# Copyright 2022 - 2026 The PyMC Labs Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Registered analysis adapters for the history-only planner.

Issue #1440 registers augmented synthetic control and interactive fixed
effects. Ordinary synthetic control stays on
:class:`~causalpy.planning.reference.HistoryOnlyPlan` and is not in
:data:`ADAPTERS`. A registered name, or an adapter instance passed to the
plan, is called instead. Groups of one to three units are one joint sample.
The group total is the draw-aligned sum of unit impacts.

The plan refuses a group or decision that the adapter's capabilities do not
allow before sampling. It is not a power curve, not a donor search, and not
Placebo in Time. Each fit is a fresh experiment. Rows after the requested
holdout are refused, so later donor outcomes cannot enter the fit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import pandas as pd
import xarray as xr

from causalpy.experiments.augmented_synthetic_control import AugmentedSyntheticControl
from causalpy.experiments.interactive_fixed_effects import (
    InteractiveFixedEffectsSyntheticControl,
)
from causalpy.experiments.model_adapter import _sampling_diagnostics
from causalpy.planning.contracts import (
    MIN_TRAINING_ROWS,
    REFERENCE_CAPABILITIES,
    RELATIVE_BASELINE_CONVENTION,
    CanonicalEvaluation,
    CanonicalFit,
    Capabilities,
    DataBoundary,
    DecisionRule,
    EffectScenario,
    EstimandDraws,
    FitDiagnostics,
    Provenance,
    WindowRequest,
    decide,
    estimand_draws,
    require_positive_int,
    require_unit_names,
    summarize_draws,
    unit_period_effect,
)
from causalpy.planning.reference import (
    _index_realized_panel,
    _planned_window,
    _prepare_history,
    _require_columns,
    _resolve_sample_kwargs,
    _select_planned_rows,
)

_IMPACT_DIMS = ("chain", "draw", "obs_ind", "treated_units")
_TRAINING_POLICY: Literal["separate-fits"] = "separate-fits"

AUGMENTED_WEIGHT_POLICY = "per_unit_dirichlet_then_draw_aligned_sum"
INTERACTIVE_FIXED_EFFECTS_WEIGHT_POLICY = "shared_factors_then_draw_aligned_sum"

AUGMENTED_PRIORS: tuple[tuple[str, str], ...] = (
    ("base_weights", "Dirichlet concentration ones, one simplex per treated unit"),
    ("tau", "HalfNormal scale 1, shared across treated units"),
    (
        "prognostic_coefficients",
        "Normal(0, tau), shared across treated units, no intercept",
    ),
    ("observation_scale", "fixed at 1; not a prior and not estimated"),
)
INTERACTIVE_FIXED_EFFECTS_PRIORS: tuple[tuple[str, str], ...] = (
    ("shared_factors", "Normal(0, 1), one column per requested factor"),
    ("shrinkage_state", "multiplicative gamma process, unused factors toward zero"),
    ("max_factors", "constructor upper bound, default 4; not a selected rank"),
)

_AUGMENTED_CAPABILITIES = Capabilities(
    estimator="augmented_synthetic_control",
    deterministic_effect_reuse=True,
    horizon_reuse=False,
    post_period_donor_data=True,
    group_support=True,
    max_group_size=3,
    allowed_decision_rules=("positive", "negative"),
    evidence=(
        ("deterministic_effect_reuse", "test_contract_journey"),
        ("horizon_reuse", "test_separate_horizon_fits"),
        ("post_period_donor_data", "test_held_out_treated_and_donor_boundary"),
        ("group_support", "test_group_draw_aligned_sum"),
        ("allowed_decision_rules", "test_contract_journey"),
    ),
)
_INTERACTIVE_FIXED_EFFECTS_CAPABILITIES = Capabilities(
    estimator="interactive_fixed_effects",
    deterministic_effect_reuse=True,
    horizon_reuse=False,
    post_period_donor_data=True,
    group_support=True,
    max_group_size=3,
    allowed_decision_rules=("positive", "negative"),
    evidence=(
        ("deterministic_effect_reuse", "test_contract_journey"),
        ("horizon_reuse", "test_separate_horizon_fits"),
        ("post_period_donor_data", "test_held_out_treated_and_donor_boundary"),
        ("group_support", "test_group_draw_aligned_sum"),
        ("allowed_decision_rules", "test_contract_journey"),
    ),
)


@dataclass(frozen=True)
class AdapterAnalysisHandoff:
    """Frozen analysis specification for one registered estimator.

    ``run`` fits the planned window and returns the same evaluation shape as
    :meth:`HistoryOnlyPlan.evaluate`. It does not sample until called.
    """

    duration: int
    sample_kwargs: tuple[tuple[str, Any], ...]
    outcome_scale: str
    training_index: pd.Index
    holdout_index: pd.Index
    max_factors: int | None
    model_class: str
    weight_policy: str
    priors: tuple[tuple[str, str], ...]
    estimand: str
    training_policy: str
    decision: DecisionRule
    treated_units: tuple[str, ...]
    control_units: tuple[str, ...]
    treatment_time: Any
    time_column: str | None
    estimator: str
    adapter: str

    def run(
        self,
        data: pd.DataFrame,
        *,
        sample_kwargs: Mapping[str, Any] | None = None,
        model: object | None = None,
        random_seed: int | None = None,
    ) -> CanonicalEvaluation:
        """Fit this specification on the planned window.

        Parameters
        ----------
        data : pandas.DataFrame
            Planning history, including the holdout tail.
        sample_kwargs : mapping or None, optional
            Settings for this call. Must still include ``draws`` and ``tune``.
            Omitted keys keep the stored specification.
        model : object or None, optional
            Rejected. The analysis class builds its own model.
        random_seed : int or None, optional
            Seed for this call when the settings do not already contain one.

        Returns
        -------
        CanonicalEvaluation
            Zero-effect evaluation on the planned window.
        """
        if model is not None:
            raise TypeError(
                f"{self.estimator} builds its model inside {self.model_class}. "
                "Pass sample_kwargs, not model. No fit was started."
            )
        stored = dict(self.sample_kwargs)
        seed = stored.get("random_seed") if random_seed is None else random_seed
        resolved = _resolve_sample_kwargs({**stored, **dict(sample_kwargs or {})}, seed)
        if "draws" not in resolved or "tune" not in resolved:
            raise TypeError(
                "sample_kwargs must include positive integer 'draws' and 'tune'. "
                "No fit was started."
            )
        adapter = _build_registered(
            self.estimator,
            sample_kwargs=resolved,
            random_seed=resolved.get("random_seed"),
            max_factors=self.max_factors,
        )
        indexed = _index_realized_panel(data, self.time_column)
        selected, excluded = _select_planned_rows(
            indexed,
            tuple(self.training_index),
            tuple(self.holdout_index),
        )
        _require_columns(selected, self.treated_units, self.control_units)
        window = _planned_window(
            selected,
            self.treatment_time,
            self.duration,
            tuple(self.training_index),
            tuple(self.holdout_index),
        )
        if not isinstance(self.decision, DecisionRule):
            raise TypeError("decision must be a DecisionRule.")
        if not isinstance(self.outcome_scale, str) or not self.outcome_scale:
            raise TypeError("outcome_scale must be a non-empty string.")
        treated = require_unit_names(
            self.treated_units,
            what="treated_units",
            min_count=1,
            max_count=3,
        )
        controls = require_unit_names(
            self.control_units,
            what="control_units",
            min_count=1,
            max_count=None,
        )
        allowed = adapter.capabilities().allowed_decision_rules
        if self.decision.direction not in allowed:
            raise ValueError(
                f"{self.estimator} does not allow a {self.decision.direction} "
                f"decision. Allowed: {allowed}. No fit was started."
            )
        fit = adapter.fit_window(
            WindowRequest(
                history=window,
                treated_units=treated,
                control_units=controls,
                treatment_time=self.treatment_time,
                duration=self.duration,
                outcome_scale=self.outcome_scale,
            )
        )
        evaluation = adapter.translate(
            fit,
            EffectScenario("absolute", 0.0),
            self.decision,
            fit_index=0,
            outcome_scale=self.outcome_scale,
        )
        return replace(
            evaluation,
            boundary=replace(
                evaluation.boundary,
                training_index=pd.Index(self.training_index),
                rows_outside_window_excluded=excluded,
            ),
            provenance=replace(evaluation.provenance, translation="observed"),
        )


class _RegisteredAdapter:
    """Shared window checks, translation, and handoff for one analysis class."""

    estimator: str
    adapter_name: str
    model_class: str
    weight_policy: str
    priors: tuple[tuple[str, str], ...]
    _capabilities: Capabilities
    _max_factors: int | None = None

    def __init__(
        self,
        *,
        model: object | None,
        sample_kwargs: Mapping[str, Any] | None,
        random_seed: int | None = None,
    ) -> None:
        if model is not None:
            raise TypeError(
                f"{self.estimator} builds its model inside {self.model_class}. "
                "Pass sample_kwargs, not model. No fit was started."
            )
        if sample_kwargs is None:
            raise TypeError(
                f"{self.estimator} requires sample_kwargs with positive integer "
                "'draws' and 'tune'. No fit was started."
            )
        self._fit_kwargs = _require_light_budget(
            _resolve_sample_kwargs(sample_kwargs, random_seed)
        )

    def capabilities(self) -> Capabilities:
        """Capability flags for this estimator."""
        return self._capabilities

    def fit_window(self, request: WindowRequest) -> CanonicalFit:
        """Fit one horizon. Does not reuse a previous experiment.

        Parameters
        ----------
        request : WindowRequest
            History and the units for one duration.
        """

        _refuse_unsupported_group(
            request.treated_units,
            self._capabilities.max_group_size,
        )
        if not request.control_units:
            raise ValueError("control_units must be nonempty. No fit was started.")
        overlap = sorted(set(request.treated_units) & set(request.control_units))
        if overlap:
            raise ValueError(
                f"Treated and control units overlap: {overlap}. No fit was started."
            )
        window, training, holdout = _window_frame(request)
        impact, counterfactual, diagnostics = self._fit_analysis(
            window,
            request.treatment_time,
            request.control_units,
            request.treated_units,
        )
        labeled_impact = _label_draws(impact, holdout, request.treated_units)
        labeled_counterfactual = _label_draws(
            counterfactual,
            holdout,
            request.treated_units,
        )
        seed = self._fit_kwargs.get("random_seed")
        return CanonicalFit(
            unit_impact=labeled_impact,
            counterfactual=labeled_counterfactual,
            diagnostics=diagnostics,
            boundary=DataBoundary(
                training_index=training,
                holdout_index=holdout,
                treated_units=request.treated_units,
                control_units=request.control_units,
                post_period_treated_in_fit=False,
                post_period_donors_in_prediction=True,
                horizon_policy=_TRAINING_POLICY,
                rows_outside_window_excluded=True,
            ),
            provenance=Provenance(
                estimator=self.estimator,
                adapter=self.adapter_name,
                model_class=self.model_class,
                random_seed=None if seed is None else int(seed),
                sample_kwargs=tuple(self._fit_kwargs.items()),
                horizon_policy=_TRAINING_POLICY,
                translation="deterministic_additive",
                uncertainty="posterior_conditional_mean",
            ),
            n_inference_fits=1,
        )

    def translate(
        self,
        fit: CanonicalFit,
        scenario: EffectScenario,
        decision: DecisionRule,
        *,
        fit_index: int,
        outcome_scale: str,
    ) -> CanonicalEvaluation:
        """Add one planned effect to each unit impact. Does not sample.

        Parameters
        ----------
        fit : CanonicalFit
            One horizon fitted without a planned effect.
        scenario : EffectScenario
            Effect to add without refitting. The magnitude is a group total.
        decision : DecisionRule
            One-sided rule applied to the translated estimand.
        fit_index : int
            Index of this horizon's single fit within the call.
        outcome_scale : str
            Label for the outcome units.
        """
        duration = int(fit.unit_impact.sizes["obs_ind"])
        n_treated = int(fit.unit_impact.sizes["treated_units"])
        delta = unit_period_effect(
            scale=scenario.scale,
            magnitude=scenario.magnitude,
            duration=duration,
            n_treated=n_treated,
            baseline=scenario.baseline,
        )
        translated = fit.unit_impact + delta
        baseline = scenario.baseline if scenario.scale == "relative" else None
        convention = RELATIVE_BASELINE_CONVENTION if baseline is not None else None
        draws = estimand_draws(
            translated,
            counterfactual=fit.counterfactual,
            baseline=baseline,
        )
        return CanonicalEvaluation(
            estimator=self.estimator,
            duration=duration,
            scenario=scenario,
            draws=draws,
            summary=summarize_draws(
                draws,
                baseline=baseline,
                baseline_convention=convention,
                treated_count=n_treated,
                duration=duration,
                outcome_scale=outcome_scale,
            ),
            decision=decide(_estimand(draws, decision.estimand), decision),
            diagnostics=fit.diagnostics,
            boundary=fit.boundary,
            provenance=fit.provenance,
            fit_index=fit_index,
            outcome_scale=outcome_scale,
        )

    def analysis_handoff(
        self,
        history: pd.DataFrame,
        treated_units: str | Sequence[str],
        control_units: str | Sequence[str],
        decision: DecisionRule,
        *,
        duration: int,
        outcome_scale: str = "outcome",
        time_column: str | None = None,
    ) -> AdapterAnalysisHandoff:
        """Return this adapter's frozen specification. Does not sample.

        Parameters
        ----------
        history : pandas.DataFrame
            Planning history. The holdout is its tail.
        treated_units : str or sequence of str
            One to three treated names.
        control_units : str or sequence of str
            Donor names.
        decision : DecisionRule
            Decision stored on the spec and applied by ``run``.
        duration : int
            Holdout length.
        outcome_scale : str, optional
            Label for the outcome units.
        time_column : str or None, optional
            Column to use as the time index.
        """
        frame = _prepare_history(history, time_column)
        periods = require_positive_int(duration, "duration")
        if len(frame) < periods + MIN_TRAINING_ROWS:
            raise ValueError(
                f"Duration {periods} leaves fewer than {MIN_TRAINING_ROWS} "
                "training rows. No fit was started."
            )
        training = frame.index[:-periods]
        holdout = frame.index[-periods:]
        treated = _unit_names(
            treated_units,
            "treated_units",
            max_count=self._capabilities.max_group_size,
        )
        controls = _unit_names(control_units, "control_units", max_count=None)
        overlap = sorted(set(treated) & set(controls))
        if overlap:
            raise ValueError(
                f"Treated and control units overlap: {overlap}. No fit was started."
            )
        _require_columns(frame, treated, controls)
        return AdapterAnalysisHandoff(
            duration=periods,
            sample_kwargs=tuple(self._fit_kwargs.items()),
            outcome_scale=outcome_scale,
            training_index=training,
            holdout_index=holdout,
            max_factors=self._max_factors,
            model_class=self.model_class,
            weight_policy=self.weight_policy,
            priors=self.priors,
            estimand="draw_aligned_unit_impact_sum",
            training_policy=_TRAINING_POLICY,
            decision=decision,
            treated_units=treated,
            control_units=controls,
            treatment_time=holdout[0],
            time_column=time_column,
            estimator=self.estimator,
            adapter=self.adapter_name,
        )

    def _fit_analysis(
        self,
        window: pd.DataFrame,
        treatment_time: Any,
        control_units: tuple[str, ...],
        treated_units: tuple[str, ...],
    ) -> tuple[xr.DataArray, xr.DataArray, FitDiagnostics]:
        raise NotImplementedError


class AugmentedSyntheticControlAdapter(_RegisteredAdapter):
    """Plan with :class:`AugmentedSyntheticControl`.

    Groups of one to three units are one joint sample. Weights stay on each
    unit's simplex. The group total sums those unit impacts inside each draw.
    """

    estimator = "augmented_synthetic_control"
    adapter_name = "AugmentedSyntheticControlAdapter"
    model_class = "AugmentedSyntheticControl"
    weight_policy = AUGMENTED_WEIGHT_POLICY
    priors = AUGMENTED_PRIORS
    _capabilities = _AUGMENTED_CAPABILITIES

    def _fit_analysis(
        self,
        window: pd.DataFrame,
        treatment_time: Any,
        control_units: tuple[str, ...],
        treated_units: tuple[str, ...],
    ) -> tuple[xr.DataArray, xr.DataArray, FitDiagnostics]:
        experiment = AugmentedSyntheticControl(
            window,
            treatment_time,
            control_units,
            treated_units,
        )
        experiment.fit(sample_kwargs=dict(self._fit_kwargs))
        if experiment.idata is None:
            raise RuntimeError(
                "AugmentedSyntheticControl did not return inference data."
            )
        posterior = experiment.idata.posterior
        return (
            posterior["treatment_effect"],
            posterior["counterfactual"],
            _diagnostics_from_idata(experiment.idata),
        )


class InteractiveFixedEffectsAdapter(_RegisteredAdapter):
    """Plan with :class:`InteractiveFixedEffectsSyntheticControl`.

    Groups of one to three units share one control-panel factor sample. The
    group total sums unit impacts inside each draw. ``max_factors`` is the
    analysis class's upper bound, not a selected rank.

    Parameters
    ----------
    model : None
        Rejected. The analysis class builds its own model.
    sample_kwargs : mapping or None
        Forwarded to ``fit``. Must include ``draws`` and ``tune``.
    random_seed : int or None
        Seed applied when the settings do not already contain one.
    max_factors : int, default 4
        Upper bound forwarded to the analysis class.
    """

    estimator = "interactive_fixed_effects"
    adapter_name = "InteractiveFixedEffectsAdapter"
    model_class = "InteractiveFixedEffectsSyntheticControl"
    weight_policy = INTERACTIVE_FIXED_EFFECTS_WEIGHT_POLICY
    priors = INTERACTIVE_FIXED_EFFECTS_PRIORS
    _capabilities = _INTERACTIVE_FIXED_EFFECTS_CAPABILITIES

    def __init__(
        self,
        *,
        model: object | None,
        sample_kwargs: Mapping[str, Any] | None,
        random_seed: int | None = None,
        max_factors: int = 4,
    ) -> None:
        self._max_factors = require_positive_int(max_factors, "max_factors")
        super().__init__(
            model=model,
            sample_kwargs=sample_kwargs,
            random_seed=random_seed,
        )

    def _fit_analysis(
        self,
        window: pd.DataFrame,
        treatment_time: Any,
        control_units: tuple[str, ...],
        treated_units: tuple[str, ...],
    ) -> tuple[xr.DataArray, xr.DataArray, FitDiagnostics]:
        # WidePanel stores a named index as an obs_ind coordinate. A
        # timezone-aware coordinate cannot be subtracted after sampling, so
        # the experiment sees naive instants. Convert the treatment instant
        # into the index zone first; stripping each zone's wall clock would
        # train on a different row. Returned labels stay on the request index.
        frame = window.copy()
        treatment = treatment_time
        if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
            if getattr(treatment, "tzinfo", None) is not None:
                treatment = pd.Timestamp(treatment).tz_convert(frame.index.tz)
            frame.index = frame.index.tz_localize(None)
            if getattr(treatment, "tzinfo", None) is not None:
                treatment = treatment.tz_localize(None)
        frame.index = frame.index.rename(None)
        experiment = InteractiveFixedEffectsSyntheticControl(
            frame,
            treatment,
            control_units,
            treated_units,
            max_factors=self._max_factors if self._max_factors is not None else 4,
        )
        experiment.fit(**dict(self._fit_kwargs))
        result = experiment.result
        if (
            result is None
            or result.impact_post is None
            or result.predictions_post is None
        ):
            raise RuntimeError(
                "InteractiveFixedEffectsSyntheticControl did not return a posterior impact."
            )
        return (
            result.impact_post,
            result.predictions_post,
            _unavailable_diagnostics(
                "InteractiveFixedEffectsSyntheticControl.fit does not retain "
                "InferenceData, so divergences, R-hat, and bulk ESS are not "
                "available from the composed result."
            ),
        )


def _build_registered(
    estimator: str,
    *,
    sample_kwargs: Mapping[str, Any] | None,
    random_seed: int | None,
    max_factors: int | None,
) -> _RegisteredAdapter:
    cls = ADAPTERS[estimator]
    if cls is InteractiveFixedEffectsAdapter:
        return InteractiveFixedEffectsAdapter(
            model=None,
            sample_kwargs=sample_kwargs,
            random_seed=random_seed,
            max_factors=4 if max_factors is None else max_factors,
        )
    if max_factors is not None:
        raise TypeError(
            "max_factors is an interactive fixed-effects setting. No fit was started."
        )
    return cls(model=None, sample_kwargs=sample_kwargs, random_seed=random_seed)


def analysis_handoff(
    history: pd.DataFrame,
    treated_units: str | Sequence[str],
    control_units: str | Sequence[str],
    decision: DecisionRule,
    *,
    estimator: str,
    duration: int,
    sample_kwargs: Mapping[str, Any] | None = None,
    model: object | None = None,
    random_seed: int | None = None,
    outcome_scale: str = "outcome",
    time_column: str | None = None,
    max_factors: int | None = None,
) -> AdapterAnalysisHandoff:
    """Return the frozen spec for a registered estimator. Does not sample.

    Ordinary synthetic control uses :meth:`HistoryOnlyPlan.analysis_handoff`.
    This function is the analysis path for the estimators in :data:`ADAPTERS`.

    Parameters
    ----------
    history : pandas.DataFrame
        Planning history. The holdout is its tail.
    treated_units : str or sequence of str
        One to three treated names.
    control_units : str or sequence of str
        Donor names.
    decision : DecisionRule
        Decision stored on the spec.
    estimator : str
        ``augmented_synthetic_control`` or ``interactive_fixed_effects``.
    duration : int
        Holdout length.
    sample_kwargs : mapping or None, optional
        Sampling settings. Must include ``draws`` and ``tune``.
    model : object or None, optional
        Rejected. These analysis classes build their own models.
    random_seed : int or None, optional
        Seed applied when the settings do not already contain one.
    outcome_scale : str, optional
        Label for the outcome units.
    time_column : str or None, optional
        Column to use as the time index.
    max_factors : int or None, optional
        Interactive fixed-effects upper bound. Rejected for other estimators.
    """
    if estimator == "synthetic_control":
        raise ValueError(
            "synthetic_control uses HistoryOnlyPlan.analysis_handoff. "
            "No fit was started."
        )
    if estimator not in ADAPTERS:
        known = ", ".join(ADAPTERS)
        raise ValueError(
            f"Unknown registered estimator {estimator!r}. Registered: {known}. "
            "No fit was started."
        )
    if model is not None:
        raise TypeError(
            f"{estimator} builds its model inside the analysis class. "
            "Pass sample_kwargs, not model. No fit was started."
        )
    adapter = _build_registered(
        estimator,
        sample_kwargs=sample_kwargs,
        random_seed=random_seed,
        max_factors=max_factors,
    )
    return adapter.analysis_handoff(
        history,
        treated_units,
        control_units,
        decision,
        duration=duration,
        outcome_scale=outcome_scale,
        time_column=time_column,
    )


ADAPTERS: dict[str, type[_RegisteredAdapter]] = {
    "augmented_synthetic_control": AugmentedSyntheticControlAdapter,
    "interactive_fixed_effects": InteractiveFixedEffectsAdapter,
}


def capability_matrix() -> tuple[Capabilities, ...]:
    """Return the tested capability row for each planning method.

    Vanilla synthetic control is the reference row. The other rows are the
    registered adapters. Evidence names are tests, not claims.
    """
    return (
        REFERENCE_CAPABILITIES,
        _AUGMENTED_CAPABILITIES,
        _INTERACTIVE_FIXED_EFFECTS_CAPABILITIES,
    )


def _require_light_budget(sample_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(sample_kwargs)
    for name in ("draws", "tune"):
        value = resolved.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TypeError(
                f"sample_kwargs[{name!r}] must be a positive integer. "
                "No fit was started."
            )
    chains = resolved.get("chains")
    if chains is not None and (
        isinstance(chains, bool) or not isinstance(chains, int) or chains <= 0
    ):
        raise TypeError(
            "sample_kwargs['chains'] must be a positive integer. No fit was started."
        )
    return resolved


def _unit_names(
    names: str | Sequence[str],
    what: str,
    *,
    max_count: int | None,
) -> tuple[str, ...]:
    return require_unit_names(names, what=what, min_count=1, max_count=max_count)


def _refuse_unsupported_group(
    treated_units: tuple[str, ...], max_group_size: int
) -> None:
    if len(treated_units) > max_group_size:
        raise ValueError(
            f"treated_units has {len(treated_units)} names; this adapter supports "
            f"at most {max_group_size}. No fit was started."
        )


def _window_frame(request: WindowRequest) -> tuple[pd.DataFrame, pd.Index, pd.Index]:
    periods = require_positive_int(request.duration, "duration")
    if len(request.history.index) < periods + MIN_TRAINING_ROWS:
        raise ValueError(
            "The history has fewer than two training rows before the holdout. "
            "No fit was started."
        )
    training = request.history.index[:-periods]
    holdout = request.history.index[-periods:]
    if holdout[0] != request.treatment_time:
        raise ValueError(
            "treatment_time must be the first holdout label. "
            "Rows after the requested window are refused, not sliced. "
            "No fit was started."
        )
    return request.history, training, holdout


def _label_draws(
    draws: xr.DataArray,
    holdout: pd.Index,
    treated_units: tuple[str, ...],
) -> xr.DataArray:
    labeled = draws
    if "post_time" in labeled.dims and "obs_ind" not in labeled.dims:
        labeled = labeled.rename({"post_time": "obs_ind"})
    if "treated_units" not in labeled.dims:
        if len(treated_units) != 1:
            raise ValueError(
                "A multi-unit impact is missing the treated_units dimension. "
                "No fit was started."
            )
        labeled = labeled.expand_dims(treated_units=[treated_units[0]])
    missing = [name for name in _IMPACT_DIMS if name not in labeled.dims]
    if missing:
        raise ValueError(f"Impact draws are missing {missing}. No fit was started.")
    labeled = labeled.transpose(*_IMPACT_DIMS)
    current = tuple(labeled.coords["treated_units"].values.tolist())
    if current != treated_units:
        labeled = labeled.sel(treated_units=list(treated_units))
    if int(labeled.sizes["obs_ind"]) != len(holdout):
        raise ValueError(
            "Impact length does not match the planned holdout. No fit was started."
        )
    if int(labeled.sizes["treated_units"]) != len(treated_units):
        raise ValueError(
            "Fitted treated-unit count does not match the request. No fit was started."
        )
    return labeled.assign_coords(
        obs_ind=("obs_ind", list(holdout)),
        treated_units=("treated_units", list(treated_units)),
    ).copy(deep=True)


def _estimand(draws: EstimandDraws, name: str) -> xr.DataArray:
    chosen = getattr(draws, name)
    if not isinstance(chosen, xr.DataArray):
        raise ValueError(f"Estimand {name!r} is not available on these draws.")
    return chosen


def _diagnostics_from_idata(idata: Any) -> FitDiagnostics:
    tree = _idata_tree(idata)
    if tree is None:
        return _unavailable_diagnostics(
            "InferenceData could not be read as a posterior DataTree."
        )
    summary = _sampling_diagnostics(tree)
    status = summary["diagnostics_status"]
    if status not in {"available", "partial", "unavailable"}:
        status = "unavailable"
    return FitDiagnostics(
        divergences=summary["divergences"],
        max_rhat=summary["max_rhat"],
        min_bulk_ess=summary["min_bulk_ess"],
        diagnostics_status=status,
        diagnostics_reason=summary["diagnostics_reason"],
    )


def _idata_tree(idata: Any) -> xr.DataTree | None:
    if isinstance(idata, xr.DataTree):
        return idata
    posterior = getattr(idata, "posterior", None)
    if not isinstance(posterior, xr.Dataset):
        return None
    groups: dict[str, xr.Dataset] = {}
    for name in ("posterior", "sample_stats"):
        group = getattr(idata, name, None)
        if isinstance(group, xr.Dataset):
            groups[name] = group
    if "posterior" not in groups:
        return None
    return xr.DataTree.from_dict(groups)


def _unavailable_diagnostics(reason: str) -> FitDiagnostics:
    return FitDiagnostics(
        divergences=None,
        max_rhat=None,
        min_bulk_ess=None,
        diagnostics_status="unavailable",
        diagnostics_reason=reason,
    )
