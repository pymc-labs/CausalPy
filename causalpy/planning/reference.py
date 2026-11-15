# Copyright 2022 - 2026 The PyMC Labs Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""History-only planning entry point and the synthetic-control reference path.

Construction validates inputs and does not sample. :meth:`HistoryOnlyPlan.evaluate`
fits the selected adapter once per requested horizon, then translates that fit
across effects. The reference adapter is ordinary synthetic control on one
treated unit. A registered adapter, or one passed in, is called instead.
Groups of two or three units belong to that adapter, not to a second entry point.
The plan refuses a group or decision that adapter did not declare before
``fit_window``.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr

from causalpy.custom_exceptions import DataException
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.input_data import to_pandas_with_time_index
from causalpy.planning.contracts import (
    DECLARED_ESTIMATORS,
    MIN_TRAINING_ROWS,
    REFERENCE_CAPABILITIES,
    RELATIVE_BASELINE_CONVENTION,
    UNIMPLEMENTED_ESTIMATORS,
    CanonicalEvaluation,
    CanonicalFit,
    Capabilities,
    DataBoundary,
    DecisionRule,
    EffectScenario,
    EstimandDraws,
    FitDiagnostics,
    PlanningResult,
    Provenance,
    WindowRequest,
    decide,
    default_relative_floor,
    estimand_draws,
    require_finite,
    require_positive_int,
    require_relative_baseline,
    require_unit_names,
    summarize_draws,
    unit_period_effect,
)
from causalpy.planning.seams import EstimatorAdapter
from causalpy.pymc_models import WeightedSumFitter

_ADAPTER_MODULE = "causalpy.planning.adapters"
_IMPACT_DIMS = ("chain", "draw", "obs_ind", "treated_units")


def _registered_adapters() -> dict[str, Any]:
    """Return adapters registered by issue #1440, if that module exists.

    The reference path does not discover plugins. It imports one module,
    ``causalpy.planning.adapters``, and reads its ``ADAPTERS`` dict. Until
    that module exists, declared unimplemented names fail before sampling.
    """
    try:
        module = importlib.import_module(_ADAPTER_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == _ADAPTER_MODULE:
            return {}
        raise
    registered = getattr(module, "ADAPTERS", {})
    if not isinstance(registered, dict):
        raise TypeError(f"{_ADAPTER_MODULE}.ADAPTERS must be a dict.")
    return registered


class SyntheticControlAdapter:
    """Fit one treated unit with :class:`WeightedSumFitter` and translate effects.

    This is the working reference for the :class:`EstimatorAdapter` protocol.
    It fits on pre-period outcomes, predicts the holdout from post-period donor
    outcomes, and adds a planned effect to the impact path without refitting.

    Parameters
    ----------
    model : WeightedSumFitter or None
        Caller-supplied fitter. Mutually exclusive with ``sample_kwargs``.
    sample_kwargs : mapping or None
        Sampling settings used to build a fresh fitter when ``model`` is
        omitted.
    random_seed : int or None
        Seed applied when the resolved settings do not already contain one.
    treated_units : tuple of str
        The one treated unit this reference adapter accepts. A longer group
        is refused here, before sampling.
    """

    def __init__(
        self,
        *,
        model: WeightedSumFitter | None,
        sample_kwargs: Mapping[str, Any] | None,
        random_seed: int | None,
        treated_units: tuple[str, ...],
    ) -> None:
        if len(treated_units) != 1:
            raise ValueError(_group_message(treated_units))
        if model is not None and type(model) is not WeightedSumFitter:
            raise TypeError(
                "The synthetic_control reference adapter fits WeightedSumFitter "
                f"only. Got {type(model).__name__}. Other estimators are the "
                "#1440 seam. No fit was started."
            )
        if model is not None and sample_kwargs is not None:
            raise TypeError("Pass model or sample_kwargs, not both.")
        if model is None and sample_kwargs is None:
            raise TypeError(
                "Pass sample_kwargs or a WeightedSumFitter. The reference path "
                "does not start a default-length PyMC sample."
            )
        self._model = model
        if model is None:
            self._fit_kwargs = _resolve_sample_kwargs(sample_kwargs, random_seed)
        else:
            self._fit_kwargs = _resolve_sample_kwargs(
                dict(model.sample_kwargs),
                random_seed,
            )
        self._sample_kwargs = self._fit_kwargs

    def capabilities(self) -> Capabilities:
        """Return the evidence-backed capability flags for this adapter."""
        return REFERENCE_CAPABILITIES

    def fit_window(self, request: WindowRequest) -> CanonicalFit:
        """Fit one horizon. The effect is not an input and is not baked in.

        Parameters
        ----------
        request : WindowRequest
            History slice, units, treatment time, and sampling settings.
        """
        if len(request.treated_units) != 1:
            raise ValueError(_group_message(request.treated_units))
        if request.duration < 1:
            raise ValueError("duration must be a positive integer.")
        frame = request.history
        if len(frame) - request.duration < MIN_TRAINING_ROWS:
            raise ValueError(
                f"Duration {request.duration} leaves fewer than "
                f"{MIN_TRAINING_ROWS} training rows. No fit was started."
            )
        holdout = frame.index[-request.duration :]
        training = frame.index[: -request.duration]
        if holdout[0] != request.treatment_time:
            raise ValueError(
                "treatment_time must be the first label of the holdout window."
            )
        experiment = SyntheticControl(
            frame,
            request.treatment_time,
            control_units=list(request.control_units),
            treated_units=list(request.treated_units),
            model=self._fresh_model(),
        ).fit()
        result = experiment.result
        if (
            result is None
            or result.impact_post is None
            or result.predictions_post is None
        ):
            raise RuntimeError("SyntheticControl did not return a posterior impact.")
        impact = _label_window(
            result.impact_post,
            holdout,
            request.treated_units,
        )
        counterfactual = _label_window(
            result.predictions_post,
            holdout,
            request.treated_units,
        )
        return CanonicalFit(
            unit_impact=impact,
            counterfactual=counterfactual,
            diagnostics=_diagnostics(experiment),
            boundary=DataBoundary(
                training_index=training.copy(),
                holdout_index=holdout.copy(),
                treated_units=request.treated_units,
                control_units=request.control_units,
                post_period_treated_in_fit=False,
                post_period_donors_in_prediction=True,
                horizon_policy="separate-fits",
                rows_outside_window_excluded=True,
            ),
            provenance=_provenance(
                self._fit_kwargs,
                translation="deterministic_additive",
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
        """Add one planned effect to a fitted impact path. Does not sample.

        Parameters
        ----------
        fit : CanonicalFit
            One horizon fitted without a planned effect.
        scenario : EffectScenario
            Effect to add without refitting.
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
        chosen = _estimand(draws, decision.estimand)
        return CanonicalEvaluation(
            estimator="synthetic_control",
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
            decision=decide(chosen, decision),
            diagnostics=fit.diagnostics,
            boundary=fit.boundary,
            provenance=fit.provenance,
            fit_index=fit_index,
            outcome_scale=outcome_scale,
        )

    def _fresh_model(self) -> WeightedSumFitter:
        if self._model is None:
            return WeightedSumFitter(sample_kwargs=dict(self._fit_kwargs))
        cloned = self._model._clone()
        if not isinstance(cloned, WeightedSumFitter):
            raise TypeError(
                "Cloning the reference model did not return WeightedSumFitter."
            )
        cloned.sample_kwargs = dict(self._fit_kwargs)
        return cloned


class HistoryOnlyPlan:
    """Plan an intervention from history. Construction does not sample.

    Parameters
    ----------
    history : pandas.DataFrame
        Outcomes through the last period the plan may use. The intervention
        is placed at the end. Rows after a requested holdout are not used
        because the holdout is the tail of this frame.
    treated_units : str or sequence of str
        One to three unit names. A string is one name, not a sequence of
        characters. The reference adapter accepts one name and rejects a
        longer group before sampling.
    control_units : str or sequence of str
        Donor names. They must not overlap the treated names.
    decision : DecisionRule
        One-sided rule on an absolute estimand.
    estimator : str, optional
        ``"synthetic_control"`` is the reference path. A name registered in
        ``causalpy.planning.adapters.ADAPTERS`` is constructed and called.
        Augmented synthetic control and interactive fixed effects stay
        declared seams for issue #1440 until that registry exists.
    adapter : EstimatorAdapter or None, optional
        A constructed adapter. When given, the plan calls its protocol
        methods and does not build one. Mutually exclusive with ``model``,
        ``sample_kwargs``, and ``random_seed``.
    model : object or None, optional
        Forwarded to the selected adapter. Mutually exclusive with
        ``sample_kwargs``. The reference adapter accepts only
        :class:`WeightedSumFitter`. A registered adapter may accept its own
        model.
    sample_kwargs : mapping or None, optional
        Passed to a fresh fitter when ``model`` is omitted. Required on the
        reference path so a default-length sample cannot start by accident.
    time_column : str or None, optional
        Column to use as the time index when ``history`` does not already
        have one.
    random_seed : int or None, optional
        Stored in ``sample_kwargs`` when that mapping has no seed. Rejected
        when it disagrees with a seed already in ``sample_kwargs`` or on
        ``model``. A registered adapter receives the resolved mapping.
    outcome_scale : str, optional
        Label for the outcome units. It does not convert values.
    """

    def __init__(
        self,
        history: pd.DataFrame,
        treated_units: str | Sequence[str],
        control_units: str | Sequence[str],
        decision: DecisionRule,
        *,
        estimator: str = "synthetic_control",
        adapter: EstimatorAdapter | None = None,
        model: object | None = None,
        sample_kwargs: Mapping[str, Any] | None = None,
        time_column: str | None = None,
        random_seed: int | None = None,
        outcome_scale: str = "outcome",
    ) -> None:
        if not isinstance(decision, DecisionRule):
            raise TypeError("decision must be a DecisionRule.")
        if not isinstance(outcome_scale, str) or not outcome_scale:
            raise TypeError("outcome_scale must be a non-empty string.")
        self._history = _prepare_history(history, time_column)
        self._time_column = time_column
        self._treated_units = require_unit_names(
            treated_units,
            what="treated_units",
            min_count=1,
            max_count=3,
        )
        self._control_units = require_unit_names(
            control_units,
            what="control_units",
            min_count=1,
            max_count=None,
        )
        _require_columns(self._history, self._treated_units, self._control_units)
        if adapter is not None:
            if not isinstance(estimator, str) or not estimator:
                raise TypeError("estimator must be a string.")
            self._estimator = estimator
            self._adapter = _adopt_adapter(
                adapter,
                estimator,
                model,
                sample_kwargs,
                random_seed,
            )
        else:
            self._estimator = _require_estimator(estimator)
            self._adapter = _construct_adapter(
                self._estimator,
                model,
                sample_kwargs,
                random_seed,
                self._treated_units,
            )
        _require_supported(
            self._adapter, self._estimator, self._treated_units, decision
        )
        self._decision = decision
        self._outcome_scale = outcome_scale

    @property
    def capabilities(self) -> Capabilities:
        """Capability flags for the selected estimator."""
        return self._adapter.capabilities()

    def analysis_handoff(self, *, duration: int) -> AnalysisHandoff:
        """Return an executable analysis spec. Does not sample.

        The reference path stores the model and sampling settings that
        ``evaluate`` fits, including a caller-supplied model's priors.

        Parameters
        ----------
        duration : int
            Holdout length. The treatment time is the first stored holdout
            label. ``run`` selects those labels plus the planned training
            labels and drops every other row.
        """
        if not isinstance(self._adapter, SyntheticControlAdapter):
            raise ValueError(
                "analysis_handoff reproduces the synthetic_control reference "
                f"fit. {self._estimator} must supply its own analysis path. "
                "No fit was started."
            )
        periods = require_positive_int(duration, "duration")
        self._require_duration(periods)
        template = self._adapter._fresh_model()
        return AnalysisHandoff(
            treated_units=self._treated_units,
            control_units=self._control_units,
            treatment_time=self._history.index[-periods],
            duration=periods,
            decision=self._decision,
            estimator=self._estimator,
            sample_kwargs=tuple(template.sample_kwargs.items()),
            outcome_scale=self._outcome_scale,
            training_index=tuple(self._history.index[:-periods]),
            holdout_index=tuple(self._history.index[-periods:]),
            time_column=self._time_column,
            model=template,
        )

    def evaluate(
        self,
        durations: int | Sequence[int],
        *,
        absolute_effects: Sequence[float] | None = None,
        relative_effects: Sequence[float] | None = None,
        relative_baseline: float | None = None,
        relative_floor: float | None = None,
    ) -> PlanningResult:
        """Fit each horizon once and translate it across the requested effects.

        Parameters
        ----------
        durations : int or sequence of int
            Holdout lengths. Each length is a separate fit. A shorter holdout
            is not treated as a prefix of a longer one.
        absolute_effects : sequence of float or None, optional
            Group-total per-period increments on the outcome scale.
        relative_effects : sequence of float or None, optional
            Fractions of ``relative_baseline``. Omitted unless a baseline is
            declared.
        relative_baseline : float or None, optional
            Finite scalar on the cumulative total outcome scale. Required when
            relative effects are requested. Signed and near-zero values are
            rejected before sampling.
        relative_floor : float or None, optional
            Smallest baseline that is still treated as defined. The default is
            ``1e-8`` times the median absolute pre-period treated outcome.
            Pass ``0`` only when a strictly positive baseline is the convention
            you mean.

        Returns
        -------
        PlanningResult
            One evaluation per duration and effect. ``n_inference_fits`` is
            the sum of the adapter's reported fits, not the number of effects.
        """
        _require_supported(
            self._adapter, self._estimator, self._treated_units, self._decision
        )
        requested = self._durations(durations)
        scenarios = self._scenarios(
            requested,
            absolute_effects=absolute_effects,
            relative_effects=relative_effects,
            relative_baseline=relative_baseline,
            relative_floor=relative_floor,
        )
        evaluations: list[CanonicalEvaluation] = []
        fits = 0
        for fit_index, duration in enumerate(requested):
            fit = self._adapter.fit_window(self._window(duration))
            if (
                isinstance(self._adapter, SyntheticControlAdapter)
                and fit.n_inference_fits != 1
            ):
                raise RuntimeError(
                    "The reference adapter must use one fit per horizon, "
                    f"got {fit.n_inference_fits}."
                )
            fits += int(fit.n_inference_fits)
            for scenario in scenarios:
                evaluations.append(
                    self._adapter.translate(
                        fit,
                        scenario,
                        self._decision,
                        fit_index=fit_index,
                        outcome_scale=self._outcome_scale,
                    )
                )
        return PlanningResult(
            evaluations=tuple(evaluations),
            n_inference_fits=fits,
            capabilities=self.capabilities,
        )

    def _window(self, duration: int) -> WindowRequest:
        return WindowRequest(
            history=self._history,
            treated_units=self._treated_units,
            control_units=self._control_units,
            treatment_time=self._history.index[-duration],
            duration=duration,
            outcome_scale=self._outcome_scale,
        )

    def _durations(self, durations: int | Sequence[int]) -> tuple[int, ...]:
        if isinstance(durations, bool):
            raise TypeError(
                "durations must be a positive integer or a sequence of them."
            )
        if isinstance(durations, (int, np.integer)):
            requested: tuple[int, ...] = (int(durations),)
        elif isinstance(durations, Sequence) and not isinstance(durations, str):
            requested = tuple(
                require_positive_int(item, "duration") for item in durations
            )
        else:
            raise TypeError(
                "durations must be a positive integer or a sequence of them."
            )
        if not requested:
            raise ValueError("Pass at least one duration.")
        if len(requested) != len(set(requested)):
            raise ValueError(
                "Durations must be unique. Each horizon is a separate fit."
            )
        for duration in requested:
            self._require_duration(duration)
        return requested

    def _require_duration(self, duration: int) -> None:
        training = len(self._history) - duration
        if training < MIN_TRAINING_ROWS:
            raise ValueError(
                f"Duration {duration} leaves {training} training rows; at least "
                f"{MIN_TRAINING_ROWS} are required. No fit was started."
            )

    def _scenarios(
        self,
        durations: tuple[int, ...],
        *,
        absolute_effects: Sequence[float] | None,
        relative_effects: Sequence[float] | None,
        relative_baseline: float | None,
        relative_floor: float | None,
    ) -> tuple[EffectScenario, ...]:
        absolute = _effect_values(absolute_effects, "absolute_effects")
        relative = _effect_values(relative_effects, "relative_effects")
        if not absolute and not relative:
            raise ValueError("Pass at least one absolute or relative effect.")
        baseline = None
        if relative:
            floor = self._floor(durations, relative_floor)
            baseline = require_relative_baseline(relative_baseline, floor=floor)
        scenarios = [EffectScenario("absolute", value) for value in absolute]
        scenarios.extend(
            EffectScenario("relative", value, baseline) for value in relative
        )
        return tuple(scenarios)

    def _floor(self, durations: tuple[int, ...], declared: float | None) -> float:
        if declared is not None:
            floor = require_finite(declared, "relative_floor")
            if floor < 0.0:
                raise ValueError("relative_floor must be non-negative.")
            return floor
        floors = [
            default_relative_floor(
                self._history.iloc[:-duration][list(self._treated_units)].to_numpy(
                    dtype=float
                )
            )
            for duration in durations
        ]
        return max(floors)


@dataclass(frozen=True)
class AnalysisHandoff:
    """Executable spec for the analysis path. ``run`` fits the realized panel.

    The planning evaluation does not call this. ``run`` is today's
    :class:`~causalpy.experiments.synthetic_control.SyntheticControl` analysis
    on the planned training labels plus the stored planned holdout labels.
    Rows outside that exact window are dropped, including an out-of-order or
    duplicated row. A missing planned label, a duplicated planned label, fewer
    than two training rows, or a selected window that is not strictly
    increasing is refused before sampling. It does not inject a planned effect.
    The stored ``model`` is the fitter ``evaluate`` uses, including custom
    priors, unless the caller passes a replacement. ``training_index``,
    ``holdout_index``, and ``time_column`` are the planned labels and the
    column that produced them.
    """

    treated_units: tuple[str, ...]
    control_units: tuple[str, ...]
    treatment_time: Any
    duration: int
    decision: DecisionRule
    estimator: str
    sample_kwargs: tuple[tuple[str, Any], ...]
    outcome_scale: str
    training_index: tuple[Any, ...]
    holdout_index: tuple[Any, ...]
    time_column: str | None
    model: WeightedSumFitter | None = field(default=None, compare=False, hash=False)

    def run(
        self,
        data: pd.DataFrame,
        *,
        model: WeightedSumFitter | None = None,
        sample_kwargs: Mapping[str, Any] | None = None,
        time_column: str | None = None,
        random_seed: int | None = None,
    ) -> CanonicalEvaluation:
        """Fit the planned window and return the observed impact.

        Parameters
        ----------
        data : pandas.DataFrame
            Realized outcomes. Rows outside the planned training labels and
            stored holdout labels are dropped, including an out-of-order or
            duplicated row. A missing planned label, a duplicated planned
            label, fewer than two training rows, or a selected window that is
            not strictly increasing is refused before sampling. This is not
            the planning history unless the caller passes that frame on purpose.
        model : WeightedSumFitter or None, optional
            Overrides the model stored on the handoff.
        sample_kwargs : mapping or None, optional
            Builds a fresh fitter and overrides the stored sampling settings.
            Not accepted together with ``model``.
        time_column : str or None, optional
            Overrides the column stored on the handoff. Omitted uses that
            stored column.
        random_seed : int or None, optional
            Used when the resolved sampling settings have no seed. Rejected
            when it disagrees with the seed already on the model.
        """
        if self.estimator != "synthetic_control" or len(self.treated_units) != 1:
            raise ValueError(
                "The reference handoff runs one-unit synthetic control only. "
                "No fit was started."
            )
        column = self.time_column if time_column is None else time_column
        indexed = _index_realized_panel(data, column)
        selected, excluded = _select_planned_rows(
            indexed,
            self.training_index,
            self.holdout_index,
        )
        _require_columns(selected, self.treated_units, self.control_units)
        frame = _planned_window(
            selected,
            self.treatment_time,
            self.duration,
            self.training_index,
            self.holdout_index,
        )
        if model is not None and sample_kwargs is not None:
            raise TypeError("Pass model or sample_kwargs, not both.")
        if model is not None:
            resolved_model: WeightedSumFitter | None = model
            resolved_kwargs: Mapping[str, Any] | None = None
        elif sample_kwargs is not None or self.model is None:
            resolved_model = None
            resolved_kwargs = (
                dict(self.sample_kwargs) if sample_kwargs is None else sample_kwargs
            )
        else:
            resolved_model = self.model
            resolved_kwargs = None
        adapter = SyntheticControlAdapter(
            model=resolved_model,
            sample_kwargs=resolved_kwargs,
            random_seed=random_seed,
            treated_units=self.treated_units,
        )
        fit = adapter.fit_window(
            WindowRequest(
                history=frame,
                treated_units=self.treated_units,
                control_units=self.control_units,
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


def _adopt_adapter(
    adapter: EstimatorAdapter,
    estimator: str,
    model: object | None,
    sample_kwargs: Mapping[str, Any] | None,
    random_seed: int | None,
) -> EstimatorAdapter:
    if model is not None or sample_kwargs is not None or random_seed is not None:
        raise TypeError(
            "Pass an adapter or model/sample_kwargs/random_seed, not both. "
            "The adapter already carries its fit settings. No fit was started."
        )
    try:
        caps = adapter.capabilities()
    except AttributeError as exc:
        raise TypeError(
            "adapter must implement EstimatorAdapter. No fit was started."
        ) from exc
    named = getattr(caps, "estimator", None)
    if named != estimator:
        raise ValueError(
            f"adapter capabilities name {named!r}, not estimator={estimator!r}. "
            "No fit was started."
        )
    return adapter


def _fit_arguments(
    model: object | None,
    sample_kwargs: Mapping[str, Any] | None,
    random_seed: int | None,
) -> tuple[object | None, Mapping[str, Any] | None, int | None]:
    """Resolve one seed and reject a model paired with sampling settings."""
    if model is not None and sample_kwargs is not None:
        raise TypeError("Pass model or sample_kwargs, not both. No fit was started.")
    if model is not None:
        existing = getattr(model, "sample_kwargs", None)
        if existing is not None:
            resolved = _resolve_sample_kwargs(existing, random_seed)
            return model, None, resolved.get("random_seed")
        if random_seed is not None:
            resolved = _resolve_sample_kwargs(None, random_seed)
            return model, None, resolved.get("random_seed")
        return model, None, None
    if sample_kwargs is None:
        return None, None, random_seed
    resolved = _resolve_sample_kwargs(sample_kwargs, random_seed)
    return None, resolved, resolved.get("random_seed")


def _require_supported(
    adapter: EstimatorAdapter,
    estimator: str,
    treated: tuple[str, ...],
    decision: DecisionRule,
) -> None:
    """Refuse a group or decision the adapter did not declare."""
    try:
        caps = adapter.capabilities()
    except AttributeError as exc:
        raise TypeError(
            "adapter must implement EstimatorAdapter. No fit was started."
        ) from exc
    named = getattr(caps, "estimator", None)
    if named != estimator:
        raise ValueError(
            f"adapter capabilities name {named!r}, not estimator={estimator!r}. "
            "No fit was started."
        )
    max_group = getattr(caps, "max_group_size", None)
    if isinstance(max_group, bool) or not isinstance(max_group, int):
        raise TypeError("adapter capabilities max_group_size must be an int.")
    group_support = bool(getattr(caps, "group_support", False))
    if len(treated) > max_group or (len(treated) > 1 and not group_support):
        raise ValueError(
            f"{estimator} supports at most {max_group} treated units. "
            f"Got {len(treated)}. No fit was started."
        )
    allowed = tuple(getattr(caps, "allowed_decision_rules", ()))
    if decision.direction not in allowed:
        raise ValueError(
            f"{estimator} does not allow a {decision.direction} decision. "
            f"Allowed: {allowed}. No fit was started."
        )


def _construct_adapter(
    estimator: str,
    model: object | None,
    sample_kwargs: Mapping[str, Any] | None,
    random_seed: int | None,
    treated: tuple[str, ...],
) -> EstimatorAdapter:
    model_arg, kwargs_arg, seed_arg = _fit_arguments(model, sample_kwargs, random_seed)
    if estimator == "synthetic_control":
        return SyntheticControlAdapter(
            model=cast("WeightedSumFitter | None", model_arg),
            sample_kwargs=kwargs_arg,
            random_seed=seed_arg,
            treated_units=treated,
        )
    cls = _registered_adapters()[estimator]
    try:
        return cls(model=model_arg, sample_kwargs=kwargs_arg, random_seed=seed_arg)
    except TypeError as exc:
        raise TypeError(
            f"{estimator} adapter construction failed: {exc}. No fit was started."
        ) from exc


def _planned_window(
    frame: pd.DataFrame,
    treatment_time: Any,
    duration: int,
    training_index: tuple[Any, ...],
    holdout_index: tuple[Any, ...],
) -> pd.DataFrame:
    periods = require_positive_int(duration, "duration")
    labels = tuple(training_index)
    holdout_labels = tuple(holdout_index)
    if len(labels) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Planned training window has {len(labels)} rows. At least "
            f"{MIN_TRAINING_ROWS} are required. No fit was started."
        )
    if len(set(labels)) != len(labels):
        raise ValueError("Planned training labels must be unique. No fit was started.")
    missing = [label for label in labels if label not in frame.index]
    if missing:
        raise ValueError(
            "Planned training labels are missing from the realized panel. "
            "No fit was started."
        )
    for label in labels:
        try:
            precedes = label < treatment_time
        except TypeError as exc:
            raise ValueError(
                "Planned training labels must be comparable with treatment_time. "
                "No fit was started."
            ) from exc
        if not bool(precedes):
            raise ValueError(
                "Planned training labels must precede treatment_time. "
                "No fit was started."
            )
    if len(holdout_labels) != periods:
        raise ValueError(
            f"Planned holdout has {len(holdout_labels)} labels, not the "
            f"requested duration {periods}. No fit was started."
        )
    if len(set(holdout_labels)) != len(holdout_labels):
        raise ValueError("Planned holdout labels must be unique. No fit was started.")
    if holdout_labels[0] != treatment_time:
        raise ValueError(
            "The first planned holdout label must be treatment_time. "
            "No fit was started."
        )
    overlap = [label for label in holdout_labels if label in set(labels)]
    if overlap:
        raise ValueError(
            "Planned holdout labels overlap the planned training labels. "
            "No fit was started."
        )
    missing_holdout = [label for label in holdout_labels if label not in frame.index]
    if missing_holdout:
        raise ValueError(
            "Planned holdout labels are missing from the realized panel. "
            "No fit was started."
        )
    for earlier, later in zip(holdout_labels, holdout_labels[1:], strict=False):
        try:
            increases = earlier < later
        except TypeError as exc:
            raise ValueError(
                "Planned holdout labels must be comparable with treatment_time. "
                "No fit was started."
            ) from exc
        if not bool(increases):
            raise ValueError(
                "Planned holdout labels must start at treatment_time and "
                "increase. No fit was started."
            )
    training = frame.loc[list(labels)]
    holdout = frame.loc[list(holdout_labels)]
    window = pd.concat([training, holdout])
    values = list(window.index)
    for earlier, later in zip(values, values[1:], strict=False):
        try:
            increases = earlier < later
        except TypeError as exc:
            raise ValueError(
                "The planned window is not strictly increasing. No fit was started."
            ) from exc
        if not bool(increases):
            raise ValueError(
                "The planned window is not strictly increasing. No fit was started."
            )
    return window


def _require_estimator(estimator: str) -> str:
    if not isinstance(estimator, str):
        raise TypeError("estimator must be a string.")
    if estimator == "synthetic_control":
        return estimator
    if estimator in _registered_adapters():
        return estimator
    if estimator in UNIMPLEMENTED_ESTIMATORS:
        raise ValueError(
            f"{estimator} is a declared adapter seam for #1440. It is not "
            "implemented on this reference path. No fit was started."
        )
    known = ", ".join(DECLARED_ESTIMATORS)
    raise ValueError(
        f"Unknown estimator {estimator!r}. Declared estimators: {known}. "
        "No fit was started."
    )


def _group_message(treated: tuple[str, ...]) -> str:
    return (
        "The synthetic_control reference adapter fits one treated unit, "
        f"got {len(treated)}: {treated}. Group fitting for sizes 1-3 is the "
        "#1440 adapter seam. It must sum unit-level posterior impacts within "
        "each draw. A raw sum of treated outcomes is not comparable to a "
        "simplex donor combination, and summing marginal intervals is not "
        "group inference. No fit was started."
    )


def _prepare_history(history: pd.DataFrame, time_column: str | None) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame):
        raise TypeError("history must be a pandas DataFrame.")
    frame = (
        history.copy()
        if time_column is None
        else to_pandas_with_time_index(history, time_column)
    )
    if not frame.index.is_unique:
        raise ValueError("History index must be unique. No fit was started.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(
            "History index must increase. The holdout is the tail of that "
            "index, while SyntheticControl splits on label order. No fit was "
            "started."
        )
    if not all(isinstance(column, str) for column in frame.columns):
        raise TypeError("History columns must be string unit names.")
    if len(frame) < MIN_TRAINING_ROWS + 1:
        raise ValueError(
            f"History needs at least {MIN_TRAINING_ROWS + 1} rows so a duration "
            "of 1 still leaves two training rows. No fit was started."
        )
    return frame


def _index_realized_panel(data: pd.DataFrame, time_column: str | None) -> pd.DataFrame:
    """Index a realized panel without requiring a unique, increasing tail."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("history must be a pandas DataFrame.")
    if time_column is None:
        frame = data.copy()
    else:
        if time_column not in data.columns:
            raise DataException(
                f"`time_column` '{time_column}' is not a column of `data`. "
                f"Available columns: {list(data.columns)}."
            )
        if not (isinstance(data.index, pd.RangeIndex) and data.index.name is None):
            raise DataException(
                f"`data` already has a meaningful index "
                f"({data.index.name or 'unnamed'}, {type(data.index).__name__}), and "
                f"`time_column` '{time_column}' was also given. Setting the column as "
                "the index would drop the existing one, which would silently change the "
                "time axis. Pass only one: drop `time_column` to keep the index, or "
                "call `.reset_index(drop=True)` on the data to discard the index."
            )
        frame = data.set_index(time_column)
    if not all(isinstance(column, str) for column in frame.columns):
        raise TypeError("History columns must be string unit names.")
    return frame


def _select_planned_rows(
    frame: pd.DataFrame,
    training_index: tuple[Any, ...],
    holdout_index: tuple[Any, ...],
) -> tuple[pd.DataFrame, bool]:
    """Select stored labels by identity. Outside rows are dropped, not checked."""
    planned = tuple(training_index) + tuple(holdout_index)
    duplicated = [label for label in planned if int((frame.index == label).sum()) > 1]
    if duplicated:
        raise ValueError(
            f"Planned label {duplicated[0]!r} is duplicated in the realized panel. "
            "No fit was started."
        )
    excluded = not bool(frame.index.isin(planned).all())
    present = [label for label in planned if label in frame.index]
    narrowed = frame.loc[frame.index.isin(planned)]
    selected = narrowed.loc[present].copy() if present else narrowed.iloc[0:0].copy()
    return selected, excluded


def _require_columns(
    frame: pd.DataFrame,
    treated: tuple[str, ...],
    controls: tuple[str, ...],
) -> None:
    missing = [name for name in treated + controls if name not in frame.columns]
    if missing:
        raise ValueError(f"History is missing units {missing}. No fit was started.")
    overlap = sorted(set(treated) & set(controls))
    if overlap:
        raise ValueError(
            f"Treated and control units overlap: {overlap}. No fit was started."
        )
    try:
        values = frame[list(treated + controls)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Treated and control outcomes must be numeric.") from exc
    if not np.isfinite(values).all():
        raise ValueError(
            "Treated and control outcomes must be finite. No fit was started."
        )


def _effect_values(effects: Sequence[float] | None, name: str) -> tuple[float, ...]:
    if effects is None:
        return ()
    if isinstance(effects, (str, bytes)) or not isinstance(effects, Sequence):
        raise TypeError(f"{name} must be a sequence of finite numbers.")
    return tuple(require_finite(value, name) for value in effects)


def _resolve_sample_kwargs(
    sample_kwargs: Mapping[str, Any] | None,
    random_seed: int | None,
) -> dict[str, Any]:
    if sample_kwargs is None:
        kwargs: dict[str, Any] = {}
    else:
        if not isinstance(sample_kwargs, Mapping):
            raise TypeError("sample_kwargs must be a mapping.")
        kwargs = dict(sample_kwargs)
    if random_seed is not None:
        if isinstance(random_seed, bool) or not isinstance(
            random_seed, (int, np.integer)
        ):
            raise TypeError("random_seed must be an integer.")
        existing = kwargs.get("random_seed")
        if existing is not None and existing != int(random_seed):
            raise ValueError("random_seed disagrees with sample_kwargs['random_seed'].")
        kwargs["random_seed"] = int(random_seed)
    seed = kwargs.get("random_seed")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
    ):
        raise TypeError("sample_kwargs['random_seed'] must be an integer.")
    return kwargs


def _label_window(
    array: xr.DataArray,
    holdout: pd.Index,
    treated: tuple[str, ...],
) -> xr.DataArray:
    labeled = array
    if "treated_units" not in labeled.dims:
        if len(treated) != 1:
            raise ValueError(
                "A multi-unit impact is missing the treated_units dimension."
            )
        labeled = labeled.expand_dims(treated_units=[treated[0]])
    missing = [dim for dim in _IMPACT_DIMS if dim not in labeled.dims]
    if missing:
        raise ValueError(f"Impact is missing dimensions {missing}.")
    ordered = labeled.transpose(*_IMPACT_DIMS)
    if int(ordered.sizes["obs_ind"]) != len(holdout):
        raise ValueError(
            "Fitted holdout length does not match the requested duration: "
            f"{int(ordered.sizes['obs_ind'])} != {len(holdout)}."
        )
    if int(ordered.sizes["treated_units"]) != len(treated):
        raise ValueError("Fitted treated-unit count does not match the request.")
    return ordered.assign_coords(
        obs_ind=("obs_ind", list(holdout)),
        treated_units=("treated_units", list(treated)),
    ).copy(deep=True)


def _diagnostics(experiment: SyntheticControl) -> FitDiagnostics:
    raw = experiment._model_backend.sampling_diagnostics()
    return FitDiagnostics(
        divergences=raw["divergences"],
        max_rhat=raw["max_rhat"],
        min_bulk_ess=raw["min_bulk_ess"],
        diagnostics_status=raw["diagnostics_status"],
        diagnostics_reason=raw["diagnostics_reason"],
    )


def _provenance(
    sample_kwargs: Mapping[str, Any],
    *,
    translation: str,
) -> Provenance:
    seed = sample_kwargs.get("random_seed")
    if translation not in ("deterministic_additive", "observed"):
        raise ValueError(f"Unknown translation {translation!r}.")
    return Provenance(
        estimator="synthetic_control",
        adapter="SyntheticControlAdapter",
        model_class="WeightedSumFitter",
        random_seed=None if seed is None else int(seed),
        sample_kwargs=tuple(sample_kwargs.items()),
        horizon_policy="separate-fits",
        translation=translation,  # type: ignore[arg-type]
        uncertainty="posterior_conditional_mean",
    )


def _estimand(draws: EstimandDraws, name: str) -> xr.DataArray:
    chosen = getattr(draws, name)
    if not isinstance(chosen, xr.DataArray):
        raise ValueError(f"Estimand {name!r} is not available on these draws.")
    return chosen
