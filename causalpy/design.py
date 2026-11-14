#   Copyright 2026 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Fixed-target planning for Bayesian simplex synthetic control.

:class:`SyntheticControlPlanner` fits one weighted-sum synthetic control per
requested horizon, then translates that fitted counterfactual across an
absolute and relative lift grid. Construction and :meth:`screen` do not
sample. This is not a power curve, not a second probability engine, and not
:class:`~causalpy.checks.placebo_in_time.PlaceboInTime`.

The draft design surface previously attached to
:class:`~causalpy.experiments.synthetic_control.SyntheticControl`
(``from_pre_period``, ``validate_design``, ``donor_pool_quality``, and
``power_analysis``) is not part of this API. ``power_analysis`` is not a
compatibility promise. ``DressRehearsalCheck``, ``causalpy.experiments.sc_results``,
and the tests for the noise-injection curve, the ``CheckResult`` wrapper,
posterior-weight donor quality, and ``from_pre_period`` were dropped rather
than adapted.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from causalpy._arviz_compat import hdi_bounds
from causalpy.constants import HDI_PROB
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.input_data import to_pandas_with_time_index
from causalpy.pymc_models import WeightedSumFitter
from causalpy.utils import check_convex_hull_violation

__all__ = [
    "DecisionRule",
    "HorizonEvaluation",
    "LiftScore",
    "PlannerResult",
    "ScreenResult",
    "SyntheticControlPlanner",
]

HORIZON_POLICY = "separate-fits"
_RELATIVE_FLOOR_SCALE = 1e-8
_MIN_TRAINING_ROWS = 2
_ESTIMANDS = ("cumulative", "average")
_SCALES = ("absolute", "relative")

_TWO_SIDED = (
    "Two-sided decisions are not supported. Pass direction='positive' or "
    "direction='negative' as the single explicit one-sided rule. Taking the "
    "absolute effect is not a two-sided decision."
)
_GROUPS = (
    "Groups are not supported. Pass one treated unit name as a string. "
    "Averaging relative effects across unequal units is not defined."
)
_ESTIMATOR = (
    "SyntheticControlPlanner supports Bayesian simplex synthetic control only "
    "(WeightedSumFitter, estimator='synthetic_control'). Softmax, augmented, "
    "interactive fixed-effects, scikit-learn, and grouped estimators are not "
    "supported."
)


def _require_real(name: str, value: object) -> float:
    """Return a finite real number, rejecting booleans and non-numeric values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return number


@dataclass(frozen=True)
class DecisionRule:
    """One explicit one-sided decision applied after effect translation.

    ``probability`` is the tail probability the decision requires. It is not
    the HDI mass used for interval coverage. Truth coverage of a translated
    interval is not this decision and is not a detection curve.

    Attributes
    ----------
    direction : {'positive', 'negative'}
        ``positive`` detects when the share of draws above ``threshold`` is
        at least ``probability``. ``negative`` uses draws below ``threshold``.
        A draw equal to ``threshold`` does not count. ``two-sided`` is rejected.
    estimand : {'cumulative', 'average'}
        Which translated aggregate the decision reads. The other estimand is
        still reported.
    threshold : float
        Finite decision threshold on the translated estimand.
    probability : float
        Required tail probability, strictly above 0 and at most 1.
    """

    direction: Literal["positive", "negative"]
    estimand: Literal["cumulative", "average"]
    threshold: float
    probability: float

    def __post_init__(self) -> None:
        if self.direction == "two-sided":
            raise ValueError(_TWO_SIDED)
        if self.direction not in {"positive", "negative"}:
            raise ValueError(
                f"Unsupported decision direction {self.direction!r}. "
                "Pass 'positive' or 'negative'. " + _TWO_SIDED
            )
        if self.estimand not in _ESTIMANDS:
            raise ValueError(
                f"Unsupported estimand {self.estimand!r}. "
                "Pass 'cumulative' or 'average'."
            )
        threshold = _require_real("threshold", self.threshold)
        probability = _require_real("probability", self.probability)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "probability", probability)
        if not 0.0 < probability <= 1.0:
            raise ValueError(
                "probability must be strictly above 0 and at most 1, "
                f"got {probability}."
            )


@dataclass(frozen=True)
class LiftScore:
    """One translated lift on one horizon and one estimand.

    ``counterfactual`` is the fitted posterior mean path shared by every lift
    on this horizon, including the paired null. ``effect`` is that path
    translated to this lift and aggregated to ``estimand``. ``truth_covers``
    reports whether the translated HDI contains the injected truth. It is not
    ``detected``. ``tail_probability`` and ``detected`` are empty unless this
    score's estimand is the declared decision. An empty detection is not a
    non-detection.
    """

    effect_scale: Literal["absolute", "relative"]
    lift: float
    estimand: Literal["cumulative", "average"]
    mean: float
    hdi_lower: float
    hdi_upper: float
    hdi_prob: float
    truth: float
    truth_covers: bool
    tail_probability: float | None
    detected: bool | None
    effect: np.ndarray
    counterfactual: xr.DataArray


@dataclass(frozen=True)
class HorizonEvaluation:
    """One horizon's single fit and the lifts translated from it.

    A failed fit has empty ``scores`` and a ``fit_error``. Empty scores are
    not non-detections.
    """

    duration: int
    treatment_time: Any
    training_index: pd.Index
    holdout_index: pd.Index
    status: Literal["historical-descriptive", "fit-failed"]
    fit_error: str | None
    n_inference_fits: int
    counterfactual: xr.DataArray | None
    baseline_impact: xr.DataArray | None
    observed_holdout: xr.DataArray | None
    applied_relative_floor: float | None
    observation_noise_rate: np.ndarray | None
    y_hat_distribution: str | None
    user_priors: dict[str, Any]
    scores: tuple[LiftScore, ...]

    def __repr__(self) -> str:
        return (
            "HorizonEvaluation("
            f"duration={self.duration}, status={self.status!r}, "
            f"scores={len(self.scores)})"
        )


@dataclass(frozen=True)
class ScreenResult:
    """Donor-pool diagnostics computed from observed history only.

    This is a historical description, not a search result and not a launch
    recommendation. Weight concentration is omitted because it requires a
    posterior.
    """

    treated_unit: str
    control_units: tuple[str, ...]
    periods: pd.Index
    correlation: dict[str, float]
    convex_hull_passes: bool
    convex_hull_violations: int
    convex_hull_pct_above: float
    convex_hull_pct_below: float

    def summary(self) -> pd.DataFrame:
        """Return one row per donor with its correlation and the shared hull."""
        return pd.DataFrame(
            {
                "donor": list(self.control_units),
                "correlation": [self.correlation[name] for name in self.control_units],
                "convex_hull_passes": self.convex_hull_passes,
                "convex_hull_violations": self.convex_hull_violations,
            }
        )


@dataclass(frozen=True)
class PlannerResult:
    """Translated rehearsal for one fixed treated unit.

    ``status`` is ``historical-descriptive`` when every requested horizon
    fitted, ``fit-failed`` when every horizon failed, and ``mixed`` when at
    least one of each occurred. A failed fit does not report a detection.
    ``n_inference_fits`` counts fit attempts, not lift-grid rows.
    ``horizon_policy`` is ``separate-fits``: each duration is fitted on its
    own tail. Prefix reuse is not enabled.
    """

    status: Literal["historical-descriptive", "fit-failed", "mixed"]
    fit_error: str | None
    decision: DecisionRule
    estimator: str
    treated_unit: str
    control_units: tuple[str, ...]
    relative_floor: float | None
    relative_baseline_convention: str
    horizon_policy: str
    random_seed: int | None
    n_inference_fits: int
    horizons: tuple[HorizonEvaluation, ...]

    def __repr__(self) -> str:
        return (
            "PlannerResult("
            f"status={self.status!r}, treated_unit={self.treated_unit!r}, "
            f"horizons={len(self.horizons)}, "
            f"n_inference_fits={self.n_inference_fits})"
        )

    def recommendation(self) -> str:
        """Return the only claim this result supports."""
        if self.status == "fit-failed":
            detail = f" {self.fit_error}" if self.fit_error else ""
            return "No detection is reported: the counterfactual fit failed." + detail
        if self.status == "mixed":
            return (
                "No single recommendation: at least one horizon failed to fit, "
                "and a failed fit is not a non-detection. Successful horizons "
                "are historical-descriptive only. Truth coverage of a translated "
                "interval is not a detection curve and is not a launch recommendation."
            )
        return (
            "Historical-descriptive rehearsal only. Truth coverage of a translated "
            "interval is not a detection curve and is not a launch recommendation."
        )

    def summary(self) -> pd.DataFrame:
        """Return one row per translated score, plus a row for each failed fit.

        Failed rows leave the decision columns empty. An empty detection is
        not a non-detection.
        """
        rows: list[dict[str, Any]] = []
        for horizon in self.horizons:
            if horizon.status == "fit-failed":
                rows.append(
                    {
                        "duration": horizon.duration,
                        "effect_scale": pd.NA,
                        "lift": pd.NA,
                        "estimand": pd.NA,
                        "mean": pd.NA,
                        "hdi_lower": pd.NA,
                        "hdi_upper": pd.NA,
                        "truth": pd.NA,
                        "truth_covers": pd.NA,
                        "tail_probability": pd.NA,
                        "detected": pd.NA,
                        "decision_estimand": self.decision.estimand,
                        "status": horizon.status,
                        "fit_error": horizon.fit_error,
                    }
                )
                continue
            for score in horizon.scores:
                rows.append(
                    {
                        "duration": horizon.duration,
                        "effect_scale": score.effect_scale,
                        "lift": score.lift,
                        "estimand": score.estimand,
                        "mean": score.mean,
                        "hdi_lower": score.hdi_lower,
                        "hdi_upper": score.hdi_upper,
                        "truth": score.truth,
                        "truth_covers": score.truth_covers,
                        "tail_probability": (
                            pd.NA
                            if score.tail_probability is None
                            else score.tail_probability
                        ),
                        "detected": pd.NA if score.detected is None else score.detected,
                        "decision_estimand": self.decision.estimand,
                        "status": horizon.status,
                        "fit_error": pd.NA,
                    }
                )
        return pd.DataFrame(rows)

    def plot(
        self,
        *,
        horizon: int | None = None,
        effect_scale: Literal["absolute", "relative"] | None = None,
        estimand: Literal["cumulative", "average"] | None = None,
        figsize: tuple[float, float] = (7.0, 4.5),
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot one horizon's translated mean and HDI against the lift grid.

        Parameters
        ----------
        horizon : int, optional
            Duration to plot. The default is the first successful horizon.
        effect_scale : {'absolute', 'relative'}, optional
            Lift scale to plot. The default prefers relative when that scale
            was requested, otherwise absolute.
        estimand : {'cumulative', 'average'}, optional
            Aggregate to plot. The default is the decision estimand.
        figsize : tuple of float, default (7.0, 4.5)
            Figure size in inches.

        Returns
        -------
        tuple of matplotlib Figure and Axes
            The translated-effect figure. Interval coverage is not drawn as a
            detection curve.
        """
        chosen = _select_horizon(self, horizon)
        if chosen.status == "fit-failed":
            raise ValueError(
                "Cannot plot a failed fit. A failed fit is not a non-detection. "
                f"{chosen.fit_error}"
            )
        scale = effect_scale or _default_scale(chosen)
        shown_estimand = estimand or self.decision.estimand
        if scale not in _SCALES:
            raise ValueError(
                f"effect_scale must be 'absolute' or 'relative', got {scale!r}."
            )
        if shown_estimand not in _ESTIMANDS:
            raise ValueError(
                f"estimand must be 'cumulative' or 'average', got {shown_estimand!r}."
            )
        scores = sorted(
            (
                score
                for score in chosen.scores
                if score.effect_scale == scale and score.estimand == shown_estimand
            ),
            key=lambda score: score.lift,
        )
        if not scores:
            raise ValueError(
                f"No {scale} {shown_estimand} scores on the {chosen.duration}-row horizon."
            )
        lifts = [score.lift for score in scores]
        means = [score.mean for score in scores]
        lower = [score.hdi_lower for score in scores]
        upper = [score.hdi_upper for score in scores]
        fig, ax = plt.subplots(figsize=figsize)
        ax.fill_between(
            lifts,
            lower,
            upper,
            alpha=0.25,
            label=f"{chosen.scores[0].hdi_prob:.0%} translated HDI",
        )
        ax.plot(lifts, means, marker="o", label="translated mean")
        if shown_estimand == self.decision.estimand:
            ax.axhline(
                self.decision.threshold, color="0.4", linestyle="--", linewidth=1
            )
            detected = [score for score in scores if score.detected]
            if detected:
                ax.scatter(
                    [score.lift for score in detected],
                    [score.mean for score in detected],
                    zorder=3,
                    label="decision rule met",
                )
        ax.set_xlabel(f"Injected {scale} lift")
        ax.set_ylabel(f"Translated {shown_estimand} effect")
        ax.set_title(
            f"Horizon {chosen.duration}: translated {shown_estimand} {scale} effect.\n"
            "Interval coverage of the injected truth is not a detection curve."
        )
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig, ax


class SyntheticControlPlanner:
    """Plan one fixed Bayesian simplex synthetic-control test.

    Construction stores a copy of the history and an unfitted model template.
    It does not sample. :meth:`evaluate` fits each requested horizon once and
    translates that fit. :meth:`screen` reports donor correlation and convex-hull
    coverage without fitting.

    Each duration holds out its own tail and is fitted separately. A shorter
    duration is not a prefix of a longer one: it trains on rows the longer
    horizon holds out. Prefix reuse is not enabled. Enabling it would require
    an equivalence test against these separate fits and a data-boundary test
    showing the reused fit does not see donor outcomes past the shorter
    horizon.

    Parameters
    ----------
    data : pandas.DataFrame
        Observed history. The last ``duration`` rows are the planned holdout,
        not an observed post-period. At least three rows are required.
    treated_unit : str
        The single unit the decision concerns. A sequence is a group and is
        rejected.
    control_units : sequence of str
        Donor pool. The treated unit cannot be a donor.
    decision : DecisionRule
        The one explicit one-sided decision.
    estimator : str, default 'synthetic_control'
        Must be ``synthetic_control``. Any other name is rejected.
    model : WeightedSumFitter, optional
        Unfitted template. The default is a new :class:`WeightedSumFitter`.
        The planner clones it before fitting, so this object is not mutated
        and is not sampled at construction. Other model classes are rejected.
    random_seed : int, optional
        Seed written onto each cloned fit unless ``evaluate`` passes
        ``sample_kwargs['random_seed']``.
    relative_floor : float, optional
        Declared minimum observed holdout outcome for a relative lift. When
        omitted, the floor is ``1e-8`` times the median absolute training
        outcome. A declared floor replaces that scale rule. Non-positive
        outcomes are always rejected: no signed-baseline convention is offered.
        Absolute lifts do not use this floor.
    time_column : str, optional
        Column that becomes the history index at construction. The analysis
        fit receives that indexed frame and does not see the column again.
    min_donor_correlation : float, default 0.0
        Warning threshold forwarded to
        :class:`~causalpy.experiments.synthetic_control.SyntheticControl`.
        It does not drop donors and does not turn screening into a search.
    auto_scale_sigma : bool, default True
        Forwarded to the analysis fit. A custom ``y_hat`` prior still overrides
        the data-scaled observation-noise prior.
    """

    horizon_policy = HORIZON_POLICY

    def __init__(
        self,
        data: pd.DataFrame,
        treated_unit: str,
        control_units: Sequence[str],
        decision: DecisionRule,
        *,
        estimator: str = "synthetic_control",
        model: WeightedSumFitter | None = None,
        random_seed: int | None = None,
        relative_floor: float | None = None,
        time_column: str | None = None,
        min_donor_correlation: float = 0.0,
        auto_scale_sigma: bool = True,
    ) -> None:
        if not isinstance(treated_unit, str):
            raise TypeError(_GROUPS)
        if estimator != "synthetic_control":
            raise ValueError(f"Unsupported estimator {estimator!r}. " + _ESTIMATOR)
        if model is not None and not isinstance(model, WeightedSumFitter):
            raise TypeError(
                f"Unsupported estimator model {type(model).__name__}. " + _ESTIMATOR
            )
        if not isinstance(decision, DecisionRule):
            raise TypeError("decision must be a DecisionRule.")
        if not isinstance(auto_scale_sigma, bool):
            raise TypeError("auto_scale_sigma must be a bool.")
        if relative_floor is not None:
            relative_floor = _require_real("relative_floor", relative_floor)
            if relative_floor < 0.0:
                raise ValueError(
                    "relative_floor must be at least 0. A negative floor would "
                    "accept signed baselines, and no signed-baseline convention "
                    "is offered."
                )
        if random_seed is not None and (
            isinstance(random_seed, bool) or not isinstance(random_seed, int)
        ):
            raise TypeError("random_seed must be an int or None.")
        min_donor_correlation = _require_real(
            "min_donor_correlation", min_donor_correlation
        )
        if not -1.0 <= min_donor_correlation <= 1.0:
            raise ValueError(
                "min_donor_correlation must be between -1 and 1, "
                f"got {min_donor_correlation}."
            )
        if isinstance(control_units, (str, bytes)) or not isinstance(
            control_units, Sequence
        ):
            raise TypeError(
                "control_units must be a sequence of unit-name strings, "
                "not a single string."
            )
        donors = tuple(control_units)
        if not donors:
            raise ValueError("control_units must name at least one donor.")
        if any(not isinstance(name, str) for name in donors):
            raise TypeError("control_units must be a sequence of unit-name strings.")
        if len(set(donors)) != len(donors):
            raise ValueError("control_units contains a duplicate donor.")
        if treated_unit in donors:
            raise ValueError("treated_unit cannot also be a control unit.")
        frame = to_pandas_with_time_index(data, time_column, argument_name="data")
        missing = [
            name for name in (treated_unit, *donors) if name not in frame.columns
        ]
        if missing:
            raise ValueError(f"history is missing columns {missing}.")
        if not frame.index.is_unique:
            raise ValueError("history index must be unique.")
        if not frame.index.is_monotonic_increasing:
            raise ValueError("history index must be sorted ascending.")
        if len(frame) < _MIN_TRAINING_ROWS + 1:
            raise ValueError(
                "history needs at least three rows: two to train and one holdout."
            )
        outcomes = frame[[treated_unit, *donors]].to_numpy(dtype=float)
        if not np.isfinite(outcomes).all():
            raise ValueError(
                "history contains non-finite outcomes in the selected units."
            )
        if model is None:
            template_kwargs: dict[str, Any] = {"progressbar": False}
            if random_seed is not None:
                template_kwargs["random_seed"] = random_seed
            model = WeightedSumFitter(sample_kwargs=template_kwargs)
        self.history = frame.copy()
        self.treated_unit = treated_unit
        self.control_units = donors
        self.decision = decision
        self.estimator = estimator
        self.model = model
        self.random_seed = random_seed
        self.relative_floor = relative_floor
        self.time_column = time_column
        self.min_donor_correlation = min_donor_correlation
        self.auto_scale_sigma = auto_scale_sigma

    def screen(
        self, *, periods: pd.Index | Sequence[Any] | None = None
    ) -> ScreenResult:
        """Report donor correlation and convex-hull coverage without fitting.

        Parameters
        ----------
        periods : index or sequence, optional
            History labels to screen. The default is the full stored history.
            This restriction is not a search over candidate windows.

        Returns
        -------
        ScreenResult
            Correlations and the convex-hull summary. No posterior weights.
        """
        if periods is None:
            frame = self.history
        else:
            labels = pd.Index(periods)
            missing = labels.difference(self.history.index)
            if len(missing):
                raise KeyError(
                    f"screen periods {list(missing)!r} are not in the stored history."
                )
            frame = self.history.loc[labels]
        if len(frame) < 2:
            raise ValueError(
                "screen() needs at least two observations to report a correlation."
            )
        treated = frame[self.treated_unit]
        correlation = {
            name: float(treated.corr(frame[name])) for name in self.control_units
        }
        hull = check_convex_hull_violation(
            treated.to_numpy(dtype=float),
            frame[list(self.control_units)].to_numpy(dtype=float),
        )
        return ScreenResult(
            treated_unit=self.treated_unit,
            control_units=self.control_units,
            periods=frame.index,
            correlation=correlation,
            convex_hull_passes=bool(hull["passes"]),
            convex_hull_violations=int(hull["n_violations"]),
            convex_hull_pct_above=float(hull["pct_above"]),
            convex_hull_pct_below=float(hull["pct_below"]),
        )

    def evaluate(
        self,
        *,
        durations: Sequence[int],
        relative_lifts: Sequence[float] | None = None,
        absolute_lifts: Sequence[float] | None = None,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> PlannerResult:
        """Fit each horizon once and translate that fit across the lift grid.

        Parameters
        ----------
        durations : sequence of int
            Holdout lengths, in trailing history rows. Each length is fitted
            separately. Duplicate lengths are fitted once, in first-seen order.
        relative_lifts, absolute_lifts : sequence of float, optional
            Injected lifts. At least one scale is required. A zero lift is
            always added to each requested scale as the paired null. Relative
            lifts use the treated unit's observed holdout outcome as the
            denominator and are rejected when that baseline is undefined.
        sample_kwargs : dict, optional
            Overrides merged onto the cloned model's sampling arguments for
            this call only. They do not change the template stored on the planner.

        Returns
        -------
        PlannerResult
            Translated scores, or an explicit ``fit-failed`` status. Increasing
            the lift grid does not add inference fits.
        """
        if isinstance(durations, (str, bytes)) or not isinstance(durations, Sequence):
            raise TypeError("durations must be a sequence of positive integers.")
        if not len(durations):
            raise ValueError("Pass at least one duration.")
        requested: list[int] = []
        for duration in durations:
            if isinstance(duration, bool) or not isinstance(duration, int):
                raise TypeError(
                    f"each duration must be a positive integer, got {duration!r}."
                )
            if duration < 1:
                raise ValueError(f"duration must be at least 1, got {duration}.")
            training_rows = len(self.history) - duration
            if training_rows < _MIN_TRAINING_ROWS:
                raise ValueError(
                    f"duration {duration} leaves {training_rows} training rows in "
                    f"a history of {len(self.history)}. At least "
                    f"{_MIN_TRAINING_ROWS} training rows are required. A longer "
                    "duration is not a prefix of a shorter one; fit a shorter "
                    "horizon separately."
                )
            if duration not in requested:
                requested.append(duration)
        relative = _ordered_lifts("relative_lifts", relative_lifts)
        absolute = _ordered_lifts("absolute_lifts", absolute_lifts)
        if relative is None and absolute is None:
            raise ValueError(
                "Pass at least one absolute or relative lift. A zero lift is "
                "added as the paired null on each requested scale."
            )
        windows = [
            _window(self.history, duration, self.treated_unit) for duration in requested
        ]
        if relative is not None:
            for window in windows:
                _require_relative_baseline(
                    window.observed.to_numpy(dtype=float),
                    window.training_observed.to_numpy(dtype=float),
                    declared_floor=self.relative_floor,
                    unit=self.treated_unit,
                    holdout_index=window.holdout_index,
                )
        horizons: list[HorizonEvaluation] = []
        for window in windows:
            horizons.append(
                self._fit_horizon(
                    window,
                    relative=relative,
                    absolute=absolute,
                    sample_kwargs=sample_kwargs,
                )
            )
        failed = [horizon for horizon in horizons if horizon.status == "fit-failed"]
        if len(failed) == len(horizons):
            status: Literal["historical-descriptive", "fit-failed", "mixed"] = (
                "fit-failed"
            )
        elif failed:
            status = "mixed"
        else:
            status = "historical-descriptive"
        return PlannerResult(
            status=status,
            fit_error=failed[0].fit_error if failed else None,
            decision=self.decision,
            estimator=self.estimator,
            treated_unit=self.treated_unit,
            control_units=self.control_units,
            relative_floor=self.relative_floor,
            relative_baseline_convention=(
                "observed-holdout" if relative is not None else "not-requested"
            ),
            horizon_policy=self.horizon_policy,
            random_seed=self.random_seed,
            n_inference_fits=sum(horizon.n_inference_fits for horizon in horizons),
            horizons=tuple(horizons),
        )

    def _fit_horizon(
        self,
        window: "_Window",
        *,
        relative: tuple[float, ...] | None,
        absolute: tuple[float, ...] | None,
        sample_kwargs: dict[str, Any] | None,
    ) -> HorizonEvaluation:
        template = self._fit_template(sample_kwargs)
        experiment = SyntheticControl(
            self.history.copy(),
            window.treatment_time,
            control_units=list(self.control_units),
            treated_units=[self.treated_unit],
            model=template,
            min_donor_correlation=self.min_donor_correlation,
            auto_scale_sigma=self.auto_scale_sigma,
        )
        try:
            experiment.fit()
        except Exception as exc:
            return _failed_horizon(window, f"{type(exc).__name__}: {exc}")
        if experiment.datapre.index.tolist() != window.training_index.tolist():
            raise RuntimeError(
                "The analysis fit trained on a different window than the "
                "requested horizon. Separate horizon fits are the contract; "
                "refusing to score a drifted boundary."
            )
        if experiment.datapost.index.tolist() != window.holdout_index.tolist():
            raise RuntimeError(
                "The analysis fit held out a different window than the "
                "requested horizon."
            )
        counterfactual = _select_unit(
            experiment.result.predictions_post, self.treated_unit
        )
        baseline_impact = _select_unit(experiment.result.impact_post, self.treated_unit)
        observed = xr.DataArray(
            experiment.datapost[self.treated_unit].to_numpy(dtype=float),
            dims="obs_ind",
            coords={"obs_ind": baseline_impact.coords["obs_ind"]},
        )
        floor = None
        if relative is not None:
            floor = _applied_relative_floor(
                window.training_observed.to_numpy(dtype=float),
                self.relative_floor,
            )
        scores = _score_translated_lifts(
            baseline_impact=baseline_impact,
            observed=observed,
            counterfactual=counterfactual,
            relative_lifts=relative,
            absolute_lifts=absolute,
            decision=self.decision,
        )
        return HorizonEvaluation(
            duration=window.duration,
            treatment_time=window.treatment_time,
            training_index=window.training_index,
            holdout_index=window.holdout_index,
            status="historical-descriptive",
            fit_error=None,
            n_inference_fits=1,
            counterfactual=counterfactual,
            baseline_impact=baseline_impact,
            observed_holdout=observed,
            applied_relative_floor=floor,
            observation_noise_rate=_observation_noise_rate(experiment.model),
            y_hat_distribution=str(experiment.model.priors["y_hat"].distribution),
            user_priors=dict(experiment.model._user_priors or {}),
            scores=scores,
        )

    def _fit_template(self, sample_kwargs: dict[str, Any] | None) -> WeightedSumFitter:
        cloned = self.model._clone()
        merged = dict(cloned.sample_kwargs)
        if self.random_seed is not None:
            merged["random_seed"] = self.random_seed
        if sample_kwargs:
            merged.update(sample_kwargs)
        merged.setdefault("progressbar", False)
        cloned.sample_kwargs = merged
        return cloned


@dataclass(frozen=True)
class _Window:
    duration: int
    treatment_time: Any
    training_index: pd.Index
    holdout_index: pd.Index
    observed: pd.Series
    training_observed: pd.Series


def _window(history: pd.DataFrame, duration: int, treated_unit: str) -> _Window:
    holdout_index = history.index[-duration:]
    training_index = history.index[:-duration]
    return _Window(
        duration=duration,
        treatment_time=holdout_index[0],
        training_index=training_index,
        holdout_index=holdout_index,
        observed=history.loc[holdout_index, treated_unit],
        training_observed=history.loc[training_index, treated_unit],
    )


def _failed_horizon(window: _Window, message: str) -> HorizonEvaluation:
    return HorizonEvaluation(
        duration=window.duration,
        treatment_time=window.treatment_time,
        training_index=window.training_index,
        holdout_index=window.holdout_index,
        status="fit-failed",
        fit_error=message,
        n_inference_fits=1,
        counterfactual=None,
        baseline_impact=None,
        observed_holdout=None,
        applied_relative_floor=None,
        observation_noise_rate=None,
        y_hat_distribution=None,
        user_priors={},
        scores=(),
    )


def _ordered_lifts(
    name: str,
    requested: Sequence[float] | None,
) -> tuple[float, ...] | None:
    if requested is None:
        return None
    if isinstance(requested, (str, bytes)) or not isinstance(requested, Sequence):
        raise TypeError(f"{name} must be a sequence of finite lifts.")
    if len(requested) == 0:
        raise ValueError(
            f"{name} is empty. Omit the scale, or pass at least one lift. "
            "A zero lift is added automatically as the paired null."
        )
    ordered = [0.0]
    for lift in requested:
        number = _require_real(name, lift)
        if any(np.isclose(number, existing) for existing in ordered):
            continue
        ordered.append(number)
    return tuple(ordered)


def _applied_relative_floor(
    training_observed: np.ndarray,
    declared_floor: float | None,
) -> float:
    if declared_floor is not None:
        return float(declared_floor)
    return _RELATIVE_FLOOR_SCALE * float(np.median(np.abs(training_observed)))


def _require_relative_baseline(
    observed: np.ndarray,
    training_observed: np.ndarray,
    *,
    declared_floor: float | None,
    unit: str,
    holdout_index: pd.Index,
) -> float:
    """Reject a relative lift whose observed-holdout denominator is undefined."""
    scale = float(np.median(np.abs(training_observed)))
    if declared_floor is None and not (_RELATIVE_FLOOR_SCALE * scale) > 0.0:
        raise ValueError(
            "Relative effects are undefined because the training outcomes do not "
            "set a strictly positive near-zero floor. The scale rule is "
            f"{_RELATIVE_FLOOR_SCALE:g} times the median absolute training outcome "
            f"({scale}). Declare relative_floor to choose a convention. "
            "relative_floor=0 accepts every strictly positive holdout and still "
            "rejects non-positive outcomes. No signed-baseline convention is offered."
        )
    floor = _applied_relative_floor(training_observed, declared_floor)
    invalid = ~np.isfinite(observed) | (observed <= 0.0) | (observed <= floor)
    if np.any(invalid):
        labels = [repr(label) for label in holdout_index[invalid]]
        convention = (
            f"The declared relative_floor is {floor}."
            if declared_floor is not None
            else (
                "No relative_floor was declared, so the floor is "
                f"{_RELATIVE_FLOOR_SCALE:g} times the median absolute training "
                f"outcome ({floor})."
            )
        )
        raise ValueError(
            "Relative effects are undefined for holdout observation(s) "
            f"{labels} on treated unit {unit!r}. A relative effect is the lift "
            "times that unit's observed holdout outcome, not the counterfactual "
            "mean. Non-positive observations are rejected because no "
            "signed-baseline convention is offered. " + convention
        )
    return floor


def _select_unit(draws: xr.DataArray, unit: str) -> xr.DataArray:
    if "treated_units" not in draws.dims:
        return draws
    return draws.sel(treated_units=unit)


def _observation_noise_rate(model: WeightedSumFitter) -> np.ndarray | None:
    y_hat = model.priors.get("y_hat")
    if y_hat is None:
        return None
    sigma = y_hat.parameters.get("sigma")
    if sigma is None or not hasattr(sigma, "parameters"):
        return None
    lam = sigma.parameters.get("lam")
    if lam is None:
        return None
    return np.asarray(lam, dtype=float).copy()


def _score_translated_lifts(
    *,
    baseline_impact: xr.DataArray,
    observed: xr.DataArray,
    counterfactual: xr.DataArray,
    relative_lifts: tuple[float, ...] | None,
    absolute_lifts: tuple[float, ...] | None,
    decision: DecisionRule,
) -> tuple[LiftScore, ...]:
    """Translate one fitted impact path without refitting.

    Adding a constant to every draw and to the injected truth leaves HDI
    coverage unchanged. Coverage is therefore not a detection curve. The
    decision reads the tail probability of the translated draws for its own
    estimand only. The other estimand keeps its translated interval and leaves
    detection empty.
    """

    scores: list[LiftScore] = []
    requested = (("relative", relative_lifts), ("absolute", absolute_lifts))
    for scale, lifts in requested:
        if lifts is None:
            continue
        for lift in lifts:
            delta = (
                observed * lift if scale == "relative" else xr.full_like(observed, lift)
            )
            translated = baseline_impact + delta
            for estimand in _ESTIMANDS:
                effect = (
                    translated.sum("obs_ind")
                    if estimand == "cumulative"
                    else translated.mean("obs_ind")
                )
                truth = float(
                    delta.sum().item()
                    if estimand == "cumulative"
                    else delta.mean().item()
                )
                lower, upper = hdi_bounds(effect, prob=HDI_PROB, dim=("chain", "draw"))
                flat = np.asarray(effect.values, dtype=float).reshape(-1)
                if estimand == decision.estimand:
                    if decision.direction == "positive":
                        tail = float(np.mean(flat > decision.threshold))
                    else:
                        tail = float(np.mean(flat < decision.threshold))
                    detected: bool | None = bool(tail >= decision.probability)
                else:
                    tail = None
                    detected = None
                scores.append(
                    LiftScore(
                        effect_scale=scale,  # type: ignore[arg-type]
                        lift=float(lift),
                        estimand=estimand,  # type: ignore[arg-type]
                        mean=float(effect.mean().item()),
                        hdi_lower=lower,
                        hdi_upper=upper,
                        hdi_prob=HDI_PROB,
                        truth=truth,
                        truth_covers=bool(lower <= truth <= upper),
                        tail_probability=tail,
                        detected=detected,
                        effect=flat.copy(),
                        counterfactual=counterfactual,
                    )
                )
    return tuple(scores)


def _select_horizon(result: PlannerResult, duration: int | None) -> HorizonEvaluation:
    if duration is None:
        for horizon in result.horizons:
            if horizon.status == "historical-descriptive":
                return horizon
        return result.horizons[0]
    for horizon in result.horizons:
        if horizon.duration == duration:
            return horizon
    raise KeyError(f"No horizon with duration {duration}.")


def _default_scale(horizon: HorizonEvaluation) -> str:
    scales = {score.effect_scale for score in horizon.scores}
    if "relative" in scales:
        return "relative"
    if "absolute" in scales:
        return "absolute"
    raise ValueError("The selected horizon has no scores to plot.")
