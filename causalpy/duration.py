#   Copyright 2022 - 2026 The PyMC Labs Developers
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
"""Common-origin duration comparison for a fixed synthetic-control design.

This is not the trailing-tail rehearsal in
:meth:`~causalpy.design.SyntheticControlPlanner.evaluate`. Each requested
duration is its own fit. Training is the pre-origin prefix from
:func:`~causalpy.historical_validation.validate_schedule`, so a shorter
holdout is a prefix of a longer one and the training index does not depend
on the duration. Prefix reuse stays disabled: no equivalence test covers it.

The fit and the lift translation are the historical-replay and planner
scorers. Lifts do not add fits. Nothing here selects a duration, emits a
launch recommendation, or upgrades a historical-descriptive replay into a
power claim.
"""

from __future__ import annotations

import hashlib
import platform
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

import causalpy.historical_validation as historical_validation
from causalpy.pymc_models import WeightedSumFitter
from causalpy.version import __version__

if TYPE_CHECKING:
    from causalpy.design import SyntheticControlPlanner
    from causalpy.experiments.synthetic_control import SyntheticControl
    from causalpy.historical_validation import (
        HistoricalSchedule,
        WindowPlan,
        WindowReplay,
    )

__all__ = [
    "DEFAULT_MAX_FITS",
    "MAX_WINDOW_BYTES",
    "AnalysisSpecification",
    "DurationAudit",
    "DurationComparison",
    "DurationPreview",
    "DurationProfile",
    "DurationRow",
    "DurationWindow",
    "OriginEligibility",
]

DEFAULT_MAX_FITS = 24
"""Counterfactual-fit ceiling, including the learned-null upper bound.

Eight origins across three durations is 24. A learned-null fit is at most
one extra model per duration and is inside this ceiling, not added on top.
Pass ``max_fits`` to raise or lower it. Zero is a valid refusal.
"""

MAX_WINDOW_BYTES = 32 * 1024 * 1024
"""Per-window impact-array ceiling. A larger estimate refuses the comparison."""

_IMPACT_ARRAYS = 4
"""Conservative count: baseline, counterfactual, one translated effect, one copy."""

_NO_RECOMMENDATION = "No supported recommendation."
_TABLE_RECOMMENDATION = "not-supported"
_SCALES = ("absolute", "relative")
_REFUSAL_STATUSES = (
    "insufficient-history",
    "budget-exhausted",
    "memory-exhausted",
)
DurationStatus = Literal[
    "ready",
    "historical-descriptive",
    "fit-failed",
    "mixed",
    "insufficient-history",
    "budget-exhausted",
    "memory-exhausted",
]
AuditStatus = Literal["historical-descriptive", "inconclusive", "limitation"]


@dataclass(frozen=True, eq=False)
class OriginEligibility:
    """Whether one requested origin can support every requested duration.

    Attributes
    ----------
    origin : hashable
        History-index label, as stored on the schedule.
    eligible : bool
        True only when every requested duration has a usable window here.
    reason : str
        Empty when eligible. Otherwise the windows that failed.
    """

    origin: Any
    eligible: bool
    reason: str


@dataclass(frozen=True, eq=False)
class DurationPreview:
    """Cost and eligibility before any sampling.

    ``status`` is ``ready`` only when at least one origin can support every
    requested duration and the fit and memory ceilings hold. A preview does
    not sample and does not select a duration.

    Attributes
    ----------
    status : str
        ``ready``, ``insufficient-history``, ``budget-exhausted``, or
        ``memory-exhausted``.
    reasons : tuple of str
        Why the preview is not ready, or the ready-count note.
    durations, target_lift, target_scale
        The comparison the caller asked for. The target includes the
        auto-added zero when that is the requested lift.
    eligible_origins, ineligible
        Common-origin split. An origin that fails any duration is ineligible
        and is not fitted.
    expected_counterfactual_fits, learned_null_upper_bound, expected_fits
        Counterfactual fits are eligible origins times durations. The
        learned-null term is an upper bound of one per duration when at
        least two origins are eligible, else zero. The budget uses the sum.
    estimated_window_bytes : int or None
        Four float64 arrays at the longest duration. None when draws are
        not on the template or the supplied ``sample_kwargs`` yet.
    windows : tuple of WindowPlan
        Eligible selection windows. Empty when none are eligible.
    """

    status: DurationStatus
    reasons: tuple[str, ...]
    durations: tuple[int, ...]
    target_lift: float
    target_scale: str
    eligible_origins: tuple[Any, ...]
    ineligible: tuple[OriginEligibility, ...]
    expected_counterfactual_fits: int
    learned_null_upper_bound: int
    expected_fits: int
    estimated_window_bytes: int | None
    max_window_bytes: int
    max_fits: int
    fit_cost: float | None
    expected_cost: float | None
    windows: tuple[Any, ...]


@dataclass(frozen=True, eq=False)
class DurationRow:
    """One duration after its own replay. Evidence is not upgraded.

    Attributes
    ----------
    evidence_level : str
        Replay evidence for this duration. Never ``model-conditional``.
    calibration_evidence_level : str or None
        Learned-null record, when one was built. Model-conditional is not a
        recommendation and is not copied into ``evidence_level``.
    recommendation : str
        Always ``not-supported``.
    historical_detection_count, historical_detection_share
        How often the frozen decision fired at the target lift. The share's
        denominator is the requested eligible windows, so a failure stays in
        the denominator and is not a detection.
    """

    duration: int
    target_lift: float
    target_scale: str
    evidence_level: str
    calibration_evidence_level: str | None
    persistent_bias: bool | None
    zero_lift_bias_cumulative: float | None
    zero_lift_bias_average: float | None
    zero_lift_mean_absolute_error: float | None
    historical_detection_count: int
    historical_detection_share: float | None
    scored_count: int
    failed_count: int
    requested_count: int
    not_run_count: int
    n_inference_fits: int
    cost: float | None
    recovery_mean_error: float | None
    truth_covers_share: float | None
    recommendation: str = _TABLE_RECOMMENDATION


@dataclass(frozen=True, eq=False)
class DurationWindow:
    """Scalar record for one origin and duration. Draws are not retained."""

    origin: Any
    duration: int
    status: str
    reason: str | None
    training_index: tuple[Any, ...]
    holdout_index: tuple[Any, ...]
    bias_cumulative: float | None
    bias_average: float | None
    estimand: str | None = None
    mean: float | None = None
    hdi_lower: float | None = None
    hdi_upper: float | None = None
    truth: float | None = None
    truth_covers: bool | None = None
    detected: bool | None = None
    tail_probability: float | None = None


@dataclass(frozen=True, eq=False)
class DurationComparison:
    """Historical comparison across durations. It does not select one.

    Attributes
    ----------
    status : str
        Worst per-duration replay status, or a refusal status when nothing
        was fitted. ``mixed`` means some durations scored and some failed.
    evidence_level : str or None
        Replay evidence. None when no replay ran. Never upgraded, and never
        taken from the calibration record.
    persistent_bias : bool or None
        True when any duration's zero-lift replay reports persistent bias.
        The bias is not corrected.
    fit_seconds, score_seconds
        Wall time inside the shared fit, and wall time translating that fit.
        Translation is not a second fit.
    sample_kwargs : dict
        The merged settings that were used or would have been used. A refusal
        does not reduce them.
    """

    status: DurationStatus
    evidence_level: str | None
    reasons: tuple[str, ...]
    preview: DurationPreview
    rows: tuple[DurationRow, ...]
    windows: tuple[DurationWindow, ...]
    persistent_bias: bool | None
    n_inference_fits: int
    cost: float | None
    fit_seconds: float
    score_seconds: float
    sample_kwargs: dict[str, Any]
    random_seed: int | None
    history_fingerprint: str

    def summary(self) -> pd.DataFrame:
        """Return one row per duration.

        Returns
        -------
        pandas.DataFrame
            Evidence, zero-lift bias, detection counts, and cost. The
            recommendation column is ``not-supported`` on every row. An
            empty frame keeps the columns when nothing was fitted.
        """
        columns = (
            "duration",
            "target_lift",
            "target_scale",
            "evidence_level",
            "calibration_evidence_level",
            "persistent_bias",
            "zero_lift_bias_cumulative",
            "zero_lift_bias_average",
            "zero_lift_mean_absolute_error",
            "historical_detection_count",
            "historical_detection_share",
            "scored_count",
            "failed_count",
            "requested_count",
            "not_run_count",
            "n_inference_fits",
            "cost",
            "recovery_mean_error",
            "truth_covers_share",
            "recommendation",
        )
        if not self.rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(
            [{name: getattr(row, name) for name in columns} for row in self.rows]
        )

    def detail(self) -> pd.DataFrame:
        """Return one row per origin, duration, and estimand.

        Returns
        -------
        pandas.DataFrame
            Translated target-lift summaries. A window that was not scored
            appears once, with an empty estimand, so a failure is visible.
        """
        columns = (
            "origin",
            "duration",
            "estimand",
            "mean",
            "hdi_lower",
            "hdi_upper",
            "truth",
            "truth_covers",
            "detected",
            "tail_probability",
            "status",
            "reason",
        )
        if not self.windows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(
            [
                {name: getattr(window, name) for name in columns}
                for window in self.windows
            ]
        )

    def recommendation(self) -> str:
        """Return the unsupported-recommendation text.

        Returns
        -------
        str
            States the status when the result is not a historical
            description, and states the evidence level. No duration was
            selected. Persistent bias, when present, is named and not
            corrected.
        """
        return _recommendation_text(
            self.status,
            self.evidence_level,
            self.persistent_bias,
            selected_duration=None,
        )

    def plot(
        self,
        *,
        axes: tuple[Any, Any] | None = None,
        figsize: tuple[float, float] = (9.0, 4.0),
    ) -> tuple[Any, tuple[Any, Any]]:
        """Plot zero-lift bias and historical detection counts.

        Parameters
        ----------
        axes : tuple of matplotlib Axes, optional
            ``(bias, detection)`` axes. When omitted, a new figure is created.
        figsize : tuple of float, default (9.0, 4.0)
            Figure size in inches. Ignored when ``axes`` is supplied.

        Returns
        -------
        tuple of Figure and tuple of Axes
            Left panel: zero-lift cumulative bias against duration. Right
            panel: historical detection counts at the target lift. The title
            states that this is not a power curve and not a recommendation.

        Raises
        ------
        ValueError
            If no window was scored. A refusal is not drawn as a curve.
        """
        import matplotlib.pyplot as plt

        scored = []
        seen: set[tuple[Any, int]] = set()
        for window in self.windows:
            if window.bias_cumulative is None:
                continue
            key = (window.origin, window.duration)
            if key in seen:
                continue
            seen.add(key)
            scored.append(window)
        if not scored:
            raise ValueError(
                "No scored windows to plot. A failed or refused comparison "
                "is not a detection curve and not a recommendation."
            )
        if axes is None:
            fig, axis_pair = plt.subplots(1, 2, figsize=figsize)
            bias_axis, detection_axis = axis_pair
            created = True
        else:
            if len(axes) != 2:
                raise ValueError("axes must be a pair of matplotlib Axes.")
            bias_axis, detection_axis = axes
            fig = bias_axis.figure
            created = False
        for window in scored:
            bias_axis.scatter(
                window.duration,
                window.bias_cumulative,
                color="C0",
                alpha=0.75,
                label="origin" if window is scored[0] else None,
            )
        means = [
            (row.duration, row.zero_lift_bias_cumulative)
            for row in self.rows
            if row.zero_lift_bias_cumulative is not None
        ]
        if means:
            bias_axis.scatter(
                [item[0] for item in means],
                [item[1] for item in means],
                marker="D",
                color="C1",
                zorder=3,
                label="mean across origins",
            )
        bias_axis.axhline(0.0, color="0.4", linewidth=1)
        bias_axis.set_xlabel("Duration (holdout rows)")
        bias_axis.set_ylabel("Zero-lift cumulative bias")
        durations = [row.duration for row in self.rows]
        bias_axis.set_xticks(durations)
        bias_axis.legend(frameon=False)
        detection_axis.bar(
            durations,
            [row.historical_detection_count for row in self.rows],
            color="C0",
            width=0.6,
        )
        detection_axis.set_xlabel("Duration (holdout rows)")
        detection_axis.set_ylabel("Historical detections at the target lift")
        detection_axis.set_xticks(durations)
        fig.suptitle(
            "Historical description, not a power curve and not a recommendation"
        )
        if created:
            fig.tight_layout()
        return fig, (bias_axis, detection_axis)

    def freeze(self, duration: int) -> AnalysisSpecification:
        """Record one evaluated duration without fitting the audit.

        Parameters
        ----------
        duration : int
            A duration that was in the evaluated grid. This is the caller's
            choice, not a library recommendation.

        Returns
        -------
        AnalysisSpecification
            Frozen units, cadence, estimator, priors, and this duration.
            The comparison's own result is unchanged.

        Raises
        ------
        ValueError
            If ``duration`` was not evaluated, or the comparison was refused
            before fitting.
        """
        if self.status in _REFUSAL_STATUSES:
            raise ValueError(
                f"Cannot freeze a duration from a {self.status} comparison. "
                "The duration was not evaluated, and freeze does not select "
                "a replacement."
            )
        if isinstance(duration, bool) or not isinstance(duration, int):
            raise TypeError(
                f"duration must be a Python int, got {type(duration).__name__}."
            )
        if duration not in self.preview.durations:
            raise ValueError(
                f"duration {duration} was not evaluated. Evaluated durations "
                f"are {list(self.preview.durations)}. Freeze does not add one."
            )
        return _freeze_specification(self, duration)


@dataclass(frozen=True, eq=False)
class AnalysisSpecification:
    """Frozen analysis for one caller-selected duration.

    ``execute`` rebuilds the planned :class:`~causalpy.experiments.synthetic_control.SyntheticControl`
    from the stored template. ``audit`` fits only the reserved audit origins
    at this duration. Neither method selects a different duration.

    The history fingerprint records the frame used at freeze. It is provenance,
    not a lock: ``execute`` does not refuse a different frame.
    """

    selected_duration: int
    status_at_freeze: str
    evidence_level_at_freeze: str | None
    persistent_bias_at_freeze: bool | None
    treated_unit: str
    control_units: tuple[str, ...]
    estimator: str
    decision_direction: Literal["positive", "negative"]
    decision_estimand: Literal["cumulative", "average"]
    decision_threshold: float
    decision_probability: float
    relative_floor: float | None
    time_column: str | None
    min_donor_correlation: float
    auto_scale_sigma: bool
    selection_origins: tuple[Any, ...]
    audit_origins: tuple[Any, ...]
    exclusions: tuple[Any, ...]
    cadence: Any
    min_training_periods: int
    durations: tuple[int, ...]
    target_lift: float
    target_scale: str
    fit_cost: float | None
    sample_kwargs: dict[str, Any]
    priors: dict[str, Any]
    random_seed: int | None
    history_fingerprint: str
    causalpy_version: str
    template: WeightedSumFitter
    recommendation_text: str

    def recommendation(self) -> str:
        """Return the unsupported-recommendation text for the frozen choice.

        Returns
        -------
        str
            Names the recorded duration and states that recording it is not
            a launch recommendation.
        """
        return self.recommendation_text

    def audit(self, history: pd.DataFrame) -> DurationAudit:
        """Fit the reserved audit origins at the frozen duration only.

        Parameters
        ----------
        history : pandas.DataFrame
            Frame the audit schedule is validated against. A raw frame that
            still has ``time_column`` is indexed first. The planner's already
            indexed history, whose index name is that column, is accepted.
            Selection origins are not refit. The selected duration is not an
            argument and is not changed.

        Returns
        -------
        DurationAudit
            ``historical-descriptive`` when every audit window scored and
            the replay is not persistently biased, ``limitation`` when the
            audit scored but does not support a stronger claim, and
            ``inconclusive`` when it could not be scored. The recommendation
            is unsupported in every case.
        """
        return _audit_specification(self, history)

    def execute(self, history: pd.DataFrame, treatment_time: Any) -> SyntheticControl:
        """Fit the frozen specification as the planned analysis.

        Parameters
        ----------
        history : pandas.DataFrame
            Analysis frame. It is not required to be the comparison frame.
            A raw frame that still has ``time_column`` is indexed first. The
            planner's already indexed history, whose index name is that
            column, is accepted. Training is every non-excluded row strictly
            before ``treatment_time``. The post-period is the frozen holdout of
            ``selected_duration`` labels under the frozen cadence. Later
            rows, cadence gaps, and exclusions are not fit. An exclusion
            equal to ``treatment_time`` is refused rather than shifting the
            split. If the frame cannot supply that holdout, ``execute``
            raises rather than fitting a shorter or different window.
        treatment_time : index label
            Post-period start. The frozen duration is the planned holdout
            length, not every row at and after this label.

        Returns
        -------
        SyntheticControl
            Fitted experiment built from the stored units, template, priors,
            and sample settings. Draws are not reduced.
        """
        return _execute_specification(self, history, treatment_time)


@dataclass(frozen=True, eq=False)
class DurationAudit:
    """Untouched-origin replay at the already frozen duration.

    Attributes
    ----------
    status : {'historical-descriptive', 'inconclusive', 'limitation'}
        Workflow result. A clean historical description still has no
        supported recommendation.
    evidence_level : str or None
        Replay evidence. Not upgraded, and not a launch claim.
    selected_duration : int
        The duration that was already frozen. The audit does not replace it.
    """

    status: AuditStatus
    evidence_level: str | None
    selected_duration: int
    persistent_bias: bool | None
    n_inference_fits: int
    scored_count: int
    failed_count: int
    requested_count: int
    reasons: tuple[str, ...]
    recommendation_text: str

    def recommendation(self) -> str:
        """Return the unsupported-recommendation text.

        Returns
        -------
        str
            States the audit status and that the frozen duration was kept.
        """
        return self.recommendation_text


@dataclass(frozen=True, eq=False)
class DurationProfile:
    """One cold run and one warm run of the same comparison.

    ``peak_rss_bytes`` is the process high-water mark from
    ``resource.getrusage``, not a per-call delta. On macOS that value is
    already bytes; elsewhere it is converted from kilobytes. PyTensor's
    compiled cache is included in the process peak and is not subtracted.

    Attributes
    ----------
    cold_seconds, warm_seconds
        End-to-end wall time of each ``compare_durations`` call.
    cold_fit_seconds, warm_fit_seconds, cold_score_seconds, warm_score_seconds
        Fit versus translation split recorded inside each comparison.
    """

    cold_seconds: float
    warm_seconds: float
    cold_fit_seconds: float
    warm_fit_seconds: float
    cold_score_seconds: float
    warm_score_seconds: float
    peak_rss_bytes: int
    platform: str
    python_version: str
    numpy_version: str
    pymc_version: str
    pytensor_version: str
    causalpy_version: str
    sample_kwargs: dict[str, Any]
    n_inference_fits: int
    status: str
    note: str


def _recommendation_text(
    status: str,
    evidence_level: str | None,
    persistent_bias: bool | None,
    *,
    selected_duration: int | None,
) -> str:
    parts = [_NO_RECOMMENDATION]
    if selected_duration is None:
        parts.append("No duration was selected.")
    else:
        parts.append(
            f"The recorded duration is {selected_duration}. "
            "Recording a duration is the caller's choice, not a recommendation."
        )
    if status == "historical-descriptive" and evidence_level is not None:
        parts.append(
            f"Evidence level: {evidence_level}. "
            "This is a historical description, not a power curve and not a launch claim."
        )
    elif evidence_level is None:
        parts.append(f"Status: {status}. No replay evidence is available.")
    else:
        parts.append(
            f"Status: {status}. Evidence level: {evidence_level}. "
            "This is not a historical description, not a power curve, and not a launch claim."
        )
    if persistent_bias:
        parts.append(
            "Persistent null bias is visible and is not corrected. "
            "The qualification engine has not landed."
        )
    parts.append("Extrapolation beyond the evaluated durations is not a result.")
    return " ".join(parts)


def _require_sequence(name: str, values: Sequence[Any]) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence, not a single string.")
    return tuple(values)


def _require_durations(durations: Sequence[int]) -> tuple[int, ...]:
    values = _require_sequence("durations", durations)
    if not values:
        raise ValueError("durations must name at least one duration.")
    checked: list[int] = []
    for duration in values:
        if isinstance(duration, bool) or not isinstance(duration, int):
            raise TypeError(
                f"duration must be a Python int, got {type(duration).__name__}."
            )
        if duration < 1:
            raise ValueError(f"duration must be at least 1, got {duration}.")
        checked.append(duration)
    if len(checked) != len(set(checked)):
        raise ValueError(
            "durations must be unique. Each horizon is a separate fit; "
            "prefix reuse is not enabled."
        )
    return tuple(checked)


def _require_max_fits(max_fits: int) -> int:
    if isinstance(max_fits, bool) or not isinstance(max_fits, int):
        raise TypeError(
            f"max_fits must be a Python int, got {type(max_fits).__name__}."
        )
    if max_fits < 0:
        raise ValueError(f"max_fits must be at least 0, got {max_fits}.")
    return max_fits


def _require_fit_cost(fit_cost: float | None) -> float | None:
    if fit_cost is None:
        return None
    if isinstance(fit_cost, bool) or not isinstance(fit_cost, (int, float)):
        raise TypeError(
            f"fit_cost must be a real number or None, got {type(fit_cost).__name__}."
        )
    cost = float(fit_cost)
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError(f"fit_cost must be finite and at least 0, got {fit_cost}.")
    return cost


def _require_min_training(min_training_periods: int) -> int:
    if isinstance(min_training_periods, bool) or not isinstance(
        min_training_periods, int
    ):
        raise TypeError(
            "min_training_periods must be a Python int, got "
            f"{type(min_training_periods).__name__}."
        )
    if min_training_periods < 1:
        raise ValueError(
            f"min_training_periods must be at least 1, got {min_training_periods}."
        )
    return min_training_periods


def _resolve_target(
    relative_lifts: Sequence[float] | None,
    absolute_lifts: Sequence[float] | None,
    target_lift: float,
    target_scale: str | None,
) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None, float, str]:
    from causalpy.design import _ordered_lifts, _require_real

    relative = _ordered_lifts("relative_lifts", relative_lifts)
    absolute = _ordered_lifts("absolute_lifts", absolute_lifts)
    if relative is None and absolute is None:
        raise ValueError(
            "Pass relative_lifts or absolute_lifts. At least one scale is required."
        )
    lift = _require_real("target_lift", target_lift)

    def _contains(grid: tuple[float, ...] | None) -> bool:
        return grid is not None and any(np.isclose(lift, item) for item in grid)

    in_relative = _contains(relative)
    in_absolute = _contains(absolute)
    if target_scale is None:
        if in_relative and in_absolute:
            raise ValueError(
                "target_lift is in both grids. Pass target_scale so the "
                "detection column has one denominator."
            )
        if in_relative:
            scale = "relative"
        elif in_absolute:
            scale = "absolute"
        else:
            raise ValueError(
                "target_lift is not in the requested lift grid. Zero is added "
                "automatically as the paired null; other lifts must be passed."
            )
    elif target_scale not in _SCALES:
        raise ValueError(
            f"target_scale must be 'absolute' or 'relative', got {target_scale!r}."
        )
    else:
        scale = target_scale
        grid = relative if scale == "relative" else absolute
        if grid is None or not _contains(grid):
            raise ValueError(
                f"target_lift {lift:g} is not in the {scale} lift grid. "
                "Zero is added automatically as the paired null."
            )
    return relative, absolute, float(lift), scale


def _window_bytes(chains: int, draws: int, duration: int) -> int:
    return _IMPACT_ARRAYS * chains * draws * duration * 8


def _merged_sample_kwargs(
    planner: SyntheticControlPlanner, sample_kwargs: dict[str, Any] | None
) -> dict[str, Any]:
    if sample_kwargs is not None and not isinstance(sample_kwargs, dict):
        raise TypeError(
            f"sample_kwargs must be a dict or None, got {type(sample_kwargs).__name__}."
        )
    merged = dict(planner.model.sample_kwargs)
    if planner.random_seed is not None:
        merged.setdefault("random_seed", planner.random_seed)
    if sample_kwargs:
        merged.update(sample_kwargs)
    merged.setdefault("progressbar", False)
    if "draws" in merged and "chains" not in merged:
        merged["chains"] = 4
    return merged


def _chains_and_draws(sample_kwargs: dict[str, Any]) -> tuple[int, int] | None:
    draws = sample_kwargs.get("draws")
    if draws is None:
        return None
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError(
            f"sample_kwargs['draws'] must be a positive int, got {draws!r}."
        )
    chains = sample_kwargs.get("chains", 4)
    if isinstance(chains, bool) or not isinstance(chains, int) or chains < 1:
        raise ValueError(
            f"sample_kwargs['chains'] must be a positive int, got {chains!r}."
        )
    return chains, draws


def _fingerprint(frame: pd.DataFrame, units: Sequence[str]) -> str:
    ordered = frame.loc[:, list(units)]
    hashed = pd.util.hash_pandas_object(ordered, index=True)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def _eligibility(
    schedule: HistoricalSchedule, durations: tuple[int, ...]
) -> tuple[tuple[Any, ...], tuple[OriginEligibility, ...]]:
    selection = [window for window in schedule.windows if window.role == "selection"]
    origins = list(dict.fromkeys(window.origin for window in selection))
    eligible: list[Any] = []
    reports: list[OriginEligibility] = []
    for origin in origins:
        origin_windows = [window for window in selection if window.origin == origin]
        failed = [window for window in origin_windows if window.status != "usable"]
        if failed or len(origin_windows) != len(durations):
            reason = "; ".join(
                f"duration {window.horizon}: {', '.join(window.reasons) or window.status}"
                for window in failed
            )
            if not reason:
                reason = "a requested duration has no window at this origin."
            reports.append(OriginEligibility(origin, False, reason))
            continue
        eligible.append(origin)
        reports.append(OriginEligibility(origin, True, ""))
    return tuple(eligible), tuple(reports)


def _learned_null_upper_bound(n_origins: int, n_durations: int) -> int:
    if n_origins < 2:
        return 0
    return n_durations


@dataclass(frozen=True, eq=False)
class _Prepared:
    preview: DurationPreview
    schedule: HistoricalSchedule
    relative_lifts: tuple[float, ...] | None
    absolute_lifts: tuple[float, ...] | None
    merged_kwargs: dict[str, Any]
    fingerprint: str
    selection_origins: tuple[Any, ...]
    audit_origins: tuple[Any, ...]
    exclusions: tuple[Any, ...]
    cadence: Any
    min_training_periods: int


def _prepare(
    planner: SyntheticControlPlanner,
    durations: Sequence[int],
    *,
    selection_origins: Sequence[Any],
    audit_origins: Sequence[Any],
    relative_lifts: Sequence[float] | None,
    absolute_lifts: Sequence[float] | None,
    target_lift: float,
    target_scale: str | None,
    exclusions: Sequence[Any],
    cadence: Any,
    max_fits: int,
    fit_cost: float | None,
    min_training_periods: int,
    sample_kwargs: dict[str, Any] | None,
) -> _Prepared:
    checked_durations = _require_durations(durations)
    origins = _require_sequence("selection_origins", selection_origins)
    audits = _require_sequence("audit_origins", audit_origins)
    if not origins:
        raise ValueError("selection_origins must name at least one origin.")
    checked_exclusions = _require_sequence("exclusions", exclusions)
    checked_max_fits = _require_max_fits(max_fits)
    checked_cost = _require_fit_cost(fit_cost)
    checked_training = _require_min_training(min_training_periods)
    relative, absolute, lift, scale = _resolve_target(
        relative_lifts, absolute_lifts, target_lift, target_scale
    )
    if not isinstance(planner.model, WeightedSumFitter):
        raise TypeError(
            "Duration comparison uses the shipped Bayesian simplex evaluator. "
            f"Got {type(planner.model).__name__}."
        )
    merged = _merged_sample_kwargs(planner, sample_kwargs)
    request = historical_validation.ScheduleRequest(
        selection_origins=origins,
        horizons=checked_durations,
        audit_origins=audits,
        exclusions=checked_exclusions,
        cadence=cadence,
        min_training_periods=checked_training,
    )
    schedule = historical_validation.validate_schedule(
        planner.history,
        planner.treated_unit,
        planner.control_units,
        request,
    )
    eligible, reports = _eligibility(schedule, checked_durations)
    eligible_windows = tuple(
        window
        for window in schedule.windows
        if window.role == "selection"
        and window.origin in eligible
        and window.status == "usable"
    )
    counterfactual = len(eligible) * len(checked_durations)
    null_bound = _learned_null_upper_bound(len(eligible), len(checked_durations))
    expected = counterfactual + null_bound
    resolved = _chains_and_draws(merged)
    estimated = None
    if resolved is not None:
        chains, draws = resolved
        estimated = _window_bytes(chains, draws, max(checked_durations))
    reasons: list[str] = []
    status: DurationStatus = "ready"
    if not eligible:
        status = "insufficient-history"
        reasons.append(
            "No origin has a usable window for every requested duration. "
            "Origins that fail any duration are ineligible and are not fitted."
        )
    elif expected > checked_max_fits:
        status = "budget-exhausted"
        reasons.append(
            f"Expected fits are {expected} "
            f"({counterfactual} counterfactual + {null_bound} learned-null "
            f"upper bound), which exceeds max_fits={checked_max_fits}. "
            "No fit was started and sampler settings were not reduced."
        )
    elif estimated is not None and estimated > MAX_WINDOW_BYTES:
        status = "memory-exhausted"
        reasons.append(
            f"One window at duration {max(checked_durations)} is estimated at "
            f"{estimated} bytes, above the {MAX_WINDOW_BYTES}-byte cap. "
            "No fit was started."
        )
    else:
        reasons.append(
            f"{len(eligible)} origin(s) support every requested duration. "
            f"Expected counterfactual fits: {counterfactual}."
        )
        if estimated is None:
            reasons.append(
                "Draws are not on the template yet. compare_durations will "
                "refuse a heavy default rather than guess a memory ceiling."
            )
    ineligible = tuple(report for report in reports if not report.eligible)
    preview = DurationPreview(
        status=status,
        reasons=tuple(reasons),
        durations=checked_durations,
        target_lift=lift,
        target_scale=scale,
        eligible_origins=eligible,
        ineligible=ineligible,
        expected_counterfactual_fits=counterfactual,
        learned_null_upper_bound=null_bound,
        expected_fits=expected,
        estimated_window_bytes=estimated,
        max_window_bytes=MAX_WINDOW_BYTES,
        max_fits=checked_max_fits,
        fit_cost=checked_cost,
        expected_cost=None if checked_cost is None else expected * checked_cost,
        windows=eligible_windows,
    )
    return _Prepared(
        preview=preview,
        schedule=schedule,
        relative_lifts=relative,
        absolute_lifts=absolute,
        merged_kwargs=merged,
        fingerprint=_fingerprint(
            planner.history, (planner.treated_unit, *planner.control_units)
        ),
        selection_origins=origins,
        audit_origins=audits,
        exclusions=checked_exclusions,
        cadence=cadence,
        min_training_periods=checked_training,
    )


def _empty_comparison(
    prepared: _Prepared, random_seed: int | None
) -> DurationComparison:
    return DurationComparison(
        status=prepared.preview.status,
        evidence_level=None,
        reasons=prepared.preview.reasons,
        preview=prepared.preview,
        rows=(),
        windows=(),
        persistent_bias=None,
        n_inference_fits=0,
        cost=0.0 if prepared.preview.fit_cost is not None else None,
        fit_seconds=0.0,
        score_seconds=0.0,
        sample_kwargs=dict(prepared.merged_kwargs),
        random_seed=random_seed,
        history_fingerprint=prepared.fingerprint,
    )


def _check_relative_baselines(
    planner: SyntheticControlPlanner,
    windows: Sequence[WindowPlan],
    *,
    needed: bool,
) -> None:
    if not needed:
        return
    from causalpy.design import _require_relative_baseline

    observed = planner.history[planner.treated_unit].to_numpy(dtype=float)
    index = planner.history.index
    for window in windows:
        positions = index.get_indexer(window.holdout_index)
        if np.any(positions < 0):
            raise ValueError(
                f"origin {window.origin!r} holdout is not in the history index."
            )
        training_positions = index.get_indexer(window.training_index)
        _require_relative_baseline(
            observed[positions],
            observed[training_positions],
            declared_floor=planner.relative_floor,
            unit=planner.treated_unit,
            holdout_index=window.holdout_index,
        )


def _translate_window(
    experiment: SyntheticControl,
    planner: SyntheticControlPlanner,
    prepared: _Prepared,
    window: WindowPlan,
) -> list[DurationWindow]:
    import xarray as xr

    from causalpy.design import _score_translated_lifts, _select_unit

    baseline = _select_unit(experiment.result.impact_post, planner.treated_unit)
    counterfactual = _select_unit(
        experiment.result.predictions_post, planner.treated_unit
    )
    observed = xr.DataArray(
        experiment.datapost[planner.treated_unit].to_numpy(dtype=float),
        dims="obs_ind",
        coords={"obs_ind": baseline.coords["obs_ind"]},
    )
    scores = _score_translated_lifts(
        baseline_impact=baseline,
        observed=observed,
        counterfactual=counterfactual,
        relative_lifts=prepared.relative_lifts,
        absolute_lifts=prepared.absolute_lifts,
        decision=planner.decision,
    )
    target = prepared.preview.target_lift
    scale = prepared.preview.target_scale
    kept: list[DurationWindow] = []
    for score in scores:
        if score.effect_scale != scale or not np.isclose(score.lift, target):
            continue
        tail = score.tail_probability
        kept.append(
            DurationWindow(
                origin=window.origin,
                duration=window.horizon,
                status="scored",
                reason=None,
                training_index=tuple(window.training_index),
                holdout_index=tuple(window.holdout_index),
                bias_cumulative=None,
                bias_average=None,
                estimand=score.estimand,
                mean=float(score.mean),
                hdi_lower=float(score.hdi_lower),
                hdi_upper=float(score.hdi_upper),
                truth=float(score.truth),
                truth_covers=bool(score.truth_covers),
                detected=score.detected,
                tail_probability=None if tail is None else float(tail),
            )
        )
    if not kept:
        raise RuntimeError(
            "The shared translator did not return the target lift. "
            "That is a duration-workflow bug, not a non-detection."
        )
    return kept


def _attach_bias(
    translated: Sequence[DurationWindow], replay: WindowReplay
) -> tuple[DurationWindow, ...]:
    if not translated:
        return (
            DurationWindow(
                origin=replay.plan.origin,
                duration=replay.plan.horizon,
                status=replay.status,
                reason=replay.reason,
                training_index=tuple(replay.plan.training_index),
                holdout_index=tuple(replay.plan.holdout_index),
                bias_cumulative=replay.bias_cumulative,
                bias_average=replay.bias_average,
            ),
        )
    return tuple(
        DurationWindow(
            origin=window.origin,
            duration=window.duration,
            status=window.status,
            reason=window.reason,
            training_index=window.training_index,
            holdout_index=window.holdout_index,
            bias_cumulative=replay.bias_cumulative,
            bias_average=replay.bias_average,
            estimand=window.estimand,
            mean=window.mean,
            hdi_lower=window.hdi_lower,
            hdi_upper=window.hdi_upper,
            truth=window.truth,
            truth_covers=window.truth_covers,
            detected=window.detected,
            tail_probability=window.tail_probability,
        )
        for window in translated
    )


def _row_from_windows(
    duration: int,
    prepared: _Prepared,
    replays: Sequence[WindowReplay],
    translated: Sequence[DurationWindow],
    *,
    evidence_level: str,
    persistent_bias: bool | None,
    bias_cumulative: float | None,
    bias_average: float | None,
    calibration_level: str | None,
    n_inference_fits: int,
) -> DurationRow:
    scored = [window for window in replays if window.status == "scored"]
    failed = [window for window in replays if window.status == "fit-failed"]
    not_run = [window for window in replays if window.status == "not-run"]
    decision_rows = [
        window
        for window in translated
        if window.estimand is not None and window.status == "scored"
    ]
    detections = sum(1 for window in decision_rows if window.detected is True)
    requested = len(replays)
    share = None if requested == 0 else detections / requested
    errors = [
        window.mean - window.truth
        for window in decision_rows
        if window.mean is not None and window.truth is not None
    ]
    covers = [
        window.truth_covers
        for window in decision_rows
        if window.truth_covers is not None
    ]
    absolute_errors = [
        abs(float(window.bias_cumulative))
        for window in scored
        if window.bias_cumulative is not None
    ]
    cost = prepared.preview.fit_cost
    return DurationRow(
        duration=duration,
        target_lift=prepared.preview.target_lift,
        target_scale=prepared.preview.target_scale,
        evidence_level=evidence_level,
        calibration_evidence_level=calibration_level,
        persistent_bias=persistent_bias,
        zero_lift_bias_cumulative=bias_cumulative,
        zero_lift_bias_average=bias_average,
        zero_lift_mean_absolute_error=(
            float(np.mean(absolute_errors)) if absolute_errors else None
        ),
        historical_detection_count=detections,
        historical_detection_share=share,
        scored_count=len(scored),
        failed_count=len(failed),
        requested_count=requested,
        not_run_count=len(not_run),
        n_inference_fits=n_inference_fits,
        cost=None if cost is None else n_inference_fits * cost,
        recovery_mean_error=float(np.mean(errors)) if errors else None,
        truth_covers_share=(float(np.mean(covers)) if covers else None),
    )


def _decision_rows(
    translated: Sequence[DurationWindow], estimand: str
) -> tuple[DurationWindow, ...]:
    return tuple(window for window in translated if window.estimand == estimand)


def _evaluate(
    planner: SyntheticControlPlanner, prepared: _Prepared
) -> DurationComparison:
    started = time.perf_counter()
    template = historical_validation._replay_template(
        planner.model, prepared.merged_kwargs, planner.random_seed
    )
    template_ready = time.perf_counter()
    rows: list[DurationRow] = []
    windows: list[DurationWindow] = []
    fit_seconds = template_ready - started
    score_seconds = 0.0
    total_fits = 0
    evidence_levels: list[str] = []
    persistent_flags: list[bool | None] = []
    for duration in prepared.preview.durations:
        plans = [
            window for window in prepared.preview.windows if window.horizon == duration
        ]
        replays: list[WindowReplay] = []
        translated: list[DurationWindow] = []
        duration_fits = 0
        for plan in plans:
            captured: list[list[DurationWindow]] = []
            score_mark: list[float] = []

            def _on_scored(
                experiment: SyntheticControl,
                window: WindowPlan,
                _captured: list[list[DurationWindow]] = captured,
                _mark: list[float] = score_mark,
            ) -> None:
                _mark.append(time.perf_counter())
                _captured.append(
                    _translate_window(experiment, planner, prepared, window)
                )

            fit_started = time.perf_counter()
            replay, attempts = historical_validation._replay_window(
                planner.history,
                planner.treated_unit,
                planner.control_units,
                plan,
                planner.decision,
                template=template,
                donor_correlation=planner.min_donor_correlation,
                auto_scale_sigma=planner.auto_scale_sigma,
                fit_audit=False,
                on_scored=_on_scored,
            )
            fit_finished = time.perf_counter()
            duration_fits += attempts
            if score_mark:
                fit_seconds += score_mark[0] - fit_started
                score_seconds += fit_finished - score_mark[0]
            else:
                fit_seconds += fit_finished - fit_started
            replays.append(replay)
            batch = captured[0] if captured and replay.status == "scored" else []
            translated.extend(_attach_bias(batch, replay))
        evidence = historical_validation._replay_evidence(replays)
        null_started = time.perf_counter()
        calibration, null_fits = historical_validation._calibration_from_windows(
            replays,
            rope_half_width=0.0,
            threshold=planner.decision.threshold,
            fit_learned_null=True,
            sample_kwargs=dict(template.sample_kwargs),
            random_seed=planner.random_seed,
        )
        fit_seconds += time.perf_counter() - null_started
        duration_fits += null_fits
        total_fits += duration_fits
        evidence_levels.append(evidence[0])
        persistent_flags.append(evidence[5])
        decision_translated = _decision_rows(translated, planner.decision.estimand)
        # Keep both estimands on the comparison, but the row's detection and
        # recovery numbers follow the declared estimand only.
        other = [
            window
            for window in translated
            if window.estimand != planner.decision.estimand
        ]
        windows.extend(decision_translated)
        windows.extend(other)
        rows.append(
            _row_from_windows(
                duration,
                prepared,
                replays,
                decision_translated,
                evidence_level=evidence[0],
                persistent_bias=evidence[5],
                bias_cumulative=evidence[2],
                bias_average=evidence[3],
                calibration_level=calibration.evidence_level,
                n_inference_fits=duration_fits,
            )
        )
    status, evidence_level = _aggregate_status(evidence_levels)
    persistent = (
        True
        if any(flag is True for flag in persistent_flags)
        else (False if any(flag is False for flag in persistent_flags) else None)
    )
    cost = prepared.preview.fit_cost
    return DurationComparison(
        status=status,
        evidence_level=evidence_level,
        reasons=prepared.preview.reasons,
        preview=prepared.preview,
        rows=tuple(rows),
        windows=tuple(windows),
        persistent_bias=persistent,
        n_inference_fits=total_fits,
        cost=None if cost is None else total_fits * cost,
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
        sample_kwargs=dict(template.sample_kwargs),
        random_seed=planner.random_seed,
        history_fingerprint=prepared.fingerprint,
    )


def _aggregate_status(levels: Sequence[str]) -> tuple[DurationStatus, str]:
    unique = set(levels)
    if unique == {"historical-descriptive"}:
        return "historical-descriptive", "historical-descriptive"
    if unique == {"fit-failed"}:
        return "fit-failed", "fit-failed"
    if unique == {"insufficient-history"}:
        return "insufficient-history", "insufficient-history"
    if "historical-descriptive" in unique:
        return "mixed", "historical-descriptive"
    if "fit-failed" in unique:
        return "fit-failed", "fit-failed"
    return "insufficient-history", "insufficient-history"


def _preview_durations(
    planner: SyntheticControlPlanner,
    durations: Sequence[int],
    *,
    selection_origins: Sequence[Any],
    audit_origins: Sequence[Any] = (),
    relative_lifts: Sequence[float] | None = None,
    absolute_lifts: Sequence[float] | None = None,
    target_lift: float,
    target_scale: str | None = None,
    exclusions: Sequence[Any] = (),
    cadence: Any = None,
    max_fits: int = DEFAULT_MAX_FITS,
    fit_cost: float | None = None,
    min_training_periods: int = 2,
    sample_kwargs: dict[str, Any] | None = None,
) -> DurationPreview:
    prepared = _prepare(
        planner,
        durations,
        selection_origins=selection_origins,
        audit_origins=audit_origins,
        relative_lifts=relative_lifts,
        absolute_lifts=absolute_lifts,
        target_lift=target_lift,
        target_scale=target_scale,
        exclusions=exclusions,
        cadence=cadence,
        max_fits=max_fits,
        fit_cost=fit_cost,
        min_training_periods=min_training_periods,
        sample_kwargs=sample_kwargs,
    )
    return prepared.preview


def _compare_durations(
    planner: SyntheticControlPlanner,
    durations: Sequence[int],
    *,
    selection_origins: Sequence[Any],
    audit_origins: Sequence[Any] = (),
    relative_lifts: Sequence[float] | None = None,
    absolute_lifts: Sequence[float] | None = None,
    target_lift: float,
    target_scale: str | None = None,
    exclusions: Sequence[Any] = (),
    cadence: Any = None,
    max_fits: int = DEFAULT_MAX_FITS,
    fit_cost: float | None = None,
    min_training_periods: int = 2,
    sample_kwargs: dict[str, Any] | None = None,
) -> DurationComparison:
    prepared = _prepare(
        planner,
        durations,
        selection_origins=selection_origins,
        audit_origins=audit_origins,
        relative_lifts=relative_lifts,
        absolute_lifts=absolute_lifts,
        target_lift=target_lift,
        target_scale=target_scale,
        exclusions=exclusions,
        cadence=cadence,
        max_fits=max_fits,
        fit_cost=fit_cost,
        min_training_periods=min_training_periods,
        sample_kwargs=sample_kwargs,
    )
    if prepared.preview.status != "ready":
        return _remember(
            _empty_comparison(prepared, planner.random_seed), planner, prepared
        )
    _check_relative_baselines(
        planner,
        prepared.preview.windows,
        needed=prepared.relative_lifts is not None,
    )
    resolved = _chains_and_draws(prepared.merged_kwargs)
    if resolved is None:
        raise ValueError(
            "sample_kwargs must include 'draws' before a duration comparison "
            "fits. The default sampler is not a safe memory bound."
        )
    return _remember(_evaluate(planner, prepared), planner, prepared)


def _rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _profile_durations(
    planner: SyntheticControlPlanner,
    durations: Sequence[int],
    *,
    selection_origins: Sequence[Any],
    audit_origins: Sequence[Any] = (),
    relative_lifts: Sequence[float] | None = None,
    absolute_lifts: Sequence[float] | None = None,
    target_lift: float,
    target_scale: str | None = None,
    exclusions: Sequence[Any] = (),
    cadence: Any = None,
    max_fits: int = DEFAULT_MAX_FITS,
    fit_cost: float | None = None,
    min_training_periods: int = 2,
    sample_kwargs: dict[str, Any] | None = None,
) -> DurationProfile:
    import pymc as pm
    import pytensor

    def _run() -> tuple[float, DurationComparison]:
        started = time.perf_counter()
        comparison = _compare_durations(
            planner,
            durations,
            selection_origins=selection_origins,
            audit_origins=audit_origins,
            relative_lifts=relative_lifts,
            absolute_lifts=absolute_lifts,
            target_lift=target_lift,
            target_scale=target_scale,
            exclusions=exclusions,
            cadence=cadence,
            max_fits=max_fits,
            fit_cost=fit_cost,
            min_training_periods=min_training_periods,
            sample_kwargs=sample_kwargs,
        )
        return time.perf_counter() - started, comparison

    cold_seconds, cold = _run()
    warm_seconds, warm = _run()
    return DurationProfile(
        cold_seconds=cold_seconds,
        warm_seconds=warm_seconds,
        cold_fit_seconds=cold.fit_seconds,
        warm_fit_seconds=warm.fit_seconds,
        cold_score_seconds=cold.score_seconds,
        warm_score_seconds=warm.score_seconds,
        peak_rss_bytes=_rss_bytes(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        pymc_version=pm.__version__,
        pytensor_version=pytensor.__version__,
        causalpy_version=__version__,
        sample_kwargs=dict(warm.sample_kwargs),
        n_inference_fits=warm.n_inference_fits,
        status=warm.status,
        note=(
            "peak_rss_bytes is the process high-water mark, including the "
            "PyTensor cache. It is not a per-call allocation and not a "
            "recommendation to launch."
        ),
    )


def _freeze_specification(
    comparison: DurationComparison, duration: int
) -> AnalysisSpecification:
    planner = getattr(comparison, "_duration_planner", None)
    prepared = getattr(comparison, "_duration_prepared", None)
    if planner is None or prepared is None:
        raise RuntimeError(
            "freeze lost the planner context. Call it on the comparison "
            "returned by SyntheticControlPlanner.compare_durations."
        )
    template = planner.model._clone()
    template.sample_kwargs = dict(comparison.sample_kwargs)
    text = _recommendation_text(
        comparison.status,
        comparison.evidence_level,
        comparison.persistent_bias,
        selected_duration=duration,
    )
    return AnalysisSpecification(
        selected_duration=duration,
        status_at_freeze=comparison.status,
        evidence_level_at_freeze=comparison.evidence_level,
        persistent_bias_at_freeze=comparison.persistent_bias,
        treated_unit=planner.treated_unit,
        control_units=planner.control_units,
        estimator=planner.estimator,
        decision_direction=planner.decision.direction,
        decision_estimand=planner.decision.estimand,
        decision_threshold=planner.decision.threshold,
        decision_probability=planner.decision.probability,
        relative_floor=planner.relative_floor,
        time_column=planner.time_column,
        min_donor_correlation=planner.min_donor_correlation,
        auto_scale_sigma=planner.auto_scale_sigma,
        selection_origins=prepared.selection_origins,
        audit_origins=prepared.audit_origins,
        exclusions=prepared.exclusions,
        cadence=prepared.cadence,
        min_training_periods=prepared.min_training_periods,
        durations=prepared.preview.durations,
        target_lift=prepared.preview.target_lift,
        target_scale=prepared.preview.target_scale,
        fit_cost=prepared.preview.fit_cost,
        sample_kwargs=dict(comparison.sample_kwargs),
        priors=dict(planner.model._user_priors or {}),
        random_seed=planner.random_seed,
        history_fingerprint=comparison.history_fingerprint,
        causalpy_version=__version__,
        template=template,
        recommendation_text=text,
    )


def _remember(
    comparison: DurationComparison,
    planner: SyntheticControlPlanner,
    prepared: _Prepared,
) -> DurationComparison:
    object.__setattr__(comparison, "_duration_planner", planner)
    object.__setattr__(comparison, "_duration_prepared", prepared)
    return comparison


def _analysis_frame(history: pd.DataFrame, time_column: str | None) -> pd.DataFrame:
    """Return history indexed by the frozen time column.

    The planner moves ``time_column`` onto the index at construction and does
    not keep it as a column. Audit and execute accept that panel. A raw frame
    that still has the column is indexed by the existing converter. A missing
    column whose index name does not match remains an error.
    """
    from causalpy.input_data import to_pandas_with_time_index

    frame = history
    if (
        time_column is not None
        and time_column not in history.columns
        and history.index.name == time_column
    ):
        frame = history.reset_index()
    return to_pandas_with_time_index(frame, time_column, argument_name="history")


def _audit_specification(
    spec: AnalysisSpecification, history: pd.DataFrame
) -> DurationAudit:
    from causalpy.design import DecisionRule

    if not spec.audit_origins:
        return DurationAudit(
            status="inconclusive",
            evidence_level=None,
            selected_duration=spec.selected_duration,
            persistent_bias=None,
            n_inference_fits=0,
            scored_count=0,
            failed_count=0,
            requested_count=0,
            reasons=(
                "No audit origins were reserved. The audit was not scored "
                "and the frozen duration was not changed.",
            ),
            recommendation_text=_recommendation_text(
                "inconclusive",
                None,
                None,
                selected_duration=spec.selected_duration,
            ),
        )
    frame = _analysis_frame(history, spec.time_column)
    request = historical_validation.ScheduleRequest(
        selection_origins=spec.audit_origins,
        horizons=(spec.selected_duration,),
        exclusions=spec.exclusions,
        cadence=spec.cadence,
        min_training_periods=spec.min_training_periods,
    )
    schedule = historical_validation.validate_schedule(
        frame,
        spec.treated_unit,
        spec.control_units,
        request,
    )
    planned = [window for window in schedule.windows if window.role == "selection"]
    usable = [window for window in planned if window.status == "usable"]
    if not usable:
        return DurationAudit(
            status="inconclusive",
            evidence_level="insufficient-history",
            selected_duration=spec.selected_duration,
            persistent_bias=None,
            n_inference_fits=0,
            scored_count=0,
            failed_count=0,
            requested_count=len(planned),
            reasons=(
                "The frozen duration has no usable audit window. "
                "The selected duration was not changed.",
            ),
            recommendation_text=_recommendation_text(
                "inconclusive",
                "insufficient-history",
                None,
                selected_duration=spec.selected_duration,
            ),
        )
    decision = DecisionRule(
        spec.decision_direction,
        spec.decision_estimand,
        threshold=spec.decision_threshold,
        probability=spec.decision_probability,
    )
    template = historical_validation._replay_template(
        spec.template, spec.sample_kwargs, spec.random_seed
    )
    replays: list[WindowReplay] = []
    fits = 0
    for window in planned:
        if window.status != "usable":
            replays.append(
                historical_validation.WindowReplay(
                    plan=window,
                    status="not-run",
                    reason=window.reason,
                )
            )
            continue
        replay, attempts = historical_validation._replay_window(
            frame,
            spec.treated_unit,
            spec.control_units,
            window,
            decision,
            template=template,
            donor_correlation=spec.min_donor_correlation,
            auto_scale_sigma=spec.auto_scale_sigma,
            fit_audit=False,
        )
        replays.append(replay)
        fits += attempts
    evidence = historical_validation._replay_evidence(tuple(replays))
    level = evidence[0]
    persistent = evidence[5]
    scored = sum(window.status == "scored" for window in replays)
    failed = sum(window.status == "fit-failed" for window in replays)
    if level == "insufficient-history" or scored == 0:
        status: AuditStatus = "inconclusive"
    elif persistent or failed or level == "fit-failed":
        status = "limitation"
    elif scored == len(planned):
        status = "historical-descriptive"
    else:
        status = "inconclusive"
    text = _recommendation_text(
        status, level, persistent, selected_duration=spec.selected_duration
    )
    return DurationAudit(
        status=status,
        evidence_level=level,
        selected_duration=spec.selected_duration,
        persistent_bias=persistent,
        n_inference_fits=fits,
        scored_count=scored,
        failed_count=failed,
        requested_count=len(replays),
        reasons=evidence[1],
        recommendation_text=text,
    )


def _execute_specification(
    spec: AnalysisSpecification, history: pd.DataFrame, treatment_time: Any
) -> SyntheticControl:
    from causalpy.experiments.synthetic_control import SyntheticControl

    frame = _analysis_frame(history, spec.time_column)
    if spec.exclusions:
        excluded = historical_validation._stored_labels(frame.index, spec.exclusions)
        position = historical_validation._label_position(frame.index, treatment_time)
        if position is not None and frame.index[position] in excluded:
            raise ValueError(
                "treatment_time is an excluded label; refusing to shift the split."
            )
    request = historical_validation.ScheduleRequest(
        selection_origins=(treatment_time,),
        horizons=(spec.selected_duration,),
        exclusions=spec.exclusions,
        cadence=spec.cadence,
        min_training_periods=spec.min_training_periods,
    )
    schedule = historical_validation.validate_schedule(
        frame,
        spec.treated_unit,
        spec.control_units,
        request,
    )
    planned = [window for window in schedule.windows if window.role == "selection"]
    window = planned[0]
    if window.status != "usable":
        raise ValueError(
            "The frozen duration cannot be executed at "
            f"{treatment_time!r}: {window.reason}"
        )
    analysis = frame.loc[
        list(window.analysis_index), [spec.treated_unit, *spec.control_units]
    ]
    template = historical_validation._replay_template(
        spec.template, spec.sample_kwargs, spec.random_seed
    )
    experiment = SyntheticControl(
        analysis.copy(),
        window.origin,
        control_units=list(spec.control_units),
        treated_units=[spec.treated_unit],
        model=template,
        min_donor_correlation=spec.min_donor_correlation,
        auto_scale_sigma=spec.auto_scale_sigma,
    )
    return experiment.fit()
