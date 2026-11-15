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
"""Planning contracts shared by the reference path and later seams.

These types are the public result of a history-only plan. Search, power,
reporting, and plotting read them. They do not carry an estimator object,
an InferenceData tree, or a field named ``power`` for a posterior tail.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, fields
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from causalpy._arviz_compat import hdi_bounds
from causalpy.constants import HDI_PROB

DRAW_ALIGNED_UNIT_SUM = "draw_aligned_unit_impact_sum"
RELATIVE_FLOOR_SCALE = 1e-8
MIN_TRAINING_ROWS = 2
GROUP_SIZE_MIN = 1
GROUP_SIZE_MAX = 3
FIXED_DESIGN_FIT_CEILING = 64
RELATIVE_BASELINE_CONVENTION = (
    "relative_total = cumulative_total / relative_baseline, draw by draw. "
    "The baseline is one finite scalar on the cumulative total outcome "
    "scale. It is not an average of unit-level percentages."
)
OUTCOME_SCALE = "outcome"
CANDIDATE_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "treated",
    "donors",
    "n_donors",
    "estimand",
    "member_weight",
    "cost",
    "outcome_share",
    "proxy_score",
    "proxy_role",
    "boundary",
    "status",
    "filter_reasons",
    "shortlist_rank",
)
CANDIDATE_FILTER_REASONS: tuple[str, ...] = (
    "excluded_unit",
    "not_treatment_eligible",
    "not_donor_eligible",
    "missing_required",
    "donor_depleted",
    "spillover_donor",
    "over_budget",
    "cost_unknown",
    "outcome_share",
    "representativeness",
    "duplicate",
    "group_size",
)
_PROXY_ROLES: tuple[str, ...] = (
    "preperiod_weighted_proportion",
    "unavailable",
)
EnumerationPolicy = Literal["exact", "seeded_hash_sample"]

Direction = Literal["positive", "negative"]
EstimandName = Literal[
    "cumulative_total",
    "average_per_period",
    "average_per_treated_unit",
]
EffectScale = Literal["absolute", "relative"]
QualificationScope = Literal["fixed_design", "post_search"]
UncertaintyKind = Literal[
    "posterior_conditional_mean",
    "posterior_predictive",
    "repeated_data_estimate",
]
Translation = Literal["deterministic_additive", "observed"]
DiagnosticsStatus = Literal["available", "partial", "unavailable"]

DECLARED_ESTIMATORS: tuple[str, ...] = (
    "synthetic_control",
    "augmented_synthetic_control",
    "interactive_fixed_effects",
)
UNIMPLEMENTED_ESTIMATORS: tuple[str, ...] = (
    "augmented_synthetic_control",
    "interactive_fixed_effects",
)

REJECTED_ESTIMATOR_MAPPINGS: tuple[tuple[str, str], ...] = (
    (
        "raw_sum_versus_simplex",
        "A Dirichlet simplex sums to 1, so the donor combination is on one "
        "unit's outcome scale. A raw sum of treated outcomes is not on that "
        "scale and is not a readout of WeightedSumFitter.",
    ),
    (
        "marginal_interval_sum",
        "Adding per-unit intervals is not the interval of the sum. Group "
        "inference sums impacts inside each posterior draw, then computes "
        "the interval of that sum.",
    ),
    (
        "average_of_relative_percentages",
        "Relative percentages are not averaged across unequal units. A "
        "relative group effect divides the draw-aligned total by an explicit "
        "positive baseline.",
    ),
    (
        "aggregate_first_shared_weights",
        "Averaging treated units and then fitting one shared simplex is a "
        "different estimator. It is not a readout of the existing per-unit "
        "simplex posterior and would change the reported uncertainty.",
    ),
)


def _reject_bool(value: object, name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not a bool.")


def require_finite(value: object, name: str) -> float:
    """Return a finite real number, rejecting bools and missing values.

    Parameters
    ----------
    value : object
        Candidate number.
    name : str
        Name used in the error.
    """
    _reject_bool(value, name)
    if isinstance(value, (int, float, np.floating, np.integer)):
        number = float(value)
        if np.isfinite(number):
            return number
    raise TypeError(f"{name} must be a finite real number.")


def require_probability(value: object, name: str) -> float:
    """Return a probability in ``(0, 1]``.

    Parameters
    ----------
    value : object
        Candidate probability.
    name : str
        Name used in the error.
    """
    number = require_finite(value, name)
    if not 0.0 < number <= 1.0:
        raise ValueError(f"{name} must be in (0, 1].")
    return number


def require_positive_int(value: object, name: str) -> int:
    """Return a positive integer, rejecting bools and non-integral values.

    Parameters
    ----------
    value : object
        Candidate integer.
    name : str
        Name used in the error.
    """
    _reject_bool(value, name)
    if isinstance(value, (int, np.integer)):
        number = int(value)
        if number >= 1:
            return number
    raise TypeError(f"{name} must be a positive integer.")


def require_unit_names(
    units: object,
    *,
    what: str,
    min_count: int,
    max_count: int | None,
) -> tuple[str, ...]:
    """Return unique non-empty unit names.

    A string is one unit name, not a sequence of characters. ``"ab"`` is one
    unit. A sequence of names is a group.

    Parameters
    ----------
    units : str or sequence of str
        One unit name or a group of names.
    what : str
        Name used in the error.
    min_count : int
        Minimum number of names.
    max_count : int or None
        Maximum number of names, or ``None`` for no maximum.
    """
    names: tuple[str, ...]
    if isinstance(units, str):
        names = (units,)
    elif isinstance(units, Sequence):
        names = tuple(units)
    else:
        raise TypeError(f"{what} must be a unit name or a sequence of names.")
    if names and not all(isinstance(name, str) and name for name in names):
        raise TypeError(f"{what} must be non-empty strings.")
    if len(names) != len(set(names)):
        raise ValueError(f"{what} must be unique.")
    if len(names) < min_count:
        raise ValueError(f"{what} needs at least {min_count} name(s).")
    if max_count is not None and len(names) > max_count:
        raise ValueError(f"{what} accepts at most {max_count} name(s).")
    return names


def default_relative_floor(pre_treated: np.ndarray) -> float:
    """Return ``1e-8`` times the median absolute pre-period treated outcome.

    A non-positive scale is undefined. The floor must not collapse to zero
    and then accept a near-zero baseline.

    Parameters
    ----------
    pre_treated : ndarray
        Pre-period treated outcomes on the outcome scale.

    Raises
    ------
    ValueError
        If the outcomes are empty, non-finite, or have a non-positive
        median absolute value.
    """
    values = np.asarray(pre_treated, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Pre-period treated outcomes must be non-empty and finite.")
    scale = float(np.median(np.abs(values)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "Relative effects need a strictly positive pre-period treated "
            "scale. The median absolute pre-period outcome is not positive, "
            "so no baseline convention is defined. No fit was started."
        )
    return RELATIVE_FLOOR_SCALE * scale


def require_relative_baseline(
    baseline: object,
    *,
    floor: float,
) -> float:
    """Accept one strictly positive baseline strictly above ``floor``.

    Near-zero, signed, and missing baselines are rejected. There is no
    silent convention that turns a negative holdout into a positive effect.

    Parameters
    ----------
    baseline : object
        Declared cumulative baseline on the outcome scale.
    floor : float
        Exclusive lower bound. The baseline must be strictly above it.
    """
    if baseline is None:
        raise ValueError(
            "Relative effects require relative_baseline, a finite scalar on "
            "the cumulative total outcome scale. No baseline was declared, "
            "so no fit was started."
        )
    number = require_finite(baseline, "relative_baseline")
    floor_value = require_finite(floor, "relative_floor")
    if floor_value < 0.0:
        raise ValueError("relative_floor must be non-negative.")
    if number <= 0.0 or number <= floor_value:
        raise ValueError(
            "Relative effects need a baseline that is strictly positive and "
            f"strictly above relative_floor={floor_value!r}. "
            f"relative_baseline={number!r} was rejected. A signed or "
            "near-zero baseline is not a convention this planner will infer. "
            "No fit was started."
        )
    return number


def unit_period_effect(
    *,
    scale: EffectScale,
    magnitude: float,
    duration: int,
    n_treated: int,
    baseline: float | None = None,
) -> float:
    """Return the per-unit per-period additive increment for one scenario.

    An absolute magnitude is the group-total per-period increment on the
    outcome scale, split equally across treated units. A singleton receives
    the whole increment. A scenario of 100 therefore means 100 on the group
    total, not 100 on each unit.

    A relative magnitude is a fraction of the cumulative baseline. The
    cumulative group increment is ``magnitude * baseline``, spread evenly
    across periods and then across treated units. A relative scenario of
    100 is ``100 * baseline`` on the group total, not 100 on the outcome scale.

    Parameters
    ----------
    scale : {"absolute", "relative"}
        Whether ``magnitude`` is an outcome-scale total or a baseline fraction.
    magnitude : float
        Group-total effect. Absolute values are per period; relative values
        are fractions of the cumulative baseline.
    duration : int
        Number of intervention periods.
    n_treated : int
        Number of treated units sharing the group total.
    baseline : float or None, optional
        Cumulative baseline. Required for a relative effect and rejected for
        an absolute effect.
    """
    if scale not in ("absolute", "relative"):
        raise ValueError("scale must be 'absolute' or 'relative'.")
    effect = require_finite(magnitude, "magnitude")
    periods = require_positive_int(duration, "duration")
    units = require_positive_int(n_treated, "n_treated")
    if scale == "absolute":
        if baseline is not None:
            raise ValueError("An absolute effect does not take a baseline.")
        group_per_period = effect
    else:
        if baseline is None:
            raise ValueError("A relative effect requires a baseline.")
        group_per_period = effect * require_finite(baseline, "baseline") / periods
    return group_per_period / units


def aggregate_unit_impacts(
    unit_impact: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Sum unit impacts inside each posterior draw.

    This is the only readout of the existing per-unit simplex that satisfies
    the group-total contract. The returned cumulative total keeps the ``chain``
    and ``draw`` coordinates. Averaging those draws, or adding their marginal
    intervals, is not this function.

    Parameters
    ----------
    unit_impact : DataArray
        Posterior conditional-mean impacts with dimensions
        ``(chain, draw, obs_ind, treated_units)``.

    Returns
    -------
    cumulative_total, average_per_period, average_per_treated_unit
        Each is a ``(chain, draw)`` array on the outcome scale.
    """
    dims = tuple(unit_impact.dims)
    if dims != ("chain", "draw", "obs_ind", "treated_units"):
        raise ValueError(
            "unit_impact must have dimensions "
            "(chain, draw, obs_ind, treated_units), "
            f"got {dims}."
        )
    if unit_impact.sizes["obs_ind"] < 1 or unit_impact.sizes["treated_units"] < 1:
        raise ValueError("unit_impact must contain at least one period and one unit.")
    if not np.isfinite(np.asarray(unit_impact.values)).all():
        raise ValueError("unit_impact must be finite.")
    total = unit_impact.sum(dim=("obs_ind", "treated_units"), keep_attrs=False)
    duration = int(unit_impact.sizes["obs_ind"])
    n_treated = int(unit_impact.sizes["treated_units"])
    return total, total / duration, total / n_treated


def estimand_draws(
    unit_impact: xr.DataArray,
    *,
    counterfactual: xr.DataArray,
    baseline: float | None,
) -> EstimandDraws:
    """Build labeled estimand draws from a unit-impact array.

    Parameters
    ----------
    unit_impact : DataArray
        Posterior conditional-mean impacts with dimensions
        ``(chain, draw, obs_ind, treated_units)``.
    counterfactual : DataArray
        Untranslated counterfactual on the same observation coordinates.
    baseline : float or None
        Cumulative baseline used for ``relative_total``, or ``None`` when
        relative effects were not requested.
    """
    total, per_period, per_unit = aggregate_unit_impacts(unit_impact)
    relative = None if baseline is None else total / baseline
    return EstimandDraws(
        unit_impact=unit_impact,
        counterfactual=counterfactual,
        cumulative_total=total,
        average_per_period=per_period,
        average_per_treated_unit=per_unit,
        relative_total=relative,
        uncertainty="posterior_conditional_mean",
    )


def summarize_draws(
    draws: EstimandDraws,
    *,
    baseline: float | None,
    baseline_convention: str | None,
    treated_count: int,
    duration: int,
    outcome_scale: str,
) -> EstimandSummary:
    """Summarize posterior conditional-mean draws. This is not a power estimate.

    Parameters
    ----------
    draws : EstimandDraws
        Draw-level estimands from one translated fit.
    baseline : float or None
        Cumulative baseline recorded on the summary, or ``None``.
    baseline_convention : str or None
        How that baseline was obtained, or ``None``.
    treated_count : int
        Number of treated units in the fit.
    duration : int
        Number of intervention periods in the fit.
    outcome_scale : str
        Label for the outcome units.
    """
    total_lower, total_upper = _interval(draws.cumulative_total)
    period_lower, period_upper = _interval(draws.average_per_period)
    unit_lower, unit_upper = _interval(draws.average_per_treated_unit)
    relative_mean = None
    relative_lower = None
    relative_upper = None
    if draws.relative_total is not None:
        relative_mean = float(draws.relative_total.mean().item())
        relative_lower, relative_upper = _interval(draws.relative_total)
    return EstimandSummary(
        cumulative_total_mean=float(draws.cumulative_total.mean().item()),
        cumulative_total_hdi=(total_lower, total_upper),
        average_per_period_mean=float(draws.average_per_period.mean().item()),
        average_per_period_hdi=(period_lower, period_upper),
        average_per_treated_unit_mean=float(
            draws.average_per_treated_unit.mean().item()
        ),
        average_per_treated_unit_hdi=(unit_lower, unit_upper),
        relative_total_mean=relative_mean,
        relative_total_hdi=(
            None
            if relative_lower is None or relative_upper is None
            else (relative_lower, relative_upper)
        ),
        relative_baseline=baseline,
        relative_baseline_convention=baseline_convention,
        treated_count=treated_count,
        duration=duration,
        hdi_probability=HDI_PROB,
        outcome_scale=outcome_scale,
    )


def decide(draws: xr.DataArray, rule: DecisionRule) -> AnalysisDecision:
    """Decide from posterior conditional-mean draws.

    ``posterior_tail_probability`` is the share of draws on the requested
    side of the threshold. It is not repeated-experiment power and it is not
    written into a field named ``power``. Draws equal to the threshold do not
    count as evidence past it. The HDI uses :data:`causalpy.constants.HDI_PROB`
    and is a separate field from the decision probability.

    Parameters
    ----------
    draws : DataArray
        Estimand draws with dimensions ``(chain, draw)``.
    rule : DecisionRule
        One-sided direction, estimand, and threshold.
    """
    if tuple(draws.dims) != ("chain", "draw"):
        raise ValueError(
            "Decision draws must have dimensions (chain, draw), "
            f"got {tuple(draws.dims)}."
        )
    values = np.asarray(draws.values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Decision draws must be non-empty and finite.")
    if rule.direction == "positive":
        tail = float(np.mean(values > rule.threshold))
    else:
        tail = float(np.mean(values < rule.threshold))
    lower, upper = _interval(draws)
    return AnalysisDecision(
        direction=rule.direction,
        estimand=rule.estimand,
        threshold=rule.threshold,
        required_probability=rule.probability,
        posterior_tail_probability=tail,
        decided=tail >= rule.probability,
        hdi_lower=lower,
        hdi_upper=upper,
        hdi_covers_threshold=bool(lower <= rule.threshold <= upper),
        hdi_probability=HDI_PROB,
    )


def _interval(draws: xr.DataArray) -> tuple[float, float]:
    lower, upper = hdi_bounds(draws, prob=HDI_PROB, dim=("chain", "draw"))
    return float(lower), float(upper)


@dataclass(frozen=True)
class DecisionRule:
    """One-sided decision on one estimand.

    Two-sided and sequential rules are not supported. Passing those names
    raises before a plan is built and before any sampling.
    """

    direction: Direction
    estimand: EstimandName
    threshold: float
    probability: float

    def __post_init__(self) -> None:
        if self.direction not in ("positive", "negative"):
            raise ValueError(
                "Only positive and negative one-sided decisions are supported. "
                f"Got {self.direction!r}. Two-sided and sequential rules are "
                "deferred and were not implemented. No fit was started."
            )
        if self.estimand not in (
            "cumulative_total",
            "average_per_period",
            "average_per_treated_unit",
        ):
            raise ValueError(
                "estimand must be cumulative_total, average_per_period, or "
                f"average_per_treated_unit. Got {self.estimand!r}."
            )
        object.__setattr__(
            self, "threshold", require_finite(self.threshold, "threshold")
        )
        object.__setattr__(
            self,
            "probability",
            require_probability(self.probability, "probability"),
        )


@dataclass(frozen=True)
class EffectScenario:
    """One planned effect. The magnitude follows :func:`unit_period_effect`."""

    scale: EffectScale
    magnitude: float
    baseline: float | None = None

    def __post_init__(self) -> None:
        if self.scale not in ("absolute", "relative"):
            raise ValueError("scale must be 'absolute' or 'relative'.")
        object.__setattr__(
            self, "magnitude", require_finite(self.magnitude, "magnitude")
        )
        if self.scale == "absolute" and self.baseline is not None:
            raise ValueError("An absolute effect does not take a baseline.")
        if self.scale == "relative":
            if self.baseline is None:
                raise ValueError("A relative effect requires a baseline.")
            object.__setattr__(
                self,
                "baseline",
                require_finite(self.baseline, "baseline"),
            )


@dataclass(frozen=True)
class PowerCriterion:
    """Target for a repeated-data power proof. This object does not compute power.

    ``target_power`` and ``false_positive_tolerance`` are probabilities of an
    analysis decision over a declared scenario distribution. They are not
    posterior tail probabilities. ``qualification_scope`` separates a
    fixed-design proof from a post-search proof. This contract does not
    authorize the post-search proof.
    """

    target_power: float
    false_positive_tolerance: float
    precision_target: float | None
    monte_carlo_error_tolerance: float
    qualification_scope: QualificationScope

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_power",
            require_probability(self.target_power, "target_power"),
        )
        object.__setattr__(
            self,
            "false_positive_tolerance",
            require_probability(
                self.false_positive_tolerance, "false_positive_tolerance"
            ),
        )
        if self.precision_target is not None:
            object.__setattr__(
                self,
                "precision_target",
                require_finite(self.precision_target, "precision_target"),
            )
        object.__setattr__(
            self,
            "monte_carlo_error_tolerance",
            require_probability(
                self.monte_carlo_error_tolerance,
                "monte_carlo_error_tolerance",
            ),
        )
        if self.qualification_scope not in ("fixed_design", "post_search"):
            raise ValueError("qualification_scope must be fixed_design or post_search.")


@dataclass(frozen=True)
class FitDiagnostics:
    """Sampler diagnostics copied from the fitted experiment.

    These fields do not say whether the fit is good enough to launch.
    """

    divergences: int | None
    max_rhat: float | None
    min_bulk_ess: float | None
    diagnostics_status: DiagnosticsStatus
    diagnostics_reason: str | None


@dataclass(frozen=True)
class DataBoundary:
    """Rows and units the fit was allowed to use."""

    training_index: pd.Index
    holdout_index: pd.Index
    treated_units: tuple[str, ...]
    control_units: tuple[str, ...]
    post_period_treated_in_fit: bool
    post_period_donors_in_prediction: bool
    horizon_policy: Literal["separate-fits"]
    rows_outside_window_excluded: bool


@dataclass(frozen=True)
class Provenance:
    """How a translated result was produced. Not a sampling-quality verdict."""

    estimator: str
    adapter: str
    model_class: str
    random_seed: int | None
    sample_kwargs: tuple[tuple[str, Any], ...]
    horizon_policy: Literal["separate-fits"]
    translation: Translation
    uncertainty: UncertaintyKind


@dataclass(frozen=True)
class EstimandDraws:
    """Posterior conditional-mean draws, not posterior predictive draws.

    ``unit_impact`` has dimensions ``chain``, ``draw``, ``obs_ind``, and
    ``treated_units``. The three totals keep ``chain`` and ``draw``.
    ``counterfactual`` is the untranslated fitted mean on the same axes as
    ``unit_impact``. Adding a planned effect does not change it.
    """

    unit_impact: xr.DataArray
    counterfactual: xr.DataArray
    cumulative_total: xr.DataArray
    average_per_period: xr.DataArray
    average_per_treated_unit: xr.DataArray
    relative_total: xr.DataArray | None
    uncertainty: UncertaintyKind


@dataclass(frozen=True)
class EstimandSummary:
    """Means and HDIs of the posterior conditional-mean estimands."""

    cumulative_total_mean: float
    cumulative_total_hdi: tuple[float, float]
    average_per_period_mean: float
    average_per_period_hdi: tuple[float, float]
    average_per_treated_unit_mean: float
    average_per_treated_unit_hdi: tuple[float, float]
    relative_total_mean: float | None
    relative_total_hdi: tuple[float, float] | None
    relative_baseline: float | None
    relative_baseline_convention: str | None
    treated_count: int
    duration: int
    hdi_probability: float
    outcome_scale: str


@dataclass(frozen=True)
class AnalysisDecision:
    """The analysis decision on one estimand.

    There is no ``power`` field. Repeated-experiment power is a different
    quantity and is not stored here.
    """

    direction: Direction
    estimand: EstimandName
    threshold: float
    required_probability: float
    posterior_tail_probability: float
    decided: bool
    hdi_lower: float
    hdi_upper: float
    hdi_covers_threshold: bool
    hdi_probability: float


@dataclass(frozen=True)
class CanonicalEvaluation:
    """One duration and one effect, after translation."""

    estimator: str
    duration: int
    scenario: EffectScenario
    draws: EstimandDraws
    summary: EstimandSummary
    decision: AnalysisDecision
    diagnostics: FitDiagnostics
    boundary: DataBoundary
    provenance: Provenance
    fit_index: int
    outcome_scale: str


@dataclass(frozen=True)
class PlanningResult:
    """History-only planning result.

    ``repeated_experiment_power`` and ``historical_detection_share`` are
    always ``None`` on this path. ``power_status`` says they were not
    computed. They are not aliases of ``posterior_tail_probability``.
    """

    evaluations: tuple[CanonicalEvaluation, ...]
    n_inference_fits: int
    capabilities: Capabilities
    repeated_experiment_power: None = None
    historical_detection_share: None = None
    power_status: Literal["not_computed"] = "not_computed"


@dataclass(frozen=True)
class Capabilities:
    """What an adapter has evidence for. Unset flags are not implied."""

    estimator: str
    deterministic_effect_reuse: bool
    horizon_reuse: bool
    post_period_donor_data: bool
    group_support: bool
    max_group_size: int
    allowed_decision_rules: tuple[Direction, ...]
    evidence: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class WindowRequest:
    """One horizon for an adapter to fit. No effect has been applied."""

    history: pd.DataFrame
    treated_units: tuple[str, ...]
    control_units: tuple[str, ...]
    treatment_time: Any
    duration: int
    outcome_scale: str


@dataclass(frozen=True)
class CanonicalFit:
    """A fitted window before effect translation.

    ``n_inference_fits`` is 1 for the reference adapter. The fit does not
    contain an InferenceData object or an experiment instance.
    """

    unit_impact: xr.DataArray
    counterfactual: xr.DataArray
    diagnostics: FitDiagnostics
    boundary: DataBoundary
    provenance: Provenance
    n_inference_fits: int


@dataclass(frozen=True)
class PowerRequest:
    """Inputs for a #1439 power proof. This object does not run one."""

    scenario: EffectScenario
    decision: DecisionRule
    criterion: PowerCriterion
    estimator: str
    group_size: int
    duration: int

    def __post_init__(self) -> None:
        if self.estimator not in DECLARED_ESTIMATORS:
            raise ValueError(f"Unknown estimator {self.estimator!r}.")
        if not GROUP_SIZE_MIN <= self.group_size <= GROUP_SIZE_MAX:
            raise ValueError("group_size must be from 1 to 3.")
        object.__setattr__(
            self, "duration", require_positive_int(self.duration, "duration")
        )


@dataclass(frozen=True)
class RepeatedExperimentPower:
    """A power estimate. #1439 fills this. It is not a posterior tail."""

    probability: float
    monte_carlo_error: float
    n_replicates: int
    qualification_scope: QualificationScope
    estimator: str
    decision: DecisionRule


def _require_nonnegative_int(value: object, name: str) -> int:
    _reject_bool(value, name)
    if isinstance(value, (int, np.integer)) and int(value) >= 0:
        return int(value)
    raise TypeError(f"{name} must be a nonnegative integer.")


def _require_seed(value: object) -> int:
    _reject_bool(value, "random_seed")
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError("random_seed must be an int.")


def _require_group_bounds(
    min_group_size: object, max_group_size: object
) -> tuple[int, int]:
    _reject_bool(min_group_size, "min_group_size")
    _reject_bool(max_group_size, "max_group_size")
    if (
        isinstance(min_group_size, (int, np.integer))
        and isinstance(max_group_size, (int, np.integer))
        and GROUP_SIZE_MIN
        <= int(min_group_size)
        <= int(max_group_size)
        <= GROUP_SIZE_MAX
    ):
        return int(min_group_size), int(max_group_size)
    raise ValueError("Candidate group sizes must lie in 1..3.")


def _known_names(
    value: object, what: str, units: set[str], *, min_count: int
) -> tuple[str, ...]:
    names = require_unit_names(value, what=what, min_count=min_count, max_count=None)
    unknown = [name for name in names if name not in units]
    if unknown:
        raise ValueError(f"{what} names an unknown unit {unknown[0]!r}.")
    return names


def _optional_names(
    value: object, what: str, units: set[str]
) -> tuple[str, ...] | None:
    if value is None:
        return None
    return _known_names(value, what, units, min_count=0)


def _require_disjoint(
    left: tuple[str, ...], right: tuple[str, ...], message: str
) -> None:
    if set(left) & set(right):
        raise ValueError(message)


def _require_groups(
    value: object,
    what: str,
    units: set[str],
    *,
    min_count: int,
) -> tuple[tuple[str, ...], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{what} must be a sequence of unit groups.")
    return tuple(
        _known_names(group, what, units, min_count=min_count) for group in value
    )


def _require_spillovers(
    value: object,
    units: set[str],
) -> tuple[tuple[str, str], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("spillover_exclusions must be a sequence of pairs.")
    pairs: list[tuple[str, str]] = []
    for item in value:
        if (
            isinstance(item, str)
            or not isinstance(item, Sequence)
            or len(tuple(item)) != 2
        ):
            raise TypeError("spillover_exclusions must be pairs of unit names.")
        source, contaminated = tuple(item)
        if source == contaminated:
            raise ValueError("spillover_exclusions cannot contain a self-pair.")
        pair = _known_names(
            (source, contaminated), "spillover_exclusions", units, min_count=2
        )
        if pair in pairs:
            raise ValueError("spillover_exclusions must be unique.")
        pairs.append((pair[0], pair[1]))
    return tuple(pairs)


def _require_costs(
    value: object,
    units: set[str],
) -> tuple[tuple[str, float], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("unit_costs must be a sequence of (unit, cost) pairs.")
    costs: list[tuple[str, float]] = []
    seen: set[str] = set()
    for item in value:
        if (
            isinstance(item, str)
            or not isinstance(item, Sequence)
            or len(tuple(item)) != 2
        ):
            raise TypeError("unit_costs must be (unit, cost) pairs.")
        name, raw = tuple(item)
        (unit,) = _known_names((name,), "unit_costs", units, min_count=1)
        if unit in seen:
            raise ValueError("unit_costs names must be unique.")
        seen.add(unit)
        costs.append((unit, require_finite(raw, "unit_costs")))
    return tuple(costs)


def _require_covariates(
    value: object,
    units: set[str],
) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("unit_covariates must be a sequence of covariate tables.")
    tables: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    seen: set[str] = set()
    for item in value:
        if (
            isinstance(item, str)
            or not isinstance(item, Sequence)
            or len(tuple(item)) != 2
        ):
            raise TypeError("unit_covariates entries must be (covariate, values).")
        covariate, raw_values = tuple(item)
        if not isinstance(covariate, str) or not covariate:
            raise TypeError("covariate names must be non-empty strings.")
        if covariate in seen:
            raise ValueError("unit_covariates names must be unique.")
        seen.add(covariate)
        if isinstance(raw_values, str) or not isinstance(raw_values, Sequence):
            raise TypeError("covariate values must be (unit, value) pairs.")
        values: list[tuple[str, float]] = []
        seen_units: set[str] = set()
        for pair in raw_values:
            if (
                isinstance(pair, str)
                or not isinstance(pair, Sequence)
                or len(tuple(pair)) != 2
            ):
                raise TypeError("covariate values must be (unit, value) pairs.")
            name, raw = tuple(pair)
            (unit,) = _known_names((name,), "unit_covariates", units, min_count=1)
            if unit in seen_units:
                raise ValueError("unit_covariates units must be unique.")
            seen_units.add(unit)
            values.append((unit, require_finite(raw, "unit_covariates")))
        tables.append((covariate, tuple(values)))
    return tuple(tables)


def _require_share(value: object, name: str) -> float | None:
    if value is None:
        return None
    number = require_finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return number


def _require_budget(value: object) -> float | None:
    if value is None:
        return None
    number = require_finite(value, "budget")
    if number < 0.0:
        raise ValueError("budget must be nonnegative.")
    return number


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (np.ndarray, pd.Series, pd.Index)):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _candidate_id(treated: tuple[str, ...], donors: tuple[str, ...]) -> str:
    return json.dumps([list(treated), list(donors)], separators=(",", ":"))


def _exact_index_label(index: pd.Index, selection_end: object) -> object:
    """Return the one index label equal to ``selection_end``.

    ``DatetimeIndex`` membership treats ``"2020-06"`` as June. ``in`` and a
    label slice are not an exact-label check.
    """
    matches: list[object] = []
    for label in index:
        try:
            equal = label == selection_end
        except TypeError as exc:
            raise ValueError(
                "selection_end is not comparable with the history index."
            ) from exc
        if not isinstance(equal, (bool, np.bool_)):
            raise ValueError("selection_end is not comparable with the history index.")
        if bool(equal):
            matches.append(label)
    if len(matches) != 1:
        raise ValueError("selection_end is not in the history index.")
    return matches[0]


def _rows_at_or_before(index: pd.Index, cutoff: object) -> list[bool]:
    flags: list[bool] = []
    for label in index:
        try:
            compared = label <= cutoff
        except TypeError as exc:
            raise ValueError(
                "selection_end is not comparable with the history index."
            ) from exc
        if not isinstance(compared, (bool, np.bool_)):
            raise ValueError("selection_end is not comparable with the history index.")
        flags.append(bool(compared))
    return flags


def _require_finite_outcomes(selected: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for name in columns:
        for value in selected[name]:
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"history outcomes must be numeric. Got {name!r}.")
            if not np.isfinite(value):
                raise ValueError(f"history outcomes must be finite. Got {name!r}.")


def _require_selection_history(
    history: object,
    selection_end: object,
    required_columns: tuple[str, ...],
) -> tuple[pd.DataFrame | None, object]:
    if history is None:
        if selection_end is not None:
            raise ValueError("selection_end requires history.")
        return None, None
    if selection_end is None:
        raise ValueError("selection_end is required when history is set.")
    if not isinstance(history, pd.DataFrame):
        raise TypeError("history must be a pandas DataFrame.")
    if isinstance(history.index, pd.MultiIndex):
        raise TypeError("history index must be a single time index.")
    if not history.index.is_unique:
        raise ValueError("history index must be unique.")
    if not all(isinstance(column, str) and column for column in history.columns):
        raise TypeError("history columns must be non-empty strings.")
    if history.columns.duplicated().any():
        raise ValueError("history columns must be unique.")
    missing = [name for name in required_columns if name not in history.columns]
    if missing:
        raise ValueError(f"history is missing units {missing}.")
    cutoff = _exact_index_label(history.index, selection_end)
    selected = history.iloc[_rows_at_or_before(history.index, cutoff)].copy()
    _require_finite_outcomes(selected, required_columns)
    if not selected.index.is_monotonic_increasing:
        raise ValueError("history index must increase.")
    return selected, cutoff


@dataclass(frozen=True)
class BalanceConstraint:
    """One equal-weight covariate gap. #1441 computes it; this object does not.

    The values are the supplied ``unit_covariates`` scalars for this
    covariate. They are not history columns and not a time average. The gap
    is the absolute difference between the equal-weight mean of the treated
    units and the equal-weight mean of ``units`` minus ``excluded_units``.
    That population includes the treated units. An excluded unit does not
    need the covariate. A treated or population unit missing the value, an
    empty population, or a covariate absent from ``unit_covariates`` filters
    that candidate with reason ``representativeness``. No imputation and no
    division by zero. Do not compare against the donor pool. A gap above
    ``max_abs_mean_gap`` filters; a gap equal to the cap passes. Construction
    still accepts a constraint whose covariate is absent.

    Notes
    -----
    covariate : str
        Supplied covariate name. Not estimated.
    max_abs_mean_gap : float
        Nonnegative finite cap on the absolute gap.
    """

    covariate: str
    max_abs_mean_gap: float

    def __post_init__(self) -> None:
        if not isinstance(self.covariate, str) or not self.covariate:
            raise TypeError("covariate must be a non-empty string.")
        gap = require_finite(self.max_abs_mean_gap, "max_abs_mean_gap")
        if gap < 0.0:
            raise ValueError("max_abs_mean_gap must be nonnegative.")
        object.__setattr__(self, "max_abs_mean_gap", gap)


class _ScreenValue:
    """Value equality for a frozen contract that stores a DataFrame copy."""

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return _fields_equal(self, other)

    def __hash__(self) -> int:
        raise TypeError(f"unhashable type: {type(self).__name__!r}")


def _fields_equal(left: Any, right: Any) -> bool:
    for field in fields(left):
        own = object.__getattribute__(left, field.name)
        other = object.__getattribute__(right, field.name)
        if isinstance(own, pd.DataFrame) or isinstance(other, pd.DataFrame):
            if not (
                isinstance(own, pd.DataFrame)
                and isinstance(other, pd.DataFrame)
                and own.equals(other)
            ):
                return False
            continue
        if own != other:
            return False
    return True


@dataclass(frozen=True, eq=False)
class CandidateRequest(_ScreenValue):
    """Inputs for a #1441 candidate screen. Construction does not generate or fit.

    Screening fits nothing. Every accepted table has ``n_inference_fits``
    equal to 0. The group estimand on every candidate row is
    ``DRAW_ALIGNED_UNIT_SUM``. ``member_weight`` is ``1/k`` for a treated
    group of size ``k``, not a fitted simplex.

    Identity is the sorted treated tuple, plus the sorted donor tuple when
    pools vary. Duplicates and permutations collapse. That collapse is
    #1441's job. This object stores the inputs only.

    Outcome share, when history is set, is the sum of selected-window values
    over the treated units divided by the sum of selected-window values over
    ``units`` minus ``excluded_units``. That denominator includes the treated
    units. A denominator that is not strictly positive filters with reason
    ``outcome_share`` and is not used as a divisor. ``history is None``
    stores a null share. A set outcome-share bound with no history filters
    with reason ``outcome_share`` and is not compared against zero.
    Construction does not reject that combination. A computed share outside a
    set bound filters with ``outcome_share`` and the row stores that share.

    ``history`` stores a copy of the rows whose labels are at or before the
    exact ``selection_end`` label, including that row. Reading ``history``
    returns another copy. A partial date string such as ``"2020-06"`` is not
    a label. Selected outcomes must be finite reals, so ``None`` and
    non-finite values are refused. Later rows are not part of the request.
    An always-control unit is donor-eligible unless a spillover removes it
    from one candidate. When ``donor_eligible`` is set, every
    ``always_control`` name is in it.

    The proxy is not implemented here. The score is the equal-weight mean of
    per-unit RMSE from ``WeightedProportion.fit`` on the selected window.
    ``WeightedProportion.mse`` is that RMSE. Fit each treated unit against
    that candidate's donors. Do not fit the summed series. Lower is better.
    Rank 1 is the lowest score. If history is missing, a selected outcome is
    missing, a unit has no donors, or the fit fails, the role is
    ``unavailable``, the score is null, and rank falls back to cost, then
    ``candidate_id``. If every score is null, skip both proxy slots and fill
    the remaining slots by cost, then ``candidate_id``. Missing cost sorts
    after every known cost and cannot win a cost slot. If every cost is
    missing, skip both cost slots. The stable labels are
    ``preperiod_weighted_proportion`` and ``unavailable``.

    A space at or below ``exact_ceiling`` is enumerated in full. Above that
    ceiling, #1441 ranks identities by the SHA-256 lowercase hex digest of
    the UTF-8 JSON ``[random_seed, sorted treated, sorted donors or null]``
    with separators ``,`` and ``:`` and ``ensure_ascii`` true. Donors are in
    the key only when ``donor_pools`` is set; otherwise the third element is
    ``null``. Lowest means lexicographic order of that digest. Equal digests
    keep the lexicographically smaller ``candidate_id``. It keeps the lowest
    ``max_generated``, reports ``n_space`` as the full count, and sets
    ``partial=True``. It raises if ``max_generated`` is ``None``.

    The shortlist keeps at most ``shortlist_size`` passing rows and does not
    grow the cap. Fill boundary slots in this order, skipping a candidate
    already kept and stopping when the cap is full: lowest cost, highest
    cost, lowest proxy, highest proxy, then one passing group of each integer
    size from ``min_group_size`` to ``max_group_size``. Skip a proxy slot
    when every score is null. Skip a size that has no passing group. Those
    rows have ``boundary=True``. Fill the remaining slots by proxy ascending,
    then cost ascending, then ``candidate_id``. No history, and an all-null
    proxy, use cost, then id. A generator must not invent a different policy.

    Notes
    -----
    units : tuple of str
        Population the names below are drawn from.
    min_group_size : int
        Smallest treated set. From 1 to 3, and no larger than ``max_group_size``.
    max_group_size : int
        Largest treated set. From 1 to 3.
    treatment_eligible : tuple of str or None, optional
        Units that may be treated. ``None`` means every unit, then the
        exclusions below.
    donor_eligible : tuple of str or None, optional
        Units that may donate. ``None`` means every unit, then the exclusions
        below.
    always_control : tuple of str, optional
        Never treated. Donor-eligible unless a spillover rule removes them
        from that candidate.
    excluded_units : tuple of str, optional
        Neither treated nor donor.
    required_treated : tuple of str, optional
        Every generated group contains all of these.
    supplied_groups : tuple of tuple of str or None, optional
        Screen only these treated sets. ``None`` means enumerate. Mutually
        exclusive with ``fixed_treated``. Construction checks names, not
        eligibility, and does not build donors. #1441 attaches every matching
        reason, in this order: length outside the size bounds is
        ``group_size``; an ``excluded_units`` member is ``excluded_unit``; an
        ``always_control`` member, or a member absent from
        ``treatment_eligible`` when that tuple is set, is
        ``not_treatment_eligible``; a missing ``required_treated`` name is
        ``missing_required``. ``fixed_treated`` still rejects those cases.
    fixed_treated : tuple of str or None, optional
        One treated set, size inside the min/max bounds. Mutually exclusive
        with ``supplied_groups``.
    donor_pools : tuple of tuple of str or None, optional
        Only with ``fixed_treated``. Each pool is one candidate. ``None``
        means one pool: the eligible complement after spillover removal.
    spillover_exclusions : tuple of tuple of str, optional
        Directed ``(source, contaminated)`` pairs. If source is treated,
        contaminated cannot donate to that candidate. Not inferred, and does
        not forbid co-treatment.
    min_remaining_donors : int, optional
        After removing treated members and spillover-forbidden donors. A
        treated unit never counts. At least 1.
    history : DataFrame or None, optional
        Time-indexed outcomes. ``None`` means outcome share and the proxy are
        unavailable, not zero.
    selection_end : label or None, optional
        Required exactly when ``history`` is set. The row at this label is
        included. Later labels are not part of the request.
    min_outcome_share : float or None, optional
        Closed lower bound in ``[0, 1]``. ``None`` means unbounded.
    max_outcome_share : float or None, optional
        Closed upper bound in ``[0, 1]``. ``None`` means unbounded.
    unit_costs : tuple of tuple, optional
        Supplied ``(unit, cost)`` pairs. A missing unit is unknown, not zero.
    budget : float or None, optional
        Reject a treated-cost sum above it. Unknown cost with a budget set is
        a filter, not a zero.
    unit_covariates : tuple of tuple, optional
        Supplied ``(covariate, ((unit, value), ...))``. Not estimated.
    balance_constraints : tuple of BalanceConstraint, optional
        Equal-weight treated-versus-population gaps. See
        :class:`BalanceConstraint`.
    exact_ceiling : int, optional
        A space at or below this is enumerated in full. At least 1.
    max_generated : int or None, optional
        Above the ceiling, keep this many by seeded hash. ``None`` means raise
        rather than sample. At least 1 when set.
    shortlist_size : int, optional
        Cap after filters. At least 1.
    random_seed : int
        Required keyword. Keys the hash sample and tie-breaks. Not iteration
        order.
    """

    units: tuple[str, ...]
    min_group_size: int
    max_group_size: int
    _: KW_ONLY
    treatment_eligible: tuple[str, ...] | None = None
    donor_eligible: tuple[str, ...] | None = None
    always_control: tuple[str, ...] = ()
    excluded_units: tuple[str, ...] = ()
    required_treated: tuple[str, ...] = ()
    supplied_groups: tuple[tuple[str, ...], ...] | None = None
    fixed_treated: tuple[str, ...] | None = None
    donor_pools: tuple[tuple[str, ...], ...] | None = None
    spillover_exclusions: tuple[tuple[str, str], ...] = ()
    min_remaining_donors: int = 1
    history: pd.DataFrame | None = None
    selection_end: Any | None = None
    min_outcome_share: float | None = None
    max_outcome_share: float | None = None
    unit_costs: tuple[tuple[str, float], ...] = ()
    budget: float | None = None
    unit_covariates: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    balance_constraints: tuple[BalanceConstraint, ...] = ()
    exact_ceiling: int = 5000
    max_generated: int | None = None
    shortlist_size: int = 10
    random_seed: int

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        if name == "history" and isinstance(value, pd.DataFrame):
            return value.copy()
        return value

    def __post_init__(self) -> None:
        min_group_size, max_group_size = _require_group_bounds(
            self.min_group_size, self.max_group_size
        )
        units = require_unit_names(
            self.units, what="units", min_count=1, max_count=None
        )
        unit_set = set(units)
        treatment_eligible = _optional_names(
            self.treatment_eligible, "treatment_eligible", unit_set
        )
        donor_eligible = _optional_names(
            self.donor_eligible, "donor_eligible", unit_set
        )
        always_control = _known_names(
            self.always_control, "always_control", unit_set, min_count=0
        )
        excluded_units = _known_names(
            self.excluded_units, "excluded_units", unit_set, min_count=0
        )
        required_treated = _known_names(
            self.required_treated, "required_treated", unit_set, min_count=0
        )
        if len(required_treated) > max_group_size:
            raise ValueError("required_treated is longer than max_group_size.")
        _require_disjoint(
            always_control,
            excluded_units,
            "always_control and excluded_units overlap.",
        )
        _require_disjoint(
            always_control,
            required_treated,
            "always_control and required_treated overlap.",
        )
        _require_disjoint(
            excluded_units,
            required_treated,
            "excluded_units and required_treated overlap.",
        )
        if treatment_eligible is not None:
            _require_disjoint(
                always_control,
                treatment_eligible,
                "always_control must be absent from treatment_eligible.",
            )
            _require_disjoint(
                excluded_units,
                treatment_eligible,
                "excluded_units must be absent from treatment_eligible.",
            )
            missing = [
                name for name in required_treated if name not in treatment_eligible
            ]
            if missing:
                raise ValueError("required_treated must be treatment-eligible.")
        if donor_eligible is not None:
            _require_disjoint(
                excluded_units,
                donor_eligible,
                "excluded_units must be absent from donor_eligible.",
            )
            omitted_controls = [
                name for name in always_control if name not in donor_eligible
            ]
            if omitted_controls:
                raise ValueError(
                    "always_control must be donor-eligible. "
                    f"Got {omitted_controls[0]!r}."
                )
        if self.supplied_groups is not None and self.fixed_treated is not None:
            raise ValueError(
                "supplied_groups and fixed_treated are mutually exclusive."
            )
        supplied_groups = (
            None
            if self.supplied_groups is None
            else _require_groups(
                self.supplied_groups, "supplied_groups", unit_set, min_count=1
            )
        )
        fixed_treated = (
            None
            if self.fixed_treated is None
            else _known_names(
                self.fixed_treated, "fixed_treated", unit_set, min_count=1
            )
        )
        if fixed_treated is not None:
            if not min_group_size <= len(fixed_treated) <= max_group_size:
                raise ValueError(
                    "fixed_treated size must lie in min_group_size..max_group_size."
                )
            if not set(required_treated) <= set(fixed_treated):
                raise ValueError("fixed_treated must contain required_treated.")
            _require_disjoint(
                fixed_treated,
                always_control,
                "fixed_treated cannot include always_control.",
            )
            _require_disjoint(
                fixed_treated,
                excluded_units,
                "fixed_treated cannot include excluded_units.",
            )
            if treatment_eligible is not None and not set(fixed_treated) <= set(
                treatment_eligible
            ):
                raise ValueError("fixed_treated must be treatment-eligible.")
        if self.donor_pools is not None and fixed_treated is None:
            raise ValueError("donor_pools requires fixed_treated.")
        donor_pools = (
            None
            if self.donor_pools is None
            else _require_groups(self.donor_pools, "donor_pools", unit_set, min_count=0)
        )
        if donor_pools is not None and fixed_treated is not None:
            treated = set(fixed_treated)
            excluded = set(excluded_units)
            donors = None if donor_eligible is None else set(donor_eligible)
            for pool in donor_pools:
                if treated & set(pool):
                    raise ValueError("donor_pools cannot include a treated unit.")
                if excluded & set(pool):
                    raise ValueError("donor_pools cannot include excluded_units.")
                if donors is not None and not set(pool) <= donors:
                    raise ValueError("donor_pools must be donor-eligible.")
        spillover_exclusions = _require_spillovers(self.spillover_exclusions, unit_set)
        unit_costs = _require_costs(self.unit_costs, unit_set)
        unit_covariates = _require_covariates(self.unit_covariates, unit_set)
        if isinstance(self.balance_constraints, str) or not isinstance(
            self.balance_constraints, Sequence
        ):
            raise TypeError(
                "balance_constraints must be a sequence of BalanceConstraint."
            )
        balance: list[BalanceConstraint] = []
        seen_covariates: set[str] = set()
        for item in self.balance_constraints:
            if not isinstance(item, BalanceConstraint):
                raise TypeError(
                    "balance_constraints must contain BalanceConstraint values."
                )
            if item.covariate in seen_covariates:
                raise ValueError("balance_constraints covariates must be unique.")
            seen_covariates.add(item.covariate)
            balance.append(item)
        min_outcome_share = _require_share(self.min_outcome_share, "min_outcome_share")
        max_outcome_share = _require_share(self.max_outcome_share, "max_outcome_share")
        if (
            min_outcome_share is not None
            and max_outcome_share is not None
            and min_outcome_share > max_outcome_share
        ):
            raise ValueError("min_outcome_share must not exceed max_outcome_share.")
        population = tuple(name for name in units if name not in set(excluded_units))
        history, selection_end = _require_selection_history(
            self.history, self.selection_end, population
        )
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "min_group_size", min_group_size)
        object.__setattr__(self, "max_group_size", max_group_size)
        object.__setattr__(self, "treatment_eligible", treatment_eligible)
        object.__setattr__(self, "donor_eligible", donor_eligible)
        object.__setattr__(self, "always_control", always_control)
        object.__setattr__(self, "excluded_units", excluded_units)
        object.__setattr__(self, "required_treated", required_treated)
        object.__setattr__(self, "supplied_groups", supplied_groups)
        object.__setattr__(self, "fixed_treated", fixed_treated)
        object.__setattr__(self, "donor_pools", donor_pools)
        object.__setattr__(self, "spillover_exclusions", spillover_exclusions)
        object.__setattr__(
            self,
            "min_remaining_donors",
            require_positive_int(self.min_remaining_donors, "min_remaining_donors"),
        )
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "selection_end", selection_end)
        object.__setattr__(self, "min_outcome_share", min_outcome_share)
        object.__setattr__(self, "max_outcome_share", max_outcome_share)
        object.__setattr__(self, "unit_costs", unit_costs)
        object.__setattr__(self, "budget", _require_budget(self.budget))
        object.__setattr__(self, "unit_covariates", unit_covariates)
        object.__setattr__(self, "balance_constraints", tuple(balance))
        object.__setattr__(
            self,
            "exact_ceiling",
            require_positive_int(self.exact_ceiling, "exact_ceiling"),
        )
        max_generated = self.max_generated
        if max_generated is not None:
            max_generated = require_positive_int(max_generated, "max_generated")
        object.__setattr__(self, "max_generated", max_generated)
        object.__setattr__(
            self,
            "shortlist_size",
            require_positive_int(self.shortlist_size, "shortlist_size"),
        )
        object.__setattr__(self, "random_seed", _require_seed(self.random_seed))


def _row_names(value: object, what: str, *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{what} must be a non-empty sequence of unit names.")
    names = tuple(value)
    if not allow_empty and not names:
        raise ValueError(f"{what} must be a non-empty sequence of unit names.")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{what} must be a non-empty sequence of unit names.")
    if len(set(names)) != len(names):
        raise ValueError(f"{what} must be unique unit names.")
    if names != tuple(sorted(names)):
        raise ValueError(f"{what} must be sorted.")
    return names


def _require_shortlist_rank(rank: object) -> int:
    if isinstance(rank, bool) or not isinstance(
        rank, (int, float, np.integer, np.floating)
    ):
        raise ValueError("shortlist_rank must be a positive integer.")
    number = float(rank)
    if not np.isfinite(number) or not number.is_integer() or number < 1:
        raise ValueError("shortlist_rank must be a positive integer.")
    return int(number)


def _require_proxy(role: object, score: object) -> None:
    if role not in _PROXY_ROLES:
        raise ValueError(
            "proxy_role must be preperiod_weighted_proportion or unavailable. "
            f"Got {role!r}."
        )
    if role == "unavailable":
        if not _is_missing(score):
            raise ValueError("unavailable proxy_score must be null.")
        return
    if (
        _is_missing(score)
        or isinstance(score, bool)
        or not isinstance(score, (int, float, np.integer, np.floating))
        or not np.isfinite(float(score))
    ):
        raise ValueError("proxy_score must be finite when the proxy was fit.")


def _require_candidate_rows(
    frame: pd.DataFrame,
    n_enumerated: int,
    n_filtered: int,
    n_shortlisted: int,
) -> pd.DataFrame:
    if len(frame) != n_enumerated:
        raise ValueError("frame length must equal n_enumerated.")
    extra = [name for name in frame.columns if name not in CANDIDATE_COLUMNS]
    if extra:
        raise ValueError(
            f"frame has columns outside CANDIDATE_COLUMNS. Got {extra[0]!r}."
        )
    copied = frame.loc[:, list(CANDIDATE_COLUMNS)].copy()
    filtered = 0
    ranks: list[int] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for position in range(len(copied)):
        row = copied.iloc[position]
        treated = _row_names(row["treated"], "treated", allow_empty=False)
        donors = _row_names(row["donors"], "donors", allow_empty=True)
        if set(treated) & set(donors):
            raise ValueError("donors must be disjoint from treated.")
        n_donors = row["n_donors"]
        if isinstance(n_donors, bool) or not isinstance(n_donors, (int, np.integer)):
            raise TypeError("n_donors must equal len(donors).")
        if int(n_donors) != len(donors):
            raise ValueError("n_donors must equal len(donors).")
        if row["candidate_id"] != _candidate_id(treated, donors):
            raise ValueError(
                "candidate_id must be the sorted treated and donor identity."
            )
        identity = (treated, donors)
        if identity in seen:
            raise ValueError("candidate identity must be unique.")
        seen.add(identity)
        weight = row["member_weight"]
        if isinstance(weight, bool) or not isinstance(
            weight, (int, float, np.integer, np.floating)
        ):
            raise TypeError("member_weight must be a real number.")
        if not np.isfinite(weight) or abs(float(weight) - (1.0 / len(treated))) > 1e-12:
            raise ValueError("member_weight must be 1/k.")
        if row["estimand"] != DRAW_ALIGNED_UNIT_SUM:
            raise ValueError("frame estimand must be DRAW_ALIGNED_UNIT_SUM.")
        _require_proxy(row["proxy_role"], row["proxy_score"])
        share = row["outcome_share"]
        if not _is_missing(share) and (
            isinstance(share, bool)
            or not isinstance(share, (int, float, np.integer, np.floating))
            or not np.isfinite(float(share))
        ):
            raise ValueError("outcome_share must be null or finite.")
        raw_reasons = row["filter_reasons"]
        if isinstance(raw_reasons, str) or not isinstance(raw_reasons, Sequence):
            raise TypeError("filter_reasons must be a sequence of reason codes.")
        reasons = tuple(raw_reasons)
        unknown = [
            reason for reason in reasons if reason not in CANDIDATE_FILTER_REASONS
        ]
        if unknown:
            raise ValueError(
                f"filter_reasons must use CANDIDATE_FILTER_REASONS. Got {unknown[0]!r}."
            )
        rank = row["shortlist_rank"]
        has_rank = not _is_missing(rank)
        if reasons and has_rank:
            raise ValueError("a filtered row cannot have a shortlist_rank.")
        boundary = row["boundary"]
        if isinstance(boundary, np.bool_):
            boundary = bool(boundary)
        if not isinstance(boundary, bool):
            raise TypeError("boundary must be a bool.")
        if reasons:
            if row["status"] != "filtered" or boundary:
                raise ValueError("status must match the row's rank and reasons.")
            filtered += 1
        elif has_rank:
            if row["status"] != "shortlisted":
                raise ValueError("status must match the row's rank and reasons.")
            ranks.append(_require_shortlist_rank(rank))
        else:
            if row["status"] != "enumerated" or boundary:
                raise ValueError("status must match the row's rank and reasons.")
        if not GROUP_SIZE_MIN <= len(treated) <= GROUP_SIZE_MAX and (
            "group_size" not in reasons
        ):
            raise ValueError("a group outside 1..3 must carry group_size.")
    if filtered != n_filtered:
        raise ValueError(
            "n_filtered must equal the number of rows with filter_reasons."
        )
    if len(ranks) != n_shortlisted:
        raise ValueError("n_shortlisted must equal the number of ranked rows.")
    if sorted(ranks) != list(range(1, n_shortlisted + 1)):
        raise ValueError("shortlist_rank must be 1..n_shortlisted without duplicates.")
    return copied


@dataclass(frozen=True, eq=False)
class CandidateTable(_ScreenValue):
    """Every enumerated identity, including rows a filter removed.

    ``shortlist_rank`` is null off the shortlist. ``partial`` is true only
    when ``n_enumerated`` is below ``n_space``. An exact table uses
    ``enumeration_policy="exact"`` and enumerates the whole space. A partial
    table uses ``enumeration_policy="seeded_hash_sample"``. ``estimand`` is
    ``DRAW_ALIGNED_UNIT_SUM``. ``n_inference_fits`` is 0: screening fits
    nothing. The frame is a copy of ``CANDIDATE_COLUMNS`` only. Reading
    ``frame`` returns another copy. Its length equals ``n_enumerated``.
    ``candidate_id`` is the JSON ``[sorted treated, sorted donors]`` with
    separators ``,`` and ``:``. Duplicate identities are refused.
    ``outcome_share`` is null or finite. Rows with a
    reason equal ``n_filtered`` and have status ``filtered``, a null rank,
    and ``boundary=False``. Ranked rows equal ``n_shortlisted``, have status
    ``shortlisted``, and use ranks ``1..n_shortlisted`` once each, including
    an integral float such as ``1.0``. An unranked passing row has status
    ``enumerated``. ``member_weight`` is ``1/k``. Reasons come from
    ``CANDIDATE_FILTER_REASONS``. Ranking, hashing, and boundary rows follow
    :class:`CandidateRequest`. This object does not apply them.

    Notes
    -----
    frame : DataFrame
        One row per enumerated identity. Columns are ``CANDIDATE_COLUMNS``.
    n_space : int
        Full identity count before sampling and filters.
    n_enumerated : int
        Identities actually generated. Equals ``n_space`` when exact.
    n_filtered : int
        Enumerated identities that failed a filter.
    n_shortlisted : int
        Passed identities kept in the cap.
    partial : bool
        True only when ``n_enumerated`` is below ``n_space``.
    enumeration_policy : {"exact", "seeded_hash_sample"}
        How ``n_enumerated`` was chosen.
    estimand : str
        Must be ``DRAW_ALIGNED_UNIT_SUM``.
    n_inference_fits : int
        Must be 0.
    """

    frame: pd.DataFrame
    _: KW_ONLY
    n_space: int
    n_enumerated: int
    n_filtered: int
    n_shortlisted: int
    partial: bool
    enumeration_policy: EnumerationPolicy
    estimand: str
    n_inference_fits: int

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        if name == "frame" and isinstance(value, pd.DataFrame):
            return value.copy()
        return value

    def __post_init__(self) -> None:
        if not isinstance(self.frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame.")
        missing = [name for name in CANDIDATE_COLUMNS if name not in self.frame.columns]
        if missing:
            raise ValueError(f"frame is missing columns {missing}.")
        n_space = _require_nonnegative_int(self.n_space, "n_space")
        n_enumerated = _require_nonnegative_int(self.n_enumerated, "n_enumerated")
        n_filtered = _require_nonnegative_int(self.n_filtered, "n_filtered")
        n_shortlisted = _require_nonnegative_int(self.n_shortlisted, "n_shortlisted")
        if n_enumerated > n_space:
            raise ValueError("n_enumerated cannot exceed n_space.")
        if n_filtered > n_enumerated:
            raise ValueError("n_filtered cannot exceed n_enumerated.")
        if n_shortlisted > n_enumerated - n_filtered:
            raise ValueError("n_shortlisted cannot exceed the unfiltered count.")
        if not isinstance(self.partial, bool):
            raise TypeError("partial must be a bool.")
        if self.partial != (n_enumerated < n_space):
            raise ValueError(
                "partial must be true only when n_enumerated is below n_space."
            )
        if self.enumeration_policy not in ("exact", "seeded_hash_sample"):
            raise ValueError("enumeration_policy must be exact or seeded_hash_sample.")
        if self.partial and self.enumeration_policy != "seeded_hash_sample":
            raise ValueError("a partial table uses seeded_hash_sample.")
        if not self.partial and self.enumeration_policy != "exact":
            raise ValueError("an exact table is not a sample.")
        if self.estimand != DRAW_ALIGNED_UNIT_SUM:
            raise ValueError("estimand must be DRAW_ALIGNED_UNIT_SUM.")
        if isinstance(self.n_inference_fits, bool) or self.n_inference_fits != 0:
            raise ValueError("n_inference_fits must be 0.")
        frame = _require_candidate_rows(
            self.frame, n_enumerated, n_filtered, n_shortlisted
        )
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "n_space", n_space)
        object.__setattr__(self, "n_enumerated", n_enumerated)
        object.__setattr__(self, "n_filtered", n_filtered)
        object.__setattr__(self, "n_shortlisted", n_shortlisted)
        object.__setattr__(self, "n_inference_fits", 0)


@dataclass(frozen=True)
class ValidationCell:
    """One named proof the later power and group work may run.

    ``max_fits`` is a ceiling, not a sample-size recommendation. Screening
    is not a cell: candidate screening stays at zero fits.
    """

    name: str
    owner: str
    question: str
    qualification_scope: QualificationScope
    max_fits: int
    environment: Literal["nightly"]
    purpose: str


# Sixty-four light fits is a few minutes per cell and a Monte Carlo standard
# error of about 0.06 near one half. That can show whether a declared
# decision fires. It is too coarse to treat as a launch study. Post-search
# qualification is a separate budget owned by #319 and is not in this matrix.
# The group cell uses the same ceiling: a size 2-3 proof is one joint fit
# per horizon, not a search, so the ceiling does not grow with group size.
VALIDATION_MATRIX: tuple[ValidationCell, ...] = (
    ValidationCell(
        name="correctly_specified_noisy_reference",
        owner="#1439",
        question="Does the declared decision fire when the effect is real and the model matches?",
        qualification_scope="fixed_design",
        max_fits=FIXED_DESIGN_FIT_CEILING,
        environment="nightly",
        purpose="Reference recovery under independent noise.",
    ),
    ValidationCell(
        name="temporal_dependence",
        owner="#1439",
        question="Does the decision still behave when outcomes are serially dependent?",
        qualification_scope="fixed_design",
        max_fits=FIXED_DESIGN_FIT_CEILING,
        environment="nightly",
        purpose="Reference recovery under AR(1) dependence.",
    ),
    ValidationCell(
        name="persistent_misfit",
        owner="#1439",
        question="Is a pre-period misfit reported as misfit rather than as power?",
        qualification_scope="fixed_design",
        max_fits=FIXED_DESIGN_FIT_CEILING,
        environment="nightly",
        purpose="A bad pre-period fit must not be labeled as detection power.",
    ),
    ValidationCell(
        name="fixed_design_qualification",
        owner="#1439",
        question="What is the repeated-data probability of the declared decision before any search?",
        qualification_scope="fixed_design",
        max_fits=FIXED_DESIGN_FIT_CEILING,
        environment="nightly",
        purpose="Fixed-design power. Not a posterior tail probability.",
    ),
    ValidationCell(
        name="group_draw_aligned_sum",
        owner="#1440",
        question=(
            "Does a declared group of two or three preserve the draw-aligned "
            "sum of unit impacts?"
        ),
        qualification_scope="fixed_design",
        max_fits=FIXED_DESIGN_FIT_CEILING,
        environment="nightly",
        purpose="Group proof. One joint fit per horizon, not a search.",
    ),
)


REFERENCE_CAPABILITIES = Capabilities(
    estimator="synthetic_control",
    deterministic_effect_reuse=True,
    horizon_reuse=False,
    post_period_donor_data=True,
    group_support=False,
    max_group_size=1,
    allowed_decision_rules=("positive", "negative"),
    evidence=(
        ("deterministic_effect_reuse", "test_general_journey_valid_example"),
        ("horizon_reuse", "test_horizons_are_separate_fits"),
        ("post_period_donor_data", "test_post_period_donor_boundary"),
        ("group_support", "test_geo_journey_refuses_pair_before_sampling"),
        ("allowed_decision_rules", "test_decision_fields_are_not_power"),
    ),
)
