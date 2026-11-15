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

from collections.abc import Sequence
from dataclasses import dataclass
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
    units: str | Sequence[str],
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

    Parameters
    ----------
    pre_treated : ndarray
        Pre-period treated outcomes on the outcome scale.
    """
    values = np.asarray(pre_treated, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Pre-period treated outcomes must be non-empty and finite.")
    return RELATIVE_FLOOR_SCALE * float(np.median(np.abs(values)))


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
    the whole increment.

    A relative magnitude is a fraction of the cumulative baseline. The
    cumulative group increment is ``magnitude * baseline``, spread evenly
    across periods and then across treated units. A scenario of 100 therefore
    means 100 on the group total, not 100 on each unit.

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


@dataclass(frozen=True)
class CandidateRequest:
    """Inputs for a #1441 candidate screen. Screening fits nothing."""

    units: tuple[str, ...]
    min_group_size: int
    max_group_size: int
    constraints: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not GROUP_SIZE_MIN
            <= self.min_group_size
            <= self.max_group_size
            <= GROUP_SIZE_MAX
        ):
            raise ValueError("Candidate group sizes must lie in 1..3.")


@dataclass(frozen=True)
class CandidateTable:
    """Plot-ready candidate rows. No posterior and no estimator class."""

    frame: pd.DataFrame


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
