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
"""Repeated-experiment power for the synthetic-control reference path.

Import this module directly. It is not re-exported from
``causalpy.planning`` because that package surface belongs to issue #1438.

Power is the share of repeated experiments whose analysis decision fires.
It is not a posterior tail probability, not HDI coverage, and not a
historical detection count. The exact ROPE engine in
``causalpy.checks.operating_characteristics`` is not used: that engine
assumes a normal learned-null and a ROPE decision, and this decision is a
one-sided tail rule applied to translated draws. No sigmoid is fit.

:class:`ReferencePowerEvaluator` closes over a
:class:`~causalpy.planning.reference.HistoryOnlyPlan` and the history that
plan was built from. The repeated-data budget is derived from the request's
Monte Carlo tolerance and :data:`~causalpy.planning.contracts.FIXED_DESIGN_FIT_CEILING`.
The sampler seed is the plan's seed. The scenario family is a constructor
argument because :class:`~causalpy.planning.contracts.PowerRequest` has no
family field, and this issue cannot edit that contract. The family is
recorded on every result. It is not stashed on the request.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from causalpy.planning.contracts import (
    FIXED_DESIGN_FIT_CEILING,
    MIN_TRAINING_ROWS,
    DecisionRule,
    EffectScale,
    EstimandSummary,
    PowerCriterion,
    PowerRequest,
    RepeatedExperimentPower,
    require_finite,
    require_positive_int,
    require_probability,
    unit_period_effect,
)
from causalpy.planning.reference import HistoryOnlyPlan

SCENARIO_FAMILIES: tuple[str, ...] = (
    "correctly_specified_noisy_reference",
    "temporal_dependence",
    "persistent_misfit",
    "fixed_design_qualification",
)
# Order is part of the data-seed contract. Do not reorder it.
NOISE_TO_DONOR_SD = 0.05
TEMPORAL_AR_COEFFICIENT = 0.7
MISFIT_TREND_PER_DONOR_SD = 3.0
MISFIT_RMSE_TO_NOISE = 3.0
DATA_SEED_SCHEME = (
    "numpy.random.SeedSequence([sampler_seed, replicate_index, family_code]), "
    "where family_code is the index of scenario_family in SCENARIO_FAMILIES. "
    "The sampler seed is fixed across replicates. The replicate index defines "
    "the experiment. A new sampler seed on the same series is not a new experiment."
)
CONDITIONAL_ON = (
    "Donor paths are the observed control columns of the closed-over history. "
    "They are fixed across replicates and are not regenerated."
)
MECHANISM = (
    "Each replicate builds one untreated panel from the closed-over donors. "
    "HistoryOnlyPlan.evaluate then fits the synthetic-control adapter once per "
    "duration and translates the effect grid without refitting. Horizon reuse "
    "is off. Treatment is injected once, by that translation, after the "
    "untreated series exists. The repeated-experiment probability is the share "
    "of replicates whose AnalysisDecision.decided is true. It is not a "
    "posterior tail probability. The decision is a one-sided rule on translated "
    "draws. Monte Carlo error is the binomial standard error of that decision "
    "rate. Effect-estimation uncertainty is the mean HDI width of the decision "
    "estimand. Those quantities are different fields. The exact ROPE engine in "
    "operating_characteristics is not used, and no sigmoid is fit."
)
MDE_CONVENTION = (
    "Linear interpolation in the decision direction between the last evaluated "
    "effect whose estimated power is below the target and the first whose "
    "estimated power is at least the target. Zero is always evaluated. If the "
    "zero effect already meets the target, the status is null_control_failure: "
    "that rate is a broken null, not a minimum detectable effect of zero. If the "
    "smallest evaluated nonzero directional effect already meets the target, that "
    "effect is the minimum detectable effect and a smaller effect is not "
    "inferred. If the largest evaluated directional effect is still below the "
    "target, the status is out_of_range. The grid does not extrapolate."
)
_GENERATED: dict[str, str] = {
    "correctly_specified_noisy_reference": (
        "correctly_specified_noisy_reference: the treated unit's untreated "
        "series is an equal-weight combination of the observed donors plus "
        "independent Gaussian noise. Each replicate draws a new series. "
        "Treatment is not in the series the estimator fits."
    ),
    "fixed_design_qualification": (
        "fixed_design_qualification uses the same generator as "
        "correctly_specified_noisy_reference. This name is the fixed-design "
        "probability question, not a different noise process."
    ),
    "temporal_dependence": (
        "temporal_dependence: the treated unit's untreated series is an "
        "equal-weight combination of the observed donors plus stationary AR(1) "
        "noise with coefficient "
        f"{TEMPORAL_AR_COEFFICIENT}. Treatment is injected once after that "
        "series exists, by translating the fitted impact."
    ),
    "persistent_misfit": (
        "persistent_misfit: the treated unit's untreated series is a time trend "
        "plus independent noise, not a convex combination of the observed "
        "donors. The pre-period simplex screen must fail. A decision rate from "
        "that series is not power."
    ),
}
MdeStatus = Literal["attained", "out_of_range", "null_control_failure"]
FRAME_COLUMNS: tuple[str, ...] = (
    "duration",
    "effect",
    "scale",
    "probability",
    "monte_carlo_error",
    "n_decided",
    "n_replicates",
    "posterior_mean",
    "posterior_tail_mean",
    "interval_width",
    "bias",
    "coverage",
    "false_positive_rate",
    "wrong_sign_rate",
    "detection",
    "failed_runs",
    "precision_met",
)


class MisfitNotPower(RuntimeError):
    """A decision rate was computed and must not be returned as power.

    Parameters
    ----------
    reason : str
        Why the decision rate was withheld.
    report : EffectByDuration
        The completed study. Its decision rate is not power.
    """

    def __init__(self, reason: str, report: EffectByDuration) -> None:
        super().__init__(reason)
        self.report = report


@dataclass(frozen=True)
class PowerPoint:
    """One duration and one translated effect.

    ``probability`` is the decision rate. ``posterior_tail_mean`` is the mean
    posterior tail and is not power. ``monte_carlo_error`` is uncertainty in
    the decision rate. ``interval_width`` is effect-estimation uncertainty.
    ``false_positive_rate`` is the decision rate at the zero effect translated
    from the same fits, not this effect's own rate. ``wrong_sign_rate`` is
    undefined, and therefore NaN, when the translated effect is zero.
    """

    duration: int
    effect: float
    scale: str
    probability: float
    monte_carlo_error: float
    n_decided: int
    n_replicates: int
    posterior_mean: float
    posterior_tail_mean: float
    interval_width: float
    bias: float
    coverage: float
    false_positive_rate: float
    wrong_sign_rate: float
    detection: float
    failed_runs: int
    precision_met: bool | None


@dataclass(frozen=True)
class InterpolatedEffect:
    """A minimum detectable effect, or an explicit out-of-range result."""

    effect: float | None
    status: MdeStatus
    lower_effect: float | None
    upper_effect: float | None
    lower_power: float | None
    upper_power: float | None
    convention: str
    reason: str | None = None


@dataclass(frozen=True)
class MinimumDetectableEffect:
    """MDE for one duration.

    Bracket Monte Carlo errors are uncertainty in the estimated power used to
    interpolate. They are not the HDI width of an effect estimate.
    ``minimum_worthwhile_effect`` is the caller's practical threshold. It is
    not this MDE and is not used to move the interpolated point.
    """

    duration: int
    target_power: float
    effect: float | None
    status: MdeStatus
    convention: str
    minimum_worthwhile_effect: float | None
    worthwhile_comparison: str | None
    reason: str | None
    lower_effect: float | None
    upper_effect: float | None
    lower_power: float | None
    upper_power: float | None
    lower_monte_carlo_error: float | None
    upper_monte_carlo_error: float | None


@dataclass(frozen=True)
class StudyRecord:
    """What the study conditioned on, what it generated, and how it seeded."""

    scenario_family: str
    conditional_on: str
    generated: str
    mechanism: str
    sampler_seed: int
    data_seed_scheme: str
    noise_scale: float
    donor_weights: tuple[float, ...]
    ar_coefficient: float | None
    horizon_reuse: bool


@dataclass(frozen=True)
class EffectByDuration:
    """Plot-ready power curve and the MDE at the declared target.

    ``precision_target`` is the caller's maximum acceptable mean HDI width on
    the decision estimand, or ``None`` when no precision target was declared.
    It is not ``criterion.monte_carlo_error_tolerance``.
    ``false_positive_within_tolerance`` compares each duration's zero-effect
    decision rate with ``criterion.false_positive_tolerance``. It is not power.
    When power is withheld, ``decision_rate_not_power`` is the mean decision
    rate at the nonzero effects, or at zero when that was the only effect.
    """

    points: tuple[PowerPoint, ...]
    minimum_detectable_effects: tuple[MinimumDetectableEffect, ...]
    study: StudyRecord
    n_fits: int
    n_replicates: int
    failed_runs: int
    failure_messages: tuple[str, ...]
    power_withheld: bool
    withhold_reason: str | None
    preperiod_rmse_ratio: float
    decision_rate_not_power: float | None
    minimum_worthwhile_effect: float | None
    precision_target: float | None
    false_positive_within_tolerance: bool | None
    decision: DecisionRule
    criterion: PowerCriterion
    estimator: str
    scale: str
    baseline: float | None
    divergences: int
    max_rhat: float | None

    def point(self, duration: int, effect: float) -> PowerPoint:
        """Return the one point for this duration and effect magnitude.

        Parameters
        ----------
        duration : int
            Holdout length of the requested point.
        effect : float
            Translated effect magnitude of the requested point.
        """
        matches = [
            item
            for item in self.points
            if item.duration == duration and item.effect == effect
        ]
        if len(matches) != 1:
            known = ", ".join(
                f"({item.duration}, {item.effect})" for item in self.points
            )
            raise ValueError(
                f"No unique point for duration={duration}, effect={effect}. "
                f"Known points: {known}."
            )
        return matches[0]

    def frame(self) -> pd.DataFrame:
        """Return one row per duration and effect, in evaluated order.

        Constant study fields are repeated so a notebook can plot without
        joining back to this object. ``minimum_worthwhile_effect`` is that
        repeated caller threshold, not the row's ``effect``.
        ``false_positive_within_tolerance`` is the study-level comparison, not
        a per-effect rate.
        """
        rows = [
            {column: getattr(item, column) for column in FRAME_COLUMNS}
            for item in self.points
        ]
        frame = pd.DataFrame(rows, columns=list(FRAME_COLUMNS))
        frame["minimum_worthwhile_effect"] = self.minimum_worthwhile_effect
        frame["scenario_family"] = self.study.scenario_family
        frame["power_withheld"] = self.power_withheld
        frame["baseline"] = self.baseline
        frame["false_positive_within_tolerance"] = self.false_positive_within_tolerance
        return frame


@dataclass(frozen=True)
class _PlanConfiguration:
    treated: str
    controls: tuple[str, ...]
    estimator: str
    sample_kwargs: tuple[tuple[str, Any], ...]
    sampler_seed: int
    outcome_scale: str
    time_column: str | None
    planned_index: tuple[Any, ...]
    model: Any


@dataclass
class _Draw:
    decided: bool
    mean: float
    tail: float
    width: float
    covers: bool
    wrong_sign: bool | None
    divergences: int
    max_rhat: float | None


def replicates_for_tolerance(
    tolerance: float,
    *,
    n_durations: int,
    ceiling: int = FIXED_DESIGN_FIT_CEILING,
) -> int:
    """Return the replicate count whose worst-case Monte Carlo SE meets tolerance.

    The worst-case binomial standard error is ``0.5 / sqrt(n)``. Each duration
    is a separate fit, so the fit count is ``n * n_durations``. This raises
    before any fit when that product exceeds ``ceiling``. It does not shrink
    ``n`` to squeeze under the ceiling.

    Parameters
    ----------
    tolerance : float
        Monte Carlo standard-error tolerance in ``(0, 1]``.
    n_durations : int
        Number of horizons. Each one is its own fit.
    ceiling : int, optional
        Maximum inference fits. Defaults to the fixed-design ceiling.

    Returns
    -------
    int
        Replicates to run for every duration.
    """
    tol = require_probability(tolerance, "monte_carlo_error_tolerance")
    horizons = require_positive_int(n_durations, "n_durations")
    limit = require_positive_int(ceiling, "ceiling")
    needed = max(1, math.ceil((0.5 / tol) ** 2 - 1e-12))
    total = needed * horizons
    if total > limit:
        raise ValueError(
            f"monte_carlo_error_tolerance={tol} needs {needed} replicates "
            f"across {horizons} separate horizons ({total} fits). The "
            f"fixed-design ceiling is {limit}. Replicates were not reduced. "
            "No fit was started."
        )
    return needed


def monte_carlo_error(probability: float, n_replicates: int) -> float:
    """Return the binomial standard error of a decision rate.

    Parameters
    ----------
    probability : float
        Decision rate in ``[0, 1]``.
    n_replicates : int
        Successful replicates in the rate.
    """
    rate = require_finite(probability, "probability")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("probability must be in [0, 1].")
    count = require_positive_int(n_replicates, "n_replicates")
    return float(np.sqrt(rate * (1.0 - rate) / count))


def interpolate_minimum_detectable_effect(
    effects: Sequence[float],
    powers: Sequence[float],
    *,
    target_power: float,
    direction: str,
) -> InterpolatedEffect:
    """Interpolate the MDE. Do not extrapolate past the evaluated grid.

    Parameters
    ----------
    effects : sequence of float
        Evaluated effect magnitudes, including zero.
    powers : sequence of float
        Decision rate at each effect. Same length as ``effects``.
    target_power : float
        Declared decision-rate target.
    direction : {"positive", "negative"}
        Decision direction. Only effects on that side of zero, plus zero,
        enter the search.

    Returns
    -------
    InterpolatedEffect
        ``out_of_range`` when the target is outside the evaluated powers.
        ``null_control_failure`` when the zero effect already meets the target.
        That is not an attained minimum detectable effect.
    """
    target = require_probability(target_power, "target_power")
    if direction not in ("positive", "negative"):
        raise ValueError("direction must be 'positive' or 'negative'.")
    if len(effects) != len(powers):
        raise ValueError("effects and powers must have the same length.")
    pairs: list[tuple[float, float]] = []
    for effect, power in zip(effects, powers, strict=True):
        magnitude = require_finite(effect, "effect")
        rate = require_finite(power, "power")
        if not 0.0 <= rate <= 1.0:
            raise ValueError("power must be in [0, 1].")
        on_side = magnitude >= 0.0 if direction == "positive" else magnitude <= 0.0
        if on_side:
            pairs.append((magnitude, rate))
    pairs.sort(key=lambda item: item[0] if direction == "positive" else -item[0])

    def _miss(reason: str) -> InterpolatedEffect:
        return InterpolatedEffect(
            None,
            "out_of_range",
            None,
            None,
            None,
            None,
            MDE_CONVENTION,
            reason,
        )

    if not pairs:
        return _miss("no effect on the decision side")
    for index, (effect, power) in enumerate(pairs):
        if power < target:
            continue
        if effect == 0.0:
            return InterpolatedEffect(
                None,
                "null_control_failure",
                effect,
                effect,
                power,
                power,
                MDE_CONVENTION,
                "The zero effect already meets the target power. That is a "
                "null-control failure, not a minimum detectable effect.",
            )
        if index == 0 or power == pairs[index - 1][1]:
            return InterpolatedEffect(
                effect, "attained", effect, effect, power, power, MDE_CONVENTION
            )
        previous_effect, previous_power = pairs[index - 1]
        weight = (target - previous_power) / (power - previous_power)
        interpolated = previous_effect + weight * (effect - previous_effect)
        return InterpolatedEffect(
            float(interpolated),
            "attained",
            previous_effect,
            effect,
            previous_power,
            power,
            MDE_CONVENTION,
        )
    return _miss("above the largest evaluated effect")


def minimum_detectable_effect(
    result: EffectByDuration,
    duration: int,
) -> MinimumDetectableEffect:
    """Return the stored MDE for one duration. Does not refit.

    Parameters
    ----------
    result : EffectByDuration
        A completed effect-by-duration study.
    duration : int
        Holdout length present in ``result``.
    """
    matches = [
        item for item in result.minimum_detectable_effects if item.duration == duration
    ]
    if len(matches) != 1:
        raise ValueError(
            f"No minimum detectable effect is stored for duration {duration}."
        )
    return matches[0]


def preperiod_simplex_rmse(donors: np.ndarray, treated: np.ndarray) -> float:
    """Return in-sample RMSE of a nonnegative donor simplex.

    This is a pre-period misfit screen, not a synthetic-control estimator and
    not a replacement for the planned model.

    Parameters
    ----------
    donors : ndarray
        Pre-period donor outcomes, shape ``(n_rows, n_donors)``.
    treated : ndarray
        Pre-period treated outcomes, shape ``(n_rows,)``.
    """
    matrix = np.asarray(donors, dtype=float)
    outcome = np.asarray(treated, dtype=float)
    if matrix.ndim != 2 or outcome.ndim != 1 or matrix.shape[0] != outcome.shape[0]:
        raise ValueError("donors must be (n_rows, n_donors) and treated (n_rows,).")
    if matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("The simplex screen needs at least two rows and one donor.")
    if not np.isfinite(matrix).all() or not np.isfinite(outcome).all():
        raise ValueError("Simplex-screen inputs must be finite.")
    n_donors = matrix.shape[1]
    start = np.full(n_donors, 1.0 / n_donors)

    def objective(weights: np.ndarray) -> float:
        """Return the squared residual of one nonnegative weight vector.

        Parameters
        ----------
        weights : ndarray
            Nonnegative donor weights, not yet constrained to sum to one.
        """
        residual = matrix @ weights - outcome
        return float(residual @ residual)

    fitted = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_donors,
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        tol=1e-12,
    )
    weights = np.asarray(fitted.x, dtype=float) if fitted.success else start
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    weights = start if total <= 0.0 else weights / total
    residual = matrix @ weights - outcome
    return float(np.sqrt(np.mean(residual * residual)))


def _family_code(family: str) -> int:
    if family not in SCENARIO_FAMILIES:
        raise ValueError(
            f"scenario_family must be one of {SCENARIO_FAMILIES}. Got {family!r}. "
            "No fit was started."
        )
    return SCENARIO_FAMILIES.index(family)


def _replicate_rng(
    sampler_seed: int, replicate: int, family: str
) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([sampler_seed, replicate, _family_code(family)])
    )


def _ar1_noise(rng: np.random.Generator, n_rows: int, rho: float) -> np.ndarray:
    shocks = rng.normal(0.0, 1.0, size=n_rows)
    noise = np.empty(n_rows)
    noise[0] = shocks[0]
    innovation = math.sqrt(1.0 - rho * rho)
    for time in range(1, n_rows):
        noise[time] = rho * noise[time - 1] + innovation * shocks[time]
    return noise


def _estimand_mean(summary: EstimandSummary, estimand: str) -> float:
    if estimand == "cumulative_total":
        return summary.cumulative_total_mean
    if estimand == "average_per_period":
        return summary.average_per_period_mean
    if estimand == "average_per_treated_unit":
        return summary.average_per_treated_unit_mean
    raise ValueError(f"Unsupported estimand {estimand!r}.")


def _true_estimand(
    *,
    scale: EffectScale,
    magnitude: float,
    duration: int,
    baseline: float | None,
    estimand: str,
) -> float:
    increment = unit_period_effect(
        scale=scale,
        magnitude=magnitude,
        duration=duration,
        n_treated=1,
        baseline=baseline,
    )
    cumulative = increment * duration
    if estimand == "cumulative_total":
        return float(cumulative)
    if estimand == "average_per_period":
        return float(increment)
    if estimand == "average_per_treated_unit":
        return float(cumulative)
    raise ValueError(f"Unsupported estimand {estimand!r}.")


def _is_wrong_sign(mean: float, truth: float) -> bool | None:
    """Return whether the estimate's sign disagrees with a nonzero truth.

    A zero effect has no sign to miss. A posterior mean that fails the decision
    threshold is not a sign error, so the rate is undefined there.
    """
    if truth > 0.0:
        return mean < 0.0
    if truth < 0.0:
        return mean > 0.0
    return None


def _wrong_sign_rate(flags: Sequence[bool | None]) -> float:
    if not flags:
        raise ValueError("A rate needs at least one successful replicate.")
    if all(flag is None for flag in flags):
        return float("nan")
    if any(flag is None for flag in flags):
        raise RuntimeError(
            "Wrong-sign is undefined for a mix of zero and nonzero effects."
        )
    return _rate([bool(flag) for flag in flags])


def _worthwhile_comparison(mde: float | None, worthwhile: float | None) -> str | None:
    if mde is None or worthwhile is None:
        return None
    if abs(mde) > abs(worthwhile):
        return "mde_larger_than_worthwhile"
    return "mde_at_or_below_worthwhile"


def _rate(flags: Sequence[bool]) -> float:
    if not flags:
        raise ValueError("A rate needs at least one successful replicate.")
    return float(np.mean(np.asarray(flags, dtype=float)))


class ReferencePowerEvaluator:
    """Repeated-experiment power for one closed-over synthetic-control plan.

    Parameters
    ----------
    plan : HistoryOnlyPlan
        Supplies the estimator, sampling settings, and sampler seed.
        Construction of this evaluator does not sample.
    history : pandas.DataFrame
        The frame the plan was built from. Donor columns are the conditional
        part of every replicate. The treated column is replaced.
    scenario_family : str
        Which declared generator to run. Recorded on the result. This is not
        a field of ``PowerRequest``.
    """

    def __init__(
        self,
        plan: HistoryOnlyPlan,
        *,
        history: pd.DataFrame,
        scenario_family: str,
    ) -> None:
        if not isinstance(plan, HistoryOnlyPlan):
            raise TypeError("plan must be a HistoryOnlyPlan. No fit was started.")
        if not isinstance(history, pd.DataFrame):
            raise TypeError("history must be a pandas DataFrame. No fit was started.")
        _family_code(scenario_family)
        self._plan = plan
        self._history = history.copy()
        self._scenario_family = scenario_family
        self._config: _PlanConfiguration | None = None

    @property
    def scenario_family(self) -> str:
        """Declared generator name. This does not sample."""
        return self._scenario_family

    @property
    def noise_scale(self) -> float:
        """Declared noise scale. Reading it does not sample."""
        config = self._configuration()
        return _noise_scale(self._history, config.controls)

    def untreated_panel(self, replicate: int) -> pd.DataFrame:
        """Return one generated untreated panel. Does not fit or inject treatment.

        Parameters
        ----------
        replicate : int
            Zero-based replicate index. It is the data-seed key, not a sampler seed.
        """
        if (
            isinstance(replicate, bool)
            or not isinstance(replicate, int)
            or replicate < 0
        ):
            raise ValueError("replicate must be a non-negative integer.")
        config = self._configuration()
        self._require_same_history(config)
        self._require_units(config)
        return self._panel(replicate, config)

    def repeated_experiment_power(
        self, request: PowerRequest
    ) -> RepeatedExperimentPower:
        """Return the decision rate for one request.

        A persistent pre-period misfit raises :class:`MisfitNotPower` after the
        study. The exception carries the report. It does not return the
        withheld rate as ``probability``.

        Parameters
        ----------
        request : PowerRequest
            Scenario, decision, criterion, estimator, group size, and duration.
            The repeated-data budget is derived from the criterion. The seed
            and scenario family are not read from this object.
        """
        if not isinstance(request, PowerRequest):
            raise TypeError("request must be a PowerRequest. No fit was started.")
        self._reject_request(request)
        report = self.effect_by_duration(
            (request.duration,),
            (request.scenario.magnitude,),
            decision=request.decision,
            criterion=request.criterion,
            scale=request.scenario.scale,
            baseline=request.scenario.baseline,
        )
        if report.power_withheld:
            reason = report.withhold_reason or "Power was withheld."
            raise MisfitNotPower(reason, report)
        point = report.point(request.duration, request.scenario.magnitude)
        return RepeatedExperimentPower(
            probability=point.probability,
            monte_carlo_error=point.monte_carlo_error,
            n_replicates=point.n_replicates,
            qualification_scope=request.criterion.qualification_scope,
            estimator=request.estimator,
            decision=request.decision,
        )

    def effect_by_duration(
        self,
        durations: Sequence[int],
        effects: Sequence[float],
        *,
        decision: DecisionRule,
        criterion: PowerCriterion,
        scale: EffectScale = "absolute",
        baseline: float | None = None,
        minimum_worthwhile_effect: float | None = None,
    ) -> EffectByDuration:
        """Fit each duration once per replicate and translate ``effects``.

        Zero is always translated, including when the caller omitted it, so
        the false-positive rate comes from the same fits as the curve. The
        fit count does not grow with the effect grid.

        Parameters
        ----------
        durations : sequence of int
            Holdout lengths. Each length is a separate fit.
        effects : sequence of float
            Effect magnitudes on ``scale``. A purely translated grid reuses fits.
        decision : DecisionRule
            One-sided rule applied to every translated estimand.
        criterion : PowerCriterion
            Target power, false-positive tolerance, precision target, and
            Monte Carlo tolerance. ``post_search`` is refused before sampling.
        scale : {"absolute", "relative"}, optional
            Scale of ``effects``. Relative effects require ``baseline``.
        baseline : float or None, optional
            Cumulative baseline for a relative effect.
        minimum_worthwhile_effect : float or None, optional
            Caller threshold in the same units as ``effects``. Recorded and
            compared with the MDE. It does not change the MDE.

        Returns
        -------
        EffectByDuration
            Power curve, MDE, and the study record. When the pre-period screen
            fails, power fields are withheld and the decision rate is stored
            as ``decision_rate_not_power``. A replicate that raises is counted
            as one fit per duration so a failed run is not under-reported.
        """
        horizons = _require_durations(durations)
        grid = _require_effects(effects)
        self._reject_controls(
            decision,
            criterion,
            scale,
            baseline,
            minimum_worthwhile_effect,
        )
        config = self._configuration()
        self._require_same_history(config)
        self._require_units(config)
        self._reject_plan(config)
        longest = max(horizons)
        if len(self._history) - longest < MIN_TRAINING_ROWS:
            raise ValueError(
                f"history has {len(self._history)} rows. Duration {longest} leaves "
                f"fewer than {MIN_TRAINING_ROWS} training rows. No fit was started."
            )
        noise_scale = _noise_scale(self._history, config.controls)
        n_replicates = replicates_for_tolerance(
            criterion.monte_carlo_error_tolerance,
            n_durations=len(horizons),
        )
        panels = [self._panel(index, config) for index in range(n_replicates)]
        ratio = _median_rmse_ratio(panels, config, horizons, noise_scale)
        withhold, reason = _screen_decision(ratio)
        if self._scenario_family == "persistent_misfit" and not withhold:
            raise RuntimeError(
                "Declared persistent_misfit was not present: median pre-period "
                f"RMSE / noise_scale was {ratio}, below {MISFIT_RMSE_TO_NOISE}. "
                "Power was not reported. No fit was started."
            )
        translated = _with_zero(grid)
        draws, failures, n_fits = self._fit_replicates(
            panels,
            config,
            horizons,
            translated,
            decision,
            scale,
            baseline,
        )
        if not any(draws.values()):
            detail = failures[0] if failures else "no successful replicate"
            raise RuntimeError(
                f"Every replicate failed ({detail}). Power was not reported."
            )
        return _assemble(
            draws=draws,
            failures=failures,
            n_fits=n_fits,
            horizons=horizons,
            translated=translated,
            decision=decision,
            criterion=criterion,
            scale=scale,
            baseline=baseline,
            worthwhile=minimum_worthwhile_effect,
            withhold=withhold,
            reason=reason,
            ratio=ratio,
            config=config,
            family=self._scenario_family,
            noise_scale=noise_scale,
            n_replicates=n_replicates,
        )

    def _configuration(self) -> _PlanConfiguration:
        if self._config is not None:
            return self._config
        handoff = self._plan.analysis_handoff(duration=1)
        seed = _require_seed(dict(handoff.sample_kwargs).get("random_seed"))
        if len(handoff.treated_units) != 1:
            raise ValueError(
                "Grouped qualification is not this issue. The reference power "
                "path needs one treated unit. No fit was started."
            )
        if handoff.model is None:
            raise RuntimeError(
                "The closed-over plan did not hand off its model. Power must "
                "fit that model, not a default fitter. No fit was started."
            )
        planned = tuple(handoff.training_index) + tuple(handoff.holdout_index)
        self._config = _PlanConfiguration(
            treated=handoff.treated_units[0],
            controls=tuple(handoff.control_units),
            estimator=handoff.estimator,
            sample_kwargs=tuple(handoff.sample_kwargs),
            sampler_seed=seed,
            outcome_scale=handoff.outcome_scale,
            time_column=handoff.time_column,
            planned_index=planned,
            model=handoff.model,
        )
        return self._config

    def _require_same_history(self, config: _PlanConfiguration) -> None:
        observed = _labels(self._history, config.time_column)
        if tuple(observed) != config.planned_index:
            raise ValueError(
                "history must be the frame the plan was built from, in the same "
                "order. A different donor path is a different study. No fit was "
                "started."
            )
        # This issue cannot add a public donor accessor to reference.py.
        prepared = self._plan._history
        observed_donors = _donor_matrix(self._history, config.controls)
        planned_donors = _donor_matrix(prepared, config.controls)
        if observed_donors.shape != planned_donors.shape or not np.array_equal(
            observed_donors, planned_donors
        ):
            raise ValueError(
                "Donor columns do not match the plan's prepared history. "
                "A different donor path is a different study. No fit was started."
            )

    def _require_units(self, config: _PlanConfiguration) -> None:
        required = (config.treated, *config.controls)
        missing = [name for name in required if name not in self._history.columns]
        if missing:
            raise ValueError(f"history is missing units {missing}. No fit was started.")

    def _reject_plan(self, config: _PlanConfiguration) -> None:
        capabilities = self._plan.capabilities
        if (
            capabilities.estimator != "synthetic_control"
            or config.estimator != "synthetic_control"
        ):
            raise ValueError(
                "This power path implements the synthetic_control reference only. "
                f"{config.estimator!r} is the #1440 seam. No fit was started."
            )
        if not capabilities.deterministic_effect_reuse:
            raise ValueError(
                "This power path translates one fit across the effect grid. The "
                "closed-over plan does not support deterministic effect reuse. "
                "No fit was started."
            )
        if capabilities.horizon_reuse:
            raise ValueError(
                "Horizon reuse is off on this power path. No fit was started."
            )

    def _reject_request(self, request: PowerRequest) -> None:
        if request.group_size != 1:
            raise ValueError(
                "Grouped qualification is not this issue. group_size must be 1. "
                "Groups of 2 or 3 are #1440 and #319. No fit was started."
            )
        if request.estimator != "synthetic_control":
            raise ValueError(
                "This power path implements the synthetic_control reference only. "
                f"{request.estimator!r} is the #1440 seam. No fit was started."
            )

    def _reject_controls(
        self,
        decision: DecisionRule,
        criterion: PowerCriterion,
        scale: str,
        baseline: float | None,
        worthwhile: float | None,
    ) -> None:
        if not isinstance(decision, DecisionRule):
            raise TypeError("decision must be a DecisionRule. No fit was started.")
        if not isinstance(criterion, PowerCriterion):
            raise TypeError("criterion must be a PowerCriterion. No fit was started.")
        if criterion.qualification_scope == "post_search":
            raise ValueError(
                "Post-search qualification is owned by #319 and is not authorized "
                "here. No fit was started."
            )
        if scale not in ("absolute", "relative"):
            raise ValueError("scale must be 'absolute' or 'relative'.")
        if scale == "relative" and (
            baseline is None or not np.isfinite(baseline) or baseline <= 0.0
        ):
            raise ValueError(
                "A relative effect requires a finite baseline strictly above zero. "
                "No fit was started."
            )
        if scale == "absolute" and baseline is not None:
            raise ValueError("An absolute effect does not take a baseline.")
        if worthwhile is not None:
            require_finite(worthwhile, "minimum_worthwhile_effect")

    def _panel(self, replicate: int, config: _PlanConfiguration) -> pd.DataFrame:
        donors = _donor_matrix(self._history, config.controls)
        weights = np.full(donors.shape[1], 1.0 / donors.shape[1])
        signal = donors @ weights
        rng = _replicate_rng(config.sampler_seed, replicate, self._scenario_family)
        noise_scale = _noise_scale(self._history, config.controls)
        n_rows = donors.shape[0]
        if self._scenario_family == "persistent_misfit":
            centered = np.arange(n_rows, dtype=float)
            centered = centered - float(centered.mean())
            donor_sd = float(np.std(donors))
            treated = (
                MISFIT_TREND_PER_DONOR_SD * donor_sd * centered
                + noise_scale * rng.normal(0.0, 1.0, size=n_rows)
            )
        else:
            noise: np.ndarray
            if self._scenario_family == "temporal_dependence":
                noise = _ar1_noise(rng, n_rows, TEMPORAL_AR_COEFFICIENT)
            else:
                noise = np.asarray(rng.normal(0.0, 1.0, size=n_rows), dtype=float)
            treated = signal + noise_scale * noise
        panel = self._history.copy()
        panel[config.treated] = pd.Series(treated, index=self._history.index)
        return panel

    def _fit_replicates(
        self,
        panels: Sequence[pd.DataFrame],
        config: _PlanConfiguration,
        horizons: tuple[int, ...],
        translated: tuple[float, ...],
        decision: DecisionRule,
        scale: EffectScale,
        baseline: float | None,
    ) -> tuple[dict[tuple[int, float], list[_Draw]], tuple[str, ...], int]:
        draws: dict[tuple[int, float], list[_Draw]] = {
            (duration, effect): [] for duration in horizons for effect in translated
        }
        failures: list[str] = []
        n_fits = 0
        for panel in panels:
            try:
                replica = HistoryOnlyPlan(
                    panel,
                    config.treated,
                    config.controls,
                    decision,
                    estimator=config.estimator,
                    model=config.model._clone(),
                    time_column=config.time_column,
                    outcome_scale=config.outcome_scale,
                )
                if scale == "absolute":
                    planning = replica.evaluate(horizons, absolute_effects=translated)
                else:
                    planning = replica.evaluate(
                        horizons,
                        relative_effects=translated,
                        relative_baseline=baseline,
                    )
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
                n_fits += len(horizons)
                continue
            if planning.capabilities.horizon_reuse:
                raise RuntimeError(
                    "Horizon reuse is off on this power path. No further fit was started."
                )
            indexes = {item.fit_index for item in planning.evaluations}
            if planning.n_inference_fits != len(horizons) or len(indexes) != len(
                horizons
            ):
                raise RuntimeError(
                    "Each duration must be its own fit. "
                    f"Got {planning.n_inference_fits} fits and indexes {sorted(indexes)} "
                    f"for {len(horizons)} durations."
                )
            n_fits += planning.n_inference_fits
            expected = len(next(iter(draws.values()))) + 1
            for evaluation in planning.evaluations:
                key = (evaluation.duration, evaluation.scenario.magnitude)
                if key not in draws:
                    raise RuntimeError(
                        "Translation returned an effect that was not requested. "
                        f"Got duration {key[0]} and effect {key[1]}."
                    )
                draws[key].append(_draw_from(evaluation, decision, scale, baseline))
            if any(len(sample) != expected for sample in draws.values()):
                raise RuntimeError(
                    "A successful replicate did not return every translated effect."
                )
        return draws, tuple(failures), n_fits


def _require_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            "The closed-over plan has no integer random_seed in sample_kwargs. "
            "Repeated experiments need a recorded seed. No fit was started."
        )
    return int(value)


def _labels(history: pd.DataFrame, time_column: str | None) -> list[Any]:
    if time_column is None:
        return list(history.index)
    if time_column not in history.columns:
        raise ValueError(
            f"history is missing time column {time_column!r}. No fit was started."
        )
    return list(history[time_column])


def _require_durations(durations: Sequence[int]) -> tuple[int, ...]:
    if isinstance(durations, (str, bytes)) or not isinstance(durations, Sequence):
        raise TypeError("durations must be a sequence of positive integers.")
    values = tuple(require_positive_int(item, "duration") for item in durations)
    if not values:
        raise ValueError("durations must not be empty. No fit was started.")
    if len(values) != len(set(values)):
        raise ValueError("durations must be unique. No fit was started.")
    return values


def _require_effects(effects: Sequence[float]) -> tuple[float, ...]:
    if isinstance(effects, (str, bytes)) or not isinstance(effects, Sequence):
        raise TypeError("effects must be a sequence of finite numbers.")
    if len(effects) == 0:
        raise ValueError("effects must not be empty. No fit was started.")
    return tuple(require_finite(item, "effect") for item in effects)


def _with_zero(effects: Sequence[float]) -> tuple[float, ...]:
    ordered: list[float] = []
    for effect in (0.0, *effects):
        if effect not in ordered:
            ordered.append(effect)
    return tuple(ordered)


def _donor_matrix(history: pd.DataFrame, controls: Sequence[str]) -> np.ndarray:
    missing = [name for name in controls if name not in history.columns]
    if missing:
        raise ValueError(f"history is missing units {missing}. No fit was started.")
    try:
        donors = history.loc[:, list(controls)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Donor columns must be finite numbers. No fit was started."
        ) from exc
    if donors.ndim != 2 or not np.isfinite(donors).all():
        raise ValueError("Donor columns must be finite numbers. No fit was started.")
    if float(np.std(donors)) <= 0.0:
        raise ValueError(
            "Observed donors must have positive variation. No fit was started."
        )
    return donors


def _noise_scale(history: pd.DataFrame, controls: Sequence[str]) -> float:
    donors = _donor_matrix(history, controls)
    return float(NOISE_TO_DONOR_SD * np.std(donors))


def _median_rmse_ratio(
    panels: Sequence[pd.DataFrame],
    config: _PlanConfiguration,
    horizons: Sequence[int],
    noise_scale: float,
) -> float:
    pre_rows = len(panels[0]) - min(horizons)
    if pre_rows < MIN_TRAINING_ROWS:
        raise ValueError(
            "The pre-period screen needs at least "
            f"{MIN_TRAINING_ROWS} training rows. No fit was started."
        )
    ratios: list[float] = []
    for panel in panels:
        donors = panel.loc[:, list(config.controls)].to_numpy(dtype=float)[:pre_rows]
        treated = panel[config.treated].to_numpy(dtype=float)[:pre_rows]
        ratios.append(preperiod_simplex_rmse(donors, treated) / noise_scale)
    return float(np.median(ratios))


def _screen_decision(ratio: float) -> tuple[bool, str | None]:
    if ratio > MISFIT_RMSE_TO_NOISE:
        return True, (
            "Pre-period misfit: simplex reconstruction failed "
            f"(median RMSE / noise_scale={ratio:.3f}, threshold "
            f"{MISFIT_RMSE_TO_NOISE}). The decision rate was not labeled as power."
        )
    return False, None


def _draw_from(
    evaluation: Any, decision: DecisionRule, scale: EffectScale, baseline: float | None
) -> _Draw:
    mean = _estimand_mean(evaluation.summary, decision.estimand)
    if not np.isfinite(mean):
        raise RuntimeError(
            "A successful evaluation returned a non-finite estimand mean."
        )
    truth = _true_estimand(
        scale=scale,
        magnitude=evaluation.scenario.magnitude,
        duration=evaluation.duration,
        baseline=baseline,
        estimand=decision.estimand,
    )
    lower = evaluation.decision.hdi_lower
    upper = evaluation.decision.hdi_upper
    rhat = evaluation.diagnostics.max_rhat
    return _Draw(
        decided=bool(evaluation.decision.decided),
        mean=float(mean),
        tail=float(evaluation.decision.posterior_tail_probability),
        width=float(upper - lower),
        covers=bool(lower <= truth <= upper),
        wrong_sign=_is_wrong_sign(float(mean), truth),
        divergences=int(evaluation.diagnostics.divergences or 0),
        max_rhat=None if rhat is None else float(rhat),
    )


def _assemble(
    *,
    draws: dict[tuple[int, float], list[_Draw]],
    failures: tuple[str, ...],
    n_fits: int,
    horizons: tuple[int, ...],
    translated: tuple[float, ...],
    decision: DecisionRule,
    criterion: PowerCriterion,
    scale: EffectScale,
    baseline: float | None,
    worthwhile: float | None,
    withhold: bool,
    reason: str | None,
    ratio: float,
    config: _PlanConfiguration,
    family: str,
    noise_scale: float,
    n_replicates: int,
) -> EffectByDuration:
    points: list[PowerPoint] = []
    withheld_rates: list[float] = []
    null_rates: list[float] = []
    nonzero_rates: list[float] = []
    rhats: list[float] = []
    divergences = 0
    for duration in horizons:
        null = draws[(duration, 0.0)]
        false_positive = _rate([item.decided for item in null])
        for effect in translated:
            sample = draws[(duration, effect)]
            decided = [item.decided for item in sample]
            rate = _rate(decided)
            if effect == 0.0:
                withheld_rates.append(rate)
                null_rates.append(rate)
            else:
                nonzero_rates.append(rate)
            probability = float("nan") if withhold else rate
            detection = float("nan") if withhold else rate
            false_positive_rate = float("nan") if withhold else false_positive
            error = float("nan") if withhold else monte_carlo_error(rate, len(sample))
            width = float(np.mean([item.width for item in sample]))
            precision_met = None
            if criterion.precision_target is not None and not withhold:
                precision_met = width <= criterion.precision_target
            points.append(
                PowerPoint(
                    duration=duration,
                    effect=effect,
                    scale=scale,
                    probability=probability,
                    monte_carlo_error=error,
                    n_decided=sum(decided),
                    n_replicates=len(sample),
                    posterior_mean=float(np.mean([item.mean for item in sample])),
                    posterior_tail_mean=float(np.mean([item.tail for item in sample])),
                    interval_width=width,
                    bias=float(
                        np.mean(
                            [
                                item.mean
                                - _true_estimand(
                                    scale=scale,
                                    magnitude=effect,
                                    duration=duration,
                                    baseline=baseline,
                                    estimand=decision.estimand,
                                )
                                for item in sample
                            ]
                        )
                    ),
                    coverage=_rate([item.covers for item in sample]),
                    false_positive_rate=false_positive_rate,
                    wrong_sign_rate=_wrong_sign_rate(
                        [item.wrong_sign for item in sample]
                    ),
                    detection=detection,
                    failed_runs=len(failures),
                    precision_met=precision_met,
                )
            )
            if effect == 0.0:
                divergences += sum(item.divergences for item in sample)
                rhats.extend(
                    item.max_rhat for item in sample if item.max_rhat is not None
                )
    mdes = tuple(
        _mde_for(points, duration, decision, criterion, worthwhile, withhold)
        for duration in horizons
    )
    n_controls = len(config.controls)
    weights = tuple([1.0 / n_controls] * n_controls)
    study = StudyRecord(
        scenario_family=family,
        conditional_on=CONDITIONAL_ON,
        generated=_GENERATED[family],
        mechanism=MECHANISM,
        sampler_seed=config.sampler_seed,
        data_seed_scheme=DATA_SEED_SCHEME,
        noise_scale=noise_scale,
        donor_weights=weights,
        ar_coefficient=TEMPORAL_AR_COEFFICIENT
        if family == "temporal_dependence"
        else None,
        horizon_reuse=False,
    )
    chosen = nonzero_rates or withheld_rates
    decision_rate = float(np.mean(chosen)) if withhold and chosen else None
    within_tolerance = (
        all(item <= criterion.false_positive_tolerance for item in null_rates)
        if null_rates
        else None
    )
    return EffectByDuration(
        points=tuple(points),
        minimum_detectable_effects=mdes,
        study=study,
        n_fits=n_fits,
        n_replicates=n_replicates,
        failed_runs=len(failures),
        failure_messages=failures,
        power_withheld=withhold,
        withhold_reason=reason,
        preperiod_rmse_ratio=ratio,
        decision_rate_not_power=decision_rate,
        minimum_worthwhile_effect=worthwhile,
        precision_target=criterion.precision_target,
        false_positive_within_tolerance=within_tolerance,
        decision=decision,
        criterion=criterion,
        estimator=config.estimator,
        scale=scale,
        baseline=baseline,
        divergences=divergences,
        max_rhat=max(rhats) if rhats else None,
    )


def _mde_for(
    points: Sequence[PowerPoint],
    duration: int,
    decision: DecisionRule,
    criterion: PowerCriterion,
    worthwhile: float | None,
    withhold: bool,
) -> MinimumDetectableEffect:
    selected = [item for item in points if item.duration == duration]
    if withhold or any(not np.isfinite(item.probability) for item in selected):
        interpolated = InterpolatedEffect(
            None,
            "out_of_range",
            None,
            None,
            None,
            None,
            MDE_CONVENTION,
            "power withheld",
        )
    else:
        interpolated = interpolate_minimum_detectable_effect(
            [item.effect for item in selected],
            [item.probability for item in selected],
            target_power=criterion.target_power,
            direction=decision.direction,
        )
    errors = {item.effect: item.monte_carlo_error for item in selected}
    return MinimumDetectableEffect(
        duration=duration,
        target_power=criterion.target_power,
        effect=interpolated.effect,
        status=interpolated.status,
        convention=interpolated.convention,
        minimum_worthwhile_effect=worthwhile,
        worthwhile_comparison=_worthwhile_comparison(interpolated.effect, worthwhile),
        reason=interpolated.reason,
        lower_effect=interpolated.lower_effect,
        upper_effect=interpolated.upper_effect,
        lower_power=interpolated.lower_power,
        upper_power=interpolated.upper_power,
        lower_monte_carlo_error=errors.get(interpolated.lower_effect)
        if interpolated.lower_effect in errors
        else None,
        upper_monte_carlo_error=errors.get(interpolated.upper_effect)
        if interpolated.upper_effect in errors
        else None,
    )
