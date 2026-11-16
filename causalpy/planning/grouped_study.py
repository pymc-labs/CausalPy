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
"""Grouped Monte Carlo study used by treatment search.

Construction samples. Not a public entry point. ``search`` owns screening,
selection, and validation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from causalpy.planning.contracts import (
    CanonicalEvaluation,
    DecisionRule,
    PowerCriterion,
    unit_period_effect,
)
from causalpy.planning.power import (
    CONDITIONAL_ON,
    MDE_CONVENTION,
    MISFIT_RMSE_TO_NOISE,
    MISFIT_TREND_PER_DONOR_SD,
    NOISE_TO_DONOR_SD,
    SCENARIO_FAMILIES,
    TEMPORAL_AR_COEFFICIENT,
    EffectByDuration,
    MinimumDetectableEffect,
    PowerPoint,
    StudyRecord,
    interpolate_minimum_detectable_effect,
    monte_carlo_error,
    preperiod_simplex_rmse,
    replicates_for_tolerance,
)
from causalpy.planning.reference import HistoryOnlyPlan

if TYPE_CHECKING:
    from causalpy.planning.search import _CandidateEvaluation

GROUPED_DATA_SEED_SCHEME = (
    "numpy.random.SeedSequence([sampler_seed, replicate_index, family_code])"
    ".spawn(1 + n_treated). Child 0 is the common shock. Later children are "
    "unit shocks in sorted treated-name order. family_code is the index in "
    "SCENARIO_FAMILIES. This is not the size-one engine stream."
)
GROUPED_MECHANISM = (
    "Each replicate replaces every treated column and leaves donor columns "
    "at their observed values. HistoryOnlyPlan.evaluate fits the selected "
    "adapter once per duration and translates the effect grid without "
    "refitting. Treatment is not in the series that is fit. The decision "
    "rate is the share of replicates whose AnalysisDecision.decided is true. "
    "ReferencePowerEvaluator rejects groups, so this extension calls the "
    "plan and the published aggregation helpers instead. It does not rank "
    "candidates and it does not refit the power engine's private assembly."
)
COMMON_SHOCK_SHARE = 0.5


@dataclass(frozen=True)
class _DiagnosticRecord:
    """Grouped sampling diagnostics for one search row.

    ``divergences`` is null when any replicate count is missing. It is not
    the power curve's required integer.
    """

    divergences: int | None
    max_rhat: float | None
    min_bulk_ess: float | None
    status: str | None
    reason: str | None


def _grouped_study(
    frame: pd.DataFrame,
    treated: tuple[str, ...],
    donors: tuple[str, ...],
    horizons: tuple[int, ...],
    effects: tuple[float, ...],
    decision: DecisionRule,
    criterion: PowerCriterion,
    worthwhile: float,
    estimator: str,
    family: str,
    scale: str,
    baseline: float | None,
    sample_kwargs: Mapping[str, object],
    outcome_scale: str,
    time_column: str | None,
    selection_end: object,
    seed: int,
    fit_ceiling: int,
) -> _CandidateEvaluation:
    # search imports this module. Import the orchestrator types only when a study runs.
    from causalpy.planning.search import (
        _CandidateEvaluation,
        _finished_evaluation,
    )

    try:
        replicates = replicates_for_tolerance(
            criterion.monte_carlo_error_tolerance,
            n_durations=len(horizons),
            ceiling=fit_ceiling,
        )
        translated = _with_zero(effects)
        noise = _noise_scale(frame, donors, selection_end)
        donor_sd = _donor_sd(frame, donors, selection_end)
        scales = {
            name: _unit_scale(frame, name, selection_end, donor_sd) for name in treated
        }
        unit_scale = max(scales.values())
        panels = [
            _grouped_panel(
                frame,
                treated,
                donors,
                family,
                seed,
                replicate,
                noise,
                selection_end,
            )
            for replicate in range(replicates)
        ]
        ratio_scales = (
            dict.fromkeys(treated, 1.0) if family == "persistent_misfit" else scales
        )
        ratio = _grouped_rmse_ratio(
            panels, treated, donors, horizons, noise, ratio_scales
        )
    except ValueError as exc:
        return _CandidateEvaluation(None, "rejected", str(exc), 0)
    if family == "persistent_misfit" and ratio <= MISFIT_RMSE_TO_NOISE:
        return _CandidateEvaluation(
            None,
            "rejected",
            (
                "Declared persistent_misfit was not present: median pre-period "
                f"RMSE / noise_scale was {ratio}, below {MISFIT_RMSE_TO_NOISE}. "
                "Power was not reported. No fit was started."
            ),
            0,
        )
    draws: dict[tuple[int, float], list[CanonicalEvaluation]] = {
        (duration, effect): [] for duration in horizons for effect in translated
    }
    failures: list[str] = []
    n_fits = 0
    for panel in panels:
        try:
            plan = HistoryOnlyPlan(
                panel,
                treated,
                donors,
                decision,
                estimator=estimator,
                sample_kwargs=sample_kwargs,
                time_column=time_column,
                outcome_scale=outcome_scale,
            )
            if scale == "absolute":
                planning = plan.evaluate(horizons, absolute_effects=translated)
            else:
                planning = plan.evaluate(
                    horizons,
                    relative_effects=translated,
                    relative_baseline=baseline,
                )
        except ValueError as exc:
            message = str(exc)
            failures.append(f"ValueError: {message}")
            if "No fit was started" not in message:
                n_fits += len(horizons)
            continue
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
            n_fits += len(horizons)
            continue
        n_fits += int(planning.n_inference_fits)
        for evaluation in planning.evaluations:
            key = (evaluation.duration, evaluation.scenario.magnitude)
            if key not in draws:
                failures.append(
                    "Translation returned an effect that was not requested."
                )
                continue
            draws[key].append(evaluation)
    if not any(draws.values()):
        detail = failures[0] if failures else "no successful replicate"
        if n_fits == 0:
            return _CandidateEvaluation(None, "rejected", detail, 0)
        return _CandidateEvaluation(
            None,
            "failed",
            f"Every replicate failed ({detail}).",
            n_fits,
            failed_runs=len(failures) or 1,
        )
    shortest = min(len(sample) for sample in draws.values())
    if shortest < replicates:
        partial = [item for sample in draws.values() for item in sample]
        return _CandidateEvaluation(
            None,
            "failed",
            (
                f"Only {shortest} of {replicates} replicates completed. "
                "The declared tolerance sample was not reached. No design "
                "was frozen from this study."
            ),
            n_fits,
            diagnostics=_summarize_diagnostics(partial),
            failed_runs=max(len(failures), replicates - shortest),
        )
    study, diagnostics = _assemble_grouped(
        draws=draws,
        failures=tuple(failures),
        n_fits=n_fits,
        horizons=horizons,
        translated=translated,
        decision=decision,
        criterion=criterion,
        scale=scale,
        baseline=baseline,
        worthwhile=worthwhile,
        ratio=ratio,
        family=family,
        noise_scale=noise,
        n_replicates=replicates,
        estimator=estimator,
        donors=donors,
        treated=treated,
        seed=seed,
        frame=frame,
        selection_end=selection_end,
    )
    return _finished_evaluation(
        study,
        n_fits,
        unit_scale=unit_scale,
        diagnostics=diagnostics,
    )


def _grouped_panel(
    history: pd.DataFrame,
    treated: tuple[str, ...],
    donors: tuple[str, ...],
    family: str,
    sampler_seed_value: int,
    replicate: int,
    noise_scale: float,
    selection_end: object,
) -> pd.DataFrame:
    """Replace treated columns. Donor columns stay observed."""
    children = np.random.SeedSequence(
        [sampler_seed_value, replicate, SCENARIO_FAMILIES.index(family)]
    ).spawn(1 + len(treated))
    panel = history.copy()
    donor_matrix = history.loc[:, list(donors)].to_numpy(dtype=float)
    n_rows = donor_matrix.shape[0]
    weights = np.full(len(donors), 1.0 / len(donors))
    combo = donor_matrix @ weights
    rho = TEMPORAL_AR_COEFFICIENT if family == "temporal_dependence" else 0.0
    common = _ar1_noise(np.random.default_rng(children[0]), n_rows, rho)
    donor_sd = _donor_sd(history, donors, selection_end)
    share = math.sqrt(COMMON_SHOCK_SHARE)
    for index, name in enumerate(sorted(treated)):
        rng = np.random.default_rng(children[index + 1])
        scale = _unit_scale(history, name, selection_end, donor_sd)
        if family == "persistent_misfit":
            centered = np.arange(n_rows, dtype=float)
            centered = centered - float(centered.mean())
            series = (
                MISFIT_TREND_PER_DONOR_SD * donor_sd * centered
                + noise_scale * rng.normal(0.0, 1.0, size=n_rows)
            )
        else:
            unit = _ar1_noise(rng, n_rows, rho)
            noise = share * common + math.sqrt(1.0 - COMMON_SHOCK_SHARE) * unit
            series = combo + noise_scale * scale * noise
        panel[name] = pd.Series(series, index=history.index)
    return panel


def _summarize_diagnostics(
    items: Sequence[CanonicalEvaluation],
) -> _DiagnosticRecord:
    """Record adapter diagnostics without turning a missing count into zero."""
    if not items:
        return _DiagnosticRecord(None, None, None, None, None)
    counts: list[int | None] = []
    rhats: list[float] = []
    ess_values: list[float] = []
    statuses: list[str] = []
    reasons: list[str] = []
    ess_complete = True
    for item in items:
        diagnostics = item.diagnostics
        raw_count = diagnostics.divergences
        count = (
            raw_count
            if isinstance(raw_count, int) and not isinstance(raw_count, bool)
            else None
        )
        counts.append(count)
        raw_status = diagnostics.diagnostics_status
        if raw_status in {"available", "partial", "unavailable"}:
            statuses.append(str(raw_status))
        elif count is None:
            statuses.append("unavailable")
        else:
            statuses.append("available")
        raw_reason = diagnostics.diagnostics_reason
        if raw_reason:
            reasons.append(str(raw_reason))
        raw_rhat = diagnostics.max_rhat
        if isinstance(raw_rhat, int | float) and not isinstance(raw_rhat, bool):
            rhat = float(raw_rhat)
            if np.isfinite(rhat):
                rhats.append(rhat)
        raw_ess = diagnostics.min_bulk_ess
        if (
            isinstance(raw_ess, int | float)
            and not isinstance(raw_ess, bool)
            and np.isfinite(float(raw_ess))
        ):
            ess_values.append(float(raw_ess))
        else:
            ess_complete = False
    total = 0
    missing_count = False
    for count in counts:
        if count is None:
            missing_count = True
            break
        total += count
    rank = {"available": 0, "partial": 1, "unavailable": 2}
    status = (
        "unavailable" if missing_count else max(statuses, key=lambda name: rank[name])
    )
    return _DiagnosticRecord(
        divergences=None if missing_count else total,
        max_rhat=max(rhats) if rhats else None,
        min_bulk_ess=min(ess_values) if ess_complete and ess_values else None,
        status=status,
        reason="; ".join(dict.fromkeys(reasons)) or None,
    )


def _assemble_grouped(
    *,
    draws: dict[tuple[int, float], list[CanonicalEvaluation]],
    failures: tuple[str, ...],
    n_fits: int,
    horizons: tuple[int, ...],
    translated: tuple[float, ...],
    decision: DecisionRule,
    criterion: PowerCriterion,
    scale: str,
    baseline: float | None,
    worthwhile: float,
    ratio: float,
    family: str,
    noise_scale: float,
    n_replicates: int,
    estimator: str,
    donors: tuple[str, ...],
    treated: tuple[str, ...],
    seed: int,
    frame: pd.DataFrame,
    selection_end: object,
) -> tuple[EffectByDuration, _DiagnosticRecord]:
    withhold = ratio > MISFIT_RMSE_TO_NOISE
    reason = (
        (
            "Pre-period misfit: simplex reconstruction failed "
            f"(median RMSE / injected noise sd={ratio:.3f}, threshold "
            f"{MISFIT_RMSE_TO_NOISE}). The decision rate was not labeled as power."
        )
        if withhold
        else None
    )
    points: list[PowerPoint] = []
    null_rates: list[float] = []
    nonzero_rates: list[float] = []
    withheld_rates: list[float] = []
    diagnostic_items: list[CanonicalEvaluation] = []
    n_treated = len(treated)
    for duration in horizons:
        null = draws[(duration, 0.0)]
        false_positive = _rate([bool(item.decision.decided) for item in null])
        for effect in translated:
            sample = draws[(duration, effect)]
            decided = [bool(item.decision.decided) for item in sample]
            rate = _rate(decided)
            if effect == 0.0:
                null_rates.append(rate)
                withheld_rates.append(rate)
            else:
                nonzero_rates.append(rate)
            means = [_estimand_mean(item.summary, decision.estimand) for item in sample]
            widths = [
                float(item.decision.hdi_upper - item.decision.hdi_lower)
                for item in sample
            ]
            truth = _grouped_truth(
                scale, effect, duration, baseline, decision.estimand, n_treated
            )
            covers = [
                bool(item.decision.hdi_lower <= truth <= item.decision.hdi_upper)
                for item in sample
            ]
            width = float(np.mean(widths))
            precision_met = None
            if criterion.precision_target is not None and not withhold:
                precision_met = width <= criterion.precision_target
            points.append(
                PowerPoint(
                    duration=duration,
                    effect=effect,
                    scale=scale,
                    probability=float("nan") if withhold else rate,
                    monte_carlo_error=float("nan")
                    if withhold
                    else monte_carlo_error(rate, len(sample)),
                    n_decided=sum(decided),
                    n_replicates=len(sample),
                    posterior_mean=float(np.mean(means)),
                    posterior_tail_mean=float(
                        np.mean(
                            [
                                item.decision.posterior_tail_probability
                                for item in sample
                            ]
                        )
                    ),
                    interval_width=width,
                    bias=float(np.mean([mean - truth for mean in means])),
                    coverage=_rate(covers),
                    false_positive_rate=float("nan") if withhold else false_positive,
                    wrong_sign_rate=_wrong_sign_rate(
                        [_wrong_sign(mean, truth) for mean in means]
                    ),
                    detection=float("nan") if withhold else rate,
                    failed_runs=len(failures),
                    precision_met=precision_met,
                )
            )
            if effect == 0.0:
                diagnostic_items.extend(sample)
    mdes = tuple(
        _mde_for(points, duration, decision, criterion, worthwhile, withhold)
        for duration in horizons
    )
    donor_sd = _donor_sd(frame, donors, selection_end)
    scales = tuple(
        _unit_scale(frame, name, selection_end, donor_sd) for name in sorted(treated)
    )
    study = StudyRecord(
        scenario_family=family,
        conditional_on=CONDITIONAL_ON,
        generated=_grouped_generated(family, scales),
        mechanism=GROUPED_MECHANISM,
        sampler_seed=seed,
        data_seed_scheme=GROUPED_DATA_SEED_SCHEME,
        noise_scale=noise_scale,
        donor_weights=tuple(1.0 / len(donors) for _ in donors)
        if family != "persistent_misfit"
        else (),
        ar_coefficient=TEMPORAL_AR_COEFFICIENT
        if family == "temporal_dependence"
        else None,
        horizon_reuse=False,
    )
    chosen = nonzero_rates or withheld_rates
    within = (
        all(item <= criterion.false_positive_tolerance for item in null_rates)
        if null_rates and not withhold
        else None
    )
    record = _summarize_diagnostics(diagnostic_items)
    return (
        EffectByDuration(
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
            decision_rate_not_power=float(np.mean(chosen))
            if withhold and chosen
            else None,
            minimum_worthwhile_effect=worthwhile,
            precision_target=criterion.precision_target,
            false_positive_within_tolerance=within,
            decision=decision,
            criterion=criterion,
            estimator=estimator,
            scale=scale,
            baseline=baseline,
            divergences=record.divergences,
            max_rhat=record.max_rhat,
        ),
        record,
    )


def _grouped_truth(
    scale: str,
    magnitude: float,
    duration: int,
    baseline: float | None,
    estimand: str,
    n_treated: int,
) -> float:
    increment = unit_period_effect(
        scale=scale,  # type: ignore[arg-type]
        magnitude=magnitude,
        duration=duration,
        n_treated=n_treated,
        baseline=baseline,
    )
    if estimand == "cumulative_total":
        return float(increment * duration * n_treated)
    if estimand == "average_per_period":
        return float(increment * n_treated)
    if estimand == "average_per_treated_unit":
        return float(increment * duration)
    raise ValueError(f"Unsupported estimand {estimand!r}.")


def _estimand_mean(summary: object, estimand: str) -> float:
    name = f"{estimand}_mean"
    value = float(getattr(summary, name))
    if not np.isfinite(value):
        raise RuntimeError(
            "A successful evaluation returned a non-finite estimand mean."
        )
    return value


def _mde_for(
    points: Sequence[PowerPoint],
    duration: int,
    decision: DecisionRule,
    criterion: PowerCriterion,
    worthwhile: float,
    withhold: bool,
) -> MinimumDetectableEffect:
    selected = [item for item in points if item.duration == duration]
    if withhold or any(not np.isfinite(item.probability) for item in selected):
        effect = None
        status = "out_of_range"
        lower = upper = lower_power = upper_power = None
        reason: str | None = "power withheld"
        convention = MDE_CONVENTION
    else:
        interpolated = interpolate_minimum_detectable_effect(
            [item.effect for item in selected],
            [item.probability for item in selected],
            target_power=criterion.target_power,
            direction=decision.direction,
        )
        effect = interpolated.effect
        status = interpolated.status
        lower = interpolated.lower_effect
        upper = interpolated.upper_effect
        lower_power = interpolated.lower_power
        upper_power = interpolated.upper_power
        reason = interpolated.reason
        convention = interpolated.convention
    errors = {item.effect: item.monte_carlo_error for item in selected}
    comparison = None
    if effect is not None:
        comparison = (
            "mde_larger_than_worthwhile"
            if abs(effect) > abs(worthwhile)
            else "mde_at_or_below_worthwhile"
        )
    return MinimumDetectableEffect(
        duration=duration,
        target_power=criterion.target_power,
        effect=effect,
        status=status,  # type: ignore[arg-type]
        convention=convention,
        minimum_worthwhile_effect=worthwhile,
        worthwhile_comparison=comparison,
        reason=reason,
        lower_effect=lower,
        upper_effect=upper,
        lower_power=lower_power,
        upper_power=upper_power,
        lower_monte_carlo_error=errors.get(lower) if lower in errors else None,
        upper_monte_carlo_error=errors.get(upper) if upper in errors else None,
    )


def _grouped_rmse_ratio(
    panels: Sequence[pd.DataFrame],
    treated: tuple[str, ...],
    donors: tuple[str, ...],
    horizons: tuple[int, ...],
    noise_scale: float,
    scales: Mapping[str, float],
) -> float:
    pre_rows = len(panels[0]) - min(horizons)
    ratios: list[float] = []
    for panel in panels:
        donor_matrix = panel.loc[:, list(donors)].to_numpy(dtype=float)[:pre_rows]
        for name in treated:
            outcome = panel[name].to_numpy(dtype=float)[:pre_rows]
            injected = noise_scale * float(scales[name])
            ratios.append(preperiod_simplex_rmse(donor_matrix, outcome) / injected)
    return float(np.median(ratios))


def _grouped_generated(family: str, scales: tuple[float, ...]) -> str:
    noise = (
        "persistent_misfit noise is noise_scale, not the unit scale. "
        if family == "persistent_misfit"
        else f"Unit noise scales relative to donor sd are {scales}. "
    )
    return (
        f"{family}: each treated column is replaced. Donors stay observed. "
        f"{noise}Common-shock share is {COMMON_SHOCK_SHARE}. "
        "persistent_misfit uses the published trend instead of the donor "
        "combination. Treatment is injected by translation after the untreated "
        "series exists."
    )


def _unit_scale(
    history: pd.DataFrame, name: str, selection_end: object, donor_sd: float
) -> float:
    """Return this unit's pre-cutoff scale relative to the donor standard deviation.

    Values below one stay below one. A non-positive scale cannot be simulated
    at the unit's own size, so it is a zero-fit refusal.
    """
    labels = list(history.index)
    end = labels.index(selection_end)
    values = history.iloc[: end + 1][name].to_numpy(dtype=float)
    std = float(np.std(values))
    if (
        not np.isfinite(std)
        or std <= 0.0
        or not np.isfinite(donor_sd)
        or donor_sd <= 0.0
    ):
        raise ValueError(
            "A treated unit must have positive finite pre-cutoff variation. "
            "No fit was started."
        )
    return std / donor_sd


def _rows_through(history: pd.DataFrame, selection_end: object) -> pd.DataFrame:
    """Return rows at or before ``selection_end``. The holdout stays out."""
    labels = list(history.index)
    return history.iloc[: labels.index(selection_end) + 1]


def _donor_sd(
    history: pd.DataFrame, donors: Sequence[str], selection_end: object
) -> float:
    matrix = (
        _rows_through(history, selection_end).loc[:, list(donors)].to_numpy(dtype=float)
    )
    return float(np.std(matrix))


def _noise_scale(
    history: pd.DataFrame, donors: Sequence[str], selection_end: object
) -> float:
    matrix = (
        _rows_through(history, selection_end).loc[:, list(donors)].to_numpy(dtype=float)
    )
    if (
        matrix.ndim != 2
        or not np.isfinite(matrix).all()
        or float(np.std(matrix)) <= 0.0
    ):
        raise ValueError(
            "Observed donors must have positive finite variation. No fit was started."
        )
    return float(NOISE_TO_DONOR_SD * np.std(matrix))


def _ar1_noise(rng: np.random.Generator, n_rows: int, rho: float) -> np.ndarray:
    shocks = rng.normal(0.0, 1.0, size=n_rows)
    if rho == 0.0:
        return np.asarray(shocks, dtype=float)
    noise = np.empty(n_rows)
    noise[0] = shocks[0]
    innovation = math.sqrt(1.0 - rho * rho)
    for step in range(1, n_rows):
        noise[step] = rho * noise[step - 1] + innovation * shocks[step]
    return noise


def _with_zero(effects: Sequence[float]) -> tuple[float, ...]:
    ordered: list[float] = []
    for effect in (0.0, *effects):
        if effect not in ordered:
            ordered.append(float(effect))
    return tuple(ordered)


def _rate(flags: Sequence[bool]) -> float:
    if not flags:
        raise ValueError("A rate needs at least one successful replicate.")
    return float(np.mean(np.asarray(flags, dtype=float)))


def _wrong_sign(mean: float, truth: float) -> bool | None:
    if truth > 0.0:
        return mean < 0.0
    if truth < 0.0:
        return mean > 0.0
    return None


def _wrong_sign_rate(flags: Sequence[bool | None]) -> float:
    if not flags or all(flag is None for flag in flags):
        return float("nan")
    if any(flag is None for flag in flags):
        raise RuntimeError(
            "Wrong-sign is undefined for a mix of zero and nonzero effects."
        )
    return _rate([bool(flag) for flag in flags])
