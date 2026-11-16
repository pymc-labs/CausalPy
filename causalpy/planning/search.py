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
"""Grouped treatment search over the published planning surface.

Screening is :class:`causalpy.planning.candidates.CandidateGenerator`.
Size-one synthetic control is
:class:`causalpy.planning.power.ReferencePowerEvaluator`. Other supported
designs are translated :class:`causalpy.planning.reference.HistoryOnlyPlan`
fits. The generator's proxy is recorded and is not a power claim. There is
no second ranking implementation.

The declared selection objective is lexicographic, not a weighted score:
among rows that meet the caller's power, false-positive, and precision
criteria, choose the minimum finite cost, then higher power, then the
smaller ``candidate_id``, then the shorter duration. A shortlist, a hash
sample, or a fit budget is not a global optimum. If no row qualifies, the
result is inconclusive and no design is frozen. A shortlist stopped by the
fit cap says so, rather than claiming those rows failed the power criteria.
``n_fits`` is repeated on every duration row of one study.
``SearchResult.n_inference_fits`` is the study total, so summing the column double-counts a multi-duration search.

Size-one diagnostics stay on the reference engine's integer field, which cannot represent a missing count. A size-one study that returns fewer successful replicates than the declared tolerance is failed and is not frozen. A post-fit reference failure charges the reserved replicate count. An inconclusive summary quotes the recorded row reasons unless every refusal is an estimator capability refusal. A rejected identity was attempted, so a fully attempted shortlist is not labeled capped. A grouped curve whose divergence count is missing is not attached to ``SearchResult.studies``, because that integer cannot hold null.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from causalpy.planning.adapters import (
    AdapterAnalysisHandoff,
)
from causalpy.planning.adapters import (
    analysis_handoff as adapter_analysis_handoff,
)
from causalpy.planning.candidates import CandidateGenerator
from causalpy.planning.contracts import (
    MIN_TRAINING_ROWS,
    BalanceConstraint,
    CandidateRequest,
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
    MisfitNotPower,
    PowerPoint,
    ReferencePowerEvaluator,
    StudyRecord,
    interpolate_minimum_detectable_effect,
    monte_carlo_error,
    preperiod_simplex_rmse,
    replicates_for_tolerance,
)
from causalpy.planning.reference import AnalysisHandoff, HistoryOnlyPlan

SELECTION_OBJECTIVE = (
    "Among completed designs that meet the declared power, false-positive, "
    "and precision criteria, choose the minimum finite cost. Ties break by "
    "higher power at the worthwhile effect, then lexicographically smaller "
    "candidate_id, then shorter duration. This is not a weighted score. A "
    "capped shortlist, a partial enumeration, or a fit budget is not a "
    "global optimum. If no design qualifies, freeze nothing."
)
EVALUATION_ORDER = (
    "Shortlist membership is the generator's rank. Power studies run in "
    "candidate_id order. The sampler seed is keyed to the caller seed, "
    "candidate_id, and durations, not to the loop index."
)
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
SEARCH_COLUMNS = (
    "candidate_id",
    "treated",
    "donors",
    "n_donors",
    "group_size",
    "duration",
    "estimator",
    "cost",
    "outcome_share",
    "proxy_score",
    "proxy_role",
    "shortlist_rank",
    "boundary",
    "screen_status",
    "evaluation_status",
    "reason",
    "power",
    "power_withheld",
    "false_positive_rate",
    "precision_met",
    "interval_width",
    "monte_carlo_error",
    "mde",
    "mde_status",
    "n_fits",
    "failed_runs",
    "divergences",
    "max_rhat",
    "min_bulk_ess",
    "diagnostics_status",
    "diagnostics_reason",
    "unit_scale",
    "eligible",
    "pareto",
    "selected",
)


@dataclass(frozen=True)
class ComputationalBudget:
    """Fit cap for one search. It does not shrink a study that has started.

    ``max_inference_fits`` counts inference fits only. Screening and
    translation are free. A candidate whose planned fits would pass the cap
    is skipped and recorded. A refusal that starts no fit is rejected and
    is not a budget skip. Replicates are never reduced to squeeze under
    the cap.
    """

    max_inference_fits: int

    def __post_init__(self) -> None:
        if isinstance(self.max_inference_fits, bool) or not isinstance(
            self.max_inference_fits, int
        ):
            raise TypeError("max_inference_fits must be a non-negative integer.")
        if self.max_inference_fits < 0:
            raise ValueError("max_inference_fits must be non-negative.")


@dataclass(frozen=True)
class FrozenDesign:
    """One explicit selection. Construction does not sample.

    ``analysis_handoff`` rebuilds the published spec for this identity.
    ``inject`` adds the planned effect only on the frozen holdout. ``audit``
    fits this identity on a later window and refuses a window that overlaps
    the planning holdout. None of the three consults another candidate.
    """

    candidate_id: str
    treated_units: tuple[str, ...]
    control_units: tuple[str, ...]
    duration: int
    estimator: str
    decision: DecisionRule
    sample_kwargs: tuple[tuple[str, object], ...]
    sampler_seed: int
    outcome_scale: str
    time_column: str | None
    history: pd.DataFrame
    training_index: tuple[object, ...]
    holdout_index: tuple[object, ...]
    scale: str
    baseline: float | None
    cost: float
    power: float
    false_positive_rate: float
    interval_width: float
    monte_carlo_error: float
    selection_objective: str

    def analysis_handoff(self) -> AnalysisHandoff | AdapterAnalysisHandoff:
        """Return the frozen analysis spec. Does not sample."""
        return _build_handoff(
            self.history,
            self.treated_units,
            self.control_units,
            self.decision,
            estimator=self.estimator,
            duration=self.duration,
            sample_kwargs=dict(self.sample_kwargs),
            outcome_scale=self.outcome_scale,
            time_column=self.time_column,
        )

    def inject(self, panel: pd.DataFrame, effect: float) -> pd.DataFrame:
        """Return a copy with ``effect`` added only on the frozen holdout.

        The increment is :func:`unit_period_effect` for this treated count.
        Training rows and donor columns are unchanged. This is not an
        all-history level shift.

        Parameters
        ----------
        panel : pandas.DataFrame
            Panel containing the frozen holdout labels and treated columns.
        effect : float
            Group-total effect on ``scale``.
        """
        if not isinstance(panel, pd.DataFrame):
            raise TypeError("panel must be a pandas DataFrame.")
        copied = panel.copy()
        missing = [label for label in self.holdout_index if label not in copied.index]
        if missing:
            raise ValueError(
                f"panel is missing frozen holdout labels {missing}. No column was changed."
            )
        absent = [name for name in self.treated_units if name not in copied.columns]
        if absent:
            raise ValueError(
                f"panel is missing treated columns {absent}. No column was changed."
            )
        increment = unit_period_effect(
            scale=self.scale,  # type: ignore[arg-type]
            magnitude=float(effect),
            duration=self.duration,
            n_treated=len(self.treated_units),
            baseline=self.baseline,
        )
        for name in self.treated_units:
            values = copied.loc[list(self.holdout_index), name].to_numpy(dtype=float)
            copied.loc[list(self.holdout_index), name] = values + increment
        return copied

    def audit(self, panel: pd.DataFrame) -> AuditReport:
        """Fit this identity on an untouched tail. Does not re-search.

        The panel's tail of the frozen duration is the audit holdout. Overlap
        with the planning holdout is refused before sampling, including when
        those labels fall in the audit training index.

        Parameters
        ----------
        panel : pandas.DataFrame
            Later history. Its tail must be disjoint from the planning holdout.
        """
        if not isinstance(panel, pd.DataFrame):
            raise TypeError("panel must be a pandas DataFrame. No fit was started.")
        spec = _build_handoff(
            panel,
            self.treated_units,
            self.control_units,
            self.decision,
            estimator=self.estimator,
            duration=self.duration,
            sample_kwargs=dict(self.sample_kwargs),
            outcome_scale=self.outcome_scale,
            time_column=self.time_column,
        )
        overlap = [
            label for label in spec.holdout_index if label in set(self.holdout_index)
        ]
        if overlap:
            raise ValueError(
                "Audit holdout overlaps the frozen planning holdout "
                f"{overlap}. That is not an untouched window. No fit was started."
            )
        contaminated = [
            label for label in spec.training_index if label in set(self.holdout_index)
        ]
        if contaminated:
            raise ValueError(
                "Audit training includes the frozen planning holdout "
                f"{contaminated}. That is not an untouched window. No fit was started."
            )
        evaluation = spec.run(panel)
        return AuditReport(
            candidate_id=self.candidate_id,
            decided=bool(evaluation.decision.decided),
            posterior_tail_probability=float(
                evaluation.decision.posterior_tail_probability
            ),
            holdout_index=tuple(spec.holdout_index),
            evaluation=evaluation,
            reason=None
            if evaluation.decision.decided
            else "The frozen design did not decide on the audit window.",
        )


@dataclass(frozen=True)
class AuditReport:
    """One frozen identity on one untouched window. Not a new search."""

    candidate_id: str
    decided: bool
    posterior_tail_probability: float
    holdout_index: tuple[object, ...]
    evaluation: object
    reason: str | None


@dataclass(frozen=True)
class SearchResult:
    """Screen, power, and one explicit selection or an inconclusive result.

    ``frame`` is the plot-ready table. Grouped rows record ``divergences``,
    ``max_rhat``, ``min_bulk_ess``, ``diagnostics_status``, and
    ``diagnostics_reason``. A missing divergence count stays null. The power
    curve's ``divergences`` integer is not that record. ``studies`` holds a
    completed or withheld curve whose divergence integer is a recorded count,
    keyed by ``candidate_id``. A grouped curve with a missing count is omitted.
    ``None`` when no design met the declared criteria.
    ``closest_candidate_id`` is the nearest completed row for a redesign note. It is not a selection.
    ``optimality`` is never ``global``. ``runtime_seconds`` is wall time for
    this call. ``peak_traced_bytes`` is the tracemalloc peak of Python
    allocations during the call, not process RSS.
    """

    frame: pd.DataFrame
    candidates: object
    studies: Mapping[str, EffectByDuration]
    frozen: FrozenDesign | None
    status: str
    reason: str | None
    closest_candidate_id: str | None
    n_inference_fits: int
    n_skipped: int
    partial: bool
    optimality: str
    enumeration_policy: str
    selection_objective: str
    evaluation_order: str
    pareto_ids: tuple[str, ...]
    runtime_seconds: float
    peak_traced_bytes: int

    def plot_data(self) -> pd.DataFrame:
        """Return a copy of the plot-ready table. Does not fit."""
        return self.frame.copy()


@dataclass(frozen=True)
class SelectionValidation:
    """Post-search rates on independent panels. Not the inner power curve.

    False positives and power are decision rates of the frozen handoff after
    selection. ``mean_interval_width`` is the mean alternate-panel interval
    width, and ``interval_width_monte_carlo_error`` is the standard error of
    that mean. The inner model-conditional curve is not reused as this proof.
    """

    n_replicates: int
    n_frozen: int
    n_inconclusive: int
    false_positive_rate: float | None
    power: float | None
    mean_interval_width: float | None
    interval_width_monte_carlo_error: float | None
    false_positive_monte_carlo_error: float | None
    power_monte_carlo_error: float | None
    confirmation_fits: int
    failures: tuple[str, ...]
    training_max_abs_diff: float
    holdout_shift: float | None
    redesign_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ScreeningComparison:
    """What the shortlist dropped, and the cost of that drop.

    Identity loss needs no fit. Decision regret is the frozen shortlist cost
    minus the frozen exhaustive cost, so a positive value means the cap chose
    a more expensive qualified design. ``cost_regret`` is ``None`` when one
    side did not freeze, or when the exhaustive pass did not evaluate every
    passing identity. ``cost_regret_reason`` says which of those applied.
    """

    shortlist_ids: tuple[str, ...]
    exhaustive_ids: tuple[str, ...]
    lost_ids: tuple[str, ...]
    shortlist_winner: str | None
    exhaustive_winner: str | None
    cost_regret: float | None
    n_inference_fits: int
    cost_regret_reason: str | None = None


def sampler_seed(caller_seed: int, candidate_id: str, durations: Sequence[int]) -> int:
    """Return the sampler seed for one candidate. Independent of loop order.

    Parameters
    ----------
    caller_seed : int
        Seed passed to :func:`search`.
    candidate_id : str
        JSON identity of the treated and donor tuples.
    durations : sequence of int
        Horizons this study will fit.
    """
    _require_seed(caller_seed, "caller_seed")
    payload = json.dumps(
        [int(caller_seed), candidate_id, [int(item) for item in durations]],
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little") % (2**31 - 1)
    return value or 1


class _SearchTrace:
    """Stop a search-owned tracemalloc session if this call raises."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.own = False
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.own = True
        tracemalloc.reset_peak()

    def finish(self) -> tuple[float, int]:
        elapsed = time.perf_counter() - self.started
        _, peak = tracemalloc.get_traced_memory()
        self.close()
        return elapsed, int(peak)

    def close(self) -> None:
        if self.own and tracemalloc.is_tracing():
            tracemalloc.stop()
        self.own = False

    def __del__(self) -> None:
        self.close()


def search(
    history: pd.DataFrame,
    units: Sequence[str],
    decision: DecisionRule,
    criterion: PowerCriterion,
    worthwhile_effect: float,
    durations: Sequence[int],
    *,
    selection_end: object,
    sample_kwargs: Mapping[str, object],
    random_seed: int,
    budget: ComputationalBudget,
    estimator: str = "synthetic_control",
    min_group_size: int = 1,
    max_group_size: int = 3,
    scale: str = "absolute",
    baseline: float | None = None,
    outcome_scale: str = "outcome",
    time_column: str | None = None,
    treatment_eligible: Sequence[str] | None = None,
    donor_eligible: Sequence[str] | None = None,
    always_control: Sequence[str] = (),
    excluded_units: Sequence[str] = (),
    required_treated: Sequence[str] = (),
    supplied_groups: Sequence[Sequence[str]] | None = None,
    fixed_treated: Sequence[str] | None = None,
    donor_pools: Sequence[Sequence[str]] | None = None,
    spillover_exclusions: Sequence[tuple[str, str]] = (),
    min_remaining_donors: int = 1,
    min_outcome_share: float | None = None,
    max_outcome_share: float | None = None,
    unit_costs: Sequence[tuple[str, float]] = (),
    cost_budget: float | None = None,
    unit_covariates: Sequence[tuple[str, Sequence[tuple[str, float]]]] = (),
    balance_constraints: Sequence[BalanceConstraint] = (),
    exact_ceiling: int = 5000,
    max_generated: int | None = None,
    shortlist_size: int = 10,
    scenario_family: str = "correctly_specified_noisy_reference",
    effect_grid: Sequence[float] | None = None,
) -> SearchResult:
    """Screen candidates, evaluate the shortlist, and freeze one design or none.

    ``history`` is the planning frame. Its tail is the holdout. Do not include
    an audit window; pass that only to :meth:`FrozenDesign.audit`. The screen
    sees rows at or before ``selection_end`` and that label must sit strictly
    before every requested holdout. The plan is not built on the truncated
    screen copy.

    Size-one ``synthetic_control`` uses :class:`ReferencePowerEvaluator`.
    A larger vanilla group is refused by that plan before sampling and is
    recorded with ``n_fits`` of 0. Augmented and interactive fixed-effects
    designs use :meth:`HistoryOnlyPlan.evaluate` on panels generated by
    extending the published conditional-on-donors mechanism. The inner
    criterion passed to the size-one engine is ``fixed_design`` because that
    engine refuses ``post_search``. The outer validation owns that scope.
    A grouped pre-period simplex ratio above the published misfit threshold
    withholds power for every estimator, and that row is not frozen. A
    constructor refusal, including a size-one plan, is rejected with zero
    fits before the budget is reserved. A size-one study that returns fewer
    successful replicates than its declared tolerance is failed and is not
    frozen. A reference failure after fitting starts charges the reserved
    replicate count, so the next candidate cannot start on an under-count.

    Parameters
    ----------
    history : pandas.DataFrame
        Planning panel. The holdout is its tail. Audit rows do not belong here.
    units : sequence of str
        Population passed to the candidate screen.
    decision : DecisionRule
        One-sided rule stored on every study and on the frozen handoff.
    criterion : PowerCriterion
        Power, false-positive, precision, and Monte Carlo targets.
    worthwhile_effect : float
        Effect the selection reads. It is not the minimum detectable effect.
    durations : sequence of int
        Holdout lengths. Each length is a separate fit.
    selection_end : object
        Last label the screen may see. Strictly before every holdout.
    sample_kwargs : mapping
        Must include positive ``draws`` and ``tune``. Must not include
        ``random_seed``; the candidate seed is derived from ``random_seed``.
    random_seed : int
        Caller seed. Candidate seeds do not depend on evaluation order.
    budget : ComputationalBudget
        Maximum inference fits. A study that would exceed it is skipped.
        A refusal that starts no fit is rejected and does not consume the cap.
    estimator : str, optional
        ``synthetic_control``, ``augmented_synthetic_control``, or
        ``interactive_fixed_effects``.
    min_group_size : int, optional
        Smallest treated group to enumerate. One to three.
    max_group_size : int, optional
        Largest treated group to enumerate. One to three.
    scale : str, optional
        ``absolute`` or ``relative``. Relative needs ``baseline``.
    baseline : float or None, optional
        Cumulative baseline for a relative effect. An absolute effect rejects
        a baseline and starts no fit. A relative effect without a finite
        baseline strictly above zero is rejected the same way.
    outcome_scale : str, optional
        ``outcome`` or ``unit``. Passed to the plan, not recomputed here.
    time_column : str or None, optional
        Column to use as the planning index. The default index is kept.
        Returned designs use that index. Pass indexed panels to ``inject``,
        ``audit``, and ``analysis_handoff().run``.
    treatment_eligible : sequence of str or None, optional
        Units that may be treated. ``None`` uses the population.
    donor_eligible : sequence of str or None, optional
        Units that may be donors. ``None`` uses the population.
    always_control : sequence of str, optional
        Units that stay donors. An always-control unit is donor-eligible
        unless a spillover removes it from one candidate.
    excluded_units : sequence of str, optional
        Names removed from the population before enumeration.
    required_treated : sequence of str, optional
        Names present in every enumerated treated group.
    supplied_groups : sequence of sequences of str or None, optional
        Treated groups to screen instead of enumerating sizes.
    fixed_treated : sequence of str or None, optional
        One treated group. ``donor_pools`` may vary around it.
    donor_pools : sequence of sequences of str or None, optional
        Alternative donor sets for ``fixed_treated``. ``None`` uses the
        complement.
    spillover_exclusions : sequence of tuple of str, optional
        ``(source, contaminated)`` pairs. A treated source blocks that
        contaminated name from its donor pool.
    min_remaining_donors : int, optional
        Smallest donor pool kept after spillover exclusions.
    min_outcome_share : float or None, optional
        Lowest selected-window share that passes. ``None`` sets no floor.
    max_outcome_share : float or None, optional
        Highest selected-window share that passes. ``None`` sets no cap.
    unit_costs : sequence of tuple, optional
        ``(unit, cost)`` pairs. A missing cost cannot win a cost slot.
    cost_budget : float or None, optional
        Passed to the screen as its cost cap. It is not the fit budget.
    unit_covariates : sequence of tuple, optional
        ``(unit, ((covariate, value), ...))`` scalars. Not history columns.
    balance_constraints : sequence of BalanceConstraint, optional
        Equal-weight covariate gaps. A gap above the cap filters.
    exact_ceiling : int, optional
        Spaces at or below this count are enumerated in full.
    max_generated : int or None, optional
        Identities kept when the space exceeds ``exact_ceiling``.
    shortlist_size : int, optional
        Passing rows evaluated. The cap does not grow.
    scenario_family : str, optional
        Declared panel generator. It does not refit.
    effect_grid : sequence of float or None, optional
        Nonzero effects to translate. The default is the worthwhile effect.
        Zero is always evaluated from the same fits.

    Returns
    -------
    SearchResult
        The screen, one row per candidate and duration, and at most one
        frozen design. ``runtime_seconds`` and ``peak_traced_bytes`` cover
        this call.
    """
    trace = _SearchTrace()
    frame = _planning_frame(history, time_column)
    time_column = None
    horizons = _require_durations(durations)
    _require_decision(decision)
    _require_criterion(criterion)
    if not isinstance(budget, ComputationalBudget):
        raise TypeError("budget must be a ComputationalBudget. No fit was started.")
    settings = _require_sample_kwargs(sample_kwargs)
    caller_seed = _require_seed(random_seed, "random_seed")
    worthwhile = _require_finite(worthwhile_effect, "worthwhile_effect")
    family = _require_family(scenario_family)
    _require_selection_end(frame, selection_end, horizons)
    effects = _effect_grid(worthwhile, effect_grid)
    try:
        planned = replicates_for_tolerance(
            criterion.monte_carlo_error_tolerance, n_durations=len(horizons)
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    request = _candidate_request(
        frame=frame,
        units=units,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        selection_end=selection_end,
        random_seed=caller_seed,
        treatment_eligible=treatment_eligible,
        donor_eligible=donor_eligible,
        always_control=always_control,
        excluded_units=excluded_units,
        required_treated=required_treated,
        supplied_groups=supplied_groups,
        fixed_treated=fixed_treated,
        donor_pools=donor_pools,
        spillover_exclusions=spillover_exclusions,
        min_remaining_donors=min_remaining_donors,
        min_outcome_share=min_outcome_share,
        max_outcome_share=max_outcome_share,
        unit_costs=unit_costs,
        cost_budget=cost_budget,
        unit_covariates=unit_covariates,
        balance_constraints=balance_constraints,
        exact_ceiling=exact_ceiling,
        max_generated=max_generated,
        shortlist_size=shortlist_size,
    )
    table = CandidateGenerator().generate(request)
    screen = table.frame
    shortlisted = [
        row for _, row in screen.iterrows() if row["status"] == "shortlisted"
    ]
    ordered = sorted(shortlisted, key=lambda row: str(row["candidate_id"]))
    studies: dict[str, EffectByDuration] = {}
    evaluations: dict[str, _CandidateEvaluation] = {}
    fits_used = 0
    skipped = 0
    inner = _fixed_design_criterion(criterion)
    for row in ordered:
        identity = str(row["candidate_id"])
        refusal = _free_refusal(
            frame,
            row,
            horizons,
            decision,
            estimator,
            settings,
            caller_seed,
            outcome_scale,
            time_column,
            scale,
            baseline,
        )
        if refusal is not None:
            evaluations[identity] = refusal
            continue
        planned_fits = planned * len(horizons)
        if fits_used + planned_fits > budget.max_inference_fits:
            evaluations[identity] = _CandidateEvaluation(
                None, "skipped", "The next study would exceed max_inference_fits.", 0
            )
            skipped += 1
            continue
        outcome = _evaluate_candidate(
            frame,
            row,
            horizons,
            effects,
            decision,
            inner,
            worthwhile,
            estimator=estimator,
            family=family,
            scale=scale,
            baseline=baseline,
            sample_kwargs=settings,
            caller_seed=caller_seed,
            outcome_scale=outcome_scale,
            time_column=time_column,
            selection_end=selection_end,
        )
        fits_used += outcome.n_fits
        if _publish_study(outcome):
            studies[identity] = outcome.study
        evaluations[identity] = outcome
    rows = _result_rows(
        screen,
        horizons,
        evaluations,
        decision,
        criterion,
        worthwhile,
        estimator,
    )
    selected, status, reason, closest = _select(rows, criterion)
    for row in rows:
        row["selected"] = (
            selected is not None
            and row["candidate_id"] == selected["candidate_id"]
            and row["duration"] == selected["duration"]
        )
    pareto = _pareto(rows)
    for row in rows:
        row["pareto"] = (row["candidate_id"], row["duration"]) in pareto
    frozen = None
    if selected is not None:
        frozen = _freeze(
            frame,
            selected,
            decision,
            settings,
            caller_seed,
            horizons,
            outcome_scale,
            time_column,
            scale,
            baseline,
        )
    optimality = _optimality(table.partial, skipped, screen, evaluations)
    elapsed, peak = trace.finish()
    return SearchResult(
        frame=pd.DataFrame(rows, columns=list(SEARCH_COLUMNS)),
        candidates=table,
        studies=studies,
        frozen=frozen,
        status=status,
        reason=reason,
        closest_candidate_id=closest,
        n_inference_fits=fits_used,
        n_skipped=skipped,
        partial=bool(table.partial or skipped),
        optimality=optimality,
        enumeration_policy=str(table.enumeration_policy),
        selection_objective=SELECTION_OBJECTIVE,
        evaluation_order=EVALUATION_ORDER,
        pareto_ids=tuple(sorted({item[0] for item in pareto})),
        runtime_seconds=elapsed,
        peak_traced_bytes=peak,
    )


def _replay_search(
    history: pd.DataFrame,
    units: Sequence[str],
    decision: DecisionRule,
    criterion: PowerCriterion,
    worthwhile_effect: float,
    durations: Sequence[int],
    *,
    estimator: str,
    min_group_size: int,
    max_group_size: int,
    selection_end: object,
    sample_kwargs: Mapping[str, object],
    random_seed: int,
    budget: ComputationalBudget,
    scale: str,
    baseline: float | None,
    outcome_scale: str,
    time_column: str | None,
    always_control: Sequence[str],
    unit_costs: Sequence[tuple[str, float]],
    shortlist_size: int,
    scenario_family: str,
    treatment_eligible: Sequence[str] | None,
    donor_eligible: Sequence[str] | None,
    excluded_units: Sequence[str],
    required_treated: Sequence[str],
    supplied_groups: Sequence[Sequence[str]] | None,
    fixed_treated: Sequence[str] | None,
    donor_pools: Sequence[Sequence[str]] | None,
    spillover_exclusions: Sequence[tuple[str, str]],
    min_remaining_donors: int,
    min_outcome_share: float | None,
    max_outcome_share: float | None,
    cost_budget: float | None,
    unit_covariates: Sequence[tuple[str, Sequence[tuple[str, float]]]],
    balance_constraints: Sequence[BalanceConstraint],
    exact_ceiling: int,
    max_generated: int | None,
    effect_grid: Sequence[float] | None,
) -> SearchResult:
    """Run one inner search with the caller's operational constraints."""
    return search(
        history,
        units,
        decision,
        criterion,
        worthwhile_effect,
        durations,
        estimator=estimator,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        selection_end=selection_end,
        sample_kwargs=sample_kwargs,
        random_seed=random_seed,
        budget=budget,
        scale=scale,
        baseline=baseline,
        outcome_scale=outcome_scale,
        time_column=time_column,
        treatment_eligible=treatment_eligible,
        donor_eligible=donor_eligible,
        always_control=always_control,
        excluded_units=excluded_units,
        required_treated=required_treated,
        supplied_groups=supplied_groups,
        fixed_treated=fixed_treated,
        donor_pools=donor_pools,
        spillover_exclusions=spillover_exclusions,
        min_remaining_donors=min_remaining_donors,
        min_outcome_share=min_outcome_share,
        max_outcome_share=max_outcome_share,
        unit_costs=unit_costs,
        cost_budget=cost_budget,
        unit_covariates=unit_covariates,
        balance_constraints=balance_constraints,
        exact_ceiling=exact_ceiling,
        max_generated=max_generated,
        shortlist_size=shortlist_size,
        scenario_family=scenario_family,
        effect_grid=effect_grid,
    )


def validate_selection(
    history: pd.DataFrame,
    units: Sequence[str],
    decision: DecisionRule,
    criterion: PowerCriterion,
    worthwhile_effect: float,
    durations: Sequence[int],
    *,
    n_replicates: int,
    random_seed: int,
    budget: ComputationalBudget,
    sample_kwargs: Mapping[str, object],
    selection_end: object,
    estimator: str = "synthetic_control",
    always_control: Sequence[str] = (),
    scenario_family: str = "correctly_specified_noisy_reference",
    min_group_size: int = 1,
    max_group_size: int = 1,
    shortlist_size: int = 2,
    scale: str = "absolute",
    baseline: float | None = None,
    outcome_scale: str = "outcome",
    time_column: str | None = None,
    unit_costs: Sequence[tuple[str, float]] = (),
    treatment_eligible: Sequence[str] | None = None,
    donor_eligible: Sequence[str] | None = None,
    excluded_units: Sequence[str] = (),
    required_treated: Sequence[str] = (),
    supplied_groups: Sequence[Sequence[str]] | None = None,
    fixed_treated: Sequence[str] | None = None,
    donor_pools: Sequence[Sequence[str]] | None = None,
    spillover_exclusions: Sequence[tuple[str, str]] = (),
    min_remaining_donors: int = 1,
    min_outcome_share: float | None = None,
    max_outcome_share: float | None = None,
    cost_budget: float | None = None,
    unit_covariates: Sequence[tuple[str, Sequence[tuple[str, float]]]] = (),
    balance_constraints: Sequence[BalanceConstraint] = (),
    exact_ceiling: int = 5000,
    max_generated: int | None = None,
    effect_grid: Sequence[float] | None = None,
) -> SelectionValidation:
    """Re-run search on independent panels and confirm the frozen design.

    Each replicate generates an untreated panel, runs :func:`search`, and,
    only if a design froze, fits that handoff on the untreated panel and on
    a copy whose holdout alone contains the worthwhile effect. The inner
    power curve is not this report. A replicate that freezes nothing is
    inconclusive, not a false positive. Operational constraints accepted by
    :func:`search` are replayed on every inner search. A size-one
    ``synthetic_control`` procedure draws each non-control column from
    :meth:`ReferencePowerEvaluator.untreated_panel` against the fixed donor
    core. A grouped or mixed procedure keeps the grouped generator. A
    confirmation that raises is not a non-detection: the rates stay null if
    any frozen confirmation failed, and ``confirmation_fits`` counts calls
    that started.

    Parameters
    ----------
    history : pandas.DataFrame
        Observed panel. Non-core columns are regenerated on each replicate.
    units : sequence of str
        Population passed to each inner search.
    decision : DecisionRule
        Rule used by the inner search and by both confirmation fits.
    criterion : PowerCriterion
        Targets for the inner search. The outer rate is not this curve.
    worthwhile_effect : float
        Effect injected into the confirmation holdout only.
    durations : sequence of int
        Holdout lengths passed to each inner search.
    n_replicates : int
        Independent outer panels. This is not the inner Monte Carlo count.
    random_seed : int
        Caller seed. Replicate panels do not depend on loop order.
    budget : ComputationalBudget
        Fit cap for each inner search.
    sample_kwargs : mapping
        Draws and tune for the inner search and the confirmation fits.
    selection_end : object
        Last label the inner screen may see.
    estimator : str, optional
        Estimator passed to each inner search.
    always_control : sequence of str
        Donor core that stays observed. Required, so the panel remains
        conditional on observed donors.
    scenario_family : str, optional
        Generator for the regenerated columns.
    min_group_size : int, optional
        Smallest treated group in each inner search.
    max_group_size : int, optional
        Largest treated group in each inner search.
    shortlist_size : int, optional
        Passing rows each inner search may evaluate.
    scale : str, optional
        ``absolute`` or ``relative``.
    baseline : float or None, optional
        Cumulative baseline when ``scale`` is relative.
    outcome_scale : str, optional
        ``outcome`` or ``unit``.
    time_column : str or None, optional
        Column to use as the planning index.
    unit_costs : sequence of tuple, optional
        ``(unit, cost)`` pairs passed to each inner search.
    treatment_eligible : sequence of str or None, optional
        Units that may be treated. Forwarded to each inner search.
    donor_eligible : sequence of str or None, optional
        Units that may be donors. Forwarded to each inner search.
    excluded_units : sequence of str, optional
        Names removed before enumeration. Forwarded to each inner search.
    required_treated : sequence of str, optional
        Names present in every enumerated treated group.
    supplied_groups : sequence of sequences of str or None, optional
        Treated groups to screen instead of enumerating sizes.
    fixed_treated : sequence of str or None, optional
        One treated group. ``donor_pools`` may vary around it.
    donor_pools : sequence of sequences of str or None, optional
        Alternative donor sets for ``fixed_treated``.
    spillover_exclusions : sequence of tuple of str, optional
        ``(source, contaminated)`` pairs forwarded to each inner search.
    min_remaining_donors : int, optional
        Smallest donor pool kept after spillover exclusions.
    min_outcome_share : float or None, optional
        Lowest selected-window share that passes.
    max_outcome_share : float or None, optional
        Highest selected-window share that passes.
    cost_budget : float or None, optional
        Screen cost cap. It is not the fit budget.
    unit_covariates : sequence of tuple, optional
        ``(unit, ((covariate, value), ...))`` scalars forwarded unchanged.
    balance_constraints : sequence of BalanceConstraint, optional
        Equal-weight covariate gaps forwarded to each inner search.
    exact_ceiling : int, optional
        Spaces at or below this count are enumerated in full.
    max_generated : int or None, optional
        Identities kept when the space exceeds ``exact_ceiling``.
    effect_grid : sequence of float or None, optional
        Nonzero effects to translate. Zero is still evaluated.

    Returns
    -------
    SelectionValidation
        Outer decision rates, their Monte Carlo errors, and the holdout check.
    """
    count = _require_seed(n_replicates, "n_replicates")
    if not always_control:
        raise ValueError(
            "validate_selection needs always_control as a fixed donor core. "
            "Regenerating every column would not be conditional on observed "
            "donors. No fit was started."
        )
    family = _require_family(scenario_family)
    frame = _planning_frame(history, time_column)
    false_positives: list[bool] = []
    detections: list[bool] = []
    widths: list[float] = []
    failures: list[str] = []
    reasons: list[str] = []
    training_diff = 0.0
    shift: float | None = None
    confirmation_fits = 0
    frozen_count = 0
    confirmation_failed = False
    reference = _size_one_reference_procedure(
        estimator,
        min_group_size,
        max_group_size,
        fixed_treated,
        supplied_groups,
    )
    panel_cache: dict[str, ReferencePowerEvaluator] = {}
    for replicate in range(count):
        panel = _validation_panel(
            frame,
            units,
            always_control,
            family,
            random_seed,
            replicate,
            selection_end,
            decision=decision,
            sample_kwargs=sample_kwargs,
            reference=reference,
            panel_cache=panel_cache,
        )
        result = _replay_search(
            panel,
            units,
            decision,
            criterion,
            worthwhile_effect,
            durations,
            estimator=estimator,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            selection_end=selection_end,
            sample_kwargs=sample_kwargs,
            random_seed=_outer_seed(random_seed, replicate),
            budget=budget,
            scale=scale,
            baseline=baseline,
            outcome_scale=outcome_scale,
            time_column=None,  # ``_planning_frame`` already indexed this panel.
            always_control=always_control,
            unit_costs=unit_costs,
            shortlist_size=shortlist_size,
            scenario_family=family,
            treatment_eligible=treatment_eligible,
            donor_eligible=donor_eligible,
            excluded_units=excluded_units,
            required_treated=required_treated,
            supplied_groups=supplied_groups,
            fixed_treated=fixed_treated,
            donor_pools=donor_pools,
            spillover_exclusions=spillover_exclusions,
            min_remaining_donors=min_remaining_donors,
            min_outcome_share=min_outcome_share,
            max_outcome_share=max_outcome_share,
            cost_budget=cost_budget,
            unit_covariates=unit_covariates,
            balance_constraints=balance_constraints,
            exact_ceiling=exact_ceiling,
            max_generated=max_generated,
            effect_grid=effect_grid,
        )
        if result.frozen is None:
            reasons.append(result.reason or "inconclusive")
            continue
        frozen_count += 1
        injected = result.frozen.inject(panel, worthwhile_effect)
        training_diff = max(
            training_diff,
            _training_diff(panel, injected, result.frozen.training_index),
        )
        shift = _holdout_shift(panel, injected, result.frozen)
        started = 0
        try:
            null_spec = result.frozen.analysis_handoff()
            started = 1
            null_eval = null_spec.run(panel)
            alt_spec = result.frozen.analysis_handoff()
            started = 2
            alt_eval = alt_spec.run(injected)
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
            confirmation_fits += started
            confirmation_failed = True
            continue
        confirmation_fits += 2
        false_positives.append(bool(null_eval.decision.decided))
        detections.append(bool(alt_eval.decision.decided))
        widths.append(float(alt_eval.decision.hdi_upper - alt_eval.decision.hdi_lower))
    identified = not confirmation_failed
    return SelectionValidation(
        n_replicates=count,
        n_frozen=frozen_count,
        n_inconclusive=count - frozen_count,
        false_positive_rate=_optional_rate(false_positives) if identified else None,
        power=_optional_rate(detections) if identified else None,
        mean_interval_width=float(np.mean(widths)) if widths and identified else None,
        interval_width_monte_carlo_error=_mean_error(widths) if identified else None,
        false_positive_monte_carlo_error=(
            _optional_error(false_positives) if identified else None
        ),
        power_monte_carlo_error=_optional_error(detections) if identified else None,
        confirmation_fits=confirmation_fits,
        failures=tuple(failures),
        training_max_abs_diff=training_diff,
        holdout_shift=shift,
        redesign_reasons=tuple(reasons),
    )


def compare_screening(
    history: pd.DataFrame,
    units: Sequence[str],
    decision: DecisionRule,
    criterion: PowerCriterion,
    worthwhile_effect: float,
    durations: Sequence[int],
    *,
    selection_end: object,
    sample_kwargs: Mapping[str, object],
    random_seed: int,
    budget: ComputationalBudget,
    shortlist_size: int,
    estimator: str = "synthetic_control",
    min_group_size: int = 1,
    max_group_size: int = 1,
    always_control: Sequence[str] = (),
    unit_costs: Sequence[tuple[str, float]] = (),
    scenario_family: str = "correctly_specified_noisy_reference",
    scale: str = "absolute",
    baseline: float | None = None,
    outcome_scale: str = "outcome",
    time_column: str | None = None,
    treatment_eligible: Sequence[str] | None = None,
    donor_eligible: Sequence[str] | None = None,
    excluded_units: Sequence[str] = (),
    required_treated: Sequence[str] = (),
    supplied_groups: Sequence[Sequence[str]] | None = None,
    fixed_treated: Sequence[str] | None = None,
    donor_pools: Sequence[Sequence[str]] | None = None,
    spillover_exclusions: Sequence[tuple[str, str]] = (),
    min_remaining_donors: int = 1,
    min_outcome_share: float | None = None,
    max_outcome_share: float | None = None,
    cost_budget: float | None = None,
    unit_covariates: Sequence[tuple[str, Sequence[tuple[str, float]]]] = (),
    balance_constraints: Sequence[BalanceConstraint] = (),
    exact_ceiling: int = 5000,
    max_generated: int | None = None,
    effect_grid: Sequence[float] | None = None,
) -> ScreeningComparison:
    """Compare a capped shortlist with the generator's unfiltered set.

    Both searches use the same caller seed, so a candidate's sampler seed
    does not depend on which pass evaluates it. They do not share study
    objects. The exhaustive cap is the count of candidates the generator did
    not filter, not a second ranking rule. Identity loss is a shortlisted
    identity that cap dropped. Cost regret is the capped winner's cost minus
    the exhaustive winner's cost only when both freeze and the exhaustive
    pass evaluated every passing identity. A skipped exhaustive row or a
    partial generator withholds that number; ``cost_regret_reason`` says why.
    Operational constraints accepted by :func:`search` are replayed on both
    passes.

    Parameters
    ----------
    history : pandas.DataFrame
        Planning panel shared by both searches. Studies are not shared.
    units : sequence of str
        Population passed to both screens.
    decision : DecisionRule
        Rule passed to both searches.
    criterion : PowerCriterion
        Targets passed to both searches.
    worthwhile_effect : float
        Effect both searches read.
    durations : sequence of int
        Holdout lengths passed to both searches.
    selection_end : object
        Last label both screens may see.
    sample_kwargs : mapping
        Draws and tune for both searches.
    random_seed : int
        Shared caller seed. A candidate's sampler seed does not depend on
        which pass evaluates it.
    budget : ComputationalBudget
        Fit cap for each search. The second pass does not reuse the first.
    shortlist_size : int
        Cap whose loss is being measured.
    estimator : str, optional
        Estimator passed to both searches.
    min_group_size : int, optional
        Smallest treated group in both screens.
    max_group_size : int, optional
        Largest treated group in both screens.
    always_control : sequence of str, optional
        Donor core passed to both screens.
    unit_costs : sequence of tuple, optional
        ``(unit, cost)`` pairs used by both screens.
    scenario_family : str, optional
        Panel generator passed to both searches.
    scale : str, optional
        ``absolute`` or ``relative``.
    baseline : float or None, optional
        Cumulative baseline when ``scale`` is relative.
    outcome_scale : str, optional
        ``outcome`` or ``unit``.
    time_column : str or None, optional
        Column to use as the planning index.
    treatment_eligible : sequence of str or None, optional
        Units that may be treated. Forwarded to both searches.
    donor_eligible : sequence of str or None, optional
        Units that may be donors. Forwarded to both searches.
    excluded_units : sequence of str, optional
        Names removed before enumeration. Forwarded to both searches.
    required_treated : sequence of str, optional
        Names present in every enumerated treated group.
    supplied_groups : sequence of sequences of str or None, optional
        Treated groups to screen instead of enumerating sizes.
    fixed_treated : sequence of str or None, optional
        One treated group. ``donor_pools`` may vary around it.
    donor_pools : sequence of sequences of str or None, optional
        Alternative donor sets for ``fixed_treated``.
    spillover_exclusions : sequence of tuple of str, optional
        ``(source, contaminated)`` pairs forwarded to both searches.
    min_remaining_donors : int, optional
        Smallest donor pool kept after spillover exclusions.
    min_outcome_share : float or None, optional
        Lowest selected-window share that passes.
    max_outcome_share : float or None, optional
        Highest selected-window share that passes.
    cost_budget : float or None, optional
        Screen cost cap. It is not the fit budget.
    unit_covariates : sequence of tuple, optional
        ``(unit, ((covariate, value), ...))`` scalars forwarded unchanged.
    balance_constraints : sequence of BalanceConstraint, optional
        Equal-weight covariate gaps forwarded to both searches.
    exact_ceiling : int, optional
        Spaces at or below this count are enumerated in full.
    max_generated : int or None, optional
        Identities kept when the space exceeds ``exact_ceiling``.
    effect_grid : sequence of float or None, optional
        Nonzero effects to translate. Zero is still evaluated.

    Returns
    -------
    ScreeningComparison
        Identity loss, cost regret, and the two search results.
    """

    def _pass(shortlist: int) -> SearchResult:
        return _replay_search(
            history,
            units,
            decision,
            criterion,
            worthwhile_effect,
            durations,
            estimator=estimator,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            selection_end=selection_end,
            sample_kwargs=sample_kwargs,
            random_seed=random_seed,
            budget=budget,
            scale=scale,
            baseline=baseline,
            outcome_scale=outcome_scale,
            time_column=time_column,
            always_control=always_control,
            unit_costs=unit_costs,
            shortlist_size=shortlist,
            scenario_family=scenario_family,
            treatment_eligible=treatment_eligible,
            donor_eligible=donor_eligible,
            excluded_units=excluded_units,
            required_treated=required_treated,
            supplied_groups=supplied_groups,
            fixed_treated=fixed_treated,
            donor_pools=donor_pools,
            spillover_exclusions=spillover_exclusions,
            min_remaining_donors=min_remaining_donors,
            min_outcome_share=min_outcome_share,
            max_outcome_share=max_outcome_share,
            cost_budget=cost_budget,
            unit_covariates=unit_covariates,
            balance_constraints=balance_constraints,
            exact_ceiling=exact_ceiling,
            max_generated=max_generated,
            effect_grid=effect_grid,
        )

    capped = _pass(shortlist_size)
    passing = int(
        (capped.candidates.frame["status"] != "filtered").sum()  # type: ignore[attr-defined]
    )
    exhaustive = _pass(max(passing, shortlist_size))
    shortlist_ids = tuple(
        sorted(
            str(item)
            for item in capped.candidates.frame.loc[  # type: ignore[attr-defined]
                capped.candidates.frame["status"] == "shortlisted", "candidate_id"  # type: ignore[attr-defined]
            ]
        )
    )
    exhaustive_ids = tuple(
        sorted(
            str(item)
            for item in exhaustive.candidates.frame.loc[  # type: ignore[attr-defined]
                exhaustive.candidates.frame["status"] == "shortlisted", "candidate_id"  # type: ignore[attr-defined]
            ]
        )
    )
    lost = tuple(item for item in exhaustive_ids if item not in set(shortlist_ids))
    regret = None
    if capped.frozen is None or exhaustive.frozen is None:
        regret_reason = (
            "One side did not freeze, so cost regret is not a decision comparison."
        )
    elif exhaustive.partial or int(
        (exhaustive.frame["evaluation_status"] == "skipped").sum()
    ):
        regret_reason = (
            "The exhaustive pass did not evaluate every passing identity, "
            "so cost regret is withheld."
        )
    else:
        regret = float(capped.frozen.cost - exhaustive.frozen.cost)
        regret_reason = None
    return ScreeningComparison(
        shortlist_ids=shortlist_ids,
        exhaustive_ids=exhaustive_ids,
        lost_ids=lost,
        shortlist_winner=None if capped.frozen is None else capped.frozen.candidate_id,
        exhaustive_winner=None
        if exhaustive.frozen is None
        else exhaustive.frozen.candidate_id,
        cost_regret=regret,
        n_inference_fits=capped.n_inference_fits + exhaustive.n_inference_fits,
        cost_regret_reason=regret_reason,
    )


def plot_search(result: SearchResult) -> Figure:
    """Plot power against cost for completed rows. Does not fit.

    Parameters
    ----------
    result : SearchResult
        A finished search. The figure reads ``frame`` and does not recompute it.
    """
    import matplotlib.pyplot as plt

    frame = result.frame
    completed = frame[frame["evaluation_status"] == "completed"].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    if completed.empty:
        ax.set_title("No completed power row")
    else:
        for size, group in completed.groupby("group_size"):
            ax.scatter(group["cost"], group["power"], label=f"group {size}")
        chosen = frame[frame["selected"]]
        if not chosen.empty:
            ax.scatter(
                chosen["cost"],
                chosen["power"],
                marker="*",
                s=160,
                label="frozen",
            )
        ax.legend()
    ax.set_xlabel("cost")
    ax.set_ylabel("power at the worthwhile effect")
    ax.set_title(result.status)
    return fig


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


@dataclass
class _CandidateEvaluation:
    study: EffectByDuration | None
    status: str
    reason: str | None
    n_fits: int
    unit_scale: float | None = None
    diagnostics: _DiagnosticRecord | None = None


def _scale_baseline_refusal(
    scale: str, baseline: float | None
) -> _CandidateEvaluation | None:
    """Reject a scale and baseline pair before any fit is reserved."""
    if scale == "relative" and (
        baseline is None or not np.isfinite(baseline) or baseline <= 0.0
    ):
        return _CandidateEvaluation(
            None,
            "rejected",
            (
                "Relative effects require a finite baseline strictly above zero. "
                "No fit was started."
            ),
            0,
        )
    if scale == "absolute" and baseline is not None:
        return _CandidateEvaluation(
            None,
            "rejected",
            "An absolute effect does not take a baseline. No fit was started.",
            0,
        )
    if scale not in {"absolute", "relative"}:
        return _CandidateEvaluation(
            None,
            "rejected",
            "scale must be 'absolute' or 'relative'. No fit was started.",
            0,
        )
    return None


def _free_refusal(
    frame: pd.DataFrame,
    row: pd.Series,
    horizons: tuple[int, ...],
    decision: DecisionRule,
    estimator: str,
    sample_kwargs: Mapping[str, object],
    caller_seed: int,
    outcome_scale: str,
    time_column: str | None,
    scale: str,
    baseline: float | None,
) -> _CandidateEvaluation | None:
    """Return a zero-fit refusal, or None when the study would sample."""
    scale_refusal = _scale_baseline_refusal(scale, baseline)
    if scale_refusal is not None:
        return scale_refusal
    treated = tuple(row["treated"])
    donors = tuple(row["donors"])
    seed = sampler_seed(caller_seed, str(row["candidate_id"]), horizons)
    settings = dict(sample_kwargs)
    settings["random_seed"] = seed
    try:
        HistoryOnlyPlan(
            frame,
            treated,
            donors,
            decision,
            estimator=estimator,
            sample_kwargs=settings,
            time_column=time_column,
            outcome_scale=outcome_scale,
        )
    except (TypeError, ValueError) as exc:
        return _CandidateEvaluation(None, "rejected", str(exc), 0)
    return _setup_refusal(frame, donors, horizons)


def _setup_refusal(
    frame: pd.DataFrame,
    donors: tuple[str, ...],
    horizons: tuple[int, ...],
) -> _CandidateEvaluation | None:
    """Reject a window or donor pool that cannot start a fit."""
    longest = max(horizons)
    if len(frame) - longest < MIN_TRAINING_ROWS:
        return _CandidateEvaluation(
            None,
            "rejected",
            (
                f"history has {len(frame)} rows. Duration {longest} leaves "
                f"fewer than {MIN_TRAINING_ROWS} training rows. No fit was started."
            ),
            0,
        )
    try:
        _noise_scale(frame, donors)
    except ValueError as exc:
        return _CandidateEvaluation(None, "rejected", str(exc), 0)
    return None


def _evaluate_candidate(
    frame: pd.DataFrame,
    row: pd.Series,
    horizons: tuple[int, ...],
    effects: tuple[float, ...],
    decision: DecisionRule,
    criterion: PowerCriterion,
    worthwhile: float,
    *,
    estimator: str,
    family: str,
    scale: str,
    baseline: float | None,
    sample_kwargs: dict[str, object],
    caller_seed: int,
    outcome_scale: str,
    time_column: str | None,
    selection_end: object,
) -> _CandidateEvaluation:
    treated = tuple(row["treated"])
    donors = tuple(row["donors"])
    identity = str(row["candidate_id"])
    seed = sampler_seed(caller_seed, identity, horizons)
    settings = dict(sample_kwargs)
    settings["random_seed"] = seed
    if estimator == "synthetic_control" and len(treated) == 1:
        return _reference_study(
            frame,
            treated,
            donors,
            horizons,
            effects,
            decision,
            criterion,
            worthwhile,
            family,
            scale,
            baseline,
            settings,
            outcome_scale,
            time_column,
        )
    return _grouped_study(
        frame,
        treated,
        donors,
        horizons,
        effects,
        decision,
        criterion,
        worthwhile,
        estimator,
        family,
        scale,
        baseline,
        settings,
        outcome_scale,
        time_column,
        selection_end,
        seed,
    )


def _reserved_reference_fits(
    criterion: PowerCriterion, horizons: tuple[int, ...]
) -> int:
    """Return the fit count a size-one study reserved before it started."""
    planned = replicates_for_tolerance(
        criterion.monte_carlo_error_tolerance, n_durations=len(horizons)
    )
    return planned * len(horizons)


def _reference_failure_fits(message: str, reserved: int) -> int:
    """Charge the reservation after fitting starts, and zero before it.

    ``Every replicate failed`` is emitted only after the engine has counted
    one fit per horizon for every planned replicate, even when an inner
    message says no fit started. A pre-fit refusal carries ``No fit was
    started`` and does not carry that wrapper.
    """
    if "Every replicate failed" in message or "No fit was started" not in message:
        return reserved
    return 0


def _reference_study(
    frame: pd.DataFrame,
    treated: tuple[str, ...],
    donors: tuple[str, ...],
    horizons: tuple[int, ...],
    effects: tuple[float, ...],
    decision: DecisionRule,
    criterion: PowerCriterion,
    worthwhile: float,
    family: str,
    scale: str,
    baseline: float | None,
    sample_kwargs: Mapping[str, object],
    outcome_scale: str,
    time_column: str | None,
) -> _CandidateEvaluation:
    try:
        plan = HistoryOnlyPlan(
            frame,
            treated,
            donors,
            decision,
            estimator="synthetic_control",
            sample_kwargs=sample_kwargs,
            time_column=time_column,
            outcome_scale=outcome_scale,
        )
    except (TypeError, ValueError) as exc:
        return _CandidateEvaluation(None, "rejected", str(exc), 0)
    try:
        evaluator = ReferencePowerEvaluator(plan, history=frame, scenario_family=family)
        # The engine refuses post_search. This module applies that scope
        # to the returned curve; the engine only computes the fixed design.
        engine_criterion = replace(criterion, qualification_scope="fixed_design")
        study = evaluator.effect_by_duration(
            horizons,
            effects,
            decision=decision,
            criterion=engine_criterion,
            scale=scale,  # type: ignore[arg-type]
            baseline=baseline,
            minimum_worthwhile_effect=worthwhile,
        )
    except MisfitNotPower as exc:
        return _CandidateEvaluation(exc.report, "withheld", str(exc), exc.report.n_fits)
    except ValueError as exc:
        return _CandidateEvaluation(None, "rejected", str(exc), 0)
    except Exception as exc:
        message = str(exc)
        fits = _reference_failure_fits(
            message, _reserved_reference_fits(criterion, horizons)
        )
        return _CandidateEvaluation(
            None, "failed", f"{type(exc).__name__}: {message}", fits
        )
    planned = replicates_for_tolerance(
        criterion.monte_carlo_error_tolerance, n_durations=len(horizons)
    )
    successful = min((point.n_replicates for point in study.points), default=0)
    if successful < planned:
        return _CandidateEvaluation(
            None,
            "failed",
            (
                f"Only {successful} of {planned} replicates completed. "
                "The declared tolerance sample was not reached. No design "
                "was frozen from this study."
            ),
            study.n_fits,
        )
    return _CandidateEvaluation(study, "completed", None, study.n_fits)


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
) -> _CandidateEvaluation:
    try:
        HistoryOnlyPlan(
            frame,
            treated,
            donors,
            decision,
            estimator=estimator,
            sample_kwargs=sample_kwargs,
            time_column=time_column,
            outcome_scale=outcome_scale,
        )
    except (TypeError, ValueError) as exc:
        return _CandidateEvaluation(None, "rejected", str(exc), 0)
    setup = _setup_refusal(frame, donors, horizons)
    if setup is not None:
        return setup
    try:
        replicates = replicates_for_tolerance(
            criterion.monte_carlo_error_tolerance, n_durations=len(horizons)
        )
        translated = _with_zero(effects)
        noise = _noise_scale(frame, donors)
        donor_sd = float(np.std(frame.loc[:, list(donors)].to_numpy(dtype=float)))
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
        ratio = _grouped_rmse_ratio(panels, treated, donors, horizons, noise, scales)
    except ValueError as exc:
        return _CandidateEvaluation(None, "rejected", str(exc), 0)
    draws: dict[tuple[int, float], list[object]] = {
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
            None, "failed", f"Every replicate failed ({detail}).", n_fits
        )
    shortest = min(len(sample) for sample in draws.values())
    if shortest < replicates:
        return _CandidateEvaluation(
            None,
            "failed",
            (
                f"Only {shortest} of {replicates} replicates completed. "
                "The declared tolerance sample was not reached. No design "
                "was frozen from this study."
            ),
            n_fits,
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
    status = "withheld" if study.power_withheld else "completed"
    return _CandidateEvaluation(
        study,
        status,
        study.withhold_reason,
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
    donor_sd = float(np.std(donor_matrix))
    share = math.sqrt(COMMON_SHOCK_SHARE)
    for index, name in enumerate(sorted(treated)):
        rng = np.random.default_rng(children[index + 1])
        scale = _unit_scale(history, name, selection_end, donor_sd)
        if family == "persistent_misfit":
            centered = np.arange(n_rows, dtype=float)
            centered = centered - float(centered.mean())
            series = (
                MISFIT_TREND_PER_DONOR_SD * donor_sd * centered
                + noise_scale * scale * rng.normal(0.0, 1.0, size=n_rows)
            )
        else:
            unit = _ar1_noise(rng, n_rows, rho)
            noise = share * common + math.sqrt(1.0 - COMMON_SHOCK_SHARE) * unit
            series = combo + noise_scale * scale * noise
        panel[name] = pd.Series(series, index=history.index)
    return panel


def _summarize_diagnostics(items: Sequence[object]) -> _DiagnosticRecord:
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
        diagnostics = getattr(item, "diagnostics", None)
        raw_count = getattr(diagnostics, "divergences", None)
        count = (
            raw_count
            if isinstance(raw_count, int) and not isinstance(raw_count, bool)
            else None
        )
        counts.append(count)
        raw_status = getattr(diagnostics, "diagnostics_status", None)
        if raw_status in {"available", "partial", "unavailable"}:
            statuses.append(str(raw_status))
        elif count is None:
            statuses.append("unavailable")
        else:
            statuses.append("available")
        raw_reason = getattr(diagnostics, "diagnostics_reason", None)
        if raw_reason:
            reasons.append(str(raw_reason))
        raw_rhat = getattr(diagnostics, "max_rhat", None)
        if isinstance(raw_rhat, int | float) and not isinstance(raw_rhat, bool):
            rhat = float(raw_rhat)
            if np.isfinite(rhat):
                rhats.append(rhat)
        if diagnostics is None or not hasattr(diagnostics, "min_bulk_ess"):
            ess_complete = False
        else:
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
    draws: dict[tuple[int, float], list[object]],
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
    diagnostic_items: list[object] = []
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
    donor_sd = float(np.std(frame.loc[:, list(donors)].to_numpy(dtype=float)))
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
            divergences=0 if record.divergences is None else record.divergences,
            max_rhat=record.max_rhat,
        ),
        record,
    )


def _result_rows(
    screen: pd.DataFrame,
    horizons: tuple[int, ...],
    evaluations: Mapping[str, _CandidateEvaluation],
    decision: DecisionRule,
    criterion: PowerCriterion,
    worthwhile: float,
    estimator: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, screen_row in screen.iterrows():
        identity = str(screen_row["candidate_id"])
        evaluation = evaluations.get(identity)
        for duration in horizons:
            point = _point_or_none(evaluation, duration, worthwhile)
            mde = _mde_or_none(evaluation, duration)
            status, reason, n_fits, failed = _row_status(screen_row, evaluation)
            power = None if point is None else _finite_or_none(point.probability)
            withheld = bool(
                evaluation and evaluation.study and evaluation.study.power_withheld
            )
            eligible = _eligible(status, power, point, criterion, screen_row)
            rows.append(
                {
                    "candidate_id": identity,
                    "treated": tuple(screen_row["treated"]),
                    "donors": tuple(screen_row["donors"]),
                    "n_donors": int(screen_row["n_donors"]),
                    "group_size": len(tuple(screen_row["treated"])),
                    "duration": duration,
                    "estimator": estimator,
                    "cost": _finite_or_none(screen_row["cost"]),
                    "outcome_share": _finite_or_none(screen_row["outcome_share"]),
                    "proxy_score": _finite_or_none(screen_row["proxy_score"]),
                    "proxy_role": screen_row["proxy_role"],
                    "shortlist_rank": screen_row["shortlist_rank"],
                    "boundary": bool(screen_row["boundary"]),
                    "screen_status": screen_row["status"],
                    "evaluation_status": status,
                    "reason": reason,
                    "power": power,
                    "power_withheld": withheld,
                    "false_positive_rate": None
                    if point is None
                    else _finite_or_none(point.false_positive_rate),
                    "precision_met": None if point is None else point.precision_met,
                    "interval_width": None if point is None else point.interval_width,
                    "monte_carlo_error": None
                    if point is None
                    else _finite_or_none(point.monte_carlo_error),
                    "mde": None if mde is None else mde.effect,
                    "mde_status": None if mde is None else mde.status,
                    "n_fits": n_fits,
                    "failed_runs": failed,
                    "divergences": None
                    if evaluation is None or evaluation.diagnostics is None
                    else evaluation.diagnostics.divergences,
                    "max_rhat": None
                    if evaluation is None or evaluation.diagnostics is None
                    else evaluation.diagnostics.max_rhat,
                    "min_bulk_ess": None
                    if evaluation is None or evaluation.diagnostics is None
                    else evaluation.diagnostics.min_bulk_ess,
                    "diagnostics_status": None
                    if evaluation is None or evaluation.diagnostics is None
                    else evaluation.diagnostics.status,
                    "diagnostics_reason": None
                    if evaluation is None or evaluation.diagnostics is None
                    else evaluation.diagnostics.reason,
                    "unit_scale": _max_unit_scale(evaluation),
                    "eligible": eligible,
                    "pareto": False,
                    "selected": False,
                    "_decision": decision,
                }
            )
    return rows


def _select(
    rows: list[dict[str, object]],
    criterion: PowerCriterion,
) -> tuple[dict[str, object] | None, str, str | None, str | None]:
    eligible = [row for row in rows if row["eligible"]]
    completed = [
        row
        for row in rows
        if row["evaluation_status"] == "completed" and row["power"] is not None
    ]
    closest = None
    if completed:
        closest = str(
            max(
                completed, key=lambda row: (float(row["power"]), -int(row["duration"]))
            )[  # type: ignore[arg-type]
                "candidate_id"
            ]
        )
    if not eligible:
        return None, "inconclusive", _redesign_reason(rows, criterion), closest
    chosen = min(
        eligible,
        key=lambda row: (
            float(row["cost"]),  # type: ignore[arg-type]
            -float(row["power"]),  # type: ignore[arg-type]
            str(row["candidate_id"]),
            int(row["duration"]),  # type: ignore[arg-type]
        ),
    )
    return chosen, "frozen", None, closest


def _eligible(
    status: str,
    power: float | None,
    point: PowerPoint | None,
    criterion: PowerCriterion,
    screen_row: pd.Series,
) -> bool:
    if status != "completed" or power is None or point is None:
        return False
    cost = _finite_or_none(screen_row["cost"])
    if cost is None:
        return False
    if power < criterion.target_power:
        return False
    false_positive = _finite_or_none(point.false_positive_rate)
    if false_positive is None or false_positive > criterion.false_positive_tolerance:
        return False
    return point.precision_met is not False


def _pareto(rows: list[dict[str, object]]) -> set[tuple[str, int]]:
    completed = [
        row
        for row in rows
        if row["evaluation_status"] == "completed" and not row["power_withheld"]
    ]
    kept: set[tuple[str, int]] = set()
    for row in completed:
        dominated = False
        for other in completed:
            if other is row:
                continue
            if _dominates(other, row):
                dominated = True
                break
        if not dominated:
            kept.add((str(row["candidate_id"]), int(row["duration"])))  # type: ignore[arg-type]
    return kept


def _dominates(left: dict[str, object], right: dict[str, object]) -> bool:
    """Return whether left is no worse on every axis and better on one.

    Power and outcome share improve when larger. Cost and interval width
    improve when smaller. A missing value is worse than any finite value
    and ties another missing value.
    """
    better = False
    for name, higher_is_better in (
        ("power", True),
        ("interval_width", False),
        ("cost", False),
        ("outcome_share", True),
    ):
        left_value = _finite_or_none(left[name])
        right_value = _finite_or_none(right[name])
        if left_value is None and right_value is None:
            continue
        if left_value is None:
            return False
        if right_value is None:
            better = True
            continue
        delta = left_value - right_value
        improved = delta > 0.0 if higher_is_better else delta < 0.0
        worse = delta < 0.0 if higher_is_better else delta > 0.0
        if worse:
            return False
        if improved:
            better = True
    return better


def _freeze(
    frame: pd.DataFrame,
    selected: dict[str, object],
    decision: DecisionRule,
    sample_kwargs: Mapping[str, object],
    caller_seed: int,
    horizons: tuple[int, ...],
    outcome_scale: str,
    time_column: str | None,
    scale: str,
    baseline: float | None,
) -> FrozenDesign:
    treated = tuple(selected["treated"])  # type: ignore[arg-type]
    donors = tuple(selected["donors"])  # type: ignore[arg-type]
    duration = int(selected["duration"])  # type: ignore[arg-type]
    identity = str(selected["candidate_id"])
    seed = sampler_seed(caller_seed, identity, horizons)
    settings = dict(sample_kwargs)
    settings["random_seed"] = seed
    spec = _build_handoff(
        frame,
        treated,
        donors,
        decision,
        estimator=str(selected["estimator"]),
        duration=duration,
        sample_kwargs=settings,
        outcome_scale=outcome_scale,
        time_column=time_column,
    )
    return FrozenDesign(
        candidate_id=identity,
        treated_units=treated,
        control_units=donors,
        duration=duration,
        estimator=str(selected["estimator"]),
        decision=decision,
        sample_kwargs=tuple(settings.items()),
        sampler_seed=seed,
        outcome_scale=outcome_scale,
        time_column=time_column,
        history=frame.copy(),
        training_index=tuple(spec.training_index),
        holdout_index=tuple(spec.holdout_index),
        scale=scale,
        baseline=baseline,
        cost=float(selected["cost"]),  # type: ignore[arg-type]
        power=float(selected["power"]),  # type: ignore[arg-type]
        false_positive_rate=float(selected["false_positive_rate"]),  # type: ignore[arg-type]
        interval_width=float(selected["interval_width"]),  # type: ignore[arg-type]
        monte_carlo_error=float(selected["monte_carlo_error"]),  # type: ignore[arg-type]
        selection_objective=SELECTION_OBJECTIVE,
    )


def _build_handoff(
    history: pd.DataFrame,
    treated: tuple[str, ...],
    donors: tuple[str, ...],
    decision: DecisionRule,
    *,
    estimator: str,
    duration: int,
    sample_kwargs: Mapping[str, object],
    outcome_scale: str,
    time_column: str | None,
) -> AnalysisHandoff | AdapterAnalysisHandoff:
    if estimator == "synthetic_control":
        plan = HistoryOnlyPlan(
            history,
            treated,
            donors,
            decision,
            estimator=estimator,
            sample_kwargs=sample_kwargs,
            time_column=time_column,
            outcome_scale=outcome_scale,
        )
        return plan.analysis_handoff(duration=duration)
    return adapter_analysis_handoff(
        history,
        treated,
        donors,
        decision,
        estimator=estimator,
        duration=duration,
        sample_kwargs=sample_kwargs,
        outcome_scale=outcome_scale,
        time_column=time_column,
    )


def _candidate_request(**kwargs: object) -> CandidateRequest:
    return CandidateRequest(
        units=tuple(kwargs["units"]),  # type: ignore[arg-type]
        min_group_size=int(kwargs["min_group_size"]),  # type: ignore[arg-type]
        max_group_size=int(kwargs["max_group_size"]),  # type: ignore[arg-type]
        history=kwargs["frame"],  # type: ignore[arg-type]
        selection_end=kwargs["selection_end"],
        random_seed=int(kwargs["random_seed"]),  # type: ignore[arg-type]
        treatment_eligible=None
        if kwargs["treatment_eligible"] is None
        else tuple(kwargs["treatment_eligible"]),  # type: ignore[arg-type]
        donor_eligible=None
        if kwargs["donor_eligible"] is None
        else tuple(kwargs["donor_eligible"]),  # type: ignore[arg-type]
        always_control=tuple(kwargs["always_control"]),  # type: ignore[arg-type]
        excluded_units=tuple(kwargs["excluded_units"]),  # type: ignore[arg-type]
        required_treated=tuple(kwargs["required_treated"]),  # type: ignore[arg-type]
        supplied_groups=None
        if kwargs["supplied_groups"] is None
        else tuple(tuple(group) for group in kwargs["supplied_groups"]),  # type: ignore[arg-type]
        fixed_treated=None
        if kwargs["fixed_treated"] is None
        else tuple(kwargs["fixed_treated"]),  # type: ignore[arg-type]
        donor_pools=None
        if kwargs["donor_pools"] is None
        else tuple(tuple(pool) for pool in kwargs["donor_pools"]),  # type: ignore[arg-type]
        spillover_exclusions=tuple(kwargs["spillover_exclusions"]),  # type: ignore[arg-type]
        min_remaining_donors=int(kwargs["min_remaining_donors"]),  # type: ignore[arg-type]
        min_outcome_share=kwargs["min_outcome_share"],  # type: ignore[arg-type]
        max_outcome_share=kwargs["max_outcome_share"],  # type: ignore[arg-type]
        unit_costs=tuple(kwargs["unit_costs"]),  # type: ignore[arg-type]
        budget=kwargs["cost_budget"],  # type: ignore[arg-type]
        unit_covariates=tuple(kwargs["unit_covariates"]),  # type: ignore[arg-type]
        balance_constraints=tuple(kwargs["balance_constraints"]),  # type: ignore[arg-type]
        exact_ceiling=int(kwargs["exact_ceiling"]),  # type: ignore[arg-type]
        max_generated=None
        if kwargs["max_generated"] is None
        else int(kwargs["max_generated"]),  # type: ignore[arg-type]
        shortlist_size=int(kwargs["shortlist_size"]),  # type: ignore[arg-type]
    )


def _size_one_reference_procedure(
    estimator: str,
    min_group_size: int,
    max_group_size: int,
    fixed_treated: Sequence[str] | None,
    supplied_groups: Sequence[Sequence[str]] | None,
) -> bool:
    """Return whether every candidate this call can study is size-one synthetic control."""
    fixed_is_size_one = fixed_treated is None or len(tuple(fixed_treated)) == 1
    groups_are_size_one = supplied_groups is None or all(
        len(tuple(group)) == 1 for group in supplied_groups
    )
    return (
        estimator == "synthetic_control"
        and min_group_size == 1
        and max_group_size == 1
        and fixed_is_size_one
        and groups_are_size_one
    )


def _reference_column(
    cache: dict[str, ReferencePowerEvaluator],
    history: pd.DataFrame,
    name: str,
    donors: tuple[str, ...],
    decision: DecisionRule,
    family: str,
    caller_seed: int,
    sample_kwargs: Mapping[str, object],
    replicate: int,
) -> pd.Series:
    """Draw one non-control column from the published size-one generator."""
    evaluator = cache.get(name)
    if evaluator is None:
        seed = sampler_seed(caller_seed, f"validation-panel:{name}", ())
        settings = dict(sample_kwargs)
        settings["random_seed"] = seed
        plan = HistoryOnlyPlan(
            history,
            (name,),
            donors,
            decision,
            estimator="synthetic_control",
            sample_kwargs=settings,
        )
        evaluator = ReferencePowerEvaluator(
            plan, history=history, scenario_family=family
        )
        cache[name] = evaluator
    return evaluator.untreated_panel(replicate)[name]


def _publish_study(outcome: _CandidateEvaluation) -> bool:
    """Keep a curve whose divergence integer is a recorded count.

    Size-one studies have no search diagnostic record. Their integer is the
    reference engine's own count. A grouped record with a missing count is
    not zero, and ``EffectByDuration.divergences`` cannot hold null.
    """
    if outcome.study is None:
        return False
    if outcome.diagnostics is None:
        return True
    return outcome.diagnostics.divergences is not None


def _validation_panel(
    history: pd.DataFrame,
    units: Sequence[str],
    always_control: Sequence[str],
    family: str,
    caller_seed: int,
    replicate: int,
    selection_end: object,
    *,
    decision: DecisionRule,
    sample_kwargs: Mapping[str, object],
    reference: bool,
    panel_cache: dict[str, ReferencePowerEvaluator],
) -> pd.DataFrame:
    """Draw the outer panel from the generator the inner path will use."""
    treated = tuple(name for name in units if name not in set(always_control))
    if not treated:
        raise ValueError(
            "validate_selection needs at least one unit outside always_control. "
            "No fit was started."
        )
    donors = tuple(always_control)
    if reference:
        panel = history.copy()
        for name in treated:
            panel[name] = _reference_column(
                panel_cache,
                history,
                name,
                donors,
                decision,
                family,
                caller_seed,
                sample_kwargs,
                replicate,
            )
        return panel
    seed = sampler_seed(caller_seed, json.dumps(list(treated)), (replicate,))
    return _grouped_panel(
        history,
        treated,
        donors,
        family,
        seed,
        replicate,
        _noise_scale(history, donors),
        selection_end,
    )


def _outer_seed(caller_seed: int, replicate: int) -> int:
    return sampler_seed(caller_seed, "validation-replicate", (replicate,))


def _training_diff(
    original: pd.DataFrame, injected: pd.DataFrame, training_index: tuple[object, ...]
) -> float:
    if not training_index:
        return 0.0
    before = original.loc[list(training_index)].to_numpy(dtype=float)
    after = injected.loc[list(training_index)].to_numpy(dtype=float)
    return float(np.max(np.abs(after - before)))


def _holdout_shift(
    original: pd.DataFrame, injected: pd.DataFrame, frozen: FrozenDesign
) -> float:
    before = original.loc[list(frozen.holdout_index), list(frozen.treated_units)]
    after = injected.loc[list(frozen.holdout_index), list(frozen.treated_units)]
    return float(np.mean(after.to_numpy(dtype=float) - before.to_numpy(dtype=float)))


def _optional_rate(flags: Sequence[bool]) -> float | None:
    if not flags:
        return None
    return float(np.mean(np.asarray(flags, dtype=float)))


def _optional_error(flags: Sequence[bool]) -> float | None:
    rate = _optional_rate(flags)
    if rate is None:
        return None
    return monte_carlo_error(rate, len(flags))


def _mean_error(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    array = np.asarray(values, dtype=float)
    return float(np.std(array, ddof=1) / np.sqrt(len(array)))


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
    return (
        f"{family}: each treated column is replaced. Donors stay observed. "
        f"Unit noise scales relative to donor sd are {scales}. Common-shock "
        f"share is {COMMON_SHOCK_SHARE}. persistent_misfit uses the published "
        "trend instead of the donor combination. Treatment is injected by "
        "translation after the untreated series exists."
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


def _noise_scale(history: pd.DataFrame, donors: Sequence[str]) -> float:
    matrix = history.loc[:, list(donors)].to_numpy(dtype=float)
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


def _point_or_none(
    evaluation: _CandidateEvaluation | None, duration: int, worthwhile: float
) -> PowerPoint | None:
    if evaluation is None or evaluation.study is None:
        return None
    matches = [
        item
        for item in evaluation.study.points
        if item.duration == duration and item.effect == worthwhile
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _mde_or_none(
    evaluation: _CandidateEvaluation | None, duration: int
) -> MinimumDetectableEffect | None:
    if evaluation is None or evaluation.study is None:
        return None
    matches = [
        item
        for item in evaluation.study.minimum_detectable_effects
        if item.duration == duration
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _row_status(
    screen_row: pd.Series, evaluation: _CandidateEvaluation | None
) -> tuple[str, str | None, int, int]:
    if screen_row["status"] == "filtered":
        reasons = screen_row["filter_reasons"]
        return "filtered", ",".join(str(item) for item in reasons), 0, 0
    if evaluation is None:
        return "not_shortlisted", "Not on the generator shortlist.", 0, 0
    failed = 0 if evaluation.study is None else int(evaluation.study.failed_runs)
    return evaluation.status, evaluation.reason, evaluation.n_fits, failed


def _max_unit_scale(evaluation: _CandidateEvaluation | None) -> float | None:
    if evaluation is None:
        return None
    return evaluation.unit_scale


_CAPABILITY_REFUSAL_MARKERS = (
    "Unknown estimator",
    "declared adapter seam",
    "reference adapter fits one treated unit",
    "reference adapter fits WeightedSumFitter",
    "Grouped qualification is not this issue",
    "supports at most",
    "does not allow a",
    "Other estimators are the",
)


def _capability_refusal(reason: object) -> bool:
    text = "" if reason is None else str(reason)
    return any(marker in text for marker in _CAPABILITY_REFUSAL_MARKERS)


def _rejected_design_reason(rows: list[dict[str, object]]) -> str:
    """Keep the estimator sentence only when every refusal is a capability."""
    distinct: list[str] = []
    for row in rows:
        if row["evaluation_status"] != "rejected" or not row["reason"]:
            continue
        text = str(row["reason"])
        if text not in distinct:
            distinct.append(text)
    if distinct and all(_capability_refusal(text) for text in distinct):
        return (
            "No supported design. The estimator refused the requested groups "
            "before sampling. No design was frozen."
        )
    detail = " ".join(distinct) if distinct else "The rows do not say why."
    return f"No supported design. {detail} No design was frozen."


def _redesign_reason(rows: list[dict[str, object]], criterion: PowerCriterion) -> str:
    statuses = {row["evaluation_status"] for row in rows}
    if statuses <= {"filtered"}:
        return (
            "No passing candidate. Widen eligibility or the donor pool. "
            "No design was frozen."
        )
    if any(row["evaluation_status"] == "skipped" for row in rows):
        return (
            "The fit budget stopped evaluation before every shortlisted "
            "design was studied. No design was frozen."
        )
    if "rejected" in statuses and not any(
        row["evaluation_status"] in {"completed", "withheld", "failed"} for row in rows
    ):
        return _rejected_design_reason(rows)
    if any(row["power_withheld"] for row in rows) and not any(
        row["evaluation_status"] == "completed" for row in rows
    ):
        return (
            "Every evaluated design withheld power. Do not launch on a "
            "pre-period misfit. No design was frozen."
        )
    completed = [
        row
        for row in rows
        if row["evaluation_status"] == "completed" and row["power"] is not None
    ]
    if not completed:
        return "No design produced a qualified power estimate. No design was frozen."
    best = max(completed, key=lambda row: float(row["power"]))  # type: ignore[arg-type]
    gaps: list[str] = []
    if float(best["power"]) < criterion.target_power:  # type: ignore[arg-type]
        gaps.append(
            f"best power {float(best['power']):.3f} is below "
            f"{criterion.target_power:.3f}"
        )
    false_positive = best["false_positive_rate"]
    if (
        false_positive is not None
        and float(false_positive) > criterion.false_positive_tolerance
    ):  # type: ignore[arg-type]
        gaps.append("false-positive rate exceeds tolerance")
    if best["precision_met"] is False:
        gaps.append("precision target was missed")
    if best["cost"] is None:
        gaps.append("cost is unknown, so the minimum-cost objective cannot choose")
    detail = "; ".join(gaps) if gaps else "the declared criteria were not met"
    return (
        f"No design met the declared criteria ({detail}). Lengthen the window, "
        "add donors, or reduce the worthwhile effect. No design was frozen."
    )


def _optimality(
    table_partial: bool,
    skipped: int,
    screen: pd.DataFrame,
    evaluations: Mapping[str, _CandidateEvaluation],
) -> str:
    if skipped or table_partial:
        return "budget_partial" if skipped else "capped"
    passing = int((screen["status"] != "filtered").sum())
    evaluated = sum(
        1
        for item in evaluations.values()
        if item.status in {"completed", "withheld", "failed", "rejected"}
    )
    if evaluated < passing:
        return "capped"
    return "shortlist"


def _planning_frame(history: pd.DataFrame, time_column: str | None) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame):
        raise TypeError("history must be a pandas DataFrame. No fit was started.")
    frame = history.copy()
    if time_column is not None:
        if time_column not in frame.columns:
            raise ValueError(
                f"history is missing time column {time_column!r}. No fit was started."
            )
        frame = frame.set_index(time_column)
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(
            "The planning index must be unique and increasing. No fit was started."
        )
    return frame


def _require_selection_end(
    frame: pd.DataFrame, selection_end: object, horizons: tuple[int, ...]
) -> None:
    labels = list(frame.index)
    if selection_end not in labels:
        raise ValueError(
            "selection_end is not an exact planning label. A partial date "
            "string is not a label. No fit was started."
        )
    earliest_holdout = labels[-max(horizons)]
    if labels.index(selection_end) >= labels.index(earliest_holdout):
        raise ValueError(
            "selection_end must be strictly before every requested holdout. "
            "Do not pass the truncated screen to the plan, and do not let the "
            "screen see the holdout. No fit was started."
        )


def _require_durations(durations: Sequence[int]) -> tuple[int, ...]:
    if isinstance(durations, (str, bytes)) or not isinstance(durations, Sequence):
        raise TypeError("durations must be a sequence of positive integers.")
    values = tuple(int(item) for item in durations)
    if not values or any(item <= 0 for item in values):
        raise ValueError("durations must be positive. No fit was started.")
    if len(values) != len(set(values)):
        raise ValueError("durations must be unique. No fit was started.")
    return values


def _require_sample_kwargs(sample_kwargs: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(sample_kwargs, Mapping):
        raise TypeError("sample_kwargs must be a mapping. No fit was started.")
    copied = dict(sample_kwargs)
    if "random_seed" in copied:
        raise ValueError(
            "Pass random_seed to search, not inside sample_kwargs. No fit was started."
        )
    for name in ("draws", "tune"):
        value = copied.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
        ):
            raise ValueError(
                f"sample_kwargs[{name!r}] must be a positive integer. "
                "No fit was started."
            )
    return copied


def _require_seed(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int. No fit was started.")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive. No fit was started.")
    return int(value)


def _require_finite(value: object, name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite. No fit was started.") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite. No fit was started.")
    return number


def _require_decision(decision: DecisionRule) -> None:
    if not isinstance(decision, DecisionRule):
        raise TypeError("decision must be a DecisionRule. No fit was started.")


def _require_criterion(criterion: PowerCriterion) -> None:
    if not isinstance(criterion, PowerCriterion):
        raise TypeError("criterion must be a PowerCriterion. No fit was started.")


def _require_family(family: str) -> str:
    if family not in SCENARIO_FAMILIES:
        raise ValueError(
            f"scenario_family must be one of {SCENARIO_FAMILIES}. Got {family!r}. "
            "No fit was started."
        )
    return family


def _fixed_design_criterion(criterion: PowerCriterion) -> PowerCriterion:
    if criterion.qualification_scope == "fixed_design":
        return criterion
    return PowerCriterion(
        target_power=criterion.target_power,
        false_positive_tolerance=criterion.false_positive_tolerance,
        precision_target=criterion.precision_target,
        monte_carlo_error_tolerance=criterion.monte_carlo_error_tolerance,
        qualification_scope="fixed_design",
    )


def _effect_grid(
    worthwhile: float, effect_grid: Sequence[float] | None
) -> tuple[float, ...]:
    if effect_grid is None:
        return (worthwhile,)
    values = tuple(_require_finite(item, "effect") for item in effect_grid)
    if worthwhile not in values:
        raise ValueError(
            "effect_grid must include the worthwhile effect. No fit was started."
        )
    return values


def _with_zero(effects: Sequence[float]) -> tuple[float, ...]:
    ordered: list[float] = []
    for effect in (0.0, *effects):
        if effect not in ordered:
            ordered.append(float(effect))
    return tuple(ordered)


def _finite_or_none(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


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
