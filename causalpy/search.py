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
"""Constrained single-target search for a fixed synthetic-control planner.

Screening uses a deterministic weighted-proportion holdout fit. The shortlist
is fitted by the existing duration comparison in candidate-id order, then
frozen and audited by the caller. A failed or unscored fit is not a handoff.
Nothing here selects a launch, a duration, or a global optimum.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from causalpy.duration import (
    DEFAULT_MAX_FITS,
    AnalysisSpecification,
    DurationComparison,
    DurationRow,
    _compare_durations,
)
from causalpy.historical_validation import _stored_labels
from causalpy.skl_models import WeightedProportion

if TYPE_CHECKING:
    from causalpy.design import SyntheticControlPlanner

__all__ = [
    "SHORTLIST_SIZE",
    "EvaluatedDesign",
    "ParetoPoint",
    "ScreenedCandidate",
    "SearchResult",
]

SHORTLIST_SIZE = 8
_SCREEN_HOLDOUT = 4
_PROXY_NAME = "weighted-proportion-holdout-rmse"
_SELECTION_POLICY = "proxy-rank-plus-boundary"
_GROUPS = (
    "Group size must be 1. Group requests are rejected; this search does not "
    "score joint assignments."
)
_NO_RECOMMENDATION = (
    "No supported recommendation. The Pareto set is the evaluated "
    "non-dominated set across historical detection share, supplied cost, and "
    "supplied relevance. It is not a global optimum, not an inferred CPIC or "
    "spend response, and not a launch recommendation. Screen scores are "
    "weighted-proportion holdout RMSE proxies, not the analysis."
)


@dataclass(frozen=True, eq=False)
class ScreenedCandidate:
    """One generated design and the screening decision made before fitting.

    Attributes
    ----------
    candidate_id : str
        Stable identity of the treated unit plus the sorted donor pool.
    treated_unit, donors : str, tuple of str
        The design that would be fitted. Donors are sorted by name so input
        order is not part of the identity.
    status : str
        ``skipped``, ``screen-failed``, ``screened``, ``unfinished``,
        ``evaluated``, or ``fit-failed``.
    reason : str
        Why the candidate stopped, or an empty string when it was evaluated.
    proxy_score : float, optional
        Holdout RMSE. Lower is a tighter in-sample match. Absent when the
        candidate was not screened.
    correlation_min, max_weight : float, optional
        Diagnostics. Neither is a veto.
    outcome_share, cost, relevance : float, optional
        Supplied-constraint measurements. Absent when that input was omitted.
    seed : int, optional
        Candidate seed. Origin seeds are derived from the same master and this
        identity; they are not this value.
    """

    candidate_id: str
    treated_unit: str
    donors: tuple[str, ...]
    status: str
    reason: str
    proxy_score: float | None = None
    correlation_min: float | None = None
    max_weight: float | None = None
    outcome_share: float | None = None
    cost: float | None = None
    relevance: float | None = None
    seed: int | None = None


@dataclass(frozen=True, eq=False)
class EvaluatedDesign:
    """A shortlist member that reached the duration comparison.

    Attributes
    ----------
    candidate_id, treated_unit, donors, seed
        The design and the candidate seed recorded on the comparison.
    comparison : DurationComparison
        The existing duration result. Its audit origins were not fitted.
    """

    candidate_id: str
    treated_unit: str
    donors: tuple[str, ...]
    seed: int
    comparison: DurationComparison


@dataclass(frozen=True, eq=False)
class ParetoPoint:
    """One evaluated duration on the sensitivity, cost, and relevance axes.

    Attributes
    ----------
    sensitivity : float
        Historical detection share at the target lift. Higher is better.
        A missing share is not a point. A failed or unscored row is not a
        point, even when its recorded share is zero.
    cost, relevance : float, optional
        Present only when the caller supplied that axis. Lower cost is better.
        Higher relevance is better.
    dominated : bool
        True when another point is at least as good on every present axis and
        strictly better on one. Equal points do not dominate each other.
    """

    candidate_id: str
    treated_unit: str
    duration: int
    sensitivity: float
    cost: float | None
    relevance: float | None
    dominated: bool


@dataclass(frozen=True, eq=False)
class SearchResult:
    """Screened designs, the evaluated shortlist, and an unaudited Pareto set.

    ``global_optimum`` is always false. ``freeze`` delegates to the chosen
    comparison and does not inspect any other candidate.

    Attributes
    ----------
    status : str
        ``budget-exhausted``, ``fit-failed``, ``mixed``,
        ``historical-descriptive``, ``insufficient-history``, or
        ``memory-exhausted``.
    mode : {"fixed-target", "free-target"}
        Which candidate space was enumerated.
    evaluation_exhaustive : bool
        True only when every screened candidate was fitted. Recall and regret
        are withheld otherwise.
    proxy_recall, proxy_regret : float, optional
        Recall is 1 when the lowest proxy RMSE is among the best evaluated
        sensitivities, else 0. Regret is the sensitivity gap between that
        proxy-best candidate and the best evaluated sensitivity. Both are
        None when evaluation was partial.
    """

    status: str
    mode: Literal["fixed-target", "free-target"]
    candidates: tuple[ScreenedCandidate, ...]
    evaluated: tuple[EvaluatedDesign, ...]
    pareto: tuple[ParetoPoint, ...]
    n_screened: int
    n_evaluated: int
    n_skipped: int
    n_failed: int
    n_unfinished: int
    n_inference_fits: int
    budget_exhausted: bool
    evaluation_exhaustive: bool
    proxy_recall: float | None
    proxy_regret: float | None
    partial_reason: str | None
    selection_policy: str
    disclosure: str
    recommendation_text: str
    sample_kwargs: dict[str, Any]
    random_seed: int

    @property
    def global_optimum(self) -> bool:
        """Search never claims a global optimum."""
        return False

    def recommendation(self) -> str:
        """Return the withheld-launch text. It does not name a winner."""
        return self.recommendation_text

    @property
    def summary(self) -> pd.DataFrame:
        """One row per generated candidate, including skipped and unfinished."""
        rows = [
            {
                "candidate_id": candidate.candidate_id,
                "treated_unit": candidate.treated_unit,
                "donors": ",".join(candidate.donors),
                "status": candidate.status,
                "reason": candidate.reason,
                "proxy_name": _PROXY_NAME if candidate.proxy_score is not None else "",
                "proxy_score": candidate.proxy_score,
                "correlation_min": candidate.correlation_min,
                "max_weight": candidate.max_weight,
                "outcome_share": candidate.outcome_share,
                "cost": candidate.cost,
                "relevance": candidate.relevance,
                "seed": candidate.seed,
            }
            for candidate in self.candidates
        ]
        return pd.DataFrame(rows)

    @property
    def pareto_table(self) -> pd.DataFrame:
        """Evaluated points, including dominated ones. Missing axes are omitted."""
        rows = [
            {
                "candidate_id": point.candidate_id,
                "treated_unit": point.treated_unit,
                "duration": point.duration,
                "sensitivity": point.sensitivity,
                "cost": point.cost,
                "relevance": point.relevance,
                "dominated": point.dominated,
            }
            for point in self.pareto
        ]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        drop = [
            column for column in ("cost", "relevance") if frame[column].isna().all()
        ]
        return frame.drop(columns=drop)

    def freeze(self, candidate_id: str, duration: int) -> AnalysisSpecification:
        """Freeze one evaluated candidate. Other candidates are not consulted.

        Parameters
        ----------
        candidate_id : str
            An evaluated design. A skipped, failed, or unfinished id raises.
        duration : int
            A scored duration on that design. A failed or unscored duration
            raises. This method does not choose a duration.

        Returns
        -------
        AnalysisSpecification
            The duration handoff. Call its ``audit`` separately; search does
            not fit reserved origins.
        """
        match = next(
            (item for item in self.evaluated if item.candidate_id == candidate_id),
            None,
        )
        if match is None:
            raise ValueError(
                f"Candidate {candidate_id!r} has no evaluated comparison to freeze. "
                "Search does not select a replacement winner."
            )
        row = next(
            (item for item in match.comparison.rows if item.duration == duration),
            None,
        )
        if row is None or row.evidence_level != "historical-descriptive":
            raise ValueError(
                f"Candidate {candidate_id!r} duration {duration} is not an assessed "
                "design. A failed or unscored fit is not a handoff, and search does "
                "not select a replacement."
            )
        return match.comparison.freeze(duration)


def _search_designs(
    planner: SyntheticControlPlanner,
    durations: Sequence[int],
    selection_origins: Sequence[Any],
    *,
    treated_unit: str | None = None,
    eligible_units: Sequence[str] | None = None,
    donor_pools: Sequence[Sequence[str]] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
    spillover_exclude: Sequence[str] = (),
    min_donors: int = 1,
    costs: Mapping[str, float] | None = None,
    max_cost: float | None = None,
    outcome_share: tuple[float, float] | None = None,
    representativeness: Mapping[str, float] | None = None,
    min_representativeness: float | None = None,
    group_size: int = 1,
    shortlist_size: int = SHORTLIST_SIZE,
    max_screened: int | None = None,
    screen_holdout: int = _SCREEN_HOLDOUT,
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
) -> SearchResult:
    """Enumerate one declared candidate space and fit its shortlist.

    The planner's constructor target is the fixed-target default. Pass
    ``eligible_units`` for free-target search. Do not pass both.
    """
    _reject_groups(group_size, treated_unit, eligible_units)
    if planner.random_seed is None:
        raise ValueError(
            "Search requires planner.random_seed so candidate and origin seeds "
            "are defined. An unseeded search is rejected."
        )
    if sample_kwargs is not None and "random_seed" in sample_kwargs:
        raise ValueError(
            "Pass the master seed as planner.random_seed. Search derives "
            "candidate and origin seeds from it; sample_kwargs['random_seed'] "
            "would override that key."
        )
    _require_positive_int(shortlist_size, "shortlist_size")
    _require_positive_int(screen_holdout, "screen_holdout")
    _require_positive_int(min_donors, "min_donors")
    if max_screened is not None:
        _require_positive_int(max_screened, "max_screened")
    if max_cost is not None and costs is None:
        raise ValueError(
            "max_cost requires a costs mapping. Missing costs are not zero."
        )
    if min_representativeness is not None and representativeness is None:
        raise ValueError(
            "min_representativeness requires a representativeness mapping. "
            "The score is not inferred."
        )
    share_bounds = _share_bounds(outcome_share)
    cost_map = None if costs is None else _number_map(costs, "costs", minimum=0.0)
    relevance_map = (
        None
        if representativeness is None
        else _number_map(representativeness, "representativeness")
    )
    include_names = None if include is None else _string_tuple(include, "include")
    exclude_names = _string_tuple(exclude, "exclude")
    spillover_names = _string_tuple(spillover_exclude, "spillover_exclude")
    _reject_overlap(include_names, exclude_names, spillover_names)
    columns = _column_names(planner.history)
    _require_known(columns, include_names, exclude_names, spillover_names)
    mode, specs = _candidate_specs(
        planner,
        treated_unit=treated_unit,
        eligible_units=eligible_units,
        donor_pools=donor_pools,
        columns=columns,
        include=include_names,
        exclude=exclude_names,
        spillover=spillover_names,
    )
    specs, duplicate_note = _collapse_specs(specs)
    universe = _universe(columns, include_names, exclude_names, spillover_names)
    selection = _selection_frame(planner.history, selection_origins, exclusions)
    share_values = None if share_bounds is None else _shares(selection, universe)
    screened, disclosure = _screen_candidates(
        specs,
        selection=selection,
        holdout=screen_holdout,
        min_donors=min_donors,
        include=include_names,
        exclude=exclude_names,
        spillover=spillover_names,
        columns=columns,
        costs=cost_map,
        max_cost=max_cost,
        shares=share_values,
        share_bounds=share_bounds,
        relevance=relevance_map,
        min_representativeness=min_representativeness,
        max_screened=max_screened,
        master_seed=planner.random_seed,
    )
    if duplicate_note:
        disclosure = _join_notes(duplicate_note, disclosure)
    shortlist, shortlist_note = _shortlist(
        [item for item in screened if item.status == "screened"],
        shortlist_size,
    )
    if shortlist_note:
        disclosure = _join_notes(disclosure, shortlist_note)
    evaluated, finished, stop_status, stop_note = _evaluate_shortlist(
        planner,
        shortlist,
        durations=durations,
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
    if stop_note:
        disclosure = _join_notes(disclosure, stop_note)
    candidates = _apply_evaluation(
        screened,
        finished,
        {item.candidate_id for item in shortlist},
        stop_status,
    )
    return _assemble(
        mode=mode,
        candidates=candidates,
        evaluated=evaluated,
        shortlist_ids={item.candidate_id for item in shortlist},
        stop_status=stop_status,
        disclosure=disclosure,
        sample_kwargs=sample_kwargs,
        random_seed=planner.random_seed,
        costs_supplied=cost_map is not None,
        relevance_supplied=relevance_map is not None,
    )


def _reject_groups(
    group_size: int,
    treated_unit: object,
    eligible_units: object,
) -> None:
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise TypeError("group_size must be the integer 1.")
    if group_size != 1:
        raise ValueError(_GROUPS)
    if treated_unit is not None and not isinstance(treated_unit, str):
        raise TypeError(_GROUPS)
    if eligible_units is None:
        return
    if isinstance(eligible_units, str) or not isinstance(eligible_units, Sequence):
        raise TypeError("eligible_units must be a sequence of unit names.")
    if any(not isinstance(unit, str) for unit in eligible_units):
        raise TypeError(_GROUPS)


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer of at least 1.")


def _share_bounds(
    bounds: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if bounds is None:
        return None
    if isinstance(bounds, str) or not isinstance(bounds, tuple) or len(bounds) != 2:
        raise TypeError("outcome_share must be a (min, max) pair of finite floats.")
    lower, upper = bounds
    if isinstance(lower, bool) or isinstance(upper, bool):
        raise TypeError("outcome_share bounds must be floats, not booleans.")
    if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
        raise TypeError("outcome_share bounds must be floats.")
    lower_f, upper_f = float(lower), float(upper)
    if not np.isfinite(lower_f) or not np.isfinite(upper_f) or lower_f > upper_f:
        raise ValueError("outcome_share bounds must be finite and min <= max.")
    return lower_f, upper_f


def _number_map(
    values: Mapping[str, float],
    name: str,
    *,
    minimum: float | None = None,
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping of unit names to finite numbers.")
    cleaned: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be unit names.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name}[{key!r}] must be a finite number.")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{name}[{key!r}] must be finite.")
        if minimum is not None and number < minimum:
            raise ValueError(f"{name}[{key!r}] must be at least {minimum}.")
        cleaned[key] = number
    return cleaned


def _string_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of unit names.")
    names = tuple(values)
    if not names:
        return ()
    if any(not isinstance(item, str) for item in names):
        raise TypeError(f"{name} must contain unit names. Grouped units are rejected.")
    if len(names) != len(set(names)):
        raise ValueError(f"{name} contains duplicate units.")
    return names


def _reject_overlap(
    include: tuple[str, ...] | None,
    exclude: tuple[str, ...],
    spillover: tuple[str, ...],
) -> None:
    excluded = set(exclude)
    buffered = set(spillover)
    if include is not None and set(include) & excluded:
        raise ValueError(
            "include and exclude overlap. A unit cannot be required and removed."
        )
    if include is not None and set(include) & buffered:
        raise ValueError(
            "include and spillover_exclude overlap. A buffer cannot be an eligible unit."
        )
    if excluded & buffered:
        raise ValueError(
            "exclude and spillover_exclude overlap. Name a unit in one list so the "
            "skip reason is unambiguous."
        )


def _column_names(history: pd.DataFrame) -> tuple[str, ...]:
    names = tuple(history.columns)
    if any(not isinstance(name, str) for name in names):
        raise TypeError("History columns must be unit names.")
    return names


def _require_known(
    columns: tuple[str, ...],
    include: tuple[str, ...] | None,
    exclude: tuple[str, ...],
    spillover: tuple[str, ...],
) -> None:
    known = set(columns)
    unknown = [
        name
        for name in (
            *(include or ()),
            *exclude,
            *spillover,
        )
        if name not in known
    ]
    if unknown:
        raise ValueError(
            f"Unknown units {unknown}. Constraints are not inferred for units "
            "outside the history."
        )


def _candidate_specs(
    planner: SyntheticControlPlanner,
    *,
    treated_unit: str | None,
    eligible_units: Sequence[str] | None,
    donor_pools: Sequence[Sequence[str]] | None,
    columns: tuple[str, ...],
    include: tuple[str, ...] | None,
    exclude: tuple[str, ...],
    spillover: tuple[str, ...],
) -> tuple[
    Literal["fixed-target", "free-target"], tuple[tuple[str, tuple[str, ...]], ...]
]:
    if eligible_units is not None and treated_unit is not None:
        raise ValueError(
            "Pass eligible_units for free-target search or treated_unit for "
            "fixed-target donor alternatives, not both."
        )
    known = set(columns)
    if eligible_units is not None:
        units = _string_tuple(eligible_units, "eligible_units")
        _require_members(known, units, "eligible_units")
        pools = _parse_pools(donor_pools, known)
        if len(pools) > 1:
            raise ValueError(
                "Free-target search accepts one donor pool, applied to every "
                "target, or none. Multiple pools are fixed-target alternatives."
            )
        base = pools[0] if pools else units
        specs = tuple(
            (target, _donors(base, target, include, exclude, spillover))
            for target in units
        )
        return "free-target", specs
    target = treated_unit if treated_unit is not None else planner.treated_unit
    if target not in known:
        raise ValueError(f"Unknown treated unit {target!r}.")
    pools = _parse_pools(donor_pools, known)
    if not pools:
        pools = (planner.control_units,)
    specs = tuple(
        (target, _donors(pool, target, include, exclude, spillover)) for pool in pools
    )
    return "fixed-target", specs


def _collapse_specs(
    specs: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], str]:
    seen: set[str] = set()
    unique: list[tuple[str, tuple[str, ...]]] = []
    for target, donors in specs:
        identity = _candidate_id(target, donors)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append((target, donors))
    note = ""
    if len(unique) < len(specs):
        note = (
            "Donor pools that matched after sorting and exclusions were "
            "collapsed. Input order did not create extra candidates."
        )
    return tuple(unique), note


def _parse_pools(
    donor_pools: Sequence[Sequence[str]] | None,
    known: set[str],
) -> tuple[tuple[str, ...], ...]:
    if donor_pools is None:
        return ()
    if isinstance(donor_pools, str) or not isinstance(donor_pools, Sequence):
        raise TypeError("donor_pools must be a sequence of donor-name sequences.")
    pools = tuple(_string_tuple(pool, "donor_pools") for pool in donor_pools)
    for pool in pools:
        _require_members(known, pool, "donor_pools")
    return pools


def _require_members(known: set[str], names: Sequence[str], label: str) -> None:
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"Unknown units {unknown} in {label}.")


def _donors(
    pool: Sequence[str],
    target: str,
    include: tuple[str, ...] | None,
    exclude: tuple[str, ...],
    spillover: tuple[str, ...],
) -> tuple[str, ...]:
    banned = set(exclude) | set(spillover) | {target}
    allowed = None if include is None else set(include)
    kept = [
        name
        for name in pool
        if name not in banned and (allowed is None or name in allowed)
    ]
    return tuple(sorted(set(kept)))


def _universe(
    columns: tuple[str, ...],
    include: tuple[str, ...] | None,
    exclude: tuple[str, ...],
    spillover: tuple[str, ...],
) -> tuple[str, ...]:
    banned = set(exclude) | set(spillover)
    source = columns if include is None else include
    return tuple(name for name in source if name not in banned)


def _selection_frame(
    history: pd.DataFrame,
    origins: Sequence[Any],
    exclusions: Sequence[Any],
) -> pd.DataFrame:
    if (
        isinstance(origins, str)
        or not isinstance(origins, Sequence)
        or len(origins) == 0
    ):
        raise ValueError(
            "selection_origins must be a non-empty sequence of index labels."
        )
    if isinstance(exclusions, (str, bytes)) or not isinstance(exclusions, Sequence):
        raise TypeError(
            "exclusions must be a sequence of index labels, not a single string."
        )
    positions = history.index.get_indexer(list(origins))
    if (positions < 0).any():
        missing = [
            origin
            for origin, position in zip(origins, positions, strict=True)
            if position < 0
        ]
        raise ValueError(f"Unknown selection origins {missing}.")
    earliest = int(positions.min())
    # The evaluator resolves aliases through the index. Raw set membership
    # keeps a timestamp string in the screening holdout that validation drops.
    excluded = set(_stored_labels(history.index, tuple(exclusions)))
    labels = [label for label in history.index[:earliest] if label not in excluded]
    return history.loc[labels]


def _shares(selection: pd.DataFrame, universe: Sequence[str]) -> dict[str, float]:
    if selection.empty:
        raise ValueError(
            "Outcome share needs selection-history rows before the earliest "
            "selection origin. There are none."
        )
    totals = selection.loc[:, list(universe)].sum(axis=0)
    if not np.isfinite(totals.to_numpy(dtype=float)).all():
        raise ValueError(
            "Outcome share received non-finite selection-history outcomes."
        )
    denominator = float(totals.sum())
    if denominator <= 0:
        raise ValueError(
            "Outcome share needs a positive sum of selection-history outcomes "
            "across the eligible universe. A non-positive total is an error, "
            "not a signed convention."
        )
    return {name: float(totals[name] / denominator) for name in universe}


def _screen_candidates(
    specs: Sequence[tuple[str, tuple[str, ...]]],
    *,
    selection: pd.DataFrame,
    holdout: int,
    min_donors: int,
    include: tuple[str, ...] | None,
    exclude: tuple[str, ...],
    spillover: tuple[str, ...],
    columns: tuple[str, ...],
    costs: dict[str, float] | None,
    max_cost: float | None,
    shares: dict[str, float] | None,
    share_bounds: tuple[float, float] | None,
    relevance: dict[str, float] | None,
    min_representativeness: float | None,
    max_screened: int | None,
    master_seed: int,
) -> tuple[tuple[ScreenedCandidate, ...], str]:
    generated = [_spec_candidate(target, donors) for target, donors in specs]
    notes: list[str] = []
    constrained: list[ScreenedCandidate] = []
    feasible: list[ScreenedCandidate] = []
    for item in generated:
        reason = _constraint_reason(
            item.treated_unit,
            item.donors,
            min_donors=min_donors,
            include=include,
            exclude=exclude,
            spillover=spillover,
            columns=columns,
            costs=costs,
            max_cost=max_cost,
            shares=shares,
            share_bounds=share_bounds,
            relevance=relevance,
            min_representativeness=min_representativeness,
        )
        if reason is None:
            feasible.append(item)
            continue
        constrained.append(
            _replace(
                item,
                status="skipped",
                reason=reason,
                **_measurements(item.treated_unit, costs, shares, relevance),
            )
        )
    if max_screened is not None and len(feasible) > max_screened:
        ranked = sorted(feasible, key=lambda item: _stable_hash(item.candidate_id))
        kept = {item.candidate_id for item in ranked[:max_screened]}
        feasible = [
            item
            if item.candidate_id in kept
            else _replace(
                item,
                status="skipped",
                reason=(
                    f"screening cap {max_screened}: kept by stable candidate-id "
                    "hash, not input order"
                ),
            )
            for item in feasible
        ]
        notes.append(
            f"Screening was capped at {max_screened} of {len(feasible)} "
            "candidates that passed include, exclude, spillover, donor, cost, "
            "share, and representativeness constraints. The kept set is the "
            "stable hash of candidate ids, not the first candidates in input "
            "order."
        )
    prepared = {item.candidate_id: item for item in (*constrained, *feasible)}
    generated = [prepared[item.candidate_id] for item in generated]
    screened = [
        _screen_one(
            item,
            selection=selection,
            holdout=holdout,
            min_donors=min_donors,
            include=include,
            exclude=exclude,
            spillover=spillover,
            columns=columns,
            costs=costs,
            max_cost=max_cost,
            shares=shares,
            share_bounds=share_bounds,
            relevance=relevance,
            min_representativeness=min_representativeness,
            master_seed=master_seed,
        )
        for item in generated
    ]
    return tuple(screened), " ".join(notes)


def _spec_candidate(target: str, donors: tuple[str, ...]) -> ScreenedCandidate:
    return ScreenedCandidate(
        candidate_id=_candidate_id(target, donors),
        treated_unit=target,
        donors=donors,
        status="pending",
        reason="",
    )


def _candidate_id(target: str, donors: tuple[str, ...]) -> str:
    return json.dumps([target, list(donors)], separators=(",", ":"))


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _screen_one(
    candidate: ScreenedCandidate,
    *,
    selection: pd.DataFrame,
    holdout: int,
    min_donors: int,
    include: tuple[str, ...] | None,
    exclude: tuple[str, ...],
    spillover: tuple[str, ...],
    columns: tuple[str, ...],
    costs: dict[str, float] | None,
    max_cost: float | None,
    shares: dict[str, float] | None,
    share_bounds: tuple[float, float] | None,
    relevance: dict[str, float] | None,
    min_representativeness: float | None,
    master_seed: int,
) -> ScreenedCandidate:
    if candidate.status == "skipped":
        return candidate
    reason = _constraint_reason(
        candidate.treated_unit,
        candidate.donors,
        min_donors=min_donors,
        include=include,
        exclude=exclude,
        spillover=spillover,
        columns=columns,
        costs=costs,
        max_cost=max_cost,
        shares=shares,
        share_bounds=share_bounds,
        relevance=relevance,
        min_representativeness=min_representativeness,
    )
    measurements = _measurements(candidate.treated_unit, costs, shares, relevance)
    if reason is not None:
        return _replace(candidate, status="skipped", reason=reason, **measurements)
    proxy = _proxy(selection, candidate.treated_unit, candidate.donors, holdout)
    seed = _keyed_seed(master_seed, candidate.candidate_id)
    if proxy.reason is not None:
        return _replace(
            candidate,
            status="screen-failed",
            reason=proxy.reason,
            seed=seed,
            **measurements,
        )
    return _replace(
        candidate,
        status="screened",
        reason="",
        proxy_score=proxy.score,
        correlation_min=proxy.correlation_min,
        max_weight=proxy.max_weight,
        seed=seed,
        **measurements,
    )


def _constraint_reason(
    target: str,
    donors: tuple[str, ...],
    *,
    min_donors: int,
    include: tuple[str, ...] | None,
    exclude: tuple[str, ...],
    spillover: tuple[str, ...],
    columns: tuple[str, ...],
    costs: dict[str, float] | None,
    max_cost: float | None,
    shares: dict[str, float] | None,
    share_bounds: tuple[float, float] | None,
    relevance: dict[str, float] | None,
    min_representativeness: float | None,
) -> str | None:
    known = set(columns)
    if target not in known:
        raise ValueError(f"Unknown treated unit {target!r}.")
    if include is not None and target not in include:
        return "not in include"
    if target in exclude:
        return "exclude"
    if target in spillover:
        return "spillover-exclude"
    if len(donors) < min_donors:
        return f"fewer than {min_donors} donors after exclusions"
    if costs is not None and target not in costs:
        raise ValueError(
            f"costs has no entry for candidate target {target!r}. Missing costs "
            "are an error, not an imputation."
        )
    if max_cost is not None and costs is not None and costs[target] > max_cost:
        return f"cost {costs[target]} exceeds max_cost {max_cost}"
    if share_bounds is not None and shares is not None:
        share = shares[target]
        lower, upper = share_bounds
        if share < lower or share > upper:
            return f"outcome share {share} is outside [{lower}, {upper}]"
    if relevance is not None and target not in relevance:
        raise ValueError(
            f"representativeness has no entry for candidate target {target!r}. "
            "The score is not inferred."
        )
    if (
        min_representativeness is not None
        and relevance is not None
        and relevance[target] < min_representativeness
    ):
        return (
            f"representativeness {relevance[target]} is below {min_representativeness}"
        )
    return None


def _measurements(
    target: str,
    costs: dict[str, float] | None,
    shares: dict[str, float] | None,
    relevance: dict[str, float] | None,
) -> dict[str, float | None]:
    return {
        "cost": None if costs is None else costs.get(target),
        "outcome_share": None if shares is None else shares.get(target),
        "relevance": None if relevance is None else relevance.get(target),
    }


@dataclass(frozen=True, eq=False)
class _Proxy:
    score: float | None = None
    correlation_min: float | None = None
    max_weight: float | None = None
    reason: str | None = None


def _proxy(
    selection: pd.DataFrame,
    target: str,
    donors: tuple[str, ...],
    holdout: int,
) -> _Proxy:
    if len(selection) < holdout + 2:
        return _Proxy(
            reason="selection history is shorter than the holdout plus two training rows"
        )
    train = selection.iloc[:-holdout]
    held = selection.iloc[-holdout:]
    if len(train) < 2 or held.empty:
        return _Proxy(reason="selection history cannot support a holdout proxy")
    y = train[target].to_numpy(dtype=float)
    x = train.loc[:, list(donors)].to_numpy(dtype=float)
    held_y = held[target].to_numpy(dtype=float)
    held_x = held.loc[:, list(donors)].to_numpy(dtype=float)
    if not np.isfinite(np.concatenate([y, held_y, x.ravel(), held_x.ravel()])).all():
        return _Proxy(reason="non-finite selection history")
    model = WeightedProportion()
    try:
        model.fit(x, y)
        coef = np.asarray(model.coef_, dtype=float).reshape(-1)
    except Exception as exc:
        return _Proxy(reason=f"screen-failed: {type(exc).__name__}: {exc}")
    if coef.shape != (len(donors),) or not np.isfinite(coef).all():
        return _Proxy(reason="screen-failed: non-finite weights")
    if np.any(coef < -1e-6) or abs(float(coef.sum()) - 1.0) > 1e-4:
        return _Proxy(reason="screen-failed: weights are not a convex combination")
    predicted = held_x @ coef
    rmse = float(np.sqrt(np.mean((held_y - predicted) ** 2)))
    if not np.isfinite(rmse):
        return _Proxy(reason="screen-failed: non-finite holdout RMSE")
    return _Proxy(
        score=rmse,
        correlation_min=_min_correlation(train, target, donors),
        max_weight=float(np.max(coef)),
    )


def _min_correlation(
    train: pd.DataFrame, target: str, donors: tuple[str, ...]
) -> float | None:
    target_values = train[target].to_numpy(dtype=float)
    if float(np.std(target_values)) == 0.0:
        return None
    coefficients: list[float] = []
    for donor in donors:
        donor_values = train[donor].to_numpy(dtype=float)
        if float(np.std(donor_values)) == 0.0:
            continue
        coefficient = float(np.corrcoef(target_values, donor_values)[0, 1])
        if np.isfinite(coefficient):
            coefficients.append(coefficient)
    if not coefficients:
        return None
    return float(min(coefficients))


def _keyed_seed(master: int, *parts: object) -> int:
    payload = json.dumps([master, *parts], separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode()).digest()
    value = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    return value if value > 0 else 1


def _replace(candidate: ScreenedCandidate, **changes: Any) -> ScreenedCandidate:
    data = {
        "candidate_id": candidate.candidate_id,
        "treated_unit": candidate.treated_unit,
        "donors": candidate.donors,
        "status": candidate.status,
        "reason": candidate.reason,
        "proxy_score": candidate.proxy_score,
        "correlation_min": candidate.correlation_min,
        "max_weight": candidate.max_weight,
        "outcome_share": candidate.outcome_share,
        "cost": candidate.cost,
        "relevance": candidate.relevance,
        "seed": candidate.seed,
    }
    data.update(changes)
    return ScreenedCandidate(**data)


_EVALUATION_ORDER = "The shortlist is fitted in candidate-id order, not input order."


def _shortlist(
    screened: Sequence[ScreenedCandidate],
    size: int,
) -> tuple[tuple[ScreenedCandidate, ...], str]:
    if not screened:
        return (), ""
    if len(screened) <= size:
        ordered = tuple(sorted(screened, key=lambda item: item.candidate_id))
        return ordered, _EVALUATION_ORDER
    chosen: dict[str, ScreenedCandidate] = {}
    boundaries = _boundary_candidates(screened)
    for candidate in boundaries:
        if len(chosen) >= size:
            break
        chosen.setdefault(candidate.candidate_id, candidate)
    remainder = sorted(
        (item for item in screened if item.candidate_id not in chosen),
        key=lambda item: (item.proxy_score, item.candidate_id),
    )
    for candidate in remainder:
        if len(chosen) >= size:
            break
        chosen[candidate.candidate_id] = candidate
    ordered = tuple(sorted(chosen.values(), key=lambda item: item.candidate_id))
    note = (
        f"Shortlist size is {size} of {len(screened)} screened candidates. "
        "Boundary candidates are kept first, then the lowest proxy scores. "
        "Ties break on candidate id. Input order is not the policy. "
        f"{_EVALUATION_ORDER} "
        "Candidates outside the shortlist were screened and not evaluated."
    )
    if len(boundaries) > size:
        note += (
            " Boundary candidates exceeded the shortlist, so the kept boundary "
            "set follows lowest cost, then highest relevance, then lowest and "
            "highest outcome share, then lowest proxy score."
        )
    return ordered, note


def _boundary_candidates(
    screened: Sequence[ScreenedCandidate],
) -> tuple[ScreenedCandidate, ...]:
    picks: list[ScreenedCandidate] = []

    def add(candidate: ScreenedCandidate) -> None:
        if all(candidate.candidate_id != item.candidate_id for item in picks):
            picks.append(candidate)

    costs = [item for item in screened if item.cost is not None]
    if costs:
        add(min(costs, key=lambda item: (item.cost, item.candidate_id)))
    relevance = [item for item in screened if item.relevance is not None]
    if relevance:
        add(max(relevance, key=lambda item: (item.relevance, item.candidate_id)))
    shares = [item for item in screened if item.outcome_share is not None]
    if shares:
        add(min(shares, key=lambda item: (item.outcome_share, item.candidate_id)))
        add(max(shares, key=lambda item: (item.outcome_share, item.candidate_id)))
    add(min(screened, key=lambda item: (item.proxy_score, item.candidate_id)))
    return tuple(picks)


def _evaluate_shortlist(
    planner: SyntheticControlPlanner,
    shortlist: Sequence[ScreenedCandidate],
    *,
    durations: Sequence[int],
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
) -> tuple[tuple[EvaluatedDesign, ...], dict[str, EvaluatedDesign], str | None, str]:
    master = planner.random_seed
    if master is None:
        raise RuntimeError("search rejected an unseeded planner.")
    remaining = max_fits
    evaluated: list[EvaluatedDesign] = []
    finished: dict[str, EvaluatedDesign] = {}
    stop_status: str | None = None
    stop_note = ""
    for candidate in shortlist:
        if remaining <= 0:
            stop_status = "budget-exhausted"
            stop_note = (
                "The fit budget was exhausted before the shortlist finished. "
                "Unfitted members follow candidate-id order, not input order. "
                "They are unfinished evidence, not rejections, and no winner "
                "was selected."
            )
            break
        if candidate.seed is None:
            raise RuntimeError("A screened candidate reached fitting without a seed.")
        child = _child_planner(planner, candidate)
        comparison = _compare_durations(
            child,
            durations,
            selection_origins=selection_origins,
            audit_origins=audit_origins,
            relative_lifts=relative_lifts,
            absolute_lifts=absolute_lifts,
            target_lift=target_lift,
            target_scale=target_scale,
            exclusions=exclusions,
            cadence=cadence,
            max_fits=remaining,
            fit_cost=fit_cost,
            min_training_periods=min_training_periods,
            sample_kwargs=sample_kwargs,
            window_seed=_origin_seed(master, candidate.candidate_id),
        )
        if comparison.status in {"budget-exhausted", "memory-exhausted"} and (
            comparison.n_inference_fits == 0
        ):
            stop_status = comparison.status
            stop_note = comparison.recommendation()
            break
        evaluated.append(
            EvaluatedDesign(
                candidate_id=candidate.candidate_id,
                treated_unit=candidate.treated_unit,
                donors=candidate.donors,
                seed=candidate.seed,
                comparison=comparison,
            )
        )
        finished[candidate.candidate_id] = evaluated[-1]
        remaining -= comparison.n_inference_fits
    return tuple(evaluated), finished, stop_status, stop_note


def _origin_seed(master: int, candidate_id: str) -> Callable[[Any], int]:
    def seed(origin: Any) -> int:
        return _keyed_seed(
            master,
            candidate_id,
            "origin",
            type(origin).__name__,
            str(origin),
        )

    return seed


def _child_planner(
    planner: SyntheticControlPlanner,
    candidate: ScreenedCandidate,
) -> SyntheticControlPlanner:
    if candidate.seed is None:
        raise RuntimeError("A screened candidate reached fitting without a seed.")
    return planner._rebind(candidate.treated_unit, candidate.donors, candidate.seed)


def _apply_evaluation(
    screened: Sequence[ScreenedCandidate],
    finished: Mapping[str, EvaluatedDesign],
    shortlist_ids: set[str],
    stop_status: str | None,
) -> tuple[ScreenedCandidate, ...]:
    stopped = stop_status in {"budget-exhausted", "memory-exhausted"}
    updated: list[ScreenedCandidate] = []
    for candidate in screened:
        design = finished.get(candidate.candidate_id)
        if design is not None:
            failed = design.comparison.status == "fit-failed" or (
                design.comparison.n_inference_fits == 0
                and design.comparison.status not in {"historical-descriptive", "mixed"}
            )
            updated.append(
                _replace(
                    candidate,
                    status="fit-failed" if failed else "evaluated",
                    reason="" if not failed else design.comparison.status,
                )
            )
            continue
        if candidate.candidate_id in shortlist_ids and stopped:
            reason = (
                "memory limit stopped the search"
                if stop_status == "memory-exhausted"
                else "fit budget exhausted before this shortlist member was fitted"
            )
            updated.append(_replace(candidate, status="unfinished", reason=reason))
            continue
        if (
            candidate.status == "screened"
            and candidate.candidate_id not in shortlist_ids
        ):
            updated.append(
                _replace(
                    candidate,
                    reason="screened; outside the shortlist and not evaluated",
                )
            )
            continue
        updated.append(candidate)
    return tuple(updated)


def _assemble(
    *,
    mode: Literal["fixed-target", "free-target"],
    candidates: tuple[ScreenedCandidate, ...],
    evaluated: tuple[EvaluatedDesign, ...],
    shortlist_ids: set[str],
    stop_status: str | None,
    disclosure: str,
    sample_kwargs: dict[str, Any] | None,
    random_seed: int,
    costs_supplied: bool,
    relevance_supplied: bool,
) -> SearchResult:
    n_screened = sum(item.proxy_score is not None for item in candidates)
    n_evaluated = sum(item.status == "evaluated" for item in candidates)
    n_skipped = sum(item.status == "skipped" for item in candidates)
    n_failed = sum(
        item.status in {"screen-failed", "fit-failed"} for item in candidates
    )
    n_unfinished = sum(item.status == "unfinished" for item in candidates)
    n_fits = sum(item.comparison.n_inference_fits for item in evaluated)
    proxy_ids = {
        item.candidate_id
        for item in candidates
        if item.proxy_score is not None and item.status != "skipped"
    }
    fitted_ids = {
        item.candidate_id for item in evaluated if item.comparison.n_inference_fits > 0
    }
    exhaustive = (
        bool(proxy_ids)
        and proxy_ids <= fitted_ids
        and n_unfinished == 0
        and stop_status is None
    )
    recall, regret, partial = _recall_regret(candidates, evaluated, exhaustive)
    status = _search_status(candidates, evaluated, stop_status)
    text = _recommendation(status, partial, disclosure)
    return SearchResult(
        status=status,
        mode=mode,
        candidates=candidates,
        evaluated=evaluated,
        pareto=_pareto(evaluated, candidates, costs_supplied, relevance_supplied),
        n_screened=n_screened,
        n_evaluated=n_evaluated,
        n_skipped=n_skipped,
        n_failed=n_failed,
        n_unfinished=n_unfinished,
        n_inference_fits=n_fits,
        budget_exhausted=status == "budget-exhausted",
        evaluation_exhaustive=exhaustive,
        proxy_recall=recall,
        proxy_regret=regret,
        partial_reason=partial,
        selection_policy=_SELECTION_POLICY,
        disclosure=disclosure,
        recommendation_text=text,
        sample_kwargs={} if sample_kwargs is None else dict(sample_kwargs),
        random_seed=random_seed,
    )


def _search_status(
    candidates: Sequence[ScreenedCandidate],
    evaluated: Sequence[EvaluatedDesign],
    stop_status: str | None,
) -> str:
    if stop_status == "memory-exhausted":
        return "memory-exhausted"
    if stop_status == "budget-exhausted":
        return "budget-exhausted"
    if not evaluated:
        return "insufficient-history"
    statuses = {item.comparison.status for item in evaluated}
    if statuses <= {"fit-failed", "insufficient-history"}:
        if statuses == {"insufficient-history"}:
            return "insufficient-history"
        return "fit-failed"
    if (
        "fit-failed" in statuses
        or "insufficient-history" in statuses
        or "mixed" in statuses
    ):
        return "mixed"
    if statuses == {"historical-descriptive"}:
        return "historical-descriptive"
    if not any(item.status == "evaluated" for item in candidates):
        return "insufficient-history"
    return "historical-descriptive"


def _recall_regret(
    candidates: Sequence[ScreenedCandidate],
    evaluated: Sequence[EvaluatedDesign],
    exhaustive: bool,
) -> tuple[float | None, float | None, str | None]:
    if not exhaustive:
        return (
            None,
            None,
            "Recall and regret are withheld because evaluation was not exhaustive. "
            "Partial evidence is not a silent winner.",
        )
    sensitivities = {
        item.candidate_id: _best_sensitivity(item.comparison) for item in evaluated
    }
    scored = {key: value for key, value in sensitivities.items() if value is not None}
    if not scored:
        return None, None, "No evaluated candidate has a historical detection share."
    best = max(scored.values())
    best_ids = {key for key, value in scored.items() if value == best}
    proxy_best = min(
        (
            item
            for item in candidates
            if item.candidate_id in scored and item.proxy_score is not None
        ),
        key=lambda item: (item.proxy_score, item.candidate_id),
        default=None,
    )
    if proxy_best is None or proxy_best.proxy_score is None:
        return (
            None,
            None,
            "No screened candidate has both a proxy score and a sensitivity.",
        )
    recall = 1.0 if proxy_best.candidate_id in best_ids else 0.0
    regret = float(best - scored[proxy_best.candidate_id])
    return recall, regret, None


def _assessed_share(row: DurationRow) -> float | None:
    """Return a scored detection share, or None when the row was not assessed."""
    if row.evidence_level != "historical-descriptive":
        return None
    share = row.historical_detection_share
    if share is None or not np.isfinite(share):
        return None
    return float(share)


def _best_sensitivity(comparison: DurationComparison) -> float | None:
    shares = [
        share for row in comparison.rows if (share := _assessed_share(row)) is not None
    ]
    if not shares:
        return None
    return float(max(shares))


def _pareto(
    evaluated: Sequence[EvaluatedDesign],
    candidates: Sequence[ScreenedCandidate],
    costs_supplied: bool,
    relevance_supplied: bool,
) -> tuple[ParetoPoint, ...]:
    by_id = {item.candidate_id: item for item in candidates}
    points: list[ParetoPoint] = []
    for design in evaluated:
        source = by_id[design.candidate_id]
        for row in design.comparison.rows:
            share = _assessed_share(row)
            if share is None:
                continue
            points.append(
                ParetoPoint(
                    candidate_id=design.candidate_id,
                    treated_unit=design.treated_unit,
                    duration=row.duration,
                    sensitivity=float(share),
                    cost=source.cost if costs_supplied else None,
                    relevance=source.relevance if relevance_supplied else None,
                    dominated=False,
                )
            )
    if not points:
        return ()
    return tuple(_with_dominance(point, points) for point in points)


def _with_dominance(
    point: ParetoPoint,
    points: Sequence[ParetoPoint],
) -> ParetoPoint:
    dominated = any(other is not point and _dominates(other, point) for other in points)
    return ParetoPoint(
        candidate_id=point.candidate_id,
        treated_unit=point.treated_unit,
        duration=point.duration,
        sensitivity=point.sensitivity,
        cost=point.cost,
        relevance=point.relevance,
        dominated=dominated,
    )


def _dominates(left: ParetoPoint, right: ParetoPoint) -> bool:
    better = False
    if left.sensitivity < right.sensitivity:
        return False
    if left.sensitivity > right.sensitivity:
        better = True
    if left.cost is not None and right.cost is not None:
        if left.cost > right.cost:
            return False
        if left.cost < right.cost:
            better = True
    if left.relevance is not None and right.relevance is not None:
        if left.relevance < right.relevance:
            return False
        if left.relevance > right.relevance:
            better = True
    return better


def _recommendation(status: str, partial: str | None, disclosure: str) -> str:
    parts = [_NO_RECOMMENDATION, _status_sentence(status)]
    if partial:
        parts.append(partial)
    if disclosure:
        parts.append(disclosure)
    return " ".join(parts)


def _status_sentence(status: str) -> str:
    sentences = {
        "budget-exhausted": (
            "Budget exhaustion is partial evidence. No winner was selected."
        ),
        "memory-exhausted": (
            "The memory limit stopped the search. No winner was selected."
        ),
        "fit-failed": "Every evaluated comparison failed. No winner was selected.",
        "mixed": "Some evaluated comparisons failed. No winner was selected.",
        "insufficient-history": ("No candidate was evaluated. No winner was selected."),
        "historical-descriptive": (
            "Evaluated comparisons are historical descriptions. No winner was selected."
        ),
    }
    return sentences.get(status, "No winner was selected.")


def _join_notes(*notes: str) -> str:
    return " ".join(note for note in notes if note)
