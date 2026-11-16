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
"""Candidate generation and inexpensive screening.

:class:`CandidateGenerator` is the public entry. It implements
:class:`causalpy.planning.seams.CandidateGenerator`. Enumeration, filters,
the seeded hash sample, the shortlist, and the proxy follow
:class:`~causalpy.planning.contracts.CandidateRequest`. This module does not
define a second ranking and does not fit a Bayesian model.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from causalpy.planning.contracts import (
    CANDIDATE_COLUMNS,
    DRAW_ALIGNED_UNIT_SUM,
    CandidateRequest,
    CandidateTable,
)
from causalpy.planning.seams import CandidateGenerator as CandidateGeneratorProtocol
from causalpy.skl_models import WeightedProportion

__all__ = ["CandidateGenerator"]

_REASON_ORDER: tuple[str, ...] = (
    "group_size",
    "excluded_unit",
    "not_treatment_eligible",
    "missing_required",
    "donor_depleted",
    "over_budget",
    "cost_unknown",
    "outcome_share",
    "representativeness",
)


@dataclass(frozen=True)
class _Identity:
    treated: tuple[str, ...]
    donors: tuple[str, ...]

    @property
    def candidate_id(self) -> str:
        return json.dumps(
            [list(self.treated), list(self.donors)],
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class _Scored:
    identity: _Identity
    reasons: tuple[str, ...]
    cost: float | None
    outcome_share: float | None
    proxy_score: float | None
    proxy_role: str


def _treatment_pool(request: CandidateRequest) -> tuple[str, ...]:
    if request.treatment_eligible is None:
        blocked = set(request.excluded_units) | set(request.always_control)
        names = [name for name in request.units if name not in blocked]
    else:
        names = list(request.treatment_eligible)
    return tuple(sorted(set(names)))


def _donor_base(request: CandidateRequest) -> tuple[str, ...]:
    if request.donor_eligible is None:
        excluded = set(request.excluded_units)
        names = [name for name in request.units if name not in excluded]
    else:
        names = list(request.donor_eligible)
    return tuple(sorted(set(names)))


def _spillover_map(request: CandidateRequest) -> dict[str, frozenset[str]]:
    mapping: dict[str, set[str]] = {}
    for source, contaminated in request.spillover_exclusions:
        mapping.setdefault(source, set()).add(contaminated)
    return {source: frozenset(banned) for source, banned in mapping.items()}


def _blocked_donors(
    treated: tuple[str, ...],
    spills: dict[str, frozenset[str]],
) -> set[str]:
    blocked = set(treated)
    for name in treated:
        blocked.update(spills.get(name, ()))
    return blocked


def _enumerated_groups(request: CandidateRequest) -> tuple[tuple[str, ...], ...]:
    required = tuple(sorted(request.required_treated))
    pool = _treatment_pool(request)
    others = tuple(name for name in pool if name not in set(required))
    groups: list[tuple[str, ...]] = []
    for size in range(request.min_group_size, request.max_group_size + 1):
        extra = size - len(required)
        if extra < 0 or extra > len(others):
            continue
        if extra == 0:
            groups.append(required)
            continue
        groups.extend(
            tuple(sorted((*required, *combo))) for combo in combinations(others, extra)
        )
    return tuple(groups)


def _identities(request: CandidateRequest) -> tuple[_Identity, ...]:
    spills = _spillover_map(request)
    base = _donor_base(request)
    seen: dict[tuple[tuple[str, ...], tuple[str, ...]], _Identity] = {}

    def add(treated: tuple[str, ...], donors: tuple[str, ...]) -> None:
        identity = _Identity(tuple(sorted(treated)), tuple(sorted(donors)))
        seen.setdefault((identity.treated, identity.donors), identity)

    if request.supplied_groups is not None:
        for group in request.supplied_groups:
            treated = tuple(sorted(group))
            add(treated, _complement_donors(treated, base, spills))
    elif request.fixed_treated is not None:
        treated = tuple(sorted(request.fixed_treated))
        if request.donor_pools is None:
            add(treated, _complement_donors(treated, base, spills))
        else:
            for pool in request.donor_pools:
                blocked = _blocked_donors(treated, spills)
                donors = tuple(sorted(name for name in pool if name not in blocked))
                add(treated, donors)
    else:
        for treated in _enumerated_groups(request):
            add(treated, _complement_donors(treated, base, spills))
    return tuple(seen.values())


def _complement_donors(
    treated: tuple[str, ...],
    donor_base: tuple[str, ...],
    spills: dict[str, frozenset[str]],
) -> tuple[str, ...]:
    blocked = _blocked_donors(treated, spills)
    return tuple(name for name in donor_base if name not in blocked)


def _digest(request: CandidateRequest, identity: _Identity) -> str:
    donors = None if request.donor_pools is None else list(identity.donors)
    payload = json.dumps(
        [request.random_seed, list(identity.treated), donors],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _select(
    request: CandidateRequest,
    identities: tuple[_Identity, ...],
) -> tuple[tuple[_Identity, ...], bool]:
    n_space = len(identities)
    if n_space <= request.exact_ceiling:
        return identities, False
    if request.max_generated is None:
        raise ValueError(
            f"Candidate space has {n_space} identities, above exact_ceiling "
            f"{request.exact_ceiling}. Set max_generated to keep a seeded hash "
            "sample. Input order is not a truncation rule."
        )
    ranked = sorted(
        identities,
        key=lambda identity: (_digest(request, identity), identity.candidate_id),
    )
    kept = tuple(ranked[: request.max_generated])
    return kept, len(kept) < n_space


def _population(request: CandidateRequest) -> tuple[str, ...]:
    excluded = set(request.excluded_units)
    return tuple(sorted(name for name in request.units if name not in excluded))


def _finite(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return value


class _FilteredShare:
    """Marker for a share that is stored as null and filtered."""


_FILTERED_SHARE = _FilteredShare()


def _reasons_cost_share(
    request: CandidateRequest,
    identity: _Identity,
    history: pd.DataFrame | None,
    population: tuple[str, ...],
) -> tuple[tuple[str, ...], float | None, float | None]:
    treated = identity.treated
    matched = dict.fromkeys(_REASON_ORDER, False)
    if not request.min_group_size <= len(treated) <= request.max_group_size:
        matched["group_size"] = True
    excluded = set(request.excluded_units)
    if any(name in excluded for name in treated):
        matched["excluded_unit"] = True
    controls = set(request.always_control)
    ineligible = any(name in controls for name in treated)
    if request.treatment_eligible is not None:
        eligible = set(request.treatment_eligible)
        ineligible = ineligible or any(name not in eligible for name in treated)
    if ineligible:
        matched["not_treatment_eligible"] = True
    if any(name not in treated for name in request.required_treated):
        matched["missing_required"] = True
    if len(identity.donors) < request.min_remaining_donors:
        matched["donor_depleted"] = True
    costs = dict(request.unit_costs)
    if all(name in costs for name in treated):
        cost = _finite(float(sum(costs[name] for name in treated)))
    else:
        cost = None
    if request.budget is not None:
        if cost is None:
            matched["cost_unknown"] = True
        elif cost > request.budget:
            matched["over_budget"] = True
    raw_share = _outcome_share(request, treated, history, population)
    share: float | None
    if isinstance(raw_share, _FilteredShare):
        matched["outcome_share"] = True
        share = None
    else:
        share = raw_share
        if share is not None and (
            (
                request.min_outcome_share is not None
                and share < request.min_outcome_share
            )
            or (
                request.max_outcome_share is not None
                and share > request.max_outcome_share
            )
        ):
            matched["outcome_share"] = True
    if _fails_balance(request, treated, population):
        matched["representativeness"] = True
    reasons = tuple(reason for reason in _REASON_ORDER if matched[reason])
    return reasons, cost, share


def _outcome_share(
    request: CandidateRequest,
    treated: tuple[str, ...],
    history: pd.DataFrame | None,
    population: tuple[str, ...],
) -> float | None | _FilteredShare:
    if history is None:
        if (
            request.min_outcome_share is not None
            or request.max_outcome_share is not None
        ):
            return _FILTERED_SHARE
        return None
    if any(name not in history.columns for name in (*treated, *population)):
        return _FILTERED_SHARE
    if not population:
        return _FILTERED_SHARE
    try:
        denominator = float(
            history.loc[:, list(population)].to_numpy(dtype=float).sum()
        )
        numerator = float(history.loc[:, list(treated)].to_numpy(dtype=float).sum())
    except (TypeError, ValueError, OverflowError):
        return _FILTERED_SHARE
    if not math.isfinite(denominator) or denominator <= 0.0:
        return _FILTERED_SHARE
    if not math.isfinite(numerator):
        return _FILTERED_SHARE
    share = numerator / denominator
    if not math.isfinite(share):
        return _FILTERED_SHARE
    return share


def _fails_balance(
    request: CandidateRequest,
    treated: tuple[str, ...],
    population: tuple[str, ...],
) -> bool:
    if not request.balance_constraints:
        return False
    covariates = {name: dict(values) for name, values in request.unit_covariates}
    for constraint in request.balance_constraints:
        values = covariates.get(constraint.covariate)
        if values is None or not population:
            return True
        needed = (*treated, *population)
        if any(name not in values for name in needed):
            return True
        treated_mean = sum(values[name] for name in treated) / len(treated)
        population_mean = sum(values[name] for name in population) / len(population)
        gap = abs(treated_mean - population_mean)
        if not math.isfinite(gap) or gap > constraint.max_abs_mean_gap:
            return True
    return False


def _proxy(
    identity: _Identity,
    history: pd.DataFrame | None,
) -> tuple[str, float | None]:
    if history is None or not identity.donors:
        return "unavailable", None
    needed = (*identity.treated, *identity.donors)
    if any(name not in history.columns for name in needed):
        return "unavailable", None
    scores: list[float] = []
    try:
        predictors = history.loc[:, list(identity.donors)].to_numpy(dtype=float)
        for unit in identity.treated:
            outcome = history.loc[:, unit].to_numpy(dtype=float)
            fitted = WeightedProportion().fit(predictors, outcome)
            score = float(fitted.mse)
            if score != score or score in (float("inf"), float("-inf")):
                return "unavailable", None
            scores.append(score)
    except Exception:
        return "unavailable", None
    return "preperiod_weighted_proportion", sum(scores) / len(scores)


def _score(
    request: CandidateRequest,
    identity: _Identity,
    history: pd.DataFrame | None,
    population: tuple[str, ...],
) -> _Scored:
    reasons, cost, share = _reasons_cost_share(request, identity, history, population)
    role, score = _proxy(identity, history)
    return _Scored(identity, reasons, cost, share, score, role)


def _extreme(
    rows: list[_Scored],
    value,
    *,
    highest: bool,
) -> _Scored | None:
    finite = [row for row in rows if value(row) is not None]
    if not finite:
        return None

    def key(row: _Scored) -> tuple[float, str]:
        number = float(value(row))
        return (-number if highest else number, row.identity.candidate_id)

    return min(finite, key=key)


def _shortlist(
    request: CandidateRequest,
    scored: list[_Scored],
) -> dict[str, tuple[int, bool]]:
    passing = [row for row in scored if not row.reasons]
    kept: list[tuple[_Scored, bool]] = []
    kept_ids: set[str] = set()

    def take(row: _Scored | None, boundary: bool) -> None:
        if row is None or len(kept) >= request.shortlist_size:
            return
        identity = row.identity.candidate_id
        if identity in kept_ids:
            return
        kept.append((row, boundary))
        kept_ids.add(identity)

    if any(row.cost is not None for row in passing):
        take(_extreme(passing, lambda row: row.cost, highest=False), True)
        take(_extreme(passing, lambda row: row.cost, highest=True), True)
    if any(row.proxy_score is not None for row in passing):
        take(_extreme(passing, lambda row: row.proxy_score, highest=False), True)
        take(_extreme(passing, lambda row: row.proxy_score, highest=True), True)
    if any(row.outcome_share is not None for row in passing):
        take(_extreme(passing, lambda row: row.outcome_share, highest=False), True)
        take(_extreme(passing, lambda row: row.outcome_share, highest=True), True)
    for size in range(request.min_group_size, request.max_group_size + 1):
        matching = [
            row
            for row in passing
            if len(row.identity.treated) == size
            and row.identity.candidate_id not in kept_ids
        ]
        if matching:
            take(min(matching, key=lambda row: row.identity.candidate_id), True)
    remaining = sorted(
        (row for row in passing if row.identity.candidate_id not in kept_ids),
        key=_remaining_key,
    )
    for row in remaining:
        if len(kept) >= request.shortlist_size:
            break
        take(row, False)
    return {
        row.identity.candidate_id: (rank, boundary)
        for rank, (row, boundary) in enumerate(kept, start=1)
    }


def _remaining_key(row: _Scored) -> tuple[object, ...]:
    return (
        row.proxy_score is None,
        0.0 if row.proxy_score is None else row.proxy_score,
        row.outcome_share is None,
        0.0 if row.outcome_share is None else -row.outcome_share,
        row.cost is None,
        0.0 if row.cost is None else row.cost,
        row.identity.candidate_id,
    )


def _missing() -> object:
    return pd.NA


def _frame(scored: list[_Scored], ranks: dict[str, tuple[int, bool]]) -> pd.DataFrame:
    ordered = sorted(scored, key=lambda row: row.identity.candidate_id)
    columns: dict[str, list[object]] = {name: [] for name in CANDIDATE_COLUMNS}
    for row in ordered:
        identity = row.identity
        placement = ranks.get(identity.candidate_id)
        if row.reasons:
            status = "filtered"
            boundary = False
            rank: object = _missing()
        elif placement is None:
            status = "enumerated"
            boundary = False
            rank = _missing()
        else:
            status = "shortlisted"
            rank, boundary = placement
        columns["candidate_id"].append(identity.candidate_id)
        columns["treated"].append(identity.treated)
        columns["donors"].append(identity.donors)
        columns["n_donors"].append(len(identity.donors))
        columns["estimand"].append(DRAW_ALIGNED_UNIT_SUM)
        columns["member_weight"].append(1.0 / len(identity.treated))
        columns["cost"].append(_missing() if row.cost is None else row.cost)
        columns["outcome_share"].append(
            _missing() if row.outcome_share is None else row.outcome_share
        )
        columns["proxy_score"].append(
            _missing() if row.proxy_score is None else row.proxy_score
        )
        columns["proxy_role"].append(row.proxy_role)
        columns["boundary"].append(boundary)
        columns["status"].append(status)
        columns["filter_reasons"].append(row.reasons)
        columns["shortlist_rank"].append(rank)
    return pd.DataFrame(
        {name: pd.Series(values, dtype=object) for name, values in columns.items()}
    )


class CandidateGenerator(CandidateGeneratorProtocol):
    """Screen treatment groups of size 1 to 3.

    Enumeration, filters, the seeded hash sample, the shortlist, and the
    pre-period weighted-proportion proxy follow :class:`CandidateRequest`.
    The proxy is not an inference fit, and ``n_inference_fits`` is 0.
    """

    def generate(self, request: CandidateRequest) -> CandidateTable:
        """Return every enumerated identity, including filtered rows.

        Parameters
        ----------
        request : CandidateRequest
            Units, eligibility, history cutoff, and the required seed.

        Returns
        -------
        CandidateTable
            Plot-ready rows. ``n_inference_fits`` is 0. Proxy score is not power.
        """
        if not isinstance(request, CandidateRequest):
            raise TypeError("request must be a CandidateRequest.")
        identities = _identities(request)
        selected, partial = _select(request, identities)
        history = request.history
        population = _population(request)
        scored = [
            _score(request, identity, history, population) for identity in selected
        ]
        ranks = _shortlist(request, scored)
        filtered = sum(1 for row in scored if row.reasons)
        return CandidateTable(
            _frame(scored, ranks),
            n_space=len(identities),
            n_enumerated=len(selected),
            n_filtered=filtered,
            n_shortlisted=len(ranks),
            partial=partial,
            enumeration_policy="seeded_hash_sample" if partial else "exact",
            estimand=DRAW_ALIGNED_UNIT_SUM,
            n_inference_fits=0,
        )
