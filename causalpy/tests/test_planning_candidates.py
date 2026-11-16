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
"""Candidate generation and screening. No Bayesian fits."""

from __future__ import annotations

import hashlib
import json
import warnings
from itertools import combinations

import pandas as pd
import pytest

from causalpy.planning.candidates import CandidateGenerator
from causalpy.planning.contracts import (
    CANDIDATE_COLUMNS,
    DRAW_ALIGNED_UNIT_SUM,
    BalanceConstraint,
    CandidateRequest,
    CandidateTable,
)
from causalpy.skl_models import WeightedProportion


def _request(**overrides: object) -> CandidateRequest:
    fields: dict[str, object] = {
        "units": ("a", "b", "c"),
        "min_group_size": 1,
        "max_group_size": 1,
        "random_seed": 0,
    }
    fields.update(overrides)
    return CandidateRequest(**fields)


def _generate(**overrides: object) -> CandidateTable:
    return CandidateGenerator().generate(_request(**overrides))


def _by_treated(table: CandidateTable) -> dict[tuple[str, ...], pd.Series]:
    frame = table.frame
    return {tuple(row.treated): row for row in frame.itertuples(index=False)}


def _history(*rows: dict[str, float], end: object = 2) -> dict[str, object]:
    frame = pd.DataFrame(
        list(rows), index=pd.Index(range(1, len(rows) + 1), name="time")
    )
    return {"history": frame, "selection_end": end}


def test_twelve_unconstrained_units_expose_298_groups() -> None:
    units = tuple("abcdefghijkl")
    table = _generate(
        units=units,
        min_group_size=1,
        max_group_size=3,
        shortlist_size=10,
    )
    treated = [tuple(row.treated) for row in table.frame.itertuples(index=False)]
    expected = [combo for size in (1, 2, 3) for combo in combinations(units, size)]
    assert table.n_space == 298
    assert table.n_enumerated == 298
    assert table.n_filtered == 0
    assert table.partial is False
    assert table.enumeration_policy == "exact"
    assert table.n_inference_fits == 0
    assert table.estimand == DRAW_ALIGNED_UNIT_SUM
    assert sorted(treated) == sorted(expected)
    assert [len(group) for group in treated].count(1) == 12
    assert [len(group) for group in treated].count(2) == 66
    assert [len(group) for group in treated].count(3) == 220
    assert table.n_shortlisted == 10
    assert list(table.frame.columns) == list(CANDIDATE_COLUMNS)


def test_exact_small_space_counts_match_combinations() -> None:
    units = ("a", "b", "c", "d")
    table = _generate(units=units, min_group_size=1, max_group_size=3)
    assert table.n_space == 14
    assert table.n_enumerated == 14
    pairs = _generate(units=units, min_group_size=2, max_group_size=2)
    assert pairs.n_space == 6
    assert {tuple(row.treated) for row in pairs.frame.itertuples(index=False)} == set(
        combinations(units, 2)
    )


def test_required_excluded_and_always_control_units() -> None:
    table = _generate(
        units=("a", "b", "c", "d"),
        min_group_size=1,
        max_group_size=2,
        treatment_eligible=("a", "b", "c"),
        always_control=("d",),
        excluded_units=(),
        required_treated=("a",),
        donor_eligible=("b", "c", "d"),
    )
    treated = {tuple(row.treated) for row in table.frame.itertuples(index=False)}
    assert treated == {("a",), ("a", "b"), ("a", "c")}
    for row in table.frame.itertuples(index=False):
        assert "d" not in row.treated
        assert "a" not in row.donors
        assert set(row.treated).isdisjoint(row.donors)


def test_treated_unit_never_donates_to_its_group() -> None:
    table = _generate(
        units=("a", "b", "c"),
        min_group_size=1,
        max_group_size=3,
        donor_eligible=("a", "b", "c"),
    )
    for row in table.frame.itertuples(index=False):
        assert set(row.treated).isdisjoint(row.donors)
        assert row.n_donors == len(row.donors)


def test_donor_depletion_filters_only_on_the_remaining_count() -> None:
    table = _generate(
        units=("a", "b"),
        min_group_size=1,
        max_group_size=2,
        min_remaining_donors=2,
    )
    assert table.n_space == 3
    assert table.n_filtered == 3
    assert set(table.frame["status"]) == {"filtered"}
    assert all(
        row.filter_reasons == ("donor_depleted",)
        for row in table.frame.itertuples(index=False)
    )


def test_spillover_removes_a_donor_without_its_own_reason() -> None:
    table = _generate(
        units=("a", "b", "c"),
        min_group_size=1,
        max_group_size=2,
        spillover_exclusions=(("a", "b"),),
        min_remaining_donors=1,
    )
    rows = _by_treated(table)
    assert rows[("a",)].donors == ("c",)
    assert rows[("a",)].filter_reasons == ()
    assert "b" not in rows[("a", "c")].donors
    assert rows[("a", "b")].status != "filtered"


def test_budget_unknown_cost_and_exact_budget() -> None:
    table = _generate(
        units=("a", "b", "c"),
        min_group_size=1,
        max_group_size=2,
        unit_costs=(("a", 3.0), ("b", 4.0)),
        budget=5.0,
    )
    rows = _by_treated(table)
    assert rows[("a",)].cost == 3.0
    assert rows[("a",)].filter_reasons == ()
    assert rows[("b",)].cost == 4.0
    assert rows[("a", "b")].cost == 7.0
    assert rows[("a", "b")].filter_reasons == ("over_budget",)
    assert rows[("c",)].cost is pd.NA or pd.isna(rows[("c",)].cost)
    assert rows[("c",)].filter_reasons == ("cost_unknown",)
    at_budget = _generate(
        units=("a", "b", "c"),
        min_group_size=2,
        max_group_size=2,
        fixed_treated=("a", "b"),
        unit_costs=(("a", 2.0), ("b", 3.0)),
        budget=5.0,
    )
    assert at_budget.n_filtered == 0
    assert at_budget.frame.iloc[0].cost == 5.0


def test_supplied_group_collects_every_matching_reason_in_order() -> None:
    table = _generate(
        units=("a", "b", "c"),
        min_group_size=1,
        max_group_size=1,
        treatment_eligible=("a",),
        always_control=("b",),
        excluded_units=("c",),
        required_treated=("a",),
        supplied_groups=(("b", "c"),),
        min_remaining_donors=2,
        budget=1.0,
        min_outcome_share=0.0,
        balance_constraints=(BalanceConstraint("income", 0.0),),
    )
    assert table.n_space == 1
    row = table.frame.iloc[0]
    assert row.filter_reasons == (
        "group_size",
        "excluded_unit",
        "not_treatment_eligible",
        "missing_required",
        "donor_depleted",
        "cost_unknown",
        "outcome_share",
        "representativeness",
    )
    assert row.status == "filtered"
    assert row.boundary is False
    assert pd.isna(row.shortlist_rank)


def test_duplicates_and_input_order_collapse() -> None:
    forward = _generate(
        units=("c", "a", "b"),
        min_group_size=1,
        max_group_size=2,
        supplied_groups=(("b", "a"), ("a", "b"), ("c",)),
    )
    reverse = _generate(
        units=("a", "b", "c"),
        min_group_size=1,
        max_group_size=2,
        supplied_groups=(("c",), ("a", "b")),
    )
    assert forward.n_space == 2
    assert reverse.n_space == 2
    assert forward.frame.equals(reverse.frame)
    assert forward == reverse


def test_fixed_target_donor_alternatives_collapse_after_spillover() -> None:
    table = _generate(
        units=("a", "b", "c"),
        min_group_size=1,
        max_group_size=1,
        fixed_treated=("a",),
        donor_pools=(("b",), ("b", "c"), ("c",), ("c", "b")),
        spillover_exclusions=(("a", "b"),),
    )
    assert table.n_space == 2
    donors = {tuple(row.donors): row for row in table.frame.itertuples(index=False)}
    assert set(donors) == {(), ("c",)}
    assert donors[()].filter_reasons == ("donor_depleted",)
    assert donors[("c",)].filter_reasons == ()
    assert donors[("c",)].treated == ("a",)
    assert "a" not in donors[("c",)].donors


def test_reproducible_and_not_input_order_truncation() -> None:
    units = ("a", "b", "c", "d")
    kwargs = {
        "units": units,
        "min_group_size": 1,
        "max_group_size": 2,
        "exact_ceiling": 2,
        "max_generated": 3,
        "random_seed": 11,
    }
    first = _generate(**kwargs)
    second = _generate(**kwargs)
    reversed_units = _generate(**{**kwargs, "units": tuple(reversed(units))})
    assert first == second
    assert first.frame.equals(reversed_units.frame)
    assert first.n_space == 10
    assert first.n_enumerated == 3
    assert first.partial is True
    assert first.enumeration_policy == "seeded_hash_sample"
    expected = _lowest_identities(units, seed=11, limit=3)
    got = {
        (tuple(row.treated), tuple(row.donors))
        for row in first.frame.itertuples(index=False)
    }
    assert got == expected


def test_full_hash_keep_is_stored_as_exact() -> None:
    table = _generate(
        units=("a", "b", "c"),
        exact_ceiling=1,
        max_generated=3,
    )
    assert table.n_space == 3
    assert table.n_enumerated == 3
    assert table.partial is False
    assert table.enumeration_policy == "exact"


def test_space_above_ceiling_without_max_generated_raises() -> None:
    with pytest.raises(ValueError, match="exact_ceiling"):
        _generate(units=("a", "b", "c"), exact_ceiling=1)


def test_outcome_share_uses_only_the_selected_window() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 1.0, 100.0],
            "b": [1.0, 1.0, 0.0],
            "c": [3.0, 3.0, 0.0],
        },
        index=pd.Index([1, 2, 3], name="time"),
    )
    table = _generate(
        units=("a", "b", "c"),
        history=frame,
        selection_end=2,
        min_outcome_share=0.3,
        max_outcome_share=0.5,
    )
    rows = _by_treated(table)
    assert rows[("a",)].outcome_share == pytest.approx(0.2)
    assert rows[("a",)].filter_reasons == ("outcome_share",)
    assert rows[("b",)].outcome_share == pytest.approx(0.2)
    assert rows[("c",)].outcome_share == pytest.approx(0.6)
    assert rows[("c",)].status == "filtered"
    selected = frame.loc[[1, 2]]
    expected = _unit_rmse(selected, "a", ("b", "c"))
    full = _unit_rmse(frame, "a", ("b", "c"))
    assert rows[("a",)].proxy_score == pytest.approx(expected)
    assert rows[("a",)].proxy_score != pytest.approx(full)
    assert rows[("a",)].proxy_role == "preperiod_weighted_proportion"


def test_share_bound_without_history_filters_and_stores_null() -> None:
    table = _generate(min_outcome_share=0.1)
    row = table.frame.iloc[0]
    assert row.filter_reasons == ("outcome_share",)
    assert pd.isna(row.outcome_share)
    assert row.proxy_role == "unavailable"
    assert pd.isna(row.proxy_score)


def test_nonpositive_denominator_does_not_divide() -> None:
    frame = pd.DataFrame(
        {"a": [-1.0, -1.0], "b": [-1.0, -1.0]},
        index=pd.Index([1, 2], name="time"),
    )
    table = _generate(
        units=("a", "b"),
        history=frame,
        selection_end=2,
    )
    row = _by_treated(table)[("a",)]
    assert row.filter_reasons == ("outcome_share",)
    assert pd.isna(row.outcome_share)


def test_representativeness_uses_the_population_mean() -> None:
    passing = _generate(
        units=("a", "b"),
        unit_covariates=(("income", (("a", 0.0), ("b", 2.0))),),
        balance_constraints=(BalanceConstraint("income", 1.0),),
    )
    failing = _generate(
        units=("a", "b"),
        unit_covariates=(("income", (("a", 0.0), ("b", 2.0))),),
        balance_constraints=(BalanceConstraint("income", 0.5),),
    )
    assert passing.n_filtered == 0
    assert failing.frame.iloc[0].filter_reasons == ("representativeness",)
    missing = _generate(
        units=("a",),
        excluded_units=("a",),
        supplied_groups=(("a",),),
        balance_constraints=(BalanceConstraint("income", 1.0),),
    )
    assert "representativeness" in missing.frame.iloc[0].filter_reasons


def test_proxy_is_the_mean_unit_rmse_not_the_summed_series() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [2.0, 4.0, 6.0, 8.0],
            "c": [1.0, 0.0, 1.0, 0.0],
        },
        index=pd.Index([1, 2, 3, 4], name="time"),
    )
    table = _generate(
        units=("a", "b", "c"),
        min_group_size=2,
        max_group_size=2,
        fixed_treated=("a", "b"),
        history=frame,
        selection_end=4,
    )
    row = table.frame.iloc[0]
    per_unit = (_unit_rmse(frame, "a", ("c",)) + _unit_rmse(frame, "b", ("c",))) / 2
    summed = _unit_rmse(
        pd.DataFrame({"ab": frame["a"] + frame["b"], "c": frame["c"]}),
        "ab",
        ("c",),
    )
    assert row.proxy_score == pytest.approx(per_unit)
    assert row.proxy_score != pytest.approx(summed)
    assert row.member_weight == pytest.approx(0.5)
    assert row.estimand == DRAW_ALIGNED_UNIT_SUM
    assert table.n_inference_fits == 0


def test_failed_proxy_fit_is_unavailable_not_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(self: WeightedProportion, X: object, y: object) -> WeightedProportion:
        raise RuntimeError("fit failed")

    monkeypatch.setattr(
        "causalpy.planning.candidates.WeightedProportion.fit",
        _fail,
    )
    frame = pd.DataFrame(
        {"a": [1.0, 2.0], "b": [1.0, 2.0]},
        index=pd.Index([1, 2], name="time"),
    )
    table = _generate(units=("a", "b"), history=frame, selection_end=2)
    row = table.frame.iloc[0]
    assert row.proxy_role == "unavailable"
    assert pd.isna(row.proxy_score)


def test_empty_donor_proxy_is_unavailable() -> None:
    table = _generate(
        units=("a", "b"),
        min_group_size=2,
        max_group_size=2,
        **_history({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0}),
    )
    row = table.frame.iloc[0]
    assert row.donors == ()
    assert row.proxy_role == "unavailable"
    assert pd.isna(row.proxy_score)
    assert "donor_depleted" in row.filter_reasons


def test_shortlist_keeps_cost_extremes_then_size_then_id() -> None:
    table = _generate(
        units=("a", "b", "c"),
        min_group_size=1,
        max_group_size=2,
        unit_costs=(("a", 1.0), ("b", 5.0), ("c", 3.0)),
        shortlist_size=4,
    )
    ranked = table.frame.dropna(subset=["shortlist_rank"]).sort_values("shortlist_rank")
    order = [tuple(treated) for treated in ranked["treated"]]
    assert order == [("a",), ("b", "c"), ("b",), ("a", "b")]
    assert list(ranked["boundary"]) == [True, True, True, True]
    assert list(ranked["shortlist_rank"]) == [1, 2, 3, 4]
    wider = _generate(
        units=("a", "b", "c"),
        min_group_size=1,
        max_group_size=2,
        unit_costs=(("a", 1.0), ("b", 5.0), ("c", 3.0)),
        shortlist_size=5,
    )
    fifth = wider.frame.loc[wider.frame["shortlist_rank"] == 5].iloc[0]
    assert tuple(fifth.treated) == ("c",)
    assert fifth.boundary is False
    assert wider.n_shortlisted == 5


def test_no_history_shortlist_skips_proxy_and_relevance_slots() -> None:
    table = _generate(
        units=("a", "b", "c"),
        min_group_size=1,
        max_group_size=3,
        shortlist_size=1,
    )
    ranked = table.frame.loc[table.frame["status"] == "shortlisted"]
    assert len(ranked) == 1
    assert ranked.iloc[0].boundary is True
    assert tuple(ranked.iloc[0].treated) == ("a",)
    assert ranked.iloc[0].proxy_role == "unavailable"
    assert table.n_inference_fits == 0


def test_screening_does_not_call_bayesian_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("screening started an inference fit")

    monkeypatch.setattr(
        "causalpy.experiments.synthetic_control.SyntheticControl.fit", boom
    )
    monkeypatch.setattr("causalpy.pymc_models.WeightedSumFitter.fit", boom)
    table = _generate(units=("a", "b", "c"), min_group_size=1, max_group_size=3)
    assert table.n_inference_fits == 0
    assert set(table.frame["proxy_role"]) == {"unavailable"}


def test_required_set_skips_sizes_that_cannot_contain_it() -> None:
    table = _generate(
        units=("a", "b", "c"),
        min_group_size=1,
        max_group_size=3,
        treatment_eligible=("a", "b"),
        required_treated=("a", "b"),
    )
    assert table.n_space == 1
    assert table.n_filtered == 0
    assert tuple(table.frame.iloc[0].treated) == ("a", "b")


def test_excluded_treated_unit_missing_from_history_filters_share() -> None:
    frame = pd.DataFrame(
        {"a": [1.0, 1.0], "b": [2.0, 2.0]},
        index=pd.Index([1, 2], name="time"),
    )
    table = _generate(
        units=("a", "b", "c"),
        excluded_units=("c",),
        supplied_groups=(("c",),),
        history=frame,
        selection_end=2,
    )
    row = table.frame.iloc[0]
    assert row.filter_reasons == ("excluded_unit", "outcome_share")
    assert pd.isna(row.outcome_share)
    assert row.proxy_role == "unavailable"
    assert pd.isna(row.proxy_score)


def test_empty_population_filters_share_without_dividing() -> None:
    frame = pd.DataFrame({"a": [1.0, 1.0]}, index=pd.Index([1, 2], name="time"))
    table = _generate(
        units=("a",),
        excluded_units=("a",),
        supplied_groups=(("a",),),
        history=frame,
        selection_end=2,
    )
    row = table.frame.iloc[0]
    assert row.filter_reasons == ("excluded_unit", "donor_depleted", "outcome_share")
    assert pd.isna(row.outcome_share)


def test_missing_covariate_value_filters_representativeness() -> None:
    table = _generate(
        units=("a", "b"),
        min_group_size=1,
        max_group_size=1,
        unit_covariates=(("income", (("a", 1.0),)),),
        balance_constraints=(BalanceConstraint("income", 5.0),),
    )
    rows = _by_treated(table)
    assert rows[("a",)].filter_reasons == ("representativeness",)
    assert rows[("b",)].filter_reasons == ("representativeness",)


def test_nonfinite_proxy_score_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NanFit:
        mse = float("nan")

        def fit(self, predictors: object, outcome: object) -> _NanFit:
            return self

    monkeypatch.setattr(
        "causalpy.planning.candidates.WeightedProportion",
        lambda: _NanFit(),
    )
    frame = pd.DataFrame(
        {"a": [1.0, 2.0], "b": [1.0, 2.0]},
        index=pd.Index([1, 2], name="time"),
    )
    table = _generate(
        units=("a", "b"),
        fixed_treated=("a",),
        history=frame,
        selection_end=2,
    )
    row = table.frame.iloc[0]
    assert row.filter_reasons == ()
    assert row.proxy_role == "unavailable"
    assert pd.isna(row.proxy_score)


def test_generate_rejects_a_non_request() -> None:
    with pytest.raises(TypeError, match="CandidateRequest"):
        CandidateGenerator().generate(object())  # type: ignore[arg-type]


def test_overflowed_reductions_filter_instead_of_raising() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cost = _generate(
            units=("a", "b", "c"),
            min_group_size=2,
            max_group_size=2,
            fixed_treated=("a", "b"),
            unit_costs=(("a", 1e308), ("b", 1e308)),
        )
        budgeted = _generate(
            units=("a", "b", "c"),
            min_group_size=2,
            max_group_size=2,
            fixed_treated=("a", "b"),
            unit_costs=(("a", -1e308), ("b", -1e308)),
            budget=1.0,
        )
        frame = pd.DataFrame(
            {"a": [1e308], "b": [1e308], "c": [1e308]},
            index=pd.Index([1], name="time"),
        )
        share = _generate(
            units=("a", "b", "c"),
            min_group_size=1,
            max_group_size=2,
            history=frame,
            selection_end=1,
        )
        balance = _generate(
            units=("a", "b", "c"),
            min_group_size=2,
            max_group_size=2,
            fixed_treated=("a", "b"),
            unit_covariates=(("income", (("a", 1e308), ("b", 1e308), ("c", 1e308))),),
            balance_constraints=(BalanceConstraint("income", 0.0),),
        )
    assert pd.isna(cost.frame.iloc[0].cost)
    assert cost.frame.iloc[0].filter_reasons == ()
    assert pd.isna(budgeted.frame.iloc[0].cost)
    assert budgeted.frame.iloc[0].filter_reasons == ("cost_unknown",)
    assert share.n_filtered == share.n_enumerated
    assert share.n_shortlisted == 0
    assert set(share.frame["filter_reasons"]) == {("outcome_share",)}
    assert share.frame["outcome_share"].isna().all()
    assert balance.frame.iloc[0].filter_reasons == ("representativeness",)


def test_overflow_boundary_filters_ignore_unit_order() -> None:
    frame = pd.DataFrame(
        {"a": [1e308], "b": [1e308], "c": [-1e308]},
        index=pd.Index([1], name="time"),
    )
    covariates = (("income", (("a", 1e308), ("b", 1e308), ("c", -1e308))),)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        shares = [
            _generate(
                units=units,
                min_group_size=1,
                max_group_size=1,
                history=frame,
                selection_end=1,
            )
            for units in (("a", "b", "c"), ("a", "c", "b"))
        ]
        balances = [
            _generate(
                units=units,
                min_group_size=1,
                max_group_size=1,
                unit_covariates=covariates,
                balance_constraints=(BalanceConstraint("income", 1e308),),
            )
            for units in (("a", "b", "c"), ("a", "c", "b"))
        ]
    for table in shares:
        assert table.n_filtered == table.n_enumerated == 3
        assert set(table.frame["filter_reasons"]) == {("outcome_share",)}
        assert table.frame["outcome_share"].isna().all()
    for table in balances:
        assert table.n_filtered == table.n_enumerated == 3
        assert set(table.frame["filter_reasons"]) == {("representativeness",)}


def test_unreadable_excluded_outcome_filters_without_dropping_siblings() -> None:
    dirty = ("bad", pd.NA, [[1.0]], 10**309)
    for value in dirty:
        frame = pd.DataFrame(
            {"a": [1.0], "b": [2.0], "c": [value]},
            index=pd.Index([1], name="time"),
        )
        table = _generate(
            excluded_units=("c",),
            supplied_groups=(("a",), ("c",)),
            history=frame,
            selection_end=1,
        )
        rows = _by_treated(table)
        assert table.n_enumerated == 2
        assert rows[("a",)].filter_reasons == ()
        assert rows[("c",)].filter_reasons == ("excluded_unit", "outcome_share")
        assert pd.isna(rows[("c",)].outcome_share)
        assert rows[("c",)].proxy_role == "unavailable"
    numeric = _generate(
        excluded_units=("c",),
        supplied_groups=(("c",),),
        history=pd.DataFrame(
            {"a": [1.0], "b": [2.0], "c": [3.0]},
            index=pd.Index([1], name="time"),
        ),
        selection_end=1,
    )
    row = numeric.frame.iloc[0]
    assert row.filter_reasons == ("excluded_unit",)
    assert row.outcome_share == pytest.approx(1.0)


def _unit_rmse(
    frame: pd.DataFrame,
    unit: str,
    donors: tuple[str, ...],
) -> float:
    fitted = WeightedProportion().fit(
        frame.loc[:, list(donors)].to_numpy(dtype=float),
        frame.loc[:, unit].to_numpy(dtype=float),
    )
    return float(fitted.mse)


def _lowest_identities(
    units: tuple[str, ...],
    *,
    seed: int,
    limit: int,
) -> set[tuple[tuple[str, ...], tuple[str, ...]]]:
    identities: list[tuple[tuple[str, ...], tuple[str, ...], str, str]] = []
    for size in (1, 2):
        for treated in combinations(units, size):
            donors = tuple(name for name in units if name not in treated)
            candidate_id = json.dumps(
                [list(treated), list(donors)],
                separators=(",", ":"),
            )
            payload = json.dumps(
                [seed, list(treated), None],
                separators=(",", ":"),
                ensure_ascii=True,
            )
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            identities.append((treated, donors, digest, candidate_id))
    identities.sort(key=lambda item: (item[2], item[3]))
    return {(item[0], item[1]) for item in identities[:limit]}
