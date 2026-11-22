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
"""Refusal and handoff contracts that the happy-path search tests do not reach."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from causalpy.design import DecisionRule, SyntheticControlPlanner
from causalpy.duration import DurationRow
from causalpy.search import (
    EvaluatedDesign,
    ParetoPoint,
    ScreenedCandidate,
    SearchResult,
    _assessed_share,
    _child_planner,
    _dominates,
    _evaluate_shortlist,
    _min_correlation,
    _proxy,
    _recall_regret,
    _screen_one,
    _search_status,
)

LIGHT = {"tune": 8, "draws": 8, "chains": 1, "cores": 1, "progressbar": False}


def _panel() -> pd.DataFrame:
    index = pd.Index(range(20), name="t")
    generator = np.random.default_rng(7)
    donor_0 = generator.normal(size=20)
    donor_1 = generator.normal(size=20)
    return pd.DataFrame(
        {
            "donor_0": donor_0,
            "donor_1": donor_1,
            "precise": 0.6 * donor_0 + 0.4 * donor_1,
            "relevant": donor_0 + 0.2,
            "other": generator.normal(size=20),
        },
        index=index,
    )


def _planner(
    frame: pd.DataFrame | None = None,
    treated: str = "precise",
    *,
    seed: int | None = 7,
) -> SyntheticControlPlanner:
    return SyntheticControlPlanner(
        _panel() if frame is None else frame,
        treated,
        ("donor_0", "donor_1"),
        DecisionRule("positive", "cumulative", threshold=0.0, probability=0.95),
        random_seed=seed,
    )


def _positive_panel() -> pd.DataFrame:
    frame = _panel()
    return frame - float(frame.min().min()) + 1.0


def _search(planner: SyntheticControlPlanner, **overrides: object):
    arguments = {
        "durations": (4,),
        "selection_origins": (8,),
        "target_lift": 1.0,
        "max_fits": 0,
    }
    arguments.update(overrides)
    return planner.search(**arguments)


def _row(level: str, share: float) -> DurationRow:
    return DurationRow(
        duration=4,
        target_lift=1.0,
        target_scale="absolute",
        evidence_level=level,
        calibration_evidence_level=None,
        persistent_bias=None,
        zero_lift_bias_cumulative=None,
        zero_lift_bias_average=None,
        zero_lift_mean_absolute_error=None,
        historical_detection_count=0,
        historical_detection_share=share,
        scored_count=1,
        failed_count=0,
        requested_count=1,
        not_run_count=0,
        n_inference_fits=1,
        cost=None,
        recovery_mean_error=None,
        truth_covers_share=None,
    )


def _comparison(status: str, fits: int = 1, rows: tuple[DurationRow, ...] = ()):
    return SimpleNamespace(
        status=status,
        n_inference_fits=fits,
        rows=rows,
        recommendation=lambda: f"{status} stopped the comparison",
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"group_size": 2}, "Group size must be 1"),
        ({"group_size": True}, "integer 1"),
        ({"eligible_units": "precise"}, "sequence of unit names"),
        ({"eligible_units": (1,)}, "Group size must be 1"),
        ({"shortlist_size": 0}, "at least 1"),
        ({"screen_holdout": True}, "at least 1"),
        ({"outcome_share": "wide"}, "pair of finite"),
        ({"outcome_share": (True, 1.0)}, "not booleans"),
        ({"outcome_share": (0.0, "high")}, "must be floats"),
        ({"outcome_share": (0.8, 0.2)}, "min <= max"),
        ({"outcome_share": (0.0, np.inf)}, "finite"),
        ({"costs": ["precise"]}, "mapping"),
        ({"costs": {1: 1.0}}, "keys must be unit names"),
        ({"costs": {"precise": "high"}}, "finite number"),
        ({"costs": {"precise": np.inf}}, "must be finite"),
        ({"costs": {"precise": -1.0}}, "at least"),
        ({"include": "precise"}, "sequence of unit names"),
        ({"include": (1,)}, "Grouped units"),
        ({"include": ("precise", "precise")}, "duplicate"),
        (
            {"include": ("precise",), "spillover_exclude": ("precise",)},
            "spillover_exclude overlap",
        ),
        (
            {"exclude": ("donor_0",), "spillover_exclude": ("donor_0",)},
            "skip reason is unambiguous",
        ),
        ({"exclude": ("missing",)}, "Unknown units"),
        (
            {"treated_unit": "precise", "eligible_units": ("relevant",)},
            "not both",
        ),
        (
            {
                "eligible_units": ("precise", "relevant"),
                "donor_pools": (("donor_0",), ("donor_1",)),
            },
            "one donor pool",
        ),
        ({"donor_pools": "donor_0"}, "donor-name sequences"),
        ({"treated_unit": "missing"}, "Unknown treated unit"),
        ({"max_cost": 1.0}, "max_cost requires"),
        ({"min_representativeness": 0.2}, "not inferred"),
        ({"sample_kwargs": {"random_seed": 3}}, "master seed"),
        ({"selection_origins": ()}, "non-empty"),
        ({"selection_origins": (99,)}, "Unknown selection origins"),
        ({"costs": {"donor_0": 1.0}}, "no entry for candidate target"),
        (
            {"representativeness": {"donor_0": 1.0}},
            "not inferred",
        ),
    ],
)
def test_search_rejects_a_bad_request_before_fitting(
    overrides: dict, match: str
) -> None:
    """A malformed request is an error, not a skipped or imputed candidate."""
    with pytest.raises((TypeError, ValueError), match=match):
        _search(_planner(), **overrides)


def test_unseeded_planner_cannot_search() -> None:
    """Candidate seeds are undefined without the planner seed."""
    with pytest.raises(ValueError, match="unseeded search is rejected"):
        _search(_planner(seed=None))


def test_rebind_rejects_a_bad_child_before_copying_history() -> None:
    """Child construction keeps the constructor's donor and seed checks."""
    planner = _planner()
    with pytest.raises(TypeError, match="not a single string"):
        planner._rebind("precise", "donor_0", 1)
    with pytest.raises(ValueError, match="at least one donor"):
        planner._rebind("precise", (), 1)
    with pytest.raises(TypeError, match="unit-name strings"):
        planner._rebind("precise", (1,), 1)
    with pytest.raises(ValueError, match="duplicate donor"):
        planner._rebind("precise", ("donor_0", "donor_0"), 1)
    with pytest.raises(ValueError, match="cannot also be a control"):
        planner._rebind("precise", ("precise",), 1)
    with pytest.raises(ValueError, match="missing columns"):
        planner._rebind("precise", ("absent",), 1)
    with pytest.raises(TypeError, match="random_seed"):
        planner._rebind("precise", ("donor_0",), True)


def test_history_columns_must_be_unit_names() -> None:
    """An integer column is not a unit name the constraints can name."""
    planner = _planner()
    renamed = planner.history.copy()
    renamed.columns = list(range(renamed.shape[1]))
    planner.history = renamed
    with pytest.raises(TypeError, match="History columns must be unit names"):
        _search(planner)


def test_default_pool_is_the_constructor_controls_and_records_share() -> None:
    """Omitting donor pools uses the constructor controls, not every column."""
    frame = _positive_panel()
    result = _search(_planner(frame), outcome_share=(0.0, 1.0))
    row = result.summary.iloc[0]
    assert row["treated_unit"] == "precise"
    assert row["donors"] == "donor_0,donor_1"
    assert row["status"] in {"screened", "unfinished"}
    selection = frame.loc[frame.index < 8, list(frame.columns)]
    expected = float(selection["precise"].sum() / selection.to_numpy().sum())
    assert row["outcome_share"] == pytest.approx(expected)
    table = result.pareto_table
    assert table.empty


def test_outcome_share_outside_the_bounds_is_a_skip_not_a_fit() -> None:
    """A supplied share bound removes the candidate before the evaluator."""
    result = _search(_planner(_positive_panel()), outcome_share=(0.99, 1.0))
    row = result.summary.iloc[0]
    assert row["status"] == "skipped"
    assert "outcome share" in row["reason"]
    assert result.n_evaluated == 0


def test_outcome_share_rejects_an_unusable_selection_history() -> None:
    """Share is an error when the pre-origin window cannot define a fraction."""
    with pytest.raises(ValueError, match="selection-history rows"):
        _search(_planner(), selection_origins=(0,), outcome_share=(0.0, 1.0))
    zeros = _panel()
    zeros.loc[zeros.index < 8] = 0.0
    with pytest.raises(ValueError, match="positive sum"):
        _search(_planner(zeros), outcome_share=(0.0, 1.0))
    non_finite = _panel()
    non_finite.loc[0, "other"] = np.inf
    with pytest.raises(ValueError, match="non-finite selection-history"):
        _search(_planner(non_finite), outcome_share=(0.0, 1.0))


def test_low_representativeness_is_skipped_when_the_score_is_supplied() -> None:
    """The supplied score can exclude a target. It is not inferred."""
    result = _search(
        _planner(),
        representativeness={"precise": 0.1, "donor_0": 1.0, "donor_1": 1.0},
        min_representativeness=0.5,
    )
    row = result.summary.iloc[0]
    assert row["status"] == "skipped"
    assert "representativeness" in row["reason"]
    assert row["relevance"] == pytest.approx(0.1)


def test_repeated_pools_collapse_and_the_collapse_is_disclosed() -> None:
    """Sorting makes reversed donor lists one candidate, not two."""
    result = _search(
        _planner(),
        donor_pools=(("donor_1", "donor_0"), ("donor_0", "donor_1")),
    )
    assert len(result.summary) == 1
    assert "collapsed" in (result.disclosure or "")


def test_short_or_non_finite_selection_history_fails_the_screen() -> None:
    """A proxy that cannot be computed is a screen failure, not an abort."""
    short = _search(_planner(), selection_origins=(3,), screen_holdout=4)
    assert short.summary.iloc[0]["status"] == "screen-failed"
    assert "holdout" in short.summary.iloc[0]["reason"]
    dirty = _panel()
    dirty.loc[0, "precise"] = np.nan
    # The constructor rejects a non-finite selected column, so the NaN has to
    # land in a column that becomes the target only at search time.
    dirty.loc[0, "precise"] = _panel().loc[0, "precise"]
    dirty.loc[0, "other"] = np.nan
    failed = _search(
        _planner(dirty, treated="precise"),
        eligible_units=("other",),
        donor_pools=(("donor_0", "donor_1"),),
    )
    assert failed.summary.iloc[0]["status"] == "screen-failed"
    assert "non-finite" in failed.summary.iloc[0]["reason"]


def test_a_flat_series_is_omitted_from_the_correlation_diagnostic() -> None:
    """Zero variance is not a correlation and does not veto the candidate."""
    frame = _panel().iloc[:8]
    assert _min_correlation(frame, "precise", ("donor_0",)) is not None
    flat_target = frame.copy()
    flat_target["precise"] = 1.0
    assert _min_correlation(flat_target, "precise", ("donor_0",)) is None
    flat_donor = frame.copy()
    flat_donor["donor_0"] = 1.0
    assert _min_correlation(
        flat_donor, "precise", ("donor_0", "donor_1")
    ) == pytest.approx(
        float(np.corrcoef(flat_donor["precise"], flat_donor["donor_1"])[0, 1])
    )
    assert _min_correlation(flat_donor, "precise", ("donor_0",)) is None


def test_screen_failure_stays_on_the_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken proxy records the failure and leaves the other candidate alone."""

    class _Boom:
        def fit(self, x, y):  # noqa: ANN001
            raise ValueError("singular")

    monkeypatch.setattr("causalpy.search.WeightedProportion", _Boom)
    result = _search(_planner(), eligible_units=("precise", "relevant"))
    assert set(result.summary["status"]) == {"screen-failed"}
    assert result.summary["reason"].str.contains("ValueError").all()
    assert result.n_evaluated == 0


def test_proxy_rejects_weights_that_are_not_a_convex_combination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-convex or non-finite weight vector is a screen failure."""

    class _Bad:
        def fit(self, x, y):  # noqa: ANN001
            self.coef_ = np.array([2.0, -1.0])

    monkeypatch.setattr("causalpy.search.WeightedProportion", _Bad)
    convex = _search(_planner())
    assert "convex combination" in convex.summary.iloc[0]["reason"]

    class _Shape:
        def fit(self, x, y):  # noqa: ANN001
            self.coef_ = np.array([1.0])

    monkeypatch.setattr("causalpy.search.WeightedProportion", _Shape)
    shape = _search(_planner())
    assert "non-finite weights" in shape.summary.iloc[0]["reason"]

    class _Finite:
        def fit(self, x, y):  # noqa: ANN001
            self.coef_ = np.array([0.5, 0.5])

    monkeypatch.setattr("causalpy.search.WeightedProportion", _Finite)
    monkeypatch.setattr("causalpy.search.np.sqrt", lambda values: np.nan)
    rmse = _search(_planner())
    assert "non-finite holdout RMSE" in rmse.summary.iloc[0]["reason"]


def test_holdout_proxy_refuses_an_empty_holdout() -> None:
    """A zero holdout cannot score a candidate."""
    proxy = _proxy(_panel().iloc[:5], "precise", ("donor_0", "donor_1"), 0)
    assert proxy.reason is not None
    assert "cannot support" in proxy.reason


def test_screen_one_records_a_constraint_the_batch_already_removed() -> None:
    """The per-candidate check still skips a pending design that has no donors."""
    pending = ScreenedCandidate("id", "precise", (), "pending", "")
    skipped = _screen_one(
        pending,
        selection=_panel().iloc[:8],
        holdout=4,
        min_donors=1,
        include=None,
        exclude=(),
        spillover=(),
        columns=tuple(_panel().columns),
        costs=None,
        max_cost=None,
        shares=None,
        share_bounds=None,
        relevance=None,
        min_representativeness=None,
        master_seed=7,
    )
    assert skipped.status == "skipped"
    assert "donors" in skipped.reason
    excluded = ScreenedCandidate("id", "precise", ("donor_0",), "pending", "")
    omitted = _screen_one(
        excluded,
        selection=_panel().iloc[:8],
        holdout=4,
        min_donors=1,
        include=("donor_0",),
        exclude=(),
        spillover=(),
        columns=tuple(_panel().columns),
        costs=None,
        max_cost=None,
        shares=None,
        share_bounds=None,
        relevance=None,
        min_representativeness=None,
        master_seed=7,
    )
    assert omitted.status == "skipped"
    assert omitted.reason == "not in include"
    removed = _screen_one(
        excluded,
        selection=_panel().iloc[:8],
        holdout=4,
        min_donors=1,
        include=None,
        exclude=("precise",),
        spillover=(),
        columns=tuple(_panel().columns),
        costs=None,
        max_cost=None,
        shares=None,
        share_bounds=None,
        relevance=None,
        min_representativeness=None,
        master_seed=7,
    )
    assert removed.status == "skipped"
    assert removed.reason == "exclude"
    unknown = ScreenedCandidate("id", "absent", ("donor_0",), "pending", "")
    with pytest.raises(ValueError, match="Unknown treated unit"):
        _screen_one(
            unknown,
            selection=_panel().iloc[:8],
            holdout=4,
            min_donors=1,
            include=None,
            exclude=(),
            spillover=(),
            columns=("donor_0", "donor_1"),
            costs=None,
            max_cost=None,
            shares=None,
            share_bounds=None,
            relevance=None,
            min_representativeness=None,
            master_seed=7,
        )


def test_shortlist_fills_by_proxy_after_the_boundary() -> None:
    """A size larger than the boundary set keeps the next-lowest proxy scores."""
    result = _search(
        _planner(treated="other"),
        eligible_units=("precise", "relevant", "other"),
        donor_pools=(("donor_0", "donor_1"),),
        shortlist_size=2,
    )
    unfinished = result.summary[result.summary["status"] == "unfinished"]
    outside = result.summary[result.summary["status"] == "screened"]
    assert len(unfinished) == 2
    assert len(outside) == 1
    assert "outside the shortlist" in outside.iloc[0]["reason"]
    assert outside.iloc[0]["proxy_score"] == max(result.summary["proxy_score"])


def test_size_one_shortlist_keeps_the_lowest_share_boundary() -> None:
    """Lowest outcome share is kept before the tightest proxy when size is one."""
    index = pd.Index(range(20), name="t")
    donor_0 = np.linspace(10.0, 20.0, 20)
    donor_1 = np.linspace(12.0, 22.0, 20)
    frame = pd.DataFrame(
        {
            "donor_0": donor_0,
            "donor_1": donor_1,
            "tight": 0.6 * donor_0 + 0.4 * donor_1,
            "loose": np.full(20, 0.1),
            "other": np.linspace(8.0, 18.0, 20),
        },
        index=index,
    )
    result = _search(
        _planner(frame, treated="other"),
        eligible_units=("loose", "tight"),
        donor_pools=(("donor_0", "donor_1"),),
        outcome_share=(0.0, 1.0),
        shortlist_size=1,
    )
    unfinished = result.summary[result.summary["status"] == "unfinished"]
    assert list(unfinished["treated_unit"]) == ["loose"]
    tight = result.summary.set_index("treated_unit").loc["tight"]
    assert tight["proxy_score"] < unfinished.iloc[0]["proxy_score"]


def test_dominance_rejects_a_worse_cost_or_relevance() -> None:
    """A better sensitivity does not hide a worse supplied cost or relevance."""
    cheap = ParetoPoint("cheap", "a", 4, 0.5, 1.0, 0.5, False)
    costly = ParetoPoint("costly", "b", 4, 0.5, 9.0, 0.5, False)
    broad = ParetoPoint("broad", "c", 4, 0.5, 1.0, 0.9, False)
    narrow = ParetoPoint("narrow", "d", 4, 0.5, 1.0, 0.1, False)
    assert not _dominates(costly, cheap)
    assert not _dominates(narrow, broad)
    assert _dominates(cheap, costly)
    assert _dominates(broad, narrow)


def test_pareto_table_omits_axes_the_caller_did_not_supply() -> None:
    """Missing cost and relevance are not columns of implied zeros."""
    point = ParetoPoint("id", "precise", 4, 0.5, None, None, False)
    result = SearchResult(
        status="historical-descriptive",
        mode="fixed-target",
        candidates=(),
        evaluated=(),
        pareto=(point,),
        n_screened=1,
        n_evaluated=1,
        n_skipped=0,
        n_failed=0,
        n_unfinished=0,
        n_inference_fits=1,
        budget_exhausted=False,
        evaluation_exhaustive=True,
        proxy_recall=None,
        proxy_regret=None,
        partial_reason=None,
        selection_policy="",
        disclosure="",
        recommendation_text="",
        sample_kwargs={},
        random_seed=7,
    )
    assert list(result.pareto_table.columns) == [
        "candidate_id",
        "treated_unit",
        "duration",
        "sensitivity",
        "dominated",
    ]


def test_comparison_stop_and_failure_statuses_do_not_select_a_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed or exhausted comparison remains evidence, not a launch."""
    seen: list[str] = []

    def install(status: str, fits: int) -> None:
        def _fake(planner, *args, **kwargs):  # noqa: ANN001
            seen.append(planner.treated_unit)
            rows = (_row("historical-descriptive", 0.4),) if status == "mixed" else ()
            this = (
                "fit-failed"
                if status == "mixed" and planner.treated_unit == "precise"
                else status
            )
            if status == "mixed" and planner.treated_unit != "precise":
                this = "historical-descriptive"
            return _comparison(
                this, fits=fits, rows=rows if this == "historical-descriptive" else ()
            )

        monkeypatch.setattr("causalpy.search._compare_durations", _fake)

    install("memory-exhausted", 0)
    memory = _search(_planner(), max_fits=2)
    assert memory.status == "memory-exhausted"
    assert memory.n_evaluated == 0
    assert "memory limit" in memory.recommendation()

    install("budget-exhausted", 0)
    budget = _search(_planner(), max_fits=2)
    assert budget.status == "budget-exhausted"
    assert budget.n_evaluated == 0
    assert budget.budget_exhausted

    install("fit-failed", 1)
    failed = _search(_planner(), max_fits=2)
    assert failed.status == "fit-failed"
    assert "failed" in failed.recommendation()

    install("insufficient-history", 0)
    thin = _search(_planner(), max_fits=2)
    assert thin.status == "insufficient-history"

    install("mixed", 1)
    mixed = _search(
        _planner(treated="other"),
        eligible_units=("precise", "relevant"),
        max_fits=4,
    )
    assert mixed.status == "mixed"
    assert seen[-2:] == ["precise", "relevant"]


def test_seedless_or_unseeded_fitting_is_refused() -> None:
    """A candidate without a seed cannot reach the duration evaluator."""
    planner = _planner()
    bare = ScreenedCandidate("id", "precise", ("donor_0",), "screened", "", seed=None)
    with pytest.raises(RuntimeError, match="without a seed"):
        _child_planner(planner, bare)
    planner.random_seed = None
    with pytest.raises(RuntimeError, match="unseeded"):
        _evaluate_shortlist(
            planner,
            (bare,),
            durations=(4,),
            selection_origins=(8,),
            audit_origins=(),
            relative_lifts=None,
            absolute_lifts=None,
            target_lift=1.0,
            target_scale=None,
            exclusions=(),
            cadence=None,
            max_fits=1,
            fit_cost=None,
            min_training_periods=2,
            sample_kwargs=None,
        )
    planner.random_seed = 7
    with pytest.raises(RuntimeError, match="without a seed"):
        _evaluate_shortlist(
            planner,
            (bare,),
            durations=(4,),
            selection_origins=(8,),
            audit_origins=(),
            relative_lifts=None,
            absolute_lifts=None,
            target_lift=1.0,
            target_scale=None,
            exclusions=(),
            cadence=None,
            max_fits=1,
            fit_cost=None,
            min_training_periods=2,
            sample_kwargs=None,
        )


def test_recall_withholds_when_share_or_proxy_is_absent() -> None:
    """Recall needs a finite assessed share and a proxy on that same candidate."""
    nan_row = _row("historical-descriptive", float("nan"))
    assert _assessed_share(nan_row) is None
    design = EvaluatedDesign("id", "precise", (), 1, SimpleNamespace(rows=(nan_row,)))
    candidate = ScreenedCandidate("id", "precise", (), "evaluated", "", proxy_score=0.2)
    recall, regret, reason = _recall_regret((candidate,), (design,), True)
    assert recall is None and regret is None
    assert reason is not None and "historical detection share" in reason
    scored = EvaluatedDesign(
        "id",
        "precise",
        (),
        1,
        SimpleNamespace(rows=(_row("historical-descriptive", 0.4),)),
    )
    unproxied = ScreenedCandidate(
        "id", "precise", (), "evaluated", "", proxy_score=None
    )
    recall, regret, reason = _recall_regret((unproxied,), (scored,), True)
    assert recall is None
    assert reason is not None and "proxy score and a sensitivity" in reason
    fallback = _search_status(
        (ScreenedCandidate("id", "precise", (), "screened", ""),),
        (
            EvaluatedDesign(
                "id",
                "precise",
                (),
                1,
                SimpleNamespace(status="budget-exhausted", rows=()),
            ),
        ),
        None,
    )
    assert fallback == "insufficient-history"
    described = _search_status(
        (ScreenedCandidate("id", "precise", (), "evaluated", ""),),
        (
            EvaluatedDesign(
                "id",
                "precise",
                (),
                1,
                SimpleNamespace(status="budget-exhausted", rows=()),
            ),
        ),
        None,
    )
    assert described == "historical-descriptive"
