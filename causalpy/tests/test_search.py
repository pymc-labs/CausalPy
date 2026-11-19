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
"""Contract tests for constrained single-target search.

Screening and shortlist policy are checked without sampling. One light search
locks the freeze handoff.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from causalpy.design import (
    DecisionRule,
    DurationAudit,
    SyntheticControlPlanner,
)
from causalpy.duration import DurationRow
from causalpy.historical_validation import WindowReplay
from causalpy.pymc_models import WeightedSumFitter
from causalpy.search import (
    EvaluatedDesign,
    ParetoPoint,
    ScreenedCandidate,
    SearchResult,
    _best_sensitivity,
    _pareto,
    _recall_regret,
    _selection_frame,
    _shortlist,
    _with_dominance,
)

LIGHT = {
    "tune": 8,
    "draws": 8,
    "chains": 1,
    "cores": 1,
    "progressbar": False,
}


def _panel() -> pd.DataFrame:
    index = pd.Index(range(20), name="t")
    generator = np.random.default_rng(7)
    donor_0 = generator.normal(size=20)
    donor_1 = generator.normal(size=20)
    precise = 0.6 * donor_0 + 0.4 * donor_1
    relevant = donor_0 + 0.2
    return pd.DataFrame(
        {
            "donor_0": donor_0,
            "donor_1": donor_1,
            "precise": precise,
            "relevant": relevant,
            "other": generator.normal(size=20),
        },
        index=index,
    )


def _planner(
    frame: pd.DataFrame,
    treated: str = "precise",
    controls: tuple[str, ...] = ("donor_0", "donor_1"),
) -> SyntheticControlPlanner:
    return SyntheticControlPlanner(
        frame,
        treated,
        controls,
        DecisionRule("positive", "cumulative", threshold=0.0, probability=0.95),
        random_seed=7,
    )


def _search(planner: SyntheticControlPlanner, **overrides: object):
    arguments = {
        "durations": (4,),
        "selection_origins": (8,),
        "target_lift": 1.0,
        "max_fits": 0,
    }
    arguments.update(overrides)
    return planner.search(**arguments)


class _CountingFitter(WeightedSumFitter):
    calls = 0

    def fit(self, *args, **kwargs):  # noqa: ANN002, ANN003
        type(self).calls += 1
        raise RuntimeError("fit was called")


def test_group_request_is_rejected() -> None:
    """A sequence treated unit is a group and is rejected before screening."""
    planner = _planner(_panel())
    with pytest.raises(TypeError, match="Group size must be 1"):
        _search(planner, treated_unit=["precise", "relevant"])


def test_constraints_are_rejected_before_inference() -> None:
    """Include and exclude overlap fails before any model is fitted."""
    planner = _planner(_panel(), treated="other")
    planner.model = _CountingFitter(sample_kwargs=LIGHT)
    _CountingFitter.calls = 0
    with pytest.raises(ValueError, match="include and exclude overlap"):
        _search(
            planner,
            eligible_units=("precise", "relevant"),
            include=("precise",),
            exclude=("precise",),
        )
    assert _CountingFitter.calls == 0


def test_unknown_unit_is_rejected_before_inference() -> None:
    """A unit absent from the history is an error, not a skipped candidate."""
    planner = _planner(_panel())
    with pytest.raises(ValueError, match="Unknown unit"):
        _search(planner, eligible_units=("missing",))


def test_cost_and_spillover_skip_before_fitting() -> None:
    """Supplied constraints remove candidates before the duration evaluator."""
    frame = _panel()
    planner = _planner(frame, treated="other")
    planner.model = _CountingFitter(sample_kwargs=LIGHT)
    _CountingFitter.calls = 0
    result = _search(
        planner,
        eligible_units=("precise", "relevant"),
        spillover_exclude=("precise",),
        costs={"relevant": 100.0, "donor_0": 1.0, "donor_1": 1.0},
        max_cost=10.0,
        sample_kwargs=LIGHT,
        max_fits=4,
    )
    reasons = dict(
        zip(result.summary["treated_unit"], result.summary["status"], strict=True)
    )
    assert reasons == {"precise": "skipped", "relevant": "skipped"}
    assert _CountingFitter.calls == 0
    assert result.n_evaluated == 0
    assert result.recommendation().startswith("No supported recommendation")


def test_seed_does_not_depend_on_eligible_order() -> None:
    """Reversing the eligible list changes neither identity nor seed."""
    frame = _panel()
    forward = _search(
        _planner(frame, treated="other"),
        eligible_units=("precise", "relevant"),
    )
    reverse = _search(
        _planner(frame, treated="other"),
        eligible_units=("relevant", "precise"),
    )
    forward_seeds = dict(
        zip(forward.summary["candidate_id"], forward.summary["seed"], strict=True)
    )
    reverse_seeds = dict(
        zip(reverse.summary["candidate_id"], reverse.summary["seed"], strict=True)
    )
    assert forward_seeds == reverse_seeds


def test_shortlist_keeps_the_lowest_cost_boundary() -> None:
    """A weaker proxy is not discarded when it is the lowest supplied cost."""
    frame = _panel()
    result = _search(
        _planner(frame, treated="other"),
        eligible_units=("precise", "relevant"),
        costs={"precise": 100.0, "relevant": 1.0},
        representativeness={"precise": 0.9, "relevant": 0.1},
        shortlist_size=1,
    )
    unfinished = result.summary[result.summary["status"] == "unfinished"]
    assert list(unfinished["treated_unit"]) == ["relevant"]
    assert result.budget_exhausted
    assert result.recommendation().startswith("No supported recommendation")
    assert "not a launch recommendation" in result.recommendation()
    assert result.global_optimum is False
    assert not hasattr(result, "selected_candidate")


def test_screening_cap_is_not_input_order() -> None:
    """Candidates past the cap are disclosed, and the kept set ignores order."""
    frame = _panel()
    units = ("precise", "relevant", "other")
    forward = _search(
        _planner(frame, treated="donor_0", controls=("donor_1",)),
        eligible_units=units,
        donor_pools=(("donor_1",),),
        max_screened=1,
    )
    reverse = _search(
        _planner(frame, treated="donor_0", controls=("donor_1",)),
        eligible_units=tuple(reversed(units)),
        donor_pools=(("donor_1",),),
        max_screened=1,
    )
    kept = set(
        forward.summary.loc[forward.summary["status"] != "skipped", "candidate_id"]
    )
    assert len(kept) == 1
    assert kept == set(
        reverse.summary.loc[reverse.summary["status"] != "skipped", "candidate_id"]
    )
    assert "not the first candidates in input order" in (forward.disclosure or "")
    assert forward.n_unfinished == 1
    assert forward.n_skipped == 2


def test_screening_cap_does_not_drop_the_only_feasible_candidate() -> None:
    """An infeasible hash-first candidate must not consume the screening cap."""
    frame = _panel()
    units = ("precise", "relevant", "other")
    donors = ("donor_1",)
    identity = {
        unit: hashlib.sha256(
            json.dumps([unit, list(donors)], separators=(",", ":")).encode()
        ).hexdigest()
        for unit in units
    }
    first = min(identity, key=identity.get)
    costs = {unit: 50.0 if unit == first else 1.0 for unit in units}
    result = _search(
        _planner(frame, treated="donor_0", controls=("donor_1",)),
        eligible_units=units,
        donor_pools=(donors,),
        costs=costs,
        max_cost=10.0,
        max_screened=1,
    )
    rows = result.summary.set_index("treated_unit")
    assert "exceeds max_cost" in rows.loc[first, "reason"]
    assert "screening cap" not in rows.loc[first, "reason"]
    screened = result.summary[result.summary["proxy_score"].notna()]
    assert len(screened) == 1
    assert screened.iloc[0]["treated_unit"] != first
    assert "passed include, exclude, spillover" in (result.disclosure or "")


def test_fixed_target_pools_are_not_a_cartesian_product() -> None:
    """Each supplied pool is one candidate, not every combination of donors."""
    result = _search(
        _planner(_panel()),
        donor_pools=(("donor_0",), ("donor_1",)),
    )
    assert list(result.summary["treated_unit"]) == ["precise", "precise"]
    assert list(result.summary["donors"]) == ["donor_0", "donor_1"]
    assert result.mode == "fixed-target"


def test_free_target_ignores_the_constructor_target() -> None:
    """Eligible units, not the planner's treated unit, are the candidates."""
    result = _search(
        _planner(_panel(), treated="other"),
        eligible_units=("precise", "relevant"),
    )
    assert set(result.summary["treated_unit"]) == {"precise", "relevant"}
    assert result.mode == "free-target"
    assert "other" not in set(result.summary["treated_unit"])


def test_correlation_is_not_a_veto() -> None:
    """A low-correlation donor remains a screened candidate with a proxy score."""
    frame = _panel()
    frame["relevant"] = np.linspace(-3.0, 3.0, len(frame))
    result = _search(
        _planner(frame, treated="other"),
        eligible_units=("relevant",),
        donor_pools=(("donor_0", "donor_1"),),
    )
    row = result.summary.iloc[0]
    assert row["status"] in {"screened", "unfinished"}
    assert pd.notna(row["proxy_score"])
    assert row["correlation_min"] < 0.2
    assert "correlation" not in str(row["reason"])


def test_pareto_keeps_non_dominated_points_and_withholds_a_winner() -> None:
    """Dominance needs a strict improvement and no worse present axis."""
    precise = ParetoPoint("precise", "precise", 4, 0.2, 10.0, 0.1, False)
    relevant = ParetoPoint("relevant", "relevant", 4, 0.8, 1.0, 0.9, False)
    tied = ParetoPoint("tied", "tied", 4, 0.8, 1.0, 0.9, False)
    points = (precise, relevant, tied)
    marked = tuple(_with_dominance(point, points) for point in points)
    assert [point.candidate_id for point in marked if point.dominated] == ["precise"]
    assert [point.candidate_id for point in marked if not point.dominated] == [
        "relevant",
        "tied",
    ]


def test_recall_is_one_only_when_the_proxy_best_is_evaluated() -> None:
    """Recall and regret compare the proxy best with the evaluated best."""
    low = SimpleNamespace(rows=(_duration_row("historical-descriptive", 0.4),))
    high = SimpleNamespace(rows=(_duration_row("historical-descriptive", 0.9),))
    evaluated = (
        EvaluatedDesign("precise", "precise", (), 1, low),
        EvaluatedDesign("relevant", "relevant", (), 2, high),
    )
    recalled = (
        ScreenedCandidate(
            "precise", "precise", (), "evaluated", "", proxy_score=0.5, seed=1
        ),
        ScreenedCandidate(
            "relevant", "relevant", (), "evaluated", "", proxy_score=0.1, seed=2
        ),
    )
    recall, regret, reason = _recall_regret(recalled, evaluated, True)
    assert reason is None
    assert recall == 1.0
    assert regret == 0.0
    missed_proxy = (
        ScreenedCandidate(
            "precise", "precise", (), "evaluated", "", proxy_score=0.1, seed=1
        ),
        recalled[1],
    )
    missed, gap, missed_reason = _recall_regret(missed_proxy, evaluated, True)
    assert missed_reason is None
    assert missed == 0.0
    assert gap == pytest.approx(0.5)
    withheld, _, withheld_reason = _recall_regret(recalled, evaluated, False)
    assert withheld is None
    assert "not exhaustive" in str(withheld_reason)


def test_freeze_refuses_a_candidate_that_was_not_evaluated() -> None:
    """Budget exhaustion does not invent a winner to freeze."""
    result = _search(_planner(_panel()), donor_pools=(("donor_0", "donor_1"),))
    assert result.n_evaluated == 0
    with pytest.raises(ValueError, match="no evaluated comparison"):
        result.freeze("precise", 4)


def test_planner_does_not_select_a_duration() -> None:
    """The landed freeze API is the comparison, not a planner selector."""
    planner = _planner(_panel())
    assert callable(planner.search)
    assert not hasattr(planner, "select_duration")
    assert not hasattr(planner, "freeze")


def test_handoff_freezes_the_chosen_candidate_and_audits_the_held_origin() -> None:
    """Search fits the selection origin, then freeze and audit fit the held one."""
    frame = _panel()
    planner = _planner(frame)
    result = planner.search(
        durations=(4,),
        selection_origins=(8,),
        audit_origins=(16,),
        absolute_lifts=(1.0,),
        target_lift=1.0,
        donor_pools=(("donor_0", "donor_1"),),
        sample_kwargs=LIGHT,
        max_fits=4,
    )
    assert result.n_evaluated == 1
    assert result.evaluation_exhaustive
    comparison = result.evaluated[0].comparison
    assert 16 not in [window.origin for window in comparison.windows]
    assert comparison.n_inference_fits == 1
    specification = result.freeze(result.evaluated[0].candidate_id, 4)
    audit = specification.audit(frame)
    assert isinstance(audit, DurationAudit)
    assert audit.selected_duration == 4
    assert audit.n_inference_fits >= 1
    assert audit.recommendation().startswith("No supported recommendation")
    assert result.proxy_recall == 1.0
    assert result.global_optimum is False


def test_frozen_audit_samples_the_candidate_seed_not_the_parent(monkeypatch) -> None:
    """Default-model freeze must not sample the parent template seed."""
    frame = _panel()
    planner = _planner(frame)
    result = planner.search(
        durations=(4,),
        selection_origins=(8,),
        audit_origins=(16,),
        absolute_lifts=(1.0,),
        target_lift=1.0,
        donor_pools=(("donor_0",), ("donor_1",)),
        sample_kwargs=LIGHT,
        max_fits=4,
    )
    assert result.n_evaluated == 2
    assert planner.model.sample_kwargs["random_seed"] == 7
    captured: list[int] = []

    def _capture(*args, **kwargs):
        captured.append(kwargs["template"].sample_kwargs["random_seed"])
        return (
            WindowReplay(
                plan=args[3],
                status="scored",
                reason=None,
                bias_cumulative=0.0,
                bias_average=0.0,
                n_inference_fits=1,
            ),
            1,
        )

    monkeypatch.setattr("causalpy.historical_validation._replay_window", _capture)
    sampled: list[int] = []
    for item in result.evaluated:
        specification = result.freeze(item.candidate_id, 4)
        assert specification.random_seed != 7
        assert specification.sample_kwargs["random_seed"] == specification.random_seed
        specification.audit(frame)
        sampled.append(specification.random_seed)
    assert len(set(sampled)) == 2
    assert captured == sampled


def _duration_row(
    level: str,
    share: float,
    *,
    duration: int = 4,
    scored: int = 1,
) -> DurationRow:
    return DurationRow(
        duration=duration,
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
        scored_count=scored,
        failed_count=0 if scored else 1,
        requested_count=1,
        not_run_count=0,
        n_inference_fits=scored,
        cost=None,
        recovery_mean_error=None,
        truth_covers_share=None,
    )


def _search_result(comparison: object, candidate_id: str = "id") -> SearchResult:
    design = EvaluatedDesign(candidate_id, "precise", ("donor_0",), 7, comparison)
    candidate = ScreenedCandidate(
        candidate_id,
        "precise",
        ("donor_0",),
        "evaluated",
        "",
        proxy_score=1.0,
        seed=7,
    )
    return SearchResult(
        status="mixed",
        mode="fixed-target",
        candidates=(candidate,),
        evaluated=(design,),
        pareto=_pareto((design,), (candidate,), False, False),
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


def test_shortlist_fits_in_candidate_id_order() -> None:
    """A shortlist that fits is fitted by candidate id, not generation order."""
    later = ScreenedCandidate(
        "zzz", "zzz", ("donor_0",), "screened", "", proxy_score=0.1, seed=1
    )
    earlier = ScreenedCandidate(
        "aaa", "aaa", ("donor_0",), "screened", "", proxy_score=9.0, seed=2
    )
    kept, note = _shortlist((later, earlier), 8)
    assert [item.candidate_id for item in kept] == ["aaa", "zzz"]
    assert "candidate-id order" in note


def test_budget_stop_discloses_candidate_id_order() -> None:
    """Budget exhaustion names the candidate-id order it actually follows."""
    result = _search(
        _planner(_panel(), treated="other"),
        eligible_units=("precise", "relevant"),
    )
    assert "candidate-id order" in result.disclosure
    assert "not input order" in result.disclosure


def test_selection_frame_resolves_exclusion_aliases() -> None:
    """Screening drops the same labels the duration evaluator drops."""
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    frame = pd.DataFrame({"precise": np.arange(12, dtype=float)}, index=index)
    screened = _selection_frame(frame, (index[8],), ("2020-01-03",))
    assert index[2] not in screened.index
    assert len(screened) == 7
    dropped = _selection_frame(_panel(), (8,), (np.int64(2),))
    assert 2 not in dropped.index
    with pytest.raises(ValueError, match="not in the history index"):
        _selection_frame(frame, (index[8],), ("not-a-label",))
    with pytest.raises(TypeError, match="not a single string"):
        _selection_frame(frame, (index[8],), "2020-01-03")


def test_audit_nan_does_not_abort_the_other_candidate(monkeypatch) -> None:
    """A reserved-audit NaN fails only its candidate, and the real gap remains."""
    frame = _panel()
    frame.loc[16, "precise"] = np.nan
    seen: list[str] = []

    def _fake(child, durations, **kwargs):
        seen.append(child.treated_unit)
        if child.treated_unit == "precise":
            assert np.isnan(child.history.loc[16, "precise"])
        return SimpleNamespace(
            status="historical-descriptive",
            rows=(_duration_row("historical-descriptive", 0.5),),
            n_inference_fits=1,
            recommendation=lambda: "No supported recommendation",
        )

    monkeypatch.setattr("causalpy.search._compare_durations", _fake)
    result = _search(
        _planner(frame, treated="other"),
        eligible_units=("precise", "relevant"),
        audit_origins=(16,),
        max_fits=4,
    )
    assert seen == ["precise", "relevant"]
    assert result.n_evaluated == 2
    assert np.isnan(frame.loc[16, "precise"])


def test_failed_fit_is_not_a_pareto_point_or_a_handoff() -> None:
    """A zero share from a failed fit is not a sensitivity and cannot be frozen."""
    failed = _duration_row("fit-failed", 0.0, scored=0)
    scored = _duration_row("historical-descriptive", 0.0, duration=8)
    comparison = SimpleNamespace(
        rows=(failed, scored),
        status="mixed",
        freeze=lambda duration: duration,
    )
    result = _search_result(comparison)
    assert [point.duration for point in result.pareto] == [8]
    assert result.pareto[0].sensitivity == 0.0
    with pytest.raises(ValueError, match="not an assessed design"):
        result.freeze("id", 4)
    assert result.freeze("id", 8) == 8
    only_failed = SimpleNamespace(rows=(failed,), status="fit-failed")
    assert (
        _pareto(
            (EvaluatedDesign("id", "precise", ("donor_0",), 7, only_failed),),
            (
                ScreenedCandidate(
                    "id",
                    "precise",
                    ("donor_0",),
                    "fit-failed",
                    "",
                    proxy_score=1.0,
                    seed=7,
                ),
            ),
            False,
            False,
        )
        == ()
    )
    assert _best_sensitivity(only_failed) is None
