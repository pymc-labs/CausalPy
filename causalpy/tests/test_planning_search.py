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
"""Grouped treatment search: routing, selection, and published-surface refusals."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from causalpy.planning.contracts import (
    Capabilities,
    DecisionRule,
    PowerCriterion,
)
from causalpy.planning.power import (
    EffectByDuration,
    MinimumDetectableEffect,
    PowerPoint,
    StudyRecord,
)
from causalpy.planning.search import (
    SEARCH_COLUMNS,
    ComputationalBudget,
    FrozenDesign,
    _pareto,
    compare_screening,
    plot_search,
    sampler_seed,
    search,
    validate_selection,
)

UNITS = ("a", "b", "c", "d")
DURATION = 4
WORTHWHILE = 8.0
LIGHT = {
    "draws": 20,
    "tune": 20,
    "chains": 1,
    "progressbar": False,
}


def _panel(n_rows: int = 16) -> pd.DataFrame:
    index = pd.RangeIndex(n_rows, name="time")
    rng = np.random.default_rng(0)
    factor = np.cumsum(rng.normal(0.0, 1.0, size=n_rows))
    frame = pd.DataFrame(index=index)
    for shift, name in enumerate(UNITS):
        frame[name] = factor + shift + rng.normal(0.0, 0.2, size=n_rows)
    return frame


def _decision() -> DecisionRule:
    return DecisionRule("positive", "cumulative_total", 0.0, 0.8)


def _criterion() -> PowerCriterion:
    return PowerCriterion(
        target_power=0.8,
        false_positive_tolerance=0.15,
        precision_target=None,
        monte_carlo_error_tolerance=0.4,
        qualification_scope="post_search",
    )


def _budget(fits: int = 500) -> ComputationalBudget:
    return ComputationalBudget(fits)


def _costs() -> tuple[tuple[str, float], ...]:
    return (("a", 1.0), ("b", 5.0), ("c", 9.0), ("d", 2.0))


def _selection_end(frame: pd.DataFrame) -> int:
    return int(frame.index[-DURATION - 1])


def _study(power: float = 0.95, false_positive: float = 0.0) -> EffectByDuration:
    point = PowerPoint(
        duration=DURATION,
        effect=WORTHWHILE,
        scale="absolute",
        probability=power,
        monte_carlo_error=0.05,
        n_decided=4,
        n_replicates=4,
        posterior_mean=WORTHWHILE,
        posterior_tail_mean=0.01,
        interval_width=1.0,
        bias=0.0,
        coverage=1.0,
        false_positive_rate=false_positive,
        wrong_sign_rate=float("nan"),
        detection=power,
        failed_runs=0,
        precision_met=None,
    )
    zero = PowerPoint(
        duration=DURATION,
        effect=0.0,
        scale="absolute",
        probability=false_positive,
        monte_carlo_error=0.05,
        n_decided=0,
        n_replicates=4,
        posterior_mean=0.0,
        posterior_tail_mean=0.4,
        interval_width=1.0,
        bias=0.0,
        coverage=1.0,
        false_positive_rate=false_positive,
        wrong_sign_rate=float("nan"),
        detection=false_positive,
        failed_runs=0,
        precision_met=None,
    )
    mde = MinimumDetectableEffect(
        duration=DURATION,
        target_power=0.8,
        effect=WORTHWHILE,
        status="attained",
        convention="published",
        minimum_worthwhile_effect=WORTHWHILE,
        worthwhile_comparison="at_or_above",
        reason=None,
        lower_effect=None,
        upper_effect=None,
        lower_power=None,
        upper_power=None,
        lower_monte_carlo_error=None,
        upper_monte_carlo_error=None,
    )
    record = StudyRecord(
        scenario_family="correctly_specified_noisy_reference",
        conditional_on="observed donors",
        generated="spy",
        mechanism="ReferencePowerEvaluator.effect_by_duration",
        sampler_seed=1,
        data_seed_scheme="spy",
        noise_scale=1.0,
        donor_weights=(),
        ar_coefficient=None,
        horizon_reuse=False,
    )
    return EffectByDuration(
        points=(zero, point),
        minimum_detectable_effects=(mde,),
        study=record,
        n_fits=4,
        n_replicates=4,
        failed_runs=0,
        failure_messages=(),
        power_withheld=False,
        withhold_reason=None,
        preperiod_rmse_ratio=0.1,
        decision_rate_not_power=None,
        minimum_worthwhile_effect=WORTHWHILE,
        precision_target=None,
        false_positive_within_tolerance=True,
        decision=_decision(),
        criterion=_criterion(),
        estimator="synthetic_control",
        scale="absolute",
        baseline=None,
        divergences=0,
        max_rhat=1.0,
    )


def _spy_engine(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []

    def _effect_by_duration(self, durations, effects, *, criterion, **kwargs):
        seen.append(criterion.qualification_scope)
        return _study()

    monkeypatch.setattr(
        "causalpy.planning.power.ReferencePowerEvaluator.effect_by_duration",
        _effect_by_duration,
    )
    return seen


def _install_group_adapter(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    fits: list[tuple[str, ...]] = []

    class _Adapter:
        def __init__(self, *, model, sample_kwargs, random_seed) -> None:
            self.sample_kwargs = sample_kwargs

        def capabilities(self) -> Capabilities:
            return Capabilities(
                estimator="augmented_synthetic_control",
                deterministic_effect_reuse=True,
                horizon_reuse=False,
                post_period_donor_data=True,
                group_support=True,
                max_group_size=3,
                allowed_decision_rules=("positive",),
                evidence=(("group_support", "contract_adapter"),),
            )

        def fit_window(self, request):
            fits.append(tuple(request.treated_units))
            return SimpleNamespace(n_inference_fits=1, duration=request.duration)

        def translate(self, fit, scenario, decision, *, fit_index, outcome_scale):
            decided = float(scenario.magnitude) >= WORTHWHILE
            return SimpleNamespace(
                duration=fit.duration,
                scenario=scenario,
                decision=SimpleNamespace(
                    decided=decided,
                    hdi_lower=float(scenario.magnitude) - 0.1,
                    hdi_upper=float(scenario.magnitude) + 0.1,
                    posterior_tail_probability=0.01 if decided else 0.5,
                ),
                summary=SimpleNamespace(
                    cumulative_total_mean=float(scenario.magnitude),
                    average_per_period_mean=float(scenario.magnitude),
                    average_per_treated_unit_mean=float(scenario.magnitude),
                ),
                diagnostics=SimpleNamespace(divergences=0, max_rhat=1.0),
            )

    monkeypatch.setattr(
        "causalpy.planning.reference._registered_adapters",
        lambda: {"augmented_synthetic_control": _Adapter},
    )
    return fits


def _run(**overrides):
    frame = overrides.pop("history", _panel())
    args = {
        "history": frame,
        "units": UNITS,
        "decision": _decision(),
        "criterion": _criterion(),
        "worthwhile_effect": WORTHWHILE,
        "durations": (DURATION,),
        "selection_end": _selection_end(frame),
        "sample_kwargs": LIGHT,
        "random_seed": 11,
        "budget": _budget(),
        "unit_costs": _costs(),
        "min_group_size": 1,
        "max_group_size": 1,
        "shortlist_size": 10,
    }
    args.update(overrides)
    return search(**args)


def test_search_is_keyword_only_for_controls() -> None:
    signature = inspect.signature(search)
    assert (
        signature.parameters["history"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert signature.parameters["selection_end"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "kwargs" not in signature.parameters


def test_bad_inputs_refuse_before_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_engine(monkeypatch)
    frame = _panel()
    with pytest.raises(ValueError, match="No fit was started"):
        _run(history=frame, selection_end=int(frame.index[-1]))
    with pytest.raises(ValueError, match="No fit was started"):
        _run(sample_kwargs={"tune": 20})
    with pytest.raises(TypeError, match="random_seed"):
        _run(random_seed=True)
    with pytest.raises(ValueError, match="scenario_family"):
        _run(scenario_family="not_a_family")
    assert calls == []


def test_zero_budget_skips_without_calling_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_engine(monkeypatch)
    result = _run(budget=_budget(0))
    assert calls == []
    assert result.n_inference_fits == 0
    assert result.frozen is None
    assert result.status == "inconclusive"
    assert set(result.frame["evaluation_status"]) == {"skipped"}
    assert result.optimality == "budget_partial"
    assert result.reason is not None
    assert "fit budget" in result.reason


def test_zero_budget_rejects_a_free_refusal_before_skipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_engine(monkeypatch)
    result = _run(budget=_budget(0), min_group_size=1, max_group_size=2)
    studied = result.frame.loc[
        ~result.frame["evaluation_status"].isin(["not_shortlisted", "filtered"])
    ]
    groups = studied.loc[studied["group_size"] > 1]
    singles = studied.loc[studied["group_size"] == 1]
    assert calls == []
    assert not groups.empty
    assert not singles.empty
    assert set(groups["evaluation_status"]) == {"rejected"}
    assert groups["n_fits"].sum() == 0
    assert groups["reason"].str.contains("one treated unit").all()
    assert set(singles["evaluation_status"]) == {"skipped"}
    assert result.n_inference_fits == 0
    assert result.optimality == "budget_partial"
    assert result.reason is not None
    assert "fit budget" in result.reason


def test_grouped_short_window_is_a_recorded_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fits = _install_group_adapter(monkeypatch)
    result = _run(
        durations=(15,),
        selection_end=0,
        estimator="augmented_synthetic_control",
        min_group_size=2,
        max_group_size=2,
        shortlist_size=2,
        budget=_budget(50),
    )
    rejected = result.frame.loc[result.frame["evaluation_status"] == "rejected"]
    assert fits == []
    assert len(rejected) == 2
    assert rejected["reason"].str.contains("fewer than 2 training rows").all()
    assert rejected["n_fits"].sum() == 0
    assert result.frozen is None
    assert result.n_inference_fits == 0


def test_grouped_constant_donors_are_rejected_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fits = _install_group_adapter(monkeypatch)
    frame = _panel()
    frame[["c", "d"]] = 1.0
    result = _run(
        history=frame,
        estimator="augmented_synthetic_control",
        min_group_size=2,
        max_group_size=2,
        fixed_treated=("a", "b"),
        shortlist_size=1,
        budget=_budget(50),
    )
    rejected = result.frame.loc[result.frame["evaluation_status"] == "rejected"]
    assert fits == []
    assert not rejected.empty
    assert rejected["reason"].str.contains("positive finite variation").all()
    assert result.n_inference_fits == 0
    assert result.frozen is None
    assert set(result.frame["evaluation_status"]) == {"rejected"}
    assert result.optimality == "shortlist"


def test_size_one_routes_through_the_published_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_engine(monkeypatch)
    result = _run()
    assert calls
    assert set(calls) == {"fixed_design"}
    assert result.frozen is not None
    assert result.frozen.treated_units == ("a",)
    assert result.frozen.cost == 1.0
    assert result.status == "frozen"
    assert result.optimality == "shortlist"
    chosen = result.frame.loc[result.frame["selected"]]
    assert len(chosen) == 1
    assert list(result.frame.columns) == list(SEARCH_COLUMNS)
    assert "estimator_class" not in result.frame.columns


def test_vanilla_group_above_one_is_rejected_with_zero_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_engine(monkeypatch)
    result = _run(min_group_size=2, max_group_size=2)
    assert calls == []
    assert result.n_inference_fits == 0
    assert result.frozen is None
    assert set(result.frame["evaluation_status"]) == {"rejected"}
    assert result.frame["n_fits"].sum() == 0
    assert result.frame["reason"].str.contains("one treated unit").all()


def test_grouped_path_uses_the_plan_and_not_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_engine(monkeypatch)
    fits = _install_group_adapter(monkeypatch)
    result = _run(
        estimator="augmented_synthetic_control",
        min_group_size=2,
        max_group_size=2,
    )
    assert calls == []
    assert fits
    assert all(len(treated) == 2 for treated in fits)
    assert result.frozen is not None
    assert result.frozen.treated_units == ("a", "d")
    assert result.frozen.cost == 3.0
    assert result.n_inference_fits == len(fits)
    assert result.frame["unit_scale"].notna().any()


def test_group_sizes_one_to_three_shrink_the_donor_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_engine(monkeypatch)
    _install_group_adapter(monkeypatch)
    result = _run(
        estimator="augmented_synthetic_control",
        min_group_size=1,
        max_group_size=3,
    )
    assert calls == []
    completed = result.frame.loc[result.frame["evaluation_status"] == "completed"]
    assert set(completed["group_size"]) == {1, 2, 3}
    donors = {
        int(size): int(completed.loc[completed["group_size"] == size, "n_donors"].max())
        for size in (1, 2, 3)
    }
    assert donors[1] > donors[2] > donors[3]
    assert result.frozen is not None
    frozen_size = len(result.frozen.treated_units)
    assert len(result.frozen.control_units) == donors[frozen_size]
    assert set(result.frozen.treated_units).isdisjoint(result.frozen.control_units)


def test_validation_reports_precision_with_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spy_engine(monkeypatch)
    state = {"n": 0}

    class _Spec:
        def run(self, panel: pd.DataFrame) -> SimpleNamespace:
            state["n"] += 1
            width = (1.0, 3.0, 5.0, 3.0)[(state["n"] - 1) // 2]
            return SimpleNamespace(
                decision=SimpleNamespace(
                    decided=state["n"] % 2 == 0,
                    hdi_lower=0.0,
                    hdi_upper=width,
                    posterior_tail_probability=0.01,
                )
            )

    monkeypatch.setattr(FrozenDesign, "analysis_handoff", lambda self: _Spec())
    frame = _panel()
    report = validate_selection(
        frame,
        UNITS,
        _decision(),
        _criterion(),
        WORTHWHILE,
        (DURATION,),
        n_replicates=4,
        random_seed=11,
        budget=_budget(),
        sample_kwargs=LIGHT,
        selection_end=_selection_end(frame),
        always_control=("b", "c"),
        unit_costs=_costs(),
    )
    assert report.n_frozen == 4
    assert report.failures == ()
    assert report.false_positive_rate == 0.0
    assert report.power == 1.0
    assert report.mean_interval_width == pytest.approx(3.0)
    assert report.interval_width_monte_carlo_error == pytest.approx(
        float(np.std([1.0, 3.0, 5.0, 3.0], ddof=1) / 2.0)
    )


def test_validate_selection_accepts_a_time_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The outer conversion indexes the column. The inner search must not look it up again."""
    calls = _spy_engine(monkeypatch)

    class _Spec:
        def run(self, panel: pd.DataFrame) -> SimpleNamespace:
            return SimpleNamespace(
                decision=SimpleNamespace(
                    decided=False,
                    hdi_lower=0.0,
                    hdi_upper=1.0,
                    posterior_tail_probability=0.4,
                )
            )

    monkeypatch.setattr(FrozenDesign, "analysis_handoff", lambda self: _Spec())
    indexed = _panel()
    report = validate_selection(
        indexed.reset_index(),
        UNITS,
        _decision(),
        _criterion(),
        WORTHWHILE,
        (DURATION,),
        n_replicates=1,
        random_seed=11,
        budget=_budget(),
        sample_kwargs=LIGHT,
        selection_end=_selection_end(indexed),
        time_column="time",
        always_control=("c", "d"),
        unit_costs=_costs(),
        min_group_size=1,
        max_group_size=1,
        shortlist_size=1,
    )
    assert calls
    assert report.n_replicates == 1
    assert report.n_frozen == 1
    assert report.failures == ()
    assert report.training_max_abs_diff == 0.0


def test_search_indexes_a_time_column_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The column is consumed once. The frozen handoff must not look it up again."""
    calls = _spy_engine(monkeypatch)
    indexed = _panel()
    result = _run(
        history=indexed.reset_index(),
        time_column="time",
        selection_end=_selection_end(indexed),
        max_group_size=1,
        shortlist_size=1,
    )
    assert calls
    assert result.frozen is not None
    assert result.frozen.time_column is None
    spec = result.frozen.analysis_handoff()
    assert spec.time_column is None
    assert list(result.frozen.holdout_index) == list(indexed.index[-DURATION:])


def test_sampler_seed_ignores_loop_order() -> None:
    first = sampler_seed(11, '["a"]', (4, 8))
    second = sampler_seed(11, '["a"]', (4, 8))
    other = sampler_seed(11, '["b"]', (4, 8))
    assert first == second
    assert first != other


def test_inject_touches_only_the_holdout(monkeypatch: pytest.MonkeyPatch) -> None:
    _spy_engine(monkeypatch)
    frame = _panel()
    result = _run(history=frame)
    frozen = result.frozen
    assert frozen is not None
    injected = frozen.inject(frame, WORTHWHILE)
    training = list(frozen.training_index)
    holdout = list(frozen.holdout_index)
    assert np.array_equal(
        frame.loc[training].to_numpy(dtype=float),
        injected.loc[training].to_numpy(dtype=float),
    )
    donor_shift = np.max(
        np.abs(
            injected.loc[holdout, list(frozen.control_units)].to_numpy(dtype=float)
            - frame.loc[holdout, list(frozen.control_units)].to_numpy(dtype=float)
        )
    )
    assert donor_shift == 0.0
    treated_shift = float(
        np.mean(
            injected.loc[holdout, list(frozen.treated_units)].to_numpy(dtype=float)
            - frame.loc[holdout, list(frozen.treated_units)].to_numpy(dtype=float)
        )
    )
    assert treated_shift == pytest.approx(WORTHWHILE)


def test_audit_refuses_an_overlapping_holdout(monkeypatch: pytest.MonkeyPatch) -> None:
    _spy_engine(monkeypatch)
    frame = _panel()
    result = _run(history=frame)
    assert result.frozen is not None

    def _boom(*_args, **_kwargs):
        raise AssertionError("audit started a fit")

    monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _boom)
    with pytest.raises(ValueError, match="overlaps"):
        result.frozen.audit(frame)


def test_planning_package_exports_search() -> None:
    from causalpy import planning

    assert planning.search is search
    assert "search" in planning.__all__
    assert "SearchResult" in planning.__all__


def test_group_size_above_three_refuses_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_engine(monkeypatch)
    with pytest.raises(ValueError, match="1..3"):
        _run(max_group_size=4)
    assert calls == []


def test_unknown_estimator_is_rejected_with_zero_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_engine(monkeypatch)
    result = _run(estimator="not_a_method")
    assert calls == []
    assert result.n_inference_fits == 0
    assert result.frozen is None
    assert result.status == "inconclusive"
    assert "No supported design" in (result.reason or "")


def test_vanilla_group_states_the_redesign(monkeypatch: pytest.MonkeyPatch) -> None:
    _spy_engine(monkeypatch)
    result = _run(min_group_size=2, max_group_size=2)
    assert result.status == "inconclusive"
    assert result.frozen is None
    assert "No supported design" in (result.reason or "")


def test_shortlist_cap_can_miss_a_cheaper_qualified_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _effect_by_duration(self, durations, effects, *, criterion, **kwargs):
        treated = tuple(self._plan._treated_units)
        power = 0.1 if treated == ("a",) else 0.95
        return _study(power=power)

    monkeypatch.setattr(
        "causalpy.planning.power.ReferencePowerEvaluator.effect_by_duration",
        _effect_by_duration,
    )
    frame = _panel()
    compared = compare_screening(
        frame,
        UNITS,
        _decision(),
        _criterion(),
        WORTHWHILE,
        (DURATION,),
        selection_end=_selection_end(frame),
        sample_kwargs=LIGHT,
        random_seed=11,
        budget=_budget(),
        shortlist_size=2,
        unit_costs=_costs(),
    )
    assert compared.lost_ids
    assert compared.shortlist_winner != compared.exhaustive_winner
    assert compared.cost_regret is not None
    assert compared.cost_regret > 0.0
    assert compared.cost_regret_reason is None


def _completed_row(
    candidate_id: str,
    *,
    power: float = 0.9,
    width: float = 1.0,
    cost: float = 1.0,
    share: float = 0.2,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "duration": DURATION,
        "evaluation_status": "completed",
        "power_withheld": False,
        "power": power,
        "interval_width": width,
        "cost": cost,
        "outcome_share": share,
    }


def test_pareto_treats_lower_cost_and_width_as_better() -> None:
    """A dearer equal row does not dominate, and a missing cost is worse."""
    cheap = _completed_row("cheap", cost=1.0)
    dear = _completed_row("dear", cost=5.0)
    missing = _completed_row("missing", cost=float("nan"))
    kept = _pareto([cheap, dear, missing])
    assert ("cheap", DURATION) in kept
    assert ("dear", DURATION) not in kept
    assert ("missing", DURATION) not in kept


def test_scale_baseline_mismatches_are_rejected_before_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A baseline pair the engine would refuse must not reserve or start a fit."""
    calls = _spy_engine(monkeypatch)
    relative = _run(
        estimator="augmented_synthetic_control",
        min_group_size=2,
        max_group_size=2,
        scale="relative",
        baseline=None,
    )
    absolute = _run(
        estimator="augmented_synthetic_control",
        min_group_size=2,
        max_group_size=2,
        scale="absolute",
        baseline=10.0,
    )
    assert calls == []
    for result in (relative, absolute):
        assert result.frozen is None
        assert result.n_inference_fits == 0
        assert set(result.frame["evaluation_status"]) == {"rejected"}
        assert int(result.frame["n_fits"].max()) == 0
    assert "No fit was started" in " ".join(relative.frame["reason"].astype(str))
    assert "does not take a baseline" in " ".join(absolute.frame["reason"].astype(str))


def _grouped_success(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    from causalpy.planning.reference import HistoryOnlyPlan

    calls: list[int] = []

    def evaluate(
        self: object,
        durations: int | Sequence[int],
        *,
        absolute_effects: Sequence[float] | None = None,
        relative_effects: Sequence[float] | None = None,
        relative_baseline: float | None = None,
        relative_floor: float | None = None,
    ) -> SimpleNamespace:
        del self, relative_effects, relative_baseline, relative_floor
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("No fit was started. replicate died")
        effects = tuple(absolute_effects or ())
        horizons = (
            (int(durations),)
            if isinstance(durations, int)
            else tuple(int(item) for item in durations)
        )
        evaluations = [
            SimpleNamespace(
                duration=duration,
                scenario=SimpleNamespace(magnitude=effect),
                summary=SimpleNamespace(cumulative_total_mean=effect),
                decision=SimpleNamespace(
                    decided=effect > 0.0,
                    hdi_lower=0.0,
                    hdi_upper=1.0,
                    posterior_tail_probability=0.01,
                ),
                diagnostics=SimpleNamespace(divergences=0, max_rhat=1.0),
            )
            for duration in horizons
            for effect in (0.0, *effects)
        ]
        return SimpleNamespace(n_inference_fits=len(horizons), evaluations=evaluations)

    monkeypatch.setattr(HistoryOnlyPlan, "evaluate", evaluate)
    return calls


def test_short_replicate_sample_does_not_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One failed replicate must not publish power from the survivor."""
    calls = _grouped_success(monkeypatch)
    result = _run(
        estimator="augmented_synthetic_control",
        min_group_size=2,
        max_group_size=2,
        shortlist_size=1,
    )
    assert len(calls) == 2
    assert result.frozen is None
    assert result.n_inference_fits == 1
    studied = result.frame[result.frame["screen_status"] == "shortlisted"]
    assert set(studied["evaluation_status"]) == {"failed"}
    assert studied["power"].isna().all()
    assert "tolerance sample was not reached" in " ".join(studied["reason"].astype(str))


def test_grouped_misfit_uses_injected_noise_not_unit_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large unit is not misfit merely because its observed scale exceeds 3."""
    from causalpy.planning.power import MISFIT_RMSE_TO_NOISE

    def evaluate(
        self: object,
        durations: int | Sequence[int],
        *,
        absolute_effects: Sequence[float] | None = None,
        relative_effects: Sequence[float] | None = None,
        relative_baseline: float | None = None,
        relative_floor: float | None = None,
    ) -> SimpleNamespace:
        del self, relative_effects, relative_baseline, relative_floor
        effects = tuple(absolute_effects or ())
        horizons = (
            (int(durations),)
            if isinstance(durations, int)
            else tuple(int(item) for item in durations)
        )
        evaluations = [
            SimpleNamespace(
                duration=duration,
                scenario=SimpleNamespace(magnitude=effect),
                summary=SimpleNamespace(cumulative_total_mean=effect),
                decision=SimpleNamespace(
                    decided=effect > 0.0,
                    hdi_lower=0.0,
                    hdi_upper=1.0,
                    posterior_tail_probability=0.01,
                ),
                diagnostics=SimpleNamespace(divergences=0, max_rhat=1.0),
            )
            for duration in horizons
            for effect in (0.0, *effects)
            for _replicate in range(2)
        ]
        return SimpleNamespace(n_inference_fits=len(horizons), evaluations=evaluations)

    from causalpy.planning.reference import HistoryOnlyPlan

    monkeypatch.setattr(HistoryOnlyPlan, "evaluate", evaluate)
    frame = _panel()
    donors = ["b", "c", "d"]
    donor_sd = float(np.std(frame[donors].to_numpy(dtype=float)))
    rng = np.random.default_rng(1)
    frame["large"] = frame[donors].mean(axis=1) + rng.normal(
        0.0, 5.0 * donor_sd, size=len(frame)
    )
    units = ("a", "large", *donors)
    result = _run(
        history=frame,
        units=units,
        estimator="augmented_synthetic_control",
        min_group_size=2,
        max_group_size=2,
        fixed_treated=("a", "large"),
        shortlist_size=1,
        unit_costs=tuple((name, 1.0) for name in units),
    )
    study = next(iter(result.studies.values()))
    assert study.preperiod_rmse_ratio < MISFIT_RMSE_TO_NOISE
    assert not bool(result.frame["power_withheld"].any())
    trend = _run(
        estimator="augmented_synthetic_control",
        min_group_size=2,
        max_group_size=2,
        shortlist_size=1,
        scenario_family="persistent_misfit",
    )
    trend_study = next(iter(trend.studies.values()))
    assert trend_study.preperiod_rmse_ratio > MISFIT_RMSE_TO_NOISE
    assert trend_study.power_withheld
    assert trend.frozen is None
    trend_rows = trend.frame[trend.frame["screen_status"] == "shortlisted"]
    assert trend_rows["power"].isna().all()
    trend_reason = " ".join(trend_rows["reason"].astype(str))
    assert "injected noise sd" in trend_reason
    assert "not labeled as power" in trend_reason
    studied = result.frame[result.frame["screen_status"] == "shortlisted"]
    assert studied["divergences"].eq(0).all()
    assert set(studied["diagnostics_status"]) == {"available"}


def _install_decisive_evaluate(
    monkeypatch: pytest.MonkeyPatch,
    diagnostics: object,
) -> None:
    def evaluate(
        self: object,
        durations: int | Sequence[int],
        *,
        absolute_effects: Sequence[float] | None = None,
        relative_effects: Sequence[float] | None = None,
        relative_baseline: float | None = None,
        relative_floor: float | None = None,
    ) -> SimpleNamespace:
        del self, relative_effects, relative_baseline, relative_floor
        effects = tuple(absolute_effects or ())
        horizons = (
            (int(durations),)
            if isinstance(durations, int)
            else tuple(int(item) for item in durations)
        )
        evaluations = [
            SimpleNamespace(
                duration=duration,
                scenario=SimpleNamespace(magnitude=effect),
                summary=SimpleNamespace(cumulative_total_mean=effect),
                decision=SimpleNamespace(
                    decided=effect > 0.0,
                    hdi_lower=0.0,
                    hdi_upper=1.0,
                    posterior_tail_probability=0.01,
                ),
                diagnostics=diagnostics,
            )
            for duration in horizons
            for effect in (0.0, *effects)
            for _replicate in range(2)
        ]
        return SimpleNamespace(n_inference_fits=len(horizons), evaluations=evaluations)

    from causalpy.planning.reference import HistoryOnlyPlan

    monkeypatch.setattr(HistoryOnlyPlan, "evaluate", evaluate)


def test_quiet_unit_keeps_its_own_noise_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A treated unit quieter than its donors is not floored to their scale."""
    from causalpy.planning.search import _unit_scale

    _install_decisive_evaluate(
        monkeypatch,
        SimpleNamespace(divergences=0, max_rhat=1.0, min_bulk_ess=100.0),
    )
    frame = _panel()
    donors = ["b", "c", "d"]
    donor_sd = float(np.std(frame[donors].to_numpy(dtype=float)))
    labels = list(frame.index)
    cutoff = labels.index(_selection_end(frame))
    rng = np.random.default_rng(2)
    pre = rng.normal(0.0, 1.0, size=cutoff + 1)
    pre = (pre - pre.mean()) / np.std(pre) * 0.45 * donor_sd
    quiet = np.zeros(len(frame))
    quiet[: cutoff + 1] = pre
    frame["quiet"] = quiet
    units = ("quiet", *donors)
    assert _unit_scale(
        frame, "quiet", _selection_end(frame), donor_sd
    ) == pytest.approx(0.45)
    result = _run(
        history=frame,
        units=units,
        estimator="augmented_synthetic_control",
        min_group_size=1,
        max_group_size=1,
        fixed_treated=("quiet",),
        shortlist_size=1,
        unit_costs=tuple((name, 1.0) for name in units),
    )
    studied = result.frame[result.frame["screen_status"] == "shortlisted"]
    assert studied["unit_scale"].iloc[0] == pytest.approx(0.45)
    assert float(studied["unit_scale"].iloc[0]) < 1.0
    assert not bool(studied["power_withheld"].any())
    assert result.frozen is not None


def test_constant_treated_unit_is_a_zero_fit_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A treated unit with no pre-cutoff variation is refused before fitting."""
    calls: list[object] = []

    def evaluate(self: object, *args: object, **kwargs: object) -> SimpleNamespace:
        del self, args, kwargs
        calls.append("evaluate")
        raise AssertionError("no fit was started")

    from causalpy.planning.reference import HistoryOnlyPlan

    monkeypatch.setattr(HistoryOnlyPlan, "evaluate", evaluate)
    frame = _panel()
    frame["flat"] = 1.0
    units = ("flat", "b", "c", "d")
    result = _run(
        history=frame,
        units=units,
        estimator="augmented_synthetic_control",
        min_group_size=1,
        max_group_size=1,
        fixed_treated=("flat",),
        shortlist_size=1,
        unit_costs=tuple((name, 1.0) for name in units),
    )
    assert calls == []
    assert result.frozen is None
    assert result.n_inference_fits == 0
    studied = result.frame[result.frame["screen_status"] == "shortlisted"]
    assert set(studied["evaluation_status"]) == {"rejected"}
    assert "No fit was started" in " ".join(studied["reason"].astype(str))


def test_size_one_constructor_refusal_does_not_consume_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty outcome scale is a zero-fit refusal, not a charged failed fit."""
    calls = _spy_engine(monkeypatch)
    result = _run(
        outcome_scale="",
        budget=ComputationalBudget(2),
        min_group_size=1,
        max_group_size=1,
        shortlist_size=2,
    )
    assert calls == []
    assert result.n_inference_fits == 0
    assert result.frozen is None
    studied = result.frame[result.frame["screen_status"] == "shortlisted"]
    assert len(studied) == 2
    assert set(studied["evaluation_status"]) == {"rejected"}
    assert studied["n_fits"].eq(0).all()
    assert "outcome_scale" in " ".join(studied["reason"].astype(str))


def test_unavailable_diagnostics_stay_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing divergence count stays null; it is not stored as zero."""
    from causalpy.planning.search import _summarize_diagnostics

    mixed = _summarize_diagnostics(
        [
            SimpleNamespace(
                diagnostics=SimpleNamespace(
                    divergences=2,
                    max_rhat=1.01,
                    min_bulk_ess=80.0,
                    diagnostics_status="available",
                    diagnostics_reason=None,
                )
            ),
            SimpleNamespace(
                diagnostics=SimpleNamespace(
                    divergences=None,
                    max_rhat=None,
                    min_bulk_ess=None,
                    diagnostics_status="unavailable",
                    diagnostics_reason="not retained",
                )
            ),
        ]
    )
    assert mixed.divergences is None
    assert mixed.status == "unavailable"
    assert mixed.reason == "not retained"
    assert mixed.max_rhat == pytest.approx(1.01)
    _install_decisive_evaluate(
        monkeypatch,
        SimpleNamespace(
            divergences=None,
            max_rhat=None,
            min_bulk_ess=None,
            diagnostics_status="unavailable",
            diagnostics_reason="not retained",
        ),
    )
    result = _run(
        estimator="augmented_synthetic_control",
        min_group_size=2,
        max_group_size=2,
        shortlist_size=1,
    )
    studied = result.frame[result.frame["screen_status"] == "shortlisted"]
    assert studied["divergences"].isna().all()
    assert set(studied["diagnostics_status"]) == {"unavailable"}
    assert set(studied["diagnostics_reason"]) == {"not retained"}
    assert result.studies == {}


def test_truncated_exhaustive_pass_withholds_cost_regret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frozen exhaustive winner is not decision regret when later rows were skipped."""
    monkeypatch.setattr(
        "causalpy.planning.power.ReferencePowerEvaluator.effect_by_duration",
        lambda self, durations, effects, *, criterion, **kwargs: _study(power=0.95),
    )
    frame = _panel()
    compared = compare_screening(
        frame,
        UNITS,
        _decision(),
        _criterion(),
        WORTHWHILE,
        (DURATION,),
        selection_end=_selection_end(frame),
        sample_kwargs=LIGHT,
        random_seed=11,
        budget=ComputationalBudget(2),
        shortlist_size=1,
        unit_costs=_costs(),
    )
    assert compared.cost_regret is None
    assert compared.cost_regret_reason is not None
    assert "exhaustive" in compared.cost_regret_reason


def test_search_records_wall_time_and_traced_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spy_engine(monkeypatch)
    result = _run()
    assert result.runtime_seconds >= 0.0
    assert result.peak_traced_bytes > 0


def test_validate_selection_injects_only_the_grouped_holdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The outer loop, not a spy of search, injects a frozen group on the holdout."""
    from causalpy.planning.reference import HistoryOnlyPlan

    seen: list[pd.DataFrame] = []

    def evaluate(
        self: object,
        durations: int | Sequence[int],
        *,
        absolute_effects: Sequence[float] | None = None,
        relative_effects: Sequence[float] | None = None,
        relative_baseline: float | None = None,
        relative_floor: float | None = None,
    ) -> SimpleNamespace:
        del self, relative_effects, relative_baseline, relative_floor
        effects = tuple(absolute_effects or ())
        horizons = (
            (int(durations),)
            if isinstance(durations, int)
            else tuple(int(item) for item in durations)
        )
        evaluations = [
            SimpleNamespace(
                duration=duration,
                scenario=SimpleNamespace(magnitude=effect),
                summary=SimpleNamespace(cumulative_total_mean=effect),
                decision=SimpleNamespace(
                    decided=effect > 0.0,
                    hdi_lower=0.0,
                    hdi_upper=1.0,
                    posterior_tail_probability=0.01,
                ),
                diagnostics=SimpleNamespace(divergences=0, max_rhat=1.0),
            )
            for duration in horizons
            for effect in (0.0, *effects)
        ]
        return SimpleNamespace(n_inference_fits=len(horizons), evaluations=evaluations)

    class _Spec:
        def run(self, panel: pd.DataFrame) -> SimpleNamespace:
            seen.append(panel.copy())
            return SimpleNamespace(
                decision=SimpleNamespace(
                    decided=True,
                    hdi_lower=0.0,
                    hdi_upper=1.0,
                    posterior_tail_probability=0.01,
                )
            )

    monkeypatch.setattr(HistoryOnlyPlan, "evaluate", evaluate)
    monkeypatch.setattr(FrozenDesign, "analysis_handoff", lambda self: _Spec())
    frame = _panel()
    report = validate_selection(
        frame,
        UNITS,
        _decision(),
        PowerCriterion(
            target_power=0.8,
            false_positive_tolerance=0.15,
            precision_target=None,
            monte_carlo_error_tolerance=0.5,
            qualification_scope="post_search",
        ),
        WORTHWHILE,
        (DURATION,),
        n_replicates=1,
        random_seed=11,
        budget=_budget(),
        sample_kwargs=LIGHT,
        selection_end=_selection_end(frame),
        always_control=("c", "d"),
        unit_costs=_costs(),
        min_group_size=2,
        max_group_size=2,
        shortlist_size=1,
        estimator="augmented_synthetic_control",
    )
    assert report.n_frozen == 1
    assert report.failures == ()
    assert report.training_max_abs_diff == 0.0
    assert report.holdout_shift is not None
    assert report.holdout_shift > 0.0
    null, injected = seen
    changed = [
        column
        for column in null.columns
        if not np.allclose(
            null[column].to_numpy(dtype=float),
            injected[column].to_numpy(dtype=float),
        )
    ]
    assert changed
    assert set(changed) <= {"a", "b"}


def test_frozen_handoff_does_not_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_engine(monkeypatch)
    result = _run()
    assert result.frozen is not None
    before = len(calls)
    handoff = result.frozen.analysis_handoff()
    assert len(calls) == before
    assert set(handoff.training_index).isdisjoint(handoff.holdout_index)
    assert result.frozen.treated_units == tuple(handoff.treated_units)


def test_plot_search_reads_the_finished_table(monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib.pyplot as plt

    _spy_engine(monkeypatch)
    figure = plot_search(_run())
    assert figure.axes
    assert figure.axes[0].get_xlabel() == "cost"
    plt.close(figure)


def test_fixed_treated_group_uses_only_the_supplied_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spy_engine(monkeypatch)
    result = _run(
        fixed_treated=("b",),
        donor_pools=(("a", "c"),),
        shortlist_size=5,
    )
    assert result.frozen is not None
    assert result.frozen.treated_units == ("b",)
    assert result.frozen.control_units == ("a", "c")
    assert set(result.frame["treated"]) == {("b",)}


def _detectable_panel(n_rows: int = 48) -> pd.DataFrame:
    """Donor-mean treated unit on a positive panel. Levels do not change the fit."""
    index = pd.RangeIndex(n_rows, name="time")
    rng = np.random.default_rng(0)
    factor = np.cumsum(rng.normal(0.0, 1.0, size=n_rows))
    frame = pd.DataFrame(index=index)
    for name in ("austin", "boston", "chicago", "denver"):
        frame[name] = factor + np.cumsum(rng.normal(0.0, 0.3, size=n_rows))
    frame["seattle"] = frame[["austin", "boston", "chicago", "denver"]].mean(axis=1)
    return frame - float(frame.min().min()) + 10.0


def _real_kwargs() -> dict[str, object]:
    return {"draws": 20, "tune": 20, "chains": 1, "progressbar": False}


def _donor_threshold(frame: pd.DataFrame, donors: Sequence[str]) -> DecisionRule:
    donor_sd = float(np.std(frame.loc[:, list(donors)].to_numpy(dtype=float)))
    return DecisionRule(
        direction="positive",
        estimand="cumulative_total",
        threshold=donor_sd,
        probability=0.8,
    )


def _post_search_criterion(tolerance: float) -> PowerCriterion:
    return PowerCriterion(
        target_power=0.8,
        false_positive_tolerance=0.15,
        precision_target=None,
        monte_carlo_error_tolerance=tolerance,
        qualification_scope="post_search",
    )


def test_three_estimators_complete_a_real_search() -> None:
    """CI exercise of the published engines. Not the scientific replicate bar."""
    frame = _detectable_panel()
    units = ("seattle", "austin", "boston", "chicago", "denver")
    costs = tuple((name, float(index + 1)) for index, name in enumerate(units))
    criterion = _post_search_criterion(0.4)
    decision = _donor_threshold(frame, ("austin", "boston", "chicago", "denver"))
    kwargs = _real_kwargs()
    reference = search(
        frame,
        units,
        decision,
        criterion,
        20.0,
        (4,),
        selection_end=frame.index[-5],
        sample_kwargs=kwargs,
        random_seed=11,
        budget=ComputationalBudget(8),
        unit_costs=costs,
        min_group_size=1,
        max_group_size=1,
        shortlist_size=1,
    )
    assert reference.n_inference_fits > 0
    assert reference.frozen is not None
    assert reference.frozen.treated_units == ("seattle",)
    for estimator in ("augmented_synthetic_control", "interactive_fixed_effects"):
        result = search(
            frame,
            units,
            decision,
            criterion,
            20.0,
            (4,),
            selection_end=frame.index[-5],
            sample_kwargs=kwargs,
            random_seed=11,
            estimator=estimator,
            fixed_treated=("austin", "boston"),
            budget=ComputationalBudget(8),
            unit_costs=costs,
        )
        assert result.n_inference_fits > 0
        assert result.frozen is not None
        assert result.frozen.treated_units == ("austin", "boston")
        studied = result.frame.loc[result.frame["evaluation_status"] == "completed"]
        assert studied["power"].notna().all()


@pytest.mark.nightly
def test_nightly_large_unit_need_not_increase_grouped_power() -> None:
    """Same group-total effect. The large unit scales the grouped noise, not the signal.

    The effect is 0.28 donor standard deviations per period: above the small
    unit's cumulative threshold and inside the noise of a unit whose observed
    scale is 2.5 donor standard deviations. Scale alone does not trip the
    pre-period screen: that screen compares reconstruction error with the
    noise actually injected, so a larger unit need not be withheld and need
    not raise power.
    """
    frame = _detectable_panel()
    donors = ["austin", "boston", "chicago", "denver"]
    donor_sd = float(np.std(frame[donors].to_numpy(dtype=float)))
    rng = np.random.default_rng(1)
    frame["large"] = frame[donors].mean(axis=1) + rng.normal(
        0.0, 2.5 * donor_sd, size=len(frame)
    )
    frame = frame - float(frame.min().min()) + 10.0
    units = ("seattle", "large", *donors)
    costs = tuple((name, 1.0) for name in units)
    criterion = _post_search_criterion(0.0625)
    effect = 0.28 * donor_sd

    def _one(treated: tuple[str, ...]) -> float:
        result = search(
            frame,
            units,
            _donor_threshold(frame, donors),
            criterion,
            effect,
            (4,),
            selection_end=frame.index[-5],
            sample_kwargs=_real_kwargs(),
            random_seed=11,
            estimator="augmented_synthetic_control",
            fixed_treated=treated,
            budget=ComputationalBudget(160),
            unit_costs=costs,
        )
        studied = result.frame.loc[result.frame["evaluation_status"] == "completed"]
        assert len(studied) == 1
        assert not bool(studied["power_withheld"].iloc[0])
        assert float(studied["false_positive_rate"].iloc[0]) <= 0.15
        assert float(studied["monte_carlo_error"].iloc[0]) <= 0.0625 + 1e-9
        return float(studied["power"].iloc[0])

    small = _one(("seattle",))
    both = _one(("seattle", "large"))
    assert both < small - 0.15


@pytest.mark.nightly
def test_nightly_selection_validation_uses_the_scientific_bar() -> None:
    """Outer decision rate at the project's 0.0625 replicate count.

    Each inner search uses the CI tolerance so the outer loop can reach 64
    panels. The reported power is that outer rate, not the inner curve.
    """
    frame = _detectable_panel()
    units = ("seattle", "austin", "boston", "chicago", "denver")
    costs = tuple((name, float(index + 1)) for index, name in enumerate(units))
    report = validate_selection(
        frame,
        units,
        _donor_threshold(frame, ("austin", "boston", "chicago", "denver")),
        _post_search_criterion(0.4),
        20.0,
        (4,),
        selection_end=frame.index[-5],
        sample_kwargs=_real_kwargs(),
        random_seed=11,
        n_replicates=64,
        min_group_size=1,
        max_group_size=1,
        shortlist_size=1,
        budget=ComputationalBudget(8),
        unit_costs=costs,
        always_control=("chicago", "denver"),
    )
    assert report.n_replicates == 64
    assert report.n_frozen == 64
    assert report.failures == ()
    assert report.power is not None
    assert report.power >= 0.8
    assert report.power_monte_carlo_error is not None
    assert report.power_monte_carlo_error <= 0.0625 + 1e-9
    assert report.false_positive_rate is not None
    assert report.false_positive_rate <= 0.15
    assert report.false_positive_monte_carlo_error is not None
    assert report.false_positive_monte_carlo_error <= 0.0625 + 1e-9
    assert report.mean_interval_width is not None
    assert report.interval_width_monte_carlo_error is not None
    assert report.training_max_abs_diff == 0.0


@pytest.mark.nightly
def test_nightly_reference_search_returns_a_numeric_power() -> None:
    """Size-one search through the real engine at the 0.0625 bar."""
    frame = _detectable_panel()
    units = ("seattle", "austin", "boston", "chicago", "denver")
    result = search(
        frame,
        units,
        _donor_threshold(frame, ("austin", "boston", "chicago", "denver")),
        _post_search_criterion(0.0625),
        20.0,
        (4,),
        selection_end=frame.index[-5],
        sample_kwargs=_real_kwargs(),
        random_seed=11,
        budget=ComputationalBudget(80),
        unit_costs=tuple((name, float(index + 1)) for index, name in enumerate(units)),
        min_group_size=1,
        max_group_size=1,
        shortlist_size=1,
    )
    assert result.n_inference_fits >= 20
    studied = result.frame.loc[result.frame["evaluation_status"] == "completed"]
    assert not studied.empty
    assert studied["power"].notna().all()
    assert studied["monte_carlo_error"].max() <= 0.0625 + 1e-9


def test_size_one_short_sample_is_failed_and_not_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A study below the declared replicate count is failed, not completed."""

    def _effect_by_duration(self, durations, effects, *, criterion, **kwargs):
        del self, durations, effects, criterion, kwargs
        study = _study()
        return replace(
            study,
            n_fits=1,
            n_replicates=1,
            points=tuple(replace(point, n_replicates=1) for point in study.points),
        )

    monkeypatch.setattr(
        "causalpy.planning.power.ReferencePowerEvaluator.effect_by_duration",
        _effect_by_duration,
    )
    result = _run(shortlist_size=1, budget=_budget(4))
    studied = result.frame.loc[result.frame["evaluation_status"] == "failed"]
    assert result.frozen is None
    assert len(studied) == 1
    assert "Only 1 of 2 replicates completed" in " ".join(studied["reason"].astype(str))
    assert int(studied["n_fits"].sum()) == 1
    assert result.n_inference_fits == 1


@pytest.mark.parametrize(
    "message",
    ["Every replicate failed.", "horizon reuse"],
)
def test_post_fit_reference_failure_charges_the_reservation(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    """A failure after fitting starts consumes the reserved replicate count."""
    calls = {"n": 0}

    def _effect_by_duration(self, durations, effects, *, criterion, **kwargs):
        del self, durations, effects, criterion, kwargs
        calls["n"] += 1
        raise RuntimeError(message)

    monkeypatch.setattr(
        "causalpy.planning.power.ReferencePowerEvaluator.effect_by_duration",
        _effect_by_duration,
    )
    result = _run(shortlist_size=3, budget=_budget(2))
    failed = result.frame.loc[result.frame["evaluation_status"] == "failed"]
    skipped = result.frame.loc[result.frame["evaluation_status"] == "skipped"]
    assert calls["n"] == 1
    assert len(failed) == 1
    assert int(failed["n_fits"].sum()) == 2
    assert result.n_inference_fits == 2
    assert len(skipped) == 2
    assert result.frozen is None


def test_pre_fit_reference_refusal_does_not_consume_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal before any fit is charged zero, so the next candidate runs."""
    calls = {"n": 0}

    def _effect_by_duration(self, durations, effects, *, criterion, **kwargs):
        del self, durations, effects, criterion, kwargs
        calls["n"] += 1
        raise RuntimeError("No fit was started.")

    monkeypatch.setattr(
        "causalpy.planning.power.ReferencePowerEvaluator.effect_by_duration",
        _effect_by_duration,
    )
    result = _run(shortlist_size=2, budget=_budget(2))
    failed = result.frame.loc[result.frame["evaluation_status"] == "failed"]
    assert calls["n"] == 2
    assert result.n_inference_fits == 0
    assert len(failed) == 2
    assert int(failed["n_fits"].sum()) == 0
    assert result.frozen is None


def test_input_refusal_summary_quotes_the_row_not_the_estimator() -> None:
    """A baseline refusal is not reported as an estimator capability refusal."""
    result = _run(
        estimator="augmented_synthetic_control",
        min_group_size=2,
        max_group_size=2,
        scale="relative",
        baseline=None,
    )
    assert result.frozen is None
    assert "baseline" in (result.reason or "")
    assert "estimator refused" not in (result.reason or "")
    assert "The rows do not say why" not in (result.reason or "")


def test_validators_replay_a_constrained_grouped_procedure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting a donor pool must not widen the procedure the validators replay."""
    seen: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def _ratio(panels, treated, donors, horizons, noise_scale, scales):
        del panels, horizons, noise_scale, scales
        seen.append((tuple(treated), tuple(donors)))
        return 0.0

    class _Adapter:
        def __init__(self, *, model, sample_kwargs, random_seed) -> None:
            del model, sample_kwargs, random_seed

        def capabilities(self) -> Capabilities:
            return Capabilities(
                estimator="augmented_synthetic_control",
                deterministic_effect_reuse=True,
                horizon_reuse=False,
                post_period_donor_data=True,
                group_support=True,
                max_group_size=3,
                allowed_decision_rules=("positive",),
                evidence=(("group_support", "contract_adapter"),),
            )

        def fit_window(self, request):
            return SimpleNamespace(n_inference_fits=1, duration=request.duration)

        def translate(self, fit, scenario, decision, *, fit_index, outcome_scale):
            del decision, fit_index, outcome_scale
            return SimpleNamespace(
                duration=fit.duration,
                scenario=scenario,
                decision=SimpleNamespace(
                    decided=False,
                    hdi_lower=0.0,
                    hdi_upper=1.0,
                    posterior_tail_probability=0.4,
                ),
                summary=SimpleNamespace(
                    cumulative_total_mean=0.0,
                    average_per_period_mean=0.0,
                    average_per_treated_unit_mean=0.0,
                ),
                diagnostics=SimpleNamespace(divergences=0, max_rhat=1.0),
            )

    monkeypatch.setattr(
        "causalpy.planning.reference._registered_adapters",
        lambda: {"augmented_synthetic_control": _Adapter},
    )
    monkeypatch.setattr(
        sys.modules["causalpy.planning.search"], "_grouped_rmse_ratio", _ratio
    )
    frame = _panel()
    shared = {
        "history": frame,
        "units": UNITS,
        "decision": _decision(),
        "criterion": _criterion(),
        "worthwhile_effect": WORTHWHILE,
        "durations": (DURATION,),
        "selection_end": _selection_end(frame),
        "sample_kwargs": LIGHT,
        "random_seed": 11,
        "budget": _budget(8),
        "estimator": "augmented_synthetic_control",
        "min_group_size": 2,
        "max_group_size": 2,
        "always_control": ("d",),
        "fixed_treated": ("a", "b"),
        "donor_pools": (("c",),),
        "spillover_exclusions": (),
        "shortlist_size": 2,
    }
    validate_selection(n_replicates=1, **shared)
    compare_screening(**shared)
    assert seen
    assert set(seen) == {(("a", "b"), ("c",))}


def test_audit_refuses_training_on_the_planning_holdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later tail can miss the planning holdout while training still contains it."""
    _spy_engine(monkeypatch)
    frame = _panel()
    result = _run(history=frame)
    assert result.frozen is not None
    extra = frame.iloc[:DURATION].copy()
    extra.index = pd.RangeIndex(
        len(frame), len(frame) + DURATION, name=frame.index.name
    )
    later = pd.concat([frame, extra])

    def _boom(*_args, **_kwargs):
        raise AssertionError("audit started a fit")

    monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _boom)
    with pytest.raises(ValueError, match="Audit training includes"):
        result.frozen.audit(later)


def test_confirmation_failure_nulls_rates_and_charges_started_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second confirmation that never starts is one fit, not a non-detection."""
    _install_decisive_evaluate(
        monkeypatch,
        SimpleNamespace(
            divergences=0,
            max_rhat=1.0,
            min_bulk_ess=80.0,
            diagnostics_status="available",
            diagnostics_reason=None,
        ),
    )
    calls = {"n": 0}

    class _Spec:
        def run(self, panel: pd.DataFrame) -> SimpleNamespace:
            del panel
            return SimpleNamespace(
                decision=SimpleNamespace(
                    decided=False,
                    hdi_lower=0.0,
                    hdi_upper=1.0,
                    posterior_tail_probability=0.4,
                )
            )

    def _handoff(self: FrozenDesign) -> _Spec:
        del self
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("second confirmation was not started")
        return _Spec()

    monkeypatch.setattr(FrozenDesign, "analysis_handoff", _handoff)
    frame = _panel()
    report = validate_selection(
        frame,
        UNITS,
        _decision(),
        _criterion(),
        WORTHWHILE,
        (DURATION,),
        n_replicates=1,
        random_seed=11,
        budget=_budget(),
        sample_kwargs=LIGHT,
        selection_end=_selection_end(frame),
        estimator="augmented_synthetic_control",
        min_group_size=2,
        max_group_size=2,
        shortlist_size=1,
        always_control=("c", "d"),
        unit_costs=_costs(),
    )
    assert report.n_frozen == 1
    assert report.confirmation_fits == 1
    assert report.power is None
    assert report.false_positive_rate is None
    assert report.mean_interval_width is None
    assert report.power_monte_carlo_error is None
    assert report.false_positive_monte_carlo_error is None
    assert report.interval_width_monte_carlo_error is None
    assert report.failures
    assert "second confirmation was not started" in report.failures[0]


def test_size_one_validation_draws_the_reference_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Size-one confirmation uses the reference generator; a group does not."""
    plans: list[tuple[str, ...]] = []
    draws: list[tuple[str, int]] = []
    panels: list[pd.DataFrame] = []

    class _Plan:
        def __init__(self, history, treated, donors, decision, **kwargs) -> None:
            del decision, kwargs
            self.history = history
            self.treated_units = tuple(treated)
            self.donors = tuple(donors)
            plans.append(self.treated_units)

    class _Evaluator:
        def __init__(self, plan, *, history, scenario_family) -> None:
            del history, scenario_family
            self.plan = plan

        def untreated_panel(self, replicate: int) -> pd.DataFrame:
            name = self.plan.treated_units[0]
            draws.append((name, replicate))
            panel = self.plan.history.copy()
            panel[name] = float(replicate) + 3.0
            return panel

    def _fake_search(*args, **kwargs):
        del kwargs
        panels.append(args[0].copy())
        return SimpleNamespace(frozen=None, reason="inconclusive")

    module = sys.modules["causalpy.planning.search"]
    monkeypatch.setattr(module, "HistoryOnlyPlan", _Plan)
    monkeypatch.setattr(module, "ReferencePowerEvaluator", _Evaluator)
    monkeypatch.setattr(module, "search", _fake_search)
    frame = _panel()
    report = validate_selection(
        frame,
        UNITS,
        _decision(),
        _criterion(),
        WORTHWHILE,
        (DURATION,),
        n_replicates=2,
        random_seed=11,
        budget=_budget(),
        sample_kwargs=LIGHT,
        selection_end=_selection_end(frame),
        always_control=("c", "d"),
        unit_costs=_costs(),
    )
    assert report.n_frozen == 0
    assert plans == [("a",), ("b",)]
    assert draws == [("a", 0), ("b", 0), ("a", 1), ("b", 1)]
    assert len(panels) == 2
    for replicate, panel in enumerate(panels):
        assert panel.loc[:, ["c", "d"]].equals(frame.loc[:, ["c", "d"]])
        assert panel["a"].eq(float(replicate) + 3.0).all()
        assert panel["b"].eq(float(replicate) + 3.0).all()

    plans.clear()
    validate_selection(
        frame,
        UNITS,
        _decision(),
        _criterion(),
        WORTHWHILE,
        (DURATION,),
        n_replicates=1,
        random_seed=11,
        budget=_budget(),
        sample_kwargs=LIGHT,
        selection_end=_selection_end(frame),
        estimator="augmented_synthetic_control",
        min_group_size=2,
        max_group_size=2,
        always_control=("c", "d"),
        unit_costs=_costs(),
    )
    assert plans == []
    plans.clear()
    validate_selection(
        frame,
        UNITS,
        _decision(),
        _criterion(),
        WORTHWHILE,
        (DURATION,),
        n_replicates=1,
        random_seed=11,
        budget=_budget(),
        sample_kwargs=LIGHT,
        selection_end=_selection_end(frame),
        estimator="synthetic_control",
        min_group_size=1,
        max_group_size=2,
        always_control=("c", "d"),
        unit_costs=_costs(),
    )
    assert plans == []


def test_withheld_size_one_curve_cannot_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reference curve with power withheld is not a completed selection."""
    withheld = replace(
        _study(),
        power_withheld=True,
        withhold_reason="Pre-period misfit. The decision rate was not labeled as power.",
    )

    def _effect_by_duration(self, durations, effects, *, criterion, **kwargs):
        del self, durations, effects, criterion, kwargs
        return withheld

    monkeypatch.setattr(
        "causalpy.planning.power.ReferencePowerEvaluator.effect_by_duration",
        _effect_by_duration,
    )
    result = _run()
    studied = result.frame[result.frame["screen_status"] == "shortlisted"]
    assert result.frozen is None
    assert set(studied["evaluation_status"]) == {"withheld"}
    assert not bool(studied["eligible"].any())
    assert studied["power"].isna().all()
    assert studied["reason"].str.contains("not labeled as power").all()


def test_grouped_tolerance_above_budget_skips_instead_of_aborting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grouped sample larger than the caller budget returns the screen."""
    from causalpy.planning.reference import HistoryOnlyPlan

    calls: list[int] = []

    def evaluate(self, durations, **kwargs):
        del self, durations, kwargs
        calls.append(1)
        raise AssertionError("the over-budget sample must not fit")

    monkeypatch.setattr(HistoryOnlyPlan, "evaluate", evaluate)
    result = _run(
        criterion=replace(_criterion(), monte_carlo_error_tolerance=0.05),
        budget=_budget(10),
        estimator="augmented_synthetic_control",
        min_group_size=1,
        max_group_size=1,
    )
    shortlisted = result.frame[result.frame["screen_status"] == "shortlisted"]
    assert result.frozen is None
    assert not shortlisted.empty
    assert set(shortlisted["evaluation_status"]) == {"skipped"}
    assert shortlisted["n_fits"].eq(0).all()
    assert shortlisted["reason"].str.contains("caller budget").all()
    assert calls == []


def test_size_one_tolerance_above_fixed_ceiling_still_raises() -> None:
    with pytest.raises(ValueError, match="fixed-design ceiling"):
        _run(criterion=replace(_criterion(), monte_carlo_error_tolerance=0.04))


def test_holdout_shift_keeps_the_largest_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later smaller group does not hide an earlier injection."""
    frame = _panel()
    end = _selection_end(frame)
    labels = list(frame.index)
    cut = labels.index(end)
    seen: list[int] = []

    def replay(*args, **kwargs):
        del args, kwargs
        seen.append(1)
        treated = ("a",) if len(seen) == 1 else ("a", "b", "c")
        frozen = FrozenDesign(
            candidate_id="-".join(treated),
            treated_units=treated,
            control_units=("d",),
            duration=DURATION,
            estimator="augmented_synthetic_control",
            decision=_decision(),
            sample_kwargs=(),
            sampler_seed=1,
            outcome_scale="outcome",
            time_column=None,
            history=frame,
            training_index=tuple(labels[: cut + 1]),
            holdout_index=tuple(labels[cut + 1 :]),
            scale="absolute",
            baseline=None,
            cost=1.0,
            power=0.9,
            false_positive_rate=0.0,
            interval_width=1.0,
            monte_carlo_error=0.05,
            selection_objective="test",
        )
        return SimpleNamespace(frozen=frozen, reason=None)

    monkeypatch.setattr(
        sys.modules["causalpy.planning.search"], "_replay_search", replay
    )
    monkeypatch.setattr(
        FrozenDesign,
        "analysis_handoff",
        lambda self: SimpleNamespace(
            run=lambda panel: SimpleNamespace(
                decision=SimpleNamespace(decided=True, hdi_lower=0.0, hdi_upper=1.0)
            )
        ),
    )
    report = validate_selection(
        frame,
        UNITS,
        _decision(),
        _criterion(),
        WORTHWHILE,
        (DURATION,),
        n_replicates=2,
        random_seed=11,
        budget=_budget(),
        sample_kwargs=LIGHT,
        selection_end=end,
        always_control=("d",),
        unit_costs=_costs(),
        estimator="augmented_synthetic_control",
        min_group_size=1,
        max_group_size=3,
    )
    assert report.n_frozen == 2
    assert report.holdout_shift == pytest.approx(WORTHWHILE)


def test_large_unit_persistent_misfit_still_withholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Published misfit noise is not scaled by a large historical unit."""
    from causalpy.planning.power import MISFIT_RMSE_TO_NOISE

    _install_decisive_evaluate(
        monkeypatch, SimpleNamespace(divergences=0, max_rhat=1.0)
    )
    frame = _panel()
    frame["a"] = frame["a"] * 100
    result = _run(
        history=frame,
        estimator="augmented_synthetic_control",
        min_group_size=1,
        max_group_size=1,
        shortlist_size=1,
        scenario_family="persistent_misfit",
    )
    study = next(iter(result.studies.values()))
    assert study.preperiod_rmse_ratio > MISFIT_RMSE_TO_NOISE
    assert study.power_withheld
    assert result.frozen is None
    studied = result.frame[result.frame["screen_status"] == "shortlisted"]
    assert studied["power"].isna().all()
    assert studied["evaluation_status"].eq("withheld").all()


def test_absent_persistent_misfit_rejects_before_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    from causalpy.planning.reference import HistoryOnlyPlan

    def evaluate(self, durations, **kwargs):
        del self, durations, kwargs
        calls.append(1)
        raise AssertionError("an absent misfit must not fit")

    monkeypatch.setattr(HistoryOnlyPlan, "evaluate", evaluate)

    monkeypatch.setattr(
        sys.modules["causalpy.planning.search"],
        "_grouped_rmse_ratio",
        lambda *args, **kwargs: 0.1,
    )
    result = _run(
        estimator="augmented_synthetic_control",
        min_group_size=1,
        max_group_size=1,
        shortlist_size=1,
        scenario_family="persistent_misfit",
    )
    shortlisted = result.frame[result.frame["screen_status"] == "shortlisted"]
    assert calls == []
    assert result.frozen is None
    assert set(shortlisted["evaluation_status"]) == {"rejected"}
    assert shortlisted["n_fits"].eq(0).all()
    assert shortlisted["reason"].str.contains("not present").all()
