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
"""Repeated-experiment power: translation oracle, refusals, and qualification cells."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from causalpy.planning.contracts import (
    MIN_TRAINING_ROWS,
    CanonicalFit,
    DataBoundary,
    DecisionRule,
    EffectScenario,
    FitDiagnostics,
    PowerCriterion,
    PowerRequest,
    Provenance,
    unit_period_effect,
)
from causalpy.planning.power import (
    FRAME_COLUMNS,
    MDE_CONVENTION,
    MECHANISM,
    MISFIT_RMSE_TO_NOISE,
    NOISE_TO_DONOR_SD,
    SCENARIO_FAMILIES,
    TEMPORAL_AR_COEFFICIENT,
    MisfitNotPower,
    ReferencePowerEvaluator,
    _donor_matrix,
    _draw_from,
    _estimand_mean,
    _median_rmse_ratio,
    _rate,
    _true_estimand,
    _with_zero,
    _wrong_sign_rate,
    interpolate_minimum_detectable_effect,
    minimum_detectable_effect,
    monte_carlo_error,
    preperiod_simplex_rmse,
    replicates_for_tolerance,
)
from causalpy.planning.reference import HistoryOnlyPlan, SyntheticControlAdapter

TREATED = "seattle"
CONTROLS = ("austin", "boston", "chicago", "denver")
LIGHT = {
    "draws": 20,
    "tune": 20,
    "chains": 1,
    "progressbar": False,
    "random_seed": 7,
}
NIGHTLY_TARGET_POWER = 0.8
NIGHTLY_FALSE_POSITIVE_TOLERANCE = 0.15
NIGHTLY_MONTE_CARLO_TOLERANCE = 0.0625
NIGHTLY_DECISION_PROBABILITY = 0.8
NIGHTLY_EFFECT_PER_NOISE = 20.0
CI_MONTE_CARLO_TOLERANCE = 0.4
CI_EFFECT_PER_NOISE = 20.0


def _history(n_rows: int = 16) -> pd.DataFrame:
    index = pd.RangeIndex(n_rows, name="time")
    rng = np.random.default_rng(0)
    factor = np.cumsum(rng.normal(0.0, 1.0, size=n_rows))
    frame = pd.DataFrame(index=index)
    for name in CONTROLS:
        shock = np.cumsum(rng.normal(0.0, 0.3, size=n_rows))
        frame[name] = factor + shock
    frame[TREATED] = frame[list(CONTROLS)].mean(axis=1)
    return frame


def _decision(direction: str = "positive", threshold: float = 0.0) -> DecisionRule:
    return DecisionRule(
        direction=direction,
        estimand="cumulative_total",
        threshold=threshold,
        probability=NIGHTLY_DECISION_PROBABILITY,
    )


def _criterion(**overrides: object) -> PowerCriterion:
    values: dict[str, object] = {
        "target_power": NIGHTLY_TARGET_POWER,
        "false_positive_tolerance": NIGHTLY_FALSE_POSITIVE_TOLERANCE,
        "precision_target": None,
        "monte_carlo_error_tolerance": CI_MONTE_CARLO_TOLERANCE,
        "qualification_scope": "fixed_design",
    }
    values.update(overrides)
    return PowerCriterion(**values)


def _plan(
    history: pd.DataFrame, decision: DecisionRule | None = None
) -> HistoryOnlyPlan:
    return HistoryOnlyPlan(
        history,
        TREATED,
        CONTROLS,
        decision or _decision(),
        estimator="synthetic_control",
        sample_kwargs=LIGHT,
        time_column=None,
        outcome_scale="outcome",
    )


def _evaluator(
    history: pd.DataFrame,
    family: str = "correctly_specified_noisy_reference",
    decision: DecisionRule | None = None,
) -> ReferencePowerEvaluator:
    return ReferencePowerEvaluator(
        _plan(history, decision),
        history=history,
        scenario_family=family,
    )


def _zero_fit(request: object, *, offset: float = 0.0) -> CanonicalFit:
    duration = request.duration
    treated = request.treated_units
    values = np.full((1, 4, duration, len(treated)), offset)
    impact = xr.DataArray(
        values,
        dims=("chain", "draw", "obs_ind", "treated_units"),
        coords={"obs_ind": np.arange(duration), "treated_units": list(treated)},
    )
    return CanonicalFit(
        unit_impact=impact,
        counterfactual=impact.copy(),
        diagnostics=FitDiagnostics(0, 1.01, 100.0, "available", None),
        boundary=DataBoundary(
            training_index=request.history.index[:-duration],
            holdout_index=request.history.index[-duration:],
            treated_units=tuple(treated),
            control_units=tuple(request.control_units),
            post_period_treated_in_fit=False,
            post_period_donors_in_prediction=True,
            horizon_policy="separate-fits",
            rows_outside_window_excluded=True,
        ),
        provenance=Provenance(
            estimator="synthetic_control",
            adapter="SyntheticControlAdapter",
            model_class="WeightedSumFitter",
            random_seed=7,
            sample_kwargs=(("random_seed", 7),),
            horizon_policy="separate-fits",
            translation="deterministic_additive",
            uncertainty="posterior_conditional_mean",
        ),
        n_inference_fits=1,
    )


def _patch_fit(monkeypatch: pytest.MonkeyPatch, *, offset: float = 0.0) -> list[int]:
    calls: list[int] = []

    def fake(self: SyntheticControlAdapter, request: object) -> CanonicalFit:
        calls.append(request.duration)
        return _zero_fit(request, offset=offset)

    monkeypatch.setattr(SyntheticControlAdapter, "fit_window", fake)
    return calls


def test_sample_size_formula_uses_the_worst_case_binomial_error() -> None:
    assert replicates_for_tolerance(0.0625, n_durations=1) == 64
    assert replicates_for_tolerance(0.4, n_durations=1) == 2
    assert replicates_for_tolerance(0.5, n_durations=1) == 1
    assert replicates_for_tolerance(0.25, n_durations=1) == 4
    with pytest.raises(ValueError, match="ceiling"):
        replicates_for_tolerance(0.0625, n_durations=2)


def test_monte_carlo_error_is_the_binomial_standard_error() -> None:
    assert monte_carlo_error(0.5, 4) == pytest.approx(0.25)
    assert monte_carlo_error(1.0, 64) == 0.0
    assert monte_carlo_error(0.0, 64) == 0.0
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        monte_carlo_error(1.1, 4)


def test_interpolation_does_not_extrapolate() -> None:
    attained = interpolate_minimum_detectable_effect(
        [0.0, 4.0, 8.0],
        [0.0, 0.5, 1.0],
        target_power=0.8,
        direction="positive",
    )
    assert attained.effect == pytest.approx(6.4)
    assert attained.status == "attained"
    assert attained.lower_effect == 4.0
    assert attained.upper_effect == 8.0
    below = interpolate_minimum_detectable_effect(
        [4.0, 8.0],
        [0.9, 1.0],
        target_power=0.8,
        direction="positive",
    )
    assert below.status == "attained"
    assert below.effect == 4.0
    assert below.reason is None
    above = interpolate_minimum_detectable_effect(
        [0.0, 4.0],
        [0.1, 0.2],
        target_power=0.8,
        direction="positive",
    )
    assert above.status == "out_of_range"
    assert above.effect is None
    exact = interpolate_minimum_detectable_effect(
        [0.0, 5.0],
        [0.0, 0.8],
        target_power=0.8,
        direction="positive",
    )
    assert exact.effect == 5.0
    assert exact.status == "attained"
    first_grid = interpolate_minimum_detectable_effect(
        [0.0, 4.0, 8.0],
        [0.0, 1.0, 1.0],
        target_power=0.8,
        direction="positive",
    )
    assert first_grid.effect == 4.0
    assert first_grid.status == "attained"
    assert first_grid.lower_effect == 4.0
    assert first_grid.upper_effect == 4.0
    negative_first = interpolate_minimum_detectable_effect(
        [0.0, -4.0, -8.0],
        [0.0, 1.0, 1.0],
        target_power=0.8,
        direction="negative",
    )
    assert negative_first.effect == -4.0
    assert negative_first.lower_effect == -4.0
    assert negative_first.upper_effect == -4.0
    broken_null = interpolate_minimum_detectable_effect(
        [0.0, 4.0],
        [0.9, 1.0],
        target_power=0.8,
        direction="positive",
    )
    assert broken_null.status == "null_control_failure"
    assert broken_null.effect is None


def test_preperiod_screen_passes_a_convex_combination_and_fails_a_trend() -> None:
    donors = np.column_stack([np.linspace(0.0, 1.0, 8), np.linspace(2.0, 3.0, 8)])
    weights = np.array([0.25, 0.75])
    signal = donors @ weights
    reference = preperiod_simplex_rmse(donors, signal + 0.01)
    trend = preperiod_simplex_rmse(donors, np.linspace(0.0, 30.0, 8))
    assert reference < trend
    assert reference < 0.1
    assert trend > 1.0


def test_mechanism_documents_the_decision_rate_and_rejects_the_rope_engine() -> None:
    text = f"{MECHANISM} {MDE_CONVENTION} {ReferencePowerEvaluator.effect_by_duration.__doc__}"
    assert "AnalysisDecision.decided" in MECHANISM
    assert "one-sided" in MECHANISM
    assert "operating_characteristics" in MECHANISM
    assert "not used" in MECHANISM
    assert "does not extrapolate" in MDE_CONVENTION
    assert "Zero" in MDE_CONVENTION
    assert "not a ROPE" in text or "not used" in text


def test_untreated_panels_are_reproducible_and_families_do_not_share_seeds() -> None:
    history = _history()
    reference = _evaluator(history)
    again = _evaluator(history)
    other = _evaluator(history, family="temporal_dependence")
    fixed = _evaluator(history, family="fixed_design_qualification")
    left = reference.untreated_panel(0)
    right = again.untreated_panel(0)
    pd.testing.assert_frame_equal(left, right)
    assert not np.allclose(left[TREATED], reference.untreated_panel(1)[TREATED])
    assert not np.allclose(left[TREATED], other.untreated_panel(0)[TREATED])
    assert not np.allclose(left[TREATED], fixed.untreated_panel(0)[TREATED])
    for name in CONTROLS:
        pd.testing.assert_series_equal(left[name], history[name])
    assert TREATED in left.columns
    assert left[TREATED].name == TREATED
    assert reference.noise_scale == pytest.approx(
        NOISE_TO_DONOR_SD * float(np.std(history[list(CONTROLS)].to_numpy()))
    )
    with pytest.raises(ValueError, match="non-negative"):
        reference.untreated_panel(-1)
    with pytest.raises(ValueError, match="non-negative"):
        reference.untreated_panel(True)  # noqa: FBT003


def test_translation_oracle_matches_the_unit_period_effect_and_fit_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_fit(monkeypatch)
    history = _history()
    evaluator = _evaluator(history)
    durations = (4, 8)
    effects = (0.0, 5.0, 10.0)
    result = evaluator.effect_by_duration(
        durations,
        effects[1:],
        decision=_decision(),
        criterion=_criterion(precision_target=1.0),
        minimum_worthwhile_effect=6.0,
    )
    assert calls == [4, 8, 4, 8]
    assert result.n_fits == 4
    assert result.n_replicates == 2
    assert result.failed_runs == 0
    assert result.power_withheld is False
    assert result.study.scenario_family == "correctly_specified_noisy_reference"
    assert result.study.sampler_seed == LIGHT["random_seed"]
    assert result.study.donor_weights == (0.25, 0.25, 0.25, 0.25)
    assert result.study.horizon_reuse is False
    assert result.study.ar_coefficient is None
    frame = result.frame()
    assert list(frame.columns[: len(FRAME_COLUMNS)]) == list(FRAME_COLUMNS)
    assert set(frame["duration"]) == set(durations)
    assert set(frame["effect"]) == set(effects)
    for duration in durations:
        for effect in effects:
            row = frame[
                (frame["duration"] == duration) & (frame["effect"] == effect)
            ].iloc[0]
            increment = unit_period_effect(
                scale="absolute",
                magnitude=effect,
                duration=duration,
                n_treated=1,
            )
            assert row["probability"] == (0.0 if effect == 0.0 else 1.0)
            assert row["false_positive_rate"] == 0.0
            if effect == 0.0:
                assert np.isnan(row["wrong_sign_rate"])
            else:
                assert row["wrong_sign_rate"] == 0.0
            assert row["coverage"] == 1.0
            assert row["bias"] == pytest.approx(0.0, abs=1e-8)
            assert row["posterior_mean"] == pytest.approx(increment * duration)
            assert row["n_replicates"] == 2
            assert row["failed_runs"] == 0
            assert row["precision_met"] == np.True_
            assert row["minimum_worthwhile_effect"] == 6.0
    assert result.false_positive_within_tolerance is True
    assert bool(frame["false_positive_within_tolerance"].iloc[0]) is True
    for item in result.minimum_detectable_effects:
        assert item.status == "attained"
        assert item.effect == 5.0
        assert item.lower_effect == 5.0
        assert item.upper_effect == 5.0
    assert result.minimum_detectable_effects[0].minimum_worthwhile_effect == 6.0
    worthwhile = minimum_detectable_effect(result, duration=4)
    assert worthwhile.worthwhile_comparison == "mde_at_or_below_worthwhile"
    zero = result.points[0]
    assert zero.effect == 0.0
    assert zero.probability == zero.false_positive_rate
    repeated = evaluator.repeated_experiment_power(
        PowerRequest(
            scenario=EffectScenario("absolute", 10.0),
            decision=_decision(),
            criterion=_criterion(),
            estimator="synthetic_control",
            group_size=1,
            duration=4,
        )
    )
    assert repeated.probability == 1.0
    assert repeated.n_replicates == 2
    assert repeated.qualification_scope == "fixed_design"


def test_relative_oracle_uses_the_baseline_and_a_negative_rule_can_be_wrong_signed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fit(monkeypatch, offset=5.0)
    history = _history()
    evaluator = _evaluator(history)
    result = evaluator.effect_by_duration(
        (4,),
        (0.5,),
        decision=_decision("negative"),
        criterion=_criterion(),
        scale="relative",
        baseline=100.0,
    )
    row = result.frame()
    row = row[row["effect"] == 0.5].iloc[0]
    increment = unit_period_effect(
        scale="relative",
        magnitude=0.5,
        duration=4,
        n_treated=1,
        baseline=100.0,
    )
    assert row["posterior_mean"] == pytest.approx(5.0 * 4 + increment * 4)
    assert row["wrong_sign_rate"] == 0.0
    zero = result.frame()
    zero = zero[zero["effect"] == 0.0].iloc[0]
    assert np.isnan(zero["wrong_sign_rate"])
    assert row["probability"] == 0.0
    large = evaluator.effect_by_duration(
        (4,),
        (-20.0,),
        decision=_decision("negative"),
        criterion=_criterion(),
        scale="absolute",
    )
    large_row = large.frame()
    large_row = large_row[large_row["effect"] == -20.0].iloc[0]
    assert large_row["probability"] == 1.0
    assert large_row["wrong_sign_rate"] == 0.0
    biased = evaluator.effect_by_duration(
        (4,),
        (-1.0,),
        decision=_decision("negative"),
        criterion=_criterion(),
        scale="absolute",
    )
    biased_row = biased.frame()
    biased_row = biased_row[biased_row["effect"] == -1.0].iloc[0]
    assert biased_row["posterior_mean"] > 0.0
    assert biased_row["wrong_sign_rate"] == 1.0


def test_zero_effect_meeting_the_target_is_a_null_control_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fit(monkeypatch, offset=5.0)
    result = _evaluator(_history()).effect_by_duration(
        (4,),
        (10.0,),
        decision=_decision(),
        criterion=_criterion(),
    )
    mde = result.minimum_detectable_effects[0]
    assert mde.status == "null_control_failure"
    assert mde.effect is None
    assert result.point(4, 0.0).probability == 1.0
    assert np.isnan(result.point(4, 0.0).wrong_sign_rate)
    assert result.false_positive_within_tolerance is False


def test_each_replica_receives_the_planned_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from causalpy.pymc_models import WeightedSumFitter

    model = WeightedSumFitter(
        priors={"beta": 3.0},
        sample_kwargs={
            "draws": 20,
            "tune": 20,
            "chains": 1,
            "progressbar": False,
            "random_seed": 13,
        },
    )
    history = _history()
    plan = HistoryOnlyPlan(
        history,
        TREATED,
        CONTROLS,
        _decision(),
        estimator="synthetic_control",
        model=model,
    )
    seen: list[dict[str, object]] = []
    original = HistoryOnlyPlan.__init__

    def record(self: HistoryOnlyPlan, *args: object, **kwargs: object) -> None:
        seen.append(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setattr(HistoryOnlyPlan, "__init__", record)
    _patch_fit(monkeypatch)
    ReferencePowerEvaluator(
        plan,
        history=history,
        scenario_family="correctly_specified_noisy_reference",
    ).effect_by_duration(
        (4,),
        (10.0,),
        decision=_decision(),
        criterion=_criterion(),
    )
    replicas = [item["model"] for item in seen]
    assert len(replicas) == 2
    assert all("sample_kwargs" not in item for item in seen)
    assert all(isinstance(item, WeightedSumFitter) for item in replicas)
    assert all(item is not model for item in replicas)
    assert len({id(item) for item in replicas}) == 2
    assert all(item._user_priors == {"beta": 3.0} for item in replicas)


def test_shifted_donors_are_refused_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_fit(monkeypatch)
    history = _history()
    shifted = history.copy()
    shifted["austin"] = shifted["austin"] + 1.0
    evaluator = ReferencePowerEvaluator(
        _plan(history),
        history=shifted,
        scenario_family="correctly_specified_noisy_reference",
    )
    with pytest.raises(ValueError, match="Donor columns do not match"):
        evaluator.effect_by_duration(
            (4,),
            (10.0,),
            decision=_decision(),
            criterion=_criterion(),
        )
    assert calls == []


def test_failures_are_counted_and_do_not_invent_a_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def flaky(self: SyntheticControlAdapter, request: object) -> CanonicalFit:
        calls.append(request.duration)
        if len(calls) == 1:
            raise RuntimeError("sampler failed")
        return _zero_fit(request)

    monkeypatch.setattr(SyntheticControlAdapter, "fit_window", flaky)
    result = _evaluator(_history()).effect_by_duration(
        (4,),
        (10.0,),
        decision=_decision(),
        criterion=_criterion(monte_carlo_error_tolerance=0.4),
    )
    assert result.failed_runs == 1
    assert result.n_fits == 2
    assert result.points[0].n_replicates == 1
    assert result.failure_messages[0].startswith("RuntimeError")
    assert result.points[0].probability in (0.0, 1.0)

    def always_fail(self: SyntheticControlAdapter, request: object) -> CanonicalFit:
        raise RuntimeError("sampler failed")

    monkeypatch.setattr(SyntheticControlAdapter, "fit_window", always_fail)
    with pytest.raises(RuntimeError, match="Every replicate failed"):
        _evaluator(_history()).effect_by_duration(
            (4,),
            (10.0,),
            decision=_decision(),
            criterion=_criterion(),
        )


def test_refusals_happen_before_any_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_fit(monkeypatch)
    history = _history()
    evaluator = _evaluator(history)
    decision = _decision()
    criterion = _criterion()
    request = PowerRequest(
        scenario=EffectScenario("absolute", 1.0),
        decision=decision,
        criterion=criterion,
        estimator="synthetic_control",
        group_size=1,
        duration=4,
    )
    before = (
        request.estimator,
        request.group_size,
        request.duration,
        request.criterion.qualification_scope,
    )
    with pytest.raises(ValueError, match="group_size"):
        evaluator.repeated_experiment_power(replace(request, group_size=2))
    with pytest.raises(ValueError, match="synthetic_control"):
        evaluator.repeated_experiment_power(
            replace(request, estimator="augmented_synthetic_control")
        )
    with pytest.raises(ValueError, match="Post-search"):
        evaluator.effect_by_duration(
            (4,),
            (1.0,),
            decision=decision,
            criterion=_criterion(qualification_scope="post_search"),
        )
    with pytest.raises(TypeError, match="DecisionRule"):
        evaluator.effect_by_duration(
            (4,),
            (1.0,),
            decision=object(),  # type: ignore[arg-type]
            criterion=criterion,
        )
    with pytest.raises(ValueError, match="baseline"):
        evaluator.effect_by_duration(
            (4,),
            (0.5,),
            decision=decision,
            criterion=criterion,
            scale="relative",
        )
    with pytest.raises(ValueError, match="must not be empty"):
        evaluator.effect_by_duration((4,), (), decision=decision, criterion=criterion)
    with pytest.raises(ValueError, match="ceiling"):
        evaluator.effect_by_duration(
            (4, 8),
            (1.0,),
            decision=decision,
            criterion=_criterion(monte_carlo_error_tolerance=0.01),
        )
    with pytest.raises(ValueError, match="training rows"):
        evaluator.effect_by_duration(
            (len(history) - MIN_TRAINING_ROWS + 1,),
            (1.0,),
            decision=decision,
            criterion=criterion,
        )
    short = history.drop(columns=["denver"])
    with pytest.raises(ValueError, match="missing units"):
        ReferencePowerEvaluator(
            _plan(history),
            history=short,
            scenario_family="correctly_specified_noisy_reference",
        ).effect_by_duration(
            (4,),
            (1.0,),
            decision=decision,
            criterion=criterion,
        )
    seedless = HistoryOnlyPlan(
        history,
        TREATED,
        CONTROLS,
        decision,
        estimator="synthetic_control",
        sample_kwargs={"draws": 20},
        time_column=None,
        outcome_scale="outcome",
    )
    with pytest.raises(ValueError, match="random_seed"):
        ReferencePowerEvaluator(
            seedless,
            history=history,
            scenario_family="correctly_specified_noisy_reference",
        ).effect_by_duration(
            (4,),
            (1.0,),
            decision=decision,
            criterion=criterion,
        )
    with pytest.raises(TypeError, match="HistoryOnlyPlan"):
        ReferencePowerEvaluator(
            object(),  # type: ignore[arg-type]
            history=history,
            scenario_family="correctly_specified_noisy_reference",
        )
    with pytest.raises(ValueError, match="scenario_family"):
        ReferencePowerEvaluator(
            _plan(history), history=history, scenario_family="search"
        )
    assert calls == []
    assert (
        request.estimator,
        request.group_size,
        request.duration,
        request.criterion.qualification_scope,
    ) == before


def test_a_non_reference_estimator_is_rejected_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_fit(monkeypatch)
    history = _history()
    plan = _plan(history)
    original = HistoryOnlyPlan.capabilities.fget

    def other(self: HistoryOnlyPlan):
        return replace(original(self), estimator="augmented_synthetic_control")

    monkeypatch.setattr(HistoryOnlyPlan, "capabilities", property(other))
    with pytest.raises(ValueError, match="synthetic_control"):
        ReferencePowerEvaluator(
            plan, history=history, scenario_family="correctly_specified_noisy_reference"
        ).effect_by_duration(
            (4,),
            (1.0,),
            decision=_decision(),
            criterion=_criterion(),
        )
    assert calls == []


def test_persistent_misfit_withholds_power_and_a_missing_misfit_does_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_fit(monkeypatch)
    history = _history()
    evaluator = _evaluator(history, family="persistent_misfit")
    result = evaluator.effect_by_duration(
        (4,),
        (10.0,),
        decision=_decision(),
        criterion=_criterion(),
    )
    assert result.power_withheld is True
    assert result.preperiod_rmse_ratio > MISFIT_RMSE_TO_NOISE
    assert np.isnan(result.points[0].probability)
    assert result.decision_rate_not_power == pytest.approx(1.0)
    assert "misfit" in (result.withhold_reason or "")
    assert calls
    with pytest.raises(MisfitNotPower, match="not labeled as power") as caught:
        evaluator.repeated_experiment_power(
            PowerRequest(
                scenario=EffectScenario("absolute", 10.0),
                decision=_decision(),
                criterion=_criterion(),
                estimator="synthetic_control",
                group_size=1,
                duration=4,
            )
        )
    assert caught.value.report.power_withheld is True
    assert np.isnan(caught.value.report.points[0].probability)
    calls.clear()
    monkeypatch.setattr(
        "causalpy.planning.power._median_rmse_ratio",
        lambda *_args, **_kwargs: 0.0,
    )
    with pytest.raises(RuntimeError, match="not present"):
        evaluator.effect_by_duration(
            (4,),
            (10.0,),
            decision=_decision(),
            criterion=_criterion(),
        )
    assert calls == []


def test_families_record_distinct_scopes_without_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fit(monkeypatch)
    history = _history()
    records = {}
    for family in SCENARIO_FAMILIES:
        result = _evaluator(history, family=family).effect_by_duration(
            (4,),
            (1.0,),
            decision=_decision(),
            criterion=_criterion(),
        )
        records[family] = result
        assert result.study.scenario_family == family
        assert family in result.study.generated
        assert result.n_fits == 2
    assert (
        records["temporal_dependence"].study.ar_coefficient == TEMPORAL_AR_COEFFICIENT
    )
    assert records["correctly_specified_noisy_reference"].study.ar_coefficient is None
    assert (
        records["fixed_design_qualification"].study.generated
        != records["correctly_specified_noisy_reference"].study.generated
    )
    assert records["persistent_misfit"].power_withheld is True
    equal = tuple(1.0 / len(CONTROLS) for _ in CONTROLS)
    assert records["persistent_misfit"].study.donor_weights == ()
    for name in SCENARIO_FAMILIES:
        if name == "persistent_misfit":
            continue
        assert records[name].study.donor_weights == equal
    assert records["correctly_specified_noisy_reference"].power_withheld is False


def test_missing_points_and_stored_mdes_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fit(monkeypatch)
    result = _evaluator(_history()).effect_by_duration(
        (4,),
        (5.0,),
        decision=_decision(),
        criterion=_criterion(),
    )
    with pytest.raises(ValueError, match="No unique point") as caught:
        result.point(4, 9.0)
    message = str(caught.value)
    assert "(4, 0.0)" in message
    assert "(4, 5.0)" in message
    with pytest.raises(ValueError, match="No minimum detectable effect"):
        minimum_detectable_effect(result, duration=8)


def test_interpolation_rejects_a_bad_grid_before_searching() -> None:
    with pytest.raises(ValueError, match="direction"):
        interpolate_minimum_detectable_effect(
            [0.0, 1.0],
            [0.0, 1.0],
            target_power=0.8,
            direction="two-sided",
        )
    with pytest.raises(ValueError, match="same length"):
        interpolate_minimum_detectable_effect(
            [0.0, 1.0],
            [0.0],
            target_power=0.8,
            direction="positive",
        )
    with pytest.raises(ValueError, match="power must be in"):
        interpolate_minimum_detectable_effect(
            [0.0, 1.0],
            [0.0, 1.1],
            target_power=0.8,
            direction="positive",
        )
    with pytest.raises(ValueError, match="power must be in"):
        interpolate_minimum_detectable_effect(
            [0.0, 1.0],
            [0.0, -0.1],
            target_power=0.8,
            direction="positive",
        )
    missing_side = interpolate_minimum_detectable_effect(
        [-2.0, -1.0],
        [0.0, 1.0],
        target_power=0.8,
        direction="positive",
    )
    assert missing_side.status == "out_of_range"
    assert missing_side.effect is None
    assert missing_side.reason == "no effect on the decision side"


def test_simplex_screen_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match=r"n_rows, n_donors"):
        preperiod_simplex_rmse(np.ones(4), np.ones(4))
    with pytest.raises(ValueError, match=r"n_rows, n_donors"):
        preperiod_simplex_rmse(np.ones((4, 1)), np.ones((4, 1)))
    with pytest.raises(ValueError, match=r"n_rows, n_donors"):
        preperiod_simplex_rmse(np.ones((4, 1)), np.ones(3))
    with pytest.raises(ValueError, match="at least two rows"):
        preperiod_simplex_rmse(np.ones((1, 1)), np.ones(1))
    with pytest.raises(ValueError, match="at least two rows"):
        preperiod_simplex_rmse(np.ones((2, 0)), np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        preperiod_simplex_rmse(
            np.array([[1.0, np.nan], [2.0, 3.0]]),
            np.ones(2),
        )
    with pytest.raises(ValueError, match="finite"):
        preperiod_simplex_rmse(np.ones((2, 1)), np.array([1.0, np.inf]))


def test_bad_inputs_are_refused_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_fit(monkeypatch)
    history = _history()
    evaluator = _evaluator(history)
    decision = _decision()
    criterion = _criterion()
    assert evaluator.scenario_family == "correctly_specified_noisy_reference"
    with pytest.raises(TypeError, match="DataFrame"):
        ReferencePowerEvaluator(
            _plan(history),
            history=history.to_numpy(),  # type: ignore[arg-type]
            scenario_family="correctly_specified_noisy_reference",
        )
    with pytest.raises(TypeError, match="PowerRequest"):
        evaluator.repeated_experiment_power(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PowerCriterion"):
        evaluator.effect_by_duration(
            (4,),
            (1.0,),
            decision=decision,
            criterion=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="scale"):
        evaluator.effect_by_duration(
            (4,),
            (1.0,),
            decision=decision,
            criterion=criterion,
            scale="multiplicative",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="does not take a baseline"):
        evaluator.effect_by_duration(
            (4,),
            (1.0,),
            decision=decision,
            criterion=criterion,
            baseline=10.0,
        )
    for durations in ("4", b"4", 4):
        with pytest.raises(TypeError, match="durations"):
            evaluator.effect_by_duration(
                durations,  # type: ignore[arg-type]
                (1.0,),
                decision=decision,
                criterion=criterion,
            )
    with pytest.raises(ValueError, match="must not be empty"):
        evaluator.effect_by_duration(
            (),
            (1.0,),
            decision=decision,
            criterion=criterion,
        )
    with pytest.raises(ValueError, match="unique"):
        evaluator.effect_by_duration(
            (4, 4),
            (1.0,),
            decision=decision,
            criterion=criterion,
        )
    for effects in ("1.0", b"1", 1.0):
        with pytest.raises(TypeError, match="effects"):
            evaluator.effect_by_duration(
                (4,),
                effects,  # type: ignore[arg-type]
                decision=decision,
                criterion=criterion,
            )
    assert calls == []


def test_a_different_history_is_a_different_study(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_fit(monkeypatch)
    history = _history()
    plan = _plan(history)
    decision = _decision()
    criterion = _criterion()

    def run(frame: pd.DataFrame) -> None:
        ReferencePowerEvaluator(
            plan,
            history=frame,
            scenario_family="correctly_specified_noisy_reference",
        ).effect_by_duration(
            (4,),
            (1.0,),
            decision=decision,
            criterion=criterion,
        )

    shifted = history.copy()
    shifted.index = shifted.index + 1
    with pytest.raises(ValueError, match="same order"):
        run(shifted)
    with pytest.raises(ValueError, match="missing units"):
        run(history.drop(columns=[TREATED]))
    text = history.copy()
    text["austin"] = "not-a-number"
    with pytest.raises(ValueError, match="finite numbers"):
        run(text)
    missing = history.copy()
    missing["austin"] = np.nan
    with pytest.raises(ValueError, match="finite numbers"):
        run(missing)
    flat = history.copy()
    for name in CONTROLS:
        flat[name] = 1.0
    with pytest.raises(ValueError, match="positive variation"):
        run(flat)
    assert calls == []


def test_time_column_labels_must_be_the_planned_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fit(monkeypatch)
    frame = _history().rename_axis("time").reset_index()
    plan = HistoryOnlyPlan(
        frame,
        TREATED,
        CONTROLS,
        _decision(),
        estimator="synthetic_control",
        sample_kwargs=LIGHT,
        time_column="time",
        outcome_scale="outcome",
    )
    result = ReferencePowerEvaluator(
        plan,
        history=frame,
        scenario_family="correctly_specified_noisy_reference",
    ).effect_by_duration(
        (4,),
        (5.0,),
        decision=_decision(),
        criterion=_criterion(),
    )
    assert result.point(4, 5.0).probability == 1.0
    assert result.study.sampler_seed == LIGHT["random_seed"]
    with pytest.raises(ValueError, match="time column"):
        ReferencePowerEvaluator(
            plan,
            history=frame.drop(columns=["time"]),
            scenario_family="correctly_specified_noisy_reference",
        ).effect_by_duration(
            (4,),
            (5.0,),
            decision=_decision(),
            criterion=_criterion(),
        )


@pytest.mark.parametrize(
    ("estimand", "expected"),
    [
        ("average_per_period", 5.0),
        ("average_per_treated_unit", 20.0),
    ],
)
def test_translated_power_uses_the_requested_estimand(
    monkeypatch: pytest.MonkeyPatch,
    estimand: str,
    expected: float,
) -> None:
    _patch_fit(monkeypatch)
    decision = replace(_decision(), estimand=estimand)
    result = _evaluator(_history(), decision=decision).effect_by_duration(
        (4,),
        (5.0,),
        decision=decision,
        criterion=_criterion(),
    )
    row = result.point(4, 5.0)
    assert row.posterior_mean == pytest.approx(expected)
    assert row.bias == pytest.approx(0.0, abs=1e-8)
    assert row.probability == 1.0


def test_an_mde_above_the_worthwhile_effect_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fit(monkeypatch)
    result = _evaluator(_history()).effect_by_duration(
        (4,),
        (5.0,),
        decision=_decision(),
        criterion=_criterion(),
        minimum_worthwhile_effect=1.0,
    )
    stored = minimum_detectable_effect(result, duration=4)
    assert stored.effect == 5.0
    assert stored.worthwhile_comparison == "mde_larger_than_worthwhile"


def test_the_closed_over_plan_must_stay_on_the_reference_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_fit(monkeypatch)
    history = _history()
    plan = _plan(history)
    decision = _decision()
    criterion = _criterion()
    original_handoff = HistoryOnlyPlan.analysis_handoff

    def grouped(self: HistoryOnlyPlan, *, duration: int) -> object:
        return replace(
            original_handoff(self, duration=duration),
            treated_units=(TREATED, "austin"),
        )

    monkeypatch.setattr(HistoryOnlyPlan, "analysis_handoff", grouped)
    with pytest.raises(ValueError, match="one treated unit"):
        ReferencePowerEvaluator(
            plan,
            history=history,
            scenario_family="correctly_specified_noisy_reference",
        ).effect_by_duration(
            (4,),
            (1.0,),
            decision=decision,
            criterion=criterion,
        )

    def missing_model(self: HistoryOnlyPlan, *, duration: int) -> object:
        return replace(original_handoff(self, duration=duration), model=None)

    monkeypatch.setattr(HistoryOnlyPlan, "analysis_handoff", missing_model)
    with pytest.raises(RuntimeError, match="did not hand off its model"):
        ReferencePowerEvaluator(
            plan,
            history=history,
            scenario_family="correctly_specified_noisy_reference",
        ).effect_by_duration(
            (4,),
            (1.0,),
            decision=decision,
            criterion=criterion,
        )

    monkeypatch.setattr(HistoryOnlyPlan, "analysis_handoff", original_handoff)
    original_capabilities = HistoryOnlyPlan.capabilities.fget

    def no_reuse(self: HistoryOnlyPlan) -> object:
        return replace(original_capabilities(self), deterministic_effect_reuse=False)

    monkeypatch.setattr(HistoryOnlyPlan, "capabilities", property(no_reuse))
    with pytest.raises(ValueError, match="deterministic effect reuse"):
        ReferencePowerEvaluator(
            plan,
            history=history,
            scenario_family="correctly_specified_noisy_reference",
        ).effect_by_duration(
            (4,),
            (1.0,),
            decision=decision,
            criterion=criterion,
        )

    def reused_horizon(self: HistoryOnlyPlan) -> object:
        return replace(original_capabilities(self), horizon_reuse=True)

    monkeypatch.setattr(HistoryOnlyPlan, "capabilities", property(reused_horizon))
    with pytest.raises(ValueError, match="Horizon reuse"):
        ReferencePowerEvaluator(
            plan,
            history=history,
            scenario_family="correctly_specified_noisy_reference",
        ).effect_by_duration(
            (4,),
            (1.0,),
            decision=decision,
            criterion=criterion,
        )
    assert calls == []


def test_translation_must_return_each_requested_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _evaluator(_history())
    decision = _decision()
    criterion = _criterion()

    def install(result: object) -> None:
        monkeypatch.setattr(
            HistoryOnlyPlan, "evaluate", lambda *_args, **_kwargs: result
        )

    install(
        SimpleNamespace(
            capabilities=SimpleNamespace(horizon_reuse=True),
            evaluations=(),
            n_inference_fits=0,
        )
    )
    with pytest.raises(RuntimeError, match="Horizon reuse is off"):
        evaluator.effect_by_duration(
            (4,),
            (1.0,),
            decision=decision,
            criterion=criterion,
        )

    install(
        SimpleNamespace(
            capabilities=SimpleNamespace(horizon_reuse=False),
            evaluations=(),
            n_inference_fits=0,
        )
    )
    with pytest.raises(RuntimeError, match="own fit"):
        evaluator.effect_by_duration(
            (4,),
            (1.0,),
            decision=decision,
            criterion=criterion,
        )

    install(
        SimpleNamespace(
            capabilities=SimpleNamespace(horizon_reuse=False),
            evaluations=(
                SimpleNamespace(fit_index=0),
                SimpleNamespace(fit_index=1),
            ),
            n_inference_fits=1,
        )
    )
    with pytest.raises(RuntimeError, match="own fit"):
        evaluator.effect_by_duration(
            (4,),
            (1.0,),
            decision=decision,
            criterion=criterion,
        )

    install(
        SimpleNamespace(
            capabilities=SimpleNamespace(horizon_reuse=False),
            evaluations=(
                SimpleNamespace(
                    fit_index=0,
                    duration=4,
                    scenario=SimpleNamespace(magnitude=99.0),
                ),
            ),
            n_inference_fits=1,
        )
    )
    with pytest.raises(RuntimeError, match="not requested"):
        evaluator.effect_by_duration(
            (4,),
            (1.0,),
            decision=decision,
            criterion=criterion,
        )

    evaluation = SimpleNamespace(
        fit_index=0,
        duration=4,
        scenario=SimpleNamespace(magnitude=0.0),
        summary=SimpleNamespace(
            cumulative_total_mean=0.0,
            average_per_period_mean=0.0,
            average_per_treated_unit_mean=0.0,
        ),
        decision=SimpleNamespace(
            decided=False,
            hdi_lower=-1.0,
            hdi_upper=1.0,
            posterior_tail_probability=0.1,
        ),
        diagnostics=SimpleNamespace(max_rhat=1.0, divergences=0),
    )
    install(
        SimpleNamespace(
            capabilities=SimpleNamespace(horizon_reuse=False),
            evaluations=(evaluation,),
            n_inference_fits=1,
        )
    )
    with pytest.raises(RuntimeError, match="every translated effect"):
        evaluator.effect_by_duration(
            (4,),
            (1.0,),
            decision=decision,
            criterion=criterion,
        )


def test_rate_and_screen_guards_refuse_undefined_inputs() -> None:
    assert _with_zero((0.0, 1.0, 1.0)) == (0.0, 1.0)
    with pytest.raises(ValueError, match="at least one"):
        _wrong_sign_rate(())
    with pytest.raises(RuntimeError, match="mix of zero and nonzero"):
        _wrong_sign_rate((None, False))
    with pytest.raises(ValueError, match="at least one"):
        _rate(())
    summary = SimpleNamespace(
        cumulative_total_mean=1.0,
        average_per_period_mean=2.0,
        average_per_treated_unit_mean=3.0,
    )
    assert _estimand_mean(summary, "average_per_period") == 2.0
    assert _estimand_mean(summary, "average_per_treated_unit") == 3.0
    with pytest.raises(ValueError, match="Unsupported estimand"):
        _estimand_mean(summary, "rope")
    with pytest.raises(ValueError, match="Unsupported estimand"):
        _true_estimand(
            scale="absolute",
            magnitude=1.0,
            duration=4,
            baseline=None,
            estimand="rope",
        )
    with pytest.raises(RuntimeError, match="non-finite"):
        _draw_from(
            SimpleNamespace(
                summary=SimpleNamespace(cumulative_total_mean=float("nan"))
            ),
            _decision(),
            "absolute",
            None,
        )
    with pytest.raises(ValueError, match="training rows"):
        _median_rmse_ratio([_history().iloc[:2]], None, (2,), 1.0)

    class _RaisingSlice:
        def to_numpy(self, dtype: object = None, **kwargs: object) -> np.ndarray:
            raise TypeError("not numeric")

    class _RaisingLoc:
        def __getitem__(self, key: object) -> _RaisingSlice:
            return _RaisingSlice()

    class _RaisingHistory(pd.DataFrame):
        @property
        def loc(self) -> _RaisingLoc:  # type: ignore[override]
            return _RaisingLoc()

    with pytest.raises(ValueError, match="finite numbers"):
        _donor_matrix(
            _RaisingHistory({name: [1.0, 2.0] for name in CONTROLS}),
            CONTROLS,
        )

    class _OneDSlice:
        def to_numpy(self, dtype: object = None, **kwargs: object) -> np.ndarray:
            return np.ones(3)

    class _OneDLoc:
        def __getitem__(self, key: object) -> _OneDSlice:
            return _OneDSlice()

    class _OneDHistory(pd.DataFrame):
        @property
        def loc(self) -> _OneDLoc:  # type: ignore[override]
            return _OneDLoc()

    with pytest.raises(ValueError, match="finite numbers"):
        _donor_matrix(
            _OneDHistory({name: [1.0, 2.0, 3.0] for name in CONTROLS}),
            CONTROLS,
        )


@pytest.mark.nightly
@pytest.mark.parametrize(
    "family",
    [
        "correctly_specified_noisy_reference",
        "temporal_dependence",
        "fixed_design_qualification",
    ],
)
def test_nightly_cells_detect_a_large_effect_and_hold_the_null(family: str) -> None:
    history = _history()
    donor_sd = float(np.std(history.loc[:, list(CONTROLS)].to_numpy(dtype=float)))
    effect = NIGHTLY_EFFECT_PER_NOISE * NOISE_TO_DONOR_SD * donor_sd
    # Cumulative truth is four donor standard deviations. The threshold is one
    # donor standard deviation, above this panel's null estimation error.
    result = _evaluator(history, family=family).effect_by_duration(
        (4,),
        (effect,),
        decision=_decision(threshold=donor_sd),
        criterion=_criterion(
            target_power=NIGHTLY_TARGET_POWER,
            false_positive_tolerance=NIGHTLY_FALSE_POSITIVE_TOLERANCE,
            monte_carlo_error_tolerance=NIGHTLY_MONTE_CARLO_TOLERANCE,
        ),
    )
    frame = result.frame()
    zero = frame[frame["effect"] == 0.0].iloc[0]
    detected = frame[frame["effect"] == effect].iloc[0]
    assert result.n_fits == replicates_for_tolerance(
        NIGHTLY_MONTE_CARLO_TOLERANCE, n_durations=1
    )
    assert result.n_fits <= 64
    assert result.failed_runs == 0
    assert result.power_withheld is False
    assert zero["false_positive_rate"] <= NIGHTLY_FALSE_POSITIVE_TOLERANCE
    assert detected["probability"] >= NIGHTLY_TARGET_POWER
    assert detected["monte_carlo_error"] <= NIGHTLY_MONTE_CARLO_TOLERANCE
    assert result.study.scenario_family == family


@pytest.mark.nightly
def test_nightly_persistent_misfit_is_not_labeled_as_power() -> None:
    history = _history()
    evaluator = _evaluator(history, family="persistent_misfit")
    with pytest.raises(MisfitNotPower, match="not labeled as power") as caught:
        evaluator.repeated_experiment_power(
            PowerRequest(
                scenario=EffectScenario("absolute", 10.0),
                decision=_decision(),
                criterion=_criterion(
                    monte_carlo_error_tolerance=NIGHTLY_MONTE_CARLO_TOLERANCE,
                ),
                estimator="synthetic_control",
                group_size=1,
                duration=4,
            )
        )
    report = caught.value.report
    assert report.power_withheld is True
    assert report.n_fits <= 64
    assert np.isnan(report.points[0].probability)
    assert report.study.scenario_family == "persistent_misfit"
