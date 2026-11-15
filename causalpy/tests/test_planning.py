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
"""Public planning contract: size-1 reference path, refusals, and seams."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.constants import HDI_PROB
from causalpy.planning import (
    FIXED_DESIGN_FIT_CEILING,
    OWNERSHIP,
    REFERENCE_CAPABILITIES,
    VALIDATION_MATRIX,
    AnalysisHandoff,
    DecisionRule,
    EffectScenario,
    HistoryOnlyPlan,
    PlanningResult,
    aggregate_unit_impacts,
    decide,
    evaluation_table,
    plot_evaluation,
    require_relative_baseline,
    unit_period_effect,
)
from causalpy.planning.contracts import REJECTED_ESTIMATOR_MAPPINGS
from causalpy.pymc_models import SoftmaxWeightedSumFitter, WeightedSumFitter

LIGHT = {
    "draws": 20,
    "tune": 20,
    "chains": 1,
    "cores": 1,
    "progressbar": False,
    "random_seed": 42,
}
TREATED = "actual"
CONTROLS = ["a", "b", "c", "d", "e", "f", "g"]
DURATION = 8


def _history() -> pd.DataFrame:
    return cp.load_data("sc").iloc[:48][CONTROLS + [TREATED]]


def _decision() -> DecisionRule:
    return DecisionRule("positive", "cumulative_total", 0.0, 0.95)


def _plan(**kwargs: object) -> HistoryOnlyPlan:
    args: dict[str, Any] = {
        "model": None,
        "sample_kwargs": LIGHT,
        "outcome_scale": "outcome",
    }
    args.update(kwargs)
    return HistoryOnlyPlan(
        _history(),
        TREATED,
        CONTROLS,
        _decision(),
        estimator="synthetic_control",
        sample_kwargs=args["sample_kwargs"],
        outcome_scale=args["outcome_scale"],
        model=args["model"],
    )


def _forbid_fit(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def _boom(*_args: object, **_kwargs: object) -> None:
        calls.append("fit")
        raise AssertionError("SyntheticControl was called")

    monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _boom)
    return calls


def _row(result: PlanningResult, scale: str, magnitude: float):
    matches = [
        row
        for row in result.evaluations
        if row.scenario.scale == scale and row.scenario.magnitude == magnitude
    ]
    assert len(matches) == 1
    return matches[0]


def _geo_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=36, freq="W")
    factor = rng.normal(0, 1, size=len(index)).cumsum()
    frame = pd.DataFrame(index=index)
    for shift, name in enumerate(("austin", "boston", "chicago", "denver", "seattle")):
        frame[name] = 100 + 0.4 * factor + rng.normal(0, 1, size=len(index)) + shift
    return frame


def test_construction_and_handoff_do_not_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_fit(monkeypatch)
    plan = _plan()
    handoff = plan.analysis_handoff(duration=DURATION)
    assert calls == []
    assert isinstance(handoff, AnalysisHandoff)
    assert handoff.treatment_time == _history().index[-DURATION]
    assert plan.capabilities.group_support is False
    assert plan.capabilities.deterministic_effect_reuse is True
    assert plan.capabilities.horizon_reuse is False
    assert plan.capabilities.post_period_donor_data is True


def test_general_journey_valid_example() -> None:
    """Domain-neutral size-1 plan reduces to today's synthetic control."""
    history = _history()
    plan = _plan()
    result = plan.evaluate(
        DURATION,
        absolute_effects=[0.0, 1.0],
        relative_effects=[0.1],
        relative_baseline=100.0,
    )
    assert result.n_inference_fits == 1
    assert result.repeated_experiment_power is None
    assert result.historical_detection_share is None
    assert result.power_status == "not_computed"
    assert {row.fit_index for row in result.evaluations} == {0}
    zero = _row(result, "absolute", 0.0)
    shifted = _row(result, "absolute", 1.0)
    relative = _row(result, "relative", 0.1)
    np.testing.assert_array_equal(
        zero.draws.counterfactual.values,
        shifted.draws.counterfactual.values,
    )
    np.testing.assert_allclose(
        shifted.draws.unit_impact.values,
        zero.draws.unit_impact.values + 1.0,
    )
    np.testing.assert_allclose(
        relative.draws.unit_impact.values,
        zero.draws.unit_impact.values + 1.25,
    )
    np.testing.assert_allclose(
        relative.draws.cumulative_total.values,
        zero.draws.cumulative_total.values + 10.0,
    )
    np.testing.assert_allclose(
        relative.draws.relative_total.values,
        relative.draws.cumulative_total.values / 100.0,
    )
    assert relative.summary.relative_baseline == 100.0
    assert relative.summary.relative_baseline_convention is not None
    np.testing.assert_allclose(
        zero.draws.average_per_period.values,
        zero.draws.cumulative_total.values / DURATION,
    )
    np.testing.assert_allclose(
        zero.draws.average_per_treated_unit.values,
        zero.draws.cumulative_total.values,
    )
    assert zero.boundary.post_period_treated_in_fit is False
    assert zero.boundary.post_period_donors_in_prediction is True
    assert zero.provenance.translation == "deterministic_additive"
    assert zero.provenance.uncertainty == "posterior_conditional_mean"
    assert zero.draws.uncertainty == "posterior_conditional_mean"
    assert tuple(zero.draws.unit_impact.dims) == (
        "chain",
        "draw",
        "obs_ind",
        "treated_units",
    )
    direct = cp.SyntheticControl(
        history,
        history.index[-DURATION],
        control_units=CONTROLS,
        treated_units=[TREATED],
        model=WeightedSumFitter(sample_kwargs=LIGHT),
    ).fit()
    expected = direct.result.impact_post
    expected_total = direct.result.impact_post_cumulative.isel(obs_ind=-1)
    if "treated_units" in expected.dims:
        expected = expected.sel(treated_units=TREATED)
        expected_total = expected_total.sel(treated_units=TREATED)
    np.testing.assert_allclose(
        zero.draws.unit_impact.sel(treated_units=TREATED).values,
        expected.values,
    )
    np.testing.assert_allclose(
        zero.draws.cumulative_total.values, expected_total.values
    )
    observed = plan.analysis_handoff(duration=DURATION).run(history)
    np.testing.assert_allclose(
        observed.draws.cumulative_total.values,
        zero.draws.cumulative_total.values,
    )
    assert observed.provenance.translation == "observed"
    table = evaluation_table(result)
    assert table["repeated_experiment_power"].isna().all()
    assert table["posterior_tail_probability"].notna().all()
    assert "power" not in table.columns
    figure = plot_evaluation(zero)
    assert figure.axes
    figure.clear()
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_general_journey_refuses_signed_relative_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_fit(monkeypatch)
    plan = _plan()
    with pytest.raises(ValueError, match="No fit was started"):
        plan.evaluate(DURATION, relative_effects=[0.1], relative_baseline=-1.0)
    with pytest.raises(ValueError, match="No fit was started"):
        plan.evaluate(DURATION, relative_effects=[0.1], relative_baseline=1e-12)
    with pytest.raises(ValueError, match="relative_baseline"):
        plan.evaluate(DURATION, relative_effects=[0.1])
    assert calls == []


def test_geo_journey_valid_example() -> None:
    """One media geo, an absolute sales effect, and a reduction to today's fit."""
    panel = _geo_panel()
    treated = "austin"
    controls = ["boston", "chicago", "denver", "seattle"]
    plan = HistoryOnlyPlan(
        panel,
        treated,
        controls,
        _decision(),
        sample_kwargs=LIGHT,
        outcome_scale="sales",
    )
    result = plan.evaluate(6, absolute_effects=[5.0])
    evaluation = result.evaluations[0]
    assert result.n_inference_fits == 1
    assert evaluation.outcome_scale == "sales"
    assert evaluation.summary.outcome_scale == "sales"
    assert evaluation.decision.posterior_tail_probability == pytest.approx(
        float(np.mean(evaluation.draws.cumulative_total.values > 0.0))
    )
    assert result.repeated_experiment_power is None
    direct = cp.SyntheticControl(
        panel,
        panel.index[-6],
        control_units=controls,
        treated_units=[treated],
        model=WeightedSumFitter(sample_kwargs=LIGHT),
    ).fit()
    expected = direct.result.impact_post_cumulative.isel(obs_ind=-1)
    if "treated_units" in expected.dims:
        expected = expected.sel(treated_units=treated)
    np.testing.assert_allclose(
        evaluation.draws.cumulative_total.values, expected.values + 5.0 * 6
    )
    handoff = plan.analysis_handoff(duration=6)
    assert handoff.treated_units == (treated,)
    assert handoff.estimator == "synthetic_control"


def test_geo_journey_refuses_pair_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_fit(monkeypatch)
    with pytest.raises(ValueError, match="#1440"):
        HistoryOnlyPlan(
            _geo_panel(),
            ["austin", "boston"],
            ["chicago", "denver", "seattle"],
            _decision(),
            sample_kwargs=LIGHT,
            outcome_scale="sales",
        )
    assert calls == []


def test_horizons_are_separate_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    original = cp.SyntheticControl

    class _Counting(original):  # type: ignore[misc, valid-type]
        def fit(self, *args: object, **kwargs: object):
            calls.append(self.treatment_time)
            return original.fit(self, *args, **kwargs)

    monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _Counting)
    result = _plan().evaluate((4, 8), absolute_effects=[0.0])
    assert result.n_inference_fits == 2
    assert len(calls) == 2
    assert calls[0] != calls[1]
    short, long = result.evaluations
    assert short.duration == 4
    assert long.duration == 8
    assert short.fit_index == 0
    assert long.fit_index == 1
    assert short.boundary.training_index.size == len(_history()) - 4
    assert long.boundary.training_index.size == len(_history()) - 8


def test_post_period_donor_boundary() -> None:
    history = _history()
    baseline = _plan().evaluate(DURATION, absolute_effects=[0.0]).evaluations[0]
    treated = history.copy()
    treated.iloc[-DURATION:, treated.columns.get_loc(TREATED)] = (
        treated.iloc[-DURATION:, treated.columns.get_loc(TREATED)] + 25
    )
    treated_fit = (
        HistoryOnlyPlan(
            treated,
            TREATED,
            CONTROLS,
            _decision(),
            sample_kwargs=LIGHT,
        )
        .evaluate(DURATION, absolute_effects=[0.0])
        .evaluations[0]
    )
    np.testing.assert_allclose(
        treated_fit.draws.counterfactual.values,
        baseline.draws.counterfactual.values,
    )
    assert not np.allclose(
        treated_fit.draws.unit_impact.values,
        baseline.draws.unit_impact.values,
    )
    donors = history.copy()
    donors.iloc[-DURATION:, donors.columns.get_loc("a")] = (
        donors.iloc[-DURATION:, donors.columns.get_loc("a")] + 25
    )
    donor_fit = (
        HistoryOnlyPlan(
            donors,
            TREATED,
            CONTROLS,
            _decision(),
            sample_kwargs=LIGHT,
        )
        .evaluate(DURATION, absolute_effects=[0.0])
        .evaluations[0]
    )
    assert not np.allclose(
        donor_fit.draws.counterfactual.values,
        baseline.draws.counterfactual.values,
    )


def test_decision_fields_are_not_power() -> None:
    draws = xr.DataArray([[0.0, 1.0, 2.0]], dims=("chain", "draw"))
    rule = DecisionRule("positive", "cumulative_total", 1.0, 0.5)
    decision = decide(draws, rule)
    assert decision.posterior_tail_probability == pytest.approx(1 / 3)
    assert decision.decided is False
    assert decision.required_probability == 0.5
    assert decision.hdi_probability == HDI_PROB
    assert "power" not in decision.__dataclass_fields__
    negative = decide(draws, DecisionRule("negative", "cumulative_total", 1.0, 0.5))
    assert negative.posterior_tail_probability == pytest.approx(1 / 3)
    assert negative.decided is False


def test_two_sided_and_sequential_decisions_are_rejected() -> None:
    with pytest.raises(ValueError, match="Two-sided"):
        DecisionRule("two-sided", "cumulative_total", 0.0, 0.95)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sequential"):
        DecisionRule("sequential", "cumulative_total", 0.0, 0.95)  # type: ignore[arg-type]


def test_draw_aligned_sum_is_not_a_marginal_interval_sum() -> None:
    unit = xr.DataArray(
        np.array([[[[0.0, 10.0]], [[10.0, 0.0]]]]),
        dims=("chain", "draw", "obs_ind", "treated_units"),
        coords={
            "chain": [0],
            "draw": [0, 1],
            "obs_ind": [0],
            "treated_units": ["a", "b"],
        },
    )
    total, per_period, per_unit = aggregate_unit_impacts(unit)
    np.testing.assert_allclose(total.values, [[10.0, 10.0]])
    np.testing.assert_allclose(per_period.values, total.values)
    np.testing.assert_allclose(per_unit.values, [[5.0, 5.0]])
    from causalpy._arviz_compat import hdi_bounds

    group_low, group_high = hdi_bounds(total, prob=HDI_PROB, dim=("chain", "draw"))
    unit_bounds = [
        hdi_bounds(
            unit.isel(treated_units=i).sum("obs_ind"),
            prob=HDI_PROB,
            dim=("chain", "draw"),
        )
        for i in range(2)
    ]
    summed_low = unit_bounds[0][0] + unit_bounds[1][0]
    summed_high = unit_bounds[0][1] + unit_bounds[1][1]
    assert (summed_low, summed_high) != pytest.approx((group_low, group_high))
    names = [name for name, _reason in REJECTED_ESTIMATOR_MAPPINGS]
    assert "marginal_interval_sum" in names
    assert "raw_sum_versus_simplex" in names
    assert "aggregate_first_shared_weights" in names
    assert "average_of_relative_percentages" in names


def test_relative_group_effect_is_not_the_average_of_ratios() -> None:
    total = xr.DataArray([[10.0]], dims=("chain", "draw"))
    group_relative = float((total / 110.0).item())
    average_of_ratios = (10.0 / 10.0 + 0.0 / 100.0) / 2.0
    assert group_relative == pytest.approx(10.0 / 110.0)
    assert group_relative != pytest.approx(average_of_ratios)
    assert unit_period_effect(
        scale="absolute",
        magnitude=100.0,
        duration=4,
        n_treated=2,
    ) == pytest.approx(50.0)
    assert unit_period_effect(
        scale="relative",
        magnitude=0.1,
        duration=4,
        n_treated=2,
        baseline=100.0,
    ) == pytest.approx(1.25)
    with pytest.raises(ValueError, match="strictly positive"):
        require_relative_baseline(-1.0, floor=1e-8)
    assert require_relative_baseline(1e-12, floor=0.0) == pytest.approx(1e-12)


def test_reporting_consumes_canonical_results_only() -> None:
    import causalpy.planning.report as report

    assert not hasattr(report, "SyntheticControl")
    draws = xr.DataArray(
        np.zeros((1, 4, 2, 1)),
        dims=("chain", "draw", "obs_ind", "treated_units"),
        coords={"treated_units": ["austin"]},
    )
    from causalpy.planning.contracts import (
        CanonicalEvaluation,
        DataBoundary,
        EstimandDraws,
        EstimandSummary,
        FitDiagnostics,
        PlanningResult,
        Provenance,
        summarize_draws,
    )
    from causalpy.planning.contracts import estimand_draws as build_draws

    built: EstimandDraws = build_draws(draws, counterfactual=draws, baseline=None)
    summary: EstimandSummary = summarize_draws(
        built,
        baseline=None,
        baseline_convention=None,
        treated_count=1,
        duration=2,
        outcome_scale="sales",
    )
    evaluation = CanonicalEvaluation(
        estimator="synthetic_control",
        duration=2,
        scenario=EffectScenario("absolute", 0.0),
        draws=built,
        summary=summary,
        decision=decide(built.cumulative_total, _decision()),
        diagnostics=FitDiagnostics(0, None, None, "unavailable", None),
        boundary=DataBoundary(
            training_index=pd.Index([0, 1]),
            holdout_index=pd.Index([2, 3]),
            treated_units=("austin",),
            control_units=("boston",),
            post_period_treated_in_fit=False,
            post_period_donors_in_prediction=True,
            horizon_policy="separate-fits",
            rows_outside_window_excluded=True,
        ),
        provenance=Provenance(
            estimator="synthetic_control",
            adapter="SyntheticControlAdapter",
            model_class="WeightedSumFitter",
            random_seed=1,
            sample_kwargs=(("draws", 20),),
            horizon_policy="separate-fits",
            translation="deterministic_additive",
            uncertainty="posterior_conditional_mean",
        ),
        fit_index=0,
        outcome_scale="sales",
    )
    result = PlanningResult(
        evaluations=(evaluation,),
        n_inference_fits=1,
        capabilities=REFERENCE_CAPABILITIES,
    )
    table = evaluation_table(result)
    assert list(table.columns) == list(report.EVALUATION_COLUMNS)
    assert table.loc[0, "repeated_experiment_power"] is None
    figure = plot_evaluation(evaluation)
    figure.clear()
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_unimplemented_estimators_fail_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_fit(monkeypatch)
    for estimator in ("augmented_synthetic_control", "interactive_fixed_effects"):
        with pytest.raises(ValueError, match="#1440"):
            HistoryOnlyPlan(
                _history(),
                TREATED,
                CONTROLS,
                _decision(),
                estimator=estimator,
                sample_kwargs=LIGHT,
            )
    with pytest.raises(ValueError, match="Unknown estimator"):
        HistoryOnlyPlan(
            _history(),
            TREATED,
            CONTROLS,
            _decision(),
            estimator="not_an_estimator",
            sample_kwargs=LIGHT,
        )
    with pytest.raises(TypeError, match="WeightedSumFitter"):
        HistoryOnlyPlan(
            _history(),
            TREATED,
            CONTROLS,
            _decision(),
            model=LinearRegression(),
        )
    with pytest.raises(TypeError, match="WeightedSumFitter"):
        HistoryOnlyPlan(
            _history(),
            TREATED,
            CONTROLS,
            _decision(),
            model=SoftmaxWeightedSumFitter(sample_kwargs=LIGHT),
        )
    assert calls == []


def test_validation_matrix_and_ownership() -> None:
    assert FIXED_DESIGN_FIT_CEILING == 64
    names = [cell.name for cell in VALIDATION_MATRIX]
    assert names == [
        "correctly_specified_noisy_reference",
        "temporal_dependence",
        "persistent_misfit",
        "fixed_design_qualification",
        "group_draw_aligned_sum",
    ]
    assert VALIDATION_MATRIX[-1].owner == "#1440"
    assert all(cell.max_fits <= FIXED_DESIGN_FIT_CEILING for cell in VALIDATION_MATRIX)
    assert all(cell.qualification_scope == "fixed_design" for cell in VALIDATION_MATRIX)
    assert all(cell.environment == "nightly" for cell in VALIDATION_MATRIX)
    assert "screening" not in names
    assert "post_search_qualification" not in names
    owners = dict(OWNERSHIP)
    assert owners["causalpy/planning/reference.py"].startswith("#1438")
    assert owners["causalpy/planning/adapters.py"].startswith("#1440")
    assert owners["causalpy/planning/power.py"].startswith("#1439")
    assert owners["causalpy/planning/candidates.py"].startswith("#1441")


def test_group_size_and_input_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _forbid_fit(monkeypatch)
    with pytest.raises(ValueError, match="at most 3"):
        HistoryOnlyPlan(
            _history(),
            ["actual", "a", "b", "c"],
            ["d", "e"],
            _decision(),
            sample_kwargs=LIGHT,
        )
    with pytest.raises(ValueError, match="overlap"):
        HistoryOnlyPlan(
            _history(),
            TREATED,
            [TREATED, "a"],
            _decision(),
            sample_kwargs=LIGHT,
        )
    plan = _plan()
    with pytest.raises(TypeError, match="positive integer"):
        plan.evaluate(True, absolute_effects=[0.0])
    with pytest.raises(ValueError, match="training rows"):
        plan.evaluate(len(_history()) - 1, absolute_effects=[0.0])
    assert calls == []


def test_unsorted_history_is_rejected_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_fit(monkeypatch)
    history = _history().iloc[::-1]
    with pytest.raises(ValueError, match="must increase"):
        HistoryOnlyPlan(
            history,
            TREATED,
            CONTROLS,
            _decision(),
            sample_kwargs=LIGHT,
        )
    assert calls == []


def test_handoff_run_applies_a_missing_seed_before_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Capture:
        def __init__(
            self, *_args: object, model: WeightedSumFitter, **_kwargs: object
        ) -> None:
            seen.update(model.sample_kwargs)
            raise RuntimeError("captured")

    monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _Capture)
    plan = HistoryOnlyPlan(
        _history(),
        TREATED,
        CONTROLS,
        _decision(),
        sample_kwargs={"draws": 20, "tune": 20, "chains": 1},
    )
    handoff = plan.analysis_handoff(duration=DURATION)
    with pytest.raises(RuntimeError, match="captured"):
        handoff.run(_history(), random_seed=7)
    assert seen["random_seed"] == 7


def _fake_capabilities(estimator: str = "augmented_synthetic_control"):
    from causalpy.planning.contracts import Capabilities

    return Capabilities(
        estimator=estimator,
        deterministic_effect_reuse=False,
        horizon_reuse=False,
        post_period_donor_data=True,
        group_support=True,
        max_group_size=3,
        allowed_decision_rules=("positive",),
        evidence=(("group_support", "test_adapter"),),
    )


class _PairAdapter:
    def __init__(self) -> None:
        self.fits: list[tuple[str, ...]] = []

    def capabilities(self):
        return _fake_capabilities()

    def fit_window(self, request):
        self.fits.append(request.treated_units)
        return type("Fit", (), {"n_inference_fits": 1})()

    def translate(self, fit, scenario, decision, *, fit_index, outcome_scale):
        return type("Row", (), {"fit_index": fit_index})()


def test_passed_adapter_is_called_for_a_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_fit(monkeypatch)
    adapter = _PairAdapter()
    plan = HistoryOnlyPlan(
        _geo_panel(),
        ["austin", "boston"],
        ["chicago", "denver"],
        _decision(),
        estimator="augmented_synthetic_control",
        adapter=adapter,
    )
    result = plan.evaluate(4, absolute_effects=[0.0])
    assert calls == []
    assert adapter.fits == [("austin", "boston")]
    assert result.n_inference_fits == 1
    with pytest.raises(ValueError, match="own analysis path"):
        plan.analysis_handoff(duration=4)


def test_registered_adapter_is_constructed_and_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[tuple[object, dict[str, object], int | None]] = []
    fits: list[tuple[str, ...]] = []

    class _Registered:
        def __init__(self, *, model, sample_kwargs, random_seed) -> None:
            constructed.append((model, dict(sample_kwargs), random_seed))

        def capabilities(self):
            return _fake_capabilities()

        def fit_window(self, request):
            fits.append(request.treated_units)
            return type("Fit", (), {"n_inference_fits": 2})()

        def translate(self, fit, scenario, decision, *, fit_index, outcome_scale):
            return type("Row", (), {"fit_index": fit_index})()

    monkeypatch.setattr(
        "causalpy.planning.reference._registered_adapters",
        lambda: {"augmented_synthetic_control": _Registered},
    )
    calls = _forbid_fit(monkeypatch)
    seedless = {key: value for key, value in LIGHT.items() if key != "random_seed"}
    plan = HistoryOnlyPlan(
        _geo_panel(),
        ["austin", "boston"],
        ["chicago", "denver"],
        _decision(),
        estimator="augmented_synthetic_control",
        sample_kwargs=seedless,
        random_seed=7,
    )
    result = plan.evaluate((4, 6), absolute_effects=[0.0])
    assert calls == []
    assert constructed[0][0] is None
    assert constructed[0][1]["draws"] == 20
    assert constructed[0][1]["random_seed"] == 7
    assert constructed[0][2] == 7
    assert fits == [("austin", "boston"), ("austin", "boston")]
    assert result.n_inference_fits == 4


def test_adapter_module_registry_is_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    calls: list[object] = []

    class _Registered:
        def __init__(self, *, model, sample_kwargs, random_seed) -> None:
            calls.append(("init", model, random_seed))

        def capabilities(self):
            return _fake_capabilities()

        def fit_window(self, request):
            calls.append(request.treated_units)
            return type("Fit", (), {"n_inference_fits": 1})()

        def translate(self, fit, scenario, decision, *, fit_index, outcome_scale):
            return type("Row", (), {"fit_index": fit_index})()

    module = types.ModuleType("causalpy.planning.adapters")
    module.ADAPTERS = {"augmented_synthetic_control": _Registered}
    monkeypatch.setitem(sys.modules, "causalpy.planning.adapters", module)
    calls_fit = _forbid_fit(monkeypatch)
    plan = HistoryOnlyPlan(
        _geo_panel(),
        ["austin", "boston"],
        ["chicago", "denver"],
        _decision(),
        estimator="augmented_synthetic_control",
        sample_kwargs=LIGHT,
    )
    result = plan.evaluate(4, absolute_effects=[0.0])
    assert calls_fit == []
    assert calls[0][0] == "init"
    assert calls[1] == ("austin", "boston")
    assert result.n_inference_fits == 1


def test_adapter_rejects_a_parallel_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _forbid_fit(monkeypatch)
    with pytest.raises(TypeError, match="not both"):
        HistoryOnlyPlan(
            _history(),
            TREATED,
            CONTROLS,
            _decision(),
            adapter=_PairAdapter(),
            estimator="augmented_synthetic_control",
            sample_kwargs=LIGHT,
        )
    assert calls == []


def test_handoff_drops_rows_after_the_planned_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Capture:
        def __init__(self, data, treatment_time, **_kwargs: object) -> None:
            seen["rows"] = len(data)
            seen["post"] = int((data.index >= treatment_time).sum())
            raise RuntimeError("captured")

    monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _Capture)
    history = _history()
    handoff = _plan().analysis_handoff(duration=4)
    step = history.index[-1] - history.index[-2]
    extra = history.iloc[:3].copy()
    extra.index = [history.index[-1] + step * (i + 1) for i in range(3)]
    longer = pd.concat([history, extra])
    with pytest.raises(RuntimeError, match="captured"):
        handoff.run(longer)
    assert seen["post"] == 4
    assert seen["rows"] == len(history)


def test_handoff_refuses_a_short_realized_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_fit(monkeypatch)
    history = _history()
    handoff = _plan().analysis_handoff(duration=4)
    with pytest.raises(ValueError, match="Planned holdout labels"):
        handoff.run(history.iloc[:-1])
    assert calls == []


def test_handoff_reuses_the_model_the_plan_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Capture:
        def __init__(self, data, treatment_time, *, model, **_kwargs: object) -> None:
            seen["draws"] = model.sample_kwargs["draws"]
            seen["seed"] = model.sample_kwargs["random_seed"]
            seen["priors"] = model._user_priors
            raise RuntimeError("captured")

    monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _Capture)
    marker = {"kept": True}
    model = WeightedSumFitter(
        sample_kwargs={"draws": 5, "tune": 5, "chains": 1, "random_seed": 1},
        priors=marker,
    )
    plan = HistoryOnlyPlan(
        _history(),
        TREATED,
        CONTROLS,
        _decision(),
        model=model,
    )
    handoff = plan.analysis_handoff(duration=4)
    assert dict(handoff.sample_kwargs)["draws"] == 5
    assert dict(handoff.sample_kwargs)["random_seed"] == 1
    assert handoff.model is not model
    with pytest.raises(RuntimeError, match="captured"):
        handoff.run(_history())
    assert seen["draws"] == 5
    assert seen["seed"] == 1
    assert seen["priors"] == marker


def test_model_seed_disagreement_is_rejected_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_fit(monkeypatch)
    model = WeightedSumFitter(sample_kwargs={**LIGHT, "random_seed": 1})
    with pytest.raises(ValueError, match="random_seed"):
        HistoryOnlyPlan(
            _history(),
            TREATED,
            CONTROLS,
            _decision(),
            model=model,
            random_seed=7,
        )
    assert calls == []


def test_zero_preperiod_scale_rejects_a_relative_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_fit(monkeypatch)
    history = _history().copy()
    history[TREATED] = 0.0
    plan = HistoryOnlyPlan(
        history,
        TREATED,
        CONTROLS,
        _decision(),
        sample_kwargs=LIGHT,
    )
    with pytest.raises(ValueError, match="not positive"):
        plan.evaluate(4, relative_effects=[0.1], relative_baseline=1.0)
    assert calls == []


def test_registered_adapter_rejects_a_conflicting_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[object] = []

    class _Registered:
        def __init__(self, *, model, sample_kwargs, random_seed) -> None:
            constructed.append(random_seed)

        def capabilities(self):
            return _fake_capabilities()

    monkeypatch.setattr(
        "causalpy.planning.reference._registered_adapters",
        lambda: {"augmented_synthetic_control": _Registered},
    )
    calls = _forbid_fit(monkeypatch)
    with pytest.raises(ValueError, match="random_seed"):
        HistoryOnlyPlan(
            _geo_panel(),
            ["austin", "boston"],
            ["chicago", "denver"],
            _decision(),
            estimator="augmented_synthetic_control",
            sample_kwargs=LIGHT,
            random_seed=7,
        )
    assert constructed == []
    assert calls == []


def test_passed_adapter_refuses_an_unsupported_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    calls = _forbid_fit(monkeypatch)

    class _Small(_PairAdapter):
        def capabilities(self):
            return replace(_fake_capabilities(), group_support=False, max_group_size=1)

    with pytest.raises(ValueError, match="at most 1"):
        HistoryOnlyPlan(
            _geo_panel(),
            ["austin", "boston"],
            ["chicago", "denver"],
            _decision(),
            estimator="augmented_synthetic_control",
            adapter=_Small(),
        )
    assert calls == []


def test_passed_adapter_refuses_an_undeclared_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_fit(monkeypatch)
    with pytest.raises(ValueError, match="does not allow a negative"):
        HistoryOnlyPlan(
            _geo_panel(),
            "austin",
            ["boston", "chicago"],
            DecisionRule("negative", "cumulative_total", 0.0, 0.95),
            estimator="augmented_synthetic_control",
            adapter=_PairAdapter(),
        )
    assert calls == []


def test_model_and_sample_kwargs_are_rejected_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_fit(monkeypatch)
    model = WeightedSumFitter(sample_kwargs=LIGHT)
    with pytest.raises(TypeError, match="not both"):
        HistoryOnlyPlan(
            _history(),
            TREATED,
            CONTROLS,
            _decision(),
            model=model,
            sample_kwargs=LIGHT,
        )
    assert calls == []


def test_handoff_drops_rows_before_the_planned_training_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Capture:
        def __init__(self, data, treatment_time, **_kwargs: object) -> None:
            seen["index"] = data.index.copy()
            seen["rows"] = len(data)
            raise RuntimeError("captured")

    monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _Capture)
    history = _history()
    handoff = _plan().analysis_handoff(duration=4)
    step = history.index[1] - history.index[0]
    prefix = history.iloc[:2].copy()
    prefix.index = [history.index[0] - step * (2 - i) for i in range(2)]
    longer = pd.concat([prefix, history])
    with pytest.raises(RuntimeError, match="captured"):
        handoff.run(longer)
    assert seen["index"][0] == history.index[0]
    assert seen["rows"] == len(history)
    assert prefix.index[0] not in seen["index"]


def test_handoff_refuses_a_missing_planned_training_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_fit(monkeypatch)
    history = _history()
    handoff = _plan().analysis_handoff(duration=4)
    with pytest.raises(ValueError, match="Planned training labels"):
        handoff.run(history.iloc[3:])
    assert calls == []


def test_handoff_drops_a_label_inserted_inside_the_planned_holdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Capture:
        def __init__(self, data, treatment_time, **_kwargs: object) -> None:
            seen["index"] = data.index.copy()
            raise RuntimeError("captured")

    monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _Capture)
    history = _history()
    handoff = _plan().analysis_handoff(duration=4)
    holdout = list(handoff.holdout_index)
    assert holdout == list(history.index[-4:])
    assert handoff.treatment_time == holdout[0]
    inserted = holdout[0] + (holdout[1] - holdout[0]) / 2
    row = history.loc[[holdout[0]]].copy()
    row.index = [inserted]
    panel = pd.concat([history, row]).sort_index()
    with pytest.raises(RuntimeError, match="captured"):
        handoff.run(panel)
    assert list(seen["index"]) == list(handoff.training_index) + holdout
    assert inserted not in seen["index"]


def test_handoff_refuses_a_replaced_planned_holdout_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_fit(monkeypatch)
    history = _history()
    handoff = _plan().analysis_handoff(duration=4)
    last = handoff.holdout_index[-1]
    step = history.index[-1] - history.index[-2]
    replaced = history.rename(index={last: last + step})
    with pytest.raises(ValueError, match="Planned holdout labels"):
        handoff.run(replaced)
    assert calls == []


def test_handoff_refuses_a_one_row_training_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_fit(monkeypatch)
    history = _history()
    panel = history.iloc[-5:]
    handoff = AnalysisHandoff(
        treated_units=(TREATED,),
        control_units=tuple(CONTROLS),
        treatment_time=panel.index[1],
        duration=4,
        decision=_decision(),
        estimator="synthetic_control",
        sample_kwargs=tuple(LIGHT.items()),
        outcome_scale="outcome",
        training_index=(panel.index[0],),
        holdout_index=tuple(panel.index[1:]),
        time_column=None,
    )
    with pytest.raises(ValueError, match="Planned training window"):
        handoff.run(panel)
    assert calls == []


def test_handoff_drops_an_out_of_order_row_outside_the_planned_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Capture:
        def __init__(self, data, treatment_time, **_kwargs: object) -> None:
            seen["index"] = data.index.copy()
            raise RuntimeError("captured")

    monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _Capture)
    history = _history()
    handoff = _plan().analysis_handoff(duration=4)
    step = history.index[1] - history.index[0]
    extra = history.iloc[[0]].copy()
    extra.index = [history.index[0] - step]
    panel = pd.concat([history, extra])
    with pytest.raises(RuntimeError, match="captured"):
        handoff.run(panel)
    planned = list(handoff.training_index) + list(handoff.holdout_index)
    assert list(seen["index"]) == planned
    assert extra.index[0] not in seen["index"]


def test_handoff_drops_a_duplicated_label_outside_the_planned_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Capture:
        def __init__(self, data, treatment_time, **_kwargs: object) -> None:
            seen["index"] = data.index.copy()
            raise RuntimeError("captured")

    monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _Capture)
    history = _history()
    handoff = _plan().analysis_handoff(duration=4)
    step = history.index[1] - history.index[0]
    duplicate = history.iloc[[0]].copy()
    duplicate.index = [history.index[-1] + step * 3]
    panel = pd.concat([history, duplicate, duplicate])
    with pytest.raises(RuntimeError, match="captured"):
        handoff.run(panel)
    planned = list(handoff.training_index) + list(handoff.holdout_index)
    assert list(seen["index"]) == planned
    assert duplicate.index[0] not in seen["index"]


def test_handoff_refuses_a_duplicated_planned_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_fit(monkeypatch)
    history = _history()
    handoff = _plan().analysis_handoff(duration=4)
    label = handoff.training_index[0]
    panel = pd.concat([history, history.loc[[label]]])
    with pytest.raises(ValueError, match="duplicated") as exc:
        handoff.run(panel)
    assert "tail" not in str(exc.value)
    assert calls == []


def test_handoff_refuses_a_non_increasing_planned_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_fit(monkeypatch)
    history = _history()
    handoff = _plan().analysis_handoff(duration=4)
    training = list(handoff.training_index)
    broken = replace(handoff, training_index=(training[1], training[0], *training[2:]))
    with pytest.raises(ValueError, match="not strictly increasing"):
        broken.run(history)
    assert calls == []


def test_handoff_drops_an_unsorted_time_column_row_outside_the_planned_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Capture:
        def __init__(self, data, treatment_time, **_kwargs: object) -> None:
            seen["index"] = data.index.copy()
            raise RuntimeError("captured")

    monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _Capture)
    frame = _history().copy()
    frame["week"] = pd.date_range("2020-01-01", periods=len(frame), freq="W")
    frame = frame.reset_index(drop=True)
    plan = HistoryOnlyPlan(
        frame,
        TREATED,
        CONTROLS,
        _decision(),
        sample_kwargs=LIGHT,
        time_column="week",
    )
    handoff = plan.analysis_handoff(duration=4)
    late = frame.iloc[[0]].copy()
    late["week"] = frame["week"].iloc[0] - pd.Timedelta(weeks=3)
    unsorted = pd.concat([frame, late], ignore_index=True)
    with pytest.raises(RuntimeError, match="captured"):
        handoff.run(unsorted)
    planned = list(handoff.training_index) + list(handoff.holdout_index)
    assert list(seen["index"]) == planned
    assert late["week"].iloc[0] not in list(seen["index"])


def test_handoff_drops_a_duplicated_time_value_outside_the_planned_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Capture:
        def __init__(self, data, treatment_time, **_kwargs: object) -> None:
            seen["index"] = data.index.copy()
            raise RuntimeError("captured")

    monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _Capture)
    frame = _history().copy()
    frame["week"] = pd.date_range("2020-01-01", periods=len(frame), freq="W")
    frame = frame.reset_index(drop=True)
    plan = HistoryOnlyPlan(
        frame,
        TREATED,
        CONTROLS,
        _decision(),
        sample_kwargs=LIGHT,
        time_column="week",
    )
    handoff = plan.analysis_handoff(duration=4)
    extra = frame.iloc[[0]].copy()
    extra["week"] = frame["week"].iloc[-1] + pd.Timedelta(weeks=5)
    panel = pd.concat([frame, extra, extra], ignore_index=True)
    with pytest.raises(RuntimeError, match="captured"):
        handoff.run(panel)
    planned = list(handoff.training_index) + list(handoff.holdout_index)
    assert list(seen["index"]) == planned
    assert extra["week"].iloc[0] not in list(seen["index"])


def test_handoff_uses_the_stored_time_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Capture:
        def __init__(self, data, treatment_time, **_kwargs: object) -> None:
            seen["rows"] = len(data)
            seen["time"] = data.index[0]
            raise RuntimeError("captured")

    monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _Capture)
    frame = _history().copy()
    frame["week"] = pd.date_range("2020-01-01", periods=len(frame), freq="W")
    frame = frame.reset_index(drop=True)
    plan = HistoryOnlyPlan(
        frame,
        TREATED,
        CONTROLS,
        _decision(),
        sample_kwargs=LIGHT,
        time_column="week",
    )
    handoff = plan.analysis_handoff(duration=4)
    assert handoff.time_column == "week"
    with pytest.raises(RuntimeError, match="captured"):
        handoff.run(frame)
    assert seen["rows"] == len(frame)
    assert seen["time"] == frame["week"].iloc[0]
