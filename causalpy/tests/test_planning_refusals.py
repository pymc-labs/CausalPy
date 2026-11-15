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
"""Refusals the reference path must raise before a sample starts."""

from __future__ import annotations

import sys
import types
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.custom_exceptions import DataException
from causalpy.planning import (
    AnalysisHandoff,
    CandidateRequest,
    DecisionRule,
    EffectScenario,
    HistoryOnlyPlan,
    PowerCriterion,
    PowerRequest,
    WindowRequest,
    aggregate_unit_impacts,
    decide,
    require_relative_baseline,
)
from causalpy.planning.contracts import (
    default_relative_floor,
    require_finite,
    require_positive_int,
)
from causalpy.planning.reference import SyntheticControlAdapter
from causalpy.pymc_models import WeightedSumFitter

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


def _history() -> pd.DataFrame:
    return cp.load_data("sc").iloc[:48][CONTROLS + [TREATED]]


def _decision() -> DecisionRule:
    return DecisionRule("positive", "cumulative_total", 0.0, 0.95)


def _plan(**kwargs: object) -> HistoryOnlyPlan:
    return HistoryOnlyPlan(
        _history(),
        TREATED,
        CONTROLS,
        _decision(),
        sample_kwargs=LIGHT,
        **kwargs,
    )


def _forbid_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("SyntheticControl was called")

    monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _boom)


def _frame(index: list[object]) -> pd.DataFrame:
    return pd.DataFrame({TREATED: 1.0, "a": 1.0}, index=index)


def _spec(**overrides: object) -> AnalysisHandoff:
    values: dict[str, object] = {
        "treated_units": (TREATED,),
        "control_units": ("a",),
        "treatment_time": 3,
        "duration": 1,
        "decision": _decision(),
        "estimator": "synthetic_control",
        "sample_kwargs": tuple(LIGHT.items()),
        "outcome_scale": "outcome",
        "training_index": (1, 2),
        "holdout_index": (3,),
        "time_column": None,
        "model": None,
    }
    values.update(overrides)
    return AnalysisHandoff(**values)


def _window(**overrides: object) -> WindowRequest:
    frame = pd.DataFrame({TREATED: [1.0, 2.0, 3.0, 4.0], "a": [1.0, 1.0, 1.0, 1.0]})
    values: dict[str, object] = {
        "history": frame,
        "treated_units": (TREATED,),
        "control_units": ("a",),
        "treatment_time": 2,
        "duration": 2,
        "outcome_scale": "outcome",
    }
    values.update(overrides)
    return WindowRequest(**values)


def _adapter(**overrides: object) -> SyntheticControlAdapter:
    values: dict[str, object] = {
        "model": None,
        "sample_kwargs": LIGHT,
        "random_seed": None,
        "treated_units": (TREATED,),
    }
    values.update(overrides)
    return SyntheticControlAdapter(**values)


def _criterion(**overrides: object) -> PowerCriterion:
    values: dict[str, object] = {
        "target_power": 0.8,
        "false_positive_tolerance": 0.05,
        "precision_target": 1.0,
        "monte_carlo_error_tolerance": 0.02,
        "qualification_scope": "fixed_design",
    }
    values.update(overrides)
    return PowerCriterion(**values)


def test_numeric_contracts_reject_bools_and_non_finite_values() -> None:
    with pytest.raises(TypeError, match="not a bool"):
        DecisionRule("positive", "cumulative_total", True, 0.95)
    with pytest.raises(TypeError, match="finite real number"):
        DecisionRule("positive", "cumulative_total", float("nan"), 0.95)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        DecisionRule("positive", "cumulative_total", 0.0, 0.0)
    with pytest.raises(ValueError, match="positive integer"):
        _plan().evaluate(0, absolute_effects=[0.0])
    with pytest.raises(ValueError, match="estimand must be"):
        DecisionRule("positive", "power", 0.0, 0.95)
    with pytest.raises(ValueError, match="scale must be"):
        EffectScenario("ratio", 1.0)
    with pytest.raises(ValueError, match="does not take a baseline"):
        EffectScenario("absolute", 1.0, baseline=2.0)
    with pytest.raises(ValueError, match="requires a baseline"):
        EffectScenario("relative", 1.0)
    with pytest.raises(TypeError, match="finite real number"):
        require_finite("no", "value")
    with pytest.raises(TypeError, match="positive integer"):
        require_positive_int(0, "duration")
    with pytest.raises(TypeError, match="positive integer"):
        require_positive_int(1.5, "duration")
    assert _criterion(precision_target=None).precision_target is None


def test_unit_names_and_history_fail_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_fit(monkeypatch)
    history = _history()
    decision = _decision()
    with pytest.raises(TypeError, match="unit name or a sequence"):
        HistoryOnlyPlan(history, 1, CONTROLS, decision, sample_kwargs=LIGHT)
    with pytest.raises(TypeError, match="non-empty strings"):
        HistoryOnlyPlan(history, "", CONTROLS, decision, sample_kwargs=LIGHT)
    with pytest.raises(ValueError, match="unique"):
        HistoryOnlyPlan(
            history, [TREATED, TREATED], ["a"], decision, sample_kwargs=LIGHT
        )
    with pytest.raises(ValueError, match="at least 1"):
        HistoryOnlyPlan(history, TREATED, [], decision, sample_kwargs=LIGHT)
    with pytest.raises(TypeError, match="DataFrame"):
        HistoryOnlyPlan(
            np.ones((5, 3)), TREATED, CONTROLS, decision, sample_kwargs=LIGHT
        )
    short = history.iloc[:2]
    with pytest.raises(ValueError, match="at least 3 rows"):
        HistoryOnlyPlan(short, TREATED, CONTROLS, decision, sample_kwargs=LIGHT)
    duplicated = history.iloc[:6].copy()
    duplicated.index = [0, 1, 2, 3, 4, 4]
    with pytest.raises(ValueError, match="must be unique"):
        HistoryOnlyPlan(duplicated, TREATED, CONTROLS, decision, sample_kwargs=LIGHT)
    renamed = history.copy()
    renamed.columns = range(renamed.shape[1])
    with pytest.raises(TypeError, match="string unit names"):
        HistoryOnlyPlan(renamed, TREATED, CONTROLS, decision, sample_kwargs=LIGHT)
    with pytest.raises(ValueError, match="missing units"):
        HistoryOnlyPlan(history, "missing", CONTROLS, decision, sample_kwargs=LIGHT)
    non_numeric = history.copy()
    non_numeric[TREATED] = "x"
    with pytest.raises(TypeError, match="must be numeric"):
        HistoryOnlyPlan(non_numeric, TREATED, CONTROLS, decision, sample_kwargs=LIGHT)
    non_finite = history.copy()
    non_finite.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        HistoryOnlyPlan(non_finite, TREATED, CONTROLS, decision, sample_kwargs=LIGHT)


def test_plan_constructor_rejects_bad_decision_scale_and_fit_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_fit(monkeypatch)
    history = _history()
    with pytest.raises(TypeError, match="DecisionRule"):
        HistoryOnlyPlan(history, TREATED, CONTROLS, "positive", sample_kwargs=LIGHT)
    with pytest.raises(TypeError, match="outcome_scale"):
        HistoryOnlyPlan(
            history,
            TREATED,
            CONTROLS,
            _decision(),
            sample_kwargs=LIGHT,
            outcome_scale="",
        )
    with pytest.raises(TypeError, match="estimator must be a string"):
        HistoryOnlyPlan(
            history, TREATED, CONTROLS, _decision(), estimator=1, sample_kwargs=LIGHT
        )
    with pytest.raises(TypeError, match="sample_kwargs or a WeightedSumFitter"):
        HistoryOnlyPlan(history, TREATED, CONTROLS, _decision())
    with pytest.raises(TypeError, match="not both"):
        SyntheticControlAdapter(
            model=WeightedSumFitter(sample_kwargs=LIGHT),
            sample_kwargs=LIGHT,
            random_seed=None,
            treated_units=(TREATED,),
        )
    with pytest.raises(TypeError, match="must be a mapping"):
        HistoryOnlyPlan(history, TREATED, CONTROLS, _decision(), sample_kwargs="draws")
    with pytest.raises(TypeError, match="random_seed must be an integer"):
        HistoryOnlyPlan(
            history,
            TREATED,
            CONTROLS,
            _decision(),
            sample_kwargs=LIGHT,
            random_seed=True,
        )
    with pytest.raises(TypeError, match="random_seed"):
        HistoryOnlyPlan(
            history,
            TREATED,
            CONTROLS,
            _decision(),
            sample_kwargs={**LIGHT, "random_seed": True},
        )
    with pytest.raises(TypeError, match="WeightedSumFitter"):
        HistoryOnlyPlan(
            history,
            TREATED,
            CONTROLS,
            _decision(),
            model=LinearRegression(),
            random_seed=1,
        )


def test_evaluate_rejects_bad_horizons_and_effects_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_fit(monkeypatch)
    plan = _plan()
    with pytest.raises(TypeError, match="positive integer or a sequence"):
        plan.evaluate("4", absolute_effects=[0.0])
    with pytest.raises(ValueError, match="at least one duration"):
        plan.evaluate([], absolute_effects=[0.0])
    with pytest.raises(ValueError, match="must be unique"):
        plan.evaluate([1, 1], absolute_effects=[0.0])
    with pytest.raises(ValueError, match="at least one absolute or relative"):
        plan.evaluate(1)
    with pytest.raises(TypeError, match="sequence of finite"):
        plan.evaluate(1, absolute_effects=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        plan.evaluate(
            1, relative_effects=[0.1], relative_baseline=1.0, relative_floor=-1
        )
    with pytest.raises(ValueError, match="strictly positive"):
        plan.evaluate(
            1, relative_effects=[0.1], relative_baseline=0.0, relative_floor=0.0
        )


def test_relative_baseline_helpers_reject_undefined_scales() -> None:
    with pytest.raises(ValueError, match="non-empty and finite"):
        default_relative_floor(np.array([]))
    with pytest.raises(ValueError, match="non-negative"):
        require_relative_baseline(1.0, floor=-1.0)
    with pytest.raises(ValueError, match="scale must be"):
        from causalpy.planning.contracts import unit_period_effect

        unit_period_effect(scale="ratio", magnitude=1.0, duration=2, n_treated=1)
    with pytest.raises(ValueError, match="does not take a baseline"):
        from causalpy.planning.contracts import unit_period_effect

        unit_period_effect(
            scale="absolute", magnitude=1.0, duration=2, n_treated=1, baseline=3.0
        )
    with pytest.raises(ValueError, match="requires a baseline"):
        from causalpy.planning.contracts import unit_period_effect

        unit_period_effect(scale="relative", magnitude=1.0, duration=2, n_treated=1)


def test_draw_contracts_reject_the_wrong_shape() -> None:
    wrong = xr.DataArray(np.ones((1, 1)), dims=("chain", "draw"))
    with pytest.raises(ValueError, match="unit_impact must have dimensions"):
        aggregate_unit_impacts(wrong)
    empty = xr.DataArray(
        np.ones((1, 1, 0, 1)),
        dims=("chain", "draw", "obs_ind", "treated_units"),
    )
    with pytest.raises(ValueError, match="at least one period"):
        aggregate_unit_impacts(empty)
    missing = xr.DataArray(
        np.array([[[[np.nan]]]]),
        dims=("chain", "draw", "obs_ind", "treated_units"),
    )
    with pytest.raises(ValueError, match="must be finite"):
        aggregate_unit_impacts(missing)
    with pytest.raises(ValueError, match="Decision draws must have dimensions"):
        decide(empty, _decision())
    blank = xr.DataArray(np.array([[]]), dims=("chain", "draw"))
    with pytest.raises(ValueError, match="non-empty and finite"):
        decide(blank, _decision())


def test_power_and_candidate_contracts_reject_undeclared_inputs() -> None:
    criterion = _criterion()
    assert criterion.qualification_scope == "fixed_design"
    assert criterion.precision_target == 1.0
    with pytest.raises(ValueError, match="qualification_scope"):
        _criterion(qualification_scope="launch")
    request = PowerRequest(
        scenario=EffectScenario("absolute", 0.0),
        decision=_decision(),
        criterion=criterion,
        estimator="synthetic_control",
        group_size=1,
        duration=4,
    )
    assert request.duration == 4
    with pytest.raises(ValueError, match="Unknown estimator"):
        PowerRequest(
            scenario=EffectScenario("absolute", 0.0),
            decision=_decision(),
            criterion=criterion,
            estimator="not_declared",
            group_size=1,
            duration=4,
        )
    with pytest.raises(ValueError, match="group_size must be from 1 to 3"):
        PowerRequest(
            scenario=EffectScenario("absolute", 0.0),
            decision=_decision(),
            criterion=criterion,
            estimator="synthetic_control",
            group_size=4,
            duration=4,
        )
    accepted = CandidateRequest(("a",), 1, 3, ())
    assert accepted.max_group_size == 3
    with pytest.raises(ValueError, match="1..3"):
        CandidateRequest(("a",), 0, 3, ())


def test_passed_adapter_must_match_the_declared_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_fit(monkeypatch)
    history = _history()
    decision = _decision()

    class _Caps:
        def __init__(self, estimator: str = "augmented_synthetic_control") -> None:
            self.estimator = estimator
            self.max_group_size = 3
            self.group_support = True
            self.allowed_decision_rules = ("positive",)

    class _Adapter:
        def __init__(self, caps: object) -> None:
            self._caps = caps
            self.calls = 0

        def capabilities(self) -> object:
            self.calls += 1
            if self.calls >= 3:
                raise AttributeError("capabilities disappeared")
            return self._caps

        def fit_window(self, request: object) -> object:
            raise AssertionError("fit")

        def translate(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("translate")

    with pytest.raises(TypeError, match="EstimatorAdapter"):
        HistoryOnlyPlan(
            history,
            TREATED,
            CONTROLS,
            decision,
            estimator="augmented_synthetic_control",
            adapter=object(),
        )
    with pytest.raises(ValueError, match="not estimator"):
        HistoryOnlyPlan(
            history,
            TREATED,
            CONTROLS,
            decision,
            estimator="augmented_synthetic_control",
            adapter=_Adapter(_Caps("synthetic_control")),
        )
    with pytest.raises(TypeError, match="estimator must be a string"):
        HistoryOnlyPlan(
            history,
            TREATED,
            CONTROLS,
            decision,
            estimator="",
            adapter=_Adapter(_Caps()),
        )
    broken_size = _Adapter(_Caps())
    broken_size._caps.max_group_size = True
    with pytest.raises(TypeError, match="max_group_size must be an int"):
        HistoryOnlyPlan(
            history,
            TREATED,
            CONTROLS,
            decision,
            estimator="augmented_synthetic_control",
            adapter=broken_size,
        )
    changing = _Adapter(_Caps())
    plan = HistoryOnlyPlan(
        history,
        TREATED,
        CONTROLS,
        decision,
        estimator="augmented_synthetic_control",
        adapter=changing,
    )
    with pytest.raises(TypeError, match="EstimatorAdapter"):
        plan.evaluate(1, absolute_effects=[0.0])

    class _Renamed:
        def __init__(self) -> None:
            self.calls = 0

        def capabilities(self) -> _Caps:
            self.calls += 1
            name = "augmented_synthetic_control" if self.calls < 3 else "other"
            return _Caps(name)

        def fit_window(self, request: object) -> object:
            raise AssertionError("fit")

        def translate(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("translate")

    renamed = _Renamed()
    plan = HistoryOnlyPlan(
        history,
        TREATED,
        CONTROLS,
        decision,
        estimator="augmented_synthetic_control",
        adapter=renamed,
    )
    with pytest.raises(ValueError, match="not estimator"):
        plan.evaluate(1, absolute_effects=[0.0])


def test_foreign_import_errors_are_reraised(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_name: str) -> None:
        raise ModuleNotFoundError("missing dependency", name="other.missing")

    monkeypatch.setattr(
        sys.modules["causalpy.planning.reference"].importlib,
        "import_module",
        _boom,
    )
    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        HistoryOnlyPlan(
            _history(),
            TREATED,
            CONTROLS,
            _decision(),
            estimator="augmented_synthetic_control",
            sample_kwargs=LIGHT,
        )


def test_adapter_registry_must_be_a_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("causalpy.planning.adapters")
    module.ADAPTERS = []
    monkeypatch.setitem(sys.modules, "causalpy.planning.adapters", module)
    monkeypatch.setattr(
        sys.modules["causalpy.planning.reference"].importlib,
        "import_module",
        lambda _name: module,
    )
    with pytest.raises(TypeError, match="ADAPTERS must be a dict"):
        HistoryOnlyPlan(
            _history(),
            TREATED,
            CONTROLS,
            _decision(),
            estimator="augmented_synthetic_control",
            sample_kwargs=LIGHT,
        )


def test_adapter_construction_errors_are_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BadAdapter:
        def __init__(
            self, *, model: object, sample_kwargs: object, random_seed: object
        ) -> None:
            raise TypeError("seed must be int")

    monkeypatch.setattr(
        sys.modules["causalpy.planning.reference"],
        "_registered_adapters",
        lambda: {"augmented_synthetic_control": _BadAdapter},
    )
    with pytest.raises(TypeError, match="adapter construction failed"):
        HistoryOnlyPlan(
            _history(),
            TREATED,
            CONTROLS,
            _decision(),
            estimator="augmented_synthetic_control",
            sample_kwargs=LIGHT,
        )


def test_reference_window_checks_run_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_fit(monkeypatch)
    adapter = _adapter()
    with pytest.raises(ValueError, match="#1440"):
        adapter.fit_window(_window(treated_units=("actual", "a")))
    with pytest.raises(ValueError, match="positive integer"):
        adapter.fit_window(_window(duration=0))
    short = pd.DataFrame({TREATED: [1.0, 2.0, 3.0], "a": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="fewer than"):
        adapter.fit_window(_window(history=short, treatment_time=1, duration=2))
    with pytest.raises(ValueError, match="first label of the holdout"):
        adapter.fit_window(_window(treatment_time=1))

    model = WeightedSumFitter(sample_kwargs=LIGHT)
    model._clone = lambda: object()
    cloned = _adapter(model=model, sample_kwargs=None)
    with pytest.raises(TypeError, match="did not return WeightedSumFitter"):
        cloned.fit_window(_window())

    class _NoImpact:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.result = None

        def fit(self) -> _NoImpact:
            return self

    monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _NoImpact)
    with pytest.raises(RuntimeError, match="posterior impact"):
        _adapter().fit_window(_window())


def test_malformed_impact_is_refused_before_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Result:
        def __init__(self, impact: xr.DataArray, prediction: xr.DataArray) -> None:
            self.impact_post = impact
            self.predictions_post = prediction

    class _Experiment:
        result: _Result

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.result = _Result(self.impact, self.prediction)

        def fit(self) -> _Experiment:
            return self

    def _install(impact: xr.DataArray, prediction: xr.DataArray) -> None:
        _Experiment.impact = impact
        _Experiment.prediction = prediction
        monkeypatch.setattr("causalpy.planning.reference.SyntheticControl", _Experiment)

    missing_unit = xr.DataArray(np.ones((1, 1, 2)), dims=("chain", "draw", "obs_ind"))
    broken = xr.DataArray(np.ones((1,)), dims=("chain",))
    _install(missing_unit, broken)
    with pytest.raises(ValueError, match="missing dimensions"):
        _adapter().fit_window(_window())

    short = xr.DataArray(
        np.ones((1, 1, 1, 1)),
        dims=("chain", "draw", "obs_ind", "treated_units"),
    )
    _install(short, short)
    with pytest.raises(ValueError, match="holdout length"):
        _adapter().fit_window(_window())

    wide = xr.DataArray(
        np.ones((1, 1, 2, 2)),
        dims=("chain", "draw", "obs_ind", "treated_units"),
    )
    _install(wide, wide)
    with pytest.raises(ValueError, match="treated-unit count"):
        _adapter().fit_window(_window())


def test_reference_fit_count_guard_rejects_a_second_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_fit(monkeypatch)
    plan = _plan()

    def _two(_request: object) -> object:
        return type("Fit", (), {"n_inference_fits": 2})()

    plan._adapter.fit_window = _two
    with pytest.raises(RuntimeError, match="one fit per horizon"):
        plan.evaluate(1, absolute_effects=[0.0])


def test_handoff_run_rejects_the_wrong_estimator_and_fit_arguments() -> None:
    handoff = _spec()
    panel = _frame([1, 2, 3])
    with pytest.raises(ValueError, match="one-unit synthetic control"):
        replace(handoff, estimator="augmented_synthetic_control").run(panel)
    with pytest.raises(TypeError, match="not both"):
        handoff.run(
            panel,
            model=WeightedSumFitter(sample_kwargs=LIGHT),
            sample_kwargs=LIGHT,
        )
    with pytest.raises(TypeError, match="WeightedSumFitter"):
        handoff.run(panel, model=LinearRegression())
    with pytest.raises(TypeError, match="must be a mapping"):
        handoff.run(panel, sample_kwargs="draws")
    with pytest.raises(TypeError, match="DataFrame"):
        handoff.run([[1, 2, 3]])
    with pytest.raises(DataException, match="not a column"):
        handoff.run(panel, time_column="week")
    named = panel.copy()
    named["week"] = [1, 2, 3]
    named.index.name = "row"
    with pytest.raises(DataException, match="meaningful index"):
        handoff.run(named, time_column="week")
    integer = panel.copy()
    integer.columns = [0, 1]
    with pytest.raises(TypeError, match="string unit names"):
        handoff.run(integer)


def test_stored_window_must_be_comparable_and_consistent() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        _spec(training_index=(1, 1)).run(_frame([1, 2, 3]))
    with pytest.raises(ValueError, match="comparable with treatment_time"):
        _spec(training_index=("a", "b"), treatment_time=1).run(_frame(["a", "b"]))
    with pytest.raises(ValueError, match="must precede"):
        _spec(training_index=(3, 4), treatment_time=2).run(_frame([3, 4]))
    with pytest.raises(ValueError, match="not the"):
        _spec(
            training_index=(1, 2), holdout_index=(3,), duration=2, treatment_time=3
        ).run(_frame([1, 2, 3]))
    with pytest.raises(ValueError, match="holdout labels must be unique"):
        _spec(
            training_index=(1, 2),
            holdout_index=(3, 3),
            duration=2,
            treatment_time=3,
        ).run(_frame([1, 2, 3]))
    with pytest.raises(ValueError, match="must be treatment_time"):
        _spec(
            training_index=(1, 2),
            holdout_index=(4, 5),
            duration=2,
            treatment_time=3,
        ).run(_frame([1, 2, 4, 5]))
    with pytest.raises(ValueError, match="overlap"):
        _spec(
            training_index=(1, 3),
            holdout_index=(5, 3),
            duration=2,
            treatment_time=5,
        ).run(_frame([1, 3, 5]))
    with pytest.raises(ValueError, match="must be comparable"):
        _spec(
            training_index=(1, 2),
            holdout_index=(3, "x"),
            duration=2,
            treatment_time=3,
        ).run(_frame([1, 2, 3, "x"]))
    with pytest.raises(ValueError, match="must start at treatment_time and"):
        _spec(
            training_index=(0, 1),
            holdout_index=(3, 2),
            duration=2,
            treatment_time=3,
        ).run(_frame([0, 1, 2, 3]))
