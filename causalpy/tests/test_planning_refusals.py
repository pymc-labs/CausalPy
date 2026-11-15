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

import json
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
    CANDIDATE_COLUMNS,
    CANDIDATE_FILTER_REASONS,
    AnalysisHandoff,
    BalanceConstraint,
    CandidateRequest,
    CandidateTable,
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
    accepted = CandidateRequest(("a",), 1, 3, random_seed=0)
    assert accepted.max_group_size == 3
    assert accepted.random_seed == 0
    with pytest.raises(ValueError, match="1..3"):
        CandidateRequest(("a",), 0, 3, random_seed=0)
    with pytest.raises(TypeError):
        CandidateRequest(("a",), 1, 3, ())
    with pytest.raises(TypeError, match="constraints"):
        CandidateRequest(("a",), 1, 3, random_seed=0, constraints=())
    with pytest.raises(TypeError, match="random_seed"):
        CandidateRequest(("a",), 1, 3)


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


def _selection_history() -> pd.DataFrame:
    return pd.DataFrame(
        {"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]},
        index=pd.Index([10, 20], name="time"),
    )


def _request(**overrides: object) -> CandidateRequest:
    fields: dict[str, object] = {
        "units": ("a", "b", "c"),
        "min_group_size": 1,
        "max_group_size": 3,
        "random_seed": 0,
    }
    fields.update(overrides)
    return CandidateRequest(**fields)


def _candidate_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": "a",
        "treated": ("a",),
        "donors": ("b",),
        "n_donors": 1,
        "estimand": "draw_aligned_unit_impact_sum",
        "member_weight": 1.0,
        "cost": 1.0,
        "outcome_share": pd.NA,
        "proxy_score": pd.NA,
        "proxy_role": "unavailable",
        "boundary": False,
        "status": "shortlisted",
        "filter_reasons": (),
        "shortlist_rank": 1,
    }
    row.update(overrides)
    if "candidate_id" not in overrides:
        treated = tuple(row["treated"])  # type: ignore[arg-type]
        donors = tuple(row["donors"])  # type: ignore[arg-type]
        row["candidate_id"] = json.dumps(
            [list(treated), list(donors)], separators=(",", ":")
        )
    return row


def _empty_candidate_frame() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=object) for name in CANDIDATE_COLUMNS})


def _table(**overrides: object) -> CandidateTable:
    fields: dict[str, object] = {
        "frame": _empty_candidate_frame(),
        "n_space": 0,
        "n_enumerated": 0,
        "n_filtered": 0,
        "n_shortlisted": 0,
        "partial": False,
        "enumeration_policy": "exact",
        "estimand": "draw_aligned_unit_impact_sum",
        "n_inference_fits": 0,
    }
    fields.update(overrides)
    return CandidateTable(**fields)


def test_candidate_constants_are_the_screen_contract() -> None:
    assert CANDIDATE_COLUMNS == (
        "candidate_id",
        "treated",
        "donors",
        "n_donors",
        "estimand",
        "member_weight",
        "cost",
        "outcome_share",
        "proxy_score",
        "proxy_role",
        "boundary",
        "status",
        "filter_reasons",
        "shortlist_rank",
    )
    assert CANDIDATE_FILTER_REASONS == (
        "excluded_unit",
        "not_treatment_eligible",
        "not_donor_eligible",
        "missing_required",
        "donor_depleted",
        "spillover_donor",
        "over_budget",
        "cost_unknown",
        "outcome_share",
        "representativeness",
        "duplicate",
        "group_size",
    )


def test_candidate_request_accepts_a_consistent_screen() -> None:
    request = _request(
        treatment_eligible=("a", "b"),
        donor_eligible=("b", "c"),
        always_control=("c",),
        required_treated=("a",),
        fixed_treated=("a", "b"),
        donor_pools=(("c",),),
        spillover_exclusions=(("a", "b"),),
        history=_selection_history(),
        selection_end=20,
        min_outcome_share=0.0,
        max_outcome_share=1.0,
        unit_costs=(("a", 1.0), ("b", 2.0)),
        budget=10.0,
        unit_covariates=(("income", (("a", 1.0), ("b", 2.0))),),
        balance_constraints=(BalanceConstraint("income", 0.5),),
        random_seed=7,
    )
    assert request.units == ("a", "b", "c")
    assert request.fixed_treated == ("a", "b")
    assert request.random_seed == 7
    assert request.exact_ceiling == 5000
    listed = _request(units=["a", "b"], max_group_size=1, always_control=["b"])
    assert listed.units == ("a", "b")
    assert listed.always_control == ("b",)


def test_candidate_request_rejects_inconsistent_screens() -> None:
    history = _selection_history()
    with pytest.raises(ValueError, match="unknown unit"):
        _request(always_control=("z",))
    with pytest.raises(ValueError, match="always_control and excluded_units overlap"):
        _request(always_control=("a",), excluded_units=("a",))
    with pytest.raises(ValueError, match="longer than max_group_size"):
        _request(required_treated=("a", "b"), max_group_size=1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _request(supplied_groups=(("a",),), fixed_treated=("a",))
    with pytest.raises(ValueError, match="donor_pools requires fixed_treated"):
        _request(donor_pools=(("b",),))
    with pytest.raises(ValueError, match="self-pair"):
        _request(spillover_exclusions=(("a", "a"),))
    with pytest.raises(ValueError, match="selection_end is required"):
        _request(history=history)
    with pytest.raises(ValueError, match="selection_end requires history"):
        _request(selection_end=20)
    with pytest.raises(ValueError, match="not in the history index"):
        _request(history=history, selection_end=99)
    with pytest.raises(ValueError, match="min_outcome_share must be in"):
        _request(min_outcome_share=-0.1)
    with pytest.raises(ValueError, match="max_outcome_share must be in"):
        _request(max_outcome_share=1.1)
    with pytest.raises(ValueError, match="must not exceed"):
        _request(min_outcome_share=0.8, max_outcome_share=0.2)
    with pytest.raises(ValueError, match="nonnegative"):
        _request(budget=-1.0)
    with pytest.raises(TypeError, match="bool"):
        _request(budget=True)
    with pytest.raises(TypeError, match="bool"):
        _request(random_seed=True)
    with pytest.raises(TypeError, match="bool"):
        _request(min_group_size=True, max_group_size=3)
    with pytest.raises(ValueError, match="must increase"):
        _request(history=history.iloc[::-1], selection_end=20)
    with pytest.raises(ValueError, match="must be unique"):
        duplicated = pd.concat([history, history])
        _request(history=duplicated, selection_end=20)
    with pytest.raises(TypeError, match="DataFrame"):
        _request(history={"a": [1.0]}, selection_end=0)
    with pytest.raises(ValueError, match="missing units"):
        _request(history=history.drop(columns="c"), selection_end=20)
    with pytest.raises(TypeError, match="numeric"):
        text = history.copy()
        text["a"] = ["x", "y"]
        _request(history=text, selection_end=20)
    with pytest.raises(ValueError, match="fixed_treated size"):
        _request(fixed_treated=("a",), min_group_size=2)
    with pytest.raises(ValueError, match="must contain required_treated"):
        _request(fixed_treated=("a",), required_treated=("b",))
    with pytest.raises(ValueError, match="cannot include a treated unit"):
        _request(fixed_treated=("a",), donor_pools=(("a", "b"),))
    with pytest.raises(ValueError, match="cannot include excluded_units"):
        _request(
            fixed_treated=("a",),
            excluded_units=("c",),
            donor_pools=(("b", "c"),),
        )
    with pytest.raises(ValueError, match="must be donor-eligible"):
        _request(
            fixed_treated=("a",),
            donor_eligible=("b",),
            donor_pools=(("c",),),
        )
    with pytest.raises(ValueError, match="absent from treatment_eligible"):
        _request(always_control=("a",), treatment_eligible=("a", "b"))
    with pytest.raises(ValueError, match="must be treatment-eligible"):
        _request(required_treated=("a",), treatment_eligible=("b",))
    with pytest.raises(ValueError, match="absent from donor_eligible"):
        _request(excluded_units=("a",), donor_eligible=("a",))
    with pytest.raises(ValueError, match="spillover_exclusions must be unique"):
        _request(spillover_exclusions=(("a", "b"), ("a", "b")))
    with pytest.raises(ValueError, match="unknown unit"):
        _request(spillover_exclusions=(("a", "z"),))
    with pytest.raises(TypeError, match="bool"):
        _request(unit_costs=(("a", True),))
    with pytest.raises(ValueError, match="unit_costs names must be unique"):
        _request(unit_costs=(("a", 1.0), ("a", 2.0)))
    with pytest.raises(ValueError, match="unit_covariates names must be unique"):
        _request(
            unit_covariates=(
                ("income", (("a", 1.0),)),
                ("income", (("b", 2.0),)),
            )
        )
    with pytest.raises(ValueError, match="max_abs_mean_gap must be nonnegative"):
        BalanceConstraint("income", -0.1)
    with pytest.raises(TypeError, match="BalanceConstraint"):
        _request(balance_constraints=(("income", 0.1),))
    with pytest.raises(ValueError, match="covariates must be unique"):
        _request(
            balance_constraints=(
                BalanceConstraint("income", 0.1),
                BalanceConstraint("income", 0.2),
            )
        )
    with pytest.raises(TypeError, match="positive integer"):
        _request(exact_ceiling=0)
    with pytest.raises(TypeError, match="positive integer"):
        _request(max_generated=0)
    with pytest.raises(TypeError, match="positive integer"):
        _request(shortlist_size=0)
    with pytest.raises(TypeError, match="positive integer"):
        _request(min_remaining_donors=0)
    with pytest.raises(ValueError, match="needs at least 1"):
        _request(units=())
    with pytest.raises(TypeError, match="sequence of unit groups"):
        _request(supplied_groups="ab")
    with pytest.raises(TypeError, match="single time index"):
        wide = history.copy()
        wide.index = pd.MultiIndex.from_arrays([history.index, history.index])
        _request(history=wide, selection_end=20)
    with pytest.raises(TypeError, match="must be an int"):
        _request(random_seed="no")
    with pytest.raises(TypeError, match="sequence of pairs"):
        _request(spillover_exclusions="ab")
    with pytest.raises(TypeError, match="pairs of unit names"):
        _request(spillover_exclusions=(("a",),))
    with pytest.raises(TypeError, match="sequence of"):
        _request(unit_costs="a")
    with pytest.raises(TypeError, match="must be"):
        _request(unit_costs=(("a",),))
    with pytest.raises(TypeError, match="sequence of covariate"):
        _request(unit_covariates="income")
    with pytest.raises(TypeError, match="entries must be"):
        _request(unit_covariates=(("income",),))
    with pytest.raises(TypeError, match="non-empty strings"):
        _request(unit_covariates=(("", (("a", 1.0),)),))
    with pytest.raises(TypeError, match="covariate values must be"):
        _request(unit_covariates=(("income", "a"),))
    with pytest.raises(TypeError, match="covariate values must be"):
        _request(unit_covariates=(("income", (("a",),)),))
    with pytest.raises(ValueError, match="units must be unique"):
        _request(unit_covariates=(("income", (("a", 1.0), ("a", 2.0))),))
    with pytest.raises(TypeError, match="non-empty strings"):
        labelled = history.copy()
        labelled.columns = [1, "b", "c"]
        _request(history=labelled, selection_end=20)
    with pytest.raises(ValueError, match="columns must be unique"):
        duplicated_columns = history.copy()
        duplicated_columns.columns = ["a", "a", "c"]
        _request(history=duplicated_columns, selection_end=20)
    with pytest.raises(TypeError, match="non-empty string"):
        BalanceConstraint("", 1.0)
    with pytest.raises(ValueError, match="must be treatment-eligible"):
        _request(fixed_treated=("a",), treatment_eligible=("b",))
    with pytest.raises(TypeError, match="sequence of BalanceConstraint"):
        _request(balance_constraints="income")
    with pytest.raises(ValueError, match="must be donor-eligible"):
        _request(always_control=("c",), donor_eligible=("b",))
    bounded = _request(min_outcome_share=0.2)
    assert bounded.history is None
    assert bounded.min_outcome_share == 0.2
    longer = pd.DataFrame(
        {"a": [1.0, 2.0, "later"], "b": [3.0, 4.0, 8.0], "c": [5.0, 6.0, 9.0]},
        index=pd.Index([10, 20, 30], name="time"),
    )
    window = _request(history=longer, selection_end=20)
    assert list(window.history.index) == [10, 20]
    longer.loc[10, "a"] = 999.0
    assert window.history.loc[10, "a"] == 1.0


def test_candidate_table_rejects_fits_and_partial_disagreement() -> None:
    accepted = _table()
    assert accepted.n_inference_fits == 0
    assert accepted.partial is False
    partial = _table(
        frame=pd.DataFrame(
            [
                _candidate_row(),
                _candidate_row(
                    treated=("b",),
                    donors=("a",),
                    filter_reasons=("group_size",),
                    shortlist_rank=pd.NA,
                    status="filtered",
                ),
            ]
        ),
        n_space=4,
        n_enumerated=2,
        n_filtered=1,
        n_shortlisted=1,
        partial=True,
        enumeration_policy="seeded_hash_sample",
    )
    assert partial.partial is True
    with pytest.raises(ValueError, match="n_inference_fits must be 0"):
        _table(n_inference_fits=1)
    with pytest.raises(ValueError, match="n_inference_fits must be 0"):
        _table(n_inference_fits=False)
    with pytest.raises(ValueError, match="partial must be true only"):
        _table(partial=True)
    with pytest.raises(ValueError, match="cannot exceed n_space"):
        _table(
            n_space=1,
            n_enumerated=2,
            partial=True,
            enumeration_policy="seeded_hash_sample",
        )
    with pytest.raises(ValueError, match="cannot exceed n_enumerated"):
        _table(n_space=1, n_enumerated=1, n_filtered=2)
    with pytest.raises(ValueError, match="unfiltered count"):
        _table(n_space=2, n_enumerated=2, n_filtered=1, n_shortlisted=2)
    with pytest.raises(TypeError, match="partial must be a bool"):
        _table(partial=1)
    with pytest.raises(ValueError, match="enumeration_policy"):
        _table(enumeration_policy="first_n")
    with pytest.raises(ValueError, match="partial table uses"):
        _table(
            n_space=2,
            n_enumerated=1,
            partial=True,
            enumeration_policy="exact",
        )
    with pytest.raises(ValueError, match="exact table is not a sample"):
        _table(enumeration_policy="seeded_hash_sample")
    with pytest.raises(ValueError, match="estimand must be DRAW_ALIGNED_UNIT_SUM"):
        _table(estimand="raw_sum_versus_simplex")
    with pytest.raises(ValueError, match="missing columns"):
        _table(frame=pd.DataFrame({"candidate_id": []}))
    with pytest.raises(TypeError, match="DataFrame"):
        _table(frame=[])
    bad_role = pd.DataFrame([_candidate_row(proxy_role="fitted_simplex")])
    with pytest.raises(ValueError, match="proxy_role"):
        _table(frame=bad_role, n_space=1, n_enumerated=1, n_shortlisted=1)
    bad_estimand = pd.DataFrame([_candidate_row(estimand="other")])
    with pytest.raises(ValueError, match="frame estimand"):
        _table(frame=bad_estimand, n_space=1, n_enumerated=1, n_shortlisted=1)
    with pytest.raises(ValueError, match="frame length must equal"):
        _table(n_space=5, n_enumerated=5, n_filtered=1, n_shortlisted=2)
    simplex = pd.DataFrame([_candidate_row(member_weight=0.9)])
    with pytest.raises(ValueError, match="member_weight must be 1/k"):
        _table(frame=simplex, n_space=1, n_enumerated=1, n_shortlisted=1)
    invented = pd.DataFrame(
        [_candidate_row(filter_reasons=("invented_reason",), shortlist_rank=pd.NA)]
    )
    with pytest.raises(ValueError, match="CANDIDATE_FILTER_REASONS"):
        _table(frame=invented, n_space=1, n_enumerated=1, n_filtered=1)
    ranked_filter = pd.DataFrame(
        [
            _candidate_row(filter_reasons=("group_size",), shortlist_rank=1),
            _candidate_row(
                treated=("b",),
                donors=("a",),
                filter_reasons=(),
                shortlist_rank=pd.NA,
                status="enumerated",
            ),
        ]
    )
    with pytest.raises(ValueError, match="cannot have a shortlist_rank"):
        _table(
            frame=ranked_filter,
            n_space=2,
            n_enumerated=2,
            n_filtered=1,
            n_shortlisted=1,
        )
    valid = pd.DataFrame([_candidate_row()])
    stored = _table(frame=valid, n_space=1, n_enumerated=1, n_shortlisted=1)
    assert stored.frame["candidate_id"].iloc[0] == '[["a"],["b"]]'
    assert stored.frame["member_weight"].iloc[0] == 1.0
    view = stored.frame
    view.loc[0, "member_weight"] = 0.2
    assert stored.frame["member_weight"].iloc[0] == 1.0
    with pytest.raises(TypeError, match="nonnegative integer"):
        _table(n_space=1.5)


def test_candidate_request_uses_the_exact_history_label() -> None:
    index = pd.to_datetime(["2020-06-01", "2020-07-01"])
    history = pd.DataFrame({"a": [1.0, 2.0]}, index=index)
    end = index[0]
    request = _request(units=("a",), history=history, selection_end=end)
    assert request.selection_end == end
    assert list(request.history.index) == [end]
    view = request.history
    view.loc[end, "a"] = 9.0
    assert request.history.loc[end, "a"] == 1.0
    leaked = request.history
    leaked.loc[pd.Timestamp("2020-08-01")] = 9.0
    assert list(request.history.index) == [end]
    integer = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=pd.Index([10, 20, 30]))
    stored_history = _request(units=("a",), history=integer, selection_end=20)
    appended = stored_history.history
    appended.loc[30] = 9.0
    assert list(stored_history.history.index) == [10, 20]
    missing_none = pd.DataFrame(index=pd.Index([10, 20, 30]))
    missing_none["a"] = pd.Series(
        [None, 2.0, 3.0], index=missing_none.index, dtype=object
    )
    with pytest.raises(TypeError, match="numeric"):
        _request(units=("a",), history=missing_none, selection_end=20)
    with pytest.raises(ValueError, match="not in the history index"):
        _request(units=("a",), history=history, selection_end="2020-06")
    missing = history.copy()
    missing.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _request(units=("a",), history=missing, selection_end=end)
    screened = _request(supplied_groups=(("a", "b"),), max_group_size=1)
    assert screened.supplied_groups == (("a", "b"),)
    control = _request(supplied_groups=(("a",),), always_control=("a",))
    assert control.supplied_groups == (("a",),)
    omitted = _request(supplied_groups=(("a",),), required_treated=("b",))
    assert omitted.supplied_groups == (("a",),)
    gap = _request(balance_constraints=(BalanceConstraint("income", 0.5),))
    assert gap.balance_constraints[0].covariate == "income"


def test_screen_contracts_compare_stored_frames_by_value() -> None:
    history = _selection_history()
    left = _request(history=history, selection_end=20)
    right = _request(history=history.copy(), selection_end=20)
    assert left == right
    assert left != _request()
    assert _table() == _table()
    with pytest.raises(TypeError):
        hash(left)
    with pytest.raises(TypeError):
        hash(_table())


def test_candidate_table_rejects_a_smuggled_or_inconsistent_row() -> None:
    ranked = pd.DataFrame(
        [
            _candidate_row(),
            _candidate_row(
                treated=("b",),
                donors=("a",),
                shortlist_rank=None,
                status="enumerated",
            ),
        ]
    )
    accepted = _table(frame=ranked, n_space=2, n_enumerated=2, n_shortlisted=1)
    assert accepted.frame["shortlist_rank"].iloc[0] == 1
    duplicate = pd.DataFrame(
        [
            _candidate_row(),
            _candidate_row(treated=("b",), donors=("a",), shortlist_rank=1),
        ]
    )
    with pytest.raises(ValueError, match="without duplicates"):
        _table(frame=duplicate, n_space=2, n_enumerated=2, n_shortlisted=2)
    extra = pd.DataFrame([_candidate_row()])
    extra["posterior"] = 1.0
    with pytest.raises(ValueError, match="outside CANDIDATE_COLUMNS"):
        _table(frame=extra, n_space=1, n_enumerated=1, n_shortlisted=1)
    with pytest.raises(ValueError, match="must be sorted"):
        _table(
            frame=pd.DataFrame([_candidate_row(treated=("b", "a"), member_weight=0.5)]),
            n_space=1,
            n_enumerated=1,
            n_shortlisted=1,
        )
    with pytest.raises(ValueError, match="disjoint"):
        _table(
            frame=pd.DataFrame([_candidate_row(donors=("a",))]),
            n_space=1,
            n_enumerated=1,
            n_shortlisted=1,
        )
    with pytest.raises(ValueError, match="candidate_id must be"):
        _table(
            frame=pd.DataFrame([_candidate_row(candidate_id="a")]),
            n_space=1,
            n_enumerated=1,
            n_shortlisted=1,
        )
    outside = _candidate_row(
        treated=("a", "b", "c", "d"),
        donors=(),
        n_donors=0,
        member_weight=0.25,
        filter_reasons=(),
        status="shortlisted",
    )
    with pytest.raises(ValueError, match="group_size"):
        _table(
            frame=pd.DataFrame([outside]), n_space=1, n_enumerated=1, n_shortlisted=1
        )
    filtered = outside | {
        "filter_reasons": ("group_size",),
        "shortlist_rank": pd.NA,
        "status": "filtered",
    }
    stored = _table(
        frame=pd.DataFrame([filtered]),
        n_space=1,
        n_enumerated=1,
        n_filtered=1,
        n_shortlisted=0,
    )
    assert stored.n_filtered == 1
    fitted = _candidate_row(
        proxy_role="preperiod_weighted_proportion", proxy_score=0.25
    )
    _table(frame=pd.DataFrame([fitted]), n_space=1, n_enumerated=1, n_shortlisted=1)
    with pytest.raises(ValueError, match="must be finite"):
        _table(
            frame=pd.DataFrame(
                [
                    _candidate_row(
                        proxy_role="preperiod_weighted_proportion",
                        proxy_score=pd.NA,
                    )
                ]
            ),
            n_space=1,
            n_enumerated=1,
            n_shortlisted=1,
        )
    disagreed = _candidate_row(
        filter_reasons=("group_size",),
        shortlist_rank=pd.NA,
        status="shortlisted",
        boundary=True,
    )
    with pytest.raises(ValueError, match="status must match"):
        _table(
            frame=pd.DataFrame([disagreed]),
            n_space=1,
            n_enumerated=1,
            n_filtered=1,
            n_shortlisted=0,
        )
    with pytest.raises(ValueError, match="n_donors must equal"):
        _table(
            frame=pd.DataFrame([_candidate_row(n_donors=5)]),
            n_space=1,
            n_enumerated=1,
            n_shortlisted=1,
        )
    with pytest.raises(ValueError, match="must be null"):
        _table(
            frame=pd.DataFrame(
                [_candidate_row(proxy_role="unavailable", proxy_score=0.2)]
            ),
            n_space=1,
            n_enumerated=1,
            n_shortlisted=1,
        )
    with pytest.raises(ValueError, match="identity must be unique"):
        _table(
            frame=pd.DataFrame(
                [
                    _candidate_row(),
                    _candidate_row(shortlist_rank=pd.NA, status="enumerated"),
                ]
            ),
            n_space=2,
            n_enumerated=2,
            n_shortlisted=1,
        )
    with pytest.raises(ValueError, match="outcome_share must be null or finite"):
        _table(
            frame=pd.DataFrame([_candidate_row(outcome_share="later")]),
            n_space=1,
            n_enumerated=1,
            n_shortlisted=1,
        )
