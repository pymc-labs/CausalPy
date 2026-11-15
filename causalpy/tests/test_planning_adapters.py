# Copyright 2022 - 2026 The PyMC Labs Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contract tests for the registered planning adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalpy.experiments.augmented_synthetic_control import AugmentedSyntheticControl
from causalpy.experiments.interactive_fixed_effects import (
    InteractiveFixedEffectsSyntheticControl,
)
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.planning import HistoryOnlyPlan
from causalpy.planning.adapters import (
    ADAPTERS,
    AUGMENTED_PRIORS,
    AUGMENTED_WEIGHT_POLICY,
    INTERACTIVE_FIXED_EFFECTS_PRIORS,
    INTERACTIVE_FIXED_EFFECTS_WEIGHT_POLICY,
    AugmentedSyntheticControlAdapter,
    analysis_handoff,
    capability_matrix,
)
from causalpy.planning.contracts import (
    REFERENCE_CAPABILITIES,
    DecisionRule,
    WindowRequest,
    aggregate_unit_impacts,
)
from causalpy.pymc_models import WeightedSumFitter

LIGHT = {
    "draws": 8,
    "tune": 8,
    "chains": 1,
    "cores": 1,
    "progressbar": False,
    "compute_convergence_checks": False,
    "random_seed": 11,
}
DURATION = 2
DONORS = ("d1", "d2", "d3")
TREATED = ("t1", "t2", "t3")
METHODS = (
    "synthetic_control",
    "augmented_synthetic_control",
    "interactive_fixed_effects",
)
REGISTERED = ("augmented_synthetic_control", "interactive_fixed_effects")


def _decision(direction: str = "positive") -> DecisionRule:
    return DecisionRule(direction, "cumulative_total", 0.0, 0.8)


def _panel(rows: int = 8) -> pd.DataFrame:
    time = np.arange(rows, dtype=float)
    factor = np.linspace(0.0, 1.0, rows)
    columns = {
        "d1": 10.0 + factor + 0.1 * time,
        "d2": 12.0 + 0.5 * factor + 0.1 * time,
        "d3": 8.0 + 1.5 * factor + 0.05 * time,
        "t1": 11.0 + factor + 0.1 * time,
        "t2": 20.0 + 0.8 * factor + 0.1 * time,
        "t3": 9.0 + 1.2 * factor + 0.1 * time,
    }
    return pd.DataFrame(columns, index=pd.RangeIndex(rows, name="time"))


def _plan(
    estimator: str,
    treated: tuple[str, ...] = ("t1",),
    *,
    history: pd.DataFrame | None = None,
    sample_kwargs: dict[str, object] | None = LIGHT,
    direction: str = "positive",
) -> HistoryOnlyPlan:
    return HistoryOnlyPlan(
        _panel() if history is None else history,
        treated,
        DONORS,
        _decision(direction),
        estimator=estimator,
        sample_kwargs=sample_kwargs,
        outcome_scale="outcome",
    )


def _native(estimator: str, frame: pd.DataFrame, treated: tuple[str, ...]):
    treatment = frame.index[-DURATION]
    if estimator == "synthetic_control":
        return SyntheticControl(
            frame,
            treatment,
            control_units=list(DONORS),
            treated_units=list(treated),
            model=WeightedSumFitter(sample_kwargs=dict(LIGHT)),
        ).fit()
    if estimator == "augmented_synthetic_control":
        return AugmentedSyntheticControl(
            frame,
            treatment,
            DONORS,
            treated,
        ).fit(sample_kwargs=dict(LIGHT))
    frame = frame.copy()
    frame.index = frame.index.rename(None)
    return InteractiveFixedEffectsSyntheticControl(
        frame,
        treatment,
        DONORS,
        treated,
        max_factors=4,
    ).fit(**dict(LIGHT))


def _native_impact(estimator: str, experiment) -> np.ndarray:
    if estimator == "augmented_synthetic_control":
        impact = experiment.idata.posterior["treatment_effect"]
    else:
        impact = experiment.result.impact_post
    if "post_time" in impact.dims:
        impact = impact.rename({"post_time": "obs_ind"})
    return np.asarray(
        impact.transpose("chain", "draw", "obs_ind", "treated_units").values
    )


def _counterfactual(evaluation) -> np.ndarray:
    return np.asarray(evaluation.draws.counterfactual.values)


def _forbid_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("analysis construction started")

    monkeypatch.setattr(AugmentedSyntheticControl, "__init__", boom)
    monkeypatch.setattr(InteractiveFixedEffectsSyntheticControl, "__init__", boom)
    monkeypatch.setattr(SyntheticControl, "__init__", boom)


def test_capability_matrix_matches_adapter_behavior() -> None:
    matrix = capability_matrix()
    assert tuple(row.estimator for row in matrix) == METHODS
    assert matrix[0] == REFERENCE_CAPABILITIES
    by_name = {row.estimator: row for row in matrix}
    assert by_name["synthetic_control"].group_support is False
    assert by_name["synthetic_control"].max_group_size == 1
    for name in REGISTERED:
        row = by_name[name]
        assert row.group_support is True
        assert row.max_group_size == 3
        assert row.deterministic_effect_reuse is True
        assert row.horizon_reuse is False
        assert row.post_period_donor_data is True
        assert row.allowed_decision_rules == ("positive", "negative")
        assert ADAPTERS[name](model=None, sample_kwargs=LIGHT).capabilities() == row
    assert "synthetic_control" not in ADAPTERS
    assert set(ADAPTERS) == set(REGISTERED)


def test_registered_names_construct_and_plan_handoff_stays_on_vanilla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_construction(monkeypatch)
    for name in REGISTERED:
        plan = _plan(name)
        assert plan.capabilities.estimator == name
        with pytest.raises(ValueError, match="own analysis path"):
            plan.analysis_handoff(duration=DURATION)
    with pytest.raises(ValueError, match="Unknown estimator"):
        _plan("not_an_estimator")


def test_frozen_specification_preserves_method_and_refuses_a_supplied_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_construction(monkeypatch)
    frame = _panel()
    augmented = analysis_handoff(
        frame,
        ("t1", "t2"),
        DONORS,
        _decision(),
        estimator="augmented_synthetic_control",
        duration=DURATION,
        sample_kwargs=LIGHT,
        outcome_scale="outcome",
    )
    assert augmented.estimator == "augmented_synthetic_control"
    assert augmented.model_class == "AugmentedSyntheticControl"
    assert augmented.weight_policy == AUGMENTED_WEIGHT_POLICY
    assert augmented.priors == AUGMENTED_PRIORS
    assert augmented.estimand == "draw_aligned_unit_impact_sum"
    assert augmented.training_policy == "separate-fits"
    assert augmented.decision == _decision()
    assert augmented.max_factors is None
    assert augmented.treated_units == ("t1", "t2")
    assert dict(augmented.sample_kwargs)["draws"] == LIGHT["draws"]
    fixed = analysis_handoff(
        frame,
        "t1",
        DONORS,
        _decision("negative"),
        estimator="interactive_fixed_effects",
        duration=DURATION,
        sample_kwargs=LIGHT,
        max_factors=2,
    )
    assert fixed.priors == INTERACTIVE_FIXED_EFFECTS_PRIORS
    assert fixed.weight_policy == INTERACTIVE_FIXED_EFFECTS_WEIGHT_POLICY
    assert fixed.max_factors == 2
    assert fixed.decision.direction == "negative"
    with pytest.raises(TypeError, match="not model"):
        analysis_handoff(
            frame,
            "t1",
            DONORS,
            _decision(),
            estimator="augmented_synthetic_control",
            duration=DURATION,
            model=object(),
            sample_kwargs=LIGHT,
        )
    with pytest.raises(TypeError, match="max_factors"):
        analysis_handoff(
            frame,
            "t1",
            DONORS,
            _decision(),
            estimator="augmented_synthetic_control",
            duration=DURATION,
            sample_kwargs=LIGHT,
            max_factors=2,
        )
    with pytest.raises(ValueError, match="HistoryOnlyPlan.analysis_handoff"):
        analysis_handoff(
            frame,
            "t1",
            DONORS,
            _decision(),
            estimator="synthetic_control",
            duration=DURATION,
            sample_kwargs=LIGHT,
        )


def test_unsupported_inputs_fail_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_construction(monkeypatch)
    frame = _panel()
    with pytest.raises(TypeError, match="adapter construction failed"):
        _plan("augmented_synthetic_control", sample_kwargs={"tune": 8})
    with pytest.raises(TypeError, match="adapter construction failed"):
        _plan("interactive_fixed_effects", sample_kwargs=None)
    with pytest.raises(TypeError, match="not model"):
        HistoryOnlyPlan(
            frame,
            "t1",
            DONORS,
            _decision(),
            estimator="augmented_synthetic_control",
            model=object(),
        )
    with pytest.raises(ValueError, match="at most 3"):
        _plan("augmented_synthetic_control", ("t1", "t2", "t3", "extra"))
    adapter = AugmentedSyntheticControlAdapter(model=None, sample_kwargs=LIGHT)
    with pytest.raises(ValueError, match="at most 3"):
        adapter.fit_window(
            WindowRequest(
                history=frame,
                treated_units=("t1", "t2", "t3", "extra"),
                control_units=DONORS,
                treatment_time=frame.index[-DURATION],
                duration=DURATION,
                outcome_scale="outcome",
            )
        )
    extra = frame.copy()
    extra.loc[int(frame.index[-1]) + 1] = frame.iloc[-1]
    with pytest.raises(ValueError, match="refused, not sliced"):
        adapter.fit_window(
            WindowRequest(
                history=extra,
                treated_units=("t1",),
                control_units=DONORS,
                treatment_time=frame.index[-DURATION],
                duration=DURATION,
                outcome_scale="outcome",
            )
        )


@pytest.mark.parametrize("estimator", METHODS)
def test_contract_journey(estimator: str, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _panel()
    seen: list[int | None] = []
    if estimator == "interactive_fixed_effects":
        real_init = InteractiveFixedEffectsSyntheticControl.__init__

        def spy(self, *args, **kwargs):
            seen.append(kwargs.get("max_factors"))
            return real_init(self, *args, **kwargs)

        monkeypatch.setattr(InteractiveFixedEffectsSyntheticControl, "__init__", spy)
    result = _plan(estimator, history=frame).evaluate(
        DURATION,
        absolute_effects=(0.0, 4.0),
        relative_effects=(0.25,),
        relative_baseline=100.0,
    )
    assert result.n_inference_fits == 1
    assert len(result.evaluations) == 3
    assert {row.fit_index for row in result.evaluations} == {0}
    zero = result.evaluations[0]
    shifted = result.evaluations[1]
    relative = result.evaluations[2]
    assert zero.estimator == estimator
    assert zero.provenance.estimator == estimator
    assert zero.provenance.translation == "deterministic_additive"
    assert zero.provenance.uncertainty == "posterior_conditional_mean"
    assert zero.provenance.horizon_policy == "separate-fits"
    assert zero.boundary.post_period_treated_in_fit is False
    assert zero.boundary.post_period_donors_in_prediction is True
    assert zero.boundary.horizon_policy == "separate-fits"
    assert tuple(zero.draws.unit_impact.coords["treated_units"].values) == ("t1",)
    assert zero.diagnostics.diagnostics_status in {
        "available",
        "partial",
        "unavailable",
    }
    if estimator == "interactive_fixed_effects":
        assert seen == [4]
        assert zero.diagnostics.diagnostics_status == "unavailable"
        assert zero.diagnostics.divergences is None
        assert zero.diagnostics.diagnostics_reason
    else:
        assert isinstance(zero.diagnostics.divergences, int)
    np.testing.assert_allclose(
        shifted.draws.cumulative_total - zero.draws.cumulative_total,
        4.0 * DURATION,
    )
    np.testing.assert_allclose(
        relative.draws.cumulative_total - zero.draws.cumulative_total,
        0.25 * 100.0,
    )
    np.testing.assert_allclose(
        relative.draws.relative_total,
        relative.draws.cumulative_total / 100.0,
    )
    native = _native(estimator, frame, ("t1",))
    np.testing.assert_allclose(
        zero.draws.unit_impact.values,
        _native_impact(estimator, native),
    )


@pytest.mark.parametrize("estimator", METHODS)
def test_separate_horizon_fits(estimator: str) -> None:
    result = _plan(estimator).evaluate((1, DURATION), absolute_effects=(0.0,))
    assert result.n_inference_fits == 2
    short, long = result.evaluations
    assert short.duration == 1
    assert long.duration == DURATION
    assert len(short.boundary.training_index) == len(long.boundary.training_index) + 1
    assert short.provenance.sample_kwargs == long.provenance.sample_kwargs
    assert short.boundary.training_index[-1] != long.boundary.training_index[-1]


@pytest.mark.parametrize("estimator", METHODS)
def test_held_out_treated_and_donor_boundary(estimator: str) -> None:
    frame = _panel()
    base = _plan(estimator, history=frame).evaluate(DURATION, absolute_effects=(0.0,))
    treated = frame.copy()
    treated.loc[treated.index[-DURATION] :, "t1"] += 25.0
    held_out = _plan(estimator, history=treated).evaluate(
        DURATION,
        absolute_effects=(0.0,),
    )
    donors = frame.copy()
    donors.loc[donors.index[-DURATION] :, list(DONORS)] += 25.0
    in_window = _plan(estimator, history=donors).evaluate(
        DURATION,
        absolute_effects=(0.0,),
    )
    np.testing.assert_allclose(
        _counterfactual(base.evaluations[0]),
        _counterfactual(held_out.evaluations[0]),
    )
    assert not np.allclose(
        _counterfactual(base.evaluations[0]),
        _counterfactual(in_window.evaluations[0]),
        atol=1.0,
    )
    np.testing.assert_allclose(
        held_out.evaluations[0].draws.unit_impact
        - base.evaluations[0].draws.unit_impact,
        25.0,
    )


@pytest.mark.parametrize("estimator", REGISTERED)
def test_group_draw_aligned_sum(estimator: str) -> None:
    frame = _panel()
    treated = ("t1", "t2")
    result = _plan(estimator, treated, history=frame).evaluate(
        DURATION,
        absolute_effects=(0.0, 4.0),
        relative_effects=(0.5,),
        relative_baseline=80.0,
    )
    assert result.n_inference_fits == 1
    zero = result.evaluations[0]
    total, per_period, per_unit = aggregate_unit_impacts(zero.draws.unit_impact)
    np.testing.assert_allclose(zero.draws.cumulative_total, total)
    np.testing.assert_allclose(zero.draws.average_per_period, per_period)
    np.testing.assert_allclose(zero.draws.average_per_treated_unit, per_unit)
    np.testing.assert_allclose(
        result.evaluations[1].draws.cumulative_total - zero.draws.cumulative_total,
        4.0 * DURATION,
    )
    np.testing.assert_allclose(
        result.evaluations[2].draws.relative_total,
        result.evaluations[2].draws.cumulative_total / 80.0,
    )
    reversed_units = zero.draws.unit_impact.sel(treated_units=["t2", "t1"])
    reversed_total, _, _ = aggregate_unit_impacts(
        reversed_units.transpose("chain", "draw", "obs_ind", "treated_units")
    )
    np.testing.assert_allclose(reversed_total, total)
    assert tuple(zero.draws.unit_impact.coords["treated_units"].values) == treated


@pytest.mark.parametrize("estimator", REGISTERED)
def test_handoff_run_matches_plan_and_drops_outside_rows(estimator: str) -> None:
    frame = _panel()
    plan = _plan(estimator, ("t1", "t2"), history=frame)
    planned = plan.evaluate(DURATION, absolute_effects=(0.0,)).evaluations[0]
    handoff = analysis_handoff(
        frame,
        ("t1", "t2"),
        DONORS,
        _decision(),
        estimator=estimator,
        duration=DURATION,
        sample_kwargs=LIGHT,
        outcome_scale="outcome",
    )
    observed = handoff.run(frame)
    assert observed.provenance.translation == "observed"
    assert observed.boundary.rows_outside_window_excluded is False
    np.testing.assert_allclose(
        observed.draws.unit_impact.values,
        planned.draws.unit_impact.values,
    )
    extended = frame.copy()
    extended.loc[int(frame.index[-1]) + 1] = frame.iloc[-1] + 40.0
    dropped = handoff.run(extended)
    assert dropped.boundary.rows_outside_window_excluded is True
    np.testing.assert_allclose(
        dropped.draws.counterfactual.values,
        observed.draws.counterfactual.values,
    )


@pytest.mark.nightly
@pytest.mark.parametrize("estimator", REGISTERED)
@pytest.mark.parametrize("size", (1, 2, 3))
def test_group_sizes_match_native_analysis(estimator: str, size: int) -> None:
    frame = _panel()
    treated = TREATED[:size]
    result = _plan(estimator, treated, history=frame).evaluate(
        DURATION,
        absolute_effects=(0.0,),
    )
    row = result.evaluations[0]
    assert result.n_inference_fits == 1
    assert tuple(row.draws.unit_impact.coords["treated_units"].values) == treated
    total, _, _ = aggregate_unit_impacts(row.draws.unit_impact)
    np.testing.assert_allclose(row.draws.cumulative_total, total)
    native = _native(estimator, frame, treated)
    np.testing.assert_allclose(
        row.draws.unit_impact.values,
        _native_impact(estimator, native),
    )


@pytest.mark.nightly
@pytest.mark.parametrize("estimator", REGISTERED)
def test_group_shock_order_and_heterogeneous_response(estimator: str) -> None:
    frame = _panel()
    treated = ("t1", "t2", "t3")
    base = (
        _plan(estimator, treated, history=frame)
        .evaluate(DURATION, absolute_effects=(0.0,))
        .evaluations[0]
    )
    shocked = frame.copy()
    shocked.loc[shocked.index[-DURATION] :, list(treated)] += 7.0
    common = (
        _plan(estimator, treated, history=shocked)
        .evaluate(DURATION, absolute_effects=(0.0,))
        .evaluations[0]
    )
    np.testing.assert_allclose(_counterfactual(base), _counterfactual(common))
    np.testing.assert_allclose(
        common.draws.cumulative_total - base.draws.cumulative_total,
        7.0 * DURATION * len(treated),
    )
    mixed = frame.copy()
    shifts = {"t1": 3.0, "t2": -2.0, "t3": 5.0}
    for name, shift in shifts.items():
        mixed.loc[mixed.index[-DURATION] :, name] += shift
    hetero = (
        _plan(estimator, treated, history=mixed)
        .evaluate(DURATION, absolute_effects=(0.0,))
        .evaluations[0]
    )
    np.testing.assert_allclose(_counterfactual(base), _counterfactual(hetero))
    np.testing.assert_allclose(
        hetero.draws.cumulative_total - base.draws.cumulative_total,
        sum(shifts.values()) * DURATION,
    )
    unit_shift = hetero.draws.unit_impact - base.draws.unit_impact
    for name, shift in shifts.items():
        np.testing.assert_allclose(unit_shift.sel(treated_units=name), shift)
    assert tuple(base.draws.unit_impact.coords["treated_units"].values) == treated


def test_negative_decision_is_allowed_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_construction(monkeypatch)
    plan = _plan("augmented_synthetic_control", direction="negative")
    assert plan.capabilities.allowed_decision_rules == ("positive", "negative")
