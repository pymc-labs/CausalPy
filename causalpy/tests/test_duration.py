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
"""Contract tests for common-origin duration comparison.

The trailing-tail evaluator stays covered by ``test_sc_design.py``. These
tests lock the separate comparison: shared training prefixes, one fit per
eligible origin and duration, refusals before sampling, and no duration
selection.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from causalpy.design import DecisionRule, SyntheticControlPlanner
from causalpy.duration import DurationWindow
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.historical_validation import WindowReplay
from causalpy.pymc_models import WeightedSumFitter

LIGHT = {
    "tune": 10,
    "draws": 10,
    "chains": 1,
    "cores": 1,
    "progressbar": False,
    "random_seed": 11,
}
CONTROLS = ("c0", "c1", "c2")
RULE = DecisionRule("positive", "cumulative", 0.0, 0.95)


def _frame() -> pd.DataFrame:
    index = pd.Index(range(8), name="t")
    c0 = np.array([0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0])
    c1 = np.array([0.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0])
    c2 = np.array([0.0, 0.0, 0.0, 8.0, 0.0, 8.0, 8.0, 8.0])
    treated = 0.5 * c0 + 0.25 * c1 + 0.25 * c2
    return pd.DataFrame({"c0": c0, "c1": c1, "c2": c2, "treated": treated}, index=index)


def _planner(**kwargs) -> SyntheticControlPlanner:
    model = kwargs.pop("model", WeightedSumFitter(sample_kwargs={}))
    return SyntheticControlPlanner(
        _frame(),
        "treated",
        CONTROLS,
        RULE,
        model=model,
        random_seed=7,
        **kwargs,
    )


def _preview(planner: SyntheticControlPlanner, **overrides):
    kwargs = {
        "audit_origins": (),
        "absolute_lifts": (1.0,),
        "target_lift": 0.0,
        "fit_cost": 2.0,
    }
    kwargs.update(overrides)
    return planner.preview_durations((2, 3), (2, 4, 6), **kwargs)


def _reject_fit(*args, **kwargs):
    raise AssertionError("a refused comparison reached a fit")


def test_invalid_duration_inputs_fail_before_fitting(monkeypatch):
    monkeypatch.setattr(SyntheticControl, "fit", _reject_fit)
    planner = _planner()
    with pytest.raises(ValueError, match="at least 1"):
        planner.preview_durations((0, 2), (2,), absolute_lifts=(1.0,), target_lift=0.0)
    with pytest.raises(ValueError, match="unique"):
        planner.preview_durations((2, 2), (2,), absolute_lifts=(1.0,), target_lift=0.0)
    with pytest.raises(TypeError, match="Python int"):
        planner.preview_durations((2.0,), (2,), absolute_lifts=(1.0,), target_lift=0.0)
    with pytest.raises(ValueError, match="not in the requested lift grid"):
        planner.preview_durations((2,), (2,), absolute_lifts=(1.0,), target_lift=0.5)
    with pytest.raises(ValueError, match="both"):
        planner.preview_durations(
            (2,),
            (2,),
            absolute_lifts=(0.0, 1.0),
            relative_lifts=(0.0, 1.0),
            target_lift=1.0,
        )
    with pytest.raises(TypeError, match="bool"):
        planner.preview_durations(
            (2,), (2,), absolute_lifts=(1.0,), target_lift=0.0, max_fits=True
        )
    with pytest.raises(ValueError, match="fit_cost"):
        planner.preview_durations(
            (2,), (2,), absolute_lifts=(1.0,), target_lift=0.0, fit_cost=-1.0
        )


def test_zero_target_lift_is_the_paired_null():
    preview = _planner().preview_durations(
        (2,), (2,), absolute_lifts=(1.0,), target_lift=0.0
    )
    assert preview.target_lift == 0.0
    assert preview.target_scale == "absolute"
    assert preview.status == "ready"


def test_common_origins_share_a_training_prefix_and_drop_partial_origins():
    preview = _preview(_planner())
    assert preview.status == "ready"
    assert preview.eligible_origins == (2, 4)
    assert [report.origin for report in preview.ineligible] == [6]
    assert preview.ineligible[0].eligible is False
    assert "duration 3" in preview.ineligible[0].reason
    by_origin: dict[int, set[tuple[int, ...]]] = {}
    for window in preview.windows:
        if window.role != "selection" or window.origin not in (2, 4):
            continue
        by_origin.setdefault(window.origin, set()).add(tuple(window.training_index))
    assert by_origin[2] == {(0, 1)}
    assert by_origin[4] == {(0, 1, 2, 3)}
    assert preview.expected_counterfactual_fits == 4
    assert preview.learned_null_upper_bound == 2
    assert preview.expected_fits == 6
    assert preview.expected_cost == 12.0


def test_budget_and_memory_refusals_do_not_fit_or_reduce_settings(monkeypatch):
    monkeypatch.setattr(SyntheticControl, "fit", _reject_fit)
    planner = _planner()
    supplied = {"draws": 20, "chains": 1, "progressbar": True}
    budget = planner.compare_durations(
        (2, 3),
        (2, 4, 6),
        audit_origins=(),
        absolute_lifts=(1.0,),
        target_lift=0.0,
        max_fits=5,
        sample_kwargs=supplied,
    )
    assert budget.status == "budget-exhausted"
    assert budget.n_inference_fits == 0
    assert budget.evidence_level is None
    assert budget.sample_kwargs["draws"] == 20
    assert budget.sample_kwargs["chains"] == 1
    assert supplied == {"draws": 20, "chains": 1, "progressbar": True}
    assert budget.recommendation().startswith("No supported recommendation.")
    with pytest.raises(ValueError, match="budget-exhausted"):
        budget.freeze(2)

    memory = planner.preview_durations(
        (2, 3),
        (2, 4),
        absolute_lifts=(1.0,),
        target_lift=0.0,
        sample_kwargs={"draws": 200_000, "chains": 4},
    )
    assert memory.status == "memory-exhausted"
    assert memory.estimated_window_bytes is not None
    assert memory.estimated_window_bytes > memory.max_window_bytes


def test_missing_draws_refuse_before_the_heavy_default(monkeypatch):
    monkeypatch.setattr(SyntheticControl, "fit", _reject_fit)
    with pytest.raises(ValueError, match="draws"):
        _planner().compare_durations((2,), (2,), absolute_lifts=(1.0,), target_lift=0.0)


def test_relative_baseline_failure_raises_before_fitting(monkeypatch):
    monkeypatch.setattr(SyntheticControl, "fit", _reject_fit)
    frame = _frame()
    frame.loc[3, "treated"] = 0.0
    planner = SyntheticControlPlanner(
        frame, "treated", CONTROLS, RULE, model=WeightedSumFitter(sample_kwargs={})
    )
    with pytest.raises(ValueError, match="Relative effects are undefined"):
        planner.compare_durations(
            (2,),
            (2,),
            relative_lifts=(0.1,),
            target_lift=0.0,
            sample_kwargs={"draws": 10, "chains": 1},
        )


def test_one_translation_per_window_and_freeze_audits_only_that_duration(monkeypatch):
    calls: list[tuple[object, int, str]] = []
    shapes: list[tuple[int, ...]] = []

    def fake_replay(*args, **kwargs):
        window = args[3]
        calls.append((window.origin, window.horizon, window.role))
        on_scored = kwargs.get("on_scored")
        if on_scored is not None:
            on_scored(
                SimpleNamespace(draws=np.zeros((1, 2, window.horizon))),
                window,
            )
        return (
            WindowReplay(
                plan=window,
                status="scored",
                reason=None,
                bias_cumulative=0.2,
                bias_average=0.1,
                cumulative_mean=0.2,
                average_mean=0.1,
                n_inference_fits=1,
                diagnostics_status="unavailable",
            ),
            1,
        )

    def fake_translate(experiment, planner, prepared, window):
        shapes.append(tuple(experiment.draws.shape))
        return [
            DurationWindow(
                origin=window.origin,
                duration=window.horizon,
                status="scored",
                reason=None,
                training_index=tuple(window.training_index),
                holdout_index=tuple(window.holdout_index),
                bias_cumulative=None,
                bias_average=None,
                estimand="cumulative",
                mean=1.0,
                hdi_lower=0.0,
                hdi_upper=2.0,
                truth=1.0,
                truth_covers=True,
                detected=True,
                tail_probability=0.9,
            )
        ]

    monkeypatch.setattr("causalpy.historical_validation._replay_window", fake_replay)
    monkeypatch.setattr("causalpy.duration._translate_window", fake_translate)
    planner = _planner()
    original = dict(planner.model.sample_kwargs)
    comparison = planner.compare_durations(
        (2, 3),
        (2,),
        audit_origins=(6,),
        absolute_lifts=(1.0,),
        target_lift=1.0,
        sample_kwargs={"draws": 10, "chains": 1},
    )
    assert planner.model.sample_kwargs == original
    assert sorted(calls) == [(2, 2, "selection"), (2, 3, "selection")]
    assert shapes == [(1, 2, 2), (1, 2, 3)]
    assert comparison.n_inference_fits == 2
    assert comparison.evidence_level == "historical-descriptive"
    assert comparison.rows[0].recommendation == "not-supported"
    assert "No supported recommendation." in comparison.recommendation()
    assert set(comparison.summary()["duration"]) == {2, 3}
    assert "origin" in comparison.detail().columns
    training = {
        (window.origin, window.duration): window.training_index
        for window in comparison.windows
        if window.estimand == "cumulative"
    }
    assert training[(2, 2)] == training[(2, 3)] == (0, 1)
    assert comparison.plot()[0] is not None

    spec = comparison.freeze(2)
    assert spec.selected_duration == 2
    assert spec.audit_origins == (6,)
    assert spec.template.sample_kwargs["draws"] == 10
    assert spec != comparison.freeze(2)
    calls.clear()
    audit = spec.audit(planner.history)
    assert audit.selected_duration == 2
    assert calls == [(6, 2, "selection")]
    assert "No supported recommendation." in audit.recommendation()
    failed = spec.audit(planner.history.iloc[:4])
    assert failed.status == "inconclusive"
    assert failed.selected_duration == 2


def test_execute_uses_the_stored_units_without_resampling_the_comparison(monkeypatch):
    seen: list[tuple[str, tuple[str, ...], object]] = []

    def fake_fit(self, *args, **kwargs):
        seen.append(
            (self.treated_units[0], tuple(self.control_units), self.treatment_time)
        )
        return self

    monkeypatch.setattr(SyntheticControl, "fit", fake_fit)
    monkeypatch.setattr(
        "causalpy.historical_validation._replay_window",
        lambda *args, **kwargs: (
            WindowReplay(
                plan=args[3],
                status="scored",
                reason=None,
                bias_cumulative=0.0,
                bias_average=0.0,
                n_inference_fits=1,
                diagnostics_status="unavailable",
            ),
            1,
        ),
    )
    monkeypatch.setattr(
        "causalpy.duration._translate_window",
        lambda experiment, planner, prepared, window: [
            DurationWindow(
                origin=window.origin,
                duration=window.horizon,
                status="scored",
                reason=None,
                training_index=tuple(window.training_index),
                holdout_index=tuple(window.holdout_index),
                bias_cumulative=0.0,
                bias_average=0.0,
                estimand="cumulative",
                mean=0.0,
                hdi_lower=-1.0,
                hdi_upper=1.0,
                truth=0.0,
                truth_covers=True,
                detected=False,
            )
        ],
    )
    planner = _planner()
    comparison = planner.compare_durations(
        (2,),
        (2, 4),
        absolute_lifts=(1.0,),
        target_lift=0.0,
        sample_kwargs={"draws": 8, "chains": 1},
    )
    seen.clear()
    fitted = comparison.freeze(2).execute(planner.history, 4)
    assert seen == [("treated", CONTROLS, 4)]
    assert fitted.model.sample_kwargs["draws"] == 8


def test_shared_engine_fits_each_eligible_window_once(monkeypatch):
    calls: list[object] = []
    real = SyntheticControl.fit

    def wrapped(self, *args, **kwargs):
        calls.append(self.treatment_time)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(SyntheticControl, "fit", wrapped)
    planner = _planner()
    comparison = planner.compare_durations(
        (2, 3),
        (2, 4),
        absolute_lifts=(1.0,),
        target_lift=0.0,
        max_fits=8,
        sample_kwargs=LIGHT,
    )
    assert sorted(calls) == [2, 2, 4, 4]
    assert comparison.status == "historical-descriptive"
    assert comparison.evidence_level == "historical-descriptive"
    assert comparison.n_inference_fits >= 4
    assert all(row.recommendation == "not-supported" for row in comparison.rows)
    assert comparison.summary()["zero_lift_bias_cumulative"].notna().all()
