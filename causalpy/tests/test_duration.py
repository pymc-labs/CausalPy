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

import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import causalpy.duration as duration
from causalpy.custom_exceptions import DataException
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
    assert "This is a historical description," in audit.recommendation()
    failed = spec.audit(planner.history.iloc[:4])
    assert failed.status == "inconclusive"
    assert failed.selected_duration == 2
    failed_text = failed.recommendation()
    assert "Status: inconclusive." in failed_text
    assert "This is a historical description," not in failed_text


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


def test_fit_failed_score_is_not_kept_as_a_detection(monkeypatch):
    translated = []

    def fake_fit(self, *args, **kwargs):
        return self

    def fake_score(experiment, window, decision):
        return WindowReplay(
            plan=window,
            status="fit-failed",
            reason=(
                "non-finite impact; a missing period was not dropped from the score."
            ),
            n_inference_fits=1,
        )

    def fake_translate(experiment, planner, prepared, window):
        translated.append(window.origin)
        return [
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
                mean=1.0,
                hdi_lower=0.5,
                hdi_upper=1.5,
                truth=0.0,
                truth_covers=False,
                detected=True,
                tail_probability=1.0,
            )
        ]

    monkeypatch.setattr(SyntheticControl, "fit", fake_fit)
    monkeypatch.setattr("causalpy.historical_validation._score_window", fake_score)
    monkeypatch.setattr("causalpy.duration._translate_window", fake_translate)
    monkeypatch.setattr(
        "causalpy.historical_validation._calibration_from_windows",
        lambda *args, **kwargs: (SimpleNamespace(evidence_level="inapplicable"), 0),
    )
    comparison = _planner().compare_durations(
        (2,),
        (2,),
        absolute_lifts=(1.0,),
        target_lift=0.0,
        sample_kwargs=LIGHT,
    )
    summary = comparison.summary()
    detail = comparison.detail()
    assert translated == []
    assert int(summary["failed_count"].sum()) == 1
    assert int(summary["historical_detection_count"].fillna(0).sum()) == 0
    assert set(detail["status"]) == {"fit-failed"}
    assert not detail["detected"].fillna(False).any()
    failed_text = comparison.recommendation()
    assert "Status: fit-failed." in failed_text
    assert "This is a historical description," not in failed_text


def test_partial_audit_is_not_a_complete_historical_description(monkeypatch):
    def fake_replay(*args, **kwargs):
        window = args[3]
        return (
            WindowReplay(
                plan=window,
                status="scored",
                reason=None,
                bias_cumulative=0.0,
                bias_average=0.0,
                n_inference_fits=1,
            ),
            1,
        )

    monkeypatch.setattr("causalpy.historical_validation._replay_window", fake_replay)
    planner = _planner()
    comparison = planner.compare_durations(
        (2,),
        (2,),
        audit_origins=(4, 6),
        exclusions=(5,),
        absolute_lifts=(1.0,),
        target_lift=0.0,
        sample_kwargs=LIGHT,
    )
    audit = comparison.freeze(2).audit(planner.history)
    assert audit.selected_duration == 2
    assert audit.requested_count == 2
    assert audit.scored_count == 1
    assert audit.failed_count == 0
    assert audit.status == "inconclusive"
    assert audit.recommendation().startswith("No supported recommendation")
    partial_text = audit.recommendation()
    assert "Status: inconclusive." in partial_text
    assert "This is a historical description," not in partial_text


def test_execute_drops_excluded_labels_and_refuses_the_excluded_split(monkeypatch):
    captured: list[tuple[list[int], object]] = []

    class _Recorder:
        def __init__(self, data, treatment_time, **kwargs):
            captured.append((list(data.index), treatment_time))

        def fit(self):
            return self

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
            ),
            1,
        ),
    )
    planner = _planner()
    spec = planner.compare_durations(
        (2,),
        (4,),
        exclusions=(1, 6),
        absolute_lifts=(1.0,),
        target_lift=0.0,
        sample_kwargs={"draws": 8, "chains": 1},
    ).freeze(2)
    monkeypatch.setattr(
        "causalpy.experiments.synthetic_control.SyntheticControl",
        _Recorder,
    )
    spec.execute(planner.history, 4)
    assert captured == [([0, 2, 3, 4, 5], 4)]
    with pytest.raises(ValueError, match="treatment_time"):
        spec.execute(planner.history, 1)
    with pytest.raises(ValueError, match="not in the history index"):
        spec.execute(planner.history.drop(index=[1]), 4)


def test_execute_fits_only_the_frozen_holdout_and_keeps_the_comparison_seed(
    monkeypatch,
):
    captured: list[tuple[list[int], object, dict[str, object]]] = []

    class _Recorder:
        def __init__(self, data, treatment_time, **kwargs):
            captured.append(
                (
                    list(data.index),
                    treatment_time,
                    dict(kwargs["model"].sample_kwargs),
                )
            )

        def fit(self):
            return self

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
            ),
            1,
        ),
    )
    planner = _planner()
    spec = planner.compare_durations(
        (2,),
        (4,),
        cadence=2,
        absolute_lifts=(1.0,),
        target_lift=0.0,
        sample_kwargs={"draws": 8, "chains": 1, "random_seed": 99},
    ).freeze(2)
    assert spec.random_seed == 7
    assert spec.sample_kwargs["random_seed"] == 99
    monkeypatch.setattr(
        "causalpy.experiments.synthetic_control.SyntheticControl",
        _Recorder,
    )
    spec.execute(planner.history, 4)
    assert len(captured) == 1
    index, treatment_time, sample_kwargs = captured[0]
    assert index == [0, 1, 2, 3, 4, 6]
    assert 5 not in index
    assert 7 not in index
    assert treatment_time == 4
    assert sample_kwargs["draws"] == 8
    assert sample_kwargs["random_seed"] == 99
    with pytest.raises(ValueError, match="cannot be executed"):
        spec.execute(planner.history.iloc[:5], 4)


def test_audit_and_execute_accept_the_planner_indexed_panel(monkeypatch):
    captured: list[list[int]] = []

    class _Recorder:
        def __init__(self, data, treatment_time, **kwargs):
            captured.append(list(data.index))

        def fit(self):
            return self

    raw = _frame().reset_index()
    planner = SyntheticControlPlanner(
        raw,
        "treated",
        CONTROLS,
        RULE,
        model=WeightedSumFitter(sample_kwargs={}),
        time_column="t",
        random_seed=7,
    )
    assert "t" not in planner.history.columns
    assert planner.history.index.name == "t"
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
            ),
            1,
        ),
    )
    spec = planner.compare_durations(
        (2,),
        (4,),
        audit_origins=(6,),
        absolute_lifts=(1.0,),
        target_lift=0.0,
        sample_kwargs={"draws": 8, "chains": 1, "random_seed": 99},
    ).freeze(2)
    assert spec.time_column == "t"
    audit = spec.audit(planner.history)
    assert audit.selected_duration == 2
    assert audit.requested_count == 1
    assert audit.status == "historical-descriptive"
    monkeypatch.setattr(
        "causalpy.experiments.synthetic_control.SyntheticControl",
        _Recorder,
    )
    spec.execute(planner.history, 4)
    spec.execute(raw, 4)
    assert captured == [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
    renamed = planner.history.rename_axis("other")
    with pytest.raises(DataException, match="is not a column"):
        spec.execute(renamed, 4)


def test_freeze_ignores_planner_mutation_after_compare(monkeypatch):
    def fake_replay(*args, **kwargs):
        window = args[3]
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
                bias_cumulative=0.0,
                bias_average=0.0,
                cumulative_mean=0.0,
                average_mean=0.0,
                n_inference_fits=1,
                diagnostics_status="unavailable",
            ),
            1,
        )

    def fake_translate(experiment, planner, prepared, window):
        return [
            DurationWindow(
                origin=window.origin,
                duration=window.horizon,
                status="scored",
                reason=None,
                training_index=tuple(window.training_index),
                holdout_index=tuple(window.holdout_index),
                bias_cumulative=0.0,
                bias_average=0.0,
            )
        ]

    monkeypatch.setattr("causalpy.historical_validation._replay_window", fake_replay)
    monkeypatch.setattr("causalpy.duration._translate_window", fake_translate)
    planner = _planner()
    comparison = planner.compare_durations(
        (2,),
        (2,),
        audit_origins=(6,),
        absolute_lifts=(1.0,),
        target_lift=1.0,
        sample_kwargs={"draws": 8, "chains": 1},
    )
    planner.treated_unit = "mutated"
    planner.control_units = ("x",)
    planner.estimator = "other"
    planner.decision = DecisionRule("negative", "average", 3.0, 0.8)
    planner.relative_floor = 9.0
    planner.time_column = "mutated"
    planner.min_donor_correlation = 0.5
    planner.auto_scale_sigma = False
    planner.random_seed = 99
    planner.model = WeightedSumFitter(sample_kwargs={"draws": 1})
    spec = comparison.freeze(2)
    assert spec.treated_unit == "treated"
    assert spec.control_units == CONTROLS
    assert spec.estimator == "synthetic_control"
    assert spec.decision_direction == "positive"
    assert spec.decision_estimand == "cumulative"
    assert spec.decision_threshold == 0.0
    assert spec.decision_probability == 0.95
    assert spec.relative_floor is None
    assert spec.time_column is None
    assert spec.min_donor_correlation == 0.0
    assert spec.auto_scale_sigma is True
    assert spec.random_seed == 7
    assert spec.audit_origins == (6,)
    assert spec.exclusions == ()
    assert spec.sample_kwargs["draws"] == 8
    assert spec.priors == {}
    again = comparison.freeze(2)
    assert again.template is not spec.template
    spec.template.sample_kwargs["draws"] = 3
    assert again.template.sample_kwargs["draws"] == 8


def test_invalid_comparison_requests_fail_before_fitting(monkeypatch):
    monkeypatch.setattr(SyntheticControl, "fit", _reject_fit)
    planner = _planner()
    with pytest.raises(TypeError, match="sequence"):
        planner.preview_durations("2", (2,), absolute_lifts=(1.0,), target_lift=0.0)
    with pytest.raises(ValueError, match="at least one duration"):
        planner.preview_durations((), (2,), absolute_lifts=(1.0,), target_lift=0.0)
    with pytest.raises(ValueError, match="at least one origin"):
        planner.preview_durations((2,), (), absolute_lifts=(1.0,), target_lift=0.0)
    with pytest.raises(ValueError, match="at least 0"):
        planner.preview_durations(
            (2,), (2,), absolute_lifts=(1.0,), target_lift=0.0, max_fits=-1
        )
    with pytest.raises(TypeError, match="real number"):
        planner.preview_durations(
            (2,), (2,), absolute_lifts=(1.0,), target_lift=0.0, fit_cost="2"
        )
    with pytest.raises(TypeError, match="Python int"):
        planner.preview_durations(
            (2,),
            (2,),
            absolute_lifts=(1.0,),
            target_lift=0.0,
            min_training_periods=True,
        )
    with pytest.raises(ValueError, match="at least 1"):
        planner.preview_durations(
            (2,), (2,), absolute_lifts=(1.0,), target_lift=0.0, min_training_periods=0
        )
    with pytest.raises(ValueError, match="relative_lifts or absolute_lifts"):
        planner.preview_durations((2,), (2,), target_lift=0.0)
    with pytest.raises(ValueError, match="absolute"):
        planner.preview_durations(
            (2,),
            (2,),
            absolute_lifts=(1.0,),
            target_lift=0.0,
            target_scale="nope",
        )
    with pytest.raises(ValueError, match="not in the relative"):
        planner.preview_durations(
            (2,),
            (2,),
            absolute_lifts=(1.0,),
            target_lift=0.0,
            target_scale="relative",
        )
    with pytest.raises(TypeError, match="dict"):
        planner.preview_durations(
            (2,),
            (2,),
            absolute_lifts=(1.0,),
            target_lift=0.0,
            sample_kwargs=["draws"],
        )
    with pytest.raises(ValueError, match="draws"):
        planner.preview_durations(
            (2,),
            (2,),
            absolute_lifts=(1.0,),
            target_lift=0.0,
            sample_kwargs={"draws": 0},
        )
    with pytest.raises(ValueError, match="chains"):
        planner.preview_durations(
            (2,),
            (2,),
            absolute_lifts=(1.0,),
            target_lift=0.0,
            sample_kwargs={"draws": 4, "chains": 0},
        )
    planner.model = object()
    with pytest.raises(TypeError, match="simplex"):
        planner.preview_durations((2,), (2,), absolute_lifts=(1.0,), target_lift=0.0)


def test_omitted_chains_default_to_four_and_an_unusable_series_is_not_fitted(
    monkeypatch,
):
    monkeypatch.setattr(SyntheticControl, "fit", _reject_fit)
    refused = _planner().compare_durations(
        (2,),
        (2,),
        absolute_lifts=(1.0,),
        target_lift=0.0,
        sample_kwargs={"draws": 4},
        max_fits=0,
    )
    assert refused.sample_kwargs["draws"] == 4
    assert refused.sample_kwargs["chains"] == 4
    assert refused.summary().empty
    assert "duration" in refused.summary().columns
    assert refused.detail().empty
    with pytest.raises(ValueError, match="not a detection curve"):
        refused.plot()

    preview = _planner().preview_durations(
        (8,), (0, 2, 4), absolute_lifts=(1.0,), target_lift=0.0
    )
    assert preview.status == "insufficient-history"
    assert preview.eligible_origins == ()


def _ready_comparison(monkeypatch, **overrides):
    def fake_replay(*args, **kwargs):
        window = args[3]
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
        return [
            DurationWindow(
                origin=window.origin,
                duration=window.horizon,
                status="scored",
                reason=None,
                training_index=tuple(window.training_index),
                holdout_index=tuple(window.holdout_index),
                estimand=estimand,
                mean=1.0,
                hdi_lower=0.0,
                hdi_upper=2.0,
                truth=1.0,
                truth_covers=True,
                detected=True,
                tail_probability=0.9,
            )
            for estimand in ("cumulative", "average")
        ]

    monkeypatch.setattr("causalpy.historical_validation._replay_window", fake_replay)
    monkeypatch.setattr("causalpy.duration._translate_window", fake_translate)
    kwargs = {
        "audit_origins": (),
        "absolute_lifts": (1.0,),
        "target_lift": 0.0,
        "sample_kwargs": LIGHT,
    }
    kwargs.update(overrides)
    return _planner().compare_durations((2,), (2, 4), **kwargs)


def test_plot_freeze_and_recommendation_keep_the_recorded_choice(monkeypatch):
    import matplotlib.pyplot as plt

    comparison = _ready_comparison(monkeypatch)
    blank = dataclasses.replace(comparison.windows[0], bias_cumulative=None)
    skipped = dataclasses.replace(comparison, windows=(blank, *comparison.windows))
    figure, axes = plt.subplots(1, 2)
    try:
        skipped.plot()
        with pytest.raises(ValueError, match="pair"):
            comparison.plot(axes=(axes[0],))
        returned, used = comparison.plot(axes=(axes[0], axes[1]))
    finally:
        plt.close(figure)
    assert used[0] is axes[0]
    assert returned is figure
    with pytest.raises(TypeError, match="Python int"):
        comparison.freeze(True)
    with pytest.raises(ValueError, match="was not evaluated"):
        comparison.freeze(9)
    spec = comparison.freeze(2)
    assert "recorded duration is 2" in spec.recommendation()
    biased = dataclasses.replace(comparison, persistent_bias=True)
    assert "Persistent null bias is visible and is not corrected." in (
        biased.recommendation()
    )
    audit = spec.audit(_frame())
    assert audit.status == "inconclusive"
    assert audit.n_inference_fits == 0
    assert audit.reasons[0].startswith("No audit origins")


def test_unscored_audit_is_inconclusive_and_a_failed_audit_is_a_limitation(
    monkeypatch,
):
    phase = {"audit": False}

    def fake_replay(*args, **kwargs):
        window = args[3]
        if phase["audit"] and window.origin == 6:
            return (
                WindowReplay(
                    plan=window,
                    status="fit-failed",
                    reason="sampling failed",
                    n_inference_fits=1,
                    diagnostics_status="unavailable",
                ),
                1,
            )
        if phase["audit"] and window.origin == 4:
            return (
                WindowReplay(
                    plan=window,
                    status="not-run",
                    reason="not scored",
                    n_inference_fits=0,
                ),
                0,
            )
        return (
            WindowReplay(
                plan=window,
                status="scored",
                reason=None,
                bias_cumulative=0.1,
                bias_average=0.1,
                cumulative_mean=0.1,
                average_mean=0.1,
                n_inference_fits=1,
                diagnostics_status="unavailable",
            ),
            1,
        )

    def fake_translate(experiment, planner, prepared, window):
        return [
            DurationWindow(
                origin=window.origin,
                duration=window.horizon,
                status="scored",
                reason=None,
                training_index=tuple(window.training_index),
                holdout_index=tuple(window.holdout_index),
                estimand="cumulative",
                mean=1.0,
                hdi_lower=0.0,
                hdi_upper=2.0,
                truth=1.0,
                truth_covers=True,
                detected=True,
                tail_probability=0.2,
            )
        ]

    monkeypatch.setattr("causalpy.historical_validation._replay_window", fake_replay)
    monkeypatch.setattr("causalpy.duration._translate_window", fake_translate)
    planner = _planner()
    comparison = planner.compare_durations(
        (2,),
        (2,),
        audit_origins=(4, 6),
        absolute_lifts=(1.0,),
        target_lift=0.0,
        sample_kwargs=LIGHT,
    )
    spec = comparison.freeze(2)
    phase["audit"] = True
    audit = spec.audit(_frame())
    assert audit.scored_count == 0
    assert audit.status == "inconclusive"

    def mixed_replay(*args, **kwargs):
        window = args[3]
        if window.origin == 6:
            return (
                WindowReplay(
                    plan=window,
                    status="fit-failed",
                    reason="sampling failed",
                    n_inference_fits=1,
                    diagnostics_status="unavailable",
                ),
                1,
            )
        return (
            WindowReplay(
                plan=window,
                status="scored",
                reason=None,
                bias_cumulative=0.1,
                bias_average=0.1,
                cumulative_mean=0.1,
                average_mean=0.1,
                n_inference_fits=1,
                diagnostics_status="unavailable",
            ),
            1,
        )

    monkeypatch.setattr("causalpy.historical_validation._replay_window", mixed_replay)
    limited = spec.audit(_frame())
    assert limited.failed_count == 1
    assert limited.scored_count == 1
    assert limited.status == "limitation"
    assert "not a historical description" in limited.recommendation()


def test_profile_measures_two_runs_without_changing_the_request(monkeypatch):
    calls = []

    def fake_compare(*args, **kwargs):
        calls.append((args[1], kwargs["selection_origins"], kwargs["sample_kwargs"]))
        return SimpleNamespace(
            fit_seconds=0.1,
            score_seconds=0.2,
            sample_kwargs={"draws": 8, "chains": 1},
            n_inference_fits=2,
            status="historical-descriptive",
        )

    monkeypatch.setattr("causalpy.duration._compare_durations", fake_compare)
    profile = _planner().profile_durations(
        (2,),
        (2, 4),
        absolute_lifts=(1.0,),
        target_lift=0.0,
        sample_kwargs=LIGHT,
    )
    assert calls == [
        ((2,), (2, 4), LIGHT),
        ((2,), (2, 4), LIGHT),
    ]
    assert profile.cold_fit_seconds == 0.1
    assert profile.warm_score_seconds == 0.2
    assert profile.n_inference_fits == 2
    assert profile.status == "historical-descriptive"
    assert profile.peak_rss_bytes >= 0
    assert "not a" in profile.note
    assert "recommendation" in profile.note


def test_rss_high_water_mark_uses_platform_units(monkeypatch):
    monkeypatch.setattr(
        duration.resource,
        "getrusage",
        lambda who: SimpleNamespace(ru_maxrss=2048),
    )
    monkeypatch.setattr(duration.sys, "platform", "darwin")
    assert duration._rss_bytes() == 2048
    monkeypatch.setattr(duration.sys, "platform", "linux")
    assert duration._rss_bytes() == 2048 * 1024
