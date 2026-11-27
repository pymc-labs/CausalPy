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
"""Refusal and accounting branches that a successful search does not enter."""

from __future__ import annotations

import importlib
from dataclasses import replace
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from causalpy.planning.evaluation import _finished_evaluation
from causalpy.planning.grouped_study import (
    _assemble_grouped,
    _grouped_rmse_ratio,
    _grouped_study,
    _grouped_truth,
    _noise_scale,
    _summarize_diagnostics,
)
from causalpy.planning.power import MisfitNotPower, unit_period_effect
from causalpy.planning.search import (
    ComputationalBudget,
    _candidate_status_counts,
    _dominates,
    _effect_grid,
    _finite_or_none,
    _fixed_design_criterion,
    _power_key,
    _present_float,
    _redesign_reason,
    _row_status,
    _SearchRow,
    _selection_key,
    _setup_refusal,
    _training_diff,
    _validation_panel,
    plot_search,
    search,
    validate_selection,
)
from causalpy.tests.test_planning_search import (
    DURATION,
    LIGHT,
    UNITS,
    WORTHWHILE,
    _budget,
    _criterion,
    _decision,
    _panel,
    _run,
    _selection_end,
    _spy_engine,
    _study,
)

search_module = importlib.import_module("causalpy.planning.search")


def _row(**overrides: object) -> _SearchRow:
    values: dict[str, object] = {
        "candidate_id": "c",
        "treated": ("a",),
        "donors": ("b", "c"),
        "n_donors": 2,
        "group_size": 1,
        "duration": DURATION,
        "estimator": "synthetic_control",
        "cost": 1.0,
        "outcome_share": 0.5,
        "proxy_score": None,
        "proxy_role": None,
        "shortlist_rank": 1,
        "boundary": False,
        "screen_status": "shortlisted",
        "evaluation_status": "completed",
        "reason": None,
        "power": 0.5,
        "power_withheld": False,
        "false_positive_rate": 0.0,
        "precision_met": True,
        "interval_width": 1.0,
        "monte_carlo_error": 0.05,
        "mde": None,
        "mde_status": None,
        "n_fits": 1,
        "failed_runs": 0,
        "divergences": 0,
        "max_rhat": 1.0,
        "min_bulk_ess": 100.0,
        "diagnostics_status": "available",
        "diagnostics_reason": None,
        "unit_scale": 1.0,
        "eligible": False,
        "pareto": False,
        "selected": False,
    }
    values.update(overrides)
    return _SearchRow(**values)  # type: ignore[arg-type]


def _evaluation(mean: float) -> SimpleNamespace:
    return SimpleNamespace(
        decision=SimpleNamespace(
            decided=True,
            hdi_lower=0.0,
            hdi_upper=2.0,
            posterior_tail_probability=0.2,
        ),
        summary=SimpleNamespace(cumulative_total_mean=mean),
    )


def _assemble(mean: float, *, precision_target: float | None = 0.1):
    frame = _panel()
    criterion = replace(_criterion(), precision_target=precision_target)
    return _assemble_grouped(
        draws={
            (DURATION, 0.0): [_evaluation(0.0)],
            (DURATION, WORTHWHILE): [_evaluation(mean)],
        },
        failures=(),
        n_fits=1,
        horizons=(DURATION,),
        translated=(WORTHWHILE,),
        decision=_decision(),
        criterion=criterion,
        scale="absolute",
        baseline=None,
        worthwhile=WORTHWHILE,
        ratio=0.1,
        family="correctly_specified_noisy_reference",
        noise_scale=1.0,
        n_replicates=1,
        estimator="augmented_synthetic_control",
        donors=("b", "c", "d"),
        treated=("a",),
        seed=11,
        frame=frame,
        selection_end=_selection_end(frame),
    )


class _Plan:
    calls: list[dict[str, object]] = []
    outcome: object = ValueError("No fit was started")

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def evaluate(self, *args: object, **kwargs: object) -> object:
        del args
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _grouped(**overrides: object):
    frame = _panel()
    values: dict[str, object] = {
        "frame": frame,
        "treated": ("a",),
        "donors": ("b", "c", "d"),
        "horizons": (DURATION,),
        "effects": (WORTHWHILE,),
        "decision": _decision(),
        "criterion": replace(_criterion(), monte_carlo_error_tolerance=0.5),
        "worthwhile": WORTHWHILE,
        "estimator": "augmented_synthetic_control",
        "family": "correctly_specified_noisy_reference",
        "scale": "absolute",
        "baseline": None,
        "sample_kwargs": LIGHT,
        "outcome_scale": "outcome",
        "time_column": None,
        "selection_end": _selection_end(frame),
        "seed": 11,
        "fit_ceiling": 4,
    }
    values.update(overrides)
    return _grouped_study(**values)  # type: ignore[arg-type]


def test_budget_rejects_a_non_integer_and_a_negative_cap() -> None:
    with pytest.raises(TypeError, match="non-negative integer"):
        ComputationalBudget(True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="non-negative integer"):
        ComputationalBudget("12")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        ComputationalBudget(-1)


def test_search_refuses_malformed_inputs_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_engine(monkeypatch)
    frame = _panel()
    reversed_frame = frame.iloc[::-1]
    with pytest.raises(TypeError, match="durations"):
        _run(durations="4")
    with pytest.raises(ValueError, match="positive"):
        _run(durations=())
    with pytest.raises(ValueError, match="unique"):
        _run(durations=(DURATION, DURATION))
    with pytest.raises(TypeError, match="mapping"):
        _run(sample_kwargs="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="random_seed"):
        _run(sample_kwargs={**LIGHT, "random_seed": 1})
    with pytest.raises(ValueError, match="positive"):
        _run(random_seed=0)
    with pytest.raises(ValueError, match="finite"):
        _run(worthwhile_effect=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        _run(worthwhile_effect="nope")
    with pytest.raises(TypeError, match="DecisionRule"):
        _run(decision="positive")
    with pytest.raises(TypeError, match="PowerCriterion"):
        _run(criterion=object())
    with pytest.raises(ValueError, match="worthwhile effect"):
        _run(effect_grid=(1.0,))
    with pytest.raises(ValueError, match="finite"):
        _run(effect_grid=(WORTHWHILE, float("nan")))
    with pytest.raises(TypeError, match="DataFrame"):
        search(
            object(),  # type: ignore[arg-type]
            UNITS,
            _decision(),
            _criterion(),
            WORTHWHILE,
            (DURATION,),
            selection_end=0,
            sample_kwargs=LIGHT,
            random_seed=11,
            budget=_budget(),
        )
    with pytest.raises(ValueError, match="time column"):
        _run(time_column="missing")
    with pytest.raises(ValueError, match="unique and increasing"):
        _run(history=reversed_frame)
    with pytest.raises(ValueError, match="training rows"):
        search(
            _panel(n_rows=4),
            UNITS,
            _decision(),
            _criterion(),
            WORTHWHILE,
            (DURATION,),
            selection_end=0,
            sample_kwargs=LIGHT,
            random_seed=11,
            budget=_budget(),
        )
    assert calls == []


def test_unknown_scale_rejects_every_candidate_without_a_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_engine(monkeypatch)
    result = _run(scale="log")
    assert calls == []
    assert result.n_inference_fits == 0
    assert result.frame["evaluation_status"].eq("rejected").all()
    assert result.frame["reason"].str.contains("absolute").all()


def test_reference_withhold_and_value_error_stay_on_the_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _withhold(self, durations, effects, *, criterion, **kwargs):
        del self, durations, effects, criterion, kwargs
        raise MisfitNotPower("pre-period misfit", _study())

    monkeypatch.setattr(
        "causalpy.planning.power.ReferencePowerEvaluator.effect_by_duration",
        _withhold,
    )
    withheld = _run()
    assert withheld.frame["evaluation_status"].eq("withheld").all()
    assert withheld.frame["n_fits"].gt(0).all()
    assert "pre-period misfit" in withheld.frame["reason"].iloc[0]

    def _collapse(self, durations, effects, *, criterion, **kwargs):
        del self, durations, effects, criterion, kwargs
        raise ValueError("donor pool collapsed")

    monkeypatch.setattr(
        "causalpy.planning.power.ReferencePowerEvaluator.effect_by_duration",
        _collapse,
    )
    rejected = _run()
    assert rejected.frame["evaluation_status"].eq("rejected").all()
    assert rejected.frame["n_fits"].eq(0).all()
    assert "donor pool collapsed" in rejected.frame["reason"].iloc[0]


def test_empty_completed_plot_and_plot_data_copy_do_not_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_engine(monkeypatch)
    result = _run(budget=_budget(0))
    assert calls == []
    copied = result.plot_data()
    copied.iloc[0, 0] = "mutated"
    assert result.frame.iloc[0, 0] != "mutated"
    figure = plot_search(result)
    assert figure.axes[0].get_legend() is None
    assert figure.axes[0].get_ylabel() == "power at the worthwhile effect"
    plt.close(figure)


def test_frozen_design_inject_and_audit_cover_refusal_and_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spy_engine(monkeypatch)
    design = _run().frozen
    assert design is not None
    with pytest.raises(TypeError, match="pandas DataFrame"):
        design.inject(object(), WORTHWHILE)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing frozen holdout"):
        design.inject(design.history.iloc[0:0], WORTHWHILE)
    with pytest.raises(ValueError, match="missing treated columns"):
        design.inject(
            design.history.drop(columns=list(design.treated_units)), WORTHWHILE
        )
    with pytest.raises(TypeError, match="pandas DataFrame"):
        design.audit(object())  # type: ignore[arg-type]

    panel = design.history.copy()
    panel.index = pd.RangeIndex(10_000, 10_000 + len(panel), name=panel.index.name)

    class _Spec:
        holdout_index = tuple(panel.index[-DURATION:])
        training_index = tuple(panel.index[:-DURATION])

        def run(self, frame: pd.DataFrame) -> SimpleNamespace:
            assert frame.index.equals(panel.index)
            return SimpleNamespace(
                decision=SimpleNamespace(decided=True, posterior_tail_probability=0.91)
            )

    monkeypatch.setattr(
        search_module, "_build_handoff", lambda *args, **kwargs: _Spec()
    )
    report = design.audit(panel)
    assert report.decided is True
    assert report.posterior_tail_probability == pytest.approx(0.91)
    assert report.reason is None
    assert report.candidate_id == design.candidate_id
    assert report.holdout_index == _Spec.holdout_index


def test_validate_selection_requires_a_fixed_donor_core() -> None:
    with pytest.raises(ValueError, match="always_control"):
        validate_selection(
            _panel(),
            UNITS,
            _decision(),
            _criterion(),
            WORTHWHILE,
            (DURATION,),
            n_replicates=1,
            random_seed=11,
            budget=_budget(),
            sample_kwargs=LIGHT,
            selection_end=0,
        )


def test_selection_helpers_refuse_missing_and_conflicting_rows() -> None:
    missing = _row(power=None, cost=None, interval_width=None, outcome_share=None)
    with pytest.raises(TypeError, match="missing power"):
        _power_key(missing)
    with pytest.raises(TypeError, match="missing cost or power"):
        _selection_key(missing)
    with pytest.raises(TypeError, match="missing numeric cost"):
        _present_float("nope", "cost")
    assert (
        _dominates(
            missing,
            _row(**dict.fromkeys(("power", "cost", "interval_width", "outcome_share"))),
        )
        is False
    )
    with pytest.raises(RuntimeError, match="once per candidate"):
        _candidate_status_counts(
            (
                _row(evaluation_status="completed"),
                _row(evaluation_status="failed"),
            )
        )
    status, reason, fits, failed = _row_status(
        pd.Series(
            {"status": "filtered", "filter_reasons": ("too_few_donors", "boundary")}
        ),
        None,
    )
    assert (status, reason, fits, failed) == (
        "filtered",
        "too_few_donors,boundary",
        0,
        0,
    )
    note = _redesign_reason(
        (
            _row(
                power=0.1,
                false_positive_rate=0.9,
                precision_met=False,
                cost=None,
            ),
        ),
        _criterion(),
    )
    assert "below" in note
    assert "false-positive" in note
    assert "precision target" in note
    assert "cost is unknown" in note
    short = _panel(n_rows=3)
    refused = _setup_refusal(short, ("b", "c"), (DURATION,), short.index[0])
    assert refused is not None
    assert refused.status == "rejected"
    assert refused.n_fits == 0
    assert "training rows" in str(refused.reason)


def test_finite_or_none_drops_missing_non_numeric_and_ambiguous_values() -> None:
    class _Unfloatable:
        def __float__(self) -> float:
            raise TypeError("no")

    assert _finite_or_none(None) is None
    assert _finite_or_none(pd.NA) is None
    assert _finite_or_none(np.array([1.0, np.nan])) is None
    assert _finite_or_none(_Unfloatable()) is None
    assert _finite_or_none(float("inf")) is None
    assert _finite_or_none(float("-inf")) is None


def test_finished_evaluation_labels_withheld_and_completed_curves() -> None:
    withheld = _finished_evaluation(
        replace(_study(), power_withheld=True, withhold_reason="misfit"),
        3,
        unit_scale=1.5,
    )
    assert withheld.status == "withheld"
    assert withheld.reason == "misfit"
    assert withheld.n_fits == 3
    assert withheld.unit_scale == 1.5
    completed = _finished_evaluation(_study(), 2)
    assert completed.status == "completed"
    assert completed.reason is None
    assert completed.failed_runs == 0


def test_grouped_truth_scales_the_unit_period_effect() -> None:
    increment = unit_period_effect(
        scale="absolute",
        magnitude=2.0,
        duration=DURATION,
        n_treated=3,
        baseline=None,
    )
    assert _grouped_truth(
        "absolute", 2.0, DURATION, None, "cumulative_total", 3
    ) == pytest.approx(increment * DURATION * 3)
    assert _grouped_truth(
        "absolute", 2.0, DURATION, None, "average_per_period", 3
    ) == pytest.approx(increment * 3)
    assert _grouped_truth(
        "absolute", 2.0, DURATION, None, "average_per_treated_unit", 3
    ) == pytest.approx(increment * DURATION)
    with pytest.raises(ValueError, match="Unsupported estimand"):
        _grouped_truth("absolute", 2.0, DURATION, None, "not_an_estimand", 3)


def test_constant_donors_refuse_a_noise_scale() -> None:
    frame = pd.DataFrame({"b": [1.0, 1.0, 1.0, 1.0], "c": [1.0, 1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="positive finite variation"):
        _noise_scale(frame, ("b", "c"), frame.index[1])


def test_diagnostics_summary_keeps_a_missing_count_null() -> None:
    empty = _summarize_diagnostics(())
    assert empty.status is None
    assert empty.divergences is None
    missing = _summarize_diagnostics(
        [
            SimpleNamespace(
                diagnostics=SimpleNamespace(
                    divergences=None,
                    diagnostics_status="other",
                    diagnostics_reason="late",
                    max_rhat=1.2,
                    min_bulk_ess=None,
                )
            )
        ]
    )
    assert missing.status == "unavailable"
    assert missing.divergences is None
    assert missing.reason == "late"
    counted = _summarize_diagnostics(
        [
            SimpleNamespace(
                diagnostics=SimpleNamespace(
                    divergences=2,
                    diagnostics_status="other",
                    diagnostics_reason=None,
                    max_rhat=1.1,
                    min_bulk_ess=50.0,
                )
            )
        ]
    )
    assert counted.status == "available"
    assert counted.divergences == 2
    assert counted.max_rhat == pytest.approx(1.1)
    assert counted.min_bulk_ess == pytest.approx(50.0)


def test_grouped_assembly_rejects_a_non_finite_mean_and_records_precision() -> None:
    with pytest.raises(RuntimeError, match="non-finite estimand mean"):
        _assemble(float("nan"))
    curve, _diagnostics = _assemble(WORTHWHILE, precision_target=0.1)
    point = curve.point(DURATION, WORTHWHILE)
    assert point.precision_met is False
    assert point.interval_width == pytest.approx(2.0)


def test_grouped_study_records_relative_effects_and_fit_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("causalpy.planning.grouped_study.HistoryOnlyPlan", _Plan)
    _Plan.calls = []
    _Plan.outcome = ValueError("No fit was started")
    rejected = _grouped(scale="relative", baseline=10.0)
    assert rejected.status == "rejected"
    assert rejected.n_fits == 0
    assert "No fit was started" in str(rejected.reason)
    assert _Plan.calls[-1]["relative_effects"] == (0.0, WORTHWHILE)
    assert _Plan.calls[-1]["relative_baseline"] == 10.0

    _Plan.outcome = ValueError("adapter exploded")
    failed = _grouped()
    assert failed.status == "failed"
    assert failed.n_fits == 1
    assert "adapter exploded" in str(failed.reason)

    _Plan.outcome = SimpleNamespace(
        n_inference_fits=2,
        evaluations=(
            SimpleNamespace(
                duration=DURATION, scenario=SimpleNamespace(magnitude=99.0)
            ),
        ),
    )
    unexpected = _grouped()
    assert unexpected.status == "failed"
    assert unexpected.n_fits == 2
    assert "not requested" in str(unexpected.reason)


def test_declared_persistent_misfit_rejects_a_calm_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "causalpy.planning.grouped_study._grouped_rmse_ratio",
        lambda *args, **kwargs: 0.1,
    )
    result = _grouped(family="persistent_misfit")
    assert result.status == "rejected"
    assert result.n_fits == 0
    assert "Declared persistent_misfit was not present" in str(result.reason)
    assert "No fit was started" in str(result.reason)


def test_grouped_rmse_ratio_refuses_a_one_row_pre_period() -> None:
    panel = pd.DataFrame({"b": [1.0, 2.0], "c": [3.0, 4.0]})
    with pytest.raises(ValueError, match="pre-period screen"):
        _grouped_rmse_ratio(
            (panel,),
            ("b",),
            ("c",),
            panel.index[0],
            1.0,
            {"b": 1.0},
        )


def test_validation_panel_refuses_a_pool_with_no_treated_unit() -> None:
    history = _panel(n_rows=4)
    with pytest.raises(ValueError, match="at least one unit outside always_control"):
        _validation_panel(
            history,
            UNITS,
            UNITS,
            "correctly_specified_noisy_reference",
            0,
            0,
            history.index[1],
            decision=_decision(),
            sample_kwargs={},
            reference=False,
            panel_cache={},
        )


def test_training_diff_is_zero_when_the_index_is_empty() -> None:
    frame = _panel(n_rows=4)
    assert _training_diff(frame, frame, ()) == 0.0


def test_redesign_reason_names_an_all_filtered_pool() -> None:
    reason = _redesign_reason(
        (_row(evaluation_status="filtered", power=None),),
        _criterion(),
    )
    assert "Widen eligibility" in reason


def test_fixed_design_criterion_keeps_an_existing_scope() -> None:
    criterion = replace(_criterion(), qualification_scope="fixed_design")
    assert _fixed_design_criterion(criterion) is criterion


def test_effect_grid_keeps_a_worthwhile_member() -> None:
    assert _effect_grid(WORTHWHILE, (0.0, WORTHWHILE)) == (0.0, WORTHWHILE)
