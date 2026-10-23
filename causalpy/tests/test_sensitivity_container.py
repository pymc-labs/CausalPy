#   Copyright 2022 - 2026 The PyMC Labs Developers
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
"""Tests for SensitivityAnalysis container and Check protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.checks.base import Check, CheckResult
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pipeline import Pipeline, PipelineContext, Step
from causalpy.steps.sensitivity import (
    _DEFAULT_CHECKS,
    SensitivityAnalysis,
    SensitivitySummary,
    register_default_check,
)

# ---------------------------------------------------------------------------
# Mock check for testing
# ---------------------------------------------------------------------------


class _PassingCheck:
    """Mock check that always passes."""

    applicable_methods = {InterruptedTimeSeries}

    def validate(self, experiment):
        """No-op validation."""
        pass

    def run(self, experiment, context):
        """Return a passing CheckResult."""
        return CheckResult(
            check_name="PassingCheck",
            passed=True,
            text="Everything looks good.",
        )


class _FailingCheck:
    """Mock check that always fails."""

    applicable_methods = {InterruptedTimeSeries}

    def validate(self, experiment):
        """No-op validation."""
        pass

    def run(self, experiment, context):
        """Return a failing CheckResult."""
        return CheckResult(
            check_name="FailingCheck",
            passed=False,
            text="Something is wrong.",
        )


class _InformationalCheck:
    """Mock check that returns informational (passed=None) result."""

    applicable_methods = {InterruptedTimeSeries}

    def validate(self, experiment):
        """No-op validation."""
        pass

    def run(self, experiment, context):
        """Return an informational CheckResult (passed=None)."""
        return CheckResult(
            check_name="InformationalCheck",
            passed=None,
            text="Here is some info.",
        )


class _WrongMethodCheck:
    """Check that only applies to SyntheticControl, not ITS."""

    applicable_methods = {SyntheticControl}

    def validate(self, experiment):
        """No-op validation."""
        pass

    def run(self, experiment, context):
        """Return a passing CheckResult."""
        return CheckResult(check_name="WrongMethodCheck", passed=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def its_context() -> PipelineContext:
    """PipelineContext with a fitted OLS ITS experiment."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({"t": np.arange(n), "y": np.random.normal(size=n)})
    model = cp.create_causalpy_compatible_class(LinearRegression())
    experiment = InterruptedTimeSeries(
        df, treatment_time=70, formula="y ~ 1 + t", model=model
    )
    ctx = PipelineContext(data=df)
    ctx.experiment = experiment
    ctx.experiment_config = {
        "method": InterruptedTimeSeries,
        "treatment_time": 70,
        "formula": "y ~ 1 + t",
    }
    return ctx


# ---------------------------------------------------------------------------
# CheckResult tests
# ---------------------------------------------------------------------------


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_defaults(self):
        r = CheckResult(check_name="test")
        assert r.passed is None
        assert r.table is None
        assert r.text == ""
        assert r.figures == []
        assert r.metadata == {}

    def test_with_table(self):
        df = pd.DataFrame({"a": [1, 2]})
        r = CheckResult(check_name="test", table=df)
        pd.testing.assert_frame_equal(r.table, df)


# ---------------------------------------------------------------------------
# Check protocol tests
# ---------------------------------------------------------------------------


class TestCheckProtocol:
    """Tests for Check protocol conformance."""

    def test_mock_satisfies_protocol(self):
        assert isinstance(_PassingCheck(), Check)

    def test_string_does_not_satisfy(self):
        assert not isinstance("not a check", Check)


# ---------------------------------------------------------------------------
# SensitivitySummary tests
# ---------------------------------------------------------------------------


class TestSensitivitySummary:
    """Tests for SensitivitySummary aggregation."""

    def test_all_pass(self):
        results = [
            CheckResult(check_name="a", passed=True, text="ok"),
            CheckResult(check_name="b", passed=True, text="also ok"),
        ]
        summary = SensitivitySummary.from_results(results)
        assert summary.all_passed is True
        assert len(summary.results) == 2
        assert "ok" in summary.text
        assert "also ok" in summary.text

    def test_one_fails(self):
        results = [
            CheckResult(check_name="a", passed=True),
            CheckResult(check_name="b", passed=False, text="bad"),
        ]
        summary = SensitivitySummary.from_results(results)
        assert summary.all_passed is False

    def test_informational_only(self):
        results = [
            CheckResult(check_name="a", passed=None, text="info"),
        ]
        summary = SensitivitySummary.from_results(results)
        assert summary.all_passed is None

    def test_empty(self):
        summary = SensitivitySummary.from_results([])
        assert summary.all_passed is None
        assert summary.results == []
        assert summary.text == ""


# ---------------------------------------------------------------------------
# SensitivityAnalysis step tests
# ---------------------------------------------------------------------------


class TestSensitivityAnalysis:
    """Tests for SensitivityAnalysis pipeline step."""

    def test_satisfies_step_protocol(self):
        step = SensitivityAnalysis(checks=[_PassingCheck()])
        assert isinstance(step, Step)

    def test_validate_accepts_valid_checks(self, its_context):
        step = SensitivityAnalysis(checks=[_PassingCheck()])
        step.validate(its_context)

    def test_validate_rejects_non_check(self, its_context):
        step = SensitivityAnalysis(checks=["not a check"])
        with pytest.raises(TypeError, match="Check protocol"):
            step.validate(its_context)

    def test_run_requires_experiment(self):
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))
        step = SensitivityAnalysis(checks=[_PassingCheck()])
        with pytest.raises(RuntimeError, match="fitted experiment"):
            step.run(ctx)

    def test_run_rejects_inapplicable_check(self, its_context):
        step = SensitivityAnalysis(checks=[_WrongMethodCheck()])
        with pytest.raises(TypeError, match="not applicable"):
            step.run(its_context)

    def test_run_collects_results(self, its_context):
        step = SensitivityAnalysis(checks=[_PassingCheck(), _InformationalCheck()])
        ctx = step.run(its_context)
        assert len(ctx.sensitivity_results) == 2
        assert ctx.sensitivity_results[0].passed is True
        assert ctx.sensitivity_results[1].passed is None

    def test_run_produces_summary(self, its_context):
        step = SensitivityAnalysis(checks=[_PassingCheck(), _FailingCheck()])
        ctx = step.run(its_context)
        summary = ctx.report
        assert isinstance(summary, SensitivitySummary)
        assert summary.all_passed is False

    def test_empty_checks(self, its_context):
        step = SensitivityAnalysis(checks=[])
        ctx = step.run(its_context)
        assert ctx.sensitivity_results == []

    def test_repr(self):
        step = SensitivityAnalysis(checks=[_PassingCheck()])
        assert "_PassingCheck" in repr(step)

    def test_default_for_returns_empty_when_no_defaults(self):
        step = SensitivityAnalysis.default_for(InterruptedTimeSeries)
        assert isinstance(step, SensitivityAnalysis)


# ---------------------------------------------------------------------------
# register_default_check tests
# ---------------------------------------------------------------------------


class TestRegisterDefaultCheck:
    """Tests for register_default_check."""

    def test_register_and_retrieve(self):
        original = _DEFAULT_CHECKS.copy()
        try:

            class _TestCheck:
                applicable_methods = {InterruptedTimeSeries}

                def __init__(self):
                    pass

                def validate(self, experiment):
                    pass

                def run(self, experiment, context):
                    return CheckResult(check_name="TestCheck", passed=True)

            register_default_check(_TestCheck, {InterruptedTimeSeries})
            step = SensitivityAnalysis.default_for(InterruptedTimeSeries)
            assert any(isinstance(c, _TestCheck) for c in step.checks)
        finally:
            _DEFAULT_CHECKS.clear()
            _DEFAULT_CHECKS.update(original)


# ---------------------------------------------------------------------------
# Integration: Pipeline with EstimateEffect + SensitivityAnalysis
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """Integration tests for Pipeline with EstimateEffect and SensitivityAnalysis."""

    def test_estimate_then_sensitivity(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({"t": np.arange(n), "y": np.random.normal(size=n)})
        model = cp.create_causalpy_compatible_class(LinearRegression())

        result = Pipeline(
            data=df,
            steps=[
                cp.EstimateEffect(
                    method=InterruptedTimeSeries,
                    treatment_time=70,
                    formula="y ~ 1 + t",
                    model=model,
                ),
                SensitivityAnalysis(checks=[_PassingCheck()]),
            ],
        ).run()

        assert result.experiment is not None
        assert result.effect_summary is not None
        assert len(result.sensitivity_results) == 1
        assert result.sensitivity_results[0].passed is True


def test_scheduled_descriptive_placebos_render_as_information():
    """The real check keeps its table and figure without inventing a verdict."""
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        {"t": np.arange(70), "y": 10 + 0.1 * np.arange(70)},
    )
    template = cp.pymc_models.LinearRegression(
        sample_kwargs={"chains": 2, "draws": 100, "tune": 100, "random_seed": 42}
    )
    experiment = cp.InterruptedTimeSeries(
        frame, treatment_time=60, formula="y ~ 1 + t", model=template._clone()
    ).fit()
    context = PipelineContext(
        data=frame,
        experiment=experiment,
        experiment_config={
            "method": cp.InterruptedTimeSeries,
            "formula": "y ~ 1 + t",
            "model": template,
        },
    )
    check = cp.checks.PlaceboInTime(
        n_folds=2,
        treatment_times=[30, 40],
        intervention_length=15,
        min_training_rows=30,
        allow_overlap=True,
        calibrate=False,
        random_seed=42,
    )
    try:
        SensitivityAnalysis(checks=[check]).run(context)
        assert context.report.all_passed is None
        result = context.sensitivity_results[0]
        assert result.passed is None
        assert result.table["status"].tolist() == ["completed", "completed"]
        assert len(result.figures) == 1
        cp.GenerateReport().run(context)
        assert 'badge-info">INFO' in context.report
        assert 'badge-pass">PASS' not in context.report
        assert 'badge-fail">FAIL' not in context.report
        assert "pseudo_treatment_time" in context.report
        assert "n_comparison" in context.report
        assert result.text in context.report
        assert context.report.count("data:image/png;base64,") >= 2
    finally:
        plt.close("all")
