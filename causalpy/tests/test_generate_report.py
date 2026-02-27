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
"""Tests for GenerateReport pipeline step."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.checks.base import CheckResult
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.pipeline import Pipeline, PipelineContext, PipelineResult, Step
from causalpy.steps.report import GenerateReport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def its_context() -> PipelineContext:
    """PipelineContext with a fitted OLS ITS experiment and effect summary."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({"t": np.arange(n), "y": np.random.normal(size=n)})
    model = cp.create_causalpy_compatible_class(LinearRegression())
    experiment = InterruptedTimeSeries(
        df, treatment_time=70, formula="y ~ 1 + t", model=model
    )
    ctx = PipelineContext(data=df)
    ctx.experiment = experiment
    ctx.effect_summary = experiment.effect_summary()
    return ctx


# ---------------------------------------------------------------------------
# GenerateReport tests
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Tests for the GenerateReport pipeline step."""

    def test_satisfies_step_protocol(self):
        assert isinstance(GenerateReport(), Step)

    def test_run_produces_html(self, its_context):
        step = GenerateReport()
        ctx = step.run(its_context)
        assert isinstance(ctx.report, str)
        assert "<!DOCTYPE html>" in ctx.report
        assert "CausalPy Analysis Report" in ctx.report

    def test_includes_effect_summary(self, its_context):
        step = GenerateReport()
        ctx = step.run(its_context)
        assert "Effect Summary" in ctx.report

    def test_excludes_effect_summary_when_disabled(self, its_context):
        step = GenerateReport(include_effect_summary=False)
        ctx = step.run(its_context)
        assert "Effect Summary" not in ctx.report

    def test_includes_sensitivity_results(self, its_context):
        its_context.sensitivity_results = [
            CheckResult(
                check_name="TestCheck",
                passed=True,
                text="All good",
                table=pd.DataFrame({"a": [1]}),
            )
        ]
        step = GenerateReport()
        ctx = step.run(its_context)
        assert "Sensitivity Analysis" in ctx.report
        assert "TestCheck" in ctx.report
        assert "PASS" in ctx.report

    def test_handles_failing_check(self, its_context):
        its_context.sensitivity_results = [
            CheckResult(
                check_name="FailCheck",
                passed=False,
                text="Something wrong",
            )
        ]
        step = GenerateReport()
        ctx = step.run(its_context)
        assert "FAIL" in ctx.report

    def test_handles_informational_check(self, its_context):
        its_context.sensitivity_results = [
            CheckResult(
                check_name="InfoCheck",
                passed=None,
                text="Just info",
            )
        ]
        step = GenerateReport()
        ctx = step.run(its_context)
        assert "INFO" in ctx.report

    def test_handles_empty_context(self):
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))
        step = GenerateReport()
        ctx = step.run(ctx)
        assert isinstance(ctx.report, str)
        assert "<!DOCTYPE html>" in ctx.report

    def test_write_to_file(self, its_context, tmp_path):
        output = tmp_path / "report.html"
        step = GenerateReport(output_file=output)
        step.run(its_context)
        assert output.exists()
        content = output.read_text()
        assert "CausalPy Analysis Report" in content

    def test_repr(self):
        step = GenerateReport(include_plots=False)
        assert "include_plots=False" in repr(step)

    def test_validate_always_passes(self):
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))
        GenerateReport().validate(ctx)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestGenerateReportPipelineIntegration:
    """Integration tests for GenerateReport within a full pipeline."""

    def test_full_pipeline(self):
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
                GenerateReport(include_plots=False),
            ],
        ).run()

        assert isinstance(result, PipelineResult)
        assert isinstance(result.report, str)
        assert "Effect Summary" in result.report


# ---------------------------------------------------------------------------
# Standalone generate_report() on BaseExperiment
# ---------------------------------------------------------------------------


class TestStandaloneGenerateReport:
    """Tests for BaseExperiment.generate_report() convenience method."""

    @pytest.fixture
    def experiment(self):
        """Return a fitted OLS ITS experiment."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({"t": np.arange(n), "y": np.random.normal(size=n)})
        model = cp.create_causalpy_compatible_class(LinearRegression())
        return InterruptedTimeSeries(
            df, treatment_time=70, formula="y ~ 1 + t", model=model
        )

    def test_returns_html_string(self, experiment):
        html = experiment.generate_report(include_plots=False)
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "CausalPy Analysis Report" in html

    def test_includes_effect_summary_by_default(self, experiment):
        html = experiment.generate_report(include_plots=False)
        assert "Effect Summary" in html

    def test_excludes_effect_summary_when_disabled(self, experiment):
        html = experiment.generate_report(
            include_plots=False, include_effect_summary=False
        )
        assert "Effect Summary" not in html

    def test_writes_to_file(self, experiment, tmp_path):
        path = tmp_path / "report.html"
        html = experiment.generate_report(include_plots=False, output_file=path)
        assert path.exists()
        assert path.read_text() == html

    def test_includes_plots(self, experiment):
        html = experiment.generate_report(include_plots=True)
        assert "data:image/png;base64," in html
