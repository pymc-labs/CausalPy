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
"""Tests for pipeline infrastructure and EstimateEffect step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import causalpy as cp
from causalpy.pipeline import Pipeline, PipelineContext, PipelineResult, Step

# ---------------------------------------------------------------------------
# Mock steps for unit-testing the pipeline orchestrator
# ---------------------------------------------------------------------------


class _MockStep:
    """Minimal Step-protocol implementation that records calls."""

    def __init__(self, name: str = "mock") -> None:
        self.name = name
        self.validated = False
        self.ran = False

    def validate(self, context: PipelineContext) -> None:
        self.validated = True

    def run(self, context: PipelineContext) -> PipelineContext:
        self.ran = True
        return context


class _FailingValidationStep:
    """Step whose validate() always raises."""

    def validate(self, context: PipelineContext) -> None:
        raise ValueError("bad config")

    def run(self, context: PipelineContext) -> PipelineContext:
        return context  # pragma: no cover


class _ContextMutatingStep:
    """Step that writes a marker into the context report field."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def validate(self, context: PipelineContext) -> None:
        pass

    def run(self, context: PipelineContext) -> PipelineContext:
        context.report = self.marker
        return context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_empty_steps_raises(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Pipeline(data=sample_df, steps=[])

    def test_non_dataframe_raises(self) -> None:
        with pytest.raises(TypeError, match="DataFrame"):
            Pipeline(data="not a df", steps=[_MockStep()])  # type: ignore[arg-type]

    def test_non_step_raises(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(TypeError, match="Step protocol"):
            Pipeline(data=sample_df, steps=["not a step"])  # type: ignore[list-item]

    def test_validates_before_running(self, sample_df: pd.DataFrame) -> None:
        failing = _FailingValidationStep()
        good = _MockStep()
        with pytest.raises(ValueError, match="bad config"):
            Pipeline(data=sample_df, steps=[good, failing]).run()
        assert good.validated
        assert not good.ran

    def test_runs_steps_sequentially(self, sample_df: pd.DataFrame) -> None:
        s1 = _MockStep("first")
        s2 = _MockStep("second")
        Pipeline(data=sample_df, steps=[s1, s2]).run()
        assert s1.validated and s1.ran
        assert s2.validated and s2.ran

    def test_context_flows_between_steps(self, sample_df: pd.DataFrame) -> None:
        result = Pipeline(
            data=sample_df,
            steps=[_ContextMutatingStep("hello")],
        ).run()
        assert result.report == "hello"

    def test_result_type(self, sample_df: pd.DataFrame) -> None:
        result = Pipeline(data=sample_df, steps=[_MockStep()]).run()
        assert isinstance(result, PipelineResult)


# ---------------------------------------------------------------------------
# PipelineContext tests
# ---------------------------------------------------------------------------


class TestPipelineContext:
    def test_defaults(self, sample_df: pd.DataFrame) -> None:
        ctx = PipelineContext(data=sample_df)
        assert ctx.experiment is None
        assert ctx.experiment_config is None
        assert ctx.effect_summary is None
        assert ctx.sensitivity_results == []
        assert ctx.report is None

    def test_data_stored(self, sample_df: pd.DataFrame) -> None:
        ctx = PipelineContext(data=sample_df)
        pd.testing.assert_frame_equal(ctx.data, sample_df)


# ---------------------------------------------------------------------------
# PipelineResult tests
# ---------------------------------------------------------------------------


class TestPipelineResult:
    def test_from_context(self, sample_df: pd.DataFrame) -> None:
        ctx = PipelineContext(data=sample_df)
        ctx.report = "test_report"
        result = PipelineResult.from_context(ctx)
        assert result.report == "test_report"
        assert result.experiment is None
        assert result.sensitivity_results == []


# ---------------------------------------------------------------------------
# Step protocol tests
# ---------------------------------------------------------------------------


class TestStepProtocol:
    def test_mock_step_satisfies_protocol(self) -> None:
        assert isinstance(_MockStep(), Step)

    def test_object_without_methods_does_not_satisfy(self) -> None:
        assert not isinstance("not a step", Step)

    def test_dataclass_with_correct_methods_satisfies(self) -> None:
        @dataclass
        class _Adhoc:
            def validate(self, context: PipelineContext) -> None:
                pass

            def run(self, context: PipelineContext) -> PipelineContext:
                return context

        assert isinstance(_Adhoc(), Step)


# ---------------------------------------------------------------------------
# EstimateEffect tests
# ---------------------------------------------------------------------------


class TestEstimateEffect:
    def test_repr(self) -> None:
        step = cp.EstimateEffect(
            method=cp.InterruptedTimeSeries,
            formula="y ~ 1 + t",
            treatment_time=70,
        )
        r = repr(step)
        assert "InterruptedTimeSeries" in r
        assert "formula=" in r

    def test_validate_rejects_non_experiment(self, sample_df: pd.DataFrame) -> None:
        step = cp.EstimateEffect(method=str)  # type: ignore[arg-type]
        ctx = PipelineContext(data=sample_df)
        with pytest.raises(TypeError, match="BaseExperiment subclass"):
            step.validate(ctx)

    def test_validate_rejects_data_kwarg(self, sample_df: pd.DataFrame) -> None:
        step = cp.EstimateEffect(
            method=cp.InterruptedTimeSeries,
            data=sample_df,
            treatment_time=70,
            formula="y ~ 1",
        )
        ctx = PipelineContext(data=sample_df)
        with pytest.raises(ValueError, match="Do not pass 'data'"):
            step.validate(ctx)

    def test_validate_accepts_valid_config(self, sample_df: pd.DataFrame) -> None:
        step = cp.EstimateEffect(
            method=cp.InterruptedTimeSeries,
            treatment_time=70,
            formula="y ~ 1",
        )
        ctx = PipelineContext(data=sample_df)
        step.validate(ctx)

    def test_satisfies_step_protocol(self) -> None:
        step = cp.EstimateEffect(
            method=cp.InterruptedTimeSeries,
            treatment_time=70,
            formula="y ~ 1",
        )
        assert isinstance(step, Step)

    def test_run_fits_experiment(self) -> None:
        """Integration test: EstimateEffect fits a real OLS ITS model."""
        np.random.seed(42)
        n = 100
        treatment_time = 70
        df = pd.DataFrame(
            {
                "t": np.arange(n),
                "y": np.random.normal(size=n),
            }
        )
        from sklearn.linear_model import LinearRegression

        model = cp.create_causalpy_compatible_class(LinearRegression())

        step = cp.EstimateEffect(
            method=cp.InterruptedTimeSeries,
            treatment_time=treatment_time,
            formula="y ~ 1 + t",
            model=model,
        )
        ctx = PipelineContext(data=df)
        step.validate(ctx)
        ctx = step.run(ctx)

        assert ctx.experiment is not None
        assert isinstance(ctx.experiment, cp.InterruptedTimeSeries)
        assert ctx.experiment_config is not None
        assert ctx.experiment_config["method"] is cp.InterruptedTimeSeries
        assert ctx.effect_summary is not None

    def test_pipeline_with_estimate_effect(self) -> None:
        """Integration test: full pipeline with a single EstimateEffect step."""
        np.random.seed(42)
        n = 100
        treatment_time = 70
        df = pd.DataFrame(
            {
                "t": np.arange(n),
                "y": np.random.normal(size=n),
            }
        )
        from sklearn.linear_model import LinearRegression

        model = cp.create_causalpy_compatible_class(LinearRegression())

        result = cp.Pipeline(
            data=df,
            steps=[
                cp.EstimateEffect(
                    method=cp.InterruptedTimeSeries,
                    treatment_time=treatment_time,
                    formula="y ~ 1 + t",
                    model=model,
                ),
            ],
        ).run()

        assert isinstance(result, PipelineResult)
        assert result.experiment is not None
        assert result.effect_summary is not None
        assert result.sensitivity_results == []
        assert result.report is None
