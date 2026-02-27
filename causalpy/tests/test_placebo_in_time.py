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
"""Tests for PlaceboInTime sensitivity check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.checks.base import Check, CheckResult
from causalpy.checks.placebo_in_time import PlaceboFoldResult, PlaceboInTime
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.pipeline import Pipeline, PipelineContext

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_its_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a simple ITS dataset with numeric index."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "t": np.arange(n),
            "y": rng.normal(size=n),
        }
    )


def _make_model():
    return cp.create_causalpy_compatible_class(LinearRegression())


@pytest.fixture
def its_data() -> pd.DataFrame:
    return _make_its_data()


@pytest.fixture
def its_experiment(its_data: pd.DataFrame) -> InterruptedTimeSeries:
    return InterruptedTimeSeries(
        its_data,
        treatment_time=150,
        formula="y ~ 1 + t",
        model=_make_model(),
    )


@pytest.fixture
def its_context(
    its_data: pd.DataFrame,
    its_experiment: InterruptedTimeSeries,
) -> PipelineContext:
    ctx = PipelineContext(data=its_data)
    ctx.experiment = its_experiment
    ctx.experiment_config = {
        "method": InterruptedTimeSeries,
        "treatment_time": 150,
        "formula": "y ~ 1 + t",
        "model": _make_model(),
    }
    return ctx


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestPlaceboInTimeConstruction:
    def test_default_n_folds(self):
        check = PlaceboInTime()
        assert check.n_folds == 3

    def test_custom_n_folds(self):
        check = PlaceboInTime(n_folds=5)
        assert check.n_folds == 5

    def test_invalid_n_folds(self):
        with pytest.raises(ValueError, match="n_folds must be >= 1"):
            PlaceboInTime(n_folds=0)

    def test_satisfies_check_protocol(self):
        assert isinstance(PlaceboInTime(), Check)

    def test_applicable_methods(self):
        check = PlaceboInTime()
        assert InterruptedTimeSeries in check.applicable_methods
        assert cp.SyntheticControl in check.applicable_methods


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestPlaceboInTimeValidation:
    def test_validate_accepts_its(self, its_experiment):
        PlaceboInTime().validate(its_experiment)

    def test_validate_rejects_experiment_without_treatment_time(self):
        class _FakeExperiment:
            pass

        check = PlaceboInTime()
        with pytest.raises(TypeError, match="treatment_time"):
            check.validate(_FakeExperiment())


# ---------------------------------------------------------------------------
# Run tests (integration with OLS)
# ---------------------------------------------------------------------------


class TestPlaceboInTimeRun:
    def test_run_produces_check_result(self, its_experiment, its_context):
        check = PlaceboInTime(n_folds=2)
        result = check.run(its_experiment, its_context)
        assert isinstance(result, CheckResult)
        assert result.check_name == "PlaceboInTime"
        assert "fold" in result.text.lower()

    def test_run_produces_fold_results(self, its_experiment, its_context):
        check = PlaceboInTime(n_folds=2)
        result = check.run(its_experiment, its_context)
        fold_results = result.metadata["fold_results"]
        assert len(fold_results) == 2
        for fr in fold_results:
            assert isinstance(fr, PlaceboFoldResult)
            assert isinstance(fr.experiment, InterruptedTimeSeries)

    def test_fold_treatment_times_are_shifted(self, its_experiment, its_context):
        check = PlaceboInTime(n_folds=2)
        result = check.run(its_experiment, its_context)
        fold_results = result.metadata["fold_results"]
        for fr in fold_results:
            assert fr.pseudo_treatment_time < its_experiment.treatment_time

    def test_single_fold(self, its_experiment, its_context):
        check = PlaceboInTime(n_folds=1)
        result = check.run(its_experiment, its_context)
        assert len(result.metadata["fold_results"]) == 1

    def test_stores_fold_results_on_check(self, its_experiment, its_context):
        check = PlaceboInTime(n_folds=2)
        check.run(its_experiment, its_context)
        assert len(check.fold_results) == 2

    def test_with_custom_factory(self, its_experiment, its_context):
        def factory(data, treatment_time):
            return InterruptedTimeSeries(
                data,
                treatment_time=treatment_time,
                formula="y ~ 1 + t",
                model=_make_model(),
            )

        check = PlaceboInTime(n_folds=2, experiment_factory=factory)
        result = check.run(its_experiment, its_context)
        assert len(result.metadata["fold_results"]) == 2

    def test_no_experiment_config_and_no_factory_raises(self, its_experiment):
        ctx = PipelineContext(data=its_experiment.data)
        ctx.experiment = its_experiment
        check = PlaceboInTime(n_folds=2)
        with pytest.raises(RuntimeError, match="experiment_config"):
            check.run(its_experiment, ctx)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestPlaceboInTimePipelineIntegration:
    def test_pipeline_with_placebo_in_time(self, its_data):
        result = Pipeline(
            data=its_data,
            steps=[
                cp.EstimateEffect(
                    method=InterruptedTimeSeries,
                    treatment_time=150,
                    formula="y ~ 1 + t",
                    model=_make_model(),
                ),
                cp.SensitivityAnalysis(
                    checks=[PlaceboInTime(n_folds=2)],
                ),
            ],
        ).run()

        assert result.experiment is not None
        assert len(result.sensitivity_results) == 1
        fold_results = result.sensitivity_results[0].metadata["fold_results"]
        assert len(fold_results) == 2
