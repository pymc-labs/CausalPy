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
from causalpy.reporting import EffectSummary
from causalpy.steps.sensitivity import _DEFAULT_CHECKS, SensitivityAnalysis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_its_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a simple ITS dataset with numeric index and no real effect."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "t": np.arange(n),
            "y": rng.normal(size=n),
        }
    )


def _make_model():
    """Create a minimal OLS-compatible model for ITS tests."""
    return cp.create_causalpy_compatible_class(LinearRegression())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def its_data() -> pd.DataFrame:
    """Minimal ITS dataset for PlaceboInTime tests."""
    return _make_its_data()


@pytest.fixture
def its_experiment(its_data: pd.DataFrame) -> InterruptedTimeSeries:
    """Fitted ITS experiment for PlaceboInTime tests."""
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
    """PipelineContext with fitted ITS experiment for PlaceboInTime tests."""
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


def test_default_n_folds():
    check = PlaceboInTime()
    assert check.n_folds == 3


def test_custom_n_folds():
    check = PlaceboInTime(n_folds=5)
    assert check.n_folds == 5


def test_invalid_n_folds():
    with pytest.raises(ValueError, match="n_folds must be >= 1"):
        PlaceboInTime(n_folds=0)


def test_satisfies_check_protocol():
    assert isinstance(PlaceboInTime(), Check)


def test_applicable_methods():
    check = PlaceboInTime()
    assert InterruptedTimeSeries in check.applicable_methods
    assert cp.SyntheticControl in check.applicable_methods


def test_repr():
    assert "n_folds=3" in repr(PlaceboInTime())


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_validate_accepts_its(its_experiment):
    PlaceboInTime().validate(its_experiment)


def test_validate_rejects_experiment_without_treatment_time():
    class _FakeExperiment:
        pass

    check = PlaceboInTime()
    with pytest.raises(TypeError, match="treatment_time"):
        check.validate(_FakeExperiment())


# ---------------------------------------------------------------------------
# Run tests (integration with OLS)
# ---------------------------------------------------------------------------


def test_run_produces_check_result(its_experiment, its_context):
    check = PlaceboInTime(n_folds=2)
    result = check.run(its_experiment, its_context)
    assert isinstance(result, CheckResult)
    assert result.check_name == "PlaceboInTime"
    assert "fold" in result.text.lower()


def test_run_produces_fold_results(its_experiment, its_context):
    check = PlaceboInTime(n_folds=2)
    result = check.run(its_experiment, its_context)
    fold_results = result.metadata["fold_results"]
    assert len(fold_results) == 2
    for fr in fold_results:
        assert isinstance(fr, PlaceboFoldResult)
        assert isinstance(fr.experiment, InterruptedTimeSeries)


def test_fold_treatment_times_are_shifted(its_experiment, its_context):
    check = PlaceboInTime(n_folds=2)
    result = check.run(its_experiment, its_context)
    fold_results = result.metadata["fold_results"]
    for fr in fold_results:
        assert fr.pseudo_treatment_time < its_experiment.treatment_time


def test_single_fold(its_experiment, its_context):
    check = PlaceboInTime(n_folds=1)
    result = check.run(its_experiment, its_context)
    assert len(result.metadata["fold_results"]) == 1


def test_run_passes_when_no_effect():
    """With large random-noise data and intercept-only model, placebo folds
    should find null effects (CI contains zero)."""
    rng = np.random.default_rng(42)
    n = 2000
    df = pd.DataFrame({"t": np.arange(n), "y": rng.normal(size=n)})
    model = _make_model()
    experiment = InterruptedTimeSeries(
        df, treatment_time=1500, formula="y ~ 1", model=model
    )
    ctx = PipelineContext(data=df)
    ctx.experiment = experiment
    ctx.experiment_config = {
        "method": InterruptedTimeSeries,
        "treatment_time": 1500,
        "formula": "y ~ 1",
        "model": _make_model(),
    }
    check = PlaceboInTime(n_folds=2)
    result = check.run(experiment, ctx)
    assert result.passed is True
    assert "PASSED" in result.text


def test_run_fails_when_spurious_effect():
    """When pre-intervention data has a level shift, PlaceboInTime should
    detect the spurious effect and report failure."""
    rng = np.random.default_rng(42)
    n = 200
    y = rng.normal(size=n)
    y[:50] += 5.0
    df = pd.DataFrame({"t": np.arange(n), "y": y})
    model = _make_model()
    experiment = InterruptedTimeSeries(
        df, treatment_time=150, formula="y ~ 1", model=model
    )
    ctx = PipelineContext(data=df)
    ctx.experiment = experiment
    ctx.experiment_config = {
        "method": InterruptedTimeSeries,
        "treatment_time": 150,
        "formula": "y ~ 1",
        "model": _make_model(),
    }
    check = PlaceboInTime(n_folds=2)
    result = check.run(experiment, ctx)
    assert result.passed is False
    assert "FAILED" in result.text


def test_fold_results_contain_effect_summary(its_experiment, its_context):
    check = PlaceboInTime(n_folds=2)
    result = check.run(its_experiment, its_context)
    for fr in result.metadata["fold_results"]:
        assert isinstance(fr.effect_summary, EffectSummary)
        assert fr.effect_summary.table is not None


def test_fold_results_contain_null_verdict(its_experiment, its_context):
    check = PlaceboInTime(n_folds=2)
    result = check.run(its_experiment, its_context)
    for fr in result.metadata["fold_results"]:
        assert fr.effect_is_null is not None
        assert isinstance(fr.effect_is_null, bool)


def test_no_mutable_state_on_check(its_experiment, its_context):
    """Check instance should not store fold_results as mutable state."""
    check = PlaceboInTime(n_folds=2)
    check.run(its_experiment, its_context)
    assert not hasattr(check, "fold_results")


def test_with_custom_factory(its_experiment, its_context):
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
    assert result.passed is not None


def test_no_experiment_config_and_no_factory_raises(its_experiment):
    ctx = PipelineContext(data=its_experiment.data)
    ctx.experiment = its_experiment
    check = PlaceboInTime(n_folds=2)
    with pytest.raises(RuntimeError, match="experiment_config"):
        check.run(its_experiment, ctx)


def test_three_period_its_intervention_length():
    """PlaceboInTime uses treatment_end_time when available."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({"t": np.arange(n), "y": rng.normal(size=n)})
    model = _make_model()
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=100,
        treatment_end_time=130,
        formula="y ~ 1 + t",
        model=model,
    )
    ctx = PipelineContext(data=df)
    ctx.experiment = experiment
    ctx.experiment_config = {
        "method": InterruptedTimeSeries,
        "treatment_time": 100,
        "treatment_end_time": 130,
        "formula": "y ~ 1 + t",
        "model": _make_model(),
    }
    check = PlaceboInTime(n_folds=2)
    result = check.run(experiment, ctx)
    assert isinstance(result, CheckResult)
    fold_results = result.metadata["fold_results"]
    for fr in fold_results:
        assert fr.pseudo_treatment_time < 100


def test_fold_fitting_failure_is_skipped():
    """When the experiment factory raises, the fold is skipped gracefully."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({"t": np.arange(n), "y": rng.normal(size=n)})
    model = _make_model()
    experiment = InterruptedTimeSeries(
        df, treatment_time=150, formula="y ~ 1 + t", model=model
    )

    call_count = 0

    def _failing_factory(data, treatment_time):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Simulated fitting failure")
        return InterruptedTimeSeries(
            data,
            treatment_time=treatment_time,
            formula="y ~ 1 + t",
            model=_make_model(),
        )

    ctx = PipelineContext(data=df)
    ctx.experiment = experiment
    check = PlaceboInTime(n_folds=2, experiment_factory=_failing_factory)
    result = check.run(experiment, ctx)
    assert "SKIPPED" in result.text
    assert "failed to fit" in result.text


def test_skipped_folds_reported_in_text():
    """When folds have too few observations or fail to fit, they are
    reported as skipped in the result text."""
    rng = np.random.default_rng(99)
    n = 10
    df = pd.DataFrame({"t": np.arange(n), "y": rng.normal(size=n)})
    model = _make_model()
    experiment = InterruptedTimeSeries(
        df, treatment_time=8, formula="y ~ 1", model=model
    )
    ctx = PipelineContext(data=df)
    ctx.experiment = experiment
    ctx.experiment_config = {
        "method": InterruptedTimeSeries,
        "treatment_time": 8,
        "formula": "y ~ 1",
        "model": _make_model(),
    }
    # intervention_length = 9 - 8 = 1; with 5 folds, earliest pseudo_tt
    # is 8 - 5*1 = 3. Data up to 3+1=4, so only 4 obs (indices 0-3).
    # The very early folds will have empty pre-treatment data and fail to fit.
    check = PlaceboInTime(n_folds=10)
    result = check.run(experiment, ctx)
    assert "SKIPPED" in result.text


# ---------------------------------------------------------------------------
# Default check registration
# ---------------------------------------------------------------------------


def test_placebo_in_time_registered_as_default():
    """PlaceboInTime should be registered for ITS and SC."""
    its_defaults = _DEFAULT_CHECKS.get(InterruptedTimeSeries, [])
    assert PlaceboInTime in its_defaults

    sc_defaults = _DEFAULT_CHECKS.get(cp.SyntheticControl, [])
    assert PlaceboInTime in sc_defaults


def test_default_for_includes_placebo_in_time():
    step = SensitivityAnalysis.default_for(InterruptedTimeSeries)
    assert any(isinstance(c, PlaceboInTime) for c in step.checks)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_pipeline_with_placebo_in_time():
    rng = np.random.default_rng(42)
    n = 2000
    data = pd.DataFrame({"t": np.arange(n), "y": rng.normal(size=n)})

    result = Pipeline(
        data=data,
        steps=[
            cp.EstimateEffect(
                method=InterruptedTimeSeries,
                treatment_time=1500,
                formula="y ~ 1",
                model=_make_model(),
            ),
            cp.SensitivityAnalysis(
                checks=[PlaceboInTime(n_folds=2)],
            ),
        ],
    ).run()

    assert result.experiment is not None
    assert len(result.sensitivity_results) == 1
    check_result = result.sensitivity_results[0]
    assert check_result.passed is True
    fold_results = check_result.metadata["fold_results"]
    assert len(fold_results) == 2
    for fr in fold_results:
        assert isinstance(fr.effect_summary, EffectSummary)
