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
"""Tests for PlaceboInTime hierarchical null model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.checks.base import Check, CheckResult
from causalpy.checks.placebo_in_time import (
    AssuranceResult,
    PlaceboFoldResult,
    PlaceboInTime,
)
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.pipeline import Pipeline, PipelineContext
from causalpy.steps.sensitivity import _DEFAULT_CHECKS, SensitivityAnalysis

# ---------------------------------------------------------------------------
# Shared sample_kwargs for fast PyMC tests
# ---------------------------------------------------------------------------

_FAST_SAMPLE_KWARGS = {
    "chains": 2,
    "draws": 100,
    "progressbar": False,
    "random_seed": 42,
}

_FAST_HIERARCHICAL_KWARGS = {
    "chains": 2,
    "draws": 50,
    "tune": 50,
    "progressbar": False,
    "random_seed": 42,
}


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


def _make_pymc_model():
    """Create a minimal PyMC model for ITS tests."""
    return cp.pymc_models.LinearRegression(sample_kwargs=_FAST_SAMPLE_KWARGS)


def _make_ols_model():
    """Create a minimal OLS-compatible model for validation tests."""
    return cp.create_causalpy_compatible_class(LinearRegression())


def _make_pymc_factory():
    """Factory that creates PyMC ITS experiments."""

    def factory(data, treatment_time):
        return InterruptedTimeSeries(
            data,
            treatment_time=treatment_time,
            formula="y ~ 1 + t",
            model=_make_pymc_model(),
        )

    return factory


# ===========================================================================
# Construction tests (unit — no sampling)
# ===========================================================================


def test_default_n_folds():
    check = PlaceboInTime()
    assert check.n_folds == 3


def test_custom_n_folds():
    check = PlaceboInTime(n_folds=5)
    assert check.n_folds == 5


def test_invalid_n_folds():
    with pytest.raises(ValueError, match="n_folds must be >= 1"):
        PlaceboInTime(n_folds=0)


def test_default_sample_kwargs():
    check = PlaceboInTime()
    assert check.sample_kwargs["draws"] == 1000
    assert check.sample_kwargs["chains"] == 4
    assert check.sample_kwargs["target_accept"] == 0.97


def test_custom_sample_kwargs():
    check = PlaceboInTime(sample_kwargs={"draws": 200, "chains": 2})
    assert check.sample_kwargs["draws"] == 200
    assert check.sample_kwargs["chains"] == 2
    assert check.sample_kwargs["target_accept"] == 0.97


def test_default_threshold_and_prior_scale():
    check = PlaceboInTime()
    assert check.threshold == 0.95
    assert check.prior_scale == 1.0


def test_custom_threshold():
    check = PlaceboInTime(threshold=0.99)
    assert check.threshold == 0.99


def test_custom_prior_scale():
    check = PlaceboInTime(prior_scale=2.0)
    assert check.prior_scale == 2.0


def test_expected_effect_prior_without_rope_raises():
    with pytest.raises(ValueError, match="rope_half_width is required"):
        PlaceboInTime(expected_effect_prior=np.array([1.0, 2.0, 3.0]))


def test_expected_effect_prior_with_rope_ok():
    check = PlaceboInTime(
        expected_effect_prior=np.array([1.0, 2.0, 3.0]),
        rope_half_width=0.5,
    )
    assert check.rope_half_width == 0.5


def test_satisfies_check_protocol():
    assert isinstance(PlaceboInTime(), Check)


def test_applicable_methods():
    check = PlaceboInTime()
    assert InterruptedTimeSeries in check.applicable_methods
    assert cp.SyntheticControl in check.applicable_methods


def test_repr_basic():
    assert "n_folds=3" in repr(PlaceboInTime())


def test_repr_with_assurance():
    check = PlaceboInTime(
        expected_effect_prior=np.array([1.0]),
        rope_half_width=0.5,
    )
    assert "assurance=True" in repr(check)


# ===========================================================================
# Validation tests (unit — no sampling)
# ===========================================================================


@pytest.mark.integration
def test_validate_accepts_pymc_its(mock_pymc_sample):
    df = _make_its_data()
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=150,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    PlaceboInTime().validate(experiment)


def test_validate_rejects_ols_model():
    df = _make_its_data()
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=150,
        formula="y ~ 1 + t",
        model=_make_ols_model(),
    )
    with pytest.raises(TypeError, match="PyMC model"):
        PlaceboInTime().validate(experiment)


def test_validate_rejects_no_treatment_time():
    class _FakeExperiment:
        pass

    with pytest.raises(TypeError, match="treatment_time"):
        PlaceboInTime().validate(_FakeExperiment())


# ===========================================================================
# ROPE decision rule tests (unit — no sampling)
# ===========================================================================


def test_rope_decision_positive():
    samples = np.full(1000, 10.0)
    result = PlaceboInTime.bayesian_rope_decision(samples, 5.0, 0.95)
    assert result == "positive"


def test_rope_decision_null():
    samples = np.full(1000, 0.0)
    result = PlaceboInTime.bayesian_rope_decision(samples, 5.0, 0.95)
    assert result == "null"


def test_rope_decision_indeterminate():
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=3.0, scale=5.0, size=1000)
    result = PlaceboInTime.bayesian_rope_decision(samples, 5.0, 0.95)
    assert result == "indeterminate"


def test_rope_decision_with_mixed_samples():
    samples = np.concatenate([np.full(960, 10.0), np.full(40, 0.0)])
    result = PlaceboInTime.bayesian_rope_decision(samples, 5.0, 0.95)
    assert result == "positive"


def test_rope_decision_barely_below_threshold():
    samples = np.concatenate([np.full(940, 10.0), np.full(60, 0.0)])
    result = PlaceboInTime.bayesian_rope_decision(samples, 5.0, 0.95)
    assert result == "indeterminate"


# ===========================================================================
# Cumulative impact extraction (integration — needs PyMC)
# ===========================================================================


@pytest.mark.integration
def test_extract_cumulative_impact(mock_pymc_sample):
    df = _make_its_data()
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=150,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    cumulative = PlaceboInTime._extract_cumulative_impact(experiment)

    assert isinstance(cumulative, xr.DataArray)
    assert "sample" in cumulative.dims
    assert cumulative.sizes["sample"] > 0


# ===========================================================================
# Full run tests (integration — needs PyMC)
# ===========================================================================


@pytest.mark.integration
def test_run_produces_check_result(mock_pymc_sample):
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)

    assert isinstance(result, CheckResult)
    assert result.check_name == "PlaceboInTime"
    assert result.passed is not None
    assert "fold" in result.text.lower()


@pytest.mark.integration
def test_run_produces_fold_results(mock_pymc_sample):
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)
    fold_results = result.metadata["fold_results"]

    assert len(fold_results) == 2
    for fr in fold_results:
        assert isinstance(fr, PlaceboFoldResult)
        assert isinstance(fr.experiment, InterruptedTimeSeries)
        assert isinstance(fr.fold_mean, float)
        assert isinstance(fr.fold_sd, float)
        assert fr.cumulative_impact_samples is not None


@pytest.mark.integration
def test_run_metadata_contains_null_distribution(mock_pymc_sample):
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)

    assert "null_samples" in result.metadata
    assert "status_quo_idata" in result.metadata
    assert "actual_cumulative_mean" in result.metadata
    assert "p_effect_outside_null" in result.metadata

    null_samples = result.metadata["null_samples"]
    assert isinstance(null_samples, np.ndarray)
    assert len(null_samples) > 0

    p = result.metadata["p_effect_outside_null"]
    assert 0.0 <= p <= 1.0


@pytest.mark.integration
def test_fold_treatment_times_are_shifted(mock_pymc_sample):
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)
    for fr in result.metadata["fold_results"]:
        assert fr.pseudo_treatment_time < experiment.treatment_time


@pytest.mark.integration
def test_single_fold(mock_pymc_sample):
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=1,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)
    assert len(result.metadata["fold_results"]) == 1


@pytest.mark.integration
def test_no_mutable_state_on_check(mock_pymc_sample):
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    check.run(experiment)
    assert not hasattr(check, "fold_results")


@pytest.mark.integration
def test_standalone_run_without_context(mock_pymc_sample):
    """Standalone use: pass experiment_factory, no PipelineContext."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)
    assert isinstance(result, CheckResult)
    assert result.passed is not None


@pytest.mark.integration
def test_standalone_no_factory_no_context_raises(mock_pymc_sample):
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(n_folds=2, sample_kwargs=_FAST_HIERARCHICAL_KWARGS)
    with pytest.raises(RuntimeError, match="experiment_config"):
        check.run(experiment)


@pytest.mark.integration
def test_run_with_context(mock_pymc_sample):
    """Pipeline-style use: context provides experiment_config."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    ctx = PipelineContext(data=df)
    ctx.experiment = experiment
    ctx.experiment_config = {
        "method": InterruptedTimeSeries,
        "treatment_time": 1500,
        "formula": "y ~ 1 + t",
        "model": _make_pymc_model(),
    }
    check = PlaceboInTime(n_folds=2, sample_kwargs=_FAST_HIERARCHICAL_KWARGS)
    result = check.run(experiment, ctx)
    assert isinstance(result, CheckResult)
    assert result.passed is not None


@pytest.mark.integration
def test_text_contains_hierarchical_summary(mock_pymc_sample):
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)
    assert "mu=" in result.text
    assert "tau=" in result.text
    assert "P(actual outside null)" in result.text


@pytest.mark.integration
def test_fold_fitting_failure_is_skipped(mock_pymc_sample):
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
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
            model=_make_pymc_model(),
        )

    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_failing_factory,
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)
    assert "SKIPPED" in result.text
    assert "failed to fit" in result.text


# ===========================================================================
# Assurance tests (integration — needs PyMC)
# ===========================================================================


@pytest.mark.integration
def test_assurance_with_numpy_array(mock_pymc_sample):
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
        expected_effect_prior=np.random.default_rng(42).normal(90, 15, size=200),
        rope_half_width=50.0,
        random_seed=42,
    )
    result = check.run(experiment)

    assert "assurance_result" in result.metadata
    assert "assurance" in result.metadata

    ar = result.metadata["assurance_result"]
    assert isinstance(ar, AssuranceResult)
    assert 0.0 <= ar.true_positive_rate <= 1.0
    assert 0.0 <= ar.false_positive_rate <= 1.0
    assert 0.0 <= ar.true_negative_rate <= 1.0
    assert 0.0 <= ar.false_negative_rate <= 1.0

    null_sum = (
        ar.false_positive_rate + ar.true_negative_rate + ar.null_indeterminate_rate
    )
    assert abs(null_sum - 1.0) < 0.01

    alt_sum = ar.true_positive_rate + ar.false_negative_rate + ar.alt_indeterminate_rate
    assert abs(alt_sum - 1.0) < 0.01


@pytest.mark.integration
def test_assurance_with_rvs_object(mock_pymc_sample):
    """Test that expected_effect_prior with .rvs() method works."""

    class _MockDistribution:
        def rvs(self, n):
            return np.random.default_rng(42).normal(90, 15, size=n)

    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
        expected_effect_prior=_MockDistribution(),
        rope_half_width=50.0,
        random_seed=42,
    )
    result = check.run(experiment)
    assert "assurance" in result.metadata
    assert isinstance(result.metadata["assurance"], float)


@pytest.mark.integration
def test_assurance_text_in_report(mock_pymc_sample):
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
        expected_effect_prior=np.full(200, 90.0),
        rope_half_width=50.0,
        random_seed=42,
    )
    result = check.run(experiment)
    assert "Bayesian assurance" in result.text
    assert "Assurance (TP rate)" in result.text
    assert "False Positive rate" in result.text


@pytest.mark.integration
def test_no_assurance_without_prior(mock_pymc_sample):
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)
    assert "assurance_result" not in result.metadata
    assert "assurance" not in result.metadata
    assert "Bayesian assurance" not in result.text


# ===========================================================================
# Default check registration
# ===========================================================================


def test_placebo_in_time_registered_as_default():
    its_defaults = _DEFAULT_CHECKS.get(InterruptedTimeSeries, [])
    assert PlaceboInTime in its_defaults

    sc_defaults = _DEFAULT_CHECKS.get(cp.SyntheticControl, [])
    assert PlaceboInTime in sc_defaults


def test_default_for_includes_placebo_in_time():
    step = SensitivityAnalysis.default_for(InterruptedTimeSeries)
    assert any(isinstance(c, PlaceboInTime) for c in step.checks)


# ===========================================================================
# Pipeline integration
# ===========================================================================


@pytest.mark.integration
def test_pipeline_with_placebo_in_time(mock_pymc_sample):
    n = 2000
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"t": np.arange(n), "y": rng.normal(size=n)})

    result = Pipeline(
        data=data,
        steps=[
            cp.EstimateEffect(
                method=InterruptedTimeSeries,
                treatment_time=1500,
                formula="y ~ 1 + t",
                model=_make_pymc_model(),
            ),
            cp.SensitivityAnalysis(
                checks=[
                    PlaceboInTime(
                        n_folds=2,
                        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
                    )
                ],
            ),
        ],
    ).run()

    assert result.experiment is not None
    assert len(result.sensitivity_results) == 1
    check_result = result.sensitivity_results[0]
    assert check_result.passed is not None
    assert "null_samples" in check_result.metadata
    fold_results = check_result.metadata["fold_results"]
    assert len(fold_results) == 2
    for fr in fold_results:
        assert isinstance(fr, PlaceboFoldResult)


@pytest.mark.integration
def test_pipeline_with_assurance(mock_pymc_sample):
    n = 2000
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"t": np.arange(n), "y": rng.normal(size=n)})

    result = Pipeline(
        data=data,
        steps=[
            cp.EstimateEffect(
                method=InterruptedTimeSeries,
                treatment_time=1500,
                formula="y ~ 1 + t",
                model=_make_pymc_model(),
            ),
            cp.SensitivityAnalysis(
                checks=[
                    PlaceboInTime(
                        n_folds=2,
                        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
                        expected_effect_prior=np.full(200, 90.0),
                        rope_half_width=50.0,
                        random_seed=42,
                    )
                ],
            ),
        ],
    ).run()

    check_result = result.sensitivity_results[0]
    assert "assurance" in check_result.metadata
    assert isinstance(check_result.metadata["assurance_result"], AssuranceResult)
