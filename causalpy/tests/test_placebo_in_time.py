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
        """Test factory."""
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
    """Test default n folds."""
    check = PlaceboInTime()
    assert check.n_folds == 3


def test_custom_n_folds():
    """Test custom n folds."""
    check = PlaceboInTime(n_folds=5)
    assert check.n_folds == 5


def test_invalid_n_folds():
    """Test invalid n folds."""
    with pytest.raises(ValueError, match="n_folds must be >= 1"):
        PlaceboInTime(n_folds=0)


def test_default_sample_kwargs():
    """Test default sample kwargs."""
    check = PlaceboInTime()
    assert check.sample_kwargs["draws"] == 1000
    assert check.sample_kwargs["chains"] == 4
    assert check.sample_kwargs["target_accept"] == 0.97


def test_custom_sample_kwargs():
    """Test custom sample kwargs."""
    check = PlaceboInTime(sample_kwargs={"draws": 200, "chains": 2})
    assert check.sample_kwargs["draws"] == 200
    assert check.sample_kwargs["chains"] == 2
    assert check.sample_kwargs["target_accept"] == 0.97


def test_default_threshold_and_prior_scale():
    """Test default threshold and prior scale."""
    check = PlaceboInTime()
    assert check.threshold == 0.95
    assert check.prior_scale == 1.0


def test_custom_threshold():
    """Test custom threshold."""
    check = PlaceboInTime(threshold=0.99)
    assert check.threshold == 0.99


def test_custom_prior_scale():
    """Test custom prior scale."""
    check = PlaceboInTime(prior_scale=2.0)
    assert check.prior_scale == 2.0


def test_expected_effect_prior_without_rope_raises():
    """Test expected effect prior without rope raises."""
    with pytest.raises(ValueError, match="rope_half_width is required"):
        PlaceboInTime(expected_effect_prior=np.array([1.0, 2.0, 3.0]))


def test_expected_effect_prior_with_rope_ok():
    """Test expected effect prior with rope ok."""
    check = PlaceboInTime(
        expected_effect_prior=np.array([1.0, 2.0, 3.0]),
        rope_half_width=0.5,
    )
    assert check.rope_half_width == 0.5


def test_satisfies_check_protocol():
    """Test satisfies check protocol."""
    assert isinstance(PlaceboInTime(), Check)


def test_applicable_methods():
    """Test applicable methods."""
    check = PlaceboInTime()
    assert InterruptedTimeSeries in check.applicable_methods
    assert cp.SyntheticControl in check.applicable_methods


def test_repr_basic():
    """Test repr basic."""
    assert "n_folds=3" in repr(PlaceboInTime())


def test_repr_with_assurance():
    """Test repr with assurance."""
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
    """Test validate accepts pymc its."""
    df = _make_its_data()
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=150,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    PlaceboInTime().validate(experiment)


def test_validate_rejects_ols_model():
    """Test validate rejects ols model."""
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
    """Test validate rejects no treatment time."""

    class _FakeExperiment:
        pass

    with pytest.raises(TypeError, match="treatment_time"):
        PlaceboInTime().validate(_FakeExperiment())


# ===========================================================================
# ROPE decision rule tests (unit — no sampling)
# ===========================================================================


def test_rope_decision_positive():
    """Test rope decision positive."""
    samples = np.full(1000, 10.0)
    result = PlaceboInTime.bayesian_rope_decision(samples, 5.0, 0.95)
    assert result == "positive"


def test_rope_decision_null():
    """Test rope decision null."""
    samples = np.full(1000, 0.0)
    result = PlaceboInTime.bayesian_rope_decision(samples, 5.0, 0.95)
    assert result == "null"


def test_rope_decision_indeterminate():
    """Test rope decision indeterminate."""
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=3.0, scale=5.0, size=1000)
    result = PlaceboInTime.bayesian_rope_decision(samples, 5.0, 0.95)
    assert result == "indeterminate"


def test_rope_decision_with_mixed_samples():
    """Test rope decision with mixed samples."""
    samples = np.concatenate([np.full(960, 10.0), np.full(40, 0.0)])
    result = PlaceboInTime.bayesian_rope_decision(samples, 5.0, 0.95)
    assert result == "positive"


def test_rope_decision_barely_below_threshold():
    """Test rope decision barely below threshold."""
    samples = np.concatenate([np.full(940, 10.0), np.full(60, 0.0)])
    result = PlaceboInTime.bayesian_rope_decision(samples, 5.0, 0.95)
    assert result == "indeterminate"


# ===========================================================================
# Cumulative impact extraction (integration — needs PyMC)
# ===========================================================================


@pytest.mark.integration
def test_extract_cumulative_impact(mock_pymc_sample):
    """Test extract cumulative impact."""
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
    """Test run produces check result."""
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
    """Test run produces fold results."""
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
    """Test run metadata contains null distribution."""
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
    """Test fold treatment times are shifted."""
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
    """Test single fold."""
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
    """Test no mutable state on check."""
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
    """Test standalone no factory no context raises."""
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
    """Test text contains hierarchical summary."""
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
    """Test fold fitting failure is skipped."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )

    call_count = 0

    def _failing_factory(data, treatment_time):
        """Factory that raises on first call to test skip logic."""
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
    """Test assurance with numpy array."""
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
    """Test assurance text in report."""
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
    """Test no assurance without prior."""
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


# ===========================================================================
# Random selection mode — construction tests (unit — no sampling)
# ===========================================================================


def test_selection_method_default():
    """Default selection method is sequential."""
    check = PlaceboInTime()
    assert check.selection_method == "sequential"


def test_selection_method_random():
    """Random selection mode stores parameters."""
    check = PlaceboInTime(
        selection_method="random",
        min_training_pct=0.40,
        min_gap=2,
        exclude_periods={"2020-01"},
        random_seed=99,
    )
    assert check.selection_method == "random"
    assert check.min_training_pct == 0.40
    assert check.min_gap == 2
    assert check.exclude_periods == {"2020-01"}


def test_invalid_selection_method():
    """Invalid selection method raises ValueError."""
    with pytest.raises(ValueError, match="selection_method"):
        PlaceboInTime(selection_method="invalid")


def test_invalid_min_training_pct():
    """min_training_pct outside (0, 1) raises ValueError."""
    with pytest.raises(ValueError, match="min_training_pct"):
        PlaceboInTime(selection_method="random", min_training_pct=0.0)
    with pytest.raises(ValueError, match="min_training_pct"):
        PlaceboInTime(selection_method="random", min_training_pct=1.0)


def test_invalid_min_gap():
    """min_gap < 1 raises ValueError."""
    with pytest.raises(ValueError, match="min_gap"):
        PlaceboInTime(selection_method="random", min_gap=0)


def test_repr_random_selection():
    """repr includes selection_method when not sequential."""
    check = PlaceboInTime(n_folds=4, selection_method="random")
    r = repr(check)
    assert "selection_method='random'" in r
    assert "n_folds=4" in r


def test_repr_sequential_omits_selection_method():
    """repr omits selection_method when sequential (default)."""
    check = PlaceboInTime(n_folds=3)
    assert "selection_method" not in repr(check)


# ===========================================================================
# Random fold selection — geometry tests (unit — no sampling)
# ===========================================================================


def test_random_fold_treatment_times_count():
    """Random selection returns exactly n_folds treatment times."""
    n = 200
    data = pd.DataFrame({"y": np.zeros(n)}, index=np.arange(n))
    check = PlaceboInTime(
        n_folds=3,
        selection_method="random",
        min_training_pct=0.20,
        random_seed=42,
    )
    times = check._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=20
    )
    assert len(times) == 3
    # All must be before treatment_time
    assert all(t < 150 for t in times)
    # Sorted
    assert times == sorted(times)


def test_random_fold_treatment_times_reproducible():
    """Same seed produces same selection."""
    n = 200
    data = pd.DataFrame({"y": np.zeros(n)}, index=np.arange(n))
    kwargs = {
        "n_folds": 3,
        "selection_method": "random",
        "min_training_pct": 0.20,
        "random_seed": 42,
    }
    times1 = PlaceboInTime(**kwargs)._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=20
    )
    times2 = PlaceboInTime(**kwargs)._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=20
    )
    assert times1 == times2


def test_random_fold_different_seeds_differ():
    """Different seeds produce different selections."""
    n = 200
    data = pd.DataFrame({"y": np.zeros(n)}, index=np.arange(n))
    times1 = PlaceboInTime(
        n_folds=3,
        selection_method="random",
        min_training_pct=0.20,
        random_seed=42,
    )._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=20
    )
    times2 = PlaceboInTime(
        n_folds=3,
        selection_method="random",
        min_training_pct=0.20,
        random_seed=99,
    )._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=20
    )
    assert times1 != times2


def test_random_fold_respects_min_gap():
    """Selected folds respect the min_gap constraint."""
    n = 200
    data = pd.DataFrame({"y": np.zeros(n)}, index=np.arange(n))
    check = PlaceboInTime(
        n_folds=3,
        selection_method="random",
        min_training_pct=0.10,
        min_gap=5,
        random_seed=42,
    )
    times = check._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=10
    )
    # Gaps between consecutive selected times should be >= min_gap
    # (since they were selected from a candidate list with min_gap spacing)
    for i in range(len(times) - 1):
        assert times[i + 1] - times[i] >= 5


def test_random_fold_respects_exclude_periods():
    """Excluded periods are not selected."""
    n = 200
    data = pd.DataFrame({"y": np.zeros(n)}, index=np.arange(n))
    # Exclude all candidates by excluding every string representation
    exclude = {str(i) for i in range(200)}
    check = PlaceboInTime(
        n_folds=1,
        selection_method="random",
        exclude_periods=exclude,
        random_seed=42,
    )
    with pytest.raises(ValueError, match="eligible candidate"):
        check._compute_random_fold_treatment_times(
            data, treatment_time=150, intervention_length=10
        )


def test_random_fold_too_few_candidates_raises():
    """Raises when there aren't enough eligible candidates."""
    # Very short pre-period
    data = pd.DataFrame({"y": np.zeros(10)}, index=np.arange(10))
    check = PlaceboInTime(
        n_folds=5,
        selection_method="random",
        min_training_pct=0.50,
        random_seed=42,
    )
    with pytest.raises(ValueError, match="eligible candidate"):
        check._compute_random_fold_treatment_times(
            data, treatment_time=8, intervention_length=2
        )


def test_random_fold_with_datetime_index():
    """Random selection works with datetime-indexed data."""
    dates = pd.date_range("2020-01-01", periods=100, freq="MS")
    data = pd.DataFrame({"y": np.zeros(100)}, index=dates)
    treatment = pd.Timestamp("2027-01-01")
    check = PlaceboInTime(
        n_folds=3,
        selection_method="random",
        min_training_pct=0.20,
        exclude_periods={"2020-06"},
        random_seed=42,
    )
    times = check._compute_random_fold_treatment_times(
        data,
        treatment_time=treatment,
        intervention_length=pd.DateOffset(months=6),
    )
    assert len(times) == 3
    assert all(t < treatment for t in times)
    # Excluded month should not appear
    for t in times:
        assert t.strftime("%Y-%m") != "2020-06"


# ===========================================================================
# Random selection — full run (integration — needs PyMC)
# ===========================================================================


@pytest.mark.integration
def test_run_random_selection(mock_pymc_sample):
    """Full run with random selection mode."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        selection_method="random",
        min_training_pct=0.20,
        random_seed=42,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)

    assert isinstance(result, CheckResult)
    assert result.check_name == "PlaceboInTime"
    assert result.passed is not None
    assert len(result.metadata["fold_results"]) == 2
    for fr in result.metadata["fold_results"]:
        assert fr.pseudo_treatment_time < experiment.treatment_time


# ===========================================================================
# Default check registration
# ===========================================================================


def test_placebo_in_time_registered_as_default():
    """Test placebo in time registered as default."""
    its_defaults = _DEFAULT_CHECKS.get(InterruptedTimeSeries, [])
    assert PlaceboInTime in its_defaults

    sc_defaults = _DEFAULT_CHECKS.get(cp.SyntheticControl, [])
    assert PlaceboInTime in sc_defaults


def test_default_for_includes_placebo_in_time():
    """Test default for includes placebo in time."""
    step = SensitivityAnalysis.default_for(InterruptedTimeSeries)
    assert any(isinstance(c, PlaceboInTime) for c in step.checks)


# ===========================================================================
# Pipeline integration
# ===========================================================================


@pytest.mark.integration
def test_pipeline_with_placebo_in_time(mock_pymc_sample):
    """Test pipeline with placebo in time."""
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
    """Test pipeline with assurance."""
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
