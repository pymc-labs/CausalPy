#   Copyright 2022 - 2025 The PyMC Labs Developers
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
"""
Tests for three-period Interrupted Time Series design.

Tests the extension of InterruptedTimeSeries to support temporary interventions
with pre-intervention, intervention, and post-intervention periods.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.custom_exceptions import BadIndexException

# Fast sampling for PyMC tests
sample_kwargs = {
    "chains": 2,
    "draws": 100,
    "tune": 50,
    "progressbar": False,
    "random_seed": 42,
}


@pytest.fixture
def datetime_data(rng):
    """Create datetime-indexed data with three periods."""
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="W")
    n_weeks = len(dates)

    # Baseline: trend + seasonality + noise
    trend = np.linspace(100, 120, n_weeks)
    season = 10 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    noise = rng.normal(0, 5, n_weeks)
    baseline = trend + season + noise

    # Add intervention effect
    treatment_idx = n_weeks // 2
    treatment_end_idx = treatment_idx + 12  # 12 weeks

    y = baseline.copy()
    y[treatment_idx:treatment_end_idx] += 50  # During intervention
    y[treatment_end_idx:] += 15  # Post-intervention (persistence)

    df = pd.DataFrame(
        {
            "y": y,
            "t": np.arange(n_weeks),
            "month": dates.month,
        },
        index=dates,
    )
    return df, dates[treatment_idx], dates[treatment_end_idx]


@pytest.fixture
def integer_data(rng):
    """Create integer-indexed data with three periods."""
    n_points = 100
    indices = np.arange(n_points)

    # Baseline: trend + noise
    trend = np.linspace(0, 10, n_points)
    noise = rng.normal(0, 1, n_points)
    baseline = trend + noise

    # Add intervention effect
    treatment_idx = 50
    treatment_end_idx = 60

    y = baseline.copy()
    y[treatment_idx:treatment_end_idx] += 5  # During intervention
    y[treatment_end_idx:] += 1.5  # Post-intervention (persistence)

    df = pd.DataFrame(
        {
            "y": y,
            "t": indices,
        },
        index=indices,
    )
    return df, treatment_idx, treatment_end_idx


# ==============================================================================
# 4.2.1 Basic Functionality
# ==============================================================================


@pytest.mark.integration
def test_three_period_pymc_datetime_index(datetime_data, mock_pymc_sample):
    """Test three-period design with PyMC model and datetime index."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    assert isinstance(result, cp.InterruptedTimeSeries)
    assert result.treatment_end_time == treatment_end_time

    # Check all new attributes exist (same for all model types)
    assert hasattr(result, "data_intervention")
    assert hasattr(result, "data_post_intervention")
    assert hasattr(result, "intervention_pred")
    assert hasattr(result, "post_intervention_pred")
    assert hasattr(result, "intervention_impact")
    assert hasattr(result, "post_intervention_impact")
    assert hasattr(result, "intervention_impact_cumulative")
    assert hasattr(result, "post_intervention_impact_cumulative")

    # Check data splits
    assert len(result.data_intervention) > 0
    assert len(result.data_post_intervention) > 0
    assert isinstance(result.data_intervention, pd.DataFrame)
    assert isinstance(result.data_post_intervention, pd.DataFrame)

    # Check PyMC-specific types
    import arviz as az
    import xarray as xr

    assert isinstance(result.intervention_pred, az.InferenceData)
    assert isinstance(result.post_intervention_pred, az.InferenceData)
    # For PyMC models, post_impact is always xarray DataArray
    assert isinstance(result.intervention_impact, xr.DataArray)
    assert isinstance(result.post_intervention_impact, xr.DataArray)


@pytest.mark.integration
def test_three_period_pymc_integer_index(integer_data, mock_pymc_sample):
    """Test three-period design with PyMC model and integer index."""
    df, treatment_time, treatment_end_time = integer_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    assert isinstance(result, cp.InterruptedTimeSeries)
    assert result.treatment_end_time == treatment_end_time

    # Check all new attributes exist (same for all model types)
    assert hasattr(result, "data_intervention")
    assert hasattr(result, "data_post_intervention")
    assert hasattr(result, "intervention_pred")
    assert hasattr(result, "post_intervention_pred")
    assert hasattr(result, "intervention_impact")
    assert hasattr(result, "post_intervention_impact")
    assert hasattr(result, "intervention_impact_cumulative")
    assert hasattr(result, "post_intervention_impact_cumulative")

    # Check data splits
    assert len(result.data_intervention) > 0
    assert len(result.data_post_intervention) > 0
    assert isinstance(result.data_intervention, pd.DataFrame)
    assert isinstance(result.data_post_intervention, pd.DataFrame)

    # Check PyMC-specific types
    import arviz as az
    import xarray as xr

    assert isinstance(result.intervention_pred, az.InferenceData)
    assert isinstance(result.post_intervention_pred, az.InferenceData)
    # For PyMC models, post_impact is always xarray DataArray
    assert isinstance(result.intervention_impact, xr.DataArray)
    assert isinstance(result.post_intervention_impact, xr.DataArray)


@pytest.mark.integration
def test_three_period_sklearn_datetime_index(datetime_data):
    """Test three-period design with sklearn model and datetime index."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=LinearRegression(),
    )

    assert isinstance(result, cp.InterruptedTimeSeries)
    assert result.treatment_end_time == treatment_end_time

    # Check all new attributes exist (same for all model types)
    assert hasattr(result, "data_intervention")
    assert hasattr(result, "data_post_intervention")
    assert hasattr(result, "intervention_pred")
    assert hasattr(result, "post_intervention_pred")
    assert hasattr(result, "intervention_impact")
    assert hasattr(result, "post_intervention_impact")
    assert hasattr(result, "intervention_impact_cumulative")
    assert hasattr(result, "post_intervention_impact_cumulative")

    # Check data splits
    assert len(result.data_intervention) > 0
    assert len(result.data_post_intervention) > 0
    assert isinstance(result.data_intervention, pd.DataFrame)
    assert isinstance(result.data_post_intervention, pd.DataFrame)

    # Check sklearn-specific types
    assert isinstance(result.intervention_pred, np.ndarray)
    assert isinstance(result.post_intervention_pred, np.ndarray)
    # For sklearn models, post_impact is also xarray DataArray (for consistency)
    import xarray as xr

    assert isinstance(result.intervention_impact, xr.DataArray)
    assert isinstance(result.post_intervention_impact, xr.DataArray)


@pytest.mark.integration
def test_three_period_sklearn_integer_index(integer_data):
    """Test three-period design with sklearn model and integer index."""
    df, treatment_time, treatment_end_time = integer_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t",
        model=LinearRegression(),
    )

    assert isinstance(result, cp.InterruptedTimeSeries)
    assert result.treatment_end_time == treatment_end_time

    # Check all new attributes exist (same for all model types)
    assert hasattr(result, "data_intervention")
    assert hasattr(result, "data_post_intervention")
    assert hasattr(result, "intervention_pred")
    assert hasattr(result, "post_intervention_pred")
    assert hasattr(result, "intervention_impact")
    assert hasattr(result, "post_intervention_impact")
    assert hasattr(result, "intervention_impact_cumulative")
    assert hasattr(result, "post_intervention_impact_cumulative")

    # Check data splits
    assert len(result.data_intervention) > 0
    assert len(result.data_post_intervention) > 0
    assert isinstance(result.data_intervention, pd.DataFrame)
    assert isinstance(result.data_post_intervention, pd.DataFrame)

    # Check sklearn-specific types
    assert isinstance(result.intervention_pred, np.ndarray)
    assert isinstance(result.post_intervention_pred, np.ndarray)
    # For sklearn models, post_impact is also xarray DataArray (for consistency)
    import xarray as xr

    assert isinstance(result.intervention_impact, xr.DataArray)
    assert isinstance(result.post_intervention_impact, xr.DataArray)


# ==============================================================================
# 4.2.2 Backward Compatibility
# ==============================================================================


@pytest.mark.integration
def test_backward_compatibility_no_treatment_end_time(datetime_data, mock_pymc_sample):
    """Test that treatment_end_time=None maintains two-period behavior."""
    df, treatment_time, _ = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    assert isinstance(result, cp.InterruptedTimeSeries)
    assert result.treatment_end_time is None

    # Check that new attributes don't exist
    assert not hasattr(result, "data_intervention")
    assert not hasattr(result, "data_post_intervention")

    # Check existing attributes still work
    assert hasattr(result, "datapre")
    assert hasattr(result, "datapost")
    assert hasattr(result, "post_pred")
    assert hasattr(result, "post_impact")


@pytest.mark.integration
def test_existing_methods_work_without_treatment_end_time(
    datetime_data, mock_pymc_sample
):
    """Test that existing methods work without modification."""
    df, treatment_time, _ = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # These should all work
    result.summary()
    fig, ax = result.plot()
    assert fig is not None
    plot_data = result.get_plot_data()
    assert isinstance(plot_data, pd.DataFrame)


# ==============================================================================
# 4.2.3 Effect Summary
# ==============================================================================


@pytest.mark.integration
def test_effect_summary_intervention_period(datetime_data, mock_pymc_sample):
    """Test effect_summary with period='intervention'."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(period="intervention")
    assert stats is not None
    assert hasattr(stats, "table")
    assert hasattr(stats, "text")
    assert "intervention" in stats.text.lower() or "during" in stats.text.lower()


@pytest.mark.integration
def test_effect_summary_post_period(datetime_data, mock_pymc_sample):
    """Test effect_summary with period='post'."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(period="post")
    assert stats is not None
    assert hasattr(stats, "table")
    assert hasattr(stats, "text")
    assert "post" in stats.text.lower()


@pytest.mark.integration
def test_effect_summary_default_behavior(datetime_data, mock_pymc_sample):
    """Test effect_summary with period=None (default behavior)."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Default should summarize all post-treatment data (backward compatible)
    stats = result.effect_summary(period=None)
    assert stats is not None

    # Without period parameter should also work
    stats2 = result.effect_summary()
    assert stats2 is not None


@pytest.mark.integration
def test_effect_summary_comparison_pymc(datetime_data, mock_pymc_sample):
    """Test that period='comparison' provides comparative summary with persistence metrics."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    comparison_summary = result.effect_summary(period="comparison")

    # Check that summary is returned (not NotImplementedError)
    assert comparison_summary is not None
    assert hasattr(comparison_summary, "table")
    assert hasattr(comparison_summary, "text")

    # Check table structure
    assert isinstance(comparison_summary.table, pd.DataFrame)
    assert "intervention" in comparison_summary.table.index
    assert "post_intervention" in comparison_summary.table.index

    # Check required columns
    assert "mean" in comparison_summary.table.columns
    assert "hdi_lower" in comparison_summary.table.columns
    assert "hdi_upper" in comparison_summary.table.columns
    assert "persistence_ratio_pct" in comparison_summary.table.columns
    assert "prob_persisted" in comparison_summary.table.columns

    # Check text contains key information
    assert "persistence" in comparison_summary.text.lower()
    assert (
        "post-intervention" in comparison_summary.text.lower()
        or "post intervention" in comparison_summary.text.lower()
    )
    assert "intervention" in comparison_summary.text.lower()


@pytest.mark.integration
def test_effect_summary_comparison_sklearn(datetime_data):
    """Test that period='comparison' works with sklearn models."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=LinearRegression(),
    )

    comparison_summary = result.effect_summary(period="comparison")

    # Check that summary is returned
    assert comparison_summary is not None
    assert hasattr(comparison_summary, "table")
    assert hasattr(comparison_summary, "text")

    # Check table structure
    assert isinstance(comparison_summary.table, pd.DataFrame)
    assert "intervention" in comparison_summary.table.index
    assert "post_intervention" in comparison_summary.table.index

    # Check required columns (OLS uses CI, not HDI)
    assert "mean" in comparison_summary.table.columns
    assert "ci_lower" in comparison_summary.table.columns
    assert "ci_upper" in comparison_summary.table.columns
    assert "persistence_ratio_pct" in comparison_summary.table.columns


@pytest.mark.integration
def test_effect_summary_comparison_persistence_ratio(datetime_data, mock_pymc_sample):
    """Test that comparison period calculates persistence ratio correctly."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    comparison_summary = result.effect_summary(period="comparison")

    # Get persistence ratio from table (in post_intervention row)
    persistence_ratio_pct = comparison_summary.table.loc[
        "post_intervention", "persistence_ratio_pct"
    ]

    # Calculate expected ratio from intervention and post means
    intervention_mean = comparison_summary.table.loc["intervention", "mean"]
    post_mean = comparison_summary.table.loc["post_intervention", "mean"]

    expected_ratio_pct = (post_mean / intervention_mean) * 100

    # Allow for small floating point differences
    assert abs(persistence_ratio_pct - expected_ratio_pct) < 1e-6


@pytest.mark.integration
def test_effect_summary_comparison_prob_persisted(datetime_data, mock_pymc_sample):
    """Test that comparison period calculates probability that effect persisted."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    comparison_summary = result.effect_summary(period="comparison")

    # Check that prob_persisted is in the table (in post_intervention row)
    assert "prob_persisted" in comparison_summary.table.columns
    prob_persisted = comparison_summary.table.loc["post_intervention", "prob_persisted"]

    # Probability should be between 0 and 1
    assert 0 <= prob_persisted <= 1


@pytest.mark.integration
def test_effect_summary_comparison_hdi_intervals(datetime_data, mock_pymc_sample):
    """Test that comparison period includes HDI intervals in text."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    comparison_summary = result.effect_summary(period="comparison")

    # Check that HDI intervals are in the table
    intervention_lower = comparison_summary.table.loc["intervention", "hdi_lower"]
    intervention_upper = comparison_summary.table.loc["intervention", "hdi_upper"]
    post_lower = comparison_summary.table.loc["post_intervention", "hdi_lower"]
    post_upper = comparison_summary.table.loc["post_intervention", "hdi_upper"]

    # Check that intervals are valid (lower < upper)
    assert intervention_lower < intervention_upper
    assert post_lower < post_upper

    # Check that text mentions HDI intervals
    assert "hdi" in comparison_summary.text.lower()
    assert "persistence" in comparison_summary.text.lower()


@pytest.mark.integration
def test_effect_summary_invalid_period_raises_error(datetime_data, mock_pymc_sample):
    """Test that invalid period parameter raises ValueError."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with pytest.raises(ValueError, match="period"):
        result.effect_summary(period="invalid")


# ==============================================================================
# 4.2.4 Validation
# ==============================================================================


def test_treatment_end_time_less_than_treatment_time_raises_error(datetime_data):
    """Test that treatment_end_time <= treatment_time raises ValueError."""
    df, treatment_time, _ = datetime_data

    # treatment_end_time before treatment_time
    with pytest.raises(ValueError, match="must be greater"):
        cp.InterruptedTimeSeries(
            df,
            treatment_time=treatment_time,
            treatment_end_time=treatment_time - pd.Timedelta(days=1),
            formula="y ~ 1 + t + C(month)",
            model=LinearRegression(),
        )

    # treatment_end_time equal to treatment_time
    with pytest.raises(ValueError, match="must be greater"):
        cp.InterruptedTimeSeries(
            df,
            treatment_time=treatment_time,
            treatment_end_time=treatment_time,
            formula="y ~ 1 + t + C(month)",
            model=LinearRegression(),
        )


def test_treatment_end_time_beyond_data_range_raises_error(datetime_data):
    """Test that treatment_end_time beyond data range raises ValueError."""
    df, treatment_time, _ = datetime_data

    future_time = df.index.max() + pd.Timedelta(days=100)

    with pytest.raises(ValueError, match="beyond the data range"):
        cp.InterruptedTimeSeries(
            df,
            treatment_time=treatment_time,
            treatment_end_time=future_time,
            formula="y ~ 1 + t + C(month)",
            model=LinearRegression(),
        )


def test_index_type_mismatch_datetime_raises_error(datetime_data):
    """Test that index type mismatches raise BadIndexException."""
    df, treatment_time, treatment_end_time = datetime_data

    # treatment_end_time as integer when index is datetime
    with pytest.raises(BadIndexException):
        cp.InterruptedTimeSeries(
            df,
            treatment_time=treatment_time,
            treatment_end_time=100,  # Wrong type
            formula="y ~ 1 + t + C(month)",
            model=LinearRegression(),
        )


def test_index_type_mismatch_integer_raises_error(integer_data):
    """Test that index type mismatches raise BadIndexException."""
    df, treatment_time, treatment_end_time = integer_data

    # treatment_end_time as Timestamp when index is integer
    with pytest.raises(BadIndexException):
        cp.InterruptedTimeSeries(
            df,
            treatment_time=treatment_time,
            treatment_end_time=pd.Timestamp("2024-01-01"),  # Wrong type
            formula="y ~ 1 + t",
            model=LinearRegression(),
        )


# ==============================================================================
# 4.2.5 Edge Cases
# ==============================================================================


@pytest.mark.integration
def test_very_short_post_intervention_period(datetime_data, mock_pymc_sample):
    """Test with very short post-intervention period."""
    df, treatment_time, _ = datetime_data

    # treatment_end_time very close to end of data
    treatment_end_time = df.index.max() - pd.Timedelta(days=1)

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    assert len(result.data_post_intervention) > 0
    assert len(result.data_post_intervention) < len(result.data_intervention)


@pytest.mark.integration
def test_treatment_end_time_at_data_boundary(datetime_data, mock_pymc_sample):
    """Test with treatment_end_time at data boundary."""
    df, treatment_time, _ = datetime_data

    # treatment_end_time at the last data point
    treatment_end_time = df.index.max()

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Post-intervention should be empty or very small
    assert len(result.data_post_intervention) <= 1


# ==============================================================================
# 4.2.6 Attributes
# ==============================================================================


@pytest.mark.integration
def test_all_new_attributes_exist(datetime_data, mock_pymc_sample):
    """Test that all new attributes exist when treatment_end_time is provided."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Data attributes
    assert hasattr(result, "data_intervention")
    assert hasattr(result, "data_post_intervention")
    assert isinstance(result.data_intervention, pd.DataFrame)
    assert isinstance(result.data_post_intervention, pd.DataFrame)

    # Prediction attributes
    assert hasattr(result, "intervention_pred")
    assert hasattr(result, "post_intervention_pred")

    # Impact attributes
    assert hasattr(result, "intervention_impact")
    assert hasattr(result, "post_intervention_impact")

    # Cumulative impact attributes
    assert hasattr(result, "intervention_impact_cumulative")
    assert hasattr(result, "post_intervention_impact_cumulative")


@pytest.mark.integration
def test_data_splits_no_overlap(datetime_data, mock_pymc_sample):
    """Test that data splits have no overlap and complete coverage."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Check no overlap
    intervention_indices = set(result.data_intervention.index)
    post_intervention_indices = set(result.data_post_intervention.index)
    assert len(intervention_indices & post_intervention_indices) == 0

    # Check complete coverage
    all_post_indices = intervention_indices | post_intervention_indices
    datapost_indices = set(result.datapost.index)
    assert all_post_indices == datapost_indices


@pytest.mark.integration
def test_cumulative_impacts_calculated_correctly(datetime_data, mock_pymc_sample):
    """Test that cumulative impact attributes are calculated correctly."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Cumulative impacts should exist and have correct shape
    assert result.intervention_impact_cumulative is not None
    assert result.post_intervention_impact_cumulative is not None

    # For PyMC, check dimensions
    if hasattr(result.intervention_impact_cumulative, "dims"):
        assert "obs_ind" in result.intervention_impact_cumulative.dims


@pytest.mark.integration
def test_intervention_pred_is_slice_of_post_pred(datetime_data, mock_pymc_sample):
    """Test that intervention_pred is a slice of post_pred, not a new computation."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # For PyMC models, check that intervention_pred is InferenceData
    assert hasattr(result.intervention_pred, "posterior_predictive")

    # Extract mu from both
    intervention_mu = result.intervention_pred.posterior_predictive["mu"]
    post_mu = result.post_pred.posterior_predictive["mu"]

    # Check that intervention_mu is a subset of post_mu
    intervention_coords = result.data_intervention.index
    post_mu_intervention = post_mu.sel(obs_ind=intervention_coords)

    # They should have the same shape
    assert intervention_mu.shape == post_mu_intervention.shape


# ==============================================================================
# 5.1 Persistence Analysis Methods
# ==============================================================================


@pytest.mark.integration
def test_analyze_persistence_pymc(datetime_data, mock_pymc_sample):
    """Test analyze_persistence() with PyMC model."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    persistence = result.analyze_persistence()

    # Check structure
    assert isinstance(persistence, dict)
    assert "mean_effect_during" in persistence
    assert "mean_effect_post" in persistence
    assert "persistence_ratio" in persistence
    assert "total_effect_during" in persistence
    assert "total_effect_post" in persistence

    # Check persistence ratio is a decimal (>= 0, can exceed 1 if post-effect > intervention-effect)
    assert isinstance(persistence["persistence_ratio"], (int, float))
    assert persistence["persistence_ratio"] >= 0
    # Note: persistence_ratio can be > 1 if post-intervention effect is larger than intervention effect

    # Check values are reasonable
    assert persistence["mean_effect_during"] is not None
    assert persistence["mean_effect_post"] is not None
    assert persistence["persistence_ratio"] is not None
    assert persistence["total_effect_during"] is not None
    assert persistence["total_effect_post"] is not None


@pytest.mark.integration
def test_analyze_persistence_sklearn(datetime_data):
    """Test analyze_persistence() with sklearn model."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=LinearRegression(),
    )

    persistence = result.analyze_persistence()

    # Check structure
    assert isinstance(persistence, dict)
    assert "mean_effect_during" in persistence
    assert "mean_effect_post" in persistence
    assert "persistence_ratio" in persistence
    assert "total_effect_during" in persistence
    assert "total_effect_post" in persistence

    # Check persistence ratio is a decimal (>= 0, can exceed 1 if post-effect > intervention-effect)
    assert isinstance(persistence["persistence_ratio"], (int, float))
    assert persistence["persistence_ratio"] >= 0
    # Note: persistence_ratio can be > 1 if post-intervention effect is larger than intervention effect

    # Check values are reasonable
    assert persistence["mean_effect_during"] is not None
    assert persistence["mean_effect_post"] is not None
    assert persistence["persistence_ratio"] is not None
    assert persistence["total_effect_during"] is not None
    assert persistence["total_effect_post"] is not None


def test_analyze_persistence_raises_error_without_treatment_end_time(
    datetime_data, mock_pymc_sample
):
    """Test that analyze_persistence() raises error when treatment_end_time is None."""
    df, treatment_time, _ = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with pytest.raises(ValueError, match="treatment_end_time"):
        result.analyze_persistence()


@pytest.mark.integration
def test_analyze_persistence_with_custom_hdi_prob(datetime_data, mock_pymc_sample):
    """Test analyze_persistence() with custom hdi_prob parameter."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    persistence = result.analyze_persistence(hdi_prob=0.90)

    # Check that results are returned (method prints internally)
    assert "mean_effect_during" in persistence
    assert "mean_effect_post" in persistence
    assert "persistence_ratio" in persistence


@pytest.mark.integration
def test_analyze_persistence_persistence_ratio_calculation(
    datetime_data, mock_pymc_sample
):
    """Test that persistence ratio is calculated correctly."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    persistence = result.analyze_persistence()

    # Persistence ratio should be post_mean / intervention_mean (as decimal, not percentage)
    expected_ratio = persistence["mean_effect_post"] / persistence["mean_effect_during"]

    # Allow for small floating point differences
    assert abs(persistence["persistence_ratio"] - expected_ratio) < 1e-6
