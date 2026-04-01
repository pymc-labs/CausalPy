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
"""
Tests for effect_summary with Interrupted Time Series (ITS) designs.

Tests both two-period (permanent intervention) and three-period (temporary intervention)
designs with PyMC and OLS models, covering various parameters and edge cases.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp

# Fast sampling for PyMC tests
sample_kwargs = {
    "chains": 2,
    "draws": 100,
    "tune": 50,
    "progressbar": False,
    "random_seed": 42,
}


@pytest.fixture
def datetime_data_2period(rng):
    """Create datetime-indexed data for two-period design."""
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="W")
    n_weeks = len(dates)

    # Baseline: trend + seasonality + noise
    trend = np.linspace(100, 120, n_weeks)
    season = 10 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    noise = rng.normal(0, 5, n_weeks)
    baseline = trend + season + noise

    # Add permanent intervention effect
    treatment_idx = n_weeks // 2
    y = baseline.copy()
    y[treatment_idx:] += 30  # Permanent effect

    df = pd.DataFrame(
        {
            "y": y,
            "t": np.arange(n_weeks),
            "month": dates.month,
        },
        index=dates,
    )
    return df, dates[treatment_idx]


@pytest.fixture
def datetime_data_3period(rng):
    """Create datetime-indexed data for three-period design."""
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="W")
    n_weeks = len(dates)

    # Baseline: trend + seasonality + noise
    trend = np.linspace(100, 120, n_weeks)
    season = 10 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    noise = rng.normal(0, 5, n_weeks)
    baseline = trend + season + noise

    # Add temporary intervention effect
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
def integer_data_2period(rng):
    """Create integer-indexed data for two-period design."""
    n_points = 100
    indices = np.arange(n_points)

    # Baseline: trend + noise
    trend = np.linspace(0, 10, n_points)
    noise = rng.normal(0, 1, n_points)
    baseline = trend + noise

    # Add permanent intervention effect
    treatment_idx = 50
    y = baseline.copy()
    y[treatment_idx:] += 5  # Permanent effect

    df = pd.DataFrame(
        {
            "y": y,
            "t": indices,
        },
        index=indices,
    )
    return df, treatment_idx


@pytest.fixture
def integer_data_3period(rng):
    """Create integer-indexed data for three-period design."""
    n_points = 100
    indices = np.arange(n_points)

    # Baseline: trend + noise
    trend = np.linspace(0, 10, n_points)
    noise = rng.normal(0, 1, n_points)
    baseline = trend + noise

    # Add temporary intervention effect
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
# Two-Period ITS Tests
# ==============================================================================


@pytest.mark.integration
def test_effect_summary_2period_pymc_datetime(datetime_data_2period, mock_pymc_sample):
    """Test effect_summary for two-period ITS with PyMC model and datetime index."""
    df, treatment_time = datetime_data_2period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Default summary (all post-treatment period)
    summary = result.effect_summary()
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")
    assert isinstance(summary.table, pd.DataFrame)
    assert isinstance(summary.text, str)
    assert len(summary.text) > 0

    # Check table has expected columns for PyMC
    assert "mean" in summary.table.columns
    assert "hdi_lower" in summary.table.columns
    assert "hdi_upper" in summary.table.columns


@pytest.mark.integration
def test_effect_summary_2period_ols_datetime(datetime_data_2period):
    """Test effect_summary for two-period ITS with OLS model and datetime index."""
    df, treatment_time = datetime_data_2period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=LinearRegression(),
    )

    # Default summary
    summary = result.effect_summary()
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")
    assert isinstance(summary.table, pd.DataFrame)

    # Check table has expected columns for OLS
    assert "mean" in summary.table.columns
    assert "ci_lower" in summary.table.columns
    assert "ci_upper" in summary.table.columns


@pytest.mark.integration
def test_effect_summary_2period_pymc_integer(integer_data_2period, mock_pymc_sample):
    """Test effect_summary for two-period ITS with PyMC model and integer index."""
    df, treatment_time = integer_data_2period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        formula="y ~ 1 + t",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    summary = result.effect_summary()
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")


@pytest.mark.integration
def test_effect_summary_2period_ols_integer(integer_data_2period):
    """Test effect_summary for two-period ITS with OLS model and integer index."""
    df, treatment_time = integer_data_2period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        formula="y ~ 1 + t",
        model=LinearRegression(),
    )

    summary = result.effect_summary()
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")


@pytest.mark.integration
def test_effect_summary_2period_with_window(datetime_data_2period, mock_pymc_sample):
    """Test effect_summary for two-period ITS with custom window."""
    df, treatment_time = datetime_data_2period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Use a window that's a subset of post-treatment period
    window_start = df.index[df.index >= treatment_time][5]
    window_end = df.index[df.index >= treatment_time][15]
    summary = result.effect_summary(window=(window_start, window_end))

    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")


@pytest.mark.integration
def test_effect_summary_2period_with_parameters(
    datetime_data_2period, mock_pymc_sample
):
    """Test effect_summary for two-period ITS with various parameters."""
    df, treatment_time = datetime_data_2period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Test with different parameters
    summary1 = result.effect_summary(cumulative=True, relative=True)
    summary2 = result.effect_summary(cumulative=False, relative=False)
    summary3 = result.effect_summary(alpha=0.1, direction="increase")
    summary4 = result.effect_summary(min_effect=1.0)

    for summary in [summary1, summary2, summary3, summary4]:
        assert summary is not None
        assert hasattr(summary, "table")
        assert hasattr(summary, "text")


# ==============================================================================
# Three-Period ITS Tests - Intervention Period
# ==============================================================================


@pytest.mark.integration
def test_effect_summary_3period_intervention_pymc_datetime(
    datetime_data_3period, mock_pymc_sample
):
    """Test effect_summary with period='intervention' for PyMC model and datetime index."""
    df, treatment_time, treatment_end_time = datetime_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    summary = result.effect_summary(period="intervention")
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")
    assert isinstance(summary.table, pd.DataFrame)
    assert isinstance(summary.text, str)

    # Check that text mentions intervention period
    text_lower = summary.text.lower()
    assert "intervention" in text_lower or "during" in text_lower

    # Check table structure
    assert "mean" in summary.table.columns
    assert "hdi_lower" in summary.table.columns
    assert "hdi_upper" in summary.table.columns


@pytest.mark.integration
def test_effect_summary_3period_intervention_ols_datetime(datetime_data_3period):
    """Test effect_summary with period='intervention' for OLS model and datetime index."""
    df, treatment_time, treatment_end_time = datetime_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=LinearRegression(),
    )

    summary = result.effect_summary(period="intervention")
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")

    # Check table structure for OLS
    assert "mean" in summary.table.columns
    assert "ci_lower" in summary.table.columns
    assert "ci_upper" in summary.table.columns


@pytest.mark.integration
def test_effect_summary_3period_intervention_pymc_integer(
    integer_data_3period, mock_pymc_sample
):
    """Test effect_summary with period='intervention' for PyMC model and integer index."""
    df, treatment_time, treatment_end_time = integer_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    summary = result.effect_summary(period="intervention")
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")


@pytest.mark.integration
def test_effect_summary_3period_intervention_ols_integer(integer_data_3period):
    """Test effect_summary with period='intervention' for OLS model and integer index."""
    df, treatment_time, treatment_end_time = integer_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t",
        model=LinearRegression(),
    )

    summary = result.effect_summary(period="intervention")
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")


# ==============================================================================
# Three-Period ITS Tests - Post-Intervention Period
# ==============================================================================


@pytest.mark.integration
def test_effect_summary_3period_post_pymc_datetime(
    datetime_data_3period, mock_pymc_sample
):
    """Test effect_summary with period='post' for PyMC model and datetime index."""
    df, treatment_time, treatment_end_time = datetime_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    summary = result.effect_summary(period="post")
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")

    # Check that text mentions post-intervention period
    text_lower = summary.text.lower()
    assert "post" in text_lower

    # Check table structure
    assert "mean" in summary.table.columns
    assert "hdi_lower" in summary.table.columns
    assert "hdi_upper" in summary.table.columns


@pytest.mark.integration
def test_effect_summary_3period_post_ols_datetime(datetime_data_3period):
    """Test effect_summary with period='post' for OLS model and datetime index."""
    df, treatment_time, treatment_end_time = datetime_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=LinearRegression(),
    )

    summary = result.effect_summary(period="post")
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")

    # Check table structure for OLS
    assert "mean" in summary.table.columns
    assert "ci_lower" in summary.table.columns
    assert "ci_upper" in summary.table.columns


@pytest.mark.integration
def test_effect_summary_3period_post_pymc_integer(
    integer_data_3period, mock_pymc_sample
):
    """Test effect_summary with period='post' for PyMC model and integer index."""
    df, treatment_time, treatment_end_time = integer_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    summary = result.effect_summary(period="post")
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")


@pytest.mark.integration
def test_effect_summary_3period_post_ols_integer(integer_data_3period):
    """Test effect_summary with period='post' for OLS model and integer index."""
    df, treatment_time, treatment_end_time = integer_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t",
        model=LinearRegression(),
    )

    summary = result.effect_summary(period="post")
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")


# ==============================================================================
# Three-Period ITS Tests - Comparison Period
# ==============================================================================


@pytest.mark.integration
def test_effect_summary_3period_comparison_pymc_datetime(
    datetime_data_3period, mock_pymc_sample
):
    """Test effect_summary with period='comparison' for PyMC model and datetime index."""
    df, treatment_time, treatment_end_time = datetime_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    summary = result.effect_summary(period="comparison")
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")

    # Check table structure
    assert isinstance(summary.table, pd.DataFrame)
    assert "intervention" in summary.table.index
    assert "post_intervention" in summary.table.index

    # Check required columns for PyMC
    assert "mean" in summary.table.columns
    assert "hdi_lower" in summary.table.columns
    assert "hdi_upper" in summary.table.columns
    assert "persistence_ratio_pct" in summary.table.columns
    assert "prob_persisted" in summary.table.columns

    # Check text contains key information
    text_lower = summary.text.lower()
    assert "persistence" in text_lower
    assert "intervention" in text_lower


@pytest.mark.integration
def test_effect_summary_3period_comparison_ols_datetime(datetime_data_3period):
    """Test effect_summary with period='comparison' for OLS model and datetime index."""
    df, treatment_time, treatment_end_time = datetime_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=LinearRegression(),
    )

    summary = result.effect_summary(period="comparison")
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")

    # Check table structure for OLS
    assert isinstance(summary.table, pd.DataFrame)
    assert "intervention" in summary.table.index
    assert "post_intervention" in summary.table.index

    # Check required columns for OLS
    assert "mean" in summary.table.columns
    assert "ci_lower" in summary.table.columns
    assert "ci_upper" in summary.table.columns
    assert "persistence_ratio_pct" in summary.table.columns


@pytest.mark.integration
def test_effect_summary_3period_comparison_pymc_integer(
    integer_data_3period, mock_pymc_sample
):
    """Test effect_summary with period='comparison' for PyMC model and integer index."""
    df, treatment_time, treatment_end_time = integer_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    summary = result.effect_summary(period="comparison")
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")

    # Check table structure
    assert "intervention" in summary.table.index
    assert "post_intervention" in summary.table.index
    assert "persistence_ratio_pct" in summary.table.columns


@pytest.mark.integration
def test_effect_summary_3period_comparison_ols_integer(integer_data_3period):
    """Test effect_summary with period='comparison' for OLS model and integer index."""
    df, treatment_time, treatment_end_time = integer_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t",
        model=LinearRegression(),
    )

    summary = result.effect_summary(period="comparison")
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")

    # Check table structure
    assert "intervention" in summary.table.index
    assert "post_intervention" in summary.table.index
    assert "persistence_ratio_pct" in summary.table.columns


# ==============================================================================
# Three-Period ITS Tests - Default Behavior (period=None)
# ==============================================================================


@pytest.mark.integration
def test_effect_summary_3period_default_pymc(datetime_data_3period, mock_pymc_sample):
    """Test effect_summary with period=None (default) for three-period ITS with PyMC."""
    df, treatment_time, treatment_end_time = datetime_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Default should summarize all post-treatment data (backward compatible)
    summary1 = result.effect_summary(period=None)
    summary2 = result.effect_summary()  # Without period parameter

    for summary in [summary1, summary2]:
        assert summary is not None
        assert hasattr(summary, "table")
        assert hasattr(summary, "text")


@pytest.mark.integration
def test_effect_summary_3period_default_ols(datetime_data_3period):
    """Test effect_summary with period=None (default) for three-period ITS with OLS."""
    df, treatment_time, treatment_end_time = datetime_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=LinearRegression(),
    )

    # Default should summarize all post-treatment data
    summary = result.effect_summary()
    assert summary is not None
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")


# ==============================================================================
# Three-Period ITS Tests - Parameters
# ==============================================================================


@pytest.mark.integration
def test_effect_summary_3period_intervention_with_parameters(
    datetime_data_3period, mock_pymc_sample
):
    """Test effect_summary with period='intervention' and various parameters."""
    df, treatment_time, treatment_end_time = datetime_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Test with different parameters
    summary1 = result.effect_summary(
        period="intervention", cumulative=True, relative=True
    )
    summary2 = result.effect_summary(
        period="intervention", cumulative=False, relative=False
    )
    summary3 = result.effect_summary(
        period="intervention", alpha=0.1, direction="increase"
    )
    summary4 = result.effect_summary(period="intervention", min_effect=1.0)

    for summary in [summary1, summary2, summary3, summary4]:
        assert summary is not None
        assert hasattr(summary, "table")
        assert hasattr(summary, "text")


@pytest.mark.integration
def test_effect_summary_3period_post_with_parameters(
    datetime_data_3period, mock_pymc_sample
):
    """Test effect_summary with period='post' and various parameters."""
    df, treatment_time, treatment_end_time = datetime_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Test with different parameters
    summary1 = result.effect_summary(period="post", cumulative=True, relative=True)
    summary2 = result.effect_summary(period="post", cumulative=False, relative=False)
    summary3 = result.effect_summary(period="post", alpha=0.1, direction="increase")

    for summary in [summary1, summary2, summary3]:
        assert summary is not None
        assert hasattr(summary, "table")
        assert hasattr(summary, "text")


# ==============================================================================
# Error Cases
# ==============================================================================


def test_effect_summary_period_without_treatment_end_time_raises_error(
    datetime_data_2period, mock_pymc_sample
):
    """Test that period parameter raises error when treatment_end_time is not provided."""
    df, treatment_time = datetime_data_2period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Should raise error for any period-specific parameter
    with pytest.raises(ValueError, match="treatment_end_time"):
        result.effect_summary(period="intervention")

    with pytest.raises(ValueError, match="treatment_end_time"):
        result.effect_summary(period="post")

    with pytest.raises(ValueError, match="treatment_end_time"):
        result.effect_summary(period="comparison")


def test_effect_summary_invalid_period_raises_error(
    datetime_data_3period, mock_pymc_sample
):
    """Test that invalid period parameter raises ValueError."""
    df, treatment_time, treatment_end_time = datetime_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with pytest.raises(ValueError, match="period must be one of"):
        result.effect_summary(period="invalid")


# ==============================================================================
# Consistency Tests
# ==============================================================================


@pytest.mark.integration
def test_effect_summary_3period_consistency_pymc(
    datetime_data_3period, mock_pymc_sample
):
    """Test that intervention and post summaries are consistent with comparison summary."""
    df, treatment_time, treatment_end_time = datetime_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Get summaries for each period
    intervention_summary = result.effect_summary(period="intervention")
    post_summary = result.effect_summary(period="post")
    comparison_summary = result.effect_summary(period="comparison")

    # Check that comparison summary means match individual period summaries
    intervention_mean_from_comparison = comparison_summary.table.loc[
        "intervention", "mean"
    ]
    post_mean_from_comparison = comparison_summary.table.loc[
        "post_intervention", "mean"
    ]

    intervention_mean_from_period = intervention_summary.table.loc["average", "mean"]
    post_mean_from_period = post_summary.table.loc["average", "mean"]

    # Allow for small floating point differences
    assert abs(intervention_mean_from_comparison - intervention_mean_from_period) < 1e-3
    assert abs(post_mean_from_comparison - post_mean_from_period) < 1e-3


@pytest.mark.integration
def test_effect_summary_3period_consistency_ols(datetime_data_3period):
    """Test that intervention and post summaries are consistent with comparison summary (OLS)."""
    df, treatment_time, treatment_end_time = datetime_data_3period

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=LinearRegression(),
    )

    # Get summaries for each period
    intervention_summary = result.effect_summary(period="intervention")
    post_summary = result.effect_summary(period="post")
    comparison_summary = result.effect_summary(period="comparison")

    # Check that comparison summary means match individual period summaries
    intervention_mean_from_comparison = comparison_summary.table.loc[
        "intervention", "mean"
    ]
    post_mean_from_comparison = comparison_summary.table.loc[
        "post_intervention", "mean"
    ]

    intervention_mean_from_period = intervention_summary.table.loc["average", "mean"]
    post_mean_from_period = post_summary.table.loc["average", "mean"]

    # Allow for small floating point differences
    assert abs(intervention_mean_from_comparison - intervention_mean_from_period) < 1e-3
    assert abs(post_mean_from_comparison - post_mean_from_period) < 1e-3
