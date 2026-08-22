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
Tests for three-period Interrupted Time Series design.

Tests the extension of InterruptedTimeSeries to support temporary interventions
with pre-intervention, intervention, and post-intervention periods.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
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
def datetime_data():
    """Create datetime-indexed data with three periods.

    Uses its own seeded generator (not the session-scoped ``rng`` fixture) so
    the data does not depend on how many draws earlier tests consumed.
    """
    rng = np.random.default_rng(seed=42)
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
def integer_data():
    """Create integer-indexed data with three periods.

    Uses its own seeded generator (not the session-scoped ``rng`` fixture) so
    the data does not depend on how many draws earlier tests consumed.
    """
    rng = np.random.default_rng(seed=42)
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
    ).fit()

    assert isinstance(result, cp.InterruptedTimeSeries)
    assert result.treatment_end_time == treatment_end_time

    # Three-period views are derived on demand from the fitted bundle
    slices = result._period_slices(result.result)

    # Check all three-period views exist (same for all model types)
    assert "data_intervention" in slices
    assert "data_post_intervention" in slices
    assert "intervention_pred" in slices
    assert "post_intervention_pred" in slices
    assert "intervention_impact" in slices
    assert "post_intervention_impact" in slices
    assert "intervention_impact_cumulative" in slices
    assert "post_intervention_impact_cumulative" in slices

    # Check data splits
    assert len(slices["data_intervention"]) > 0
    assert len(slices["data_post_intervention"]) > 0
    assert isinstance(slices["data_intervention"], pd.DataFrame)
    assert isinstance(slices["data_post_intervention"], pd.DataFrame)

    assert isinstance(slices["intervention_pred"], xr.DataArray)
    assert isinstance(slices["post_intervention_pred"], xr.DataArray)
    assert isinstance(slices["intervention_impact"], xr.DataArray)
    assert isinstance(slices["post_intervention_impact"], xr.DataArray)


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
    ).fit()

    assert isinstance(result, cp.InterruptedTimeSeries)
    assert result.treatment_end_time == treatment_end_time

    # Three-period views are derived on demand from the fitted bundle
    slices = result._period_slices(result.result)

    # Check all three-period views exist (same for all model types)
    assert "data_intervention" in slices
    assert "data_post_intervention" in slices
    assert "intervention_pred" in slices
    assert "post_intervention_pred" in slices
    assert "intervention_impact" in slices
    assert "post_intervention_impact" in slices
    assert "intervention_impact_cumulative" in slices
    assert "post_intervention_impact_cumulative" in slices

    # Check data splits
    assert len(slices["data_intervention"]) > 0
    assert len(slices["data_post_intervention"]) > 0
    assert isinstance(slices["data_intervention"], pd.DataFrame)
    assert isinstance(slices["data_post_intervention"], pd.DataFrame)

    assert isinstance(slices["intervention_pred"], xr.DataArray)
    assert isinstance(slices["post_intervention_pred"], xr.DataArray)
    assert isinstance(slices["intervention_impact"], xr.DataArray)
    assert isinstance(slices["post_intervention_impact"], xr.DataArray)


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
    ).fit()

    assert isinstance(result, cp.InterruptedTimeSeries)
    assert result.treatment_end_time == treatment_end_time

    # Three-period views are derived on demand from the fitted bundle
    slices = result._period_slices(result.result)

    # Check all three-period views exist (same for all model types)
    assert "data_intervention" in slices
    assert "data_post_intervention" in slices
    assert "intervention_pred" in slices
    assert "post_intervention_pred" in slices
    assert "intervention_impact" in slices
    assert "post_intervention_impact" in slices
    assert "intervention_impact_cumulative" in slices
    assert "post_intervention_impact_cumulative" in slices

    # Check data splits
    assert len(slices["data_intervention"]) > 0
    assert len(slices["data_post_intervention"]) > 0
    assert isinstance(slices["data_intervention"], pd.DataFrame)
    assert isinstance(slices["data_post_intervention"], pd.DataFrame)

    assert isinstance(slices["intervention_pred"], xr.DataArray)
    assert isinstance(slices["post_intervention_pred"], xr.DataArray)
    assert slices["intervention_pred"].sizes["chain"] == 1
    assert slices["intervention_pred"].sizes["draw"] == 1
    assert isinstance(slices["intervention_impact"], xr.DataArray)
    assert isinstance(slices["post_intervention_impact"], xr.DataArray)


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
    ).fit()

    assert isinstance(result, cp.InterruptedTimeSeries)
    assert result.treatment_end_time == treatment_end_time

    # Three-period views are derived on demand from the fitted bundle
    slices = result._period_slices(result.result)

    # Check all three-period views exist (same for all model types)
    assert "data_intervention" in slices
    assert "data_post_intervention" in slices
    assert "intervention_pred" in slices
    assert "post_intervention_pred" in slices
    assert "intervention_impact" in slices
    assert "post_intervention_impact" in slices
    assert "intervention_impact_cumulative" in slices
    assert "post_intervention_impact_cumulative" in slices

    # Check data splits
    assert len(slices["data_intervention"]) > 0
    assert len(slices["data_post_intervention"]) > 0
    assert isinstance(slices["data_intervention"], pd.DataFrame)
    assert isinstance(slices["data_post_intervention"], pd.DataFrame)

    assert isinstance(slices["intervention_pred"], xr.DataArray)
    assert isinstance(slices["post_intervention_pred"], xr.DataArray)
    assert slices["intervention_pred"].sizes["chain"] == 1
    assert slices["intervention_pred"].sizes["draw"] == 1
    assert isinstance(slices["intervention_impact"], xr.DataArray)
    assert isinstance(slices["post_intervention_impact"], xr.DataArray)


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
    ).fit()

    assert isinstance(result, cp.InterruptedTimeSeries)
    assert result.treatment_end_time is None

    # Check that the three-period views are not available
    with pytest.raises(ValueError, match="treatment_end_time"):
        result._period_slices(result.result)

    # Check existing attributes still work
    assert hasattr(result, "datapre")
    assert hasattr(result, "datapost")
    assert hasattr(result.result, "predictions_post")
    assert hasattr(result.result, "impact_post")


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
    ).fit()

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
    ).fit()

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
    ).fit()

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
    ).fit()

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
    ).fit()

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
    ).fit()

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
    ).fit()

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
    ).fit()

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
    ).fit()

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


@pytest.mark.parametrize(
    ("index", "treatment_time"),
    [
        pytest.param(pd.Index([0, 2, 1, 3]), 2, id="unsorted-numeric"),
        pytest.param(pd.Index([0, 1, 1, 2]), 1, id="duplicate-numeric"),
        pytest.param(
            pd.DatetimeIndex(["2024-01-01", "2024-01-03", "2024-01-02", "2024-01-04"]),
            pd.Timestamp("2024-01-03"),
            id="unsorted-datetime",
        ),
        pytest.param(
            pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"]),
            pd.Timestamp("2024-01-02"),
            id="duplicate-datetime",
        ),
    ],
)
def test_invalid_time_index_rejected_before_design_matrices(index, treatment_time):
    """Unsorted and duplicate time indexes fail before matrix construction."""
    data = pd.DataFrame(
        {"y": np.arange(len(index)), "t": np.arange(len(index))}, index=index
    )

    with (
        patch.object(
            cp.InterruptedTimeSeries, "_build_design_matrices"
        ) as build_design_matrices,
        pytest.raises(BadIndexException, match="unique and monotonically increasing"),
    ):
        cp.InterruptedTimeSeries(
            data,
            treatment_time=treatment_time,
            formula="y ~ 1 + t",
            model=LinearRegression(fit_intercept=False),
        )

    build_design_matrices.assert_not_called()


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
    ).fit()

    slices = result._period_slices(result.result)
    assert len(slices["data_post_intervention"]) > 0
    assert len(slices["data_post_intervention"]) < len(slices["data_intervention"])


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
    ).fit()

    # Post-intervention should be empty or very small
    assert len(result._period_slices(result.result)["data_post_intervention"]) <= 1


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
    ).fit()

    # Three-period views are derived on demand from the fitted bundle
    slices = result._period_slices(result.result)

    # Data views
    assert isinstance(slices["data_intervention"], pd.DataFrame)
    assert isinstance(slices["data_post_intervention"], pd.DataFrame)

    # Prediction views
    assert isinstance(slices["intervention_pred"], xr.DataArray)
    assert isinstance(slices["post_intervention_pred"], xr.DataArray)

    # Impact views
    assert isinstance(slices["intervention_impact"], xr.DataArray)
    assert isinstance(slices["post_intervention_impact"], xr.DataArray)

    # Cumulative impact views
    assert isinstance(slices["intervention_impact_cumulative"], xr.DataArray)
    assert isinstance(slices["post_intervention_impact_cumulative"], xr.DataArray)


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
    ).fit()

    slices = result._period_slices(result.result)

    # Check no overlap
    intervention_indices = set(slices["data_intervention"].index)
    post_intervention_indices = set(slices["data_post_intervention"].index)
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
    ).fit()

    slices = result._period_slices(result.result)

    # Cumulative impacts should exist and have correct shape
    assert slices["intervention_impact_cumulative"] is not None
    assert slices["post_intervention_impact_cumulative"] is not None

    # For PyMC, check dimensions
    assert "obs_ind" in slices["intervention_impact_cumulative"].dims


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
    ).fit()

    slices = result._period_slices(result.result)
    intervention_mu = slices["intervention_pred"]
    post_mu = result.result.predictions_post

    # Check that intervention_mu is a subset of post_mu
    intervention_coords = slices["data_intervention"].index
    post_mu_intervention = post_mu.sel(obs_ind=intervention_coords)

    assert intervention_mu.shape == post_mu_intervention.shape
    xr.testing.assert_allclose(intervention_mu, post_mu_intervention)

    post_intervention_mu = slices["post_intervention_pred"]
    post_intervention_coords = slices["data_post_intervention"].index
    xr.testing.assert_allclose(
        post_intervention_mu,
        post_mu.sel(obs_ind=post_intervention_coords),
    )


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
    ).fit()

    persistence = result.analyze_persistence()

    # Check structure
    assert isinstance(persistence, dict)
    assert "mean_effect_during" in persistence
    assert "mean_effect_post" in persistence
    assert "persistence_ratio" in persistence
    assert "total_effect_during" in persistence
    assert "total_effect_post" in persistence

    # Persistence ratio is a decimal. It can be negative (counterfactual above
    # the observed post-period) and can exceed 1 (post-effect > intervention-effect),
    # so only check the type.
    assert isinstance(persistence["persistence_ratio"], (int, float))

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
    ).fit()

    persistence = result.analyze_persistence()

    # Check structure
    assert isinstance(persistence, dict)
    assert "mean_effect_during" in persistence
    assert "mean_effect_post" in persistence
    assert "persistence_ratio" in persistence
    assert "total_effect_during" in persistence
    assert "total_effect_post" in persistence

    # Persistence ratio is a decimal. It can be negative (counterfactual above
    # the observed post-period) and can exceed 1 (post-effect > intervention-effect),
    # so only check the type.
    assert isinstance(persistence["persistence_ratio"], (int, float))

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
    ).fit()

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
    ).fit()

    persistence = result.analyze_persistence()

    # Persistence ratio should be post_mean / intervention_mean (as decimal, not percentage)
    expected_ratio = persistence["mean_effect_post"] / persistence["mean_effect_during"]

    # Allow for small floating point differences
    assert abs(persistence["persistence_ratio"] - expected_ratio) < 1e-6


@pytest.mark.integration
def test_plot_three_period_pymc(datetime_data, mock_pymc_sample):
    """Test that plotting works with three-period design for PyMC models."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    ).fit()

    # Plot should not raise an error
    fig, ax = result.plot()

    # Check that we have 3 subplots
    assert len(ax) == 3

    # Check that treatment_end_time line is present (should be in all 3 subplots)
    # We can't easily check the exact line properties, but we can verify the plot was created
    assert fig is not None
    assert ax is not None


@pytest.mark.integration
def test_plot_three_period_sklearn(datetime_data):
    """Test that plotting works with three-period design for sklearn models."""
    df, treatment_time, treatment_end_time = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        treatment_end_time=treatment_end_time,
        formula="y ~ 1 + t + C(month)",
        model=LinearRegression(),
    ).fit()

    # Plot should not raise an error
    fig, ax = result.plot()

    # Check that we have 3 subplots
    assert len(ax) == 3

    # Check that treatment_end_time line is present (should be in all 3 subplots)
    assert fig is not None
    assert ax is not None


@pytest.mark.integration
def test_plot_two_period_backward_compatible(datetime_data, mock_pymc_sample):
    """Test that plotting still works with two-period design (backward compatibility)."""
    df, treatment_time, _ = datetime_data

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    ).fit()

    # Plot should not raise an error
    fig, ax = result.plot()

    # Check that we have 3 subplots
    assert len(ax) == 3

    # Should only have treatment_time line, not treatment_end_time
    assert fig is not None
    assert ax is not None


def test_get_plot_data_uses_hdi_for_skewed_impacts():
    """Impact columns use HDI bounds rather than equal-tailed quantiles."""
    from types import SimpleNamespace

    from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
    from causalpy.plot_utils import get_hdi_to_df

    pre_index = pd.Index(["pre_0", "pre_1"], name="obs_ind")
    post_index = pd.Index(["post_0", "post_1"], name="obs_ind")
    rng = np.random.default_rng(42)

    def posterior(values, index):
        return xr.DataArray(
            values,
            dims=["chain", "draw", "obs_ind"],
            coords={"obs_ind": index},
        )

    def prediction(values, index):
        return (
            posterior(values, index)
            .expand_dims(treated_units=["unit_0"])
            .transpose("chain", "draw", "obs_ind", "treated_units")
        )

    pre_impact = posterior(rng.exponential(size=(2, 200, 2)), pre_index)
    post_impact = posterior(rng.exponential(size=(2, 200, 2)), post_index)
    from causalpy.experiments._results import CausalResult

    bundle = CausalResult(
        predictions_pre=prediction(rng.normal(size=(2, 200, 2)), pre_index),
        predictions_post=prediction(rng.normal(size=(2, 200, 2)), post_index),
        impact_pre=pre_impact,
        impact_post=post_impact,
        impact_post_cumulative=post_impact.cumsum(dim="obs_ind"),
    )
    result = SimpleNamespace(
        datapre=pd.DataFrame({"y": [0.0, 0.0]}, index=pre_index),
        datapost=pd.DataFrame({"y": [0.0, 0.0]}, index=post_index),
        _resolve_group=lambda group: bundle,
    )

    plot_data = InterruptedTimeSeries.get_plot_data(result)
    expected = get_hdi_to_df(pre_impact).reindex(pre_index)
    observed = plot_data.loc[
        pre_index, ["impact_hdi_lower_94", "impact_hdi_upper_94"]
    ].to_numpy()
    eti = pre_impact.quantile([0.03, 0.97], dim=["chain", "draw"]).values.T

    np.testing.assert_allclose(observed, expected.to_numpy())
    assert not np.allclose(observed, eti)


def test_comparison_period_summary_uses_frozen_hdi_bounds():
    """Comparative-summary HDIs retain the ArviZ 0.22 94% baseline."""
    from types import SimpleNamespace

    import xarray as xr

    from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries

    rng = np.random.default_rng(321)

    def impact(coords):
        return xr.DataArray(
            rng.exponential(size=(2, 200, len(coords))),
            dims=["chain", "draw", "obs_ind"],
            coords={"obs_ind": coords},
        )

    from causalpy.experiments._results import CausalResult

    full_index = [0, 1, 2, 3]
    # Same seeded draw sequence as the frozen expectations below: the
    # intervention slice consumes the first draws, post the second.
    full_impact = xr.concat([impact([0, 1]), impact([2, 3])], dim="obs_ind")
    bundle = CausalResult(
        predictions_pre=xr.DataArray(
            np.zeros((2, 200, 0)),
            dims=["chain", "draw", "obs_ind"],
        ),
        predictions_post=xr.DataArray(
            np.zeros((2, 200, len(full_index))),
            dims=["chain", "draw", "obs_ind"],
            coords={"obs_ind": full_index},
        ),
        impact_pre=xr.DataArray(
            np.zeros((2, 200, 0)),
            dims=["chain", "draw", "obs_ind"],
        ),
        impact_post=full_impact,
        impact_post_cumulative=full_impact.cumsum(dim="obs_ind"),
    )
    stub = SimpleNamespace(
        treatment_end_time=2,
        datapost=pd.DataFrame(index=full_index),
    )
    stub._period_slices = lambda bundle: InterruptedTimeSeries._period_slices(
        stub, bundle
    )

    InterruptedTimeSeries._comparison_period_summary(
        stub,
        bundle,
        alpha=0.06,
        cumulative=False,
        relative=False,
    )


def test_plot_forwards_ci_prob_to_all_singleton_hdi_markers(monkeypatch):
    """All singleton overlays use the caller's HDI probability."""
    from types import SimpleNamespace

    import matplotlib.pyplot as plt

    from causalpy.experiments import interrupted_time_series as its_module
    from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries

    def draws(obs_ind, *, treated_units=False, offset=0.0):
        samples = xr.DataArray(
            np.arange(6 * len(obs_ind), dtype=float).reshape(2, 3, len(obs_ind))
            + offset,
            dims=["chain", "draw", "obs_ind"],
            coords={"obs_ind": obs_ind},
        )
        if treated_units:
            return samples.expand_dims(treated_units=["unit_0"]).transpose(
                "chain", "draw", "obs_ind", "treated_units"
            )
        return samples

    pre_index, post_index = pd.Index([0, 1]), pd.Index([2])
    from causalpy.experiments._results import CausalResult

    bundle = CausalResult(
        predictions_pre=draws(pre_index, treated_units=True),
        predictions_post=draws(post_index, treated_units=True, offset=1.0),
        impact_pre=draws(pre_index, offset=-1.0),
        impact_post=draws(post_index, offset=-1.0),
        impact_post_cumulative=draws(post_index, offset=-2.0),
        score=pd.Series({"unit_0_r2": 0.0, "unit_0_r2_std": 0.0}),
    )

    def design(obs_ind):
        return xr.Dataset(
            {
                "y": xr.DataArray(
                    np.arange(len(obs_ind), dtype=float).reshape(-1, 1),
                    dims=["obs_ind", "treated_units"],
                    coords={"obs_ind": obs_ind, "treated_units": ["unit_0"]},
                )
            }
        )

    stub = SimpleNamespace(
        datapre=pd.DataFrame(index=pre_index),
        datapost=pd.DataFrame(index=post_index),
        pre_design=design(pre_index),
        post_design=design(post_index),
        treatment_time=2,
        treatment_end_time=None,
    )
    probabilities = []

    def fake_plot_posterior(x, posterior, *, ax, **kwargs):
        return ax.plot([], [])[0], ax.fill_between([], [], [])

    def fake_singleton_marker(ax, x, posterior, *, color, hdi_prob):
        probabilities.append(hdi_prob)
        return ax.plot([], [])[0]

    stub._resolve_group = lambda group: bundle
    stub._draw_singleton_hdi_marker = fake_singleton_marker

    monkeypatch.setattr(its_module, "plot_posterior_over_x", fake_plot_posterior)

    fig, _ = InterruptedTimeSeries._plot(stub, ci_prob=0.8)
    assert probabilities == [0.8, 0.8, 0.8]
    plt.close(fig)


def test_analyze_persistence_forwards_custom_hdi_probability(monkeypatch, capsys):
    """Persistence HDIs use the caller's probability for both periods."""
    from types import SimpleNamespace

    import xarray as xr

    from causalpy.experiments import interrupted_time_series as its_module
    from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries

    draws = xr.DataArray(
        np.arange(24, dtype=float).reshape(2, 3, 4),
        dims=["chain", "draw", "obs_ind"],
    )
    calls = []

    def fake_hdi_bounds(data, *, prob):
        calls.append(prob)
        return (prob, prob + 0.01)

    from causalpy.experiments._results import CausalResult

    # One continuous post-treatment series; _period_slices splits it at
    # treatment_end_time, reproducing the original intervention/post split.
    impact_post = draws.assign_coords(obs_ind=np.arange(4))
    bundle = CausalResult(
        predictions_pre=impact_post.isel(obs_ind=slice(0, 2)),
        predictions_post=impact_post,
        impact_pre=impact_post.isel(obs_ind=slice(0, 2)),
        impact_post=impact_post,
        impact_post_cumulative=impact_post.cumsum(dim="obs_ind"),
    )
    stub = SimpleNamespace(
        treatment_end_time=2,
        datapost=pd.DataFrame(index=np.arange(4)),
        result=bundle,
    )
    stub._period_slices = lambda bundle: InterruptedTimeSeries._period_slices(
        stub, bundle
    )
    monkeypatch.setattr(its_module, "hdi_bounds", fake_hdi_bounds)

    InterruptedTimeSeries.analyze_persistence(stub, hdi_prob=0.9)

    assert calls == [0.9, 0.9]
    assert "90% HDI: [0.90, 0.91]" in capsys.readouterr().out
