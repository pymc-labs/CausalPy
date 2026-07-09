#   Copyright 2025 - 2026 The PyMC Labs Developers
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
Tests for plot utility functions
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from causalpy.plot_utils import concat_x_y, get_hdi_to_df, plot_posterior_histogram


@pytest.mark.integration
def test_get_hdi_to_df_with_coordinate_dimensions():
    """
    Regression test for bug where get_hdi_to_df returned string coordinate values
    instead of numeric HDI values when xarray had named coordinate dimensions.

    This bug manifested in multi-cell synthetic control experiments where columns
    like 'pred_hdi_upper_94' contained the string "treated_agg" instead of
    numeric upper bound values.

    See: https://github.com/pymc-labs/CausalPy/issues/532
    """
    # Create a mock xarray DataArray similar to what's produced in synthetic control
    # with a coordinate dimension like 'treated_units'
    np.random.seed(42)
    n_chains = 2
    n_draws = 100
    n_obs = 10

    # Simulate posterior samples with a named coordinate
    data = np.random.normal(loc=5.0, scale=0.5, size=(n_chains, n_draws, n_obs))

    xr_data = xr.DataArray(
        data,
        dims=["chain", "draw", "obs_ind"],
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "obs_ind": np.arange(n_obs),
            "treated_units": "treated_agg",  # This coordinate caused the bug
        },
    )

    # Call get_hdi_to_df
    result = get_hdi_to_df(xr_data, hdi_prob=0.94)

    # Assertions to verify the bug is fixed
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"

    # Check that we have exactly 2 columns (lower and higher)
    assert result.shape[1] == 2, f"Expected 2 columns, got {result.shape[1]}"

    # Check column names
    assert "lower" in result.columns, "Should have 'lower' column"
    assert "higher" in result.columns, "Should have 'higher' column"

    # CRITICAL: Check that columns contain numeric data, not strings
    assert result["lower"].dtype in [
        np.float64,
        np.float32,
    ], f"'lower' column should be numeric, got {result['lower'].dtype}"
    assert result["higher"].dtype in [
        np.float64,
        np.float32,
    ], f"'higher' column should be numeric, got {result['higher'].dtype}"

    # Check that no string values like 'treated_agg' appear in the data
    assert not (result["lower"].astype(str).str.contains("treated_agg").any()), (
        "'lower' column should not contain coordinate string values"
    )
    assert not (result["higher"].astype(str).str.contains("treated_agg").any()), (
        "'higher' column should not contain coordinate string values"
    )

    # Verify HDI ordering
    assert (result["lower"] <= result["higher"]).all(), (
        "'lower' should be <= 'higher' for all rows"
    )

    # Verify reasonable HDI values (should be around the mean of 5.0)
    assert result["lower"].min() > 3.0, "HDI lower bounds should be reasonable"
    assert result["higher"].max() < 7.0, "HDI upper bounds should be reasonable"


@pytest.fixture
def synthetic_posterior_data():
    """Create synthetic posterior data for histogram tests."""
    np.random.seed(42)
    n_chains = 2
    n_draws = 100
    n_time_points = 20

    # Generate synthetic posterior: trend + noise
    true_mean = (
        10 + 0.1 * np.arange(n_time_points) + 0.02 * np.arange(n_time_points) ** 2
    )

    # Create posterior samples with uncertainty
    rng = np.random.default_rng(seed=42)
    posterior_samples = np.zeros((n_chains, n_draws, n_time_points))

    for chain in range(n_chains):
        for draw in range(n_draws):
            # Add some variation to the mean for each draw
            draw_mean = true_mean + rng.normal(0, 0.5, n_time_points)
            # Add observation-level noise
            posterior_samples[chain, draw, :] = draw_mean + rng.normal(
                0, 1.0, n_time_points
            )

    # Create xarray DataArray with proper dimensions and coordinates
    time_index = pd.date_range(start="2020-01-01", periods=n_time_points, freq="D")
    Y = xr.DataArray(
        posterior_samples,
        dims=["chain", "draw", "obs_ind"],
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "obs_ind": time_index,
        },
    )

    return time_index, Y


@pytest.fixture
def synthetic_posterior_data_numeric():
    """Create synthetic posterior data with numeric x values."""
    np.random.seed(42)
    n_chains = 2
    n_draws = 100
    n_time_points = 20

    # Generate synthetic posterior: trend + noise
    true_mean = (
        10 + 0.1 * np.arange(n_time_points) + 0.02 * np.arange(n_time_points) ** 2
    )

    # Create posterior samples with uncertainty
    rng = np.random.default_rng(seed=42)
    posterior_samples = np.zeros((n_chains, n_draws, n_time_points))

    for chain in range(n_chains):
        for draw in range(n_draws):
            # Add some variation to the mean for each draw
            draw_mean = true_mean + rng.normal(0, 0.5, n_time_points)
            # Add observation-level noise
            posterior_samples[chain, draw, :] = draw_mean + rng.normal(
                0, 1.0, n_time_points
            )

    # Create xarray DataArray with proper dimensions and coordinates
    x = np.arange(n_time_points)
    Y = xr.DataArray(
        posterior_samples,
        dims=["chain", "draw", "obs_ind"],
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "obs_ind": x,
        },
    )

    return x, Y


@pytest.mark.integration
def test_plot_posterior_histogram_renders_heatmap(synthetic_posterior_data):
    x, Y = synthetic_posterior_data
    fig, ax = plt.subplots()
    handles, patch = plot_posterior_histogram(x, Y, ax)
    assert len(handles) == 1
    assert patch is None
    assert ax.collections
    plt.close(fig)


@pytest.mark.integration
def test_plot_posterior_histogram_numeric_x(synthetic_posterior_data_numeric):
    x, Y = synthetic_posterior_data_numeric
    fig, ax = plt.subplots()
    plot_posterior_histogram(x, Y, ax, draw_mean=False)
    assert ax.collections
    plt.close(fig)


@pytest.mark.integration
def test_plot_posterior_histogram_x_length_mismatch_raises(synthetic_posterior_data):
    x, Y = synthetic_posterior_data
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Length of x"):
        plot_posterior_histogram(x[:3], Y, ax)
    plt.close(fig)


@pytest.mark.integration
def test_concat_x_y_histogram_shares_y_bin_edges():
    """Concatenated pre/post posterior uses one y grid spanning both periods."""
    rng = np.random.default_rng(0)
    n_pre, n_post = 8, 6
    x_pre = np.arange(n_pre)
    x_post = np.arange(n_pre, n_pre + n_post)
    Y_pre = xr.DataArray(
        rng.normal(0, 1, (2, 10, n_pre)),
        dims=["chain", "draw", "obs_ind"],
    )
    Y_post = xr.DataArray(
        rng.normal(10, 1, (2, 10, n_post)),
        dims=["chain", "draw", "obs_ind"],
    )

    fig, ax = plt.subplots()
    plot_posterior_histogram(x_pre, Y_pre, ax, draw_mean=False)
    plot_posterior_histogram(x_post, Y_post, ax, draw_mean=False)
    separate_ranges = [
        (coords[:, 1].min(), coords[:, 1].max())
        for col in ax.collections
        if (coords := col.get_coordinates()).size
    ]
    plt.close(fig)

    fig, ax = plt.subplots()
    x_all, Y_all = concat_x_y(x_pre, Y_pre, x_post, Y_post)
    plot_posterior_histogram(x_all, Y_all, ax, draw_mean=False)
    combined_range = ax.collections[0].get_coordinates()[:, 1]
    plt.close(fig)

    assert len(np.asarray(x_all)) == n_pre + n_post
    assert separate_ranges[0][1] < separate_ranges[1][0]  # disjoint y grids
    assert combined_range.min() < separate_ranges[0][0]
    assert combined_range.max() > separate_ranges[1][1]
