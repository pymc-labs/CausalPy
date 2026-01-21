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
from matplotlib.collections import PolyCollection

from causalpy.plot_utils import get_hdi_to_df, plot_xY


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
    """Create synthetic posterior data for testing plot_xY."""
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
def test_plot_xY_ribbon_hdi(synthetic_posterior_data):
    """Test ribbon plot with HDI (default behavior)."""
    x, Y = synthetic_posterior_data
    fig, ax = plt.subplots()

    h_line, h_patch = plot_xY(
        x,
        Y,
        ax=ax,
        kind="ribbon",
        ci_kind="hdi",
        ci_prob=0.94,
        label="Test HDI",
    )

    # Check return types
    assert isinstance(h_line, plt.Line2D), "Should return Line2D for mean line"
    assert h_patch is not None, "Should return PolyCollection for HDI ribbon"
    assert isinstance(h_patch, PolyCollection), (
        "Should return PolyCollection for HDI ribbon"
    )

    # Check that plot was created
    assert len(ax.lines) > 0, "Should have at least one line (mean)"
    assert len(ax.collections) > 0, "Should have at least one collection (HDI ribbon)"

    plt.close(fig)


@pytest.mark.integration
def test_plot_xY_ribbon_eti(synthetic_posterior_data):
    """Test ribbon plot with ETI."""
    x, Y = synthetic_posterior_data
    fig, ax = plt.subplots()

    h_line, h_patch = plot_xY(
        x,
        Y,
        ax=ax,
        kind="ribbon",
        ci_kind="eti",
        ci_prob=0.89,
        label="Test ETI",
    )

    # Check return types
    assert isinstance(h_line, plt.Line2D), "Should return Line2D for mean line"
    assert h_patch is not None, "Should return PolyCollection for ETI ribbon"
    assert isinstance(h_patch, PolyCollection), (
        "Should return PolyCollection for ETI ribbon"
    )

    # Check that plot was created
    assert len(ax.lines) > 0, "Should have at least one line (mean)"
    assert len(ax.collections) > 0, "Should have at least one collection (ETI ribbon)"

    plt.close(fig)


@pytest.mark.integration
def test_plot_xY_histogram(synthetic_posterior_data):
    """Test histogram visualization (2D heatmap)."""
    x, Y = synthetic_posterior_data
    fig, ax = plt.subplots()

    handles, patch = plot_xY(
        x,
        Y,
        ax=ax,
        kind="histogram",
        label="Test Histogram",
    )

    # Check return types
    assert isinstance(handles, list), "Should return list of handles"
    assert len(handles) > 0, "Should have at least one handle"
    assert patch is None, "Histogram should not return PolyCollection"

    # Check that plot was created
    # Histogram uses pcolormesh which creates a QuadMesh
    assert len(ax.collections) > 0 or len(ax.images) > 0, (
        "Should have collections or images (pcolormesh/heatmap)"
    )
    assert len(ax.lines) > 0, "Should have at least one line (mean overlay)"

    plt.close(fig)


@pytest.mark.integration
def test_plot_xY_histogram_numeric(synthetic_posterior_data_numeric):
    """Test histogram visualization with numeric x values."""
    x, Y = synthetic_posterior_data_numeric
    fig, ax = plt.subplots()

    handles, patch = plot_xY(
        x,
        Y,
        ax=ax,
        kind="histogram",
        label="Test Histogram Numeric",
    )

    # Check return types
    assert isinstance(handles, list), "Should return list of handles"
    assert len(handles) > 0, "Should have at least one handle"
    assert patch is None, "Histogram should not return PolyCollection"

    # Check that plot was created
    assert len(ax.collections) > 0 or len(ax.images) > 0, (
        "Should have collections or images (pcolormesh/heatmap)"
    )
    assert len(ax.lines) > 0, "Should have at least one line (mean overlay)"

    plt.close(fig)


@pytest.mark.integration
def test_plot_xY_spaghetti(synthetic_posterior_data):
    """Test spaghetti plot visualization."""
    x, Y = synthetic_posterior_data
    fig, ax = plt.subplots()

    handles, patch = plot_xY(
        x,
        Y,
        ax=ax,
        kind="spaghetti",
        num_samples=30,
        label="Test Spaghetti",
    )

    # Check return types
    assert isinstance(handles, list), "Should return list of handles"
    assert len(handles) > 0, "Should have at least one handle"
    assert patch is None, "Spaghetti should not return PolyCollection"

    # Check that plot was created
    # Spaghetti plot should have multiple lines (samples + mean)
    assert len(ax.lines) >= 30, "Should have at least 30 lines (samples + mean)"

    plt.close(fig)


@pytest.mark.integration
def test_plot_xY_default_behavior(synthetic_posterior_data):
    """Test that default behavior matches original (ribbon with HDI)."""
    x, Y = synthetic_posterior_data
    fig, ax = plt.subplots()

    # Test with no parameters (should default to ribbon, hdi, 0.94)
    h_line, h_patch = plot_xY(x, Y, ax=ax)

    # Check return types
    assert isinstance(h_line, plt.Line2D), "Should return Line2D for mean line"
    assert h_patch is not None, "Should return PolyCollection for HDI ribbon"

    plt.close(fig)


@pytest.mark.integration
def test_plot_xY_backward_compatibility_hdi_prob(synthetic_posterior_data):
    """Test backward compatibility with hdi_prob parameter."""
    x, Y = synthetic_posterior_data
    fig, ax = plt.subplots()

    # Test that hdi_prob still works (should override ci_prob)
    h_line, h_patch = plot_xY(x, Y, ax=ax, hdi_prob=0.95)

    # Check return types
    assert isinstance(h_line, plt.Line2D), "Should return Line2D for mean line"
    assert h_patch is not None, "Should return PolyCollection for HDI ribbon"

    plt.close(fig)


@pytest.mark.integration
def test_plot_xY_different_ci_prob(synthetic_posterior_data):
    """Test different ci_prob values."""
    x, Y = synthetic_posterior_data

    for ci_prob in [0.80, 0.89, 0.94, 0.95, 0.99]:
        fig, ax = plt.subplots()

        h_line, h_patch = plot_xY(
            x,
            Y,
            ax=ax,
            kind="ribbon",
            ci_kind="hdi",
            ci_prob=ci_prob,
        )

        # Check return types
        assert isinstance(h_line, plt.Line2D), (
            f"Should return Line2D for ci_prob={ci_prob}"
        )
        assert h_patch is not None, (
            f"Should return PolyCollection for ci_prob={ci_prob}"
        )

        plt.close(fig)


@pytest.mark.integration
def test_plot_xY_invalid_kind(synthetic_posterior_data):
    """Test that invalid kind parameter raises ValueError."""
    x, Y = synthetic_posterior_data
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="Unknown kind"):
        plot_xY(x, Y, ax=ax, kind="invalid_kind")

    plt.close(fig)


@pytest.mark.integration
def test_plot_xY_spaghetti_num_samples(synthetic_posterior_data):
    """Test spaghetti plot with different num_samples values."""
    x, Y = synthetic_posterior_data

    for num_samples in [10, 50, 100]:
        fig, ax = plt.subplots()

        handles, patch = plot_xY(
            x,
            Y,
            ax=ax,
            kind="spaghetti",
            num_samples=num_samples,
        )

        # Check return types
        assert isinstance(handles, list), (
            f"Should return list for num_samples={num_samples}"
        )
        assert patch is None, (
            f"Should not return PolyCollection for num_samples={num_samples}"
        )

        # Check that we have approximately the right number of lines
        # (samples + mean line, but may be less if num_samples > total samples)
        assert len(ax.lines) > 0, f"Should have lines for num_samples={num_samples}"

        plt.close(fig)


@pytest.mark.integration
def test_plot_xY_histogram_custom_colormap(synthetic_posterior_data):
    """Test histogram with custom colormap."""
    x, Y = synthetic_posterior_data
    fig, ax = plt.subplots()

    handles, patch = plot_xY(
        x,
        Y,
        ax=ax,
        kind="histogram",
        plot_hdi_kwargs={"cmap": "plasma", "alpha": 0.6},
    )

    # Check return types
    assert isinstance(handles, list), "Should return list of handles"
    assert patch is None, "Histogram should not return PolyCollection"

    plt.close(fig)
