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

from causalpy.plot_utils import (
    _log_hdi_type_effect_summary_once,
    _log_hdi_type_info_once,
    add_hdi_annotation,
    get_hdi_to_df,
)


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


class TestAddHdiAnnotation:
    """Tests for the add_hdi_annotation function."""

    def test_add_hdi_annotation_expectation(self):
        """Test adding HDI annotation for expectation type."""
        fig, ax = plt.subplots()
        ax.set_title("Original Title")

        add_hdi_annotation(ax, "expectation")

        title = ax.get_title()
        assert "Original Title" in title
        assert "94% HDI of model expectation (μ)" in title
        assert "excl. observation noise" in title
        plt.close(fig)

    def test_add_hdi_annotation_prediction(self):
        """Test adding HDI annotation for prediction type."""
        fig, ax = plt.subplots()
        ax.set_title("Original Title")

        add_hdi_annotation(ax, "prediction")

        title = ax.get_title()
        assert "Original Title" in title
        assert "94% HDI of posterior predictive (ŷ)" in title
        assert "incl. observation noise" in title
        plt.close(fig)

    def test_add_hdi_annotation_custom_prob(self):
        """Test adding HDI annotation with custom probability."""
        fig, ax = plt.subplots()
        ax.set_title("My Plot")

        add_hdi_annotation(ax, "expectation", hdi_prob=0.89)

        title = ax.get_title()
        assert "89% HDI" in title
        plt.close(fig)

    def test_add_hdi_annotation_empty_title(self):
        """Test adding HDI annotation when there's no existing title."""
        fig, ax = plt.subplots()
        # No title set

        add_hdi_annotation(ax, "expectation")

        title = ax.get_title()
        assert "94% HDI of model expectation (μ)" in title
        plt.close(fig)


class TestHdiTypeLogging:
    """Tests for the HDI type logging functions."""

    def test_log_hdi_type_info_once_callable(self):
        """Test that _log_hdi_type_info_once is callable without error."""
        # Clear the cache to ensure fresh state
        _log_hdi_type_info_once.cache_clear()
        # Should not raise
        _log_hdi_type_info_once()

    def test_log_hdi_type_effect_summary_once_callable(self):
        """Test that _log_hdi_type_effect_summary_once is callable without error."""
        # Clear the cache to ensure fresh state
        _log_hdi_type_effect_summary_once.cache_clear()
        # Should not raise
        _log_hdi_type_effect_summary_once()
