#   Copyright 2025 - 2025 The PyMC Labs Developers
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

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from causalpy.plot_utils import get_hdi_to_df


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
