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
Tests for the simulated data functions
"""

import numpy as np
import pandas as pd


def test_generate_multicell_geolift_data():
    """
    Test the generate_multicell_geolift_data function.
    """
    from causalpy.data.simulate_data import generate_multicell_geolift_data

    df = generate_multicell_geolift_data()
    assert isinstance(df, pd.DataFrame)
    assert np.all(df >= 0), "Found negative values in dataset"


def test_generate_geolift_data():
    """
    Test the generate_geolift_data function.
    """
    from causalpy.data.simulate_data import generate_geolift_data

    df = generate_geolift_data()
    assert isinstance(df, pd.DataFrame)
    assert np.all(df >= 0), "Found negative values in dataset"


def test_generate_regression_discontinuity_data():
    """
    Test the generate_regression_discontinuity_data function.
    """
    from causalpy.data.simulate_data import generate_regression_discontinuity_data

    df = generate_regression_discontinuity_data()
    assert isinstance(df, pd.DataFrame)
    assert "x" in df.columns
    assert "y" in df.columns
    assert "treated" in df.columns
    assert len(df) == 100  # default N value
    assert df["treated"].dtype == bool or df["treated"].dtype == np.bool_

    # Test with custom parameters
    df_custom = generate_regression_discontinuity_data(
        N=50, true_causal_impact=1.0, true_treatment_threshold=0.5
    )
    assert len(df_custom) == 50


def test_generate_synthetic_control_data():
    """
    Test the generate_synthetic_control_data function.
    """
    from causalpy.data.simulate_data import generate_synthetic_control_data

    # Test with default parameters (lowess_kwargs=None)
    df, weightings = generate_synthetic_control_data()
    assert isinstance(df, pd.DataFrame)
    assert isinstance(weightings, np.ndarray)
    assert len(df) == 100  # default N value

    # Test with explicit lowess_kwargs
    df_custom, weightings_custom = generate_synthetic_control_data(
        N=50, lowess_kwargs={"frac": 0.3, "it": 5}
    )
    assert len(df_custom) == 50
