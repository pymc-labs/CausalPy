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


# ==============================================================================
# Tests for time series data generation functions
# ==============================================================================


def test_generate_time_series_data():
    """Test the generate_time_series_data function."""
    from causalpy.data.simulate_data import generate_time_series_data

    # Test with default parameters
    df = generate_time_series_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100  # default N value
    assert "temperature" in df.columns
    assert "linear" in df.columns
    assert "causal effect" in df.columns
    assert "deaths_counterfactual" in df.columns
    assert "deaths_actual" in df.columns
    assert "intercept" in df.columns

    # Verify intercept is all ones
    assert np.all(df["intercept"] == 1.0)

    # Test with custom parameters
    df_custom = generate_time_series_data(
        N=50, treatment_time=30, beta_temp=-2, beta_linear=1.0, beta_intercept=5
    )
    assert len(df_custom) == 50


def test_generate_time_series_data_seasonal():
    """Test the generate_time_series_data_seasonal function."""
    from causalpy.data.simulate_data import generate_time_series_data_seasonal

    treatment_time = pd.to_datetime("2018-01-01")
    df = generate_time_series_data_seasonal(treatment_time)

    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert "year" in df.columns
    assert "month" in df.columns
    assert "t" in df.columns
    assert "y" in df.columns
    assert "causal effect" in df.columns
    assert "intercept" in df.columns

    # Verify intercept is all ones
    assert np.all(df["intercept"] == 1.0)

    # Verify year and month are extracted correctly
    assert df["year"].min() >= 2010
    assert df["year"].max() <= 2020
    assert df["month"].min() >= 1
    assert df["month"].max() <= 12


def test_generate_time_series_data_simple():
    """Test the generate_time_series_data_simple function."""
    from causalpy.data.simulate_data import generate_time_series_data_simple

    # Test with no slope
    treatment_time = pd.to_datetime("2015-01-01")
    df = generate_time_series_data_simple(treatment_time)

    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert "linear_trend" in df.columns
    assert "timeseries" in df.columns
    assert "causal effect" in df.columns
    assert "intercept" in df.columns

    # Verify intercept is all ones
    assert np.all(df["intercept"] == 1.0)

    # Verify causal effect is 0 before treatment and 2 after
    pre_treatment = df[df.index <= treatment_time]
    post_treatment = df[df.index > treatment_time]
    assert np.all(pre_treatment["causal effect"] == 0)
    assert np.all(post_treatment["causal effect"] == 2)

    # Test with slope
    df_with_slope = generate_time_series_data_simple(treatment_time, slope=0.5)
    assert isinstance(df_with_slope, pd.DataFrame)


# ==============================================================================
# Tests for DiD and ANCOVA data generation
# ==============================================================================


def test_generate_did():
    """Test the generate_did function."""
    from causalpy.data.simulate_data import generate_did

    df = generate_did()

    assert isinstance(df, pd.DataFrame)
    assert "group" in df.columns
    assert "t" in df.columns
    assert "unit" in df.columns
    assert "post_treatment" in df.columns
    assert "y" in df.columns

    # Verify group values
    assert set(df["group"].unique()) == {0, 1}

    # Verify time values
    assert set(df["t"].unique()) == {0.0, 1.0}

    # Verify post_treatment is boolean
    assert df["post_treatment"].dtype == bool

    # Verify there are both pre and post treatment observations
    assert df["post_treatment"].sum() > 0
    assert (~df["post_treatment"]).sum() > 0


def test_generate_ancova_data():
    """Test the generate_ancova_data function."""
    from causalpy.data.simulate_data import generate_ancova_data

    # Test with default parameters
    df = generate_ancova_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200  # default N value
    assert "group" in df.columns
    assert "pre" in df.columns
    assert "post" in df.columns

    # Verify group values
    assert set(df["group"].unique()) == {0, 1}

    # Test with custom parameters
    custom_means = np.array([5, 15])
    df_custom = generate_ancova_data(
        N=100, pre_treatment_means=custom_means, treatment_effect=5, sigma=2
    )
    assert len(df_custom) == 100


# ==============================================================================
# Tests for Event Study data generation
# ==============================================================================


def test_generate_event_study_data_basic():
    """Test the generate_event_study_data function with default parameters."""
    from causalpy.data.simulate_data import generate_event_study_data

    df = generate_event_study_data(seed=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20 * 20  # n_units * n_time = 400
    assert "unit" in df.columns
    assert "time" in df.columns
    assert "y" in df.columns
    assert "treat_time" in df.columns
    assert "treated" in df.columns

    # Verify unit and time ranges
    assert df["unit"].nunique() == 20
    assert df["time"].nunique() == 20

    # Verify treated fraction
    n_treated = df[df["treated"] == 1]["unit"].nunique()
    assert n_treated == 10  # 50% of 20 units


def test_generate_event_study_data_custom_params():
    """Test generate_event_study_data with custom parameters."""
    from causalpy.data.simulate_data import generate_event_study_data

    df = generate_event_study_data(
        n_units=10,
        n_time=15,
        treatment_time=8,
        treated_fraction=0.3,
        event_window=(-3, 3),
        seed=123,
    )

    assert len(df) == 10 * 15  # 150 rows
    assert df["unit"].nunique() == 10
    assert df["time"].nunique() == 15

    # Verify treated fraction (30% of 10 = 3 units)
    n_treated = df[df["treated"] == 1]["unit"].nunique()
    assert n_treated == 3


def test_generate_event_study_data_custom_treatment_effects():
    """Test generate_event_study_data with custom treatment effects."""
    from causalpy.data.simulate_data import generate_event_study_data

    custom_effects = {-1: 0.0, 0: 1.0, 1: 2.0, 2: 3.0}
    df = generate_event_study_data(
        n_units=10,
        n_time=10,
        treatment_time=5,
        event_window=(-1, 2),
        treatment_effects=custom_effects,
        seed=42,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100


def test_generate_event_study_data_with_predictors():
    """Test generate_event_study_data with predictor variables."""
    from causalpy.data.simulate_data import generate_event_study_data

    df = generate_event_study_data(
        n_units=10,
        n_time=10,
        treatment_time=5,
        predictor_effects={"temperature": 0.3, "humidity": -0.1},
        ar_phi=0.9,
        ar_scale=1.0,
        seed=42,
    )

    assert "temperature" in df.columns
    assert "humidity" in df.columns
    assert len(df) == 100


def test_generate_event_study_data_no_seed():
    """Test generate_event_study_data without seed (random)."""
    from causalpy.data.simulate_data import generate_event_study_data

    df = generate_event_study_data(n_units=5, n_time=5, treatment_time=3)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 25


# ==============================================================================
# Tests for utility functions
# ==============================================================================


def test_generate_ar1_series():
    """Test the generate_ar1_series function."""
    from causalpy.data.simulate_data import generate_ar1_series

    np.random.seed(42)

    # Test basic functionality
    series = generate_ar1_series(n=100, phi=0.9, scale=1.0, initial=0.0)
    assert isinstance(series, np.ndarray)
    assert len(series) == 100
    assert series[0] == 0.0  # Initial value

    # Test with different parameters
    series_custom = generate_ar1_series(n=50, phi=0.5, scale=0.5, initial=5.0)
    assert len(series_custom) == 50
    assert series_custom[0] == 5.0


def test_generate_seasonality():
    """Test the generate_seasonality function."""
    from causalpy.data.simulate_data import generate_seasonality

    np.random.seed(42)

    # Test with default parameters
    seasonality = generate_seasonality()
    assert isinstance(seasonality, np.ndarray)
    assert len(seasonality) == 12  # default n value

    # Test with custom parameters
    seasonality_custom = generate_seasonality(n=24, amplitude=2, length_scale=0.3)
    assert len(seasonality_custom) == 24


def test_periodic_kernel():
    """Test the periodic_kernel function."""
    from causalpy.data.simulate_data import periodic_kernel

    # Create test inputs
    x = np.linspace(0, 1, 10)
    x1, x2 = np.meshgrid(x, x)

    # Test with default parameters
    kernel = periodic_kernel(x1, x2)
    assert isinstance(kernel, np.ndarray)
    assert kernel.shape == (10, 10)

    # Verify kernel is symmetric
    assert np.allclose(kernel, kernel.T)

    # Verify diagonal is amplitude^2 (default 1)
    assert np.allclose(np.diag(kernel), 1.0)

    # Test with custom parameters
    kernel_custom = periodic_kernel(x1, x2, period=2, length_scale=0.5, amplitude=2)
    assert kernel_custom.shape == (10, 10)
    # Diagonal should be amplitude^2 = 4
    assert np.allclose(np.diag(kernel_custom), 4.0)


def test_create_series():
    """Test the create_series function."""
    from causalpy.data.simulate_data import create_series

    np.random.seed(42)

    # Test with default parameters
    series = create_series()
    assert isinstance(series, np.ndarray)
    assert len(series) == 52 * 4  # n * n_years = 52 * 4 = 208

    # Test with custom parameters
    series_custom = create_series(n=12, n_years=2, intercept=5)
    assert len(series_custom) == 12 * 2  # 24


def test_smoothed_gaussian_random_walk():
    """Test the _smoothed_gaussian_random_walk internal function."""
    from causalpy.data.simulate_data import _smoothed_gaussian_random_walk

    np.random.seed(42)

    x, y = _smoothed_gaussian_random_walk(
        gaussian_random_walk_mu=0.0,
        gaussian_random_walk_sigma=1.0,
        N=50,
        lowess_kwargs={"frac": 0.2, "it": 0},
    )

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(x) == 50
    assert len(y) == 50
    # Verify x is just arange
    np.testing.assert_array_equal(x, np.arange(50))
