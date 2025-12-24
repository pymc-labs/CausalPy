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
Tests for PiecewiseITS experiment class.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.data.simulate_data import generate_piecewise_its_data

# Sample kwargs for fast PyMC sampling in tests
sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


# ==============================================================================
# Unit tests for data generation
# ==============================================================================


def test_generate_piecewise_its_data_single_interruption():
    """Test data generation with single interruption."""
    df, params = generate_piecewise_its_data(
        N=100,
        interruption_times=[50],
        baseline_intercept=10.0,
        baseline_slope=0.1,
        level_changes=[5.0],
        slope_changes=[0.2],
        seed=42,
    )

    assert df.shape == (100, 5)
    assert list(df.columns) == ["t", "y", "y_true", "counterfactual", "effect"]
    assert params["baseline_intercept"] == 10.0
    assert params["baseline_slope"] == 0.1
    assert params["level_changes"] == [5.0]
    assert params["slope_changes"] == [0.2]
    assert params["interruption_times"] == [50]


def test_generate_piecewise_its_data_multiple_interruptions():
    """Test data generation with multiple interruptions."""
    df, params = generate_piecewise_its_data(
        N=150,
        interruption_times=[50, 100],
        level_changes=[3.0, -2.0],
        slope_changes=[0.1, -0.15],
        seed=42,
    )

    assert df.shape == (150, 5)
    assert len(params["interruption_times"]) == 2
    assert len(params["level_changes"]) == 2
    assert len(params["slope_changes"]) == 2


def test_generate_piecewise_its_data_level_only():
    """Test data generation with level change only (no slope change)."""
    df, params = generate_piecewise_its_data(
        N=100,
        interruption_times=[50],
        level_changes=[5.0],
        slope_changes=[0.0],
        seed=42,
    )

    # Effect should be constant after interruption
    effect_post = df.loc[df["t"] >= 50, "effect"]
    # All post-interruption effects should be 5.0 (level change only)
    assert np.allclose(effect_post, 5.0)


def test_generate_piecewise_its_data_slope_only():
    """Test data generation with slope change only (no level change)."""
    df, params = generate_piecewise_its_data(
        N=100,
        interruption_times=[50],
        level_changes=[0.0],
        slope_changes=[0.2],
        seed=42,
    )

    # Effect should be 0 at interruption time and grow linearly
    assert df.loc[df["t"] == 50, "effect"].values[0] == 0.0
    # At t=60, effect should be 0.2 * (60 - 50) = 2.0
    assert np.isclose(df.loc[df["t"] == 60, "effect"].values[0], 2.0)


def test_generate_piecewise_its_data_effect_consistency():
    """Test that y_true = counterfactual + effect."""
    df, _ = generate_piecewise_its_data(
        N=100,
        interruption_times=[50],
        level_changes=[5.0],
        slope_changes=[0.2],
        seed=42,
    )

    # y_true should equal counterfactual + effect
    expected = df["counterfactual"] + df["effect"]
    np.testing.assert_allclose(df["y_true"], expected)


def test_generate_piecewise_its_data_invalid_interruption_time():
    """Test that invalid interruption time raises error."""
    with pytest.raises(ValueError, match="outside valid range"):
        generate_piecewise_its_data(
            N=100,
            interruption_times=[150],  # Outside range [0, 99]
        )


def test_generate_piecewise_its_data_mismatched_lengths():
    """Test that mismatched level_changes length raises error."""
    with pytest.raises(ValueError, match="level_changes length"):
        generate_piecewise_its_data(
            N=100,
            interruption_times=[50],
            level_changes=[5.0, 3.0],  # Wrong length
        )


# ==============================================================================
# Unit tests for PiecewiseITS input validation
# ==============================================================================


def test_piecewise_its_missing_outcome_column():
    """Test that missing outcome column raises error."""
    df = pd.DataFrame({"t": range(100), "x": np.random.randn(100)})
    with pytest.raises(Exception, match="Outcome column"):
        cp.PiecewiseITS(
            df,
            outcome="y",  # Missing
            time="t",
            interruption_times=[50],
            model=LinearRegression(),
        )


def test_piecewise_its_missing_time_column():
    """Test that missing time column raises error."""
    df = pd.DataFrame({"y": np.random.randn(100), "x": np.random.randn(100)})
    with pytest.raises(Exception, match="Time column"):
        cp.PiecewiseITS(
            df,
            outcome="y",
            time="t",  # Missing
            interruption_times=[50],
            model=LinearRegression(),
        )


def test_piecewise_its_unsorted_interruptions():
    """Test that unsorted interruption times raises error."""
    df, _ = generate_piecewise_its_data(N=100, seed=42)
    with pytest.raises(ValueError, match="sorted"):
        cp.PiecewiseITS(
            df,
            outcome="y",
            time="t",
            interruption_times=[60, 40],  # Unsorted
            model=LinearRegression(),
        )


def test_piecewise_its_no_change_type():
    """Test that both change types disabled raises error."""
    df, _ = generate_piecewise_its_data(N=100, seed=42)
    with pytest.raises(ValueError, match="At least one"):
        cp.PiecewiseITS(
            df,
            outcome="y",
            time="t",
            interruption_times=[50],
            include_level_change=False,
            include_slope_change=False,
            model=LinearRegression(),
        )


def test_piecewise_its_empty_interruption_times():
    """Test that empty interruption times raises error."""
    df, _ = generate_piecewise_its_data(N=100, seed=42)
    with pytest.raises(ValueError, match="at least one"):
        cp.PiecewiseITS(
            df,
            outcome="y",
            time="t",
            interruption_times=[],
            model=LinearRegression(),
        )


def test_piecewise_its_interruption_outside_range():
    """Test that interruption time outside data range raises error."""
    df, _ = generate_piecewise_its_data(N=100, seed=42)
    with pytest.raises(ValueError, match="outside data range"):
        cp.PiecewiseITS(
            df,
            outcome="y",
            time="t",
            interruption_times=[150],  # Outside range
            model=LinearRegression(),
        )


# ==============================================================================
# Integration tests with OLS model
# ==============================================================================


def test_piecewise_its_ols_single_interruption():
    """Test PiecewiseITS with OLS model and single interruption."""
    df, params = generate_piecewise_its_data(
        N=100,
        interruption_times=[50],
        level_changes=[5.0],
        slope_changes=[0.2],
        noise_sigma=0.5,
        seed=42,
    )

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50],
        model=LinearRegression(),
    )

    assert isinstance(result, cp.PiecewiseITS)
    assert result.score > 0.9  # Should fit well with low noise
    assert len(result.labels) == 4  # Intercept, time, level_0, slope_0


def test_piecewise_its_ols_multiple_interruptions():
    """Test PiecewiseITS with OLS model and multiple interruptions."""
    df, params = generate_piecewise_its_data(
        N=150,
        interruption_times=[50, 100],
        level_changes=[3.0, -2.0],
        slope_changes=[0.1, -0.15],
        noise_sigma=0.5,
        seed=42,
    )

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50, 100],
        model=LinearRegression(),
    )

    assert isinstance(result, cp.PiecewiseITS)
    # 6 labels: Intercept, time, level_0, slope_0, level_1, slope_1
    assert len(result.labels) == 6


def test_piecewise_its_ols_level_only():
    """Test PiecewiseITS with OLS model and level change only."""
    df, _ = generate_piecewise_its_data(
        N=100,
        interruption_times=[50],
        level_changes=[5.0],
        slope_changes=[0.0],
        noise_sigma=0.5,
        seed=42,
    )

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50],
        include_level_change=True,
        include_slope_change=False,
        model=LinearRegression(),
    )

    # 3 labels: Intercept, time, level_0
    assert len(result.labels) == 3
    assert "level_0" in result.labels
    assert "slope_0" not in result.labels


def test_piecewise_its_ols_slope_only():
    """Test PiecewiseITS with OLS model and slope change only."""
    df, _ = generate_piecewise_its_data(
        N=100,
        interruption_times=[50],
        level_changes=[0.0],
        slope_changes=[0.2],
        noise_sigma=0.5,
        seed=42,
    )

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50],
        include_level_change=False,
        include_slope_change=True,
        model=LinearRegression(),
    )

    # 3 labels: Intercept, time, slope_0
    assert len(result.labels) == 3
    assert "slope_0" in result.labels
    assert "level_0" not in result.labels


def test_piecewise_its_ols_effect_consistency():
    """Test that effect = fitted - counterfactual for OLS models."""
    df, _ = generate_piecewise_its_data(
        N=100,
        interruption_times=[50],
        level_changes=[5.0],
        slope_changes=[0.2],
        noise_sigma=0.5,
        seed=42,
    )

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50],
        model=LinearRegression(),
    )

    # Effect should equal fitted - counterfactual
    expected_effect = np.squeeze(result.y_pred) - np.squeeze(result.y_counterfactual)
    np.testing.assert_allclose(result.effect, expected_effect)


def test_piecewise_its_ols_cumulative_effect():
    """Test cumulative effect computation for OLS models."""
    df, _ = generate_piecewise_its_data(
        N=100,
        interruption_times=[50],
        level_changes=[5.0],
        slope_changes=[0.2],
        noise_sigma=0.5,
        seed=42,
    )

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50],
        model=LinearRegression(),
    )

    # Cumulative effect should be cumsum of effect
    expected_cumulative = np.cumsum(result.effect)
    np.testing.assert_allclose(result.cumulative_effect, expected_cumulative)


def test_piecewise_its_ols_plot():
    """Test plotting for OLS models."""
    df, _ = generate_piecewise_its_data(N=100, seed=42)

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50],
        model=LinearRegression(),
    )

    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 3  # Three subplots
    plt.close(fig)


def test_piecewise_its_ols_get_plot_data():
    """Test get_plot_data for OLS models."""
    df, _ = generate_piecewise_its_data(N=100, seed=42)

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50],
        model=LinearRegression(),
    )

    plot_data = result.get_plot_data()
    assert isinstance(plot_data, pd.DataFrame)
    assert "t" in plot_data.columns
    assert "y" in plot_data.columns
    assert "fitted" in plot_data.columns
    assert "counterfactual" in plot_data.columns
    assert "effect" in plot_data.columns
    assert "cumulative_effect" in plot_data.columns


def test_piecewise_its_ols_summary():
    """Test summary method for OLS models."""
    df, _ = generate_piecewise_its_data(N=100, seed=42)

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50],
        model=LinearRegression(),
    )

    # Should not raise
    result.summary()


# ==============================================================================
# Integration tests with PyMC model
# ==============================================================================


@pytest.mark.integration
def test_piecewise_its_pymc_single_interruption(mock_pymc_sample):
    """Test PiecewiseITS with PyMC model and single interruption."""
    df, _ = generate_piecewise_its_data(
        N=100,
        interruption_times=[50],
        level_changes=[5.0],
        slope_changes=[0.2],
        noise_sigma=0.5,
        seed=42,
    )

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50],
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    assert isinstance(result, cp.PiecewiseITS)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]


@pytest.mark.integration
def test_piecewise_its_pymc_multiple_interruptions(mock_pymc_sample):
    """Test PiecewiseITS with PyMC model and multiple interruptions."""
    df, _ = generate_piecewise_its_data(
        N=150,
        interruption_times=[50, 100],
        level_changes=[3.0, -2.0],
        slope_changes=[0.1, -0.15],
        noise_sigma=0.5,
        seed=42,
    )

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50, 100],
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    assert isinstance(result, cp.PiecewiseITS)
    # 6 labels: Intercept, time, level_0, slope_0, level_1, slope_1
    assert len(result.labels) == 6


@pytest.mark.integration
def test_piecewise_its_pymc_level_only(mock_pymc_sample):
    """Test PiecewiseITS with PyMC model and level change only."""
    df, _ = generate_piecewise_its_data(
        N=100,
        interruption_times=[50],
        level_changes=[5.0],
        slope_changes=[0.0],
        noise_sigma=0.5,
        seed=42,
    )

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50],
        include_level_change=True,
        include_slope_change=False,
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # 3 labels: Intercept, time, level_0
    assert len(result.labels) == 3


@pytest.mark.integration
def test_piecewise_its_pymc_slope_only(mock_pymc_sample):
    """Test PiecewiseITS with PyMC model and slope change only."""
    df, _ = generate_piecewise_its_data(
        N=100,
        interruption_times=[50],
        level_changes=[0.0],
        slope_changes=[0.2],
        noise_sigma=0.5,
        seed=42,
    )

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50],
        include_level_change=False,
        include_slope_change=True,
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # 3 labels: Intercept, time, slope_0
    assert len(result.labels) == 3


@pytest.mark.integration
def test_piecewise_its_pymc_plot(mock_pymc_sample):
    """Test plotting for PyMC models."""
    df, _ = generate_piecewise_its_data(N=100, seed=42)

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50],
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 3  # Three subplots
    plt.close(fig)


@pytest.mark.integration
def test_piecewise_its_pymc_get_plot_data(mock_pymc_sample):
    """Test get_plot_data for PyMC models."""
    df, _ = generate_piecewise_its_data(N=100, seed=42)

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50],
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    plot_data = result.get_plot_data()
    assert isinstance(plot_data, pd.DataFrame)
    assert "t" in plot_data.columns
    assert "y" in plot_data.columns
    assert "fitted" in plot_data.columns
    assert "counterfactual" in plot_data.columns
    assert "effect" in plot_data.columns


@pytest.mark.integration
def test_piecewise_its_pymc_summary(mock_pymc_sample):
    """Test summary method for PyMC models."""
    df, _ = generate_piecewise_its_data(N=100, seed=42)

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50],
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Should not raise
    result.summary()


# ==============================================================================
# Test with control variables
# ==============================================================================


def test_piecewise_its_with_controls():
    """Test PiecewiseITS with control variables."""
    np.random.seed(42)
    N = 100
    t = np.arange(N)
    control_var = np.random.randn(N)
    y = 10 + 0.1 * t + 2 * control_var + 5 * (t >= 50) + np.random.randn(N)

    df = pd.DataFrame({"t": t, "y": y, "control": control_var})

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="t",
        interruption_times=[50],
        include_slope_change=False,
        controls=["control"],
        model=LinearRegression(),
    )

    # 4 labels: Intercept, time, level_0, control
    assert len(result.labels) == 4
    assert "control" in result.labels


# ==============================================================================
# Test with datetime index
# ==============================================================================


def test_piecewise_its_datetime_time():
    """Test PiecewiseITS with datetime time column."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    t_numeric = np.arange(100)
    y = 10 + 0.1 * t_numeric + 5 * (t_numeric >= 50) + np.random.randn(100)

    df = pd.DataFrame({"date": dates, "y": y})

    interruption_time = pd.Timestamp("2020-02-20")  # Roughly day 50

    result = cp.PiecewiseITS(
        df,
        outcome="y",
        time="date",
        interruption_times=[interruption_time],
        include_slope_change=False,
        model=LinearRegression(),
    )

    assert isinstance(result, cp.PiecewiseITS)
    assert result._time_is_datetime


def test_piecewise_its_datetime_type_mismatch():
    """Test that datetime/numeric type mismatch raises error."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    y = np.random.randn(100)

    df = pd.DataFrame({"date": dates, "y": y})

    with pytest.raises(Exception, match="datetime"):
        cp.PiecewiseITS(
            df,
            outcome="y",
            time="date",
            interruption_times=[50],  # Numeric instead of Timestamp
            model=LinearRegression(),
        )
