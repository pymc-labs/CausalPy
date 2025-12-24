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
Tests for PiecewiseITS experiment class and step/ramp transforms.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from patsy import dmatrix
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.data.simulate_data import generate_piecewise_its_data
from causalpy.transforms import RampTransform, StepTransform

# Sample kwargs for fast PyMC sampling in tests
sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


# ==============================================================================
# Unit tests for step/ramp transforms
# ==============================================================================


def test_step_transform_numeric():
    """Test step transform with numeric time."""
    transform = StepTransform()
    x = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    threshold = 50

    # Memorize
    transform.memorize_chunk(x, threshold)
    transform.memorize_finish()

    # Transform
    result = transform.transform(x, threshold)

    expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(result, expected)


def test_step_transform_datetime():
    """Test step transform with datetime time."""
    transform = StepTransform()
    x = pd.date_range("2020-01-01", periods=10, freq="D")
    threshold = "2020-01-06"

    # Memorize
    transform.memorize_chunk(x, threshold)
    transform.memorize_finish()

    # Transform
    result = transform.transform(x, threshold)

    # Days 0-4 are before threshold (2020-01-01 to 2020-01-05)
    # Days 5-9 are >= threshold (2020-01-06 to 2020-01-10)
    expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(result, expected)


def test_ramp_transform_numeric():
    """Test ramp transform with numeric time."""
    transform = RampTransform()
    x = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    threshold = 50

    # Memorize
    transform.memorize_chunk(x, threshold)
    transform.memorize_finish()

    # Transform
    result = transform.transform(x, threshold)

    expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0])
    np.testing.assert_array_equal(result, expected)


def test_ramp_transform_datetime():
    """Test ramp transform with datetime time - returns days from threshold."""
    transform = RampTransform()
    x = pd.date_range("2020-01-01", periods=10, freq="D")
    threshold = "2020-01-06"

    # Memorize
    transform.memorize_chunk(x, threshold)
    transform.memorize_finish()

    # Transform
    result = transform.transform(x, threshold)

    # Ramp in days: 0 for days before, then 0, 1, 2, 3, 4 for days at/after threshold
    expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.skip(
    reason="Known Python 3.13 + patsy + pytest interaction bug causes INTERNAL ERROR"
)
def test_transforms_with_patsy_dmatrix():
    """Test that step and ramp work correctly with patsy dmatrix.

    Note: This test works correctly but is skipped due to a pytest INTERNALERROR
    when patsy raises errors in Python 3.13. The functionality is covered by
    integration tests that use PiecewiseITS with formulas.
    """
    df = pd.DataFrame({"t": np.arange(100), "y": np.random.randn(100)})

    mat = dmatrix("1 + t + step(t, 50) + ramp(t, 50)", df)

    assert mat.shape[1] == 4  # Intercept, t, step, ramp
    assert "step(t, 50)" in mat.design_info.column_names
    assert "ramp(t, 50)" in mat.design_info.column_names

    # Verify step values
    step_col = mat[:, mat.design_info.column_names.index("step(t, 50)")]
    assert np.all(step_col[:50] == 0)
    assert np.all(step_col[50:] == 1)

    # Verify ramp values
    ramp_col = mat[:, mat.design_info.column_names.index("ramp(t, 50)")]
    assert np.all(ramp_col[:50] == 0)
    np.testing.assert_array_equal(ramp_col[50:], np.arange(50))


@pytest.mark.skip(
    reason="Known Python 3.13 + patsy + pytest interaction bug causes INTERNAL ERROR"
)
def test_transforms_with_patsy_datetime():
    """Test that step and ramp work with datetime in patsy.

    Note: This test works correctly but is skipped due to a pytest INTERNALERROR
    when patsy raises errors in Python 3.13. The functionality is covered by
    integration tests that use PiecewiseITS with datetime formulas.
    """
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "y": np.random.randn(100),
        }
    )

    mat = dmatrix("1 + step(date, '2020-02-20') + ramp(date, '2020-02-20')", df)

    assert mat.shape[1] == 3  # Intercept, step, ramp
    # Day 50 is 2020-02-20 (0-indexed)
    step_col = mat[:, mat.design_info.column_names.index("step(date, '2020-02-20')")]
    assert np.all(step_col[:50] == 0)
    assert np.all(step_col[50:] == 1)


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


def test_piecewise_its_no_step_or_ramp():
    """Test that formula without step() or ramp() raises error."""
    df, _ = generate_piecewise_its_data(N=100, seed=42)
    with pytest.raises(Exception, match="step.*ramp"):
        cp.PiecewiseITS(
            df,
            formula="y ~ 1 + t",  # No step or ramp
            model=LinearRegression(),
        )


def test_piecewise_its_missing_column():
    """Test that missing column in formula raises error."""
    from patsy import PatsyError

    df = pd.DataFrame({"t": range(100), "y": np.random.randn(100)})
    with pytest.raises(PatsyError):
        cp.PiecewiseITS(
            df,
            formula="y ~ 1 + t + step(t, 50) + nonexistent",
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
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
        model=LinearRegression(),
    )

    assert isinstance(result, cp.PiecewiseITS)
    assert result.score > 0.9  # Should fit well with low noise
    assert len(result.labels) == 4  # Intercept, time, step, ramp


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
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50) + step(t, 100) + ramp(t, 100)",
        model=LinearRegression(),
    )

    assert isinstance(result, cp.PiecewiseITS)
    # 6 labels: Intercept, t, step(t,50), ramp(t,50), step(t,100), ramp(t,100)
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
        formula="y ~ 1 + t + step(t, 50)",  # Only step, no ramp
        model=LinearRegression(),
    )

    # 3 labels: Intercept, time, step
    assert len(result.labels) == 3
    assert any("step" in label for label in result.labels)
    assert not any("ramp" in label for label in result.labels)


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
        formula="y ~ 1 + t + ramp(t, 50)",  # Only ramp, no step
        model=LinearRegression(),
    )

    # 3 labels: Intercept, time, ramp
    assert len(result.labels) == 3
    assert any("ramp" in label for label in result.labels)
    assert not any("step" in label for label in result.labels)


def test_piecewise_its_ols_mixed_effects_per_intervention():
    """Test different effects per intervention (key new capability)."""
    df, _ = generate_piecewise_its_data(
        N=150,
        interruption_times=[50, 100],
        level_changes=[5.0, 3.0],
        slope_changes=[0.0, 0.1],  # No slope change at 50, slope change at 100
        noise_sigma=0.5,
        seed=42,
    )

    # Level change only at t=50, level + slope change at t=100
    result = cp.PiecewiseITS(
        df,
        formula="y ~ 1 + t + step(t, 50) + step(t, 100) + ramp(t, 100)",
        model=LinearRegression(),
    )

    # 5 labels: Intercept, t, step(50), step(100), ramp(100)
    assert len(result.labels) == 5
    # Check we have two step terms and one ramp term
    step_count = sum(1 for label in result.labels if "step" in label)
    ramp_count = sum(1 for label in result.labels if "ramp" in label)
    assert step_count == 2
    assert ramp_count == 1


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
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
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
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
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
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
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
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
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
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
        model=LinearRegression(),
    )

    # Should not raise
    result.summary()


def test_piecewise_its_extract_interruption_times():
    """Test that interruption times are correctly extracted from formula."""
    df, _ = generate_piecewise_its_data(N=150, seed=42)

    result = cp.PiecewiseITS(
        df,
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50) + step(t, 100)",
        model=LinearRegression(),
    )

    # Should extract unique thresholds: 50 and 100
    assert 50 in result.interruption_times
    assert 100 in result.interruption_times
    assert len(result.interruption_times) == 2


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
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
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
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50) + step(t, 100) + ramp(t, 100)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    assert isinstance(result, cp.PiecewiseITS)
    # 6 labels: Intercept, t, step(50), ramp(50), step(100), ramp(100)
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
        formula="y ~ 1 + t + step(t, 50)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # 3 labels: Intercept, time, step
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
        formula="y ~ 1 + t + ramp(t, 50)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # 3 labels: Intercept, time, ramp
    assert len(result.labels) == 3


@pytest.mark.integration
def test_piecewise_its_pymc_plot(mock_pymc_sample):
    """Test plotting for PyMC models."""
    df, _ = generate_piecewise_its_data(N=100, seed=42)

    result = cp.PiecewiseITS(
        df,
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
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
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
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
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Should not raise
    result.summary()


# ==============================================================================
# Test with control variables (via formula)
# ==============================================================================


def test_piecewise_its_with_controls():
    """Test PiecewiseITS with control variables in formula."""
    np.random.seed(42)
    N = 100
    t = np.arange(N)
    control_var = np.random.randn(N)
    y = 10 + 0.1 * t + 2 * control_var + 5 * (t >= 50) + np.random.randn(N)

    df = pd.DataFrame({"t": t, "y": y, "control": control_var})

    result = cp.PiecewiseITS(
        df,
        formula="y ~ 1 + t + step(t, 50) + control",
        model=LinearRegression(),
    )

    # 4 labels: Intercept, time, step, control
    assert len(result.labels) == 4
    assert "control" in result.labels


def test_piecewise_its_with_categorical_control():
    """Test PiecewiseITS with categorical control (seasonality)."""
    np.random.seed(42)
    N = 120
    t = np.arange(N)
    month = np.tile(np.arange(1, 13), 10)  # 10 years of monthly data
    y = 10 + 0.1 * t + np.sin(month * np.pi / 6) + 5 * (t >= 60) + np.random.randn(N)

    df = pd.DataFrame({"t": t, "y": y, "month": month})

    result = cp.PiecewiseITS(
        df,
        formula="y ~ 1 + t + step(t, 60) + C(month)",
        model=LinearRegression(),
    )

    # Should have many labels due to categorical expansion
    assert len(result.labels) > 4  # Intercept + t + step + 11 month dummies


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

    result = cp.PiecewiseITS(
        df,
        formula="y ~ 1 + step(date, '2020-02-20') + ramp(date, '2020-02-20')",
        model=LinearRegression(),
    )

    assert isinstance(result, cp.PiecewiseITS)
    # Check interruption times extracted correctly
    assert "2020-02-20" in result.interruption_times


def test_piecewise_its_datetime_multiple_interruptions():
    """Test PiecewiseITS with datetime and multiple interruptions."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=150, freq="D")
    t_numeric = np.arange(150)
    y = (
        10
        + 0.1 * t_numeric
        + 3 * (t_numeric >= 50)
        + 5 * (t_numeric >= 100)
        + np.random.randn(150)
    )

    df = pd.DataFrame({"date": dates, "y": y})

    result = cp.PiecewiseITS(
        df,
        formula="y ~ 1 + step(date, '2020-02-20') + step(date, '2020-04-10')",
        model=LinearRegression(),
    )

    assert len(result.interruption_times) == 2
