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
Tests for Panel Regression with Fixed Effects
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp

# Minimal sample kwargs for fast tests
sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2, "progressbar": False}


@pytest.fixture
def small_panel_data():
    """Create a small panel dataset: 10 units, 20 time periods."""
    np.random.seed(42)
    units = [f"unit_{i}" for i in range(10)]
    periods = range(20)

    data = []
    for u_idx, u in enumerate(units):
        unit_effect = np.random.randn()
        for t in periods:
            time_effect = 0.1 * t
            treatment = 1 if (t >= 10 and u_idx < 5) else 0
            x1 = np.random.randn()
            # y = unit_effect + time_effect + treatment_effect + x1 + noise
            y = (
                unit_effect
                + time_effect
                + 2.0 * treatment
                + 0.5 * x1
                + 0.1 * np.random.randn()
            )
            data.append(
                {"unit": u, "time": t, "treatment": treatment, "x1": x1, "y": y}
            )

    return pd.DataFrame(data)


@pytest.fixture
def large_panel_data():
    """Create a larger panel dataset: 100 units, 10 time periods."""
    np.random.seed(42)
    units = [f"unit_{i}" for i in range(100)]
    periods = range(10)

    data = []
    for u_idx, u in enumerate(units):
        unit_effect = np.random.randn()
        for t in periods:
            time_effect = 0.2 * t
            treatment = 1 if t >= 5 else 0
            x1 = np.random.randn()
            y = (
                unit_effect
                + time_effect
                + 1.5 * treatment
                + 0.5 * x1
                + 0.1 * np.random.randn()
            )
            data.append(
                {"unit": u, "time": t, "treatment": treatment, "x1": x1, "y": y}
            )

    return pd.DataFrame(data)


@pytest.mark.integration
def test_panel_regression_pymc_dummies(mock_pymc_sample, small_panel_data):
    """Test PanelRegression with PyMC model using dummy variables method."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Check basic properties
    assert isinstance(result, cp.PanelRegression)
    assert result.n_units == 10
    assert result.n_periods == 20
    assert result.fe_method == "dummies"

    # Check that model was fitted
    assert hasattr(result.model, "idata")
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]

    # Check summary works
    result.summary()

    # Check plotting works
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


@pytest.mark.integration
def test_panel_regression_pymc_within(mock_pymc_sample, large_panel_data):
    """Test PanelRegression with PyMC model using within transformation."""
    result = cp.PanelRegression(
        data=large_panel_data,
        formula="y ~ treatment + x1",  # No C(unit) needed
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="within",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Check basic properties
    assert isinstance(result, cp.PanelRegression)
    assert result.n_units == 100
    assert result.n_periods == 10
    assert result.fe_method == "within"

    # Check that group means were stored
    assert "unit" in result._group_means
    assert "time" in result._group_means

    # Check plotting works
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_panel_regression_skl_dummies(small_panel_data):
    """Test PanelRegression with scikit-learn model using dummies."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    )

    # Check basic properties
    assert isinstance(result, cp.PanelRegression)
    assert result.n_units == 10
    assert result.n_periods == 20

    # Check that model was fitted
    assert hasattr(result.model, "coef_")
    assert len(result.model.coef_) == len(result.labels)

    # Check plotting works
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_panel_regression_skl_within(large_panel_data):
    """Test PanelRegression with scikit-learn model using within transformation."""
    result = cp.PanelRegression(
        data=large_panel_data,
        formula="y ~ treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="within",
        model=LinearRegression(),
    )

    # Check basic properties
    assert isinstance(result, cp.PanelRegression)
    assert result.n_units == 100
    assert result.fe_method == "within"

    # Check plotting
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_panel_regression_validation_errors(small_panel_data):
    """Test that validation errors are raised correctly."""
    # Missing unit variable
    with pytest.raises(cp.custom_exceptions.DataException, match="unit_fe_variable"):
        cp.PanelRegression(
            data=small_panel_data,
            formula="y ~ treatment + x1",
            unit_fe_variable="nonexistent",
            fe_method="within",
            model=LinearRegression(),
        )

    # Missing time variable
    with pytest.raises(cp.custom_exceptions.DataException, match="time_fe_variable"):
        cp.PanelRegression(
            data=small_panel_data,
            formula="y ~ treatment + x1",
            unit_fe_variable="unit",
            time_fe_variable="nonexistent",
            fe_method="within",
            model=LinearRegression(),
        )

    # Invalid fe_method
    with pytest.raises(ValueError, match="fe_method must be"):
        cp.PanelRegression(
            data=small_panel_data,
            formula="y ~ treatment + x1",
            unit_fe_variable="unit",
            fe_method="invalid",
            model=LinearRegression(),
        )

    # C(unit) in formula with within method
    with pytest.raises(ValueError, match="do not include C\\(unit\\)"):
        cp.PanelRegression(
            data=small_panel_data,
            formula="y ~ C(unit) + treatment + x1",
            unit_fe_variable="unit",
            fe_method="within",
            model=LinearRegression(),
        )


@pytest.mark.integration
def test_panel_regression_plot_coefficients(mock_pymc_sample, small_panel_data):
    """Test plot_coefficients method."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    fig, ax = result.plot_coefficients()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


@pytest.mark.integration
def test_panel_regression_plot_unit_effects(mock_pymc_sample, small_panel_data):
    """Test plot_unit_effects method."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    fig, ax = result.plot_unit_effects()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Should fail with within method
    result_within = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="within",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with pytest.raises(ValueError, match="only available with fe_method='dummies'"):
        result_within.plot_unit_effects()


@pytest.mark.integration
def test_panel_regression_plot_trajectories(mock_pymc_sample, small_panel_data):
    """Test plot_trajectories method."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Test with default random sampling
    fig, axes = result.plot_trajectories(n_sample=5)
    assert isinstance(fig, plt.Figure)
    assert len(axes) >= 5
    plt.close(fig)

    # Test with specific units
    fig, axes = result.plot_trajectories(units=["unit_0", "unit_1", "unit_2"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Should fail without time variable
    result_no_time = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + treatment + x1",
        unit_fe_variable="unit",
        fe_method="dummies",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with pytest.raises(ValueError, match="requires time_fe_variable"):
        result_no_time.plot_trajectories()


@pytest.mark.integration
def test_panel_regression_plot_residuals(mock_pymc_sample, small_panel_data):
    """Test plot_residuals method."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Test scatter plot
    fig, ax = result.plot_residuals(kind="scatter")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test histogram
    fig, ax = result.plot_residuals(kind="histogram")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test QQ plot
    fig, ax = result.plot_residuals(kind="qq")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


@pytest.mark.integration
def test_panel_regression_get_plot_data(mock_pymc_sample, small_panel_data):
    """Test get_plot_data methods."""
    # Bayesian
    result_bayes = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    plot_data = result_bayes.get_plot_data()
    assert isinstance(plot_data, pd.DataFrame)
    assert "y_actual" in plot_data.columns
    assert "y_fitted" in plot_data.columns
    assert "y_fitted_lower" in plot_data.columns
    assert "y_fitted_upper" in plot_data.columns
    assert "unit" in plot_data.columns
    assert "time" in plot_data.columns

    # OLS
    result_ols = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    )

    plot_data = result_ols.get_plot_data()
    assert isinstance(plot_data, pd.DataFrame)
    assert "y_actual" in plot_data.columns
    assert "y_fitted" in plot_data.columns
    assert "unit" in plot_data.columns


def test_panel_regression_one_way_fe(large_panel_data):
    """Test one-way fixed effects (unit FE only, no time FE)."""
    result = cp.PanelRegression(
        data=large_panel_data,
        formula="y ~ treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable=None,  # No time FE
        fe_method="within",
        model=LinearRegression(),
    )

    assert result.n_units == 100
    assert result.n_periods is None
    assert result.time_fe_variable is None

    # Check that only unit demeaning was applied
    assert "unit" in result._group_means
    assert "time" not in result._group_means


def test_panel_regression_two_way_fe(large_panel_data):
    """Test two-way fixed effects (unit + time FE)."""
    result = cp.PanelRegression(
        data=large_panel_data,
        formula="y ~ treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="within",
        model=LinearRegression(),
    )

    assert result.n_units == 100
    assert result.n_periods == 10

    # Check that both unit and time demeaning were applied
    assert "unit" in result._group_means
    assert "time" in result._group_means
