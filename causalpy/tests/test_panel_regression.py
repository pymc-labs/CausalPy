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
    for _u_idx, u in enumerate(units):
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
    assert len(result.model.get_coeffs()) == len(result.labels)

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

    # C(time) in formula with within method
    with pytest.raises(ValueError, match="do not include C\\(time\\)"):
        cp.PanelRegression(
            data=small_panel_data,
            formula="y ~ C(time) + treatment + x1",
            unit_fe_variable="unit",
            time_fe_variable="time",
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


def test_within_transform_boolean_treatment():
    """Boolean treatment columns must be demeaned by the within transformation."""
    np.random.seed(42)
    n_units, n_periods = 20, 10
    data = pd.DataFrame(
        [
            {
                "unit": f"u{i}",
                "time": t,
                # boolean column, not int
                "treatment": t >= 5 and i < n_units // 2,
                "y": float(i) + 2.0 * (t >= 5 and i < n_units // 2) + np.random.randn(),
            }
            for i in range(n_units)
            for t in range(n_periods)
        ]
    )
    assert data["treatment"].dtype == bool, "fixture should produce bool treatment"

    result = cp.PanelRegression(
        data=data,
        formula="y ~ treatment",
        unit_fe_variable="unit",
        fe_method="within",
        model=LinearRegression(),
    )

    # The treatment coefficient should be close to 2.0.  Without the bool
    # fix the variable would not be demeaned and the estimate would be biased.
    treatment_idx = result.labels.index("treatment")
    treatment_coef = result.model.get_coeffs()[treatment_idx]
    assert abs(treatment_coef - 2.0) < 1.0, (
        f"Treatment coefficient {treatment_coef:.2f} far from true value 2.0; "
        "boolean column may not have been demeaned"
    )


def test_summary_ols_dummies_correct_coefficients(small_panel_data, capsys):
    """summary() must print the correct coefficient values for OLS dummies."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    )

    result.summary()
    captured = capsys.readouterr().out

    # The treatment coefficient from the OLS fit should be ~2.0 (true DGP value)
    treatment_idx = result.labels.index("treatment")
    true_coef = result.model.get_coeffs()[treatment_idx]

    # The printed output should contain a value close to the true coefficient,
    # not the value of some FE dummy.  Values are rounded to 2 significant
    # figures by default so we allow tolerance for that.
    assert "treatment" in captured
    # Parse the treatment line and check the value is correct
    for line in captured.splitlines():
        if "treatment" in line and "C(" not in line:
            # Extract the numeric value from the line
            parts = line.split()
            value = float(parts[-1])
            assert abs(value - true_coef) < 0.1, (
                f"Printed value {value} does not match true coefficient {true_coef}"
            )
            break


def test_effect_summary_raises(small_panel_data):
    """effect_summary() should raise NotImplementedError."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ treatment + x1",
        unit_fe_variable="unit",
        fe_method="within",
        model=LinearRegression(),
    )
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        result.effect_summary()


def test_plot_trajectories_select_extreme(small_panel_data):
    """plot_trajectories with select='extreme' picks high/low mean-outcome units."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    )
    fig, axes = result.plot_trajectories(n_sample=4, select="extreme")
    assert isinstance(fig, plt.Figure)
    # Should have at least 4 visible subplots
    visible = [ax for ax in axes if ax.get_visible()]
    assert len(visible) >= 4
    plt.close(fig)


def test_plot_trajectories_select_high_variance(small_panel_data):
    """plot_trajectories with select='high_variance' picks high-variance units."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    )
    fig, axes = result.plot_trajectories(n_sample=4, select="high_variance")
    assert isinstance(fig, plt.Figure)
    visible = [ax for ax in axes if ax.get_visible()]
    assert len(visible) >= 4
    plt.close(fig)


def test_plot_coefficients_with_var_names(small_panel_data):
    """plot_coefficients(var_names=...) should only plot the specified coefficients."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    )
    fig, ax = result.plot_coefficients(var_names=["treatment"])
    # Should have exactly one bar (horizontal bar chart)
    assert len(ax.patches) == 1
    plt.close(fig)


def test_group_means_from_original_data(large_panel_data):
    """_group_means should contain means from original data, not demeaned data."""
    result = cp.PanelRegression(
        data=large_panel_data,
        formula="y ~ treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="within",
        model=LinearRegression(),
    )

    # The stored unit group means should match means computed directly from
    # the original data.  Use check_like=True to ignore column ordering.
    original_unit_means = large_panel_data.groupby("unit")[
        ["y", "treatment", "x1"]
    ].mean()
    pd.testing.assert_frame_equal(
        result._group_means["unit"].sort_index(),
        original_unit_means.sort_index(),
        check_names=False,
        check_like=True,
    )

    # Time group means should also come from the original data (not
    # unit-demeaned data).
    original_time_means = large_panel_data.groupby("time")[
        ["y", "treatment", "x1"]
    ].mean()
    pd.testing.assert_frame_equal(
        result._group_means["time"].sort_index(),
        original_time_means.sort_index(),
        check_names=False,
        check_like=True,
    )


def test_plot_unit_effects_ols(small_panel_data):
    """plot_unit_effects() OLS branch should produce a histogram of point estimates."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    )

    fig, ax = result.plot_unit_effects()
    assert isinstance(fig, plt.Figure)
    # Histogram should have at least one patch
    assert len(ax.patches) > 0
    plt.close(fig)


def test_plot_residuals_ols(small_panel_data):
    """plot_residuals() should work with OLS models."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    )

    fig, ax = result.plot_residuals(kind="scatter")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_trajectories_all_units(small_panel_data):
    """plot_trajectories() with n_sample >= n_units should show all units."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    )

    # n_sample=100 exceeds the 10 units, so all 10 should be plotted
    fig, axes = result.plot_trajectories(n_sample=100)
    visible = [ax for ax in axes if ax.get_visible()]
    assert len(visible) == 10
    plt.close(fig)


def test_plot_trajectories_single_unit(small_panel_data):
    """plot_trajectories() with a single unit should not raise."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    )

    fig, axes = result.plot_trajectories(units=["unit_0"])
    assert isinstance(fig, plt.Figure)
    visible = [ax for ax in axes if ax.get_visible()]
    assert len(visible) == 1
    plt.close(fig)


def test_get_plot_data_bayesian_raises_on_ols(small_panel_data):
    """get_plot_data_bayesian() should raise ValueError for non-PyMC models."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + treatment + x1",
        unit_fe_variable="unit",
        fe_method="dummies",
        model=LinearRegression(),
    )

    with pytest.raises(ValueError, match="not a PyMC model"):
        result.get_plot_data_bayesian()


def test_get_plot_data_ols_raises_on_pymc(mock_pymc_sample, small_panel_data):
    """get_plot_data_ols() should raise ValueError for non-OLS models."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + treatment + x1",
        unit_fe_variable="unit",
        fe_method="dummies",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with pytest.raises(ValueError, match="not an OLS model"):
        result.get_plot_data_ols()


def test_plot_unit_effects_no_fe_labels(small_panel_data):
    """plot_unit_effects() raises when formula has no C(unit) terms."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ treatment + x1",
        unit_fe_variable="unit",
        fe_method="dummies",
        model=LinearRegression(),
    )

    with pytest.raises(ValueError, match="No unit fixed effects found"):
        result.plot_unit_effects()
