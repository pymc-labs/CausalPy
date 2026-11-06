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
import xarray as xr
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy._arviz_compat import hdi_bound_arrays
from causalpy.pymc_models import GeneralizedLinearRegression

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
    ).fit()

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


def test_panel_regression_rejects_poisson_demeaned(small_panel_data):
    """Non-Gaussian GLRs cannot use the demeaned fixed-effects transformation."""
    with pytest.raises(ValueError, match="fe_method='demeaned' is not supported"):
        cp.PanelRegression(
            data=small_panel_data,
            formula="y ~ treatment + x1",
            unit_fe_variable="unit",
            fe_method="demeaned",
            model=GeneralizedLinearRegression(
                family="poisson",
                sample_kwargs=sample_kwargs,
            ),
        )


@pytest.mark.integration
def test_panel_regression_poisson_dummies(mock_pymc_sample, small_panel_data):
    """Poisson GLR panel models remain valid with dummy fixed effects."""
    rng = np.random.default_rng(7)
    count_panel = small_panel_data.copy()
    # Replace the continuous Gaussian outcome with genuine non-negative counts;
    # the GLR fit boundary rejects outcomes outside the Poisson support.
    count_panel["y"] = rng.poisson(
        np.exp(0.5 + 0.3 * count_panel["treatment"] + 0.1 * count_panel["x1"])
    )
    result = cp.PanelRegression(
        data=count_panel,
        formula="y ~ treatment + x1 + C(unit)",
        unit_fe_variable="unit",
        fe_method="dummies",
        model=GeneralizedLinearRegression(
            family="poisson",
            sample_kwargs=sample_kwargs,
        ),
    )
    assert isinstance(result, cp.PanelRegression)
    assert result.fe_method == "dummies"


@pytest.mark.integration
def test_panel_regression_pymc_demeaned(mock_pymc_sample, large_panel_data):
    """Test PanelRegression with PyMC model using demeaned transformation."""
    result = cp.PanelRegression(
        data=large_panel_data,
        formula="y ~ treatment + x1",  # No C(unit) needed
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="demeaned",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    ).fit()

    # Check basic properties
    assert isinstance(result, cp.PanelRegression)
    assert result.n_units == 100
    assert result.n_periods == 10
    assert result.fe_method == "demeaned"

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
    ).fit()

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


def test_panel_regression_skl_demeaned(large_panel_data):
    """Test PanelRegression with scikit-learn model using demeaned transformation."""
    result = cp.PanelRegression(
        data=large_panel_data,
        formula="y ~ treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="demeaned",
        model=LinearRegression(),
    ).fit()

    # Check basic properties
    assert isinstance(result, cp.PanelRegression)
    assert result.n_units == 100
    assert result.fe_method == "demeaned"

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
            fe_method="demeaned",
            model=LinearRegression(),
        )

    # Missing time variable
    with pytest.raises(cp.custom_exceptions.DataException, match="time_fe_variable"):
        cp.PanelRegression(
            data=small_panel_data,
            formula="y ~ treatment + x1",
            unit_fe_variable="unit",
            time_fe_variable="nonexistent",
            fe_method="demeaned",
            model=LinearRegression(),
        )

    # Invalid fe_method
    with pytest.raises(
        ValueError,
        match=(
            "fe_method must be 'dummies' \\(unpooled fixed effects\\) or 'demeaned'"
        ),
    ):
        cp.PanelRegression(
            data=small_panel_data,
            formula="y ~ treatment + x1",
            unit_fe_variable="unit",
            fe_method="invalid",
            model=LinearRegression(),
        )

    # Hierarchical naming is reserved for future partial-pooling work
    with pytest.raises(
        ValueError,
        match=(
            "fe_method must be 'dummies' \\(unpooled fixed effects\\) or 'demeaned'"
        ),
    ):
        cp.PanelRegression(
            data=small_panel_data,
            formula="y ~ treatment + x1",
            unit_fe_variable="unit",
            fe_method="hierarchical",  # type: ignore[arg-type]
            model=LinearRegression(),
        )

    # C(unit) in formula with demeaned method
    with pytest.raises(ValueError, match="do not include C\\(unit\\)"):
        cp.PanelRegression(
            data=small_panel_data,
            formula="y ~ C(unit) + treatment + x1",
            unit_fe_variable="unit",
            fe_method="demeaned",
            model=LinearRegression(),
        )

    # C(time) in formula with demeaned method
    with pytest.raises(ValueError, match="do not include C\\(time\\)"):
        cp.PanelRegression(
            data=small_panel_data,
            formula="y ~ C(time) + treatment + x1",
            unit_fe_variable="unit",
            time_fe_variable="time",
            fe_method="demeaned",
            model=LinearRegression(),
        )


@pytest.mark.parametrize(
    ("formula", "variable"),
    [
        ("y ~ C( unit ) + treatment + x1", "unit"),
        ("y ~ C(unit, Treatment) + treatment + x1", "unit"),
        ("y ~ C( time ) + treatment + x1", "time"),
        ("y ~ C(time, Sum) + treatment + x1", "time"),
    ],
)
def test_demeaned_rejects_patsy_fe_variants(small_panel_data, formula, variable):
    """Whitespace and coding options must not bypass the demeaned FE guard."""
    with pytest.raises(ValueError, match=rf"do not include C\({variable}\)"):
        cp.PanelRegression(
            data=small_panel_data,
            formula=formula,
            unit_fe_variable="unit",
            time_fe_variable="time",
            fe_method="demeaned",
            model=LinearRegression(),
        )


def test_demeaned_allows_unit_interaction_without_plain_fe(small_panel_data):
    """A unit-specific slope is not a duplicate fixed-effect intercept."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit):treatment + x1",
        unit_fe_variable="unit",
        fe_method="demeaned",
        model=LinearRegression(),
    )

    assert any("C(unit)" in label and ":treatment" in label for label in result.labels)


def test_dummy_fe_labels_follow_patsy_term_metadata(small_panel_data):
    """FE labels with whitespace or coding options are hidden and plottable."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C( unit ) + C(time, Sum) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    ).fit()

    fe_labels = set(result._get_fe_labels("unit"))
    fe_labels.update(result._get_fe_labels("time"))
    assert fe_labels
    assert fe_labels.isdisjoint(result._get_non_fe_labels())

    fig, _ = result.plot_unit_effects()
    plt.close(fig)


def test_interaction_labels_are_not_plain_fixed_effects(small_panel_data, capsys):
    """Unit interactions remain visible and are excluded from FE plots."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(unit):treatment + x1",
        unit_fe_variable="unit",
        fe_method="dummies",
        model=LinearRegression(),
    ).fit()

    unit_fe_labels = set(result._get_fe_labels("unit"))
    interaction_labels = {label for label in result.labels if ":" in label}
    assert interaction_labels
    assert unit_fe_labels.isdisjoint(interaction_labels)
    assert interaction_labels.issubset(result._get_non_fe_labels())

    result.summary()
    summary = capsys.readouterr().out
    assert all(label in summary for label in interaction_labels)


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
    ).fit()

    fig, ax = result.plot_coefficients()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    fig, ax = result.plot_coefficients(hdi_prob=0.89)
    assert isinstance(fig, plt.Figure)
    assert "89% HDI" in ax.get_title()
    plt.close(fig)


@pytest.mark.integration
@pytest.mark.filterwarnings(
    "error:Conversion of an array with ndim > 0 to a scalar:DeprecationWarning"
)
def test_panel_regression_plot_unit_effects(mock_pymc_sample, small_panel_data):
    """Test plot_unit_effects method."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    ).fit()

    fig, ax = result.plot_unit_effects()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Should fail with demeaned method
    result_demeaned = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="demeaned",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    ).fit()

    with pytest.raises(ValueError, match="only available with fe_method='dummies'"):
        result_demeaned.plot_unit_effects()


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
    ).fit()

    # Test with default random sampling
    fig, axes = result.plot_trajectories(n_sample=5)
    assert isinstance(fig, plt.Figure)
    assert len(axes) >= 5
    plt.close(fig)

    # Test predictive intervals when posterior predictive samples are available
    posterior_predictive = getattr(result.model.idata, "posterior_predictive", None)
    if posterior_predictive is not None and "y_hat" in posterior_predictive:
        fig, axes = result.plot_trajectories(n_sample=5, interval_type="predictive")
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
    ).fit()

    with pytest.raises(ValueError, match="requires time_fe_variable"):
        result_no_time.plot_trajectories()

    with pytest.raises(ValueError, match="interval_type must be"):
        result.plot_trajectories(interval_type="invalid")  # type: ignore[arg-type]


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
    ).fit()

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
    ).fit()

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
    ).fit()

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
        fe_method="demeaned",
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
        fe_method="demeaned",
        model=LinearRegression(),
    )

    assert result.n_units == 100
    assert result.n_periods == 10

    # Check that both unit and time demeaning were applied
    assert "unit" in result._group_means
    assert "time" in result._group_means


def test_demean_transform_boolean_treatment():
    """Boolean treatment columns must be demeaned by the demeaned transformation."""
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
        fe_method="demeaned",
        model=LinearRegression(),
    ).fit()

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
    ).fit()

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
        fe_method="demeaned",
        model=LinearRegression(),
    )
    with pytest.raises(
        NotImplementedError, match=r"not yet implemented for PanelRegression"
    ):
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
    ).fit()
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
    ).fit()
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
    ).fit()
    fig, ax = result.plot_coefficients(var_names=["treatment"])
    # Should have exactly one bar (horizontal bar chart)
    assert len(ax.patches) == 1
    plt.close(fig)


def test_plot_coefficients_rejects_invalid_hdi_prob(small_panel_data):
    """plot_coefficients() should validate hdi_prob bounds."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    ).fit()
    with pytest.raises(ValueError, match="hdi_prob must be between 0 and 1"):
        result.plot_coefficients(hdi_prob=1.2)


def test_panel_coefficient_plots_reject_multiple_outcomes(
    small_panel_data, monkeypatch
):
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    ).fit()
    coefficients = result._model_backend.coefficients().isel(treated_units=0, drop=True)
    coefficients = xr.concat(
        [coefficients, coefficients],
        dim=xr.IndexVariable("treated_units", ["north", "south"]),
    ).transpose("chain", "draw", "coeffs", "treated_units")
    monkeypatch.setattr(
        result._model_backend, "coefficients", lambda *, group="posterior": coefficients
    )

    with pytest.raises(ValueError, match="exactly one outcome unit"):
        result.plot_coefficients()
    with pytest.raises(ValueError, match="exactly one outcome unit"):
        result.plot_unit_effects()


def test_group_means_from_original_data(large_panel_data):
    """_group_means should contain means from original data, not demeaned data."""
    result = cp.PanelRegression(
        data=large_panel_data,
        formula="y ~ treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="demeaned",
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
    ).fit()

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
    ).fit()

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
    ).fit()

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
    ).fit()

    fig, axes = result.plot_trajectories(units=["unit_0"])
    assert isinstance(fig, plt.Figure)
    visible = [ax for ax in axes if ax.get_visible()]
    assert len(visible) == 1
    plt.close(fig)


def test_plot_unit_effects_no_fe_labels(small_panel_data):
    """plot_unit_effects() raises when formula has no C(unit) terms."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ treatment + x1",
        unit_fe_variable="unit",
        fe_method="dummies",
        model=LinearRegression(),
    ).fit()

    with pytest.raises(ValueError, match="No unit fixed effects found"):
        result.plot_unit_effects()


@pytest.mark.integration
@pytest.mark.parametrize("hdi_prob", [0.5, 0.94])
def test_panel_bayesian_coefficient_forest_preserves_axes_and_hdi_bounds(
    mock_pymc_sample,
    small_panel_data,
    hdi_prob,
):
    """Bayesian coefficient plotting stays on one Axes with explicit HDI bounds."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    ).fit()
    coeff_names = ["treatment", "x1"]

    fig, ax = result.plot_coefficients(var_names=coeff_names, hdi_prob=hdi_prob)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert [tick.get_text() for tick in ax.get_yticklabels()] == coeff_names
    assert f"{hdi_prob:.0%} HDI" in ax.get_title()

    coefficients = result.model.idata.posterior["beta"].sel(coeffs=coeff_names)
    coefficients = coefficients.squeeze("treated_units", drop=True)
    lower, upper = hdi_bound_arrays(
        coefficients,
        prob=hdi_prob,
        dim=["chain", "draw"],
    )
    error_segments = ax.collections[0].get_segments()
    for segment, expected_lower, expected_upper in zip(
        error_segments,
        lower,
        upper,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.sort(segment[:, 0]),
            [expected_lower, expected_upper],
        )
    marker_line = next(line for line in ax.lines if line.get_marker() == "o")
    np.testing.assert_allclose(
        marker_line.get_xdata(),
        coefficients.mean(dim=["chain", "draw"]).values,
    )
    np.testing.assert_allclose(marker_line.get_ydata(), np.arange(len(coeff_names)))

    plt.close(fig)


def test_panel_plot_coefficients_rejects_empty_selection(small_panel_data):
    """An empty requested coefficient list raises instead of creating a blank plot."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    ).fit()

    with pytest.raises(ValueError, match="at least one coefficient"):
        result.plot_coefficients(var_names=[])


@pytest.mark.integration
@pytest.mark.parametrize("hdi_prob", [0.5, 0.94])
@pytest.mark.parametrize("interval_type", ["mean", "predictive"])
def test_panel_trajectory_hdi_band_preserves_fitted_line_and_legend(
    mock_pymc_sample,
    small_panel_data,
    hdi_prob,
    interval_type,
):
    """Trajectory bands use the selected interval without duplicating Fitted."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    ).fit()

    fig, axes = result.plot_trajectories(
        units=["unit_0"],
        hdi_prob=hdi_prob,
        interval_type=interval_type,
    )

    assert isinstance(axes, np.ndarray)
    ax = axes[0]
    fitted_lines = [line for line in ax.lines if line.get_label() == "Fitted"]
    assert len(fitted_lines) == 1
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Actual",
        "Fitted",
    ]

    unit_indices = np.where(result.data["unit"] == "unit_0")[0]
    sort_order = np.argsort(result.data.loc[result.data["unit"] == "unit_0", "time"])
    sorted_indices = unit_indices[sort_order]
    unit_mu = result.model.idata.posterior["mu"].isel(obs_ind=sorted_indices.tolist())
    unit_mu = unit_mu.squeeze("treated_units", drop=True)
    np.testing.assert_allclose(
        fitted_lines[0].get_ydata(),
        unit_mu.mean(dim=["chain", "draw"]).values,
    )
    interval_source = (
        unit_mu
        if interval_type == "mean"
        else result.model.idata.posterior_predictive["y_hat"]
        .isel(obs_ind=sorted_indices.tolist())
        .squeeze("treated_units", drop=True)
    )
    lower, upper = hdi_bound_arrays(
        interval_source,
        prob=hdi_prob,
        dim=["chain", "draw"],
    )
    band = next(
        collection
        for collection in ax.collections
        if isinstance(collection, PolyCollection)
    )
    vertices = band.get_paths()[0].vertices[:, 1]
    for bound in np.concatenate([lower, upper]):
        assert np.isclose(vertices, bound).any()
    np.testing.assert_allclose(
        band.get_facecolor()[0],
        to_rgba(fitted_lines[0].get_color(), alpha=0.2),
    )
    plt.close(fig)


@pytest.mark.parametrize("select", ["random", "extreme", "high_variance"])
def test_panel_plot_trajectories_rejects_empty_selection(small_panel_data, select):
    """Every selection strategy rejects a non-positive sample size before plotting."""
    result = cp.PanelRegression(
        data=small_panel_data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    ).fit()

    with pytest.raises(ValueError, match="n_sample must be positive"):
        result.plot_trajectories(n_sample=0, select=select)
    with pytest.raises(ValueError, match="at least one unit"):
        result.plot_trajectories(units=[])


def _named_ols_coefs(result):
    coefs = np.asarray(result.model.coef_, dtype=float).reshape(-1)
    return {
        label: float(value) for label, value in zip(result.labels, coefs, strict=True)
    }


def _fit_panel(data, formula, method, *, time_fe=True):
    return cp.PanelRegression(
        data,
        formula=formula,
        unit_fe_variable="unit",
        time_fe_variable="time" if time_fe else None,
        fe_method=method,
        model=LinearRegression(),
    ).fit()


def _cited_interaction_panel():
    """The issue's 4 by 4 balanced panel, with a planted treatment interaction."""
    pattern = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 0,
        (0, 3): 2,
        (1, 0): 1,
        (1, 1): 0,
        (1, 2): 2,
        (1, 3): 0,
        (2, 0): 0,
        (2, 1): 2,
        (2, 2): 1,
        (2, 3): 3,
        (3, 0): 2,
        (3, 1): 0,
        (3, 2): 3,
        (3, 3): 1,
    }
    rows = []
    for unit in range(4):
        for time in range(4):
            treatment = int(unit >= 2 and time >= 2)
            x = float(pattern[(unit, time)])
            y = (
                10.0 * unit
                + 1.0 * time
                + 2.0 * treatment
                + 3.0 * x
                + 4.0 * treatment * x
            )
            rows.append(
                {
                    "unit": f"u{unit}",
                    "time": time,
                    "treatment": treatment,
                    "x": x,
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


def _constructed_term_panel(n_units, n_periods, seed):
    """Balanced panel whose formula terms are not raw columns.

    ``post:z`` and ``I(w ** 2)`` disagree with demean-then-multiply unless the
    within transform is applied after the formula is built.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n_units):
        for time in range(n_periods):
            post = int(time >= (unit % 3) + 1)
            z = float(rng.normal())
            w = float(rng.integers(0, 4))
            y = (
                7.0 * unit
                + 0.4 * time
                + 5.0 * post
                - 1.5 * z
                + 0.7 * post * z
                - 2.0 * w**2
            )
            rows.append(
                {
                    "unit": f"u{unit}",
                    "time": time,
                    "post": post,
                    "z": z,
                    "w": w,
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


def test_demeaned_balanced_panel_matches_dummy_within_transform():
    """Demeaned FE matches dummy FE on balanced panels, including constructed terms.

    An additive formula that already agreed must keep agreeing. A constructed
    term other than the cited ``treatment:x`` on the cited 4 by 4 panel must
    also match, on more than one panel shape.
    """
    cited = _cited_interaction_panel()
    interaction = "y ~ 1 + treatment + x + treatment:x"
    dummy_interaction = "y ~ 1 + C(unit) + C(time) + treatment + x + treatment:x"
    demeaned = _named_ols_coefs(_fit_panel(cited, interaction, "demeaned"))
    dummies = _named_ols_coefs(_fit_panel(cited, dummy_interaction, "dummies"))
    expected = {"treatment": 2.0, "x": 3.0, "treatment:x": 4.0}
    for name, value in expected.items():
        assert demeaned[name] == pytest.approx(value)
        assert dummies[name] == pytest.approx(value)

    additive = "y ~ 1 + treatment + x"
    dummy_additive = "y ~ 1 + C(unit) + C(time) + treatment + x"
    demeaned_additive = _named_ols_coefs(_fit_panel(cited, additive, "demeaned"))
    dummy_additive_coefs = _named_ols_coefs(
        _fit_panel(cited, dummy_additive, "dummies")
    )
    assert demeaned_additive["treatment"] == pytest.approx(
        dummy_additive_coefs["treatment"]
    )
    assert demeaned_additive["x"] == pytest.approx(dummy_additive_coefs["x"])
    assert demeaned_additive["treatment"] == pytest.approx(9.384615384615)
    assert demeaned_additive["x"] == pytest.approx(4.230769230769)

    constructed = "y ~ 1 + post + z + post:z + I(w ** 2)"
    dummy_constructed = "y ~ 1 + C(unit) + C(time) + post + z + post:z + I(w ** 2)"
    planted = {"post": 5.0, "z": -1.5, "post:z": 0.7, "I(w ** 2)": -2.0}
    for n_units, n_periods, seed in ((3, 6, 11), (5, 3, 19)):
        panel = _constructed_term_panel(n_units, n_periods, seed)
        demeaned = _named_ols_coefs(_fit_panel(panel, constructed, "demeaned"))
        dummies = _named_ols_coefs(_fit_panel(panel, dummy_constructed, "dummies"))
        for name, value in planted.items():
            assert demeaned[name] == pytest.approx(value, abs=1e-8)
            assert dummies[name] == pytest.approx(demeaned[name])

    square_rows = []
    for unit in range(4):
        for time in range(5):
            w = float((unit * 3 + time) % 4)
            square_rows.append(
                {
                    "unit": f"u{unit}",
                    "time": time,
                    "w": w,
                    "y": 3.0 * unit + 0.5 * time + 1.25 * w - 2.0 * w**2,
                }
            )
    square_panel = pd.DataFrame(square_rows)
    square_only = "y ~ 1 + w + I(w ** 2)"
    dummy_square_only = "y ~ 1 + C(unit) + C(time) + w + I(w ** 2)"
    demeaned_square = _named_ols_coefs(
        _fit_panel(square_panel, square_only, "demeaned")
    )
    dummy_square = _named_ols_coefs(
        _fit_panel(square_panel, dummy_square_only, "dummies")
    )
    assert demeaned_square["w"] == pytest.approx(1.25, abs=1e-8)
    assert demeaned_square["I(w ** 2)"] == pytest.approx(-2.0, abs=1e-8)
    assert dummy_square["w"] == pytest.approx(demeaned_square["w"])
    assert dummy_square["I(w ** 2)"] == pytest.approx(demeaned_square["I(w ** 2)"])

    one_way = "y ~ 1 + post + z + post:z + I(w ** 2)"
    dummy_one_way = "y ~ 1 + C(unit) + post + z + post:z + I(w ** 2)"
    one_way_panel = _constructed_term_panel(4, 5, 29)
    demeaned_one_way = _named_ols_coefs(
        _fit_panel(one_way_panel, one_way, "demeaned", time_fe=False)
    )
    dummy_one_way_coefs = _named_ols_coefs(
        _fit_panel(one_way_panel, dummy_one_way, "dummies", time_fe=False)
    )
    for name in planted:
        assert demeaned_one_way[name] == pytest.approx(dummy_one_way_coefs[name])


def _panel_with_internal_missing_formula_values():
    """Balanced panel with missing formula values that are not the last row."""
    rows = []
    for unit in range(4):
        for time in range(5):
            treatment = int(unit >= 2 and time >= 2)
            x = float(unit * 5 + time)
            rows.append(
                {
                    "unit": f"u{unit}",
                    "time": time,
                    "treatment": treatment,
                    "x": x,
                    "unused": 1.0,
                    "y": 100.0 + unit * 5 + time,
                }
            )
    data = pd.DataFrame(rows)
    # Not the last row. One missing outcome and one missing covariate.
    data.loc[3, "y"] = np.nan
    data.loc[7, "x"] = np.nan
    # An unused column must not decide which rows the formula keeps.
    data.loc[11, "unused"] = np.nan
    return data


def _sequential_group_demean(frame, columns, group_vars):
    demeaned = frame.loc[:, columns].astype(float).copy()
    for group_var in group_vars:
        demeaned = demeaned - demeaned.groupby(
            frame[group_var], observed=True
        ).transform("mean")
    return demeaned


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ 1 + treatment + x",
        "y ~ 1 + treatment + x + treatment:x",
    ],
)
def test_demeaned_missing_formula_rows_match_dummy_retention(formula):
    """Missing formula values must not crash demeaned init or change retained rows.

    Patsy drops incomplete formula rows before the design matrix is built. The
    within transform has to use those rows, not the full frame index, and must
    not assume the dropped row is last.
    """
    data = _panel_with_internal_missing_formula_values()
    assert int(data.index[-1]) not in {3, 7}
    demeaned = cp.PanelRegression(
        data,
        formula=formula,
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="demeaned",
        model=LinearRegression(),
    )
    dummies = cp.PanelRegression(
        data,
        formula="y ~ 1 + C(unit) + C(time) + " + formula.split("~", 1)[1],
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    )

    used = ["y", "treatment", "x"]
    retained = data.loc[data[used].notna().all(axis=1)].copy()
    assert len(retained) == len(data) - 2
    assert 3 not in retained.index
    assert 7 not in retained.index
    assert int(data.index[-1]) in retained.index
    assert 11 in retained.index

    dummy_y = np.asarray(dummies.design["y"], dtype=float).reshape(-1)
    np.testing.assert_allclose(dummy_y, retained["y"].to_numpy(dtype=float))
    assert demeaned.design["y"].shape[0] == dummies.design["y"].shape[0]
    assert demeaned.design["X"].shape[0] == retained.shape[0]

    demeaned_x = np.asarray(demeaned.design["X"], dtype=float)
    within = _sequential_group_demean(retained, ["treatment", "x"], ["unit", "time"])
    within["treatment:x"] = retained["treatment"].to_numpy(dtype=float) * retained[
        "x"
    ].to_numpy(dtype=float)
    within["treatment:x"] = _sequential_group_demean(
        within.assign(
            unit=retained["unit"].to_numpy(), time=retained["time"].to_numpy()
        ),
        ["treatment:x"],
        ["unit", "time"],
    )["treatment:x"]
    for name in ("treatment", "x", "treatment:x"):
        if name not in demeaned.labels:
            continue
        column = demeaned.labels.index(name)
        np.testing.assert_allclose(
            demeaned_x[:, column],
            within[name].to_numpy(dtype=float),
            atol=1e-8,
        )


def test_demeaned_log_interaction_matches_dummy_fe():
    """A named interaction other than treatment:x is the within transform of the product."""
    rows = []
    for unit in range(4):
        for time in range(5):
            treated = int(unit >= 1 and time >= 2)
            exposure = float(1 + unit + 2 * time)
            log_exposure = np.log(exposure)
            y = (
                4.0 * unit
                + 0.25 * time
                + 1.7 * treated
                - 0.8 * log_exposure
                + 2.4 * treated * log_exposure
            )
            rows.append(
                {
                    "unit": f"u{unit}",
                    "time": time,
                    "treated": treated,
                    "exposure": exposure,
                    "y": y,
                }
            )
    panel = pd.DataFrame(rows)
    formula = "y ~ 1 + treated + np.log(exposure) + treated:np.log(exposure)"
    dummy = (
        "y ~ 1 + C(unit) + C(time) + treated + np.log(exposure) "
        "+ treated:np.log(exposure)"
    )
    demeaned = _named_ols_coefs(_fit_panel(panel, formula, "demeaned"))
    dummies = _named_ols_coefs(_fit_panel(panel, dummy, "dummies"))
    expected = {
        "treated": 1.7,
        "np.log(exposure)": -0.8,
        "treated:np.log(exposure)": 2.4,
    }
    for name, value in expected.items():
        assert demeaned[name] == pytest.approx(value, abs=1e-8)
        assert dummies[name] == pytest.approx(value, abs=1e-8)


def test_demeaned_intercept_only_keeps_intercept_and_demeans_outcome():
    """An intercept-only formula has no regressor columns left to demean."""
    rows = []
    for unit in range(3):
        for time in range(4):
            rows.append(
                {
                    "unit": f"u{unit}",
                    "time": time,
                    "y": 10.0 * unit + 1.0 * time + 0.5,
                }
            )
    panel = pd.DataFrame(rows)
    result = cp.PanelRegression(
        panel,
        formula="y ~ 1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="demeaned",
        model=LinearRegression(),
    )
    within_y = _sequential_group_demean(panel, ["y"], ["unit", "time"])
    np.testing.assert_allclose(
        np.asarray(result.design["y"], dtype=float).reshape(-1),
        within_y["y"].to_numpy(dtype=float),
    )
    intercept = np.asarray(result.design["X"], dtype=float)
    assert intercept.shape == (len(panel), 1)
    np.testing.assert_allclose(intercept, np.ones((len(panel), 1)))
    recorded = result._group_means["unit"].copy()
    result._record_original_group_means("unit")
    pd.testing.assert_frame_equal(result._group_means["unit"], recorded)


def test_sequential_within_preserves_one_dimensional_outcome():
    """A one-column outcome vector stays one-dimensional after demeaning."""
    panel = _cited_interaction_panel()
    result = cp.PanelRegression(
        panel,
        formula="y ~ 1 + treatment",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="demeaned",
        model=LinearRegression(),
    )
    transformed = result._sequential_within(
        panel["y"].to_numpy(dtype=float),
        panel,
        ["unit"],
    )
    assert transformed.ndim == 1
    expected = panel["y"] - panel.groupby("unit", observed=True)["y"].transform("mean")
    np.testing.assert_allclose(transformed, expected.to_numpy(dtype=float))


def test_kept_row_action_reports_alignment_and_rejects_unreported_drops():
    """Row alignment must not guess when Patsy drops rows without a mask."""
    from causalpy.experiments.panel_regression import _KeptRowNAAction

    action = _KeptRowNAAction()
    assert action.handle_NA([], [], []) == []
    assert action.kept_rows is None
    data = pd.DataFrame({"y": [1.0, 2.0]})
    np.testing.assert_array_equal(
        action.rows_for(data, np.array([1.0, 2.0])),
        np.ones(2, dtype=bool),
    )
    with pytest.raises(ValueError, match="without reporting which rows were kept"):
        action.rows_for(data, np.array([1.0]))
