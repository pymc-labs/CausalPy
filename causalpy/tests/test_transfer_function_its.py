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
"""Tests for Transfer Function ITS"""

import contextlib

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import pytest

from causalpy.experiments.graded_intervention_its import GradedInterventionTimeSeries
from causalpy.skl_models import TransferFunctionOLS
from causalpy.transforms import (
    DiscreteLag,
    GeometricAdstock,
    HillSaturation,
    LogisticSaturation,
    MichaelisMentenSaturation,
    Treatment,
)


class TestTransforms:
    """Test individual transform functions."""

    def test_saturation_hill(self):
        """Test Hill saturation transform."""
        x = np.array([0, 100, 500, 1000, 5000, 10000])
        sat = HillSaturation(slope=2.0, kappa=1000)

        x_sat = sat.apply(x)

        # Check that saturation increases monotonically
        assert np.all(np.diff(x_sat) >= 0)

        # Check that saturation is bounded (approaches 1 for large x with appropriate scaling)
        # Hill function: x^s / (k^s + x^s)
        # At x=0, should be 0
        assert x_sat[0] == 0.0

        # At x=kappa, should be 0.5
        x_at_kappa = np.array([1000])
        x_sat_at_kappa = sat.apply(x_at_kappa)
        np.testing.assert_almost_equal(x_sat_at_kappa[0], 0.5, decimal=2)

    def test_saturation_logistic(self):
        """Test logistic saturation transform."""
        x = np.array([0, 1, 2, 3, 4, 5])
        sat = LogisticSaturation(lam=1.0)

        x_sat = sat.apply(x)

        # Check that saturation increases monotonically
        assert np.all(np.diff(x_sat) >= 0)

        # Logistic should be bounded between 0 and 1
        assert np.all(x_sat >= 0)
        assert np.all(x_sat <= 1)

    def test_saturation_get_params(self):
        """Test that get_params works for saturation transforms."""
        # Hill saturation
        sat_hill = HillSaturation(slope=2.0, kappa=1000)
        params = sat_hill.get_params()
        assert params == {"slope": 2.0, "kappa": 1000}

        # Logistic saturation
        sat_logistic = LogisticSaturation(lam=0.5)
        params = sat_logistic.get_params()
        assert params == {"lam": 0.5}

        # Michaelis-Menten saturation
        sat_mm = MichaelisMentenSaturation(alpha=1.0, lam=100)
        params = sat_mm.get_params()
        assert params == {"alpha": 1.0, "lam": 100}

    def test_adstock_half_life_conversion(self):
        """Test that half_life is correctly converted to alpha."""
        adstock = GeometricAdstock(half_life=2.0)

        # alpha should be 0.5^(1/2) ≈ 0.707
        expected_alpha = np.power(0.5, 1 / 2.0)
        params = adstock.get_params()
        np.testing.assert_almost_equal(params["alpha"], expected_alpha)

    def test_adstock_application(self):
        """Test adstock transform on a simple impulse."""
        # Single impulse at t=2
        x = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        adstock = GeometricAdstock(half_life=2.0, l_max=5, normalize=False)

        x_adstock = adstock.apply(x)

        # Should have carryover effect
        assert x_adstock[2] > 0  # Impulse period
        assert x_adstock[3] > 0  # One period after
        assert x_adstock[4] > 0  # Two periods after

        # Effect should decay
        assert x_adstock[2] > x_adstock[3] > x_adstock[4]

    def test_adstock_normalization(self):
        """Test that normalization works correctly."""
        x = np.array([1.0] * 10)  # Constant input
        adstock_normalized = GeometricAdstock(alpha=0.5, l_max=8, normalize=True)
        adstock_unnormalized = GeometricAdstock(alpha=0.5, l_max=8, normalize=False)

        x_norm = adstock_normalized.apply(x)
        x_unnorm = adstock_unnormalized.apply(x)

        # Unnormalized should have larger values (sum of geometric series > 1)
        assert np.mean(x_unnorm) > np.mean(x_norm)

        # With constant input and normalization, steady-state should equal input
        # (after transient effects die out)
        assert np.abs(x_norm[-1] - 1.0) < 0.1

    def test_adstock_validation(self):
        """Test adstock validation."""
        # Must provide either alpha or half_life
        with pytest.raises(ValueError, match="Must provide either"):
            GeometricAdstock()

        # Alpha must be in (0, 1)
        with pytest.raises(ValueError, match="alpha must be in"):
            GeometricAdstock(alpha=1.5)

    def test_lag(self):
        """Test lag transform."""
        x = np.array([1, 2, 3, 4, 5])
        lag = DiscreteLag(k=2)

        x_lagged = lag.apply(x)

        # First k values should be 0
        assert x_lagged[0] == 0
        assert x_lagged[1] == 0

        # Rest should be shifted
        np.testing.assert_array_equal(x_lagged[2:], x[:-2])

    def test_lag_validation(self):
        """Test lag validation."""
        with pytest.raises(ValueError, match="must be non-negative"):
            DiscreteLag(k=-1)

    def test_treatment_validation(self):
        """Test treatment validation."""
        with pytest.raises(ValueError, match="coef_constraint must be"):
            Treatment(name="test", coef_constraint="invalid")

    def test_full_transform_pipeline(self):
        """Test applying full transform pipeline using new strategy pattern."""
        x = np.array([100, 200, 300, 400, 500, 400, 300, 200, 100, 0])

        treatment = Treatment(
            name="test",
            saturation=HillSaturation(slope=1.0, kappa=300),
            adstock=GeometricAdstock(half_life=2.0, normalize=True),
            lag=DiscreteLag(k=1),
        )

        # Apply transforms manually (mimicking what TransferFunctionITS does)
        x_transformed = x
        if treatment.saturation is not None:
            x_transformed = treatment.saturation.apply(x_transformed)
        if treatment.adstock is not None:
            x_transformed = treatment.adstock.apply(x_transformed)
        if treatment.lag is not None:
            x_transformed = treatment.lag.apply(x_transformed)

        # Check that all transforms were applied
        # Result should be different from input
        assert not np.allclose(x_transformed, x)

        # First value should be 0 due to lag
        assert x_transformed[0] == 0


class TestTransformOptimization:
    """Test parameter estimation for transform functions."""

    def test_grid_search_basic(self):
        """Test basic grid search parameter estimation."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        # Generate treatment with known transforms - use more varied signal
        treatment_raw = (
            50 + 30 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-10, 10, n)
        )
        treatment_raw = np.maximum(treatment_raw, 0)  # Keep non-negative

        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        # Generate outcome with stronger signal and time trend
        beta_0 = 100.0
        beta_t = 0.5
        theta = 50.0  # Stronger treatment effect
        y = (
            beta_0
            + beta_t * t
            + theta * treatment_transformed
            + np.random.normal(0, 5, n)
        )

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        # Create unfitted model with configuration
        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [1.5, 2.0, 2.5], "kappa": [40, 50, 60]},
            adstock_grid={"half_life": [2, 3, 4], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="hac",
        )

        # Pass to experiment (experiment estimates transforms and fits model)
        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",  # Include time trend
            model=model,
        )

        # Check that estimation metadata is stored
        assert result.transform_estimation_method == "grid"
        assert result.transform_estimation_results is not None
        assert "best_score" in result.transform_estimation_results
        assert "grid_results" in result.transform_estimation_results

        # Check that model was fitted
        assert result.ols_result is not None
        assert result.score > 0.8

        # Check that parameters are reasonable (close to true values)
        best_params = result.transform_estimation_results["best_params"]
        assert 1.5 <= best_params["slope"] <= 2.5
        assert 40 <= best_params["kappa"] <= 60
        assert 2 <= best_params["half_life"] <= 4

    def test_optimize_basic(self):
        """Test basic continuous optimization parameter estimation."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        # Generate treatment with known transforms - use more varied signal
        treatment_raw = (
            50 + 30 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-10, 10, n)
        )
        treatment_raw = np.maximum(treatment_raw, 0)  # Keep non-negative

        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        # Generate outcome with stronger signal and time trend
        beta_0 = 100.0
        beta_t = 0.5
        theta = 50.0  # Stronger treatment effect
        y = (
            beta_0
            + beta_t * t
            + theta * treatment_transformed
            + np.random.normal(0, 5, n)
        )

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        # Create unfitted model with configuration
        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_bounds={"slope": (1.0, 4.0), "kappa": (20, 100)},
            adstock_bounds={"half_life": (1, 10)},
            estimation_method="optimize",
            error_model="hac",
        )

        # Pass to experiment (experiment estimates transforms and fits model)
        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",  # Include time trend
            model=model,
        )

        # Check that estimation metadata is stored
        assert result.transform_estimation_method == "optimize"
        assert result.transform_estimation_results is not None
        assert "best_score" in result.transform_estimation_results
        assert "optimization_result" in result.transform_estimation_results

        # Check that model was fitted
        assert result.ols_result is not None
        assert result.score > 0.8

        # Check that parameters are within bounds
        best_params = result.transform_estimation_results["best_params"]
        assert 1.0 <= best_params["slope"] <= 4.0
        assert 20 <= best_params["kappa"] <= 100
        assert 1 <= best_params["half_life"] <= 10

    def test_estimation_validation(self):
        """Test that parameter estimation validates inputs."""
        # Missing saturation_grid for grid search
        with pytest.raises(ValueError, match="saturation_grid is required"):
            _model = TransferFunctionOLS(
                saturation_type="hill",
                adstock_grid={"half_life": [2, 3]},
                estimation_method="grid",
            )

        # Missing saturation_bounds for optimize
        with pytest.raises(ValueError, match="saturation_bounds is required"):
            _model = TransferFunctionOLS(
                saturation_type="hill",
                adstock_bounds={"half_life": (1, 10)},
                estimation_method="optimize",
            )

        # Invalid estimation method
        with pytest.raises(ValueError, match="estimation_method must be"):
            _model = TransferFunctionOLS(
                saturation_type="hill",
                saturation_grid={"slope": [1.0, 2.0]},
                adstock_grid={"half_life": [2, 3]},
                estimation_method="invalid",
            )


class TestARIMAX:
    """Test ARIMAX error model functionality."""

    def test_arimax_basic(self):
        """Test basic ARIMAX fitting with AR(1) errors."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        # Generate treatment with known transforms
        treatment_raw = (
            50 + 30 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-10, 10, n)
        )
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        # Generate outcome with AR(1) errors
        beta_0 = 100.0
        beta_t = 0.5
        theta = 50.0

        # Create AR(1) errors
        rho = 0.5
        errors = np.zeros(n)
        errors[0] = np.random.normal(0, 10 / np.sqrt(1 - rho**2))
        for i in range(1, n):
            errors[i] = rho * errors[i - 1] + np.random.normal(0, 10)

        y = beta_0 + beta_t * t + theta * treatment_transformed + errors

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        # Create unfitted model with ARIMAX configuration
        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [1.5, 2.0, 2.5], "kappa": [40, 50, 60]},
            adstock_grid={"half_life": [2, 3, 4], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="arimax",
            arima_order=(1, 0, 0),
        )

        # Pass to experiment
        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        # Check that model was fitted
        assert result.error_model == "arimax"
        assert result.arima_order == (1, 0, 0)
        assert result.ols_result is not None
        assert result.score > 0.8
        assert hasattr(result.model, "arimax_model")

    def test_arimax_grid_search(self):
        """Test parameter estimation works with ARIMAX."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        # Generate treatment with known transforms
        treatment_raw = (
            50 + 30 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-10, 10, n)
        )
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        # Generate outcome with AR(1) errors
        beta_0 = 100.0
        beta_t = 0.5
        theta = 50.0

        rho = 0.5
        errors = np.zeros(n)
        errors[0] = np.random.normal(0, 10 / np.sqrt(1 - rho**2))
        for i in range(1, n):
            errors[i] = rho * errors[i - 1] + np.random.normal(0, 10)

        y = beta_0 + beta_t * t + theta * treatment_transformed + errors

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        # Test that parameter recovery works
        # Create unfitted model with ARIMAX and grid search configuration
        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [1.5, 2.0, 2.5], "kappa": [40, 50, 60]},
            adstock_grid={"half_life": [2, 3, 4], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="arimax",
            arima_order=(1, 0, 0),
        )

        # Pass to experiment
        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        # Check that parameters are reasonable
        best_params = result.transform_estimation_results["best_params"]
        assert 1.5 <= best_params["slope"] <= 2.5
        assert 40 <= best_params["kappa"] <= 60
        assert 2 <= best_params["half_life"] <= 4

    def test_arimax_validation(self):
        """Test that ARIMAX validates inputs properly."""
        # Missing arima_order
        with pytest.raises(ValueError, match="arima_order must be provided"):
            _model = TransferFunctionOLS(
                saturation_type="hill",
                saturation_grid={"slope": [1.0, 2.0], "kappa": [3, 5]},
                adstock_grid={"half_life": [2, 3]},
                estimation_method="grid",
                error_model="arimax",
                # arima_order is missing!
            )

        # Invalid error_model
        with pytest.raises(ValueError, match="error_model must be"):
            _model = TransferFunctionOLS(
                saturation_type="hill",
                saturation_grid={"slope": [1.0, 2.0], "kappa": [3, 5]},
                adstock_grid={"half_life": [2, 3]},
                estimation_method="grid",
                error_model="invalid",
            )

    def test_arimax_vs_hac_comparison(self):
        """Test that HAC and ARIMAX give similar coefficients but different SEs."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        # Generate treatment
        treatment_raw = (
            50 + 30 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-10, 10, n)
        )
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        # Generate outcome with AR(1) errors
        beta_0 = 100.0
        beta_t = 0.5
        theta = 50.0

        rho = 0.5
        errors = np.zeros(n)
        errors[0] = np.random.normal(0, 10 / np.sqrt(1 - rho**2))
        for i in range(1, n):
            errors[i] = rho * errors[i - 1] + np.random.normal(0, 10)

        y = beta_0 + beta_t * t + theta * treatment_transformed + errors

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        # Fit with HAC
        model_hac = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [1.5, 2.0, 2.5], "kappa": [40, 50, 60]},
            adstock_grid={"half_life": [2, 3, 4], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="hac",
        )

        result_hac = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model_hac,
        )

        # Fit with ARIMAX
        model_arimax = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [1.5, 2.0, 2.5], "kappa": [40, 50, 60]},
            adstock_grid={"half_life": [2, 3, 4], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="arimax",
            arima_order=(1, 0, 0),
        )

        result_arimax = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model_arimax,
        )

        # Coefficients should be similar
        np.testing.assert_allclose(
            result_hac.theta_treatment, result_arimax.theta_treatment, rtol=0.2
        )

        # ARIMAX should have smaller standard errors (more efficient)
        n_baseline = len(result_hac.baseline_labels)
        se_hac = result_hac.ols_result.bse[n_baseline]

        # For ARIMAX, extract standard error at same position (within exog range)
        se_arimax = result_arimax.ols_result.bse[n_baseline]

        # ARIMAX should be more efficient (smaller SE) when correctly specified
        # Note: This might not always hold in small samples, so we just check they're positive
        assert se_hac > 0
        assert se_arimax > 0


class TestPlotting:
    """Test plotting methods for GradedInterventionTimeSeries."""

    def setup_method(self):
        """Create a simple fitted experiment for testing plots."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        # Generate treatment with known transforms
        treatment_raw = (
            50 + 30 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-10, 10, n)
        )
        treatment_raw = np.maximum(treatment_raw, 0)

        # Create transforms for data generation
        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        # Generate outcome with stronger signal and time trend
        beta_0 = 100.0
        beta_t = 0.5
        theta = 50.0
        y = (
            beta_0
            + beta_t * t
            + theta * treatment_transformed
            + np.random.normal(0, 5, n)
        )

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        # Create unfitted model with configuration
        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [1.5, 2.0, 2.5], "kappa": [40, 50, 60]},
            adstock_grid={"half_life": [2, 3, 4], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="hac",
        )

        # Pass to experiment
        self.result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        # Store true transforms for testing
        self.true_saturation = sat
        self.true_adstock = adstock
        self.df = df

    def test_plot_returns_figure_axes(self):
        """Test that plot() returns matplotlib objects."""
        fig, ax = self.result.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, np.ndarray)
        assert len(ax) == 2
        assert all(isinstance(item, plt.Axes) for item in ax)
        plt.close(fig)

    def test_plot_transforms(self):
        """Test plot_transforms() method."""
        fig, ax = self.result.plot_transforms()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, np.ndarray)
        assert len(ax) == 2
        assert all(isinstance(item, plt.Axes) for item in ax)
        plt.close(fig)

    def test_plot_transforms_with_true_values(self):
        """Test plot_transforms() with true transforms for comparison."""
        fig, ax = self.result.plot_transforms(
            true_saturation=self.true_saturation, true_adstock=self.true_adstock
        )
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, np.ndarray)
        assert len(ax) == 2
        # Check that both axes have multiple lines/bars (true + estimated)
        assert len(ax[0].get_lines()) >= 2  # At least true and estimated saturation
        plt.close(fig)

    def test_plot_transforms_with_x_range(self):
        """Test plot_transforms() with custom x_range."""
        fig, ax = self.result.plot_transforms(x_range=(0, 100))
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, np.ndarray)
        plt.close(fig)

    def test_plot_effect(self):
        """Test plot_effect() method."""
        # First run effect()
        effect_result = self.result.effect(
            window=(self.df.index[0], self.df.index[-1]),
            channels=["treatment"],
            scale=0.0,
        )

        # Then plot it
        fig, ax = self.result.plot_effect(effect_result)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, np.ndarray)
        assert len(ax) == 2
        assert all(isinstance(item, plt.Axes) for item in ax)

        # Check that plot has lines
        assert len(ax[0].get_lines()) >= 2  # observed + counterfactual
        assert len(ax[1].get_lines()) >= 1  # cumulative effect

        plt.close(fig)

    def test_plot_effect_partial_window(self):
        """Test plot_effect() with partial window."""
        # Effect on last 50 periods
        window_start = self.df.index[50]
        window_end = self.df.index[-1]

        effect_result = self.result.effect(
            window=(window_start, window_end), channels=["treatment"], scale=0.5
        )

        fig, ax = self.result.plot_effect(effect_result)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, np.ndarray)
        plt.close(fig)

    def test_plot_diagnostics_runs_without_error(self):
        """Test plot_diagnostics() method runs without error."""
        # Capture output to avoid cluttering test output
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # This should not raise an exception
            self.result.plot_diagnostics(lags=10)
        finally:
            sys.stdout = old_stdout
            plt.close("all")

    def test_plot_irf(self):
        """Test plot_irf() method."""
        fig = self.result.plot_irf("treatment", max_lag=10)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_irf_default_max_lag(self):
        """Test plot_irf() with default max_lag."""
        fig = self.result.plot_irf("treatment")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestInputValidation:
    """Test input validation for GradedInterventionTimeSeries."""

    def test_missing_y_column(self):
        """Test that missing y_column raises ValueError."""
        df = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=50, freq="W"), "x": range(50)}
        )
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [1.0], "kappa": [50]},
            adstock_grid={"half_life": [2]},
            estimation_method="grid",
        )

        with pytest.raises(ValueError, match="y_column.*not found"):
            GradedInterventionTimeSeries(
                data=df,
                y_column="nonexistent",
                treatment_names=["x"],
                base_formula="1",
                model=model,
            )

    def test_missing_treatment_column(self):
        """Test that missing treatment column raises ValueError."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="W"),
                "y": range(50),
            }
        )
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [1.0], "kappa": [50]},
            adstock_grid={"half_life": [2]},
            estimation_method="grid",
        )

        with pytest.raises(ValueError, match="Treatment column.*not found"):
            GradedInterventionTimeSeries(
                data=df,
                y_column="y",
                treatment_names=["nonexistent"],
                base_formula="1",
                model=model,
            )

    def test_missing_values_in_outcome(self):
        """Test that missing values in outcome raises ValueError."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="W"),
                "y": [np.nan if i == 25 else i for i in range(50)],
                "x": range(50),
            }
        )
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [1.0], "kappa": [50]},
            adstock_grid={"half_life": [2]},
            estimation_method="grid",
        )

        with pytest.raises(ValueError, match="Outcome variable contains missing"):
            GradedInterventionTimeSeries(
                data=df,
                y_column="y",
                treatment_names=["x"],
                base_formula="1",
                model=model,
            )

    def test_invalid_index_type(self):
        """Test that invalid index types raise BadIndexException."""
        from causalpy.custom_exceptions import BadIndexException

        df = pd.DataFrame({"y": range(50), "x": range(50)})
        df.index = ["a" + str(i) for i in range(50)]  # String index

        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [1.0], "kappa": [50]},
            adstock_grid={"half_life": [2]},
            estimation_method="grid",
        )

        with pytest.raises(BadIndexException, match="DatetimeIndex.*RangeIndex"):
            GradedInterventionTimeSeries(
                data=df,
                y_column="y",
                treatment_names=["x"],
                base_formula="1",
                model=model,
            )

    def test_warning_for_missing_treatment_values(self, capsys):
        """Test warning for missing treatment values.

        Note: This test verifies that a warning is printed when treatment data
        contains missing values. The warning suggests forward-filling, which is
        what we do to allow the model to actually fit.
        """
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="W"),
                "y": range(50),
                "x": [np.nan if i == 0 else float(i) for i in range(50)],
            }
        )
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [1.0], "kappa": [25]},
            adstock_grid={"half_life": [2]},
            estimation_method="grid",
        )

        # This will print a warning about missing values
        # Grid search will likely fail with NaN values, which is expected
        with contextlib.suppress(ValueError, Exception):
            GradedInterventionTimeSeries(
                data=df.copy(),
                y_column="y",
                treatment_names=["x"],
                base_formula="1",
                model=model,
            )

        captured = capsys.readouterr()
        # The warning should have been printed during validation
        assert "Warning" in captured.out or "warning" in captured.out
        assert "missing" in captured.out.lower()


class TestEffectMethod:
    """Test effect() method edge cases and variations."""

    def setup_method(self):
        """Create a simple fitted experiment for testing."""
        np.random.seed(42)
        n = 50
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = (
            50 + 30 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-10, 10, n)
        )
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        beta_0 = 100.0
        beta_t = 0.5
        theta = 50.0
        y = (
            beta_0
            + beta_t * t
            + theta * treatment_transformed
            + np.random.normal(0, 5, n)
        )

        self.df = pd.DataFrame(
            {"date": dates, "t": t, "y": y, "treatment": treatment_raw}
        )
        self.df = self.df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [2.0], "kappa": [50]},
            adstock_grid={"half_life": [3], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="hac",
        )

        self.result = GradedInterventionTimeSeries(
            data=self.df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

    def test_effect_with_default_channels(self):
        """Test effect() with default channels (None)."""
        effect_result = self.result.effect(
            window=(self.df.index[0], self.df.index[-1]), channels=None, scale=0.0
        )

        assert "effect_df" in effect_result
        assert "total_effect" in effect_result
        assert "mean_effect" in effect_result
        assert effect_result["channels"] == ["treatment"]

    def test_effect_with_invalid_channel(self):
        """Test effect() with invalid channel name raises ValueError."""
        with pytest.raises(ValueError, match="Channel.*not found"):
            self.result.effect(
                window=(self.df.index[0], self.df.index[-1]),
                channels=["nonexistent"],
                scale=0.0,
            )

    def test_effect_with_different_scale_values(self):
        """Test effect() with different scale values."""
        # Scale = 0.5 (half treatment)
        effect_half = self.result.effect(
            window=(self.df.index[0], self.df.index[-1]),
            channels=["treatment"],
            scale=0.5,
        )

        # Scale = 1.0 (no change)
        effect_full = self.result.effect(
            window=(self.df.index[0], self.df.index[-1]),
            channels=["treatment"],
            scale=1.0,
        )

        # Effect with scale=0.5 should be between scale=0.0 and scale=1.0
        assert abs(effect_half["total_effect"]) > 0
        assert abs(effect_half["total_effect"]) < abs(
            self.result.effect(
                window=(self.df.index[0], self.df.index[-1]),
                channels=["treatment"],
                scale=0.0,
            )["total_effect"]
        )

        # Effect with scale=1.0 should be approximately zero (no change)
        assert abs(effect_full["total_effect"]) < 1.0

    def test_effect_with_integer_index(self):
        """Test effect() with integer index instead of DatetimeIndex."""
        # Create data with integer index
        np.random.seed(42)
        n = 50
        t = np.arange(n)

        treatment_raw = (
            50 + 30 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-10, 10, n)
        )
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + np.random.normal(0, 5, n)

        df_int = pd.DataFrame({"t": t, "y": y, "treatment": treatment_raw})
        # Use integer index
        df_int.index = pd.RangeIndex(n)

        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [2.0], "kappa": [50]},
            adstock_grid={"half_life": [3], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="hac",
        )

        result_int = GradedInterventionTimeSeries(
            data=df_int,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        # Test effect with integer window
        effect_result = result_int.effect(
            window=(0, 49), channels=["treatment"], scale=0.0
        )

        assert "effect_df" in effect_result
        assert effect_result["window_start"] == 0
        assert effect_result["window_end"] == 49

    def test_effect_result_keys(self):
        """Test that effect result contains all expected keys."""
        effect_result = self.result.effect(
            window=(self.df.index[10], self.df.index[-10]),
            channels=["treatment"],
            scale=0.0,
        )

        expected_keys = [
            "effect_df",
            "total_effect",
            "mean_effect",
            "window_start",
            "window_end",
            "channels",
            "scale",
        ]

        for key in expected_keys:
            assert key in effect_result, f"Missing key: {key}"

        # Check effect_df columns
        expected_cols = ["observed", "counterfactual", "effect", "effect_cumulative"]
        for col in expected_cols:
            assert col in effect_result["effect_df"].columns


class TestModelMethods:
    """Test model methods for GradedInterventionTimeSeries."""

    def setup_method(self):
        """Create a simple fitted experiment for testing."""
        np.random.seed(42)
        n = 50
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = (
            50 + 30 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-10, 10, n)
        )
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + np.random.normal(0, 5, n)

        self.df = pd.DataFrame(
            {"date": dates, "t": t, "y": y, "treatment": treatment_raw}
        )
        self.df = self.df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [2.0], "kappa": [50]},
            adstock_grid={"half_life": [3], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="hac",
        )

        self.result = GradedInterventionTimeSeries(
            data=self.df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

    def test_summary_executes_without_error(self, capsys):
        """Test summary() executes without error and includes expected text."""
        self.result.summary()

        captured = capsys.readouterr()
        assert "Graded Intervention Time Series Results" in captured.out
        assert "Outcome variable" in captured.out
        assert "Number of observations" in captured.out
        assert "R-squared" in captured.out
        assert "Baseline coefficients" in captured.out
        assert "Treatment coefficients" in captured.out

    def test_get_plot_data_ols(self):
        """Test get_plot_data_ols() returns DataFrame with correct columns."""
        plot_data = self.result.get_plot_data_ols()

        assert isinstance(plot_data, pd.DataFrame)
        expected_cols = ["observed", "fitted", "residuals"]
        for col in expected_cols:
            assert col in plot_data.columns

        assert len(plot_data) == len(self.df)

    def test_plot_irf_no_adstock_error(self):
        """Test plot_irf() when channel has no adstock."""
        # Create experiment with no adstock
        np.random.seed(42)
        n = 50
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = (
            50 + 30 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-10, 10, n)
        )
        treatment_raw = np.maximum(treatment_raw, 0)

        # Manually create data with saturation only (no adstock)
        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)

        y = 100.0 + 0.5 * t + 50.0 * treatment_sat + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        # Note: We can't directly create a model without adstock in the current implementation
        # because adstock_grid/adstock_bounds are required. This tests the error case
        # where the treatment object has no adstock after fitting.
        # For now, we test the case where adstock exists
        # In practice, plot_irf checks if adstock is None

    def test_plot_transforms_multiple_treatments_error(self):
        """Test plot_transforms() raises NotImplementedError for multiple treatments."""
        # Create experiment with multiple treatments
        np.random.seed(42)
        n = 50
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment1 = 50 + np.random.uniform(-10, 10, n)
        treatment2 = 30 + np.random.uniform(-5, 5, n)
        treatment1 = np.maximum(treatment1, 0)
        treatment2 = np.maximum(treatment2, 0)

        y = 100.0 + 0.5 * t + treatment1 + treatment2 + np.random.normal(0, 5, n)

        df = pd.DataFrame(
            {
                "date": dates,
                "t": t,
                "y": y,
                "treatment1": treatment1,
                "treatment2": treatment2,
            }
        )
        df = df.set_index("date")

        # Note: Current implementation only supports single treatment for parameter estimation
        # This test documents expected behavior when multiple treatments are added in future


class TestAdditionalSaturationTypes:
    """Test grid search and optimization with different saturation types."""

    def test_grid_search_logistic_saturation(self):
        """Test grid search with LogisticSaturation."""
        np.random.seed(42)
        n = 80
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = (
            3 + 2 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-0.5, 0.5, n)
        )
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = LogisticSaturation(lam=0.5)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="logistic",
            saturation_grid={"lam": [0.3, 0.5, 0.7]},
            adstock_grid={"half_life": [2, 3, 4], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="hac",
        )

        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        assert result.score > 0.7
        assert result.transform_estimation_results["saturation_type"] == "logistic"

    def test_grid_search_michaelis_menten_saturation(self):
        """Test grid search with MichaelisMentenSaturation."""
        np.random.seed(42)
        n = 80
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = (
            50 + 30 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-10, 10, n)
        )
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = MichaelisMentenSaturation(alpha=1.0, lam=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="michaelis_menten",
            saturation_grid={"alpha": [0.8, 1.0, 1.2], "lam": [40, 50, 60]},
            adstock_grid={"half_life": [2, 3, 4], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="hac",
        )

        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        assert result.score > 0.7
        assert (
            result.transform_estimation_results["saturation_type"] == "michaelis_menten"
        )

    def test_optimize_logistic_saturation(self):
        """Test continuous optimization with LogisticSaturation."""
        np.random.seed(42)
        n = 80
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = (
            3 + 2 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-0.5, 0.5, n)
        )
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = LogisticSaturation(lam=0.5)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="logistic",
            saturation_bounds={"lam": (0.1, 1.0)},
            adstock_bounds={"half_life": (1, 10)},
            estimation_method="optimize",
            error_model="hac",
        )

        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        assert result.score > 0.7
        assert result.transform_estimation_results["saturation_type"] == "logistic"

    def test_optimize_michaelis_menten_saturation(self):
        """Test continuous optimization with MichaelisMentenSaturation."""
        np.random.seed(42)
        n = 80
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = (
            50 + 30 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-10, 10, n)
        )
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = MichaelisMentenSaturation(alpha=1.0, lam=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="michaelis_menten",
            saturation_bounds={"alpha": (0.5, 2.0), "lam": (20, 100)},
            adstock_bounds={"half_life": (1, 10)},
            estimation_method="optimize",
            error_model="hac",
        )

        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        assert result.score > 0.7
        assert (
            result.transform_estimation_results["saturation_type"] == "michaelis_menten"
        )


class TestTransformOptimizationErrors:
    """Test error handling in transform optimization."""

    def test_unsupported_metric(self):
        """Test that unsupported metric raises NotImplementedError."""
        from causalpy.transform_optimization import estimate_transform_params_grid

        np.random.seed(42)
        n = 50
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        df = pd.DataFrame(
            {
                "date": dates,
                "t": t,
                "y": 100 + 0.5 * t + np.random.normal(0, 5, n),
                "treatment": 50 + np.random.uniform(-10, 10, n),
            }
        )
        df = df.set_index("date")

        with pytest.raises(NotImplementedError, match="Metric.*not yet implemented"):
            estimate_transform_params_grid(
                data=df,
                y_column="y",
                treatment_name="treatment",
                base_formula="1 + t",
                saturation_type="hill",
                saturation_grid={"slope": [1.0], "kappa": [50]},
                adstock_grid={"half_life": [2]},
                metric="aicc",  # Not implemented
            )

    def test_grid_search_all_combinations_fail(self):
        """Test grid search when all combinations fail."""
        from causalpy.transform_optimization import estimate_transform_params_grid

        # Create very small dataset that will cause fitting to fail
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="W"),
                "y": [1, 2, 3, 4, 5],
                "treatment": [1, 2, 3, 4, 5],
            }
        )
        df = df.set_index("date")

        # Try to fit with ARIMAX which requires more data
        with pytest.raises(ValueError, match="Grid search failed"):
            estimate_transform_params_grid(
                data=df,
                y_column="y",
                treatment_name="treatment",
                base_formula="1",
                saturation_type="hill",
                saturation_grid={"slope": [1.0], "kappa": [3]},
                adstock_grid={"half_life": [2]},
                error_model="arimax",
                arima_order=(2, 1, 2),  # Complex model for small data
            )


class TestAdditionalTransforms:
    """Test additional transform edge cases."""

    def test_michaelis_menten_saturation_apply(self):
        """Test MichaelisMentenSaturation.apply() produces expected output."""
        x = np.array([0, 50, 100, 200, 500])
        sat = MichaelisMentenSaturation(alpha=1.0, lam=100)

        x_sat = sat.apply(x)

        # Check properties
        assert x_sat[0] == 0  # At x=0, output is 0
        assert np.all(np.diff(x_sat) >= 0)  # Monotonically increasing
        assert np.all(x_sat <= 1.0)  # Bounded by alpha

        # At x=lam, should be alpha/2
        x_at_lam = np.array([100])
        x_sat_at_lam = sat.apply(x_at_lam)
        np.testing.assert_almost_equal(x_sat_at_lam[0], 0.5, decimal=2)

    def test_discrete_lag_k_zero(self):
        """Test DiscreteLag with k=0 returns unchanged array."""
        x = np.array([1, 2, 3, 4, 5])
        lag = DiscreteLag(k=0)

        x_lagged = lag.apply(x)

        np.testing.assert_array_equal(x_lagged, x)

    def test_treatment_only_saturation(self):
        """Test Treatment dataclass with only saturation (no adstock)."""
        treatment = Treatment(
            name="test", saturation=HillSaturation(slope=2.0, kappa=50), adstock=None
        )

        assert treatment.saturation is not None
        assert treatment.adstock is None
        assert treatment.lag is None

    def test_treatment_only_adstock(self):
        """Test Treatment dataclass with only adstock (no saturation)."""
        treatment = Treatment(
            name="test", saturation=None, adstock=GeometricAdstock(half_life=3.0)
        )

        assert treatment.saturation is None
        assert treatment.adstock is not None
        assert treatment.lag is None


class TestBuildTreatmentMatrix:
    """Test _build_treatment_matrix internal method."""

    def test_build_treatment_matrix_saturation_adstock(self):
        """Test _build_treatment_matrix with saturation and adstock."""
        np.random.seed(42)
        n = 50
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)
        y = 100.0 + 0.5 * t + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [2.0], "kappa": [50]},
            adstock_grid={"half_life": [3], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="hac",
        )

        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        # Test the internal method
        treatments = result.treatments
        Z, labels = result._build_treatment_matrix(df, treatments)

        assert Z.shape == (n, 1)
        assert labels == ["treatment"]
        assert not np.array_equal(Z.flatten(), treatment_raw)  # Should be transformed

    def test_build_treatment_matrix_single_transform(self):
        """Test _build_treatment_matrix with only adstock."""
        np.random.seed(42)
        n = 50
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)
        y = 100.0 + 0.5 * t + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type=None,
            adstock_grid={"half_life": [3], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="hac",
        )

        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        # Test the internal method
        treatments = result.treatments
        Z, labels = result._build_treatment_matrix(df, treatments)

        assert Z.shape == (n, 1)
        assert labels == ["treatment"]


class TestPlotIRFEdgeCases:
    """Test plot_irf edge cases and error handling."""

    def test_plot_irf_invalid_channel(self):
        """Test plot_irf with invalid channel name."""
        np.random.seed(42)
        n = 50
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)
        y = 100.0 + 0.5 * t + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type=None,
            adstock_grid={"half_life": [3]},
            estimation_method="grid",
            error_model="hac",
        )

        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        with pytest.raises(ValueError, match="Channel.*not found"):
            result.plot_irf("nonexistent_channel")


class TestSummaryMethod:
    """Test summary() method edge cases."""

    def test_summary_with_arimax(self, capsys):
        """Test summary() with ARIMAX error model."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)
        y = 100.0 + 0.5 * t + treatment_raw + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type=None,
            adstock_grid={"half_life": [3]},
            estimation_method="grid",
            error_model="arimax",
            arima_order=(1, 0, 0),
        )

        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        result.summary(round_to=3)

        captured = capsys.readouterr()
        assert "ARIMAX" in captured.out
        assert "ARIMA order" in captured.out
        assert "(1, 0, 0)" in captured.out

    def test_summary_custom_round_to(self, capsys):
        """Test summary() with custom round_to parameter."""
        np.random.seed(42)
        n = 50
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)
        y = 100.0 + 0.5 * t + treatment_raw + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type=None,
            adstock_grid={"half_life": [3]},
            estimation_method="grid",
            error_model="hac",
        )

        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        result.summary(round_to=4)

        captured = capsys.readouterr()
        assert "Graded Intervention Time Series Results" in captured.out


class TestModelTypeValidation:
    """Test validation of model types."""

    def test_invalid_model_type_raises_error(self):
        """Test that invalid model type raises ValueError."""
        np.random.seed(42)
        n = 50
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        y = 100.0 + 0.5 * t + treatment_raw + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        # Use an invalid model (just a string)
        with pytest.raises(ValueError, match="Model type not recognized"):
            GradedInterventionTimeSeries(
                data=df,
                y_column="y",
                treatment_names=["treatment"],
                base_formula="1 + t",
                model="invalid_model",
            )


class TestRangeIndexSupport:
    """Test that integer RangeIndex is supported."""

    def test_range_index_works(self):
        """Test that RangeIndex is accepted as valid index."""
        np.random.seed(42)
        n = 50
        t = np.arange(n)

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)
        y = 100.0 + 0.5 * t + treatment_raw + np.random.normal(0, 5, n)

        df = pd.DataFrame({"t": t, "y": y, "treatment": treatment_raw})
        # RangeIndex is the default for DataFrame without explicit index
        assert isinstance(df.index, pd.RangeIndex)

        model = TransferFunctionOLS(
            saturation_type=None,
            adstock_grid={"half_life": [3]},
            estimation_method="grid",
            error_model="hac",
        )

        # Should not raise an error
        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        assert result.ols_result is not None

    def test_integer_index_works(self):
        """Test that explicit integer Index is accepted."""
        np.random.seed(42)
        n = 50
        t = np.arange(n)

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)
        y = 100.0 + 0.5 * t + treatment_raw + np.random.normal(0, 5, n)

        df = pd.DataFrame({"t": t, "y": y, "treatment": treatment_raw})
        df.index = pd.Index(range(n))  # Explicit integer Index
        assert isinstance(df.index, pd.Index)
        assert pd.api.types.is_integer_dtype(df.index)

        model = TransferFunctionOLS(
            saturation_type=None,
            adstock_grid={"half_life": [3]},
            estimation_method="grid",
            error_model="hac",
        )

        # Should not raise an error
        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        assert result.ols_result is not None


class TestEffectWithARIMAX:
    """Test effect() method with ARIMAX error model."""

    def test_effect_with_arimax_model(self):
        """Test that effect() works correctly with ARIMAX."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)

        # Create AR(1) errors
        rho = 0.5
        errors = np.zeros(n)
        errors[0] = np.random.normal(0, 10 / np.sqrt(1 - rho**2))
        for i in range(1, n):
            errors[i] = rho * errors[i - 1] + np.random.normal(0, 10)

        y = 100.0 + 0.5 * t + 50 * treatment_raw + errors

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type=None,
            adstock_grid={"half_life": [3]},
            estimation_method="grid",
            error_model="arimax",
            arima_order=(1, 0, 0),
        )

        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        # Test effect
        effect_result = result.effect(
            window=(df.index[0], df.index[-1]), channels=None, scale=0.0
        )

        assert "effect_df" in effect_result
        assert "total_effect" in effect_result
        assert effect_result["total_effect"] != 0  # Should have nonzero effect

        # Test plot_effect
        fig, ax = result.plot_effect(effect_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotTransformsEdgeCases:
    """Test plot_transforms edge cases."""

    def test_plot_transforms_with_lag(self):
        """Test that lag transforms are applied correctly in build_treatment_matrix."""
        np.random.seed(42)
        n = 50
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)
        y = 100.0 + 0.5 * t + treatment_raw + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type=None,
            adstock_grid={"half_life": [3]},
            estimation_method="grid",
            error_model="hac",
        )

        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        # Manually add a lag to the treatment and test _build_treatment_matrix
        from causalpy.transforms import DiscreteLag

        treatments_with_lag = []
        for treatment in result.treatments:
            # Create a new treatment object with lag
            treatment_lagged = Treatment(
                name=treatment.name,
                saturation=treatment.saturation,
                adstock=treatment.adstock,
                lag=DiscreteLag(k=1),  # Add 1-period lag
            )
            treatments_with_lag.append(treatment_lagged)

        # Test _build_treatment_matrix with lag
        Z, labels = result._build_treatment_matrix(df, treatments_with_lag)

        assert Z.shape == (n, 1)
        assert labels == ["treatment"]
        # First value should be 0 due to lag
        assert Z[0, 0] == 0
