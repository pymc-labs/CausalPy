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
"""Tests for TransferFunctionOLS and ScikitLearnAdaptor methods."""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import numpy as np
import pandas as pd
import pytest

from causalpy.experiments.graded_intervention_its import GradedInterventionTimeSeries
from causalpy.skl_models import TransferFunctionOLS
from causalpy.transforms import GeometricAdstock, HillSaturation


class TestTransferFunctionOLSMethods:
    """Test TransferFunctionOLS model methods."""

    def setup_method(self):
        """Create fitted model for testing."""
        np.random.seed(42)
        n = 80
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

        self.model = TransferFunctionOLS(
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
            model=self.model,
        )

    def test_predict_returns_correct_shape(self):
        """Test predict() method returns correct shape."""
        X_test = self.result.X_full
        predictions = self.model.predict(X_test)

        assert predictions.shape == (len(self.df),)
        assert isinstance(predictions, np.ndarray)

    def test_predict_raises_error_before_fit(self):
        """Test predict() raises error before fit()."""
        # Create unfitted model
        model_unfitted = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [2.0], "kappa": [50]},
            adstock_grid={"half_life": [3]},
            estimation_method="grid",
        )

        X_dummy = np.random.randn(10, 3)

        with pytest.raises(ValueError, match="Model has not been fitted"):
            model_unfitted.predict(X_dummy)

    def test_get_coeffs_returns_numpy_array(self):
        """Test get_coeffs() returns coefficients as numpy array."""
        coeffs = self.model.get_coeffs()

        assert isinstance(coeffs, np.ndarray)
        assert coeffs.shape[0] > 0  # Should have at least one coefficient

    def test_print_coefficients_outputs_text(self, capsys):
        """Test print_coefficients() outputs formatted text."""
        labels = ["Intercept", "t", "treatment"]
        self.model.print_coefficients(labels, round_to=2)

        captured = capsys.readouterr()
        assert "Model coefficients:" in captured.out
        assert "Intercept" in captured.out
        assert "treatment" in captured.out

    def test_calculate_impact(self):
        """Test calculate_impact() computes difference correctly."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 32, 38, 48])

        impact = self.model.calculate_impact(y_true, y_pred)

        expected_impact = y_true - y_pred
        np.testing.assert_array_equal(impact, expected_impact)

    def test_calculate_cumulative_impact(self):
        """Test calculate_cumulative_impact() computes cumsum correctly."""
        impact = np.array([1, 2, 3, 4, 5])

        cumulative_impact = self.model.calculate_cumulative_impact(impact)

        expected_cumulative = np.cumsum(impact)
        np.testing.assert_array_equal(cumulative_impact, expected_cumulative)


class TestModelConfiguration:
    """Test various model configuration options."""

    def test_arimax_ar1_order(self):
        """Test ARIMAX with AR(1) order."""
        np.random.seed(42)
        n = 80
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        # Generate outcome with AR(1) errors
        rho = 0.5
        errors = np.zeros(n)
        errors[0] = np.random.normal(0, 10 / np.sqrt(1 - rho**2))
        for i in range(1, n):
            errors[i] = rho * errors[i - 1] + np.random.normal(0, 10)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + errors

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [2.0], "kappa": [50]},
            adstock_grid={"half_life": [3], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="arimax",
            arima_order=(1, 0, 0),  # AR(1)
        )

        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        assert result.arima_order == (1, 0, 0)
        assert result.score > 0.5

    def test_arimax_ma1_order(self):
        """Test ARIMAX with MA(1) order."""
        np.random.seed(42)
        n = 80
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + np.random.normal(0, 10, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [2.0], "kappa": [50]},
            adstock_grid={"half_life": [3], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="arimax",
            arima_order=(0, 0, 1),  # MA(1)
        )

        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        assert result.arima_order == (0, 0, 1)
        assert result.score > 0.5

    def test_arimax_arma_order(self):
        """Test ARIMAX with ARMA(2,1) order."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + np.random.normal(0, 10, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [2.0], "kappa": [50]},
            adstock_grid={"half_life": [3], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="arimax",
            arima_order=(2, 0, 1),  # ARMA(2,1)
        )

        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        assert result.arima_order == (2, 0, 1)
        assert result.score > 0.5

    def test_hac_custom_maxlags(self):
        """Test HAC with custom maxlags."""
        np.random.seed(42)
        n = 80
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        custom_maxlags = 5
        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [2.0], "kappa": [50]},
            adstock_grid={"half_life": [3], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="hac",
            hac_maxlags=custom_maxlags,
        )

        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        assert result.hac_maxlags == custom_maxlags
        assert result.score > 0.7

    def test_hac_auto_maxlags(self):
        """Test HAC with automatically determined maxlags."""
        np.random.seed(42)
        n = 80
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [2.0], "kappa": [50]},
            adstock_grid={"half_life": [3], "l_max": [12], "normalize": [True]},
            estimation_method="grid",
            error_model="hac",
            hac_maxlags=None,  # Auto
        )

        result = GradedInterventionTimeSeries(
            data=df,
            y_column="y",
            treatment_names=["treatment"],
            base_formula="1 + t",
            model=model,
        )

        # Should use rule of thumb: floor(4 * (n / 100)^(2/9))
        expected_maxlags = int(np.floor(4 * (n / 100) ** (2 / 9)))
        assert result.hac_maxlags == expected_maxlags
        assert result.score > 0.7

    def test_coef_constraint_stored_correctly(self):
        """Test coef_constraint parameter is stored correctly."""
        model_nonneg = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [2.0], "kappa": [50]},
            adstock_grid={"half_life": [3]},
            estimation_method="grid",
            coef_constraint="nonnegative",
        )

        assert model_nonneg.coef_constraint == "nonnegative"

        model_unconstrained = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [2.0], "kappa": [50]},
            adstock_grid={"half_life": [3]},
            estimation_method="grid",
            coef_constraint="unconstrained",
        )

        assert model_unconstrained.coef_constraint == "unconstrained"


class TestOptionalTransforms:
    """Test optional transform configurations (adstock-only, saturation-only)."""

    def test_adstock_only_grid(self):
        """Test model with only adstock transform (no saturation)."""
        np.random.seed(42)
        n = 80
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)

        # Apply only adstock (no saturation)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_raw)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type=None,  # No saturation
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

        # Check that saturation is None and adstock exists
        assert result.treatments[0].saturation is None
        assert result.treatments[0].adstock is not None
        assert result.score > 0.5

    def test_saturation_only_grid(self):
        """Test model with only saturation transform (no adstock)."""
        np.random.seed(42)
        n = 80
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = (
            50 + 30 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-10, 10, n)
        )
        treatment_raw = np.maximum(treatment_raw, 0)

        # Apply only saturation (no adstock)
        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_transformed = sat.apply(treatment_raw)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [2.0], "kappa": [50]},
            adstock_grid=None,  # No adstock
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

        # Check that saturation exists and adstock is None
        assert result.treatments[0].saturation is not None
        assert result.treatments[0].adstock is None
        assert result.score > 0.5

    def test_neither_transform_raises_error(self):
        """Test that model raises error if neither transform is specified."""
        with pytest.raises(
            ValueError, match="At least one of saturation_grid or adstock_grid"
        ):
            TransferFunctionOLS(
                saturation_type=None,
                saturation_grid=None,
                adstock_grid=None,
                estimation_method="grid",
                error_model="hac",
            )

    def test_saturation_type_without_grid_raises_error(self):
        """Test that specifying saturation_type without saturation_grid raises error."""
        with pytest.raises(
            ValueError, match="saturation_grid is required when saturation_type"
        ):
            TransferFunctionOLS(
                saturation_type="hill",
                saturation_grid=None,  # Missing grid
                adstock_grid={"half_life": [3]},
                estimation_method="grid",
                error_model="hac",
            )

    def test_adstock_only_optimize(self):
        """Test model with only adstock using optimize method."""
        np.random.seed(42)
        n = 80
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)

        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_raw)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type=None,  # No saturation
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

        # Check that saturation is None and adstock exists
        assert result.treatments[0].saturation is None
        assert result.treatments[0].adstock is not None
        assert result.score > 0.5

    def test_saturation_only_optimize(self):
        """Test model with only saturation using optimize method."""
        np.random.seed(42)
        n = 80
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = (
            50 + 30 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-10, 10, n)
        )
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_transformed = sat.apply(treatment_raw)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_bounds={"slope": (1, 5), "kappa": (30, 80)},
            adstock_bounds=None,  # No adstock
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

        # Check that saturation exists and adstock is None
        assert result.treatments[0].saturation is not None
        assert result.treatments[0].adstock is None
        assert result.score > 0.5

    def test_plot_transforms_with_adstock_only(self):
        """Test plot_transforms() with only adstock (single panel)."""
        np.random.seed(42)
        n = 80
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)

        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_raw)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + np.random.normal(0, 5, n)

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

        # Should create single-panel plot
        fig, axes = result.plot_transforms()

        assert len(axes) == 1  # Only one panel
        assert fig is not None

    def test_plot_transforms_with_saturation_only(self):
        """Test plot_transforms() with only saturation (single panel)."""
        np.random.seed(42)
        n = 80
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        treatment_raw = 50 + np.random.uniform(-10, 10, n)
        treatment_raw = np.maximum(treatment_raw, 0)

        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_transformed = sat.apply(treatment_raw)

        y = 100.0 + 0.5 * t + 50.0 * treatment_transformed + np.random.normal(0, 5, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        model = TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [2.0], "kappa": [50]},
            adstock_grid=None,
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

        # Should create single-panel plot
        fig, axes = result.plot_transforms()

        assert len(axes) == 1  # Only one panel
        assert fig is not None
