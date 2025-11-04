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
"""Tests for Transfer Function ITS"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import numpy as np
import pandas as pd
import pytest

from causalpy.experiments.transfer_function_its import TransferFunctionITS
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

        # Generate treatment with known transforms
        treatment_raw = np.random.uniform(0, 100, n)
        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        # Generate outcome
        beta_0 = 100.0
        theta = 5.0
        y = beta_0 + theta * treatment_transformed + np.random.normal(0, 2, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        # Estimate parameters with grid search
        # Use a coarse grid for speed
        result = TransferFunctionITS.with_estimated_transforms(
            data=df,
            y_column="y",
            treatment_name="treatment",
            base_formula="1",
            estimation_method="grid",
            saturation_type="hill",
            saturation_grid={"slope": [1.5, 2.0, 2.5], "kappa": [40, 50, 60]},
            adstock_grid={"half_life": [2, 3, 4], "l_max": [12], "normalize": [True]},
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

        # Generate treatment with known transforms
        treatment_raw = np.random.uniform(0, 100, n)
        sat = HillSaturation(slope=2.0, kappa=50)
        treatment_sat = sat.apply(treatment_raw)
        adstock = GeometricAdstock(half_life=3.0, normalize=True)
        treatment_transformed = adstock.apply(treatment_sat)

        # Generate outcome
        beta_0 = 100.0
        theta = 5.0
        y = beta_0 + theta * treatment_transformed + np.random.normal(0, 2, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        # Estimate parameters with optimization
        result = TransferFunctionITS.with_estimated_transforms(
            data=df,
            y_column="y",
            treatment_name="treatment",
            base_formula="1",
            estimation_method="optimize",
            saturation_type="hill",
            saturation_bounds={"slope": (1.0, 4.0), "kappa": (20, 100)},
            adstock_bounds={"half_life": (1, 10)},
            initial_params={"slope": 2.0, "kappa": 50, "half_life": 3.0},
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
        df = pd.DataFrame({"y": [1, 2, 3], "x": [1, 2, 3]})

        # Missing saturation_grid for grid search
        with pytest.raises(ValueError, match="saturation_grid is required"):
            TransferFunctionITS.with_estimated_transforms(
                data=df,
                y_column="y",
                treatment_name="x",
                base_formula="1",
                estimation_method="grid",
                saturation_type="hill",
                adstock_grid={"half_life": [2, 3]},
            )

        # Missing saturation_bounds for optimize
        with pytest.raises(ValueError, match="saturation_bounds is required"):
            TransferFunctionITS.with_estimated_transforms(
                data=df,
                y_column="y",
                treatment_name="x",
                base_formula="1",
                estimation_method="optimize",
                saturation_type="hill",
                adstock_bounds={"half_life": (1, 10)},
            )

        # Invalid estimation method
        with pytest.raises(ValueError, match="Unknown estimation_method"):
            TransferFunctionITS.with_estimated_transforms(
                data=df,
                y_column="y",
                treatment_name="x",
                base_formula="1",
                estimation_method="invalid",
                saturation_type="hill",
            )
