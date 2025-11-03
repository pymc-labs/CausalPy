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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import pytest

from causalpy.experiments.transfer_function_its import TransferFunctionITS
from causalpy.transforms import (
    Adstock,
    Lag,
    Saturation,
    Treatment,
    apply_treatment_transforms,
)


class TestTransforms:
    """Test individual transform functions."""

    def test_saturation_hill(self):
        """Test Hill saturation transform."""
        x = np.array([0, 100, 500, 1000, 5000, 10000])
        sat = Saturation(kind="hill", slope=2.0, kappa=1000)

        from causalpy.transforms import apply_saturation

        x_sat = apply_saturation(x, sat)

        # Check that saturation increases monotonically
        assert np.all(np.diff(x_sat) >= 0)

        # Check that saturation is bounded (approaches 1 for large x with appropriate scaling)
        # Hill function: x^s / (k^s + x^s)
        # At x=0, should be 0
        assert x_sat[0] == 0.0

        # At x=kappa, should be 0.5
        x_at_kappa = np.array([1000])
        x_sat_at_kappa = apply_saturation(x_at_kappa, sat)
        np.testing.assert_almost_equal(x_sat_at_kappa[0], 0.5, decimal=2)

    def test_saturation_logistic(self):
        """Test logistic saturation transform."""
        x = np.array([0, 1, 2, 3, 4, 5])
        sat = Saturation(kind="logistic", lam=1.0)

        from causalpy.transforms import apply_saturation

        x_sat = apply_saturation(x, sat)

        # Check that saturation increases monotonically
        assert np.all(np.diff(x_sat) >= 0)

        # Logistic should be bounded between 0 and 1
        assert np.all(x_sat >= 0)
        assert np.all(x_sat <= 1)

    def test_saturation_validation(self):
        """Test that saturation validation works."""
        # Hill requires slope and kappa
        with pytest.raises(ValueError, match="Hill saturation requires"):
            Saturation(kind="hill", slope=2.0)

        # Logistic requires lam
        with pytest.raises(ValueError, match="Logistic saturation requires"):
            Saturation(kind="logistic")

        # Invalid kind
        with pytest.raises(ValueError, match="Unknown saturation kind"):
            Saturation(kind="invalid_kind")

    def test_adstock_half_life_conversion(self):
        """Test that half_life is correctly converted to alpha."""
        adstock = Adstock(half_life=2.0)

        # alpha should be 0.5^(1/2) ≈ 0.707
        expected_alpha = np.power(0.5, 1 / 2.0)
        np.testing.assert_almost_equal(adstock.alpha, expected_alpha)

    def test_adstock_application(self):
        """Test adstock transform on a simple impulse."""
        # Single impulse at t=2
        x = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0])
        adstock = Adstock(half_life=2.0, l_max=5, normalize=False)

        from causalpy.transforms import apply_adstock

        x_adstock = apply_adstock(x, adstock)

        # Should have carryover effect
        assert x_adstock[2] > 0  # Impulse period
        assert x_adstock[3] > 0  # One period after
        assert x_adstock[4] > 0  # Two periods after

        # Effect should decay
        assert x_adstock[2] > x_adstock[3] > x_adstock[4]

    def test_adstock_normalization(self):
        """Test that normalization works correctly."""
        x = np.array([1.0] * 10)  # Constant input
        adstock_normalized = Adstock(alpha=0.5, l_max=8, normalize=True)
        adstock_unnormalized = Adstock(alpha=0.5, l_max=8, normalize=False)

        from causalpy.transforms import apply_adstock

        x_norm = apply_adstock(x, adstock_normalized)
        x_unnorm = apply_adstock(x, adstock_unnormalized)

        # Unnormalized should have larger values (sum of geometric series > 1)
        assert np.mean(x_unnorm) > np.mean(x_norm)

        # With constant input and normalization, steady-state should equal input
        # (after transient effects die out)
        assert np.abs(x_norm[-1] - 1.0) < 0.1

    def test_adstock_validation(self):
        """Test adstock validation."""
        # Must provide either alpha or half_life
        with pytest.raises(ValueError, match="Must provide either"):
            Adstock()

        # Alpha must be in (0, 1)
        with pytest.raises(ValueError, match="alpha must be in"):
            Adstock(alpha=1.5)

    def test_lag(self):
        """Test lag transform."""
        x = np.array([1, 2, 3, 4, 5])
        lag = Lag(k=2)

        from causalpy.transforms import apply_lag

        x_lagged = apply_lag(x, lag)

        # First k values should be 0
        assert x_lagged[0] == 0
        assert x_lagged[1] == 0

        # Rest should be shifted
        np.testing.assert_array_equal(x_lagged[2:], x[:-2])

    def test_lag_validation(self):
        """Test lag validation."""
        with pytest.raises(ValueError, match="must be non-negative"):
            Lag(k=-1)

    def test_treatment_validation(self):
        """Test treatment validation."""
        with pytest.raises(ValueError, match="coef_constraint must be"):
            Treatment(name="test", coef_constraint="invalid")

    def test_full_transform_pipeline(self):
        """Test applying full transform pipeline."""
        x = np.array([100, 200, 300, 400, 500, 400, 300, 200, 100, 0])

        treatment = Treatment(
            name="test",
            transforms=[
                Saturation(kind="hill", slope=1.0, kappa=300),
                Adstock(half_life=2.0, normalize=True),
                Lag(k=1),
            ],
        )

        x_transformed = apply_treatment_transforms(x, treatment)

        # Check that all transforms were applied
        # Result should be different from input
        assert not np.allclose(x_transformed, x)

        # First value should be 0 due to lag
        assert x_transformed[0] == 0


class TestTransferFunctionITS:
    """Test the TransferFunctionITS experiment class."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple test data."""
        np.random.seed(42)
        n = 100

        # Create time series
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        # True parameters
        beta_0 = 100.0  # Intercept
        beta_1 = 0.5  # Trend
        theta = 2.0  # Treatment effect

        # Generate treatment with some variation
        treatment = np.random.uniform(0, 10, n)

        # Apply simple transforms (for simplicity, just use raw treatment)
        # y = beta_0 + beta_1 * t + theta * treatment + noise
        noise = np.random.normal(0, 5, n)
        y = beta_0 + beta_1 * t + theta * treatment + noise

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment})
        df = df.set_index("date")

        return df, beta_0, beta_1, theta

    def test_basic_fit(self, simple_data):
        """Test that model fits without errors."""
        df, beta_0, beta_1, theta = simple_data

        treatment = Treatment(name="treatment", transforms=[])

        result = TransferFunctionITS(
            data=df,
            y_column="y",
            base_formula="1 + t",
            treatments=[treatment],
            hac_maxlags=4,
        )

        # Check that model fitted
        assert result.ols_result is not None
        assert result.score > 0.9  # Should fit well for clean simulated data

        # Check coefficient recovery (approximately)
        assert np.abs(result.beta_baseline[0] - beta_0) < 10  # Intercept
        assert np.abs(result.beta_baseline[1] - beta_1) < 1  # Trend
        assert np.abs(result.theta_treatment[0] - theta) < 1  # Treatment effect

    def test_with_saturation_and_adstock(self):
        """Test model with saturation and adstock transforms."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        dates = pd.date_range("2020-01-01", periods=n, freq="W")

        # Generate treatment
        treatment_raw = np.random.uniform(0, 100, n)

        # Apply known transforms
        from causalpy.transforms import apply_adstock, apply_saturation

        sat = Saturation(kind="hill", slope=1.0, kappa=50)
        treatment_sat = apply_saturation(treatment_raw, sat)

        adstock = Adstock(half_life=2.0, normalize=True)
        treatment_transformed = apply_adstock(treatment_sat, adstock)

        # Generate outcome
        beta_0 = 100.0
        theta = 5.0
        y = beta_0 + theta * treatment_transformed + np.random.normal(0, 2, n)

        df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
        df = df.set_index("date")

        # Fit model with same transforms
        treatment_spec = Treatment(
            name="treatment",
            transforms=[
                Saturation(kind="hill", slope=1.0, kappa=50),
                Adstock(half_life=2.0, normalize=True),
            ],
        )

        result = TransferFunctionITS(
            data=df,
            y_column="y",
            base_formula="1",
            treatments=[treatment_spec],
            hac_maxlags=4,
        )

        # Should recover parameters reasonably well
        assert result.score > 0.8
        assert np.abs(result.beta_baseline[0] - beta_0) < 10
        assert np.abs(result.theta_treatment[0] - theta) < 2

    def test_effect_computation(self, simple_data):
        """Test counterfactual effect computation."""
        df, beta_0, beta_1, theta = simple_data

        treatment = Treatment(name="treatment", transforms=[])

        result = TransferFunctionITS(
            data=df,
            y_column="y",
            base_formula="1 + t",
            treatments=[treatment],
            hac_maxlags=4,
        )

        # Compute effect of zeroing treatment in weeks 50-60
        effect_result = result.effect(
            window=(df.index[50], df.index[60]), channels=["treatment"], scale=0.0
        )

        # Check that result has expected keys
        assert "effect_df" in effect_result
        assert "total_effect" in effect_result
        assert "mean_effect" in effect_result

        # Effect should be positive (removing treatment should decrease outcome)
        # Since y = ... + theta * treatment, and we're zeroing treatment,
        # effect = observed - counterfactual = (... + theta * x) - (... + 0) = theta * x > 0
        assert effect_result["total_effect"] > 0

        # Check that effect_df has correct columns
        assert "observed" in effect_result["effect_df"].columns
        assert "counterfactual" in effect_result["effect_df"].columns
        assert "effect" in effect_result["effect_df"].columns
        assert "effect_cumulative" in effect_result["effect_df"].columns

    def test_effect_with_scaling(self, simple_data):
        """Test effect computation with different scaling factors."""
        df, beta_0, beta_1, theta = simple_data

        treatment = Treatment(name="treatment", transforms=[])

        result = TransferFunctionITS(
            data=df,
            y_column="y",
            base_formula="1 + t",
            treatments=[treatment],
            hac_maxlags=4,
        )

        window = (df.index[50], df.index[60])

        # Effect of 50% reduction
        effect_50 = result.effect(window=window, channels=["treatment"], scale=0.5)

        # Effect of complete removal
        effect_0 = result.effect(window=window, channels=["treatment"], scale=0.0)

        # Complete removal should have approximately 2x the effect of 50% reduction
        ratio = effect_0["total_effect"] / effect_50["total_effect"]
        assert 1.8 < ratio < 2.2  # Allow some tolerance

    def test_validation(self):
        """Test input validation."""
        df = pd.DataFrame({"t": [1, 2, 3], "y": [10, 20, 30], "x": [1, 2, 3]})

        treatment = Treatment(name="x", transforms=[])

        # Missing y_column
        with pytest.raises(ValueError, match="not found in data columns"):
            TransferFunctionITS(
                data=df,
                y_column="missing_y",
                base_formula="1 + t",
                treatments=[treatment],
            )

        # Missing treatment column
        bad_treatment = Treatment(name="missing_x", transforms=[])
        with pytest.raises(ValueError, match="not found in data columns"):
            TransferFunctionITS(
                data=df,
                y_column="y",
                base_formula="1 + t",
                treatments=[bad_treatment],
            )

    def test_plotting_methods(self, simple_data):
        """Test that plotting methods run without errors."""
        df, beta_0, beta_1, theta = simple_data

        treatment = Treatment(name="treatment", transforms=[])

        result = TransferFunctionITS(
            data=df,
            y_column="y",
            base_formula="1 + t",
            treatments=[treatment],
            hac_maxlags=4,
        )

        # Test main plot
        fig, ax = result.plot()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

        # Test IRF plot (won't work without adstock, but should handle gracefully)
        # Add a treatment with adstock for IRF test
        df_with_adstock = df.copy()
        treatment_with_adstock = Treatment(
            name="treatment", transforms=[Adstock(half_life=2.0)]
        )

        result_with_adstock = TransferFunctionITS(
            data=df_with_adstock,
            y_column="y",
            base_formula="1 + t",
            treatments=[treatment_with_adstock],
            hac_maxlags=4,
        )

        fig_irf = result_with_adstock.plot_irf("treatment")
        assert fig_irf is not None
        plt.close(fig_irf)

    def test_summary_method(self, simple_data):
        """Test that summary method runs without errors."""
        df, beta_0, beta_1, theta = simple_data

        treatment = Treatment(name="treatment", transforms=[])

        result = TransferFunctionITS(
            data=df,
            y_column="y",
            base_formula="1 + t",
            treatments=[treatment],
            hac_maxlags=4,
        )

        # Should not raise an error
        result.summary()

    def test_get_plot_data_ols(self, simple_data):
        """Test get_plot_data_ols method."""
        df, beta_0, beta_1, theta = simple_data

        treatment = Treatment(name="treatment", transforms=[])

        result = TransferFunctionITS(
            data=df,
            y_column="y",
            base_formula="1 + t",
            treatments=[treatment],
            hac_maxlags=4,
        )

        plot_data = result.get_plot_data_ols()

        assert "observed" in plot_data.columns
        assert "fitted" in plot_data.columns
        assert "residuals" in plot_data.columns
        assert len(plot_data) == len(df)
