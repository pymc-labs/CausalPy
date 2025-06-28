#   Copyright 2025 - 2025 The PyMC Labs Developers
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
Tests for multiple treated units in WeightedSumFitter
"""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from causalpy.pymc_models import WeightedSumFitter

# Use consistent sample kwargs for fast testing
sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2, "progressbar": False}


@pytest.fixture
def synthetic_control_data(rng):
    """Generate synthetic data with multiple treated and control units."""
    n_obs = 50  # Smaller for faster tests
    n_control = 4
    n_treated = 3

    # Control unit data
    control_data = {}
    for i in range(n_control):
        control_data[f"control_{i}"] = rng.normal(0, 1, n_obs)

    # Treated unit data (combinations of control units with some noise)
    treated_data = {}
    for j in range(n_treated):
        # Each treated unit is a different weighted combination of controls
        weights = rng.dirichlet(np.ones(n_control))
        treated_data[f"treated_{j}"] = sum(
            weights[i] * control_data[f"control_{i}"] for i in range(n_control)
        ) + rng.normal(0, 0.1, n_obs)

    # Create DataFrame
    df = pd.DataFrame({**control_data, **treated_data})

    # Prepare data for model
    control_units = [f"control_{i}" for i in range(n_control)]
    treated_units = [f"treated_{j}" for j in range(n_treated)]

    X = xr.DataArray(
        df[control_units].values,
        dims=["obs_ind", "coeffs"],
        coords={
            "obs_ind": df.index,
            "coeffs": control_units,
        },
    )

    y = xr.DataArray(
        df[treated_units].values,
        dims=["obs_ind", "treated_units"],
        coords={
            "obs_ind": df.index,
            "treated_units": treated_units,
        },
    )

    coords = {
        "coeffs": control_units,
        "treated_units": treated_units,
        "obs_ind": np.arange(n_obs),
    }

    return X, y, coords, control_units, treated_units


@pytest.fixture
def single_treated_data(rng):
    """Generate synthetic data with single treated unit for backward compatibility testing."""
    n_obs = 50
    n_control = 4

    # Control unit data
    control_data = {}
    for i in range(n_control):
        control_data[f"control_{i}"] = rng.normal(0, 1, n_obs)

    # Single treated unit data
    weights = rng.dirichlet(np.ones(n_control))
    treated_data = {
        "treated_0": sum(
            weights[i] * control_data[f"control_{i}"] for i in range(n_control)
        )
        + rng.normal(0, 0.1, n_obs)
    }

    # Create DataFrame
    df = pd.DataFrame({**control_data, **treated_data})

    # Prepare data for model
    control_units = [f"control_{i}" for i in range(n_control)]
    treated_units = ["treated_0"]

    X = xr.DataArray(
        df[control_units].values,
        dims=["obs_ind", "coeffs"],
        coords={
            "obs_ind": df.index,
            "coeffs": control_units,
        },
    )

    y = xr.DataArray(
        df[treated_units].values,
        dims=["obs_ind", "treated_units"],
        coords={
            "obs_ind": df.index,
            "treated_units": treated_units,
        },
    )

    coords = {
        "coeffs": control_units,
        "treated_units": treated_units,
        "obs_ind": np.arange(n_obs),
    }

    return X, y, coords, control_units, treated_units


class TestWeightedSumFitterMultiUnit:
    """Tests for WeightedSumFitter with multiple treated units."""

    def test_multi_unit_fitting(self, synthetic_control_data):
        """Test that WeightedSumFitter can fit with multiple treated units."""
        X, y, coords, control_units, treated_units = synthetic_control_data

        wsf = WeightedSumFitter(sample_kwargs=sample_kwargs)
        result = wsf.fit(X, y, coords=coords)

        # Check that fitting was successful
        assert isinstance(result, az.InferenceData)
        assert "posterior" in result.groups()
        assert "posterior_predictive" in result.groups()

    def test_multi_unit_predictions(self, synthetic_control_data):
        """Test that predictions work correctly with multiple treated units."""
        X, y, coords, control_units, treated_units = synthetic_control_data

        wsf = WeightedSumFitter(sample_kwargs=sample_kwargs)
        wsf.fit(X, y, coords=coords)

        # Test prediction
        pred = wsf.predict(X)

        # Check prediction structure
        assert isinstance(pred, az.InferenceData)
        assert "posterior_predictive" in pred.groups()

        # Check shapes - should be (chains, draws, obs_ind, treated_units)
        mu_shape = pred["posterior_predictive"]["mu"].shape
        y_hat_shape = pred["posterior_predictive"]["y_hat"].shape

        expected_shape = (
            sample_kwargs["chains"],
            sample_kwargs["draws"],
            len(X),
            len(treated_units),
        )
        assert mu_shape == expected_shape
        assert y_hat_shape == expected_shape

        # Check dimensions
        assert pred["posterior_predictive"]["mu"].dims == (
            "chain",
            "draw",
            "obs_ind",
            "treated_units",
        )
        assert pred["posterior_predictive"]["y_hat"].dims == (
            "chain",
            "draw",
            "obs_ind",
            "treated_units",
        )

    def test_multi_unit_coefficients(self, synthetic_control_data):
        """Test that coefficients are correctly structured for multiple treated units."""
        X, y, coords, control_units, treated_units = synthetic_control_data

        wsf = WeightedSumFitter(sample_kwargs=sample_kwargs)
        wsf.fit(X, y, coords=coords)

        # Extract coefficients
        beta = az.extract(wsf.idata.posterior, var_names="beta")
        sigma = az.extract(wsf.idata.posterior, var_names="sigma")

        # Check beta dimensions: should be (sample, treated_units, coeffs)
        assert "treated_units" in beta.dims
        assert "coeffs" in beta.dims
        assert len(beta.coords["treated_units"]) == len(treated_units)
        assert len(beta.coords["coeffs"]) == len(control_units)

        # Check sigma dimensions: should be (sample, treated_units)
        assert "treated_units" in sigma.dims
        assert len(sigma.coords["treated_units"]) == len(treated_units)

        # Test that coefficients are positive (Dirichlet constraint)
        assert (beta > 0).all()

        # Test that coefficients sum to 1 for each treated unit (Dirichlet constraint)
        beta_sums = beta.sum(dim="coeffs")
        np.testing.assert_allclose(beta_sums, 1.0, rtol=1e-10)

    def test_backward_compatibility_single_unit(self, single_treated_data):
        """Test that single treated unit still works (backward compatibility)."""
        X, y, coords, control_units, treated_units = single_treated_data

        wsf = WeightedSumFitter(sample_kwargs=sample_kwargs)
        result = wsf.fit(X, y, coords=coords)

        # Check that fitting was successful
        assert isinstance(result, az.InferenceData)

        # Test prediction
        pred = wsf.predict(X)

        # Now always has treated_units dimension, even for single unit
        mu_shape = pred["posterior_predictive"]["mu"].shape
        expected_shape = (sample_kwargs["chains"], sample_kwargs["draws"], len(X), 1)
        assert mu_shape == expected_shape

    def test_print_coefficients_multi_unit(self, synthetic_control_data, capsys):
        """Test that print_coefficients works correctly with multiple treated units."""
        X, y, coords, control_units, treated_units = synthetic_control_data

        wsf = WeightedSumFitter(sample_kwargs=sample_kwargs)
        wsf.fit(X, y, coords=coords)

        # Test coefficient printing
        wsf.print_coefficients(control_units, round_to=3)

        captured = capsys.readouterr()
        output = captured.out

        # Check that output contains information for each treated unit
        for unit in treated_units:
            assert f"Treated unit: {unit}" in output

        # Check that output contains each control unit name
        for control in control_units:
            assert control in output

        # Check that sigma is printed for each unit
        assert output.count("sigma") == len(treated_units)

    def test_scoring_multi_unit(self, synthetic_control_data):
        """Test that scoring works with multiple treated units."""
        X, y, coords, control_units, treated_units = synthetic_control_data

        wsf = WeightedSumFitter(sample_kwargs=sample_kwargs)
        wsf.fit(X, y, coords=coords)

        # Test scoring
        score = wsf.score(X, y)

        # Score should be a pandas Series with separate r2 and r2_std for each treated unit
        assert isinstance(score, pd.Series)

        # Check that we have r2 and r2_std for each treated unit
        for unit in treated_units:
            assert f"{unit}_r2" in score.index
            assert f"{unit}_r2_std" in score.index

            # R2 should be reasonable (between 0 and 1 typically, though can be negative)
            assert score[f"{unit}_r2"] >= -1  # R2 can be negative for very bad fits
            assert (
                score[f"{unit}_r2_std"] >= 0
            )  # Standard deviation should be non-negative

    def test_scoring_single_unit(self, single_treated_data):
        """Test that scoring works with single treated unit (backward compatibility)."""
        X, y, coords, control_units, treated_units = single_treated_data

        wsf = WeightedSumFitter(sample_kwargs=sample_kwargs)
        wsf.fit(X, y, coords=coords)

        # Test scoring
        score = wsf.score(X, y)

        # Now consistently uses treated unit name prefix even for single unit
        assert isinstance(score, pd.Series)
        assert "treated_0_r2" in score.index
        assert "treated_0_r2_std" in score.index

        # R2 should be reasonable
        assert score["treated_0_r2"] >= -1  # R2 can be negative for very bad fits
        assert (
            score["treated_0_r2_std"] >= 0
        )  # Standard deviation should be non-negative

    def test_r2_scores_differ_across_units(self, rng):
        """Test that R² scores are different for different treated units.

        This is a defensive test to ensure that each treated unit is being scored
        independently and not getting identical scores due to implementation bugs.
        """
        n_obs = 100  # Use more observations for better differentiation
        n_control = 4

        # Control unit data
        control_data = {}
        for i in range(n_control):
            control_data[f"control_{i}"] = rng.normal(0, 1, n_obs)

        # Create treated units with deliberately different quality of fit
        treated_data = {}

        # Treated unit 0: Good fit (close to control combination)
        weights_0 = rng.dirichlet(np.ones(n_control))
        treated_data["treated_0"] = sum(
            weights_0[i] * control_data[f"control_{i}"] for i in range(n_control)
        ) + rng.normal(0, 0.05, n_obs)  # Low noise

        # Treated unit 1: Medium fit
        weights_1 = rng.dirichlet(np.ones(n_control))
        treated_data["treated_1"] = sum(
            weights_1[i] * control_data[f"control_{i}"] for i in range(n_control)
        ) + rng.normal(0, 0.3, n_obs)  # Medium noise

        # Treated unit 2: Poor fit (mostly random)
        treated_data["treated_2"] = rng.normal(0, 2, n_obs)  # Largely independent

        # Create DataFrame
        df = pd.DataFrame({**control_data, **treated_data})

        # Prepare data for model
        control_units = [f"control_{i}" for i in range(n_control)]
        treated_units = ["treated_0", "treated_1", "treated_2"]

        X = xr.DataArray(
            df[control_units].values,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": df.index,
                "coeffs": control_units,
            },
        )

        y = xr.DataArray(
            df[treated_units].values,
            dims=["obs_ind", "treated_units"],
            coords={
                "obs_ind": df.index,
                "treated_units": treated_units,
            },
        )

        coords = {
            "coeffs": control_units,
            "treated_units": treated_units,
            "obs_ind": np.arange(n_obs),
        }

        # Fit model and score
        wsf = WeightedSumFitter(sample_kwargs=sample_kwargs)
        wsf.fit(X, y, coords=coords)
        scores = wsf.score(X, y)

        # Extract R² values for each treated unit
        r2_values = [scores[f"{unit}_r2"] for unit in treated_units]

        # Test that not all R² values are the same
        # Use a tolerance to avoid issues with floating point precision
        assert not np.allclose(r2_values, r2_values[0], atol=1e-6), (
            f"All R² scores are too similar: {r2_values}. "
            "This suggests the scoring might not be working correctly for individual units."
        )

        # Test that the expected ordering holds (good > medium > poor fit)
        # Note: This might occasionally fail due to randomness, but should generally hold
        # We'll just check that they're not all identical and that we have reasonable variation
        r2_std = np.std(r2_values)
        assert r2_std > 0.01, (
            f"R² standard deviation is too low ({r2_std}), suggesting insufficient variation "
            "between treated units. This might indicate a scoring implementation issue."
        )
