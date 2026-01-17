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
Tests for the InstrumentalVariable experiment class.
"""

import numpy as np
import pandas as pd
import pytest

import causalpy as cp
from causalpy.custom_exceptions import DataException

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_kwargs():
    """Minimal sampling kwargs for fast tests."""
    return {
        "tune": 5,
        "draws": 5,
        "chains": 1,
        "progressbar": False,
        "random_seed": 42,
    }


@pytest.fixture
def iv_data(rng):
    """Generate synthetic IV data with known properties."""
    N = 100
    # Endogeneity: e1 affects both X and y
    e1 = rng.normal(0, 3, N)
    e2 = rng.normal(0, 1, N)
    # Valid instrument Z (affects X but not y directly)
    Z = rng.uniform(0, 1, N)
    # Treatment X is endogenous
    X = -1 + 4 * Z + e2 + 2 * e1
    # Outcome y
    y = 2 + 3 * X + 3 * e1

    df = pd.DataFrame({"y": y, "X": X, "Z": Z})
    return {
        "data": df[["y", "X"]],
        "instruments_data": df[["X", "Z"]],
        "formula": "y ~ 1 + X",
        "instruments_formula": "X ~ 1 + Z",
    }


@pytest.fixture
def binary_treatment_data(rng):
    """Generate synthetic IV data with binary treatment."""
    N = 100
    Z1 = rng.normal(0, 1, N)
    Z2 = rng.normal(0, 1, N)
    # Binary treatment influenced by instruments
    prob = 1 / (1 + np.exp(-(0.5 * Z1 + 0.3 * Z2)))
    T = (rng.uniform(0, 1, N) < prob).astype(int)
    # Outcome
    y = 1 + 2 * T + 0.5 * Z1 + rng.normal(0, 1, N)

    df = pd.DataFrame({"y": y, "T": T, "Z1": Z1, "Z2": Z2})
    return {
        "data": df[["y", "T", "Z1"]],
        "instruments_data": df[["T", "Z1", "Z2"]],
        "formula": "y ~ 1 + T + Z1",
        "instruments_formula": "T ~ 1 + Z1 + Z2",
    }


# =============================================================================
# Test Initialization and Design Matrices
# =============================================================================


def test_iv_initialization(iv_data, sample_kwargs):
    """Test that InstrumentalVariable initializes correctly."""
    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )

    assert isinstance(result, cp.InstrumentalVariable)
    assert result.expt_type == "Instrumental Variable Regression"
    assert result.formula == iv_data["formula"]
    assert result.instruments_formula == iv_data["instruments_formula"]


def test_iv_design_matrices_shape(iv_data, sample_kwargs):
    """Test that design matrices have correct shapes."""
    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )

    # Check shapes
    n_obs = len(iv_data["data"])
    assert result.X.shape[0] == n_obs
    assert result.y.shape[0] == n_obs
    assert result.Z.shape[0] == n_obs
    assert result.t.shape[0] == n_obs

    # Check design matrix columns (intercept + covariates)
    assert result.X.shape[1] == 2  # Intercept + X
    assert result.Z.shape[1] == 2  # Intercept + Z


def test_iv_labels_extracted(iv_data, sample_kwargs):
    """Test that labels are correctly extracted from design matrices."""
    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )

    assert "Intercept" in result.labels
    assert "X" in result.labels
    assert "Intercept" in result.labels_instruments
    assert "Z" in result.labels_instruments
    assert result.outcome_variable_name == "y"
    assert result.instrument_variable_name == "X"


# =============================================================================
# Test OLS and 2SLS Methods
# =============================================================================


def test_naive_ols_fit(iv_data, sample_kwargs):
    """Test that naive OLS fit is computed."""
    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )

    # Check OLS attributes exist
    assert hasattr(result, "ols_reg")
    assert hasattr(result, "ols_beta_params")
    assert isinstance(result.ols_beta_params, dict)
    assert "Intercept" in result.ols_beta_params
    assert "X" in result.ols_beta_params


def test_2sls_fit(iv_data, sample_kwargs):
    """Test that 2SLS fit is computed."""
    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )

    # Check 2SLS attributes exist
    assert hasattr(result, "first_stage_reg")
    assert hasattr(result, "second_stage_reg")
    assert hasattr(result, "ols_beta_first_params")
    assert hasattr(result, "ols_beta_second_params")
    assert isinstance(result.ols_beta_first_params, list)
    assert isinstance(result.ols_beta_second_params, list)


# =============================================================================
# Test Input Validation
# =============================================================================


def test_iv_missing_treatment_in_instruments_data(sample_kwargs):
    """Test error when treatment variable missing from instruments_data."""
    data = pd.DataFrame({"y": [1, 2, 3], "X": [1, 2, 3]})
    instruments_data = pd.DataFrame({"Z": [1, 2, 3], "W": [4, 5, 6]})  # Missing X

    with pytest.raises(DataException):
        cp.InstrumentalVariable(
            instruments_data=instruments_data,
            data=data,
            instruments_formula="X ~ 1 + Z",
            formula="y ~ 1 + X",
            model=cp.pymc_models.InstrumentalVariableRegression(
                sample_kwargs=sample_kwargs
            ),
        )


def test_iv_missing_treatment_in_data(sample_kwargs):
    """Test error when treatment variable missing from data."""
    data = pd.DataFrame({"y": [1, 2, 3], "W": [1, 2, 3]})  # Missing X
    instruments_data = pd.DataFrame({"X": [1, 2, 3], "Z": [4, 5, 6]})

    with pytest.raises(DataException):
        cp.InstrumentalVariable(
            instruments_data=instruments_data,
            data=data,
            instruments_formula="X ~ 1 + Z",
            formula="y ~ 1 + X",
            model=cp.pymc_models.InstrumentalVariableRegression(
                sample_kwargs=sample_kwargs
            ),
        )


def test_iv_continuous_treatment_warning(iv_data, sample_kwargs):
    """Test that continuous treatment triggers a warning."""
    with pytest.warns(UserWarning, match="treatment variable is not Binary"):
        cp.InstrumentalVariable(
            instruments_data=iv_data["instruments_data"],
            data=iv_data["data"],
            instruments_formula=iv_data["instruments_formula"],
            formula=iv_data["formula"],
            model=cp.pymc_models.InstrumentalVariableRegression(
                sample_kwargs=sample_kwargs
            ),
        )


# =============================================================================
# Test Binary Treatment
# =============================================================================


def test_iv_binary_treatment_priors(binary_treatment_data, sample_kwargs):
    """Test that binary treatment uses different default priors."""
    result = cp.InstrumentalVariable(
        instruments_data=binary_treatment_data["instruments_data"],
        data=binary_treatment_data["data"],
        instruments_formula=binary_treatment_data["instruments_formula"],
        formula=binary_treatment_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
        binary_treatment=True,
    )

    # Binary treatment priors should have rho_bounds instead of eta/lkj_sd
    assert "rho_bounds" in result.priors
    assert "sigma_U" in result.priors
    assert "eta" not in result.priors


def test_iv_continuous_treatment_priors(iv_data, sample_kwargs):
    """Test that continuous treatment uses LKJ priors."""
    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )

    # Continuous treatment priors should have eta/lkj_sd
    assert "eta" in result.priors
    assert "lkj_sd" in result.priors
    assert "rho_bounds" not in result.priors


# =============================================================================
# Test Custom Priors
# =============================================================================


def test_iv_custom_priors(iv_data, sample_kwargs):
    """Test that custom priors are used when provided."""
    custom_priors = {
        "mus": [[0, 0], [0, 0]],
        "sigmas": [2, 2],
        "eta": 5,
        "lkj_sd": 2,
    }

    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
        priors=custom_priors,
    )

    # Custom priors should be stored
    assert result.priors == custom_priors


# =============================================================================
# Test Variable Selection Priors
# =============================================================================


@pytest.mark.parametrize(
    "vs_prior_type,expected_var",
    [
        ("spike_and_slab", "gamma_beta_t"),
        ("horseshoe", "tau_beta_t"),
    ],
)
def test_iv_variable_selection_priors(
    iv_data, sample_kwargs, vs_prior_type, expected_var
):
    """Test that variable selection priors create expected model variables."""
    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
        vs_prior_type=vs_prior_type,
        vs_hyperparams={"outcome": True},
    )

    assert vs_prior_type == result.vs_prior_type
    assert expected_var in result.model.named_vars


# =============================================================================
# Test Inference Data
# =============================================================================


def test_iv_idata_structure(iv_data, sample_kwargs):
    """Test that inference data has expected structure."""
    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )

    # Check idata exists and has posterior
    assert hasattr(result, "idata")
    assert hasattr(result.idata, "posterior")
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]


def test_iv_coords_set(iv_data, sample_kwargs):
    """Test that coords are correctly set."""
    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )

    assert "instruments" in result.coords
    assert "covariates" in result.coords
    assert result.coords["instruments"] == result.labels_instruments
    assert result.coords["covariates"] == result.labels


# =============================================================================
# Test Not Implemented Methods
# =============================================================================


@pytest.mark.parametrize("method", ["plot", "summary", "effect_summary"])
def test_iv_not_implemented_methods(iv_data, sample_kwargs, method):
    """Test that unimplemented methods raise NotImplementedError."""
    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )

    with pytest.raises(NotImplementedError):
        getattr(result, method)()


def test_iv_get_plot_data_not_implemented(iv_data, sample_kwargs):
    """Test that get_plot_data raises NotImplementedError."""
    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )

    with pytest.raises(NotImplementedError):
        result.get_plot_data()


# =============================================================================
# Test Predictive Distribution
# =============================================================================


def test_iv_sample_predictive_distribution(iv_data, sample_kwargs):
    """Test that predictive distribution can be sampled."""
    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )

    result.model.sample_predictive_distribution(ppc_sampler="pymc")
    assert hasattr(result.idata, "posterior_predictive")


# =============================================================================
# Test with Real Data
# =============================================================================


def test_iv_with_risk_data(sample_kwargs):
    """Integration test using the risk dataset."""
    df = cp.load_data("risk")
    instruments_formula = "risk ~ 1 + logmort0"
    formula = "loggdp ~ 1 + risk"
    instruments_data = df[["risk", "logmort0"]]
    data = df[["loggdp", "risk"]]

    result = cp.InstrumentalVariable(
        instruments_data=instruments_data,
        data=data,
        instruments_formula=instruments_formula,
        formula=formula,
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )

    assert isinstance(result, cp.InstrumentalVariable)
    assert result.outcome_variable_name == "loggdp"
    assert result.instrument_variable_name == "risk"
