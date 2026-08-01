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

import builtins

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

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
# Test Sampling Defaults
# =============================================================================


@pytest.mark.parametrize(
    ("sample_kwargs", "expected_cores"),
    [
        (None, 1),
        ({}, 1),
        ({"cores": None}, 1),
        ({"cores": 2}, 2),
        ({"cores": 0}, 0),
    ],
)
def test_iv_sampling_defaults_copy_and_preserve_overrides(
    sample_kwargs, expected_cores
):
    """Test IV sampling defaults are safe without changing caller configuration."""
    original = None if sample_kwargs is None else dict(sample_kwargs)

    model = cp.pymc_models.InstrumentalVariableRegression(sample_kwargs=sample_kwargs)

    assert model.sample_kwargs["cores"] == expected_cores
    if sample_kwargs is not None:
        assert sample_kwargs == original
        assert model.sample_kwargs is not sample_kwargs


def test_iv_sampling_default_does_not_change_other_pymc_models():
    """Test the temporary sampling default is limited to IV models."""
    assert "cores" not in cp.pymc_models.LinearRegression().sample_kwargs


def test_iv_default_sampling_kwargs_are_forwarded_without_ppc(monkeypatch):
    """Test default IV sampling uses one core and skips posterior prediction."""
    sampled_kwargs = {}
    idata = az.InferenceData()

    def sample(**kwargs):
        sampled_kwargs.update(kwargs)
        return idata

    def posterior_predictive(*args, **kwargs):
        pytest.fail("ppc_sampler=None must skip posterior predictive sampling")

    monkeypatch.setattr(pm, "sample", sample)
    monkeypatch.setattr(pm, "sample_posterior_predictive", posterior_predictive)
    model = cp.pymc_models.InstrumentalVariableRegression(
        sample_kwargs={"draws": 7, "tune": 3, "progressbar": False}
    )
    X = np.array([[1.0, 0.5], [1.0, 1.5]])
    Z = np.array([[1.0, 0.2], [1.0, 0.8]])
    y = np.array([[1.0], [2.0]])
    t = np.array([[0.5], [1.5]])

    model.fit(
        X=X,
        Z=Z,
        y=y,
        t=t,
        coords={"instruments": ["Intercept", "Z"], "covariates": ["Intercept", "X"]},
        priors={
            "mus": [[0.0, 0.0], [0.0, 0.0]],
            "sigmas": [1.0, 1.0],
            "eta": 2,
            "lkj_sd": 1,
        },
    )

    assert sampled_kwargs["cores"] == 1
    assert sampled_kwargs["draws"] == 7
    assert sampled_kwargs["tune"] == 3
    assert sampled_kwargs["progressbar"] is False
    assert model.idata is idata


def test_iv_default_model_uses_safe_sampling_kwargs(monkeypatch, iv_data):
    """Test the experiment's default IV model receives the safe core default."""
    sampled_kwargs = {}

    def sample(**kwargs):
        sampled_kwargs.update(kwargs)
        return az.InferenceData()

    monkeypatch.setattr(pm, "sample", sample)

    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
    )

    assert result.model.sample_kwargs["cores"] == 1
    assert sampled_kwargs["cores"] == 1


def test_iv_default_sampling_completes_with_two_chains(iv_data):
    """Test default IV sampling completes safely with two sequential chains."""
    model = cp.pymc_models.InstrumentalVariableRegression(
        sample_kwargs={
            "tune": 5,
            "draws": 5,
            "chains": 2,
            "progressbar": False,
            "random_seed": 42,
            "compute_convergence_checks": False,
        }
    )

    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=model,
    )

    assert model.sample_kwargs["cores"] == 1
    assert result.idata.posterior.sizes["chain"] == 2
    assert result.idata.posterior.sizes["draw"] == 5


def _set_jax_import(monkeypatch, jax_module):
    """Patch the direct JAX import used by IV posterior prediction."""
    original_import = builtins.__import__

    def import_jax(name, *args, **kwargs):
        if name == "jax":
            if isinstance(jax_module, BaseException):
                raise jax_module
            return jax_module
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_jax)


class _Idata:
    """Minimal inference-data stand-in for posterior predictive tests."""

    def extend(self, other):
        """Record posterior predictive output."""
        self.extended = other


@pytest.mark.parametrize("use_default", [True, False])
def test_iv_jax_ppc_reports_missing_jax_without_fallback(monkeypatch, use_default):
    """Test requested JAX PPC has a clear missing-dependency error."""
    model = cp.pymc_models.InstrumentalVariableRegression()
    model.idata = object()
    calls = 0

    def posterior_predictive(*args, **kwargs):
        nonlocal calls
        calls += 1

    _set_jax_import(monkeypatch, ModuleNotFoundError("No module named 'jax'"))
    monkeypatch.setattr(pm, "sample_posterior_predictive", posterior_predictive)

    with pytest.raises(ImportError, match="requires JAX"):
        if use_default:
            model.sample_predictive_distribution()
        else:
            model.sample_predictive_distribution(ppc_sampler="jax")

    assert calls == 0


def test_iv_jax_ppc_uses_jax_compilation(monkeypatch):
    """Test available JAX PPC uses the JAX compilation mode."""
    model = cp.pymc_models.InstrumentalVariableRegression(
        sample_kwargs={"random_seed": 42}
    )
    model.idata = _Idata()
    posterior_predictive = object()
    calls = {}
    _set_jax_import(monkeypatch, object())

    def sample(idata, **kwargs):
        calls["idata"] = idata
        calls.update(kwargs)
        return posterior_predictive

    monkeypatch.setattr(pm, "sample_posterior_predictive", sample)

    model.sample_predictive_distribution()

    assert calls == {
        "idata": model.idata,
        "random_seed": 42,
        "compile_kwargs": {"mode": "JAX"},
        "extend_inferencedata": True,
    }


@pytest.mark.parametrize(
    "error",
    [
        ImportError("PPC import failure"),
        RuntimeError("PPC runtime failure"),
        ModuleNotFoundError("PPC module failure"),
    ],
)
@pytest.mark.parametrize("use_default", [True, False])
def test_iv_jax_ppc_preserves_non_dependency_errors(monkeypatch, use_default, error):
    """Test JAX PPC failures are not rewritten as missing-JAX errors."""
    model = cp.pymc_models.InstrumentalVariableRegression()
    model.idata = _Idata()
    _set_jax_import(monkeypatch, object())

    def posterior_predictive(*args, **kwargs):
        raise error

    monkeypatch.setattr(pm, "sample_posterior_predictive", posterior_predictive)

    with pytest.raises(type(error), match=str(error)):
        if use_default:
            model.sample_predictive_distribution()
        else:
            model.sample_predictive_distribution(ppc_sampler="jax")


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


def test_iv_accepts_transformed_treatment_lhs(mock_pymc_sample, sample_kwargs):
    """The parsed first-stage treatment is used throughout IV fitting."""
    Z = np.linspace(0.1, 1, 20)
    X = np.exp(1 + 2 * Z)
    y = 3 * np.log(X)
    df = pd.DataFrame({"y": y, "X": X, "Z": Z})

    with pytest.warns(UserWarning, match="treatment variable is not Binary"):
        result = cp.InstrumentalVariable(
            instruments_data=df[["X", "Z"]],
            data=df[["y", "X"]],
            instruments_formula="np.log(X) ~ 1 + Z",
            formula="y ~ 1 + np.log(X)",
            model=cp.pymc_models.InstrumentalVariableRegression(
                sample_kwargs=sample_kwargs
            ),
        )

    np.testing.assert_allclose(result.t.ravel(), np.log(X))
    assert result.instrument_variable_name == "np.log(X)"
    assert hasattr(result, "second_stage_reg")


def test_iv_rejects_interaction_with_transformed_treatment(sample_kwargs):
    """Unsupported transformed-treatment interactions fail before fitting."""
    Z = np.linspace(0.1, 1, 20)
    X = np.exp(1 + 2 * Z)
    df = pd.DataFrame({"y": 3 * np.log(X), "X": X, "Z": Z})

    with pytest.raises(DataException, match="Interactions with a transformed"):
        cp.InstrumentalVariable(
            instruments_data=df[["X", "Z"]],
            data=df,
            instruments_formula="np.log(X) ~ 1 + Z",
            formula="y ~ 1 + np.log(X)*Z",
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


@pytest.mark.parametrize("method", ["plot", "effect_summary"])
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


def test_iv_has_no_unsupported_get_plot_data(iv_data, sample_kwargs):
    """Instrumental variables expose no generic plot-data method."""
    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )

    with pytest.raises(AttributeError, match="get_plot_data"):
        result.get_plot_data()


# =============================================================================
# Test Predictive Distribution
# =============================================================================


def test_iv_sample_predictive_distribution(iv_data, sample_kwargs):
    """PyMC PPC sampler mutates the fitted DataTree with posterior_predictive."""
    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )

    assert isinstance(result.idata, xr.DataTree)
    assert "posterior_predictive" not in result.idata
    posterior_before_ppc = result.idata["posterior"].to_dataset().copy()

    result.model.sample_predictive_distribution(ppc_sampler="pymc")

    assert isinstance(result.idata, xr.DataTree)
    assert "posterior_predictive" in result.idata
    assert "prior_predictive" in result.idata
    xr.testing.assert_identical(
        result.idata["posterior"].to_dataset(), posterior_before_ppc
    )


def test_iv_default_jax_ppc_mutates_existing_datatree(
    iv_data, sample_kwargs, monkeypatch
):
    """Default JAX PPC sampler mutates an existing DataTree in place.

    JAX itself is optional in the test env; the DataTree mutation contract is
    exercised by stubbing ``pm.sample_posterior_predictive`` while asserting the
    default call uses ``compile_kwargs={"mode": "JAX"}`` and
    ``extend_inferencedata=True``.
    """
    import pymc as pm

    result = cp.InstrumentalVariable(
        instruments_data=iv_data["instruments_data"],
        data=iv_data["data"],
        instruments_formula=iv_data["instruments_formula"],
        formula=iv_data["formula"],
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )

    assert isinstance(result.idata, xr.DataTree)
    assert "posterior_predictive" not in result.idata
    original_idata = result.idata

    def fake_sample_posterior_predictive(idata, *args, **kwargs):
        assert kwargs.get("extend_inferencedata") is True
        assert kwargs.get("compile_kwargs") == {"mode": "JAX"}
        # Mimic in-place mutation from extend_inferencedata=True
        idata["posterior_predictive"] = xr.Dataset(
            {
                "likelihood": (
                    ("chain", "draw", "likelihood_dim_0"),
                    np.zeros((1, 1, 1)),
                )
            }
        )
        return idata

    monkeypatch.setattr(
        pm, "sample_posterior_predictive", fake_sample_posterior_predictive
    )
    _set_jax_import(monkeypatch, object())
    result.model.sample_predictive_distribution()  # default ppc_sampler="jax"

    assert result.idata is original_idata
    assert isinstance(result.idata, xr.DataTree)
    assert "posterior_predictive" in result.idata


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
