#   Copyright 2026 - 2026 The PyMC Labs Developers
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
"""Regression coverage for model lifecycle review findings."""

import warnings

import numpy as np
import pytest
import xarray as xr

from causalpy.pymc_models import (
    InstrumentalVariableRegression,
    LinearRegression,
    PropensityScore,
)


def test_iv_clone_preserves_configuration_without_sharing_sampler_settings():
    """Sensitivity checks can clone IV models and tune the clone independently."""
    model = InstrumentalVariableRegression(
        sample_kwargs={"draws": 3, "cores": 1},
        prior_sample_kwargs={"draws": 7, "random_seed": 42},
        priors={"custom": 1},
    )
    cloned = model._clone()
    assert isinstance(cloned, InstrumentalVariableRegression)
    assert cloned.sample_kwargs == model.sample_kwargs
    assert cloned.prior_sample_kwargs == model.prior_sample_kwargs
    assert cloned.priors == model.priors
    assert cloned.idata is None
    cloned.sample_kwargs["draws"] = 11
    cloned.prior_sample_kwargs["draws"] = 13
    assert model.sample_kwargs["draws"] == 3
    assert model.prior_sample_kwargs["draws"] == 7
    assert model._clone(priors={"replacement": 2}).priors == {"replacement": 2}


@pytest.mark.parametrize("change", ["covariates", "treatment", "rows"])
def test_propensity_rebuild_rejects_changed_design(change):
    """A reused propensity graph cannot silently weight a different dataset."""
    X = np.arange(8, dtype=float).reshape(8, 1)
    treatment = np.tile([0, 1], 4)
    coords = {"obs_ind": np.arange(8), "coeffs": ["x"]}
    model = PropensityScore()
    model.build(X, treatment, coords=coords)
    model.build(X.copy(), treatment.copy(), coords=coords)
    if change == "covariates":
        X = X + 1
    elif change == "treatment":
        treatment = 1 - treatment
    else:
        X, treatment = X[:-1], treatment[:-1]
    with pytest.raises(RuntimeError, match="already built with different inputs"):
        model.build(X, treatment, coords=coords)
    np.testing.assert_array_equal(model["X"].get_value(), np.arange(8).reshape(8, 1))
    np.testing.assert_array_equal(model["t"].get_value(), np.tile([0, 1], 4))


@pytest.mark.parametrize("change", ["time", "labels", "dimension"])
def test_rebuild_rejects_changed_xarray_coordinates(change):
    """Equal numeric designs with different labeled axes are not interchangeable."""
    X = xr.DataArray(
        np.ones((4, 1)),
        dims=["obs_ind", "coeffs"],
        coords={
            "obs_ind": np.arange("2020-01-01", "2020-01-05", dtype="datetime64[D]"),
            "coeffs": np.array(["intercept"], dtype=object),
        },
    )
    y = xr.DataArray(np.ones((4, 1)), dims=["obs_ind", "treated_units"])
    model = LinearRegression()
    model.build(X, y)
    model.build(X.copy(deep=True), y.copy(deep=True))
    if change == "time":
        X = X.assign_coords(obs_ind=X.obs_ind + np.timedelta64(365, "D"))
    elif change == "labels":
        X = X.assign_coords(coeffs=np.array(["slope"], dtype=object))
    else:
        X = X.rename(coeffs="features")
    with pytest.raises(RuntimeError, match="already built with different inputs"):
        model.build(X, y)


@pytest.fixture
def regression_inputs():
    X = xr.DataArray(
        np.arange(6, dtype=float).reshape(6, 1), dims=["obs_ind", "coeffs"]
    )
    y = xr.DataArray(
        np.array([0.1, 1.2, 1.9, 3.1, 3.8, 5.2]).reshape(6, 1),
        dims=["obs_ind", "treated_units"],
    )
    return X, y


@pytest.fixture
def iv_inputs():
    treatment = np.array([0.2, 0.8, 1.3, 1.7, 2.1, 2.8])
    return {
        "X": np.column_stack([np.ones(6), treatment]),
        "Z": np.column_stack([np.ones(6), np.arange(6)]),
        "y": np.array([0.5, 1.0, 1.8, 1.9, 2.5, 3.0]),
        "t": treatment,
        "coords": {
            "instruments": ["Intercept", "Z"],
            "covariates": ["Intercept", "t"],
        },
        "priors": {
            "mus": [[0.0, 0.0], [0.0, 0.0]],
            "sigmas": [1.0, 1.0],
            "eta": 2,
            "lkj_sd": 1,
        },
    }


@pytest.fixture
def small_sample_kwargs():
    return {
        "draws": 2,
        "tune": 5,
        "chains": 1,
        "cores": 1,
        "random_seed": 12,
        "progressbar": False,
        "compute_convergence_checks": False,
    }


@pytest.mark.parametrize("backend", ["regression", "iv"])
def test_refit_replaces_predictive_draws_without_raw_overwrite_warning(
    backend, regression_inputs, iv_inputs, small_sample_kwargs
):
    """Posterior refits refresh predictive draws without leaking PyMC merge warnings."""
    if backend == "iv":
        model = InstrumentalVariableRegression(sample_kwargs=small_sample_kwargs)
        model.build(**iv_inputs, ppc_sampler="pymc")
    else:
        model = LinearRegression(sample_kwargs=small_sample_kwargs)
        model.build(*regression_inputs)
    model.sample_posterior()
    before = model.idata["posterior_predictive"].to_dataset().copy(deep=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.sample_posterior(draws=3, random_seed=19)
    assert model.idata["posterior_predictive"].sizes["draw"] == 3
    assert not model.idata["posterior_predictive"].to_dataset().equals(before)
    assert not any("extend_inferencedata" in str(item.message) for item in caught)
