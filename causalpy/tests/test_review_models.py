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

from causalpy.experiments.model_adapter import PyMCModelAdapter
from causalpy.pymc_models import (
    InstrumentalVariableRegression,
    LinearRegression,
    PropensityScore,
)

pytestmark = pytest.mark.usefixtures("real_pymc_sampling")


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


def test_iv_predictive_optout_drops_previous_posterior_predictions(
    iv_inputs, small_sample_kwargs
):
    """An explicit predictive opt-out cannot expose draws from an earlier fit."""
    model = InstrumentalVariableRegression(sample_kwargs=small_sample_kwargs)
    model.fit(**iv_inputs, ppc_sampler="pymc")
    assert model.idata["posterior_predictive"].sizes["draw"] == 2
    prior = model.idata["prior"].to_dataset().copy(deep=True)
    prior_predictive = model.idata["prior_predictive"].to_dataset().copy(deep=True)
    previous_posterior = model.idata["posterior"].to_dataset().copy(deep=True)
    model.sample_kwargs.update(draws=3, random_seed=19)
    model.fit(**iv_inputs, ppc_sampler=None)
    assert model.idata["posterior"].sizes["draw"] == 3
    assert not model.idata["posterior"].to_dataset().equals(previous_posterior)
    assert "posterior_predictive" not in model.idata.children
    xr.testing.assert_identical(model.idata["prior"].to_dataset(), prior)
    xr.testing.assert_identical(
        model.idata["prior_predictive"].to_dataset(), prior_predictive
    )


def test_base_mapping_build_rejects_unsupported_model():
    """Lazy adapter builds report unsupported mappings, not a missing attribute."""
    data = {"unit": xr.DataArray([1.0], dims=["obs_ind"])}
    with pytest.raises(TypeError, match="does not support mapping-valued inputs"):
        PyMCModelAdapter(LinearRegression()).build(X=data, y=data)


def test_custom_mapping_build_keeps_sampling_lazy(regression_inputs):
    """Custom models can opt into the build hook without invoking eager fit."""

    class MappingRegression(LinearRegression):
        def build_mapping(self, X, y, coords=None):
            self.build(X["unit"], y["unit"], coords=coords)

        def fit_mapping(self, X, y, coords=None):
            pytest.fail("A lazy build must not invoke the eager fit hook")

    X, y = regression_inputs
    model = MappingRegression(prior_sample_kwargs={"draws": 3, "random_seed": 7})
    adapter = PyMCModelAdapter(model)
    adapter.build(X={"unit": X}, y={"unit": y})
    assert model.idata is None
    adapter.sample_prior_predictive()
    assert model.require_group("prior").sizes["draw"] == 3
    np.testing.assert_allclose(
        model.require_group("prior").mu,
        np.einsum("ij,cdkj->cdik", X.values, model.require_group("prior").beta.values),
    )


def test_sampler_overrides_rearm_training_data_and_isolate_draw_groups(
    regression_inputs, small_sample_kwargs
):
    """Real draws honor per-call sizes and never mix prior and posterior values."""
    X, y = regression_inputs
    model = LinearRegression(
        sample_kwargs=small_sample_kwargs,
        prior_sample_kwargs={"draws": 5, "random_seed": 23},
    )
    model.build(X, y)
    model.sample_posterior()
    model.sample_prior_predictive()
    assert model.require_group("posterior").sizes["draw"] == 2
    assert model.require_group("prior").sizes["draw"] == 5
    posterior = model.require_group("posterior").copy(deep=True)
    posterior_predictive = (
        model.idata["posterior_predictive"].to_dataset().copy(deep=True)
    )

    # Prediction leaves the graph armed with a different design and outcome
    # placeholder. Prior resampling must restore both original training nodes.
    model.predict(X.isel(obs_ind=slice(0, 2)) + 100, group="prior")
    model.sample_prior_predictive(draws=7, random_seed=29)
    prior = model.require_group("prior").copy(deep=True)
    prior_predictive = model.idata["prior_predictive"].to_dataset().copy(deep=True)
    assert prior.sizes["draw"] == 7
    assert prior_predictive.sizes["draw"] == 7
    assert prior.sizes["obs_ind"] == X.sizes["obs_ind"]
    np.testing.assert_allclose(
        prior.mu, np.einsum("ij,cdkj->cdik", X.values, prior.beta.values)
    )
    np.testing.assert_array_equal(model["y"].get_value(), y.values)
    xr.testing.assert_identical(model.require_group("posterior"), posterior)
    xr.testing.assert_identical(
        model.idata["posterior_predictive"].to_dataset(), posterior_predictive
    )

    # Posterior overrides win over constructor settings for this call only,
    # while the independent prior phase remains byte-for-byte unchanged.
    model.sample_posterior(draws=3, random_seed=31)
    posterior = model.require_group("posterior")
    assert posterior.sizes["draw"] == 3
    assert model.idata["posterior_predictive"].sizes["draw"] == 3
    np.testing.assert_allclose(
        posterior.mu, np.einsum("ij,cdkj->cdik", X.values, posterior.beta.values)
    )
    xr.testing.assert_identical(model.require_group("prior"), prior)
    xr.testing.assert_identical(
        model.idata["prior_predictive"].to_dataset(), prior_predictive
    )
    assert model.sample_kwargs["draws"] == 2
    assert model.prior_sample_kwargs == {"draws": 5, "random_seed": 23}
