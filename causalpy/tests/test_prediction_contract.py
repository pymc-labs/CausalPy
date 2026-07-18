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
"""Contract tests for outcome-scale ``mu`` semantics in causal impact."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from causalpy.pymc_models import LinearRegression, PyMCModel

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


def _single_unit_coords(n_obs: int, n_coeffs: int = 2) -> dict:
    return {
        "obs_ind": np.arange(n_obs),
        "coeffs": [f"x{i}" for i in range(n_coeffs)],
        "treated_units": ["unit_0"],
    }


def _design_matrix(n_obs: int, rng: np.random.Generator) -> xr.DataArray:
    x = rng.normal(size=(n_obs, 2))
    x[:, 0] = 1.0
    return xr.DataArray(
        x,
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": np.arange(n_obs), "coeffs": ["Intercept", "x1"]},
    )


class PoissonLogLinkModel(PyMCModel):
    """Poisson regression with ``mu = exp(eta)`` on the count scale."""

    def build_model(self, X, y, coords):
        with self:
            self.add_coords(coords)
            X_ = pm.Data(name="X", value=X, dims=["obs_ind", "coeffs"])
            y_ = pm.Data(name="y", value=y, dims=["obs_ind", "treated_units"])
            beta = pm.Normal("beta", mu=0, sigma=1, dims=["treated_units", "coeffs"])
            eta = pm.Deterministic(
                "eta",
                pm.math.dot(X_, beta.T),
                dims=["obs_ind", "treated_units"],
            )
            mu = pm.Deterministic(
                "mu", pm.math.exp(eta), dims=["obs_ind", "treated_units"]
            )
            pm.Poisson("y_hat", mu=mu, observed=y_, dims=["obs_ind", "treated_units"])


class BernoulliLogitModel(PyMCModel):
    """Bernoulli regression with ``mu = sigmoid(eta)`` on the probability scale."""

    def build_model(self, X, y, coords):
        with self:
            self.add_coords(coords)
            X_ = pm.Data(name="X", value=X, dims=["obs_ind", "coeffs"])
            y_ = pm.Data(name="y", value=y, dims=["obs_ind", "treated_units"])
            beta = pm.Normal("beta", mu=0, sigma=1, dims=["treated_units", "coeffs"])
            eta = pm.Deterministic(
                "eta",
                pm.math.dot(X_, beta.T),
                dims=["obs_ind", "treated_units"],
            )
            mu = pm.Deterministic(
                "mu",
                pm.math.sigmoid(eta),
                dims=["obs_ind", "treated_units"],
            )
            pm.Bernoulli("y_hat", p=mu, observed=y_, dims=["obs_ind", "treated_units"])


def test_linear_regression_mu_matches_outcome_scale_impact(rng, mock_pymc_sample):
    """Identity-link Gaussian ``mu`` yields outcome-scale impact draws."""
    n_obs = 12
    X = _design_matrix(n_obs, rng)
    beta = np.array([[0.5, 1.2]])
    y = xr.DataArray(
        (X.data @ beta.T + rng.normal(scale=0.5, size=(n_obs, 1))).astype(float),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
    )
    coords = _single_unit_coords(n_obs)

    model = LinearRegression(sample_kwargs={**sample_kwargs, "random_seed": 42})
    model.fit(X, y, coords)
    pred = model.predict(X, coords)

    impact = model.calculate_impact(y, pred)
    mu = pred.posterior_predictive["mu"]
    expected = (y - mu).transpose(..., "obs_ind")

    xr.testing.assert_allclose(impact, expected)
    assert impact.dims[-1] == "obs_ind"


def test_poisson_log_link_mu_is_expected_count(rng, mock_pymc_sample):
    """Compliant Poisson models expose ``exp(eta)`` as ``mu`` for count-scale impact."""
    n_obs = 15
    X = _design_matrix(n_obs, rng)
    rate = np.exp(X.data @ np.array([[0.2, 0.4]]).T)
    counts = rng.poisson(lam=np.clip(rate, 0.1, None)).astype(float)
    y = xr.DataArray(
        counts,
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
    )
    coords = _single_unit_coords(n_obs)

    model = PoissonLogLinkModel(sample_kwargs={**sample_kwargs, "random_seed": 7})
    model.fit(X, y, coords)
    pred = model.predict(X, coords)

    mu = pred.posterior_predictive["mu"]
    assert float(mu.min()) >= 0.0
    impact = model.calculate_impact(y, pred)
    assert impact.dims[-1] == "obs_ind"
    # Impact mean should be on the count scale, not the log scale.
    assert abs(float(impact.mean()) - float((y - mu).mean())) < 1e-6


def test_bernoulli_logit_mu_is_probability(rng, mock_pymc_sample):
    """Compliant Bernoulli models expose ``sigmoid(eta)`` as ``mu``."""
    n_obs = 20
    X = _design_matrix(n_obs, rng)
    logits = X.data @ np.array([[-0.5, 1.0]]).T
    probs = 1.0 / (1.0 + np.exp(-logits))
    outcomes = rng.binomial(1, probs).astype(float)
    y = xr.DataArray(
        outcomes,
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
    )
    coords = _single_unit_coords(n_obs)

    model = BernoulliLogitModel(sample_kwargs={**sample_kwargs, "random_seed": 11})
    model.fit(X, y, coords)
    pred = model.predict(X, coords)

    mu = pred.posterior_predictive["mu"]
    assert float(mu.min()) >= 0.0
    assert float(mu.max()) <= 1.0
    impact = model.calculate_impact(y, pred)
    xr.testing.assert_allclose(impact, (y - mu).transpose(..., "obs_ind"))


def test_link_scale_mu_would_mix_units():
    """Using link-scale ``eta`` as ``mu`` produces contrasts in the wrong units."""
    n_obs = 3
    counts = xr.DataArray(
        np.array([8.0, 10.0, 12.0]),
        dims=["obs_ind"],
        coords={"obs_ind": np.arange(n_obs)},
    )
    eta = xr.DataArray(
        np.log([8.0, 10.0, 12.0]),
        dims=["obs_ind"],
        coords={"obs_ind": np.arange(n_obs)},
    )
    mu = np.exp(eta)

    wrong_impact = counts - eta
    right_impact = counts - mu

    assert not np.allclose(wrong_impact, right_impact)
    # Link-scale subtraction is not centered near zero when counts match exp(eta).
    assert abs(float(wrong_impact.mean())) > 1.0
    assert abs(float(right_impact.mean())) < 1e-10


def test_calculate_impact_uses_mu_not_y_hat(rng, mock_pymc_sample):
    """Impact excludes observation noise by using ``mu`` rather than ``y_hat``."""
    n_obs = 10
    X = _design_matrix(n_obs, rng)
    y = xr.DataArray(
        (
            X.data @ np.array([[0.0, 1.0]]).T + rng.normal(scale=1.0, size=(n_obs, 1))
        ).astype(float),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
    )
    coords = _single_unit_coords(n_obs)

    model = LinearRegression(sample_kwargs={**sample_kwargs, "random_seed": 3})
    model.fit(X, y, coords)
    pred = model.predict(X, coords)

    impact_from_mu = model.calculate_impact(y, pred)
    noise_inclusive = y - pred.posterior_predictive["y_hat"]

    with pytest.raises(AssertionError):
        xr.testing.assert_allclose(impact_from_mu, noise_inclusive)
