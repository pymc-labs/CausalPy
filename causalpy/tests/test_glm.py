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
"""Tests for built-in generalized linear regression and GLM experiment paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from pymc_extras.prior import Prior

import causalpy as cp
from causalpy.data.simulate_data import (
    generate_ancova_data,
    generate_piecewise_its_data,
    generate_staggered_did_data,
)
from causalpy.pymc_models import (
    GeneralizedLinearRegression,
    LinearRegression,
    PyMCModel,
    _validate_family_link,
    model_uses_identity_link,
)
from causalpy.tests.conftest import setup_regression_kink_data

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2, "progressbar": False}


def _poisson_glr(
    random_seed: int | None = None, **kwargs
) -> GeneralizedLinearRegression:
    """Poisson GLR with conservative coefficient priors for fast mocked sampling."""
    sample = {**sample_kwargs, **kwargs.pop("sample_kwargs", {})}
    if random_seed is not None:
        sample["random_seed"] = random_seed
    priors = kwargs.pop(
        "priors",
        {
            "beta": Prior("Normal", mu=0, sigma=0.5, dims=["treated_units", "coeffs"]),
        },
    )
    return GeneralizedLinearRegression(
        family="poisson",
        priors=priors,
        sample_kwargs=sample,
        **kwargs,
    )


def _single_unit_coords(n_obs: int, n_coeffs: int = 2) -> dict:
    coeffs = [f"x{i}" for i in range(n_coeffs)]
    return {
        "obs_ind": np.arange(n_obs),
        "coeffs": coeffs,
        "treated_units": ["unit_0"],
    }


def _design_matrix(
    n_obs: int, rng: np.random.Generator, n_coeffs: int = 2
) -> xr.DataArray:
    x = rng.normal(size=(n_obs, n_coeffs))
    x[:, 0] = 1.0
    coeffs = [f"x{i}" for i in range(n_coeffs)]
    return xr.DataArray(
        x,
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": np.arange(n_obs), "coeffs": coeffs},
    )


@pytest.mark.parametrize(
    ("family", "link"),
    [
        ("gaussian", "identity"),
        ("poisson", "log"),
        ("negative_binomial", "log"),
        ("bernoulli", "logit"),
    ],
)
def test_glr_canonical_family_link_pairs(family, link, rng, mock_pymc_sample):
    """Each curated family/link pair builds ``eta``, ``mu``, and ``y_hat``."""
    n_obs = 12
    X = _design_matrix(n_obs, rng)
    if family == "bernoulli":
        logits = X.data @ np.array([[-0.2, 0.5]]).T
        y_vals = rng.binomial(1, 1 / (1 + np.exp(-logits))).astype(float)
    elif family == "poisson":
        rate = np.exp(X.data @ np.array([[0.1, 0.2]]).T)
        y_vals = rng.poisson(np.clip(rate, 0.1, None)).astype(float)
    elif family == "negative_binomial":
        rate = np.exp(X.data @ np.array([[0.0, 0.3]]).T)
        y_vals = rng.negative_binomial(n=1.0, p=1 / (1 + rate)).astype(float)
    else:
        y_vals = X.data @ np.array([[0.2, 0.4]]).T + rng.normal(
            scale=0.5, size=(n_obs, 1)
        )

    y = xr.DataArray(
        y_vals,
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
    )
    model = GeneralizedLinearRegression(
        family=family,
        link=link,
        sample_kwargs={**sample_kwargs, "random_seed": 1},
    )
    model.fit(X, y, _single_unit_coords(n_obs))
    pred = model.predict(X)
    assert "eta" in model.named_vars
    assert "mu" in model.named_vars
    assert "y_hat" in model.named_vars
    assert "mu" in pred.posterior_predictive
    assert pred.posterior_predictive["mu"].dims[-2:] == ("obs_ind", "treated_units")


@pytest.mark.parametrize(
    ("family", "link"),
    [
        ("poisson", "identity"),
        ("gaussian", "log"),
        ("bernoulli", "log"),
    ],
)
def test_glr_rejects_non_canonical_pairs(family, link):
    """Only canonical family/link pairings are accepted."""
    with pytest.raises(ValueError, match="Invalid family/link pair"):
        GeneralizedLinearRegression(family=family, link=link)


def test_glr_default_link_is_canonical():
    """Omitting ``link`` resolves to the canonical pairing."""
    model = GeneralizedLinearRegression(family="poisson")
    assert model.link == "log"


def test_glr_score_skipped_for_non_gaussian(rng, mock_pymc_sample):
    """Non-Gaussian GLRs return ``None`` from ``score()``."""
    n_obs = 10
    X = _design_matrix(n_obs, rng)
    rate = np.exp(X.data @ np.array([[0.1, 0.2]]).T)
    y = xr.DataArray(
        rng.poisson(np.clip(rate, 0.1, None)).astype(float),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
    )
    model = GeneralizedLinearRegression(
        family="poisson",
        sample_kwargs={**sample_kwargs, "random_seed": 3},
    )
    model.fit(X, y, _single_unit_coords(n_obs))
    assert model.score(X, y) is None


def test_glr_gaussian_score_matches_linear_regression(rng, mock_pymc_sample):
    """Gaussian GLR scoring matches ``LinearRegression``."""
    n_obs = 12
    X = _design_matrix(n_obs, rng)
    y = xr.DataArray(
        (
            X.data @ np.array([[0.2, 0.4]]).T + rng.normal(scale=0.5, size=(n_obs, 1))
        ).astype(float),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
    )
    coords = _single_unit_coords(n_obs)
    kwargs = {**sample_kwargs, "random_seed": 9}
    glr = GeneralizedLinearRegression(family="gaussian", sample_kwargs=kwargs)
    lr = LinearRegression(sample_kwargs=kwargs)
    glr.fit(X, y, coords)
    lr.fit(X, y, coords)
    glr_score = glr.score(X, y)
    lr_score = lr.score(X, y)
    assert glr_score is not None and lr_score is not None
    assert glr_score.equals(lr_score)


def test_glr_clone_preserves_configuration():
    """Cloning preserves family, link, priors, and sampling kwargs."""
    custom_priors = {
        "beta": Prior("Normal", mu=0, sigma=10, dims=["treated_units", "coeffs"])
    }
    original = GeneralizedLinearRegression(
        family="negative_binomial",
        sample_kwargs={"draws": 5, "tune": 5, "chains": 1},
        priors=custom_priors,
    )
    cloned = original._clone()
    assert cloned is not original
    assert cloned.idata is None
    assert cloned.family == "negative_binomial"
    assert cloned.link == "log"
    assert cloned.sample_kwargs == original.sample_kwargs
    assert cloned._user_priors is original._user_priors


def test_linear_regression_clone_stays_gaussian_identity():
    """``LinearRegression._clone()`` keeps the Gaussian identity specialization."""
    original = LinearRegression(sample_kwargs={"draws": 5, "tune": 5, "chains": 1})
    cloned = original._clone()
    assert isinstance(cloned, LinearRegression)
    assert cloned.family == "gaussian"
    assert cloned.link == "identity"
    assert model_uses_identity_link(cloned)


def test_glr_negative_binomial_alpha_prior(rng, mock_pymc_sample):
    """User ``alpha`` priors are wired into the Negative Binomial graph."""
    n_obs = 12
    X = _design_matrix(n_obs, rng)
    rate = np.exp(X.data @ np.array([[0.0, 0.2]]).T)
    y = xr.DataArray(
        rng.negative_binomial(n=1.0, p=1 / (1 + rate)).astype(float),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
    )
    model = GeneralizedLinearRegression(
        family="negative_binomial",
        priors={"alpha": Prior("Exponential", lam=2.0, dims=["treated_units"])},
        sample_kwargs={**sample_kwargs, "random_seed": 5},
    )
    model.fit(X, y, _single_unit_coords(n_obs))
    assert "alpha" in model.idata.posterior


def _poisson_did_data(rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "group": [0, 0, 0, 0, 1, 1, 1, 1],
            "t": [0, 0, 1, 1, 0, 0, 1, 1],
            "unit": np.arange(8),
        }
    )
    df["post_treatment"] = df["t"] == 1
    eta = 1.0 + 1.2 * df["group"].astype(float) * df["post_treatment"].astype(float)
    df["y"] = rng.poisson(np.exp(eta))
    return df


def test_poisson_did_att_uses_g_computation(mock_pymc_sample, rng):
    """Poisson DiD reports a response-scale ATT rather than a link coefficient."""
    df = _poisson_did_data(rng)
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=_poisson_glr(random_seed=11),
    )
    assert isinstance(result.causal_impact, xr.DataArray)
    assert "coeffs" not in result.causal_impact.dims
    assert {"chain", "draw"}.issubset(set(result.causal_impact.dims))


def test_poisson_prepostnegd_att_uses_g_computation(mock_pymc_sample, rng):
    """Poisson PrePostNEGD averages treated-unit response-scale contrasts."""
    df = generate_ancova_data(N=40, seed=7).copy()
    eta = 1.0 + 0.8 * df["group"] + 0.05 * df["pre"]
    df["post"] = rng.poisson(np.exp(eta.astype(float)))
    result = cp.PrePostNEGD(
        df,
        formula="post ~ 1 + C(group) + pre",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=_poisson_glr(random_seed=13),
    )
    assert isinstance(result.causal_impact, xr.DataArray)
    assert "coeffs" not in result.causal_impact.dims


def test_poisson_rd_plot_without_score(mock_pymc_sample, rng, rd_data):
    """Regression discontinuity plots cleanly when ``score()`` is skipped."""
    df = rd_data.copy()
    df["y"] = rng.poisson(np.clip(np.exp(0.2 + df["x"]), 0.1, 15.0))
    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + bs(x, df=6) + treated",
        treatment_threshold=0.5,
        epsilon=0.001,
        model=_poisson_glr(random_seed=17),
    )
    assert result.score is None
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert "Bayesian fit on data" in ax.get_title()


def test_poisson_rk_plot_without_score(mock_pymc_sample, rng):
    """Regression kink plots cleanly when ``score()`` is skipped."""
    kink = 0.5
    df = setup_regression_kink_data(kink)
    df["y"] = rng.poisson(np.clip(np.exp(0.2 + 0.5 * df["x"]), 0.1, 15.0))
    result = cp.RegressionKink(
        df,
        formula=f"y ~ 1 + x + I((x-{kink})*treated)",
        kink_point=kink,
        model=_poisson_glr(random_seed=19),
    )
    assert result.score is None
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert "Bayesian fit on all data" in ax.get_title()


def test_poisson_piecewise_its_smoke(mock_pymc_sample, rng):
    """Piecewise ITS runs with Poisson GLR and response-scale impact."""
    df, _ = generate_piecewise_its_data(
        N=80,
        interruption_times=[40],
        level_changes=[0.5],
        slope_changes=[0.01],
        noise_sigma=0.1,
        seed=3,
    )
    df["t"] = df["t"] / df["t"].max()
    df["y"] = rng.poisson(np.clip(np.exp(-0.5 + 0.2 * df["t"]), 0.1, 10.0))
    result = cp.PiecewiseITS(
        df,
        formula="y ~ 1 + t + step(t, 40) + ramp(t, 40)",
        model=_poisson_glr(
            random_seed=23,
            priors={
                "beta": Prior(
                    "Normal", mu=0, sigma=0.1, dims=["treated_units", "coeffs"]
                ),
            },
        ),
    )
    assert result.score is None
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)


def test_poisson_staggered_did_smoke(mock_pymc_sample, rng):
    """Staggered DiD runs with Poisson GLR on count outcomes."""
    df = generate_staggered_did_data(
        n_units=20,
        n_time_periods=10,
        treatment_cohorts={4: 6, 7: 6},
        seed=4,
    )
    df["y"] = rng.poisson(np.clip(np.exp(0.2 + 0.05 * df["time"]), 0.1, 15.0))
    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        treatment_time_variable_name="treatment_time",
        model=_poisson_glr(random_seed=29),
    )
    assert isinstance(result, cp.StaggeredDifferenceInDifferences)


def _make_poisson_its_data(
    n: int = 120, effect: float = 3.0, seed: int = 21
) -> tuple[pd.DataFrame, int, float]:
    rng = np.random.default_rng(seed)
    treatment_time = 90
    t = np.arange(n) / max(n - 1, 1)
    post = (np.arange(n) >= treatment_time).astype(float)
    lam = np.maximum(1.0 + 0.5 * t + effect * post, 0.1)
    return (
        pd.DataFrame({"t": t, "y": rng.poisson(lam)}),
        treatment_time,
        effect,
    )


def test_poisson_its_recovers_simulated_effect(mock_pymc_sample):
    """Poisson ITS exposes count-scale cumulative impact on the post period."""
    effect = 3.0
    df, treatment_time, _ = _make_poisson_its_data(effect=effect)
    post_mean = df.loc[df.index >= treatment_time, "y"].mean()
    pre_mean = df.loc[df.index < treatment_time, "y"].mean()
    assert post_mean > pre_mean
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t",
        model=_poisson_glr(
            random_seed=31,
            priors={
                "beta": Prior(
                    "Normal", mu=0, sigma=0.1, dims=["treated_units", "coeffs"]
                ),
            },
        ),
    )
    cumulative = result.post_impact_cumulative
    if "treated_units" in cumulative.dims:
        cumulative = cumulative.isel(treated_units=0)
    assert cumulative.dims[-1] == "obs_ind"
    assert np.isfinite(float(cumulative.isel(obs_ind=-1).mean()))


def test_validate_family_link_rejects_unknown_family():
    """Unknown families are rejected at construction time."""
    with pytest.raises(ValueError, match="Unsupported family"):
        _validate_family_link("gamma", None)


def test_validate_family_link_rejects_unknown_link():
    """Unknown links are rejected before family pairing checks."""
    with pytest.raises(ValueError, match="Unsupported link"):
        _validate_family_link("gaussian", "probit")


def test_model_uses_identity_link_defaults_to_true_for_legacy_models():
    """Models without ``uses_identity_link`` are treated as identity-link."""
    assert model_uses_identity_link(PyMCModel()) is True


def test_glr_build_model_adds_default_treated_units_coord(rng, mock_pymc_sample):
    """``build_model`` injects a default ``treated_units`` coordinate when missing."""
    n_obs = 8
    X = _design_matrix(n_obs, rng)
    y = xr.DataArray(
        rng.poisson(2, size=(n_obs, 1)).astype(float),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
    )
    coords = {"obs_ind": np.arange(n_obs), "coeffs": ["x0", "x1"]}
    model = GeneralizedLinearRegression(
        family="poisson",
        sample_kwargs={**sample_kwargs, "random_seed": 2},
    )
    model.fit(X, y, coords)
    assert "treated_units" in model.coords


def test_glr_print_coefficients_requires_fit():
    """Printing coefficients before fitting raises a runtime error."""
    model = GeneralizedLinearRegression(family="poisson")
    with pytest.raises(RuntimeError, match="Model has not been fit"):
        model.print_coefficients(["x0"])


def test_glr_gaussian_print_coefficients_includes_sigma(capsys, rng, mock_pymc_sample):
    """Gaussian GLR summaries include ``y_hat_sigma``."""
    n_obs = 8
    X = _design_matrix(n_obs, rng)
    y = xr.DataArray(
        (
            X.data @ np.array([[0.1, 0.2]]).T + rng.normal(scale=0.1, size=(n_obs, 1))
        ).astype(float),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
    )
    model = GeneralizedLinearRegression(
        family="gaussian",
        sample_kwargs={**sample_kwargs, "random_seed": 4},
    )
    model.fit(X, y, _single_unit_coords(n_obs))
    model.print_coefficients(["x0", "x1"])
    assert "y_hat_sigma" in capsys.readouterr().out


def test_glr_negative_binomial_print_coefficients_includes_alpha(
    capsys, rng, mock_pymc_sample
):
    """Negative Binomial summaries include ``alpha`` when present."""
    n_obs = 10
    X = _design_matrix(n_obs, rng)
    rate = np.exp(X.data @ np.array([[0.0, 0.1]]).T)
    y = xr.DataArray(
        rng.negative_binomial(n=1.0, p=1 / (1 + rate)).astype(float),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
    )
    model = GeneralizedLinearRegression(
        family="negative_binomial",
        sample_kwargs={**sample_kwargs, "random_seed": 6},
    )
    model.fit(X, y, _single_unit_coords(n_obs))
    model.print_coefficients(["x0", "x1"])
    assert "alpha" in capsys.readouterr().out


def test_glr_print_coefficients_without_sigma(capsys, rng, mock_pymc_sample):
    """Poisson summaries omit Gaussian ``sigma`` rows."""
    n_obs = 8
    X = _design_matrix(n_obs, rng)
    rate = np.exp(X.data @ np.array([[0.1, 0.2]]).T)
    y = xr.DataArray(
        rng.poisson(np.clip(rate, 0.1, None)).astype(float),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
    )
    model = GeneralizedLinearRegression(
        family="poisson",
        sample_kwargs={**sample_kwargs, "random_seed": 37},
    )
    model.fit(X, y, _single_unit_coords(n_obs))
    model.print_coefficients(["x0", "x1"])
    captured = capsys.readouterr().out
    assert "y_hat_sigma" not in captured
