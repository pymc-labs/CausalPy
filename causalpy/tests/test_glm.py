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

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from patsy import build_design_matrices, dmatrix
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
    _create_family_likelihood,
    _validate_family,
)
from causalpy.tests.conftest import setup_regression_kink_data

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2, "progressbar": False}


class FixedMuRegression(PyMCModel):
    """Deterministic response-scale ``mu`` for g-computation estimand tests."""

    def __init__(
        self,
        column_weights: dict[int, float],
        *,
        link: str = "log",
    ) -> None:
        super().__init__()
        self.column_weights = column_weights
        self.link = link
        self.predict_calls: list[dict[str, object]] = []

    def build_model(
        self, X: xr.DataArray, y: xr.DataArray, coords: dict | None
    ) -> None:
        with self:
            pass

    def fit(
        self, X: xr.DataArray, y: xr.DataArray, coords: dict | None = None
    ) -> az.InferenceData:
        self.idata = az.from_dict(posterior={"beta": np.array([[[1.0]]])})
        return self.idata

    def predict(
        self,
        X: xr.DataArray,
        coords: dict | None = None,
        out_of_sample: bool | None = False,
        var_names: list[str] | tuple[str, ...] | None = None,
        **kwargs,
    ):
        if self.idata is None:
            raise RuntimeError("Model has not been fit")
        x_arr = np.asarray(X)
        self.predict_calls.append(
            {"n_obs": x_arr.shape[0], "var_names": list(var_names or [])}
        )
        eta = np.zeros(x_arr.shape[0], dtype=float)
        for col_idx, weight in self.column_weights.items():
            eta += x_arr[:, col_idx] * weight
        if self.link == "log":
            mu = np.exp(eta)
        elif self.link == "identity":
            mu = eta
        else:
            raise ValueError(f"Unsupported link {self.link!r}")
        mu = mu[:, np.newaxis]
        mu_da = xr.DataArray(
            mu[np.newaxis, np.newaxis, :, :],
            dims=["chain", "draw", "obs_ind", "treated_units"],
            coords={
                "chain": [0],
                "draw": [0],
                "obs_ind": np.arange(mu.shape[0]),
                "treated_units": ["unit_0"],
            },
        )
        return az.InferenceData(posterior_predictive=xr.Dataset({"mu": mu_da}))


def _expected_log_link_att(
    factual: np.ndarray, counter: np.ndarray, column_weights: dict[int, float]
) -> float:
    eta_f = np.zeros(factual.shape[0], dtype=float)
    eta_c = np.zeros(counter.shape[0], dtype=float)
    for col_idx, weight in column_weights.items():
        eta_f += factual[:, col_idx] * weight
        eta_c += counter[:, col_idx] * weight
    return float(np.mean(np.exp(eta_f) - np.exp(eta_c)))


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
    "family",
    ["gaussian", "poisson", "negative_binomial", "bernoulli"],
)
def test_glr_canonical_family_link_pairs(family, rng, mock_pymc_sample):
    """Each curated family builds ``eta``, ``mu``, and ``y_hat`` with its canonical link."""
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
        sample_kwargs={**sample_kwargs, "random_seed": 1},
    )
    model.fit(X, y, _single_unit_coords(n_obs))
    assert (
        model.link
        == {
            "gaussian": "identity",
            "poisson": "log",
            "negative_binomial": "log",
            "bernoulli": "logit",
        }[family]
    )
    pred = model.predict(X)
    assert "eta" in model.named_vars
    assert "mu" in model.named_vars
    assert "y_hat" in model.named_vars
    assert "mu" in pred.posterior_predictive
    assert pred.posterior_predictive["mu"].dims[-2:] == ("obs_ind", "treated_units")


def test_glr_rejects_unknown_family():
    """Unknown families are rejected at construction time."""
    with pytest.raises(ValueError, match="Unsupported family"):
        GeneralizedLinearRegression(family="gamma")  # type: ignore[arg-type]


def test_glr_rejects_y_hat_prior_for_non_gaussian():
    """Non-Gaussian GLRs reject custom observation priors."""
    with pytest.raises(ValueError, match="y_hat"):
        GeneralizedLinearRegression(
            family="poisson",
            priors={"y_hat": Prior("Normal", mu=0, sigma=1)},
        )


def test_glr_gaussian_allows_y_hat_prior_override():
    """Gaussian GLRs keep ``y_hat`` customization for ``LinearRegression`` parity."""
    model = GeneralizedLinearRegression(
        family="gaussian",
        priors={"y_hat": Prior("Normal", sigma=2.0, dims=["obs_ind", "treated_units"])},
    )
    assert "y_hat" in model.priors


class _CustomPoissonGLR(GeneralizedLinearRegression):
    default_priors = {
        "beta": Prior("Normal", mu=1.0, sigma=0.25, dims=["treated_units", "coeffs"]),
    }


class _CustomPoissonGLRMixin:
    default_priors = {
        "beta": Prior("Normal", mu=1.0, sigma=0.25, dims=["treated_units", "coeffs"]),
    }


class _MergedPoissonGLR(_CustomPoissonGLRMixin, GeneralizedLinearRegression):
    """Mixin supplies class defaults; property merge is accessed explicitly."""


def test_glr_subclass_default_priors_override():
    """Subclass ``default_priors`` merge into family defaults."""
    model = _CustomPoissonGLR(family="poisson")
    assert model.default_priors["beta"].parameters["mu"] == 1.0
    assert model.default_priors["beta"].parameters["sigma"] == 0.25


def test_glr_default_priors_property_merges_mro_class_defaults():
    """The GLR ``default_priors`` property merges non-empty MRO class dicts."""
    model = _MergedPoissonGLR(family="poisson")
    priors = GeneralizedLinearRegression.default_priors.__get__(
        model, _MergedPoissonGLR
    )
    assert priors["beta"].parameters["mu"] == 1.0
    assert "y_hat" in priors


def test_linear_regression_default_priors_merge_class_attribute():
    """``LinearRegression`` class-level defaults merge through the GLR property."""
    model = LinearRegression()
    priors = GeneralizedLinearRegression.default_priors.__get__(model, LinearRegression)
    assert "beta" in priors
    assert "y_hat" in priors


def test_glr_predict_mu_only(rng, mock_pymc_sample):
    """``predict(..., var_names=['mu'])`` returns only requested variables."""
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
    pred = model.predict(X, var_names=["mu"])
    assert list(pred.posterior_predictive.data_vars) == ["mu"]


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


def test_poisson_did_att_exact_g_computation():
    """DiD ATT matches a hand-computed response-scale contrast."""
    df = pd.DataFrame(
        {
            "group": [0, 0, 0, 0, 1, 1, 1, 1],
            "t": [0, 0, 1, 1, 0, 0, 1, 1],
            "unit": np.arange(8),
            "post_treatment": [False, False, True, True, False, False, True, True],
            "g_post": np.arange(8, dtype=float),
            "y": np.ones(8),
        }
    )
    design = dmatrix(
        "1 + post_treatment*group + g_post",
        df,
        return_type="dataframe",
    )
    labels = list(design.design_info.column_names)
    weights = {
        labels.index("post_treatment[T.True]:group"): np.log(2.0),
        labels.index("g_post"): 0.4,
    }
    model = FixedMuRegression(weights, link="log")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + post_treatment*group + g_post",
        time_variable_name="t",
        group_variable_name="group",
        model=model,
    )
    treated_post = df.query("group == 1 and post_treatment")
    (x_factual,) = build_design_matrices(
        [result._x_design_info],
        treated_post.drop(columns=["y"]),
        return_type="dataframe",
    )
    x_counter = x_factual.copy()
    for i, label in enumerate(result.labels):
        if result._is_treatment_interaction(label):
            x_counter.iloc[:, i] = 0
    expected = _expected_log_link_att(
        np.asarray(x_factual), np.asarray(x_counter), weights
    )
    assert float(result.causal_impact.mean()) == pytest.approx(expected)
    assert "coeffs" not in result.causal_impact.dims


def test_poisson_prepostnegd_att_exact_g_computation():
    """PrePostNEGD uses two bulk ``mu`` predictions and an interaction-aware ATT."""
    df = pd.DataFrame(
        {
            "group": [1, 1, 1, 0, 0, 0],
            "pre": [0.0, 1.0, 2.0, 0.5, 1.5, 2.5],
            "post": np.ones(6),
        }
    )
    design = dmatrix(
        "1 + C(group) + pre + C(group):pre",
        df,
        return_type="dataframe",
    )
    labels = list(design.design_info.column_names)
    weights = {
        labels.index("C(group)[T.1]"): 0.2,
        labels.index("C(group)[T.1]:pre"): 0.5,
        labels.index("pre"): 0.1,
    }
    model = FixedMuRegression(weights, link="log")
    result = cp.PrePostNEGD(
        df,
        formula="post ~ 1 + C(group) + pre + C(group):pre",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=model,
    )
    treated = df[df["group"] == 1]
    x_treated_df = treated.assign(group=1)
    x_control_df = treated.assign(group=0)
    (x_treated,) = build_design_matrices(
        [result._x_design_info], x_treated_df, return_type="dataframe"
    )
    (x_control,) = build_design_matrices(
        [result._x_design_info], x_control_df, return_type="dataframe"
    )
    expected = _expected_log_link_att(
        np.asarray(x_treated), np.asarray(x_control), weights
    )
    assert float(result.causal_impact.mean()) == pytest.approx(expected)
    mu_calls = [call for call in model.predict_calls if call["var_names"] == ["mu"]]
    assert len(mu_calls) == 2
    assert mu_calls[0]["n_obs"] == len(treated)
    assert mu_calls[1]["n_obs"] == len(treated)


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
    """Piecewise ITS runs with Poisson GLR and active interruption terms."""
    df, _ = generate_piecewise_its_data(
        N=80,
        interruption_times=[40],
        level_changes=[0.5],
        slope_changes=[0.01],
        noise_sigma=0.1,
        seed=3,
    )
    interruption = 40 / df["t"].max()
    df["t"] = df["t"] / df["t"].max()
    df["y"] = rng.poisson(np.clip(np.exp(-0.5 + 0.2 * df["t"]), 0.1, 10.0))
    step_col = (df["t"] >= interruption).astype(float)
    ramp_col = np.clip(df["t"] - interruption, 0.0, None)
    assert step_col.sum() > 0
    assert ramp_col.max() > 0
    result = cp.PiecewiseITS(
        df,
        formula=f"y ~ 1 + t + step(t, {interruption}) + ramp(t, {interruption})",
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


def test_validate_family_rejects_unknown_family():
    """Unknown families are rejected by the shared validator."""
    with pytest.raises(ValueError, match="Unsupported family"):
        _validate_family("gamma")


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


def test_create_family_likelihood_nb_scalar_alpha_fallback():
    """NB likelihood uses a scalar ``alpha`` when no dispersion prior is supplied."""
    priors = {"y_hat": Prior("NegativeBinomial", dims=["obs_ind", "treated_units"])}
    coords = {"obs_ind": np.arange(4), "treated_units": ["unit_0"]}
    with pm.Model(coords=coords) as model:
        mu = pm.Data("mu", np.ones((4, 1)), dims=["obs_ind", "treated_units"])
        y_obs = pm.Data("y", np.ones((4, 1)), dims=["obs_ind", "treated_units"])
        _create_family_likelihood(
            "negative_binomial", mu=mu, y_obs=y_obs, priors=priors
        )
    assert "y_hat" in model.named_vars


def test_glr_print_coefficients_multi_unit_header(capsys, rng, mock_pymc_sample):
    """Gaussian summaries print a header when multiple treated units are present."""
    n_obs = 8
    X = _design_matrix(n_obs, rng)
    y = xr.DataArray(
        rng.normal(size=(n_obs, 2)),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0", "unit_1"]},
    )
    coords = {
        "obs_ind": np.arange(n_obs),
        "coeffs": ["x0", "x1"],
        "treated_units": ["unit_0", "unit_1"],
    }
    model = GeneralizedLinearRegression(
        family="gaussian",
        sample_kwargs={**sample_kwargs, "random_seed": 45},
    )
    model.fit(X, y, coords)
    model.print_coefficients(["x0", "x1"])
    captured = capsys.readouterr().out
    assert "Treated unit: unit_0" in captured
    assert "Treated unit: unit_1" in captured


def test_glr_print_coefficients_reads_sigma_variable(capsys, rng, mock_pymc_sample):
    """Gaussian summaries accept a legacy ``sigma`` posterior variable name."""
    n_obs = 8
    X = _design_matrix(n_obs, rng)
    y = xr.DataArray(
        rng.normal(size=(n_obs, 1)),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
    )
    model = GeneralizedLinearRegression(
        family="gaussian",
        sample_kwargs={**sample_kwargs, "random_seed": 49},
    )
    model.fit(X, y, _single_unit_coords(n_obs))
    idata = model.idata.copy()
    posterior = idata.posterior.rename({"y_hat_sigma": "sigma"})
    idata.posterior = posterior
    model.idata = idata
    model.print_coefficients(["x0", "x1"])
    assert "y_hat_sigma" in capsys.readouterr().out


def test_glr_print_coefficients_requires_sigma_variable(rng, mock_pymc_sample):
    """Gaussian summaries error when neither ``sigma`` nor ``y_hat_sigma`` exist."""
    n_obs = 8
    X = _design_matrix(n_obs, rng)
    y = xr.DataArray(
        rng.normal(size=(n_obs, 1)),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
    )
    model = GeneralizedLinearRegression(
        family="gaussian",
        sample_kwargs={**sample_kwargs, "random_seed": 48},
    )
    model.fit(X, y, _single_unit_coords(n_obs))
    idata = model.idata.copy()
    idata.posterior = idata.posterior.drop_vars(["y_hat_sigma"], errors="ignore")
    model.idata = idata
    with pytest.raises(ValueError, match="Neither 'sigma' nor 'y_hat_sigma'"):
        model.print_coefficients(["x0", "x1"])


def test_fixed_mu_regression_identity_link_and_guardrails():
    """Deterministic test double supports identity link and fit/predict guards."""
    model = FixedMuRegression({0: 2.0}, link="identity")
    X = xr.DataArray(
        [[1.0]], dims=["obs_ind", "coeffs"], coords={"obs_ind": [0], "coeffs": ["x0"]}
    )
    with pytest.raises(RuntimeError, match="Model has not been fit"):
        model.predict(X)
    model.fit(X, xr.DataArray([[1.0]], dims=["obs_ind", "treated_units"]))
    bad = FixedMuRegression({0: 1.0}, link="logit")
    bad.fit(X, xr.DataArray([[1.0]], dims=["obs_ind", "treated_units"]))
    with pytest.raises(ValueError, match="Unsupported link"):
        bad.predict(X)


def test_did_att_requires_treated_post_rows():
    """DiD g-computation rejects designs with no treated post-treatment rows."""
    experiment = object.__new__(cp.DifferenceInDifferences)
    experiment.group_variable_name = "group"
    experiment.post_treatment_variable_name = "post_treatment"
    experiment.outcome_variable_name = "y"
    experiment.data = pd.DataFrame(
        {
            "group": [0, 0, 1, 1],
            "post_treatment": [False, True, False, False],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(ValueError, match="No treated post-treatment observations"):
        experiment._att_from_g_computation()


def test_prepost_att_requires_treated_rows():
    """PrePostNEGD g-computation rejects designs with no treated rows."""
    experiment = object.__new__(cp.PrePostNEGD)
    experiment.group_variable_name = "group"
    experiment.outcome_variable_name = "post"
    experiment.data = pd.DataFrame(
        {"group": [0, 0], "pre": [1.0, 2.0], "post": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="No treated observations"):
        experiment._att_from_g_computation()
