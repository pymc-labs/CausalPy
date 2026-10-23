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
"""Known-truth reproductions of the basis-expansion prior-scale bug (#935)."""

import importlib.metadata
import json
from time import perf_counter

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pymc_extras.prior import Prior

import causalpy as cp


class _ZeroComponent:
    """Remove optional components when testing the regression prior alone."""

    def apply(self, t):
        return pm.math.zeros_like(t)


@pytest.fixture
def basis_inputs():
    """A nonconstant outcome with intercept, predictor, and zero columns."""
    dates = pd.date_range("2023-01-01", periods=8)
    X = xr.DataArray(
        np.column_stack([np.ones(8), np.linspace(-2, 2, 8), np.zeros(8)]),
        dims=("obs_ind", "coeffs"),
        coords={"obs_ind": dates, "coeffs": ["Intercept", "x", "zero"]},
    )
    y = xr.DataArray(
        np.arange(50.0, 58.0)[:, None],
        dims=("obs_ind", "treated_units"),
        coords={"obs_ind": dates, "treated_units": ["y"]},
    )
    return X, y


@pytest.mark.parametrize("unit_scale", [1e-9, 1.0, 1e6])
def test_basis_expansion_prior_scales_follow_outcome_units(basis_inputs, unit_scale):
    """All default parameter scales change together under a change of units."""
    X, y = basis_inputs
    model = cp.pymc_models.BayesianBasisExpansionTimeSeries()
    baseline = model.priors_from_data(X, y)
    scaled = model.priors_from_data(X, y * unit_scale)
    for name, parameter in (
        ("beta", "sigma"),
        ("sigma", "sigma"),
        ("delta", "b"),
        ("fourier_beta", "b"),
    ):
        np.testing.assert_allclose(
            scaled[name].parameters[parameter],
            np.asarray(baseline[name].parameters[parameter]) * unit_scale,
        )


def test_basis_expansion_coefficient_scale_accounts_for_predictor_units(basis_inputs):
    """Rescaling a design column inversely rescales its coefficient prior."""
    X, y = basis_inputs
    model = cp.pymc_models.BayesianBasisExpansionTimeSeries()
    baseline = model.priors_from_data(X, y)["beta"].parameters["sigma"]
    scaled = model.priors_from_data(X * [1, 1000, 1], y)["beta"].parameters["sigma"]
    np.testing.assert_allclose(scaled, baseline / [1, 1000, 1])
    assert np.isfinite(scaled).all()


@pytest.mark.parametrize("auto_scale_priors", [True, False])
def test_basis_expansion_explicit_priors_reach_default_components(
    basis_inputs, auto_scale_priors
):
    """User priors replace every automatically chosen prior in the actual graph."""
    pytest.importorskip("pymc_marketing")
    X, y = basis_inputs
    priors = {
        "beta": Prior("Normal", mu=123, sigma=2, dims="coeffs"),
        "sigma": Prior("HalfNormal", sigma=7, dims="treated_units"),
        "delta": Prior("Laplace", mu=0, b=13, dims="changepoint"),
        "fourier_beta": Prior("Laplace", mu=0, b=17, dims="fourier"),
    }
    model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        priors=priors, prior_sigma=99, auto_scale_priors=auto_scale_priors
    )
    model.build(X, y)
    for name, scale in (("beta", 2), ("sigma", 7), ("delta", 13), ("fourier_beta", 17)):
        np.testing.assert_allclose(model[name].owner.inputs[-1].eval(), scale)
        assert model.priors[name] is priors[name]


@pytest.mark.parametrize("auto_scale_priors", [True, False])
@pytest.mark.parametrize("build_method", ["build", "build_model"])
def test_basis_expansion_explicit_noise_scale_is_preserved(
    basis_inputs, auto_scale_priors, build_method
):
    X, y = basis_inputs
    model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        prior_sigma=123,
        auto_scale_priors=auto_scale_priors,
        trend_component=_ZeroComponent(),
        seasonality_component=_ZeroComponent(),
    )
    getattr(model, build_method)(X, y, coords=None)
    np.testing.assert_allclose(model["sigma"].owner.inputs[-1].eval(), 123)


@pytest.mark.parametrize("prior_sigma", [0, -1, np.inf, np.nan])
def test_basis_expansion_rejects_invalid_explicit_noise_scale(prior_sigma):
    with pytest.raises(ValueError, match="finite and strictly positive"):
        cp.pymc_models.BayesianBasisExpansionTimeSeries(prior_sigma=prior_sigma)


@pytest.mark.parametrize(
    "values",
    [[0.0] * 8, [50_000.0] * 8, [1e-12] * 8, [50_000.0], [1.0, np.nextafter(1.0, 2.0)]],
)
def test_basis_expansion_degenerate_spread_has_finite_graph(values):
    """No-spread data warn and use a documented finite, positive fallback."""
    dates = pd.date_range("2023-01-01", periods=len(values))
    X = xr.DataArray(
        np.ones((len(values), 1)),
        dims=("obs_ind", "coeffs"),
        coords={"obs_ind": dates, "coeffs": ["Intercept"]},
    )
    y = xr.DataArray(
        np.asarray(values)[:, None],
        dims=("obs_ind", "treated_units"),
        coords={"obs_ind": dates, "treated_units": ["y"]},
    )
    model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        trend_component=_ZeroComponent(), seasonality_component=_ZeroComponent()
    )
    with pytest.warns(UserWarning, match="Cannot estimate the fitted outcome spread"):
        model.build(X, y)
    expected = max(abs(value) for value in values) or 1.0
    assert model.priors["sigma"].parameters["sigma"] == expected
    assert np.isfinite(model.compile_logp()(model.initial_point()))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("invalid_input", ["outcome", "predictor"])
def test_basis_expansion_rejects_nonfinite_calibration_data(
    basis_inputs, invalid, invalid_input
):
    X, y = basis_inputs
    X, y = X.copy(), y.copy()
    (y if invalid_input == "outcome" else X).values[0, 0] = invalid
    model = cp.pymc_models.BayesianBasisExpansionTimeSeries()
    with pytest.raises(
        ValueError, match="finite fitted outcomes and finite predictors"
    ):
        model.priors_from_data(X, y)


def test_basis_expansion_requires_nonempty_single_outcome(basis_inputs):
    X, y = basis_inputs
    model = cp.pymc_models.BayesianBasisExpansionTimeSeries()
    with pytest.raises(ValueError, match="non-empty"):
        model.priors_from_data(X[:0], y[:0])
    with pytest.raises(ValueError, match="one treated outcome"):
        model.priors_from_data(X, xr.concat([y, y], dim="treated_units"))
    with pytest.raises(TypeError, match="DataArray inputs"):
        model.priors_from_data({"X": X}, {"y": y})


def test_basis_expansion_preserves_subclass_default(basis_inputs):
    X, y = basis_inputs

    class CustomDefault(cp.pymc_models.BayesianBasisExpansionTimeSeries):
        default_priors = {
            **cp.pymc_models.BayesianBasisExpansionTimeSeries.default_priors,
            "beta": Prior("Normal", mu=7, sigma=3, dims="coeffs"),
        }

    model = CustomDefault(
        trend_component=_ZeroComponent(), seasonality_component=_ZeroComponent()
    )
    model.build(X, y)
    np.testing.assert_equal(model["beta"].owner.inputs[-2].eval(), 7)


def test_basis_expansion_custom_components_and_clone_are_preserved(basis_inputs):
    """Automatic scaling must not rewrite any part of a supplied component."""
    X, y = basis_inputs

    class CustomComponent:
        def __init__(self, name, scale):
            self.name = name
            self.prior = Prior("Normal", mu=0, sigma=scale)

        def apply(self, t):
            return self.prior.create_variable(self.name) * pm.math.ones_like(t)

    trend = CustomComponent("custom_trend", 123)
    seasonality = CustomComponent("custom_seasonality", 456)
    user_priors = {"beta": Prior("Normal", mu=0, sigma=3, dims="coeffs")}
    model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        trend_component=trend,
        seasonality_component=seasonality,
        prior_sigma=7,
        auto_scale_priors=False,
        priors=user_priors,
    )
    preserved_clone = model._clone()
    preserved_clone.build(X, y)
    np.testing.assert_equal(preserved_clone["beta"].owner.inputs[-1].eval(), 3)
    clone = model._clone(priors={"beta": Prior("Normal", mu=0, sigma=9, dims="coeffs")})
    assert clone.priors["beta"].parameters["sigma"] == 9
    assert clone.auto_scale_priors is False
    assert clone.prior_sigma == 7
    assert clone._custom_trend_component is trend
    assert clone._custom_seasonality_component is seasonality
    # Build with scaling enabled as well, while preserving both custom objects.
    model.auto_scale_priors = True
    model.build(X, y)
    for name, scale in (("custom_trend", 123), ("custom_seasonality", 456)):
        np.testing.assert_equal(model[name].owner.inputs[-1].eval(), scale)
    assert model._get_trend_component() is trend
    assert model._get_seasonality_component() is seasonality


def test_basis_expansion_calibration_ignores_post_intervention_outcomes():
    """Changing only held-out outcomes must leave every calibrated prior fixed."""
    models = []
    dates = pd.date_range("2023-01-01", periods=12)
    for post_value in (1e-12, 1e12):
        data = pd.DataFrame(
            {"y": np.r_[np.arange(8.0) + 50, np.full(4, post_value)]}, index=dates
        )
        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            trend_component=_ZeroComponent(), seasonality_component=_ZeroComponent()
        )
        experiment = cp.InterruptedTimeSeries(
            data, treatment_time=dates[8], formula="y ~ 1", model=model
        )
        experiment.build()
        models.append(model)
        np.testing.assert_equal(model["y"].get_value().ravel(), data.y.iloc[:8])
    for name, parameter in (("beta", "sigma"), ("sigma", "sigma")):
        np.testing.assert_equal(
            models[0].priors[name].parameters[parameter],
            models[1].priors[name].parameters[parameter],
        )


def test_basis_expansion_inspected_components_receive_calibrated_priors(basis_inputs):
    """Inspecting lazy components before build must not freeze legacy priors."""
    pytest.importorskip("pymc_marketing")
    X, y = basis_inputs
    model = cp.pymc_models.BayesianBasisExpansionTimeSeries()
    old_trend = model._get_trend_component()
    old_seasonality = model._get_seasonality_component()
    model.build(X, y)
    assert model._get_trend_component() is not old_trend
    assert model._get_seasonality_component() is not old_seasonality
    assert model._get_trend_component().priors["delta"] == model.priors["delta"]
    assert model._get_seasonality_component().prior == model.priors["fourier_beta"]


@pytest.mark.parametrize("n_changepoints", [1, 2, 10])
def test_basis_expansion_initial_slope_scale_is_independent_of_knots(
    basis_inputs, n_changepoints
):
    """Adding knots must not tighten the initial slope's training contribution."""
    X, y = basis_inputs
    model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        n_changepoints_trend=n_changepoints
    )
    scales = model.priors_from_data(X, y)["delta"].parameters["b"]
    elapsed_days = (X.obs_ind.values - X.obs_ind.values[0]) / np.timedelta64(1, "D")
    implied_spread = np.sqrt(2) * scales[0] * np.std(elapsed_days, ddof=1) / 365.25
    assert implied_spread == pytest.approx(float(y.std(ddof=1)))


@pytest.mark.parametrize("n_changepoints", [1, 10])
def test_basis_expansion_component_prior_predictive_scales(
    basis_inputs, n_changepoints
):
    """Verify contribution scales using draws from the real component graph."""
    pytest.importorskip("pymc_marketing")
    X, y = basis_inputs
    model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        n_changepoints_trend=n_changepoints
    )
    model.build(X, y)
    with model:
        prior = pm.sample_prior_predictive(
            draws=4_000,
            random_seed=935,
            var_names=["trend_component", "season_component"],
        )["prior"].to_dataset()
    spread = float(y.std(ddof=1))
    season_sd = prior["season_component"].std(("chain", "draw")).values
    np.testing.assert_allclose(season_sd, spread, rtol=0.08)
    expected_trend_variance = float(
        prior["trend_component"].var("obs_ind", ddof=1).mean(("chain", "draw"))
    )
    assert (spread * 0.92) ** 2 < expected_trend_variance < 2 * (spread * 1.08) ** 2


def test_basis_expansion_beta_override_reaches_graph():
    """A documented user prior must affect the distribution in the graph."""
    dates = pd.date_range("2023-01-01", periods=4)
    X = xr.DataArray(
        np.ones((4, 1)),
        dims=("obs_ind", "coeffs"),
        coords={"obs_ind": dates, "coeffs": ["Intercept"]},
    )
    y = xr.DataArray(
        np.full((4, 1), 123.0),
        dims=("obs_ind", "treated_units"),
        coords={"obs_ind": dates, "treated_units": ["y"]},
    )
    prior = Prior("Normal", mu=123, sigma=2, dims="coeffs")
    model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        trend_component=_ZeroComponent(),
        seasonality_component=_ZeroComponent(),
        priors={"beta": prior},
    )
    model.build(X, y)

    assert model.priors["beta"] is prior
    np.testing.assert_equal(model["beta"].owner.inputs[-2].eval(), 123)


@pytest.mark.integration
@pytest.mark.correctness
@pytest.mark.nightly
@pytest.mark.parametrize(
    "unit_scale, auto_scale_priors, n_changepoints, data_seed",
    [
        (1.0, True, 1, 0),
        (1_000.0, True, 1, 0),
        (0.001, True, 1, 0),
        (1.0, False, 1, 0),
        (1_000.0, False, 1, 0),
        (1.0, True, 10, 0),
        (1_000.0, True, 10, 0),
        (1.0, True, 1, 42),
        (1_000.0, True, 1, 42),
        (1.0, True, 10, 42),
        (1_000.0, True, 10, 42),
    ],
)
def test_basis_expansion_known_truth_scale_comparison(
    unit_scale,
    auto_scale_priors,
    n_changepoints,
    data_seed,
    real_pymc_sampling,
    tmp_path,
):
    """Outcome units should preserve recovery with a correctly specified basis.

    The annual Fourier basis matches the DGP exactly. The single-changepoint cases test parameter recovery; the ten-changepoint cases test forecasts without claiming recovery of individual local slopes. Seed 0 is the development reproducer, and seed 42 confirms the frozen criteria on fresh data. Data and MCMC seeds are matched within each unit comparison; all reported estimates and errors are converted back to the original units. Diagnostics and parameter intervals are recorded before assertions, including interval misses.
    """
    pytest.importorskip("pymc_marketing")
    n, n_pre = 800, 700
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    elapsed_days = np.arange(n)
    seasonality = 8 * np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)
    truth = 50 + 0.02 * elapsed_days + seasonality
    outcome = truth + np.random.default_rng(data_seed).normal(0, 3, n)
    data = pd.DataFrame({"y": unit_scale * outcome}, index=dates)
    model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        n_order=1 if n_changepoints == 1 else 3,
        n_changepoints_trend=n_changepoints,
        auto_scale_priors=auto_scale_priors,
        sample_kwargs={
            "chains": 2,
            "cores": 1,
            "draws": 600,
            "tune": 600,
            "target_accept": 0.95,
            "random_seed": 935,
            "progressbar": False,
        },
        prior_sample_kwargs={"draws": 50, "random_seed": 935},
    )
    experiment = cp.InterruptedTimeSeries(
        data=data,
        treatment_time=dates[n_pre],
        formula="y ~ 1",
        model=model,
    )
    start = perf_counter()
    experiment.fit()
    posterior = experiment.idata["posterior"].to_dataset()
    sample_stats = experiment.idata["sample_stats"].to_dataset()
    parameter_names = ["beta", "delta", "fourier_beta", "sigma"]
    parameters = posterior[parameter_names]
    mean = parameters.mean(("chain", "draw"))
    estimates = {
        "intercept": float(mean["beta"].values[0] / unit_scale),
        "slope_per_day": float(mean["delta"].values[0] / unit_scale / 365.25),
        "seasonal_amplitude": float(
            np.linalg.norm(mean["fourier_beta"].values) / unit_scale
        ),
        "observation_sigma": float(mean["sigma"].values[0] / unit_scale),
    }
    prediction = (
        xr.concat(
            [experiment.result.predictions_pre, experiment.result.predictions_post],
            dim="obs_ind",
        ).isel(treated_units=0)
        / unit_scale
    )
    prediction_mean = prediction.mean(("chain", "draw")).values
    summary = {
        "unit_scale": unit_scale,
        "auto_scale_priors": auto_scale_priors,
        "n_changepoints": n_changepoints,
        "n_order": model.n_order,
        "data_seed": data_seed,
        "sampling_seed": 935,
        "n_pre": n_pre,
        "n_post": n - n_pre,
        "chains": 2,
        "draws_per_chain": 600,
        "tune_per_chain": 600,
        "elapsed_seconds": perf_counter() - start,
        "truth": {
            "intercept": 50,
            "slope_per_day": 0.02,
            "seasonal_amplitude": 8,
            "observation_sigma": 3,
        },
        "slope_definition": "Initial segment: delta[0] / 365.25",
        "initial_slope_posterior_sd": float(
            parameters["delta"].isel(changepoint=0).std(("chain", "draw"))
            / unit_scale
            / 365.25
        ),
        "estimates": estimates,
        "parameter_intervals_94": {
            name: (
                parameters[name].quantile([0.03, 0.97], ("chain", "draw")).values
                / unit_scale
            ).tolist()
            for name in parameter_names
        },
        "pre_truth_rmse": float(
            np.sqrt(np.mean((prediction_mean[:n_pre] - truth[:n_pre]) ** 2))
        ),
        "post_truth_rmse": float(
            np.sqrt(np.mean((prediction_mean[n_pre:] - truth[n_pre:]) ** 2))
        ),
        "divergences": int(sample_stats["diverging"].sum()),
        "max_rhat": float(az.rhat(parameters).to_array().max()),
        "min_bulk_ess": float(az.ess(parameters, method="bulk").to_array().min()),
        "min_bfmi": float(az.bfmi(sample_stats["energy"]).min()),
        "versions": {
            package: importlib.metadata.version(package)
            for package in (
                "pymc",
                "pytensor",
                "pymc-extras",
                "pymc-marketing",
                "arviz",
            )
        },
    }
    (tmp_path / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        tmp_path / "predictions.npz",
        truth=truth,
        observed=outcome,
        mean=prediction_mean,
        lower=prediction.quantile(0.03, ("chain", "draw")).values,
        upper=prediction.quantile(0.97, ("chain", "draw")).values,
    )
    print(json.dumps(summary, indent=2))

    # Check sampling separately from recovery: a converged posterior can still be badly biased by the priors.
    assert summary["divergences"] == 0
    assert summary["max_rhat"] < 1.05
    assert summary["min_bulk_ess"] > 100
    if not auto_scale_priors and unit_scale == 1_000:
        # Pin the opt-out to the original failure as a matched negative control.
        assert estimates["seasonal_amplitude"] < 1
        assert summary["post_truth_rmse"] > 10
        return
    assert abs(estimates["intercept"] - 50) < 2
    if n_changepoints == 1:
        assert abs(estimates["slope_per_day"] - 0.02) < 0.005
    assert abs(estimates["seasonal_amplitude"] - 8) < 1
    assert abs(estimates["observation_sigma"] - 3) < 0.5
    assert summary["post_truth_rmse"] < 2
