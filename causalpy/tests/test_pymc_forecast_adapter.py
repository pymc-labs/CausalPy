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
"""Tests for the pymc-forecast model-provider adapter behind
InterruptedTimeSeries (issue #1013)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr

import causalpy as cp
from causalpy.experiments.model_adapter import (
    PyMCForecastAdapter,
    make_model_adapter,
)
from causalpy.pymc_forecast_models import PyMCForecastModel

pymc_forecast = pytest.importorskip("pymc_forecast")

sample_kwargs = {
    "draws": 200,
    "tune": 200,
    "chains": 2,
}

TRUE_EFFECT = 2.0


def linear_model(h, covariates):
    """Static regression on the patsy design matrix (intercept + trend)."""
    beta = pm.Normal("beta", 0.0, 2.0, dims="covariate")
    sigma = pm.HalfNormal("sigma", 1.0)
    mu = pt.dot(covariates.values, beta)
    pymc_forecast.predict(
        h,
        lambda name, m, dims, observed: pm.Normal(
            name, m, sigma, dims=dims, observed=observed
        ),
        mu,
    )


class LocalLevel(pymc_forecast.ForecastingModel):  # type: ignore[name-defined]
    """Covariate-free local-level (random-walk drift) model."""

    def model(self, h, covariates):
        drift = self.time_series(
            "drift", lambda name, dims: pm.Normal(name, 0.0, 0.1, dims=dims)
        )
        sigma = pm.HalfNormal("sigma", 1.0)
        self.predict(
            lambda name, m, dims, observed: pm.Normal(
                name, m, sigma, dims=dims, observed=observed
            ),
            pt.cumsum(drift),
        )


@pytest.fixture(scope="module")
def its_data():
    """Linear trend with a known level shift after treatment."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    t = np.arange(100)
    y = 2.0 + 0.05 * t + rng.normal(0, 0.3, 100)
    y[70:] += TRUE_EFFECT
    df = pd.DataFrame({"y": y, "t": t.astype(float)}, index=dates)
    return df, dates[70]


def make_forecast_model():
    return PyMCForecastModel(
        linear_model,
        forecaster_kwargs=dict(sample_kwargs),
        num_samples=200,
        random_seed=42,
    )


@pytest.fixture(scope="module")
def forecast_result(its_data):
    df, treatment_time = its_data
    return cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t",
        model=make_forecast_model(),
    )


@pytest.fixture(scope="module")
def pymc_result(its_data):
    df, treatment_time = its_data
    return cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t",
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={**sample_kwargs, "progressbar": False, "random_seed": 42}
        ),
    )


@pytest.mark.integration
class TestRoundTripAgainstPyMCBackend:
    """Fit pre / forecast post-as-untreated / calculate_impact on draw-level
    samples, checked against the existing native PyMC path."""

    def test_output_contract_matches_pymc_backend(self, forecast_result, pymc_result):
        """Draw-level posterior-predictive output mirrors the native backend."""
        for result in (forecast_result, pymc_result):
            mu = result.post_pred["posterior_predictive"]["mu"]
            assert mu.dims == ("chain", "draw", "obs_ind", "treated_units")
            assert list(mu.coords["treated_units"].values) == ["unit_0"]
            pd.testing.assert_index_equal(
                pd.Index(mu.coords["obs_ind"].values),
                result.datapost.index,
                check_names=False,
            )

    def test_impact_recovers_true_effect(self, forecast_result, pymc_result):
        """Both backends recover the simulated level shift at the draw level."""
        for result in (forecast_result, pymc_result):
            impact = result.post_impact
            assert set(impact.dims) == {"chain", "draw", "obs_ind", "treated_units"}
            assert impact.dims[-1] == "obs_ind"
            mean_impact = float(
                impact.mean(("chain", "draw")).isel(treated_units=0).mean()
            )
            assert mean_impact == pytest.approx(TRUE_EFFECT, abs=0.5)
        forecast_mean = float(
            forecast_result.post_impact.mean(("chain", "draw"))
            .isel(treated_units=0)
            .mean()
        )
        pymc_mean = float(
            pymc_result.post_impact.mean(("chain", "draw")).isel(treated_units=0).mean()
        )
        assert forecast_mean == pytest.approx(pymc_mean, abs=0.5)

    def test_cumulative_impact(self, forecast_result):
        cum = forecast_result.post_impact_cumulative
        assert "obs_ind" in cum.dims
        last = float(cum.isel(obs_ind=-1).mean(("chain", "draw")).squeeze())
        n_post = len(forecast_result.datapost)
        assert last == pytest.approx(TRUE_EFFECT * n_post, rel=0.4)

    def test_score_matches_pymc_shape(self, forecast_result, pymc_result):
        assert list(forecast_result.score.index) == list(pymc_result.score.index)
        assert forecast_result.score["unit_0_r2"] > 0.7

    def test_plot_and_summaries_smoke(self, forecast_result, capsys):
        fig, ax = forecast_result.plot(show=False)
        assert len(ax) == 3
        forecast_result.summary()
        assert "Model parameters:" in capsys.readouterr().out
        summary = forecast_result.effect_summary()
        assert len(summary.text) > 0
        plot_df = forecast_result.get_plot_data_bayesian()
        assert {"prediction", "impact"}.issubset(plot_df.columns)

    def test_experiment_reports_bayesian_backend(self, forecast_result):
        assert forecast_result._model_backend.is_bayesian
        assert forecast_result._model_backend.kind == "pymc-forecast"
        assert forecast_result.idata.posterior is not None

    def test_mu_is_noise_free(self, forecast_result):
        """mu carries the upstream noise-free latent (mu/mu_future), so it is
        strictly narrower than the posterior predictive y_hat."""
        for pred in (forecast_result.pre_pred, forecast_result.post_pred):
            pp = pred["posterior_predictive"]
            mu_spread = float(pp["mu"].std(("chain", "draw")).mean())
            y_hat_spread = float(pp["y_hat"].std(("chain", "draw")).mean())
            assert mu_spread < y_hat_spread
        # impact is computed from mu, i.e. excludes observation noise
        impact_spread = float(forecast_result.post_impact.std(("chain", "draw")).mean())
        post_pp = forecast_result.post_pred["posterior_predictive"]
        assert impact_spread == pytest.approx(
            float(post_pp["mu"].std(("chain", "draw")).mean()), rel=1e-6
        )

    def test_predictions_are_draw_coherent(self, forecast_result):
        """One posterior is drawn at fit time and shared by every predictive
        call: mu is a deterministic function of the shared draws, so repeated
        calls reproduce it exactly, and pre/post mu come from the same draws
        (checked through the linear model: mu = X @ beta draw-for-draw)."""
        model = forecast_result.model
        posterior = forecast_result.idata.posterior
        for X, pred, out_of_sample in (
            (forecast_result.pre_design["X"], forecast_result.pre_pred, False),
            (forecast_result.post_design["X"], forecast_result.post_pred, True),
        ):
            mu = pred["posterior_predictive"]["mu"]
            expected = xr.dot(
                posterior["beta"],
                X.rename({"coeffs": "covariate"}),
                dim="covariate",
            )
            np.testing.assert_allclose(
                mu.isel(treated_units=0).transpose("chain", "draw", "obs_ind").values,
                expected.transpose("chain", "draw", "obs_ind").values,
                rtol=1e-5,
            )
            again = model.predict(X, out_of_sample=out_of_sample)
            np.testing.assert_allclose(
                mu.values, again["posterior_predictive"]["mu"].values, rtol=1e-6
            )


@pytest.mark.integration
def test_covariate_free_future_index_path(its_data):
    """A covariate-free model forecasts over the post-period index via
    ``forecast(future_index=...)``."""
    df, treatment_time = its_data
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 0",
        model=PyMCForecastModel(
            LocalLevel(),
            forecaster_kwargs=dict(sample_kwargs),
            num_samples=200,
            random_seed=42,
        ),
    )
    mu = result.post_pred["posterior_predictive"]["mu"]
    assert mu.dims == ("chain", "draw", "obs_ind", "treated_units")
    pd.testing.assert_index_equal(
        pd.Index(mu.coords["obs_ind"].values),
        result.datapost.index,
        check_names=False,
    )
    # A local level frozen at treatment time underestimates the trend, but the
    # level shift must dominate the impact estimate.
    mean_impact = float(
        result.post_impact.mean(("chain", "draw")).isel(treated_units=0).mean()
    )
    assert mean_impact > TRUE_EFFECT / 2


def _design_arrays(n_units: int = 1):
    obs_ind = pd.date_range("2020-01-01", periods=10, freq="D")
    X = xr.DataArray(
        np.random.default_rng(0).normal(size=(10, 1)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": obs_ind, "coeffs": ["x"]},
    )
    y = xr.DataArray(
        np.random.default_rng(1).normal(size=(10, n_units)),
        dims=["obs_ind", "treated_units"],
        coords={
            "obs_ind": obs_ind,
            "treated_units": [f"unit_{i}" for i in range(n_units)],
        },
    )
    return X, y


def test_multiple_treated_units_rejected():
    X, y = _design_arrays(n_units=2)
    with pytest.raises(ValueError, match="single treated unit"):
        make_forecast_model().fit(X, y)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="has not been fit"):
        make_forecast_model().predict(_design_arrays()[0])


def test_adapter_resolution_and_gating():
    """make_model_adapter wraps the model and experiments must opt in."""
    model = make_forecast_model()
    adapter = make_model_adapter(
        model,
        default_model_class=None,
        supports_bayes=True,
        supports_ols=True,
        supports_pymc_forecast=True,
    )
    assert isinstance(adapter, PyMCForecastAdapter)
    assert adapter.is_bayesian
    assert not adapter.is_ols
    assert adapter.model is model
    with pytest.raises(ValueError, match="pymc-forecast models not supported"):
        make_model_adapter(
            model,
            default_model_class=None,
            supports_bayes=True,
            supports_ols=True,
            supports_pymc_forecast=False,
        )


def test_unfit_adapter_has_no_idata_or_coefficients():
    adapter = make_model_adapter(
        make_forecast_model(),
        default_model_class=None,
        supports_bayes=True,
        supports_ols=True,
        supports_pymc_forecast=True,
    )
    with pytest.raises(RuntimeError, match="has not been fit"):
        _ = adapter.idata
    with pytest.raises(NotImplementedError, match="design-matrix coefficients"):
        adapter.coefficients()


@pytest.mark.integration
def test_three_period_design(its_data):
    """treatment_end_time splitting works on the forecast backend's output."""
    df, treatment_time = its_data
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t",
        model=make_forecast_model(),
        treatment_end_time=df.index[85],
    )
    assert result.intervention_pred.posterior_predictive["mu"].sizes["obs_ind"] == 15
    assert (
        result.post_intervention_pred.posterior_predictive["mu"].sizes["obs_ind"] == 15
    )
    summary = result.effect_summary(period="comparison")
    assert "persistence" in summary.text
