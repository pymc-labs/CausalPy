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

import sys
from importlib.metadata import version

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
from causalpy.pymc_forecast_models import PyMCForecastModel, _import_pymc_forecast

pymc_forecast = pytest.importorskip("pymc_forecast")


@pytest.fixture(scope="module", autouse=True)
def real_pymc_sampling():
    """Undo the session-scoped ``mock_pymc_sample`` patch for this module.

    The conftest registers PyMC's ``mock_sample`` (prior sampling instead of
    MCMC) session-wide, so once any earlier test module requests it,
    ``pm.sample`` stays mocked for the rest of the run. The round-trip tests
    here assert effect recovery against ground truth, which is meaningless
    under prior sampling — restore the real sampler for this module only.
    """
    import pymc as pm
    import pymc.sampling.mcmc

    patched = pm.sample
    pm.sample = pymc.sampling.mcmc.sample
    yield
    pm.sample = patched


# The y ~ 1 + t design leaves intercept and slope strongly correlated
# (unscaled t), so give NUTS a real warmup budget: tune=200 produced a
# badly adapted mass matrix and a biased posterior on CI's platform.
sample_kwargs = {
    "draws": 500,
    "tune": 1000,
    "chains": 2,
}

fast_sample_kwargs = {
    "draws": 100,
    "tune": 100,
    "chains": 2,
    "progressbar": False,
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


def make_fast_forecast_model():
    """Small HMC schedule for forecast-placebo control-flow tests."""
    return PyMCForecastModel(
        linear_model,
        forecaster_kwargs=dict(fast_sample_kwargs),
        num_samples=50,
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
    """Fit pre / forecast post-as-untreated / compare canonical draw-level impacts by subtracting posterior ``mu`` from observed outcomes."""

    def test_output_contract_matches_pymc_backend(self, forecast_result, pymc_result):
        """Draw-level posterior-predictive output mirrors the native backend."""
        for result in (forecast_result, pymc_result):
            mu = result.post_pred
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
            assert set(impact.dims) == {"chain", "draw", "obs_ind"}
            assert impact.dims[-1] == "obs_ind"
            mean_impact = float(impact.mean(("chain", "draw")).mean())
            assert mean_impact == pytest.approx(TRUE_EFFECT, abs=0.5)
        forecast_mean = float(
            forecast_result.post_impact.mean(("chain", "draw")).mean()
        )
        pymc_mean = float(pymc_result.post_impact.mean(("chain", "draw")).mean())
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
        plot_df = forecast_result.get_plot_data()
        assert {"prediction", "impact"}.issubset(plot_df.columns)

    def test_summaries_and_plot_data_without_plotting(
        self, forecast_result, capsys, monkeypatch
    ):
        """Forecast summary and plot-data contracts remain observable when
        generic plotting is unavailable."""

        def fail_if_plot_called(*args, **kwargs):
            raise AssertionError("summary helpers must not call plot()")

        monkeypatch.setattr(forecast_result, "plot", fail_if_plot_called)
        forecast_result.summary()
        assert "Model parameters:" in capsys.readouterr().out
        summary = forecast_result.effect_summary()
        assert len(summary.text) > 0
        plot_df = forecast_result.get_plot_data()
        assert {"prediction", "impact"}.issubset(plot_df.columns)

    def test_experiment_reports_bayesian_backend(self, forecast_result):
        assert forecast_result._model_backend.is_bayesian
        assert forecast_result._model_backend.kind == "pymc-forecast"
        assert forecast_result.idata.posterior is not None

    def test_mu_is_noise_free(self, forecast_result):
        """mu carries the upstream noise-free latent (mu/mu_future), so it is
        strictly narrower than the posterior predictive y_hat."""
        for X, mu, out_of_sample in (
            (forecast_result.pre_design["X"], forecast_result.pre_pred, False),
            (forecast_result.post_design["X"], forecast_result.post_pred, True),
        ):
            full_prediction = forecast_result.model.predict(
                X, out_of_sample=out_of_sample
            )
            y_hat = full_prediction["posterior_predictive"]["y_hat"]
            mu_spread = float(mu.std(("chain", "draw")).mean())
            y_hat_spread = float(y_hat.std(("chain", "draw")).mean())
            assert mu_spread < y_hat_spread
        # impact is computed from mu, i.e. excludes observation noise
        impact_spread = float(forecast_result.post_impact.std(("chain", "draw")).mean())
        assert impact_spread == pytest.approx(
            float(forecast_result.post_pred.std(("chain", "draw")).mean()), rel=1e-6
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
            mu = pred
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
            forecaster_kwargs=dict(fast_sample_kwargs),
            num_samples=50,
            random_seed=42,
        ),
    )
    mu = result.post_pred
    assert mu.dims == ("chain", "draw", "obs_ind", "treated_units")
    pd.testing.assert_index_equal(
        pd.Index(mu.coords["obs_ind"].values),
        result.datapost.index,
        check_names=False,
    )
    # A local level frozen at treatment time underestimates the trend, but the
    # level shift must dominate the impact estimate.
    mean_impact = float(result.post_impact.mean(("chain", "draw")).mean())
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
    assert adapter.supports_idata
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
    assert adapter.supports_idata
    assert adapter.idata is None
    with pytest.raises(RuntimeError, match="has not been fit"):
        adapter.require_idata()
    with pytest.raises(NotImplementedError, match="design-matrix coefficients"):
        adapter.coefficients()


def test_print_coefficients_before_fit_raises():
    with pytest.raises(RuntimeError, match="has not been fit"):
        make_forecast_model().print_coefficients([])


def test_print_coefficients_without_scalar_parameters(capsys):
    """Time-varying forecasting latents do not require scalar parameters."""
    model = make_forecast_model()
    posterior = xr.Dataset({"latent": (("chain", "draw", "time"), np.ones((1, 2, 3)))})
    model.idata = xr.DataTree.from_dict({"posterior": posterior})

    model.print_coefficients([])

    assert capsys.readouterr().out == (
        "Model parameters:\n  (no scalar parameters in posterior)\n"
    )


def test_print_coefficients_skips_non_scalar_parameters(capsys):
    """Only scalar posterior variables are reported as model parameters."""
    model = make_forecast_model()
    posterior = xr.Dataset(
        {
            "sigma": (("chain", "draw"), np.full((1, 2), 1.23456)),
            "latent": (("chain", "draw", "time"), np.ones((1, 2, 3))),
        }
    )
    model.idata = xr.DataTree.from_dict({"posterior": posterior})

    model.print_coefficients([], round_to=3)

    output = capsys.readouterr().out
    assert "sigma" in output
    assert "1.23" in output
    assert "latent" not in output


@pytest.mark.integration
def test_three_period_design(its_data):
    """treatment_end_time splitting works on the forecast backend's output."""
    df, treatment_time = its_data
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t",
        model=make_fast_forecast_model(),
        treatment_end_time=df.index[85],
    )
    assert result.intervention_pred.sizes["obs_ind"] == 15
    assert result.post_intervention_pred.sizes["obs_ind"] == 15
    summary = result.effect_summary(period="comparison")
    assert "persistence" in summary.text


def test_statespace_models_rejected():
    """Statespace backends lack a noise-free latent (pymc-forecast#50)."""
    with pytest.raises(NotImplementedError, match="pymc-forecast/issues/50"):
        PyMCForecastModel(linear_model, forecaster=pymc_forecast.StatespaceForecaster)


def test_statespace_model_instances_rejected():
    """The StatespaceModel branch cannot silently use noisy predictions."""

    class UnsupportedStatespaceModel(pymc_forecast.StatespaceModel):
        def statespace(self, data, covariates):
            raise AssertionError(
                "The rejection guard must run before model construction."
            )

        def priors(self, ss_mod, data, covariates):
            raise AssertionError(
                "The rejection guard must run before model construction."
            )

    with pytest.raises(NotImplementedError, match="pymc-forecast/issues/50"):
        PyMCForecastModel(UnsupportedStatespaceModel())


def test_clone_returns_unfitted_copy_with_same_config():
    """clone_model dispatches to _clone; the copy is unfitted but identically
    configured, so sensitivity checks can refit the same model spec."""
    from causalpy.checks.base import clone_model

    model = make_forecast_model()
    cloned = clone_model(model)
    assert isinstance(cloned, PyMCForecastModel)
    assert cloned is not model
    assert cloned.model_fn is model.model_fn
    assert cloned.forecaster_cls is model.forecaster_cls
    assert cloned.forecaster_kwargs == model.forecaster_kwargs
    assert cloned.forecaster is not model.forecaster
    assert cloned.num_samples == model.num_samples
    assert cloned.random_seed == model.random_seed
    assert cloned.idata is None


def test_fit_produces_datatree_with_posterior():
    """fit() wraps the thinned posterior subsample as a DataTree."""
    model = make_forecast_model()
    posterior = xr.Dataset(
        {
            "beta": (("chain", "draw", "coeffs"), np.ones((1, 2, 1))),
            "sigma": (("chain", "draw"), np.ones((1, 2))),
        },
        coords={"coeffs": ["x"]},
    )

    class FakeForecaster:
        def fit(self, *args, **kwargs):
            return None

        def draw_posterior(self, n, random_seed=None):
            assert n == model.num_samples
            return posterior

    model.forecaster = FakeForecaster()
    X, y = _design_arrays()
    idata = model.fit(X, y)

    assert isinstance(idata, xr.DataTree)
    assert "posterior" in idata
    assert model.idata is idata
    assert set(idata["posterior"].to_dataset().data_vars) == {"beta", "sigma"}


def test_to_inference_data_yields_datatree_posterior_predictive():
    """_to_inference_data returns a DataTree with a posterior_predictive group."""
    model = make_forecast_model()
    model._treated_units = ["unit_0"]
    obs_ind = pd.date_range("2020-01-01", periods=3, freq="D").to_numpy()
    mu = xr.DataArray(np.ones((1, 2, 3)), dims=("chain", "draw", "obs_ind"))
    y_hat = mu.copy()

    out = model._to_inference_data(mu, y_hat, obs_ind)

    assert isinstance(out, xr.DataTree)
    assert "posterior_predictive" in out
    pp = out["posterior_predictive"]
    assert "mu" in pp and "y_hat" in pp
    assert pp["mu"].dims == ("chain", "draw", "obs_ind", "treated_units")
    np.testing.assert_array_equal(
        pp["mu"].coords["obs_ind"].values.astype("datetime64[ns]"),
        obs_ind.astype("datetime64[ns]"),
    )


def test_to_inference_data_renames_series_dimension():
    """Upstream multi-series predictions retain their unit coordinate."""
    model = make_forecast_model()
    obs_ind = pd.date_range("2020-01-01", periods=3, freq="D").to_numpy()
    samples = xr.DataArray(
        np.ones((1, 2, 3, 2)),
        dims=("chain", "draw", "obs_ind", "series"),
        coords={"series": ["unit_a", "unit_b"]},
    )

    out = model._to_inference_data(samples, samples.copy(), obs_ind)

    pp = out["posterior_predictive"]
    assert pp["mu"].dims == ("chain", "draw", "obs_ind", "treated_units")
    assert list(pp["mu"].treated_units.values) == ["unit_a", "unit_b"]


@pytest.mark.integration
def test_fit_idata_exposes_full_fit_result(forecast_result):
    """fit_idata is the full NUTS result; idata the thinned draw-coherent
    posterior used for prediction."""
    model = forecast_result.model
    full = model.fit_idata
    assert isinstance(full, xr.DataTree)
    assert hasattr(full, "sample_stats")
    assert full.posterior.sizes["draw"] == sample_kwargs["draws"]
    thinned = forecast_result.idata
    assert thinned.posterior.sizes["chain"] == 1
    assert thinned.posterior.sizes["draw"] == model.num_samples

    with pytest.raises(RuntimeError, match="has not been fit"):
        _ = make_forecast_model().fit_idata


@pytest.mark.integration
class TestPlaceboInTime:
    """PlaceboInTime accepts the forecast backend and refits it via _clone."""

    def test_validate_accepts_forecast_backend(self, forecast_result):
        cp.checks.PlaceboInTime(n_folds=2).validate(forecast_result)

    def test_placebo_run_clones_and_refits(self, its_data, forecast_result):
        from causalpy.checks.base import clone_model

        df, _ = its_data
        base_model = make_fast_forecast_model()

        def factory(data, treatment_time):
            return cp.InterruptedTimeSeries(
                data,
                treatment_time,
                formula="y ~ 1 + t",
                model=clone_model(base_model),
            )

        # ``n_folds=1`` is forced by this fixture's geometry, not by convenience: the intervention window is 29 daily observations (``dates[70]`` through ``dates[99]``), so a fold is only eligible when at least 29 pre-treatment rows precede its pseudo treatment time. Two sequential folds would sit at ``dates[12]`` and ``dates[41]``, and the first is skipped as ``insufficient_pre_period``, so the fixture can never admit two eligible sequential folds. This test is about PlaceboInTime accepting and refitting the forecast backend, so one fitted fold exercises it fully; do not raise ``n_folds`` here.
        check = cp.checks.PlaceboInTime(
            n_folds=1,
            experiment_factory=factory,
            sample_kwargs=dict(fast_sample_kwargs),
            random_seed=42,
        )
        result = check.run(forecast_result)
        assert len(result.metadata["fold_results"]) == 1
        for fold in result.metadata["fold_results"]:
            fold_model = fold.experiment.model
            assert isinstance(fold_model, PyMCForecastModel)
            assert fold_model is not base_model
            assert fold_model.idata is not None

    def test_placebo_context_factory_clones_forecast_model(
        self, its_data, forecast_result
    ):
        """The PipelineContext factory path preserves the forecast backend."""
        df, treatment_time = its_data
        configured_model = make_fast_forecast_model()
        context = cp.PipelineContext(data=df)
        context.experiment = forecast_result
        context.experiment_config = {
            "method": cp.InterruptedTimeSeries,
            "treatment_time": treatment_time,
            "formula": "y ~ 1 + t",
            "model": configured_model,
        }

        check = cp.checks.PlaceboInTime(
            n_folds=1,
            sample_kwargs=dict(fast_sample_kwargs),
            random_seed=42,
        )
        result = check.run(forecast_result, context)
        fold_model = result.metadata["fold_results"][0].experiment.model
        assert isinstance(fold_model, PyMCForecastModel)
        assert fold_model is not configured_model
        assert fold_model.idata is not None


def test_pymc_forecast_02_prediction_schema_contract():
    """The optional extra's pinned 0.2 schema matches adapter consumption."""
    assert version("pymc-forecast").startswith("0.2.")
    assert {
        "OBS_VAR": pymc_forecast.OBS_VAR,
        "MU_VAR": pymc_forecast.MU_VAR,
        "FORECAST_VAR": pymc_forecast.FORECAST_VAR,
        "MU_FORECAST_VAR": pymc_forecast.MU_FORECAST_VAR,
        "TIME_DIM": pymc_forecast.TIME_DIM,
        "FUTURE_DIM": pymc_forecast.FUTURE_DIM,
    } == {
        "OBS_VAR": "obs",
        "MU_VAR": "mu",
        "FORECAST_VAR": "forecast",
        "MU_FORECAST_VAR": "mu_future",
        "TIME_DIM": "time",
        "FUTURE_DIM": "time_future",
    }
    assert callable(pymc_forecast.prediction_samples)


def test_missing_forecast_extra_has_actionable_install_error(monkeypatch):
    """The optional dependency boundary fails only at forecast-model creation."""
    monkeypatch.setitem(sys.modules, "pymc_forecast", None)
    with pytest.raises(ImportError, match=r"pip install causalpy\[forecast\]") as error:
        _import_pymc_forecast()
    assert "pymc-forecast[extras]>=0.2,<0.3" in str(error.value)


def test_default_forecaster_is_hmc(forecast_result):
    """The default backend executes the documented HMC forecaster."""
    assert isinstance(forecast_result.model.forecaster, pymc_forecast.HMCForecaster)


def test_default_forecaster_configuration_is_empty():
    """An omitted forecaster configuration constructs a usable HMC backend."""
    model = PyMCForecastModel(linear_model)
    assert isinstance(model.forecaster, pymc_forecast.HMCForecaster)
    assert model.forecaster_kwargs == {}


@pytest.mark.integration
@pytest.mark.parametrize(
    ("forecaster", "forecaster_kwargs"),
    [
        pytest.param(
            pymc_forecast.Forecaster,
            {"num_steps": 1_000, "progressbar": False},
            id="advi",
        ),
        pytest.param(
            pymc_forecast.PathfinderForecaster,
            {
                "pathfinder_kwargs": {
                    "num_paths": 1,
                    "num_draws": 50,
                    "num_draws_per_path": 50,
                    "num_elbo_draws": 5,
                    "parallel": False,
                },
                "progressbar": False,
            },
            id="pathfinder",
        ),
    ],
)
def test_approximate_forecasters_fit_and_predict_canonical_containers(
    its_data, forecaster, forecaster_kwargs
):
    """ADVI and Pathfinder retain full fitted DataTrees and emit canonical
    prediction DataArrays without making an accuracy claim."""
    df, treatment_time = its_data
    num_samples = 10
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t",
        model=PyMCForecastModel(
            linear_model,
            forecaster=forecaster,
            forecaster_kwargs=forecaster_kwargs,
            num_samples=num_samples,
            random_seed=42,
        ),
    )

    model = result.model
    assert isinstance(model.forecaster, forecaster)
    assert model.forecaster.is_fitted
    assert isinstance(model.idata, xr.DataTree)
    assert model.idata.posterior.sizes["draw"] == num_samples
    if forecaster is pymc_forecast.Forecaster:
        with pytest.raises(AttributeError, match="does not retain a full DataTree"):
            _ = model.fit_idata
    else:
        assert isinstance(model.fit_idata, xr.DataTree)
    assert list(result.score.index) == ["unit_0_r2", "unit_0_r2_std"]
    for pred, expected_index in (
        (result.pre_pred, result.datapre.index),
        (result.post_pred, result.datapost.index),
    ):
        assert isinstance(pred, xr.DataArray)
        assert pred.name == "mu"
        assert pred.dims == ("chain", "draw", "obs_ind", "treated_units")
        pd.testing.assert_index_equal(
            pd.Index(pred.obs_ind.values), expected_index, check_names=False
        )
