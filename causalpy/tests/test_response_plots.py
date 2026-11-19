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
"""Outcome target selection through public plot and export APIs."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.data.simulate_data import generate_staggered_did_data
from causalpy.experiments.interactive_fixed_effects import (
    InteractiveFixedEffectsSyntheticControl,
)
from causalpy.experiments.model_adapter import PyMCForecastAdapter
from causalpy.plot_utils import get_hdi_to_df


@pytest.fixture(
    params=["its", "piecewise", "synthetic", "did", "prepost", "rd", "kink"]
)
def outcome_experiment(
    request, mock_pymc_sample, did_data, anova1_data, rd_data, sc_data
):
    # Mocked sampling checks public wiring, not posterior recovery.
    model = cp.pymc_models.LinearRegression(
        sample_kwargs={
            "draws": 20,
            "tune": 20,
            "chains": 2,
            "cores": 1,
            "random_seed": 41,
        }
    )
    if request.param == "its":
        data = pd.DataFrame(
            {"t": np.arange(16), "y": np.arange(16) + 2.0},
            index=pd.date_range("2020-01-01", periods=16),
        )
        experiment = cp.InterruptedTimeSeries(
            data, data.index[8], formula="y ~ 1 + t", model=model
        )
    elif request.param == "piecewise":
        data = pd.DataFrame({"t": np.arange(8), "y": np.arange(8) + 2.0})
        experiment = cp.PiecewiseITS(
            data, formula="y ~ 1 + step(t, 6) + step(t, 2)", model=model
        )
    elif request.param == "synthetic":
        experiment = cp.SyntheticControl(
            sc_data,
            70,
            control_units=["a", "b", "c"],
            treated_units=["actual"],
            model=cp.pymc_models.WeightedSumFitter(sample_kwargs=model.sample_kwargs),
        )
    elif request.param == "did":
        experiment = cp.DifferenceInDifferences(
            did_data,
            formula="y ~ 1 + group + t + group:post_treatment",
            time_variable_name="t",
            group_variable_name="group",
            model=model,
        )
    elif request.param == "prepost":
        experiment = cp.PrePostNEGD(
            anova1_data,
            formula="post ~ 1 + C(group) + pre",
            group_variable_name="group",
            pretreatment_variable_name="pre",
            model=model,
        )
    elif request.param == "rd":
        experiment = cp.RegressionDiscontinuity(
            rd_data, formula="y ~ 1 + x + treated", treatment_threshold=0.5, model=model
        )
    else:
        data = pd.DataFrame({"x": np.linspace(-1, 1, 20)})
        data["treated"] = data.x >= 0
        data["y"] = data.x + data.x * data.treated
        experiment = cp.RegressionKink(
            data, formula="y ~ 1 + x + I(x*treated)", kink_point=0, model=model
        )
    return experiment.fit()


@pytest.mark.parametrize("group", ["prior", "posterior"])
def test_outcome_plots_select_target_without_changing_effect_results(
    outcome_experiment, group
):
    experiment = outcome_experiment
    stored = experiment._require_bundle(group)
    summary = experiment.effect_summary().table.copy()
    fig, _axes = experiment.plot(
        group=group, response_type="prediction", ci_prob=0.5, show=False
    )
    assert any(f"{group} predictive outcome" in note.get_text() for note in fig.texts)
    assert experiment._require_bundle(group) is stored
    pd.testing.assert_frame_equal(experiment.effect_summary().table, summary)
    plt.close(fig)
    fig, _axes = experiment.plot(
        group=group,
        response_type="expectation",
        show=False,
        show_response_annotation=False,
    )
    assert not any("Intervals:" in note.get_text() for note in fig.texts)
    plt.close(fig)


def test_piecewise_prediction_preserves_structural_effects_and_earliest_window(
    monkeypatch,
):
    data = pd.DataFrame({"t": np.arange(8), "y": np.arange(8) + 2.0})
    experiment = cp.PiecewiseITS(
        data, formula="y ~ 1 + step(t, 6) + step(t, 2)", model=LinearRegression()
    ).fit()
    stored = experiment.result

    def noisy(X, **kwargs):
        mean = experiment._model_backend.predict(X)
        noise = xr.DataArray(
            [-2.0, -1.0, 1.0, 2.0], dims="draw", coords={"draw": range(4)}
        )
        return mean.isel(draw=0, drop=True) + noise

    monkeypatch.setattr(experiment._model_backend, "predict_response", noisy)
    fitted, counterfactual = experiment._response_predictions(
        stored, response_type="prediction", group="posterior"
    )
    assert counterfactual.sel(obs_ind=experiment.datapost.index).sizes["obs_ind"] == 6
    assert fitted.sizes["draw"] == 4
    assert experiment.result is stored
    frame = experiment.get_plot_data(response_type="prediction")
    np.testing.assert_allclose(frame.effect, stored.impact_pre.mean(("chain", "draw")))


def test_its_predictive_export_and_render_share_aligned_draw_trajectories(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"t": range(8), "y": np.arange(8) + 10.0}, index=dates)
    experiment = cp.InterruptedTimeSeries(
        data, dates[4], formula="y ~ 1 + t", model=LinearRegression()
    ).fit()
    stored = experiment.result
    baseline = experiment.get_plot_data()

    def noisy(X, **kwargs):
        mean = experiment._model_backend.predict(X).isel(draw=0, drop=True)
        noise = xr.DataArray(
            np.arange(-4, 5, dtype=float), dims="draw", coords={"draw": range(9)}
        )
        signs = xr.DataArray(
            (-1.0) ** np.arange(len(X)), dims="obs_ind", coords={"obs_ind": X.obs_ind}
        )
        return (mean + noise * signs).transpose(
            "chain", "draw", "obs_ind", "treated_units"
        )

    monkeypatch.setattr(experiment._model_backend, "predict_response", noisy)
    selected = experiment._bundle_for_response(
        stored, response_type="prediction", group="posterior"
    )
    xr.testing.assert_equal(
        selected.impact_post,
        (experiment.post_design["y"] - selected.predictions_post).squeeze(
            "treated_units"
        ),
    )
    xr.testing.assert_equal(
        selected.impact_post_cumulative, selected.impact_post.cumsum("obs_ind")
    )
    # Opposite noise in adjacent periods cancels within each draw. Summing marginal endpoints would leave a false interval.
    assert (
        float(selected.impact_post_cumulative.isel(obs_ind=-1).std("draw").max())
        < 1e-12
    )
    assert float(selected.impact_post.std("draw").min()) > 0
    exported = experiment.get_plot_data(response_type="prediction", hdi_prob=0.5)
    assert exported.index.equals(dates.rename("obs_ind"))
    assert exported.filter(like="hdi").notna().all().all()
    expected = get_hdi_to_df(selected.impact_post, hdi_prob=0.5)
    np.testing.assert_allclose(
        exported.loc[dates[4] :, ["impact_hdi_lower_50", "impact_hdi_upper_50"]],
        expected,
    )
    from causalpy.experiments import interrupted_time_series as module

    plotted = []
    original = module.plot_posterior_over_x

    def capture(x, draws, *args, **kwargs):
        plotted.append(draws)
        return original(x, draws, *args, **kwargs)

    monkeypatch.setattr(module, "plot_posterior_over_x", capture)
    fig, axes = experiment.plot(response_type="prediction", ci_prob=0.5, show=False)
    assert "Predictive" in axes[1].get_ylabel()
    assert any(
        np.array_equal(draws.values, selected.impact_post.values) for draws in plotted
    )
    assert any(
        np.array_equal(draws.values, selected.impact_post_cumulative.values)
        for draws in plotted
    )
    assert experiment.result is stored
    pd.testing.assert_frame_equal(experiment.get_plot_data(), baseline)
    plt.close(fig)


@pytest.fixture
def staggered_experiment():
    data = generate_staggered_did_data(
        n_units=6, n_time_periods=6, treatment_cohorts={3: 2}, seed=42
    )
    return cp.StaggeredDifferenceInDifferences(
        data,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        treatment_time_variable_name="treatment_time",
        model=LinearRegression(),
    ).fit()


def test_staggered_predictive_exports_match_both_views_and_preserve_att(
    staggered_experiment, monkeypatch
):
    experiment = staggered_experiment
    stored = experiment.result
    baseline = experiment.get_plot_data()

    def noisy(X, **kwargs):
        mean = experiment._model_backend.predict(X).isel(draw=0, drop=True)
        noise = xr.DataArray(
            np.arange(-4, 5, dtype=float), dims="draw", coords={"draw": range(9)}
        )
        return (mean + noise).transpose("chain", "draw", "obs_ind", "treated_units")

    monkeypatch.setattr(experiment._model_backend, "predict_response", noisy)
    # Keep the canonical container draw-based, so both event and cohort renderers use interval bars.
    mean = noisy(xr.DataArray(experiment.X_full, dims=("obs_ind", "coeffs")))
    experiment._assign_bundle("posterior", replace(stored, y_pred=mean))
    stored = experiment.result
    exported = experiment.get_plot_data(response_type="prediction", hdi_prob=0.5)
    assert exported.attrs["quantity"] == "predictive difference"
    assert exported.attrs["interval_kind"] == "eti"
    for view in (experiment.plot, experiment.plot_group_time):
        fig, axes = view(response_type="prediction", hdi_prob=0.5, show=False)
        assert "Predictive difference" in axes[0].get_ylabel()
        assert "50% ETI" in fig.texts[0].get_text()
        plt.close(fig)
    expected = experiment._bundle_for_response(
        stored, response_type="prediction", group="posterior", hdi_prob=0.5
    ).att_event_time
    pd.testing.assert_frame_equal(exported, expected)
    assert experiment.result is stored
    pd.testing.assert_frame_equal(experiment.get_plot_data(), baseline)


def test_panel_export_selects_noisy_outcomes_and_coefficient_plot_rejects_them(
    monkeypatch,
):
    data = pd.DataFrame(
        {
            "unit": np.repeat(["a", "b"], 4),
            "t": list(range(4)) * 2,
            "y": np.arange(8, dtype=float),
        }
    )
    experiment = cp.PanelRegression(
        data,
        formula="y ~ 1 + t + C(unit)",
        unit_fe_variable="unit",
        time_fe_variable="t",
        model=LinearRegression(),
    ).fit()
    baseline = experiment.get_plot_data()
    with pytest.raises(ValueError, match="no observation-noise draws"):
        experiment.get_plot_data(response_type="prediction")
    with pytest.raises(ValueError, match="shows coefficients"):
        experiment.plot(response_type="prediction")

    def selected(X, **kwargs):
        mean = experiment._model_backend.predict(X).isel(draw=0, drop=True)
        noise = xr.DataArray([-2.0, 2.0], dims="draw", coords={"draw": [0, 1]})
        return (mean + noise).transpose("chain", "draw", "obs_ind", "treated_units")

    monkeypatch.setattr(experiment._model_backend, "predict_response", selected)
    predictive = experiment.get_plot_data(response_type="prediction")
    np.testing.assert_allclose(predictive.y_fitted, baseline.y_fitted, atol=1e-12)
    np.testing.assert_allclose(
        predictive.y_fitted_upper - predictive.y_fitted_lower, 3.8
    )


@pytest.mark.parametrize(
    "experiment_class,message",
    [
        (cp.SyntheticDifferenceInDifferences, "models weights"),
        (InteractiveFixedEffectsSyntheticControl, "expectation trajectories only"),
    ],
)
def test_outcome_models_without_noise_reject_prediction_explicitly(
    experiment_class, message
):
    experiment = experiment_class.__new__(experiment_class)
    with pytest.raises(ValueError, match=message):
        experiment.plot(response_type="prediction")


def test_forecast_adapter_selects_outcomes_and_rejects_prior():
    draws = xr.DataArray(
        np.arange(8, dtype=float).reshape(1, 2, 2, 2),
        dims=("chain", "draw", "obs_ind", "treated_units"),
    )
    tree = xr.DataTree()
    tree["posterior_predictive"] = xr.Dataset({"mu": draws, "y_hat": draws + 2})

    class Model:
        def predict(self, **kwargs):
            return tree

    adapter = PyMCForecastAdapter.__new__(PyMCForecastAdapter)
    adapter._model = Model()
    xr.testing.assert_equal(
        adapter.predict_response(None, quantity="y_hat"),
        tree["posterior_predictive"]["y_hat"],
    )
    with pytest.raises(cp.GroupNotSampledException):
        adapter.predict_response(None, quantity="y_hat", group="prior")


@pytest.mark.parametrize("group", ["prior", "posterior"])
def test_piecewise_uses_genuine_counterfactual_outcomes_in_plots_and_export(
    monkeypatch, group, mock_pymc_sample
):
    from causalpy.experiments import piecewise_its as module

    data = pd.DataFrame({"t": range(8), "y": [1, 1, 5, 5, 5, 5, 9, 9]})
    experiment = cp.PiecewiseITS(
        data,
        formula="y ~ 1 + step(t, 6) + step(t, 2)",
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={"draws": 20, "tune": 20, "chains": 2, "cores": 1}
        ),
    ).fit()
    stored = experiment._require_bundle(group)

    def discrete(X, **kwargs):
        # Distinct factual/counterfactual laws expose subtraction of an expected effect from factual noise.
        is_counterfactual = not np.asarray(X)[:, experiment._interruption_cols].any()
        values = np.full((1, 8, len(X), 1), int(is_counterfactual), dtype=float)
        return xr.DataArray(
            values,
            dims=("chain", "draw", "obs_ind", "treated_units"),
            coords={"obs_ind": X.obs_ind, "treated_units": ["unit_0"]},
        )

    monkeypatch.setattr(experiment._model_backend, "predict_response", discrete)
    exported = experiment.get_plot_data(
        group=group, response_type="prediction", hdi_prob=0.5
    )
    np.testing.assert_array_equal(exported.counterfactual, 1)
    np.testing.assert_array_equal(exported.counterfactual_hdi_lower_50, 1)
    np.testing.assert_array_equal(exported.counterfactual_hdi_upper_50, 1)
    captured = []
    original = module.plot_posterior_over_x

    def capture(x, draws, **kwargs):
        captured.append(draws.copy())
        return original(x, draws, **kwargs)

    monkeypatch.setattr(module, "plot_posterior_over_x", capture)
    fig, _axes = experiment.plot(
        group=group, response_type="prediction", ci_prob=0.5, show=False
    )
    np.testing.assert_array_equal(captured[1], 1)
    assert experiment._require_bundle(group) is stored
    plt.close(fig)


def test_staggered_predictive_interval_can_exclude_its_mean(
    staggered_experiment, monkeypatch
):
    experiment = staggered_experiment
    monkeypatch.setattr(
        experiment, "_observed_outcome", pd.Series(0.0, index=experiment.data.index)
    )

    def rare(X, **kwargs):
        values = np.zeros((1, 100, len(X), 1))
        values[:, -1] = 1
        return xr.DataArray(
            values,
            dims=("chain", "draw", "obs_ind", "treated_units"),
            coords={"obs_ind": X.obs_ind, "treated_units": ["unit_0"]},
        )

    monkeypatch.setattr(experiment._model_backend, "predict_response", rare)
    frame = experiment.get_plot_data(response_type="prediction")
    assert (frame.att < frame.att_lower).all()
    fig, axes = experiment.plot(response_type="prediction", show=False)
    # Vertical interval segments stay at zero while the separately drawn mean is negative.
    segments = [
        segment
        for collection in axes[0].collections
        if hasattr(collection, "get_segments")
        for segment in collection.get_segments()
        if len(segment)
    ]
    assert segments
    assert all(np.allclose(segment[:, 1], 0) for segment in segments)
    assert any(
        np.any(line.get_ydata() < 0) for line in axes[0].lines if len(line.get_ydata())
    )
    plt.close(fig)


def test_equal_poisson_samples_do_not_disable_observation_noise():
    from causalpy.experiments.model_adapter import PyMCModelAdapter, build_coords
    from causalpy.tests.test_prediction_contract import PoissonLogLinkModel

    X = xr.DataArray(
        np.zeros((2, 1)),
        dims=("obs_ind", "coeffs"),
        coords={"obs_ind": [0, 1], "coeffs": ["Intercept"]},
    )
    y = xr.DataArray(
        np.ones((2, 1)),
        dims=("obs_ind", "treated_units"),
        coords={"obs_ind": [0, 1], "treated_units": ["unit_0"]},
    )
    model = PoissonLogLinkModel(sample_kwargs={"random_seed": 0})
    model.build(X, y, build_coords(["Intercept"], 2))
    model.sample_prior_predictive(draws=2, random_seed=0)
    draws = PyMCModelAdapter(model).predict_response(
        X.isel(obs_ind=[0]), quantity="y_hat", group="prior"
    )
    # Poisson(1) has variance one, even when this finite draw happens to equal its mean.
    np.testing.assert_array_equal(draws, 1)
    assert not cp.pymc_models.StateSpaceTimeSeries.supports_observation_noise


@pytest.mark.parametrize("ci_kind", ["hdi", "eti"])
def test_singleton_marker_renders_requested_interval_endpoints(ci_kind):
    from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
    from causalpy.plot_utils import _interval_bound_values

    draws = xr.DataArray(
        np.array([0, 0, 0, 0, 2, 3, 5, 9]).reshape(1, 8, 1),
        dims=("chain", "draw", "obs_ind"),
    )
    expected = _interval_bound_values([0], draws, ci_prob=0.5, ci_kind=ci_kind)
    fig, ax = plt.subplots()
    errorbar = InterruptedTimeSeries._draw_singleton_hdi_marker(
        ax, [0], draws, color="C0", hdi_prob=0.5, ci_kind=ci_kind
    )
    segment = errorbar.lines[2][0].get_segments()[0]
    np.testing.assert_allclose(segment[:, 1], np.array(expected).ravel())
    plt.close(fig)
