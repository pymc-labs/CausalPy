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
"""Contract tests for the recursive forecasting backend through ITS."""

import builtins

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.linear_model import Ridge

import causalpy as cp
from causalpy.skl_models import SkforecastModel


def test_skforecast_missing_extra(monkeypatch):
    import_module = builtins.__import__

    def without_skforecast(name, *args, **kwargs):
        if name == "skforecast.recursive":
            raise ModuleNotFoundError("No module named 'skforecast'", name="skforecast")
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_skforecast)
    with pytest.raises(ImportError, match=r"causalpy\[skforecast\]"):
        SkforecastModel(object(), calibration_size=5)


@pytest.fixture
def forecaster_class():
    return pytest.importorskip("skforecast.recursive").ForecasterRecursive


@pytest.mark.parametrize("calendar", [False, True])
def test_native_recursive_paths_and_canonical_result(forecaster_class, calendar):
    rng = np.random.default_rng(12)
    n_train, n_pre, n_obs = 26, 33, 43
    index = (
        pd.date_range("2020-01-01", periods=n_obs, freq="MS")
        if calendar
        else pd.RangeIndex(5, 5 + n_obs * 3, 3)
    )
    input_index = pd.DatetimeIndex(index.to_numpy()) if calendar else index
    data = pd.DataFrame(
        {"y": rng.normal(size=n_obs), "t": np.arange(n_obs)}, index=input_index
    )
    data["season"] = np.arange(n_obs) % 3
    formula = "y ~ 1 + center(t) + C(season)" if calendar else "y ~ 1"
    exog = (
        pd.DataFrame(
            {
                "C(season)[T.1]": (data.season == 1).astype(float),
                "C(season)[T.2]": (data.season == 2).astype(float),
                "center(t)": data.t - data.t.iloc[:n_train].mean(),
            },
            index=index,
        )
        if calendar
        else None
    )

    def rows(start, stop):
        return None if exog is None else exog.iloc[start:stop]

    def make_forecaster():
        return forecaster_class(estimator=Ridge(), lags=[1, 3])

    native = make_forecaster()
    native_y = data.y.set_axis(index)
    native.fit(native_y.iloc[:n_train], exog=rows(0, n_train))
    tail_predictions = pd.concat(
        [
            native.predict(1, last_window=native_y.iloc[i - 3 : i], exog=rows(i, i + 1))
            for i in range(n_train, n_pre)
        ]
    )
    native.set_out_sample_residuals(native_y.iloc[n_train:n_pre], tail_predictions)
    expected = native.predict_bootstrapping(
        n_obs - n_pre,
        last_window=native_y.iloc[n_pre - 3 : n_pre],
        exog=rows(n_pre, n_obs),
        n_boot=40,
        random_state=43,
        use_in_sample_residuals=False,
        use_binned_residuals=False,
    )
    stale_window = native.predict_bootstrapping(
        n_obs - n_pre,
        last_window=native_y.iloc[n_train - 3 : n_train],
        exog=rows(n_train, n_train + n_obs - n_pre),
        n_boot=40,
        random_state=43,
        use_in_sample_residuals=False,
        use_binned_residuals=False,
    )
    assert not np.allclose(expected.to_numpy(), stale_window.to_numpy())

    template = make_forecaster()
    model = SkforecastModel(
        template, calibration_size=n_pre - n_train, n_boot=40, random_state=43
    )
    exp = cp.InterruptedTimeSeries(data, index[n_pre], formula, model=model).build()
    assert exp._model_backend.is_built
    assert not exp.is_fitted
    assert not template.is_fitted
    exp.fit()
    result = exp.result
    paths = result.counterfactual_paths
    assert paths.dims == ("simulation", "obs_ind", "treated_units")
    assert paths.obs_ind.to_index().equals(index[n_pre:])
    assert paths.treated_units.values.tolist() == ["unit_0"]
    np.testing.assert_array_equal(paths.values[:, :, 0], expected.to_numpy().T)
    prediction = result.predictions_post
    assert prediction.dims == ("chain", "draw", "obs_ind", "treated_units")
    assert prediction.sizes["chain"] == prediction.sizes["draw"] == 1
    xr.testing.assert_equal(
        prediction.isel(chain=0, draw=0, drop=True), paths.mean("simulation")
    )
    assert result.predictions_pre.obs_ind.to_index().equals(index[:n_pre])
    np.testing.assert_array_equal(
        result.predictions_pre.values.ravel()[n_train:], tail_predictions.to_numpy()
    )
    assert np.isnan(result.predictions_pre.values.ravel()[:3]).all()
    assert np.isnan(result.impact_pre.values.ravel()[:3]).all()

    shifted = data.copy()
    shifted.loc[index[n_pre:], "y"] += 10
    comparison = cp.InterruptedTimeSeries(
        shifted, index[n_pre], formula, model=model, treatment_end_time=index[n_pre + 2]
    ).fit()
    assert not template.is_fitted
    xr.testing.assert_identical(paths, comparison.result.counterfactual_paths)
    xr.testing.assert_allclose(comparison.result.impact_post, result.impact_post + 10)
    assert exp.result.counterfactual_paths is paths


@pytest.fixture
def series():
    rng = np.random.default_rng(17)
    values = np.full(90, 20.0)
    for i in range(1, len(values)):
        values[i] = 20 + 0.85 * (values[i - 1] - 20) + rng.normal()
    return pd.DataFrame({"y": values, "t": np.arange(len(values))})


@pytest.fixture
def forecast_model(forecaster_class):
    return SkforecastModel(
        forecaster_class(estimator=Ridge(), lags=2), calibration_size=20, n_boot=300
    )


@pytest.mark.parametrize("prob", [0.8, 0.95])
def test_path_intervals_agree_across_plot_data_summary_persistence_and_report(
    series, forecast_model, prob, monkeypatch
):
    from matplotlib import pyplot as plt

    from causalpy.pipeline import PipelineContext

    start, end, stop = 66, 76, 83
    series.loc[start : end - 1, "y"] += 15
    series.loc[end:, "y"] += 7
    exp = cp.InterruptedTimeSeries(
        series, start, "y ~ 1", model=forecast_model, treatment_end_time=end
    ).fit()

    def reject_resampling(*args, **kwargs):
        pytest.fail("Reading a fitted result must reuse the stored simulations")

    monkeypatch.setattr(
        type(forecast_model.forecaster), "predict_bootstrapping", reject_resampling
    )
    paths = exp.result.counterfactual_paths.values[:, :, 0]
    impact = series.y.iloc[start:].to_numpy() - paths

    def interval(values):
        return np.quantile(values, [(1 - prob) / 2, (1 + prob) / 2], axis=0)

    plot_data = exp.get_plot_data(hdi_prob=prob)
    fig, axes = exp.plot(ci_prob=prob, show=False)
    for ax, (name, samples) in zip(
        axes,
        {
            "pred": paths,
            "impact": impact,
            "cumulative_impact": impact.cumsum(axis=1),
        }.items(),
        strict=True,
    ):
        lower, upper = interval(samples)
        cols = [
            f"{name}_pi_{bound}_{prob:.0%}".rstrip("%") for bound in ("lower", "upper")
        ]
        assert plot_data.loc[: start - 1, cols].isna().all().all()
        np.testing.assert_allclose(
            plot_data.loc[start:, cols].to_numpy().T, [lower, upper]
        )
        vertices = ax.collections[-1].get_paths()[0].vertices
        for x, lo, hi in zip(series.index[start:], lower, upper, strict=True):
            y_at_x = vertices[vertices[:, 0] == x, 1]
            assert y_at_x.min() == pytest.approx(lo)
            assert y_at_x.max() == pytest.approx(hi)
    plt.close(fig)
    low, high = interval(paths)
    np.testing.assert_allclose(
        plot_data.loc[start:, f"impact_pi_lower_{prob * 100:.0f}"],
        series.y.loc[start:] - high,
    )
    np.testing.assert_allclose(
        plot_data.loc[start:, f"impact_pi_upper_{prob * 100:.0f}"],
        series.y.loc[start:] - low,
    )

    summary = exp.effect_summary(alpha=1 - prob, period="post", window=(end + 1, stop))
    bounds = ["pi_lower", "pi_upper"]
    window_impact = impact[:, end + 1 - start : stop + 1 - start]
    window_paths = paths[:, end + 1 - start : stop + 1 - start]
    for row, values, lift in (
        (
            "average",
            window_impact.mean(axis=1),
            (100 * window_impact / window_paths).mean(axis=1),
        ),
        (
            "cumulative",
            window_impact.sum(axis=1),
            100 * window_impact.sum(axis=1) / window_paths.sum(axis=1),
        ),
    ):
        np.testing.assert_allclose(summary.table.loc[row, bounds], interval(values))
        assert summary.table.loc[row, "mean"] == pytest.approx(values.mean())
        np.testing.assert_allclose(
            summary.table.loc[row, ["relative_pi_lower", "relative_pi_upper"]],
            interval(lift),
        )
        assert summary.table.loc[row, "relative_mean"] == pytest.approx(lift.mean())
    assert not np.allclose(
        summary.table.loc["cumulative", bounds], interval(window_impact).sum(axis=1)
    )
    assert list(
        exp.effect_summary(
            period="intervention",
            window=(start, end - 1),
            cumulative=False,
            relative=False,
        ).table.columns
    ) == ["mean", *bounds]

    comparison = exp.effect_summary(period="comparison", alpha=1 - prob)
    during = impact[:, : end - start].mean(axis=1)
    after = impact[:, end - start :].mean(axis=1)
    ratio = after / during
    ratio_cols = ["persistence_pi_lower", "persistence_pi_upper"]
    np.testing.assert_allclose(
        comparison.table.loc["post_intervention", ratio_cols], 100 * interval(ratio)
    )
    persistence = exp.analyze_persistence(hdi_prob=prob)
    assert persistence["persistence_ratio"] == pytest.approx(ratio.mean())
    np.testing.assert_allclose(
        [
            persistence["persistence_ratio_pi_lower"],
            persistence["persistence_ratio_pi_upper"],
        ],
        interval(ratio),
    )
    np.testing.assert_allclose(
        [
            persistence["total_effect_post_pi_lower"],
            persistence["total_effect_post_pi_upper"],
        ],
        interval(impact[:, end - start :].sum(axis=1)),
    )
    without_options = exp.effect_summary(
        period="comparison", relative=False, cumulative=False
    )
    assert list(without_options.table.columns) == ["mean", *bounds]

    def report_plot(*, ci_prob):
        assert ci_prob == pytest.approx(prob)
        return fig, axes

    monkeypatch.setattr(exp, "plot", report_plot)
    context = PipelineContext(data=series, experiment=exp, effect_summary=summary)
    report = cp.GenerateReport().run(context).report
    assert f"{prob:.0%} bootstrap prediction intervals for realized impact" in report
    assert "pi_lower" in report and "cumulative" in report
    assert "exchangeable innovations" in report and "data:image/png" in report
    assert f"{stop - end} observations from {end + 1} through {stop}" in report
    assert all(key not in report for key in ("p_value", "hdi_lower", "ROPE"))


def test_shifted_outcomes_shift_intervals_and_window_total(series, forecast_model):
    start, stop = 70, 77
    shift = np.arange(len(series) - start) + 3
    shifted = series.copy()
    shifted.loc[start:, "y"] += shift
    a = cp.InterruptedTimeSeries(series, start, "y ~ 0", model=forecast_model).fit()
    b = cp.InterruptedTimeSeries(
        shifted, start, "y ~ 0", model=forecast_model, treatment_end_time=stop
    ).fit()
    a_data, b_data = a.get_plot_data(hdi_prob=0.9), b.get_plot_data(hdi_prob=0.9)
    for bound in ("lower", "upper"):
        column = f"pred_pi_{bound}_90"
        pd.testing.assert_series_equal(a_data[column], b_data[column])
        column = f"impact_pi_{bound}_90"
        np.testing.assert_allclose(
            b_data.loc[start:, column] - a_data.loc[start:, column], shift
        )
    a_table = a.effect_summary(window=(start, stop - 1)).table
    b_table = b.effect_summary(period="intervention").table
    np.testing.assert_allclose(
        b_table.loc["cumulative", ["mean", "pi_lower", "pi_upper"]]
        - a_table.loc["cumulative", ["mean", "pi_lower", "pi_upper"]],
        shift[: stop - start].sum(),
    )


def test_singleton_prediction_intervals_are_visible(series, forecast_model):
    from matplotlib import pyplot as plt

    exp = cp.InterruptedTimeSeries(series, 89, "y ~ 1", model=forecast_model).fit()
    data = exp.get_plot_data(hdi_prob=0.85).iloc[-1]
    fig, axes = exp.plot(ci_prob=0.85, show=False)
    for ax, name in zip(axes, ("pred", "impact", "cumulative_impact"), strict=True):
        expected = [data[f"{name}_pi_lower_85"], data[f"{name}_pi_upper_85"]]
        segment = ax.collections[-1].get_segments()[0]
        np.testing.assert_allclose(segment[:, 1], expected)
        assert expected[1] > expected[0]
    plt.close(fig)


def test_swapping_models_relearns_preprocessing_and_invalidates_results(
    series, forecast_model
):
    from sklearn.linear_model import LinearRegression

    start = 70
    exp = cp.InterruptedTimeSeries(
        series, start, "y ~ 1 + center(t)", model=LinearRegression(fit_intercept=False)
    ).fit()
    assert exp.pre_design.X.sel(coeffs="center(t)").values[0] == -(start - 1) / 2
    exp.model = forecast_model
    assert not exp.is_fitted and exp.idata is None
    assert exp.pre_design.X.sel(coeffs="center(t)").values[0] == -(start - 20 - 1) / 2
    exp.fit()
    paths = exp.result.counterfactual_paths
    with pytest.raises(ValueError, match="nonempty training prefix"):
        exp.model = SkforecastModel(forecast_model.forecaster, calibration_size=start)
    assert exp.result.counterfactual_paths is paths
    exp.model = SkforecastModel(forecast_model.forecaster, calibration_size=10)
    assert not exp.is_fitted
    assert exp.pre_design.X.sel(coeffs="center(t)").values[0] == -(start - 10 - 1) / 2
    exp.fit()
    assert not np.array_equal(exp.result.counterfactual_paths, paths)
    exp.model = LinearRegression(fit_intercept=False)
    assert not exp.is_fitted
    assert exp.pre_design.X.sel(coeffs="center(t)").values[0] == -(start - 1) / 2
    assert exp.fit().result.counterfactual_paths is None


@pytest.mark.parametrize(
    ("drop", "error"),
    [
        (30, "regular datetime or integer grid"),
        (80, "regular datetime or integer grid"),
    ],
)
def test_forecast_rejects_gaps_without_renumbering(series, forecast_model, drop, error):
    from causalpy.custom_exceptions import BadIndexException

    with pytest.raises(BadIndexException, match=error):
        cp.InterruptedTimeSeries(
            series.drop(index=drop), 70, "y ~ 1", model=forecast_model
        ).fit()


def test_forecast_requires_history_controls_and_supported_template(
    series, forecaster_class
):
    from skforecast.direct import ForecasterDirect

    with pytest.raises(ValueError, match="requires.*ForecasterRecursive"):
        SkforecastModel(
            ForecasterDirect(estimator=Ridge(), steps=2, lags=2), calibration_size=4
        )
    with pytest.raises(ValueError, match="differentiation=None"):
        SkforecastModel(
            forecaster_class(estimator=Ridge(), lags=2, differentiation=1),
            calibration_size=4,
        )
    template = forecaster_class(estimator=Ridge(), lags=5)
    for controls, name in (
        ({"calibration_size": 1}, "calibration_size"),
        ({"calibration_size": 2, "n_boot": 1}, "n_boot"),
        ({"calibration_size": 2, "random_state": -1}, "random_state"),
    ):
        with pytest.raises(ValueError, match=name):
            SkforecastModel(template, **controls)
    with pytest.raises(ValueError, match="training prefix.*window_size"):
        cp.InterruptedTimeSeries(
            series, 8, "y ~ 1", model=SkforecastModel(template, 3)
        ).fit()
    template.fit(series.y)
    with pytest.raises(ValueError, match="unfitted"):
        SkforecastModel(template, 4)


def test_degenerate_calibration_and_zero_denominator_are_reported(
    forecaster_class, capsys
):
    from sklearn.dummy import DummyRegressor

    from causalpy.custom_exceptions import PriorPredictiveNotSupportedException

    data = pd.DataFrame({"y": np.zeros(40)})
    data.loc[35:, "y"] = 10
    model = SkforecastModel(
        forecaster_class(
            estimator=DummyRegressor(), lags=2, binner_kwargs={"n_bins": 2}
        ),
        calibration_size=5,
        n_boot=5,
    )
    exp = cp.InterruptedTimeSeries(
        data, 30, "y ~ 1", model=model, treatment_end_time=35
    )
    with pytest.raises(PriorPredictiveNotSupportedException):
        exp.sample_prior_predictive()
    with pytest.warns(UserWarning, match="calibration errors are degenerate"):
        exp.fit()
    with pytest.raises(NotImplementedError, match="coefficient table"):
        exp.print_coefficients()
    exp.summary()
    printed = capsys.readouterr().out
    assert "5 finite one-step errors" in printed
    assert "held-out one-step forecasts" in printed
    assert "degenerate" in printed
    summary = exp.effect_summary()
    assert summary.table["relative_mean"].isna().all()
    assert "unavailable (unstable denominator)" in summary.text
    assert np.isnan(exp.analyze_persistence()["persistence_ratio"])

    from causalpy.checks.persistence import PersistenceCheck
    from causalpy.pipeline import PipelineContext

    context = PipelineContext(data=data, experiment=exp, effect_summary=summary)
    check = PersistenceCheck().run(exp, context)
    context.sensitivity_results.append(check)
    ratio = check.table.set_index("metric").loc["persistence_ratio"]
    assert ratio[["value", "pi_lower", "pi_upper"]].isna().all()
    report = cp.GenerateReport(include_plots=False).run(context).report
    assert "persistence ratio is unavailable" in report
    assert "effect persistence ratio is nan" not in report


@pytest.mark.correctness
def test_recursive_bootstrap_covers_known_ar_realizations_and_totals(forecaster_class):
    rng = np.random.default_rng(20261)
    n_pre, horizon, injected_effect = 1200, 8, 4.0
    ar = 0.8
    history = np.zeros(n_pre)
    for i, error in enumerate(rng.normal(size=n_pre - 1), start=1):
        history[i] = ar * history[i - 1] + error

    n_future = 5000
    future = np.empty((n_future, horizon))
    state = np.full(n_future, history[-1])
    for i in range(horizon):
        state = ar * state + rng.normal(size=n_future)
        future[:, i] = state

    data = pd.DataFrame({"y": np.r_[history, future[0] + injected_effect]})
    model = SkforecastModel(
        forecaster_class(estimator=Ridge(alpha=0), lags=1),
        calibration_size=600,
        n_boot=4000,
    )
    exp = cp.InterruptedTimeSeries(data, n_pre, "y ~ 1", model=model).fit()
    paths = exp.result.counterfactual_paths.values[:, :, 0]

    for observed, samples, truth in (
        (future[:, -1] + injected_effect, paths[:, -1], injected_effect),
        (
            (future + injected_effect).sum(axis=1),
            paths.sum(axis=1),
            injected_effect * horizon,
        ),
    ):
        lower, upper = np.quantile(samples, [0.025, 0.975])
        coverage = np.mean((observed - upper <= truth) & (truth <= observed - lower))
        # Conditional binomial SE is about 0.003 at 5,000 futures. The wider
        # tolerance also allows estimation and empirical-tail error from one
        # 600-row training prefix and 600-row calibration window.
        assert 0.91 < coverage < 0.985


def test_forecast_rejects_unsupported_formula_and_nonfinite_data(
    series, forecast_model
):
    start = 70
    with pytest.raises(ValueError, match="single continuous outcome"):
        cp.InterruptedTimeSeries(series, start, "y + t ~ 1", model=forecast_model)
    with pytest.raises(ValueError, match="supported by InterruptedTimeSeries"):
        cp.SyntheticControl(series, start, ["t"], ["y"], model=forecast_model)
    for time, column, formula, error in (
        (start - 1, "y", "y ~ 1", "finite pre-intervention outcomes"),
        (start, "y", "y ~ 1", "finite post-intervention outcomes"),
        (start, "t", "y ~ 1 + t", "finite exogenous inputs"),
    ):
        bad_data = series.copy()
        bad_data[column] = bad_data[column].astype(float)
        bad_data.loc[time, column] = np.inf
        with pytest.raises(ValueError, match=error):
            cp.InterruptedTimeSeries(
                bad_data, start, formula, model=forecast_model
            ).fit()


def test_forecast_calibration_requires_two_finite_errors(
    series, forecast_model, monkeypatch
):
    cls = type(forecast_model.forecaster)
    native_predict = cls.predict

    def nonfinite_calibration(self, steps, last_window, **kwargs):
        predictions = native_predict(self, steps, last_window, **kwargs)
        if predictions.index[0] >= 50:
            predictions[:] = np.nan
        return predictions

    monkeypatch.setattr(cls, "predict", nonfinite_calibration)
    exp = cp.InterruptedTimeSeries(series, 70, "y ~ 1", model=forecast_model)
    with pytest.raises(TypeError, match="Set the bootstrap controls"):
        exp.fit(random_seed=5)
    with pytest.raises(ValueError, match="at least two finite calibration errors"):
        exp.fit()
    assert not exp.is_fitted


def test_forecast_comparison_validates_window_and_probability(series, forecast_model):
    exp = cp.InterruptedTimeSeries(
        series, 70, "y ~ 1", model=forecast_model, treatment_end_time=80
    ).fit()
    with pytest.raises(ValueError, match="requires observations in both periods"):
        exp.effect_summary(period="comparison", window=(70, 79))
    with pytest.raises(ValueError, match="Prediction interval probability"):
        exp.get_plot_data(hdi_prob=0)


@pytest.mark.parametrize("declared", [False, True])
def test_business_day_cadence_uses_full_axis(forecaster_class, declared):
    index = pd.bdate_range("2026-01-05", periods=12)
    if not declared:
        index = pd.DatetimeIndex(index, freq=None)
        assert index[:4].inferred_freq == "D" and index.inferred_freq == "B"
    data = pd.DataFrame(
        {"y": np.random.default_rng(32).normal(size=len(index))}, index=index
    )
    model = SkforecastModel(forecaster_class(estimator=Ridge(), lags=1), 2, n_boot=10)
    exp = cp.InterruptedTimeSeries(data, index[4], "y ~ 1", model=model).fit()
    assert exp.result.counterfactual_paths.obs_ind.to_index().equals(index[4:])


@pytest.mark.parametrize("origin", [25, 45])
def test_utc_anchored_grid_preserves_observation_timezone(
    series, forecast_model, origin
):
    utc = pd.date_range("2026-02-15", periods=len(series), freq="D", tz="UTC")
    local = utc.tz_convert("Europe/London")
    formula = "y ~ 1 + t"
    exp = cp.InterruptedTimeSeries(
        series.set_axis(local),
        local[origin],
        formula,
        model=forecast_model,
        treatment_end_time=local[origin + 10],
    ).fit()
    reference = cp.InterruptedTimeSeries(
        series.set_axis(utc),
        utc[origin],
        formula,
        model=forecast_model,
        treatment_end_time=utc[origin + 10],
    ).fit()
    paths = exp.result.counterfactual_paths
    assert paths.obs_ind.to_index().equals(local[origin:])
    np.testing.assert_allclose(
        paths.values, reference.result.counterfactual_paths.values
    )
    pd.testing.assert_frame_equal(
        exp.effect_summary(period="comparison").table,
        reference.effect_summary(period="comparison").table,
    )


def test_forecast_keeps_a_real_regressor_named_intercept(forecast_model):
    rng = np.random.default_rng(21)
    regressor = rng.normal(size=90)
    data = pd.DataFrame(
        {"y": 50 * regressor + rng.normal(size=90), "Intercept": regressor}
    )
    aliased = data.rename(columns={"Intercept": "indicator"})
    exp = cp.InterruptedTimeSeries(
        data, 70, "y ~ 0 + Intercept", model=forecast_model
    ).fit()
    reference = cp.InterruptedTimeSeries(
        aliased, 70, "y ~ 0 + indicator", model=forecast_model
    ).fit()
    omitted = cp.InterruptedTimeSeries(data, 70, "y ~ 0", model=forecast_model).fit()
    np.testing.assert_allclose(
        exp.result.counterfactual_paths.values,
        reference.result.counterfactual_paths.values,
    )
    assert not np.allclose(
        exp.result.counterfactual_paths.values,
        omitted.result.counterfactual_paths.values,
    )


def test_calibration_reporting_distinguishes_upstream_pool_cap(forecaster_class):
    size = 10_010
    data = pd.DataFrame({"y": np.random.default_rng(61).normal(size=size + 44)})
    model = SkforecastModel(forecaster_class(estimator=Ridge(), lags=1), size, n_boot=8)
    exp = cp.InterruptedTimeSeries(data, size + 40, "y ~ 1", model=model).fit()
    attrs = exp.result.counterfactual_paths.attrs
    assert attrs["calibration_count"] == size
    assert attrs["residual_pool_count"] == 10_000
    text = exp.effect_summary(relative=False).text
    assert f"measured {size} finite one-step errors" in text
    assert "sampled from a pool of 10000" in text
