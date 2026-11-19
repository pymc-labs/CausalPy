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


@pytest.mark.parametrize("prob", [0.29, 0.8, 0.95])
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


@pytest.mark.parametrize(
    ("level", "percent"), [(0.29, "29"), (0.9999, "99.99"), (1.0, "100")]
)
def test_prediction_interval_labels_match_requested_level(
    series, forecast_model, level, percent, capsys
):
    from matplotlib import pyplot as plt

    from causalpy.checks.persistence import PersistenceCheck
    from causalpy.pipeline import PipelineContext

    exp = cp.InterruptedTimeSeries(
        series, 66, "y ~ 1", model=forecast_model, treatment_end_time=76
    ).fit()
    fig, axes = exp.plot(ci_prob=level, show=False)
    _, labels = axes[0].get_legend_handles_labels()
    plt.close(fig)
    assert f"{percent}% bootstrap prediction interval" in labels

    plot_data = exp.get_plot_data(hdi_prob=level)
    assert {column for column in plot_data if "_pi_" in column} == {
        f"{name}_pi_{bound}_{percent}"
        for name in ("pred", "impact", "cumulative_impact")
        for bound in ("lower", "upper")
    }

    result = PersistenceCheck(hdi_prob=level).run(
        exp, PipelineContext(data=series, experiment=exp)
    )
    assert f"{percent}% bootstrap prediction intervals" in result.text
    assert f"{percent}% bootstrap prediction intervals" in capsys.readouterr().out
    summary = exp.effect_summary(alpha=1 - level)
    assert f"{percent}% bootstrap prediction intervals" in summary.text


@pytest.mark.parametrize("alpha", [-1e-17, 1.0])
def test_prediction_summary_rejects_alpha_outside_interval_domain(
    series, forecast_model, alpha
):
    exp = cp.InterruptedTimeSeries(series, 70, "y ~ 1", model=forecast_model).fit()
    with pytest.raises(ValueError, match="alpha must be in"):
        exp.effect_summary(alpha=alpha)


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


def test_local_calendar_uses_wall_time_across_dst(series, forecaster_class):
    local = pd.date_range(
        "2026-02-01 19:00", periods=len(series), freq="D", tz="America/New_York"
    )
    wall_clock = local.tz_localize(None)
    model = SkforecastModel(forecaster_class(estimator=Ridge(), lags=1), 10, n_boot=10)
    exp = cp.InterruptedTimeSeries(
        series.set_axis(local), local[55], "y ~ 1 + t", model=model
    ).fit()
    reference = cp.InterruptedTimeSeries(
        series.set_axis(wall_clock), wall_clock[55], "y ~ 1 + t", model=model
    ).fit()
    paths = exp.result.counterfactual_paths
    assert paths.obs_ind.to_index().equals(local[55:])
    np.testing.assert_allclose(
        paths.values, reference.result.counterfactual_paths.values
    )
    assert exp.get_plot_data().index.equals(local)


def test_single_level_categorical_outcome_is_rejected(series, forecast_model):
    series["status"] = "closed"
    with pytest.raises(ValueError, match="single continuous outcome"):
        cp.InterruptedTimeSeries(series, 70, "status ~ 1", model=forecast_model)


@pytest.mark.parametrize("cadence", ["D", "W-MON", "MS", "24h"])
def test_native_calendar_features_keep_source_timezone(forecaster_class, cadence):
    from skforecast.preprocessing import CalendarFeatures

    index = pd.date_range("2026-02-15", periods=90, freq=cadence, tz="UTC").tz_convert(
        "America/New_York"
    )
    y = pd.Series(
        8 * index.hour + np.random.default_rng(13).normal(size=len(index)), index=index
    )
    train, pre = 48, 63
    template = forecaster_class(
        estimator=Ridge(),
        lags=1,
        calendar_features=CalendarFeatures(features=["hour"], encoding=None),
    )
    # Compare the configured hour feature and the exact simulated paths to the
    # upstream API on the same timestamp grid and observed history.
    native = forecaster_class(
        estimator=Ridge(),
        lags=1,
        calendar_features=CalendarFeatures(features=["hour"], encoding=None),
    )
    native.fit(y.iloc[:train])
    predicted = pd.concat(
        [
            native.predict(steps=1, last_window=y.iloc[i - 1 : i])
            for i in range(train, pre)
        ]
    )
    native.set_out_sample_residuals(y.iloc[train:pre], predicted)
    expected = native.predict_bootstrapping(
        steps=len(y) - pre,
        last_window=y.iloc[pre - 1 : pre],
        n_boot=20,
        random_state=55,
        use_in_sample_residuals=False,
        use_binned_residuals=False,
    )
    exp = cp.InterruptedTimeSeries(
        y.to_frame(name="y"),
        index[pre],
        "y ~ 1",
        model=SkforecastModel(template, pre - train, n_boot=20, random_state=55),
    ).fit()
    paths = exp.result.counterfactual_paths
    assert paths.obs_ind.to_index().equals(expected.index)
    np.testing.assert_allclose(paths.values[:, :, 0], expected.to_numpy().T)


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ 1 + y",
        "y ~ I(y.shift(1, fill_value=0))",
        'y ~ 1 + Q("y")',
        'Q("y") ~ y',
        'y ~ I(Q("y").shift(1, fill_value=0))',
        'y ~ Q(name="y")',
        'Q(name="y") ~ y',
        'y ~ Q("" + "y")',
    ],
)
def test_observed_outcome_is_rejected_as_forecast_predictor(
    series, forecast_model, formula
):
    with pytest.raises(
        ValueError, match="Configure outcome lags on ForecasterRecursive"
    ):
        cp.InterruptedTimeSeries(series, 70, formula, model=forecast_model)


@pytest.mark.parametrize("dtype", ["Int64", "UInt64", "int64[pyarrow]"])
def test_nullable_integer_time_column_uses_its_logical_grid(
    series, forecast_model, dtype
):
    grid = pd.RangeIndex(100, 100 + 2 * len(series), 2)
    data = series.assign(week=pd.array(grid, dtype=dtype))
    start = grid[70]
    exp = cp.InterruptedTimeSeries(
        data, start, "y ~ 1 + t", time_column="week", model=forecast_model
    ).fit()
    reference = cp.InterruptedTimeSeries(
        series.set_axis(grid), start, "y ~ 1 + t", model=forecast_model
    ).fit()
    paths = exp.result.counterfactual_paths
    assert paths.obs_ind.to_index().dtype == data.week.dtype
    np.testing.assert_array_equal(paths.obs_ind.to_index().to_numpy(), grid[70:])
    np.testing.assert_array_equal(
        paths.values, reference.result.counterfactual_paths.values
    )


@pytest.mark.parametrize("formula", ["y ~ 1", "y ~ 1 + t"])
def test_arrow_microsecond_time_column_preserves_labels(
    series, forecast_model, formula
):
    import pyarrow as pa

    grid = pd.date_range("2026-01-01", periods=len(series), freq="D")
    data = pa.table(
        {
            "date": pa.array(
                grid.to_numpy(dtype="datetime64[us]"), type=pa.timestamp("us")
            ),
            "y": series.y,
            "t": series.t,
        }
    )
    exp = cp.InterruptedTimeSeries(
        data, grid[70], formula, time_column="date", model=forecast_model
    ).fit()
    reference = cp.InterruptedTimeSeries(
        series.set_axis(grid), grid[70], formula, model=forecast_model
    ).fit()
    paths = exp.result.counterfactual_paths
    assert paths.obs_ind.to_index().equals(grid[70:].as_unit("us"))
    np.testing.assert_array_equal(
        paths.values, reference.result.counterfactual_paths.values
    )


@pytest.mark.parametrize("origin", [20, 42])
def test_nonexistent_native_wall_time_falls_back_to_elapsed_daily_grid(
    series, forecaster_class, origin
):
    utc = pd.date_range("2026-03-01 01:30", periods=len(series), freq="D", tz="UTC")
    local = utc.tz_convert("Europe/London")
    model = SkforecastModel(forecaster_class(estimator=Ridge(), lags=1), 8, n_boot=10)
    exp = cp.InterruptedTimeSeries(
        series.set_axis(local), local[origin], "y ~ 1 + t", model=model
    ).fit()
    reference = cp.InterruptedTimeSeries(
        series.set_axis(utc), utc[origin], "y ~ 1 + t", model=model
    ).fit()
    paths = exp.result.counterfactual_paths
    assert paths.obs_ind.to_index().equals(local[origin:])
    np.testing.assert_array_equal(
        paths.values, reference.result.counterfactual_paths.values
    )


@pytest.mark.parametrize("time_of_day", ["04:00", "05:00"])
def test_fallback_calendar_still_uses_observed_local_hour(
    forecaster_class, time_of_day
):
    from skforecast.preprocessing import CalendarFeatures

    index = pd.date_range(
        f"2019-01-01 {time_of_day}", periods=90, freq="MS", tz="UTC"
    ).tz_convert("America/New_York")
    local_hour = index.hour.to_numpy(dtype=float)
    data = pd.DataFrame(
        {
            "y": 30
            + 8 * local_hour
            + np.random.default_rng(186).normal(size=len(index)),
            "local_hour": local_hour,
        },
        index=index,
    )
    wrapper = SkforecastModel(
        forecaster_class(
            estimator=Ridge(),
            lags=1,
            calendar_features=CalendarFeatures(features=["hour"], encoding=None),
        ),
        calibration_size=15,
        n_boot=20,
    )
    manual = SkforecastModel(
        forecaster_class(estimator=Ridge(), lags=1), calibration_size=15, n_boot=20
    )
    exp = cp.InterruptedTimeSeries(data, index[63], "y ~ 1", model=wrapper).fit()
    reference = cp.InterruptedTimeSeries(
        data, index[63], "y ~ 0 + local_hour", model=manual
    ).fit()
    paths = exp.result.counterfactual_paths
    assert paths.obs_ind.to_index().equals(index[63:])
    np.testing.assert_allclose(
        paths.values, reference.result.counterfactual_paths.values
    )


def test_fallback_uses_observed_timestamps_for_training_weights(forecaster_class):
    utc = pd.date_range("2019-01-01 04:00", periods=90, freq="MS", tz="UTC")
    local = utc.tz_convert("America/New_York")
    values = np.random.default_rng(233).normal(size=len(utc))

    def local_weights(index):
        return np.where(index.day == 1, 1.0, 12.0)

    def utc_weights(index):
        return local_weights(index.tz_convert(local.tz))

    def fit(index, weight_func):
        data = pd.DataFrame({"y": values}, index=index)
        model = SkforecastModel(
            forecaster_class(estimator=Ridge(), lags=1, weight_func=weight_func),
            calibration_size=16,
            n_boot=25,
        )
        return cp.InterruptedTimeSeries(data, index[64], "y ~ 1", model=model).fit()

    result = fit(local, local_weights).result
    reference = fit(utc, utc_weights).result
    np.testing.assert_array_equal(
        result.counterfactual_paths.values, reference.counterfactual_paths.values
    )


def test_plot_data_clears_predictive_columns_when_refit_changes_origin(
    series, forecast_model
):
    initial = cp.InterruptedTimeSeries(
        series, 60, "y ~ 1 + t", model=forecast_model
    ).fit()
    initial_data = initial.get_plot_data(hdi_prob=0.8)
    generated = ["cumulative_impact"] + [
        column for column in initial_data if "_pi_" in column
    ]
    assert initial_data.loc[60:69, generated].notna().all().all()
    result = cp.InterruptedTimeSeries(
        initial_data, 70, "y ~ 1 + t", model=forecast_model
    ).fit()
    clean = cp.InterruptedTimeSeries(
        series, 70, "y ~ 1 + t", model=forecast_model
    ).fit()
    data = result.get_plot_data(hdi_prob=0.8)
    assert data.loc[:69, generated].isna().all().all()
    pd.testing.assert_frame_equal(
        data[generated], clean.get_plot_data(hdi_prob=0.8)[generated]
    )


def test_held_out_covariates_cannot_change_training_preprocessing(
    series, forecast_model
):
    start = 70
    split = start - forecast_model.calibration_size
    changed = series.copy()
    changed.loc[split : start - 1, "t"] += 20 * np.arange(1, start - split + 1)
    formula = "y ~ I(t - t.mean())"
    reference = cp.InterruptedTimeSeries(
        series, start, formula, model=forecast_model
    ).fit()
    exp = cp.InterruptedTimeSeries(changed, start, formula, model=forecast_model).fit()
    train = slice(0, split)
    learner_values = series.t.iloc[train].to_numpy(dtype=float)
    np.testing.assert_array_equal(
        exp.pre_design.X.isel(obs_ind=train).values[:, 0],
        learner_values - learner_values.mean(),
    )
    np.testing.assert_array_equal(
        reference.result.predictions_pre.isel(obs_ind=train),
        exp.result.predictions_pre.isel(obs_ind=train),
    )


@pytest.mark.parametrize(
    "formula", ["I(y-y.mean()) ~ 1", "np.log(y) ~ 1", "np.log(y) ~ y"]
)
def test_skforecast_requires_original_outcome_units(series, forecast_model, formula):
    with pytest.raises(ValueError, match="outcome column in its original units"):
        cp.InterruptedTimeSeries(series, 70, formula, model=forecast_model)


def test_quoted_outcome_with_native_transformer_stays_on_original_scale(
    series, forecaster_class
):
    from sklearn.preprocessing import StandardScaler

    model = SkforecastModel(
        forecaster_class(estimator=Ridge(), lags=2, transformer_y=StandardScaler()),
        calibration_size=20,
        n_boot=50,
    )
    named = series.rename(columns={"y": "observed result"})
    exp = cp.InterruptedTimeSeries(
        named, 70, 'Q(name="observed result") ~ 1', model=model
    ).fit()
    expected = cp.InterruptedTimeSeries(series, 70, "y ~ 1", model=model).fit()
    np.testing.assert_array_equal(
        exp.result.counterfactual_paths.values,
        expected.result.counterfactual_paths.values,
    )
    np.testing.assert_allclose(
        exp.result.impact_post.values.ravel(),
        series.y.iloc[70:].to_numpy() - exp.result.predictions_post.values.ravel(),
    )


def test_formula_namespace_does_not_include_experiment_locals(series, forecast_model):
    with pytest.raises(ValueError, match="name 'pre' is not defined"):
        cp.InterruptedTimeSeries(
            series,
            70,
            "y ~ I(pd.concat([pre, post]).loc[t.index, 'y'])",
            model=forecast_model,
        )


def test_explicit_formula_namespace_supports_stateful_and_numpy_terms(
    series, forecast_model
):
    split = 70 - forecast_model.calibration_size
    experiment = cp.InterruptedTimeSeries(
        series, 70, "y ~ center(t) + I(np.sqrt(t))", model=forecast_model
    )
    np.testing.assert_allclose(
        experiment.pre_design.X.sel(coeffs="center(t)").values,
        series.t.iloc[:70] - series.t.iloc[:split].mean(),
    )
    np.testing.assert_allclose(
        experiment.post_design.X.sel(coeffs="I(np.sqrt(t))").values,
        np.sqrt(series.t.iloc[70:]),
    )


@pytest.mark.parametrize("preserve_index", [True, False])
def test_native_exog_transformer_sees_observed_clock(forecaster_class, preserve_index):
    from sklearn.preprocessing import FunctionTransformer

    utc = pd.date_range("2019-01-01 04:00", periods=90, freq="MS", tz="UTC")
    local = utc.tz_convert("America/New_York")
    hour = local.hour.to_numpy(dtype=float)
    data = pd.DataFrame(
        {
            "y": 30 + 8 * hour + np.random.default_rng(233).normal(size=len(utc)),
            "hour": hour,
        },
        index=local,
    )

    def local_hour(frame):
        transformed = frame.assign(hour=frame.index.hour.astype(float))
        return transformed if preserve_index else transformed.reset_index(drop=True)

    def fit(transformer):
        model = SkforecastModel(
            forecaster_class(estimator=Ridge(), lags=1, transformer_exog=transformer),
            calibration_size=16,
            n_boot=25,
        )
        return cp.InterruptedTimeSeries(data, local[64], "y ~ hour", model=model).fit()

    native = fit(FunctionTransformer(local_hour))
    explicit = fit(None)
    xr.testing.assert_identical(
        native.result.counterfactual_paths, explicit.result.counterfactual_paths
    )


@pytest.mark.parametrize("labelled_output", [False, True])
def test_native_outcome_transformer_sees_original_name_and_clock(
    forecaster_class, labelled_output
):
    from sklearn.preprocessing import StandardScaler

    index = pd.date_range(
        "2019-01-01 04:00", periods=90, freq="MS", tz="UTC"
    ).tz_convert("America/New_York")
    outcome = "observed result"
    data = pd.DataFrame(
        {outcome: np.random.default_rng(237).normal(size=len(index))}, index=index
    )
    expected_train = index[:50]

    class NamedScaler(StandardScaler):
        def fit_transform(self, X, y=None, **fit_params):
            assert X.columns.tolist() == [outcome]
            assert X.index.equals(expected_train)
            scaled = super().fit_transform(X, y, **fit_params)
            return pd.DataFrame(
                scaled,
                columns=X.columns,
                index=X.index if labelled_output else None,
            )

    def fit(transformer):
        model = SkforecastModel(
            forecaster_class(estimator=Ridge(), lags=1, transformer_y=transformer),
            calibration_size=14,
            n_boot=25,
        )
        return cp.InterruptedTimeSeries(
            data, index[64], 'Q("observed result") ~ 1', model=model
        ).fit()

    xr.testing.assert_identical(
        fit(NamedScaler()).result.counterfactual_paths,
        fit(StandardScaler()).result.counterfactual_paths,
    )


def test_native_rolling_features_see_observed_clock(forecaster_class):
    from skforecast.preprocessing import RollingFeatures

    index = pd.date_range(
        "2019-01-01 04:00", periods=90, freq="MS", tz="UTC"
    ).tz_convert("America/New_York")
    data = pd.DataFrame(
        {"y": np.random.default_rng(330).normal(size=len(index))}, index=index
    )

    class ObservedRollingFeatures(RollingFeatures):
        def transform_batch(self, X):
            assert X.index.equals(index[:50])
            return super().transform_batch(X)

    def fit(features):
        model = SkforecastModel(
            forecaster_class(estimator=Ridge(), lags=None, window_features=features),
            calibration_size=14,
            n_boot=25,
        )
        return cp.InterruptedTimeSeries(data, index[64], "y ~ 1", model=model).fit()

    xr.testing.assert_identical(
        fit(
            ObservedRollingFeatures(stats="mean", window_sizes=2)
        ).result.counterfactual_paths,
        fit(RollingFeatures(stats="mean", window_sizes=2)).result.counterfactual_paths,
    )


@pytest.mark.parametrize("clock_fallback", [False, True])
@pytest.mark.parametrize(
    "preprocessor", ["transformer_exog", "transformer_y", "calendar_features"]
)
def test_native_preprocessing_rejects_reordered_observation_labels(
    forecaster_class, clock_fallback, preprocessor
):
    from skforecast.preprocessing import CalendarFeatures
    from sklearn.preprocessing import FunctionTransformer

    from causalpy.custom_exceptions import BadIndexException

    index = pd.date_range("2019-01-01 04:00", periods=90, freq="MS", tz="UTC")
    if clock_fallback:
        index = index.tz_convert("America/New_York")
    t = np.arange(len(index))
    data = pd.DataFrame({"y": 30 + 8 * t, "t": t}, index=index)

    class ReorderedCalendar(CalendarFeatures):
        def transform(self, X):
            return super().transform(X).iloc[::-1]

    def reverse_labelled(frame):
        return frame.iloc[::-1] if isinstance(frame, pd.DataFrame) else frame

    transformer = (
        ReorderedCalendar(features=["month"])
        if preprocessor == "calendar_features"
        else FunctionTransformer(reverse_labelled, check_inverse=False)
    )
    model = SkforecastModel(
        forecaster_class(estimator=Ridge(), lags=1, **{preprocessor: transformer}),
        calibration_size=16,
        n_boot=25,
    )
    formula = "y ~ t" if preprocessor == "transformer_exog" else "y ~ 1"
    experiment = cp.InterruptedTimeSeries(data, index[64], formula, model=model)
    with pytest.raises(
        BadIndexException, match="keep their observation labels in order"
    ):
        experiment.fit()
