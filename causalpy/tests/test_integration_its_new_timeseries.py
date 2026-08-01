#   Copyright 2025 - 2026 The PyMC Labs Developers
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
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt

import causalpy as cp


@pytest.mark.integration
def test_its_with_bsts_model():
    """InterruptedTimeSeries integration using BayesianBasisExpansionTimeSeries."""
    pytest.importorskip(
        "pymc_marketing", reason="pymc-marketing optional for default BSTS components"
    )
    # Prepare data
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
        .rename(columns={"y": "y"})
    )
    treatment_time = pd.to_datetime("2017-01-01")

    # Keep test fast
    sample_kwargs = {
        "chains": 1,
        "draws": 60,
        "tune": 30,
        "progressbar": False,
        "random_seed": 123,
    }

    model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        n_order=2, n_changepoints_trend=5, sample_kwargs=sample_kwargs
    )

    # Simple formula (intercept only) avoids exogenous regressors if desired
    # but we still pass it through patsy for consistency with the experiment
    result = cp.InterruptedTimeSeries(
        data=df[["y"]],
        treatment_time=treatment_time,
        formula="y ~ 1",
        model=model,
    )

    # Basic checks
    assert isinstance(result, cp.InterruptedTimeSeries)
    assert isinstance(result.idata, xr.DataTree)

    # Plot and plot data
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray)

    plot_data = result.get_plot_data()
    assert isinstance(plot_data, pd.DataFrame)
    expected_columns = {
        "prediction",
        "pred_hdi_lower_94",
        "pred_hdi_upper_94",
        "impact",
        "impact_hdi_lower_94",
        "impact_hdi_upper_94",
    }
    assert expected_columns.issubset(set(plot_data.columns))


@pytest.mark.integration
def test_its_with_state_space_model():
    """InterruptedTimeSeries integration using StateSpaceTimeSeries.

    Skips when pymc-extras is not installed.
    """
    # Skip if pymc-extras is not available
    try:
        from pymc_extras.statespace import structural  # noqa: F401
    except ImportError:
        pytest.skip("pymc-extras is required for StateSpaceTimeSeries tests")

    # Synthetic data: short daily series for speed
    rng = np.random.default_rng(seed=42)
    dates = pd.date_range(start="2020-01-01", periods=80, freq="D")
    trend = np.linspace(0, 1.0, len(dates))
    season = 0.5 * np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 7)
    noise = rng.normal(0, 0.2, len(dates))
    y = trend + season + noise
    df = pd.DataFrame({"y": y}, index=dates)

    treatment_time = dates[50]

    sample_kwargs = {
        "chains": 1,
        "draws": 40,
        "tune": 20,
        "progressbar": False,
        "random_seed": 7,
    }

    model = cp.pymc_models.StateSpaceTimeSeries(
        level_order=2,
        seasonal_length=7,
        sample_kwargs=sample_kwargs,
        mode="FAST_COMPILE",
    )

    result = cp.InterruptedTimeSeries(
        data=df[["y"]],
        treatment_time=treatment_time,
        formula="y ~ 1",
        model=model,
    )

    assert isinstance(result, cp.InterruptedTimeSeries)
    assert isinstance(result.idata, xr.DataTree)

    # In-sample predictions should be available
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray)

    # Plot data should include expected columns
    plot_data = result.get_plot_data()
    assert isinstance(plot_data, pd.DataFrame)
    expected_columns = {
        "prediction",
        "pred_hdi_lower_94",
        "pred_hdi_upper_94",
        "impact",
        "impact_hdi_lower_94",
        "impact_hdi_upper_94",
    }
    assert expected_columns.issubset(set(plot_data.columns))


@pytest.mark.integration
def test_state_space_predict_and_score():
    """Test StateSpaceTimeSeries predict and score methods directly."""
    # Skip if pymc-extras is not available
    try:
        from pymc_extras.statespace import structural  # noqa: F401
    except ImportError:
        pytest.skip("pymc-extras is required for StateSpaceTimeSeries tests")

    # Create simple synthetic data
    rng = np.random.default_rng(seed=42)
    dates = pd.date_range(start="2020-01-01", periods=60, freq="D")
    trend = np.linspace(0, 1.0, len(dates))
    season = 0.5 * np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 7)
    noise = rng.normal(0, 0.1, len(dates))
    y = trend + season + noise

    # Split into train/test
    train_dates = dates[:50]
    test_dates = dates[50:]
    y_train = xr.DataArray(
        y[:50, np.newaxis],
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": train_dates, "treated_units": ["unit_0"]},
    )
    X_train = xr.DataArray(
        np.zeros((len(train_dates), 0)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": train_dates, "coeffs": []},
    )
    X_test = xr.DataArray(
        np.zeros((len(test_dates), 0)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": test_dates, "coeffs": []},
    )

    sample_kwargs = {
        "chains": 1,
        "draws": 40,
        "tune": 20,
        "progressbar": False,
        "random_seed": 7,
    }

    model = cp.pymc_models.StateSpaceTimeSeries(
        level_order=2,
        seasonal_length=7,
        sample_kwargs=sample_kwargs,
        mode="FAST_COMPILE",
    )

    # Fit the model.
    model.fit(X=X_train, y=y_train)

    # Test in-sample prediction.
    pred_in_sample = model.predict(X=X_train, out_of_sample=False)
    assert isinstance(pred_in_sample, xr.DataTree)
    assert "posterior_predictive" in pred_in_sample
    in_sample_pp = pred_in_sample["posterior_predictive"].to_dataset()
    assert {"y_hat", "mu"} <= set(in_sample_pp.data_vars)
    np.testing.assert_array_equal(
        in_sample_pp.coords["obs_ind"].values, X_train.coords["obs_ind"].values
    )

    # Test out-of-sample prediction with the target datetime coordinates.
    pred_out_of_sample = model.predict(X=X_test, out_of_sample=True)
    assert isinstance(pred_out_of_sample, xr.DataTree)
    assert "posterior_predictive" in pred_out_of_sample
    posterior_predictive = pred_out_of_sample["posterior_predictive"].to_dataset()
    assert "y_hat" in posterior_predictive
    np.testing.assert_array_equal(
        posterior_predictive.coords["obs_ind"].values, X_test.coords["obs_ind"].values
    )

    score = model.score(X=X_train, y=y_train)
    assert isinstance(score, pd.Series)
    assert "unit_0_r2" in score.index
    assert "unit_0_r2_std" in score.index


@pytest.mark.integration
def test_state_space_custom_components():
    """Test StateSpaceTimeSeries custom component validation."""
    # Skip if pymc-extras is not available
    try:
        from pymc_extras.statespace import structural  # noqa: F401
    except ImportError:
        pytest.skip("pymc-extras is required for StateSpaceTimeSeries tests")

    class BadComponent:
        """Component without apply method"""

        pass

    sample_kwargs = {"chains": 1, "draws": 10, "progressbar": False}

    # Test invalid trend component
    with pytest.raises(
        ValueError,
        match="Custom trend_component must have an 'apply' method",
    ):
        cp.pymc_models.StateSpaceTimeSeries(
            trend_component=BadComponent(),
            sample_kwargs=sample_kwargs,
        )

    # Test invalid seasonality component
    with pytest.raises(
        ValueError,
        match="Custom seasonality_component must have an 'apply' method",
    ):
        cp.pymc_models.StateSpaceTimeSeries(
            seasonality_component=BadComponent(),
            sample_kwargs=sample_kwargs,
        )


@pytest.mark.integration
def test_state_space_error_conditions():
    """Test StateSpaceTimeSeries error handling."""
    # Skip if pymc-extras is not available
    try:
        from pymc_extras.statespace import structural  # noqa: F401
    except ImportError:
        pytest.skip("pymc-extras is required for StateSpaceTimeSeries tests")

    rng = np.random.default_rng(seed=42)
    dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
    y_values = rng.normal(0, 1, (len(dates), 1))
    y = xr.DataArray(
        y_values,
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": dates, "treated_units": ["unit_0"]},
    )

    sample_kwargs = {"chains": 1, "draws": 10, "tune": 10, "progressbar": False}

    model = cp.pymc_models.StateSpaceTimeSeries(
        level_order=2,
        seasonal_length=7,
        sample_kwargs=sample_kwargs,
        mode="FAST_COMPILE",
    )

    # y is required
    with pytest.raises(ValueError, match="y must be provided"):
        model.fit(X=None, y=None)

    # A treated_units dimension is required
    y_without_units = xr.DataArray(
        y_values[:, 0], dims=["obs_ind"], coords={"obs_ind": dates}
    )
    with pytest.raises(ValueError, match="requires a treated_units dimension"):
        model.fit(X=None, y=y_without_units)

    # Exactly one treated unit is supported
    y_two_units = xr.DataArray(
        rng.normal(0, 1, (len(dates), 2)),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": dates, "treated_units": ["unit_0", "unit_1"]},
    )
    with pytest.raises(ValueError, match="supports exactly one treated unit"):
        model.fit(X=None, y=y_two_units)

    # Non-datetime obs_ind with no datetime_index fallback
    y_integer_index = xr.DataArray(
        y_values,
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(len(dates)), "treated_units": ["unit_0"]},
    )
    with pytest.raises(ValueError, match="must contain datetime values"):
        model.fit(X=None, y=y_integer_index)

    # The datetime_index fallback must be a pd.DatetimeIndex
    with pytest.raises(ValueError, match="must be a pd.DatetimeIndex"):
        model.fit(
            X=None,
            y=y_integer_index,
            coords={"datetime_index": np.arange(len(dates))},
        )

    # Fit a model for the prediction error tests
    model2 = cp.pymc_models.StateSpaceTimeSeries(
        level_order=2,
        seasonal_length=7,
        sample_kwargs=sample_kwargs,
        mode="FAST_COMPILE",
    )
    model2.fit(X=None, y=y)

    # Out-of-sample prediction requires X carrying the forecast datetimes
    with pytest.raises(ValueError, match="X must be provided for out-of-sample"):
        model2.predict(X=None, out_of_sample=True)

    X_without_obs_ind = xr.DataArray(np.zeros((5, 0)), dims=["obs_ind", "coeffs"])
    with pytest.raises(ValueError, match="X must have 'obs_ind' coordinate"):
        model2.predict(X=X_without_obs_ind, out_of_sample=True)

    X_integer_index = xr.DataArray(
        np.zeros((5, 0)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": np.arange(5), "coeffs": []},
    )
    with pytest.raises(ValueError, match="must contain datetime values"):
        model2.predict(X=X_integer_index, out_of_sample=True)

    # Predict before fit
    unfitted_model = cp.pymc_models.StateSpaceTimeSeries(
        level_order=2,
        seasonal_length=7,
        sample_kwargs=sample_kwargs,
        mode="FAST_COMPILE",
    )
    with pytest.raises(RuntimeError, match="Model must be fit before"):
        unfitted_model.predict(X=None)
