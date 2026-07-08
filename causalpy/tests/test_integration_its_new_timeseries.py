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
import arviz as az
import numpy as np
import pandas as pd
import pytest
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
    assert isinstance(result.idata, az.InferenceData)

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
    season = 0.5 * np.sin(2 * np.pi * dates.dayofyear / 7)
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
    assert isinstance(result.idata, az.InferenceData)

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
        import pymc_extras.statespace.structural  # noqa: F401
    except ImportError:
        pytest.skip("pymc-extras is required for StateSpaceTimeSeries tests")

    # Create simple synthetic data
    rng = np.random.default_rng(seed=42)
    dates = pd.date_range(start="2020-01-01", periods=60, freq="D")
    trend = np.linspace(0, 1.0, len(dates))
    season = 0.5 * np.sin(2 * np.pi * dates.dayofyear / 7)
    noise = rng.normal(0, 0.1, len(dates))
    y = trend + season + noise

    # Split into train/test
    train_dates = dates[:50]
    test_dates = dates[50:]
    y_train = y[:50]

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
        mode="PyMC",
    )

    # Fit the model
    coords_train = {"datetime_index": train_dates}
    model.fit(X=None, y=y_train, coords=coords_train)

    # Test in-sample prediction (out_of_sample=False)
    pred_in_sample = model.predict(X=None, coords=coords_train, out_of_sample=False)
    assert pred_in_sample is not None
    assert "posterior_predictive" in pred_in_sample or "y_hat" in pred_in_sample

    # Test out-of-sample prediction (out_of_sample=True)
    coords_test = {"datetime_index": test_dates}
    pred_out_of_sample = model.predict(X=None, coords=coords_test, out_of_sample=True)
    assert pred_out_of_sample is not None
    # StateSpaceTimeSeries.predict returns xr.Dataset for out_of_sample
    assert "y_hat" in pred_out_of_sample or "forecast_observed" in pred_out_of_sample

    # Test score method
    score = model.score(X=None, y=y_train, coords=coords_train)
    assert isinstance(score, pd.Series)
    assert "r2" in score


@pytest.mark.integration
def test_state_space_custom_components():
    """Test StateSpaceTimeSeries custom component validation."""
    # Skip if pymc-extras is not available
    try:
        import pymc_extras.statespace.structural  # noqa: F401
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
        import pymc_extras.statespace.structural  # noqa: F401
    except ImportError:
        pytest.skip("pymc-extras is required for StateSpaceTimeSeries tests")

    rng = np.random.default_rng(seed=42)
    dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
    y = rng.normal(0, 1, len(dates))

    sample_kwargs = {"chains": 1, "draws": 10, "progressbar": False}

    model = cp.pymc_models.StateSpaceTimeSeries(
        level_order=2,
        seasonal_length=7,
        sample_kwargs=sample_kwargs,
        mode="PyMC",
    )

    # Test missing coords
    with pytest.raises(ValueError, match="coords must be provided"):
        model.fit(X=None, y=y, coords=None)

    # Test missing datetime_index in coords
    with pytest.raises(
        ValueError,
        match="coords must contain 'datetime_index' of type pandas.DatetimeIndex",
    ):
        model.fit(X=None, y=y, coords={"some_other_key": dates})

    # Test invalid datetime_index type
    with pytest.raises(
        ValueError,
        match="coords must contain 'datetime_index' of type pandas.DatetimeIndex",
    ):
        model.fit(X=None, y=y, coords={"datetime_index": np.arange(len(dates))})

    # Fit a model for predict error tests
    model2 = cp.pymc_models.StateSpaceTimeSeries(
        level_order=2,
        seasonal_length=7,
        sample_kwargs=sample_kwargs,
        mode="PyMC",
    )
    model2.fit(X=None, y=y, coords={"datetime_index": dates})

    # Test predict with out_of_sample=True but coords=None
    with pytest.raises(
        ValueError, match="coords must be provided for out-of-sample prediction"
    ):
        model2.predict(X=None, coords=None, out_of_sample=True)

    # Test predict with out_of_sample=True but invalid datetime_index
    with pytest.raises(
        ValueError,
        match="coords must contain 'datetime_index' for prediction period",
    ):
        model2.predict(
            X=None,
            coords={"datetime_index": np.arange(10)},
            out_of_sample=True,
        )

    # Test predict before fit
    unfitted_model = cp.pymc_models.StateSpaceTimeSeries(
        level_order=2,
        seasonal_length=7,
        sample_kwargs=sample_kwargs,
        mode="PyMC",
    )
    with pytest.raises(RuntimeError, match="Model must be fit before"):
        unfitted_model.predict(X=None, coords={"datetime_index": dates})
