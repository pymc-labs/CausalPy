#   Copyright 2025 - 2025 The PyMC Labs Developers
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
        import pymc_extras.statespace.structural  # noqa: F401
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
        mode="PyMC",
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
