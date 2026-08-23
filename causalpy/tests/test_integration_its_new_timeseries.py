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
    y = np.asarray(trend + season + noise)

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

    # Predict before fit raises
    unfitted_model = cp.pymc_models.StateSpaceTimeSeries(
        level_order=2,
        seasonal_length=7,
        sample_kwargs=sample_kwargs,
        mode="FAST_COMPILE",
    )
    with pytest.raises(RuntimeError, match="Model must be fit before"):
        unfitted_model.predict(X=None)


@pytest.mark.integration
def test_its_with_state_space_covariates():
    """ITS + StateSpaceTimeSeries with exogenous covariates end to end."""
    try:
        from pymc_extras.statespace import structural  # noqa: F401
    except ImportError:
        pytest.skip("pymc-extras is required for StateSpaceTimeSeries tests")

    rng = np.random.default_rng(seed=42)
    n = 100
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    season = 0.5 * np.sin(2 * np.pi * dates.dayofyear / 7)
    y = 5 + 0.05 * np.arange(n) + season + 2.0 * x1 - 1.5 * x2 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2}, index=dates)

    model = cp.pymc_models.StateSpaceTimeSeries(
        level_order=2,
        seasonal_length=7,
        sample_kwargs={
            "chains": 1,
            "draws": 100,
            "tune": 100,
            "progressbar": False,
            "random_seed": 7,
        },
    )

    # patsy adds an Intercept column; the model drops it with a warning
    with pytest.warns(UserWarning, match="Dropping the 'Intercept' column"):
        result = cp.InterruptedTimeSeries(
            data=df,
            treatment_time=dates[80],
            formula="y ~ 1 + x1 + x2",
            model=model,
        )

    # Covariates entered the model: beta_exog exists with the right coords.
    # No posterior-accuracy assertions here: the suite mocks pm.sample
    # session-wide (see conftest mock_pymc_sample), so draws come from the
    # prior. Numerical recovery is exercised outside the test suite.
    assert "beta_exog" in result.idata.posterior
    assert list(result.idata.posterior["beta_exog"].coords["state_exog"].values) == [
        "x1",
        "x2",
    ]

    # Counterfactual and impact have the post-period shape and finite values
    n_post = n - 80
    assert result.post_impact.sizes["obs_ind"] == n_post
    assert np.isfinite(result.post_impact.values).all()


@pytest.mark.integration
def test_its_with_state_space_variable_selection(mock_pymc_sample):
    """ITS + StateSpaceTimeSeries with spike-and-slab covariate selection.

    Structure-only assertions: the suite mocks pm.sample session-wide,
    so posterior values come from the prior.
    """
    try:
        from pymc_extras.statespace import structural  # noqa: F401
    except ImportError:
        pytest.skip("pymc-extras is required for StateSpaceTimeSeries tests")

    rng = np.random.default_rng(seed=42)
    n = 90
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 5 + 2.0 * x1 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3}, index=dates)

    model = cp.pymc_models.StateSpaceTimeSeries(
        level_order=1,
        seasonal_length=7,
        sample_kwargs={
            "chains": 1,
            "draws": 50,
            "tune": 50,
            "progressbar": False,
            "random_seed": 7,
        },
        vs_prior_type="spike_and_slab",
    )

    result = cp.InterruptedTimeSeries(
        data=df,
        treatment_time=dates[70],
        formula="y ~ 0 + x1 + x2 + x3",
        model=model,
    )

    assert "beta_exog" in result.idata.posterior
    assert "gamma_beta_exog" in result.idata.posterior

    incl = model.get_inclusion_probabilities()
    assert isinstance(incl, pd.DataFrame)
    assert len(incl) == 3
    assert ((incl["prob"] >= 0) & (incl["prob"] <= 1)).all()

    assert np.isfinite(result.post_impact.values).all()
