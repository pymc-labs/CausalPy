#   Copyright 2022 - 2025 The PyMC Labs Developers
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
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel
from sklearn.linear_model import LinearRegression

import causalpy as cp


@pytest.mark.integration
def test_did():
    """
    Test Difference in Differences (DID) Sci-Kit Learn experiment.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.DifferenceInDifferences returns correct type
    """
    data = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        treated=1,
        untreated=0,
        model=LinearRegression(),
    )
    assert isinstance(data, pd.DataFrame)
    assert isinstance(result, cp.DifferenceInDifferences)
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_rd_drinking():
    """
    Test Regression Discontinuity Sci-Kit Learn experiment on drinking age data.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.RegressionDiscontinuity returns correct type
    """

    df = (
        cp.load_data("drinking")
        .rename(columns={"agecell": "age"})
        .assign(treated=lambda df_: df_.age > 21)
    )
    result = cp.RegressionDiscontinuity(
        df,
        formula="all ~ 1 + age + treated",
        running_variable_name="age",
        model=LinearRegression(),
        treatment_threshold=21,
        epsilon=0.001,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.RegressionDiscontinuity)
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_its():
    """
    Test Interrupted Time Series Sci-Kit Learn experiment.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.InterruptedTimeSeries returns correct type
    """

    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=LinearRegression(),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.InterruptedTimeSeries)
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    # For multi-panel plots, ax should be an array of axes
    assert isinstance(ax, np.ndarray) and all(
        isinstance(item, plt.Axes) for item in ax
    ), "ax must be a numpy.ndarray of plt.Axes"
    plot_data = result.get_plot_data()
    assert isinstance(plot_data, pd.DataFrame), "The returned object is not a pandas DataFrame"
    expected_columns = ['prediction', 'impact']
    assert set(expected_columns).issubset(set(plot_data.columns)), f"DataFrame is missing expected columns {expected_columns}"


@pytest.mark.integration
def test_sc():
    """
    Test Synthetic Control Sci-Kit Learn experiment.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.SyntheticControl returns correct type
    """
    df = cp.load_data("sc")
    treatment_time = 70
    result = cp.SyntheticControl(
        df,
        treatment_time,
        formula="actual ~ 0 + a + b + c + d + e + f + g",
        model=cp.skl_models.WeightedProportion(),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.SyntheticControl)
    result.summary()

    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    # For multi-panel plots, ax should be an array of axes
    assert isinstance(ax, np.ndarray) and all(
        isinstance(item, plt.Axes) for item in ax
    ), "ax must be a numpy.ndarray of plt.Axes"

    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    # For multi-panel plots, ax should be an array of axes
    assert isinstance(ax, np.ndarray) and all(
        isinstance(item, plt.Axes) for item in ax
    ), "ax must be a numpy.ndarray of plt.Axes"
    plot_data = result.get_plot_data()
    assert isinstance(plot_data, pd.DataFrame), "The returned object is not a pandas DataFrame"
    expected_columns = ['prediction', 'impact']
    assert set(expected_columns).issubset(set(plot_data.columns)), f"DataFrame is missing expected columns {expected_columns}"


@pytest.mark.integration
def test_rd_linear_main_effects():
    """
    Test Regression Discontinuity Sci-Kit Learn experiment main effects.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.RegressionDiscontinuity returns correct type
    """
    data = cp.load_data("rd")
    result = cp.RegressionDiscontinuity(
        data,
        formula="y ~ 1 + x + treated",
        model=LinearRegression(),
        treatment_threshold=0.5,
        epsilon=0.001,
    )
    assert isinstance(data, pd.DataFrame)
    assert isinstance(result, cp.RegressionDiscontinuity)
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_rd_linear_main_effects_bandwidth():
    """
    Test Regression Discontinuity Sci-Kit Learn experiment, main effects with
    bandwidth parameter.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.RegressionDiscontinuity returns correct type
    """
    data = cp.load_data("rd")
    result = cp.RegressionDiscontinuity(
        data,
        formula="y ~ 1 + x + treated",
        model=LinearRegression(),
        treatment_threshold=0.5,
        epsilon=0.001,
        bandwidth=0.3,
    )
    assert isinstance(data, pd.DataFrame)
    assert isinstance(result, cp.RegressionDiscontinuity)
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_rd_linear_with_interaction():
    """
    Test Regression Discontinuity Sci-Kit Learn experiment with interaction.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.RegressionDiscontinuity returns correct type
    """
    data = cp.load_data("rd")
    result = cp.RegressionDiscontinuity(
        data,
        formula="y ~ 1 + x + treated + x:treated",
        model=LinearRegression(),
        treatment_threshold=0.5,
        epsilon=0.001,
    )
    assert isinstance(data, pd.DataFrame)
    assert isinstance(result, cp.RegressionDiscontinuity)
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_rd_linear_with_gaussian_process():
    """
    Test Regression Discontinuity Sci-Kit Learn experiment with Gaussian process model.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.RegressionDiscontinuity returns correct type
    """
    data = cp.load_data("rd")
    kernel = 1.0 * ExpSineSquared(1.0, 5.0) + WhiteKernel(1e-1)
    result = cp.RegressionDiscontinuity(
        data,
        formula="y ~ 1 + x + treated",
        model=GaussianProcessRegressor(kernel=kernel),
        model_kwargs={"kernel": kernel},
        treatment_threshold=0.5,
        epsilon=0.001,
    )
    assert isinstance(data, pd.DataFrame)
    assert isinstance(result, cp.RegressionDiscontinuity)
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


# DEPRECATION WARNING TESTS ============================================================


def test_did_deprecation_warning():
    """Test that the old DifferenceInDifferences class raises a deprecation warning."""

    with pytest.warns(DeprecationWarning):
        data = cp.load_data("did")
        result = cp.skl_experiments.DifferenceInDifferences(
            data,
            formula="y ~ 1 + group*post_treatment",
            time_variable_name="t",
            group_variable_name="group",
            treated=1,
            untreated=0,
            model=LinearRegression(),
        )
        assert isinstance(result, cp.DifferenceInDifferences)


def test_its_deprecation_warning():
    """Test that the old InterruptedTimeSeries class raises a deprecation warning."""

    with pytest.warns(DeprecationWarning):
        df = (
            cp.load_data("its")
            .assign(date=lambda x: pd.to_datetime(x["date"]))
            .set_index("date")
        )
        treatment_time = pd.to_datetime("2017-01-01")
        result = cp.skl_experiments.InterruptedTimeSeries(
            df,
            treatment_time,
            formula="y ~ 1 + t + C(month)",
            model=LinearRegression(),
        )
        assert isinstance(result, cp.InterruptedTimeSeries)


def test_sc_deprecation_warning():
    """Test that the old SyntheticControl class raises a deprecation warning."""

    with pytest.warns(DeprecationWarning):
        df = cp.load_data("sc")
        treatment_time = 70
        result = cp.skl_experiments.SyntheticControl(
            df,
            treatment_time,
            formula="actual ~ 0 + a + b + c + d + e + f + g",
            model=cp.skl_models.WeightedProportion(),
        )
        assert isinstance(result, cp.SyntheticControl)


def test_rd_deprecation_warning():
    """Test that the old RegressionDiscontinuity class raises a deprecation warning."""

    with pytest.warns(DeprecationWarning):
        data = cp.load_data("rd")
        result = cp.skl_experiments.RegressionDiscontinuity(
            data,
            formula="y ~ 1 + x + treated",
            model=LinearRegression(),
            treatment_threshold=0.5,
            epsilon=0.001,
        )
        assert isinstance(result, cp.RegressionDiscontinuity)
