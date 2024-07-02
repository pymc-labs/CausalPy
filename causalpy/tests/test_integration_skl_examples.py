#   Copyright 2024 The PyMC Labs Developers
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

import causalpy as cp
from causalpy.skl_models import LinearRegression


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
        model=cp.skl_models.LinearRegression(),
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
        treatment_threshold=0.5,
        epsilon=0.001,
    )
    assert isinstance(data, pd.DataFrame)
    assert isinstance(result, cp.RegressionDiscontinuity)
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
