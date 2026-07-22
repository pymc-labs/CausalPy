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
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from patsy import build_design_matrices
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel
from sklearn.linear_model import LinearRegression

import causalpy as cp


@pytest.mark.integration
def test_did(did_data):
    """
    Test Difference in Differences (DID) scikit-learn experiment.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.DifferenceInDifferences returns correct type
    """
    data = did_data
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
    with pytest.raises(NotImplementedError):
        result.get_plot_data()


@pytest.mark.integration
def test_did_causal_impact_order_independent_ols(did_data):
    """
    Regression test: causal_impact must not depend on which variable is
    written first in the DiD interaction term.

    Previously, DifferenceInDifferences.algorithm() looked up the OLS
    interaction coefficient using a single concatenated substring
    ("group:post_treatment"), which only matched patsy's column naming when
    the formula wrote the group variable first. Writing the formula the
    other way round (post_treatment*group) fit an identical model but
    silently produced causal_impact=None instead of the real value.
    """
    data = did_data

    result_group_first = cp.DifferenceInDifferences(
        data.copy(),
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )
    result_post_first = cp.DifferenceInDifferences(
        data.copy(),
        formula="y ~ 1 + post_treatment*group",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )

    assert result_group_first.causal_impact is not None
    assert result_post_first.causal_impact is not None
    assert result_group_first.causal_impact == pytest.approx(
        result_post_first.causal_impact
    )


@pytest.mark.integration
def test_rd_drinking():
    """
    Test Regression Discontinuity scikit-learn experiment on drinking age data.

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
    assert result.pred_discon.dims == (
        "chain",
        "draw",
        "obs_ind",
        "treated_units",
    )
    (discontinuity_design,) = build_design_matrices(
        [result._x_design_info], result.x_discon
    )
    legacy_prediction = result.model.predict(np.asarray(discontinuity_design))
    expected = np.squeeze(legacy_prediction[1]) - np.squeeze(legacy_prediction[0])
    assert np.asarray(result.discontinuity_at_threshold).item() == pytest.approx(
        expected
    )
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    with pytest.raises(NotImplementedError):
        result.get_plot_data()


@pytest.mark.integration
def test_its(its_data):
    """
    Test Interrupted Time Series scikit-learn experiment.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.InterruptedTimeSeries returns correct type
    3. the method get_plot_data returns a DataFrame with expected columns
    """

    df = its_data
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
    # Test get_plot_data with default parameters
    plot_data = result.get_plot_data()
    assert isinstance(plot_data, pd.DataFrame), (
        "The returned object is not a pandas DataFrame"
    )
    expected_columns = ["prediction", "impact"]
    assert set(expected_columns).issubset(set(plot_data.columns)), (
        f"DataFrame is missing expected columns {expected_columns}"
    )


@pytest.mark.integration
def test_sc(sc_data):
    """
    Test Synthetic Control scikit-learn experiment.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.SyntheticControl returns correct type
    3. the method get_plot_data returns a DataFrame with expected columns
    """
    df = sc_data
    treatment_time = 70
    result = cp.SyntheticControl(
        df,
        treatment_time,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
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
    # Test get_plot_data with default parameters
    plot_data = result.get_plot_data()
    assert isinstance(plot_data, pd.DataFrame), (
        "The returned object is not a pandas DataFrame"
    )
    expected_columns = ["prediction", "impact"]
    assert set(expected_columns).issubset(set(plot_data.columns)), (
        f"DataFrame is missing expected columns {expected_columns}"
    )


@pytest.mark.integration
def test_sc_datetime_treatment_time_plot(geolift1_data):
    """Test SyntheticControl plotting with datetime treatment_time and sklearn model."""
    df = geolift1_data
    treatment_time = pd.to_datetime("2022-01-01")

    result = cp.SyntheticControl(
        df,
        treatment_time,
        control_units=["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus"],
        treated_units=["Denmark"],
        model=cp.skl_models.WeightedProportion(),
    )

    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray) and all(
        isinstance(item, plt.Axes) for item in ax
    ), "ax must be a numpy.ndarray of plt.Axes"


@pytest.mark.integration
def test_rd_linear_main_effects(rd_data):
    """
    Test Regression Discontinuity scikit-learn experiment main effects.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.RegressionDiscontinuity returns correct type
    """
    data = rd_data
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
def test_rd_linear_main_effects_bandwidth(rd_data):
    """
    Test Regression Discontinuity scikit-learn experiment, main effects with
    bandwidth parameter.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.RegressionDiscontinuity returns correct type
    """
    data = rd_data
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
def test_rd_linear_main_effects_bandwidth_custom_running_variable():
    """
    Test Regression Discontinuity scikit-learn experiment with bandwidth parameter
    and custom running variable name.

    This test verifies the bug fix where the bandwidth parameter was hardcoding 'x'
    instead of using the user-specified running_variable_name.

    Creates synthetic data with custom column name and checks:
    1. RegressionDiscontinuity works with bandwidth and custom running variable name
    2. The model completes successfully
    3. Plot can be generated
    """
    # Create synthetic data with custom running variable name
    df = pd.DataFrame(
        {
            "my_running_var": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "outcome": [1, 2, 3, 4, 10, 11, 12],
            "treated": [False, False, False, False, True, True, True],
        }
    )

    # This should work without errors (previously failed with "name 'x' is not defined")
    result = cp.RegressionDiscontinuity(
        df,
        formula="outcome ~ 1 + my_running_var + treated",
        running_variable_name="my_running_var",
        model=LinearRegression(),
        treatment_threshold=0.45,
        bandwidth=0.2,
    )

    assert isinstance(result, cp.RegressionDiscontinuity)
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_rd_linear_with_interaction(rd_data):
    """
    Test Regression Discontinuity scikit-learn experiment with interaction.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.RegressionDiscontinuity returns correct type
    """
    data = rd_data
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
def test_rd_linear_with_gaussian_process(rd_data):
    """
    Test Regression Discontinuity scikit-learn experiment with Gaussian process model.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.RegressionDiscontinuity returns correct type
    """
    data = rd_data
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
