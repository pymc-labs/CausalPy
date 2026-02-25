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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel
from sklearn.linear_model import LinearRegression

import causalpy as cp


@pytest.mark.integration
def test_did():
    """
    Test Difference in Differences (DID) scikit-learn experiment.

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
    with pytest.raises(NotImplementedError):
        result.get_plot_data()


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
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    with pytest.raises(NotImplementedError):
        result.get_plot_data()


@pytest.mark.integration
def test_its():
    """
    Test Interrupted Time Series scikit-learn experiment.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.InterruptedTimeSeries returns correct type
    3. the method get_plot_data returns a DataFrame with expected columns
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
def test_sc():
    """
    Test Synthetic Control scikit-learn experiment.

    Loads data and checks:
    1. data is a dataframe
    2. skl_experiements.SyntheticControl returns correct type
    3. the method get_plot_data returns a DataFrame with expected columns
    """
    df = cp.load_data("sc")
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
def test_rd_linear_main_effects():
    """
    Test Regression Discontinuity scikit-learn experiment main effects.

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
    Test Regression Discontinuity scikit-learn experiment, main effects with
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
def test_rd_linear_with_interaction():
    """
    Test Regression Discontinuity scikit-learn experiment with interaction.

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
    Test Regression Discontinuity scikit-learn experiment with Gaussian process model.

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


@pytest.mark.integration
def test_graded_intervention_time_series_end_to_end():
    """
    Test Graded Intervention Time Series end-to-end workflow.

    This integration test exercises the full workflow:
    1. Create data
    2. Configure TransferFunctionOLS model
    3. Run GradedInterventionTimeSeries experiment
    4. Call all major methods: plot(), plot_transforms(), effect(), plot_effect(), summary()
    5. Verify all methods work together
    """
    # Generate synthetic data
    np.random.seed(42)
    n = 80
    t = np.arange(n)
    dates = pd.date_range("2020-01-01", periods=n, freq="W")

    # Create treatment with known transforms
    treatment_raw = 50 + 30 * np.sin(2 * np.pi * t / 20) + np.random.uniform(-10, 10, n)
    treatment_raw = np.maximum(treatment_raw, 0)

    # Generate outcome (we don't know true transforms, model will estimate them)
    y = 100.0 + 0.5 * t + 2.0 * treatment_raw + np.random.normal(0, 10, n)

    df = pd.DataFrame({"date": dates, "t": t, "y": y, "treatment": treatment_raw})
    df = df.set_index("date")

    # Create TransferFunctionOLS model
    model = cp.skl_models.TransferFunctionOLS(
        saturation_type="hill",
        saturation_grid={"slope": [1.0, 2.0, 3.0], "kappa": [40, 50, 60]},
        adstock_grid={"half_life": [2, 3, 4], "l_max": [12], "normalize": [True]},
        estimation_method="grid",
        error_model="hac",
    )

    # Run experiment
    result = cp.GradedInterventionTimeSeries(
        data=df,
        y_column="y",
        treatment_names=["treatment"],
        base_formula="1 + t",
        model=model,
    )

    # Verify experiment result
    assert isinstance(result, cp.GradedInterventionTimeSeries)
    assert result.score > 0.5  # Reasonable fit

    # Test plot() method
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray)
    assert len(ax) == 2
    plt.close(fig)

    # Test plot_transforms() method
    fig, ax = result.plot_transforms()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray)
    assert len(ax) == 2
    plt.close(fig)

    # Test effect() method
    effect_result = result.effect(
        window=(df.index[0], df.index[-1]), channels=["treatment"], scale=0.0
    )
    assert "effect_df" in effect_result
    assert "total_effect" in effect_result
    assert "mean_effect" in effect_result
    assert isinstance(effect_result["effect_df"], pd.DataFrame)

    # Test plot_effect() method
    fig, ax = result.plot_effect(effect_result)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray)
    assert len(ax) == 2
    plt.close(fig)

    # Test summary() method (capture output to avoid cluttering test output)
    import io
    import sys

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result.summary()
        output = sys.stdout.getvalue()
        assert "Graded Intervention Time Series Results" in output
        assert "Outcome variable" in output
        assert "Treatment coefficients" in output
    finally:
        sys.stdout = old_stdout

    # Test plot_diagnostics() method
    sys.stdout = io.StringIO()
    try:
        result.plot_diagnostics(lags=10)
    finally:
        sys.stdout = old_stdout
        plt.close("all")

    # Test get_plot_data_ols() method
    plot_data = result.get_plot_data_ols()
    assert isinstance(plot_data, pd.DataFrame)
    assert "observed" in plot_data.columns
    assert "fitted" in plot_data.columns
    assert "residuals" in plot_data.columns
