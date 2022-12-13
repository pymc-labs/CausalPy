import pandas as pd
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel
from sklearn.linear_model import LinearRegression

import causalpy as cp


@pytest.mark.integration
def test_did():
    data = cp.load_data("did")
    result = cp.skl_experiments.DifferenceInDifferences(
        data,
        formula="y ~ 1 + group + t + treated:group",
        time_variable_name="t",
        prediction_model=LinearRegression(),
    )
    assert isinstance(data, pd.DataFrame)
    assert isinstance(result, cp.skl_experiments.DifferenceInDifferences)


@pytest.mark.integration
def test_rd_drinking():
    df = (
        cp.load_data("drinking")
        .rename(columns={"agecell": "age"})
        .assign(treated=lambda df_: df_.age > 21)
    )
    result = cp.skl_experiments.RegressionDiscontinuity(
        df,
        formula="all ~ 1 + age + treated",
        running_variable_name="age",
        prediction_model=LinearRegression(),
        treatment_threshold=21,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.skl_experiments.RegressionDiscontinuity)


@pytest.mark.integration
def test_its():
    df = cp.load_data("its")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.skl_experiments.SyntheticControl(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        prediction_model=LinearRegression(),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.skl_experiments.SyntheticControl)


@pytest.mark.integration
def test_sc():
    df = cp.load_data("sc")
    treatment_time = 70
    result = cp.skl_experiments.SyntheticControl(
        df,
        treatment_time,
        formula="actual ~ 0 + a + b + c + d + e + f + g",
        prediction_model=cp.skl_models.WeightedProportion(),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.skl_experiments.SyntheticControl)


@pytest.mark.integration
def test_rd_linear_main_effects():
    data = cp.load_data("rd")
    result = cp.skl_experiments.RegressionDiscontinuity(
        data,
        formula="y ~ 1 + x + treated",
        prediction_model=LinearRegression(),
        treatment_threshold=0.5,
    )
    assert isinstance(data, pd.DataFrame)
    assert isinstance(result, cp.skl_experiments.RegressionDiscontinuity)


@pytest.mark.integration
def test_rd_linear_with_interaction():
    data = cp.load_data("rd")
    result = cp.skl_experiments.RegressionDiscontinuity(
        data,
        formula="y ~ 1 + x + treated + x:treated",
        prediction_model=LinearRegression(),
        treatment_threshold=0.5,
    )
    assert isinstance(data, pd.DataFrame)
    assert isinstance(result, cp.skl_experiments.RegressionDiscontinuity)


@pytest.mark.integration
def test_rd_linear_with_gaussian_process():
    data = cp.load_data("rd")
    kernel = 1.0 * ExpSineSquared(1.0, 5.0) + WhiteKernel(1e-1)
    result = cp.skl_experiments.RegressionDiscontinuity(
        data,
        formula="y ~ 1 + x + treated",
        prediction_model=GaussianProcessRegressor(kernel=kernel),
        treatment_threshold=0.5,
    )
    assert isinstance(data, pd.DataFrame)
    assert isinstance(result, cp.skl_experiments.RegressionDiscontinuity)
