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
import arviz as az
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

import causalpy as cp

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


@pytest.mark.integration
def test_did(mock_pymc_sample):
    """
    Test Difference in Differences (DID) PyMC experiment.

    Loads data and checks:
    1. data is a dataframe
    2. pymc_experiements.DifferenceInDifferences returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    """
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.DifferenceInDifferences)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    with pytest.raises(NotImplementedError):
        result.get_plot_data()


# TODO: set up fixture for the banks dataset


@pytest.mark.integration
def test_did_banks_simple(mock_pymc_sample):
    """
    Test simple Differences In Differences Experiment on the 'banks' data set.

    :code: `formula="bib ~ 1 + district * post_treatment"`

    Loads, transforms data and checks:
    1. data is a dataframe
    2. pymc_experiements.DifferenceInDifferences returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data

    """
    treatment_time = 1930.5
    df = (
        cp.load_data("banks")
        .filter(items=["bib6", "bib8", "year"])
        .rename(columns={"bib6": "Sixth District", "bib8": "Eighth District"})
        .groupby("year")
        .median()
    )
    # SET TREATMENT TIME TO ZERO =========
    df.index = df.index - treatment_time
    treatment_time = 0
    # ====================================
    df.reset_index(level=0, inplace=True)
    df_long = pd.melt(
        df,
        id_vars=["year"],
        value_vars=["Sixth District", "Eighth District"],
        var_name="district",
        value_name="bib",
    ).sort_values("year")
    df_long["unit"] = df_long["district"]
    df_long["post_treatment"] = df_long.year >= treatment_time
    df_long = df_long.replace({"district": {"Sixth District": 1, "Eighth District": 0}})

    result = cp.DifferenceInDifferences(
        # df_long[df_long.year.isin([1930, 1931])],
        df_long[df_long.year.isin([-0.5, 0.5])],
        formula="bib ~ 1 + district * post_treatment",
        time_variable_name="year",
        group_variable_name="district",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.DifferenceInDifferences)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_did_banks_multi(mock_pymc_sample):
    """
    Test multiple regression Differences In Differences Experiment on the 'banks'
    data set.

    :code: `formula="bib ~ 1 + year + district + post_treatment + district:post_treatment"` # noqa: E501

    Loads, transforms data and checks:
    1. data is a dataframe
    2. pymc_experiements.DifferenceInDifferences returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    """
    treatment_time = 1930.5
    df = (
        cp.load_data("banks")
        .filter(items=["bib6", "bib8", "year"])
        .rename(columns={"bib6": "Sixth District", "bib8": "Eighth District"})
        .groupby("year")
        .median()
    )
    # SET TREATMENT TIME TO ZERO =========
    df.index = df.index - treatment_time
    treatment_time = 0
    # ====================================
    df.reset_index(level=0, inplace=True)
    df_long = pd.melt(
        df,
        id_vars=["year"],
        value_vars=["Sixth District", "Eighth District"],
        var_name="district",
        value_name="bib",
    ).sort_values("year")
    df_long["unit"] = df_long["district"]
    df_long["post_treatment"] = df_long.year >= treatment_time
    df_long = df_long.replace({"district": {"Sixth District": 1, "Eighth District": 0}})

    result = cp.DifferenceInDifferences(
        df_long,
        formula="bib ~ 1 + year + district + post_treatment + district:post_treatment",
        time_variable_name="year",
        group_variable_name="district",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.DifferenceInDifferences)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_rd(mock_pymc_sample):
    """
    Test Regression Discontinuity experiment.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.RegressionDiscontinuity returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    """
    df = cp.load_data("rd")
    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + bs(x, df=6) + treated",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=0.5,
        epsilon=0.001,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.RegressionDiscontinuity)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    with pytest.raises(NotImplementedError):
        result.get_plot_data()


@pytest.mark.integration
def test_rd_bandwidth(mock_pymc_sample):
    """
    Test Regression Discontinuity experiment with bandwidth parameter.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.RegressionDiscontinuity returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    """
    df = cp.load_data("rd")
    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + x + treated + x:treated",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=0.5,
        epsilon=0.001,
        bandwidth=0.3,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.RegressionDiscontinuity)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_rd_drinking(mock_pymc_sample):
    """
    Test Regression Discontinuity experiment on drinking age data.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.RegressionDiscontinuity returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
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
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=21,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.RegressionDiscontinuity)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def setup_regression_kink_data(kink):
    """Set up data for regression kink design tests"""
    # define parameters for data generation
    seed = 42
    rng = np.random.default_rng(seed)
    N = 50
    kink = 0.5
    beta = [0, -1, 0, 2, 0]
    sigma = 0.05
    # generate data
    x = rng.uniform(-1, 1, N)
    y = reg_kink_function(x, beta, kink) + rng.normal(0, sigma, N)
    return pd.DataFrame({"x": x, "y": y, "treated": x >= kink})


def reg_kink_function(x, beta, kink):
    """Utility function for regression kink design. Returns a piecewise linear function
    evaluated at x with a kink at kink and parameters beta"""
    return (
        beta[0]
        + beta[1] * x
        + beta[2] * x**2
        + beta[3] * (x - kink) * (x >= kink)
        + beta[4] * (x - kink) ** 2 * (x >= kink)
    )


@pytest.mark.integration
def test_rkink(mock_pymc_sample):
    """
    Test Regression Kink design.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.RegressionKink returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    """
    kink = 0.5
    df = setup_regression_kink_data(kink)
    result = cp.RegressionKink(
        df,
        formula=f"y ~ 1 + x + I((x-{kink})*treated)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        kink_point=kink,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.RegressionKink)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    with pytest.raises(NotImplementedError):
        result.get_plot_data()


@pytest.mark.integration
def test_rkink_bandwidth(mock_pymc_sample):
    """
    Test Regression Kink experiment with bandwidth parameter.

    Generates synthetic data and checks:
    1. data is a dataframe
    2. causalpy.RegressionKink returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    """
    kink = 0.5
    df = setup_regression_kink_data(kink)
    result = cp.RegressionKink(
        df,
        formula=f"y ~ 1 + x + I((x-{kink})*treated)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        kink_point=kink,
        bandwidth=0.3,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.RegressionKink)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_its(mock_pymc_sample):
    """
    Test Interrupted Time-Series experiment.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.InterruptedTimeSeries returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    5. the method get_plot_data returns a DataFrame with expected columns
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
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.InterruptedTimeSeries)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
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
    expected_columns = [
        "prediction",
        "pred_hdi_lower_94",
        "pred_hdi_upper_94",
        "impact",
        "impact_hdi_lower_94",
        "impact_hdi_upper_94",
    ]
    assert set(expected_columns).issubset(set(plot_data.columns)), (
        f"DataFrame is missing expected columns {expected_columns}"
    )


@pytest.mark.integration
def test_its_no_treatment_time():
    """
    Test Interrupted Time-Series experiment on COVID data with an unknown treatment time.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.InterruptedtimeSeries returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    5. the method get_plot_data returns a DataFrame with expected columns
    """

    df = (
        cp.load_data("covid")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = (pd.to_datetime("2014-01-01"), pd.to_datetime("2022-01-01"))

    # Assert that we correctfully raise a value if the given model can't predict InterventionTime
    with pytest.raises(cp.custom_exceptions.ModelException) as exc_info:
        cp.InterruptedTimeSeries(
            df,
            treatment_time,
            formula="standardize(deaths) ~ 0 + t + C(month) + standardize(temp)",  # noqa E501
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )
    assert (
        "If treatment_time is None, provided model must have a 'set_time_range' method"
        in str(exc_info.value)
    )

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="standardize(deaths) ~ 0 + t + C(month) + standardize(temp)",  # noqa E501
        model=cp.pymc_models.InterventionTimeEstimator(
            treatment_effect_type=["impulse", "level", "trend"],
            sample_kwargs=sample_kwargs,
        ),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.InterruptedTimeSeries)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
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
    expected_columns = [
        "prediction",
        "pred_hdi_lower_94",
        "pred_hdi_upper_94",
        "impact",
        "impact_hdi_lower_94",
        "impact_hdi_upper_94",
    ]
    assert set(expected_columns).issubset(set(plot_data.columns)), (
        f"DataFrame is missing expected columns {expected_columns}"
    )


@pytest.mark.integration
def test_its_covid(mock_pymc_sample):
    """
    Test Interrupted Time-Series experiment on COVID data.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.InterruptedtimeSeries returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    5. the method get_plot_data returns a DataFrame with expected columns
    """

    df = (
        cp.load_data("covid")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2020-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="standardize(deaths) ~ 0 + standardize(t) + C(month) + standardize(temp)",  # noqa E501
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.InterruptedTimeSeries)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
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
    expected_columns = [
        "prediction",
        "pred_hdi_lower_94",
        "pred_hdi_upper_94",
        "impact",
        "impact_hdi_lower_94",
        "impact_hdi_upper_94",
    ]
    assert set(expected_columns).issubset(set(plot_data.columns)), (
        f"DataFrame is missing expected columns {expected_columns}"
    )


@pytest.mark.integration
def test_sc(mock_pymc_sample):
    """
    Test Synthetic Control experiment.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.SyntheticControl returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    5. the method get_plot_data returns a DataFrame with expected columns
    """

    df = cp.load_data("sc")
    treatment_time = 70
    result = cp.SyntheticControl(
        df,
        treatment_time,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.SyntheticControl)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
    result.summary()

    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    # For multi-panel plots, ax should be an array of axes
    assert isinstance(ax, np.ndarray) and all(
        isinstance(item, plt.Axes) for item in ax
    ), "ax must be a numpy.ndarray of plt.Axes"

    fig, ax = result.plot(plot_predictors=True)
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
    expected_columns = [
        "prediction",
        "pred_hdi_lower_94",
        "pred_hdi_upper_94",
        "impact",
        "impact_hdi_lower_94",
        "impact_hdi_upper_94",
    ]
    assert set(expected_columns).issubset(set(plot_data.columns)), (
        f"DataFrame is missing expected columns {expected_columns}"
    )


@pytest.mark.integration
def test_sc_brexit(mock_pymc_sample):
    """
    Test Synthetic Control experiment on Brexit data.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.SyntheticControl returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    5. the method get_plot_data returns a DataFrame with expected columns
    """

    df = (
        cp.load_data("brexit")
        .assign(Time=lambda x: pd.to_datetime(x["Time"]))
        .set_index("Time")
        .loc[lambda x: x.index >= "2009-01-01"]
        .drop(["Japan", "Italy", "US", "Spain"], axis=1)
    )
    treatment_time = pd.to_datetime("2016 June 24")
    target_country = "UK"
    all_countries = df.columns
    other_countries = all_countries.difference({target_country})
    all_countries = list(all_countries)
    other_countries = list(other_countries)
    result = cp.SyntheticControl(
        df,
        treatment_time,
        control_units=other_countries,
        treated_units=[target_country],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.SyntheticControl)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
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
    expected_columns = [
        "prediction",
        "pred_hdi_lower_94",
        "pred_hdi_upper_94",
        "impact",
        "impact_hdi_lower_94",
        "impact_hdi_upper_94",
    ]
    assert set(expected_columns).issubset(set(plot_data.columns)), (
        f"DataFrame is missing expected columns {expected_columns}"
    )


@pytest.mark.integration
def test_ancova(mock_pymc_sample):
    """
    Test Pre-PostNEGD experiment on anova1 data.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.PrePostNEGD returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    """
    df = cp.load_data("anova1")
    result = cp.PrePostNEGD(
        df,
        formula="post ~ 1 + C(group) + pre",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.PrePostNEGD)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    # For multi-panel plots, ax should be an array of axes
    assert isinstance(ax, np.ndarray) and all(
        isinstance(item, plt.Axes) for item in ax
    ), "ax must be a numpy.ndarray of plt.Axes"


@pytest.mark.integration
def test_geolift1(mock_pymc_sample):
    """
    Test Synthetic Control experiment on geo lift data.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.SyntheticControl returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    """
    df = (
        cp.load_data("geolift1")
        .assign(time=lambda x: pd.to_datetime(x["time"]))
        .set_index("time")
    )
    treatment_time = pd.to_datetime("2022-01-01")
    result = cp.SyntheticControl(
        df,
        treatment_time,
        control_units=["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus"],
        treated_units=["Denmark"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.SyntheticControl)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    # For multi-panel plots, ax should be an array of axes
    assert isinstance(ax, np.ndarray) and all(
        isinstance(item, plt.Axes) for item in ax
    ), "ax must be a numpy.ndarray of plt.Axes"


@pytest.mark.integration
def test_iv_reg(mock_pymc_sample):
    df = cp.load_data("risk")
    instruments_formula = "risk  ~ 1 + logmort0"
    formula = "loggdp ~  1 + risk"
    instruments_data = df[["risk", "logmort0"]]
    data = df[["loggdp", "risk"]]

    result = cp.InstrumentalVariable(
        instruments_data=instruments_data,
        data=data,
        instruments_formula=instruments_formula,
        formula=formula,
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )
    result.model.sample_predictive_distribution(ppc_sampler="pymc")
    assert isinstance(df, pd.DataFrame)
    assert isinstance(data, pd.DataFrame)
    assert isinstance(instruments_data, pd.DataFrame)
    assert isinstance(result, cp.InstrumentalVariable)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
    with pytest.raises(NotImplementedError):
        result.get_plot_data()


@pytest.mark.integration
def test_inverse_prop(mock_pymc_sample):
    """Test the InversePropensityWeighting class."""
    df = cp.load_data("nhefs")
    sample_kwargs = {
        "tune": 100,
        "draws": 500,
        "chains": 2,
        "cores": 2,
        "random_seed": 100,
    }
    result = cp.InversePropensityWeighting(
        df,
        formula="trt ~ 1 + age + race",
        outcome_variable="outcome",
        weighting_scheme="robust",
        model=cp.pymc_models.PropensityScore(sample_kwargs=sample_kwargs),
    )
    assert isinstance(result.idata, az.InferenceData)
    ps = result.idata.posterior["p"].mean(dim=("chain", "draw"))
    w1, w2, _, _ = result.make_doubly_robust_adjustment(ps)
    assert isinstance(w1, pd.Series)
    assert isinstance(w2, pd.Series)
    w1, w2, n1, nw = result.make_raw_adjustments(ps)
    assert isinstance(w1, pd.Series)
    assert isinstance(w2, pd.Series)
    w1, w2, n1, n2 = result.make_robust_adjustments(ps)
    assert isinstance(w1, pd.Series)
    assert isinstance(w2, pd.Series)
    w1, w2, n1, n2 = result.make_overlap_adjustments(ps)
    assert isinstance(w1, pd.Series)
    assert isinstance(w2, pd.Series)
    ate_list = result.get_ate(0, result.idata)
    assert isinstance(ate_list, list)
    ate_list = result.get_ate(0, result.idata, method="raw")
    assert isinstance(ate_list, list)
    ate_list = result.get_ate(0, result.idata, method="robust")
    assert isinstance(ate_list, list)
    ate_list = result.get_ate(0, result.idata, method="overlap")
    assert isinstance(ate_list, list)
    fig, axs = result.plot_ate(prop_draws=1, ate_draws=10)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axs, list)
    assert all(isinstance(ax, plt.Axes) for ax in axs)
    fig, axs = result.plot_balance_ecdf("age")
    assert isinstance(fig, plt.Figure)
    assert isinstance(axs, list)
    assert all(isinstance(ax, plt.Axes) for ax in axs)
    with pytest.raises(NotImplementedError):
        result.get_plot_data()
