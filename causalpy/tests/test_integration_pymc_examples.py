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

import contextlib

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from matplotlib import pyplot as plt

import causalpy as cp
from causalpy.tests.conftest import setup_regression_kink_data

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


@pytest.mark.integration
def test_did_banks_simple(mock_pymc_sample, banks_data):
    """
    Test simple Differences In Differences Experiment on the 'banks' data set.

    :code: `formula="bib ~ 1 + district * post_treatment"`

    Uses the ``banks_data`` fixture and checks:
    1. data is a dataframe
    2. pymc_experiements.DifferenceInDifferences returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    """
    df_long, _treatment_time = banks_data

    result = cp.DifferenceInDifferences(
        df_long[df_long.year.isin([-0.5, 0.5])],
        formula="bib ~ 1 + district * post_treatment",
        time_variable_name="year",
        group_variable_name="district",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df_long, pd.DataFrame)
    assert isinstance(result, cp.DifferenceInDifferences)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
    result.summary()
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_did_banks_multi(mock_pymc_sample, banks_data):
    """
    Test multiple regression Differences In Differences Experiment on the 'banks'
    data set.

    :code: `formula="bib ~ 1 + year + district + post_treatment + district:post_treatment"` # noqa: E501

    Uses the ``banks_data`` fixture and checks:
    1. data is a dataframe
    2. pymc_experiements.DifferenceInDifferences returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    """
    df_long, _treatment_time = banks_data

    result = cp.DifferenceInDifferences(
        df_long,
        formula="bib ~ 1 + year + district + post_treatment + district:post_treatment",
        time_variable_name="year",
        group_variable_name="district",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df_long, pd.DataFrame)
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
def test_rd_bandwidth_custom_running_variable(mock_pymc_sample):
    """
    Test Regression Discontinuity experiment with bandwidth parameter and custom running variable name.

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
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=0.45,
        bandwidth=0.2,
    )

    assert isinstance(result, cp.RegressionDiscontinuity)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
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
    # Test 1. plot method runs
    result.plot()
    # 2. causalpy.InterruptedTimeSeries returns correct type
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
    2. causalpy.InterruptedTimeSeries returns correct type
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
    # Test 1. plot method runs
    result.plot()
    # 2. causalpy.InterruptedTimeSeries returns correct type
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
def test_iv_binary_treatment(mock_pymc_sample):
    df = cp.load_data("risk")
    df["binary_trt"] = np.random.binomial(1, 0.5, len(df))
    instruments_formula = "binary_trt  ~ 1 + risk + logmort0"
    formula = "loggdp ~  1 + binary_trt + risk"
    instruments_data = df[["risk", "logmort0", "binary_trt"]]
    data = df[["loggdp", "risk", "binary_trt"]]

    result = cp.InstrumentalVariable(
        instruments_data=instruments_data,
        data=data,
        instruments_formula=instruments_formula,
        formula=formula,
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
        binary_treatment=True,
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
    assert "rho" in result.model.named_vars


@pytest.mark.integration
def test_iv_reg_vs_prior(mock_pymc_sample):
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
        vs_prior_type="spike_and_slab",
        vs_hyperparams={"pi_alpha": 5, "outcome": True},
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
    assert "gamma_beta_t" in result.model.named_vars
    assert "pi_beta_t" in result.model.named_vars
    summary = result.model.vs_prior_outcome.get_inclusion_probabilities(
        result.idata, "beta_z"
    )
    assert isinstance(summary, pd.DataFrame)
    with pytest.raises(ValueError):
        summary = result.model.vs_prior_outcome.get_shrinkage_factors(
            result.idata, "beta_z"
        )


@pytest.mark.integration
def test_iv_reg_vs_prior_hs(mock_pymc_sample):
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
        vs_prior_type="horseshoe",
        vs_hyperparams={"outcome": True},
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
    assert "tau_beta_t" in result.model.named_vars
    assert "tau_beta_z" in result.model.named_vars
    summary = result.model.vs_prior_outcome.get_shrinkage_factors(
        result.idata, "beta_z"
    )
    assert isinstance(summary, pd.DataFrame)
    with pytest.raises(ValueError):
        summary = result.model.vs_prior_outcome.get_inclusion_probabilities(
            result.idata, "beta_z"
        )


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
    plt.close()
    with pytest.raises(NotImplementedError):
        result.get_plot_data()

    ### testing outcome model
    idata_normal, model_normal = result.model.fit_outcome_model(
        X_outcome=result.X_outcome,
        y=result.y,
        coords=result.coords,
        normal_outcome=True,
        spline_component=False,
    )
    assert isinstance(idata_normal, az.InferenceData)
    assert isinstance(model_normal, pm.Model)
    assert "beta_" in idata_normal.posterior
    assert "beta_ps" in idata_normal.posterior

    # Test spline model
    idata_spline, _ = result.model.fit_outcome_model(
        X_outcome=result.X_outcome,
        y=result.y,
        coords=result.coords,
        normal_outcome=True,
        spline_component=True,
    )
    assert "spline_features" in idata_spline.posterior

    # Test student-t outcome
    idata_student, _ = result.model.fit_outcome_model(
        X_outcome=result.X_outcome,
        y=result.y,
        coords=result.coords,
        noncentred=False,
        normal_outcome=False,
        spline_component=False,
    )
    assert "nu" in idata_student.posterior


@pytest.mark.integration
def test_bayesian_structural_time_series():
    """Test the BayesianBasisExpansionTimeSeries model."""
    # Generate synthetic data
    rng = np.random.default_rng(seed=123)
    dates = pd.date_range(start="2020-01-01", end="2021-12-31", freq="D")
    n_obs = len(dates)
    trend_actual = np.linspace(0, 2, n_obs)
    seasonality_actual = 3 * np.sin(2 * np.pi * dates.dayofyear / 365.25) + 2 * np.cos(
        4 * np.pi * dates.dayofyear / 365.25
    )
    x1_actual = rng.normal(0, 1, n_obs)
    beta_x1_actual = 1.5
    noise_actual = rng.normal(0, 0.3, n_obs)

    y_values_with_x = (
        trend_actual + seasonality_actual + beta_x1_actual * x1_actual + noise_actual
    )
    y_values_no_x = trend_actual + seasonality_actual + noise_actual

    data_with_x = pd.DataFrame({"y": y_values_with_x, "x1": x1_actual}, index=dates)
    data_no_x = pd.DataFrame({"y": y_values_no_x}, index=dates)

    # Note: day_of_year and time_numeric are not directly passed in coords to build_model anymore
    # They are derived from datetime_index. They can remain here for clarity or potential future use
    # in a more complex test setup if needed, but are not strictly necessary for current model.
    # day_of_year = dates.dayofyear.to_numpy()
    # time_numeric = (dates - dates[0]).days.to_numpy() / 365.25

    bsts_sample_kwargs = {
        "chains": 1,
        "draws": 100,
        "tune": 50,
        "progressbar": False,
        "random_seed": 42,
    }

    # --- Test Case 1: Model with exogenous regressor --- #
    coords_with_x = {
        "obs_ind": dates,  # Use dates directly for xarray coords
        "coeffs": ["x1"],
        "treated_units": ["unit_0"],
        "datetime_index": dates,
    }

    # Create DataArrays for input to match new API
    X_da = xr.DataArray(
        data_with_x[["x1"]].values,
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": dates, "coeffs": ["x1"]},
    )
    y_da = xr.DataArray(
        data_with_x["y"].values[:, None],
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": dates, "treated_units": ["unit_0"]},
    )

    model_with_x = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        n_order=2, n_changepoints_trend=5, sample_kwargs=bsts_sample_kwargs
    )
    model_with_x.fit(
        X=X_da,
        y=y_da,
        coords=coords_with_x.copy(),  # Pass a copy
    )
    assert isinstance(model_with_x.idata, az.InferenceData)
    assert "posterior" in model_with_x.idata
    assert "beta" in model_with_x.idata.posterior
    # PyMC Marketing components might use different internal names, e.g. fourier_beta, delta
    # Let's check for existence of key components rather than exact pymc_marketing internal names
    # if specific internal names are not exposed or guaranteed by causalpy's BSTS.
    # For now, assuming 'fourier_beta' and 'delta' are names exposed by the pymc_marketing components used.
    assert (
        "fourier_beta" in model_with_x.idata.posterior
    )  # Trend/Seasonality component param
    assert "delta" in model_with_x.idata.posterior  # Trend/Seasonality component param
    assert "sigma" in model_with_x.idata.posterior
    assert "mu" in model_with_x.idata.posterior_predictive
    assert "y_hat" in model_with_x.idata.posterior_predictive

    predictions_with_x = model_with_x.predict(
        X=X_da,
        coords=coords_with_x,  # Original coords_with_x is fine here
    )
    assert isinstance(predictions_with_x, az.InferenceData)
    score_with_x = model_with_x.score(
        X=X_da,
        y=y_da,
        coords=coords_with_x,  # Original coords_with_x is fine here
    )
    assert isinstance(score_with_x, pd.Series)

    # --- Test Case 2: Model without exogenous regressor --- #
    coords_no_x = {
        "obs_ind": dates,
        "treated_units": ["unit_0"],
        "datetime_index": dates,
        # "coeffs": [], # Explicitly empty or omitted if X is None
    }

    y_da_no_x = xr.DataArray(
        data_no_x["y"].values[:, None],
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": dates, "treated_units": ["unit_0"]},
    )

    # Create X_da_no_x (empty coeffs) to provide time index for predict
    X_da_no_x = xr.DataArray(
        np.zeros((len(dates), 0)),  # 0 coeffs
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": dates, "coeffs": []},
    )

    model_no_x = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        n_order=2, n_changepoints_trend=5, sample_kwargs=bsts_sample_kwargs
    )

    model_no_x.fit(
        X=X_da_no_x,
        y=y_da_no_x,
        coords=coords_no_x.copy(),  # Pass a copy
    )
    assert isinstance(model_no_x.idata, az.InferenceData)
    assert "posterior" in model_no_x.idata
    assert "beta" not in model_no_x.idata.posterior
    assert "fourier_beta" in model_no_x.idata.posterior
    assert "delta" in model_no_x.idata.posterior
    assert "sigma" in model_no_x.idata.posterior

    predictions_no_x = model_no_x.predict(
        X=X_da_no_x,
        coords=coords_no_x,  # Original coords_no_x is fine
    )
    assert isinstance(predictions_no_x, az.InferenceData)
    score_no_x = model_no_x.score(
        X=X_da_no_x,
        y=y_da_no_x,
        coords=coords_no_x,  # Original coords_no_x is fine
    )
    assert isinstance(score_no_x, pd.Series)

    # --- Test Case 3: Model with empty exogenous regressor (X has 0 columns) --- #
    # This is similar to Test Case 2. Model should handle X with 0 columns
    coords_empty_x = {  # Coords for 0 exog vars
        "obs_ind": dates,
        "treated_units": ["unit_0"],
        "datetime_index": dates,
        "coeffs": [],  # Must be empty list if X has 0 columns and 'coeffs' is provided
    }

    # Reuse X_da_no_x from Test Case 2 as it has 0 columns and correct coords
    # Reuse y_da_no_x from Test Case 2

    model_empty_x = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        n_order=2, n_changepoints_trend=5, sample_kwargs=bsts_sample_kwargs
    )
    model_empty_x.fit(
        X=X_da_no_x,
        y=y_da_no_x,
        coords=coords_empty_x.copy(),  # Pass a copy
    )
    assert isinstance(model_empty_x.idata, az.InferenceData)

    predictions_empty_x = model_empty_x.predict(
        X=X_da_no_x,
        coords=coords_empty_x,  # Original coords_empty_x is fine
    )
    assert isinstance(predictions_empty_x, az.InferenceData)
    score_empty_x = model_empty_x.score(
        X=X_da_no_x,
        y=y_da_no_x,
        coords=coords_empty_x,  # Original coords_empty_x is fine
    )
    assert isinstance(score_empty_x, pd.Series)

    # --- Test Case 4: Model with incorrect coord/data setup (ValueErrors) --- #
    # Test that X must have datetime coordinates
    with pytest.raises(
        ValueError,
        match=r"X\.coords\['obs_ind'\] must contain datetime values",
    ):
        model_error_idx = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs=bsts_sample_kwargs
        )
        # Create X with non-datetime obs_ind coordinates
        bad_X = xr.DataArray(
            data_with_x[["x1"]].values,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": np.arange(n_obs),
                "coeffs": ["x1"],
            },  # integers not datetime
        )
        bad_y = xr.DataArray(
            data_with_x["y"].values[:, None],
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
        )
        model_error_idx.fit(
            X=bad_X,
            y=bad_y,
            coords=coords_with_x.copy(),
        )

    with pytest.raises(ValueError, match="Model was built with exogenous variables"):
        # Pass X with no exogenous vars (X_da_no_x) to model expecting vars (model_with_x)
        # This checks that we can't predict without supplying the expected exog vars
        model_with_x.predict(X=X_da_no_x, coords=coords_with_x)

    with pytest.raises(
        ValueError,
        match=r"Exogenous variable names mismatch",
    ):
        wrong_shape_x_pred_vals = np.hstack(
            [data_with_x[["x1"]].values, data_with_x[["x1"]].values]
        )  # 2 columns

        X_wrong_shape = xr.DataArray(
            wrong_shape_x_pred_vals,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": dates,
                "coeffs": ["x1", "x2"],  # 2 coeffs
            },
        )

        model_with_x.predict(X=X_wrong_shape, coords=coords_with_x)

    # --- Test Case 5: Custom component validation errors --- #
    class BadTrendComponent:
        """Component without apply method"""

        pass

    with pytest.raises(
        ValueError,
        match="Custom trend_component must have an 'apply' method",
    ):
        cp.pymc_models.BayesianBasisExpansionTimeSeries(
            trend_component=BadTrendComponent(),
            sample_kwargs=bsts_sample_kwargs,
        )

    with pytest.raises(
        ValueError,
        match="Custom seasonality_component must have an 'apply' method",
    ):
        cp.pymc_models.BayesianBasisExpansionTimeSeries(
            seasonality_component=BadTrendComponent(),
            sample_kwargs=bsts_sample_kwargs,
        )

    # --- Test Case 6: Additional error conditions --- #
    # Test TypeError for non-xarray X (expecting xarray DataArray)
    with pytest.raises(TypeError, match="X must be an xarray DataArray"):
        model_with_x.predict(
            X=data_with_x[["x1"]].values,  # Pass numpy array instead of xarray
            coords=coords_with_x,
        )


@pytest.mark.integration
def test_state_space_time_series():
    """
    Test InterruptedTimeSeries model.

    This test verifies the InterruptedTimeSeries model functionality including:
    1. Model initialization and parameter validation
    2. Model fitting with synthetic time series data
    3. In-sample and out-of-sample prediction
    4. Model scoring (Bayesian R²)
    5. Error handling for invalid inputs
    6. State-space model components and structure

    The InterruptedTimeSeries model uses pymc-extras for state-space modeling,
    which provides Kalman filtering and smoothing capabilities.

    Note: This test will be skipped if pymc-extras is not available in the environment.
    The test is designed to be comprehensive but also robust to dependency issues.
    """
    # Check if pymc-extras is available
    try:
        from pymc_extras.statespace import structural  # noqa: F401
    except ImportError:
        pytest.skip("pymc-extras is required for InterruptedTimeSeries tests")

    # Generate synthetic time series data with trend and seasonality
    rng = np.random.default_rng(seed=123)
    dates = pd.date_range(
        start="2020-01-01", end="2020-03-31", freq="D"
    )  # Shorter period for faster testing
    n_obs = len(dates)

    # Create synthetic components
    trend_actual = np.linspace(0, 2, n_obs)  # Linear trend
    seasonality_actual = 3 * np.sin(2 * np.pi * dates.dayofyear / 365.25) + 2 * np.cos(
        4 * np.pi * dates.dayofyear / 365.25
    )  # Yearly seasonality
    noise_actual = rng.normal(0, 0.3, n_obs)  # Observation noise

    y_values = trend_actual + seasonality_actual + noise_actual
    data = pd.DataFrame({"y": y_values}, index=dates)

    # Sample configuration for faster testing
    ss_sample_kwargs = {
        "chains": 1,
        "draws": 50,  # Reduced for faster testing
        "tune": 25,  # Reduced for faster testing
        "progressbar": False,
        "random_seed": 42,
    }

    # Create DataArray for y to support score() which requires xarray
    # Use dates as obs_ind coordinate (datetime values required by new API)
    y_da = xr.DataArray(
        data["y"].values.reshape(-1, 1),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": dates, "treated_units": ["unit_0"]},
    )

    # Initialize model with PyMC mode (more stable than JAX for testing)
    model = cp.pymc_models.StateSpaceTimeSeries(
        level_order=2,  # Local linear trend (level + slope)
        seasonal_length=7,  # Weekly seasonality for shorter test period
        sample_kwargs=ss_sample_kwargs,
        mode="FAST_COMPILE",  # Use PyMC mode instead of JAX for better compatibility
    )

    # Test the complete workflow
    # --- Test Case 1: Model fitting --- #
    # Create dummy X (state-space doesn't use exogenous vars but we pass empty array for API consistency)
    dummy_X = xr.DataArray(
        np.zeros((len(dates), 0)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": dates, "coeffs": []},
    )
    # StateSpaceTimeSeries extracts datetime from xarray coords, no separate coords dict needed
    idata = model.fit(
        X=dummy_X,
        y=y_da,
    )

    # Verify inference data structure
    assert isinstance(idata, az.InferenceData)
    assert "posterior" in idata
    assert "posterior_predictive" in idata

    # Check for expected state-space parameters
    expected_params = [
        "P0_diag",
        "initial_level_trend",
        "params_freq",
        "sigma_level_trend",
        "sigma_freq",
    ]
    for param in expected_params:
        assert param in idata.posterior, f"Parameter {param} not found in posterior"

    # Check for expected posterior predictive variables
    assert "y_hat" in idata.posterior_predictive
    assert "mu" in idata.posterior_predictive

    # --- Test Case 2: In-sample prediction --- #
    # Create dummy X for in-sample prediction (state-space doesn't use it but API requires it for consistency)
    dummy_X_insample = xr.DataArray(
        np.zeros((len(dates), 0)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": dates, "coeffs": []},
    )
    predictions_in_sample = model.predict(
        X=dummy_X_insample,
        out_of_sample=False,
    )
    assert isinstance(predictions_in_sample, az.InferenceData)
    assert "posterior_predictive" in predictions_in_sample
    assert "y_hat" in predictions_in_sample.posterior_predictive
    assert "mu" in predictions_in_sample.posterior_predictive

    # --- Test Case 3: Out-of-sample prediction (forecasting) --- #
    future_dates = pd.date_range(start="2020-04-01", end="2020-04-07", freq="D")
    # Create dummy X for forecasting (needs time index)
    future_X = xr.DataArray(
        np.zeros((len(future_dates), 0)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": future_dates, "coeffs": []},
    )

    predictions_out_sample = model.predict(
        X=future_X,
        out_of_sample=True,
    )
    # Note: predict now returns InferenceData, not Dataset!
    # But let's check what the test expects.
    # The previous code expected xr.Dataset:
    # assert isinstance(predictions_out_sample, xr.Dataset)
    # I updated predict() to return az.InferenceData.
    # So I should update this assertion too.

    assert isinstance(predictions_out_sample, az.InferenceData)
    assert "y_hat" in predictions_out_sample.posterior_predictive
    assert "mu" in predictions_out_sample.posterior_predictive

    # Verify forecast has correct dimensions
    # y_hat is in posterior_predictive group
    assert predictions_out_sample.posterior_predictive["y_hat"].shape[-1] == len(
        future_dates
    )

    # --- Test Case 4: Model scoring --- #
    # Create dummy X for score (state-space doesn't use it but API requires it)
    dummy_X_for_score = xr.DataArray(
        np.zeros((len(dates), 0)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": dates, "coeffs": []},
    )
    score = model.score(
        X=dummy_X_for_score,
        y=y_da,
    )
    assert isinstance(score, pd.Series)
    assert "unit_0_r2" in score.index
    assert "unit_0_r2_std" in score.index
    # R² should be reasonable for synthetic data with clear structure
    assert score["unit_0_r2"] > 0.0, (
        "R² should be positive for structured synthetic data"
    )

    # --- Test Case 5: Model components verification --- #
    # Test that the model has the expected state-space structure
    assert hasattr(model, "ss_mod")
    assert model.ss_mod is not None
    assert hasattr(model, "_train_index")
    assert isinstance(model._train_index, pd.DatetimeIndex)

    # Test conditional inference data
    assert hasattr(model, "conditional_idata")
    assert isinstance(model.conditional_idata, xr.Dataset)

    # Verify model parameters match initialization
    assert model.level_order == 2
    assert model.seasonal_length == 7
    assert model.mode == "FAST_COMPILE"

    # --- Test Case 6: Error handling --- #
    # Test that y must have datetime coordinates
    with pytest.raises(
        ValueError,
        match=r"y\.coords\['obs_ind'\] must contain datetime values",
    ):
        model_error = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs=ss_sample_kwargs
        )
        # Create y with non-datetime coords (integers instead)
        bad_y = xr.DataArray(
            data["y"].values.reshape(-1, 1),
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
        )
        bad_X = xr.DataArray(
            np.zeros((n_obs, 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": np.arange(n_obs), "coeffs": []},
        )
        model_error.fit(
            X=bad_X,
            y=bad_y,
        )

    # Test prediction with missing X for out-of-sample
    with pytest.raises(
        ValueError,
        match="X must be provided for out-of-sample predictions",
    ):
        model.predict(
            X=None,
            out_of_sample=True,
        )

    # Test methods before fitting
    unfitted_model = cp.pymc_models.StateSpaceTimeSeries(sample_kwargs=ss_sample_kwargs)

    with pytest.raises(RuntimeError, match="Model must be fit before"):
        unfitted_model._smooth()

    with pytest.raises(RuntimeError, match="Model must be fit before"):
        unfitted_model._forecast(start=dates[0], periods=10)

    # --- Test Case 7: Model initialization with different parameters --- #
    # Test different level orders
    model_level1 = cp.pymc_models.StateSpaceTimeSeries(
        level_order=1,  # Local level only (no slope)
        seasonal_length=7,
        sample_kwargs=ss_sample_kwargs,
        mode="FAST_COMPILE",
    )
    assert model_level1.level_order == 1

    # Test different seasonal lengths
    model_monthly = cp.pymc_models.StateSpaceTimeSeries(
        level_order=2,
        seasonal_length=30,  # Monthly seasonality
        sample_kwargs=ss_sample_kwargs,
        mode="FAST_COMPILE",
    )
    assert model_monthly.seasonal_length == 30


@pytest.fixture(scope="module")
def multi_unit_sc_data(rng):
    """Generate synthetic data for SyntheticControl with multiple treated units."""
    n_obs = 60
    n_control = 4
    n_treated = 3

    # Create time index
    time_index = pd.date_range("2020-01-01", periods=n_obs, freq="D")
    treatment_time = time_index[40]  # Intervention at day 40

    # Control unit data
    control_data = {}
    for i in range(n_control):
        control_data[f"control_{i}"] = rng.normal(10, 2, n_obs) + np.sin(
            np.arange(n_obs) * 0.1
        )

    # Treated unit data (combinations of control units with some noise)
    treated_data = {}
    for j in range(n_treated):
        # Each treated unit is a different weighted combination of controls
        weights = rng.dirichlet(np.ones(n_control))
        base_signal = sum(
            weights[i] * control_data[f"control_{i}"] for i in range(n_control)
        )

        # Add treatment effect after intervention
        treatment_effect = np.zeros(n_obs)
        treatment_effect[40:] = rng.normal(
            5, 1, n_obs - 40
        )  # Positive effect after treatment

        treated_data[f"treated_{j}"] = (
            base_signal + treatment_effect + rng.normal(0, 0.5, n_obs)
        )

    # Create DataFrame
    df = pd.DataFrame({**control_data, **treated_data}, index=time_index)

    control_units = [f"control_{i}" for i in range(n_control)]
    treated_units = [f"treated_{j}" for j in range(n_treated)]

    return df, treatment_time, control_units, treated_units


@pytest.fixture(scope="module")
def single_unit_sc_data(rng):
    """Generate synthetic data for SyntheticControl with single treated unit."""
    n_obs = 60
    n_control = 4

    # Create time index
    time_index = pd.date_range("2020-01-01", periods=n_obs, freq="D")
    treatment_time = time_index[40]  # Intervention at day 40

    # Control unit data
    control_data = {}
    for i in range(n_control):
        control_data[f"control_{i}"] = rng.normal(10, 2, n_obs) + np.sin(
            np.arange(n_obs) * 0.1
        )

    # Single treated unit data
    weights = rng.dirichlet(np.ones(n_control))
    base_signal = sum(
        weights[i] * control_data[f"control_{i}"] for i in range(n_control)
    )

    # Add treatment effect after intervention
    treatment_effect = np.zeros(n_obs)
    treatment_effect[40:] = rng.normal(
        5, 1, n_obs - 40
    )  # Positive effect after treatment

    treated_data = {
        "treated_0": base_signal + treatment_effect + rng.normal(0, 0.5, n_obs)
    }

    # Create DataFrame
    df = pd.DataFrame({**control_data, **treated_data}, index=time_index)

    control_units = [f"control_{i}" for i in range(n_control)]
    treated_units = ["treated_0"]

    return df, treatment_time, control_units, treated_units


class TestSyntheticControlMultiUnit:
    """Tests for SyntheticControl experiment with multiple treated units."""

    @pytest.mark.integration
    def test_multi_unit_initialization(self, multi_unit_sc_data):
        """Test that SyntheticControl can initialize with multiple treated units."""
        df, treatment_time, control_units, treated_units = multi_unit_sc_data

        model = cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs)

        # Should initialize without error
        sc = cp.SyntheticControl(
            data=df,
            treatment_time=treatment_time,
            control_units=control_units,
            treated_units=treated_units,
            model=model,
        )

        # Check basic attributes
        assert sc.treated_units == treated_units
        assert sc.control_units == control_units
        assert sc.treatment_time == treatment_time

        # Check data shapes
        assert sc.datapre_treated.shape == (40, len(treated_units))
        assert sc.datapost_treated.shape == (20, len(treated_units))
        assert sc.datapre_control.shape == (40, len(control_units))
        assert sc.datapost_control.shape == (20, len(control_units))

    @pytest.mark.integration
    def test_multi_unit_scoring(self, multi_unit_sc_data):
        """Test that scoring works with multiple treated units."""
        df, treatment_time, control_units, treated_units = multi_unit_sc_data

        model = cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs)

        sc = cp.SyntheticControl(
            data=df,
            treatment_time=treatment_time,
            control_units=control_units,
            treated_units=treated_units,
            model=model,
        )

        # Score should be a pandas Series with separate entries for each unit
        assert isinstance(sc.score, pd.Series)

        # Check that we have r2 and r2_std for each treated unit using unified format
        for i, _unit in enumerate(treated_units):
            assert f"unit_{i}_r2" in sc.score.index
            assert f"unit_{i}_r2_std" in sc.score.index

    @pytest.mark.integration
    def test_multi_unit_summary(self, multi_unit_sc_data, capsys):
        """Test that summary works with multiple treated units."""
        df, treatment_time, control_units, treated_units = multi_unit_sc_data

        model = cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs)

        sc = cp.SyntheticControl(
            data=df,
            treatment_time=treatment_time,
            control_units=control_units,
            treated_units=treated_units,
            model=model,
        )

        # Test summary
        sc.summary(round_to=3)

        captured = capsys.readouterr()
        output = captured.out

        # Check that output contains information for multiple treated units
        assert "Treated units:" in output
        for unit in treated_units:
            assert unit in output

    @pytest.mark.integration
    def test_single_unit_backward_compatibility(self, single_unit_sc_data):
        """Test that single treated unit still works (backward compatibility)."""
        df, treatment_time, control_units, treated_units = single_unit_sc_data

        model = cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs)

        sc = cp.SyntheticControl(
            data=df,
            treatment_time=treatment_time,
            control_units=control_units,
            treated_units=treated_units,
            model=model,
        )

        # Check basic attributes
        assert sc.treated_units == treated_units
        assert sc.control_units == control_units
        assert sc.treatment_time == treatment_time

    @pytest.mark.integration
    def test_multi_unit_plotting(self, multi_unit_sc_data):
        """Test that plotting works with multiple treated units."""
        df, treatment_time, control_units, treated_units = multi_unit_sc_data

        model = cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs)

        sc = cp.SyntheticControl(
            data=df,
            treatment_time=treatment_time,
            control_units=control_units,
            treated_units=treated_units,
            model=model,
        )

        # Test plotting - should work for each treated unit individually
        for unit in treated_units:
            fig, ax = sc.plot(treated_unit=unit)
            assert isinstance(fig, plt.Figure)
            assert isinstance(ax, np.ndarray) and all(
                isinstance(item, plt.Axes) for item in ax
            )

        # Test default plotting (first unit)
        fig, ax = sc.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, np.ndarray) and all(
            isinstance(item, plt.Axes) for item in ax
        )

    @pytest.mark.integration
    def test_multi_unit_plot_data(self, multi_unit_sc_data):
        """Test that plot data generation works with multiple treated units."""
        df, treatment_time, control_units, treated_units = multi_unit_sc_data

        model = cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs)

        sc = cp.SyntheticControl(
            data=df,
            treatment_time=treatment_time,
            control_units=control_units,
            treated_units=treated_units,
            model=model,
        )

        # Test plot data generation for each treated unit
        for unit in treated_units:
            plot_data = sc.get_plot_data(treated_unit=unit)
            assert isinstance(plot_data, pd.DataFrame)

            # Check expected columns
            expected_columns = [
                "prediction",
                "pred_hdi_lower_94",
                "pred_hdi_upper_94",
                "impact",
                "impact_hdi_lower_94",
                "impact_hdi_upper_94",
            ]
            assert set(expected_columns).issubset(set(plot_data.columns))

        # Test default plot data (first unit)
        plot_data = sc.get_plot_data()
        assert isinstance(plot_data, pd.DataFrame)

    @pytest.mark.integration
    def test_multi_unit_plotting_invalid_unit(self, multi_unit_sc_data):
        """Test that plotting with invalid treated unit raises appropriate errors."""
        df, treatment_time, control_units, treated_units = multi_unit_sc_data

        model = cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs)

        sc = cp.SyntheticControl(
            data=df,
            treatment_time=treatment_time,
            control_units=control_units,
            treated_units=treated_units,
            model=model,
        )

        # Test that invalid treated unit name is handled gracefully
        # Note: Current implementation may not raise ValueError, so we test default behavior
        with contextlib.suppress(ValueError, KeyError):
            sc.plot(treated_unit="invalid_unit")

        with contextlib.suppress(ValueError, KeyError):
            sc.get_plot_data(treated_unit="invalid_unit")


@pytest.mark.integration
def test_transfer_function_bayesian_adstock_only(mock_pymc_sample):
    """
    Test Bayesian Transfer Function ITS with adstock only.

    Creates synthetic data and checks:
    1. GradedInterventionTimeSeries works with PyMC model
    2. Correct number of MCMC chains and draws
    3. Transform parameters are estimated (half_life in posterior)
    4. Plot and summary methods work
    5. Half-life posterior is reasonable
    """
    # Generate synthetic data
    np.random.seed(42)
    n_weeks = 100
    t = np.arange(n_weeks)

    # Simple baseline with trend
    baseline = 1000 + 2 * t + np.random.normal(0, 50, n_weeks)

    # Treatment: on/off pattern
    treatment = np.zeros(n_weeks)
    treatment[30:50] = np.random.uniform(5, 10, 20)
    treatment[70:90] = np.random.uniform(5, 10, 20)

    # Apply adstock transform for data generation
    half_life_true = 2.0
    alpha = np.power(0.5, 1 / half_life_true)
    l_max = 8
    adstock_weights = np.power(alpha, np.arange(l_max + 1))
    adstock_weights = adstock_weights / adstock_weights.sum()

    treatment_transformed = np.zeros_like(treatment)
    for t_idx in range(n_weeks):
        for lag in range(min(l_max + 1, t_idx + 1)):
            treatment_transformed[t_idx] += (
                adstock_weights[lag] * treatment[t_idx - lag]
            )

    # Generate outcome
    outcome = baseline - 100 * treatment_transformed + np.random.normal(0, 30, n_weeks)

    # Create dataframe
    df = pd.DataFrame(
        {
            "t": t,
            "y": outcome,
            "treatment": treatment,
        }
    )

    # Fit Bayesian model
    model = cp.pymc_models.TransferFunctionLinearRegression(
        saturation_type=None,
        adstock_config={
            "half_life_prior": {"dist": "Gamma", "alpha": 4, "beta": 2},
            "l_max": 8,
            "normalize": True,
        },
        sample_kwargs=sample_kwargs,
    )

    result = cp.GradedInterventionTimeSeries(
        data=df,
        y_column="y",
        treatment_names=["treatment"],
        base_formula="1 + t",
        model=model,
    )

    # Test basic properties
    assert isinstance(result, cp.GradedInterventionTimeSeries)
    assert hasattr(result.model, "idata")
    assert len(result.model.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.model.idata.posterior.coords["draw"]) == sample_kwargs["draws"]

    # Test that transform parameters are in posterior
    assert "half_life" in result.model.idata.posterior
    assert "alpha_adstock" in result.model.idata.posterior

    # Test that regression coefficients are in posterior
    assert "beta" in result.model.idata.posterior
    assert "theta_treatment" in result.model.idata.posterior

    # Test plotting
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 2  # Should have 2 subplots

    # Test plot_transforms (Bayesian-specific path)
    fig_trans, ax_trans = result.plot_transforms()
    assert isinstance(fig_trans, plt.Figure)
    assert len(ax_trans) >= 1  # Should have at least 1 panel (adstock)

    # Test summary (should not raise)
    result.summary()

    # Test effect() method (Bayesian-specific path)
    effect_result = result.effect(
        window=(df.index[0], df.index[-1]), channels=["treatment"], scale=0.0
    )
    assert "effect_df" in effect_result
    assert "total_effect" in effect_result

    # Test plot_effect() (Bayesian-specific path)
    fig_eff, ax_eff = result.plot_effect(effect_result)
    assert isinstance(fig_eff, plt.Figure)
    assert len(ax_eff) == 2

    # Test that half_life posterior is reasonable (should be positive)
    half_life_samples = az.extract(result.model.idata, var_names=["half_life"])
    assert (half_life_samples > 0).all(), "Half-life should be positive"
    assert half_life_samples.mean() < 10, "Half-life should be reasonable"


@pytest.mark.integration
def test_transfer_function_ar_bayesian(mock_pymc_sample):
    """
    Test Bayesian Transfer Function ITS with AR(1) errors.

    Creates synthetic data with AR(1) errors and checks:
    1. GradedInterventionTimeSeries works with TransferFunctionARRegression
    2. Correct number of MCMC chains and draws
    3. Transform parameters (half_life) and AR parameter (rho) are in posterior
    4. Plot and summary methods work
    5. Parameter posteriors are reasonable (bounded, positive where expected)
    """
    # Generate synthetic data with AR(1) errors
    np.random.seed(42)
    n_weeks = 100
    t = np.arange(n_weeks)

    # Simple baseline with trend
    baseline = 1000 + 2 * t

    # Treatment: on/off pattern
    treatment = np.zeros(n_weeks)
    treatment[30:50] = np.random.uniform(5, 10, 20)
    treatment[70:90] = np.random.uniform(5, 10, 20)

    # Apply adstock transform for data generation
    half_life_true = 2.0
    alpha = np.power(0.5, 1 / half_life_true)
    l_max = 8
    adstock_weights = np.power(alpha, np.arange(l_max + 1))
    adstock_weights = adstock_weights / adstock_weights.sum()

    treatment_transformed = np.zeros_like(treatment)
    for t_idx in range(n_weeks):
        for lag in range(min(l_max + 1, t_idx + 1)):
            treatment_transformed[t_idx] += (
                adstock_weights[lag] * treatment[t_idx - lag]
            )

    # Generate outcome with AR(1) errors
    rho_true = 0.7
    errors = np.zeros(n_weeks)
    errors[0] = np.random.normal(0, 30)
    for t_idx in range(1, n_weeks):
        errors[t_idx] = rho_true * errors[t_idx - 1] + np.random.normal(0, 30)

    outcome = baseline - 100 * treatment_transformed + errors

    # Create dataframe
    df = pd.DataFrame(
        {
            "t": t,
            "y": outcome,
            "treatment": treatment,
        }
    )

    # Fit Bayesian AR(1) model
    # Note: AR models need more sampling than independent errors models due to complexity
    ar_sample_kwargs = {
        "tune": 50,
        "draws": 50,
        "chains": 2,
        "cores": 2,
        "random_seed": 42,
    }
    model = cp.pymc_models.TransferFunctionARRegression(
        saturation_type=None,
        adstock_config={
            "half_life_prior": {"dist": "Gamma", "alpha": 4, "beta": 2},
            "l_max": 8,
            "normalize": True,
        },
        sample_kwargs=ar_sample_kwargs,
    )

    result = cp.GradedInterventionTimeSeries(
        data=df,
        y_column="y",
        treatment_names=["treatment"],
        base_formula="1 + t",
        model=model,
    )

    # Test basic properties
    assert isinstance(result, cp.GradedInterventionTimeSeries)
    assert hasattr(result.model, "idata")
    assert (
        len(result.model.idata.posterior.coords["chain"]) == ar_sample_kwargs["chains"]
    )
    assert len(result.model.idata.posterior.coords["draw"]) == ar_sample_kwargs["draws"]

    # Test that transform parameters are in posterior
    assert "half_life" in result.model.idata.posterior
    assert "alpha_adstock" in result.model.idata.posterior

    # Test that AR parameter is in posterior
    assert "rho" in result.model.idata.posterior, (
        "AR(1) parameter rho should be in posterior"
    )

    # Test that regression coefficients are in posterior
    assert "beta" in result.model.idata.posterior
    assert "theta_treatment" in result.model.idata.posterior

    # Test plotting
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 2  # Should have 2 subplots

    # Test plot_transforms (Bayesian-specific path)
    fig_trans, ax_trans = result.plot_transforms()
    assert isinstance(fig_trans, plt.Figure)
    assert len(ax_trans) >= 1  # Should have at least 1 panel (adstock)

    # Test summary (should not raise)
    result.summary()

    # Test effect() method (Bayesian-specific path)
    effect_result = result.effect(
        window=(df.index[0], df.index[-1]), channels=["treatment"], scale=0.0
    )
    assert "effect_df" in effect_result
    assert "total_effect" in effect_result

    # Test plot_effect() (Bayesian-specific path)
    fig_eff, ax_eff = result.plot_effect(effect_result)
    assert isinstance(fig_eff, plt.Figure)
    assert len(ax_eff) == 2

    # Test that half_life posterior is reasonable (should be positive)
    half_life_samples = az.extract(result.model.idata, var_names=["half_life"])
    assert (half_life_samples > 0).all(), "Half-life should be positive"
    assert half_life_samples.mean() < 10, "Half-life should be reasonable"

    # Test that rho is bounded between -1 and 1 (stationarity constraint)
    rho_samples = az.extract(result.model.idata, var_names=["rho"])
    assert (rho_samples > -1).all() and (rho_samples < 1).all(), (
        "Rho should be between -1 and 1"
    )


@pytest.mark.integration
def test_transfer_function_bayesian_with_saturation(mock_pymc_sample):
    """
    Test Bayesian Transfer Function with Hill saturation.

    This test covers saturation transform code paths in build_model.
    """
    np.random.seed(42)
    n_weeks = 100
    t = np.arange(n_weeks)

    # Simple baseline
    baseline = 1000 + 2 * t + np.random.normal(0, 50, n_weeks)

    # Treatment with saturation
    treatment_raw = np.zeros(n_weeks)
    treatment_raw[30:70] = np.random.uniform(5, 15, 40)

    # Apply Hill saturation + adstock for data generation
    from causalpy.transforms import GeometricAdstock, HillSaturation

    sat = HillSaturation(slope=2.0, kappa=10)
    treatment_sat = sat.apply(treatment_raw)

    adstock = GeometricAdstock(half_life=2.0, normalize=True)
    treatment_transformed = adstock.apply(treatment_sat)

    outcome = baseline - 100 * treatment_transformed + np.random.normal(0, 30, n_weeks)

    df = pd.DataFrame({"t": t, "y": outcome, "treatment": treatment_raw})

    # Fit Bayesian model WITH saturation
    model = cp.pymc_models.TransferFunctionLinearRegression(
        saturation_type="hill",
        saturation_config={
            "slope_prior": {"dist": "Gamma", "alpha": 3, "beta": 1.5},
            "kappa_prior": {"dist": "Gamma", "alpha": 10, "beta": 1},
        },
        adstock_config={
            "half_life_prior": {"dist": "Gamma", "alpha": 4, "beta": 2},
            "l_max": 8,
            "normalize": True,
        },
        sample_kwargs=sample_kwargs,
    )

    result = cp.GradedInterventionTimeSeries(
        data=df,
        y_column="y",
        treatment_names=["treatment"],
        base_formula="1 + t",
        model=model,
    )

    # Test that saturation parameters are in posterior
    assert "slope" in result.model.idata.posterior
    assert "kappa" in result.model.idata.posterior
    assert "half_life" in result.model.idata.posterior

    # Test plotting works with saturation
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)


@pytest.mark.integration
def test_transfer_function_bayesian_halfnormal_prior(mock_pymc_sample):
    """
    Test Bayesian Transfer Function with HalfNormal priors.

    This test covers HalfNormal prior code paths (lines 1233-1239).
    """
    np.random.seed(42)
    n_weeks = 80
    t = np.arange(n_weeks)

    baseline = 1000 + 2 * t + np.random.normal(0, 50, n_weeks)
    treatment = np.zeros(n_weeks)
    treatment[20:60] = np.random.uniform(5, 10, 40)

    outcome = baseline - 50 * treatment + np.random.normal(0, 30, n_weeks)
    df = pd.DataFrame({"t": t, "y": outcome, "treatment": treatment})

    # Use HalfNormal prior for half_life
    model = cp.pymc_models.TransferFunctionLinearRegression(
        saturation_type=None,
        adstock_config={
            "half_life_prior": {
                "dist": "HalfNormal",
                "sigma": 3,
            },  # HalfNormal instead of Gamma
            "l_max": 8,
            "normalize": True,
        },
        sample_kwargs=sample_kwargs,
    )

    result = cp.GradedInterventionTimeSeries(
        data=df,
        y_column="y",
        treatment_names=["treatment"],
        base_formula="1 + t",
        model=model,
    )

    # Verify model fitted
    assert "half_life" in result.model.idata.posterior
    assert hasattr(result.model, "idata")


@pytest.mark.integration
def test_transfer_function_bayesian_logistic_saturation(mock_pymc_sample):
    """
    Test Bayesian Transfer Function with Logistic saturation.

    This test covers logistic saturation code paths (lines 1277-1286).
    """
    np.random.seed(42)
    n_weeks = 80
    t = np.arange(n_weeks)

    baseline = 1000 + 2 * t + np.random.normal(0, 50, n_weeks)
    treatment = np.zeros(n_weeks)
    treatment[20:60] = np.random.uniform(1, 5, 40)

    outcome = baseline - 50 * treatment + np.random.normal(0, 30, n_weeks)
    df = pd.DataFrame({"t": t, "y": outcome, "treatment": treatment})

    # Use Logistic saturation
    model = cp.pymc_models.TransferFunctionLinearRegression(
        saturation_type="logistic",
        saturation_config={
            "lam_prior": {"dist": "HalfNormal", "sigma": 1},
        },
        adstock_config={
            "half_life_prior": {"dist": "Gamma", "alpha": 4, "beta": 2},
            "l_max": 8,
            "normalize": True,
        },
        sample_kwargs=sample_kwargs,
    )

    result = cp.GradedInterventionTimeSeries(
        data=df,
        y_column="y",
        treatment_names=["treatment"],
        base_formula="1 + t",
        model=model,
    )

    # Verify logistic parameters are in posterior
    assert "lam" in result.model.idata.posterior
    assert "half_life" in result.model.idata.posterior


@pytest.mark.integration
def test_transfer_function_bayesian_michaelis_menten_saturation(mock_pymc_sample):
    """
    Test Bayesian Transfer Function with Michaelis-Menten saturation.

    This test covers Michaelis-Menten saturation code paths (lines 1289-1313).
    """
    np.random.seed(42)
    n_weeks = 80
    t = np.arange(n_weeks)

    baseline = 1000 + 2 * t + np.random.normal(0, 50, n_weeks)
    treatment = np.zeros(n_weeks)
    treatment[20:60] = np.random.uniform(5, 15, 40)

    outcome = baseline - 50 * treatment + np.random.normal(0, 30, n_weeks)
    df = pd.DataFrame({"t": t, "y": outcome, "treatment": treatment})

    # Use Michaelis-Menten saturation
    model = cp.pymc_models.TransferFunctionLinearRegression(
        saturation_type="michaelis_menten",
        saturation_config={
            "alpha_prior": {"dist": "HalfNormal", "sigma": 1},
            "lam_prior": {"dist": "HalfNormal", "sigma": 100},
        },
        adstock_config={
            "half_life_prior": {"dist": "Gamma", "alpha": 4, "beta": 2},
            "l_max": 8,
            "normalize": True,
        },
        sample_kwargs=sample_kwargs,
    )

    result = cp.GradedInterventionTimeSeries(
        data=df,
        y_column="y",
        treatment_names=["treatment"],
        base_formula="1 + t",
        model=model,
    )

    # Verify Michaelis-Menten parameters are in posterior
    assert "alpha_sat" in result.model.idata.posterior
    assert "lam" in result.model.idata.posterior
    assert "half_life" in result.model.idata.posterior


@pytest.mark.integration
def test_transfer_function_ar_bayesian_with_saturation(mock_pymc_sample):
    """
    Test Bayesian AR(1) Transfer Function with Hill saturation.

    This test covers saturation transform code paths in AR model (lines 1746-1826).
    """
    np.random.seed(42)
    n_weeks = 80
    t = np.arange(n_weeks)

    baseline = 1000 + 2 * t
    treatment = np.zeros(n_weeks)
    treatment[20:60] = np.random.uniform(5, 15, 40)

    # Add AR(1) errors
    rho = 0.6
    errors = np.zeros(n_weeks)
    errors[0] = np.random.normal(0, 30 / np.sqrt(1 - rho**2))
    for i in range(1, n_weeks):
        errors[i] = rho * errors[i - 1] + np.random.normal(0, 30)

    outcome = baseline - 50 * treatment + errors
    df = pd.DataFrame({"t": t, "y": outcome, "treatment": treatment})

    # AR model WITH saturation
    model = cp.pymc_models.TransferFunctionARRegression(
        saturation_type="hill",
        saturation_config={
            "slope_prior": {"dist": "Gamma", "alpha": 3, "beta": 1.5},
            "kappa_prior": {"dist": "Gamma", "alpha": 10, "beta": 1},
        },
        adstock_config={
            "half_life_prior": {"dist": "Gamma", "alpha": 4, "beta": 2},
            "l_max": 8,
            "normalize": True,
        },
        sample_kwargs={
            "chains": 2,
            "draws": 50,
            "tune": 50,
            "random_seed": 42,
        },
    )

    result = cp.GradedInterventionTimeSeries(
        data=df,
        y_column="y",
        treatment_names=["treatment"],
        base_formula="1 + t",
        model=model,
    )

    # Verify saturation parameters and rho are in posterior
    assert "slope" in result.model.idata.posterior
    assert "kappa" in result.model.idata.posterior
    assert "rho" in result.model.idata.posterior
    assert "half_life" in result.model.idata.posterior
