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
import pymc as pm
import pytensor.tensor as pt
import pytest
from matplotlib import pyplot as plt

import causalpy as cp

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


@pytest.mark.integration
def test_did():
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
def test_did_banks_simple():
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
def test_did_banks_multi():
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
def test_rd():
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
def test_rd_bandwidth():
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
def test_rd_drinking():
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
def test_rkink():
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
def test_rkink_bandwidth():
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
def test_its():
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
def test_its_covid():
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
def test_sc():
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
        formula="actual ~ 0 + a + b + c + d + e + f + g",
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
def test_sc_brexit():
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
    formula = target_country + " ~ " + "0 + " + " + ".join(other_countries)
    result = cp.SyntheticControl(
        df,
        treatment_time,
        formula=formula,
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
def test_ancova():
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
def test_geolift1():
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
        formula="""Denmark ~ 0 + Austria + Belgium + Bulgaria + Croatia + Cyprus
        + Czech_Republic""",
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
def test_iv_reg():
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
def test_inverse_prop():
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


# DEPRECATION WARNING TESTS ============================================================


def test_did_deprecation_warning():
    """Test that the old DifferenceInDifferences class raises a deprecation warning."""

    with pytest.warns(DeprecationWarning):
        df = cp.load_data("did")
        result = cp.pymc_experiments.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group*post_treatment",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )
        assert isinstance(result, cp.DifferenceInDifferences)


def test_rd_deprecation_warning():
    """Test that the old RegressionDiscontinuity class raises a deprecation warning."""

    with pytest.warns(DeprecationWarning):
        df = cp.load_data("rd")
        result = cp.pymc_experiments.RegressionDiscontinuity(
            df,
            formula="y ~ 1 + bs(x, df=6) + treated",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
            treatment_threshold=0.5,
            epsilon=0.001,
        )
        assert isinstance(result, cp.RegressionDiscontinuity)


def test_rk_deprecation_warning():
    """Test that the old RegressionKink class raises a deprecation warning."""

    with pytest.warns(DeprecationWarning):
        kink = 0.5
        df = setup_regression_kink_data(kink)
        result = cp.pymc_experiments.RegressionKink(
            df,
            formula=f"y ~ 1 + x + I((x-{kink})*treated)",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
            kink_point=kink,
        )
        assert isinstance(result, cp.RegressionKink)


def test_its_deprecation_warning():
    """Test that the old InterruptedTimeSeries class raises a deprecation warning."""

    with pytest.warns(DeprecationWarning):
        df = (
            cp.load_data("its")
            .assign(date=lambda x: pd.to_datetime(x["date"]))
            .set_index("date")
        )
        treatment_time = pd.to_datetime("2017-01-01")
        result = cp.pymc_experiments.InterruptedTimeSeries(
            df,
            treatment_time,
            formula="y ~ 1 + t + C(month)",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )
        assert isinstance(result, cp.InterruptedTimeSeries)


def test_sc_deprecation_warning():
    """Test that the old SyntheticControl class raises a deprecation warning."""

    with pytest.warns(DeprecationWarning):
        df = cp.load_data("sc")
        treatment_time = 70
        result = cp.pymc_experiments.SyntheticControl(
            df,
            treatment_time,
            formula="actual ~ 0 + a + b + c + d + e + f + g",
            model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
        )
        assert isinstance(result, cp.SyntheticControl)


def test_ancova_deprecation_warning():
    """Test that the old PrePostNEGD class raises a deprecation warning."""

    with pytest.warns(DeprecationWarning):
        df = cp.load_data("anova1")
        result = cp.pymc_experiments.PrePostNEGD(
            df,
            formula="post ~ 1 + C(group) + pre",
            group_variable_name="group",
            pretreatment_variable_name="pre",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )
        assert isinstance(result, cp.PrePostNEGD)


def test_iv_deprecation_warning():
    """Test that the old InstrumentalVariable class raises a deprecation warning."""

    with pytest.warns(DeprecationWarning):
        df = cp.load_data("risk")
        instruments_formula = "risk  ~ 1 + logmort0"
        formula = "loggdp ~  1 + risk"
        instruments_data = df[["risk", "logmort0"]]
        data = df[["loggdp", "risk"]]
        result = cp.pymc_experiments.InstrumentalVariable(
            instruments_data=instruments_data,
            data=data,
            instruments_formula=instruments_formula,
            formula=formula,
            model=cp.pymc_models.InstrumentalVariableRegression(
                sample_kwargs=sample_kwargs
            ),
        )
        assert isinstance(result, cp.InstrumentalVariable)


@pytest.mark.integration
def test_bayesian_structural_time_series():
    """Test the BayesianStructuralTimeSeries model."""
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
    )  # noqa E501
    y_values_no_x = trend_actual + seasonality_actual + noise_actual

    data_with_x = pd.DataFrame({"y": y_values_with_x, "x1": x1_actual}, index=dates)
    data_no_x = pd.DataFrame({"y": y_values_no_x}, index=dates)

    # Prepare time features for the model
    day_of_year = dates.dayofyear.to_numpy()
    time_numeric = (dates - dates[0]).days.to_numpy() / 365.25

    # Define sample_kwargs for speed
    bsts_sample_kwargs = {
        "chains": 1,
        "draws": 100,
        "tune": 50,
        "progressbar": False,
        "random_seed": 42,  # noqa E501
    }

    # --- Test Case 1: Model with exogenous regressor --- #
    coords_with_x = {
        "obs_ind": np.arange(n_obs),
        "coeffs": ["x1"],
        "time_for_seasonality": day_of_year,
        "time_for_trend": time_numeric,
    }
    model_with_x = cp.pymc_models.BayesianStructuralTimeSeries(
        n_order=2, n_changepoints_trend=5, sample_kwargs=bsts_sample_kwargs
    )
    model_with_x.fit(
        X=data_with_x[["x1"]],
        y=data_with_x["y"].values.reshape(-1, 1),
        coords=coords_with_x,
    )
    assert isinstance(model_with_x.idata, az.InferenceData)
    assert "posterior" in model_with_x.idata
    assert "beta" in model_with_x.idata.posterior  # For exogenous regressor
    assert "fourier_beta" in model_with_x.idata.posterior  # Corrected name
    assert "delta" in model_with_x.idata.posterior  # Corrected name
    assert "sigma" in model_with_x.idata.posterior
    assert "mu" in model_with_x.idata.posterior_predictive
    assert "y_hat" in model_with_x.idata.posterior_predictive

    # Test predict and score
    predictions_with_x = model_with_x.predict(
        X=data_with_x[["x1"]],
        time_for_trend_pred=time_numeric,
        time_for_seasonality_pred=day_of_year,
    )
    assert "y_hat" in predictions_with_x.posterior_predictive
    assert "mu" in predictions_with_x.posterior_predictive
    assert predictions_with_x.posterior_predictive["y_hat"].shape == (
        bsts_sample_kwargs["chains"],
        bsts_sample_kwargs["draws"],
        n_obs,
    )  # noqa E501
    score_with_x = model_with_x.score(
        X=data_with_x[["x1"]],
        y=data_with_x["y"].values,
        time_for_trend_pred=time_numeric,
        time_for_seasonality_pred=day_of_year,
    )
    assert isinstance(score_with_x, pd.Series)
    assert "r2" in score_with_x

    # --- Test Case 2: Model without exogenous regressor (X=None) --- #
    coords_no_x = {
        "obs_ind": np.arange(n_obs),
        "time_for_seasonality": day_of_year,
        "time_for_trend": time_numeric,
    }
    model_no_x = cp.pymc_models.BayesianStructuralTimeSeries(
        n_order=2, n_changepoints_trend=5, sample_kwargs=bsts_sample_kwargs
    )
    # Fit with X=None
    model_no_x.fit(
        X=None,  # Explicitly None
        y=data_no_x["y"].values.reshape(-1, 1),
        coords=coords_no_x,
    )
    assert isinstance(model_no_x.idata, az.InferenceData)
    assert "beta" not in model_no_x.idata.posterior  # No exogenous regressor beta
    assert "fourier_beta" in model_no_x.idata.posterior  # Corrected name
    assert "delta" in model_no_x.idata.posterior  # Corrected name

    # Test predict and score (X=None)
    # The _data_setter will raise ValueError if model has X and X_pred is None.
    # But if model was built with X=None, _data_setter should not expect X_pred.
    predictions_no_x = model_no_x.predict(
        X=None,  # X=None for predict
        time_for_trend_pred=time_numeric,
        time_for_seasonality_pred=day_of_year,
    )
    assert "y_hat" in predictions_no_x.posterior_predictive
    score_no_x = model_no_x.score(
        X=None,
        y=data_no_x["y"].values,  # X=None for score
        time_for_trend_pred=time_numeric,
        time_for_seasonality_pred=day_of_year,
    )
    assert isinstance(score_no_x, pd.Series)

    # --- Test Case 3: Model with empty exogenous regressor (X with shape (n,0)) --- #
    # This case represents e.g. `y ~ 0 + trend + season` if patsy produced X with 0 cols
    # For build_model, X.shape[1] == 0 means no beta is created.
    model_empty_x = cp.pymc_models.BayesianStructuralTimeSeries(
        n_order=2, n_changepoints_trend=5, sample_kwargs=bsts_sample_kwargs
    )
    empty_x_array = np.empty((n_obs, 0))
    model_empty_x.fit(
        X=empty_x_array,  # X with zero columns
        y=data_no_x["y"].values.reshape(-1, 1),
        coords=coords_no_x,  # No "coeffs" needed as X has no columns
    )
    assert isinstance(model_empty_x.idata, az.InferenceData)
    assert "beta" not in model_empty_x.idata.posterior

    # Predict with empty X array
    predictions_empty_x = model_empty_x.predict(
        X=empty_x_array,
        time_for_trend_pred=time_numeric,
        time_for_seasonality_pred=day_of_year,
    )
    assert "y_hat" in predictions_empty_x.posterior_predictive
    score_empty_x = model_empty_x.score(
        X=empty_x_array,
        y=data_no_x["y"].values,
        time_for_trend_pred=time_numeric,
        time_for_seasonality_pred=day_of_year,
    )
    assert isinstance(score_empty_x, pd.Series)

    # --- Test Case 4: Error handling for missing coords --- #
    incomplete_coords = {"obs_ind": np.arange(n_obs)}
    model_err = cp.pymc_models.BayesianStructuralTimeSeries(
        sample_kwargs=bsts_sample_kwargs
    )  # noqa E501
    with pytest.raises(ValueError, match="'time_for_trend' must be provided"):
        model_err.build_model(
            X=None, y=data_no_x["y"].values.reshape(-1, 1), coords=incomplete_coords
        )  # noqa E501

    incomplete_coords_trend = {
        "obs_ind": np.arange(n_obs),
        "time_for_trend": time_numeric,
    }  # noqa E501
    with pytest.raises(ValueError, match="'time_for_seasonality' must be provided"):
        model_err.build_model(
            X=None,
            y=data_no_x["y"].values.reshape(-1, 1),
            coords=incomplete_coords_trend,
        )  # noqa E501

    coords_for_x_no_coeffs = {
        "obs_ind": np.arange(n_obs),
        "time_for_seasonality": day_of_year,
        "time_for_trend": time_numeric,
        # No "coeffs"
    }
    with pytest.raises(
        ValueError, match="'coeffs' must be provided in coords when X is not None"
    ):
        model_err.build_model(
            X=data_with_x[["x1"]],
            y=data_with_x["y"].values.reshape(-1, 1),
            coords=coords_for_x_no_coeffs,
        )  # noqa E501

    # Test _data_setter error when model has X, but X_pred is None
    with pytest.raises(
        ValueError,
        match="Model was built with exogenous variable X. New X data \\(X_pred\\) must be provided for prediction, not None.",
    ):
        model_with_x.predict(
            X=None,
            time_for_trend_pred=time_numeric,
            time_for_seasonality_pred=day_of_year,
        )

    # --- Test Case 5: Custom components and ImportError handling --- #
    # Define dummy custom components satisfying the expected interface
    class CustomTrendComponent:
        def __init__(self, name_prefix="custom_trend"):
            self.name_prefix = name_prefix

        def _build_components(self, X) -> pt.TensorVariable:
            # Minimalistic trend: a simple learnable slope
            # Ensure this is compatible with how LinearTrend from pymc-marketing builds
            # its parameters (e.g. `delta` and `k`).
            # For this test, we'll keep it simple and ensure variable names don't clash
            # or are expected if we were to check for them.
            # A real custom component would have its own PyMC variables.
            custom_slope = pm.Normal(f"{self.name_prefix}_slope", mu=0, sigma=1)
            return X * custom_slope

        def apply(self, X) -> pt.TensorVariable:
            return self._build_components(X)

    class CustomSeasonalityComponent:
        def __init__(self, name_prefix="custom_season"):
            self.name_prefix = name_prefix

        def _build_components(self, X) -> pt.TensorVariable:
            # Minimalistic seasonality: a simple learnable offset
            # Similar to trend, a real one would be more complex.
            custom_offset = pm.Normal(f"{self.name_prefix}_offset", mu=0, sigma=1)
            # X here would be day_of_year or similar, but for this dummy, just use it
            # to ensure the shape is broadcastable if X is scalar for offset.
            # Or, make it independent of X if it's just an offset for all time points.
            return pm.math.zeros_like(X) + custom_offset  # Make it broadcast

        def apply(self, X) -> pt.TensorVariable:
            return self._build_components(X)

    # Test with custom trend only
    custom_trend = CustomTrendComponent()
    model_custom_trend = cp.pymc_models.BayesianStructuralTimeSeries(
        trend_component=custom_trend,  # Corrected parameter name
        sample_kwargs=bsts_sample_kwargs,
    )
    model_custom_trend.fit(
        X=None, y=data_no_x["y"].values.reshape(-1, 1), coords=coords_no_x
    )
    assert "custom_trend_slope" in model_custom_trend.idata.posterior
    assert "fourier_beta" in model_custom_trend.idata.posterior  # Default seasonality

    # Test with custom seasonality only
    custom_season = CustomSeasonalityComponent()
    model_custom_season = cp.pymc_models.BayesianStructuralTimeSeries(
        seasonality_component=custom_season,  # Corrected parameter name
        sample_kwargs=bsts_sample_kwargs,
    )
    model_custom_season.fit(
        X=None, y=data_no_x["y"].values.reshape(-1, 1), coords=coords_no_x
    )
    assert "custom_season_offset" in model_custom_season.idata.posterior
    assert "delta" in model_custom_season.idata.posterior  # Default trend

    # Test with both custom trend and seasonality
    model_both_custom = cp.pymc_models.BayesianStructuralTimeSeries(
        trend_component=custom_trend,  # Corrected parameter name
        seasonality_component=custom_season,  # Corrected parameter name
        sample_kwargs=bsts_sample_kwargs,
    )
    model_both_custom.fit(
        X=None, y=data_no_x["y"].values.reshape(-1, 1), coords=coords_no_x
    )
    assert "custom_trend_slope" in model_both_custom.idata.posterior
    assert "custom_season_offset" in model_both_custom.idata.posterior
    assert "fourier_beta" not in model_both_custom.idata.posterior
    assert "delta" not in model_both_custom.idata.posterior
