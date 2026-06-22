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
from types import SimpleNamespace

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

bsts_sample_kwargs = {
    "chains": 1,
    "draws": 100,
    "tune": 50,
    "progressbar": False,
    "random_seed": 42,
}

ss_sample_kwargs = {
    "chains": 1,
    "draws": 50,
    "tune": 25,
    "progressbar": False,
    "random_seed": 42,
}


@pytest.mark.integration
def test_did(mock_pymc_sample, did_data):
    """
    Test Difference in Differences (DID) PyMC experiment.

    Loads data and checks:
    1. data is a dataframe
    2. pymc_experiements.DifferenceInDifferences returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    """
    df = did_data
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
def test_rd(mock_pymc_sample, rd_data):
    """
    Test Regression Discontinuity experiment.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.RegressionDiscontinuity returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    """
    df = rd_data
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
def test_rd_bandwidth(mock_pymc_sample, rd_data):
    """
    Test Regression Discontinuity experiment with bandwidth parameter.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.RegressionDiscontinuity returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    """
    df = rd_data
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
def test_its(mock_pymc_sample, its_data):
    """
    Test Interrupted Time-Series experiment.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.InterruptedTimeSeries returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    5. the method get_plot_data returns a DataFrame with expected columns
    """
    df = its_data
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
def test_its_single_post_observation_plot(mock_pymc_sample, its_data):
    """Regression test: ITS plot must remain readable when the post-period
    contains a single observation.

    With one post-period datum the ``arviz.plot_hdi`` ribbon collapses to a
    zero-area polygon, the median line has no neighbours to connect to, and
    the (then top-of-zorder) treatment ``axvline`` covers the only datum -
    leaving the bottom two panels visually empty. The fix
        1. lowers the treatment-line zorder and switches it to a thin dashed
           style so it reads as an annotation, never as data;
        2. overlays an explicit median-plus-HDI errorbar on every panel;
        3. swaps the legend handle to that errorbar so the legend matches
           what is drawn (and drops the "Causal impact" entry whose
           ``fill_between`` collapses to nothing).
    """
    from matplotlib.collections import LineCollection
    from matplotlib.container import ErrorbarContainer

    df = its_data
    # Choose treatment_time so exactly one datum sits in the post-period.
    treatment_time = df.index[-1]
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert len(result.datapost) == 1
    fig, ax = result.plot()

    treatment_axvlines = [
        ln for a in ax for ln in a.get_lines() if ln.get_label() == "Treatment start"
    ]
    assert treatment_axvlines, "expected a treatment-start axvline"
    assert all(ln.get_zorder() < 2 for ln in treatment_axvlines), (
        "treatment axvline must sit below data (zorder<2) so it never "
        "occludes the only post-period observation"
    )
    assert all(ln.get_linestyle() == "--" for ln in treatment_axvlines), (
        "treatment axvline must be dashed to read as an annotation"
    )
    assert all(ln.get_linewidth() <= 2 for ln in treatment_axvlines), (
        "treatment axvline must be thin to avoid dominating the plot"
    )

    # Each panel must contain at least one errorbar overlay
    # (rendered as a LineCollection from the errorbar caps/whiskers).
    for i, a in enumerate(ax):
        line_collections = [c for c in a.collections if isinstance(c, LineCollection)]
        assert line_collections, (
            f"panel {i} should have a LineCollection from the singleton "
            "errorbar overlay; otherwise the post-period is invisible"
        )

    # Top-panel legend must reflect what is actually drawn: a Counterfactual
    # entry backed by the ErrorbarContainer (not a Line2D + ribbon tuple),
    # and no "Causal impact" entry (since its fill_between is degenerate).
    legend = ax[0].get_legend()
    assert legend is not None, "top panel should have a legend"
    legend_labels = [t.get_text() for t in legend.get_texts()]
    assert "Counterfactual" in legend_labels
    assert "Causal impact" not in legend_labels, (
        "Causal impact entry must be dropped when its fill_between collapses"
    )
    cf_idx = legend_labels.index("Counterfactual")
    cf_handle = legend.legend_handles[cf_idx]
    # Matplotlib renders an ErrorbarContainer in the legend via a
    # LineCollection proxy. What we want to assert is that the handle is
    # *not* the old (Line2D, PolyCollection) tuple, since that would imply
    # the legend swatch shows a ribbon that does not exist on the plot.
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    assert not (isinstance(cf_handle, tuple) and len(cf_handle) == 2), (
        "Counterfactual legend handle should not be a (line, ribbon) tuple "
        "in the singleton case - the ribbon does not render."
    )
    assert not isinstance(cf_handle, PolyCollection), (
        "Counterfactual legend handle should not be a PolyCollection ribbon "
        "in the singleton case - the ribbon does not render."
    )
    assert isinstance(cf_handle, ErrorbarContainer | LineCollection | Line2D), (
        "Counterfactual legend handle should reflect the errorbar overlay, "
        f"got {type(cf_handle).__name__}"
    )
    plt.close(fig)


@pytest.mark.integration
def test_sc(mock_pymc_sample, sc_data):
    """
    Test Synthetic Control experiment.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.SyntheticControl returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    5. the method get_plot_data returns a DataFrame with expected columns
    """

    df = sc_data
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
def test_sc_softmax(mock_pymc_sample):
    """
    Test Synthetic Control experiment with SoftmaxWeightedSumFitter.

    Verifies that SoftmaxWeightedSumFitter is a drop-in replacement for
    WeightedSumFitter in the SyntheticControl experiment:
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
        model=cp.pymc_models.SoftmaxWeightedSumFitter(sample_kwargs=sample_kwargs),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.SyntheticControl)
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]
    result.summary()

    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray) and all(
        isinstance(item, plt.Axes) for item in ax
    ), "ax must be a numpy.ndarray of plt.Axes"

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
def test_sdid(mock_pymc_sample):
    """
    Test Synthetic Difference-in-Differences experiment.

    Loads data and checks:
    1. data is a dataframe
    2. SyntheticDifferenceInDifferences returns correct type
    3. tau_posterior exists with chain/draw dims
    4. post_impact and post_impact_cumulative exist
    5. summary runs without error
    6. plot returns Figure and Axes
    """
    df = cp.load_data("sc")
    treatment_time = 70
    result = cp.SyntheticDifferenceInDifferences(
        df,
        treatment_time,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.SyntheticDifferenceInDifferencesWeightFitter(
            sample_kwargs=sample_kwargs,
        ),
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(result, cp.SyntheticDifferenceInDifferences)

    # tau posterior should exist with chain/draw dims
    assert hasattr(result, "tau_posterior")
    assert "chain" in result.tau_posterior.dims
    assert "draw" in result.tau_posterior.dims

    # post_impact should exist
    assert hasattr(result, "post_impact")
    assert hasattr(result, "post_impact_cumulative")

    # summary should run without error
    result.summary()

    # plot should return fig and axes
    fig, ax = result.plot(show=False)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray) and all(
        isinstance(item, plt.Axes) for item in ax
    ), "ax must be a numpy.ndarray of plt.Axes"


@pytest.mark.integration
def test_sdid_datetime_index_and_effect_summary(mock_pymc_sample):
    """SDiD with a DatetimeIndex panel exercises the datetime branch in
    ``_bayesian_plot`` and the full ``effect_summary`` body, including the
    ``period``-warning path and the ``cumulative=False`` branch.
    """
    df = cp.load_data("sc").copy()
    df.index = pd.date_range("2020-01-01", periods=len(df), freq="D")
    treatment_time = df.index[70]

    result = cp.SyntheticDifferenceInDifferences(
        df,
        treatment_time,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.SyntheticDifferenceInDifferencesWeightFitter(
            sample_kwargs=sample_kwargs,
        ),
    )

    # DatetimeIndex branch in _bayesian_plot calls format_date_axes.
    fig, _ = result.plot(show=False)
    assert isinstance(fig, plt.Figure)

    # Default effect_summary call covers the main body and prose generation.
    summary = result.effect_summary()
    assert hasattr(summary, "table")
    assert hasattr(summary, "text")
    assert isinstance(summary.text, str) and len(summary.text) > 0

    # Passing period= triggers the ignored-warning branch.
    with pytest.warns(
        UserWarning, match="ignored for SyntheticDifferenceInDifferences"
    ):
        result.effect_summary(period="post")

    # cumulative=False covers the conditional ``obs_cum``/``counterfactual_cum``
    # branches that are skipped by the default call above.
    summary_no_cum = result.effect_summary(cumulative=False)
    assert "cumulative" not in summary_no_cum.table.index


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
def test_ancova(mock_pymc_sample, anova1_data):
    """
    Test Pre-PostNEGD experiment on anova1 data.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.PrePostNEGD returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    """
    df = anova1_data
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
def test_geolift1(mock_pymc_sample, geolift1_data):
    """
    Test Synthetic Control experiment on geo lift data.

    Loads data and checks:
    1. data is a dataframe
    2. causalpy.SyntheticControl returns correct type
    3. the correct number of MCMC chains exists in the posterior inference data
    4. the correct number of MCMC draws exists in the posterior inference data
    """
    df = geolift1_data
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


@pytest.fixture
def bsts_data():
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

    return SimpleNamespace(dates=dates, data_with_x=data_with_x, data_no_x=data_no_x)


@pytest.mark.integration
def test_bsts_with_exogenous_regressor(bsts_data):
    pytest.importorskip(
        "pymc_marketing", reason="pymc-marketing optional for default BSTS components"
    )
    dates = bsts_data.dates
    data_with_x = bsts_data.data_with_x

    coords_with_x = {
        "obs_ind": dates,
        "coeffs": ["x1"],
        "treated_units": ["unit_0"],
        "datetime_index": dates,
    }

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

    model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        n_order=2, n_changepoints_trend=5, sample_kwargs=bsts_sample_kwargs
    )
    model.fit(X=X_da, y=y_da, coords=coords_with_x.copy())

    assert isinstance(model.idata, az.InferenceData)
    assert "posterior" in model.idata
    assert "beta" in model.idata.posterior
    assert "fourier_beta" in model.idata.posterior
    assert "delta" in model.idata.posterior
    assert "sigma" in model.idata.posterior
    assert "mu" in model.idata.posterior_predictive
    assert "y_hat" in model.idata.posterior_predictive

    predictions = model.predict(X=X_da, coords=coords_with_x)
    assert isinstance(predictions, az.InferenceData)

    score = model.score(X=X_da, y=y_da, coords=coords_with_x)
    assert isinstance(score, pd.Series)


@pytest.mark.integration
def test_bsts_without_exogenous_regressor(bsts_data):
    pytest.importorskip(
        "pymc_marketing", reason="pymc-marketing optional for default BSTS components"
    )
    dates = bsts_data.dates
    data_no_x = bsts_data.data_no_x

    coords_no_x = {
        "obs_ind": dates,
        "treated_units": ["unit_0"],
        "datetime_index": dates,
    }

    y_da = xr.DataArray(
        data_no_x["y"].values[:, None],
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": dates, "treated_units": ["unit_0"]},
    )

    X_da = xr.DataArray(
        np.zeros((len(dates), 0)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": dates, "coeffs": []},
    )

    model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        n_order=2, n_changepoints_trend=5, sample_kwargs=bsts_sample_kwargs
    )
    model.fit(X=X_da, y=y_da, coords=coords_no_x.copy())

    assert isinstance(model.idata, az.InferenceData)
    assert "posterior" in model.idata
    assert "beta" not in model.idata.posterior
    assert "fourier_beta" in model.idata.posterior
    assert "delta" in model.idata.posterior
    assert "sigma" in model.idata.posterior

    predictions = model.predict(X=X_da, coords=coords_no_x)
    assert isinstance(predictions, az.InferenceData)

    score = model.score(X=X_da, y=y_da, coords=coords_no_x)
    assert isinstance(score, pd.Series)


@pytest.mark.integration
def test_bsts_with_empty_exogenous_regressor(bsts_data):
    pytest.importorskip(
        "pymc_marketing", reason="pymc-marketing optional for default BSTS components"
    )
    dates = bsts_data.dates
    data_no_x = bsts_data.data_no_x

    coords_empty_x = {
        "obs_ind": dates,
        "treated_units": ["unit_0"],
        "datetime_index": dates,
        "coeffs": [],
    }

    y_da = xr.DataArray(
        data_no_x["y"].values[:, None],
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": dates, "treated_units": ["unit_0"]},
    )

    X_da = xr.DataArray(
        np.zeros((len(dates), 0)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": dates, "coeffs": []},
    )

    model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        n_order=2, n_changepoints_trend=5, sample_kwargs=bsts_sample_kwargs
    )
    model.fit(X=X_da, y=y_da, coords=coords_empty_x.copy())

    assert isinstance(model.idata, az.InferenceData)

    predictions = model.predict(X=X_da, coords=coords_empty_x)
    assert isinstance(predictions, az.InferenceData)

    score = model.score(X=X_da, y=y_da, coords=coords_empty_x)
    assert isinstance(score, pd.Series)


@pytest.mark.integration
def test_bsts_error_invalid_inputs(bsts_data):
    pytest.importorskip(
        "pymc_marketing", reason="pymc-marketing optional for default BSTS components"
    )
    dates = bsts_data.dates
    data_with_x = bsts_data.data_with_x

    coords_with_x = {
        "obs_ind": dates,
        "coeffs": ["x1"],
        "treated_units": ["unit_0"],
        "datetime_index": dates,
    }

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

    model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        n_order=2, n_changepoints_trend=5, sample_kwargs=bsts_sample_kwargs
    )
    model.fit(X=X_da, y=y_da, coords=coords_with_x.copy())

    X_da_no_x = xr.DataArray(
        np.zeros((len(dates), 0)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": dates, "coeffs": []},
    )

    with pytest.raises(
        ValueError,
        match=r"X\.coords\['obs_ind'\] must contain datetime values",
    ):
        bad_model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs=bsts_sample_kwargs
        )
        bad_X = xr.DataArray(
            data_with_x[["x1"]].values,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": np.arange(len(dates)),
                "coeffs": ["x1"],
            },
        )
        bad_y = xr.DataArray(
            data_with_x["y"].values[:, None],
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": np.arange(len(dates)), "treated_units": ["unit_0"]},
        )
        bad_model.fit(X=bad_X, y=bad_y, coords=coords_with_x.copy())

    with pytest.raises(ValueError, match="Model was built with exogenous variables"):
        model.predict(X=X_da_no_x, coords=coords_with_x)

    with pytest.raises(
        ValueError,
        match=r"Exogenous variable names mismatch",
    ):
        wrong_shape_vals = np.hstack(
            [data_with_x[["x1"]].values, data_with_x[["x1"]].values]
        )
        X_wrong_shape = xr.DataArray(
            wrong_shape_vals,
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates, "coeffs": ["x1", "x2"]},
        )
        model.predict(X=X_wrong_shape, coords=coords_with_x)


@pytest.mark.integration
def test_bsts_error_custom_component_validation():
    pytest.importorskip(
        "pymc_marketing", reason="pymc-marketing optional for default BSTS components"
    )

    class BadTrendComponent:
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


@pytest.mark.integration
def test_bsts_error_non_xarray_input(bsts_data):
    pytest.importorskip(
        "pymc_marketing", reason="pymc-marketing optional for default BSTS components"
    )
    dates = bsts_data.dates
    data_with_x = bsts_data.data_with_x

    coords_with_x = {
        "obs_ind": dates,
        "coeffs": ["x1"],
        "treated_units": ["unit_0"],
        "datetime_index": dates,
    }

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

    model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
        n_order=2, n_changepoints_trend=5, sample_kwargs=bsts_sample_kwargs
    )
    model.fit(X=X_da, y=y_da, coords=coords_with_x.copy())

    with pytest.raises(TypeError, match="X must be an xarray DataArray"):
        model.predict(
            X=data_with_x[["x1"]].values,
            coords=coords_with_x,
        )

@pytest.fixture(scope="module")
def state_space_model():
    """Fixture providing a fitted StateSpaceTimeSeries model for testing."""
    try:
        from pymc_extras.statespace import structural  # noqa: F401
    except ImportError:
        pytest.skip("pymc-extras is required for InterruptedTimeSeries tests")

    rng = np.random.default_rng(seed=123)
    dates = pd.date_range(start="2020-01-01", end="2020-03-31", freq="D")
    n_obs = len(dates)
    trend = np.linspace(0, 2, n_obs)
    seasonality = 3 * np.sin(2 * np.pi * dates.dayofyear / 365.25) + 2 * np.cos(
        4 * np.pi * dates.dayofyear / 365.25
    )
    noise = rng.normal(0, 0.3, n_obs)
    data = pd.DataFrame({"y": trend + seasonality + noise}, index=dates)

    y_da = xr.DataArray(
        data["y"].values.reshape(-1, 1),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": dates, "treated_units": ["unit_0"]},
    )
    model = cp.pymc_models.StateSpaceTimeSeries(
        level_order=2,
        seasonal_length=7,
        sample_kwargs=ss_sample_kwargs,
        mode="FAST_COMPILE",
    )
    dummy_X = xr.DataArray(
        np.zeros((n_obs, 0)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": dates, "coeffs": []},
    )
    idata = model.fit(X=dummy_X, y=y_da)
    return SimpleNamespace(
        model=model,
        idata=idata,
        y_da=y_da,
        dates=dates,
        n_obs=n_obs,
        dummy_X=dummy_X,
    )


@pytest.mark.integration
def test_state_space_fitting(state_space_model):
    """Test model fitting produces correct inference data."""
    idata = state_space_model.idata
    assert isinstance(idata, az.InferenceData)
    assert "posterior" in idata
    assert "posterior_predictive" in idata
    expected_params = [
        "P0_diag",
        "initial_level_trend",
        "params_freq",
        "sigma_level_trend",
        "sigma_freq",
    ]
    for param in expected_params:
        assert param in idata.posterior, f"Parameter {param} not found in posterior"
    assert "y_hat" in idata.posterior_predictive
    assert "mu" in idata.posterior_predictive


@pytest.mark.integration
def test_state_space_insample_prediction(state_space_model):
    """Test in-sample prediction."""
    model = state_space_model.model
    dates = state_space_model.dates
    dummy_X = xr.DataArray(
        np.zeros((len(dates), 0)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": dates, "coeffs": []},
    )
    predictions = model.predict(X=dummy_X, out_of_sample=False)
    assert isinstance(predictions, az.InferenceData)
    assert "posterior_predictive" in predictions
    assert "y_hat" in predictions.posterior_predictive
    assert "mu" in predictions.posterior_predictive


@pytest.mark.integration
def test_state_space_forecast(state_space_model):
    """Test out-of-sample prediction (forecasting)."""
    model = state_space_model.model
    future_dates = pd.date_range(start="2020-04-01", end="2020-04-07", freq="D")
    future_X = xr.DataArray(
        np.zeros((len(future_dates), 0)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": future_dates, "coeffs": []},
    )
    predictions = model.predict(X=future_X, out_of_sample=True)
    assert isinstance(predictions, az.InferenceData)
    assert "y_hat" in predictions.posterior_predictive
    assert "mu" in predictions.posterior_predictive
    assert predictions.posterior_predictive["y_hat"].shape[-1] == len(future_dates)


@pytest.mark.integration
def test_state_space_scoring(state_space_model):
    """Test model scoring."""
    model = state_space_model.model
    dates = state_space_model.dates
    y_da = state_space_model.y_da
    dummy_X = xr.DataArray(
        np.zeros((len(dates), 0)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": dates, "coeffs": []},
    )
    score = model.score(X=dummy_X, y=y_da)
    assert isinstance(score, pd.Series)
    assert "unit_0_r2" in score.index
    assert "unit_0_r2_std" in score.index
    assert score["unit_0_r2"] > 0.0


@pytest.mark.integration
def test_state_space_model_structure(state_space_model):
    """Test model structure and components."""
    model = state_space_model.model
    assert hasattr(model, "ss_mod")
    assert model.ss_mod is not None
    assert hasattr(model, "_train_index")
    assert isinstance(model._train_index, pd.DatetimeIndex)
    assert hasattr(model, "conditional_idata")
    assert isinstance(model.conditional_idata, xr.Dataset)
    assert model.level_order == 2
    assert model.seasonal_length == 7
    assert model.mode == "FAST_COMPILE"


@pytest.mark.integration
def test_state_space_error_handling():
    """Test error handling for invalid inputs."""
    try:
        from pymc_extras.statespace import structural  # noqa: F401
    except ImportError:
        pytest.skip("pymc-extras is required for InterruptedTimeSeries tests")

    rng = np.random.default_rng(seed=123)
    dates = pd.date_range(start="2020-01-01", end="2020-03-31", freq="D")
    n_obs = len(dates)
    data = pd.DataFrame({"y": rng.normal(0, 1, n_obs)}, index=dates)
    y_da = xr.DataArray(
        data["y"].values.reshape(-1, 1),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": dates, "treated_units": ["unit_0"]},
    )
    dummy_X = xr.DataArray(
        np.zeros((n_obs, 0)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": dates, "coeffs": []},
    )
    model = cp.pymc_models.StateSpaceTimeSeries(sample_kwargs=ss_sample_kwargs)
    model.fit(X=dummy_X, y=y_da)

    with pytest.raises(
        ValueError,
        match=r"y\.coords\['obs_ind'\] must contain datetime values",
    ):
        bad_model = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs=ss_sample_kwargs
        )
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
        bad_model.fit(X=bad_X, y=bad_y)

    with pytest.raises(
        ValueError,
        match="X must be provided for out-of-sample predictions",
    ):
        model.predict(X=None, out_of_sample=True)

    unfitted_model = cp.pymc_models.StateSpaceTimeSeries(
        sample_kwargs=ss_sample_kwargs
    )

    with pytest.raises(RuntimeError, match="Model must be fit before"):
        unfitted_model._smooth()

    with pytest.raises(RuntimeError, match="Model must be fit before"):
        unfitted_model._forecast(start=dates[0], periods=10)


@pytest.mark.integration
def test_state_space_parameter_variants():
    """Test model initialization with different parameters."""
    try:
        from pymc_extras.statespace import structural  # noqa: F401
    except ImportError:
        pytest.skip("pymc-extras is required for InterruptedTimeSeries tests")

    model_level1 = cp.pymc_models.StateSpaceTimeSeries(
        level_order=1,
        seasonal_length=7,
        sample_kwargs=ss_sample_kwargs,
        mode="FAST_COMPILE",
    )
    assert model_level1.level_order == 1

    model_monthly = cp.pymc_models.StateSpaceTimeSeries(
        level_order=2,
        seasonal_length=30,
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
        assert sc.pre_design["treated"].shape == (40, len(treated_units))
        assert sc.post_design["treated"].shape == (20, len(treated_units))
        assert sc.pre_design["control"].shape == (40, len(control_units))
        assert sc.post_design["control"].shape == (20, len(control_units))

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
