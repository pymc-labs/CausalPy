"""
Unit tests for pymc_experiments.py
"""

import matplotlib.pyplot as plt
import pandas as pd

import causalpy as cp

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


def test_summary_intervention():
    # Load the data
    df = cp.load_data("sc")

    # Set the parameters
    treatment_time = 70  # or any other specific value you're testing with
    seed = 42  # A seed for reproducibility, adjust as necessary

    # Create an instance of SyntheticControl
    result = cp.pymc_experiments.SyntheticControl(
        df,
        treatment_time,
        formula="actual ~ 0 + a + b + c + d + e + f + g",
        model=cp.pymc_models.WeightedSumFitter(
            sample_kwargs={"target_accept": 0.95, "random_seed": seed}
        ),
    )

    # Call the summary function with "intervention" version
    summary_df = result.summary(version="intervention")

    alpha = 0.05
    correction = False

    # Invoke the power_plot method
    power_plot_fig = result.power_plot(alpha=alpha, correction=correction)

    alpha = 0.05
    correction = False

    # Invoke the power_summary method
    power_summary_df = result.power_summary(alpha=alpha, correction=correction)

    # Test the properties of the summary_df
    assert isinstance(
        summary_df, pd.DataFrame
    ), "Summary should return a DataFrame for 'intervention' version"
    assert isinstance(power_plot_fig, plt.Figure), "Should return a plt.Figure"
    assert isinstance(power_summary_df, pd.DataFrame), "Should return a DataFrame"


def test_did_summary():
    """Test that the summary stat function returns a string."""
    df = cp.load_data("did")
    result = cp.pymc_experiments.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    print(type(result._causal_impact_summary_stat()))
    assert isinstance(result._causal_impact_summary_stat(), str)


def test_regression_kink_gradient_change():
    """Test function to numerically calculate the change in gradient around the kink
    point in regression kink designs"""
    # test no change in gradient
    assert cp.pymc_experiments.RegressionKink._eval_gradient_change(-1, 0, 1, 1) == 0.0
    assert cp.pymc_experiments.RegressionKink._eval_gradient_change(1, 0, -1, 1) == 0.0
    assert cp.pymc_experiments.RegressionKink._eval_gradient_change(0, 0, 0, 1) == 0.0
    # test positive change in gradient
    assert cp.pymc_experiments.RegressionKink._eval_gradient_change(0, 0, 1, 1) == 1.0
    assert cp.pymc_experiments.RegressionKink._eval_gradient_change(0, 0, 2, 1) == 2.0
    assert cp.pymc_experiments.RegressionKink._eval_gradient_change(-1, -1, 2, 1) == 3.0
    assert cp.pymc_experiments.RegressionKink._eval_gradient_change(-1, 0, 2, 1) == 1.0
    # test negative change in gradient
    assert cp.pymc_experiments.RegressionKink._eval_gradient_change(0, 0, -1, 1) == -1.0
    assert cp.pymc_experiments.RegressionKink._eval_gradient_change(0, 0, -2, 1) == -2.0
    assert (
        cp.pymc_experiments.RegressionKink._eval_gradient_change(-1, -1, -2, 1) == -1.0
    )
    assert cp.pymc_experiments.RegressionKink._eval_gradient_change(1, 0, -2, 1) == -1.0
