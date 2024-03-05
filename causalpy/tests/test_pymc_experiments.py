"""
Unit tests for pymc_experiments.py
"""

import causalpy as cp

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


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
