"""
Unit tests for pymc_experiments.py
"""
import arviz as az
import pandas as pd

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
    round_to = None
    print(type(result._causal_impact_summary_stat(round_to)))
    print(result._causal_impact_summary_stat(round_to))
    assert isinstance(result._causal_impact_summary_stat(round_to), str)


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


def test_inverse_prop():
    df = cp.load_data("nhefs")
    result = cp.pymc_experiments.InversePropensityWeighting(
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
