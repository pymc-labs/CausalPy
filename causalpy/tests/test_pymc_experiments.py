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
