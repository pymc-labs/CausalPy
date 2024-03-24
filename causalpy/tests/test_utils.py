"""
Tests for utility functions
"""

import numpy as np
import pandas as pd

from causalpy.utils import (
    _is_variable_dummy_coded,
    _series_has_2_levels,
    compute_bayesian_tail_probability,
    round_num,
)


def test_dummy_coding():
    """Test if the function to check if a variable is dummy coded works correctly"""
    assert _is_variable_dummy_coded(pd.Series([False, True, False, True])) is True
    assert _is_variable_dummy_coded(pd.Series([False, True, False, "frog"])) is False
    assert _is_variable_dummy_coded(pd.Series([0, 0, 1, 0, 1])) is True
    assert _is_variable_dummy_coded(pd.Series([0, 0, 1, 0, 2])) is False
    assert _is_variable_dummy_coded(pd.Series([0, 0.5, 1, 0, 1])) is False


def test_2_level_series():
    """Test if the function to check if a variable has 2 levels works correctly"""
    assert _series_has_2_levels(pd.Series(["a", "a", "b"])) is True
    assert _series_has_2_levels(pd.Series(["a", "a", "b", "c"])) is False
    assert _series_has_2_levels(pd.Series(["coffee", "tea", "coffee"])) is True
    assert _series_has_2_levels(pd.Series(["water", "tea", "coffee"])) is False
    assert _series_has_2_levels(pd.Series([0, 1, 0, 1])) is True
    assert _series_has_2_levels(pd.Series([0, 1, 0, 2])) is False


def test_round_num():
    """Test if the function to round numbers works correctly"""
    assert round_num(0.12345, None) == "0.12"
    assert round_num(0.12345, 0) == "0.1"
    assert round_num(0.12345, 1) == "0.1"
    assert round_num(0.12345, 2) == "0.12"
    assert round_num(0.12345, 3) == "0.123"
    assert round_num(0.12345, 4) == "0.1235"
    assert round_num(0.12345, 5) == "0.12345"
    assert round_num(0.12345, 6) == "0.12345"
    assert round_num(123.456, None) == "123"
    assert round_num(123.456, 1) == "123"
    assert round_num(123.456, 2) == "123"
    assert round_num(123.456, 3) == "123"
    assert round_num(123.456, 4) == "123.5"
    assert round_num(123.456, 5) == "123.46"
    assert round_num(123.456, 6) == "123.456"
    assert round_num(123.456, 7) == "123.456"


def test_compute_bayesian_tail_probability():
    """
    Re-running all tests for the compute_bayesian_tail_probability function with the corrected understanding
    and expectations for various scenarios.
    """
    # Test 1: Posterior is a standard normal distribution, x = mean = 0
    posterior_standard_normal = np.random.normal(0, 1, 10000)
    x_at_mean = 0
    prob_at_mean = compute_bayesian_tail_probability(
        posterior_standard_normal, x_at_mean
    )
    assert np.isclose(prob_at_mean, 1, atol=0.05), f"Expected 1, got {prob_at_mean}"

    # Test 2: Posterior is a standard normal distribution, x = 1
    x_one_std_above = 1
    prob_one_std_above = compute_bayesian_tail_probability(
        posterior_standard_normal, x_one_std_above
    )
    assert (
        0 < prob_one_std_above < 1
    ), "Probability should decrease from 1 as x moves away from mean"

    # Test 3: Posterior is a standard normal distribution, x well outside the distribution
    x_far_out = 5
    prob_far_out = compute_bayesian_tail_probability(
        posterior_standard_normal, x_far_out
    )
    # Expect a very low probability for a value far outside the distribution
    assert prob_far_out < 0.01, f"Expected a value < 0.01, got {prob_far_out}"

    # Test 4: Posterior is a normal distribution with mean=5, std=2, x = mean
    posterior_shifted = np.random.normal(5, 2, 10000)
    x_at_shifted_mean = 5
    prob_at_shifted_mean = compute_bayesian_tail_probability(
        posterior_shifted, x_at_shifted_mean
    )
    # Expect the probability at the mean of a shifted distribution to be close to 1
    assert np.isclose(
        prob_at_shifted_mean, 1, atol=0.05
    ), f"Expected 1, got {prob_at_shifted_mean}"
