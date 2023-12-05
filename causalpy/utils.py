"""
Utility functions
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def _is_variable_dummy_coded(series: pd.Series) -> bool:
    """Check if a data in the provided Series is dummy coded. It should be 0 or 1
    only."""
    return len(set(series).difference(set([0, 1]))) == 0


def _series_has_2_levels(series: pd.Series) -> bool:
    """Check that the variable in the provided Series has 2 levels"""
    return len(pd.Categorical(series).categories) == 2


def compute_bayesian_tail_probability(posterior, x) -> float:
    """
    Calculate the probability of a given value being in a distribution defined by the posterior,

    Args:
    - data: a list or array-like object containing the data to define the distribution
    - x: a numeric value for which to calculate the probability of being in the distribution

    Returns:
    - prob: a numeric value representing the probability of x being in the distribution
    """
    lower_bound, upper_bound = min(posterior), max(posterior)
    mean, std = np.mean(posterior), np.std(posterior)

    cdf_lower = norm.cdf(lower_bound, mean, std)
    cdf_upper = 1 - norm.cdf(upper_bound, mean, std)
    cdf_x = norm.cdf(x, mean, std)

    if cdf_x <= 0.5:
        probability = 2 * (cdf_x - cdf_lower) / (1 - cdf_lower - cdf_upper)
    else:
        probability = 2 * (1 - cdf_x + cdf_lower) / (1 - cdf_lower - cdf_upper)

    return abs(round(probability, 2))
