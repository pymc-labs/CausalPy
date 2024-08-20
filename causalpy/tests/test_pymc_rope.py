#   Copyright 2024 The PyMC Labs Developers
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
"""
Test suite for the ROPE class to estimate Bayesian Power Analysis.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from causalpy.pymc_rope import ROPE, AlphaValueError, CorrectionValueError


@pytest.fixture
def rope_instance():
    # Create sample posterior data
    post_y = np.random.normal(0, 1, 1000)
    mu_data = np.random.normal(0, 2, (3, 3000, 1000))
    mu_xarray = xr.DataArray(mu_data, dims=["chain", "draw", "obs_ind"], name="mu")
    post_pred = {"posterior_predictive": {"mu": mu_xarray}}
    return ROPE(post_y, post_pred)


@pytest.mark.parametrize("invalid_alpha", [-0.1, 1.1])
def test_validate_alpha_error(rope_instance, invalid_alpha):
    with pytest.raises(AlphaValueError):
        rope_instance._validate_alpha(invalid_alpha)


# Test for CorrectionValueError
@pytest.mark.parametrize("invalid_correction", [42, "invalid", [1, 2, 3]])
def test_validate_correction_error(rope_instance, invalid_correction):
    with pytest.raises(CorrectionValueError):
        rope_instance._validate_correction(invalid_correction)


@pytest.mark.parametrize(
    "invalid_correction_dict",
    [{"cum": 1.0, "mean": 1.0}, {"cumulative": "invalid", "mean": 1.0}],
)
def test_validate_correction_dict_error(rope_instance, invalid_correction_dict):
    with pytest.raises(CorrectionValueError):
        rope_instance._validate_correction(invalid_correction_dict)


@pytest.mark.parametrize(
    "invalid_correction_series",
    [
        pd.Series([1.0, 2.0], index=["cum", "mean"]),
        pd.Series(["invalid", 1.0], index=["cumulative", "mean"]),
    ],
)
def test_validate_correction_series_error(rope_instance, invalid_correction_series):
    with pytest.raises(CorrectionValueError):
        rope_instance._validate_correction(invalid_correction_series)


@pytest.mark.parametrize(
    "posterior, x",
    [
        (np.random.normal(0, 1, 1000), 0),
        (np.random.normal(0, 1, 1000), 1),
        (np.random.normal(0, 1, 1000), -1),
    ],
)
def test_compute_bayesian_tail_probability_positive(rope_instance, posterior, x):
    """
    Validate that the results from Bayesian tail probability are always positive.
    """
    prob = rope_instance.compute_bayesian_tail_probability(posterior, x)
    assert prob >= 0, f"Probability is negative: {prob}"


@pytest.mark.parametrize(
    "posterior, x",
    [
        (np.random.normal(0, 1, 1000), 3),
        (np.random.normal(0, 1, 1000), -3),
    ],
)
def test_bayesian_tail_probability_low_outside_rope(rope_instance, posterior, x):
    """
    Validate that the Bayesian tail probability is low outside the ROPE.
    """
    prob = rope_instance.compute_bayesian_tail_probability(posterior, x)
    assert prob < 0.05, f"Probability is not low outside the ROPE: {prob}"


@pytest.mark.parametrize(
    "posterior",
    [
        np.random.normal(0, 1, 1000),
    ],
)
def test_bayesian_tail_probability_close_to_one_at_mean(rope_instance, posterior):
    """
    Validate that the Bayesian tail probability is close to one when the value is equal to the mean.
    """
    mean_value = np.mean(posterior)
    prob = rope_instance.compute_bayesian_tail_probability(posterior, mean_value)
    assert np.isclose(
        prob, 1, atol=0.1
    ), f"Probability is not close to one at the mean: {prob}"


# Test for Posterior MDE Calculation
def test_calculate_posterior_mde(rope_instance):
    alpha = 0.05
    correction = False
    results = rope_instance._calculate_posterior_mde(alpha, correction)
    assert "posterior_estimation" in results
    assert "credible_interval" in results
    assert "posterior_mde" in results
    assert isinstance(results["posterior_estimation"], dict)
    assert isinstance(results["credible_interval"], dict)
    assert isinstance(results["posterior_mde"], dict)


# Test for Causal Effect Summary
def test_causal_effect_summary(rope_instance):
    alpha = 0.05
    summary_df = rope_instance.causal_effect_summary(alpha=alpha)
    assert isinstance(summary_df, pd.DataFrame)
    expected_columns = {
        "posterior_estimation",
        "results",
        "credible_interval",
        "bayesian_tail_probability",
        "causal_effect",
    }
    assert set(summary_df.columns).issubset(expected_columns)


# Test for Power Distribution Plot
def test_plot_power_distribution(rope_instance):
    alpha = 0.05
    fig, axs = rope_instance.plot_power_distribution(alpha=alpha)
    assert isinstance(fig, plt.Figure)
    assert len(axs) == 2  # Check that there are two subplots
    for ax in axs:
        assert ax.get_title()  # Ensure each subplot has a title
