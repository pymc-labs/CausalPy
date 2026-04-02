#   Copyright 2026 - 2026 The PyMC Labs Developers
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
Tests for experiment init refactors and new helpers.
"""

import numpy as np
import pandas as pd
import pytest

import causalpy as cp

SAMPLE_KWARGS = {
    "tune": 5,
    "draws": 10,
    "chains": 1,
    "progressbar": False,
    "random_seed": 42,
}


def test_interrupted_time_series_datapre_datapost_properties() -> None:
    """datapre/datapost should reflect treatment time split."""
    df = cp.load_data("its")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    treatment_time = pd.Timestamp("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=treatment_time,
        formula="y ~ 1 + t",
        model=cp.pymc_models.LinearRegression(sample_kwargs=SAMPLE_KWARGS),
    )

    assert result.data.index.name == "obs_ind"
    assert result.datapre.index.max() < treatment_time
    assert result.datapost.index.min() >= treatment_time


def test_synthetic_control_datapre_datapost_properties() -> None:
    """datapre/datapost should reflect treatment time split."""
    df = cp.load_data("sc")
    treatment_time = 70
    result = cp.SyntheticControl(
        df,
        treatment_time=treatment_time,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=SAMPLE_KWARGS),
    )

    assert result.data.index.name == "obs_ind"
    assert result.datapre.index.max() < treatment_time
    assert result.datapost.index.min() >= treatment_time


@pytest.mark.integration
def test_regression_kink_gradient_change_uses_epsilon() -> None:
    """Gradient change should respect the instance epsilon value."""
    rng = np.random.default_rng(42)
    n_obs = 50
    kink = 0.5
    beta = [0, -1, 0, 2, 0]
    sigma = 0.05
    x = rng.uniform(-1, 1, n_obs)
    y = (
        beta[0]
        + beta[1] * x
        + beta[2] * x**2
        + beta[3] * (x - kink) * (x >= kink)
        + beta[4] * ((x - kink) ** 2) * (x >= kink)
        + rng.normal(0, sigma, n_obs)
    )
    df = pd.DataFrame({"x": x, "y": y, "treated": x >= kink})

    result = cp.RegressionKink(  # type: ignore[abstract]
        df,
        formula=f"y ~ 1 + x + I((x-{kink})*treated)",
        kink_point=kink,
        epsilon=0.01,
        model=cp.pymc_models.LinearRegression(sample_kwargs=SAMPLE_KWARGS),
    )

    mu_kink_left, mu_kink, mu_kink_right = result._probe_kink_point()
    expected = result._eval_gradient_change(
        mu_kink_left, mu_kink, mu_kink_right, result.epsilon
    )
    np.testing.assert_allclose(result.gradient_change.values, expected.values)
