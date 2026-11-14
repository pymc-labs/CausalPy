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
"""Posterior recovery checks for generalized linear regression paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pymc_extras.prior import Prior

import causalpy as cp
from causalpy._arviz_compat import hdi_bounds
from causalpy.constants import HDI_PROB
from causalpy.pymc_models import GeneralizedLinearRegression

sample_kwargs = {
    "tune": 800,
    "draws": 800,
    "chains": 1,
    "cores": 1,
    "progressbar": False,
    "random_seed": 42,
}


def _poisson_glr(**kwargs) -> GeneralizedLinearRegression:
    sample = {**sample_kwargs, **kwargs.pop("sample_kwargs", {})}
    priors = kwargs.pop(
        "priors",
        {
            "beta": Prior(
                "Normal",
                mu=np.log(20.0),
                sigma=0.05,
                dims=["treated_units", "coeffs"],
            ),
        },
    )
    return GeneralizedLinearRegression(
        family="poisson",
        priors=priors,
        sample_kwargs=sample,
        **kwargs,
    )


@pytest.mark.slow
@pytest.mark.correctness
def test_poisson_its_posterior_recovers_level_shift(real_pymc_sampling):
    """Poisson ITS HDI covers the known log-scale level shift and cumulative impact."""
    n = 80
    treatment_time = 50
    log_baseline = np.log(20.0)
    log_post_add = np.log(1.5)
    t = np.arange(n, dtype=float)
    log_mu = log_baseline + (t >= treatment_time) * log_post_add
    # Deterministic means avoid Poisson noise flakiness while still using a count model.
    y = np.exp(log_mu)
    df = pd.DataFrame({"t": t, "y": y})

    daily_effect = float(np.exp(log_baseline + log_post_add) - np.exp(log_baseline))
    n_post = int((t >= treatment_time).sum())
    cumulative_effect = daily_effect * n_post

    experiment = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1",
        model=_poisson_glr(),
    )
    experiment.fit()
    post_mean = experiment.result.impact_post.mean(dim="obs_ind")
    if "treated_units" in post_mean.dims:
        post_mean = post_mean.isel(treated_units=0)
    post_lower, post_upper = hdi_bounds(post_mean, prob=HDI_PROB)
    assert post_lower <= daily_effect <= post_upper

    cumulative = experiment.result.impact_post_cumulative
    if "treated_units" in cumulative.dims:
        cumulative = cumulative.isel(treated_units=0)
    final_cumulative = cumulative.isel(obs_ind=-1)
    cumulative_lower, cumulative_upper = hdi_bounds(final_cumulative, prob=HDI_PROB)
    assert cumulative_lower <= cumulative_effect <= cumulative_upper
