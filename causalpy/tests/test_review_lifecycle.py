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
"""Regression coverage for lazy lifecycle review findings."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.custom_exceptions import (
    PriorPredictiveNotSupportedException,
)


def make_its(model=None):
    data = pd.DataFrame({"t": np.arange(12), "y": np.arange(12) + 0.2})
    return cp.InterruptedTimeSeries(
        data,
        treatment_time=8,
        formula="y ~ 1 + t",
        model=model
        if model is not None
        else cp.pymc_models.LinearRegression(
            sample_kwargs={"draws": 5, "tune": 5, "chains": 1, "progressbar": False},
            prior_sample_kwargs={"draws": 7, "random_seed": 12},
        ),
    )


@pytest.mark.parametrize(
    "reader", ["prior_result", "plot", "get_plot_data", "effect_summary"]
)
def test_unsupported_prior_reads_raise_capability_error(reader):
    experiment = make_its(LinearRegression(fit_intercept=False))
    with pytest.raises(PriorPredictiveNotSupportedException):
        if reader == "prior_result":
            _ = experiment.prior_result
        else:
            getattr(experiment, reader)(group="prior")
