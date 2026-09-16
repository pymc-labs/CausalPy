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
import xarray as xr
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.custom_exceptions import (
    GroupNotSampledException,
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


@pytest.mark.parametrize("bundleless", [False, True])
def test_auto_prior_failure_invalidates_posterior(
    monkeypatch, bundleless, mock_pymc_sample
):
    experiment = make_its()
    if bundleless:
        experiment._supports_results = False
    failure = ValueError("invalid prior specification")

    def fail_prior(**kwargs):
        raise failure

    monkeypatch.setattr(experiment, "sample_prior_predictive", fail_prior)
    with pytest.raises(ValueError) as caught:
        experiment.fit()
    assert caught.value is failure
    assert not experiment.is_fitted
    assert "posterior" not in experiment.idata.children
    with pytest.raises(GroupNotSampledException):
        experiment.plot(show=False)


def test_sampled_model_assignment_preserves_both_experiments(mock_pymc_sample):
    original = make_its().fit()
    incoming = make_its().fit()
    model_before = original.model
    result_before = original.result
    prior_before = original.prior_result
    original_draws = original.idata.copy(deep=True)
    incoming_draws = incoming.idata.copy(deep=True)

    with pytest.raises(ValueError, match="fresh model"):
        original.model = incoming.model

    assert original.model is model_before
    assert original.result is result_before
    assert original.prior_result is prior_before
    xr.testing.assert_identical(original.idata, original_draws)
    xr.testing.assert_identical(incoming.idata, incoming_draws)


def make_panel(model=None, outcome_shift=0):
    data = pd.DataFrame(
        {
            "unit": np.repeat(["a", "b"], 6),
            "time": np.tile(np.arange(6), 2),
            "x": np.tile([0, 1, 0, 1, 0, 1], 2),
            "y": np.arange(12) + outcome_shift,
        }
    )
    return cp.PanelRegression(
        data,
        formula="y ~ 1 + x",
        unit_fe_variable="unit",
        time_fe_variable="time",
        model=model
        if model is not None
        else cp.pymc_models.LinearRegression(
            sample_kwargs={"draws": 5, "tune": 5, "chains": 1, "progressbar": False},
            prior_sample_kwargs={"draws": 7, "random_seed": 12},
        ),
    )


@pytest.mark.parametrize("phase", ["fit", "sample_prior_predictive"])
def test_constructor_rejects_sampled_panel_model(phase, mock_pymc_sample):
    first = make_panel()
    getattr(first, phase)()
    draws_before = first.idata.copy(deep=True)
    with pytest.raises(ValueError, match="fresh model"):
        make_panel(first.model, outcome_shift=100)
    xr.testing.assert_identical(first.idata, draws_before)
