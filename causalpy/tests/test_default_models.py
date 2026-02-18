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
"""
Tests that experiment classes use default PyMC models when model=None.
"""

import numpy as np
import pandas as pd
import pytest

import causalpy as cp
from causalpy.data.simulate_data import generate_staggered_did_data
from causalpy.pymc_models import (
    InstrumentalVariableRegression,
    LinearRegression,
    PropensityScore,
    WeightedSumFitter,
)


def _setup_regression_kink_data(kink=0.5):
    """Set up data for regression kink design tests."""
    rng = np.random.default_rng(42)
    N = 50
    beta = [0, -1, 0, 2, 0]
    sigma = 0.05
    x = rng.uniform(-1, 1, N)
    y = (
        beta[0]
        + beta[1] * x
        + beta[2] * x**2
        + beta[3] * (x - kink) * (x >= kink)
        + beta[4] * (x - kink) ** 2 * (x >= kink)
        + rng.normal(0, sigma, N)
    )
    return pd.DataFrame({"x": x, "y": y, "treated": x >= kink})


def _iv_data():
    """Generate synthetic IV data."""
    rng = np.random.default_rng(42)
    N = 50
    e1 = rng.normal(0, 3, N)
    e2 = rng.normal(0, 1, N)
    Z = rng.uniform(0, 1, N)
    X = -1 + 4 * Z + e2 + 2 * e1
    y = 2 + 3 * X + 3 * e1
    df = pd.DataFrame({"y": y, "X": X, "Z": Z})
    return {
        "data": df[["y", "X"]],
        "instruments_data": df[["X", "Z"]],
        "formula": "y ~ 1 + X",
        "instruments_formula": "X ~ 1 + Z",
    }


@pytest.fixture
def sample_kwargs():
    """Minimal sampling kwargs for fast tests."""
    return {
        "tune": 5,
        "draws": 5,
        "chains": 1,
        "progressbar": False,
        "random_seed": 42,
    }


@pytest.mark.parametrize(
    "experiment_class,init_kwargs,expected_model_class",
    [
        (
            cp.DifferenceInDifferences,
            {
                "data": lambda: cp.load_data("did"),
                "formula": "y ~ 1 + group*post_treatment",
                "time_variable_name": "t",
                "group_variable_name": "group",
            },
            LinearRegression,
        ),
        (
            cp.InterruptedTimeSeries,
            {
                "data": lambda: (
                    cp.load_data("its")
                    .assign(date=lambda x: pd.to_datetime(x["date"]))
                    .set_index("date")
                ),
                "treatment_time": pd.to_datetime("2017-01-01"),
                "formula": "y ~ 1 + t + C(month)",
            },
            LinearRegression,
        ),
        (
            cp.RegressionDiscontinuity,
            {
                "data": lambda: cp.load_data("rd"),
                "formula": "y ~ 1 + bs(x, df=6) + treated",
                "treatment_threshold": 0.5,
                "epsilon": 0.001,
            },
            LinearRegression,
        ),
        (
            cp.RegressionKink,
            {
                "data": lambda: _setup_regression_kink_data(),
                "formula": "y ~ 1 + x + I((x-0.5)*treated)",
                "kink_point": 0.5,
            },
            LinearRegression,
        ),
        (
            cp.PrePostNEGD,
            {
                "data": lambda: cp.load_data("anova1"),
                "formula": "post ~ 1 + C(group) + pre",
                "group_variable_name": "group",
                "pretreatment_variable_name": "pre",
            },
            LinearRegression,
        ),
        (
            cp.SyntheticControl,
            {
                "data": lambda: cp.load_data("sc"),
                "treatment_time": 70,
                "control_units": ["a", "b", "c", "d", "e", "f", "g"],
                "treated_units": ["actual"],
            },
            WeightedSumFitter,
        ),
        (
            cp.StaggeredDifferenceInDifferences,
            {
                "data": lambda: generate_staggered_did_data(
                    n_units=30,
                    n_time_periods=15,
                    treatment_cohorts={5: 10, 10: 10},
                    seed=42,
                ),
                "formula": "y ~ 1 + C(unit) + C(time)",
                "unit_variable_name": "unit",
                "time_variable_name": "time",
                "treated_variable_name": "treated",
                "treatment_time_variable_name": "treatment_time",
            },
            LinearRegression,
        ),
        (
            cp.InstrumentalVariable,
            lambda: _iv_data(),
            InstrumentalVariableRegression,
        ),
        (
            cp.InversePropensityWeighting,
            {
                "data": lambda: cp.load_data("nhefs"),
                "formula": "trt ~ 1 + age + race",
                "outcome_variable": "outcome",
                "weighting_scheme": "robust",
            },
            PropensityScore,
        ),
    ],
    ids=[
        "DifferenceInDifferences",
        "InterruptedTimeSeries",
        "RegressionDiscontinuity",
        "RegressionKink",
        "PrePostNEGD",
        "SyntheticControl",
        "StaggeredDifferenceInDifferences",
        "InstrumentalVariable",
        "InversePropensityWeighting",
    ],
)
@pytest.mark.integration
def test_experiment_uses_default_model_when_model_is_none(
    mock_pymc_sample, experiment_class, init_kwargs, expected_model_class
):
    """Each experiment class uses its default PyMC model when model=None.

    Construction exercises the full pipeline (design matrices, fit via
    algorithm()) so this also validates that the default model is wired
    correctly, not just instantiated.
    """
    if callable(init_kwargs):
        kwargs = init_kwargs()
    else:
        kwargs = {k: (v() if callable(v) else v) for k, v in init_kwargs.items()}

    result = experiment_class(**kwargs, model=None)

    assert isinstance(result.model, expected_model_class)
    assert hasattr(result, "idata")


def test_missing_default_model_class_raises_valueerror():
    """Subclass without _default_model_class raises ValueError, not AttributeError."""
    from causalpy.experiments.base import BaseExperiment

    class _NoDefaultExperiment(BaseExperiment):  # pragma: no cover
        supports_bayes = True
        supports_ols = True

        def _bayesian_plot(self):
            pass

        def _ols_plot(self):
            pass

        def get_plot_data_bayesian(self):
            pass

        def get_plot_data_ols(self):
            pass

        def effect_summary(self):
            pass

    with pytest.raises(ValueError, match="model not set or passed"):
        _NoDefaultExperiment(model=None)


@pytest.mark.integration
def test_explicit_model_takes_precedence_over_default(mock_pymc_sample, sample_kwargs):
    """Explicitly passed model is used instead of default."""
    df = cp.load_data("did")
    explicit_model = cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs)

    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=explicit_model,
    )

    assert result.model is explicit_model
