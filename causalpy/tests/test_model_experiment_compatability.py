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
Test exceptions are raised when an experiment object is provided a model type (e.g.
`PyMCModel` or `ScikitLearnModel`) that is not supported by the experiment object.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp

CustomLinearRegression = cp.create_causalpy_compatible_class(LinearRegression)


# TODO: THE TWO FUNCTIONS BELOW ARE COPIED FROM causalpy/tests/test_regression_kink.py


def setup_regression_kink_data(kink):
    """Set up data for regression kink design tests"""
    # define parameters for data generation
    seed = 42
    rng = np.random.default_rng(seed)
    N = 50
    kink = 0.5
    beta = [0, -1, 0, 2, 0]
    sigma = 0.05
    # generate data
    x = rng.uniform(-1, 1, N)
    y = reg_kink_function(x, beta, kink) + rng.normal(0, sigma, N)
    return pd.DataFrame({"x": x, "y": y, "treated": x >= kink})


def reg_kink_function(x, beta, kink):
    """Utility function for regression kink design. Returns a piecewise linear function
    evaluated at x with a kink at kink and parameters beta"""
    return (
        beta[0]
        + beta[1] * x
        + beta[2] * x**2
        + beta[3] * (x - kink) * (x >= kink)
        + beta[4] * (x - kink) ** 2 * (x >= kink)
    )


# Test that a ValueError is raised when a ScikitLearnModel is provided to a RegressionKink object
def test_olsmodel_and_regressionkink():
    """RegressionKink does not support OLS models, so a ValueError should be raised"""

    with pytest.raises(ValueError):
        kink = 0.5
        df = setup_regression_kink_data(kink)
        _ = cp.RegressionKink(
            df,
            formula=f"y ~ 1 + x + I((x-{kink})*treated)",
            model=CustomLinearRegression(),
            kink_point=kink,
        )


# Test that a ValueError is raised when a ScikitLearnModel is provided to a InstrumentalVariable object
def test_olsmodel_and_iv():
    """InstrumentalVariable does not support OLS models, so a ValueError should be raised"""

    with pytest.raises(ValueError):
        df = cp.load_data("risk")
        instruments_formula = "risk  ~ 1 + logmort0"
        formula = "loggdp ~  1 + risk"
        instruments_data = df[["risk", "logmort0"]]
        data = df[["loggdp", "risk"]]
        _ = cp.InstrumentalVariable(
            instruments_data=instruments_data,
            data=data,
            instruments_formula=instruments_formula,
            formula=formula,
            model=CustomLinearRegression(),
        )


# Test that a ValueError is raised when a ScikitLearnModel is provided to a PrePostNEGD object
def test_olsmodel_and_prepostnegd():
    """PrePostNEGD does not support OLS models, so a ValueError should be raised"""

    with pytest.raises(ValueError):
        df = cp.load_data("anova1")
        _ = cp.PrePostNEGD(
            df,
            formula="post ~ 1 + C(group) + pre",
            group_variable_name="group",
            pretreatment_variable_name="pre",
            model=CustomLinearRegression(),
        )


# Test that a ValueError is raised when a ScikitLearnModel is provided to a InversePropensityWeighting object
def test_olsmodel_and_ipw():
    """InversePropensityWeighting does not support OLS models, so a ValueError should be raised"""

    with pytest.raises(ValueError):
        df = cp.load_data("nhefs")
        _ = cp.InversePropensityWeighting(
            df,
            formula="trt ~ 1 + age + race",
            outcome_variable="outcome",
            weighting_scheme="robust",
            model=CustomLinearRegression(),
        )
