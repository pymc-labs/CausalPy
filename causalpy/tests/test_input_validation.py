#   Copyright 2025 The PyMC Labs Developers
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
"""Input validation tests"""

import numpy as np  # noqa: I001
import pandas as pd
import pytest

import causalpy as cp
from causalpy.custom_exceptions import BadIndexException
from causalpy.custom_exceptions import DataException, FormulaException

from sklearn.linear_model import LinearRegression


sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}

# DiD


def test_did_validation_post_treatment_formula():
    """Test that we get a FormulaException if do not include post_treatment in the
    formula"""
    df = pd.DataFrame(
        {
            "group": [0, 0, 1, 1],
            "t": [0, 1, 0, 1],
            "unit": [0, 0, 1, 1],
            "post_treatment": [0, 1, 0, 1],
            "y": [1, 2, 3, 4],
        }
    )

    with pytest.raises(FormulaException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group*post_SOMETHING",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )

    with pytest.raises(FormulaException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group*post_SOMETHING",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )


def test_did_validation_post_treatment_data():
    """Test that we get a DataException if do not include post_treatment in the data"""
    df = pd.DataFrame(
        {
            "group": [0, 0, 1, 1],
            "t": [0, 1, 0, 1],
            "unit": [0, 0, 1, 1],
            #    "post_treatment": [0, 1, 0, 1],
            "y": [1, 2, 3, 4],
        }
    )

    with pytest.raises(DataException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group*post_treatment",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )

    with pytest.raises(DataException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group*post_treatment",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )


def test_did_validation_unit_data():
    """Test that we get a DataException if do not include unit in the data"""
    df = pd.DataFrame(
        {
            "group": [0, 0, 1, 1],
            "t": [0, 1, 0, 1],
            # "unit": [0, 0, 1, 1],
            "post_treatment": [0, 1, 0, 1],
            "y": [1, 2, 3, 4],
        }
    )

    with pytest.raises(DataException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group*post_treatment",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )

    with pytest.raises(DataException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group*post_treatment",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )


def test_did_validation_group_dummy_coded():
    """Test that we get a DataException if the group variable is not dummy coded"""
    df = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "t": [0, 1, 0, 1],
            "unit": [0, 0, 1, 1],
            "post_treatment": [0, 1, 0, 1],
            "y": [1, 2, 3, 4],
        }
    )

    with pytest.raises(DataException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group*post_treatment",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )

    with pytest.raises(DataException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group*post_treatment",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )


# Synthetic Control


def test_sc_input_error():
    """Confirm that a BadIndexException is raised treatment_time is pd.Timestamp
    and df.index is not pd.DatetimeIndex."""
    with pytest.raises(BadIndexException):
        df = cp.load_data("sc")
        treatment_time = pd.to_datetime("2016 June 24")
        _ = cp.SyntheticControl(
            df,
            treatment_time,
            formula="actual ~ 0 + a + b + c + d + e + f + g",
            model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
        )

    with pytest.raises(BadIndexException):
        df = cp.load_data("sc")
        treatment_time = pd.to_datetime("2016 June 24")
        _ = cp.SyntheticControl(
            df,
            treatment_time,
            formula="actual ~ 0 + a + b + c + d + e + f + g",
            model=cp.skl_models.WeightedProportion(),
        )


def test_sc_brexit_input_error():
    """Confirm a BadIndexException is raised if the data index is datetime and the
    treatment time is not pd.Timestamp."""
    with pytest.raises(BadIndexException):
        df = cp.load_data("brexit")
        df["Time"] = pd.to_datetime(df["Time"])
        df.set_index("Time", inplace=True)
        df = df.iloc[df.index > "2009", :]
        treatment_time = "2016 June 24"  # NOTE This is not of type pd.Timestamp
        df = df.drop(["Japan", "Italy", "US", "Spain"], axis=1)
        target_country = "UK"
        all_countries = df.columns
        other_countries = all_countries.difference({target_country})
        all_countries = list(all_countries)
        other_countries = list(other_countries)
        formula = target_country + " ~ " + "0 + " + " + ".join(other_countries)
        _ = cp.SyntheticControl(
            df,
            treatment_time,
            formula=formula,
            model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
        )


# Pre-post NEGD


def test_ancova_validation_2_levels():
    """Test that we get a DataException group variable is not dummy coded"""
    df = pd.DataFrame(
        {
            "group": [0, 0, 1, 2],
            "pre": [1, 1, 3, 4],
            "post": [1, 2, 3, 4],
        }
    )

    with pytest.raises(DataException):
        _ = cp.PrePostNEGD(
            df,
            formula="post ~ 1 + C(group) + pre",
            group_variable_name="group",
            pretreatment_variable_name="pre",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )


# Regression discontinuity


def test_rd_validation_treated_in_formula():
    """Test that we get a FormulaException if treated is not in the model formula"""
    df = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "treated": [0, 0, 1, 1],
            "y": [1, 1, 2, 2],
        }
    )

    with pytest.raises(FormulaException):
        _ = cp.RegressionDiscontinuity(
            df,
            formula="y ~ 1 + x",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
            treatment_threshold=0.5,
        )

    with pytest.raises(FormulaException):
        _ = cp.RegressionDiscontinuity(
            df,
            formula="y ~ 1 + x",
            model=LinearRegression(),
            treatment_threshold=0.5,
        )


def test_rd_validation_treated_is_dummy():
    """Test that we get a DataException if treated is not dummy coded"""
    df = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "treated": ["control", "control", "treated", "treated"],
            "y": [1, 1, 2, 2],
        }
    )

    with pytest.raises(DataException):
        _ = cp.RegressionDiscontinuity(
            df,
            formula="y ~ 1 + x + treated",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
            treatment_threshold=0.5,
        )

    with pytest.raises(DataException):
        _ = cp.RegressionDiscontinuity(
            df,
            formula="y ~ 1 + x + treated",
            model=LinearRegression(),
            treatment_threshold=0.5,
        )


def test_iv_treatment_var_is_present():
    """Test the treatment variable is present for Instrumental Variable experiment"""
    data = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 5]})
    instruments_formula = "risk  ~ 1 + logmort0"
    formula = "loggdp ~  1 + risk"
    instruments_data = pd.DataFrame({"z": [1, 3, 4], "w": [2, 3, 4]})

    with pytest.raises(DataException):
        _ = cp.InstrumentalVariable(
            instruments_data=instruments_data,
            data=data,
            instruments_formula=instruments_formula,
            formula=formula,
            model=cp.pymc_models.InstrumentalVariableRegression(
                sample_kwargs=sample_kwargs
            ),
        )


# Regression kink design


def setup_regression_kink_data(kink):
    """Set up data for regression kink design tests"""
    # define parameters for data generation
    seed = 42
    rng = np.random.default_rng(seed)
    N = 50
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


def test_rkink_bandwidth_check():
    """Test that we get exceptions when bandwidth parameter is <= 0"""
    with pytest.raises(ValueError):
        kink = 0.5
        df = setup_regression_kink_data(kink)
        _ = cp.RegressionKink(
            df,
            formula=f"y ~ 1 + x + I((x-{kink})*treated)",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
            kink_point=kink,
            bandwidth=0,
        )

    with pytest.raises(ValueError):
        kink = 0.5
        df = setup_regression_kink_data(kink)
        _ = cp.RegressionKink(
            df,
            formula=f"y ~ 1 + x + I((x-{kink})*treated)",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
            kink_point=kink,
            bandwidth=-1,
        )


def test_rkink_epsilon_check():
    """Test that we get exceptions when epsilon parameter is <= 0"""
    with pytest.raises(ValueError):
        kink = 0.5
        df = setup_regression_kink_data(kink)
        _ = cp.RegressionKink(
            df,
            formula=f"y ~ 1 + x + I((x-{kink})*treated)",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
            kink_point=kink,
            epsilon=0,
        )

    with pytest.raises(ValueError):
        kink = 0.5
        df = setup_regression_kink_data(kink)
        _ = cp.RegressionKink(
            df,
            formula=f"y ~ 1 + x + I((x-{kink})*treated)",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
            kink_point=kink,
            epsilon=-1,
        )
