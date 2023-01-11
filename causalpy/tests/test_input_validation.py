import pandas as pd
import pytest

import causalpy as cp
from causalpy.custom_exceptions import DataException, FormulaException

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
        _ = cp.pymc_experiments.DifferenceInDifferences(
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
        _ = cp.pymc_experiments.DifferenceInDifferences(
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
        _ = cp.pymc_experiments.DifferenceInDifferences(
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
        _ = cp.pymc_experiments.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group*post_treatment",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )
