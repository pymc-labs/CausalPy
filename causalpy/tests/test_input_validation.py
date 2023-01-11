import pandas as pd
import pytest

import causalpy as cp
from causalpy.custom_exceptions import BadIndexException  # NOQA
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


# Synthetic Control


def test_sc_input_error():
    """Confirm that a BadIndexException is raised treatment_time is pd.Timestamp
    and df.index is not pd.DatetimeIndex."""
    with pytest.raises(BadIndexException):
        df = cp.load_data("sc")
        treatment_time = pd.to_datetime("2016 June 24")
        _ = cp.pymc_experiments.SyntheticControl(
            df,
            treatment_time,
            formula="actual ~ 0 + a + b + c + d + e + f + g",
            model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
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
        _ = cp.pymc_experiments.SyntheticControl(
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
        _ = cp.pymc_experiments.PrePostNEGD(
            df,
            formula="post ~ 1 + C(group) + pre",
            group_variable_name="group",
            pretreatment_variable_name="pre",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )
