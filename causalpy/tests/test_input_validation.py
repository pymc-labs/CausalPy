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
"""Input validation tests"""

import numpy as np  # noqa: I001
import pandas as pd
import pytest
from matplotlib import pyplot as plt

import causalpy as cp
from causalpy.custom_exceptions import BadIndexException
from causalpy.custom_exceptions import DataException, FormulaException
from causalpy.tests.conftest import setup_regression_kink_data

from sklearn.linear_model import LinearRegression


sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}

# DiD


def test_did_validation_post_treatment_formula():
    """Test that we get a FormulaException for invalid formulas and missing post_treatment variables"""
    df = pd.DataFrame(
        {
            "group": [0, 0, 1, 1],
            "t": [0, 1, 0, 1],
            "unit": [0, 0, 1, 1],
            "post_treatment": [0, 1, 0, 1],
            "male": [0, 1, 0, 1],  # Additional variable for testing
            "y": [1, 2, 3, 4],
        }
    )

    df_with_custom = pd.DataFrame(
        {
            "group": [0, 0, 1, 1],
            "t": [0, 1, 0, 1],
            "unit": [0, 0, 1, 1],
            "custom_post": [0, 1, 0, 1],  # Custom column name
            "y": [1, 2, 3, 4],
        }
    )

    # Test 1: Missing post_treatment variable in formula
    with pytest.raises(FormulaException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group*post_SOMETHING",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )

    # Test 2: Missing post_treatment variable in formula (duplicate test)
    with pytest.raises(FormulaException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group*post_SOMETHING",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )

    # Test 3: Custom post_treatment_variable_name but formula uses different name
    with pytest.raises(FormulaException):
        _ = cp.DifferenceInDifferences(
            df_with_custom,
            formula="y ~ 1 + group*post_treatment",  # Formula uses 'post_treatment'
            time_variable_name="t",
            group_variable_name="group",
            post_treatment_variable_name="custom_post",  # But user specifies 'custom_post'
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )

    # Test 4: Default post_treatment_variable_name but formula uses different name
    with pytest.raises(FormulaException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group*custom_post",  # Formula uses 'custom_post'
            time_variable_name="t",
            group_variable_name="group",
            # post_treatment_variable_name defaults to "post_treatment"
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )

    # Test 5: Repeated interaction terms (should be invalid)
    with pytest.raises(FormulaException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group + group*post_treatment + group*post_treatment",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )

    # Test 6: Three-way interactions using * (should be invalid)
    with pytest.raises(FormulaException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group + group*post_treatment*male",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )

    # Test 7: Three-way interactions using : (should be invalid)
    with pytest.raises(FormulaException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group + group:post_treatment:male",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )

    # Test 8: Multiple different interaction terms using * (should be invalid)
    with pytest.raises(FormulaException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group + group*post_treatment + group*male",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )

    # Test 9: Multiple different interaction terms using : (should be invalid)
    with pytest.raises(FormulaException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group + group:post_treatment + group:male",
            time_variable_name="t",
            group_variable_name="group",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )

    # Test 10: Mixed issues - multiple terms + three-way interaction (should be invalid)
    with pytest.raises(FormulaException):
        _ = cp.DifferenceInDifferences(
            df,
            formula="y ~ 1 + group + group*post_treatment + group:post_treatment:male",
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

    # Test 2: Custom post_treatment_variable_name but column doesn't exist in data
    df_with_post = pd.DataFrame(
        {
            "group": [0, 0, 1, 1],
            "t": [0, 1, 0, 1],
            "unit": [0, 0, 1, 1],
            "post_treatment": [0, 1, 0, 1],  # Data has 'post_treatment'
            "y": [1, 2, 3, 4],
        }
    )

    with pytest.raises(DataException):
        _ = cp.DifferenceInDifferences(
            df_with_post,
            formula="y ~ 1 + group*custom_post",  # Formula uses 'custom_post'
            time_variable_name="t",
            group_variable_name="group",
            post_treatment_variable_name="custom_post",  # User specifies 'custom_post'
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
            control_units=["a", "b", "c", "d", "e", "f", "g"],
            treated_units=["actual"],
            model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
        )

    with pytest.raises(BadIndexException):
        df = cp.load_data("sc")
        treatment_time = pd.to_datetime("2016 June 24")
        _ = cp.SyntheticControl(
            df,
            treatment_time,
            control_units=["a", "b", "c", "d", "e", "f", "g"],
            treated_units=["actual"],
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
        _ = cp.SyntheticControl(
            df,
            treatment_time,
            control_units=other_countries,
            treated_units=[target_country],
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


# RegressionDiscontinuity


def setup_regression_discontinuity_data(threshold=0.5):
    """Create data for a regression discontinuity test."""
    np.random.seed(42)
    x = np.random.uniform(0, 1, 100)
    treated = np.where(x > threshold, 1, 0)
    y = 2 * x + treated + np.random.normal(0, 1, 100)
    return pd.DataFrame({"x": x, "treated": treated, "y": y})


def test_regression_discontinuity_int_treatment():
    """Test that RegressionDiscontinuity works with integer treatment variables."""
    threshold = 0.5
    df = setup_regression_discontinuity_data(threshold)
    assert df["treated"].dtype == np.int64  # Ensure treatment is int

    # This should work now with our fix
    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + x + treated + x:treated",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=threshold,
    )

    # Check that the treatment variable was converted to bool
    assert result.data["treated"].dtype == bool


def test_regression_discontinuity_bool_treatment():
    """Test that RegressionDiscontinuity works with boolean treatment variables."""
    threshold = 0.5
    df = setup_regression_discontinuity_data(threshold)
    df["treated"] = df["treated"].astype(bool)  # Convert to bool
    assert df["treated"].dtype == bool  # Ensure treatment is bool

    # This should work as before
    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + x + treated + x:treated",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=threshold,
    )

    # Check that the treatment variable is still bool
    assert result.data["treated"].dtype == bool


def test_rd_donut_hole_zero_same_as_default():
    """Test that donut_hole=0 reproduces current behavior (no filtering)."""
    threshold = 0.5
    df = setup_regression_discontinuity_data(threshold)

    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + x + treated + x:treated",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=threshold,
        donut_hole=0.0,
    )

    # With donut_hole=0, fit_data should equal data
    assert len(result.fit_data) == len(result.data)


def test_rd_donut_hole_filters_data():
    """Test that donut_hole filters out observations near threshold."""
    threshold = 0.5
    df = setup_regression_discontinuity_data(threshold)

    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + x + treated + x:treated",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=threshold,
        donut_hole=0.1,
    )

    # fit_data should have fewer observations than data
    assert len(result.fit_data) < len(result.data)

    # No observations in fit_data should be within 0.1 of threshold
    x_vals = result.fit_data["x"]
    assert all(np.abs(x_vals - threshold) >= 0.1)


def test_rd_donut_hole_with_bandwidth():
    """Test that donut_hole works correctly with bandwidth."""
    threshold = 0.5
    df = setup_regression_discontinuity_data(threshold)

    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + x + treated + x:treated",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=threshold,
        bandwidth=0.3,
        donut_hole=0.05,
    )

    # Check that fit_data respects both constraints
    x_vals = result.fit_data["x"]
    assert all(np.abs(x_vals - threshold) <= 0.3)  # within bandwidth
    assert all(np.abs(x_vals - threshold) >= 0.05)  # outside donut


def test_rd_donut_hole_validation_negative():
    """Test that negative donut_hole raises DataException."""
    threshold = 0.5
    df = setup_regression_discontinuity_data(threshold)

    with pytest.raises(DataException):
        cp.RegressionDiscontinuity(
            df,
            formula="y ~ 1 + x + treated + x:treated",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
            treatment_threshold=threshold,
            donut_hole=-0.1,
        )


def test_rd_donut_hole_validation_exceeds_bandwidth():
    """Test that donut_hole >= bandwidth raises DataException."""
    threshold = 0.5
    df = setup_regression_discontinuity_data(threshold)

    with pytest.raises(DataException):
        cp.RegressionDiscontinuity(
            df,
            formula="y ~ 1 + x + treated + x:treated",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
            treatment_threshold=threshold,
            bandwidth=0.3,
            donut_hole=0.3,  # Equal to bandwidth, should fail
        )

    with pytest.raises(DataException):
        cp.RegressionDiscontinuity(
            df,
            formula="y ~ 1 + x + treated + x:treated",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
            treatment_threshold=threshold,
            bandwidth=0.3,
            donut_hole=0.4,  # Greater than bandwidth, should fail
        )


def test_rd_running_variable_name_not_x():
    """Test that running_variable_name works correctly with non-default names."""
    np.random.seed(42)
    threshold = 21
    age = np.random.uniform(18, 25, 100)
    treated = np.where(age >= threshold, 1, 0)
    y = 2 * age + treated * 0.5 + np.random.normal(0, 1, 100)
    df = pd.DataFrame({"age": age, "treated": treated, "y": y})

    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + age + treated + age:treated",
        running_variable_name="age",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=threshold,
        bandwidth=2.0,
        donut_hole=0.5,
    )

    # Check that filtering works with non-default running variable name
    age_vals = result.fit_data["age"]
    assert all(np.abs(age_vals - threshold) <= 2.0)
    assert all(np.abs(age_vals - threshold) >= 0.5)


def test_rd_few_datapoints_warning():
    """Test that a warning is raised when bandwidth/donut_hole filter too aggressively."""
    threshold = 0.5
    df = setup_regression_discontinuity_data(threshold)

    # Use aggressive bandwidth that leaves very few datapoints
    with pytest.warns(UserWarning, match="remaining datapoints"):
        cp.RegressionDiscontinuity(
            df,
            formula="y ~ 1 + x + treated + x:treated",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
            treatment_threshold=threshold,
            bandwidth=0.05,  # Very narrow bandwidth
        )


def test_rd_few_datapoints_warning_with_donut():
    """Test warning when both bandwidth and donut_hole are mentioned."""
    threshold = 0.5
    df = setup_regression_discontinuity_data(threshold)

    # Use aggressive settings that leave very few datapoints
    with pytest.warns(UserWarning, match="bandwidth.*donut_hole"):
        cp.RegressionDiscontinuity(
            df,
            formula="y ~ 1 + x + treated + x:treated",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
            treatment_threshold=threshold,
            bandwidth=0.1,
            donut_hole=0.08,  # Large donut relative to bandwidth
        )


def test_rd_unrecognized_model_type():
    """Test that an unrecognized model type raises ValueError."""
    threshold = 0.5
    df = setup_regression_discontinuity_data(threshold)

    class FakeModel:
        """A fake model that is neither PyMCModel nor RegressorMixin."""

        pass

    with pytest.raises(ValueError, match="Model type not recognized"):
        cp.RegressionDiscontinuity(
            df,
            formula="y ~ 1 + x + treated + x:treated",
            model=FakeModel(),
            treatment_threshold=threshold,
        )


def test_rd_ols_plot_with_donut_hole():
    """Test that OLS plot shows donut hole boundary lines."""
    threshold = 0.5
    df = setup_regression_discontinuity_data(threshold)

    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + x + treated + x:treated",
        model=LinearRegression(),
        treatment_threshold=threshold,
        donut_hole=0.1,
    )

    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Check that donut boundary lines were added (2 orange dashed lines)
    donut_lines = [
        line
        for line in ax.get_lines()
        if line.get_linestyle() == "--" and line.get_color() == "orange"
    ]
    assert len(donut_lines) == 2, "Expected 2 donut boundary lines"
    plt.close(fig)


def test_rd_bayesian_plot_with_donut_hole():
    """Test that Bayesian plot shows donut hole boundary lines."""
    threshold = 0.5
    df = setup_regression_discontinuity_data(threshold)

    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + x + treated + x:treated",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=threshold,
        donut_hole=0.1,
    )

    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Check that donut boundary lines were added (2 orange dashed lines)
    donut_lines = [
        line
        for line in ax.get_lines()
        if line.get_linestyle() == "--" and line.get_color() == "orange"
    ]
    assert len(donut_lines) == 2, "Expected 2 donut boundary lines"
    plt.close(fig)


# Synthetic Control - Convex Hull Assumption


def test_synthetic_control_convex_hull_warning():
    """Test that SyntheticControl issues a warning when convex hull assumption is violated"""
    # Create synthetic data where treated is above all controls
    np.random.seed(42)
    n_time = 50
    time_idx = np.arange(n_time)

    # Create control units that are consistently lower than the treated
    controls = pd.DataFrame(
        {
            "control_1": 1.0 + 0.5 * time_idx + np.random.normal(0, 0.5, n_time),
            "control_2": 0.5 + 0.5 * time_idx + np.random.normal(0, 0.5, n_time),
            "control_3": 0.8 + 0.5 * time_idx + np.random.normal(0, 0.5, n_time),
        }
    )

    # Create treated unit that is consistently above all controls
    treated = 5.0 + 0.5 * time_idx + np.random.normal(0, 0.5, n_time)

    df = controls.copy()
    df["treated"] = treated

    treatment_time = 30

    # Should issue a warning
    with pytest.warns(UserWarning, match="Convex hull assumption may be violated"):
        result = cp.SyntheticControl(
            df,
            treatment_time,
            control_units=["control_1", "control_2", "control_3"],
            treated_units=["treated"],
            model=cp.skl_models.WeightedProportion(),
        )

    # The model should still run and produce results
    assert isinstance(result, cp.SyntheticControl)


def test_synthetic_control_no_warning_when_assumption_satisfied():
    """Test that SyntheticControl does not issue a warning when assumption is satisfied"""
    # Create synthetic data where treated is within control range
    np.random.seed(42)
    n_time = 50
    time_idx = np.arange(n_time)

    # Create control units with varying levels that span a wide range
    controls = pd.DataFrame(
        {
            "control_1": 1.0 + 0.5 * time_idx + np.random.normal(0, 0.3, n_time),
            "control_2": 0.0 + 0.5 * time_idx + np.random.normal(0, 0.3, n_time),
            "control_3": 4.0 + 0.5 * time_idx + np.random.normal(0, 0.3, n_time),
        }
    )

    # Create treated unit that falls within the control range
    treated = 2.0 + 0.5 * time_idx + np.random.normal(0, 0.2, n_time)

    df = controls.copy()
    df["treated"] = treated

    treatment_time = 30

    # Should NOT issue a warning
    import warnings

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        result = cp.SyntheticControl(
            df,
            treatment_time,
            control_units=["control_1", "control_2", "control_3"],
            treated_units=["treated"],
            model=cp.skl_models.WeightedProportion(),
        )

    # Check that no UserWarning about convex hull was issued
    convex_hull_warnings = [w for w in warning_list if "Convex hull" in str(w.message)]
    assert len(convex_hull_warnings) == 0

    # The model should run successfully
    assert isinstance(result, cp.SyntheticControl)
