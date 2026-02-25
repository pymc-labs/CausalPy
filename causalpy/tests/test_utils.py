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
Tests for utility functions
"""

import numpy as np
import pandas as pd
import pytest

import causalpy as cp
from causalpy.utils import (
    _is_variable_dummy_coded,
    _series_has_2_levels,
    check_convex_hull_violation,
    extract_lift_for_mmm,
    get_interaction_terms,
    round_num,
)


def test_dummy_coding():
    """Test if the function to check if a variable is dummy coded works correctly"""
    assert _is_variable_dummy_coded(pd.Series([False, True, False, True])) is True
    assert _is_variable_dummy_coded(pd.Series([False, True, False, "frog"])) is False
    assert _is_variable_dummy_coded(pd.Series([0, 0, 1, 0, 1])) is True
    assert _is_variable_dummy_coded(pd.Series([0, 0, 1, 0, 2])) is False
    assert _is_variable_dummy_coded(pd.Series([0, 0.5, 1, 0, 1])) is False


def test_2_level_series():
    """Test if the function to check if a variable has 2 levels works correctly"""
    assert _series_has_2_levels(pd.Series(["a", "a", "b"])) is True
    assert _series_has_2_levels(pd.Series(["a", "a", "b", "c"])) is False
    assert _series_has_2_levels(pd.Series(["coffee", "tea", "coffee"])) is True
    assert _series_has_2_levels(pd.Series(["water", "tea", "coffee"])) is False
    assert _series_has_2_levels(pd.Series([0, 1, 0, 1])) is True
    assert _series_has_2_levels(pd.Series([0, 1, 0, 2])) is False


def test_round_num():
    """Test if the function to round numbers works correctly"""
    assert round_num(0.12345, None) == "0.12"
    assert round_num(0.12345, 0) == "0.1"
    assert round_num(0.12345, 1) == "0.1"
    assert round_num(0.12345, 2) == "0.12"
    assert round_num(0.12345, 3) == "0.123"
    assert round_num(0.12345, 4) == "0.1235"
    assert round_num(0.12345, 5) == "0.12345"
    assert round_num(0.12345, 6) == "0.12345"
    assert round_num(123.456, None) == "123"
    assert round_num(123.456, 1) == "123"
    assert round_num(123.456, 2) == "123"
    assert round_num(123.456, 3) == "123"
    assert round_num(123.456, 4) == "123.5"
    assert round_num(123.456, 5) == "123.46"
    assert round_num(123.456, 6) == "123.456"
    assert round_num(123.456, 7) == "123.456"


def test_get_interaction_terms():
    """Test if the function to extract interaction terms from formulas works correctly"""
    # No interaction terms
    assert get_interaction_terms("y ~ x1 + x2 + x3") == []
    assert get_interaction_terms("y ~ 1 + x1 + x2") == []

    # Single interaction term with '*'
    assert get_interaction_terms("y ~ x1 + x2*x3") == ["x2*x3"]
    assert get_interaction_terms("y ~ 1 + group*post_treatment") == [
        "group*post_treatment"
    ]

    # Single interaction term with ':'
    assert get_interaction_terms("y ~ x1 + x2:x3") == ["x2:x3"]
    assert get_interaction_terms("y ~ 1 + group:post_treatment") == [
        "group:post_treatment"
    ]

    # Multiple interaction terms
    assert get_interaction_terms("y ~ x1*x2 + x3*x4") == ["x1*x2", "x3*x4"]
    assert get_interaction_terms("y ~ a:b + c*d") == ["a:b", "c*d"]

    # Three-way interaction
    assert get_interaction_terms("y ~ x1*x2*x3") == ["x1*x2*x3"]
    assert get_interaction_terms("y ~ a:b:c") == ["a:b:c"]

    # Formula with spaces (should be handled correctly)
    assert get_interaction_terms("y ~ x1 + x2 * x3") == ["x2*x3"]
    assert get_interaction_terms("y ~ 1 + group * post_treatment") == [
        "group*post_treatment"
    ]

    # Mixed main effects and interactions
    assert get_interaction_terms("y ~ 1 + x1 + x2 + x1*x2") == ["x1*x2"]
    assert get_interaction_terms("y ~ x1 + x2*x3 + x4") == ["x2*x3"]

    # Formula with subtraction (edge case)
    assert get_interaction_terms("y ~ x1*x2 - x3") == ["x1*x2"]


def test_check_convex_hull_violation_passes():
    """Test convex hull check when treated series is within control range"""
    # Treated series is within the range of controls at all time points
    treated = np.array([1.0, 2.0, 3.0])
    controls = np.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]])

    result = check_convex_hull_violation(treated, controls)

    assert result["passes"] is True
    assert result["n_violations"] == 0
    assert result["pct_above"] == 0.0
    assert result["pct_below"] == 0.0


def test_check_convex_hull_violation_above():
    """Test convex hull check when treated series is above control range"""
    # Treated series is above the control range for some points
    treated = np.array([5.0, 2.0, 3.0])
    controls = np.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]])

    result = check_convex_hull_violation(treated, controls)

    assert result["passes"] is False
    assert result["n_violations"] == 1
    assert result["pct_above"] == 100.0 / 3
    assert result["pct_below"] == 0.0


def test_check_convex_hull_violation_below():
    """Test convex hull check when treated series is below control range"""
    # Treated series is below the control range for some points
    treated = np.array([1.0, 0.5, 3.0])
    controls = np.array([[1.5, 2.5], [1.5, 2.5], [2.5, 3.5]])

    result = check_convex_hull_violation(treated, controls)

    assert result["passes"] is False
    assert result["n_violations"] == 2
    assert result["pct_above"] == 0.0
    assert result["pct_below"] == 200.0 / 3


def test_check_convex_hull_violation_both():
    """Test convex hull check when treated series is both above and below control range"""
    # Treated series is outside the control range in both directions
    treated = np.array([5.0, 0.5, 3.0])
    controls = np.array([[1.0, 2.0], [1.5, 2.5], [2.5, 3.5]])

    result = check_convex_hull_violation(treated, controls)

    assert result["passes"] is False
    assert result["n_violations"] == 2
    assert result["pct_above"] == 100.0 / 3
    assert result["pct_below"] == 100.0 / 3


def test_check_convex_hull_violation_boundary():
    """Test convex hull check when treated series is exactly on control boundaries"""
    # Treated series is exactly at the min and max of controls (should pass)
    treated = np.array([0.5, 2.5, 3.5])
    controls = np.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]])

    result = check_convex_hull_violation(treated, controls)

    assert result["passes"] is True
    assert result["n_violations"] == 0
    assert result["pct_above"] == 0.0
    assert result["pct_below"] == 0.0


def test_check_convex_hull_violation_empty_series():
    """Test convex hull check with empty arrays (no pre-intervention data)"""
    # Edge case: empty treated series (can occur when treatment_time is at start)
    treated = np.array([])
    controls = np.array([]).reshape(0, 3)  # 0 timepoints, 3 controls

    result = check_convex_hull_violation(treated, controls)

    # Should pass without error and return safe defaults
    assert result["passes"] is True
    assert result["n_violations"] == 0
    assert result["pct_above"] == 0.0
    assert result["pct_below"] == 0.0


# ============================================================================
# Tests for extract_lift_for_mmm
# ============================================================================


@pytest.fixture(scope="module")
def sc_result_single_unit():
    """Fixture for a SyntheticControl result with a single treated unit."""
    df = cp.load_data("sc")
    treatment_time = 70
    return cp.SyntheticControl(
        df,
        treatment_time,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.WeightedSumFitter(
            sample_kwargs={
                "target_accept": 0.95,
                "random_seed": 42,
                "progressbar": False,
                "draws": 100,
                "tune": 100,
            }
        ),
    )


@pytest.fixture(scope="module")
def sc_result_multi_unit():
    """Fixture for a SyntheticControl result with multiple treated units."""
    df = cp.load_data("geolift_multi_cell")
    # Set the time column as index and convert to datetime
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    treatment_time = pd.Timestamp("2020-01-05")
    return cp.SyntheticControl(
        df,
        treatment_time,
        control_units=["u1", "u2", "u3", "u4", "u5", "u6"],
        treated_units=["t1", "t2"],
        model=cp.pymc_models.WeightedSumFitter(
            sample_kwargs={
                "target_accept": 0.95,
                "random_seed": 42,
                "progressbar": False,
                "draws": 100,
                "tune": 100,
            }
        ),
    )


def test_extract_lift_for_mmm_single_unit(sc_result_single_unit):
    """Test extract_lift_for_mmm with a single treated unit."""
    result = extract_lift_for_mmm(
        sc_result_single_unit,
        channel="tv_campaign",
        x=0.0,
        delta_x=50000,
        aggregate="mean",
    )

    # Check that we get a DataFrame with the right structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1  # One treated unit
    assert list(result.columns) == [
        "channel",
        "geo",
        "x",
        "delta_x",
        "delta_y",
        "sigma",
    ]

    # Check values
    assert result.iloc[0]["channel"] == "tv_campaign"
    assert result.iloc[0]["geo"] == "actual"
    assert result.iloc[0]["x"] == 0.0
    assert result.iloc[0]["delta_x"] == 50000

    # delta_y and sigma should be floats
    assert isinstance(result.iloc[0]["delta_y"], float)
    assert isinstance(result.iloc[0]["sigma"], float)

    # sigma should be positive (it's a std)
    assert result.iloc[0]["sigma"] > 0


def test_extract_lift_for_mmm_multi_unit(sc_result_multi_unit):
    """Test extract_lift_for_mmm with multiple treated units."""
    result = extract_lift_for_mmm(
        sc_result_multi_unit,
        channel="radio",
        x=1000.0,
        delta_x=5000,
        aggregate="mean",
    )

    # Check that we get a DataFrame with one row per treated unit
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Two treated units (t1, t2)
    assert list(result.columns) == [
        "channel",
        "geo",
        "x",
        "delta_x",
        "delta_y",
        "sigma",
    ]

    # Check that both geos are represented
    geos = set(result["geo"])
    assert geos == {"t1", "t2"}

    # Check that channel and spend values are correct for all rows
    for _, row in result.iterrows():
        assert row["channel"] == "radio"
        assert row["x"] == 1000.0
        assert row["delta_x"] == 5000
        assert isinstance(row["delta_y"], float)
        assert isinstance(row["sigma"], float)
        assert row["sigma"] > 0


def test_extract_lift_for_mmm_sum_aggregate(sc_result_single_unit):
    """Test extract_lift_for_mmm with sum aggregation."""
    result_mean = extract_lift_for_mmm(
        sc_result_single_unit,
        channel="tv",
        x=0.0,
        delta_x=1000,
        aggregate="mean",
    )

    result_sum = extract_lift_for_mmm(
        sc_result_single_unit,
        channel="tv",
        x=0.0,
        delta_x=1000,
        aggregate="sum",
    )

    # Sum should be larger than mean (assuming multiple post-intervention periods)
    # The sum aggregates all periods, mean averages them
    n_post_periods = len(sc_result_single_unit.datapost)
    assert n_post_periods > 1

    # The sum of lift should roughly equal mean * n_periods
    # (with some tolerance for numerical precision)
    assert abs(result_sum.iloc[0]["delta_y"]) > abs(result_mean.iloc[0]["delta_y"])


def test_extract_lift_for_mmm_raises_for_ols():
    """Test that extract_lift_for_mmm raises an error for OLS models."""
    df = cp.load_data("sc")
    treatment_time = 70

    # Use sklearn model (OLS)
    from sklearn.linear_model import LinearRegression

    ols_result = cp.SyntheticControl(
        df,
        treatment_time,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.create_causalpy_compatible_class(LinearRegression)(),
    )

    with pytest.raises(ValueError, match="Bayesian"):
        extract_lift_for_mmm(
            ols_result,
            channel="tv",
            x=0.0,
            delta_x=1000,
        )
