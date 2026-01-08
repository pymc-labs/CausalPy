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

from causalpy.utils import (
    _is_variable_dummy_coded,
    _series_has_2_levels,
    check_convex_hull_violation,
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
