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

import pandas as pd

from causalpy.utils import (
    _is_variable_dummy_coded,
    _series_has_2_levels,
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
