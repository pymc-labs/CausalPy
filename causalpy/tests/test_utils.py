"""
Tests for utility functions
"""

import pandas as pd

from causalpy.utils import _is_variable_dummy_coded, _series_has_2_levels


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
