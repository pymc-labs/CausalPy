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
"""Backward-compatibility tests for deprecated design-matrix aliases.

Each test verifies that:
1. Accessing the deprecated attribute triggers ``DeprecationWarning``.
2. The data returned by the deprecated path is identical to the new API.
"""

import warnings

import pandas as pd
import pytest
import xarray.testing as xrt
from sklearn.linear_model import LinearRegression

import causalpy as cp

# ---------------------------------------------------------------------------
# Helpers – lightweight experiment instances (OLS for speed)
# ---------------------------------------------------------------------------


def _make_did() -> cp.DifferenceInDifferences:
    df = cp.load_data("did")
    return cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )


def _make_rd() -> cp.RegressionDiscontinuity:
    df = cp.load_data("rd")
    return cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + x + treated + x:treated",
        treatment_threshold=0.5,
        model=LinearRegression(),
    )


def _make_its(mock_pymc_sample) -> cp.InterruptedTimeSeries:
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    return cp.InterruptedTimeSeries(
        df,
        treatment_time=pd.to_datetime("2017-01-01"),
        formula="y ~ 1 + t",
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={"random_seed": 42, "progressbar": False}
        ),
    )


def _make_sc(mock_pymc_sample) -> cp.SyntheticControl:
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
            }
        ),
    )


# ---------------------------------------------------------------------------
# Parametrised tests – formula-based (design["X"] / design["y"])
# ---------------------------------------------------------------------------

_FORMULA_CASES = [
    ("X", "design", "X"),
    ("y", "design", "y"),
]


@pytest.mark.parametrize("old_attr,dataset_attr,key", _FORMULA_CASES)
def test_deprecated_alias_did(old_attr, dataset_attr, key):
    result = _make_did()
    with pytest.warns(DeprecationWarning, match=old_attr):
        old_val = getattr(result, old_attr)
    new_val = getattr(result, dataset_attr)[key]
    xrt.assert_identical(old_val, new_val)


@pytest.mark.parametrize("old_attr,dataset_attr,key", _FORMULA_CASES)
def test_deprecated_alias_rd(old_attr, dataset_attr, key):
    result = _make_rd()
    with pytest.warns(DeprecationWarning, match=old_attr):
        old_val = getattr(result, old_attr)
    new_val = getattr(result, dataset_attr)[key]
    xrt.assert_identical(old_val, new_val)


# ---------------------------------------------------------------------------
# Parametrised tests – pre/post split (ITS)
# ---------------------------------------------------------------------------

_ITS_CASES = [
    ("pre_X", "pre_design", "X"),
    ("pre_y", "pre_design", "y"),
    ("post_X", "post_design", "X"),
    ("post_y", "post_design", "y"),
]


@pytest.mark.parametrize("old_attr,dataset_attr,key", _ITS_CASES)
def test_deprecated_alias_its(mock_pymc_sample, old_attr, dataset_attr, key):
    result = _make_its(mock_pymc_sample)
    with pytest.warns(DeprecationWarning, match=old_attr):
        old_val = getattr(result, old_attr)
    new_val = getattr(result, dataset_attr)[key]
    xrt.assert_identical(old_val, new_val)


# ---------------------------------------------------------------------------
# Parametrised tests – synthetic control
# ---------------------------------------------------------------------------

_SC_CASES = [
    ("datapre_control", "pre_design", "control"),
    ("datapre_treated", "pre_design", "treated"),
    ("datapost_control", "post_design", "control"),
    ("datapost_treated", "post_design", "treated"),
]


@pytest.mark.parametrize("old_attr,dataset_attr,key", _SC_CASES)
def test_deprecated_alias_sc(mock_pymc_sample, old_attr, dataset_attr, key):
    result = _make_sc(mock_pymc_sample)
    with pytest.warns(DeprecationWarning, match=old_attr):
        old_val = getattr(result, old_attr)
    new_val = getattr(result, dataset_attr)[key]
    xrt.assert_identical(old_val, new_val)


# ---------------------------------------------------------------------------
# Parametrised tests – synthetic difference-in-differences
# ---------------------------------------------------------------------------


def _make_sdid(mock_pymc_sample) -> cp.SyntheticDifferenceInDifferences:
    df = cp.load_data("sc")
    treatment_time = 70
    return cp.SyntheticDifferenceInDifferences(
        df,
        treatment_time,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.SyntheticDifferenceInDifferencesWeightFitter(
            sample_kwargs={"random_seed": 42, "progressbar": False}
        ),
    )


@pytest.mark.parametrize("old_attr,dataset_attr,key", _SC_CASES)
def test_deprecated_alias_sdid(mock_pymc_sample, old_attr, dataset_attr, key):
    result = _make_sdid(mock_pymc_sample)
    with pytest.warns(DeprecationWarning, match=old_attr):
        old_val = getattr(result, old_attr)
    new_val = getattr(result, dataset_attr)[key]
    xrt.assert_identical(old_val, new_val)


# ---------------------------------------------------------------------------
# ConvexHullCheck should NOT trigger any DeprecationWarning
# ---------------------------------------------------------------------------


def test_convex_hull_check_no_deprecation_warning(mock_pymc_sample):
    """ConvexHullCheck.run() must not trigger DeprecationWarning internally."""
    from causalpy.checks.convex_hull import ConvexHullCheck
    from causalpy.pipeline import PipelineContext

    sc = _make_sc(mock_pymc_sample)
    check = ConvexHullCheck()
    ctx = PipelineContext(data=sc.data)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        check.run(sc, ctx)


# ---------------------------------------------------------------------------
# AttributeError for truly missing attributes
# ---------------------------------------------------------------------------


def test_attribute_error_for_missing():
    """Accessing a truly missing attribute still raises AttributeError."""
    result = _make_did()
    with pytest.raises(AttributeError, match="no_such_attribute"):
        _ = result.no_such_attribute
