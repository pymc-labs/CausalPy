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
"""Tests for the 1.0 removal of deprecated design-matrix aliases."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.experiments.base import BaseExperiment

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


def _make_panel(_mock_pymc_sample) -> cp.PanelRegression:
    data = pd.DataFrame(
        {
            "unit": [0, 0, 1, 1],
            "time": [0, 1, 0, 1],
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 1.0, 1.0, 2.0],
        }
    )
    return cp.PanelRegression(
        data,
        formula="y ~ x",
        unit_fe_variable="unit",
        time_fe_variable="time",
        model=LinearRegression(),
    )


def _make_piecewise(_mock_pymc_sample) -> cp.PiecewiseITS:
    return cp.PiecewiseITS(
        pd.DataFrame({"t": [0, 1, 2, 3], "y": [0.0, 1.0, 3.0, 4.0]}),
        formula="y ~ 1 + t + step(t, 2)",
        model=LinearRegression(),
    )


def _make_prepost(mock_pymc_sample) -> cp.PrePostNEGD:
    return cp.PrePostNEGD(
        cp.load_data("anova1"),
        formula="post ~ 1 + C(group) + pre",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={
                "chains": 1,
                "cores": 1,
                "draws": 5,
                "progressbar": False,
                "random_seed": 42,
                "tune": 5,
            }
        ),
    )


def _make_regression_kink(mock_pymc_sample) -> cp.RegressionKink:
    kink = 0.5
    x = np.linspace(-1, 1, 10)
    data = pd.DataFrame(
        {
            "x": x,
            "y": x + np.where(x >= kink, x - kink, 0),
            "treated": x >= kink,
        }
    )
    return cp.RegressionKink(
        data,
        formula=f"y ~ 1 + x + I((x - {kink}) * treated)",
        kink_point=kink,
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={
                "chains": 1,
                "cores": 1,
                "draws": 5,
                "progressbar": False,
                "random_seed": 42,
                "tune": 5,
            }
        ),
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
            sample_kwargs={
                "chains": 1,
                "cores": 1,
                "draws": 5,
                "progressbar": False,
                "random_seed": 42,
                "tune": 5,
            }
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
                "chains": 1,
                "cores": 1,
                "draws": 5,
                "progressbar": False,
                "random_seed": 42,
                "target_accept": 0.95,
                "tune": 5,
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


def _assert_removed_alias(result, old_attr, dataset_attr, key):
    """Assert the supported Dataset replacement works and the alias is absent."""
    assert key in getattr(result, dataset_attr)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(AttributeError, match=old_attr):
            getattr(result, old_attr)


_MIGRATED_EXPERIMENT_CLASSES = [
    cp.DifferenceInDifferences,
    cp.InterruptedTimeSeries,
    cp.PanelRegression,
    cp.PiecewiseITS,
    cp.PrePostNEGD,
    cp.RegressionDiscontinuity,
    cp.RegressionKink,
    cp.SyntheticControl,
    cp.SyntheticDifferenceInDifferences,
]


def test_deprecated_design_alias_forwarding_is_absent():
    """The base class no longer exposes the metadata or forwarding hook."""
    assert "_deprecated_design_aliases" not in BaseExperiment.__dict__
    assert "__getattr__" not in BaseExperiment.__dict__


@pytest.mark.parametrize("experiment_class", _MIGRATED_EXPERIMENT_CLASSES)
def test_deprecated_design_alias_metadata_is_absent(experiment_class):
    """Migrated experiments no longer retain inert alias metadata."""
    assert "_deprecated_design_aliases" not in experiment_class.__dict__


@pytest.mark.parametrize("old_attr,dataset_attr,key", _FORMULA_CASES)
def test_removed_alias_did(old_attr, dataset_attr, key):
    _assert_removed_alias(_make_did(), old_attr, dataset_attr, key)


@pytest.mark.parametrize("old_attr,dataset_attr,key", _FORMULA_CASES)
def test_removed_alias_rd(old_attr, dataset_attr, key):
    _assert_removed_alias(_make_rd(), old_attr, dataset_attr, key)


@pytest.mark.parametrize(
    "factory",
    [_make_panel, _make_piecewise, _make_prepost, _make_regression_kink],
)
@pytest.mark.parametrize("old_attr,dataset_attr,key", _FORMULA_CASES)
def test_removed_alias_remaining_formula_experiments(
    mock_pymc_sample, factory, old_attr, dataset_attr, key
):
    _assert_removed_alias(factory(mock_pymc_sample), old_attr, dataset_attr, key)


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
def test_removed_alias_its(mock_pymc_sample, old_attr, dataset_attr, key):
    _assert_removed_alias(_make_its(mock_pymc_sample), old_attr, dataset_attr, key)


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
def test_removed_alias_sc(mock_pymc_sample, old_attr, dataset_attr, key):
    _assert_removed_alias(_make_sc(mock_pymc_sample), old_attr, dataset_attr, key)


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
            sample_kwargs={
                "chains": 1,
                "cores": 1,
                "draws": 5,
                "progressbar": False,
                "random_seed": 42,
                "tune": 5,
            }
        ),
    )


@pytest.mark.parametrize("old_attr,dataset_attr,key", _SC_CASES)
def test_removed_alias_sdid(mock_pymc_sample, old_attr, dataset_attr, key):
    _assert_removed_alias(_make_sdid(mock_pymc_sample), old_attr, dataset_attr, key)


# ---------------------------------------------------------------------------
# AttributeError for truly missing attributes
# ---------------------------------------------------------------------------


def test_attribute_error_for_missing():
    """Accessing a truly missing attribute still raises AttributeError."""
    result = _make_did()
    with pytest.raises(AttributeError, match="no_such_attribute"):
        _ = result.no_such_attribute
