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
Tests for the IPW treatment-column guard.

A bool or multi-level treatment column is expanded by patsy into one dummy
column per level, so the treatment design matrix has more than one column and
cannot back the single ``trt`` column of the outcome frame. The guard rejects
it at construction with a DataException instead of an opaque pandas error.
"""

import pytest

import causalpy as cp
from causalpy.custom_exceptions import DataException


@pytest.fixture
def nhefs():
    return cp.load_data("nhefs")


def test_bool_treatment_is_rejected_with_data_exception(nhefs):
    df = nhefs.assign(trt=nhefs["trt"].astype(bool))
    with pytest.raises(DataException, match="single 0-1 column"):
        cp.InversePropensityWeighting(
            df,
            formula="trt ~ 1 + age + race",
            outcome_variable="outcome",
            weighting_scheme="robust",
        )


def test_two_level_string_treatment_is_rejected_with_data_exception(nhefs):
    df = nhefs.assign(trt=nhefs["trt"].map({0: "no", 1: "yes"}))
    with pytest.raises(DataException, match="single 0-1 column"):
        cp.InversePropensityWeighting(
            df,
            formula="trt ~ 1 + age + race",
            outcome_variable="outcome",
            weighting_scheme="robust",
        )


def test_single_01_column_treatment_still_constructs(nhefs):
    exp = cp.InversePropensityWeighting(
        nhefs,
        formula="trt ~ 1 + age + race",
        outcome_variable="outcome",
        weighting_scheme="robust",
    )
    assert exp.t.shape[-1] == 1
