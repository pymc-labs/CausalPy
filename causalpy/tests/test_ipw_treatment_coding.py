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
"""IPW reads a boolean treatment as 0/1 and rejects text treatments (#1365)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import causalpy as cp
from causalpy.custom_exceptions import DataException

TRT = pd.Series([0, 1, 0, 1, 0, 0, 1, 1])


def _frame(trt=TRT):
    """Eight units, both arms, outcomes that are not a shift of each other."""
    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70, 80, 90],
            "race": [0, 1, 0, 1, 0, 1, 0, 1],
            "trt": trt,
            "outcome": [1.0, 4.0, 9.0, 2.5, 8.0, 3.0, 12.0, 6.5],
        }
    )


def _ipw(frame, lhs="trt"):
    return cp.InversePropensityWeighting(
        frame,
        formula=f"{lhs} ~ 1 + age + race",
        outcome_variable="outcome",
        weighting_scheme="robust",
    )


def _posterior(n, propensity):
    idata = xr.DataTree()
    idata["posterior"] = xr.Dataset(
        {"p": (("chain", "draw", "obs_ind"), np.full((1, 1, n), propensity))}
    )
    return idata


@pytest.mark.parametrize(
    ("trt", "lhs"),
    [
        pytest.param(TRT.astype(bool), "trt", id="bool"),
        pytest.param(TRT.astype(bool).astype(object), "trt", id="object-bool"),
        pytest.param(TRT, "I(trt == 1)", id="expression"),
        pytest.param(
            TRT.astype(bool), "C(trt, levels=[True, False])", id="true-level-first"
        ),
    ],
)
def test_boolean_treatment_matches_the_integer_treatment(trt, lhs):
    """Patsy codes a boolean treatment as two dummies; IPW keeps the True one."""
    expected = _ipw(_frame())
    experiment = _ipw(_frame(trt), lhs)

    assert experiment.t.shape == (len(TRT), 1)
    np.testing.assert_array_equal(experiment.t, expected.t)
    np.testing.assert_array_equal(experiment.X_outcome, expected.X_outcome)
    experiment.model.idata = _posterior(len(TRT), 0.4)
    expected.model.idata = _posterior(len(TRT), 0.4)
    for method in ["raw", "robust", "overlap", "doubly_robust"]:
        np.testing.assert_allclose(
            experiment.get_ate(0, experiment.idata, method=method),
            expected.get_ate(0, expected.idata, method=method),
        )


@pytest.mark.parametrize(
    ("trt", "lhs"),
    [
        pytest.param(TRT.map({0: "no", 1: "yes"}), "trt", id="text"),
        pytest.param(TRT.map({0: "False", 1: "True"}), "trt", id="text-true-false"),
        pytest.param(TRT.astype("category"), "trt", id="category"),
        pytest.param(TRT.astype(bool), "C(trt, Sum)", id="sum-contrast"),
        pytest.param(TRT, "trt + race", id="two-terms"),
    ],
)
def test_multi_column_treatment_is_rejected(trt, lhs):
    """A treatment that is not one 0/1 column raises a DataException."""
    with pytest.raises(DataException, match="must be one 0/1 or boolean column"):
        _ipw(_frame(trt), lhs)
