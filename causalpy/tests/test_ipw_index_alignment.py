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
"""IPW weights each row by its own propensity, not by index label."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import causalpy as cp


def _frame(index):
    """Eight units, both arms, outcomes that are not a shift of each other."""
    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70, 80, 90],
            "race": [0, 1, 0, 1, 0, 1, 0, 1],
            "trt": [0, 1, 0, 1, 0, 0, 1, 1],
            "outcome": [1.0, 4.0, 9.0, 2.5, 8.0, 3.0, 12.0, 6.5],
        },
        index=index,
    )


def _ipw(frame):
    return cp.InversePropensityWeighting(
        frame,
        formula="trt ~ 1 + age + race",
        outcome_variable="outcome",
        weighting_scheme="robust",
    )


def _posterior(n, propensity):
    posterior = xr.Dataset(
        {
            "p": (
                ("chain", "draw", "obs_ind"),
                np.full((1, 1, n), propensity),
            )
        }
    )
    idata = xr.DataTree()
    idata["posterior"] = posterior
    return idata


def _positional_contrasts(frame, propensity):
    """Horvitz-Thompson, raw, and overlap contrasts in stored row order."""
    y = frame["outcome"].to_numpy(dtype=float)
    t = frame["trt"].to_numpy(dtype=float)
    ps = np.full(len(frame), propensity)
    p_of_t = np.mean(t)
    i_robust = np.where(t == 1, p_of_t / ps, (1 - p_of_t) / (1 - ps))
    n_trt = np.sum(t == 1)
    n_ntrt = np.sum(t == 0)
    robust = (
        np.sum(y[t == 1] * i_robust[t == 1]) / n_trt
        - np.sum(y[t == 0] * i_robust[t == 0]) / n_ntrt
    )
    ps_arm = np.where(t == 1, ps, 1 - ps)
    i_raw = 1 / ps_arm
    raw = np.sum(y[t == 1] * i_raw[t == 1]) / len(y) - np.sum(
        y[t == 0] * i_raw[t == 0]
    ) / len(y)
    i_overlap = np.where(t == 1, 1 - ps, ps)
    overlap = np.sum(y[t == 1] * i_overlap[t == 1]) / np.sum(
        i_overlap[t == 1]
    ) - np.sum(y[t == 0] * i_overlap[t == 0]) / np.sum(i_overlap[t == 0])
    return raw, robust, overlap


@pytest.mark.parametrize(
    "index",
    [
        list(range(4, 12)),
        list(range(100, 108)),
        [100, 2, 7, 50, 3, 8, 0, 11],
    ],
)
def test_get_ate_pairs_each_outcome_with_its_own_propensity(index):
    """A non-positional index must not reattach outcomes by label.

    ``0..n-1`` overlap shifts rows. No overlap drops every outcome and returns
    0. A permuted index that still intersects ``0..n-1`` pairs the wrong units
    even if a fix only special-cases a slice.
    """
    frame = _frame(index)
    experiment = _ipw(frame)
    experiment.model.idata = _posterior(len(frame), 0.4)
    expected_raw, expected_robust, expected_overlap = _positional_contrasts(frame, 0.4)

    raw = experiment.get_ate(0, experiment.idata, method="raw")[0]
    robust = experiment.get_ate(0, experiment.idata, method="robust")[0]
    overlap = experiment.get_ate(0, experiment.idata, method="overlap")[0]

    np.testing.assert_allclose(raw, expected_raw, atol=1e-12)
    np.testing.assert_allclose(robust, expected_robust, atol=1e-12)
    np.testing.assert_allclose(overlap, expected_overlap, atol=1e-12)
    assert expected_raw != 0.0
    assert not np.isclose(raw, 0.0)


def test_default_index_matches_the_positional_contrast():
    """RangeIndex inputs keep the same contrast the non-aligned cases require."""
    frame = _frame(range(8))
    experiment = _ipw(frame)
    experiment.model.idata = _posterior(len(frame), 0.4)
    expected_raw, expected_robust, _expected_overlap = _positional_contrasts(frame, 0.4)

    np.testing.assert_allclose(
        experiment.get_ate(0, experiment.idata, method="raw")[0],
        expected_raw,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        experiment.get_ate(0, experiment.idata, method="robust")[0],
        expected_robust,
        atol=1e-12,
    )


def test_doubly_robust_uses_the_row_outcome_not_the_index_label():
    """Doubly robust OLS must fit and residualize each row's own outcome."""
    frame = _frame([100, 2, 7, 50, 3, 8, 0, 11])
    experiment = _ipw(frame)
    propensity = np.full(len(frame), 0.4)
    ate, treated, control = experiment._compute_ate_doubly_robust(propensity)

    y = frame["outcome"].to_numpy(dtype=float)
    t = frame["trt"].to_numpy(dtype=float)
    design = np.column_stack(
        [
            np.ones(len(frame)),
            frame["age"].to_numpy(dtype=float),
            frame["race"].to_numpy(dtype=float),
        ]
    )
    control_rows = t == 0
    treated_rows = t == 1
    m0 = np.linalg.lstsq(design[control_rows], y[control_rows], rcond=None)[0]
    m1 = np.linalg.lstsq(design[treated_rows], y[treated_rows], rcond=None)[0]
    m0_pred = design @ m0
    m1_pred = design @ m1
    expected_control = np.mean((1 - t) * (y - m0_pred) / (1 - propensity) + m0_pred)
    expected_treated = np.mean(t * (y - m1_pred) / propensity + m1_pred)

    np.testing.assert_allclose(control, expected_control, atol=1e-10)
    np.testing.assert_allclose(treated, expected_treated, atol=1e-10)
    np.testing.assert_allclose(ate, expected_treated - expected_control, atol=1e-10)
