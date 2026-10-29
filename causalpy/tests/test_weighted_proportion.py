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
"""Regression tests for WeightedProportion root-mean-squared loss."""

import numpy as np
import pytest

import causalpy as cp
from causalpy.skl_models import WeightedProportion


def _noiseless_donors():
    """Donor matrix whose only convex combination of the outcome is ``[1, 0, 0]``."""
    rng = np.random.default_rng(0)
    donors = rng.normal(size=(40, 3))
    donors[:, 1] += 0.2
    donors[:, 2] += 0.4
    outcome = donors @ np.array([1.0, 0.0, 0.0])
    return donors, outcome


def _rmse(donors, outcome, weights):
    residual = np.ravel(outcome) - donors @ np.ravel(weights)
    return np.sqrt(np.mean(residual**2))


def test_loss_does_not_broadcast_a_row_of_weights_against_a_column_outcome():
    donors, outcome = _noiseless_donors()
    weights = np.atleast_2d(np.array([1.0, 0.0, 0.0]))
    column = outcome.reshape(-1, 1)

    reported = WeightedProportion().loss(weights, donors, column)

    assert reported == pytest.approx(0.0)
    assert reported == pytest.approx(_rmse(donors, column, weights))
    # A 1-d outcome against an (n, 1) prediction used to broadcast to (n, n).
    broadcast = np.sqrt(np.mean((outcome - donors @ weights.T) ** 2))
    assert broadcast > 1.0
    assert WeightedProportion().loss(weights, donors, outcome) != pytest.approx(
        broadcast
    )


def test_column_outcome_recovers_the_same_weights_as_a_vector_outcome():
    donors, outcome = _noiseless_donors()
    vector_fit = WeightedProportion().fit(donors, outcome)
    column_fit = WeightedProportion().fit(donors, outcome.reshape(-1, 1))

    np.testing.assert_allclose(np.ravel(vector_fit.coef_), [1.0, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(np.ravel(column_fit.coef_), np.ravel(vector_fit.coef_))
    assert vector_fit.mse == pytest.approx(0.0, abs=1e-8)
    assert column_fit.mse == pytest.approx(vector_fit.mse)
    np.testing.assert_allclose(np.ravel(column_fit.predict(donors)), outcome, atol=1e-8)


def test_synthetic_control_stores_preperiod_rmse():
    data = cp.load_data("sc")
    result = cp.SyntheticControl(
        data,
        70,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=WeightedProportion(),
    ).fit()
    donors = np.asarray(result.pre_design["control"])
    outcome = np.asarray(result.pre_design["treated"])
    expected = _rmse(donors, outcome, result.model.coef_)

    assert result.model.mse == pytest.approx(expected)
    assert result.model.mse == pytest.approx(0.2361371152075593)
