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
"""Deterministic 2SLS recovery and instrument-frame alignment."""

import numpy as np
import pandas as pd
import pytest

import causalpy as cp

_PLANTED_TREATMENT = 3.0
_PLANTED_INTERCEPT = 2.0


def _planted_frame():
    """Outcome whose IV moment is exact and whose OLS slope is not.

    The structural error is the residual of the first-stage shock after
    projection onto the instrument, so two-stage least squares recovers the
    planted treatment coefficient. Ordinary least squares does not.
    """
    rng = np.random.default_rng(0)
    n = 200
    instrument = rng.normal(size=n)
    first_stage_shock = rng.normal(size=n)
    instrument_design = np.column_stack([np.ones(n), instrument])
    projection = instrument_design @ np.linalg.pinv(instrument_design)
    structural_error = first_stage_shock - projection @ first_stage_shock
    treatment = -1.0 + 4.0 * instrument + first_stage_shock
    outcome = _PLANTED_INTERCEPT + _PLANTED_TREATMENT * treatment + structural_error
    return pd.DataFrame({"y": outcome, "X": treatment, "Z": instrument})


def _fit_reference(frame):
    with pytest.warns(UserWarning, match="treatment variable is not Binary"):
        return cp.InstrumentalVariable(
            instruments_data=frame[["X", "Z"]],
            data=frame[["y", "X"]],
            instruments_formula="X ~ 1 + Z",
            formula="y ~ 1 + X",
        )


def test_2sls_recovers_planted_treatment_coefficient():
    """The second-stage treatment entry is the planted effect, not OLS."""
    frame = _planted_frame()
    experiment = _fit_reference(frame)
    treatment_index = experiment.labels.index("X")
    treatment = experiment.ols_beta_second_params[treatment_index]

    assert treatment == pytest.approx(_PLANTED_TREATMENT, abs=1e-8)
    assert treatment > 0
    assert abs(treatment - experiment.ols_beta_params["X"]) > 0.01


def test_reversed_instrument_rows_raise_before_2sls(monkeypatch):
    """A reversed instrument index is rejected before the 2SLS fit."""
    frame = _planted_frame()
    called = []

    def _forbidden(self):
        called.append(True)
        raise AssertionError("get_2SLS_fit ran on a reversed instrument frame")

    monkeypatch.setattr(cp.InstrumentalVariable, "get_2SLS_fit", _forbidden)
    with pytest.raises(ValueError, match="same order"):
        cp.InstrumentalVariable(
            instruments_data=frame[["X", "Z"]].iloc[::-1],
            data=frame[["y", "X"]],
            instruments_formula="X ~ 1 + Z",
            formula="y ~ 1 + X",
        )
    assert called == []
