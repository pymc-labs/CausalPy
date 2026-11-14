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
"""Closed-form PrePostNEGD coefficient recovery. No MCMC, no OLS backend."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import causalpy as cp
from causalpy.pymc_models import PyMCModel


class _ClosedFormCoefficientModel(PyMCModel):
    """Solve the experiment's design with least squares and skip NUTS.

    ``predict`` is unused by the coefficient assertion. It returns a zero
    predictive tree so scenario bookkeeping in ``fit`` can finish.
    """

    def __init__(self) -> None:
        super().__init__()
        self.supports_prior_predictive = False

    def build(self, X, y, coords=None):  # noqa: ARG002
        self._X = X
        self._y = y
        self._built = True

    def sample_posterior(self, **kwargs):  # noqa: ARG002
        design = np.asarray(self._X.values, dtype=float)
        outcome = np.asarray(self._y.values, dtype=float).reshape(-1)
        beta = np.linalg.lstsq(design, outcome, rcond=None)[0]
        labels = [str(name) for name in self._X.coords["coeffs"].values]
        posterior = xr.Dataset(
            {
                "beta": xr.DataArray(
                    beta.reshape(1, 1, -1),
                    dims=["chain", "draw", "coeffs"],
                    coords={"coeffs": labels},
                )
            }
        )
        self.idata = xr.DataTree.from_dict({"posterior": posterior})
        return self.idata

    def predict(self, X, coords=None, out_of_sample=False, group="posterior"):  # noqa: ARG002
        n_obs = np.asarray(X).shape[0]
        mu = xr.DataArray(
            np.zeros((1, 1, n_obs, 1)),
            dims=["chain", "draw", "obs_ind", "treated_units"],
            coords={
                "obs_ind": np.arange(n_obs),
                "treated_units": ["unit_0"],
            },
        )
        return xr.DataTree.from_dict({"posterior_predictive": xr.Dataset({"mu": mu})})

    def score(self, X, y, coords=None):  # noqa: ARG002
        return pd.Series({"r2": 0.0, "r2_std": 0.0})


def test_prepostnegd_recovers_planted_group_shift(mock_pymc_sample):
    """The reported coefficient is the planted group shift, not the raw contrast.

    ``post = 1.5 + 4 * group + 0.7 * pre``. Group 1 also has higher pre-period
    values, so the raw post-period mean difference is 6.1. A zero effect, a
    swapped group column, or that raw contrast must fail.
    """
    del mock_pymc_sample
    frame = pd.DataFrame(
        {
            "group": [0, 0, 0, 1, 1, 1],
            "pre": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "post": [1.5, 2.2, 2.9, 7.6, 8.3, 9.0],
        }
    )
    raw_contrast = float(
        frame.loc[frame["group"] == 1, "post"].mean()
        - frame.loc[frame["group"] == 0, "post"].mean()
    )
    assert raw_contrast == pytest.approx(6.1)
    hand_design = np.column_stack(
        [np.ones(len(frame)), frame["group"].to_numpy(), frame["pre"].to_numpy()]
    )
    hand_beta = np.linalg.lstsq(hand_design, frame["post"].to_numpy(), rcond=None)[0]
    np.testing.assert_allclose(hand_beta, [1.5, 4.0, 0.7])

    experiment = cp.PrePostNEGD(
        frame,
        formula="post ~ 1 + C(group) + pre",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=_ClosedFormCoefficientModel(),
    )
    experiment.fit()

    impact = float(np.asarray(experiment.result.causal_impact).reshape(-1)[0])
    assert impact == pytest.approx(4.0)
    assert impact != pytest.approx(raw_contrast)
    assert impact != pytest.approx(0.0)
