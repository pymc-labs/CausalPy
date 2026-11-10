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
"""Closed-form synthetic DiD weight recovery. No MCMC."""

import numpy as np
import pandas as pd
import xarray as xr

import causalpy as cp
from causalpy.pymc_models import PyMCModel


def _simplex_intercept_weights(
    donors: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray]:
    """Least squares with an intercept and weights summing to one.

    ``donors`` has one column per weighted series. The last weight is
    ``1 - sum(other weights)``, so the constraint is exact and a full-rank
    reduced design has one solution.
    """
    donors = np.asarray(donors, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)
    reduced = donors[:, :-1] - donors[:, -1:]
    design = np.column_stack([np.ones(donors.shape[0]), reduced])
    coef = np.linalg.lstsq(design, target - donors[:, -1], rcond=None)[0]
    free = coef[1:]
    weights = np.concatenate([free, [1.0 - float(np.sum(free))]])
    return float(coef[0]), weights


class _ClosedFormWeightModel(PyMCModel):
    """Replace the softmax weight sampler with the simplex least-squares solution.

    ``fit`` still builds the experiment's design matrices and reads the stored
    posterior. The only thing skipped is NUTS.
    """

    def __init__(self) -> None:
        super().__init__()
        self.supports_prior_predictive = False

    def build_mapping(self, X, y, coords=None):  # noqa: ARG002
        self._X = X
        self._y = y
        self._built = True

    def sample_posterior(self, **kwargs):  # noqa: ARG002
        unit_x = self._X["unit"].transpose("obs_ind", "coeffs")
        time_x = self._X["time"].transpose("coeffs", "obs_ind")
        omega0, omega = _simplex_intercept_weights(
            np.asarray(unit_x.values),
            np.asarray(self._y["unit"].values),
        )
        _lambda0, lam = _simplex_intercept_weights(
            np.asarray(time_x.values),
            np.asarray(self._y["time"].values),
        )
        posterior = xr.Dataset(
            {
                "omega": xr.DataArray(
                    omega.reshape(1, 1, -1),
                    dims=["chain", "draw", "coeffs"],
                    coords={"coeffs": list(unit_x.coords["coeffs"].values)},
                ),
                "omega0": xr.DataArray([[omega0]], dims=["chain", "draw"]),
                "lam": xr.DataArray(
                    lam.reshape(1, 1, -1),
                    dims=["chain", "draw", "obs_ind"],
                    coords={"obs_ind": np.asarray(time_x.coords["obs_ind"].values)},
                ),
            }
        )
        self.idata = xr.DataTree.from_dict({"posterior": posterior})
        return self.idata


def test_sdid_weight_estimation_recovers_planted_simplex_weights(mock_pymc_sample):
    """Unit and time weights are the unique simplex fit, not an injected vector.

    The treated pre-period mean is ``0.5 + 0.25 * donor_a + 0.75 * donor_b``.
    Post-period control means identify time weights ``[0.25, 0.75]`` with a
    zero intercept. Equal weights and the swapped vector are different
    solutions. The weight assertion is what rejects them.
    """
    del mock_pymc_sample
    panel = pd.DataFrame(
        {
            "time": [0, 1, 2, 3],
            "donor_a": [0.0, 1.0, 0.75, 0.75],
            "donor_b": [1.0, 3.0, 2.5, 2.5],
            "alpha": [1.5, 4.0, 6.5625, 6.5625],
            "beta": [1.0, 2.0, 6.5625, 6.5625],
        }
    )
    pre = panel.loc[panel["time"] < 2]
    treated_mean = pre[["alpha", "beta"]].mean(axis=1).to_numpy()
    unit_intercept, unit_weights = _simplex_intercept_weights(
        pre[["donor_a", "donor_b"]].to_numpy(),
        treated_mean,
    )
    post_control_mean = panel.loc[panel["time"] >= 2, ["donor_a", "donor_b"]].mean()
    time_intercept, time_weights = _simplex_intercept_weights(
        pre[["donor_a", "donor_b"]].to_numpy().T,
        post_control_mean.to_numpy(),
    )
    # The literals below are the plant. This only checks that the panel still
    # has that unique solution, so a later edit cannot make the oracle vacuous.
    np.testing.assert_allclose(unit_intercept, 0.5)
    np.testing.assert_allclose(unit_weights, [0.25, 0.75])
    np.testing.assert_allclose(time_intercept, 0.0, atol=1e-8)
    np.testing.assert_allclose(time_weights, [0.25, 0.75])
    assert not np.allclose(unit_weights, 0.5)

    experiment = cp.SyntheticDifferenceInDifferences(
        panel,
        treatment_time=2,
        control_units=["donor_a", "donor_b"],
        treated_units=["alpha", "beta"],
        time_column="time",
        model=_ClosedFormWeightModel(),
    )
    experiment.fit()

    posterior = experiment.idata["posterior"]
    omega = posterior["omega"].sel(coeffs=["donor_a", "donor_b"])
    lam = posterior["lam"]
    np.testing.assert_allclose(np.asarray(omega).reshape(-1), [0.25, 0.75])
    np.testing.assert_allclose(list(lam.coords["obs_ind"].values), [0, 1])
    np.testing.assert_allclose(np.asarray(lam).reshape(-1), [0.25, 0.75])
    # Zero only if finalize drops the double difference. Weight errors are
    # caught above; this panel's tau is not a second weight check.
    np.testing.assert_allclose(
        np.asarray(experiment.result.tau_posterior).reshape(-1),
        [4.0],
    )
