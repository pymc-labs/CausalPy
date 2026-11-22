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
"""Regression-kink gradient_change from stubbed probe predictions. No sampling."""

import numpy as np
import pandas as pd
import xarray as xr

import causalpy as cp
from causalpy.pymc_models import PyMCModel

_KINK = 0.5
_EPSILON = 0.2
_PROBE_X = np.array([_KINK - _EPSILON, _KINK, _KINK + _EPSILON])
_PROBE_TREATED = np.array([0.0, 1.0, 1.0])


def _probe_design(running: np.ndarray, treated: np.ndarray) -> np.ndarray:
    """Design the product builds for probes: intercept, running, kink interaction."""
    interaction = (running - _KINK) * treated
    return np.column_stack([np.ones(running.shape[0]), running, interaction])


def _planted_response(design: np.ndarray) -> np.ndarray:
    """Two deterministic responses of the probe design. Shape ``(draw, obs)``.

    Draw 0 uses the running variable, so a center-for-end swap changes the
    finite difference. Draw 1 uses only the interaction column.
    """
    running = design[:, 1]
    interaction = design[:, 2]
    return np.vstack(
        [
            10.0 * running + 3.0 * interaction,
            1.0 + 4.0 * interaction,
        ]
    )


def _finite_difference(mu: np.ndarray, epsilon: float) -> np.ndarray:
    return (mu[:, 2] - mu[:, 1]) / epsilon - (mu[:, 1] - mu[:, 0]) / epsilon


class _ProbePredictionModel(PyMCModel):
    """Return a closed-form response and record the design passed to predict."""

    def __init__(self) -> None:
        super().__init__()
        self.supports_prior_predictive = False
        self.predict_rows: list[np.ndarray] = []
        self.labels: list[str] = []

    def build(self, X, y, coords=None):  # noqa: ARG002
        self.labels = [str(name) for name in X.coords["coeffs"].values]
        self._built = True

    def sample_posterior(self, **kwargs):  # noqa: ARG002
        self.idata = xr.DataTree()
        return self.idata

    def predict(self, X, coords=None, out_of_sample=False, group="posterior"):  # noqa: ARG002
        rows = np.asarray(X, dtype=float)
        self.predict_rows.append(rows.copy())
        running_index = self.labels.index("x")
        interaction_index = next(
            index for index, name in enumerate(self.labels) if "treated" in name
        )
        design = np.column_stack(
            [np.ones(rows.shape[0]), rows[:, running_index], rows[:, interaction_index]]
        )
        mu = _planted_response(design)
        prediction = xr.DataArray(
            mu[None, :, :, None],
            dims=["chain", "draw", "obs_ind", "treated_units"],
            coords={
                "obs_ind": np.arange(rows.shape[0]),
                "treated_units": ["unit_0"],
            },
        )
        return xr.DataTree.from_dict(
            {"posterior_predictive": xr.Dataset({"mu": prediction})}
        )

    def score(self, X, y, coords=None):  # noqa: ARG002
        return pd.Series({"r2": 0.0, "r2_std": 0.0})


def test_regression_kink_gradient_change_uses_probe_predictions(mock_pymc_sample):
    """gradient_change is the finite difference of the three probe rows.

    The stored scalar is symmetric in the two end probes, so an end swap would
    leave it unchanged. The recorded probe design must therefore be the kink
    probes in order ``[kink - epsilon, kink, kink + epsilon]`` with treated
    coding ``[0, 1, 1]``. Moving the center probe changes the scalar.
    """
    del mock_pymc_sample
    frame = pd.DataFrame(
        {
            "x": [-1.0, -0.2, 0.4, 1.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "treated": [0, 0, 1, 1],
        }
    )
    model = _ProbePredictionModel()
    experiment = cp.RegressionKink(
        frame,
        formula=f"y ~ 1 + x + I((x-{_KINK})*treated)",
        kink_point=_KINK,
        epsilon=_EPSILON,
        model=model,
    )
    experiment.fit()

    probe_calls = [rows for rows in model.predict_rows if rows.shape[0] == 3]
    assert len(probe_calls) == 1
    probe = probe_calls[0]
    expected_design = _probe_design(_PROBE_X, _PROBE_TREATED)
    running_index = model.labels.index("x")
    interaction_index = next(
        index for index, name in enumerate(model.labels) if "treated" in name
    )
    np.testing.assert_allclose(probe[:, running_index], expected_design[:, 1])
    np.testing.assert_allclose(probe[:, interaction_index], expected_design[:, 2])
    expected = _finite_difference(_planted_response(expected_design), _EPSILON)
    np.testing.assert_allclose(expected, [3.0, 4.0])
    np.testing.assert_allclose(
        np.asarray(experiment.result.gradient_change).reshape(-1),
        expected,
    )
