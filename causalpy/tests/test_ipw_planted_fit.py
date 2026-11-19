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
"""``fit`` plus ``get_ate`` against a planted propensity and outcome."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import causalpy as cp

# Hand contrast, not a call into ``_compute_ate_*``.
#
# Stored rows are (x, t, y, ps) = (0, 1, 10, 0.25), (1, 1, 0, 0.75),
# (0, 0, 4, 0.25), (1, 0, 8, 0.75). The robust weight is P(t=1) / ps on
# the treated rows and P(t=0) / (1 - ps) on the control rows, and P(t) is
# 1/2. Treated weighted mean is (10 * 2 + 0 * 2/3) / 2 = 10. Control
# weighted mean is (4 * 2/3 + 8 * 2) / 2 = 28/3. Their difference is 2/3.
# The unweighted arm difference is 5 - 6 = -1, so a propensity-blind
# contrast has the wrong sign.
_PLANTED_PS = np.array([0.25, 0.75, 0.25, 0.75])
_PLANTED_ATE = 2.0 / 3.0
_UNWEIGHTED_DIFFERENCE = -1.0
_BETA = np.array(
    [
        np.log(0.25 / 0.75),
        np.log(0.75 / 0.25) - np.log(0.25 / 0.75),
    ]
)


def _frame():
    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0],
            "trt": [1, 1, 0, 0],
            "outcome": [10.0, 0.0, 4.0, 8.0],
        },
        index=[10, 20, 30, 40],
    )


def _planted_sample(*args, **kwargs):
    """Return the propensity implied by the design ``fit`` just built."""
    import pymc as pm

    model = pm.modelcontext(kwargs.get("model"))
    design = np.asarray(model._build_data_nodes["X"], dtype=float)
    propensity = 1.0 / (1.0 + np.exp(-(design @ _BETA)))
    posterior = xr.Dataset(
        {
            "p": (
                ("chain", "draw", "obs_ind"),
                propensity.reshape(1, 1, -1),
            )
        }
    )
    idata = xr.DataTree()
    idata["posterior"] = posterior
    return idata


def _empty_predictive(*args, **kwargs):
    idata = xr.DataTree()
    idata["posterior_predictive"] = xr.Dataset()
    return idata


def test_fit_get_ate_recovers_planted_propensity_and_outcome(
    mock_pymc_sample, monkeypatch
):
    """Public fit stores the planted propensity and get_ate uses the outcome.

    The sampler stand-in is a function of the built design, not a copy of
    the expected propensity vector. Prior sampling is disabled so this check
    does not draw. ``mock_pymc_sample`` remains requested so a later edit
    that reaches ``pm.sample`` still cannot run NUTS.
    """
    import pymc as pm

    monkeypatch.setattr(pm, "sample", _planted_sample)
    monkeypatch.setattr(pm, "sample_posterior_predictive", _empty_predictive)
    monkeypatch.setattr(
        cp.pymc_models.PropensityScore,
        "supports_prior_predictive",
        False,
    )
    experiment = cp.InversePropensityWeighting(
        _frame(),
        formula="trt ~ 1 + x",
        outcome_variable="outcome",
        weighting_scheme="robust",
        model=cp.pymc_models.PropensityScore(
            sample_kwargs={"draws": 1, "tune": 1, "chains": 1, "progressbar": False}
        ),
    ).fit()

    stored = experiment.idata["posterior"]["p"].stack(z=("chain", "draw"))[:, 0].values
    ate = experiment.get_ate(0, experiment.idata, method="robust")[0]

    np.testing.assert_allclose(stored, _PLANTED_PS, atol=1e-8)
    assert ate == pytest.approx(_PLANTED_ATE, abs=1e-8)
    assert ate > 0
    assert not np.isclose(ate, _UNWEIGHTED_DIFFERENCE)
