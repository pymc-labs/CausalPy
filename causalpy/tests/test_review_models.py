#   Copyright 2026 - 2026 The PyMC Labs Developers
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
"""Regression coverage for model lifecycle review findings."""

import numpy as np
import pytest

from causalpy.pymc_models import InstrumentalVariableRegression, PropensityScore


def test_iv_clone_preserves_configuration_without_sharing_sampler_settings():
    """Sensitivity checks can clone IV models and tune the clone independently."""
    model = InstrumentalVariableRegression(
        sample_kwargs={"draws": 3, "cores": 1},
        prior_sample_kwargs={"draws": 7, "random_seed": 42},
        priors={"custom": 1},
    )
    cloned = model._clone()
    assert isinstance(cloned, InstrumentalVariableRegression)
    assert cloned.sample_kwargs == model.sample_kwargs
    assert cloned.prior_sample_kwargs == model.prior_sample_kwargs
    assert cloned.priors == model.priors
    assert cloned.idata is None
    cloned.sample_kwargs["draws"] = 11
    cloned.prior_sample_kwargs["draws"] = 13
    assert model.sample_kwargs["draws"] == 3
    assert model.prior_sample_kwargs["draws"] == 7
    assert model._clone(priors={"replacement": 2}).priors == {"replacement": 2}


@pytest.mark.parametrize("change", ["covariates", "treatment", "rows"])
def test_propensity_rebuild_rejects_changed_design(change):
    """A reused propensity graph cannot silently weight a different dataset."""
    X = np.arange(8, dtype=float).reshape(8, 1)
    treatment = np.tile([0, 1], 4)
    coords = {"obs_ind": np.arange(8), "coeffs": ["x"]}
    model = PropensityScore()
    model.build(X, treatment, coords=coords)
    model.build(X.copy(), treatment.copy(), coords=coords)
    if change == "covariates":
        X = X + 1
    elif change == "treatment":
        treatment = 1 - treatment
    else:
        X, treatment = X[:-1], treatment[:-1]
    with pytest.raises(RuntimeError, match="already built with different inputs"):
        model.build(X, treatment, coords=coords)
    np.testing.assert_array_equal(model["X"].get_value(), np.arange(8).reshape(8, 1))
    np.testing.assert_array_equal(model["t"].get_value(), np.tile([0, 1], 4))
