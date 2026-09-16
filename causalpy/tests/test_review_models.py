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

from causalpy.pymc_models import InstrumentalVariableRegression


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
