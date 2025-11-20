#   Copyright 2022 - 2025 The PyMC Labs Developers
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

import numpy as np
import pymc as pm
import pytest

from causalpy.variable_selection_priors import (
    HorseshoePrior,
    SpikeAndSlabPrior,
    VariableSelectionPrior,
    create_variable_selection_prior,
)


@pytest.fixture
def sample_data():
    """Generate sample design matrix for testing."""
    rng = np.random.default_rng(42)
    n_obs = 100
    n_features = 5
    X = rng.normal(size=(n_obs, n_features))
    return X


@pytest.fixture
def coords():
    """Generate sample coordinates for PyMC models."""
    return {"features": [f"x_{i}" for i in range(5)]}


def test_create_variable_in_model_context(coords):
    """Test that create_variable works in PyMC model context."""
    prior = SpikeAndSlabPrior(dims="features")

    with pm.Model(coords=coords) as model:
        beta = prior.create_variable("beta")

        # Check that beta was created
        assert "beta" in model.named_vars
        assert beta.name == "beta"

        # Check that intermediate variables were created
        assert "pi_beta" in model.named_vars
        assert "beta_raw" in model.named_vars
        assert "gamma_beta" in model.named_vars


def test_create_variable_in_model_context_horseshoe(coords):
    """Test that create_variable works in PyMC model context."""
    prior = HorseshoePrior(dims="features")

    with pm.Model(coords=coords) as model:
        beta = prior.create_variable("beta")

        # Check that beta was created
        assert "beta" in model.named_vars
        assert beta.name == "beta"

        # Check that intermediate variables were created
        assert "tau_beta" in model.named_vars
        assert "lambda_beta" in model.named_vars
        assert "c2_beta" in model.named_vars
        assert "lambda_tilde_beta" in model.named_vars
        assert "beta_raw" in model.named_vars


def test_create_prior_spike_and_slab(coords):
    """Test create_prior for spike-and-slab."""
    vs_prior = VariableSelectionPrior("spike_and_slab")

    with pm.Model(coords=coords) as model:
        beta = vs_prior.create_prior(name="beta", n_params=5, dims="features")

        assert "beta" in model.named_vars
        assert beta.name == "beta"


def test_create_prior_horseshoe(coords, sample_data):
    """Test create_prior for horseshoe."""
    vs_prior = VariableSelectionPrior("horseshoe")

    with pm.Model(coords=coords) as model:
        beta = vs_prior.create_prior(
            name="beta", n_params=5, dims="features", X=sample_data
        )

        assert "beta" in model.named_vars
        assert beta.name == "beta"


def test_convenience_function_with_custom_hyperparams(coords):
    """Test convenience function with custom hyperparameters."""
    with pm.Model(coords=coords) as model:
        _ = create_variable_selection_prior(
            prior_type="spike_and_slab",
            name="beta",
            n_params=5,
            dims="features",
            hyperparams={"slab_sigma": 5},
        )

        assert "beta" in model.named_vars
