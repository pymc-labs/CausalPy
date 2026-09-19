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
    vs_prior = VariableSelectionPrior("spike_and_slab", hyperparams={"pi_alpha": 5})

    with pm.Model(coords=coords) as model:
        beta = vs_prior.create_prior(name="beta", n_params=5, dims="features")

        assert "beta" in model.named_vars
        assert beta.name == "beta"


def _create_spike_and_slab_prior(
    *, n_params=20, instance_hyperparams=None, call_hyperparams=None
):
    """Create a spike-and-slab prior and return its resolved implementation."""
    coords = {"features": [f"x_{i}" for i in range(n_params)]}
    vs_prior = VariableSelectionPrior(
        "spike_and_slab", hyperparams=instance_hyperparams
    )
    with pm.Model(coords=coords):
        vs_prior.create_prior(
            name="beta",
            n_params=n_params,
            dims="features",
            hyperparams=call_hyperparams,
        )
    assert isinstance(vs_prior._prior_instance, SpikeAndSlabPrior)
    return vs_prior._prior_instance


def test_expected_num_nonzero_instance_pair_resolves_sparse_prior():
    """Instance-level expected size maps to the requested Beta parameters."""
    prior = _create_spike_and_slab_prior(
        instance_hyperparams={
            "expected_num_nonzero": 1,
            "pi_concentration": 40,
        }
    )

    assert prior.pi_alpha == pytest.approx(2)
    assert prior.pi_beta == pytest.approx(38)


def test_expected_num_nonzero_call_pair_allows_boundary_heavy_prior():
    """An explicit low concentration may intentionally produce alpha below one."""
    prior = _create_spike_and_slab_prior(
        call_hyperparams={
            "expected_num_nonzero": 1,
            "pi_concentration": 4,
        }
    )

    assert prior.pi_alpha == pytest.approx(0.2)
    assert prior.pi_beta == pytest.approx(3.8)


def test_expected_num_nonzero_pair_can_be_split_across_layers():
    """Instance and call values combine using the ordinary precedence rules."""
    prior = _create_spike_and_slab_prior(
        instance_hyperparams={"expected_num_nonzero": 1},
        call_hyperparams={"pi_concentration": 40},
    )

    assert prior.pi_alpha == pytest.approx(2)
    assert prior.pi_beta == pytest.approx(38)


def test_call_hyperparams_override_one_expected_size_pair_member():
    """Call-level values override the matching instance-level value."""
    prior = _create_spike_and_slab_prior(
        instance_hyperparams={
            "expected_num_nonzero": 1,
            "pi_concentration": 40,
        },
        call_hyperparams={"pi_concentration": 4},
    )

    assert prior.pi_alpha == pytest.approx(0.2)
    assert prior.pi_beta == pytest.approx(3.8)


def test_default_spike_and_slab_parameters_are_unchanged():
    """The convenience pair does not change the existing default prior."""
    prior = _create_spike_and_slab_prior()

    assert prior.pi_alpha == 2
    assert prior.pi_beta == 2


@pytest.mark.parametrize("missing_key", ["expected_num_nonzero", "pi_concentration"])
def test_expected_size_pair_is_mandatory(missing_key):
    """Supplying only one convenience value is an error."""
    with pytest.raises(ValueError, match="must be provided together"):
        _create_spike_and_slab_prior(instance_hyperparams={missing_key: 1})


@pytest.mark.parametrize("explicit_key", ["pi_alpha", "pi_beta"])
def test_expected_size_pair_conflicts_with_explicit_beta_shapes(explicit_key):
    """Explicit shape keys conflict even when their value equals a default."""
    with pytest.raises(ValueError, match="cannot be combined"):
        _create_spike_and_slab_prior(
            instance_hyperparams={
                "expected_num_nonzero": 1,
                "pi_concentration": 40,
                explicit_key: 2,
            }
        )


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("expected_num_nonzero", 0, "greater than 0"),
        ("expected_num_nonzero", 20, "less than n_params"),
        ("expected_num_nonzero", np.nan, "finite numeric"),
        ("expected_num_nonzero", np.inf, "finite numeric"),
        ("expected_num_nonzero", "1", "finite numeric"),
        ("pi_concentration", 0, "positive finite"),
        ("pi_concentration", -1, "positive finite"),
        ("pi_concentration", np.nan, "positive finite"),
        ("pi_concentration", np.inf, "positive finite"),
        ("pi_concentration", "40", "positive finite"),
    ],
)
def test_expected_size_pair_validates_values(key, value, match):
    """The convenience values must be finite and within their valid ranges."""
    hyperparams = {"expected_num_nonzero": 1, "pi_concentration": 40}
    hyperparams[key] = value

    with pytest.raises(ValueError, match=match):
        _create_spike_and_slab_prior(instance_hyperparams=hyperparams)


def test_create_prior_horseshoe(coords, sample_data):
    """Test create_prior for horseshoe."""
    vs_prior = VariableSelectionPrior("horseshoe")

    with pm.Model(coords=coords) as model:
        beta = vs_prior.create_prior(
            name="beta", n_params=5, dims="features", X=sample_data
        )

        assert "beta" in model.named_vars
        assert beta.name == "beta"


def test_create_prior_normal(coords, sample_data):
    """Test create_prior for horseshoe."""
    vs_prior = VariableSelectionPrior("normal")

    with pm.Model(coords=coords) as model:
        beta = vs_prior.create_prior(name="beta", n_params=5, dims="features")

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
