"""
CausalPy Test Configuration

Functions:
* rng: random number generator with session level scope
"""
import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Random number generator that can persist through a pytest session"""
    seed: int = sum(map(ord, "causalpy"))
    return np.random.default_rng(seed=seed)
