import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    seed: int = sum(map(ord, "causalpy"))
    return np.random.default_rng(seed=seed)
