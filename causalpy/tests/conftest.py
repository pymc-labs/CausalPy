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
"""
CausalPy Test Configuration

Functions:
* rng: random number generator with session level scope
"""

import numpy as np
import pytest
from pymc.testing import mock_sample, mock_sample_setup_and_teardown


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Random number generator that can persist through a pytest session"""
    seed: int = sum(map(ord, "causalpy"))
    return np.random.default_rng(seed=seed)


mock_pymc_sample = pytest.fixture(mock_sample_setup_and_teardown, scope="session")


@pytest.fixture(autouse=True)
def mock_sample_for_doctest(request):
    if not request.config.getoption("--doctest-modules", default=False):
        return

    import pymc as pm

    pm.sample = mock_sample
