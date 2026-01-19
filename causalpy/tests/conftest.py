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
"""
CausalPy Test Configuration

Functions:
* rng: random number generator with session level scope
"""

import numpy as np
import pytest

# Try to use PyMC's testing helpers if available; otherwise, fall back to no-op fixtures
try:  # pragma: no cover - conditional import for compatibility across PyMC versions
    from pymc.testing import mock_sample, mock_sample_setup_and_teardown  # type: ignore

    _HAVE_PYMC_TESTING = True
except Exception:  # pragma: no cover
    mock_sample = None  # type: ignore
    mock_sample_setup_and_teardown = None  # type: ignore
    _HAVE_PYMC_TESTING = False


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Random number generator that can persist through a pytest session"""
    seed: int = sum(map(ord, "causalpy"))
    return np.random.default_rng(seed=seed)


if _HAVE_PYMC_TESTING:
    mock_pymc_sample = pytest.fixture(mock_sample_setup_and_teardown, scope="session")
else:

    @pytest.fixture(scope="session")
    def mock_pymc_sample():  # pragma: no cover - compatibility no-op
        # No-op fixture to satisfy tests when PyMC testing helpers are unavailable
        yield


@pytest.fixture(autouse=True)
def mock_sample_for_doctest(request):
    if not request.config.getoption("--doctest-modules", default=False):
        return

    if not _HAVE_PYMC_TESTING or mock_sample is None:
        return
    import pymc as pm

    pm.sample = mock_sample
