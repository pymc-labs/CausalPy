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
import pandas as pd
import pytest

import causalpy as cp

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


def reg_kink_function(x, beta, kink):
    """Piecewise linear function with a kink, for regression kink design tests."""
    return (
        beta[0]
        + beta[1] * x
        + beta[2] * x**2
        + beta[3] * (x - kink) * (x >= kink)
        + beta[4] * (x - kink) ** 2 * (x >= kink)
    )


def setup_regression_kink_data(kink):
    """Generate synthetic data for regression kink design tests."""
    rng = np.random.default_rng(42)
    N = 50
    beta = [0, -1, 0, 2, 0]
    sigma = 0.05
    x = rng.uniform(-1, 1, N)
    y = reg_kink_function(x, beta, kink) + rng.normal(0, sigma, N)
    return pd.DataFrame({"x": x, "y": y, "treated": x >= kink})


@pytest.fixture()
def banks_data():
    """Load and reshape the banks dataset for DiD integration tests.

    Returns (df_long, treatment_time) where treatment_time has been
    shifted to 0 so that tests don't depend on absolute years.
    """
    treatment_time = 1930.5
    df = (
        cp.load_data("banks")
        .filter(items=["bib6", "bib8", "year"])
        .rename(columns={"bib6": "Sixth District", "bib8": "Eighth District"})
        .groupby("year")
        .median()
    )
    df.index = df.index - treatment_time
    treatment_time = 0
    df.reset_index(level=0, inplace=True)
    df_long = pd.melt(
        df,
        id_vars=["year"],
        value_vars=["Sixth District", "Eighth District"],
        var_name="district",
        value_name="bib",
    ).sort_values("year")
    df_long["unit"] = df_long["district"]
    df_long["post_treatment"] = df_long.year >= treatment_time
    df_long = df_long.replace({"district": {"Sixth District": 1, "Eighth District": 0}})
    return df_long, treatment_time
