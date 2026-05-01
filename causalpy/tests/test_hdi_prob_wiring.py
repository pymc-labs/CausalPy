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
Regression tests verifying that ``hdi_prob`` is wired through Bayesian plot
methods so that user-supplied values actually change the rendered HDI bands.

These tests guard against the regression described in GitHub issue
`pymc-labs/CausalPy#890`_, where ``result.plot(hdi_prob=...)`` was silently
swallowed by ``**kwargs`` rather than reaching the underlying
:func:`causalpy.plot_utils.plot_xY` calls.

.. _pymc-labs/CausalPy#890: https://github.com/pymc-labs/CausalPy/issues/890
"""

from unittest.mock import patch

import pandas as pd
import pytest

import causalpy as cp
from causalpy.constants import HDI_PROB

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


def _record_hdi_prob_calls(monkey_target: str):
    """Patch ``plot_xY`` to capture the ``hdi_prob`` kwarg of every call.

    Returns a context manager and the list that will be populated with the
    recorded values (as floats).
    """
    import causalpy.plot_utils as plot_utils

    real_plot_xY = plot_utils.plot_xY
    recorded: list[float] = []

    def spy(*args, **kwargs):
        recorded.append(kwargs.get("hdi_prob", HDI_PROB))
        return real_plot_xY(*args, **kwargs)

    return patch(monkey_target, side_effect=spy), recorded


@pytest.fixture(scope="module")
def fitted_its(its_data):
    """Fit a single ITS once and reuse across tests in this module."""
    treatment_time = pd.to_datetime("2017-01-01")
    return cp.InterruptedTimeSeries(
        its_data,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )


@pytest.mark.integration
@pytest.mark.parametrize("hdi_prob", [0.50, 0.75, 0.99])
def test_its_plot_threads_hdi_prob_to_plot_xY(mock_pymc_sample, fitted_its, hdi_prob):
    """User-supplied ``hdi_prob`` reaches every ``plot_xY`` call in ITS plot."""
    patcher, recorded = _record_hdi_prob_calls(
        "causalpy.experiments.interrupted_time_series.plot_xY"
    )
    with patcher:
        fitted_its.plot(hdi_prob=hdi_prob)

    assert len(recorded) > 0, "plot_xY was not invoked at all"
    assert all(value == hdi_prob for value in recorded), (
        f"Expected every plot_xY call to receive hdi_prob={hdi_prob}, "
        f"but recorded values were {recorded}"
    )


@pytest.mark.integration
def test_its_plot_default_hdi_prob_matches_constant(mock_pymc_sample, fitted_its):
    """When ``hdi_prob`` is omitted, the canonical default is forwarded."""
    patcher, recorded = _record_hdi_prob_calls(
        "causalpy.experiments.interrupted_time_series.plot_xY"
    )
    with patcher:
        fitted_its.plot()

    assert len(recorded) > 0, "plot_xY was not invoked at all"
    assert all(value == HDI_PROB for value in recorded), (
        f"Expected every plot_xY call to receive the default HDI_PROB="
        f"{HDI_PROB}, but recorded values were {recorded}"
    )
