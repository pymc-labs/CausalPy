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
Tests for IPW plotting with extreme propensity scores.

Regression tests for issue #645: plot_ate() and plot_balance_ecdf() crash
with ValueError when propensity scores include 0.0 or 1.0 due to
unguarded division.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import causalpy as cp

sample_kwargs = {
    "tune": 50,
    "draws": 100,
    "chains": 2,
    "cores": 2,
    "random_seed": 42,
}


@pytest.fixture(scope="module")
def ipw_result(mock_pymc_sample):
    """Create a fitted IPW result for testing."""
    df = cp.load_data("nhefs")
    return cp.InversePropensityWeighting(
        df,
        formula="trt ~ 1 + age + race",
        outcome_variable="outcome",
        weighting_scheme="robust",
        model=cp.pymc_models.PropensityScore(sample_kwargs=sample_kwargs),
    )


@pytest.fixture
def extreme_idata(ipw_result):
    """Create idata with some propensity scores at 0.0 and 1.0."""
    import copy

    idata = copy.deepcopy(ipw_result.idata)
    idata.posterior["p"][:, :, :5] = 0.0
    idata.posterior["p"][:, :, 5:10] = 1.0
    return idata


class TestPlotAteExtremeScores:
    """plot_ate must not crash when propensity scores hit 0 or 1."""

    @pytest.mark.parametrize("method", ["raw", "robust", "overlap"])
    def test_plot_ate_no_crash(self, ipw_result, extreme_idata, method):
        """Verify plot_ate renders without error for each weighting scheme."""
        fig, axs = ipw_result.plot_ate(
            idata=extreme_idata, method=method, prop_draws=1, ate_draws=5
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotBalanceEcdfExtremeScores:
    """plot_balance_ecdf must not crash when propensity scores hit 0 or 1."""

    @pytest.mark.parametrize("scheme", ["raw", "robust", "overlap"])
    def test_plot_balance_ecdf_no_crash(self, ipw_result, extreme_idata, scheme):
        """Verify plot_balance_ecdf renders without error for each weighting scheme."""
        fig, axs = ipw_result.plot_balance_ecdf(
            "age", idata=extreme_idata, weighting_scheme=scheme
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPreparePs:
    """Unit tests for _prepare_ps clipping behavior."""

    def test_clips_zeros(self, ipw_result):
        """Scores at 0.0 are clipped to eps."""
        ps = np.array([0.0, 0.5, 1.0])
        clipped = ipw_result._prepare_ps(ps)
        assert clipped[0] > 0.0
        assert clipped[2] < 1.0
        assert clipped[1] == 0.5

    def test_warns_on_extreme(self, ipw_result):
        """A warning is emitted when extreme scores are detected."""
        ps = np.array([0.0, 0.5, 1.0])
        with pytest.warns(UserWarning, match="Extreme propensity scores"):
            ipw_result._prepare_ps(ps)

    def test_no_warn_on_safe(self, ipw_result):
        """No warning when all scores are within bounds."""
        ps = np.array([0.3, 0.5, 0.7])
        # Should not warn
        clipped = ipw_result._prepare_ps(ps)
        np.testing.assert_array_equal(ps, clipped)
