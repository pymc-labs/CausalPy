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
Unit tests for the InversePropensityWeighting get_ate method refactoring.

These tests verify that each of the extracted ATE computation methods
(_compute_ate_robust, _compute_ate_raw, _compute_ate_overlap, _compute_ate_doubly_robust)
work correctly and that the dispatch mechanism in get_ate functions as expected.
"""

import numpy as np
import pytest

import causalpy as cp


@pytest.fixture(scope="module")
def ipw_result(mock_pymc_sample):
    """Create a fitted InversePropensityWeighting result for testing.

    Uses minimal sampling parameters to speed up tests while still
    providing realistic posterior samples for ATE computation.
    """
    df = cp.load_data("nhefs")
    sample_kwargs = {
        "tune": 50,
        "draws": 100,
        "chains": 2,
        "cores": 2,
        "random_seed": 42,
    }
    result = cp.InversePropensityWeighting(
        df,
        formula="trt ~ 1 + age + race",
        outcome_variable="outcome",
        weighting_scheme="robust",
        model=cp.pymc_models.PropensityScore(sample_kwargs=sample_kwargs),
    )
    return result


@pytest.fixture
def propensity_scores(ipw_result):
    """Extract propensity scores from the first posterior sample."""
    return ipw_result.idata["posterior"]["p"].stack(z=("chain", "draw"))[:, 0].values


class TestComputeAteRobust:
    """Tests for the _compute_ate_robust method."""

    def test_returns_tuple_of_three_floats(self, ipw_result, propensity_scores):
        """Verify the method returns a tuple with three numeric values."""
        result = ipw_result._compute_ate_robust(propensity_scores)

        assert isinstance(result, tuple)
        assert len(result) == 3
        ate, trt, ntrt = result
        assert isinstance(ate, (int, float, np.floating))
        assert isinstance(trt, (int, float, np.floating))
        assert isinstance(ntrt, (int, float, np.floating))

    def test_ate_equals_trt_minus_ntrt(self, ipw_result, propensity_scores):
        """Verify ATE is computed as the difference between treated and non-treated."""
        ate, trt, ntrt = ipw_result._compute_ate_robust(propensity_scores)

        # Note: With mock sampling, propensity scores may be extreme (all 1s or 0s),
        # causing NaN values. We check the relationship holds or both are NaN.
        assert np.isclose(ate, trt - ntrt, equal_nan=True)


class TestComputeAteRaw:
    """Tests for the _compute_ate_raw method."""

    def test_returns_tuple_of_three_floats(self, ipw_result, propensity_scores):
        """Verify the method returns a tuple with three numeric values."""
        result = ipw_result._compute_ate_raw(propensity_scores)

        assert isinstance(result, tuple)
        assert len(result) == 3
        ate, trt, ntrt = result
        assert isinstance(ate, (int, float, np.floating))
        assert isinstance(trt, (int, float, np.floating))
        assert isinstance(ntrt, (int, float, np.floating))

    def test_ate_equals_trt_minus_ntrt(self, ipw_result, propensity_scores):
        """Verify ATE is computed as the difference between treated and non-treated."""
        ate, trt, ntrt = ipw_result._compute_ate_raw(propensity_scores)

        # Note: With mock sampling, propensity scores may be extreme (all 1s or 0s),
        # causing NaN values. We check the relationship holds or both are NaN.
        assert np.isclose(ate, trt - ntrt, equal_nan=True)


class TestComputeAteOverlap:
    """Tests for the _compute_ate_overlap method."""

    def test_returns_tuple_of_three_floats(self, ipw_result, propensity_scores):
        """Verify the method returns a tuple with three numeric values."""
        result = ipw_result._compute_ate_overlap(propensity_scores)

        assert isinstance(result, tuple)
        assert len(result) == 3
        ate, trt, ntrt = result
        assert isinstance(ate, (int, float, np.floating))
        assert isinstance(trt, (int, float, np.floating))
        assert isinstance(ntrt, (int, float, np.floating))

    def test_ate_equals_trt_minus_ntrt(self, ipw_result, propensity_scores):
        """Verify ATE is computed as the difference between treated and non-treated."""
        ate, trt, ntrt = ipw_result._compute_ate_overlap(propensity_scores)

        assert np.isclose(ate, trt - ntrt)


class TestComputeAteDoublyRobust:
    """Tests for the _compute_ate_doubly_robust method."""

    def test_returns_tuple_of_three_floats(self, ipw_result, propensity_scores):
        """Verify the method returns a tuple with three numeric values."""
        result = ipw_result._compute_ate_doubly_robust(propensity_scores)

        assert isinstance(result, tuple)
        assert len(result) == 3
        ate, trt, ntrt = result
        assert isinstance(ate, (int, float, np.floating))
        assert isinstance(trt, (int, float, np.floating))
        assert isinstance(ntrt, (int, float, np.floating))

    def test_ate_equals_trt_minus_ntrt(self, ipw_result, propensity_scores):
        """Verify ATE is computed as the difference between treated and non-treated."""
        ate, trt, ntrt = ipw_result._compute_ate_doubly_robust(propensity_scores)

        # Note: With mock sampling, propensity scores may be extreme (all 1s or 0s),
        # causing NaN values. We check the relationship holds or both are NaN.
        assert np.isclose(ate, trt - ntrt, equal_nan=True)


class TestGetAteDispatch:
    """Tests for the get_ate dispatch mechanism."""

    def test_get_ate_dispatches_to_robust(self, ipw_result):
        """Verify get_ate with method='robust' uses _compute_ate_robust."""
        ate_list = ipw_result.get_ate(0, ipw_result.idata, method="robust")
        ps = ipw_result.idata["posterior"]["p"].stack(z=("chain", "draw"))[:, 0].values
        ate, trt, ntrt = ipw_result._compute_ate_robust(ps)

        assert isinstance(ate_list, list)
        assert len(ate_list) == 3
        assert np.isclose(ate_list[0], ate, equal_nan=True)
        assert np.isclose(ate_list[1], trt, equal_nan=True)
        assert np.isclose(ate_list[2], ntrt, equal_nan=True)

    def test_get_ate_dispatches_to_raw(self, ipw_result):
        """Verify get_ate with method='raw' uses _compute_ate_raw."""
        ate_list = ipw_result.get_ate(0, ipw_result.idata, method="raw")
        ps = ipw_result.idata["posterior"]["p"].stack(z=("chain", "draw"))[:, 0].values
        ate, trt, ntrt = ipw_result._compute_ate_raw(ps)

        assert isinstance(ate_list, list)
        assert len(ate_list) == 3
        assert np.isclose(ate_list[0], ate, equal_nan=True)
        assert np.isclose(ate_list[1], trt, equal_nan=True)
        assert np.isclose(ate_list[2], ntrt, equal_nan=True)

    def test_get_ate_dispatches_to_overlap(self, ipw_result):
        """Verify get_ate with method='overlap' uses _compute_ate_overlap."""
        ate_list = ipw_result.get_ate(0, ipw_result.idata, method="overlap")
        ps = ipw_result.idata["posterior"]["p"].stack(z=("chain", "draw"))[:, 0].values
        ate, trt, ntrt = ipw_result._compute_ate_overlap(ps)

        assert isinstance(ate_list, list)
        assert len(ate_list) == 3
        assert np.isclose(ate_list[0], ate, equal_nan=True)
        assert np.isclose(ate_list[1], trt, equal_nan=True)
        assert np.isclose(ate_list[2], ntrt, equal_nan=True)

    def test_get_ate_dispatches_to_doubly_robust_by_default(self, ipw_result):
        """Verify get_ate with no method specified uses _compute_ate_doubly_robust."""
        ate_list = ipw_result.get_ate(0, ipw_result.idata)
        ps = ipw_result.idata["posterior"]["p"].stack(z=("chain", "draw"))[:, 0].values
        ate, trt, ntrt = ipw_result._compute_ate_doubly_robust(ps)

        assert isinstance(ate_list, list)
        assert len(ate_list) == 3
        assert np.isclose(ate_list[0], ate, equal_nan=True)
        assert np.isclose(ate_list[1], trt, equal_nan=True)
        assert np.isclose(ate_list[2], ntrt, equal_nan=True)

    def test_get_ate_with_explicit_doubly_robust(self, ipw_result):
        """Verify get_ate with method='doubly_robust' uses _compute_ate_doubly_robust."""
        ate_list = ipw_result.get_ate(0, ipw_result.idata, method="doubly_robust")
        ps = ipw_result.idata["posterior"]["p"].stack(z=("chain", "draw"))[:, 0].values
        ate, trt, ntrt = ipw_result._compute_ate_doubly_robust(ps)

        assert isinstance(ate_list, list)
        assert len(ate_list) == 3
        assert np.isclose(ate_list[0], ate, equal_nan=True)
        assert np.isclose(ate_list[1], trt, equal_nan=True)
        assert np.isclose(ate_list[2], ntrt, equal_nan=True)

    def test_get_ate_with_unknown_method_falls_back_to_doubly_robust(self, ipw_result):
        """Verify get_ate with unknown method falls back to doubly_robust."""
        ate_list = ipw_result.get_ate(0, ipw_result.idata, method="unknown_method")
        ps = ipw_result.idata["posterior"]["p"].stack(z=("chain", "draw"))[:, 0].values
        ate, trt, ntrt = ipw_result._compute_ate_doubly_robust(ps)

        assert isinstance(ate_list, list)
        assert len(ate_list) == 3
        assert np.isclose(ate_list[0], ate, equal_nan=True)
        assert np.isclose(ate_list[1], trt, equal_nan=True)
        assert np.isclose(ate_list[2], ntrt, equal_nan=True)
