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
"""Tests for method-specific sensitivity checks."""

from __future__ import annotations

import warnings
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.text import Text
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.checks.bandwidth import BandwidthSensitivity
from causalpy.checks.base import Check, CheckResult
from causalpy.checks.leave_one_out import LeaveOneOut
from causalpy.checks.mccrary import McCraryDensityTest, _adjacent_bin_counts
from causalpy.checks.placebo_in_space import (
    _PLACEBO_COLOUR,
    _TREATED_COLOUR,
    PlaceboInSpace,
    _mspe_stats,
    _permutation_pvalue,
)
from causalpy.experiments._results import CausalResult
from causalpy.experiments.regression_discontinuity import RegressionDiscontinuity
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pipeline import PipelineContext

# ---------------------------------------------------------------------------
# BandwidthSensitivity tests
# ---------------------------------------------------------------------------


class TestBandwidthSensitivity:
    """Tests for BandwidthSensitivity (Regression Discontinuity)."""

    def test_satisfies_check_protocol(self):
        assert isinstance(BandwidthSensitivity(), Check)

    def test_applicable_methods(self):
        check = BandwidthSensitivity()
        assert RegressionDiscontinuity in check.applicable_methods

    def test_validate_rejects_non_rd(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({"t": np.arange(n), "y": np.random.normal(size=n)})
        model = cp.create_causalpy_compatible_class(LinearRegression())
        its = cp.InterruptedTimeSeries(
            df, treatment_time=70, formula="y ~ 1 + t", model=model
        )
        with pytest.raises(TypeError, match="RegressionDiscontinuity"):
            BandwidthSensitivity().validate(its)

    def test_run_on_rd(self):
        df = cp.load_data("rd")
        model = cp.create_causalpy_compatible_class(LinearRegression())
        rd = RegressionDiscontinuity(
            df,
            formula="y ~ 1 + x + treated + x:treated",
            treatment_threshold=0.5,
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = rd
        ctx.experiment_config = {
            "method": RegressionDiscontinuity,
            "formula": "y ~ 1 + x + treated + x:treated",
            "treatment_threshold": 0.5,
            "model": cp.create_causalpy_compatible_class(LinearRegression()),
        }

        check = BandwidthSensitivity(bandwidths=[0.5, np.inf])
        check.validate(rd)
        result = check.run(rd, ctx)

        assert isinstance(result, CheckResult)
        assert result.check_name == "BandwidthSensitivity"
        assert result.table is not None
        assert len(result.table) == 2


# ---------------------------------------------------------------------------
# LeaveOneOut tests
# ---------------------------------------------------------------------------


class TestLeaveOneOut:
    """Tests for LeaveOneOut (Synthetic Control)."""

    def test_satisfies_check_protocol(self):
        assert isinstance(LeaveOneOut(), Check)

    def test_applicable_methods(self):
        assert SyntheticControl in LeaveOneOut().applicable_methods

    def test_validate_rejects_non_sc(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({"t": np.arange(n), "y": np.random.normal(size=n)})
        model = cp.create_causalpy_compatible_class(LinearRegression())
        its = cp.InterruptedTimeSeries(
            df, treatment_time=70, formula="y ~ 1 + t", model=model
        )
        with pytest.raises(TypeError, match="SyntheticControl"):
            LeaveOneOut().validate(its)

    def test_run_on_sc(self):
        df = cp.load_data("sc")
        controls = ["a", "b", "c", "d", "e", "f", "g"]
        model = cp.create_causalpy_compatible_class(LinearRegression())
        sc = SyntheticControl(
            df,
            treatment_time=70,
            control_units=controls,
            treated_units=["actual"],
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = sc
        ctx.experiment_config = {
            "method": SyntheticControl,
            "treatment_time": 70,
            "control_units": controls,
            "treated_units": ["actual"],
            "model": cp.create_causalpy_compatible_class(LinearRegression()),
        }

        check = LeaveOneOut()
        check.validate(sc)
        result = check.run(sc, ctx)

        assert isinstance(result, CheckResult)
        assert result.check_name == "LeaveOneOut"
        assert result.table is not None
        assert len(result.table) == len(controls)


# ---------------------------------------------------------------------------
# PlaceboInSpace tests
# ---------------------------------------------------------------------------


class TestPlaceboInSpace:
    """Tests for PlaceboInSpace (Synthetic Control)."""

    def test_satisfies_check_protocol(self):
        assert isinstance(PlaceboInSpace(), Check)

    def test_applicable_methods(self):
        assert SyntheticControl in PlaceboInSpace().applicable_methods

    def test_validate_rejects_non_sc(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({"t": np.arange(n), "y": np.random.normal(size=n)})
        model = cp.create_causalpy_compatible_class(LinearRegression())
        its = cp.InterruptedTimeSeries(
            df, treatment_time=70, formula="y ~ 1 + t", model=model
        )
        with pytest.raises(TypeError, match="SyntheticControl"):
            PlaceboInSpace().validate(its)

    def test_run_on_sc(self):
        df = cp.load_data("sc")
        controls = ["a", "b", "c"]
        model = cp.create_causalpy_compatible_class(LinearRegression())
        sc = SyntheticControl(
            df,
            treatment_time=70,
            control_units=controls,
            treated_units=["actual"],
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = sc
        ctx.experiment_config = {
            "method": SyntheticControl,
            "treatment_time": 70,
            "control_units": controls,
            "treated_units": ["actual"],
            "model": cp.create_causalpy_compatible_class(LinearRegression()),
        }

        check = PlaceboInSpace()
        check.validate(sc)
        result = check.run(sc, ctx)

        assert isinstance(result, CheckResult)
        assert result.check_name == "PlaceboInSpace"
        assert result.table is not None


# ---------------------------------------------------------------------------
# McCraryDensityTest tests
# ---------------------------------------------------------------------------


class TestMcCraryDensityTest:
    """Tests for McCraryDensityTest (Regression Discontinuity)."""

    def test_satisfies_check_protocol(self):
        assert isinstance(McCraryDensityTest(), Check)

    def test_applicable_methods(self):
        assert RegressionDiscontinuity in McCraryDensityTest().applicable_methods

    def test_validate_rejects_non_rd(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({"t": np.arange(n), "y": np.random.normal(size=n)})
        model = cp.create_causalpy_compatible_class(LinearRegression())
        its = cp.InterruptedTimeSeries(
            df, treatment_time=70, formula="y ~ 1 + t", model=model
        )
        with pytest.raises(TypeError, match="RegressionDiscontinuity"):
            McCraryDensityTest().validate(its)

    def test_run_on_rd(self):
        df = cp.load_data("rd")
        model = cp.create_causalpy_compatible_class(LinearRegression())
        rd = RegressionDiscontinuity(
            df,
            formula="y ~ 1 + x + treated + x:treated",
            treatment_threshold=0.5,
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = rd

        check = McCraryDensityTest()
        check.validate(rd)
        result = check.run(rd, ctx)

        assert isinstance(result, CheckResult)
        assert result.check_name == "McCraryDensityTest"
        assert result.passed is not None
        assert result.table is not None
        assert "z_statistic" in result.metadata
        assert "p_value" in result.metadata

    def test_balanced_data_passes(self):
        """Symmetric data around threshold should pass."""
        np.random.seed(42)
        x = np.concatenate(
            [np.random.uniform(0, 0.5, 50), np.random.uniform(0.5, 1, 50)]
        )
        y = np.random.normal(size=100)
        df = pd.DataFrame({"x": x, "y": y, "treated": (x >= 0.5).astype(int)})
        model = cp.create_causalpy_compatible_class(LinearRegression())
        rd = RegressionDiscontinuity(
            df,
            formula="y ~ 1 + x + treated",
            treatment_threshold=0.5,
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = rd

        result = McCraryDensityTest().run(rd, ctx)
        assert result.passed

    def test_empty_data_returns_none_passed(self):
        """When no data exists around the threshold, return inconclusive."""
        from unittest.mock import Mock

        mock_rd = Mock()
        mock_rd.treatment_threshold = 100
        mock_rd.running_variable_name = "x"
        mock_rd.data = pd.DataFrame({"x": [], "y": []})
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))
        result = McCraryDensityTest().run(mock_rd, ctx)
        assert result.passed is None
        assert "No observations" in result.text

    def _running_result(self, x, *, n_bins, threshold=0.0, alpha=0.05):
        from unittest.mock import Mock

        mock_rd = Mock()
        mock_rd.treatment_threshold = threshold
        mock_rd.running_variable_name = "x"
        mock_rd.data = pd.DataFrame({"x": np.asarray(x, dtype=float)})
        ctx = PipelineContext(data=mock_rd.data)
        return McCraryDensityTest(n_bins=n_bins, alpha=alpha).run(mock_rd, ctx)

    def test_compares_bins_adjacent_to_threshold(self):
        """Far-from-cutoff mass must not enter the density comparison.

        Global counts on this sample are balanced (101 vs 101). The bins
        touching the threshold are not (1 vs 100) when ``n_bins`` is 5.
        """
        x = np.concatenate(
            [
                np.full(100, -5.0),
                np.array([-0.1]),
                np.full(100, 0.1),
                np.array([5.0]),
            ]
        )

        local = self._running_result(x, n_bins=5)
        assert local.passed is False
        assert local.table.loc[0, "n_below"] == 1
        assert local.table.loc[0, "n_above"] == 100
        assert local.table.loc[0, "p_value"] == 0.0
        assert "below: 1, above: 100" in local.text

        # One bin on each side covers the whole support, which is balanced.
        whole = self._running_result(x, n_bins=1)
        assert whole.passed is True
        assert whole.table.loc[0, "n_below"] == 101
        assert whole.table.loc[0, "n_above"] == 101

    def test_local_balance_is_not_the_global_count(self):
        """A globally lopsided sample can still pass on the adjacent bins."""
        x = np.concatenate(
            [
                np.full(100, -5.0),
                np.full(50, -0.1),
                np.full(50, 0.1),
                np.array([5.0]),
            ]
        )
        result = self._running_result(x, n_bins=5)
        assert result.passed is True
        assert result.table.loc[0, "n_below"] == 50
        assert result.table.loc[0, "n_above"] == 50

    def test_n_bins_changes_which_mass_is_adjacent(self):
        """A coarser histogram must pull in mass a finer one leaves outside."""
        x = np.concatenate([np.full(30, -1.2), np.full(4, 0.2), np.full(30, 3.0)])
        fine = self._running_result(x, n_bins=5, threshold=0.0)
        coarse = self._running_result(x, n_bins=1, threshold=0.0)
        assert fine.table.loc[0, "n_below"] == 0
        assert int(fine.table.loc[0, "n_above"]) == 4
        assert coarse.table.loc[0, "n_below"] == 30
        assert coarse.table.loc[0, "n_above"] == 34

    def test_empty_adjacent_bin_is_not_a_pass(self):
        """An empty bin beside a positive bin is a density jump, not balance."""
        issue = np.concatenate(
            [np.full(100, -5.0), np.array([-0.1]), np.full(100, 0.1), np.array([5.0])]
        )
        narrow = self._running_result(issue, n_bins=50)
        assert narrow.table.loc[0, "n_below"] == 1
        assert narrow.table.loc[0, "n_above"] == 0
        assert narrow.passed is False
        assert "no evidence of manipulation" not in narrow.text

        one_sided = np.concatenate(
            [np.full(30, -1.2), np.full(4, 0.2), np.full(30, 3.0)]
        )
        fine = self._running_result(one_sided, n_bins=5)
        assert fine.table.loc[0, "n_below"] == 0
        assert int(fine.table.loc[0, "n_above"]) == 4
        assert fine.passed is False
        assert "no evidence of manipulation" not in fine.text

    def test_nonfinite_values_do_not_replace_adjacent_bins(self):
        """A non-finite running value must not widen the histogram to every point."""
        issue = np.concatenate(
            [np.full(100, -5.0), np.array([-0.1]), np.full(100, 0.1), np.array([5.0])]
        )
        for extra in (np.inf, -np.inf, np.nan):
            result = self._running_result(np.concatenate([issue, [extra]]), n_bins=5)
            assert result.passed is False
            assert result.table.loc[0, "n_below"] == 1
            assert result.table.loc[0, "n_above"] == 100
            assert "below: 1, above: 100" in result.text

    def test_left_endpoint_counted_when_span_fits_in_one_bin(self):
        """A one-bin left side includes its minimum even when the rebuilt edge misses it.

        ``threshold - width`` sits above the minimum for these samples, so a
        comparison against that edge drops the whole left side. For ``-1.9`` the
        rebuilt edge is exactly one ulp above the minimum. The right side already
        counts a side that fits in one bin; the left side must too.
        """
        threshold = 0.7
        cited = np.array([-0.2, 1.0])
        # Not the cited endpoint. ``0.7 - (0.7 - -1.9)`` is one ulp above -1.9.
        other = np.array([-1.9, 1.0])
        for sample, expected, one_ulp in (
            (cited, (1, 1), False),
            (np.concatenate([np.full(50, -0.2), np.full(50, 1.0)]), (50, 50), False),
            (other, (1, 1), True),
        ):
            minimum = float(np.min(sample[sample < threshold]))
            span = max(threshold - minimum, float(np.max(sample)) - threshold)
            edge = threshold - span
            assert edge > minimum
            if one_ulp:
                assert edge == np.nextafter(minimum, np.inf)
            assert _adjacent_bin_counts(sample, threshold, 1) == expected
            result = self._running_result(sample, n_bins=1, threshold=threshold)
            assert result.passed is True
            assert result.table.loc[0, "n_below"] == expected[0]
            assert result.table.loc[0, "n_above"] == expected[1]
            assert "no evidence of manipulation" in result.text

        # Integer sample at threshold 0 does not hit the ulp miss and still passes.
        integers = np.concatenate([np.full(100, -1), np.full(102, 1)])
        assert _adjacent_bin_counts(integers, 0.0, 1) == (100, 102)
        integer_result = self._running_result(integers, n_bins=1, threshold=0.0)
        assert integer_result.passed is True

    def test_mass_outside_longer_right_window_stays_out(self):
        """A longer right span still excludes mass beyond the adjacent bin."""
        # width = 5 / 5 = 1. 5.0 is outside [0, 1); -0.2 fits in the left bin.
        x = np.array([-0.2, 0.2, 5.0])
        assert _adjacent_bin_counts(x, 0.0, 5) == (1, 1)
        result = self._running_result(x, n_bins=5, threshold=0.0)
        assert result.table.loc[0, "n_below"] == 1
        assert result.table.loc[0, "n_above"] == 1

    def test_degenerate_histograms_do_not_invent_a_bin(self):
        """Invalid bin counts and zero-width or one-sided samples stay explicit."""
        with pytest.raises(ValueError, match="n_bins"):
            _adjacent_bin_counts(np.array([-1.0, 1.0]), 0.0, 0)

        # Every finite point sits on the threshold, so there is no width to split.
        assert _adjacent_bin_counts(np.array([0.5, 0.5]), 0.5, 4) == (0, 2)
        # A missing side must not set the width, and the present side is counted.
        assert _adjacent_bin_counts(np.array([0.2, 1.5]), 0.0, 1) == (0, 2)
        assert _adjacent_bin_counts(np.array([-1.5, -0.2]), 0.0, 1) == (2, 0)


# ---------------------------------------------------------------------------
# Edge-case and error-path tests
# ---------------------------------------------------------------------------


class TestBandwidthSensitivityEdgeCases:
    """Edge-case tests for BandwidthSensitivity."""

    def test_run_missing_experiment_config_raises(self):
        df = cp.load_data("rd")
        model = cp.create_causalpy_compatible_class(LinearRegression())
        rd = RegressionDiscontinuity(
            df,
            formula="y ~ 1 + x + treated + x:treated",
            treatment_threshold=0.5,
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = rd
        with pytest.raises(RuntimeError, match="experiment_config"):
            BandwidthSensitivity().run(rd, ctx)

    def test_run_handles_failing_bandwidth(self):
        df = cp.load_data("rd")
        model = cp.create_causalpy_compatible_class(LinearRegression())
        rd = RegressionDiscontinuity(
            df,
            formula="y ~ 1 + x + treated + x:treated",
            treatment_threshold=0.5,
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = rd
        ctx.experiment_config = {
            "method": RegressionDiscontinuity,
            "formula": "y ~ 1 + x + treated + x:treated",
            "treatment_threshold": 0.5,
            "model": cp.create_causalpy_compatible_class(LinearRegression()),
        }
        check = BandwidthSensitivity(bandwidths=[0.001])
        result = check.run(rd, ctx)
        assert isinstance(result, CheckResult)
        assert result.table is not None

    def test_run_handles_fitting_exception(self):
        df = cp.load_data("rd")
        model = cp.create_causalpy_compatible_class(LinearRegression())
        rd = RegressionDiscontinuity(
            df,
            formula="y ~ 1 + x + treated + x:treated",
            treatment_threshold=0.5,
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = rd
        ctx.experiment_config = {
            "method": RegressionDiscontinuity,
            "formula": "y ~ 1 + x + treated + x:treated",
            "treatment_threshold": 0.5,
            "model": model,
        }
        with patch.object(
            RegressionDiscontinuity,
            "__init__",
            side_effect=RuntimeError("simulated failure"),
        ):
            result = BandwidthSensitivity(bandwidths=[0.5]).run(rd, ctx)
        assert result.table is not None
        assert "error" in result.table.columns


class TestLeaveOneOutEdgeCases:
    """Edge-case tests for LeaveOneOut."""

    def test_run_missing_experiment_config_raises(self):
        df = cp.load_data("sc")
        model = cp.create_causalpy_compatible_class(LinearRegression())
        sc = SyntheticControl(
            df,
            treatment_time=70,
            control_units=["a", "b", "c"],
            treated_units=["actual"],
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = sc
        with pytest.raises(RuntimeError, match="experiment_config"):
            LeaveOneOut().run(sc, ctx)

    def test_run_with_single_control(self):
        df = cp.load_data("sc")
        model = cp.create_causalpy_compatible_class(LinearRegression())
        sc = SyntheticControl(
            df,
            treatment_time=70,
            control_units=["a"],
            treated_units=["actual"],
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = sc
        ctx.experiment_config = {
            "method": SyntheticControl,
            "treatment_time": 70,
            "control_units": ["a"],
            "treated_units": ["actual"],
            "model": model,
        }
        result = LeaveOneOut().run(sc, ctx)
        assert result.passed is None
        assert "fewer than 2" in result.text

    def test_run_handles_fitting_failure(self):
        df = cp.load_data("sc")
        controls = ["a", "b"]
        model = cp.create_causalpy_compatible_class(LinearRegression())
        sc = SyntheticControl(
            df,
            treatment_time=70,
            control_units=controls,
            treated_units=["actual"],
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = sc
        ctx.experiment_config = {
            "method": SyntheticControl,
            "treatment_time": 70,
            "control_units": controls,
            "treated_units": ["actual"],
            "model": model,
        }
        with patch.object(
            SyntheticControl,
            "__init__",
            side_effect=RuntimeError("simulated failure"),
        ):
            result = LeaveOneOut().run(sc, ctx)
        assert result.table is not None
        assert "error" in result.table.columns


class TestPlaceboInSpaceEdgeCases:
    """Edge-case tests for PlaceboInSpace."""

    def test_run_missing_experiment_config_raises(self):
        df = cp.load_data("sc")
        model = cp.create_causalpy_compatible_class(LinearRegression())
        sc = SyntheticControl(
            df,
            treatment_time=70,
            control_units=["a", "b", "c"],
            treated_units=["actual"],
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = sc
        with pytest.raises(RuntimeError, match="experiment_config"):
            PlaceboInSpace().run(sc, ctx)

    def test_run_with_single_control(self):
        df = cp.load_data("sc")
        model = cp.create_causalpy_compatible_class(LinearRegression())
        sc = SyntheticControl(
            df,
            treatment_time=70,
            control_units=["a"],
            treated_units=["actual"],
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = sc
        ctx.experiment_config = {
            "method": SyntheticControl,
            "treatment_time": 70,
            "control_units": ["a"],
            "treated_units": ["actual"],
            "model": model,
        }
        result = PlaceboInSpace().run(sc, ctx)
        assert result.passed is None
        assert "fewer than 2" in result.text

    def test_run_handles_fitting_failure(self):
        df = cp.load_data("sc")
        controls = ["a", "b"]
        model = cp.create_causalpy_compatible_class(LinearRegression())
        sc = SyntheticControl(
            df,
            treatment_time=70,
            control_units=controls,
            treated_units=["actual"],
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = sc
        ctx.experiment_config = {
            "method": SyntheticControl,
            "treatment_time": 70,
            "control_units": controls,
            "treated_units": ["actual"],
            "model": model,
        }
        with patch.object(
            SyntheticControl,
            "__init__",
            side_effect=RuntimeError("simulated failure"),
        ):
            result = PlaceboInSpace().run(sc, ctx)
        assert result.table is not None
        assert "error" in result.table.columns

    def test_run_skips_unit_with_no_donors(self):
        """When a control is also listed as treated, it should be skipped."""
        df = cp.load_data("sc")
        model = cp.create_causalpy_compatible_class(LinearRegression())
        sc = SyntheticControl(
            df,
            treatment_time=70,
            control_units=["a", "b"],
            treated_units=["actual"],
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = sc
        ctx.experiment_config = {
            "method": SyntheticControl,
            "treatment_time": 70,
            "control_units": ["a", "b"],
            "treated_units": ["a", "actual"],
            "model": model,
        }
        result = PlaceboInSpace().run(sc, ctx)
        assert isinstance(result, CheckResult)


# ---------------------------------------------------------------------------
# PlaceboInSpace MSPE-ratio tests
# ---------------------------------------------------------------------------


def _run_placebo_in_space(
    fit=True, treated=("actual",), config_treated=None, failing=(), check=None
):
    """Run the check on the bundled synthetic-control data.

    The experiment fits ``treated`` against controls a, b and c.
    ``config_treated`` is the treated list the config names, which can differ
    from the units the experiment fitted; it defaults to ``treated``.  The
    placebo fits for the units in ``failing`` raise.
    """
    if config_treated is None:
        config_treated = treated
    df = cp.load_data("sc")
    controls = ["a", "b", "c"]
    sc = SyntheticControl(
        df,
        treatment_time=70,
        control_units=controls,
        treated_units=list(treated),
        model=cp.create_causalpy_compatible_class(LinearRegression()),
    )
    if fit:
        sc.fit()
    ctx = PipelineContext(data=df)
    ctx.experiment = sc
    ctx.experiment_config = {
        "method": SyntheticControl,
        "treatment_time": 70,
        "control_units": controls,
        "treated_units": list(config_treated),
        "model": cp.create_causalpy_compatible_class(LinearRegression()),
    }
    real_init = SyntheticControl.__init__

    def init(self, *args, **kwargs):
        if set(kwargs["treated_units"]) & set(failing):
            raise RuntimeError("simulated failure")
        real_init(self, *args, **kwargs)

    with patch.object(SyntheticControl, "__init__", init):
        if check is None:
            check = PlaceboInSpace()
        return check.run(sc, ctx)


def _fake_result(pre_residuals, post_residuals, unit="actual"):
    """A result bundle whose impact arrays are the given residuals.

    Residuals are ``(obs,)``, repeated over two draws so the posterior mean is
    the residual itself, or ``(draw, obs)`` to vary them across draws.
    ``_mspe_stats`` reads only the impact fields, so the other fields reuse
    the same arrays.
    """

    def impact(values):
        values = np.atleast_2d(values)
        if len(values) == 1:
            values = np.tile(values, (2, 1))
        return xr.DataArray(
            values[np.newaxis, :, :, np.newaxis],
            dims=("chain", "draw", "obs_ind", "treated_units"),
            coords={"treated_units": [unit]},
        )

    pre = impact(pre_residuals)
    post = impact(post_residuals)
    return CausalResult(
        predictions_pre=pre,
        predictions_post=post,
        impact_pre=pre,
        impact_post=post,
        impact_post_cumulative=post,
    )


def _make_mspe_check_result(
    ratios=(1.0, 2.0, 3.0),
    baseline_ratios=None,
    units=("a", "b", "c"),
):
    """Build a CheckResult with hand-set ratios, skipping the model fits.

    ``baseline_ratios`` maps each actual treated unit to its ratio; pass an
    empty mapping for a result with no treated baseline at all.
    """
    if baseline_ratios is None:
        baseline_ratios = {"actual": 5.0}
    baseline_mspe = {
        unit: {"pre_mspe": 1.0, "post_mspe": ratio, "mspe_ratio": ratio}
        for unit, ratio in baseline_ratios.items()
    }
    return CheckResult(
        check_name="PlaceboInSpace",
        table=pd.DataFrame(
            {
                "placebo_treated": list(units),
                "pre_mspe": [1.0] * len(units),
                "post_mspe": list(ratios),
                "mspe_ratio": list(ratios),
            }
        ),
        metadata={"baseline_mspe": baseline_mspe},
    )


def _figure_texts(fig):
    """Collect every rendered string in a figure.

    plotnine draws the subtitle and legend labels as free text artists rather
    than through ``Axes.set_title``, so assertions read them from here.
    """
    return [text.get_text() for text in fig.findobj(Text)]


def _bar_colours(fig):
    """Face colours of the drawn bars, in plotted order."""
    poly = next(c for c in fig.axes[0].collections if isinstance(c, PolyCollection))
    return [to_hex(colour) for colour in poly.get_facecolor()]


def _tick_labels(fig):
    """Unit labels on the bar axis, in plotted order."""
    return [t.get_text() for t in fig.axes[0].get_yticklabels()]


class TestPlaceboInSpaceMspeRatio:
    """Tests for the MSPE columns and the ratio plot."""

    def test_mspe_stats_matches_hand_computed_errors(self):
        """The reported quantity is the MSPE ratio, not its square root.

        Pre-period residuals of 3 and 4 give an MSPE of 12.5, post-period
        residuals of 6 and 8 give 50, so the ratio is 4.  The square-root
        (RMSPE) version of the same ratio would be 2, which is what makes
        this test able to detect the wrong statistic.
        """
        result = _fake_result([3.0, 4.0], [6.0, 8.0])

        stats = _mspe_stats(result, "actual")

        assert stats["pre_mspe"] == pytest.approx(12.5)
        assert stats["post_mspe"] == pytest.approx(50.0)
        assert stats["mspe_ratio"] == pytest.approx(4.0)

    def test_mspe_stats_separates_the_degenerate_cases(self):
        """A zero pre-period error is infinite or undefined, never both."""
        positive_over_zero = _mspe_stats(_fake_result([0.0, 0.0], [1.0, 1.0]), "actual")
        zero_over_zero = _mspe_stats(_fake_result([0.0, 0.0], [0.0, 0.0]), "actual")
        non_finite_pre = _mspe_stats(_fake_result([np.nan, 1.0], [1.0, 1.0]), "actual")

        assert positive_over_zero["mspe_ratio"] == float("inf")
        assert np.isnan(zero_over_zero["mspe_ratio"])
        assert np.isnan(non_finite_pre["mspe_ratio"])

    def test_mspe_stats_averages_over_the_posterior_before_squaring(self):
        """Residuals of +2 and -2 across two draws average to 0 before squaring.

        Squaring each draw first would give an MSPE of 4 instead.
        """
        result = _fake_result([[2.0, 2.0, 2.0], [-2.0, -2.0, -2.0]], [2.0, 2.0])

        stats = _mspe_stats(result, "actual")

        assert stats["pre_mspe"] == pytest.approx(0.0)
        assert stats["post_mspe"] == pytest.approx(4.0)
        assert stats["mspe_ratio"] == float("inf")

    def test_run_reports_mspe_columns(self):
        """Every successful placebo fit carries its pre, post and ratio."""
        result = _run_placebo_in_space()

        for column in ("pre_mspe", "post_mspe", "mspe_ratio"):
            assert column in result.table.columns
        assert np.isfinite(result.table["mspe_ratio"]).all()
        np.testing.assert_allclose(
            result.table["mspe_ratio"],
            result.table["post_mspe"] / result.table["pre_mspe"],
        )

    def test_run_text_points_to_the_ratio_only_when_it_exists(self):
        """The text names the `mspe_ratio` column only when the table has it."""
        assert "mspe_ratio" in _run_placebo_in_space().text

        result = _run_placebo_in_space(failing=("a", "b", "c"))

        assert "mspe_ratio" not in result.table.columns
        assert "mspe_ratio" not in result.text

    def test_run_reports_baseline_mspe(self):
        """The actual treated unit's MSPEs land in metadata."""
        result = _run_placebo_in_space()

        baseline = result.metadata["baseline_mspe"]["actual"]
        assert set(baseline) == {"pre_mspe", "post_mspe", "mspe_ratio"}
        assert baseline["mspe_ratio"] == pytest.approx(
            baseline["post_mspe"] / baseline["pre_mspe"]
        )

    def test_run_skips_baseline_for_unfitted_treated_unit(self):
        """A treated unit the experiment never fitted gets no baseline MSPE.

        "a" is named treated in the config but absent from the fitted
        experiment, so selecting it would raise rather than yield a baseline.
        """
        result = _run_placebo_in_space(config_treated=("a", "actual"))

        assert set(result.metadata["baseline_mspe"]) == {"actual"}

    def test_run_skips_a_treated_unit_listed_as_control(self):
        """A unit listed as both control and treated is not its own placebo."""
        result = _run_placebo_in_space(treated=("a",))

        assert list(result.table["placebo_treated"]) == ["b", "c"]
        assert set(result.metadata["baseline_mspe"]) == {"a"}

    def test_run_skips_baseline_for_an_unfitted_experiment(self):
        """An experiment that was never fitted has no impact arrays to read."""
        result = _run_placebo_in_space(fit=False)

        assert result.metadata["baseline_mspe"] == {}
        assert np.isfinite(result.table["mspe_ratio"]).all()

    def test_run_surfaces_errors_in_the_statistic(self):
        """A bug in the MSPE computation is not reported as a failed fit."""
        with (
            patch(
                "causalpy.checks.placebo_in_space._mspe_stats",
                side_effect=KeyError("broken"),
            ),
            pytest.raises(KeyError, match="broken"),
        ):
            _run_placebo_in_space()

    def test_plot_accepts_a_treated_unit_listed_as_control(self):
        """The unit appears once, as treated, so the plot does not see it twice."""
        fig = PlaceboInSpace.plot_mspe_ratio(_run_placebo_in_space(treated=("a",)))

        assert sorted(_tick_labels(fig)) == ["a", "b", "c"]
        plt.close(fig)

    def test_plot_accepts_the_result_of_a_real_run(self):
        """run() and the plot agree on the contract, without a hand-built result.

        Every other plot test builds its ``CheckResult`` directly, so this is
        the only one that would catch the two drifting apart.
        """
        result = _run_placebo_in_space()

        fig = PlaceboInSpace.plot_mspe_ratio(result)

        labels = _tick_labels(fig)
        assert set(labels) == {"a", "b", "c", "actual"}
        treated_position = labels.index("actual")
        assert _bar_colours(fig)[treated_position] == _TREATED_COLOUR.lower()
        plt.close(fig)

    def test_plot_drops_a_unit_whose_placebo_fit_failed(self):
        """A failed fit leaves a NaN ratio, which the plot drops with a warning.

        The undefined path has two origins and they reach it differently: a
        zero post-period error over a zero pre-period error is set to NaN by
        ``_mspe_stats``, while a failed fit never gets there and leaves the
        column NaN.
        """
        result = _run_placebo_in_space(failing=("b",))

        failed = result.table.loc[
            result.table["error"].notna(), "placebo_treated"
        ].tolist()
        assert failed == ["b"]
        assert result.table["mspe_ratio"].isna().sum() == 1

        with pytest.warns(UserWarning, match="failed placebo fit"):
            fig = PlaceboInSpace.plot_mspe_ratio(result)

        assert "b" not in _tick_labels(fig)
        plt.close(fig)

    def test_plot_returns_a_figure(self):
        """The plot draws one bar per unit, treated included."""
        fig = PlaceboInSpace.plot_mspe_ratio(_make_mspe_check_result())

        assert isinstance(fig, Figure)
        assert _tick_labels(fig) == [
            "a",
            "b",
            "c",
            "actual",
        ]
        plt.close(fig)

    def test_plot_controls_are_keyword_only(self):
        """Only the result is positional, so the controls stay reorderable."""
        with pytest.raises(TypeError):
            PlaceboInSpace.plot_mspe_ratio(_make_mspe_check_result(), "California")

    def test_plot_honours_title_and_figsize(self):
        """The caller's title and size survive plotnine's layout pass."""
        fig = PlaceboInSpace.plot_mspe_ratio(
            _make_mspe_check_result(), title="California", figsize=(5.0, 4.0)
        )

        assert tuple(fig.get_size_inches()) == (5.0, 4.0)
        assert "California" in _figure_texts(fig)
        plt.close(fig)

    def test_plot_highlights_the_treated_unit(self):
        """The treated bar is drawn in the highlight colour, donors are not."""
        fig = PlaceboInSpace.plot_mspe_ratio(_make_mspe_check_result())

        colours = _bar_colours(fig)
        # Bars follow the ratio ordering, so the treated unit (5.0) is last.
        assert colours[-1] == _TREATED_COLOUR.lower()
        assert set(colours[:-1]) == {_PLACEBO_COLOUR.lower()}
        plt.close(fig)

    def test_plot_orders_units_by_ratio(self):
        """A treated unit in the middle of the pack is drawn there."""
        fig = PlaceboInSpace.plot_mspe_ratio(
            _make_mspe_check_result(ratios=(1.0, 4.0, 6.0))
        )

        assert _tick_labels(fig) == [
            "a",
            "b",
            "actual",
            "c",
        ]
        plt.close(fig)

    def test_plot_reports_the_permutation_pvalue(self):
        """The treated unit ranking first over 3 donors gives p = 1/4."""
        fig = PlaceboInSpace.plot_mspe_ratio(_make_mspe_check_result())

        assert any("actual: p = 0.250" in text for text in _figure_texts(fig))
        plt.close(fig)

    def test_plot_ranks_each_treated_unit_against_itself_only(self):
        """One treated unit's p-value does not depend on the others.

        With placebo ratios 1, 2 and 3, each of the two treated units beats
        every donor, so both must report 1/4.  Ranking them within the whole
        plotted frame instead would give 0.4 and 0.2.
        """
        result = _make_mspe_check_result(baseline_ratios={"first": 5.0, "second": 10.0})

        fig = PlaceboInSpace.plot_mspe_ratio(result)

        texts = _figure_texts(fig)
        assert any("first: p = 0.250" in text for text in texts)
        assert any("second: p = 0.250" in text for text in texts)
        plt.close(fig)

    def test_plot_skips_a_treated_unit_with_an_undefined_ratio(self):
        """An unrankable treated unit gets no annotation, the other still does."""
        result = _make_mspe_check_result(
            baseline_ratios={"good": 5.0, "broken": np.nan}
        )

        with pytest.warns(UserWarning, match="undefined"):
            fig = PlaceboInSpace.plot_mspe_ratio(result)

        texts = _figure_texts(fig)
        assert any("good: p = 0.250" in text for text in texts)
        assert not any("broken: p" in text for text in texts)
        plt.close(fig)

    def test_plot_can_suppress_the_pvalue(self):
        """``show_pvalue=False`` drops the annotation."""
        fig = PlaceboInSpace.plot_mspe_ratio(
            _make_mspe_check_result(), show_pvalue=False
        )

        assert not any("p = " in text for text in _figure_texts(fig))
        plt.close(fig)

    def test_plot_omits_the_pvalue_without_a_baseline(self):
        """With no treated baseline the donors still plot, and the missing p-value warns."""
        with pytest.warns(UserWarning, match="No p-value to show"):
            fig = PlaceboInSpace.plot_mspe_ratio(
                _make_mspe_check_result(baseline_ratios={})
            )

        assert _tick_labels(fig) == ["a", "b", "c"]
        assert not any("p = " in text for text in _figure_texts(fig))
        plt.close(fig)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = PlaceboInSpace.plot_mspe_ratio(
                _make_mspe_check_result(baseline_ratios={}), show_pvalue=False
            )
        plt.close(fig)

    def test_plot_keeps_infinite_ratios_in_the_pvalue(self):
        """An undrawable donor still outranks the treated unit.

        Donor ``b`` has a defined infinite ratio, so the reference set is
        1, inf, 3 plus the treated 5: two of the four are at least as large,
        giving p = 0.5.  Dropping it from the denominator would report 1/3.
        """
        result = _make_mspe_check_result(ratios=(1.0, np.inf, 3.0))

        with pytest.warns(UserWarning, match="infinite") as record:
            fig = PlaceboInSpace.plot_mspe_ratio(result)

        # The warning points at this call, not at the helper that raised it.
        assert record[0].filename == __file__
        assert _tick_labels(fig) == ["a", "c", "actual"]
        assert any("actual: p = 0.500" in text for text in _figure_texts(fig))
        plt.close(fig)

    def test_plot_excludes_undefined_ratios_from_the_pvalue(self):
        """An unranked donor leaves the denominator as well as the figure.

        Donor ``b`` has an undefined ratio, so the reference set is 1 and 3
        plus the treated 5, giving p = 1/3.
        """
        result = _make_mspe_check_result(ratios=(1.0, np.nan, 3.0))

        with pytest.warns(UserWarning, match="undefined"):
            fig = PlaceboInSpace.plot_mspe_ratio(result)

        assert _tick_labels(fig) == [
            "a",
            "c",
            "actual",
        ]
        assert any("actual: p = 0.333" in text for text in _figure_texts(fig))
        plt.close(fig)

    @pytest.mark.parametrize(
        "table",
        [
            pytest.param(
                pd.DataFrame({"placebo_treated": ["a"], "mean": [1.0]}),
                id="predates-mspe-reporting",
            ),
            pytest.param(None, id="no-table"),
        ],
    )
    def test_plot_raises_without_mspe_columns(self, table):
        """A result with no `mspe_ratio` column has nothing to plot."""
        result = CheckResult(check_name="PlaceboInSpace", table=table)

        with pytest.raises(ValueError, match="mspe_ratio"):
            PlaceboInSpace.plot_mspe_ratio(result)

    def test_plot_raises_when_no_ratio_can_be_drawn(self):
        """Dropping every unit leaves an empty figure, so raise instead."""
        result = _make_mspe_check_result(
            ratios=(np.inf, np.inf), baseline_ratios={}, units=("a", "b")
        )

        with (
            pytest.warns(UserWarning, match="infinite"),
            pytest.raises(ValueError, match="finite MSPE ratio"),
        ):
            PlaceboInSpace.plot_mspe_ratio(result)

    def test_permutation_pvalue_counts_ties(self):
        """Units tied with the treated unit count against it."""
        ratios = np.array([1.0, 2.0, 2.0, 4.0])

        assert _permutation_pvalue(ratios, 2.0) == pytest.approx(0.75)
        assert _permutation_pvalue(ratios, 4.0) == pytest.approx(0.25)


def test_placebo_in_space_attaches_figure_to_report():
    """Automatic figures reach reports, while opting out preserves statistics."""
    result = _run_placebo_in_space()
    context = PipelineContext(data=pd.DataFrame(), sensitivity_results=[result])

    report = cp.GenerateReport().run(context).report

    assert len(result.figures) == 1
    assert "PlaceboInSpace figure" in report
    assert "data:image/png;base64," in report
    plt.close(result.figures[0])

    disabled = _run_placebo_in_space(check=PlaceboInSpace(make_figures=False))
    assert disabled.figures == []
    pd.testing.assert_frame_equal(disabled.table, result.table)


@pytest.mark.parametrize("ratio", [np.nan, np.inf], ids=["undefined", "infinite"])
def test_placebo_in_space_skips_figure_without_finite_ratios(ratio):
    """Non-finite ratios remain in the result with an explanation for the missing figure."""
    stats = {
        "pre_mspe": 0.0,
        "post_mspe": 1.0 if np.isinf(ratio) else 0.0,
        "mspe_ratio": ratio,
    }
    with patch("causalpy.checks.placebo_in_space._mspe_stats", return_value=stats):
        result = _run_placebo_in_space()

    assert result.figures == []
    assert result.passed is None
    assert "no unit has a finite MSPE ratio" in result.text
    np.testing.assert_equal(result.table["mspe_ratio"].to_numpy(), [ratio] * 3)
    np.testing.assert_equal(
        result.metadata["baseline_mspe"]["actual"]["mspe_ratio"], ratio
    )
