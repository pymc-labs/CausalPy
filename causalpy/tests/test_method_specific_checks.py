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

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.checks.bandwidth import BandwidthSensitivity
from causalpy.checks.base import Check, CheckResult
from causalpy.checks.leave_one_out import LeaveOneOut
from causalpy.checks.mccrary import McCraryDensityTest
from causalpy.checks.placebo_in_space import (
    PlaceboInSpace,
    _mspe_stats,
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
    fit=True, treated=("actual",), config_treated=None, failing=()
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
        return PlaceboInSpace().run(sc, ctx)


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


class TestPlaceboInSpaceMspeRatio:
    """Tests for the MSPE columns."""

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
