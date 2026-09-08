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

from types import SimpleNamespace
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
from causalpy.checks.mccrary import McCraryDensityTest
from causalpy.checks.placebo_in_space import (
    _PLACEBO_COLOUR,
    _TREATED_COLOUR,
    PlaceboInSpace,
    _mspe_stats,
    _permutation_pvalue,
)
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

    def test_run_skips_baseline_for_unfitted_treated_unit(self):
        """A treated unit the experiment never fitted gets no baseline MSPE."""
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
        # "a" is named treated in the config but absent from the fitted
        # experiment, so selecting it would raise rather than yield a baseline.
        assert set(result.metadata["baseline_mspe"]) == {"actual"}


# ---------------------------------------------------------------------------
# PlaceboInSpace MSPE-ratio tests
# ---------------------------------------------------------------------------


def _run_placebo_in_space():
    """Run the check on the bundled synthetic-control data."""
    df = cp.load_data("sc")
    controls = ["a", "b", "c"]
    sc = SyntheticControl(
        df,
        treatment_time=70,
        control_units=controls,
        treated_units=["actual"],
        model=cp.create_causalpy_compatible_class(LinearRegression()),
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
    return PlaceboInSpace().run(sc, ctx)


def _fake_experiment(pre_residuals, post_residuals, unit="actual"):
    """An object exposing just the impact arrays ``_mspe_stats`` reads.

    The residuals are constant across the posterior, so the posterior mean is
    the residual itself and the expected MSPE can be written down by hand.
    """

    def impact(values):
        return xr.DataArray(
            np.tile(values, (1, 2, 1, 1)).transpose(0, 1, 3, 2),
            dims=("chain", "draw", "obs_ind", "treated_units"),
            coords={"treated_units": [unit]},
        )

    return SimpleNamespace(
        pre_impact=impact([pre_residuals]),
        post_impact=impact([post_residuals]),
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
    metadata = {}
    if baseline_ratios:
        metadata["baseline_mspe"] = {
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
        metadata=metadata,
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


class TestPlaceboInSpaceMspeRatio:
    """Tests for the MSPE columns and the ratio plot."""

    def test_mspe_stats_matches_hand_computed_errors(self):
        """The reported quantity is the MSPE ratio, not its square root.

        Pre-period residuals of 3 and 4 give an MSPE of 12.5, post-period
        residuals of 6 and 8 give 50, so the ratio is 4.  The square-root
        (RMSPE) version of the same ratio would be 2, which is what makes
        this test able to detect the wrong statistic.
        """
        experiment = _fake_experiment([3.0, 4.0], [6.0, 8.0])

        stats = _mspe_stats(experiment, "actual")

        assert stats["pre_mspe"] == pytest.approx(12.5)
        assert stats["post_mspe"] == pytest.approx(50.0)
        assert stats["mspe_ratio"] == pytest.approx(4.0)

    def test_mspe_stats_separates_the_degenerate_cases(self):
        """A zero pre-period error is infinite or undefined, never both."""
        positive_over_zero = _mspe_stats(
            _fake_experiment([0.0, 0.0], [1.0, 1.0]), "actual"
        )
        zero_over_zero = _mspe_stats(_fake_experiment([0.0, 0.0], [0.0, 0.0]), "actual")
        non_finite_pre = _mspe_stats(
            _fake_experiment([np.nan, 1.0], [1.0, 1.0]), "actual"
        )

        assert positive_over_zero["mspe_ratio"] == float("inf")
        assert np.isnan(zero_over_zero["mspe_ratio"])
        assert np.isnan(non_finite_pre["mspe_ratio"])

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

    def test_run_reports_baseline_mspe(self):
        """The actual treated unit's MSPEs land in metadata."""
        result = _run_placebo_in_space()

        baseline = result.metadata["baseline_mspe"]["actual"]
        assert set(baseline) == {"pre_mspe", "post_mspe", "mspe_ratio"}
        assert baseline["mspe_ratio"] == pytest.approx(
            baseline["post_mspe"] / baseline["pre_mspe"]
        )

    def test_plot_accepts_the_result_of_a_real_run(self):
        """run() and the plot agree on the contract, without a hand-built result.

        Every other plot test builds its ``CheckResult`` directly, so this is
        the only one that would catch the two drifting apart.
        """
        result = _run_placebo_in_space()

        fig = PlaceboInSpace.plot_mspe_ratio(result)

        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
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
        real_init = SyntheticControl.__init__
        attempts = []

        def fail_once(self, *args, **kwargs):
            # The first call builds the experiment itself; the placebo re-fits
            # follow, so failing on the second breaks exactly one donor.
            attempts.append(None)
            if len(attempts) == 2:
                raise RuntimeError("simulated failure")
            return real_init(self, *args, **kwargs)

        with patch.object(SyntheticControl, "__init__", fail_once):
            result = _run_placebo_in_space()

        assert "error" in result.table.columns
        failed = result.table.loc[
            result.table["error"].notna(), "placebo_treated"
        ].tolist()
        assert len(failed) == 1
        assert result.table["mspe_ratio"].isna().sum() == 1

        with pytest.warns(UserWarning, match="failed placebo fit"):
            fig = PlaceboInSpace.plot_mspe_ratio(result)

        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert failed[0] not in labels
        plt.close(fig)

    def test_plot_returns_a_figure(self):
        """The plot draws one bar per unit, treated included."""
        fig = PlaceboInSpace.plot_mspe_ratio(_make_mspe_check_result())

        assert isinstance(fig, Figure)
        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == [
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

        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == [
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

    def test_plot_can_suppress_the_pvalue(self):
        """``show_pvalue=False`` drops the annotation."""
        fig = PlaceboInSpace.plot_mspe_ratio(
            _make_mspe_check_result(), show_pvalue=False
        )

        assert not any("p = " in text for text in _figure_texts(fig))
        plt.close(fig)

    def test_plot_omits_the_pvalue_without_a_baseline(self):
        """With no treated baseline the donors still plot, without inference."""
        fig = PlaceboInSpace.plot_mspe_ratio(
            _make_mspe_check_result(baseline_ratios={})
        )

        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["a", "b", "c"]
        assert not any("p = " in text for text in _figure_texts(fig))
        plt.close(fig)

    def test_plot_keeps_infinite_ratios_in_the_pvalue(self):
        """An undrawable donor still outranks the treated unit.

        Donor ``b`` has a defined infinite ratio, so the reference set is
        1, inf, 3 plus the treated 5: two of the four are at least as large,
        giving p = 0.5.  Dropping it from the denominator would report 1/3.
        """
        result = _make_mspe_check_result(ratios=(1.0, np.inf, 3.0))

        with pytest.warns(UserWarning, match="infinite"):
            fig = PlaceboInSpace.plot_mspe_ratio(result)

        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == [
            "a",
            "c",
            "actual",
        ]
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

        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == [
            "a",
            "c",
            "actual",
        ]
        assert any("actual: p = 0.333" in text for text in _figure_texts(fig))
        plt.close(fig)

    def test_plot_raises_without_mspe_columns(self):
        """A result predating MSPE reporting cannot be plotted."""
        result = CheckResult(
            check_name="PlaceboInSpace",
            table=pd.DataFrame({"placebo_treated": ["a"], "mean": [1.0]}),
        )

        with pytest.raises(ValueError, match="mspe_ratio"):
            PlaceboInSpace.plot_mspe_ratio(result)

    def test_plot_raises_when_the_run_produced_no_table(self):
        """A check that bailed out early has nothing to plot."""
        with pytest.raises(ValueError, match="mspe_ratio"):
            PlaceboInSpace.plot_mspe_ratio(CheckResult(check_name="PlaceboInSpace"))

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
