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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
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
        """A treated unit the experiment never fitted gets no baseline RMSPE."""
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
        assert set(result.metadata["baseline_rmspe"]) == {"actual"}


# ---------------------------------------------------------------------------
# PlaceboInSpace RMSPE-ratio tests
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


def _make_rmspe_check_result(
    ratios=(1.0, 2.0, 3.0),
    baseline_ratio=5.0,
    units=("a", "b", "c"),
):
    """Build a CheckResult with hand-set ratios, skipping the model fits."""
    metadata = {}
    if baseline_ratio is not None:
        metadata["baseline_rmspe"] = {
            "actual": {
                "pre_rmspe": 1.0,
                "post_rmspe": baseline_ratio,
                "rmspe_ratio": baseline_ratio,
            }
        }
    return CheckResult(
        check_name="PlaceboInSpace",
        table=pd.DataFrame(
            {
                "placebo_treated": list(units),
                "pre_rmspe": [1.0] * len(units),
                "post_rmspe": list(ratios),
                "rmspe_ratio": list(ratios),
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


class TestPlaceboInSpaceRmspeRatio:
    """Tests for the RMSPE columns and the ratio plot."""

    def test_run_reports_rmspe_columns(self):
        """Every successful placebo fit carries its pre, post and ratio."""
        result = _run_placebo_in_space()

        for column in ("pre_rmspe", "post_rmspe", "rmspe_ratio"):
            assert column in result.table.columns
        assert np.isfinite(result.table["rmspe_ratio"]).all()
        np.testing.assert_allclose(
            result.table["rmspe_ratio"],
            result.table["post_rmspe"] / result.table["pre_rmspe"],
        )

    def test_run_reports_baseline_rmspe(self):
        """The actual treated unit's RMSPEs land in metadata."""
        result = _run_placebo_in_space()

        baseline = result.metadata["baseline_rmspe"]["actual"]
        assert set(baseline) == {"pre_rmspe", "post_rmspe", "rmspe_ratio"}
        assert baseline["rmspe_ratio"] == pytest.approx(
            baseline["post_rmspe"] / baseline["pre_rmspe"]
        )

    def test_plot_returns_a_figure(self):
        """The plot draws one bar per unit, treated included."""
        fig = PlaceboInSpace.plot_rmspe_ratio(_make_rmspe_check_result())

        assert isinstance(fig, Figure)
        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == [
            "a",
            "b",
            "c",
            "actual",
        ]
        plt.close(fig)

    def test_plot_honours_title_and_figsize(self):
        """The caller's title and size survive plotnine's layout pass."""
        fig = PlaceboInSpace.plot_rmspe_ratio(
            _make_rmspe_check_result(), title="California", figsize=(5.0, 4.0)
        )

        assert tuple(fig.get_size_inches()) == (5.0, 4.0)
        assert "California" in _figure_texts(fig)
        plt.close(fig)

    def test_plot_highlights_the_treated_unit(self):
        """The treated bar is drawn in the highlight colour, donors are not."""
        fig = PlaceboInSpace.plot_rmspe_ratio(_make_rmspe_check_result())

        colours = _bar_colours(fig)
        # Bars follow the ratio ordering, so the treated unit (5.0) is last.
        assert colours[-1] == _TREATED_COLOUR.lower()
        assert set(colours[:-1]) == {_PLACEBO_COLOUR.lower()}
        plt.close(fig)

    def test_plot_orders_units_by_ratio(self):
        """A treated unit in the middle of the pack is drawn there."""
        fig = PlaceboInSpace.plot_rmspe_ratio(
            _make_rmspe_check_result(ratios=(1.0, 4.0, 6.0), baseline_ratio=5.0)
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
        fig = PlaceboInSpace.plot_rmspe_ratio(_make_rmspe_check_result())

        assert any("actual: p = 0.250" in text for text in _figure_texts(fig))
        plt.close(fig)

    def test_plot_can_suppress_the_pvalue(self):
        """``show_pvalue=False`` drops the annotation."""
        fig = PlaceboInSpace.plot_rmspe_ratio(
            _make_rmspe_check_result(), show_pvalue=False
        )

        assert not any("p = " in text for text in _figure_texts(fig))
        plt.close(fig)

    def test_plot_omits_the_pvalue_without_a_baseline(self):
        """With no treated baseline the donors still plot, without inference."""
        fig = PlaceboInSpace.plot_rmspe_ratio(
            _make_rmspe_check_result(baseline_ratio=None)
        )

        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["a", "b", "c"]
        assert not any("p = " in text for text in _figure_texts(fig))
        plt.close(fig)

    def test_plot_warns_and_drops_non_finite_ratios(self):
        """A donor with a zero pre-period RMSPE is dropped, loudly."""
        result = _make_rmspe_check_result(ratios=(1.0, np.inf, 3.0))

        with pytest.warns(UserWarning, match="non-finite"):
            fig = PlaceboInSpace.plot_rmspe_ratio(result)

        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == [
            "a",
            "c",
            "actual",
        ]
        plt.close(fig)

    def test_plot_raises_without_rmspe_columns(self):
        """A result predating RMSPE reporting cannot be plotted."""
        result = CheckResult(
            check_name="PlaceboInSpace",
            table=pd.DataFrame({"placebo_treated": ["a"], "mean": [1.0]}),
        )

        with pytest.raises(ValueError, match="rmspe_ratio"):
            PlaceboInSpace.plot_rmspe_ratio(result)

    def test_plot_raises_when_the_run_produced_no_table(self):
        """A check that bailed out early has nothing to plot."""
        with pytest.raises(ValueError, match="rmspe_ratio"):
            PlaceboInSpace.plot_rmspe_ratio(CheckResult(check_name="PlaceboInSpace"))

    def test_plot_raises_when_every_ratio_is_non_finite(self):
        """Dropping every unit leaves an empty figure, so raise instead."""
        result = _make_rmspe_check_result(
            ratios=(np.inf, np.inf), baseline_ratio=None, units=("a", "b")
        )

        with (
            pytest.warns(UserWarning, match="non-finite"),
            pytest.raises(ValueError, match="finite RMSPE ratio"),
        ):
            PlaceboInSpace.plot_rmspe_ratio(result)

    def test_permutation_pvalue_counts_ties(self):
        """Units tied with the treated unit count against it."""
        ratios = np.array([1.0, 2.0, 2.0, 4.0])

        assert _permutation_pvalue(ratios, 2.0) == pytest.approx(0.75)
        assert _permutation_pvalue(ratios, 4.0) == pytest.approx(0.25)
