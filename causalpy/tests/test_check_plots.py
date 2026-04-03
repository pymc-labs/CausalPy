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
"""Tests for sensitivity check plot functions."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causalpy.checks._plot_helpers import forest_plot, null_distribution_plot
from causalpy.checks.base import CheckResult
from causalpy.checks.leave_one_out import LeaveOneOut
from causalpy.checks.placebo_in_space import PlaceboInSpace
from causalpy.checks.placebo_in_time import PlaceboInTime
from causalpy.checks.prior_sensitivity import PriorSensitivity

matplotlib.use("Agg")


@dataclass
class _MockEffectSummary:
    """Minimal stand-in for EffectSummary used in plot tests."""

    table: pd.DataFrame | None
    text: str = ""


def _baseline_stats() -> _MockEffectSummary:
    """Return a mock EffectSummary with typical baseline values."""
    return _MockEffectSummary(
        table=pd.DataFrame([{"mean": -25.0, "hdi_lower": -35.0, "hdi_upper": -15.0}])
    )


class TestForestPlot:
    """Tests for the shared forest_plot helper."""

    def test_returns_figure_and_axes(self) -> None:
        """forest_plot returns a (Figure, Axes) tuple."""
        table = pd.DataFrame(
            [
                {"label": "A", "mean": 1.0, "hdi_lower": 0.5, "hdi_upper": 1.5},
                {"label": "B", "mean": 2.0, "hdi_lower": 1.0, "hdi_upper": 3.0},
            ]
        )
        fig, ax = forest_plot(table, label_col="label", title="test")
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_with_baseline_row(self) -> None:
        """Baseline row is prepended and labelled correctly."""
        table = pd.DataFrame(
            [{"label": "A", "mean": 1.0, "hdi_lower": 0.5, "hdi_upper": 1.5}]
        )
        baseline = {"mean": 0.0, "hdi_lower": -0.5, "hdi_upper": 0.5}
        fig, ax = forest_plot(
            table,
            label_col="label",
            baseline_row=baseline,
            baseline_label="base",
        )
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "base" in ytick_labels
        assert "A" in ytick_labels
        plt.close(fig)

    def test_with_highlight(self) -> None:
        """Highlighted row renders without error."""
        table = pd.DataFrame(
            [
                {"label": "A", "mean": 1.0, "hdi_lower": 0.5, "hdi_upper": 1.5},
                {"label": "B", "mean": 2.0, "hdi_lower": 1.0, "hdi_upper": 3.0},
            ]
        )
        fig, ax = forest_plot(
            table, label_col="label", highlight_label="B", highlight_color="C3"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_skips_error_rows(self) -> None:
        """Rows with error values are excluded from the plot."""
        table = pd.DataFrame(
            [
                {
                    "label": "A",
                    "mean": 1.0,
                    "hdi_lower": 0.5,
                    "hdi_upper": 1.5,
                    "error": np.nan,
                },
                {
                    "label": "B",
                    "mean": np.nan,
                    "hdi_lower": np.nan,
                    "hdi_upper": np.nan,
                    "error": "failed",
                },
            ]
        )
        fig, ax = forest_plot(table, label_col="label")
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "A" in ytick_labels
        assert "B" not in ytick_labels
        plt.close(fig)

    def test_auto_figsize(self) -> None:
        """Auto-computed figure height grows with the number of rows."""
        rows = [
            {"label": f"unit_{i}", "mean": float(i), "hdi_lower": 0.0, "hdi_upper": 2.0}
            for i in range(20)
        ]
        table = pd.DataFrame(rows)
        fig, _ = forest_plot(table, label_col="label")
        h = fig.get_size_inches()[1]
        assert h > 5  # auto-sized for many rows
        plt.close(fig)


class TestNullDistributionPlot:
    """Tests for the null_distribution_plot helper."""

    def test_returns_figure_and_axes(self) -> None:
        """null_distribution_plot returns a (Figure, Axes) tuple."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 10, size=200)
        fig, ax = null_distribution_plot(samples, actual_effect=-100.0)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_title_includes_p_outside(self) -> None:
        """p_outside value appears in the auto-generated title."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 10, size=200)
        fig, ax = null_distribution_plot(samples, actual_effect=-100.0, p_outside=0.987)
        assert "0.987" in ax.get_title()
        plt.close(fig)

    def test_custom_title(self) -> None:
        """Custom title overrides the default."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 10, size=200)
        fig, ax = null_distribution_plot(
            samples, actual_effect=-100.0, title="Custom title"
        )
        assert ax.get_title() == "Custom title"
        plt.close(fig)


class TestPlaceboInSpacePlot:
    """Tests for PlaceboInSpace.plot()."""

    def test_plot_with_baseline(self) -> None:
        """Baseline treated unit is appended and highlighted."""
        table = pd.DataFrame(
            [
                {
                    "placebo_treated": "StateA",
                    "mean": -5.0,
                    "hdi_lower": -10.0,
                    "hdi_upper": 0.0,
                },
                {
                    "placebo_treated": "StateB",
                    "mean": -2.0,
                    "hdi_lower": -8.0,
                    "hdi_upper": 4.0,
                },
            ]
        )
        cr = CheckResult(check_name="PlaceboInSpace", table=table)
        fig, ax = PlaceboInSpace.plot(
            cr,
            baseline_stats=_baseline_stats(),
            treated_label="California (treated)",
        )
        assert isinstance(fig, plt.Figure)
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "California (treated)" in ytick_labels
        assert "StateA" in ytick_labels
        plt.close(fig)

    def test_plot_without_baseline(self) -> None:
        """Plot works without baseline stats."""
        table = pd.DataFrame(
            [{"placebo_treated": "X", "mean": 1.0, "hdi_lower": 0.0, "hdi_upper": 2.0}]
        )
        cr = CheckResult(check_name="PlaceboInSpace", table=table)
        fig, ax = PlaceboInSpace.plot(cr)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlaceboInTimePlot:
    """Tests for PlaceboInTime.plot()."""

    def test_plot(self) -> None:
        """PlaceboInTime.plot renders with p_outside in the title."""
        rng = np.random.default_rng(42)
        cr = CheckResult(
            check_name="PlaceboInTime",
            metadata={
                "null_samples": rng.normal(0, 10, size=200),
                "actual_cumulative_mean": -150.0,
                "p_effect_outside_null": 0.995,
            },
        )
        fig, ax = PlaceboInTime.plot(cr)
        assert isinstance(fig, plt.Figure)
        assert "0.995" in ax.get_title()
        plt.close(fig)


class TestLeaveOneOutPlot:
    """Tests for LeaveOneOut.plot()."""

    def test_plot_with_baseline(self) -> None:
        """Baseline 'all donors' row is prepended and drop- labels are shown."""
        table = pd.DataFrame(
            [
                {
                    "dropped_unit": "StateA",
                    "mean": -24.0,
                    "hdi_lower": -34.0,
                    "hdi_upper": -14.0,
                },
                {
                    "dropped_unit": "StateB",
                    "mean": -26.0,
                    "hdi_lower": -36.0,
                    "hdi_upper": -16.0,
                },
            ]
        )
        cr = CheckResult(check_name="LeaveOneOut", table=table)
        fig, ax = LeaveOneOut.plot(cr, baseline_stats=_baseline_stats())
        assert isinstance(fig, plt.Figure)
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "all donors" in ytick_labels
        assert "drop StateA" in ytick_labels
        plt.close(fig)


class TestPriorSensitivityPlot:
    """Tests for PriorSensitivity.plot()."""

    def test_plot_with_baseline(self) -> None:
        """Baseline prior row is prepended alongside alternative prior rows."""
        table = pd.DataFrame(
            [
                {
                    "prior_spec": "diffuse (conc=0.1)",
                    "mean": -24.0,
                    "hdi_lower": -40.0,
                    "hdi_upper": -8.0,
                },
                {
                    "prior_spec": "tight (conc=10)",
                    "mean": -25.5,
                    "hdi_lower": -32.0,
                    "hdi_upper": -19.0,
                },
            ]
        )
        cr = CheckResult(check_name="PriorSensitivity", table=table)
        fig, ax = PriorSensitivity.plot(
            cr,
            baseline_stats=_baseline_stats(),
            baseline_label="baseline (conc=1)",
        )
        assert isinstance(fig, plt.Figure)
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "baseline (conc=1)" in ytick_labels
        assert "diffuse (conc=0.1)" in ytick_labels
        plt.close(fig)
