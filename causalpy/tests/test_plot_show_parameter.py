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
Tests for the plot() method's show parameter and legend_kwargs functionality.

This module tests that the plot() method correctly handles the show parameter
to control automatic plot display in Jupyter notebooks, and that legend_kwargs
preserves existing legend content while allowing customization.
"""

from unittest.mock import patch

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

import causalpy as cp

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


@pytest.mark.integration
def test_plot_show_parameter_default_true_pymc(mock_pymc_sample, did_data):
    """
    Test that plot() calls plt.show() by default (show=True) for PyMC models.

    This ensures plots auto-display in Jupyter notebooks when using
    the pattern: fig, ax = result.plot()
    """
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with patch("matplotlib.pyplot.show") as mock_show:
        fig, ax = result.plot()
        # Verify plt.show() was called (default behavior)
        mock_show.assert_called_once()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_plot_show_parameter_explicit_true_pymc(mock_pymc_sample, did_data):
    """
    Test that plot(show=True) calls plt.show() for PyMC models.
    """
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with patch("matplotlib.pyplot.show") as mock_show:
        fig, ax = result.plot(show=True)
        # Verify plt.show() was called
        mock_show.assert_called_once()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_plot_show_parameter_false_pymc(mock_pymc_sample, did_data):
    """
    Test that plot(show=False) does NOT call plt.show() for PyMC models.

    This allows users to modify the figure before displaying it manually.
    """
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with patch("matplotlib.pyplot.show") as mock_show:
        fig, ax = result.plot(show=False)
        # Verify plt.show() was NOT called
        mock_show.assert_not_called()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_plot_show_parameter_default_true_skl(did_data):
    """
    Test that plot() calls plt.show() by default (show=True) for scikit-learn models.
    """
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )

    with patch("matplotlib.pyplot.show") as mock_show:
        fig, ax = result.plot()
        # Verify plt.show() was called (default behavior)
        mock_show.assert_called_once()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_plot_show_parameter_false_skl(did_data):
    """
    Test that plot(show=False) does NOT call plt.show() for scikit-learn models.
    """
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )

    with patch("matplotlib.pyplot.show") as mock_show:
        fig, ax = result.plot(show=False)
        # Verify plt.show() was NOT called
        mock_show.assert_not_called()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)


# ---------------------------------------------------------------------------
# legend_kwargs tests
# ---------------------------------------------------------------------------


def _legend_handle_count(legend):
    """Return the number of handles in a legend (cross-version)."""
    handles = getattr(legend, "legend_handles", getattr(legend, "legendHandles", None))
    return len(handles) if handles is not None else 0


@pytest.mark.integration
def test_legend_kwargs_preserves_labels_pymc(mock_pymc_sample, did_data):
    """
    Test that legend_kwargs preserves existing legend labels for PyMC models.

    DiD Bayesian plots create custom legends with (line, patch) tuple handles.
    Passing legend_kwargs must not drop those entries.
    """
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Plot without legend_kwargs to get baseline
    with patch("matplotlib.pyplot.show"):
        _, ax_baseline = result.plot()
    baseline_legend = ax_baseline.get_legend()
    baseline_labels = [t.get_text() for t in baseline_legend.get_texts()]
    baseline_handle_count = _legend_handle_count(baseline_legend)
    plt.close("all")

    # Plot with legend_kwargs — in-place mutation must preserve everything
    with patch("matplotlib.pyplot.show"):
        _, ax_custom = result.plot(legend_kwargs={"loc": "lower left"})
    custom_legend = ax_custom.get_legend()
    custom_labels = [t.get_text() for t in custom_legend.get_texts()]

    assert custom_legend is not None, "Legend must still exist after legend_kwargs"
    assert custom_labels == baseline_labels, "Legend labels must be preserved"
    assert _legend_handle_count(custom_legend) == baseline_handle_count, (
        "Legend handle count must be preserved"
    )
    plt.close("all")


@pytest.mark.integration
def test_legend_kwargs_preserves_labels_skl(did_data):
    """
    Test that legend_kwargs preserves existing legend labels for OLS models.
    """
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )

    # Plot without legend_kwargs to get baseline
    with patch("matplotlib.pyplot.show"):
        _, ax_baseline = result.plot()
    baseline_legend = ax_baseline.get_legend()
    baseline_labels = [t.get_text() for t in baseline_legend.get_texts()]
    baseline_handle_count = _legend_handle_count(baseline_legend)
    plt.close("all")

    # Plot with legend_kwargs
    with patch("matplotlib.pyplot.show"):
        _, ax_custom = result.plot(legend_kwargs={"loc": "lower left"})
    custom_legend = ax_custom.get_legend()
    custom_labels = [t.get_text() for t in custom_legend.get_texts()]

    assert custom_legend is not None, "Legend must still exist after legend_kwargs"
    assert custom_labels == baseline_labels, "Legend labels must be preserved"
    assert _legend_handle_count(custom_legend) == baseline_handle_count, (
        "Legend handle count must be preserved"
    )
    plt.close("all")


@pytest.mark.integration
def test_legend_kwargs_changes_location_pymc(mock_pymc_sample, did_data):
    """
    Test that legend_kwargs with loc does not crash and preserves the legend.
    """
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Get baseline labels
    with patch("matplotlib.pyplot.show"):
        _, ax_baseline = result.plot()
    baseline_labels = [t.get_text() for t in ax_baseline.get_legend().get_texts()]
    plt.close("all")

    # Apply loc change — must preserve labels and not crash
    with patch("matplotlib.pyplot.show"):
        fig, ax = result.plot(legend_kwargs={"loc": "lower right"})
    assert isinstance(fig, plt.Figure)
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == baseline_labels
    plt.close("all")


@pytest.mark.integration
def test_legend_kwargs_bbox_to_anchor_triggers_layout(mock_pymc_sample, did_data):
    """
    Test that bbox_to_anchor triggers fig.tight_layout() to avoid clipping.
    """
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with (
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.figure.Figure.tight_layout") as mock_tl,
    ):
        fig, ax = result.plot(
            legend_kwargs={"loc": "upper left", "bbox_to_anchor": (1.04, 1)},
        )
    assert isinstance(fig, plt.Figure)
    assert ax.get_legend() is not None
    mock_tl.assert_called_once()
    plt.close("all")


@pytest.mark.integration
def test_legend_kwargs_frameon_and_title(mock_pymc_sample, did_data):
    """
    Test that frameon and title kwargs are applied in place.
    """
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with patch("matplotlib.pyplot.show"):
        _, ax = result.plot(
            legend_kwargs={"frameon": False, "title": "My Legend"},
        )
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_frame_on() is False
    assert legend.get_title().get_text() == "My Legend"
    plt.close("all")


@pytest.mark.integration
def test_legend_kwargs_unsupported_key_raises(mock_pymc_sample, did_data):
    """
    Test that unsupported legend_kwargs keys raise TypeError.
    """
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with (
        pytest.raises(TypeError, match="not supported"),
        patch("matplotlib.pyplot.show"),
    ):
        result.plot(legend_kwargs={"ncol": 2})
    plt.close("all")


@pytest.mark.integration
def test_legend_kwargs_bbox_transform_without_anchor_raises(mock_pymc_sample, did_data):
    """
    Test that bbox_transform without bbox_to_anchor raises TypeError.
    """
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with (
        pytest.raises(TypeError, match="bbox_transform requires bbox_to_anchor"),
        patch("matplotlib.pyplot.show"),
    ):
        result.plot(legend_kwargs={"bbox_transform": None})
    plt.close("all")


@pytest.mark.integration
def test_legend_kwargs_preserves_fontsize(mock_pymc_sample, did_data):
    """
    Test that existing legend fontsize is preserved when not overridden,
    and that an explicit fontsize override takes effect.
    """
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Get baseline fontsize
    with patch("matplotlib.pyplot.show"):
        _, ax_baseline = result.plot()
    baseline_fontsize = ax_baseline.get_legend().get_texts()[0].get_fontsize()
    plt.close("all")

    # legend_kwargs without fontsize — fontsize must be unchanged
    with patch("matplotlib.pyplot.show"):
        _, ax_custom = result.plot(legend_kwargs={"loc": "lower right"})
    assert ax_custom.get_legend().get_texts()[0].get_fontsize() == baseline_fontsize
    plt.close("all")

    # legend_kwargs with explicit fontsize override
    with patch("matplotlib.pyplot.show"):
        _, ax_override = result.plot(legend_kwargs={"fontsize": 8})
    assert ax_override.get_legend().get_texts()[0].get_fontsize() == 8
    plt.close("all")


@pytest.mark.integration
def test_legend_kwargs_none_is_noop(mock_pymc_sample, did_data):
    """
    Test that legend_kwargs=None (default) does not alter the plot.
    """
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with patch("matplotlib.pyplot.show"):
        fig, ax = result.plot(legend_kwargs=None)
    assert isinstance(fig, plt.Figure)
    assert ax.get_legend() is not None
    plt.close("all")


@pytest.mark.integration
def test_legend_kwargs_multi_axis_its(mock_pymc_sample, its_data):
    """
    Test legend_kwargs on a multi-axis experiment (ITS has 3 axes).

    This exercises the numpy-array / list-of-axes flattening logic and
    verifies that legends on the first axes are preserved.
    """
    result = cp.InterruptedTimeSeries(
        its_data,
        formula="y ~ 1 + t",
        treatment_time=pd.to_datetime("2017-01-01"),
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Get baseline legend from first axes
    with patch("matplotlib.pyplot.show"):
        _, axes_baseline = result.plot()
    baseline_legend = axes_baseline[0].get_legend()
    baseline_labels = [t.get_text() for t in baseline_legend.get_texts()]
    plt.close("all")

    # Apply legend_kwargs — should preserve labels on all legend-bearing axes
    with patch("matplotlib.pyplot.show"):
        fig, axes_custom = result.plot(legend_kwargs={"loc": "lower left"})
    assert isinstance(fig, plt.Figure)
    custom_legend = axes_custom[0].get_legend()
    assert custom_legend is not None
    custom_labels = [t.get_text() for t in custom_legend.get_texts()]
    assert custom_labels == baseline_labels
    plt.close("all")
