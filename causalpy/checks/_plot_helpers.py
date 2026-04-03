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
Shared plotting helpers for sensitivity checks.

Private module — these functions are used by individual check classes
to produce figures for ``CheckResult.figures``.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def forest_plot(
    table: pd.DataFrame,
    label_col: str,
    *,
    baseline_row: dict[str, Any] | None = None,
    baseline_label: str = "baseline",
    xlabel: str = "Average causal impact",
    title: str = "",
    figsize: tuple[float, float] | None = None,
    baseline_color: str = "C0",
    comparison_color: str = "C1",
    highlight_color: str = "C3",
    highlight_label: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw a forest plot of HDI intervals from a check result table.

    Parameters
    ----------
    table : pd.DataFrame
        Must contain ``label_col``, ``mean``, ``hdi_lower``, ``hdi_upper``.
        Rows with an ``error`` column that is non-null are skipped.
    label_col : str
        Column name to use for y-axis labels.
    baseline_row : dict, optional
        If provided, prepend a baseline row (e.g. from the original
        experiment's effect summary) drawn in ``baseline_color``.
    baseline_label : str
        Label for the baseline row.
    xlabel : str
        X-axis label.
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size. Auto-computed from the number of rows if not given.
    baseline_color : str
        Color for the baseline row.
    comparison_color : str
        Color for comparison rows.
    highlight_color : str
        Color for the highlighted row (see ``highlight_label``).
    highlight_label : str or None
        If given, the row whose ``label_col`` matches this value is
        drawn in ``highlight_color`` instead of ``comparison_color``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
    """
    valid = table.dropna(subset=["mean", "hdi_lower", "hdi_upper"])
    if "error" in valid.columns:
        valid = valid[valid["error"].isna() | (valid["error"] == "")]

    labels: list[str] = []
    means: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    colors: list[str] = []
    widths: list[float] = []
    marker_sizes: list[int] = []

    if baseline_row is not None:
        labels.append(baseline_label)
        means.append(float(baseline_row["mean"]))
        lo.append(float(baseline_row["hdi_lower"]))
        hi.append(float(baseline_row["hdi_upper"]))
        colors.append(baseline_color)
        widths.append(3.0)
        marker_sizes.append(8)

    for _, row in valid.iterrows():
        lbl = str(row[label_col])
        labels.append(lbl)
        means.append(float(row["mean"]))
        lo.append(float(row["hdi_lower"]))
        hi.append(float(row["hdi_upper"]))
        if highlight_label is not None and lbl == highlight_label:
            colors.append(highlight_color)
            widths.append(3.0)
            marker_sizes.append(8)
        else:
            colors.append(comparison_color)
            widths.append(2.5)
            marker_sizes.append(7)

    n = len(labels)
    if figsize is None:
        figsize = (8, max(3, 0.3 * n + 1.0))

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = list(range(n))

    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    for i, (m, lower, h, c, w, ms) in enumerate(
        zip(means, lo, hi, colors, widths, marker_sizes, strict=True)
    ):
        ax.plot([lower, h], [i, i], color=c, lw=w, solid_capstyle="round")
        ax.plot(m, i, "o", color=c, markersize=ms, zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig, ax


def null_distribution_plot(
    null_samples: np.ndarray,
    actual_effect: float,
    *,
    p_outside: float | None = None,
    xlabel: str = "Cumulative causal impact",
    title: str | None = None,
    figsize: tuple[float, float] = (7, 3.5),
) -> tuple[plt.Figure, plt.Axes]:
    """Draw a histogram of null-distribution samples vs the actual effect.

    Parameters
    ----------
    null_samples : array-like
        Posterior predictive draws from the status-quo model.
    actual_effect : float
        The observed cumulative effect to compare against.
    p_outside : float, optional
        P(actual outside null). Shown in the title when provided.
    xlabel : str
        X-axis label.
    title : str, optional
        Plot title. Auto-generated if not provided.
    figsize : tuple
        Figure size.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        null_samples,
        bins=40,
        density=True,
        color="C0",
        alpha=0.5,
        edgecolor="C0",
        label="Null distribution (status quo)",
    )
    ax.axvline(
        actual_effect,
        color="C3",
        lw=2.5,
        ls="-",
        label=f"Actual effect = {actual_effect:.1f}",
    )
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")

    if title is None:
        if p_outside is not None:
            title = (
                f"Placebo-in-time: actual effect vs. null distribution "
                f"(P outside null = {p_outside:.3f})"
            )
        else:
            title = "Placebo-in-time: actual effect vs. null distribution"

    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax
