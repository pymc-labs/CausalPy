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
Plotting utility functions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import polars as pl
import tidydraws as td
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from causalpy.constants import HDI_PROB, LEGEND_FONT_SIZE

HISTOGRAM_PANEL_THEME = p9.theme(
    panel_background=p9.element_rect(fill="white"),
    panel_grid_major=p9.element_blank(),
    panel_grid_minor=p9.element_blank(),
)

_VALID_POSTERIOR_KINDS = frozenset({"ribbon", "histogram", "spaghetti"})
_VALID_CI_KINDS = frozenset({"hdi", "eti"})


@dataclass(frozen=True)
class PlotSpec:
    """Declarative plot plus optional post-draw matplotlib overlay."""

    plot: Any
    overlay: Callable[[plt.Figure, list[plt.Axes]], None] | None = None
    n_panels: int | None = None


def panel_axes(fig: plt.Figure, n: int | None = None) -> list[plt.Axes]:
    """Facet axes from a plotnine ``.draw()`` figure, excluding colorbars.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure returned by :meth:`plotnine.ggplot.draw`.
    n : int, optional
        When set, return at most this many panel axes.
    """
    axes = [a for a in fig.axes if a.get_subplotspec() is not None]
    return axes[:n] if n is not None else axes


def as_axes_result(axes: list[plt.Axes]) -> plt.Axes | np.ndarray:
    """Normalize a panel list to a single Axes or ndarray for public return.

    Parameters
    ----------
    axes : list of matplotlib.axes.Axes
        Panel axes discovered via :func:`panel_axes`.
    """
    if len(axes) == 1:
        return axes[0]
    return np.asarray(axes)


def to_axes_list(ax: plt.Axes | np.ndarray | list[plt.Axes]) -> list[plt.Axes]:
    """Normalize public multi-panel returns that promise a list of axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, numpy.ndarray, or list
        Single axes, ndarray of axes, or list from :func:`as_axes_result`.
    """
    if isinstance(ax, list):
        return ax
    if isinstance(ax, np.ndarray):
        return list(ax.flat)
    return [ax]


def validate_posterior_plot_options(
    kind: str,
    *,
    ci_kind: str = "hdi",
    num_samples: int = 50,
) -> None:
    """Validate shared posterior plot kwargs once at the plotting boundary.

    Parameters
    ----------
    kind : str
        Posterior rendering mode.
    ci_kind : str, optional
        Credible interval type for ribbon summaries.
    num_samples : int, optional
        Number of spaghetti paths to sample.
    """
    if kind not in _VALID_POSTERIOR_KINDS:
        msg = f"Unknown kind: {kind!r}. Must be 'ribbon', 'histogram', or 'spaghetti'."
        raise ValueError(msg)
    if ci_kind not in _VALID_CI_KINDS:
        msg = f"Unknown ci_kind: {ci_kind!r}. Must be 'hdi' or 'eti'."
        raise ValueError(msg)
    if num_samples <= 0:
        msg = f"num_samples must be positive, got {num_samples}."
        raise ValueError(msg)


def _validate_ci_kind(ci_kind: str) -> None:
    if ci_kind not in _VALID_CI_KINDS:
        msg = f"Unknown ci_kind: {ci_kind!r}. Must be 'hdi' or 'eti'."
        raise ValueError(msg)


def _validate_num_samples(num_samples: int) -> None:
    if num_samples <= 0:
        msg = f"num_samples must be positive, got {num_samples}."
        raise ValueError(msg)


def scale_for_x_column(x: pd.Series | np.ndarray) -> Any:
    """Choose a plotnine x scale that matches the column dtype.

    Parameters
    ----------
    x : pandas.Series or numpy.ndarray
        Values used on the shared x aesthetic.
    """
    series = pd.Series(x)
    if pd.api.types.is_bool_dtype(series) or isinstance(
        series.dtype, pd.CategoricalDtype
    ):
        return p9.guides()
    if not pd.api.types.is_numeric_dtype(series):
        return p9.guides()
    breaks = sorted(series.astype(float).unique())
    return p9.scale_x_continuous(breaks=breaks)


def coord_xlim_for_column(
    x: pd.Series | np.ndarray,
    *,
    padding: float = 0.15,
) -> Any:
    """Return ``coord_cartesian`` x limits for numeric x columns only.

    Parameters
    ----------
    x : pandas.Series or numpy.ndarray
        Values used on the shared x aesthetic.
    padding : float, optional
        Extra span added beyond the maximum x value for annotations.
    """
    series = pd.Series(x)
    if pd.api.types.is_bool_dtype(series) or isinstance(
        series.dtype, pd.CategoricalDtype
    ):
        return p9.coord_cartesian()
    if not pd.api.types.is_numeric_dtype(series):
        return p9.coord_cartesian()
    vals = series.astype(float).to_numpy()
    span = float(np.ptp(vals) or 1.0)
    return p9.coord_cartesian(
        xlim=(float(np.min(vals)) - 0.05, float(np.max(vals)) + padding * span)
    )


def sample_draw_lines(
    draws: pl.DataFrame,
    num_samples: int,
    *,
    sort_by: str | list[str],
) -> pl.DataFrame:
    """Subsample posterior draws and tag each with a unique ``_draw_id``.

    Parameters
    ----------
    draws : polars.DataFrame
        Long posterior draws with ``chain`` and ``draw`` columns.
    num_samples : int
        Maximum number of draw lines to keep.
    sort_by : str or list of str
        Column(s) passed to :meth:`polars.DataFrame.sort`.
    """
    tagged = draws.with_columns(
        (pl.col("chain") * 1_000_000 + pl.col("draw")).alias("_draw_id")
    )
    ids = tagged.select("_draw_id").unique()
    chosen = ids.sample(n=min(num_samples, ids.height), seed=42)
    return tagged.join(chosen, on="_draw_id").sort(sort_by)


def _filter_treated_unit(draws: pl.DataFrame, treated_unit: str | None) -> pl.DataFrame:
    if treated_unit is not None and "treated_units" in draws.columns:
        draws = draws.filter(pl.col("treated_units") == treated_unit)
    if "treated_units" in draws.columns and draws["treated_units"].n_unique() == 1:
        return draws.drop("treated_units")
    return draws


def prediction_draws(
    pred: Any,
    newdata: pd.DataFrame,
    *,
    var_name: str = "mu",
    treated_unit: str | None = None,
) -> pl.DataFrame:
    """Extract posterior predictive draws once for summary and spaghetti.

    Parameters
    ----------
    pred : Any
        Prediction container for :func:`tidydraws.prediction_draws`.
    newdata : pandas.DataFrame
        Grid passed as ``newdata`` to tidydraws.
    var_name : str, optional
        Posterior variable name. Defaults to ``"mu"``.
    treated_unit : str, optional
        When draws include ``treated_units``, filter to this unit.
    """
    draws = td.prediction_draws(
        pred, newdata=newdata, var_name=var_name, idata_group="posterior_predictive"
    )
    return _filter_treated_unit(draws, treated_unit)


def dataarray_draws(
    da: xr.DataArray,
    *,
    var_name: str = "mu",
    treated_unit: str | None = None,
) -> pl.DataFrame:
    """Convert a posterior DataArray to the same long form as tidydraws.

    Parameters
    ----------
    da : xarray.DataArray
        Posterior samples with ``chain`` and ``draw`` dimensions.
    var_name : str, optional
        Name for the posterior value column.
    treated_unit : str, optional
        Treated unit to select when that dimension is present.
    """
    if treated_unit is not None:
        da = da.sel(treated_units=treated_unit)
    elif "treated_units" in da.dims:
        da = da.isel(treated_units=0)
    return _filter_treated_unit(
        pl.from_pandas(da.to_dataframe(name=var_name).reset_index()),
        treated_unit,
    )


def label_draws(
    draws: pl.DataFrame,
    *,
    series: str,
    panel: str | None = None,
) -> pl.DataFrame:
    """Attach plotting identity to a canonical posterior-draw table.

    Parameters
    ----------
    draws : polars.DataFrame
        Canonical long posterior draws.
    series : str
        Series label.
    panel : str, optional
        Facet label.
    """
    labels = [pl.lit(series).alias("series")]
    if panel is not None:
        labels.append(pl.lit(panel).alias("panel"))
    return draws.with_columns(labels)


def summarize_draws(
    draws: pl.DataFrame,
    *,
    group_by: str | list[str],
    ci_prob: float,
    interval: Literal["hdi", "eti"] = "hdi",
    var_name: str = "mu",
) -> pd.DataFrame:
    """Summarize one canonical long posterior-draw table.

    Parameters
    ----------
    draws : polars.DataFrame
        Canonical long posterior draws.
    group_by : str or list of str
        Columns identifying one plotted point.
    ci_prob : float
        Credible interval probability mass.
    interval : {"hdi", "eti"}, optional
        Interval type.
    var_name : str, optional
        Posterior value column.
    """
    return (
        td.point_interval(
            draws,
            var_name,
            group_by=group_by,
            probs=(ci_prob,),
            point="mean",
            interval=interval,
        )
        .sort(group_by)
        .to_pandas()
    )


def spaghetti_draws(
    draws: pl.DataFrame,
    *,
    group_by: str | list[str],
    num_samples: int,
    sort_by: str | list[str] | None = None,
) -> pd.DataFrame:
    """Sample complete posterior paths from one canonical draw table.

    Parameters
    ----------
    draws : polars.DataFrame
        Canonical long posterior draws.
    group_by : str or list of str
        Columns that order each path.
    num_samples : int
        Maximum number of complete paths to retain.
    sort_by : str or list of str, optional
        Explicit output sort columns.
    """
    _validate_num_samples(num_samples)
    sampled = sample_draw_lines(draws, num_samples, sort_by=sort_by or group_by)
    identity = [col for col in ("panel", "series") if col in sampled.columns]
    if identity:
        sampled = sampled.with_columns(
            pl.concat_str(
                [pl.col(col) for col in identity] + [pl.col("_draw_id")],
                separator=":",
            ).alias("_line_id")
        )
    else:
        sampled = sampled.with_columns(pl.col("_draw_id").alias("_line_id"))
    return sampled.to_pandas()


def posterior_kind_layers(
    draws: pl.DataFrame,
    kind: Literal["ribbon", "histogram", "spaghetti"],
    *,
    x: str,
    group_by: str | list[str],
    ci_prob: float,
    interval: Literal["hdi", "eti"] = "hdi",
    num_samples: int = 50,
    var_name: str = "mu",
    colors: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[Any]]:
    """Summarize canonical draws and build the requested posterior layers.

    Parameters
    ----------
    draws : polars.DataFrame
        Canonical posterior draws. A ``series`` column enables colored
        multi-series layers; a ``panel`` column enables facets.
    kind : {"ribbon", "histogram", "spaghetti"}
        Posterior rendering mode.
    x : str
        Column used for the horizontal axis.
    group_by : str or list of str
        Columns identifying one summarized posterior point.
    ci_prob : float
        Credible interval probability mass.
    interval : {"hdi", "eti"}, optional
        Credible interval type.
    num_samples : int, optional
        Complete posterior paths retained for spaghetti mode.
    var_name : str, optional
        Posterior value column.
    colors : dict, optional
        Fixed fill colors by series for histogram mode.

    Returns
    -------
    tuple[pandas.DataFrame, list]
        Point/interval summaries and plotnine layers.
    """
    validate_posterior_plot_options(kind, ci_kind=interval, num_samples=num_samples)
    bands = summarize_draws(
        draws,
        group_by=group_by,
        ci_prob=ci_prob,
        interval=interval,
        var_name=var_name,
    )
    has_series = "series" in draws.columns
    line_aes = (
        p9.aes(x, var_name, color="series") if has_series else p9.aes(x, var_name)
    )
    line_bands = (
        bands.groupby("series", observed=True)
        .filter(lambda group: len(group) > 1)
        .reset_index(drop=True)
        if has_series
        else bands
        if len(bands) > 1
        else bands.iloc[0:0]
    )
    mean_line = [p9.geom_line(line_bands, line_aes)] if not line_bands.empty else []

    if kind == "histogram":
        layers: list[Any] = []
        frame = draws.to_pandas()
        series = list(frame["series"].drop_duplicates()) if has_series else [None]
        for label in series:
            subset = frame if label is None else frame.loc[frame["series"] == label]
            fill = "grey" if label is None else (colors or {}).get(label, "grey")
            layers.append(
                p9.geom_bin_2d(
                    subset,
                    p9.aes(x, var_name, alpha=p9.after_stat("count")),
                    bins=(max(int(subset[x].nunique()), 1), 50),
                    drop=True,
                    fill=fill,
                    raster=True,
                    inherit_aes=False,
                    show_legend=False,
                )
            )
        layers.extend(
            [
                p9.scale_alpha_continuous(range=(0.0, 0.85)),
                p9.guides(alpha="none"),
                *mean_line,
            ]
        )
        return bands, layers

    if kind == "spaghetti":
        paths = spaghetti_draws(
            draws,
            group_by=group_by,
            num_samples=num_samples,
        )
        layers = [
            p9.geom_line(
                paths,
                p9.aes(x, var_name, group="_line_id", color="series")
                if has_series
                else p9.aes(x, var_name, group="_line_id"),
                alpha=0.1,
                size=0.3,
                show_legend=False,
            ),
            *mean_line,
        ]
        return bands, layers

    return bands, [
        p9.geom_ribbon(
            bands,
            p9.aes(
                x,
                ymin=f"{var_name}_lower",
                ymax=f"{var_name}_upper",
                fill="series",
            )
            if has_series
            else p9.aes(
                x,
                ymin=f"{var_name}_lower",
                ymax=f"{var_name}_upper",
            ),
            alpha=0.3,
            show_legend=False,
        ),
        *mean_line,
    ]


def _categorize_panels(
    frames: list[pd.DataFrame],
    panels: list[str] | tuple[str, ...],
) -> None:
    for frame in frames:
        frame["panel"] = pd.Categorical(frame["panel"], categories=panels, ordered=True)


@dataclass(frozen=True)
class CausalPanelData:
    """Posterior quantities and observations for a three-panel causal plot."""

    fitted: pl.DataFrame
    counterfactual: pl.DataFrame
    post_effect: pl.DataFrame
    cumulative_effect: pl.DataFrame
    observations: pd.DataFrame
    pre_effect: pl.DataFrame | None = None


def _causal_panel_draws(
    data: CausalPanelData,
    *,
    panel_titles: tuple[str, str, str],
    series_labels: dict[str, str],
) -> pl.DataFrame:
    top, middle, bottom = panel_titles
    parts = [
        label_draws(data.fitted, series=series_labels["fitted"], panel=top),
        label_draws(
            data.counterfactual,
            series=series_labels["counterfactual"],
            panel=top,
        ),
        label_draws(
            data.post_effect,
            series=series_labels["post_effect"],
            panel=middle,
        ),
        label_draws(
            data.cumulative_effect,
            series=series_labels["cumulative_effect"],
            panel=bottom,
        ),
    ]
    if data.pre_effect is not None:
        parts.append(
            label_draws(
                data.pre_effect,
                series=series_labels["pre_effect"],
                panel=middle,
            )
        )
    return pl.concat(parts, how="diagonal_relaxed")


def _observations_for_plot(
    observations: pd.DataFrame,
    *,
    x: str,
    panel: str,
) -> pd.DataFrame:
    obs = observations.rename(columns={"value": "y"})
    obs["series"] = "Observations"
    obs["panel"] = panel
    return obs


def _derive_effect_area(
    intervals: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    panel_titles: tuple[str, str, str],
    series_labels: dict[str, str],
    shade_outcome: bool,
    x: str,
    post_index: pd.Index | None,
) -> pd.DataFrame | None:
    top, middle, _bottom = panel_titles
    counterfactual_label = series_labels["counterfactual"]
    post_effect_label = series_labels["post_effect"]
    post_prediction = intervals[
        (intervals["panel"] == top) & (intervals["series"] == counterfactual_label)
    ]
    post_impact = intervals[
        (intervals["panel"] == middle) & (intervals["series"] == post_effect_label)
    ]
    areas: list[pd.DataFrame] = []
    if (
        shade_outcome
        and not post_prediction.empty
        and (post_index is None or len(post_index) > 1)
    ):
        obs_frame = observations.rename(columns={"value": "y"})
        if post_index is not None:
            post_obs = obs_frame.loc[obs_frame[x].isin(post_index.tolist()), [x, "y"]]
        else:
            post_obs = obs_frame.loc[obs_frame[x].isin(post_prediction[x]), [x, "y"]]
        areas.append(
            post_prediction[[x, "mu"]]
            .merge(post_obs, on=x)
            .rename(columns={"mu": "y1", "y": "y2"})
            .assign(panel=top)
        )
    if not post_impact.empty and (post_index is None or len(post_index) > 1):
        areas.append(
            post_impact[[x, "mu"]]
            .rename(columns={"mu": "y1"})
            .assign(y2=0.0, panel=middle)
        )
    if not areas:
        return None
    return pd.concat(areas, ignore_index=True)


def _derive_singleton_intervals(
    intervals: pd.DataFrame,
    *,
    panel_titles: tuple[str, str, str],
    series_labels: dict[str, str],
    post_index: pd.Index | None,
) -> pd.DataFrame:
    if post_index is None or len(post_index) != 1:
        return intervals.iloc[0:0].copy()
    top, middle, bottom = panel_titles
    counterfactual_label = series_labels["counterfactual"]
    post_effect_label = series_labels["post_effect"]
    cumulative_label = series_labels["cumulative_effect"]
    return pd.concat(
        [
            intervals[
                (intervals["panel"] == top)
                & (intervals["series"] == counterfactual_label)
            ],
            intervals[
                (intervals["panel"] == middle)
                & (intervals["series"] == post_effect_label)
            ],
            intervals[
                (intervals["panel"] == bottom)
                & (intervals["series"] == cumulative_label)
            ],
        ],
        ignore_index=True,
    )


def add_causal_panel_legend(
    ax: plt.Axes,
    *,
    labels: list[str],
    colors: dict[str, str],
    area_labels: set[str] | None = None,
) -> None:
    """Add the Matplotlib legend required by the public ``legend_kwargs`` API.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Top panel that owns the legend.
    labels : list of str
        Ordered legend entries.
    colors : dict
        Color for each legend entry.
    area_labels : set of str, optional
        Entries represented by filled areas rather than lines.
    """
    area_labels = area_labels or set()
    handles = []
    for label in labels:
        if label in area_labels:
            handles.append(Patch(facecolor=colors[label], alpha=0.25))
        elif label == "Observations":
            handles.append(
                Line2D([0], [0], color=colors[label], marker=".", linestyle="")
            )
        else:
            handles.append(Line2D([0], [0], color=colors[label]))
    ax.legend(handles=handles, labels=labels, fontsize=LEGEND_FONT_SIZE)


def build_causal_panel_plot(
    panel_data: CausalPanelData,
    *,
    panels: tuple[str, str, str],
    series_labels: dict[str, str],
    colors: dict[str, str],
    kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
    ci_prob: float = HDI_PROB,
    interval: Literal["hdi", "eti"] = "hdi",
    num_samples: int = 50,
    x: str = "obs_ind",
    shade_fill: str = "#1f77b4",
    figsize: tuple[float, float] = (7, 11),
    zero_linetype: str | None = None,
    zero_alpha: float = 1.0,
    post_index: pd.Index | None = None,
    singleton_color: str = "#ff7f0e",
    shade_outcome: bool = True,
) -> Any:
    """Three-panel causal-impact layout shared by ITS, SC, SDiD, and PiecewiseITS.

    Derives interval summaries, shading, posterior layers, and singleton
    point-ranges from explicit ``CausalPanelData`` quantities.

    Parameters
    ----------
    panel_data : CausalPanelData
        Semantic draws and observations from an experiment extractor.
    panels : tuple of str
        Ordered facet labels (top, middle, bottom).
    series_labels : dict
        Display labels keyed by explicit posterior quantity names.
    colors : dict
        Series name to color mapping for rendered layers.
    kind : {"ribbon", "histogram", "spaghetti"}, optional
        Posterior rendering mode.
    ci_prob : float, optional
        Credible interval probability mass.
    interval : {"hdi", "eti"}, optional
        Interval type.
    num_samples : int, optional
        Spaghetti paths to sample when ``kind="spaghetti"``.
    x : str, optional
        Shared x column name. Defaults to ``"obs_ind"``.
    shade_fill : str, optional
        Fill color for causal-impact shading ribbons.
    figsize : tuple of float, optional
        plotnine ``figure_size``.
    zero_linetype : str, optional
        ``geom_hline`` linetype for zero reference lines.
    zero_alpha : float, optional
        Alpha for zero reference lines.
    post_index : pandas.Index, optional
        Post-period index for outcome shading and singleton point-ranges.
    singleton_color : str, optional
        Color for singleton post-period point-ranges.
    shade_outcome : bool, optional
        Whether to shade the top-panel counterfactual gap.
    """
    validate_posterior_plot_options(kind, ci_kind=interval, num_samples=num_samples)

    panel_titles: tuple[str, str, str] = (panels[0], panels[1], panels[2])
    panel_draws = _causal_panel_draws(
        panel_data,
        panel_titles=panel_titles,
        series_labels=series_labels,
    )
    grouping = ["panel", "series", x]
    intervals, posterior_layers = posterior_kind_layers(
        panel_draws,
        kind,
        x=x,
        group_by=grouping,
        ci_prob=ci_prob,
        interval=interval,
        num_samples=num_samples,
        colors=colors,
    )
    obs = _observations_for_plot(panel_data.observations, x=x, panel=panels[0])
    effect_area = _derive_effect_area(
        intervals,
        panel_data.observations,
        panel_titles=panel_titles,
        series_labels=series_labels,
        shade_outcome=shade_outcome,
        x=x,
        post_index=post_index,
    )
    singleton_intervals = _derive_singleton_intervals(
        intervals,
        panel_titles=panel_titles,
        series_labels=series_labels,
        post_index=post_index,
    )
    if not singleton_intervals.empty:
        singleton_intervals["panel"] = pd.Categorical(
            singleton_intervals["panel"], categories=panels, ordered=True
        )

    frames = [intervals, obs]
    if effect_area is not None:
        frames.append(effect_area)
    _categorize_panels(frames, panels)

    mid, bot = panels[1], panels[2]
    zero_df = pd.DataFrame({"yintercept": [0.0, 0.0], "panel": [mid, bot]})
    _categorize_panels([zero_df], panels)

    p = p9.ggplot()
    if effect_area is not None:
        p = p + p9.geom_ribbon(
            effect_area,
            p9.aes(x, ymin="y1", ymax="y2"),
            fill=shade_fill,
            alpha=0.25,
        )
    for layer in posterior_layers:
        p += layer
    hline_kwargs: dict[str, Any] = {"color": "black", "alpha": zero_alpha}
    if zero_linetype is not None:
        hline_kwargs["linetype"] = zero_linetype
    scales = [p9.scale_color_manual(values=colors, name="")]
    if kind != "histogram":
        scales.append(p9.scale_fill_manual(values=colors, name=""))
    plot_theme: dict[str, Any] = {
        "figure_size": figsize,
        "panel_spacing_y": 0.06,
        "plot_margin_bottom": 0.08,
    }
    p = (
        p
        + p9.geom_point(obs, p9.aes(x, "y", color="series"), size=1)
        + p9.geom_hline(zero_df, p9.aes(yintercept="yintercept"), **hline_kwargs)
        + p9.facet_wrap("panel", ncol=1, scales="free_y")
        + scales[0]
        + (scales[1] if len(scales) > 1 else p9.guides())
        + p9.labs(x="", y="")
        + p9.theme(**plot_theme)
    )
    if kind == "histogram":
        p = p + HISTOGRAM_PANEL_THEME
    if not singleton_intervals.empty:
        p += p9.geom_pointrange(
            singleton_intervals,
            p9.aes(x, "mu", ymin="mu_lower", ymax="mu_upper"),
            color=singleton_color,
            size=0.5,
            show_legend=False,
        )
    return p
