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

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import tidydraws as td
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pandas.api.extensions import ExtensionArray
from plotnine import (
    aes,
    element_blank,
    element_rect,
    facet_wrap,
    geom_hline,
    geom_line,
    geom_point,
    geom_ribbon,
    geom_tile,
    guides,
    labs,
    scale_color_manual,
    scale_fill_continuous,
    scale_fill_manual,
    theme,
)

from causalpy.constants import HDI_PROB, LEGEND_FONT_SIZE

HISTOGRAM_PANEL_THEME = theme(
    panel_background=element_rect(fill="white"),
    panel_grid_major=element_blank(),
    panel_grid_minor=element_blank(),
)


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


def interval_kind(ci_kind: Literal["hdi", "eti"]) -> Literal["hdi", "eti"]:
    """Map public ``ci_kind`` to tidydraws ``interval`` argument.

    Parameters
    ----------
    ci_kind : {"hdi", "eti"}
        Credible interval type from experiment ``plot()`` APIs.
    """
    return "eti" if ci_kind == "eti" else "hdi"


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
    sampled = sample_draw_lines(draws, num_samples, sort_by=sort_by or group_by)
    identity = [col for col in ("panel", "series") if col in sampled.columns]
    if identity:
        sampled = sampled.with_columns(
            pl.concat_str(
                [pl.col(col) for col in identity] + [pl.col("_draw_id")],
                separator=":",
            ).alias("_line_id")
        )
    return sampled.to_pandas()


def posterior_kind_layers(
    bands: pd.DataFrame,
    kind: Literal["ribbon", "histogram", "spaghetti"],
    *,
    x: str,
    y: str = "mu",
    spaghetti_df: pd.DataFrame | None = None,
    histogram_tiles: pd.DataFrame | None = None,
    ymin: str = "mu_lower",
    ymax: str = "mu_upper",
    spaghetti_group: str = "_draw_id",
) -> list[Any]:
    """plotnine layers for ribbon, spaghetti, or histogram (mean-line) modes.

    Parameters
    ----------
    bands : pandas.DataFrame
        Mean + interval summary from tidydraws.
    kind : {"ribbon", "histogram", "spaghetti"}
        Posterior rendering mode.
    x, y : str
        Column names for the x axis and mean line.
    spaghetti_df : pandas.DataFrame, optional
        Sampled draws when ``kind="spaghetti"``.
    histogram_tiles : pandas.DataFrame, optional
        Tidy tile grid from :func:`posterior_histogram_tiles` when
        ``kind="histogram"``.
    ymin, ymax : str, optional
        Interval bound columns for ribbon mode.
    spaghetti_group : str, optional
        Grouping column for spaghetti lines.
    """
    if kind == "histogram":
        layers: list[Any] = []
        if histogram_tiles is not None:
            layers.extend(histogram_tile_layers(histogram_tiles, x))
        layers.append(geom_line(bands, aes(x, y, color="series")))
        return layers
    if kind == "spaghetti":
        if spaghetti_df is None:
            msg = "spaghetti_df is required when kind='spaghetti'"
            raise ValueError(msg)
        return [
            geom_line(
                spaghetti_df,
                aes(x, y, group=spaghetti_group, color="series"),
                alpha=0.1,
                size=0.3,
                show_legend=False,
            ),
            geom_line(bands, aes(x, y, color="series")),
        ]
    return [
        geom_ribbon(
            bands,
            aes(x, ymin=ymin, ymax=ymax, fill="series"),
            alpha=0.3,
            show_legend=False,
        ),
        geom_line(bands, aes(x, y, color="series")),
    ]


def add_posterior_kind(
    p: Any,
    bands: pd.DataFrame,
    kind: Literal["ribbon", "histogram", "spaghetti"],
    *,
    x: str,
    y: str = "mu",
    spaghetti_df: pd.DataFrame | None = None,
    histogram_tiles: pd.DataFrame | None = None,
    ymin: str = "mu_lower",
    ymax: str = "mu_upper",
    spaghetti_group: str = "_draw_id",
) -> Any:
    """Append ribbon/spaghetti/histogram layers to a plotnine ggplot.

    Parameters
    ----------
    p : plotnine.ggplot
        Base ggplot object.
    bands : pandas.DataFrame
        Mean + interval summary from tidydraws.
    kind : {"ribbon", "histogram", "spaghetti"}
        Posterior rendering mode.
    x, y : str
        Column names for the x axis and mean line.
    spaghetti_df : pandas.DataFrame, optional
        Sampled draws when ``kind="spaghetti"``.
    histogram_tiles : pandas.DataFrame, optional
        Tidy tile grid when ``kind="histogram"``.
    ymin, ymax : str, optional
        Interval bound columns for ribbon mode.
    spaghetti_group : str, optional
        Grouping column for spaghetti lines.
    """
    for layer in posterior_kind_layers(
        bands,
        kind,
        x=x,
        y=y,
        spaghetti_df=spaghetti_df,
        histogram_tiles=histogram_tiles,
        ymin=ymin,
        ymax=ymax,
        spaghetti_group=spaghetti_group,
    ):
        p = p + layer
    return p


def _categorize_panels(frames: list[pd.DataFrame], panels: list[str]) -> None:
    for frame in frames:
        frame["panel"] = pd.Categorical(frame["panel"], categories=panels, ordered=True)


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
    bands: pd.DataFrame,
    obs: pd.DataFrame,
    *,
    panels: list[str],
    colors: dict[str, str],
    show_panel_titles: bool = False,
    kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
    x: str = "obs_ind",
    shade_df: pd.DataFrame | None = None,
    shade_fill: str = "#1f77b4",
    spaghetti_df: pd.DataFrame | None = None,
    histogram_tiles: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (7, 11),
    zero_linetype: str | None = None,
    zero_alpha: float = 1.0,
) -> Any:
    """Three-panel causal-impact layout shared by ITS, SC, SDiD, and PiecewiseITS.

    Parameters
    ----------
    bands : pandas.DataFrame
        Mean + interval summaries with ``panel`` and ``series`` columns.
    obs : pandas.DataFrame
        Observed points for the top panel.
    panels : list of str
        Ordered facet labels (top, middle, bottom).
    colors : dict
        Series name to color mapping.
    show_panel_titles : bool, optional
        Whether to display facet labels as panel titles.
    kind : {"ribbon", "histogram", "spaghetti"}, optional
        Posterior rendering mode.
    x : str, optional
        Shared x column name. Defaults to ``"obs_ind"``.
    shade_df : pandas.DataFrame, optional
        Ribbon data for causal-impact shading.
    shade_fill : str, optional
        Fill color for the shade ribbon.
    spaghetti_df : pandas.DataFrame, optional
        Sampled draws when ``kind="spaghetti"``.
    histogram_tiles : pandas.DataFrame, optional
        Tidy posterior heatmap tiles when ``kind="histogram"``.
    figsize : tuple of float, optional
        plotnine ``figure_size``.
    zero_linetype : str, optional
        ``geom_hline`` linetype for zero reference lines.
    zero_alpha : float, optional
        Alpha for zero reference lines.
    """
    from plotnine import ggplot

    frames = [bands, obs]
    if shade_df is not None:
        frames.append(shade_df)
    if spaghetti_df is not None:
        frames.append(spaghetti_df)
    _categorize_panels(frames, panels)

    mid, bot = panels[1], panels[2]
    zero_df = pd.DataFrame({"yintercept": [0.0, 0.0], "panel": [mid, bot]})
    _categorize_panels([zero_df], panels)

    p = ggplot()
    if shade_df is not None:
        p = p + geom_ribbon(
            shade_df,
            aes(x, ymin="y1", ymax="y2"),
            fill=shade_fill,
            alpha=0.25,
        )
    p = add_posterior_kind(
        p,
        bands,
        kind,
        x=x,
        spaghetti_df=spaghetti_df,
        histogram_tiles=histogram_tiles,
    )
    hline_kwargs: dict[str, Any] = {"color": "black", "alpha": zero_alpha}
    if zero_linetype is not None:
        hline_kwargs["linetype"] = zero_linetype
    scales = [scale_color_manual(values=colors, name="")]
    if kind != "histogram":
        scales.append(scale_fill_manual(values=colors, name=""))
    plot_theme: dict[str, Any] = {
        "figure_size": figsize,
        "panel_spacing_y": 0.06,
        "plot_margin_bottom": 0.08,
    }
    if not show_panel_titles:
        plot_theme.update(
            strip_text=element_blank(),
            strip_background=element_blank(),
        )
    p = (
        p
        + geom_point(obs, aes(x, "y", color="series"), size=1)
        + geom_hline(zero_df, aes(yintercept="yintercept"), **hline_kwargs)
        + facet_wrap("panel", ncol=1, scales="free_y")
        + scales[0]
        + (scales[1] if len(scales) > 1 else guides())
        + guides(color="none", fill="none")
        + labs(x="", y="")
        + theme(**plot_theme)
    )
    if kind == "histogram":
        p = p + HISTOGRAM_PANEL_THEME
    return p


def histogram_tile_layers(
    tiles: pd.DataFrame,
    x: str,
    *,
    y: str | None = None,
    alpha: float = 0.85,
) -> list[Any]:
    """plotnine layers for a posterior heatmap grid.

    Parameters
    ----------
    tiles : pandas.DataFrame
        Tidy tile grid from :func:`posterior_histogram_tiles`.
    x : str
        Column name for tile x centers.
    y : str, optional
        Column name for tile y centers; inferred when omitted.
    alpha : float, optional
        Tile opacity.
    """
    y_col = y or next(col for col in ("y", "_y", "mu") if col in tiles.columns)
    fill_max = float(tiles["density"].max()) if len(tiles) else 1.0
    return [
        geom_tile(
            tiles,
            aes(x, y_col, fill="density", width="width", height="height"),
            alpha=alpha,
            show_legend=False,
            inherit_aes=False,
        ),
        scale_fill_continuous(
            cmap_name="Greys",
            limits=(0.0, fill_max if fill_max > 0 else 1.0),
        ),
    ]


def _histogram_count_grid(
    draws: pl.DataFrame,
    x: str,
    y_edges: np.ndarray | None = None,
    *,
    var_name: str = "mu",
    n_bins: int = 50,
) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """Posterior draw proportions per y-bin and x value."""
    frame = draws.select(x, var_name).to_pandas()
    x_values = pd.Index(frame[x].drop_duplicates()).sort_values()
    if x_values.empty:
        raise ValueError("posterior draws contain no x values")
    if y_edges is None:
        vals = frame[var_name].to_numpy()
        y_min = float(np.nanmin(vals))
        y_max = float(np.nanmax(vals))
        y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        y_edges = np.linspace(y_min - y_pad, y_max + y_pad, n_bins + 1)
    density = np.column_stack(
        [
            np.histogram(frame.loc[frame[x] == x_value, var_name], bins=y_edges)[0]
            / max(int((frame[x] == x_value).sum()), 1)
            for x_value in x_values
        ]
    )
    return x_values, y_edges, density


def posterior_histogram_tiles(
    draws: pl.DataFrame,
    x: str,
    *,
    x_col: str | None = None,
    y_col: str = "y",
    var_name: str = "mu",
    y_edges: np.ndarray | None = None,
    panel: str | None = None,
) -> pd.DataFrame:
    """Build a tidy ``geom_tile`` grid from canonical long posterior draws.

    Parameters
    ----------
    draws : polars.DataFrame
        Long posterior draws containing the x and value columns.
    x : str
        X column in ``draws``.
    x_col, y_col : str, optional
        Output column names. ``x_col`` defaults to ``x``.
    var_name : str, optional
        Posterior value column. Defaults to ``"mu"``.
    y_edges : numpy.ndarray, optional
        Shared bin edges from :func:`histogram_y_edges`.
    panel : str, optional
        Facet label when plotting faceted panels.
    """
    x_values, y_edges, density = _histogram_count_grid(
        draws,
        x,
        y_edges=y_edges,
        var_name=var_name,
    )
    x_left, x_right = _x_mesh_edges(x_values)
    x_arr = x_values.to_numpy()
    x_centers: Any
    if isinstance(x_values, pd.DatetimeIndex):
        x_centers = x_arr
    elif np.issubdtype(x_arr.dtype, np.datetime64):
        x_centers = pd.to_datetime(x_arr)
    else:
        x_centers = (x_left + x_right) / 2.0

    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    heights = np.diff(y_edges)
    _, is_dt = _x_as_numeric_mesh(x_values)
    widths: Any
    if is_dt:
        widths = (pd.to_datetime(x_right) - pd.to_datetime(x_left)) / np.timedelta64(
            1, "D"
        )
        widths = np.asarray(widths, dtype=float)
    else:
        widths = x_right - x_left

    output_x = x_col or x
    tiles = pd.DataFrame(
        {
            output_x: np.repeat(x_centers, len(y_centers)),
            y_col: np.tile(y_centers, len(x_centers)),
            "width": np.repeat(widths, len(y_centers)),
            "height": np.tile(heights, len(x_centers)),
            "density": density.T.reshape(-1),
        }
    )
    if panel is not None:
        tiles["panel"] = panel
    return tiles


def histogram_y_edges(
    *draw_frames: pl.DataFrame,
    var_name: str = "mu",
    n_bins: int = 50,
) -> np.ndarray:
    """Shared y-bin edges for multiple posterior heatmaps on one axes.

    Parameters
    ----------
    *draw_frames : polars.DataFrame
        One or more canonical long posterior-draw tables.
    var_name : str, optional
        Posterior value column. Defaults to ``"mu"``.
    n_bins : int, optional
        Number of histogram bins.
    """
    if not draw_frames:
        raise ValueError("histogram_y_edges requires at least one draw frame")
    vals = np.concatenate(
        [draws.get_column(var_name).to_numpy() for draws in draw_frames]
    )
    y_min = float(np.nanmin(vals))
    y_max = float(np.nanmax(vals))
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    return np.linspace(y_min - y_pad, y_max + y_pad, n_bins + 1)


def _x_as_numeric_mesh(
    x: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
) -> tuple[np.ndarray, bool]:
    """Convert x to floats for mesh edges; return (values, is_datetime)."""
    if isinstance(x, pd.DatetimeIndex):
        return mdates.date2num(x.to_numpy()), True
    x_arr = np.asarray(x)
    if np.issubdtype(x_arr.dtype, np.datetime64):
        return mdates.date2num(pd.to_datetime(x_arr)), True
    if x_arr.dtype == object:
        try:
            dt = pd.to_datetime(x_arr)
            if pd.api.types.is_datetime64_any_dtype(dt):
                return mdates.date2num(dt), True
        except (ValueError, TypeError):
            pass
    return np.asarray(x_arr, dtype=float), False


def _x_mesh_edges(
    x: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
) -> tuple[np.ndarray, np.ndarray]:
    """Left/right x bounds per observation for heatmap cells."""
    x_num, is_dt = _x_as_numeric_mesh(x)
    if len(x_num) == 1:
        x_edges = np.array([x_num[0] - 0.5, x_num[0] + 0.5])
    else:
        dx = np.diff(x_num)
        x_edges = np.zeros(len(x_num) + 1)
        x_edges[0] = x_num[0] - dx[0] / 2
        x_edges[-1] = x_num[-1] + dx[-1] / 2
        x_edges[1:-1] = x_num[:-1] + dx / 2
    if is_dt:
        idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(x)))
        x_left = pd.DatetimeIndex(
            [pd.Timestamp(t) for t in mdates.num2date(x_edges[:-1])]
        )
        x_right = pd.DatetimeIndex(
            [pd.Timestamp(t) for t in mdates.num2date(x_edges[1:])]
        )
        if idx.tz is None:
            x_left = x_left.tz_localize(None)
            x_right = x_right.tz_localize(None)
        else:
            x_left = x_left.tz_convert(idx.tz)
            x_right = x_right.tz_convert(idx.tz)
        return np.asarray(x_left), np.asarray(x_right)
    return x_edges[:-1], x_edges[1:]


def get_hdi_to_df(
    x: xr.DataArray,
    hdi_prob: float = HDI_PROB,
) -> pd.DataFrame:
    """Calculate and recover HDI intervals.

    Parameters
    ----------
    x : xr.DataArray
        Xarray data array.
    hdi_prob : float, optional
        The size of the HDI. Defaults to
        :data:`~causalpy.constants.HDI_PROB` (currently 0.94).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the HDI intervals with 'lower' and 'higher'
        columns.
    """
    hdi_result = az.hdi(x, hdi_prob=hdi_prob)
    data_var = list(hdi_result.data_vars)[0]
    hdi_df = hdi_result[data_var].to_dataframe()[[data_var]].unstack(level="hdi")
    hdi_df.columns = hdi_df.columns.droplevel(0)
    return hdi_df
