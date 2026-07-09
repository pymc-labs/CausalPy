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

from causalpy.constants import HDI_PROB

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
        return draws.filter(pl.col("treated_units") == treated_unit)
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


def prediction_summary(
    pred: Any,
    newdata: pd.DataFrame,
    *,
    group_by: str,
    ci_prob: float,
    interval: Literal["hdi", "eti"] = "hdi",
    var_name: str = "mu",
    treated_unit: str | None = None,
    draws: pl.DataFrame | None = None,
) -> pd.DataFrame:
    """Tidy mean + credible interval from posterior predictive draws.

    Parameters
    ----------
    pred : Any
        ArviZ ``InferenceData`` or prediction container for
        :func:`tidydraws.prediction_draws`.
    newdata : pandas.DataFrame
        Grid passed as ``newdata`` to tidydraws.
    group_by : str
        Column to summarise over.
    ci_prob : float
        Credible interval probability mass.
    interval : {"hdi", "eti"}, optional
        Interval type for :func:`tidydraws.point_interval`.
    var_name : str, optional
        Posterior variable name. Defaults to ``"mu"``.
    treated_unit : str, optional
        When draws include ``treated_units``, filter to this unit.
    draws : polars.DataFrame, optional
        Pre-extracted draws from :func:`prediction_draws`.
    """
    if draws is None:
        draws = prediction_draws(
            pred, newdata, var_name=var_name, treated_unit=treated_unit
        )
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


def prediction_spaghetti(
    pred: Any,
    newdata: pd.DataFrame,
    *,
    group_by: str,
    num_samples: int,
    var_name: str = "mu",
    treated_unit: str | None = None,
    sort_by: str | list[str] | None = None,
    draws: pl.DataFrame | None = None,
) -> pd.DataFrame:
    """Sampled posterior draw lines from a prediction object.

    Parameters
    ----------
    pred : Any
        Prediction container for :func:`tidydraws.prediction_draws`.
    newdata : pandas.DataFrame
        Grid passed as ``newdata`` to tidydraws.
    group_by : str
        x-axis column in the returned frame.
    num_samples : int
        Number of draw lines to sample.
    var_name : str, optional
        Posterior variable name. Defaults to ``"mu"``.
    treated_unit : str, optional
        When draws include ``treated_units``, filter to this unit.
    sort_by : str or list of str, optional
        Sort columns; defaults to ``group_by``.
    draws : polars.DataFrame, optional
        Pre-extracted draws from :func:`prediction_draws`.
    """
    if draws is None:
        draws = prediction_draws(
            pred, newdata, var_name=var_name, treated_unit=treated_unit
        )
    return sample_draw_lines(
        draws, num_samples, sort_by=sort_by or group_by
    ).to_pandas()


def prediction_bundle(
    pred: Any,
    newdata: pd.DataFrame,
    *,
    group_by: str,
    ci_prob: float,
    interval: Literal["hdi", "eti"] = "hdi",
    var_name: str = "mu",
    treated_unit: str | None = None,
    kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
    num_samples: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Summary and optional spaghetti from a single tidydraws extraction.

    Parameters
    ----------
    pred : Any
        Prediction container for :func:`prediction_draws`.
    newdata : pandas.DataFrame
        Grid passed as ``newdata`` to tidydraws.
    group_by : str
        Column to summarise over.
    ci_prob : float
        Credible interval probability mass.
    interval : {"hdi", "eti"}, optional
        Interval type for :func:`tidydraws.point_interval`.
    var_name : str, optional
        Posterior variable name. Defaults to ``"mu"``.
    treated_unit : str, optional
        When draws include ``treated_units``, filter to this unit.
    kind : {"ribbon", "histogram", "spaghetti"}, optional
        When ``"spaghetti"``, also return sampled draw lines.
    num_samples : int, optional
        Number of draw lines when ``kind="spaghetti"``.
    """
    draws = prediction_draws(
        pred, newdata, var_name=var_name, treated_unit=treated_unit
    )
    summary = prediction_summary(
        pred,
        newdata,
        group_by=group_by,
        ci_prob=ci_prob,
        interval=interval,
        var_name=var_name,
        draws=draws,
    )
    spaghetti_df = None
    if kind == "spaghetti":
        spaghetti_df = prediction_spaghetti(
            pred,
            newdata,
            group_by=group_by,
            num_samples=num_samples,
            var_name=var_name,
            draws=draws,
        )
    return summary, spaghetti_df


def da_summary(
    da: xr.DataArray,
    *,
    group_by: str,
    ci_prob: float,
    interval: Literal["hdi", "eti"] = "hdi",
    treated_unit: str | None = None,
) -> pd.DataFrame:
    """Tidy mean + credible interval from a posterior DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        Posterior samples with ``chain`` and ``draw`` dimensions.
    group_by : str
        Column to summarise over after tidying.
    ci_prob : float
        Credible interval probability mass.
    interval : {"hdi", "eti"}, optional
        Interval type for :func:`tidydraws.point_interval`.
    treated_unit : str, optional
        Select this ``treated_units`` level when present.
    """
    if treated_unit is not None:
        da = da.sel(treated_units=treated_unit)
    elif hasattr(da, "dims") and "treated_units" in da.dims:
        da = da.isel(treated_units=0)
    tidy = pl.from_pandas(da.to_dataframe(name="mu").reset_index())
    return (
        td.point_interval(
            tidy,
            "mu",
            group_by=group_by,
            probs=(ci_prob,),
            point="mean",
            interval=interval,
        )
        .sort(group_by)
        .to_pandas()
    )


def da_spaghetti(
    da: xr.DataArray,
    *,
    group_by: str,
    num_samples: int,
    treated_unit: str | None = None,
    sort_by: str | list[str] | None = None,
) -> pd.DataFrame:
    """Sampled posterior draw lines from a DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        Posterior samples with ``chain`` and ``draw`` dimensions.
    group_by : str
        x-axis column in the returned frame.
    num_samples : int
        Number of draw lines to sample.
    treated_unit : str, optional
        Select this ``treated_units`` level when present.
    sort_by : str or list of str, optional
        Sort columns; defaults to ``group_by``.
    """
    if treated_unit is not None:
        da = da.sel(treated_units=treated_unit)
    elif hasattr(da, "dims") and "treated_units" in da.dims:
        da = da.isel(treated_units=0)
    tidy = pl.from_pandas(da.to_dataframe(name="mu").reset_index())
    return sample_draw_lines(tidy, num_samples, sort_by=sort_by or group_by).to_pandas()


@dataclass(frozen=True)
class HistogramLayer:
    """One posterior heatmap source for :func:`concat_histogram_tiles`."""

    x: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray
    Y: xr.DataArray
    panel: str | None = None
    y_edges: np.ndarray | None = None


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
        Tidy tile grid from :func:`concat_histogram_tiles` when
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


def build_causal_panel_plot(
    bands: pd.DataFrame,
    obs: pd.DataFrame,
    *,
    panels: list[str],
    colors: dict[str, str],
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
    p = (
        p
        + geom_point(obs, aes(x, "y", color="series"), size=1)
        + geom_hline(zero_df, aes(yintercept="yintercept"), **hline_kwargs)
        + facet_wrap("panel", ncol=1, scales="free_y")
        + scales[0]
        + (scales[1] if len(scales) > 1 else guides())
        + guides(color="none", fill="none")
        + labs(x="", y="")
        + theme(
            strip_text=element_blank(),
            strip_background=element_blank(),
            figure_size=figsize,
            panel_spacing_y=0.06,
            plot_margin_bottom=0.08,
        )
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
    Y: xr.DataArray,
    y_edges: np.ndarray | None = None,
    *,
    n_bins: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior draw counts per (y-bin, x-index) for one xarray series."""
    Y_flat = Y.stack(sample=("chain", "draw"))
    time_dim = _time_dim_name(Y)
    n_time = Y.sizes[time_dim]
    if y_edges is None:
        vals = Y_flat.values
        y_min = float(np.nanmin(vals))
        y_max = float(np.nanmax(vals))
        y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        y_edges = np.linspace(y_min - y_pad, y_max + y_pad, n_bins + 1)
    n_bins_actual = len(y_edges) - 1
    hist2d = np.zeros((n_bins_actual, n_time), dtype=float)
    for t in range(n_time):
        col = Y_flat.isel({time_dim: t}).values.ravel()
        counts, _ = np.histogram(col, bins=y_edges)
        hist2d[:, t] = counts
    return y_edges, hist2d


def posterior_histogram_tiles(
    x: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
    Y: xr.DataArray,
    *,
    x_col: str = "x",
    y_col: str = "y",
    y_edges: np.ndarray | None = None,
    panel: str | None = None,
    normalize: Literal["proportion", "column"] = "proportion",
) -> pd.DataFrame:
    """Tidy tile grid for ``geom_tile`` posterior heatmaps.

    Parameters
    ----------
    x : array-like
        x-axis values aligned with ``Y``'s time dimension.
    Y : xarray.DataArray
        Posterior samples with ``chain`` and ``draw`` dimensions.
    x_col, y_col : str
        Column names in the returned frame.
    y_edges : numpy.ndarray, optional
        Shared bin edges from :func:`histogram_y_edges`.
    panel : str, optional
        Facet label when plotting faceted panels.
    normalize : {"proportion", "column"}, optional
        ``proportion`` divides counts by the MCMC sample size at each x;
        ``column`` scales each x column by its own maximum (legacy look).
    """
    n_x = len(np.asarray(x))
    if n_x != Y.sizes[_time_dim_name(Y)]:
        msg = (
            f"Length of x ({n_x}) != length of time dimension "
            f"{_time_dim_name(Y)!r} ({Y.sizes[_time_dim_name(Y)]})"
        )
        raise ValueError(msg)

    y_edges, hist2d = _histogram_count_grid(Y, y_edges=y_edges)
    if normalize == "column":
        col_max = hist2d.max(axis=0, keepdims=True)
        density = np.divide(
            hist2d,
            col_max + 1e-12,
            out=np.zeros_like(hist2d, dtype=float),
            where=col_max > 0,
        )
    else:
        n_samples = float(Y.stack(sample=("chain", "draw")).sizes["sample"])
        density = hist2d / n_samples

    x_left, x_right = _x_mesh_edges(x)
    x_arr = np.asarray(x)
    x_centers: Any
    if isinstance(x, pd.DatetimeIndex):
        x_centers = x_arr
    elif np.issubdtype(np.asarray(x_arr).dtype, np.datetime64):
        x_centers = pd.to_datetime(x_arr)
    else:
        x_centers = (x_left + x_right) / 2.0

    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    heights = np.diff(y_edges)
    _, is_dt = _x_as_numeric_mesh(x)
    widths: Any
    if is_dt:
        widths = (pd.to_datetime(x_right) - pd.to_datetime(x_left)) / np.timedelta64(
            1, "D"
        )
        widths = np.asarray(widths, dtype=float)
    else:
        widths = x_right - x_left

    rows: list[dict[str, Any]] = []
    for t in range(n_x):
        for b in range(len(y_centers)):
            row = {
                x_col: x_centers[t],
                y_col: float(y_centers[b]),
                "width": float(widths[t]),
                "height": float(heights[b]),
                "density": float(density[b, t]),
            }
            if panel is not None:
                row["panel"] = panel
            rows.append(row)
    return pd.DataFrame(rows)


def concat_histogram_tiles(
    layers: list[HistogramLayer],
    *,
    x_col: str = "x",
    y_col: str = "y",
    normalize: Literal["proportion", "column"] = "proportion",
) -> pd.DataFrame:
    """Concatenate tidy heatmap tiles for multiple posterior series/panels.

    Parameters
    ----------
    layers : list of HistogramLayer
        One heatmap source per series or facet panel.
    x_col, y_col : str
        Column names passed through to :func:`posterior_histogram_tiles`.
    normalize : {"proportion", "column"}, optional
        Density scaling passed through to :func:`posterior_histogram_tiles`.
    """
    if not layers:
        raise ValueError("concat_histogram_tiles requires at least one layer")
    parts = [
        posterior_histogram_tiles(
            layer.x,
            layer.Y,
            x_col=x_col,
            y_col=y_col,
            y_edges=layer.y_edges,
            panel=layer.panel,
            normalize=normalize,
        )
        for layer in layers
    ]
    return pd.concat(parts, ignore_index=True)


def _time_dim_name(Y: xr.DataArray) -> str:
    time_dims = [d for d in Y.dims if d not in ("chain", "draw")]
    if len(time_dims) != 1:
        msg = (
            "concat_x_y expects Y with exactly one non-chain/draw dimension; "
            f"got {time_dims!r}"
        )
        raise ValueError(msg)
    return time_dims[0]


def _concat_x_values(
    x_left: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
    x_right: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
) -> pd.DatetimeIndex | np.ndarray | pd.Index:
    if isinstance(x_left, pd.DatetimeIndex) and isinstance(x_right, pd.DatetimeIndex):
        return x_left.append(x_right)
    combined = np.concatenate([np.asarray(x_left), np.asarray(x_right)])
    if isinstance(x_left, (pd.DatetimeIndex, pd.Index)):
        return pd.Index(combined)
    return combined


def concat_x_y(
    x_left: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
    Y_left: xr.DataArray,
    x_right: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
    Y_right: xr.DataArray,
) -> tuple[pd.DatetimeIndex | np.ndarray | pd.Index, xr.DataArray]:
    """Concatenate pre/post x and posterior arrays along time.

    Parameters
    ----------
    x_left : array-like
        x values for the left segment.
    x_right : array-like
        x values for the right segment.
    Y_left, Y_right : xarray.DataArray
        Posterior arrays aligned with each segment.
    """
    dim_left = _time_dim_name(Y_left)
    dim_right = _time_dim_name(Y_right)
    Y_right_aligned = (
        Y_right.rename({dim_right: dim_left}) if dim_right != dim_left else Y_right
    )
    Y = xr.concat([Y_left, Y_right_aligned], dim=dim_left)
    x = _concat_x_values(x_left, x_right)
    n_x = len(np.asarray(x))
    if n_x != Y.sizes[dim_left]:
        msg = (
            f"concat_x_y length mismatch: x has {n_x} points, "
            f"Y has {Y.sizes[dim_left]} along {dim_left!r}"
        )
        raise ValueError(msg)
    return x, Y


def histogram_y_edges(*Ys: xr.DataArray, n_bins: int = 50) -> np.ndarray:
    """Shared y-bin edges for multiple posterior heatmaps on one axes.

    Parameters
    ----------
    *Ys : xarray.DataArray
        One or more posterior arrays.
    n_bins : int, optional
        Number of histogram bins.
    """
    if not Ys:
        raise ValueError("histogram_y_edges requires at least one DataArray")
    vals = np.concatenate(
        [np.asarray(Y.stack(sample=("chain", "draw")).values).ravel() for Y in Ys]
    )
    y_min = float(np.nanmin(vals))
    y_max = float(np.nanmax(vals))
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    return np.linspace(y_min - y_pad, y_max + y_pad, n_bins + 1)


def plot_posterior_histogram(
    x: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
    Y: xr.DataArray,
    ax: plt.Axes,
    style: dict[str, Any] | None = None,
    label: str | None = None,
    y_edges: np.ndarray | None = None,
    *,
    draw_mean: bool = True,
    normalize: Literal["proportion", "column"] = "proportion",
) -> tuple[list[Line2D], None]:
    """Draw a posterior heatmap on matplotlib axes (test/legacy helper).

    Production plots use :func:`posterior_histogram_tiles` with ``geom_tile``.

    Parameters
    ----------
    x : array-like
        x-axis values aligned with ``Y``'s time dimension.
    Y : xarray.DataArray
        Posterior samples with ``chain`` and ``draw`` dimensions.
    ax : matplotlib.axes.Axes
        Target axes for the heatmap.
    style : dict, optional
        Matplotlib style overrides (``cmap``, ``alpha``, ``color``).
    label : str, optional
        Legend label for the posterior mean line.
    y_edges : numpy.ndarray, optional
        Shared bin edges from :func:`histogram_y_edges`.
    draw_mean : bool, optional
        Whether to overlay the posterior mean line.
    normalize : {"proportion", "column"}, optional
        Density scaling for the heatmap cells.
    """
    style = style or {}
    n_x = len(np.asarray(x))
    if n_x != Y.sizes[_time_dim_name(Y)]:
        msg = (
            f"Length of x ({n_x}) != length of time dimension "
            f"{_time_dim_name(Y)!r} ({Y.sizes[_time_dim_name(Y)]})"
        )
        raise ValueError(msg)

    y_edges_arr, hist2d = _histogram_count_grid(Y, y_edges=y_edges)
    if normalize == "column":
        col_max = hist2d.max(axis=0, keepdims=True)
        density = np.divide(
            hist2d,
            col_max + 1e-12,
            out=np.zeros_like(hist2d, dtype=float),
            where=col_max > 0,
        )
    else:
        n_samples = float(Y.stack(sample=("chain", "draw")).sizes["sample"])
        density = hist2d / n_samples

    x_left, x_right = _x_mesh_edges(x)
    x_num, is_dt = _x_as_numeric_mesh(x)
    if is_dt:
        x_edges = np.concatenate(
            [[mdates.date2num(x_left[0])], mdates.date2num(x_right)]
        )
    else:
        x_edges = np.concatenate([[x_left[0]], x_right])

    ax.pcolormesh(
        x_edges,
        y_edges_arr,
        density,
        cmap=style.get("cmap", "Greys"),
        vmin=0.0,
        vmax=float(density.max()) if density.size else 1.0,
        shading="flat",
        alpha=float(style.get("alpha", 0.85)),
        zorder=0.5,
    )
    if is_dt:
        ax.xaxis_date()

    if not draw_mean:
        return [], None

    mean_vals = np.asarray(Y.mean(dim=["chain", "draw"]).values, dtype=float).ravel()
    (mean_line,) = ax.plot(
        x_num,
        mean_vals,
        ls="-",
        color=style.get("color", "C0"),
        label=label or "Posterior mean",
    )
    return ([mean_line], None)


def overlay_posterior_histograms(
    axes: list[plt.Axes],
    layers: list[tuple[Any, xr.DataArray, dict[str, Any]]],
    *,
    style: dict[str, Any] | None = None,
) -> None:
    """Legacy matplotlib overlay helper kept for compatibility tests.

    Parameters
    ----------
    axes : list of matplotlib.axes.Axes
        Target axes, one per layer.
    layers : list of tuple
        Each ``(x, Y, style)`` triple for :func:`plot_posterior_histogram`.
    style : dict, optional
        Base style merged into each layer's style dict.
    """
    base = {"cmap": "Greys", "alpha": 0.85, **(style or {})}
    for ax, (x_vals, y_da, extra) in zip(axes, layers, strict=True):
        plot_posterior_histogram(x_vals, y_da, ax, {**base, **extra}, draw_mean=False)


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
