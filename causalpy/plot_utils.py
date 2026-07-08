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

import warnings
from typing import Any, Literal, TypedDict

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from pandas.api.extensions import ExtensionArray

from causalpy.constants import HDI_PROB


class _PlotXYStyle(TypedDict):
    """Typed kwargs bundle forwarded from ``_bayesian_plot`` to every ``plot_xY`` call."""

    ci_prob: float
    kind: Literal["ribbon", "histogram", "spaghetti"]
    ci_kind: Literal["hdi", "eti"]
    num_samples: int


def plot_xY(
    x: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
    Y: xr.DataArray,
    ax: plt.Axes,
    plot_hdi_kwargs: dict[str, Any] | None = None,
    ci_prob: float = HDI_PROB,
    label: str | None = None,
    kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
    ci_kind: Literal["hdi", "eti"] = "hdi",
    num_samples: int = 50,
    # Backward compatibility: hdi_prob was in original API
    hdi_prob: float | None = None,
) -> tuple[Line2D | list[Line2D], PolyCollection | None]:
    """Plot posterior intervals or samples.

    Parameters
    ----------
    x : pd.DatetimeIndex, np.ndarray, pd.Index, pd.Series, or ExtensionArray
        Pandas datetime index or numpy array of x-axis values.
    Y : xr.DataArray
        Xarray data array of y-axis data.
    ax : plt.Axes
        Matplotlib axes object.
    plot_hdi_kwargs : dict, optional
        Keyword arguments for line, band, heatmap, or sample styling (passed through
        to matplotlib / ArviZ helpers depending on ``kind`` and ``ci_kind``).
    ci_prob : float, optional
        The size of the credible interval. Defaults to
        :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
    label : str, optional
        The plot label.
    kind : {"ribbon", "histogram", "spaghetti"}, optional
        Type of visualization. Default is "ribbon".
    ci_kind : {"hdi", "eti"}, optional
        Type of interval for ribbon plots. Default is "hdi".
    num_samples : int, optional
        Number of posterior samples to plot for spaghetti visualization.
        Default is 50.
    hdi_prob : float, optional
        Backward-compatibility alias for ``ci_prob`` (same meaning as in earlier
        releases). There is no deprecation schedule; it may remain indefinitely.

    Returns
    -------
    tuple
        Depends on ``kind``:

        - ``kind="ribbon"``: ``(Line2D, PolyCollection)`` — mean line and
          interval band (HDI or ETI).
        - ``kind="histogram"`` or ``"spaghetti"``: ``(list[Line2D], None)`` —
          sample/mean lines and no single band patch.

        Experiment :meth:`~causalpy.experiments.base.BaseExperiment.plot` code
        that builds legends from ``plot_xY`` return values should only assume
        the ribbon shape when it passes ``kind="ribbon"`` (the default) through
        to :func:`plot_xY`.
    """
    # Handle backward compatibility: hdi_prob was in original API
    if hdi_prob is not None:
        ci_prob = hdi_prob

    if kind != "ribbon" and ci_kind != "hdi":
        warnings.warn(
            f"ci_kind={ci_kind!r} is ignored when kind={kind!r}. "
            "ci_kind only applies to kind='ribbon'.",
            UserWarning,
            stacklevel=2,
        )
    if kind != "spaghetti" and num_samples != 50:
        warnings.warn(
            f"num_samples={num_samples} is ignored when kind={kind!r}. "
            "num_samples only applies to kind='spaghetti'.",
            UserWarning,
            stacklevel=2,
        )

    if kind == "ribbon":
        return _plot_ribbon(x, Y, ax, plot_hdi_kwargs, ci_prob, label, ci_kind)
    elif kind == "histogram":
        return _plot_histogram(x, Y, ax, plot_hdi_kwargs, label)
    elif kind == "spaghetti":
        return _plot_spaghetti(x, Y, ax, plot_hdi_kwargs, num_samples, label)
    else:
        raise ValueError(
            f"Unknown kind: {kind}. Must be 'ribbon', 'histogram', or 'spaghetti'."
        )


def _equal_tailed_interval(
    Y: xr.DataArray, prob: float
) -> tuple[xr.DataArray, xr.DataArray]:
    """Equal-tailed interval using posterior quantiles (no arviz_stats dependency)."""
    q_lo = (1.0 - prob) / 2.0
    q_hi = 1.0 - q_lo
    stacked = Y.stack(sample=("chain", "draw"))
    lower = stacked.quantile(q_lo, dim="sample", skipna=True)
    upper = stacked.quantile(q_hi, dim="sample", skipna=True)
    return lower, upper


def _plot_ribbon(
    x: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
    Y: xr.DataArray,
    ax: plt.Axes,
    plot_hdi_kwargs: dict[str, Any] | None,
    ci_prob: float,
    label: str | None,
    ci_kind: Literal["hdi", "eti"],
) -> tuple[Line2D, PolyCollection]:
    """Plot ribbon visualization with HDI or ETI intervals."""
    if plot_hdi_kwargs is None:
        plot_hdi_kwargs = {}

    # Separate fill_kwargs for az.plot_hdi, as ax.plot doesn't accept them
    line_kwargs = plot_hdi_kwargs.copy()
    if "fill_kwargs" in line_kwargs:
        del line_kwargs["fill_kwargs"]

    # Plot mean line
    (h_line,) = ax.plot(
        x,
        Y.mean(dim=["chain", "draw"]),
        ls="-",
        **line_kwargs,
        label=label,
    )

    # Plot interval ribbon
    if ci_kind == "hdi":
        # Use ArviZ's plot_hdi for HDI
        ax_hdi = az.plot_hdi(
            x,
            Y,
            hdi_prob=ci_prob,
            ax=ax,
            smooth=False,
            **plot_hdi_kwargs,
        )
    else:  # ci_kind == "eti"
        lower, upper = _equal_tailed_interval(Y, ci_prob)
        lower_vals = np.asarray(lower.values, dtype=float).ravel()
        upper_vals = np.asarray(upper.values, dtype=float).ravel()
        n_x = len(np.asarray(x))
        if lower_vals.size != n_x or upper_vals.size != n_x:
            msg = (
                "ETI ribbon: length mismatch between x and interval bounds "
                f"(x={n_x}, lower={lower_vals.size}, upper={upper_vals.size})."
            )
            raise ValueError(msg)

        # Extract fill_kwargs if provided
        fill_kwargs = plot_hdi_kwargs.get("fill_kwargs", {})
        line_color = plot_hdi_kwargs.get("color", "C0")
        fill_color = fill_kwargs.get("color", line_color)
        fill_alpha = fill_kwargs.get("alpha", 0.3)

        ax.fill_between(
            x,
            lower_vals,
            upper_vals,
            color=fill_color,
            alpha=fill_alpha,
            **{k: v for k, v in fill_kwargs.items() if k not in ["color", "alpha"]},
        )
        ax_hdi = ax

    # Return handle to patch. We get a list of the children of the axis. Filter for just
    # the PolyCollection objects. Take the last one.
    if ci_kind == "hdi":
        h_patch = list(
            filter(lambda x: isinstance(x, PolyCollection), ax_hdi.get_children())
        )[-1]
    else:  # ci_kind == "eti"
        # For ETI, we used fill_between which creates a PolyCollection
        # Get the last PolyCollection from the axes
        h_patch = (
            list(
                filter(lambda x: isinstance(x, PolyCollection), ax_hdi.get_children())
            )[-1]
            if any(isinstance(x, PolyCollection) for x in ax_hdi.get_children())
            else None
        )
    return (h_line, h_patch)


def _x_as_numeric_mesh(
    x: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
) -> tuple[np.ndarray, bool]:
    """Convert x to floats for pcolormesh edges; return (values, is_datetime)."""
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


def _plot_histogram(
    x: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
    Y: xr.DataArray,
    ax: plt.Axes,
    plot_hdi_kwargs: dict[str, Any] | None,
    label: str | None,
) -> tuple[list[Line2D], None]:
    """Plot histogram visualization of the posterior as a 2D heatmap.

    Columns are time points (x), rows are y-value bins; cell values are
    per-time histogram counts, column-normalized for display. The posterior
    mean line is overlaid on top.
    """
    if plot_hdi_kwargs is None:
        plot_hdi_kwargs = {}

    Y_flat = Y.stack(sample=("chain", "draw"))
    time_dims = [d for d in Y.dims if d not in ("chain", "draw")]
    if len(time_dims) != 1:
        msg = (
            "plot_xY histogram expects Y with exactly one non-chain/draw dimension; "
            f"got {time_dims!r}"
        )
        raise ValueError(msg)
    time_dim = time_dims[0]
    n_time = Y.sizes[time_dim]
    n_x = len(np.asarray(x))
    if n_x != n_time:
        msg = f"Length of x ({n_x}) != length of time dimension {time_dim!r} ({n_time})"
        raise ValueError(msg)

    y_min = float(np.nanmin(Y_flat.values))
    y_max = float(np.nanmax(Y_flat.values))
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    y_edges = np.linspace(y_min - y_pad, y_max + y_pad, 51)
    n_bins = len(y_edges) - 1

    hist2d = np.zeros((n_bins, n_time), dtype=float)
    for t in range(n_time):
        col = Y_flat.isel({time_dim: t}).values.ravel()
        counts, _ = np.histogram(col, bins=y_edges)
        hist2d[:, t] = counts

    col_max = hist2d.max(axis=0, keepdims=True)
    hist2d_norm = np.divide(
        hist2d,
        col_max + 1e-12,
        out=np.zeros_like(hist2d, dtype=float),
        where=col_max > 0,
    )

    x_num, is_dt = _x_as_numeric_mesh(x)
    if len(x_num) == 1:
        x_edges = np.array([x_num[0] - 0.5, x_num[0] + 0.5])
    else:
        dx = np.diff(x_num)
        x_edges = np.zeros(len(x_num) + 1)
        x_edges[0] = x_num[0] - dx[0] / 2
        x_edges[-1] = x_num[-1] + dx[-1] / 2
        x_edges[1:-1] = x_num[:-1] + dx / 2

    cmap = plot_hdi_kwargs.get("cmap", "viridis")
    alpha = float(plot_hdi_kwargs.get("alpha", 0.85))
    color_line = plot_hdi_kwargs.get("color", "C0")

    ax.pcolormesh(
        x_edges,
        y_edges,
        hist2d_norm,
        cmap=cmap,
        shading="flat",
        alpha=alpha,
    )
    if is_dt:
        ax.xaxis_date()

    mean_y = Y.mean(dim=["chain", "draw"])
    mean_vals = np.asarray(mean_y.values, dtype=float).ravel()
    if mean_vals.size != n_time:
        msg = f"Mean line length {mean_vals.size} != n_time {n_time}"
        raise ValueError(msg)

    (mean_line,) = ax.plot(
        x_num,
        mean_vals,
        ls="-",
        color=color_line,
        label=label if label else "Posterior mean",
    )
    return ([mean_line], None)


def _plot_spaghetti(
    x: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
    Y: xr.DataArray,
    ax: plt.Axes,
    plot_hdi_kwargs: dict[str, Any] | None,
    num_samples: int,
    label: str | None,
) -> tuple[list[Line2D], None]:
    """Plot spaghetti plot with random posterior samples."""
    if plot_hdi_kwargs is None:
        plot_hdi_kwargs = {}

    # Flatten posterior samples across chains and draws
    Y_flat = Y.stack(sample=("chain", "draw"))
    n_samples_total = Y_flat.sizes["sample"]

    # Randomly select samples
    n_draw = min(num_samples, n_samples_total)
    rng = np.random.default_rng(seed=42)
    sample_indices = rng.choice(n_samples_total, size=n_draw, replace=False)

    # Plot each selected sample as a line
    handles = []
    color = plot_hdi_kwargs.get("color", "C0")
    alpha = plot_hdi_kwargs.get("alpha", 0.1)

    for idx in sample_indices:
        sample_data = Y_flat.isel(sample=idx)
        h = ax.plot(
            x,
            sample_data.values,
            color=color,
            alpha=alpha,
            linewidth=0.5,
            label=label if idx == sample_indices[0] else None,
        )
        handles.extend(h)

    # Plot mean line on top
    mean_line = ax.plot(
        x,
        Y.mean(dim=["chain", "draw"]),
        ls="-",
        color=plot_hdi_kwargs.get("color", "C0"),
        linewidth=2,
        label="Posterior mean",
    )
    handles.extend(mean_line)

    return (handles, None)


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

    # Get the data variable name (typically 'mu' or 'x')
    # We select only the data variable column to exclude coordinates like 'treated_units'
    data_var = list(hdi_result.data_vars)[0]

    # Convert to DataFrame, select only the data variable column, then unstack
    # This prevents coordinate values (like 'treated_agg') from appearing as columns
    hdi_df = hdi_result[data_var].to_dataframe()[[data_var]].unstack(level="hdi")

    # Remove the top level of column MultiIndex to get just 'lower' and 'higher'
    hdi_df.columns = hdi_df.columns.droplevel(0)

    return hdi_df
