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

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from pandas.api.extensions import ExtensionArray

from causalpy._arviz_compat import hdi, hdi_bound_arrays, hdi_bounds
from causalpy.constants import HDI_PROB

# Re-exported for existing importers; the canonical home is causalpy.utils so
# non-plotting modules (e.g. reporting) can use it without importing plotting.
from causalpy.utils import has_posterior_draws as has_posterior_draws
from causalpy.utils import round_num


def extract_r2_score(
    score: pd.Series, unit_index: int = 0
) -> tuple[float, float | None]:
    """Extract ``(r2, r2_std)`` from the canonical score container.

    Every backend returns a :class:`pandas.Series` with ``unit_{i}_r2``
    entries. ``r2_std`` is ``None`` when the container carries no posterior
    dispersion.

    Parameters
    ----------
    score : pd.Series
        Canonical score container as stored on ``experiment.score``.
    unit_index : int, optional
        Index of the treated unit whose score to extract. Defaults to 0.
    """
    key = f"unit_{unit_index}_r2"
    if key not in score.index:
        raise ValueError(
            f"Score container is missing required {key!r}; "
            "expected one unit_{i}_r2 entry per treated unit, "
            f"got {list(score.index)}."
        )
    std = score.get(f"{key}_std")
    return float(score[key]), None if std is None else float(std)


def format_r2_score(
    score: pd.Series,
    *,
    unit_index: int = 0,
    round_to: int | None = 2,
    context: str = "",
) -> str:
    """Format a canonical score for an experiment plot title.

    Parameters
    ----------
    score : pd.Series
        Canonical score container.
    unit_index : int, default 0
        Index of the treated unit whose score to format.
    round_to : int, optional
        Number of significant figures. Defaults to 2.
    context : str, default ""
        Text appended to the :math:`R^2` label, such as ``"on fit data"``.
    """
    r2, r2_std = extract_r2_score(score, unit_index)
    label = "Bayesian $R^2$" if r2_std is not None else "$R^2$"
    title = f"{label}{f' {context}' if context else ''} = {round_num(r2, round_to)}"
    if r2_std is not None:
        title += f" (std = {round_num(r2_std, round_to)})"
    return title


class _PosteriorPlotStyle(TypedDict):
    """Typed kwargs bundle forwarded from experiment ``_plot`` methods to every ``plot_posterior_over_x`` call."""

    ci_prob: float
    kind: Literal["ribbon", "histogram", "spaghetti"]
    ci_kind: Literal["hdi", "eti"]
    num_samples: int


def plot_posterior_over_x(
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
    """Plot a posterior :class:`xarray.DataArray` along an x-axis.

    Dispatches on ``kind`` to render ribbon (mean + interval band), spaghetti
    (posterior draw lines), or histogram (2D density heatmap) visualizations.

    Parameters
    ----------
    x : pd.DatetimeIndex, np.ndarray, pd.Index, pd.Series, or ExtensionArray
        Values for the x-axis (e.g. time, a running variable, or a prediction
        grid). Need not be temporal.
    Y : xr.DataArray
        Posterior samples with ``chain`` and ``draw`` dimensions and one
        dimension aligned with ``x``.
    ax : plt.Axes
        Matplotlib axes object.
    plot_hdi_kwargs : dict, optional
        Keyword arguments for local Matplotlib line, band, heatmap, or sample
        styling. Ribbon bands accept nested ``fill_kwargs``.
    ci_prob : float, optional
        Credible interval width when ``kind="ribbon"``. Defaults to
        :data:`~causalpy.constants.HDI_PROB` (currently 0.94). Ignored for
        other kinds.
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
        that builds legends from ``plot_posterior_over_x`` return values should only assume
        the ribbon shape when it passes ``kind="ribbon"`` (the default) through
        to :func:`plot_posterior_over_x`.
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


def _interval_bound_values(
    x: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
    Y: xr.DataArray,
    *,
    ci_prob: float,
    ci_kind: Literal["hdi", "eti"],
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated posterior interval bounds aligned one-to-one with ``x``."""
    if ci_kind == "hdi":
        lower, upper = hdi_bound_arrays(
            Y,
            prob=ci_prob,
            dim=["chain", "draw"],
        )
    else:
        eti_lower, eti_upper = _equal_tailed_interval(Y, ci_prob)
        preserved_dims = list(eti_lower.dims)
        if len(preserved_dims) != 1:
            msg = (
                "Vector ETI bounds require exactly one preserved dimension; "
                f"got {preserved_dims!r}"
            )
            raise ValueError(msg)
        lower, upper = eti_lower.to_numpy(), eti_upper.to_numpy()

    lower_vals = np.asarray(lower, dtype=float).ravel()
    upper_vals = np.asarray(upper, dtype=float).ravel()
    n_x = len(np.asarray(x))
    if lower_vals.size != n_x or upper_vals.size != n_x:
        msg = (
            f"{ci_kind.upper()} ribbon: length mismatch between x and interval bounds "
            f"(x={n_x}, lower={lower_vals.size}, upper={upper_vals.size})."
        )
        raise ValueError(msg)
    return lower_vals, upper_vals


def _plot_interval_band(
    x: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
    Y: xr.DataArray,
    ax: plt.Axes,
    *,
    ci_prob: float,
    ci_kind: Literal["hdi", "eti"],
    plot_hdi_kwargs: dict[str, Any],
    interval_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> PolyCollection:
    """Draw a validated posterior interval band without a summary line."""
    if interval_bounds is None:
        lower_vals, upper_vals = _interval_bound_values(
            x,
            Y,
            ci_prob=ci_prob,
            ci_kind=ci_kind,
        )
    else:
        lower_vals, upper_vals = interval_bounds

    fill_kwargs = plot_hdi_kwargs.get("fill_kwargs", {})
    line_color = plot_hdi_kwargs.get("color", "C0")
    fill_color = fill_kwargs.get("color", line_color)
    fill_alpha = fill_kwargs.get("alpha", 0.3)
    return ax.fill_between(
        x,
        lower_vals,
        upper_vals,
        color=fill_color,
        alpha=fill_alpha,
        **{
            key: value
            for key, value in fill_kwargs.items()
            if key not in ["color", "alpha"]
        },
    )


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

    line_kwargs = plot_hdi_kwargs.copy()
    line_kwargs.pop("fill_kwargs", None)

    interval_bounds = _interval_bound_values(
        x,
        Y,
        ci_prob=ci_prob,
        ci_kind=ci_kind,
    )
    (h_line,) = ax.plot(
        x,
        Y.mean(dim=["chain", "draw"]),
        ls="-",
        **line_kwargs,
        label=label,
    )
    band_kwargs = {**plot_hdi_kwargs, "color": h_line.get_color()}
    h_patch = _plot_interval_band(
        x,
        Y,
        ax,
        ci_prob=ci_prob,
        ci_kind=ci_kind,
        plot_hdi_kwargs=band_kwargs,
        interval_bounds=interval_bounds,
    )
    return h_line, h_patch


def plot_scalar_posterior(
    draws: xr.DataArray,
    *,
    ax: plt.Axes,
    ci_prob: float = HDI_PROB,
    ref_val: float | None = 0,
    round_to: int | None = None,
) -> plt.Axes:
    """Draw a scalar posterior density, explicit HDI, and optional reference value.

    Parameters
    ----------
    draws : xr.DataArray
        Scalar posterior draws with ``chain`` and ``draw`` dimensions. Singleton
        non-sample dimensions are squeezed before plotting.
    ax : plt.Axes
        Matplotlib axes to draw on.
    ci_prob : float, default=HDI_PROB
        Probability mass of the explicit highest density interval.
    ref_val : float, optional
        Reference value shown as a vertical dashed line. Defaults to 0.
    round_to : int, optional
        Minimum significant figures used in the summary annotation.

    Returns
    -------
    plt.Axes
        The axes supplied by the caller after drawing.

    Raises
    ------
    ValueError
        If the posterior is non-scalar after squeezing or has no finite draws.
    """
    if {"chain", "draw"} - set(draws.dims):
        msg = "Scalar posterior plotting requires both chain and draw dimensions."
        raise ValueError(msg)
    squeeze_dims = [
        dim
        for dim in draws.dims
        if dim not in {"chain", "draw"} and draws.sizes[dim] == 1
    ]
    posterior = draws.squeeze(dim=squeeze_dims, drop=True) if squeeze_dims else draws
    non_sample_dims = [dim for dim in posterior.dims if dim not in {"chain", "draw"}]
    if non_sample_dims:
        msg = (
            "Scalar posterior plotting requires only chain and draw dimensions after "
            f"squeezing singleton dimensions; got {non_sample_dims!r}."
        )
        raise ValueError(msg)

    flat = np.asarray(
        posterior.stack(sample=("chain", "draw")).values,
        dtype=float,
    ).ravel()
    finite_draws = flat[np.isfinite(flat)]
    if finite_draws.size == 0:
        msg = "Scalar posterior plotting requires at least one finite posterior draw."
        raise ValueError(msg)

    lower, upper = hdi_bounds(finite_draws, prob=ci_prob)
    mean = float(finite_draws.mean())
    ax.hist(finite_draws, bins="auto", density=True, color="C0", alpha=0.5)
    ax.plot(
        [lower, upper],
        [0, 0],
        color="C0",
        linewidth=4,
        solid_capstyle="butt",
        label=f"{ci_prob:.0%} HDI",
    )
    if ref_val is not None:
        ax.axvline(ref_val, color="black", linestyle="--", label="Reference value")
    ax.text(
        0.98,
        0.98,
        (
            f"Mean: {round_num(mean, round_to)}\n"
            f"{ci_prob:.0%} HDI "
            f"[{round_num(lower, round_to)}, {round_num(upper, round_to)}]"
        ),
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
    )
    return ax


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

    Columns are positions along ``x``; rows are y-value bins. Cell values are
    per-column histogram counts, column-normalized for display. The posterior
    mean line is overlaid on top.
    """
    if plot_hdi_kwargs is None:
        plot_hdi_kwargs = {}

    Y_flat = Y.stack(sample=("chain", "draw"))
    time_dims = [d for d in Y.dims if d not in ("chain", "draw")]
    if len(time_dims) != 1:
        msg = (
            "plot_posterior_over_x histogram expects Y with exactly one non-chain/draw dimension; "
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
    hdi_result = hdi(x, prob=hdi_prob)
    # Drop non-dimension coordinates (e.g. scalar treated_units) so they do not
    # become DataFrame columns after unstack — regression for #532.
    hdi_result = hdi_result.reset_coords(drop=True)
    if hdi_result.dims == ("hdi",):
        return pd.DataFrame(
            [hdi_result.sel(hdi=["lower", "higher"]).values],
            columns=["lower", "higher"],
        )

    lower = hdi_result.sel(hdi="lower")
    if any(lower.sizes[dim] == 0 for dim in lower.dims):
        return pd.DataFrame(
            index=lower.to_dataframe(name="lower").index,
            columns=["lower", "higher"],
            dtype=float,
        )

    hdi_df = hdi_result.to_dataframe(name="hdi_value")[["hdi_value"]].unstack(
        level="hdi"
    )
    hdi_df.columns = hdi_df.columns.droplevel(0)
    hdi_df.columns.name = None
    # Force deterministic column order for SC iloc[:, [0, -1]] consumers
    return hdi_df[["lower", "higher"]]
