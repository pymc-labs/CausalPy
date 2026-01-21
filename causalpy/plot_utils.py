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

from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from arviz_stats import eti
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from pandas.api.extensions import ExtensionArray


def plot_xY(
    x: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
    Y: xr.DataArray,
    ax: plt.Axes,
    plot_hdi_kwargs: dict[str, Any] | None = None,
    ci_prob: float = 0.94,
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
        Dictionary of keyword arguments passed to ax.plot().
    ci_prob : float, optional
        The size of the credible interval. Default is 0.94 (matching current behavior).
    label : str, optional
        The plot label.
    kind : {"ribbon", "histogram", "spaghetti"}, optional
        Type of visualization. Default is "ribbon".
    ci_kind : {"hdi", "eti"}, optional
        Type of interval for ribbon plots. Default is "hdi" (matching current behavior).
    num_samples : int, optional
        Number of posterior samples to plot for spaghetti visualization.
        Default is 50.
    hdi_prob : float, optional
        Backward compatibility alias for `ci_prob`. If provided, overrides `ci_prob`.
        This parameter existed in the original API.

    Returns
    -------
    tuple
        Tuple of (Line2D or list[Line2D], PolyCollection or None) handles
        for the plot line(s) and interval patch (if applicable).
    """
    # Handle backward compatibility: hdi_prob was in original API
    if hdi_prob is not None:
        ci_prob = hdi_prob

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
        # Compute ETI using arviz_stats
        eti_result = eti(Y, prob=ci_prob)

        # ETI returns a DataArray with 'ci_bound' dimension containing 'lower' and 'upper'
        # Extract lower and upper bounds
        lower = eti_result.sel(ci_bound="lower")
        upper = eti_result.sel(ci_bound="upper")

        # Extract fill_kwargs if provided
        fill_kwargs = plot_hdi_kwargs.get("fill_kwargs", {})
        line_color = plot_hdi_kwargs.get("color", "C0")
        fill_color = fill_kwargs.get("color", line_color)
        fill_alpha = fill_kwargs.get("alpha", 0.3)

        # Plot ETI using fill_between
        ax.fill_between(
            x,
            lower.values,
            upper.values,
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


def _plot_histogram(
    x: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
    Y: xr.DataArray,
    ax: plt.Axes,
    plot_hdi_kwargs: dict[str, Any] | None,
    label: str | None,
) -> tuple[list[Line2D], None]:
    """Plot histogram visualization of posterior distribution."""
    if plot_hdi_kwargs is None:
        plot_hdi_kwargs = {}

    # Flatten posterior samples across chains and draws
    # Shape: [n_time_points, n_samples_total]
    Y_flat = Y.stack(sample=("chain", "draw"))

    # Select a few key time points for visualization
    n_time_points = len(x)
    if n_time_points <= 5:
        time_indices = list(range(n_time_points))
    else:
        # Select first, middle, and last time points
        time_indices = [0, n_time_points // 2, n_time_points - 1]

    handles = []
    for i, idx in enumerate(time_indices):
        # Get posterior samples for this time point
        samples = Y_flat.isel(obs_ind=idx).values

        # Create histogram
        counts, bins = np.histogram(samples, bins=30)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Normalize to make it visible on the plot
        max_count = counts.max()
        normalized_counts = (
            counts / max_count * 0.1 if max_count > 0 else counts
        )  # Scale to 10% of y-axis range

        # Plot histogram bars at the time point
        color = plot_hdi_kwargs.get("color", f"C{i}")
        alpha = plot_hdi_kwargs.get("alpha", 0.6)

        h = ax.barh(
            bin_centers,
            normalized_counts,
            left=x[idx],
            height=bins[1] - bins[0],
            color=color,
            alpha=alpha,
            label=label if i == 0 else None,
        )
        handles.extend(h)

    # Plot mean line
    mean_line = ax.plot(
        x,
        Y.mean(dim=["chain", "draw"]),
        ls="-",
        color=plot_hdi_kwargs.get("color", "C0"),
        label=label if label else "Posterior mean",
    )
    handles.extend(mean_line)

    return (handles, None)


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
    n_samples = min(num_samples, n_samples_total)
    rng = np.random.default_rng(seed=42)
    sample_indices = rng.choice(n_samples_total, size=n_samples, replace=False)

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
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """Calculate and recover HDI intervals.

    Parameters
    ----------
    x : xr.DataArray
        Xarray data array.
    hdi_prob : float, optional
        The size of the HDI. Default is 0.94.

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
