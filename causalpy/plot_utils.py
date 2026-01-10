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
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from pandas.api.extensions import ExtensionArray

# Type alias for HDI type parameter
HdiType = Literal["expectation", "prediction"]


def add_hdi_annotation(
    fig: plt.Figure,
    hdi_type: HdiType,
    hdi_prob: float = 0.94,
) -> None:
    """Add a text annotation to a figure explaining what the HDI represents.

    This function adds small text at the bottom of the figure to indicate
    whether the HDI (Highest Density Interval) represents:
    - Model expectation (μ): excludes observation noise
    - Posterior predictive (ŷ): includes observation noise

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure to annotate.
    hdi_type : {"expectation", "prediction"}
        The type of HDI being displayed:
        - "expectation": HDI of the model expectation (μ), which excludes
          observation noise. Shows uncertainty from model parameters only.
        - "prediction": HDI of the posterior predictive (ŷ), which includes
          observation noise. Shows the full predictive uncertainty.
    hdi_prob : float, optional
        The probability mass of the HDI. Default is 0.94.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> add_hdi_annotation(fig, "expectation")  # doctest: +SKIP
    >>> add_hdi_annotation(fig, "prediction", hdi_prob=0.89)  # doctest: +SKIP
    """
    hdi_pct = int(hdi_prob * 100)

    if hdi_type == "expectation":
        text = (
            f"Shaded regions show {hdi_pct}% HDI of model expectation (μ), "
            "excluding observation noise"
        )
    else:
        text = (
            f"Shaded regions show {hdi_pct}% HDI of posterior predictive (ŷ), "
            "including observation noise"
        )

    fig.text(
        0.5,
        0.01,
        text,
        ha="center",
        va="bottom",
        fontsize=8,
        fontstyle="italic",
        color="gray",
    )


def plot_xY(
    x: pd.DatetimeIndex | np.ndarray | pd.Index | pd.Series | ExtensionArray,
    Y: xr.DataArray,
    ax: plt.Axes,
    plot_hdi_kwargs: dict[str, Any] | None = None,
    hdi_prob: float = 0.94,
    label: str | None = None,
) -> tuple[Line2D, PolyCollection]:
    """Plot HDI intervals.

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
    hdi_prob : float, optional
        The size of the HDI. Default is 0.94.
    label : str, optional
        The plot label.

    Returns
    -------
    tuple
        Tuple of (Line2D, PolyCollection) handles for the plot line and
        HDI patch.
    """

    if plot_hdi_kwargs is None:
        plot_hdi_kwargs = {}

    # Separate fill_kwargs for az.plot_hdi, as ax.plot doesn't accept them
    line_kwargs = plot_hdi_kwargs.copy()
    if "fill_kwargs" in line_kwargs:
        del line_kwargs["fill_kwargs"]

    (h_line,) = ax.plot(
        x,
        Y.mean(dim=["chain", "draw"]),
        ls="-",
        **line_kwargs,  # Use kwargs without fill_kwargs
        label=label,  # Use the provided label for the mean line
    )
    ax_hdi = az.plot_hdi(
        x,
        Y,
        hdi_prob=hdi_prob,
        ax=ax,
        smooth=False,  # To prevent warning about resolution with few data points
        # Pass original plot_hdi_kwargs which might include fill_kwargs for fill_between
        **plot_hdi_kwargs,
    )
    # Return handle to patch. We get a list of the children of the axis. Filter for just
    # the PolyCollection objects. Take the last one.
    h_patch = list(
        filter(lambda x: isinstance(x, PolyCollection), ax_hdi.get_children())
    )[-1]
    return (h_line, h_patch)


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
