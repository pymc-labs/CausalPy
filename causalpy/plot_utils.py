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

import logging
from functools import lru_cache
from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from pandas.api.extensions import ExtensionArray

# Type alias for response type parameter
ResponseType = Literal["expectation", "prediction"]

# Module-level logger
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _log_response_type_info_once() -> None:
    """Log response type information for plots once per session.

    This function uses lru_cache to ensure the message is only logged once,
    regardless of how many times plot() is called.
    """
    logger.info(
        "Plot intervals use response_type='expectation' by default (model mean, excluding "
        "observation noise). For full predictive uncertainty including observation "
        "noise, use response_type='prediction'. To annotate plots with this information, "
        "use show_hdi_annotation=True."
    )


@lru_cache(maxsize=1)
def _log_response_type_effect_summary_once() -> None:
    """Log response type information for effect_summary once per session.

    This function uses lru_cache to ensure the message is only logged once,
    regardless of how many times effect_summary() is called.
    """
    logger.info(
        "Effect size intervals use response_type='expectation' by default (model mean, excluding "
        "observation noise). For full predictive uncertainty including observation "
        "noise, use response_type='prediction'."
    )


def add_hdi_annotation(
    ax: plt.Axes,
    response_type: ResponseType,
    hdi_prob: float = 0.94,
) -> None:
    """Add HDI type information to an axes title.

    This function appends a line to the existing title of the given axes
    to indicate whether the HDI (Highest Density Interval) represents:
    - Model expectation (μ): excludes observation noise
    - Posterior predictive (ŷ): includes observation noise

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axes whose title should be updated.
    response_type : {"expectation", "prediction"}
        The response type used for the HDI:
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
    >>> ax.set_title("My Plot")
    Text(...)
    >>> add_hdi_annotation(ax, "expectation")  # doctest: +SKIP
    """
    hdi_pct = int(hdi_prob * 100)

    if response_type == "expectation":
        annotation = (
            f"Shaded: {hdi_pct}% HDI of model expectation (μ), excl. observation noise"
        )
    else:
        annotation = (
            f"Shaded: {hdi_pct}% HDI of posterior predictive (ŷ), "
            "incl. observation noise"
        )

    # Get existing title and append annotation
    current_title = ax.get_title()
    new_title = f"{current_title}\n{annotation}" if current_title else annotation
    ax.set_title(new_title)


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
