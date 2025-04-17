#   Copyright 2022 - 2025 The PyMC Labs Developers
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

from typing import Any, Dict, Optional, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D


def plot_xY(
    x: Union[pd.DatetimeIndex, np.array],
    Y: xr.DataArray,
    ax: plt.Axes,
    plot_hdi_kwargs: Optional[Dict[str, Any]] = None,
    hdi_prob: float = 0.94,
    label: Union[str, None] = None,
) -> Tuple[Line2D, PolyCollection]:
    """
    Utility function to plot HDI intervals.

    :param x:
        Pandas datetime index or numpy array of x-axis values
    :param y:
        Xarray data array of y-axis data
    :param ax:
        Matplotlib ax object
    :param plot_hdi_kwargs:
        Dictionary of keyword arguments passed to ax.plot()
    :param hdi_prob:
        The size of the HDI, default is 0.94
    :param label:
        The plot label
    """

    if plot_hdi_kwargs is None:
        plot_hdi_kwargs = {}

    (h_line,) = ax.plot(
        x,
        Y.mean(dim=["chain", "draw"]),
        ls="-",
        **plot_hdi_kwargs,
        label=f"{label}",
    )
    ax_hdi = az.plot_hdi(
        x,
        Y,
        hdi_prob=hdi_prob,
        fill_kwargs={
            "alpha": 0.25,
            "label": " ",
        },
        smooth=False,
        ax=ax,
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
    """
    Utility function to calculate and recover HDI intervals.

    :param x:
        Xarray data array
    :param hdi_prob:
        The size of the HDI, default is 0.94
    """
    hdi = (
        az.hdi(x, hdi_prob=hdi_prob)
        .to_dataframe()
        .unstack(level="hdi")
        .droplevel(0, axis=1)
    )
    return hdi
