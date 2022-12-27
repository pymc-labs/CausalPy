from typing import Any, Dict, Optional, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import PolyCollection


def plot_xY(
    x: Union[pd.DatetimeIndex, np.array],
    Y: xr.DataArray,
    ax: plt.Axes,
    plot_hdi_kwargs: Optional[Dict[str, Any]] = None,
    hdi_prob: float = 0.94,
    label: Optional[str] = "",
    include_label: bool = True,
):
    """Utility function to plot HDI intervals."""

    if plot_hdi_kwargs is None:
        plot_hdi_kwargs = {}

    (h_line,) = ax.plot(
        x,
        Y.mean(dim=["chain", "draw"]),
        ls="-",
        **plot_hdi_kwargs,
        label=f"{label}" if include_label else None,
    )
    ax_hdi = az.plot_hdi(
        x,
        Y,
        hdi_prob=hdi_prob,
        fill_kwargs={
            "alpha": 0.25,
            "label": " ",  # f"{hdi_prob*100}% HDI" if include_label else None,
        },
        smooth=False,
        ax=ax,
        **plot_hdi_kwargs,
    )
    # Return handle to patch.
    # We get a list of the childen of the axis
    # Filter for just the PolyCollection objects
    # Take the last one
    h_patch = list(
        filter(lambda x: isinstance(x, PolyCollection), ax_hdi.get_children())
    )[-1]

    # if include_label:
    #     handles, labels = ax.get_legend_handles_labels()
    #     ax.legend(
    #         handles=[(h1, h2) for h1, h2 in zip(handles[::2], handles[1::2])],
    #         # labels=[l1 + " + " + l2 for l1, l2 in zip(labels[::2], labels[1::2])],
    #         labels=[l1 for l1 in labels[::2]],
    #     )
    return h_line, h_patch
