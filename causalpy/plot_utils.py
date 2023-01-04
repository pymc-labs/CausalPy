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
    """Utility function to plot HDI intervals."""

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
    # Return handle to patch. We get a list of the childen of the axis. Filter for just
    # the PolyCollection objects. Take the last one.
    h_patch = list(
        filter(lambda x: isinstance(x, PolyCollection), ax_hdi.get_children())
    )[-1]
    return (h_line, h_patch)
