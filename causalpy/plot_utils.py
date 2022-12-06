from typing import Any, Dict, Optional, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def plot_xY(
    x: Union[pd.DatetimeIndex, np.array],
    Y: xr.DataArray,
    ax: plt.Axes,
    plot_hdi_kwargs: Optional[Dict[str, Any]] = {},
    hdi_prob: Optional[float] = 0.94,
    include_label: Optional[bool] = True,
) -> None:
    """Utility function to plot HDI intervals."""

    az.plot_hdi(
        x,
        Y,
        hdi_prob=hdi_prob,
        fill_kwargs={
            "alpha": 0.25,
            "label": f"{hdi_prob*100}% HDI" if include_label else None,
        },
        smooth=False,
        ax=ax,
        **plot_hdi_kwargs,
    )
    ax.plot(
        x,
        Y.mean(dim=["chain", "draw"]),
        color="k",
        label="Posterior mean" if include_label else None,
    )
