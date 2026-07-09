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

import arviz as az
import pandas as pd
import xarray as xr

from causalpy.constants import HDI_PROB


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
