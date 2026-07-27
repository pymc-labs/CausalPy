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
Dataframe-agnostic input handling.

CausalPy accepts any eager dataframe that `Narwhals
<https://narwhals-dev.github.io/narwhals/>`_ supports (pandas, Polars,
PyArrow, Modin, cuDF, and others) at its experiment boundaries. Inputs are
converted to pandas immediately, because the modelling pipeline (patsy,
statsmodels, scikit-learn, PyMC, and the plotting code) works on pandas and
NumPy objects. Everything CausalPy returns is therefore still pandas-backed.

Pandas inputs retain their index. Dataframes from other libraries have no
index concept, so conversion produces a default ``RangeIndex``.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import pandas as pd
from narwhals.typing import IntoDataFrame

__all__ = ["DataFrameLike", "to_pandas"]

type DataFrameLike = IntoDataFrame
"""Any eager dataframe Narwhals supports: pandas, Polars, PyArrow, and others.

Use this alias to annotate arguments that CausalPy normalizes with
:func:`to_pandas`. Lazy frames are not included, because the modelling
pipeline needs materialized data.
"""


def to_pandas(
    data: DataFrameLike | Any, *, argument_name: str = "data"
) -> pd.DataFrame:
    """Convert a dataframe-like input to a pandas ``DataFrame``.

    Pandas inputs are copied so that later index renaming does not mutate the
    caller's dataframe. Inputs from other dataframe libraries are converted
    through Narwhals and end up with a default ``RangeIndex``.

    Parameters
    ----------
    data : dataframe-like
        Any eager dataframe supported by Narwhals, such as a pandas, Polars,
        or PyArrow table.
    argument_name : str, default "data"
        Name of the calling argument, used in the error message.

    Returns
    -------
    pandas.DataFrame
        A pandas dataframe that the caller owns and may modify in place.

    Raises
    ------
    TypeError
        If the input is not a dataframe that Narwhals can convert eagerly.

    Examples
    --------
    >>> import pandas as pd
    >>> from causalpy.input_data import to_pandas
    >>> to_pandas(pd.DataFrame({"y": [1, 2]}))
       y
    0  1
    1  2
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()

    try:
        return nw.from_native(data, eager_only=True, pass_through=False).to_pandas()
    except TypeError as error:
        raise TypeError(
            f"`{argument_name}` must be a dataframe supported by Narwhals "
            "(pandas, Polars, PyArrow, and others). "
            f"Got {type(data).__name__}."
        ) from error
