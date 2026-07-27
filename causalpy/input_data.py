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
<https://narwhals-dev.github.io/narwhals/>`_ supports at its experiment
boundaries. The test suite exercises pandas, Polars, and PyArrow; the rest of
the Narwhals-supported libraries follow from the same conversion but are not
covered here. Lazy frames are rejected, because the modelling pipeline needs
materialized data.

Inputs are converted to pandas immediately, because that pipeline (patsy,
statsmodels, scikit-learn, PyMC, and the plotting code) works on pandas and
NumPy objects.

**Outputs are unchanged.** Everything CausalPy returns is still pandas-backed:
``experiment.data``, the dataframes on :mod:`causalpy.reporting`, and the
loaders in :mod:`causalpy.data` all hand back pandas regardless of what was
passed in. This module widens what you may pass, not what you get back.

Pandas inputs retain their index. Dataframes from other libraries have no
index concept, so conversion produces a default ``RangeIndex``. Experiments
that read the index as a time axis take a ``time_column`` argument instead;
see :func:`to_pandas_with_time_index`.
"""

from __future__ import annotations

import narwhals as nw
import pandas as pd
from narwhals.typing import IntoDataFrame

from causalpy.custom_exceptions import DataException

__all__ = ["DataFrameLike", "to_pandas", "to_pandas_with_time_index"]

type DataFrameLike = IntoDataFrame
"""Any eager dataframe Narwhals supports: pandas, Polars, PyArrow, and others.

Use this alias to annotate arguments that CausalPy normalizes with
:func:`to_pandas`. Lazy frames are not included, because the modelling
pipeline needs materialized data.
"""


def to_pandas(data: DataFrameLike, *, argument_name: str = "data") -> pd.DataFrame:
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

    # Only the rejection of a non-dataframe is translated. Errors raised while
    # converting a dataframe Narwhals did accept are left alone, so a genuine
    # conversion failure is not reported as "this is not a dataframe".
    try:
        frame = nw.from_native(data, eager_only=True, pass_through=False)
    except TypeError as error:
        # A lazy frame is the confusing case: the library is supported, the
        # object is not, so say that rather than leaving the caller to guess.
        hint = (
            " Lazy frames are not accepted, because the modelling pipeline "
            "needs materialized data. Call `.collect()` first."
            if hasattr(data, "collect")
            else ""
        )
        raise TypeError(
            f"`{argument_name}` must be an eager dataframe supported by "
            "Narwhals, such as pandas, Polars, or PyArrow. "
            f"Got {type(data).__name__}.{hint}"
        ) from error
    return frame.to_pandas()


def to_pandas_with_time_index(
    data: DataFrameLike,
    time_column: str | None = None,
    *,
    argument_name: str = "data",
) -> pd.DataFrame:
    """Convert a dataframe-like input and put its time axis on the index.

    Experiments that compare observations against a ``treatment_time`` need a
    time axis. Pandas callers have historically supplied it as the dataframe
    index. Dataframes from other libraries have no index, so those callers must
    name the column that holds the time axis.

    Parameters
    ----------
    data : dataframe-like
        Any eager dataframe supported by Narwhals.
    time_column : str or None, default None
        Column holding the time axis. When given, it becomes the index. When
        None, the pandas index of ``data`` is used, which requires a pandas
        input.
    argument_name : str, default "data"
        Name of the calling argument, used in error messages.

    Returns
    -------
    pandas.DataFrame
        A pandas dataframe indexed by the time axis.

    Raises
    ------
    DataException
        If ``time_column`` is missing from the data, or if a non-pandas input
        arrives without a ``time_column``.

    Examples
    --------
    >>> import pandas as pd
    >>> from causalpy.input_data import to_pandas_with_time_index
    >>> frame = pd.DataFrame({"t": [1, 2], "y": [3, 4]})
    >>> result = to_pandas_with_time_index(frame, time_column="t")
    >>> result.index.name
    't'
    >>> result.index.tolist()
    [1, 2]
    >>> result.columns.tolist()
    ['y']
    """
    has_pandas_index = isinstance(data, pd.DataFrame)
    frame = to_pandas(data, argument_name=argument_name)

    if time_column is None:
        if not has_pandas_index:
            raise DataException(
                f"`{argument_name}` is not a pandas dataframe, so it carries no "
                "index to use as the time axis. Pass `time_column` naming the "
                "column that holds the time axis."
            )
        return frame

    if time_column not in frame.columns:
        raise DataException(
            f"`time_column` '{time_column}' is not a column of `{argument_name}`. "
            f"Available columns: {list(frame.columns)}."
        )
    return frame.set_index(time_column)
