#   Copyright 2025 - 2026 The PyMC Labs Developers
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
Utility functions for date axis formatting and datetime index validation.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from causalpy.custom_exceptions import BadIndexException


def validate_treatment_time_against_index(
    index: pd.Index,
    treatment_time: int | float | pd.Timestamp,
    *,
    name: str = "treatment_time",
) -> None:
    """Check that a treatment boundary can be compared against ``data.index``.

    Experiments slice their data with comparisons such as
    ``data.index < treatment_time``. That is only well defined when the boundary
    is missing-free, of the same broad kind as the index (datetime vs numeric),
    and — for :class:`pandas.DatetimeIndex` — matches the index's
    timezone-awareness. Timezone-aware values in *different* zones compare
    correctly in pandas, so only a naive/aware mismatch is rejected.

    Parameters
    ----------
    index : pd.Index
        The index the boundary will be compared against.
    treatment_time : int, float, or pd.Timestamp
        The treatment boundary to validate.
    name : str, default "treatment_time"
        Name used in error messages, e.g. ``"treatment_end_time"``.

    Raises
    ------
    BadIndexException
        If ``treatment_time`` cannot be compared against ``index``.
    """
    if pd.isna(treatment_time):
        raise BadIndexException(f"{name} must not be missing.")

    if isinstance(index, pd.DatetimeIndex):
        if not isinstance(treatment_time, pd.Timestamp):
            raise BadIndexException(
                f"If data.index is DatetimeIndex, {name} must be pd.Timestamp."
            )
        if (index.tz is None) != (treatment_time.tz is None):
            raise BadIndexException(
                f"{name} and data.index must either both be timezone-aware or "
                "both be timezone-naive."
            )
    elif isinstance(treatment_time, pd.Timestamp):
        raise BadIndexException(
            f"If data.index is not DatetimeIndex, {name} must not be pd.Timestamp."
        )


def _combine_datetime_indices(
    index1: pd.DatetimeIndex, index2: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """
    Combine two DatetimeIndex objects into a single sorted DatetimeIndex.

    Parameters
    ----------
    index1 : pd.DatetimeIndex
        First datetime index
    index2 : pd.DatetimeIndex
        Second datetime index

    Returns
    -------
    pd.DatetimeIndex
        Combined and sorted datetime index
    """
    return pd.DatetimeIndex(index1.tolist() + index2.tolist()).sort_values()


def format_date_axis(
    ax: plt.Axes,
    date_index: pd.DatetimeIndex,
    maxticks: int = 8,
) -> None:
    """
    Apply intelligent date formatting to x-axis using AutoDateLocator.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to format
    date_index : pd.DatetimeIndex
        The datetime index being plotted on the x-axis
    maxticks : int
        Maximum number of ticks to display (default 8)
    """
    locator = mdates.AutoDateLocator(minticks=3, maxticks=maxticks)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Rotate labels: vertical (-90) for long series, horizontal (0) otherwise
    if len(date_index) > 1:
        date_span = date_index.max() - date_index.min()
        num_years = date_span.days / 365.25
        if num_years > 3:
            ax.tick_params(axis="x", labelrotation=-90)
        else:
            ax.tick_params(axis="x", labelrotation=0)


def format_date_axes(axes: list[plt.Axes], date_index: pd.DatetimeIndex) -> None:
    """
    Apply intelligent date formatting to multiple axes with shared x-axis.

    Parameters
    ----------
    axes : list of plt.Axes
        List of matplotlib axes objects to format
    date_index : pd.DatetimeIndex
        The datetime index being plotted on the x-axis
    """
    if len(axes) == 0:
        return

    # Apply formatting to the bottom-most axis
    format_date_axis(axes[-1], date_index)
