#   Copyright 2025 - 2025 The PyMC Labs Developers
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
Utility functions for intelligent date axis formatting.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


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


def format_date_axis(ax: plt.Axes, date_index: pd.DatetimeIndex) -> None:
    """
    Apply intelligent date formatting to x-axis based on date range.

    This function automatically selects appropriate date formatters and locators
    based on the span of dates being plotted. It aims to:
    - Prevent overlapping x-axis labels
    - Use appropriate granularity (years, months, weeks, days)
    - Set intelligent major and minor ticks/gridlines

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to format
    date_index : pd.DatetimeIndex
        The datetime index being plotted on the x-axis

    Notes
    -----
    This function uses matplotlib's built-in date formatters and locators,
    which provide good automatic behavior for most date ranges.
    """
    if len(date_index) == 0:
        return

    # Calculate the span of dates
    date_span = date_index.max() - date_index.min()
    days_span = date_span.days

    # Strategy: Use matplotlib's AutoDateLocator and ConciseDateFormatter
    # which provide intelligent automatic date formatting
    
    # Calculate number of years for better decisions
    num_years = days_span / 365.25

    if days_span > 365 * 6:  # More than 6 years
        # Use yearly major ticks, no minor ticks (too cluttered)
        # For very long series, space out the year labels more
        if num_years > 15:
            # Every 2 years for very long series
            major_locator = mdates.YearLocator(2)
        else:
            major_locator = mdates.YearLocator()
        minor_locator = mdates.YearLocator()  # Minor at every year
        major_formatter = mdates.DateFormatter("%Y")

    elif days_span > 365:  # 1-6 years
        # Use yearly major ticks, monthly minor ticks
        major_locator = mdates.YearLocator()
        minor_locator = mdates.MonthLocator()
        major_formatter = mdates.DateFormatter("%Y")

    elif days_span > 90:  # 3-12 months
        # Use monthly major ticks
        major_locator = mdates.MonthLocator()
        minor_locator = mdates.MonthLocator(bymonthday=15)
        major_formatter = mdates.DateFormatter("%Y-%m")

    elif days_span > 30:  # 1-3 months
        # Use bi-weekly major ticks
        major_locator = mdates.WeekdayLocator(byweekday=mdates.MO, interval=2)
        minor_locator = mdates.WeekdayLocator(byweekday=mdates.MO)
        major_formatter = mdates.DateFormatter("%Y-%m-%d")

    else:  # Less than 1 month
        # Use weekly major ticks
        major_locator = mdates.WeekdayLocator(byweekday=mdates.MO)
        minor_locator = mdates.DayLocator()
        major_formatter = mdates.DateFormatter("%Y-%m-%d")

    # Apply formatters and locators
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(major_formatter)
    ax.xaxis.set_minor_locator(minor_locator)

    # Rotate labels for better readability
    # For very long series (>8 years), use vertical rotation to prevent overlap
    if num_years > 8:
        ax.tick_params(axis="x", labelrotation=-90)
    elif days_span <= 365 * 3:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Enable minor grid lines for better readability
    # For long series with years, only show minor gridlines if <= 6 years
    if num_years > 6:
        # Only major grid for very long series
        ax.grid(True, which="major", linestyle="-", alpha=0.5)
    else:
        # Both major and minor for shorter series
        ax.grid(True, which="minor", linestyle=":", alpha=0.3)
        ax.grid(True, which="major", linestyle="-", alpha=0.5)


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
    # Only format the bottom-most axis to avoid duplicate labels
    if len(axes) > 0:
        format_date_axis(axes[-1], date_index)
