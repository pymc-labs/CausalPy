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
"""Tests for date formatting utilities"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causalpy.date_utils import (
    _combine_datetime_indices,
    format_date_axes,
    format_date_axis,
)


class TestCombineDatetimeIndices:
    """Tests for _combine_datetime_indices helper function"""

    def test_combines_and_sorts_indices(self):
        """Test that indices are combined and sorted correctly"""
        index1 = pd.DatetimeIndex(["2020-03-01", "2020-01-01", "2020-02-01"])
        index2 = pd.DatetimeIndex(["2020-06-01", "2020-04-01", "2020-05-01"])

        result = _combine_datetime_indices(index1, index2)

        expected = pd.DatetimeIndex(
            [
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
                "2020-04-01",
                "2020-05-01",
                "2020-06-01",
            ]
        )
        assert result.equals(expected)

    def test_handles_empty_indices(self):
        """Test that empty indices are handled correctly"""
        index1 = pd.DatetimeIndex([])
        index2 = pd.DatetimeIndex(["2020-01-01", "2020-02-01"])

        result = _combine_datetime_indices(index1, index2)

        assert len(result) == 2
        assert result.equals(pd.DatetimeIndex(["2020-01-01", "2020-02-01"]))


class TestFormatDateAxis:
    """Tests for format_date_axis function"""

    def test_uses_auto_date_locator(self):
        """Test that AutoDateLocator is used for automatic tick positioning"""
        dates = pd.date_range("2010-01-01", "2020-01-01", freq="ME")
        fig, ax = plt.subplots()
        ax.plot(dates, np.random.randn(len(dates)))

        format_date_axis(ax, dates)

        # Check that major locator is AutoDateLocator
        assert isinstance(ax.xaxis.get_major_locator(), mdates.AutoDateLocator)
        plt.close(fig)

    def test_uses_concise_date_formatter(self):
        """Test that ConciseDateFormatter is used for concise labels"""
        dates = pd.date_range("2020-01-01", "2022-01-01", freq="ME")
        fig, ax = plt.subplots()
        ax.plot(dates, np.random.randn(len(dates)))

        format_date_axis(ax, dates)

        # Check that major formatter is ConciseDateFormatter
        assert isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)
        plt.close(fig)

    def test_long_date_range_rotates_labels_vertically(self):
        """Test that long date ranges (>3 years) use vertical rotation (-90)"""
        dates = pd.date_range("2010-01-01", "2020-01-01", freq="ME")
        fig, ax = plt.subplots()
        ax.plot(dates, np.random.randn(len(dates)))

        format_date_axis(ax, dates)

        # Check rotation is -90 (or 270) for long series
        rotation = ax.xaxis.get_ticklabels()[0].get_rotation()
        assert rotation in (-90, 270.0)
        plt.close(fig)

    def test_short_date_range_keeps_labels_horizontal(self):
        """Test that short date ranges (<=3 years) keep labels horizontal (0)"""
        dates = pd.date_range("2020-01-01", "2022-01-01", freq="ME")
        fig, ax = plt.subplots()
        ax.plot(dates, np.random.randn(len(dates)))

        format_date_axis(ax, dates)

        # Check rotation is 0 for short series
        rotation = ax.xaxis.get_ticklabels()[0].get_rotation()
        assert rotation == 0
        plt.close(fig)

    def test_empty_date_index_does_not_error(self):
        """Test that empty date index doesn't cause errors"""
        dates = pd.DatetimeIndex([])
        fig, ax = plt.subplots()

        # Should not raise an error
        format_date_axis(ax, dates)
        plt.close(fig)

    def test_single_date_does_not_error(self):
        """Test that single date doesn't cause errors"""
        dates = pd.DatetimeIndex(["2023-01-01"])
        fig, ax = plt.subplots()
        ax.plot(dates, [1])

        # Should not raise an error
        format_date_axis(ax, dates)
        plt.close(fig)

    def test_maxticks_parameter_is_accepted(self):
        """Test that maxticks parameter is accepted without error"""
        dates = pd.date_range("2020-01-01", "2022-01-01", freq="ME")
        fig, ax = plt.subplots()
        ax.plot(dates, np.random.randn(len(dates)))

        # Should not raise an error with custom maxticks
        format_date_axis(ax, dates, maxticks=5)

        # Verify AutoDateLocator is still used
        assert isinstance(ax.xaxis.get_major_locator(), mdates.AutoDateLocator)
        plt.close(fig)


class TestFormatDateAxes:
    """Tests for format_date_axes function"""

    def test_formats_multiple_axes(self):
        """Test that format_date_axes works with multiple axes"""
        dates = pd.date_range("2020-01-01", "2022-01-01", freq="ME")
        fig, axes = plt.subplots(3, 1, sharex=True)

        for ax in axes:
            ax.plot(dates, np.random.randn(len(dates)))

        format_date_axes(axes, dates)

        # Check that bottom axis has AutoDateLocator
        assert isinstance(axes[-1].xaxis.get_major_locator(), mdates.AutoDateLocator)
        plt.close(fig)

    def test_empty_axes_list_does_not_error(self):
        """Test that empty axes list doesn't cause errors"""
        dates = pd.date_range("2020-01-01", "2022-01-01", freq="ME")

        # Should not raise an error
        format_date_axes([], dates)

    def test_formats_only_bottom_axis(self):
        """Test that only the bottom axis is formatted (to avoid duplicate labels)"""
        dates = pd.date_range("2020-01-01", "2022-01-01", freq="ME")
        fig, axes = plt.subplots(2, 1, sharex=True)

        for ax in axes:
            ax.plot(dates, np.random.randn(len(dates)))

        format_date_axes(axes, dates)

        # Bottom axis should have AutoDateLocator
        assert isinstance(axes[-1].xaxis.get_major_locator(), mdates.AutoDateLocator)

        plt.close(fig)
