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
Tests for the explicit time axis on index-based experiments.

InterruptedTimeSeries, SyntheticControl, and SyntheticDifferenceInDifferences
compare observations against a treatment time. Pandas callers supply that axis
as the dataframe index. Callers using another dataframe library have no index
and must name the time column instead.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.custom_exceptions import DataException
from causalpy.input_data import to_pandas_with_time_index

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


def to_polars_with_time(data: pd.DataFrame, time_name: str) -> pl.DataFrame:
    """Move a pandas index into a named column and convert to Polars."""
    return pl.from_pandas(data.rename_axis(time_name).reset_index())


class TestToPandasWithTimeIndex:
    """Unit tests for the shared helper."""

    def test_pandas_without_time_column_keeps_index(self):
        """A pandas input without time_column keeps its own index."""
        frame = pd.DataFrame({"y": [1, 2]}, index=pd.Index([10, 11], name="t"))
        result = to_pandas_with_time_index(frame)
        assert result.index.tolist() == [10, 11]

    def test_time_column_becomes_index(self):
        """time_column moves out of the columns and onto the index."""
        frame = pd.DataFrame({"t": [10, 11], "y": [1, 2]})
        result = to_pandas_with_time_index(frame, time_column="t")
        assert result.index.tolist() == [10, 11]
        assert "t" not in result.columns

    def test_polars_time_column_becomes_index(self):
        """A Polars input with time_column gets a real time index."""
        frame = pl.DataFrame({"t": [10, 11], "y": [1, 2]})
        result = to_pandas_with_time_index(frame, time_column="t")
        assert result.index.tolist() == [10, 11]

    def test_polars_without_time_column_raises(self):
        """A non-pandas input without time_column is rejected, not guessed."""
        frame = pl.DataFrame({"t": [10, 11], "y": [1, 2]})
        with pytest.raises(DataException, match="Pass `time_column`"):
            to_pandas_with_time_index(frame)

    def test_missing_time_column_raises(self):
        """A time_column that is not in the data is rejected."""
        frame = pl.DataFrame({"t": [10, 11], "y": [1, 2]})
        with pytest.raises(DataException, match="is not a column of"):
            to_pandas_with_time_index(frame, time_column="date")

    def test_time_column_conflicting_with_named_index_raises(self):
        """A named index plus a time_column is ambiguous, so it is refused."""
        frame = pd.DataFrame(
            {"t": [100, 200], "y": [1, 2]}, index=pd.Index([7, 8], name="old")
        )
        with pytest.raises(DataException, match="already has a meaningful index"):
            to_pandas_with_time_index(frame, time_column="t")

    def test_time_column_conflicting_with_datetime_index_raises(self):
        """A DatetimeIndex plus a time_column is refused rather than dropped."""
        frame = pd.DataFrame(
            {"t": [100, 200], "y": [1, 2]},
            index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        )
        with pytest.raises(DataException, match="already has a meaningful index"):
            to_pandas_with_time_index(frame, time_column="t")

    def test_time_column_with_default_index_is_allowed(self):
        """An unnamed RangeIndex is not meaningful, so time_column is fine."""
        frame = pd.DataFrame({"t": [100, 200], "y": [1, 2]})
        result = to_pandas_with_time_index(frame, time_column="t")
        assert result.index.tolist() == [100, 200]


class TestInterruptedTimeSeries:
    """InterruptedTimeSeries with an explicit time column."""

    def test_polars_matches_pandas(self, its_simple_data):
        """The Polars run with time_column matches the pandas run."""
        treatment_time = pd.to_datetime("2015-01-01")

        from_pandas = cp.InterruptedTimeSeries(
            its_simple_data,
            treatment_time,
            formula="timeseries ~ 1 + linear_trend",
            model=LinearRegression(),
        )
        from_polars = cp.InterruptedTimeSeries(
            to_polars_with_time(its_simple_data, "date"),
            treatment_time,
            formula="timeseries ~ 1 + linear_trend",
            model=LinearRegression(),
            time_column="date",
        )

        assert from_polars.datapre.index.equals(from_pandas.datapre.index)
        assert from_polars.datapost.index.equals(from_pandas.datapost.index)
        np.testing.assert_allclose(
            np.asarray(from_polars.post_impact), np.asarray(from_pandas.post_impact)
        )

    def test_time_column_on_pandas_input(self, its_simple_data):
        """A pandas caller can use time_column instead of setting the index."""
        treatment_time = pd.to_datetime("2015-01-01")
        flat = its_simple_data.rename_axis("date").reset_index()

        result = cp.InterruptedTimeSeries(
            flat,
            treatment_time,
            formula="timeseries ~ 1 + linear_trend",
            model=LinearRegression(),
            time_column="date",
        )
        assert isinstance(result.data.index, pd.DatetimeIndex)

    def test_polars_without_time_column_raises(self, its_simple_data):
        """Without time_column the Polars input is refused, not silently ranked."""
        with pytest.raises(DataException, match="Pass `time_column`"):
            cp.InterruptedTimeSeries(
                to_polars_with_time(its_simple_data, "date"),
                pd.to_datetime("2015-01-01"),
                formula="timeseries ~ 1 + linear_trend",
                model=LinearRegression(),
            )

    def test_time_column_with_existing_index_raises(self, its_simple_data):
        """Passing time_column alongside a DatetimeIndex is refused."""
        flat = its_simple_data.rename_axis("date").reset_index()
        both = flat.set_index(pd.Index(range(len(flat)), name="row"))
        with pytest.raises(DataException, match="already has a meaningful index"):
            cp.InterruptedTimeSeries(
                both,
                pd.to_datetime("2015-01-01"),
                formula="timeseries ~ 1 + linear_trend",
                model=LinearRegression(),
                time_column="date",
            )

    def test_pandas_input_index_is_not_renamed(self, its_simple_data):
        """The caller's index name survives construction."""
        data = its_simple_data.copy()
        data.index.name = "my_dates"
        cp.InterruptedTimeSeries(
            data,
            pd.to_datetime("2015-01-01"),
            formula="timeseries ~ 1 + linear_trend",
            model=LinearRegression(),
        )
        assert data.index.name == "my_dates"


class TestSyntheticControl:
    """SyntheticControl with an explicit time column."""

    def test_polars_matches_pandas(self, sc_data, mock_pymc_sample):
        """The Polars run with time_column matches the pandas run."""
        treatment_time = 70
        control_units = ["a", "b", "c", "d", "e", "f", "g"]
        treated_units = ["actual"]

        def build(frame, time_column=None):
            return cp.SyntheticControl(
                frame,
                treatment_time,
                control_units=control_units,
                treated_units=treated_units,
                model=LinearRegression(),
                time_column=time_column,
            )

        from_pandas = build(sc_data)
        from_polars = build(to_polars_with_time(sc_data, "time"), time_column="time")

        assert from_polars.datapre.index.equals(from_pandas.datapre.index)
        assert from_polars.datapost.index.equals(from_pandas.datapost.index)

    def test_polars_without_time_column_raises(self, sc_data):
        """Without time_column the Polars input is refused."""
        with pytest.raises(DataException, match="Pass `time_column`"):
            cp.SyntheticControl(
                to_polars_with_time(sc_data, "time"),
                70,
                control_units=["a", "b", "c", "d", "e", "f", "g"],
                treated_units=["actual"],
                model=LinearRegression(),
            )


class TestSyntheticDifferenceInDifferences:
    """SyntheticDifferenceInDifferences with an explicit time column."""

    def test_polars_matches_pandas(self, sc_data, mock_pymc_sample):
        """The Polars run with time_column matches the pandas run."""
        treatment_time = 70
        control_units = ["a", "b", "c", "d", "e", "f", "g"]
        treated_units = ["actual"]

        def build(frame, time_column=None):
            return cp.SyntheticDifferenceInDifferences(
                frame,
                treatment_time,
                control_units=control_units,
                treated_units=treated_units,
                time_column=time_column,
            )

        from_pandas = build(sc_data)
        from_polars = build(to_polars_with_time(sc_data, "time"), time_column="time")

        assert from_polars.datapre.index.equals(from_pandas.datapre.index)
        assert from_polars.datapost.index.equals(from_pandas.datapost.index)

    def test_polars_without_time_column_raises(self, sc_data):
        """Without time_column the Polars input is refused."""
        with pytest.raises(DataException, match="Pass `time_column`"):
            cp.SyntheticDifferenceInDifferences(
                to_polars_with_time(sc_data, "time"),
                70,
                control_units=["a", "b", "c", "d", "e", "f", "g"],
                treated_units=["actual"],
            )
