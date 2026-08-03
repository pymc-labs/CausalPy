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
from causalpy.checks.placebo_in_time import PlaceboInTime
from causalpy.custom_exceptions import BadIndexException, DataException
from causalpy.input_data import to_pandas_with_time_index
from causalpy.pipeline import PipelineContext

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

    def test_pandas_without_time_column_keeps_unsorted_duplicate_index(self):
        """Native pandas indexes remain untouched without explicit time_column."""
        frame = pd.DataFrame({"y": [1, 2, 3]}, index=pd.Index([2, 0, 0], name="t"))
        result = to_pandas_with_time_index(frame)
        assert result.index.equals(frame.index)

    def test_pandas_without_time_column_preserves_named_datetime_index(self):
        """The legacy pandas-index path retains its time-axis type and name."""
        index = pd.date_range("2020-01-01", periods=2, name="date")
        frame = pd.DataFrame({"y": [1, 2]}, index=index)
        result = to_pandas_with_time_index(frame)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.equals(index)
        assert result.index.name == "date"

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

    def test_time_column_conflicting_with_named_range_index_raises(self):
        """A named RangeIndex is meaningful and cannot be overwritten."""
        frame = pd.DataFrame({"t": [100, 200], "y": [1, 2]})
        frame.index.name = "row"
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

    def test_duplicate_time_column_raises(self):
        """A time axis with repeats is refused."""
        frame = pl.DataFrame({"t": [1, 1, 2], "y": [3, 4, 5]})
        with pytest.raises(DataException, match="duplicate values"):
            to_pandas_with_time_index(frame, time_column="t")

    def test_unsorted_time_column_raises(self):
        """An out-of-order time axis is refused rather than silently accepted.

        A dataframe library with no index carries no row-order guarantee, so
        this is an easy accident. The pre/post split is value-based and would
        survive it, but anything order-dependent downstream would not.
        """
        frame = pl.DataFrame({"t": [3, 1, 2], "y": [4, 5, 6]})
        with pytest.raises(DataException, match="is not sorted"):
            to_pandas_with_time_index(frame, time_column="t")


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
        result = cp.InterruptedTimeSeries(
            data,
            pd.to_datetime("2015-01-01"),
            formula="timeseries ~ 1 + linear_trend",
            model=LinearRegression(),
        )
        assert data.index.name == "my_dates"
        assert result.data.index.name == "obs_ind"


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


class TestNativePandasTimeIndex:
    """Index-based experiments preserve a pandas time axis without time_column."""

    @pytest.mark.parametrize(
        "experiment", ["SyntheticControl", "SyntheticDifferenceInDifferences"]
    )
    def test_native_datetime_index_is_preserved(
        self, sc_data, mock_pymc_sample, experiment
    ):
        """A named DatetimeIndex remains the native time axis for SC and SDiD."""
        index = pd.date_range("2020-01-01", periods=len(sc_data), freq="D", name="date")
        data = sc_data.copy()
        data.index = index
        treatment_time = index[70]
        kwargs = {
            "control_units": ["a", "b", "c", "d", "e", "f", "g"],
            "treated_units": ["actual"],
        }
        if experiment == "SyntheticControl":
            kwargs["model"] = LinearRegression()
        result = getattr(cp, experiment)(data, treatment_time, **kwargs)

        expected_index = index.rename("obs_ind")
        assert data.index.equals(index)
        assert data.index.name == "date"
        assert result.data.index.equals(expected_index)
        assert result.data.index.name == "obs_ind"
        assert result.datapre.index.equals(expected_index[:70])
        assert result.datapost.index.equals(expected_index[70:])


class TestTimestampMismatchMessage:
    """The treatment-time mismatch error states the right requirement.

    The ``treatment_time`` branch read "must be pd.Timestamp" on the path that
    fires precisely because it already is one. The sibling
    ``treatment_end_time`` branch had it right, so this was a copy-paste slip.
    Reachable from any backend, but a ``time_column`` holding strings rather
    than dates lands here, which is what surfaced it.
    """

    @staticmethod
    def _build(name, data):
        """Construct one of the three classes with a Timestamp treatment time."""
        units = {
            "control_units": ["a", "b", "c", "d", "e", "f", "g"],
            "treated_units": ["actual"],
        }
        treatment_time = pd.Timestamp("2020-01-01")
        if name == "InterruptedTimeSeries":
            return cp.InterruptedTimeSeries(
                data, treatment_time, formula="actual ~ 1 + a", model=LinearRegression()
            )
        if name == "SyntheticControl":
            return cp.SyntheticControl(
                data, treatment_time, model=LinearRegression(), **units
            )
        return cp.SyntheticDifferenceInDifferences(data, treatment_time, **units)

    @pytest.mark.parametrize(
        "experiment",
        [
            "InterruptedTimeSeries",
            "SyntheticControl",
            "SyntheticDifferenceInDifferences",
        ],
    )
    def test_non_datetime_index_with_timestamp_treatment_time(
        self, sc_data, experiment, mock_pymc_sample
    ):
        """A string time axis plus a Timestamp treatment time explains itself.

        Checked on all three classes, since the slip was copy-pasted into each
        of them rather than living in one shared place.
        """
        as_text = sc_data.copy()
        # Zero-padded so the string index stays sorted. Plain str() would give
        # "10" < "9", and InterruptedTimeSeries checks monotonicity first, so
        # it would raise about ordering before reaching the type check.
        as_text.index = [f"{i:04d}" for i in range(len(as_text))]
        with pytest.raises(BadIndexException, match="must not be pd.Timestamp"):
            self._build(experiment, as_text)


class TestUnsortedTimeColumnEndToEnd:
    """An unsorted time column is refused at the experiment boundary.

    SyntheticControl and SyntheticDifferenceInDifferences validate only the
    treatment-time type, not index order, so before this guard they accepted a
    shuffled time axis and scrambled it silently. InterruptedTimeSeries already
    refused it via its own index validation.
    """

    @pytest.mark.parametrize(
        "experiment", ["SyntheticControl", "SyntheticDifferenceInDifferences"]
    )
    def test_shuffled_time_column_raises(self, sc_data, experiment):
        """Shuffling the rows of an otherwise valid frame is refused."""
        shuffled = sc_data.sample(frac=1.0, random_state=1)
        with pytest.raises(DataException, match="is not sorted"):
            getattr(cp, experiment)(
                to_polars_with_time(shuffled, "time"),
                70,
                control_units=["a", "b", "c", "d", "e", "f", "g"],
                treated_units=["actual"],
                time_column="time",
            )

    @pytest.mark.parametrize(
        "experiment", ["SyntheticControl", "SyntheticDifferenceInDifferences"]
    )
    def test_duplicate_time_column_raises(self, sc_data, experiment):
        """A duplicate explicit time axis is refused at both public boundaries."""
        duplicate = sc_data.rename_axis("time").reset_index()
        duplicate.loc[1, "time"] = duplicate.loc[0, "time"]
        with pytest.raises(DataException, match="duplicate values"):
            getattr(cp, experiment)(
                pl.from_pandas(duplicate),
                70,
                control_units=["a", "b", "c", "d", "e", "f", "g"],
                treated_units=["actual"],
                time_column="time",
            )


class TestSensitivityRefit:
    """Checks that re-fit an experiment must not replay ``time_column``.

    ``PlaceboInTime`` hands its factory a slice of ``experiment.data``, which
    is already normalized: the time column has been moved onto the index, so
    replaying the argument raised ``DataException`` on every fold. The check
    swallows per-fold failures, so the user saw an inconclusive result rather
    than an error. The other checks re-fit from the caller's original data and
    do still need the argument.
    """

    def _context(self):
        """Build a pipeline context from a Polars frame using time_column."""
        n = 200
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "date": np.arange(n),
                "trend": np.arange(n) / n,
                "y": rng.normal(size=n),
            }
        )
        step = cp.steps.EstimateEffect(
            cp.InterruptedTimeSeries,
            treatment_time=150,
            formula="y ~ 1 + trend",
            model=LinearRegression(),
            time_column="date",
        )
        return step.run(PipelineContext(data=pl.from_pandas(frame)))

    def test_config_still_carries_time_column(self):
        """The recorded config keeps it, since other checks re-fit raw data."""
        context = self._context()
        assert context.experiment_config["time_column"] == "date"
        assert "date" not in context.experiment.data.columns

    def test_placebo_factory_refits_without_time_column(self):
        """The placebo factory rebuilds from normalized data without raising."""
        context = self._context()
        factory = PlaceboInTime(n_folds=2)._get_factory(context)
        refit = factory(context.experiment.data.iloc[:100], 75)
        assert isinstance(refit, cp.InterruptedTimeSeries)
