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
Tests for dataframe-agnostic input handling
"""

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from causalpy.input_data import to_pandas


def test_pandas_input_is_returned_unchanged():
    """A pandas input keeps its columns, values, and index."""
    data = pd.DataFrame({"y": [1, 2], "x": [3.0, 4.0]}, index=[10, 11])
    result = to_pandas(data)
    pd.testing.assert_frame_equal(result, data)


def test_pandas_input_is_copied():
    """The result is a copy, so mutating it leaves the caller's frame alone."""
    data = pd.DataFrame({"y": [1, 2]}, index=pd.Index([10, 11], name="t"))
    result = to_pandas(data)
    result.index.name = "obs_ind"
    result.loc[10, "y"] = 99
    assert data.index.name == "t"
    assert data.loc[10, "y"] == 1
    data.index.name = "caller"
    data.loc[10, "y"] = 0
    assert result.index.name == "obs_ind"
    assert result.loc[10, "y"] == 99


def test_polars_input_is_converted():
    """A Polars frame converts to pandas with a default RangeIndex."""
    result = to_pandas(pl.DataFrame({"y": [1, 2], "x": [3.0, 4.0]}))
    expected = pd.DataFrame({"y": [1, 2], "x": [3.0, 4.0]})
    pd.testing.assert_frame_equal(result, expected)
    assert isinstance(result.index, pd.RangeIndex)


def test_pyarrow_input_is_converted():
    """A PyArrow table converts to pandas."""
    result = to_pandas(pa.table({"y": [1, 2], "x": [3.0, 4.0]}))
    assert list(result.columns) == ["y", "x"]
    assert result["y"].tolist() == [1, 2]


def test_polars_datetime_column_survives_conversion():
    """Datetime columns keep a datetime dtype after conversion."""
    dates = pl.datetime_range(
        pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"), "1d", eager=True
    )
    result = to_pandas(pl.DataFrame({"t": dates, "y": [1, 2, 3]}))
    assert pd.api.types.is_datetime64_any_dtype(result["t"])


@pytest.mark.parametrize(
    "bad_input",
    [[1, 2], {"y": [1, 2]}, None, pl.Series([1, 2])],
    ids=["list", "dict", "none", "series"],
)
def test_non_dataframe_input_raises(bad_input):
    """Anything that is not an eager dataframe raises TypeError."""
    with pytest.raises(TypeError, match="must be an eager dataframe"):
        to_pandas(bad_input)


def test_lazy_frame_raises():
    """Lazy frames are rejected, since CausalPy needs materialized data."""
    with pytest.raises(TypeError, match="must be an eager dataframe"):
        to_pandas(pl.LazyFrame({"y": [1, 2]}))


def test_lazy_frame_error_points_at_collect():
    """The lazy case says what to do, since the library itself is supported.

    Telling a Polars caller that Polars is supported and that their object is
    not a supported dataframe is the confusing part, so the message names the
    fix rather than leaving them to guess.
    """
    with pytest.raises(TypeError, match=r"Call `\.collect\(\)` first"):
        to_pandas(pl.LazyFrame({"y": [1, 2]}))


def test_non_dataframe_error_has_no_lazy_hint():
    """The collect hint stays out of the way for inputs that are not frames."""
    with pytest.raises(TypeError) as excinfo:
        to_pandas([1, 2])
    assert "collect" not in str(excinfo.value)


def test_error_message_names_the_argument():
    """The error message names the offending argument."""
    with pytest.raises(TypeError, match="`donor_data` must be an eager dataframe"):
        to_pandas([1, 2], argument_name="donor_data")
