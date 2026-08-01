#   Copyright 2026 - 2026 The PyMC Labs Developers
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
"""Behavioral regressions for pandas 2.3 and 3 compatibility."""

from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.custom_exceptions import BadIndexException, DataException
from causalpy.data.simulate_data import generate_time_series_data_simple
from causalpy.date_utils import validate_treatment_time_against_index
from causalpy.formula_utils import _normalize_patsy_data, build_formula_matrices

# Fixed offsets rather than named zones: tzdata is not guaranteed to be
# installed in every environment the matrix builds.
MINUS_FIVE = timezone(timedelta(hours=-5))


@pytest.mark.parametrize(
    "unit",
    [
        pd.Series(["north", "north", "south", "south"]),
        pd.Series(["north", "north", "south", "south"], dtype="string"),
    ],
    ids=["inferred-strings", "StringDtype"],
)
def test_formula_matrices_accept_string_columns_without_mutating_input(unit):
    """Patsy formulas accept pandas 3-style strings without changing the frame."""
    data = pd.DataFrame({"unit": unit, "y": [1.0, 2.0, 3.0, 4.0]})
    original = data.copy(deep=True)

    y, X = build_formula_matrices("y ~ 1 + C(unit)", data)

    assert y.shape == (4, 1)
    assert X.shape == (4, 2)
    pd.testing.assert_frame_equal(data, original)


def test_patsy_normalization_leaves_non_string_extension_dtypes_unchanged():
    """Only extension strings are normalized before Patsy consumes mixed frames."""
    data = pd.DataFrame(
        {
            "unit": pd.Series(["north", "south"], dtype="string"),
            "category": pd.Series(pd.Categorical(["a", "b"])),
            "number": pd.Series([1, 2], dtype="Int64"),
            "y": [1.0, 2.0],
        }
    )
    original = data.copy(deep=True)

    normalized = _normalize_patsy_data(data)

    assert normalized is not data
    assert normalized["unit"].dtype == object
    assert isinstance(normalized["category"].dtype, pd.CategoricalDtype)
    assert normalized["number"].dtype == data["number"].dtype

    y, X = build_formula_matrices("y ~ 1 + C(unit) + C(category)", data)

    assert y.shape == (2, 1)
    assert X.shape == (2, 3)
    pd.testing.assert_frame_equal(data, original)


def test_patsy_normalization_is_a_no_op_for_nan_backed_string_columns():
    """Only ``pd.NA``-sentinel string columns need the Patsy #206 workaround.

    Plain string literals give an object column on pandas 2.3 and the default
    ``str`` dtype (``np.nan`` sentinel) on pandas 3. Neither trips Patsy, so
    neither should pay for a copy.
    """
    data = pd.DataFrame({"unit": ["north", None, "south"], "y": [1.0, 2.0, 3.0]})

    assert _normalize_patsy_data(data) is data

    y, X = build_formula_matrices("y ~ 1 + C(unit)", data)

    assert y.shape == (2, 1)
    assert X.shape == (2, 2)


def test_formula_matrices_preserve_extension_string_missingness():
    """Extension-string missing values retain Patsy's normal row-dropping behavior."""
    data = pd.DataFrame(
        {
            "unit": pd.Series(["north", "south", pd.NA, "north"], dtype="string"),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    original = data.copy(deep=True)

    y, X = build_formula_matrices("y ~ 1 + C(unit)", data)

    assert y.shape == (3, 1)
    assert X.shape == (3, 2)
    assert pd.isna(data.loc[2, "unit"])
    pd.testing.assert_frame_equal(data, original)


def test_regression_discontinuity_normalizes_nullable_integers_without_mutation():
    """Nullable integer indicators become boolean only on the experiment-owned frame."""
    x = np.linspace(0.0, 1.0, 20)
    data = pd.DataFrame(
        {
            "x": x,
            "treated": pd.Series((x >= 0.5).astype(int), dtype="Int64"),
            "y": 1.0 + 2.0 * x + (x >= 0.5),
        }
    )
    original = data.copy(deep=True)

    result = cp.RegressionDiscontinuity(
        data,
        formula="y ~ 1 + x + treated + x:treated",
        treatment_threshold=0.5,
        model=LinearRegression(),
    )

    assert pd.api.types.is_bool_dtype(result.data["treated"])
    assert pd.api.types.is_integer_dtype(data["treated"])
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize(
    "unit",
    [
        pd.Series(["north", "north", "south", "south"]),
        pd.Series(["north", "north", "south", "south"], dtype="string"),
    ],
    ids=["inferred-strings", "StringDtype"],
)
def test_panel_regression_uses_observed_categorical_groups(unit):
    """Demeaning ignores unused category levels and accepts string fixed effects."""
    data = pd.DataFrame(
        {
            "unit": unit,
            "time": pd.Categorical([0, 1, 0, 1], categories=[0, 1, 2]),
            "treatment": pd.Series([False, True, False, True], dtype="boolean"),
            "y": [1.0, 3.0, 2.0, 4.0],
        }
    )
    original = data.copy(deep=True)

    result = cp.PanelRegression(
        data,
        formula="y ~ treatment",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="demeaned",
        model=LinearRegression(),
    )

    assert list(result._group_means["unit"].index) == ["north", "south"]
    assert list(result._group_means["time"].index) == [0, 1]
    pd.testing.assert_frame_equal(data, original)


def test_panel_regression_rejects_missing_string_fixed_effects():
    """Missing fixed-effect identifiers are rejected before grouping can drop them."""
    data = pd.DataFrame(
        {
            "unit": pd.Series(["north", pd.NA, "south", "south"], dtype="string"),
            "time": [0, 1, 0, 1],
            "treatment": [False, True, False, True],
            "y": [1.0, 3.0, 2.0, 4.0],
        }
    )
    original = data.copy(deep=True)

    with pytest.raises(
        DataException,
        match="Fixed-effect variable 'unit' must not contain missing values",
    ):
        cp.PanelRegression(
            data,
            formula="y ~ treatment",
            unit_fe_variable="unit",
            time_fe_variable="time",
            fe_method="demeaned",
            model=LinearRegression(),
        )

    pd.testing.assert_frame_equal(data, original)


def test_staggered_did_creates_float_event_times_without_mutating_input():
    """Never-treated rows receive missing float event times without mutating callers."""
    data = pd.DataFrame(
        {
            "unit": [0, 0, 1, 1, 2, 2],
            "time": [0, 1, 0, 1, 0, 1],
            "treated": [0, 1, 0, 1, 0, 0],
            "y": [1.0, 3.0, 2.0, 4.0, 1.5, 1.7],
        }
    )
    original = data.copy(deep=True)

    result = cp.StaggeredDifferenceInDifferences(
        data,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        model=LinearRegression(),
    )

    assert pd.api.types.is_float_dtype(result.data_["event_time"])
    assert result.data_.loc[result.data_["unit"] == 2, "event_time"].isna().all()
    pd.testing.assert_frame_equal(data, original)


def test_time_series_generator_has_explicit_datetime_treatment_boundary():
    """Generated series retains a datetime index and exclusive treatment boundary."""
    treatment_time = pd.Timestamp("2015-01-31")

    data = generate_time_series_data_simple(treatment_time=treatment_time, seed=42)

    expected_index = pd.date_range(
        start="2010-01-01", end="2020-01-01", freq="ME", name="date"
    )
    pd.testing.assert_index_equal(data.index, expected_index, exact=True)
    assert data.loc[:treatment_time, "causal effect"].eq(0).all()
    assert data.loc[data.index > treatment_time, "causal effect"].eq(2).all()


def test_interrupted_time_series_predicts_with_string_extension_covariates():
    """Fit and prediction matrices handle ``StringDtype`` at the Patsy boundary."""
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    data = pd.DataFrame(
        {
            "y": np.arange(len(dates), dtype=float),
            "segment": pd.array(
                ["north", "south", "north", "south"] * 3, dtype="string"
            ),
        },
        index=dates,
    )
    original = data.copy(deep=True)

    result = cp.InterruptedTimeSeries(
        data,
        treatment_time=dates[6],
        formula="y ~ 1 + C(segment)",
        model=LinearRegression(),
    )

    assert result.pre_design["X"].sizes["obs_ind"] == 6
    assert result.post_design["X"].sizes["obs_ind"] == 6
    assert result.pre_design.indexes["obs_ind"].equals(
        result.pre_pred.indexes["obs_ind"]
    )
    assert result.post_design.indexes["obs_ind"].equals(
        result.post_pred.indexes["obs_ind"]
    )
    pd.testing.assert_frame_equal(data, original)


def test_interrupted_time_series_preserves_timezone_aware_boundary_and_input():
    """Timezone-aware treatment dates split periods correctly and reject ``NaT``."""
    dates = pd.date_range("2020-01-01", periods=12, freq="MS", tz="UTC")
    data = pd.DataFrame(
        {"y": np.arange(len(dates), dtype=float)},
        index=dates,
    )
    original = data.copy(deep=True)
    treatment_time = dates[6]

    result = cp.InterruptedTimeSeries(
        data,
        treatment_time=treatment_time,
        formula="y ~ 1",
        model=LinearRegression(),
    )

    assert result.datapre.index.max() == dates[5]
    assert result.datapost.index.min() == treatment_time
    assert result.pre_design.indexes["obs_ind"].equals(
        result.pre_pred.indexes["obs_ind"]
    )
    assert result.post_design.indexes["obs_ind"].equals(
        result.post_pred.indexes["obs_ind"]
    )
    pd.testing.assert_frame_equal(data, original)

    with pytest.raises(BadIndexException, match="treatment_time must not be missing"):
        cp.InterruptedTimeSeries(
            data,
            treatment_time=pd.NaT,
            formula="y ~ 1",
            model=LinearRegression(),
        )


@pytest.mark.parametrize(
    "index, treatment_time, match",
    [
        (
            pd.date_range("2020-01-01", periods=3, freq="MS"),
            pd.NaT,
            "treatment_time must not be missing",
        ),
        (
            pd.RangeIndex(3),
            np.nan,
            "treatment_time must not be missing",
        ),
        (
            pd.date_range("2020-01-01", periods=3, freq="MS"),
            1,
            "If data.index is DatetimeIndex, treatment_time must be pd.Timestamp",
        ),
        (
            pd.RangeIndex(3),
            pd.Timestamp("2020-01-01"),
            "If data.index is not DatetimeIndex, treatment_time must not be "
            "pd.Timestamp",
        ),
        (
            pd.date_range("2020-01-01", periods=3, freq="MS", tz="UTC"),
            pd.Timestamp("2020-02-01"),
            "either both be timezone-aware or both be timezone-naive",
        ),
        (
            pd.date_range("2020-01-01", periods=3, freq="MS"),
            pd.Timestamp("2020-02-01", tz="UTC"),
            "either both be timezone-aware or both be timezone-naive",
        ),
    ],
    ids=[
        "NaT-datetime-index",
        "nan-numeric-index",
        "numeric-boundary-datetime-index",
        "timestamp-boundary-numeric-index",
        "naive-boundary-aware-index",
        "aware-boundary-naive-index",
    ],
)
def test_validate_treatment_time_rejects_incomparable_boundaries(
    index, treatment_time, match
):
    """Boundaries that cannot be compared against the index are rejected."""
    with pytest.raises(BadIndexException, match=match):
        validate_treatment_time_against_index(index, treatment_time)


def test_validate_treatment_time_uses_the_supplied_name_in_messages():
    """The helper reports the caller's parameter name, e.g. treatment_end_time."""
    with pytest.raises(BadIndexException, match="treatment_end_time must not be"):
        validate_treatment_time_against_index(
            pd.RangeIndex(3), np.nan, name="treatment_end_time"
        )


@pytest.mark.parametrize(
    "index, treatment_time",
    [
        (pd.RangeIndex(3), 1),
        (pd.Index([0.0, 1.0, 2.0]), 1.5),
        (pd.date_range("2020-01-01", periods=3, freq="MS"), pd.Timestamp("2020-02-01")),
        (
            pd.date_range("2020-01-01", periods=3, freq="MS", tz="UTC"),
            pd.Timestamp("2020-02-01", tz="UTC"),
        ),
        (
            pd.date_range("2020-01-01", periods=3, freq="MS", tz="UTC"),
            pd.Timestamp("2020-02-01", tz=MINUS_FIVE),
        ),
    ],
    ids=[
        "int-index",
        "float-index",
        "naive-datetime",
        "same-timezone",
        "different-timezones",
    ],
)
def test_validate_treatment_time_accepts_comparable_boundaries(index, treatment_time):
    """Comparable boundaries pass, including tz-aware pairs in different zones."""
    validate_treatment_time_against_index(index, treatment_time)
    # The comparison the experiments actually perform must not raise.
    assert (index < treatment_time).dtype == bool


def test_interrupted_time_series_validates_the_treatment_end_boundary():
    """``treatment_end_time`` gets the same missing-value and timezone checks."""
    dates = pd.date_range("2020-01-01", periods=12, freq="MS", tz="UTC")
    data = pd.DataFrame({"y": np.arange(len(dates), dtype=float)}, index=dates)

    with pytest.raises(
        BadIndexException, match="treatment_end_time must not be missing"
    ):
        cp.InterruptedTimeSeries(
            data,
            treatment_time=dates[4],
            formula="y ~ 1",
            model=LinearRegression(),
            treatment_end_time=pd.NaT,
        )

    with pytest.raises(
        BadIndexException,
        match="treatment_end_time and data.index must either both be timezone-aware",
    ):
        cp.InterruptedTimeSeries(
            data,
            treatment_time=dates[4],
            formula="y ~ 1",
            model=LinearRegression(),
            treatment_end_time=pd.Timestamp("2020-09-01"),
        )


def test_interrupted_time_series_rejects_timezone_naive_treatment_time():
    """A naive boundary against a tz-aware index fails before any slicing."""
    dates = pd.date_range("2020-01-01", periods=12, freq="MS", tz="UTC")
    data = pd.DataFrame({"y": np.arange(len(dates), dtype=float)}, index=dates)

    with pytest.raises(
        BadIndexException,
        match="treatment_time and data.index must either both be timezone-aware",
    ):
        cp.InterruptedTimeSeries(
            data,
            treatment_time=pd.Timestamp("2020-07-01"),
            formula="y ~ 1",
            model=LinearRegression(),
        )


def _synthetic_control_data(tz=None):
    """Build a small donor/treated panel indexed by dates in ``tz``."""
    dates = pd.date_range("2020-01-01", periods=12, freq="MS", tz=tz)
    rng = np.random.default_rng(42)
    control = rng.normal(size=(len(dates), 2)) + 10.0
    return pd.DataFrame(
        {
            "c0": control[:, 0],
            "c1": control[:, 1],
            "t0": control.mean(axis=1) + 1.0,
        },
        index=dates,
    )


@pytest.mark.parametrize(
    "experiment",
    [cp.SyntheticControl, cp.SyntheticDifferenceInDifferences],
    ids=["SyntheticControl", "SyntheticDifferenceInDifferences"],
)
@pytest.mark.parametrize(
    "treatment_time, match",
    [
        (pd.NaT, "treatment_time must not be missing"),
        (
            pd.Timestamp("2020-07-01"),
            "treatment_time and data.index must either both be timezone-aware",
        ),
    ],
    ids=["missing", "timezone-mismatch"],
)
def test_synthetic_experiments_validate_treatment_time(
    experiment, treatment_time, match
):
    """Both synthetic designs share the treatment-time boundary contract."""
    data = _synthetic_control_data(tz="UTC")

    with pytest.raises(BadIndexException, match=match):
        experiment(
            data,
            treatment_time=treatment_time,
            control_units=["c0", "c1"],
            treated_units=["t0"],
            model=LinearRegression(),
        )


def _staggered_did_data():
    """Build a small staggered-adoption panel with an explicit adoption time."""
    return pd.DataFrame(
        {
            "unit": [0, 0, 1, 1, 2, 2],
            "time": [0, 1, 0, 1, 0, 1],
            "treated": [0, 1, 0, 1, 0, 0],
            "adoption": [1.0, 1.0, 1.0, 1.0, np.inf, np.inf],
            "y": [1.0, 3.0, 2.0, 4.0, 1.5, 1.7],
        }
    )


@pytest.mark.parametrize(
    "column, kwargs, match",
    [
        (
            "unit",
            {},
            "Required column 'unit' must not contain missing values",
        ),
        (
            "time",
            {},
            "Required column 'time' must not contain missing values",
        ),
        (
            "treated",
            {},
            "Treated column 'treated' must not contain missing values",
        ),
        (
            "adoption",
            {"treatment_time_variable_name": "adoption"},
            "Treatment time column 'adoption' must not contain missing values",
        ),
    ],
    ids=["unit", "time", "treated", "treatment-time"],
)
def test_staggered_did_rejects_missing_values_in_grouping_columns(
    column, kwargs, match
):
    """Missing keys would be silently dropped by ``groupby``, so reject them."""
    data = _staggered_did_data().astype({column: "Float64"})
    data.loc[0, column] = pd.NA

    with pytest.raises(DataException, match=match):
        cp.StaggeredDifferenceInDifferences(
            data,
            formula="y ~ 1 + C(unit) + C(time)",
            unit_variable_name="unit",
            time_variable_name="time",
            model=LinearRegression(),
            **kwargs,
        )
