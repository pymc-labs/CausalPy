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
"""String thresholds for step and ramp follow the datetime column timezone."""

from datetime import UTC, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from patsy import PatsyError

from causalpy.formula_utils import build_formula_matrices
from causalpy.transforms import _threshold_in_column_timezone

MINUS_FIVE = timezone(timedelta(hours=-5))
# Fixed offsets rather than named zones: tzdata is not guaranteed to be installed.
PLUS_FIVE = timezone(timedelta(hours=5))


def _effect_column(design, transform_name: str) -> np.ndarray:
    names = list(design.design_info.column_names)
    matches = [
        i for i, name in enumerate(names) if name.startswith(f"{transform_name}(")
    ]
    assert len(matches) == 1, names
    return np.asarray(design)[:, matches[0]]


@pytest.mark.parametrize(
    ("tz", "start", "threshold", "expected_step", "expected_ramp"),
    [
        (
            "UTC",
            "2020-01-01",
            "2020-01-03",
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 2.0],
        ),
        (
            PLUS_FIVE,
            "2020-06-01",
            "2020-06-02",
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ),
    ],
    ids=["utc-daily", "plus-five-wall-time"],
)
@pytest.mark.parametrize("transform_name", ["step", "ramp"])
def test_string_threshold_uses_column_timezone(
    tz, start, threshold, expected_step, expected_ramp, transform_name
):
    """A string threshold is a wall time in the column timezone, not UTC-naive."""
    periods = len(expected_step)
    frame = pd.DataFrame(
        {
            "date": pd.date_range(start, periods=periods, freq="D", tz=tz),
            "y": range(periods),
        }
    )
    formula = f'y ~ 1 + {transform_name}(date, "{threshold}")'

    _y, design = build_formula_matrices(formula, frame)

    expected = expected_step if transform_name == "step" else expected_ramp
    np.testing.assert_allclose(_effect_column(design, transform_name), expected)


def test_hourly_utc_minus_five_flips_at_local_midnight():
    """Local midnight is the threshold on an hourly UTC-05 column.

    Rows are 2020-01-02 22:00 through 2020-01-03 02:00. The row equal to
    ``2020-01-03`` is 1 for step and 0 for ramp.
    """
    frame = pd.DataFrame(
        {
            "date": pd.date_range(
                "2020-01-02 22:00", periods=5, freq="h", tz=MINUS_FIVE
            ),
            "y": range(5),
        }
    )
    expected = {
        "step": [0.0, 0.0, 1.0, 1.0, 1.0],
        "ramp": [0.0, 0.0, 0.0, 1.0 / 24.0, 2.0 / 24.0],
    }

    for transform_name, values in expected.items():
        formula = f'y ~ 1 + {transform_name}(date, "2020-01-03")'
        _y, design = build_formula_matrices(formula, frame)
        column = _effect_column(design, transform_name)
        np.testing.assert_allclose(column, values)
        # The third row is exactly the threshold instant.
        if transform_name == "step":
            assert column[2] == 1.0
        else:
            assert column[2] == 0.0


def test_naive_column_and_numeric_threshold_are_unchanged():
    """Naive datetime columns and numeric thresholds keep the previous comparison."""
    naive = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=5, freq="D"), "y": range(5)}
    )
    expected = {
        "step": [0.0, 0.0, 1.0, 1.0, 1.0],
        "ramp": [0.0, 0.0, 0.0, 1.0, 2.0],
    }
    for transform_name, values in expected.items():
        formula = f'y ~ 1 + {transform_name}(date, "2020-01-03")'
        _y, design = build_formula_matrices(formula, naive)
        np.testing.assert_allclose(_effect_column(design, transform_name), values)

    numeric = pd.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "y": range(5)})
    numeric_expected = {
        "step": [0.0, 0.0, 1.0, 1.0, 1.0],
        "ramp": [0.0, 0.0, 0.0, 1.0, 2.0],
    }
    for transform_name, values in numeric_expected.items():
        formula = f"y ~ 1 + {transform_name}(t, 2)"
        _y, design = build_formula_matrices(formula, numeric)
        np.testing.assert_allclose(_effect_column(design, transform_name), values)


def test_naive_column_does_not_strip_an_aware_threshold():
    """Deleting the naive-column return must not strip an aware threshold.

    Falling through to ``tz_convert(None)`` would drop the timezone and compare
    wall times. The early return keeps the aware timestamp, so both formulas
    still raise.
    """
    frame = pd.DataFrame(
        {"date": pd.date_range("2020-03-01", periods=4, freq="D"), "y": range(4)}
    )
    aware = pd.Timestamp("2020-03-02 15:30", tz=UTC)

    for transform_name in ("step", "ramp"):
        formula = f"y ~ 1 + {transform_name}(date, aware)"
        with pytest.raises(PatsyError, match="tz-naive and tz-aware"):
            build_formula_matrices(formula, frame)

    origin = pd.Timestamp(frame["date"].min())
    kept = _threshold_in_column_timezone(aware, origin)
    assert kept.tzinfo is not None
    assert kept == aware


def test_already_aware_threshold_is_compared_as_that_instant():
    """An aware threshold is converted into the column timezone, then compared.

    ``2020-01-03 05:00:00+00:00`` is local midnight on the UTC-05 column, not
    05:00 wall time. Skipping ``tz_convert`` leaves the UTC offset in place.
    """
    frame = pd.DataFrame(
        {
            "date": pd.date_range(
                "2020-01-02 22:00", periods=5, freq="h", tz=MINUS_FIVE
            ),
            "y": range(5),
        }
    )
    threshold = "2020-01-03 05:00:00+00:00"
    expected = {
        "step": [0.0, 0.0, 1.0, 1.0, 1.0],
        "ramp": [0.0, 0.0, 0.0, 1.0 / 24.0, 2.0 / 24.0],
    }
    for transform_name, values in expected.items():
        formula = f'y ~ 1 + {transform_name}(date, "{threshold}")'
        _y, design = build_formula_matrices(formula, frame)
        np.testing.assert_allclose(_effect_column(design, transform_name), values)

    origin = pd.Timestamp(frame["date"].min())
    aligned = _threshold_in_column_timezone(threshold, origin)
    assert aligned.utcoffset() == timedelta(hours=-5)
    assert aligned == pd.Timestamp(threshold)
