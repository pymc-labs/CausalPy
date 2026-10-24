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

from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from causalpy.formula_utils import build_formula_matrices

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
