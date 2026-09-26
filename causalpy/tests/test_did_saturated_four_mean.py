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
"""Saturated 2x2 difference-in-differences equals the four cell-mean contrast."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import causalpy as cp

CELL_MEANS = {(0, 0): 1.0, (0, 1): 3.0, (1, 0): 2.0, (1, 1): 8.0}


def _saturated_did_frame(n_per_cell: int = 5) -> pd.DataFrame:
    rows = []
    unit = 0
    for group in (0, 1):
        for post in (0, 1):
            for _ in range(n_per_cell):
                rows.append(
                    {
                        "unit": f"u{unit}",
                        "t": post,
                        "group": group,
                        "post_treatment": post,
                        "y": CELL_MEANS[(group, post)],
                    }
                )
                unit += 1
    return pd.DataFrame(rows)


def test_saturated_did_interaction_equals_four_mean_contrast():
    """In a saturated 2x2 design the interaction is the four cell-mean contrast."""
    frame = _saturated_did_frame()
    four_mean = (CELL_MEANS[(1, 1)] - CELL_MEANS[(1, 0)]) - (
        CELL_MEANS[(0, 1)] - CELL_MEANS[(0, 0)]
    )
    result = cp.DifferenceInDifferences(
        frame,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(fit_intercept=False),
    )
    impact = float(np.asarray(result.causal_impact).reshape(-1)[0])
    np.testing.assert_allclose(impact, four_mean, atol=1e-12)
    assert four_mean == 4.0
