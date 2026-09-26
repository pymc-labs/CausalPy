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
"""Covariate-adjusted DiD reports the interaction, not the four cell means."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import causalpy as cp

CELL_MEANS = {(0, 0): 1.0, (0, 1): 3.0, (1, 0): 2.0, (1, 1): 8.0}


def _cell_mean(frame: pd.DataFrame, group: int, post: int) -> float:
    mask = (frame["group"] == group) & (frame["post_treatment"] == post)
    return float(frame.loc[mask, "y"].mean())


def test_covariate_adjusted_did_is_not_the_four_mean_contrast():
    """With another covariate, causal impact stays the interaction coefficient.

    The saturated cells have interaction 4. Adding ``2 x`` inside cells moves
    the four observed means, and the report must not follow those means.
    """
    rng = np.random.default_rng(3)
    rows = []
    unit = 0
    for group in (0, 1):
        for post in (0, 1):
            for _ in range(5):
                rows.append(
                    {
                        "unit": f"u{unit}",
                        "t": post,
                        "group": group,
                        "post_treatment": post,
                        "x": float(rng.normal()),
                        "y": CELL_MEANS[(group, post)],
                    }
                )
                unit += 1
    frame = pd.DataFrame(rows)
    frame["y"] = frame["y"] + 2.0 * frame["x"]
    four_mean = (_cell_mean(frame, 1, 1) - _cell_mean(frame, 1, 0)) - (
        _cell_mean(frame, 0, 1) - _cell_mean(frame, 0, 0)
    )
    result = cp.DifferenceInDifferences(
        frame,
        formula="y ~ 1 + x + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(fit_intercept=False),
    )
    impact = float(np.asarray(result.causal_impact).reshape(-1)[0])
    reported = float(result.effect_summary().table.loc["treatment_effect", "mean"])

    np.testing.assert_allclose(impact, 4.0, atol=1e-8)
    np.testing.assert_allclose(reported, impact, atol=1e-8)
    assert abs(impact - four_mean) > 0.1
