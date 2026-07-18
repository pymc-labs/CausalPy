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
"""Regression tests for continuous datetime predictors in Patsy formulas."""

import numpy as np
import pandas as pd
import pytest
from patsy import build_design_matrices
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.formula_utils import build_formula_matrices
from causalpy.transforms import ElapsedDaysTransform


def test_bare_datetime_predictor_is_continuous():
    dates = pd.date_range("2020-01-01", periods=4, freq="MS")
    data = pd.DataFrame({"date": dates, "y": np.arange(4, dtype=float)})

    _, X = build_formula_matrices("y ~ 1 + date", data)

    assert X.design_info.column_names == ["Intercept", "elapsed(date)"]
    np.testing.assert_allclose(
        np.asarray(X)[:, 1],
        (dates - dates.min()).days,
    )


def test_elapsed_days_transform_tracks_origin_across_chunks():
    transform = ElapsedDaysTransform()
    later_dates = pd.date_range("2020-01-05", periods=2, freq="D")
    earlier_dates = pd.date_range("2020-01-01", periods=2, freq="D")

    transform.memorize_chunk(later_dates)
    transform.memorize_chunk(earlier_dates)
    transform.memorize_finish()

    assert transform._origin == pd.Timestamp("2020-01-01")
    np.testing.assert_array_equal(transform.transform(later_dates), [4.0, 5.0])


def test_elapsed_days_transform_requires_fitted_origin():
    with pytest.raises(RuntimeError, match="origin"):
        ElapsedDaysTransform().transform(pd.date_range("2020-01-01", periods=1))


def test_datetime_predictor_uses_fitted_origin_out_of_sample():
    dates = pd.date_range("2020-01-01", periods=8, freq="MS")
    data = pd.DataFrame({"date": dates, "y": np.arange(8, dtype=float)})

    y_pre, X_pre = build_formula_matrices("y ~ 1 + date", data.iloc[:4])
    _, X_post = build_design_matrices(
        [y_pre.design_info, X_pre.design_info], data.iloc[4:]
    )

    assert X_post.design_info.column_names == X_pre.design_info.column_names
    np.testing.assert_allclose(
        np.asarray(X_post)[:, 1],
        (dates[4:] - dates[0]).days,
    )


def test_C_datetime_predictor_remains_categorical():
    dates = pd.date_range("2020-01-01", periods=4, freq="MS")
    data = pd.DataFrame({"date": dates, "y": np.arange(4, dtype=float)})

    _, X = build_formula_matrices("y ~ 1 + C(date)", data)

    assert X.shape == (4, 4)
    assert all("C(date)" in label for label in X.design_info.column_names[1:])


def test_datetime_predictor_in_interaction_is_continuous():
    dates = pd.date_range("2020-01-01", periods=4, freq="MS")
    data = pd.DataFrame(
        {"date": dates, "group": [0, 1, 0, 1], "y": np.arange(4, dtype=float)}
    )

    _, X = build_formula_matrices("y ~ 1 + date:group", data)

    assert X.design_info.column_names == ["Intercept", "elapsed(date):group"]


def test_interrupted_time_series_predicts_with_bare_datetime_predictor():
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    data = pd.DataFrame(
        {"date": dates, "y": 10 + 0.5 * np.arange(len(dates), dtype=float)}
    ).set_index("date", drop=False)

    result = cp.InterruptedTimeSeries(
        data,
        treatment_time=pd.Timestamp("2021-07-01"),
        formula="y ~ 1 + date",
        model=LinearRegression(),
    )

    assert result.labels == ["Intercept", "elapsed(date)"]
    assert result.pre_design["X"].shape[1] == result.post_design["X"].shape[1] == 2
    assert len(result.post_pred) == len(result.datapost)


def test_piecewise_its_uses_single_datetime_baseline_column():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    t = np.arange(len(dates))
    data = pd.DataFrame(
        {
            "date": dates,
            "y": 10 + 0.1 * t + 5 * (t >= 50) + 0.2 * np.maximum(0, t - 50),
        }
    )

    result = cp.PiecewiseITS(
        data,
        formula="y ~ 1 + date + step(date, '2020-02-20') + ramp(date, '2020-02-20')",
        model=LinearRegression(),
    )

    assert result.labels == [
        "Intercept",
        "elapsed(date)",
        "step(date, '2020-02-20')",
        "ramp(date, '2020-02-20')",
    ]
