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

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.custom_exceptions import FormulaException


def _panel_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit": ["a", "a", "b", "b"],
            "time": [1, 2, 1, 2],
            "x": [1.0, 2.0, 2.0, 4.0],
            "y": [1.5, 2.0, 3.5, 4.0],
        }
    )


def test_panel_regression_within_demeans_by_unit():
    df = _panel_df()
    result = cp.PanelRegression(
        data=df,
        formula="y ~ x",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="within",
        model=LinearRegression(),
    )
    group_means = result.model_data.groupby("unit")[["y", "x"]].mean()
    assert np.allclose(group_means.values, 0.0)


def test_panel_regression_within_rejects_unit_dummies():
    df = _panel_df()
    with pytest.raises(FormulaException):
        cp.PanelRegression(
            data=df,
            formula="y ~ C(unit) + x",
            unit_fe_variable="unit",
            time_fe_variable="time",
            fe_method="within",
            model=LinearRegression(),
        )


def test_panel_regression_dummies_requires_unit_term():
    df = _panel_df()
    with pytest.raises(FormulaException):
        cp.PanelRegression(
            data=df,
            formula="y ~ x",
            unit_fe_variable="unit",
            time_fe_variable="time",
            fe_method="dummies",
            model=LinearRegression(),
        )


def test_panel_regression_plot_data_ols():
    df = _panel_df()
    result = cp.PanelRegression(
        data=df,
        formula="y ~ C(unit) + x",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=LinearRegression(),
    )
    plot_data = result.get_plot_data_ols()
    assert "prediction" in plot_data.columns
    assert "residual" in plot_data.columns
