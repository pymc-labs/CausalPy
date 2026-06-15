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
"""Engine-agnostic formula design transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from patsy import DesignInfo, build_design_matrices, dmatrices


class DesignTransform(Protocol):
    """Rebuild design matrices for new data without exposing the formula engine."""

    labels: list[str]
    outcome_name: str | None

    def transform_x(
        self,
        new_data: pd.DataFrame,
        *,
        return_type: Literal["matrix", "dataframe"] = "matrix",
    ) -> np.ndarray | pd.DataFrame:
        """Rebuild predictor design matrices for new data.

        Parameters
        ----------
        new_data : pandas.DataFrame
            Data used to rebuild the design matrix.
        return_type : {"matrix", "dataframe"}, default "matrix"
            Return format for the rebuilt predictors.
        """

    def transform_y(self, new_data: pd.DataFrame) -> np.ndarray:
        """Rebuild the outcome vector for new data.

        Parameters
        ----------
        new_data : pandas.DataFrame
            Data used to rebuild the outcome vector.
        """


@dataclass
class PatsyDesignTransform:
    """Patsy-backed :class:`DesignTransform`."""

    _x_design_info: DesignInfo
    _y_design_info: DesignInfo | None
    labels: list[str]
    outcome_name: str | None = None

    def transform_x(
        self,
        new_data: pd.DataFrame,
        *,
        return_type: Literal["matrix", "dataframe"] = "matrix",
    ) -> np.ndarray | pd.DataFrame:
        """Rebuild predictor design matrices for new data.

        Parameters
        ----------
        new_data : pandas.DataFrame
            Data used to rebuild the design matrix.
        return_type : {"matrix", "dataframe"}, default "matrix"
            Return format for the rebuilt predictors.
        """
        (new_x,) = build_design_matrices(
            [self._x_design_info], new_data, return_type=return_type
        )
        if return_type == "dataframe":
            return new_x
        return np.asarray(new_x)

    def transform_y(self, new_data: pd.DataFrame) -> np.ndarray:
        """Rebuild the outcome vector for new data.

        Parameters
        ----------
        new_data : pandas.DataFrame
            Data used to rebuild the outcome vector.
        """
        if self._y_design_info is None:
            msg = "No outcome design metadata; transform_y is unavailable."
            raise ValueError(msg)
        (new_y,) = build_design_matrices([self._y_design_info], new_data)
        return np.asarray(new_y)


def build_patsy_design(
    formula: str, data: pd.DataFrame
) -> tuple[PatsyDesignTransform, np.ndarray, np.ndarray]:
    """Build a design transform and raw fit arrays from a patsy formula.

    Parameters
    ----------
    formula : str
        Patsy model formula.
    data : pandas.DataFrame
        Input data used to fit the design metadata.
    """
    # eval_env=1 resolves custom transforms (step/ramp) from the caller's module.
    y, X = dmatrices(formula, data, eval_env=1)
    transform = PatsyDesignTransform(
        _x_design_info=X.design_info,
        _y_design_info=y.design_info,
        labels=list(X.design_info.column_names),
        outcome_name=y.design_info.column_names[0],
    )
    return transform, np.asarray(X), np.asarray(y)


def build_patsy_formula_sides(
    formula: str, data: pd.DataFrame
) -> tuple[PatsyDesignTransform, PatsyDesignTransform, np.ndarray, np.ndarray]:
    """Build LHS/RHS transforms from a single patsy formula.

    Parameters
    ----------
    formula : str
        Patsy model formula.
    data : pandas.DataFrame
        Input data used to fit the design metadata.
    """
    lhs, rhs = dmatrices(formula, data, eval_env=1)
    treatment = PatsyDesignTransform(
        _x_design_info=lhs.design_info,
        _y_design_info=None,
        labels=[lhs.design_info.column_names[0]],
        outcome_name=lhs.design_info.column_names[0],
    )
    covariates = PatsyDesignTransform(
        _x_design_info=rhs.design_info,
        _y_design_info=None,
        labels=list(rhs.design_info.column_names),
        outcome_name=None,
    )
    return treatment, covariates, np.asarray(lhs), np.asarray(rhs)
