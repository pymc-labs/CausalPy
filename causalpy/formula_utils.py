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
"""Helpers for building Patsy design matrices from CausalPy formulas."""

from __future__ import annotations

from typing import Any

import pandas as pd
from patsy import EvalEnvironment, EvalFactor, ModelDesc, Term, dmatrices

from causalpy.transforms import elapsed, ramp, step


def _datetime_columns(data: pd.DataFrame) -> set[str]:
    """Return datetime column names that can appear as bare Patsy factors."""
    return {
        column
        for column in data.columns
        if isinstance(column, str)
        and pd.api.types.is_datetime64_any_dtype(data[column])
    }


def _rewrite_datetime_terms(
    terms: list[Term], datetime_columns: set[str]
) -> list[Term]:
    """Replace bare datetime factors with the stateful elapsed-time transform."""
    return [
        Term(
            [
                EvalFactor(f"elapsed({factor.code})")
                if factor.code in datetime_columns
                else factor
                for factor in term.factors
            ]
        )
        for term in terms
    ]


def datetime_continuous_formula(formula: str, data: pd.DataFrame) -> ModelDesc:
    """Make bare datetime predictors continuous while preserving explicit transforms.

    Bare datetime factors such as ``date`` become ``elapsed(date)``. Expressions
    such as ``C(date)``, ``step(date, ...)``, and ``ramp(date, ...)`` are preserved
    exactly, so users retain Patsy's explicit categorical syntax and CausalPy's
    existing datetime intervention transforms.

    Parameters
    ----------
    formula : str
        Patsy formula to rewrite.
    data : pd.DataFrame
        Data used to identify datetime columns.
    """
    model_desc = ModelDesc.from_formula(formula)
    return ModelDesc(
        model_desc.lhs_termlist,
        _rewrite_datetime_terms(model_desc.rhs_termlist, _datetime_columns(data)),
    )


def build_formula_matrices(
    formula: str, data: pd.DataFrame, **kwargs: Any
) -> tuple[Any, Any]:
    """Build Patsy matrices with bare datetime predictors encoded as elapsed days.

    The stateful ``elapsed`` transform stores its origin in Patsy's ``design_info``.
    Calls to :func:`patsy.build_design_matrices` therefore use the same fitted origin
    for new rows.

    Parameters
    ----------
    formula : str
        Patsy formula to evaluate.
    data : pd.DataFrame
        Data used to build the design matrices.
    **kwargs : Any
        Keyword arguments forwarded to :func:`patsy.dmatrices`.
    """
    eval_env = EvalEnvironment.capture(1).with_outer_namespace(
        {"elapsed": elapsed, "ramp": ramp, "step": step}
    )
    return dmatrices(
        datetime_continuous_formula(formula, data),
        data,
        eval_env=eval_env,
        **kwargs,
    )
