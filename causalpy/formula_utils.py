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

import numpy as np
import pandas as pd
from patsy import (
    EvalEnvironment,
    EvalFactor,
    ModelDesc,
    Term,
    dmatrices,
)
from patsy import (
    build_design_matrices as patsy_build_design_matrices,
)

from causalpy.transforms import elapsed, ramp, step


def _datetime_columns(data: pd.DataFrame) -> set[str]:
    """Return datetime column names that can appear as bare Patsy factors."""
    return {
        column
        for column in data.columns
        if isinstance(column, str)
        and pd.api.types.is_datetime64_any_dtype(data[column])
    }


def _uses_pd_na_string_dtype(dtype: Any) -> bool:
    """Report whether ``dtype`` is a string dtype whose missing value is ``pd.NA``.

    Parameters
    ----------
    dtype : Any
        A pandas dtype, typically an entry of ``DataFrame.dtypes``.

    Returns
    -------
    bool
        ``True`` for nullable (``pd.NA``-sentinel) string dtypes, including
        Arrow-backed strings.
    """
    if isinstance(dtype, pd.StringDtype):
        # pandas 3's default inferred ``str`` dtype is also a StringDtype, but
        # it uses ``np.nan`` as its missing value and needs no conversion.
        return dtype.na_value is pd.NA
    return isinstance(dtype, pd.ArrowDtype) and pd.api.types.is_string_dtype(dtype)


def _normalize_patsy_data(data: pd.DataFrame) -> pd.DataFrame:
    """Convert ``pd.NA``-backed string columns to Patsy-compatible object columns.

    Patsy evaluates missingness with ``bool(value)``, which raises
    ``TypeError: boolean value of NA is ambiguous`` for nullable string columns
    (upstream Patsy #206). This affects both pandas 2.3 and pandas 3, and only
    for the ``pd.NA`` sentinel: pandas 3's default inferred ``str`` dtype uses
    ``np.nan`` and passes through untouched, as do categorical, numeric, object
    and datetime columns.

    Affected columns are rewritten to object dtype with ``np.nan`` for missing
    values on a frame this function owns, so the caller's DataFrame is never
    modified. Frames with no affected column are returned as-is, without a copy.
    """
    string_column_positions = [
        position
        for position, dtype in enumerate(data.dtypes)
        if _uses_pd_na_string_dtype(dtype)
    ]
    if not string_column_positions:
        return data

    # Positional access rather than by name: patsy is given whatever frame the
    # caller built, which may carry duplicate column labels.
    normalized_data = data.copy(deep=False)
    for position in string_column_positions:
        values = data.iloc[:, position].to_numpy(dtype=object, na_value=np.nan)
        # An explicit object Series, not a bare ndarray: pandas 3 would
        # otherwise re-infer its own string dtype from the object array.
        # ``isetitem`` accepts a Series at runtime; pandas-stubs omits it.
        normalized_data.isetitem(
            position,
            pd.Series(values, index=data.index, dtype=object),  # type: ignore[arg-type]
        )
    return normalized_data


def build_design_matrices(
    design_infos: list[Any], data: pd.DataFrame, **kwargs: Any
) -> list[Any]:
    """Build Patsy matrices from fitted design information.

    This applies the same extension-string normalization as
    :func:`build_formula_matrices`, so predictions using Patsy's fitted
    ``design_info`` remain compatible with pandas 2.3 and 3.

    Parameters
    ----------
    design_infos : list[Any]
        Patsy design information fitted on the training data.
    data : pd.DataFrame
        New data used to construct matrices with the fitted design information.
    **kwargs : Any
        Keyword arguments forwarded to :func:`patsy.build_design_matrices`.

    Returns
    -------
    list[Any]
        Patsy design matrices corresponding to ``design_infos``.
    """
    return patsy_build_design_matrices(
        design_infos, _normalize_patsy_data(data), **kwargs
    )


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
    data_for_patsy = _normalize_patsy_data(data)
    eval_env = EvalEnvironment.capture(1).with_outer_namespace(
        {"elapsed": elapsed, "ramp": ramp, "step": step}
    )
    return dmatrices(
        datetime_continuous_formula(formula, data_for_patsy),
        data_for_patsy,
        eval_env=eval_env,
        **kwargs,
    )
