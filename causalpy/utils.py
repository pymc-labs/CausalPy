#   Copyright 2022 - 2025 The PyMC Labs Developers
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
"""
Utility functions
"""

import re
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr


def _is_variable_dummy_coded(series: pd.Series) -> bool:
    """Check if a data in the provided Series is dummy coded. It should be 0 or 1
    only."""
    return len(set(series).difference(set([0, 1]))) == 0


def _series_has_2_levels(series: pd.Series) -> bool:
    """Check that the variable in the provided Series has 2 levels"""
    return len(pd.Categorical(series).categories) == 2


def round_num(n: float, round_to: int | None) -> str:
    """Return a string representing a number with significant figures.

    Parameters
    ----------
    n : float
        Number to round.
    round_to : int, optional
        Number of significant figures. If None, defaults to 2.

    Returns
    -------
    str
        String representation of the number with specified significant
        figures.
    """
    sig_figs = _format_sig_figs(n, round_to)
    return f"{n:.{sig_figs}g}"


def _format_sig_figs(value: float, default: int | None = None) -> int:
    """Get a default number of significant figures.

    Gives the integer part or `default`, whichever is bigger.

    Examples
    --------
    0.1234 --> 0.12
    1.234  --> 1.2
    12.34  --> 12
    123.4  --> 123
    """
    if default is None:
        default = 2
    if value == 0:
        return 1
    return max(int(np.log10(np.abs(value))) + 1, default)


def convert_to_string(x: Union[float, xr.DataArray], round_to: int | None = 2) -> str:
    """Convert numeric inputs to a formatted string representation.

    Parameters
    ----------
    x : float or xr.DataArray
        The numeric value or xarray DataArray to convert.
    round_to : int, optional
        Number of significant figures to round to. Defaults to 2.

    Returns
    -------
    str
        Formatted string representation. For floats, returns rounded
        decimal. For DataArrays, returns mean with 94% credible interval.

    Raises
    ------
    ValueError
        If `x` is neither a float nor an xarray DataArray.
    """
    if isinstance(x, float):
        # In the case of a float, we return the number rounded to 2 decimal places
        return f"{x:.2f}"
    elif isinstance(x, xr.DataArray):
        # In the case of an xarray object, we return the mean and 94% CI
        percentiles = x.quantile([0.03, 1 - 0.03]).values
        ci = (
            r"$CI_{94\%}$"
            + f"[{round_num(percentiles[0], round_to)}, {round_num(percentiles[1], round_to)}]"
        )
        return f"{x.mean().values:.2f}" + ci
    else:
        raise ValueError(
            "Type not supported. Please provide a float or an xarray object."
        )


def get_interaction_terms(formula: str) -> list[str]:
    """
    Extract interaction terms from a statistical model formula.

    Parameters
    ----------
    formula : str
        A statistical model formula string (e.g., "y ~ x1 + x2*x3")

    Returns
    -------
    list[str]
        A list of interaction terms (those containing '*' or ':')

    Examples
    --------
    >>> get_interaction_terms("y ~ 1 + x1 + x2*x3")
    ['x2*x3']
    >>> get_interaction_terms("y ~ x1:x2 + x3")
    ['x1:x2']
    >>> get_interaction_terms("y ~ x1 + x2 + x3")
    []
    """
    # Define interaction indicators
    INTERACTION_INDICATORS = ["*", ":"]

    # Remove whitespace
    formula_clean = formula.replace(" ", "")

    # Extract right-hand side of the formula
    rhs = formula_clean.split("~")[1]

    # Split terms by '+' or '-' while keeping them intact
    terms = re.split(r"(?=[+-])", rhs)

    # Clean up terms and get interaction terms (those with '*' or ':')
    interaction_terms = []
    for term in terms:
        # Remove leading + or - for processing
        clean_term = term.lstrip("+-")
        if any(indicator in clean_term for indicator in INTERACTION_INDICATORS):
            interaction_terms.append(clean_term)

    return interaction_terms
