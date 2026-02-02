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
"""
Utility functions
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from causalpy.experiments.synthetic_control import SyntheticControl


def _is_variable_dummy_coded(series: pd.Series) -> bool:
    """Check if a data in the provided Series is dummy coded. It should be 0 or 1
    only."""
    return len(set(series).difference({0, 1})) == 0


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


def convert_to_string(x: float | xr.DataArray, round_to: int | None = 2) -> str:
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


def check_convex_hull_violation(
    treated_series: np.ndarray, control_matrix: np.ndarray
) -> dict:
    """
    Check if treated series values fall within the range of control series.

    For each time point, verify that:
    min(controls) <= treated <= max(controls)

    This is a necessary (but not sufficient) condition for the treated unit
    to lie within the convex hull of control units.

    Parameters
    ----------
    treated_series : np.ndarray
        1D array of treated unit values (shape: n_timepoints)
    control_matrix : np.ndarray
        2D array of control unit values (shape: n_timepoints x n_controls)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'passes': bool - whether the check passes
        - 'n_violations': int - number of time points with violations
        - 'pct_above': float - percentage of points where treated > max(controls)
        - 'pct_below': float - percentage of points where treated < min(controls)

    Examples
    --------
    >>> treated = np.array([1.0, 2.0, 3.0])
    >>> controls = np.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]])
    >>> result = check_convex_hull_violation(treated, controls)
    >>> result["passes"]
    True
    """
    control_min = control_matrix.min(axis=1)
    control_max = control_matrix.max(axis=1)

    above = treated_series > control_max
    below = treated_series < control_min

    n_points = len(treated_series)
    if n_points == 0:
        return {
            "passes": True,
            "n_violations": 0,
            "pct_above": 0.0,
            "pct_below": 0.0,
        }
    return {
        "passes": not (above.any() or below.any()),
        "n_violations": int(above.sum() + below.sum()),
        "pct_above": float(100 * above.sum() / n_points),
        "pct_below": float(100 * below.sum() / n_points),
    }


def extract_lift_for_mmm(
    sc_result: SyntheticControl,
    channel: str,
    x: float,
    delta_x: float,
    aggregate: Literal["mean", "sum"] = "mean",
) -> pd.DataFrame:
    """
    Extract lift test results from a Synthetic Control analysis for MMM calibration.

    This function extracts lift estimates from a fitted SyntheticControl model in a
    format compatible with PyMC-Marketing's ``add_lift_test_measurements()`` method.
    This enables using geo-level lift test results to calibrate Media Mix Models.

    Parameters
    ----------
    sc_result : SyntheticControl
        A fitted SyntheticControl model with one or more treated units. The model
        must have been fit with a Bayesian (PyMC) model to provide posterior
        distributions for uncertainty quantification.
    channel : str
        Name of the marketing channel being tested (e.g., "tv", "radio", "digital").
        This should match the channel names used in your MMM.
    x : float
        Baseline spend level for the channel before the test period. For channels
        with zero pre-test spend, use 0.0.
    delta_x : float
        The change in spend during the test period (i.e., test spend minus baseline
        spend). For a new channel activation, this equals the total test spend.
    aggregate : {"mean", "sum"}, default="mean"
        How to aggregate the causal impact across post-intervention time periods:

        - "mean": Average lift per time period. Use this for rate-based outcomes
          (e.g., weekly sales rate) or when your MMM operates at the same time
          granularity as the experiment.
        - "sum": Total cumulative lift across all post-intervention periods. Use
          this for cumulative outcomes or when you want total campaign impact.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per treated geo, containing columns:

        - ``channel``: The marketing channel name (from input parameter)
        - ``geo``: The treated geo identifier (from sc_result.treated_units)
        - ``x``: Pre-test spend level (from input parameter)
        - ``delta_x``: Spend change during test (from input parameter)
        - ``delta_y``: Mean lift estimate from the posterior distribution
        - ``sigma``: Standard deviation of the lift estimate from the posterior

    Raises
    ------
    ValueError
        If the model is not a Bayesian (PyMC) model, as uncertainty quantification
        requires posterior samples.

    See Also
    --------
    PyMC-Marketing lift test calibration :
        https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_lift_test.html

    Notes
    -----
    This function is designed for integration with PyMC-Marketing's MMM calibration
    workflow. The output DataFrame can be passed directly to
    ``MMM.add_lift_test_measurements()`` to inform the model's saturation curves
    with experimental evidence.

    For more information on lift test calibration in MMMs, see the PyMC-Marketing
    documentation: https://github.com/pymc-labs/pymc-marketing

    Examples
    --------
    >>> import causalpy as cp  # doctest: +SKIP
    >>> # Fit a multi-geo synthetic control model
    >>> result = cp.SyntheticControl(  # doctest: +SKIP
    ...     df,
    ...     treatment_time,
    ...     control_units=["geo_a", "geo_b", "geo_c"],
    ...     treated_units=["geo_x", "geo_y"],
    ...     model=cp.pymc_models.WeightedSumFitter(
    ...         sample_kwargs={"progressbar": False}
    ...     ),
    ... )
    >>> # Extract lift results for MMM calibration
    >>> df_lift = cp.extract_lift_for_mmm(  # doctest: +SKIP
    ...     result,
    ...     channel="tv_campaign",
    ...     x=0.0,  # No pre-test TV spend
    ...     delta_x=50000,  # $50k test spend
    ...     aggregate="mean",
    ... )
    >>> # The resulting DataFrame can be used with PyMC-Marketing:
    >>> # mmm.add_lift_test_measurements(df_lift)  # doctest: +SKIP
    """
    from causalpy.pymc_models import PyMCModel

    # Validate that we have a Bayesian model
    if not isinstance(sc_result.model, PyMCModel):
        raise ValueError(
            "extract_lift_for_mmm requires a Bayesian (PyMC) model for uncertainty "
            "quantification. OLS models do not provide posterior distributions needed "
            "for the 'sigma' (uncertainty) column."
        )

    treated_units = sc_result.treated_units
    results = []

    for unit in treated_units:
        # Get posterior samples for this unit's causal impact
        unit_impact = sc_result.post_impact.sel(treated_units=unit)

        # Aggregate across time periods
        if aggregate == "mean":
            # Average lift per time period
            lift_samples = unit_impact.mean(dim="obs_ind")
        else:  # sum
            # Total cumulative lift
            lift_samples = unit_impact.sum(dim="obs_ind")

        # Extract mean and std from the posterior
        delta_y = float(lift_samples.mean().values)
        sigma = float(lift_samples.std().values)

        results.append(
            {
                "channel": channel,
                "geo": str(unit),
                "x": x,
                "delta_x": delta_x,
                "delta_y": delta_y,
                "sigma": sigma,
            }
        )

    return pd.DataFrame(results)
