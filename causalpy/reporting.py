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
Reporting utilities for causal inference experiments.

This module provides statistical summaries and prose reports for causal effects.
The reporting functions automatically compute appropriate statistics based on the
model type (Bayesian/PyMC or Frequentist/OLS).

For detailed explanations of the reported statistics (HDI, ROPE, p-values, etc.)
and their interpretation, see the documentation:
https://causalpy.readthedocs.io/en/latest/knowledgebase/reporting_statistics.html
"""

from dataclasses import dataclass
from typing import Literal, Optional, Union

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import t


@dataclass
class EffectSummary:
    """Container for effect summary statistics and prose report.

    Attributes
    ----------
    table : pd.DataFrame
        DataFrame containing summary statistics (mean, median, HDI, tail probabilities)
    text : str
        Formatted prose summary of the effect
    """

    table: pd.DataFrame
    text: str


# ==============================================================================
# Helper functions for common operations
# ==============================================================================


def _extract_hdi_bounds(
    hdi_result: Union[xr.Dataset, xr.DataArray], hdi_prob: float = 0.95
) -> tuple[float, float]:
    """Extract HDI lower and upper bounds from arviz.hdi result.

    Handles both Dataset (when arviz returns Dataset) and DataArray formats.

    Parameters
    ----------
    hdi_result : xr.Dataset or xr.DataArray
        Result from arviz.hdi()
    hdi_prob : float
        HDI probability (not used in extraction but kept for signature consistency)

    Returns
    -------
    tuple[float, float]
        Lower and upper HDI bounds
    """
    if isinstance(hdi_result, xr.Dataset):
        hdi_data = list(hdi_result.data_vars.values())[0]
        lower = float(hdi_data.sel(hdi="lower").values)
        upper = float(hdi_data.sel(hdi="higher").values)
    else:
        lower = float(hdi_result.sel(hdi="lower").values)
        upper = float(hdi_result.sel(hdi="higher").values)
    return lower, upper


def _compute_tail_probabilities(
    effect: xr.DataArray, direction: Literal["increase", "decrease", "two-sided"]
) -> dict[str, float]:
    """Compute tail probabilities based on direction.

    Parameters
    ----------
    effect : xr.DataArray
        Effect posterior draws
    direction : {"increase", "decrease", "two-sided"}
        Direction for tail probability

    Returns
    -------
    dict[str, float]
        Dictionary with keys: 'p_gt_0', 'p_lt_0', or 'p_two_sided'+'prob_of_effect'
    """
    if direction == "increase":
        return {"p_gt_0": float((effect > 0).mean().values)}
    elif direction == "decrease":
        return {"p_lt_0": float((effect < 0).mean().values)}
    else:  # two-sided
        p_gt = float((effect > 0).mean().values)
        p_lt = float((effect < 0).mean().values)
        p_two_sided = 2 * min(p_gt, p_lt)
        return {"p_two_sided": p_two_sided, "prob_of_effect": 1 - p_two_sided}


def _compute_rope_probability(
    effect: xr.DataArray,
    min_effect: float,
    direction: Literal["increase", "decrease", "two-sided"],
) -> float:
    """Compute Region of Practical Equivalence probability.

    Parameters
    ----------
    effect : xr.DataArray
        Effect posterior draws
    min_effect : float
        Minimum effect size threshold
    direction : {"increase", "decrease", "two-sided"}
        Direction for ROPE calculation

    Returns
    -------
    float
        Probability that effect exceeds min_effect threshold
    """
    if direction == "two-sided":
        return float((np.abs(effect) > min_effect).mean().values)
    elif direction == "increase":
        return float((effect > min_effect).mean().values)
    elif direction == "decrease":
        return float((effect < -min_effect).mean().values)


def _format_number(x: float, decimals: int = 2) -> str:
    """Format number for prose output.

    Parameters
    ----------
    x : float
        Number to format
    decimals : int
        Number of decimal places

    Returns
    -------
    str
        Formatted number string
    """
    return f"{x:.{decimals}f}"


# ==============================================================================
# Unified scalar effect statistics (DiD, RD, RKink)
# ==============================================================================


def _compute_statistics_scalar(
    effect: xr.DataArray,
    hdi_prob: float = 0.95,
    direction: Literal["increase", "decrease", "two-sided"] = "increase",
    min_effect: Optional[float] = None,
) -> dict[str, float]:
    """Compute statistics for scalar causal effects (DiD, RD, RKink).

    Works for any scalar effect with posterior draws (chain, draw dimensions).

    Parameters
    ----------
    effect : xr.DataArray
        Scalar effect with posterior draws (must have chain, draw dimensions)
    hdi_prob : float
        Probability for HDI interval
    direction : {"increase", "decrease", "two-sided"}
        Direction for tail probability calculation
    min_effect : float, optional
        Minimum effect size for ROPE analysis

    Returns
    -------
    dict[str, float]
        Dictionary containing mean, median, HDI bounds, tail probabilities, and optionally ROPE
    """
    stats = {
        "mean": float(effect.mean(dim=["chain", "draw"]).values),
        "median": float(effect.median(dim=["chain", "draw"]).values),
    }

    # HDI using helper
    hdi_result = az.hdi(effect, hdi_prob=hdi_prob)
    stats["hdi_lower"], stats["hdi_upper"] = _extract_hdi_bounds(hdi_result)

    # Tail probabilities using helper
    stats.update(_compute_tail_probabilities(effect, direction))

    # ROPE using helper
    if min_effect is not None:
        stats["p_rope"] = _compute_rope_probability(effect, min_effect, direction)

    return stats


def _generate_table_scalar(
    stats: dict[str, float], index_name: str = "effect"
) -> pd.DataFrame:
    """Generate summary table for scalar effects (DiD, RD, RKink).

    Parameters
    ----------
    stats : dict[str, float]
        Statistics dictionary from _compute_statistics_scalar()
    index_name : str
        Name for the table index (e.g., "treatment_effect", "discontinuity")

    Returns
    -------
    pd.DataFrame
        Summary table with one row
    """
    row = {
        "mean": stats["mean"],
        "median": stats["median"],
        "hdi_lower": stats["hdi_lower"],
        "hdi_upper": stats["hdi_upper"],
    }

    # Add tail probabilities (whichever are present)
    for key in ["p_gt_0", "p_lt_0", "p_two_sided", "prob_of_effect", "p_rope"]:
        if key in stats:
            row[key] = stats[key]

    return pd.DataFrame([row], index=[index_name])


def _generate_prose_scalar(
    stats: dict[str, float],
    effect_name: str,
    alpha: float = 0.05,
    direction: Literal["increase", "decrease", "two-sided"] = "increase",
) -> str:
    """Generate prose summary for scalar effects.

    Parameters
    ----------
    stats : dict[str, float]
        Statistics dictionary from _compute_statistics_scalar()
    effect_name : str
        Name of the effect for prose (e.g., "average treatment effect",
        "discontinuity at threshold", "change in gradient at the kink point")
    alpha : float
        Significance level for HDI interval
    direction : {"increase", "decrease", "two-sided"}
        Direction for tail probability

    Returns
    -------
    str
        Prose summary of the effect
    """
    hdi_pct = int((1 - alpha) * 100)
    mean = stats["mean"]
    lower = stats["hdi_lower"]
    upper = stats["hdi_upper"]

    # Direction-specific text
    if direction == "increase":
        p_val = stats.get("p_gt_0", 0.0)
        direction_text = "increase"
    elif direction == "decrease":
        p_val = stats.get("p_lt_0", 0.0)
        direction_text = "decrease"
    else:
        p_val = stats.get("prob_of_effect", 0.0)
        direction_text = "effect"

    prose = (
        f"The {effect_name} was {_format_number(mean)} "
        f"({hdi_pct}% HDI [{_format_number(lower)}, {_format_number(upper)}]), "
        f"with a posterior probability of an {direction_text} of {_format_number(p_val, 3)}."
    )

    return prose


def _detect_experiment_type(result):
    """Detect experiment type from result attributes."""
    if hasattr(result, "discontinuity_at_threshold"):
        return "rd"  # Regression Discontinuity
    elif hasattr(result, "gradient_change"):
        return "rkink"  # Regression Kink
    elif hasattr(result, "causal_impact") and not hasattr(result, "post_impact"):
        return "did"  # Difference-in-Differences or ANCOVA/PrePostNEGD
    elif hasattr(result, "post_impact"):
        return "its_or_sc"  # ITS or Synthetic Control
    else:
        raise ValueError(
            "Unknown experiment type. Result must have 'discontinuity_at_threshold' (RD), "
            "'gradient_change' (Regression Kink), 'causal_impact' (DiD/ANCOVA), "
            "or 'post_impact' (ITS/Synthetic Control) attribute."
        )


def _effect_summary_did(
    result,
    direction: Literal["increase", "decrease", "two-sided"] = "increase",
    alpha: float = 0.05,
    min_effect: Optional[float] = None,
):
    """Generate effect summary for Difference-in-Differences experiments."""
    causal_impact = result.causal_impact

    # For DiD, causal_impact should be an xarray.DataArray with posterior draws
    if not isinstance(causal_impact, xr.DataArray):
        raise ValueError(
            "For DiD experiments, causal_impact must be an xarray.DataArray with "
            "posterior draws. OLS models are not supported for uncertainty quantification. "
            "Please use a PyMC model."
        )

    # Compute statistics using unified function
    hdi_prob = 1 - alpha
    stats = _compute_statistics_scalar(
        causal_impact, hdi_prob=hdi_prob, direction=direction, min_effect=min_effect
    )

    # Generate table and prose using unified functions
    table = _generate_table_scalar(stats, index_name="treatment_effect")
    text = _generate_prose_scalar(
        stats, "average treatment effect", alpha=alpha, direction=direction
    )

    return EffectSummary(table=table, text=text)


def _effect_summary_rd(
    result,
    direction: Literal["increase", "decrease", "two-sided"] = "increase",
    alpha: float = 0.05,
    min_effect: Optional[float] = None,
):
    """Generate effect summary for Regression Discontinuity experiments."""
    discontinuity = result.discontinuity_at_threshold

    # Check if PyMC (xarray) or OLS (scalar)
    is_pymc = isinstance(discontinuity, xr.DataArray)

    if is_pymc:
        # PyMC model: use unified scalar functions
        hdi_prob = 1 - alpha
        stats = _compute_statistics_scalar(
            discontinuity, hdi_prob=hdi_prob, direction=direction, min_effect=min_effect
        )
        table = _generate_table_scalar(stats, index_name="discontinuity")
        text = _generate_prose_scalar(
            stats, "discontinuity at threshold", alpha=alpha, direction=direction
        )
    else:
        # OLS model: calculate from model
        stats = _compute_statistics_rd_ols(result, alpha=alpha)
        table = _generate_table_rd_ols(stats)
        text = _generate_prose_rd_ols(stats, alpha=alpha)

    return EffectSummary(table=table, text=text)


# ==============================================================================
# Window and counterfactual extraction helpers
# ==============================================================================


def _select_treated_unit(
    data: xr.DataArray, treated_unit: Optional[str]
) -> xr.DataArray:
    """Select a specific treated unit from multi-unit xarray data.

    Parameters
    ----------
    data : xr.DataArray
        Data with treated_units dimension
    treated_unit : str or None
        Name of treated unit to select. If None, selects first unit.

    Returns
    -------
    xr.DataArray
        Data for the selected treated unit
    """
    # Validate coordinate/dimension size match
    if "treated_units" in data.dims:
        coord_len = len(data.coords["treated_units"])
        dim_size = data.sizes["treated_units"]
        if coord_len != dim_size:
            # Shape mismatch - slice to match coordinates
            data = data.isel(treated_units=slice(0, coord_len))

    if treated_unit is not None:
        return data.sel(treated_units=treated_unit)
    else:
        return data.isel(treated_units=0)


def _select_treated_unit_numpy(
    data: np.ndarray, result, treated_unit: Optional[str]
) -> np.ndarray:
    """Select a specific treated unit from multi-dimensional numpy array.

    Parameters
    ----------
    data : np.ndarray
        Multi-dimensional array where second dimension is treated units
    result
        Experiment result object with treated_units attribute
    treated_unit : str or None
        Name of treated unit to select. If None, selects first unit.

    Returns
    -------
    np.ndarray
        Data for the selected treated unit (1D)
    """
    if treated_unit is not None and hasattr(result, "treated_units"):
        unit_idx = result.treated_units.index(treated_unit)
        return data[:, unit_idx]
    else:
        return data[:, 0]


def _extract_window(result, window, treated_unit=None):
    """Extract windowed impact data based on window specification.

    Assumes result.post_impact is properly shaped xarray or numpy array.

    Parameters
    ----------
    result
        Experiment result object with post_impact and datapost attributes
    window : str, tuple, or slice
        Window specification: "post", (start, end) tuple, or slice object
    treated_unit : str, optional
        For multi-unit experiments, specify which treated unit to analyze

    Returns
    -------
    tuple
        (windowed_impact, window_coords) where windowed_impact is the data
        and window_coords is the corresponding index
    """
    post_impact = result.post_impact

    # Check if PyMC (xarray with chain/draw dims) or OLS
    is_pymc = isinstance(post_impact, xr.DataArray) and (
        "chain" in post_impact.dims or "draw" in post_impact.dims
    )

    # Handle treated_unit selection using helper functions
    if isinstance(post_impact, xr.DataArray) and "treated_units" in post_impact.dims:
        post_impact = _select_treated_unit(post_impact, treated_unit)
    elif (
        not isinstance(post_impact, xr.DataArray)
        and hasattr(post_impact, "ndim")
        and post_impact.ndim > 1
    ):
        post_impact = _select_treated_unit_numpy(post_impact, result, treated_unit)

    # Convert OLS xarray to numpy for consistent handling
    if not is_pymc and isinstance(post_impact, xr.DataArray):
        post_impact = np.squeeze(post_impact.values)

    # Ensure OLS data is numpy array
    if not is_pymc and not isinstance(post_impact, np.ndarray):
        post_impact = np.asarray(post_impact)

    # Extract window coordinates based on window specification
    if window == "post":
        # Use all post-treatment time points
        window_coords = result.datapost.index
    elif isinstance(window, tuple) and len(window) == 2:
        # Handle (start, end) tuple
        start, end = window
        if isinstance(result.datapost.index, pd.DatetimeIndex):
            # Datetime index - convert to timestamps if needed
            if not isinstance(start, pd.Timestamp):
                start = pd.Timestamp(start)
            if not isinstance(end, pd.Timestamp):
                end = pd.Timestamp(end)
            window_coords = result.datapost.index[
                (result.datapost.index >= start) & (result.datapost.index <= end)
            ]
        else:
            # Integer index - filter by value
            start_val = int(start)
            end_val = int(end)
            mask = (result.datapost.index >= start_val) & (
                result.datapost.index <= end_val
            )
            window_coords = result.datapost.index[mask]
    elif isinstance(window, slice):
        # Handle slice object
        if isinstance(result.datapost.index, pd.DatetimeIndex):
            # For datetime, slice works directly
            window_coords = result.datapost.index[window]
        else:
            # For integer indices, convert slice to value-based filtering
            start_val = (
                int(window.start)
                if window.start is not None
                else result.datapost.index.min()
            )
            stop_val = (
                int(window.stop)
                if window.stop is not None
                else result.datapost.index.max() + 1
            )
            step = int(window.step) if window.step is not None else 1
            # Create boolean mask for values in range
            mask = (result.datapost.index >= start_val) & (
                result.datapost.index < stop_val
            )
            window_coords = result.datapost.index[mask][::step]
    else:
        raise ValueError(
            f"window must be 'post', a tuple (start, end), or a slice. Got {type(window)}"
        )

    # Apply window selection to post_impact
    if window == "post":
        # No filtering needed - use all data
        windowed_impact = post_impact
    elif is_pymc:
        # PyMC: use xarray's named dimension selection
        windowed_impact = post_impact.sel(obs_ind=window_coords)
    else:
        # OLS: convert window_coords to integer indices and select from numpy array
        indices = [result.datapost.index.get_loc(coord) for coord in window_coords]
        windowed_impact = post_impact[indices]

    # Validate window is not empty
    if len(window_coords) == 0:
        raise ValueError("Window contains no time points")

    return windowed_impact, window_coords


def _extract_counterfactual(result, window_coords, treated_unit=None):
    """Extract counterfactual predictions for the window.

    Reuses logic from _extract_window for consistency.

    Parameters
    ----------
    result
        Experiment result object with post_pred attribute
    window_coords : pd.Index
        Window coordinates from _extract_window
    treated_unit : str, optional
        For multi-unit experiments, specify which treated unit to analyze

    Returns
    -------
    xr.DataArray or np.ndarray
        Counterfactual predictions for the window
    """
    post_pred = result.post_pred

    # PyMC: Extract from InferenceData
    if hasattr(post_pred, "posterior_predictive"):
        # PyMC model - InferenceData object
        counterfactual = post_pred.posterior_predictive["mu"]

        # Handle treated_unit selection using helper
        if "treated_units" in counterfactual.dims:
            counterfactual = _select_treated_unit(counterfactual, treated_unit)

        # Select window using named dimension
        counterfactual = counterfactual.sel(obs_ind=window_coords)
        return counterfactual

    elif isinstance(post_pred, dict) and "posterior_predictive" in post_pred:
        # PyMC model - dict format (fallback)
        counterfactual = post_pred["posterior_predictive"]["mu"]

        # Handle treated_unit selection using helper
        if "treated_units" in counterfactual.dims:
            counterfactual = _select_treated_unit(counterfactual, treated_unit)

        # Select window using named dimension
        counterfactual = counterfactual.sel(obs_ind=window_coords)
        return counterfactual

    # OLS: Handle xarray or numpy
    if isinstance(post_pred, xr.DataArray):
        # OLS with xarray (e.g., SyntheticControl)
        # Select treated_unit using helper
        if "treated_units" in post_pred.dims:
            post_pred = _select_treated_unit(post_pred, treated_unit)

        # Convert window_coords to integer indices for isel
        indices = [result.datapost.index.get_loc(coord) for coord in window_coords]
        counterfactual = post_pred.isel(obs_ind=indices).values
        return np.squeeze(counterfactual)
    else:
        # OLS with numpy array
        # Convert window_coords to indices
        indices = [result.datapost.index.get_loc(coord) for coord in window_coords]
        counterfactual = post_pred[indices]

        # Handle treated_unit for multi-unit numpy arrays using helper
        if hasattr(counterfactual, "ndim") and counterfactual.ndim > 1:
            counterfactual = _select_treated_unit_numpy(
                counterfactual, result, treated_unit
            )

        return np.squeeze(counterfactual)


def _compute_statistics(
    impact,
    counterfactual,
    hdi_prob=0.95,
    direction="increase",
    cumulative=True,
    relative=True,
    min_effect=None,
):
    """Compute all summary statistics from posterior draws."""
    stats = {}

    # Average effect over window
    avg_effect = impact.mean(dim="obs_ind")
    stats["avg"] = {
        "mean": float(avg_effect.mean(dim=["chain", "draw"]).values),
        "median": float(avg_effect.median(dim=["chain", "draw"]).values),
    }

    # HDI for average
    hdi_avg = az.hdi(avg_effect, hdi_prob=hdi_prob)
    # Extract lower and upper bounds from HDI Dataset
    # Handle both Dataset and DataArray returns
    if isinstance(hdi_avg, xr.Dataset):
        hdi_data = list(hdi_avg.data_vars.values())[0]
        stats["avg"]["hdi_lower"] = float(hdi_data.sel(hdi="lower").values)
        stats["avg"]["hdi_upper"] = float(hdi_data.sel(hdi="higher").values)
    else:
        # If it's a DataArray, extract directly
        stats["avg"]["hdi_lower"] = float(hdi_avg.sel(hdi="lower").values)
        stats["avg"]["hdi_upper"] = float(hdi_avg.sel(hdi="higher").values)

    # Tail probabilities for average
    if direction == "increase":
        stats["avg"]["p_gt_0"] = float((avg_effect > 0).mean().values)
    elif direction == "decrease":
        stats["avg"]["p_lt_0"] = float((avg_effect < 0).mean().values)
    else:  # two-sided
        p_gt = float((avg_effect > 0).mean().values)
        p_lt = float((avg_effect < 0).mean().values)
        p_two_sided = 2 * min(p_gt, p_lt)
        stats["avg"]["p_two_sided"] = p_two_sided
        stats["avg"]["prob_of_effect"] = 1 - p_two_sided

    # ROPE for average
    if min_effect is not None:
        if direction == "two-sided":
            stats["avg"]["p_rope"] = float(
                (np.abs(avg_effect) > min_effect).mean().values
            )
        else:
            stats["avg"]["p_rope"] = float((avg_effect > min_effect).mean().values)

    # Cumulative effect
    if cumulative:
        # Use cumulative sum over window
        cum_effect = impact.cumsum(dim="obs_ind")
        # Take final value (cumulative over entire window)
        cum_final = cum_effect.isel(obs_ind=-1)

        stats["cum"] = {
            "mean": float(cum_final.mean(dim=["chain", "draw"]).values),
            "median": float(cum_final.median(dim=["chain", "draw"]).values),
        }

        # HDI for cumulative
        hdi_cum = az.hdi(cum_final, hdi_prob=hdi_prob)
        if isinstance(hdi_cum, xr.Dataset):
            hdi_cum_data = list(hdi_cum.data_vars.values())[0]
            stats["cum"]["hdi_lower"] = float(hdi_cum_data.sel(hdi="lower").values)
            stats["cum"]["hdi_upper"] = float(hdi_cum_data.sel(hdi="higher").values)
        else:
            stats["cum"]["hdi_lower"] = float(hdi_cum.sel(hdi="lower").values)
            stats["cum"]["hdi_upper"] = float(hdi_cum.sel(hdi="higher").values)

        # Tail probabilities for cumulative
        if direction == "increase":
            stats["cum"]["p_gt_0"] = float((cum_final > 0).mean().values)
        elif direction == "decrease":
            stats["cum"]["p_lt_0"] = float((cum_final < 0).mean().values)
        else:  # two-sided
            p_gt = float((cum_final > 0).mean().values)
            p_lt = float((cum_final < 0).mean().values)
            p_two_sided = 2 * min(p_gt, p_lt)
            stats["cum"]["p_two_sided"] = p_two_sided
            stats["cum"]["prob_of_effect"] = 1 - p_two_sided

        # ROPE for cumulative
        if min_effect is not None:
            if direction == "two-sided":
                stats["cum"]["p_rope"] = float(
                    (np.abs(cum_final) > min_effect).mean().values
                )
            else:
                stats["cum"]["p_rope"] = float((cum_final > min_effect).mean().values)

    # Relative effects
    if relative:
        epsilon = 1e-8  # Guard against division by zero
        counterfactual_mean = counterfactual.mean(dim="obs_ind")
        rel_avg = (avg_effect / (counterfactual_mean + epsilon)) * 100

        stats["avg"]["relative_mean"] = float(
            rel_avg.mean(dim=["chain", "draw"]).values
        )

        hdi_rel_avg = az.hdi(rel_avg, hdi_prob=hdi_prob)
        if isinstance(hdi_rel_avg, xr.Dataset):
            hdi_rel_avg_data = list(hdi_rel_avg.data_vars.values())[0]
            stats["avg"]["relative_hdi_lower"] = float(
                hdi_rel_avg_data.sel(hdi="lower").values
            )
            stats["avg"]["relative_hdi_upper"] = float(
                hdi_rel_avg_data.sel(hdi="higher").values
            )
        else:
            stats["avg"]["relative_hdi_lower"] = float(
                hdi_rel_avg.sel(hdi="lower").values
            )
            stats["avg"]["relative_hdi_upper"] = float(
                hdi_rel_avg.sel(hdi="higher").values
            )

        if cumulative:
            # Relative cumulative: (cumulative effect / cumulative counterfactual) * 100
            counterfactual_cum = counterfactual.cumsum(dim="obs_ind").isel(obs_ind=-1)
            rel_cum = (cum_final / (counterfactual_cum + epsilon)) * 100

            stats["cum"]["relative_mean"] = float(
                rel_cum.mean(dim=["chain", "draw"]).values
            )

            hdi_rel_cum = az.hdi(rel_cum, hdi_prob=hdi_prob)
            if isinstance(hdi_rel_cum, xr.Dataset):
                hdi_rel_cum_data = list(hdi_rel_cum.data_vars.values())[0]
                stats["cum"]["relative_hdi_lower"] = float(
                    hdi_rel_cum_data.sel(hdi="lower").values
                )
                stats["cum"]["relative_hdi_upper"] = float(
                    hdi_rel_cum_data.sel(hdi="higher").values
                )
            else:
                stats["cum"]["relative_hdi_lower"] = float(
                    hdi_rel_cum.sel(hdi="lower").values
                )
                stats["cum"]["relative_hdi_upper"] = float(
                    hdi_rel_cum.sel(hdi="higher").values
                )

    return stats


def _generate_table(stats, cumulative=True, relative=True):
    """Generate DataFrame table from statistics."""
    rows = []
    row_names = []

    # Average row
    avg_row = {
        "mean": stats["avg"]["mean"],
        "median": stats["avg"]["median"],
        "hdi_lower": stats["avg"]["hdi_lower"],
        "hdi_upper": stats["avg"]["hdi_upper"],
    }

    # Add tail probabilities
    if "p_gt_0" in stats["avg"]:
        avg_row["p_gt_0"] = stats["avg"]["p_gt_0"]
    if "p_lt_0" in stats["avg"]:
        avg_row["p_lt_0"] = stats["avg"]["p_lt_0"]
    if "p_two_sided" in stats["avg"]:
        avg_row["p_two_sided"] = stats["avg"]["p_two_sided"]
        avg_row["prob_of_effect"] = stats["avg"]["prob_of_effect"]

    # Add ROPE
    if "p_rope" in stats["avg"]:
        avg_row["p_rope"] = stats["avg"]["p_rope"]

    # Add relative
    if relative and "relative_mean" in stats["avg"]:
        avg_row["relative_mean"] = stats["avg"]["relative_mean"]
        avg_row["relative_hdi_lower"] = stats["avg"]["relative_hdi_lower"]
        avg_row["relative_hdi_upper"] = stats["avg"]["relative_hdi_upper"]

    rows.append(avg_row)
    row_names.append("average")

    # Cumulative row
    if cumulative:
        cum_row = {
            "mean": stats["cum"]["mean"],
            "median": stats["cum"]["median"],
            "hdi_lower": stats["cum"]["hdi_lower"],
            "hdi_upper": stats["cum"]["hdi_upper"],
        }

        # Add tail probabilities
        if "p_gt_0" in stats["cum"]:
            cum_row["p_gt_0"] = stats["cum"]["p_gt_0"]
        if "p_lt_0" in stats["cum"]:
            cum_row["p_lt_0"] = stats["cum"]["p_lt_0"]
        if "p_two_sided" in stats["cum"]:
            cum_row["p_two_sided"] = stats["cum"]["p_two_sided"]
            cum_row["prob_of_effect"] = stats["cum"]["prob_of_effect"]

        # Add ROPE
        if "p_rope" in stats["cum"]:
            cum_row["p_rope"] = stats["cum"]["p_rope"]

        # Add relative
        if relative and "relative_mean" in stats["cum"]:
            cum_row["relative_mean"] = stats["cum"]["relative_mean"]
            cum_row["relative_hdi_lower"] = stats["cum"]["relative_hdi_lower"]
            cum_row["relative_hdi_upper"] = stats["cum"]["relative_hdi_upper"]

        rows.append(cum_row)
        row_names.append("cumulative")

    df = pd.DataFrame(rows, index=row_names)
    return df


def _generate_prose(
    stats,
    window_coords,
    alpha=0.05,
    direction="increase",
    cumulative=True,
    relative=True,
):
    """Generate prose summary from statistics."""
    hdi_pct = int((1 - alpha) * 100)

    # Format window string
    if len(window_coords) > 0:
        start_str = str(window_coords[0])
        end_str = str(window_coords[-1])
        window_str = f"{start_str} to {end_str}"
    else:
        window_str = "post-period"

    # Average effect prose
    avg_mean = stats["avg"]["mean"]
    avg_lower = stats["avg"]["hdi_lower"]
    avg_upper = stats["avg"]["hdi_upper"]

    # Format numbers
    def fmt_num(x, decimals=2):
        return f"{x:.{decimals}f}"

    # Tail probability text
    if direction == "increase":
        p_val = stats["avg"].get("p_gt_0", 0.0)
        direction_text = "increase"
    elif direction == "decrease":
        p_val = stats["avg"].get("p_lt_0", 0.0)
        direction_text = "decrease"
    else:  # two-sided
        p_val = stats["avg"].get("prob_of_effect", 0.0)
        direction_text = "effect"

    prose_parts = [
        f"Post-period ({window_str}), the average effect was {fmt_num(avg_mean)} "
        f"({hdi_pct}% HDI [{fmt_num(avg_lower)}, {fmt_num(avg_upper)}]), "
        f"with a posterior probability of an {direction_text} of {fmt_num(p_val, 3)}."
    ]

    # Cumulative effect prose
    if cumulative:
        cum_mean = stats["cum"]["mean"]
        cum_lower = stats["cum"]["hdi_lower"]
        cum_upper = stats["cum"]["hdi_upper"]

        if direction == "increase":
            cum_p_val = stats["cum"].get("p_gt_0", 0.0)
        elif direction == "decrease":
            cum_p_val = stats["cum"].get("p_lt_0", 0.0)
        else:  # two-sided
            cum_p_val = stats["cum"].get("prob_of_effect", 0.0)

        prose_parts.append(
            f"The cumulative effect was {fmt_num(cum_mean)} "
            f"({hdi_pct}% HDI [{fmt_num(cum_lower)}, {fmt_num(cum_upper)}]); "
            f"probability of an {direction_text} {fmt_num(cum_p_val, 3)}."
        )

    # Relative effect prose
    if relative and "relative_mean" in stats["avg"]:
        rel_mean = stats["avg"]["relative_mean"]
        rel_lower = stats["avg"]["relative_hdi_lower"]
        rel_upper = stats["avg"]["relative_hdi_upper"]

        prose_parts.append(
            f"Relative to the counterfactual, this equals {fmt_num(rel_mean)}% on average "
            f"({hdi_pct}% HDI [{fmt_num(rel_lower)}%, {fmt_num(rel_upper)}%])."
        )

    return " ".join(prose_parts)


def _compute_statistics_ols(
    impact,
    counterfactual,
    alpha=0.05,
    cumulative=True,
    relative=True,
):
    """Compute summary statistics for OLS models (time-series experiments).

    Parameters
    ----------
    impact : np.ndarray
        Impact values (y_true - y_pred) as 1D numpy array
    counterfactual : np.ndarray
        Counterfactual predictions as 1D numpy array
    alpha : float
        Significance level
    cumulative : bool
        Whether to compute cumulative statistics
    relative : bool
        Whether to compute relative statistics

    Returns
    -------
    dict
        Dictionary of statistics
    """
    stats = {}

    # Average effect over window
    avg_effect = np.mean(impact)
    n = len(impact)
    # Calculate standard error of mean
    se_avg = np.std(impact, ddof=1) / np.sqrt(n)
    # Degrees of freedom
    df = n - 1
    # t-critical value
    t_critical = t.ppf(1 - alpha / 2, df=df)
    ci_lower = avg_effect - t_critical * se_avg
    ci_upper = avg_effect + t_critical * se_avg
    # Two-sided p-value
    t_stat = avg_effect / se_avg
    p_value = 2 * (1 - t.cdf(abs(t_stat), df=df))

    stats["avg"] = {
        "mean": float(avg_effect),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
    }

    # Cumulative effect
    if cumulative:
        cum_effect = np.sum(impact)
        # Standard error of sum (assuming independence)
        se_cum = np.std(impact, ddof=1) * np.sqrt(n)
        ci_cum_lower = cum_effect - t_critical * se_cum
        ci_cum_upper = cum_effect + t_critical * se_cum
        t_stat_cum = cum_effect / se_cum if se_cum > 0 else 0
        p_value_cum = 2 * (1 - t.cdf(abs(t_stat_cum), df=df))

        stats["cum"] = {
            "mean": float(cum_effect),
            "ci_lower": float(ci_cum_lower),
            "ci_upper": float(ci_cum_upper),
            "p_value": float(p_value_cum),
        }

    # Relative effect
    if relative:
        # Relative effect as percentage change
        relative_effect = (impact / counterfactual) * 100
        rel_mean = np.mean(relative_effect)
        se_rel = np.std(relative_effect, ddof=1) / np.sqrt(n)
        ci_rel_lower = rel_mean - t_critical * se_rel
        ci_rel_upper = rel_mean + t_critical * se_rel

        stats["avg"]["relative_mean"] = float(rel_mean)
        stats["avg"]["relative_ci_lower"] = float(ci_rel_lower)
        stats["avg"]["relative_ci_upper"] = float(ci_rel_upper)

        if cumulative:
            # Cumulative relative effect
            cum_relative = np.sum(relative_effect)
            se_cum_rel = np.std(relative_effect, ddof=1) * np.sqrt(n)
            ci_cum_rel_lower = cum_relative - t_critical * se_cum_rel
            ci_cum_rel_upper = cum_relative + t_critical * se_cum_rel

            stats["cum"]["relative_mean"] = float(cum_relative)
            stats["cum"]["relative_ci_lower"] = float(ci_cum_rel_lower)
            stats["cum"]["relative_ci_upper"] = float(ci_cum_rel_upper)

    return stats


def _compute_statistics_did_ols(
    result,
    alpha=0.05,
):
    """Compute statistics for DiD scalar effect with OLS model.

    Parameters
    ----------
    result
        Experiment result object with OLS model
    alpha : float
        Significance level

    Returns
    -------
    dict
        Dictionary of statistics
    """
    causal_impact = result.causal_impact  # scalar

    # Calculate standard error from model residuals
    # Get fitted values and residuals
    y_pred = result.model.predict(result.X)
    residuals = result.y - y_pred
    mse = np.mean(residuals**2)
    n, p = result.X.shape
    df = n - p

    # Find the interaction term coefficient index
    interaction_term = (
        f"{result.group_variable_name}:{result.post_treatment_variable_name}"
    )
    coeff_idx = None
    for i, label in enumerate(result.labels):
        if interaction_term in label:
            coeff_idx = i
            break

    if coeff_idx is None:
        raise ValueError(f"Could not find interaction term {interaction_term} in model")

    # Calculate standard error for this coefficient
    X = result.X
    try:
        # Try to get X as numpy array
        if hasattr(X, "values"):
            X = X.values
        elif hasattr(X, "data"):
            X = X.data
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(mse * XtX_inv[coeff_idx, coeff_idx])
    except (np.linalg.LinAlgError, AttributeError):
        # Fallback: use simple approximation
        se = np.std(residuals) / np.sqrt(n)

    # t-critical value
    t_critical = t.ppf(1 - alpha / 2, df=df)
    ci_lower = causal_impact - t_critical * se
    ci_upper = causal_impact + t_critical * se
    # Two-sided p-value
    t_stat = causal_impact / se if se > 0 else 0
    p_value = 2 * (1 - t.cdf(abs(t_stat), df=df))

    stats = {
        "mean": float(causal_impact),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
    }
    return stats


def _generate_table_ols(stats, cumulative=True, relative=True):
    """Generate summary table for OLS models."""
    rows = []
    row_names = []

    # Average row
    avg_row = {
        "mean": stats["avg"]["mean"],
        "ci_lower": stats["avg"]["ci_lower"],
        "ci_upper": stats["avg"]["ci_upper"],
        "p_value": stats["avg"]["p_value"],
    }

    # Add relative
    if relative and "relative_mean" in stats["avg"]:
        avg_row["relative_mean"] = stats["avg"]["relative_mean"]
        avg_row["relative_ci_lower"] = stats["avg"]["relative_ci_lower"]
        avg_row["relative_ci_upper"] = stats["avg"]["relative_ci_upper"]

    rows.append(avg_row)
    row_names.append("average")

    # Cumulative row
    if cumulative:
        cum_row = {
            "mean": stats["cum"]["mean"],
            "ci_lower": stats["cum"]["ci_lower"],
            "ci_upper": stats["cum"]["ci_upper"],
            "p_value": stats["cum"]["p_value"],
        }

        # Add relative
        if relative and "relative_mean" in stats["cum"]:
            cum_row["relative_mean"] = stats["cum"]["relative_mean"]
            cum_row["relative_ci_lower"] = stats["cum"]["relative_ci_lower"]
            cum_row["relative_ci_upper"] = stats["cum"]["relative_ci_upper"]

        rows.append(cum_row)
        row_names.append("cumulative")

    df = pd.DataFrame(rows, index=row_names)
    return df


def _generate_prose_ols(
    stats,
    window_coords,
    alpha=0.05,
    cumulative=True,
    relative=True,
):
    """Generate prose summary for OLS models."""
    ci_pct = int((1 - alpha) * 100)

    # Format window string
    if len(window_coords) > 0:
        start_str = str(window_coords[0])
        end_str = str(window_coords[-1])
        window_str = f"{start_str} to {end_str}"
    else:
        window_str = "post-period"

    # Format numbers
    def fmt_num(x, decimals=2):
        return f"{x:.{decimals}f}"

    # Average effect prose
    avg_mean = stats["avg"]["mean"]
    avg_lower = stats["avg"]["ci_lower"]
    avg_upper = stats["avg"]["ci_upper"]
    p_val = stats["avg"]["p_value"]

    prose_parts = [
        f"Post-period ({window_str}), the average effect was {fmt_num(avg_mean)} "
        f"({ci_pct}% CI [{fmt_num(avg_lower)}, {fmt_num(avg_upper)}]), "
        f"with a p-value of {fmt_num(p_val, 3)}."
    ]

    # Cumulative effect prose
    if cumulative:
        cum_mean = stats["cum"]["mean"]
        cum_lower = stats["cum"]["ci_lower"]
        cum_upper = stats["cum"]["ci_upper"]
        cum_p_val = stats["cum"]["p_value"]

        prose_parts.append(
            f"The cumulative effect was {fmt_num(cum_mean)} "
            f"({ci_pct}% CI [{fmt_num(cum_lower)}, {fmt_num(cum_upper)}]); "
            f"p-value {fmt_num(cum_p_val, 3)}."
        )

    # Relative effect prose
    if relative and "relative_mean" in stats["avg"]:
        rel_mean = stats["avg"]["relative_mean"]
        rel_lower = stats["avg"]["relative_ci_lower"]
        rel_upper = stats["avg"]["relative_ci_upper"]

        prose_parts.append(
            f"Relative to the counterfactual, this equals {fmt_num(rel_mean)}% on average "
            f"({ci_pct}% CI [{fmt_num(rel_lower)}%, {fmt_num(rel_upper)}%])."
        )

    return " ".join(prose_parts)


def _generate_table_did_ols(stats):
    """Generate summary table for DiD with OLS model."""
    row = {
        "mean": stats["mean"],
        "ci_lower": stats["ci_lower"],
        "ci_upper": stats["ci_upper"],
        "p_value": stats["p_value"],
    }
    df = pd.DataFrame([row], index=["treatment_effect"])
    return df


def _generate_prose_did_ols(stats, alpha=0.05):
    """Generate prose summary for DiD with OLS model."""
    ci_pct = int((1 - alpha) * 100)

    def fmt_num(x, decimals=2):
        return f"{x:.{decimals}f}"

    mean = stats["mean"]
    lower = stats["ci_lower"]
    upper = stats["ci_upper"]
    p_val = stats["p_value"]

    prose = (
        f"The treatment effect was {fmt_num(mean)} "
        f"({ci_pct}% CI [{fmt_num(lower)}, {fmt_num(upper)}]), "
        f"with a p-value of {fmt_num(p_val, 3)}."
    )

    return prose


def _compute_statistics_rd_ols(result, alpha=0.05):
    """Compute statistics for RD scalar effect with OLS model."""
    discontinuity = result.discontinuity_at_threshold  # scalar

    # Calculate standard error from model
    y_pred = result.model.predict(result.X)
    residuals = result.y - y_pred
    mse = np.mean(residuals**2)
    n, p = result.X.shape
    df = n - p

    # Find the treated coefficient index
    coeff_idx = None
    for i, label in enumerate(result.labels):
        if "treated" in label.lower() and ":" in label:
            coeff_idx = i
            break

    if coeff_idx is None:
        # Fallback: use simple approximation
        se = np.std(residuals) / np.sqrt(n)
    else:
        # Calculate standard error for this coefficient
        X = result.X
        try:
            if hasattr(X, "values"):
                X = X.values
            elif hasattr(X, "data"):
                X = X.data
            XtX_inv = np.linalg.inv(X.T @ X)
            se = np.sqrt(mse * XtX_inv[coeff_idx, coeff_idx])
        except (np.linalg.LinAlgError, AttributeError):
            se = np.std(residuals) / np.sqrt(n)

    # t-critical value
    t_critical = t.ppf(1 - alpha / 2, df=df)
    ci_lower = discontinuity - t_critical * se
    ci_upper = discontinuity + t_critical * se
    # Two-sided p-value
    t_stat = discontinuity / se if se > 0 else 0
    p_value = 2 * (1 - t.cdf(abs(t_stat), df=df))

    stats = {
        "mean": float(discontinuity),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
    }
    return stats


def _generate_table_rd_ols(stats):
    """Generate summary table for RD with OLS model."""
    row = {
        "mean": stats["mean"],
        "ci_lower": stats["ci_lower"],
        "ci_upper": stats["ci_upper"],
        "p_value": stats["p_value"],
    }
    df = pd.DataFrame([row], index=["discontinuity"])
    return df


def _generate_prose_rd_ols(stats, alpha=0.05):
    """Generate prose summary for RD with OLS model."""
    ci_pct = int((1 - alpha) * 100)

    def fmt_num(x, decimals=2):
        return f"{x:.{decimals}f}"

    mean = stats["mean"]
    lower = stats["ci_lower"]
    upper = stats["ci_upper"]
    p_val = stats["p_value"]

    prose = (
        f"The discontinuity at threshold was {fmt_num(mean)} "
        f"({ci_pct}% CI [{fmt_num(lower)}, {fmt_num(upper)}]), "
        f"with a p-value of {fmt_num(p_val, 3)}."
    )

    return prose


# ==============================================================================
# Regression Kink handler functions
# ==============================================================================


def _effect_summary_rkink(
    result,
    direction: Literal["increase", "decrease", "two-sided"] = "increase",
    alpha: float = 0.05,
    min_effect: Optional[float] = None,
):
    """Generate effect summary for Regression Kink experiments."""
    gradient_change = result.gradient_change

    # Check if PyMC (xarray) or OLS (scalar)
    is_pymc = isinstance(gradient_change, xr.DataArray)

    if is_pymc:
        # PyMC model: use unified scalar functions
        hdi_prob = 1 - alpha
        stats = _compute_statistics_scalar(
            gradient_change,
            hdi_prob=hdi_prob,
            direction=direction,
            min_effect=min_effect,
        )
        table = _generate_table_scalar(stats, index_name="gradient_change")
        text = _generate_prose_scalar(
            stats,
            "change in gradient at the kink point",
            alpha=alpha,
            direction=direction,
        )
    else:
        # OLS model: Not currently supported for RegressionKink, but structure is here
        stats = _compute_statistics_rkink_ols(result, alpha=alpha)
        table = _generate_table_rkink_ols(stats)
        text = _generate_prose_rkink_ols(stats, alpha=alpha)

    return EffectSummary(table=table, text=text)


def _compute_statistics_rkink_ols(result, alpha=0.05):
    """Compute statistics for Regression Kink scalar effect with OLS model.

    TODO: Implement OLS support for Regression Kink
    - Extract gradient change coefficient from model
    - Calculate standard error from regression
    - Compute confidence intervals and p-values
    - Follow pattern from _compute_statistics_rd_ols()
    """
    raise NotImplementedError(
        "OLS models are not currently supported for Regression Kink experiments. "
        "Please use a PyMC model for full statistical inference. "
        "If OLS support is needed, see _compute_statistics_rd_ols() for implementation pattern."
    )


def _generate_table_rkink_ols(stats):
    """Generate DataFrame table for Regression Kink with OLS model.

    TODO: This is a placeholder implementation.
    Will be used when _compute_statistics_rkink_ols() is implemented.
    """
    # Placeholder for future OLS support
    data = {
        "metric": ["gradient_change"],
        "mean": [stats["mean"]],
        "CI_lower": [stats["ci_lower"]],
        "CI_upper": [stats["ci_upper"]],
        "p_value": [stats["p_value"]],
    }
    return pd.DataFrame(data)


def _generate_prose_rkink_ols(stats, alpha=0.05):
    """Generate prose summary for Regression Kink with OLS model.

    TODO: This is a placeholder implementation.
    Will be used when _compute_statistics_rkink_ols() is implemented.
    """
    # Placeholder for future OLS support
    ci_pct = int((1 - alpha) * 100)

    def fmt_num(x, decimals=2):
        return f"{x:.{decimals}f}"

    mean = stats["mean"]
    lower = stats["ci_lower"]
    upper = stats["ci_upper"]
    p_val = stats["p_value"]

    prose = (
        f"The change in gradient at the kink point was {fmt_num(mean)} "
        f"({ci_pct}% CI [{fmt_num(lower)}, {fmt_num(upper)}]), "
        f"with a p-value of {fmt_num(p_val, 3)}."
    )

    return prose
