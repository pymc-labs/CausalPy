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
"""

from dataclasses import dataclass

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
    direction="increase",
    alpha=0.05,
    min_effect=None,
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

    # Compute statistics for scalar effect
    hdi_prob = 1 - alpha
    stats = _compute_statistics_did(
        causal_impact,
        hdi_prob=hdi_prob,
        direction=direction,
        min_effect=min_effect,
    )

    # Generate table
    table = _generate_table_did(stats)

    # Generate prose
    text = _generate_prose_did(stats, alpha=alpha, direction=direction)

    return EffectSummary(table=table, text=text)


def _effect_summary_rd(
    result,
    direction="increase",
    alpha=0.05,
    min_effect=None,
):
    """Generate effect summary for Regression Discontinuity experiments."""
    discontinuity = result.discontinuity_at_threshold

    # Check if PyMC (xarray) or OLS (scalar)
    is_pymc = isinstance(discontinuity, xr.DataArray)

    if is_pymc:
        # PyMC model: use posterior draws
        hdi_prob = 1 - alpha
        stats = _compute_statistics_rd(
            discontinuity,
            hdi_prob=hdi_prob,
            direction=direction,
            min_effect=min_effect,
        )
        table = _generate_table_rd(stats)
        text = _generate_prose_rd(stats, alpha=alpha, direction=direction)
    else:
        # OLS model: calculate from model
        stats = _compute_statistics_rd_ols(result, alpha=alpha)
        table = _generate_table_rd_ols(stats)
        text = _generate_prose_rd_ols(stats, alpha=alpha)

    return EffectSummary(table=table, text=text)


def _compute_statistics_did(
    causal_impact,
    hdi_prob=0.95,
    direction="increase",
    min_effect=None,
):
    """Compute statistics for DiD scalar effect."""
    stats = {
        "mean": float(causal_impact.mean(dim=["chain", "draw"]).values),
        "median": float(causal_impact.median(dim=["chain", "draw"]).values),
    }

    # HDI
    hdi_result = az.hdi(causal_impact, hdi_prob=hdi_prob)
    if isinstance(hdi_result, xr.Dataset):
        hdi_data = list(hdi_result.data_vars.values())[0]
        stats["hdi_lower"] = float(hdi_data.sel(hdi="lower").values)
        stats["hdi_upper"] = float(hdi_data.sel(hdi="higher").values)
    else:
        stats["hdi_lower"] = float(hdi_result.sel(hdi="lower").values)
        stats["hdi_upper"] = float(hdi_result.sel(hdi="higher").values)

    # Tail probabilities
    if direction == "increase":
        stats["p_gt_0"] = float((causal_impact > 0).mean().values)
    elif direction == "decrease":
        stats["p_lt_0"] = float((causal_impact < 0).mean().values)
    else:  # two-sided
        p_gt = float((causal_impact > 0).mean().values)
        p_lt = float((causal_impact < 0).mean().values)
        p_two_sided = 2 * min(p_gt, p_lt)
        stats["p_two_sided"] = p_two_sided
        stats["prob_of_effect"] = 1 - p_two_sided

    # ROPE
    if min_effect is not None:
        if direction == "two-sided":
            stats["p_rope"] = float((np.abs(causal_impact) > min_effect).mean().values)
        else:
            stats["p_rope"] = float((causal_impact > min_effect).mean().values)

    return stats


def _generate_table_did(stats):
    """Generate DataFrame table for DiD (single row, no cumulative/relative)."""
    row = {
        "mean": stats["mean"],
        "median": stats["median"],
        "hdi_lower": stats["hdi_lower"],
        "hdi_upper": stats["hdi_upper"],
    }

    # Add tail probabilities
    if "p_gt_0" in stats:
        row["p_gt_0"] = stats["p_gt_0"]
    if "p_lt_0" in stats:
        row["p_lt_0"] = stats["p_lt_0"]
    if "p_two_sided" in stats:
        row["p_two_sided"] = stats["p_two_sided"]
        row["prob_of_effect"] = stats["prob_of_effect"]

    # Add ROPE
    if "p_rope" in stats:
        row["p_rope"] = stats["p_rope"]

    df = pd.DataFrame([row], index=["treatment_effect"])
    return df


def _generate_prose_did(stats, alpha=0.05, direction="increase"):
    """Generate prose summary for DiD scalar effect."""
    hdi_pct = int((1 - alpha) * 100)

    def fmt_num(x, decimals=2):
        return f"{x:.{decimals}f}"

    mean = stats["mean"]
    lower = stats["hdi_lower"]
    upper = stats["hdi_upper"]

    # Tail probability text
    if direction == "increase":
        p_val = stats.get("p_gt_0", 0.0)
        direction_text = "increase"
    elif direction == "decrease":
        p_val = stats.get("p_lt_0", 0.0)
        direction_text = "decrease"
    else:  # two-sided
        p_val = stats.get("prob_of_effect", 0.0)
        direction_text = "effect"

    prose = (
        f"The average treatment effect was {fmt_num(mean)} "
        f"({hdi_pct}% HDI [{fmt_num(lower)}, {fmt_num(upper)}]), "
        f"with a posterior probability of an {direction_text} of {fmt_num(p_val, 3)}."
    )

    return prose


def _extract_window(result, window, treated_unit=None):
    """Extract windowed impact data based on window specification."""
    post_impact = result.post_impact

    # Check if PyMC (xarray with chain/draw dims) or OLS (numpy array or xarray without chains)
    # For OLS Synthetic Control, post_impact can be xarray but without chain/draw dimensions
    is_pymc = isinstance(post_impact, xr.DataArray) and (
        "chain" in post_impact.dims or "draw" in post_impact.dims
    )

    if is_pymc:
        # Handle treated_unit selection for multi-unit experiments
        if "treated_units" in post_impact.dims:
            if treated_unit is not None:
                post_impact = post_impact.sel(treated_units=treated_unit)
            else:
                # Use first unit if multiple exist
                post_impact = post_impact.isel(treated_units=0)
    else:
        # OLS model - handle treated_unit if multi-dimensional
        # For OLS, post_impact might be xarray (from SyntheticControl) or numpy array
        if isinstance(post_impact, xr.DataArray):
            # OLS with xarray (e.g., SyntheticControl)
            if "treated_units" in post_impact.dims:
                # Check for shape mismatch between data and coordinates
                # This can happen when post_impact is calculated incorrectly in SyntheticControl
                treated_units_coord_len = len(post_impact.coords["treated_units"])
                treated_units_dim_size = post_impact.sizes["treated_units"]

                if treated_units_coord_len != treated_units_dim_size:
                    # Shape mismatch - take only the slice that matches the coordinates
                    # This typically means we need the first `treated_units_coord_len` elements
                    post_impact = post_impact.isel(
                        treated_units=slice(0, treated_units_coord_len)
                    )

                # Now select the specific treated_unit
                if treated_unit is not None:
                    post_impact = post_impact.sel(treated_units=treated_unit)
                else:
                    post_impact = post_impact.isel(treated_units=0)
            # Convert to numpy for consistent handling - do this BEFORE window selection
            # Squeeze to remove any single-element dimensions
            post_impact = np.squeeze(post_impact.values)
        elif hasattr(post_impact, "ndim") and post_impact.ndim > 1:
            # OLS with numpy array (multi-dimensional)
            if treated_unit is not None and hasattr(result, "treated_units"):
                unit_idx = result.treated_units.index(treated_unit)
                post_impact = post_impact[:, unit_idx]
            else:
                post_impact = post_impact[:, 0]
        # Ensure post_impact is a numpy array at this point
        if not isinstance(post_impact, np.ndarray):
            post_impact = np.asarray(post_impact)

    # Determine window coordinates
    if window == "post":
        # Use all post-treatment time points
        window_coords = result.datapost.index
        if is_pymc:
            windowed_impact = post_impact
        else:
            # OLS: post_impact is now a numpy array
            windowed_impact = post_impact
    elif isinstance(window, tuple) and len(window) == 2:
        start, end = window
        # Handle datetime vs integer indices
        if isinstance(result.datapost.index, pd.DatetimeIndex):
            if not isinstance(start, pd.Timestamp):
                start = pd.Timestamp(start)
            if not isinstance(end, pd.Timestamp):
                end = pd.Timestamp(end)
            window_coords = result.datapost.index[
                (result.datapost.index >= start) & (result.datapost.index <= end)
            ]
            if is_pymc:
                windowed_impact = post_impact.sel(obs_ind=window_coords)
            else:
                # OLS: convert window_coords to indices
                # post_impact is now a numpy array
                indices = [
                    result.datapost.index.get_loc(coord) for coord in window_coords
                ]
                windowed_impact = post_impact[indices]
        else:
            # Integer index
            # Ensure start and end are comparable with the index
            # Convert to native Python int to avoid type issues
            start_val = int(start)
            end_val = int(end)
            # Use result.datapost.index for filtering, then match with post_impact coordinates
            mask = (result.datapost.index >= start_val) & (
                result.datapost.index <= end_val
            )
            window_coords = result.datapost.index[mask]
            if is_pymc:
                windowed_impact = post_impact.sel(obs_ind=window_coords)
            else:
                # OLS: convert window_coords to indices
                # post_impact is now a numpy array
                indices = [
                    result.datapost.index.get_loc(coord) for coord in window_coords
                ]
                windowed_impact = post_impact[indices]
    elif isinstance(window, slice):
        # Slice window - handle differently for datetime vs integer indices
        if isinstance(result.datapost.index, pd.DatetimeIndex):
            # For datetime, slice works directly
            window_coords = result.datapost.index[window]
        else:
            # For integer indices, convert slice to value-based filtering
            # slice(start, stop, step) -> get all values in [start, stop)
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
            if step == 1:
                mask = (result.datapost.index >= start_val) & (
                    result.datapost.index < stop_val
                )
                window_coords = result.datapost.index[mask]
            else:
                # For non-unit step, filter then apply step
                mask = (result.datapost.index >= start_val) & (
                    result.datapost.index < stop_val
                )
                filtered = result.datapost.index[mask]
                window_coords = filtered[::step]
        if is_pymc:
            windowed_impact = post_impact.sel(obs_ind=window_coords)
        else:
            # OLS: convert window_coords to indices
            indices = [result.datapost.index.get_loc(coord) for coord in window_coords]
            windowed_impact = post_impact[indices]
    else:
        raise ValueError(
            f"window must be 'post', a tuple (start, end), or a slice. Got {type(window)}"
        )

    # Validate window is not empty
    if len(window_coords) == 0:
        raise ValueError("Window contains no time points")

    return windowed_impact, window_coords


def _extract_counterfactual(result, window_coords, treated_unit=None):
    """Extract counterfactual predictions for the window."""
    post_pred = result.post_pred

    # Check if PyMC (InferenceData) or OLS (numpy array or xarray)
    # PyMC models return InferenceData which has posterior_predictive attribute
    if hasattr(post_pred, "posterior_predictive"):
        # PyMC model - InferenceData object
        # Extract mu (posterior expectation)
        counterfactual = post_pred.posterior_predictive["mu"]
    elif isinstance(post_pred, dict) and "posterior_predictive" in post_pred:
        # PyMC model - dict format (fallback)
        counterfactual = post_pred["posterior_predictive"]["mu"]
    else:
        # OLS model - post_pred is numpy array or xarray
        if isinstance(post_pred, xr.DataArray):
            # OLS with xarray (e.g., SyntheticControl)
            # First select the treated_unit if multi-dimensional
            if "treated_units" in post_pred.dims:
                # Check for shape mismatch between data and coordinates
                treated_units_coord_len = len(post_pred.coords["treated_units"])
                treated_units_dim_size = post_pred.sizes["treated_units"]

                if treated_units_coord_len != treated_units_dim_size:
                    # Shape mismatch - take only the slice that matches the coordinates
                    post_pred = post_pred.isel(
                        treated_units=slice(0, treated_units_coord_len)
                    )

                # Now select the specific treated_unit
                if treated_unit is not None:
                    post_pred = post_pred.sel(treated_units=treated_unit)
                else:
                    post_pred = post_pred.isel(treated_units=0)

            # Then select the window using integer indices
            if isinstance(window_coords, pd.Index):
                # Find indices in datapost.index that match window_coords
                indices = [
                    result.datapost.index.get_loc(coord) for coord in window_coords
                ]
                # Use isel for integer-based indexing on xarray
                counterfactual = post_pred.isel(obs_ind=indices)
            else:
                # If window_coords is already indices
                counterfactual = post_pred.isel(obs_ind=window_coords)

            # Convert to numpy and squeeze to remove single-element dims
            counterfactual = np.squeeze(counterfactual.values)
        else:
            # OLS with numpy array
            # Convert window_coords to indices
            if isinstance(window_coords, pd.Index):
                indices = [
                    result.datapost.index.get_loc(coord) for coord in window_coords
                ]
                counterfactual = post_pred[indices]
            else:
                counterfactual = post_pred[window_coords]

            # Handle treated_unit for multi-unit numpy arrays
            if hasattr(counterfactual, "ndim") and counterfactual.ndim > 1:
                if treated_unit is not None and hasattr(result, "treated_units"):
                    unit_idx = result.treated_units.index(treated_unit)
                    counterfactual = counterfactual[:, unit_idx]
                else:
                    counterfactual = counterfactual[:, 0]

        return counterfactual

    # Handle treated_unit selection (PyMC only)
    if "treated_units" in counterfactual.dims:
        if treated_unit is not None:
            counterfactual = counterfactual.sel(treated_units=treated_unit)
        else:
            counterfactual = counterfactual.isel(treated_units=0)

    # Select window (PyMC only)
    counterfactual = counterfactual.sel(obs_ind=window_coords)

    return counterfactual


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


def _compute_statistics_rd(
    discontinuity,
    hdi_prob=0.95,
    direction="increase",
    min_effect=None,
):
    """Compute statistics for RD scalar effect (PyMC)."""
    stats = {
        "mean": float(discontinuity.mean(dim=["chain", "draw"]).values),
        "median": float(discontinuity.median(dim=["chain", "draw"]).values),
    }

    # HDI
    hdi_result = az.hdi(discontinuity, hdi_prob=hdi_prob)
    if isinstance(hdi_result, xr.Dataset):
        hdi_data = list(hdi_result.data_vars.values())[0]
        stats["hdi_lower"] = float(hdi_data.sel(hdi="lower").values)
        stats["hdi_upper"] = float(hdi_data.sel(hdi="higher").values)
    else:
        stats["hdi_lower"] = float(hdi_result.sel(hdi="lower").values)
        stats["hdi_upper"] = float(hdi_result.sel(hdi="higher").values)

    # Tail probabilities
    if direction == "increase":
        stats["p_gt_0"] = float((discontinuity > 0).mean().values)
    elif direction == "decrease":
        stats["p_lt_0"] = float((discontinuity < 0).mean().values)
    else:  # two-sided
        p_gt = float((discontinuity > 0).mean().values)
        p_lt = float((discontinuity < 0).mean().values)
        p_two_sided = 2 * min(p_gt, p_lt)
        stats["p_two_sided"] = p_two_sided
        stats["prob_of_effect"] = 1 - p_two_sided

    # ROPE
    if min_effect is not None:
        if direction == "two-sided":
            stats["p_rope"] = float((np.abs(discontinuity) > min_effect).mean().values)
        else:
            stats["p_rope"] = float((discontinuity > min_effect).mean().values)

    return stats


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


def _generate_table_rd(stats):
    """Generate summary table for RD (PyMC)."""
    row = {
        "mean": stats["mean"],
        "median": stats["median"],
        "hdi_lower": stats["hdi_lower"],
        "hdi_upper": stats["hdi_upper"],
    }

    # Add tail probabilities
    if "p_gt_0" in stats:
        row["p_gt_0"] = stats["p_gt_0"]
    if "p_lt_0" in stats:
        row["p_lt_0"] = stats["p_lt_0"]
    if "p_two_sided" in stats:
        row["p_two_sided"] = stats["p_two_sided"]
        row["prob_of_effect"] = stats["prob_of_effect"]
    if "p_rope" in stats:
        row["p_rope"] = stats["p_rope"]

    df = pd.DataFrame([row], index=["discontinuity"])
    return df


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


def _generate_prose_rd(stats, alpha=0.05, direction="increase"):
    """Generate prose summary for RD (PyMC)."""
    hdi_pct = int((1 - alpha) * 100)

    def fmt_num(x, decimals=2):
        return f"{x:.{decimals}f}"

    mean = stats["mean"]
    lower = stats["hdi_lower"]
    upper = stats["hdi_upper"]

    # Tail probability text
    if direction == "increase":
        p_val = stats.get("p_gt_0", 0.0)
        direction_text = "increase"
    elif direction == "decrease":
        p_val = stats.get("p_lt_0", 0.0)
        direction_text = "decrease"
    else:  # two-sided
        p_val = stats.get("prob_of_effect", 0.0)
        direction_text = "effect"

    prose = (
        f"The discontinuity at threshold was {fmt_num(mean)} "
        f"({hdi_pct}% HDI [{fmt_num(lower)}, {fmt_num(upper)}]), "
        f"with a posterior probability of an {direction_text} of {fmt_num(p_val, 3)}."
    )

    return prose


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
    direction="increase",
    alpha=0.05,
    min_effect=None,
):
    """Generate effect summary for Regression Kink experiments."""
    gradient_change = result.gradient_change

    # Check if PyMC (xarray) or OLS (scalar)
    is_pymc = isinstance(gradient_change, xr.DataArray)

    if is_pymc:
        # PyMC model: use posterior draws
        hdi_prob = 1 - alpha
        stats = _compute_statistics_rkink(
            gradient_change,
            hdi_prob=hdi_prob,
            direction=direction,
            min_effect=min_effect,
        )
        table = _generate_table_rkink(stats)
        text = _generate_prose_rkink(stats, alpha=alpha, direction=direction)
    else:
        # OLS model: Not currently supported for RegressionKink, but structure is here
        stats = _compute_statistics_rkink_ols(result, alpha=alpha)
        table = _generate_table_rkink_ols(stats)
        text = _generate_prose_rkink_ols(stats, alpha=alpha)

    return EffectSummary(table=table, text=text)


def _compute_statistics_rkink(
    gradient_change,
    hdi_prob=0.95,
    direction="increase",
    min_effect=None,
):
    """Compute statistics for Regression Kink scalar effect (PyMC)."""
    stats = {
        "mean": float(gradient_change.mean(dim=["chain", "draw"]).values),
        "median": float(gradient_change.median(dim=["chain", "draw"]).values),
    }

    # HDI
    hdi_result = az.hdi(gradient_change, hdi_prob=hdi_prob)
    if isinstance(hdi_result, xr.Dataset):
        hdi_data = list(hdi_result.data_vars.values())[0]
        stats["hdi_lower"] = float(hdi_data.sel(hdi="lower").values)
        stats["hdi_upper"] = float(hdi_data.sel(hdi="higher").values)
    else:
        stats["hdi_lower"] = float(hdi_result.sel(hdi="lower").values)
        stats["hdi_upper"] = float(hdi_result.sel(hdi="higher").values)

    # Tail probabilities
    if direction == "increase":
        stats["p_gt_0"] = float((gradient_change > 0).mean().values)
    elif direction == "decrease":
        stats["p_lt_0"] = float((gradient_change < 0).mean().values)
    else:  # two-sided
        p_gt = float((gradient_change > 0).mean().values)
        p_lt = float((gradient_change < 0).mean().values)
        p_two_sided = 2 * min(p_gt, p_lt)
        stats["p_two_sided"] = p_two_sided
        stats["prob_of_effect"] = 1 - p_two_sided

    # ROPE
    if min_effect is not None:
        if direction == "two-sided":
            stats["p_rope"] = float(
                (np.abs(gradient_change) > min_effect).mean().values
            )
        else:
            stats["p_rope"] = float((gradient_change > min_effect).mean().values)

    return stats


def _compute_statistics_rkink_ols(result, alpha=0.05):
    """Compute statistics for Regression Kink scalar effect with OLS model."""
    # Note: RegressionKink currently only supports PyMC models
    # This is a placeholder for future OLS support
    raise NotImplementedError(
        "OLS models are not currently supported for Regression Kink experiments. "
        "Please use a PyMC model."
    )


def _generate_table_rkink(stats):
    """Generate DataFrame table for Regression Kink (PyMC)."""
    data = {
        "metric": ["gradient_change"],
        "mean": [stats["mean"]],
        "median": [stats["median"]],
        "HDI_lower": [stats["hdi_lower"]],
        "HDI_upper": [stats["hdi_upper"]],
    }

    # Add direction-specific columns
    if "p_gt_0" in stats:
        data["P(effect>0)"] = [stats["p_gt_0"]]
    elif "p_lt_0" in stats:
        data["P(effect<0)"] = [stats["p_lt_0"]]
    elif "p_two_sided" in stats:
        data["P(two-sided)"] = [stats["p_two_sided"]]
        data["P(effect)"] = [stats["prob_of_effect"]]

    # Add ROPE if present
    if "p_rope" in stats:
        data["P(|effect|>min_effect)"] = [stats["p_rope"]]

    return pd.DataFrame(data)


def _generate_table_rkink_ols(stats):
    """Generate DataFrame table for Regression Kink with OLS model."""
    # Placeholder for future OLS support
    data = {
        "metric": ["gradient_change"],
        "mean": [stats["mean"]],
        "CI_lower": [stats["ci_lower"]],
        "CI_upper": [stats["ci_upper"]],
        "p_value": [stats["p_value"]],
    }
    return pd.DataFrame(data)


def _generate_prose_rkink(stats, alpha=0.05, direction="increase"):
    """Generate prose summary for Regression Kink (PyMC)."""
    hdi_pct = int((1 - alpha) * 100)

    def fmt_num(x, decimals=2):
        return f"{x:.{decimals}f}"

    mean = stats["mean"]
    median = stats["median"]
    lower = stats["hdi_lower"]
    upper = stats["hdi_upper"]

    prose_parts = [
        f"The change in gradient at the kink point had a mean of {fmt_num(mean)} "
        f"(median: {fmt_num(median)}, {hdi_pct}% HDI [{fmt_num(lower)}, {fmt_num(upper)}])."
    ]

    # Add tail probability info
    if direction == "increase":
        prob = stats["p_gt_0"]
        prose_parts.append(
            f" There is a {fmt_num(prob * 100, 1)}% posterior probability "
            f"that the gradient change is positive."
        )
    elif direction == "decrease":
        prob = stats["p_lt_0"]
        prose_parts.append(
            f" There is a {fmt_num(prob * 100, 1)}% posterior probability "
            f"that the gradient change is negative."
        )
    else:  # two-sided
        prob = stats["prob_of_effect"]
        prose_parts.append(
            f" There is a {fmt_num(prob * 100, 1)}% posterior probability "
            f"of a non-zero gradient change (two-sided test)."
        )

    # Add ROPE info
    if "p_rope" in stats:
        p_rope = stats["p_rope"]
        prose_parts.append(
            f" The probability that the absolute gradient change exceeds "
            f"the practical significance threshold is {fmt_num(p_rope * 100, 1)}%."
        )

    return "".join(prose_parts)


def _generate_prose_rkink_ols(stats, alpha=0.05):
    """Generate prose summary for Regression Kink with OLS model."""
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
