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
    if hasattr(result, "causal_impact") and not hasattr(result, "post_impact"):
        return "did"
    elif hasattr(result, "post_impact"):
        return "its_or_sc"  # ITS or Synthetic Control
    else:
        raise ValueError(
            "Unknown experiment type. Result must have either 'causal_impact' "
            "(DiD) or 'post_impact' (ITS/Synthetic Control) attribute."
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

    # Handle treated_unit selection for multi-unit experiments
    if "treated_units" in post_impact.dims:
        if treated_unit is not None:
            post_impact = post_impact.sel(treated_units=treated_unit)
        else:
            # Use first unit if multiple exist
            post_impact = post_impact.isel(treated_units=0)

    # Determine window coordinates
    if window == "post":
        # Use all post-treatment time points
        window_coords = result.datapost.index
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
            windowed_impact = post_impact.sel(obs_ind=window_coords)
        else:
            # Integer index
            window_coords = result.datapost.index[
                (result.datapost.index >= start) & (result.datapost.index <= end)
            ]
            windowed_impact = post_impact.sel(obs_ind=window_coords)
    elif isinstance(window, slice):
        # Integer slice
        window_coords = result.datapost.index[window]
        windowed_impact = post_impact.sel(obs_ind=window_coords)
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

    # Extract mu (posterior expectation)
    counterfactual = post_pred["posterior_predictive"]["mu"]

    # Handle treated_unit selection
    if "treated_units" in counterfactual.dims:
        if treated_unit is not None:
            counterfactual = counterfactual.sel(treated_units=treated_unit)
        else:
            counterfactual = counterfactual.isel(treated_units=0)

    # Select window
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
