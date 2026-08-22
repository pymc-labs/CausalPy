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
Reporting utilities for causal inference experiments.

This module provides statistical summaries and prose reports for causal effects.
The reporting functions automatically compute appropriate statistics based on the
model type (Bayesian/PyMC or Frequentist/OLS).

For detailed explanations of the reported statistics (HDI, ROPE, p-values, etc.)
and their interpretation, see the documentation:
https://causalpy.readthedocs.io/en/latest/knowledgebase/reporting_statistics.html
"""

from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import t

from causalpy._arviz_compat import hdi_bounds
from causalpy.constants import HDI_PROB
from causalpy.experiments._results import (
    CausalResult,
    CoefficientResult,
    DiscontinuityResult,
    KinkResult,
    StaggeredDifferenceInDifferencesResult,
)
from causalpy.utils import _as_scalar, has_posterior_draws


@dataclass
class EffectSummary:
    """Container for effect summary statistics and prose report.

    Attributes
    ----------
    table : pd.DataFrame
        DataFrame containing summary statistics (mean, median, HDI, tail probabilities)
    text : str
        Detailed multi-paragraph narrative report with
        observed vs counterfactual breakdown, statistical credibility assessment,
        and assumptions/guidance.
    """

    table: pd.DataFrame
    text: str


@dataclass(frozen=True)
class _BayesianDecision:
    """Internal Bayesian HDI/ROPE decision consumed by reporting prose."""

    conclusion: Literal[
        "practically_significant",
        "practically_equivalent_to_zero",
        "inconclusive",
        "descriptive",
    ]
    framework: Literal["hdi_rope", "descriptive"]
    interval: tuple[float, float]
    rope: tuple[float, float] | None
    tail_label: Literal["increase", "decrease", "two-sided"]
    tail_probability: float
    posterior_mass_below_rope: float | None
    posterior_mass_inside_rope: float | None
    posterior_mass_above_rope: float | None


class _ScalarBayesianStats(TypedDict, total=False):
    """Numerical scalar summary fields plus the prose decision."""

    mean: float
    median: float
    hdi_lower: float
    hdi_upper: float
    p_gt_0: float
    p_lt_0: float
    p_two_sided: float
    prob_of_effect: float
    p_rope: float
    decision: _BayesianDecision


__all__ = ["EffectSummary"]


# ==============================================================================
# Helper functions for common operations
# ==============================================================================


def _posterior_probability(indicator: xr.DataArray, effect: xr.DataArray) -> float:
    """Return an indicator's posterior mean while excluding non-finite draws."""
    return _as_scalar(indicator.where(np.isfinite(effect)).mean(skipna=True))


def _finite_posterior_draws(effect: xr.DataArray) -> xr.DataArray:
    """Mask non-finite posterior draws and reject an empty finite posterior."""
    finite_effect = effect.where(np.isfinite(effect))
    if not bool(np.isfinite(effect).any()):
        raise ValueError("Effect posterior contains no finite draws.")
    return finite_effect


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
        return {"p_gt_0": _posterior_probability(effect > 0, effect)}
    elif direction == "decrease":
        return {"p_lt_0": _posterior_probability(effect < 0, effect)}
    else:  # two-sided
        p_gt = _posterior_probability(effect > 0, effect)
        p_lt = _posterior_probability(effect < 0, effect)
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
        return _posterior_probability(np.abs(effect) > min_effect, effect)
    elif direction == "increase":
        return _posterior_probability(effect > min_effect, effect)
    elif direction == "decrease":
        return _posterior_probability(effect < -min_effect, effect)


def _validate_min_effect(min_effect: float | None) -> float | None:
    """Validate and normalize a supplied ROPE threshold."""
    if min_effect is None:
        return None
    if not np.isfinite(min_effect) or min_effect < 0:
        raise ValueError("min_effect must be finite and non-negative.")
    return float(min_effect)


def _make_bayesian_decision(
    effect: xr.DataArray,
    *,
    hdi_lower: float,
    hdi_upper: float,
    tail_probabilities: dict[str, float],
    direction: Literal["increase", "decrease", "two-sided"],
    min_effect: float | None,
) -> _BayesianDecision:
    """Construct the immutable decision used by Bayesian prose renderers."""
    tail_key = {
        "increase": "p_gt_0",
        "decrease": "p_lt_0",
        "two-sided": "p_two_sided",
    }[direction]
    interval = (hdi_lower, hdi_upper)
    tail_probability = tail_probabilities[tail_key]
    min_effect = _validate_min_effect(min_effect)

    if min_effect is None:
        return _BayesianDecision(
            conclusion="descriptive",
            framework="descriptive",
            interval=interval,
            rope=None,
            tail_label=direction,
            tail_probability=tail_probability,
            posterior_mass_below_rope=None,
            posterior_mass_inside_rope=None,
            posterior_mass_above_rope=None,
        )

    rope = (-min_effect, min_effect)
    posterior_mass_below_rope = _posterior_probability(effect < rope[0], effect)
    posterior_mass_inside_rope = _posterior_probability(
        (effect >= rope[0]) & (effect <= rope[1]), effect
    )
    posterior_mass_above_rope = _posterior_probability(effect > rope[1], effect)

    conclusion: Literal[
        "practically_significant",
        "practically_equivalent_to_zero",
        "inconclusive",
    ]
    if hdi_upper < rope[0] or hdi_lower > rope[1]:
        conclusion = "practically_significant"
    elif rope[0] <= hdi_lower and hdi_upper <= rope[1]:
        conclusion = "practically_equivalent_to_zero"
    else:
        conclusion = "inconclusive"

    return _BayesianDecision(
        conclusion=conclusion,
        framework="hdi_rope",
        interval=interval,
        rope=rope,
        tail_label=direction,
        tail_probability=tail_probability,
        posterior_mass_below_rope=posterior_mass_below_rope,
        posterior_mass_inside_rope=posterior_mass_inside_rope,
        posterior_mass_above_rope=posterior_mass_above_rope,
    )


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


def _format_probability_as_percent(probability: float) -> str:
    """Format a probability as a percentage without truncating its precision."""
    return f"{np.format_float_positional(probability * 100, unique=True, trim='-')}%"


def _format_rope_bound(value: float) -> str:
    """Format a ROPE bound using the shortest round-trip-safe representation."""
    if value == 0:
        return "0"
    return repr(float(value)).removesuffix(".0")


def _render_bayesian_decision(decision: _BayesianDecision, coverage: str) -> str:
    """Render the decision-owned Bayesian tail and optional ROPE interpretation."""
    if decision.tail_label == "increase":
        parts = [
            "The posterior probability of an increase is "
            f"{_format_number(decision.tail_probability, 3)}."
        ]
    elif decision.tail_label == "decrease":
        parts = [
            "The posterior probability of a decrease is "
            f"{_format_number(decision.tail_probability, 3)}."
        ]
    else:
        parts = [
            "The two-sided tail probability is "
            f"{_format_number(decision.tail_probability, 3)}."
        ]

    if decision.framework == "descriptive":
        return " ".join(parts)

    rope = decision.rope
    if rope is None:
        raise ValueError("An HDI/ROPE decision requires ROPE bounds.")
    rope_lower, rope_upper = map(_format_rope_bound, rope)

    if decision.conclusion == "practically_significant":
        parts.append(
            f"Using the closed ROPE [{rope_lower}, {rope_upper}], the {coverage} HDI "
            "is entirely outside the ROPE; the effect is practically significant."
        )
    elif decision.conclusion == "practically_equivalent_to_zero":
        parts.append(
            f"Using the closed ROPE [{rope_lower}, {rope_upper}], the {coverage} HDI "
            "is entirely inside the ROPE; the effect is practically equivalent to zero."
        )
    else:
        parts.append(
            f"Using the closed ROPE [{rope_lower}, {rope_upper}], the {coverage} HDI "
            "overlaps the ROPE; the result is inconclusive."
        )

    below = decision.posterior_mass_below_rope
    inside = decision.posterior_mass_inside_rope
    above = decision.posterior_mass_above_rope
    if below is None or inside is None or above is None:
        raise ValueError("An HDI/ROPE decision requires posterior ROPE masses.")
    parts.append(
        f"Posterior mass is {_format_number(below, 3)} below, "
        f"{_format_number(inside, 3)} inside, and {_format_number(above, 3)} "
        "above the ROPE."
    )
    return " ".join(parts)


# ==============================================================================
# Unified scalar effect statistics (DiD, RD, RKink)
# ==============================================================================


def _compute_statistics_scalar(
    effect: xr.DataArray,
    hdi_prob: float = 0.95,
    direction: Literal["increase", "decrease", "two-sided"] = "increase",
    min_effect: float | None = None,
) -> _ScalarBayesianStats:
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
        Finite, non-negative ROPE half-width. The generated decision uses the
        closed interval ``[-min_effect, min_effect]`` when supplied.

    Returns
    -------
    _ScalarBayesianStats
        Numerical summary fields plus the internal decision used for prose.
    """
    min_effect = _validate_min_effect(min_effect)
    effect = _finite_posterior_draws(effect)

    stats: _ScalarBayesianStats = {
        "mean": _as_scalar(effect.mean(dim=["chain", "draw"])),
        "median": _as_scalar(effect.median(dim=["chain", "draw"])),
    }

    stats["hdi_lower"], stats["hdi_upper"] = hdi_bounds(effect, prob=hdi_prob)
    tail_probabilities = _compute_tail_probabilities(effect, direction)
    if direction == "increase":
        stats["p_gt_0"] = tail_probabilities["p_gt_0"]
    elif direction == "decrease":
        stats["p_lt_0"] = tail_probabilities["p_lt_0"]
    else:
        stats["p_two_sided"] = tail_probabilities["p_two_sided"]
        stats["prob_of_effect"] = tail_probabilities["prob_of_effect"]

    if min_effect is not None:
        stats["p_rope"] = _compute_rope_probability(effect, min_effect, direction)

    stats["decision"] = _make_bayesian_decision(
        effect,
        hdi_lower=stats["hdi_lower"],
        hdi_upper=stats["hdi_upper"],
        tail_probabilities=tail_probabilities,
        direction=direction,
        min_effect=min_effect,
    )
    return stats


def _generate_table_scalar(
    stats: _ScalarBayesianStats, index_name: str = "effect"
) -> pd.DataFrame:
    """Generate summary table for scalar effects (DiD, RD, RKink)."""
    row = {
        "mean": stats["mean"],
        "median": stats["median"],
        "hdi_lower": stats["hdi_lower"],
        "hdi_upper": stats["hdi_upper"],
    }

    if "p_gt_0" in stats:
        row["p_gt_0"] = stats["p_gt_0"]
    if "p_lt_0" in stats:
        row["p_lt_0"] = stats["p_lt_0"]
    if "p_two_sided" in stats:
        row["p_two_sided"] = stats["p_two_sided"]
    if "prob_of_effect" in stats:
        row["prob_of_effect"] = stats["prob_of_effect"]
    if "p_rope" in stats:
        row["p_rope"] = stats["p_rope"]

    return pd.DataFrame([row], index=[index_name])


def _generate_prose_scalar(
    stats: _ScalarBayesianStats,
    effect_name: str,
    alpha: float = 0.05,
    direction: Literal["increase", "decrease", "two-sided"] = "increase",
) -> str:
    """Generate prose summary for scalar effects."""
    hdi_coverage = _format_probability_as_percent(1 - alpha)
    decision = stats["decision"]
    mean = stats["mean"]
    lower, upper = decision.interval

    return (
        f"The {effect_name} was {_format_number(mean)} "
        f"({hdi_coverage} HDI [{_format_number(lower)}, {_format_number(upper)}]). "
        f"{_render_bayesian_decision(decision, hdi_coverage)}"
    )


def _detect_experiment_type(result):
    """Detect experiment type from its result bundle."""
    bundle = result.result
    if isinstance(bundle, DiscontinuityResult):
        return "rd"  # Regression Discontinuity
    elif isinstance(bundle, KinkResult):
        return "rkink"  # Regression Kink
    elif isinstance(bundle, StaggeredDifferenceInDifferencesResult):
        return "staggered_did"  # Staggered Difference-in-Differences
    elif isinstance(bundle, CoefficientResult):
        return "did"  # Difference-in-Differences or ANCOVA/PrePostNEGD
    elif isinstance(bundle, CausalResult):
        return "its_or_sc"  # ITS or Synthetic Control
    else:
        raise ValueError(
            "Unknown experiment type. Result must have 'discontinuity_at_threshold' (RD), "
            "'gradient_change' (Regression Kink), 'att_event_time_' (Staggered DiD), "
            "'causal_impact' (DiD/ANCOVA), or 'post_impact' (ITS/Synthetic Control) attribute."
        )


def _apply_prior_grouping(text: str, group: str) -> str:
    """Frame prose as a prior plausibility statement for prior-group bundles."""
    if group == "prior":
        return f"Prior predictive check (not a causal estimate): {text}"
    return text


def _effect_summary_did(
    bundle,
    *,
    direction: Literal["increase", "decrease", "two-sided"] = "increase",
    alpha: float = 0.05,
    min_effect: float | None = None,
    group: Literal["prior", "posterior"] = "posterior",
):
    """Generate effect summary for Difference-in-Differences experiments."""
    causal_impact = bundle.causal_impact

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
    text = _apply_prior_grouping(
        _generate_prose_scalar(
            stats, "average treatment effect", alpha=alpha, direction=direction
        ),
        group,
    )

    return EffectSummary(table=table, text=text)


def _effect_summary_staggered_did(
    experiment,
    bundle=None,
    *,
    group: Literal["prior", "posterior"] = "posterior",
    direction: Literal["increase", "decrease", "two-sided"] = "increase",
    alpha: float = 0.06,
    min_effect: float | None = None,
):
    """Generate effect summary for Staggered Difference-in-Differences experiments.

    Reports event-time ATT estimates with pre-treatment placebo check and
    post-treatment dynamic effects.

    Parameters
    ----------
    experiment
        StaggeredDifferenceInDifferences experiment (supplies the
        deterministic cohort list)
    bundle
        The resolved group result bundle providing the event-time ATT table;
        defaults to ``experiment.result`` when omitted
    direction : {"increase", "decrease", "two-sided"}
        Direction for interpretation
    alpha : float, default=0.06
        Probability mass outside the credible interval. The HDI probability
        is computed as (1 - alpha). Default 0.06 gives 94% HDI, matching
        ArviZ's default. Only used as fallback if the result bundle doesn't
        store the HDI probability used during interval computation.
    min_effect : float, optional
        Not used for staggered DiD, kept for API consistency

    Returns
    -------
    EffectSummary
        Summary with table of event-time ATTs and prose interpretation
    """
    result_bundle = bundle if bundle is not None else experiment.result
    att_et = result_bundle.att_event_time.copy()

    # Separate pre-treatment (placebo) and post-treatment effects
    pre_treatment = att_et[att_et["event_time"] < 0]
    post_treatment = att_et[att_et["event_time"] >= 0]
    if "identified" in post_treatment.columns:
        post_treatment = post_treatment[post_treatment["identified"]]
    if "identified" in pre_treatment.columns:
        pre_treatment = pre_treatment[pre_treatment["identified"]]

    # Build summary table with all event-time effects
    table = att_et.copy()

    # Generate prose summary
    prose_parts = []

    # Overall ATT (average across all post-treatment periods)
    if len(post_treatment) > 0:
        avg_post_att = post_treatment["att"].mean()
        if "att_lower" in post_treatment.columns:
            # Bayesian model - use stored hdi_prob from experiment
            avg_lower = post_treatment["att_lower"].mean()
            avg_upper = post_treatment["att_upper"].mean()
            # Use the HDI probability that was actually used to compute the intervals
            hdi_prob = getattr(result_bundle, "hdi_prob", 1 - alpha)
            hdi_pct = int(hdi_prob * 100)
            prose_parts.append(
                f"Staggered DiD analysis: The average post-treatment effect "
                f"across event-times was {avg_post_att:.2f} "
                f"(average {hdi_pct}% HDI [{avg_lower:.2f}, {avg_upper:.2f}])."
            )
        else:
            # OLS model
            prose_parts.append(
                f"Staggered DiD analysis: The average post-treatment effect "
                f"across event-times was {avg_post_att:.2f}."
            )

    # Pre-treatment placebo check
    if len(pre_treatment) > 0:
        avg_pre_att = pre_treatment["att"].mean()
        # When post-treatment effects exist and are non-zero, use a relative threshold.
        # When the average post-treatment effect is (near) zero, fall back to a small
        # absolute threshold for the placebo to avoid spuriously flagging violations.
        if len(post_treatment) > 0:
            if abs(avg_post_att) > 0:
                placebo_ok = abs(avg_pre_att) < 0.1 * abs(avg_post_att)
            else:
                # No detectable average treatment effect; treat very small pre-treatment
                # effects as consistent with parallel trends.
                placebo_ok = abs(avg_pre_att) < 1e-6
        else:
            placebo_ok = True

        if placebo_ok:
            prose_parts.append(
                f"Pre-treatment placebo check: Average pre-treatment effect was "
                f"{avg_pre_att:.2f}, consistent with parallel trends assumption."
            )
        else:
            prose_parts.append(
                f"Pre-treatment placebo check: Average pre-treatment effect was "
                f"{avg_pre_att:.2f}. This may indicate violation of parallel trends."
            )

    # Number of cohorts
    n_cohorts = len(experiment.cohorts)
    prose_parts.append(f"Analysis includes {n_cohorts} treatment cohort(s).")

    text = " ".join(prose_parts)
    if group == "prior":
        text = _apply_prior_grouping(text, group)

    return EffectSummary(table=table, text=text)


def _effect_summary_rd(
    bundle,
    *,
    direction: Literal["increase", "decrease", "two-sided"] = "increase",
    alpha: float = 0.05,
    min_effect: float | None = None,
    experiment=None,
    group: Literal["prior", "posterior"] = "posterior",
):
    """Generate effect summary for Regression Discontinuity experiments."""
    discontinuity = bundle.discontinuity_at_threshold

    if has_posterior_draws(discontinuity):
        # Posterior draws present: use unified scalar functions
        hdi_prob = 1 - alpha
        stats = _compute_statistics_scalar(
            discontinuity, hdi_prob=hdi_prob, direction=direction, min_effect=min_effect
        )
        table = _generate_table_scalar(stats, index_name="discontinuity")
        text = _apply_prior_grouping(
            _generate_prose_scalar(
                stats, "discontinuity at threshold", alpha=alpha, direction=direction
            ),
            group,
        )
    elif experiment is None:
        raise TypeError(
            "_effect_summary_rd() requires the fitted experiment on the OLS "
            "path: pass experiment=self so residuals can be computed from "
            "the fitted model."
        )
    else:
        # OLS model: calculate from model
        stats = _compute_statistics_rd_ols(experiment, alpha=alpha)
        table = _generate_table_rd_ols(stats)
        text = _apply_prior_grouping(_generate_prose_rd_ols(stats, alpha=alpha), group)

    return EffectSummary(table=table, text=text)


# ==============================================================================
# Window and counterfactual extraction helpers
# ==============================================================================


def _select_treated_unit(data: xr.DataArray, treated_unit: str | None) -> xr.DataArray:
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


def _extract_window(
    post_impact: xr.DataArray,
    post_index: pd.Index,
    window: str | tuple | slice,
    treated_unit: str | None = None,
):
    """Extract windowed impact data based on window specification.

    Parameters
    ----------
    post_impact : xr.DataArray
        The group bundle's ``impact_post`` container with canonical
        prediction dimensions (singleton ``chain``/``draw`` for OLS backends).
    post_index : pd.Index
        Index of the post-treatment period (e.g. ``experiment.datapost.index``).
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
    impact = post_impact

    if "treated_units" in impact.dims:
        impact = _select_treated_unit(impact, treated_unit)

    # Extract window coordinates based on window specification
    if window == "post":
        # Use all post-treatment time points
        window_coords = post_index
    elif isinstance(window, tuple) and len(window) == 2:
        # Handle (start, end) tuple
        start, end = window
        if isinstance(post_index, pd.DatetimeIndex):
            # Datetime index - convert to timestamps if needed
            if not isinstance(start, pd.Timestamp):
                start = pd.Timestamp(start)
            if not isinstance(end, pd.Timestamp):
                end = pd.Timestamp(end)
            window_coords = post_index[(post_index >= start) & (post_index <= end)]
        else:
            # Integer index - filter by value
            start_val = int(start)
            end_val = int(end)
            mask = (post_index >= start_val) & (post_index <= end_val)
            window_coords = post_index[mask]
    elif isinstance(window, slice):
        # Handle slice object
        if isinstance(post_index, pd.DatetimeIndex):
            # For datetime, slice works directly
            window_coords = post_index[window]
        else:
            # For integer indices, convert slice to value-based filtering
            start_val = (
                int(window.start) if window.start is not None else post_index.min()
            )
            stop_val = (
                int(window.stop) if window.stop is not None else post_index.max() + 1
            )
            step = int(window.step) if window.step is not None else 1
            # Create boolean mask for values in range
            mask = (post_index >= start_val) & (post_index < stop_val)
            window_coords = post_index[mask][::step]
    else:
        raise ValueError(
            f"window must be 'post', a tuple (start, end), or a slice. Got {type(window)}"
        )

    # Apply window selection to post_impact
    windowed_impact = impact if window == "post" else impact.sel(obs_ind=window_coords)

    # Validate window is not empty
    if len(window_coords) == 0:
        raise ValueError("Window contains no time points")

    return windowed_impact, window_coords


def _extract_counterfactual(
    post_pred: xr.DataArray,
    window_coords,
    treated_unit: str | None = None,
):
    """Extract counterfactual predictions for the window.

    Parameters
    ----------
    post_pred : xr.DataArray
        The group bundle's ``predictions_post`` container with canonical
        prediction dimensions (singleton ``chain``/``draw`` for OLS backends).
    window_coords : pd.Index
        Window coordinates from :func:`_extract_window`
    treated_unit : str, optional
        For multi-unit experiments, specify which treated unit to analyze

    Returns
    -------
    xr.DataArray
        Counterfactual predictions for the window
    """
    pred = post_pred
    if "treated_units" in pred.dims:
        pred = _select_treated_unit(pred, treated_unit)
    return pred.sel(obs_ind=window_coords)


def _effect_summary_timeseries(
    windowed_impact: xr.DataArray,
    counterfactual: xr.DataArray,
    window_coords,
    *,
    direction: Literal["increase", "decrease", "two-sided"] = "increase",
    alpha: float = 0.05,
    cumulative: bool = True,
    relative: bool = True,
    min_effect: float | None = None,
    prefix: str = "Post-period",
    experiment_type: str | None = None,
    group: Literal["prior", "posterior"] = "posterior",
) -> EffectSummary:
    """Build an :class:`EffectSummary` for time-series experiments (ITS, SC,
    Piecewise ITS) from canonical impact/counterfactual containers.

    The single irreducible statistical branch lives here, keyed on the
    container itself: predictions carrying posterior draws are summarized
    with HDIs, tail probabilities, and ROPE, while single-draw (point
    estimate) predictions report frequentist t-based intervals — an HDI
    computed from a singleton draw would be silently meaningless.

    Parameters
    ----------
    windowed_impact : xr.DataArray
        Causal impact in the analysis window with canonical prediction
        dimensions.
    counterfactual : xr.DataArray
        Counterfactual predictions in the analysis window with canonical
        prediction dimensions.
    window_coords : pd.Index
        Window coordinates from :func:`_extract_window`.
    direction : {"increase", "decrease", "two-sided"}, default "increase"
        Selects the Bayesian tail probability reported in prose and the
        direction-sensitive ``p_rope`` table column. It does not change the
        symmetric HDI+ROPE conclusion.
    alpha : float, default 0.05
        Interval tail mass. Bayesian ``effect_summary`` reports an HDI with
        probability ``1 - alpha`` (95% by default), independently of the
        project-wide :data:`~causalpy.constants.HDI_PROB` setting.
    cumulative : bool, default True
        Whether to include cumulative effect statistics.
    relative : bool, default True
        Whether to include relative effect statistics.
    min_effect : float, optional
        Finite, non-negative ROPE half-width. Supplying it uses the closed
        symmetric ROPE ``[-min_effect, min_effect]`` for the three-way
        HDI+ROPE conclusion; omitting it leaves prose descriptive.
    prefix : str, default "Post-period"
        Prefix for prose generation.
    experiment_type : str, optional
        Experiment tag ("its", "sc", "piecewise_its") for tailored
        assumptions text.
    """
    if has_posterior_draws(windowed_impact):
        hdi_prob = 1 - alpha
        stats = _compute_statistics(
            windowed_impact,
            counterfactual,
            hdi_prob=hdi_prob,
            direction=direction,
            cumulative=cumulative,
            relative=relative,
            min_effect=min_effect,
        )
        table = _generate_table(stats, cumulative=cumulative, relative=relative)

        cf_avg = _as_scalar(counterfactual.mean(dim=["obs_ind", "chain", "draw"]))
        obs_avg = cf_avg + stats["avg"]["mean"]
        cf_cum = _as_scalar(
            counterfactual.sum(dim="obs_ind").mean(dim=["chain", "draw"])
        )
        obs_cum = cf_cum + stats["cum"]["mean"] if cumulative else None

        text = _generate_prose_detailed(
            stats,
            window_coords,
            alpha=alpha,
            direction=direction,
            cumulative=cumulative,
            relative=relative,
            prefix=prefix,
            observed_avg=obs_avg,
            counterfactual_avg=cf_avg,
            observed_cum=obs_cum,
            counterfactual_cum=cf_cum if cumulative else None,
            experiment_type=experiment_type,
        )
    else:
        impact_array = np.asarray(windowed_impact.isel(chain=0, draw=0))
        counterfactual_array = np.asarray(counterfactual.isel(chain=0, draw=0))

        stats = _compute_statistics_ols(
            impact_array,
            counterfactual_array,
            alpha=alpha,
            cumulative=cumulative,
            relative=relative,
        )
        table = _generate_table_ols(stats, cumulative=cumulative, relative=relative)

        cf_avg = float(np.mean(counterfactual_array))
        obs_avg = cf_avg + stats["avg"]["mean"]
        cf_cum = float(np.sum(counterfactual_array))
        obs_cum = cf_cum + stats["cum"]["mean"] if cumulative else None

        text = _generate_prose_detailed_ols(
            stats,
            window_coords,
            alpha=alpha,
            cumulative=cumulative,
            relative=relative,
            prefix=prefix,
            observed_avg=obs_avg,
            counterfactual_avg=cf_avg,
            observed_cum=obs_cum,
            counterfactual_cum=cf_cum if cumulative else None,
            experiment_type=experiment_type,
        )

    if group == "prior":
        text = _apply_prior_grouping(text, group)

    return EffectSummary(table=table, text=text)


def _compute_statistics(
    impact,
    counterfactual,
    hdi_prob=HDI_PROB,
    direction="increase",
    cumulative=True,
    relative=True,
    min_effect=None,
    time_dim="obs_ind",
):
    """Compute all summary statistics from posterior draws.

    Notes
    -----
    All in-tree callers pass ``hdi_prob`` explicitly (typically derived from
    ``effect_summary``'s ``alpha`` as ``hdi_prob = 1 - alpha``), so this
    default is effectively unused; it is set to :data:`HDI_PROB` to keep the
    project-wide convention consistent.
    """
    min_effect = _validate_min_effect(min_effect)

    stats = {}

    # Average effect over window
    avg_effect = impact.mean(dim=time_dim)
    avg_effect = _finite_posterior_draws(avg_effect)
    stats["avg"] = {
        "mean": _as_scalar(avg_effect.mean(dim=["chain", "draw"])),
        "median": _as_scalar(avg_effect.median(dim=["chain", "draw"])),
    }

    # HDI for average
    stats["avg"]["hdi_lower"], stats["avg"]["hdi_upper"] = hdi_bounds(
        avg_effect, prob=hdi_prob
    )

    # Tail probabilities for average
    avg_tail_probabilities = _compute_tail_probabilities(avg_effect, direction)
    stats["avg"].update(avg_tail_probabilities)

    # ROPE for average
    if min_effect is not None:
        stats["avg"]["p_rope"] = _compute_rope_probability(
            avg_effect, min_effect, direction
        )

    stats["avg"]["decision"] = _make_bayesian_decision(
        avg_effect,
        hdi_lower=stats["avg"]["hdi_lower"],
        hdi_upper=stats["avg"]["hdi_upper"],
        tail_probabilities=avg_tail_probabilities,
        direction=direction,
        min_effect=min_effect,
    )

    # Cumulative effect
    if cumulative:
        # Use cumulative sum over window
        cum_effect = impact.cumsum(dim=time_dim)
        # Take final value (cumulative over entire window)
        cum_final = cum_effect.isel({time_dim: -1})
        cum_final = _finite_posterior_draws(cum_final)

        stats["cum"] = {
            "mean": _as_scalar(cum_final.mean(dim=["chain", "draw"])),
            "median": _as_scalar(cum_final.median(dim=["chain", "draw"])),
        }

        # HDI for cumulative
        stats["cum"]["hdi_lower"], stats["cum"]["hdi_upper"] = hdi_bounds(
            cum_final, prob=hdi_prob
        )

        # Tail probabilities for cumulative
        cum_tail_probabilities = _compute_tail_probabilities(cum_final, direction)
        stats["cum"].update(cum_tail_probabilities)

        # ROPE for cumulative
        if min_effect is not None:
            stats["cum"]["p_rope"] = _compute_rope_probability(
                cum_final, min_effect, direction
            )

        stats["cum"]["decision"] = _make_bayesian_decision(
            cum_final,
            hdi_lower=stats["cum"]["hdi_lower"],
            hdi_upper=stats["cum"]["hdi_upper"],
            tail_probabilities=cum_tail_probabilities,
            direction=direction,
            min_effect=min_effect,
        )

    # Relative effects
    if relative:
        epsilon = 1e-8  # Guard against division by zero
        counterfactual_mean = counterfactual.mean(dim=time_dim)
        rel_avg = (avg_effect / (counterfactual_mean + epsilon)) * 100

        stats["avg"]["relative_mean"] = _as_scalar(rel_avg.mean(dim=["chain", "draw"]))

        (
            stats["avg"]["relative_hdi_lower"],
            stats["avg"]["relative_hdi_upper"],
        ) = hdi_bounds(rel_avg, prob=hdi_prob)

        if cumulative:
            # Relative cumulative: (cumulative effect / cumulative counterfactual) * 100
            counterfactual_cum = counterfactual.cumsum(dim=time_dim).isel(
                {time_dim: -1}
            )
            rel_cum = (cum_final / (counterfactual_cum + epsilon)) * 100

            stats["cum"]["relative_mean"] = _as_scalar(
                rel_cum.mean(dim=["chain", "draw"])
            )

            (
                stats["cum"]["relative_hdi_lower"],
                stats["cum"]["relative_hdi_upper"],
            ) = hdi_bounds(rel_cum, prob=hdi_prob)

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


def _generate_prose_detailed(
    stats,
    window_coords,
    alpha=0.05,
    direction="increase",
    cumulative=True,
    relative=True,
    prefix="Post-period",
    observed_avg: float | None = None,
    counterfactual_avg: float | None = None,
    observed_cum: float | None = None,
    counterfactual_cum: float | None = None,
    experiment_type: str | None = None,
):
    """Generate detailed multi-paragraph narrative report.

    This function produces a comprehensive plain-language interpretation of the
    causal effect, including observed vs counterfactual values, statistical
    credibility assessment, assumptions, and guidance on interpretation.

    Parameters
    ----------
    stats : dict
        Statistics dictionary from _compute_statistics()
    window_coords : pd.Index
        Window coordinates for the analysis period
    alpha : float, default=0.05
        Significance level for HDI interval
    direction : {"increase", "decrease", "two-sided"}, default="increase"
        Direction for tail probability interpretation.
    cumulative : bool, default=True
        Whether cumulative effects were computed
    relative : bool, default=True
        Whether relative effects were computed
    prefix : str, default="Post-period"
        Prefix describing the analysis window
    observed_avg : float, optional
        Average observed response in the analysis window
    counterfactual_avg : float, optional
        Average counterfactual prediction in the analysis window
    observed_cum : float, optional
        Cumulative observed response in the analysis window
    counterfactual_cum : float, optional
        Cumulative counterfactual prediction in the analysis window
    experiment_type : str, optional
        Type of experiment ("its", "sc", "piecewise_its") for tailored
        assumptions text

    Returns
    -------
    str
        Detailed multi-paragraph narrative report
    """
    hdi_coverage = _format_probability_as_percent(1 - alpha)

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

    # The attached decision is the sole source for the reported average HDI.
    # This keeps the interval, tail, and optional ROPE verdict coherent.
    decision = stats["avg"]["decision"]
    avg_mean = stats["avg"]["mean"]
    avg_lower, avg_upper = decision.interval

    # Paragraph 1: Observed vs counterfactual (average)
    paragraphs = []

    if observed_avg is not None and counterfactual_avg is not None:
        # Counterfactual interval: since effect = observed - counterfactual,
        # counterfactual = observed - effect, so the HDI of the counterfactual
        # is [observed - effect_upper, observed - effect_lower].
        cf_interval_lower = observed_avg - avg_upper
        cf_interval_upper = observed_avg - avg_lower

        para1 = (
            f"During the {prefix} ({window_str}), the response variable had "
            f"an average value of approx. {fmt_num(observed_avg)}. By contrast, in the "
            f"absence of an intervention, we would have expected an average response of "
            f"{fmt_num(counterfactual_avg)}. The {hdi_coverage} interval of this counterfactual "
            f"prediction is [{fmt_num(cf_interval_lower)}, "
            f"{fmt_num(cf_interval_upper)}]. Subtracting this prediction "
            f"from the observed response yields an estimate of the causal effect the "
            f"intervention had on the response variable. This effect is {fmt_num(avg_mean)} "
            f"with a {hdi_coverage} interval of [{fmt_num(avg_lower)}, {fmt_num(avg_upper)}]."
        )
    else:
        para1 = (
            f"During the {prefix} ({window_str}), the estimated average causal "
            f"effect of the intervention is {fmt_num(avg_mean)} "
            f"({hdi_coverage} HDI [{fmt_num(avg_lower)}, {fmt_num(avg_upper)}]). "
            f"This represents the difference between the observed response and the "
            f"counterfactual prediction of what would have occurred without the intervention."
        )
    paragraphs.append(para1)

    # Paragraph 2: Cumulative effect (if applicable)
    if cumulative and "cum" in stats:
        cumulative_decision = stats["cum"]["decision"]
        cum_mean = stats["cum"]["mean"]
        cum_lower, cum_upper = cumulative_decision.interval

        if observed_cum is not None and counterfactual_cum is not None:
            cum_cf_lower = observed_cum - cum_upper
            cum_cf_upper = observed_cum - cum_lower

            para2 = (
                f"Summing up the individual data points during the {prefix}, "
                f"the response variable had an overall value of {fmt_num(observed_cum)}. "
                f"By contrast, had the intervention not taken place, we would have expected "
                f"a sum of {fmt_num(counterfactual_cum)}. The {hdi_coverage} interval of this "
                f"prediction is [{fmt_num(cum_cf_lower)}, {fmt_num(cum_cf_upper)}]. "
                f"The cumulative effect is {fmt_num(cum_mean)} with a {hdi_coverage} HDI "
                f"[{fmt_num(cum_lower)}, {fmt_num(cum_upper)}]."
            )
        else:
            para2 = (
                f"The cumulative effect over the {prefix} "
                f"was {fmt_num(cum_mean)} ({hdi_coverage} HDI [{fmt_num(cum_lower)}, "
                f"{fmt_num(cum_upper)}])."
            )
        paragraphs.append(para2)

    # Paragraph 3: posterior summaries rendered from attached decisions.
    credibility_parts = [_render_bayesian_decision(decision, hdi_coverage)]
    if cumulative and "cum" in stats:
        credibility_parts.append(
            "For the cumulative effect, "
            f"{_render_bayesian_decision(cumulative_decision, hdi_coverage)}"
        )

    if relative and "relative_mean" in stats["avg"]:
        rel_mean = stats["avg"]["relative_mean"]
        rel_lower = stats["avg"]["relative_hdi_lower"]
        rel_upper = stats["avg"]["relative_hdi_upper"]
        credibility_parts.append(
            f"Relative to the counterfactual, the effect represents a "
            f"{fmt_num(rel_mean)}% change ({hdi_coverage} HDI [{fmt_num(rel_lower)}%, "
            f"{fmt_num(rel_upper)}%])."
        )

    para3 = " ".join(credibility_parts)
    paragraphs.append(para3)

    # Paragraph 4: Assumptions and guidance
    para4 = _assumptions_text(experiment_type)
    para4 += (
        "We recommend inspecting model fit, examining pre-intervention trends, "
        "and conducting sensitivity analyses (e.g., placebo tests) to support "
        "any causal conclusions drawn from this analysis."
    )
    paragraphs.append(para4)

    return "\n\n".join(paragraphs)


def _assumptions_text(experiment_type: str | None = None) -> str:
    """Return the assumptions preamble tailored to the experiment type.

    Parameters
    ----------
    experiment_type : str, optional
        One of "its", "sc", "piecewise_its", or None for a generic default.

    Returns
    -------
    str
        Assumptions preamble (ends with a trailing space for appending guidance).
    """
    if experiment_type == "its":
        return (
            "This analysis assumes that the relationship between the time-based "
            "predictors and the response observed during the pre-intervention period "
            "remains stable throughout the post-intervention period. If the formula "
            "includes external covariates, it further assumes they were not themselves "
            "affected by the intervention. "
        )
    elif experiment_type == "sc":
        return (
            "This analysis assumes that the control units used to construct the "
            "synthetic counterfactual were not themselves affected by the intervention, "
            "and that the pre-treatment relationship between control and treated units "
            "remains stable throughout the post-treatment period. "
        )
    else:
        return (
            "This analysis assumes that the covariates used to construct the "
            "counterfactual were not themselves affected by the intervention. It also "
            "assumes that the relationship between the covariates and the response "
            "observed during the pre-intervention period remains stable throughout "
            "the post-intervention period. "
        )


def _generate_prose_detailed_ols(
    stats,
    window_coords,
    alpha=0.05,
    cumulative=True,
    relative=True,
    prefix="Post-period",
    observed_avg: float | None = None,
    counterfactual_avg: float | None = None,
    observed_cum: float | None = None,
    counterfactual_cum: float | None = None,
    experiment_type: str | None = None,
):
    """Generate detailed multi-paragraph narrative report for OLS models.

    This function produces a comprehensive plain-language interpretation of the
    causal effect from OLS models, including observed vs counterfactual values,
    statistical significance assessment, assumptions, and guidance on interpretation.

    Parameters
    ----------
    stats : dict
        Statistics dictionary from _compute_statistics_ols()
    window_coords : pd.Index
        Window coordinates for the analysis period
    alpha : float, default=0.05
        Significance level for CI interval
    cumulative : bool, default=True
        Whether cumulative effects were computed
    relative : bool, default=True
        Whether relative effects were computed
    prefix : str, default="Post-period"
        Prefix describing the analysis window
    observed_avg : float, optional
        Average observed response in the analysis window
    counterfactual_avg : float, optional
        Average counterfactual prediction in the analysis window
    observed_cum : float, optional
        Cumulative observed response in the analysis window
    counterfactual_cum : float, optional
        Cumulative counterfactual prediction in the analysis window
    experiment_type : str, optional
        Type of experiment ("its", "sc", "piecewise_its") for tailored
        assumptions text

    Returns
    -------
    str
        Detailed multi-paragraph narrative report
    """
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

    # Extract statistics
    avg_mean = stats["avg"]["mean"]
    avg_lower = stats["avg"]["ci_lower"]
    avg_upper = stats["avg"]["ci_upper"]
    p_val = stats["avg"]["p_value"]

    # Paragraph 1: Average effect description
    paragraphs = []

    if observed_avg is not None and counterfactual_avg is not None:
        cf_interval_lower = observed_avg - avg_upper
        cf_interval_upper = observed_avg - avg_lower

        para1 = (
            f"During the {prefix} ({window_str}), the response variable had "
            f"an average value of approx. {fmt_num(observed_avg)}. By contrast, in the "
            f"absence of an intervention, we would have expected an average response of "
            f"{fmt_num(counterfactual_avg)}. The {ci_pct}% confidence interval of this "
            f"counterfactual prediction is [{fmt_num(cf_interval_lower)}, "
            f"{fmt_num(cf_interval_upper)}]. Subtracting this prediction "
            f"from the observed response yields an estimate of the causal effect the "
            f"intervention had on the response variable. This effect is {fmt_num(avg_mean)} "
            f"with a {ci_pct}% confidence interval of [{fmt_num(avg_lower)}, "
            f"{fmt_num(avg_upper)}]."
        )
    else:
        para1 = (
            f"During the {prefix} ({window_str}), the estimated average causal "
            f"effect of the intervention is {fmt_num(avg_mean)} "
            f"({ci_pct}% CI [{fmt_num(avg_lower)}, {fmt_num(avg_upper)}]). "
            f"This represents the difference between the observed response and the "
            f"counterfactual prediction of what would have occurred without the intervention."
        )
    paragraphs.append(para1)

    # Paragraph 2: Cumulative effect (if applicable)
    if cumulative and "cum" in stats:
        cum_mean = stats["cum"]["mean"]
        cum_lower = stats["cum"]["ci_lower"]
        cum_upper = stats["cum"]["ci_upper"]

        if observed_cum is not None and counterfactual_cum is not None:
            cum_cf_lower = observed_cum - cum_upper
            cum_cf_upper = observed_cum - cum_lower

            para2 = (
                f"Summing up the individual data points during the {prefix}, "
                f"the response variable had an overall value of {fmt_num(observed_cum)}. "
                f"By contrast, had the intervention not taken place, we would have expected "
                f"a sum of {fmt_num(counterfactual_cum)}. The {ci_pct}% confidence interval "
                f"of this prediction is [{fmt_num(cum_cf_lower)}, {fmt_num(cum_cf_upper)}]."
            )
        else:
            para2 = (
                f"The cumulative effect over the {prefix} "
                f"was {fmt_num(cum_mean)} ({ci_pct}% CI [{fmt_num(cum_lower)}, "
                f"{fmt_num(cum_upper)}])."
            )
        paragraphs.append(para2)

    # Paragraph 3: Statistical summary
    ci_excludes_zero = (avg_lower > 0) or (avg_upper < 0)

    significance_parts = []
    if ci_excludes_zero:
        significance_parts.append(
            f"The {ci_pct}% confidence interval of the effect [{fmt_num(avg_lower)}, "
            f"{fmt_num(avg_upper)}] does not include zero (p-value {fmt_num(p_val, 3)})."
        )
    else:
        significance_parts.append(
            f"The {ci_pct}% confidence interval of the effect [{fmt_num(avg_lower)}, "
            f"{fmt_num(avg_upper)}] includes zero (p-value {fmt_num(p_val, 3)})."
        )

    if relative and "relative_mean" in stats["avg"]:
        rel_mean = stats["avg"]["relative_mean"]
        rel_lower = stats["avg"]["relative_ci_lower"]
        rel_upper = stats["avg"]["relative_ci_upper"]
        significance_parts.append(
            f"Relative to the counterfactual, the effect represents a "
            f"{fmt_num(rel_mean)}% change ({ci_pct}% CI [{fmt_num(rel_lower)}%, "
            f"{fmt_num(rel_upper)}%])."
        )

    para3 = " ".join(significance_parts)
    paragraphs.append(para3)

    # Paragraph 4: Assumptions and guidance
    para4 = _assumptions_text(experiment_type)
    para4 += (
        "We recommend inspecting model fit, examining pre-intervention trends, "
        "and conducting sensitivity analyses (e.g., placebo tests) to support "
        "any causal conclusions drawn from this analysis."
    )
    paragraphs.append(para4)

    return "\n\n".join(paragraphs)


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


def _point_residuals(experiment) -> np.ndarray:
    """In-sample point residuals via the canonical prediction container.

    Uses the model adapter's canonical ``predict`` output collapsed over
    ``chain``/``draw``, so the t-based point-summary path works for any
    backend. This path is only reached for singleton containers
    (``chain * draw == 1``), where the mean is exactly the single point
    estimate; taking the mean (rather than the first draw) keeps the helper
    well-defined even if a many-draw container ever slips through.

    ``y`` may have shape ``(n, 1)`` with dims ``(obs_ind, treated_units)``
    while the fitted values are conceptually ``(n,)``; both are flattened to
    1-D so they align positionally on ``obs_ind`` (letting xarray align them
    would broadcast against ``treated_units`` and produce an ``(n, n)``
    array).
    """
    y = np.asarray(experiment.design["y"]).reshape(-1)
    pred = experiment._model_backend.predict(X=np.asarray(experiment.design["X"]))
    y_fitted = np.asarray(pred.mean(dim=["chain", "draw"])).reshape(-1)
    return y - y_fitted


def _compute_statistics_did_ols(
    experiment,
    alpha=0.05,
):
    """Compute statistics for DiD scalar effect with OLS model.

    Parameters
    ----------
    experiment
        Fitted DiD experiment with OLS model
    alpha : float
        Significance level

    Returns
    -------
    dict
        Dictionary of statistics
    """
    causal_impact = _as_scalar(experiment.result.causal_impact)

    # Calculate standard error from model residuals
    residuals = _point_residuals(experiment)
    X_da = experiment.design["X"]
    n, p = X_da.shape
    df = n - p
    # Unbiased estimator of the residual variance: SSR / (n - p), consistent
    # with the degrees of freedom used below for the t-distribution.
    mse = np.sum(residuals**2) / df

    # Find the interaction term coefficient index. patsy names interaction
    # columns by formula order (e.g. "post_treatment[T.True]:group" for a
    # formula written as "post_treatment*group"), so match structurally via
    # the same helper the experiment uses to locate the causal_impact
    # coefficient, rather than a concatenated "group:post_treatment" string.
    coeff_idx = next(
        (
            i
            for i, label in enumerate(experiment.labels)
            if experiment._is_treatment_interaction(label)
        ),
        None,
    )

    if coeff_idx is None:
        raise ValueError(
            f"Could not find interaction term between '{experiment.group_variable_name}' "
            f"and '{experiment.post_treatment_variable_name}' in model"
        )

    X = X_da
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


def _compute_statistics_rd_ols(experiment, alpha=0.05):
    """Compute statistics for RD scalar effect with OLS model."""
    discontinuity = _as_scalar(experiment.result.discontinuity_at_threshold)

    # Calculate standard error from model residuals
    residuals = _point_residuals(experiment)
    X_da = experiment.design["X"]
    n, p = X_da.shape
    df = n - p
    # Unbiased estimator of the residual variance: SSR / (n - p), consistent
    # with the degrees of freedom used below for the t-distribution.
    mse = np.sum(residuals**2) / df

    try:
        threshold_design = np.asarray(experiment.x_discon_design, dtype=float)
    except AttributeError as err:
        raise ValueError(
            "Cannot compute the RD threshold-contrast standard error because "
            "the threshold design rows are unavailable."
        ) from err
    except (TypeError, ValueError) as err:
        raise ValueError(
            "RD threshold design must be a finite numeric two-row array."
        ) from err

    if threshold_design.ndim != 2 or threshold_design.shape[0] != 2:
        raise ValueError(
            "RD threshold design must contain exactly two rows: below threshold "
            "followed by above threshold."
        )

    if not np.isfinite(threshold_design).all():
        raise ValueError("RD threshold design must be a finite numeric two-row array.")

    X = np.asarray(X_da)
    if threshold_design.shape[1] != X.shape[1]:
        raise ValueError(
            "RD threshold design must have the same number of columns as the "
            "fitted design matrix."
        )

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            "Cannot compute the RD threshold-contrast standard error because "
            "X.T @ X is singular."
        ) from err

    # discontinuity_at_threshold is the prediction above the threshold minus
    # the prediction below it, so its uncertainty must use that same contrast.
    contrast = threshold_design[1] - threshold_design[0]
    se = np.sqrt(mse * contrast @ XtX_inv @ contrast)

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
    bundle,
    *,
    direction: Literal["increase", "decrease", "two-sided"] = "increase",
    alpha: float = 0.05,
    min_effect: float | None = None,
    group: Literal["prior", "posterior"] = "posterior",
):
    """Generate effect summary for Regression Kink experiments."""
    gradient_change = bundle.gradient_change

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
        text = _apply_prior_grouping(
            _generate_prose_scalar(
                stats,
                "change in gradient at the kink point",
                alpha=alpha,
                direction=direction,
            ),
            group,
        )
    else:
        raise NotImplementedError(
            "OLS models are not currently supported for Regression Kink experiments. "
            "Please use a PyMC model for full statistical inference. "
            "If OLS support is needed, see _compute_statistics_rd_ols() "
            "for the implementation pattern."
        )

    return EffectSummary(table=table, text=text)
