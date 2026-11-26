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
"""Response-scale expected-effect contrasts.

Private to this package. Callers supply response-scale conditional means
from ``predict()``. This module does not read realized outcomes, does not
apply an inverse link, and does not fit a GLM.
"""

from __future__ import annotations

from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

UNTREATED_ONLY_REASON = (
    "expected_effect unavailable: this fit only supplies the untreated "
    "trajectory (mu_untreated)"
)

EffectChoice = Literal["expected_effect", "observed_gap"]


class HeadlineEffect(NamedTuple):
    """One resolved series for a plot or summary."""

    pointwise: xr.DataArray | None
    cumulative: xr.DataArray | None
    point_title: str
    cumulative_title: str
    label: str
    reason: str | None
    shade_gap: bool


def select_headline_effect(
    bundle: Any,
    effect: str,
    *,
    treated_unit: str | None = None,
) -> HeadlineEffect:
    """Resolve ``observed_gap`` or ``expected_effect`` once.

    ``observed_gap`` is realized minus counterfactual. ``expected_effect`` is
    the stored response-scale contrast, or the reason a counterfactual-only
    fit cannot supply it. A treated unit is selected when that dimension is
    present.

    Parameters
    ----------
    bundle : CausalResult
        Result whose impact and expected-effect fields are already stored.
    effect : {"expected_effect", "observed_gap"}
        Series to resolve.
    treated_unit : str, optional
        Unit selected when the series has a ``treated_units`` dimension.

    Returns
    -------
    HeadlineEffect
        Pointwise series, cumulative series, titles, and shade flag.

    Raises
    ------
    ValueError
        If ``effect`` is not one of the two stored series.
    """
    if effect == "observed_gap":
        pointwise = _select_unit(bundle.impact_post, treated_unit)
        cumulative = _select_unit(bundle.impact_post_cumulative, treated_unit)
        return HeadlineEffect(
            pointwise=pointwise,
            cumulative=cumulative,
            point_title="Observed gap",
            cumulative_title="Cumulative observed gap",
            label="Observed gap",
            reason=None,
            shade_gap=True,
        )
    if effect != "expected_effect":
        raise ValueError(
            f"effect must be 'expected_effect' or 'observed_gap', got {effect!r}."
        )
    series = getattr(bundle, "expected_effect", None)
    if series is None:
        reason = getattr(bundle, "expected_effect_reason", None)
        return HeadlineEffect(
            pointwise=None,
            cumulative=None,
            point_title="Expected effect unavailable",
            cumulative_title="Cumulative expected effect unavailable",
            label="Expected effect unavailable",
            reason=reason,
            shade_gap=False,
        )
    pointwise = _select_unit(series, treated_unit)
    return HeadlineEffect(
        pointwise=pointwise,
        cumulative=pointwise.cumsum(dim="obs_ind"),
        point_title="Expected effect",
        cumulative_title="Cumulative expected effect",
        label="Expected effect",
        reason=None,
        shade_gap=False,
    )


def _select_unit(series: xr.DataArray, treated_unit: str | None) -> xr.DataArray:
    if treated_unit is None or "treated_units" not in series.dims:
        return series
    return series.sel(treated_units=treated_unit)


def response_contrast(
    mu_treated: xr.DataArray,
    mu_untreated: xr.DataArray,
    *,
    weights: xr.DataArray | None = None,
    reduce_dims: tuple[str, ...] = ("obs_ind", "treated_units"),
) -> xr.DataArray:
    """Average ``mu_treated - mu_untreated`` over the target dimensions.

    Both arrays must already be response-scale expectations aligned on the
    shared sample dimensions. ``weights``, when given, are broadcast to the
    difference and applied only on ``reduce_dims``. Singleton ``chain`` and
    ``draw`` dimensions from an adapter prediction are kept. They are not
    fabricated and are not squeezed to a Python float.

    Parameters
    ----------
    mu_treated : xarray.DataArray
        Response-scale expectation under intervention.
    mu_untreated : xarray.DataArray
        Response-scale expectation under no intervention.
    weights : xarray.DataArray, optional
        Weights broadcast onto ``reduce_dims``. Equal weights when omitted.
    reduce_dims : tuple of str, default ("obs_ind", "treated_units")
        Dimensions averaged before returning the contrast.

    Returns
    -------
    xarray.DataArray
        The averaged contrast.
    """
    difference = mu_treated - mu_untreated
    present = tuple(dim for dim in reduce_dims if dim in difference.dims)
    if weights is not None and present:
        aligned = weights.broadcast_like(difference)
        weighted = difference * aligned
        denominator = aligned.sum(dim=present)
        return weighted.sum(dim=present) / denominator
    if present:
        return difference.mean(dim=present)
    return difference


def cumulative_sum(series: xr.DataArray) -> xr.DataArray:
    """Running sum along ``obs_ind``, or ``series`` when that dim is absent.

    Parameters
    ----------
    series : xarray.DataArray
        Pointwise contrast ordered on ``obs_ind``.

    Returns
    -------
    xarray.DataArray
        Cumulative sum, or ``series`` when ``obs_ind`` is absent.
    """
    if "obs_ind" not in series.dims:
        return series
    return series.cumsum(dim="obs_ind")


def headline_mean(series: xr.DataArray) -> float:
    """Posterior mean of within-draw window averages.

    Parameters
    ----------
    series : xarray.DataArray
        Window of response-scale contrasts.

    Returns
    -------
    float
        Mean of within-draw averages.
    """
    reduced = series.mean(dim="obs_ind") if "obs_ind" in series.dims else series
    return float(np.asarray(reduced.mean()).reshape(()))


def cumulative_total(series: xr.DataArray) -> float:
    """Posterior mean of within-draw sums.

    Parameters
    ----------
    series : xarray.DataArray
        Window of response-scale contrasts.

    Returns
    -------
    float
        Mean of within-draw sums.
    """
    if "obs_ind" not in series.dims:
        return headline_mean(series)
    total = series.sum(dim="obs_ind")
    return float(np.asarray(total.mean()).reshape(()))


def unavailable_summary(
    reason: str | None = None,
    *,
    group: Literal["prior", "posterior"] = "posterior",
):
    """Effect summary for a fit that cannot supply ``mu_treated``.

    Parameters
    ----------
    reason : str, optional
        Why the treated mean is missing. Defaults to the untreated-only
        explanation.
    group : {"prior", "posterior"}, default "posterior"
        Draw group named in the summary prose.

    Returns
    -------
    EffectSummary
        A one-row table whose status is ``unavailable``.
    """
    from causalpy.reporting import EffectSummary, _apply_prior_grouping

    text = _apply_prior_grouping(reason or UNTREATED_ONLY_REASON, group)
    table = pd.DataFrame({"status": ["unavailable"]}, index=["expected_effect"])
    return EffectSummary(table=table, text=text)


def summarize_expected_effect(
    series: xr.DataArray | None,
    *,
    reason: str | None = None,
    cumulative: bool = True,
    group: Literal["prior", "posterior"] = "posterior",
) -> Any:
    """Headline the mean expected effect, or explain why it is unavailable.

    Callers slice ``series`` to the requested window before calling.
    Bayesian series keep their sampling dimensions. A series without
    ``obs_ind`` is summarized by its mean.

    Parameters
    ----------
    series : xarray.DataArray, float, or None
        Windowed expected effect. ``None`` reports ``reason``.
    reason : str, optional
        Explanation used when ``series`` is missing.
    cumulative : bool, default True
        Whether to append the cumulative sum.
    group : {"prior", "posterior"}, default "posterior"
        Draw group named in the summary prose.

    Returns
    -------
    EffectSummary
        Headline table and prose, or an unavailable summary.
    """
    if series is None:
        return unavailable_summary(reason, group=group)
    from causalpy.reporting import EffectSummary, _apply_prior_grouping

    mean = headline_mean(series)
    row: dict[str, float] = {"mean": mean}
    text = (
        f"Expected effect = {mean:.4f}. "
        "Headline is the mean of within-window averages of mu_treated minus "
        "mu_untreated."
    )
    if cumulative:
        total = cumulative_total(series)
        row["cumulative_sum"] = total
        text += f" Cumulative sum = {total:.4f}."
    table = pd.DataFrame([row], index=["expected_effect"])
    return EffectSummary(table=table, text=_apply_prior_grouping(text, group))
