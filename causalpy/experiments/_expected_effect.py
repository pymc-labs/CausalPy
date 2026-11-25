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

from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

UNTREATED_ONLY_REASON = (
    "expected_effect unavailable: this fit only supplies the untreated "
    "trajectory (mu_untreated)"
)

EffectChoice = Literal["expected_effect", "observed_gap"]


def response_contrast(
    mu_treated: xr.DataArray,
    mu_untreated: xr.DataArray,
    *,
    weights: xr.DataArray | None = None,
    reduce_dims: tuple[str, ...] = ("obs_ind", "treated_units"),
    point: bool = False,
) -> xr.DataArray | float:
    """Average ``mu_treated - mu_untreated`` over the target dimensions.

    Both arrays must already be response-scale expectations aligned on the
    shared sample dimensions. ``weights``, when given, are broadcast to the
    difference and applied only on ``reduce_dims``. ``point=True`` drops
    singleton ``chain`` and ``draw`` dimensions and returns a Python float.

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
    point : bool, default False
        Drop singleton sampling dimensions and return a Python float.

    Returns
    -------
    xarray.DataArray or float
        The averaged contrast, or a float when ``point`` is true.
    """
    difference = mu_treated - mu_untreated
    present = tuple(dim for dim in reduce_dims if dim in difference.dims)
    if weights is not None and present:
        aligned = weights.broadcast_like(difference)
        weighted = difference * aligned
        denominator = aligned.sum(dim=present)
        reduced = weighted.sum(dim=present) / denominator
    elif present:
        reduced = difference.mean(dim=present)
    else:
        reduced = difference
    if point:
        return as_point(reduced)
    return reduced


def as_point(value: xr.DataArray) -> float:
    """Return a Python float after dropping singleton sampling dimensions.

    Parameters
    ----------
    value : xarray.DataArray
        Contrast whose ``chain`` and ``draw`` sizes are 1 or absent.

    Returns
    -------
    float
        The point contrast.
    """
    squeezed = value
    for dim in ("chain", "draw"):
        if dim not in squeezed.dims:
            continue
        if squeezed.sizes[dim] != 1:
            raise ValueError(
                f"Point contrast still has {dim} size {int(squeezed.sizes[dim])}."
            )
        squeezed = squeezed.squeeze(dim, drop=True)
    return float(np.asarray(squeezed).reshape(()))


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


def headline_mean(series: xr.DataArray | float) -> float:
    """Posterior mean of within-draw window averages, or the point average.

    Parameters
    ----------
    series : xarray.DataArray or float
        Window of response-scale contrasts, or one point contrast.

    Returns
    -------
    float
        Mean of within-draw averages, or the point average.
    """
    if not isinstance(series, xr.DataArray):
        return float(series)
    reduced = series.mean(dim="obs_ind") if "obs_ind" in series.dims else series
    return float(np.asarray(reduced.mean()).reshape(()))


def cumulative_total(series: xr.DataArray | float) -> float:
    """Posterior mean of within-draw sums, or the point sum.

    Parameters
    ----------
    series : xarray.DataArray or float
        Window of response-scale contrasts, or one point contrast.

    Returns
    -------
    float
        Mean of within-draw sums, or the point sum.
    """
    if not isinstance(series, xr.DataArray):
        return float(series)
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
    series: xr.DataArray | float | None,
    *,
    reason: str | None = None,
    datapost_index: pd.Index | None = None,
    cumulative: bool = True,
    group: Literal["prior", "posterior"] = "posterior",
) -> Any:
    """Headline the mean expected effect, or explain why it is unavailable.

    Bayesian series keep their sampling dimensions. The headline is the mean
    of the within-draw window averages. A point series is summarized by its
    window mean alone: no credible interval and no posterior probability.
    ``datapost_index`` is accepted so callers can name the window; the series
    is already the post-intervention contrast.

    Parameters
    ----------
    series : xarray.DataArray, float, or None
        Stored expected effect. ``None`` reports ``reason``.
    reason : str, optional
        Explanation used when ``series`` is missing.
    datapost_index : pandas.Index, optional
        Accepted so callers can name the window. Not used for the headline.
    cumulative : bool, default True
        Whether to append the cumulative sum.
    group : {"prior", "posterior"}, default "posterior"
        Draw group named in the summary prose.

    Returns
    -------
    EffectSummary
        Headline table and prose, or an unavailable summary.
    """
    del datapost_index
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
