#   Copyright 2025 - 2026 The PyMC Labs Developers
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
"""Shared wide-panel counterfactual helpers.

``SyntheticControl`` calls these. Later wide-panel counterfactual experiments
should call them too, rather than subclassing ``SyntheticControl``. This module
is not a public API.

The post-period treated outcomes are stored so impact can be computed. They
are not a fit input. Callers must not pass them to a model, and must not use
them to tune a penalty or a rank.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from causalpy.constants import HDI_PROB, LEGEND_FONT_SIZE
from causalpy.date_utils import _combine_datetime_indices, format_date_axes
from causalpy.experiments._results import CausalResult
from causalpy.plot_utils import (
    _PosteriorPlotStyle,
    get_hdi_to_df,
    has_posterior_draws,
    plot_posterior_over_x,
)
from causalpy.reporting import (
    EffectSummary,
    _effect_summary_timeseries,
    _extract_counterfactual,
    _extract_window,
)


def panel_period_frames(
    data: pd.DataFrame, treatment_time: int | float | pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a wide panel at ``treatment_time``.

    Pre-period rows have ``index < treatment_time``. Post-period rows have
    ``index >= treatment_time``. The treatment time itself is post-period.

    Parameters
    ----------
    data : pandas.DataFrame
        Wide panel indexed by time.
    treatment_time : int, float, or pandas.Timestamp
        First post-period time. Rows at this time are post-period.

    Returns
    -------
    tuple of pandas.DataFrame
        Pre-period frame, then post-period frame.
    """
    pre = data[data.index < treatment_time]
    post = data[data.index >= treatment_time]
    return pre, post


def wide_panel_design(
    frame: pd.DataFrame, control_units: list[str], treated_units: list[str]
) -> xr.Dataset:
    """Bundle one period of a wide unit panel into control and treated arrays.

    ``control`` has dims ``("obs_ind", "coeffs")``. ``treated`` has dims
    ``("obs_ind", "treated_units")``.

    Parameters
    ----------
    frame : pandas.DataFrame
        One period of the wide panel.
    control_units : list of str
        Donor columns. These become the ``coeffs`` coordinate.
    treated_units : list of str
        Treated columns. These become the ``treated_units`` coordinate.

    Returns
    -------
    xarray.Dataset
        ``control`` and ``treated`` arrays for the period.
    """
    control = frame[control_units]
    treated = frame[treated_units]
    return xr.Dataset(
        {
            "control": xr.DataArray(
                control,
                dims=["obs_ind", "coeffs"],
                coords={"obs_ind": control.index, "coeffs": control_units},
            ),
            "treated": xr.DataArray(
                treated,
                dims=["obs_ind", "treated_units"],
                coords={"obs_ind": treated.index, "treated_units": treated_units},
            ),
        }
    )


def counterfactual_impacts(
    treated_pre: xr.DataArray,
    predictions_pre: xr.DataArray,
    treated_post: xr.DataArray,
    predictions_post: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Subtract the counterfactual from the observed treated outcomes.

    Impact is aligned on ``obs_ind``. A coordinate mismatch is an error, not a
    silent reindex. Cumulative impact is the running sum of the post-period
    impact.

    Parameters
    ----------
    treated_pre : xarray.DataArray
        Observed treated outcomes before intervention.
    predictions_pre : xarray.DataArray
        Counterfactual predictions on the same pre-period ``obs_ind``.
    treated_post : xarray.DataArray
        Observed treated outcomes from the treatment time onward.
    predictions_post : xarray.DataArray
        Counterfactual predictions on the same post-period ``obs_ind``.

    Returns
    -------
    tuple of xarray.DataArray
        Pre-period impact, post-period impact, and cumulative post-period impact.
    """
    # Impact relies on exact obs_ind alignment; a mismatch (e.g. a bare
    # ndarray X getting arange coords) would silently corrupt the subtraction.
    assert treated_pre.obs_ind.equals(predictions_pre.obs_ind)
    assert treated_post.obs_ind.equals(predictions_post.obs_ind)
    impact_pre = (treated_pre - predictions_pre).transpose(
        ..., "obs_ind", "treated_units"
    )
    impact_post = (treated_post - predictions_post).transpose(
        ..., "obs_ind", "treated_units"
    )
    return impact_pre, impact_post, impact_post.cumsum(dim="obs_ind")


def resolve_treated_unit(treated_units: list[str], treated_unit: str | None) -> str:
    """Return ``treated_unit``, or the first name when it is omitted.

    Parameters
    ----------
    treated_units : list of str
        Treated-unit names stored on the experiment.
    treated_unit : str or None
        Requested unit. ``None`` selects the first name.

    Returns
    -------
    str
        The resolved treated-unit name.

    Raises
    ------
    ValueError
        If ``treated_unit`` is not in ``treated_units``.
    """
    treated_unit = treated_unit if treated_unit is not None else treated_units[0]
    if treated_unit not in treated_units:
        raise ValueError(
            f"treated_unit '{treated_unit}' not found. Available units: {treated_units}"
        )
    return treated_unit


def convert_treatment_time_for_axis(
    axis: plt.Axes, treatment_time: int | float | pd.Timestamp
) -> int | float | pd.Timestamp:
    """Convert treatment time into the plotting units expected by an axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis whose x-axis units should receive the treatment time.
    treatment_time : int, float, or pandas.Timestamp
        Treatment time in the experiment's index units.

    Returns
    -------
    int, float, or pandas.Timestamp
        The converted coordinate, or the original value when conversion fails.
    """
    try:
        return axis.xaxis.convert_units(treatment_time)
    except (TypeError, ValueError):
        return treatment_time


def panel_plot_frame(
    datapre: pd.DataFrame,
    datapost: pd.DataFrame,
    bundle: CausalResult,
    treated_units: list[str],
    *,
    treated_unit: str | None = None,
    hdi_prob: float = HDI_PROB,
) -> pd.DataFrame:
    """Build the observed-plus-prediction frame for one treated unit.

    HDI columns are included only when the prediction container carries
    posterior draws. Point-estimate backends return ``prediction`` and
    ``impact`` only.

    Parameters
    ----------
    datapre : pandas.DataFrame
        Observed pre-period panel.
    datapost : pandas.DataFrame
        Observed post-period panel.
    bundle : CausalResult
        Predictions and impacts for the requested draw group.
    treated_units : list of str
        Treated-unit names stored on the experiment.
    treated_unit : str or None, optional
        Unit to extract. ``None`` selects the first name.
    hdi_prob : float, optional
        Probability mass of the HDI columns. Ignored for point estimates.

    Returns
    -------
    pandas.DataFrame
        Pre-period and post-period rows with prediction and impact columns.
    """
    with_uncertainty = has_posterior_draws(bundle.predictions_pre)
    hdi_pct = int(round(hdi_prob * 100))
    pre_data = datapre.copy()
    post_data = datapost.copy()
    treated_unit = resolve_treated_unit(treated_units, treated_unit)

    pre_pred = bundle.predictions_pre.sel(treated_units=treated_unit)
    post_pred = bundle.predictions_post.sel(treated_units=treated_unit)
    pre_impact = bundle.impact_pre.sel(treated_units=treated_unit)
    post_impact = bundle.impact_post.sel(treated_units=treated_unit)

    pre_data["prediction"] = pre_pred.mean(dim=["chain", "draw"]).values
    post_data["prediction"] = post_pred.mean(dim=["chain", "draw"]).values

    if with_uncertainty:
        pred_lower_col = f"pred_hdi_lower_{hdi_pct}"
        pred_upper_col = f"pred_hdi_upper_{hdi_pct}"
        pre_hdi = get_hdi_to_df(pre_pred, hdi_prob=hdi_prob)
        post_hdi = get_hdi_to_df(post_pred, hdi_prob=hdi_prob)
        pre_data[[pred_lower_col, pred_upper_col]] = pre_hdi.iloc[:, [0, -1]].values
        post_data[[pred_lower_col, pred_upper_col]] = post_hdi.iloc[:, [0, -1]].values

    pre_data["impact"] = pre_impact.mean(dim=["chain", "draw"]).values
    post_data["impact"] = post_impact.mean(dim=["chain", "draw"]).values

    if with_uncertainty:
        impact_lower_col = f"impact_hdi_lower_{hdi_pct}"
        impact_upper_col = f"impact_hdi_upper_{hdi_pct}"
        pre_impact_hdi = get_hdi_to_df(pre_impact, hdi_prob=hdi_prob)
        post_impact_hdi = get_hdi_to_df(post_impact, hdi_prob=hdi_prob)
        pre_data[[impact_lower_col, impact_upper_col]] = pre_impact_hdi.iloc[
            :, [0, -1]
        ].values
        post_data[[impact_lower_col, impact_upper_col]] = post_impact_hdi.iloc[
            :, [0, -1]
        ].values

    return pd.concat([pre_data, post_data])


def _format_panel_dates(
    axes: plt.Axes | list[plt.Axes] | Any,
    pre_index: pd.Index,
    post_index: pd.Index,
) -> None:
    if isinstance(pre_index, pd.DatetimeIndex):
        full_index = _combine_datetime_indices(
            pd.DatetimeIndex(pre_index),
            pd.DatetimeIndex(post_index),
        )
        format_date_axes(axes, full_index)


def plot_panel_prior_check(
    *,
    pre_index: pd.Index,
    post_index: pd.Index,
    pre_pred: xr.DataArray,
    post_pred: xr.DataArray,
    pre_treated: xr.DataArray,
    post_treated: xr.DataArray,
    pre_control: xr.DataArray,
    post_control: xr.DataArray,
    treatment_time: int | float | pd.Timestamp,
    style: _PosteriorPlotStyle,
    figsize: tuple[float, float],
    plot_predictors: bool,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Render the reduced prior-check panel.

    Prior-implied bands are typically far wider than the data, so the impact
    panels are dropped rather than autoscaled into uselessness.

    Parameters
    ----------
    pre_index : pandas.Index
        Pre-period time index.
    post_index : pandas.Index
        Post-period time index.
    pre_pred : xarray.DataArray
        Prior counterfactual for one treated unit, before intervention.
    post_pred : xarray.DataArray
        Prior counterfactual for one treated unit, from intervention onward.
    pre_treated : xarray.DataArray
        Observed pre-period outcome for the plotted unit.
    post_treated : xarray.DataArray
        Observed post-period outcome for the plotted unit.
    pre_control : xarray.DataArray
        Pre-period donor trajectories.
    post_control : xarray.DataArray
        Post-period donor trajectories.
    treatment_time : int, float, or pandas.Timestamp
        Time drawn as the intervention line.
    style : dict
        Interval style forwarded to the posterior plotting helper.
    figsize : tuple of float
        Figure size in inches.
    plot_predictors : bool
        Whether to overlay donor trajectories.

    Returns
    -------
    tuple
        The figure and its single axes.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    h_line, h_patch = plot_posterior_over_x(
        pre_index,
        pre_pred,
        ax=ax,
        **style,
        plot_hdi_kwargs={"color": "C0"},
    )
    ax.plot(pre_index, pre_treated, "k.", label="Observations")
    plot_posterior_over_x(
        post_index,
        post_pred,
        ax=ax,
        **style,
        plot_hdi_kwargs={"color": "C1"},
    )
    ax.plot(post_index, post_treated, "k.", zorder=3)
    converted = convert_treatment_time_for_axis(ax, treatment_time)
    ax.axvline(x=converted, ls="-", lw=3, color="r", zorder=1.5)
    ax.legend(
        handles=[tuple(h_line) if isinstance(h_line, list) else (h_line, h_patch)],
        labels=["Prior counterfactual"],
        fontsize=LEGEND_FONT_SIZE,
    )
    ax.set(title="Prior predictive check")
    if plot_predictors:
        ax.plot(pre_index, pre_control, "-", c=[0.8, 0.8, 0.8], zorder=1)
        ax.plot(post_index, post_control, "-", c=[0.8, 0.8, 0.8], zorder=1)
    _format_panel_dates([ax], pre_index, post_index)
    return fig, [ax]


def plot_panel_counterfactual(
    *,
    pre_index: pd.Index,
    post_index: pd.Index,
    pre_pred: xr.DataArray,
    post_pred: xr.DataArray,
    pre_impact: xr.DataArray,
    post_impact: xr.DataArray,
    post_impact_cumulative: xr.DataArray,
    pre_treated: xr.DataArray,
    post_treated: xr.DataArray,
    pre_control: xr.DataArray,
    post_control: xr.DataArray,
    treatment_time: int | float | pd.Timestamp,
    title: str,
    style: _PosteriorPlotStyle,
    figsize: tuple[float, float],
    plot_predictors: bool,
) -> tuple[plt.Figure, Any]:
    """Render the three-panel counterfactual, impact, and cumulative figure.

    Parameters
    ----------
    pre_index : pandas.Index
        Pre-period time index.
    post_index : pandas.Index
        Post-period time index.
    pre_pred : xarray.DataArray
        Counterfactual for one treated unit, before intervention.
    post_pred : xarray.DataArray
        Counterfactual for one treated unit, from intervention onward.
    pre_impact : xarray.DataArray
        Pre-period impact for the plotted unit.
    post_impact : xarray.DataArray
        Post-period impact for the plotted unit.
    post_impact_cumulative : xarray.DataArray
        Cumulative post-period impact for the plotted unit.
    pre_treated : xarray.DataArray
        Observed pre-period outcome for the plotted unit.
    post_treated : xarray.DataArray
        Observed post-period outcome for the plotted unit.
    pre_control : xarray.DataArray
        Pre-period donor trajectories.
    post_control : xarray.DataArray
        Post-period donor trajectories.
    treatment_time : int, float, or pandas.Timestamp
        Time drawn as the intervention line.
    title : str
        Title of the counterfactual panel.
    style : dict
        Interval style forwarded to the posterior plotting helper.
    figsize : tuple of float
        Figure size in inches.
    plot_predictors : bool
        Whether to overlay donor trajectories.

    Returns
    -------
    tuple
        The figure and its three axes.
    """
    counterfactual_label = "Counterfactual"
    with_uncertainty = has_posterior_draws(pre_pred)
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize)
    handles: list[Any] = []
    labels: list[str] = []
    if with_uncertainty:
        h_line, h_patch = plot_posterior_over_x(
            pre_index,
            pre_pred,
            ax=ax[0],
            **style,
            plot_hdi_kwargs={"color": "C0"},
        )
        handles.append((h_line, h_patch))
        labels.append("Pre-intervention period")

        (h,) = ax[0].plot(pre_index, pre_treated, "k.", label="Observations")
        handles.append(h)
        labels.append("Observations")

        h_line, h_patch = plot_posterior_over_x(
            post_index,
            post_pred,
            ax=ax[0],
            **style,
            plot_hdi_kwargs={"color": "C1"},
        )
        handles.append((h_line, h_patch))
        labels.append(counterfactual_label)
        ax[0].plot(post_index, post_treated, "k.")
    else:
        ax[0].plot(pre_treated["obs_ind"], pre_treated, "k.")
        ax[0].plot(post_treated["obs_ind"], post_treated, "k.")
        ax[0].plot(
            pre_index,
            pre_pred.mean(dim=["chain", "draw"]),
            c="k",
            label="model fit",
        )
        ax[0].plot(
            post_index,
            post_pred.mean(dim=["chain", "draw"]),
            label=counterfactual_label,
            ls=":",
            c="k",
        )

    h = ax[0].fill_between(
        post_index,
        y1=post_pred.mean(dim=["chain", "draw"]).values,
        y2=post_treated.values,
        color="C0",
        alpha=0.25,
        label="Causal impact",
    )
    if with_uncertainty:
        handles.append(h)
        labels.append("Causal impact")

    ax[0].set(title=title)

    if with_uncertainty:
        plot_posterior_over_x(
            pre_index,
            pre_impact,
            ax=ax[1],
            **style,
            plot_hdi_kwargs={"color": "C0"},
        )
        plot_posterior_over_x(
            post_index,
            post_impact,
            ax=ax[1],
            **style,
            plot_hdi_kwargs={"color": "C1"},
        )
    else:
        ax[1].plot(pre_index, pre_impact.mean(dim=["chain", "draw"]), "k.")
        ax[1].plot(
            post_index,
            post_impact.mean(dim=["chain", "draw"]),
            "k.",
            label=counterfactual_label,
        )
    ax[1].axhline(y=0, c="k")
    ax[1].fill_between(
        post_index,
        y1=post_impact.mean(dim=["chain", "draw"]),
        color="C0",
        alpha=0.25,
        label="Causal impact",
    )
    ax[1].set(title="Causal Impact")

    if with_uncertainty:
        plot_posterior_over_x(
            post_index,
            post_impact_cumulative,
            ax=ax[2],
            **style,
            plot_hdi_kwargs={"color": "C1"},
        )
    else:
        ax[2].plot(
            post_index,
            post_impact_cumulative.mean(dim=["chain", "draw"]),
            c="k",
        )
    ax[2].axhline(y=0, c="k")
    ax[2].set(title="Cumulative Causal Impact")

    for i in [0, 1, 2]:
        converted = convert_treatment_time_for_axis(ax[i], treatment_time)
        ax[i].axvline(
            x=converted,
            ls="-",
            lw=3,
            color="r",
            label=None if with_uncertainty else "Treatment time",
        )

    if with_uncertainty:
        ax[0].legend(
            handles=(h_tuple for h_tuple in handles),
            labels=labels,
            fontsize=LEGEND_FONT_SIZE,
        )
    else:
        ax[0].legend(fontsize=LEGEND_FONT_SIZE)

    if plot_predictors:
        ax[0].plot(pre_index, pre_control, "-", c=[0.8, 0.8, 0.8], zorder=1)
        ax[0].plot(post_index, post_control, "-", c=[0.8, 0.8, 0.8], zorder=1)

    _format_panel_dates(ax, pre_index, post_index)
    return fig, ax


def panel_effect_summary(
    bundle: CausalResult,
    post_index: pd.Index,
    *,
    group: Literal["prior", "posterior"],
    window: Literal["post"] | tuple | slice = "post",
    direction: Literal["increase", "decrease", "two-sided"] = "increase",
    alpha: float = 0.05,
    cumulative: bool = True,
    relative: bool = True,
    min_effect: float | None = None,
    treated_unit: str | None = None,
    prefix: str = "Post-period",
    experiment_type: str = "sc",
) -> EffectSummary:
    """Summarize a two-period panel counterfactual over ``window``.

    Parameters
    ----------
    bundle : CausalResult
        Predictions and impacts for the requested draw group.
    post_index : pandas.Index
        Post-period time index used to resolve ``window``.
    group : {"prior", "posterior"}
        Draw group being summarized.
    window : {"post"}, tuple, or slice, optional
        Post-period window passed to the shared summary extractor.
    direction : {"increase", "decrease", "two-sided"}, optional
        Effect direction used by the probability statement.
    alpha : float, optional
        Tail probability used by the summary.
    cumulative : bool, optional
        Whether the summary uses the cumulative impact.
    relative : bool, optional
        Whether the summary includes a relative effect.
    min_effect : float or None, optional
        Minimum effect used by the probability statement.
    treated_unit : str or None, optional
        Unit to summarize. ``None`` selects the first treated unit.
    prefix : str, optional
        Label prefix for the summary.
    experiment_type : str, optional
        Experiment token forwarded to the shared summary.

    Returns
    -------
    EffectSummary
        The windowed post-period effect summary.
    """
    windowed_impact, window_coords = _extract_window(
        bundle.impact_post,
        post_index,
        window,
        treated_unit=treated_unit,
    )
    counterfactual = _extract_counterfactual(
        bundle.predictions_post, window_coords, treated_unit=treated_unit
    )
    return _effect_summary_timeseries(
        windowed_impact,
        counterfactual,
        window_coords,
        direction=direction,
        alpha=alpha,
        cumulative=cumulative,
        relative=relative,
        min_effect=min_effect,
        prefix=prefix,
        experiment_type=experiment_type,
        group=group,
    )
