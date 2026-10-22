# Copyright 2025 - 2026 The PyMC Labs Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Private wide-panel counterfactual seam.

``WidePanel`` is the call surface. ``SyntheticControl`` holds one. Later
wide-panel counterfactual experiments should hold one too, rather than
subclassing ``SyntheticControl``. This module is not a public API.

Shared: the treatment time is post-period (rows at ``treatment_time`` are
post); impact is observed minus counterfactual, aligned on ``obs_ind``; the
three-panel counterfactual figure and the reduced prior-check figure; the
observed-plus-prediction plot frame, donor columns included; effect-summary
windowing over the post period.

Not shared: donor names as ``coeffs`` (that rename belongs to
``WeightedSumFitter`` and stays inside ``SyntheticControl``);
``ModelAdapter.predict`` on post-period donor rows; the score or R² title;
``isinstance(SyntheticControl)`` checks; synthetic difference-in-differences.

Fit rule: donor outcomes at every time, and treated outcomes strictly before
``treatment_time``, may enter a fit or a tuning choice. Treated outcomes at
or after ``treatment_time`` are impact and display only. Penalty and rank
choice use that same allowed set. There is no ``treated_post`` accessor and
no ``treated(period)`` twin of ``control(period)``. ``plot`` and
``plot_data`` read the stored post-period treated series themselves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
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


def resolve_treated_unit(treated_units: Sequence[str], treated_unit: str | None) -> str:
    """Return ``treated_unit``, or the first name when it is omitted.

    Parameters
    ----------
    treated_units : sequence of str
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
    names = list(treated_units)
    treated_unit = treated_unit if treated_unit is not None else names[0]
    if treated_unit not in names:
        raise ValueError(
            f"treated_unit '{treated_unit}' not found. Available units: {names}"
        )
    return treated_unit


def _obs_ind_text(values: xr.DataArray) -> str:
    return ", ".join(str(value) for value in values.obs_ind.values.tolist())


def _require_aligned_obs_ind(
    treated: xr.DataArray,
    predicted: xr.DataArray,
    period: str,
    *,
    name: str = "predictions",
) -> None:
    """Reject a series whose time index does not match the treated series.

    ``name`` is the right-hand label. ``impacts`` keeps the default
    ``predictions``. Plot paths pass ``impact`` or ``cumulative impact`` when
    that field is the mismatch.
    """
    if treated.obs_ind.equals(predicted.obs_ind):
        return
    raise ValueError(
        f"{period} obs_ind mismatch: "
        f"treated=[{_obs_ind_text(treated)}], "
        f"{name}=[{_obs_ind_text(predicted)}]"
    )


def _require_treated_units_dim(
    predicted: xr.DataArray, units: Sequence[str], period: str
) -> None:
    """Reject a prediction that would broadcast onto every treated unit."""
    if "treated_units" in predicted.dims:
        return
    raise ValueError(
        f"{period} predictions lack a treated_units dimension. "
        f"Available units: {list(units)}"
    )


def _counterfactual_impacts(
    treated_pre: xr.DataArray,
    predictions_pre: xr.DataArray,
    treated_post: xr.DataArray,
    predictions_post: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Subtract the counterfactual from the observed treated outcomes.

    Called only by :meth:`WidePanel.impacts`. Impact is aligned on ``obs_ind``.
    A coordinate mismatch is an error, not a silent reindex. A prediction
    without a ``treated_units`` dimension is an error, not a broadcast onto
    every treated unit. Cumulative impact is the running sum of the post-period
    impact.
    """
    units = treated_pre.coords["treated_units"].values.tolist()
    _require_treated_units_dim(predictions_pre, units, "pre-period")
    _require_treated_units_dim(predictions_post, units, "post-period")
    _require_aligned_obs_ind(treated_pre, predictions_pre, "pre-period")
    _require_aligned_obs_ind(treated_post, predictions_post, "post-period")
    impact_pre = (treated_pre - predictions_pre).transpose(
        ..., "obs_ind", "treated_units"
    )
    impact_post = (treated_post - predictions_post).transpose(
        ..., "obs_ind", "treated_units"
    )
    return impact_pre, impact_post, impact_post.cumsum(dim="obs_ind")


def _convert_treatment_time_for_axis(
    axis: plt.Axes, treatment_time: int | float | pd.Timestamp
) -> int | float | pd.Timestamp:
    """Convert treatment time into the plotting units expected by an axis."""
    try:
        return axis.xaxis.convert_units(treatment_time)
    except (TypeError, ValueError):
        return treatment_time


def _format_panel_dates(
    axes: plt.Axes | list[plt.Axes] | Any,
    pre_index: pd.Index,
    post_index: pd.Index,
) -> None:
    """Format datetime axes without changing the figure's returned axes object.

    ``list(ax)`` on a single ``Axes`` iterates child artists, so a lone axes
    is wrapped instead. An ndarray from ``plt.subplots`` is copied to a list
    for ``format_date_axes`` and is still returned unchanged by the plotter.
    """
    if isinstance(pre_index, pd.DatetimeIndex):
        full_index = _combine_datetime_indices(
            pd.DatetimeIndex(pre_index),
            pd.DatetimeIndex(post_index),
        )
        axis_list = [axes] if isinstance(axes, plt.Axes) else list(axes)
        format_date_axes(axis_list, full_index)


@dataclass(frozen=True)
class WidePanel:
    """Stored wide-panel split for a counterfactual experiment.

    Not a public API. See the module docstring for what this seam shares and
    what it does not. ``control`` and ``treated_pre`` are the fit-facing reads.
    Post-period treated outcomes are read through :meth:`_treated_post` only.
    """

    treatment_time: int | float | pd.Timestamp
    control_units: tuple[str, ...]
    treated_units: tuple[str, ...]
    pre: pd.DataFrame
    post: pd.DataFrame

    @classmethod
    def from_frame(
        cls,
        data: pd.DataFrame,
        treatment_time: int | float | pd.Timestamp,
        control_units: Sequence[str],
        treated_units: Sequence[str],
    ) -> WidePanel:
        """Split ``data`` once and store the two period frames.

        Pre-period rows have ``index < treatment_time``. Post-period rows have
        ``index >= treatment_time``. The treatment time itself is post-period.
        The stored frames are copies, so later reads do not re-split ``data``.

        Parameters
        ----------
        data : pandas.DataFrame
            Wide panel indexed by time.
        treatment_time : int, float, or pandas.Timestamp
            First post-period time. Rows at this time are post-period.
        control_units : sequence of str
            Donor columns.
        treated_units : sequence of str
            Treated columns.

        Returns
        -------
        WidePanel
            The stored split.
        """
        pre = data.loc[data.index < treatment_time].copy()
        post = data.loc[data.index >= treatment_time].copy()
        return cls(
            treatment_time=treatment_time,
            control_units=tuple(control_units),
            treated_units=tuple(treated_units),
            pre=pre,
            post=post,
        )

    def control(self, period: Literal["pre", "post"]) -> xr.DataArray:
        """Return donor outcomes for ``period``.

        Dims are ``("obs_ind", "control_units")``. This is not the
        ``WeightedSumFitter`` ``coeffs`` axis.

        Parameters
        ----------
        period : {"pre", "post"}
            Which stored frame to read.

        Returns
        -------
        xarray.DataArray
            Donor outcomes on ``obs_ind × control_units``.
        """
        frame = self._period_frame(period)
        units = list(self.control_units)
        return xr.DataArray(
            frame[units],
            dims=["obs_ind", "control_units"],
            coords={"obs_ind": frame.index, "control_units": units},
        )

    @property
    def treated_pre(self) -> xr.DataArray:
        """Observed treated outcomes strictly before ``treatment_time``.

        Dims are ``("obs_ind", "treated_units")``. This series may enter a
        unit-level fit. Post-period treated outcomes are not available through
        a matching accessor.
        """
        return self._treated_array(self.pre)

    def impacts(
        self,
        predictions_pre: xr.DataArray,
        predictions_post: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Return pre-period, post-period, and cumulative post-period impact.

        Impact is observed minus counterfactual. The post-period treated series
        is read from the stored frame and is not an argument. Each prediction
        must carry a ``treated_units`` dimension.

        Parameters
        ----------
        predictions_pre : xarray.DataArray
            Counterfactual predictions on the pre-period ``obs_ind``.
        predictions_post : xarray.DataArray
            Counterfactual predictions on the post-period ``obs_ind``.

        Returns
        -------
        tuple of xarray.DataArray
            Pre-period impact, post-period impact, and cumulative post-period
            impact.
        """
        return _counterfactual_impacts(
            self.treated_pre,
            predictions_pre,
            self._treated_post(),
            predictions_post,
        )

    def plot(
        self,
        bundle: CausalResult,
        *,
        group: Literal["prior", "posterior"],
        treated_unit: str | None,
        title: str,
        style: _PosteriorPlotStyle,
        figsize: tuple[float, float],
        plot_predictors: bool,
    ) -> tuple[plt.Figure, list[plt.Axes]] | tuple[plt.Figure, np.ndarray]:
        """Draw the prior-check figure or the three-panel posterior figure.

        The prior path returns a one-element list of axes. The posterior path
        returns the ndarray from ``plt.subplots``. Observed post-period outcomes
        are read from the stored frame. ``title`` is the score text owned by
        the caller; the prior figure ignores it.

        Parameters
        ----------
        bundle : CausalResult
            Predictions and impacts for ``group``.
        group : {"prior", "posterior"}
            ``"prior"`` draws the reduced figure. ``"posterior"`` draws three
            panels.
        treated_unit : str or None
            Unit to draw. ``None`` selects the first treated name.
        title : str
            Title of the posterior counterfactual panel.
        style : dict
            Interval style forwarded to the posterior plotting helper.
        figsize : tuple of float
            Figure size in inches.
        plot_predictors : bool
            Whether to overlay donor trajectories.

        Returns
        -------
        tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]] or tuple[matplotlib.figure.Figure, numpy.ndarray]
            Posterior axes are the ndarray from ``plt.subplots``. Prior axes
            are a one-element list.
        """
        unit = resolve_treated_unit(self.treated_units, treated_unit)
        self._require_aligned_bundle(bundle, unit)
        pre_pred = bundle.predictions_pre.sel(treated_units=unit)
        post_pred = bundle.predictions_post.sel(treated_units=unit)
        pre_treated = self.treated_pre.sel(treated_units=unit)
        post_treated = self._treated_post().sel(treated_units=unit)
        pre_control = self.control("pre")
        post_control = self.control("post")
        if group == "prior":
            return _plot_prior_check(
                pre_index=self.pre.index,
                post_index=self.post.index,
                pre_pred=pre_pred,
                post_pred=post_pred,
                pre_treated=pre_treated,
                post_treated=post_treated,
                pre_control=pre_control,
                post_control=post_control,
                treatment_time=self.treatment_time,
                style=style,
                figsize=figsize,
                plot_predictors=plot_predictors,
            )
        if group != "posterior":
            raise ValueError(f"group must be 'prior' or 'posterior', got {group!r}.")
        return _plot_counterfactual(
            pre_index=self.pre.index,
            post_index=self.post.index,
            pre_pred=pre_pred,
            post_pred=post_pred,
            pre_impact=bundle.impact_pre.sel(treated_units=unit),
            post_impact=bundle.impact_post.sel(treated_units=unit),
            post_impact_cumulative=bundle.impact_post_cumulative.sel(
                treated_units=unit
            ),
            pre_treated=pre_treated,
            post_treated=post_treated,
            pre_control=pre_control,
            post_control=post_control,
            treatment_time=self.treatment_time,
            title=title,
            style=style,
            figsize=figsize,
            plot_predictors=plot_predictors,
        )

    def plot_data(
        self,
        bundle: CausalResult,
        *,
        treated_unit: str | None = None,
        hdi_prob: float = HDI_PROB,
    ) -> pd.DataFrame:
        """Build the observed-plus-prediction frame for one treated unit.

        The frame keeps every stored column, including donors. HDI columns are
        included only when the prediction container carries posterior draws.

        Parameters
        ----------
        bundle : CausalResult
            Predictions and impacts for the requested draw group.
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
        pre_data = self.pre.copy()
        post_data = self.post.copy()
        unit = resolve_treated_unit(self.treated_units, treated_unit)
        self._require_aligned_bundle(bundle, unit)
        pre_pred = bundle.predictions_pre.sel(treated_units=unit)
        post_pred = bundle.predictions_post.sel(treated_units=unit)
        pre_impact = bundle.impact_pre.sel(treated_units=unit)
        post_impact = bundle.impact_post.sel(treated_units=unit)

        pre_data["prediction"] = pre_pred.mean(dim=["chain", "draw"]).values
        post_data["prediction"] = post_pred.mean(dim=["chain", "draw"]).values
        if with_uncertainty:
            pred_lower_col = f"pred_hdi_lower_{hdi_pct}"
            pred_upper_col = f"pred_hdi_upper_{hdi_pct}"
            pre_hdi = get_hdi_to_df(pre_pred, hdi_prob=hdi_prob)
            post_hdi = get_hdi_to_df(post_pred, hdi_prob=hdi_prob)
            pre_data[[pred_lower_col, pred_upper_col]] = pre_hdi.iloc[:, [0, -1]].values
            post_data[[pred_lower_col, pred_upper_col]] = post_hdi.iloc[
                :, [0, -1]
            ].values

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

    def effect_summary(
        self,
        bundle: CausalResult,
        *,
        group: Literal["prior", "posterior"],
        experiment_type: str,
        window: Literal["post"] | tuple | slice = "post",
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        cumulative: bool = True,
        relative: bool = True,
        min_effect: float | None = None,
        treated_unit: str | None = None,
        prefix: str = "Post-period",
    ) -> EffectSummary:
        """Summarize the stored post period over ``window``.

        The bundle must share the stored time index. A mismatch raises the same
        ``ValueError`` as :meth:`plot`.

        Parameters
        ----------
        bundle : CausalResult
            Predictions and impacts for the requested draw group.
        group : {"prior", "posterior"}
            Draw group being summarized.
        experiment_type : str
            Experiment token forwarded to the shared summary. ``"sc"`` selects
            synthetic-control assumptions. Callers that are not synthetic control
            must pass their own token; there is no ``"sc"`` default.
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

        Returns
        -------
        EffectSummary
            The windowed post-period effect summary.
        """
        unit = resolve_treated_unit(self.treated_units, treated_unit)
        self._require_aligned_bundle(bundle, unit)
        windowed_impact, window_coords = _extract_window(
            bundle.impact_post,
            self.post.index,
            window,
            treated_unit=unit,
        )
        counterfactual = _extract_counterfactual(
            bundle.predictions_post, window_coords, treated_unit=unit
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

    def _period_frame(self, period: Literal["pre", "post"]) -> pd.DataFrame:
        if period == "pre":
            return self.pre
        if period == "post":
            return self.post
        raise ValueError(f"period must be 'pre' or 'post', got {period!r}.")

    def _treated_array(self, frame: pd.DataFrame) -> xr.DataArray:
        units = list(self.treated_units)
        return xr.DataArray(
            frame[units],
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": frame.index, "treated_units": units},
        )

    def _treated_post(self) -> xr.DataArray:
        """Observed post-period treated outcomes. Impact and display only."""
        return self._treated_array(self.post)

    def _require_aligned_bundle(self, bundle: CausalResult, unit: str) -> None:
        """Reject a bundle whose time index is not the stored period index."""
        treated_pre = self.treated_pre.sel(treated_units=unit)
        treated_post = self._treated_post().sel(treated_units=unit)
        checks = (
            (treated_pre, bundle.predictions_pre, "pre-period", "predictions"),
            (treated_post, bundle.predictions_post, "post-period", "predictions"),
            (treated_pre, bundle.impact_pre, "pre-period", "impact"),
            (treated_post, bundle.impact_post, "post-period", "impact"),
            (
                treated_post,
                bundle.impact_post_cumulative,
                "post-period",
                "cumulative impact",
            ),
        )
        for treated, array, period, name in checks:
            _require_aligned_obs_ind(
                treated, array.sel(treated_units=unit), period, name=name
            )


def _plot_prior_check(
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

    Called only by :meth:`WidePanel.plot`. Prior-implied bands are typically
    far wider than the data, so the impact panels are dropped.
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
    converted = _convert_treatment_time_for_axis(ax, treatment_time)
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


def _plot_counterfactual(
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
) -> tuple[plt.Figure, np.ndarray]:
    """Render the three-panel counterfactual, impact, and cumulative figure.

    Called only by :meth:`WidePanel.plot`. The returned axes are the ndarray
    from ``plt.subplots``, not a list.
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
        ax[0].plot(pre_index, pre_treated, "k.")
        ax[0].plot(post_index, post_treated, "k.")
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
        converted = _convert_treatment_time_for_axis(ax[i], treatment_time)
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
