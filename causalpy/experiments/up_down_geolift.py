#   Copyright 2026 - 2026 The PyMC Labs Developers
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
"""Arm-aware geo lift analysis built on synthetic control."""

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.base import RegressorMixin

from causalpy.pymc_models import PyMCModel, SoftmaxWeightedSumFitter

from .synthetic_control import SyntheticControl


class UpDownGeoLift(SyntheticControl):
    """Estimate signed up and down geo effects against unchanged controls.

    ``arms`` maps each of ``"up"``, ``"down"``, and ``"control"`` to a
    nonempty sequence of geo columns in a wide revenue panel. All geos share
    one intervention date. Fitting uses the same synthetic-control model as
    :class:`SyntheticControl`; the arm methods aggregate its joint posterior
    draws without fitting separate models or treating down geos as donors.

    Attribution requires adequate pre-period donor fit, no spillovers into
    controls, and no concurrent geo-specific changes correlated with arms.
    Arm labels describe assignment, not the amount of media actually delivered.

    Parameters
    ----------
    data : pd.DataFrame
        Wide revenue panel with a unique, sorted time index and geo columns.
    treatment_time : int, float, or pd.Timestamp
        First intervention period, shared by all up and down geos.
    arms : mapping of str to sequence of str
        Nonempty and disjoint ``up``, ``down``, and ``control`` geo groups.
    model : PyMCModel or RegressorMixin, optional
        Counterfactual model. Defaults to :class:`SoftmaxWeightedSumFitter`.
    min_donor_correlation : float, default 0.0
        Minimum pre-period correlation before warning about a donor.
    auto_scale_sigma : bool, default True
        Whether to scale the stock observation-noise prior by pre-period data.
    """

    supports_ols = False
    supports_bayes = True
    _default_model_class = SoftmaxWeightedSumFitter

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_time: int | float | pd.Timestamp,
        arms: Mapping[str, Sequence[str]],
        model: PyMCModel | RegressorMixin | None = None,
        *,
        min_donor_correlation: float = 0.0,
        auto_scale_sigma: bool = True,
    ) -> None:
        if set(arms) != {"up", "down", "control"}:
            raise ValueError("arms must contain exactly 'up', 'down', and 'control'.")
        normalized = {arm: list(arms[arm]) for arm in ("up", "down", "control")}
        if any(not geos for geos in normalized.values()):
            raise ValueError("Each arm must contain at least one geo.")
        all_geos = sum(normalized.values(), [])
        if len(all_geos) != len(set(all_geos)):
            raise ValueError("Geo assignments overlap or contain duplicates.")
        if not data.index.is_unique or not data.index.is_monotonic_increasing:
            raise ValueError("Revenue panel time index must be unique and sorted.")
        if not data.columns.is_unique:
            raise ValueError("Revenue panel geo columns must be unique.")
        missing = set(all_geos) - set(data.columns)
        if missing:
            raise ValueError(
                f"Assigned geos are missing from revenue: {sorted(missing)}"
            )
        if data[all_geos].isna().any().any():
            raise ValueError("Revenue panel contains missing values in assigned geos.")
        self.outcome_unit = data.attrs.get("unit")
        self.arms = normalized
        self.geo_arm = {geo: arm for arm, geos in normalized.items() for geo in geos}
        super().__init__(
            data=data,
            treatment_time=treatment_time,
            control_units=normalized["control"],
            treated_units=normalized["up"] + normalized["down"],
            model=model,
            min_donor_correlation=min_donor_correlation,
            auto_scale_sigma=auto_scale_sigma,
        )

    @property
    def impact_draws(self) -> xr.DataArray:
        """Signed per-period impact with aligned joint geo posterior draws."""
        return self.result.impact_post.assign_coords(
            arm=("treated_units", [self.geo_arm[geo] for geo in self.treated_units])
        )

    @property
    def arm_impact_draws(self) -> xr.DataArray:
        """Draw-wise mean impact across geos in each intervention arm."""
        impact = self.impact_draws
        return xr.concat(
            [
                impact.sel(treated_units=self.arms[arm])
                .mean(dim="treated_units")
                .drop_vars("arm", errors="ignore")
                for arm in ("up", "down")
            ],
            dim=pd.Index(["up", "down"], name="arm"),
        )

    def _selected_window(
        self,
        start: int | float | pd.Timestamp | None,
        end: int | float | pd.Timestamp | None,
    ) -> pd.Index:
        index = self.datapost.index
        window_start = index[0] if start is None else start
        window_end = index[-1] if end is None else end
        selected = index[(index >= window_start) & (index <= window_end)]
        if (
            len(selected) == 0
            or selected[0] != window_start
            or selected[-1] != window_end
        ):
            raise ValueError(
                "Window boundaries must be observed post-intervention periods."
            )
        return selected

    def aggregate_draws(
        self,
        *,
        start: int | float | pd.Timestamp | None = None,
        end: int | float | pd.Timestamp | None = None,
        aggregate: Literal["mean", "sum"] = "mean",
    ) -> xr.Dataset:
        """Aggregate a common inclusive post-intervention window within draws.

        The ``geo`` variable retains a ``treated_units`` dimension and ``arm_mean``
        retains separate up and down values. ``mean`` yields effect per outcome
        period; ``sum`` yields cumulative effect over the selected periods.

        Parameters
        ----------
        start : int, float, or pd.Timestamp, optional
            First included post-intervention period. Defaults to the first post period.
        end : int, float, or pd.Timestamp, optional
            Last included post-intervention period. Defaults to the last post period.
        aggregate : {"mean", "sum"}, default "mean"
            Draw-wise time aggregation over the inclusive window.

        Returns
        -------
        xr.Dataset
            Joint geo and arm effect draws with window metadata.
        """
        if aggregate not in ("mean", "sum"):
            raise ValueError("aggregate must be 'mean' or 'sum'.")
        selected = self._selected_window(start, end)
        geo = self.result.impact_post.sel(obs_ind=selected)
        arm = self.arm_impact_draws.sel(obs_ind=selected)
        return xr.Dataset(
            {
                "geo": getattr(geo, aggregate)(dim="obs_ind"),
                "arm_mean": getattr(arm, aggregate)(dim="obs_ind"),
            },
            attrs={
                "window_start": str(selected[0]),
                "window_end": str(selected[-1]),
                "n_periods": len(selected),
                "aggregate": aggregate,
            },
        )

    def arm_effect_table(
        self,
        *,
        start: int | float | pd.Timestamp | None = None,
        end: int | float | pd.Timestamp | None = None,
        aggregate: Literal["mean", "sum"] = "mean",
    ) -> pd.DataFrame:
        """Summarize geo and arm effects with 94% equal-tailed intervals.

        Parameters
        ----------
        start : int, float, or pd.Timestamp, optional
            First included post-intervention period.
        end : int, float, or pd.Timestamp, optional
            Last included post-intervention period.
        aggregate : {"mean", "sum"}, default "mean"
            Draw-wise time aggregation over the inclusive window.

        Returns
        -------
        pd.DataFrame
            One row per treated geo and one row per intervention arm, with
            posterior mean, standard deviation, interval, and window metadata.
        """
        draws = self.aggregate_draws(start=start, end=end, aggregate=aggregate)
        rows: list[tuple[str, str | None, str, np.ndarray]] = []
        for geo in self.treated_units:
            samples = draws["geo"].sel(treated_units=geo).to_numpy().ravel()
            rows.append(("geo", geo, self.geo_arm[geo], samples))
        for arm in ("up", "down"):
            samples = draws["arm_mean"].sel(arm=arm).to_numpy().ravel()
            rows.append(("arm", None, arm, samples))
        return pd.DataFrame(
            [
                {
                    "level": level,
                    "geo": geo,
                    "arm": arm,
                    "mean": float(np.mean(samples)),
                    "sd": float(np.std(samples)),
                    "lower_94": float(np.quantile(samples, 0.03)),
                    "upper_94": float(np.quantile(samples, 0.97)),
                    **draws.attrs,
                }
                for level, geo, arm, samples in rows
            ]
        )

    def to_mmm_lift(
        self,
        *,
        spend_baseline: pd.DataFrame,
        spend_realized: pd.DataFrame,
        channel: str,
        spend_unit: str,
        outcome_unit: str,
        start: int | float | pd.Timestamp | None = None,
        end: int | float | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Export per-period geo lift rows for scalar MMM saturation calibration.

        Both spend frames require ``(geo, channel)`` MultiIndex columns. Their
        values must be in ``spend_unit`` per outcome period, and revenue must be
        in ``outcome_unit`` per the same period. ``x`` is mean baseline spend;
        ``delta_x`` is mean realized minus baseline spend; ``delta_y`` and
        ``sigma`` summarize posterior draws of mean signed revenue impact.
        The selected window is inclusive and identical for spend and outcome.
        Baseline spend and the delivered change must each be stable across the
        window because a nonlinear saturation curve evaluated at mean spend
        generally differs from the mean of period-level responses.
        The six scalar likelihood columns are accompanied by arm, window, and
        unit metadata. The joint posterior remains in :attr:`impact_draws`.

        Parameters
        ----------
        spend_baseline : pd.DataFrame
            Business-as-usual spend panel with ``(geo, channel)`` columns and
            ``attrs['unit']`` matching ``spend_unit``.
        spend_realized : pd.DataFrame
            Delivered spend panel on the same period grid and in the same units.
        channel : str
            Tested channel name used in the MMM.
        spend_unit : str
            Spend unit per outcome period, matching both spend frame attributes.
        outcome_unit : str
            Revenue unit per outcome period, matching the fitted revenue attribute.
        start : int, float, or pd.Timestamp, optional
            First included post-intervention period.
        end : int, float, or pd.Timestamp, optional
            Last included post-intervention period.

        Returns
        -------
        pd.DataFrame
            One scalar lift row per treated geo with signed effect and spend
            change, posterior standard deviation, and unit and window metadata.
        """
        if not channel or not spend_unit or not outcome_unit:
            raise ValueError("channel, spend_unit, and outcome_unit are required.")
        if self.outcome_unit != outcome_unit:
            raise ValueError("outcome_unit must match revenue.attrs['unit'].")
        selected = self._selected_window(start, end)
        for label, frame in (
            ("spend_baseline", spend_baseline),
            ("spend_realized", spend_realized),
        ):
            if frame.attrs.get("unit") != spend_unit:
                raise ValueError(f"spend_unit must match {label}.attrs['unit'].")
            if (
                not isinstance(frame.columns, pd.MultiIndex)
                or frame.columns.nlevels != 2
            ):
                raise ValueError(
                    f"{label} must have (geo, channel) MultiIndex columns."
                )
            if not frame.columns.is_unique:
                raise ValueError(f"{label} geo/channel columns must be unique.")
            if not frame.index.is_unique or not selected.isin(frame.index).all():
                raise ValueError(f"{label} does not align with the outcome window.")
            missing = [
                (geo, channel)
                for geo in self.treated_units
                if (geo, channel) not in frame
            ]
            if missing:
                raise ValueError(f"{label} is missing geo/channel values: {missing}")

        geo_draws = self.aggregate_draws(
            start=selected[0], end=selected[-1], aggregate="mean"
        )["geo"]
        rows = []
        for geo in self.treated_units:
            baseline = spend_baseline.loc[selected, (geo, channel)].to_numpy(
                dtype=float
            )
            realized = spend_realized.loc[selected, (geo, channel)].to_numpy(
                dtype=float
            )
            if not np.isfinite(baseline).all() or not np.isfinite(realized).all():
                raise ValueError(
                    f"Spend values are missing or nonfinite for geo '{geo}'."
                )
            if (baseline < 0).any() or (realized < 0).any():
                raise ValueError(f"Spend cannot fall below zero for geo '{geo}'.")
            change = realized - baseline
            if not np.allclose(baseline, baseline[0]) or not np.allclose(
                change, change[0]
            ):
                raise ValueError(
                    f"Baseline spend and delivered change must be stable across "
                    f"the scalar MMM window for geo '{geo}'."
                )
            arm = self.geo_arm[geo]
            if (arm == "up" and (change < 0).any()) or (
                arm == "down" and (change > 0).any()
            ):
                raise ValueError(
                    f"Delivered spend direction conflicts with '{arm}' assignment for '{geo}'."
                )
            if change.mean() == 0:
                raise ValueError(
                    f"Delivered spend change must be nonzero for geo '{geo}' "
                    "in the scalar MMM lift likelihood."
                )
            samples = geo_draws.sel(treated_units=geo).to_numpy().ravel()
            delta_y = float(samples.mean())
            sigma = float(samples.std())
            if delta_y == 0:
                raise ValueError(
                    f"Estimated lift must be nonzero for geo '{geo}' "
                    "in the scalar MMM lift likelihood."
                )
            if change.mean() * delta_y < 0:
                raise ValueError(
                    f"Estimated lift for geo '{geo}' has the opposite sign from its "
                    "delivered spend change, which the scalar MMM lift likelihood "
                    "does not accept. Inspect the estimate before calibration."
                )
            if sigma <= 0:
                raise ValueError(
                    f"Posterior lift uncertainty must be positive for geo '{geo}'."
                )
            rows.append(
                {
                    "channel": channel,
                    "geo": geo,
                    "x": float(baseline.mean()),
                    "delta_x": float(change.mean()),
                    "delta_y": delta_y,
                    "sigma": sigma,
                    "arm": arm,
                    "window_start": selected[0],
                    "window_end": selected[-1],
                    "n_periods": len(selected),
                    "spend_unit": spend_unit,
                    "outcome_unit": outcome_unit,
                }
            )
        return pd.DataFrame(rows)
