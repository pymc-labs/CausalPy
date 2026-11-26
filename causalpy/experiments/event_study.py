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
"""Event study / dynamic difference-in-differences for a common treatment time."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.base import RegressorMixin

from causalpy.constants import HDI_PROB
from causalpy.custom_exceptions import DataException, FormulaException
from causalpy.experiments._results import EventStudyResult
from causalpy.experiments.model_adapter import build_coords
from causalpy.formula_utils import build_formula_matrices
from causalpy.input_data import DataFrameLike, to_pandas
from causalpy.pymc_models import LinearRegression, PyMCModel
from causalpy.reporting import EffectSummary, _effect_summary_event_study
from causalpy.utils import round_num

from .base import BaseExperiment


class EventStudy(BaseExperiment[EventStudyResult]):
    """Estimate dynamic treatment effects when every treated unit is treated together.

    This is the classical event-study regression: unit and time fixed effects,
    plus a dummy for each event time other than a reference period. Event time
    is calendar time minus the unit's treatment time. Coefficients are
    treatment effects relative to the reference event time, not relative to
    zero calendar time.

    The design requires one common treatment time. Staggered adoption is
    rejected; use :class:`~causalpy.experiments.staggered_did.StaggeredDifferenceInDifferences`
    for that design. This estimator does not implement the Callaway and
    Sant'Anna or Sun and Abraham heterogeneous-timing corrections.

    Configuration validates the panel and builds the design. Sampling happens
    in :meth:`fit`. Draw-derived coefficients live on
    :class:`~causalpy.experiments._results.EventStudyResult`, not on the
    experiment.

    Parameters
    ----------
    data : DataFrameLike
        Panel with one row per unit-time observation.
    formula : str
        Patsy formula for the outcome, fixed effects, and covariates, such as
        ``"y ~ 1 + C(unit) + C(time)"``. Event-time dummies are added by the
        experiment; do not put them in the formula.
    unit_col : str
        Column identifying units.
    time_col : str
        Column identifying time periods. Must be numeric enough that
        differences are event times.
    treat_time_col : str
        Column of treatment times. ``NaN`` or ``inf`` marks a never-treated unit.
    event_window : tuple of int, default=(-5, 5)
        Inclusive ``(minimum, maximum)`` event times to estimate. The minimum
        must be strictly less than the maximum.
    reference_event_time : int, default=-1
        Omitted event time, normalized to zero. Must lie inside ``event_window``.
        ``-1`` is the last pre-treatment period.
    model : PyMCModel, RegressorMixin, or None, default None
        Statistical model. ``None`` uses :class:`~causalpy.pymc_models.LinearRegression`.

    Examples
    --------
    >>> import causalpy as cp
    >>> from sklearn.linear_model import LinearRegression
    >>> from causalpy.data.simulate_data import generate_event_study_data
    >>> df = generate_event_study_data(n_units=8, n_time=8, treatment_time=4, seed=1)
    >>> experiment = cp.EventStudy(
    ...     df,
    ...     formula="y ~ 1 + C(unit) + C(time)",
    ...     unit_col="unit",
    ...     time_col="time",
    ...     treat_time_col="treat_time",
    ...     event_window=(-2, 2),
    ...     model=cp.create_causalpy_compatible_class(LinearRegression()),
    ... )
    >>> experiment.is_fitted
    False
    """

    supports_ols = True
    supports_bayes = True
    _default_model_class = LinearRegression

    def __init__(
        self,
        data: DataFrameLike,
        formula: str,
        unit_col: str,
        time_col: str,
        treat_time_col: str,
        event_window: tuple[int, int] = (-5, 5),
        reference_event_time: int = -1,
        model: PyMCModel | RegressorMixin | None = None,
    ) -> None:
        super().__init__(model=model)
        pandas_data = to_pandas(data)
        pandas_data.index.name = "obs_ind"
        self.data = pandas_data
        self.expt_type = "Event Study"
        self.formula = formula
        self.unit_col = unit_col
        self.time_col = time_col
        self.treat_time_col = treat_time_col
        self.event_window = event_window
        self.reference_event_time = reference_event_time
        self.input_validation()
        self._compute_event_time()
        self._build_design()

    def input_validation(self) -> None:
        """Validate the formula, panel keys, window, and common treatment time."""
        if not self.formula or "~" not in self.formula:
            raise FormulaException(
                "Formula must be provided in the form 'outcome ~ predictors'"
            )

        required_cols = [self.unit_col, self.time_col, self.treat_time_col]
        for col in required_cols:
            if col not in self.data.columns:
                raise DataException(f"Required column '{col}' not found in data")

        if self.event_window[0] >= self.event_window[1]:
            raise DataException(
                f"event_window[0] ({self.event_window[0]}) must be less than "
                f"event_window[1] ({self.event_window[1]})"
            )

        if not (
            self.event_window[0] <= self.reference_event_time <= self.event_window[1]
        ):
            raise DataException(
                f"reference_event_time ({self.reference_event_time}) must be within "
                f"event_window {self.event_window}"
            )

        duplicates = self.data.duplicated(subset=[self.unit_col, self.time_col])
        if duplicates.any():
            raise DataException(
                "Data contains duplicate unit-time observations. "
                "Each unit should have at most one observation per time period."
            )

        treated_times = self._treated_times()
        if len(treated_times) > 1:
            raise DataException(
                "All treated units must have the same treatment time. "
                f"Found {len(treated_times)} different treatment times: "
                f"{sorted(treated_times)}. "
                "Staggered adoption designs are not supported here; use "
                "StaggeredDifferenceInDifferences."
            )
        if len(treated_times) == 0:
            raise DataException(
                "No treated units found. Mark never-treated units with NaN or inf "
                f"in '{self.treat_time_col}', and give treated units one finite time."
            )

    def _is_control(self) -> pd.Series:
        """Return a row mask for never-treated units."""
        treat_time = self.data[self.treat_time_col]
        return treat_time.isna() | np.isinf(treat_time.to_numpy(dtype=float))

    def _treated_times(self) -> np.ndarray:
        """Return the unique finite treatment times."""
        treated = self.data.loc[~self._is_control(), self.treat_time_col]
        return pd.unique(treated)

    def _compute_event_time(self) -> None:
        """Store event time and warn once if a unit's time grid has gaps."""
        self.data["_event_time"] = np.nan
        treated_mask = ~self._is_control()
        self.data.loc[treated_mask, "_event_time"] = (
            self.data.loc[treated_mask, self.time_col]
            - self.data.loc[treated_mask, self.treat_time_col]
        )

        for unit in self.data[self.unit_col].unique():
            times = np.sort(
                self.data.loc[self.data[self.unit_col] == unit, self.time_col]
                .unique()
                .astype(float)
            )
            if len(times) > 1 and len(np.unique(np.diff(times))) > 1:
                warnings.warn(
                    "Non-consecutive time periods detected (gaps in time series). "
                    "Event time computation assumes time differences are meaningful. "
                    "Please verify gaps are intentional.",
                    UserWarning,
                    stacklevel=2,
                )
                break

        self.data["_in_event_window"] = (
            self.data["_event_time"] >= self.event_window[0]
        ) & (self.data["_event_time"] <= self.event_window[1])

    def _build_design(self) -> None:
        """Build the formula design and append event-time dummies."""
        original_size = len(self.data)
        y, X = build_formula_matrices(self.formula, self.data, return_type="dataframe")
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.data = self.data.loc[X.index]

        filtered_size = len(self.data)
        if original_size and (original_size - filtered_size) / original_size > 0.5:
            pct_lost = (original_size - filtered_size) / original_size * 100
            warnings.warn(
                f"Patsy filtering removed {pct_lost:.1f}% of observations due to "
                "NaN values. Results may be unreliable with substantial data loss. "
                "Consider imputation or checking data quality.",
                UserWarning,
                stacklevel=2,
            )

        unit_counts = self.data.groupby(self.unit_col, observed=True).size()
        if unit_counts.nunique() > 1:
            warnings.warn(
                "Data contains an unbalanced panel (different units have different "
                "numbers of observations). Event Study methodology assumes balanced "
                "panel data. Results may be unreliable. Please verify this is "
                "intentional.",
                UserWarning,
                stacklevel=2,
            )

        self.outcome_variable_name = y.design_info.column_names[0]
        event_times = list(range(self.event_window[0], self.event_window[1] + 1))
        self.event_times = event_times
        self.event_times_non_ref = [
            k for k in event_times if k != self.reference_event_time
        ]
        self.event_time_labels = [f"event_time_{k}" for k in self.event_times_non_ref]
        collision = sorted(set(self.event_time_labels) & set(X.columns))
        if collision:
            raise DataException(
                "Formula columns collide with event-time dummies: "
                f"{collision}. Remove those terms from the formula; the "
                "experiment adds event-time indicators itself."
            )

        dummies = pd.DataFrame(index=self.data.index)
        for label, k in zip(
            self.event_time_labels, self.event_times_non_ref, strict=True
        ):
            dummies[label] = (
                (self.data["_event_time"] == k) & self.data["_in_event_window"]
            ).astype(float)
        design = pd.concat([X, dummies], axis=1)
        self.labels = list(design.columns)
        n = len(design)
        self.design = self._build_design_dataset(
            design.to_numpy(dtype=float),
            np.asarray(y, dtype=float),
            obs_ind=np.arange(n),
            coeffs=self.labels,
        )

    def _fit_inputs(
        self,
    ) -> tuple[xr.DataArray, xr.DataArray, dict[str, Any]]:
        """Return the design matrices and coordinates for build."""
        X = self.design["X"]
        return X, self.design["y"], build_coords(self.labels, X.shape[0])

    def _coefficient_draws(self, group: Literal["prior", "posterior"]) -> xr.DataArray:
        """Return coefficients with dimensions ``chain``, ``draw``, ``coeffs``."""
        coeffs = self._model_backend.coefficients(group=group)
        if "coeffs" not in coeffs.dims:
            raise ValueError("Model coefficients are missing a 'coeffs' dimension.")
        extra = [dim for dim in coeffs.dims if dim not in {"chain", "draw", "coeffs"}]
        for dim in extra:
            if coeffs.sizes[dim] != 1:
                raise ValueError(
                    f"Cannot reduce coefficient dimension {dim!r} of size "
                    f"{coeffs.sizes[dim]} to a scalar event-time effect."
                )
            coeffs = coeffs.squeeze(dim, drop=True)
        return coeffs.transpose("chain", "draw", "coeffs")

    def _finalize(self, group: Literal["prior", "posterior"]) -> None:
        """Stack event-time coefficient draws, with the reference period at zero."""
        coeffs = self._coefficient_draws(group)
        pieces: list[xr.DataArray] = []
        for k in self.event_times:
            if k == self.reference_event_time:
                piece = xr.zeros_like(coeffs.isel(coeffs=0, drop=True))
            else:
                piece = coeffs.sel(coeffs=f"event_time_{k}", drop=True)
            pieces.append(piece.expand_dims(event_time=[k]))
        effects = xr.concat(pieces, dim="event_time")
        effects = effects.transpose("chain", "draw", "event_time")
        score = None
        if group == "posterior":
            score = self._model_backend.score(self.design["X"], self.design["y"])
        self._assign_bundle(
            group, EventStudyResult(event_time_effects=effects, score=score)
        )

    def get_event_time_summary(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        hdi_prob: float = HDI_PROB,
        round_to: int | None = None,
    ) -> pd.DataFrame:
        """Return event-time means and, when draws exist, interval bounds.

        Parameters
        ----------
        group : {"prior", "posterior"}, default "posterior"
            Draw group to summarize. ``"posterior"`` requires :meth:`fit`.
            ``"prior"`` requires :meth:`sample_prior_predictive`.
        hdi_prob : float, default 0.94
            Interval probability. Ignored when the fit has no posterior draws.
        round_to : int, optional
            Decimal places. ``None`` leaves the raw values.
        """
        from causalpy.utils import has_posterior_draws

        bundle = self._require_bundle(group)
        effects = bundle.event_time_effects
        rows = []
        uncertainty = has_posterior_draws(effects)
        for k in effects.coords["event_time"].values.tolist():
            draw = effects.sel(event_time=k)
            is_reference = int(k) == self.reference_event_time
            row: dict[str, Any] = {
                "event_time": int(k),
                "mean": float(draw.mean()),
                "is_reference": is_reference,
            }
            if uncertainty:
                if is_reference:
                    row["std"] = np.nan
                    row["hdi_lower"] = np.nan
                    row["hdi_upper"] = np.nan
                else:
                    from causalpy._arviz_compat import hdi_bounds

                    lower, upper = hdi_bounds(draw, prob=hdi_prob)
                    row["std"] = float(draw.std())
                    row["hdi_lower"] = lower
                    row["hdi_upper"] = upper
            rows.append(row)
        table = pd.DataFrame(rows)
        if round_to is not None:
            numeric = table.select_dtypes(include="number").columns.difference(
                ["event_time"]
            )
            table[numeric] = table[numeric].map(
                lambda value: value if pd.isna(value) else round_num(value, round_to)
            )
        return table

    def summary(self, round_to: int | None = None) -> None:
        """Print the design and the posterior event-time coefficients.

        Parameters
        ----------
        round_to : int, optional
            Decimal places for printed coefficients. ``None`` uses two.
        """
        bundle = self._require_bundle("posterior")
        print(f"{'=' * 24}Event Study{'=' * 24}")
        print(f"Formula: {self.formula}")
        print(f"Event window: {self.event_window}")
        print(f"Reference event time: {self.reference_event_time}")
        print(f"Units: {self.data[self.unit_col].nunique()}")
        print(f"Observations used: {len(self.data)}")
        if bundle.score is not None:
            print(f"Score: {bundle.score.to_dict()}")
        print(self.get_event_time_summary(round_to=round_to).to_string(index=False))
        self.print_coefficients(round_to)

    def plot(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        hdi_prob: float = HDI_PROB,
        figsize: tuple[float, float] = (8, 5),
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot event-time coefficients.

        Parameters
        ----------
        group : {"prior", "posterior"}, default "posterior"
            Which draw group to plot. Both groups draw the coefficient path.
            ``"prior"`` requires :meth:`sample_prior_predictive`; ``"posterior"``
            requires :meth:`fit`.
        hdi_prob : float, default 0.94
            Interval probability drawn as error bars when the group has
            posterior draws. Ignored for point-estimate fits.
        figsize : tuple of float, default=(8, 5)
            Passed to :func:`matplotlib.pyplot.subplots`.
        show : bool, default True
            Whether to display the figure before returning it.
        legend_kwargs : dict, optional
            Legend styling applied in place. Supported keys are ``loc``,
            ``bbox_to_anchor``, ``fontsize``, ``frameon``, and ``title``.
        """
        return self._render_plot(
            show=show,
            legend_kwargs=legend_kwargs,
            group=group,
            hdi_prob=hdi_prob,
            figsize=figsize,
        )

    def _plot(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        hdi_prob: float = HDI_PROB,
        figsize: tuple[float, float] = (8, 5),
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Draw the event-time coefficient path for one group."""
        from causalpy.utils import has_posterior_draws

        table = self.get_event_time_summary(group=group, hdi_prob=hdi_prob)
        fig, ax = plt.subplots(figsize=figsize)
        estimated = table.loc[~table["is_reference"].astype(bool)]
        reference = table.loc[table["is_reference"].astype(bool)]
        if has_posterior_draws(self._require_bundle(group).event_time_effects):
            yerr = np.vstack(
                [
                    estimated["mean"] - estimated["hdi_lower"],
                    estimated["hdi_upper"] - estimated["mean"],
                ]
            )
            ax.errorbar(
                estimated["event_time"],
                estimated["mean"],
                yerr=yerr,
                fmt="o",
                capsize=3,
                label=f"{hdi_prob:.0%} interval",
            )
            if not reference.empty:
                ax.plot(
                    reference["event_time"],
                    reference["mean"],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="none",
                    markeredgecolor="C0",
                    label="reference (fixed at 0)",
                )
        else:
            ax.plot(
                table["event_time"],
                table["mean"],
                "o",
                label="point estimate",
            )
        ax.axhline(0, color="0.4", linewidth=1)
        ax.axvline(self.reference_event_time, color="0.4", linestyle="--", linewidth=1)
        title = (
            "Prior event-time coefficients"
            if group == "prior"
            else "Event-time treatment effects"
        )
        ax.set(xlabel="Event time", ylabel="Effect relative to reference", title=title)
        ax.legend()
        return fig, ax

    def get_plot_data(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        hdi_prob: float = HDI_PROB,
    ) -> pd.DataFrame:
        """Return the event-time table used by :meth:`plot`.

        Parameters
        ----------
        group : {"prior", "posterior"}, default "posterior"
            Draw group to read.
        hdi_prob : float, default 0.94
            Interval probability when the group has posterior draws.
        """
        return self.get_event_time_summary(group=group, hdi_prob=hdi_prob)

    def effect_summary(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        min_effect: float | None = None,
        include_pretrend_check: bool = True,
    ) -> EffectSummary:
        """Summarize dynamic effects relative to the reference event time.

        Parameters
        ----------
        group : {"prior", "posterior"}, default "posterior"
            Which draw group to summarize. ``"prior"`` requires
            :meth:`sample_prior_predictive` and is a design check, not a
            causal estimate. ``"posterior"`` requires :meth:`fit`.
        direction : {"increase", "decrease", "two-sided"}, default "increase"
            Tail probability reported for each non-reference event time when
            posterior draws exist.
        alpha : float, default 0.05
            Probability outside the reported interval (``1 - alpha``).
        min_effect : float, optional
            Half-width of the region of practical equivalence. Ignored for
            point-estimate fits.
        include_pretrend_check : bool, default True
            Whether the prose reports pre-treatment intervals that exclude zero.
            Point-estimate fits have no interval check either way.
        """
        bundle = self._require_bundle(group)
        return _effect_summary_event_study(
            bundle,
            event_window=self.event_window,
            reference_event_time=self.reference_event_time,
            direction=direction,
            alpha=alpha,
            min_effect=min_effect,
            include_pretrend_check=include_pretrend_check,
            group=group,
        )
