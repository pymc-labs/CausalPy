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
Hierarchical interrupted time series for multi-unit panels with unit-specific
launch times. Supports instant level-shift, event-study/placebo, and
saturating (Hill-curve) treatment-effect parameterizations.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from patsy import dmatrix

from causalpy._arviz_compat import hdi
from causalpy.constants import HDI_PROB
from causalpy.pymc_models import HierarchicalLaunchITS, PyMCModel
from causalpy.reporting import (
    EffectSummary,
    _apply_prior_grouping,
    _compute_statistics,
    _generate_table,
)

from ._results import HierarchicalInterruptedTimeSeriesResult
from .base import BaseExperiment


def _fourier_terms(t: np.ndarray, period: float, K: int) -> np.ndarray:
    """Fourier basis of order ``K`` for period ``period``."""
    t = np.asarray(t, dtype=float)
    cols = []
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * t / period))
        cols.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(cols) if cols else np.empty((len(t), 0))


def _assign_bins(tau: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Assign each ``tau`` to a half-open bin ``[edges[k], edges[k+1])``.

    Returns ``-1`` for rows that fall outside every bin.
    """
    tau = np.asarray(tau)
    out = np.full(tau.shape, -1, dtype=np.int64)
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        mask = (tau >= lo) & (tau < hi)
        out[mask] = k
    return out


def _hdi_bounds(samples: xr.DataArray, ci_prob: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(low, high)`` HDI bounds for a chain/draw(/extra-dim) DataArray.

    Works for both scalar parameters (``chain``, ``draw`` only) and per-unit /
    per-bin vector parameters (an extra dim such as ``unit`` or
    ``event_bin``), returning 0-d or 1-d numpy arrays respectively.
    """
    bounds = hdi(samples, prob=ci_prob, dim=["chain", "draw"])
    return bounds.sel(hdi="lower").values, bounds.sel(hdi="higher").values


def _validate_ci_prob(ci_prob: float) -> None:
    """Raise unless ``ci_prob`` is a valid HDI probability in ``(0, 1)``."""
    if not 0 < ci_prob < 1:
        raise ValueError(f"ci_prob must be in (0, 1), got {ci_prob!r}")


class HierarchicalInterruptedTimeSeries(
    BaseExperiment[HierarchicalInterruptedTimeSeriesResult]
):
    """Hierarchical ITS for multi-unit panels with unit-specific launch times.

    Unlike :class:`~causalpy.experiments.interrupted_time_series.InterruptedTimeSeries`
    (single unit, single treatment time), this experiment accepts a long-format
    panel where every unit has its *own* launch time. Per-unit intercepts,
    covariate slopes and launch "lift" are partially pooled through a
    hierarchical PyMC model, which lets the model borrow strength across units
    and produces a population-level predictive distribution useful for
    forecasting the effect of a *new* unit.

    Construction validates and prepares the data only; call :meth:`fit` to
    sample (``cp.HierarchicalInterruptedTimeSeries(...).fit()``), optionally
    after :meth:`sample_prior_predictive` for prior checks. Draw-derived
    predictions live in :attr:`result` / :attr:`prior_result`.

    Four effect parameterizations are available:

    - ``effect_type="instant"`` — a single post-launch lift per unit,
      ``lift[unit] ~ Normal(mu_lift, sigma_lift)``.
    - ``effect_type="event_study"`` — dynamic per-bin effects over post-launch
      event time (pre-launch is the implicit reference).
    - ``effect_type="placebo"`` — the event-study form extended with pre-launch
      "leads" used as placebos to test the no-anticipation assumption.
    - ``effect_type="saturation"`` — the post-launch effect follows a Hill
      (logistic-type) saturation curve in event time, ramping up smoothly and
      asymptoting at a per-unit ceiling lift ``L``, rather than jumping
      instantly to its final size. See :class:`~causalpy.pymc_models.HierarchicalLaunchITS`.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel. Must contain ``unit_col``, ``time_col`` (numeric),
        ``treatment_time_col`` (numeric, same units as ``time_col``) and any
        columns referenced by ``formula``.
    formula : str
        Patsy formula for the covariate design, e.g. ``"sales ~ 0 + emails + price"``.
        An intercept should *not* be included — the hierarchical ``alpha`` term
        plays that role and an intercept column will be dropped with a warning.
    unit_col : str
        Column identifying units (e.g. product id).
    time_col : str
        Numeric **shared calendar** time index (e.g. weeks since a common
        origin), so the same value means the same calendar period for every
        unit. Seasonality, unit trends and calendar effects are all defined on
        this clock; a per-series counter that restarts at 0 for each unit
        would silently misalign them. Datetime columns are not supported
        directly; convert to an integer index first.
    treatment_time_col : str
        Numeric column with each unit's launch time (same units as ``time_col``).
    effect_type : {"instant", "event_study", "placebo", "saturation"}, default="instant"
        The effect parameterization.
    bin_edges : sequence of float, optional
        Post-launch bin edges (in units of ``time_col``). Required for
        ``effect_type="event_study"`` and ``"placebo"``. Every post-launch
        observation must fall in a bin, so start at 0 and make the final edge
        exceed the largest observed event time (a large final edge gives an
        open-ended last bin); otherwise a ``ValueError`` is raised.
    placebo_edges : sequence of float, optional
        Pre-launch (negative) bin edges for ``effect_type="placebo"``. Rows
        with ``tau`` below the smallest edge are the implicit reference.
    seasonality : dict, optional
        Shared Fourier seasonality spec, e.g. ``{"period": 52, "K": 2}``.
        If ``None`` (default), no seasonality term is included.
    ar_residuals : bool, default=False
        If ``True``, add hierarchical AR(1) residuals per unit via
        ``pytensor.scan``. Requires a balanced panel (all units observed at
        the same time steps). The AR coefficient is partially pooled:
        ``rho[unit] ~ tanh(Normal(mu_rho, sigma_rho))``.
    calendar_effects : bool, default=False
        If ``True``, add a shared calendar-time effect: one random walk over
        the distinct values of ``time_col``, added to every unit's baseline
        and kept in the counterfactual. It absorbs market-wide shocks that hit
        all units at the same calendar time, identified from units that are
        still pre-launch at that time. It is projected off the intercept,
        linear trend and (if used) seasonal terms. A ``UserWarning`` is
        emitted when some calendar times have post-launch rows but no
        pre-launch units; see :meth:`launch_overlap`.
    model : HierarchicalLaunchITS, optional
        A custom model instance. If ``None``, a default is constructed.

    Notes
    -----
    **Data preparation.** Put every unit on one shared calendar index that
    spans the whole study window (potentially several years), and include
    pre-launch history for every unit. Launches then occur at different
    calendar times, while the effect itself is measured in event time
    ``tau = time_col - treatment_time_col``, which is what makes launches
    comparable and lets their effects be pooled.

    **Identification.** Each unit is an interrupted time series: the launch
    is a known interruption, and the counterfactual is the unit's baseline
    (intercept, linear trend, covariates, seasonality) carried past it. The
    jump at launch is identified by the design, provided nothing else
    changes at the same moment. How far that identification reaches depends
    on the model:

    - The trend is estimated jointly with the effect on all rows, not on
      pre-launch rows alone. The effect parameterization (constant lift,
      piecewise-constant bins, Hill curve) is what separates post-launch
      dynamics from drift, so an effect whose shape the chosen
      ``effect_type`` cannot represent can leak into the trend.
    - Without ``calendar_effects``, nothing absorbs a shock common to all
      units at one calendar time beyond covariates and seasonality; such a
      shock is attributed to the launch effect of units already launched.
      This is pure ITS pooled in event time.
    - With ``calendar_effects=True``, staggered launches act like a
      not-yet-treated comparison: at each calendar time, pre-launch units
      identify the common shock. This only holds where pre-launch units
      exist; after the last launch the calendar effect and long-run effects
      are separated only by priors (see :meth:`launch_overlap`).
    - Partial pooling (shared ``mu_gamma``, ``mu_beta``, seasonality and
      hyperpriors) is a modelling assumption: each unit's counterfactual is
      partly informed by the others.

    The ``placebo`` effect type tests the no-anticipation part of these
    assumptions.
    """

    supports_ols = False
    supports_bayes = True
    _default_model_class = HierarchicalLaunchITS

    expt_type = "Hierarchical ITS (staggered launch)"

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        unit_col: str,
        time_col: str,
        treatment_time_col: str,
        effect_type: Literal[
            "instant", "event_study", "placebo", "saturation"
        ] = "instant",
        bin_edges: Sequence[float] | None = None,
        placebo_edges: Sequence[float] | None = None,
        seasonality: dict | None = None,
        ar_residuals: bool = False,
        calendar_effects: bool = False,
        model: PyMCModel | None = None,
    ) -> None:
        super().__init__(model=model)
        if not isinstance(self.model, HierarchicalLaunchITS):
            raise TypeError(
                "HierarchicalInterruptedTimeSeries requires a "
                "HierarchicalLaunchITS model instance."
            )

        self.data = data
        self.formula = formula
        self.unit_col = unit_col
        self.time_col = time_col
        self.treatment_time_col = treatment_time_col
        self.effect_type = effect_type
        self.bin_edges = list(bin_edges) if bin_edges is not None else None
        self.placebo_edges = list(placebo_edges) if placebo_edges is not None else None
        self.seasonality = seasonality
        self.ar_residuals = ar_residuals
        self.calendar_effects = calendar_effects

        self._validate_inputs()
        self._prepare_data()

    # ------------------------------------------------------------------ setup

    def _validate_inputs(self) -> None:
        """Check that required columns exist and arguments are consistent."""
        required = {self.unit_col, self.time_col, self.treatment_time_col}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for col in (self.time_col, self.treatment_time_col):
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                raise ValueError(
                    f"Column {col!r} must be numeric (convert datetimes to an "
                    "integer time index first)."
                )

        if self.effect_type not in (
            "instant",
            "event_study",
            "placebo",
            "saturation",
        ):
            raise ValueError(
                "effect_type must be 'instant', 'event_study', 'placebo' or "
                f"'saturation', got {self.effect_type!r}"
            )
        if self.effect_type in ("event_study", "placebo") and not self.bin_edges:
            raise ValueError(
                f"effect_type={self.effect_type!r} requires `bin_edges` "
                "(post-launch bin edges)."
            )
        if self.effect_type == "placebo" and not self.placebo_edges:
            raise ValueError(
                "effect_type='placebo' requires `placebo_edges` (pre-launch leads)."
            )
        if self.bin_edges is not None and list(self.bin_edges) != sorted(
            self.bin_edges
        ):
            raise ValueError("bin_edges must be sorted in ascending order")
        if self.placebo_edges is not None and list(self.placebo_edges) != sorted(
            self.placebo_edges
        ):
            raise ValueError("placebo_edges must be sorted in ascending order")
        if self.bin_edges is not None and len(self.bin_edges) < 2:
            raise ValueError(
                "bin_edges must have at least 2 entries to define a bin, got "
                f"{len(self.bin_edges)}"
            )
        if self.placebo_edges is not None and len(self.placebo_edges) < 2:
            raise ValueError(
                "placebo_edges must have at least 2 entries to define a bin, "
                f"got {len(self.placebo_edges)}"
            )
        if (
            self.effect_type == "placebo"
            and self.bin_edges is not None
            and self.placebo_edges is not None
            and max(self.placebo_edges) > min(self.bin_edges)
        ):
            raise ValueError(
                "placebo_edges must not overlap bin_edges: pre-launch leads "
                f"(max={max(self.placebo_edges):g}) must not extend past the "
                f"first post-launch bin edge (min={min(self.bin_edges):g})."
            )
        if self.seasonality is not None:
            missing_keys = {"period", "K"} - set(self.seasonality)
            if missing_keys:
                raise ValueError(
                    f"seasonality dict is missing required key(s): "
                    f"{sorted(missing_keys)}"
                )
            if not float(self.seasonality["period"]) > 0:
                raise ValueError(
                    "seasonality['period'] must be > 0, got "
                    f"{self.seasonality['period']!r}"
                )
            if not int(self.seasonality["K"]) >= 1:
                raise ValueError(
                    f"seasonality['K'] must be >= 1, got {self.seasonality['K']!r}"
                )

        inconsistent = (
            self.data.groupby(self.unit_col)[self.treatment_time_col].nunique() > 1
        )
        if inconsistent.any():
            bad_units = sorted(inconsistent[inconsistent].index.tolist())
            raise ValueError(
                f"treatment_time_col {self.treatment_time_col!r} is not constant "
                f"within units: {bad_units}. Each unit must have a single launch time."
            )

    def _prepare_data(self) -> None:
        """Build design matrices, unit indices, and effect indicators from data."""
        df = (
            self.data.copy()
            .sort_values([self.unit_col, self.time_col])
            .reset_index(drop=True)
        )

        # outcome
        outcome = self.formula.split("~")[0].strip()
        if outcome not in df.columns:
            raise ValueError(f"Outcome variable {outcome!r} not in data")
        self.outcome_variable_name = outcome

        # Covariate design via patsy (RHS only)
        rhs = self.formula.split("~", 1)[1]
        X_design = dmatrix(rhs, df, return_type="dataframe")
        if "Intercept" in X_design.columns:
            X_design = X_design.drop(columns=["Intercept"])
        self.labels = list(X_design.columns)

        X_values = X_design.to_numpy(dtype=float)
        # Standardize covariates (column-wise z-score) for scale-free priors
        if X_values.shape[1] > 0:
            self._x_mean = X_values.mean(axis=0)
            self._x_std = X_values.std(axis=0)
            self._x_std[self._x_std == 0] = 1.0
            X_values = (X_values - self._x_mean) / self._x_std
        else:
            self._x_mean = np.zeros(0)
            self._x_std = np.ones(0)

        # Unit index
        units = pd.Categorical(df[self.unit_col])
        self._unit_categories = list(units.categories)
        unit_idx = np.asarray(units.codes, dtype=np.int64)
        n_units = len(self._unit_categories)

        # Within-unit time index for AR residuals (rectangular panel required)
        self._within_unit_tidx: np.ndarray | None = None
        self._n_time_steps: int | None = None
        if self.ar_residuals:
            counts = np.bincount(unit_idx)
            if counts.min() != counts.max():
                raise ValueError(
                    "ar_residuals=True requires a balanced panel "
                    "(all units must have the same number of time steps)."
                )
            self._n_time_steps = int(counts[0])
            # Derive within-unit sequential time index from groupby ordering
            self._within_unit_tidx = (
                df.groupby(self.unit_col, sort=False)
                .cumcount()
                .to_numpy(dtype=np.int64)
            )

        # Event time
        tau = (df[self.time_col] - df[self.treatment_time_col]).to_numpy()
        self._tau = tau
        self._tau_since = np.clip(tau, 0, None)

        # Standardised time index for hierarchical time trends
        t_raw = df[self.time_col].to_numpy(dtype=float)
        self._time_mean = float(t_raw.mean())
        self._time_std = float(t_raw.std())
        if self._time_std == 0:
            self._time_std = 1.0
        self._time = (t_raw - self._time_mean) / self._time_std

        # Fourier seasonality
        if self.seasonality is not None:
            period = float(self.seasonality["period"])
            K = int(self.seasonality["K"])
            F = _fourier_terms(df[self.time_col].to_numpy(), period=period, K=K)
            fourier_labels = [
                f"f_{trig}_{k}" for k in range(1, K + 1) for trig in ("sin", "cos")
            ]
        else:
            F = None
            fourier_labels = None

        # Shared calendar-time grid and the projection that keeps the calendar
        # random walk orthogonal to the intercept, linear trend and seasonality
        self._calendar_idx: np.ndarray | None = None
        self._calendar_projection: np.ndarray | None = None
        calendar_grid: np.ndarray | None = None
        if self.calendar_effects:
            calendar_grid = np.unique(t_raw)
            self._calendar_idx = np.searchsorted(calendar_grid, t_raw).astype(np.int64)
            grid_std = (calendar_grid - self._time_mean) / self._time_std
            basis = [np.ones_like(grid_std), grid_std]
            if self.seasonality is not None:
                basis.append(
                    _fourier_terms(
                        calendar_grid,
                        period=float(self.seasonality["period"]),
                        K=int(self.seasonality["K"]),
                    )
                )
            B = np.column_stack(basis)
            U, s, _ = np.linalg.svd(B, full_matrices=False)
            Q = U[:, s > s.max() * 1e-10]
            self._calendar_projection = np.eye(len(calendar_grid)) - Q @ Q.T

        # Effect design
        post = None
        D = None
        event_bin_labels: list[str] | None = None
        if self.effect_type == "instant":
            post = (tau >= 0).astype(float)
        elif self.effect_type == "event_study":
            edges = [float(e) for e in self.bin_edges]  # type: ignore[union-attr]
            bins = _assign_bins(tau, edges)
            K_bins = len(edges) - 1
            D = np.zeros((len(tau), K_bins), dtype=float)
            mask = bins >= 0
            D[mask, bins[mask]] = 1.0
            event_bin_labels = [
                f"[{edges[k]:g},{edges[k + 1]:g})" for k in range(K_bins)
            ]
        elif self.effect_type == "saturation":
            pass  # effect uses tau_since directly; no `post` indicator needed
        else:  # placebo
            pre = [float(e) for e in self.placebo_edges]  # type: ignore[union-attr]
            post_edges = [float(e) for e in self.bin_edges]  # type: ignore[union-attr]
            pre_bins = _assign_bins(tau, pre)
            post_bins = _assign_bins(tau, post_edges)
            K_pre = len(pre) - 1
            K_post = len(post_edges) - 1
            K_total = K_pre + K_post
            D = np.zeros((len(tau), K_total), dtype=float)
            pre_mask = pre_bins >= 0
            D[pre_mask, pre_bins[pre_mask]] = 1.0
            post_mask = post_bins >= 0
            D[post_mask, K_pre + post_bins[post_mask]] = 1.0
            event_bin_labels = [
                f"pre[{pre[k]:g},{pre[k + 1]:g})" for k in range(K_pre)
            ] + [
                f"post[{post_edges[k]:g},{post_edges[k + 1]:g})" for k in range(K_post)
            ]
            self._n_pre_bins = K_pre
            self._n_post_bins = K_post

        if (
            self.effect_type in ("event_study", "placebo")
            and D is not None
            and D.sum() == 0
        ):
            raise ValueError(
                f"No observations fall within any bin for effect_type="
                f"{self.effect_type!r}; check that `bin_edges`"
                f"{' / `placebo_edges`' if self.effect_type == 'placebo' else ''} "
                "cover the observed event-time range (time_col - "
                "treatment_time_col)."
            )

        # Rows outside every bin get a zero effect, i.e. they act as the
        # untreated reference. That is intended for pre-launch rows but would
        # silently anchor the counterfactual on treated periods for any
        # post-launch row not covered by `bin_edges`.
        if self.effect_type in ("event_study", "placebo"):
            post_edges_all = [float(e) for e in self.bin_edges]  # type: ignore[union-attr]
            uncovered = (tau >= 0) & (_assign_bins(tau, post_edges_all) < 0)
            if uncovered.any():
                raise ValueError(
                    f"{int(uncovered.sum())} post-launch observation(s) fall "
                    "outside every `bin_edges` bin (observed post-launch event "
                    f"time spans [{tau[tau >= 0].min():g}, "
                    f"{tau[tau >= 0].max():g}]; bins span "
                    f"[{post_edges_all[0]:g}, {post_edges_all[-1]:g})). "
                    "Uncovered post-launch rows would be treated as untreated "
                    "baseline. Start `bin_edges` at 0 and extend the final "
                    "edge past the largest event time (e.g. a large final "
                    "edge for an open-ended last bin)."
                )

        # Assemble xarray DataArrays
        obs_ind = np.arange(len(df))
        self.X = xr.DataArray(
            X_values,
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": obs_ind, "coeffs": self.labels},
        )
        self.y = xr.DataArray(
            df[outcome].to_numpy(dtype=float).reshape(-1, 1),
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": obs_ind, "treated_units": ["unit_0"]},
        )
        # Row order of every design/prediction array along `obs_ind`; any
        # per-row lookup into the input data must go through this frame.
        self._sorted_data = df
        self._n_units = n_units
        self._unit_idx = unit_idx
        self._F = F
        self._D = D
        self._post = post
        self._event_bin_labels = event_bin_labels
        self._fourier_labels = fourier_labels

        # Coordinates for the model
        coords: dict[str, Any] = {
            "coeffs": self.labels,
            "obs_ind": obs_ind,
            "treated_units": ["unit_0"],
            "unit": [str(c) for c in self._unit_categories],
        }
        if fourier_labels is not None:
            coords["fourier"] = fourier_labels
        if event_bin_labels is not None:
            coords["event_bin"] = event_bin_labels
        if self._n_time_steps is not None:
            coords["time_step"] = np.arange(self._n_time_steps)
        if calendar_grid is not None:
            coords["calendar_time"] = calendar_grid
        self._coords = coords

        if self.calendar_effects:
            self._warn_calendar_overlap()

    def launch_overlap(self) -> pd.DataFrame:
        """Count pre- and post-launch units observed at each calendar time.

        Shows where staggered launches provide a not-yet-launched comparison:
        a calendar effect (``calendar_effects=True``) is identified from
        data only at calendar times with ``n_pre > 0``. Where ``n_pre == 0``
        and ``n_post > 0``, common shocks and launch effects are separated
        by priors and functional form alone.

        Returns
        -------
        pd.DataFrame
            Indexed by the distinct values of ``time_col``, with columns
            ``n_pre`` and ``n_post`` (number of units observed pre-launch /
            post-launch at that time) and ``identified`` (``n_pre > 0``).
        """
        df = self._sorted_data
        is_post = pd.Series(self._tau >= 0, index=df.index)
        counts = (
            df.assign(_post=is_post)
            .groupby([self.time_col, "_post"])[self.unit_col]
            .nunique()
            .unstack("_post", fill_value=0)
            .reindex(columns=[False, True], fill_value=0)
        )
        out = pd.DataFrame(
            {
                "n_pre": counts[False].astype(int),
                "n_post": counts[True].astype(int),
            }
        )
        out.index.name = self.time_col
        out["identified"] = out["n_pre"] > 0
        return out

    def _warn_calendar_overlap(self) -> None:
        """Warn when calendar effects lack a pre-launch comparison somewhere."""
        overlap = self.launch_overlap()
        unsupported = overlap[(overlap["n_pre"] == 0) & (overlap["n_post"] > 0)]
        if unsupported.empty:
            return
        n_rows = int(np.isin(self._sorted_data[self.time_col], unsupported.index).sum())
        warnings.warn(
            f"calendar_effects=True: {len(unsupported)} calendar time(s) "
            f"(from {unsupported.index.min():g} to {unsupported.index.max():g}, "
            f"{n_rows} rows) have post-launch units but no pre-launch units, "
            "so the calendar effect there is not identified by a "
            "not-yet-launched comparison and is separated from launch effects "
            "only by priors. Consider truncating the window; see "
            "launch_overlap().",
            UserWarning,
            stacklevel=4,
        )

    def _aux(self, *, effect_on: bool = True) -> dict[str, Any]:
        """Build the aux dict passed to the model.

        When ``effect_on=False`` the effect-design inputs are zeroed so that
        posterior-predictive sampling yields the counterfactual ``mu``.
        """
        aux: dict[str, Any] = {
            "effect_type": self.effect_type,
            "unit_idx": self._unit_idx,
        }
        if self._time is not None:
            aux["time"] = self._time
        if self._within_unit_tidx is not None:
            aux["within_unit_tidx"] = self._within_unit_tidx
            aux["n_time_steps"] = self._n_time_steps
        if self._calendar_idx is not None:
            aux["calendar_idx"] = self._calendar_idx
            aux["calendar_projection"] = self._calendar_projection
        if self._F is not None:
            aux["F"] = self._F
        if self.effect_type == "instant":
            post = self._post
            if not effect_on and post is not None:
                post = np.zeros_like(post)
            aux["post"] = post
        elif self.effect_type == "saturation":
            tau_since = self._tau_since
            if not effect_on:
                tau_since = np.zeros_like(tau_since)
            aux["tau_since"] = tau_since
        else:
            D = self._D
            if not effect_on and D is not None:
                if self.effect_type == "placebo":
                    # Keep pre-launch lead columns active; zero only post-launch bins
                    D = D.copy()
                    D[:, self._n_pre_bins :] = 0.0
                else:
                    D = np.zeros_like(D)
            aux["D"] = D
        return aux

    # ------------------------------------------------------------------ fit

    @property
    def _launch_model(self) -> HierarchicalLaunchITS:
        """The backend model, typed as the hierarchical launch model."""
        model = self.model
        assert isinstance(model, HierarchicalLaunchITS)
        return model

    def build(self) -> Self:
        """Construct the model graph without sampling anything.

        Registers the auxiliary event-time design (unit index, effect
        indicators, seasonality, AR and calendar inputs) on the model before
        delegating to :meth:`BaseExperiment.build`, since the hierarchical
        model's graph depends on it as well as on ``(X, y, coords)``.

        Returns
        -------
        Self
            The same experiment, for chaining.
        """
        if not self._model_backend.is_built:
            self._launch_model.set_aux(self._aux(effect_on=True))
        return super().build()

    def _fit_inputs(self) -> tuple[xr.DataArray, xr.DataArray, dict[str, Any]]:
        """Return the standardized design, outcome and coordinates for build."""
        return self.X, self.y, self._coords

    def _predict(
        self, *, effect_on: bool, group: Literal["prior", "posterior"]
    ) -> xr.DataArray:
        """Expected outcomes for every row with the launch effect on or off."""
        with self._launch_model.prediction_aux(self._aux(effect_on=effect_on)):
            return self._model_backend.predict(X=self.X, group=group)

    def _finalize(self, group: Literal["prior", "posterior"]) -> None:
        """Compute the group's predictions, counterfactual and impact.

        The posterior bundle's ``score`` is derived from ``observed_pred``
        rather than the adapter's ``score()``, which would re-run the
        posterior predictive. Prior draws are not scored.
        """
        observed_pred = self._predict(effect_on=True, group=group)
        counterfactual_pred = self._predict(effect_on=False, group=group)
        # Impact relies on exact obs_ind alignment with y.
        assert self.y.obs_ind.equals(counterfactual_pred.obs_ind)
        impact = (
            self.y.isel(treated_units=0) - counterfactual_pred.isel(treated_units=0)
        ).transpose(..., "obs_ind")
        score = None
        if group == "posterior":
            score = self._launch_model.score_from_prediction(observed_pred, self.y)
        self._assign_bundle(
            group,
            HierarchicalInterruptedTimeSeriesResult(
                observed_pred=observed_pred,
                counterfactual_pred=counterfactual_pred,
                impact=impact,
                score=score,
            ),
        )

    def _draws(self, group: Literal["prior", "posterior"]) -> xr.Dataset:
        """Parameter draws for ``group``, raising if it has not been sampled."""
        self._resolve_group(group)
        return self._model_backend.require_idata()[group].to_dataset()

    # ---------------------------------------------------------------- output

    def summary(self, round_to: int | None = None) -> None:
        """Print a short summary of the fitted hierarchical model.

        Parameters
        ----------
        round_to : int, optional
            Unused; accepted for interface parity with other experiments.
        """
        post = self._draws("posterior")
        print(f"{self.expt_type}")
        print(f"Formula: {self.formula}")
        print(f"Effect type: {self.effect_type}")
        print(f"Units: {self._n_units}")
        if self.calendar_effects:
            cal_scale = float(post["sigma_calendar"].mean())
            print(f"Calendar effects: on   E[sigma_calendar] = {cal_scale:.3g}")
        if self.effect_type == "instant":
            mu = float(post["mu_lift"].mean())
            sd = float(post["sigma_lift"].mean())
            print(f"E[mu_lift] = {mu:.3g}   E[sigma_lift] = {sd:.3g}")
        elif self.effect_type == "saturation":
            L = float(xr.ufuncs.exp(post["mu_logL"]).mean())
            k = float(xr.ufuncs.exp(post["mu_logk"]).mean())
            s = float(post["s"].mean())
            print(
                f"E[L] (ceiling lift) = {L:.3g}   "
                f"E[k] (half-saturation time) = {k:.3g}   "
                f"E[s] (Hill exponent) = {s:.3g}"
            )
        else:
            mu_delta = post["mu_delta"].mean(("chain", "draw")).values
            for label, val in zip(self._event_bin_labels or [], mu_delta, strict=False):
                print(f"  {label:<18}  mu_delta = {val:+.3g}")
            if self.effect_type == "placebo":
                print(self._placebo_check_text())

    def _placebo_check_text(
        self,
        ci_prob: float = HDI_PROB,
        *,
        group: Literal["prior", "posterior"] = "posterior",
    ) -> str:
        """Return a one-line pass/fail summary of pre-launch bins.

        Parameters
        ----------
        ci_prob : float
            HDI probability mass used to test whether each pre-launch bin's
            interval contains zero. Defaults to
            :data:`~causalpy.constants.HDI_PROB`.
        group : {"prior", "posterior"}, default "posterior"
            Draw group to test.
        """
        mu_delta = self._draws(group)["mu_delta"]
        n_pre = getattr(self, "_n_pre_bins", 0)
        if n_pre == 0:
            return "Placebo check: no pre-launch bins"
        lo, hi = _hdi_bounds(mu_delta, ci_prob)
        pre_lo, pre_hi = lo[:n_pre], hi[:n_pre]
        contains_zero = (pre_lo <= 0) & (pre_hi >= 0)
        status = "PASS" if contains_zero.all() else "FAIL"
        return (
            f"Placebo check: {status} "
            f"({int(contains_zero.sum())}/{n_pre} pre-launch bins contain 0 "
            f"within the {int(ci_prob * 100)}% HDI)"
        )

    def predictive_for_new_unit(
        self,
        size: int | None = None,
        random_seed: int | None = None,
        *,
        group: Literal["prior", "posterior"] = "posterior",
    ) -> np.ndarray:
        """Draw from the population predictive distribution of a new unit's effect.

        For ``effect_type='instant'`` returns samples from
        ``Normal(mu_lift, sigma_lift)``; for event-study / placebo variants
        returns an array shaped ``(draws, n_bins)`` from
        ``Normal(mu_delta, sigma_delta)``; for ``effect_type='saturation'``
        returns an array shaped ``(draws, 3)`` with columns ``[L, k, s]``
        (ceiling lift, half-saturation time, Hill exponent) for a
        hypothetical new unit.

        Parameters
        ----------
        size : int, optional
            Number of draws to return. Defaults to the number of available
            posterior samples.
        random_seed : int, optional
            Seed for the random number generator used to draw from the
            population-level distribution.
        group : {"prior", "posterior"}, default "posterior"
            Draw group of the population hyperparameters. ``"prior"`` gives
            the prior predictive for a new unit's effect.
        """
        post = self._draws(group)
        rng = np.random.default_rng(random_seed)
        if self.effect_type == "instant":
            mu = post["mu_lift"].values.flatten()
            sd = post["sigma_lift"].values.flatten()
            n = size or mu.size
            idx = rng.integers(0, mu.size, size=n)
            return rng.normal(mu[idx], sd[idx])
        if self.effect_type == "saturation":
            mu_logL = post["mu_logL"].values.flatten()
            sigma_logL = post["sigma_logL"].values.flatten()
            mu_logk = post["mu_logk"].values.flatten()
            sigma_logk = post["sigma_logk"].values.flatten()
            s_samples = post["s"].values.flatten()
            n = size or mu_logL.size
            idx = rng.integers(0, mu_logL.size, size=n)
            L = np.exp(rng.normal(mu_logL[idx], sigma_logL[idx]))
            k = np.exp(rng.normal(mu_logk[idx], sigma_logk[idx]))
            return np.column_stack([L, k, s_samples[idx]])
        mu = (
            post["mu_delta"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", ...)
            .values
        )
        sd = (
            post["sigma_delta"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", ...)
            .values
        )
        n = size or mu.shape[0]
        idx = rng.integers(0, mu.shape[0], size=n)
        return rng.normal(mu[idx], sd[idx])

    # ------------------------------------------------------------------ plot

    def plot(
        self,
        *,
        ci_prob: float = HDI_PROB,
        group: Literal["prior", "posterior"] = "posterior",
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple:
        """Plot the hierarchical model's population-level effect.

        Dispatches on ``effect_type``: a forest plot of per-unit lifts plus
        the population distribution of ``mu_lift`` for ``"instant"``, the
        population Hill saturation curve for ``"saturation"``, or an
        event-study plot of per-bin population effects for
        ``"event_study"`` / ``"placebo"``. See :meth:`plot_unit` to plot a
        single unit's observed vs. counterfactual trajectory instead.

        Parameters
        ----------
        ci_prob : float
            Probability mass of the HDI band drawn around per-unit lifts,
            the saturation curve, or per-bin event-study effects. Must be
            in ``(0, 1)``. Defaults to :data:`~causalpy.constants.HDI_PROB`.
        group : {"prior", "posterior"}, default "posterior"
            Draw group to plot. ``"prior"`` (after
            :meth:`sample_prior_predictive`) shows the prior-implied effects.
        show : bool
            Whether to automatically display the plot. Defaults to ``True``.
        legend_kwargs : dict, optional
            Keyword arguments to adjust legend placement and styling.
            Supported keys: ``loc``, ``bbox_to_anchor``, ``fontsize``,
            ``frameon``, ``title`` (``bbox_transform`` is accepted alongside
            ``bbox_to_anchor``).

        Returns
        -------
        fig, ax
            Matplotlib figure and axes (a pair of axes for ``"instant"``,
            a single axes otherwise).
        """
        _validate_ci_prob(ci_prob)
        return self._render_plot(
            show=show, legend_kwargs=legend_kwargs, group=group, ci_prob=ci_prob
        )

    def _plot(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        ci_prob: float = HDI_PROB,
        **kwargs: Any,
    ) -> tuple:
        """Dispatch to the appropriate plot method based on effect type.

        Parameters
        ----------
        group : {"prior", "posterior"}, default "posterior"
            Draw group to plot.
        ci_prob : float
            Probability mass of the HDI band. Defaults to
            :data:`~causalpy.constants.HDI_PROB`.
        **kwargs
            Accepted for compatibility with :meth:`BaseExperiment._plot`;
            not consumed.
        """
        draws = self._draws(group)
        label = "Posterior" if group == "posterior" else "Prior"
        if self.effect_type == "instant":
            return self._plot_instant(draws, ci_prob=ci_prob, label=label)
        if self.effect_type == "saturation":
            return self._plot_saturation(draws, ci_prob=ci_prob)
        return self._plot_event_study(draws, ci_prob=ci_prob)

    def _plot_instant(self, draws: xr.Dataset, *, ci_prob: float, label: str):
        """Forest plot of per-unit lifts and the distribution of mu_lift."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Forest of per-unit lifts
        lift = draws["lift"]
        means = lift.mean(("chain", "draw")).values
        lo, hi = _hdi_bounds(lift, ci_prob)
        y_pos = np.arange(len(means))
        axes[0].errorbar(
            means, y_pos, xerr=np.vstack([means - lo, hi - means]), fmt="o", color="C0"
        )
        axes[0].axvline(0, color="grey", lw=0.8, ls="--")
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels([str(u) for u in self._unit_categories])
        axes[0].set_xlabel("per-unit lift")
        axes[0].set_title(f"{label} lift by unit")

        # Population distribution of mu_lift with its HDI
        mu_lift = draws["mu_lift"]
        mu_lo, mu_hi = _hdi_bounds(mu_lift, ci_prob)
        axes[1].hist(mu_lift.values.ravel(), bins=40, color="C0", alpha=0.6)
        axes[1].axvline(float(mu_lift.mean()), color="C0", label="mean")
        axes[1].axvspan(
            float(mu_lo),
            float(mu_hi),
            color="C0",
            alpha=0.15,
            label=f"{int(ci_prob * 100)}% HDI",
        )
        axes[1].set_xlabel(r"$\mu_{lift}$")
        axes[1].set_title(r"Population mean $\mu_{lift}$")
        axes[1].legend()
        fig.tight_layout()
        return fig, axes

    def _plot_saturation(self, draws: xr.Dataset, *, ci_prob: float):
        """Plot the population-level Hill saturation curve with an HDI band."""
        max_tau = float(np.nanmax(self._tau_since)) if len(self._tau_since) else 1.0
        grid = xr.DataArray(np.linspace(0, max(max_tau, 1.0), 100), dims="tau_since")

        L = xr.ufuncs.exp(draws["mu_logL"])
        k = xr.ufuncs.exp(draws["mu_logk"])
        s = draws["s"]
        hill_curve = grid**s / (k**s + grid**s + 1e-6)
        effect_draws = L * hill_curve
        mean = effect_draws.mean(("chain", "draw")).values
        lo, hi = _hdi_bounds(effect_draws, ci_prob)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(grid.values, mean, color="C0")
        ax.fill_between(
            grid.values,
            lo,
            hi,
            color="C0",
            alpha=0.2,
            label=f"{int(ci_prob * 100)}% HDI",
        )
        ax.axvline(0, color="red", ls="--", label="launch")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("weeks since launch (time_col units)")
        ax.set_ylabel("population effect (ceiling-scaled Hill curve)")
        ax.set_title("Dynamic launch effect (saturation)")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def _plot_event_study(self, draws: xr.Dataset, *, ci_prob: float):
        """Event-study plot of population bin effects with HDI error bars."""
        mu_delta = draws["mu_delta"]
        mean = mu_delta.mean(("chain", "draw")).values
        lo, hi = _hdi_bounds(mu_delta, ci_prob)
        labels = self._event_bin_labels or [str(i) for i in range(len(mean))]
        x = np.arange(len(mean))

        fig, ax = plt.subplots(figsize=(10, 5))
        if self.effect_type == "placebo":
            n_pre = getattr(self, "_n_pre_bins", 0)
            ax.errorbar(
                x[:n_pre],
                mean[:n_pre],
                yerr=np.vstack([mean[:n_pre] - lo[:n_pre], hi[:n_pre] - mean[:n_pre]]),
                fmt="o",
                color="grey",
                label="pre-launch (placebo)",
            )
            ax.errorbar(
                x[n_pre:],
                mean[n_pre:],
                yerr=np.vstack([mean[n_pre:] - lo[n_pre:], hi[n_pre:] - mean[n_pre:]]),
                fmt="o-",
                color="C0",
                label="post-launch",
            )
            ax.axvline(n_pre - 0.5, color="red", ls="--", label="launch")
            ax.legend()
        else:
            ax.errorbar(
                x,
                mean,
                yerr=np.vstack([mean - lo, hi - mean]),
                fmt="o-",
                color="C0",
            )
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(r"$\mu_\delta$ (population effect)")
        ax.set_title(f"Dynamic launch effect ({self.effect_type})")
        fig.tight_layout()
        return fig, ax

    def plot_unit(
        self,
        unit_id: Any = 0,
        *,
        ci_prob: float = HDI_PROB,
        group: Literal["prior", "posterior"] = "posterior",
    ) -> tuple:
        """Plot observed vs counterfactual and causal impact for a single unit.

        Parameters
        ----------
        unit_id : Any
            The unit identifier (as it appears in the ``unit_col`` column of the
            input data) to plot. Matched by equality, so any dtype present in
            ``unit_col`` works.
        ci_prob : float
            Probability mass of the HDI band drawn around the causal impact.
            Must be in ``(0, 1)``. Defaults to
            :data:`~causalpy.constants.HDI_PROB`.
        group : {"prior", "posterior"}, default "posterior"
            Draw group whose predictions are plotted.

        Returns
        -------
        fig, (ax1, ax2)
            Matplotlib figure and axes. Top panel shows observed data, fitted
            mean (with effect) and counterfactual mean (without effect). Bottom
            panel shows the causal impact with HDI.
        """
        _validate_ci_prob(ci_prob)
        bundle = self._require_bundle(group)
        df = self._sorted_data
        mask = (df[self.unit_col] == unit_id).to_numpy()
        if not mask.any():
            raise ValueError(
                f"unit_id={unit_id!r} not found in column {self.unit_col!r}"
            )
        t = df.loc[mask, self.time_col].values
        y_obs = df.loc[mask, self.outcome_variable_name].values
        launch = df.loc[mask, self.treatment_time_col].iloc[0]

        obs_mu = bundle.observed_pred.mean(("chain", "draw")).values.flatten()[mask]
        cf_mu = bundle.counterfactual_pred.mean(("chain", "draw")).values.flatten()[
            mask
        ]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        # Top panel: observed data + model fits
        ax1.scatter(t, y_obs, s=8, alpha=0.4, color="black", label="observed")
        ax1.plot(t, obs_mu, color="C0", label="fitted (with effect)")
        ax1.plot(t, cf_mu, color="C1", ls="--", label="counterfactual (no effect)")
        ax1.axvline(launch, color="red", ls=":", label=f"launch ({launch:g})")
        ax1.legend(fontsize=9)
        ax1.set_ylabel(self.outcome_variable_name)
        ax1.set_title(f"Unit {unit_id}: observed vs counterfactual")

        # Bottom panel: causal impact (mean + HDI)
        impact_unit = bundle.impact.isel(obs_ind=np.flatnonzero(mask))
        impact_mean = impact_unit.mean(("chain", "draw")).values.flatten()
        impact_lo, impact_hi = _hdi_bounds(impact_unit, ci_prob)
        ax2.plot(t, impact_mean, color="C2")
        ax2.fill_between(
            t,
            impact_lo.flatten(),
            impact_hi.flatten(),
            color="C2",
            alpha=0.2,
            label=f"{int(ci_prob * 100)}% HDI",
        )
        ax2.axhline(0, color="grey", lw=0.5)
        ax2.axvline(launch, color="red", ls=":")
        ax2.set_xlabel(self.time_col)
        ax2.set_ylabel("causal impact")
        ax2.set_title(f"Unit {unit_id}: {group} causal impact")
        ax2.legend()
        fig.tight_layout()
        return fig, (ax1, ax2)

    # ------------------------------------------------------------ reporting

    def print_coefficients(self, round_to: int | None = None) -> None:
        """Print population-level coefficient summaries for the hierarchical model.

        ``mu_beta``/``sigma_beta`` are fit on the internally standardized
        (z-scored) covariate design, so an additional "original scale" block
        is printed by dividing back through the per-column standard
        deviation recorded during :meth:`_prepare_data`.

        Parameters
        ----------
        round_to : int, optional
            Unused; accepted for interface parity with other experiments.

        Raises
        ------
        GroupNotSampledException
            If the experiment has not been fitted yet.
        """
        post = self._draws("posterior")
        print("Model coefficients (population level):")
        for name in ("mu_beta", "sigma_beta"):
            if name in post:
                vals = post[name].mean(("chain", "draw")).values
                print(f"  {name}: {vals}")
        if "mu_beta" in post and len(self._x_std):
            orig = post["mu_beta"].mean(("chain", "draw")).values / self._x_std
            print("  mu_beta (original scale, i.e. per unit of raw covariate):")
            for label, val in zip(self.labels, orig, strict=False):
                print(f"    {label}: {val:.4g}")
        if self.effect_type == "instant":
            print(f"  mu_lift: {float(post['mu_lift'].mean()):.4g}")
            print(f"  sigma_lift: {float(post['sigma_lift'].mean()):.4g}")
        elif self.effect_type == "saturation":
            print(
                f"  L (ceiling lift): {float(xr.ufuncs.exp(post['mu_logL']).mean()):.4g}"
            )
            print(
                f"  k (half-saturation time): {float(xr.ufuncs.exp(post['mu_logk']).mean()):.4g}"
            )
            print(f"  s (Hill exponent): {float(post['s'].mean()):.4g}")
        else:  # event_study / placebo
            for i, label in enumerate(self._event_bin_labels or []):
                val = float(post["mu_delta"].isel(event_bin=i).mean())
                print(f"  mu_delta[{label}]: {val:.4g}")

    def effect_summary(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        window: Literal["post"] | tuple | slice = "post",
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        cumulative: bool = True,
        relative: bool = True,
        min_effect: float | None = None,
        treated_unit: str | None = None,
        period: Literal["intervention", "post", "comparison"] | None = None,
        prefix: str = "Post-period",
    ) -> EffectSummary:
        """Summarize the causal effect on the outcome scale.

        The effect is ``observed - counterfactual`` (the counterfactual
        switches the launch effect off), pooled over every unit-period row
        whose event time ``tau = time_col - treatment_time_col`` falls in
        ``window``. The ``average`` row is the mean effect per unit-period
        row; the ``cumulative`` row is the total incremental outcome summed
        over those rows (across all units). Use :meth:`parameter_summary` for
        the population-level model parameters instead.

        Parameters
        ----------
        group : {"prior", "posterior"}, default "posterior"
            Draw group to summarize. ``"prior"`` (after
            :meth:`sample_prior_predictive`) frames the prose as a prior
            plausibility check, not a causal estimate.
        window : "post", tuple, or slice, default="post"
            Event-time window. ``"post"`` selects all post-launch rows
            (``tau >= 0``). A ``(start, end)`` tuple or ``slice(start, end)``
            selects ``start <= tau <= end`` (``None`` leaves a side open).
        direction : {"increase", "decrease", "two-sided"}, default="increase"
            Direction for the tail probability.
        alpha : float, default=0.05
            Significance level for HDI intervals; the effective HDI
            probability is ``hdi_prob = 1 - alpha``.
        cumulative : bool, default=True
            Whether to include the cumulative (total) effect row.
        relative : bool, default=True
            Whether to include effects relative to the counterfactual (%).
        min_effect : float, optional
            Region of practical equivalence threshold; adds a ``p_rope``
            column when given.
        treated_unit : str, optional
            Restrict the summary to one unit, given as it appears in
            ``unit_col`` (compared as strings). Defaults to pooling all units.
        period : {"intervention", "post", "comparison"}, optional
            Not supported; passing a non-``None`` value emits a warning and
            is ignored.
        prefix : str, optional
            Prefix for prose generation. Defaults to ``"Post-period"``.

        Returns
        -------
        EffectSummary
            Object with .table (DataFrame) and .text (str) attributes.
        """
        if period is not None:
            warnings.warn(
                "effect_summary() parameter 'period' is not yet supported "
                "by HierarchicalInterruptedTimeSeries and will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        bundle = self._require_bundle(group)
        df = self._sorted_data
        rows_mask = self._event_time_window_mask(window)
        if treated_unit is not None:
            unit_mask = (df[self.unit_col].astype(str) == str(treated_unit)).to_numpy()
            if not unit_mask.any():
                raise ValueError(
                    f"treated_unit={treated_unit!r} not found in column "
                    f"{self.unit_col!r}"
                )
            rows_mask &= unit_mask
        if not rows_mask.any():
            raise ValueError(
                f"No observations fall in window={window!r}"
                + (f" for treated_unit={treated_unit!r}" if treated_unit else "")
                + "."
            )

        idx = np.flatnonzero(rows_mask)
        impact = bundle.impact.isel(obs_ind=idx)
        counterfactual = bundle.counterfactual_pred.isel(treated_units=0, obs_ind=idx)
        hdi_prob = 1 - alpha
        stats = _compute_statistics(
            impact,
            counterfactual,
            hdi_prob=hdi_prob,
            direction=direction,
            cumulative=cumulative,
            relative=relative,
            min_effect=min_effect,
        )
        table = _generate_table(stats, cumulative=cumulative, relative=relative)

        n_units = int(df.loc[rows_mask, self.unit_col].nunique())
        ci_pct = int(hdi_prob * 100)

        def _describe(key: str, label: str) -> str:
            """Return one prose sentence for the ``avg`` or ``cum`` stats."""
            s = stats[key]
            line = (
                f"{label}: {s['mean']:.2f} ({ci_pct}% HDI "
                f"[{s['hdi_lower']:.2f}, {s['hdi_upper']:.2f}])"
            )
            if "p_gt_0" in s:
                line += f", P(effect > 0) = {s['p_gt_0']:.2f}"
            elif "p_lt_0" in s:
                line += f", P(effect < 0) = {s['p_lt_0']:.2f}"
            else:
                line += f", P(effect) = {s['prob_of_effect']:.2f}"
            if "relative_mean" in s:
                line += f"; {s['relative_mean']:.1f}% relative to the counterfactual"
            return line + "."

        text_lines = [
            f"{prefix}: {self.expt_type}, effect_type={self.effect_type!r}, "
            f"{n_units} unit(s), {len(idx)} unit-period observations "
            f"(event-time window {window!r}).",
            _describe("avg", "Average effect per unit-period"),
        ]
        if cumulative:
            text_lines.append(_describe("cum", "Cumulative effect (all rows)"))
        if self.effect_type == "placebo":
            text_lines.append(self._placebo_check_text(group=group))
        text = _apply_prior_grouping("\n".join(text_lines), group)
        return EffectSummary(table=table, text=text)

    def _event_time_window_mask(
        self, window: Literal["post"] | tuple | slice
    ) -> np.ndarray:
        """Boolean mask over ``obs_ind`` selecting rows whose event time is in ``window``."""
        if isinstance(window, str):
            if window != "post":
                raise ValueError(
                    f"window must be 'post', a (start, end) tuple or a slice, "
                    f"got {window!r}"
                )
            return self._tau >= 0
        if isinstance(window, slice):
            if window.step is not None:
                raise ValueError("window slices must not have a step")
            start, end = window.start, window.stop
        elif isinstance(window, tuple) and len(window) == 2:
            start, end = window
        else:
            raise ValueError(
                f"window must be 'post', a (start, end) tuple or a slice, got {window!r}"
            )
        mask = np.ones(self._tau.shape, dtype=bool)
        if start is not None:
            mask &= self._tau >= start
        if end is not None:
            mask &= self._tau <= end
        return mask

    def parameter_summary(
        self,
        *,
        alpha: float = 0.05,
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        group: Literal["prior", "posterior"] = "posterior",
    ) -> pd.DataFrame:
        """Return summaries of the population-level effect parameters.

        Reports posterior mean and HDI for ``mu_lift``/``sigma_lift``
        (instant), each ``mu_delta`` bin (event-study / placebo), or
        ``L``/``k``/``s`` (saturation).

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for HDI intervals; the effective HDI
            probability is ``hdi_prob = 1 - alpha``.
        direction : {"increase", "decrease", "two-sided"}, default="increase"
            Direction for the tail-probability column reported per row:
            ``P(effect > 0)``, ``P(effect < 0)``, or ``P(effect != 0)``.
        group : {"prior", "posterior"}, default "posterior"
            Draw group to summarize.

        Returns
        -------
        pd.DataFrame
            One row per parameter, indexed by parameter name.
        """
        post = self._draws(group)
        rows = []
        hdi_prob = 1 - alpha

        def _row(name: str, samples: xr.DataArray) -> dict[str, Any]:
            """Build a summary row dict for a single parameter."""
            mean = float(samples.mean())
            lo_arr, hi_arr = _hdi_bounds(samples, hdi_prob)
            lo, hi = float(lo_arr), float(hi_arr)
            if direction == "increase":
                prob_directional = float((samples > 0).mean())
                prob_col = "prob_positive"
            elif direction == "decrease":
                prob_directional = float((samples < 0).mean())
                prob_col = "prob_negative"
            else:  # two-sided
                prob_directional = float((samples != 0).mean())
                prob_col = "prob_nonzero"
            return {
                "parameter": name,
                "mean": mean,
                f"hdi_{int(hdi_prob * 100)}_low": lo,
                f"hdi_{int(hdi_prob * 100)}_high": hi,
                prob_col: prob_directional,
            }

        if self.effect_type == "instant":
            rows.append(_row("mu_lift", post["mu_lift"]))
            rows.append(_row("sigma_lift", post["sigma_lift"]))
        elif self.effect_type == "saturation":
            rows.append(_row("L (ceiling lift)", xr.ufuncs.exp(post["mu_logL"])))
            rows.append(
                _row("k (half-saturation time)", xr.ufuncs.exp(post["mu_logk"]))
            )
            rows.append(_row("s (Hill exponent)", post["s"]))
        else:
            for i, label in enumerate(self._event_bin_labels or []):
                rows.append(
                    _row(f"mu_delta[{label}]", post["mu_delta"].isel(event_bin=i))
                )

        return pd.DataFrame(rows).set_index("parameter")
