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
from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from patsy import dmatrix

from causalpy.constants import HDI_PROB
from causalpy.pymc_models import HierarchicalLaunchITS, PyMCModel
from causalpy.reporting import EffectSummary

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

    Wraps :func:`arviz.hdi`, which returns a ``Dataset`` keyed by the
    DataArray's ``name`` with an added ``hdi`` dim. Works for both scalar
    parameters (``chain``, ``draw`` only) and per-unit / per-bin vector
    parameters (an extra dim such as ``unit`` or ``event_bin``), returning
    0-d or 1-d numpy arrays respectively.
    """
    name = samples.name or "x"
    hdi_ds = az.hdi(samples.rename(name), hdi_prob=ci_prob)
    hdi_da = hdi_ds[name]
    low = hdi_da.sel(hdi="lower").values
    high = hdi_da.sel(hdi="higher").values
    return low, high


class HierarchicalInterruptedTimeSeries(BaseExperiment):
    """Hierarchical ITS for multi-unit panels with unit-specific launch times.

    Unlike :class:`~causalpy.experiments.interrupted_time_series.InterruptedTimeSeries`
    (single unit, single treatment time), this experiment accepts a long-format
    panel where every unit has its *own* launch time. Per-unit intercepts,
    covariate slopes and launch "lift" are partially pooled through a
    hierarchical PyMC model, which lets the model borrow strength across units
    and produces a population-level predictive distribution useful for
    forecasting the effect of a *new* unit.

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
        Numeric time index column (e.g. ``week_idx``). Datetime columns are not
        supported directly; convert to an integer index first.
    treatment_time_col : str
        Numeric column with each unit's launch time (same units as ``time_col``).
    effect_type : {"instant", "event_study", "placebo", "saturation"}, default="instant"
        The effect parameterization.
    bin_edges : sequence of float, optional
        Post-launch bin edges (in units of ``time_col``). Required for
        ``effect_type="event_study"`` and ``"placebo"``.
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
    model : HierarchicalLaunchITS, optional
        A custom model instance. If ``None``, a default is constructed.
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

        self._validate_inputs()
        self._prepare_data()
        self.algorithm()

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
        self._coords = coords

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

    def algorithm(self) -> None:
        """Fit model, compute observed/counterfactual predictions and impact."""
        model: HierarchicalLaunchITS = self.model  # type: ignore[assignment]
        model.fit(
            X=self.X, y=self.y, coords=self._coords, aux=self._aux(effect_on=True)
        )
        # Posterior predictive for observed design and counterfactual. `score`
        # is derived from `observed_pred` (rather than calling `model.score()`,
        # which would internally re-run `predict()`) to avoid sampling the
        # posterior predictive distribution a third time.
        self.observed_pred = model.predict(X=self.X, aux=self._aux(effect_on=True))
        self.score = model.score_from_prediction(self.observed_pred, self.y)
        self.counterfactual_pred = model.predict(
            X=self.X, aux=self._aux(effect_on=False)
        )
        self.impact = model.calculate_impact(self.y, self.counterfactual_pred)

    # ---------------------------------------------------------------- output

    def summary(self, round_to: int | None = None) -> None:
        """Print a short summary of the fitted hierarchical model.

        Parameters
        ----------
        round_to : int, optional
            Unused; accepted for interface parity with other experiments.
        """
        print(f"{self.expt_type}")
        print(f"Formula: {self.formula}")
        print(f"Effect type: {self.effect_type}")
        print(f"Units: {self._n_units}")
        if self.effect_type == "instant":
            post = self.model.idata.posterior  # type: ignore[union-attr]
            mu = float(post["mu_lift"].mean())
            sd = float(post["sigma_lift"].mean())
            print(f"E[mu_lift] = {mu:.3g}   E[sigma_lift] = {sd:.3g}")
        elif self.effect_type == "saturation":
            post = self.model.idata.posterior  # type: ignore[union-attr]
            L = float(np.exp(post["mu_logL"]).mean())
            k = float(np.exp(post["mu_logk"]).mean())
            s = float(post["s"].mean())
            print(
                f"E[L] (ceiling lift) = {L:.3g}   "
                f"E[k] (half-saturation time) = {k:.3g}   "
                f"E[s] (Hill exponent) = {s:.3g}"
            )
        else:
            post = self.model.idata.posterior  # type: ignore[union-attr]
            mu_delta = post["mu_delta"].mean(("chain", "draw")).values
            for label, val in zip(self._event_bin_labels or [], mu_delta, strict=False):
                print(f"  {label:<18}  mu_delta = {val:+.3g}")
            if self.effect_type == "placebo":
                print(self._placebo_check_text())

    def _placebo_check_text(self, ci_prob: float = HDI_PROB) -> str:
        """Return a one-line pass/fail summary of pre-launch bins.

        Parameters
        ----------
        ci_prob : float
            HDI probability mass used to test whether each pre-launch bin's
            interval contains zero. Defaults to
            :data:`~causalpy.constants.HDI_PROB`.
        """
        post = self.model.idata.posterior  # type: ignore[union-attr]
        mu_delta = post["mu_delta"]
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
        self, size: int | None = None, random_seed: int | None = None
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
        """
        if self.model.idata is None:
            raise RuntimeError("Model is not fitted")
        post = self.model.idata.posterior
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
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple:
        """Plot the fitted hierarchical model's population-level effect.

        Dispatches on ``effect_type``: a forest plot of per-unit lifts plus
        the population posterior for ``"instant"``, the population Hill
        saturation curve for ``"saturation"``, or an event-study plot of
        per-bin population effects for ``"event_study"`` / ``"placebo"``.
        See :meth:`plot_unit` to plot a single unit's observed vs.
        counterfactual trajectory instead.

        Parameters
        ----------
        ci_prob : float
            Probability mass of the HDI band drawn around per-unit lifts,
            the saturation curve, or per-bin event-study effects. Must be
            in ``(0, 1]``. Defaults to :data:`~causalpy.constants.HDI_PROB`.
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
        if not 0 < ci_prob <= 1:
            raise ValueError(f"ci_prob must be in (0, 1], got {ci_prob!r}")
        return self._render_plot(
            show=show, legend_kwargs=legend_kwargs, ci_prob=ci_prob
        )

    def _bayesian_plot(self, *, ci_prob: float = HDI_PROB):
        """Dispatch to the appropriate plot method based on effect type."""
        if self.effect_type == "instant":
            return self._plot_instant(ci_prob=ci_prob)
        if self.effect_type == "saturation":
            return self._plot_saturation(ci_prob=ci_prob)
        return self._plot_event_study(ci_prob=ci_prob)

    def _plot_instant(self, *, ci_prob: float = HDI_PROB):
        """Forest plot of per-unit lifts and population posterior of mu_lift."""
        post = self.model.idata.posterior  # type: ignore[union-attr]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Forest of per-unit lifts
        lift = post["lift"]
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
        axes[0].set_title("Posterior lift by unit")

        # Population posterior of mu_lift
        az.plot_posterior(
            self.model.idata, var_names=["mu_lift"], hdi_prob=ci_prob, ax=axes[1]
        )
        axes[1].set_title(r"Population mean $\mu_{lift}$")
        return fig, axes

    def _plot_saturation(self, *, ci_prob: float = HDI_PROB):
        """Plot the population-level Hill saturation curve with an HDI band."""
        post = self.model.idata.posterior  # type: ignore[union-attr]
        max_tau = float(np.nanmax(self._tau_since)) if len(self._tau_since) else 1.0
        grid = xr.DataArray(np.linspace(0, max(max_tau, 1.0), 100), dims="tau_since")

        L = np.exp(post["mu_logL"])
        k = np.exp(post["mu_logk"])
        s = post["s"]
        hill_curve = grid**s / (k**s + grid**s + 1e-6)
        effect_draws = L * hill_curve
        mean = effect_draws.mean(("chain", "draw")).values
        lo, hi = _hdi_bounds(effect_draws.rename("effect"), ci_prob)

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

    def _plot_event_study(self, *, ci_prob: float = HDI_PROB):
        """Event-study plot of population bin effects with HDI error bars."""
        post = self.model.idata.posterior  # type: ignore[union-attr]
        mu_delta = post["mu_delta"]
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

    def plot_unit(self, unit_id: Any = 0, *, ci_prob: float = HDI_PROB):
        """Plot observed vs counterfactual and causal impact for a single unit.

        Parameters
        ----------
        unit_id : Any
            The unit identifier (as it appears in the ``unit_col`` column of the
            input data) to plot. Matched by equality, so any dtype present in
            ``unit_col`` works.
        ci_prob : float
            Probability mass of the HDI band drawn around the posterior
            causal impact. Must be in ``(0, 1]``. Defaults to
            :data:`~causalpy.constants.HDI_PROB`.

        Returns
        -------
        fig, (ax1, ax2)
            Matplotlib figure and axes. Top panel shows observed data, fitted
            mean (with effect) and counterfactual mean (without effect). Bottom
            panel shows the posterior causal impact with HDI.
        """
        if not 0 < ci_prob <= 1:
            raise ValueError(f"ci_prob must be in (0, 1], got {ci_prob!r}")
        df = self.data
        mask = df[self.unit_col] == unit_id
        if not mask.any():
            raise ValueError(
                f"unit_id={unit_id!r} not found in column {self.unit_col!r}"
            )
        t = df.loc[mask, self.time_col].values
        y_obs = df.loc[mask, self.outcome_variable_name].values
        launch = df.loc[mask, self.treatment_time_col].iloc[0]

        obs_mu = (
            self.observed_pred.posterior_predictive["mu"]
            .mean(("chain", "draw"))
            .values.flatten()[mask]
        )
        cf_mu = (
            self.counterfactual_pred.posterior_predictive["mu"]
            .mean(("chain", "draw"))
            .values.flatten()[mask]
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        # Top panel: observed data + model fits
        ax1.scatter(t, y_obs, s=8, alpha=0.4, color="black", label="observed")
        ax1.plot(t, obs_mu, color="C0", label="fitted (with effect)")
        ax1.plot(t, cf_mu, color="C1", ls="--", label="counterfactual (no effect)")
        ax1.axvline(launch, color="red", ls=":", label=f"launch ({launch:g})")
        ax1.legend(fontsize=9)
        ax1.set_ylabel(self.outcome_variable_name)
        ax1.set_title(f"Unit {unit_id}: observed vs counterfactual")

        # Bottom panel: causal impact (posterior mean + HDI)
        impact_unit = self.impact.isel(obs_ind=np.where(mask)[0])
        impact_mean = impact_unit.mean(("chain", "draw")).values.flatten()
        impact_lo, impact_hi = _hdi_bounds(impact_unit.rename("impact"), ci_prob)
        impact_lo = impact_lo.flatten()
        impact_hi = impact_hi.flatten()
        ax2.plot(t, impact_mean, color="C2")
        ax2.fill_between(
            t,
            impact_lo,
            impact_hi,
            color="C2",
            alpha=0.2,
            label=f"{int(ci_prob * 100)}% HDI",
        )
        ax2.axhline(0, color="grey", lw=0.5)
        ax2.axvline(launch, color="red", ls=":")
        ax2.set_xlabel(self.time_col)
        ax2.set_ylabel("causal impact")
        ax2.set_title(f"Unit {unit_id}: posterior causal impact")
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
        """
        post = self.model.idata.posterior  # type: ignore[union-attr]
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
            print(f"  L (ceiling lift): {float(np.exp(post['mu_logL']).mean()):.4g}")
            print(
                f"  k (half-saturation time): {float(np.exp(post['mu_logk']).mean()):.4g}"
            )
            print(f"  s (Hill exponent): {float(post['s'].mean()):.4g}")
        else:  # event_study / placebo
            for i, label in enumerate(self._event_bin_labels or []):
                val = float(post["mu_delta"].isel(event_bin=i).mean())
                print(f"  mu_delta[{label}]: {val:.4g}")

    def effect_summary(
        self,
        *,
        window: Literal["post"] | tuple | slice = "post",
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        cumulative: bool = True,
        relative: bool = True,
        min_effect: float | None = None,
        treated_unit: str | None = None,
        period: Literal["intervention", "post", "comparison"] | None = None,
        prefix: str = "Post-period",
        **kwargs: Any,
    ) -> EffectSummary:
        """Return a compact summary of the population-level effect.

        Reports posterior mean and HDI for ``mu_lift`` (instant), each
        ``mu_delta`` bin (event-study / placebo), or ``L``/``k``/``s``
        (saturation).

        Parameters
        ----------
        window : str, tuple, or slice, default="post"
            Not supported; passing a non-default value emits a warning and
            is ignored.
        direction : {"increase", "decrease", "two-sided"}, default="increase"
            Direction for the tail-probability column reported per row:
            ``P(effect > 0)``, ``P(effect < 0)``, or ``P(effect != 0)``.
        alpha : float, default=0.05
            Significance level for HDI intervals; the effective HDI
            probability is ``hdi_prob = 1 - alpha``.
        cumulative : bool, default=True
            Not supported; passing a non-default value emits a warning and
            is ignored.
        relative : bool, default=True
            Not supported; passing a non-default value emits a warning and
            is ignored.
        min_effect : float, optional
            Not supported; passing a non-``None`` value emits a warning and
            is ignored.
        treated_unit : str, optional
            Not supported; passing a non-``None`` value emits a warning and
            is ignored.
        period : {"intervention", "post", "comparison"}, optional
            Not supported; passing a non-``None`` value emits a warning and
            is ignored.
        prefix : str, optional
            Prefix for prose generation. Defaults to ``"Post-period"``.
        **kwargs
            Reserved for forward-compatibility; not consumed by this
            implementation.

        Returns
        -------
        EffectSummary
            Object with .table (DataFrame) and .text (str) attributes.
        """
        _unsupported = {
            "window": (window, "post"),
            "cumulative": (cumulative, True),
            "relative": (relative, True),
            "period": (period, None),
            "min_effect": (min_effect, None),
            "treated_unit": (treated_unit, None),
        }
        for param_name, (val, default) in _unsupported.items():
            if val != default:
                warnings.warn(
                    f"effect_summary() parameter {param_name!r} is not yet supported "
                    "by HierarchicalInterruptedTimeSeries and will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )

        post = self.model.idata.posterior  # type: ignore[union-attr]
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
            rows.append(_row("L (ceiling lift)", np.exp(post["mu_logL"])))
            rows.append(_row("k (half-saturation time)", np.exp(post["mu_logk"])))
            rows.append(_row("s (Hill exponent)", post["s"]))
        else:
            for i, label in enumerate(self._event_bin_labels or []):
                rows.append(
                    _row(f"mu_delta[{label}]", post["mu_delta"].isel(event_bin=i))
                )

        table = pd.DataFrame(rows).set_index("parameter")
        text_lines = [
            f"{prefix}: {self.expt_type}",
            f"Effect type: {self.effect_type}",
            f"Units: {self._n_units}",
        ]
        if self.effect_type == "placebo":
            text_lines.append(self._placebo_check_text())
        text = "\n".join(text_lines)
        return EffectSummary(table=table, text=text)
