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
Staggered Difference in Differences (Imputation-based)

This module implements the imputation-based staggered DiD estimator, following
the approach of Borusyak, Jaravel, and Spiess (2024). It handles settings where
different units receive treatment at different times.
"""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from patsy import dmatrices
from sklearn.base import RegressorMixin

from causalpy.custom_exceptions import DataException, FormulaException
from causalpy.pymc_models import PyMCModel

from .base import BaseExperiment

LEGEND_FONT_SIZE = 12


class StaggeredDifferenceInDifferences(BaseExperiment):
    """A class to analyse data from staggered adoption Difference-in-Differences settings.

    This estimator uses an imputation-based approach: it fits a model on untreated
    observations only (pre-treatment periods for eventually-treated units plus all
    periods for never-treated units), then predicts counterfactual outcomes for all
    observations. Treatment effects are computed as the difference between observed
    and predicted outcomes for treated observations.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas dataframe with panel data (unit x time observations).
    formula : str
        A statistical model formula. Recommended: "y ~ 1 + C(unit) + C(time)"
        for unit and time fixed effects.
    unit_variable_name : str
        Name of the column identifying units.
    time_variable_name : str
        Name of the column identifying time periods.
    treated_variable_name : str, optional
        Name of the column indicating treatment status (0/1). Defaults to "treated".
    treatment_time_variable_name : str, optional
        Name of the column containing unit-level treatment time (G_i).
        If None, treatment time is inferred from the treated_variable_name column.
    never_treated_value : Any, optional
        Value indicating never-treated units in treatment_time column.
        Defaults to np.inf.
    model : PyMCModel or RegressorMixin, optional
        A model for the untreated outcome. Defaults to None.
    event_window : tuple[int, int], optional
        Tuple (min_event_time, max_event_time) to restrict event-time aggregation.
        If None, uses all available event-times.
    reference_event_time : int, optional
        Event-time index associated with plots (reserved for future use).
        Defaults to -1.

    Attributes
    ----------
    data_ : pd.DataFrame
        Augmented data with G (treatment time), event_time, y_hat0 (counterfactual),
        and tau_hat (treatment effect) columns.
    att_group_time_ : pd.DataFrame
        Group-time ATT estimates: ATT(g, t) for each cohort g and calendar time t.
    att_event_time_ : pd.DataFrame
        Event-time ATT estimates: ATT(e) for each event-time e = t - G.

    Example
    -------
    >>> import causalpy as cp
    >>> from causalpy.data.simulate_data import generate_staggered_did_data
    >>> df = generate_staggered_did_data(n_units=30, n_time_periods=15, seed=42)
    >>> result = cp.StaggeredDifferenceInDifferences(
    ...     df,
    ...     formula="y ~ 1 + C(unit) + C(time)",
    ...     unit_variable_name="unit",
    ...     time_variable_name="time",
    ...     treated_variable_name="treated",
    ...     treatment_time_variable_name="treatment_time",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={
    ...             "tune": 100,
    ...             "draws": 200,
    ...             "chains": 2,
    ...             "progressbar": False,
    ...         }
    ...     ),
    ... )  # doctest: +SKIP

    References
    ----------
    Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting Event Study Designs:
    Robust and Efficient Estimation. Review of Economic Studies.
    """

    supports_ols = True
    supports_bayes = True

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        unit_variable_name: str,
        time_variable_name: str,
        treated_variable_name: str = "treated",
        treatment_time_variable_name: str | None = None,
        never_treated_value: Any = np.inf,
        model: PyMCModel | RegressorMixin | None = None,
        event_window: tuple[int, int] | None = None,
        reference_event_time: int = -1,
        **kwargs: dict,
    ) -> None:
        # NOTE: kwargs is accepted for API compatibility with other experiment classes
        # and is intentionally not used inside this constructor.
        super().__init__(model=model)

        # Store parameters
        self.expt_type = "Staggered Difference in Differences"
        self.formula = formula
        self.unit_variable_name = unit_variable_name
        self.time_variable_name = time_variable_name
        self.treated_variable_name = treated_variable_name
        self.treatment_time_variable_name = treatment_time_variable_name
        self.never_treated_value = never_treated_value
        self.event_window = event_window
        self.reference_event_time = reference_event_time

        # Make a copy of data to avoid modifying the original
        data = data.copy()
        data.index.name = "obs_ind"

        # Input validation
        self.data = data
        self.input_validation()

        # Step 1: Compute treatment time G_i for each unit
        self._compute_treatment_times()

        # Step 2: Compute event time for each observation
        self._compute_event_times()

        # Step 3: Identify untreated observations (training set)
        self._identify_untreated_observations()

        # Step 4: Build design matrices
        self._build_design_matrices()

        # Step 5: Fit model on untreated observations
        self._fit_model()

        # Step 6: Predict counterfactuals for all observations
        self._predict_counterfactuals()

        # Step 7: Compute treatment effects
        self._compute_treatment_effects()

        # Step 8: Aggregate to group-time and event-time ATTs
        self._aggregate_effects()

    def input_validation(self) -> None:
        """Validate the input data and parameters."""
        # Check required columns exist
        required_cols = [
            self.unit_variable_name,
            self.time_variable_name,
        ]

        for col in required_cols:
            if col not in self.data.columns:
                raise DataException(f"Required column '{col}' not found in data")

        # Check treated variable exists (either directly or via treatment_time)
        if self.treatment_time_variable_name is not None:
            if self.treatment_time_variable_name not in self.data.columns:
                raise DataException(
                    f"Treatment time column '{self.treatment_time_variable_name}' "
                    "not found in data"
                )
        elif self.treated_variable_name not in self.data.columns:
            raise DataException(
                f"Treated column '{self.treated_variable_name}' not found in data. "
                "Either provide treated_variable_name or treatment_time_variable_name."
            )

        # Validate formula contains outcome variable
        outcome_match = self.formula.split("~")[0].strip()
        if outcome_match not in self.data.columns:
            raise FormulaException(
                f"Outcome variable '{outcome_match}' from formula not found in data"
            )

        # Validate absorbing treatment (once treated, always treated)
        self._validate_absorbing_treatment()

    def _validate_absorbing_treatment(self) -> None:
        """Validate that treatment is absorbing (once treated, always treated)."""
        if self.treated_variable_name not in self.data.columns:
            # Will infer from treatment_time, skip validation here
            return

        for unit in self.data[self.unit_variable_name].unique():
            unit_data = self.data[
                self.data[self.unit_variable_name] == unit
            ].sort_values(self.time_variable_name)
            treated_values = unit_data[self.treated_variable_name].values

            # Find first treated period
            treated_indices = np.where(treated_values == 1)[0]
            if len(treated_indices) == 0:
                continue  # Never treated

            first_treated_idx = treated_indices[0]

            # Check all subsequent periods are also treated
            if not np.all(treated_values[first_treated_idx:] == 1):
                raise DataException(
                    f"Treatment is not absorbing for unit {unit}. "
                    "Once a unit is treated, it must remain treated in all "
                    "subsequent periods."
                )

    def _compute_treatment_times(self) -> None:
        """Compute treatment time G_i for each unit."""
        if self.treatment_time_variable_name is not None:
            # Use provided treatment time column
            # Get unique treatment time per unit
            g_map = (
                self.data.groupby(self.unit_variable_name)[
                    self.treatment_time_variable_name
                ]
                .first()
                .to_dict()
            )
            self.data["G"] = self.data[self.unit_variable_name].map(g_map)
        else:
            # Infer from treated variable: G = min{t : D_it = 1}
            g_map = {}
            for unit in self.data[self.unit_variable_name].unique():
                unit_data = self.data[self.data[self.unit_variable_name] == unit]
                treated_times = unit_data.loc[
                    unit_data[self.treated_variable_name] == 1, self.time_variable_name
                ]
                if len(treated_times) == 0:
                    g_map[unit] = self.never_treated_value
                else:
                    g_map[unit] = treated_times.min()
            self.data["G"] = self.data[self.unit_variable_name].map(g_map)

        # Store unique cohorts (excluding never-treated)
        self.cohorts = sorted(
            [g for g in self.data["G"].unique() if g != self.never_treated_value]
        )

    def _compute_event_times(self) -> None:
        """Compute event time (t - G) for each observation."""
        self.data["event_time"] = self.data[self.time_variable_name] - self.data["G"]
        # Set event_time to NaN for never-treated units
        self.data.loc[self.data["G"] == self.never_treated_value, "event_time"] = np.nan

    def _identify_untreated_observations(self) -> None:
        """Identify untreated observations for the training set."""
        # Untreated if: (t < G) OR (never-treated)
        is_never_treated = self.data["G"] == self.never_treated_value
        is_pre_treatment = self.data[self.time_variable_name] < self.data["G"]
        self.data["_is_untreated"] = is_never_treated | is_pre_treatment

        # Verify we have some training data
        n_untreated = self.data["_is_untreated"].sum()
        if n_untreated == 0:
            raise DataException(
                "No untreated observations found. Cannot fit the model. "
                "Ensure there are never-treated units or pre-treatment periods."
            )

    def _build_design_matrices(self) -> None:
        """Build design matrices using patsy."""
        # Build design matrix for the full data
        y, X = dmatrices(self.formula, self.data)
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.outcome_variable_name = y.design_info.column_names[0]

        # Store full design matrix
        self.X_full = np.asarray(X)
        self.y_full = np.asarray(y)

        # Get untreated subset for training
        untreated_mask = np.asarray(self.data["_is_untreated"].values, dtype=bool)
        self.X_train = self.X_full[untreated_mask]
        self.y_train = self.y_full[untreated_mask]

    def _fit_model(self) -> None:
        """Fit the model on untreated observations only."""
        # Convert to xarray for PyMC models
        n_train = self.X_train.shape[0]

        if isinstance(self.model, PyMCModel):
            X_train_xr = xr.DataArray(
                self.X_train,
                dims=["obs_ind", "coeffs"],
                coords={
                    "obs_ind": np.arange(n_train),
                    "coeffs": self.labels,
                },
            )
            y_train_xr = xr.DataArray(
                self.y_train,
                dims=["obs_ind", "treated_units"],
                coords={"obs_ind": np.arange(n_train), "treated_units": ["unit_0"]},
            )
            COORDS = {
                "coeffs": self.labels,
                "obs_ind": np.arange(n_train),
                "treated_units": ["unit_0"],
            }
            self.model.fit(X=X_train_xr, y=y_train_xr, coords=COORDS)
        elif isinstance(self.model, RegressorMixin):
            if hasattr(self.model, "fit_intercept"):
                self.model.fit_intercept = False
            self.model.fit(X=self.X_train, y=self.y_train)
        else:
            raise ValueError("Model type not recognized")

    def _predict_counterfactuals(self) -> None:
        """Predict counterfactual outcomes for all observations."""
        n_full = self.X_full.shape[0]

        if isinstance(self.model, PyMCModel):
            X_full_xr = xr.DataArray(
                self.X_full,
                dims=["obs_ind", "coeffs"],
                coords={
                    "obs_ind": np.arange(n_full),
                    "coeffs": self.labels,
                },
            )
            self.y_pred = self.model.predict(X=X_full_xr)

            # Extract posterior mean for y_hat0
            y_hat0_mean = (
                self.y_pred["posterior_predictive"]
                .mu.mean(dim=["chain", "draw"])
                .isel(treated_units=0)
                .values
            )
            self.data["y_hat0"] = y_hat0_mean
        elif isinstance(self.model, RegressorMixin):
            self.y_pred = self.model.predict(self.X_full)
            self.data["y_hat0"] = np.squeeze(self.y_pred)
        else:
            raise ValueError("Model type not recognized")

    def _compute_treatment_effects(self) -> None:
        """Compute treatment effects tau_hat = y - y_hat0 for treated observations."""
        self.data["tau_hat"] = np.nan  # Initialize with NaN
        treated_mask = ~self.data["_is_untreated"]
        self.data.loc[treated_mask, "tau_hat"] = (
            self.data.loc[treated_mask, self.outcome_variable_name]
            - self.data.loc[treated_mask, "y_hat0"]
        )

        # Store augmented data
        self.data_ = self.data.copy()

    def _aggregate_effects(self) -> None:
        """Aggregate effects to group-time and event-time ATTs.

        This method aggregates individual treatment effects into:
        1. Group-time ATTs: ATT(g, t) for each cohort g and calendar time t
        2. Event-time ATTs: ATT(e) for each event-time e = t - G

        For event-time ATTs, this includes both:
        - Post-treatment effects (event_time >= 0): actual treatment effects
        - Pre-treatment effects (event_time < 0): placebo/residual checks

        Pre-treatment effects are computed as residuals (y - y_hat0) for
        eventually-treated units before they receive treatment. These serve
        as a placebo check - if the parallel trends assumption holds, they
        should be centered around zero.
        """
        treated_data = self.data[~self.data["_is_untreated"]].copy()

        # Also get pre-treatment data for eventually-treated units (placebo check)
        # These are observations where: G != never_treated_value AND event_time < 0
        is_eventually_treated = self.data["G"] != self.never_treated_value
        is_pre_treatment = self.data["event_time"] < 0
        pretreatment_data = self.data[is_eventually_treated & is_pre_treatment].copy()

        if isinstance(self.model, PyMCModel):
            self._aggregate_effects_bayesian(treated_data, pretreatment_data)
        else:
            self._aggregate_effects_ols(treated_data, pretreatment_data)

    def _aggregate_effects_bayesian(
        self,
        treated_data: pd.DataFrame,
        pretreatment_data: pd.DataFrame,
        hdi_prob: float = 0.94,
    ) -> None:
        """Aggregate effects for Bayesian model with posterior uncertainty.

        Parameters
        ----------
        treated_data : pd.DataFrame
            DataFrame containing only treated observations (event_time >= 0)
        pretreatment_data : pd.DataFrame
            DataFrame containing pre-treatment observations from eventually-treated
            units (event_time < 0) for placebo check
        hdi_prob : float, optional
            Probability mass for the HDI interval bounds, by default 0.94
        """
        # Store the HDI probability used for interval computation
        self.hdi_prob_ = hdi_prob
        lower_pct = (1 - hdi_prob) / 2 * 100
        upper_pct = (1 + hdi_prob) / 2 * 100

        # Get posterior draws for mu
        mu_draws = self.y_pred["posterior_predictive"].mu.isel(treated_units=0)

        # Get observed y for all observations
        y_observed = np.asarray(self.data[self.outcome_variable_name].values)

        # Compute tau draws for all observations
        # tau_draws has shape (chain, draw, obs_ind)
        tau_draws_all = y_observed - mu_draws.values

        # Get treated observation indices for group-time ATTs
        _is_untreated = np.asarray(self.data["_is_untreated"].values, dtype=bool)
        treated_mask = ~_is_untreated
        treated_indices = np.where(treated_mask)[0]
        tau_draws_treated = tau_draws_all[:, :, treated_indices]
        event_time_treated = np.asarray(treated_data["event_time"].values)

        # --- Group-time ATTs (post-treatment only) ---
        gt_groups = treated_data.groupby(["G", self.time_variable_name]).groups
        att_gt_rows: list[dict] = []
        for key, idx in gt_groups.items():
            g_val = key[0]  # type: ignore[index]
            t_val = key[1]  # type: ignore[index]
            # Find positions in treated_indices
            positions = [np.where(treated_indices == i)[0][0] for i in idx]
            tau_gt = tau_draws_treated[:, :, positions].mean(axis=2)
            att_gt_rows.append(
                {
                    "cohort": g_val,
                    "time": t_val,
                    "att": float(tau_gt.mean()),
                    "att_lower": float(np.percentile(tau_gt, lower_pct)),
                    "att_upper": float(np.percentile(tau_gt, upper_pct)),
                }
            )
        self.att_group_time_ = pd.DataFrame(att_gt_rows)

        # --- Event-time ATTs (including pre-treatment placebo) ---
        att_et_rows: list[dict] = []

        # Pre-treatment placebo effects (event_time < 0)
        if len(pretreatment_data) > 0:
            pretreat_indices = pretreatment_data.index.values
            pretreat_idx_positions = np.array(
                [np.where(self.data.index == idx)[0][0] for idx in pretreat_indices]
            )
            tau_draws_pretreat = tau_draws_all[:, :, pretreat_idx_positions]
            event_time_pretreat = np.asarray(pretreatment_data["event_time"].values)

            event_times_pre = np.unique(
                event_time_pretreat[~np.isnan(event_time_pretreat)]
            )
            # Apply event window filter if specified
            if self.event_window is not None:
                event_times_pre = event_times_pre[
                    (event_times_pre >= self.event_window[0])
                    & (event_times_pre <= self.event_window[1])
                ]

            for e in sorted(event_times_pre):
                e_mask = event_time_pretreat == e
                if e_mask.sum() == 0:
                    continue
                positions_arr = np.where(e_mask)[0]
                tau_e = tau_draws_pretreat[:, :, positions_arr].mean(axis=2)
                att_et_rows.append(
                    {
                        "event_time": int(e),
                        "att": float(tau_e.mean()),
                        "att_lower": float(np.percentile(tau_e, lower_pct)),
                        "att_upper": float(np.percentile(tau_e, upper_pct)),
                        "n_obs": int(e_mask.sum()),
                    }
                )

        # Post-treatment effects (event_time >= 0)
        event_times_post = np.unique(event_time_treated[~np.isnan(event_time_treated)])
        if self.event_window is not None:
            event_times_post = event_times_post[
                (event_times_post >= self.event_window[0])
                & (event_times_post <= self.event_window[1])
            ]

        for e in sorted(event_times_post):
            e_mask = event_time_treated == e
            if e_mask.sum() == 0:
                continue
            positions_arr = np.where(e_mask)[0]
            tau_e = tau_draws_treated[:, :, positions_arr].mean(axis=2)
            att_et_rows.append(
                {
                    "event_time": int(e),
                    "att": float(tau_e.mean()),
                    "att_lower": float(np.percentile(tau_e, lower_pct)),
                    "att_upper": float(np.percentile(tau_e, upper_pct)),
                    "n_obs": int(e_mask.sum()),
                }
            )

        self.att_event_time_ = pd.DataFrame(att_et_rows)

    def _aggregate_effects_ols(
        self, treated_data: pd.DataFrame, pretreatment_data: pd.DataFrame
    ) -> None:
        """Aggregate effects for OLS model (point estimates only).

        Parameters
        ----------
        treated_data : pd.DataFrame
            DataFrame containing only treated observations (event_time >= 0)
        pretreatment_data : pd.DataFrame
            DataFrame containing pre-treatment observations from eventually-treated
            units (event_time < 0) for placebo check
        """
        # --- Group-time ATTs (post-treatment only) ---
        att_gt = (
            treated_data.groupby(["G", self.time_variable_name])["tau_hat"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        att_gt.columns = ["cohort", "time", "att", "att_std", "n_obs"]
        self.att_group_time_ = att_gt

        # --- Event-time ATTs (including pre-treatment placebo) ---
        # Compute tau_hat for pre-treatment observations (residuals)
        if len(pretreatment_data) > 0:
            pretreatment_data = pretreatment_data.copy()
            pretreatment_data["tau_hat"] = (
                pretreatment_data[self.outcome_variable_name]
                - pretreatment_data["y_hat0"]
            )

        # Combine pre-treatment and post-treatment for event-time aggregation
        event_data = pd.concat([pretreatment_data, treated_data], ignore_index=True)

        # Apply event window filter if specified
        if self.event_window is not None:
            event_data = event_data[
                (event_data["event_time"] >= self.event_window[0])
                & (event_data["event_time"] <= self.event_window[1])
            ]

        att_et = (
            event_data.groupby("event_time")["tau_hat"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        att_et.columns = ["event_time", "att", "att_std", "n_obs"]
        att_et["event_time"] = att_et["event_time"].astype(int)
        self.att_event_time_ = att_et

    def summary(self, round_to: int | None = 2) -> None:
        """Print summary of main results.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals for rounding. Defaults to 2.
        """
        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print(f"Number of units: {self.data[self.unit_variable_name].nunique()}")
        print(f"Number of time periods: {self.data[self.time_variable_name].nunique()}")
        print(f"Treatment cohorts: {self.cohorts}")
        print(
            f"Never-treated units: {(self.data['G'] == self.never_treated_value).sum() // self.data[self.time_variable_name].nunique()}"
        )
        print("\nEvent-time estimates:")
        att_et = self.att_event_time_.copy()
        # Add indicator column for clarity
        att_et["type"] = att_et["event_time"].apply(
            lambda x: "placebo" if x < 0 else "ATT"
        )
        # Reorder columns to put type first
        cols = ["event_time", "type"] + [
            c for c in att_et.columns if c not in ["event_time", "type"]
        ]
        print(att_et[cols].to_string(index=False))
        print("\nModel coefficients:")
        self.print_coefficients(round_to)

    def _bayesian_plot(
        self, round_to: int | None = None, **kwargs: dict
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot event-study results for Bayesian model.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals for rounding in plot titles.

        Returns
        -------
        tuple[plt.Figure, list[plt.Axes]]
            Figure and axes objects.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        att_et = self.att_event_time_.copy()

        # Separate pre-treatment (placebo) and post-treatment (ATT)
        pre_treatment = att_et[att_et["event_time"] < 0]
        post_treatment = att_et[att_et["event_time"] >= 0]

        # Plot pre-treatment placebo estimates (different style)
        if len(pre_treatment) > 0:
            ax.errorbar(
                pre_treatment["event_time"],
                pre_treatment["att"],
                yerr=[
                    pre_treatment["att"] - pre_treatment["att_lower"],
                    pre_treatment["att_upper"] - pre_treatment["att"],
                ],
                fmt="s",  # Square markers for placebo
                capsize=4,
                capthick=2,
                markersize=7,
                color="gray",
                alpha=0.7,
                label="Placebo estimate (94% HDI)",
            )

        # Plot post-treatment ATT estimates
        if len(post_treatment) > 0:
            ax.errorbar(
                post_treatment["event_time"],
                post_treatment["att"],
                yerr=[
                    post_treatment["att"] - post_treatment["att_lower"],
                    post_treatment["att_upper"] - post_treatment["att"],
                ],
                fmt="o",
                capsize=4,
                capthick=2,
                markersize=8,
                color="C0",
                label="ATT estimate (94% HDI)",
            )

        # Add horizontal line at zero
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.7)

        # Add vertical line at event_time = 0 (treatment onset)
        ax.axvline(x=-0.5, color="red", linestyle="-", linewidth=2, alpha=0.7)

        # Shade pre-treatment region
        event_min = att_et["event_time"].min()
        if event_min < 0:
            ax.axvspan(
                event_min - 0.5,
                -0.5,
                alpha=0.1,
                color="gray",
            )

        # Labels and formatting
        ax.set_xlabel("Event Time (periods relative to treatment)", fontsize=12)
        ax.set_ylabel("Effect Estimate", fontsize=12)
        ax.set_title("Staggered DiD Event Study", fontsize=14)
        ax.legend(fontsize=LEGEND_FONT_SIZE)

        # Set integer ticks for event time
        ax.set_xticks(att_et["event_time"].values)

        return fig, [ax]

    def _ols_plot(
        self, round_to: int | None = None, **kwargs: dict
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot event-study results for OLS model.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals for rounding in plot titles.

        Returns
        -------
        tuple[plt.Figure, list[plt.Axes]]
            Figure and axes objects.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        att_et = self.att_event_time_.copy()

        # Separate pre-treatment (placebo) and post-treatment (ATT)
        pre_treatment = att_et[att_et["event_time"] < 0]
        post_treatment = att_et[att_et["event_time"] >= 0]

        # Plot pre-treatment placebo estimates (different style)
        if len(pre_treatment) > 0:
            ax.scatter(
                pre_treatment["event_time"],
                pre_treatment["att"],
                s=60,
                color="gray",
                marker="s",  # Square markers for placebo
                zorder=3,
                alpha=0.7,
                label="Placebo estimate",
            )
            # Add error bars if std available
            if "att_std" in pre_treatment.columns:
                se = pre_treatment["att_std"] / np.sqrt(pre_treatment["n_obs"])
                ax.errorbar(
                    pre_treatment["event_time"],
                    pre_treatment["att"],
                    yerr=1.96 * se,
                    fmt="none",
                    capsize=4,
                    capthick=2,
                    color="gray",
                    alpha=0.5,
                )

        # Plot post-treatment ATT estimates
        if len(post_treatment) > 0:
            ax.scatter(
                post_treatment["event_time"],
                post_treatment["att"],
                s=80,
                color="C0",
                zorder=3,
                label="ATT estimate",
            )
            # Add error bars if std available
            if "att_std" in post_treatment.columns:
                se = post_treatment["att_std"] / np.sqrt(post_treatment["n_obs"])
                ax.errorbar(
                    post_treatment["event_time"],
                    post_treatment["att"],
                    yerr=1.96 * se,
                    fmt="none",
                    capsize=4,
                    capthick=2,
                    color="C0",
                    alpha=0.7,
                )

        # Add horizontal line at zero
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.7)

        # Add vertical line at event_time = 0 (treatment onset)
        ax.axvline(x=-0.5, color="red", linestyle="-", linewidth=2, alpha=0.7)

        # Shade pre-treatment region
        event_min = att_et["event_time"].min()
        if event_min < 0:
            ax.axvspan(
                event_min - 0.5,
                -0.5,
                alpha=0.1,
                color="gray",
            )

        # Labels and formatting
        ax.set_xlabel("Event Time (periods relative to treatment)", fontsize=12)
        ax.set_ylabel("Effect Estimate", fontsize=12)
        ax.set_title("Staggered DiD Event Study", fontsize=14)
        ax.legend(fontsize=LEGEND_FONT_SIZE)

        # Set integer ticks for event time
        ax.set_xticks(att_et["event_time"].values)

        return fig, [ax]

    def get_plot_data_bayesian(self, hdi_prob: float = 0.94) -> pd.DataFrame:
        """Get plotting data for Bayesian model.

        Parameters
        ----------
        hdi_prob : float, optional
            Probability for HDI interval. Defaults to 0.94.

        Returns
        -------
        pd.DataFrame
            DataFrame with event_time, att, att_lower, att_upper columns.
            Includes both pre-treatment (placebo) and post-treatment effects.
        """
        # If the requested hdi_prob matches what was used during aggregation,
        # return the pre-computed results
        stored_hdi_prob = getattr(self, "hdi_prob_", 0.94)
        if np.isclose(hdi_prob, stored_hdi_prob):
            return self.att_event_time_.copy()

        # Recompute intervals with the requested hdi_prob
        lower_pct = (1 - hdi_prob) / 2 * 100
        upper_pct = (1 + hdi_prob) / 2 * 100

        # Get posterior draws for mu
        mu_draws = self.y_pred["posterior_predictive"].mu.isel(treated_units=0)

        # Get observed y for all observations
        y_observed = np.asarray(self.data[self.outcome_variable_name].values)

        # Compute tau draws for all observations
        tau_draws_all = y_observed - mu_draws.values

        att_et_rows: list[dict] = []

        # Pre-treatment placebo effects (eventually-treated units, event_time < 0)
        is_eventually_treated = self.data["G"] != self.never_treated_value
        is_pre_treatment = self.data["event_time"] < 0
        pretreatment_data = self.data[is_eventually_treated & is_pre_treatment].copy()

        if len(pretreatment_data) > 0:
            pretreat_indices = pretreatment_data.index.values
            pretreat_idx_positions = np.array(
                [np.where(self.data.index == idx)[0][0] for idx in pretreat_indices]
            )
            tau_draws_pretreat = tau_draws_all[:, :, pretreat_idx_positions]
            event_time_pretreat = np.asarray(pretreatment_data["event_time"].values)

            event_times_pre = np.unique(
                event_time_pretreat[~np.isnan(event_time_pretreat)]
            )
            if self.event_window is not None:
                event_times_pre = event_times_pre[
                    (event_times_pre >= self.event_window[0])
                    & (event_times_pre <= self.event_window[1])
                ]

            for e in sorted(event_times_pre):
                e_mask = event_time_pretreat == e
                if e_mask.sum() == 0:
                    continue
                positions_arr = np.where(e_mask)[0]
                tau_e = tau_draws_pretreat[:, :, positions_arr].mean(axis=2)
                att_et_rows.append(
                    {
                        "event_time": int(e),
                        "att": float(tau_e.mean()),
                        "att_lower": float(np.percentile(tau_e, lower_pct)),
                        "att_upper": float(np.percentile(tau_e, upper_pct)),
                        "n_obs": int(e_mask.sum()),
                    }
                )

        # Post-treatment effects (treated observations, event_time >= 0)
        _is_untreated = np.asarray(self.data["_is_untreated"].values, dtype=bool)
        treated_mask = ~_is_untreated
        treated_indices = np.where(treated_mask)[0]
        tau_draws_treated = tau_draws_all[:, :, treated_indices]

        treated_data = self.data[~self.data["_is_untreated"]].copy()
        event_time_treated = np.asarray(treated_data["event_time"].values)

        event_times_post = np.unique(event_time_treated[~np.isnan(event_time_treated)])
        if self.event_window is not None:
            event_times_post = event_times_post[
                (event_times_post >= self.event_window[0])
                & (event_times_post <= self.event_window[1])
            ]

        for e in sorted(event_times_post):
            e_mask = event_time_treated == e
            if e_mask.sum() == 0:
                continue
            positions_arr = np.where(e_mask)[0]
            tau_e = tau_draws_treated[:, :, positions_arr].mean(axis=2)
            att_et_rows.append(
                {
                    "event_time": int(e),
                    "att": float(tau_e.mean()),
                    "att_lower": float(np.percentile(tau_e, lower_pct)),
                    "att_upper": float(np.percentile(tau_e, upper_pct)),
                    "n_obs": int(e_mask.sum()),
                }
            )

        return pd.DataFrame(att_et_rows)

    def get_plot_data_ols(self) -> pd.DataFrame:
        """Get plotting data for OLS model.

        Returns
        -------
        pd.DataFrame
            DataFrame with event_time, att, att_std, n_obs columns.
        """
        return self.att_event_time_.copy()
