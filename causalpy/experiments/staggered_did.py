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
"""Staggered Difference in Differences (Imputation-based).

This module implements the imputation-based staggered DiD estimator, following
the approach of Borusyak, Jaravel, and Spiess (2024). It handles settings where
different units receive treatment at different times.
"""

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from patsy import PatsyError
from sklearn.base import RegressorMixin

from causalpy.constants import HDI_PROB, LEGEND_FONT_SIZE
from causalpy.custom_exceptions import DataException, FormulaException
from causalpy.experiments._results import StaggeredDifferenceInDifferencesResult
from causalpy.experiments.model_adapter import build_coords
from causalpy.formula_utils import build_formula_matrices
from causalpy.input_data import DataFrameLike, to_pandas
from causalpy.plot_utils import has_posterior_draws, plot_posterior_over_x
from causalpy.pymc_models import LinearRegression, PyMCModel
from causalpy.reporting import EffectSummary

from .base import BaseExperiment


class StaggeredDifferenceInDifferences(
    BaseExperiment[StaggeredDifferenceInDifferencesResult]
):
    """A class to analyse data from staggered adoption Difference-in-Differences settings.

    This class implements the Borusyak, Jaravel, and Spiess (BJS, 2024)
    imputation estimator for staggered adoption settings. It fits a model on
    untreated observations only (pre-treatment periods for eventually-treated
    units plus all periods for never-treated units), then predicts
    counterfactual outcomes for all observations. Treatment effects are computed
    as the difference between observed and predicted outcomes for treated
    observations.

    Parameters
    ----------
    data : dataframe-like
        Panel data (unit x time observations) as any eager dataframe Narwhals
        supports, such as pandas, Polars, or PyArrow. Converted to pandas
        internally.
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
        A model for the untreated outcome. Defaults to LinearRegression.
    event_window : tuple[int, int], optional
        Tuple (min_event_time, max_event_time) to restrict event-time aggregation.
        If None, uses all available event-times.
    reference_event_time : int, optional
        Event-time index associated with plots (reserved for future use).
        Defaults to -1.

    Attributes
    ----------
    Aggregated estimates live on the result bundle exposed through
    :attr:`result`: ``att_group_time`` and ``att_event_time`` DataFrames
    (each including an ``identified`` column; non-identified cells have
    ``NaN`` estimates), the raw counterfactual draws ``y_pred``, and the
    ``hdi_prob`` used during effect aggregation.
    non_identified_periods_ : set
        Calendar periods with no untreated observations.
    non_identified_cohorts_ : set
        Treatment cohorts with at least one non-identified post-treatment ATT(g, t).

    Notes
    -----
    **Estimate extraction**

    The Borusyak-Jaravel-Spiess imputation estimator fits the untreated outcome model using only observations that are not yet treated or never treated. It predicts each treated observation's untreated potential outcome, subtracts that prediction from the observed outcome, and averages the resulting one-sided contrasts into group-time and event-time ATTs. Bayesian aggregation retains posterior uncertainty in ``mu``; OLS aggregation uses point predictions and standard-error approximations.

    Like Interrupted Time Series, this fit-predict-subtract procedure is a reduced-form estimator. The corresponding structural contrast is a saturated regression as in Wooldridge's extended two-way fixed effects (ETWFE) framework, which CausalPy does not currently implement.

    This estimator requires the following identifying assumptions:

    1. **Absorbing treatment**: Once a unit receives treatment, it must remain
       treated in all subsequent periods. Treatment cannot be reversed or
       temporarily suspended. This is validated at runtime.
    2. **Parallel trends**: In the absence of treatment, treated and control
       units would have followed parallel outcome trajectories.
    3. **No anticipation**: Units do not change their behavior in anticipation
       of future treatment.
    4. **Untreated support at each calendar period**: The time fixed effect
       :math:`\\gamma_t` for calendar period :math:`t` is identified only if at
       least one unit is untreated in that period. Without never-treated units,
       post-treatment effects for the last-treated cohort (and any calendar
       periods where every unit is already treated) are not identified. CausalPy
       warns when this condition fails and marks the affected ``ATT(g, t)`` and
       ``ATT(e)`` cells as non-identified in the output tables.

    **Panel Balance**: This implementation supports both balanced and unbalanced panel
    data. While balanced panels (where each unit is observed in every time period) are
    common in staggered DiD applications, the imputation-based approach of Borusyak et
    al. (2024) can accommodate unbalanced panels. The key requirement is that treatment
    timing is well-defined for each unit, not that all units are observed in all periods.
    Unit and observation counts in the summary output are computed without assuming
    balanced panels.

    **Lazy lifecycle**

    Construction only validates inputs and builds design matrices — no
    sampling happens. Call :meth:`fit` to sample posterior draws and
    populate :attr:`result`, or :meth:`sample_prior_predictive` to run a
    prior predictive check (inspect it via ``plot(group="prior")``).

    References
    ----------
    Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting Event Study Designs:
    Robust and Efficient Estimation. Review of Economic Studies.

    Examples
    --------
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
    ... ).fit()  # doctest: +SKIP
    """

    supports_ols = True
    supports_bayes = True
    _default_model_class = LinearRegression

    def __init__(
        self,
        data: DataFrameLike,
        formula: str,
        unit_variable_name: str,
        time_variable_name: str,
        treated_variable_name: str = "treated",
        treatment_time_variable_name: str | None = None,
        never_treated_value: Any = np.inf,
        model: PyMCModel | RegressorMixin | None = None,
        event_window: tuple[int, int] | None = None,
        reference_event_time: int = -1,
    ) -> None:
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

        # to_pandas returns a copy, so the caller's dataframe is left alone
        data = to_pandas(data)
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

        # Step 3b: Check calendar-period identification support
        self._check_att_identification()

        # Step 4: Build design matrices
        self._build_design_matrices()

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
            if self.data[col].isna().any():
                raise DataException(
                    f"Required column '{col}' must not contain missing values"
                )

        # Check treated variable exists (either directly or via treatment_time)
        if self.treatment_time_variable_name is not None:
            if self.treatment_time_variable_name not in self.data.columns:
                raise DataException(
                    f"Treatment time column '{self.treatment_time_variable_name}' "
                    "not found in data"
                )
            if self.data[self.treatment_time_variable_name].isna().any():
                raise DataException(
                    f"Treatment time column '{self.treatment_time_variable_name}' "
                    "must not contain missing values"
                )
        elif self.treated_variable_name not in self.data.columns:
            raise DataException(
                f"Treated column '{self.treated_variable_name}' not found in data. "
                "Either provide treated_variable_name or treatment_time_variable_name."
            )
        elif self.data[self.treated_variable_name].isna().any():
            raise DataException(
                f"Treated column '{self.treated_variable_name}' "
                "must not contain missing values"
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
                self.data.groupby(self.unit_variable_name, observed=True)[
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
        # Construct a floating result before adding missing never-treated values:
        # pandas 3 disallows dtype-changing in-place assignment.
        event_time = (self.data[self.time_variable_name] - self.data["G"]).astype(float)
        self.data["event_time"] = event_time.where(
            self.data["G"] != self.never_treated_value, np.nan
        )

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

    def _get_periods_without_untreated_support(self) -> set[Any]:
        """Return calendar periods with zero untreated observations."""
        untreated_periods = set(
            self.data.loc[self.data["_is_untreated"], self.time_variable_name].unique()
        )
        all_periods = set(self.data[self.time_variable_name].unique())
        return all_periods - untreated_periods

    def _get_non_identified_cohorts(self, periods: set[Any]) -> set[Any]:
        """Return cohorts with post-treatment cells in non-identified periods."""
        non_identified_cohorts: set[Any] = set()
        for cohort in self.cohorts:
            for period in periods:
                if period >= cohort:
                    non_identified_cohorts.add(cohort)
                    break
        return non_identified_cohorts

    def _check_att_identification(self) -> None:
        """Detect non-identified ATT cells and warn when untreated support is missing."""
        self.non_identified_periods_ = self._get_periods_without_untreated_support()
        self.non_identified_cohorts_ = self._get_non_identified_cohorts(
            self.non_identified_periods_
        )

        if not self.non_identified_periods_:
            return

        periods_str = ", ".join(str(p) for p in sorted(self.non_identified_periods_))
        cohorts_str = ", ".join(str(c) for c in sorted(self.non_identified_cohorts_))
        warnings.warn(
            "No untreated observations in calendar period(s) "
            f"{{{periods_str}}}; treatment effects for cohort(s) "
            f"{{{cohorts_str}}} are not identified at the affected post-treatment "
            "cells. Provide never-treated units or restrict the event window. "
            "Non-identified ATT(g, t) and ATT(e) cells are marked in the output "
            "tables (identified=False) with NaN estimates.",
            UserWarning,
            stacklevel=2,
        )

    def _is_calendar_period_identified(self, period: Any) -> bool:
        """Return whether calendar period ``period`` has untreated support."""
        return period not in self.non_identified_periods_

    def _is_event_time_att_identified(self, event_time: int) -> bool:
        """Return whether aggregated ATT(e) is identified."""
        for cohort in self.cohorts:
            period = cohort + event_time
            has_contributing_obs = (
                (self.data["G"] == cohort)
                & (self.data[self.time_variable_name] == period)
                & (self.data["event_time"] == event_time)
            ).any()
            if has_contributing_obs and not self._is_calendar_period_identified(period):
                return False
        return True

    def _mark_non_identified_att_rows(self, att_df: pd.DataFrame) -> pd.DataFrame:
        """Add ``identified`` column and mask non-identified point estimates."""
        if len(att_df) == 0:
            att_df = att_df.copy()
            att_df["identified"] = pd.Series(dtype=bool)
            return att_df

        att_df = att_df.copy()
        if "cohort" in att_df.columns and "time" in att_df.columns:
            att_df["identified"] = att_df["time"].map(
                self._is_calendar_period_identified
            )
        elif "event_time" in att_df.columns:
            att_df["identified"] = att_df["event_time"].apply(
                lambda e: self._is_event_time_att_identified(int(e))
            )
        else:
            att_df["identified"] = True

        value_columns = [
            col
            for col in ("att", "att_lower", "att_upper", "att_std")
            if col in att_df.columns
        ]
        for col in value_columns:
            att_df.loc[~att_df["identified"], col] = np.nan
        return att_df

    def _build_design_matrices(self) -> None:
        """Build design matrices using patsy."""
        # Build design matrix for the full data
        try:
            y, X = build_formula_matrices(self.formula, self.data)
        except PatsyError as err:
            raise FormulaException(f"Unable to evaluate formula: {err}") from err
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.outcome_variable_name = y.design_info.column_names[0]

        # Store full design matrix
        self.X_full = np.asarray(X)
        self.y_full = np.asarray(y)
        self._observed_outcome = pd.Series(self.y_full.ravel(), index=self.data.index)

        # Get untreated subset for training
        untreated_mask = np.asarray(self.data["_is_untreated"].values, dtype=bool)
        self.X_train = self.X_full[untreated_mask]
        self.y_train = self.y_full[untreated_mask]

    def _fit_inputs(self) -> tuple[xr.DataArray, xr.DataArray, dict[str, Any]]:
        """Return the untreated-subset design matrices and coordinates for build.

        The model is built (and later sampled) on untreated observations
        only: pre-treatment periods of eventually-treated units plus all
        periods of never-treated units.
        """
        n_train = self.X_train.shape[0]
        X_train = xr.DataArray(
            self.X_train,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": np.arange(n_train),
                "coeffs": self.labels,
            },
        )
        y_train = xr.DataArray(
            self.y_train,
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": np.arange(n_train), "treated_units": ["unit_0"]},
        )
        return X_train, y_train, build_coords(self.labels, n_train)

    def _finalize(self, group: Literal["prior", "posterior"]) -> None:
        """Compute the group's result bundle from its draws and assign it.

        The base class samples the requested draw group before calling
        this, so no fitting happens here. The body is the historical
        predict-and-aggregate pipeline: predict counterfactual outcomes
        for every observation conditioned on ``group``, aggregate the
        resulting treatment effects into group-time and event-time ATT
        tables, and store everything in a result bundle. Prior draws are
        aggregated identically so prior plausibility checks can reuse the
        same readers.
        """
        n_full = self.X_full.shape[0]
        X_full_xr = xr.DataArray(
            self.X_full,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": np.arange(n_full),
                "coeffs": self.labels,
            },
        )
        y_pred = self._model_backend.predict(X=X_full_xr, group=group)

        treated_data = self.data[~self.data["_is_untreated"]].copy()

        # Also get pre-treatment data for eventually-treated units (placebo
        # check): G != never_treated_value AND event_time < 0
        is_eventually_treated = self.data["G"] != self.never_treated_value
        is_pre_treatment = self.data["event_time"] < 0
        pretreatment_data = self.data[is_eventually_treated & is_pre_treatment].copy()

        # The two helpers compute genuinely different statistics: HDI bounds
        # need posterior draws, sample dispersion needs only point residuals.
        if has_posterior_draws(y_pred):
            att_group_time, att_event_time, hdi_prob = self._aggregate_effects_bayesian(
                y_pred=y_pred,
                treated_data=treated_data,
                pretreatment_data=pretreatment_data,
            )
        else:
            att_group_time, att_event_time = self._aggregate_effects_ols(
                y_pred=y_pred,
                treated_data=treated_data,
                pretreatment_data=pretreatment_data,
            )
            hdi_prob = float(HDI_PROB)

        bundle = StaggeredDifferenceInDifferencesResult(
            att_group_time=att_group_time,
            att_event_time=att_event_time,
            y_pred=y_pred,
            hdi_prob=hdi_prob,
        )
        self._assign_bundle(group, bundle)

    def _aggregate_effects_bayesian(
        self,
        y_pred: xr.DataArray,
        treated_data: pd.DataFrame,
        pretreatment_data: pd.DataFrame,
        hdi_prob: float = HDI_PROB,
    ) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        """Aggregate effects for a draw-carrying prediction container.

        Parameters
        ----------
        y_pred : xr.DataArray
            Counterfactual draws for every observation.
        treated_data : pd.DataFrame
            DataFrame containing only treated observations (event_time >= 0)
        pretreatment_data : pd.DataFrame
            DataFrame containing pre-treatment observations from eventually-treated
            units (event_time < 0) for placebo check
        hdi_prob : float, optional
            Probability mass for the HDI interval bounds. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, float]
            The group-time ATT table, the event-time ATT table, and the HDI
            probability used for the interval bounds.
        """
        lower_pct = (1 - hdi_prob) / 2 * 100
        upper_pct = (1 + hdi_prob) / 2 * 100

        # Get posterior draws for mu
        mu_draws = y_pred.isel(treated_units=0)

        # Get observed y for all observations
        y_observed = self._observed_outcome.to_numpy()

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
        gt_groups = treated_data.groupby(
            ["G", self.time_variable_name], observed=True
        ).groups
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
        att_group_time = self._mark_non_identified_att_rows(pd.DataFrame(att_gt_rows))

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

        att_event_time = self._mark_non_identified_att_rows(pd.DataFrame(att_et_rows))

        return att_group_time, att_event_time, hdi_prob

    def _aggregate_effects_ols(
        self,
        y_pred: xr.DataArray,
        treated_data: pd.DataFrame,
        pretreatment_data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate effects for an OLS model (point estimates only).

        Parameters
        ----------
        y_pred : xr.DataArray
            Point counterfactual predictions for every observation (with
            singleton ``chain``/``draw`` dimensions).
        treated_data : pd.DataFrame
            DataFrame containing only treated observations (event_time >= 0)
        pretreatment_data : pd.DataFrame
            DataFrame containing pre-treatment observations from eventually-treated
            units (event_time < 0) for placebo check

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            The group-time ATT table and the event-time ATT table.
        """
        # Point counterfactual predictions per observation; treatment
        # effects are observed outcome minus prediction.
        y_hat0 = y_pred.mean(dim=["chain", "draw"]).isel(treated_units=0).values
        treated_positions = self.data.index.get_indexer(treated_data.index)
        treated_data["tau_hat"] = (
            self._observed_outcome.loc[treated_data.index].to_numpy()
            - y_hat0[treated_positions]
        )

        # --- Group-time ATTs (post-treatment only) ---
        att_gt = (
            treated_data.groupby(["G", self.time_variable_name], observed=True)[
                "tau_hat"
            ]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        att_gt.columns = ["cohort", "time", "att", "att_std", "n_obs"]
        att_group_time = self._mark_non_identified_att_rows(att_gt)

        # --- Event-time ATTs (including pre-treatment placebo) ---
        # Compute tau_hat for pre-treatment observations (residuals)
        if len(pretreatment_data) > 0:
            positions = self.data.index.get_indexer(pretreatment_data.index)
            pretreatment_data["tau_hat"] = (
                self._observed_outcome.loc[pretreatment_data.index].to_numpy()
                - y_hat0[positions]
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
            event_data.groupby("event_time", observed=True)["tau_hat"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        att_et.columns = ["event_time", "att", "att_std", "n_obs"]
        att_et["event_time"] = att_et["event_time"].astype(int)
        att_event_time = self._mark_non_identified_att_rows(att_et)

        return att_group_time, att_event_time

    def summary(
        self, round_to: int | None = 2, include_group_time: bool = False
    ) -> None:
        """Print summary of main results.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals for rounding. Defaults to 2.
        include_group_time : bool
            Whether to print the disaggregated cohort-by-calendar-time
            ``ATT(g, t)`` table after the event-time estimates. Defaults to
            ``False``.
        """
        bundle = self.result
        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print(f"Number of units: {self.data[self.unit_variable_name].nunique()}")
        print(f"Number of time periods: {self.data[self.time_variable_name].nunique()}")
        print(f"Treatment cohorts: {self.cohorts}")
        n_never_treated = self.data.loc[
            self.data["G"] == self.never_treated_value, self.unit_variable_name
        ].nunique()
        print(f"Never-treated units: {n_never_treated}")
        print("\nEvent-time estimates:")
        att_et = bundle.att_event_time.copy()
        # Add indicator column for clarity
        att_et["type"] = att_et["event_time"].apply(
            lambda x: "placebo" if x < 0 else "ATT"
        )
        # Reorder columns to put type first
        cols = ["event_time", "type"] + [
            c for c in att_et.columns if c not in ["event_time", "type"]
        ]
        print(att_et[cols].to_string(index=False))
        if include_group_time:
            print("\nGroup-time estimates:")
            print(bundle.att_group_time.to_string(index=False))
        print("\nModel coefficients:")
        self.print_coefficients(round_to)

    def plot(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        hdi_prob: float | None = None,
        figsize: tuple[float, float] = (10, 6),
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot the staggered difference-in-differences event study.

        Parameters
        ----------
        group : {"prior", "posterior"}, default "posterior"
            Which draw group to plot. ``"prior"`` renders a single-panel
            prior predictive check — the observed aggregate outcome against
            the prior-implied counterfactual — and requires
            :meth:`sample_prior_predictive`; ``"posterior"`` (default)
            renders the event study and requires :meth:`fit`. The two
            groups intentionally return different axes layouts.
        hdi_prob : float, optional
            Probability mass of the highest density interval shown by the
            error bars. Unlike most other CausalPy experiments, ``hdi_prob``
            for staggered DiD is fixed during effect aggregation and stored
            on the result bundle. If supplied here, the value must match
            ``result.hdi_prob``; otherwise a :class:`ValueError` is raised.
            Pass ``None`` (the default) to plot using the stored value.
            Ignored for OLS models and for ``group="prior"``.
        figsize : tuple of (float, float)
            Width and height of the figure in inches, passed to
            :func:`matplotlib.pyplot.subplots`. Defaults to ``(10, 6)``.
        show : bool
            Whether to automatically display the plot. Defaults to ``True``.
        legend_kwargs : dict, optional
            Keyword arguments to adjust legend placement and styling.
            Supported keys: ``loc``, ``bbox_to_anchor``, ``fontsize``,
            ``frameon``, ``title`` (``bbox_transform`` is accepted alongside
            ``bbox_to_anchor``). The existing legend is modified **in
            place** so that custom handles are preserved.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure that was created.
        ax : list[matplotlib.axes.Axes]
            A single-element list containing the event-study axes.
        """
        return self._render_plot(
            show=show,
            legend_kwargs=legend_kwargs,
            group=group,
            hdi_prob=hdi_prob,
            figsize=figsize,
        )

    def plot_group_time(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        hdi_prob: float | None = None,
        layout: Literal["facet", "overlay"] = "facet",
        x_axis: Literal["event_time", "calendar_time"] = "event_time",
        include_placebo: bool = True,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot cohort-specific ``ATT(g, t)`` trajectories.

        Parameters
        ----------
        group : {"prior", "posterior"}, default "posterior"
            Which draw group to plot. ``"prior"`` renders a single-panel
            prior predictive check and requires
            :meth:`sample_prior_predictive`; ``"posterior"`` (default)
            renders the cohort trajectories and requires :meth:`fit`.
        hdi_prob : float, optional
            Probability mass of the highest density interval shown by the
            uncertainty bands. As with :meth:`plot`, Bayesian ``ATT(g, t)``
            bounds are computed during effect aggregation and stored on the
            result bundle. If supplied here, the value must match
            ``result.hdi_prob``; otherwise a :class:`ValueError` is raised.
            Pass ``None`` (the default) to plot using the stored value.
            Ignored for OLS models and for ``group="prior"``.
        layout : {"facet", "overlay"}
            Plot layout. ``"facet"`` draws one row per cohort and
            ``"overlay"`` draws all cohorts on a single axes. Defaults to
            ``"facet"``.
        x_axis : {"event_time", "calendar_time"}
            Time scale for the cohort trajectories. ``"event_time"`` plots
            each cohort against periods since treatment, giving an
            ``ATT(g, e)`` view derived from ``ATT(g, t)``. ``"calendar_time"``
            plots each cohort against calendar time ``t``. Defaults to
            ``"event_time"``.
        include_placebo : bool
            Whether to include pre-treatment residual estimates for
            eventually-treated cohorts as placebo diagnostics. Defaults to
            ``True``.
        figsize : tuple of (float, float), optional
            Width and height of the figure in inches, passed to
            :func:`matplotlib.pyplot.subplots`. Defaults to a height scaled by
            the number of cohorts when ``layout="facet"`` and ``(10, 6)``
            when ``layout="overlay"``.
        show : bool
            Whether to automatically display the plot. Defaults to ``True``.
        legend_kwargs : dict, optional
            Keyword arguments to adjust legend placement and styling.
            Supported keys: ``loc``, ``bbox_to_anchor``, ``fontsize``,
            ``frameon``, ``title`` (``bbox_transform`` is accepted alongside
            ``bbox_to_anchor``). The existing legend is modified **in place**
            so that custom handles are preserved.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure that was created.
        ax : list[matplotlib.axes.Axes]
            Axes containing the cohort trajectories. The list has one axes
            per cohort when ``layout="facet"`` and one axes when
            ``layout="overlay"``.
        """
        return self._render_plot(
            show=show,
            legend_kwargs=legend_kwargs,
            group=group,
            hdi_prob=hdi_prob,
            layout=layout,
            x_axis=x_axis,
            include_placebo=include_placebo,
            figsize=figsize,
            view="group_time",
        )

    def _plot(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        hdi_prob: float | None = None,
        figsize: tuple[float, float] | None = (10, 6),
        view: Literal["event_time", "group_time"] = "event_time",
        layout: Literal["facet", "overlay"] = "facet",
        x_axis: Literal["event_time", "calendar_time"] = "event_time",
        include_placebo: bool = True,
        **kwargs: Any,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot the event study or cohort trajectories.

        Consumes the resolved group bundle injected by
        :meth:`~causalpy.experiments.base.BaseExperiment._render_plot`.

        Parameters
        ----------
        group : {"prior", "posterior"}
            ``"prior"`` renders the reduced single-panel prior-check figure
            via :meth:`_plot_prior_checks`; ``"posterior"`` renders the
            event study or cohort trajectories.
        hdi_prob : float, optional
            Probability mass of the highest density interval shown by the
            error bars. Unlike most other CausalPy experiments, ``hdi_prob``
            for ``StaggeredDiD`` is fixed during effect aggregation (see
            ``_aggregate_effects_bayesian``) and stored on the result
            bundle. If supplied here, the value must match
            ``bundle.hdi_prob``; otherwise a :class:`ValueError` is raised.
            Pass ``None`` (the default) to plot using the stored value.
            Ignored for point-estimate models.
        figsize : tuple of (float, float), optional
            Width and height of the figure in inches. Defaults to ``(10, 6)``.
        view : {"event_time", "group_time"}, optional
            Plot view to render. ``"event_time"`` draws the aggregated event
            study and ``"group_time"`` draws cohort-specific ``ATT(g, t)``
            trajectories. Defaults to ``"event_time"``.
        layout : {"facet", "overlay"}, optional
            Plot layout for the ``"group_time"`` view. Defaults to
            ``"facet"``.
        x_axis : {"event_time", "calendar_time"}, optional
            Time scale for the ``"group_time"`` view. Defaults to
            ``"event_time"``.
        include_placebo : bool, optional
            Whether to include pre-treatment residual estimates in the
            ``"group_time"`` view. Defaults to ``True``.

        Returns
        -------
        tuple[plt.Figure, list[plt.Axes]]
            Figure and axes objects.
        """
        bundle = self._require_bundle(group)
        if group == "prior":
            return self._plot_prior_checks(bundle=bundle)

        with_uncertainty = has_posterior_draws(bundle.y_pred)
        if with_uncertainty and hdi_prob is not None and hdi_prob != bundle.hdi_prob:
            raise ValueError(
                "StaggeredDiD HDI bounds are computed during effect "
                "aggregation, not at plot time. The stored HDI probability "
                f"is {bundle.hdi_prob}, but plot() received hdi_prob="
                f"{hdi_prob}. To plot at a different HDI probability, "
                "re-fit the experiment so that aggregation uses the desired "
                "value, or omit hdi_prob to use the stored value."
            )
        if view == "group_time":
            return self._plot_group_time(
                bundle=bundle,
                figsize=figsize,
                layout=layout,
                x_axis=x_axis,
                include_placebo=include_placebo,
            )
        if view != "event_time":
            raise ValueError("view must be 'event_time' or 'group_time'")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        att_et = bundle.att_event_time.copy()

        # Separate pre-treatment (placebo) and post-treatment (ATT)
        pre_treatment = att_et[att_et["event_time"] < 0]
        post_treatment = att_et[att_et["event_time"] >= 0]

        # Plot pre-treatment placebo estimates (different style)
        if len(pre_treatment) > 0:
            if with_uncertainty:
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
                    label=f"Placebo estimate ({int(bundle.hdi_prob * 100)}% HDI)",
                )
            else:
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
            if with_uncertainty:
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
                    label=f"ATT estimate ({int(bundle.hdi_prob * 100)}% HDI)",
                )
            else:
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

    def _plot_prior_checks(
        self, *, bundle: StaggeredDifferenceInDifferencesResult
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Render the reduced prior-check panel set.

        The observed aggregate outcome (mean across units per calendar
        period) is drawn against the prior-implied counterfactual.
        Prior-implied bands are typically far wider than the data, so
        impact-style panels are dropped rather than autoscaled into
        uselessness — the question a prior check answers is whether the
        prior counterfactual is plausible against the observed series, and
        one panel suffices.
        """
        pred = bundle.y_pred.isel(treated_units=0)
        obs_df = pd.DataFrame(
            {
                self.time_variable_name: self.data[self.time_variable_name].to_numpy(),
                "y": self._observed_outcome.to_numpy(),
            }
        )
        period_means = obs_df.groupby(self.time_variable_name, sort=True)["y"].mean()
        periods = period_means.index.to_numpy()
        time_vals = obs_df[self.time_variable_name].to_numpy()
        agg_pred = xr.concat(
            [
                pred.isel(obs_ind=np.where(time_vals == p)[0]).mean("obs_ind")
                for p in periods
            ],
            dim="obs_ind",
        )

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        h_line, h_patch = plot_posterior_over_x(
            periods,
            agg_pred,
            ax=ax,
            ci_prob=bundle.hdi_prob,
            kind="ribbon",
            ci_kind="hdi",
            plot_hdi_kwargs={"color": "C0"},
        )
        ax.plot(periods, period_means.to_numpy(), "k.", label="Observations")
        ax.set(
            title="Prior predictive check",
            xlabel=str(self.time_variable_name),
            ylabel=self.outcome_variable_name,
        )
        ax.legend(
            handles=[(h_line, h_patch)],
            labels=["Prior counterfactual"],
            fontsize=LEGEND_FONT_SIZE,
        )
        return fig, [ax]

    def _plot_group_time(
        self,
        *,
        bundle: StaggeredDifferenceInDifferencesResult,
        figsize: tuple[float, float] | None = None,
        layout: Literal["facet", "overlay"] = "facet",
        x_axis: Literal["event_time", "calendar_time"] = "event_time",
        include_placebo: bool = True,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot cohort-time ``ATT(g, t)`` trajectories."""
        att_gt, x_col, x_label, y_label = self._get_group_time_plot_data(
            bundle=bundle, x_axis=x_axis, include_placebo=include_placebo
        )
        cohort_groups = list(att_gt.groupby("cohort", observed=True, sort=True))
        sharex = x_axis == "event_time"
        fig, axes = self._make_group_time_axes(
            att_gt=att_gt,
            layout=layout,
            figsize=figsize,
            sharex=sharex,
            sharey=layout == "facet",
        )

        for cohort_idx, (cohort, cohort_data) in enumerate(cohort_groups):
            ax = axes[cohort] if layout == "facet" else axes["overlay"]
            self._plot_group_time_segment(
                ax=ax,
                cohort_data=cohort_data[cohort_data["type"] == "placebo"],
                x_col=x_col,
                line_type="placebo",
                color="gray" if layout == "facet" else f"C{cohort_idx % 10}",
                label=(
                    "Placebo estimate"
                    if layout == "facet"
                    else f"Cohort {cohort} placebo"
                ),
            )
            self._plot_group_time_segment(
                ax=ax,
                cohort_data=cohort_data[cohort_data["type"] == "ATT"],
                x_col=x_col,
                line_type="ATT",
                color="C0" if layout == "facet" else f"C{cohort_idx % 10}",
                label="ATT estimate" if layout == "facet" else f"Cohort {cohort} ATT",
            )
            self._format_group_time_axis(
                ax=ax,
                cohort=cohort if layout == "facet" else None,
                x_label=self._get_group_time_axis_label(
                    x_label=x_label,
                    layout=layout,
                    sharex=sharex,
                    axis_index=cohort_idx,
                    n_axes=len(cohort_groups),
                ),
                y_label=y_label,
                x_axis=x_axis,
                treatment_time=cohort,
            )
            ax.legend(fontsize=LEGEND_FONT_SIZE)

        if layout == "overlay":
            axes["overlay"].legend(title="Treatment cohort", fontsize=LEGEND_FONT_SIZE)

        return fig, list(axes.values())

    def _get_group_time_plot_data(
        self,
        *,
        bundle: StaggeredDifferenceInDifferencesResult,
        x_axis: Literal["event_time", "calendar_time"],
        include_placebo: bool,
    ) -> tuple[pd.DataFrame, str, str, str]:
        """Return cohort-time data with the requested plotting time scale."""
        if x_axis not in {"event_time", "calendar_time"}:
            raise ValueError("x_axis must be 'event_time' or 'calendar_time'")

        att_gt = bundle.att_group_time.sort_values(["cohort", "time"]).copy()
        att_gt["type"] = "ATT"
        if include_placebo:
            att_gt = pd.concat(
                [self._get_group_time_placebo_data(bundle=bundle), att_gt],
                ignore_index=True,
                sort=False,
            ).sort_values(["cohort", "time"])

        y_label = "ATT(g, e)" if x_axis == "event_time" else "ATT(g, t)"
        if include_placebo:
            y_label = f"{y_label} / placebo"

        if x_axis == "event_time":
            att_gt["event_time"] = att_gt["time"] - att_gt["cohort"]
            return (
                att_gt,
                "event_time",
                "Event Time (periods relative to treatment)",
                y_label,
            )
        return att_gt, "time", "Calendar Time", y_label

    def _get_group_time_placebo_data(
        self, *, bundle: StaggeredDifferenceInDifferencesResult
    ) -> pd.DataFrame:
        """Return cohort-time placebo estimates for eventually-treated units.

        The two helpers compute genuinely different statistics: HDI bounds
        need posterior draws, sample dispersion needs only point residuals.
        """
        if has_posterior_draws(bundle.y_pred):
            return self._get_group_time_placebo_data_bayesian(bundle=bundle)
        return self._get_group_time_placebo_data_ols(bundle=bundle)

    def _get_group_time_placebo_observations(self) -> pd.DataFrame:
        """Return pre-treatment observations for eventually-treated units."""
        is_eventually_treated = self.data["G"] != self.never_treated_value
        is_pre_treatment = self.data["event_time"] < 0
        return self.data[is_eventually_treated & is_pre_treatment].copy()

    def _get_group_time_placebo_data_bayesian(
        self, *, bundle: StaggeredDifferenceInDifferencesResult
    ) -> pd.DataFrame:
        """Return Bayesian cohort-time placebo estimates with HDI bounds."""
        pretreatment_data = self._get_group_time_placebo_observations()
        if len(pretreatment_data) == 0:
            return pd.DataFrame()

        hdi_prob = bundle.hdi_prob
        lower_pct = (1 - hdi_prob) / 2 * 100
        upper_pct = (1 + hdi_prob) / 2 * 100
        mu_draws = bundle.y_pred.isel(treated_units=0)
        y_observed = self._observed_outcome.to_numpy()
        tau_draws_all = y_observed - mu_draws.values

        att_gt_rows: list[dict[str, Any]] = []
        gt_groups = pretreatment_data.groupby(
            ["G", self.time_variable_name], observed=True
        ).groups
        for key, idx in gt_groups.items():
            g_val = key[0]  # type: ignore[index]
            t_val = key[1]  # type: ignore[index]
            positions = [np.where(self.data.index == i)[0][0] for i in idx]
            tau_gt = tau_draws_all[:, :, positions].mean(axis=2)
            att_gt_rows.append(
                {
                    "cohort": g_val,
                    "time": t_val,
                    "att": float(tau_gt.mean()),
                    "att_lower": float(np.percentile(tau_gt, lower_pct)),
                    "att_upper": float(np.percentile(tau_gt, upper_pct)),
                    "n_obs": len(positions),
                    "type": "placebo",
                }
            )
        return pd.DataFrame(att_gt_rows)

    def _get_group_time_placebo_data_ols(
        self, *, bundle: StaggeredDifferenceInDifferencesResult
    ) -> pd.DataFrame:
        """Return OLS cohort-time placebo residual estimates."""
        pretreatment_data = self._get_group_time_placebo_observations()
        if len(pretreatment_data) == 0:
            return pd.DataFrame()

        y_hat0 = bundle.y_pred.mean(dim=["chain", "draw"]).isel(treated_units=0).values
        positions = self.data.index.get_indexer(pretreatment_data.index)
        pretreatment_data["tau_hat"] = (
            self._observed_outcome.loc[pretreatment_data.index].to_numpy()
            - y_hat0[positions]
        )
        att_gt = (
            pretreatment_data.groupby(["G", self.time_variable_name], observed=True)[
                "tau_hat"
            ]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        att_gt.columns = ["cohort", "time", "att", "att_std", "n_obs"]
        att_gt["type"] = "placebo"
        return att_gt

    def _make_group_time_axes(
        self,
        att_gt: pd.DataFrame,
        layout: Literal["facet", "overlay"],
        figsize: tuple[float, float] | None,
        sharex: bool,
        sharey: bool,
    ) -> tuple[plt.Figure, dict[Any, plt.Axes]]:
        """Create axes for cohort trajectory plots."""
        cohorts = list(att_gt["cohort"].drop_duplicates())
        if layout == "overlay":
            fig, ax = plt.subplots(
                1, 1, figsize=figsize or (10, 6), layout="constrained"
            )
            return fig, {"overlay": ax}
        if layout != "facet":
            raise ValueError("layout must be 'facet' or 'overlay'")

        fig_height = max(2.5 * len(cohorts), 3.0)
        fig, axes_arr = plt.subplots(
            len(cohorts),
            1,
            figsize=figsize or (10, fig_height),
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
            layout="constrained",
        )
        return fig, {
            cohort: axes_arr[row_idx, 0] for row_idx, cohort in enumerate(cohorts)
        }

    def _format_group_time_axis(
        self,
        ax: plt.Axes,
        cohort: Any | None,
        x_label: str,
        y_label: str,
        x_axis: Literal["event_time", "calendar_time"],
        treatment_time: Any,
    ) -> None:
        """Apply shared formatting for cohort trajectory axes."""
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        if x_axis == "event_time":
            ax.axvline(x=-0.5, color="red", linestyle="-", linewidth=1, alpha=0.5)
        elif cohort is not None:
            ax.axvline(
                x=treatment_time - 0.5,
                color="red",
                linestyle="-",
                linewidth=1,
                alpha=0.5,
            )
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        if cohort is None:
            ax.set_title("Staggered DiD Cohort Trajectories", fontsize=14)
        else:
            ax.set_title(f"Cohort {cohort}", fontsize=12)

    def _get_group_time_axis_label(
        self,
        x_label: str,
        layout: Literal["facet", "overlay"],
        sharex: bool,
        axis_index: int,
        n_axes: int,
    ) -> str:
        """Return an x-axis label only where it helps the figure."""
        if layout == "facet" and sharex and axis_index < n_axes - 1:
            return ""
        return x_label

    def _plot_group_time_segment(
        self,
        ax: plt.Axes,
        cohort_data: pd.DataFrame,
        x_col: str,
        line_type: Literal["placebo", "ATT"],
        color: str,
        label: str,
    ) -> None:
        """Plot one placebo or ATT segment for a cohort.

        Uncertainty rendering keys on which columns the aggregation produced:
        HDI bounds (``att_lower`` / ``att_upper``) draw a shaded band, sample
        dispersion (``att_std`` / ``n_obs``) draws 1.96-SE error bars, and
        bare point estimates draw a plain line.
        """
        if len(cohort_data) == 0:
            return

        marker = "s" if line_type == "placebo" else "o"
        linestyle = "--" if line_type == "placebo" else "-"
        if {"att_lower", "att_upper"}.issubset(cohort_data.columns):
            alpha = 0.15 if line_type == "placebo" else 0.2
            ax.plot(
                cohort_data[x_col],
                cohort_data["att"],
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=label,
            )
            ax.fill_between(
                cohort_data[x_col],
                cohort_data["att_lower"],
                cohort_data["att_upper"],
                color=color,
                alpha=alpha,
            )
        elif {"att_std", "n_obs"}.issubset(cohort_data.columns):
            se = cohort_data["att_std"] / np.sqrt(cohort_data["n_obs"])
            ax.errorbar(
                cohort_data[x_col],
                cohort_data["att"],
                yerr=1.96 * se,
                fmt=f"{marker}{linestyle}",
                capsize=4,
                capthick=2,
                color=color,
                label=label,
            )
        else:
            ax.plot(
                cohort_data[x_col],
                cohort_data["att"],
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=label,
            )

    def get_plot_data(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        hdi_prob: float = HDI_PROB,
    ) -> pd.DataFrame:
        """Get event-time plotting data.

        Parameters
        ----------
        group : {"prior", "posterior"}, default "posterior"
            Which draw group to summarize. ``"prior"`` requires
            :meth:`sample_prior_predictive`; ``"posterior"`` requires
            :meth:`fit`.
        hdi_prob : float, optional
            Probability for HDI interval. Only used by models carrying
            draws; when it differs from the value stored on the result
            bundle, the intervals are recomputed. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).

        Returns
        -------
        pd.DataFrame
            DataFrame with ``event_time`` and ``att`` columns plus
            ``att_lower`` / ``att_upper`` HDI bounds (draw-carrying models)
            or ``att_std`` / ``n_obs`` dispersion columns (point estimates).
            Includes both pre-treatment (placebo) and post-treatment
            effects. Not cached on the experiment.
        """
        bundle = self._require_bundle(group)
        # If there are no draws, or the requested hdi_prob matches what was
        # used during aggregation, return the pre-computed results
        stored_hdi_prob = bundle.hdi_prob
        if not has_posterior_draws(bundle.y_pred) or np.isclose(
            hdi_prob, stored_hdi_prob
        ):
            return bundle.att_event_time.copy()

        # Recompute intervals with the requested hdi_prob
        lower_pct = (1 - hdi_prob) / 2 * 100
        upper_pct = (1 + hdi_prob) / 2 * 100

        # Get draws for mu
        mu_draws = bundle.y_pred.isel(treated_units=0)

        # Get observed y for all observations
        y_observed = self._observed_outcome.to_numpy()

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

        return self._mark_non_identified_att_rows(pd.DataFrame(att_et_rows))

    def effect_summary(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        min_effect: float | None = None,
    ) -> EffectSummary:
        """
        Generate a decision-ready summary of causal effects for Staggered Difference-in-Differences.

        Parameters
        ----------
        group : {"prior", "posterior"}, default "posterior"
            Which draw group to summarize. ``"prior"`` requires
            :meth:`sample_prior_predictive` and produces prior-appropriate
            prose — a prior plausibility statement, not a causal estimate;
            ``"posterior"`` requires :meth:`fit`.
        direction : {"increase", "decrease", "two-sided"}, default="increase"
            Direction for tail probability calculation (PyMC only, ignored for OLS).
        alpha : float, default=0.05
            Significance level for HDI/CI intervals (1-alpha confidence level).
        min_effect : float, optional
            Region of Practical Equivalence (ROPE) threshold (PyMC only, ignored for OLS).

        Returns
        -------
        EffectSummary
            Object with .table (DataFrame) and .text (str) attributes
        """
        # Resolve the requested group's bundle; the helper reads ATT tables
        # from it and frames prior-group prose as a plausibility check.
        bundle = self._require_bundle(group)
        from causalpy.reporting import _effect_summary_staggered_did

        return _effect_summary_staggered_did(
            self,
            bundle,
            group=group,
            direction=direction,
            alpha=alpha,
            min_effect=min_effect,
        )
