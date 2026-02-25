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
Graded Intervention Time Series Experiment

This module implements experiments for estimating the causal effects of graded
interventions (e.g., media spend, policy intensity) in single-market time series
using transfer functions that model saturation and adstock (carryover) effects.

The experiment works with the TransferFunctionOLS model class (from skl_models)
to provide a complete causal inference workflow including visualization,
diagnostics, and counterfactual effect estimation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.base import RegressorMixin
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from causalpy.custom_exceptions import BadIndexException
from causalpy.pymc_models import PyMCModel
from causalpy.transforms import Treatment
from causalpy.utils import round_num

from .base import BaseExperiment

if TYPE_CHECKING:
    from causalpy.reporting import EffectSummary

LEGEND_FONT_SIZE = 12


class GradedInterventionTimeSeries(BaseExperiment):
    """
    Interrupted time series experiment with graded interventions and transfer functions.

    This experiment class handles causal inference for time series with graded
    (non-binary) interventions, incorporating saturation and adstock effects.
    Following the standard CausalPy pattern, it takes data and an unfitted model,
    performs transform parameter estimation, fits the model, and provides
    visualization, diagnostics, and counterfactual effect estimation.

    Typical workflow:
    1. Create an unfitted TransferFunctionOLS model with configuration
    2. Pass data + model to this experiment class
    3. Experiment estimates transforms, fits model, and provides results
    4. Use experiment methods for visualization and effect estimation

    Fitting Procedure
    -----------------
    The experiment uses a nested optimization approach to estimate transform parameters
    and fit the regression model:

    **Outer Loop (Transform Parameter Estimation):**
    The experiment searches for optimal saturation and adstock parameters either via
    grid search (exhaustive evaluation of discrete parameter combinations) or continuous
    optimization (gradient-based search). For grid search with N saturation parameter
    combinations and M adstock parameter combinations, all N x M combinations are
    evaluated.

    **Inner Loop (Model Fitting):**
    For each candidate set of transform parameters, the raw treatment variable is
    transformed by applying saturation (diminishing returns) and adstock (carryover
    effects). The transformed treatment is combined with baseline predictors to create
    a full design matrix, and an OLS or ARIMAX model is fitted. The model that achieves
    the lowest root mean squared error (RMSE) determines the selected parameters.

    This nested approach is computationally efficient because OLS has a closed-form
    solution requiring only matrix operations, making each individual model fit very
    fast. For ARIMAX error models, numerical optimization is required for each fit,
    increasing computational cost but providing explicit modeling of autocorrelation
    structure.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data with datetime or numeric index.
    y_column : str
        Name of the outcome variable column.
    treatment_names : List[str]
        List of treatment variable names (e.g., ["comm_intensity"]).
    base_formula : str
        Patsy formula for baseline model (e.g., "1 + t + temperature").
    model : TransferFunctionOLS
        UNFITTED model with configuration for transform parameter estimation.

    Attributes
    ----------
    data : pd.DataFrame
        Input data.
    y : np.ndarray
        Outcome variable values.
    X_baseline : np.ndarray
        Baseline design matrix.
    Z_treatment : np.ndarray
        Treatment design matrix.
    X_full : np.ndarray
        Full design matrix.
    predictions : np.ndarray
        Fitted values from model.
    residuals : np.ndarray
        Model residuals.
    score : float
        R-squared of the model.

    Examples
    --------
    .. code-block:: python

        import causalpy as cp

        # Step 1: Create UNFITTED model with configuration
        model = cp.skl_models.TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [1.0, 2.0, 3.0], "kappa": [3, 5, 7]},
            adstock_grid={"half_life": [2, 3, 4, 5]},
            estimation_method="grid",
            error_model="hac",
        )

        # Step 2: Pass to experiment (experiment estimates transforms and fits model)
        result = cp.GradedInterventionTimeSeries(
            data=df,
            y_column="water_consumption",
            treatment_names=["comm_intensity"],
            base_formula="1 + t + temperature + rainfall",
            model=model,
        )

        # Step 3: Use experiment methods
        result.summary()
        result.plot()
        result.plot_diagnostics()
        effect = result.effect(window=(df.index[0], df.index[-1]), scale=0.0)
    """

    expt_type = "Graded Intervention Time Series"
    supports_ols = True
    supports_bayes = True

    def __init__(
        self,
        data: pd.DataFrame,
        y_column: str,
        treatment_names: list[str],
        base_formula: str,
        model=None,
        **kwargs,
    ):
        """
        Initialize experiment with data and unfitted model (standard CausalPy pattern).

        This method:
        1. Validates inputs and builds baseline design matrix
        2. Estimates transform parameters for each treatment
        3. Applies transforms and builds full design matrix
        4. Calls model.fit(X_full, y)
        5. Extracts results for visualization and analysis

        Parameters
        ----------
        data : pd.DataFrame
            Time series data.
        y_column : str
            Name of outcome variable.
        treatment_names : List[str]
            List of treatment variable names (e.g., ["comm_intensity"]).
        base_formula : str
            Patsy formula for baseline model.
        model : TransferFunctionOLS
            UNFITTED model with configuration for transform estimation.
        """
        super().__init__(model=model)

        # Validate inputs
        self._validate_inputs(data, y_column, treatment_names)

        # Store attributes
        self.data = data.copy()
        self.y_column = y_column
        self.treatment_names = treatment_names
        self.base_formula = base_formula

        # Extract outcome variable
        self.y = np.asarray(data[y_column].values)

        # Build baseline design matrix (like other experiments do)
        self.X_baseline = np.asarray(dmatrix(base_formula, data))
        self.baseline_labels = dmatrix(base_formula, data).design_info.column_names

        # ====================================================================
        # Conditional logic: Bayesian vs OLS models
        # ====================================================================
        if isinstance(self.model, PyMCModel):
            # ================================================================
            # BAYESIAN MODEL PATH
            # ================================================================
            # For Bayesian models, transforms are estimated jointly within PyMC
            # So we skip grid/optimize parameter estimation and pass raw data

            import xarray as xr

            # Convert to xarray format (like DifferenceInDifferences does)
            self.X = xr.DataArray(
                self.X_baseline,
                dims=["obs_ind", "coeffs"],
                coords={
                    "obs_ind": np.arange(self.X_baseline.shape[0]),
                    "coeffs": self.baseline_labels,
                },
            )
            self.y = xr.DataArray(  # type: ignore[assignment]
                self.y.reshape(-1, 1),
                dims=["obs_ind", "treated_units"],
                coords={
                    "obs_ind": np.arange(len(self.y)),
                    "treated_units": ["unit_0"],
                },
            )

            # Get raw treatment data
            treatment_raw = np.column_stack(
                [np.asarray(data[name].values) for name in treatment_names]
            )
            treatment_data = xr.DataArray(
                treatment_raw,
                dims=["obs_ind", "treatment_names"],
                coords={
                    "obs_ind": np.arange(treatment_raw.shape[0]),
                    "treatment_names": treatment_names,
                },
            )

            # Setup coordinates for PyMC
            COORDS = {
                "coeffs": self.baseline_labels,
                "obs_ind": np.arange(self.X_baseline.shape[0]),
                "treated_units": ["unit_0"],
                "treatment_names": treatment_names,
            }

            # Fit Bayesian model (subclass overrides accept treatment_data)
            self.model.fit(  # type: ignore[call-arg]
                X=self.X,
                y=self.y,  # type: ignore[arg-type]
                coords=COORDS,
                treatment_data=treatment_data,
            )

            # Store for later use
            self.treatment_labels = treatment_names
            self.treatments = None  # Not used for Bayesian (transforms in model)

        elif isinstance(self.model, RegressorMixin):
            # ================================================================
            # OLS MODEL PATH
            # ================================================================
            # Estimate transform parameters for each treatment
            from causalpy.transform_optimization import (
                estimate_transform_params_grid,
                estimate_transform_params_optimize,
            )
            from causalpy.transforms import Treatment

            self.treatments = []
            Z_columns = []
            self.treatment_labels = []

            for name in treatment_names:
                # Run parameter estimation using model configuration
                if self.model.estimation_method == "grid":
                    est_results = estimate_transform_params_grid(
                        data=data,
                        y_column=y_column,
                        treatment_name=name,
                        base_formula=base_formula,
                        saturation_type=self.model.saturation_type,
                        saturation_grid=self.model.saturation_grid,
                        adstock_grid=self.model.adstock_grid,
                        coef_constraint=self.model.coef_constraint,
                        hac_maxlags=self.model.hac_maxlags,
                        error_model=self.model.error_model,
                        arima_order=self.model.arima_order,
                    )
                    search_space = {
                        "saturation_grid": self.model.saturation_grid,
                        "adstock_grid": self.model.adstock_grid,
                    }
                elif self.model.estimation_method == "optimize":
                    est_results = estimate_transform_params_optimize(
                        data=data,
                        y_column=y_column,
                        treatment_name=name,
                        base_formula=base_formula,
                        saturation_type=self.model.saturation_type,
                        saturation_bounds=self.model.saturation_bounds,
                        adstock_bounds=self.model.adstock_bounds,
                        initial_params=None,
                        coef_constraint=self.model.coef_constraint,
                        hac_maxlags=self.model.hac_maxlags,
                        method="L-BFGS-B",
                        error_model=self.model.error_model,
                        arima_order=self.model.arima_order,
                    )
                    search_space = {
                        "saturation_bounds": self.model.saturation_bounds,
                        "adstock_bounds": self.model.adstock_bounds,
                    }

                # Store estimation metadata on model
                self.model.transform_estimation_results = est_results
                self.model.transform_search_space = search_space

                # Create Treatment with estimated transforms
                treatment = Treatment(
                    name=name,
                    saturation=est_results["best_saturation"],
                    adstock=est_results["best_adstock"],
                    coef_constraint=self.model.coef_constraint,
                )
                self.treatments.append(treatment)

                # Apply transforms
                x_raw = np.asarray(data[name].values)
                x_transformed = x_raw
                if treatment.saturation is not None:
                    x_transformed = treatment.saturation.apply(x_transformed)
                if treatment.adstock is not None:
                    x_transformed = treatment.adstock.apply(x_transformed)
                if treatment.lag is not None:
                    x_transformed = treatment.lag.apply(x_transformed)

                Z_columns.append(x_transformed)
                self.treatment_labels.append(name)

            # Build full design matrix
            self.Z_treatment = np.column_stack(Z_columns)
            self.X_full = np.column_stack([self.X_baseline, self.Z_treatment])
            self.all_labels = self.baseline_labels + self.treatment_labels

            # Store treatments on model for later use
            self.model.treatments = self.treatments

            # Fit the model (standard CausalPy pattern)
            self.model.fit(X=self.X_full, y=self.y)

            # Extract results from fitted model
            self.ols_result = self.model.ols_result
            self.predictions = self.model.ols_result.fittedvalues
            self.residuals = self.model.ols_result.resid
            self.score = self.model.score

            # Extract coefficients (handling ARIMAX correctly)
            if self.model.error_model == "arimax":
                # ARIMAX: extract only exogenous coefficients
                n_exog = self.ols_result.model.k_exog
                exog_params = self.ols_result.params[:n_exog]
                n_baseline = self.X_baseline.shape[1]
                self.beta_baseline = exog_params[:n_baseline]
                self.theta_treatment = exog_params[n_baseline:]
            else:
                # OLS: all params are regression coefficients
                n_baseline = self.X_baseline.shape[1]
                self.beta_baseline = self.ols_result.params[:n_baseline]
                self.theta_treatment = self.ols_result.params[n_baseline:]

            # Store model metadata for summary output
            self.error_model = self.model.error_model
            self.hac_maxlags = self.model.hac_maxlags
            self.arima_order = self.model.arima_order
            self.transform_estimation_method = self.model.estimation_method
            self.transform_estimation_results = self.model.transform_estimation_results
            self.transform_search_space = self.model.transform_search_space
        else:
            raise ValueError("Model type not recognized")

    def _validate_inputs(
        self,
        data: pd.DataFrame,
        y_column: str,
        treatment_names: list[str],
    ) -> None:
        """Validate input data and parameters."""
        # Check that y_column exists
        if y_column not in data.columns:
            raise ValueError(f"y_column '{y_column}' not found in data columns")

        # Check that treatment columns exist
        for name in treatment_names:
            if name not in data.columns:
                raise ValueError(f"Treatment column '{name}' not found in data columns")

        # Check for missing values in outcome
        if data[y_column].isna().any():
            raise ValueError("Outcome variable contains missing values")

        # Warn about missing values in treatment columns
        for name in treatment_names:
            if data[name].isna().any():
                print(
                    f"Warning: Treatment column '{name}' contains missing values. "
                    f"Consider forward-filling if justified by the context."
                )

        # Check that we have a time index
        valid_index_types = (pd.DatetimeIndex, pd.RangeIndex)
        is_valid_index = isinstance(data.index, valid_index_types) or (
            isinstance(data.index, pd.Index)
            and pd.api.types.is_integer_dtype(data.index)
        )

        if not is_valid_index:
            raise BadIndexException(
                "Data index must be DatetimeIndex, RangeIndex, or integer Index for time series"
            )

    def _build_treatment_matrix(
        self, data: pd.DataFrame, treatments: list[Treatment]
    ) -> tuple[np.ndarray, list[str]]:
        """Build the treatment design matrix by applying transforms.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with treatment columns.
        treatments : List[Treatment]
            Treatment specifications.

        Returns
        -------
        Z : np.ndarray
            Treatment design matrix (n_obs, n_treatments).
        labels : List[str]
            Column labels for treatments.
        """
        Z_columns = []
        labels = []

        for treatment in treatments:
            # Get raw exposure series
            x_raw = np.asarray(data[treatment.name].values)

            # Apply transform pipeline: Saturation → Adstock → Lag
            x_transformed = x_raw
            if treatment.saturation is not None:
                x_transformed = treatment.saturation.apply(x_transformed)
            if treatment.adstock is not None:
                x_transformed = treatment.adstock.apply(x_transformed)
            if treatment.lag is not None:
                x_transformed = treatment.lag.apply(x_transformed)

            Z_columns.append(x_transformed)
            labels.append(treatment.name)

        Z = np.column_stack(Z_columns)
        return Z, labels

    def effect_summary(self, **kwargs: Any) -> EffectSummary:
        """Generate a decision-ready summary of causal effects.

        Not yet implemented for GradedInterventionTimeSeries. Use the
        :meth:`effect` method for counterfactual effect estimation.
        """
        raise NotImplementedError(
            "effect_summary is not yet implemented for GradedInterventionTimeSeries. "
            "Use the .effect() method for counterfactual effect estimation."
        )

    def effect(
        self,
        window: tuple[pd.Timestamp | int, pd.Timestamp | int],
        channels: list[str] | None = None,
        scale: float = 0.0,
    ) -> dict[str, Any]:
        """Estimate the causal effect of scaling treatment channels in a time window.

        This method computes a counterfactual scenario by scaling the specified
        treatment channels in the given window, reapplying all transforms with
        the same parameters, and comparing to the observed outcome.

        For Bayesian models, returns posterior distributions of effects with credible intervals.
        For OLS models, returns point estimates.

        Parameters
        ----------
        window : Tuple[Union[pd.Timestamp, int], Union[pd.Timestamp, int]]
            Start and end of the effect window (inclusive).
        channels : List[str], optional
            List of treatment channel names to scale. If None, scales all channels.
        scale : float, default=0.0
            Scaling factor for the counterfactual (0.0 = remove treatment).

        Returns
        -------
        result : Dict
            Dictionary containing:
            - "effect_df": DataFrame with observed, counterfactual, effect, cumulative effect
            - "total_effect": Total effect in window (mean for Bayesian)
            - "mean_effect": Mean effect per period in window
            For Bayesian models, also includes HDI bounds for counterfactual and effect.

        Examples
        --------
        .. code-block:: python

            # Estimate effect of removing treatment completely
            effect = result.effect(
                window=(df.index[0], df.index[-1]),
                channels=["comm_intensity"],
                scale=0.0,
            )
            print(f"Total effect: {effect['total_effect']:.2f}")
        """
        # Route to appropriate method based on model type
        if isinstance(self.model, PyMCModel):
            return self._bayesian_effect(window=window, channels=channels, scale=scale)
        else:
            return self._ols_effect(window=window, channels=channels, scale=scale)

    def _ols_effect(
        self,
        window: tuple[pd.Timestamp | int, pd.Timestamp | int],
        channels: list[str] | None = None,
        scale: float = 0.0,
    ) -> dict[str, Any]:
        """Estimate the causal effect for OLS models (point estimates)."""
        # Default to all channels if not specified
        if channels is None:
            channels = self.treatment_names

        # Validate channels
        for ch in channels:
            if ch not in self.treatment_names:
                raise ValueError(f"Channel '{ch}' not found in treatments")

        # Get window mask
        window_start, window_end = window
        if isinstance(self.data.index, pd.DatetimeIndex):
            mask = (
                self.data.index >= window_start  # type: ignore[operator]
            ) & (
                self.data.index <= window_end  # type: ignore[operator]
            )
        else:
            mask = (self.data.index >= window_start) & (self.data.index <= window_end)

        # Create counterfactual data by scaling specified channels in the window
        data_cf = self.data.copy()
        for channel in channels:
            data_cf.loc[mask, channel] = scale * data_cf.loc[mask, channel]

        # Reapply transforms to counterfactual data
        Z_cf, _ = self._build_treatment_matrix(data_cf, self.treatments)

        # Predict counterfactual
        X_cf_full = np.column_stack([self.X_baseline, Z_cf])

        # For ARIMAX, extract only exogenous coefficients (exclude ARIMA params)
        if hasattr(self.ols_result.model, "k_exog"):
            # ARIMAX: params includes exog coefficients + ARIMA params
            exog_params = self.ols_result.params[: self.ols_result.model.k_exog]
            y_cf = X_cf_full @ exog_params
        else:
            # OLS: all params are regression coefficients
            y_cf = X_cf_full @ self.ols_result.params

        # Compute effect
        effect = self.y - y_cf

        # Create result DataFrame
        effect_df = pd.DataFrame(
            {
                "observed": self.y,
                "counterfactual": y_cf,
                "effect": effect,
                "effect_cumulative": np.cumsum(effect),
            },
            index=self.data.index,
        )

        # Filter to window for summary statistics
        window_effect = effect[mask]

        result = {
            "effect_df": effect_df,
            "total_effect": float(np.sum(window_effect)),
            "mean_effect": float(np.mean(window_effect)),
            "window_start": window_start,
            "window_end": window_end,
            "channels": channels,
            "scale": scale,
        }

        return result

    def plot_effect(
        self,
        effect_result: dict,
        **kwargs,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot counterfactual effect analysis results.

        Creates a 2-panel figure showing:
        1. Observed vs counterfactual outcome
        2. Cumulative effect over time

        For Bayesian models, shows posterior mean with 94% credible intervals.
        For OLS models, shows point estimates.

        Parameters
        ----------
        effect_result : dict
            Result dictionary from effect() method containing:
            - effect_df: DataFrame with observed, counterfactual, effect columns
            - window_start, window_end: Effect window boundaries
            - channels: List of scaled channels
            - scale: Scaling factor used
            - total_effect: Total effect in window
            - mean_effect: Mean effect per period in window

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : array of matplotlib.axes.Axes
            Array of 2 axes objects (top: observed vs counterfactual, bottom: cumulative effect).

        Examples
        --------
        .. code-block:: python

            # Estimate effect of removing treatment
            effect_result = result.effect(
                window=(df.index[0], df.index[-1]),
                channels=["comm_intensity"],
                scale=0.0,
            )
            fig, ax = result.plot_effect(effect_result)
        """
        # Route to appropriate plot method based on model type
        if isinstance(self.model, PyMCModel):
            return self._bayesian_plot_effect(effect_result=effect_result, **kwargs)
        else:
            return self._ols_plot_effect(effect_result=effect_result, **kwargs)

    def _ols_plot_effect(
        self,
        effect_result: dict,
        **kwargs,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot counterfactual effect analysis for OLS models."""
        # Extract data from effect result
        effect_df = effect_result["effect_df"]
        window_start = effect_result.get("window_start")
        window_end = effect_result.get("window_end")

        # Create 2-panel subplot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # ============================================================================
        # TOP PANEL: Observed vs Counterfactual
        # ============================================================================
        axes[0].plot(
            effect_df.index,
            effect_df["observed"],
            label="Observed",
            linewidth=1.5,
        )
        axes[0].plot(
            effect_df.index,
            effect_df["counterfactual"],
            label="Counterfactual",
            linewidth=1.5,
            linestyle="--",
        )

        # Shade the effect region
        axes[0].fill_between(
            effect_df.index,
            effect_df["observed"],
            effect_df["counterfactual"],
            alpha=0.3,
            color="C2",
            label="Effect",
        )

        axes[0].set_ylabel(self.y_column, fontsize=11)
        axes[0].set_title("Observed vs Counterfactual", fontsize=12, fontweight="bold")
        axes[0].legend(fontsize=LEGEND_FONT_SIZE)
        axes[0].grid(True, alpha=0.3)

        # Add window boundaries if specified
        if window_start is not None and window_end is not None:
            axes[0].axvline(x=window_start, color="red", linestyle=":", alpha=0.5)
            axes[0].axvline(x=window_end, color="red", linestyle=":", alpha=0.5)

        # ============================================================================
        # BOTTOM PANEL: Cumulative effect
        # ============================================================================
        axes[1].plot(
            effect_df.index,
            effect_df["effect_cumulative"],
            linewidth=2,
            color="C2",
        )
        axes[1].fill_between(
            effect_df.index,
            0,
            effect_df["effect_cumulative"],
            alpha=0.3,
            color="C2",
        )
        axes[1].axhline(y=0, color="k", linestyle="--", linewidth=1)
        axes[1].set_ylabel("Cumulative Effect", fontsize=11)
        axes[1].set_xlabel("Time", fontsize=11)
        axes[1].set_title("Cumulative Effect Over Time", fontsize=12, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

        # Add window boundaries
        if window_start is not None and window_end is not None:
            axes[1].axvline(x=window_start, color="red", linestyle=":", alpha=0.5)
            axes[1].axvline(x=window_end, color="red", linestyle=":", alpha=0.5)

        plt.tight_layout()
        return fig, axes

    def plot(self, round_to: int | None = 2, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """Plot the model fit and results.

        Creates a 2-panel figure showing:
        1. Observed vs fitted values
        2. Residuals over time

        Parameters
        ----------
        round_to : int, optional
            Number of decimal places for rounding displayed values.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : array of matplotlib.axes.Axes
        """
        # Route to appropriate plot method based on model type
        if isinstance(self.model, PyMCModel):
            return self._bayesian_plot(round_to=round_to, **kwargs)
        else:
            return self._ols_plot(round_to=round_to, **kwargs)

    def _ols_plot(
        self, round_to: int | None = 2, **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        """Generate OLS-specific plots."""
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Top panel: Observed vs fitted
        ax[0].plot(
            self.data.index, self.y, "o", label="Observed", alpha=0.6, markersize=4
        )
        ax[0].plot(
            self.data.index,
            self.predictions,
            "-",
            label="Fitted",
            linewidth=2,
            color="C1",
        )
        ax[0].set_ylabel("Outcome")
        ax[0].set_title(f"Model Fit: R² = {round_num(self.score, round_to)}")
        ax[0].legend(fontsize=LEGEND_FONT_SIZE)
        ax[0].grid(True, alpha=0.3)

        # Bottom panel: Residuals
        ax[1].plot(self.data.index, self.residuals, "o-", alpha=0.6, markersize=3)
        ax[1].axhline(y=0, color="k", linestyle="--", linewidth=1)
        ax[1].set_ylabel("Residuals")
        ax[1].set_xlabel("Time")
        ax[1].set_title("Model Residuals")
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def plot_irf(self, channel: str, max_lag: int | None = None) -> plt.Figure | None:
        """Plot the Impulse Response Function (IRF) for a treatment channel.

        Shows how a one-unit impulse in the (saturated) exposure propagates over
        time through the adstock transformation.

        Parameters
        ----------
        channel : str
            Name of the treatment channel.
        max_lag : int, optional
            Maximum lag to display.

        Returns
        -------
        fig : matplotlib.figure.Figure

        Examples
        --------
        .. code-block:: python

            fig = result.plot_irf("comm_intensity", max_lag=12)
        """
        # Find the treatment
        treatment = None
        for t in self.treatments:
            if t.name == channel:
                treatment = t
                break

        if treatment is None:
            raise ValueError(f"Channel '{channel}' not found in treatments")

        # Extract adstock transform
        adstock = treatment.adstock

        if adstock is None:
            print(f"No adstock transform found for channel '{channel}'")
            return None

        # Get alpha parameter from adstock transform
        adstock_params = adstock.get_params()
        alpha = adstock_params.get("alpha")

        if alpha is None:
            raise ValueError(
                f"Adstock transform for channel '{channel}' has alpha=None. "
                "This should not happen if half_life or alpha was provided."
            )

        # Generate IRF (adstock weights)
        if max_lag is None:
            max_lag = adstock_params.get("l_max", 12)

        lags = np.arange(max_lag + 1)
        weights = alpha**lags

        normalize = adstock_params.get("normalize", True)
        if normalize:
            weights = weights / weights.sum()

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(lags, weights, alpha=0.7, color="C0")
        ax.set_xlabel("Lag (periods)")
        ax.set_ylabel("Weight")

        # Calculate half-life
        half_life_calc = np.log(0.5) / np.log(alpha)

        ax.set_title(
            f"Impulse Response Function: {channel}\n"
            f"(alpha={alpha:.3f}, half_life={half_life_calc:.2f}, "
            f"normalize={normalize})"
        )
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig

    def plot_transforms(
        self,
        true_saturation=None,
        true_adstock=None,
        x_range=None,
        **kwargs,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot estimated transformation curves (saturation and/or adstock).

        Creates a figure with 1-2 panels depending on which transforms are present:
        - Saturation curve (input exposure -> saturated exposure) if saturation exists
        - Adstock weights over time (lag distribution) if adstock exists

        Parameters
        ----------
        true_saturation : SaturationTransform, optional
            True saturation transform for comparison (e.g., from simulation).
            If provided, will be overlaid as a dashed line.
        true_adstock : AdstockTransform, optional
            True adstock transform for comparison (e.g., from simulation).
            If provided, will be overlaid as gray bars.
        x_range : tuple of (min, max), optional
            Range for saturation curve x-axis. If None, uses data range.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : list of matplotlib.axes.Axes
            List of axes objects (1 or 2 panels depending on which transforms exist).

        Examples
        --------
        .. code-block:: python

            # Plot estimated transforms only
            fig, ax = result.plot_transforms()

            # Compare to true transforms (simulation study)
            fig, ax = result.plot_transforms(
                true_saturation=HillSaturation(slope=2.0, kappa=50),
                true_adstock=GeometricAdstock(half_life=3.0, normalize=True),
            )
        """
        # Route to appropriate plot method based on model type
        if isinstance(self.model, PyMCModel):
            return self._bayesian_plot_transforms(
                true_saturation=true_saturation,
                true_adstock=true_adstock,
                x_range=x_range,
                **kwargs,
            )
        else:
            return self._ols_plot_transforms(
                true_saturation=true_saturation,
                true_adstock=true_adstock,
                x_range=x_range,
                **kwargs,
            )

    def _ols_plot_transforms(
        self,
        true_saturation=None,
        true_adstock=None,
        x_range=None,
        **kwargs,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot estimated transformation curves for OLS models."""
        # Currently only supports single treatment
        if len(self.treatments) != 1:
            raise NotImplementedError(
                "plot_transforms() currently only supports single treatment analysis"
            )

        treatment = self.treatments[0]
        est_saturation = treatment.saturation
        est_adstock = treatment.adstock

        # Check which transforms exist
        has_saturation = est_saturation is not None
        has_adstock = est_adstock is not None

        if not has_saturation and not has_adstock:
            raise ValueError(
                "No transforms to plot (both saturation and adstock are None). "
                "At least one transform must be specified."
            )

        # Determine number of panels based on available transforms
        n_panels = int(has_saturation) + int(has_adstock)

        # Create subplot with appropriate number of panels
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))

        # Make axes a list for consistent indexing
        if n_panels == 1:
            axes = [axes]

        panel_idx = 0

        # ============================================================================
        # SATURATION PLOT (if present)
        # ============================================================================
        if est_saturation is not None:
            ax = axes[panel_idx]
            # Determine x range
            if x_range is None:
                # Use range from data
                x_raw = np.asarray(self.data[treatment.name].values)
                x_min, x_max = x_raw.min(), x_raw.max()
                # Add some padding
                x_padding = (x_max - x_min) * 0.1
                x_range = (max(0, x_min - x_padding), x_max + x_padding)

            x_sat = np.linspace(x_range[0], x_range[1], 100)

            # Plot true saturation if provided
            if true_saturation is not None:
                y_true_sat = true_saturation.apply(x_sat)
                ax.plot(
                    x_sat,
                    y_true_sat,
                    "k--",
                    linewidth=2.5,
                    label="True",
                    alpha=0.8,
                )

            # Plot estimated saturation
            y_est_sat = est_saturation.apply(x_sat)
            ax.plot(x_sat, y_est_sat, "C0-", linewidth=2.5, label="Estimated")

            ax.set_xlabel(f"{treatment.name} (raw)", fontsize=11)
            ax.set_ylabel("Saturated Value", fontsize=11)
            ax.set_title("Saturation Function", fontsize=12, fontweight="bold")
            ax.legend(fontsize=LEGEND_FONT_SIZE, framealpha=0.9)
            ax.grid(True, alpha=0.3)

            # Add parameter text
            est_params = est_saturation.get_params()
            param_text = "Estimated:\n"
            for key, val in est_params.items():
                if key not in ["alpha", "l_max", "normalize"]:  # Skip adstock params
                    param_text += f"  {key}={val:.2f}\n"

            if true_saturation is not None:
                true_params = true_saturation.get_params()
                param_text += "\nTrue:\n"
                for key, val in true_params.items():
                    if key not in ["alpha", "l_max", "normalize"]:
                        param_text += f"  {key}={val:.2f}\n"

            ax.text(
                0.05,
                0.95,
                param_text.strip(),
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
            )
            panel_idx += 1

        # ============================================================================
        # ADSTOCK PLOT (if present)
        # ============================================================================
        if est_adstock is not None:
            ax = axes[panel_idx]
            est_adstock_params = est_adstock.get_params()
            l_max = est_adstock_params.get("l_max", 12)
            lags = np.arange(l_max + 1)

            # Compute estimated adstock weights
            est_alpha = est_adstock_params["alpha"]
            est_weights = est_alpha**lags
            normalize = est_adstock_params.get("normalize", True)
            if normalize:
                est_weights = est_weights / est_weights.sum()

            # Plot true adstock if provided
            if true_adstock is not None:
                true_adstock_params = true_adstock.get_params()
                true_alpha = true_adstock_params["alpha"]
                true_weights = true_alpha**lags
                if true_adstock_params.get("normalize", True):
                    true_weights = true_weights / true_weights.sum()

                # Line plot instead of bars
                ax.plot(
                    lags,
                    true_weights,
                    "k--",
                    linewidth=2.5,
                    label="True",
                    alpha=0.8,
                )

                # Estimated adstock line
                ax.plot(
                    lags,
                    est_weights,
                    "C0-",
                    linewidth=2.5,
                    label="Estimated",
                    alpha=0.8,
                )
            else:
                # Single line for estimated only
                ax.plot(
                    lags,
                    est_weights,
                    "C0-",
                    linewidth=2.5,
                    label="Estimated",
                    alpha=0.8,
                )

            ax.set_xlabel("Lag (periods)", fontsize=11)
            ax.set_ylabel("Adstock Weight", fontsize=11)
            ax.set_title(
                "Adstock Function (Carryover Effect)", fontsize=12, fontweight="bold"
            )
            ax.legend(fontsize=LEGEND_FONT_SIZE, framealpha=0.9)
            ax.grid(True, alpha=0.3, axis="y")

            # Add parameter text
            param_text = "Estimated:\n"
            half_life_est = np.log(0.5) / np.log(est_alpha)
            param_text += f"  half_life={half_life_est:.2f}\n"
            param_text += f"  alpha={est_alpha:.3f}\n"

            if true_adstock is not None:
                true_alpha = true_adstock_params["alpha"]
                half_life_true = np.log(0.5) / np.log(true_alpha)
                param_text += "\nTrue:\n"
                param_text += f"  half_life={half_life_true:.2f}\n"
                param_text += f"  alpha={true_alpha:.3f}\n"

            ax.text(
                0.95,
                0.95,
                param_text.strip(),
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
            )

        plt.tight_layout()
        return fig, axes

    def plot_diagnostics(self, lags: int = 20) -> None:
        """Display diagnostic plots and tests for model residuals.

        Shows:
        1. ACF (autocorrelation function) plot
        2. PACF (partial autocorrelation function) plot
        3. Ljung-Box test for residual autocorrelation

        Parameters
        ----------
        lags : int, default=20
            Number of lags to display.
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # ACF plot
        plot_acf(self.residuals, lags=lags, ax=axes[0], alpha=0.05)
        axes[0].set_title("Residual Autocorrelation Function (ACF)")

        # PACF plot
        plot_pacf(self.residuals, lags=lags, ax=axes[1], alpha=0.05, method="ywm")
        axes[1].set_title("Residual Partial Autocorrelation Function (PACF)")

        plt.tight_layout()
        plt.show()

        # Ljung-Box test
        lb_result = acorr_ljungbox(self.residuals, lags=lags, return_df=True)

        print("\n" + "=" * 60)
        print("Ljung-Box Test for Residual Autocorrelation")
        print("=" * 60)
        print("H0: Residuals are independently distributed (no autocorrelation)")
        print("If p-value < 0.05, reject H0 (autocorrelation present)")
        print("-" * 60)

        # Show summary for a few key lags
        key_lags = [1, 5, 10, lags]
        for lag in key_lags:
            if lag <= len(lb_result):
                row = lb_result.iloc[lag - 1]
                sig = (
                    "***"
                    if row["lb_pvalue"] < 0.01
                    else ("*" if row["lb_pvalue"] < 0.05 else "")
                )
                print(
                    f"Lag {lag:2d}: LB statistic = {row['lb_stat']:8.3f}, "
                    f"p-value = {row['lb_pvalue']:.4f} {sig}"
                )

        print("-" * 60)
        if lb_result["lb_pvalue"].min() < 0.05:
            print(
                "⚠ Warning: Significant residual autocorrelation detected.\n"
                "  - HAC standard errors (if used) account for this in coefficient inference.\n"
                "  - Consider adding more baseline controls or adjusting transform parameters."
            )
        else:
            print("✓ No significant residual autocorrelation detected.")
        print("=" * 60)

    def summary(self, round_to: int | None = None) -> None:
        """Print a summary of the model results.

        Parameters
        ----------
        round_to : int, optional
            Number of decimal places for rounding.
        """
        import arviz as az

        if round_to is None:
            round_to = 2

        print("=" * 80)
        print("Graded Intervention Time Series Results")
        print("=" * 80)
        print(f"Outcome variable: {self.y_column}")
        print(
            f"Number of observations: {len(self.y) if isinstance(self.y, np.ndarray) else self.y.shape[0]}"
        )

        if isinstance(self.model, PyMCModel):
            # ============================================================
            # BAYESIAN MODEL SUMMARY
            # ============================================================
            print("Model type: Bayesian (PyMC)")

            print("-" * 80)
            print("Transform parameters (Posterior Mean [94% HDI]):")

            assert self.model.idata is not None
            # Extract transform parameters
            if "half_life" in self.model.idata.posterior:
                half_life_post = az.extract(self.model.idata, var_names=["half_life"])
                half_life_mean = float(half_life_post.mean())
                half_life_hdi = az.hdi(
                    self.model.idata, var_names=["half_life"], hdi_prob=0.94
                )["half_life"]
                print(
                    f"  half_life: {round_num(half_life_mean, round_to)} [{round_num(float(half_life_hdi.sel(hdi='lower').values), round_to)}, {round_num(float(half_life_hdi.sel(hdi='higher').values), round_to)}]"
                )

            # Saturation parameters if present
            if "slope" in self.model.idata.posterior:
                slope_post = az.extract(self.model.idata, var_names=["slope"])
                slope_mean = float(slope_post.mean())
                slope_hdi = az.hdi(
                    self.model.idata, var_names=["slope"], hdi_prob=0.94
                )["slope"]
                print(
                    f"  slope: {round_num(slope_mean, round_to)} [{round_num(float(slope_hdi.sel(hdi='lower').values), round_to)}, {round_num(float(slope_hdi.sel(hdi='higher').values), round_to)}]"
                )

            if "kappa" in self.model.idata.posterior:
                kappa_post = az.extract(self.model.idata, var_names=["kappa"])
                kappa_mean = float(kappa_post.mean())
                kappa_hdi = az.hdi(
                    self.model.idata, var_names=["kappa"], hdi_prob=0.94
                )["kappa"]
                print(
                    f"  kappa: {round_num(kappa_mean, round_to)} [{round_num(float(kappa_hdi.sel(hdi='lower').values), round_to)}, {round_num(float(kappa_hdi.sel(hdi='higher').values), round_to)}]"
                )

            print("-" * 80)
            print("Baseline coefficients (Posterior Mean [94% HDI]):")
            beta_post = az.extract(self.model.idata, var_names=["beta"]).squeeze()
            # Compute HDI for all betas at once - HDI uses generic dimension names
            beta_hdi_all = az.hdi(self.model.idata, var_names=["beta"], hdi_prob=0.94)[
                "beta"
            ]

            # Determine the dimension structure of the HDI result
            # For multi-unit case: (treated_units, coeffs, hdi)
            # For single-unit case: (coeffs, hdi) or just (hdi)
            for i, label in enumerate(self.baseline_labels):
                if beta_post.ndim > 1:
                    beta_i = beta_post[0, i]  # First treated unit
                    # HDI for multi-dimensional: select first unit and i-th coefficient
                    if "treated_units" in beta_hdi_all.dims:
                        beta_hdi = beta_hdi_all.isel(treated_units=0, coeffs=i)
                    else:
                        beta_hdi = beta_hdi_all.isel(coeffs=i)
                else:
                    beta_i = beta_post[i]
                    # Single unit case
                    if "coeffs" in beta_hdi_all.dims:
                        beta_hdi = beta_hdi_all.isel(coeffs=i)
                    else:
                        # Scalar case
                        beta_hdi = beta_hdi_all
                beta_mean = float(beta_i.mean())
                print(
                    f"  {label:20s}: {round_num(beta_mean, round_to)} [{round_num(float(beta_hdi.sel(hdi='lower').values), round_to)}, {round_num(float(beta_hdi.sel(hdi='higher').values), round_to)}]"
                )

            print("-" * 80)
            print("Treatment coefficients (Posterior Mean [94% HDI]):")
            theta_post = az.extract(
                self.model.idata, var_names=["theta_treatment"]
            ).squeeze()
            # Compute HDI for all thetas at once - HDI uses generic dimension names
            theta_hdi_all = az.hdi(
                self.model.idata, var_names=["theta_treatment"], hdi_prob=0.94
            )["theta_treatment"]

            for i, label in enumerate(self.treatment_labels):
                if theta_post.ndim > 1:
                    theta_i = theta_post[0, i]  # First treated unit
                    # HDI for multi-dimensional
                    if "treated_units" in theta_hdi_all.dims:
                        theta_hdi = theta_hdi_all.isel(treated_units=0, coeffs=i)
                    else:
                        theta_hdi = theta_hdi_all.isel(coeffs=i)
                else:
                    theta_i = theta_post[i] if theta_post.ndim > 0 else theta_post
                    # Single unit/treatment case
                    if (
                        "coeffs" in theta_hdi_all.dims
                        and theta_hdi_all.sizes["coeffs"] > 1
                    ):
                        theta_hdi = theta_hdi_all.isel(coeffs=i)
                    else:
                        # Scalar or single coefficient case
                        theta_hdi = theta_hdi_all
                theta_mean = float(theta_i.mean())
                print(
                    f"  {label:20s}: {round_num(theta_mean, round_to)} [{round_num(float(theta_hdi.sel(hdi='lower').values), round_to)}, {round_num(float(theta_hdi.sel(hdi='higher').values), round_to)}]"
                )

            print("=" * 80)
        else:
            # ============================================================
            # OLS MODEL SUMMARY
            # ============================================================
            print(f"R-squared: {round_num(self.score, round_to)}")
            print(f"Error model: {self.error_model.upper()}")
            if self.error_model == "hac":
                print(
                    f"  HAC max lags: {self.hac_maxlags} "
                    f"(robust SEs accounting for {self.hac_maxlags} periods of autocorrelation)"
                )
            elif self.error_model == "arimax":
                p, d, q = self.arima_order
                print(f"  ARIMA order: ({p}, {d}, {q})")
                print(f"    p={p}: AR order, d={d}: differencing, q={q}: MA order")
            print("-" * 80)
            print("Baseline coefficients:")
            for label, coef, se in zip(
                self.baseline_labels,
                self.beta_baseline,
                self.ols_result.bse[: len(self.baseline_labels)],
                strict=False,
            ):
                coef_rounded = round_num(coef, round_to)
                se_rounded = round_num(se, round_to)
                print(f"  {label:20s}: {coef_rounded:>10} (SE: {se_rounded})")
            print("-" * 80)
            print("Treatment coefficients:")
            n_baseline = len(self.baseline_labels)

            # For ARIMAX, we need to extract only the treatment SEs from exogenous params
            if self.error_model == "arimax":
                n_exog = self.ols_result.model.k_exog
                treatment_se = self.ols_result.bse[n_baseline:n_exog]
            else:
                treatment_se = self.ols_result.bse[n_baseline:]

            for label, coef, se in zip(
                self.treatment_labels,
                self.theta_treatment,
                treatment_se,
                strict=False,
            ):
                coef_rounded = round_num(coef, round_to)
                se_rounded = round_num(se, round_to)
                print(f"  {label:20s}: {coef_rounded:>10} (SE: {se_rounded})")
            print("=" * 80)

    # Methods required by BaseExperiment
    def _bayesian_plot(
        self, round_to: int | None = 2, **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        """Generate Bayesian-specific plots with credible intervals."""
        import arviz as az

        assert self.model.idata is not None
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Extract posterior predictions (mu is in posterior group as Deterministic)
        mu_posterior = az.extract(self.model.idata, group="posterior", var_names="mu")

        # Get mean and HDI
        mu_mean = mu_posterior.mean(dim="sample").values.flatten()
        # Compute HDI from posterior
        mu_hdi = az.hdi(self.model.idata.posterior, var_names=["mu"], hdi_prob=0.94)[
            "mu"
        ]
        mu_lower = mu_hdi.sel(hdi="lower").values.flatten()
        mu_upper = mu_hdi.sel(hdi="higher").values.flatten()

        # Top panel: Observed vs fitted with credible interval
        ax[0].plot(
            self.data.index,
            self.y.values.flatten() if hasattr(self.y, "values") else self.y,
            "o",
            label="Observed",
            alpha=0.6,
            markersize=4,
        )
        ax[0].plot(
            self.data.index,
            mu_mean,
            "-",
            label="Posterior Mean",
            linewidth=2,
            color="C1",
        )
        ax[0].fill_between(
            self.data.index,
            mu_lower,
            mu_upper,
            alpha=0.3,
            color="C1",
            label="94% HDI",
        )
        ax[0].set_ylabel("Outcome")
        ax[0].set_title("Bayesian Model Fit")
        ax[0].legend(fontsize=LEGEND_FONT_SIZE)
        ax[0].grid(True, alpha=0.3)

        # Bottom panel: Residuals with uncertainty
        y_obs = np.asarray(
            self.y.values.flatten() if hasattr(self.y, "values") else self.y
        )
        residuals = y_obs - mu_mean
        ax[1].plot(self.data.index, residuals, "o-", alpha=0.6, markersize=3)
        ax[1].axhline(y=0, color="k", linestyle="--", linewidth=1)
        ax[1].set_ylabel("Residuals")
        ax[1].set_xlabel("Time")
        ax[1].set_title("Model Residuals")
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def get_plot_data_bayesian(self, *args, **kwargs):
        """Get plot data for Bayesian results.

        Returns
        -------
        pd.DataFrame
            DataFrame with observed and posterior mean fitted values.
        """
        import arviz as az

        # Extract posterior predictions
        mu_posterior = az.extract(
            self.model.idata, group="posterior_predictive", var_names="mu"
        )
        mu_mean = mu_posterior.mean(dim="sample").values.flatten()

        y_obs = np.asarray(
            self.y.values.flatten() if hasattr(self.y, "values") else self.y
        )

        return pd.DataFrame(
            {
                "observed": y_obs,
                "fitted": mu_mean,
                "residuals": y_obs - mu_mean,
            },
            index=self.data.index,
        )

    def _bayesian_plot_transforms(
        self,
        true_saturation=None,
        true_adstock=None,
        x_range=None,
        **kwargs,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot estimated transformation curves for Bayesian models with credible intervals."""
        import arviz as az

        assert self.model.idata is not None
        # Check which transforms are present in the posterior
        has_saturation = (
            "slope" in self.model.idata.posterior
            or "kappa" in self.model.idata.posterior
        )
        has_adstock = "half_life" in self.model.idata.posterior

        if not has_saturation and not has_adstock:
            raise ValueError(
                "No transforms to plot (no transform parameters found in posterior). "
                "At least one transform must be specified."
            )

        # Determine number of panels
        n_panels = int(has_saturation) + int(has_adstock)

        # Create subplot
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))

        # Make axes a list for consistent indexing
        if n_panels == 1:
            axes = [axes]

        panel_idx = 0

        # ============================================================================
        # SATURATION PLOT (if present)
        # ============================================================================
        if has_saturation:
            ax = axes[panel_idx]
            # TODO: Implement Bayesian saturation plotting when needed
            # For now, skip saturation (since the current example uses adstock only)
            ax.text(
                0.5,
                0.5,
                "Bayesian saturation\nplotting not yet\nimplemented",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            panel_idx += 1

        # ============================================================================
        # ADSTOCK PLOT (if present)
        # ============================================================================
        if has_adstock:
            ax = axes[panel_idx]

            # Extract posterior samples of half_life
            half_life_post = az.extract(self.model.idata, var_names=["half_life"])
            l_max = self.model.adstock_config.get("l_max", 8)
            lags = np.arange(l_max + 1)

            # Compute adstock weights for all posterior samples
            weights_list = []
            for half_life_sample in half_life_post.values:
                alpha_sample = np.power(0.5, 1 / half_life_sample)
                weights_sample = alpha_sample**lags
                # Normalize if needed
                if self.model.adstock_config.get("normalize", True):
                    weights_sample = weights_sample / weights_sample.sum()
                weights_list.append(weights_sample)
            weights_posterior = np.array(weights_list)  # (n_samples, n_lags)

            # Compute 94% credible interval bounds at each lag
            weights_lower = np.percentile(weights_posterior, 3, axis=0)
            weights_upper = np.percentile(weights_posterior, 97, axis=0)

            # Compute posterior mean
            half_life_mean = float(half_life_post.mean())
            alpha_mean = np.power(0.5, 1 / half_life_mean)
            weights_mean = alpha_mean**lags
            if self.model.adstock_config.get("normalize", True):
                weights_mean = weights_mean / weights_mean.sum()

            # Plot true adstock if provided
            if true_adstock is not None:
                true_adstock_params = true_adstock.get_params()
                true_alpha = true_adstock_params["alpha"]
                true_weights = true_alpha**lags
                if true_adstock_params.get("normalize", True):
                    true_weights = true_weights / true_weights.sum()

                ax.plot(
                    lags,
                    true_weights,
                    "k--",
                    linewidth=2.5,
                    label="True",
                    alpha=0.8,
                    zorder=10,
                )

            # Plot Bayesian uncertainty as shaded region
            ax.fill_between(
                lags,
                weights_lower,
                weights_upper,
                color="C2",
                alpha=0.25,
                label="Bayesian 94% HDI",
                zorder=1,
            )

            # Plot Bayesian posterior mean
            ax.plot(
                lags,
                weights_mean,
                "C2-",
                linewidth=3,
                label="Bayesian Mean",
                alpha=1.0,
                zorder=11,
            )

            ax.set_xlabel("Lag (periods)", fontsize=11)
            ax.set_ylabel("Adstock Weight", fontsize=11)
            ax.set_title(
                "Adstock Function (Carryover Effect)", fontsize=12, fontweight="bold"
            )
            ax.legend(fontsize=LEGEND_FONT_SIZE, framealpha=0.9)
            ax.grid(True, alpha=0.3, axis="y")

            # Add parameter text with posterior summary
            param_text = "Bayesian Posterior:\n"
            half_life_hdi = az.hdi(
                self.model.idata, var_names=["half_life"], hdi_prob=0.94
            )["half_life"]
            param_text += f"  half_life={half_life_mean:.2f}\n"
            param_text += f"    [94% HDI: {float(half_life_hdi.sel(hdi='lower').values):.2f}, {float(half_life_hdi.sel(hdi='higher').values):.2f}]\n"

            if true_adstock is not None:
                true_half_life = np.log(0.5) / np.log(true_alpha)
                param_text += "\nTrue:\n"
                param_text += f"  half_life={true_half_life:.2f}\n"

            ax.text(
                0.95,
                0.95,
                param_text.strip(),
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
            )

        plt.tight_layout()
        return fig, axes

    def _bayesian_effect(
        self,
        window: tuple[pd.Timestamp | int, pd.Timestamp | int],
        channels: list[str] | None = None,
        scale: float = 0.0,
    ) -> dict[str, Any]:
        """Estimate the causal effect for Bayesian models with posterior uncertainty."""
        import arviz as az

        assert self.model.idata is not None
        # Default to all channels if not specified
        if channels is None:
            channels = self.treatment_labels

        # Validate channels
        for ch in channels:
            if ch not in self.treatment_labels:
                raise ValueError(f"Channel '{ch}' not found in treatments")

        # Get window mask
        window_start, window_end = window
        if isinstance(self.data.index, pd.DatetimeIndex):
            mask = (
                self.data.index >= window_start  # type: ignore[operator]
            ) & (
                self.data.index <= window_end  # type: ignore[operator]
            )
        else:
            mask = (self.data.index >= window_start) & (self.data.index <= window_end)

        # Create counterfactual data by scaling specified channels in the window
        data_cf = self.data.copy()
        for channel in channels:
            data_cf.loc[mask, channel] = scale * data_cf.loc[mask, channel]

        # Get counterfactual treatment data
        treatment_raw_cf = np.column_stack(
            [np.asarray(data_cf[name].values) for name in self.treatment_labels]
        )

        # Extract posterior samples
        beta_post = az.extract(
            self.model.idata, var_names=["beta"]
        )  # (sample, units, features)
        theta_post = az.extract(
            self.model.idata, var_names=["theta_treatment"]
        )  # (sample, units, treatments)

        # Extract transform parameter samples
        n_samples = len(beta_post.sample)
        y_cf_samples = []

        # For each posterior sample, compute counterfactual prediction
        for i in range(n_samples):
            # Get parameter values for this sample
            if "half_life" in self.model.idata.posterior:
                half_life_sample = float(
                    az.extract(self.model.idata, var_names=["half_life"]).isel(sample=i)
                )
            else:
                half_life_sample = None

            # Apply transforms with posterior sample parameters
            treatment_transformed_cf = treatment_raw_cf.copy().astype(np.float64)
            if half_life_sample is not None:
                # Apply adstock with this sample's half_life (numpy implementation)
                alpha_sample = np.power(0.5, 1 / half_life_sample)
                l_max = self.model.adstock_config.get("l_max", 8)
                normalize = self.model.adstock_config.get("normalize", True)

                # Compute adstock weights
                lags = np.arange(l_max + 1)
                weights = alpha_sample**lags
                if normalize:
                    weights = weights / weights.sum()

                # Apply convolution for each treatment column
                treatment_transformed_cf = np.zeros_like(
                    treatment_raw_cf, dtype=np.float64
                )
                for col_idx in range(treatment_raw_cf.shape[1]):
                    treatment_transformed_cf[:, col_idx] = np.convolve(
                        treatment_raw_cf[:, col_idx], weights, mode="same"
                    )

            # Get coefficients for this sample (first unit only for single-unit case)
            if beta_post.ndim > 2:
                beta_sample = beta_post.isel(sample=i, treated_units=0).values.astype(
                    np.float64
                )
            else:
                beta_sample = beta_post.isel(sample=i).values.astype(np.float64)

            if theta_post.ndim > 2:
                theta_sample = theta_post.isel(sample=i, treated_units=0).values.astype(
                    np.float64
                )
            else:
                theta_sample = theta_post.isel(sample=i).values.astype(np.float64)

            # Compute counterfactual prediction (ensure float64)
            baseline_pred = np.asarray(self.X.values, dtype=np.float64) @ beta_sample
            treatment_pred = treatment_transformed_cf @ theta_sample
            y_cf_sample = baseline_pred + treatment_pred
            y_cf_samples.append(y_cf_sample.flatten())

        y_cf_arr = np.array(y_cf_samples)  # Shape: (n_samples, n_obs)

        # Compute posterior mean and HDI
        y_cf_mean = y_cf_arr.mean(axis=0)
        y_cf_lower = np.percentile(y_cf_arr, 3, axis=0)
        y_cf_upper = np.percentile(y_cf_arr, 97, axis=0)

        # Get observed data
        y_obs = np.asarray(
            self.y.values.flatten() if hasattr(self.y, "values") else self.y
        )

        # Compute effect
        effect_samples = y_obs[np.newaxis, :] - y_cf_arr  # Shape: (n_samples, n_obs)
        effect_mean = effect_samples.mean(axis=0)
        effect_lower = np.percentile(effect_samples, 3, axis=0)
        effect_upper = np.percentile(effect_samples, 97, axis=0)

        # Cumulative effect
        effect_cumulative_samples = np.cumsum(effect_samples, axis=1)
        effect_cumulative_mean = effect_cumulative_samples.mean(axis=0)
        effect_cumulative_lower = np.percentile(effect_cumulative_samples, 3, axis=0)
        effect_cumulative_upper = np.percentile(effect_cumulative_samples, 97, axis=0)

        # Create result DataFrame
        effect_df = pd.DataFrame(
            {
                "observed": y_obs,
                "counterfactual": y_cf_mean,
                "counterfactual_lower": y_cf_lower,
                "counterfactual_upper": y_cf_upper,
                "effect": effect_mean,
                "effect_lower": effect_lower,
                "effect_upper": effect_upper,
                "effect_cumulative": effect_cumulative_mean,
                "effect_cumulative_lower": effect_cumulative_lower,
                "effect_cumulative_upper": effect_cumulative_upper,
            },
            index=self.data.index,
        )

        # Filter to window for summary statistics
        window_effect_samples = effect_samples[:, mask]

        result = {
            "effect_df": effect_df,
            "total_effect": float(window_effect_samples.sum(axis=1).mean()),
            "total_effect_lower": float(
                np.percentile(window_effect_samples.sum(axis=1), 3)
            ),
            "total_effect_upper": float(
                np.percentile(window_effect_samples.sum(axis=1), 97)
            ),
            "mean_effect": float(window_effect_samples.mean(axis=1).mean()),
            "window_start": window_start,
            "window_end": window_end,
            "channels": channels,
            "scale": scale,
        }

        return result

    def _bayesian_plot_effect(
        self,
        effect_result: dict,
        **kwargs,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot counterfactual effect analysis for Bayesian models with credible intervals."""
        # Extract data from effect result
        effect_df = effect_result["effect_df"]
        window_start = effect_result.get("window_start")
        window_end = effect_result.get("window_end")

        # Create 2-panel subplot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # ============================================================================
        # TOP PANEL: Observed vs Counterfactual with credible intervals
        # ============================================================================
        # Plot counterfactual with uncertainty
        axes[0].plot(
            effect_df.index,
            effect_df["counterfactual"],
            label="Counterfactual (Posterior Mean)",
            linewidth=1.5,
            linestyle="--",
            color="C1",
        )
        axes[0].fill_between(
            effect_df.index,
            effect_df["counterfactual_lower"],
            effect_df["counterfactual_upper"],
            alpha=0.2,
            color="C1",
            label="Counterfactual 94% HDI",
        )

        # Plot observed
        axes[0].plot(
            effect_df.index,
            effect_df["observed"],
            label="Observed",
            linewidth=1.5,
            color="C0",
        )

        # Shade the effect region
        axes[0].fill_between(
            effect_df.index,
            effect_df["observed"],
            effect_df["counterfactual"],
            alpha=0.3,
            color="C2",
            label="Effect",
        )

        axes[0].set_ylabel(self.y_column, fontsize=11)
        axes[0].set_title("Observed vs Counterfactual", fontsize=12, fontweight="bold")
        axes[0].legend(fontsize=LEGEND_FONT_SIZE)
        axes[0].grid(True, alpha=0.3)

        # Add window boundaries if specified
        if window_start is not None and window_end is not None:
            axes[0].axvline(x=window_start, color="red", linestyle=":", alpha=0.5)
            axes[0].axvline(x=window_end, color="red", linestyle=":", alpha=0.5)

        # ============================================================================
        # BOTTOM PANEL: Cumulative effect with credible intervals
        # ============================================================================
        axes[1].plot(
            effect_df.index,
            effect_df["effect_cumulative"],
            linewidth=2,
            color="C2",
            label="Cumulative Effect (Posterior Mean)",
        )
        axes[1].fill_between(
            effect_df.index,
            effect_df["effect_cumulative_lower"],
            effect_df["effect_cumulative_upper"],
            alpha=0.3,
            color="C2",
            label="94% HDI",
        )
        axes[1].axhline(y=0, color="k", linestyle="--", linewidth=1)
        axes[1].set_ylabel("Cumulative Effect", fontsize=11)
        axes[1].set_xlabel("Time", fontsize=11)
        axes[1].set_title("Cumulative Effect Over Time", fontsize=12, fontweight="bold")
        axes[1].legend(fontsize=LEGEND_FONT_SIZE)
        axes[1].grid(True, alpha=0.3)

        # Add window boundaries
        if window_start is not None and window_end is not None:
            axes[1].axvline(x=window_start, color="red", linestyle=":", alpha=0.5)
            axes[1].axvline(x=window_end, color="red", linestyle=":", alpha=0.5)

        plt.tight_layout()
        return fig, axes

    def get_plot_data_ols(self) -> pd.DataFrame:
        """Get plot data for OLS results.

        Returns
        -------
        pd.DataFrame
            DataFrame with observed, fitted, and residual values.
        """
        return pd.DataFrame(
            {
                "observed": self.y,
                "fitted": self.predictions,
                "residuals": self.residuals,
            },
            index=self.data.index,
        )
