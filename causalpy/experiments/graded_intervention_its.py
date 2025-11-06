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
Graded Intervention Time Series Experiment

This module implements experiments for estimating the causal effects of graded
interventions (e.g., media spend, policy intensity) in single-market time series
using transfer functions that model saturation and adstock (carryover) effects.

The experiment works with the TransferFunctionOLS model class (from skl_models)
to provide a complete causal inference workflow including visualization,
diagnostics, and counterfactual effect estimation.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.base import RegressorMixin
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from causalpy.custom_exceptions import BadIndexException
from causalpy.transforms import Treatment
from causalpy.utils import round_num

from .base import BaseExperiment

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
    1. Create an UNFITTED TransferFunctionOLS model with configuration
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
    combinations and M adstock parameter combinations, all N × M combinations are
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
    supports_bayes = False  # Future extension

    def __init__(
        self,
        data: pd.DataFrame,
        y_column: str,
        treatment_names: List[str],
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
        self.y = data[y_column].values

        # Build baseline design matrix (like other experiments do)
        self.X_baseline = np.asarray(dmatrix(base_formula, data))
        self.baseline_labels = dmatrix(base_formula, data).design_info.column_names

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
            x_raw = data[name].values
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
        if isinstance(self.model, RegressorMixin):
            self.model.fit(X=self.X_full, y=self.y)
        else:
            raise ValueError("Model type not recognized")

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

    def _validate_inputs(
        self,
        data: pd.DataFrame,
        y_column: str,
        treatment_names: List[str],
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
        self, data: pd.DataFrame, treatments: List[Treatment]
    ) -> Tuple[np.ndarray, List[str]]:
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
            x_raw = data[treatment.name].values

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

    def effect(
        self,
        window: Tuple[Union[pd.Timestamp, int], Union[pd.Timestamp, int]],
        channels: Optional[List[str]] = None,
        scale: float = 0.0,
    ) -> Dict[str, Union[pd.DataFrame, float]]:
        """Estimate the causal effect of scaling treatment channels in a time window.

        This method computes a counterfactual scenario by scaling the specified
        treatment channels in the given window, reapplying all transforms with
        the same parameters, and comparing to the observed outcome.

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
            - "total_effect": Total effect in window
            - "mean_effect": Mean effect per period in window

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
            mask = (self.data.index >= window_start) & (self.data.index <= window_end)
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
        effect_result: Dict,
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot counterfactual effect analysis results.

        Creates a 2-panel figure showing:
        1. Observed vs counterfactual outcome
        2. Cumulative effect over time

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

    def plot(
        self, round_to: Optional[int] = 2, **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
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
        return self._ols_plot(round_to=round_to, **kwargs)

    def _ols_plot(
        self, round_to: Optional[int] = 2, **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
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

    def plot_irf(self, channel: str, max_lag: Optional[int] = None) -> plt.Figure:
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
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot estimated saturation and adstock transformation curves.

        Creates a 2-panel figure showing:
        1. Saturation curve (input exposure -> saturated exposure)
        2. Adstock weights over time (lag distribution)

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
        ax : array of matplotlib.axes.Axes
            Array of 2 axes objects (left: saturation, right: adstock).

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
        # Currently only supports single treatment
        if len(self.treatments) != 1:
            raise NotImplementedError(
                "plot_transforms() currently only supports single treatment analysis"
            )

        treatment = self.treatments[0]
        est_saturation = treatment.saturation
        est_adstock = treatment.adstock

        # Create 2-panel subplot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ============================================================================
        # LEFT PLOT: Saturation curves
        # ============================================================================
        if est_saturation is not None:
            # Determine x range
            if x_range is None:
                # Use range from data
                x_raw = self.data[treatment.name].values
                x_min, x_max = x_raw.min(), x_raw.max()
                # Add some padding
                x_padding = (x_max - x_min) * 0.1
                x_range = (max(0, x_min - x_padding), x_max + x_padding)

            x_sat = np.linspace(x_range[0], x_range[1], 100)

            # Plot true saturation if provided
            if true_saturation is not None:
                y_true_sat = true_saturation.apply(x_sat)
                axes[0].plot(
                    x_sat,
                    y_true_sat,
                    "k--",
                    linewidth=2.5,
                    label="True",
                    alpha=0.8,
                )

            # Plot estimated saturation
            y_est_sat = est_saturation.apply(x_sat)
            axes[0].plot(x_sat, y_est_sat, "C0-", linewidth=2.5, label="Estimated")

            axes[0].set_xlabel(f"{treatment.name} (raw)", fontsize=11)
            axes[0].set_ylabel("Saturated Value", fontsize=11)
            axes[0].set_title("Saturation Function", fontsize=12, fontweight="bold")
            axes[0].legend(fontsize=LEGEND_FONT_SIZE, framealpha=0.9)
            axes[0].grid(True, alpha=0.3)

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

            axes[0].text(
                0.05,
                0.95,
                param_text.strip(),
                transform=axes[0].transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        else:
            axes[0].text(
                0.5,
                0.5,
                "No saturation transform",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
            axes[0].set_title("Saturation Function", fontsize=12, fontweight="bold")

        # ============================================================================
        # RIGHT PLOT: Adstock weights
        # ============================================================================
        if est_adstock is not None:
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

                width = 0.35
                axes[1].bar(
                    lags - width / 2,
                    true_weights,
                    width,
                    alpha=0.8,
                    label="True",
                    color="gray",
                )
                axes[1].bar(
                    lags + width / 2,
                    est_weights,
                    width,
                    alpha=0.8,
                    label="Estimated",
                    color="C0",
                )
            else:
                axes[1].bar(lags, est_weights, alpha=0.7, color="C0", label="Estimated")

            axes[1].set_xlabel("Lag (periods)", fontsize=11)
            axes[1].set_ylabel("Adstock Weight", fontsize=11)
            axes[1].set_title(
                "Adstock Function (Carryover Effect)", fontsize=12, fontweight="bold"
            )
            axes[1].legend(fontsize=LEGEND_FONT_SIZE, framealpha=0.9)
            axes[1].grid(True, alpha=0.3, axis="y")

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

            axes[1].text(
                0.95,
                0.95,
                param_text.strip(),
                transform=axes[1].transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        else:
            axes[1].text(
                0.5,
                0.5,
                "No adstock transform",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_title(
                "Adstock Function (Carryover Effect)", fontsize=12, fontweight="bold"
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

    def summary(self, round_to: Optional[int] = None) -> None:
        """Print a summary of the model results.

        Parameters
        ----------
        round_to : int, optional
            Number of decimal places for rounding.
        """
        if round_to is None:
            round_to = 2

        print("=" * 80)
        print("Graded Intervention Time Series Results")
        print("=" * 80)
        print(f"Outcome variable: {self.y_column}")
        print(f"Number of observations: {len(self.y)}")
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
        ):
            coef_rounded = round_num(coef, round_to)
            se_rounded = round_num(se, round_to)
            print(f"  {label:20s}: {coef_rounded:>10} (SE: {se_rounded})")
        print("=" * 80)

    # Methods required by BaseExperiment
    def _bayesian_plot(self, *args, **kwargs):
        """Bayesian plotting not yet implemented."""
        raise NotImplementedError("Bayesian inference not yet supported")

    def get_plot_data_bayesian(self, *args, **kwargs):
        """Bayesian plot data not yet implemented."""
        raise NotImplementedError("Bayesian inference not yet supported")

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
