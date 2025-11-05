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
    It works with a pre-fitted TransferFunctionOLS model to provide visualization,
    diagnostics, and counterfactual effect estimation.

    Typical workflow:
    1. Create and fit a TransferFunctionOLS model using the with_estimated_transforms() method
    2. Pass the fitted model to this experiment class
    3. Use experiment methods for visualization and effect estimation

    Parameters
    ----------
    data : pd.DataFrame
        Time series data with datetime or numeric index.
    y_column : str
        Name of the outcome variable column.
    treatment_name : str
        Name of the treatment variable column.
    base_formula : str
        Patsy formula for baseline model.
    treatments : List[Treatment]
        List of Treatment objects with configured transforms.
    model : TransferFunctionOLS
        Pre-fitted model instance.

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
    >>> import causalpy as cp
    >>> # Step 1: Create and fit model
    >>> model = cp.skl_models.TransferFunctionOLS.with_estimated_transforms(
    ...     data=df,
    ...     y_column="water_consumption",
    ...     treatment_name="comm_intensity",
    ...     base_formula="1 + t + temperature + rainfall",
    ...     estimation_method="grid",
    ...     saturation_grid={"slope": [1.0, 2.0, 3.0], "kappa": [3, 5, 7]},
    ...     adstock_grid={"half_life": [2, 3, 4, 5]},
    ...     error_model="hac",
    ... )
    >>> # Step 2: Create experiment with fitted model
    >>> result = cp.GradedInterventionTimeSeries(
    ...     data=df,
    ...     y_column="water_consumption",
    ...     treatment_name="comm_intensity",
    ...     base_formula="1 + t + temperature + rainfall",
    ...     treatments=model.treatments,
    ...     model=model,
    ... )
    >>> # Step 3: Use experiment methods
    >>> result.summary()
    >>> result.plot()
    >>> result.diagnostics()
    >>> effect = result.effect(window=(df.index[0], df.index[-1]), scale=0.0)
    """

    expt_type = "Graded Intervention Time Series"
    supports_ols = True
    supports_bayes = False  # Future extension

    def __init__(
        self,
        data: pd.DataFrame,
        y_column: str,
        treatment_name: str,
        base_formula: str,
        treatments: List[Treatment],
        model=None,
        **kwargs,
    ):
        """
        Initialize experiment with pre-configured treatments and fitted model.

        The model should be a fitted TransferFunctionOLS instance. For most use cases,
        create the model using TransferFunctionOLS.with_estimated_transforms() first.
        """
        super().__init__(model=model)

        # Validate model
        if model is None:
            raise ValueError(
                "A fitted model is required. Use TransferFunctionOLS.with_estimated_transforms() "
                "to create and fit a model, then pass it to this experiment class."
            )

        # Validate inputs
        self._validate_inputs(data, y_column, treatments)

        # Store attributes
        self.data = data.copy()
        self.y_column = y_column
        self.treatment_name = treatment_name  # Store for backwards compatibility
        self.base_formula = base_formula
        self.treatments = treatments
        self.treatment_names = [t.name for t in treatments]

        # Extract outcome variable
        self.y = data[y_column].values

        # Build baseline design matrix
        self.X_baseline = np.asarray(dmatrix(base_formula, data))
        self.baseline_labels = dmatrix(base_formula, data).design_info.column_names

        # Build treatment design matrix
        self.Z_treatment, self.treatment_labels = self._build_treatment_matrix(
            data, treatments
        )

        # Combine matrices
        self.X_full = np.column_stack([self.X_baseline, self.Z_treatment])
        self.all_labels = self.baseline_labels + self.treatment_labels

        # Extract information from fitted model
        self.model = model
        self.ols_result = model.ols_result
        self.predictions = model.ols_result.fittedvalues
        self.residuals = model.ols_result.resid
        self.score = model.score

        # Extract coefficients (handling ARIMAX correctly)
        if hasattr(model, "error_model") and model.error_model == "arimax":
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
        self.error_model = getattr(model, "error_model", "hac")
        self.hac_maxlags = getattr(model, "hac_maxlags", None)
        self.arima_order = getattr(model, "arima_order", None)
        self.transform_estimation_method = getattr(
            model, "transform_estimation_method", None
        )
        self.transform_estimation_results = getattr(
            model, "transform_estimation_results", None
        )
        self.transform_search_space = getattr(model, "transform_search_space", None)

    def _validate_inputs(
        self,
        data: pd.DataFrame,
        y_column: str,
        treatments: List[Treatment],
    ) -> None:
        """Validate input data and parameters."""
        # Check that y_column exists
        if y_column not in data.columns:
            raise ValueError(f"y_column '{y_column}' not found in data columns")

        # Check that treatment columns exist
        for treatment in treatments:
            if treatment.name not in data.columns:
                raise ValueError(
                    f"Treatment column '{treatment.name}' not found in data columns"
                )

        # Check for missing values in outcome
        if data[y_column].isna().any():
            raise ValueError("Outcome variable contains missing values")

        # Warn about missing values in treatment columns
        for treatment in treatments:
            if data[treatment.name].isna().any():
                print(
                    f"Warning: Treatment column '{treatment.name}' contains missing values. "
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
        >>> # Estimate effect of removing treatment completely
        >>> effect = result.effect(
        ...     window=(df.index[0], df.index[-1]),
        ...     channels=["comm_intensity"],
        ...     scale=0.0,
        ... )
        >>> print(f"Total effect: {effect['total_effect']:.2f}")
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
        >>> result.plot_irf("comm_intensity", max_lag=12)
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

    def diagnostics(self, lags: int = 20) -> None:
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
