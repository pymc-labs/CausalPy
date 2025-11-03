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
Transfer Function Interrupted Time Series Analysis

This module implements Transfer-Function ITS for estimating the causal effects
of graded interventions in single-market time series using saturation and
adstock transforms.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from causalpy.custom_exceptions import BadIndexException
from causalpy.transforms import Adstock, Treatment, apply_treatment_transforms
from causalpy.utils import round_num

from .base import BaseExperiment

LEGEND_FONT_SIZE = 12


class TransferFunctionITS(BaseExperiment):
    """
    Transfer Function Interrupted Time Series experiment class.

    This experiment estimates the causal effect of graded interventions (e.g., media
    spend, policy intensity) in a single market using transfer functions that model
    saturation and adstock (carryover) effects.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data with datetime or numeric index. Must contain the outcome
        variable and treatment exposure columns.
    y_column : str
        Name of the outcome variable column in data.
    base_formula : str
        Patsy formula for the baseline model (trend, seasonality, controls).
        Example: "1 + t + np.sin(2*np.pi*t/52) + np.cos(2*np.pi*t/52)"
        where t is a time index. FUTURE: Custom helpers like trend(),
        season_fourier(), holidays() can be added.
    treatments : List[Treatment]
        List of Treatment objects specifying channels and their transforms.
    hac_maxlags : int, optional
        Maximum lags for Newey-West HAC covariance estimation. Default is
        int(4 * (n / 100) ** (2/9)) as suggested by Newey & West.
    model : None
        Not used in MVP (OLS only), but parameter kept for future Bayesian
        extension compatibility with CausalPy architecture.

    Attributes
    ----------
    data : pd.DataFrame
        Input data with time index.
    y : np.ndarray
        Outcome variable values.
    y_column : str
        Name of outcome variable.
    base_formula : str
        Baseline model formula.
    treatments : List[Treatment]
        Treatment specifications.
    ols_result : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model with HAC standard errors.
    beta_baseline : np.ndarray
        Baseline model coefficients.
    theta_treatment : np.ndarray
        Treatment effect coefficients.
    predictions : np.ndarray
        Fitted values.
    residuals : np.ndarray
        Model residuals.

    Examples
    --------
    >>> import causalpy as cp
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample data
    >>> dates = pd.date_range("2020-01-01", periods=104, freq="W")
    >>> df = pd.DataFrame(
    ...     {
    ...         "date": dates,
    ...         "sales": np.random.normal(1000, 100, 104),
    ...         "tv_spend": np.random.uniform(0, 10000, 104),
    ...     }
    ... )
    >>> df = df.set_index("date")
    >>> # Add time index for formula
    >>> df["t"] = np.arange(len(df))
    >>> # Define treatment with saturation and adstock
    >>> treatment = cp.Treatment(
    ...     name="tv_spend",
    ...     transforms=[
    ...         cp.Saturation(kind="hill", slope=2.0, kappa=5000),
    ...         cp.Adstock(half_life=3, normalize=True),
    ...     ],
    ... )
    >>> # Fit model
    >>> result = cp.TransferFunctionITS(
    ...     data=df,
    ...     y_column="sales",
    ...     base_formula="1 + t",
    ...     treatments=[treatment],
    ...     hac_maxlags=8,
    ... )
    >>> # Estimate effect of zeroing TV spend in weeks 50-60
    >>> effect_result = result.effect(
    ...     window=(df.index[50], df.index[60]), channels=["tv_spend"], scale=0.0
    ... )
    >>> # Plot results
    >>> result.plot()
    >>> # Show diagnostics
    >>> result.diagnostics()

    Notes
    -----
    **MVP Limitations:**
    - OLS with HAC standard errors only (no Bayesian inference)
    - Point estimates only (no bootstrap uncertainty intervals)
    - Fixed transform parameters (no grid search)
    - Basic diagnostics only

    **Future Extensions:**
    - Grid search for optimal transform parameters (estimate_transforms=True)
    - Bootstrap or asymptotic confidence intervals for effects
    - Additional error models (GLSAR, ARIMAX)
    - Bayesian inference via PyMC model (reusing transform pipeline)
    - Custom formula helpers (trend(), season_fourier(), holidays())
    - Multi-channel collinearity diagnostics
    - Placebo tests and boundary sensitivity analysis

    The architecture is designed to support future Bayesian extension by:
    - Using the same transform pipeline for both OLS and future PyMC models
    - Following CausalPy's model dispatch pattern (supports_ols, supports_bayes)
    - Storing transforms separately from estimation
    """

    supports_ols = True
    supports_bayes = False  # FUTURE: Will be True when PyMC model is implemented

    def __init__(
        self,
        data: pd.DataFrame,
        y_column: str,
        base_formula: str,
        treatments: List[Treatment],
        hac_maxlags: Optional[int] = None,
        model=None,
        **kwargs,
    ) -> None:
        """Initialize and fit the Transfer Function ITS model."""
        # For MVP, we only support OLS. The model parameter is kept for future
        # compatibility with CausalPy's architecture.
        if model is not None:
            raise NotImplementedError(
                "Custom models not yet supported. MVP uses OLS with HAC standard errors only."
            )

        # Validate inputs
        self._validate_inputs(data, y_column, base_formula, treatments)

        # Store attributes
        self.data = data.copy()
        self.y_column = y_column
        self.base_formula = base_formula
        self.treatments = treatments
        self.treatment_names = [t.name for t in treatments]

        # Extract outcome variable
        self.y = data[y_column].values

        # Build baseline design matrix from formula
        # FUTURE: Add custom formula helpers like trend(), season_fourier(), holidays()
        self.X_baseline = np.asarray(dmatrix(base_formula, data))
        self.baseline_labels = dmatrix(base_formula, data).design_info.column_names

        # Build treatment design matrix by applying transforms
        self.Z_treatment, self.treatment_labels = self._build_treatment_matrix(
            data, treatments
        )

        # Combine baseline and treatment matrices
        self.X_full = np.column_stack([self.X_baseline, self.Z_treatment])
        self.all_labels = self.baseline_labels + self.treatment_labels

        # Fit OLS with HAC standard errors
        if hac_maxlags is None:
            # Newey & West (1994) rule of thumb
            n = len(self.y)
            hac_maxlags = int(np.floor(4 * (n / 100) ** (2 / 9)))

        self.hac_maxlags = hac_maxlags

        # Fit the model
        self.ols_result = sm.OLS(self.y, self.X_full).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_maxlags}
        )

        # Extract coefficients
        n_baseline = self.X_baseline.shape[1]
        self.beta_baseline = self.ols_result.params[:n_baseline]
        self.theta_treatment = self.ols_result.params[n_baseline:]

        # Store predictions and residuals
        self.predictions = self.ols_result.fittedvalues
        self.residuals = self.ols_result.resid

        # Store score (R-squared)
        self.score = self.ols_result.rsquared

    def _validate_inputs(
        self,
        data: pd.DataFrame,
        y_column: str,
        base_formula: str,
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
        # Note: pd.Int64Index was removed in pandas 2.0+, now it's just pd.Index with int64 dtype
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

        Notes
        -----
        FUTURE: When implementing grid search for transform parameters, this
        method will be called repeatedly with different parameter settings.
        """
        Z_columns = []
        labels = []

        for treatment in treatments:
            # Get raw exposure series
            x_raw = data[treatment.name].values

            # Apply transform pipeline
            x_transformed = apply_treatment_transforms(x_raw, treatment)

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
            Start and end of the effect window (inclusive). Can be datetime
            timestamps (for DatetimeIndex) or integers (for numeric index).
        channels : List[str], optional
            List of treatment channel names to scale. If None, scales all channels.
        scale : float, default=0.0
            Scaling factor for the counterfactual. 0.0 means zero out the channels
            (estimate total effect of removing treatment). 0.5 means reduce by 50%, etc.

        Returns
        -------
        result : Dict
            Dictionary containing:
            - "effect_df": pd.DataFrame with columns:
                - "observed": Observed outcome
                - "counterfactual": Counterfactual prediction
                - "effect": Observed - counterfactual (causal impact)
                - "effect_cumulative": Cumulative sum of effect
            - "total_effect": float, sum of effects in window
            - "mean_effect": float, mean effect per period in window

        Examples
        --------
        >>> # Estimate effect of completely removing TV spend in weeks 50-60
        >>> effect = result.effect(
        ...     window=(df.index[50], df.index[60]), channels=["tv_spend"], scale=0.0
        ... )
        >>> print(f"Total effect: {effect['total_effect']:.2f}")
        >>> print(f"Mean weekly effect: {effect['mean_effect']:.2f}")

        Notes
        -----
        **Counterfactual Computation:**
        1. Identify observations in the specified window
        2. Scale the raw exposure series for specified channels by the scale factor
        3. Reapply all transforms (saturation, adstock, lag) with the original
           fitted parameters to the scaled exposures
        4. Predict counterfactual outcome using the fitted baseline and treatment
           coefficients with the counterfactual treatment matrix
        5. Compute effect as observed - counterfactual

        **Important:** The transforms are recomputed with the same parameters, but
        the transformed values will differ due to the scaled raw inputs. For example,
        if adstock is applied, the carryover from periods before the window is
        correctly incorporated.

        **MVP Note:** Returns point estimates only. FUTURE: Add bootstrap or
        asymptotic confidence intervals for the effects.
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

        Creates a 3-panel figure showing:
        1. Observed vs fitted values
        2. Residuals over time
        3. Model R-squared

        Parameters
        ----------
        round_to : int, optional
            Number of decimal places for rounding displayed values. Default is 2.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : array of matplotlib.axes.Axes
            Array of axes objects.
        """
        return self._ols_plot(round_to=round_to, **kwargs)

    def _ols_plot(
        self, round_to: Optional[int] = 2, **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Generate OLS-specific plots.

        Parameters
        ----------
        round_to : int, optional
            Number of decimal places for rounding. Default is 2.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : array of matplotlib.axes.Axes
        """
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
            Maximum lag to display. If None, uses the l_max from the channel's
            Adstock transform, or 12 if no adstock is present.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.

        Examples
        --------
        >>> result.plot_irf("tv_spend")

        Notes
        -----
        The IRF shows the adstock weights, which represent how a one-unit increase
        in (saturated) exposure at time t affects the transformed variable at
        t, t+1, t+2, etc. If normalize=True was used, the weights sum to 1.
        """
        # Find the treatment
        treatment = None
        for t in self.treatments:
            if t.name == channel:
                treatment = t
                break

        if treatment is None:
            raise ValueError(f"Channel '{channel}' not found in treatments")

        # Extract adstock parameters
        adstock = None
        for transform in treatment.transforms:
            if isinstance(transform, Adstock):
                adstock = transform
                break

        if adstock is None:
            print(f"No adstock transform found for channel '{channel}'")
            return None

        # Verify alpha is set (should be set by __post_init__)
        if adstock.alpha is None:
            raise ValueError(
                f"Adstock transform for channel '{channel}' has alpha=None. "
                "This should not happen if half_life or alpha was provided."
            )

        # Generate IRF (adstock weights)
        if max_lag is None:
            max_lag = adstock.l_max

        lags = np.arange(max_lag + 1)
        weights = adstock.alpha**lags

        if adstock.normalize:
            weights = weights / weights.sum()

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(lags, weights, alpha=0.7, color="C0")
        ax.set_xlabel("Lag (periods)")
        ax.set_ylabel("Weight")

        # Calculate half-life: alpha^h = 0.5, so h = log(0.5) / log(alpha)
        half_life_calc = np.log(0.5) / np.log(adstock.alpha)

        ax.set_title(
            f"Impulse Response Function: {channel}\n"
            f"(alpha={adstock.alpha:.3f}, half_life={half_life_calc:.2f}, "
            f"normalize={adstock.normalize})"
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
            Number of lags to display in ACF/PACF plots and use in Ljung-Box test.

        Notes
        -----
        **Interpreting Diagnostics:**
        - **ACF/PACF**: Should show no significant autocorrelation at most lags
          (values should be within confidence bands). Significant autocorrelation
          suggests the model may be misspecified or that HAC standard errors are
          needed (which we already use).
        - **Ljung-Box test**: Tests the null hypothesis that residuals are
          independently distributed. Large p-values (> 0.05) suggest no significant
          autocorrelation. Small p-values indicate autocorrelation remains.

        **If diagnostics show problems:**
        - Residual autocorrelation is addressed by HAC standard errors (already used)
        - If strong patterns remain, consider:
          - Adding more baseline controls (trend, seasonality)
          - Adjusting adstock parameters (longer/shorter memory)
          - FUTURE: Use GLSAR or ARIMAX error models for explicit AR structure

        **MVP Note:** Basic diagnostics only. FUTURE: Add placebo tests,
        boundary sensitivity, multi-channel collinearity warnings.
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
                "  - HAC standard errors (already used) account for this in coefficient inference.\n"
                "  - Consider adding more baseline controls or adjusting transform parameters.\n"
                "  - FUTURE: GLSAR or ARIMAX models can explicitly model AR structure."
            )
        else:
            print("✓ No significant residual autocorrelation detected.")
        print("=" * 60)

    def summary(self, round_to: Optional[int] = None) -> None:
        """Print a summary of the model results.

        Parameters
        ----------
        round_to : int, optional
            Number of decimal places for rounding. Default is 2.
        """
        # Set default rounding
        if round_to is None:
            round_to = 2

        print("=" * 80)
        print("Transfer Function Interrupted Time Series Results")
        print("=" * 80)
        print(f"Outcome variable: {self.y_column}")
        print(f"Number of observations: {len(self.y)}")
        print(f"R-squared: {round_num(self.score, round_to)}")
        print(f"HAC max lags: {self.hac_maxlags}")
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
        for label, coef, se in zip(
            self.treatment_labels,
            self.theta_treatment,
            self.ols_result.bse[n_baseline:],
        ):
            coef_rounded = round_num(coef, round_to)
            se_rounded = round_num(se, round_to)
            print(f"  {label:20s}: {coef_rounded:>10} (SE: {se_rounded})")
        print("=" * 80)

    # Methods required by BaseExperiment (for future Bayesian support)
    def _bayesian_plot(self, *args, **kwargs):
        """Bayesian plotting not yet implemented."""
        raise NotImplementedError("Bayesian inference not yet supported for TFITS")

    def get_plot_data_bayesian(self, *args, **kwargs):
        """Bayesian plot data not yet implemented."""
        raise NotImplementedError("Bayesian inference not yet supported for TFITS")

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
