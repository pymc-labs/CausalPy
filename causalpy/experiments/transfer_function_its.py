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

Parameter Estimation
--------------------
Transform parameters (saturation and adstock) are estimated via nested optimization:

1. **Outer Loop**: Search over transform parameters (saturation slope/kappa,
   adstock half-life) using either:
   - Grid search: Exhaustive evaluation of discrete parameter combinations
   - Continuous optimization: scipy.optimize.minimize for faster convergence

2. **Inner Loop**: For each candidate set of transform parameters:
   - Apply transforms to the treatment variable
   - Fit OLS model with HAC standard errors
   - Compute RMSE as the optimization metric

3. **Selection**: The transform parameters that yield the lowest RMSE are selected
   as the final estimates. These parameters, along with the OLS coefficients from
   the best-fitting model, define the complete fitted model.

This nested approach is efficient because OLS has a closed-form solution, making
the inner loop fast even when evaluating many parameter combinations.
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
from causalpy.transforms import Treatment
from causalpy.utils import round_num

from .base import BaseExperiment

LEGEND_FONT_SIZE = 12


class TransferFunctionITS(BaseExperiment):
    """
    Transfer Function Interrupted Time Series experiment class.

    This experiment estimates the causal effect of graded interventions (e.g., media
    spend, policy intensity) in a single market using transfer functions that model
    saturation and adstock (carryover) effects.

    Transform parameters (saturation and adstock) are estimated from the data via
    grid search or continuous optimization to find the best fit.

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
        Treatment specifications with estimated transforms.
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
    transform_estimation_method : str
        Method used for parameter estimation ("grid" or "optimize").
    transform_estimation_results : dict
        Full results from parameter estimation including best_score, best_params.
    transform_search_space : dict
        Parameter grids or bounds that were searched.

    Examples
    --------
    >>> import causalpy as cp
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample data
    >>> dates = pd.date_range("2022-01-01", periods=104, freq="W")
    >>> df = pd.DataFrame(
    ...     {
    ...         "date": dates,
    ...         "water_consumption": np.random.normal(5000, 500, 104),
    ...         "comm_intensity": np.random.uniform(0, 10, 104),
    ...         "temperature": 25 + 10 * np.sin(2 * np.pi * np.arange(104) / 52),
    ...         "rainfall": 8 - 8 * np.sin(2 * np.pi * np.arange(104) / 52),
    ...     }
    ... )
    >>> df = df.set_index("date")
    >>> df["t"] = np.arange(len(df))
    >>>
    >>> # Estimate transform parameters via grid search
    >>> result = cp.TransferFunctionITS.with_estimated_transforms(
    ...     data=df,
    ...     y_column="water_consumption",
    ...     treatment_name="comm_intensity",
    ...     base_formula="1 + t + temperature + rainfall",
    ...     estimation_method="grid",
    ...     saturation_type="hill",
    ...     saturation_grid={"slope": [1.0, 2.0, 3.0], "kappa": [3, 5, 7]},
    ...     adstock_grid={"half_life": [2, 3, 4, 5]},
    ... )
    >>>
    >>> # View estimated parameters
    >>> print(result.transform_estimation_results["best_params"])
    >>>
    >>> # Estimate effect of policy over entire period
    >>> effect_result = result.effect(
    ...     window=(df.index[0], df.index[-1]), channels=["comm_intensity"], scale=0.0
    ... )
    >>> print(f"Total effect: {effect_result['total_effect']:.2f}")
    >>>
    >>> # Visualize results
    >>> result.plot()
    >>> result.diagnostics()

    Notes
    -----
    **Instantiation:**
    Models are created via the `with_estimated_transforms()` class method, which
    estimates optimal transform parameters from the data. Direct instantiation
    is not supported.

    **Transform Estimation:**
    Two methods are available:
    - Grid search: Exhaustive search over discrete parameter values (slower, guaranteed best)
    - Continuous optimization: Uses scipy.optimize (faster, may find local optima)

    **Future Extensions:**
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

    def _init_from_treatments(
        self,
        data: pd.DataFrame,
        y_column: str,
        base_formula: str,
        treatments: List[Treatment],
        hac_maxlags: Optional[int] = None,
        model=None,
        **kwargs,
    ) -> None:
        """Initialize and fit the Transfer Function ITS model with given treatments.

        This is a private method called by with_estimated_transforms().
        Users should not call this directly - use with_estimated_transforms() instead.
        """
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

        # Transform estimation metadata (set by with_estimated_transforms)
        self.transform_estimation_method = None  # "grid", "optimize", or None
        self.transform_estimation_results = None  # Full results dict
        self.transform_search_space = None  # Grid or bounds that were searched

    @classmethod
    def with_estimated_transforms(
        cls,
        data: pd.DataFrame,
        y_column: str,
        treatment_name: str,
        base_formula: str,
        estimation_method: str = "grid",
        saturation_type: str = "hill",
        coef_constraint: str = "nonnegative",
        hac_maxlags: Optional[int] = None,
        **estimation_kwargs,
    ) -> "TransferFunctionITS":
        """
        Create a TransferFunctionITS with transform parameters estimated from data.

        This method estimates optimal saturation and adstock parameters via grid
        search or continuous optimization, then creates a TransferFunctionITS
        instance with those estimated transforms.

        Parameters
        ----------
        data : pd.DataFrame
            Time series data with datetime or numeric index.
        y_column : str
            Name of the outcome variable column in data.
        treatment_name : str
            Name of the treatment variable column in data.
        base_formula : str
            Patsy formula for the baseline model (trend, seasonality, controls).
        estimation_method : str, default="grid"
            Method for parameter estimation: "grid" or "optimize".
            - "grid": Grid search over discrete parameter values
            - "optimize": Continuous optimization using scipy.optimize
        saturation_type : str, default="hill"
            Type of saturation function: "hill", "logistic", or "michaelis_menten".
        coef_constraint : str, default="nonnegative"
            Constraint on treatment coefficient ("nonnegative" or "unconstrained").
        hac_maxlags : int, optional
            Maximum lags for HAC standard errors. If None, uses rule of thumb.
        **estimation_kwargs
            Additional keyword arguments for the estimation method:

            For grid search (estimation_method="grid"):
            - saturation_grid : dict
                Dictionary mapping parameter names to lists of values.
                E.g., {"slope": [1.0, 2.0], "kappa": [3, 5, 7]}
            - adstock_grid : dict
                Dictionary mapping parameter names to lists of values.
                E.g., {"half_life": [2, 3, 4]}

            For optimization (estimation_method="optimize"):
            - saturation_bounds : dict
                Dictionary mapping parameter names to (min, max) tuples.
                E.g., {"slope": (0.5, 5.0), "kappa": (2, 10)}
            - adstock_bounds : dict
                Dictionary mapping parameter names to (min, max) tuples.
                E.g., {"half_life": (1, 10)}
            - initial_params : dict, optional
                Initial parameter values for optimization.
            - method : str, default="L-BFGS-B"
                Scipy optimization method.

        Returns
        -------
        TransferFunctionITS
            Fitted model with estimated transform parameters.

        Examples
        --------
        >>> # Grid search example
        >>> result = TransferFunctionITS.with_estimated_transforms(
        ...     data=df,
        ...     y_column="water_consumption",
        ...     treatment_name="comm_intensity",
        ...     base_formula="1 + t + temperature + rainfall",
        ...     estimation_method="grid",
        ...     saturation_type="hill",
        ...     saturation_grid={"slope": [1.0, 2.0, 3.0], "kappa": [3, 5, 7]},
        ...     adstock_grid={"half_life": [2, 3, 4, 5]},
        ... )
        >>> print(f"Best RMSE: {result.transform_estimation_results['best_score']:.2f}")

        >>> # Optimization example
        >>> result = TransferFunctionITS.with_estimated_transforms(
        ...     data=df,
        ...     y_column="water_consumption",
        ...     treatment_name="comm_intensity",
        ...     base_formula="1 + t + temperature + rainfall",
        ...     estimation_method="optimize",
        ...     saturation_type="hill",
        ...     saturation_bounds={"slope": (0.5, 5.0), "kappa": (2, 10)},
        ...     adstock_bounds={"half_life": (1, 10)},
        ...     initial_params={"slope": 2.0, "kappa": 5.0, "half_life": 4.0},
        ... )

        Notes
        -----
        This method performs nested optimization:
        - Outer loop: Search over transform parameters
        - Inner loop: Fit OLS for each set of transform parameters
        - Objective: Minimize RMSE

        Grid search is exhaustive but can be slow for large grids. Continuous
        optimization is faster but may find local optima. Consider using grid
        search first to find good starting points for optimization.
        """
        from causalpy.transform_optimization import (
            estimate_transform_params_grid,
            estimate_transform_params_optimize,
        )

        # Run parameter estimation
        if estimation_method == "grid":
            if "saturation_grid" not in estimation_kwargs:
                raise ValueError(
                    "saturation_grid is required for grid search method. "
                    "E.g., saturation_grid={'slope': [1.0, 2.0], 'kappa': [3, 5]}"
                )
            if "adstock_grid" not in estimation_kwargs:
                raise ValueError(
                    "adstock_grid is required for grid search method. "
                    "E.g., adstock_grid={'half_life': [2, 3, 4]}"
                )

            est_results = estimate_transform_params_grid(
                data=data,
                y_column=y_column,
                treatment_name=treatment_name,
                base_formula=base_formula,
                saturation_type=saturation_type,
                saturation_grid=estimation_kwargs["saturation_grid"],
                adstock_grid=estimation_kwargs["adstock_grid"],
                coef_constraint=coef_constraint,
                hac_maxlags=hac_maxlags,
            )

            search_space = {
                "saturation_grid": estimation_kwargs["saturation_grid"],
                "adstock_grid": estimation_kwargs["adstock_grid"],
            }

        elif estimation_method == "optimize":
            if "saturation_bounds" not in estimation_kwargs:
                raise ValueError(
                    "saturation_bounds is required for optimize method. "
                    "E.g., saturation_bounds={'slope': (0.5, 5.0), 'kappa': (2, 10)}"
                )
            if "adstock_bounds" not in estimation_kwargs:
                raise ValueError(
                    "adstock_bounds is required for optimize method. "
                    "E.g., adstock_bounds={'half_life': (1, 10)}"
                )

            est_results = estimate_transform_params_optimize(
                data=data,
                y_column=y_column,
                treatment_name=treatment_name,
                base_formula=base_formula,
                saturation_type=saturation_type,
                saturation_bounds=estimation_kwargs["saturation_bounds"],
                adstock_bounds=estimation_kwargs["adstock_bounds"],
                initial_params=estimation_kwargs.get("initial_params"),
                coef_constraint=coef_constraint,
                hac_maxlags=hac_maxlags,
                method=estimation_kwargs.get("method", "L-BFGS-B"),
            )

            search_space = {
                "saturation_bounds": estimation_kwargs["saturation_bounds"],
                "adstock_bounds": estimation_kwargs["adstock_bounds"],
                "initial_params": estimation_kwargs.get("initial_params"),
                "method": estimation_kwargs.get("method", "L-BFGS-B"),
            }

        else:
            raise ValueError(
                f"Unknown estimation_method: {estimation_method}. "
                "Use 'grid' or 'optimize'."
            )

        # Create Treatment with best transforms
        from causalpy.transforms import Treatment

        treatment = Treatment(
            name=treatment_name,
            saturation=est_results["best_saturation"],
            adstock=est_results["best_adstock"],
            coef_constraint=coef_constraint,
        )

        # Create TransferFunctionITS instance and initialize with estimated transforms
        result = cls.__new__(cls)
        result._init_from_treatments(
            data=data,
            y_column=y_column,
            base_formula=base_formula,
            treatments=[treatment],
            hac_maxlags=hac_maxlags,
        )

        # Store estimation metadata
        result.transform_estimation_method = estimation_method
        result.transform_estimation_results = est_results
        result.transform_search_space = search_space

        return result

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

            # Apply transform pipeline using strategy pattern
            # Transforms are applied in order: Saturation → Adstock → Lag
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

        # Extract adstock transform (now directly accessible via treatment.adstock)
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

        # Calculate half-life: alpha^h = 0.5, so h = log(0.5) / log(alpha)
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
