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
"""Custom scikit-learn models for causal inference"""

from functools import partial
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix
from scipy.optimize import fmin_slsqp
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel

from causalpy.utils import round_num


class ScikitLearnAdaptor:
    """Base class for scikit-learn models that can be used for causal inference."""

    def calculate_impact(self, y_true, y_pred):
        """Calculate the causal impact of the intervention."""
        return y_true - y_pred

    def calculate_cumulative_impact(self, impact):
        """Calculate the cumulative impact intervention."""
        return np.cumsum(impact)

    def print_coefficients(self, labels, round_to=None) -> None:
        """Print the coefficients of the model with the corresponding labels."""
        print("Model coefficients:")
        coef_ = self.get_coeffs()
        # Determine the width of the longest label
        max_label_length = max(len(name) for name in labels)
        # Print each coefficient with formatted alignment
        for name, val in zip(labels, coef_):
            # Left-align the name
            formatted_name = f"{name:<{max_label_length}}"
            # Right-align the value with width 10
            formatted_val = f"{round_num(val, round_to):>10}"
            print(f"  {formatted_name}\t{formatted_val}")

    def get_coeffs(self):
        """Get the coefficients of the model as a numpy array."""
        return np.squeeze(self.coef_)


class WeightedProportion(ScikitLearnAdaptor, LinearModel, RegressorMixin):
    """Weighted proportion model for causal inference. Used for synthetic control
    methods for example"""

    def loss(self, W, X, y):
        """Compute root mean squared loss with data X, weights W, and predictor y"""
        return np.sqrt(np.mean((y - np.dot(X, W.T)) ** 2))

    def fit(self, X, y):
        """Fit model on data X with predictor y"""
        w_start = [1 / X.shape[1]] * X.shape[1]
        coef_ = fmin_slsqp(
            partial(self.loss, X=X, y=y),
            np.array(w_start),
            f_eqcons=lambda w: np.sum(w) - 1,
            bounds=[(0.0, 1.0)] * len(w_start),
            disp=False,
        )
        self.coef_ = np.atleast_2d(coef_)  # return as column vector
        self.mse = self.loss(W=self.coef_, X=X, y=y)
        return self

    def predict(self, X):
        """Predict results for data X"""
        return np.dot(X, self.coef_.T)


class TransferFunctionOLS(ScikitLearnAdaptor, LinearModel, RegressorMixin):
    """
    OLS model with transfer functions for graded interventions.

    This model supports:
    - HAC (Newey-West) standard errors for robust inference (default)
    - ARIMAX error models for explicit autocorrelation modeling
    - Saturation and adstock transforms for treatment effects

    The model is designed to work with the GradedInterventionTimeSeries experiment
    class. Use the `with_estimated_transforms()` factory method to estimate
    transform parameters and fit the model in one step.

    Parameters
    ----------
    error_model : str, default="hac"
        Error model specification: "hac" or "arimax".
        - "hac": HAC (Newey-West) standard errors. Robust to autocorrelation.
        - "arimax": ARIMA(p,d,q) errors with exogenous variables.
    arima_order : tuple of (int, int, int), optional
        ARIMA order (p, d, q) when error_model="arimax". Required for ARIMAX.
    hac_maxlags : int, optional
        Maximum lags for HAC standard errors. If None, uses Newey-West rule of thumb.

    Attributes
    ----------
    ols_result : statsmodels regression result
        Fitted OLS or ARIMAX model result.
    treatments : List[Treatment]
        Treatment specifications with transform objects.
    X_baseline : np.ndarray
        Baseline design matrix.
    X_full : np.ndarray
        Full design matrix (baseline + treatments).
    score : float
        R-squared of the model.
    coef_ : np.ndarray
        Model coefficients (for sklearn compatibility).

    Examples
    --------
    >>> # Use factory method to estimate transforms and fit model
    >>> model = TransferFunctionOLS.with_estimated_transforms(
    ...     data=df,
    ...     y_column="outcome",
    ...     treatment_name="exposure",
    ...     base_formula="1 + t",
    ...     estimation_method="grid",
    ...     saturation_grid={"slope": [1.0, 2.0], "kappa": [3, 5]},
    ...     adstock_grid={"half_life": [2, 3, 4]},
    ...     error_model="hac",
    ... )
    >>> # Use with experiment class
    >>> from causalpy import GradedInterventionTimeSeries
    >>> result = GradedInterventionTimeSeries(
    ...     data=df,
    ...     y_column="outcome",
    ...     treatment_name="exposure",
    ...     base_formula="1 + t",
    ...     treatments=model.treatments,
    ...     model=model,
    ... )
    """

    def __init__(
        self,
        error_model: str = "hac",
        arima_order: Optional[Tuple[int, int, int]] = None,
        hac_maxlags: Optional[int] = None,
    ):
        """Initialize model with error structure specification."""
        self.error_model = error_model
        self.arima_order = arima_order
        self.hac_maxlags = hac_maxlags

        # Validate error model
        if error_model not in ["hac", "arimax"]:
            raise ValueError(
                f"error_model must be 'hac' or 'arimax', got '{error_model}'"
            )
        if error_model == "arimax" and arima_order is None:
            raise ValueError(
                "arima_order must be provided when error_model='arimax'. "
                "E.g., arima_order=(1, 0, 0) for AR(1) errors"
            )

        # Initialize attributes (set by fit())
        self.ols_result = None
        self.treatments = None
        self.X_baseline = None
        self.X_full = None
        self.y = None
        self.baseline_labels = None
        self.treatment_labels = None
        self.score = None
        self.coef_ = None  # For sklearn compatibility

        # For ARIMAX models
        self.arimax_model = None

        # Transform estimation metadata (set by with_estimated_transforms())
        self.transform_estimation_method = None
        self.transform_estimation_results = None
        self.transform_search_space = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit OLS model with HAC or ARIMAX error structure.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (n_obs, n_features).
        y : np.ndarray
            Outcome variable (n_obs,).

        Returns
        -------
        self : TransferFunctionOLS
            Fitted model.
        """
        self.y = y
        self.X_full = X

        # Fit model with chosen error structure
        if self.error_model == "hac":
            # Fit OLS with HAC standard errors
            if self.hac_maxlags is None:
                # Newey & West (1994) rule of thumb
                n = len(y)
                self.hac_maxlags = int(np.floor(4 * (n / 100) ** (2 / 9)))

            self.ols_result = sm.OLS(y, X).fit(
                cov_type="HAC", cov_kwds={"maxlags": self.hac_maxlags}
            )

        elif self.error_model == "arimax":
            # Fit ARIMAX model
            import warnings

            from statsmodels.tsa.statespace.sarimax import SARIMAX

            # Suppress convergence warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.arimax_model = SARIMAX(y, exog=X, order=self.arima_order)
                self.ols_result = self.arimax_model.fit(
                    disp=0,
                    maxiter=200,
                    method="lbfgs",
                )

        # Compute R-squared
        if hasattr(self.ols_result, "rsquared"):
            self.score = self.ols_result.rsquared
        else:
            # For ARIMAX, compute R-squared manually
            residuals = self.ols_result.resid
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            self.score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Store coefficients for sklearn compatibility
        self.coef_ = self.ols_result.params.reshape(1, -1)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (n_obs, n_features).

        Returns
        -------
        y_pred : np.ndarray
            Predicted values (n_obs,).
        """
        if self.ols_result is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        return X @ self.ols_result.params

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
        error_model: str = "hac",
        arima_order: Optional[Tuple[int, int, int]] = None,
        **estimation_kwargs,
    ) -> "TransferFunctionOLS":
        """
        Factory method: estimate transform parameters and return fitted model.

        This method performs the complete workflow:
        1. Estimate optimal saturation and adstock parameters
        2. Create Treatment objects with estimated transforms
        3. Build design matrices
        4. Fit the model
        5. Return fitted instance

        Parameters
        ----------
        data : pd.DataFrame
            Time series data with datetime or numeric index.
        y_column : str
            Name of the outcome variable column in data.
        treatment_name : str
            Name of the treatment variable column in data.
        base_formula : str
            Patsy formula for the baseline model (e.g., "1 + t + temperature").
        estimation_method : str, default="grid"
            Method for parameter estimation: "grid" or "optimize".
        saturation_type : str, default="hill"
            Type of saturation function: "hill", "logistic", or "michaelis_menten".
        coef_constraint : str, default="nonnegative"
            Constraint on treatment coefficient.
        hac_maxlags : int, optional
            Maximum lags for HAC standard errors.
        error_model : str, default="hac"
            Error model: "hac" or "arimax".
        arima_order : tuple of (int, int, int), optional
            ARIMA order (p, d, q) when error_model="arimax".
        **estimation_kwargs
            Additional keyword arguments for estimation:
            - For grid: saturation_grid, adstock_grid
            - For optimize: saturation_bounds, adstock_bounds, initial_params

        Returns
        -------
        TransferFunctionOLS
            Fitted model with estimated transform parameters.

        Examples
        --------
        >>> model = TransferFunctionOLS.with_estimated_transforms(
        ...     data=df,
        ...     y_column="water_consumption",
        ...     treatment_name="comm_intensity",
        ...     base_formula="1 + t + temperature + rainfall",
        ...     estimation_method="grid",
        ...     saturation_type="hill",
        ...     saturation_grid={"slope": [1.0, 2.0, 3.0], "kappa": [3, 5, 7]},
        ...     adstock_grid={"half_life": [2, 3, 4, 5]},
        ...     error_model="hac",
        ... )
        """
        from causalpy.transform_optimization import (
            estimate_transform_params_grid,
            estimate_transform_params_optimize,
        )
        from causalpy.transforms import Treatment

        # Validate error model parameters
        if error_model not in ["hac", "arimax"]:
            raise ValueError(
                f"error_model must be 'hac' or 'arimax', got '{error_model}'"
            )
        if error_model == "arimax" and arima_order is None:
            raise ValueError(
                "arima_order must be provided when error_model='arimax'. "
                "E.g., arima_order=(1, 0, 0) for AR(1) errors"
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
                error_model=error_model,
                arima_order=arima_order,
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
                error_model=error_model,
                arima_order=arima_order,
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
        treatment = Treatment(
            name=treatment_name,
            saturation=est_results["best_saturation"],
            adstock=est_results["best_adstock"],
            coef_constraint=coef_constraint,
        )

        # Build design matrices
        y = data[y_column].values
        X_baseline = np.asarray(dmatrix(base_formula, data))
        baseline_labels = dmatrix(base_formula, data).design_info.column_names

        # Build treatment matrix by applying transforms
        x_raw = data[treatment_name].values
        x_transformed = x_raw
        if treatment.saturation is not None:
            x_transformed = treatment.saturation.apply(x_transformed)
        if treatment.adstock is not None:
            x_transformed = treatment.adstock.apply(x_transformed)
        if treatment.lag is not None:
            x_transformed = treatment.lag.apply(x_transformed)

        Z_treatment = x_transformed.reshape(-1, 1)
        treatment_labels = [treatment_name]

        # Combine matrices
        X_full = np.column_stack([X_baseline, Z_treatment])

        # Create and fit model instance
        model = cls(
            error_model=error_model,
            arima_order=arima_order,
            hac_maxlags=hac_maxlags,
        )
        model.fit(X_full, y)

        # Store additional metadata
        model.treatments = [treatment]
        model.X_baseline = X_baseline
        model.baseline_labels = baseline_labels
        model.treatment_labels = treatment_labels
        model.transform_estimation_method = estimation_method
        model.transform_estimation_results = est_results
        model.transform_search_space = search_space

        return model


def create_causalpy_compatible_class(
    estimator: type[RegressorMixin],
) -> type[RegressorMixin]:
    """This function takes a scikit-learn estimator and returns a new class that is
    compatible with CausalPy."""
    _add_mixin_methods(estimator, ScikitLearnAdaptor)
    return estimator


def _add_mixin_methods(model_instance, mixin_class):
    """Utility function to bind mixin methods to an existing model instance."""
    for attr_name in dir(mixin_class):
        attr = getattr(mixin_class, attr_name)
        if callable(attr) and not attr_name.startswith("__"):
            # Bind the method to the instance
            method = attr.__get__(model_instance, model_instance.__class__)
            setattr(model_instance, attr_name, method)
    return model_instance
