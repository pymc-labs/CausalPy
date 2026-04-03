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
"""Custom scikit-learn models for causal inference"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import statsmodels.api as sm
from scipy.optimize import fmin_slsqp
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel

from causalpy.utils import round_num


class ScikitLearnAdaptor:
    """Base class for scikit-learn models that can be used for causal inference."""

    coef_: np.ndarray

    def calculate_impact(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate the causal impact of the intervention."""
        return y_true - y_pred

    def calculate_cumulative_impact(self, impact: np.ndarray) -> np.ndarray:
        """Calculate the cumulative impact intervention."""
        return np.cumsum(impact)

    def print_coefficients(
        self, labels: list[str], round_to: int | None = None
    ) -> None:
        """Print the coefficients of the model with the corresponding labels.

        Parameters
        ----------
        labels : list of str
            List of strings representing the coefficient names.
        round_to : int, optional
            Number of significant figures to round to. Defaults to None,
            in which case 2 significant figures are used.
        """
        print("Model coefficients:")
        coef_ = self.get_coeffs()
        # Determine the width of the longest label
        max_label_length = max(len(name) for name in labels)
        # Print each coefficient with formatted alignment
        for name, val in zip(labels, coef_, strict=False):
            # Left-align the name
            formatted_name = f"{name:<{max_label_length}}"
            # Right-align the value with width 10
            formatted_val = (
                f"{round_num(val, round_to if round_to is not None else 2):>10}"
            )
            print(f"  {formatted_name}\t{formatted_val}")

    def get_coeffs(self) -> np.ndarray:
        """Get the coefficients of the model as a numpy array."""
        return np.squeeze(self.coef_)


class WeightedProportion(ScikitLearnAdaptor, LinearModel, RegressorMixin):
    """Weighted proportion model for causal inference. Used for synthetic control
    methods for example"""

    def loss(self, W: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Compute root mean squared loss with data X, weights W, and predictor y"""
        return np.sqrt(np.mean((y - np.dot(X, W.T)) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> WeightedProportion:
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict results for data X"""
        return np.dot(X, self.coef_.T)


class TransferFunctionOLS(ScikitLearnAdaptor, LinearModel, RegressorMixin):
    """
    OLS model with transfer functions for graded interventions.

    This model supports:
    - HAC (Newey-West) standard errors for robust inference (default)
    - ARIMAX error models for explicit autocorrelation modeling
    - Saturation and adstock transforms for treatment effects

    This model is designed to work with the GradedInterventionTimeSeries experiment
    class following the standard CausalPy pattern where the experiment handles data
    preparation and calls model.fit().

    Parameters
    ----------
    saturation_type : str, default="hill"
        Type of saturation function: "hill", "logistic", or "michaelis_menten".
    saturation_grid : dict, optional
        For grid search: dict mapping parameter names to lists of values.
        E.g., {"slope": [1.0, 2.0], "kappa": [3, 5]}.
    saturation_bounds : dict, optional
        For optimization: dict mapping parameter names to (min, max) tuples.
        E.g., {"slope": (0.5, 5.0), "kappa": (2, 10)}.
    adstock_grid : dict, optional
        For grid search: dict mapping parameter names to lists of values.
        E.g., {"half_life": [2, 3, 4]}.
    adstock_bounds : dict, optional
        For optimization: dict mapping parameter names to (min, max) tuples.
        E.g., {"half_life": (1, 10)}.
    estimation_method : str, default="grid"
        Method for parameter estimation: "grid" or "optimize".
    error_model : str, default="hac"
        Error model specification: "hac" or "arimax".
    arima_order : tuple of (int, int, int), optional
        ARIMA order (p, d, q) when error_model="arimax".
    hac_maxlags : int, optional
        Maximum lags for HAC standard errors.
    coef_constraint : str, default="nonnegative"
        Constraint on treatment coefficients.

    Attributes
    ----------
    ols_result : statsmodels regression result
        Fitted OLS or ARIMAX model result.
    treatments : List[Treatment]
        Treatment specifications with transform objects.
    score : float
        R-squared of the model.
    coef_ : np.ndarray
        Model coefficients (for sklearn compatibility).

    Examples
    --------
    .. code-block:: python

        # Create unfitted model with configuration
        model = cp.skl_models.TransferFunctionOLS(
            saturation_type="hill",
            saturation_grid={"slope": [1.0, 2.0], "kappa": [3, 5]},
            adstock_grid={"half_life": [2, 3, 4]},
            estimation_method="grid",
            error_model="hac",
        )

        # Use with experiment class (experiment calls fit())
        result = cp.GradedInterventionTimeSeries(
            data=df,
            y_column="outcome",
            treatment_names=["exposure"],
            base_formula="1 + t",
            model=model,
        )
    """

    def __init__(
        self,
        saturation_type: str = "hill",
        saturation_grid: dict | None = None,
        saturation_bounds: dict | None = None,
        adstock_grid: dict | None = None,
        adstock_bounds: dict | None = None,
        estimation_method: str = "grid",
        error_model: str = "hac",
        arima_order: tuple[int, int, int] | None = None,
        hac_maxlags: int | None = None,
        coef_constraint: str = "nonnegative",
    ):
        """Initialize model with configuration parameters."""
        # Store configuration
        self.saturation_type = saturation_type
        self.saturation_grid = saturation_grid
        self.saturation_bounds = saturation_bounds
        self.adstock_grid = adstock_grid
        self.adstock_bounds = adstock_bounds
        self.estimation_method = estimation_method
        self.error_model = error_model
        self.arima_order = arima_order
        self.hac_maxlags = hac_maxlags
        self.coef_constraint = coef_constraint

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

        # Validate estimation method and required parameters
        if estimation_method == "grid":
            # At least one transform must be specified
            if saturation_grid is None and adstock_grid is None:
                raise ValueError(
                    "At least one of saturation_grid or adstock_grid must be provided for grid search. "
                    "To use only adstock: set saturation_type=None and provide adstock_grid. "
                    "To use only saturation: provide saturation_grid and set adstock_grid=None."
                )
            # If saturation_type is specified, grid must be provided
            if saturation_type is not None and saturation_grid is None:
                raise ValueError(
                    f"saturation_grid is required when saturation_type='{saturation_type}'. "
                    "E.g., saturation_grid={'lam': [0.2, 0.5, 0.8]}"
                )
        elif estimation_method == "optimize":
            # At least one transform must be specified
            if saturation_bounds is None and adstock_bounds is None:
                raise ValueError(
                    "At least one of saturation_bounds or adstock_bounds must be provided for optimize method. "
                    "To use only adstock: set saturation_type=None and provide adstock_bounds. "
                    "To use only saturation: provide saturation_bounds and set adstock_bounds=None."
                )
            # If saturation_type is specified, bounds must be provided
            if saturation_type is not None and saturation_bounds is None:
                raise ValueError(
                    f"saturation_bounds is required when saturation_type='{saturation_type}'. "
                    "E.g., saturation_bounds={'lam': (0.1, 1.0)}"
                )
        else:
            raise ValueError(
                f"estimation_method must be 'grid' or 'optimize', got '{estimation_method}'"
            )

        # Initialize attributes (set by fit())
        self.ols_result: Any = None
        self.treatments: list[Any] | None = None
        self.score: float | None = None
        self.coef_: np.ndarray | None = None  # type: ignore[assignment]
        self.arimax_model: Any = None

        # Transform estimation metadata (set by fit())
        self.transform_estimation_results: dict[str, Any] | None = None
        self.transform_search_space: dict[str, Any] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit OLS model with HAC/ARIMAX errors.

        Note: This method expects X to already contain the transformed treatment
        variables. Transform parameter estimation is handled by the experiment class.

        Parameters
        ----------
        X : np.ndarray
            Full design matrix (n_obs, n_features) including baseline AND
            transformed treatment variables.
        y : np.ndarray
            Outcome variable (n_obs,).

        Returns
        -------
        self : TransferFunctionOLS
            Fitted model.
        """
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


def create_causalpy_compatible_class(
    estimator: type[RegressorMixin],
) -> type[RegressorMixin]:
    """This function takes a scikit-learn estimator and returns a new class that is
    compatible with CausalPy."""
    _add_mixin_methods(estimator, ScikitLearnAdaptor)
    return estimator


def _add_mixin_methods(
    model_instance: RegressorMixin, mixin_class: type
) -> RegressorMixin:
    """Utility function to bind mixin methods to an existing model instance."""
    for attr_name in dir(mixin_class):
        attr = getattr(mixin_class, attr_name)
        if callable(attr) and not attr_name.startswith("__"):
            # Bind the method to the instance
            method = attr.__get__(model_instance, model_instance.__class__)
            setattr(model_instance, attr_name, method)
    return model_instance
