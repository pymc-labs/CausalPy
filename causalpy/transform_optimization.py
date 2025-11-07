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
Transform parameter estimation for Transfer Function ITS.

This module provides functions to estimate optimal saturation and adstock
parameters via grid search or continuous optimization. The key challenge is
that we need to estimate transform parameters jointly with the OLS coefficients,
creating a nested optimization problem.

The nested optimization works as follows:
1. Outer loop: Search over transform parameters (saturation + adstock)
2. Inner loop: For each set of transform parameters, fit OLS with HAC errors
3. Objective: Minimize RMSE (or maximize R-squared)

Since OLS has a closed-form solution, this is computationally tractable.
"""

from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix
from scipy.optimize import minimize

from causalpy.transforms import (
    GeometricAdstock,
    HillSaturation,
    LogisticSaturation,
    MichaelisMentenSaturation,
)

# FUTURE: Implement AICc metric for model comparison
# FUTURE: Implement out-of-sample validation (split data into estimation/validation)
# FUTURE: Add support for multiple channels with joint optimization
# FUTURE: Add parallelization for grid search
# FUTURE: Implement Bayesian optimization for parameter search


def _fit_ols_with_transforms(
    data: pd.DataFrame,
    y_column: str,
    treatment_name: str,
    base_formula: str,
    saturation,
    adstock,
    lag=None,
    hac_maxlags: Optional[int] = None,
    error_model: str = "hac",
    arima_order: Optional[Tuple[int, int, int]] = None,
) -> Tuple[float, sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Fit OLS model with specific transform parameters.

    This is the inner loop of the nested optimization. Given specific
    transform parameters, we fit OLS and return the RMSE.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y_column : str
        Name of outcome variable.
    treatment_name : str
        Name of treatment variable.
    base_formula : str
        Patsy formula for baseline predictors.
    saturation : SaturationTransform or None
        Saturation transform object.
    adstock : AdstockTransform or None
        Adstock transform object.
    lag : LagTransform or None
        Lag transform object.
    hac_maxlags : int, optional
        Maximum lags for HAC standard errors (ignored if error_model="arimax").
    error_model : str, default="hac"
        Error model: "hac" for HAC standard errors or "arimax" for ARIMAX.
    arima_order : tuple of (int, int, int), optional
        ARIMA order (p, d, q) when error_model="arimax".

    Returns
    -------
    rmse : float
        Root mean squared error of the fit.
    ols_result : RegressionResultsWrapper
        Fitted OLS or ARIMAX model object.
    """
    # Build baseline design matrix
    X_baseline = np.asarray(dmatrix(base_formula, data))

    # Apply transforms to treatment variable
    x_raw = data[treatment_name].values
    x_transformed = x_raw

    if saturation is not None:
        x_transformed = saturation.apply(x_transformed)
    if adstock is not None:
        x_transformed = adstock.apply(x_transformed)
    if lag is not None:
        x_transformed = lag.apply(x_transformed)

    # Build full design matrix
    X_full = np.column_stack([X_baseline, x_transformed])

    # Get outcome
    y = data[y_column].values

    # Fit model with chosen error structure
    if error_model == "hac":
        # Fit OLS with HAC standard errors
        if hac_maxlags is None:
            n = len(y)
            hac_maxlags = int(np.floor(4 * (n / 100) ** (2 / 9)))

        ols_result = sm.OLS(y, X_full).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_maxlags}
        )
    elif error_model == "arimax":
        # Fit ARIMAX model
        import warnings

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # ARIMAX requires at least as many observations as parameters
        # Quick validation
        n_obs = len(y)
        n_params = X_full.shape[1] + sum(arima_order)  # exog params + ARIMA params
        if n_obs < n_params + 10:  # Need some degrees of freedom
            raise ValueError(
                f"ARIMAX requires more observations. Have {n_obs}, need at least {n_params + 10}"
            )

        # Suppress convergence warnings during grid search
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arimax_model = SARIMAX(y, exog=X_full, order=arima_order)
            ols_result = arimax_model.fit(
                disp=0,
                maxiter=200,
                method="lbfgs",
            )
    else:
        raise ValueError(f"error_model must be 'hac' or 'arimax', got '{error_model}'")

    # Compute RMSE
    residuals = ols_result.resid
    rmse = np.sqrt(np.mean(residuals**2))

    return rmse, ols_result


def estimate_transform_params_grid(
    data: pd.DataFrame,
    y_column: str,
    treatment_name: str,
    base_formula: str,
    saturation_type: Optional[str],
    saturation_grid: Optional[Dict[str, List[float]]],
    adstock_grid: Optional[Dict[str, List[float]]],
    coef_constraint: str = "nonnegative",
    hac_maxlags: Optional[int] = None,
    metric: str = "rmse",
    error_model: str = "hac",
    arima_order: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, Any]:
    """
    Estimate transform parameters via grid search.

    Searches over all combinations of saturation and adstock parameters,
    fitting OLS or ARIMAX for each combination and selecting the one with lowest RMSE.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with treatment and outcome variables.
    y_column : str
        Name of outcome variable in data.
    treatment_name : str
        Name of treatment variable in data.
    base_formula : str
        Patsy formula for baseline predictors (e.g., "1 + t + temperature").
    saturation_type : str
        Type of saturation function: "hill", "logistic", or "michaelis_menten".
    saturation_grid : dict
        Dictionary mapping parameter names to lists of values to try.
        For "hill": {"slope": [...], "kappa": [...]}
        For "logistic": {"lam": [...]}
        For "michaelis_menten": {"alpha": [...], "lam": [...]}
    adstock_grid : dict
        Dictionary mapping parameter names to lists of values to try.
        Example: {"half_life": [2, 3, 4], "l_max": [12], "normalize": [True]}
    coef_constraint : str, default="nonnegative"
        Constraint on treatment coefficient.
    hac_maxlags : int, optional
        Maximum lags for HAC standard errors. If None, uses rule of thumb.
        Ignored if error_model="arimax".
    metric : str, default="rmse"
        Optimization metric. Currently only "rmse" is supported.
        FUTURE: "aicc", "oos_rmse"
    error_model : str, default="hac"
        Error model: "hac" for HAC standard errors or "arimax" for ARIMAX.
    arima_order : tuple of (int, int, int), optional
        ARIMA order (p, d, q) when error_model="arimax". Required if error_model="arimax".

    Returns
    -------
    dict
        Dictionary with keys:
        - "best_saturation": Best saturation transform object
        - "best_adstock": Best adstock transform object
        - "best_score": Best RMSE achieved
        - "best_params": Dictionary of best parameters
        - "grid_results": DataFrame with all tried combinations

    Examples
    --------
    .. code-block:: python

        result = estimate_transform_params_grid(
            data=df,
            y_column="water_consumption",
            treatment_name="comm_intensity",
            base_formula="1 + t + temperature + rainfall",
            saturation_type="hill",
            saturation_grid={"slope": [1.0, 2.0, 3.0], "kappa": [3, 5, 7]},
            adstock_grid={"half_life": [2, 3, 4, 5]},
        )
        print(f"Best RMSE: {result['best_score']:.2f}")
        print(f"Best params: {result['best_params']}")

    Notes
    -----
    The grid search evaluates all combinations of parameters, so computational
    cost grows multiplicatively with grid size. For fine-grained search,
    consider using estimate_transform_params_optimize() instead.
    """
    if metric != "rmse":
        raise NotImplementedError(f"Metric '{metric}' not yet implemented. Use 'rmse'.")

    # Generate saturation combinations (or single None if not used)
    if saturation_type is not None and saturation_grid is not None:
        sat_param_names = list(saturation_grid.keys())
        sat_param_values = list(saturation_grid.values())
        sat_combinations = list(product(*sat_param_values))
    else:
        sat_param_names = []
        sat_combinations = [None]  # Single "no saturation" combination

    # Generate adstock combinations (or single None if not used)
    if adstock_grid is not None:
        adstock_param_names = list(adstock_grid.keys())
        adstock_param_values = list(adstock_grid.values())
        adstock_combinations = list(product(*adstock_param_values))
    else:
        adstock_param_names = []
        adstock_combinations = [None]  # Single "no adstock" combination

    # Store results
    results = []
    best_score = float("inf")
    best_saturation = None
    best_adstock = None
    best_params = None

    # Grid search over all combinations
    for sat_params in sat_combinations:
        # Create saturation object or None
        if sat_params is not None:
            sat_kwargs = dict(zip(sat_param_names, sat_params))

            if saturation_type == "hill":
                saturation = HillSaturation(**sat_kwargs)
            elif saturation_type == "logistic":
                saturation = LogisticSaturation(**sat_kwargs)
            elif saturation_type == "michaelis_menten":
                saturation = MichaelisMentenSaturation(**sat_kwargs)
            else:
                raise ValueError(f"Unknown saturation type: {saturation_type}")
        else:
            saturation = None
            sat_kwargs = {}

        for adstock_params in adstock_combinations:
            # Create adstock object or None
            if adstock_params is not None:
                adstock_kwargs = dict(zip(adstock_param_names, adstock_params))
                adstock = GeometricAdstock(**adstock_kwargs)
            else:
                adstock = None
                adstock_kwargs = {}

            # Fit OLS with these transforms
            try:
                score, ols_result = _fit_ols_with_transforms(
                    data=data,
                    y_column=y_column,
                    treatment_name=treatment_name,
                    base_formula=base_formula,
                    saturation=saturation,
                    adstock=adstock,
                    lag=None,
                    hac_maxlags=hac_maxlags,
                    error_model=error_model,
                    arima_order=arima_order,
                )

                # Store result
                result_dict = {
                    **{f"sat_{k}": v for k, v in sat_kwargs.items()},
                    **{f"adstock_{k}": v for k, v in adstock_kwargs.items()},
                    "score": score,
                }
                # Add R-squared if available (OLS has it, ARIMAX doesn't)
                if hasattr(ols_result, "rsquared"):
                    result_dict["r_squared"] = ols_result.rsquared
                results.append(result_dict)

                # Update best if this is better
                if score < best_score:
                    best_score = score
                    best_saturation = saturation
                    best_adstock = adstock
                    best_params = {**sat_kwargs, **adstock_kwargs}

            except Exception:
                # If fitting fails (e.g., singular matrix, ARIMAX convergence), skip this combination
                # Silently continue - grid search tries many combinations and some may fail
                continue

    # Check if we found any valid combinations
    # Note: best_saturation can be None if using adstock-only, so check results instead
    if len(results) == 0 or best_params is None:
        # Provide more helpful error message
        if error_model == "arimax":
            raise ValueError(
                "Grid search failed: no valid ARIMAX parameter combinations found. "
                "This often happens when ARIMAX cannot converge for any parameter combination. "
                "Try: (1) using error_model='hac' instead, (2) increasing sample size, "
                "(3) simplifying the grid (fewer parameters), or (4) checking data quality."
            )
        else:
            raise ValueError(
                "Grid search failed: no valid parameter combinations found."
            )

    # Convert results to DataFrame
    grid_results = pd.DataFrame(results)

    return {
        "best_saturation": best_saturation,
        "best_adstock": best_adstock,
        "best_score": best_score,
        "best_params": best_params,
        "grid_results": grid_results,
        "saturation_type": saturation_type,
        "coef_constraint": coef_constraint,
    }


def estimate_transform_params_optimize(
    data: pd.DataFrame,
    y_column: str,
    treatment_name: str,
    base_formula: str,
    saturation_type: Optional[str],
    saturation_bounds: Optional[Dict[str, Tuple[float, float]]],
    adstock_bounds: Optional[Dict[str, Tuple[float, float]]],
    initial_params: Optional[Dict[str, float]] = None,
    coef_constraint: str = "nonnegative",
    hac_maxlags: Optional[int] = None,
    method: str = "L-BFGS-B",
    metric: str = "rmse",
    error_model: str = "hac",
    arima_order: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, Any]:
    """
    Estimate transform parameters via continuous optimization.

    Uses scipy.optimize.minimize to find optimal saturation and adstock
    parameters by minimizing RMSE. This is generally faster than grid search
    for fine-grained optimization, but may find local optima.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with treatment and outcome variables.
    y_column : str
        Name of outcome variable in data.
    treatment_name : str
        Name of treatment variable in data.
    base_formula : str
        Patsy formula for baseline predictors (e.g., "1 + t + temperature").
    saturation_type : str
        Type of saturation function: "hill", "logistic", or "michaelis_menten".
    saturation_bounds : dict
        Dictionary mapping parameter names to (min, max) tuples.
        For "hill": {"slope": (0.5, 5.0), "kappa": (1, 10)}
        For "logistic": {"lam": (0.01, 2.0)}
        For "michaelis_menten": {"alpha": (0.5, 2.0), "lam": (10, 1000)}
    adstock_bounds : dict
        Dictionary mapping parameter names to (min, max) tuples.
        Example: {"half_life": (1, 10)}
    initial_params : dict, optional
        Initial parameter values for optimization. If None, uses midpoint of bounds.
    coef_constraint : str, default="nonnegative"
        Constraint on treatment coefficient.
    hac_maxlags : int, optional
        Maximum lags for HAC standard errors. If None, uses rule of thumb.
        Ignored if error_model="arimax".
    method : str, default="L-BFGS-B"
        Scipy optimization method. Must support bounds (e.g., "L-BFGS-B", "TNC").
    metric : str, default="rmse"
        Optimization metric. Currently only "rmse" is supported.
        FUTURE: "aicc", "oos_rmse"
    error_model : str, default="hac"
        Error model: "hac" for HAC standard errors or "arimax" for ARIMAX.
    arima_order : tuple of (int, int, int), optional
        ARIMA order (p, d, q) when error_model="arimax". Required if error_model="arimax".

    Returns
    -------
    dict
        Dictionary with keys:
        - "best_saturation": Best saturation transform object
        - "best_adstock": Best adstock transform object
        - "best_score": Best RMSE achieved
        - "best_params": Dictionary of best parameters
        - "optimization_result": scipy OptimizeResult object

    Examples
    --------
    .. code-block:: python

        result = estimate_transform_params_optimize(
            data=df,
            y_column="water_consumption",
            treatment_name="comm_intensity",
            base_formula="1 + t + temperature + rainfall",
            saturation_type="hill",
            saturation_bounds={"slope": (0.5, 5.0), "kappa": (2, 10)},
            adstock_bounds={"half_life": (1, 10)},
            initial_params={"slope": 2.0, "kappa": 5.0, "half_life": 4.0},
        )
        print(f"Optimized RMSE: {result['best_score']:.2f}")

    Notes
    -----
    This method uses continuous optimization which can be faster than grid search
    but may find local optima. Good initial parameters can improve results.
    Consider running grid search first to find good starting points.
    """
    if metric != "rmse":
        raise NotImplementedError(f"Metric '{metric}' not yet implemented. Use 'rmse'.")

    # Determine parameter names and bounds
    sat_param_names = (
        list(saturation_bounds.keys()) if saturation_bounds is not None else []
    )
    adstock_param_names = (
        list(adstock_bounds.keys()) if adstock_bounds is not None else []
    )
    all_param_names = sat_param_names + adstock_param_names

    bounds_list = []
    if saturation_bounds is not None:
        bounds_list.extend([saturation_bounds[k] for k in sat_param_names])
    if adstock_bounds is not None:
        bounds_list.extend([adstock_bounds[k] for k in adstock_param_names])

    # Set initial parameters
    if initial_params is None:
        # Use midpoint of bounds
        initial_params = {}
        all_bounds = {}
        if saturation_bounds is not None:
            all_bounds.update(saturation_bounds)
        if adstock_bounds is not None:
            all_bounds.update(adstock_bounds)
        for k, (lo, hi) in all_bounds.items():
            initial_params[k] = (lo + hi) / 2

    x0 = np.array([initial_params[k] for k in all_param_names])

    # Define objective function
    def objective(params_array):
        """Objective function: returns RMSE for given parameters."""
        # Unpack parameters
        param_dict = dict(zip(all_param_names, params_array))
        sat_kwargs = {k: param_dict[k] for k in sat_param_names}
        adstock_kwargs = {k: param_dict[k] for k in adstock_param_names}

        # Create saturation transform object or None
        if saturation_type is not None and len(sat_kwargs) > 0:
            if saturation_type == "hill":
                saturation = HillSaturation(**sat_kwargs)
            elif saturation_type == "logistic":
                saturation = LogisticSaturation(**sat_kwargs)
            elif saturation_type == "michaelis_menten":
                saturation = MichaelisMentenSaturation(**sat_kwargs)
            else:
                raise ValueError(f"Unknown saturation type: {saturation_type}")
        else:
            saturation = None

        # Create adstock transform object or None
        if len(adstock_kwargs) > 0:
            # Adstock always uses half_life, add defaults if not provided
            if "l_max" not in adstock_kwargs:
                adstock_kwargs["l_max"] = 12
            if "normalize" not in adstock_kwargs:
                adstock_kwargs["normalize"] = True
            adstock = GeometricAdstock(**adstock_kwargs)
        else:
            adstock = None

        # Fit OLS and return RMSE
        try:
            score, _ = _fit_ols_with_transforms(
                data=data,
                y_column=y_column,
                treatment_name=treatment_name,
                base_formula=base_formula,
                saturation=saturation,
                adstock=adstock,
                lag=None,
                hac_maxlags=hac_maxlags,
                error_model=error_model,
                arima_order=arima_order,
            )
            return score
        except Exception as e:
            # If fitting fails, return large penalty
            print(f"Optimization failed at params {param_dict}: {e}")
            return 1e10

    # Run optimization
    opt_result = minimize(
        objective,
        x0=x0,
        method=method,
        bounds=bounds_list,
        options={"disp": False},
    )

    if not opt_result.success:
        print(f"Warning: Optimization did not converge. Message: {opt_result.message}")

    # Extract best parameters
    best_params_array = opt_result.x
    best_params = dict(zip(all_param_names, best_params_array))
    best_score = opt_result.fun

    # Create best transform objects
    sat_kwargs = {k: best_params[k] for k in sat_param_names}
    adstock_kwargs = {k: best_params[k] for k in adstock_param_names}

    # Create best saturation transform or None
    if saturation_type is not None and len(sat_kwargs) > 0:
        if saturation_type == "hill":
            best_saturation = HillSaturation(**sat_kwargs)
        elif saturation_type == "logistic":
            best_saturation = LogisticSaturation(**sat_kwargs)
        elif saturation_type == "michaelis_menten":
            best_saturation = MichaelisMentenSaturation(**sat_kwargs)
        else:
            raise ValueError(f"Unknown saturation type: {saturation_type}")
    else:
        best_saturation = None

    # Create best adstock transform or None
    if len(adstock_kwargs) > 0:
        # Add defaults for adstock if not optimized
        if "l_max" not in adstock_kwargs:
            adstock_kwargs["l_max"] = 12
        if "normalize" not in adstock_kwargs:
            adstock_kwargs["normalize"] = True
        best_adstock = GeometricAdstock(**adstock_kwargs)
    else:
        best_adstock = None

    return {
        "best_saturation": best_saturation,
        "best_adstock": best_adstock,
        "best_score": best_score,
        "best_params": best_params,
        "optimization_result": opt_result,
        "saturation_type": saturation_type,
        "coef_constraint": coef_constraint,
    }
