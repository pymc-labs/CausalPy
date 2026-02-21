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
Piecewise Interrupted Time Series Analysis (Segmented Regression)
"""

import re
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from patsy import dmatrices
from sklearn.base import RegressorMixin

from causalpy.custom_exceptions import FormulaException
from causalpy.plot_utils import plot_xY
from causalpy.pymc_models import PyMCModel
from causalpy.reporting import EffectSummary
from causalpy.transforms import ramp, step  # noqa: F401
from causalpy.utils import round_num

from .base import BaseExperiment

LEGEND_FONT_SIZE = 12


class PiecewiseITS(BaseExperiment):
    """
    Piecewise Interrupted Time Series (Segmented Regression) experiment.

    This class implements segmented-regression / piecewise linear models for
    Interrupted Time Series analysis with **known** interruption dates. Unlike
    the standard :class:`InterruptedTimeSeries` which fits a model to pre-intervention
    data and forecasts a counterfactual, `PiecewiseITS` fits **one model to the
    full time series** and estimates explicit level and/or slope changes at each
    interruption.

    The model uses patsy formulas with custom `step()` and `ramp()` transforms:

    - ``step(time, threshold)``: Creates a binary indicator (1 if time >= threshold)
      for level changes
    - ``ramp(time, threshold)``: Creates a ramp function (max(0, time - threshold))
      for slope changes

    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame containing the time series data.
    formula : str
        A patsy formula specifying the model. Must include at least one
        ``step()`` or ``ramp()`` term, and all such terms must use the same
        time variable. Example:
        ``"y ~ 1 + t + step(t, 50) + ramp(t, 50)"``
    model : PyMCModel or RegressorMixin, optional
        A PyMC (Bayesian) or sklearn (OLS) model. If None, defaults to a PyMC
        LinearRegression model.
    **kwargs
        Additional keyword arguments passed to the model.

    Attributes
    ----------
    formula : str
        The patsy formula used for the model.
    interruption_times : list
        Canonicalized interruption thresholds extracted from the formula.
    labels : list[str]
        Names of all coefficients in the design matrix.
    effect : xr.DataArray or np.ndarray
        Pointwise causal effect (observed - counterfactual).
    cumulative_effect : xr.DataArray or np.ndarray
        Cumulative causal effect over time.

    Examples
    --------
    >>> import causalpy as cp
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Generate simple piecewise data
    >>> np.random.seed(42)
    >>> t = np.arange(100)
    >>> y = (
    ...     10
    ...     + 0.1 * t
    ...     + 5 * (t >= 50)
    ...     + 0.2 * np.maximum(0, t - 50)
    ...     + np.random.normal(0, 1, 100)
    ... )
    >>> df = pd.DataFrame({"t": t, "y": y})
    >>> result = cp.PiecewiseITS(
    ...     df,
    ...     formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={"random_seed": 42, "progressbar": False}
    ...     ),
    ... )

    **Different effects per intervention:**

    >>> # Level change only at t=50, level + slope change at t=100
    >>> result = cp.PiecewiseITS(
    ...     df,
    ...     formula="y ~ 1 + t + step(t, 50) + step(t, 100) + ramp(t, 100)",
    ...     model=...,
    ... )  # doctest: +SKIP

    **With datetime thresholds:**

    >>> df["date"] = pd.date_range("2020-01-01", periods=100, freq="D")
    >>> result = cp.PiecewiseITS(
    ...     df,
    ...     formula="y ~ 1 + date + step(date, '2020-02-20') + ramp(date, '2020-02-20')",
    ...     model=...,
    ... )  # doctest: +SKIP

    Notes
    -----
    The counterfactual is computed by setting all step/ramp terms to zero,
    representing what would have happened without the interventions.

    The `step` and `ramp` transforms are patsy stateful transforms that handle
    both numeric and datetime time columns. For datetime, thresholds can be
    specified as strings (e.g., '2020-06-01') or pd.Timestamp objects.

    References
    ----------
    - Wagner AK, et al. (2002). Segmented regression analysis of interrupted
      time series studies in medication use research. Journal of Clinical
      Pharmacy and Therapeutics.
    - Lopez Bernal J, et al. (2017). Interrupted time series regression for
      the evaluation of public health interventions: a tutorial. Int J Epidemiol.
    """

    expt_type = "Piecewise Interrupted Time Series"
    supports_ols = True
    supports_bayes = True

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        model: PyMCModel | RegressorMixin | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(model=model)

        # Store configuration
        self.formula = formula
        self.data = data.copy()

        # Rename the index to "obs_ind" for consistency
        self.data.index.name = "obs_ind"

        # Input validation
        self._validate_inputs()

        # Parse and validate step/ramp terms before any downstream logic.
        self._step_ramp_terms = self._parse_step_ramp_terms()
        self.time_col = self._extract_time_column(self._step_ramp_terms)
        self.interruption_times = self._extract_and_canonicalize_interruption_times(
            self._step_ramp_terms, self.time_col
        )

        # Parse formula with patsy (step and ramp are available in namespace)
        y, X = dmatrices(formula, self.data)
        self.outcome_variable_name = y.design_info.column_names[0]
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = list(X.design_info.column_names)

        # Convert to numpy arrays
        y_array = np.asarray(y)
        X_array = np.asarray(X)

        n_obs = X_array.shape[0]

        # Convert to xarray DataArrays
        self.X = xr.DataArray(
            X_array,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": np.arange(n_obs),
                "coeffs": self.labels,
            },
        )

        self.y = xr.DataArray(
            y_array,
            dims=["obs_ind", "treated_units"],
            coords={
                "obs_ind": np.arange(n_obs),
                "treated_units": ["unit_0"],
            },
        )

        # Track which columns are interruption-related (for counterfactual)
        self._interruption_cols = self._get_interruption_column_indices()

        # Fit the model to the full time series
        if isinstance(self.model, PyMCModel):
            COORDS: dict[str, Any] = {
                "coeffs": self.labels,
                "obs_ind": np.arange(self.X.shape[0]),
                "treated_units": ["unit_0"],
            }
            self.model.fit(X=self.X, y=self.y, coords=COORDS)
        elif isinstance(self.model, RegressorMixin):
            if hasattr(self.model, "fit_intercept"):
                self.model.fit_intercept = False
            self.model.fit(X=self.X, y=self.y.isel(treated_units=0))
        else:
            raise ValueError("Model type not recognized")

        # Compute predictions (fitted values)
        self.y_pred = self.model.predict(X=self.X)

        # Score the model fit
        if isinstance(self.model, PyMCModel):
            self.score = self.model.score(X=self.X, y=self.y)
        elif isinstance(self.model, RegressorMixin):
            self.score = self.model.score(X=self.X, y=self.y.isel(treated_units=0))

        # Compute counterfactual and effects
        self._compute_counterfactual_and_effects()

    def _validate_inputs(self) -> None:
        """Validate input data and formula."""
        # Check formula contains at least one step() or ramp() term
        if "step(" not in self.formula and "ramp(" not in self.formula:
            raise FormulaException(
                "Formula must contain at least one step() or ramp() term. "
                "Example: 'y ~ 1 + t + step(t, 50) + ramp(t, 50)'"
            )

    def _parse_step_ramp_terms(self) -> list[dict[str, str]]:
        """Parse step/ramp terms into structured metadata."""
        pattern = r"(step|ramp)\s*\(\s*(\w+)\s*,\s*([^)]+?)\s*\)"
        matches = re.findall(pattern, self.formula)
        return [
            {"transform": transform, "variable": variable, "raw_threshold": threshold}
            for transform, variable, threshold in matches
        ]

    def _extract_time_column(self, terms: list[dict[str, str]]) -> str:
        """Extract and validate the unique time column used by step/ramp calls."""
        variables = {term["variable"] for term in terms}
        if len(variables) != 1:
            raise FormulaException(
                "All step()/ramp() terms must use exactly one time variable in this "
                "version of PiecewiseITS. Mixed variables are not yet supported."
            )
        time_col = next(iter(variables))
        if time_col not in self.data.columns:
            raise FormulaException(
                f"Time variable '{time_col}' from step()/ramp() terms is not present "
                "in the input data."
            )
        return time_col

    def _extract_and_canonicalize_interruption_times(
        self, terms: list[dict[str, str]], time_col: str
    ) -> list[int | float | pd.Timestamp]:
        """Extract thresholds and canonicalize them to numeric or Timestamp."""
        time_values = self.data[time_col]
        is_datetime = pd.api.types.is_datetime64_any_dtype(time_values)
        is_numeric = pd.api.types.is_numeric_dtype(time_values)

        if not is_datetime and not is_numeric:
            raise FormulaException(
                f"Time variable '{time_col}' must be numeric or datetime-like to be "
                "used with step()/ramp()."
            )

        canonical_thresholds: list[int | float | pd.Timestamp] = []
        for term in terms:
            raw_threshold = term["raw_threshold"].strip().strip("'\"")

            if is_datetime:
                try:
                    value: int | float | pd.Timestamp = pd.to_datetime(
                        raw_threshold, errors="raise"
                    )
                except (TypeError, ValueError) as exc:
                    raise FormulaException(
                        f"Invalid datetime threshold '{term['raw_threshold']}' for "
                        f"time variable '{time_col}'."
                    ) from exc
            else:
                try:
                    if re.fullmatch(r"[-+]?\d+", raw_threshold):
                        value = int(raw_threshold)
                    else:
                        value = float(raw_threshold)
                except ValueError as exc:
                    raise FormulaException(
                        f"Invalid numeric threshold '{term['raw_threshold']}' for "
                        f"time variable '{time_col}'."
                    ) from exc

            if value not in canonical_thresholds:
                canonical_thresholds.append(value)

        return canonical_thresholds

    def _get_interruption_column_indices(self) -> list[int]:
        """Get indices of columns related to interruptions (step/ramp terms)."""
        indices = []
        for i, label in enumerate(self.labels):
            # Patsy labels step/ramp terms like "step(t, 50)" or "ramp(t, 50)"
            if "step(" in label or "ramp(" in label:
                indices.append(i)
        return indices

    def _compute_counterfactual_and_effects(self) -> None:
        """
        Compute the counterfactual (no intervention) and causal effects.

        The counterfactual is computed by setting step/ramp terms to zero.
        Also creates post_impact, datapost, and post_pred attributes for
        compatibility with effect_summary() from BaseExperiment.
        """
        # Create design matrix for counterfactual (zero out interruption columns)
        X_cf = self.X.copy()
        for idx in self._interruption_cols:
            X_cf[:, idx] = 0

        # Compute counterfactual predictions
        if isinstance(self.model, PyMCModel):
            self.y_counterfactual = self.model.predict(X=X_cf)

            # Extract mu for fitted and counterfactual
            y_pred_mu = self.y_pred["posterior_predictive"]["mu"]
            y_cf_mu = self.y_counterfactual["posterior_predictive"]["mu"]

            # Handle treated_units dimension if present
            if "treated_units" in y_pred_mu.dims:
                y_pred_mu = y_pred_mu.isel(treated_units=0)
            if "treated_units" in y_cf_mu.dims:
                y_cf_mu = y_cf_mu.isel(treated_units=0)

            # Compute effect as fitted - counterfactual
            self.effect = y_pred_mu - y_cf_mu

            # Cumulative effect
            self.cumulative_effect = self.effect.cumsum(dim="obs_ind")

        elif isinstance(self.model, RegressorMixin):
            self.y_counterfactual = self.model.predict(X=X_cf)

            # Compute effect
            self.effect = np.squeeze(self.y_pred) - np.squeeze(self.y_counterfactual)

            # Cumulative effect
            self.cumulative_effect = np.cumsum(self.effect)

        # Create compatibility attributes for effect_summary() from BaseExperiment
        # These represent the post-intervention portion (after the first interruption)
        self._create_post_intervention_attributes()

    def _create_post_intervention_attributes(self) -> None:
        """
        Create post_impact, datapost, and post_pred attributes for effect_summary().

        These attributes make PiecewiseITS compatible with the effect_summary()
        method inherited from BaseExperiment, which expects ITS-like attributes.

        The "post-intervention" portion is defined as all observations at or after
        the first interruption time.
        """
        if not self.interruption_times:
            # No interruptions - all data is "pre-intervention"
            # Create empty post-intervention attributes
            self.datapost = self.data.iloc[0:0]  # Empty DataFrame
            return

        # Get the first interruption time
        first_interruption = self.interruption_times[0]
        time_col = self.time_col

        # Post-intervention = time >= first_interruption (inclusive)
        post_mask = self.data[time_col] >= first_interruption

        # Create datapost - the post-intervention data
        self.datapost = self.data[post_mask].copy()
        self.datapost.index.name = "obs_ind"

        # Get indices for post-intervention period
        post_indices = np.where(np.asarray(post_mask))[0]

        # Create post_impact - the effects after the first interruption
        if isinstance(self.model, PyMCModel):
            # For PyMC models, effect is an xarray.DataArray
            # Select using obs_ind coordinate
            self.post_impact = self.effect.isel(obs_ind=post_indices)

            # Create post_pred - counterfactual predictions for post-intervention
            # This needs to be an InferenceData-like object for extract_counterfactual
            y_cf_mu = self.y_counterfactual["posterior_predictive"]["mu"]
            if "treated_units" in y_cf_mu.dims:
                y_cf_mu = y_cf_mu.isel(treated_units=0)
            post_cf_mu = y_cf_mu.isel(obs_ind=post_indices)

            # Update the coordinates to match datapost.index
            post_cf_mu = post_cf_mu.assign_coords(obs_ind=self.datapost.index)

            # Create an InferenceData-like dict structure
            self.post_pred = {
                "posterior_predictive": {"mu": post_cf_mu},
            }

            # Update post_impact coordinates to match datapost.index
            self.post_impact = self.post_impact.assign_coords(
                obs_ind=self.datapost.index
            )

        elif isinstance(self.model, RegressorMixin):
            # For OLS models, effect and counterfactual are numpy arrays
            self.post_impact = self.effect[post_indices]
            self.post_pred = np.squeeze(self.y_counterfactual)[post_indices]

    def summary(self, round_to: int | None = None) -> None:
        """Print summary of main results and model coefficients.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2.
        """
        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print(f"Interruption times: {self.interruption_times}")
        self.print_coefficients(round_to)

    def _bayesian_plot(
        self, round_to: int | None = 2, **kwargs: dict[str, Any]
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plot the results for Bayesian models.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals for rounding. Defaults to 2.

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure.
        ax : list[plt.Axes]
            List of axes objects.
        """
        time_values = self.data[self.time_col].values

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

        # TOP PLOT: Observed, Fitted, and Counterfactual
        # Observed data
        (h_obs,) = ax[0].plot(
            time_values,
            self.y.isel(treated_units=0),
            "k.",
            label="Observations",
        )

        # Fitted values (mu)
        y_pred_mu = self.y_pred["posterior_predictive"]["mu"]
        if "treated_units" in y_pred_mu.dims:
            y_pred_mu = y_pred_mu.isel(treated_units=0)
        h_line_fit, h_patch_fit = plot_xY(
            time_values,
            y_pred_mu,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C0"},
        )

        # Counterfactual
        y_cf_mu = self.y_counterfactual["posterior_predictive"]["mu"]
        if "treated_units" in y_cf_mu.dims:
            y_cf_mu = y_cf_mu.isel(treated_units=0)
        h_line_cf, h_patch_cf = plot_xY(
            time_values,
            y_cf_mu,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C1"},
        )

        # Title with R^2
        r2_val = None
        try:
            if isinstance(self.score, pd.Series):
                if "unit_0_r2" in self.score.index:
                    r2_val = self.score["unit_0_r2"]
                elif "r2" in self.score.index:
                    r2_val = self.score["r2"]
        except Exception:
            pass

        title_str = "Piecewise ITS: Bayesian $R^2$"
        if r2_val is not None:
            title_str += f" = {round_num(r2_val, round_to)}"
        ax[0].set(title=title_str, ylabel=self.outcome_variable_name)

        handles = [h_obs, (h_line_fit, h_patch_fit), (h_line_cf, h_patch_cf)]
        labels_legend = ["Observations", "Fitted", "Counterfactual"]

        # MIDDLE PLOT: Causal Effect
        plot_xY(
            time_values,
            self.effect,
            ax=ax[1],
            plot_hdi_kwargs={"color": "C2"},
        )
        ax[1].axhline(y=0, c="k", linestyle="--", alpha=0.5)
        ax[1].fill_between(
            time_values,
            y1=self.effect.mean(dim=["chain", "draw"]).values,
            alpha=0.25,
            color="C2",
        )
        ax[1].set(title="Causal Effect", ylabel="Effect")

        # BOTTOM PLOT: Cumulative Effect
        plot_xY(
            time_values,
            self.cumulative_effect,
            ax=ax[1 + 1],
            plot_hdi_kwargs={"color": "C3"},
        )
        ax[2].axhline(y=0, c="k", linestyle="--", alpha=0.5)
        ax[2].set(title="Cumulative Causal Effect", ylabel="Cumulative Effect")

        # Add vertical lines for interruptions
        for i, t_k in enumerate(self.interruption_times):
            for a in ax:
                a.axvline(
                    x=t_k,
                    ls="-",
                    lw=2,
                    color="red",
                    alpha=0.7,
                    label=f"Interruption {i}" if a == ax[0] else None,
                )
            handles.append(plt.Line2D([0], [0], color="red", lw=2))
            labels_legend.append(f"Interruption {i}")

        ax[0].legend(handles=handles, labels=labels_legend, fontsize=LEGEND_FONT_SIZE)

        plt.tight_layout()
        return fig, ax

    def _ols_plot(
        self, round_to: int | None = 2, **kwargs: dict[str, Any]
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plot the results for OLS models.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals for rounding. Defaults to 2.

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure.
        ax : list[plt.Axes]
            List of axes objects.
        """
        time_values = self.data[self.time_col].values

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

        # TOP PLOT: Observed, Fitted, and Counterfactual
        ax[0].plot(time_values, self.y.values, "k.", label="Observations")
        ax[0].plot(time_values, self.y_pred, "C0-", label="Fitted", linewidth=2)
        ax[0].plot(
            time_values,
            self.y_counterfactual,
            "C1--",
            label="Counterfactual",
            linewidth=2,
        )

        title_str = f"Piecewise ITS: $R^2$ = {round_num(float(self.score), round_to)}"
        ax[0].set(title=title_str, ylabel=self.outcome_variable_name)

        # MIDDLE PLOT: Causal Effect
        ax[1].plot(time_values, self.effect, "C2-", linewidth=2)
        ax[1].fill_between(time_values, y1=self.effect, alpha=0.25, color="C2")
        ax[1].axhline(y=0, c="k", linestyle="--", alpha=0.5)
        ax[1].set(title="Causal Effect", ylabel="Effect")

        # BOTTOM PLOT: Cumulative Effect
        ax[2].plot(time_values, self.cumulative_effect, "C3-", linewidth=2)
        ax[2].axhline(y=0, c="k", linestyle="--", alpha=0.5)
        ax[2].set(title="Cumulative Causal Effect", ylabel="Cumulative Effect")

        # Add vertical lines for interruptions
        for i, t_k in enumerate(self.interruption_times):
            for a in ax:
                a.axvline(
                    x=t_k,
                    ls="-",
                    lw=2,
                    color="red",
                    alpha=0.7,
                    label=f"Interruption {i}" if a == ax[0] else None,
                )

        ax[0].legend(fontsize=LEGEND_FONT_SIZE)

        plt.tight_layout()
        return fig, ax

    def get_plot_data_bayesian(self, hdi_prob: float = 0.94) -> pd.DataFrame:
        """
        Recover the data of the experiment along with prediction and effect information.

        Parameters
        ----------
        hdi_prob : float, default=0.94
            Probability for the highest density interval.

        Returns
        -------
        pd.DataFrame
            DataFrame containing observed data, predictions, and effects.
        """
        hdi_pct = int(round(hdi_prob * 100))

        # Get time values
        time_values = self.data[self.time_col].values

        # Extract predictions
        y_pred_mu = self.y_pred["posterior_predictive"]["mu"]
        if "treated_units" in y_pred_mu.dims:
            y_pred_mu = y_pred_mu.isel(treated_units=0)

        y_cf_mu = self.y_counterfactual["posterior_predictive"]["mu"]
        if "treated_units" in y_cf_mu.dims:
            y_cf_mu = y_cf_mu.isel(treated_units=0)

        # Helper to extract HDI bounds from az.hdi() result (which returns a Dataset)
        def _get_hdi_bounds(
            hdi_result: xr.Dataset,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Extract lower and upper bounds from az.hdi result."""
            data_var = list(hdi_result.data_vars)[0]
            hdi_data = hdi_result[data_var]
            lower = hdi_data.sel(hdi="lower").values.flatten()
            upper = hdi_data.sel(hdi="higher").values.flatten()
            return lower, upper

        # Compute means and HDIs
        fitted_mean = y_pred_mu.mean(dim=["chain", "draw"]).values
        fitted_hdi = az.hdi(y_pred_mu, hdi_prob=hdi_prob)
        fitted_lower, fitted_upper = _get_hdi_bounds(fitted_hdi)

        cf_mean = y_cf_mu.mean(dim=["chain", "draw"]).values
        cf_hdi = az.hdi(y_cf_mu, hdi_prob=hdi_prob)
        cf_lower, cf_upper = _get_hdi_bounds(cf_hdi)

        effect_mean = self.effect.mean(dim=["chain", "draw"]).values
        effect_hdi = az.hdi(self.effect, hdi_prob=hdi_prob)
        effect_lower, effect_upper = _get_hdi_bounds(effect_hdi)

        cum_effect_mean = self.cumulative_effect.mean(dim=["chain", "draw"]).values
        cum_effect_hdi = az.hdi(self.cumulative_effect, hdi_prob=hdi_prob)
        cum_effect_lower, cum_effect_upper = _get_hdi_bounds(cum_effect_hdi)

        # Build DataFrame
        result = pd.DataFrame(
            {
                self.time_col: time_values,
                self.outcome_variable_name: self.y.isel(treated_units=0).values,
                "fitted": fitted_mean,
                f"fitted_hdi_lower_{hdi_pct}": fitted_lower,
                f"fitted_hdi_upper_{hdi_pct}": fitted_upper,
                "counterfactual": cf_mean,
                f"counterfactual_hdi_lower_{hdi_pct}": cf_lower,
                f"counterfactual_hdi_upper_{hdi_pct}": cf_upper,
                "effect": effect_mean,
                f"effect_hdi_lower_{hdi_pct}": effect_lower,
                f"effect_hdi_upper_{hdi_pct}": effect_upper,
                "cumulative_effect": cum_effect_mean,
                f"cumulative_effect_hdi_lower_{hdi_pct}": cum_effect_lower,
                f"cumulative_effect_hdi_upper_{hdi_pct}": cum_effect_upper,
            }
        )

        self.plot_data = result
        return result

    def get_plot_data_ols(self) -> pd.DataFrame:
        """
        Recover the data of the experiment along with prediction and effect information.

        Returns
        -------
        pd.DataFrame
            DataFrame containing observed data, predictions, and effects.
        """
        time_values = self.data[self.time_col].values

        result = pd.DataFrame(
            {
                self.time_col: time_values,
                self.outcome_variable_name: self.y.values.flatten(),
                "fitted": np.squeeze(self.y_pred),
                "counterfactual": np.squeeze(self.y_counterfactual),
                "effect": self.effect,
                "cumulative_effect": self.cumulative_effect,
            }
        )

        self.plot_data = result
        return result

    def effect_summary(
        self,
        *,
        window: Literal["post"] | tuple | slice = "post",
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        cumulative: bool = True,
        relative: bool = True,
        min_effect: float | None = None,
        treated_unit: str | None = None,
        period: Literal["intervention", "post", "comparison"] | None = None,
        prefix: str = "Post-period",
        **kwargs: Any,
    ) -> EffectSummary:
        """Generate a decision-ready summary of PiecewiseITS causal effects."""
        from causalpy.reporting import (
            _compute_statistics,
            _compute_statistics_ols,
            _extract_counterfactual,
            _extract_window,
            _generate_prose,
            _generate_prose_ols,
            _generate_table,
            _generate_table_ols,
        )

        if period is not None:
            raise ValueError(
                "period is not supported for PiecewiseITS. "
                "Use window to restrict the post-period summary."
            )

        windowed_impact, window_coords = _extract_window(
            self, window, treated_unit=treated_unit
        )
        counterfactual = _extract_counterfactual(
            self, window_coords, treated_unit=treated_unit
        )

        if isinstance(self.model, PyMCModel):
            hdi_prob = 1 - alpha
            stats = _compute_statistics(
                windowed_impact,
                counterfactual,
                hdi_prob=hdi_prob,
                direction=direction,
                cumulative=cumulative,
                relative=relative,
                min_effect=min_effect,
            )
            table = _generate_table(stats, cumulative=cumulative, relative=relative)
            text = _generate_prose(
                stats,
                window_coords,
                alpha=alpha,
                direction=direction,
                cumulative=cumulative,
                relative=relative,
                prefix=prefix,
            )
        else:
            impact_array = np.asarray(windowed_impact)
            counterfactual_array = np.asarray(counterfactual)
            stats = _compute_statistics_ols(
                impact_array,
                counterfactual_array,
                alpha=alpha,
                cumulative=cumulative,
                relative=relative,
            )
            table = _generate_table_ols(stats, cumulative=cumulative, relative=relative)
            text = _generate_prose_ols(
                stats,
                window_coords,
                alpha=alpha,
                cumulative=cumulative,
                relative=relative,
                prefix=prefix,
            )

        return EffectSummary(table=table, text=text)
