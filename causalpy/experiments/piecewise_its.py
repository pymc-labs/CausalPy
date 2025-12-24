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
Piecewise Interrupted Time Series Analysis (Segmented Regression)
"""

import warnings
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.base import RegressorMixin

from causalpy.custom_exceptions import BadIndexException, DataException
from causalpy.plot_utils import plot_xY
from causalpy.pymc_models import PyMCModel
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

    The model specification is:

    .. math::

        y_t = \\beta_0 + \\beta_1 t + \\sum_{k=1}^{K} (\\beta_{2k} I_k(t) + \\beta_{3k} R_k(t)) + \\epsilon_t

    Where:
    - :math:`\\beta_0` is the baseline intercept
    - :math:`\\beta_1` is the baseline slope
    - :math:`I_k(t) = 1[t \\geq T_k]` is a step function for level change at interruption k
    - :math:`R_k(t) = \\max(0, t - T_k)` is a ramp function for slope change at interruption k
    - :math:`T_k` are the known interruption times

    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame containing the time series data.
    outcome : str
        Column name for the outcome variable.
    time : str
        Column name for the time variable. Can be numeric or datetime.
    interruption_times : list
        List of known interruption times. Must be sorted and within data range.
        Type should match the time column (int/float or pd.Timestamp).
    include_level_change : bool, default=True
        Whether to include step functions (immediate level changes) at interruptions.
    include_slope_change : bool, default=True
        Whether to include ramp functions (slope changes) at interruptions.
    controls : list[str], optional
        List of control variable column names to include in the model.
    model : PyMCModel or RegressorMixin, optional
        A PyMC (Bayesian) or sklearn (OLS) model. If None, defaults to a PyMC
        LinearRegression model.
    **kwargs
        Additional keyword arguments passed to the model.

    Attributes
    ----------
    interruption_times : list
        The known interruption times.
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
    ...     outcome="y",
    ...     time="t",
    ...     interruption_times=[50],
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={"random_seed": 42, "progressbar": False}
    ...     ),
    ... )

    Notes
    -----
    The counterfactual is computed by setting all interruption terms to zero,
    representing what would have happened without the interventions.

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
        outcome: str,
        time: str,
        interruption_times: list[int | float | pd.Timestamp],
        include_level_change: bool = True,
        include_slope_change: bool = True,
        controls: list[str] | None = None,
        model: PyMCModel | RegressorMixin | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(model=model)

        # Store configuration
        self.outcome = outcome
        self.time_col = time
        self.interruption_times = list(interruption_times)
        self.include_level_change = include_level_change
        self.include_slope_change = include_slope_change
        self.controls = controls or []
        self.data = data.copy()

        # Input validation
        self._validate_inputs()

        # Rename the index to "obs_ind" for consistency
        self.data.index.name = "obs_ind"

        # Generate design matrix with step and ramp features
        self.X, self.y, self.labels = self._generate_design_matrix()

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
        """Validate input data and configuration."""
        # Check outcome column exists
        if self.outcome not in self.data.columns:
            raise DataException(f"Outcome column '{self.outcome}' not found in data.")

        # Check time column exists
        if self.time_col not in self.data.columns:
            raise DataException(f"Time column '{self.time_col}' not found in data.")

        # Check control columns exist
        for ctrl in self.controls:
            if ctrl not in self.data.columns:
                raise DataException(f"Control column '{ctrl}' not found in data.")

        # Check at least one change type is enabled
        if not self.include_level_change and not self.include_slope_change:
            raise ValueError(
                "At least one of include_level_change or include_slope_change must be True."
            )

        # Check interruption times is non-empty
        if not self.interruption_times:
            raise ValueError("interruption_times must contain at least one time point.")

        # Check interruption times are sorted
        if self.interruption_times != sorted(self.interruption_times):
            raise ValueError("interruption_times must be sorted in ascending order.")

        # Validate time types match
        time_values = self.data[self.time_col]
        is_datetime_time = pd.api.types.is_datetime64_any_dtype(time_values)
        first_interrupt = self.interruption_times[0]
        is_datetime_interrupt = isinstance(first_interrupt, pd.Timestamp)

        if is_datetime_time and not is_datetime_interrupt:
            raise BadIndexException(
                "If time column is datetime, interruption_times must be pd.Timestamp."
            )
        if not is_datetime_time and is_datetime_interrupt:
            raise BadIndexException(
                "If time column is numeric, interruption_times must not be pd.Timestamp."
            )

        # Check interruption times are within data range
        time_min = time_values.min()
        time_max = time_values.max()
        for t_k in self.interruption_times:
            if t_k < time_min or t_k > time_max:
                raise ValueError(
                    f"Interruption time {t_k} is outside data range [{time_min}, {time_max}]."
                )

        # Warn about closely spaced interruptions (potential collinearity)
        if len(self.interruption_times) > 1:
            if is_datetime_time:
                time_range_val = (
                    pd.Timestamp(time_max) - pd.Timestamp(time_min)
                ).total_seconds()
            else:
                time_range_val = float(time_max) - float(time_min)
            for i in range(1, len(self.interruption_times)):
                if is_datetime_time:
                    diff_val = (
                        pd.Timestamp(self.interruption_times[i])
                        - pd.Timestamp(self.interruption_times[i - 1])
                    ).total_seconds()
                else:
                    diff_val = float(self.interruption_times[i]) - float(
                        self.interruption_times[i - 1]
                    )
                if diff_val < time_range_val * 0.05:  # Less than 5% of total range
                    warnings.warn(
                        f"Interruption times {self.interruption_times[i - 1]} and "
                        f"{self.interruption_times[i]} are very close together. "
                        "This may cause collinearity issues.",
                        UserWarning,
                        stacklevel=2,
                    )

    def _generate_design_matrix(
        self,
    ) -> tuple[xr.DataArray, xr.DataArray, list[str]]:
        """
        Generate the design matrix with intercept, time, step/ramp features,
        and optional control variables.

        Returns
        -------
        X : xr.DataArray
            Design matrix with shape (n_obs, n_features).
        y : xr.DataArray
            Outcome variable with shape (n_obs, 1).
        labels : list[str]
            Names of the columns in X.
        """
        n_obs = len(self.data)
        time_values = self.data[self.time_col].values

        # Convert datetime to numeric for model fitting
        time_numeric: np.ndarray
        self._time_origin: pd.Timestamp | None
        if pd.api.types.is_datetime64_any_dtype(time_values):
            dt_index = pd.to_datetime(time_values)
            time_numeric = (dt_index - dt_index.min()).total_seconds().values / (
                24 * 3600
            )  # Convert to days
            self._time_is_datetime = True
            self._time_origin = dt_index.min()
        else:
            time_numeric = np.asarray(time_values, dtype=float)
            self._time_is_datetime = False
            self._time_origin = None

        # Store numeric time for later use
        self._time_numeric = time_numeric

        # Build design matrix columns
        columns = {"Intercept": np.ones(n_obs), "time": time_numeric}
        labels = ["Intercept", "time"]

        # Add step and ramp features for each interruption
        for k, t_k in enumerate(self.interruption_times):
            if self._time_is_datetime and self._time_origin is not None:
                t_k_numeric = (
                    pd.Timestamp(t_k) - self._time_origin
                ).total_seconds() / (24 * 3600)
            else:
                t_k_numeric = float(t_k)

            if self.include_level_change:
                col_name = f"level_{k}"
                columns[col_name] = (time_numeric >= t_k_numeric).astype(float)
                labels.append(col_name)

            if self.include_slope_change:
                col_name = f"slope_{k}"
                columns[col_name] = np.maximum(0, time_numeric - t_k_numeric)
                labels.append(col_name)

        # Add control variables
        for ctrl in self.controls:
            columns[ctrl] = np.asarray(self.data[ctrl].values, dtype=float)
            labels.append(ctrl)

        # Create design matrix
        X_array: np.ndarray = np.column_stack(
            [np.asarray(columns[label]) for label in labels]
        )

        X = xr.DataArray(
            X_array,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": np.arange(n_obs),
                "coeffs": labels,
            },
        )

        # Create outcome array
        y_array: np.ndarray = np.asarray(self.data[self.outcome].values).reshape(-1, 1)
        y = xr.DataArray(
            y_array,
            dims=["obs_ind", "treated_units"],
            coords={
                "obs_ind": np.arange(n_obs),
                "treated_units": ["unit_0"],
            },
        )

        return X, y, labels

    def _get_interruption_column_indices(self) -> list[int]:
        """Get indices of columns related to interruptions (for counterfactual)."""
        indices = []
        for i, label in enumerate(self.labels):
            if label.startswith("level_") or label.startswith("slope_"):
                indices.append(i)
        return indices

    def _compute_counterfactual_and_effects(self) -> None:
        """
        Compute the counterfactual (no intervention) and causal effects.

        The counterfactual is computed by setting interruption terms to zero.
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

    def summary(self, round_to: int | None = None) -> None:
        """Print summary of main results and model coefficients.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2.
        """
        print(f"{self.expt_type:=^80}")
        print(f"Outcome: {self.outcome}")
        print(f"Time column: {self.time_col}")
        print(f"Interruption times: {self.interruption_times}")
        print(f"Level change: {self.include_level_change}")
        print(f"Slope change: {self.include_slope_change}")
        if self.controls:
            print(f"Controls: {self.controls}")
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
        ax[0].set(title=title_str, ylabel=self.outcome)

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
        ax[0].set(title=title_str, ylabel=self.outcome)

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
                self.outcome: self.y.isel(treated_units=0).values,
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
                self.outcome: self.y.values.flatten(),
                "fitted": np.squeeze(self.y_pred),
                "counterfactual": np.squeeze(self.y_counterfactual),
                "effect": self.effect,
                "cumulative_effect": self.cumulative_effect,
            }
        )

        self.plot_data = result
        return result
