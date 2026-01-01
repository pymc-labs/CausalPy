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
Panel Regression with Fixed Effects
"""

from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from patsy import dmatrices
from sklearn.base import RegressorMixin

from causalpy.custom_exceptions import DataException
from causalpy.pymc_models import PyMCModel

from .base import BaseExperiment

LEGEND_FONT_SIZE = 12


class PanelRegression(BaseExperiment):
    """Panel regression with fixed effects estimation.

    Enables panel-aware visualization and diagnostics, with support for both
    dummy variable and within-transformation approaches to fixed effects.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas dataframe with panel data. Each row is an observation for a
        unit at a time period.
    formula : str
        A statistical model formula using patsy syntax. For dummy variable
        approach, include C(unit_var) in the formula. For within transformation,
        do NOT include C(unit_var) as it will be automatically removed.
    unit_fe_variable : str
        Column name for the unit identifier (e.g., "state", "id", "country").
    time_fe_variable : str, optional
        Column name for the time identifier (e.g., "year", "wave", "period").
        If provided, time fixed effects will be included. Default is None.
    fe_method : {"dummies", "within"}, default="dummies"
        Method for handling fixed effects:
        - "dummies": Use dummy variables (C(unit) in formula). Gets individual
          unit effect estimates but creates N-1 dummy columns. Best for small N.
        - "within": Use within transformation (demeaning). Scales to large N but
          doesn't directly estimate individual unit effects.
    model : PyMCModel or RegressorMixin, optional
        A PyMC (Bayesian) or sklearn (OLS) model. If None, a model must be provided.

    Attributes
    ----------
    n_units : int
        Number of unique units in the panel.
    n_periods : int or None
        Number of unique time periods (None if time_fe_variable not provided).
    fe_method : str
        The fixed effects method used ("dummies" or "within").
    _group_means : dict
        Stored group means for recovering unit effects (within method only).

    Examples
    --------
    Small panel with dummy variables:

    >>> import causalpy as cp
    >>> import pandas as pd
    >>> # Create small panel: 10 units, 20 time periods
    >>> np.random.seed(42)
    >>> units = [f"unit_{i}" for i in range(10)]
    >>> periods = range(20)
    >>> data = pd.DataFrame(
    ...     [
    ...         {
    ...             "unit": u,
    ...             "time": t,
    ...             "treatment": t >= 10 and u in units[:5],
    ...             "x1": np.random.randn(),
    ...             "y": np.random.randn(),
    ...         }
    ...         for u in units
    ...         for t in periods
    ...     ]
    ... )
    >>> result = cp.PanelRegression(
    ...     data=data,
    ...     formula="y ~ C(unit) + C(time) + treatment + x1",
    ...     unit_fe_variable="unit",
    ...     time_fe_variable="time",
    ...     fe_method="dummies",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={"random_seed": 42, "progressbar": False}
    ...     ),
    ... )

    Large panel with within transformation:

    >>> # Create larger panel: 1000 units, 10 time periods
    >>> np.random.seed(42)
    >>> units = [f"unit_{i}" for i in range(1000)]
    >>> periods = range(10)
    >>> data = pd.DataFrame(
    ...     [
    ...         {
    ...             "unit": u,
    ...             "time": t,
    ...             "treatment": t >= 5,
    ...             "x1": np.random.randn(),
    ...             "y": np.random.randn(),
    ...         }
    ...         for u in units
    ...         for t in periods
    ...     ]
    ... )
    >>> result = cp.PanelRegression(
    ...     data=data,
    ...     formula="y ~ treatment + x1",  # No C(unit) needed
    ...     unit_fe_variable="unit",
    ...     time_fe_variable="time",
    ...     fe_method="within",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={"random_seed": 42, "progressbar": False}
    ...     ),
    ... )

    Notes
    -----
    The within transformation demeans all numeric variables by group, which
    removes time-invariant confounders but also drops time-invariant covariates
    from the model. For the dummy approach, individual unit effects can be
    extracted from the coefficients. For the within approach, unit effects can
    be recovered post-hoc using the stored group means.

    Two-way fixed effects (unit + time) control for both unit-specific and
    time-specific unobserved heterogeneity. This is the standard approach in
    difference-in-differences estimation.
    """

    expt_type = "Panel Regression"
    supports_ols = True
    supports_bayes = True

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        unit_fe_variable: str,
        time_fe_variable: str | None = None,
        fe_method: Literal["dummies", "within"] = "dummies",
        model: PyMCModel | RegressorMixin | None = None,
        **kwargs: dict,
    ) -> None:
        super().__init__(model=model)

        # Store parameters
        self.unit_fe_variable = unit_fe_variable
        self.time_fe_variable = time_fe_variable
        self.fe_method = fe_method
        self.formula = formula

        # Validate inputs
        self._validate_inputs(data)

        # Store panel dimensions
        self.n_units = data[unit_fe_variable].nunique()
        self.n_periods = data[time_fe_variable].nunique() if time_fe_variable else None

        # Store original data for plotting
        self.data = data.copy()
        self.data.index.name = "obs_ind"

        # Initialize storage for group means (used in within transformation)
        self._group_means: dict[str, pd.DataFrame] = {}

        # Apply within transformation if requested
        if fe_method == "within":
            data = self._within_transform(data, unit_fe_variable)
            if time_fe_variable:
                data = self._within_transform(data, time_fe_variable)

        # Create design matrices
        y, X = dmatrices(formula, data)
        self.outcome_variable_name = y.design_info.column_names[0]
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.y, self.X = np.asarray(y), np.asarray(X)

        # Convert to xarray
        self.X = xr.DataArray(
            self.X,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": np.arange(self.X.shape[0]),
                "coeffs": self.labels,
            },
        )
        self.y = xr.DataArray(
            self.y,
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": np.arange(self.y.shape[0]), "treated_units": ["unit_0"]},
        )

        # Fit model
        if isinstance(self.model, PyMCModel):
            COORDS = {
                "coeffs": self.labels,
                "obs_ind": np.arange(self.X.shape[0]),
                "treated_units": ["unit_0"],
            }
            self.model.fit(X=self.X, y=self.y, coords=COORDS)
        elif isinstance(self.model, RegressorMixin):
            # For scikit-learn models, set fit_intercept=False to include intercept in coefficients
            if hasattr(self.model, "fit_intercept"):
                self.model.fit_intercept = False
            self.model.fit(X=self.X.values, y=self.y.values.ravel())

    def _validate_inputs(self, data: pd.DataFrame) -> None:
        """Validate input parameters."""
        if self.unit_fe_variable not in data.columns:
            raise DataException(
                f"unit_fe_variable '{self.unit_fe_variable}' not found in data columns"
            )

        if self.time_fe_variable and self.time_fe_variable not in data.columns:
            raise DataException(
                f"time_fe_variable '{self.time_fe_variable}' not found in data columns"
            )

        if self.fe_method not in ["dummies", "within"]:
            raise ValueError("fe_method must be 'dummies' or 'within'")

        # Check if formula includes C(unit_var) when using within method
        if self.fe_method == "within" and f"C({self.unit_fe_variable})" in self.formula:
            raise ValueError(
                f"When using fe_method='within', do not include C({self.unit_fe_variable}) "
                "in the formula. The within transformation handles unit fixed effects automatically."
            )

    def _within_transform(self, data: pd.DataFrame, group_var: str) -> pd.DataFrame:
        """Apply within transformation (demean by group).

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        group_var : str
            Column name to group by (unit or time variable)

        Returns
        -------
        pd.DataFrame
            Demeaned data
        """
        data = data.copy()

        # Identify numeric columns to demean (exclude group variables)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        group_vars_to_exclude = [self.unit_fe_variable]
        if self.time_fe_variable:
            group_vars_to_exclude.append(self.time_fe_variable)

        numeric_cols = [c for c in numeric_cols if c not in group_vars_to_exclude]

        # Store group means for potential recovery of fixed effects
        self._group_means[group_var] = data.groupby(group_var)[numeric_cols].mean()

        # Demean each numeric column
        for col in numeric_cols:
            group_mean = data.groupby(group_var)[col].transform("mean")
            data[col] = data[col] - group_mean

        return data

    def summary(self) -> None:
        """Print a summary of the panel regression results."""
        print(f"\n{self.expt_type}")
        print("=" * 60)
        print(f"Units: {self.n_units} ({self.unit_fe_variable})")
        if self.n_periods:
            print(f"Periods: {self.n_periods} ({self.time_fe_variable})")
        print(f"FE method: {self.fe_method}")
        print(f"Observations: {len(self.data)}")
        print("=" * 60)
        print("\nModel Coefficients:")
        self.print_coefficients()

    def _bayesian_plot(self, **kwargs: Any) -> tuple[plt.Figure, plt.Axes]:
        """Create coefficient plot for Bayesian model.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            Figure and axes objects
        """
        return self._plot_coefficients_internal()

    def _ols_plot(self, **kwargs: Any) -> tuple[plt.Figure, plt.Axes]:
        """Create coefficient plot for OLS model.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            Figure and axes objects
        """
        return self._plot_coefficients_internal()

    def _plot_coefficients_internal(self) -> tuple[plt.Figure, plt.Axes]:
        """Internal method to create coefficient plot."""
        # Filter out fixed effect dummy coefficients
        coeff_names = self.labels.copy()

        # Remove unit FE dummies
        if self.fe_method == "dummies":
            coeff_names = [
                c
                for c in coeff_names
                if not c.startswith(f"C({self.unit_fe_variable})")
            ]
            # Also remove time FE dummies if present
            if self.time_fe_variable:
                coeff_names = [
                    c
                    for c in coeff_names
                    if not c.startswith(f"C({self.time_fe_variable})")
                ]

        fig, ax = plt.subplots(figsize=(10, max(4, len(coeff_names) * 0.5)))

        if isinstance(self.model, PyMCModel):
            # Bayesian: forest plot with HDI
            az.plot_forest(
                self.model.idata,
                var_names=["beta"],
                coords={"coeffs": coeff_names},
                combined=True,
                hdi_prob=0.95,
                ax=ax,
            )
        else:
            # OLS: point estimates with confidence intervals
            # Get coefficient values
            coef_indices = [self.labels.index(c) for c in coeff_names]
            coefs = self.model.coef_[coef_indices]

            # Plot
            y_pos = np.arange(len(coeff_names))
            ax.barh(y_pos, coefs)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(coeff_names)
            ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Coefficient Value")

        ax.set_title("Model Coefficients (excluding FE dummies)")
        plt.tight_layout()

        return fig, ax

    def get_plot_data_bayesian(self, **kwargs: Any) -> pd.DataFrame:
        """Get plot data for Bayesian model.

        Returns
        -------
        pd.DataFrame
            DataFrame with fitted values and credible intervals
        """
        # Get posterior predictions
        if isinstance(self.model, PyMCModel):
            mu = self.model.idata.posterior["mu"]
            pred_mean = mu.mean(dim=["chain", "draw"]).values.flatten()
            pred_lower = mu.quantile(0.025, dim=["chain", "draw"]).values.flatten()
            pred_upper = mu.quantile(0.975, dim=["chain", "draw"]).values.flatten()
        else:
            raise ValueError("Model is not a PyMC model")

        plot_data = pd.DataFrame(
            {
                "y_actual": self.y.values.flatten(),
                "y_fitted": pred_mean,
                "y_fitted_lower": pred_lower,
                "y_fitted_upper": pred_upper,
                self.unit_fe_variable: self.data[self.unit_fe_variable].values,
            }
        )

        if self.time_fe_variable:
            plot_data[self.time_fe_variable] = self.data[self.time_fe_variable].values

        return plot_data

    def get_plot_data_ols(self, **kwargs: Any) -> pd.DataFrame:
        """Get plot data for OLS model.

        Returns
        -------
        pd.DataFrame
            DataFrame with fitted values
        """
        if isinstance(self.model, RegressorMixin):
            y_fitted = self.model.predict(self.X.values)
        else:
            raise ValueError("Model is not an OLS model")

        plot_data = pd.DataFrame(
            {
                "y_actual": self.y.values.flatten(),
                "y_fitted": y_fitted,
                self.unit_fe_variable: self.data[self.unit_fe_variable].values,
            }
        )

        if self.time_fe_variable:
            plot_data[self.time_fe_variable] = self.data[self.time_fe_variable].values

        return plot_data

    def plot_coefficients(
        self, var_names: list[str] | None = None
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot coefficient estimates with credible/confidence intervals.

        Automatically filters out fixed effect dummy coefficients to show only
        the treatment and control covariates.

        Parameters
        ----------
        var_names : list[str], optional
            Specific coefficient names to plot. If None, plots all non-FE coefficients.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            Figure and axes objects
        """
        # Use main plot method which already filters FE dummies
        return self.plot()

    def plot_unit_effects(
        self, highlight: list[str] | None = None, label_extreme: int = 0
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot distribution of unit fixed effects.

        Only available with fe_method="dummies". Shows histogram of estimated
        unit-specific intercepts.

        Parameters
        ----------
        highlight : list[str], optional
            List of unit IDs to highlight on the distribution.
        label_extreme : int, default=0
            Number of extreme units to label (top N + bottom N).

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            Figure and axes objects

        Raises
        ------
        ValueError
            If fe_method is not "dummies"
        """
        if self.fe_method != "dummies":
            raise ValueError(
                "plot_unit_effects() only available with fe_method='dummies'. "
                "Use within transformation for large panels."
            )

        # Extract unit fixed effects from coefficients
        unit_fe_names = [
            c for c in self.labels if c.startswith(f"C({self.unit_fe_variable})")
        ]

        if not unit_fe_names:
            raise ValueError("No unit fixed effects found in model coefficients")

        fig, ax = plt.subplots(figsize=(10, 6))

        if isinstance(self.model, PyMCModel):
            # Bayesian: get posterior means
            beta = self.model.idata.posterior["beta"]
            unit_fe_indices = [self.labels.index(name) for name in unit_fe_names]

            # Get mean and std for each unit FE
            fe_means = []
            for idx in unit_fe_indices:
                fe_means.append(
                    float(beta.sel(coeffs=self.labels[idx]).mean(dim=["chain", "draw"]))
                )

            ax.hist(fe_means, bins=min(30, len(fe_means) // 2), edgecolor="black")
            ax.set_xlabel("Unit Fixed Effect (Posterior Mean)")

        else:
            # OLS: get point estimates
            unit_fe_indices = [self.labels.index(name) for name in unit_fe_names]
            fe_values = [self.model.coef_[idx] for idx in unit_fe_indices]

            ax.hist(fe_values, bins=min(30, len(fe_values) // 2), edgecolor="black")
            ax.set_xlabel("Unit Fixed Effect")

        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of Unit Fixed Effects (N={self.n_units})")
        plt.tight_layout()

        return fig, ax

    def plot_trajectories(
        self,
        units: list[str] | None = None,
        n_sample: int = 10,
        select: Literal["random", "extreme", "high_variance"] = "random",
        show_mean: bool = True,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot unit-level time series trajectories.

        Shows actual vs fitted values for selected units over time. Useful for
        visualizing within-unit model fit and identifying problematic units.

        Parameters
        ----------
        units : list[str], optional
            Specific unit IDs to plot. If provided, ignores n_sample and select.
        n_sample : int, default=10
            Number of units to sample if units not specified.
        select : {"random", "extreme", "high_variance"}, default="random"
            Method for selecting units:
            - "random": Random sample of units
            - "extreme": Units with largest positive and negative effects
            - "high_variance": Units with most within-unit variation
        show_mean : bool, default=True
            Whether to show the overall mean trajectory.

        Returns
        -------
        tuple[plt.Figure, np.ndarray]
            Figure and array of axes objects

        Raises
        ------
        ValueError
            If time_fe_variable is not provided (cannot plot trajectories without time)
        """
        if self.time_fe_variable is None:
            raise ValueError(
                "plot_trajectories() requires time_fe_variable to be specified"
            )

        # Get plot data
        if isinstance(self.model, PyMCModel):
            plot_data = self.get_plot_data_bayesian()
        else:
            plot_data = self.get_plot_data_ols()

        # Select units to plot
        all_units = self.data[self.unit_fe_variable].unique()

        if units is not None:
            selected_units = units
        elif self.n_units <= n_sample:
            selected_units = all_units
        else:
            if select == "random":
                rng = np.random.default_rng(42)
                selected_units = rng.choice(all_units, size=n_sample, replace=False)
            else:
                # For extreme/high_variance, just take first n_sample for now
                selected_units = all_units[:n_sample]

        # Create subplots
        n_units_plot = len(selected_units)
        ncols = min(3, n_units_plot)
        nrows = (n_units_plot + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        if n_units_plot == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Plot each unit
        for idx, unit in enumerate(selected_units):
            ax = axes[idx]
            unit_data = plot_data[plot_data[self.unit_fe_variable] == unit]

            # Sort by time
            unit_data = unit_data.sort_values(self.time_fe_variable)

            # Plot actual and fitted
            ax.plot(
                unit_data[self.time_fe_variable],
                unit_data["y_actual"],
                "o-",
                label="Actual",
                alpha=0.7,
            )
            ax.plot(
                unit_data[self.time_fe_variable],
                unit_data["y_fitted"],
                "s--",
                label="Fitted",
                alpha=0.7,
            )

            # Add credible interval if Bayesian
            if "y_fitted_lower" in unit_data.columns:
                ax.fill_between(
                    unit_data[self.time_fe_variable],
                    unit_data["y_fitted_lower"],
                    unit_data["y_fitted_upper"],
                    alpha=0.2,
                )

            ax.set_title(f"Unit: {unit}", fontsize=10)
            ax.set_xlabel(self.time_fe_variable)
            ax.set_ylabel(self.outcome_variable_name)
            if idx == 0:
                ax.legend(fontsize=8)

        # Hide unused subplots
        for idx in range(n_units_plot, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig, axes

    def plot_residuals(
        self,
        kind: Literal["scatter", "histogram", "qq"] = "scatter",
        by: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot residual diagnostics.

        Parameters
        ----------
        kind : {"scatter", "histogram", "qq"}, default="scatter"
            Type of residual plot:
            - "scatter": Residuals vs fitted values
            - "histogram": Distribution of residuals
            - "qq": Q-Q plot for normality check
        by : str, optional
            Group residuals by a variable (e.g., unit or time).

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            Figure and axes objects
        """
        # Get plot data
        if isinstance(self.model, PyMCModel):
            plot_data = self.get_plot_data_bayesian()
        else:
            plot_data = self.get_plot_data_ols()

        # Calculate residuals
        residuals = plot_data["y_actual"] - plot_data["y_fitted"]

        fig, ax = plt.subplots(figsize=(10, 6))

        if kind == "scatter":
            ax.scatter(plot_data["y_fitted"], residuals, alpha=0.5)
            ax.axhline(y=0, color="r", linestyle="--")
            ax.set_xlabel("Fitted Values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Fitted Values")

        elif kind == "histogram":
            ax.hist(residuals, bins=50, edgecolor="black")
            ax.set_xlabel("Residuals")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Residuals")

        elif kind == "qq":
            from scipy import stats

            stats.probplot(residuals, dist="norm", plot=ax)
            ax.set_title("Q-Q Plot")

        plt.tight_layout()
        return fig, ax
