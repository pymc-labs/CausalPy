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
Event Study / Dynamic Difference-in-Differences
"""

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from patsy import dmatrices
from sklearn.base import RegressorMixin

from causalpy.custom_exceptions import DataException, FormulaException
from causalpy.pymc_models import PyMCModel
from causalpy.utils import round_num

from .base import BaseExperiment

LEGEND_FONT_SIZE = 12


class EventStudy(BaseExperiment):
    """A class to analyze data from Event Study / Dynamic DiD settings.

    Event studies estimate dynamic treatment effects over "event time" (time relative
    to treatment). This allows researchers to:

    1. Examine pre-treatment trends (placebo checks for parallel trends assumption)
    2. Estimate how treatment effects evolve over time after treatment
    3. Visualize the full time path of causal effects

    The model estimates:

    .. math::

        Y_{it} = \\alpha_i + \\lambda_t + \\sum_{k \\neq k_0} \\beta_k \\cdot \\mathbf{1}\\{E_{it} = k\\} + \\varepsilon_{it}

    where:
    - :math:`\\alpha_i` are unit fixed effects
    - :math:`\\lambda_t` are time fixed effects
    - :math:`E_{it} = t - G_i` is event time (time relative to treatment)
    - :math:`\\beta_k` are the dynamic treatment effects at event time k
    - :math:`k_0` is the reference (omitted) event time
    - :math:`\\mathbf{1}\\{E_{it} = k\\}` is the indicator function: equals 1 when the
      condition :math:`E_{it} = k` is true (i.e., when observation it is at event time k),
      and 0 otherwise

    **Implementation via dummy variables:** The indicator function notation is equivalent
    to creating dummy (binary) variables for each event time. Internally, this class
    creates one dummy variable for each event time k in the event window, where the dummy
    equals 1 for treated observations at that specific event time and 0 otherwise. One
    event time (the reference period, typically k=-1) is omitted to avoid perfect
    multicollinearity. The estimated regression coefficient :math:`\\beta_k` for each
    dummy variable represents the Average Treatment Effect on the Treated (ATT) at event
    time k, measured relative to the reference period.

    .. warning::

        This implementation uses a standard two-way fixed effects (TWFE) estimator,
        which requires **simultaneous treatment timing** (all treated units receive
        treatment at the same time). Staggered adoption designs, where different units
        are treated at different times, can produce biased estimates when treatment
        effects vary across cohorts. See Sun & Abraham (2021) for details.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with unit, time, outcome, and treatment time columns.
    formula : str
        A patsy-style formula specifying the model. Should include the outcome variable
        on the left-hand side and fixed effects on the right-hand side. Use ``C(column)``
        syntax for categorical fixed effects. Example: ``"y ~ C(unit) + C(time)"``.
        Event-time dummies are added automatically by the class.
    unit_col : str
        Name of the column identifying units (must match a term in the formula).
    time_col : str
        Name of the column identifying time periods (must match a term in the formula).
    treat_time_col : str
        Name of the column containing treatment time for each unit.
        Use NaN or np.inf for never-treated (control) units.
    event_window : tuple[int, int]
        Range of event times to include: (K_min, K_max).
        Default is (-5, 5).
    reference_event_time : int
        Event time to use as reference (omitted) category.
        Default is -1 (one period before treatment).
    model : PyMCModel or RegressorMixin, optional
        Model for estimation. Defaults to None.

    Example
    --------
    >>> import causalpy as cp
    >>> from causalpy.data.simulate_data import generate_event_study_data
    >>> df = generate_event_study_data(
    ...     n_units=20, n_time=20, treatment_time=10, seed=42
    ... )
    >>> result = cp.EventStudy(
    ...     df,
    ...     formula="y ~ C(unit) + C(time)",
    ...     unit_col="unit",
    ...     time_col="time",
    ...     treat_time_col="treat_time",
    ...     event_window=(-5, 5),
    ...     reference_event_time=-1,
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={
    ...             "tune": 20,
    ...             "draws": 20,
    ...             "chains": 2,
    ...             "progressbar": False,
    ...             "random_seed": 42,
    ...         }
    ...     ),
    ... )  # doctest: +SKIP
    """

    supports_ols = True
    supports_bayes = True

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        unit_col: str,
        time_col: str,
        treat_time_col: str,
        event_window: tuple[int, int] = (-5, 5),
        reference_event_time: int = -1,
        model: PyMCModel | RegressorMixin | None = None,
        **kwargs: dict,
    ) -> None:
        super().__init__(model=model)
        self.data = data.copy()
        self.expt_type = "Event Study"
        self.formula = formula
        self.unit_col = unit_col
        self.time_col = time_col
        self.treat_time_col = treat_time_col
        self.event_window = event_window
        self.reference_event_time = reference_event_time

        # Validate inputs
        self.input_validation()

        # Compute event time for each observation
        self._compute_event_time()

        # Build design matrix with FEs and event-time dummies
        self._build_design_matrix()

        # Fit model
        if isinstance(self.model, PyMCModel):
            COORDS = {
                "coeffs": self.labels,
                "obs_ind": np.arange(self.X.shape[0]),
                "treated_units": ["unit_0"],
            }
            self.model.fit(X=self.X, y=self.y, coords=COORDS)
        elif isinstance(self.model, RegressorMixin):
            if hasattr(self.model, "fit_intercept"):
                self.model.fit_intercept = False
            self.model.fit(X=self.X, y=self.y)
        else:
            raise ValueError("Model type not recognized")

        # Extract event-time coefficients
        self._extract_event_time_coefficients()

    def input_validation(self) -> None:
        """Validate input data, formula, and parameters."""
        # Check formula is provided
        if not self.formula or "~" not in self.formula:
            raise FormulaException(
                "Formula must be provided in the form 'outcome ~ predictors'"
            )

        # Check required columns exist for event time computation
        required_cols = [self.unit_col, self.time_col, self.treat_time_col]
        for col in required_cols:
            if col not in self.data.columns:
                raise DataException(f"Required column '{col}' not found in data")

        # Check event window is valid
        if self.event_window[0] >= self.event_window[1]:
            raise DataException(
                f"event_window[0] ({self.event_window[0]}) must be less than "
                f"event_window[1] ({self.event_window[1]})"
            )

        # Check reference event time is in window
        if not (
            self.event_window[0] <= self.reference_event_time <= self.event_window[1]
        ):
            raise DataException(
                f"reference_event_time ({self.reference_event_time}) must be within "
                f"event_window {self.event_window}"
            )

        # Check for duplicate unit-time observations
        duplicates = self.data.duplicated(subset=[self.unit_col, self.time_col])
        if duplicates.any():
            raise DataException(
                "Data contains duplicate unit-time observations. "
                "Each unit should have at most one observation per time period."
            )

        # Check that all treated units have the same treatment time
        # (staggered adoption is not currently supported)
        # Filter out control units (marked with NaN or np.inf)
        is_control = self.data[self.treat_time_col].apply(
            lambda x: pd.isna(x) or np.isinf(x)
        )
        treated_times = self.data.loc[~is_control, self.treat_time_col].unique()
        if len(treated_times) > 1:
            raise DataException(
                f"All treated units must have the same treatment time. "
                f"Found {len(treated_times)} different treatment times: "
                f"{sorted(treated_times)}. "
                f"Staggered adoption designs are not currently supported."
            )

    def _compute_event_time(self) -> None:
        """Compute event time (time relative to treatment) for each observation."""
        self.data["_event_time"] = np.nan

        # For treated units, compute event time
        # Treated units are those with finite, non-NaN treatment times
        is_control = self.data[self.treat_time_col].apply(
            lambda x: pd.isna(x) or np.isinf(x)
        )
        treated_mask = ~is_control
        self.data.loc[treated_mask, "_event_time"] = (
            self.data.loc[treated_mask, self.time_col]
            - self.data.loc[treated_mask, self.treat_time_col]
        )

        # Mark observations in the event window
        self.data["_in_event_window"] = (
            self.data["_event_time"] >= self.event_window[0]
        ) & (self.data["_event_time"] <= self.event_window[1])

    def _build_design_matrix(self) -> None:
        """Build design matrix using patsy formula plus event-time dummies."""
        # Parse formula with patsy to get y and X (including FEs and covariates)
        y, X = dmatrices(self.formula, self.data)
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info

        # Extract outcome variable name from formula
        self.outcome_variable_name = y.design_info.column_names[0]

        # Convert patsy output to DataFrames for manipulation
        X_df = pd.DataFrame(
            X, columns=X.design_info.column_names, index=self.data.index
        )

        # Build event-time dummies (excluding reference event time)
        event_times = list(range(self.event_window[0], self.event_window[1] + 1))
        event_times_non_ref = [k for k in event_times if k != self.reference_event_time]
        self.event_times_non_ref = event_times_non_ref

        event_time_dummies = pd.DataFrame(index=self.data.index)
        for k in event_times_non_ref:
            col_name = f"event_time_{k}"
            # 1 if treated and at event time k, 0 otherwise
            event_time_dummies[col_name] = (
                (self.data["_event_time"] == k) & self.data["_in_event_window"]
            ).astype(float)

        # Combine patsy design matrix with event-time dummies
        X_df = pd.concat([X_df, event_time_dummies], axis=1)

        self.labels = list(X_df.columns)
        self.event_time_labels = [f"event_time_{k}" for k in event_times_non_ref]

        # Convert to xarray
        self.X = xr.DataArray(
            X_df.values,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": np.arange(X_df.shape[0]),
                "coeffs": self.labels,
            },
        )

        y_values = np.asarray(y).reshape(-1, 1)
        self.y = xr.DataArray(
            y_values,
            dims=["obs_ind", "treated_units"],
            coords={
                "obs_ind": np.arange(len(y_values)),
                "treated_units": ["unit_0"],
            },
        )

    def _extract_event_time_coefficients(self) -> None:
        """Extract event-time coefficients from fitted model."""
        self.event_time_coeffs = {}

        if isinstance(self.model, PyMCModel):
            assert self.model.idata is not None
            beta = self.model.idata.posterior["beta"]
            coeffs_coord = beta.coords["coeffs"].values

            for k in self.event_times_non_ref:
                label = f"event_time_{k}"
                if label in coeffs_coord:
                    idx = list(coeffs_coord).index(label)
                    self.event_time_coeffs[k] = beta.isel(coeffs=idx)

            # Add reference event time as zero
            self.event_time_coeffs[self.reference_event_time] = xr.DataArray(0.0)

        elif isinstance(self.model, RegressorMixin):
            coeffs = self.model.get_coeffs()
            coeff_dict = dict(zip(self.labels, coeffs, strict=False))

            for k in self.event_times_non_ref:
                label = f"event_time_{k}"
                if label in coeff_dict:
                    self.event_time_coeffs[k] = coeff_dict[label]

            # Add reference event time as zero
            self.event_time_coeffs[self.reference_event_time] = 0.0

    def summary(self, round_to: int | None = 2, hdi_prob: float = 0.94) -> None:
        """Print summary of event-time coefficients.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals for rounding. Defaults to 2.
        hdi_prob : float, optional
            Probability mass for the highest density interval. Defaults to 0.94.
        """
        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print(f"Event window: {self.event_window}")
        print(f"Reference event time: {self.reference_event_time}")
        print("\nEvent-time coefficients (beta_k):")
        print("-" * 60)

        # Sort by event time
        sorted_times = sorted(self.event_time_coeffs.keys())

        # Compute HDI bounds labels
        hdi_lower_pct = (1 - hdi_prob) / 2 * 100
        hdi_upper_pct = (1 - (1 - hdi_prob) / 2) * 100

        if isinstance(self.model, PyMCModel):
            print(
                f"{'Event Time':>12} {'Mean':>10} {'SD':>10} "
                f"{f'HDI {hdi_lower_pct:.0f}%':>10} {f'HDI {hdi_upper_pct:.0f}%':>10}"
            )
            print("-" * 60)
            for k in sorted_times:
                coeff = self.event_time_coeffs[k]
                if k == self.reference_event_time:
                    print(f"{k:>12} {'0 (ref)':>10} {'-':>10} {'-':>10} {'-':>10}")
                else:
                    mean_val = float(coeff.mean())
                    std_val = float(coeff.std())
                    hdi = az.hdi(coeff.values.flatten(), hdi_prob=hdi_prob)
                    print(
                        f"{k:>12} "
                        f"{round_num(mean_val, round_to):>10} "
                        f"{round_num(std_val, round_to):>10} "
                        f"{round_num(hdi[0], round_to):>10} "
                        f"{round_num(hdi[1], round_to):>10}"
                    )
        else:
            print(f"{'Event Time':>12} {'Coefficient':>15}")
            print("-" * 60)
            for k in sorted_times:
                coeff = self.event_time_coeffs[k]
                if k == self.reference_event_time:
                    print(f"{k:>12} {'0 (ref)':>15}")
                else:
                    print(f"{k:>12} {round_num(coeff, round_to):>15}")

        print("-" * 60)
        self.print_coefficients(round_to)

    def get_event_time_summary(
        self, round_to: int | None = 2, hdi_prob: float = 0.94
    ) -> pd.DataFrame:
        """Get event-time coefficients as a DataFrame.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals for rounding. Defaults to 2.
        hdi_prob : float, optional
            Probability mass for the highest density interval. Defaults to 0.94.

        Returns
        -------
        pd.DataFrame
            DataFrame with event-time coefficient statistics.
        """
        sorted_times = sorted(self.event_time_coeffs.keys())
        rows = []

        # Compute HDI bounds labels
        hdi_lower_pct = (1 - hdi_prob) / 2 * 100
        hdi_upper_pct = (1 - (1 - hdi_prob) / 2) * 100
        hdi_lower_col = f"hdi_{hdi_lower_pct:.0f}%"
        hdi_upper_col = f"hdi_{hdi_upper_pct:.0f}%"

        for k in sorted_times:
            coeff = self.event_time_coeffs[k]
            if k == self.reference_event_time:
                row = {
                    "event_time": k,
                    "mean": 0.0,
                    "std": 0.0,
                    hdi_lower_col: 0.0,
                    hdi_upper_col: 0.0,
                    "is_reference": True,
                }
            elif isinstance(self.model, PyMCModel):
                hdi = az.hdi(coeff.values.flatten(), hdi_prob=hdi_prob)
                mean_val = float(coeff.mean())
                std_val = float(coeff.std())
                hdi_lower_val = float(hdi[0])
                hdi_upper_val = float(hdi[1])
                row = {
                    "event_time": k,
                    "mean": np.round(mean_val, round_to)
                    if round_to is not None
                    else mean_val,
                    "std": np.round(std_val, round_to)
                    if round_to is not None
                    else std_val,
                    hdi_lower_col: np.round(hdi_lower_val, round_to)
                    if round_to is not None
                    else hdi_lower_val,
                    hdi_upper_col: np.round(hdi_upper_val, round_to)
                    if round_to is not None
                    else hdi_upper_val,
                    "is_reference": False,
                }
            else:
                mean_val = float(coeff)
                row = {
                    "event_time": k,
                    "mean": np.round(mean_val, round_to)
                    if round_to is not None
                    else mean_val,
                    "std": np.nan,
                    hdi_lower_col: np.nan,
                    hdi_upper_col: np.nan,
                    "is_reference": False,
                }
            rows.append(row)

        return pd.DataFrame(rows)

    def _bayesian_plot(
        self,
        round_to: int | None = 2,
        figsize: tuple[float, float] = (10, 6),
        hdi_prob: float = 0.94,
        **kwargs: dict,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot event-study coefficients with credible intervals (Bayesian).

        Parameters
        ----------
        round_to : int, optional
            Number of decimals for rounding. Defaults to 2.
        figsize : tuple[float, float], optional
            Figure size in inches (width, height). Defaults to (10, 6).
        hdi_prob : float, optional
            Probability mass for the highest density interval. Defaults to 0.94.
        **kwargs : dict
            Additional keyword arguments (currently unused).

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            The matplotlib Figure and Axes objects.
        """
        fig, ax = plt.subplots(figsize=figsize)

        sorted_times = sorted(self.event_time_coeffs.keys())
        means_list: list[float] = []
        lower_list: list[float] = []
        upper_list: list[float] = []

        for k in sorted_times:
            coeff = self.event_time_coeffs[k]
            if k == self.reference_event_time:
                means_list.append(0.0)
                lower_list.append(0.0)
                upper_list.append(0.0)
            else:
                hdi = az.hdi(coeff.values.flatten(), hdi_prob=hdi_prob)
                means_list.append(float(coeff.mean()))
                lower_list.append(float(hdi[0]))
                upper_list.append(float(hdi[1]))

        means = np.array(means_list)
        lower = np.array(lower_list)
        upper = np.array(upper_list)

        # Plot coefficients with error bars
        hdi_pct = int(hdi_prob * 100)
        ax.errorbar(
            sorted_times,
            means,
            yerr=[means - lower, upper - means],
            fmt="o",
            capsize=4,
            capthick=2,
            markersize=8,
            color="C0",
            label=f"Event-time coefficient ({hdi_pct}% HDI)",
        )

        # Add horizontal line at zero
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        # Add vertical line at k=0 (treatment time)
        ax.axvline(
            x=0,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Treatment (k=0)",
        )

        # Shade pre-treatment period
        ax.axvspan(
            self.event_window[0] - 0.5,
            -0.5,
            alpha=0.1,
            color="blue",
            label="Pre-treatment",
        )

        # Mark reference period
        ax.scatter(
            [self.reference_event_time],
            [0],
            marker="s",
            s=100,
            color="orange",
            zorder=5,
            label=f"Reference (k={self.reference_event_time})",
        )

        ax.set_xlabel("Event Time (k)", fontsize=12)
        ax.set_ylabel(r"$\beta_k$ (Treatment Effect)", fontsize=12)
        ax.set_title("Event Study: Dynamic Treatment Effects", fontsize=14)
        ax.set_xticks(sorted_times)
        ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
        ax.grid(True, alpha=0.3)

        return fig, ax

    def _ols_plot(
        self,
        round_to: int | None = 2,
        figsize: tuple[float, float] = (10, 6),
        **kwargs: dict,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot event-study coefficients (OLS).

        Parameters
        ----------
        round_to : int, optional
            Number of decimals for rounding. Defaults to 2.
        figsize : tuple[float, float], optional
            Figure size in inches (width, height). Defaults to (10, 6).
        **kwargs : dict
            Additional keyword arguments (currently unused).

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            The matplotlib Figure and Axes objects.
        """
        fig, ax = plt.subplots(figsize=figsize)

        sorted_times = sorted(self.event_time_coeffs.keys())
        coeffs = []

        for k in sorted_times:
            if k == self.reference_event_time:
                coeffs.append(0.0)
            else:
                coeffs.append(float(self.event_time_coeffs[k]))

        # Plot coefficients
        ax.plot(
            sorted_times,
            coeffs,
            "o-",
            markersize=8,
            color="C0",
            label="Event-time coefficient",
        )

        # Add horizontal line at zero
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        # Add vertical line at k=0 (treatment time)
        ax.axvline(
            x=0,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Treatment (k=0)",
        )

        # Shade pre-treatment period
        ax.axvspan(
            self.event_window[0] - 0.5,
            -0.5,
            alpha=0.1,
            color="blue",
            label="Pre-treatment",
        )

        # Mark reference period
        ax.scatter(
            [self.reference_event_time],
            [0],
            marker="s",
            s=100,
            color="orange",
            zorder=5,
            label=f"Reference (k={self.reference_event_time})",
        )

        ax.set_xlabel("Event Time (k)", fontsize=12)
        ax.set_ylabel(r"$\beta_k$ (Treatment Effect)", fontsize=12)
        ax.set_title("Event Study: Dynamic Treatment Effects", fontsize=14)
        ax.set_xticks(sorted_times)
        ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
        ax.grid(True, alpha=0.3)

        return fig, ax

    def get_plot_data_bayesian(
        self, hdi_prob: float = 0.94, **kwargs: dict
    ) -> pd.DataFrame:
        """Get plot data for Bayesian model."""
        return self.get_event_time_summary(hdi_prob=hdi_prob)

    def get_plot_data_ols(self, **kwargs: dict) -> pd.DataFrame:
        """Get plot data for OLS model."""
        return self.get_event_time_summary(hdi_prob=0.94)
