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
"""Panel Regression with Fixed Effects."""

from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import dmatrices
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    facet_wrap,
    geom_col,
    geom_errorbarh,
    geom_histogram,
    geom_line,
    geom_point,
    geom_ribbon,
    geom_vline,
    ggplot,
    guides,
    labs,
    theme,
)
from scipy import stats
from sklearn.base import RegressorMixin

from causalpy.constants import HDI_PROB
from causalpy.custom_exceptions import DataException
from causalpy.experiments.model_adapter import build_coords
from causalpy.pymc_models import PyMCModel
from causalpy.reporting import EffectSummary
from causalpy.utils import round_num

from .base import BaseExperiment


class PanelRegression(BaseExperiment):
    """Panel regression with fixed effects estimation.

    Enables panel-aware visualization and diagnostics, with support for both
    unpooled dummy-variable and demeaned (de-meaned) fixed effects.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas dataframe with panel data. Each row is an observation for a
        unit at a time period.
    formula : str
        A statistical model formula using patsy syntax. For the unpooled
        dummy-variable fixed-effects approach, include ``C(unit_var)`` (and
        optionally ``C(time_var)``) in the formula. For the demeaned
        transformation, do NOT include those ``C(...)`` terms; fixed effects
        are removed by transformation before fitting.
    unit_fe_variable : str
        Column name for the unit identifier (e.g., "state", "id", "country").
    time_fe_variable : str, optional
        Column name for the time identifier (e.g., "year", "wave", "period").
        If provided, time fixed effects will be included. Default is None.
    fe_method : {"dummies", "demeaned"}, default="dummies"
        Method for handling fixed effects:

        - "dummies": Use unpooled dummy-variable fixed effects
          (``C(unit)``/``C(time)`` in formula). Gets individual unit effect
          estimates but creates N-1 dummy columns. Best for small N.
        - "demeaned": Use demeaned (de-meaned) transformation. Scales to large N
          but doesn't directly estimate individual unit effects.
    model : PyMCModel or RegressorMixin, optional
        A PyMC (Bayesian) or sklearn (OLS) model. If None, a model must be provided.
    **kwargs
        Additional keyword arguments forwarded to :class:`BaseExperiment`.

    Attributes
    ----------
    n_units : int
        Number of unique units in the panel.
    n_periods : int or None
        Number of unique time periods (None if time_fe_variable not provided).
    fe_method : str
        The fixed effects method used ("dummies" or "demeaned").
    _group_means : dict
        Stored group means for recovering unit effects (demeaned method only).

    Notes
    -----
    The demeaned transformation (de-meaning by group) removes time-invariant
    confounders but also drops time-invariant covariates from the model. For
    the ``"dummies"`` approach (unpooled FE), individual unit effects can be
    extracted from the coefficients. For the demeaned approach, unit effects
    can be recovered post-hoc using the stored group means (``_group_means``),
    which are always computed from the original (pre-demeaning) data.

    This class does not yet implement hierarchical/partial-pooling fixed
    effects. Those semantics are intentionally kept out of scope here so
    ``fe_method="dummies"`` remains an accurate label for the current
    unpooled estimator.

    Two-way fixed effects (unit + time) control for both unit-specific and
    time-specific unobserved heterogeneity. This is the standard approach in
    difference-in-differences estimation.

    **Balanced vs unbalanced panels**: A panel is *balanced* when every unit
    is observed in every time period; otherwise it is *unbalanced* (e.g. unit
    entry/exit, missing waves). When both unit and time fixed effects are
    requested with ``fe_method="demeaned"``, the sequential demeaning
    (first by unit, then by time) is algebraically equivalent to the standard
    two-way demeaned transformation only for balanced panels. For unbalanced
    panels, iterative alternating demeaning would be needed for exact
    convergence; the single-pass approximation used here may introduce small
    biases. Unbalanced panels are common in practice (e.g. firm or worker
    panels with attrition); for heavily unbalanced data, consider checking
    sensitivity or using dedicated FE packages that implement iterative
    two-way demeaning (e.g. reghdfe, pyfixest).

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
    ...             "treatment": int(t >= 10 and u in units[:5]),
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

    Large panel with demeaned transformation:

    >>> # Create larger panel: 1000 units, 10 time periods
    >>> np.random.seed(42)
    >>> units = [f"unit_{i}" for i in range(1000)]
    >>> periods = range(10)
    >>> data = pd.DataFrame(
    ...     [
    ...         {
    ...             "unit": u,
    ...             "time": t,
    ...             "treatment": int(t >= 5),
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
    ...     fe_method="demeaned",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={"random_seed": 42, "progressbar": False}
    ...     ),
    ... )
    """

    supports_ols = True
    supports_bayes = True
    _deprecated_design_aliases = {"X": ("design", "X"), "y": ("design", "y")}

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        unit_fe_variable: str,
        time_fe_variable: str | None = None,
        fe_method: Literal["dummies", "demeaned"] = "dummies",
        model: PyMCModel | RegressorMixin | None = None,
        **kwargs: dict,
    ) -> None:
        super().__init__(model=model)

        # Rename the index to "obs_ind" (on original, before copying)
        data.index.name = "obs_ind"
        self.data = data
        self.expt_type = "Panel Regression"
        self.formula = formula
        self.unit_fe_variable = unit_fe_variable
        self.time_fe_variable = time_fe_variable
        self.fe_method = fe_method

        # Store a copy of original data for recovering group means in demeaned
        # transformation.  Other experiment classes don't need this because
        # they don't demean the data before fitting.
        self._original_data = data.copy()

        # Initialize storage for group means (used in demeaned transformation)
        self._group_means: dict[str, pd.DataFrame] = {}

        # Pipeline (matches pattern of other experiment classes)
        self.input_validation()

        # Store panel dimensions (after validation confirms columns exist)
        self.n_units = data[unit_fe_variable].nunique()
        self.n_periods = data[time_fe_variable].nunique() if time_fe_variable else None
        self._build_design_matrices()
        self._prepare_data()
        self.algorithm()

    def input_validation(self) -> None:
        """Validate input parameters."""
        if self.unit_fe_variable not in self.data.columns:
            raise DataException(
                f"unit_fe_variable '{self.unit_fe_variable}' not found in data columns"
            )

        if self.time_fe_variable and self.time_fe_variable not in self.data.columns:
            raise DataException(
                f"time_fe_variable '{self.time_fe_variable}' not found in data columns"
            )

        if self.fe_method not in ["dummies", "demeaned"]:
            raise ValueError(
                "fe_method must be 'dummies' (unpooled fixed effects) or 'demeaned'"
            )

        # Check if formula includes C(unit_var) or C(time_var) when using demeaned method
        if (
            self.fe_method == "demeaned"
            and f"C({self.unit_fe_variable})" in self.formula
        ):
            raise ValueError(
                f"When using fe_method='demeaned', do not include C({self.unit_fe_variable}) "
                "in the formula. The demeaned transformation handles unit fixed effects automatically."
            )

        if (
            self.fe_method == "demeaned"
            and self.time_fe_variable
            and f"C({self.time_fe_variable})" in self.formula
        ):
            raise ValueError(
                f"When using fe_method='demeaned', do not include C({self.time_fe_variable}) "
                "in the formula. The demeaned transformation handles time fixed effects automatically."
            )

    def _build_design_matrices(self) -> None:
        """Build design matrices from formula and data using patsy.

        For ``fe_method="demeaned"`` this first applies the demeaned
        transformation (de-meaning by unit, and optionally by time) before
        constructing the patsy design matrices.
        """
        data = self._original_data.copy()

        # Apply demeaned transformation if requested
        if self.fe_method == "demeaned":
            data = self._demean_transform(data, self.unit_fe_variable)
            if self.time_fe_variable:
                # TODO: Use iterative alternating demeaning for unbalanced panels
                # (single-pass is exact only for balanced; see docstring Notes).
                data = self._demean_transform(data, self.time_fe_variable)

        y, X = dmatrices(self.formula, data)
        self.outcome_variable_name = y.design_info.column_names[0]
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self._y_raw, self._X_raw = np.asarray(y), np.asarray(X)

    def _prepare_data(self) -> None:
        """Bundle design matrices into an ``xr.Dataset``."""
        n = self._X_raw.shape[0]
        self.design = self._build_design_dataset(
            self._X_raw,
            self._y_raw,
            obs_ind=np.arange(n),
            coeffs=self.labels,
        )
        del self._X_raw, self._y_raw

    def algorithm(self) -> None:
        """Run the experiment algorithm: fit the model."""
        X = self.design["X"]
        y = self.design["y"]

        self._model_backend.fit(
            X=X,
            y=y,
            coords=build_coords(self.labels, X.shape[0]),
        )

    def _demean_transform(self, data: pd.DataFrame, group_var: str) -> pd.DataFrame:
        """Apply demeaned transformation (demean by group).

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

        Notes
        -----
        When two-way fixed effects are requested (both unit and time), the
        sequential single-pass demeaning (first by unit, then by time) is
        algebraically equivalent to the standard two-way demeaned transformation

        .. math:: \\tilde{y}_{it} = y_{it} - \\bar{y}_{i\\cdot}
                   - \\bar{y}_{\\cdot t} + \\bar{y}_{\\cdot\\cdot}

        **only for balanced panels** (every unit observed in every period).
        For unbalanced panels the single pass is an approximation; iterative
        alternating demeaning would be needed for exact convergence.

        Group means stored in ``_group_means`` are always computed from the
        **original** (pre-demeaning) data so that unit and time effects can
        be recovered post-hoc without confusion.
        """
        data = data.copy()

        # Identify numeric and boolean columns to demean (exclude group variables).
        # Boolean columns (e.g. treatment indicators) must be included; pandas
        # select_dtypes(include=[np.number]) excludes bool.
        numeric_cols = data.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        group_vars_to_exclude = [self.unit_fe_variable]
        if self.time_fe_variable:
            group_vars_to_exclude.append(self.time_fe_variable)

        numeric_cols = [c for c in numeric_cols if c not in group_vars_to_exclude]

        # Cast boolean columns to float so that demeaning produces correct
        # numeric results (bool - float would otherwise raise or produce
        # unexpected dtypes).
        for col in numeric_cols:
            if data[col].dtype == "bool":
                data[col] = data[col].astype(float)

        # Store group means from the ORIGINAL data (before any demeaning in
        # prior calls) so that fixed effects can be recovered post-hoc.
        if group_var not in self._group_means:
            self._group_means[group_var] = self._original_data.groupby(group_var)[
                numeric_cols
            ].mean()

        # Demean each numeric column
        for col in numeric_cols:
            group_mean = data.groupby(group_var)[col].transform("mean")
            data[col] = data[col] - group_mean

        return data

    def _get_non_fe_labels(self) -> list[str]:
        """Return coefficient labels with FE dummy names filtered out.

        For ``fe_method="dummies"`` this removes all ``C(unit_fe_variable)``
        and ``C(time_fe_variable)`` labels.  For ``fe_method="demeaned"`` it
        returns all labels unchanged (there are no dummy columns).
        """
        coeff_labels = self.labels.copy()
        if self.fe_method == "dummies":
            coeff_labels = [
                c
                for c in coeff_labels
                if not c.startswith(f"C({self.unit_fe_variable})")
            ]
            if self.time_fe_variable:
                coeff_labels = [
                    c
                    for c in coeff_labels
                    if not c.startswith(f"C({self.time_fe_variable})")
                ]
        return coeff_labels

    def summary(self, round_to: int | None = None) -> None:
        """Print a summary of the panel regression results.

        Parameters
        ----------
        round_to : int, optional
            Number of significant figures to round to. Defaults to None,
            in which case 2 significant figures are used.
        """
        print(f"\n{self.expt_type}")
        print("=" * 60)
        print(f"Units: {self.n_units} ({self.unit_fe_variable})")
        if self.n_periods:
            print(f"Periods: {self.n_periods} ({self.time_fe_variable})")
        print(f"FE method: {self.fe_method}")
        print(f"Observations: {self.design['X'].shape[0]}")
        print("=" * 60)

        coeff_labels = self._get_non_fe_labels()

        if self.fe_method == "dummies" and len(coeff_labels) < len(self.labels):
            n_hidden = len(self.labels) - len(coeff_labels)
            print(
                f"\nNote: {n_hidden} fixed effect coefficients not shown "
                "(use print_coefficients() to see all)"
            )

        print("\nModel Coefficients:")
        if self._model_backend.is_bayesian:
            # PyMC print_coefficients uses coordinate-based lookup so a
            # filtered label list works correctly.
            self.model.print_coefficients(coeff_labels, round_to)
        else:
            # For OLS models the base print_coefficients uses positional zip
            # which would pair filtered labels with the wrong coefficient
            # values.  We do our own index-based lookup instead.
            coefs = self.model.get_coeffs()
            max_label_length = max(len(name) for name in coeff_labels)
            rd = round_to if round_to is not None else 2
            print("Model coefficients:")
            for name in coeff_labels:
                idx = self.labels.index(name)
                formatted_name = f"{name:<{max_label_length}}"
                formatted_val = f"{round_num(coefs[idx], rd):>10}"
                print(f"  {formatted_name}\t{formatted_val}")

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
        """Generate a decision-ready summary of causal effects.

        .. note::
            ``effect_summary()`` is not yet implemented for
            ``PanelRegression``.  Panel fixed-effects models estimate
            regression coefficients rather than time-varying causal impacts,
            so the standard ITS/SC-style effect summary does not directly
            apply.  Use :meth:`summary` for coefficient-level inference.

        Parameters
        ----------
        window : str, tuple, or slice, default "post"
            Time window for analysis (placeholder; not consumed).
        direction : {"increase", "decrease", "two-sided"}, default "increase"
            Direction for tail probability calculation.
        alpha : float, default 0.05
            Significance level for HDI/CI intervals.
        cumulative : bool, default True
            Whether to include cumulative effect statistics.
        relative : bool, default True
            Whether to include relative effect statistics.
        min_effect : float, optional
            Region of Practical Equivalence (ROPE) threshold.
        treated_unit : str, optional
            Treated unit selector for multi-unit experiments.
        period : {"intervention", "post", "comparison"}, optional
            Period selector for three-period designs.
        prefix : str, default "Post-period"
            Prefix for prose generation.
        **kwargs
            Reserved for forward-compatibility.

        Raises
        ------
        NotImplementedError
            Always raised; this method is a placeholder for future work.
        """
        raise NotImplementedError(
            "effect_summary() is not yet implemented for PanelRegression. "
            "Panel fixed-effects models estimate regression coefficients rather "
            "than time-varying causal impacts. Use summary() for coefficient-level "
            "inference."
        )

    def plot(
        self,
        *,
        hdi_prob: float = HDI_PROB,
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the panel regression coefficients.

        Bayesian models render a forest plot with HDI intervals; OLS models
        render a bar plot of point estimates. To plot only a subset of
        coefficients (or to customise the figure size), call
        :meth:`plot_coefficients` directly.

        Parameters
        ----------
        hdi_prob : float
            Probability mass of the highest density interval drawn around
            each posterior coefficient via :func:`arviz.plot_forest`. Must
            be in ``(0, 1]``. Ignored for OLS models. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
        show : bool
            Whether to automatically display the plot. Defaults to ``True``.
        legend_kwargs : dict, optional
            Keyword arguments to adjust legend placement and styling.
            Supported keys: ``loc``, ``bbox_to_anchor``, ``fontsize``,
            ``frameon``, ``title`` (``bbox_transform`` is accepted alongside
            ``bbox_to_anchor``). The existing legend is modified **in
            place** so that custom handles are preserved.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure that was created.
        ax : matplotlib.axes.Axes
            The axes object containing the coefficient plot.
        """
        return self._render_plot(
            show=show,
            legend_kwargs=legend_kwargs,
            hdi_prob=hdi_prob,
        )

    def _bayesian_plot(
        self, hdi_prob: float = HDI_PROB, **kwargs: Any
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create coefficient plot for Bayesian model.

        Parameters
        ----------
        hdi_prob : float, optional
            Probability mass of the highest density interval drawn around each
            posterior coefficient via :func:`arviz.plot_forest`. Must be in
            ``(0, 1]``. Defaults to :data:`~causalpy.constants.HDI_PROB`
            (currently 0.94).

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            Figure and axes objects
        """
        return self._plot_coefficients_internal(hdi_prob=hdi_prob)

    def _ols_plot(self, **kwargs: Any) -> tuple[plt.Figure, plt.Axes]:
        """Create coefficient plot for OLS model.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            Figure and axes objects
        """
        return self._plot_coefficients_internal()

    def _plot_coefficients_internal(
        self, var_names: list[str] | None = None, hdi_prob: float = HDI_PROB
    ) -> tuple[plt.Figure, plt.Axes]:
        """Internal method to create coefficient plot.

        Parameters
        ----------
        var_names : list[str], optional
            Specific coefficient names to plot.  If ``None``, plots all
            non-FE coefficients (as determined by ``_get_non_fe_labels``).
        hdi_prob : float
            Probability mass for the HDI interval when plotting Bayesian
            coefficients. Must be in (0, 1). Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
        """
        if not 0 < hdi_prob < 1:
            raise ValueError("hdi_prob must be between 0 and 1")

        coeff_names = var_names if var_names is not None else self._get_non_fe_labels()
        fig_height = max(4, len(coeff_names) * 0.5)
        figsize = (10, fig_height)

        if self._model_backend.is_bayesian:
            beta = self.model.idata.posterior["beta"].sel(coeffs=coeff_names)  # type: ignore[union-attr]
            if "treated_units" in beta.dims:
                beta = beta.squeeze("treated_units", drop=True)
            hdi = az.hdi(beta, hdi_prob=hdi_prob)
            means = beta.mean(dim=["chain", "draw"])
            tidy = pd.DataFrame(
                {
                    "coeffs": coeff_names,
                    "mean": means.values,
                    "lower": hdi["beta"].sel(hdi="lower").values,
                    "higher": hdi["beta"].sel(hdi="higher").values,
                }
            )
            tidy["coeffs"] = pd.Categorical(
                tidy["coeffs"], categories=coeff_names, ordered=True
            )
            title = f"Model Coefficients with {hdi_prob:.0%} HDI"
            p = (
                ggplot(tidy, aes(x="mean", y="coeffs"))
                + geom_errorbarh(aes(xmin="lower", xmax="higher"), height=0.2, size=0.6)
                + geom_point(size=2)
                + geom_vline(xintercept=0, color="black", linetype="dashed", alpha=0.8)
                + labs(title="", x="Coefficient Value", y="")
                + theme(figure_size=figsize)
            )
        else:
            coef_indices = [self.labels.index(c) for c in coeff_names]
            coefs = self.model.get_coeffs()[coef_indices]
            tidy = pd.DataFrame({"coeffs": coeff_names, "coef": coefs})
            tidy["coeffs"] = pd.Categorical(
                tidy["coeffs"], categories=coeff_names, ordered=True
            )
            title = "Model Coefficients"
            p = (
                ggplot(tidy, aes(x="coef", y="coeffs"))
                + geom_col(fill="#1f77b4")
                + geom_vline(xintercept=0, color="black", linetype="dashed", alpha=0.8)
                + coord_flip()
                + labs(title="", x="Coefficient Value", y="")
                + theme(figure_size=figsize)
            )

        fig = p.draw()
        axes = [a for a in fig.axes if a.get_subplotspec() is not None]
        ax = axes[0]
        ax.set_title(title)
        return fig, ax

    def get_plot_data_bayesian(self, **kwargs: Any) -> pd.DataFrame:
        """Get plot data for Bayesian model.

        Parameters
        ----------
        **kwargs
            Reserved for forward-compatibility; not consumed by this
            implementation.

        Returns
        -------
        pd.DataFrame
            DataFrame with fitted values and credible intervals
        """
        # Get posterior predictions
        if self._model_backend.is_bayesian:
            mu = self.model.idata.posterior["mu"]  # type: ignore[union-attr]
            pred_mean = mu.mean(dim=["chain", "draw"]).values.flatten()
            pred_lower = mu.quantile(0.025, dim=["chain", "draw"]).values.flatten()
            pred_upper = mu.quantile(0.975, dim=["chain", "draw"]).values.flatten()
        else:
            raise ValueError("Model is not a PyMC model")

        plot_data = pd.DataFrame(
            {
                "y_actual": self.design["y"].values.flatten(),
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

        Parameters
        ----------
        **kwargs
            Reserved for forward-compatibility; not consumed by this
            implementation.

        Returns
        -------
        pd.DataFrame
            DataFrame with fitted values
        """
        if self._model_backend.is_ols:
            y_fitted = np.squeeze(self.model.predict(self.design["X"]))
        else:
            raise ValueError("Model is not an OLS model")

        plot_data = pd.DataFrame(
            {
                "y_actual": self.design["y"].values.flatten(),
                "y_fitted": y_fitted,
                self.unit_fe_variable: self.data[self.unit_fe_variable].values,
            }
        )

        if self.time_fe_variable:
            plot_data[self.time_fe_variable] = self.data[self.time_fe_variable].values

        return plot_data

    def plot_coefficients(
        self, var_names: list[str] | None = None, hdi_prob: float = HDI_PROB
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot coefficient estimates with credible/confidence intervals.

        Automatically filters out fixed effect dummy coefficients to show only
        the treatment and control covariates.

        Parameters
        ----------
        var_names : list[str], optional
            Specific coefficient names to plot.  Names must match the patsy
            design-matrix labels (e.g. ``"treatment"``, ``"x1"``).
            If ``None``, plots all non-FE coefficients.
        hdi_prob : float
            Probability mass for the HDI interval when plotting Bayesian
            coefficients. Must be in (0, 1). Ignored for OLS models.
            Defaults to :data:`~causalpy.constants.HDI_PROB` (currently 0.94).

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            Figure and axes objects
        """
        return self._plot_coefficients_internal(var_names=var_names, hdi_prob=hdi_prob)

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
                "Use demeaned transformation for large panels."
            )

        # Extract unit fixed effects from coefficients
        unit_fe_names = [
            c for c in self.labels if c.startswith(f"C({self.unit_fe_variable})")
        ]

        if not unit_fe_names:
            raise ValueError("No unit fixed effects found in model coefficients")

        if self._model_backend.is_bayesian:
            beta = self.model.idata.posterior["beta"]  # type: ignore[union-attr]
            unit_fe_indices = [self.labels.index(name) for name in unit_fe_names]
            fe_means = [
                beta.sel(coeffs=self.labels[idx])
                .mean(dim=["chain", "draw"])
                .squeeze("treated_units", drop=True)
                .item()
                for idx in unit_fe_indices
            ]
            x_label = "Unit Fixed Effect (Posterior Mean)"
            values = fe_means
        else:
            unit_fe_indices = [self.labels.index(name) for name in unit_fe_names]
            coefs = self.model.get_coeffs()
            values = [coefs[idx] for idx in unit_fe_indices]
            x_label = "Unit Fixed Effect"

        n_bins = min(30, max(1, len(values) // 2))
        tidy = pd.DataFrame({"value": values})
        title = f"Distribution of Unit Fixed Effects (N={self.n_units})"
        p = (
            ggplot(tidy, aes(x="value"))
            + geom_histogram(bins=n_bins, fill="#1f77b4", color="black", alpha=0.7)
            + labs(x=x_label, y="Count", title="")
            + theme(figure_size=(10, 6))
        )
        fig = p.draw()
        axes = [a for a in fig.axes if a.get_subplotspec() is not None]
        ax = axes[0]
        ax.set_title(title)
        return fig, ax

    def plot_trajectories(
        self,
        units: list[str] | None = None,
        n_sample: int = 10,
        select: Literal["random", "extreme", "high_variance"] = "random",
        show_mean: bool = True,
        hdi_prob: float = HDI_PROB,
        interval_type: Literal["mean", "predictive"] = "mean",
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
        hdi_prob : float
            Probability mass for the HDI credible interval (Bayesian models
            only). Defaults to :data:`~causalpy.constants.HDI_PROB`
            (currently 0.94). Common alternative values are 0.89 or 0.5.
        interval_type : {"mean", "predictive"}, default="mean"
            Which uncertainty interval to show for Bayesian models:

            - "mean": HDI of posterior ``mu`` (uncertainty in expected value)
            - "predictive": HDI of posterior predictive ``y_hat``
              (includes observation noise)

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
        if interval_type not in {"mean", "predictive"}:
            raise ValueError("interval_type must be 'mean' or 'predictive'")

        # Check if model is Bayesian
        is_bayesian = self._model_backend.is_bayesian

        # Get posterior for HDI plotting (Bayesian only)
        if is_bayesian:
            mu = self.model.idata.posterior["mu"]  # type: ignore[union-attr]
            if interval_type == "predictive":
                posterior_predictive = getattr(
                    self.model.idata,
                    "posterior_predictive",
                    None,  # type: ignore[union-attr]
                )
                if posterior_predictive is None or "y_hat" not in posterior_predictive:
                    raise ValueError(
                        "interval_type='predictive' requires posterior predictive "
                        "samples ('y_hat') in idata.posterior_predictive"
                    )
                interval_source = posterior_predictive["y_hat"]
            else:
                interval_source = mu

        # Select units to plot
        all_units = self.data[self.unit_fe_variable].unique()

        selected_units: np.ndarray | list[str]
        if units is not None:
            selected_units = units
        elif self.n_units <= n_sample:
            selected_units = all_units  # type: ignore[assignment]
        else:
            if select == "random":
                rng = np.random.default_rng(42)
                selected_units = rng.choice(all_units, size=n_sample, replace=False)  # type: ignore[assignment]
            elif select == "extreme":
                # Select units with the largest and smallest mean outcomes
                unit_means = self.data.groupby(self.unit_fe_variable)[
                    self.outcome_variable_name
                ].mean()
                n_each = max(1, n_sample // 2)
                top = unit_means.nlargest(n_each).index.tolist()
                bottom = unit_means.nsmallest(n_sample - n_each).index.tolist()
                selected_units = top + bottom
            elif select == "high_variance":
                # Select units with the most within-unit variation
                unit_var = self.data.groupby(self.unit_fe_variable)[
                    self.outcome_variable_name
                ].var()
                selected_units = unit_var.nlargest(n_sample).index.tolist()

        # Build tidy frames for a faceted plotnine base.
        n_units_plot = len(selected_units)
        ncols = min(3, n_units_plot)
        nrows = (n_units_plot + ncols - 1) // ncols
        figsize = (5 * ncols, 3 * nrows)

        obs_rows: list[dict[str, Any]] = []
        fit_rows: list[dict[str, Any]] = []
        ribbon_rows: list[dict[str, Any]] = []

        for unit in selected_units:
            unit_label = f"Unit: {unit}"
            unit_mask = self.data[self.unit_fe_variable] == unit
            unit_obs_indices = np.where(unit_mask)[0]
            time_vals = np.asarray(
                self.data.loc[unit_mask, self.time_fe_variable].values
            )
            sort_order = np.argsort(time_vals)
            sorted_time_vals = time_vals[sort_order]
            sorted_obs_indices = unit_obs_indices[sort_order]
            y_actual = self.design["y"].values.flatten()[sorted_obs_indices]

            for t, y in zip(sorted_time_vals, y_actual, strict=True):
                obs_rows.append(
                    {"unit_label": unit_label, "time": t, "y": y, "series": "Actual"}
                )

            if is_bayesian:
                unit_mu = mu.isel(obs_ind=sorted_obs_indices.tolist())
                if "treated_units" in unit_mu.dims:
                    unit_mu = unit_mu.squeeze("treated_units", drop=True)
                unit_interval = interval_source.isel(
                    obs_ind=sorted_obs_indices.tolist()
                )
                if "treated_units" in unit_interval.dims:
                    unit_interval = unit_interval.squeeze("treated_units", drop=True)
                hdi = az.hdi(unit_interval, hdi_prob=hdi_prob)
                var_name = next(iter(hdi.data_vars))
                fit_mean = unit_mu.mean(dim=["chain", "draw"]).values
                lower = hdi[var_name].sel(hdi="lower").values
                upper = hdi[var_name].sel(hdi="higher").values
                for t, y, ymin, ymax in zip(
                    sorted_time_vals, fit_mean, lower, upper, strict=True
                ):
                    fit_rows.append({"unit_label": unit_label, "time": t, "y": y})
                    ribbon_rows.append(
                        {
                            "unit_label": unit_label,
                            "time": t,
                            "ymin": ymin,
                            "ymax": ymax,
                        }
                    )
            else:
                y_fitted = np.squeeze(self.model.predict(self.design["X"]))[
                    sorted_obs_indices
                ]
                for t, y in zip(sorted_time_vals, y_fitted, strict=True):
                    fit_rows.append({"unit_label": unit_label, "time": t, "y": y})

        unit_labels = [f"Unit: {u}" for u in selected_units]
        obs_df = pd.DataFrame(obs_rows)
        fit_df = pd.DataFrame(fit_rows)
        obs_df["unit_label"] = pd.Categorical(
            obs_df["unit_label"], categories=unit_labels, ordered=True
        )
        fit_df["unit_label"] = pd.Categorical(
            fit_df["unit_label"], categories=unit_labels, ordered=True
        )

        p = ggplot()
        if is_bayesian and ribbon_rows:
            ribbon_df = pd.DataFrame(ribbon_rows)
            ribbon_df["unit_label"] = pd.Categorical(
                ribbon_df["unit_label"], categories=unit_labels, ordered=True
            )
            p = p + geom_ribbon(
                ribbon_df,
                aes("time", ymin="ymin", ymax="ymax"),
                fill="#1f77b4",
                alpha=0.2,
                inherit_aes=False,
            )
        p = (
            p
            + geom_line(
                fit_df,
                aes("time", "y"),
                color="#ff7f0e",
                linetype="dashed",
                alpha=0.7,
            )
            + geom_point(obs_df, aes("time", "y"), color="black", alpha=0.7)
            + geom_line(obs_df, aes("time", "y"), color="black", alpha=0.7)
            + facet_wrap("unit_label", ncol=ncols, scales="free_y")
            + guides(color="none")
            + labs(x="", y="")
            + theme(
                strip_text=element_blank(),
                strip_background=element_blank(),
                figure_size=figsize,
            )
        )

        fig = p.draw()
        axes = np.asarray(
            [a for a in fig.axes if a.get_subplotspec() is not None][:n_units_plot]
        )
        for ax, unit in zip(axes, selected_units, strict=True):
            ax.set_title(f"Unit: {unit}", fontsize=10)
            ax.set_xlabel(self.time_fe_variable)
            ax.set_ylabel(self.outcome_variable_name)
        if len(axes) > 0:
            axes[0].legend(
                handles=[
                    plt.Line2D([0], [0], color="black", marker="o", linestyle="-"),
                    plt.Line2D([0], [0], color="#ff7f0e", marker="s", linestyle="--"),
                ],
                labels=["Actual", "Fitted"],
                fontsize=8,
            )
        return fig, axes

    def plot_residuals(
        self,
        kind: Literal["scatter", "histogram", "qq"] = "scatter",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot residual diagnostics.

        Parameters
        ----------
        kind : {"scatter", "histogram", "qq"}, default="scatter"
            Type of residual plot:

            - "scatter": Residuals vs fitted values
            - "histogram": Distribution of residuals
            - "qq": Q-Q plot for normality check

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            Figure and axes objects
        """
        # Get plot data
        if self._model_backend.is_bayesian:
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
            stats.probplot(residuals, dist="norm", plot=ax)
            # Update colors to match the rest of the plots
            ax.get_lines()[0].set_markerfacecolor("C0")
            ax.get_lines()[0].set_markeredgecolor("C0")
            ax.get_lines()[1].set_color("C1")
            ax.set_title("Q-Q Plot")

        plt.tight_layout()
        return fig, ax
