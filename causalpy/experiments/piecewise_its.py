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
"""Piecewise Interrupted Time Series Analysis (Segmented Regression)."""

import ast
import re
import warnings
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from patsy import ModelDesc
from sklearn.base import RegressorMixin

from causalpy.constants import HDI_PROB, LEGEND_FONT_SIZE
from causalpy.custom_exceptions import FormulaException
from causalpy.experiments.model_adapter import build_coords
from causalpy.formula_utils import build_formula_matrices
from causalpy.plot_utils import _PosteriorPlotStyle, plot_posterior_over_x
from causalpy.pymc_models import LinearRegression, PyMCModel
from causalpy.reporting import EffectSummary
from causalpy.transforms import ramp, step  # noqa: F401
from causalpy.utils import _as_scalar, round_num

from .base import BaseExperiment


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
        Pointwise causal effect (fitted expectation - counterfactual expectation).
    cumulative_effect : xr.DataArray or np.ndarray
        Cumulative causal effect over time.

    Notes
    -----
    **Estimate extraction**

    One model is fitted to the full time series. The no-intervention counterfactual is predicted after setting every ``step()`` and ``ramp()`` design-matrix column to zero, and the pointwise effect is the fitted conditional expectation minus that counterfactual expectation. Bayesian backends contrast posterior ``mu`` values, OLS contrasts point predictions, and the cumulative effect is the running sum.

    The `step` and `ramp` transforms are patsy stateful transforms that handle
    both numeric and datetime time columns. For datetime, thresholds can be
    specified as strings (e.g., '2020-06-01') or pd.Timestamp objects.

    Bare datetime predictors are represented as continuous elapsed days. Use
    ``C(date)`` when date fixed effects are intended instead.

    References
    ----------
    - Wagner AK, et al. (2002). Segmented regression analysis of interrupted
      time series studies in medication use research. Journal of Clinical
      Pharmacy and Therapeutics.
    - Lopez Bernal J, et al. (2017). Interrupted time series regression for
      the evaluation of public health interventions: a tutorial. Int J Epidemiol.

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
    """

    supports_ols = True
    supports_bayes = True
    _default_model_class = LinearRegression
    _deprecated_design_aliases = {"X": ("design", "X"), "y": ("design", "y")}

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        model: PyMCModel | RegressorMixin | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model)

        # Store configuration
        self.expt_type = "Piecewise Interrupted Time Series"
        self.formula = formula
        self.data = data.copy()

        # Rename the index to "obs_ind" for consistency
        self.data.index.name = "obs_ind"

        # Parse and validate step/ramp terms before any downstream logic.
        self._step_ramp_terms = self._parse_step_ramp_terms()
        self._validate_inputs()
        self.time_col = self._extract_time_column(self._step_ramp_terms)
        self.interruption_times = self._extract_and_canonicalize_interruption_times(
            self._step_ramp_terms, self.time_col
        )

        # Parse formula with datetime-aware Patsy handling.
        y, X = build_formula_matrices(formula, self.data)
        self.outcome_variable_name = y.design_info.column_names[0]
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = list(X.design_info.column_names)

        # Convert to numpy arrays
        y_array = np.asarray(y)
        X_array = np.asarray(X)

        n_obs = X_array.shape[0]

        # Bundle into xr.Dataset
        self.design = self._build_design_dataset(
            X_array,
            y_array,
            obs_ind=np.arange(n_obs),
            coeffs=self.labels,
        )

        # Track which columns are interruption-related (for counterfactual)
        self._interruption_cols = self._get_interruption_column_indices()

        X = self.design["X"]
        y = self.design["y"]

        self._model_backend.fit(
            X=X,
            y=y,
            coords=build_coords(self.labels, X.shape[0]),
        )

        self.y_pred = self._model_backend.predict(X=X)
        self.score = self._model_backend.score(X=X, y=y)

        # Compute counterfactual and effects
        self._compute_counterfactual_and_effects()

    def _validate_inputs(self) -> None:
        """Validate input data and formula."""
        # Check formula contains at least one step() or ramp() term
        if not self._step_ramp_terms:
            raise FormulaException(
                "Formula must contain at least one step() or ramp() term. "
                "Example: 'y ~ 1 + t + step(t, 50) + ramp(t, 50)'"
            )

    @staticmethod
    def _parse_step_ramp_factor(factor_name: str) -> list[dict[str, str]]:
        """Extract bare ``step`` and ``ramp`` calls from one Patsy factor."""
        expression = ast.parse(factor_name, mode="eval")
        calls = []
        for node in ast.walk(expression):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"step", "ramp"}
                and len(node.args) == 2
                and isinstance(node.args[0], ast.Name)
            ):
                threshold = ast.get_source_segment(factor_name, node.args[1])
                calls.append(
                    {
                        "transform": node.func.id,
                        "variable": node.args[0].id,
                        "raw_threshold": threshold or ast.unparse(node.args[1]),
                    }
                )
        return calls

    def _parse_step_ramp_terms(self) -> list[dict[str, str]]:
        """Parse step/ramp terms into structured metadata."""
        return [
            call
            for term in ModelDesc.from_formula(self.formula).rhs_termlist
            for factor in term.factors
            for call in self._parse_step_ramp_factor(factor.name())
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
            raw_threshold = term["raw_threshold"].strip()
            if (
                len(raw_threshold) >= 2
                and raw_threshold[0] in "'\""
                and raw_threshold[-1] == raw_threshold[0]
            ):
                raw_threshold = raw_threshold[1:-1]

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
        indices: list[int] = []
        for term in self._x_design_info.terms:
            if any(
                self._parse_step_ramp_factor(factor.name()) for factor in term.factors
            ):
                term_slice = self._x_design_info.term_slices[term]
                indices.extend(range(term_slice.start, term_slice.stop))
        return indices

    def _compute_counterfactual_and_effects(self) -> None:
        """
        Compute the counterfactual (no intervention) and causal effects.

        The counterfactual is computed by setting step/ramp terms to zero.
        Also creates post_impact, datapost, and post_pred attributes for
        compatibility with effect_summary() from BaseExperiment.
        """
        # Create design matrix for counterfactual (zero out interruption columns)
        X_cf = self.design["X"].copy()
        for idx in self._interruption_cols:
            X_cf[:, idx] = 0

        # Compute counterfactual predictions
        if self._model_backend.is_bayesian:
            self.y_counterfactual = self._model_backend.predict(X=X_cf)

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

        elif self._model_backend.is_ols:
            self.y_counterfactual = self._model_backend.predict(X=X_cf)

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
        if self._model_backend.is_bayesian:
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

        elif self._model_backend.is_ols:
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

    def plot(
        self,
        *,
        round_to: int | None = 2,
        ci_prob: float = HDI_PROB,
        hdi_prob: float | None = None,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        figsize: tuple[float, float] = (10, 10),
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot the piecewise interrupted time-series results.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round numerical results in the figure
            title. Defaults to 2. Use ``None`` to render raw numbers.
        ci_prob : float
            Probability mass of the highest density interval drawn around the
            fitted, counterfactual, causal effect, and cumulative effect
            bands. Must be in ``(0, 1]``. Ignored for OLS models. Defaults
            to :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
        hdi_prob : float, optional
            Deprecated. Use ``ci_prob`` instead.
        kind : {"ribbon", "histogram", "spaghetti"}, optional
            How posterior uncertainty is rendered via
            :func:`~causalpy.plot_utils.plot_posterior_over_x`. Defaults to ``"ribbon"``.
            For ``"spaghetti"``, legends use draw lines rather than a shaded
            band. For ``"histogram"``, uncertainty is shown as a 2D density
            heatmap with a mean line overlay (no ribbon patch for legends).
        ci_kind : {"hdi", "eti"}, optional
            Credible interval type when ``kind="ribbon"``. Defaults to
            ``"hdi"``.
        num_samples : int, optional
            Number of posterior draws when ``kind="spaghetti"``. Defaults
            to 50. Ignored for other kinds.

        figsize : tuple of (float, float)
            Width and height of the figure in inches, passed to
            :func:`matplotlib.pyplot.subplots`. Defaults to ``(10, 10)``.
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
        ax : list[matplotlib.axes.Axes]
            The three axes (top: observed, fitted and counterfactual;
            middle: causal effect; bottom: cumulative effect).
        """
        if hdi_prob is not None:
            warnings.warn(
                "hdi_prob is deprecated and will be removed in a future release. "
                "Use ci_prob instead.",
                FutureWarning,
                stacklevel=2,
            )
            ci_prob = hdi_prob
        return self._render_plot(
            show=show,
            legend_kwargs=legend_kwargs,
            round_to=round_to,
            ci_prob=ci_prob,
            kind=kind,
            ci_kind=ci_kind,
            num_samples=num_samples,
            figsize=figsize,
        )

    def _bayesian_plot(
        self,
        round_to: int | None = 2,
        ci_prob: float = HDI_PROB,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        figsize: tuple[float, float] = (10, 10),
        **kwargs: Any,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plot the results for Bayesian models.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals for rounding. Defaults to 2.
        hdi_prob : float, optional
            Probability mass of the highest density interval drawn around the
            fitted, counterfactual, causal effect, and cumulative effect bands.
            Must be in ``(0, 1]``. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
        figsize : tuple of (float, float), optional
            Width and height of the figure in inches. Defaults to ``(10, 10)``.

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure.
        ax : list[plt.Axes]
            List of axes objects.
        """
        style: _PosteriorPlotStyle = {
            "ci_prob": ci_prob,
            "kind": kind,
            "ci_kind": ci_kind,
            "num_samples": num_samples,
        }
        time_values = self.data[self.time_col].values

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize)

        # TOP PLOT: Observed, Fitted, and Counterfactual
        # Observed data
        (h_obs,) = ax[0].plot(
            time_values,
            self.design["y"].isel(treated_units=0),
            "k.",
            label="Observations",
        )

        # Fitted values (mu)
        y_pred_mu = self.y_pred["posterior_predictive"]["mu"]
        if "treated_units" in y_pred_mu.dims:
            y_pred_mu = y_pred_mu.isel(treated_units=0)
        h_line_fit, h_patch_fit = plot_posterior_over_x(
            time_values,
            y_pred_mu,
            ax=ax[0],
            **style,
            plot_hdi_kwargs={"color": "C0"},
        )

        # Counterfactual
        y_cf_mu = self.y_counterfactual["posterior_predictive"]["mu"]
        if "treated_units" in y_cf_mu.dims:
            y_cf_mu = y_cf_mu.isel(treated_units=0)
        h_line_cf, h_patch_cf = plot_posterior_over_x(
            time_values,
            y_cf_mu,
            ax=ax[0],
            **style,
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
        plot_posterior_over_x(
            time_values,
            self.effect,
            ax=ax[1],
            **style,
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
        plot_posterior_over_x(
            time_values,
            self.cumulative_effect,
            ax=ax[2],
            **style,
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
        self,
        round_to: int | None = 2,
        figsize: tuple[float, float] = (10, 10),
        **kwargs: Any,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plot the results for OLS models.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals for rounding. Defaults to 2.
        figsize : tuple of (float, float), optional
            Width and height of the figure in inches. Defaults to ``(10, 10)``.

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure.
        ax : list[plt.Axes]
            List of axes objects.
        """
        time_values = self.data[self.time_col].values

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize)

        # TOP PLOT: Observed, Fitted, and Counterfactual
        ax[0].plot(time_values, self.design["y"].values, "k.", label="Observations")
        ax[0].plot(time_values, self.y_pred, "C0-", label="Fitted", linewidth=2)
        ax[0].plot(
            time_values,
            self.y_counterfactual,
            "C1--",
            label="Counterfactual",
            linewidth=2,
        )

        title_str = (
            f"Piecewise ITS: $R^2$ = {round_num(_as_scalar(self.score), round_to)}"
        )
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

    def get_plot_data_bayesian(self, hdi_prob: float = HDI_PROB) -> pd.DataFrame:
        """
        Recover the data of the experiment along with prediction and effect information.

        Parameters
        ----------
        hdi_prob : float
            Probability for the highest density interval. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).

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
                self.outcome_variable_name: self.design["y"]
                .isel(treated_units=0)
                .values,
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
                self.outcome_variable_name: self.design["y"].values.flatten(),
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
        """Generate a decision-ready summary of PiecewiseITS causal effects.

        Parameters
        ----------
        window : str, tuple, or slice, default "post"
            Time window for analysis (see :meth:`BaseExperiment.effect_summary`).
        direction : {"increase", "decrease", "two-sided"}, default "increase"
            Direction for tail probability calculation (PyMC only).
        alpha : float, default 0.05
            Significance level for HDI/CI intervals (1-alpha confidence).
        cumulative : bool, default True
            Whether to include cumulative effect statistics.
        relative : bool, default True
            Whether to include relative effect statistics.
        min_effect : float, optional
            Region of Practical Equivalence (ROPE) threshold (PyMC only).
        treated_unit : str, optional
            Multi-unit experiments select which unit to analyse.
        period : None
            Not supported by PiecewiseITS; pass ``None``.
        prefix : str, default "Post-period"
            Prefix for prose generation.
        **kwargs
            Reserved for forward-compatibility.
        """
        from causalpy.reporting import (
            _compute_statistics,
            _compute_statistics_ols,
            _extract_counterfactual,
            _extract_window,
            _generate_prose_detailed,
            _generate_prose_detailed_ols,
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

        if self._model_backend.is_bayesian:
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

            time_dim = "obs_ind"
            cf_avg = _as_scalar(counterfactual.mean(dim=[time_dim, "chain", "draw"]))
            obs_avg = cf_avg + stats["avg"]["mean"]
            cf_cum = _as_scalar(
                counterfactual.sum(dim=time_dim).mean(dim=["chain", "draw"])
            )
            obs_cum = cf_cum + stats["cum"]["mean"] if cumulative else None

            text = _generate_prose_detailed(
                stats,
                window_coords,
                alpha=alpha,
                direction=direction,
                cumulative=cumulative,
                relative=relative,
                prefix=prefix,
                observed_avg=obs_avg,
                counterfactual_avg=cf_avg,
                observed_cum=obs_cum,
                counterfactual_cum=cf_cum if cumulative else None,
                experiment_type="piecewise_its",
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

            cf_avg = float(np.mean(counterfactual_array))
            obs_avg = cf_avg + stats["avg"]["mean"]
            cf_cum = float(np.sum(counterfactual_array))
            obs_cum = cf_cum + stats["cum"]["mean"] if cumulative else None

            text = _generate_prose_detailed_ols(
                stats,
                window_coords,
                alpha=alpha,
                cumulative=cumulative,
                relative=relative,
                prefix=prefix,
                observed_avg=obs_avg,
                counterfactual_avg=cf_avg,
                observed_cum=obs_cum,
                counterfactual_cum=cf_cum if cumulative else None,
                experiment_type="piecewise_its",
            )

        return EffectSummary(table=table, text=text)
