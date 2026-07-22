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
import plotnine as p9
import polars as pl
import xarray as xr
from matplotlib import pyplot as plt
from patsy import ModelDesc
from sklearn.base import RegressorMixin

from causalpy.constants import HDI_PROB
from causalpy.custom_exceptions import FormulaException
from causalpy.experiments.model_adapter import build_coords
from causalpy.formula_utils import build_formula_matrices
from causalpy.plot_utils import (
    CausalPanelData,
    PlotSpec,
    build_causal_panel_plot,
    dataarray_draws,
    extract_r2_score,
    has_posterior_draws,
)
from causalpy.pymc_models import LinearRegression, PyMCModel
from causalpy.reporting import EffectSummary
from causalpy.transforms import ramp, step  # noqa: F401
from causalpy.utils import round_num

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
        Pointwise causal effect (observed - counterfactual).
    cumulative_effect : xr.DataArray or np.ndarray
        Cumulative causal effect over time.

    Notes
    -----
    The counterfactual is computed by setting all step/ramp terms to zero,
    representing what would have happened without the interventions.

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

        self.y_counterfactual = self._model_backend.predict(X=X_cf)
        self.effect = self.y_pred.isel(treated_units=0) - self.y_counterfactual.isel(
            treated_units=0
        )
        self.cumulative_effect = self.effect.cumsum(dim="obs_ind")

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

        self.post_impact = self.effect.isel(obs_ind=post_indices).assign_coords(
            obs_ind=self.datapost.index
        )
        self.post_pred = self.y_counterfactual.isel(obs_ind=post_indices).assign_coords(
            obs_ind=self.datapost.index
        )

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

    def _causal_panel_data(self) -> CausalPanelData:
        """Extract semantic long-form draws and observations for plotting."""
        time_values = self.data[self.time_col].to_numpy()
        time_lookup = pl.from_pandas(
            pd.DataFrame({"obs_ind": self.data.index, "t": time_values})
        )
        observations = pd.DataFrame(
            {
                "t": time_values,
                "value": self.design["y"].isel(treated_units=0).to_numpy(),
            }
        )
        return CausalPanelData(
            fitted=dataarray_draws(self.y_pred).join(time_lookup, on="obs_ind"),
            counterfactual=dataarray_draws(self.y_counterfactual).join(
                time_lookup, on="obs_ind"
            ),
            post_effect=dataarray_draws(self.effect).join(time_lookup, on="obs_ind"),
            cumulative_effect=dataarray_draws(self.cumulative_effect).join(
                time_lookup, on="obs_ind"
            ),
            observations=observations,
        )

    def _plot(
        self,
        round_to: int | None = 2,
        ci_prob: float = HDI_PROB,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        figsize: tuple[float, float] = (10, 10),
        **kwargs: Any,
    ) -> PlotSpec:
        """Build the piecewise ITS plot from tidy declarative layers.

        Consumes the canonical prediction container from any backend.
        Point-estimate backends (singleton ``chain``/``draw``) collapse the
        uncertainty layers to bare mean lines.
        """
        # Title with R^2; scores carrying a dispersion entry render as Bayesian
        r2_val, r2_std_val = extract_r2_score(self.score)
        assert r2_val is not None  # both backends' score containers carry R^2
        label = "Bayesian $R^2$" if r2_std_val is not None else "$R^2$"
        panels = (
            f"Piecewise ITS: {label} = {round_num(r2_val, round_to)}",
            "Causal Effect",
            "Cumulative Causal Effect",
        )
        plot_data = self._causal_panel_data()
        series_labels = {
            "fitted": "Fitted",
            "counterfactual": "Counterfactual",
            "post_effect": "effect",
            "cumulative_effect": "cumulative",
        }
        colors = {
            "Fitted": "#1f77b4",
            "Counterfactual": "#ff7f0e",
            "Observations": "black",
            "effect": "#2ca02c",
            "cumulative": "#d62728",
        }
        p = build_causal_panel_plot(
            plot_data,
            panels=panels,
            series_labels=series_labels,
            colors=colors,
            kind=kind,
            ci_prob=ci_prob,
            interval=ci_kind,
            num_samples=num_samples,
            x="t",
            shade_fill="#2ca02c",
            figsize=figsize,
            zero_linetype="dashed",
            zero_alpha=0.5,
            shade_outcome=False,
        )
        p += p9.geom_vline(
            pd.DataFrame({"t": self.interruption_times}),
            p9.aes(xintercept="t"),
            color="red",
            size=1.5,
            alpha=0.7,
        )

        def add_labels(_fig: plt.Figure, axes: list[plt.Axes]) -> None:
            axes[0].set_ylabel(self.outcome_variable_name)
            axes[1].set_ylabel("Effect")
            axes[2].set_ylabel("Cumulative Effect")

        return PlotSpec(p, overlay=add_labels, n_panels=3)

    def get_plot_data(self, hdi_prob: float = HDI_PROB) -> pd.DataFrame:
        """
        Recover the data of the experiment along with prediction and effect information.

        HDI columns are included only when the prediction container carries
        posterior draws.

        Parameters
        ----------
        hdi_prob : float
            Probability for the highest density interval. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94). Ignored
            when the prediction container has no posterior draws.

        Returns
        -------
        pd.DataFrame
            DataFrame containing observed data, predictions, and effects.
        """
        with_uncertainty = has_posterior_draws(self.y_pred)
        hdi_pct = int(round(hdi_prob * 100))

        # Get time values
        time_values = self.data[self.time_col].values

        # Extract predictions
        y_pred_mu = self.y_pred.isel(treated_units=0)
        y_cf_mu = self.y_counterfactual.isel(treated_units=0)

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

        # Build DataFrame column-by-column so HDI columns interleave with the
        # quantities they describe
        data: dict[str, Any] = {
            self.time_col: time_values,
            self.outcome_variable_name: self.design["y"].isel(treated_units=0).values,
            "fitted": y_pred_mu.mean(dim=["chain", "draw"]).values,
        }
        if with_uncertainty:
            fitted_lower, fitted_upper = _get_hdi_bounds(
                az.hdi(y_pred_mu, hdi_prob=hdi_prob)
            )
            data[f"fitted_hdi_lower_{hdi_pct}"] = fitted_lower
            data[f"fitted_hdi_upper_{hdi_pct}"] = fitted_upper

        data["counterfactual"] = y_cf_mu.mean(dim=["chain", "draw"]).values
        if with_uncertainty:
            cf_lower, cf_upper = _get_hdi_bounds(az.hdi(y_cf_mu, hdi_prob=hdi_prob))
            data[f"counterfactual_hdi_lower_{hdi_pct}"] = cf_lower
            data[f"counterfactual_hdi_upper_{hdi_pct}"] = cf_upper

        data["effect"] = self.effect.mean(dim=["chain", "draw"]).values
        if with_uncertainty:
            effect_lower, effect_upper = _get_hdi_bounds(
                az.hdi(self.effect, hdi_prob=hdi_prob)
            )
            data[f"effect_hdi_lower_{hdi_pct}"] = effect_lower
            data[f"effect_hdi_upper_{hdi_pct}"] = effect_upper

        data["cumulative_effect"] = self.cumulative_effect.mean(
            dim=["chain", "draw"]
        ).values
        if with_uncertainty:
            cum_lower, cum_upper = _get_hdi_bounds(
                az.hdi(self.cumulative_effect, hdi_prob=hdi_prob)
            )
            data[f"cumulative_effect_hdi_lower_{hdi_pct}"] = cum_lower
            data[f"cumulative_effect_hdi_upper_{hdi_pct}"] = cum_upper

        result = pd.DataFrame(data)

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
            _effect_summary_timeseries,
            _extract_counterfactual,
            _extract_window,
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
        return _effect_summary_timeseries(
            self,
            windowed_impact,
            counterfactual,
            window_coords,
            direction=direction,
            alpha=alpha,
            cumulative=cumulative,
            relative=relative,
            min_effect=min_effect,
            prefix=prefix,
            experiment_type="piecewise_its",
        )
