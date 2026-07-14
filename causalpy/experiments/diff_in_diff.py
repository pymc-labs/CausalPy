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
"""Difference in differences."""

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import xarray as xr
from matplotlib import pyplot as plt
from patsy import ModelDesc, build_design_matrices, dmatrices
from plotnine import (
    aes,
    annotate,
    arrow,
    geom_point,
    geom_violin,
    ggplot,
    guides,
    labs,
    scale_color_manual,
    scale_fill_manual,
    theme,
)
from sklearn.base import RegressorMixin

from causalpy.constants import HDI_PROB, LEGEND_FONT_SIZE
from causalpy.custom_exceptions import (
    DataException,
    FormulaException,
)
from causalpy.experiments.model_adapter import build_coords
from causalpy.plot_utils import (
    HISTOGRAM_PANEL_THEME,
    PlotSpec,
    add_causal_panel_legend,
    coord_xlim_for_column,
    label_draws,
    posterior_kind_layers,
    prediction_draws,
    scale_for_x_column,
)
from causalpy.pymc_models import LinearRegression, PyMCModel
from causalpy.reporting import (
    EffectSummary,
    _compute_statistics_did_ols,
    _effect_summary_did,
    _generate_prose_did_ols,
    _generate_table_did_ols,
)
from causalpy.utils import (
    _as_scalar,
    _is_variable_dummy_coded,
    convert_to_string,
    round_num,
)

from .base import BaseExperiment


@dataclass(frozen=True)
class _DiDPlotData:
    """Tidy tables and annotation coordinates consumed by the DiD plot."""

    scatter: pd.DataFrame
    draws: pl.DataFrame
    counterfactual_draws: pd.DataFrame
    time_points: np.ndarray
    arrow_x: Any
    treatment_y: float
    counterfactual_y: float


class DifferenceInDifferences(BaseExperiment):
    """A class to analyse data from Difference in Difference settings.

    .. note::

        There is no pre/post intervention data distinction for DiD, we fit
        all the data available.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas dataframe.
    formula : str
        A statistical model formula.
    time_variable_name : str
        Name of the data column for the time variable.
    group_variable_name : str
        Name of the data column for the group variable.
    post_treatment_variable_name : str, optional
        Name of the data column indicating post-treatment period.
        Defaults to "post_treatment".
    model : PyMCModel or RegressorMixin, optional
        A PyMC model for difference in differences. Defaults to LinearRegression.
    **kwargs
        Additional keyword arguments forwarded to :class:`BaseExperiment`.

    Examples
    --------
    >>> import causalpy as cp
    >>> df = cp.load_data("did")
    >>> seed = 42
    >>> result = cp.DifferenceInDifferences(
    ...     df,
    ...     formula="y ~ 1 + group*post_treatment",
    ...     time_variable_name="t",
    ...     group_variable_name="group",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={
    ...             "target_accept": 0.95,
    ...             "random_seed": seed,
    ...             "progressbar": False,
    ...         }
    ...     ),
    ... )
    """

    supports_ols = True
    supports_bayes = True
    _default_model_class = LinearRegression
    _deprecated_design_aliases = {"X": ("design", "X"), "y": ("design", "y")}

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        time_variable_name: str,
        group_variable_name: str,
        post_treatment_variable_name: str = "post_treatment",
        model: PyMCModel | RegressorMixin | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model)
        self.causal_impact: xr.DataArray | float | None
        # rename the index to "obs_ind"
        data.index.name = "obs_ind"
        self.data = data
        self.expt_type = "Difference in Differences"
        self.formula = formula
        self.time_variable_name = time_variable_name
        self.group_variable_name = group_variable_name
        self.post_treatment_variable_name = post_treatment_variable_name
        self.input_validation()
        self._build_design_matrices()
        self._prepare_data()
        self.algorithm()

    def _build_design_matrices(self) -> None:
        """Build design matrices from formula and data using patsy."""
        y, X = dmatrices(self.formula, self.data)
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self._y_raw, self._X_raw = np.asarray(y), np.asarray(X)
        self.outcome_variable_name = y.design_info.column_names[0]

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
        """Run the experiment algorithm: fit model, predict, and calculate causal impact."""
        X = self.design["X"]
        y = self.design["y"]

        self._model_backend.fit(
            X=X,
            y=y,
            coords=build_coords(self.labels, X.shape[0]),
        )

        # predicted outcome for control group
        self.x_pred_control = (
            self.data
            # just the untreated group
            .query(f"{self.group_variable_name} == 0")
            # drop the outcome variable
            .drop(self.outcome_variable_name, axis=1)
            # We may have multiple units per time point, we only want one time point
            .groupby(self.time_variable_name)
            .first()
            .reset_index()
        )
        if self.x_pred_control.empty:
            raise ValueError("x_pred_control is empty")
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred_control)
        self.y_pred_control = self._model_backend.predict(np.asarray(new_x))

        # predicted outcome for treatment group
        self.x_pred_treatment = (
            self.data
            # just the treated group
            .query(f"{self.group_variable_name} == 1")
            # drop the outcome variable
            .drop(self.outcome_variable_name, axis=1)
            # We may have multiple units per time point, we only want one time point
            .groupby(self.time_variable_name)
            .first()
            .reset_index()
        )
        if self.x_pred_treatment.empty:
            raise ValueError("x_pred_treatment is empty")
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred_treatment)
        self.y_pred_treatment = self._model_backend.predict(np.asarray(new_x))

        # predicted outcome for counterfactual. This is given by removing the influence
        # of the interaction term between the group and the post_treatment variable
        self.x_pred_counterfactual = (
            self.data
            # just the treated group
            .query(f"{self.group_variable_name} == 1")
            # just the treatment period(s)
            .query(f"{self.post_treatment_variable_name} == True")
            # drop the outcome variable
            .drop(self.outcome_variable_name, axis=1)
            # We may have multiple units per time point, we only want one time point
            .groupby(self.time_variable_name)
            .first()
            .reset_index()
        )
        if self.x_pred_counterfactual.empty:
            raise ValueError("x_pred_counterfactual is empty")
        (new_x,) = build_design_matrices(
            [self._x_design_info], self.x_pred_counterfactual, return_type="dataframe"
        )
        # INTERVENTION: set the interaction term between the group and the
        # post_treatment variable to zero. This is the counterfactual.
        for i, label in enumerate(self.labels):
            if self._is_treatment_interaction(label):
                new_x.iloc[:, i] = 0
        self.y_pred_counterfactual = self._model_backend.predict(np.asarray(new_x))

        # calculate causal impact
        if self._model_backend.is_bayesian:
            assert self.model.idata is not None
            # This is the coefficient on the interaction term
            coeff_names = self.model.idata.posterior.coords["coeffs"].data
            for i, label in enumerate(coeff_names):
                if self._is_treatment_interaction(label):
                    self.causal_impact = self.model.idata.posterior["beta"].isel(
                        {"coeffs": i}
                    )
        elif self._model_backend.is_ols:
            # This is the coefficient on the interaction term
            coef_map = dict(
                zip(self.labels, self._model_backend.coefficients(), strict=False)
            )
            matched_key = next(
                (key for key in coef_map if self._is_treatment_interaction(key)),
                None,
            )
            att = coef_map.get(matched_key) if matched_key is not None else None
            self.causal_impact = att
        else:
            raise ValueError("Model type not recognized")

    def input_validation(self) -> None:
        """Validate the input data and model formula for correctness."""
        # Validate formula structure and interaction interaction terms
        self._validate_formula_interaction_terms()

        # Check if post_treatment_variable_name is in data columns
        if self.post_treatment_variable_name not in self.data.columns:
            raise DataException(
                f"Missing required column '{self.post_treatment_variable_name}' in dataset"
            )

        if "unit" not in self.data.columns:
            raise DataException(
                "Require a `unit` column to label unique units. This is used for plotting purposes"  # noqa: E501
            )

        if not _is_variable_dummy_coded(self.data[self.group_variable_name]):
            raise DataException(
                f"""The grouping variable {self.group_variable_name} should be dummy
                coded. Consisting of 0's and 1's only."""
            )

    def _validate_formula_interaction_terms(self) -> None:
        """
        Validate that the formula contains exactly one interaction term, that it
        is between the group and post-treatment variables, and that it is not a
        three-way or higher-order interaction.

        Raises FormulaException if no interaction term is found, if more than one
        interaction term is found, if any interaction term has more than 2
        variables, or if the single interaction term does not involve both the
        group and post-treatment variables.
        """
        interaction_terms = [
            term
            for term in ModelDesc.from_formula(self.formula).rhs_termlist
            if len(term.factors) > 1
        ]

        # Check for interaction terms with more than 2 variables
        for term in interaction_terms:
            if len(term.factors) > 2:
                raise FormulaException(
                    f"Formula contains interaction term with more than 2 variables: {term.name()}. "
                    "Three-way or higher-order interactions are not supported as they complicate interpretation of the causal effect."
                )

        if len(interaction_terms) > 1:
            interaction_term_names = [term.name() for term in interaction_terms]
            raise FormulaException(
                f"Formula contains {len(interaction_terms)} interaction terms: {interaction_term_names}. "
                "Multiple interaction terms are not currently supported as they complicate interpretation of the causal effect."
            )

        # A DiD formula must contain exactly one interaction term, and it must be
        # between the group and post-treatment variables: that term is what
        # identifies the causal effect (see `algorithm`, which reads it back off
        # the fitted model). Without this check a formula with no interaction
        # term, or an interaction between unrelated variables, would pass
        # validation and only fail later when `causal_impact` is accessed.
        if len(interaction_terms) == 0 or not (
            self._is_treatment_interaction(interaction_terms[0].name())
        ):
            raise FormulaException(
                "Formula must contain exactly one interaction term between the "
                f"group variable '{self.group_variable_name}' and the "
                f"post-treatment variable '{self.post_treatment_variable_name}' "
                f"(e.g. '{self.group_variable_name}*{self.post_treatment_variable_name}'). "
                "This interaction term identifies the difference-in-differences causal effect."
            )

    def _is_treatment_interaction(self, term: str) -> bool:
        """Whether a term is exactly the group/post-treatment interaction."""
        factors = {
            factor.split("[", maxsplit=1)[0]
            for factor in term.replace("*", ":").split(":")
        }
        return len(factors) == 2 and all(
            any(factor in {name, f"C({name})"} for factor in factors)
            for name in (
                self.group_variable_name,
                self.post_treatment_variable_name,
            )
        )

    def summary(self, round_to: int | None = 2) -> None:
        """Print summary of main results and model coefficients.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2. Use
            ``None`` to return raw numbers.
        """
        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print("\nResults:")
        print(self._causal_impact_summary_stat(round_to))
        self.print_coefficients(round_to)

    def _causal_impact_summary_stat(self, round_to: int | None = None) -> str:
        """Computes the mean and credible interval bounds for the causal impact."""
        return f"Causal impact = {convert_to_string(self.causal_impact, round_to=round_to)}"

    def plot(
        self,
        *,
        round_to: int | None = None,
        ci_prob: float = HDI_PROB,
        hdi_prob: float | None = None,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the difference-in-differences results.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round numerical results in the figure
            title. Defaults to ``None``, in which case 2 significant figures
            are used.
        ci_prob : float
            Probability mass of the highest density interval drawn around the
            posterior predictive bands for the control, treatment, and
            counterfactual trajectories. Must be in ``(0, 1]``. Ignored for
            OLS models. Defaults to :data:`~causalpy.constants.HDI_PROB`
            (currently 0.94).
        hdi_prob : float, optional
            Deprecated. Use ``ci_prob`` instead.
        kind : {"ribbon", "spaghetti", "histogram"}, optional
            How posterior uncertainty is rendered. Defaults to ``"ribbon"``
            (mean + credible band).
        ci_kind : {"hdi", "eti"}, optional
            Credible interval type when ``kind="ribbon"``. Defaults to
            ``"hdi"``.
        num_samples : int, optional
            Number of posterior draws to overlay when ``kind="spaghetti"``.
            Defaults to 50.
        figsize : tuple of (float, float), optional
            Width and height of the figure in inches, passed to
            :func:`matplotlib.pyplot.subplots`. Defaults to ``None`` (use
            matplotlib's default).
        show : bool
            Whether to automatically display the plot. Defaults to ``True``.
            Set to ``False`` if you want to modify the figure before
            displaying it.
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
            The axes object containing the plot.
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

    def _prepare_bayesian_plot_data(
        self,
    ) -> _DiDPlotData:
        """Prepare observed, posterior, and annotation data for plotting."""
        tcol = self.time_variable_name
        ycol = self.outcome_variable_name
        levels = sorted(self.data[self.group_variable_name].unique())
        group_to_series = {levels[0]: "Control group", levels[1]: "Treatment group"}
        scatter = self.data[[tcol, ycol]].copy()
        scatter["series"] = (
            self.data[self.group_variable_name].map(group_to_series).to_numpy()
        )

        control_grid = self.x_pred_control.reset_index(drop=True)
        control_grid["obs_ind"] = range(len(control_grid))
        treatment_grid = self.x_pred_treatment.reset_index(drop=True)
        treatment_grid["obs_ind"] = range(len(treatment_grid))
        counterfactual_grid = self.x_pred_counterfactual.reset_index(drop=True)
        counterfactual_grid["obs_ind"] = range(len(counterfactual_grid))
        control = prediction_draws(self.y_pred_control, control_grid)
        treatment = prediction_draws(self.y_pred_treatment, treatment_grid)
        counterfactual = prediction_draws(
            self.y_pred_counterfactual, counterfactual_grid
        )
        time_points = self.x_pred_counterfactual[tcol].to_numpy()
        draw_parts = [
            label_draws(control, series="Control group"),
            label_draws(treatment, series="Treatment group"),
        ]
        if len(time_points) > 1:
            draw_parts.append(label_draws(counterfactual, series="Counterfactual"))
        draws = pl.concat(draw_parts, how="diagonal_relaxed")

        treatment_y = _as_scalar(
            self.y_pred_treatment["posterior_predictive"]
            .mu.isel({"obs_ind": 1})
            .mean()
            .data
        )
        counterfactual_y = _as_scalar(
            self.y_pred_counterfactual["posterior_predictive"].mu.mean().data
        )
        treatment_times = self.x_pred_treatment[tcol].to_numpy()
        arrow_x: Any
        if np.issubdtype(treatment_times.dtype, np.datetime64):
            time_min = pd.Timestamp(treatment_times.min())
            time_max = pd.Timestamp(treatment_times.max())
            arrow_x = time_max + 0.1 * (time_max - time_min)
        elif np.issubdtype(treatment_times.dtype, np.number):
            time_span = float(np.ptp(treatment_times.astype(float)))
            arrow_x = float(np.max(treatment_times)) + 0.1 * time_span
        else:
            arrow_x = treatment_times[-1]
        return _DiDPlotData(
            scatter=scatter,
            draws=draws,
            counterfactual_draws=counterfactual.to_pandas(),
            time_points=time_points,
            arrow_x=arrow_x,
            treatment_y=treatment_y,
            counterfactual_y=counterfactual_y,
        )

    def _bayesian_plot(
        self,
        round_to: int | None = None,
        ci_prob: float = HDI_PROB,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> PlotSpec:
        """Build the Bayesian DiD plot from tidy data and declarative layers."""
        tcol = self.time_variable_name
        ycol = self.outcome_variable_name
        colors = {
            "Control group": "#1f77b4",
            "Treatment group": "#ff7f0e",
            "Counterfactual": "#2ca02c",
        }
        plot_data = self._prepare_bayesian_plot_data()
        _, posterior_layers = posterior_kind_layers(
            plot_data.draws,
            kind,
            x=tcol,
            group_by=["series", tcol],
            ci_prob=ci_prob,
            interval=ci_kind,
            num_samples=num_samples,
            colors=colors,
        )
        p = ggplot() + geom_point(
            plot_data.scatter, aes(tcol, ycol, color="series"), size=1.5
        )
        for layer in posterior_layers:
            p += layer
        x_values = plot_data.scatter[tcol]
        p = (
            p
            + scale_color_manual(values=colors, name="")
            + (
                scale_fill_manual(values=colors, name="")
                if kind == "ribbon"
                else guides()
            )
            + scale_for_x_column(x_values)
            + coord_xlim_for_column(x_values)
            + labs(title=self._causal_impact_summary_stat(round_to), x=tcol, y=ycol)
        )
        if figsize is not None:
            p += theme(figure_size=figsize)
        if kind == "histogram":
            p = p + HISTOGRAM_PANEL_THEME
        if len(plot_data.time_points) == 1:
            p = p + geom_violin(
                plot_data.counterfactual_draws,
                aes(tcol, "mu"),
                width=0.2,
                fill="#1f77b4",
                color=None,
                alpha=0.5,
                show_legend=False,
                inherit_aes=False,
            )
        p += annotate(
            "segment",
            x=plot_data.arrow_x,
            xend=plot_data.arrow_x,
            y=plot_data.treatment_y,
            yend=plot_data.counterfactual_y,
            arrow=arrow(length=0.1, ends="first", type="closed"),
            color="green",
            size=1.5,
        )
        p += annotate(
            "text",
            x=plot_data.arrow_x,
            y=np.mean([plot_data.counterfactual_y, plot_data.treatment_y]),
            label="causal\nimpact",
            color="green",
            ha="left",
        )

        def overlay(_fig: plt.Figure, axes: list[plt.Axes]) -> None:
            ax = axes[0]
            legend_labels = ["Control group", "Treatment group"]
            if len(plot_data.time_points) > 1:
                legend_labels.append("Counterfactual")
            add_causal_panel_legend(
                ax,
                labels=legend_labels,
                colors=colors,
            )

        return PlotSpec(p, overlay=overlay, n_panels=1)

    def _ols_plot(
        self,
        round_to: int | None = 2,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Generate plot for difference-in-differences.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2.
        figsize : tuple of (float, float), optional
            Width and height of the figure in inches. Defaults to ``None``
            (use matplotlib's default).
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot raw data
        sns.lineplot(
            self.data,
            x=self.time_variable_name,
            y=self.outcome_variable_name,
            hue="group",
            units="unit",
            estimator=None,
            alpha=0.25,
            ax=ax,
        )
        # Plot model fit to control group
        ax.plot(
            self.x_pred_control[self.time_variable_name],
            self.y_pred_control,
            "o",
            c="C0",
            markersize=10,
            label="model fit (control group)",
        )
        # Plot model fit to treatment group
        ax.plot(
            self.x_pred_treatment[self.time_variable_name],
            self.y_pred_treatment,
            "o",
            c="C1",
            markersize=10,
            label="model fit (treatment group)",
        )
        # Plot counterfactual - post-test for treatment group IF no treatment
        # had occurred.
        ax.plot(
            self.x_pred_counterfactual[self.time_variable_name],
            self.y_pred_counterfactual,
            "go",
            markersize=10,
            label="counterfactual",
        )
        y_pred_counterfactual_scalar = _as_scalar(self.y_pred_counterfactual)
        y_pred_treatment_post_scalar = _as_scalar(self.y_pred_treatment[1])
        # arrow to label the causal impact
        ax.annotate(
            "",
            xy=(1.05, y_pred_counterfactual_scalar),
            xycoords="data",
            xytext=(1.05, y_pred_treatment_post_scalar),
            textcoords="data",
            arrowprops={"arrowstyle": "<->", "color": "green", "lw": 3},
        )
        ax.annotate(
            "causal\nimpact",
            xy=(
                1.05,
                np.mean([y_pred_counterfactual_scalar, y_pred_treatment_post_scalar]),
            ),
            xycoords="data",
            xytext=(5, 0),
            textcoords="offset points",
            color="green",
            va="center",
        )
        # formatting
        # In OLS context, causal_impact should be a float, but mypy doesn't know this
        causal_impact_value = (
            float(self.causal_impact) if self.causal_impact is not None else 0.0
        )
        ax.set(
            xlim=[-0.05, 1.1],
            xticks=[0, 1],
            xticklabels=["pre", "post"],
            title=f"Causal impact = {round_num(causal_impact_value, round_to)}",
        )
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        return fig, ax

    def effect_summary(
        self,
        *,
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        min_effect: float | None = None,
        **kwargs: Any,
    ) -> EffectSummary:
        """
        Generate a decision-ready summary of causal effects for Difference-in-Differences.

        Parameters
        ----------
        direction : {"increase", "decrease", "two-sided"}, default="increase"
            Direction for tail probability calculation (PyMC only, ignored for OLS).
        alpha : float, default=0.05
            Significance level for HDI/CI intervals (1-alpha confidence level).
        min_effect : float, optional
            Region of Practical Equivalence (ROPE) threshold (PyMC only, ignored for OLS).
        **kwargs
            Reserved for forward-compatibility; not consumed by this
            implementation.

        Returns
        -------
        EffectSummary
            Object with .table (DataFrame) and .text (str) attributes
        """
        is_pymc = self._model_backend.is_bayesian

        if is_pymc:
            return _effect_summary_did(
                self,
                direction=direction,
                alpha=alpha,
                min_effect=min_effect,
            )
        else:
            # OLS DiD
            stats = _compute_statistics_did_ols(self, alpha=alpha)
            table = _generate_table_did_ols(stats)
            text = _generate_prose_did_ols(stats, alpha=alpha)
            return EffectSummary(table=table, text=text)
