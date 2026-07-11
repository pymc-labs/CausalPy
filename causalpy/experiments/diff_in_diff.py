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
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import pyplot as plt
from patsy import build_design_matrices, dmatrices
from sklearn.base import RegressorMixin

from causalpy.constants import HDI_PROB, LEGEND_FONT_SIZE
from causalpy.custom_exceptions import (
    DataException,
    FormulaException,
)
from causalpy.experiments.model_adapter import build_coords
from causalpy.plot_utils import _PosteriorPlotStyle, plot_posterior_over_x
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
    get_interaction_terms,
    round_num,
)

from .base import BaseExperiment


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
            if (
                self.post_treatment_variable_name in label
                and self.group_variable_name in label
            ):
                new_x.iloc[:, i] = 0
        self.y_pred_counterfactual = self._model_backend.predict(np.asarray(new_x))

        # calculate causal impact
        if self._model_backend.is_bayesian:
            assert self.model.idata is not None
            # This is the coefficient on the interaction term
            coeff_names = self.model.idata.posterior.coords["coeffs"].data
            for i, label in enumerate(coeff_names):
                if (
                    self.post_treatment_variable_name in label
                    and self.group_variable_name in label
                ):
                    self.causal_impact = self.model.idata.posterior["beta"].isel(
                        {"coeffs": i}
                    )
        elif self._model_backend.is_ols:
            # This is the coefficient on the interaction term. Match by checking
            # both variable names independently (as the Bayesian branch above
            # does), rather than a single concatenated substring, since patsy
            # may name the interaction term with either variable first (e.g.
            # "post_treatment[T.True]:group" if the formula writes
            # "post_treatment*group" instead of "group*post_treatment").
            coef_map = dict(
                zip(self.labels, self._model_backend.coefficients(), strict=False)
            )
            matched_key = next(
                (
                    k
                    for k in coef_map
                    if self.post_treatment_variable_name in k
                    and self.group_variable_name in k
                ),
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
        # Check if post_treatment_variable_name is in formula
        if self.post_treatment_variable_name not in self.formula:
            raise FormulaException(
                f"Missing required variable '{self.post_treatment_variable_name}' in formula"
            )

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
        # Define interaction indicators
        INTERACTION_INDICATORS = ["*", ":"]

        # Get interaction terms
        interaction_terms = get_interaction_terms(self.formula)

        # Check for interaction terms with more than 2 variables (more than one '*' or ':')
        for term in interaction_terms:
            total_indicators = sum(
                term.count(indicator) for indicator in INTERACTION_INDICATORS
            )
            if (
                total_indicators >= 2
            ):  # 3 or more variables (e.g., a*b*c or a:b:c has 2 symbols)
                raise FormulaException(
                    f"Formula contains interaction term with more than 2 variables: {term}. "
                    "Three-way or higher-order interactions are not supported as they complicate interpretation of the causal effect."
                )

        if len(interaction_terms) > 1:
            raise FormulaException(
                f"Formula contains {len(interaction_terms)} interaction terms: {interaction_terms}. "
                "Multiple interaction terms are not currently supported as they complicate interpretation of the causal effect."
            )

        # A DiD formula must contain exactly one interaction term, and it must be
        # between the group and post-treatment variables: that term is what
        # identifies the causal effect (see `algorithm`, which reads it back off
        # the fitted model). Without this check a formula with no interaction
        # term, or an interaction between unrelated variables, would pass
        # validation and only fail later when `causal_impact` is accessed.
        if len(interaction_terms) == 0 or not (
            self.group_variable_name in interaction_terms[0]
            and self.post_treatment_variable_name in interaction_terms[0]
        ):
            raise FormulaException(
                "Formula must contain exactly one interaction term between the "
                f"group variable '{self.group_variable_name}' and the "
                f"post-treatment variable '{self.post_treatment_variable_name}' "
                f"(e.g. '{self.group_variable_name}*{self.post_treatment_variable_name}'). "
                "This interaction term identifies the difference-in-differences causal effect."
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

    def _bayesian_plot(
        self,
        round_to: int | None = None,
        ci_prob: float = HDI_PROB,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the results.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2. Use ``None``
            to return raw numbers.
        hdi_prob : float, optional
            Probability mass of the highest density interval drawn around the
            posterior predictive bands for the control, treatment, and
            counterfactual trajectories. Must be in ``(0, 1]``. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
        figsize : tuple of (float, float), optional
            Width and height of the figure in inches. Defaults to ``None``
            (use matplotlib's default).
        """
        style: _PosteriorPlotStyle = {
            "ci_prob": ci_prob,
            "kind": kind,
            "ci_kind": ci_kind,
            "num_samples": num_samples,
        }

        def _plot_causal_impact_arrow(results, ax):
            """
            draw a vertical arrow between `y_pred_counterfactual` and
            `y_pred_counterfactual`
            """
            # Calculate y values to plot the arrow between
            y_pred_treatment = (
                results.y_pred_treatment["posterior_predictive"]
                .mu.isel({"obs_ind": 1})
                .mean()
                .data
            )
            y_pred_counterfactual = (
                results.y_pred_counterfactual["posterior_predictive"].mu.mean().data
            )
            y_pred_treatment_scalar = _as_scalar(y_pred_treatment)
            y_pred_counterfactual_scalar = _as_scalar(y_pred_counterfactual)
            # Calculate the x position to plot at
            # Note that we force to be float to avoid a type error using np.ptp with boolean
            # values
            diff = np.ptp(
                np.array(
                    results.x_pred_treatment[results.time_variable_name].values
                ).astype(float)
            )
            x = (
                np.max(results.x_pred_treatment[results.time_variable_name].values)
                + 0.1 * diff
            )
            # Plot the arrow
            ax.annotate(
                "",
                xy=(x, y_pred_counterfactual_scalar),
                xycoords="data",
                xytext=(x, y_pred_treatment_scalar),
                textcoords="data",
                arrowprops={"arrowstyle": "<-", "color": "green", "lw": 3},
            )
            # Plot text annotation next to arrow
            ax.annotate(
                "causal\nimpact",
                xy=(
                    x,
                    np.mean([y_pred_counterfactual_scalar, y_pred_treatment_scalar]),
                ),
                xycoords="data",
                xytext=(5, 0),
                textcoords="offset points",
                color="green",
                va="center",
            )

        fig, ax = plt.subplots(figsize=figsize)

        # Plot raw data
        sns.scatterplot(
            self.data,
            x=self.time_variable_name,
            y=self.outcome_variable_name,
            hue=self.group_variable_name,
            alpha=1,
            legend=False,
            markers=True,
            ax=ax,
        )

        # Plot model fit to control group
        time_points = self.x_pred_control[self.time_variable_name].values
        h_line, h_patch = plot_posterior_over_x(
            time_points,
            self.y_pred_control["posterior_predictive"].mu.isel(treated_units=0),
            ax=ax,
            **style,
            plot_hdi_kwargs={"color": "C0"},
            label="Control group",
        )
        handles = [(h_line, h_patch)]
        labels = ["Control group"]

        # Plot model fit to treatment group
        time_points = self.x_pred_control[self.time_variable_name].values
        h_line, h_patch = plot_posterior_over_x(
            time_points,
            self.y_pred_treatment["posterior_predictive"].mu.isel(treated_units=0),
            ax=ax,
            **style,
            plot_hdi_kwargs={"color": "C1"},
            label="Treatment group",
        )
        handles.append((h_line, h_patch))
        labels.append("Treatment group")

        # Plot counterfactual - post-test for treatment group IF no treatment
        # had occurred.
        time_points = self.x_pred_counterfactual[self.time_variable_name].values
        if len(time_points) == 1:
            y_pred_cf = az.extract(
                self.y_pred_counterfactual,
                group="posterior_predictive",
                var_names="mu",
            )
            # Select single unit data for plotting
            y_pred_cf_single = y_pred_cf.isel(treated_units=0)
            violin_data = (
                y_pred_cf_single.values
                if hasattr(y_pred_cf_single, "values")
                else y_pred_cf_single
            )
            parts = ax.violinplot(
                violin_data.T,
                positions=self.x_pred_counterfactual[self.time_variable_name].values,
                showmeans=False,
                showmedians=False,
                widths=0.2,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("C0")
                pc.set_edgecolor("None")
                pc.set_alpha(0.5)
        else:
            h_line, h_patch = plot_posterior_over_x(
                time_points,
                self.y_pred_counterfactual.posterior_predictive.mu.isel(
                    treated_units=0
                ),
                ax=ax,
                **style,
                plot_hdi_kwargs={"color": "C2"},
                label="Counterfactual",
            )
            handles.append((h_line, h_patch))
            labels.append("Counterfactual")

        # arrow to label the causal impact
        _plot_causal_impact_arrow(self, ax)

        # formatting
        ax.set(
            xticks=self.x_pred_treatment[self.time_variable_name].values,
            title=self._causal_impact_summary_stat(round_to),
        )
        ax.legend(
            handles=(h_tuple for h_tuple in handles),
            labels=labels,
            fontsize=LEGEND_FONT_SIZE,
        )
        return fig, ax

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
