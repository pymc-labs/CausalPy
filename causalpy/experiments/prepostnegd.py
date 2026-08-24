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
"""Pretest/posttest nonequivalent group design."""

from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import pyplot as plt

from causalpy.constants import HDI_PROB, LEGEND_FONT_SIZE
from causalpy.custom_exceptions import (
    DataException,
)
from causalpy.experiments._results import CoefficientResult, GroupComparisonScenario
from causalpy.experiments.model_adapter import build_coords
from causalpy.formula_utils import build_design_matrices, build_formula_matrices
from causalpy.input_data import DataFrameLike, to_pandas
from causalpy.plot_utils import (
    _PosteriorPlotStyle,
    has_posterior_draws,
    plot_posterior_over_x,
    plot_scalar_posterior,
)
from causalpy.pymc_models import LinearRegression, PyMCModel
from causalpy.reporting import EffectSummary, _effect_summary_did
from causalpy.utils import _is_variable_dummy_coded, round_num

from .base import BaseExperiment


class PrePostNEGD(BaseExperiment[CoefficientResult]):
    """
    A class to analyse data from pretest/posttest designs.

    Parameters
    ----------
    data : dataframe-like
        Any eager dataframe Narwhals supports, such as pandas, Polars, or
        PyArrow. Converted to pandas internally.
    formula : str
        A statistical model formula.
    group_variable_name : str
        Name of the column in ``data`` for the group variable; should be
        either binary or boolean.
    pretreatment_variable_name : str
        Name of the column in ``data`` for the pretreatment variable.
    model : PyMCModel, optional
        A PyMC model. Defaults to :class:`LinearRegression`.

    Notes
    -----
    **Estimate extraction**

    The reported ``causal_impact`` is the posterior coefficient on the treatment-group term, conditional on the pretreatment outcome and any other formula covariates. Treated and untreated prediction curves are also computed for visualization, but they do not determine the reported scalar effect. With the current additive identity-link model, the treatment coefficient equals the corresponding conditional prediction contrast.

    Lazy lifecycle: construction only validates inputs and builds design
    matrices — nothing is sampled. Call :meth:`fit` to draw posterior
    samples (and :meth:`sample_prior_predictive` for prior predictive
    checks); read methods such as :meth:`plot`, :meth:`summary`, and
    :meth:`effect_summary` require the corresponding draw group.

    Examples
    --------
    >>> import causalpy as cp
    >>> df = cp.load_data("anova1")
    >>> seed = 42
    >>> result = cp.PrePostNEGD(
    ...     df,
    ...     formula="post ~ 1 + C(group) + pre",
    ...     group_variable_name="group",
    ...     pretreatment_variable_name="pre",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={
    ...             "target_accept": 0.95,
    ...             "random_seed": seed,
    ...             "progressbar": False,
    ...         }
    ...     ),
    ... ).fit()
    >>> result.summary(round_to=1)  # doctest: +SKIP
    ==================Pretest/posttest Nonequivalent Group Design===================
    Formula: post ~ 1 + C(group) + pre
    <BLANKLINE>
    Results:
    Causal impact = 2, $CI_{94%}$[2, 2]
    Model coefficients:
        Intercept      -0.5, 94% HDI [-1, 0.2]
        C(group)[T.1]  2, 94% HDI [2, 2]
        pre            1, 94% HDI [1, 1]
        y_hat_sigma    0.5, 94% HDI [0.5, 0.6]
    """

    supports_ols = False
    supports_bayes = True
    _default_model_class = LinearRegression

    def __init__(
        self,
        data: DataFrameLike,
        formula: str,
        group_variable_name: str,
        pretreatment_variable_name: str,
        model: PyMCModel | None = None,
    ) -> None:
        super().__init__(model=model)
        self.pred_xi: np.ndarray
        self.data = to_pandas(data)
        self.data.index.name = "obs_ind"
        self.expt_type = "Pretest/posttest Nonequivalent Group Design"
        self.formula = formula
        self.group_variable_name = group_variable_name
        self.pretreatment_variable_name = pretreatment_variable_name
        self.input_validation()
        # Interpolated pretest grid for the treated/untreated scenario
        # predictions. Deterministic — derived from the data, never draws.
        self.pred_xi = np.linspace(
            np.min(self.data[self.pretreatment_variable_name]),
            np.max(self.data[self.pretreatment_variable_name]),
            200,
        )
        self._build_design_matrices()
        self._prepare_data()

    def _build_design_matrices(self) -> None:
        """Build design matrices from formula and data using patsy."""
        y, X = build_formula_matrices(self.formula, self.data)
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self._y_raw, self._X_raw = np.asarray(y), np.asarray(X)
        self.outcome_variable_name = y.design_info.column_names[0]

    def _prepare_data(self) -> None:
        """Bundle design matrices into an ``xr.Dataset``."""
        self.design = self._build_design_dataset(
            self._X_raw,
            self._y_raw,
            obs_ind=self.data.index,
            coeffs=self.labels,
        )
        del self._X_raw, self._y_raw

    def _fit_inputs(
        self,
    ) -> tuple[xr.Dataset, xr.Dataset, dict[str, Any]]:
        """Return the design matrices and coordinates for build."""
        X = self.design["X"]
        # Backend-identity checks are justified here: capability validation
        # (trust boundary), not statistical dispatch.
        if self._model_backend.is_ols:
            raise NotImplementedError("Not implemented for OLS model")
        if not self._model_backend.is_bayesian:
            raise ValueError("Model type not recognized")
        return (
            X,
            self.design["y"],
            build_coords(self.labels, X.shape[0]),
        )

    def _finalize(self, group: Literal["prior", "posterior"]) -> None:
        """Compute the group's result bundle from its draws and assign it.

        The body is the historical ``algorithm()`` prediction and contrast
        stage with the draw group threaded through prediction and
        coefficient reads. The two scenario frames are deterministic
        functions of the data and formula; only their predictions carry
        the requested draw group.
        """
        # Calculate the posterior predictive for the treatment and control for an
        # interpolated set of pretest values
        # untreated
        x_pred_untreated = pd.DataFrame(
            {
                self.pretreatment_variable_name: self.pred_xi,
                self.group_variable_name: np.zeros(self.pred_xi.shape),
            }
        )
        (new_x_untreated,) = build_design_matrices(
            [self._x_design_info], x_pred_untreated
        )
        pred_untreated = self._model_backend.predict(
            X=np.asarray(new_x_untreated), group=group
        )
        # treated
        x_pred_treated = pd.DataFrame(
            {
                self.pretreatment_variable_name: self.pred_xi,
                self.group_variable_name: np.ones(self.pred_xi.shape),
            }
        )
        (new_x_treated,) = build_design_matrices([self._x_design_info], x_pred_treated)
        pred_treated = self._model_backend.predict(
            X=np.asarray(new_x_treated), group=group
        )

        # Evaluate causal impact as equal to the treatment effect
        causal_impact = self._model_backend.coefficients(group=group).sel(
            coeffs=self._get_treatment_effect_coeff()
        )

        bundle = CoefficientResult(
            causal_impact=causal_impact,
            scenario_control=GroupComparisonScenario(
                inputs=x_pred_untreated, prediction=pred_untreated
            ),
            scenario_treated=GroupComparisonScenario(
                inputs=x_pred_treated, prediction=pred_treated
            ),
            score=None,
        )
        self._assign_bundle(group, bundle)

    def input_validation(self) -> None:
        """Validate the input data and model formula for correctness."""
        if not _is_variable_dummy_coded(self.data[self.group_variable_name]):
            raise DataException(
                f"""
                There must be 2 levels of the grouping variable
                {self.group_variable_name}. I.e. the treated and untreated.
                """
            )

    def _get_treatment_effect_coeff(self) -> str:
        """Find the beta regression coefficient corresponding to the
        group (i.e. treatment) effect.
        For example if self.group_variable_name is 'group' and
        the labels are `['Intercept', 'C(group)[T.1]', 'pre']`
        then we want `C(group)[T.1]`.
        """
        for label in self.labels:
            if (self.group_variable_name in label) & (":" not in label):
                return label

        raise NameError("Unable to find coefficient name for the treatment effect")

    def _causal_impact_summary_stat(self, round_to: int | None = 2) -> str:
        """Computes the mean and credible interval bounds for the causal impact."""
        causal_impact = self.result.causal_impact
        percentiles = causal_impact.quantile(
            [(1 - HDI_PROB) / 2, 1 - (1 - HDI_PROB) / 2]
        ).values
        ci = (
            rf"$CI_{{{HDI_PROB * 100:.0f}\%}}$"
            + f"[{round_num(percentiles[0], round_to)}, {round_num(percentiles[1], round_to)}]"
        )
        causal_impact = f"{round_num(causal_impact.mean(), round_to)}, "
        return f"Causal impact = {causal_impact + ci}"

    def summary(self, round_to: int | None = None) -> None:
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

    def plot(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        round_to: int | None = None,
        ci_prob: float = HDI_PROB,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        figsize: tuple[float, float] = (7, 9),
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot the pre-post non-equivalent group design results.

        Parameters
        ----------
        group : {"prior", "posterior"}, default "posterior"
            Which draw group to plot. ``"prior"`` renders the reduced
            prior-check panel set — prior-implied treated and untreated
            prediction curves against the observed data only — and requires
            :meth:`sample_prior_predictive`; ``"posterior"`` (default)
            renders the full two-panel layout with the estimated treatment
            effect posterior and requires :meth:`fit`.
            The two groups intentionally return different axes layouts.
        round_to : int, optional
            Number of decimals used to round numerical results in the figure.
            Defaults to ``None``, in which case 2 significant figures are
            used.
        ci_prob : float
            Probability mass of the highest density interval drawn around the
            posterior predictive bands for the control and treatment groups,
            and around the posterior of the estimated treatment effect.
            Must be in ``(0, 1]``. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
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
            :func:`matplotlib.pyplot.subplots`. Defaults to ``(7, 9)``.
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
            The two axes (top: scatter and posterior predictive bands,
            bottom: estimated treatment effect posterior).
        """
        return self._render_plot(
            show=show,
            legend_kwargs=legend_kwargs,
            group=group,
            round_to=round_to,
            ci_prob=ci_prob,
            kind=kind,
            ci_kind=ci_kind,
            num_samples=num_samples,
            figsize=figsize,
        )

    def _plot(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        round_to: int | None = None,
        ci_prob: float = HDI_PROB,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        figsize: tuple[float, float] = (7, 9),
        **kwargs: Any,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Generate plot for ANOVA-like experiments with non-equivalent group designs.

        Consumes the resolved group bundle injected by
        :meth:`~causalpy.experiments.base.BaseExperiment._render_plot`.

        Parameters
        ----------
        group : {"prior", "posterior"}
            ``"prior"`` renders the reduced single-panel prior-check figure
            via :meth:`_plot_prior_checks`; ``"posterior"`` renders the full
            two-panel layout.
        round_to : int, optional
            Number of decimals used to round results. Defaults to ``None``. Use
            ``None`` to return raw numbers.
        ci_prob : float, optional
            Probability mass of the highest density interval drawn around the
            posterior predictive bands for the control and treatment groups,
            and around the posterior of the estimated treatment effect.
            Must be in ``(0, 1]``. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
        figsize : tuple of (float, float), optional
            Width and height of the figure in inches. Defaults to ``(7, 9)``.
        """
        bundle = self._require_bundle(group)
        if group == "prior":
            return self._plot_prior_checks(bundle=bundle)

        pred_untreated = bundle.scenario_control.prediction
        pred_treated = bundle.scenario_treated.prediction

        style: _PosteriorPlotStyle = {
            "ci_prob": ci_prob,
            "kind": kind,
            "ci_kind": ci_kind,
            "num_samples": num_samples,
        }
        fig, ax = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
        )

        # Plot raw data
        sns.scatterplot(
            x="pre",
            y="post",
            hue="group",
            alpha=0.5,
            data=self.data,
            legend=True,
            ax=ax[0],
        )
        ax[0].set(xlabel="Pretest", ylabel="Posttest")

        # plot posterior predictive of untreated
        h_line, h_patch = plot_posterior_over_x(
            self.pred_xi,
            pred_untreated.isel(treated_units=0),
            ax=ax[0],
            **style,
            plot_hdi_kwargs={"color": "C0"},
            label="Control group",
        )
        handles = [(h_line, h_patch)]
        labels = ["Control group"]

        # plot posterior predictive of treated
        h_line, h_patch = plot_posterior_over_x(
            self.pred_xi,
            pred_treated.isel(treated_units=0),
            ax=ax[0],
            **style,
            plot_hdi_kwargs={"color": "C1"},
            label="Treatment group",
        )
        handles.append((h_line, h_patch))
        labels.append("Treatment group")

        ax[0].legend(
            handles=(h_tuple for h_tuple in handles),
            labels=labels,
            fontsize=LEGEND_FONT_SIZE,
        )

        # Plot estimated causal impact / treatment effect
        plot_scalar_posterior(
            bundle.causal_impact,
            ax=ax[1],
            ci_prob=ci_prob,
            ref_val=0,
            round_to=round_to,
        )
        ax[1].set(title="Estimated treatment effect")
        return fig, ax

    def _plot_prior_checks(
        self, *, bundle: CoefficientResult
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Render the reduced prior-check panel set.

        Prior-implied bands are typically far wider than the data, so the
        treatment-effect posterior panel is dropped rather than autoscaled
        into uselessness. The question a prior check answers is whether the
        prior-implied treated/untreated curves are plausible against the
        observed data — one panel suffices.
        """
        style: _PosteriorPlotStyle = {
            "ci_prob": HDI_PROB,
            "kind": "ribbon",
            "ci_kind": "hdi",
            "num_samples": 50,
        }

        fig, ax = plt.subplots(figsize=(7, 4))

        # Plot raw data
        sns.scatterplot(
            x="pre",
            y="post",
            hue="group",
            alpha=0.5,
            data=self.data,
            legend=True,
            ax=ax,
        )
        ax.set(xlabel="Pretest", ylabel="Posttest")

        # plot prior predictive of untreated
        h_line, h_patch = plot_posterior_over_x(
            self.pred_xi,
            bundle.scenario_control.prediction.isel(treated_units=0),
            ax=ax,
            **style,
            plot_hdi_kwargs={"color": "C0"},
            label="Control group",
        )
        handles = [(h_line, h_patch)]
        labels = ["Control group"]

        # plot prior predictive of treated
        h_line, h_patch = plot_posterior_over_x(
            self.pred_xi,
            bundle.scenario_treated.prediction.isel(treated_units=0),
            ax=ax,
            **style,
            plot_hdi_kwargs={"color": "C1"},
            label="Treatment group",
        )
        handles.append((h_line, h_patch))
        labels.append("Treatment group")

        ax.legend(
            handles=(h_tuple for h_tuple in handles),
            labels=labels,
            fontsize=LEGEND_FONT_SIZE,
        )
        ax.set(title="Prior predictive check")
        return fig, [ax]

    def effect_summary(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        min_effect: float | None = None,
    ) -> EffectSummary:
        """
        Generate a decision-ready summary of causal effects for PrePostNEGD.

        Parameters
        ----------
        group : {"prior", "posterior"}, default "posterior"
            Which draw group to summarize. ``"prior"`` requires
            :meth:`sample_prior_predictive` and produces prior-appropriate
            prose — under a neutral prior, ``P(effect > 0)`` should sit near
            0.5, so a tail probability far from 0.5 flags a design-matrix or
            prior-specification problem rather than a causal finding.
            ``"posterior"`` requires :meth:`fit`.
        direction : {"increase", "decrease", "two-sided"}, default="increase"
            Direction for tail probability calculation (PyMC only).
        alpha : float, default=0.05
            Significance level for HDI/CI intervals (1-alpha confidence level).
        min_effect : float, optional
            Region of Practical Equivalence (ROPE) threshold (PyMC only).

        Returns
        -------
        EffectSummary
            Object with .table (DataFrame) and .text (str) attributes
        """
        bundle = self._require_bundle(group)
        if not has_posterior_draws(bundle.scenario_control.prediction):
            # Unreachable via the constructor (supports_ols is False), but an
            # OLS backend must never reach the draw-based helper below.
            raise NotImplementedError("Not implemented for OLS model")
        return _effect_summary_did(
            bundle,
            direction=direction,
            alpha=alpha,
            min_effect=min_effect,
            group=group,
        )
