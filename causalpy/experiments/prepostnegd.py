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

import warnings
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import tidydraws as td
import xarray as xr
from matplotlib import pyplot as plt
from patsy import build_design_matrices, dmatrices
from plotnine import (
    aes,
    facet_wrap,
    geom_density,
    geom_line,
    geom_point,
    geom_ribbon,
    geom_vline,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    theme,
)

from causalpy.constants import HDI_PROB, LEGEND_FONT_SIZE
from causalpy.custom_exceptions import (
    DataException,
)
from causalpy.experiments.model_adapter import build_coords
from causalpy.plot_utils import _PosteriorPlotStyle, plot_posterior_over_x
from causalpy.pymc_models import LinearRegression, PyMCModel
from causalpy.reporting import EffectSummary, _effect_summary_did
from causalpy.utils import _is_variable_dummy_coded, round_num

from .base import BaseExperiment


class PrePostNEGD(BaseExperiment):
    """
    A class to analyse data from pretest/posttest designs.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas dataframe.
    formula : str
        A statistical model formula.
    group_variable_name : str
        Name of the column in ``data`` for the group variable; should be
        either binary or boolean.
    pretreatment_variable_name : str
        Name of the column in ``data`` for the pretreatment variable.
    model : PyMCModel, optional
        A PyMC model. Defaults to :class:`LinearRegression`.
    **kwargs
        Additional keyword arguments forwarded to :class:`BaseExperiment`.

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
    ... )
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
    _deprecated_design_aliases = {"X": ("design", "X"), "y": ("design", "y")}

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        group_variable_name: str,
        pretreatment_variable_name: str,
        model: PyMCModel | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model)
        self.causal_impact: xr.DataArray
        self.pred_xi: np.ndarray
        self.pred_untreated: az.InferenceData
        self.pred_treated: az.InferenceData
        self.data = data
        self.expt_type = "Pretest/posttest Nonequivalent Group Design"
        self.formula = formula
        self.group_variable_name = group_variable_name
        self.pretreatment_variable_name = pretreatment_variable_name
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
        self.design = self._build_design_dataset(
            self._X_raw,
            self._y_raw,
            obs_ind=self.data.index,
            coeffs=self.labels,
        )
        del self._X_raw, self._y_raw

    def algorithm(self) -> None:
        """Run the experiment algorithm: fit model, predict, and calculate causal impact."""
        X = self.design["X"]
        y = self.design["y"]

        if self._model_backend.is_ols:
            raise NotImplementedError("Not implemented for OLS model")
        if not self._model_backend.is_bayesian:
            raise ValueError("Model type not recognized")

        self._model_backend.fit(
            X=X,
            y=y,
            coords=build_coords(self.labels, X.shape[0]),
        )

        assert self.model.idata is not None
        # Calculate the posterior predictive for the treatment and control for an
        # interpolated set of pretest values
        # get the model predictions of the observed data
        self.pred_xi = np.linspace(
            np.min(self.data[self.pretreatment_variable_name]),
            np.max(self.data[self.pretreatment_variable_name]),
            200,
        )
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
        self.pred_untreated = self.model.predict(X=np.asarray(new_x_untreated))
        # treated
        x_pred_treated = pd.DataFrame(
            {
                self.pretreatment_variable_name: self.pred_xi,
                self.group_variable_name: np.ones(self.pred_xi.shape),
            }
        )
        (new_x_treated,) = build_design_matrices([self._x_design_info], x_pred_treated)
        self.pred_treated = self.model.predict(X=np.asarray(new_x_treated))

        # Evaluate causal impact as equal to the treatment effect
        self.causal_impact = self.model.idata.posterior["beta"].sel(
            {"coeffs": self._get_treatment_effect_coeff()}
        )

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
        percentiles = self.causal_impact.quantile(
            [(1 - HDI_PROB) / 2, 1 - (1 - HDI_PROB) / 2]
        ).values
        ci = (
            rf"$CI_{{{HDI_PROB * 100:.0f}\%}}$"
            + f"[{round_num(percentiles[0], round_to)}, {round_num(percentiles[1], round_to)}]"
        )
        causal_impact = f"{round_num(self.causal_impact.mean(), round_to)}, "
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
        round_to: int | None = None,
        ci_prob: float = HDI_PROB,
        hdi_prob: float | None = None,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        figsize: tuple[float, float] = (7, 9),
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> ggplot | tuple[plt.Figure, list[plt.Axes]]:
        """Plot the pre-post non-equivalent group design results.

        Parameters
        ----------
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
        plotnine.ggplot or tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]
            For ``kind="ribbon"`` (default), a two-facet :class:`plotnine.ggplot`
            (top: scatter + posterior predictive bands; bottom: estimated
            treatment effect posterior). Call ``.draw()`` for the matplotlib
            figure. For ``kind="spaghetti"`` or ``"histogram"`` a ``(fig, ax)``
            tuple is returned (legacy matplotlib path).
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
        figsize: tuple[float, float] = (7, 9),
        **kwargs: Any,
    ) -> ggplot | tuple[plt.Figure, list[plt.Axes]]:
        """Generate a plotnine plot for pretest/posttest nonequivalent group designs.

        Returns a two-facet :class:`plotnine.ggplot` for ``kind="ribbon"``: the
        top facet shows the pre/post scatter plus control and treatment
        posterior predictive bands; the bottom facet shows the estimated
        treatment effect posterior as a density with a reference line at zero
        and the credible interval bounds (replacing ``az.plot_posterior``).

        ponytail: ``spaghetti`` and ``histogram`` route to the legacy matplotlib
        method (multi-panel spaghetti/heatmap have no clean plotnine form);
        upgrade path tracked in #988. Both return a ``(fig, ax)`` tuple.
        """
        if kind != "ribbon":
            return self._bayesian_plot_mpl(
                round_to=round_to,
                ci_prob=ci_prob,
                kind=kind,
                ci_kind=ci_kind,
                num_samples=num_samples,
                figsize=figsize,
            )

        top = "Pretest vs posttest"
        bottom = "Estimated treatment effect"
        interval = "eti" if ci_kind == "eti" else "hdi"

        # Top facet: observed data as (_x=pre, _y=post).
        scatter = self.data[["pre", "post", "group"]].rename(
            columns={"pre": "_x", "post": "_y"}
        )
        scatter["group"] = scatter["group"].astype(str)
        scatter["panel"] = top

        # Top facet: posterior predictive bands for each group. tidydraws keeps
        # the pretest grid as a column, dropping the isel(treated_units=0) guard.
        def _band(pred, series):
            newdata = pd.DataFrame({"pre": np.asarray(self.pred_xi)})
            newdata["obs_ind"] = range(len(newdata))
            draws = td.prediction_draws(
                pred, newdata=newdata, var_name="mu", idata_group="posterior_predictive"
            )
            summ = (
                td.point_interval(
                    draws,
                    "mu",
                    group_by="pre",
                    probs=(ci_prob,),
                    point="mean",
                    interval=interval,
                )
                .sort("pre")
                .to_pandas()
                .rename(columns={"pre": "_x", "mu": "_y"})
            )
            summ["series"] = series
            summ["panel"] = top
            return summ

        bands = pd.concat(
            [
                _band(self.pred_untreated, "Control group"),
                _band(self.pred_treated, "Treatment group"),
            ]
        )

        # Bottom facet: treatment effect posterior samples + interval summary.
        effect = np.asarray(self.causal_impact).ravel()
        eff_summary = td.point_interval(
            pl.DataFrame({"effect": effect, "_g": 0}),
            "effect",
            group_by="_g",
            probs=(ci_prob,),
            point="mean",
            interval=interval,
        )
        effect_df = pd.DataFrame({"_x": effect, "panel": bottom})
        refs = pd.DataFrame(
            {
                "_x": [
                    0.0,
                    float(eff_summary["effect_lower"][0]),
                    float(eff_summary["effect_upper"][0]),
                ],
                "panel": bottom,
                "ref": ["zero", "hdi", "hdi"],
            }
        )
        mean_label = (
            f"mean = {round_num(eff_summary['effect'][0], round_to)}\n"
            f"{ci_prob * 100:.0f}% CI [{round_num(eff_summary['effect_lower'][0], round_to)}, "
            f"{round_num(eff_summary['effect_upper'][0], round_to)}]"
        )

        # Order facets so the scatter is on top (plotnine follows factor levels).
        panels = [top, bottom]
        for frame in (scatter, bands, effect_df, refs):
            frame["panel"] = pd.Categorical(
                frame["panel"], categories=panels, ordered=True
            )

        colors = {
            "0": "#1f77b4",
            "1": "#ff7f0e",
            "Control group": "#1f77b4",
            "Treatment group": "#ff7f0e",
        }
        return (
            ggplot()
            + geom_point(scatter, aes("_x", "_y", color="group"), alpha=0.5)
            + geom_ribbon(
                bands,
                aes("_x", ymin="mu_lower", ymax="mu_upper", fill="series"),
                alpha=0.3,
            )
            + geom_line(bands, aes("_x", "_y", color="series"))
            + geom_density(effect_df, aes("_x"))
            + geom_vline(
                refs[refs["ref"] == "zero"], aes(xintercept="_x"), color="grey"
            )
            + geom_vline(
                refs[refs["ref"] == "hdi"],
                aes(xintercept="_x"),
                color="black",
                linetype="dashed",
            )
            + facet_wrap("panel", ncol=1, scales="free")
            + scale_color_manual(values=colors, name="")
            + scale_fill_manual(values=colors, name="")
            + labs(x="", y="", title=mean_label)
            + theme(panel_spacing=0.25)
        )

    def _bayesian_plot_mpl(
        self,
        round_to: int | None = None,
        ci_prob: float = HDI_PROB,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        figsize: tuple[float, float] = (7, 9),
        **kwargs: Any,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Legacy matplotlib plot, retained for the ``spaghetti`` and ``histogram`` kinds.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2. Use ``None``
            to return raw numbers.
        hdi_prob : float, optional
            Probability mass of the highest density interval drawn around the
            posterior predictive bands for the control and treatment groups,
            and around the posterior of the estimated treatment effect.
            Must be in ``(0, 1]``. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
        figsize : tuple of (float, float), optional
            Width and height of the figure in inches. Defaults to ``(7, 9)``.
        """
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
            self.pred_untreated["posterior_predictive"].mu.isel(treated_units=0),
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
            self.pred_treated["posterior_predictive"].mu.isel(treated_units=0),
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

        # Plot estimated caual impact / treatment effect
        az.plot_posterior(
            self.causal_impact,
            ref_val=0,
            ax=ax[1],
            round_to=round_to,
            hdi_prob=ci_prob,
        )
        ax[1].set(title="Estimated treatment effect")
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
        Generate a decision-ready summary of causal effects for PrePostNEGD.

        Parameters
        ----------
        direction : {"increase", "decrease", "two-sided"}, default="increase"
            Direction for tail probability calculation (PyMC only).
        alpha : float, default=0.05
            Significance level for HDI/CI intervals (1-alpha confidence level).
        min_effect : float, optional
            Region of Practical Equivalence (ROPE) threshold (PyMC only).
        **kwargs
            Reserved for forward-compatibility; not consumed by this
            implementation.

        Returns
        -------
        EffectSummary
            Object with .table (DataFrame) and .text (str) attributes
        """
        return _effect_summary_did(
            self,
            direction=direction,
            alpha=alpha,
            min_effect=min_effect,
        )
