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
"""Regression discontinuity design."""

import re  # noqa: I001
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from patsy import ModelDesc, build_design_matrices, dmatrices
from plotnine import (
    aes,
    geom_point,
    geom_vline,
    ggplot,
    labs,
    scale_color_manual,
)
from sklearn.base import RegressorMixin
from causalpy.experiments.model_adapter import build_coords
from causalpy.custom_exceptions import (
    DataException,
    FormulaException,
)
from causalpy.constants import HDI_PROB, LEGEND_FONT_SIZE
from causalpy.plot_utils import (
    HISTOGRAM_PANEL_THEME,
    add_posterior_kind,
    interval_kind,
    posterior_histogram_tiles,
    prediction_draws,
    spaghetti_draws,
    summarize_draws,
)
from causalpy.pymc_models import LinearRegression, PyMCModel
from causalpy.reporting import EffectSummary, _effect_summary_rd
from causalpy.utils import (
    _as_scalar,
    _is_variable_dummy_coded,
    convert_to_string,
    round_num,
)

from .base import BaseExperiment


@dataclass(frozen=True)
class _RDPlotData:
    """Tidy tables consumed by the declarative RD plot."""

    points: pd.DataFrame
    intervals: pd.DataFrame
    posterior_paths: pd.DataFrame | None
    posterior_density: pd.DataFrame | None
    colors: dict[str, str]


class RegressionDiscontinuity(BaseExperiment):
    """
    A class to analyse sharp regression discontinuity experiments.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas dataframe.
    formula : str
        A statistical model formula.
    treatment_threshold : float
        A scalar threshold value at which the treatment is applied.
    model : PyMCModel, RegressorMixin, or None, default None
        A PyMC or sklearn model. Defaults to :class:`LinearRegression`.
    running_variable_name : str, default "x"
        The name of the predictor variable that the treatment threshold is
        based upon.
    epsilon : float, default 0.001
        A small scalar value which determines how far above and below the
        treatment threshold to evaluate the causal impact.
    bandwidth : float, default np.inf
        Data outside of the bandwidth (relative to the discontinuity) is not
        used to fit the model.
    donut_hole : float, default 0.0
        Observations within this distance from the treatment threshold are
        excluded from model fitting. Used as a robustness check when
        observations closest to the threshold may be problematic (e.g., due
        to manipulation or heaping). Must be non-negative and less than
        ``bandwidth`` if ``bandwidth`` is finite.
    **kwargs
        Additional keyword arguments forwarded to :class:`BaseExperiment`.

    Examples
    --------
    >>> import causalpy as cp
    >>> df = cp.load_data("rd")
    >>> seed = 42
    >>> result = cp.RegressionDiscontinuity(
    ...     df,
    ...     formula="y ~ 1 + x + treated + x:treated",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={
    ...             "draws": 100,
    ...             "target_accept": 0.95,
    ...             "random_seed": seed,
    ...             "progressbar": False,
    ...         },
    ...     ),
    ...     treatment_threshold=0.5,
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
        treatment_threshold: float,
        model: PyMCModel | RegressorMixin | None = None,
        running_variable_name: str = "x",
        epsilon: float = 0.001,
        bandwidth: float = np.inf,
        donut_hole: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model)
        self.expt_type = "Regression Discontinuity"
        self.data = data
        self.formula = formula
        self.running_variable_name = running_variable_name
        self.treatment_threshold = treatment_threshold
        self.epsilon = epsilon
        self.bandwidth = bandwidth
        self.donut_hole = donut_hole
        self.input_validation()
        self._build_design_matrices()
        self._prepare_data()
        self.algorithm()

    def _build_design_matrices(self) -> None:
        """Build design matrices from formula and data, applying bandwidth and donut hole filtering."""
        x_vals = self.data[self.running_variable_name]
        c = self.treatment_threshold
        mask = pd.Series(True, index=self.data.index)

        if self.bandwidth is not np.inf:
            mask &= np.abs(x_vals - c) <= self.bandwidth

        if self.donut_hole > 0:
            mask &= np.abs(x_vals - c) >= self.donut_hole

        self.fit_data = self.data.loc[mask]

        if len(self.fit_data) <= 10:
            filter_desc = []
            if self.bandwidth is not np.inf:
                filter_desc.append(f"bandwidth={self.bandwidth}")
            if self.donut_hole > 0:
                filter_desc.append(f"donut_hole={self.donut_hole}")
            if filter_desc:
                msg = (
                    f"Choice of {' and '.join(filter_desc)} parameters has led to only "
                    f"{len(self.fit_data)} remaining datapoints. "
                    f"Consider adjusting these parameters."
                )
            else:
                msg = f"Only {len(self.fit_data)} datapoints in the dataset."
            warnings.warn(msg, UserWarning, stacklevel=2)

        y, X = dmatrices(self.formula, self.fit_data)

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
        """Run the experiment algorithm: fit model, predict, and calculate discontinuity."""
        X = self.design["X"]
        y = self.design["y"]

        self._model_backend.fit(
            X=X,
            y=y,
            coords=build_coords(self.labels, X.shape[0]),
        )

        self.score = self._model_backend.score(X=X, y=y)

        # get the model predictions of the observed data
        if self.bandwidth is not np.inf:
            fmin = self.treatment_threshold - self.bandwidth
            fmax = self.treatment_threshold + self.bandwidth
            xi = np.linspace(fmin, fmax, 200)
        else:
            xi = np.linspace(
                np.min(self.data[self.running_variable_name]),
                np.max(self.data[self.running_variable_name]),
                200,
            )
        self.x_pred = pd.DataFrame(
            {self.running_variable_name: xi, "treated": self._is_treated(xi)}
        )
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred)
        self.pred = self._model_backend.predict(X=np.asarray(new_x))

        # calculate discontinuity by evaluating the difference in model expectation on
        # either side of the discontinuity
        # NOTE: `"treated": np.array([0, 1])`` assumes treatment is applied above
        # (not below) the threshold
        self.x_discon = pd.DataFrame(
            {
                self.running_variable_name: np.array(
                    [
                        self.treatment_threshold - self.epsilon,
                        self.treatment_threshold + self.epsilon,
                    ]
                ),
                "treated": np.array([0, 1]),
            }
        )
        (new_x,) = build_design_matrices([self._x_design_info], self.x_discon)
        self.pred_discon = self.model.predict(X=np.asarray(new_x))

        # ******** THIS IS SUBOPTIMAL AT THE MOMENT ************************************
        if self._model_backend.is_bayesian:
            self.discontinuity_at_threshold = (
                self.pred_discon["posterior_predictive"].sel(obs_ind=1)["mu"]
                - self.pred_discon["posterior_predictive"].sel(obs_ind=0)["mu"]
            )
        else:
            self.discontinuity_at_threshold = np.squeeze(
                self.pred_discon[1]
            ) - np.squeeze(self.pred_discon[0])
        # ******************************************************************************

    def input_validation(self) -> None:
        """Validate the input data and model formula for correctness."""
        if not any(
            re.search(r"\btreated\b", factor.name())
            for term in ModelDesc.from_formula(self.formula).rhs_termlist
            for factor in term.factors
        ):
            raise FormulaException(
                "A predictor called `treated` should be in the formula RHS"
            )

        if "treated" not in self.data.columns:
            raise DataException(
                "A dummy-coded `treated` column should be present in the data"
            )

        if not _is_variable_dummy_coded(self.data["treated"]):
            raise DataException(
                """The treated variable should be dummy coded. Consisting of 0's and 1's only."""  # noqa: E501
            )

        # Validate donut_hole parameter
        if self.donut_hole < 0:
            raise ValueError("donut_hole must be non-negative.")

        if self.bandwidth is not np.inf and self.donut_hole >= self.bandwidth:
            raise ValueError(
                f"donut_hole ({self.donut_hole}) must be less than bandwidth "
                f"({self.bandwidth}) when bandwidth is finite."
            )

        # Convert integer treated variable to boolean if needed
        if self.data["treated"].dtype in ["int64", "int32"]:
            # Make a copy to avoid SettingWithCopyWarning
            self.data = self.data.copy()
            self.data["treated"] = self.data["treated"].astype(bool)

    def _is_treated(self, x: np.ndarray | pd.Series) -> np.ndarray:
        """Returns ``True`` if `x` is greater than or equal to the treatment threshold.

        .. warning::

            Assumes treatment is given to those ABOVE the treatment threshold.
        """
        return np.greater_equal(x, self.treatment_threshold)

    def summary(self, round_to: int | None = None) -> None:
        """
        Print summary of main results and model coefficients.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2. Use
            ``None`` to return raw numbers.
        """
        print("Regression Discontinuity experiment")
        print(f"Formula: {self.formula}")
        print(f"Running variable: {self.running_variable_name}")
        print(f"Threshold on running variable: {self.treatment_threshold}")
        print(f"Bandwidth: {self.bandwidth}")
        print(f"Donut hole: {self.donut_hole}")
        print(f"Observations used for fit: {len(self.fit_data)}")
        print("\nResults:")
        print(
            f"Discontinuity at threshold = {convert_to_string(self.discontinuity_at_threshold)}"
        )
        print("\n")
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
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> ggplot | tuple[plt.Figure, plt.Axes]:
        """Plot the regression discontinuity results.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round numerical results in the figure
            title (e.g. the Bayesian :math:`R^2`). Defaults to 2. Use
            ``None`` to render raw numbers.
        ci_prob : float
            Probability mass of the highest density interval drawn around the
            posterior predictive band, and the central credible interval
            reported in the figure title for the discontinuity at threshold.
            Must be in ``(0, 1]``. Ignored for OLS models. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
        hdi_prob : float, optional
            Deprecated. Use ``ci_prob`` instead.
        kind : {"ribbon", "spaghetti", "histogram"}, optional
            How posterior uncertainty is rendered. Defaults to ``"ribbon"``
            (mean + credible band). ``"spaghetti"`` draws individual posterior
            predictive lines. ``"histogram"`` uses a matplotlib density heatmap
            overlay on the plotnine base.
        ci_kind : {"hdi", "eti"}, optional
            Credible interval type when ``kind="ribbon"``. Defaults to
            ``"hdi"``.
        num_samples : int, optional
            Number of posterior draws when ``kind="spaghetti"``. Defaults
            to 50. Ignored for other kinds.

        figsize : tuple of (float, float), optional
            Unused for the plotnine path; retained for API compatibility.
        show : bool
            Whether to automatically display the plot. Defaults to ``True``.
        legend_kwargs : dict, optional
            Keyword arguments to adjust legend placement and styling.
            Supported keys: ``loc``, ``bbox_to_anchor``, ``fontsize``,
            ``frameon``, ``title`` (``bbox_transform`` is accepted alongside
            ``bbox_to_anchor``). Applied only when the return value is a
            matplotlib ``(fig, ax)`` tuple (e.g. OLS plots).

        Returns
        -------
        plotnine.ggplot or tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            A :class:`plotnine.ggplot` for Bayesian ``ribbon`` / ``spaghetti``
            plots (call ``.draw()`` for the matplotlib figure). OLS plots
            still return a ``(fig, ax)`` tuple.
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
        *,
        ci_prob: float,
        interval: Literal["hdi", "eti"],
        kind: Literal["ribbon", "histogram", "spaghetti"],
        num_samples: int,
    ) -> _RDPlotData:
        """Prepare observed and posterior tables for plotting."""
        xcol = self.running_variable_name
        points = self.data.copy()
        has_exclusion = len(self.fit_data) < len(self.data)
        point_label = "fit data" if has_exclusion else "data"
        points["series"] = (
            np.where(
                points.index.isin(self.fit_data.index), "fit data", "excluded data"
            )
            if has_exclusion
            else "data"
        )
        newdata = self.x_pred.reset_index(drop=True)
        newdata["obs_ind"] = range(len(newdata))
        draws = prediction_draws(self.pred, newdata)
        intervals = summarize_draws(
            draws,
            group_by=xcol,
            ci_prob=ci_prob,
            interval=interval,
        ).assign(series="Posterior mean")
        posterior_paths = (
            spaghetti_draws(
                draws,
                group_by=xcol,
                num_samples=num_samples,
            ).assign(series="Posterior mean")
            if kind == "spaghetti"
            else None
        )
        colors = {point_label: "black", "Posterior mean": "#ff7f0e"}
        if has_exclusion:
            colors["excluded data"] = "lightgray"
        return _RDPlotData(
            points=points,
            intervals=intervals,
            posterior_paths=posterior_paths,
            posterior_density=(
                posterior_histogram_tiles(draws, xcol) if kind == "histogram" else None
            ),
            colors=colors,
        )

    def _bayesian_plot(
        self,
        round_to: int | None = 2,
        ci_prob: float = HDI_PROB,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> ggplot:
        """Build the Bayesian RD plot from tidy declarative layers."""
        xcol = self.running_variable_name
        ycol = self.outcome_variable_name
        plot_data = self._prepare_bayesian_plot_data(
            ci_prob=ci_prob,
            interval=interval_kind(ci_kind),
            kind=kind,
            num_samples=num_samples,
        )
        p = ggplot() + geom_point(
            plot_data.points, aes(xcol, ycol, color="series"), size=1.5
        )
        p = add_posterior_kind(
            p,
            plot_data.intervals,
            kind,
            x=xcol,
            spaghetti_df=plot_data.posterior_paths,
            histogram_tiles=plot_data.posterior_density,
        )

        title_info = f"{round_num(self.score['unit_0_r2'], round_to)} (std = {round_num(self.score['unit_0_r2_std'], round_to)})"
        r2 = f"Bayesian $R^2$ on fit data = {title_info}"
        percentiles = self.discontinuity_at_threshold.quantile(
            [(1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2]
        ).values
        ci = (
            rf"$CI_{{{ci_prob * 100:.0f}\%}}$"
            + f"[{round_num(percentiles[0], round_to)}, {round_num(percentiles[1], round_to)}]"
        )
        discon = f"Discontinuity at threshold = {round_num(self.discontinuity_at_threshold.mean(), round_to)}, "

        # Treatment threshold (and optional donut boundaries) as legend-mapped
        # vertical lines.
        thr_df = pd.DataFrame(
            {
                "xintercept": [self.treatment_threshold],
                "series": ["treatment threshold"],
            }
        )
        plot_data.colors["treatment threshold"] = "red"
        p = p + geom_vline(
            thr_df, aes(xintercept="xintercept", color="series"), size=1.5
        )
        if self.donut_hole > 0:
            donut_df = pd.DataFrame(
                {
                    "xintercept": [
                        self.treatment_threshold - self.donut_hole,
                        self.treatment_threshold + self.donut_hole,
                    ],
                    "series": ["donut boundary", "donut boundary"],
                }
            )
            p = p + geom_vline(
                donut_df,
                aes(xintercept="xintercept", color="series"),
                linetype="dashed",
                size=1,
            )
            plot_data.colors["donut boundary"] = "orange"

        p = (
            p
            + scale_color_manual(values=plot_data.colors, name="")
            + labs(title=r2 + "\n" + discon + ci, x=xcol, y=ycol)
        )
        if kind == "histogram":
            p = p + HISTOGRAM_PANEL_THEME

        return p

    def _ols_plot(
        self,
        round_to: int | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Generate plot for regression discontinuity designs.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results.
        figsize : tuple of (float, float), optional
            Width and height of the figure in inches. Defaults to ``None``
            (use matplotlib's default).
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot data: use two layers only when there are excluded observations
        has_exclusion = len(self.fit_data) < len(self.data)
        if has_exclusion:
            sns.scatterplot(
                self.data,
                x=self.running_variable_name,
                y=self.outcome_variable_name,
                color="lightgray",
                ax=ax,
                label="excluded data",
            )
        sns.scatterplot(
            self.fit_data,
            x=self.running_variable_name,
            y=self.outcome_variable_name,
            color="k",
            ax=ax,
            label="fit data" if has_exclusion else "data",
        )

        # Plot model fit to data
        ax.plot(
            self.x_pred[self.running_variable_name],
            self.pred,
            "k",
            markersize=10,
            label="model fit",
        )

        # create strings to compose title
        r2 = f"$R^2$ on fit data = {round_num(_as_scalar(self.score), round_to)}"
        discon = f"Discontinuity at threshold = {round_num(self.discontinuity_at_threshold, round_to)}"
        ax.set(title=r2 + "\n" + discon)

        # Treatment threshold line
        ax.axvline(
            x=self.treatment_threshold,
            ls="-",
            lw=3,
            color="r",
            label="treatment threshold",
        )

        # Add donut hole boundary lines if donut_hole > 0
        if self.donut_hole > 0:
            ax.axvline(
                x=self.treatment_threshold - self.donut_hole,
                ls="--",
                lw=2,
                color="orange",
                label="donut boundary",
            )
            ax.axvline(
                x=self.treatment_threshold + self.donut_hole,
                ls="--",
                lw=2,
                color="orange",
            )

        ax.legend(fontsize=LEGEND_FONT_SIZE)
        return (fig, ax)

    def effect_summary(
        self,
        *,
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        min_effect: float | None = None,
        **kwargs: Any,
    ) -> EffectSummary:
        """
        Generate a decision-ready summary of causal effects for Regression Discontinuity.

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
        return _effect_summary_rd(
            self,
            direction=direction,
            alpha=alpha,
            min_effect=min_effect,
        )
