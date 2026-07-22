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

"""Regression kink design."""

import re  # noqa: I001
import warnings
from dataclasses import dataclass


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import polars as pl
from patsy import ModelDesc, build_design_matrices
import xarray as xr
from causalpy.formula_utils import build_formula_matrices
from causalpy.experiments.model_adapter import build_coords
from causalpy.plot_utils import (
    HISTOGRAM_PANEL_THEME,
    PlotSpec,
    dataarray_draws,
    label_draws,
    posterior_kind_layers,
)

from causalpy.pymc_models import LinearRegression, PyMCModel
from causalpy.reporting import EffectSummary, _effect_summary_rkink

from causalpy.constants import HDI_PROB

from .base import BaseExperiment
from typing import Any, Literal
from causalpy.utils import _is_variable_dummy_coded, round_num
from causalpy.custom_exceptions import (
    DataException,
    FormulaException,
)


@dataclass(frozen=True)
class _RKPlotData:
    """Tidy tables consumed by the declarative RK plot."""

    points: pd.DataFrame
    draws: pl.DataFrame


class RegressionKink(BaseExperiment):
    """A class to analyse regression kink designs.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas dataframe.
    formula : str
        A statistical model formula.
    kink_point : float
        A scalar value at which the kink occurs.
    model : PyMCModel, optional
        A PyMC model. Defaults to :class:`LinearRegression`.
    running_variable_name : str, default "x"
        The name of the running variable column.
    epsilon : float, default 0.001
        A small scalar for evaluating the causal impact above/below the kink.
    bandwidth : float, default np.inf
        Data outside of the bandwidth (relative to the kink) is not used to
        fit the model.
    **kwargs
        Additional keyword arguments forwarded to :class:`BaseExperiment`.
    """

    supports_ols = False
    supports_bayes = True
    _default_model_class = LinearRegression
    _deprecated_design_aliases = {"X": ("design", "X"), "y": ("design", "y")}

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        kink_point: float,
        model: PyMCModel | None = None,
        running_variable_name: str = "x",
        epsilon: float = 0.001,
        bandwidth: float = np.inf,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model)
        self.expt_type = "Regression Kink"
        self.data = data
        self.formula = formula
        self.running_variable_name = running_variable_name
        self.kink_point = kink_point
        self.epsilon = epsilon
        self.bandwidth = bandwidth
        self.input_validation()
        self._build_design_matrices()
        self._prepare_data()
        self.algorithm()

    def _build_design_matrices(self) -> None:
        """Build design matrices from formula and data, applying bandwidth filtering."""
        if self.bandwidth is not np.inf:
            fmin = self.kink_point - self.bandwidth
            fmax = self.kink_point + self.bandwidth
            filtered_data = self.data.query(f"{fmin} <= x <= {fmax}")
            if len(filtered_data) <= 10:
                warnings.warn(
                    f"Choice of bandwidth parameter has lead to only {len(filtered_data)} remaining datapoints. Consider increasing the bandwidth parameter.",  # noqa: E501
                    UserWarning,
                    stacklevel=2,
                )
            y, X = build_formula_matrices(self.formula, filtered_data)
        else:
            y, X = build_formula_matrices(self.formula, self.data)

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
        """Run the experiment algorithm: fit model, predict, and evaluate gradient change."""
        X = self.design["X"]
        y = self.design["y"]

        COORDS = build_coords(self.labels, X.shape[0])
        self._model_backend.fit(X=X, y=y, coords=COORDS)

        self.score = self._model_backend.score(X=X, y=y)

        # get the model predictions of the observed data
        if self.bandwidth is not np.inf:
            fmin = self.kink_point - self.bandwidth
            fmax = self.kink_point + self.bandwidth
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

        # evaluate gradient change around kink point
        mu_kink_left, mu_kink, mu_kink_right = self._probe_kink_point()
        self.gradient_change = self._eval_gradient_change(
            mu_kink_left, mu_kink, mu_kink_right, self.epsilon
        )

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

        if self.bandwidth <= 0:
            raise ValueError("The bandwidth must be greater than zero.")

        if self.epsilon <= 0:
            raise ValueError("Epsilon must be greater than zero.")

    @staticmethod
    def _eval_gradient_change(
        mu_kink_left: xr.DataArray,
        mu_kink: xr.DataArray,
        mu_kink_right: xr.DataArray,
        epsilon: float,
    ) -> xr.DataArray:
        """Evaluate the gradient change at the kink point.
        It works by evaluating the model below the kink point, at the kink point,
        and above the kink point.
        This is a static method for ease of testing.
        """
        gradient_left = (mu_kink - mu_kink_left) / epsilon
        gradient_right = (mu_kink_right - mu_kink) / epsilon
        gradient_change = gradient_right - gradient_left
        return gradient_change

    def _probe_kink_point(self) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Probe the kink point to evaluate the predicted outcome at the kink point and
        either side."""
        # Create a dataframe to evaluate predicted outcome at the kink point and either
        # side
        x_predict = pd.DataFrame(
            {
                self.running_variable_name: np.array(
                    [
                        self.kink_point - self.epsilon,
                        self.kink_point,
                        self.kink_point + self.epsilon,
                    ]
                ),
                "treated": np.array([0, 1, 1]),
            }
        )
        (new_x,) = build_design_matrices([self._x_design_info], x_predict)
        predicted = self._model_backend.predict(X=np.asarray(new_x))
        mu_kink_left = predicted.sel(obs_ind=0)
        mu_kink = predicted.sel(obs_ind=1)
        mu_kink_right = predicted.sel(obs_ind=2)
        return mu_kink_left, mu_kink, mu_kink_right

    def _is_treated(self, x: np.ndarray | pd.Series) -> np.ndarray:
        """Returns ``True`` if `x` is greater than or equal to the treatment threshold."""  # noqa: E501
        return np.greater_equal(x, self.kink_point)

    def summary(self, round_to: int | None = 2) -> None:
        """Print summary of main results and model coefficients.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2. Use
            ``None`` to return raw numbers.
        """
        print(
            f"""
        {self.expt_type:=^80}
        Formula: {self.formula}
        Running variable: {self.running_variable_name}
        Kink point on running variable: {self.kink_point}

        Results:
        Change in slope at kink point = {round_num(self.gradient_change.mean(), round_to)}
        """
        )
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
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the regression kink results.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round numerical results in the figure
            title (e.g. the Bayesian :math:`R^2`). Defaults to 2. Use
            ``None`` to render raw numbers.
        ci_prob : float
            Probability mass of the highest density interval drawn around the
            posterior predictive band, and the central credible interval
            reported in the figure title for the change in gradient at the
            kink point. Must be in ``(0, 1]``. Defaults to
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

        figsize : tuple of (float, float), optional
            Width and height of the figure in inches, passed to
            :func:`matplotlib.pyplot.subplots`. Defaults to ``None`` (use
            matplotlib's default).
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

    def _prepare_plot_data(self) -> _RKPlotData:
        """Prepare observed and posterior tables for plotting."""
        points = self.data.copy()
        points["series"] = "data"
        newdata = self.x_pred.reset_index(drop=True)
        newdata["obs_ind"] = range(len(newdata))
        draws = label_draws(
            dataarray_draws(self.pred).join(pl.from_pandas(newdata), on="obs_ind"),
            series="Posterior mean",
        )
        return _RKPlotData(
            points=points,
            draws=draws,
        )

    def _plot(
        self,
        round_to: int | None = 2,
        ci_prob: float = HDI_PROB,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> PlotSpec:
        """Build the RK plot from tidy declarative layers."""
        xcol = self.running_variable_name
        ycol = self.outcome_variable_name
        round_digits = round_to if round_to is not None else 2
        color_values = {
            "data": "black",
            "Posterior mean": "#ff7f0e",
            "treatment threshold": "red",
        }
        plot_data = self._prepare_plot_data()
        _, posterior_layers = posterior_kind_layers(
            plot_data.draws,
            kind,
            x=xcol,
            group_by=["series", xcol],
            ci_prob=ci_prob,
            interval=ci_kind,
            num_samples=num_samples,
            colors=color_values,
        )

        p = p9.ggplot() + p9.geom_point(
            plot_data.points, p9.aes(xcol, ycol, color="series"), size=1.5
        )
        for layer in posterior_layers:
            p += layer

        title_info = (
            f"{round_num(self.score['unit_0_r2'], round_digits)} "
            f"(std = {round_num(self.score['unit_0_r2_std'], round_digits)})"
        )
        r2 = f"Bayesian $R^2$ on all data = {title_info}"
        percentiles = self.gradient_change.quantile(
            [(1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2]
        ).values
        ci = (
            rf"$CI_{{{ci_prob * 100:.0f}\%}}$"
            + f"[{round_num(percentiles[0], round_digits)}, "
            f"{round_num(percentiles[1], round_digits)}]"
        )
        grad_change = f"Change in gradient = {round_num(self.gradient_change.mean(), round_digits)}, "

        thr_df = pd.DataFrame(
            {"xintercept": [self.kink_point], "series": ["treatment threshold"]}
        )
        p = p + p9.geom_vline(
            thr_df, p9.aes(xintercept="xintercept", color="series"), size=1.5
        )

        p = (
            p
            + p9.scale_color_manual(values=color_values, name="")
            + p9.labs(title=r2 + "\n" + grad_change + ci, x=xcol, y=ycol)
        )
        if figsize is not None:
            p += p9.theme(figure_size=figsize)
        if kind == "histogram":
            p = p + HISTOGRAM_PANEL_THEME

        return PlotSpec(p, n_panels=1)

    def effect_summary(
        self,
        *,
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        min_effect: float | None = None,
        **kwargs: Any,
    ) -> EffectSummary:
        """
        Generate a decision-ready summary of causal effects for Regression Kink.

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
        return _effect_summary_rkink(
            self,
            direction=direction,
            alpha=alpha,
            min_effect=min_effect,
        )
