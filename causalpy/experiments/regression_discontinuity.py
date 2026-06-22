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

import warnings  # noqa: I001
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from patsy import build_design_matrices, dmatrices
from sklearn.base import RegressorMixin
from causalpy.experiments.model_adapter import build_coords
from causalpy.custom_exceptions import (
    DataException,
    FormulaException,
)
from causalpy.constants import HDI_PROB, LEGEND_FONT_SIZE
from causalpy.plot_utils import plot_xY
from causalpy.pymc_models import LinearRegression, PyMCModel
from causalpy.reporting import EffectSummary, _effect_summary_rd
from causalpy.utils import (
    _as_scalar,
    _is_variable_dummy_coded,
    convert_to_string,
    round_num,
)

from .base import BaseExperiment


@dataclass
class _Config:
    """Container for regression discontinuity configuration parameters."""

    formula: str
    running_variable_name: str
    treatment_threshold: float
    epsilon: float
    bandwidth: float
    donut_hole: float


@dataclass
class _DesignMatrices:
    """Container for design matrix data and metadata."""

    fit_data: pd.DataFrame
    y_design_info: Any
    x_design_info: Any
    labels: list[str]
    outcome_variable_name: str


@dataclass
class _Results:
    """Container for algorithm results."""

    score: Any
    x_pred: pd.DataFrame
    pred: Any
    x_discon: pd.DataFrame
    pred_discon: Any
    discontinuity_at_threshold: Any


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

    Example
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
    expt_type = "Regression Discontinuity"
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
        self.data = data
        self.config = _Config(
            formula=formula,
            running_variable_name=running_variable_name,
            treatment_threshold=treatment_threshold,
            epsilon=epsilon,
            bandwidth=bandwidth,
            donut_hole=donut_hole,
        )
        self.input_validation()
        y_raw, X_raw = self._build_design_matrices()
        self._prepare_data(y_raw, X_raw)
        self.algorithm()

    @property
    def treatment_threshold(self) -> float:
        """Backward-compatible access to config.treatment_threshold."""
        return self.config.treatment_threshold

    @property
    def running_variable_name(self) -> str:
        """Backward-compatible access to config.running_variable_name."""
        return self.config.running_variable_name

    @property
    def formula(self) -> str:
        """Backward-compatible access to config.formula."""
        return self.config.formula

    @property
    def epsilon(self) -> float:
        """Backward-compatible access to config.epsilon."""
        return self.config.epsilon

    @property
    def bandwidth(self) -> float:
        """Backward-compatible access to config.bandwidth."""
        return self.config.bandwidth

    @property
    def donut_hole(self) -> float:
        """Backward-compatible access to config.donut_hole."""
        return self.config.donut_hole

    @property
    def fit_data(self) -> pd.DataFrame:
        """Backward-compatible access to design_matrices.fit_data."""
        return self.design_matrices.fit_data

    @property
    def labels(self) -> list[str]:
        """Backward-compatible access to design_matrices.labels."""
        return self.design_matrices.labels

    @labels.setter
    def labels(self, value: list[str]) -> None:
        self.design_matrices.labels = value

    @property
    def outcome_variable_name(self) -> str:
        """Backward-compatible access to design_matrices.outcome_variable_name."""
        return self.design_matrices.outcome_variable_name

    @property
    def score(self) -> Any:
        """Backward-compatible access to results.score."""
        return self.results.score

    @property
    def x_pred(self) -> pd.DataFrame:
        """Backward-compatible access to results.x_pred."""
        return self.results.x_pred

    @property
    def pred(self) -> Any:
        """Backward-compatible access to results.pred."""
        return self.results.pred

    @property
    def x_discon(self) -> pd.DataFrame:
        """Backward-compatible access to results.x_discon."""
        return self.results.x_discon

    @property
    def pred_discon(self) -> Any:
        """Backward-compatible access to results.pred_discon."""
        return self.results.pred_discon

    @property
    def discontinuity_at_threshold(self) -> Any:
        """Backward-compatible access to results.discontinuity_at_threshold."""
        return self.results.discontinuity_at_threshold

    def _build_design_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Build design matrices from formula and data, applying bandwidth and donut hole filtering."""
        x_vals = self.data[self.config.running_variable_name]
        c = self.config.treatment_threshold
        mask = pd.Series(True, index=self.data.index)

        if self.config.bandwidth is not np.inf:
            mask &= np.abs(x_vals - c) <= self.config.bandwidth

        if self.config.donut_hole > 0:
            mask &= np.abs(x_vals - c) >= self.config.donut_hole

        fit_data = self.data.loc[mask]

        if len(fit_data) <= 10:
            filter_desc = []
            if self.config.bandwidth is not np.inf:
                filter_desc.append(f"bandwidth={self.config.bandwidth}")
            if self.config.donut_hole > 0:
                filter_desc.append(f"donut_hole={self.config.donut_hole}")
            if filter_desc:
                msg = (
                    f"Choice of {' and '.join(filter_desc)} parameters has led to only "
                    f"{len(fit_data)} remaining datapoints. "
                    f"Consider adjusting these parameters."
                )
            else:
                msg = f"Only {len(fit_data)} datapoints in the dataset."
            warnings.warn(msg, UserWarning, stacklevel=2)

        y, X = dmatrices(self.config.formula, fit_data)

        self.design_matrices = _DesignMatrices(
            fit_data=fit_data,
            y_design_info=y.design_info,
            x_design_info=X.design_info,
            labels=X.design_info.column_names,
            outcome_variable_name=y.design_info.column_names[0],
        )
        return np.asarray(y), np.asarray(X)

    def _prepare_data(self, y_raw: np.ndarray, X_raw: np.ndarray) -> None:
        """Bundle design matrices into an ``xr.Dataset``."""
        n = X_raw.shape[0]
        self.design = self._build_design_dataset(
            X_raw,
            y_raw,
            obs_ind=np.arange(n),
            coeffs=self.design_matrices.labels,
        )

    def algorithm(self) -> None:
        """Run the experiment algorithm: fit model, predict, and calculate discontinuity."""
        X = self.design["X"]
        y = self.design["y"]

        self._model_backend.fit(
            X=X,
            y=y,
            coords=build_coords(self.design_matrices.labels, X.shape[0]),
        )

        score = self._model_backend.score(X=X, y=y)

        # get the model predictions of the observed data
        if self.config.bandwidth is not np.inf:
            fmin = self.config.treatment_threshold - self.config.bandwidth
            fmax = self.config.treatment_threshold + self.config.bandwidth
            xi = np.linspace(fmin, fmax, 200)
        else:
            xi = np.linspace(
                np.min(self.data[self.config.running_variable_name]),
                np.max(self.data[self.config.running_variable_name]),
                200,
            )
        x_pred = pd.DataFrame(
            {self.config.running_variable_name: xi, "treated": self._is_treated(xi)}
        )
        (new_x,) = build_design_matrices([self.design_matrices.x_design_info], x_pred)
        pred = self._model_backend.predict(X=np.asarray(new_x))

        # calculate discontinuity by evaluating the difference in model expectation on
        # either side of the discontinuity
        # NOTE: `"treated": np.array([0, 1])`` assumes treatment is applied above
        # (not below) the threshold
        x_discon = pd.DataFrame(
            {
                self.config.running_variable_name: np.array(
                    [
                        self.config.treatment_threshold - self.config.epsilon,
                        self.config.treatment_threshold + self.config.epsilon,
                    ]
                ),
                "treated": np.array([0, 1]),
            }
        )
        (new_x,) = build_design_matrices([self.design_matrices.x_design_info], x_discon)
        pred_discon = self.model.predict(X=np.asarray(new_x))

        # ******** THIS IS SUBOPTIMAL AT THE MOMENT ************************************
        if self._model_backend.is_bayesian:
            discontinuity_at_threshold = (
                pred_discon["posterior_predictive"].sel(obs_ind=1)["mu"]
                - pred_discon["posterior_predictive"].sel(obs_ind=0)["mu"]
            )
        else:
            discontinuity_at_threshold = np.squeeze(
                pred_discon[1]
            ) - np.squeeze(pred_discon[0])
        # ******************************************************************************

        self.results = _Results(
            score=score,
            x_pred=x_pred,
            pred=pred,
            x_discon=x_discon,
            pred_discon=pred_discon,
            discontinuity_at_threshold=discontinuity_at_threshold,
        )

    def input_validation(self) -> None:
        """Validate the input data and model formula for correctness."""
        if "treated" not in self.formula:
            raise FormulaException(
                "A predictor called `treated` should be in the formula"
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
        hdi_prob: float = HDI_PROB,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the regression discontinuity results.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round numerical results in the figure
            title (e.g. the Bayesian :math:`R^2`). Defaults to 2. Use
            ``None`` to render raw numbers.
        hdi_prob : float
            Probability mass of the highest density interval drawn around the
            posterior predictive band, and the central credible interval
            reported in the figure title for the discontinuity at threshold.
            Must be in ``(0, 1]``. Ignored for OLS models. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
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
        return self._render_plot(
            show=show,
            legend_kwargs=legend_kwargs,
            round_to=round_to,
            hdi_prob=hdi_prob,
            figsize=figsize,
        )

    def _bayesian_plot(
        self,
        round_to: int | None = 2,
        hdi_prob: float = HDI_PROB,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Generate plot for regression discontinuity designs.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2. Use ``None``
            to return raw numbers.
        hdi_prob : float, optional
            Probability mass of the highest density interval drawn around the
            posterior predictive band, and the central credible interval
            reported in the figure title for the discontinuity at threshold.
            Must be in ``(0, 1]``. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
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
        plot_xY(
            self.x_pred[self.running_variable_name],
            self.pred["posterior_predictive"].mu.isel(treated_units=0),
            ax=ax,
            hdi_prob=hdi_prob,
            plot_hdi_kwargs={"color": "C1"},
            label="Posterior mean",
        )

        # create strings to compose title
        title_info = f"{round_num(self.score['unit_0_r2'], round_to)} (std = {round_num(self.score['unit_0_r2_std'], round_to)})"
        r2 = f"Bayesian $R^2$ on fit data = {title_info}"
        percentiles = self.discontinuity_at_threshold.quantile(
            [(1 - hdi_prob) / 2, 1 - (1 - hdi_prob) / 2]
        ).values
        ci = (
            rf"$CI_{{{hdi_prob * 100:.0f}\%}}$"
            + f"[{round_num(percentiles[0], round_to)}, {round_num(percentiles[1], round_to)}]"
        )
        discon = f"""
            Discontinuity at threshold = {round_num(self.discontinuity_at_threshold.mean(), round_to)},
            """
        ax.set(title=r2 + "\n" + discon + ci)

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
