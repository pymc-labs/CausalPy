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
"""
Regression discontinuity design
"""

import warnings  # noqa: I001


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from patsy import build_design_matrices, dmatrices
from sklearn.base import RegressorMixin
import xarray as xr
from causalpy.custom_exceptions import (
    DataException,
    FormulaException,
)
from causalpy.plot_utils import plot_xY
from causalpy.pymc_models import LinearRegression, PyMCModel
from causalpy.utils import _is_variable_dummy_coded, convert_to_string, round_num

from .base import BaseExperiment
from causalpy.reporting import EffectSummary, _effect_summary_rd
from typing import Any, Literal

LEGEND_FONT_SIZE = 12


class RegressionDiscontinuity(BaseExperiment):
    """
    A class to analyse sharp regression discontinuity experiments.

    :param data:
        A pandas dataframe
    :param formula:
        A statistical model formula
    :param treatment_threshold:
        A scalar threshold value at which the treatment is applied
    :param model:
        A PyMC or sklearn model. Defaults to LinearRegression.
    :param running_variable_name:
        The name of the predictor variable that the treatment threshold is based upon
    :param epsilon:
        A small scalar value which determines how far above and below the treatment
        threshold to evaluate the causal impact.
    :param bandwidth:
        Data outside of the bandwidth (relative to the discontinuity) is not used to fit
        the model.
    :param donut_hole:
        Observations within this distance from the treatment threshold are excluded from
        model fitting. Used as a robustness check when observations closest to the
        threshold may be problematic (e.g., due to manipulation or heaping). Defaults
        to 0.0 (no exclusion). Must be non-negative and less than bandwidth if bandwidth
        is finite.

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
    _default_model_class = LinearRegression

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
        **kwargs: dict,
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
        self.y, self.X = np.asarray(y), np.asarray(X)
        self.outcome_variable_name = y.design_info.column_names[0]

    def _prepare_data(self) -> None:
        """Convert design matrices to xarray DataArrays."""
        self.X = xr.DataArray(
            self.X,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": np.arange(self.X.shape[0]),
                "coeffs": self.labels,
            },
        )
        self.y = xr.DataArray(
            self.y,
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": np.arange(self.y.shape[0]), "treated_units": ["unit_0"]},
        )

    def algorithm(self) -> None:
        """Run the experiment algorithm: fit model, predict, and calculate discontinuity."""
        # fit model
        if isinstance(self.model, PyMCModel):
            # fit the model to the observed (pre-intervention) data
            COORDS = {
                "coeffs": self.labels,
                "obs_ind": np.arange(self.X.shape[0]),
                "treated_units": ["unit_0"],
            }
            self.model.fit(X=self.X, y=self.y, coords=COORDS)
        elif isinstance(self.model, RegressorMixin):
            self.model.fit(X=self.X, y=self.y)
        else:
            raise ValueError("Model type not recognized")

        # score the goodness of fit to all data
        self.score = self.model.score(X=self.X, y=self.y)

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
        self.pred = self.model.predict(X=np.asarray(new_x))

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
        if isinstance(self.model, PyMCModel):
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
        """Validate the input data and model formula for correctness"""
        if "treated" not in self.formula:
            raise FormulaException(
                "A predictor called `treated` should be in the formula"
            )

        if _is_variable_dummy_coded(self.data["treated"]) is False:
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
        Print summary of main results and model coefficients

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
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

    def _bayesian_plot(
        self, round_to: int | None = 2, **kwargs: dict
    ) -> tuple[plt.Figure, plt.Axes]:
        """Generate plot for regression discontinuity designs."""
        fig, ax = plt.subplots()

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
            plot_hdi_kwargs={"color": "C1"},
            label="Posterior mean",
        )

        # create strings to compose title
        title_info = f"{round_num(self.score['unit_0_r2'], round_to)} (std = {round_num(self.score['unit_0_r2_std'], round_to)})"
        r2 = f"Bayesian $R^2$ on fit data = {title_info}"
        percentiles = self.discontinuity_at_threshold.quantile([0.03, 1 - 0.03]).values
        ci = (
            r"$CI_{94\%}$"
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
        self, round_to: int | None = None, **kwargs: dict
    ) -> tuple[plt.Figure, plt.Axes]:
        """Generate plot for regression discontinuity designs."""
        fig, ax = plt.subplots()

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
        r2 = f"$R^2$ on fit data = {round_num(float(self.score), round_to)}"
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
