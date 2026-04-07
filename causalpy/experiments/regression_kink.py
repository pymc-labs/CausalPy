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
Regression kink design
"""

import warnings  # noqa: I001


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from patsy import build_design_matrices, dmatrices
import xarray as xr
from causalpy.plot_utils import plot_xY

from causalpy.pymc_models import LinearRegression, PyMCModel
from causalpy.reporting import EffectSummary, _effect_summary_rkink

from .base import BaseExperiment
from typing import Any, Literal
from causalpy.utils import round_num
from causalpy.custom_exceptions import (
    DataException,
    FormulaException,
)
from causalpy.utils import _is_variable_dummy_coded


LEGEND_FONT_SIZE = 12


class RegressionKink(BaseExperiment):
    """A class to analyse regression kink designs.

    :param data:
        A pandas dataframe
    :param formula:
        A statistical model formula
    :param kink_point:
        A scalar value at which the kink occurs
    :param model:
        A PyMC model. Defaults to LinearRegression.
    :param running_variable_name:
        The name of the running variable column
    :param epsilon:
        A small scalar for evaluating the causal impact above/below the kink
    :param bandwidth:
        Data outside of the bandwidth (relative to the kink) is not used to fit
        the model.
    """

    supports_ols = False
    supports_bayes = True
    _default_model_class = LinearRegression

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        kink_point: float,
        model: PyMCModel | None = None,
        running_variable_name: str = "x",
        epsilon: float = 0.001,
        bandwidth: float = np.inf,
        **kwargs: dict,
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
            y, X = dmatrices(self.formula, filtered_data, return_type="dataframe")
            filtered_data = filtered_data.loc[X.index]
            self.data = filtered_data
        else:
            y, X = dmatrices(self.formula, self.data, return_type="dataframe")
            self.data = self.data.loc[X.index]

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
        """Run the experiment algorithm: fit model, predict, and evaluate gradient change."""
        COORDS = {
            "coeffs": self.labels,
            "obs_ind": np.arange(self.X.shape[0]),
            "treated_units": ["unit_0"],
        }
        self.model.fit(X=self.X, y=self.y, coords=COORDS)

        # score the goodness of fit to all data
        self.score = self.model.score(X=self.X, y=self.y)

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
        self.pred = self.model.predict(X=np.asarray(new_x))

        # evaluate gradient change around kink point
        mu_kink_left, mu_kink, mu_kink_right = self._probe_kink_point()
        self.gradient_change = self._eval_gradient_change(
            mu_kink_left, mu_kink, mu_kink_right, self.epsilon
        )

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
        predicted = self.model.predict(X=np.asarray(new_x))
        # extract predicted mu values
        mu_kink_left = predicted["posterior_predictive"].sel(obs_ind=0)["mu"]
        mu_kink = predicted["posterior_predictive"].sel(obs_ind=1)["mu"]
        mu_kink_right = predicted["posterior_predictive"].sel(obs_ind=2)["mu"]
        return mu_kink_left, mu_kink, mu_kink_right

    def _is_treated(self, x: np.ndarray | pd.Series) -> np.ndarray:
        """Returns ``True`` if `x` is greater than or equal to the treatment threshold."""  # noqa: E501
        return np.greater_equal(x, self.kink_point)

    def summary(self, round_to: int | None = 2) -> None:
        """Print summary of main results and model coefficients.

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers
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

    def _bayesian_plot(
        self, round_to: int | None = 2, **kwargs: dict
    ) -> tuple[plt.Figure, plt.Axes]:
        """Generate plot for regression kink designs."""
        fig, ax = plt.subplots()
        # Plot raw data
        sns.scatterplot(
            self.data,
            x=self.running_variable_name,
            y=self.outcome_variable_name,
            c="k",  # hue="treated",
            ax=ax,
        )

        # Plot model fit to data
        h_line, h_patch = plot_xY(
            self.x_pred[self.running_variable_name],
            self.pred["posterior_predictive"].mu.isel(treated_units=0),
            ax=ax,
            plot_hdi_kwargs={"color": "C1"},
        )
        handles = [(h_line, h_patch)]
        labels = ["Posterior mean"]

        # create strings to compose title
        title_info = f"{round_num(self.score['unit_0_r2'], round_to if round_to is not None else 2)} (std = {round_num(self.score['unit_0_r2_std'], round_to if round_to is not None else 2)})"
        r2 = f"Bayesian $R^2$ on all data = {title_info}"
        percentiles = self.gradient_change.quantile([0.03, 1 - 0.03]).values
        ci = (
            r"$CI_{94\%}$"
            + f"[{round_num(percentiles[0], round_to if round_to is not None else 2)}, {round_num(percentiles[1], round_to if round_to is not None else 2)}]"
        )
        grad_change = f"""
            Change in gradient = {round_num(self.gradient_change.mean(), round_to if round_to is not None else 2)},
            """
        ax.set(title=r2 + "\n" + grad_change + ci)
        # Intervention line
        ax.axvline(
            x=self.kink_point,
            ls="-",
            lw=3,
            color="r",
            label="treatment threshold",
        )
        ax.legend(
            handles=(h_tuple for h_tuple in handles),
            labels=labels,
            fontsize=LEGEND_FONT_SIZE,
        )
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
        Generate a decision-ready summary of causal effects for Regression Kink.

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
        return _effect_summary_rkink(
            self,
            direction=direction,
            alpha=alpha,
            min_effect=min_effect,
        )
