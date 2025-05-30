#   Copyright 2022 - 2025 The PyMC Labs Developers
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
from causalpy.pymc_models import PyMCModel
from causalpy.utils import _is_variable_dummy_coded, convert_to_string, round_num

from .base import BaseExperiment

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
        A PyMC model
    :param running_variable_name:
        The name of the predictor variable that the treatment threshold is based upon
    :param epsilon:
        A small scalar value which determines how far above and below the treatment
        threshold to evaluate the causal impact.
    :param bandwidth:
        Data outside of the bandwidth (relative to the discontinuity) is not used to fit
        the model.

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

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        treatment_threshold: float,
        model=None,
        running_variable_name: str = "x",
        epsilon: float = 0.001,
        bandwidth: float = np.inf,
        **kwargs,
    ):
        super().__init__(model=model)
        self.expt_type = "Regression Discontinuity"
        self.data = data
        self.formula = formula
        self.running_variable_name = running_variable_name
        self.treatment_threshold = treatment_threshold
        self.epsilon = epsilon
        self.bandwidth = bandwidth
        self.input_validation()

        if self.bandwidth is not np.inf:
            fmin = self.treatment_threshold - self.bandwidth
            fmax = self.treatment_threshold + self.bandwidth
            filtered_data = self.data.query(f"{fmin} <= x <= {fmax}")
            if len(filtered_data) <= 10:
                warnings.warn(
                    f"Choice of bandwidth parameter has lead to only {len(filtered_data)} remaining datapoints. Consider increasing the bandwidth parameter.",  # noqa: E501
                    UserWarning,
                )
            y, X = dmatrices(formula, filtered_data)
        else:
            y, X = dmatrices(formula, self.data)

        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.y, self.X = np.asarray(y), np.asarray(X)
        self.outcome_variable_name = y.design_info.column_names[0]

        # turn into xarray.DataArray's
        self.X = xr.DataArray(
            self.X,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": np.arange(self.X.shape[0]),
                "coeffs": self.labels,
            },
        )
        self.y = xr.DataArray(
            self.y[:, 0],
            dims=["obs_ind"],
            coords={"obs_ind": np.arange(self.y.shape[0])},
        )

        # fit model
        if isinstance(self.model, PyMCModel):
            # fit the model to the observed (pre-intervention) data
            COORDS = {"coeffs": self.labels, "obs_ind": np.arange(self.X.shape[0])}
            self.model.fit(X=self.X, y=self.y, coords=COORDS)
        elif isinstance(self.model, RegressorMixin):
            self.model.fit(X=self.X, y=self.y)
        else:
            raise ValueError("Model type not recognized")

        # score the goodness of fit to all data
        self.score = self.model.score(X=self.X, y=self.y)

        # get the model predictions of the observed data
        if self.bandwidth is not np.inf:
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

    def input_validation(self):
        """Validate the input data and model formula for correctness"""
        if "treated" not in self.formula:
            raise FormulaException(
                "A predictor called `treated` should be in the formula"
            )

        if _is_variable_dummy_coded(self.data["treated"]) is False:
            raise DataException(
                """The treated variable should be dummy coded. Consisting of 0's and 1's only."""  # noqa: E501
            )

    def _is_treated(self, x):
        """Returns ``True`` if `x` is greater than or equal to the treatment threshold.

        .. warning::

            Assumes treatment is given to those ABOVE the treatment threshold.
        """
        return np.greater_equal(x, self.treatment_threshold)

    def summary(self, round_to=None) -> None:
        """
        Print summary of main results and model coefficients

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """
        print("Difference in Differences experiment")
        print(f"Formula: {self.formula}")
        print(f"Running variable: {self.running_variable_name}")
        print(f"Threshold on running variable: {self.treatment_threshold}")
        print("\nResults:")
        print(
            f"Discontinuity at threshold = {convert_to_string(self.discontinuity_at_threshold)}"
        )
        print("\n")
        self.print_coefficients(round_to)

    def _bayesian_plot(self, round_to=None, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """Generate plot for regression discontinuity designs."""
        fig, ax = plt.subplots()
        # Plot raw data
        sns.scatterplot(
            self.data,
            x=self.running_variable_name,
            y=self.outcome_variable_name,
            c="k",
            ax=ax,
        )

        # Plot model fit to data
        h_line, h_patch = plot_xY(
            self.x_pred[self.running_variable_name],
            self.pred["posterior_predictive"].mu,
            ax=ax,
            plot_hdi_kwargs={"color": "C1"},
        )
        handles = [(h_line, h_patch)]
        labels = ["Posterior mean"]

        # create strings to compose title
        title_info = f"{round_num(self.score.r2, round_to)} (std = {round_num(self.score.r2_std, round_to)})"
        r2 = f"Bayesian $R^2$ on all data = {title_info}"
        percentiles = self.discontinuity_at_threshold.quantile([0.03, 1 - 0.03]).values
        ci = (
            r"$CI_{94\%}$"
            + f"[{round_num(percentiles[0], round_to)}, {round_num(percentiles[1], round_to)}]"
        )
        discon = f"""
            Discontinuity at threshold = {round_num(self.discontinuity_at_threshold.mean(), round_to)},
            """
        ax.set(title=r2 + "\n" + discon + ci)
        # Intervention line
        ax.axvline(
            x=self.treatment_threshold,
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
        return (fig, ax)

    def _ols_plot(self, round_to=None, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """Generate plot for regression discontinuity designs."""
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
        ax.plot(
            self.x_pred[self.running_variable_name],
            self.pred,
            "k",
            markersize=10,
            label="model fit",
        )
        # create strings to compose title
        r2 = f"$R^2$ on all data = {round_num(self.score, round_to)}"
        discon = f"Discontinuity at threshold = {round_num(self.discontinuity_at_threshold, round_to)}"
        ax.set(title=r2 + "\n" + discon)
        # Intervention line
        ax.axvline(
            x=self.treatment_threshold,
            ls="-",
            lw=3,
            color="r",
            label="treatment threshold",
        )
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        # TODO: have to convert ax into list because it is somehow a numpy.ndarray
        return (fig, ax)
