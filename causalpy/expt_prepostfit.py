#   Copyright 2024 The PyMC Labs Developers
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
from typing import Union

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrices

from causalpy.data_validation import PrePostFitDataValidator
from causalpy.experiments import ExperimentalDesign
from causalpy.pymc_models import PyMCModel


class PrePostFit(ExperimentalDesign, PrePostFitDataValidator):
    """
    A base class for quasi-experimental designs where parameter estimation is based on
    just pre-intervention data. This class is not directly invoked by the user.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_time: Union[int, float, pd.Timestamp],
        formula: str,
        model=None,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._input_validation(data, treatment_time)
        self.treatment_time = treatment_time
        # set experiment type - usually done in subclasses
        self.expt_type = "Pre-Post Fit"
        # split data in to pre and post intervention
        self.datapre = data[data.index < self.treatment_time]
        self.datapost = data[data.index >= self.treatment_time]

        self.formula = formula

        # set things up with pre-intervention data
        y, X = dmatrices(formula, self.datapre)
        self.outcome_variable_name = y.design_info.column_names[0]
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.pre_y, self.pre_X = np.asarray(y), np.asarray(X)
        # process post-intervention data
        (new_y, new_x) = build_design_matrices(
            [self._y_design_info, self._x_design_info], self.datapost
        )
        self.post_X = np.asarray(new_x)
        self.post_y = np.asarray(new_y)

        # fit the model to the observed (pre-intervention) data
        # ******** THIS IS SUBOPTIMAL AT THE MOMENT ************************************
        if isinstance(self.model, PyMCModel):
            COORDS = {"coeffs": self.labels, "obs_indx": np.arange(self.pre_X.shape[0])}
            self.model.fit(X=self.pre_X, y=self.pre_y, coords=COORDS)
        else:
            self.model.fit(X=self.pre_X, y=self.pre_y)
        # ******************************************************************************

        # score the goodness of fit to the pre-intervention data
        self.score = self.model.score(X=self.pre_X, y=self.pre_y)

        # get the model predictions of the observed (pre-intervention) data
        self.pre_pred = self.model.predict(X=self.pre_X)

        # calculate the counterfactual
        self.post_pred = self.model.predict(X=self.post_X)
        self.pre_impact = self.model.calculate_impact(self.pre_y[:, 0], self.pre_pred)
        self.post_impact = self.model.calculate_impact(
            self.post_y[:, 0], self.post_pred
        )
        self.post_impact_cumulative = self.model.calculate_cumulative_impact(
            self.post_impact
        )

    def plot(self):
        """
        Plot the results

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """
        # Get a BayesianPlotComponent or OLSPlotComponent depending on the model
        plot_component = self.model.get_plot_component()
        fig, ax = plot_component.plot_pre_post(self)
        return fig, ax

    def summary(self, round_to=None) -> None:
        """Print summary of main results and model coefficients.

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers
        """
        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        self.print_coefficients(round_to)


class InterruptedTimeSeries(PrePostFit):
    """
    A wrapper around PrePostFit class

    :param data:
        A pandas dataframe
    :param treatment_time:
        The time when treatment occured, should be in reference to the data index
    :param formula:
        A statistical model formula
    :param model:
        A PyMC model

    Example
    --------
    >>> import causalpy as cp
    >>> df = (
    ...     cp.load_data("its")
    ...     .assign(date=lambda x: pd.to_datetime(x["date"]))
    ...     .set_index("date")
    ... )
    >>> treatment_time = pd.to_datetime("2017-01-01")
    >>> seed = 42
    >>> result = cp.InterruptedTimeSeries(
    ...     df,
    ...     treatment_time,
    ...     formula="y ~ 1 + t + C(month)",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={
    ...             "target_accept": 0.95,
    ...             "random_seed": seed,
    ...             "progressbar": False,
    ...         }
    ...     )
    ... )
    """

    expt_type = "Interrupted Time Series"


class SyntheticControl(PrePostFit):
    """A wrapper around the PrePostFit class

    :param data:
        A pandas dataframe
    :param treatment_time:
        The time when treatment occured, should be in reference to the data index
    :param formula:
        A statistical model formula
    :param model:
        A PyMC model

    Example
    --------
    >>> import causalpy as cp
    >>> df = cp.load_data("sc")
    >>> treatment_time = 70
    >>> seed = 42
    >>> result = cp.SyntheticControl(
    ...     df,
    ...     treatment_time,
    ...     formula="actual ~ 0 + a + b + c + d + e + f + g",
    ...     model=cp.pymc_models.WeightedSumFitter(
    ...         sample_kwargs={
    ...             "target_accept": 0.95,
    ...             "random_seed": seed,
    ...             "progressbar": False,
    ...         }
    ...     ),
    ... )
    """

    expt_type = "SyntheticControl"
