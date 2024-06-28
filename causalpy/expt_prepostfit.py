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
from causalpy.pymc_models import BayesianModel


class PrePostFit(ExperimentalDesign, PrePostFitDataValidator):
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
        if isinstance(self.model, BayesianModel):
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
        # Get a BayesianPlotComponent or OLSPlotComponent depending on the model
        plot_component = self.model.get_plot_component()
        plot_component.plot_pre_post(self)

    def print_coefficients(self):
        self.model.print_coefficients(self.labels)


class InterruptedTimeSeries(PrePostFit):
    expt_type = "Interrupted Time Series"


class SyntheticControl(PrePostFit):
    expt_type = "SyntheticControl"
