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

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrices

from causalpy.data_validation import RDDataValidator
from causalpy.experiments import ExperimentalDesign
from causalpy.pymc_models import PyMCModel


class RegressionDiscontinuity(ExperimentalDesign, RDDataValidator):
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
        super().__init__(model=model, **kwargs)
        self.expt_type = "Regression Discontinuity"
        self.data = data
        self.formula = formula
        self.running_variable_name = running_variable_name
        self.treatment_threshold = treatment_threshold
        self.epsilon = epsilon
        self.bandwidth = bandwidth
        self._input_validation()

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

        # ******** THIS IS SUBOPTIMAL AT THE MOMENT ************************************
        if isinstance(self.model, PyMCModel):
            # fit the model to the observed (pre-intervention) data
            COORDS = {"coeffs": self.labels, "obs_indx": np.arange(self.X.shape[0])}
            self.model.fit(X=self.X, y=self.y, coords=COORDS)
        else:
            self.model.fit(X=self.X, y=self.y)
        # ******************************************************************************

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

    def _is_treated(self, x):
        """Returns ``True`` if `x` is greater than or equal to the treatment threshold.

        .. warning::

            Assumes treatment is given to those ABOVE the treatment threshold.
        """
        return np.greater_equal(x, self.treatment_threshold)

    def plot(self):
        # Get a BayesianPlotComponent or OLSPlotComponent depending on the model
        plot_component = self.model.get_plot_component()
        plot_component.plot_regression_disctontinuity(self)

    def summary(self, round_to=None) -> None:
        """
        Print text output summarising the results

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """
        print("Difference in Differences experiment")
        print(f"Formula: {self.formula}")
        print(f"Running variable: {self.running_variable_name}")
        print(f"Threshold on running variable: {self.treatment_threshold}")
        print("\nResults:")
        print(f"Discontinuity at threshold = {self.discontinuity_at_threshold:.2f}")
        print("\n")
        self.print_coefficients(round_to)
