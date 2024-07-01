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

from causalpy.data_validation import RegressionKinkDataValidator
from causalpy.experiments import ExperimentalDesign
from causalpy.utils import round_num


class RegressionKink(ExperimentalDesign, RegressionKinkDataValidator):
    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        kink_point: float,
        model=None,
        running_variable_name: str = "x",
        epsilon: float = 0.001,
        bandwidth: float = np.inf,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.expt_type = "Regression Kink"
        self.data = data
        self.formula = formula
        self.running_variable_name = running_variable_name
        self.kink_point = kink_point
        self.epsilon = epsilon
        self.bandwidth = bandwidth
        self._input_validation()

        if self.bandwidth is not np.inf:
            fmin = self.kink_point - self.bandwidth
            fmax = self.kink_point + self.bandwidth
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

        COORDS = {"coeffs": self.labels, "obs_indx": np.arange(self.X.shape[0])}
        self.model.fit(X=self.X, y=self.y, coords=COORDS)

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

        # evaluate gradient change around kink point
        mu_kink_left, mu_kink, mu_kink_right = self._probe_kink_point()
        self.gradient_change = self._eval_gradient_change(
            mu_kink_left, mu_kink, mu_kink_right, epsilon
        )

    @staticmethod
    def _eval_gradient_change(mu_kink_left, mu_kink, mu_kink_right, epsilon):
        """Evaluate the gradient change at the kink point.
        It works by evaluating the model below the kink point, at the kink point,
        and above the kink point.
        This is a static method for ease of testing.
        """
        gradient_left = (mu_kink - mu_kink_left) / epsilon
        gradient_right = (mu_kink_right - mu_kink) / epsilon
        gradient_change = gradient_right - gradient_left
        return gradient_change

    def _probe_kink_point(self):
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

    def _is_treated(self, x):
        """Returns ``True`` if `x` is greater than or equal to the treatment threshold."""  # noqa: E501
        return np.greater_equal(x, self.kink_point)

    def plot(self):
        # Get a BayesianPlotComponent or OLSPlotComponent depending on the model
        plot_component = self.model.get_plot_component()
        plot_component.plot_regression_kink(self)

    def summary(self, round_to=None) -> None:
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
