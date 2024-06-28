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
"""Custom scikit-learn models for causal inference"""

from functools import partial

import numpy as np
from scipy.optimize import fmin_slsqp
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import LinearModel

from causalpy.plotting import OLSPlotComponent, PlotComponent
from causalpy.utils import round_num


class WeightedProportion(LinearModel, RegressorMixin):
    """
    Model which minimises sum squared error subject to:

    - All weights are bound between 0-1
    - Weights sum to 1.

    Inspiration taken from this blog post
    https://towardsdatascience.com/understanding-synthetic-control-methods-dd9a291885a1

    Example
    --------
    >>> import numpy as np
    >>> from causalpy.skl_models import WeightedProportion
    >>> rng = np.random.default_rng(seed=42)
    >>> X = rng.normal(loc=0, scale=1, size=(20,2))
    >>> y = rng.normal(loc=0, scale=1, size=(20,))
    >>> wp = WeightedProportion()
    >>> wp.fit(X, y)
    WeightedProportion()
    >>> wp.coef_
    array([[0.36719946, 0.63280054]])
    >>> X_new = rng.normal(loc=0, scale=1, size=(10,2))
    >>> wp.predict(X_new)
    array(...)
    """

    def loss(self, W, X, y):
        """Compute root mean squared loss with data X, weights W, and predictor y"""
        return np.sqrt(np.mean((y - np.dot(X, W.T)) ** 2))

    def fit(self, X, y):
        """Fit model on data X with predictor y"""
        w_start = [1 / X.shape[1]] * X.shape[1]
        coef_ = fmin_slsqp(
            partial(self.loss, X=X, y=y),
            np.array(w_start),
            f_eqcons=lambda w: np.sum(w) - 1,
            bounds=[(0.0, 1.0)] * len(w_start),
            disp=False,
        )
        self.coef_ = np.atleast_2d(coef_)  # return as column vector
        self.mse = self.loss(W=self.coef_, X=X, y=y)
        return self

    def predict(self, X):
        """Predict results for data X"""
        return np.dot(X, self.coef_.T)


class OLSLinearRegression(LinearRegression):
    # def fit(self, X, y, **kwargs):
    #     # Remove any additional keyword arguments that are not accepted by the original fit method
    #     valid_kwargs = self._get_valid_fit_kwargs()
    #     kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}

    #     # Call the original fit method with the modified keyword arguments
    #     super().fit(X, y, **kwargs)

    # def _get_valid_fit_kwargs(self):
    #     # Get the list of valid keyword arguments accepted by the original fit method
    #     valid_kwargs = set(LinearRegression().get_params().keys())
    #     return valid_kwargs

    def calculate_impact(self, y_true, y_pred):
        return y_true - y_pred

    def calculate_cumulative_impact(self, impact):
        return np.cumsum(impact)

    # def plot_model_fit(self, ax):
    #     ax.plot(self.datapre.index, self.pre_pred, c="k", label="model fit")

    def get_plot_component(self) -> PlotComponent:
        return OLSPlotComponent()

    def print_coefficients(self, labels, round_to=None) -> None:
        print("Model coefficients:")
        coef_ = self.get_coeffs()
        # Determine the width of the longest label
        max_label_length = max(len(name) for name in labels)
        # Print each coefficient with formatted alignment
        for name, val in zip(labels, coef_):
            # Left-align the name
            formatted_name = f"{name:<{max_label_length}}"
            # Right-align the value with width 10
            formatted_val = f"{round_num(val, round_to):>10}"
            print(f"  {formatted_name}\t{formatted_val}")

    def get_coeffs(self):
        return np.squeeze(self.coef_)
