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
"""Custom scikit-learn models for causal inference"""

from functools import partial

import numpy as np
from scipy.optimize import fmin_slsqp
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel

from causalpy.utils import round_num


class ScikitLearnAdaptor:
    """Base class for scikit-learn models that can be used for causal inference."""

    def calculate_impact(self, y_true, y_pred):
        """Calculate the causal impact of the intervention."""
        return y_true - np.squeeze(y_pred)

    def calculate_cumulative_impact(self, impact):
        """Calculate the cumulative impact intervention."""
        return np.cumsum(impact)

    def print_coefficients(self, labels, round_to=None) -> None:
        """Print the coefficients of the model with the corresponding labels."""
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
        """Get the coefficients of the model as a numpy array."""
        return np.squeeze(self.coef_)


class WeightedProportion(ScikitLearnAdaptor, LinearModel, RegressorMixin):
    """Weighted proportion model for causal inference. Used for synthetic control
    methods for example"""

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


def create_causalpy_compatible_class(
    estimator: type[RegressorMixin],
) -> type[RegressorMixin]:
    """This function takes a scikit-learn estimator and returns a new class that is
    compatible with CausalPy."""
    _add_mixin_methods(estimator, ScikitLearnAdaptor)
    return estimator


def _add_mixin_methods(model_instance, mixin_class):
    """Utility function to bind mixin methods to an existing model instance."""
    for attr_name in dir(mixin_class):
        attr = getattr(mixin_class, attr_name)
        if callable(attr) and not attr_name.startswith("__"):
            # Bind the method to the instance
            method = attr.__get__(model_instance, model_instance.__class__)
            setattr(model_instance, attr_name, method)
    return model_instance
