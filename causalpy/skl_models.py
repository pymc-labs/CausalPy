from functools import partial

import numpy as np
from scipy.optimize import fmin_slsqp
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel


class WeightedProportion(LinearModel, RegressorMixin):
    """
    Model which minimises sum squared error subject to:

    - All weights are bound between 0-1
    - Weights sum to 1.

    Inspiration taken from this blog post
    https://towardsdatascience.com/understanding-synthetic-control-methods-dd9a291885a1
    """

    def loss(self, W, X, y):
        return np.sqrt(np.mean((y - np.dot(X, W.T)) ** 2))

    def fit(self, X, y):
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
        return np.dot(X, self.coef_.T)
