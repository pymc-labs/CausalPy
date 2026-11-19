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
"""Custom scikit-learn models for causal inference."""

from dataclasses import dataclass
from functools import partial
from numbers import Integral
from typing import Any

import numpy as np
from scipy.optimize import fmin_slsqp
from sklearn.base import RegressorMixin, is_classifier
from sklearn.linear_model._base import LinearModel


@dataclass(frozen=True, init=False)
class SkforecastModel:
    """Configure a recursive skforecast model for interrupted time series.

    Requires the optional ``causalpy[skforecast]`` extra. CausalPy fits an
    owned copy of the template so it can be reused across experiments.

    Parameters
    ----------
    forecaster : skforecast.recursive.ForecasterRecursive
        Unfitted single-series regression forecaster. Set its lags, features,
        estimator, and estimator seed here. Use ``differentiation=None``.
    calibration_size : int
        Number of pre-intervention rows reserved for held-out one-step
        calibration. Must be at least two. The preceding training prefix
        must be longer than the forecaster's lag and feature window.
    n_boot : int, default 1000
        Number of future paths to simulate. Must be at least two.
    random_state : int, default 123
        Non-negative seed for residual sampling.
    """

    forecaster: Any
    calibration_size: int
    n_boot: int
    random_state: int

    def __init__(
        self,
        forecaster: Any,
        calibration_size: int,
        *,
        n_boot: int = 1000,
        random_state: int = 123,
    ) -> None:
        _validate_recursive_forecaster(forecaster)
        for name, value, minimum in (
            ("calibration_size", calibration_size, 2),
            ("n_boot", n_boot, 2),
            ("random_state", random_state, 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        object.__setattr__(self, "forecaster", forecaster)
        object.__setattr__(self, "calibration_size", int(calibration_size))
        object.__setattr__(self, "n_boot", int(n_boot))
        object.__setattr__(self, "random_state", int(random_state))


def _validate_recursive_forecaster(forecaster: Any) -> None:
    try:
        from skforecast.recursive import ForecasterRecursive
    except ModuleNotFoundError as exc:
        if exc.name != "skforecast":
            raise
        raise ImportError(
            "Install 'causalpy[skforecast]' to use SkforecastModel."
        ) from exc

    if not isinstance(forecaster, ForecasterRecursive) or is_classifier(
        forecaster.estimator
    ):
        raise ValueError(
            "SkforecastModel requires a single-series regression ForecasterRecursive. "
            "Use ForecasterRecursive with a regression estimator."
        )
    if forecaster.is_fitted:
        raise ValueError("Pass an unfitted ForecasterRecursive template.")
    if forecaster.differentiation is not None:
        raise ValueError(
            "SkforecastModel requires differentiation=None for one-step calibration."
        )


class ScikitLearnAdaptor:
    """Base class for scikit-learn models that can be used for causal inference."""

    coef_: np.ndarray

    def get_coeffs(self) -> np.ndarray:
        """Get the coefficients of the model as a numpy array."""
        return np.squeeze(self.coef_)


class WeightedProportion(ScikitLearnAdaptor, LinearModel, RegressorMixin):
    """Weighted proportion model for causal inference.

    Used for synthetic control methods, for example.
    """

    def loss(self, W: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Compute root mean squared loss with data X, weights W, and predictor y.

        Parameters
        ----------
        W : np.ndarray
            Convex combination weights.
        X : np.ndarray
            Donor matrix.
        y : np.ndarray
            Treated unit outcomes in the pre-treatment period.
        """
        weights = np.ravel(W)
        prediction = np.ravel(np.dot(X, weights))
        outcome = np.ravel(y)
        return np.sqrt(np.mean((outcome - prediction) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeightedProportion":
        """Fit model on data X with predictor y.

        Parameters
        ----------
        X : np.ndarray
            Donor matrix.
        y : np.ndarray
            Treated unit outcomes in the pre-treatment period.
        """
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict results for data X.

        Parameters
        ----------
        X : np.ndarray
            Donor matrix to predict from.
        """
        return np.dot(X, self.coef_.T)


def create_causalpy_compatible_class(
    estimator: type[RegressorMixin],
) -> type[RegressorMixin]:
    """This function takes a scikit-learn estimator and returns a new class that is
    compatible with CausalPy.

    Parameters
    ----------
    estimator : type[RegressorMixin]
        A scikit-learn estimator class to augment.
    """
    _add_mixin_methods(estimator, ScikitLearnAdaptor)
    return estimator


def _add_mixin_methods(
    model_instance: RegressorMixin, mixin_class: type
) -> RegressorMixin:
    """Utility function to bind mixin methods to an existing model instance."""
    for attr_name in dir(mixin_class):
        attr = getattr(mixin_class, attr_name)
        if callable(attr) and not attr_name.startswith("__"):
            # Bind the method to the instance
            method = attr.__get__(model_instance, model_instance.__class__)
            setattr(model_instance, attr_name, method)
    return model_instance
