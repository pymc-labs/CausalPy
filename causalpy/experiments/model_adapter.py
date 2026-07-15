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
"""Backend adapters for experiment model fitting and prediction."""

from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal

import arviz as az
import numpy as np
import xarray as xr
from sklearn.base import RegressorMixin, clone

from causalpy.pymc_forecast_models import PyMCForecastModel
from causalpy.pymc_models import PyMCModel
from causalpy.skl_models import create_causalpy_compatible_class

BackendKind = Literal["pymc", "sklearn", "pymc-forecast"]


def build_coords(
    coeffs: list[str] | tuple[str, ...],
    n_obs: int,
    *,
    treated_units: tuple[str, ...] | list[str] = ("unit_0",),
    **extra: Any,
) -> dict[str, Any]:
    """Build the standard PyMC coordinate dict for regression experiments.

    Parameters
    ----------
    coeffs : list of str or tuple of str
        Coefficient / predictor names for the ``coeffs`` coord.
    n_obs : int
        Number of observations; used to build ``obs_ind`` as ``np.arange(n_obs)``.
    treated_units : list of str or tuple of str, default ``("unit_0",)``
        Names for the treated-unit dimension of ``y``.
    **extra
        Additional coordinate entries merged into the result (e.g.
        ``datetime_index`` for ITS).
    """
    return {
        "coeffs": list(coeffs),
        "obs_ind": np.arange(n_obs),
        "treated_units": list(treated_units),
        **extra,
    }


def _sklearn_array(value: Any) -> np.ndarray:
    """Coerce xarray or array-like inputs to a numpy array for sklearn."""
    if isinstance(value, xr.DataArray):
        return np.asarray(value.data)
    return np.asarray(value)


def _sklearn_y(y: Any) -> np.ndarray:
    """Coerce outcome arrays to sklearn's preferred 1D shape when possible.

    Collapses a single trailing treated-units column to 1D. Genuine multi-output
    ``y`` (>1 column) is passed through unchanged; experiments whose sklearn
    backend cannot fit multiple outcomes (e.g. synthetic control's
    ``WeightedProportion``) must reject that case upstream at construction.
    """
    arr = _sklearn_array(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return np.squeeze(arr, axis=1)
    return arr


class ModelAdapter(ABC):
    """Experiment-agnostic wrapper around a CausalPy statistical backend."""

    @property
    @abstractmethod
    def model(self) -> PyMCModel | RegressorMixin | PyMCForecastModel:
        """The underlying model instance."""

    @property
    @abstractmethod
    def kind(self) -> BackendKind:
        """Backend identifier."""

    @property
    def is_bayesian(self) -> bool:
        """Whether the backend is Bayesian (PyMC or pymc-forecast)."""
        return self.kind in ("pymc", "pymc-forecast")

    @property
    def is_ols(self) -> bool:
        """Whether the backend is OLS/sklearn."""
        return self.kind == "sklearn"

    @property
    @abstractmethod
    def idata(self) -> az.InferenceData:
        """Return InferenceData for Bayesian models."""

    @abstractmethod
    def fit(
        self,
        X: Any,
        y: Any,
        *,
        coords: dict[str, Any] | None = None,
    ) -> Any:
        """Fit the model with backend-appropriate conventions.

        Parameters
        ----------
        X : array-like or xarray.DataArray
            Predictor matrix.
        y : array-like or xarray.DataArray
            Outcome vector or matrix.
        coords : dict, optional
            Coordinate metadata for PyMC models. Ignored by sklearn backends.
        """

    @abstractmethod
    def predict(
        self,
        X: Any,
        *,
        out_of_sample: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Predict with backend-appropriate conventions.

        Parameters
        ----------
        X : array-like or xarray.DataArray
            Predictor matrix for which to generate predictions.
        out_of_sample : bool, default False
            Whether predictions are out-of-sample. Used by PyMC backends only.
        **kwargs
            Additional keyword arguments forwarded to the underlying model.
        """

    @abstractmethod
    def score(self, X: Any, y: Any, **kwargs: Any) -> Any:
        """Score predictions against observed outcomes.

        Parameters
        ----------
        X : array-like or xarray.DataArray
            Predictor matrix.
        y : array-like or xarray.DataArray
            Observed outcomes.
        **kwargs
            Additional keyword arguments forwarded to the underlying model.
        """

    @abstractmethod
    def coefficients(self) -> np.ndarray:
        """Return point estimates of model coefficients."""

    @abstractmethod
    def print_coefficients(
        self, labels: list[str], round_to: int | None = None
    ) -> None:
        """Print model coefficients with labels.

        Parameters
        ----------
        labels : list of str
            Coefficient names aligned with the fitted model.
        round_to : int, optional
            Number of significant figures to round to.
        """


class PyMCModelAdapter(ModelAdapter):
    """Adapter for :class:`~causalpy.pymc_models.PyMCModel` backends.

    Parameters
    ----------
    model : PyMCModel
        Fitted or unfitted PyMC backend model.
    """

    def __init__(self, model: PyMCModel) -> None:
        self._model = model

    @property
    def model(self) -> PyMCModel:
        """The underlying PyMC model."""
        return self._model

    @property
    def kind(self) -> BackendKind:
        """Backend identifier."""
        return "pymc"

    @property
    def idata(self) -> az.InferenceData:
        """Return the model's InferenceData object."""
        return self._model.idata

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        coords: dict[str, Any] | None = None,
    ) -> az.InferenceData:
        """Fit the PyMC model.

        Parameters
        ----------
        X : array-like or xarray.DataArray
            Predictor matrix.
        y : array-like or xarray.DataArray
            Outcome vector or matrix.
        coords : dict, optional
            Coordinate metadata for the PyMC model.
        """
        return self._model.fit(X=X, y=y, coords=coords)

    def predict(
        self,
        X: Any,
        *,
        out_of_sample: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Predict using the PyMC model.

        Parameters
        ----------
        X : array-like or xarray.DataArray
            Predictor matrix for which to generate predictions.
        out_of_sample : bool, default False
            Whether predictions are out-of-sample.
        **kwargs
            Additional keyword arguments forwarded to the underlying model.
        """
        return self._model.predict(X=X, out_of_sample=out_of_sample, **kwargs)

    def score(self, X: Any, y: Any, **kwargs: Any) -> Any:
        """Score predictions from the PyMC model.

        Parameters
        ----------
        X : array-like or xarray.DataArray
            Predictor matrix.
        y : array-like or xarray.DataArray
            Observed outcomes.
        **kwargs
            Additional keyword arguments forwarded to the underlying model.
        """
        return self._model.score(X=X, y=y, **kwargs)

    def coefficients(self) -> np.ndarray:
        """Return posterior mean coefficients."""
        if self._model.idata is None:
            raise RuntimeError("Model has not been fit yet.")
        beta = self._model.idata.posterior["beta"]
        return beta.mean(dim=["chain", "draw"]).values

    def print_coefficients(
        self, labels: list[str], round_to: int | None = None
    ) -> None:
        """Print PyMC model coefficients.

        Parameters
        ----------
        labels : list of str
            Coefficient names aligned with the fitted model.
        round_to : int, optional
            Number of significant figures to round to.
        """
        self._model.print_coefficients(labels, round_to)


class SklearnModelAdapter(ModelAdapter):
    """Adapter for sklearn :class:`~sklearn.base.RegressorMixin` backends.

    Parameters
    ----------
    model : RegressorMixin
        CausalPy-compatible sklearn backend model.
    """

    def __init__(self, model: RegressorMixin) -> None:
        self._model = model

    @property
    def model(self) -> RegressorMixin:
        """The underlying sklearn model."""
        return self._model

    @property
    def kind(self) -> BackendKind:
        """Backend identifier."""
        return "sklearn"

    @property
    def idata(self) -> az.InferenceData:
        """OLS models do not expose InferenceData."""
        raise AttributeError("OLS models do not have idata.")

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        coords: dict[str, Any] | None = None,
    ) -> Any:
        """Fit the sklearn model.

        Parameters
        ----------
        X : array-like
            Predictor matrix.
        y : array-like
            Outcome vector or matrix.
        coords : dict, optional
            Ignored for sklearn backends.
        """
        return self._model.fit(X=_sklearn_array(X), y=_sklearn_y(y))

    def predict(
        self,
        X: Any,
        *,
        out_of_sample: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Predict using the sklearn model.

        Parameters
        ----------
        X : array-like
            Predictor matrix for which to generate predictions.
        out_of_sample : bool, default False
            Ignored for sklearn backends.
        **kwargs
            Additional keyword arguments forwarded to the underlying model.
        """
        return self._model.predict(X=_sklearn_array(X), **kwargs)

    def score(self, X: Any, y: Any, **kwargs: Any) -> Any:
        """Score predictions from the sklearn model.

        Parameters
        ----------
        X : array-like
            Predictor matrix.
        y : array-like
            Observed outcomes.
        **kwargs
            Additional keyword arguments forwarded to the underlying model.
        """
        return self._model.score(X=_sklearn_array(X), y=_sklearn_y(y), **kwargs)

    def coefficients(self) -> np.ndarray:
        """Return fitted sklearn coefficients."""
        return self._model.get_coeffs()

    def print_coefficients(
        self, labels: list[str], round_to: int | None = None
    ) -> None:
        """Print sklearn model coefficients.

        Parameters
        ----------
        labels : list of str
            Coefficient names aligned with the fitted model.
        round_to : int, optional
            Number of significant figures to round to.
        """
        self._model.print_coefficients(labels, round_to)


class PyMCForecastAdapter(ModelAdapter):
    """Adapter for :class:`~causalpy.pymc_forecast_models.PyMCForecastModel`
    backends.

    The wrapped model already speaks CausalPy's Bayesian conventions
    (``mu``/``y_hat`` posterior-predictive output on ``obs_ind`` /
    ``treated_units`` coords), so this adapter is pure delegation.

    Parameters
    ----------
    model : PyMCForecastModel
        Wrapped ``pymc_forecast`` backend model.
    """

    def __init__(self, model: PyMCForecastModel) -> None:
        self._model = model

    @property
    def model(self) -> PyMCForecastModel:
        """The underlying pymc-forecast wrapper."""
        return self._model

    @property
    def kind(self) -> BackendKind:
        """Backend identifier."""
        return "pymc-forecast"

    @property
    def idata(self) -> az.InferenceData:
        """Return the model's InferenceData (posterior draws)."""
        if self._model.idata is None:
            raise RuntimeError("Model has not been fit yet.")
        return self._model.idata

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        coords: dict[str, Any] | None = None,
    ) -> az.InferenceData:
        """Fit the forecasting model on the pre-period.

        Parameters
        ----------
        X : xarray.DataArray
            Design matrix with dims ``["obs_ind", "coeffs"]``.
        y : xarray.DataArray
            Outcome with dims ``["obs_ind", "treated_units"]``.
        coords : dict, optional
            Coordinate metadata; ignored (real coordinates are read from
            ``X`` and ``y``).
        """
        return self._model.fit(X=X, y=y, coords=coords)

    def predict(
        self,
        X: Any,
        *,
        out_of_sample: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Predict in-sample or forecast the counterfactual.

        Parameters
        ----------
        X : xarray.DataArray
            Design matrix for which to generate predictions.
        out_of_sample : bool, default False
            ``True`` draws the post-period counterfactual via the model's
            forecasting path.
        **kwargs
            Additional keyword arguments forwarded to the underlying model.
        """
        return self._model.predict(X=X, out_of_sample=out_of_sample, **kwargs)

    def score(self, X: Any, y: Any, **kwargs: Any) -> Any:
        """Score in-sample predictions with the Bayesian :math:`R^2`.

        Parameters
        ----------
        X : xarray.DataArray
            Design matrix.
        y : xarray.DataArray
            Observed outcomes.
        **kwargs
            Additional keyword arguments forwarded to the underlying model.
        """
        return self._model.score(X=X, y=y, **kwargs)

    def coefficients(self) -> np.ndarray:
        """Forecasting models have no design-matrix coefficients."""
        raise NotImplementedError(
            "pymc-forecast models do not expose design-matrix coefficients; "
            "inspect the fitted posterior via `.idata` instead."
        )

    def print_coefficients(
        self, labels: list[str], round_to: int | None = None
    ) -> None:
        """Print posterior summaries of the model's scalar parameters.

        Parameters
        ----------
        labels : list of str
            Design-matrix labels; ignored by forecasting models.
        round_to : int, optional
            Number of significant figures to round to.
        """
        self._model.print_coefficients(labels, round_to)


def _prepare_sklearn_model(model: RegressorMixin) -> RegressorMixin:
    """Clone, augment, and validate a sklearn estimator for CausalPy."""
    try:
        model = clone(model)
    except TypeError:
        model = copy.deepcopy(model)
    model = create_causalpy_compatible_class(model)
    if getattr(model, "fit_intercept", False):
        warnings.warn(
            f"{type(model).__name__} had fit_intercept=True, but CausalPy "
            "requires fit_intercept=False because the intercept is already "
            "included in the design matrix by patsy. A cloned copy of the "
            "model with fit_intercept=False will be used; the original "
            "instance is unchanged.",
            UserWarning,
            stacklevel=3,
        )
        model.fit_intercept = False
    return model


def make_model_adapter(
    model: PyMCModel | RegressorMixin | PyMCForecastModel | None,
    *,
    default_model_class: type[PyMCModel] | None,
    supports_bayes: bool,
    supports_ols: bool,
    supports_pymc_forecast: bool = False,
) -> ModelAdapter:
    """Resolve, validate, and wrap a model in a backend adapter.

    Parameters
    ----------
    model : PyMCModel, RegressorMixin, PyMCForecastModel, or None
        User-supplied model instance, or ``None`` to use the default.
    default_model_class : type[PyMCModel] or None
        PyMC model class used when ``model`` is ``None``.
    supports_bayes : bool
        Whether the experiment supports Bayesian backends.
    supports_ols : bool
        Whether the experiment supports OLS/sklearn backends.
    supports_pymc_forecast : bool, default False
        Whether the experiment supports pymc-forecast backends.

    Returns
    -------
    ModelAdapter
        Backend-specific adapter wrapping the resolved model.
    """
    if isinstance(model, RegressorMixin):
        model = _prepare_sklearn_model(model)

    if model is None and default_model_class is not None:
        model = default_model_class()

    if model is None:
        raise ValueError("model not set or passed.")

    if isinstance(model, PyMCModel):
        if not supports_bayes:
            raise ValueError("Bayesian models not supported.")
        return PyMCModelAdapter(model)

    if isinstance(model, RegressorMixin):
        if not supports_ols:
            raise ValueError("OLS models not supported.")
        return SklearnModelAdapter(model)

    if isinstance(model, PyMCForecastModel):
        if not supports_pymc_forecast:
            raise ValueError("pymc-forecast models not supported.")
        return PyMCForecastAdapter(model)

    raise ValueError("Unsupported model type")
