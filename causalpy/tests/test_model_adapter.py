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
"""Tests for experiment backend model adapters."""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from causalpy.experiments.model_adapter import (
    ModelAdapter,
    PyMCForecastAdapter,
    PyMCModelAdapter,
    SklearnModelAdapter,
    make_model_adapter,
)
from causalpy.pymc_models import LinearRegression as PyMCLinearRegression
from causalpy.skl_models import create_causalpy_compatible_class

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2, "progressbar": False}


def _prediction_tree() -> xr.DataTree:
    """Return the minimal nested prediction container consumed by adapters."""
    mu = xr.DataArray(
        np.arange(6, dtype=float).reshape(1, 1, 3, 2),
        dims=("chain", "draw", "obs_ind", "treated_units"),
        coords={
            "chain": [0],
            "draw": [0],
            "obs_ind": [10, 11, 12],
            "treated_units": ["a", "b"],
            "auxiliary": ("obs_ind", ["first", "second", "third"]),
        },
    )
    return xr.DataTree.from_dict({"posterior_predictive": xr.Dataset({"mu": mu})})


class _RecordingBayesianBackend:
    """Record the exact controls forwarded by an adapter."""

    def __init__(self) -> None:
        self.predict_calls: list[dict[str, object]] = []
        self.score_calls: list[dict[str, object]] = []

    def predict(
        self,
        *,
        X: object,
        coords: dict[str, object] | None,
        out_of_sample: bool,
    ) -> xr.DataTree:
        self.predict_calls.append(
            {
                "X": X,
                "coords": coords,
                "out_of_sample": out_of_sample,
            }
        )
        return _prediction_tree()

    def score(
        self, *, X: object, y: object, coords: dict[str, object] | None
    ) -> pd.Series:
        self.score_calls.append({"X": X, "y": y, "coords": coords})
        return pd.Series({"unit_0_r2": 0.9})


class _FixedPredictionRegressor:
    """Minimal sklearn-shaped regressor with deterministic predictions."""

    def __init__(self, predictions: np.ndarray) -> None:
        self.predictions = predictions
        self.predict_inputs: list[np.ndarray] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.predict_inputs.append(X)
        return self.predictions


def test_make_model_adapter_default_pymc():
    adapter = make_model_adapter(
        None,
        default_model_class=PyMCLinearRegression,
        supports_bayes=True,
        supports_ols=True,
    )
    assert isinstance(adapter, PyMCModelAdapter)
    assert adapter.is_bayesian
    assert not adapter.is_ols
    assert adapter.supports_idata
    assert adapter.kind == "pymc"


def test_make_model_adapter_explicit_pymc():
    model = PyMCLinearRegression()
    adapter = make_model_adapter(
        model,
        default_model_class=PyMCLinearRegression,
        supports_bayes=True,
        supports_ols=True,
    )
    assert adapter.model is model
    assert adapter.is_bayesian


def test_make_model_adapter_sklearn_coercion_and_fit_intercept_warning():
    model = LinearRegression(fit_intercept=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter = make_model_adapter(
            model,
            default_model_class=PyMCLinearRegression,
            supports_bayes=True,
            supports_ols=True,
        )

    assert isinstance(adapter, SklearnModelAdapter)
    assert model.fit_intercept is True
    assert adapter.model.fit_intercept is False
    assert adapter.is_ols
    assert not adapter.is_bayesian
    assert not adapter.supports_idata
    assert any("fit_intercept" in str(w.message) for w in caught)


def test_make_model_adapter_bayes_not_supported():
    with pytest.raises(ValueError, match="Bayesian models not supported"):
        make_model_adapter(
            PyMCLinearRegression(),
            default_model_class=None,
            supports_bayes=False,
            supports_ols=True,
        )


def test_make_model_adapter_ols_not_supported():
    with pytest.raises(ValueError, match="OLS models not supported"):
        make_model_adapter(
            LinearRegression(fit_intercept=False),
            default_model_class=None,
            supports_bayes=True,
            supports_ols=False,
        )


def test_make_model_adapter_no_model_no_default_raises():
    with pytest.raises(ValueError, match="model not set or passed"):
        make_model_adapter(
            None,
            default_model_class=None,
            supports_bayes=True,
            supports_ols=True,
        )


def test_sklearn_adapter_has_explicit_idata_capability():
    adapter = make_model_adapter(
        LinearRegression(fit_intercept=False),
        default_model_class=None,
        supports_bayes=True,
        supports_ols=True,
    )
    assert not adapter.supports_idata
    assert adapter.idata is None
    with pytest.raises(TypeError, match="does not support InferenceData"):
        adapter.require_idata()


def test_unfit_pymc_adapter_requires_fitted_idata():
    adapter = PyMCModelAdapter(PyMCLinearRegression())
    assert adapter.supports_idata
    assert adapter.idata is None
    with pytest.raises(RuntimeError, match="has not been fit"):
        adapter.require_idata()


def test_sklearn_adapter_fit_predict_score():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    y = X @ np.array([1.0, -0.5]) + rng.normal(scale=0.1, size=20)
    adapter = SklearnModelAdapter(
        create_causalpy_compatible_class(LinearRegression(fit_intercept=False))
    )
    adapter.fit(X, y)
    mu = adapter.predict(X)
    score = adapter.score(X, y)
    coeffs = adapter.coefficients()

    assert mu.dims == ("chain", "draw", "obs_ind", "treated_units")
    assert mu.shape == (1, 1, 20, 1)
    np.testing.assert_allclose(mu.squeeze(), adapter.model.predict(X))
    assert list(score.index) == ["unit_0_r2"]
    assert score["unit_0_r2"] > 0.9
    assert coeffs.shape == (2,)


def test_pymc_adapter_fit_predict_score(mock_pymc_sample):
    rng = np.random.default_rng(0)

    X = xr.DataArray(
        rng.normal(size=(10, 2)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": np.arange(10), "coeffs": ["a", "b"]},
    )
    y = xr.DataArray(
        rng.normal(size=(10, 1)),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(10), "treated_units": ["unit_0"]},
    )
    coords = {
        "coeffs": ["a", "b"],
        "obs_ind": np.arange(10),
        "treated_units": ["unit_0"],
    }
    adapter = PyMCModelAdapter(PyMCLinearRegression(sample_kwargs=sample_kwargs))
    adapter.fit(X, y, coords=coords)
    assert adapter.idata is not None
    assert adapter.require_idata() is adapter.idata
    mu = adapter.predict(X)
    score = adapter.score(X, y)
    coeffs = adapter.coefficients()

    assert mu.dims == ("chain", "draw", "obs_ind", "treated_units")
    assert mu.shape == (
        sample_kwargs["chains"],
        sample_kwargs["draws"],
        len(X),
        1,
    )
    assert list(score.index) == ["unit_0_r2", "unit_0_r2_std"]
    assert np.squeeze(coeffs).shape == (2,)


def test_panel_regression_requires_explicit_model():
    """PanelRegression has no default model class."""
    import pandas as pd

    import causalpy as cp

    df = pd.DataFrame({"unit": ["a"], "time": [0], "y": [1.0], "x1": [1.0]})
    with pytest.raises(ValueError, match="model not set or passed"):
        cp.PanelRegression(
            df,
            formula="y ~ x1",
            unit_fe_variable="unit",
            time_fe_variable="time",
            model=None,
        )


def test_base_experiment_exposes_model_backend(did_data):
    """Concrete experiments expose _model_backend after construction."""
    import causalpy as cp

    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(fit_intercept=False),
    )
    assert result._model_backend.is_ols
    assert result.model is result._model_backend.model
    assert result.idata is None


@pytest.mark.parametrize(
    ("method", "argument_count", "keyword_defaults"),
    [
        (ModelAdapter.predict, 1, {"coords": None, "out_of_sample": False}),
        (PyMCModelAdapter.predict, 1, {"coords": None, "out_of_sample": False}),
        (
            PyMCForecastAdapter.predict,
            1,
            {"coords": None, "out_of_sample": False},
        ),
        (
            SklearnModelAdapter.predict,
            1,
            {"coords": None, "out_of_sample": False},
        ),
        (ModelAdapter.score, 2, {"coords": None}),
        (PyMCModelAdapter.score, 2, {"coords": None}),
        (PyMCForecastAdapter.score, 2, {"coords": None}),
        (
            SklearnModelAdapter.score,
            2,
            {
                "coords": None,
                "sample_weight": None,
                "multioutput": "raw_values",
                "force_finite": True,
            },
        ),
    ],
)
def test_adapter_predict_score_signatures_are_explicit_and_keyword_only(
    method, argument_count, keyword_defaults
):
    """Adapter controls are explicit and cannot absorb accidental keywords."""
    signature = inspect.signature(method)
    parameters = signature.parameters
    assert not {
        parameter.kind
        for parameter in parameters.values()
        if parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    assert {
        name: parameters[name].default for name in keyword_defaults
    } == keyword_defaults
    assert all(
        parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in keyword_defaults
    )

    arguments = [object(), *([object()] * argument_count)]
    with pytest.raises(TypeError):
        signature.bind(*arguments, __causalpy_signature_typo__=None)
    with pytest.raises(TypeError):
        signature.bind(*arguments, None)


@pytest.mark.parametrize(
    ("adapter_class", "coords", "out_of_sample"),
    [
        (PyMCModelAdapter, None, False),
        (PyMCModelAdapter, {"obs_ind": [10, 11, 12]}, True),
        (PyMCForecastAdapter, None, False),
        (PyMCForecastAdapter, {"obs_ind": [10, 11, 12]}, True),
    ],
)
def test_bayesian_adapters_delegate_only_named_prediction_controls(
    adapter_class, coords, out_of_sample
):
    """Bayesian adapters preserve their explicit backend controls exactly."""
    backend = _RecordingBayesianBackend()
    adapter = adapter_class(backend)
    X = object()
    y = object()

    prediction = adapter.predict(X, coords=coords, out_of_sample=out_of_sample)
    score = adapter.score(X, y, coords=coords)

    assert backend.predict_calls == [
        {"X": X, "coords": coords, "out_of_sample": out_of_sample}
    ]
    assert backend.score_calls == [{"X": X, "y": y, "coords": coords}]
    assert prediction.dims == ("chain", "draw", "obs_ind", "treated_units")
    assert "auxiliary" not in prediction.coords
    assert score.to_dict() == {"unit_0_r2": 0.9}


def test_sklearn_adapter_explicit_prediction_and_score_controls():
    """Sklearn controls preserve raw per-output weighted R-squared scores."""
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = np.array(
        [
            [0.0, 3.0],
            [2.0, 2.0],
            [4.0, 1.0],
            [8.0, -1.0],
        ]
    )
    predictions = np.array(
        [
            [0.2, 2.4],
            [1.2, 2.5],
            [4.8, 0.4],
            [6.5, -0.2],
        ]
    )
    sample_weight = np.array([1.0, 2.0, 1.0, 3.0])
    model = _FixedPredictionRegressor(predictions)
    adapter = SklearnModelAdapter(model)

    ignored_coords = {"obs_ind": [90, 91, 92, 93]}
    prediction = adapter.predict(X, coords=ignored_coords, out_of_sample=True)
    in_sample_prediction = adapter.predict(X, coords=None, out_of_sample=False)
    score = adapter.score(
        X,
        y,
        coords=ignored_coords,
        sample_weight=sample_weight,
        force_finite=False,
    )
    expected = r2_score(
        y,
        predictions,
        sample_weight=sample_weight,
        multioutput="raw_values",
        force_finite=False,
    )

    assert prediction.shape == (1, 1, 4, 2)
    np.testing.assert_allclose(prediction.isel(chain=0, draw=0).values, predictions)
    np.testing.assert_array_equal(prediction.obs_ind.values, np.arange(len(X)))
    np.testing.assert_allclose(
        in_sample_prediction.isel(chain=0, draw=0).values, predictions
    )
    np.testing.assert_array_equal(model.predict_inputs[0], X)
    np.testing.assert_allclose(score.to_numpy(), expected)
    assert list(score.index) == ["unit_0_r2", "unit_1_r2"]
    np.testing.assert_allclose(
        adapter.score(X, y, sample_weight=None).to_numpy(),
        r2_score(y, predictions, multioutput="raw_values"),
    )
    assert adapter.score(X, y, multioutput="raw_values").equals(adapter.score(X, y))
    constant_y = np.ones_like(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected_nonfinite = r2_score(
            constant_y,
            predictions,
            multioutput="raw_values",
            force_finite=False,
        )
        actual_nonfinite = adapter.score(
            X,
            constant_y,
            force_finite=False,
        ).to_numpy()
    np.testing.assert_equal(actual_nonfinite, expected_nonfinite)
    with pytest.raises(ValueError, match="multioutput"):
        adapter.score(X, y, multioutput=None)
    with pytest.raises(ValueError, match="multioutput"):
        adapter.score(X, y, multioutput="uniform_average")
    with pytest.raises(ValueError):
        adapter.score(X, y, sample_weight=sample_weight[:-1])
    np.testing.assert_allclose(
        adapter.score(X, y, coords=None).to_numpy(),
        adapter.score(X, y).to_numpy(),
    )
    with pytest.raises(TypeError):
        adapter.score(X, y, unexpected=True)
