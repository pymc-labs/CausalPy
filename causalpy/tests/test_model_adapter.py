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

import warnings

import numpy as np
import pytest
import xarray as xr
from sklearn.linear_model import LinearRegression

from causalpy.experiments.model_adapter import (
    PyMCModelAdapter,
    SklearnModelAdapter,
    make_model_adapter,
)
from causalpy.pymc_models import LinearRegression as PyMCLinearRegression
from causalpy.skl_models import create_causalpy_compatible_class

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2, "progressbar": False}


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


def test_sklearn_adapter_idata_raises():
    adapter = make_model_adapter(
        LinearRegression(fit_intercept=False),
        default_model_class=None,
        supports_bayes=True,
        supports_ols=True,
    )
    with pytest.raises(AttributeError, match="OLS models do not have idata"):
        _ = adapter.idata


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
    assert isinstance(score, float)
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
    assert score is not None
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
