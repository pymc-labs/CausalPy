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
"""Tests for sklearn input normalization and build_coords."""

from __future__ import annotations

import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression

from causalpy.experiments.model_adapter import (
    SklearnModelAdapter,
    build_coords,
)
from causalpy.skl_models import create_causalpy_compatible_class


def test_build_coords_defaults():
    coords = build_coords(["a", "b"], 5)
    assert coords["coeffs"] == ["a", "b"]
    assert list(coords["obs_ind"]) == [0, 1, 2, 3, 4]
    assert coords["treated_units"] == ["unit_0"]


def test_build_coords_extra_and_treated_units():
    coords = build_coords(
        ["x"],
        3,
        treated_units=["u1", "u2"],
        datetime_index=[1, 2, 3],
    )
    assert coords["treated_units"] == ["u1", "u2"]
    assert coords["datetime_index"] == [1, 2, 3]


def test_sklearn_adapter_2d_y_matches_1d_slice():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    y_1d = X @ np.array([1.0, -0.5]) + rng.normal(scale=0.1, size=20)
    y_2d = y_1d.reshape(-1, 1)

    model = create_causalpy_compatible_class(LinearRegression(fit_intercept=False))
    adapter_1d = SklearnModelAdapter(model)
    adapter_2d = SklearnModelAdapter(
        create_causalpy_compatible_class(LinearRegression(fit_intercept=False))
    )

    adapter_1d.fit(X, y_1d)
    adapter_2d.fit(X, y_2d)

    assert np.allclose(adapter_1d.coefficients(), adapter_2d.coefficients())
    assert adapter_1d.score(X, y_1d) == adapter_2d.score(X, y_2d)
    assert np.allclose(adapter_1d.predict(X), adapter_2d.predict(X))


def test_sklearn_adapter_xarray_inputs_match_numpy():
    rng = np.random.default_rng(1)
    X_np = rng.normal(size=(15, 2))
    y_np = X_np @ np.array([0.5, 1.0]) + rng.normal(scale=0.1, size=15)

    X_xr = xr.DataArray(
        X_np,
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": np.arange(15), "coeffs": ["a", "b"]},
    )
    y_xr = xr.DataArray(
        y_np.reshape(-1, 1),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": np.arange(15), "treated_units": ["unit_0"]},
    )

    adapter_np = SklearnModelAdapter(
        create_causalpy_compatible_class(LinearRegression(fit_intercept=False))
    )
    adapter_xr = SklearnModelAdapter(
        create_causalpy_compatible_class(LinearRegression(fit_intercept=False))
    )

    adapter_np.fit(X_np, y_np)
    adapter_xr.fit(X_xr, y_xr)

    assert np.allclose(adapter_np.coefficients(), adapter_xr.coefficients())
    assert adapter_np.score(X_np, y_np) == adapter_xr.score(X_xr, y_xr)
    assert np.allclose(adapter_np.predict(X_np), adapter_xr.predict(X_xr))
    mu = adapter_xr.predict_mu(X_xr)
    assert mu.dims == ("chain", "draw", "obs_ind", "treated_units")
    np.testing.assert_array_equal(mu.coords["obs_ind"], X_xr.coords["obs_ind"])
    np.testing.assert_array_equal(
        mu.coords["treated_units"], y_xr.coords["treated_units"]
    )
    np.testing.assert_allclose(mu.squeeze(), adapter_xr.predict(X_xr))


def test_sklearn_adapter_predict_ignores_out_of_sample():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(10, 1))
    y = rng.normal(size=10)
    adapter = SklearnModelAdapter(
        create_causalpy_compatible_class(LinearRegression(fit_intercept=False))
    )
    adapter.fit(X, y)
    preds_default = adapter.predict(X)
    preds_oos = adapter.predict(X, out_of_sample=True)
    assert np.allclose(preds_default, preds_oos)


def test_sklearn_predict_mu_preserves_multi_output_coordinates():
    X = xr.DataArray(
        np.arange(16).reshape(8, 2),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": np.arange(10, 18), "coeffs": ["a", "b"]},
    )
    y = xr.DataArray(
        np.column_stack((X[:, 0] + X[:, 1], X[:, 0] - X[:, 1])),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": X.obs_ind, "treated_units": ["north", "south"]},
    )
    adapter = SklearnModelAdapter(
        create_causalpy_compatible_class(LinearRegression(fit_intercept=False))
    )
    adapter.fit(X, y)

    mu = adapter.predict_mu(X)

    assert mu.dims == ("chain", "draw", "obs_ind", "treated_units")
    assert mu.shape == (1, 1, 8, 2)
    np.testing.assert_array_equal(mu.obs_ind, X.obs_ind)
    np.testing.assert_array_equal(mu.treated_units, y.treated_units)
    np.testing.assert_allclose(mu.squeeze(("chain", "draw")), adapter.predict(X))
