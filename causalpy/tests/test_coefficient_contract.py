#   Copyright 2026 - 2026 The PyMC Labs Developers
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
"""Contract tests for canonical model coefficient containers."""

import arviz as az
import numpy as np
import pytest
import xarray as xr
from sklearn.linear_model import LinearRegression

from causalpy.experiments.model_adapter import (
    PyMCForecastAdapter,
    PyMCModelAdapter,
    SklearnModelAdapter,
)
from causalpy.pymc_forecast_models import PyMCForecastModel
from causalpy.pymc_models import LinearRegression as PyMCLinearRegression
from causalpy.skl_models import create_causalpy_compatible_class


def test_sklearn_coefficients_embed_point_estimate_as_singleton_draws():
    X = xr.DataArray(
        np.arange(12).reshape(6, 2),
        dims=("obs_ind", "coeffs"),
        coords={"obs_ind": np.arange(6), "coeffs": ["intercept", "slope"]},
    )
    y = xr.DataArray(
        (X[:, 0] - 2 * X[:, 1]).values[:, None],
        dims=("obs_ind", "treated_units"),
        coords={"obs_ind": X.obs_ind, "treated_units": ["outcome"]},
    )
    adapter = SklearnModelAdapter(
        create_causalpy_compatible_class(LinearRegression(fit_intercept=False))
    )
    adapter.fit(X, y)

    coefficients = adapter.coefficients()

    assert coefficients.name == "coefficients"
    assert coefficients.dims == ("chain", "draw", "coeffs", "treated_units")
    assert coefficients.shape == (1, 1, 2, 1)
    np.testing.assert_array_equal(coefficients.coeffs, X.coeffs)
    np.testing.assert_array_equal(coefficients.treated_units, y.treated_units)
    np.testing.assert_allclose(
        coefficients.mean(("chain", "draw")).squeeze("treated_units"),
        adapter.model.get_coeffs(),
    )


def test_sklearn_coefficients_preserve_multi_output_orientation():
    X = xr.DataArray(
        np.arange(16).reshape(8, 2),
        dims=("obs_ind", "coeffs"),
        coords={"obs_ind": np.arange(8), "coeffs": ["a", "b"]},
    )
    y = xr.DataArray(
        np.column_stack((X[:, 0] + X[:, 1], X[:, 0] - X[:, 1])),
        dims=("obs_ind", "treated_units"),
        coords={"obs_ind": X.obs_ind, "treated_units": ["north", "south"]},
    )
    adapter = SklearnModelAdapter(
        create_causalpy_compatible_class(LinearRegression(fit_intercept=False))
    )
    adapter.fit(X, y)

    coefficients = adapter.coefficients()

    assert coefficients.dims == ("chain", "draw", "coeffs", "treated_units")
    assert coefficients.shape == (1, 1, 2, 2)
    np.testing.assert_array_equal(coefficients.treated_units, y.treated_units)
    np.testing.assert_allclose(
        coefficients.squeeze(("chain", "draw")).values,
        adapter.model.coef_.T,
    )


def test_sklearn_coefficients_reject_invalid_shapes_and_coordinates():
    X = np.arange(12).reshape(6, 2)
    y = X @ np.array([1.0, -1.0])
    adapter = SklearnModelAdapter(
        create_causalpy_compatible_class(LinearRegression(fit_intercept=False))
    )
    adapter.fit(X, y)

    adapter._coeffs = np.asarray(["only_one"])
    with pytest.raises(ValueError, match="does not match the predictors"):
        adapter.coefficients()

    adapter._coeffs = np.asarray(["a", "b"])
    adapter._treated_units = np.asarray(["north", "south"])
    with pytest.raises(ValueError, match="output columns do not match"):
        adapter.coefficients()

    adapter.model.__dict__["coef_"] = np.zeros((1, 1, 2))
    with pytest.raises(ValueError, match="Expected sklearn coefficients"):
        adapter.coefficients()

    adapter.model.__dict__["coef_"] = np.zeros((2, 2))
    adapter._treated_units = np.asarray(["north"])
    with pytest.raises(ValueError, match="output rows do not match"):
        adapter.coefficients()


def test_pymc_coefficients_preserve_draws_and_normalize_dimension_order():
    draws = xr.DataArray(
        np.arange(24).reshape(2, 3, 1, 4),
        dims=("chain", "draw", "treated_units", "coeffs"),
        coords={
            "chain": [0, 1],
            "draw": [0, 1, 2],
            "treated_units": ["outcome"],
            "coeffs": ["a", "b", "c", "d"],
        },
    )
    model = PyMCLinearRegression()
    model.idata = az.InferenceData(posterior=xr.Dataset({"beta": draws}))

    coefficients = PyMCModelAdapter(model).coefficients()

    assert coefficients.name == "coefficients"
    assert coefficients.dims == ("chain", "draw", "coeffs", "treated_units")
    xr.testing.assert_equal(
        coefficients,
        draws.transpose("chain", "draw", "coeffs", "treated_units").rename(
            "coefficients"
        ),
    )
    np.testing.assert_allclose(
        coefficients.mean(("chain", "draw")).values,
        draws.mean(("chain", "draw")).transpose("coeffs", "treated_units").values,
    )


@pytest.mark.parametrize(
    ("variable", "source_dim"),
    [("b", "coeffs"), ("beta_z", "covariates")],
)
def test_pymc_coefficients_normalize_supported_model_families(variable, source_dim):
    draws = xr.DataArray(
        np.arange(12).reshape(2, 3, 2),
        dims=("chain", "draw", source_dim),
        coords={"chain": [0, 1], "draw": [0, 1, 2], source_dim: ["a", "b"]},
    )
    model = PyMCLinearRegression()
    model.idata = az.InferenceData(posterior=xr.Dataset({variable: draws}))

    coefficients = PyMCModelAdapter(model).coefficients()

    assert coefficients.dims == ("chain", "draw", "coeffs")
    np.testing.assert_array_equal(coefficients.coeffs, ["a", "b"])
    np.testing.assert_array_equal(coefficients, draws)


@pytest.mark.parametrize(
    ("dims", "shape", "match"),
    [
        (("draw", "coeffs"), (3, 2), "must include dimensions"),
        (
            ("chain", "draw", "coeffs", "extra"),
            (2, 3, 2, 1),
            "unsupported dimensions",
        ),
    ],
)
def test_pymc_coefficients_reject_noncanonical_dimensions(dims, shape, match):
    draws = xr.DataArray(np.zeros(shape), dims=dims)
    model = PyMCLinearRegression()
    model.idata = az.InferenceData(posterior=xr.Dataset({"beta": draws}))

    with pytest.raises(ValueError, match=match):
        PyMCModelAdapter(model).coefficients()


def test_shared_print_coefficients_dispatches_on_draw_count(capsys):
    X = xr.DataArray(
        np.arange(12).reshape(6, 2),
        dims=("obs_ind", "coeffs"),
        coords={"obs_ind": np.arange(6), "coeffs": ["a", "b"]},
    )
    y = xr.DataArray(
        (X[:, 0] - X[:, 1]).values[:, None],
        dims=("obs_ind", "treated_units"),
        coords={"obs_ind": X.obs_ind, "treated_units": ["outcome"]},
    )
    sklearn_adapter = SklearnModelAdapter(
        create_causalpy_compatible_class(LinearRegression(fit_intercept=False))
    )
    sklearn_adapter.fit(X, y)
    sklearn_adapter.print_coefficients(["a", "b"])
    point_output = capsys.readouterr().out

    draws = xr.DataArray(
        np.arange(12).reshape(2, 3, 2),
        dims=("chain", "draw", "coeffs"),
        coords={"chain": [0, 1], "draw": [0, 1, 2], "coeffs": ["a", "b"]},
    )
    pymc_model = PyMCLinearRegression()
    pymc_model.idata = az.InferenceData(posterior=xr.Dataset({"beta": draws}))
    PyMCModelAdapter(pymc_model).print_coefficients(["a", "b"])
    posterior_output = capsys.readouterr().out

    assert "HDI" not in point_output
    assert "94% HDI" in posterior_output


def test_pymc_forecast_coefficients_and_printing_are_unsupported():
    adapter = PyMCForecastAdapter(PyMCForecastModel.__new__(PyMCForecastModel))

    with pytest.raises(NotImplementedError, match="do not expose"):
        adapter.coefficients()
    with pytest.raises(NotImplementedError, match="do not expose"):
        adapter.print_coefficients(["a"])
