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
"""Regression tests for the ArviZ 1.x HDI compatibility boundary."""

import arviz as az
import numpy as np
import pytest
import xarray as xr

from causalpy._arviz_compat import hdi, hdi_bound_arrays, hdi_bounds
from causalpy.constants import HDI_PROB


@pytest.fixture
def normal_draws() -> xr.DataArray:
    """Frozen chain/draw values captured against ArviZ 0.22 HDI behavior."""
    return xr.DataArray(
        np.random.default_rng(42).normal(size=(2, 200)),
        dims=["chain", "draw"],
    )


def test_hdi_normalizes_arviz_stats_ci_bound(monkeypatch):
    def fake_hdi(data, prob=None, **kwargs):
        assert prob == HDI_PROB
        return xr.DataArray(
            [1.0, 3.0], dims=["ci_bound"], coords={"ci_bound": ["lower", "upper"]}
        )

    monkeypatch.setattr("causalpy._arviz_compat.az.hdi", fake_hdi)

    result = hdi(xr.DataArray([0.0, 1.0]))

    assert result.dims == ("hdi",)
    assert list(result.hdi.values) == ["lower", "higher"]
    assert result.values == pytest.approx([1.0, 3.0])


def test_hdi_normalizes_historical_single_variable_dataset_result(monkeypatch):
    def fake_hdi(data, prob=None, **kwargs):
        return xr.Dataset(
            {
                "effect": xr.DataArray(
                    [0.2, 0.8],
                    dims=["hdi"],
                    coords={"hdi": ["lower", "higher"]},
                )
            }
        )

    monkeypatch.setattr("causalpy._arviz_compat.az.hdi", fake_hdi)

    result = hdi(xr.DataArray([0.0, 1.0]))

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("hdi",)
    assert list(result.hdi.values) == ["lower", "higher"]


def test_hdi_rejects_multi_variable_dataset_result(monkeypatch):
    def fake_hdi(data, prob=None, **kwargs):
        return xr.Dataset(
            {
                "effect": xr.DataArray(
                    [0.2, 0.8],
                    dims=["ci_bound"],
                    coords={"ci_bound": ["lower", "upper"]},
                ),
                "other": xr.DataArray(
                    [0.1, 0.9],
                    dims=["ci_bound"],
                    coords={"ci_bound": ["lower", "upper"]},
                ),
            }
        )

    monkeypatch.setattr("causalpy._arviz_compat.az.hdi", fake_hdi)

    with pytest.raises(ValueError, match="exactly one data variable"):
        hdi(xr.DataArray([0.0, 1.0]))


def test_hdi_rejects_unsupported_dataset_and_non_array_inputs():
    with pytest.raises(TypeError, match="xarray.DataArray or numpy.ndarray"):
        hdi([0.0, 1.0])
    with pytest.raises(TypeError, match="xarray.DataArray or numpy.ndarray"):
        hdi(xr.Dataset({"effect": xr.DataArray([0.0, 1.0])}))


def test_hdi_rejects_precomputed_interval_input():
    interval = xr.DataArray(
        [0.0, 1.0], dims=["hdi"], coords={"hdi": ["lower", "higher"]}
    )

    with pytest.raises(ValueError, match="already computed interval"):
        hdi(interval)


def test_hdi_preserves_one_non_sample_dimension():
    draws = xr.DataArray(
        np.random.default_rng(0).normal(size=(2, 40, 3)),
        dims=["chain", "draw", "obs_ind"],
        coords={"obs_ind": [3, 1, 2]},
    )

    result = hdi(draws)
    lower, upper = hdi_bound_arrays(draws)

    assert result.dims == ("obs_ind", "hdi")
    assert result.obs_ind.values.tolist() == [3, 1, 2]
    assert lower.shape == upper.shape == (3,)
    assert np.all(lower <= upper)


def test_hdi_bound_arrays_preserves_a_singleton_vector_dimension():
    draws = xr.DataArray(
        np.random.default_rng(3).normal(size=(2, 40, 1)),
        dims=["chain", "draw", "obs_ind"],
        coords={"obs_ind": [3]},
    )

    lower, upper = hdi_bound_arrays(draws)

    assert lower.shape == upper.shape == (1,)
    assert lower[0] <= upper[0]


def test_hdi_bounds_squeezes_singleton_dimensions_and_rejects_vectors():
    singleton = xr.DataArray(
        np.random.default_rng(1).normal(size=(2, 40, 1)),
        dims=["chain", "draw", "treated_units"],
        coords={"treated_units": ["unit_0"]},
    )
    vector = xr.concat([singleton, singleton + 1], dim="treated_units").assign_coords(
        treated_units=["unit_0", "unit_1"]
    )

    lower, upper = hdi_bounds(singleton)

    assert lower < upper
    with pytest.raises(ValueError, match="remaining dims"):
        hdi_bounds(vector)


def test_hdi_bound_arrays_rejects_multiple_preserved_dimensions():
    draws = xr.DataArray(
        np.random.default_rng(1).normal(size=(2, 40, 2, 3)),
        dims=["chain", "draw", "treated_units", "obs_ind"],
    )

    with pytest.raises(ValueError, match="exactly one preserved dimension"):
        hdi_bound_arrays(draws)


def test_hdi_bound_arrays_rejects_an_extra_dimension_when_obs_ind_is_singleton():
    draws = xr.DataArray(
        np.random.default_rng(3).normal(size=(2, 40, 1, 2)),
        dims=["chain", "draw", "obs_ind", "treated_units"],
        coords={"obs_ind": [3], "treated_units": ["unit_0", "unit_1"]},
    )

    with pytest.raises(ValueError, match="exactly one preserved dimension"):
        hdi_bound_arrays(draws)


def test_hdi_raw_one_dimensional_input_is_unchanged(monkeypatch):
    seen = {}

    def fake_hdi(data, prob=None, **kwargs):
        seen["shape"] = np.asarray(data).shape
        seen["prob"] = prob
        return np.array([-1.0, 1.0])

    monkeypatch.setattr("causalpy._arviz_compat.az.hdi", fake_hdi)

    assert hdi_bounds(np.arange(10.0)) == pytest.approx((-1.0, 1.0))
    assert seen == {"shape": (10,), "prob": HDI_PROB}


def test_hdi_rejects_non_scalar_ndarray_results(monkeypatch):
    monkeypatch.setattr(
        "causalpy._arviz_compat.az.hdi",
        lambda data, prob=None, **kwargs: np.array([[-1.0, 1.0]]),
    )

    with pytest.raises(ValueError, match="shape \\(2,\\)"):
        hdi(np.arange(10.0))


@pytest.mark.parametrize("prob", [None, np.nan, 0.0, 1.0, 1.1, True, "0.94", b"0.94"])
def test_hdi_rejects_probabilities_that_would_inherit_or_exceed_defaults(prob):
    with pytest.raises(ValueError, match="finite real number in"):
        hdi_bounds(np.arange(10.0), prob=prob)


def test_hdi_raw_chain_draw_requires_explicit_pooling(monkeypatch):
    seen = {}

    def fake_hdi(data, prob=None, **kwargs):
        seen["shape"] = np.asarray(data).shape
        return np.array([-1.0, 1.0])

    monkeypatch.setattr("causalpy._arviz_compat.az.hdi", fake_hdi)
    draws = np.zeros((4, 100))

    with pytest.raises(ValueError, match="flatten_chains_draws=True"):
        hdi_bounds(draws)

    assert hdi_bounds(draws, flatten_chains_draws=True) == pytest.approx((-1.0, 1.0))
    assert seen["shape"] == (400,)


@pytest.mark.parametrize(
    ("data", "dim", "match"),
    [
        (np.zeros((2, 3, 4)), None, "two-dimensional"),
        (np.zeros((2, 3)), "draw", "cannot be combined with dim"),
        (xr.DataArray(np.zeros((2, 3)), dims=["chain", "draw"]), None, "raw ndarray"),
    ],
)
def test_hdi_rejects_invalid_raw_pooling_requests(data, dim, match):
    with pytest.raises((TypeError, ValueError), match=match):
        hdi(data, dim=dim, flatten_chains_draws=True)


def test_fixed_seed_legacy_hdi_baseline(normal_draws):
    """ArviZ 0.22 HDI@0.94 for default_rng(42).normal(size=(2, 200))."""
    expected = (-1.7577283913566313, 1.732311605409944)

    labeled = hdi_bounds(normal_draws)
    pooled = hdi_bounds(normal_draws.values, flatten_chains_draws=True)
    flattened = hdi_bounds(normal_draws.values.ravel())

    assert labeled == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert pooled == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert flattened == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_default_hdi_prob_is_explicit_094_hdi_not_default_or_eti():
    """ArviZ 0.22 HDI@0.94 for default_rng(42).exponential(size=(2, 500))."""
    skew = xr.DataArray(
        np.random.default_rng(42).exponential(size=(2, 500)),
        dims=["chain", "draw"],
    )
    expected = (0.0011048458009492535, 2.903332935595461)
    eti = tuple(np.quantile(skew.values, [0.03, 0.97]))
    default_probability = tuple(az.hdi(skew).values)

    default_bounds = hdi_bounds(skew)
    explicit_bounds = hdi_bounds(skew, prob=HDI_PROB)

    assert default_bounds == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert explicit_bounds == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert default_bounds != pytest.approx(default_probability)
    assert default_bounds != pytest.approx(eti)


def test_hdi_skips_missing_draws(normal_draws):
    """Missing posterior draws retain a finite, ordered HDI."""
    draws = normal_draws.copy(deep=True)
    draws.values[0, 0] = np.nan

    expected = tuple(az.hdi(draws, prob=HDI_PROB, skipna=True).values)
    observed = hdi_bounds(draws)

    assert observed == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert np.isfinite(observed).all()
    assert observed[0] <= observed[1]
