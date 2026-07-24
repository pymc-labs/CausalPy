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
"""Tests for causalpy._arviz_compat HDI helpers."""

import numpy as np
import pytest
import xarray as xr

from causalpy._arviz_compat import hdi, hdi_bound_arrays, hdi_bounds
from causalpy.constants import HDI_PROB
from causalpy.plot_utils import get_hdi_to_df


def test_hdi_normalizes_ci_bound_upper_to_hdi_higher(monkeypatch):
    def fake_hdi(data, prob=None, **kwargs):
        assert prob == HDI_PROB
        return xr.DataArray(
            [1.0, 3.0], dims=["ci_bound"], coords={"ci_bound": ["lower", "upper"]}
        )

    monkeypatch.setattr("causalpy._arviz_compat.az.hdi", fake_hdi)
    out = hdi(xr.DataArray([0.0, 1.0]))
    assert list(out.dims) == ["hdi"]
    assert list(out.coords["hdi"].values) == ["lower", "higher"]
    assert float(out.sel(hdi="lower")) == pytest.approx(1.0)
    assert float(out.sel(hdi="higher")) == pytest.approx(3.0)


def test_hdi_dataset_input_returns_dataarray(monkeypatch):
    def fake_hdi(data, prob=None, **kwargs):
        return xr.Dataset(
            {
                "effect": xr.DataArray(
                    [0.2, 0.8],
                    dims=["ci_bound"],
                    coords={"ci_bound": ["lower", "upper"]},
                )
            }
        )

    monkeypatch.setattr("causalpy._arviz_compat.az.hdi", fake_hdi)
    out = hdi(xr.Dataset({"effect": xr.DataArray([1.0, 2.0])}))
    assert isinstance(out, xr.DataArray)
    assert list(out.coords["hdi"].values) == ["lower", "higher"]


def test_hdi_preserves_obs_ind_dimension():
    rng = np.random.default_rng(0)
    da = xr.DataArray(
        rng.normal(size=(2, 40, 3)),
        dims=["chain", "draw", "obs_ind"],
        coords={"obs_ind": [0, 1, 2]},
    )
    out = hdi(da, prob=0.94)
    assert "obs_ind" in out.dims
    assert out.sizes["obs_ind"] == 3
    assert list(out.coords["hdi"].values) == ["lower", "higher"]
    lower, upper = hdi_bound_arrays(da, prob=0.94)
    assert lower.shape == (3,)
    assert upper.shape == (3,)
    assert np.all(lower <= upper)


def test_ndarray_non_1d_is_raveled(monkeypatch):
    seen = {}

    def fake_hdi(data, prob=None, **kwargs):
        seen["shape"] = np.asarray(data).shape
        return np.asarray([-1.0, 1.0])

    monkeypatch.setattr("causalpy._arviz_compat.az.hdi", fake_hdi)
    arr = np.zeros((4, 100))
    lower, upper = hdi_bounds(arr, prob=0.94)
    assert seen["shape"] == (400,)
    assert lower == pytest.approx(-1.0)
    assert upper == pytest.approx(1.0)


def test_hdi_always_passes_explicit_prob(monkeypatch):
    seen = {}

    def fake_hdi(data, prob=None, **kwargs):
        seen["prob"] = prob
        return np.asarray([0.0, 1.0])

    monkeypatch.setattr("causalpy._arviz_compat.az.hdi", fake_hdi)
    hdi_bounds(np.arange(10.0))
    assert seen["prob"] == HDI_PROB


def test_fixed_seed_legacy_baseline_hdi_bounds():
    """Deterministic HDI bounds for a frozen chain×draw array at prob=0.94.

    Baseline captured with arviz 0.22.0 via ``az.hdi(..., hdi_prob=0.94)`` on this
    seeded array; arviz-stats 1.2.0 labeled chain×draw path with ``prob=0.94``
    matches exactly (verified in CausalPy and mamba_temp_env_issue_1041 envs).
    """
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=2.0, scale=1.0, size=(4, 200))
    da = xr.DataArray(samples, dims=["chain", "draw"])
    lower, upper = hdi_bounds(da, prob=0.94)
    assert lower == pytest.approx(0.21329248863192207, rel=1e-6, abs=1e-6)
    assert upper == pytest.approx(3.8478250129560454, rel=1e-6, abs=1e-6)


def test_get_hdi_to_df_column_order_lower_higher():
    rng = np.random.default_rng(1)
    da = xr.DataArray(
        rng.normal(size=(2, 30, 5)),
        dims=["chain", "draw", "obs_ind"],
    )
    result = get_hdi_to_df(da, hdi_prob=0.94)
    assert list(result.columns) == ["lower", "higher"]
    assert (result["lower"] <= result["higher"]).all()
