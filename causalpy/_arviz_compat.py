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
"""Internal ArviZ Stats 1.x compatibility helpers for HDI computation.

CausalPy call sites should use these helpers instead of calling :func:`arviz.hdi`
directly. The wrappers always pass an explicit ``prob`` (defaulting to
:data:`~causalpy.constants.HDI_PROB`) so we keep 0.94 HDI semantics rather than
inheriting arviz-stats' 0.89 ETI default, and they normalize return values to a
stable :class:`xarray.DataArray` with dimension ``hdi`` and coordinates
``lower`` / ``higher``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import arviz as az
import numpy as np
import xarray as xr

from causalpy.constants import HDI_PROB
from causalpy.utils import _as_scalar

__all__ = ["hdi", "hdi_bound_arrays", "hdi_bounds"]


def _prepare_hdi_input(data: Any) -> Any:
    """Flatten non-1D ndarray inputs to preserve legacy draw-pooled semantics."""
    if isinstance(data, np.ndarray) and data.ndim != 1:
        return np.ravel(data)
    return data


def _normalize_hdi_result(result: Any) -> xr.DataArray:
    """Normalize arviz/arviz-stats HDI output to dim ``hdi`` / coords lower|higher."""
    if isinstance(result, np.ndarray):
        values = np.asarray(result, dtype=float).reshape(-1)
        if values.size != 2:
            msg = f"Expected scalar HDI ndarray of length 2, got shape {result.shape}"
            raise ValueError(msg)
        return xr.DataArray(
            values,
            dims=["hdi"],
            coords={"hdi": ["lower", "higher"]},
        )

    if isinstance(result, xr.Dataset):
        result = list(result.data_vars.values())[0]

    if not isinstance(result, xr.DataArray):
        msg = f"Unsupported HDI result type: {type(result)!r}"
        raise TypeError(msg)

    da = result
    if "ci_bound" in da.dims:
        da = da.rename({"ci_bound": "hdi"})
    if "hdi" not in da.dims:
        msg = f"HDI result missing expected bound dimension; dims={da.dims}"
        raise ValueError(msg)

    labels = [str(v) for v in da.coords["hdi"].values]
    rename_map = {}
    if "upper" in labels and "higher" not in labels:
        rename_map["upper"] = "higher"
    if rename_map:
        da = da.assign_coords(hdi=[rename_map.get(label, label) for label in labels])

    # Keep a deterministic coordinate order for downstream iloc consumers.
    return da.sel(hdi=["lower", "higher"])


def hdi(
    data: Any,
    *,
    prob: float = HDI_PROB,
    dim: str | Sequence[str] | None = None,
) -> xr.DataArray:
    """Compute HDI and return a CausalPy-stable :class:`~xarray.DataArray`.

    Parameters
    ----------
    data : Any
        Draws as :class:`~xarray.DataArray`, :class:`~xarray.Dataset`, or
        array-like. Non-1D :class:`numpy.ndarray` inputs are raveled before the
        call so ``(chain, draw)`` arrays keep legacy flattened-draw semantics.
    prob : float, default HDI_PROB
        Probability mass for the highest density interval. Defaults to
        :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
    dim : str or sequence of str, optional
        Optional sample dimension(s) forwarded to :func:`arviz.hdi`.

    Returns
    -------
    xr.DataArray
        Interval bounds with dimension ``hdi`` and coordinates ``lower`` /
        ``higher``. Non-sample dimensions (e.g. ``obs_ind``) are preserved.
    """
    prepared = _prepare_hdi_input(data)
    kwargs: dict[str, Any] = {"prob": prob}
    if dim is not None:
        kwargs["dim"] = dim
    return _normalize_hdi_result(az.hdi(prepared, **kwargs))


def hdi_bounds(
    data: Any,
    *,
    prob: float = HDI_PROB,
    dim: str | Sequence[str] | None = None,
) -> tuple[float, float]:
    """Return scalar ``(lower, upper)`` HDI bounds.

    Parameters
    ----------
    data : Any
        Draws as :class:`~xarray.DataArray`, :class:`~xarray.Dataset`, or
        array-like. Non-1D :class:`numpy.ndarray` inputs are raveled.
    prob : float, default HDI_PROB
        Probability mass for the highest density interval.
    dim : str or sequence of str, optional
        Optional sample dimension(s) forwarded to :func:`arviz.hdi`.

    Returns
    -------
    tuple of float
        Lower and upper HDI bounds.
    """
    result = hdi(data, prob=prob, dim=dim)
    return _as_scalar(result.sel(hdi="lower")), _as_scalar(result.sel(hdi="higher"))


def hdi_bound_arrays(
    data: Any,
    *,
    prob: float = HDI_PROB,
    dim: str | Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return vector ``(lower, upper)`` HDI bounds over non-sample dims.

    Parameters
    ----------
    data : Any
        Draws as :class:`~xarray.DataArray`, :class:`~xarray.Dataset`, or
        array-like. Non-1D :class:`numpy.ndarray` inputs are raveled.
    prob : float, default HDI_PROB
        Probability mass for the highest density interval.
    dim : str or sequence of str, optional
        Optional sample dimension(s) forwarded to :func:`arviz.hdi`.

    Returns
    -------
    tuple of numpy.ndarray
        Flattened lower and upper bound arrays over preserved non-sample
        dimensions.
    """
    result = hdi(data, prob=prob, dim=dim)
    lower = np.asarray(result.sel(hdi="lower").values).reshape(-1)
    upper = np.asarray(result.sel(hdi="higher").values).reshape(-1)
    return lower, upper
