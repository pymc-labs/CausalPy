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
"""Internal ArviZ Stats 1.x compatibility helpers for HDI computations.

CausalPy calls these helpers instead of :func:`arviz.hdi`. They pass the
probability explicitly and normalize ArviZ's interval representation to the
historical ``hdi`` dimension with ``lower`` and ``higher`` bounds.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any

import arviz as az
import numpy as np
import xarray as xr

from causalpy.constants import HDI_PROB

__all__ = ["hdi", "hdi_bound_arrays", "hdi_bounds"]


def _validate_prob(prob: float | None) -> float:
    """Return a valid HDI probability without inheriting an ArviZ default."""
    if isinstance(prob, bool) or not isinstance(prob, Real):
        msg = f"HDI probability must be a finite real number in (0, 1), got {prob!r}"
        raise ValueError(msg)
    probability = float(prob)
    if not np.isfinite(probability) or not 0 < probability < 1:
        msg = f"HDI probability must be a finite real number in (0, 1), got {prob!r}"
        raise ValueError(msg)
    return probability


def _prepare_hdi_input(
    data: Any,
    *,
    dim: str | Sequence[str] | None,
    flatten_chains_draws: bool,
) -> xr.DataArray | np.ndarray:
    """Validate HDI input and optionally pool an unlabeled chain/draw array."""
    if isinstance(data, xr.DataArray):
        if flatten_chains_draws:
            msg = "flatten_chains_draws is only valid for a raw ndarray"
            raise TypeError(msg)
        if {"hdi", "ci_bound"} & set(data.dims):
            msg = "HDI cannot be computed from already computed interval bounds"
            raise ValueError(msg)
        return data

    if not isinstance(data, np.ndarray):
        msg = f"HDI input must be an xarray.DataArray or numpy.ndarray, got {type(data)!r}"
        raise TypeError(msg)

    if data.ndim == 1:
        if flatten_chains_draws:
            msg = "flatten_chains_draws requires a two-dimensional raw ndarray"
            raise ValueError(msg)
        return data

    if not flatten_chains_draws:
        msg = (
            "Raw ndarray HDI input must be one-dimensional; pass "
            "flatten_chains_draws=True only for an unlabeled (chain, draw) array"
        )
        raise ValueError(msg)
    if data.ndim != 2:
        msg = "flatten_chains_draws requires a two-dimensional raw ndarray"
        raise ValueError(msg)
    if dim is not None:
        msg = "flatten_chains_draws cannot be combined with dim"
        raise ValueError(msg)
    return data.ravel()


def _normalize_hdi_result(result: Any) -> xr.DataArray:
    """Return stable ``hdi=['lower', 'higher']`` bounds from an ArviZ result."""
    if isinstance(result, np.ndarray):
        values = np.asarray(result, dtype=float)
        if values.shape != (2,):
            msg = f"Expected scalar HDI ndarray of shape (2,), got shape {result.shape}"
            raise ValueError(msg)
        return xr.DataArray(
            values,
            dims=["hdi"],
            coords={"hdi": ["lower", "higher"]},
        )

    if isinstance(result, xr.Dataset):
        if len(result.data_vars) != 1:
            msg = "HDI Dataset result must contain exactly one data variable"
            raise ValueError(msg)
        result = next(iter(result.data_vars.values()))

    if not isinstance(result, xr.DataArray):
        msg = f"Unsupported HDI result type: {type(result)!r}"
        raise TypeError(msg)

    if "ci_bound" in result.dims:
        if "hdi" in result.dims:
            msg = "HDI result cannot contain both ci_bound and hdi dimensions"
            raise ValueError(msg)
        result = result.rename({"ci_bound": "hdi"})
    if "hdi" not in result.dims or "hdi" not in result.coords:
        msg = f"HDI result missing expected bound dimension; dims={result.dims}"
        raise ValueError(msg)

    labels = [str(value) for value in result.coords["hdi"].values]
    if len(labels) != 2 or set(labels) not in ({"lower", "upper"}, {"lower", "higher"}):
        msg = f"Unexpected HDI bound labels: {labels!r}"
        raise ValueError(msg)
    normalized_labels = ["higher" if label == "upper" else label for label in labels]
    result = result.assign_coords(hdi=normalized_labels)
    return result.sel(hdi=["lower", "higher"])


def hdi(
    data: Any,
    *,
    prob: float = HDI_PROB,
    dim: str | Sequence[str] | None = None,
    flatten_chains_draws: bool = False,
) -> xr.DataArray:
    """Compute an HDI with explicit CausalPy probability semantics.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        Posterior draws.
    prob : float, default=HDI_PROB
        Probability mass of the HDI.
    dim : str or sequence of str, optional
        Sample dimensions to reduce.
    flatten_chains_draws : bool, default=False
        Whether to pool an unlabeled raw ``(chain, draw)`` array.

    Returns
    -------
    xarray.DataArray
        HDI bounds with normalized ``hdi=['lower', 'higher']`` labels.
    """
    probability = _validate_prob(prob)
    prepared = _prepare_hdi_input(
        data,
        dim=dim,
        flatten_chains_draws=flatten_chains_draws,
    )
    kwargs: dict[str, Any] = {"prob": probability, "skipna": True}
    if dim is not None:
        kwargs["dim"] = dim
    return _normalize_hdi_result(az.hdi(prepared, **kwargs))


def hdi_bounds(
    data: Any,
    *,
    prob: float = HDI_PROB,
    dim: str | Sequence[str] | None = None,
    flatten_chains_draws: bool = False,
) -> tuple[float, float]:
    """Return scalar lower and upper HDI bounds.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        Posterior draws.
    prob : float, default=HDI_PROB
        Probability mass of the HDI.
    dim : str or sequence of str, optional
        Sample dimensions to reduce.
    flatten_chains_draws : bool, default=False
        Whether to pool an unlabeled raw ``(chain, draw)`` array.

    Returns
    -------
    tuple of float
        Lower and upper HDI bounds.

    Notes
    -----
    Singleton non-bound dimensions are squeezed to preserve legacy scalar paths.
    Any non-singleton dimension left after HDI reduction is rejected.
    """
    result = hdi(
        data,
        prob=prob,
        dim=dim,
        flatten_chains_draws=flatten_chains_draws,
    ).squeeze(drop=True)
    remaining_dims = set(result.dims) - {"hdi"}
    if remaining_dims:
        # Dimension names are Hashable in xarray's typing, so sort them as strings.
        dim_names = sorted(str(dim) for dim in remaining_dims)
        msg = f"Scalar HDI bounds require reduced draws; remaining dims={dim_names!r}"
        raise ValueError(msg)
    return float(result.sel(hdi="lower").item()), float(result.sel(hdi="higher").item())


def hdi_bound_arrays(
    data: Any,
    *,
    prob: float = HDI_PROB,
    dim: str | Sequence[str] | None = None,
    flatten_chains_draws: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper bounds over exactly one preserved dimension.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        Posterior draws.
    prob : float, default=HDI_PROB
        Probability mass of the HDI.
    dim : str or sequence of str, optional
        Sample dimensions to reduce.
    flatten_chains_draws : bool, default=False
        Whether to pool an unlabeled raw ``(chain, draw)`` array.

    Returns
    -------
    tuple of numpy.ndarray
        Lower and upper HDI bounds.
    """
    result = hdi(
        data,
        prob=prob,
        dim=dim,
        flatten_chains_draws=flatten_chains_draws,
    )
    preserved_dims = [name for name in result.dims if name != "hdi"]
    if len(preserved_dims) != 1:
        msg = (
            "Vector HDI bounds require exactly one preserved dimension; "
            f"got {preserved_dims!r}"
        )
        raise ValueError(msg)
    return (
        np.asarray(result.sel(hdi="lower").values),
        np.asarray(result.sel(hdi="higher").values),
    )
