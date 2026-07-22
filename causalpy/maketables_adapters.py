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
"""Internal adapters for optional maketables plugin support.

This module intentionally does not import ``maketables``. It provides an internal
adapter interface that BaseExperiment can delegate to when external tools inspect
``__maketables_*`` attributes/methods.
"""

from __future__ import annotations

from typing import Any, Protocol

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from causalpy.constants import HDI_PROB
from causalpy.experiments.model_adapter import ModelAdapter


class MaketablesAdapter(Protocol):
    """Protocol for backend-specific maketables extraction."""

    def stat(self, experiment: Any, key: str) -> Any:
        """Return a single model-level statistic by key.

        Parameters
        ----------
        experiment : Any
            Fitted CausalPy experiment.
        key : str
            Statistic identifier (see :meth:`default_stat_keys`).
        """
        ...

    def vcov_info(self, experiment: Any) -> dict[str, Any]:
        """Return variance-covariance metadata dict.

        Parameters
        ----------
        experiment : Any
            Fitted CausalPy experiment.
        """
        ...

    def stat_labels(self, experiment: Any) -> dict[str, str] | None:
        """Return display labels for statistics.

        Parameters
        ----------
        experiment : Any
            Fitted CausalPy experiment.
        """
        ...

    def default_stat_keys(self, experiment: Any) -> list[str] | None:
        """Return ordered list of default statistic keys.

        Parameters
        ----------
        experiment : Any
            Fitted CausalPy experiment.
        """
        ...


def _safe_observation_count(experiment: Any) -> int | None:
    """Best-effort observation count across experiment classes."""
    design = getattr(experiment, "design", None)
    if isinstance(design, xr.Dataset) and "X" in design:
        return int(design["X"].shape[0])
    for attr in ("data", "datapre", "datapost"):
        obj = getattr(experiment, attr, None)
        if obj is None:
            continue
        if hasattr(obj, "shape"):
            shape = obj.shape
            if shape and len(shape) > 0:
                return int(shape[0])
    return None


def _safe_r2_value(experiment: Any) -> float | None:
    """Best-effort model score extraction without assuming one score format."""
    score_obj = getattr(experiment, "score", None)
    if score_obj is None:
        return None
    try:
        if isinstance(score_obj, pd.Series):
            r2_like = score_obj[[idx for idx in score_obj.index if "r2" in str(idx)]]
            if r2_like.empty:
                return None
            mean_like = r2_like[[idx for idx in r2_like.index if "std" not in str(idx)]]
            target = mean_like if not mean_like.empty else r2_like
            target_numeric = pd.to_numeric(target, errors="coerce")
            mean_value = float(target_numeric.mean(skipna=True))
            if np.isnan(mean_value):
                return None
            return mean_value
        if isinstance(score_obj, (int, float, np.integer, np.floating)):
            return float(score_obj)
    except Exception:
        return None
    return None


def _canonical_frame(
    labels: list[str],
    b: np.ndarray,
    se: np.ndarray,
    p: np.ndarray,
    ci95l: np.ndarray | None = None,
    ci95u: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build canonical maketables coefficient DataFrame."""
    if ci95l is None:
        ci95l = np.full(len(labels), np.nan)
    if ci95u is None:
        ci95u = np.full(len(labels), np.nan)

    frame = pd.DataFrame(
        {
            "b": b,
            "se": se,
            "p": p,
            "t": np.full(len(labels), np.nan),
            "ci95l": ci95l,
            "ci95u": ci95u,
        },
        index=labels,
    )
    frame.index.name = "Coefficient"
    return frame


def _extract_hdi_bounds(
    hdi_result: xr.Dataset | xr.DataArray, var_name: str | None = None
) -> tuple[float, float]:
    """Extract lower/higher values from an ArviZ HDI result."""
    if isinstance(hdi_result, xr.Dataset):
        if var_name is not None and var_name in hdi_result.data_vars:
            hdi_data = hdi_result[var_name]
        else:
            hdi_data = list(hdi_result.data_vars.values())[0]
    else:
        hdi_data = hdi_result

    lower = float(hdi_data.sel(hdi="lower").values)
    upper = float(hdi_data.sel(hdi="higher").values)
    return lower, upper


def _get_maketables_hdi_prob(experiment: Any) -> float:
    """Resolve HDI probability for maketables export.

    Priority:
    1) explicit user override via BaseExperiment.set_maketables_options()
    2) experiment-specific stored value (e.g. staggered_did hdi_prob_)
    3) project-wide default :data:`causalpy.constants.HDI_PROB`
    """
    hdi_prob = getattr(experiment, "_maketables_hdi_prob", None)
    if hdi_prob is None:
        hdi_prob = getattr(experiment, "hdi_prob_", HDI_PROB)
    if hdi_prob is None:
        hdi_prob = HDI_PROB

    try:
        hdi_prob = float(hdi_prob)
    except (TypeError, ValueError) as err:
        msg = f"Invalid HDI probability for maketables export: {hdi_prob!r}"
        raise ValueError(msg) from err

    if not 0 < hdi_prob < 1:
        msg = f"HDI probability must be in (0, 1), got {hdi_prob!r}"
        raise ValueError(msg)
    return hdi_prob


def coefficient_table(experiment: Any) -> pd.DataFrame:
    """Build a maketables coefficient frame from the canonical container.

    Parameters
    ----------
    experiment : Any
        Fitted experiment exposing labels and a model adapter.

    Returns
    -------
    pd.DataFrame
        Coefficient estimates and available uncertainty statistics.
    """
    labels = list(getattr(experiment, "labels", []))
    if not labels:
        raise ValueError("Experiment has no coefficient labels for maketables export.")
    coef_draws = experiment._model_backend.coefficients()

    if (
        "treated_units" in coef_draws.dims
        and int(coef_draws.sizes["treated_units"]) > 1
    ):
        msg = (
            "Ambiguous multi-treated-unit coefficient table for maketables. "
            "Provide an explicit treated unit selection before exporting."
        )
        raise ValueError(msg)
    if "treated_units" in coef_draws.dims:
        coef_draws = coef_draws.isel(treated_units=0)

    try:
        coef_draws = coef_draws.sel(coeffs=labels)
    except KeyError as err:
        raise ValueError(
            "Coefficient labels do not match the fitted model: "
            f"expected {labels!r}, got {list(coef_draws.coords['coeffs'].values)!r}."
        ) from err

    mean = coef_draws.mean(dim=["chain", "draw"]).values.astype(float)
    if coef_draws.sizes["chain"] * coef_draws.sizes["draw"] == 1:
        n = len(labels)
        nans = np.full(n, np.nan)
        return _canonical_frame(labels=labels, b=mean, se=nans, p=nans)

    std = coef_draws.std(dim=["chain", "draw"]).values.astype(float)
    hdi_prob = _get_maketables_hdi_prob(experiment)
    ci95l = np.empty(len(labels), dtype=float)
    ci95u = np.empty(len(labels), dtype=float)
    for i, coeff_name in enumerate(labels):
        coeff_hdi = az.hdi(coef_draws.sel(coeffs=coeff_name), hdi_prob=hdi_prob)
        lower, upper = _extract_hdi_bounds(coeff_hdi)
        ci95l[i] = lower
        ci95u[i] = upper

    # Bayesian p-value semantics are deliberately not inferred from draws.
    p_vals = np.full(len(labels), np.nan)
    return _canonical_frame(
        labels=labels, b=mean, se=std, p=p_vals, ci95l=ci95l, ci95u=ci95u
    )


class PyMCMaketablesAdapter:
    """Adapter for experiments backed by PyMCModel."""

    def stat(self, experiment: Any, key: str) -> Any:
        """Return a single Bayesian model-level statistic by key.

        Parameters
        ----------
        experiment : Any
            Fitted CausalPy experiment with a PyMC model.
        key : str
            Statistic identifier.
        """
        stats: dict[str, Any] = {
            "N": _safe_observation_count(experiment),
            "r2": _safe_r2_value(experiment),
            "model_type": "bayesian",
            "experiment_type": type(experiment).__name__,
            "se_type": "Bayesian posterior",
        }
        return stats.get(key)

    def vcov_info(self, experiment: Any) -> dict[str, Any]:
        """Return Bayesian posterior variance-covariance metadata.

        Parameters
        ----------
        experiment : Any
            Fitted CausalPy experiment with a PyMC model.
        """
        return {"se_type": "Bayesian posterior", "vcov": None}

    def stat_labels(self, experiment: Any) -> dict[str, str] | None:
        """Return display labels for Bayesian model statistics.

        Parameters
        ----------
        experiment : Any
            Fitted CausalPy experiment with a PyMC model.
        """
        return {"N": "N", "r2": "Bayesian R2", "se_type": "SE type"}

    def default_stat_keys(self, experiment: Any) -> list[str] | None:
        """Return ordered list of default statistic keys for Bayesian models.

        Parameters
        ----------
        experiment : Any
            Fitted CausalPy experiment with a PyMC model.
        """
        keys = ["N"]
        if _safe_r2_value(experiment) is not None:
            keys.append("r2")
        return keys


class SklearnMaketablesAdapter:
    """Adapter for experiments backed by sklearn RegressorMixin."""

    def stat(self, experiment: Any, key: str) -> Any:
        """Return a single OLS model-level statistic by key.

        Parameters
        ----------
        experiment : Any
            Fitted CausalPy experiment with an sklearn model.
        key : str
            Statistic identifier.
        """
        stats: dict[str, Any] = {
            "N": _safe_observation_count(experiment),
            "r2": _safe_r2_value(experiment),
            "model_type": "ols",
            "experiment_type": type(experiment).__name__,
            "se_type": "Not available",
        }
        return stats.get(key)

    def vcov_info(self, experiment: Any) -> dict[str, Any]:
        """Return OLS variance-covariance metadata.

        Parameters
        ----------
        experiment : Any
            Fitted CausalPy experiment with an sklearn model.
        """
        return {"se_type": "Not available", "vcov": None}

    def stat_labels(self, experiment: Any) -> dict[str, str] | None:
        """Return display labels for OLS model statistics.

        Parameters
        ----------
        experiment : Any
            Fitted CausalPy experiment with an sklearn model.
        """
        return {"N": "N", "r2": "R2", "se_type": "SE type"}

    def default_stat_keys(self, experiment: Any) -> list[str] | None:
        """Return ordered list of default statistic keys for OLS models.

        Parameters
        ----------
        experiment : Any
            Fitted CausalPy experiment with an sklearn model.
        """
        keys = ["N"]
        if _safe_r2_value(experiment) is not None:
            keys.append("r2")
        return keys


def get_maketables_adapter(model_adapter: ModelAdapter) -> MaketablesAdapter:
    """Return the maketables adapter for a model adapter.

    Parameters
    ----------
    model_adapter : ModelAdapter
        CausalPy backend adapter.
    """
    if model_adapter.kind == "pymc":
        return PyMCMaketablesAdapter()
    if model_adapter.kind == "sklearn":
        return SklearnMaketablesAdapter()
    msg = f"Unsupported model backend for maketables export: {model_adapter.kind!r}"
    raise TypeError(msg)
