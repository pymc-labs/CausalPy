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
"""Boundaries for the interactive fixed-effects call surface."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from causalpy.experiments._results import CausalResult
from causalpy.experiments.interactive_fixed_effects import (
    InteractiveFixedEffectsResult,
    InteractiveFixedEffectsSyntheticControl,
    pre_period_score,
    untreated_counterfactual,
)
from causalpy.utils import _bayesian_r2_score


def _panel(post_shift: float = 0.0) -> pd.DataFrame:
    index = [0, 1, 2, 3]
    frame = pd.DataFrame(
        {
            "donor_a": [1.0, 2.0, 3.0, 4.0],
            "donor_b": [0.5, 0.5, 0.5, 0.5],
            "treated_a": [1.0, 1.5, 9.0, 9.5],
            "treated_b": [2.0, 2.5, 8.0, 8.5],
        },
        index=index,
    )
    frame.loc[frame.index >= 2, ["treated_a", "treated_b"]] += post_shift
    return frame


def _components(max_factors: int = 2) -> dict[str, xr.DataArray]:
    chain = [0]
    draw = [0, 1]
    times = [0, 1, 2, 3]
    treated = ["treated_a", "treated_b"]
    factors = [f"f{i}" for i in range(max_factors)]
    unit = xr.DataArray(
        np.array([[[1.0, 3.0], [1.5, 3.5]]]),
        dims=("chain", "draw", "treated_units"),
        coords={"chain": chain, "draw": draw, "treated_units": treated},
    )
    time = xr.DataArray(
        np.array([[[0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.3]]]),
        dims=("chain", "draw", "obs_ind"),
        coords={"chain": chain, "draw": draw, "obs_ind": times},
    )
    loadings = xr.DataArray(
        np.ones((1, 2, 2, max_factors)),
        dims=("chain", "draw", "treated_units", "factors"),
        coords={
            "chain": chain,
            "draw": draw,
            "treated_units": treated,
            "factors": factors,
        },
    )
    loadings = loadings.copy()
    loadings.loc[{"treated_units": "treated_b", "factors": "f1"}] = 0.0
    factor_draws = xr.DataArray(
        np.zeros((1, 2, 4, max_factors)),
        dims=("chain", "draw", "obs_ind", "factors"),
        coords={
            "chain": chain,
            "draw": draw,
            "obs_ind": times,
            "factors": factors,
        },
    )
    factor_draws = factor_draws.copy()
    factor_draws.loc[{"factors": "f0"}] = 0.5
    factor_draws.loc[{"factors": "f1", "draw": 1}] = 2.0
    scales = xr.DataArray(
        np.array([[[0.4, 0.01], [0.2, 1e-6]]]),
        dims=("chain", "draw", "factors"),
        coords={"chain": chain, "draw": draw, "factors": factors},
    )
    return {
        "unit_intercept": unit,
        "time_intercept": time,
        "loadings": loadings,
        "factors": factor_draws,
        "shrinkage_state": scales,
    }


def _experiment(post_shift: float = 0.0) -> InteractiveFixedEffectsSyntheticControl:
    return InteractiveFixedEffectsSyntheticControl(
        _panel(post_shift),
        2,
        ["donor_a", "donor_b"],
        ["treated_a", "treated_b"],
        max_factors=2,
    )


def test_counterfactual_keeps_both_intercepts() -> None:
    parts = _components()
    got = untreated_counterfactual(
        parts["unit_intercept"],
        parts["time_intercept"],
        parts["loadings"],
        parts["factors"],
    )
    product = (parts["loadings"] * parts["factors"]).sum("factors")
    expected = parts["unit_intercept"] + parts["time_intercept"] + product
    xr.testing.assert_allclose(got, expected)
    assert not np.allclose(got, product)
    assert not np.allclose(got, parts["unit_intercept"] + product)
    assert not np.allclose(got, parts["unit_intercept"] + parts["time_intercept"])


def test_constructor_has_max_factors_and_not_n_factors() -> None:
    signature = inspect.signature(InteractiveFixedEffectsSyntheticControl)
    assert "max_factors" in signature.parameters
    assert "n_factors" not in signature.parameters
    experiment = _experiment()
    assert experiment.max_factors == 2
    assert not hasattr(experiment, "n_factors")
    assert not hasattr(experiment, "effective_rank")


def test_result_is_causal_result_subclass_with_scale_draws() -> None:
    experiment = _experiment()
    result = experiment.compose_result(**_components())
    assert isinstance(result, CausalResult)
    assert isinstance(result, InteractiveFixedEffectsResult)
    assert result.shrinkage_state.sizes["factors"] == experiment.max_factors
    assert result.shrinkage_state.dtype.kind == "f"
    assert not hasattr(result, "n_factors")
    assert not hasattr(result, "effective_rank")
    assert set(CausalResult.__dataclass_fields__) <= set(
        InteractiveFixedEffectsResult.__dataclass_fields__
    )
    assert "shrinkage_state" not in CausalResult.__dataclass_fields__


def test_impact_uses_observed_minus_counterfactual_and_post_outcomes_only() -> None:
    parts = _components()
    baseline = _experiment().compose_result(**parts)
    shifted = _experiment(post_shift=10.0).compose_result(**parts)
    xr.testing.assert_allclose(baseline.predictions_pre, shifted.predictions_pre)
    xr.testing.assert_allclose(baseline.predictions_post, shifted.predictions_post)
    xr.testing.assert_allclose(baseline.shrinkage_state, shifted.shrinkage_state)
    pd.testing.assert_series_equal(baseline.score, shifted.score)
    assert not np.allclose(baseline.impact_post, shifted.impact_post)
    xr.testing.assert_allclose(shifted.impact_post, baseline.impact_post + 10.0)
    observed_post = _experiment().panel._treated_post()
    manual = (observed_post - baseline.predictions_post).transpose(
        ..., "obs_ind", "treated_units"
    )
    xr.testing.assert_allclose(baseline.impact_post, manual)


def test_score_matches_pre_period_bayesian_r2_only() -> None:
    experiment = _experiment()
    result = experiment.compose_result(**_components())
    treated_pre = experiment.panel.treated_pre
    stacked = result.predictions_pre.stack(sample=("chain", "draw"))
    for i, unit in enumerate(treated_pre.coords["treated_units"].values):
        unit_mu = stacked.sel(treated_units=unit).transpose("sample", "obs_ind")
        expected = _bayesian_r2_score(
            treated_pre.sel(treated_units=unit).to_numpy(),
            unit_mu.to_numpy(),
        )
        assert result.score[f"unit_{i}_r2"] == pytest.approx(float(expected["r2"]))
        assert result.score[f"unit_{i}_r2_std"] == pytest.approx(
            float(expected["r2_std"])
        )
    both = xr.concat([result.predictions_pre, result.predictions_post], dim="obs_ind")
    both_y = xr.concat([treated_pre, experiment.panel._treated_post()], dim="obs_ind")
    leaked = pre_period_score(both, both_y)
    assert not np.allclose(result.score.to_numpy(), leaked.to_numpy())


def test_missing_used_values_are_rejected() -> None:
    frame = _panel()
    frame.loc[1, "treated_a"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        InteractiveFixedEffectsSyntheticControl(
            frame,
            2,
            ["donor_a", "donor_b"],
            ["treated_a", "treated_b"],
            max_factors=2,
        )


def test_integer_shrinkage_state_is_not_a_rank() -> None:
    experiment = _experiment()
    parts = _components()
    parts["shrinkage_state"] = xr.DataArray(
        np.array([[[1, 0], [1, 0]]]),
        dims=("chain", "draw", "factors"),
    )
    with pytest.raises(TypeError, match="integer rank"):
        experiment.compose_result(**parts)


def test_shrinkage_state_length_is_the_upper_bound_not_a_selected_rank() -> None:
    experiment = _experiment()
    parts = _components()
    parts["shrinkage_state"] = parts["shrinkage_state"].isel(factors=slice(0, 1))
    parts["loadings"] = parts["loadings"].isel(factors=slice(0, 1))
    parts["factors"] = parts["factors"].isel(factors=slice(0, 1))
    with pytest.raises(ValueError, match="max_factors"):
        experiment.compose_result(**parts)
