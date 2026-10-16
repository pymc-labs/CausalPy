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
"""Boundaries for interactive fixed-effects synthetic control."""

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
    untreated_counterfactual,
)
from causalpy.tests.known_truth_panels import donor_shock, latent_factor

_FIT = {"draws": 8, "tune": 20, "chains": 1, "random_seed": 7, "progressbar": False}


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
    factor_draws = xr.DataArray(
        np.zeros((1, 2, 4, max_factors)),
        dims=("chain", "draw", "obs_ind", "factors"),
        coords={"chain": chain, "draw": draw, "obs_ind": times, "factors": factors},
    )
    factor_draws = factor_draws.copy()
    factor_draws.loc[{"factors": "f0"}] = 0.5
    scales = xr.DataArray(
        np.array([[[0.4, 0.01], [0.2, 1e-6]]])[:, :, :max_factors],
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


def test_constructor_has_max_factors_and_not_n_factors() -> None:
    signature = inspect.signature(InteractiveFixedEffectsSyntheticControl)
    assert "max_factors" in signature.parameters
    assert "n_factors" not in signature.parameters
    experiment = _experiment()
    assert not hasattr(experiment, "n_factors")
    assert not hasattr(experiment, "effective_rank")


def test_result_is_causal_result_subclass_with_scale_draws() -> None:
    experiment = _experiment()
    result = experiment.compose_result(**_components())
    assert isinstance(result, InteractiveFixedEffectsResult)
    assert isinstance(result, CausalResult)
    assert result.shrinkage_state.sizes["factors"] == 2
    assert "shrinkage_state" not in CausalResult.__dataclass_fields__


def test_impact_uses_observed_minus_counterfactual() -> None:
    experiment = _experiment()
    shifted = _experiment(post_shift=4.0)
    parts = _components()
    baseline = experiment.compose_result(**parts)
    moved = shifted.compose_result(**parts)
    xr.testing.assert_allclose(baseline.predictions_post, moved.predictions_post)
    assert not np.allclose(baseline.impact_post, moved.impact_post)


def test_missing_used_values_are_rejected() -> None:
    frame = _panel()
    frame.loc[0, "donor_a"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        InteractiveFixedEffectsSyntheticControl(
            frame, 2, ["donor_a", "donor_b"], ["treated_a"], max_factors=1
        )


def test_integer_shrinkage_state_is_not_a_rank() -> None:
    experiment = _experiment()
    parts = _components()
    parts["shrinkage_state"] = parts["shrinkage_state"].astype(int)
    with pytest.raises(TypeError, match="integer rank"):
        experiment.compose_result(**parts)


def test_control_graph_excludes_treated_post_outcomes() -> None:
    frame = _panel()
    frame.loc[frame.index >= 2, "treated_a"] = 12345.0
    experiment = InteractiveFixedEffectsSyntheticControl(
        frame, 2, ["donor_a", "donor_b"], ["treated_a", "treated_b"], max_factors=2
    )
    experiment.build()
    model = experiment._control_model
    assert model is not None
    observed = model.rvs_to_values[model["control_y"]].eval()
    assert observed.shape == (4, 2)
    assert not np.any(observed == 12345.0)
    np.testing.assert_allclose(observed[:, 0], frame["donor_a"].to_numpy())


def test_treated_pre_updates_scale_not_factors_or_time_intercepts(monkeypatch) -> None:
    """Treated pre-period moves the shared scale, not factors or time intercepts.

    Captures both ``pm.sample`` calls inside ``fit``: the control-panel sample
    and the conditional sample. A joint factor update, or a residual scale held
    fixed from the control panel, fails.
    """
    from pytensor.graph.traversal import ancestors

    import causalpy.experiments.interactive_fixed_effects as ife

    panel = latent_factor
    baseline_frame = panel.data.copy()
    shifted = baseline_frame.copy()
    pre = shifted.index < panel.treatment_time
    shifted.loc[pre, list(panel.treated_units)] += 25.0
    shifted.loc[
        shifted.index[shifted.index < panel.treatment_time][0], panel.treated_units[0]
    ] = 12345.0
    captured: list[tuple] = []
    real_sample = ife.pm.sample

    def spy(*args, **kwargs):
        idata = real_sample(*args, **kwargs)
        captured.append((kwargs["model"], idata))
        return idata

    monkeypatch.setattr(ife.pm, "sample", spy)

    def fit(frame):
        return ife.InteractiveFixedEffectsSyntheticControl(
            frame,
            panel.treatment_time,
            panel.control_units,
            panel.treated_units,
            max_factors=3,
        ).fit(**_FIT)

    baseline = fit(baseline_frame)
    moved = fit(shifted)
    assert len(captured) == 4
    base_control, base_conditional = captured[0], captured[1]
    moved_control, moved_conditional = captured[2], captured[3]

    def assert_split(control_model, conditional_model, frame) -> None:
        assert {rv.name for rv in control_model.free_RVs} == {
            "mgp_delta",
            "sigma",
            "control_unit_intercept",
            "time_intercept",
            "shared_factors",
            "control_loadings",
        }
        assert {rv.name for rv in control_model.observed_RVs} == {"control_y"}
        control_observed = control_model.rvs_to_values[
            control_model["control_y"]
        ].eval()
        assert control_observed.shape[1] == len(panel.control_units)
        assert not np.any(control_observed == 12345.0)
        treated_pre = frame.loc[
            frame.index < panel.treatment_time, list(panel.treated_units)
        ]
        assert not np.any(np.isin(control_observed, treated_pre.to_numpy()))
        assert {rv.name for rv in conditional_model.free_RVs} == {
            "sigma",
            "unit_intercept",
            "loadings",
        }
        assert {rv.name for rv in conditional_model.observed_RVs} == {
            "control_resid",
            "treated_pre",
        }
        assert "shared_factors" not in conditional_model.named_vars
        assert "time_intercept" not in conditional_model.named_vars

        sigma = conditional_model["sigma"]
        assert sigma in set(ancestors([conditional_model["treated_pre"]]))
        treated_observed = conditional_model.rvs_to_values[
            conditional_model["treated_pre"]
        ].eval()
        if frame is baseline_frame:
            assert not np.any(treated_observed == 12345.0)
        else:
            assert np.any(treated_observed == 12345.0)

    assert_split(base_control[0], base_conditional[0], baseline_frame)
    assert_split(moved_control[0], moved_conditional[0], shifted)

    base_posterior = ife._as_dataset(base_control[1])
    moved_posterior = ife._as_dataset(moved_control[1])
    xr.testing.assert_allclose(
        base_posterior["shared_factors"], moved_posterior["shared_factors"]
    )
    xr.testing.assert_allclose(
        base_posterior["time_intercept"], moved_posterior["time_intercept"]
    )
    xr.testing.assert_allclose(
        base_posterior["shrinkage_state"], moved_posterior["shrinkage_state"]
    )
    base_sigma = ife._as_dataset(base_conditional[1])["sigma"]
    moved_sigma = ife._as_dataset(moved_conditional[1])["sigma"]
    assert not np.allclose(base_sigma, moved_sigma)
    assert float(moved_sigma.mean()) > float(base_sigma.mean())
    assert baseline.result is not None and moved.result is not None
    assert not np.allclose(
        baseline.result.predictions_pre, moved.result.predictions_pre
    )


def test_known_truth_smoke_and_post_outcomes_do_not_move_the_trajectory() -> None:
    panel = latent_factor
    data = panel.data.copy()
    data["treated_b"] = panel.untreated["treated"]
    data.loc[data.index >= panel.treatment_time, "treated_b"] += 5.0
    baseline = InteractiveFixedEffectsSyntheticControl(
        data,
        panel.treatment_time,
        panel.control_units,
        ("treated", "treated_b"),
        max_factors=3,
    ).fit(**_FIT)
    shifted = data.copy()
    shifted.loc[shifted.index >= panel.treatment_time, ["treated", "treated_b"]] += 9.0
    moved = InteractiveFixedEffectsSyntheticControl(
        shifted,
        panel.treatment_time,
        panel.control_units,
        ("treated", "treated_b"),
        max_factors=3,
    ).fit(**_FIT)
    assert baseline.result is not None
    assert moved.result is not None
    xr.testing.assert_allclose(
        baseline.result.predictions_post, moved.result.predictions_post
    )
    xr.testing.assert_allclose(
        baseline.result.predictions_pre, moved.result.predictions_pre
    )
    xr.testing.assert_allclose(
        baseline.result.shrinkage_state, moved.result.shrinkage_state
    )
    assert not np.allclose(baseline.result.impact_post, moved.result.impact_post)
    assert baseline.result.shrinkage_state.sizes["factors"] == 3
    assert baseline.result.predictions_post.sizes["treated_units"] == 2
    assert set(baseline.result.score.index) == {
        "unit_0_r2",
        "unit_0_r2_std",
        "unit_1_r2",
        "unit_1_r2_std",
    }


def test_donor_order_changes_observed_columns_not_the_treated_post_cells() -> None:
    panel = latent_factor
    forward = InteractiveFixedEffectsSyntheticControl(
        panel.data,
        panel.treatment_time,
        panel.control_units,
        panel.treated_units,
        max_factors=2,
    )
    reversed_units = tuple(reversed(panel.control_units))
    backward = InteractiveFixedEffectsSyntheticControl(
        panel.data,
        panel.treatment_time,
        reversed_units,
        panel.treated_units,
        max_factors=2,
    )
    forward.build()
    backward.build()
    forward_model = forward._control_model
    backward_model = backward._control_model
    assert forward_model is not None and backward_model is not None
    forward_y = forward_model.rvs_to_values[forward_model["control_y"]].eval()
    backward_y = backward_model.rvs_to_values[backward_model["control_y"]].eval()
    np.testing.assert_allclose(forward_y, np.flip(backward_y, axis=1))
    treated_post = panel.data.loc[
        panel.data.index >= panel.treatment_time, "treated"
    ].iloc[0]
    assert treated_post not in forward_y


@pytest.mark.nightly
def test_nightly_recovers_latent_factor_path() -> None:
    panel = latent_factor
    result = (
        InteractiveFixedEffectsSyntheticControl(
            panel.data,
            panel.treatment_time,
            panel.control_units,
            panel.treated_units,
            max_factors=4,
        )
        .fit(draws=40, tune=80, chains=1, random_seed=3, progressbar=False)
        .result
    )
    assert result is not None
    mean = result.predictions_post.mean(("chain", "draw")).sel(treated_units="treated")
    truth = panel.untreated.loc[
        panel.untreated.index >= panel.treatment_time, "treated"
    ]
    assert np.max(np.abs(mean.values - truth.to_numpy())) < 1.0


@pytest.mark.nightly
def test_nightly_misspecified_donor_shock_still_marginalizes_shrinkage() -> None:
    panel = donor_shock
    low = InteractiveFixedEffectsSyntheticControl(
        panel.data,
        panel.treatment_time,
        panel.control_units,
        panel.treated_units,
        max_factors=2,
    ).fit(draws=20, tune=40, chains=1, random_seed=5, progressbar=False)
    high = InteractiveFixedEffectsSyntheticControl(
        panel.data,
        panel.treatment_time,
        panel.control_units,
        panel.treated_units,
        max_factors=4,
    ).fit(draws=20, tune=40, chains=1, random_seed=5, progressbar=False)
    assert low.result is not None and high.result is not None
    assert low.result.shrinkage_state.sizes["factors"] == 2
    assert high.result.shrinkage_state.sizes["factors"] == 4
    assert low.result.predictions_post.dims == high.result.predictions_post.dims
