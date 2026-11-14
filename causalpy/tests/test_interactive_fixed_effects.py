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
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from causalpy.experiments._results import CausalResult
from causalpy.experiments.interactive_fixed_effects import (
    _RESIDUAL_FLOOR_FRACTION,
    InteractiveFixedEffectsResult,
    InteractiveFixedEffectsSyntheticControl,
    _as_dataset,
    _control_initvals,
    _control_outcome_spread,
    _scale_log_posterior,
    _updated_residual_scale,
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
    """Treated pre-period updates the one shared scale, not factors or intercepts.

    A unit observation scale fails because ``sigma`` is not an ancestor of the
    control likelihood. A second residual scale fails the free-RV sets: the
    control graph has the only one, and the conditional graph has none. Draws
    held fixed from the control panel fail the comparison of the scale the
    loading likelihood uses. That comparison is the updated draws, not a
    second short chain.
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

    fit(baseline_frame)
    fit(shifted)
    assert len(captured) == 4
    base_control, base_conditional = captured[0], captured[1]
    moved_control, moved_conditional = captured[2], captured[3]

    def assert_split(control_model, conditional_model, frame) -> None:
        assert {rv.name for rv in control_model.free_RVs} == {
            "mgp_delta",
            "sigma_raw",
            "control_unit_intercept",
            "time_intercept",
            "shared_factors",
            "control_loadings",
        }
        sigma = control_model["sigma"]
        assert sigma in set(ancestors([control_model["control_y"]]))
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
            "unit_intercept",
            "loadings",
        }
        assert "sigma" not in conditional_model.named_vars
        treated = conditional_model["treated_pre"]
        treated_ancestors = set(ancestors([treated]))
        assert conditional_model["residual_scale"] in treated_ancestors
        assert [
            rv.name
            for rv in conditional_model.free_RVs
            if rv in treated_ancestors and rv.name not in {"unit_intercept", "loadings"}
        ] == []
        assert {rv.name for rv in conditional_model.observed_RVs} == {"treated_pre"}
        assert "shared_factors" not in conditional_model.named_vars
        assert "time_intercept" not in conditional_model.named_vars
        scale = np.asarray(conditional_model["residual_scale"].eval())
        assert scale.ndim == 1
        assert not np.allclose(scale, 1.0)
        treated_observed = conditional_model.rvs_to_values[treated].eval()
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
    xr.testing.assert_allclose(base_posterior["sigma"], moved_posterior["sigma"])
    base_scale = np.asarray(base_conditional[0]["residual_scale"].eval())
    moved_scale = np.asarray(moved_conditional[0]["residual_scale"].eval())
    assert float(np.mean(moved_scale)) > float(np.mean(base_scale)) + 1.0


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


def test_constructor_rejects_a_non_positive_integer_factor_bound() -> None:
    """A factor bound is an integer count, not a rank, a bool, or a float."""
    frame = _panel()
    donors = ["donor_a", "donor_b"]
    treated = ["treated_a"]
    with pytest.raises(TypeError, match="not a selected rank"):
        InteractiveFixedEffectsSyntheticControl(
            frame, 2, donors, treated, max_factors=True
        )
    with pytest.raises(TypeError, match="not a selected rank"):
        InteractiveFixedEffectsSyntheticControl(
            frame, 2, donors, treated, max_factors=1.5
        )
    with pytest.raises(ValueError, match="at least 1"):
        InteractiveFixedEffectsSyntheticControl(
            frame, 2, donors, treated, max_factors=0
        )
    with pytest.raises(TypeError, match="DataFrame"):
        InteractiveFixedEffectsSyntheticControl(frame.to_numpy(), 2, donors, treated)


def test_constructor_rejects_empty_overlapping_or_missing_units() -> None:
    """Donors and treated names must be non-empty, disjoint, and present."""
    frame = _panel()
    with pytest.raises(ValueError, match="one or more treated units"):
        InteractiveFixedEffectsSyntheticControl(frame, 2, [], ["treated_a"])
    with pytest.raises(ValueError, match="one or more treated units"):
        InteractiveFixedEffectsSyntheticControl(frame, 2, ["donor_a"], [])
    with pytest.raises(ValueError, match="overlap"):
        InteractiveFixedEffectsSyntheticControl(
            frame, 2, ["donor_a", "donor_b"], ["donor_a"]
        )
    with pytest.raises(KeyError, match="missing"):
        InteractiveFixedEffectsSyntheticControl(frame, 2, ["donor_a"], ["absent"])


def test_constructor_rejects_a_panel_missing_either_period() -> None:
    """The treatment time must leave both a pre-period and a post-period."""
    frame = _panel()
    donors = ["donor_a", "donor_b"]
    treated = ["treated_a"]
    with pytest.raises(ValueError, match="both a pre-period and a post-period"):
        InteractiveFixedEffectsSyntheticControl(frame, 0, donors, treated)
    with pytest.raises(ValueError, match="both a pre-period and a post-period"):
        InteractiveFixedEffectsSyntheticControl(frame, 4, donors, treated)


def test_unit_names_follow_construction_order() -> None:
    """Donor and treated properties keep the order supplied at construction."""
    experiment = InteractiveFixedEffectsSyntheticControl(
        _panel(),
        2,
        ["donor_b", "donor_a"],
        ["treated_b", "treated_a"],
    )
    assert experiment.control_units == ("donor_b", "donor_a")
    assert experiment.treated_units == ("treated_b", "treated_a")


def test_build_reuses_the_control_graph() -> None:
    """A second build returns the same graph instead of sampling a new one."""
    experiment = _experiment()
    first = experiment.build()
    second = experiment.build()
    assert first is experiment
    assert second._control_model is first._control_model


def test_plot_requires_a_result_and_honours_show() -> None:
    """plot draws the composed posterior and calls show only when asked."""
    experiment = _experiment()
    with pytest.raises(RuntimeError, match=r"fit\(\)"):
        experiment.plot(show=False)
    experiment.result = experiment.compose_result(**_components())
    with patch("causalpy.experiments.interactive_fixed_effects.plt.show") as show:
        figure, axes = experiment.plot(
            show=False, treated_unit="treated_b", round_to=2, plot_predictors=True
        )
        show.assert_not_called()
        assert isinstance(figure, plt.Figure)
        assert len(axes) == 3
        plt.close(figure)
        figure, _ = experiment.plot(show=True)
        show.assert_called_once()
        plt.close(figure)


def test_compose_result_rejects_a_bad_shrinkage_or_factor_axis() -> None:
    """Component draws must be scales on the shared factor axis, not a rank."""
    experiment = _experiment()
    parts = _components()
    with pytest.raises(TypeError, match="DataArray of scales"):
        experiment.compose_result(**{**parts, "shrinkage_state": [0.2, 0.3]})
    wrong_dims = parts["shrinkage_state"].rename(factors="rank")
    with pytest.raises(ValueError, match="shrinkage_state dims"):
        experiment.compose_result(**{**parts, "shrinkage_state": wrong_dims})
    short = parts["shrinkage_state"].isel(factors=slice(0, 1))
    with pytest.raises(ValueError, match="one scale per factor"):
        experiment.compose_result(**{**parts, "shrinkage_state": short})
    loadings = parts["loadings"].rename(factors="rank")
    with pytest.raises(ValueError, match="loadings must have a 'factors' dimension"):
        experiment.compose_result(**{**parts, "loadings": loadings})
    short_loadings = parts["loadings"].isel(factors=slice(0, 1))
    with pytest.raises(ValueError, match="factor axis must have length"):
        experiment.compose_result(**{**parts, "loadings": short_loadings})
    factors = parts["factors"].isel(obs_ind=0, drop=True)
    with pytest.raises(ValueError, match="shared fit"):
        experiment.compose_result(**{**parts, "factors": factors})


def test_control_initvals_skip_the_svd_fill_when_the_factor_axis_is_empty() -> None:
    """A zero-length factor axis leaves the SVD start at zero."""

    class _Factors:
        def eval(self) -> int:
            return 0

    class _Model:
        dim_lengths = {"factors": _Factors()}

    control_y = xr.DataArray(
        np.ones((3, 2)),
        dims=("obs_ind", "control_units"),
        coords={"obs_ind": [0, 1, 2], "control_units": ["a", "b"]},
    )
    start = _control_initvals(_Model(), control_y)
    assert start["shared_factors"].shape == (3, 0)
    assert start["control_loadings"].shape == (2, 0)
    assert start["mgp_delta"].shape == (0,)


def test_scale_posterior_rejects_a_non_positive_covariance() -> None:
    """A singular or non-finite covariance does not enter the scale update."""
    centered = np.ones((2, 1))
    singular = _scale_log_posterior(
        0.0,
        gram=-np.eye(2),
        centered=centered,
        control_rss=1.0,
        n_control=1,
        spread=1.0,
    )
    with np.errstate(invalid="ignore"):
        missing = _scale_log_posterior(
            0.0,
            gram=np.full((2, 2), np.nan),
            centered=centered,
            control_rss=1.0,
            n_control=1,
            spread=1.0,
        )
    assert singular == -1e300
    assert missing == -1e300


def test_scale_posterior_rejects_a_sub_floor_draw() -> None:
    """A residual scale below the control-panel floor is not in the posterior."""
    above = _scale_log_posterior(
        float(np.log(0.2)),
        gram=np.eye(2),
        centered=np.zeros((2, 1)),
        control_rss=1.0,
        n_control=1,
        spread=1.0,
    )
    below = _scale_log_posterior(
        float(np.log(0.01)),
        gram=np.eye(2),
        centered=np.zeros((2, 1)),
        control_rss=1.0,
        n_control=1,
        spread=1.0,
    )
    assert np.isfinite(above)
    assert below == -1e300


def test_control_residual_scale_cannot_reach_an_exact_fit() -> None:
    """A noise-free control panel cannot put the residual scale on an exact-fit ridge."""
    panel = latent_factor
    experiment = InteractiveFixedEffectsSyntheticControl(
        panel.data,
        panel.treatment_time,
        panel.control_units,
        panel.treated_units,
        max_factors=2,
    )
    experiment.build()
    model = experiment._control_model
    assert model is not None
    floor = _RESIDUAL_FLOOR_FRACTION * _control_outcome_spread(
        experiment._control_outcomes()
    )
    sigma = float(np.asarray(model["sigma"].eval({model["sigma_raw"]: np.array(0.0)})))
    assert sigma == pytest.approx(floor)
    assert sigma > 0.0


def test_non_finite_treated_pre_period_does_not_move_the_scale() -> None:
    """A non-finite gradient stops the update and keeps the control-panel scale."""
    sigma = np.array([1.0, 2.0])
    updated = _updated_residual_scale(
        sigma=sigma,
        control_rss=np.array([1.0, 1.0]),
        n_control=2,
        factors_pre=np.zeros((2, 3, 1)),
        shrinkage=np.ones((2, 1)),
        treated_centered=np.full((2, 3, 1), np.nan),
        spread=1.0,
    )
    np.testing.assert_allclose(updated, sigma)


def test_flat_scale_posterior_uses_the_gradient_sign() -> None:
    """A nearly flat posterior still takes a finite step, and stops at the floor."""
    updated = _updated_residual_scale(
        sigma=np.array([1.0, 1e-8]),
        control_rss=np.array([0.0, 0.0]),
        n_control=1,
        factors_pre=np.zeros((2, 2, 1)),
        shrinkage=np.ones((2, 1)),
        treated_centered=np.zeros((2, 2, 1)),
        spread=1.0,
    )
    assert np.all(np.isfinite(updated))
    assert updated[0] < 1.0
    assert np.all(updated >= _RESIDUAL_FLOOR_FRACTION)


def test_failed_line_search_keeps_the_current_scale(monkeypatch) -> None:
    """If no trial fraction improves the posterior, the scale draw stays put."""
    start = float(np.log(1.0))

    def _decline(log_scale: float, **kwargs: object) -> float:
        if abs(log_scale - start) <= 1e-3:
            return 0.0
        return -1.0

    monkeypatch.setattr(
        "causalpy.experiments.interactive_fixed_effects._scale_log_posterior",
        _decline,
    )
    updated = _updated_residual_scale(
        sigma=np.array([1.0]),
        control_rss=np.array([1.0]),
        n_control=1,
        factors_pre=np.zeros((1, 2, 1)),
        shrinkage=np.ones((1, 1)),
        treated_centered=np.zeros((1, 2, 1)),
        spread=1.0,
    )
    assert updated[0] == pytest.approx(1.0)


def test_non_finite_curvature_keeps_the_scale(monkeypatch) -> None:
    """A finite gradient with a non-finite curvature does not move the scale."""
    start = float(np.log(1.0))

    def _curved(log_scale: float, **kwargs: object) -> float:
        if abs(log_scale - start) < 1e-8:
            return np.nan
        return 0.0

    monkeypatch.setattr(
        "causalpy.experiments.interactive_fixed_effects._scale_log_posterior",
        _curved,
    )
    updated = _updated_residual_scale(
        sigma=np.array([1.0]),
        control_rss=np.array([1.0]),
        n_control=1,
        factors_pre=np.zeros((1, 2, 1)),
        shrinkage=np.ones((1, 1)),
        treated_centered=np.zeros((1, 2, 1)),
        spread=1.0,
    )
    assert updated[0] == pytest.approx(1.0)


def test_control_posterior_coercion_accepts_a_mapping_and_rejects_other_objects() -> (
    None
):
    """A posterior holder is a Dataset, a mapping of variables, or an error."""

    class _Mapping(dict):
        @property
        def data_vars(self) -> dict:
            return self

    class _Holder:
        def __init__(self, dataset: object) -> None:
            self.dataset = dataset

    got = _as_dataset(_Holder(_Mapping(x=("t", [1.0, 2.0]))))
    assert got["x"].values.tolist() == [1.0, 2.0]
    with pytest.raises(TypeError, match="control posterior"):
        _as_dataset(object())
    with pytest.raises(TypeError, match="control posterior"):
        _as_dataset(_Holder(object()))
