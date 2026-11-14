#   Copyright 2025 - 2026 The PyMC Labs Developers
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
"""Known-truth panels are exact, noiseless, and not a public API."""

import importlib

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

import causalpy
import causalpy.data
import causalpy.experiments
from causalpy.experiments._panel_counterfactual import WidePanel
from causalpy.tests import known_truth_panels as panels

_PANELS = (
    panels.exact_convex_hull,
    panels.systematic_inexact,
    panels.latent_factor,
    panels.donor_shock,
    panels.placebo,
)
_FIELDS = {
    "data",
    "untreated",
    "effect",
    "treatment_time",
    "control_units",
    "treated_units",
}


def _pre(frame: pd.DataFrame, treatment_time: int) -> pd.DataFrame:
    return frame.loc[frame.index < treatment_time]


def _post(frame: pd.DataFrame, treatment_time: int) -> pd.DataFrame:
    return frame.loc[frame.index >= treatment_time]


def _simplex_weights(controls: pd.DataFrame, target: pd.Series):
    width = controls.shape[1]
    design = np.vstack([controls.to_numpy(), np.ones(width)])
    target_eq = np.concatenate([target.to_numpy(), [1.0]])
    return linprog(
        np.zeros(width),
        A_eq=design,
        b_eq=target_eq,
        bounds=(0, None),
        method="highs",
    )


def _residual(controls: pd.DataFrame, target: pd.Series, weights: np.ndarray) -> float:
    fitted = controls.to_numpy() @ weights
    return float(np.max(np.abs(fitted - target.to_numpy())))


@pytest.mark.parametrize("panel", _PANELS)
def test_identity_and_frozen_fields(panel) -> None:
    """``data`` equals ``untreated + effect``, and the value exposes only the six fields."""
    assert set(panel.__dataclass_fields__) == _FIELDS
    assert panel.data.equals(panel.untreated + panel.effect)
    assert np.array_equal(
        panel.data.to_numpy(),
        (panel.untreated + panel.effect).to_numpy(),
    )
    assert isinstance(panel.treatment_time, int)
    assert not isinstance(panel.treatment_time, bool)
    assert isinstance(panel.control_units, tuple)
    assert isinstance(panel.treated_units, tuple)
    assert len(panel.treated_units) == 1
    assert len(panel.control_units) >= 2
    assert all(
        isinstance(unit, str) for unit in panel.control_units + panel.treated_units
    )
    units = panel.control_units + panel.treated_units
    assert set(panel.data.columns) == set(units)
    assert (
        list(panel.data.columns)
        == list(panel.untreated.columns)
        == list(panel.effect.columns)
    )
    assert panel.data.index.equals(panel.untreated.index)
    assert panel.data.index.equals(panel.effect.index)
    assert pd.api.types.is_integer_dtype(panel.data.index)
    assert panel.treatment_time in panel.data.index
    assert (panel.effect[list(panel.control_units)].to_numpy() == 0).all()
    pre_effect = _pre(panel.effect, panel.treatment_time)
    assert (pre_effect.to_numpy() == 0).all()
    split = WidePanel.from_frame(
        panel.data,
        panel.treatment_time,
        panel.control_units,
        panel.treated_units,
    )
    assert panel.treatment_time in split.post.index
    assert panel.treatment_time not in split.pre.index
    assert split.pre.index.equals(_pre(panel.data, panel.treatment_time).index)
    assert split.post.index.equals(_post(panel.data, panel.treatment_time).index)


def test_exact_convex_hull_on_data_and_untreated() -> None:
    """Treated pre-period path is an exact convex combination, and the post effect is nonzero."""
    panel = panels.exact_convex_hull
    treated = panel.treated_units[0]
    controls = list(panel.control_units)
    for frame in (panel.untreated, panel.data):
        pre = _pre(frame, panel.treatment_time)
        fit = _simplex_weights(pre[controls], pre[treated])
        assert fit.success
        assert _residual(pre[controls], pre[treated], fit.x) == 0.0
        assert np.all(fit.x >= 0)
        assert fit.x.sum() == pytest.approx(1.0)
    post_effect = _post(panel.effect, panel.treatment_time)[treated]
    assert (post_effect.to_numpy() != 0).any()


def test_systematic_inexact_has_no_simplex_weights() -> None:
    """No nonnegative weights that sum to 1 reproduce the treated pre-period path."""
    panel = panels.systematic_inexact
    treated = panel.treated_units[0]
    controls = list(panel.control_units)
    for frame in (panel.untreated, panel.data):
        pre = _pre(frame, panel.treatment_time)
        fit = _simplex_weights(pre[controls], pre[treated])
        assert not fit.success
    post_effect = _post(panel.effect, panel.treatment_time)[treated]
    assert (post_effect.to_numpy() != 0).any()


def test_latent_factor_is_exact_rank_two() -> None:
    """Untreated outcomes are an exact rank-2 factor product, not a convex combination."""
    panel = panels.latent_factor
    treated = panel.treated_units[0]
    controls = list(panel.control_units)
    untreated = panel.untreated.to_numpy()
    singular = np.linalg.svd(untreated, compute_uv=False)
    assert np.linalg.matrix_rank(untreated) == 2
    assert singular[1] > 1e-8 * singular[0]
    assert singular[2] <= 1e-8 * singular[0]
    pre_data = _pre(panel.data, panel.treatment_time)
    assert np.linalg.matrix_rank(pre_data.to_numpy()) == 2
    pre = _pre(panel.untreated, panel.treatment_time)
    fit = _simplex_weights(pre[controls], pre[treated])
    assert not fit.success
    post_effect = _post(panel.effect, panel.treatment_time)[treated]
    assert (post_effect.to_numpy() != 0).any()


def test_donor_shock_is_not_in_the_treated_untreated_path() -> None:
    """One control post path leaves the combination that still generates treated."""
    panel = panels.donor_shock
    treated = panel.treated_units[0]
    controls = list(panel.control_units)
    pre = _pre(panel.untreated, panel.treatment_time)
    post = _post(panel.untreated, panel.treatment_time)
    fit = _simplex_weights(pre[controls], pre[treated])
    assert fit.success
    weights = fit.x
    assert _residual(pre[controls], pre[treated], weights) == 0.0
    assert _residual(post[controls], post[treated], weights) > 0.0
    shocked = [
        unit
        for unit in controls
        if not np.all(post[unit].to_numpy() == pre[unit].iloc[-1])
    ]
    assert len(shocked) == 1
    replaced = post[controls].copy()
    replaced[shocked[0]] = pre[shocked[0]].iloc[-1]
    assert _residual(replaced, post[treated], weights) == 0.0
    assert not np.allclose(
        post[treated].to_numpy(), post[controls].to_numpy() @ weights
    )
    assert (panel.effect[treated].to_numpy() == 0).all()


def test_placebo_effect_is_identically_zero() -> None:
    """Placebo plants no intervention in any cell."""
    panel = panels.placebo
    assert (panel.effect.to_numpy() == 0).all()
    assert panel.data.equals(panel.untreated)


def test_reload_is_deterministic() -> None:
    """Building the panels does not draw observation noise."""
    before = panels.exact_convex_hull.data.copy()
    importlib.reload(panels)
    assert before.equals(panels.exact_convex_hull.data)
    assert panels.exact_convex_hull.data.equals(
        panels.exact_convex_hull.untreated + panels.exact_convex_hull.effect
    )


def test_panels_are_not_public_api() -> None:
    """The module is not exported from the public packages."""
    assert not hasattr(causalpy, "known_truth_panels")
    assert not hasattr(causalpy.data, "known_truth_panels")
    assert not hasattr(causalpy.experiments, "known_truth_panels")
    assert "_KnownTruthPanel" not in getattr(causalpy, "__all__", [])
