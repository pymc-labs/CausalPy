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
"""Perturbation check and known-truth interval report.

The default suite runs the perturbation check and one fixed-seed smoke.
Refits of the other panel-model cells are nightly. They score the five
existing panels only.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from causalpy.constants import HDI_PROB
from causalpy.tests.known_truth_report import (
    MODELS,
    NOTE_PATH,
    PANELS,
    POSTERIOR_TARGET,
    fit_panel,
    posterior_draws,
    score_fit,
)

_ROW = re.compile(
    r"^\| (?P<panel>[a-z_]+) \| (?P<model>[A-Za-z]+) \| "
    r"(?P<posterior>[a-z_]+) \| (?P<time>\d+) \| (?P<inside>[01]) \|$"
)
_PERTURB = {
    "draws": 20,
    "tune": 20,
    "chains": 1,
    "cores": 1,
    "random_seed": 7,
    "progressbar": False,
}
_SHIFT = 5.0


def _recorded_rows() -> dict[tuple[str, str, str, int], int]:
    rows: dict[tuple[str, str, str, int], int] = {}
    for line in NOTE_PATH.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line)
        if match is None:
            continue
        key = (
            match.group("panel"),
            match.group("model"),
            match.group("posterior"),
            int(match.group("time")),
        )
        rows[key] = int(match.group("inside"))
    return rows


def test_note_names_models_intervals_and_both_regimes() -> None:
    """The committed note is the surface #1240 cites."""
    text = NOTE_PATH.read_text(encoding="utf-8")
    assert "HDI_PROB" in text
    assert f"{HDI_PROB}" in text
    assert "not a frequentist coverage rate" in text
    assert "nominal coverage" not in text.lower()
    assert "%" not in text
    assert "latent_factor" in text
    assert "InteractiveFixedEffectsSyntheticControl" in text
    assert "improves on `SyntheticControl`" in text
    assert "does not by itself establish causal validity" in text
    assert "donor_shock" in text
    assert "CausalResult.impact_post" in text
    rows = _recorded_rows()
    expected = {
        (panel, model, posterior, time)
        for panel in PANELS
        for model, targets in POSTERIOR_TARGET.items()
        for posterior in targets
        for time in (4, 5, 6, 7)
    }
    assert set(rows) == expected
    assert set(MODELS) == set(POSTERIOR_TARGET)


@pytest.mark.parametrize("model_name", MODELS)
def test_treated_post_perturbation_changes_impact_not_untreated(
    model_name: str,
) -> None:
    """Treated post outcomes change impact, not the untreated trajectory."""
    panel_name = "placebo"
    panel = PANELS[panel_name]
    baseline = fit_panel(model_name, panel_name, sample=_PERTURB)
    shifted = panel.data.copy()
    post = shifted.index >= panel.treatment_time
    unit = panel.treated_units[0]
    shifted.loc[post, unit] = shifted.loc[post, unit] + _SHIFT
    perturbed = fit_panel(model_name, panel_name, data=shifted, sample=_PERTURB)
    before = posterior_draws(model_name, baseline, panel_name)
    after = posterior_draws(model_name, perturbed, panel_name)
    untreated_name = (
        "counterfactual"
        if model_name == "AugmentedSyntheticControl"
        else "predictions_post"
    )
    impact_name = (
        "treatment_effect"
        if model_name == "AugmentedSyntheticControl"
        else "impact_post"
    )
    np.testing.assert_allclose(
        before[untreated_name].values,
        after[untreated_name].values,
        atol=1e-8,
    )
    impact_shift = after[impact_name].values - before[impact_name].values
    np.testing.assert_allclose(impact_shift, _SHIFT, atol=1e-6)


def test_fixed_seed_smoke_matches_note() -> None:
    """One fixed-seed inside/outside cell matches the committed note."""
    rows = _recorded_rows()
    scored = score_fit("SyntheticControl", "placebo")
    for (posterior, time), flag in scored.items():
        assert rows[("placebo", "SyntheticControl", posterior, time)] == flag


@pytest.mark.nightly
@pytest.mark.parametrize("panel_name", list(PANELS))
@pytest.mark.parametrize("model_name", MODELS)
def test_nightly_refit_matches_note(panel_name: str, model_name: str) -> None:
    """Refit the recorded cell. Update the note if this disagrees."""
    if panel_name == "placebo" and model_name == "SyntheticControl":
        pytest.skip("covered by the default-suite smoke")
    rows = _recorded_rows()
    scored = score_fit(model_name, panel_name)
    for (posterior, time), flag in scored.items():
        assert rows[(panel_name, model_name, posterior, time)] == flag
