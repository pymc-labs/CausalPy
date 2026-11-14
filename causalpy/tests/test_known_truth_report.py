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
"""Perturbation check for known-truth panels.

Shifting treated post-period outcomes moves impact and leaves the untreated
trajectory unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalpy.tests.known_truth_report import (
    MODELS,
    PANELS,
    fit_panel,
    posterior_draws,
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
