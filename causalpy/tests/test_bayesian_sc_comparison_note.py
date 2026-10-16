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
"""Contract for the cross-model Bayesian synthetic-control note and notebook."""

from pathlib import Path

import pandas as pd

from causalpy.experiments.augmented_synthetic_control import AugmentedSyntheticControl
from causalpy.experiments.interactive_fixed_effects import (
    InteractiveFixedEffectsResult,
)

_REPO = Path(__file__).resolve().parents[2]
_NOTE = (
    _REPO
    / "docs"
    / "source"
    / "knowledgebase"
    / "comparing-bayesian-synthetic-control.md"
)
_NOTEBOOK = (
    _REPO
    / "docs"
    / "source"
    / "notebooks"
    / "comparing-bayesian-synthetic-control-pymc.ipynb"
)

# Names the comparison must not read. Absence is not a defect.
_DROPPED = (
    "baseline_weights",
    "effective_weights",
    "negative_weight_magnitude",
    "time_varying_correction",
)


def test_methods_note_cites_phase_2_and_does_not_reopen_dropped_quantities():
    text = _NOTE.read_text(encoding="utf-8")
    notebook = _NOTEBOOK.read_text(encoding="utf-8")
    combined = text + notebook

    assert "bayesian_synthetic_control_known_truth_report.md" in text
    assert "latent_factor" in text
    assert "donor_shock" in text
    assert "causalpy.tests" not in notebook
    assert "known_truth_panels" not in notebook
    assert "nominal coverage" in text
    assert "WeightedSumFitter" in text
    assert "not a frequentist coverage rate" in text
    assert "pre-intervention" in text

    for name in _DROPPED:
        assert name not in combined
    assert "fix to the owning model" not in text
    assert "shrinkage_state" in text
    assert "pre-period prediction error" in text


def test_augmented_model_exposes_only_the_frozen_inference_names():
    times = range(6)
    frame = pd.DataFrame(
        {
            "c0": [0.0, 1, 0, 1, 0, 1],
            "c1": [1.0, 0, 1, 0, 1, 0],
            "treated": [0.5, 0.5, 0.5, 0.5, 2.5, 2.5],
        },
        index=times,
    )
    experiment = AugmentedSyntheticControl(
        frame,
        4,
        ["c0", "c1"],
        ["treated"],
    )
    names = set(experiment.model.named_vars)
    assert {
        "base_weights",
        "prognostic_coefficients",
        "counterfactual",
        "treatment_effect",
        "tau",
    } <= names
    for name in _DROPPED:
        assert name not in names


def test_factor_result_exposes_shrinkage_and_pre_period_score():
    fields = InteractiveFixedEffectsResult.__dataclass_fields__
    assert "shrinkage_state" in fields
    assert "score" in fields
    assert "n_factors" not in fields
