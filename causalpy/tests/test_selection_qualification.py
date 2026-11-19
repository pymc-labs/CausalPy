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
"""Repeated selection procedure for issue #319.

Ordinary CI runs one null panel and one alternative panel. Each freezes an
evaluated candidate and audits the reserved origin. The repeated study is
nightly, capped at 100 panels per scenario, and is not expanded past that
cap. It repeats that freeze-before-audit procedure. It does not claim that
the screen recovers an operationally relevant geo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalpy.design import DecisionRule, SyntheticControlPlanner
from causalpy.pymc_models import WeightedSumFitter

N_PANELS = 100
LIGHT = {
    "draws": 8,
    "tune": 8,
    "chains": 1,
    "cores": 1,
    "progressbar": False,
}
_ELIGIBLE = ("target", "other", "donor_0", "donor_1")
_COSTS = {"target": 10.0, "other": 1.0, "donor_0": 5.0, "donor_1": 5.0}
_RELEVANCE = {"target": 0.2, "other": 0.8, "donor_0": 0.1, "donor_1": 0.1}


def _panel(seed: int, *, alternative: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.Index(range(20), name="t")
    donor_0 = rng.normal(size=20)
    donor_1 = 0.7 * donor_0 + rng.normal(scale=0.3, size=20)
    target = 0.6 * donor_0 + 0.4 * donor_1
    other = 0.25 * donor_0 + rng.normal(scale=0.9, size=20)
    if alternative:
        target = target + 1.0
    return pd.DataFrame(
        {"target": target, "other": other, "donor_0": donor_0, "donor_1": donor_1},
        index=index,
    )


def _run(seed: int, *, alternative: bool):
    frame = _panel(seed, alternative=alternative)
    planner = SyntheticControlPlanner(
        frame,
        "target",
        ["donor_0", "donor_1"],
        DecisionRule("positive", "cumulative", threshold=0.0, probability=0.95),
        model=WeightedSumFitter(sample_kwargs=LIGHT),
        random_seed=seed,
    )
    return planner.search(
        durations=(4,),
        selection_origins=(8,),
        audit_origins=(16,),
        eligible_units=_ELIGIBLE,
        costs=_COSTS,
        representativeness=_RELEVANCE,
        target_lift=1.0,
        absolute_lifts=(1.0,),
        sample_kwargs=LIGHT,
        max_fits=8,
    )


def _assert_withheld(result) -> None:
    assert result.global_optimum is False
    recommendation = result.recommendation()
    assert recommendation.startswith("No supported recommendation")
    assert "No winner was selected." in recommendation
    assert "not a launch recommendation" in recommendation
    assert "not a global optimum" in recommendation
    assert "recommend launching" not in recommendation
    assert result.n_screened == len(_ELIGIBLE)
    assert {item.treated_unit for item in result.candidates} == set(_ELIGIBLE)


def _assert_freeze_before_audit(result, frame: pd.DataFrame) -> None:
    """Freeze one evaluated candidate. Audit must not select a replacement."""
    _assert_withheld(result)
    evaluated = [
        item for item in result.evaluated if item.comparison.n_inference_fits > 0
    ]
    assert evaluated, "the procedure produced no candidate that could be frozen"
    chosen = evaluated[0]
    assert 16 not in [window.origin for window in chosen.comparison.windows]
    frozen = result.freeze(chosen.candidate_id, 4)
    assert frozen.treated_unit == chosen.treated_unit
    assert frozen.control_units == chosen.donors
    assert frozen.selected_duration == 4
    for other in evaluated[1:]:
        competitor = result.freeze(other.candidate_id, 4)
        assert competitor.treated_unit == other.treated_unit
        assert competitor.control_units == other.donors
        assert competitor.treated_unit != frozen.treated_unit or (
            competitor.control_units != frozen.control_units
        )
    audit = frozen.audit(frame)
    assert audit.selected_duration == frozen.selected_duration
    assert frozen.treated_unit == chosen.treated_unit
    assert frozen.control_units == chosen.donors
    assert audit.recommendation().startswith("No supported recommendation")


def test_selection_procedure_withholds_launch_on_null_and_alternative() -> None:
    """One panel of each scenario. Freeze precedes the untouched audit."""
    null_frame = _panel(0, alternative=False)
    alternative_frame = _panel(1, alternative=True)
    null = _run(0, alternative=False)
    alternative = _run(1, alternative=True)
    _assert_freeze_before_audit(null, null_frame)
    _assert_freeze_before_audit(alternative, alternative_frame)
    assert int(null.summary["seed"].iloc[0]) != int(alternative.summary["seed"].iloc[0])


@pytest.mark.nightly
def test_selection_procedure_repeats_on_capped_panels() -> None:
    """Repeat freeze-before-audit. Do not expand past 100 panels per scenario."""
    for alternative, offset in ((False, 0), (True, 10_000)):
        seeds: set[int] = set()
        for panel in range(N_PANELS):
            frame = _panel(panel + offset, alternative=alternative)
            result = _run(panel + offset, alternative=alternative)
            _assert_freeze_before_audit(result, frame)
            seeds.add(int(result.summary["seed"].iloc[0]))
        assert len(seeds) == N_PANELS
