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
"""Fast qualification checks. The repeated-data study is nightly."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog

from causalpy.design import SyntheticControlPlanner
from causalpy.historical_validation import validate_schedule
from causalpy.tests.qualification_panels import (
    data_generator,
    generate_untreated,
    historical_replay_frame,
    historical_schedule_request,
    inject_treatment,
    mean_frame,
    residual_frame,
    sampler_seed,
)
from causalpy.tests.qualification_policy import (
    EligibilityInput,
    binomial_rate,
    decide,
    operating_input_from_summary,
)
from causalpy.tests.qualification_protocol import (
    DONORS,
    PROTOCOL,
    SCENARIOS,
    TREATED,
    WEIGHTS,
)
from causalpy.tests.qualification_study import load_committed_report, write_evidence

_FORBIDDEN = (
    "recommend launching",
    "prospective power",
    "validated false-positive",
    "universal pass threshold",
)


def _input(**overrides: object) -> EligibilityInput:
    payload = {
        "kind": "operating-summary",
        "scenario": SCENARIOS[0],
        "estimator": PROTOCOL.estimator,
        "n_requested": PROTOCOL.n_panels,
        "n_failed": 0,
        "failures_in_denominator": True,
        "windows_independent": True,
        "null_tolerance_requested": None,
        "claimed_detection_se": None,
        "launch_requested": False,
        "prospective_power_requested": False,
        "numeric_match_is_validation": False,
        "depends_on_translation": False,
        "translation_identity_ok": True,
        "correctly_specified": True,
        "calibration_applicable": False,
        "detection_target_applies": False,
        "bias_interval_covers_zero": True,
    }
    payload.update(overrides)
    return EligibilityInput(**payload)


def test_protocol_withholds_null_tolerance_and_names_the_duration_gap() -> None:
    """The HDI and the tail probability are not a false-positive tolerance."""
    assert PROTOCOL.null_error_tolerance is None
    assert PROTOCOL.decision_probability != 1.0 - PROTOCOL.coverage_hdi_prob
    assert "gap: no public duration-freeze API" in PROTOCOL.selected_duration_audit
    assert PROTOCOL.n_panels == 100
    assert PROTOCOL.max_claimed_detection_se == 0.04


def test_planner_has_no_public_duration_freeze() -> None:
    """The gap is unmet. A later freeze API must fail this until the audit exists."""
    public = [name for name in dir(SyntheticControlPlanner) if not name.startswith("_")]
    assert not any("freeze" in name or name == "select_duration" for name in public)


def test_reference_mean_is_the_declared_convex_combination() -> None:
    """The correctly specified mean is exact. Noise is added later, once."""
    frame = mean_frame(PROTOCOL.n_time, misfit=False)
    reconstructed = frame[list(DONORS)].to_numpy() @ np.asarray(WEIGHTS)
    assert np.allclose(reconstructed, frame[TREATED].to_numpy())


def test_misfit_mean_is_outside_the_donor_simplex() -> None:
    """A least-squares residual is not the contract. The constrained system is."""
    frame = mean_frame(PROTOCOL.n_time, misfit=True)
    donors = frame[list(DONORS)].to_numpy()
    treated = frame[TREATED].to_numpy()
    result = linprog(
        np.zeros(len(DONORS)),
        A_eq=np.vstack([donors, np.ones((1, len(DONORS)))]),
        b_eq=np.concatenate([treated, [1.0]]),
        bounds=[(0.0, None)] * len(DONORS),
        method="highs",
    )
    assert not result.success


def test_treatment_is_injected_once_onto_the_untreated_path() -> None:
    """A second call must not add another residual draw."""
    panel = generate_untreated(SCENARIOS[0], 0)
    assert np.allclose(
        panel.untreated.to_numpy(), (panel.mean + panel.residual).to_numpy()
    )
    injected = inject_treatment(panel.untreated, PROTOCOL.large_lift)
    delta = injected[TREATED].to_numpy() - panel.untreated[TREATED].to_numpy()
    assert np.allclose(delta[: -PROTOCOL.duration], 0.0)
    assert np.allclose(delta[-PROTOCOL.duration :], PROTOCOL.large_lift)
    again = inject_treatment(panel.untreated, PROTOCOL.large_lift)
    assert np.allclose(again.to_numpy(), injected.to_numpy())


def test_panel_streams_do_not_share_a_residual() -> None:
    """Repeating a seed on the same panel is the same dataset, not a new one."""
    first = generate_untreated(SCENARIOS[0], 0)
    repeat = generate_untreated(SCENARIOS[0], 0)
    other = generate_untreated(SCENARIOS[0], 1)
    serial = generate_untreated(SCENARIOS[2], 0)
    assert np.allclose(first.untreated.to_numpy(), repeat.untreated.to_numpy())
    assert not np.allclose(first.residual.to_numpy(), other.residual.to_numpy())
    assert not np.allclose(first.residual.to_numpy(), serial.residual.to_numpy())
    assert sampler_seed(1, 0) != sampler_seed(1, 1)
    assert sampler_seed(1, 0) != sampler_seed(3, 0)


def test_serial_residual_is_autocorrelated() -> None:
    """AR(1) is the residual process. Lag-1 correlation of an iid series is not."""
    serial = residual_frame(data_generator(3, 7), 400, serial=True)[TREATED].to_numpy()
    independent = residual_frame(data_generator(1, 7), 400, serial=False)[
        TREATED
    ].to_numpy()
    serial_corr = float(np.corrcoef(serial[1:], serial[:-1])[0, 1])
    independent_corr = float(np.corrcoef(independent[1:], independent[:-1])[0, 1])
    assert serial_corr > 0.4
    assert abs(independent_corr) < 0.2


def test_historical_schedule_overlaps_and_reserves_the_audit() -> None:
    """Dependent windows are the point. A holdout intersection must still fail."""
    frame = historical_replay_frame()
    schedule = validate_schedule(frame, TREATED, DONORS, historical_schedule_request())
    assert schedule.audit_reserved
    assert schedule.selection_audit_separated
    assert schedule.overlapping_pairs
    assert all(window.status == "usable" for window in schedule.windows)
    assert sum(window.role == "audit" for window in schedule.windows) == 1


def test_launch_and_null_tolerance_are_never_supported() -> None:
    """A numeric tolerance, including the HDI complement, does not become a claim."""
    launch = decide(_input(kind="launch", launch_requested=True))
    tolerance = decide(_input(kind="null-tolerance", null_tolerance_requested=0.06))
    duration = decide(_input(kind="duration-selection"))
    power = decide(_input(kind="prospective-power", prospective_power_requested=True))
    for decision in (launch, tolerance, duration, power):
        assert decision.status == "inconclusive"
        assert decision.permitted_claim is None
    assert "launch-not-qualified" in launch.withheld
    assert "null-tolerance-withheld" in tolerance.withheld
    assert "duration-audit-gap" in duration.withheld


def test_operating_claim_stays_descriptive_when_the_panel_count_is_short() -> None:
    """Fewer than 100 panels withholds the protocol precision claim."""
    decision = decide(_input(n_requested=20, n_failed=1, failures_in_denominator=True))
    assert decision.status == "descriptive"
    assert "panel-count-below-protocol" in decision.withheld
    assert decision.permitted_claim is not None
    assert "not a launch recommendation" in decision.permitted_claim
    for phrase in _FORBIDDEN:
        assert phrase not in decision.permitted_claim


def test_dropping_failures_is_inconclusive() -> None:
    """A denominator that omits failures is narrower than the acceptance rule."""
    decision = decide(_input(n_failed=3, failures_in_denominator=False))
    assert decision.status == "inconclusive"
    assert decision.permitted_claim is None
    assert "failures-dropped-from-denominator" in decision.withheld


def test_tighter_detection_se_and_broken_translation_are_inconclusive() -> None:
    """The design's 0.04 bound is not a target to beat after seeing rates."""
    tight = decide(_input(claimed_detection_se=0.01))
    broken = decide(_input(depends_on_translation=True, translation_identity_ok=False))
    assert tight.status == "inconclusive"
    assert broken.status == "inconclusive"
    assert "detection-se-tighter-than-protocol" in tight.withheld
    assert "translation-identity-failed" in broken.withheld


def test_supported_operating_claim_does_not_say_the_target_was_achieved() -> None:
    """Support means the count may be stated. It is not a power result."""
    decision = decide(_input())
    assert decision.status == "supported"
    assert decision.permitted_claim is not None
    assert "not a launch recommendation" in decision.permitted_claim
    assert "was achieved" in decision.permitted_claim
    for phrase in _FORBIDDEN:
        assert phrase not in decision.permitted_claim


def test_misfit_recovery_cannot_be_called_unbiased() -> None:
    """A bias interval that covers zero does not rescue a structural misfit."""
    decision = decide(
        _input(
            kind="recovery-summary",
            scenario=SCENARIOS[1],
            correctly_specified=False,
            bias_interval_covers_zero=True,
        )
    )
    assert decision.status == "descriptive"
    assert "unbiasedness-withheld-misfit" in decision.withheld


def test_misfit_operating_count_is_descriptive() -> None:
    """A structurally biased scenario does not receive a supported count claim."""
    decision = decide(
        _input(
            scenario=SCENARIOS[1],
            correctly_specified=False,
        )
    )
    assert decision.status == "descriptive"
    assert decision.permitted_claim is not None
    assert "biased process" in decision.permitted_claim
    assert "misfit-operating-descriptive" in decision.withheld
    for phrase in _FORBIDDEN:
        assert phrase not in decision.permitted_claim


def test_serial_recovery_is_not_labeled_as_the_level_trend_misfit() -> None:
    """Serial correlation withholds unbiasedness for a different reason."""
    decision = decide(
        _input(
            kind="recovery-summary",
            scenario=SCENARIOS[2],
            correctly_specified=False,
            bias_interval_covers_zero=True,
        )
    )
    assert decision.status == "descriptive"
    assert "unbiasedness-withheld-not-reference" in decision.withheld
    assert "unbiasedness-withheld-misfit" not in decision.withheld


def test_plug_in_error_below_the_bound_is_not_qualified_precision() -> None:
    """A zero or sub-0.04 plug-in error is not a precision achievement."""
    certain = binomial_rate(PROTOCOL.n_panels, PROTOCOL.n_panels, independent=True)
    tight = binomial_rate(97, PROTOCOL.n_panels, independent=True)
    wide = binomial_rate(38, PROTOCOL.n_panels, independent=True)
    assert certain.plug_in_standard_error == 0.0
    assert certain.plug_in_standard_error_is_qualified_precision is False
    assert tight.plug_in_standard_error < PROTOCOL.max_claimed_detection_se
    assert tight.plug_in_standard_error_is_qualified_precision is False
    assert wide.plug_in_standard_error >= PROTOCOL.max_claimed_detection_se
    assert wide.plug_in_standard_error_is_qualified_precision is True


def test_dependent_windows_have_no_binomial_interval() -> None:
    """Historical replay must not receive the independent-panel interval."""
    with pytest.raises(ValueError, match="Dependent windows"):
        binomial_rate(1, 3, independent=False)
    decision = decide(
        _input(
            kind="historical-replay",
            scenario="historical-replay",
            windows_independent=False,
            claimed_detection_se=0.2,
        )
    )
    assert decision.status == "inconclusive"


def test_inapplicable_curve_does_not_invent_a_substitute() -> None:
    """Numerical exactness is not qualification, and a missing curve stays missing."""
    decision = decide(
        _input(
            kind="model-conditional-operating-comparison",
            calibration_applicable=False,
            assumptions_named=("fold-sd mixture",),
        )
    )
    assert decision.status == "inconclusive"
    assert decision.permitted_claim is None


def test_short_study_cannot_be_written_as_evidence(tmp_path) -> None:
    """A smoke run must not replace the 100-panel report."""
    with pytest.raises(ValueError, match="short study"):
        write_evidence({"short_study": True, "n_panels_requested": 1}, tmp_path)


def test_committed_report_matches_the_policy_and_the_panel_counts() -> None:
    """Ordinary CI re-decides the stored summary. It does not resample."""
    report = load_committed_report()
    assert report["protocol_id"] == PROTOCOL.protocol_id
    assert report["n_panels_requested"] == PROTOCOL.n_panels
    assert report["launch_recommendation_qualified"] is False
    assert report["null_tolerance"] is None
    assert "gap: no public duration-freeze API" in report["selected_duration_audit"]
    assert "not a launch recommendation" in (REPORT_TEXT := _markdown())
    for phrase in _FORBIDDEN:
        assert phrase not in REPORT_TEXT
    for scenario in report["scenarios"]:
        panels = scenario["panels"]
        assert len(panels) == PROTOCOL.n_panels
        assert len({tuple(panel["data_seed_entropy"]) for panel in panels}) == len(
            panels
        )
        for lift in scenario["lifts"]:
            key = f"{lift['lift']:.2f}"
            failed = [panel for panel in panels if panel["status"] == "fit-failed"]
            detected = [
                panel
                for panel in panels
                if panel["status"] != "fit-failed"
                and panel["lifts"][key]["positive_detected"]
            ]
            assert lift["n_failed"] == len(failed)
            assert lift["n_detected"] == len(detected)
            assert lift["failures_in_denominator"] is True
            assert lift["claimed_detection_se"] is None
            assert lift["plug_in_standard_error_is_qualified_precision"] is (
                lift["plug_in_standard_error"] >= PROTOCOL.max_claimed_detection_se
            )
            if scenario["scenario"] == SCENARIOS[1]:
                assert lift["eligibility"]["status"] == "descriptive"
                assert "misfit-operating-descriptive" in lift["eligibility"]["withheld"]
            decision = decide(operating_input_from_summary(lift))
            assert decision.status == lift["eligibility"]["status"]
            assert decision.permitted_claim == lift["eligibility"]["permitted_claim"]
    assert "Hull warnings" in REPORT_TEXT
    assert "not a qualified precision claim" in REPORT_TEXT
    assert "95 of 100" in REPORT_TEXT


@pytest.mark.nightly
def test_repeated_data_study_obeys_the_protocol() -> None:
    """Re-run the bounded study. Do not require bit-identical draws.

    The committed report is one machine's evidence. This check is the
    independent re-run. It rejects a launch claim, a numeric null tolerance,
    a dropped failure, and a detection-probability SE tighter than 0.04.
    """
    from causalpy.tests.qualification_study import run_qualification

    report = run_qualification(progress=True)
    assert report["short_study"] is False
    assert report["n_panels_requested"] == PROTOCOL.n_panels
    assert report["launch_recommendation_qualified"] is False
    assert report["null_tolerance"] is None
    for scenario in report["scenarios"]:
        assert len(scenario["panels"]) == PROTOCOL.n_panels
        for lift in scenario["lifts"]:
            assert lift["failures_in_denominator"] is True
            assert lift["claimed_detection_se"] is None
            assert lift["n_requested"] == PROTOCOL.n_panels
            assert lift["plug_in_standard_error_is_qualified_precision"] is (
                lift["plug_in_standard_error"] >= PROTOCOL.max_claimed_detection_se
            )
            if scenario["scenario"] == SCENARIOS[1]:
                assert lift["eligibility"]["status"] == "descriptive"
            decision = decide(operating_input_from_summary(lift))
            assert decision.status == lift["eligibility"]["status"]
            if lift["eligibility"]["permitted_claim"]:
                assert (
                    "not a launch recommendation"
                    in lift["eligibility"]["permitted_claim"]
                )
                for phrase in _FORBIDDEN:
                    assert phrase not in lift["eligibility"]["permitted_claim"]
    historical = report["historical_replay"]
    assert historical["binomial_standard_error"] is None
    assert historical["fit_audit"] is False
    assert historical["curve_validated"] is False


def _markdown() -> str:
    from causalpy.tests.qualification_study import REPORT_MD

    return REPORT_MD.read_text(encoding="utf-8")
