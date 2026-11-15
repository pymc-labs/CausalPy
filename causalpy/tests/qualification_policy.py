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
"""Executable recommendation-eligibility policy for issue #914.

The policy never issues a launch recommendation. A supported claim is a
predeclared statement of what the named evidence permits. It is not a
universal pass threshold and not prospective power.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import binomtest

from causalpy.tests.qualification_protocol import (
    ESTIMATOR,
    MISFIT_SCENARIO,
    NULL_TOLERANCE_REASON,
    PROTOCOL,
    REFERENCE_SCENARIO,
    SCENARIOS,
)

ClaimStatus = Literal["supported", "descriptive", "model-conditional", "inconclusive"]

LAUNCH_NOT_QUALIFIED = "launch-not-qualified"
PROSPECTIVE_POWER_WITHHELD = "prospective-power-withheld"
NULL_TOLERANCE_WITHHELD = "null-tolerance-withheld"
DURATION_AUDIT_GAP = "duration-audit-gap"
DEPENDENT_WINDOWS = "dependent-windows-not-binomial"
PANEL_COUNT_BELOW_PROTOCOL = "panel-count-below-protocol"
FAILURES_IN_CLAIM = "failures-included-not-a-precision-claim"
FAILURES_DROPPED = "failures-dropped-from-denominator"
DETECTION_SE_TIGHTER_THAN_PROTOCOL = "detection-se-tighter-than-protocol"
TRANSLATION_IDENTITY_FAILED = "translation-identity-failed"
UNBIASEDNESS_WITHHELD_MISFIT = "unbiasedness-withheld-misfit"
UNBIASEDNESS_WITHHELD_NOT_REFERENCE = "unbiasedness-withheld-not-reference"
MISFIT_OPERATING_DESCRIPTIVE = "misfit-operating-descriptive"
UNBIASEDNESS_WITHHELD_BIAS = "unbiasedness-withheld-bias-interval"
UNBIASEDNESS_WITHHELD_FAILURES = "unbiasedness-withheld-failures"
UNBIASEDNESS_WITHHELD_NO_TOLERANCE = "unbiasedness-withheld-no-tolerance"
HULL_WARNINGS_CONFOUND_LABEL = "hull-warnings-confound-reference"
CURVE_INAPPLICABLE = "curve-inapplicable"
NUMERIC_MATCH_NOT_VALIDATION = "numeric-match-not-validation"
ESTIMATOR_MISMATCH = "estimator-mismatch"
UNKNOWN_SCENARIO = "unknown-scenario"
HISTORICAL_NOT_BINOMIAL = "historical-windows-not-binomial"
ASSUMPTIONS_NOT_NAMED = "assumptions-not-named"


@dataclass(frozen=True)
class RateSummary:
    """A count, a rate, and the uncertainty the protocol permits."""

    count: int
    denominator: int
    rate: float | None
    interval_low: float | None
    interval_high: float | None
    plug_in_standard_error: float | None
    plug_in_standard_error_is_qualified_precision: bool
    interval_method: str


@dataclass(frozen=True)
class EligibilityInput:
    """One claim request. Fields the caller omits are not silently granted."""

    kind: str
    scenario: str
    estimator: str
    n_requested: int
    n_failed: int
    failures_in_denominator: bool
    windows_independent: bool
    null_tolerance_requested: float | None
    claimed_detection_se: float | None
    launch_requested: bool
    prospective_power_requested: bool
    numeric_match_is_validation: bool
    depends_on_translation: bool
    translation_identity_ok: bool
    calibration_applicable: bool
    detection_target_applies: bool
    bias_interval_covers_zero: bool
    assumptions_named: tuple[str, ...] = ()
    hull_warnings: int = 0


@dataclass(frozen=True)
class EligibilityDecision:
    """What the protocol permits. ``permitted_claim`` is None when nothing is."""

    status: ClaimStatus
    permitted_claim: str | None
    withheld: tuple[str, ...]
    reasons: tuple[str, ...]


def binomial_rate(count: int, denominator: int, *, independent: bool) -> RateSummary:
    """Clopper-Pearson interval for independent panels.

    Dependent windows must not receive this interval. The plug-in standard
    error is a calculation. It is a qualified precision claim only at the
    protocol panel count, with every requested panel finished and independent.
    """
    if not independent:
        raise ValueError(
            "Dependent windows do not have a binomial standard error. "
            "Report the count and say the windows are dependent."
        )
    if denominator < 0 or count < 0 or count > denominator:
        raise ValueError(
            f"count {count} is not in 0..{denominator} for a binomial rate."
        )
    if denominator == 0:
        return RateSummary(
            count=0,
            denominator=0,
            rate=None,
            interval_low=None,
            interval_high=None,
            plug_in_standard_error=None,
            plug_in_standard_error_is_qualified_precision=False,
            interval_method="undefined-empty-denominator",
        )
    rate = count / denominator
    interval = binomtest(count, denominator).proportion_ci(
        confidence_level=PROTOCOL.rate_interval_level, method="exact"
    )
    plug_in = float(np.sqrt(rate * (1.0 - rate) / denominator))
    # A plug-in error below 0.04 is a calculation, not a qualified precision
    # claim, even when every requested panel finished.
    qualified = (
        denominator == PROTOCOL.n_panels
        and plug_in >= PROTOCOL.max_claimed_detection_se
    )
    return RateSummary(
        count=count,
        denominator=denominator,
        rate=rate,
        interval_low=float(interval.low),
        interval_high=float(interval.high),
        plug_in_standard_error=plug_in,
        plug_in_standard_error_is_qualified_precision=qualified,
        interval_method="clopper-pearson",
    )


def _withheld_always(item: EligibilityInput) -> list[str]:
    withheld = [LAUNCH_NOT_QUALIFIED, PROSPECTIVE_POWER_WITHHELD, DURATION_AUDIT_GAP]
    if item.null_tolerance_requested is not None or item.kind == "null-tolerance":
        withheld.append(NULL_TOLERANCE_WITHHELD)
    if (
        item.claimed_detection_se is not None
        and item.claimed_detection_se < PROTOCOL.max_claimed_detection_se
    ):
        withheld.append(DETECTION_SE_TIGHTER_THAN_PROTOCOL)
    return withheld


def _operating_blockers(item: EligibilityInput) -> list[str]:
    blockers: list[str] = []
    if item.estimator != ESTIMATOR:
        blockers.append(ESTIMATOR_MISMATCH)
    if item.scenario not in SCENARIOS:
        blockers.append(UNKNOWN_SCENARIO)
    if not item.windows_independent:
        blockers.append(DEPENDENT_WINDOWS)
    if item.n_requested != PROTOCOL.n_panels:
        blockers.append(PANEL_COUNT_BELOW_PROTOCOL)
    if item.n_failed > 0:
        blockers.append(FAILURES_IN_CLAIM)
    if not item.failures_in_denominator:
        blockers.append(FAILURES_DROPPED)
    if item.depends_on_translation and not item.translation_identity_ok:
        blockers.append(TRANSLATION_IDENTITY_FAILED)
    if item.launch_requested:
        blockers.append(LAUNCH_NOT_QUALIFIED)
    if item.prospective_power_requested:
        blockers.append(PROSPECTIVE_POWER_WITHHELD)
    if item.null_tolerance_requested is not None:
        blockers.append(NULL_TOLERANCE_WITHHELD)
    if (
        item.claimed_detection_se is not None
        and item.claimed_detection_se < PROTOCOL.max_claimed_detection_se
    ):
        blockers.append(DETECTION_SE_TIGHTER_THAN_PROTOCOL)
    if item.scenario == MISFIT_SCENARIO:
        blockers.append(MISFIT_OPERATING_DESCRIPTIVE)
    return blockers


def _operating_claim(item: EligibilityInput) -> str:
    claim = (
        f"Under protocol {PROTOCOL.protocol_id}, scenario {item.scenario!r}, "
        f"estimator {item.estimator!r}, and the shipped positive cumulative "
        f"decision rule, the study reports the observed detection count with "
        f"failures included in the denominator of {item.n_requested} requested "
        f"panels. {item.n_failed} fit failures are non-detections, not omitted "
        f"panels. This is not a launch recommendation, and it is not a claim "
        f"that the detection target of {PROTOCOL.detection_target} was achieved."
    )
    if item.scenario == MISFIT_SCENARIO:
        claim += (
            " This scenario is the persistent level and trend misfit. The count "
            "describes that biased process. It is not evidence that the "
            "estimator recovered an effect."
        )
    return claim


def _recovery_blockers(item: EligibilityInput) -> list[str]:
    """Unbiasedness is withheld unless a bias tolerance was predeclared.

    The protocol has no such tolerance. An interval that covers zero is a
    count, not support. Hull warnings on the reference also block calling
    that scenario a correctly specified recovery diagnostic.
    """
    blockers = [UNBIASEDNESS_WITHHELD_NO_TOLERANCE]
    if item.scenario == MISFIT_SCENARIO:
        blockers.append(UNBIASEDNESS_WITHHELD_MISFIT)
    elif item.scenario != REFERENCE_SCENARIO:
        blockers.append(UNBIASEDNESS_WITHHELD_NOT_REFERENCE)
    if item.n_failed != 0:
        blockers.append(UNBIASEDNESS_WITHHELD_FAILURES)
    if item.n_requested != PROTOCOL.n_panels:
        blockers.append(PANEL_COUNT_BELOW_PROTOCOL)
    if not item.windows_independent:
        blockers.append(DEPENDENT_WINDOWS)
    if not item.bias_interval_covers_zero:
        blockers.append(UNBIASEDNESS_WITHHELD_BIAS)
    if item.estimator != ESTIMATOR:
        blockers.append(ESTIMATOR_MISMATCH)
    if item.scenario == REFERENCE_SCENARIO and item.hull_warnings > 0:
        blockers.append(HULL_WARNINGS_CONFOUND_LABEL)
    return blockers


def _comparison_blockers(item: EligibilityInput) -> list[str]:
    blockers: list[str] = []
    if not item.calibration_applicable:
        blockers.append(CURVE_INAPPLICABLE)
    if item.numeric_match_is_validation:
        blockers.append(NUMERIC_MATCH_NOT_VALIDATION)
    if not item.assumptions_named:
        blockers.append(ASSUMPTIONS_NOT_NAMED)
    return blockers


def _recovery_claim(item: EligibilityInput, blockers: list[str]) -> str:
    """State the interval. Do not call covering zero approximate unbiasedness."""
    if item.bias_interval_covers_zero:
        interval = "The Monte Carlo bias interval covers zero. "
    else:
        interval = "The Monte Carlo bias interval does not cover zero. "
    parts = [
        interval,
        "Unbiasedness is withheld. No bias tolerance was predeclared. ",
    ]
    if UNBIASEDNESS_WITHHELD_MISFIT in blockers:
        parts.append(
            "This scenario is the persistent level and trend misfit, so the "
            "interval is not a recovery diagnostic. "
        )
    elif UNBIASEDNESS_WITHHELD_NOT_REFERENCE in blockers:
        parts.append(
            "This scenario is not the reference scenario, so the interval is "
            "not a recovery diagnostic. "
        )
    if HULL_WARNINGS_CONFOUND_LABEL in blockers:
        parts.append(
            f"{item.hull_warnings} convex-hull warnings are recorded, so this "
            "is not a correctly specified recovery diagnostic. "
        )
    parts.append("This is not a launch recommendation.")
    return "".join(parts)


def decide(item: EligibilityInput) -> EligibilityDecision:
    """Apply the predeclared rules. Does not look at posterior draws."""
    withheld = tuple(dict.fromkeys(_withheld_always(item)))
    if item.kind in {
        "launch",
        "prospective-power",
        "null-tolerance",
        "duration-selection",
    }:
        return EligibilityDecision(
            status="inconclusive",
            permitted_claim=None,
            withheld=withheld,
            reasons=(
                "This claim kind is not qualified by the protocol. "
                + NULL_TOLERANCE_REASON
                + " "
                + PROTOCOL.selected_duration_audit,
            ),
        )
    if item.kind == "historical-replay":
        reasons = [
            "Historical windows overlap. The count is product evidence, not a "
            "binomial standard error and not an independent experiment."
        ]
        status: ClaimStatus = "descriptive"
        extra = list(withheld)
        if not item.windows_independent:
            extra.append(HISTORICAL_NOT_BINOMIAL)
        if item.claimed_detection_se is not None:
            status = "inconclusive"
            reasons.append(
                "A binomial standard error was attached to dependent windows."
            )
        return EligibilityDecision(
            status=status,
            permitted_claim=(
                "Historical replay is descriptive product evidence. It does "
                "not carry a binomial standard error, and it is not a launch "
                "recommendation."
            ),
            withheld=tuple(dict.fromkeys(extra)),
            reasons=tuple(reasons),
        )
    if item.kind == "model-conditional-operating-comparison":
        blockers = _comparison_blockers(item)
        refused = [
            code
            for code in (CURVE_INAPPLICABLE, ASSUMPTIONS_NOT_NAMED)
            if code in blockers
        ]
        if refused:
            reasons = []
            if CURVE_INAPPLICABLE in refused:
                reasons.append(
                    "The calibration record is inapplicable. No substitute "
                    "curve is invented."
                )
            if ASSUMPTIONS_NOT_NAMED in refused:
                reasons.append(
                    "A model-conditional comparison must name its assumptions. "
                    "None were named."
                )
            return EligibilityDecision(
                status="inconclusive",
                permitted_claim=None,
                withheld=tuple(dict.fromkeys([*withheld, *refused])),
                reasons=tuple(reasons),
            )
        named = ", ".join(item.assumptions_named)
        return EligibilityDecision(
            status="model-conditional",
            permitted_claim=(
                "Model-conditional comparison under the named assumptions: "
                f"{named}. Numerical agreement is not validation, and this is "
                "not a launch recommendation."
            ),
            withheld=tuple(dict.fromkeys([*withheld, NUMERIC_MATCH_NOT_VALIDATION])),
            reasons=tuple(blockers)
            or ("Assumptions are named and the curve is applicable.",),
        )
    if item.kind == "recovery-summary":
        blockers = _recovery_blockers(item)
        return EligibilityDecision(
            status="descriptive",
            permitted_claim=_recovery_claim(item, blockers),
            withheld=tuple(dict.fromkeys([*withheld, *blockers])),
            reasons=tuple(blockers),
        )
    if item.kind != "operating-summary":
        raise ValueError(f"Unknown eligibility kind {item.kind!r}.")
    blockers = _operating_blockers(item)
    # A supported operating claim states the observed count under the protocol.
    # Failures, a short panel, or a broken translation keep it descriptive or
    # inconclusive. They do not become a tighter claim.
    hard = {
        FAILURES_DROPPED,
        TRANSLATION_IDENTITY_FAILED,
        DEPENDENT_WINDOWS,
        ESTIMATOR_MISMATCH,
        UNKNOWN_SCENARIO,
        DETECTION_SE_TIGHTER_THAN_PROTOCOL,
        NULL_TOLERANCE_WITHHELD,
        LAUNCH_NOT_QUALIFIED,
        PROSPECTIVE_POWER_WITHHELD,
    }
    if any(code in hard for code in blockers):
        return EligibilityDecision(
            status="inconclusive",
            permitted_claim=None,
            withheld=tuple(dict.fromkeys([*withheld, *blockers])),
            reasons=tuple(blockers),
        )
    if blockers:
        return EligibilityDecision(
            status="descriptive",
            permitted_claim=_operating_claim(item),
            withheld=tuple(dict.fromkeys([*withheld, *blockers])),
            reasons=tuple(blockers),
        )
    return EligibilityDecision(
        status="supported",
        permitted_claim=_operating_claim(item),
        withheld=withheld,
        reasons=(
            "Protocol panel count, independent panels, failures included, and "
            "the shipped decision rule are all present. The claim does not "
            "say the detection target was achieved.",
        ),
    )


def operating_input_from_summary(summary: Mapping[str, object]) -> EligibilityInput:
    """Build the operating claim from a stored numeric summary.

    The ordinary CI test uses this so a hand-edited report cannot claim a
    status the policy would refuse.
    """
    return EligibilityInput(
        kind="operating-summary",
        scenario=str(summary["scenario"]),
        estimator=str(summary["estimator"]),
        n_requested=int(summary["n_requested"]),
        n_failed=int(summary["n_failed"]),
        failures_in_denominator=bool(summary["failures_in_denominator"]),
        windows_independent=bool(summary["windows_independent"]),
        null_tolerance_requested=summary["null_tolerance_requested"],  # type: ignore[arg-type]
        claimed_detection_se=summary["claimed_detection_se"],  # type: ignore[arg-type]
        launch_requested=bool(summary["launch_requested"]),
        prospective_power_requested=bool(summary["prospective_power_requested"]),
        numeric_match_is_validation=bool(summary["numeric_match_is_validation"]),
        depends_on_translation=bool(summary["depends_on_translation"]),
        translation_identity_ok=bool(summary["translation_identity_ok"]),
        calibration_applicable=bool(summary["calibration_applicable"]),
        detection_target_applies=bool(summary["detection_target_applies"]),
        bias_interval_covers_zero=bool(summary["bias_interval_covers_zero"]),
    )


def recovery_input_from_summary(
    summary: Mapping[str, object], *, hull_warnings: int
) -> EligibilityInput:
    """Build the recovery claim from a stored lift and its scenario hull count.

    The ordinary CI test uses this so a stored ``supported`` label cannot
    outlive the policy. The hull count is the scenario total, not a lift field.
    """
    return EligibilityInput(
        kind="recovery-summary",
        scenario=str(summary["scenario"]),
        estimator=str(summary["estimator"]),
        n_requested=int(summary["n_requested"]),
        n_failed=int(summary["n_failed"]),
        failures_in_denominator=True,
        windows_independent=True,
        null_tolerance_requested=None,
        claimed_detection_se=None,
        launch_requested=False,
        prospective_power_requested=False,
        numeric_match_is_validation=False,
        depends_on_translation=False,
        translation_identity_ok=True,
        calibration_applicable=False,
        detection_target_applies=False,
        bias_interval_covers_zero=bool(summary["bias_interval_covers_zero"]),
        hull_warnings=hull_warnings,
    )
