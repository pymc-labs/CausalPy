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
"""Markdown view of a finished qualification report.

This module does not sample. ``write_evidence`` is the only caller.
"""

from __future__ import annotations

from typing import Any

from causalpy.tests.qualification_protocol import PROTOCOL, REFERENCE_SCENARIO


def render_markdown(report: dict[str, Any]) -> str:
    """One markdown view of the JSON. It does not add a claim the JSON lacks."""
    lines = [
        f"# Synthetic-control qualification report ({report['protocol_id']})",
        "",
        "This is not a launch recommendation. The selected-duration audit is unmet.",
        "",
        report["selected_duration_audit"],
        "",
        report["null_tolerance_reason"],
        "",
        report["fold_sd_assessment"],
        "",
        f"Panels requested: {report['n_panels_requested']}.",
        (
            "Translation identity matches: "
            f"{report['translation_identity']['matches']}. "
            "When it matches, the bias of the cumulative estimand is the same "
            "at every lift within a scenario, so those bias rows are not "
            "independent estimates."
        ),
        (
            "Launch recommendation qualified: "
            f"{report['launch_recommendation_qualified']}."
        ),
        (
            "Supported means the protocol conditions for stating the observed "
            "count were met. It does not mean the detection target was achieved, "
            "and a plug-in standard error below "
            f"{PROTOCOL.max_claimed_detection_se} is not a qualified precision claim."
        ),
        "",
        "## Scenario counts",
        "",
        (
            "| Scenario | Lift | Detected | Wrong sign | Null negative tail | "
            "Failed | Truth covered | Bias mean | Bias covers zero | "
            "Plug-in SE | SE qualified | Operating | Recovery |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scenario in report["scenarios"]:
        for lift in scenario["lifts"]:
            bias = lift["bias"]
            lines.append(
                "| {scenario} | {lift:.2f} | {detected} | {wrong} | {null_tail} | "
                "{failed} | {covered}/{denom} | {mean:.3f} | {covers} | "
                "{se:.4f} | {qualified} | {operating} | {recovery} |".format(
                    scenario=scenario["scenario"],
                    lift=float(lift["lift"]),
                    detected=lift["n_detected"],
                    wrong=lift["n_wrong_sign"],
                    null_tail=lift["n_negative_at_null"],
                    failed=lift["n_failed"],
                    covered=lift["n_truth_covers"],
                    denom=lift["coverage_denominator"],
                    mean=float(bias["mean"]),
                    covers=bias["covers_zero"],
                    se=float(lift["plug_in_standard_error"]),
                    qualified=lift["plug_in_standard_error_is_qualified_precision"],
                    operating=lift["eligibility"]["status"],
                    recovery=lift["recovery"]["status"],
                )
            )
    lines.extend(["", "## Hull warnings", ""])
    for scenario in report["scenarios"]:
        lines.append(
            f"{scenario['scenario']}: {scenario['hull_warnings']} of "
            f"{report['n_panels_requested']} panels warned that a pre-period "
            "point was outside the donor range. Panels were not dropped."
        )
    lines.extend(
        [
            "",
            _reference_hull_sentence(report),
            "",
            "## Comparison",
            "",
            (
                "Curve probabilities use cumulative effects 0, 2, and 8. "
                "Audit rates are the audit half, panels 50 through 99, at "
                "per-period lifts 0, 0.25, and 1. The detected column above "
                "counts all 100 panels, so the two rates are not the same "
                "denominator. Paired probabilities are a Normal restatement "
                "of each audit panel's own posterior mean and standard "
                "deviation. Agreement with the audit rate is not validation "
                "and is not transport."
            ),
            "",
            (
                "| Scenario | Curve p_detect | Audit rates | Paired mean p_detect | "
                "Mean-SD correlation | Divergences | Status |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for scenario in report["scenarios"]:
        comparison = scenario.get("comparison") or {}
        curve = comparison.get("curve") or {}
        audit = comparison.get("audit_empirical") or {}
        paired = (comparison.get("paired_normal") or {}).get("mean_p_detect") or {}
        lines.append(
            "| {scenario} | {curve} | {audit} | {paired} | {corr} | {div} | {status} |".format(
                scenario=scenario["scenario"],
                curve=_fmt_rates(curve.get("p_detect")),
                audit=_fmt_mapping(audit, "rate"),
                paired=_fmt_rates(list(paired.values()) if paired else None),
                corr=_fmt_optional(comparison.get("mean_sd_correlation")),
                div=comparison.get("divergences"),
                status=comparison.get("status"),
            )
        )
    historical = report.get("historical_replay") or {}
    lines.extend(
        [
            "",
            "## Historical replay",
            "",
            (
                f"Evidence level: {historical.get('evidence_level')}. "
                f"Selection windows detected {historical.get('selection_detected')} "
                f"of {historical.get('selection_scored')} scored windows at the "
                "zero lift. Binomial standard error: "
                f"{historical.get('binomial_standard_error')}. "
                f"Audit fit: {historical.get('fit_audit')}. "
                "Calibration evidence level: "
                f"{historical.get('calibration_evidence_level')}. "
                f"Curve validated: {historical.get('curve_validated')}."
            ),
            "",
            "| Origin | Role | Status | Detected |",
            "| --- | --- | --- | --- |",
        ]
    )
    for window in historical.get("windows") or []:
        lines.append(
            "| {origin} | {role} | {status} | {detected} |".format(
                origin=window.get("origin"),
                role=window.get("role"),
                status=window.get("status"),
                detected=window.get("detected"),
            )
        )
    lines.extend(["", "## Claims", ""])
    for scenario in report["scenarios"]:
        for lift in scenario["lifts"]:
            claim = lift["eligibility"]["permitted_claim"]
            if claim:
                lines.extend([f"Lift {float(lift['lift']):.2f}. {claim}", ""])
            recovery = lift["recovery"]["permitted_claim"]
            if recovery:
                lines.extend(
                    [f"Recovery at lift {float(lift['lift']):.2f}. {recovery}", ""]
                )
        comparison = scenario.get("comparison") or {}
        comparison_claim = (comparison.get("eligibility") or {}).get("permitted_claim")
        if comparison_claim:
            lines.extend([comparison_claim, ""])
    historical_claim = (historical.get("eligibility") or {}).get("permitted_claim")
    if historical_claim:
        lines.extend([historical_claim, ""])
    lines.extend(
        [
            "Numeric agreement between a curve and a rate is not validation.",
            "",
            f"Source SHA: {report['source']['git_sha']}.",
            *(
                [
                    "Product modules were at that commit. Qualification modules were untracked when the draws were generated:",
                    "",
                    "```text",
                    report["source"]["git_status"].rstrip(),
                    "```",
                    "",
                ]
                if report["source"].get("git_status")
                else []
            ),
            report["source"].get("label_note") or "",
            "",
        ]
    )
    return "\n".join(lines)


def _reference_hull_sentence(report: dict[str, Any]) -> str:
    """Name the reference hull count. Do not hardcode the confounding claim."""
    reference = next(
        scenario
        for scenario in report["scenarios"]
        if scenario["scenario"] == REFERENCE_SCENARIO
    )
    count = int(reference["hull_warnings"])
    requested = int(report["n_panels_requested"])
    if count:
        return (
            f"The reference scenario recorded {count} of {requested} "
            "convex-hull warnings. That count confounds the label, so it is "
            "not a correctly specified recovery diagnostic. The amplitude was "
            "not changed after the counts were seen."
        )
    return (
        f"The reference scenario recorded {count} of {requested} "
        "convex-hull warnings. The amplitude was not changed after the "
        "counts were seen."
    )


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.3f}"


def _fmt_rates(values: Any) -> str:
    if not values:
        return "None"
    return ", ".join(f"{float(item):.3f}" for item in values)


def _fmt_mapping(mapping: dict[str, Any], field: str) -> str:
    if not mapping:
        return "None"
    return ", ".join(f"{key}={float(item[field]):.3f}" for key, item in mapping.items())
