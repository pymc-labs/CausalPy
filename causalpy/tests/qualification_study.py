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
"""Repeated-data qualification study for issue #914.

Ordinary CI does not run this module's sampling. The nightly test does. The
committed report is the local run's evidence; a later nightly run checks the
same protocol and does not require bit-identical draws.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from causalpy.checks.placebo_in_time import (
    _fit_learned_null,
    _NullModelUnidentifiedError,
)
from causalpy.design import SyntheticControlPlanner
from causalpy.historical_validation import (
    _LEARNED_NULL_PRIOR_SCALE,
    CalibrationRecord,
    replay_history,
    validate_schedule,
)
from causalpy.tests.qualification_panels import (
    DONORS,
    TREATED,
    generate_untreated,
    historical_replay_frame,
    historical_schedule_request,
    inject_treatment,
    sampler_seed,
    scenario_code,
)
from causalpy.tests.qualification_policy import (
    EligibilityInput,
    binomial_rate,
    decide,
    operating_input_from_summary,
    recovery_input_from_summary,
)
from causalpy.tests.qualification_protocol import (
    ABSOLUTE_LIFTS,
    FOLD_SD_ASSESSMENT,
    HISTORICAL_CODE,
    PROTOCOL,
    REFERENCE_SCENARIO,
    SAMPLE_KWARGS,
    SCENARIOS,
    sample_kwargs,
)
from causalpy.tests.qualification_report import render_markdown

EVIDENCE_DIR = Path(__file__).parent / "qualification_evidence"
REPORT_JSON = EVIDENCE_DIR / "qualification_report.json"
REPORT_MD = EVIDENCE_DIR / "qualification_report.md"
_IDENTITY_TOLERANCE = 1e-6
_MONTE_CARLO_Z = 1.96


def _jsonable(value: Any) -> Any:
    """Convert numpy values so the evidence file is plain JSON."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.floating | float):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer | int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    return value


def _git(args: list[str]) -> str:
    """Read one git fact from the worktree. A failure is recorded, not hidden."""
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return f"unavailable: {completed.stderr.strip()}"
    return completed.stdout.strip()


def _source_identity() -> dict[str, str]:
    """Commit, dirty state, and package versions. Not a golden posterior."""
    packages = ("numpy", "pandas", "scipy", "pymc", "pytensor", "arviz", "causalpy")
    versions = {name: version(name) for name in packages}
    return {
        "git_sha": _git(["rev-parse", "HEAD"]),
        "git_status": _git(["status", "--porcelain"]),
        "versions": versions,
    }


def _cumulative_scores(result: Any) -> dict[float, Any]:
    """Absolute cumulative scores keyed by lift. Other estimands are unused."""
    horizon = result.horizons[0]
    found: dict[float, Any] = {}
    for score in horizon.scores:
        if score.estimand == "cumulative" and score.effect_scale == "absolute":
            found[float(score.lift)] = score
    return found


def _negative_detected(draws: np.ndarray) -> tuple[bool, float]:
    """Negative tail of the same draws the positive rule reads.

    This is not the operating-characteristics ROPE wrong-sign probability.
    At a positive lift it is the wrong-sign decision. At lift zero it is the
    other null-region tail and is not called a wrong-sign.
    """
    tail = float(np.mean(np.asarray(draws, dtype=float) < PROTOCOL.decision_threshold))
    return tail >= PROTOCOL.decision_probability, tail


def _lift_record(score: Any) -> dict[str, Any]:
    """Summarise one score and drop the draw vector."""
    draws = np.asarray(score.effect, dtype=float)
    negative, negative_tail = _negative_detected(draws)
    direct_positive = float(np.mean(draws > PROTOCOL.decision_threshold))
    if score.detected != (direct_positive >= PROTOCOL.decision_probability):
        raise RuntimeError("Study decision helper disagreed with LiftScore.detected.")
    return {
        "positive_detected": bool(score.detected),
        "negative_detected": negative,
        "positive_tail": direct_positive,
        "negative_tail": negative_tail,
        "mean": float(score.mean),
        "sd": float(np.std(draws, ddof=1)),
        "hdi_lower": float(score.hdi_lower),
        "hdi_upper": float(score.hdi_upper),
        "truth": float(score.truth),
        "truth_covers": bool(score.truth_covers),
    }


def _fit_frame(frame: Any, seed: int) -> dict[str, Any]:
    """One shipped planner fit. A failure is a counted non-detection, not a crash."""
    planner = SyntheticControlPlanner(
        frame,
        TREATED,
        DONORS,
        PROTOCOL.decision_rule(),
        estimator=PROTOCOL.estimator,
        random_seed=seed,
    )
    kwargs = sample_kwargs(seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        try:
            result = planner.evaluate(
                durations=[PROTOCOL.duration],
                absolute_lifts=list(ABSOLUTE_LIFTS),
                sample_kwargs=kwargs,
            )
        except Exception as exc:
            return {
                "status": "fit-failed",
                "fit_error": f"{type(exc).__name__}: {exc}",
                "hull_warnings": 0,
                "lifts": {},
            }
    if result.status == "fit-failed":
        return {
            "status": "fit-failed",
            "fit_error": result.fit_error,
            "hull_warnings": _hull_warning_count(caught),
            "lifts": {},
        }
    return {
        "status": result.status,
        "fit_error": None,
        "hull_warnings": _hull_warning_count(caught),
        "lifts": {
            _lift_key(lift): _lift_record(score)
            for lift, score in _cumulative_scores(result).items()
        },
    }


def _hull_warning_count(caught: list[warnings.WarningMessage]) -> int:
    return sum("convex hull" in str(item.message).lower() for item in caught)


def _lift_key(lift: float) -> str:
    return f"{lift:.2f}"


def _fit_panel(scenario: str, panel_index: int) -> dict[str, Any]:
    """Fit one untreated panel. Treatment is not injected into this fit."""
    panel = generate_untreated(scenario, panel_index)
    seed = sampler_seed(scenario_code(scenario), panel_index)
    fitted = _fit_frame(panel.untreated, seed)
    role = "calibration" if panel_index < PROTOCOL.calibration_panels else "audit"
    return {
        "panel_index": panel_index,
        "role": role,
        "data_seed_entropy": list(panel.data_seed_entropy),
        "sampler_seed": seed,
        **fitted,
    }


def translation_identity() -> dict[str, Any]:
    """Compare one translated lift with one injected treatment on the same seed.

    The planner adds the lift after the fit. If that is exact, the translated
    cumulative mean equals the injected-path mean and both equal the untreated
    mean plus lift times duration. A miss is a product defect for #721. It
    withholds nonzero detection claims. It does not stop the null study.
    """
    panel = generate_untreated(REFERENCE_SCENARIO, 0)
    seed = sampler_seed(scenario_code(REFERENCE_SCENARIO), 0)
    untreated = _fit_frame(panel.untreated, seed)
    injected_frame = inject_treatment(panel.untreated, PROTOCOL.large_lift)
    injected = _fit_frame(injected_frame, seed)
    truth_shift = PROTOCOL.cumulative_truth(PROTOCOL.large_lift)
    key = _lift_key(PROTOCOL.large_lift)
    zero = _lift_key(0.0)
    if untreated["status"] == "fit-failed" or injected["status"] == "fit-failed":
        return {
            "matches": False,
            "reason": "identity fit failed",
            "untreated_status": untreated["status"],
            "injected_status": injected["status"],
            "truth_shift": truth_shift,
        }
    null_mean = untreated["lifts"][zero]["mean"]
    translated_mean = untreated["lifts"][key]["mean"]
    injected_mean = injected["lifts"][zero]["mean"]
    translation_shift = translated_mean - null_mean
    injection_shift = injected_mean - null_mean
    matches = (
        abs(translation_shift - truth_shift) <= _IDENTITY_TOLERANCE
        and abs(injection_shift - truth_shift) <= _IDENTITY_TOLERANCE
        and abs(translated_mean - injected_mean) <= _IDENTITY_TOLERANCE
        and untreated["lifts"][key]["positive_detected"]
        == injected["lifts"][zero]["positive_detected"]
    )
    return {
        "matches": matches,
        "reason": "exact-translation"
        if matches
        else "translation-disagrees-with-injection",
        "truth_shift": truth_shift,
        "translation_shift": translation_shift,
        "injection_shift": injection_shift,
        "translated_detected": untreated["lifts"][key]["positive_detected"],
        "injected_detected": injected["lifts"][zero]["positive_detected"],
        "tolerance": _IDENTITY_TOLERANCE,
    }


def _monte_carlo_interval(values: list[float]) -> dict[str, Any]:
    """Normal interval for a Monte Carlo mean. Not a posterior interval."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count < 2:
        return {
            "n": count,
            "mean": None if count == 0 else float(array[0]),
            "interval_low": None,
            "interval_high": None,
            "covers_zero": False,
            "method": "undefined",
        }
    mean = float(array.mean())
    half = _MONTE_CARLO_Z * float(array.std(ddof=1)) / np.sqrt(count)
    low = mean - half
    high = mean + half
    return {
        "n": count,
        "mean": mean,
        "interval_low": low,
        "interval_high": high,
        "covers_zero": bool(low <= 0.0 <= high),
        "method": "normal-monte-carlo-mean",
    }


def _summarise_lift(
    scenario: str,
    lift: float,
    panels: list[dict[str, Any]],
    *,
    identity_ok: bool,
) -> dict[str, Any]:
    """Counts use the requested panel count. Failures are non-detections."""
    key = _lift_key(lift)
    failed = [panel for panel in panels if panel["status"] == "fit-failed"]
    completed = [panel for panel in panels if panel["status"] != "fit-failed"]
    detected = [
        panel for panel in completed if panel["lifts"][key]["positive_detected"]
    ]
    wrong_sign = [
        panel
        for panel in completed
        if lift > 0.0 and panel["lifts"][key]["negative_detected"]
    ]
    negative_at_null = [
        panel
        for panel in completed
        if lift == 0.0 and panel["lifts"][key]["negative_detected"]
    ]
    covered = [panel for panel in completed if panel["lifts"][key]["truth_covers"]]
    biases = [
        panel["lifts"][key]["mean"] - panel["lifts"][key]["truth"]
        for panel in completed
    ]
    rate = binomial_rate(len(detected), len(panels), independent=True)
    bias = _monte_carlo_interval(biases)
    summary = {
        "scenario": scenario,
        "estimator": PROTOCOL.estimator,
        "lift": lift,
        "cumulative_truth": PROTOCOL.cumulative_truth(lift),
        "n_requested": len(panels),
        "n_failed": len(failed),
        "n_completed": len(completed),
        "n_detected": len(detected),
        "n_wrong_sign": len(wrong_sign),
        "n_negative_at_null": len(negative_at_null),
        "n_truth_covers": len(covered),
        "coverage_denominator": len(completed),
        "failures_in_denominator": True,
        "windows_independent": True,
        "depends_on_translation": lift != 0.0,
        "translation_identity_ok": identity_ok,
        "calibration_applicable": False,
        "detection_target_applies": lift == PROTOCOL.large_lift,
        "bias_interval_covers_zero": bool(bias["covers_zero"]),
        "null_tolerance_requested": None,
        "claimed_detection_se": None,
        "launch_requested": False,
        "prospective_power_requested": False,
        "numeric_match_is_validation": False,
        "rate": rate.rate,
        "interval_low": rate.interval_low,
        "interval_high": rate.interval_high,
        "plug_in_standard_error": rate.plug_in_standard_error,
        "plug_in_standard_error_is_qualified_precision": (
            rate.plug_in_standard_error_is_qualified_precision
        ),
        "interval_method": rate.interval_method,
        "bias": bias,
    }
    decision = decide(operating_input_from_summary(summary))
    summary["eligibility"] = {
        "status": decision.status,
        "permitted_claim": decision.permitted_claim,
        "withheld": list(decision.withheld),
        "reasons": list(decision.reasons),
    }
    return summary


def _calibration_inputs(
    panels: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Raw zero-lift means and posterior SDs. A short or non-positive set is unused."""
    selected = [
        panel
        for panel in panels
        if panel["role"] == "calibration" and panel["status"] != "fit-failed"
    ]
    if len(selected) != PROTOCOL.calibration_panels:
        return None
    means = np.array(
        [panel["lifts"]["0.00"]["mean"] for panel in selected], dtype=float
    )
    sds = np.array([panel["lifts"]["0.00"]["sd"] for panel in selected], dtype=float)
    if not np.isfinite(means).all() or not np.isfinite(sds).all() or np.any(sds <= 0.0):
        return None
    return means, sds


def _record_from_samples(
    samples: np.ndarray,
    means: np.ndarray,
    sds: np.ndarray,
    *,
    reasons: tuple[str, ...],
) -> CalibrationRecord:
    """Product record. Independence stays false because the constructor requires it."""
    return CalibrationRecord(
        evidence_level="model-conditional",
        reasons=reasons,
        applicable=True,
        curve_validated=False,
        null_samples=np.ravel(samples),
        fold_sds=sds,
        fold_means=means,
        rope_half_width=0.0,
        threshold=PROTOCOL.decision_probability,
        scored_window_count=int(means.size),
        overlapping_pair_count=0,
        shared_training_pair_count=0,
        model="learned-null-normal",
    )


def _curve_at(record: CalibrationRecord, effects: list[float]) -> dict[str, Any]:
    """Read the shipped curve. Do not treat a numerical match as validation."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        curve = record.conditional_curve(
            effect_sizes=effects, mde_target=PROTOCOL.detection_target
        )
    return {
        "effect_sizes": [float(item) for item in curve.effect_sizes],
        "p_detect": [float(item) for item in curve.p_detect],
        "p_null": [float(item) for item in curve.p_null],
        "p_wrong_sign": [float(item) for item in curve.p_wrong_sign],
        "warnings": [str(item.message) for item in caught],
        "curve_validated": False,
    }


def _operating_comparison(
    scenario: str, panels: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    """Compare actual decisions with the learned-null curve. Not a pass/fail test."""
    inputs = _calibration_inputs(panels)
    assumptions = (
        "rope_half_width=0 so the engine positive rule matches the shipped tail rule",
        "threshold is the 0.95 tail probability, not the effect threshold of 0",
        "fold SDs are posterior SDs of cumulative impact, not residual SDs",
        "the engine crosses every null draw with every fold SD",
        "a calibration curve does not transport by itself",
        "the learned-null fitter floors fold SDs below 1e-6 internally; the record stores raw SDs",
    )
    base = {
        "scenario": scenario,
        "assumptions": list(assumptions),
        "calibration_applicable": False,
        "status": "inconclusive",
    }
    if inputs is None:
        base["reason"] = (
            "calibration half was incomplete or had a non-positive posterior SD"
        )
        return base
    means, sds = inputs
    try:
        idata, theta = _fit_learned_null(
            means,
            sds,
            prior_scale=_LEARNED_NULL_PRIOR_SCALE,
            sample_kwargs=dict(SAMPLE_KWARGS),
            random_seed=seed,
        )
    except _NullModelUnidentifiedError as exc:
        base["reason"] = f"learned-null unidentified: {exc}"
        return base
    except Exception as exc:
        base["reason"] = f"learned-null fit failed: {type(exc).__name__}: {exc}"
        return base
    divergences = _divergence_count(idata)
    if divergences is None or divergences > 0:
        base["reason"] = (
            "learned-null divergences were positive or could not be counted; "
            "the curve was not fed"
        )
        base["divergences"] = divergences
        return base
    samples = np.ravel(np.asarray(theta, dtype=float))
    reasons = (
        "Simulation panels are independent. windows_independent stays false "
        "because CalibrationRecord rejects independent windows. Binomial "
        "uncertainty is attached outside this record.",
        *assumptions,
    )
    record = _record_from_samples(samples, means, sds, reasons=reasons)
    effects = [PROTOCOL.cumulative_truth(lift) for lift in (0.0, *ABSOLUTE_LIFTS)]
    curve = _curve_at(record, effects)
    audit = [panel for panel in panels if panel["role"] == "audit"]
    paired = _paired_predictions(audit, effects)
    correlation = _correlation(means, sds)
    return {
        "scenario": scenario,
        "assumptions": list(assumptions),
        "calibration_applicable": True,
        "status": "model-conditional",
        "divergences": divergences,
        "n_null_samples": int(samples.size),
        "mean_sd_correlation": correlation,
        "curve": curve,
        "audit_empirical": _audit_rates(audit),
        "paired_normal": paired,
        "mixture_note": (
            "The curve crosses calibration means' learned-null draws with every "
            "calibration fold SD. The paired values use each audit panel's own "
            "mean and SD. Neither is an observation-noise standard error."
        ),
        "eligibility": _comparison_decision(scenario, assumptions),
    }


def _divergence_count(idata: Any) -> int | None:
    try:
        return int(np.asarray(idata.sample_stats["diverging"]).sum())
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _correlation(means: np.ndarray, sds: np.ndarray) -> float | None:
    if means.size < 3 or float(np.std(means)) == 0.0 or float(np.std(sds)) == 0.0:
        return None
    return float(np.corrcoef(means, sds)[0, 1])


def _paired_predictions(
    audit: list[dict[str, Any]], effects: list[float]
) -> dict[str, Any]:
    """One-panel records. A failed audit panel is omitted and counted."""
    predicted: dict[str, list[float]] = {_lift_key(effect): [] for effect in effects}
    used = 0
    for panel in audit:
        if panel["status"] == "fit-failed":
            continue
        mean = np.array([panel["lifts"]["0.00"]["mean"]], dtype=float)
        sd = np.array([panel["lifts"]["0.00"]["sd"]], dtype=float)
        if not np.isfinite(sd[0]) or sd[0] <= 0.0:
            continue
        record = _record_from_samples(
            mean,
            mean,
            sd,
            reasons=("paired audit panel; not a crossed mixture",),
        )
        curve = _curve_at(record, effects)
        for effect, rate in zip(effects, curve["p_detect"], strict=True):
            predicted[_lift_key(effect)].append(rate)
        used += 1
    return {
        "n_used": used,
        "n_audit": len(audit),
        "mean_p_detect": {
            key: (float(np.mean(values)) if values else None)
            for key, values in predicted.items()
        },
    }


def _audit_rates(audit: list[dict[str, Any]]) -> dict[str, Any]:
    """Actual audit decisions. Failures stay in the denominator."""
    rates = {}
    for lift in (0.0, *ABSOLUTE_LIFTS):
        key = _lift_key(lift)
        detected = 0
        for panel in audit:
            if (
                panel["status"] != "fit-failed"
                and panel["lifts"][key]["positive_detected"]
            ):
                detected += 1
        rates[key] = {
            "n_detected": detected,
            "n_requested": len(audit),
            "rate": detected / len(audit) if audit else None,
        }
    return rates


def _comparison_decision(scenario: str, assumptions: tuple[str, ...]) -> dict[str, Any]:
    decision = decide(
        EligibilityInput(
            kind="model-conditional-operating-comparison",
            scenario=scenario,
            estimator=PROTOCOL.estimator,
            n_requested=PROTOCOL.n_panels,
            n_failed=0,
            failures_in_denominator=True,
            windows_independent=True,
            null_tolerance_requested=None,
            claimed_detection_se=None,
            launch_requested=False,
            prospective_power_requested=False,
            numeric_match_is_validation=False,
            depends_on_translation=False,
            translation_identity_ok=True,
            calibration_applicable=True,
            detection_target_applies=False,
            bias_interval_covers_zero=False,
            assumptions_named=assumptions,
        )
    )
    return {
        "status": decision.status,
        "permitted_claim": decision.permitted_claim,
        "withheld": list(decision.withheld),
        "reasons": list(decision.reasons),
    }


def _historical_replay(seed: int) -> dict[str, Any]:
    """One generated history. Dependent windows. No binomial standard error."""
    frame = historical_replay_frame()
    request = historical_schedule_request()
    schedule = validate_schedule(frame, TREATED, DONORS, request)
    replay = replay_history(
        frame,
        TREATED,
        DONORS,
        schedule,
        PROTOCOL.decision_rule(),
        random_seed=seed,
        sample_kwargs=sample_kwargs(seed),
        fit_audit=False,
        rope_half_width=0.0,
        threshold=PROTOCOL.decision_probability,
    )
    windows = []
    for window in replay.windows:
        windows.append(
            {
                "role": window.plan.role,
                "origin": _label(window.plan.origin),
                "status": window.status,
                "detected": window.detected,
                "cumulative_mean": window.cumulative_mean,
                "cumulative_sd": window.cumulative_sd,
                "reason": window.reason,
            }
        )
    scored = [
        row
        for row in windows
        if row["role"] == "selection" and row["detected"] is not None
    ]
    detected = sum(1 for row in scored if row["detected"])
    curve_at_null = None
    if replay.calibration.applicable:
        curve_at_null = _curve_at(replay.calibration, [0.0])["p_detect"][0]
    decision = decide(
        EligibilityInput(
            kind="historical-replay",
            scenario="historical-replay",
            estimator=PROTOCOL.estimator,
            n_requested=len(windows),
            n_failed=sum(1 for row in windows if row["status"] == "fit-failed"),
            failures_in_denominator=True,
            windows_independent=False,
            null_tolerance_requested=None,
            claimed_detection_se=None,
            launch_requested=False,
            prospective_power_requested=False,
            numeric_match_is_validation=False,
            depends_on_translation=False,
            translation_identity_ok=True,
            calibration_applicable=bool(replay.calibration.applicable),
            detection_target_applies=False,
            bias_interval_covers_zero=False,
        )
    )
    return {
        "evidence_level": replay.evidence_level,
        "reasons": list(replay.reasons),
        "windows": windows,
        "selection_detected": detected,
        "selection_scored": len(scored),
        "binomial_standard_error": None,
        "calibration_applicable": bool(replay.calibration.applicable),
        "calibration_evidence_level": replay.calibration.evidence_level,
        "calibration_reasons": list(replay.calibration.reasons),
        "curve_validated": False,
        "curve_p_detect_at_zero": curve_at_null,
        "fit_audit": False,
        "audit_note": (
            "The audit origin was reserved and not fitted. That is the product's "
            "audit reservation. It is not a duration freeze and does not close "
            "the selected-duration gap."
        ),
        "eligibility": {
            "status": decision.status,
            "permitted_claim": decision.permitted_claim,
            "withheld": list(decision.withheld),
            "reasons": list(decision.reasons),
        },
    }


def _label(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _recovery_decision(
    summary: dict[str, Any], *, hull_warnings: int
) -> dict[str, Any]:
    decision = decide(recovery_input_from_summary(summary, hull_warnings=hull_warnings))
    return {
        "status": decision.status,
        "permitted_claim": decision.permitted_claim,
        "withheld": list(decision.withheld),
        "reasons": list(decision.reasons),
    }


def run_qualification(
    *, n_panels: int | None = None, progress: bool = True
) -> dict[str, Any]:
    """Run the predeclared study. A short run must not be written as evidence."""
    requested = PROTOCOL.n_panels if n_panels is None else n_panels
    if requested < 1 or requested > PROTOCOL.n_panels:
        raise ValueError(
            f"n_panels must be in 1..{PROTOCOL.n_panels}, got {requested}."
        )
    started = time.perf_counter()
    _silence_samplers()
    identity = translation_identity()
    scenarios = []
    for scenario in SCENARIOS:
        panels = []
        for index in range(requested):
            if progress and index % 10 == 0:
                print(
                    f"qualification {scenario} panel {index}/{requested}",
                    flush=True,
                )
            panels.append(_fit_panel(scenario, index))
        lifts = [
            _summarise_lift(
                scenario, lift, panels, identity_ok=bool(identity["matches"])
            )
            for lift in (0.0, *ABSOLUTE_LIFTS)
        ]
        hull_warnings = sum(panel["hull_warnings"] for panel in panels)
        for lift_summary in lifts:
            lift_summary["recovery"] = _recovery_decision(
                lift_summary, hull_warnings=hull_warnings
            )
        comparison = None
        if requested == PROTOCOL.n_panels:
            comparison = _operating_comparison(
                scenario,
                panels,
                sampler_seed(scenario_code(scenario), PROTOCOL.n_panels),
            )
        scenarios.append(
            {
                "scenario": scenario,
                "panels": panels,
                "lifts": lifts,
                "comparison": comparison,
                "hull_warnings": sum(panel["hull_warnings"] for panel in panels),
            }
        )
    historical = None
    if requested == PROTOCOL.n_panels:
        historical = _historical_replay(sampler_seed(HISTORICAL_CODE, 0))
    elapsed = time.perf_counter() - started
    defects = []
    if not identity["matches"]:
        defects.append(
            {
                "id": "translation-identity",
                "returned_to": 721,
                "summary": (
                    "The injected-path cumulative mean did not match the "
                    "translated lift on the same untreated fit."
                ),
            }
        )
    return _jsonable(
        {
            "protocol_id": PROTOCOL.protocol_id,
            "n_panels_requested": requested,
            "short_study": requested != PROTOCOL.n_panels,
            "elapsed_seconds": elapsed,
            "source": _source_identity(),
            "sample_kwargs": SAMPLE_KWARGS,
            "selected_duration_audit": PROTOCOL.selected_duration_audit,
            "null_tolerance": PROTOCOL.null_error_tolerance,
            "null_tolerance_reason": (
                "No numeric null-error tolerance is claimed. The 94% HDI is an "
                "interval convention, and the 0.95 tail probability is the "
                "decision rule. Neither is a false-positive tolerance."
            ),
            "launch_recommendation_qualified": False,
            "translation_identity": identity,
            "product_defects": defects,
            "scenarios": scenarios,
            "historical_replay": historical,
            "fold_sd_assessment": FOLD_SD_ASSESSMENT,
        }
    )


def _silence_samplers() -> None:
    """Keep a 300-fit log readable. Does not hide a Python exception."""
    logging.getLogger("pymc").setLevel(logging.ERROR)
    logging.getLogger("pytensor").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="Only .* samples")
    warnings.filterwarnings("ignore", message="The number of samples is too small")


def write_evidence(report: dict[str, Any], directory: Path | None = None) -> Path:
    """Write the 100-panel report. A short study is refused."""
    if report["short_study"] or report["n_panels_requested"] != PROTOCOL.n_panels:
        raise ValueError("Refusing to write a short study as qualification evidence.")
    target = EVIDENCE_DIR if directory is None else directory
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "qualification_report.json"
    md_path = target / "qualification_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path


def load_committed_report() -> dict[str, Any]:
    """Load the committed evidence. Missing evidence is a test failure."""
    if not REPORT_JSON.is_file():
        raise FileNotFoundError(
            f"Committed qualification evidence is missing: {REPORT_JSON}"
        )
    return json.loads(REPORT_JSON.read_text(encoding="utf-8"))
