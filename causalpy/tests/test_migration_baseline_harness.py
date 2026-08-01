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
"""Tests for the permanent PyMC migration baseline comparison harness."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pymc as pm
import pytest
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "migration_baseline" / "harness.py"

# The shared ``conftest`` registers ``mock_pymc_sample`` at *session* scope, so
# once any earlier test in the run triggers it, ``pymc.sample`` (and
# ``pymc.Flat`` / ``pymc.HalfFlat``) stay monkeypatched to the fast
# prior-predictive mock for the remainder of the session. That mock produces no
# ``sample_stats`` group, which the baseline harness requires, so the capture
# test below would fail purely because of suite ordering. Capture the genuine
# callables at import time -- collection always completes before any fixture
# runs -- so the capture test can restore real reduced MCMC regardless of which
# tests ran before it.
_GENUINE_PYMC_SAMPLING = {
    name: getattr(pm, name) for name in ("sample", "Flat", "HalfFlat")
}


@pytest.fixture
def genuine_pymc_sampling():
    """Force real PyMC sampling for a test despite the session-scoped mock."""
    saved = {name: getattr(pm, name) for name in _GENUINE_PYMC_SAMPLING}
    for name, value in _GENUINE_PYMC_SAMPLING.items():
        setattr(pm, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(pm, name, value)


# Decoded artifact JSON: a heterogeneous, arbitrarily nested mapping that the harness itself types as ``dict[str, Any]`` and validates at runtime. These tests deliberately reach into and mutate that structure at arbitrary depths, including inserting keys the schema does not register, so a TypedDict would reject the very tampering being asserted.
ArtifactPayload = dict[str, Any]


def _load_harness_module():
    spec = importlib.util.spec_from_file_location(
        "migration_baseline_harness", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _summary(
    *,
    mean: float,
    posterior_sd: float,
    mcse_mean: float,
    hdi_lower: float,
    hdi_upper: float,
) -> dict[str, float]:
    return {
        "mean": mean,
        "posterior_sd": posterior_sd,
        "mcse_mean": mcse_mean,
        "hdi_lower": hdi_lower,
        "hdi_upper": hdi_upper,
    }


def _valid_artifact(
    harness,
    tmp_path: Path,
    *,
    stack: str,
    role: str,
    capture_id: str,
    batch_id: str = "00000000-0000-4000-8000-000000000000",
    harness_sha256: str = "a" * 64,
) -> ArtifactPayload:
    """Build schema-complete evidence without importing a sampling stack."""
    root = (tmp_path / stack / "checkout").resolve()
    prefix = (tmp_path / stack / "prefix").resolve()
    expected_major = harness.STACK_RUNTIME_MAJORS[stack]
    dependencies = {
        "arviz": f"{expected_major['arviz']}.0.0",
        "causalpy": "0.8.0",
        "numpy": "2.0.0",
        "pandas": "3.0.0",
        "pymc": f"{expected_major['pymc']}.0.0",
        "pytensor": f"{expected_major['pytensor']}.0.0",
        "xarray": "2026.1.0",
    }
    cases = []
    for case_name, manifest in harness._scenario_manifest(
        harness.REGISTERED_SAMPLING
    ).items():
        series = []
        for series_name, series_manifest in manifest["series"].items():
            metrics = []
            for metric in series_manifest["metrics"]:
                metrics.append(
                    {
                        **copy.deepcopy(metric),
                        "draw_digest": "d" * 64,
                        "summary": {
                            "mean": 1.0,
                            "posterior_sd": 1.0,
                            "mcse_mean": 0.01,
                            "rhat": 1.0,
                            "ess_bulk": 800.0,
                            "ess_tail": 800.0,
                            "hdi_lower": 0.0,
                            "hdi_upper": 2.0,
                        },
                    }
                )
            series.append(
                {
                    "name": series_name,
                    "semantics": copy.deepcopy(series_manifest["semantics"]),
                    "metrics": metrics,
                }
            )
        cases.append(
            {
                "name": case_name,
                "fixture": copy.deepcopy(manifest["fixture"]),
                "sampling_quality": {
                    "divergences": 0,
                    "tree_depth_source": "tree_depth",
                    "tree_depth_events": 0,
                    "max_observed_tree_depth": 7,
                    "finite_values": True,
                },
                "effect_summary": copy.deepcopy(manifest["effect_summary"]),
                "counterfactual": copy.deepcopy(manifest["counterfactual"]),
                "series": series,
            }
        )
    return {
        "schema_version": harness.ARTIFACT_SCHEMA_VERSION,
        "suite": harness.SUITE_NAME,
        "provenance": {
            "stack": stack,
            "expected_commit": harness.STACK_COMMITS[stack],
            "actual_commit": harness.STACK_COMMITS[stack],
            "repo_root": str(root),
            "causalpy_path": str(root / "causalpy" / "__init__.py"),
            "checkout_clean": True,
            "capture_role": role,
            "capture_ordinal": harness.CAPTURE_ROLES[role][1],
            "capture_id": capture_id,
            "batch_id": batch_id,
            "harness_path": str(tmp_path / "harness.py"),
            "harness_sha256": harness_sha256,
            "harness_commit": "b" * 40,
            "harness_git_blob_sha256": harness_sha256,
            "harness_checkout_clean": True,
            "python": "3.12.0",
            "python_implementation": "CPython",
            "platform": "Darwin-25.0.0-arm64",
            "machine": "arm64",
            "dependencies": dependencies,
            "runtime": {
                "executable": str(prefix / "bin" / "python"),
                "prefix": str(prefix),
                "module_paths": {
                    name: str(prefix / "lib" / f"{name}.py")
                    for name in (
                        "arviz",
                        "numpy",
                        "pandas",
                        "pymc",
                        "pytensor",
                        "xarray",
                    )
                },
                "causalpy_editable_target": str(root),
            },
        },
        "protocol": harness._protocol(True, harness.REGISTERED_SAMPLING),
        "cases": cases,
    }


def _valid_artifact_batch(harness, tmp_path: Path) -> tuple[ArtifactPayload, ...]:
    """Build four role-complete artifacts for comparator integrity tests."""
    capture_ids = (
        "00000000-0000-4000-8000-000000000001",
        "00000000-0000-4000-8000-000000000002",
        "00000000-0000-4000-8000-000000000003",
        "00000000-0000-4000-8000-000000000004",
    )
    roles = (
        "reference_first",
        "reference_second",
        "candidate_first",
        "candidate_second",
    )
    stacks = ("pymc5", "pymc5", "pymc6", "pymc6")
    return tuple(
        _valid_artifact(
            harness,
            tmp_path,
            stack=stack,
            role=role,
            capture_id=capture_id,
        )
        for stack, role, capture_id in zip(stacks, roles, capture_ids, strict=True)
    )


def _fake_harness_identity() -> dict[str, str | bool]:
    """Provide the clean comparator identity expected by synthetic artifacts."""
    return {
        "path": "/comparator/harness.py",
        "sha256": "a" * 64,
        "commit": "b" * 40,
        "git_blob_sha256": "a" * 64,
        "checkout_clean": True,
    }


def test_registered_mean_gates_accept_small_independent_drift() -> None:
    """The comparator combines independent MCSE values and pooled posterior SDs."""
    harness = _load_harness_module()
    reference = _summary(
        mean=5.0,
        posterior_sd=2.0,
        mcse_mean=0.1,
        hdi_lower=1.0,
        hdi_upper=9.0,
    )
    candidate = _summary(
        mean=5.1,
        posterior_sd=2.0,
        mcse_mean=0.1,
        hdi_lower=1.1,
        hdi_upper=9.1,
    )

    result = harness._compare_scalar_summaries(reference, candidate)

    assert result["absolute_mean_gate_passed"]
    assert result["standardized_mean_drift_gate_passed"]
    assert math.isclose(result["combined_mcse"], math.hypot(0.1, 0.1))
    assert math.isclose(result["standardized_mean_drift"], 0.05)


def test_standardized_gate_rejects_drift_hidden_by_large_mcse() -> None:
    """Large MCSE must not make a substantively large posterior shift acceptable."""
    harness = _load_harness_module()
    reference = _summary(
        mean=5.0,
        posterior_sd=2.0,
        mcse_mean=0.1,
        hdi_lower=1.0,
        hdi_upper=9.0,
    )
    candidate = _summary(
        mean=5.5,
        posterior_sd=2.0,
        mcse_mean=0.1,
        hdi_lower=1.5,
        hdi_upper=9.5,
    )

    result = harness._compare_scalar_summaries(reference, candidate)

    assert result["absolute_mean_gate_passed"]
    assert not result["standardized_mean_drift_gate_passed"]
    assert math.isclose(result["standardized_mean_drift"], 0.25)


def test_degenerate_posterior_uses_absolute_gate_without_nan() -> None:
    """A zero pooled SD uses the registered absolute rule instead of a NaN ratio."""
    harness = _load_harness_module()
    reference = _summary(
        mean=0.0,
        posterior_sd=0.0,
        mcse_mean=0.0,
        hdi_lower=0.0,
        hdi_upper=0.0,
    )
    candidate = _summary(
        mean=2e-6,
        posterior_sd=0.0,
        mcse_mean=0.0,
        hdi_lower=2e-6,
        hdi_upper=2e-6,
    )

    result = harness._compare_scalar_summaries(reference, candidate)

    assert result["standardized_mean_drift"] is None
    assert not result["absolute_mean_gate_passed"]
    assert not result["standardized_mean_drift_gate_passed"]


def test_hdi_containment_remains_diagnostic_only() -> None:
    """The returned containment flag is separate from the two numerical hard gates."""
    harness = _load_harness_module()
    reference = _summary(
        mean=0.0,
        posterior_sd=10.0,
        mcse_mean=1.0,
        hdi_lower=-0.01,
        hdi_upper=0.01,
    )
    candidate = _summary(
        mean=0.1,
        posterior_sd=10.0,
        mcse_mean=1.0,
        hdi_lower=0.09,
        hdi_upper=0.11,
    )

    result = harness._compare_scalar_summaries(reference, candidate)

    assert result["absolute_mean_gate_passed"]
    assert result["standardized_mean_drift_gate_passed"]
    assert not result["mutual_hdi_containment_diagnostic"]


def test_pymc5_singleton_effect_is_canonicalized_before_table_binding() -> None:
    """A PyMC 5-shaped singleton effect must yield the scalar table selector."""
    harness = _load_harness_module()
    sampling = harness.REGISTERED_SAMPLING
    pymc5_effect = xr.DataArray(
        np.arange(sampling.chains * sampling.draws, dtype=float).reshape(
            sampling.chains, sampling.draws, 1
        ),
        dims=("chain", "draw", "treated_units"),
        coords={
            "chain": np.arange(sampling.chains),
            "draw": np.arange(sampling.draws),
            "treated_units": ["unit_0"],
        },
        name="beta",
    )

    class StableArviZ:
        @staticmethod
        def hdi(values, *, prob):
            assert prob == harness.HDI_PROB
            return np.array([np.min(values), np.max(values)])

        @staticmethod
        def mcse(_values, *, method):
            assert method == "mean"
            return 0.01

        @staticmethod
        def rhat(_values, *, method):
            assert method == "rank"
            return 1.0

        @staticmethod
        def ess(_values, *, method, prob=None):
            if method == "tail":
                assert prob == harness.TAIL_ESS_PROB
            else:
                assert method == "bulk"
                assert prob is None
            return 800.0

    captured = harness._capture_series(
        "did.causal_impact",
        harness._canonical_scalar_effect(pymc5_effect),
        StableArviZ,
        np,
        sampling,
    )

    assert captured["semantics"]["dims"] == ["chain", "draw"]
    assert captured["metrics"][0]["selector"] == {}
    assert harness._series_metric(captured, {})["id"] == "did.causal_impact"

    multi_unit_effect = xr.concat(
        [pymc5_effect, pymc5_effect], dim="treated_units"
    ).assign_coords(treated_units=["unit_0", "unit_1"])
    with pytest.raises(harness.HarnessError, match="unexpected non-sample"):
        harness._canonical_scalar_effect(multi_unit_effect)


def test_tail_ess_probability_is_explicit_across_arviz_api_variants() -> None:
    """Tail ESS must not inherit a version-dependent default probability."""
    harness = _load_harness_module()
    protocol = harness._protocol(False, harness.REGISTERED_SAMPLING)
    assert protocol["evidence_validity"]["tail_ess_prob"] == [
        0.05,
        0.95,
    ]
    calls = []

    class ArviZOne:
        @staticmethod
        def ess(_draws, *, method, prob):
            calls.append(("arviz-one", method, prob))
            return 801.0

    class LegacyArviZ:
        @staticmethod
        def ess(_draws, *, method, prob=None):
            calls.append(("legacy", method, prob))
            return 802.0

    assert harness._tail_ess(object(), ArviZOne) == 801.0
    assert harness._tail_ess(object(), LegacyArviZ) == 802.0
    assert calls == [
        ("arviz-one", "tail", (0.05, 0.95)),
        ("legacy", "tail", (0.05, 0.95)),
    ]


def test_capture_rejects_dirty_pinned_checkout_before_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pinned revision alone cannot make uncommitted source attributable."""
    harness = _load_harness_module()

    def fake_run_git(_repo_root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return harness.STACK_COMMITS["pymc6"]
        assert arguments == ("status", "--porcelain")
        return " M causalpy/experiments/diff_in_diff.py"

    def unexpected_import(_repo_root: Path) -> None:
        pytest.fail("dirty source must be rejected before importing CausalPy")

    monkeypatch.setattr(harness, "_run_git", fake_run_git)
    monkeypatch.setattr(harness, "_import_capture_dependencies", unexpected_import)

    with pytest.raises(harness.HarnessError, match="checkout must be clean"):
        harness._capture_artifact(
            "pymc6",
            tmp_path,
            batch_id="00000000-0000-4000-8000-000000000000",
            capture_role="candidate_first",
        )


def test_did_capture_fixture_passes_constructor_validation_without_mcmc() -> None:
    """The pinned DiD fixture reaches real constructor validation with OLS only."""
    harness = _load_harness_module()
    records_json = json.dumps(harness._records_payload(harness.DID_RECORDS))
    script = "\n".join(
        [
            "import json",
            "import pandas as pd",
            "import causalpy as cp",
            "from sklearn.linear_model import LinearRegression",
            f"data = pd.DataFrame(json.loads({records_json!r}))",
            'assert set(data.loc[data["group"] == 0, "unit"]) == {"control"}',
            'assert set(data.loc[data["group"] == 1, "unit"]) == {"treated"}',
            "result = cp.DifferenceInDifferences(",
            "    data,",
            '    formula="y ~ 1 + group * post_treatment",',
            '    time_variable_name="t",',
            '    group_variable_name="group",',
            "    model=LinearRegression(),",
            ")",
            "assert result.causal_impact is not None",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _report_capture_evidence(
    *,
    stack: str,
    fixture_sha256: str,
    capture_id: str,
) -> ArtifactPayload:
    """Build one concise capture-evidence record for report rendering."""
    return {
        "provenance": {
            "capture_id": capture_id,
            "checkout_clean": True,
            "causalpy_path": f"/checkouts/{stack}/causalpy/__init__.py",
            "harness_sha256": "harness-sha256",
            "harness_commit": "harness-commit",
            "python": "3.12.0",
            "python_implementation": "CPython",
            "platform": "Darwin-25.0.0-arm64",
            "machine": "arm64",
            "runtime": {
                "prefix": f"/prefixes/{stack}",
                "executable": f"/prefixes/{stack}/bin/python",
                "causalpy_editable_target": f"/checkouts/{stack}",
                "module_paths": {
                    name: f"/prefixes/{stack}/lib/{name}.py"
                    for name in (
                        "arviz",
                        "numpy",
                        "pandas",
                        "pymc",
                        "pytensor",
                        "xarray",
                    )
                },
            },
            "dependencies": {
                "arviz": "0.22.0",
                "pymc": "6.2.0",
                "pytensor": "2.37.0",
            },
        },
        "cases": [
            {
                "name": "difference_in_differences",
                "fixture_sha256": fixture_sha256,
                "sampling_quality": {
                    "divergences": 0,
                    "tree_depth_source": "tree_depth",
                    "tree_depth_events": 0,
                    "max_observed_tree_depth": 7,
                    "finite_values": True,
                },
                "metric_count": 3,
                "max_rhat": 1.001,
                "min_ess_bulk": 800.0,
                "min_ess_tail": 750.0,
            }
        ],
    }


def test_report_records_artifact_provenance_and_observed_validity() -> None:
    """Generated attachments must include every REPORT_TEMPLATE evidence field."""
    harness = _load_harness_module()
    capture_ids = {
        "reference_first": "capture-reference-first",
        "reference_second": "capture-reference-second",
        "candidate_first": "capture-candidate-first",
        "candidate_second": "capture-candidate-second",
    }
    evidence = {
        "reference_first": _report_capture_evidence(
            stack="pymc5",
            fixture_sha256="fixture-reference-first",
            capture_id=capture_ids["reference_first"],
        ),
        "reference_second": _report_capture_evidence(
            stack="pymc5",
            fixture_sha256="fixture-reference-second",
            capture_id=capture_ids["reference_second"],
        ),
        "candidate_first": _report_capture_evidence(
            stack="pymc6",
            fixture_sha256="fixture-candidate-first",
            capture_id=capture_ids["candidate_first"],
        ),
        "candidate_second": _report_capture_evidence(
            stack="pymc6",
            fixture_sha256="fixture-candidate-second",
            capture_id=capture_ids["candidate_second"],
        ),
    }
    comparison = {
        "passed": True,
        "reference": {"stack": "pymc5", "commit": "reference-commit"},
        "candidate": {"stack": "pymc6", "commit": "candidate-commit"},
        "comparator": {
            "harness_path": "/harness.py",
            "harness_commit": "harness-commit",
            "harness_sha256": "harness-sha256",
            "harness_git_blob_sha256": "harness-blob-sha256",
            "harness_checkout_clean": True,
        },
        "capture_batch": {
            "batch_id": "batch-id",
            "capture_ids": capture_ids,
        },
        "cross_stack_runtime": {
            "distinct_prefixes": True,
            "same_platform": True,
            "same_machine": True,
            "same_python": True,
            "same_python_implementation": True,
            "same_numpy": True,
            "same_pandas": True,
            "same_xarray": True,
            "passed": True,
        },
        "within_stack_repeatability": {
            "reference": {"metric_count": 3, "passed": True},
            "candidate": {"metric_count": 3, "passed": True},
        },
        "hard_gates": {
            "protocol_equal": True,
            "same_executing_harness_version": True,
            "fresh_capture_batch_integrity": True,
            "isolated_stack_runtime": True,
            "within_stack_repeatability": True,
            "semantic_table_and_coordinate_equality": True,
            "absolute_mean_delta": True,
            "standardized_mean_drift": True,
        },
        "artifact_inputs": [
            {
                "role": "reference_first",
                "path": "/evidence/pymc5-run-1.json",
                "sha256": "artifact-reference-first",
            },
            {
                "role": "reference_second",
                "path": "/evidence/pymc5-run-2.json",
                "sha256": "artifact-reference-second",
            },
            {
                "role": "candidate_first",
                "path": "/evidence/pymc6-run-1.json",
                "sha256": "artifact-candidate-first",
            },
            {
                "role": "candidate_second",
                "path": "/evidence/pymc6-run-2.json",
                "sha256": "artifact-candidate-second",
            },
        ],
        "capture_evidence": evidence,
        "cases": [
            {
                "name": "difference_in_differences",
                "semantic_equality": {"passed": True, "failures": []},
                "metrics": [
                    {
                        "id": "did.causal_impact",
                        "reference_mean": 1.0,
                        "candidate_mean": 1.001,
                        "absolute_mean_delta": 0.001,
                        "absolute_tolerance": 0.01,
                        "standardized_mean_drift": 0.001,
                        "reference_hdi": [0.8, 1.2],
                        "candidate_hdi": [0.81, 1.21],
                        "mutual_hdi_containment_diagnostic": True,
                        "passed": True,
                    }
                ],
            }
        ],
    }

    report = harness.render_report(comparison)

    assert "| Reference 0.94 HDI | Candidate 0.94 HDI |" in report
    assert "| [0.8, 1.2] | [0.81, 1.21] |" in report
    assert "## Comparator identity" in report
    assert "harness-commit" in report
    assert "harness-blob-sha256" in report
    assert "Immutable scenario manifest version" in report
    assert "## Capture provenance" in report
    assert "`/evidence/pymc5-run-1.json`" in report
    assert "`artifact-reference-first`" in report
    assert "All sampled checkouts recorded an empty" in report
    assert "True" in report
    assert "## Capture validity evidence" in report
    assert "`fixture-candidate-second`" in report
    assert "tree_depth / 0 / 7" in report
    assert "Finite values" in report
    assert "cores=1" in report
    assert "maximum tree depth `12`" in report
    assert "mandatory on local macOS" in report
    assert "Imported module paths:" in report
    assert "Matching shared NumPy, pandas, and xarray versions: pass" in report
    assert "750" in report


def test_artifact_manifest_rejects_deleted_output_and_substituted_fixture(
    tmp_path: Path,
) -> None:
    """Self-consistent evidence cannot omit an output or replace fixed inputs."""
    harness = _load_harness_module()
    artifact = _valid_artifact(
        harness,
        tmp_path,
        stack="pymc5",
        role="reference_first",
        capture_id="00000000-0000-4000-8000-000000000001",
    )
    harness._validate_artifact(artifact)

    missing_series = copy.deepcopy(artifact)
    missing_series["cases"][0]["series"].pop()
    with pytest.raises(harness.HarnessError, match="series differ from manifest"):
        harness._validate_artifact(missing_series)

    unknown_case_field = copy.deepcopy(artifact)
    unknown_case_field["cases"][0]["unexpected"] = "cannot be ignored"
    with pytest.raises(harness.HarnessError, match="invalid keys"):
        harness._validate_artifact(unknown_case_field)

    substituted_fixture = copy.deepcopy(artifact)
    substituted_fixture["cases"][0]["fixture"]["records"][0]["y"] = 999.0
    with pytest.raises(harness.HarnessError, match="fixture hash does not match"):
        harness._validate_artifact(substituted_fixture)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("draw_digest", "g" * 64, "invalid draw digest"),
        ("posterior_sd", -1.0, "negative posterior SD"),
        ("mcse_mean", -0.01, "negative MCSE"),
        ("rhat", False, "must be a real number"),
    ],
)
def test_artifact_validation_rejects_impossible_serialized_diagnostics(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    """Untrusted JSON cannot turn malformed diagnostics into accepted evidence."""
    harness = _load_harness_module()
    artifact = _valid_artifact(
        harness,
        tmp_path,
        stack="pymc5",
        role="reference_first",
        capture_id="00000000-0000-4000-8000-000000000001",
    )
    metric = artifact["cases"][0]["series"][0]["metrics"][0]
    if field == "draw_digest":
        metric[field] = value
    else:
        metric["summary"][field] = value

    with pytest.raises(harness.HarnessError, match=message):
        harness._validate_artifact(artifact)


def test_artifact_validation_rejects_self_reported_tree_depth_saturation(
    tmp_path: Path,
) -> None:
    """A zero event count cannot conceal a saturated observed tree depth."""
    harness = _load_harness_module()
    artifact = _valid_artifact(
        harness,
        tmp_path,
        stack="pymc5",
        role="reference_first",
        capture_id="00000000-0000-4000-8000-000000000001",
    )
    quality = artifact["cases"][0]["sampling_quality"]
    quality["tree_depth_source"] = "tree_depth"
    quality["tree_depth_events"] = 0
    quality["max_observed_tree_depth"] = harness.REGISTERED_SAMPLING.max_treedepth

    with pytest.raises(harness.HarnessError, match="tree-depth saturation"):
        harness._validate_artifact(artifact)


def test_capture_runtime_rejects_wrong_pymc_major_before_sampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A mislabeled PyMC 5 environment cannot begin a PyMC 6 capture."""
    harness = _load_harness_module()
    prefix = tmp_path / "prefix"
    repo_root = tmp_path / "checkout"

    def module(version: str, name: str):
        return type(
            name,
            (),
            {
                "__version__": version,
                "__file__": str(prefix / "lib" / f"{name}.py"),
            },
        )

    dependencies = {
        "az": module("1.0.0", "arviz"),
        "cp": module("0.8.0", "causalpy"),
        "np": module("2.0.0", "numpy"),
        "pd": module("3.0.0", "pandas"),
        "pm": module("5.0.0", "pymc"),
        "pt": module("3.0.0", "pytensor"),
        "xr": module("2026.1.0", "xarray"),
    }
    monkeypatch.setattr(harness.sys, "prefix", str(prefix))
    monkeypatch.setattr(harness.sys, "executable", str(prefix / "bin" / "python"))
    monkeypatch.setattr(harness, "_editable_causalpy_target", lambda: repo_root)

    with pytest.raises(harness.HarnessError, match="requires imported pymc major 6"):
        harness._capture_runtime_provenance("pymc6", dependencies, repo_root)


def test_comparator_requires_the_executing_harness_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Four matching self-reported digests cannot satisfy a changed comparator."""
    harness = _load_harness_module()
    artifacts = list(_valid_artifact_batch(harness, tmp_path))
    artifacts[0]["provenance"]["harness_sha256"] = "c" * 64
    artifacts[0]["provenance"]["harness_git_blob_sha256"] = "c" * 64
    monkeypatch.setattr(harness, "_harness_identity", _fake_harness_identity)

    with pytest.raises(harness.HarnessError, match="executing comparator"):
        harness.compare_artifacts(*artifacts)


def test_comparator_requires_the_executing_harness_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Matching source bytes alone cannot bind evidence to a different commit."""
    harness = _load_harness_module()
    artifacts = list(_valid_artifact_batch(harness, tmp_path))
    artifacts[0]["provenance"]["harness_commit"] = "c" * 40
    monkeypatch.setattr(harness, "_harness_identity", _fake_harness_identity)

    with pytest.raises(harness.HarnessError, match="harness_commit"):
        harness.compare_artifacts(*artifacts)


def test_comparator_requires_four_unique_role_bound_capture_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Copied evidence cannot impersonate an independent capture process."""
    harness = _load_harness_module()
    artifacts = list(_valid_artifact_batch(harness, tmp_path))
    artifacts[3]["provenance"]["capture_id"] = artifacts[2]["provenance"]["capture_id"]
    monkeypatch.setattr(harness, "_harness_identity", _fake_harness_identity)

    with pytest.raises(harness.HarnessError, match="distinct capture IDs"):
        harness.compare_artifacts(*artifacts)


def test_cross_stack_prefix_reuse_fails_the_runtime_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A comparison cannot call one reused prefix an isolated migration run."""
    harness = _load_harness_module()
    artifacts = list(_valid_artifact_batch(harness, tmp_path))
    reference_prefix = artifacts[0]["provenance"]["runtime"]["prefix"]
    for artifact in artifacts[2:]:
        runtime = artifact["provenance"]["runtime"]
        old_prefix = runtime["prefix"]
        runtime["prefix"] = reference_prefix
        runtime["executable"] = runtime["executable"].replace(
            old_prefix, reference_prefix
        )
        runtime["module_paths"] = {
            name: path.replace(old_prefix, reference_prefix)
            for name, path in runtime["module_paths"].items()
        }
    monkeypatch.setattr(harness, "_harness_identity", _fake_harness_identity)

    comparison = harness.compare_artifacts(*artifacts)

    assert not comparison["cross_stack_runtime"]["distinct_prefixes"]
    assert not comparison["hard_gates"]["isolated_stack_runtime"]
    assert not comparison["passed"]


def test_same_stack_summary_mismatch_fails_repeatability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same raw draws cannot hide a changed serialized posterior summary."""
    harness = _load_harness_module()
    artifacts = list(_valid_artifact_batch(harness, tmp_path))
    summary = artifacts[1]["cases"][0]["series"][0]["metrics"][0]["summary"]
    summary["mean"] = 1.001
    monkeypatch.setattr(harness, "_harness_identity", _fake_harness_identity)

    comparison = harness.compare_artifacts(*artifacts)

    repeatability = comparison["within_stack_repeatability"]["reference"]
    assert not repeatability["passed"]
    assert repeatability["mismatched_summary_metric_ids"] == ["did.causal_impact"]
    assert not comparison["passed"]


def test_same_stack_sampling_quality_mismatch_fails_repeatability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same posterior draws cannot hide changed capture-validity evidence."""
    harness = _load_harness_module()
    artifacts = list(_valid_artifact_batch(harness, tmp_path))
    artifacts[1]["cases"][0]["sampling_quality"]["max_observed_tree_depth"] = 8
    monkeypatch.setattr(harness, "_harness_identity", _fake_harness_identity)

    comparison = harness.compare_artifacts(*artifacts)

    repeatability = comparison["within_stack_repeatability"]["reference"]
    assert not repeatability["passed"]
    assert repeatability["mismatched_sampling_quality_cases"] == [
        "difference_in_differences"
    ]
    assert not comparison["passed"]


def test_cross_stack_shared_dependency_drift_fails_runtime_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A shared NumPy change cannot be labeled solely as a PyMC migration."""
    harness = _load_harness_module()
    artifacts = list(_valid_artifact_batch(harness, tmp_path))
    for artifact in artifacts[2:]:
        artifact["provenance"]["dependencies"]["numpy"] = "2.1.0"
    monkeypatch.setattr(harness, "_harness_identity", _fake_harness_identity)

    comparison = harness.compare_artifacts(*artifacts)

    assert not comparison["cross_stack_runtime"]["same_numpy"]
    assert not comparison["hard_gates"]["isolated_stack_runtime"]
    assert not comparison["passed"]


def test_cross_stack_python_drift_fails_runtime_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A changed interpreter cannot be labeled solely as a PyMC migration."""
    harness = _load_harness_module()
    artifacts = list(_valid_artifact_batch(harness, tmp_path))
    for artifact in artifacts[2:]:
        artifact["provenance"]["python"] = "3.13.0"
    monkeypatch.setattr(harness, "_harness_identity", _fake_harness_identity)

    comparison = harness.compare_artifacts(*artifacts)

    assert not comparison["cross_stack_runtime"]["same_python"]
    assert not comparison["hard_gates"]["isolated_stack_runtime"]
    assert not comparison["passed"]


def test_artifact_loader_rejects_duplicate_keys_and_reads_each_input_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reported artifact digests must be derived from the decision byte buffers."""
    harness = _load_harness_module()
    duplicate_key_path = tmp_path / "duplicate.json"
    duplicate_key_path.write_text(
        '{"schema_version": 2, "schema_version": 2}', encoding="utf-8"
    )
    with pytest.raises(harness.HarnessError, match="duplicate JSON key"):
        harness._read_artifact_input(duplicate_key_path)

    artifacts = _valid_artifact_batch(harness, tmp_path)
    paths = [tmp_path / f"artifact-{index}.json" for index in range(len(artifacts))]
    for path, artifact in zip(paths, artifacts, strict=True):
        path.write_text(json.dumps(artifact), encoding="utf-8")
    original_read_bytes = Path.read_bytes
    reads: list[Path] = []

    def counting_read_bytes(path: Path) -> bytes:
        if path in paths:
            reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    inputs = harness._load_distinct_artifacts(paths)
    metadata = harness._artifact_input_metadata(inputs)

    assert reads == paths
    assert [item["sha256"] for item in metadata] == [
        artifact_input.sha256 for artifact_input in inputs
    ]


def test_capture_command_rejects_existing_destination_before_sampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed rerun cannot silently preserve or replace stale evidence."""
    harness = _load_harness_module()
    output = tmp_path / "evidence" / "pymc5-run-1.json"
    output.parent.mkdir()
    output.write_text("stale", encoding="utf-8")
    args = argparse.Namespace(
        stack="pymc5",
        capture_role="reference_first",
        batch_id="00000000-0000-4000-8000-000000000000",
        repo_root=tmp_path / "checkout",
        output=output,
    )

    def unexpected_capture(*_args, **_kwargs):
        pytest.fail("an existing destination must be rejected before sampling")

    monkeypatch.setattr(harness, "_capture_artifact", unexpected_capture)

    with pytest.raises(harness.HarnessError, match="already exists"):
        harness._capture_command(args)


def test_registered_manifest_sample_coordinates_track_the_registered_protocol() -> None:
    """The fixed output manifest must describe the registered posterior size."""
    harness = _load_harness_module()
    sampling = harness.REGISTERED_SAMPLING
    manifest = harness._scenario_manifest(sampling)

    for case in manifest.values():
        for series in case["series"].values():
            coords = series["semantics"]["coords"]
            assert coords["chain"] == list(range(sampling.chains))
            assert coords["draw"] == list(range(sampling.draws))
            assert series["semantics"]["shape"][:2] == [
                sampling.chains,
                sampling.draws,
            ]


# A reduced posterior with relaxed convergence thresholds. This protocol never
# produces migration evidence: it exists so the capture path -- the CausalPy
# public API calls, the 0.94 HDI extraction, the draw-wise R-squared, the
# counterfactual dimensions and the effect-table binding check -- is executed
# against the installed stack in ordinary CI time. The registered thresholds of
# REGISTERED_SAMPLING are what every serialized artifact is validated against.
VERIFICATION_SAMPLING_FIELDS = {
    "chains": 2,
    "draws": 500,
    "tune": 500,
    "max_rhat": 1.05,
    "min_ess_bulk": 100.0,
    "min_ess_tail": 100.0,
}


def _verification_sampling(harness):
    registered = harness.REGISTERED_SAMPLING
    return harness.SamplingProtocol(
        master_seed=registered.master_seed,
        target_accept=registered.target_accept,
        max_treedepth=registered.max_treedepth,
        **VERIFICATION_SAMPLING_FIELDS,
    )


def _assert_case_matches_manifest(harness, case: dict[str, Any], expected) -> None:
    """Assert a captured case reproduces its fixed manifest contract exactly."""
    assert harness._json_equal(case["fixture"], expected["fixture"])
    assert harness._json_equal(case["counterfactual"], expected["counterfactual"])
    assert case["effect_summary"]["alpha"] == expected["effect_summary"]["alpha"]
    assert case["effect_summary"]["hdi_prob"] == harness.HDI_PROB
    assert harness._json_equal(
        case["effect_summary"]["metric_bindings"],
        expected["effect_summary"]["metric_bindings"],
    )
    assert (
        case["effect_summary"]["table"]["index"]
        == (expected["effect_summary"]["table"]["index"])
    )
    assert harness._json_equal(
        case["effect_summary"]["table"]["hdi"],
        expected["effect_summary"]["table"]["hdi"],
    )

    captured_series = {series["name"]: series for series in case["series"]}
    assert set(captured_series) == set(expected["series"])
    for name, expected_series in expected["series"].items():
        series = captured_series[name]
        assert harness._json_equal(series["semantics"], expected_series["semantics"]), (
            f"{name} semantics drifted from the fixed manifest"
        )
        assert [metric["id"] for metric in series["metrics"]] == [
            metric["id"] for metric in expected_series["metrics"]
        ]
        assert [metric["selector"] for metric in series["metrics"]] == [
            metric["selector"] for metric in expected_series["metrics"]
        ]

    assert case["sampling_quality"]["divergences"] == 0
    assert case["sampling_quality"]["finite_values"] is True


@pytest.mark.slow
@pytest.mark.integration
def test_capture_reproduces_the_fixed_manifest_on_the_installed_stack(
    genuine_pymc_sampling,
) -> None:
    """Both fixed scenarios must actually capture against the installed CausalPy.

    Every other test in this module drives the harness with synthesized
    artifacts, so nothing else here would notice if the CausalPy public API the
    capture path depends on (``effect_summary(alpha=...)``,
    ``_model_backend.predict``, ``design``/``pre_design``, ``causal_impact``,
    ``y_pred_counterfactual``, ``post_impact``, ``pre_pred``/``post_pred``) or
    the ArviZ HDI/ESS API drifted. Without this, such a break would surface only
    after hours of coordinator sampling.
    """
    harness = _load_harness_module()
    sampling = _verification_sampling(harness)
    dependencies = harness._import_capture_dependencies(REPO_ROOT)
    manifest = harness._scenario_manifest(sampling)

    captured = {
        "difference_in_differences": harness._capture_difference_in_differences(
            dependencies, sampling
        ),
        "synthetic_control": harness._capture_synthetic_control(dependencies, sampling),
    }
    assert set(captured) == set(manifest)
    for case_name, case in captured.items():
        assert case["name"] == case_name
        _assert_case_matches_manifest(harness, case, manifest[case_name])

    did_series = {
        series["name"]: series
        for series in captured["difference_in_differences"]["series"]
    }
    treatment_effect = did_series["did.causal_impact"]["metrics"][0]["summary"]
    assert treatment_effect["mean"] > 0, (
        "the fixed DiD fixture encodes a positive treatment effect"
    )
    assert treatment_effect["hdi_lower"] < treatment_effect["hdi_upper"]

    synthetic_series = {
        series["name"]: series for series in captured["synthetic_control"]["series"]
    }
    average_impact = synthetic_series["sc.post_average_impact"]["metrics"][0]["summary"]
    assert average_impact["hdi_lower"] > 0, (
        "the fixed synthetic-control fixture encodes a positive post-period impact"
    )

    for series_name, series in (
        ("did.draw_wise_r2", did_series["did.draw_wise_r2"]),
        ("sc.draw_wise_r2", synthetic_series["sc.draw_wise_r2"]),
    ):
        for metric in series["metrics"]:
            mean = metric["summary"]["mean"]
            assert 0.0 < mean <= 1.0, f"{series_name} is not a variance ratio: {mean}"


def test_harness_identity_binds_the_executing_file_to_its_committed_blob(
    tmp_path: Path,
) -> None:
    """The first gate of every capture must work in a real checkout.

    ``_harness_identity`` is monkeypatched away in every comparator test, so
    nothing else here would notice if its repository-root, blob-path or
    cleanliness resolution broke -- and it runs before any sampling, so a
    coordinator would discover the break only when starting a capture. Load a
    committed copy from a throwaway repository so the assertion does not depend
    on the state of the checkout running the tests.
    """
    root = tmp_path / "checkout"
    (root / "scripts" / "migration_baseline").mkdir(parents=True)
    copy_path = root / "scripts" / "migration_baseline" / "harness.py"
    copy_path.write_bytes(SCRIPT_PATH.read_bytes())
    (root / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    git = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com"]
    subprocess.run([*git, "init", "-q", "."], cwd=root, check=True)
    subprocess.run([*git, "add", "-A"], cwd=root, check=True)
    subprocess.run([*git, "commit", "-qm", "harness"], cwd=root, check=True)
    spec = importlib.util.spec_from_file_location("committed_harness", copy_path)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = harness
    spec.loader.exec_module(harness)

    identity = harness._harness_identity()

    assert identity["path"] == str(copy_path)
    assert identity["sha256"] == identity["git_blob_sha256"]
    assert identity["sha256"] == harness._sha256_bytes(SCRIPT_PATH.read_bytes())
    assert harness._COMMIT_PATTERN.fullmatch(str(identity["commit"]))
    assert identity["checkout_clean"] is True

    copy_path.write_bytes(copy_path.read_bytes() + b"\n# uncommitted edit\n")
    with pytest.raises(harness.HarnessError, match="must be clean"):
        harness._harness_identity()
