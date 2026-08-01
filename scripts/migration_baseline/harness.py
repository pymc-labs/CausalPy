"""Capture and compare the pinned PyMC 5 and PyMC 6 migration baselines.

The capture command deliberately imports CausalPy from ``--repo-root`` after
checking its exact Git revision. Run it once in each dedicated environment,
twice per stack, then pass all four JSON artifacts to ``compare``. Comparison
uses posterior summaries and semantic schemas, never cross-stack draw equality.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

SUITE_NAME = "causalpy-pymc-migration-baseline"
ARTIFACT_SCHEMA_VERSION = 2
COMPARISON_SCHEMA_VERSION = 2
SCENARIO_VERSION = 2

PYMC5_COMMIT = "79c0a87072fd4653bfaed1eb085f965594c7f03a"
PYMC6_COMMIT = "18a524a1a8512aaa21c46e0ccddbc54501c9eb1a"
STACK_COMMITS = {"pymc5": PYMC5_COMMIT, "pymc6": PYMC6_COMMIT}
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
STACK_RUNTIME_MAJORS = {
    "pymc5": {"arviz": 0, "pymc": 5, "pytensor": 2},
    "pymc6": {"arviz": 1, "pymc": 6, "pytensor": 3},
}
CAPTURE_ROLES = {
    "reference_first": ("pymc5", 1),
    "reference_second": ("pymc5", 2),
    "candidate_first": ("pymc6", 1),
    "candidate_second": ("pymc6", 2),
}

HDI_PROB = 0.94
EFFECT_SUMMARY_ALPHA = 0.06
CORES = 1
TAIL_ESS_PROB = (0.05, 0.95)

ABSOLUTE_TOLERANCE_FLOOR = 1e-6
RELATIVE_TOLERANCE_FLOOR = 1e-4
MCSE_MULTIPLIER = 4.0
MAX_STANDARDIZED_DRIFT = 0.1
DEGENERATE_POSTERIOR_SD = 1e-12


@dataclass(frozen=True)
class SamplingProtocol:
    """Sampler settings and capture-validity thresholds of one capture run.

    ``REGISTERED_SAMPLING`` is the only protocol that produces migration
    evidence: every serialized artifact is validated against it. The values are
    explicit parameters rather than module constants so the capture pipeline can
    be exercised against the installed stack at a reduced posterior size without
    mutating the registered protocol.
    """

    chains: int
    draws: int
    tune: int
    master_seed: int
    target_accept: float
    max_treedepth: int
    max_rhat: float
    min_ess_bulk: float
    min_ess_tail: float


REGISTERED_SAMPLING = SamplingProtocol(
    chains=4,
    draws=1_000,
    tune=1_000,
    master_seed=1048,
    target_accept=0.95,
    max_treedepth=12,
    max_rhat=1.01,
    min_ess_bulk=400.0,
    min_ess_tail=400.0,
)

# DifferenceInDifferences requires stable unit labels even when the formula
# omits them.
DID_RECORDS: tuple[dict[str, Any], ...] = (
    {"unit": "control", "t": 0, "group": 0, "post_treatment": False, "y": 10.00},
    {"unit": "control", "t": 1, "group": 0, "post_treatment": False, "y": 10.65},
    {"unit": "control", "t": 2, "group": 0, "post_treatment": False, "y": 11.10},
    {"unit": "control", "t": 3, "group": 0, "post_treatment": False, "y": 11.82},
    {"unit": "control", "t": 4, "group": 0, "post_treatment": False, "y": 12.34},
    {"unit": "control", "t": 5, "group": 0, "post_treatment": False, "y": 12.81},
    {"unit": "control", "t": 6, "group": 0, "post_treatment": True, "y": 13.15},
    {"unit": "control", "t": 7, "group": 0, "post_treatment": True, "y": 13.72},
    {"unit": "control", "t": 8, "group": 0, "post_treatment": True, "y": 14.10},
    {"unit": "control", "t": 9, "group": 0, "post_treatment": True, "y": 14.70},
    {"unit": "control", "t": 10, "group": 0, "post_treatment": True, "y": 15.10},
    {"unit": "control", "t": 11, "group": 0, "post_treatment": True, "y": 15.60},
    {"unit": "treated", "t": 0, "group": 1, "post_treatment": False, "y": 11.30},
    {"unit": "treated", "t": 1, "group": 1, "post_treatment": False, "y": 11.75},
    {"unit": "treated", "t": 2, "group": 1, "post_treatment": False, "y": 12.40},
    {"unit": "treated", "t": 3, "group": 1, "post_treatment": False, "y": 13.00},
    {"unit": "treated", "t": 4, "group": 1, "post_treatment": False, "y": 13.42},
    {"unit": "treated", "t": 5, "group": 1, "post_treatment": False, "y": 13.98},
    {"unit": "treated", "t": 6, "group": 1, "post_treatment": True, "y": 15.30},
    {"unit": "treated", "t": 7, "group": 1, "post_treatment": True, "y": 15.90},
    {"unit": "treated", "t": 8, "group": 1, "post_treatment": True, "y": 16.40},
    {"unit": "treated", "t": 9, "group": 1, "post_treatment": True, "y": 17.00},
    {"unit": "treated", "t": 10, "group": 1, "post_treatment": True, "y": 17.50},
    {"unit": "treated", "t": 11, "group": 1, "post_treatment": True, "y": 18.20},
)

SYNTHETIC_CONTROL_RECORDS: tuple[dict[str, Any], ...] = (
    {"t": 0, "a": 10.00, "b": 9.80, "c": 10.40, "actual": 10.02},
    {"t": 1, "a": 10.30, "b": 10.05, "c": 10.20, "actual": 10.17},
    {"t": 2, "a": 10.55, "b": 10.30, "c": 10.65, "actual": 10.49},
    {"t": 3, "a": 10.25, "b": 10.50, "c": 10.70, "actual": 10.45},
    {"t": 4, "a": 10.85, "b": 10.75, "c": 10.90, "actual": 10.82},
    {"t": 5, "a": 11.10, "b": 10.95, "c": 11.25, "actual": 11.07},
    {"t": 6, "a": 11.35, "b": 11.20, "c": 11.10, "actual": 11.24},
    {"t": 7, "a": 11.15, "b": 11.35, "c": 11.50, "actual": 11.28},
    {"t": 8, "a": 11.65, "b": 11.55, "c": 11.80, "actual": 11.57},
    {"t": 9, "a": 11.95, "b": 11.75, "c": 11.60, "actual": 11.81},
    {"t": 10, "a": 12.20, "b": 12.05, "c": 12.25, "actual": 12.14},
    {"t": 11, "a": 12.05, "b": 12.30, "c": 12.50, "actual": 12.20},
    {"t": 12, "a": 12.55, "b": 12.45, "c": 12.70, "actual": 14.34},
    {"t": 13, "a": 12.85, "b": 12.70, "c": 12.90, "actual": 14.58},
    {"t": 14, "a": 13.10, "b": 12.90, "c": 13.20, "actual": 14.89},
    {"t": 15, "a": 12.90, "b": 13.10, "c": 13.35, "actual": 14.91},
    {"t": 16, "a": 13.40, "b": 13.30, "c": 13.55, "actual": 15.33},
    {"t": 17, "a": 13.70, "b": 13.55, "c": 13.80, "actual": 15.62},
    {"t": 18, "a": 13.95, "b": 13.75, "c": 14.10, "actual": 15.92},
    {"t": 19, "a": 14.25, "b": 14.00, "c": 14.30, "actual": 16.23},
)


class HarnessError(RuntimeError):
    """Raised when a baseline artifact is not valid migration evidence."""


def _is_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` resolves below ``root``."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _script_repository_root() -> Path:
    """Return the tracked repository containing this permanent harness."""
    return Path(__file__).resolve().parents[2]


def _require_new_external_output(path: Path, *tracked_roots: Path) -> Path:
    """Reject tracked or pre-existing destinations before expensive capture work."""
    resolved = _require_external_output(path, *tracked_roots)
    if resolved.exists() or resolved.is_symlink():
        raise HarnessError(f"Evidence destination already exists: {resolved}")
    return resolved


def _require_external_output(path: Path, *tracked_roots: Path) -> Path:
    """Reject generated evidence paths inside a tracked checkout."""
    resolved = path.expanduser().resolve()
    for root in tracked_roots:
        if _is_within(resolved, root):
            raise HarnessError(
                f"Generated evidence must be outside tracked checkouts, got {resolved} "
                f"inside {root.resolve()}."
            )
    return resolved


def _canonical_value(value: Any) -> Any:
    """Convert scalar metadata to deterministic JSON-compatible values."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _canonical_value(value.item())
        except ValueError:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HarnessError(f"Non-finite value in artifact metadata: {value!r}")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    return str(value)


def _canonical_json(value: Any) -> str:
    """Return canonical JSON for deterministic hashes and artifact serialization."""
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(text: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of a byte payload."""
    return hashlib.sha256(value).hexdigest()


def _records_payload(records: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Copy fixed fixture records into JSON-safe artifact data."""
    return [
        {key: _canonical_value(value) for key, value in row.items()} for row in records
    ]


def _fixture_payload(name: str, records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Return fixed, serialized input data and its content hash."""
    payload = _records_payload(records)
    return {
        "name": name,
        "records": payload,
        "sha256": _sha256_text(_canonical_json(payload)),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    """Create text atomically without replacing any existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise HarnessError(
                f"Evidence destination already exists: {path}"
            ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write sorted, strict JSON atomically."""
    _assert_finite_json(payload)
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, serialized)


def _assert_finite_json(value: Any, path: str = "artifact") -> None:
    """Reject NaN and infinity anywhere in a JSON-compatible object."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HarnessError(f"{path} contains non-finite float {value!r}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_json(item, f"{path}.{key}")


def _run_git(repo_root: Path, *args: str) -> str:
    """Run a read-only Git query against an explicitly selected repository."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise HarnessError(f"Git query failed in {repo_root}: {message}")
    return result.stdout.strip()


def _run_git_bytes(repo_root: Path, *args: str) -> bytes:
    """Run a read-only Git query and return its exact byte output."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        if not message:
            message = result.stdout.decode("utf-8", errors="replace").strip()
        raise HarnessError(f"Git query failed in {repo_root}: {message}")
    return result.stdout


def _harness_identity() -> dict[str, str | bool]:
    """Bind the executing comparator to a clean tracked harness blob."""
    repo_root = _script_repository_root().resolve()
    script_path = Path(__file__).resolve()
    try:
        relative_path = script_path.relative_to(repo_root).as_posix()
    except ValueError as error:
        raise HarnessError(
            "Harness file is outside its claimed repository root"
        ) from error
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise HarnessError(f"Harness Git revision is not a full SHA-1: {commit!r}")
    dirty_paths = _run_git(repo_root, "status", "--porcelain")
    if dirty_paths:
        raise HarnessError(
            "Harness checkout must be clean before capture or comparison; found "
            f"uncommitted paths:\n{dirty_paths}"
        )
    harness_sha256 = _sha256_bytes(script_path.read_bytes())
    git_blob_sha256 = _sha256_bytes(
        _run_git_bytes(repo_root, "show", f"HEAD:{relative_path}")
    )
    if harness_sha256 != git_blob_sha256:
        raise HarnessError(
            f"Executing harness bytes do not match the clean Git blob at {commit}."
        )
    return {
        "path": str(script_path),
        "sha256": harness_sha256,
        "commit": commit,
        "git_blob_sha256": git_blob_sha256,
        "checkout_clean": True,
    }


def _activate_repository(repo_root: Path) -> Path:
    """Put exactly one checkout first on ``sys.path`` and return its source path."""
    resolved = repo_root.expanduser().resolve()
    source = resolved / "causalpy"
    if not source.is_dir():
        raise HarnessError(f"{resolved} does not contain a causalpy source directory")
    sys.path.insert(0, str(resolved))
    return resolved


def _import_capture_dependencies(repo_root: Path) -> dict[str, Any]:
    """Import sampling dependencies only after pinning the requested checkout."""
    resolved = _activate_repository(repo_root)
    import arviz as az
    import numpy as np
    import pandas as pd
    import pymc as pm
    import pytensor as pt
    import xarray as xr

    import causalpy as cp

    module_path = Path(cp.__file__).resolve()
    if not _is_within(module_path, resolved):
        raise HarnessError(
            "CausalPy was not imported from --repo-root: "
            f"expected below {resolved}, imported {module_path}."
        )
    return {
        "az": az,
        "cp": cp,
        "np": np,
        "pd": pd,
        "pm": pm,
        "pt": pt,
        "xr": xr,
    }


def _module_version(module: Any, name: str) -> str:
    """Return a version from the imported module rather than package metadata."""
    version = getattr(module, "__version__", None)
    if not isinstance(version, str) or not version:
        raise HarnessError(f"Imported {name} module does not expose a version")
    return version


def _module_major(version: str, name: str) -> int:
    """Parse one imported package major version."""
    match = re.fullmatch(r"([0-9]+)(?:\.[0-9]+)*(?:[+.-].*)?", version)
    if match is None:
        raise HarnessError(f"Imported {name} version is not parseable: {version!r}")
    return int(match.group(1))


def _module_path(module: Any, name: str, prefix: Path) -> str:
    """Require an imported runtime dependency to come from the active prefix."""
    location = getattr(module, "__file__", None)
    if not isinstance(location, str) or not location:
        raise HarnessError(f"Imported {name} module does not expose a file path")
    resolved = Path(location).resolve()
    if not _is_within(resolved, prefix):
        raise HarnessError(
            f"Imported {name} module is outside sys.prefix {prefix}: {resolved}"
        )
    return str(resolved)


def _editable_causalpy_target() -> Path:
    """Read the active CausalPy editable-install target from direct_url metadata."""
    try:
        distribution = importlib.metadata.distribution("causalpy")
        direct_url = distribution.read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError as error:
        raise HarnessError(
            "Active environment has no CausalPy distribution metadata"
        ) from error
    if direct_url is None:
        raise HarnessError("Active CausalPy distribution has no direct_url.json")
    try:
        payload = json.loads(direct_url)
    except json.JSONDecodeError as error:
        raise HarnessError("Active CausalPy direct_url.json is malformed") from error
    if not isinstance(payload, dict):
        raise HarnessError("Active CausalPy direct_url.json is not an object")
    directory_info = payload.get("dir_info")
    if (
        not isinstance(directory_info, dict)
        or directory_info.get("editable") is not True
        or not isinstance(payload.get("url"), str)
    ):
        raise HarnessError("Active CausalPy distribution is not an editable install")
    parsed = urlparse(payload["url"])
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise HarnessError("Active CausalPy editable target is not a local file URL")
    return Path(unquote(parsed.path)).resolve()


def _capture_runtime_provenance(
    stack: str, dependencies: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    """Validate and serialize the imported stack identity before sampling."""
    prefix = Path(sys.prefix).resolve()
    executable = Path(sys.executable).resolve()
    if not _is_within(executable, prefix):
        raise HarnessError(
            f"Python executable is outside sys.prefix {prefix}: {executable}"
        )
    modules = {
        "arviz": dependencies["az"],
        "causalpy": dependencies["cp"],
        "numpy": dependencies["np"],
        "pandas": dependencies["pd"],
        "pymc": dependencies["pm"],
        "pytensor": dependencies["pt"],
        "xarray": dependencies["xr"],
    }
    versions = {name: _module_version(module, name) for name, module in modules.items()}
    for name, expected_major in STACK_RUNTIME_MAJORS[stack].items():
        if _module_major(versions[name], name) != expected_major:
            raise HarnessError(
                f"{stack} capture requires imported {name} major {expected_major}, "
                f"got {versions[name]!r}"
            )
    editable_target = _editable_causalpy_target()
    if editable_target != repo_root:
        raise HarnessError(
            "Active CausalPy editable-install target differs from --repo-root: "
            f"{editable_target} != {repo_root}"
        )
    module_paths = {
        name: _module_path(module, name, prefix)
        for name, module in modules.items()
        if name != "causalpy"
    }
    return {
        "dependencies": versions,
        "runtime": {
            "executable": str(executable),
            "prefix": str(prefix),
            "module_paths": module_paths,
            "causalpy_editable_target": str(editable_target),
        },
    }


def _supports_nuts_sampler(pm: Any) -> bool:
    """Return whether the installed PyMC exposes an explicit NUTS-sampler choice."""
    try:
        return "nuts_sampler" in inspect.signature(pm.sample).parameters
    except (TypeError, ValueError):
        return False


def _sample_kwargs(pm: Any, sampling: SamplingProtocol) -> dict[str, Any]:
    """Build the serialized-chain sampling configuration of one protocol."""
    kwargs: dict[str, Any] = {
        "chains": sampling.chains,
        "cores": CORES,
        "draws": sampling.draws,
        "tune": sampling.tune,
        "random_seed": sampling.master_seed,
        "progressbar": False,
        "target_accept": sampling.target_accept,
        "max_treedepth": sampling.max_treedepth,
    }
    if _supports_nuts_sampler(pm):
        kwargs["nuts_sampler"] = "pymc"
    return kwargs


def _protocol(
    supports_nuts_sampler: bool, sampling: SamplingProtocol
) -> dict[str, Any]:
    """Return the preregistered protocol recorded in every capture artifact."""
    return {
        "scenario_version": SCENARIO_VERSION,
        "hdi_prob": HDI_PROB,
        "effect_summary_alpha": EFFECT_SUMMARY_ALPHA,
        "draw_wise_r2": {
            "definition": (
                "var_obs(mu) / (var_obs(mu) + var_obs(y - mu)) with ddof=0 "
                "for every chain, draw, and treated unit"
            ),
            "prediction_kind": "conditional_expected_mu",
        },
        "sampling": {
            "sampler": "pymc",
            "chains": sampling.chains,
            "cores": CORES,
            "draws": sampling.draws,
            "tune": sampling.tune,
            "master_seed": sampling.master_seed,
            "target_accept": sampling.target_accept,
            "max_treedepth": sampling.max_treedepth,
            "nuts_sampler_argument_used": supports_nuts_sampler,
        },
        "evidence_validity": {
            "max_rhat": sampling.max_rhat,
            "min_ess_bulk": sampling.min_ess_bulk,
            "min_ess_tail": sampling.min_ess_tail,
            "tail_ess_prob": list(TAIL_ESS_PROB),
            "divergences": 0,
            "reject_tree_depth_saturation_when_exposed": True,
        },
        "migration_hard_gates": {
            "absolute_mean_delta": (
                "abs(candidate_mean - reference_mean) <= "
                "max(4 * hypot(reference_mcse, candidate_mcse), "
                "1e-6 + 1e-4 * abs(reference_mean))"
            ),
            "standardized_mean_drift": (
                "abs(candidate_mean - reference_mean) / pooled_posterior_sd <= 0.1"
            ),
            "semantic_equality": "effect-table schema and prediction coordinates",
            "hdi_containment": "diagnostic only",
        },
    }


def _group(idata: Any, name: str) -> Any:
    """Read an InferenceData or DataTree group through the common public shape."""
    group = getattr(idata, name, None)
    if group is None:
        try:
            group = idata[name]
        except (KeyError, TypeError):
            group = None
    if group is None:
        raise HarnessError(f"Inference output is missing the required {name!r} group")
    dataset = getattr(group, "ds", None)
    return dataset if dataset is not None else group


def _sampling_quality(
    idata: Any, np: Any, sampling: SamplingProtocol
) -> dict[str, Any]:
    """Reject divergent or tree-depth-saturated chains before comparing outputs."""
    sample_stats = _group(idata, "sample_stats")
    if "diverging" not in sample_stats:
        raise HarnessError("Inference output is missing sample_stats.diverging")
    divergences = int(np.asarray(sample_stats["diverging"].values, dtype=int).sum())
    if divergences:
        raise HarnessError(
            f"Divergent chains invalidate baseline evidence ({divergences} draws)"
        )

    tree_depth_status = "not-exposed"
    tree_depth_events = 0
    max_observed_tree_depth: int | None = None
    if "reached_max_treedepth" in sample_stats:
        tree_depth_events = int(
            np.asarray(sample_stats["reached_max_treedepth"].values, dtype=int).sum()
        )
        tree_depth_status = "reached_max_treedepth"
    elif "tree_depth" in sample_stats:
        tree_depth = np.asarray(sample_stats["tree_depth"].values, dtype=int)
        max_observed_tree_depth = int(tree_depth.max())
        tree_depth_events = int((tree_depth >= sampling.max_treedepth).sum())
        tree_depth_status = "tree_depth"
    if tree_depth_events:
        raise HarnessError(
            "Tree-depth saturation invalidates baseline evidence "
            f"({tree_depth_events} draws; source={tree_depth_status})."
        )

    return {
        "divergences": divergences,
        "tree_depth_source": tree_depth_status,
        "tree_depth_events": tree_depth_events,
        "max_observed_tree_depth": max_observed_tree_depth,
    }


def _finite_scalar(value: Any, label: str, np: Any) -> float:
    """Convert one scalar diagnostic to a finite Python float."""
    array = np.asarray(value)
    if array.size != 1:
        raise HarnessError(f"{label} must be scalar, got shape {array.shape}")
    result = float(array.reshape(()))
    if not math.isfinite(result):
        raise HarnessError(f"{label} is non-finite: {result!r}")
    return result


def _draw_digest(draws: Any, np: Any) -> str:
    """Hash a raw draw matrix only for same-stack repeatability checks."""
    values = np.ascontiguousarray(np.asarray(draws, dtype="<f8"))
    header = _canonical_json({"dtype": "float64-le", "shape": list(values.shape)})
    digest = hashlib.sha256(header.encode("utf-8"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _hdi_interval(flattened_draws: Any, az: Any, np: Any) -> Any:
    """Call the installed ArviZ HDI API with an explicitly requested probability."""
    try:
        parameter_names = inspect.signature(az.hdi).parameters
    except (TypeError, ValueError):
        parameter_names = {}
    keyword = "prob" if "prob" in parameter_names else "hdi_prob"
    return np.asarray(az.hdi(flattened_draws, **{keyword: HDI_PROB}), dtype=float)


def _tail_ess(draws: Any, az: Any) -> Any:
    """Calculate tail ESS with one explicit probability policy across ArviZ APIs."""
    try:
        return az.ess(draws, method="tail", prob=TAIL_ESS_PROB)
    except TypeError as error:
        raise HarnessError(
            "Installed ArviZ cannot calculate tail ESS with the registered "
            f"probability pair {TAIL_ESS_PROB!r}"
        ) from error


def _summary_from_draws(
    draws: Any, az: Any, np: Any, label: str, sampling: SamplingProtocol
) -> dict[str, float]:
    """Summarize one chain-by-draw scalar with validity diagnostics."""
    values = np.asarray(draws, dtype=float)
    expected_shape = (sampling.chains, sampling.draws)
    if values.shape != expected_shape:
        raise HarnessError(
            f"{label} must have shape {expected_shape}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise HarnessError(f"{label} contains non-finite posterior draws")

    flattened = values.reshape(-1)
    interval = _hdi_interval(flattened, az, np)
    if interval.shape != (2,):
        raise HarnessError(f"{label} returned invalid HDI shape {interval.shape}")

    summary = {
        "mean": _finite_scalar(values.mean(), f"{label} mean", np),
        "posterior_sd": _finite_scalar(values.std(ddof=1), f"{label} posterior SD", np),
        "mcse_mean": _finite_scalar(
            az.mcse(values, method="mean"), f"{label} MCSE", np
        ),
        "rhat": _finite_scalar(az.rhat(values, method="rank"), f"{label} R-hat", np),
        "ess_bulk": _finite_scalar(
            az.ess(values, method="bulk"), f"{label} bulk ESS", np
        ),
        "ess_tail": _finite_scalar(_tail_ess(values, az), f"{label} tail ESS", np),
        "hdi_lower": _finite_scalar(interval[0], f"{label} HDI lower", np),
        "hdi_upper": _finite_scalar(interval[1], f"{label} HDI upper", np),
    }
    if summary["posterior_sd"] < 0:
        raise HarnessError(f"{label} has a negative posterior SD")
    if summary["mcse_mean"] < 0:
        raise HarnessError(f"{label} has a negative MCSE")
    if summary["rhat"] <= 0:
        raise HarnessError(f"{label} has a non-positive R-hat")
    if summary["hdi_lower"] > summary["hdi_upper"]:
        raise HarnessError(f"{label} has inverted {HDI_PROB:.2f} HDI bounds")
    if summary["rhat"] > sampling.max_rhat:
        raise HarnessError(
            f"{label} has R-hat {summary['rhat']:.6g}, above {sampling.max_rhat:.6g}"
        )
    if summary["ess_bulk"] < sampling.min_ess_bulk:
        raise HarnessError(
            f"{label} has bulk ESS {summary['ess_bulk']:.6g}, "
            f"below {sampling.min_ess_bulk:.6g}"
        )
    if summary["ess_tail"] < sampling.min_ess_tail:
        raise HarnessError(
            f"{label} has tail ESS {summary['ess_tail']:.6g}, "
            f"below {sampling.min_ess_tail:.6g}"
        )
    return summary


def _coordinate_values(data: Any, dimension: str) -> list[Any]:
    """Return one dimension coordinate in semantic JSON form."""
    if dimension not in data.coords:
        raise HarnessError(f"Prediction output is missing coordinate {dimension!r}")
    coordinate = data.coords[dimension]
    if coordinate.ndim != 1 or coordinate.dims != (dimension,):
        raise HarnessError(
            f"Coordinate {dimension!r} must be one-dimensional over itself, got "
            f"dims={coordinate.dims!r}"
        )
    return [_canonical_value(value) for value in coordinate.values.tolist()]


def _array_semantics(data: Any) -> dict[str, Any]:
    """Capture names, order, shapes, and values of all meaningful dimensions."""
    dimensions = list(data.dims)
    if dimensions[:2] != ["chain", "draw"]:
        raise HarnessError(
            "Posterior output must begin with canonical ('chain', 'draw') dimensions, "
            f"got {tuple(data.dims)!r}"
        )
    return {
        "dims": dimensions,
        "shape": [int(size) for size in data.shape],
        "coords": {
            dimension: _coordinate_values(data, dimension) for dimension in dimensions
        },
    }


def _metric_id(series_name: str, selector: dict[str, Any]) -> str:
    """Build a stable identifier for one scalar extracted from a draw series."""
    if not selector:
        return series_name
    rendered = ", ".join(f"{key}={value}" for key, value in selector.items())
    return f"{series_name}[{rendered}]"


def _series_manifest(
    name: str, dimensions: tuple[str, ...], coordinates: dict[str, list[Any]]
) -> dict[str, Any]:
    """Build the exact expected semantics and scalar selectors for one series."""
    semantics = {
        "dims": list(dimensions),
        "shape": [len(coordinates[dimension]) for dimension in dimensions],
        "coords": {dimension: coordinates[dimension] for dimension in dimensions},
    }
    selectors: list[dict[str, Any]] = [{}]
    for dimension in dimensions[2:]:
        selectors = [
            {**selector, dimension: value}
            for selector in selectors
            for value in coordinates[dimension]
        ]
    return {
        "semantics": semantics,
        "metrics": [
            {"id": _metric_id(name, selector), "selector": selector}
            for selector in selectors
        ],
    }


def _effect_table_manifest(index: list[str]) -> dict[str, Any]:
    """Return the stable public effect-summary table contract."""
    return {
        "shape": [len(index), 5],
        "columns": ["mean", "median", "hdi_lower", "hdi_upper", "p_gt_0"],
        "index": index,
        "index_name": None,
        "hdi": {
            "probability": HDI_PROB,
            "lower_column": "hdi_lower",
            "upper_column": "hdi_upper",
        },
    }


def _scenario_manifest(sampling: SamplingProtocol) -> dict[str, dict[str, Any]]:
    """Return the immutable fixed-output contract derived from fixture records."""
    sample_coordinates = {
        "chain": list(range(sampling.chains)),
        "draw": list(range(sampling.draws)),
    }
    did_counterfactual_count = sum(
        row["group"] == 1 and row["post_treatment"] for row in DID_RECORDS
    )
    did_counterfactual_coordinates = {
        **sample_coordinates,
        "obs_ind": list(range(did_counterfactual_count)),
        "treated_units": ["unit_0"],
    }
    singleton_unit_coordinates = {
        **sample_coordinates,
        "treated_units": ["unit_0"],
    }
    synthetic_unit_coordinates = {
        **sample_coordinates,
        "treated_units": ["actual"],
    }
    synthetic_counterfactual_coordinates = {
        **sample_coordinates,
        "obs_ind": [row["t"] for row in SYNTHETIC_CONTROL_RECORDS if row["t"] >= 12],
        "treated_units": ["actual"],
    }
    did_bindings = [
        {
            "table_row": "treatment_effect",
            "series": "did.causal_impact",
            "selector": {},
        }
    ]
    synthetic_bindings = [
        {
            "table_row": "average",
            "series": "sc.post_average_impact",
            "selector": {"treated_units": "actual"},
        },
        {
            "table_row": "cumulative",
            "series": "sc.post_cumulative_impact",
            "selector": {"treated_units": "actual"},
        },
    ]
    return {
        "difference_in_differences": {
            "fixture": _fixture_payload("fixed_difference_in_differences", DID_RECORDS),
            "effect_summary": {
                "alpha": EFFECT_SUMMARY_ALPHA,
                "hdi_prob": HDI_PROB,
                "table": _effect_table_manifest(["treatment_effect"]),
                "metric_bindings": did_bindings,
            },
            "counterfactual": {
                "series": "did.counterfactual_mu",
                "prediction_kind": "conditional_expected_mu",
            },
            "series": {
                "did.causal_impact": _series_manifest(
                    "did.causal_impact",
                    ("chain", "draw"),
                    sample_coordinates,
                ),
                "did.draw_wise_r2": _series_manifest(
                    "did.draw_wise_r2",
                    ("chain", "draw", "treated_units"),
                    singleton_unit_coordinates,
                ),
                "did.counterfactual_mu": _series_manifest(
                    "did.counterfactual_mu",
                    ("chain", "draw", "obs_ind", "treated_units"),
                    did_counterfactual_coordinates,
                ),
            },
        },
        "synthetic_control": {
            "fixture": _fixture_payload(
                "fixed_synthetic_control", SYNTHETIC_CONTROL_RECORDS
            ),
            "effect_summary": {
                "alpha": EFFECT_SUMMARY_ALPHA,
                "hdi_prob": HDI_PROB,
                "table": _effect_table_manifest(["average", "cumulative"]),
                "metric_bindings": synthetic_bindings,
            },
            "counterfactual": {
                "series": "sc.counterfactual_mu",
                "prediction_kind": "conditional_expected_mu",
            },
            "series": {
                "sc.post_average_impact": _series_manifest(
                    "sc.post_average_impact",
                    ("chain", "draw", "treated_units"),
                    synthetic_unit_coordinates,
                ),
                "sc.post_cumulative_impact": _series_manifest(
                    "sc.post_cumulative_impact",
                    ("chain", "draw", "treated_units"),
                    synthetic_unit_coordinates,
                ),
                "sc.draw_wise_r2": _series_manifest(
                    "sc.draw_wise_r2",
                    ("chain", "draw", "treated_units"),
                    synthetic_unit_coordinates,
                ),
                "sc.counterfactual_mu": _series_manifest(
                    "sc.counterfactual_mu",
                    ("chain", "draw", "obs_ind", "treated_units"),
                    synthetic_counterfactual_coordinates,
                ),
            },
        },
    }


def _capture_series(
    name: str,
    data: Any,
    az: Any,
    np: Any,
    sampling: SamplingProtocol,
    *,
    expected_dims: tuple[str, ...] | None = None,
    expected_name: str | None = None,
) -> dict[str, Any]:
    """Capture scalar summaries and same-stack draw digests from a posterior array."""
    if expected_dims is not None and tuple(data.dims) != expected_dims:
        raise HarnessError(
            f"{name} dimensions changed: expected {expected_dims!r}, "
            f"got {tuple(data.dims)!r}"
        )
    if expected_name is not None and data.name != expected_name:
        raise HarnessError(
            f"{name} must be extracted from {expected_name!r}, got {data.name!r}"
        )
    value_dimensions = [
        dimension for dimension in data.dims if dimension not in {"chain", "draw"}
    ]
    ordered = data.transpose("chain", "draw", *value_dimensions)
    semantics = _array_semantics(ordered)
    values = np.asarray(ordered.values, dtype=float)
    value_shape = values.shape[2:]
    indexes = [()] if not value_shape else list(np.ndindex(value_shape))
    metrics: list[dict[str, Any]] = []

    for index in indexes:
        selector = {
            dimension: semantics["coords"][dimension][offset]
            for dimension, offset in zip(value_dimensions, index, strict=True)
        }
        draws = values[(slice(None), slice(None), *index)]
        metric_name = _metric_id(name, selector)
        metrics.append(
            {
                "id": metric_name,
                "selector": selector,
                "draw_digest": _draw_digest(draws, np),
                "summary": _summary_from_draws(draws, az, np, metric_name, sampling),
            }
        )
    return {"name": name, "semantics": semantics, "metrics": metrics}


def _canonical_scalar_effect(data: Any) -> Any:
    """Normalize only the known singleton backend dimension of a scalar DiD effect."""
    if tuple(data.dims[:2]) != ("chain", "draw"):
        raise HarnessError(
            "Scalar effect must begin with canonical ('chain', 'draw') dimensions, "
            f"got {tuple(data.dims)!r}"
        )
    value_dimensions = tuple(
        dimension for dimension in data.dims if dimension not in {"chain", "draw"}
    )
    if not value_dimensions:
        return data
    if value_dimensions == ("treated_units",) and data.sizes["treated_units"] == 1:
        return data.isel({"treated_units": 0}, drop=True)
    raise HarnessError(
        "Scalar DiD effect has an unexpected non-sample dimension contract: "
        f"{value_dimensions!r}"
    )


def _coordinates_match(left: Any, right: Any, dimension: str, np: Any) -> bool:
    """Return whether two xarray coordinates have the same values and order."""
    return np.array_equal(left.coords[dimension].values, right.coords[dimension].values)


def _draw_wise_r2(observed: Any, expected_mu: Any, xr: Any, np: Any) -> Any:
    """Recompute CausalPy's Bayesian R-squared without collapsing posterior draws."""
    expected_dims = ("chain", "draw", "obs_ind", "treated_units")
    if tuple(expected_mu.dims) != expected_dims:
        raise HarnessError(
            "Expected-value predictions must have canonical dimensions "
            f"{expected_dims!r}, got {tuple(expected_mu.dims)!r}"
        )
    if expected_mu.name != "mu":
        raise HarnessError(
            "Draw-wise R-squared requires conditional expected 'mu', "
            f"got {expected_mu.name!r}"
        )
    if tuple(observed.dims) != ("obs_ind", "treated_units"):
        raise HarnessError(
            "Observed outcomes must have ('obs_ind', 'treated_units') dimensions, "
            f"got {tuple(observed.dims)!r}"
        )
    for dimension in ("obs_ind", "treated_units"):
        if not _coordinates_match(observed, expected_mu, dimension, np):
            raise HarnessError(
                f"Observed and expected-value {dimension!r} coordinates differ"
            )

    prediction = np.asarray(expected_mu.values, dtype=float)
    outcome = np.asarray(observed.values, dtype=float)
    if not np.isfinite(prediction).all() or not np.isfinite(outcome).all():
        raise HarnessError(
            "Draw-wise R-squared requires finite outcomes and expected values"
        )
    variance_of_prediction = np.var(prediction, axis=2, ddof=0)
    variance_of_residual = np.var(
        outcome[None, None, :, :] - prediction,
        axis=2,
        ddof=0,
    )
    denominator = variance_of_prediction + variance_of_residual
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise HarnessError(
            "Draw-wise R-squared has a non-positive or non-finite denominator"
        )
    r2 = variance_of_prediction / denominator
    if not np.isfinite(r2).all():
        raise HarnessError("Draw-wise R-squared contains non-finite values")
    return xr.DataArray(
        r2,
        dims=("chain", "draw", "treated_units"),
        coords={
            "chain": expected_mu.coords["chain"],
            "draw": expected_mu.coords["draw"],
            "treated_units": expected_mu.coords["treated_units"],
        },
        name="draw_wise_r2",
    )


def _table_semantics(table: Any) -> dict[str, Any]:
    """Capture the non-numeric semantic contract of an effect-summary table."""
    required = {"mean", "hdi_lower", "hdi_upper"}
    columns = [str(column) for column in table.columns.tolist()]
    missing = required.difference(columns)
    if missing:
        raise HarnessError(
            "Effect-summary table is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )
    return {
        "shape": [int(table.shape[0]), int(table.shape[1])],
        "columns": columns,
        "index": [_canonical_value(value) for value in table.index.tolist()],
        "index_name": _canonical_value(table.index.name),
        "hdi": {
            "probability": HDI_PROB,
            "lower_column": "hdi_lower",
            "upper_column": "hdi_upper",
        },
    }


def _series_metric(series: dict[str, Any], selector: dict[str, Any]) -> dict[str, Any]:
    """Return one captured metric selected by all non-sample coordinates."""
    for metric in series["metrics"]:
        if metric["selector"] == selector:
            return metric
    raise HarnessError(
        f"Missing metric in series {series['name']!r} for selector {selector!r}"
    )


def _verify_effect_table(
    table: Any,
    bindings: list[dict[str, Any]],
    series_by_name: dict[str, dict[str, Any]],
    np: Any,
) -> None:
    """Verify that the public table reports unrounded 94% HDI summary statistics."""
    for binding in bindings:
        row = binding["table_row"]
        if row not in table.index:
            raise HarnessError(f"Effect-summary table is missing expected row {row!r}")
        metric = _series_metric(series_by_name[binding["series"]], binding["selector"])
        expected = metric["summary"]
        for column, statistic in (
            ("mean", "mean"),
            ("hdi_lower", "hdi_lower"),
            ("hdi_upper", "hdi_upper"),
        ):
            reported = float(table.loc[row, column])
            if not math.isfinite(reported):
                raise HarnessError(
                    f"Effect-summary table {row!r}/{column!r} is non-finite"
                )
            if not np.isclose(reported, expected[statistic], rtol=1e-10, atol=1e-10):
                raise HarnessError(
                    f"Effect-summary table {row!r}/{column!r} does not match the "
                    f"unrounded {HDI_PROB:.2f} posterior summary."
                )


def _capture_difference_in_differences(
    dependencies: dict[str, Any], sampling: SamplingProtocol
) -> dict[str, Any]:
    """Capture a representative coefficient-based counterfactual analysis."""
    cp = dependencies["cp"]
    np = dependencies["np"]
    pd = dependencies["pd"]
    xr = dependencies["xr"]
    az = dependencies["az"]
    sample_kwargs = _sample_kwargs(dependencies["pm"], sampling)

    data = pd.DataFrame(_records_payload(DID_RECORDS))
    result = cp.DifferenceInDifferences(
        data,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=dict(sample_kwargs)),
    )
    quality = _sampling_quality(result.idata, np, sampling)
    effect_summary = result.effect_summary(alpha=EFFECT_SUMMARY_ALPHA)
    fitted_mu = result._model_backend.predict(result.design["X"])
    draw_wise_r2 = _draw_wise_r2(result.design["y"], fitted_mu, xr, np)

    series = [
        _capture_series(
            "did.causal_impact",
            _canonical_scalar_effect(result.causal_impact),
            az,
            np,
            sampling,
        ),
        _capture_series("did.draw_wise_r2", draw_wise_r2, az, np, sampling),
        _capture_series(
            "did.counterfactual_mu",
            result.y_pred_counterfactual,
            az,
            np,
            sampling,
            expected_dims=("chain", "draw", "obs_ind", "treated_units"),
            expected_name="mu",
        ),
    ]
    series_by_name = {item["name"]: item for item in series}
    bindings = [
        {
            "table_row": "treatment_effect",
            "series": "did.causal_impact",
            "selector": {},
        }
    ]
    _verify_effect_table(effect_summary.table, bindings, series_by_name, np)
    quality["finite_values"] = True

    return {
        "name": "difference_in_differences",
        "fixture": _fixture_payload("fixed_difference_in_differences", DID_RECORDS),
        "sampling_quality": quality,
        "effect_summary": {
            "alpha": EFFECT_SUMMARY_ALPHA,
            "hdi_prob": HDI_PROB,
            "table": _table_semantics(effect_summary.table),
            "metric_bindings": bindings,
        },
        "counterfactual": {
            "series": "did.counterfactual_mu",
            "prediction_kind": "conditional_expected_mu",
        },
        "series": series,
    }


def _capture_synthetic_control(
    dependencies: dict[str, Any], sampling: SamplingProtocol
) -> dict[str, Any]:
    """Capture a representative simplex-weighted counterfactual analysis."""
    cp = dependencies["cp"]
    np = dependencies["np"]
    pd = dependencies["pd"]
    xr = dependencies["xr"]
    az = dependencies["az"]
    sample_kwargs = _sample_kwargs(dependencies["pm"], sampling)

    data = pd.DataFrame(_records_payload(SYNTHETIC_CONTROL_RECORDS)).set_index("t")
    result = cp.SyntheticControl(
        data,
        treatment_time=12,
        control_units=["a", "b", "c"],
        treated_units=["actual"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=dict(sample_kwargs)),
    )
    quality = _sampling_quality(result.idata, np, sampling)
    effect_summary = result.effect_summary(
        alpha=EFFECT_SUMMARY_ALPHA,
        cumulative=True,
        relative=False,
    )
    post_average_impact = result.post_impact.mean(dim="obs_ind")
    post_cumulative_impact = result.post_impact_cumulative.isel(obs_ind=-1)
    draw_wise_r2 = _draw_wise_r2(result.pre_design["treated"], result.pre_pred, xr, np)

    series = [
        _capture_series(
            "sc.post_average_impact", post_average_impact, az, np, sampling
        ),
        _capture_series(
            "sc.post_cumulative_impact", post_cumulative_impact, az, np, sampling
        ),
        _capture_series("sc.draw_wise_r2", draw_wise_r2, az, np, sampling),
        _capture_series(
            "sc.counterfactual_mu",
            result.post_pred,
            az,
            np,
            sampling,
            expected_dims=("chain", "draw", "obs_ind", "treated_units"),
            expected_name="mu",
        ),
    ]
    series_by_name = {item["name"]: item for item in series}
    bindings = [
        {
            "table_row": "average",
            "series": "sc.post_average_impact",
            "selector": {"treated_units": "actual"},
        },
        {
            "table_row": "cumulative",
            "series": "sc.post_cumulative_impact",
            "selector": {"treated_units": "actual"},
        },
    ]
    _verify_effect_table(effect_summary.table, bindings, series_by_name, np)
    quality["finite_values"] = True

    return {
        "name": "synthetic_control",
        "fixture": _fixture_payload(
            "fixed_synthetic_control", SYNTHETIC_CONTROL_RECORDS
        ),
        "sampling_quality": quality,
        "effect_summary": {
            "alpha": EFFECT_SUMMARY_ALPHA,
            "hdi_prob": HDI_PROB,
            "table": _table_semantics(effect_summary.table),
            "metric_bindings": bindings,
        },
        "counterfactual": {
            "series": "sc.counterfactual_mu",
            "prediction_kind": "conditional_expected_mu",
        },
        "series": series,
    }


def _canonical_uuid(value: Any, label: str) -> str:
    """Require a canonical lowercase UUID for batch and capture identities."""
    if not isinstance(value, str):
        raise HarnessError(f"{label} must be a canonical UUID string")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as error:
        raise HarnessError(f"{label} is not a valid UUID: {value!r}") from error
    normalized = str(parsed)
    if value != normalized:
        raise HarnessError(f"{label} must use canonical lowercase UUID form")
    return normalized


def _capture_artifact(
    stack: str, repo_root: Path, *, batch_id: str, capture_role: str
) -> dict[str, Any]:
    """Run one role-bound fixed scenario capture in a coordinator batch."""
    if capture_role not in CAPTURE_ROLES:
        raise HarnessError(f"Unknown capture role {capture_role!r}")
    expected_stack, capture_ordinal = CAPTURE_ROLES[capture_role]
    if stack != expected_stack:
        raise HarnessError(
            f"Capture role {capture_role!r} requires stack {expected_stack!r}, "
            f"got {stack!r}"
        )
    normalized_batch_id = _canonical_uuid(batch_id, "batch ID")
    expected_commit = STACK_COMMITS[stack]
    resolved_root = repo_root.expanduser().resolve()
    actual_commit = _run_git(resolved_root, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise HarnessError(
            f"{stack} capture requires {expected_commit}, found {actual_commit}. "
            "Do not attribute later changes to this migration baseline."
        )
    dirty_paths = _run_git(resolved_root, "status", "--porcelain")
    if dirty_paths:
        raise HarnessError(
            f"{stack} capture checkout must be clean; found uncommitted paths:\n"
            f"{dirty_paths}"
        )
    harness_identity = _harness_identity()

    dependencies = _import_capture_dependencies(resolved_root)
    runtime_provenance = _capture_runtime_provenance(stack, dependencies, resolved_root)
    cp = dependencies["cp"]
    cases = [
        _capture_difference_in_differences(dependencies, REGISTERED_SAMPLING),
        _capture_synthetic_control(dependencies, REGISTERED_SAMPLING),
    ]
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "suite": SUITE_NAME,
        "provenance": {
            "stack": stack,
            "expected_commit": expected_commit,
            "actual_commit": actual_commit,
            "repo_root": str(resolved_root),
            "causalpy_path": str(Path(cp.__file__).resolve()),
            "checkout_clean": True,
            "capture_role": capture_role,
            "capture_ordinal": capture_ordinal,
            "capture_id": str(uuid.uuid4()),
            "batch_id": normalized_batch_id,
            "harness_path": harness_identity["path"],
            "harness_sha256": harness_identity["sha256"],
            "harness_commit": harness_identity["commit"],
            "harness_git_blob_sha256": harness_identity["git_blob_sha256"],
            "harness_checkout_clean": harness_identity["checkout_clean"],
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            **runtime_provenance,
        },
        "protocol": _protocol(
            _supports_nuts_sampler(dependencies["pm"]), REGISTERED_SAMPLING
        ),
        "cases": cases,
    }
    _validate_artifact(
        artifact, expected_harness_sha256=str(harness_identity["sha256"])
    )
    return artifact


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys at every nesting level."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class ArtifactInput:
    """One immutable artifact buffer used for both decision and reported hash."""

    path: Path
    sha256: str
    artifact: dict[str, Any]


def _read_artifact_input(path: Path) -> ArtifactInput:
    """Read, hash, parse, and validate one artifact from exactly one byte buffer."""
    try:
        contents = path.read_bytes()
        payload = json.loads(
            contents.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise HarnessError(f"Could not read artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise HarnessError(f"Artifact {path} must contain a JSON object")
    _assert_finite_json(payload, str(path))
    _validate_artifact(payload)
    return ArtifactInput(path=path, sha256=_sha256_bytes(contents), artifact=payload)


def _case_map(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index artifact cases while rejecting duplicated names."""
    cases = artifact.get("cases")
    if not isinstance(cases, list):
        raise HarnessError("Artifact cases must be a list")
    result: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            raise HarnessError("Each artifact case must have a string name")
        name = case["name"]
        if name in result:
            raise HarnessError(f"Artifact has duplicate case {name!r}")
        result[name] = case
    return result


def _series_map(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index case series while rejecting duplicate names."""
    series = case.get("series")
    if not isinstance(series, list):
        raise HarnessError(f"Case {case.get('name')!r} must contain a series list")
    result: dict[str, dict[str, Any]] = {}
    for item in series:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise HarnessError(f"Case {case.get('name')!r} has malformed series")
        name = item["name"]
        if name in result:
            raise HarnessError(
                f"Case {case.get('name')!r} has duplicate series {name!r}"
            )
        result[name] = item
    return result


def _metric_map(series: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index scalar metrics while rejecting duplicate identifiers."""
    metrics = series.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise HarnessError(f"Series {series.get('name')!r} has no captured metrics")
    result: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        if not isinstance(metric, dict) or not isinstance(metric.get("id"), str):
            raise HarnessError(f"Series {series.get('name')!r} has malformed metric")
        metric_id = metric["id"]
        if metric_id in result:
            raise HarnessError(
                f"Series {series.get('name')!r} has duplicate metric {metric_id!r}"
            )
        result[metric_id] = metric
    return result


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    """Require one untrusted JSON object to have exactly its registered keys."""
    if not isinstance(value, dict):
        raise HarnessError(f"{label} must be a JSON object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unknown = sorted(actual.difference(expected))
        details = []
        if missing:
            details.append(f"missing {missing!r}")
        if unknown:
            details.append(f"unknown {unknown!r}")
        raise HarnessError(f"{label} has invalid keys ({'; '.join(details)})")
    return value


def _require_string(value: Any, label: str) -> str:
    """Require a non-empty JSON string."""
    if not isinstance(value, str) or not value:
        raise HarnessError(f"{label} must be a non-empty string")
    return value


def _require_real(value: Any, label: str) -> float:
    """Require a finite non-boolean JSON number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarnessError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise HarnessError(f"{label} must be finite")
    return result


def _require_nonnegative_int(value: Any, label: str) -> int:
    """Require a nonnegative non-boolean JSON integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HarnessError(f"{label} must be a nonnegative integer")
    return value


def _json_equal(left: Any, right: Any) -> bool:
    """Compare canonical JSON while retaining JSON scalar types."""
    try:
        return _canonical_json(left) == _canonical_json(right)
    except (TypeError, ValueError):
        return False


def _validate_metric(metric: dict[str, Any]) -> None:
    """Validate one serialized scalar posterior metric."""
    _require_exact_keys(
        metric,
        {"id", "selector", "draw_digest", "summary"},
        "Metric",
    )
    metric_id = _require_string(metric["id"], "Metric ID")
    if not isinstance(metric["selector"], dict) or not all(
        isinstance(key, str) for key in metric["selector"]
    ):
        raise HarnessError(f"Metric {metric_id!r} has an invalid selector")
    digest = metric["draw_digest"]
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise HarnessError(f"Metric {metric_id!r} has an invalid draw digest")
    summary = _require_exact_keys(
        metric["summary"],
        {
            "mean",
            "posterior_sd",
            "mcse_mean",
            "rhat",
            "ess_bulk",
            "ess_tail",
            "hdi_lower",
            "hdi_upper",
        },
        f"Metric {metric_id!r} summary",
    )
    values = {
        key: _require_real(value, f"Metric {metric_id!r} {key}")
        for key, value in summary.items()
    }
    if values["posterior_sd"] < 0:
        raise HarnessError(f"Metric {metric_id!r} has a negative posterior SD")
    if values["mcse_mean"] < 0:
        raise HarnessError(f"Metric {metric_id!r} has a negative MCSE")
    if values["rhat"] <= 0:
        raise HarnessError(f"Metric {metric_id!r} has a non-positive R-hat")
    if values["hdi_lower"] > values["hdi_upper"]:
        raise HarnessError(f"Metric {metric_id!r} has inverted HDI bounds")
    if values["rhat"] > REGISTERED_SAMPLING.max_rhat:
        raise HarnessError(f"Metric {metric_id!r} exceeds the registered R-hat")
    if values["ess_bulk"] < REGISTERED_SAMPLING.min_ess_bulk:
        raise HarnessError(f"Metric {metric_id!r} fails the registered bulk ESS")
    if values["ess_tail"] < REGISTERED_SAMPLING.min_ess_tail:
        raise HarnessError(f"Metric {metric_id!r} fails the registered tail ESS")


def _validate_semantics(semantics: Any, label: str) -> None:
    """Validate the shape-coordinate relationship before manifest comparison."""
    semantics = _require_exact_keys(
        semantics,
        {"dims", "shape", "coords"},
        f"Series {label!r} semantics",
    )
    dimensions = semantics["dims"]
    if (
        not isinstance(dimensions, list)
        or len(dimensions) < 2
        or any(
            not isinstance(dimension, str) or not dimension for dimension in dimensions
        )
        or len(set(dimensions)) != len(dimensions)
        or dimensions[:2] != ["chain", "draw"]
    ):
        raise HarnessError(f"Series {label!r} has invalid sample dimensions")
    shape = semantics["shape"]
    if not isinstance(shape, list) or len(shape) != len(dimensions):
        raise HarnessError(f"Series {label!r} has an invalid shape")
    for index, size in enumerate(shape):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise HarnessError(
                f"Series {label!r} has invalid size at dimension {dimensions[index]!r}"
            )
    coordinates = semantics["coords"]
    if not isinstance(coordinates, dict) or set(coordinates) != set(dimensions):
        raise HarnessError(f"Series {label!r} has invalid coordinate keys")
    for dimension, size in zip(dimensions, shape, strict=True):
        values = coordinates[dimension]
        if not isinstance(values, list) or len(values) != size:
            raise HarnessError(
                f"Series {label!r} coordinate {dimension!r} does not match shape"
            )


def _validate_sampling_quality(quality: Any, case_name: str) -> None:
    """Validate typed per-case capture-validity evidence."""
    quality = _require_exact_keys(
        quality,
        {
            "divergences",
            "tree_depth_source",
            "tree_depth_events",
            "max_observed_tree_depth",
            "finite_values",
        },
        f"Case {case_name!r} sampling quality",
    )
    if (
        _require_nonnegative_int(
            quality["divergences"], f"Case {case_name!r} divergences"
        )
        != 0
    ):
        raise HarnessError(f"Case {case_name!r} is not divergence-free")
    source = quality["tree_depth_source"]
    if source not in {"not-exposed", "reached_max_treedepth", "tree_depth"}:
        raise HarnessError(f"Case {case_name!r} has invalid tree-depth source")
    if (
        _require_nonnegative_int(
            quality["tree_depth_events"], f"Case {case_name!r} tree-depth events"
        )
        != 0
    ):
        raise HarnessError(f"Case {case_name!r} has tree-depth saturation")
    maximum = quality["max_observed_tree_depth"]
    if source == "tree_depth":
        maximum = _require_nonnegative_int(
            maximum, f"Case {case_name!r} maximum tree depth"
        )
        if maximum >= REGISTERED_SAMPLING.max_treedepth:
            raise HarnessError(f"Case {case_name!r} has tree-depth saturation")
    elif maximum is not None:
        raise HarnessError(
            f"Case {case_name!r} reported a tree-depth maximum without tree_depth"
        )
    if quality["finite_values"] is not True:
        raise HarnessError(f"Case {case_name!r} did not prove finite captured values")


def _validate_series(series: Any, expected_name: str, expected: dict[str, Any]) -> None:
    """Validate every expected series, selector, and metric cardinality."""
    series = _require_exact_keys(
        series,
        {"name", "semantics", "metrics"},
        f"Series {expected_name!r}",
    )
    if series["name"] != expected_name:
        raise HarnessError(
            f"Series manifest expected {expected_name!r}, got {series['name']!r}"
        )
    _validate_semantics(series["semantics"], expected_name)
    if not _json_equal(series["semantics"], expected["semantics"]):
        raise HarnessError(f"Series {expected_name!r} semantics differ from manifest")
    metrics = series["metrics"]
    if not isinstance(metrics, list) or len(metrics) != len(expected["metrics"]):
        raise HarnessError(f"Series {expected_name!r} has the wrong metric cardinality")
    for metric, expected_metric in zip(metrics, expected["metrics"], strict=True):
        _validate_metric(metric)
        if metric["id"] != expected_metric["id"] or not _json_equal(
            metric["selector"], expected_metric["selector"]
        ):
            raise HarnessError(
                f"Series {expected_name!r} metric selector differs from manifest"
            )


def _validate_case(case: dict[str, Any]) -> None:
    """Validate a case against the immutable fixture and output manifest."""
    case = _require_exact_keys(
        case,
        {
            "name",
            "fixture",
            "sampling_quality",
            "effect_summary",
            "counterfactual",
            "series",
        },
        "Case",
    )
    case_name = _require_string(case["name"], "Case name")
    manifest = _scenario_manifest(REGISTERED_SAMPLING)
    if case_name not in manifest:
        raise HarnessError(f"Case {case_name!r} is not registered")
    expected = manifest[case_name]
    fixture = _require_exact_keys(
        case["fixture"],
        {"name", "records", "sha256"},
        f"Case {case_name!r} fixture",
    )
    if not isinstance(fixture["records"], list):
        raise HarnessError(f"Case {case_name!r} fixture records must be a list")
    if fixture["sha256"] != _sha256_text(_canonical_json(fixture["records"])):
        raise HarnessError(f"Case {case_name!r} fixture hash does not match records")
    if not _json_equal(fixture, expected["fixture"]):
        raise HarnessError(f"Case {case_name!r} fixture differs from manifest")
    _validate_sampling_quality(case["sampling_quality"], case_name)
    if not _json_equal(case["effect_summary"], expected["effect_summary"]):
        raise HarnessError(
            f"Case {case_name!r} effect-summary contract differs from manifest"
        )
    if not _json_equal(case["counterfactual"], expected["counterfactual"]):
        raise HarnessError(
            f"Case {case_name!r} counterfactual contract differs from manifest"
        )
    series = case["series"]
    if not isinstance(series, list):
        raise HarnessError(f"Case {case_name!r} series must be a list")
    series_by_name = _series_map(case)
    if set(series_by_name) != set(expected["series"]):
        raise HarnessError(f"Case {case_name!r} series differ from manifest")
    for series_name, series_manifest in expected["series"].items():
        _validate_series(series_by_name[series_name], series_name, series_manifest)


def _validate_runtime_provenance(provenance: dict[str, Any], stack: str) -> None:
    """Validate imported-environment identity and editable-install binding."""
    runtime = _require_exact_keys(
        provenance["runtime"],
        {"executable", "prefix", "module_paths", "causalpy_editable_target"},
        "Artifact runtime provenance",
    )
    prefix = Path(_require_string(runtime["prefix"], "Runtime prefix"))
    executable = Path(_require_string(runtime["executable"], "Runtime executable"))
    repo_root = Path(_require_string(provenance["repo_root"], "Repository root"))
    editable_target = Path(
        _require_string(
            runtime["causalpy_editable_target"], "CausalPy editable-install target"
        )
    )
    if not prefix.is_absolute() or not executable.is_absolute():
        raise HarnessError("Runtime prefix and executable must be absolute paths")
    if not repo_root.is_absolute() or not editable_target.is_absolute():
        raise HarnessError(
            "Repository root and CausalPy editable-install target must be absolute"
        )
    if not _is_within(executable, prefix):
        raise HarnessError("Runtime executable is outside its recorded prefix")
    if editable_target.resolve() != repo_root.resolve():
        raise HarnessError(
            "CausalPy editable-install target differs from recorded repository root"
        )
    module_paths = _require_exact_keys(
        runtime["module_paths"],
        {"arviz", "numpy", "pandas", "pymc", "pytensor", "xarray"},
        "Runtime module paths",
    )
    for name, path in module_paths.items():
        resolved = Path(_require_string(path, f"Runtime {name} module path"))
        if not resolved.is_absolute() or not _is_within(resolved, prefix):
            raise HarnessError(
                f"Runtime {name} module path is outside its recorded prefix"
            )
    dependencies = _require_exact_keys(
        provenance["dependencies"],
        {"arviz", "causalpy", "numpy", "pandas", "pymc", "pytensor", "xarray"},
        "Artifact dependency versions",
    )
    for name, version in dependencies.items():
        _require_string(version, f"Imported {name} version")
    for name, expected_major in STACK_RUNTIME_MAJORS[stack].items():
        if _module_major(dependencies[name], name) != expected_major:
            raise HarnessError(
                f"Artifact {stack} provenance has imported {name} major "
                f"{dependencies[name]!r}, expected {expected_major}"
            )


def _validate_artifact(
    artifact: dict[str, Any], *, expected_harness_sha256: str | None = None
) -> None:
    """Validate exact pins, runtime, protocol, manifest, and evidence types."""
    try:
        artifact = _require_exact_keys(
            artifact,
            {"schema_version", "suite", "provenance", "protocol", "cases"},
            "Artifact",
        )
        if artifact["schema_version"] != ARTIFACT_SCHEMA_VERSION:
            raise HarnessError("Unsupported artifact schema version")
        if artifact["suite"] != SUITE_NAME:
            raise HarnessError("Artifact comes from a different comparison suite")
        provenance = _require_exact_keys(
            artifact["provenance"],
            {
                "stack",
                "expected_commit",
                "actual_commit",
                "repo_root",
                "causalpy_path",
                "checkout_clean",
                "capture_role",
                "capture_ordinal",
                "capture_id",
                "batch_id",
                "harness_path",
                "harness_sha256",
                "harness_commit",
                "harness_git_blob_sha256",
                "harness_checkout_clean",
                "python",
                "python_implementation",
                "platform",
                "machine",
                "dependencies",
                "runtime",
            },
            "Artifact provenance",
        )
        stack = provenance["stack"]
        if stack not in STACK_COMMITS:
            raise HarnessError(f"Artifact has unknown stack {stack!r}")
        expected_commit = STACK_COMMITS[stack]
        if provenance["expected_commit"] != expected_commit:
            raise HarnessError(
                f"Artifact has an unexpected registered commit for {stack}"
            )
        if provenance["actual_commit"] != expected_commit:
            raise HarnessError(
                f"Artifact actual commit differs from the pinned {stack} revision"
            )
        if provenance["checkout_clean"] is not True:
            raise HarnessError("Artifact does not prove its sampled checkout was clean")
        capture_role = provenance["capture_role"]
        if capture_role not in CAPTURE_ROLES:
            raise HarnessError(f"Artifact has unknown capture role {capture_role!r}")
        expected_stack, expected_ordinal = CAPTURE_ROLES[capture_role]
        if stack != expected_stack or provenance["capture_ordinal"] != expected_ordinal:
            raise HarnessError("Artifact capture role does not match stack and ordinal")
        _canonical_uuid(provenance["capture_id"], "Artifact capture ID")
        _canonical_uuid(provenance["batch_id"], "Artifact batch ID")
        for key in (
            "repo_root",
            "causalpy_path",
            "harness_path",
            "python",
            "python_implementation",
            "platform",
            "machine",
        ):
            _require_string(provenance[key], f"Artifact provenance {key}")
        if not _is_within(
            Path(provenance["causalpy_path"]), Path(provenance["repo_root"])
        ):
            raise HarnessError(
                "Artifact CausalPy import path is outside its recorded checkout"
            )
        for key in ("harness_sha256", "harness_git_blob_sha256"):
            digest = provenance[key]
            if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                raise HarnessError(f"Artifact has invalid {key!r}")
        if provenance["harness_sha256"] != provenance["harness_git_blob_sha256"]:
            raise HarnessError("Artifact harness bytes differ from its Git blob")
        harness_commit = provenance["harness_commit"]
        if (
            not isinstance(harness_commit, str)
            or _COMMIT_PATTERN.fullmatch(harness_commit) is None
        ):
            raise HarnessError("Artifact has an invalid harness commit")
        if provenance["harness_checkout_clean"] is not True:
            raise HarnessError("Artifact does not prove its harness checkout was clean")
        if expected_harness_sha256 is not None:
            if _SHA256_PATTERN.fullmatch(expected_harness_sha256) is None:
                raise HarnessError("Comparator has an invalid executing harness digest")
            if provenance["harness_sha256"] != expected_harness_sha256:
                raise HarnessError(
                    "Artifact harness digest does not match the executing comparator"
                )
        _validate_runtime_provenance(provenance, stack)
        protocol = artifact["protocol"]
        if not isinstance(protocol, dict):
            raise HarnessError("Artifact is missing the preregistered protocol")
        sampling_record = protocol.get("sampling")
        if not isinstance(sampling_record, dict) or not isinstance(
            sampling_record.get("nuts_sampler_argument_used"), bool
        ):
            raise HarnessError("Artifact has an invalid NUTS sampler capability flag")
        if not _json_equal(
            protocol,
            _protocol(
                sampling_record["nuts_sampler_argument_used"], REGISTERED_SAMPLING
            ),
        ):
            raise HarnessError("Artifact protocol differs from the registered protocol")
        cases = _case_map(artifact)
        manifest = _scenario_manifest(REGISTERED_SAMPLING)
        if set(cases) != set(manifest):
            raise HarnessError(
                f"Artifact cases must be {sorted(manifest)!r}, got {sorted(cases)!r}"
            )
        for case in cases.values():
            _validate_case(case)
    except HarnessError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise HarnessError(f"Artifact has malformed evidence: {error}") from error


def _comparable_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Drop runtime implementation details while retaining every statistical setting."""
    comparable = json.loads(_canonical_json(protocol))
    comparable["sampling"].pop("nuts_sampler_argument_used", None)
    return comparable


def _runtime_identity(provenance: dict[str, Any]) -> dict[str, Any]:
    """Return the stable environment fields that repeats must share exactly."""
    return {
        key: provenance[key]
        for key in (
            "actual_commit",
            "repo_root",
            "causalpy_path",
            "checkout_clean",
            "harness_sha256",
            "harness_commit",
            "harness_git_blob_sha256",
            "harness_checkout_clean",
            "python",
            "python_implementation",
            "platform",
            "machine",
            "dependencies",
            "runtime",
        )
    }


def _repeatability(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Verify exact same-stack draw, summary, and validity reproducibility."""
    first_provenance = first["provenance"]
    second_provenance = second["provenance"]
    if first_provenance["stack"] != second_provenance["stack"]:
        raise HarnessError("Repeatability artifacts must come from the same stack")
    if not _json_equal(
        _runtime_identity(first_provenance), _runtime_identity(second_provenance)
    ):
        raise HarnessError("Repeatability artifacts have different runtime provenance")
    if not _json_equal(first["protocol"], second["protocol"]):
        raise HarnessError("Repeatability artifacts use different protocols")

    first_cases = _case_map(first)
    second_cases = _case_map(second)
    if set(first_cases) != set(second_cases):
        raise HarnessError("Repeatability artifacts have different scenario sets")

    draw_digest_mismatches: list[str] = []
    summary_mismatches: list[str] = []
    sampling_quality_mismatches: list[str] = []
    metric_count = 0
    for case_name in sorted(first_cases):
        first_case = first_cases[case_name]
        second_case = second_cases[case_name]
        if not _json_equal(first_case["fixture"], second_case["fixture"]):
            raise HarnessError(f"Repeatability fixture changed for {case_name!r}")
        if not _json_equal(
            first_case["sampling_quality"], second_case["sampling_quality"]
        ):
            sampling_quality_mismatches.append(case_name)
        first_series = _series_map(first_case)
        second_series = _series_map(second_case)
        if set(first_series) != set(second_series):
            raise HarnessError(f"Repeatability series changed for {case_name!r}")
        for series_name in sorted(first_series):
            first_item = first_series[series_name]
            second_item = second_series[series_name]
            if not _json_equal(first_item["semantics"], second_item["semantics"]):
                raise HarnessError(
                    f"Repeatability coordinate semantics changed for {series_name!r}"
                )
            first_metrics = _metric_map(first_item)
            second_metrics = _metric_map(second_item)
            if set(first_metrics) != set(second_metrics):
                raise HarnessError(f"Repeatability metrics changed for {series_name!r}")
            for metric_id in sorted(first_metrics):
                metric_count += 1
                first_metric = first_metrics[metric_id]
                second_metric = second_metrics[metric_id]
                if first_metric["draw_digest"] != second_metric["draw_digest"]:
                    draw_digest_mismatches.append(metric_id)
                if not _json_equal(first_metric["summary"], second_metric["summary"]):
                    summary_mismatches.append(metric_id)
    return {
        "stack": first_provenance["stack"],
        "metric_count": metric_count,
        "passed": not (
            draw_digest_mismatches or summary_mismatches or sampling_quality_mismatches
        ),
        "mismatched_draw_digest_metric_ids": draw_digest_mismatches,
        "mismatched_summary_metric_ids": summary_mismatches,
        "mismatched_sampling_quality_cases": sampling_quality_mismatches,
    }


def _validate_capture_batch(artifacts: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Require the four designated fresh roles in one coordinator batch."""
    expected_roles = (
        "reference_first",
        "reference_second",
        "candidate_first",
        "candidate_second",
    )
    batch_ids: set[str] = set()
    capture_ids: set[str] = set()
    for artifact, expected_role in zip(artifacts, expected_roles, strict=True):
        provenance = artifact["provenance"]
        if provenance["capture_role"] != expected_role:
            raise HarnessError(
                f"Artifact position requires capture role {expected_role!r}, got "
                f"{provenance['capture_role']!r}"
            )
        batch_ids.add(provenance["batch_id"])
        capture_ids.add(provenance["capture_id"])
    if len(batch_ids) != 1:
        raise HarnessError("All four captures must use one shared coordinator batch ID")
    if len(capture_ids) != len(artifacts):
        raise HarnessError("All four captures must have distinct capture IDs")
    return {
        "batch_id": next(iter(batch_ids)),
        "capture_ids": {
            role: artifact["provenance"]["capture_id"]
            for role, artifact in zip(expected_roles, artifacts, strict=True)
        },
    }


def _cross_stack_runtime_gate(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, bool]:
    """Require isolated prefixes while holding shared runtime dependencies fixed."""
    reference_provenance = reference["provenance"]
    candidate_provenance = candidate["provenance"]
    reference_dependencies = reference_provenance["dependencies"]
    candidate_dependencies = candidate_provenance["dependencies"]
    distinct_prefixes = (
        reference_provenance["runtime"]["prefix"]
        != candidate_provenance["runtime"]["prefix"]
    )
    same_platform = reference_provenance["platform"] == candidate_provenance["platform"]
    same_machine = reference_provenance["machine"] == candidate_provenance["machine"]
    same_python = reference_provenance["python"] == candidate_provenance["python"]
    same_python_implementation = (
        reference_provenance["python_implementation"]
        == candidate_provenance["python_implementation"]
    )
    same_numpy = reference_dependencies["numpy"] == candidate_dependencies["numpy"]
    same_pandas = reference_dependencies["pandas"] == candidate_dependencies["pandas"]
    same_xarray = reference_dependencies["xarray"] == candidate_dependencies["xarray"]
    return {
        "distinct_prefixes": distinct_prefixes,
        "same_platform": same_platform,
        "same_machine": same_machine,
        "same_python": same_python,
        "same_python_implementation": same_python_implementation,
        "same_numpy": same_numpy,
        "same_pandas": same_pandas,
        "same_xarray": same_xarray,
        "passed": (
            distinct_prefixes
            and same_platform
            and same_machine
            and same_python
            and same_python_implementation
            and same_numpy
            and same_pandas
            and same_xarray
        ),
    }


def _semantic_check(
    reference_case: dict[str, Any], candidate_case: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the hard schema and coordinate equality gate for one scenario."""
    failures: list[str] = []
    if reference_case["fixture"] != candidate_case["fixture"]:
        failures.append("serialized fixed fixture differs")
    if reference_case["effect_summary"] != candidate_case["effect_summary"]:
        failures.append("effect-summary table schema or 0.94 HDI metadata differs")
    if reference_case["counterfactual"] != candidate_case["counterfactual"]:
        failures.append("counterfactual prediction contract differs")

    reference_series = _series_map(reference_case)
    candidate_series = _series_map(candidate_case)
    if set(reference_series) != set(candidate_series):
        failures.append("captured series names differ")
    for series_name in sorted(set(reference_series).intersection(candidate_series)):
        reference_item = reference_series[series_name]
        candidate_item = candidate_series[series_name]
        if reference_item["semantics"] != candidate_item["semantics"]:
            failures.append(f"coordinate semantics differ for {series_name}")
        reference_metrics = _metric_map(reference_item)
        candidate_metrics = _metric_map(candidate_item)
        if set(reference_metrics) != set(candidate_metrics):
            failures.append(f"metric selectors differ for {series_name}")
    return {"passed": not failures, "failures": failures}


def _compare_scalar_summaries(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Apply only the preregistered numerical migration gates to one scalar."""
    reference_mean = float(reference["mean"])
    candidate_mean = float(candidate["mean"])
    difference = abs(candidate_mean - reference_mean)
    combined_mcse = math.hypot(
        float(reference["mcse_mean"]), float(candidate["mcse_mean"])
    )
    absolute_tolerance = max(
        MCSE_MULTIPLIER * combined_mcse,
        ABSOLUTE_TOLERANCE_FLOOR + RELATIVE_TOLERANCE_FLOOR * abs(reference_mean),
    )
    absolute_passed = difference <= absolute_tolerance

    pooled_posterior_sd = math.sqrt(
        (float(reference["posterior_sd"]) ** 2 + float(candidate["posterior_sd"]) ** 2)
        / 2.0
    )
    if pooled_posterior_sd <= DEGENERATE_POSTERIOR_SD:
        standardized_drift: float | None = None
        standardized_passed = absolute_passed
        standardized_rule = "degenerate posterior: use absolute mean gate"
    else:
        standardized_drift = difference / pooled_posterior_sd
        standardized_passed = standardized_drift <= MAX_STANDARDIZED_DRIFT
        standardized_rule = "abs(delta) / pooled posterior SD <= 0.1"

    reference_contains_candidate_mean = (
        float(reference["hdi_lower"]) <= candidate_mean <= float(reference["hdi_upper"])
    )
    candidate_contains_reference_mean = (
        float(candidate["hdi_lower"]) <= reference_mean <= float(candidate["hdi_upper"])
    )
    return {
        "reference_mean": reference_mean,
        "candidate_mean": candidate_mean,
        "absolute_mean_delta": difference,
        "combined_mcse": combined_mcse,
        "absolute_tolerance": absolute_tolerance,
        "absolute_mean_gate_passed": absolute_passed,
        "pooled_posterior_sd": pooled_posterior_sd,
        "standardized_mean_drift": standardized_drift,
        "standardized_mean_drift_rule": standardized_rule,
        "standardized_mean_drift_gate_passed": standardized_passed,
        "reference_hdi": [
            float(reference["hdi_lower"]),
            float(reference["hdi_upper"]),
        ],
        "candidate_hdi": [
            float(candidate["hdi_lower"]),
            float(candidate["hdi_upper"]),
        ],
        "reference_hdi_contains_candidate_mean": reference_contains_candidate_mean,
        "candidate_hdi_contains_reference_mean": candidate_contains_reference_mean,
        "mutual_hdi_containment_diagnostic": (
            reference_contains_candidate_mean and candidate_contains_reference_mean
        ),
    }


def _compare_case(
    reference_case: dict[str, Any], candidate_case: dict[str, Any]
) -> dict[str, Any]:
    """Compare independent posterior summaries for one fixed scenario."""
    semantic = _semantic_check(reference_case, candidate_case)
    reference_series = _series_map(reference_case)
    candidate_series = _series_map(candidate_case)
    metric_results: list[dict[str, Any]] = []

    for series_name in sorted(set(reference_series).intersection(candidate_series)):
        reference_metrics = _metric_map(reference_series[series_name])
        candidate_metrics = _metric_map(candidate_series[series_name])
        for metric_id in sorted(set(reference_metrics).intersection(candidate_metrics)):
            comparison = _compare_scalar_summaries(
                reference_metrics[metric_id]["summary"],
                candidate_metrics[metric_id]["summary"],
            )
            comparison["id"] = metric_id
            comparison["series"] = series_name
            comparison["passed"] = (
                comparison["absolute_mean_gate_passed"]
                and comparison["standardized_mean_drift_gate_passed"]
            )
            metric_results.append(comparison)

    return {
        "name": reference_case["name"],
        "semantic_equality": semantic,
        "metrics": metric_results,
        "passed": semantic["passed"] and all(item["passed"] for item in metric_results),
    }


def _capture_evidence(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return the provenance and observed validity diagnostics for one capture."""
    cases: list[dict[str, Any]] = []
    for case_name, case in sorted(_case_map(artifact).items()):
        metrics = [
            metric
            for series in _series_map(case).values()
            for metric in _metric_map(series).values()
        ]
        summaries = [metric["summary"] for metric in metrics]
        cases.append(
            {
                "name": case_name,
                "fixture_sha256": case["fixture"]["sha256"],
                "sampling_quality": case["sampling_quality"],
                "metric_count": len(metrics),
                "max_rhat": max(summary["rhat"] for summary in summaries),
                "min_ess_bulk": min(summary["ess_bulk"] for summary in summaries),
                "min_ess_tail": min(summary["ess_tail"] for summary in summaries),
            }
        )
    return {"provenance": artifact["provenance"], "cases": cases}


def compare_artifacts(
    reference_first: dict[str, Any],
    reference_second: dict[str, Any],
    candidate_first: dict[str, Any],
    candidate_second: dict[str, Any],
) -> dict[str, Any]:
    """Compare one fresh, role-bound PyMC 5/PyMC 6 evidence batch."""
    comparator = _harness_identity()
    artifacts = (
        reference_first,
        reference_second,
        candidate_first,
        candidate_second,
    )
    comparator_sha256 = str(comparator["sha256"])
    for artifact in artifacts:
        _validate_artifact(artifact, expected_harness_sha256=comparator_sha256)
        provenance = artifact["provenance"]
        for artifact_key, comparator_key in (
            ("harness_commit", "commit"),
            ("harness_git_blob_sha256", "git_blob_sha256"),
        ):
            if provenance[artifact_key] != comparator[comparator_key]:
                raise HarnessError(
                    f"Artifact {artifact_key} does not match the executing comparator"
                )
    if any(
        artifact["provenance"]["stack"] != "pymc5"
        for artifact in (reference_first, reference_second)
    ):
        raise HarnessError(
            "Reference artifacts must be captured from the pinned PyMC 5 stack"
        )
    if any(
        artifact["provenance"]["stack"] != "pymc6"
        for artifact in (candidate_first, candidate_second)
    ):
        raise HarnessError(
            "Candidate artifacts must be captured from the pinned PyMC 6 stack"
        )

    capture_batch = _validate_capture_batch(artifacts)
    reference_repeatability = _repeatability(reference_first, reference_second)
    candidate_repeatability = _repeatability(candidate_first, candidate_second)
    cross_stack_runtime = _cross_stack_runtime_gate(reference_first, candidate_first)
    reference_cases = _case_map(reference_first)
    candidate_cases = _case_map(candidate_first)
    if set(reference_cases) != set(candidate_cases):
        raise HarnessError(
            "Reference and candidate artifacts have different scenario sets"
        )

    protocol_equal = _json_equal(
        _comparable_protocol(reference_first["protocol"]),
        _comparable_protocol(candidate_first["protocol"]),
    )
    harness_equal = all(
        artifact["provenance"]["harness_sha256"] == comparator_sha256
        and artifact["provenance"]["harness_commit"] == comparator["commit"]
        and artifact["provenance"]["harness_git_blob_sha256"]
        == comparator["git_blob_sha256"]
        for artifact in artifacts
    )
    cases = [
        _compare_case(reference_cases[name], candidate_cases[name])
        for name in sorted(reference_cases)
    ]
    metric_results = [metric for case in cases for metric in case["metrics"]]
    semantic_passed = all(case["semantic_equality"]["passed"] for case in cases)
    absolute_passed = all(
        metric["absolute_mean_gate_passed"] for metric in metric_results
    )
    standardized_passed = all(
        metric["standardized_mean_drift_gate_passed"] for metric in metric_results
    )
    repeatability_passed = (
        reference_repeatability["passed"] and candidate_repeatability["passed"]
    )
    runtime_passed = cross_stack_runtime["passed"]
    passed = (
        protocol_equal
        and harness_equal
        and repeatability_passed
        and runtime_passed
        and semantic_passed
        and absolute_passed
        and standardized_passed
    )

    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "suite": SUITE_NAME,
        "reference": {
            "stack": "pymc5",
            "commit": reference_first["provenance"]["actual_commit"],
        },
        "candidate": {
            "stack": "pymc6",
            "commit": candidate_first["provenance"]["actual_commit"],
        },
        "comparator": {
            "harness_path": comparator["path"],
            "harness_sha256": comparator_sha256,
            "harness_commit": comparator["commit"],
            "harness_git_blob_sha256": comparator["git_blob_sha256"],
            "harness_checkout_clean": comparator["checkout_clean"],
        },
        "capture_batch": capture_batch,
        "cross_stack_runtime": cross_stack_runtime,
        "protocol_equal": protocol_equal,
        "harness_equal": harness_equal,
        "artifact_inputs": [],
        "capture_evidence": {
            "reference_first": _capture_evidence(reference_first),
            "reference_second": _capture_evidence(reference_second),
            "candidate_first": _capture_evidence(candidate_first),
            "candidate_second": _capture_evidence(candidate_second),
        },
        "within_stack_repeatability": {
            "reference": reference_repeatability,
            "candidate": candidate_repeatability,
        },
        "hard_gates": {
            "protocol_equal": protocol_equal,
            "same_executing_harness_version": harness_equal,
            "fresh_capture_batch_integrity": True,
            "isolated_stack_runtime": runtime_passed,
            "within_stack_repeatability": repeatability_passed,
            "semantic_table_and_coordinate_equality": semantic_passed,
            "absolute_mean_delta": absolute_passed,
            "standardized_mean_drift": standardized_passed,
        },
        "cases": cases,
        "passed": passed,
    }


def _format_number(value: float | None) -> str:
    """Format optional report values without introducing non-finite JSON values."""
    return "n/a" if value is None else f"{value:.8g}"


def _format_interval(bounds: list[float]) -> str:
    """Render one closed posterior interval for the Markdown comparison table."""
    lower, upper = bounds
    return f"[{_format_number(lower)}, {_format_number(upper)}]"


def render_report(comparison: dict[str, Any]) -> str:
    """Render an issue-attachment-ready Markdown report from a comparison result."""
    status = "PASS" if comparison["passed"] else "FAIL"
    reference = comparison["reference"]
    candidate = comparison["candidate"]
    comparator = comparison["comparator"]
    capture_batch = comparison["capture_batch"]
    cross_stack_runtime = comparison["cross_stack_runtime"]
    reference_repeat = comparison["within_stack_repeatability"]["reference"]
    candidate_repeat = comparison["within_stack_repeatability"]["candidate"]
    artifact_inputs = {item["role"]: item for item in comparison["artifact_inputs"]}
    capture_evidence = comparison["capture_evidence"]
    host_runtime_match = all(
        cross_stack_runtime[key]
        for key in (
            "same_platform",
            "same_machine",
            "same_python",
            "same_python_implementation",
        )
    )
    shared_dependency_versions_match = all(
        cross_stack_runtime[key] for key in ("same_numpy", "same_pandas", "same_xarray")
    )
    capture_labels = {
        "reference_first": "PyMC 5 / capture 1",
        "reference_second": "PyMC 5 / capture 2",
        "candidate_first": "PyMC 6 / capture 1",
        "candidate_second": "PyMC 6 / capture 2",
    }
    lines = [
        "# PyMC 5 → PyMC 6 migration baseline comparison",
        "",
        f"**Result:** {status}",
        "",
        "## Scope and attribution",
        "",
        f"- Reference: PyMC 5 CausalPy commit `{reference['commit']}`.",
        f"- Candidate: PyMC 6 migration commit `{candidate['commit']}`.",
        "- All sampled checkouts recorded an empty `git status --porcelain` before CausalPy import.",
        "- This report measures only the pinned migration delta. It must not be used to attribute behavior introduced after the PyMC 6 migration commit to the migration itself.",
        "- The artifacts contain independent posterior summaries; no raw PyMC 5 draw is compared to a PyMC 6 draw.",
        "",
        "## Comparator identity",
        "",
        f"- Executing harness: `{comparator['harness_path']}`.",
        f"- Harness Git commit: `{comparator['harness_commit']}`.",
        f"- Harness SHA-256: `{comparator['harness_sha256']}`.",
        f"- Checked Git blob SHA-256: `{comparator['harness_git_blob_sha256']}`.",
        f"- Harness checkout clean: `{comparator['harness_checkout_clean']}`.",
        "- Every input artifact was required to bind this executing harness SHA-256, Git blob SHA-256, and Git commit.",
        "",
        "## Pre-registered protocol",
        "",
        f"- Explicit HDI probability: `{HDI_PROB}` (`alpha={EFFECT_SUMMARY_ALPHA}`).",
        f"- Sampling: PyMC NUTS, `{REGISTERED_SAMPLING.chains}` serialized chains (`cores={CORES}`), `{REGISTERED_SAMPLING.tune}` tune iterations, `{REGISTERED_SAMPLING.draws}` retained draws, master seed `{REGISTERED_SAMPLING.master_seed}`, target acceptance `{REGISTERED_SAMPLING.target_accept}`, maximum tree depth `{REGISTERED_SAMPLING.max_treedepth}`.",
        "- `cores=1` is mandatory on local macOS to avoid Accelerate/numba fork failures and intentionally retained on every platform to serialize the chain schedule.",
        "- The harness captures a coefficient-based Difference-in-Differences scenario and a simplex-weighted Synthetic Control scenario from fixed serialized input records embedded in each artifact.",
        "- Every captured scalar must be finite, divergence-free, non-tree-depth-saturated when that statistic is exposed, have rank R-hat at most 1.01, and have bulk and tail ESS at least 400 before it is evidence.",
        f"- Tail ESS uses explicit probabilities `{TAIL_ESS_PROB}` on both stacks.",
        f"- Immutable scenario manifest version: `{SCENARIO_VERSION}`.",
        "",
        "## Fresh capture batch and isolated environments",
        "",
        f"- Coordinator batch ID: `{capture_batch['batch_id']}`.",
        "- The four role-bound capture UUIDs are distinct:",
    ]
    for role, label in capture_labels.items():
        lines.append(f"  - {label}: `{capture_batch['capture_ids'][role]}`.")
    lines.extend(
        [
            f"- Distinct PyMC environment prefixes: {'pass' if cross_stack_runtime['distinct_prefixes'] else 'fail'}.",
            (
                "- Matching platform, machine, Python version, and Python "
                "implementation: "
                f"{'pass' if host_runtime_match else 'fail'}."
            ),
            (
                "- Matching shared NumPy, pandas, and xarray versions: "
                f"{'pass' if shared_dependency_versions_match else 'fail'}."
            ),
            "",
            "## Within-stack deterministic repeatability",
            "",
            "| Stack | Metrics checked | Exact same-stack result |",
            "|---|---:|---|",
            f"| PyMC 5 | {reference_repeat['metric_count']} | {'pass' if reference_repeat['passed'] else 'fail'} |",
            f"| PyMC 6 | {candidate_repeat['metric_count']} | {'pass' if candidate_repeat['passed'] else 'fail'} |",
            "",
            "Exact raw-draw digests, posterior summaries, and sampling-quality "
            "evidence are checked only within a stack to establish repeatability. "
            "Raw draws are not used for any cross-stack decision.",
            "",
            "## Hard migration gates",
            "",
            "1. `abs(candidate_mean - reference_mean) <= max(4 * hypot(reference_mcse, candidate_mcse), 1e-6 + 1e-4 * abs(reference_mean))`.",
            "2. `abs(candidate_mean - reference_mean) / pooled_posterior_sd <= 0.1`, with a degenerate posterior using the absolute gate because a standardized denominator is undefined.",
            "3. Effect-summary table schemas, requested 0.94 HDI labels, metric selectors, prediction dimensions, and coordinate values must be semantically equal.",
            "",
            "Mutual mean-in-94%-HDI containment is diagnostic only. It is reported below but never changes the pass/fail decision.",
            "",
            "## Gate summary",
            "",
            "| Gate | Result |",
            "|---|---|",
        ]
    )
    for gate, passed in comparison["hard_gates"].items():
        lines.append(f"| {gate.replace('_', ' ')} | {'pass' if passed else 'fail'} |")
    lines.extend(
        [
            "",
            "## Capture provenance",
            "",
            "| Capture | Capture UUID | Artifact path | Artifact SHA-256 | Clean checkout | Runtime prefix | CausalPy editable target | CausalPy import | Harness SHA-256 | Harness commit | Package versions |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for role, label in capture_labels.items():
        evidence = capture_evidence[role]
        provenance = evidence["provenance"]
        dependencies = provenance["dependencies"]
        dependency_versions = ", ".join(
            f"{name} {version}" for name, version in sorted(dependencies.items())
        )
        artifact_input = artifact_inputs.get(role, {})
        lines.append(
            "| "
            f"{label} | `{provenance['capture_id']}` | "
            f"`{artifact_input.get('path', 'not supplied')}` | "
            f"`{artifact_input.get('sha256', 'not supplied')}` | "
            f"{provenance['checkout_clean']} | "
            f"`{provenance['runtime']['prefix']}` | "
            f"`{provenance['runtime']['causalpy_editable_target']}` | "
            f"`{provenance['causalpy_path']}` | "
            f"`{provenance['harness_sha256']}` | "
            f"`{provenance['harness_commit']}` | "
            f"{dependency_versions} |"
        )
    lines.extend(["", "Capture platforms:"])
    for role, label in capture_labels.items():
        provenance = capture_evidence[role]["provenance"]
        lines.append(
            f"- {label}: Python {provenance['python']} "
            f"({provenance['python_implementation']}) on "
            f"{provenance['platform']} ({provenance['machine']}); executable "
            f"`{provenance['runtime']['executable']}`."
        )
    lines.extend(["", "Imported module paths:"])
    for role, label in capture_labels.items():
        module_paths = capture_evidence[role]["provenance"]["runtime"]["module_paths"]
        rendered_paths = ", ".join(
            f"`{name}`: `{path}`" for name, path in sorted(module_paths.items())
        )
        lines.append(f"- {label}: {rendered_paths}.")
    lines.extend(
        [
            "",
            "## Capture validity evidence",
            "",
            "| Capture | Scenario | Fixture SHA-256 | Divergences | Tree-depth source / events / max | Finite values | Max R-hat | Min bulk ESS | Min tail ESS |",
            "|---|---|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for role, label in capture_labels.items():
        for case in capture_evidence[role]["cases"]:
            quality = case["sampling_quality"]
            tree_depth = (
                f"{quality['tree_depth_source']} / "
                f"{quality['tree_depth_events']} / "
                f"{_format_number(quality['max_observed_tree_depth'])}"
            )
            lines.append(
                "| "
                f"{label} | {case['name']} | `{case['fixture_sha256']}` | "
                f"{quality['divergences']} | "
                f"{tree_depth} | {'pass' if quality['finite_values'] else 'fail'} | "
                f"{_format_number(case['max_rhat'])} | "
                f"{_format_number(case['min_ess_bulk'])} | "
                f"{_format_number(case['min_ess_tail'])} |"
            )
    lines.extend(
        [
            "",
            "## Metric comparison",
            "",
            f"| Scenario | Metric | Reference mean | Candidate mean | Reference {HDI_PROB:.2f} HDI | Candidate {HDI_PROB:.2f} HDI | Absolute delta | Tolerance | Standardized drift | Mutual HDI diagnostic | Hard gates |",
            "|---|---|---:|---:|---|---|---:|---:|---:|---|---|",
        ]
    )
    for case in comparison["cases"]:
        for metric in case["metrics"]:
            diagnostic = "yes" if metric["mutual_hdi_containment_diagnostic"] else "no"
            lines.append(
                "| "
                f"{case['name']} | `{metric['id']}` | "
                f"{_format_number(metric['reference_mean'])} | "
                f"{_format_number(metric['candidate_mean'])} | "
                f"{_format_interval(metric['reference_hdi'])} | "
                f"{_format_interval(metric['candidate_hdi'])} | "
                f"{_format_number(metric['absolute_mean_delta'])} | "
                f"{_format_number(metric['absolute_tolerance'])} | "
                f"{_format_number(metric['standardized_mean_drift'])} | "
                f"{diagnostic} | {'pass' if metric['passed'] else 'fail'} |"
            )
    lines.extend(
        [
            "",
            "## Semantic contracts",
            "",
            "| Scenario | Effect table and coordinate equality | Details |",
            "|---|---|---|",
        ]
    )
    for case in comparison["cases"]:
        semantic = case["semantic_equality"]
        details = "; ".join(semantic["failures"]) if semantic["failures"] else "matched"
        lines.append(
            f"| {case['name']} | {'pass' if semantic['passed'] else 'fail'} | {details} |"
        )
    lines.extend(
        [
            "",
            "## Reuse for #157 correctness work",
            "",
            "The fixed fixtures, unrounded posterior summaries, explicit 0.94 HDI extraction, draw-wise R-squared computation, and schema checks are reusable test inputs. They are migration evidence only: #157 correctness tests must assert simulated-data ground truth with separately registered posterior-SD-unit bounds rather than treating the PyMC 5 posterior as truth.",
            "",
            "## Attachment checklist",
            "",
            "- Attach this rendered report and its four JSON artifacts to #1048 after coordinator execution.",
            "- Preserve the artifact SHA-256 metadata, fresh batch UUID, comparator identity, and command log with the attachment.",
            "- If a later commit is under investigation, create a separately named feature comparison; this harness intentionally rejects a source revision other than the two pinned commits above.",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_distinct_artifacts(paths: list[Path]) -> list[ArtifactInput]:
    """Read four independently named artifact buffers exactly once each."""
    resolved_paths = [path.expanduser().resolve() for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise HarnessError(
            "All four artifact paths must be distinct independent captures"
        )
    return [_read_artifact_input(path) for path in resolved_paths]


def _artifact_input_metadata(inputs: list[ArtifactInput]) -> list[dict[str, Any]]:
    """Record hashes from the exact artifact byte buffers used for comparison."""
    roles = (
        "reference_first",
        "reference_second",
        "candidate_first",
        "candidate_second",
    )
    return [
        {
            "role": role,
            "path": str(artifact_input.path),
            "sha256": artifact_input.sha256,
        }
        for role, artifact_input in zip(roles, inputs, strict=True)
    ]


def _capture_command(args: argparse.Namespace) -> int:
    """Execute one fresh role-bound capture and write an external JSON artifact."""
    output = _require_new_external_output(
        args.output,
        _script_repository_root(),
        args.repo_root.expanduser().resolve(),
    )
    artifact = _capture_artifact(
        args.stack,
        args.repo_root,
        batch_id=args.batch_id,
        capture_role=args.capture_role,
    )
    _atomic_write_json(output, artifact)
    print(output)
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    """Compare four immutable artifact buffers and create fresh decision outputs."""
    paths = [
        args.reference_first,
        args.reference_second,
        args.candidate_first,
        args.candidate_second,
    ]
    inputs = _load_distinct_artifacts(paths)
    artifacts = [artifact_input.artifact for artifact_input in inputs]
    resolved_paths = [artifact_input.path for artifact_input in inputs]
    tracked_roots = [_script_repository_root()]
    for artifact in artifacts:
        tracked_roots.append(Path(artifact["provenance"]["repo_root"]))
    output = _require_new_external_output(args.output, *tracked_roots)
    report_path = _require_new_external_output(args.report, *tracked_roots)
    if output == report_path:
        raise HarnessError(
            "Comparison JSON output and Markdown report must use different paths"
        )
    if output in resolved_paths or report_path in resolved_paths:
        raise HarnessError(
            "Comparison outputs must not overwrite an input evidence artifact"
        )

    comparison = compare_artifacts(*artifacts)
    comparison["artifact_inputs"] = _artifact_input_metadata(inputs)
    _atomic_write_json(output, comparison)
    _atomic_write_text(report_path, render_report(comparison))
    print(output)
    print(report_path)
    return 0 if comparison["passed"] else 1


def _build_parser() -> argparse.ArgumentParser:
    """Build the two-command CLI without exposing unregistered run settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture",
        help="Sample one pinned stack and write a strict baseline JSON artifact.",
    )
    capture.add_argument(
        "--capture-role",
        choices=sorted(CAPTURE_ROLES),
        required=True,
        help="Designated role in the four-capture coordinator batch.",
    )
    capture.add_argument(
        "--batch-id",
        required=True,
        help="Canonical UUID shared by all four fresh capture processes.",
    )
    capture.add_argument("--stack", choices=sorted(STACK_COMMITS), required=True)
    capture.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Pinned CausalPy checkout that must supply the imported package.",
    )
    capture.add_argument(
        "--output",
        type=Path,
        required=True,
        help="External JSON destination; tracked checkouts are rejected.",
    )
    capture.set_defaults(handler=_capture_command)

    compare = subparsers.add_parser(
        "compare",
        help="Verify within-stack repeats and compare independent stack summaries.",
    )
    compare.add_argument("--reference-first", type=Path, required=True)
    compare.add_argument("--reference-second", type=Path, required=True)
    compare.add_argument("--candidate-first", type=Path, required=True)
    compare.add_argument("--candidate-second", type=Path, required=True)
    compare.add_argument(
        "--output",
        type=Path,
        required=True,
        help="External comparison JSON destination; tracked checkouts are rejected.",
    )
    compare.add_argument(
        "--report",
        type=Path,
        required=True,
        help="External Markdown attachment destination; tracked checkouts are rejected.",
    )
    compare.set_defaults(handler=_compare_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and return an evidence-oriented process status."""
    args = _build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except HarnessError as error:
        print(f"migration baseline harness: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
