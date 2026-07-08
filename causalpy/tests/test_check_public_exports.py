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
"""Tests for ``scripts/check_public_exports.py``."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_public_exports.py"
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("check_public_exports", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script_module():
    return _load_script_module()


def test_repo_export_wiring_is_current(script_module) -> None:
    """The repository export wiring should pass the checker."""
    assert script_module.check_exports() == []


def test_detects_synthetic_did_in_experiment_exports(script_module) -> None:
    """Synthetic DiD must be exported from ``experiments/__init__.py``."""
    discovered = script_module.discover_experiment_class_names(
        REPO_ROOT / "causalpy" / "experiments"
    )
    assert "SyntheticDifferenceInDifferences" in discovered

    experiments_init = REPO_ROOT / "causalpy" / "experiments" / "__init__.py"
    _, exp_imports = script_module._parse_init_exports(experiments_init)
    assert "SyntheticDifferenceInDifferences" in exp_imports


def test_cli_exits_zero_when_exports_are_current() -> None:
    """CLI ``--check`` should succeed on the current repository."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--check"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert result.stdout == ""


def test_cli_requires_check_flag() -> None:
    """The CLI should require an explicit ``--check`` flag."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert "--check is required" in result.stderr
