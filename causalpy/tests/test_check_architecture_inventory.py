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
"""Tests for ``scripts/check_architecture_inventory.py``."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_architecture_inventory.py"
SCRIPTS_DIR = REPO_ROOT / "scripts"
ARCHITECTURE_PATH = REPO_ROOT / "ARCHITECTURE.md"


def _load_script_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "check_architecture_inventory", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script_module():
    return _load_script_module()


def test_repo_architecture_inventory_is_current(script_module) -> None:
    """The repository ``ARCHITECTURE.md`` inventory should match the code."""
    assert script_module.check_inventory(ARCHITECTURE_PATH) == []


def test_parses_architecture_inventory_table(script_module) -> None:
    """The parser should read experiment rows from ``ARCHITECTURE.md``."""
    rows = script_module._parse_architecture_inventory(ARCHITECTURE_PATH)
    assert "SyntheticDifferenceInDifferences" in rows
    assert rows["SyntheticDifferenceInDifferences"]["backends"] == "OLS + Bayes"


def test_introspected_backends_match_known_experiments(script_module) -> None:
    """Introspection should surface backend support and plot stubs."""
    inventory = script_module.introspected_inventory()
    assert inventory["RegressionKink"].backends == "Bayes only"
    assert inventory["PanelRegression"].default_model is None
    assert inventory["InstrumentalVariable"].plot_is_stub is True


def test_print_markdown_includes_all_experiments(script_module) -> None:
    """``--print-markdown`` output should include every experiment class."""
    markdown = script_module.print_markdown(ARCHITECTURE_PATH)
    assert "| Class | Method | Backends | Notable quirk |" in markdown
    assert "`SyntheticDifferenceInDifferences`" in markdown


def test_cli_exits_zero_when_inventory_is_current() -> None:
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


def test_cli_detects_backend_drift(tmp_path: Path, script_module) -> None:
    """Backend mismatches in the doc table should be reported as errors."""
    architecture = tmp_path / "ARCHITECTURE.md"
    architecture.write_text(
        "\n".join(
            [
                "## Experiment Inventory",
                "",
                "| Class | Method | Backends | Notable quirk |",
                "|-------|--------|----------|---------------|",
                "| `RegressionKink` | RKD | OLS + Bayes | wrong |",
            ]
        )
    )
    errors = script_module.check_inventory(architecture)
    assert any(
        "RegressionKink" in line and "backends mismatch" in line for line in errors
    )
