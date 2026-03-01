#   Copyright 2022 - 2026 The PyMC Labs Developers
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
"""Tests for the notebook schema validation script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_notebook, new_output

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_SCRIPT = REPO_ROOT / "scripts" / "validate_notebooks.py"


def _run_validator(notebook_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR_SCRIPT), str(notebook_path)],
        check=False,
        capture_output=True,
        text=True,
    )


def _build_notebook() -> dict:
    return new_notebook(
        cells=[
            new_code_cell(
                source="print('hello')",
                outputs=[
                    new_output(output_type="stream", name="stdout", text="hello\n")
                ],
            )
        ]
    )


def test_validate_notebooks_accepts_valid_notebook(tmp_path: Path) -> None:
    notebook_path = tmp_path / "valid.ipynb"
    with notebook_path.open("w", encoding="utf-8") as notebook_file:
        nbformat.write(_build_notebook(), notebook_file)

    result = _run_validator(notebook_path)

    assert result.returncode == 0
    assert result.stderr == ""


def test_validate_notebooks_reports_schema_details_for_invalid_output(
    tmp_path: Path,
) -> None:
    notebook_path = tmp_path / "invalid.ipynb"
    notebook = _build_notebook()
    notebook["cells"][0]["outputs"][0].pop("name")

    with notebook_path.open("w", encoding="utf-8") as notebook_file:
        json.dump(notebook, notebook_file)

    result = _run_validator(notebook_path)

    assert result.returncode == 1
    assert str(notebook_path) in result.stderr
    assert "cell[0]" in result.stderr
    assert "output[0]" in result.stderr
    assert "required property" in result.stderr
