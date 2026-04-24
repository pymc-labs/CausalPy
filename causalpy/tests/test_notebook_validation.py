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
import pytest
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_SCRIPT = REPO_ROOT / "scripts" / "validate_notebooks.py"
DOCS_NOTEBOOKS_DIR = REPO_ROOT / "docs" / "source" / "notebooks"


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


def _write_notebook(notebook_path: Path, notebook: dict) -> None:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    with notebook_path.open("w", encoding="utf-8") as notebook_file:
        nbformat.write(notebook, notebook_file)


def _write_docs_notebook(tmp_path: Path, name: str, cells: list) -> Path:
    notebook = new_notebook(cells=cells)
    notebook_path = tmp_path / "docs" / "source" / "notebooks" / name
    _write_notebook(notebook_path, notebook)
    return notebook_path


def test_validate_notebooks_accepts_valid_notebook(tmp_path: Path) -> None:
    notebook_path = tmp_path / "valid.ipynb"
    _write_notebook(notebook_path, _build_notebook())

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


def test_docs_notebook_with_single_h1_passes(tmp_path: Path) -> None:
    notebook_path = _write_docs_notebook(
        tmp_path,
        "single_h1.ipynb",
        cells=[
            new_markdown_cell(source="# Lone Title\n\nSome intro text."),
            new_markdown_cell(source="## Subsection"),
            new_code_cell(source="x = 1"),
        ],
    )

    result = _run_validator(notebook_path)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("description", "cells", "expected_count"),
    [
        (
            "no_h1",
            [
                new_markdown_cell(source="## Only a subsection\n\nNo top-level."),
                new_code_cell(source="x = 1"),
            ],
            0,
        ),
        (
            "multiple_h1_in_one_cell",
            [
                new_markdown_cell(source="# First\n\nIntro\n\n# Second\n\nMore."),
            ],
            2,
        ),
        (
            "multiple_h1_across_cells",
            [
                new_markdown_cell(source="# First"),
                new_markdown_cell(source="# Second"),
                new_markdown_cell(source="# Third"),
            ],
            3,
        ),
    ],
)
def test_docs_notebook_with_wrong_h1_count_fails(
    tmp_path: Path,
    description: str,
    cells: list,
    expected_count: int,
) -> None:
    notebook_path = _write_docs_notebook(tmp_path, f"{description}.ipynb", cells=cells)

    result = _run_validator(notebook_path)

    assert result.returncode == 1
    assert str(notebook_path) in result.stderr
    assert f"found {expected_count}" in result.stderr
    assert "exactly one top-level (#) markdown heading" in result.stderr


def test_python_comments_in_code_cells_do_not_count_as_h1(tmp_path: Path) -> None:
    notebook_path = _write_docs_notebook(
        tmp_path,
        "code_comments.ipynb",
        cells=[
            new_markdown_cell(source="# The Real Title"),
            new_code_cell(
                source=("# Calculate average weekly spend\n# Another comment\nx = 1")
            ),
        ],
    )

    result = _run_validator(notebook_path)

    assert result.returncode == 0, result.stderr


def test_fenced_code_block_in_markdown_does_not_count_as_h1(
    tmp_path: Path,
) -> None:
    """Replicates the its_lift_test.ipynb pattern: ```python``` block with
    `# comment` inside a markdown cell must not register as additional H1s."""
    markdown_with_fenced_python = (
        "## Key outputs\n"
        "\n"
        "```python\n"
        "# Calculate average weekly spend during the promo period\n"
        "# Extract mean lift statistics from the ITS analysis\n"
        "spend = 100\n"
        "```\n"
        "\n"
        "Some closing text."
    )
    notebook_path = _write_docs_notebook(
        tmp_path,
        "fenced_python_in_markdown.ipynb",
        cells=[
            new_markdown_cell(source="# Real Title"),
            new_markdown_cell(source=markdown_with_fenced_python),
        ],
    )

    result = _run_validator(notebook_path)

    assert result.returncode == 0, result.stderr


def test_tilde_fenced_block_in_markdown_does_not_count_as_h1(
    tmp_path: Path,
) -> None:
    markdown_with_tilde_fence = "Example:\n\n~~~python\n# Not a heading\ny = 2\n~~~\n"
    notebook_path = _write_docs_notebook(
        tmp_path,
        "tilde_fenced.ipynb",
        cells=[
            new_markdown_cell(source="# Real Title"),
            new_markdown_cell(source=markdown_with_tilde_fence),
        ],
    )

    result = _run_validator(notebook_path)

    assert result.returncode == 0, result.stderr


def test_h1_check_only_applies_to_docs_notebooks(tmp_path: Path) -> None:
    """Notebooks outside docs/source/notebooks/ are exempt from the H1 rule."""
    notebook = new_notebook(
        cells=[
            new_markdown_cell(source="# First"),
            new_markdown_cell(source="# Second"),
            new_markdown_cell(source="# Third"),
        ]
    )
    notebook_path = tmp_path / "scratch_notebook.ipynb"
    _write_notebook(notebook_path, notebook)

    result = _run_validator(notebook_path)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


def test_all_docs_notebooks_pass_h1_check() -> None:
    """Regression test: every checked-in docs notebook must satisfy the rule."""
    if not DOCS_NOTEBOOKS_DIR.is_dir():
        pytest.skip("docs/source/notebooks directory not present in this checkout")

    notebooks = sorted(DOCS_NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        pytest.skip("no notebooks found under docs/source/notebooks")

    result = subprocess.run(
        [sys.executable, str(VALIDATOR_SCRIPT), *map(str, notebooks)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
