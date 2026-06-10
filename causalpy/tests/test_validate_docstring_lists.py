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
"""Tests for ``scripts/validate_docstring_lists.py``.

These tests guard the local pre-commit hook added in #892. Without them the
fixer/checker pair could silently regress (the docstring fixes were authored
before the checker existed, so we need explicit positive evidence that the
checker actually rejects every shape of offender it claims to cover).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_docstring_lists.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "validate_docstring_lists", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script_module():
    return _load_script_module()


@pytest.mark.parametrize("marker", ["-", "*", "+"])
def test_flags_inline_bullet_list_for_each_marker(
    tmp_path: Path, script_module, marker: str
) -> None:
    """The checker must flag every bullet-marker style RST/Markdown allows."""
    src = (
        f"def f():\n"
        f'    """Summary.\n'
        f"\n"
        f"    Returns\n"
        f"    -------\n"
        f"    dict\n"
        f"        Dictionary with keys:\n"
        f'        {marker} "a": value\n'
        f'        {marker} "b": value\n'
        f'    """\n'
        f"    return {{}}\n"
    )
    target = tmp_path / "broken.py"
    target.write_text(src)

    findings = script_module.check_file(target)

    assert findings, f"checker missed inline list with marker {marker!r}"
    assert all(path == target for path, _, _ in findings)


def test_passes_when_blank_line_separates_bullets(
    tmp_path: Path, script_module
) -> None:
    """Properly formatted lists must not produce findings."""
    src = (
        "def f():\n"
        '    """Summary.\n'
        "\n"
        "    Returns\n"
        "    -------\n"
        "    dict\n"
        "        Dictionary with keys:\n"
        "\n"
        '        - "a": value\n'
        '        - "b": value\n'
        '    """\n'
        "    return {}\n"
    )
    target = tmp_path / "clean.py"
    target.write_text(src)

    assert script_module.check_file(target) == []


def test_does_not_flag_bullet_continuation_lines(tmp_path: Path, script_module) -> None:
    """A second bullet whose preceding line is a wrapped first bullet must
    not be flagged. This is the false-positive class the checker explicitly
    avoids by treating bullet-prefixed previous lines (and their wraps) as
    list context rather than introductory prose.
    """
    src = (
        "def f():\n"
        '    """Summary.\n'
        "\n"
        "    Notes\n"
        "    -----\n"
        "    Cases:\n"
        "\n"
        "    - First bullet whose description wraps onto a second line\n"
        "      and ends without a colon\n"
        "    - Second bullet, immediately after the wrap line\n"
        '    """\n'
        "    return None\n"
    )
    target = tmp_path / "wrap.py"
    target.write_text(src)

    assert script_module.check_file(target) == []


def test_does_not_flag_list_when_intro_lacks_trailing_colon(
    tmp_path: Path, script_module
) -> None:
    """The checker is intentionally conservative: it only flags the ``intro:``
    pattern, since that's the unambiguous Napoleon-collapse case from #892.
    Lists introduced by other prose (no trailing colon) are not flagged to
    avoid false positives on prose that happens to be followed by a list.
    """
    src = (
        "def f():\n"
        '    """Summary.\n'
        "\n"
        "    Some prose without a trailing colon\n"
        "    - this is not flagged\n"
        '    """\n'
        "    return None\n"
    )
    target = tmp_path / "no_colon.py"
    target.write_text(src)

    assert script_module.check_file(target) == []


def test_cli_exits_nonzero_when_offenders_present(tmp_path: Path) -> None:
    """End-to-end CLI smoke test: invoking the script with a broken file
    must exit with a non-zero status, matching how pre-commit drives it.
    """
    broken = (
        "def f():\n"
        '    """Summary.\n'
        "\n"
        "    Returns\n"
        "    -------\n"
        "    dict\n"
        "        Dictionary with keys:\n"
        '        - "a": value\n'
        '    """\n'
        "    return {}\n"
    )
    target = tmp_path / "broken.py"
    target.write_text(broken)

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Inline bullet list detected" in result.stdout


def test_cli_exits_zero_on_clean_file(tmp_path: Path) -> None:
    """The CLI must succeed when given a properly-formatted file."""
    clean = (
        "def f():\n"
        '    """Summary.\n'
        "\n"
        "    Returns\n"
        "    -------\n"
        "    dict\n"
        "        Dictionary with keys:\n"
        "\n"
        '        - "a": value\n'
        '    """\n'
        "    return {}\n"
    )
    target = tmp_path / "clean.py"
    target.write_text(clean)

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == ""
