"""Validate that bullet lists in Python docstrings have a blank line before them.

This catches the rendering defect tracked in issue #892, where Sphinx's Napoleon
extension collapses an inline ``-`` bullet list (one that follows a non-blank
prose line ending in ``:``) into a single paragraph of literal hyphens. The fix
is to leave a blank line between the introductory sentence and the list. RST
requires that blank line; numpydoc-validation and ruff/pydocstyle don't enforce
it because they don't inspect section bodies.

Usage
-----

Run on a list of files (as pre-commit invokes it):

    python scripts/validate_docstring_lists.py path/to/file.py [...]

Or scan the package by default:

    python scripts/validate_docstring_lists.py

Exits with code 1 if any offending docstrings are found and prints their
locations; exits 0 otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _docstring_node_lineno(node: ast.AST) -> int:
    """Line number where this node's docstring starts (1-indexed)."""
    if isinstance(node, ast.Module):
        return 1
    body = getattr(node, "body", None)
    if not body or not isinstance(body[0], ast.Expr):
        return getattr(node, "lineno", 1)
    return body[0].lineno


# RST and CommonMark both accept ``-``, ``*``, and ``+`` as bullet markers,
# each followed by whitespace. Track all three so we don't miss alternative
# bullet styles (or false-positive on continuation lines that use them).
_BULLET_PREFIXES: tuple[str, ...] = ("- ", "* ", "+ ")


def _find_offenders_in_docstring(doc: str, base_lineno: int) -> list[int]:
    """Return absolute line numbers of bullet lines lacking a leading blank.

    A line is flagged when it starts with a bullet marker (``-``, ``*``, or
    ``+`` followed by whitespace) and the immediately preceding line is
    non-blank, ends with ``:``, and is not itself a bullet line. This matches
    the Napoleon-collapse pattern from #892 and avoids false positives on
    bullet-continuation lines.
    """
    offenders: list[int] = []
    lines = doc.split("\n")
    for i in range(1, len(lines)):
        cur_stripped = lines[i].lstrip()
        prev_stripped = lines[i - 1].rstrip()
        if not cur_stripped.startswith(_BULLET_PREFIXES):
            continue
        if not prev_stripped:
            continue
        prev_lstripped = prev_stripped.lstrip()
        if prev_lstripped.startswith(_BULLET_PREFIXES):
            continue
        if not prev_stripped.endswith(":"):
            continue
        offenders.append(base_lineno + i)
    return offenders


def check_file(path: Path) -> list[tuple[Path, int, str]]:
    """Return a list of (path, lineno, qualified_name) findings for a file."""
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    findings: list[tuple[Path, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        doc = ast.get_docstring(node, clean=False)
        if not doc:
            continue
        base = _docstring_node_lineno(node)
        for lineno in _find_offenders_in_docstring(doc, base):
            name = getattr(node, "name", "<module>")
            findings.append((path, lineno, name))
    return findings


def _iter_target_files(args: list[str]) -> list[Path]:
    if args:
        return [Path(a) for a in args if a.endswith(".py")]
    return list(Path("causalpy").rglob("*.py"))


def main(argv: list[str]) -> int:
    findings: list[tuple[Path, int, str]] = []
    for path in _iter_target_files(argv):
        findings.extend(check_file(path))

    if not findings:
        return 0

    print(
        "Inline bullet list detected without a preceding blank line in the "
        "following docstrings. Sphinx Napoleon will render these as a single "
        "paragraph of prose with the bullet markers inlined (see issue #892). "
        "Insert a blank line between the introductory line and the first bullet."
    )
    print()
    for path, lineno, name in findings:
        print(f"  {path}:{lineno}: {name}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
