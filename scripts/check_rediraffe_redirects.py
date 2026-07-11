#!/usr/bin/env python3
"""Check that notebook renames/deletes have matching rediraffe redirects.

Compares ``docs/source/notebooks/`` against a base git branch (default:
``upstream/main``, then ``origin/main``, then ``main``). Any deleted or
renamed notebook/markdown page must appear as a key in ``rediraffe_redirects``
in ``docs/source/conf.py``. All redirect targets must exist on disk.

This mirrors the ``rediraffecheckdiff`` Sphinx builder without loading the
full docs stack during prek.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONF_PY = REPO_ROOT / "docs" / "source" / "conf.py"
NOTEBOOKS_DIR = REPO_ROOT / "docs" / "source" / "notebooks"
DOCS_SOURCE = REPO_ROOT / "docs" / "source"


def load_redirects() -> dict[str, str]:
    tree = ast.parse(CONF_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "rediraffe_redirects":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, dict):
                        raise TypeError("rediraffe_redirects must be a dict")
                    return value
    raise RuntimeError("rediraffe_redirects not found in docs/source/conf.py")


def resolve_compare_branch(explicit: str | None) -> str:
    candidates = [
        explicit,
        os.environ.get("REDIRAFFE_COMPARE_BRANCH"),
        "upstream/main",
        "origin/main",
        "main",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        result = subprocess.run(
            ["git", "rev-parse", "--verify", candidate],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return candidate
    msg = "Could not resolve a compare branch for rediraffe redirect checks."
    raise RuntimeError(msg)


def path_to_docname(path: str) -> str:
    rel = Path(path).resolve().relative_to(DOCS_SOURCE.resolve())
    return rel.with_suffix("").as_posix()


def target_exists(docname: str) -> bool:
    base = DOCS_SOURCE / docname
    return base.with_suffix(".ipynb").exists() or base.with_suffix(".md").exists()


def changed_notebook_paths(compare_branch: str) -> list[tuple[str, str]]:
    """Return ``(status, old_path)`` for deletes/renames under notebooks/."""
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            f"{compare_branch}...HEAD",
            "--",
            "docs/source/notebooks",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    changes: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") and len(parts) >= 2:
            old_path = parts[1]
            if old_path.endswith((".ipynb", ".md")) and Path(old_path).name not in {
                "index.md",
                "gallery.yaml",
            }:
                changes.append(("rename", old_path))
        elif status == "D" and len(parts) >= 2:
            old_path = parts[1]
            if old_path.endswith((".ipynb", ".md")) and Path(old_path).name not in {
                "index.md",
                "gallery.yaml",
            }:
                changes.append(("delete", old_path))
    return changes


def check_redirects(compare_branch: str | None = None) -> list[str]:
    errors: list[str] = []
    branch = resolve_compare_branch(compare_branch)
    redirects = load_redirects()

    for _kind, old_path in changed_notebook_paths(branch):
        docname = path_to_docname(old_path)
        if docname not in redirects:
            errors.append(
                f"{old_path} was removed or renamed since {branch} but "
                f"'{docname}' is missing from rediraffe_redirects in conf.py"
            )

    for _src, dst in redirects.items():
        if not target_exists(dst):
            errors.append(
                f"rediraffe_redirects target '{dst}' has no matching "
                ".ipynb or .md under docs/source/"
            )

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-branch",
        help="Git ref to diff against (default: upstream/main, origin/main, main)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = check_redirects(args.compare_branch)
    if errors:
        print("Rediraffe redirect check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            "\nAdd old docname -> new docname entries to rediraffe_redirects in "
            "docs/source/conf.py. Keep prior keys when renaming again.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
