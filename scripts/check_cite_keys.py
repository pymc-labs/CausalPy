"""Check that every cited bibtex key exists in the docs bibliography.

Sphinx only warns (``bibtex.key_not_found``) when a ``{cite}`` role names an
unknown key, and the docs build does not fail on warnings, so a misspelled key
can render as ``[?]`` on ``main`` unnoticed (issue #911). This script reads the
``.bib`` files listed in ``bibtex_bibfiles`` in ``docs/source/conf.py`` and
scans ``.md``, ``.rst`` and notebook markdown cells under ``docs/source`` for
MyST (``{cite:p}`key```) and RST (``:footcite:t:`key```) citation roles.

Keys are compared case-insensitively, as pybtex does when Sphinx resolves them.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_SOURCE = REPO_ROOT / "docs" / "source"

_BIB_ENTRY = re.compile(r"@(\w+)\s*[{(]\s*([^,\s]+)\s*,")
_NON_ENTRY_TYPES = {"comment", "preamble", "string"}
_CITE_ROLE = re.compile(
    r"(?:\{(?:foot)?cite(?::[a-z]+)?\}|:(?:foot)?cite(?::[a-z]+)?:)`([^`]+)`"
)


def load_bibfiles(docs_source: Path) -> list[Path]:
    """Return the ``.bib`` paths listed in ``bibtex_bibfiles`` in ``conf.py``."""
    tree = ast.parse((docs_source / "conf.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "bibtex_bibfiles":
                    return [docs_source / name for name in ast.literal_eval(node.value)]
    raise RuntimeError("bibtex_bibfiles not found in conf.py")


def load_keys(bibfiles: list[Path]) -> set[str]:
    """Return the lower-cased entry keys defined in ``bibfiles``."""
    keys: set[str] = set()
    for bibfile in bibfiles:
        for entry_type, key in _BIB_ENTRY.findall(bibfile.read_text(encoding="utf-8")):
            if entry_type.lower() not in _NON_ENTRY_TYPES:
                keys.add(key.lower())
    return keys


def _texts(path: Path) -> list[str]:
    """Return the text that can hold citations: the file, or a notebook's markdown cells."""
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
        return [
            "".join(cell["source"])
            if isinstance(cell["source"], list)
            else cell["source"]
            for cell in cells
            if cell.get("cell_type") == "markdown"
        ]
    return [path.read_text(encoding="utf-8")]


def find_unknown_citations(docs_source: Path) -> list[tuple[Path, str]]:
    """Return ``(file, key)`` for every cited key missing from the bibliography."""
    keys = load_keys(load_bibfiles(docs_source))
    unknown: list[tuple[Path, str]] = []
    paths = sorted(
        path
        for suffix in ("*.md", "*.rst", "*.ipynb")
        for path in docs_source.rglob(suffix)
        if "_build" not in path.parts and ".ipynb_checkpoints" not in path.parts
    )
    for path in paths:
        for text in _texts(path):
            for match in _CITE_ROLE.finditer(text):
                for key in match.group(1).split(","):
                    key = key.strip()
                    if key and key.lower() not in keys:
                        unknown.append((path, key))
    return unknown


def main() -> int:
    unknown = find_unknown_citations(DOCS_SOURCE)
    if not unknown:
        return 0
    print(
        "Citation keys not found in the docs bibliography. They render as [?]. "
        "Fix the key or add the entry to docs/source/references.bib:"
    )
    for path, key in unknown:
        print(f"  {path.relative_to(REPO_ROOT)}: {key}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
