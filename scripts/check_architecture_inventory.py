"""Compare or emit the experiment inventory in ARCHITECTURE.md.

Introspects concrete ``BaseExperiment`` subclasses and checks that
``ARCHITECTURE.md`` documents each class with the correct backend support.
Use ``--print-markdown`` to regenerate table rows for paste into the doc.

Usage
-----

    python scripts/check_architecture_inventory.py --check
    python scripts/check_architecture_inventory.py --print-markdown

Exits with code 1 on drift when ``--check`` is used; exits 0 otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPTS_DIR.parent


def _load_ast_introspection():
    path = _SCRIPTS_DIR / "_ast_introspection.py"
    spec = importlib.util.spec_from_file_location("ast_introspection", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_ast = _load_ast_introspection()
ExperimentMetadata = _ast.ExperimentMetadata
discover_experiment_metadata = _ast.discover_experiment_metadata
ARCHITECTURE_PATH = REPO_ROOT / "ARCHITECTURE.md"
INVENTORY_HEADER = "## Experiment Inventory"


@dataclass(frozen=True)
class ExperimentInventoryRow:
    class_name: str
    backends: str
    default_model: str | None
    plot_is_stub: bool

    @classmethod
    def from_metadata(cls, metadata: ExperimentMetadata) -> ExperimentInventoryRow:
        """Build an inventory row from AST-derived experiment metadata."""
        return cls(
            class_name=metadata.class_name,
            backends=metadata.backends,
            default_model=metadata.default_model,
            plot_is_stub=metadata.plot_is_stub,
        )


def _parse_architecture_inventory(path: Path) -> dict[str, dict[str, str]]:
    """Parse the experiment inventory markdown table from ``ARCHITECTURE.md``."""
    text = path.read_text(encoding="utf-8")
    start = text.find(INVENTORY_HEADER)
    if start == -1:
        msg = f"{path}: missing {INVENTORY_HEADER!r} section"
        raise ValueError(msg)

    section = text[start:]
    table_start = section.find("| Class |")
    if table_start == -1:
        msg = f"{path}: experiment inventory table not found"
        raise ValueError(msg)

    table_lines: list[str] = []
    for line in section[table_start:].splitlines():
        if not line.startswith("|"):
            break
        table_lines.append(line)

    rows: dict[str, dict[str, str]] = {}
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        class_name = cells[0].strip("`")
        rows[class_name] = {
            "method": cells[1],
            "backends": cells[2],
            "quirk": cells[3] if len(cells) > 3 else "",
        }
    return rows


def _markdown_table_row(row: ExperimentInventoryRow, doc_row: dict[str, str]) -> str:
    """Format one experiment-inventory markdown table row."""
    method = doc_row.get("method", "TBD")
    quirk = doc_row.get("quirk", "")
    if row.default_model is None and "model required" not in quirk.lower():
        quirk = quirk or "No `_default_model_class`; model required"
    if row.plot_is_stub and "plot()" not in quirk:
        quirk = "no unified `plot()`" if not quirk else f"{quirk}; no unified `plot()`"
    return f"| `{row.class_name}` | {method} | {row.backends} | {quirk} |"


def introspected_inventory() -> dict[str, ExperimentInventoryRow]:
    """Build inventory rows by introspecting experiment class definitions."""
    metadata = discover_experiment_metadata(REPO_ROOT / "causalpy" / "experiments")
    return {
        name: ExperimentInventoryRow.from_metadata(meta)
        for name, meta in metadata.items()
    }


def check_inventory(path: Path = ARCHITECTURE_PATH) -> list[str]:
    """Compare ``ARCHITECTURE.md`` against introspected experiment metadata."""
    errors: list[str] = []
    truth = introspected_inventory()
    documented = _parse_architecture_inventory(path)

    missing = set(truth) - set(documented)
    extra = set(documented) - set(truth)
    if missing:
        errors.append(
            f"  ARCHITECTURE.md missing experiment rows: {', '.join(sorted(missing))}"
        )
    if extra:
        errors.append(
            f"  ARCHITECTURE.md stale experiment rows: {', '.join(sorted(extra))}"
        )

    for class_name, row in sorted(truth.items()):
        if class_name not in documented:
            continue
        doc_backends = documented[class_name]["backends"]
        if doc_backends != row.backends:
            errors.append(
                f"  {class_name}: backends mismatch "
                f"(doc={doc_backends!r}, code={row.backends!r})"
            )
    return errors


def print_markdown(path: Path = ARCHITECTURE_PATH) -> str:
    """Render a markdown table snippet for the experiment inventory."""
    truth = introspected_inventory()
    documented = _parse_architecture_inventory(path)
    header = (
        "| Class | Method | Backends | Notable quirk |\n"
        "|-------|--------|----------|---------------|"
    )
    body = "\n".join(
        _markdown_table_row(truth[name], documented.get(name, {}))
        for name in sorted(truth)
    )
    return f"{header}\n{body}"


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run or print the inventory check."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 when ARCHITECTURE.md inventory drifts from the code.",
    )
    group.add_argument(
        "--print-markdown",
        action="store_true",
        help="Print a markdown table snippet for the experiment inventory.",
    )
    args = parser.parse_args(argv)

    if args.print_markdown:
        print(print_markdown())
        return 0

    errors = check_inventory()
    if not errors:
        return 0

    print(
        "ARCHITECTURE.md experiment inventory drift detected. "
        "Update the table under '## Experiment Inventory' or run "
        "`python scripts/check_architecture_inventory.py --print-markdown` "
        "to regenerate rows."
    )
    print()
    for line in errors:
        print(line)
    return 1


if __name__ == "__main__":
    sys.exit(main())
