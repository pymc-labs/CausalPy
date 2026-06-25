"""Verify public API export wiring for experiments and checks.

Ensures every concrete ``BaseExperiment`` subclass is imported and listed in
``causalpy/experiments/__init__.py`` ``__all__`` and ``causalpy/__init__.py``
``__all__``, and that each package's imports stay in sync with ``__all__``.
Concrete ``Check`` implementations in ``causalpy/checks/`` must likewise appear
in ``causalpy/checks/__init__.py``.

Usage
-----

    python scripts/check_public_exports.py --check

Exits with code 1 when drift is detected; exits 0 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
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
discover_check_class_names = _ast.discover_check_class_names
discover_experiment_class_names = _ast.discover_experiment_class_names


def _parse_init_exports(path: Path) -> tuple[set[str], set[str]]:
    """Return ``__all__`` names and imported public names from an ``__init__.py``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    all_names: set[str] = set()
    imported_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__all__"
                    and isinstance(node.value, (ast.List, ast.Tuple))
                ):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            all_names.add(elt.value)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "*":
                    continue
                imported_names.add(alias.asname or alias.name)
    return all_names, imported_names


def _format_set_diff(label: str, missing: set[str], extra: set[str]) -> list[str]:
    """Format missing/extra set differences as indented error lines."""
    lines: list[str] = []
    if missing:
        lines.append(f"  {label} missing: {', '.join(sorted(missing))}")
    if extra:
        lines.append(f"  {label} extra: {', '.join(sorted(extra))}")
    return lines


def check_exports() -> list[str]:
    """Return human-readable error lines; empty list means success."""
    errors: list[str] = []

    discovered_experiments = discover_experiment_class_names(
        REPO_ROOT / "causalpy" / "experiments"
    )
    experiments_init = REPO_ROOT / "causalpy" / "experiments" / "__init__.py"
    package_init = REPO_ROOT / "causalpy" / "__init__.py"
    checks_init = REPO_ROOT / "causalpy" / "checks" / "__init__.py"

    exp_all, exp_imports = _parse_init_exports(experiments_init)
    pkg_all, pkg_imports = _parse_init_exports(package_init)
    checks_all, checks_imports = _parse_init_exports(checks_init)
    discovered_checks = discover_check_class_names(REPO_ROOT / "causalpy" / "checks")

    errors.extend(
        _format_set_diff(
            "experiments/__init__.py vs discovered BaseExperiment subclasses",
            discovered_experiments - exp_all,
            exp_all - discovered_experiments,
        )
    )
    if exp_all != exp_imports:
        errors.append(
            "  experiments/__init__.py: __all__ and imports are out of sync "
            f"(only in __all__: {sorted(exp_all - exp_imports)}; "
            f"only imported: {sorted(exp_imports - exp_all)})"
        )

    errors.extend(
        _format_set_diff(
            "causalpy/__init__.py vs experiments/__init__.py",
            exp_all - pkg_all,
            {name for name in pkg_all - exp_all if name in discovered_experiments},
        )
    )
    for name in exp_all:
        if name not in pkg_imports:
            errors.append(f"  causalpy/__init__.py missing import for {name}")

    errors.extend(
        _format_set_diff(
            "checks/__init__.py vs discovered Check implementations",
            discovered_checks - checks_all,
            set(),
        )
    )
    missing_check_imports = checks_all - checks_imports
    if missing_check_imports:
        errors.append(
            "  checks/__init__.py missing imports for __all__ names: "
            f"{sorted(missing_check_imports)}"
        )

    return errors


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the export wiring check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 when export wiring drifts from the codebase.",
    )
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("--check is required")

    errors = check_exports()
    if not errors:
        return 0

    print(
        "Public export wiring drift detected. Update causalpy/__init__.py, "
        "causalpy/experiments/__init__.py, and causalpy/checks/__init__.py "
        "so imports and __all__ match the concrete experiment and check classes."
    )
    print()
    for line in errors:
        print(line)
    return 1


if __name__ == "__main__":
    sys.exit(main())
