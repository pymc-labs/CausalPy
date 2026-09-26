"""Verify public API export and documentation wiring.

Ensures every concrete ``BaseExperiment`` subclass is imported and listed in
``causalpy/experiments/__init__.py`` ``__all__`` and ``causalpy/__init__.py``
``__all__``, and that each package's imports stay in sync with ``__all__``.
Concrete ``Check`` implementations in ``causalpy/checks/`` must likewise appear
in ``causalpy/checks/__init__.py``. The Tier 1 ``causalpy.__all__`` surface must
be locally bound and documented by the ``causalpy`` ``automodule`` in
``docs/source/api/index.md``.

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
PACKAGE_INIT_PATH = REPO_ROOT / "causalpy" / "__init__.py"
API_INDEX_PATH = REPO_ROOT / "docs" / "source" / "api" / "index.md"
TOP_LEVEL_MODULE = "causalpy"


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
    """Return ``__all__`` names and top-level bindings from an ``__init__.py``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    all_names: set[str] = set()
    bound_names: set[str] = set()
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
                elif isinstance(target, ast.Name):
                    bound_names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound_names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    bound_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound_names.add(alias.asname or alias.name.partition(".")[0])
        elif isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)):
            bound_names.add(node.name)
    return all_names, bound_names


def _parse_automodule_options(path: Path) -> dict[str, list[set[str]]]:
    """Return reStructuredText ``automodule`` option sets keyed by module."""
    directives: dict[str, list[set[str]]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_index, line in enumerate(lines):
        directive = line.strip()
        if not directive.startswith(".. automodule::"):
            continue

        _, _, module = directive.partition("::")
        module = module.strip()
        if not module:
            continue

        options: set[str] = set()
        for option_line in lines[line_index + 1 :]:
            if not option_line.strip():
                continue
            if not option_line[0].isspace():
                break

            option = option_line.strip()
            if not option.startswith(":"):
                continue
            name, separator, _ = option[1:].partition(":")
            if separator:
                options.add(name)

        directives.setdefault(module, []).append(options)
    return directives


def check_top_level_exports(package_init: Path = PACKAGE_INIT_PATH) -> list[str]:
    """Return errors when a Tier 1 export is not locally bound."""
    all_names, bound_names = _parse_init_exports(package_init)
    return _format_set_diff(
        "causalpy/__init__.py __all__ vs top-level bindings",
        all_names - bound_names,
        set(),
    )


def check_top_level_api_docs(api_index: Path = API_INDEX_PATH) -> list[str]:
    """Return errors when Sphinx cannot render the Tier 1 root surface."""
    directives = _parse_automodule_options(api_index)
    top_level_directives = directives.get(TOP_LEVEL_MODULE, [])
    if not top_level_directives:
        return [
            "  Sphinx API index is missing an ``.. automodule:: causalpy`` directive."
        ]
    if not any("members" in options for options in top_level_directives):
        return [
            "  The ``causalpy`` automodule must include ``:members:`` so "
            "``causalpy.__all__`` is documented."
        ]
    if not any(
        {"members", "undoc-members"} <= options for options in top_level_directives
    ):
        return [
            "  The ``causalpy`` automodule must include ``:undoc-members:`` "
            "with ``:members:`` so every ``causalpy.__all__`` export is rendered."
        ]
    if not any(
        {"members", "undoc-members", "imported-members"} <= options
        for options in top_level_directives
    ):
        return [
            "  The ``causalpy`` automodule must include ``:imported-members:`` "
            "with ``:members:`` and ``:undoc-members:`` so root re-exports are "
            "documented."
        ]
    return []


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
    package_init = PACKAGE_INIT_PATH
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
    errors.extend(check_top_level_exports(package_init))
    errors.extend(check_top_level_api_docs())

    return errors


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the export/documentation wiring check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 when export or documentation wiring drifts from the codebase.",
    )
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("--check is required")

    errors = check_exports()
    if not errors:
        return 0

    print(
        "Public API export/documentation wiring drift detected. Update "
        "causalpy/__init__.py, causalpy/experiments/__init__.py, "
        "causalpy/checks/__init__.py, and docs/source/api/index.md so exports "
        "and Sphinx coverage match the public API policy."
    )
    print()
    for line in errors:
        print(line)
    return 1


if __name__ == "__main__":
    sys.exit(main())
