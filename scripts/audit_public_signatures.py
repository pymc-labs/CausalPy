"""Inventory variadic signatures on CausalPy's documented public surface.

The audit uses source ASTs rather than importing CausalPy, so it is safe to run in
an uninstalled checkout. Public membership follows the documented API manifest:
``docs/source/api/index.md`` roots, package ``__all__`` exports, top-level
callable re-exports, and the Tier 4 ``BaseExperiment`` protocol hooks.

Usage
-----

    python scripts/audit_public_signatures.py

The report is intended for issue and review evidence. It does not modify files
or validate imports.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
PACKAGE_ROOT = REPO_ROOT / "causalpy"
API_INDEX_PATH = REPO_ROOT / "docs" / "source" / "api" / "index.md"
ROOT_MODULE = "causalpy"

TIER_4_CALLABLES: dict[tuple[str, str], frozenset[str]] = {
    (
        "causalpy.experiments.base",
        "BaseExperiment",
    ): frozenset(
        {
            "__maketables_coef_table__",
            "__maketables_stat__",
            "__maketables_depvar__",
            "__maketables_vcov_info__",
            "__maketables_stat_labels__",
            "__maketables_default_stat_keys__",
        }
    ),
    ("causalpy.pymc_forecast_models", "PyMCForecastModel"): frozenset({"_clone"}),
    ("causalpy.pymc_models", "PyMCModel"): frozenset({"_clone"}),
    (
        "causalpy.pymc_models",
        "BayesianBasisExpansionTimeSeries",
    ): frozenset({"_clone"}),
    ("causalpy.pymc_models", "StateSpaceTimeSeries"): frozenset({"_clone"}),
}


@dataclass(frozen=True)
class ImportedSymbol:
    """A symbol imported by a package ``__init__.py``."""

    module: str
    name: str | None


@dataclass(frozen=True)
class CallableSignature:
    """A public callable declaration discovered without importing project code."""

    qualified_name: str
    source_path: Path
    line: int
    signature: str
    variadics: tuple[str, ...]


def _module_path(module: str) -> Path:
    """Return the source path for a package-qualified module name."""
    relative_parts = module.split(".")[1:]
    candidate = PACKAGE_ROOT.joinpath(*relative_parts)
    package_init = candidate / "__init__.py"
    if package_init.exists():
        return package_init
    return candidate.with_suffix(".py")


def _read_tree(module: str) -> ast.Module:
    """Parse one CausalPy source module."""
    path = _module_path(module)
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _all_names(tree: ast.Module) -> list[str]:
    """Return literal names from a module-level ``__all__`` declaration."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        return [
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
    return []


def _resolve_from_module(package: str, node: ast.ImportFrom) -> str:
    """Resolve an import-from node against its containing package."""
    if node.level == 0:
        return node.module or ""

    package_parts = package.split(".")
    base_parts = package_parts[: len(package_parts) - node.level + 1]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _package_exports(package: str) -> dict[str, ImportedSymbol]:
    """Return public ``__all__`` bindings and their static origins."""
    tree = _read_tree(package)
    bindings: dict[str, ImportedSymbol] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = _resolve_from_module(package, node)
            for alias in node.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = ImportedSymbol(
                        module=module,
                        name=alias.name,
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.partition(".")[0]] = ImportedSymbol(
                    module=alias.name,
                    name=None,
                )

    return {name: bindings[name] for name in _all_names(tree) if name in bindings}


def _documented_module_roots() -> list[str]:
    """Read module roots from the authoritative Sphinx API manifest."""
    roots: list[str] = []
    in_autosummary = False
    for line in API_INDEX_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == ".. autosummary::":
            in_autosummary = True
            continue
        if not in_autosummary:
            continue
        if stripped == "```":
            break
        if not stripped or stripped.startswith(":"):
            continue
        roots.append(f"{ROOT_MODULE}.{stripped}")
    return roots


def _function_variadics(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    """Return declared ``*args`` and ``**kwargs`` parameter spellings."""
    names: list[str] = []
    if node.args.vararg is not None:
        names.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg is not None:
        names.append(f"**{node.args.kwarg.arg}")
    return tuple(names)


def _callable_signature(
    module: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    class_name: str | None = None,
) -> CallableSignature:
    """Construct inventory metadata for one callable declaration."""
    path = _module_path(module)
    components = [module]
    if class_name is not None:
        components.append(class_name)
    components.append(node.name)
    return CallableSignature(
        qualified_name=".".join(components),
        source_path=path.relative_to(REPO_ROOT),
        line=node.lineno,
        signature=f"({ast.unparse(node.args)})",
        variadics=_function_variadics(node),
    )


def _class_callables(module: str, node: ast.ClassDef) -> list[CallableSignature]:
    """Return public and Tier 4 callable declarations owned by one public class."""
    tier_4_methods = TIER_4_CALLABLES.get((module, node.name), frozenset())
    callables: list[CallableSignature] = []
    for member in node.body:
        if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if (
            member.name != "__init__"
            and member.name.startswith("_")
            and member.name not in tier_4_methods
        ):
            continue
        callables.append(_callable_signature(module, member, class_name=node.name))
    return callables


def _module_callables(module: str) -> list[CallableSignature]:
    """Return every non-private callable declared by a documented module."""
    tree = _read_tree(module)
    callables: list[CallableSignature] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                callables.append(_callable_signature(module, node))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            callables.extend(_class_callables(module, node))
    return callables


def _exported_symbol_callables(symbol: ImportedSymbol) -> list[CallableSignature]:
    """Return public callables for one exported function or class binding."""
    if symbol.name is None:
        return []

    tree = _read_tree(symbol.module)
    for node in tree.body:
        if (
            not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            or node.name != symbol.name
        ):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return [_callable_signature(symbol.module, node)]
        if isinstance(node, ast.ClassDef):
            return _class_callables(symbol.module, node)
    return []


def _collect_public_callables() -> list[CallableSignature]:
    """Collect documented, exported, and Tier 4 public callables once each."""
    discovered: dict[tuple[Path, int], CallableSignature] = {}

    def add(callables: list[CallableSignature]) -> None:
        for callable_signature in callables:
            discovered[(callable_signature.source_path, callable_signature.line)] = (
                callable_signature
            )

    for module in _documented_module_roots():
        path = _module_path(module)
        if path.name == "__init__.py":
            for symbol in _package_exports(module).values():
                add(_exported_symbol_callables(symbol))
        else:
            add(_module_callables(module))

    for symbol in _package_exports(ROOT_MODULE).values():
        add(_exported_symbol_callables(symbol))

    base_tree = _read_tree("causalpy.experiments.base")
    base_class = next(
        node
        for node in base_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BaseExperiment"
    )
    add(_class_callables("causalpy.experiments.base", base_class))

    return sorted(discovered.values(), key=lambda item: item.qualified_name)


def _markdown_report(callables: list[CallableSignature]) -> str:
    """Render a review-ready Markdown inventory."""
    variadic = [item for item in callables if item.variadics]
    lines = [
        "# CausalPy public-signature survey",
        "",
        f"- Public callable declarations surveyed: **{len(callables)}**",
        f"- Declarations with `*args` or `**kwargs`: **{len(variadic)}**",
        "- Scope: documented API roots in `docs/source/api/index.md`, package `__all__` exports, top-level callable re-exports, and Tier 4 `BaseExperiment.__maketables_*__` / model `_clone()` hooks.",
        "- Excluded: tests and private implementation details outside that Tier 4 protocol.",
    ]
    if not variadic:
        lines.append("No variadic public signatures found.")
        return "\n".join(lines)

    lines.extend(
        [
            "| Public callable | Source | Variadic parameter(s) |",
            "| --- | --- | --- |",
        ]
    )
    for item in variadic:
        lines.append(
            f"| `{item.qualified_name}{item.signature}` | `{item.source_path}:{item.line}` | `{', '.join(item.variadics)}` |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and print the static audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(_markdown_report(_collect_public_callables()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
