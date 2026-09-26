"""Inventory variadic signatures on CausalPy's documented public surface.

The audit uses source ASTs rather than importing CausalPy, so it is safe to run in
an uninstalled checkout. Public membership follows the documented API manifest:
``docs/source/api/index.md`` roots, package ``__all__`` exports, top-level
callable re-exports, named Tier 4 hooks, and explicit Tier 3 exclusions.

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


TIER_3_MODULE_EXCLUSIONS: dict[str, str] = {
    "causalpy.experiments.model_adapter": (
        "Backend adapter plumbing is stored only on "
        "BaseExperiment._model_backend; supported user access is "
        "BaseExperiment.model."
    ),
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


def _reject_tier_3_module(module: str) -> None:
    """Reject a Tier 3 module when a manifest or export would promote it."""
    if reason := TIER_3_MODULE_EXCLUSIONS.get(module):
        raise ValueError(
            f"{module} is explicitly Tier 3 and cannot be audited as public: {reason}"
        )


def _read_tree(module: str) -> ast.Module:
    """Parse one CausalPy source module."""
    path = _module_path(module)
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _all_names(tree: ast.Module) -> list[str] | None:
    """Return a literal module-level ``__all__``, or ``None`` when absent."""
    names: list[str] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in targets
        ):
            continue
        if not isinstance(value, (ast.List, ast.Tuple)):
            continue
        names = [
            element.value
            for element in value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
    return names


def _resolve_from_module(package: str, node: ast.ImportFrom) -> str:
    """Resolve an import-from node against its containing package or module."""
    if node.level == 0:
        return node.module or ""

    package_parts = package.split(".")
    if _module_path(package).name != "__init__.py":
        package_parts.pop()
    base_parts = package_parts[: len(package_parts) - node.level + 1]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _wildcard_export_names(
    module: str, *, seen: frozenset[str] = frozenset()
) -> set[str]:
    """Return names made public by ``from module import *``, recursively."""
    if module in seen:
        return set()
    tree = _read_tree(module)
    if (names := _all_names(tree)) is not None:
        return set(names)

    next_seen = seen | {module}
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            imported_module = _resolve_from_module(module, node)
            for alias in node.names:
                if alias.name == "*":
                    names.update(
                        _wildcard_export_names(imported_module, seen=next_seen)
                    )
                elif not (alias.asname or alias.name).startswith("_"):
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name.partition(".")[0]
                for alias in node.names
                if not (alias.asname or alias.name.partition(".")[0]).startswith("_")
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(
                target.id
                for target in targets
                if isinstance(target, ast.Name) and not target.id.startswith("_")
            )
    return names


def _assignment_source(
    value: ast.expr, bindings: dict[str, ImportedSymbol]
) -> ImportedSymbol | None:
    """Resolve a simple static alias to its imported symbol."""
    if isinstance(value, ast.Name):
        return bindings.get(value.id)

    attributes: list[str] = []
    while isinstance(value, ast.Attribute):
        attributes.append(value.attr)
        value = value.value
    if not isinstance(value, ast.Name) or not (source := bindings.get(value.id)):
        return None

    module = source.module
    if source.name is not None:
        submodule = f"{module}.{source.name}"
        if not _module_path(submodule).exists():
            return None
        module = submodule
    for attribute in reversed(attributes[1:]):
        submodule = f"{module}.{attribute}"
        if not _module_path(submodule).exists():
            return None
        module = submodule
    return ImportedSymbol(module=module, name=attributes[0])


def _import_bindings(
    package: str, tree: ast.Module
) -> tuple[dict[str, ImportedSymbol], list[str], dict[str, int]]:
    """Return ordered static import bindings, star-import sources, and lines."""
    bindings: dict[str, ImportedSymbol] = {}
    wildcard_modules: list[str] = []
    binding_lines: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = _resolve_from_module(package, node)
            for alias in node.names:
                if alias.name == "*":
                    wildcard_modules.append(module)
                    for name in _wildcard_export_names(module):
                        bindings[name] = ImportedSymbol(module=module, name=name)
                        binding_lines[name] = node.lineno
                else:
                    name = alias.asname or alias.name
                    bindings[name] = ImportedSymbol(module=module, name=alias.name)
                    binding_lines[name] = node.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                binding_name = alias.asname or alias.name.partition(".")[0]
                bindings[binding_name] = ImportedSymbol(
                    module=alias.name if alias.asname else binding_name,
                    name=None,
                )
                binding_lines[binding_name] = node.lineno
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is None:
                continue
            source = _assignment_source(value, bindings)
            if source is not None:
                for target in targets:
                    if isinstance(target, ast.Name):
                        bindings[target.id] = source
                        binding_lines[target.id] = node.lineno
    return bindings, wildcard_modules, binding_lines


def _package_exports(package: str) -> dict[str, ImportedSymbol]:
    """Return public ``__all__`` bindings and their static origins."""
    tree = _read_tree(package)
    bindings, wildcard_modules, _ = _import_bindings(package, tree)

    exports: dict[str, ImportedSymbol] = {}
    for name in _all_names(tree) or []:
        if name in bindings:
            exports[name] = bindings[name]
            continue
        for module in reversed(wildcard_modules):
            if name in _wildcard_export_names(module):
                exports[name] = ImportedSymbol(module=module, name=name)
                break
    return exports


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
        _reject_tier_3_module(f"{ROOT_MODULE}.{stripped}")
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


def _exported_symbol_callables(
    symbol: ImportedSymbol,
    *,
    seen: frozenset[tuple[str, str | None]] = frozenset(),
) -> list[CallableSignature]:
    """Return public callables for one exported binding, resolving re-exports."""
    _reject_tier_3_module(symbol.module)
    if symbol.name is None:
        return []

    identity = (symbol.module, symbol.name)
    if identity in seen:
        return []
    next_seen = seen | {identity}

    submodule = _module_path(f"{symbol.module}.{symbol.name}")
    if submodule.exists():
        _reject_tier_3_module(f"{symbol.module}.{symbol.name}")
        return []

    tree = _read_tree(symbol.module)
    declaration = next(
        (
            node
            for node in reversed(tree.body)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == symbol.name
        ),
        None,
    )
    bindings, wildcard_modules, binding_lines = _import_bindings(symbol.module, tree)
    if (source := bindings.get(symbol.name)) and (
        declaration is None or binding_lines[symbol.name] > declaration.lineno
    ):
        return _exported_symbol_callables(source, seen=next_seen)
    if declaration is not None:
        if isinstance(declaration, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return [_callable_signature(symbol.module, declaration)]
        return _class_callables(symbol.module, declaration)
    for module in reversed(wildcard_modules):
        if symbol.name in _wildcard_export_names(module):
            return _exported_symbol_callables(
                ImportedSymbol(module=module, name=symbol.name),
                seen=next_seen,
            )
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
        "- Explicit Tier 3 exclusion: `causalpy.experiments.model_adapter` is backend plumbing stored on `BaseExperiment._model_backend`; supported user access is `BaseExperiment.model`.",
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
