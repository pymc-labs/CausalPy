"""AST helpers for structural checks without importing ``causalpy``."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentMetadata:
    class_name: str
    supports_ols: bool
    supports_bayes: bool
    default_model: str | None
    plot_is_stub: bool

    @property
    def backends(self) -> str:
        if self.supports_ols and self.supports_bayes:
            return "OLS + Bayes"
        if self.supports_bayes:
            return "Bayes only"
        if self.supports_ols:
            return "OLS only"
        return "none"


def _bases_include_base_experiment(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "BaseExperiment":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseExperiment":
            return True
    return False


def _class_bool_constant(node: ast.ClassDef, name: str) -> bool:
    for item in node.body:
        if not isinstance(item, ast.Assign):
            continue
        for target in item.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == name
                and isinstance(item.value, ast.Constant)
                and isinstance(item.value.value, bool)
            ):
                return item.value.value
    return False


def _class_name_constant(node: ast.ClassDef, name: str) -> str | None:
    for item in node.body:
        if not isinstance(item, ast.Assign):
            continue
        for target in item.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == name
                and isinstance(item.value, ast.Name)
            ):
                return item.value.id
    return None


def _plot_raises_not_implemented(node: ast.ClassDef) -> bool:
    for item in node.body:
        if not isinstance(item, ast.FunctionDef) or item.name != "plot":
            continue
        for stmt in ast.walk(item):
            if not isinstance(stmt, ast.Raise) or stmt.exc is None:
                continue
            exc = stmt.exc
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                return True
            if (
                isinstance(exc, ast.Call)
                and isinstance(exc.func, ast.Name)
                and exc.func.id == "NotImplementedError"
            ):
                return True
    return False


def _class_has_applicable_methods(node: ast.ClassDef) -> bool:
    for item in node.body:
        if (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id == "applicable_methods"
        ):
            return True
        if not isinstance(item, ast.Assign):
            continue
        for target in item.targets:
            if isinstance(target, ast.Name) and target.id == "applicable_methods":
                return True
    return False


def _experiment_metadata_from_class(node: ast.ClassDef) -> ExperimentMetadata:
    return ExperimentMetadata(
        class_name=node.name,
        supports_ols=_class_bool_constant(node, "supports_ols"),
        supports_bayes=_class_bool_constant(node, "supports_bayes"),
        default_model=_class_name_constant(node, "_default_model_class"),
        plot_is_stub=_plot_raises_not_implemented(node),
    )


def discover_experiment_metadata(
    experiments_dir: Path,
) -> dict[str, ExperimentMetadata]:
    """Return experiment metadata keyed by class name."""
    metadata: dict[str, ExperimentMetadata] = {}
    for path in sorted(experiments_dir.glob("*.py")):
        if path.name in {"__init__.py", "base.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and _bases_include_base_experiment(node):
                metadata[node.name] = _experiment_metadata_from_class(node)
    return metadata


def discover_experiment_class_names(experiments_dir: Path) -> set[str]:
    """Return concrete ``BaseExperiment`` subclass names."""
    return set(discover_experiment_metadata(experiments_dir))


def discover_check_class_names(checks_dir: Path) -> set[str]:
    """Return check class names that declare ``applicable_methods``."""
    names: set[str] = set()
    for path in sorted(checks_dir.glob("*.py")):
        if path.name in {"__init__.py", "base.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and _class_has_applicable_methods(node):
                names.add(node.name)
    return names
