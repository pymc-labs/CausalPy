#   Copyright 2022 - 2026 The PyMC Labs Developers
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
"""Signature-drift invariants for CausalPy's documented public callables.

Issue #886 established explicit, keyword-only concrete ``plot()`` contracts.
Issue #896 broadens that discipline to the whole documented public surface:
Sphinx, IDE autocomplete, ``inspect.signature``, and ``help()`` must expose
all supported arguments, and misspelled arguments must fail rather than
silently disappear into a generic catch-all.

The AST-only survey in ``scripts/audit_public_signatures.py`` is the single
scope definition shared by these runtime invariants and the issue inventory.
Only narrowly documented dynamic or third-party forwarders may keep
``**kwargs``; each requires an exact exemption below.

.. _pymc-labs/CausalPy#886: https://github.com/pymc-labs/CausalPy/issues/886
.. _pymc-labs/CausalPy#896: https://github.com/pymc-labs/CausalPy/issues/896
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
import pytest

import causalpy
from causalpy.experiments.base import BaseExperiment
from causalpy.pipeline import PipelineContext
from causalpy.steps import EstimateEffect


class _AuditedCallable(Protocol):
    """Minimal metadata interface shared with the AST-only survey."""

    qualified_name: str
    variadics: tuple[str, ...]


class _SignatureAudit(Protocol):
    """Typed interface exposed by the dynamically loaded AST survey."""

    def _collect_public_callables(self) -> list[_AuditedCallable]: ...

    def _markdown_report(self, callables: list[_AuditedCallable]) -> str: ...

    def _package_exports(self, package: str) -> dict[str, Any]: ...


def _all_base_experiment_subclasses() -> list[type]:
    """Import every experiment module and collect concrete subclasses."""
    package = importlib.import_module("causalpy.experiments")
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):  # pragma: no cover
            continue
        importlib.import_module(f"{package.__name__}.{module_info.name}")

    subclasses: list[type] = []
    seen: set[int] = set()
    stack: list[type] = list(BaseExperiment.__subclasses__())
    while stack:
        cls = stack.pop()
        if id(cls) in seen:  # pragma: no cover
            continue
        seen.add(id(cls))
        subclasses.append(cls)
        stack.extend(cls.__subclasses__())
    return subclasses


def _experiments_with_plot() -> Iterable[type]:
    """Yield concrete subclasses that declare an explicit public ``plot``."""
    for cls in _all_base_experiment_subclasses():
        if "plot" in cls.__dict__:  # pragma: no branch
            yield cls


_OVERRIDING_SUBCLASSES = list(_experiments_with_plot())
_EXPECTED_EXPERIMENTS = {
    "DifferenceInDifferences",
    "InstrumentalVariable",
    "InterruptedTimeSeries",
    "InversePropensityWeighting",
    "PanelRegression",
    "PiecewiseITS",
    "PrePostNEGD",
    "RegressionDiscontinuity",
    "RegressionKink",
    "StaggeredDifferenceInDifferences",
    "SyntheticControl",
    "SyntheticDifferenceInDifferences",
}


def _experiment_class(class_name: str) -> type:
    """Return one discovered concrete experiment by class name."""
    return next(cls for cls in _OVERRIDING_SUBCLASSES if cls.__name__ == class_name)


def test_every_concrete_subclass_declares_plot() -> None:
    """Every concrete experiment owns an explicit public ``plot`` method."""
    names = {cls.__name__ for cls in _OVERRIDING_SUBCLASSES}
    missing = _EXPECTED_EXPERIMENTS - names
    assert not missing, (
        "Expected every concrete experiment to declare its own plot(); "
        f"missing: {sorted(missing)}; discovered: {sorted(names)}"
    )


_POSTERIOR_OVER_X_PLOT_CLASSES = [
    "InterruptedTimeSeries",
    "DifferenceInDifferences",
    "PrePostNEGD",
    "RegressionDiscontinuity",
    "RegressionKink",
    "SyntheticControl",
    "SyntheticDifferenceInDifferences",
    "PiecewiseITS",
]


@pytest.mark.parametrize("class_name", _POSTERIOR_OVER_X_PLOT_CLASSES)
def test_posterior_plot_exposes_viz_kind(class_name: str) -> None:
    """Posterior-over-x plots expose their uncertainty-rendering controls."""
    parameters = inspect.signature(_experiment_class(class_name).__dict__["plot"])
    for parameter in ("kind", "ci_kind", "num_samples"):
        assert parameter in parameters.parameters, (
            f"{class_name}.plot() is missing {parameter!r}; all posterior-over-x "
            "plot classes must expose kind, ci_kind, and num_samples."
        )


@pytest.mark.parametrize(
    "cls",
    _OVERRIDING_SUBCLASSES,
    ids=lambda cls: cls.__name__,
)
def test_public_plot_has_no_var_positional_or_var_keyword(cls: type) -> None:
    """``plot`` must not declare ``*args`` or ``**kwargs`` at the surface."""
    parameters = inspect.signature(cls.__dict__["plot"]).parameters.values()
    bad = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    assert not bad, (
        f"{cls.__name__}.plot has VAR_POSITIONAL/VAR_KEYWORD parameter(s) "
        f"{[parameter.name for parameter in bad]}; replace them with explicit "
        "named parameters per issues #886 and #896."
    )


@pytest.mark.parametrize(
    "cls",
    _OVERRIDING_SUBCLASSES,
    ids=lambda cls: cls.__name__,
)
def test_public_plot_parameters_are_documented(cls: type) -> None:
    """Every named public ``plot`` parameter appears in its docstring."""
    plot_method = cls.__dict__["plot"]
    doc = plot_method.__doc__ or ""
    missing: list[str] = []
    for name, parameter in inspect.signature(plot_method).parameters.items():
        if name == "self":
            continue
        if parameter.kind in (  # pragma: no cover
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        numpydoc_pattern = re.compile(rf"^\s*{re.escape(name)}\s*:\s", re.MULTILINE)
        sphinx_pattern = re.compile(rf":param\s+(?:\S+\s+)?{re.escape(name)}\s*:")
        if not (numpydoc_pattern.search(doc) or sphinx_pattern.search(doc)):
            missing.append(name)
    assert not missing, (
        f"{cls.__name__}.plot signature has parameter(s) {missing} that are not "
        "documented in the docstring."
    )


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("causalpy.experiments.instrumental_variable", "InstrumentalVariable"),
        (
            "causalpy.experiments.inverse_propensity_weighting",
            "InversePropensityWeighting",
        ),
    ],
)
def test_stub_plot_raises_not_implemented(module_name: str, class_name: str) -> None:
    """Explicit unsupported ``plot`` methods fail loudly."""
    cls = getattr(importlib.import_module(module_name), class_name)
    with pytest.raises(NotImplementedError):
        object.__new__(cls).plot()


def test_base_experiment_has_no_public_plot() -> None:
    """The shared plot dispatcher remains protected rather than inherited."""
    assert "plot" not in BaseExperiment.__dict__
    assert hasattr(BaseExperiment, "_render_plot")
    assert _OVERRIDING_SUBCLASSES, "no concrete subclasses discovered"
    assert causalpy.__name__ == "causalpy"


def _load_signature_audit() -> _SignatureAudit:
    """Load the AST survey without importing project implementation modules."""
    audit_path = Path(__file__).parents[2] / "scripts" / "audit_public_signatures.py"
    spec = importlib.util.spec_from_file_location("public_signature_audit", audit_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(_SignatureAudit, module)


_SIGNATURE_AUDIT = _load_signature_audit()
_PUBLIC_SIGNATURES = _SIGNATURE_AUDIT._collect_public_callables()


def _configure_temporary_audit_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    root_init: str,
    experiments_init: str = "",
    module_adapter: str = "class ModelAdapter:\n    pass\n",
) -> None:
    """Point the AST audit at a tiny package with a controlled export graph."""
    package = tmp_path / "causalpy"
    experiments = package / "experiments"
    experiments.mkdir(parents=True)
    (package / "__init__.py").write_text(root_init, encoding="utf-8")
    (experiments / "__init__.py").write_text(experiments_init, encoding="utf-8")
    (experiments / "model_adapter.py").write_text(module_adapter, encoding="utf-8")
    api_index = tmp_path / "index.md"
    api_index.write_text(".. autosummary::\n```\n", encoding="utf-8")
    monkeypatch.setattr(_SIGNATURE_AUDIT, "PACKAGE_ROOT", package)
    monkeypatch.setattr(_SIGNATURE_AUDIT, "API_INDEX_PATH", api_index)


def test_adapter_tier_three_exclusion_is_reported_and_absent():
    """The private adapter seam is an explicit audit boundary, not an accident."""
    report = _SIGNATURE_AUDIT._markdown_report(_PUBLIC_SIGNATURES)
    assert "Explicit Tier 3 exclusion" in report
    assert "`causalpy.experiments.model_adapter`" in report
    assert "BaseExperiment._model_backend" in report
    assert not {
        candidate.qualified_name
        for candidate in _PUBLIC_SIGNATURES
        if candidate.qualified_name.startswith("causalpy.experiments.model_adapter.")
    }


def test_audit_rejects_adapter_as_documented_api_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Tier 3 adapter plumbing cannot enter the docs-derived public inventory."""
    api_index = tmp_path / "index.md"
    api_index.write_text(
        ".. autosummary::\n  experiments.model_adapter\n```\n", encoding="utf-8"
    )
    monkeypatch.setattr(_SIGNATURE_AUDIT, "API_INDEX_PATH", api_index)

    with pytest.raises(ValueError, match="model_adapter.*Tier 3"):
        _SIGNATURE_AUDIT._collect_public_callables()


@pytest.mark.parametrize(
    ("root_init", "experiments_init"),
    [
        (
            "from .experiments.model_adapter import ModelAdapter\n"
            '__all__ = ["ModelAdapter"]\n',
            "",
        ),
        (
            'from .experiments import PublicAdapter\n__all__ = ["PublicAdapter"]\n',
            "from .model_adapter import ModelAdapter as PublicAdapter\n",
        ),
        (
            "from .experiments.model_adapter import ModelAdapter as PublicAdapter\n"
            '__all__ = ["PublicAdapter"]\n',
            "",
        ),
        (
            "import causalpy.experiments.model_adapter as adapter\n"
            '__all__ = ["adapter"]\n',
            "",
        ),
        (
            "from .experiments import model_adapter as adapter\n"
            '__all__ = ["adapter"]\n',
            "",
        ),
        (
            "from .experiments.model_adapter import ModelAdapter\n"
            "PublicAdapter = ModelAdapter\n"
            '__all__ = ["PublicAdapter"]\n',
            "",
        ),
        (
            'from .experiments.model_adapter import *\n__all__ = ["ModelAdapter"]\n',
            "",
        ),
        (
            "from .experiments.model_adapter import ModelAdapter\n"
            '__all__: list[str] = ["ModelAdapter"]\n',
            "",
        ),
        (
            "from .experiments import model_adapter as adapter\n"
            "PublicAdapter = adapter.ModelAdapter\n"
            '__all__ = ["PublicAdapter"]\n',
            "",
        ),
        (
            'from .experiments import PublicAdapter\n__all__ = ["PublicAdapter"]\n',
            "from .model_adapter import ModelAdapter\n"
            "PublicAdapter: object = ModelAdapter\n",
        ),
        (
            'from .experiments import ModelAdapter\n__all__ = ["ModelAdapter"]\n',
            "from .model_adapter import *\n",
        ),
        (
            'from .experiments import *\n__all__ = ["ModelAdapter"]\n',
            "from .model_adapter import *\n",
        ),
        (
            "__all__ = []\n"
            "from .experiments.model_adapter import ModelAdapter\n"
            '__all__ = ["ModelAdapter"]\n',
            "",
        ),
        (
            "import causalpy.experiments.model_adapter\n"
            "PublicAdapter = causalpy.experiments.model_adapter.ModelAdapter\n"
            '__all__ = ["PublicAdapter"]\n',
            "",
        ),
        (
            "import causalpy.experiments as adapter\n"
            "PublicAdapter = adapter.model_adapter.ModelAdapter\n"
            '__all__ = ["PublicAdapter"]\n',
            "",
        ),
    ],
)
def test_audit_rejects_adapter_export_promotion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    root_init: str,
    experiments_init: str,
):
    """Every supported static re-export form preserves the Tier 3 exclusion."""
    _configure_temporary_audit_package(
        monkeypatch,
        tmp_path,
        root_init=root_init,
        experiments_init=experiments_init,
    )

    with pytest.raises(ValueError, match="model_adapter.*Tier 3"):
        _SIGNATURE_AUDIT._collect_public_callables()


def test_audit_resolves_star_imports_by_actual_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A later unrelated star import cannot hide an earlier adapter binding."""
    _configure_temporary_audit_package(
        monkeypatch,
        tmp_path,
        root_init=(
            "from .experiments.model_adapter import *\n"
            "from .safe import *\n"
            '__all__ = ["ModelAdapter"]\n'
        ),
        module_adapter="class ModelAdapter:\n    pass\n",
    )
    (tmp_path / "causalpy" / "safe.py").write_text(
        "class Safe:\n    pass\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="model_adapter.*Tier 3"):
        _SIGNATURE_AUDIT._collect_public_callables()


def test_audit_honors_later_star_import_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A later adapter star import overrides an earlier safe binding."""
    _configure_temporary_audit_package(
        monkeypatch,
        tmp_path,
        root_init=(
            "from .safe import ModelAdapter\n"
            "from .experiments.model_adapter import *\n"
            '__all__ = ["ModelAdapter"]\n'
        ),
    )
    (tmp_path / "causalpy" / "safe.py").write_text(
        "class ModelAdapter:\n    pass\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="model_adapter.*Tier 3"):
        _SIGNATURE_AUDIT._collect_public_callables()


def test_audit_honors_later_explicit_import_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A later explicit safe binding overrides an adapter star import."""
    _configure_temporary_audit_package(
        monkeypatch,
        tmp_path,
        root_init=(
            "from .experiments.model_adapter import *\n"
            "from .safe import ModelAdapter\n"
            '__all__ = ["ModelAdapter"]\n'
        ),
    )
    (tmp_path / "causalpy" / "safe.py").write_text(
        "class ModelAdapter:\n    pass\n", encoding="utf-8"
    )

    exported = _SIGNATURE_AUDIT._package_exports("causalpy")["ModelAdapter"]
    assert exported.module == "causalpy.safe"


def test_audit_honors_later_reexport_over_local_declaration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A later import overrides an earlier local declaration of the same name."""
    _configure_temporary_audit_package(
        monkeypatch,
        tmp_path,
        root_init='from .safe import ModelAdapter\n__all__ = ["ModelAdapter"]\n',
    )
    (tmp_path / "causalpy" / "safe.py").write_text(
        "class ModelAdapter:\n    pass\n"
        "from .experiments.model_adapter import ModelAdapter\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_adapter.*Tier 3"):
        _SIGNATURE_AUDIT._collect_public_callables()


def test_audit_honors_empty_all_during_star_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """An explicit empty ``__all__`` exports no names through a star import."""
    _configure_temporary_audit_package(
        monkeypatch,
        tmp_path,
        root_init='from .safe import *\n__all__ = ["Safe"]\n',
    )
    (tmp_path / "causalpy" / "safe.py").write_text(
        "class Safe:\n    pass\n__all__ = []\n", encoding="utf-8"
    )

    assert _SIGNATURE_AUDIT._package_exports("causalpy") == {}


def test_audit_stops_at_nonpublic_reexport_cycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Cycle protection avoids recursive static export resolution."""
    _configure_temporary_audit_package(
        monkeypatch,
        tmp_path,
        root_init="",
    )
    package = tmp_path / "causalpy"
    (package / "alpha.py").write_text("from .beta import Thing\n", encoding="utf-8")
    (package / "beta.py").write_text("from .alpha import Thing\n", encoding="utf-8")
    audit = cast(Any, _SIGNATURE_AUDIT)
    symbol_type = audit.ImportedSymbol
    resolver = audit._exported_symbol_callables

    assert resolver(symbol_type(module="causalpy.alpha", name="Thing")) == []


_FORWARDER_EXEMPTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "causalpy.steps.estimate_effect.EstimateEffect.__init__": (
        "kwargs",
        ("Other Parameters", "integrator-provided", "TypeError"),
    ),
    "causalpy.utils.plot_correlations": (
        "kwargs",
        ("Other Parameters", "seaborn.heatmap", "third-party forwarder"),
    ),
    "causalpy.experiments.base.BaseExperiment.fit": (
        "kwargs",
        ("Forwarded to the posterior sampler", "sample_kwargs"),
    ),
    "causalpy.experiments.base.BaseExperiment.sample_prior_predictive": (
        "kwargs",
        ("Forwarded to :func:`pymc.sample_prior_predictive`", "prior_sample_kwargs"),
    ),
    "causalpy.experiments.instrumental_variable.InstrumentalVariable.fit": (
        "kwargs",
        ("Keyword arguments are forwarded to", "ppc_sampler"),
    ),
    "causalpy.pymc_models.InstrumentalVariableRegression.sample_posterior": (
        "kwargs",
        ("Keyword arguments override ``sample_kwargs``.", "ppc_sampler"),
    ),
    "causalpy.pymc_models.PyMCModel.sample_posterior": (
        "kwargs",
        ("Keyword arguments override ``sample_kwargs`` for this call only.",),
    ),
    "causalpy.pymc_models.PyMCModel.sample_prior_predictive": (
        "kwargs",
        ("Keyword arguments override ``prior_sample_kwargs`` for this call only.",),
    ),
    "causalpy.pymc_models.StateSpaceTimeSeries.sample_posterior": (
        "kwargs",
        ("Keyword arguments override ``sample_kwargs``.",),
    ),
}
_NON_FORWARDING_PUBLIC_SIGNATURES = [
    candidate
    for candidate in _PUBLIC_SIGNATURES
    if candidate.qualified_name not in _FORWARDER_EXEMPTIONS
]


def _resolve_public_callable(
    qualified_name: str,
) -> tuple[Callable[..., Any], object | None]:
    """Resolve an AST-inventoried callable and its owning object."""
    parts = qualified_name.split(".")
    for module_end in range(len(parts), 0, -1):
        try:
            target: object = importlib.import_module(".".join(parts[:module_end]))
        except ModuleNotFoundError:
            continue

        owner: object | None = None
        for part in parts[module_end:]:
            owner = target
            target = getattr(target, part)
            if isinstance(target, property):
                assert target.fget is not None
                target = target.fget
        return cast(Callable[..., Any], target), owner
    raise AssertionError(f"Unable to resolve public callable {qualified_name!r}")


def test_forwarder_exemptions_exactly_match_variadic_inventory() -> None:
    """The allowlist has no stale entries and no unreviewed additions."""
    variadic_names = {
        candidate.qualified_name
        for candidate in _PUBLIC_SIGNATURES
        if candidate.variadics
    }
    assert variadic_names == set(_FORWARDER_EXEMPTIONS)


@pytest.mark.parametrize(
    "candidate",
    _PUBLIC_SIGNATURES,
    ids=lambda candidate: candidate.qualified_name,
)
def test_public_signatures_have_no_unexplained_variadics(
    candidate: _AuditedCallable,
) -> None:
    """Only explicitly justified public forwarders may retain ``**kwargs``."""
    allowed = _FORWARDER_EXEMPTIONS.get(candidate.qualified_name)
    if allowed is not None:
        parameter, _ = allowed
        assert candidate.variadics == (f"**{parameter}",)
        return

    assert not candidate.variadics, (
        f"{candidate.qualified_name} declares {candidate.variadics}; expose every "
        "supported parameter explicitly or add a narrowly documented forwarder "
        "exception under issue #896."
    )


@pytest.mark.parametrize(
    "candidate",
    _NON_FORWARDING_PUBLIC_SIGNATURES,
    ids=lambda candidate: candidate.qualified_name,
)
def test_public_signatures_reject_unexpected_keywords(
    candidate: _AuditedCallable,
) -> None:
    """Every non-forwarding public signature rejects a misspelled keyword."""
    callable_obj, _ = _resolve_public_callable(candidate.qualified_name)
    with pytest.raises(TypeError):
        inspect.signature(callable_obj).bind_partial(__causalpy_signature_typo__=None)


@pytest.mark.parametrize(
    ("qualified_name", "expected_parameter", "required_fragments"),
    [
        (qualified_name, parameter, fragments)
        for qualified_name, (parameter, fragments) in _FORWARDER_EXEMPTIONS.items()
    ],
)
def test_public_forwarder_exemptions_are_narrowly_documented(
    qualified_name: str,
    expected_parameter: str,
    required_fragments: tuple[str, ...],
) -> None:
    """Each variadic exception documents its exact forwarding contract."""
    callable_obj, owner = _resolve_public_callable(qualified_name)
    variadics = [
        parameter
        for parameter in inspect.signature(callable_obj).parameters.values()
        if parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    assert [(parameter.kind, parameter.name) for parameter in variadics] == [
        (inspect.Parameter.VAR_KEYWORD, expected_parameter)
    ]

    documented_object = owner if qualified_name.endswith(".__init__") else callable_obj
    doc = inspect.getdoc(documented_object) or ""
    # Every forwarder documents its kwargs contract under a uniform
    # "Other Parameters" section (numpydoc); the per-entry fragments then
    # pin each exact forwarding surface.
    assert "Other Parameters" in doc, (
        f"{qualified_name} must document its forwarding contract under an "
        "'Other Parameters' section"
    )
    for fragment in required_fragments:
        assert fragment in doc


def test_base_experiment_owns_live_lifecycle_verbs() -> None:
    """The lazy lifecycle verbs are real methods on BaseExperiment.

    Under the lazy lifecycle (issue #1092) ``fit`` and
    ``sample_prior_predictive`` are the shared, implemented verbs every
    experiment inherits; ``get_plot_data`` remains per-experiment rather
    than a generic inherited dispatcher.
    """
    assert "fit" in BaseExperiment.__dict__
    assert "sample_prior_predictive" in BaseExperiment.__dict__
    assert "get_plot_data" not in BaseExperiment.__dict__


_GET_PLOT_DATA_EXPERIMENTS = {
    "InterruptedTimeSeries",
    "PanelRegression",
    "PiecewiseITS",
    "StaggeredDifferenceInDifferences",
    "SyntheticControl",
}
_NO_GET_PLOT_DATA_EXPERIMENTS = {
    "DifferenceInDifferences",
    "InstrumentalVariable",
    "InversePropensityWeighting",
    "PrePostNEGD",
    "RegressionDiscontinuity",
    "RegressionKink",
    "SyntheticDifferenceInDifferences",
}


def test_supported_get_plot_data_methods_are_concrete_and_keyword_only() -> None:
    """Supported plot-data views own explicit keyword-only signatures."""
    for class_name in _GET_PLOT_DATA_EXPERIMENTS:
        method = _experiment_class(class_name).__dict__.get("get_plot_data")
        assert method is not None, f"{class_name} must own get_plot_data()."
        parameters = inspect.signature(method).parameters.values()
        assert all(
            parameter.name == "self" or parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in parameters
        ), f"{class_name}.get_plot_data() must expose optional arguments by keyword."


def test_unsupported_get_plot_data_is_not_accidentally_inherited() -> None:
    """Unsupported experiments expose no misleading generic plot-data API."""
    for class_name in _NO_GET_PLOT_DATA_EXPERIMENTS:
        cls = _experiment_class(class_name)
        assert "get_plot_data" not in cls.__dict__
        assert not hasattr(cls, "get_plot_data")


def test_every_concrete_subclass_declares_effect_summary() -> None:
    """Each experiment owns its explicit effect-summary contract."""
    for cls in _OVERRIDING_SUBCLASSES:
        assert "effect_summary" in cls.__dict__, (
            f"{cls.__name__} must declare effect_summary() rather than inherit "
            "a generic base signature."
        )


def test_representative_cutover_signatures_reject_typos_at_call_time() -> None:
    """Changed concrete API families reject typos before executing their bodies."""
    did = _experiment_class("DifferenceInDifferences")
    did_instance: Any = object.__new__(did)
    did_init = cast(Callable[..., Any], did.__dict__["__init__"])
    did_effect_summary = cast(Callable[..., Any], did.__dict__["effect_summary"])
    with pytest.raises(TypeError):
        did_init(
            did_instance,
            data=None,
            formula="",
            time_variable_name="",
            group_variable_name="",
            __causalpy_signature_typo__=None,
        )
    with pytest.raises(TypeError):
        did_effect_summary(did_instance, __causalpy_signature_typo__=None)

    pymc_predict = cast(Callable[..., Any], causalpy.pymc_models.PyMCModel.predict)
    pymc_score = cast(Callable[..., Any], causalpy.pymc_models.PyMCModel.score)
    with pytest.raises(TypeError):
        pymc_predict(None, X=None, __causalpy_signature_typo__=None)
    with pytest.raises(TypeError):
        pymc_score(None, X=None, y=None, __causalpy_signature_typo__=None)


def test_estimate_effect_validates_constructor_keywords_before_running() -> None:
    """The dynamic constructor forwarder reports built-in typos during validation."""
    step = EstimateEffect(
        _experiment_class("DifferenceInDifferences"),
        __causalpy_signature_typo__=None,
    )
    with pytest.raises(TypeError, match="Invalid constructor arguments"):
        step.validate(PipelineContext(data=pd.DataFrame()))


def test_estimate_effect_validates_missing_constructor_arguments() -> None:
    """The dynamic constructor forwarder rejects incomplete built-in setup."""
    step = EstimateEffect(
        _experiment_class("DifferenceInDifferences"),
        formula="",
    )
    with pytest.raises(TypeError, match="Invalid constructor arguments"):
        step.validate(PipelineContext(data=pd.DataFrame()))


# Floor for the AST inventory that drives every parametrized invariant in this module. The inventory is discovered from ``docs/source/api/index.md`` and the package ``__all__`` exports, so deleting a documentation entry or an export silently shrinks it — and a shrunken inventory makes the parametrized tests below pass by simply not running. Raise this floor when the public surface grows; never lower it to make a failure go away.
_MINIMUM_PUBLIC_SIGNATURE_COUNT = 200

# Representative members that must always be inventoried: every concrete experiment constructor plus one model, checks, and pipeline member. These pin the inventory's breadth, not just its size, so dropping a whole family (all of ``checks/*``, say) cannot hide behind the count floor.
_INVENTORY_CANARIES = (
    "causalpy.experiments.diff_in_diff.DifferenceInDifferences.__init__",
    "causalpy.experiments.instrumental_variable.InstrumentalVariable.__init__",
    "causalpy.experiments.interrupted_time_series.InterruptedTimeSeries.__init__",
    "causalpy.experiments.inverse_propensity_weighting.InversePropensityWeighting.__init__",
    "causalpy.experiments.panel_regression.PanelRegression.__init__",
    "causalpy.experiments.piecewise_its.PiecewiseITS.__init__",
    "causalpy.experiments.prepostnegd.PrePostNEGD.__init__",
    "causalpy.experiments.regression_discontinuity.RegressionDiscontinuity.__init__",
    "causalpy.experiments.regression_kink.RegressionKink.__init__",
    "causalpy.experiments.staggered_did.StaggeredDifferenceInDifferences.__init__",
    "causalpy.experiments.synthetic_control.SyntheticControl.__init__",
    "causalpy.experiments.synthetic_difference_in_differences.SyntheticDifferenceInDifferences.__init__",
    "causalpy.pymc_models.PyMCModel.fit",
    "causalpy.checks.outcome_falsification.OutcomeFalsification.__init__",
    "causalpy.checks.placebo_in_time.PlaceboInTime.run",
    "causalpy.pipeline.Pipeline.__init__",
    "causalpy.pipeline.Pipeline.run",
    "causalpy.steps.estimate_effect.EstimateEffect.__init__",
)

_SCOPE_SHRINK_DIAGNOSTIC = (
    "The AST survey in scripts/audit_public_signatures.py derives its scope from "
    "docs/source/api/index.md and the package __all__ exports, so the usual cause is a "
    "deleted autosummary entry in docs/source/api/index.md or a name dropped from an "
    "__all__. Restore the documentation/export entry instead of relaxing this test: "
    "every invariant in this module is parametrized over the inventory and silently "
    "stops checking anything that falls out of scope."
)


def test_public_signature_inventory_does_not_shrink() -> None:
    """A shrinking inventory silently disables the parametrized invariants."""
    assert len(_PUBLIC_SIGNATURES) >= _MINIMUM_PUBLIC_SIGNATURE_COUNT, (
        f"Public signature inventory collapsed to {len(_PUBLIC_SIGNATURES)} "
        f"declarations, below the floor of {_MINIMUM_PUBLIC_SIGNATURE_COUNT}. "
        f"{_SCOPE_SHRINK_DIAGNOSTIC}"
    )


@pytest.mark.parametrize("qualified_name", _INVENTORY_CANARIES)
def test_public_signature_inventory_covers_canary_members(qualified_name: str) -> None:
    """Named public members stay inside the audited scope."""
    inventoried = {candidate.qualified_name for candidate in _PUBLIC_SIGNATURES}
    assert qualified_name in inventoried, (
        f"{qualified_name} is no longer in the audited public-signature inventory "
        f"({len(inventoried)} declarations). {_SCOPE_SHRINK_DIAGNOSTIC}"
    )
