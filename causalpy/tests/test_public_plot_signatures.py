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
"""
Signature-drift invariants for the public ``plot()`` method on every
:class:`~causalpy.experiments.base.BaseExperiment` subclass.

These tests pin the rollout of explicit ``plot()`` signatures tracked under
GitHub issue `pymc-labs/CausalPy#886`_. They guarantee that:

1. Every experiment subclass that overrides the public ``plot()`` exposes
   only explicit, named parameters at the public surface (no ``*args`` and
   no ``**kwargs``), so that Sphinx, IDE autocomplete, ``inspect.signature``,
   and ``help()`` all show the real, supported knobs.
2. Every parameter in the public ``plot()`` signature is documented in the
   method docstring's ``Parameters`` block (or, for stub overrides, at
   least appears in the body of the docstring).

If you legitimately need to relax the signature contract (e.g. by adding a
new ``**kwargs`` to a public ``plot()``), update the umbrella issue and this
test together; do not silently weaken the invariant.

.. _pymc-labs/CausalPy#886: https://github.com/pymc-labs/CausalPy/issues/886
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from collections.abc import Iterable

import pytest

import causalpy
from causalpy.experiments.base import BaseExperiment


def _all_base_experiment_subclasses() -> list[type]:
    """Import every ``causalpy.experiments`` submodule and collect concrete
    :class:`~causalpy.experiments.base.BaseExperiment` subclasses.

    Importing the submodules ensures the subclasses are registered before we
    walk ``__subclasses__()``.
    """
    pkg = importlib.import_module("causalpy.experiments")
    for module_info in pkgutil.iter_modules(pkg.__path__):
        if module_info.name.startswith("_"):  # pragma: no cover
            continue
        importlib.import_module(f"{pkg.__name__}.{module_info.name}")

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
    """Yield concrete subclasses that declare their own ``plot``.

    :class:`~causalpy.experiments.base.BaseExperiment` deliberately does
    **not** define a public ``plot`` method — the shared dispatcher lives
    in the protected helper ``_render_plot`` so that every subclass is
    forced to declare its own explicit, documented ``plot`` (issue
    `#886 <https://github.com/pymc-labs/CausalPy/issues/886>`_). Every
    concrete subclass therefore should appear here, even those whose
    ``plot`` is a stub raising :class:`NotImplementedError` (the stub
    itself must still have an explicit signature and matching
    docstring).
    """
    for cls in _all_base_experiment_subclasses():
        if "plot" in cls.__dict__:  # pragma: no branch
            yield cls


_OVERRIDING_SUBCLASSES = list(_experiments_with_plot())


def test_every_concrete_subclass_declares_plot() -> None:
    """Every concrete experiment subclass declares its own ``plot``.

    Because :class:`~causalpy.experiments.base.BaseExperiment` no longer
    provides a public ``plot``, a subclass that forgets to declare one
    will raise :class:`AttributeError` on ``result.plot()``. This test
    guards against that regression by asserting every known concrete
    subclass shows up in the discovery walk.
    """
    names = sorted(cls.__name__ for cls in _OVERRIDING_SUBCLASSES)
    expected = {
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
    }
    missing = expected - set(names)
    assert not missing, (
        f"Expected every concrete experiment to declare its own plot(); "
        f"missing: {sorted(missing)}; discovered: {names}"
    )


@pytest.mark.parametrize(
    "cls",
    _OVERRIDING_SUBCLASSES,
    ids=lambda c: c.__name__,
)
def test_public_plot_has_no_var_positional_or_var_keyword(cls: type) -> None:
    """``plot`` must not declare ``*args`` or ``**kwargs`` at the public surface.

    The whole point of issue #886 is that a bare ``**kwargs`` silently
    swallows real, supported parameters. Mirror PyMC, scikit-learn, ArviZ,
    and statsmodels: enumerate the parameters explicitly.
    """
    plot_method = cls.__dict__["plot"]
    sig = inspect.signature(plot_method)
    bad = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    assert not bad, (
        f"{cls.__name__}.plot has VAR_POSITIONAL/VAR_KEYWORD parameter(s) "
        f"{[p.name for p in bad]}; replace with explicit named parameters "
        f"per issue #886."
    )


@pytest.mark.parametrize(
    "cls",
    _OVERRIDING_SUBCLASSES,
    ids=lambda c: c.__name__,
)
def test_public_plot_parameters_are_documented(cls: type) -> None:
    """Every named parameter on ``plot`` must appear in the method docstring.

    We accept either a numpydoc ``param : type`` line or a Sphinx-style
    ``:param name:`` line for the parameter name. The looser check avoids
    coupling this invariant to a single docstring style while still catching
    drift between the signature and the documentation.
    """
    plot_method = cls.__dict__["plot"]
    sig = inspect.signature(plot_method)
    doc = plot_method.__doc__ or ""
    missing: list[str] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (  # pragma: no cover
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        # numpydoc: "name : type"
        numpydoc_pattern = re.compile(rf"^\s*{re.escape(name)}\s*:\s", re.MULTILINE)
        # Sphinx: ":param name:" or ":param type name:"
        sphinx_pattern = re.compile(rf":param\s+(?:\S+\s+)?{re.escape(name)}\s*:")
        if not (
            numpydoc_pattern.search(doc) or sphinx_pattern.search(doc)
        ):  # pragma: no cover
            missing.append(name)
    assert not missing, (
        f"{cls.__name__}.plot signature has parameter(s) {missing} that "
        f"are not documented in the docstring. Add a numpydoc 'name : type' "
        f"entry under the Parameters block."
    )


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("causalpy.experiments.instrumental_variable", "InstrumentalVariable"),
        (
            "causalpy.experiments.inverse_propensity_weighting",
            "InversePropensityWeighting",
        ),
    ],
)
def test_stub_plot_raises_not_implemented(module_name: str, class_name: str) -> None:
    """Stub ``plot()`` overrides must raise :class:`NotImplementedError`.

    :class:`~causalpy.experiments.instrumental_variable.InstrumentalVariable`
    and :class:`~causalpy.experiments.inverse_propensity_weighting.InversePropensityWeighting`
    declare explicit kwarg-only ``plot()`` signatures purely to satisfy the
    structural invariant from issue #886; both bodies are stubs that must
    surface that fact loudly to callers rather than failing silently or
    producing an empty figure. We bypass ``__init__`` with
    :func:`object.__new__` because the stub bodies do not depend on instance
    state and constructing real experiments would slow this invariant check
    down considerably without adding coverage.
    """
    cls = getattr(importlib.import_module(module_name), class_name)
    instance = object.__new__(cls)
    with pytest.raises(NotImplementedError):
        instance.plot()


def test_base_experiment_has_no_public_plot() -> None:
    """The base class deliberately offers no public ``plot``.

    The shared dispatcher lives in the protected helper ``_render_plot``
    so that subclasses can't passively inherit a generic ``*args,
    **kwargs`` signature and re-introduce the discoverability problem
    described in #886. If this test starts failing, somebody has added a
    public ``plot`` back to :class:`BaseExperiment`; either remove it or
    update the design contract documented in ``AGENTS.md``.
    """
    assert "plot" not in BaseExperiment.__dict__, (
        "BaseExperiment must not declare a public plot(); the shared "
        "dispatcher is _render_plot. See AGENTS.md and issue #886."
    )
    assert hasattr(BaseExperiment, "_render_plot"), (
        "BaseExperiment should define the protected _render_plot helper "
        "that subclass plot() methods delegate to."
    )
    # Sanity: the discovery walk must find at least one subclass; otherwise
    # the parametrised invariants above are trivially passing.
    assert _OVERRIDING_SUBCLASSES, "no concrete subclasses discovered"
    assert causalpy.__name__ == "causalpy"
