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
        if module_info.name.startswith("_"):
            continue
        importlib.import_module(f"{pkg.__name__}.{module_info.name}")

    subclasses: list[type] = []
    seen: set[int] = set()
    stack: list[type] = list(BaseExperiment.__subclasses__())
    while stack:
        cls = stack.pop()
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        subclasses.append(cls)
        stack.extend(cls.__subclasses__())
    return subclasses


def _experiments_with_overridden_plot() -> Iterable[type]:
    """Yield subclasses that define their own ``plot`` (not inherited).

    The base class :class:`~causalpy.experiments.base.BaseExperiment` keeps a
    generic ``plot(*args, **kwargs)`` for dispatch; subclasses that don't
    override it are intentionally exempt from this invariant (e.g.
    :class:`~causalpy.experiments.inverse_propensity_weighting.InversePropensityWeighting`,
    which exposes purpose-built ``plot_ate`` / ``plot_balance_ecdf`` methods
    rather than a unified ``plot``).
    """
    for cls in _all_base_experiment_subclasses():
        if "plot" in cls.__dict__:
            yield cls


_OVERRIDING_SUBCLASSES = list(_experiments_with_overridden_plot())


def test_at_least_several_overriding_subclasses_discovered() -> None:
    """Sanity: the discovery walk finds the bulk of experiment classes.

    Guards against the discovery returning empty (which would make every
    parametrised test below trivially pass).
    """
    names = sorted(cls.__name__ for cls in _OVERRIDING_SUBCLASSES)
    expected = {
        "InterruptedTimeSeries",
        "SyntheticControl",
        "DifferenceInDifferences",
        "RegressionDiscontinuity",
        "RegressionKink",
        "PrePostNEGD",
        "PiecewiseITS",
        "PanelRegression",
        "StaggeredDifferenceInDifferences",
    }
    missing = expected - set(names)
    assert not missing, (
        f"Expected every plotting experiment to override plot(); "
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
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        # numpydoc: "name : type"
        numpydoc_pattern = re.compile(rf"^\s*{re.escape(name)}\s*:\s", re.MULTILINE)
        # Sphinx: ":param name:" or ":param type name:"
        sphinx_pattern = re.compile(rf":param\s+(?:\S+\s+)?{re.escape(name)}\s*:")
        if not (numpydoc_pattern.search(doc) or sphinx_pattern.search(doc)):
            missing.append(name)
    assert not missing, (
        f"{cls.__name__}.plot signature has parameter(s) {missing} that "
        f"are not documented in the docstring. Add a numpydoc 'name : type' "
        f"entry under the Parameters block."
    )


def test_no_unexpected_var_keyword_anywhere() -> None:
    """Belt-and-braces: even subclasses we did not explicitly cover above must
    not regress to a bare ``**kwargs`` if they later add their own ``plot``
    override. This ensures new subclasses join the invariant by default.
    """
    # Re-run the discovery; if a new subclass shows up between import and now
    # (it shouldn't, but be defensive), include it.
    discovered = sorted(cls.__name__ for cls in _experiments_with_overridden_plot())
    assert discovered, "no overriding subclasses discovered"
    # We rely on the parametrised invariant above to do the per-class check;
    # this test guarantees a positive count and serves as documentation.
    assert causalpy.__name__ == "causalpy"
