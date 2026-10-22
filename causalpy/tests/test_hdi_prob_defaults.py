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
Invariant tests pinning the project-wide ``HDI_PROB`` default across all
public methods that accept an ``hdi_prob`` keyword argument.

If a future change introduces a method that takes ``hdi_prob`` with a default
that diverges from :data:`causalpy.constants.HDI_PROB`, these tests will fail
and prompt the author to either align the default with the constant or
deliberately update this invariant.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import Any

import pytest

from causalpy.constants import HDI_PROB


def _hdi_prob_defaulted_methods() -> Iterable[tuple[str, Callable[..., Any]]]:
    """Yield ``(qualified_name, callable)`` for every public function/method
    in ``causalpy`` whose signature has an ``hdi_prob`` parameter with a
    non-empty default value.

    Excluded:

      * Private callables (any path component starting with ``_``).
      * The ``causalpy.tests`` package.
      * ``hdi_prob`` parameters with no default (kwarg-only required arg).
      * ``hdi_prob`` parameters defaulted to ``None`` (sentinel for "no
        override" semantics, e.g.
        :meth:`BaseExperiment.set_maketables_options`).
    """
    import importlib
    import pkgutil

    import causalpy

    seen: set[int] = set()

    # Defensive branches below (private modules, import failures, callables
    # without a usable signature, etc.) are hit during the walk depending on
    # which optional dependencies and modules happen to be importable; they
    # are exempt from coverage so the safety net does not become a CI
    # liability.
    for module_info in pkgutil.walk_packages(
        causalpy.__path__, prefix=f"{causalpy.__name__}."
    ):
        name = module_info.name
        if name.startswith("causalpy.tests"):
            continue
        if any(part.startswith("_") for part in name.split(".")[1:]):
            continue  # pragma: no cover
        try:
            mod = importlib.import_module(name)
        except Exception:  # pragma: no cover
            continue

        for attr_name, attr in vars(mod).items():
            if attr_name.startswith("_"):
                continue
            if not callable(attr):
                continue
            if id(attr) in seen:
                continue
            seen.add(id(attr))

            attr_module = getattr(attr, "__module__", None)
            if not isinstance(attr_module, str) or not attr_module.startswith(
                "causalpy"
            ):
                continue
            attr_qualname = getattr(attr, "__qualname__", None)
            if not isinstance(attr_qualname, str):
                continue  # pragma: no cover

            candidates: list[tuple[str, Callable[..., Any]]] = []
            if inspect.isclass(attr):
                for member_name, member in vars(attr).items():
                    if member_name.startswith("_"):
                        continue
                    if not callable(member):
                        continue
                    candidates.append(
                        (f"{attr_module}.{attr_qualname}.{member_name}", member)
                    )
            else:
                candidates.append((f"{attr_module}.{attr_qualname}", attr))

            for qualname, fn in candidates:
                try:
                    sig = inspect.signature(fn)
                except (TypeError, ValueError):  # pragma: no cover
                    continue
                param = sig.parameters.get("hdi_prob")
                if param is None:
                    continue
                if param.default is inspect.Parameter.empty:
                    continue  # pragma: no cover
                if param.default is None:
                    continue
                yield qualname, fn


def test_at_least_one_hdi_prob_defaulted_method_discovered() -> None:
    """Sanity: the discovery helper finds at least a few real methods.

    This guards against the pkgutil walk silently returning nothing (which
    would make the parametrised invariant trivially pass).
    """
    methods = list(_hdi_prob_defaulted_methods())
    assert len(methods) >= 3, (
        f"Expected to find several public methods with an hdi_prob default; "
        f"got {len(methods)}: {[name for name, _ in methods]}"
    )


@pytest.mark.parametrize(
    ("qualname", "fn"),
    list(_hdi_prob_defaulted_methods()),
    ids=lambda v: v if isinstance(v, str) else "fn",
)
def test_hdi_prob_default_matches_constant(
    qualname: str, fn: Callable[..., Any]
) -> None:
    """Every public callable with an ``hdi_prob`` default must use ``HDI_PROB``.

    See :issue:`889`. If you intentionally need a different default, update
    this invariant in the same PR and document the divergence.
    """
    sig = inspect.signature(fn)
    default = sig.parameters["hdi_prob"].default
    assert default == HDI_PROB, (
        f"{qualname} has hdi_prob default {default!r}; "
        f"expected HDI_PROB ({HDI_PROB!r}). Either align the default with "
        f"HDI_PROB or deliberately update test_hdi_prob_defaults.py."
    )
