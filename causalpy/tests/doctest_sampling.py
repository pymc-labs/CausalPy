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
"""Pytest plugin that neutralises real MCMC sampling in the doctest leg.

Loaded explicitly from the doctest CI/Make command via
``-p causalpy.tests.doctest_sampling``.

Why a plugin and not a conftest fixture
---------------------------------------
An autouse fixture defined in ``causalpy/tests/conftest.py`` only applies to
items collected *under that conftest's own directory*. The doctests are
collected from ``causalpy/*.py`` -- outside ``causalpy/tests/`` -- so such a
fixture can never reach them, and every model-fitting doctest sampled for real
(multi-core by default). A plugin loaded with ``-p`` registers its hooks
globally, independent of the collection tree, so the mock actually applies to
the doctests.

Why here and not a package-root ``causalpy/conftest.py``
--------------------------------------------------------
This module lives under ``causalpy/tests/``, which is excluded from the built
wheel by ``[tool.setuptools.packages.find] exclude = ["causalpy.test*"]``. No
test-only sampling shim ships to users. Excluding a single ``conftest.py``
*module* from a setuptools wheel, by contrast, is not cleanly expressible
(``packages.find`` excludes packages, and ``MANIFEST.in`` only shapes the
sdist).

What it does
------------
When ``--doctest-modules`` is active it drives PyMC's own
:func:`pymc.testing.mock_sample_setup_and_teardown` -- the *same* setup/teardown
that backs the normal suite's ``mock_pymc_sample`` fixture, so the mock is
defined in exactly one place. That swaps :func:`pymc.sample` for a fast
prior-predictive stand-in and swaps ``Flat``/``HalfFlat`` for proper priors that
prior-predictive can actually draw. On top of that it booby-traps the real MCMC
entry points so that if a doctest *ever* reaches the real sampler -- i.e. the
mock stops applying for any reason -- the run fails loudly and immediately
instead of sampling for real and wedging the job.
"""

import pytest

#: Real MCMC entry points in ``pymc.sampling.mcmc`` that ``pymc.sample`` dispatches
#: to. ``mock_sample`` routes through ``sample_prior_predictive`` (in
#: ``pymc.sampling.forward``) and never calls any of these, so guarding them
#: cannot false-fire on the mock.
_FORBIDDEN_MCMC_ENTRY_POINTS = (
    "_sample_many",
    "_mp_sample",
    "_sample_population",
    "_sample_external_nuts",
)

# Session state stashed between configure and unconfigure. Restored on teardown
# so the process-global patches do not leak, even though in practice the doctest
# leg runs in a throwaway process of its own.
_mock_gen = None
_guard_originals: dict = {}


def pytest_configure(config: pytest.Config) -> None:
    """Install the doctest sampling mock when running ``--doctest-modules``.

    Parameters
    ----------
    config : pytest.Config
        The active pytest configuration; used only to detect
        ``--doctest-modules``.
    """
    if config.getoption("--doctest-modules", default=False):
        _install_doctest_mock()


def pytest_unconfigure(config: pytest.Config) -> None:
    """Undo whatever :func:`pytest_configure` installed.

    Parameters
    ----------
    config : pytest.Config
        The active pytest configuration (unused; required by the hook
        signature).
    """
    _restore_doctest_mock()


def _install_doctest_mock() -> None:
    """Swap in the mock sampler and arm the real-sampler guard, process-wide."""
    global _mock_gen
    # Keep configure/unconfigure symmetric: a second install would capture the
    # already-mocked state as its "original" and a single restore would then
    # leak the mock. In practice a ``-p`` plugin configures once, but guard
    # against double registration.
    if _mock_gen is not None:
        return

    from pymc.sampling import mcmc
    from pymc.testing import mock_sample_setup_and_teardown

    # The fix. A bare import (no try/except) is deliberate: if PyMC's mock helper
    # is unavailable we must fail loudly here, never silently fall through to
    # real sampling.
    _mock_gen = mock_sample_setup_and_teardown()
    next(_mock_gen)  # run setup: swap pm.sample, pm.Flat, pm.HalfFlat

    # The proof. ``getattr`` first so an upstream rename fails loudly *here*
    # (AttributeError at configure time) rather than ``setattr`` silently
    # attaching a dead attribute and leaving the real sampler unguarded.
    def _forbidden(*_args, **_kwargs):
        raise RuntimeError(
            "A doctest reached the real MCMC sampler: the doctest sampling mock "
            "is not in force. See causalpy/tests/doctest_sampling.py."
        )

    for name in _FORBIDDEN_MCMC_ENTRY_POINTS:
        _guard_originals[name] = getattr(mcmc, name)
        setattr(mcmc, name, _forbidden)


def _restore_doctest_mock() -> None:
    """Restore ``pm.sample``/``Flat``/``HalfFlat`` and the guarded entry points."""
    global _mock_gen
    if _mock_gen is None:
        return

    from pymc.sampling import mcmc

    next(_mock_gen, None)  # run teardown: restore pm.sample, pm.Flat, pm.HalfFlat
    _mock_gen = None
    while _guard_originals:
        name, original = _guard_originals.popitem()
        setattr(mcmc, name, original)
