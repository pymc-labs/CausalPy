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
"""Tests for the doctest sampling plugin (``causalpy/tests/doctest_sampling.py``).

These guard the mechanism that keeps the ``--doctest-modules`` CI leg from
running real MCMC: that the plugin (a) swaps ``pm.sample`` for the fast mock and
(b) booby-traps the real MCMC entry points so a regression fails loudly instead
of sampling for real. The guard is only useful if it targets functions that
still exist and actually raises, so both are asserted here.
"""

import pymc as pm
import pytest
from pymc.sampling import mcmc
from pymc.testing import mock_sample

from causalpy.tests import doctest_sampling

# The real MCMC entry points the plugin forbids during the doctest leg. Kept in
# one place so the "these names still exist" check and the "these actually
# raise" check cannot drift apart.
GUARDED_ENTRY_POINTS = (
    "_sample_many",
    "_mp_sample",
    "_sample_population",
    "_sample_external_nuts",
)


class _FakeConfig:
    """Minimal stand-in for ``pytest.Config`` exposing only ``getoption``."""

    def __init__(self, doctest_modules: bool):
        self._doctest_modules = doctest_modules

    def getoption(self, name, default=False):
        if name == "--doctest-modules":
            return self._doctest_modules
        return default


def test_guarded_entry_points_still_exist():
    """The plugin can only guard functions that exist; a rename must be caught.

    If PyMC renames one of these internals the plugin would silently guard
    nothing and doctests would sample for real again, so pin the names down.
    """
    for name in GUARDED_ENTRY_POINTS:
        assert hasattr(mcmc, name), f"pymc.sampling.mcmc.{name} no longer exists"


def test_plugin_is_noop_without_doctest_modules(monkeypatch):
    """Outside the doctest leg the plugin must not touch global sampling state."""
    original_sample = pm.sample
    monkeypatch.setattr(pm, "sample", original_sample)

    doctest_sampling.pytest_configure(_FakeConfig(doctest_modules=False))

    assert pm.sample is original_sample


def test_plugin_installs_mock_and_arms_guard(monkeypatch):
    """Under ``--doctest-modules`` the mock is installed and the guard is armed."""
    # Register the originals with monkeypatch so its teardown restores the
    # process-wide state the plugin mutates, keeping the rest of the suite clean.
    monkeypatch.setattr(pm, "sample", pm.sample)
    for name in GUARDED_ENTRY_POINTS:
        monkeypatch.setattr(mcmc, name, getattr(mcmc, name))

    doctest_sampling.pytest_configure(_FakeConfig(doctest_modules=True))

    # (a) The fix: pm.sample is the fast prior-predictive stand-in.
    assert pm.sample is mock_sample

    # (b) The proof: every real MCMC entry point now refuses to run, so a
    #     doctest that reaches the real sampler fails immediately.
    for name in GUARDED_ENTRY_POINTS:
        with pytest.raises(RuntimeError, match="real MCMC sampler"):
            getattr(mcmc, name)()
