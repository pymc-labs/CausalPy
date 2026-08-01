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

The plugin keeps the ``--doctest-modules`` CI leg from running real MCMC. The
acceptance criterion is not "the leg is green" (it was green while broken) but
"a check that would fail if a doctest ever reached the real sampler". That check
has to exercise the *real* pytest option plumbing and the real ``-p`` load, so
the core proofs here run pytest in a subprocess against a throwaway probe module
rather than calling the hook with a hand-rolled fake config. Two cheap in-process
checks (an upstream-rename catcher and the no-op-outside-the-leg guarantee)
round it out.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from pymc.sampling import mcmc

from causalpy.tests import doctest_sampling

# Repo root: this file is <root>/causalpy/tests/test_doctest_sampling.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_ARG = "causalpy.tests.doctest_sampling"

# A probe whose doctest asserts the mock is installed. Under the plugin it prints
# ``mock_sample``; without it, the real ``sample``.
_MOCK_PROBE = textwrap.dedent(
    '''\
    """Throwaway probe module."""


    def probe():
        """
        >>> import pymc as pm
        >>> print(pm.sample.__name__)
        mock_sample
        """
    '''
)

# A probe whose doctest actually fits a model and calls ``pm.sample()``. Under
# the plugin the mock returns fast and the guard stays silent -- the whole point
# of the mechanism (model-fitting doctests run without real MCMC).
_MODEL_PROBE = textwrap.dedent(
    '''\
    """Throwaway probe module."""


    def probe():
        """
        >>> import pymc as pm
        >>> with pm.Model():
        ...     _ = pm.Normal("x", 0, 1)
        ...     idata = pm.sample()
        >>> "posterior" in idata
        True
        """
    '''
)

# A probe whose doctest reaches a real MCMC entry point directly. Under the
# plugin the guard raises; without it, calling the real internal with no
# arguments raises some *other* error, so the expected RuntimeError is the
# specific signal that the guard is armed.
_GUARD_PROBE = textwrap.dedent(
    '''\
    """Throwaway probe module."""


    def probe():
        """
        >>> from pymc.sampling import mcmc
        >>> mcmc._mp_sample()
        Traceback (most recent call last):
        ...
        RuntimeError: A doctest reached the real MCMC sampler: ...
        """
    '''
)


def _run_doctest(tmp_path, source, *, load_plugin):
    """Run ``pytest --doctest-modules`` on a probe module in a subprocess.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory to write the probe into and run from. Running from a tmp dir
        (no ``pyproject.toml``) keeps pytest's config clean -- no coverage gate,
        no repo ``addopts``.
    source : str
        Source of the probe module.
    load_plugin : bool
        Whether to pass ``-p causalpy.tests.doctest_sampling``.

    Returns
    -------
    subprocess.CompletedProcess
        The finished pytest invocation.
    """
    (tmp_path / "probe.py").write_text(source)
    args = [
        sys.executable,
        "-m",
        "pytest",
        "--doctest-modules",
        "-o",
        "doctest_optionflags=ELLIPSIS",
        "-p",
        "no:cacheprovider",
    ]
    if load_plugin:
        args += ["-p", _PLUGIN_ARG]
    args.append("probe.py")
    return subprocess.run(
        args, cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False
    )


def test_mock_applied_in_real_doctest_leg(tmp_path):
    """With the plugin loaded, ``pm.sample`` is the mock for the doctests.

    Exercises the real ``--doctest-modules`` detection and ``-p`` load, which the
    in-process fake-config approach cannot.
    """
    result = _run_doctest(tmp_path, _MOCK_PROBE, load_plugin=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_model_fitting_doctest_runs_under_mock(tmp_path):
    """A doctest that fits a model and calls ``pm.sample()`` passes under the mock.

    The positive complement to ``test_guard_fires...``: it proves the mock and
    the guard coexist -- real model-fitting doctests run fast without tripping
    the guard -- and would catch a future PyMC change where ``mock_sample`` began
    routing through a guarded internal.
    """
    result = _run_doctest(tmp_path, _MODEL_PROBE, load_plugin=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_plugin_is_load_bearing(tmp_path):
    """Without the plugin the mock is absent -- proving ``-p`` is load-bearing.

    This is the regression that would let the doctest leg sample for real again
    if the ``-p`` flag were dropped from the CI/Make command.
    """
    result = _run_doctest(tmp_path, _MOCK_PROBE, load_plugin=False)
    # Pin the *reason* for the failure to the doctest mismatch (real ``sample``
    # where ``mock_sample`` was expected), so an unrelated import/collection
    # error cannot make this pass for the wrong reason.
    assert result.returncode != 0, result.stdout + result.stderr
    assert "mock_sample" in result.stdout, result.stdout + result.stderr


def test_guard_fires_on_real_sampler_in_real_doctest_leg(tmp_path):
    """A doctest that reaches a real MCMC entry point fails loudly, not slowly.

    This is the acceptance-criterion check: it would fail (here, by *not*
    raising the expected RuntimeError) if the guard were not armed during an
    actual ``--doctest-modules`` run.
    """
    result = _run_doctest(tmp_path, _GUARD_PROBE, load_plugin=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_guarded_entry_points_still_exist():
    """The plugin can only guard functions that exist; a rename must be caught.

    If PyMC renames one of these internals the plugin's ``getattr`` would raise
    at configure time (fail loud), but this cheap check flags the drift directly
    without needing to run the doctest leg.
    """
    for name in doctest_sampling._FORBIDDEN_MCMC_ENTRY_POINTS:
        assert hasattr(mcmc, name), f"pymc.sampling.mcmc.{name} no longer exists"


class _FakeConfig:
    """Minimal stand-in for ``pytest.Config``; the hooks read one option."""

    def __init__(self, *, doctest_modules):
        self._doctest_modules = doctest_modules

    def getoption(self, name, default=False):
        assert name == "--doctest-modules"
        return self._doctest_modules


def _sentinel(qualified_name):
    """Build a unique, identifiable stand-in for one patched global.

    Parameters
    ----------
    qualified_name : str
        Name the sentinel stands in for, used in ``__name__`` and in the error
        raised if anything actually calls it.

    Returns
    -------
    callable
        A fresh function object, distinct from every other object in the
        process.
    """

    def _never_call(*_args, **_kwargs):
        raise AssertionError(
            f"the sentinel standing in for {qualified_name} was called"
        )

    _never_call.__name__ = f"sentinel_{qualified_name.replace('.', '_')}"
    return _never_call


def _pin_sampling_globals(monkeypatch):
    """Pin every global the plugin patches to a unique sentinel, for one test.

    The in-process hook tests assert that install *replaces* these globals and
    that restore puts back *exactly* what was there. Read against whatever the
    live values happen to be, both halves of that are order-dependent and can
    pass vacuously: ``mock_pymc_sample`` in ``causalpy/tests/conftest.py`` is
    session-scoped, so the first test file that uses it leaves ``pm.sample`` set
    to ``mock_sample`` for the rest of the session. A test that then asserted
    ``pm.sample.__name__ != "mock_sample"`` after restore would fail even though
    the plugin behaved perfectly -- it restored the mock it was correctly handed
    as the original.

    Sentinels remove the coupling. Each is a fresh object that cannot collide
    with ``mock_sample`` or with the real PyMC callables, so "install swapped
    it" and "restore returned this exact object" are both decidable no matter
    what ran earlier. ``monkeypatch`` reverts the pinning at teardown.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to install and later revert the sentinels.

    Returns
    -------
    dict
        Maps ``"pm.sample"``, ``"pm.Flat"``, ``"pm.HalfFlat"`` and each name in
        ``_FORBIDDEN_MCMC_ENTRY_POINTS`` to the sentinel now bound there.
    """
    import pymc as pm

    # The plugin's own bookkeeping is module-global too, so a sibling test that
    # fails part-way (or a mutant that breaks restore) would otherwise hand the
    # next test a half-installed plugin -- and the ``_mock_gen is not None``
    # idempotence guard would silently turn its install into a no-op, letting
    # the test pass without exercising anything. Pin it to the not-installed
    # state so each test starts from a known plugin state as well as a known
    # PyMC state.
    monkeypatch.setattr(doctest_sampling, "_mock_gen", None)
    monkeypatch.setattr(doctest_sampling, "_guard_originals", {})

    pinned = {}
    for name in ("sample", "Flat", "HalfFlat"):
        sentinel = _sentinel(f"pm.{name}")
        monkeypatch.setattr(pm, name, sentinel)
        pinned[f"pm.{name}"] = sentinel
    for name in doctest_sampling._FORBIDDEN_MCMC_ENTRY_POINTS:
        sentinel = _sentinel(f"mcmc.{name}")
        monkeypatch.setattr(mcmc, name, sentinel)
        pinned[name] = sentinel
    return pinned


def test_configure_is_noop_without_doctest_modules(monkeypatch):
    """Outside the doctest leg the plugin must not touch global sampling state."""
    import pymc as pm

    pinned = _pin_sampling_globals(monkeypatch)

    doctest_sampling.pytest_configure(_FakeConfig(doctest_modules=False))

    assert pm.sample is pinned["pm.sample"]
    assert pm.Flat is pinned["pm.Flat"]
    assert pm.HalfFlat is pinned["pm.HalfFlat"]
    for name in doctest_sampling._FORBIDDEN_MCMC_ENTRY_POINTS:
        assert getattr(mcmc, name) is pinned[name]
    assert doctest_sampling._mock_gen is None


def test_hooks_install_and_restore_inside_the_doctest_leg(monkeypatch):
    """The pytest entry points themselves arm and disarm the mock.

    ``test_install_and_restore_round_trip`` covers the private helpers; this
    covers the wiring from ``pytest_configure``/``pytest_unconfigure`` to them,
    which otherwise only runs in the subprocess tests' child process.

    The contract asserted here is a *round trip* -- restore hands back the exact
    objects install was given -- not "``pm.sample`` is unmocked afterwards". The
    latter is not the plugin's to promise: whatever ``pm.sample`` happened to be
    at install time is what restore must reinstate, mock or not. See
    ``_pin_sampling_globals`` for why that distinction decides whether this test
    is order-dependent.
    """
    import pymc as pm

    pinned = _pin_sampling_globals(monkeypatch)

    doctest_sampling.pytest_configure(_FakeConfig(doctest_modules=True))
    try:
        assert pm.sample.__name__ == "mock_sample"
        assert pm.Flat is pm.Normal
        assert pm.HalfFlat is pm.HalfNormal
    finally:
        doctest_sampling.pytest_unconfigure(_FakeConfig(doctest_modules=True))

    assert doctest_sampling._mock_gen is None
    # Restore must hand back the exact objects install was given, not merely
    # "something that isn't the mock".
    assert pm.sample is pinned["pm.sample"]
    assert pm.Flat is pinned["pm.Flat"]
    assert pm.HalfFlat is pinned["pm.HalfFlat"]
    for name in doctest_sampling._FORBIDDEN_MCMC_ENTRY_POINTS:
        assert getattr(mcmc, name) is pinned[name]


def test_install_is_idempotent(monkeypatch):
    """A second install must not capture the mocked state as its "original".

    If it did, the single matching restore would leave the mock in place for
    every later test in the process.
    """
    import pymc as pm

    pinned = _pin_sampling_globals(monkeypatch)

    doctest_sampling._install_doctest_mock()
    try:
        first_gen = doctest_sampling._mock_gen
        doctest_sampling._install_doctest_mock()
        assert doctest_sampling._mock_gen is first_gen
    finally:
        doctest_sampling._restore_doctest_mock()

    # The single restore matching the two installs must leave nothing mocked.
    assert pm.sample is pinned["pm.sample"]
    for name in doctest_sampling._FORBIDDEN_MCMC_ENTRY_POINTS:
        assert getattr(mcmc, name) is pinned[name]


def test_restore_without_install_is_a_noop(monkeypatch):
    """``pytest_unconfigure`` also fires when configure never installed.

    The not-installed precondition is pinned rather than inherited from
    whatever ran before, so this cannot fail as collateral from an unrelated
    test leaving the plugin half-installed.
    """
    pinned = _pin_sampling_globals(monkeypatch)

    doctest_sampling._restore_doctest_mock()

    assert doctest_sampling._mock_gen is None
    # A restore with nothing to undo must not touch the globals either.
    import pymc as pm

    assert pm.sample is pinned["pm.sample"]
    for name in doctest_sampling._FORBIDDEN_MCMC_ENTRY_POINTS:
        assert getattr(mcmc, name) is pinned[name]


def test_install_and_restore_round_trip(monkeypatch):
    """In-process check that install arms mock+guard and restore undoes both.

    The subprocess tests prove the mechanism through real pytest, but that logic
    never runs in this (coverage-measured) process. This exercises
    ``_install_doctest_mock``/``_restore_doctest_mock`` directly, and is the only
    place teardown is asserted in-process. ``monkeypatch`` snapshots the globals
    so state is restored even if an assertion fails mid-way.
    """
    import pymc as pm

    pinned = _pin_sampling_globals(monkeypatch)

    doctest_sampling._install_doctest_mock()
    try:
        assert pm.sample.__name__ == "mock_sample"
        assert pm.Flat is pm.Normal
        assert pm.HalfFlat is pm.HalfNormal
        for name in doctest_sampling._FORBIDDEN_MCMC_ENTRY_POINTS:
            with pytest.raises(RuntimeError, match="real MCMC sampler"):
                getattr(mcmc, name)()
    finally:
        doctest_sampling._restore_doctest_mock()

    # Teardown restored every patched name to the exact original object.
    assert pm.sample is pinned["pm.sample"]
    assert pm.Flat is pinned["pm.Flat"]
    assert pm.HalfFlat is pinned["pm.HalfFlat"]
    for name in doctest_sampling._FORBIDDEN_MCMC_ENTRY_POINTS:
        assert getattr(mcmc, name) is pinned[name]
    assert doctest_sampling._mock_gen is None


@pytest.mark.parametrize("relpath", [".github/workflows/ci.yml", "Makefile"])
def test_doctest_command_loads_the_plugin(relpath):
    """The load-bearing ``-p`` flag must stay on the doctest command at source.

    ``test_plugin_is_load_bearing`` proves the flag matters; this proves the two
    real invocations still carry it, so a hand-edit cannot silently remove it.

    Assumes the doctest command stays on a single line (true for both files); a
    future reformat splitting the flags across lines would need this updated.
    """
    text = (_REPO_ROOT / relpath).read_text()
    doctest_lines = [ln for ln in text.splitlines() if "--doctest-modules" in ln]
    assert doctest_lines, f"no --doctest-modules command found in {relpath}"
    for line in doctest_lines:
        assert f"-p {_PLUGIN_ARG}" in line, (
            f"doctest command in {relpath} does not load the plugin: {line!r}"
        )
