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


def test_configure_is_noop_without_doctest_modules(monkeypatch):
    """Outside the doctest leg the plugin must not touch global sampling state."""
    import pymc as pm

    original_sample = pm.sample
    monkeypatch.setattr(pm, "sample", original_sample)

    class _FakeConfig:
        def getoption(self, name, default=False):
            return default  # --doctest-modules not set

    doctest_sampling.pytest_configure(_FakeConfig())

    assert pm.sample is original_sample


def test_install_and_restore_round_trip(monkeypatch):
    """In-process check that install arms mock+guard and restore undoes both.

    The subprocess tests prove the mechanism through real pytest, but that logic
    never runs in this (coverage-measured) process. This exercises
    ``_install_doctest_mock``/``_restore_doctest_mock`` directly, and is the only
    place teardown is asserted in-process. ``monkeypatch`` snapshots the globals
    so state is restored even if an assertion fails mid-way.
    """
    import pymc as pm

    monkeypatch.setattr(pm, "sample", pm.sample)
    monkeypatch.setattr(pm, "Flat", pm.Flat)
    monkeypatch.setattr(pm, "HalfFlat", pm.HalfFlat)
    for name in doctest_sampling._FORBIDDEN_MCMC_ENTRY_POINTS:
        monkeypatch.setattr(mcmc, name, getattr(mcmc, name))
    originals = {
        n: getattr(mcmc, n) for n in doctest_sampling._FORBIDDEN_MCMC_ENTRY_POINTS
    }

    doctest_sampling._install_doctest_mock()
    try:
        assert pm.sample.__name__ == "mock_sample"
        for name in doctest_sampling._FORBIDDEN_MCMC_ENTRY_POINTS:
            with pytest.raises(RuntimeError, match="real MCMC sampler"):
                getattr(mcmc, name)()
    finally:
        doctest_sampling._restore_doctest_mock()

    # Teardown restored every patched name to the exact original object.
    for name, original in originals.items():
        assert getattr(mcmc, name) is original
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
