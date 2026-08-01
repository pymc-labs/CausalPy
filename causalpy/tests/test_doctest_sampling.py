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


def test_plugin_is_load_bearing(tmp_path):
    """Without the plugin the mock is absent -- proving ``-p`` is load-bearing.

    This is the regression that would let the doctest leg sample for real again
    if the ``-p`` flag were dropped from the CI/Make command.
    """
    result = _run_doctest(tmp_path, _MOCK_PROBE, load_plugin=False)
    assert result.returncode != 0, result.stdout + result.stderr


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


@pytest.mark.parametrize("relpath", [".github/workflows/ci.yml", "Makefile"])
def test_doctest_command_loads_the_plugin(relpath):
    """The load-bearing ``-p`` flag must stay on the doctest command at source.

    ``test_plugin_is_load_bearing`` proves the flag matters; this proves the two
    real invocations still carry it, so a hand-edit cannot silently remove it.
    """
    text = (_REPO_ROOT / relpath).read_text()
    doctest_lines = [ln for ln in text.splitlines() if "--doctest-modules" in ln]
    assert doctest_lines, f"no --doctest-modules command found in {relpath}"
    for line in doctest_lines:
        assert f"-p {_PLUGIN_ARG}" in line, (
            f"doctest command in {relpath} does not load the plugin: {line!r}"
        )
