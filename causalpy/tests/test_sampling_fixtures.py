#   Copyright 2026 - 2026 The PyMC Labs Developers
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
"""Sampler fixtures restore PyMC globals when their requesting module finishes."""

import os
import subprocess
import sys
import textwrap

import pytest

_PROBES = {
    "mock": """
        import pymc as pm
        import pytest
        from pymc.testing import mock_sample

        @pytest.fixture(scope="module")
        def mocked_sampler(mock_pymc_sample):
            return pm.sample

        def test_mock(mocked_sampler):
            assert mocked_sampler is mock_sample
            assert pm.Flat is pm.Normal
            assert pm.HalfFlat is pm.HalfNormal
    """,
    "real": """
        import pymc as pm
        from pymc.distributions.continuous import Flat, HalfFlat
        from pymc.sampling.mcmc import sample

        def test_real(real_pymc_sampling):
            assert pm.sample is sample
            assert pm.Flat is Flat
            assert pm.HalfFlat is HalfFlat
    """,
    "genuine": """
        import pymc as pm
        from pymc.distributions.continuous import Flat, HalfFlat
        from pymc.sampling.mcmc import sample

        def test_genuine(genuine_pymc_sampling):
            assert pm.sample is sample
            assert pm.Flat is Flat
            assert pm.HalfFlat is HalfFlat
    """,
    "unrequested": """
        import pymc as pm
        from pymc.distributions.continuous import Flat, HalfFlat
        from pymc.sampling.mcmc import sample

        def test_unrequested():
            assert pm.sample is sample
            assert pm.Flat is Flat
            assert pm.HalfFlat is HalfFlat
    """,
}


def _run_fixture_probes(tmp_path, sources):
    """Run probe modules with the actual shared sampling fixtures."""
    (tmp_path / "conftest.py").write_text(
        'pytest_plugins = ["causalpy.tests.conftest"]\n', encoding="utf-8"
    )
    modules = []
    for index, source in enumerate(sources):
        module = tmp_path / f"test_{index}_sampling.py"
        module.write_text(textwrap.dedent(source), encoding="utf-8")
        modules.append(str(module))

    # Probe order is part of the assertion, so do not inherit outer test filters or parallel scheduling.
    probe_env = os.environ.copy()
    probe_env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", *modules],
        cwd=tmp_path,
        env=probe_env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize("fixture_order", [("mock", "real"), ("real", "mock")])
def test_sampler_fixtures_restore_globals_between_modules(tmp_path, fixture_order):
    """A later module gets genuine sampling without requesting an undo fixture."""
    pytest.importorskip("pymc.testing")
    result = _run_fixture_probes(
        tmp_path,
        [
            _PROBES[name]
            for name in (
                fixture_order[0],
                "unrequested",
                fixture_order[1],
                "unrequested",
            )
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 passed" in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize("fixture_order", [("mock", "genuine"), ("genuine", "mock")])
def test_genuine_sampling_restores_mock_within_module(tmp_path, fixture_order):
    """A genuine fit can run among mocked tests without changing their priors."""
    pytest.importorskip("pymc.testing")
    mixed_source = "\n".join(textwrap.dedent(_PROBES[name]) for name in fixture_order)
    mixed_source += textwrap.dedent(
        """
        def test_mock_after(mocked_sampler):
            assert pm.sample is mocked_sampler is mock_sample
            assert pm.Flat is pm.Normal
            assert pm.HalfFlat is pm.HalfNormal
        """
    )
    result = _run_fixture_probes(tmp_path, [mixed_source, _PROBES["unrequested"]])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 passed" in result.stdout, result.stdout + result.stderr
