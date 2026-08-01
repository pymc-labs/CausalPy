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
When ``--doctest-modules`` is active it replaces :func:`pymc.sample` with
:func:`pymc.testing.mock_sample` (fast prior-predictive stand-in), and it
booby-traps the real MCMC entry points so that if a doctest *ever* reaches the
real sampler -- i.e. the mock stops applying for any reason -- the run fails
loudly and immediately instead of sampling for real and wedging the job.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Install the doctest sampling mock and the real-sampler guard.

    Parameters
    ----------
    config : pytest.Config
        The active pytest configuration; used only to detect
        ``--doctest-modules``.
    """
    if not config.getoption("--doctest-modules", default=False):
        return

    import pymc as pm
    from pymc.sampling import mcmc
    from pymc.testing import mock_sample

    # The fix: doctests that call ``pm.sample`` get the fast prior-predictive
    # stand-in instead of real MCMC.
    pm.sample = mock_sample

    # The proof: ``mock_sample`` routes through ``pm.sample_prior_predictive``
    # (in ``pymc.sampling.forward``) and never touches these MCMC entry points.
    # Real sampling always does. Patching them to raise turns "a doctest reached
    # the real sampler" into an immediate, unambiguous failure rather than a
    # six-hour hang.
    def _forbidden(*_args, **_kwargs):
        raise RuntimeError(
            "A doctest reached the real MCMC sampler: the doctest sampling mock "
            "is not in force. See causalpy/tests/doctest_sampling.py."
        )

    for _name in (
        "_sample_many",
        "_mp_sample",
        "_sample_population",
        "_sample_external_nuts",
    ):
        setattr(mcmc, _name, _forbidden)
