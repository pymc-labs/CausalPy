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
"""Posterior plot contracts retained during the plotting-stack transition."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from causalpy._arviz_compat import hdi_bound_arrays
from causalpy.plot_utils import plot_posterior_over_x


@pytest.fixture
def posterior_draws() -> xr.DataArray:
    """Return deterministic posterior draws over three observations."""
    return xr.DataArray(
        np.arange(24, dtype=float).reshape(2, 4, 3),
        dims=("chain", "draw", "obs_ind"),
        coords={"obs_ind": [0, 1, 2]},
    )


@pytest.mark.parametrize("kind", ["ribbon", "histogram", "spaghetti"])
def test_posterior_plot_kinds_render_with_default_hdi(
    posterior_draws: xr.DataArray,
    kind: Literal["ribbon", "histogram", "spaghetti"],
) -> None:
    """Every supported uncertainty kind renders at CausalPy's default 0.94 HDI."""
    figure, axis = plt.subplots()
    try:
        plot_posterior_over_x(
            posterior_draws.obs_ind.to_numpy(), posterior_draws, ax=axis, kind=kind
        )
        assert axis.has_data()
    finally:
        plt.close(figure)


def test_ribbon_bounds_match_the_094_hdi(posterior_draws: xr.DataArray) -> None:
    """The visible ribbon retains the public default 0.94 HDI probability."""
    figure, axis = plt.subplots()
    try:
        _, patch = plot_posterior_over_x(
            posterior_draws.obs_ind.to_numpy(), posterior_draws, ax=axis
        )
        assert patch is not None
        lower, upper = hdi_bound_arrays(posterior_draws, prob=0.94)
        vertices = patch.get_paths()[0].vertices
        assert vertices[:, 1].min() == pytest.approx(np.min(lower))
        assert vertices[:, 1].max() == pytest.approx(np.max(upper))
    finally:
        plt.close(figure)
