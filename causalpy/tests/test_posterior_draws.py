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
"""Tests for uncertainty detection in prediction containers."""

import numpy as np
import pytest
import xarray as xr

from causalpy.utils import has_posterior_draws


@pytest.mark.parametrize(
    ("dims", "shape", "expected"),
    [
        (("obs_ind", "treated_units"), (5, 3), False),
        (("chain", "draw", "obs_ind", "treated_units"), (1, 1, 5, 3), False),
        (("chain", "draw", "obs_ind", "treated_units"), (1, 4, 5, 3), True),
        (("chain", "draw", "obs_ind", "treated_units"), (4, 1, 5, 3), True),
        (("obs_ind", "treated_units", "draw", "chain"), (5, 3, 2, 2), True),
        (("draw", "obs_ind"), (4, 5), True),
        (("chain", "obs_ind"), (4, 5), True),
        (("draw", "obs_ind"), (1, 5), False),
        (("chain", "obs_ind"), (1, 5), False),
    ],
    ids=[
        "no-sampling-dimensions",
        "point-estimate",
        "multiple-draws",
        "multiple-chains",
        "reordered-dimensions",
        "draw-only",
        "chain-only",
        "singleton-draw-only",
        "singleton-chain-only",
    ],
)
def test_has_posterior_draws_uses_sampling_dimensions(dims, shape, expected):
    """Only sampling dimensions determine whether uncertainty is available."""
    predictions = xr.DataArray(np.zeros(shape), dims=dims)

    assert has_posterior_draws(predictions) is expected
