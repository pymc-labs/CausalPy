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
"""Tests for causalpy.experiments._design helpers."""

import numpy as np
import pandas as pd
import pytest

from causalpy.experiments._design import PatsyDesignTransform


def test_patsy_design_transform_y_requires_outcome_metadata():
    """transform_y raises when the transform was built without outcome metadata."""
    transform = PatsyDesignTransform(
        _x_design_info=None,
        _y_design_info=None,
        labels=["Intercept"],
        outcome_name=None,
    )
    with pytest.raises(ValueError, match="transform_y is unavailable"):
        transform.transform_y(pd.DataFrame({"y": [1.0]}))


def test_patsy_design_transform_y_rebuilds_outcome():
    """transform_y rebuilds the outcome vector from stored patsy metadata."""
    data = pd.DataFrame({"y": [1.0, 2.0], "x": [0.0, 1.0]})
    from causalpy.experiments._design import build_patsy_design

    transform, _, y = build_patsy_design("y ~ x", data)
    rebuilt = transform.transform_y(data)
    np.testing.assert_allclose(rebuilt, y)
