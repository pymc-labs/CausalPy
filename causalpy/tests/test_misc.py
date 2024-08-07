#   Copyright 2024 The PyMC Labs Developers
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
"""
Miscellaneous unit tests
"""

import causalpy as cp

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


def test_regression_kink_gradient_change():
    """Test function to numerically calculate the change in gradient around the kink
    point in regression kink designs"""
    # test no change in gradient
    assert cp.RegressionKink._eval_gradient_change(-1, 0, 1, 1) == 0.0
    assert cp.RegressionKink._eval_gradient_change(1, 0, -1, 1) == 0.0
    assert cp.RegressionKink._eval_gradient_change(0, 0, 0, 1) == 0.0
    # test positive change in gradient
    assert cp.RegressionKink._eval_gradient_change(0, 0, 1, 1) == 1.0
    assert cp.RegressionKink._eval_gradient_change(0, 0, 2, 1) == 2.0
    assert cp.RegressionKink._eval_gradient_change(-1, -1, 2, 1) == 3.0
    assert cp.RegressionKink._eval_gradient_change(-1, 0, 2, 1) == 1.0
    # test negative change in gradient
    assert cp.RegressionKink._eval_gradient_change(0, 0, -1, 1) == -1.0
    assert cp.RegressionKink._eval_gradient_change(0, 0, -2, 1) == -2.0
    assert cp.RegressionKink._eval_gradient_change(-1, -1, -2, 1) == -1.0
    assert cp.RegressionKink._eval_gradient_change(1, 0, -2, 1) == -1.0
