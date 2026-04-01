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
"""Compatibility tests for optional transformer APIs in pymc_models."""

from causalpy.pymc_models import _call_geometric_adstock


def test_call_geometric_adstock_supports_axis_signature():
    calls: dict[str, object] = {}

    def fake_geometric_adstock(x, *, alpha, l_max, normalize, axis=0, mode=None):  # noqa: ANN001
        calls.update(
            {
                "x": x,
                "alpha": alpha,
                "l_max": l_max,
                "normalize": normalize,
                "axis": axis,
                "mode": mode,
            }
        )
        return "axis-result"

    result = _call_geometric_adstock(
        fake_geometric_adstock,
        "signal",
        alpha=0.5,
        l_max=12,
        normalize=True,
        mode="After",
    )

    assert result == "axis-result"
    assert calls == {
        "x": "signal",
        "alpha": 0.5,
        "l_max": 12,
        "normalize": True,
        "axis": 0,
        "mode": "After",
    }


def test_call_geometric_adstock_supports_dim_signature():
    calls: dict[str, object] = {}

    def fake_geometric_adstock(x, *, alpha, l_max, normalize, dim, mode=None):  # noqa: ANN001
        calls.update(
            {
                "x": x,
                "alpha": alpha,
                "l_max": l_max,
                "normalize": normalize,
                "dim": dim,
                "mode": mode,
            }
        )
        return "dim-result"

    result = _call_geometric_adstock(
        fake_geometric_adstock,
        "signal",
        alpha=0.5,
        l_max=12,
        normalize=True,
        mode="After",
    )

    assert result == "dim-result"
    assert calls == {
        "x": "signal",
        "alpha": 0.5,
        "l_max": 12,
        "normalize": True,
        "dim": "obs_ind",
        "mode": "After",
    }
