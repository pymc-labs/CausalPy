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

import pytensor.tensor as pt

from causalpy.pymc_models import (
    _call_geometric_adstock,
    _call_seasonality_component_apply,
)


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

    class FakeXTensorResult:
        def __init__(self, values: str) -> None:
            self.values = values

    def fake_geometric_adstock(x, *, alpha, l_max, normalize, dim, mode=None):  # noqa: ANN001
        calls.update(
            {
                "has_type": hasattr(x, "type"),
                "alpha": alpha,
                "l_max": l_max,
                "normalize": normalize,
                "dim": dim,
                "mode": mode,
            }
        )
        return FakeXTensorResult("dim-result")

    result = _call_geometric_adstock(
        fake_geometric_adstock,
        pt.vector("signal"),
        alpha=0.5,
        l_max=12,
        normalize=True,
        mode="After",
    )

    assert result == "dim-result"
    assert calls == {
        "has_type": True,
        "alpha": 0.5,
        "l_max": 12,
        "normalize": True,
        "dim": "obs_ind",
        "mode": "After",
    }


def test_call_seasonality_component_apply_supports_tensor_input():
    calls: dict[str, object] = {}

    class FakeSeasonalityComponent:
        def apply(self, dayofperiod):  # noqa: ANN001
            calls["dayofperiod"] = dayofperiod
            return "seasonality-result"

    signal = pt.vector("signal")

    result = _call_seasonality_component_apply(FakeSeasonalityComponent(), signal)

    assert result == "seasonality-result"
    assert calls == {"dayofperiod": signal}


def test_call_seasonality_component_apply_retries_with_xtensor():
    calls: dict[str, object] = {"attempts": 0}

    class FakeXTensorResult:
        def __init__(self, values: str) -> None:
            self.values = values

    class FakeSeasonalityComponent:
        def apply(self, dayofperiod):  # noqa: ANN001
            calls["attempts"] += 1
            calls[f"has_dims_{calls['attempts']}"] = hasattr(dayofperiod.type, "dims")
            if not hasattr(dayofperiod.type, "dims"):
                raise TypeError(
                    "non-scalar TensorVariable cannot be converted to XTensorVariable without dims."
                )
            return FakeXTensorResult("xtensor-result")

    result = _call_seasonality_component_apply(
        FakeSeasonalityComponent(),
        pt.vector("signal"),
    )

    assert result == "xtensor-result"
    assert calls == {
        "attempts": 2,
        "has_dims_1": False,
        "has_dims_2": True,
    }
