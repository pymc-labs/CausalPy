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
    _call_saturation_transform,
    _call_seasonality_component_apply,
    _call_time_component_apply,
    _uses_xtensor_api,
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
        def apply(self, dayofperiod, result_callback=None):  # noqa: ANN001, ARG002
            calls["dayofperiod"] = dayofperiod
            return "seasonality-result"

    signal = pt.vector("signal")

    result = _call_seasonality_component_apply(FakeSeasonalityComponent(), signal)

    assert result == "seasonality-result"
    assert calls == {"dayofperiod": signal}


def test_call_seasonality_component_apply_retries_with_xtensor():
    calls: dict[str, object] = {}

    class FakeXTensorResult:
        def __init__(self, values: str) -> None:
            self.values = values

    class FakeSeasonalityComponent:
        def apply(self, dayofperiod, sum=True):  # noqa: ANN001, FBT002
            calls["has_dims"] = hasattr(dayofperiod.type, "dims")
            calls["sum"] = sum
            return FakeXTensorResult("xtensor-result")

    result = _call_seasonality_component_apply(
        FakeSeasonalityComponent(),
        pt.vector("signal"),
    )

    assert result == "xtensor-result"
    assert calls == {
        "has_dims": True,
        "sum": True,
    }


def test_call_time_component_apply_supports_xtensor_source():
    calls: dict[str, object] = {}

    class FakeXTensorResult:
        def __init__(self, values: str) -> None:
            self.values = values

    class FakeTimeComponent:
        def apply(self, t):  # noqa: ANN001
            # as_xtensor compatibility path
            calls["has_dims"] = hasattr(t.type, "dims")
            return FakeXTensorResult("xtensor-time-result")

    result = _call_time_component_apply(FakeTimeComponent(), pt.vector("signal"))

    assert result == "xtensor-time-result"
    assert calls == {"has_dims": True}


def test_uses_xtensor_api_falls_back_to_code_names(monkeypatch):
    namespace: dict[str, object] = {}
    exec(  # noqa: S102
        "def fake_transform(x):\n    return as_xtensor(x)\n",
        {},
        namespace,
    )
    fake_transform = namespace["fake_transform"]

    def raise_oserror(_function):  # noqa: ANN001
        raise OSError("source unavailable")

    monkeypatch.setattr("causalpy.pymc_models.inspect.getsource", raise_oserror)

    assert _uses_xtensor_api(fake_transform)


def test_call_saturation_transform_supports_tensor_input():
    calls: dict[str, object] = {}

    def fake_saturation_transform(x, *, slope, kappa):  # noqa: ANN001
        calls.update({"x": x, "slope": slope, "kappa": kappa})
        return "saturation-result"

    signal = pt.vector("signal")

    result = _call_saturation_transform(
        fake_saturation_transform,
        signal,
        slope=2.0,
        kappa=10.0,
    )

    assert result == "saturation-result"
    assert calls == {
        "x": signal,
        "slope": 2.0,
        "kappa": 10.0,
    }


def test_call_saturation_transform_supports_xtensor_input():
    calls: dict[str, object] = {}

    class FakeXTensorResult:
        def __init__(self, values: str) -> None:
            self.values = values

    def fake_saturation_transform(x, *, lam):  # noqa: ANN001
        # as_xtensor compatibility path
        calls.update({"has_dims": hasattr(x.type, "dims"), "lam": lam})
        return FakeXTensorResult("xtensor-saturation-result")

    result = _call_saturation_transform(
        fake_saturation_transform,
        pt.vector("signal"),
        lam=0.5,
    )

    assert result == "xtensor-saturation-result"
    assert calls == {
        "has_dims": True,
        "lam": 0.5,
    }
