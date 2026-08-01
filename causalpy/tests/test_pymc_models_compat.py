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
"""Compatibility tests for optional pymc-marketing components in pymc_models."""

import numpy as np
import pytensor.tensor as pt
import pytest
import xarray as xr

from causalpy.pymc_models import (
    BayesianBasisExpansionTimeSeries,
    StateSpaceTimeSeries,
    _call_seasonality_component_apply,
    _call_time_component_apply,
    _uses_xtensor_api,
)


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


def test_call_seasonality_component_apply_supports_xtensor_signature():
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


def test_call_time_component_apply_supports_tensor_input():
    calls: dict[str, object] = {}

    class FakeTimeComponent:
        def apply(self, t):  # noqa: ANN001
            calls["t"] = t
            return "time-result"

    signal = pt.vector("signal")

    result = _call_time_component_apply(FakeTimeComponent(), signal)

    assert result == "time-result"
    assert calls == {"t": signal}


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


def test_state_space_defaults_use_level_trend(monkeypatch):
    """State-space defaults dispatch to the supported pymc-extras trend component."""
    statespace = pytest.importorskip("pymc_extras.statespace")
    structural = statespace.structural
    calls: list[int] = []

    class FakeLevelTrend:
        def __init__(self, *, order: int) -> None:
            calls.append(order)

    monkeypatch.setattr(structural, "LevelTrend", FakeLevelTrend)

    model = StateSpaceTimeSeries(level_order=1)

    trend = model._get_trend_component()

    assert type(trend) is FakeLevelTrend
    assert calls == [model.level_order]
    assert model._get_trend_component() is trend


def test_pymc_marketing_v1_defaults_build_and_sample():
    """Default BSTS components work with the PyMC-6-compatible marketing branch."""
    pytest.importorskip(
        "pymc_marketing", reason="pymc-marketing optional for default BSTS components"
    )
    from pymc_marketing.mmm import LinearTrend, YearlyFourier
    from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation

    dates = np.arange(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-15"),
        dtype="datetime64[D]",
    )
    X = xr.DataArray(
        np.empty((len(dates), 0)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": dates, "coeffs": []},
    )
    y = xr.DataArray(
        np.linspace(0.0, 1.0, len(dates))[:, None],
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": dates, "treated_units": ["unit_0"]},
    )
    model = BayesianBasisExpansionTimeSeries(
        n_order=1,
        n_changepoints_trend=2,
        sample_kwargs={
            "chains": 1,
            "cores": 1,
            "draws": 10,
            "tune": 10,
            "progressbar": False,
            "random_seed": 42,
        },
    )

    assert isinstance(model._get_trend_component(), LinearTrend)
    assert isinstance(model._get_seasonality_component(), YearlyFourier)
    assert callable(geometric_adstock)
    assert callable(logistic_saturation)

    idata = model.fit(X, y)

    assert {"trend_component", "season_component"} <= set(idata.posterior.data_vars)
    assert idata.posterior.sizes["chain"] == 1
    assert idata.posterior.sizes["draw"] == 10
