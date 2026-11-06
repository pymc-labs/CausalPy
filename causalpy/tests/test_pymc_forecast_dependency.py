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
"""Tests for the optional pymc-forecast dependency boundary."""

import runpy
import sys
from importlib import metadata
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

from causalpy.pymc_forecast_models import PyMCForecastModel, _import_pymc_forecast


@pytest.fixture
def forecast_dependency(monkeypatch):
    package = ModuleType("pymc_forecast")
    monkeypatch.setitem(sys.modules, "pymc_forecast", package)
    get_version = Mock(return_value="0.3.0")
    monkeypatch.setattr(metadata, "version", get_version)
    return package, get_version


@pytest.mark.parametrize(
    ("installed_version", "supported"),
    [
        ("0.2.0", False),
        ("0.3.0", True),
        ("0.3.0.post1", True),
        ("0.3.0+local", True),
        ("0.3.0rc1", False),
        ("0.3.1rc1", False),
        ("0.4.0rc1", False),
        ("0.4.0", False),
        ("0.30.0", False),
    ],
)
def test_forecast_version_requirement(
    forecast_dependency, installed_version, supported
):
    package, get_version = forecast_dependency
    get_version.return_value = installed_version
    if supported:
        assert _import_pymc_forecast() is package
    else:
        forecaster = Mock()
        with pytest.raises(
            ImportError, match="requires pymc-forecast>=0.3,<0.4"
        ) as error:
            PyMCForecastModel(Mock(), forecaster=forecaster)
        assert installed_version in str(error.value)
        assert "pip install --upgrade" in str(error.value)
        forecaster.assert_not_called()


@pytest.mark.parametrize("installed_version", [None, "unknown"])
def test_forecast_version_metadata_errors(forecast_dependency, installed_version):
    _, get_version = forecast_dependency
    if installed_version is None:
        get_version.side_effect = metadata.PackageNotFoundError("pymc-forecast")
    else:
        get_version.return_value = installed_version
    with pytest.raises(ImportError, match="version") as error:
        _import_pymc_forecast()
    assert ">=0.3,<0.4" in str(error.value)
    assert "pip install" in str(error.value)


def test_missing_forecast_extra_has_actionable_install_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymc_forecast", None)
    with pytest.raises(ImportError, match=r"pip install causalpy\[forecast\]") as error:
        _import_pymc_forecast()
    assert "pymc-forecast[extras]>=0.3,<0.4" in str(error.value)


@pytest.mark.parametrize(
    "module_name",
    ["test_pymc_forecast_adapter.py", "test_pymc_forecast_damped_trend.py"],
)
def test_forecast_integration_version_skips(
    monkeypatch, forecast_dependency, module_name
):
    package, _ = forecast_dependency
    module_path = str(Path(__file__).with_name(module_name))
    for installed_version in ("0.2.0", "0.3.1rc1", "0.4.0"):
        package.__version__ = installed_version
        with pytest.raises(pytest.skip.Exception):
            runpy.run_path(module_path)

    package.__version__ = "0.3.0"
    for name in ("ForecastingModel", "Forecaster", "PathfinderForecaster"):
        monkeypatch.setattr(package, name, object, raising=False)
    guard = Mock(side_effect=ImportError("guard regression"))
    monkeypatch.setattr("causalpy.pymc_forecast_models._import_pymc_forecast", guard)
    try:
        runpy.run_path(module_path)
    except pytest.skip.Exception as err:
        pytest.fail(f"Supported dependency was skipped: {err}")
    guard.assert_not_called()
