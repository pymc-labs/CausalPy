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
"""Contract tests for the seeded visual-regression plot script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "generate_plots.py"


def _load_plot_script():
    """Load the standalone plot script without invoking its CLI entry point."""
    spec = importlib.util.spec_from_file_location(
        "causalpy_generate_plots", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plot script at {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seeded_plot_builders_are_reproducible(monkeypatch):
    """Synthetic panel data are reproducible and sampling configuration is seeded."""
    script = _load_plot_script()
    captured_data: list[pd.DataFrame] = []

    class FakePanelRegression:
        """Capture synthetic panel data without fitting an experiment."""

        def __init__(self, *, data, **kwargs):
            captured_data.append(data.copy())

    monkeypatch.setattr(script.cp, "PanelRegression", FakePanelRegression)

    script.build_panel("ols")
    script.build_panel("ols")

    assert script.lr().sample_kwargs["random_seed"] == 42
    assert script.SAMPLE_KWARGS["random_seed"] == 42
    pd.testing.assert_frame_equal(captured_data[0], captured_data[1])


def test_ols_plot_builders_match_current_experiment_constructors():
    """Every OLS-capable builder constructs the current experiment API."""
    script = _load_plot_script()
    ols_builders = {
        "its",
        "sc",
        "did",
        "piecewise_its",
        "rd",
        "staggered_did",
        "panel",
    }
    bayes_only_builders = {"sdid", "prepostnegd", "rkink"}

    assert set(script.BUILDERS) == ols_builders | bayes_only_builders

    for name in ols_builders:
        result, plots = script.BUILDERS[name]("ols")
        assert result is not None
        assert plots
        assert all(callable(plot_call) for _, plot_call in plots)

    for name in bayes_only_builders:
        with pytest.raises(KeyError):
            script.BUILDERS[name]("ols")
