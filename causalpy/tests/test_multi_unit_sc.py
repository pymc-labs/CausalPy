#   Copyright 2025 - 2025 The PyMC Labs Developers
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
Tests for multiple treated units in SyntheticControl experiment
"""

import numpy as np
import pandas as pd
import pytest

from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pymc_models import WeightedSumFitter

# Use consistent sample kwargs for fast testing
sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2, "progressbar": False}


@pytest.fixture
def multi_unit_sc_data(rng):
    """Generate synthetic data for SyntheticControl with multiple treated units."""
    n_obs = 60
    n_control = 4
    n_treated = 3

    # Create time index
    time_index = pd.date_range("2020-01-01", periods=n_obs, freq="D")
    treatment_time = time_index[40]  # Intervention at day 40

    # Control unit data
    control_data = {}
    for i in range(n_control):
        control_data[f"control_{i}"] = rng.normal(10, 2, n_obs) + np.sin(
            np.arange(n_obs) * 0.1
        )

    # Treated unit data (combinations of control units with some noise)
    treated_data = {}
    for j in range(n_treated):
        # Each treated unit is a different weighted combination of controls
        weights = rng.dirichlet(np.ones(n_control))
        base_signal = sum(
            weights[i] * control_data[f"control_{i}"] for i in range(n_control)
        )

        # Add treatment effect after intervention
        treatment_effect = np.zeros(n_obs)
        treatment_effect[40:] = rng.normal(
            5, 1, n_obs - 40
        )  # Positive effect after treatment

        treated_data[f"treated_{j}"] = (
            base_signal + treatment_effect + rng.normal(0, 0.5, n_obs)
        )

    # Create DataFrame
    df = pd.DataFrame({**control_data, **treated_data}, index=time_index)

    control_units = [f"control_{i}" for i in range(n_control)]
    treated_units = [f"treated_{j}" for j in range(n_treated)]

    return df, treatment_time, control_units, treated_units


@pytest.fixture
def single_unit_sc_data(rng):
    """Generate synthetic data for SyntheticControl with single treated unit."""
    n_obs = 60
    n_control = 4

    # Create time index
    time_index = pd.date_range("2020-01-01", periods=n_obs, freq="D")
    treatment_time = time_index[40]  # Intervention at day 40

    # Control unit data
    control_data = {}
    for i in range(n_control):
        control_data[f"control_{i}"] = rng.normal(10, 2, n_obs) + np.sin(
            np.arange(n_obs) * 0.1
        )

    # Single treated unit data
    weights = rng.dirichlet(np.ones(n_control))
    base_signal = sum(
        weights[i] * control_data[f"control_{i}"] for i in range(n_control)
    )

    # Add treatment effect after intervention
    treatment_effect = np.zeros(n_obs)
    treatment_effect[40:] = rng.normal(
        5, 1, n_obs - 40
    )  # Positive effect after treatment

    treated_data = {
        "treated_0": base_signal + treatment_effect + rng.normal(0, 0.5, n_obs)
    }

    # Create DataFrame
    df = pd.DataFrame({**control_data, **treated_data}, index=time_index)

    control_units = [f"control_{i}" for i in range(n_control)]
    treated_units = ["treated_0"]

    return df, treatment_time, control_units, treated_units


class TestSyntheticControlMultiUnit:
    """Tests for SyntheticControl experiment with multiple treated units."""

    def test_multi_unit_initialization(self, multi_unit_sc_data):
        """Test that SyntheticControl can initialize with multiple treated units."""
        df, treatment_time, control_units, treated_units = multi_unit_sc_data

        model = WeightedSumFitter(sample_kwargs=sample_kwargs)

        # Should initialize without error
        sc = SyntheticControl(
            data=df,
            treatment_time=treatment_time,
            control_units=control_units,
            treated_units=treated_units,
            model=model,
        )

        # Check basic attributes
        assert sc.treated_units == treated_units
        assert sc.control_units == control_units
        assert sc.treatment_time == treatment_time

        # Check data shapes
        assert sc.datapre_treated.shape == (40, len(treated_units))
        assert sc.datapost_treated.shape == (20, len(treated_units))
        assert sc.datapre_control.shape == (40, len(control_units))
        assert sc.datapost_control.shape == (20, len(control_units))

    def test_multi_unit_scoring(self, multi_unit_sc_data):
        """Test that scoring works with multiple treated units."""
        df, treatment_time, control_units, treated_units = multi_unit_sc_data

        model = WeightedSumFitter(sample_kwargs=sample_kwargs)

        sc = SyntheticControl(
            data=df,
            treatment_time=treatment_time,
            control_units=control_units,
            treated_units=treated_units,
            model=model,
        )

        # Score should be a pandas Series with separate entries for each unit
        assert isinstance(sc.score, pd.Series)

        # Check that we have r2 and r2_std for each treated unit
        for unit in treated_units:
            assert f"{unit}_r2" in sc.score.index
            assert f"{unit}_r2_std" in sc.score.index

    def test_multi_unit_summary(self, multi_unit_sc_data, capsys):
        """Test that summary works with multiple treated units."""
        df, treatment_time, control_units, treated_units = multi_unit_sc_data

        model = WeightedSumFitter(sample_kwargs=sample_kwargs)

        sc = SyntheticControl(
            data=df,
            treatment_time=treatment_time,
            control_units=control_units,
            treated_units=treated_units,
            model=model,
        )

        # Test summary
        sc.summary(round_to=3)

        captured = capsys.readouterr()
        output = captured.out

        # Check that output contains information for multiple treated units
        assert "Treated units:" in output
        for unit in treated_units:
            assert unit in output

    def test_single_unit_backward_compatibility(self, single_unit_sc_data):
        """Test that single treated unit still works (backward compatibility)."""
        df, treatment_time, control_units, treated_units = single_unit_sc_data

        model = WeightedSumFitter(sample_kwargs=sample_kwargs)

        sc = SyntheticControl(
            data=df,
            treatment_time=treatment_time,
            control_units=control_units,
            treated_units=treated_units,
            model=model,
        )

        # Check basic attributes
        assert sc.treated_units == treated_units
        assert len(sc.treated_units) == 1

        # Score should still work
        assert isinstance(sc.score, pd.Series)
        assert "r2" in sc.score.index
        assert "r2_std" in sc.score.index

    def test_multi_unit_plotting(self, multi_unit_sc_data):
        """Test that plotting works with multiple treated units."""
        df, treatment_time, control_units, treated_units = multi_unit_sc_data

        model = WeightedSumFitter(sample_kwargs=sample_kwargs)

        sc = SyntheticControl(
            data=df,
            treatment_time=treatment_time,
            control_units=control_units,
            treated_units=treated_units,
            model=model,
        )

        # Should be able to create plots without error
        # Test default (first unit)
        fig, ax = sc.plot()

        # Test specific unit by index
        fig2, ax2 = sc.plot(treated_unit=1)

        # Test specific unit by name
        fig3, ax3 = sc.plot(treated_unit="treated_2")

        # Check that we got the expected plot structure
        for axes in [ax, ax2, ax3]:
            assert len(axes) == 3  # Three subplots as expected

            # Check titles contain appropriate info
            title = axes[0].get_title()
            assert (
                "R²" in title or "R^2" in title
            )  # Score information should be displayed
            assert "Unit:" in title  # Unit information should be displayed

    def test_multi_unit_plot_data(self, multi_unit_sc_data):
        """Test that plot data generation works with multiple treated units."""
        df, treatment_time, control_units, treated_units = multi_unit_sc_data

        model = WeightedSumFitter(sample_kwargs=sample_kwargs)

        sc = SyntheticControl(
            data=df,
            treatment_time=treatment_time,
            control_units=control_units,
            treated_units=treated_units,
            model=model,
        )

        # Should be able to generate plot data without error
        plot_df = sc.get_plot_data_bayesian()

        # Check that dataframe has expected structure
        assert isinstance(plot_df, pd.DataFrame)
        assert "prediction" in plot_df.columns
        assert "impact" in plot_df.columns

        # Check that we have data for both pre and post periods
        assert len(plot_df) == len(df)

    def test_multi_unit_plotting_invalid_unit(self, multi_unit_sc_data):
        """Test that plotting with invalid treated unit raises appropriate errors."""
        df, treatment_time, control_units, treated_units = multi_unit_sc_data

        model = WeightedSumFitter(sample_kwargs=sample_kwargs)

        sc = SyntheticControl(
            data=df,
            treatment_time=treatment_time,
            control_units=control_units,
            treated_units=treated_units,
            model=model,
        )

        # Test invalid index
        with pytest.raises(ValueError, match="treated_unit index.*out of range"):
            sc.plot(treated_unit=10)

        # Test invalid name
        with pytest.raises(ValueError, match="treated_unit.*not found"):
            sc.plot(treated_unit="invalid_unit")

        # Test invalid type
        with pytest.raises(
            ValueError, match="treated_unit must be.*integer.*string.*None"
        ):
            sc.plot(treated_unit=3.14)
