#   Copyright 2022 - 2025 The PyMC Labs Developers
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

"""Tests for Event Study experiment class."""

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.custom_exceptions import DataException
from causalpy.data.simulate_data import generate_event_study_data

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2, "progressbar": False}


# ============================================================================
# Unit Tests for Data Simulation
# ============================================================================


def test_generate_event_study_data_shape():
    """Test that generate_event_study_data produces correct shape."""
    df = generate_event_study_data(n_units=20, n_time=15, seed=42)
    assert df.shape == (20 * 15, 5)
    assert list(df.columns) == ["unit", "time", "y", "treat_time", "treated"]


def test_generate_event_study_data_treated_fraction():
    """Test that treated fraction is correctly applied."""
    df = generate_event_study_data(
        n_units=100, n_time=10, treated_fraction=0.3, seed=42
    )
    n_treated_units = df[df["treated"] == 1]["unit"].nunique()
    assert n_treated_units == 30


def test_generate_event_study_data_treatment_time():
    """Test that treatment time is correctly assigned."""
    df = generate_event_study_data(
        n_units=20, n_time=20, treatment_time=10, treated_fraction=0.5, seed=42
    )
    # Treated units should have treat_time = 10
    treated_df = df[df["treated"] == 1]
    assert (treated_df["treat_time"] == 10).all()
    # Control units should have NaN treat_time
    control_df = df[df["treated"] == 0]
    assert control_df["treat_time"].isna().all()


def test_generate_event_study_data_reproducibility():
    """Test that seed produces reproducible data."""
    df1 = generate_event_study_data(n_units=10, n_time=10, seed=42)
    df2 = generate_event_study_data(n_units=10, n_time=10, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_event_study_data_custom_treatment_effects():
    """Test that custom treatment effects are applied."""
    treatment_effects = {-1: 0.0, 0: 1.0, 1: 2.0}
    df = generate_event_study_data(
        n_units=20,
        n_time=20,
        treatment_time=10,
        event_window=(-1, 1),
        treatment_effects=treatment_effects,
        seed=42,
    )
    assert df.shape == (400, 5)


# ============================================================================
# Unit Tests for Input Validation
# ============================================================================


def test_event_study_missing_column():
    """Test that missing columns raise DataException."""
    df = pd.DataFrame(
        {"unit": [0, 1], "time": [0, 0], "y": [1.0, 2.0]}
    )  # missing treat_time

    with pytest.raises(DataException, match="Required column 'treat_time' not found"):
        cp.EventStudy(
            df,
            unit_col="unit",
            time_col="time",
            outcome_col="y",
            treat_time_col="treat_time",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )


def test_event_study_invalid_event_window():
    """Test that invalid event window raises DataException."""
    df = generate_event_study_data(n_units=10, n_time=10, seed=42)

    with pytest.raises(DataException, match="event_window\\[0\\].*must be less than"):
        cp.EventStudy(
            df,
            unit_col="unit",
            time_col="time",
            outcome_col="y",
            treat_time_col="treat_time",
            event_window=(5, -5),  # Invalid: min > max
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )


def test_event_study_reference_outside_window():
    """Test that reference event time outside window raises DataException."""
    df = generate_event_study_data(n_units=10, n_time=10, seed=42)

    with pytest.raises(DataException, match="reference_event_time.*must be within"):
        cp.EventStudy(
            df,
            unit_col="unit",
            time_col="time",
            outcome_col="y",
            treat_time_col="treat_time",
            event_window=(-3, 3),
            reference_event_time=-5,  # Outside window
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )


def test_event_study_duplicate_observations():
    """Test that duplicate unit-time observations raise DataException."""
    df = pd.DataFrame(
        {
            "unit": [0, 0, 1, 1],  # Duplicate (0, 0)
            "time": [0, 0, 0, 1],
            "y": [1.0, 2.0, 3.0, 4.0],
            "treat_time": [5.0, 5.0, np.nan, np.nan],
        }
    )

    with pytest.raises(DataException, match="duplicate unit-time observations"):
        cp.EventStudy(
            df,
            unit_col="unit",
            time_col="time",
            outcome_col="y",
            treat_time_col="treat_time",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )


# ============================================================================
# Integration Tests with PyMC
# ============================================================================


@pytest.mark.integration
def test_event_study_pymc(mock_pymc_sample):
    """Test EventStudy with PyMC model."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        unit_col="unit",
        time_col="time",
        outcome_col="y",
        treat_time_col="treat_time",
        event_window=(-5, 5),
        reference_event_time=-1,
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Check result type
    assert isinstance(result, cp.EventStudy)

    # Check idata structure
    assert len(result.idata.posterior.coords["chain"]) == sample_kwargs["chains"]
    assert len(result.idata.posterior.coords["draw"]) == sample_kwargs["draws"]

    # Check event-time coefficients were extracted
    assert len(result.event_time_coeffs) == 11  # -5 to +5 inclusive
    assert result.reference_event_time in result.event_time_coeffs


@pytest.mark.integration
def test_event_study_pymc_summary(mock_pymc_sample):
    """Test EventStudy summary method with PyMC model."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        unit_col="unit",
        time_col="time",
        outcome_col="y",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Summary should not raise
    result.summary()

    # get_event_time_summary should return DataFrame
    summary_df = result.get_event_time_summary()
    assert isinstance(summary_df, pd.DataFrame)
    assert "event_time" in summary_df.columns
    assert "mean" in summary_df.columns
    assert len(summary_df) == 7  # -3 to +3 inclusive


@pytest.mark.integration
def test_event_study_pymc_plot(mock_pymc_sample):
    """Test EventStudy plot method with PyMC model."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        unit_col="unit",
        time_col="time",
        outcome_col="y",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


@pytest.mark.integration
def test_event_study_pymc_get_plot_data(mock_pymc_sample):
    """Test EventStudy get_plot_data method with PyMC model."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        unit_col="unit",
        time_col="time",
        outcome_col="y",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    plot_data = result.get_plot_data()
    assert isinstance(plot_data, pd.DataFrame)


# ============================================================================
# Integration Tests with sklearn
# ============================================================================


def test_event_study_sklearn():
    """Test EventStudy with sklearn model."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        unit_col="unit",
        time_col="time",
        outcome_col="y",
        treat_time_col="treat_time",
        event_window=(-5, 5),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    # Check result type
    assert isinstance(result, cp.EventStudy)

    # Check event-time coefficients were extracted
    assert len(result.event_time_coeffs) == 11  # -5 to +5 inclusive
    assert result.reference_event_time in result.event_time_coeffs

    # Reference coefficient should be 0
    assert result.event_time_coeffs[result.reference_event_time] == 0.0


def test_event_study_sklearn_summary():
    """Test EventStudy summary method with sklearn model."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        unit_col="unit",
        time_col="time",
        outcome_col="y",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    # Summary should not raise
    result.summary()

    # get_event_time_summary should return DataFrame
    summary_df = result.get_event_time_summary()
    assert isinstance(summary_df, pd.DataFrame)
    assert "event_time" in summary_df.columns
    assert "mean" in summary_df.columns
    assert len(summary_df) == 7  # -3 to +3 inclusive


def test_event_study_sklearn_plot():
    """Test EventStudy plot method with sklearn model."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        unit_col="unit",
        time_col="time",
        outcome_col="y",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


# ============================================================================
# Tests for Treatment Effect Recovery
# ============================================================================


def test_event_study_sklearn_recovers_effects():
    """Test that EventStudy with sklearn roughly recovers known treatment effects."""
    # Create data with known treatment effects
    treatment_effects = dict.fromkeys(range(-3, 0), 0.0)  # No pre-treatment effects
    treatment_effects.update({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})  # Constant post effect

    df = generate_event_study_data(
        n_units=100,  # More units for better estimation
        n_time=20,
        treatment_time=10,
        treated_fraction=0.5,
        event_window=(-3, 3),
        treatment_effects=treatment_effects,
        unit_fe_sigma=0.5,
        time_fe_sigma=0.3,
        noise_sigma=0.1,
        seed=42,
    )

    result = cp.EventStudy(
        df,
        unit_col="unit",
        time_col="time",
        outcome_col="y",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    # Pre-treatment coefficients should be close to 0 (relative to reference)
    for k in [-3, -2]:  # -1 is reference
        coeff = result.event_time_coeffs[k]
        assert abs(coeff) < 0.5, f"Pre-treatment coeff at k={k} should be near 0"

    # Post-treatment coefficients should be close to 1 (relative to reference=0)
    for k in [0, 1, 2, 3]:
        coeff = result.event_time_coeffs[k]
        assert 0.5 < coeff < 1.5, f"Post-treatment coeff at k={k} should be near 1"


# ============================================================================
# Edge Cases
# ============================================================================


def test_event_study_narrow_event_window():
    """Test EventStudy with narrow event window."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        unit_col="unit",
        time_col="time",
        outcome_col="y",
        treat_time_col="treat_time",
        event_window=(-1, 1),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    # Should have 3 coefficients: -1 (ref), 0, 1
    assert len(result.event_time_coeffs) == 3


def test_event_study_all_control_units():
    """Test EventStudy with all control units (edge case)."""
    df = generate_event_study_data(
        n_units=20, n_time=20, treatment_time=10, treated_fraction=0.0, seed=42
    )

    # All units are control, so no event-time coefficients can be estimated
    # The model should still run but event-time dummies will all be 0
    result = cp.EventStudy(
        df,
        unit_col="unit",
        time_col="time",
        outcome_col="y",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    # Check that result was created
    assert isinstance(result, cp.EventStudy)
