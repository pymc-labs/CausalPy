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
"""
Tests for StaggeredDifferenceInDifferences experiment class.
"""

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.data.simulate_data import generate_staggered_did_data

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


# ==============================================================================
# Integration Tests
# ==============================================================================


@pytest.mark.integration
def test_staggered_did_pymc(mock_pymc_sample):
    """
    Test StaggeredDifferenceInDifferences with PyMC model.

    Checks:
    1. Result is correct type
    2. Augmented data contains expected columns
    3. ATT tables are computed
    4. Plot can be generated
    """
    df = generate_staggered_did_data(
        n_units=30,
        n_time_periods=15,
        treatment_cohorts={5: 10, 10: 10},
        seed=42,
    )

    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        treatment_time_variable_name="treatment_time",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Check result type
    assert isinstance(result, cp.StaggeredDifferenceInDifferences)

    # Check augmented data
    assert "G" in result.data_.columns
    assert "event_time" in result.data_.columns
    assert "y_hat0" in result.data_.columns
    assert "tau_hat" in result.data_.columns

    # Check ATT tables exist
    assert hasattr(result, "att_group_time_")
    assert hasattr(result, "att_event_time_")
    assert len(result.att_event_time_) > 0

    # Check ATT table columns for Bayesian
    assert "event_time" in result.att_event_time_.columns
    assert "att" in result.att_event_time_.columns
    assert "att_lower" in result.att_event_time_.columns
    assert "att_upper" in result.att_event_time_.columns

    # Check plot
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


@pytest.mark.integration
def test_staggered_did_sklearn():
    """
    Test StaggeredDifferenceInDifferences with sklearn LinearRegression.
    """
    df = generate_staggered_did_data(
        n_units=30,
        n_time_periods=15,
        treatment_cohorts={5: 10, 10: 10},
        seed=42,
    )

    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        treatment_time_variable_name="treatment_time",
        model=LinearRegression(),
    )

    # Check result type
    assert isinstance(result, cp.StaggeredDifferenceInDifferences)

    # Check augmented data
    assert "G" in result.data_.columns
    assert "tau_hat" in result.data_.columns

    # Check ATT tables
    assert len(result.att_event_time_) > 0
    assert "event_time" in result.att_event_time_.columns
    assert "att" in result.att_event_time_.columns

    # Check plot
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


@pytest.mark.integration
def test_staggered_did_recovers_known_effect_sklearn():
    """
    Test that StaggeredDifferenceInDifferences recovers known treatment effects.

    Uses synthetic data with known constant treatment effect.
    Uses sklearn model for numerical accuracy (mock PyMC doesn't do real MCMC).
    """
    # Generate data with known constant effect of 2.0
    constant_effect = 2.0
    df = generate_staggered_did_data(
        n_units=50,
        n_time_periods=20,
        treatment_cohorts={5: 15, 10: 15, 15: 10},
        treatment_effects={0: constant_effect, 1: constant_effect, 2: constant_effect},
        sigma=0.1,  # Low noise for better recovery
        seed=123,
    )

    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        treatment_time_variable_name="treatment_time",
        model=LinearRegression(),
    )

    # Check that recovered effects are close to true effect
    # (with some tolerance for noise)
    avg_att = result.att_event_time_["att"].mean()
    assert abs(avg_att - constant_effect) < 0.5, (
        f"Recovered ATT {avg_att:.2f} is too far from true effect {constant_effect}"
    )


# ==============================================================================
# Unit Tests - Input Validation
# ==============================================================================


def test_staggered_did_missing_unit_column():
    """Test that missing unit column raises DataException."""
    df = pd.DataFrame(
        {
            "time": [0, 1, 2, 0, 1, 2],
            "treated": [0, 0, 1, 0, 0, 0],
            "y": [1, 2, 3, 1, 2, 2],
        }
    )

    with pytest.raises(cp.custom_exceptions.DataException, match="unit"):
        cp.StaggeredDifferenceInDifferences(
            df,
            formula="y ~ 1 + C(time)",
            unit_variable_name="unit",  # Does not exist
            time_variable_name="time",
            model=LinearRegression(),
        )


def test_staggered_did_missing_time_column():
    """Test that missing time column raises DataException."""
    df = pd.DataFrame(
        {
            "unit": [0, 0, 0, 1, 1, 1],
            "treated": [0, 0, 1, 0, 0, 0],
            "y": [1, 2, 3, 1, 2, 2],
        }
    )

    with pytest.raises(cp.custom_exceptions.DataException, match="time"):
        cp.StaggeredDifferenceInDifferences(
            df,
            formula="y ~ 1 + C(unit)",
            unit_variable_name="unit",
            time_variable_name="time",  # Does not exist
            model=LinearRegression(),
        )


def test_staggered_did_missing_treated_column():
    """Test that missing treated column raises DataException."""
    df = pd.DataFrame(
        {
            "unit": [0, 0, 0, 1, 1, 1],
            "time": [0, 1, 2, 0, 1, 2],
            "y": [1, 2, 3, 1, 2, 2],
        }
    )

    with pytest.raises(cp.custom_exceptions.DataException, match="treated"):
        cp.StaggeredDifferenceInDifferences(
            df,
            formula="y ~ 1 + C(unit) + C(time)",
            unit_variable_name="unit",
            time_variable_name="time",
            treated_variable_name="treated",  # Does not exist
            model=LinearRegression(),
        )


def test_staggered_did_non_absorbing_treatment():
    """Test that non-absorbing treatment raises DataException."""
    # Create data where unit 0 switches treatment off (not absorbing)
    df = pd.DataFrame(
        {
            "unit": [0, 0, 0, 0, 1, 1, 1, 1],
            "time": [0, 1, 2, 3, 0, 1, 2, 3],
            "treated": [0, 1, 0, 1, 0, 0, 0, 0],  # Unit 0 switches back to 0
            "y": [1, 2, 3, 4, 1, 2, 2, 3],
        }
    )

    with pytest.raises(cp.custom_exceptions.DataException, match="absorbing"):
        cp.StaggeredDifferenceInDifferences(
            df,
            formula="y ~ 1 + C(unit) + C(time)",
            unit_variable_name="unit",
            time_variable_name="time",
            treated_variable_name="treated",
            model=LinearRegression(),
        )


def test_staggered_did_formula_missing_outcome():
    """Test that formula with missing outcome raises FormulaException."""
    df = generate_staggered_did_data(
        n_units=30, n_time_periods=10, treatment_cohorts={5: 10}, seed=42
    )

    with pytest.raises(cp.custom_exceptions.FormulaException):
        cp.StaggeredDifferenceInDifferences(
            df,
            formula="nonexistent_y ~ 1 + C(unit) + C(time)",
            unit_variable_name="unit",
            time_variable_name="time",
            treated_variable_name="treated",
            model=LinearRegression(),
        )


# ==============================================================================
# Unit Tests - Core Functionality
# ==============================================================================


def test_no_treated_in_training_set():
    """Verify that no treated observations enter the training set."""
    df = generate_staggered_did_data(
        n_units=20,
        n_time_periods=10,
        treatment_cohorts={5: 10},
        seed=42,
    )

    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        model=LinearRegression(),
    )

    # Check that training set only contains untreated observations
    # The untreated observations are marked in data_["_is_untreated"]
    training_mask = result.data_["_is_untreated"]
    treated_in_training = result.data_.loc[training_mask, "treated"].sum()

    assert treated_in_training == 0, (
        f"Found {treated_in_training} treated observations in training set"
    )


def test_never_treated_and_not_yet_treated_as_controls():
    """Verify both never-treated and not-yet-treated units are used as controls."""
    # Create data with:
    # - 5 never-treated units
    # - 5 units treated at time 5
    df = generate_staggered_did_data(
        n_units=10,
        n_time_periods=10,
        treatment_cohorts={5: 5},  # 5 treated, 5 never-treated
        seed=42,
    )

    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        treatment_time_variable_name="treatment_time",
        model=LinearRegression(),
    )

    # Check never-treated units are in training set (all periods)
    never_treated = result.data_[result.data_["G"] == np.inf]
    assert all(never_treated["_is_untreated"]), (
        "Never-treated units should all be in training set"
    )

    # Check not-yet-treated observations are in training set
    eventually_treated = result.data_[result.data_["G"] != np.inf]
    pre_treatment = eventually_treated[
        eventually_treated["time"] < eventually_treated["G"]
    ]
    assert all(pre_treatment["_is_untreated"]), (
        "Pre-treatment periods for eventually-treated units should be in training set"
    )


def test_treatment_time_inference():
    """Test that treatment time is correctly inferred from treated column."""
    df = pd.DataFrame(
        {
            "unit": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            "time": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            "treated": [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
            "y": [1, 2, 5, 6, 1, 4, 5, 6, 1, 2, 2, 3],
        }
    )

    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        model=LinearRegression(),
    )

    # Check treatment times
    assert result.data_.loc[result.data_["unit"] == 0, "G"].iloc[0] == 2
    assert result.data_.loc[result.data_["unit"] == 1, "G"].iloc[0] == 1
    assert result.data_.loc[result.data_["unit"] == 2, "G"].iloc[0] == np.inf


def test_event_time_computation():
    """Test that event time is correctly computed."""
    df = pd.DataFrame(
        {
            "unit": [0, 0, 0, 0, 1, 1, 1, 1],
            "time": [0, 1, 2, 3, 0, 1, 2, 3],
            "treated": [0, 0, 1, 1, 0, 0, 0, 0],
            "treatment_time": [2, 2, 2, 2, np.inf, np.inf, np.inf, np.inf],
            "y": [1, 2, 5, 6, 1, 2, 2, 3],
        }
    )

    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treatment_time_variable_name="treatment_time",
        model=LinearRegression(),
    )

    # Check event times for unit 0 (treated at time 2)
    unit0_data = result.data_[result.data_["unit"] == 0]
    expected_event_times = [-2, -1, 0, 1]  # time - G = time - 2
    actual_event_times = unit0_data["event_time"].tolist()
    assert actual_event_times == expected_event_times

    # Check event times for never-treated unit 1 (should be NaN)
    unit1_data = result.data_[result.data_["unit"] == 1]
    assert all(pd.isna(unit1_data["event_time"]))


def test_event_window_filtering():
    """Test that event_window parameter correctly filters event-time ATTs."""
    df = generate_staggered_did_data(
        n_units=30,
        n_time_periods=15,
        treatment_cohorts={5: 15},
        seed=42,
    )

    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        event_window=(-2, 3),  # Only event times -2 to 3
        model=LinearRegression(),
    )

    # Check that event times are within window
    event_times = result.att_event_time_["event_time"].values
    assert all(event_times >= -2), "Event times should be >= -2"
    assert all(event_times <= 3), "Event times should be <= 3"


# ==============================================================================
# Unit Tests - Summary and Reporting
# ==============================================================================


def test_staggered_did_summary():
    """Test that summary() runs without error."""
    df = generate_staggered_did_data(
        n_units=30, n_time_periods=10, treatment_cohorts={5: 10, 8: 10}, seed=42
    )

    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        model=LinearRegression(),
    )

    # Should not raise
    result.summary()


def test_staggered_did_get_plot_data():
    """Test get_plot_data methods return expected data."""
    df = generate_staggered_did_data(
        n_units=30, n_time_periods=10, treatment_cohorts={5: 10, 8: 10}, seed=42
    )

    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        model=LinearRegression(),
    )

    plot_data = result.get_plot_data()

    assert isinstance(plot_data, pd.DataFrame)
    assert "event_time" in plot_data.columns
    assert "att" in plot_data.columns


@pytest.mark.integration
def test_staggered_did_effect_summary(mock_pymc_sample):
    """Test effect_summary() for staggered DiD."""
    df = generate_staggered_did_data(
        n_units=30, n_time_periods=10, treatment_cohorts={5: 10, 8: 10}, seed=42
    )

    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    summary = result.effect_summary()

    assert hasattr(summary, "table")
    assert hasattr(summary, "text")
    assert isinstance(summary.table, pd.DataFrame)
    assert isinstance(summary.text, str)
    assert "Staggered DiD" in summary.text


# ==============================================================================
# Synthetic Data Generator Tests
# ==============================================================================


def test_generate_staggered_did_data():
    """Test the synthetic data generator."""
    df = generate_staggered_did_data(
        n_units=50,
        n_time_periods=20,
        treatment_cohorts={5: 10, 10: 10, 15: 10},
        seed=42,
    )

    assert isinstance(df, pd.DataFrame)

    # Check required columns
    required_cols = ["unit", "time", "treated", "treatment_time", "y", "y0", "tau"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # Check dimensions
    assert len(df) == 50 * 20  # n_units * n_time_periods

    # Check treatment cohorts
    assert df["treatment_time"].nunique() == 4  # 3 cohorts + never-treated (inf)

    # Check absorbing treatment
    for unit in df["unit"].unique():
        unit_data = df[df["unit"] == unit].sort_values("time")
        treated = unit_data["treated"].values
        # Once treated, should stay treated
        first_treated = np.where(treated == 1)[0]
        if len(first_treated) > 0:
            assert all(treated[first_treated[0] :] == 1)


def test_generate_staggered_did_data_too_many_units():
    """Test that too many units in cohorts raises ValueError."""
    with pytest.raises(ValueError, match="exceeds n_units"):
        generate_staggered_did_data(
            n_units=10,
            treatment_cohorts={5: 20},  # More than n_units
        )
