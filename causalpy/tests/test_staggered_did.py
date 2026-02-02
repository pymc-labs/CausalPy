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

    model = LinearRegression(fit_intercept=True)
    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        treatment_time_variable_name="treatment_time",
        model=model,
    )

    # Check result type
    assert isinstance(result, cp.StaggeredDifferenceInDifferences)
    assert model.fit_intercept is True
    assert result.model.fit_intercept is False

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

    # Check that recovered post-treatment effects are close to true effect
    # (with some tolerance for noise)
    # Note: att_event_time_ now includes pre-treatment placebo effects, so filter to post
    post_treatment = result.att_event_time_[result.att_event_time_["event_time"] >= 0]
    avg_att = post_treatment["att"].mean()
    assert abs(avg_att - constant_effect) < 0.5, (
        f"Recovered ATT {avg_att:.2f} is too far from true effect {constant_effect}"
    )

    # Verify pre-treatment placebo effects are close to zero
    pre_treatment = result.att_event_time_[result.att_event_time_["event_time"] < 0]
    if len(pre_treatment) > 0:
        avg_pre_att = pre_treatment["att"].mean()
        assert abs(avg_pre_att) < 0.5, (
            f"Pre-treatment placebo effect {avg_pre_att:.2f} should be close to zero"
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


def test_staggered_did_unbalanced_panel():
    """Test that StaggeredDifferenceInDifferences handles unbalanced panel data.

    Verifies that the implementation correctly handles panels where units are not
    observed in all time periods. This tests the fix for summary() which previously
    assumed balanced panels when counting never-treated units.
    """
    # Generate balanced panel
    df = generate_staggered_did_data(
        n_units=20, n_time_periods=10, treatment_cohorts={5: 10}, seed=42
    )

    # Create unbalanced panel by dropping some observations randomly
    rng = np.random.default_rng(123)
    drop_mask = rng.random(len(df)) < 0.1  # Drop ~10% of observations
    df_unbalanced = df[~drop_mask].reset_index(drop=True)

    # Verify we actually have unbalanced data
    obs_per_unit = df_unbalanced.groupby("unit").size()
    assert obs_per_unit.nunique() > 1, "Panel should be unbalanced"

    # Run experiment - should not error
    result = cp.StaggeredDifferenceInDifferences(
        df_unbalanced,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        treatment_time_variable_name="treatment_time",
        model=LinearRegression(),
    )

    # Basic sanity checks
    assert result.att_event_time_ is not None
    assert len(result.att_event_time_) > 0

    # Verify summary runs without error (this exercises the fixed counting logic)
    result.summary()


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


@pytest.mark.integration
def test_staggered_did_hdi_prob_stored_and_reported(mock_pymc_sample):
    """Test that Bayesian results store hdi_prob_ and report it correctly in prose.

    This verifies the fix for the mismatch between computed interval bounds
    (94% by default) and reported percentage in effect_summary prose.
    """
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

    # Verify hdi_prob_ is stored on the result
    assert hasattr(result, "hdi_prob_"), "Bayesian result should have hdi_prob_ attr"
    assert result.hdi_prob_ == 0.94, "Default hdi_prob_ should be 0.94"

    # Verify effect summary prose reports the correct percentage
    summary = result.effect_summary()
    assert "94% HDI" in summary.text, (
        f"Effect summary should report '94% HDI' but got: {summary.text}"
    )


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


# ==============================================================================
# Additional Coverage Tests
# ==============================================================================


def test_staggered_did_explicit_treatment_time_column():
    """Test using treatment_time_variable_name explicitly provided."""
    df = pd.DataFrame(
        {
            "unit": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            "time": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            "treated": [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
            "treatment_time": [2, 2, 2, 2, 1, 1, 1, 1, np.inf, np.inf, np.inf, np.inf],
            "y": [1, 2, 5, 6, 1, 4, 5, 6, 1, 2, 2, 3],
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

    # Check treatment times are correctly read from column
    assert result.data_.loc[result.data_["unit"] == 0, "G"].iloc[0] == 2
    assert result.data_.loc[result.data_["unit"] == 1, "G"].iloc[0] == 1
    assert result.data_.loc[result.data_["unit"] == 2, "G"].iloc[0] == np.inf


def test_staggered_did_missing_treatment_time_column():
    """Test that missing treatment_time column raises DataException."""
    df = pd.DataFrame(
        {
            "unit": [0, 0, 0, 1, 1, 1],
            "time": [0, 1, 2, 0, 1, 2],
            "treated": [0, 0, 1, 0, 0, 0],
            "y": [1, 2, 3, 1, 2, 2],
        }
    )

    with pytest.raises(cp.custom_exceptions.DataException, match="treatment_time"):
        cp.StaggeredDifferenceInDifferences(
            df,
            formula="y ~ 1 + C(unit) + C(time)",
            unit_variable_name="unit",
            time_variable_name="time",
            treatment_time_variable_name="nonexistent_treatment_time",
            model=LinearRegression(),
        )


def test_staggered_did_group_time_att_structure():
    """Test that group-time ATT table has correct structure."""
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
        model=LinearRegression(),
    )

    # Check group-time ATT table structure
    assert isinstance(result.att_group_time_, pd.DataFrame)
    assert "cohort" in result.att_group_time_.columns
    assert "time" in result.att_group_time_.columns
    assert "att" in result.att_group_time_.columns

    # Check we have multiple cohorts
    assert len(result.att_group_time_["cohort"].unique()) >= 2


def test_staggered_did_no_untreated_observations():
    """Test that having no untreated observations raises DataException."""
    # All units treated from time 0
    df = pd.DataFrame(
        {
            "unit": [0, 0, 0, 1, 1, 1],
            "time": [0, 1, 2, 0, 1, 2],
            "treated": [1, 1, 1, 1, 1, 1],
            "y": [1, 2, 3, 1, 2, 2],
        }
    )

    with pytest.raises(cp.custom_exceptions.DataException, match="No untreated"):
        cp.StaggeredDifferenceInDifferences(
            df,
            formula="y ~ 1 + C(unit) + C(time)",
            unit_variable_name="unit",
            time_variable_name="time",
            treated_variable_name="treated",
            model=LinearRegression(),
        )


def test_staggered_did_custom_never_treated_value():
    """Test using custom never_treated_value parameter."""
    df = pd.DataFrame(
        {
            "unit": [0, 0, 0, 0, 1, 1, 1, 1],
            "time": [0, 1, 2, 3, 0, 1, 2, 3],
            "treated": [0, 0, 1, 1, 0, 0, 0, 0],
            # Use -1 as never-treated value instead of inf
            "treatment_time": [2, 2, 2, 2, -1, -1, -1, -1],
            "y": [1, 2, 5, 6, 1, 2, 2, 3],
        }
    )

    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treatment_time_variable_name="treatment_time",
        never_treated_value=-1,
        model=LinearRegression(),
    )

    # Check never-treated unit is correctly identified
    assert result.data_.loc[result.data_["unit"] == 1, "G"].iloc[0] == -1
    # Never-treated should have NaN event_time
    assert all(pd.isna(result.data_.loc[result.data_["unit"] == 1, "event_time"]))


def test_staggered_did_does_not_modify_original_data():
    """Test that original data is not modified."""
    df = generate_staggered_did_data(
        n_units=20,
        n_time_periods=10,
        treatment_cohorts={5: 10},
        seed=42,
    )

    # Store original columns
    original_columns = set(df.columns)
    original_shape = df.shape

    _ = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        model=LinearRegression(),
    )

    # Original data should be unchanged
    assert set(df.columns) == original_columns
    assert df.shape == original_shape
    assert "G" not in df.columns
    assert "event_time" not in df.columns


def test_staggered_did_cohorts_attribute():
    """Test that cohorts attribute is correctly populated."""
    df = generate_staggered_did_data(
        n_units=30,
        n_time_periods=15,
        treatment_cohorts={5: 10, 10: 10, 15: 5},
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

    # Check cohorts attribute
    assert hasattr(result, "cohorts")
    assert isinstance(result.cohorts, list)
    # Should have 3 cohorts (not including never-treated)
    assert 5 in result.cohorts
    assert 10 in result.cohorts
    # Cohorts should be sorted
    assert result.cohorts == sorted(result.cohorts)


def test_staggered_did_labels_attribute():
    """Test that labels attribute is correctly populated."""
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

    # Check labels attribute
    assert hasattr(result, "labels")
    assert isinstance(result.labels, list)
    assert len(result.labels) > 0
    # Should include intercept
    assert "Intercept" in result.labels


def test_staggered_did_ols_att_std_column():
    """Test that OLS ATT table includes standard deviation column."""
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
        model=LinearRegression(),
    )

    # Check OLS-specific columns
    assert "att_std" in result.att_event_time_.columns
    assert "n_obs" in result.att_event_time_.columns


def test_staggered_did_dynamic_effects_recovery():
    """Test recovery of dynamic (time-varying) treatment effects."""
    # Generate data with dynamic effects that increase over time
    dynamic_effects = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}

    df = generate_staggered_did_data(
        n_units=50,
        n_time_periods=20,
        treatment_cohorts={5: 20, 10: 20},
        treatment_effects=dynamic_effects,
        sigma=0.1,  # Low noise
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

    # Check that effects are increasing (qualitatively correct pattern)
    att_et = result.att_event_time_
    post_treatment = att_et[att_et["event_time"] >= 0].sort_values("event_time")

    if len(post_treatment) >= 3:
        # Later event-times should have higher effects on average
        early_effect = post_treatment.iloc[0]["att"]
        late_effect = post_treatment.iloc[-1]["att"]
        assert late_effect > early_effect, (
            "Dynamic effects should show increasing pattern"
        )


def test_staggered_did_plot_elements_ols():
    """Test that OLS plot contains expected elements."""
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
        model=LinearRegression(),
    )

    fig, axes = result.plot()

    # Check plot structure
    assert len(axes) == 1
    ax = axes[0]

    # Check axis labels
    assert "Event Time" in ax.get_xlabel()
    assert "Effect Estimate" in ax.get_ylabel()

    # Check title
    assert "Staggered DiD" in ax.get_title()

    # Check legend exists
    assert ax.get_legend() is not None

    plt.close(fig)


@pytest.mark.integration
def test_staggered_did_plot_elements_bayesian(mock_pymc_sample):
    """Test that Bayesian plot contains expected elements."""
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
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    fig, axes = result.plot()

    # Check plot structure
    assert len(axes) == 1
    ax = axes[0]

    # Check axis labels
    assert "Event Time" in ax.get_xlabel()
    assert "Effect Estimate" in ax.get_ylabel()

    plt.close(fig)


def test_staggered_did_n_obs_column():
    """Test that n_obs column correctly counts treated observations per event-time."""
    # Create small controlled dataset
    df = pd.DataFrame(
        {
            "unit": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "time": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            # Units 0 and 1 treated at time 2, units 2 and 3 never treated
            "treated": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "treatment_time": [
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
            ],
            "y": [1, 2, 5, 6, 1, 2, 5, 6, 1, 2, 2, 3, 1, 2, 2, 3],
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

    # Check n_obs for each event time
    # event_time 0: 2 units (0 and 1) at time 2
    # event_time 1: 2 units (0 and 1) at time 3
    att_et = result.att_event_time_
    e0_obs = att_et.loc[att_et["event_time"] == 0, "n_obs"].values[0]
    e1_obs = att_et.loc[att_et["event_time"] == 1, "n_obs"].values[0]

    assert e0_obs == 2, f"Expected 2 observations at event_time 0, got {e0_obs}"
    assert e1_obs == 2, f"Expected 2 observations at event_time 1, got {e1_obs}"


@pytest.mark.integration
def test_staggered_did_bayesian_uncertainty_reasonable(mock_pymc_sample):
    """Test that Bayesian uncertainty intervals are reasonable."""
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
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    att_et = result.att_event_time_

    # Check that all rows have uncertainty bounds
    assert all(att_et["att_lower"].notna())
    assert all(att_et["att_upper"].notna())

    # Lower bound should be less than upper bound
    assert all(att_et["att_lower"] <= att_et["att_upper"])

    # Point estimate should be within bounds
    assert all(att_et["att"] >= att_et["att_lower"])
    assert all(att_et["att"] <= att_et["att_upper"])


def test_staggered_did_reference_event_time_param():
    """Test that reference_event_time parameter is stored correctly."""
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
        reference_event_time=-2,
        model=LinearRegression(),
    )

    assert result.reference_event_time == -2


def test_staggered_did_expt_type():
    """Test that experiment type is correctly set."""
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

    assert result.expt_type == "Staggered Difference in Differences"


def test_staggered_did_outcome_variable_name():
    """Test that outcome variable name is correctly extracted from formula."""
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

    assert result.outcome_variable_name == "y"


def test_staggered_did_x_full_shape():
    """Test that design matrix has correct shape."""
    n_units = 20
    n_time = 10
    df = generate_staggered_did_data(
        n_units=n_units,
        n_time_periods=n_time,
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

    # Full design matrix should have n_units * n_time rows
    assert result.X_full.shape[0] == n_units * n_time
    # Columns: intercept + (n_units - 1) unit dummies + (n_time - 1) time dummies
    expected_cols = 1 + (n_units - 1) + (n_time - 1)
    assert result.X_full.shape[1] == expected_cols


def test_staggered_did_training_data_shape():
    """Test that training data only contains untreated observations."""
    df = generate_staggered_did_data(
        n_units=30,
        n_time_periods=15,
        treatment_cohorts={5: 10, 10: 10},  # 10 never-treated
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

    # Training data should be subset of full data
    assert result.X_train.shape[0] < result.X_full.shape[0]
    assert result.X_train.shape[1] == result.X_full.shape[1]

    # Number of training observations should match _is_untreated
    expected_train = result.data_["_is_untreated"].sum()
    assert result.X_train.shape[0] == expected_train


@pytest.mark.integration
def test_staggered_did_get_plot_data_bayesian(mock_pymc_sample):
    """Test get_plot_data_bayesian method."""
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
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    plot_data = result.get_plot_data_bayesian()

    assert isinstance(plot_data, pd.DataFrame)
    assert "event_time" in plot_data.columns
    assert "att" in plot_data.columns
    assert "att_lower" in plot_data.columns
    assert "att_upper" in plot_data.columns


@pytest.mark.integration
def test_staggered_did_get_plot_data_bayesian_hdi_prob_respected(mock_pymc_sample):
    """Test that get_plot_data_bayesian respects the hdi_prob parameter.

    This verifies the fix for the bug where hdi_prob was accepted but ignored,
    always returning pre-computed 94% intervals.
    """
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
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Get intervals with default 94% HDI
    plot_data_94 = result.get_plot_data_bayesian(hdi_prob=0.94)

    # Get intervals with narrower 80% HDI
    plot_data_80 = result.get_plot_data_bayesian(hdi_prob=0.80)

    # Get intervals with wider 99% HDI
    plot_data_99 = result.get_plot_data_bayesian(hdi_prob=0.99)

    # Verify structure is correct for all
    for df_plot in [plot_data_94, plot_data_80, plot_data_99]:
        assert "att_lower" in df_plot.columns
        assert "att_upper" in df_plot.columns

    # Point estimates should be the same regardless of hdi_prob
    assert np.allclose(plot_data_94["att"].values, plot_data_80["att"].values)
    assert np.allclose(plot_data_94["att"].values, plot_data_99["att"].values)

    # Narrower HDI (80%) should have smaller intervals than 94%
    # interval_width = upper - lower
    width_94 = plot_data_94["att_upper"] - plot_data_94["att_lower"]
    width_80 = plot_data_80["att_upper"] - plot_data_80["att_lower"]
    width_99 = plot_data_99["att_upper"] - plot_data_99["att_lower"]

    # 80% intervals should be narrower than 94%
    assert all(width_80 <= width_94), (
        "80% HDI intervals should be narrower than 94% HDI intervals"
    )

    # 99% intervals should be wider than 94%
    assert all(width_99 >= width_94), (
        "99% HDI intervals should be wider than 94% HDI intervals"
    )


def test_staggered_did_get_plot_data_ols():
    """Test get_plot_data_ols method."""
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
        model=LinearRegression(),
    )

    plot_data = result.get_plot_data_ols()

    assert isinstance(plot_data, pd.DataFrame)
    assert "event_time" in plot_data.columns
    assert "att" in plot_data.columns
    assert "att_std" in plot_data.columns


def test_staggered_did_only_never_treated_as_controls():
    """Test with only never-treated units as controls (no pre-treatment periods)."""
    # All treated units are treated from time 0
    df = pd.DataFrame(
        {
            "unit": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            # Units 0, 1 treated from start, units 2, 3 never treated
            "treated": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "treatment_time": [
                0,
                0,
                0,
                0,
                0,
                0,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
            ],
            "y": [5, 6, 7, 5, 6, 7, 1, 2, 3, 1, 2, 3],
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

    # Training set should only contain never-treated observations
    training_mask = result.data_["_is_untreated"]
    training_units = result.data_.loc[training_mask, "unit"].unique()

    # Only units 2 and 3 should be in training set
    assert set(training_units) == {2, 3}


def test_staggered_did_supports_ols_bayes_attrs():
    """Test that supports_ols and supports_bayes are correctly set."""
    assert cp.StaggeredDifferenceInDifferences.supports_ols is True
    assert cp.StaggeredDifferenceInDifferences.supports_bayes is True


def test_staggered_did_skip_absorbing_validation_when_using_treatment_time():
    """Test that absorbing treatment validation is skipped when using treatment_time_variable_name."""
    # This covers line 215 - early return in _validate_absorbing_treatment
    # when treated_variable_name is not in data columns
    df = pd.DataFrame(
        {
            "unit": [0, 0, 0, 1, 1, 1],
            "time": [0, 1, 2, 0, 1, 2],
            # No 'treated' column - only treatment_time
            "treatment_time": [2, 2, 2, np.inf, np.inf, np.inf],
            "y": [1, 2, 5, 1, 2, 3],
        }
    )

    # Should not raise even though there's no treated column
    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treatment_time_variable_name="treatment_time",
        model=LinearRegression(),
    )

    assert result.data_.loc[result.data_["unit"] == 0, "G"].iloc[0] == 2


def test_staggered_did_unrecognized_model_type_fit():
    """Test that unrecognized model type raises ValueError during fit."""

    class UnknownModel:
        """A model type that is neither PyMCModel nor RegressorMixin."""

        pass

    df = generate_staggered_did_data(
        n_units=20,
        n_time_periods=10,
        treatment_cohorts={5: 10},
        seed=42,
    )

    with pytest.raises(ValueError, match="Model type not recognized"):
        cp.StaggeredDifferenceInDifferences(
            df,
            formula="y ~ 1 + C(unit) + C(time)",
            unit_variable_name="unit",
            time_variable_name="time",
            treated_variable_name="treated",
            model=UnknownModel(),  # type: ignore[arg-type]
        )


def test_staggered_did_single_cohort():
    """Test with a single treatment cohort."""
    df = generate_staggered_did_data(
        n_units=20,
        n_time_periods=15,
        treatment_cohorts={7: 10},  # Single cohort
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

    # Should have exactly one cohort
    assert len(result.cohorts) == 1
    assert result.cohorts[0] == 7


def test_staggered_did_many_cohorts():
    """Test with many treatment cohorts."""
    df = generate_staggered_did_data(
        n_units=60,
        n_time_periods=20,
        treatment_cohorts={3: 10, 6: 10, 9: 10, 12: 10, 15: 10},  # 5 cohorts
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

    # Should have 5 cohorts
    assert len(result.cohorts) == 5
    assert result.cohorts == [3, 6, 9, 12, 15]


def test_staggered_did_late_treatment():
    """Test with treatment occurring late in the panel."""
    df = generate_staggered_did_data(
        n_units=30,
        n_time_periods=20,
        treatment_cohorts={18: 15},  # Treatment very late
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

    # Should have few post-treatment event times
    att_et = result.att_event_time_
    assert len(att_et[att_et["event_time"] >= 0]) <= 2  # Only event times 0 and 1


def test_staggered_did_early_treatment():
    """Test with treatment occurring early in the panel."""
    df = generate_staggered_did_data(
        n_units=30,
        n_time_periods=20,
        treatment_cohorts={2: 15},  # Treatment very early
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

    # Should have many post-treatment event times
    att_et = result.att_event_time_
    assert len(att_et[att_et["event_time"] >= 0]) >= 10


def test_staggered_did_event_window_restricts_negative():
    """Test that event_window correctly restricts pre-treatment event times."""
    df = generate_staggered_did_data(
        n_units=30,
        n_time_periods=20,
        treatment_cohorts={10: 15},
        seed=42,
    )

    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        event_window=(-3, 5),  # Restrict pre-treatment to -3
        model=LinearRegression(),
    )

    att_et = result.att_event_time_
    # No event times should be less than -3
    assert all(att_et["event_time"] >= -3)
    # No event times should be greater than 5
    assert all(att_et["event_time"] <= 5)


@pytest.mark.integration
def test_staggered_did_group_time_att_bayesian(mock_pymc_sample):
    """Test group-time ATT structure for Bayesian model."""
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
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Check Bayesian group-time ATT structure
    att_gt = result.att_group_time_
    assert isinstance(att_gt, pd.DataFrame)
    assert "cohort" in att_gt.columns
    assert "time" in att_gt.columns
    assert "att" in att_gt.columns
    assert "att_lower" in att_gt.columns
    assert "att_upper" in att_gt.columns


def test_staggered_did_plot_only_post_treatment():
    """Test plot when event_window excludes pre-treatment (no shading)."""
    df = generate_staggered_did_data(
        n_units=30,
        n_time_periods=20,
        treatment_cohorts={5: 15},
        seed=42,
    )

    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        event_window=(0, 10),  # Only post-treatment event times
        model=LinearRegression(),
    )

    # Plot should work even without pre-treatment period
    fig, axes = result.plot()
    assert len(axes) == 1

    # Check that all event times are >= 0
    att_et = result.att_event_time_
    assert all(att_et["event_time"] >= 0)

    plt.close(fig)


@pytest.mark.integration
def test_staggered_did_plot_only_post_treatment_bayesian(mock_pymc_sample):
    """Test Bayesian plot when event_window excludes pre-treatment."""
    df = generate_staggered_did_data(
        n_units=30,
        n_time_periods=20,
        treatment_cohorts={5: 15},
        seed=42,
    )

    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        event_window=(0, 10),  # Only post-treatment event times
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Plot should work even without pre-treatment period
    fig, axes = result.plot()
    assert len(axes) == 1

    plt.close(fig)


def test_staggered_did_sklearn_model_without_fit_intercept():
    """Test with sklearn model that doesn't have fit_intercept attribute."""
    from sklearn.neighbors import KNeighborsRegressor

    df = generate_staggered_did_data(
        n_units=20,
        n_time_periods=10,
        treatment_cohorts={5: 10},
        seed=42,
    )

    # KNeighborsRegressor doesn't have fit_intercept
    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        model=KNeighborsRegressor(n_neighbors=3),
    )

    # Should still work
    assert hasattr(result, "att_event_time_")
    assert len(result.att_event_time_) > 0


def test_staggered_did_att_event_time_includes_pre_and_post_treatment():
    """Test that att_event_time_ includes both pre and post-treatment event times.

    This verifies the design: ATT estimates are computed for both:
    - Post-treatment periods (event_time >= 0): actual treatment effects
    - Pre-treatment periods (event_time < 0): placebo check for parallel trends
    """
    df = generate_staggered_did_data(
        n_units=30,
        n_time_periods=15,
        treatment_cohorts={10: 15},  # Late treatment = many pre-treatment periods
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

    att_et = result.att_event_time_

    # ATT table should include both pre-treatment (placebo) and post-treatment effects
    pre_treatment_et = att_et[att_et["event_time"] < 0]
    post_treatment_et = att_et[att_et["event_time"] >= 0]

    assert len(pre_treatment_et) > 0, (
        "ATT table should include pre-treatment event times for placebo check"
    )
    assert len(post_treatment_et) > 0, (
        "ATT table should include post-treatment event times"
    )

    # Pre-treatment effects should be close to zero (placebo check)
    # Since data is simulated with true treatment effect only after treatment
    avg_pre_effect = pre_treatment_et["att"].mean()
    assert abs(avg_pre_effect) < 1.0, (
        f"Pre-treatment placebo effects should be close to zero, got {avg_pre_effect}"
    )

    # Plot should work with pre-treatment shading
    fig, axes = result.plot()
    assert len(axes) == 1
    plt.close(fig)
