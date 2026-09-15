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

"""Tests for Event Study experiment class."""

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.custom_exceptions import DataException, FormulaException
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
    """Test that treatment time is correctly set."""
    df = generate_event_study_data(n_units=20, n_time=15, treatment_time=10, seed=42)
    treated_df = df[df["treated"] == 1]
    assert (treated_df["treat_time"] == 10).all()

    control_df = df[df["treated"] == 0]
    assert control_df["treat_time"].isna().all()


def test_generate_event_study_data_treatment_effect():
    """Test that treatment effect is applied correctly."""
    # Generate data with zero treatment effect
    df_no_effect = generate_event_study_data(
        n_units=100,
        n_time=20,
        treatment_time=10,
        treatment_effects={0: 0.0, 1: 0.0, 2: 0.0},
        seed=42,
    )
    # Generate data with positive treatment effect
    df_with_effect = generate_event_study_data(
        n_units=100,
        n_time=20,
        treatment_time=10,
        treatment_effects={0: 5.0, 1: 5.0, 2: 5.0},
        seed=42,
    )

    # Post-treatment mean should be higher with treatment effect
    treated_post_no_effect = df_no_effect[
        (df_no_effect["treated"] == 1) & (df_no_effect["time"] >= 10)
    ]["y"].mean()
    treated_post_with_effect = df_with_effect[
        (df_with_effect["treated"] == 1) & (df_with_effect["time"] >= 10)
    ]["y"].mean()

    assert treated_post_with_effect > treated_post_no_effect


# ============================================================================
# Unit Tests for EventStudy Input Validation
# ============================================================================


def test_event_study_missing_formula():
    """Test that EventStudy raises error when formula is missing."""
    df = generate_event_study_data(n_units=20, n_time=15, seed=42)

    with pytest.raises(FormulaException, match="Formula must be provided"):
        cp.EventStudy(
            df,
            formula="",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )


def test_event_study_missing_column():
    """Test that EventStudy raises error when required column is missing."""
    df = generate_event_study_data(n_units=20, n_time=15, seed=42)

    with pytest.raises(DataException, match="Required column .* not found"):
        cp.EventStudy(
            df,
            formula="y ~ C(unit) + C(time)",
            unit_col="nonexistent_col",
            time_col="time",
            treat_time_col="treat_time",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )


def test_event_study_invalid_event_window():
    """Test that EventStudy raises error for invalid event window."""
    df = generate_event_study_data(n_units=20, n_time=15, seed=42)

    with pytest.raises(DataException, match="event_window\\[0\\] .* must be less than"):
        cp.EventStudy(
            df,
            formula="y ~ C(unit) + C(time)",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            event_window=(5, 3),  # Invalid: start > end
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )


def test_event_study_reference_outside_window():
    """Test that EventStudy raises error when reference is outside window."""
    df = generate_event_study_data(n_units=20, n_time=15, seed=42)

    with pytest.raises(
        DataException, match="reference_event_time .* must be within event_window"
    ):
        cp.EventStudy(
            df,
            formula="y ~ C(unit) + C(time)",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            event_window=(-3, 3),
            reference_event_time=-5,  # Outside window
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )


def test_event_study_duplicate_observations():
    """Test that EventStudy raises error for duplicate unit-time observations."""
    df = generate_event_study_data(n_units=20, n_time=15, seed=42)
    # Add a duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(DataException, match="duplicate unit-time observations"):
        cp.EventStudy(
            df,
            formula="y ~ C(unit) + C(time)",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )


def test_event_study_staggered_adoption():
    """Test that EventStudy raises error for staggered adoption."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)
    # Manually create staggered adoption by changing some treatment times
    treated_units = df[df["treated"] == 1]["unit"].unique()
    df.loc[df["unit"] == treated_units[0], "treat_time"] = 12  # Different time

    with pytest.raises(
        DataException, match="All treated units must have the same treatment time"
    ):
        cp.EventStudy(
            df,
            formula="y ~ C(unit) + C(time)",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )


# ============================================================================
# Unit Tests for NaN Handling
# ============================================================================


@pytest.mark.parametrize("model_type", ["pymc", "sklearn"])
@pytest.mark.parametrize("nan_location", ["outcome", "unit", "multiple"])
def test_event_study_handles_nan_values(model_type, nan_location):
    """Test that EventStudy handles NaN values by filtering rows."""
    df = generate_event_study_data(n_units=20, n_time=15, seed=42)

    # Inject NaN values based on nan_location
    # Note: We avoid setting 'time' to NaN because it would create duplicates
    # in the (unit, time) key, which EventStudy validates against before patsy filtering
    if nan_location == "outcome":
        df.loc[5:9, "y"] = np.nan  # 5 rows with NaN in outcome
    elif nan_location == "unit":
        df.loc[10:14, "unit"] = np.nan  # 5 rows with NaN in unit
    else:  # multiple
        df.loc[5, "y"] = np.nan
        df.loc[12, "unit"] = np.nan  # 2 rows total

    original_rows = len(df)

    # Choose model based on model_type
    if model_type == "pymc":
        model = cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs)
    else:
        model = LinearRegression()

    # Should not raise ValueError about shape mismatch
    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-5, 5),
        reference_event_time=-1,
        model=model,
    )

    # Verify that NaN rows were filtered
    if nan_location == "outcome" or nan_location == "unit":
        expected_rows = original_rows - 5
    else:  # multiple
        expected_rows = original_rows - 2

    # Check that arrays have consistent shapes
    assert result.X.shape[0] == expected_rows
    assert result.y.shape[0] == expected_rows
    assert len(result.data) == expected_rows  # Filtered data

    # Check that model was successfully fit
    assert hasattr(result, "event_time_coeffs")
    assert len(result.event_time_coeffs) > 0


# ============================================================================
# Integration Tests for EventStudy with PyMC
# ============================================================================


def test_event_study_pymc_basic():
    """Test basic EventStudy functionality with PyMC model."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-5, 5),
        reference_event_time=-1,
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={**sample_kwargs, "random_seed": 42}
        ),
    )

    # Check that event time coefficients were extracted
    assert hasattr(result, "event_time_coeffs")
    assert len(result.event_time_coeffs) == 11  # -5 to 5 inclusive

    # Check that reference event time is zero
    assert result.event_time_coeffs[-1] == 0.0

    # Check that model was fit
    assert result.model.idata is not None


def test_event_study_pymc_summary():
    """Test EventStudy summary method with PyMC model."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={**sample_kwargs, "random_seed": 42}
        ),
    )

    # Summary should not raise
    result.summary()

    # get_event_time_summary should return DataFrame
    summary_df = result.get_event_time_summary()
    assert isinstance(summary_df, pd.DataFrame)
    assert "event_time" in summary_df.columns
    assert "mean" in summary_df.columns
    assert "std" in summary_df.columns
    assert "is_reference" in summary_df.columns


def test_event_study_pymc_plot():
    """Test EventStudy plotting with PyMC model."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={**sample_kwargs, "random_seed": 42}
        ),
    )

    # Plot should not raise
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_event_study_pymc_get_plot_data():
    """Test get_plot_data_bayesian method."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={**sample_kwargs, "random_seed": 42}
        ),
    )

    plot_data = result.get_plot_data_bayesian()
    assert isinstance(plot_data, pd.DataFrame)
    assert "event_time" in plot_data.columns
    assert "mean" in plot_data.columns


# ============================================================================
# Integration Tests for EventStudy with scikit-learn
# ============================================================================


def test_event_study_skl_basic():
    """Test basic EventStudy functionality with sklearn model."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-5, 5),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    # Check that event time coefficients were extracted
    assert hasattr(result, "event_time_coeffs")
    assert len(result.event_time_coeffs) == 11  # -5 to 5 inclusive

    # Check that reference event time is zero
    assert result.event_time_coeffs[-1] == 0.0


def test_event_study_skl_summary():
    """Test EventStudy summary method with sklearn model."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
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


def test_event_study_skl_plot():
    """Test EventStudy plotting with sklearn model."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    # Plot should not raise
    fig, ax = result.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_event_study_skl_get_plot_data():
    """Test get_plot_data_ols method."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    plot_data = result.get_plot_data_ols()
    assert isinstance(plot_data, pd.DataFrame)
    assert "event_time" in plot_data.columns
    assert "mean" in plot_data.columns


# ============================================================================
# Tests for Different Event Windows and Reference Periods
# ============================================================================


def test_event_study_different_reference():
    """Test EventStudy with different reference event times."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    # Test with reference at -2
    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-2,
        model=LinearRegression(),
    )

    assert result.event_time_coeffs[-2] == 0.0
    assert -1 in result.event_time_coeffs  # -1 should have a coefficient


def test_event_study_asymmetric_window():
    """Test EventStudy with asymmetric event window."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-2, 5),  # Asymmetric
        reference_event_time=-1,
        model=LinearRegression(),
    )

    # Should have coefficients from -2 to 5
    assert len(result.event_time_coeffs) == 8
    assert min(result.event_time_coeffs.keys()) == -2
    assert max(result.event_time_coeffs.keys()) == 5


# ============================================================================
# Tests for Control Units
# ============================================================================


def test_event_study_with_never_treated():
    """Test EventStudy with never-treated control units."""
    df = generate_event_study_data(
        n_units=30, n_time=20, treatment_time=10, treated_fraction=0.5, seed=42
    )

    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    # Should work without error
    assert hasattr(result, "event_time_coeffs")
    assert len(result.event_time_coeffs) == 7


# ============================================================================
# Tests for effect_summary
# ============================================================================


def test_event_study_effect_summary_pymc():
    """Test EventStudy effect_summary method with PyMC model."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={**sample_kwargs, "random_seed": 42}
        ),
    )

    # Test effect_summary returns EffectSummary with table and text
    effect = result.effect_summary()
    assert isinstance(effect.table, pd.DataFrame)
    assert isinstance(effect.text, str)
    assert "event_time" in effect.table.columns
    assert "mean" in effect.table.columns

    # Check prose mentions key elements
    assert "Event study" in effect.text
    assert "k=" in effect.text


def test_event_study_effect_summary_skl():
    """Test EventStudy effect_summary method with sklearn model."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    # Test effect_summary returns EffectSummary with table and text
    effect = result.effect_summary()
    assert isinstance(effect.table, pd.DataFrame)
    assert isinstance(effect.text, str)
    assert "event_time" in effect.table.columns
    assert "mean" in effect.table.columns

    # Check prose mentions key elements
    assert "Event study" in effect.text
    assert "k=" in effect.text


def test_event_study_get_event_time_summary_rounding():
    """Test that round_to parameter works correctly in get_event_time_summary."""
    # Generate test data
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    # Test with PyMC model
    result_pymc = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={**sample_kwargs, "random_seed": 42}
        ),
    )

    # Test with round_to=3
    summary_rounded = result_pymc.get_event_time_summary(round_to=3)

    # Check that numeric columns have at most 3 decimal places
    for col in ["mean", "std", "hdi_3%", "hdi_97%"]:
        for val in summary_rounded[col]:
            if not pd.isna(val):
                # Convert to string and check decimal places
                val_str = f"{val:.10f}"  # Get full precision string
                if "." in val_str:
                    decimal_part = val_str.split(".")[1].rstrip("0")
                    assert len(decimal_part) <= 3, (
                        f"Value {val} in column {col} has more than 3 decimal places"
                    )

    # Test with round_to=None (full precision)
    summary_full = result_pymc.get_event_time_summary(round_to=None)

    # Values should be different (more precision)
    # Check at least one non-reference value has more precision
    non_ref_rows = summary_full[~summary_full["is_reference"]]
    if len(non_ref_rows) > 0:
        # At least one value should have different precision
        assert not summary_rounded["mean"].equals(
            summary_full["mean"]
        ) or not summary_rounded["std"].equals(summary_full["std"])

    # Test with SKL model
    result_skl = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    # Test with round_to=2
    summary_skl_rounded = result_skl.get_event_time_summary(round_to=2)

    # Check that mean column has at most 2 decimal places
    for val in summary_skl_rounded["mean"]:
        if not pd.isna(val):
            val_str = f"{val:.10f}"
            if "." in val_str:
                decimal_part = val_str.split(".")[1].rstrip("0")
                assert len(decimal_part) <= 2, (
                    f"Value {val} has more than 2 decimal places"
                )

    # Test with round_to=None for SKL
    summary_skl_full = result_skl.get_event_time_summary(round_to=None)

    # Values should potentially be different
    non_ref_rows_skl = summary_skl_full[~summary_skl_full["is_reference"]]
    if len(non_ref_rows_skl) > 0:
        # Check that we get full precision values
        assert summary_skl_full["mean"].dtype == np.float64


# ============================================================================
# Unit Tests for _compute_event_time Method
# ============================================================================


def test_compute_event_time_treatment_at_boundaries():
    """Test when treatment occurs at first or last time period in data."""
    # Test treatment at first time period
    df_first = generate_event_study_data(
        n_units=20, n_time=15, treatment_time=0, seed=42
    )

    result_first = cp.EventStudy(
        df_first,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-2, 5),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    # Should work without errors
    assert hasattr(result_first, "event_time_coeffs")
    # Check that event times were computed correctly for treated units
    treated_df = result_first.data[result_first.data["_event_time"].notna()]
    assert treated_df["_event_time"].min() == 0  # Earliest event time is 0

    # Test treatment at last time period
    df_last = generate_event_study_data(
        n_units=20, n_time=15, treatment_time=14, seed=42
    )

    result_last = cp.EventStudy(
        df_last,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-5, 2),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    # Should work without errors
    assert hasattr(result_last, "event_time_coeffs")
    # Check that event times were computed correctly
    treated_df = result_last.data[result_last.data["_event_time"].notna()]
    assert treated_df["_event_time"].max() == 0  # Latest event time is 0


def test_compute_event_time_observations_outside_window():
    """Verify observations with event times outside window are handled correctly."""
    df = generate_event_study_data(n_units=20, n_time=30, treatment_time=15, seed=42)

    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    # Check that _in_event_window marks correct observations
    # For treated units, only observations within [-3, 3] event time should be marked
    treated_in_window = result.data[
        result.data["_event_time"].notna() & result.data["_in_event_window"]
    ]
    # All observations in window should have event time in [-3, 3]
    assert (treated_in_window["_event_time"] >= -3).all()
    assert (treated_in_window["_event_time"] <= 3).all()

    # Check that observations outside window are NOT marked
    treated_outside_window = result.data[
        result.data["_event_time"].notna() & ~result.data["_in_event_window"]
    ]
    # All observations outside window should have event time outside [-3, 3]
    assert (
        (treated_outside_window["_event_time"] < -3)
        | (treated_outside_window["_event_time"] > 3)
    ).all()

    # But all data should be retained (not filtered out)
    assert len(result.data) > 0
    assert result.data["_event_time"].notna().sum() > 0


# ============================================================================
# Edge Cases for Panel Data Structure (with Warnings)
# ============================================================================


def test_event_study_unbalanced_panel_emits_warning():
    """Test with unbalanced panel AND verify warning is emitted."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    # Create unbalanced panel by randomly dropping 20% of observations from random units
    np.random.seed(42)
    units_to_modify = np.random.choice(df["unit"].unique(), size=5, replace=False)
    indices_to_drop = []
    for unit in units_to_modify:
        unit_indices = df[df["unit"] == unit].index
        n_to_drop = int(len(unit_indices) * 0.2)
        drop_indices = np.random.choice(unit_indices, size=n_to_drop, replace=False)
        indices_to_drop.extend(drop_indices)

    df_unbalanced = df.drop(indices_to_drop)

    # Verify panel is actually unbalanced
    unit_counts = df_unbalanced.groupby("unit").size()
    assert unit_counts.nunique() > 1, "Panel should be unbalanced"

    # Should emit warning about unbalanced panel
    with pytest.warns(UserWarning, match="unbalanced panel"):
        result = cp.EventStudy(
            df_unbalanced,
            formula="y ~ C(unit) + C(time)",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            event_window=(-3, 3),
            reference_event_time=-1,
            model=LinearRegression(),
        )

    # Should still work correctly
    assert hasattr(result, "event_time_coeffs")
    assert len(result.event_time_coeffs) == 7  # -3 to 3 inclusive


def test_event_study_gaps_in_time_periods_emits_warning():
    """Test with non-consecutive time periods AND verify warning is emitted."""
    df = generate_event_study_data(n_units=20, n_time=15, treatment_time=7, seed=42)

    # Keep only specific time values to create gaps
    time_values_to_keep = [0, 1, 2, 5, 6, 7, 10, 11, 12]
    df_gaps = df[df["time"].isin(time_values_to_keep)].copy()

    # Adjust treat_time to match available times (treatment still at time 7)
    # This is already at 7 which is in our list

    # Should emit warning about gaps in time
    with pytest.warns(UserWarning, match="Non-consecutive time periods"):
        result = cp.EventStudy(
            df_gaps,
            formula="y ~ C(unit) + C(time)",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            event_window=(-3, 3),
            reference_event_time=-1,
            model=LinearRegression(),
        )

    # Event times should still be computed correctly based on actual time differences
    assert hasattr(result, "event_time_coeffs")
    # Check that event times are computed correctly (e.g., time 10 - treat_time 7 = event_time 3)
    treated_df = result.data[result.data["_event_time"].notna()]
    time_10_rows = treated_df[treated_df["time"] == 10]
    if len(time_10_rows) > 0:
        assert (time_10_rows["_event_time"] == 3).all()


def test_event_study_single_treated_unit():
    """Test minimal treated group (only 1 treated unit)."""
    # Generate data with only 1 treated unit out of 20
    df = generate_event_study_data(
        n_units=20, n_time=20, treatment_time=10, treated_fraction=1 / 20, seed=42
    )

    # Verify only 1 unit is treated
    n_treated = df[df["treated"] == 1]["unit"].nunique()
    assert n_treated == 1

    # Test with PyMC
    result_pymc = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={**sample_kwargs, "random_seed": 42}
        ),
    )

    # Should work without errors
    assert hasattr(result_pymc, "event_time_coeffs")
    assert len(result_pymc.event_time_coeffs) == 7

    # Test with sklearn
    result_skl = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-3, 3),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    # Should work without errors
    assert hasattr(result_skl, "event_time_coeffs")
    assert len(result_skl.event_time_coeffs) == 7


# ============================================================================
# Edge Cases for Filtering and Empty Results
# ============================================================================


def test_event_study_no_observations_in_window():
    """Test when event window excludes all treated observations."""
    df = generate_event_study_data(n_units=20, n_time=15, treatment_time=10, seed=42)

    # Event window beyond data range - no treated observations will fall in [15, 20]
    # since time only goes up to 14
    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(15, 20),
        reference_event_time=15,
        model=LinearRegression(),
    )

    # Should work, but event-time dummies will be all zeros (no variation)
    assert hasattr(result, "event_time_coeffs")
    # All event time dummies should be zero since no observations fall in window
    # The coefficients might be zero or very small
    # Just verify the model completed without error
    assert result.X is not None
    assert result.y is not None

    # TODO: Consider adding validation to warn when no observations fall in event window


def test_event_study_extreme_nan_patterns_emits_warning():
    """Test with NaN values that filter out >50% of data AND verify warning."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    # Set 60% of outcome values to NaN
    np.random.seed(42)
    n_to_nan = int(len(df) * 0.6)
    indices_to_nan = np.random.choice(df.index, size=n_to_nan, replace=False)
    df.loc[indices_to_nan, "y"] = np.nan

    # Should emit warning about extreme data loss
    with pytest.warns(UserWarning, match="removed.*% of observations"):
        result = cp.EventStudy(
            df,
            formula="y ~ C(unit) + C(time)",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            event_window=(-3, 3),
            reference_event_time=-1,
            model=LinearRegression(),
        )

    # Should still work if enough observations remain
    assert hasattr(result, "event_time_coeffs")
    # Verify significant data loss occurred
    original_size = len(df)
    filtered_size = len(result.data)
    pct_kept = filtered_size / original_size * 100
    assert pct_kept < 50  # More than 50% was filtered out


def test_event_study_nan_in_treat_time_for_treated_unit():
    """Test edge case where treated unit has NaN in treat_time_col."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    # Manually set one treated unit's treat_time to NaN in some rows
    # This creates an inconsistent state (shouldn't happen in valid data)
    treated_units = df[df["treated"] == 1]["unit"].unique()
    if len(treated_units) > 0:
        first_treated = treated_units[0]
        # Set treat_time to NaN for half the rows of this unit
        unit_mask = df["unit"] == first_treated
        unit_indices = df[unit_mask].index
        indices_to_nan = unit_indices[: len(unit_indices) // 2]
        df.loc[indices_to_nan, "treat_time"] = np.nan

        # This should cause the unit to be partially treated as control
        # The behavior depends on how the code handles this edge case
        result = cp.EventStudy(
            df,
            formula="y ~ C(unit) + C(time)",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            event_window=(-3, 3),
            reference_event_time=-1,
            model=LinearRegression(),
        )

        # Document actual behavior: rows with NaN treat_time will have NaN event_time
        # and will not be marked as in_event_window
        assert hasattr(result, "event_time_coeffs")
        # Some rows of the unit should have NaN event_time
        unit_data = result.data[result.data["unit"] == first_treated]
        assert unit_data["_event_time"].isna().any()

        # TODO: Consider adding validation to check for inconsistent treat_time
        # within a unit (some NaN, some not)


# ============================================================================
# Edge Cases for Event Window Boundaries
# ============================================================================


def test_event_study_single_period_window():
    """Test with minimal event window (just k=-1 and k=0)."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    # Test with PyMC
    result_pymc = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-1, 0),
        reference_event_time=-1,
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={**sample_kwargs, "random_seed": 42}
        ),
    )

    # Should work with only one non-reference coefficient (k=0)
    assert hasattr(result_pymc, "event_time_coeffs")
    assert len(result_pymc.event_time_coeffs) == 2  # k=-1 (reference) and k=0
    assert -1 in result_pymc.event_time_coeffs
    assert 0 in result_pymc.event_time_coeffs
    # Reference should be zero
    assert float(result_pymc.event_time_coeffs[-1]) == 0.0

    # Test with sklearn
    result_skl = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-1, 0),
        reference_event_time=-1,
        model=LinearRegression(),
    )

    # Should work with only one non-reference coefficient (k=0)
    assert hasattr(result_skl, "event_time_coeffs")
    assert len(result_skl.event_time_coeffs) == 2
    assert result_skl.event_time_coeffs[-1] == 0.0


def test_event_study_reference_at_min_boundary():
    """Test with reference event time at window minimum."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-5, 5),
        reference_event_time=-5,
        model=LinearRegression(),
    )

    # Coefficient for k=-5 should be zero, all others estimated
    assert hasattr(result, "event_time_coeffs")
    assert len(result.event_time_coeffs) == 11  # -5 to 5 inclusive
    assert result.event_time_coeffs[-5] == 0.0  # Reference is zero
    # All other event times should be present
    for k in range(-4, 6):
        assert k in result.event_time_coeffs


def test_event_study_reference_at_max_boundary():
    """Test with reference event time at window maximum."""
    df = generate_event_study_data(n_units=20, n_time=20, treatment_time=10, seed=42)

    result = cp.EventStudy(
        df,
        formula="y ~ C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-5, 5),
        reference_event_time=5,
        model=LinearRegression(),
    )

    # Coefficient for k=5 should be zero, all others estimated
    assert hasattr(result, "event_time_coeffs")
    assert len(result.event_time_coeffs) == 11  # -5 to 5 inclusive
    assert result.event_time_coeffs[5] == 0.0  # Reference is zero
    # All other event times should be present
    for k in range(-5, 5):
        assert k in result.event_time_coeffs
