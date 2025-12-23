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
        n_units=100, n_time=20, treatment_time=10, treatment_effect=0.0, seed=42
    )
    # Generate data with positive treatment effect
    df_with_effect = generate_event_study_data(
        n_units=100, n_time=20, treatment_time=10, treatment_effect=5.0, seed=42
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
        model=cp.skl_models.LinearRegression(),
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
        model=cp.skl_models.LinearRegression(),
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
        model=cp.skl_models.LinearRegression(),
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
        model=cp.skl_models.LinearRegression(),
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
        model=cp.skl_models.LinearRegression(),
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
        model=cp.skl_models.LinearRegression(),
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
        model=cp.skl_models.LinearRegression(),
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
