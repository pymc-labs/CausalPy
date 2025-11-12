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
Tests for reporting utilities.
"""

import numpy as np
import pandas as pd
import pytest

import causalpy as cp
from causalpy.reporting import EffectSummary

sample_kwargs = {
    "chains": 2,
    "draws": 100,
    "progressbar": False,
    "random_seed": 42,
}


@pytest.mark.integration
def test_effect_summary_basic(mock_pymc_sample):
    """Test basic effect_summary functionality with ITS."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary()

    # Check return type
    assert isinstance(stats, EffectSummary)
    assert hasattr(stats, "table")
    assert hasattr(stats, "text")

    # Check table structure
    assert isinstance(stats.table, pd.DataFrame)
    assert "average" in stats.table.index
    assert "mean" in stats.table.columns
    assert "median" in stats.table.columns
    assert "hdi_lower" in stats.table.columns
    assert "hdi_upper" in stats.table.columns

    # Check text is a string
    assert isinstance(stats.text, str)
    assert len(stats.text) > 0


@pytest.mark.integration
def test_effect_summary_with_cumulative(mock_pymc_sample):
    """Test effect_summary with cumulative effects."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(cumulative=True)

    assert "average" in stats.table.index
    assert "cumulative" in stats.table.index


@pytest.mark.integration
def test_effect_summary_without_cumulative(mock_pymc_sample):
    """Test effect_summary without cumulative effects."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(cumulative=False)

    assert "average" in stats.table.index
    assert "cumulative" not in stats.table.index


@pytest.mark.integration
def test_effect_summary_with_relative(mock_pymc_sample):
    """Test effect_summary with relative effects."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(relative=True)

    assert "relative_mean" in stats.table.columns
    assert "relative_hdi_lower" in stats.table.columns
    assert "relative_hdi_upper" in stats.table.columns


@pytest.mark.integration
def test_effect_summary_direction_increase(mock_pymc_sample):
    """Test effect_summary with direction='increase'."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(direction="increase")

    assert "p_gt_0" in stats.table.columns
    assert 0 <= stats.table.loc["average", "p_gt_0"] <= 1


@pytest.mark.integration
def test_effect_summary_direction_decrease(mock_pymc_sample):
    """Test effect_summary with direction='decrease'."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(direction="decrease")

    assert "p_lt_0" in stats.table.columns
    assert 0 <= stats.table.loc["average", "p_lt_0"] <= 1


@pytest.mark.integration
def test_effect_summary_direction_two_sided(mock_pymc_sample):
    """Test effect_summary with direction='two-sided'."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(direction="two-sided")

    assert "p_two_sided" in stats.table.columns
    assert "prob_of_effect" in stats.table.columns
    assert 0 <= stats.table.loc["average", "p_two_sided"] <= 1
    assert 0 <= stats.table.loc["average", "prob_of_effect"] <= 1


@pytest.mark.integration
def test_effect_summary_window_datetime(mock_pymc_sample):
    """Test effect_summary with datetime window."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Get post-period dates
    post_dates = result.datapost.index
    window_start = post_dates[0]
    window_end = post_dates[len(post_dates) // 2]

    stats = result.effect_summary(window=(window_start, window_end))

    assert isinstance(stats, EffectSummary)
    assert (
        window_start.strftime("%Y-%m-%d") in stats.text
        or str(window_start) in stats.text
    )


@pytest.mark.integration
def test_effect_summary_window_integer(mock_pymc_sample):
    """Test effect_summary with integer index window."""
    # Create data with integer index
    np.random.seed(42)
    n_pre = 50
    n_post = 30
    t_pre = np.arange(n_pre)
    t_post = np.arange(n_pre, n_pre + n_post)

    y_pre = 10 + 0.5 * t_pre + np.random.normal(0, 1, n_pre)
    y_post = 15 + 0.5 * t_post + np.random.normal(0, 1, n_post)

    df = pd.DataFrame(
        {
            "y": np.concatenate([y_pre, y_post]),
            "t": np.concatenate([t_pre, t_post]),
        },
        index=np.concatenate([t_pre, t_post]),
    )

    treatment_time = 50
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Test with tuple window
    stats1 = result.effect_summary(window=(55, 65))
    assert isinstance(stats1, EffectSummary)

    # Test with slice window
    stats2 = result.effect_summary(window=slice(55, 65))
    assert isinstance(stats2, EffectSummary)


@pytest.mark.integration
def test_effect_summary_alpha(mock_pymc_sample):
    """Test effect_summary with custom alpha."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(alpha=0.1)  # 90% HDI

    # Check that HDI is in text (should mention 90%)
    assert "90%" in stats.text


@pytest.mark.integration
def test_effect_summary_rope(mock_pymc_sample):
    """Test effect_summary with ROPE (min_effect)."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(min_effect=1.0)

    assert "p_rope" in stats.table.columns
    assert 0 <= stats.table.loc["average", "p_rope"] <= 1


@pytest.mark.integration
def test_effect_summary_ols_its(mock_pymc_sample):
    """Test effect_summary with OLS model for ITS."""
    from sklearn.linear_model import LinearRegression

    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=LinearRegression(),
    )

    stats = result.effect_summary()

    assert isinstance(stats, EffectSummary)
    assert "average" in stats.table.index
    assert "mean" in stats.table.columns
    assert "ci_lower" in stats.table.columns
    assert "ci_upper" in stats.table.columns
    assert "p_value" in stats.table.columns
    # OLS tables should NOT have posterior metrics
    assert "median" not in stats.table.columns
    assert "hdi_lower" not in stats.table.columns
    assert "hdi_upper" not in stats.table.columns
    assert "p_gt_0" not in stats.table.columns


@pytest.mark.integration
def test_effect_summary_ols_did(mock_pymc_sample):
    """Test effect_summary with OLS model for DiD."""
    from sklearn.linear_model import LinearRegression

    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )

    stats = result.effect_summary()

    assert isinstance(stats, EffectSummary)
    assert "treatment_effect" in stats.table.index
    assert "mean" in stats.table.columns
    assert "ci_lower" in stats.table.columns
    assert "ci_upper" in stats.table.columns
    assert "p_value" in stats.table.columns
    # OLS tables should NOT have posterior metrics
    assert "median" not in stats.table.columns
    assert "hdi_lower" not in stats.table.columns
    assert "hdi_upper" not in stats.table.columns


@pytest.mark.integration
def test_effect_summary_ols_sc(mock_pymc_sample):
    """Test effect_summary with OLS model for Synthetic Control."""
    from sklearn.linear_model import LinearRegression

    df = cp.load_data("sc")
    treatment_time = 70
    result = cp.SyntheticControl(
        df,
        treatment_time,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=LinearRegression(),
    )

    stats = result.effect_summary(treated_unit="actual")

    assert isinstance(stats, EffectSummary)
    assert "average" in stats.table.index
    assert "mean" in stats.table.columns
    assert "ci_lower" in stats.table.columns
    assert "ci_upper" in stats.table.columns
    assert "p_value" in stats.table.columns


@pytest.mark.integration
def test_effect_summary_rd_pymc(mock_pymc_sample):
    """Test effect_summary with Regression Discontinuity (PyMC)."""
    df = cp.load_data("rd")
    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + x + treated + x:treated",
        treatment_threshold=0.5,
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary()

    assert isinstance(stats, EffectSummary)
    assert "discontinuity" in stats.table.index
    assert "mean" in stats.table.columns
    assert "hdi_lower" in stats.table.columns
    assert "hdi_upper" in stats.table.columns


@pytest.mark.integration
def test_effect_summary_rd_ols(mock_pymc_sample):
    """Test effect_summary with Regression Discontinuity (OLS)."""
    from sklearn.linear_model import LinearRegression

    df = cp.load_data("rd")
    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + x + treated + x:treated",
        treatment_threshold=0.5,
        model=LinearRegression(),
    )

    stats = result.effect_summary()

    assert isinstance(stats, EffectSummary)
    assert "discontinuity" in stats.table.index
    assert "mean" in stats.table.columns
    assert "ci_lower" in stats.table.columns
    assert "ci_upper" in stats.table.columns
    assert "p_value" in stats.table.columns
    # OLS tables should NOT have posterior metrics
    assert "median" not in stats.table.columns
    assert "hdi_lower" not in stats.table.columns
    assert "hdi_upper" not in stats.table.columns


@pytest.mark.integration
def test_effect_summary_rkink_pymc(mock_pymc_sample):
    """Test effect_summary with Regression Kink (PyMC)."""
    # Generate data for regression kink analysis
    rng = np.random.default_rng(42)
    kink_point = 0.5
    beta = [1, 0.5, 0, 0.5, 0]  # Parameters for the piecewise function
    N = 100
    x = rng.uniform(-1, 1, N)
    treated = (x >= kink_point).astype(int)
    y = (
        beta[0]
        + beta[1] * x
        + beta[2] * x**2
        + beta[3] * (x - kink_point) * treated
        + beta[4] * (x - kink_point) ** 2 * treated
        + rng.normal(0, 0.1, N)
    )
    df = pd.DataFrame({"x": x, "y": y, "treated": treated})

    result = cp.RegressionKink(
        df,
        formula="y ~ 1 + x + I(x**2) + I((x-0.5)*treated) + I(((x-0.5)**2)*treated)",
        kink_point=kink_point,
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary()

    assert isinstance(stats, EffectSummary)
    assert "gradient_change" in stats.table.index
    assert "mean" in stats.table.columns
    assert "median" in stats.table.columns
    assert "hdi_lower" in stats.table.columns
    assert "hdi_upper" in stats.table.columns
    assert "p_gt_0" in stats.table.columns


@pytest.mark.integration
def test_effect_summary_rkink_directions(mock_pymc_sample):
    """Test effect_summary with Regression Kink with different directions."""
    # Generate data
    rng = np.random.default_rng(42)
    kink_point = 0.5
    beta = [1, 0.5, 0, -0.5, 0]  # Negative gradient change
    N = 100
    x = rng.uniform(-1, 1, N)
    treated = (x >= kink_point).astype(int)
    y = (
        beta[0]
        + beta[1] * x
        + beta[2] * x**2
        + beta[3] * (x - kink_point) * treated
        + beta[4] * (x - kink_point) ** 2 * treated
        + rng.normal(0, 0.1, N)
    )
    df = pd.DataFrame({"x": x, "y": y, "treated": treated})

    result = cp.RegressionKink(
        df,
        formula="y ~ 1 + x + I(x**2) + I((x-0.5)*treated) + I(((x-0.5)**2)*treated)",
        kink_point=kink_point,
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Test increase
    stats_increase = result.effect_summary(direction="increase")
    assert "p_gt_0" in stats_increase.table.columns

    # Test decrease
    stats_decrease = result.effect_summary(direction="decrease")
    assert "p_lt_0" in stats_decrease.table.columns

    # Test two-sided
    stats_two_sided = result.effect_summary(direction="two-sided")
    assert "p_two_sided" in stats_two_sided.table.columns
    assert "prob_of_effect" in stats_two_sided.table.columns


@pytest.mark.integration
def test_effect_summary_rkink_rope(mock_pymc_sample):
    """Test effect_summary with Regression Kink with ROPE."""
    # Generate data
    rng = np.random.default_rng(42)
    kink_point = 0.5
    beta = [1, 0.5, 0, 0.5, 0]
    N = 100
    x = rng.uniform(-1, 1, N)
    treated = (x >= kink_point).astype(int)
    y = (
        beta[0]
        + beta[1] * x
        + beta[2] * x**2
        + beta[3] * (x - kink_point) * treated
        + beta[4] * (x - kink_point) ** 2 * treated
        + rng.normal(0, 0.1, N)
    )
    df = pd.DataFrame({"x": x, "y": y, "treated": treated})

    result = cp.RegressionKink(
        df,
        formula="y ~ 1 + x + I(x**2) + I((x-0.5)*treated) + I(((x-0.5)**2)*treated)",
        kink_point=kink_point,
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(min_effect=0.2)
    assert "p_rope" in stats.table.columns


@pytest.mark.integration
def test_effect_summary_empty_window_error(mock_pymc_sample):
    """Test that effect_summary raises error for empty window."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Create window that doesn't overlap with post-period
    future_date = pd.to_datetime("2100-01-01")
    with pytest.raises(ValueError, match="no time points"):
        result.effect_summary(window=(future_date, future_date + pd.Timedelta(days=1)))


@pytest.mark.integration
def test_effect_summary_hdi_coverage(mock_pymc_sample):
    """Test that HDI intervals are properly ordered."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary()

    # Check HDI ordering
    assert (
        stats.table.loc["average", "hdi_lower"]
        <= stats.table.loc["average", "hdi_upper"]
    )
    if "cumulative" in stats.table.index:
        assert (
            stats.table.loc["cumulative", "hdi_lower"]
            <= stats.table.loc["cumulative", "hdi_upper"]
        )


@pytest.mark.integration
def test_effect_summary_tail_probabilities_match(mock_pymc_sample):
    """Test that tail probabilities match manual calculations."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(direction="increase")

    # Manually calculate P(effect > 0)
    avg_effect = result.post_impact.mean(dim="obs_ind").isel(treated_units=0)
    manual_p_gt_0 = float((avg_effect > 0).mean().values)

    # Should match (within floating point precision)
    assert abs(stats.table.loc["average", "p_gt_0"] - manual_p_gt_0) < 1e-10


@pytest.mark.integration
def test_effect_summary_synthetic_control(mock_pymc_sample):
    """Test effect_summary with Synthetic Control experiment (single treated unit)."""
    df = cp.load_data("sc")
    treatment_time = 70
    result = cp.SyntheticControl(
        df,
        treatment_time,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(treated_unit="actual")

    assert isinstance(stats, EffectSummary)
    assert "average" in stats.table.index
    assert "cumulative" in stats.table.index
    assert "mean" in stats.table.columns
    assert "hdi_lower" in stats.table.columns
    assert "hdi_upper" in stats.table.columns
    assert isinstance(stats.text, str)
    assert len(stats.text) > 0


@pytest.mark.integration
def test_effect_summary_synthetic_control_multi_unit(mock_pymc_sample):
    """Test effect_summary with Synthetic Control experiment (multiple treated units)."""
    # Create multi-unit synthetic control data
    np.random.seed(42)
    n_obs = 60
    n_control = 4
    n_treated = 2

    # Create time index
    time_index = pd.date_range("2020-01-01", periods=n_obs, freq="D")
    treatment_time = time_index[40]

    # Control unit data
    control_data = {}
    for i in range(n_control):
        control_data[f"control_{i}"] = np.random.normal(10, 2, n_obs) + np.sin(
            np.arange(n_obs) * 0.1
        )

    # Treated unit data
    treated_data = {}
    for j in range(n_treated):
        weights = np.random.dirichlet(np.ones(n_control))
        base_signal = sum(
            weights[i] * control_data[f"control_{i}"] for i in range(n_control)
        )
        treatment_effect = np.zeros(n_obs)
        treatment_effect[40:] = np.random.normal(5, 1, n_obs - 40)
        treated_data[f"treated_{j}"] = (
            base_signal + treatment_effect + np.random.normal(0, 0.5, n_obs)
        )

    df = pd.DataFrame({**control_data, **treated_data}, index=time_index)
    control_units = [f"control_{i}" for i in range(n_control)]
    treated_units = [f"treated_{j}" for j in range(n_treated)]

    result = cp.SyntheticControl(
        df,
        treatment_time,
        control_units=control_units,
        treated_units=treated_units,
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )

    # Test with first treated unit
    stats1 = result.effect_summary(treated_unit="treated_0")
    assert isinstance(stats1, EffectSummary)
    assert "average" in stats1.table.index

    # Test with second treated unit
    stats2 = result.effect_summary(treated_unit="treated_1")
    assert isinstance(stats2, EffectSummary)
    assert "average" in stats2.table.index

    # Test without specifying unit (should use first)
    stats3 = result.effect_summary()
    assert isinstance(stats3, EffectSummary)


@pytest.mark.integration
def test_effect_summary_synthetic_control_window(mock_pymc_sample):
    """Test effect_summary with Synthetic Control using window specification."""
    df = cp.load_data("sc")
    treatment_time = 70
    result = cp.SyntheticControl(
        df,
        treatment_time,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )

    # Test with integer window
    post_indices = result.datapost.index
    window_start = post_indices[0]
    window_end = post_indices[10]  # First 11 post-period points

    stats = result.effect_summary(
        window=(window_start, window_end), treated_unit="actual"
    )

    assert isinstance(stats, EffectSummary)
    assert str(window_start) in stats.text or str(int(window_start)) in stats.text


@pytest.mark.integration
def test_effect_summary_did(mock_pymc_sample):
    """Test effect_summary with Difference-in-Differences experiment."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary()

    assert isinstance(stats, EffectSummary)
    assert "treatment_effect" in stats.table.index
    assert "mean" in stats.table.columns
    assert "median" in stats.table.columns
    assert "hdi_lower" in stats.table.columns
    assert "hdi_upper" in stats.table.columns
    assert isinstance(stats.text, str)
    assert len(stats.text) > 0
    # DiD should not have cumulative or relative effects
    assert "cumulative" not in stats.table.index


@pytest.mark.integration
def test_effect_summary_did_direction_increase(mock_pymc_sample):
    """Test effect_summary with DiD and direction='increase'."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(direction="increase")

    assert "p_gt_0" in stats.table.columns
    assert 0 <= stats.table.loc["treatment_effect", "p_gt_0"] <= 1


@pytest.mark.integration
def test_effect_summary_did_direction_decrease(mock_pymc_sample):
    """Test effect_summary with DiD and direction='decrease'."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(direction="decrease")

    assert "p_lt_0" in stats.table.columns
    assert 0 <= stats.table.loc["treatment_effect", "p_lt_0"] <= 1


@pytest.mark.integration
def test_effect_summary_did_direction_two_sided(mock_pymc_sample):
    """Test effect_summary with DiD and direction='two-sided'."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(direction="two-sided")

    assert "p_two_sided" in stats.table.columns
    assert "prob_of_effect" in stats.table.columns
    assert 0 <= stats.table.loc["treatment_effect", "p_two_sided"] <= 1
    assert 0 <= stats.table.loc["treatment_effect", "prob_of_effect"] <= 1


@pytest.mark.integration
def test_effect_summary_did_rope(mock_pymc_sample):
    """Test effect_summary with DiD and ROPE."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(min_effect=1.0)

    assert "p_rope" in stats.table.columns
    assert 0 <= stats.table.loc["treatment_effect", "p_rope"] <= 1


@pytest.mark.integration
def test_effect_summary_did_ols_error(mock_pymc_sample):
    """Test that effect_summary works for DiD with OLS model (OLS is now supported)."""
    from sklearn.linear_model import LinearRegression

    df = cp.load_data("did")
    ols_model = cp.skl_models.create_causalpy_compatible_class(LinearRegression)()
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=ols_model,
    )

    # OLS is now supported for DiD, so this should not raise an error
    stats = result.effect_summary()
    assert isinstance(stats, EffectSummary)
    assert "treatment_effect" in stats.table.index
    assert "mean" in stats.table.columns
    assert "ci_lower" in stats.table.columns
    assert "ci_upper" in stats.table.columns
    assert "p_value" in stats.table.columns


@pytest.mark.integration
def test_effect_summary_did_hdi_coverage(mock_pymc_sample):
    """Test that HDI intervals are properly ordered for DiD."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary()

    # Check HDI ordering
    assert (
        stats.table.loc["treatment_effect", "hdi_lower"]
        <= stats.table.loc["treatment_effect", "hdi_upper"]
    )


# ==============================================================================
# Tests for new helper functions
# ==============================================================================


def test_extract_hdi_bounds_dataset():
    """Test _extract_hdi_bounds with xr.Dataset input."""
    import xarray as xr

    from causalpy.reporting import _extract_hdi_bounds

    # Create a mock HDI result as Dataset
    data = xr.DataArray([1.0, 3.0], dims=["hdi"], coords={"hdi": ["lower", "higher"]})
    hdi_dataset = xr.Dataset({"effect": data})

    lower, upper = _extract_hdi_bounds(hdi_dataset)

    assert lower == 1.0
    assert upper == 3.0


def test_extract_hdi_bounds_dataarray():
    """Test _extract_hdi_bounds with xr.DataArray input."""
    import xarray as xr

    from causalpy.reporting import _extract_hdi_bounds

    # Create a mock HDI result as DataArray
    hdi_dataarray = xr.DataArray(
        [1.0, 3.0], dims=["hdi"], coords={"hdi": ["lower", "higher"]}
    )

    lower, upper = _extract_hdi_bounds(hdi_dataarray)

    assert lower == 1.0
    assert upper == 3.0


def test_compute_tail_probabilities_increase():
    """Test _compute_tail_probabilities with direction='increase'."""
    import xarray as xr

    from causalpy.reporting import _compute_tail_probabilities

    # Create mock effect posterior with 60% positive values
    effect = xr.DataArray([0.5, 1.0, 1.5, -0.5, -1.0])

    result = _compute_tail_probabilities(effect, "increase")

    assert "p_gt_0" in result
    assert result["p_gt_0"] == 0.6  # 3 out of 5 are positive


def test_compute_tail_probabilities_decrease():
    """Test _compute_tail_probabilities with direction='decrease'."""
    import xarray as xr

    from causalpy.reporting import _compute_tail_probabilities

    # Create mock effect posterior with 40% negative values
    effect = xr.DataArray([0.5, 1.0, 1.5, -0.5, -1.0])

    result = _compute_tail_probabilities(effect, "decrease")

    assert "p_lt_0" in result
    assert result["p_lt_0"] == 0.4  # 2 out of 5 are negative


def test_compute_tail_probabilities_two_sided():
    """Test _compute_tail_probabilities with direction='two-sided'."""
    import xarray as xr

    from causalpy.reporting import _compute_tail_probabilities

    # Create mock effect posterior
    effect = xr.DataArray([0.5, 1.0, 1.5, -0.5, -1.0])

    result = _compute_tail_probabilities(effect, "two-sided")

    assert "p_two_sided" in result
    assert "prob_of_effect" in result
    # p_two_sided = 2 * min(0.6, 0.4) = 0.8
    assert abs(result["p_two_sided"] - 0.8) < 1e-10
    assert abs(result["prob_of_effect"] - 0.2) < 1e-10


def test_compute_rope_probability_two_sided():
    """Test _compute_rope_probability with direction='two-sided'."""
    import xarray as xr

    from causalpy.reporting import _compute_rope_probability

    # Create mock effect posterior
    effect = xr.DataArray([0.5, 1.0, 1.5, -0.5, -1.5])

    result = _compute_rope_probability(effect, min_effect=1.0, direction="two-sided")

    # |effect| > 1.0 for 3 values: 1.5, -1.5, (1.0 is not > 1.0)
    assert result == 0.4  # 2 out of 5


def test_compute_rope_probability_one_sided():
    """Test _compute_rope_probability with one-sided direction."""
    import xarray as xr

    from causalpy.reporting import _compute_rope_probability

    # Create mock effect posterior
    effect = xr.DataArray([0.5, 1.0, 1.5, -0.5, -1.5])

    result = _compute_rope_probability(effect, min_effect=1.0, direction="increase")

    # effect > 1.0 for 1 value: 1.5
    assert result == 0.2  # 1 out of 5


def test_compute_rope_probability_decrease():
    """Test _compute_rope_probability with direction='decrease'."""
    import xarray as xr

    from causalpy.reporting import _compute_rope_probability

    # Create mock effect posterior with negative values
    effect = xr.DataArray([0.5, -1.5, -2.5, -0.5, -3.0])

    result = _compute_rope_probability(effect, min_effect=2.0, direction="decrease")

    # effect < -2.0 for 2 values: -2.5, -3.0
    assert result == 0.4  # 2 out of 5


def test_format_number():
    """Test _format_number helper."""
    from causalpy.reporting import _format_number

    assert _format_number(3.14159, decimals=2) == "3.14"
    assert _format_number(3.14159, decimals=3) == "3.142"
    assert _format_number(10.0, decimals=1) == "10.0"
    assert _format_number(0.001, decimals=4) == "0.0010"


def test_select_treated_unit():
    """Test _select_treated_unit helper."""
    import xarray as xr

    from causalpy.reporting import _select_treated_unit

    # Create mock data with multiple treated units
    data = xr.DataArray(
        [[1, 2], [3, 4], [5, 6]],
        dims=["time", "treated_units"],
        coords={"time": [0, 1, 2], "treated_units": ["unit_a", "unit_b"]},
    )

    # Select by name
    result = _select_treated_unit(data, "unit_a")
    # Check values and dims, not exact coordinate structure
    np.testing.assert_array_equal(result.values, np.array([1, 3, 5]))
    assert "time" in result.dims
    assert "treated_units" not in result.dims

    # Select first when None provided
    result = _select_treated_unit(data, None)
    np.testing.assert_array_equal(result.values, np.array([1, 3, 5]))
    assert "time" in result.dims
    assert "treated_units" not in result.dims


def test_select_treated_unit_numpy():
    """Test _select_treated_unit_numpy helper."""
    from causalpy.reporting import _select_treated_unit_numpy

    # Create mock result object
    class MockResult:
        treated_units = ["unit_a", "unit_b", "unit_c"]

    result = MockResult()

    # Create mock 2D numpy array (time x units)
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Select by name
    selected = _select_treated_unit_numpy(data, result, "unit_b")
    np.testing.assert_array_equal(selected, np.array([2, 5, 8]))

    # Select first when None provided
    selected = _select_treated_unit_numpy(data, result, None)
    np.testing.assert_array_equal(selected, np.array([1, 4, 7]))


# ==============================================================================
# Tests for error handling
# ==============================================================================


def test_detect_experiment_type_unknown():
    """Test _detect_experiment_type raises error for unknown experiment type."""
    from causalpy.reporting import _detect_experiment_type

    # Create mock result with no recognized attributes
    class MockResult:
        some_other_attribute = "value"

    result = MockResult()

    with pytest.raises(ValueError, match="Unknown experiment type"):
        _detect_experiment_type(result)


def test_detect_experiment_type_prepostnegd():
    """Test _detect_experiment_type correctly identifies PrePostNEGD (has causal_impact but not post_impact)."""
    from causalpy.reporting import _detect_experiment_type

    # Create mock result like PrePostNEGD
    class MockPrePostNEGD:
        causal_impact = None

    result = MockPrePostNEGD()

    experiment_type = _detect_experiment_type(result)
    assert experiment_type == "did"


def test_extract_window_invalid_type():
    """Test _extract_window raises error for invalid window type."""
    from causalpy.reporting import _extract_window

    # Create a minimal mock result
    class MockResult:
        post_impact = np.array([1, 2, 3])
        datapost = pd.DataFrame({"y": [1, 2, 3]}, index=[0, 1, 2])

    result = MockResult()

    # Invalid window type (not "post", tuple, or slice)
    with pytest.raises(ValueError, match="window must be"):
        _extract_window(result, window=[1, 2, 3])  # list is invalid


@pytest.mark.integration
def test_compute_statistics_did_ols_missing_interaction_term(mock_pymc_sample):
    """Test _compute_statistics_did_ols error when interaction term is not found."""
    from sklearn.linear_model import LinearRegression

    from causalpy.reporting import _compute_statistics_did_ols

    df = cp.load_data("did")

    # Create DiD result
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )

    # Manually corrupt the labels to trigger error
    result.labels = ["Intercept", "some_other_term"]

    with pytest.raises(ValueError, match="Could not find interaction term"):
        _compute_statistics_did_ols(result, alpha=0.05)


@pytest.mark.integration
def test_compute_statistics_rd_ols_fallback_path(mock_pymc_sample):
    """Test _compute_statistics_rd_ols uses fallback when coefficient not found."""
    from sklearn.linear_model import LinearRegression

    from causalpy.reporting import _compute_statistics_rd_ols

    df = cp.load_data("rd")
    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + x + treated + x:treated",
        treatment_threshold=0.5,
        model=LinearRegression(),
    )

    # Manually corrupt the labels to trigger fallback
    original_labels = result.labels
    result.labels = ["Intercept", "x", "some_other_term"]

    # Should not raise error, but use fallback SE calculation
    stats = _compute_statistics_rd_ols(result, alpha=0.05)

    # Restore labels
    result.labels = original_labels

    assert "mean" in stats
    assert "ci_lower" in stats
    assert "ci_upper" in stats
    assert "p_value" in stats


# ==============================================================================
# Tests for edge cases and data handling
# ==============================================================================


def test_select_treated_unit_with_multiple_units():
    """Test _select_treated_unit correctly selects from multiple units."""
    import xarray as xr

    from causalpy.reporting import _select_treated_unit

    # Create data with multiple treated units
    data = xr.DataArray(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dims=["time", "treated_units"],
        coords={"time": [0, 1, 2], "treated_units": ["unit_a", "unit_b", "unit_c"]},
    )

    # Select unit_a
    result = _select_treated_unit(data, "unit_a")
    assert "time" in result.dims
    assert "treated_units" not in result.dims
    np.testing.assert_array_equal(result.values, np.array([1, 4, 7]))

    # Select unit_b
    result = _select_treated_unit(data, "unit_b")
    np.testing.assert_array_equal(result.values, np.array([2, 5, 8]))

    # Select unit_c
    result = _select_treated_unit(data, "unit_c")
    np.testing.assert_array_equal(result.values, np.array([3, 6, 9]))


@pytest.mark.integration
def test_extract_window_slice_with_step(mock_pymc_sample):
    """Test _extract_window with slice having step parameter."""
    # Create data with integer index
    np.random.seed(42)
    n_pre = 50
    n_post = 30
    t_pre = np.arange(n_pre)
    t_post = np.arange(n_pre, n_pre + n_post)

    y_pre = 10 + 0.5 * t_pre + np.random.normal(0, 1, n_pre)
    y_post = 15 + 0.5 * t_post + np.random.normal(0, 1, n_post)

    df = pd.DataFrame(
        {
            "y": np.concatenate([y_pre, y_post]),
            "t": np.concatenate([t_pre, t_post]),
        },
        index=np.concatenate([t_pre, t_post]),
    )

    treatment_time = 50
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Test with slice having step
    stats = result.effect_summary(window=slice(50, 70, 2))  # Every other point
    assert isinstance(stats, EffectSummary)
    # Window should have approximately half the points
    assert len(str(stats.text)) > 0


@pytest.mark.integration
def test_relative_effects_with_near_zero_counterfactual(mock_pymc_sample):
    """Test that relative effects handle division by near-zero counterfactual (epsilon protection)."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics

    # Create mock data with near-zero counterfactual
    impact = xr.DataArray(
        np.random.normal(1.0, 0.1, (2, 10, 5)),
        dims=["chain", "draw", "obs_ind"],
        coords={"chain": [0, 1], "draw": range(10), "obs_ind": range(5)},
    )

    # Counterfactual with values very close to zero
    counterfactual = xr.DataArray(
        np.random.normal(0.0001, 0.00001, (2, 10, 5)),
        dims=["chain", "draw", "obs_ind"],
        coords={"chain": [0, 1], "draw": range(10), "obs_ind": range(5)},
    )

    # Should not raise division by zero error
    stats = _compute_statistics(
        impact,
        counterfactual,
        hdi_prob=0.95,
        direction="increase",
        cumulative=True,
        relative=True,
        min_effect=None,
    )

    # Check that relative statistics were computed
    assert "relative_mean" in stats["avg"]
    assert np.isfinite(stats["avg"]["relative_mean"])


@pytest.mark.integration
def test_extract_counterfactual_dict_format(mock_pymc_sample):
    """Test _extract_counterfactual with dict format PyMC results."""
    from causalpy.reporting import _extract_counterfactual

    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Convert InferenceData to dict format
    post_pred_dict = {"posterior_predictive": result.post_pred.posterior_predictive}
    original_post_pred = result.post_pred
    result.post_pred = post_pred_dict

    # Should handle dict format
    window_coords = result.datapost.index[:10]
    counterfactual = _extract_counterfactual(result, window_coords, treated_unit=None)

    # Restore original
    result.post_pred = original_post_pred

    assert counterfactual is not None
    assert hasattr(counterfactual, "shape")


@pytest.mark.integration
def test_compute_statistics_ols_small_sample(mock_pymc_sample):
    """Test _compute_statistics_ols with small sample size."""
    from causalpy.reporting import _compute_statistics_ols

    # Very small sample
    impact = np.array([1.0, 2.0, 1.5])
    counterfactual = np.array([0.5, 0.6, 0.7])

    stats = _compute_statistics_ols(
        impact,
        counterfactual,
        alpha=0.05,
        cumulative=True,
        relative=True,
    )

    assert "avg" in stats
    assert "cum" in stats
    assert "mean" in stats["avg"]
    assert "ci_lower" in stats["avg"]
    assert "ci_upper" in stats["avg"]
    assert "p_value" in stats["avg"]


def test_generate_table_scalar_all_tail_probabilities():
    """Test _generate_table_scalar includes all tail probability columns."""
    from causalpy.reporting import _generate_table_scalar

    # Stats with all possible tail probability keys
    stats = {
        "mean": 2.5,
        "median": 2.4,
        "hdi_lower": 1.0,
        "hdi_upper": 4.0,
        "p_gt_0": 0.95,
        "p_lt_0": 0.05,
        "p_two_sided": 0.10,
        "prob_of_effect": 0.90,
        "p_rope": 0.85,
    }

    table = _generate_table_scalar(stats, index_name="test_effect")

    assert "p_gt_0" in table.columns
    assert "p_lt_0" in table.columns
    assert "p_two_sided" in table.columns
    assert "prob_of_effect" in table.columns
    assert "p_rope" in table.columns
    assert table.loc["test_effect", "p_gt_0"] == 0.95


# ==============================================================================
# Unit tests for scalar effect helper functions
# ==============================================================================


def test_compute_statistics_scalar_increase():
    """Test _compute_statistics_scalar with direction='increase'."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics_scalar

    # Create mock effect with known properties
    effect = xr.DataArray(
        np.array([[0.5, 1.0, 1.5, 2.0], [0.6, 1.1, 1.6, 2.1]]),
        dims=["chain", "draw"],
        coords={"chain": [0, 1], "draw": [0, 1, 2, 3]},
    )

    stats = _compute_statistics_scalar(
        effect, hdi_prob=0.95, direction="increase", min_effect=None
    )

    assert "mean" in stats
    assert "median" in stats
    assert "hdi_lower" in stats
    assert "hdi_upper" in stats
    assert "p_gt_0" in stats
    assert stats["p_gt_0"] == 1.0  # All values are positive


def test_compute_statistics_scalar_decrease():
    """Test _compute_statistics_scalar with direction='decrease'."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics_scalar

    # Create mock effect with negative values
    effect = xr.DataArray(
        np.array([[-0.5, -1.0, -1.5, -2.0], [-0.6, -1.1, -1.6, -2.1]]),
        dims=["chain", "draw"],
        coords={"chain": [0, 1], "draw": [0, 1, 2, 3]},
    )

    stats = _compute_statistics_scalar(
        effect, hdi_prob=0.95, direction="decrease", min_effect=None
    )

    assert "p_lt_0" in stats
    assert stats["p_lt_0"] == 1.0  # All values are negative


def test_compute_statistics_scalar_two_sided():
    """Test _compute_statistics_scalar with direction='two-sided'."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics_scalar

    # Create mock effect with mixed values (60% positive, 40% negative)
    effect = xr.DataArray(
        np.array([[0.5, 1.0, 1.5, -0.5, -1.0]]),
        dims=["chain", "draw"],
        coords={"chain": [0], "draw": [0, 1, 2, 3, 4]},
    )

    stats = _compute_statistics_scalar(
        effect, hdi_prob=0.95, direction="two-sided", min_effect=None
    )

    assert "p_two_sided" in stats
    assert "prob_of_effect" in stats
    # p_two_sided = 2 * min(0.6, 0.4) = 0.8
    assert abs(stats["p_two_sided"] - 0.8) < 1e-10
    assert abs(stats["prob_of_effect"] - 0.2) < 1e-10


def test_compute_statistics_scalar_with_rope():
    """Test _compute_statistics_scalar with ROPE (min_effect)."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics_scalar

    # Create mock effect
    effect = xr.DataArray(
        np.array([[0.5, 1.0, 1.5, 2.0, 2.5]]),
        dims=["chain", "draw"],
        coords={"chain": [0], "draw": [0, 1, 2, 3, 4]},
    )

    stats = _compute_statistics_scalar(
        effect, hdi_prob=0.95, direction="increase", min_effect=1.2
    )

    assert "p_rope" in stats
    # Values > 1.2 are: 1.5, 2.0, 2.5 = 3 out of 5 = 0.6
    assert stats["p_rope"] == 0.6


def test_compute_statistics_scalar_with_rope_two_sided():
    """Test _compute_statistics_scalar with ROPE and two-sided direction."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics_scalar

    # Create mock effect with both positive and negative values
    effect = xr.DataArray(
        np.array([[0.5, 1.5, -1.5, 2.0, -0.3]]),
        dims=["chain", "draw"],
        coords={"chain": [0], "draw": [0, 1, 2, 3, 4]},
    )

    stats = _compute_statistics_scalar(
        effect, hdi_prob=0.95, direction="two-sided", min_effect=1.0
    )

    assert "p_rope" in stats
    # |effect| > 1.0 for: 1.5, -1.5, 2.0 = 3 out of 5 = 0.6
    assert stats["p_rope"] == 0.6


def test_generate_table_scalar_basic():
    """Test _generate_table_scalar with basic stats."""
    from causalpy.reporting import _generate_table_scalar

    stats = {
        "mean": 2.5,
        "median": 2.4,
        "hdi_lower": 1.0,
        "hdi_upper": 4.0,
        "p_gt_0": 0.95,
    }

    table = _generate_table_scalar(stats, index_name="effect")

    assert "effect" in table.index
    assert table.loc["effect", "mean"] == 2.5
    assert table.loc["effect", "median"] == 2.4
    assert table.loc["effect", "hdi_lower"] == 1.0
    assert table.loc["effect", "hdi_upper"] == 4.0
    assert table.loc["effect", "p_gt_0"] == 0.95


def test_generate_prose_scalar_increase():
    """Test _generate_prose_scalar with direction='increase'."""
    from causalpy.reporting import _generate_prose_scalar

    stats = {
        "mean": 2.5,
        "hdi_lower": 1.0,
        "hdi_upper": 4.0,
        "p_gt_0": 0.95,
    }

    prose = _generate_prose_scalar(
        stats, "average treatment effect", alpha=0.05, direction="increase"
    )

    assert "average treatment effect" in prose
    assert "2.50" in prose
    assert "95% HDI" in prose
    assert "1.00" in prose
    assert "4.00" in prose
    assert "0.950" in prose
    assert "increase" in prose


def test_generate_prose_scalar_decrease():
    """Test _generate_prose_scalar with direction='decrease'."""
    from causalpy.reporting import _generate_prose_scalar

    stats = {
        "mean": -2.5,
        "hdi_lower": -4.0,
        "hdi_upper": -1.0,
        "p_lt_0": 0.98,
    }

    prose = _generate_prose_scalar(
        stats, "treatment effect", alpha=0.05, direction="decrease"
    )

    assert "treatment effect" in prose
    assert "-2.50" in prose
    assert "0.980" in prose
    assert "decrease" in prose


def test_generate_prose_scalar_two_sided():
    """Test _generate_prose_scalar with direction='two-sided'."""
    from causalpy.reporting import _generate_prose_scalar

    stats = {
        "mean": 2.5,
        "hdi_lower": 1.0,
        "hdi_upper": 4.0,
        "prob_of_effect": 0.85,
    }

    prose = _generate_prose_scalar(
        stats, "discontinuity", alpha=0.05, direction="two-sided"
    )

    assert "discontinuity" in prose
    assert "2.50" in prose
    assert "0.850" in prose
    assert "effect" in prose  # "effect" (not "increase" or "decrease")


# ==============================================================================
# Unit tests for time-series helper functions
# ==============================================================================


def test_generate_table_with_all_options():
    """Test _generate_table includes all columns when all stats are present."""
    from causalpy.reporting import _generate_table

    stats = {
        "avg": {
            "mean": 2.5,
            "median": 2.4,
            "hdi_lower": 1.0,
            "hdi_upper": 4.0,
            "p_gt_0": 0.95,
            "p_rope": 0.85,
            "relative_mean": 50.0,
            "relative_hdi_lower": 20.0,
            "relative_hdi_upper": 80.0,
        },
        "cum": {
            "mean": 50.0,
            "median": 49.0,
            "hdi_lower": 30.0,
            "hdi_upper": 70.0,
            "p_gt_0": 0.98,
            "p_rope": 0.90,
            "relative_mean": 100.0,
            "relative_hdi_lower": 60.0,
            "relative_hdi_upper": 140.0,
        },
    }

    table = _generate_table(stats, cumulative=True, relative=True)

    assert "average" in table.index
    assert "cumulative" in table.index
    assert "mean" in table.columns
    assert "relative_mean" in table.columns
    assert "p_gt_0" in table.columns
    assert "p_rope" in table.columns
    assert table.loc["average", "mean"] == 2.5
    assert table.loc["cumulative", "mean"] == 50.0


def test_generate_table_without_cumulative():
    """Test _generate_table excludes cumulative row when cumulative=False."""
    from causalpy.reporting import _generate_table

    stats = {
        "avg": {
            "mean": 2.5,
            "median": 2.4,
            "hdi_lower": 1.0,
            "hdi_upper": 4.0,
            "p_gt_0": 0.95,
        }
    }

    table = _generate_table(stats, cumulative=False, relative=False)

    assert "average" in table.index
    assert "cumulative" not in table.index


def test_generate_table_without_relative():
    """Test _generate_table excludes relative columns when relative=False."""
    from causalpy.reporting import _generate_table

    stats = {
        "avg": {
            "mean": 2.5,
            "median": 2.4,
            "hdi_lower": 1.0,
            "hdi_upper": 4.0,
            "p_gt_0": 0.95,
        }
    }

    table = _generate_table(stats, cumulative=False, relative=False)

    assert "relative_mean" not in table.columns


def test_generate_table_with_two_sided():
    """Test _generate_table includes two-sided probability columns."""
    from causalpy.reporting import _generate_table

    stats = {
        "avg": {
            "mean": 2.5,
            "median": 2.4,
            "hdi_lower": 1.0,
            "hdi_upper": 4.0,
            "p_two_sided": 0.10,
            "prob_of_effect": 0.90,
        }
    }

    table = _generate_table(stats, cumulative=False, relative=False)

    assert "p_two_sided" in table.columns
    assert "prob_of_effect" in table.columns
    assert table.loc["average", "p_two_sided"] == 0.10
    assert table.loc["average", "prob_of_effect"] == 0.90


def test_generate_prose_basic():
    """Test _generate_prose generates proper text."""
    from causalpy.reporting import _generate_prose

    stats = {
        "avg": {
            "mean": 2.5,
            "hdi_lower": 1.0,
            "hdi_upper": 4.0,
            "p_gt_0": 0.95,
        }
    }

    window_coords = pd.Index([10, 11, 12, 13, 14])

    prose = _generate_prose(
        stats,
        window_coords,
        alpha=0.05,
        direction="increase",
        cumulative=False,
        relative=False,
    )

    assert "Post-period" in prose
    assert "10 to 14" in prose
    assert "2.50" in prose
    assert "95% HDI" in prose
    assert "1.00" in prose
    assert "4.00" in prose
    assert "0.950" in prose
    assert "increase" in prose


def test_generate_prose_with_cumulative():
    """Test _generate_prose includes cumulative effect text."""
    from causalpy.reporting import _generate_prose

    stats = {
        "avg": {
            "mean": 2.5,
            "hdi_lower": 1.0,
            "hdi_upper": 4.0,
            "p_gt_0": 0.95,
        },
        "cum": {
            "mean": 50.0,
            "hdi_lower": 30.0,
            "hdi_upper": 70.0,
            "p_gt_0": 0.98,
        },
    }

    window_coords = pd.Index([10, 11, 12])

    prose = _generate_prose(
        stats,
        window_coords,
        alpha=0.05,
        direction="increase",
        cumulative=True,
        relative=False,
    )

    assert "cumulative effect" in prose
    assert "50.00" in prose
    assert "30.00" in prose
    assert "70.00" in prose


def test_generate_prose_with_relative():
    """Test _generate_prose includes relative effect text."""
    from causalpy.reporting import _generate_prose

    stats = {
        "avg": {
            "mean": 2.5,
            "hdi_lower": 1.0,
            "hdi_upper": 4.0,
            "p_gt_0": 0.95,
            "relative_mean": 50.0,
            "relative_hdi_lower": 20.0,
            "relative_hdi_upper": 80.0,
        }
    }

    window_coords = pd.Index([10, 11, 12])

    prose = _generate_prose(
        stats,
        window_coords,
        alpha=0.05,
        direction="increase",
        cumulative=False,
        relative=True,
    )

    assert "Relative to the counterfactual" in prose
    assert "50.00%" in prose
    assert "20.00%" in prose
    assert "80.00%" in prose


def test_generate_table_ols_basic():
    """Test _generate_table_ols with basic OLS stats."""
    from causalpy.reporting import _generate_table_ols

    stats = {
        "avg": {
            "mean": 2.5,
            "ci_lower": 1.0,
            "ci_upper": 4.0,
            "p_value": 0.05,
        }
    }

    table = _generate_table_ols(stats, cumulative=False, relative=False)

    assert "average" in table.index
    assert "mean" in table.columns
    assert "ci_lower" in table.columns
    assert "ci_upper" in table.columns
    assert "p_value" in table.columns
    assert table.loc["average", "mean"] == 2.5


def test_generate_table_ols_with_cumulative():
    """Test _generate_table_ols includes cumulative row."""
    from causalpy.reporting import _generate_table_ols

    stats = {
        "avg": {
            "mean": 2.5,
            "ci_lower": 1.0,
            "ci_upper": 4.0,
            "p_value": 0.05,
        },
        "cum": {
            "mean": 50.0,
            "ci_lower": 30.0,
            "ci_upper": 70.0,
            "p_value": 0.01,
        },
    }

    table = _generate_table_ols(stats, cumulative=True, relative=False)

    assert "average" in table.index
    assert "cumulative" in table.index
    assert table.loc["cumulative", "mean"] == 50.0


def test_generate_table_ols_with_relative():
    """Test _generate_table_ols includes relative columns."""
    from causalpy.reporting import _generate_table_ols

    stats = {
        "avg": {
            "mean": 2.5,
            "ci_lower": 1.0,
            "ci_upper": 4.0,
            "p_value": 0.05,
            "relative_mean": 50.0,
            "relative_ci_lower": 20.0,
            "relative_ci_upper": 80.0,
        }
    }

    table = _generate_table_ols(stats, cumulative=False, relative=True)

    assert "relative_mean" in table.columns
    assert "relative_ci_lower" in table.columns
    assert "relative_ci_upper" in table.columns
    assert table.loc["average", "relative_mean"] == 50.0


def test_generate_prose_ols_basic():
    """Test _generate_prose_ols generates proper text."""
    from causalpy.reporting import _generate_prose_ols

    stats = {
        "avg": {
            "mean": 2.5,
            "ci_lower": 1.0,
            "ci_upper": 4.0,
            "p_value": 0.05,
        }
    }

    window_coords = pd.Index([10, 11, 12])

    prose = _generate_prose_ols(
        stats,
        window_coords,
        alpha=0.05,
        cumulative=False,
        relative=False,
    )

    assert "Post-period" in prose
    assert "10 to 12" in prose
    assert "2.50" in prose
    assert "95% CI" in prose
    assert "1.00" in prose
    assert "4.00" in prose
    assert "p-value of 0.050" in prose


def test_generate_prose_ols_with_cumulative():
    """Test _generate_prose_ols includes cumulative effect text."""
    from causalpy.reporting import _generate_prose_ols

    stats = {
        "avg": {
            "mean": 2.5,
            "ci_lower": 1.0,
            "ci_upper": 4.0,
            "p_value": 0.05,
        },
        "cum": {
            "mean": 50.0,
            "ci_lower": 30.0,
            "ci_upper": 70.0,
            "p_value": 0.01,
        },
    }

    window_coords = pd.Index([10, 11, 12])

    prose = _generate_prose_ols(
        stats,
        window_coords,
        alpha=0.05,
        cumulative=True,
        relative=False,
    )

    assert "cumulative effect" in prose
    assert "50.00" in prose
    assert "p-value 0.010" in prose


# ==============================================================================
# Integration tests for PrePostNEGD experiment
# ==============================================================================


@pytest.mark.integration
def test_effect_summary_prepostnegd_pymc(mock_pymc_sample):
    """Test effect_summary with PrePostNEGD experiment (PyMC)."""
    df = cp.load_data("anova1")
    result = cp.PrePostNEGD(
        df,
        formula="post ~ 1 + C(group) + pre",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary()

    assert isinstance(stats, EffectSummary)
    assert "treatment_effect" in stats.table.index
    assert "mean" in stats.table.columns
    assert "median" in stats.table.columns
    assert "hdi_lower" in stats.table.columns
    assert "hdi_upper" in stats.table.columns
    assert isinstance(stats.text, str)
    assert len(stats.text) > 0
    # PrePostNEGD should not have cumulative or relative effects (like DiD)
    assert "cumulative" not in stats.table.index


@pytest.mark.integration
def test_effect_summary_prepostnegd_directions(mock_pymc_sample):
    """Test effect_summary with PrePostNEGD with different directions."""
    df = cp.load_data("anova1")
    result = cp.PrePostNEGD(
        df,
        formula="post ~ 1 + C(group) + pre",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    # Test increase
    stats_increase = result.effect_summary(direction="increase")
    assert "p_gt_0" in stats_increase.table.columns

    # Test decrease
    stats_decrease = result.effect_summary(direction="decrease")
    assert "p_lt_0" in stats_decrease.table.columns

    # Test two-sided
    stats_two_sided = result.effect_summary(direction="two-sided")
    assert "p_two_sided" in stats_two_sided.table.columns
    assert "prob_of_effect" in stats_two_sided.table.columns


@pytest.mark.integration
def test_effect_summary_prepostnegd_rope(mock_pymc_sample):
    """Test effect_summary with PrePostNEGD with ROPE."""
    df = cp.load_data("anova1")
    result = cp.PrePostNEGD(
        df,
        formula="post ~ 1 + C(group) + pre",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(min_effect=0.5)

    assert "p_rope" in stats.table.columns
    assert 0 <= stats.table.loc["treatment_effect", "p_rope"] <= 1


# ==============================================================================
# Integration tests for additional parameter combinations
# ==============================================================================


@pytest.mark.integration
def test_effect_summary_its_relative_false(mock_pymc_sample):
    """Test effect_summary with ITS and relative=False."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(relative=False)

    assert isinstance(stats, EffectSummary)
    assert "relative_mean" not in stats.table.columns


@pytest.mark.integration
def test_effect_summary_ols_cumulative_false(mock_pymc_sample):
    """Test effect_summary with OLS model and cumulative=False."""
    from sklearn.linear_model import LinearRegression

    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=LinearRegression(),
    )

    stats = result.effect_summary(cumulative=False)

    assert isinstance(stats, EffectSummary)
    assert "cumulative" not in stats.table.index
    assert "average" in stats.table.index


@pytest.mark.integration
def test_effect_summary_ols_relative_false(mock_pymc_sample):
    """Test effect_summary with OLS model and relative=False."""
    from sklearn.linear_model import LinearRegression

    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=LinearRegression(),
    )

    stats = result.effect_summary(relative=False)

    assert isinstance(stats, EffectSummary)
    assert "relative_mean" not in stats.table.columns


@pytest.mark.integration
def test_effect_summary_rope_with_two_sided_its(mock_pymc_sample):
    """Test effect_summary with ROPE and two-sided direction for ITS."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(direction="two-sided", min_effect=1.0)

    assert "p_rope" in stats.table.columns
    assert 0 <= stats.table.loc["average", "p_rope"] <= 1
    if "cumulative" in stats.table.index:
        assert 0 <= stats.table.loc["cumulative", "p_rope"] <= 1


@pytest.mark.integration
def test_effect_summary_rope_with_two_sided_did(mock_pymc_sample):
    """Test effect_summary with ROPE and two-sided direction for DiD."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(direction="two-sided", min_effect=0.5)

    assert "p_rope" in stats.table.columns
    assert 0 <= stats.table.loc["treatment_effect", "p_rope"] <= 1


@pytest.mark.integration
def test_effect_summary_rd_two_sided_with_rope(mock_pymc_sample):
    """Test effect_summary with RD, two-sided direction, and ROPE."""
    df = cp.load_data("rd")
    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + x + treated + x:treated",
        treatment_threshold=0.5,
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(direction="two-sided", min_effect=0.5)

    assert "p_rope" in stats.table.columns
    assert "p_two_sided" in stats.table.columns
    assert "prob_of_effect" in stats.table.columns
    assert 0 <= stats.table.loc["discontinuity", "p_rope"] <= 1


@pytest.mark.integration
def test_effect_summary_sc_cumulative_false(mock_pymc_sample):
    """Test effect_summary with Synthetic Control and cumulative=False."""
    df = cp.load_data("sc")
    treatment_time = 70
    result = cp.SyntheticControl(
        df,
        treatment_time,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(cumulative=False, treated_unit="actual")

    assert isinstance(stats, EffectSummary)
    assert "cumulative" not in stats.table.index
    assert "average" in stats.table.index


@pytest.mark.integration
def test_effect_summary_sc_relative_false(mock_pymc_sample):
    """Test effect_summary with Synthetic Control and relative=False."""
    df = cp.load_data("sc")
    treatment_time = 70
    result = cp.SyntheticControl(
        df,
        treatment_time,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(relative=False, treated_unit="actual")

    assert isinstance(stats, EffectSummary)
    assert "relative_mean" not in stats.table.columns


@pytest.mark.integration
def test_effect_summary_ols_both_false(mock_pymc_sample):
    """Test effect_summary with OLS model, cumulative=False and relative=False."""
    from sklearn.linear_model import LinearRegression

    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=LinearRegression(),
    )

    stats = result.effect_summary(cumulative=False, relative=False)

    assert isinstance(stats, EffectSummary)
    assert "cumulative" not in stats.table.index
    assert "relative_mean" not in stats.table.columns
    assert "average" in stats.table.index


@pytest.mark.integration
def test_effect_summary_pymc_both_false(mock_pymc_sample):
    """Test effect_summary with PyMC model, cumulative=False and relative=False."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(cumulative=False, relative=False)

    assert isinstance(stats, EffectSummary)
    assert "cumulative" not in stats.table.index
    assert "relative_mean" not in stats.table.columns
    assert "average" in stats.table.index
