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
    assert stats.table["metric"].iloc[0] == "gradient_change"
    assert "mean" in stats.table.columns
    assert "median" in stats.table.columns
    assert "HDI_lower" in stats.table.columns
    assert "HDI_upper" in stats.table.columns
    assert "P(effect>0)" in stats.table.columns


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
    assert "P(effect>0)" in stats_increase.table.columns

    # Test decrease
    stats_decrease = result.effect_summary(direction="decrease")
    assert "P(effect<0)" in stats_decrease.table.columns

    # Test two-sided
    stats_two_sided = result.effect_summary(direction="two-sided")
    assert "P(two-sided)" in stats_two_sided.table.columns
    assert "P(effect)" in stats_two_sided.table.columns


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
    assert "P(|effect|>min_effect)" in stats.table.columns


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
