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
Tests for reporting utilities.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import causalpy as cp
from causalpy.reporting import EffectSummary, _BayesianDecision

sample_kwargs = {
    "chains": 2,
    "draws": 100,
    "progressbar": False,
    "random_seed": 42,
}


@pytest.mark.integration
def test_effect_summary_basic(mock_pymc_sample, its_data):
    """Test basic effect_summary functionality with ITS."""
    df = its_data
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
def test_effect_summary_with_cumulative(mock_pymc_sample, its_data):
    """Test effect_summary with cumulative effects."""
    df = its_data
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
def test_effect_summary_without_cumulative(mock_pymc_sample, its_data):
    """Test effect_summary without cumulative effects."""
    df = its_data
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
def test_effect_summary_with_relative(mock_pymc_sample, its_data):
    """Test effect_summary with relative effects."""
    df = its_data
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
def test_effect_summary_direction_increase(mock_pymc_sample, its_data):
    """Test effect_summary with direction='increase'."""
    df = its_data
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
def test_effect_summary_direction_decrease(mock_pymc_sample, its_data):
    """Test effect_summary with direction='decrease'."""
    df = its_data
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
def test_effect_summary_direction_two_sided(mock_pymc_sample, its_data):
    """Test effect_summary with direction='two-sided'."""
    df = its_data
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
def test_effect_summary_window_datetime(mock_pymc_sample, its_data):
    """Test effect_summary with datetime window."""
    df = its_data
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
    rng = np.random.default_rng(42)
    n_pre = 50
    n_post = 30
    t_pre = np.arange(n_pre)
    t_post = np.arange(n_pre, n_pre + n_post)

    y_pre = 10 + 0.5 * t_pre + rng.normal(0, 1, n_pre)
    y_post = 15 + 0.5 * t_post + rng.normal(0, 1, n_post)

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
def test_effect_summary_alpha(mock_pymc_sample, its_data):
    """Test effect_summary with custom alpha."""
    df = its_data
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
def test_effect_summary_rope(mock_pymc_sample, its_data):
    """Test effect_summary with ROPE (min_effect)."""
    df = its_data
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
    assert "Using the closed ROPE [-1, 1]" in stats.text
    assert "Posterior mass is" in stats.text
    assert any(
        verdict in stats.text
        for verdict in (
            "the effect is practically significant.",
            "the effect is practically equivalent to zero.",
            "the result is inconclusive.",
        )
    )


@pytest.mark.integration
def test_effect_summary_ols_its(mock_pymc_sample, its_data):
    """Test effect_summary with OLS model for ITS."""
    from sklearn.linear_model import LinearRegression

    df = its_data
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
def test_effect_summary_ols_did(mock_pymc_sample, did_data):
    """Test effect_summary with OLS model for DiD."""
    from sklearn.linear_model import LinearRegression

    df = did_data
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
def test_effect_summary_ols_did_residuals_are_per_observation(did_data):
    """``_compute_statistics_did_ols`` must compute one residual per
    observation, not an (n, n) array, and use the unbiased SSR/(n-p)
    estimator of the residual variance (matching the ``df = n - p`` already
    used for the t-distribution critical value).

    ``y_da`` (shape ``(n, 1)``, dims ``obs_ind x treated_units``) minus a bare
    ``(n,)`` ``y_pred`` array used to broadcast positionally against the
    *last* axis (``treated_units``, size 1) instead of ``obs_ind``, producing
    an ``(n, n)`` array of every observation's y minus every *other*
    observation's prediction; that inflated the reported SE by roughly 70x
    on this dataset. Separately, the residual variance was estimated as
    SSR/n (biased) rather than SSR/(n-p) (unbiased). With both bugs fixed,
    the reported SE/CI should match an independent statsmodels OLS fit
    almost exactly, not just up to some remaining conversion factor.
    """
    from sklearn.linear_model import LinearRegression

    df = did_data
    formula = "y ~ 1 + group * post_treatment"

    result = cp.DifferenceInDifferences(
        df,
        formula=formula,
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )

    X_da = result.design["X"]
    y_da = result.design["y"]
    y_pred = result.model.predict(X_da)
    residuals = np.asarray(y_da).reshape(-1) - np.asarray(y_pred).reshape(-1)
    n = X_da.shape[0]
    assert residuals.shape == (n,)

    stats = result.effect_summary()
    row = stats.table.loc["treatment_effect"]

    import statsmodels.formula.api as smf
    from scipy.stats import t as t_dist

    sm_fit = smf.ols(formula, data=df).fit()
    interaction_col = next(
        name for name in sm_fit.params.index if "group" in name and "post" in name
    )
    unbiased_se = sm_fit.bse[interaction_col]
    expected_ci_lower, expected_ci_upper = sm_fit.conf_int(alpha=0.05).loc[
        interaction_col
    ]

    t_crit = t_dist.ppf(1 - 0.05 / 2, df=n - X_da.shape[1])
    reported_se = (row["ci_upper"] - row["ci_lower"]) / (2 * t_crit)

    assert reported_se == pytest.approx(unbiased_se, rel=1e-8)
    assert row["ci_lower"] == pytest.approx(expected_ci_lower, rel=1e-6)
    assert row["ci_upper"] == pytest.approx(expected_ci_upper, rel=1e-6)
    # Guard against regressing to either the (n, n) broadcast bug (~70x
    # inflation) or the biased SSR/n denominator (~5% understatement).
    biased_mse = np.mean(residuals**2)
    XtX_inv = np.linalg.inv(np.asarray(X_da).T @ np.asarray(X_da))
    coeff_idx = next(
        i
        for i, label in enumerate(result.labels)
        if "group" in label and "post_treatment" in label and ":" in label
    )
    biased_se = np.sqrt(biased_mse * XtX_inv[coeff_idx, coeff_idx])
    assert reported_se != pytest.approx(biased_se, rel=1e-3)


@pytest.mark.integration
def test_effect_summary_ols_sc(mock_pymc_sample, sc_data):
    """Test effect_summary with OLS model for Synthetic Control."""
    from sklearn.linear_model import LinearRegression

    df = sc_data
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
def test_effect_summary_rd_pymc(mock_pymc_sample, rd_data):
    """Test effect_summary with Regression Discontinuity (PyMC)."""
    df = rd_data
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
def test_effect_summary_rd_ols(mock_pymc_sample, rd_data):
    """Test effect_summary with Regression Discontinuity (OLS)."""
    from sklearn.linear_model import LinearRegression

    df = rd_data
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
def test_effect_summary_ols_rd_residuals_are_per_observation(rd_data):
    """Regression-pin for RD OLS ``effect_summary()`` intervals.

    The pre-#1049 ``_compute_statistics_rd_ols`` subtracted a bare ``(n,)``
    ``y_pred`` array from the ``(n, 1)`` y DataArray, broadcasting to an
    ``(n, n)`` residual matrix (every observation's y minus every *other*
    observation's prediction) and inflating the MSE ~9x on this dataset, so
    the reported standard errors were ~3x too wide. ``_point_residuals``
    fixed that. Separately, the residual variance was estimated as SSR/n
    (biased) rather than SSR/(n-p) (unbiased), inconsistent with the
    ``df = n - p`` already used for the t-distribution critical value. With
    both bugs fixed, this test pins the corrected residual shape and SE
    against an independent statsmodels fit almost exactly, not just up to
    some remaining conversion factor.
    """
    from sklearn.linear_model import LinearRegression

    from causalpy.reporting import _point_residuals

    formula = "y ~ 1 + x + treated + x:treated"
    result = cp.RegressionDiscontinuity(
        rd_data,
        formula=formula,
        treatment_threshold=0.5,
        model=LinearRegression(),
    )

    n, p = result.design["X"].shape
    residuals = _point_residuals(result)
    assert residuals.shape == (n,)

    stats = result.effect_summary()
    row = stats.table.loc["discontinuity"]

    import statsmodels.formula.api as smf
    from scipy.stats import t as t_dist

    sm_fit = smf.ols(formula, data=result.fit_data).fit()
    interaction_col = next(name for name in sm_fit.params.index if "x:treated" in name)
    unbiased_se = sm_fit.bse[interaction_col]

    t_crit = t_dist.ppf(1 - 0.05 / 2, df=n - p)
    reported_se = (row["ci_upper"] - row["ci_lower"]) / (2 * t_crit)

    assert reported_se == pytest.approx(unbiased_se, rel=1e-8)
    # Guard against regressing to either the (n, n) broadcast bug (~3x
    # inflation) or the biased SSR/n denominator understatement.
    biased_mse = np.mean(residuals**2)
    XtX_inv = np.linalg.inv(
        np.asarray(result.design["X"]).T @ np.asarray(result.design["X"])
    )
    coeff_idx = next(
        i
        for i, label in enumerate(result.labels)
        if "treated" in label.lower() and ":" in label
    )
    biased_se = np.sqrt(biased_mse * XtX_inv[coeff_idx, coeff_idx])
    assert reported_se != pytest.approx(biased_se, rel=1e-3)


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


def test_effect_summary_rkink_ols_raises():
    """The OLS path for Regression Kink should raise NotImplementedError."""
    from types import SimpleNamespace

    from causalpy.reporting import _effect_summary_rkink

    mock_result = SimpleNamespace(gradient_change=1.5)
    with pytest.raises(
        NotImplementedError, match="OLS models are not currently supported"
    ):
        _effect_summary_rkink(mock_result)


@pytest.mark.integration
def test_effect_summary_empty_window_error(mock_pymc_sample, its_data):
    """Test that effect_summary raises error for empty window."""
    df = its_data
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
def test_effect_summary_hdi_coverage(mock_pymc_sample, its_data):
    """Test that HDI intervals are properly ordered."""
    df = its_data
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
def test_effect_summary_tail_probabilities_match(mock_pymc_sample, its_data):
    """Test that tail probabilities match manual calculations."""
    df = its_data
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    stats = result.effect_summary(direction="increase")

    # Manually calculate P(effect > 0)
    avg_effect = result.post_impact.mean(dim="obs_ind")
    manual_p_gt_0 = float((avg_effect > 0).mean().values)

    # Should match (within floating point precision)
    assert abs(stats.table.loc["average", "p_gt_0"] - manual_p_gt_0) < 1e-10


@pytest.mark.integration
def test_effect_summary_synthetic_control(mock_pymc_sample, sc_data):
    """Test effect_summary with Synthetic Control experiment (single treated unit)."""
    df = sc_data
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
    rng = np.random.default_rng(42)
    n_obs = 60
    n_control = 4
    n_treated = 2

    # Create time index
    time_index = pd.date_range("2020-01-01", periods=n_obs, freq="D")
    treatment_time = time_index[40]

    # Control unit data
    control_data = {}
    for i in range(n_control):
        control_data[f"control_{i}"] = rng.normal(10, 2, n_obs) + np.sin(
            np.arange(n_obs) * 0.1
        )

    # Treated unit data
    treated_data = {}
    for j in range(n_treated):
        weights = rng.dirichlet(np.ones(n_control))
        base_signal = sum(
            weights[i] * control_data[f"control_{i}"] for i in range(n_control)
        )
        treatment_effect = np.zeros(n_obs)
        treatment_effect[40:] = rng.normal(5, 1, n_obs - 40)
        treated_data[f"treated_{j}"] = (
            base_signal + treatment_effect + rng.normal(0, 0.5, n_obs)
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
def test_effect_summary_synthetic_control_window(mock_pymc_sample, sc_data):
    """Test effect_summary with Synthetic Control using window specification."""
    df = sc_data
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
def test_effect_summary_did(mock_pymc_sample, did_data):
    """Test effect_summary with Difference-in-Differences experiment."""
    df = did_data
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
def test_effect_summary_did_direction_increase(mock_pymc_sample, did_data):
    """Test effect_summary with DiD and direction='increase'."""
    df = did_data
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
def test_effect_summary_did_direction_decrease(mock_pymc_sample, did_data):
    """Test effect_summary with DiD and direction='decrease'."""
    df = did_data
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
def test_effect_summary_did_direction_two_sided(mock_pymc_sample, did_data):
    """Test effect_summary with DiD and direction='two-sided'."""
    df = did_data
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
def test_effect_summary_did_rope(mock_pymc_sample, did_data):
    """Test effect_summary with DiD and ROPE."""
    df = did_data
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
    assert "Using the closed ROPE [-1, 1]" in stats.text
    assert "Posterior mass is" in stats.text


@pytest.mark.integration
def test_effect_summary_did_ols_error(mock_pymc_sample, did_data):
    """Test that effect_summary works for DiD with OLS model (OLS is now supported)."""
    from sklearn.linear_model import LinearRegression

    df = did_data
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
def test_effect_summary_did_hdi_coverage(mock_pymc_sample, did_data):
    """Test that HDI intervals are properly ordered for DiD."""
    df = did_data
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


def _fixed_bayesian_decision(
    interval,
    tail_label,
    tail_probability,
    *,
    conclusion="descriptive",
    rope=None,
    masses=(None, None, None),
):
    """Create a fixed decision for direct prose fixtures."""
    below, inside, above = masses
    return _BayesianDecision(
        conclusion=conclusion,
        framework="descriptive" if rope is None else "hdi_rope",
        interval=interval,
        rope=rope,
        tail_label=tail_label,
        tail_probability=tail_probability,
        posterior_mass_below_rope=below,
        posterior_mass_inside_rope=inside,
        posterior_mass_above_rope=above,
    )


@pytest.mark.parametrize(
    ("direction", "tail_probability"),
    [
        ("increase", 0.5),
        ("decrease", 0.25),
        ("two-sided", 0.5),
    ],
)
def test_make_bayesian_decision_without_rope_uses_requested_tail(
    direction, tail_probability
):
    """No-ROPE decisions are descriptive and retain their requested tail."""
    import xarray as xr

    from causalpy.reporting import _compute_tail_probabilities, _make_bayesian_decision

    effect = xr.DataArray([2.0, -1.0, 0.0, 2.0])
    decision = _make_bayesian_decision(
        effect,
        hdi_lower=-1.0,
        hdi_upper=2.0,
        tail_probabilities=_compute_tail_probabilities(effect, direction),
        direction=direction,
        min_effect=None,
    )

    assert decision.conclusion == "descriptive"
    assert decision.framework == "descriptive"
    assert decision.rope is None
    assert decision.posterior_mass_below_rope is None
    assert decision.posterior_mass_inside_rope is None
    assert decision.posterior_mass_above_rope is None
    assert decision.tail_label == direction
    assert decision.tail_probability == tail_probability


@pytest.mark.parametrize(
    ("interval", "conclusion"),
    [
        ((1.01, 2.0), "practically_significant"),
        ((-2.0, -1.01), "practically_significant"),
        ((-1.0, 1.0), "practically_equivalent_to_zero"),
        ((1.0, 2.0), "inconclusive"),
        ((-2.0, -1.0), "inconclusive"),
        ((-0.5, 1.01), "inconclusive"),
    ],
)
def test_make_bayesian_decision_uses_closed_rope_geometry(interval, conclusion):
    """HDI verdicts use strict non-overlap and inclusive closed-ROPE endpoints."""
    import xarray as xr

    from causalpy.reporting import _make_bayesian_decision

    decision = _make_bayesian_decision(
        xr.DataArray([-2.0, 0.0, 2.0]),
        hdi_lower=interval[0],
        hdi_upper=interval[1],
        tail_probabilities={"p_gt_0": 0.5},
        direction="increase",
        min_effect=1.0,
    )

    assert decision.conclusion == conclusion
    assert decision.rope == (-1.0, 1.0)


@pytest.mark.parametrize(
    ("interval", "conclusion"),
    [
        ((np.nextafter(1.0, -np.inf), 2.0), "inconclusive"),
        ((1.0, 2.0), "inconclusive"),
        ((np.nextafter(1.0, np.inf), 2.0), "practically_significant"),
        ((-2.0, np.nextafter(-1.0, -np.inf)), "practically_significant"),
        ((-2.0, -1.0), "inconclusive"),
        ((-2.0, np.nextafter(-1.0, np.inf)), "inconclusive"),
    ],
)
def test_make_bayesian_decision_compares_raw_rope_boundaries(interval, conclusion):
    """ROPE geometry must not round endpoints before comparing them."""
    import xarray as xr

    from causalpy.reporting import _make_bayesian_decision

    decision = _make_bayesian_decision(
        xr.DataArray([-2.0, 0.0, 2.0]),
        hdi_lower=interval[0],
        hdi_upper=interval[1],
        tail_probabilities={"p_gt_0": 0.5},
        direction="increase",
        min_effect=1.0,
    )

    assert decision.conclusion == conclusion


def test_make_bayesian_decision_partitions_mass_with_rope_boundary_ties():
    """Boundary draws belong inside the closed ROPE and masses partition exactly."""
    import xarray as xr

    from causalpy.reporting import _make_bayesian_decision

    decision = _make_bayesian_decision(
        xr.DataArray([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]),
        hdi_lower=-1.5,
        hdi_upper=1.5,
        tail_probabilities={"p_gt_0": 3 / 7},
        direction="increase",
        min_effect=1.0,
    )

    assert decision.posterior_mass_below_rope == pytest.approx(1 / 7)
    assert decision.posterior_mass_inside_rope == pytest.approx(5 / 7)
    assert decision.posterior_mass_above_rope == pytest.approx(1 / 7)
    assert (
        decision.posterior_mass_below_rope
        + decision.posterior_mass_inside_rope
        + decision.posterior_mass_above_rope
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("min_effect", [-1.0, np.nan, np.inf, -np.inf])
def test_make_bayesian_decision_rejects_invalid_rope_thresholds(min_effect):
    """Negative and non-finite ROPE thresholds are rejected."""
    import xarray as xr

    from causalpy.reporting import _make_bayesian_decision

    with pytest.raises(ValueError, match="finite and non-negative"):
        _make_bayesian_decision(
            xr.DataArray([0.0]),
            hdi_lower=0.0,
            hdi_upper=0.0,
            tail_probabilities={"p_gt_0": 0.0},
            direction="increase",
            min_effect=min_effect,
        )


def test_make_bayesian_decision_accepts_zero_width_rope():
    """A zero threshold remains a closed point ROPE with a mass partition."""
    import xarray as xr

    from causalpy.reporting import _make_bayesian_decision

    decision = _make_bayesian_decision(
        xr.DataArray([-1.0, 0.0, 1.0]),
        hdi_lower=0.0,
        hdi_upper=0.0,
        tail_probabilities={"p_gt_0": 1 / 3},
        direction="increase",
        min_effect=0.0,
    )

    assert decision.rope == (0.0, 0.0)
    assert decision.conclusion == "practically_equivalent_to_zero"
    assert decision.posterior_mass_below_rope == pytest.approx(1 / 3)
    assert decision.posterior_mass_inside_rope == pytest.approx(1 / 3)
    assert decision.posterior_mass_above_rope == pytest.approx(1 / 3)


def test_bayesian_decision_ignores_nonfinite_draws_in_tail_and_rope_masses():
    """Finite posterior draws alone determine Bayesian tail and ROPE masses."""
    import xarray as xr

    from causalpy.reporting import _compute_tail_probabilities, _make_bayesian_decision

    effect = xr.DataArray([-2.0, 0.0, 2.0, np.nan])
    tail_probabilities = _compute_tail_probabilities(effect, "increase")
    decision = _make_bayesian_decision(
        effect,
        hdi_lower=-2.0,
        hdi_upper=2.0,
        tail_probabilities=tail_probabilities,
        direction="increase",
        min_effect=1.0,
    )

    assert decision.tail_probability == pytest.approx(1 / 3)
    assert decision.posterior_mass_below_rope == pytest.approx(1 / 3)
    assert decision.posterior_mass_inside_rope == pytest.approx(1 / 3)
    assert decision.posterior_mass_above_rope == pytest.approx(1 / 3)


def test_scalar_statistics_filter_nonfinite_posterior_draws():
    """All scalar Bayesian summaries use the same finite posterior draws."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics_scalar

    effect = xr.DataArray([[-2.0, np.inf, 2.0]], dims=["chain", "draw"])
    stats = _compute_statistics_scalar(effect, min_effect=1.0)

    assert stats["mean"] == pytest.approx(0.0)
    assert stats["median"] == pytest.approx(0.0)
    assert np.isfinite(stats["hdi_lower"])
    assert np.isfinite(stats["hdi_upper"])
    assert stats["p_gt_0"] == pytest.approx(0.5)
    assert stats["decision"].posterior_mass_below_rope == pytest.approx(0.5)
    assert stats["decision"].posterior_mass_above_rope == pytest.approx(0.5)

    with pytest.raises(ValueError, match="contains no finite draws"):
        _compute_statistics_scalar(
            xr.DataArray([[np.nan, np.inf]], dims=["chain", "draw"])
        )


def test_time_series_statistics_filter_nonfinite_posterior_draws():
    """Average and cumulative Bayesian summaries share finite posterior draws."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics

    impact = xr.DataArray(
        [[[-0.5, -0.5], [np.inf, np.inf], [1.5, 1.5]]],
        dims=["chain", "draw", "obs_ind"],
    )
    stats = _compute_statistics(
        impact,
        xr.zeros_like(impact),
        min_effect=1.0,
        relative=False,
    )

    assert stats["avg"]["mean"] == pytest.approx(0.5)
    assert stats["cum"]["mean"] == pytest.approx(1.0)
    assert stats["avg"]["median"] == pytest.approx(0.5)
    assert stats["cum"]["median"] == pytest.approx(1.0)
    for summary in stats.values():
        assert np.isfinite(summary["hdi_lower"])
        assert np.isfinite(summary["hdi_upper"])
        assert summary["p_gt_0"] == pytest.approx(0.5)
        assert summary["decision"].posterior_mass_inside_rope == pytest.approx(0.5)
        assert summary["decision"].posterior_mass_above_rope == pytest.approx(0.5)

    with pytest.raises(ValueError, match="contains no finite draws"):
        _compute_statistics(
            xr.full_like(impact, np.inf),
            xr.zeros_like(impact),
            relative=False,
        )


def test_effect_summary_helpers_use_alpha_for_scalar_and_time_series_hdis():
    """Both Bayesian summary assembly paths use HDI coverage ``1 - alpha``."""
    import xarray as xr

    from causalpy._arviz_compat import hdi_bounds
    from causalpy.reporting import _effect_summary_did, _effect_summary_timeseries

    alpha = 0.025
    effect = xr.DataArray(np.arange(101, dtype=float)[None, :], dims=["chain", "draw"])
    expected_bounds = hdi_bounds(effect, prob=1 - alpha)

    scalar = _effect_summary_did(SimpleNamespace(causal_impact=effect), alpha=alpha)
    assert tuple(
        scalar.table.loc["treatment_effect", ["hdi_lower", "hdi_upper"]]
    ) == pytest.approx(expected_bounds)
    assert "97.5% HDI" in scalar.text

    impact = xr.DataArray(
        np.repeat(effect.values[:, :, None], 2, axis=2),
        dims=["chain", "draw", "obs_ind"],
        coords={"obs_ind": [0, 1]},
    )
    time_series = _effect_summary_timeseries(
        impact,
        xr.zeros_like(impact),
        pd.Index([0, 1], name="obs_ind"),
        alpha=alpha,
        cumulative=False,
        relative=False,
    )
    assert tuple(
        time_series.table.loc["average", ["hdi_lower", "hdi_upper"]]
    ) == pytest.approx(expected_bounds)
    assert "97.5% interval" in time_series.text


def test_effect_summary_timeseries_renders_distinct_cumulative_rope_decision():
    """Public time-series prose renders the cumulative decision, not the average one."""
    import xarray as xr

    from causalpy.reporting import _effect_summary_timeseries

    impact = xr.DataArray(
        np.full((1, 20, 3), 0.5),
        dims=["chain", "draw", "obs_ind"],
        coords={"obs_ind": [0, 1, 2]},
    )
    summary = _effect_summary_timeseries(
        impact,
        xr.zeros_like(impact),
        pd.Index([0, 1, 2], name="obs_ind"),
        min_effect=1.0,
        relative=False,
    )

    assert "The cumulative effect is 1.50 with a 95% HDI [1.50, 1.50]." in summary.text
    assert (
        "For the cumulative effect, The posterior probability of an increase is 1.000. "
        "Using the closed ROPE [-1, 1], the 95% HDI is entirely outside the ROPE; "
        "the effect is practically significant. Posterior mass is 0.000 below, "
        "0.000 inside, and 1.000 above the ROPE."
    ) in summary.text


def test_effect_summary_did_distinguishes_zero_rope_from_no_rope():
    """A public summary retains a point ROPE and strict ``p_rope`` at zero."""
    import xarray as xr

    from causalpy.reporting import _effect_summary_did

    summary = _effect_summary_did(
        SimpleNamespace(
            causal_impact=xr.DataArray([[-1.0, 0.0, 1.0]], dims=["chain", "draw"])
        ),
        min_effect=0.0,
    )

    assert summary.table.loc["treatment_effect", "p_rope"] == pytest.approx(1 / 3)
    assert "Using the closed ROPE [0, 0]" in summary.text


def test_compute_statistics_time_series_rope_uses_requested_direction():
    """Average and cumulative ``p_rope`` use strict requested-direction tails."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics

    draw_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    impact = xr.DataArray(
        np.tile(draw_values[None, :, None], (1, 1, 2)),
        dims=["chain", "draw", "obs_ind"],
    )
    counterfactual = xr.zeros_like(impact)

    expected = {
        "increase": (1 / 5, 2 / 5),
        "decrease": (1 / 5, 2 / 5),
        "two-sided": (2 / 5, 4 / 5),
    }
    for direction, (avg_expected, cum_expected) in expected.items():
        stats = _compute_statistics(
            impact,
            counterfactual,
            hdi_prob=0.95,
            direction=direction,
            cumulative=True,
            relative=False,
            min_effect=1.0,
        )

        assert stats["avg"]["p_rope"] == pytest.approx(avg_expected)
        assert stats["cum"]["p_rope"] == pytest.approx(cum_expected)


def test_compute_statistics_builds_distinct_average_and_cumulative_decisions():
    """Cumulative ROPE geometry is computed independently of average geometry."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics

    impact = xr.DataArray(np.full((1, 5, 2), 0.6), dims=["chain", "draw", "obs_ind"])
    stats = _compute_statistics(
        impact,
        xr.zeros_like(impact),
        hdi_prob=0.95,
        direction="increase",
        cumulative=True,
        relative=False,
        min_effect=1.0,
    )

    assert stats["avg"]["decision"].conclusion == "practically_equivalent_to_zero"
    assert stats["cum"]["decision"].conclusion == "practically_significant"


def test_compute_statistics_scalar_hdi_golden_unmonkeypatched():
    """Approved fixed-seed HDI golden via reporting scalar helpers (no monkeypatch).

    Uses ``default_rng(42).normal(size=(2, 200))`` through
    ``_compute_statistics_scalar(..., hdi_prob=0.94)`` and
    ``_generate_table_scalar``. Bounds match the approved baseline to 1e-12.
    """
    import xarray as xr

    from causalpy.reporting import _compute_statistics_scalar, _generate_table_scalar

    rng = np.random.default_rng(42)
    effect = xr.DataArray(rng.normal(size=(2, 200)), dims=["chain", "draw"])
    stats = _compute_statistics_scalar(effect, hdi_prob=0.94)
    table = _generate_table_scalar(stats)

    assert stats["hdi_lower"] == pytest.approx(
        -1.7577283913566313, rel=1e-12, abs=1e-12
    )
    assert stats["hdi_upper"] == pytest.approx(1.732311605409944, rel=1e-12, abs=1e-12)
    assert table.loc["effect", "hdi_lower"] == pytest.approx(
        -1.7577283913566313, rel=1e-12, abs=1e-12
    )
    assert table.loc["effect", "hdi_upper"] == pytest.approx(
        1.732311605409944, rel=1e-12, abs=1e-12
    )
    assert "decision" not in table.columns


def test_compute_statistics_scalar_singleton_treated_units():
    """Singleton ``treated_units`` must squeeze through scalar HDI reporting."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics_scalar, _generate_table_scalar

    rng = np.random.default_rng(42)
    effect = xr.DataArray(
        rng.normal(size=(2, 200, 1)),
        dims=["chain", "draw", "treated_units"],
        coords={"treated_units": ["unit_a"]},
    )
    stats = _compute_statistics_scalar(effect, hdi_prob=0.94)
    table = _generate_table_scalar(stats)

    assert stats["hdi_lower"] == pytest.approx(
        -1.7577283913566313, rel=1e-12, abs=1e-12
    )
    assert stats["hdi_upper"] == pytest.approx(1.732311605409944, rel=1e-12, abs=1e-12)
    assert isinstance(stats["mean"], float)
    assert isinstance(table.loc["effect", "hdi_lower"], float)


def test_compute_statistics_scalar_unreduced_treated_units_raises():
    """Unreduced multi-value ``treated_units`` must fail closed in scalar reporting."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics_scalar

    rng = np.random.default_rng(42)
    effect = xr.DataArray(
        rng.normal(size=(2, 50, 2)),
        dims=["chain", "draw", "treated_units"],
        coords={"treated_units": ["a", "b"]},
    )
    with pytest.raises(ValueError):
        _compute_statistics_scalar(effect, hdi_prob=0.94)


def test_as_scalar_handles_singleton_arrays():
    """_as_scalar should work for both scalar and singleton-array values."""
    import xarray as xr

    from causalpy.utils import _as_scalar

    assert _as_scalar(np.array(2.5)) == 2.5
    assert _as_scalar(np.array([2.5])) == 2.5
    assert _as_scalar(xr.DataArray([2.5], dims=["treated_units"])) == 2.5


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


def test_compute_statistics_rope_decrease():
    """Regression test: ROPE in _compute_statistics must use effect < -min_effect
    for direction='decrease', not effect > min_effect (which was the bug)."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics

    # 8 posterior draws, 3 time points.
    # All draws are strongly negative (around -5).
    rng = np.random.default_rng(42)
    draws = rng.normal(loc=-5.0, scale=0.5, size=(1, 200, 3))
    impact = xr.DataArray(
        draws,
        dims=["chain", "draw", "obs_ind"],
        coords={"obs_ind": [0, 1, 2]},
    )
    counterfactual = xr.DataArray(
        np.ones((1, 200, 3)) * 10.0,
        dims=["chain", "draw", "obs_ind"],
        coords={"obs_ind": [0, 1, 2]},
    )

    stats = _compute_statistics(
        impact,
        counterfactual,
        hdi_prob=0.95,
        direction="decrease",
        cumulative=True,
        relative=False,
        min_effect=1.0,
    )

    # With draws around -5, virtually all should satisfy effect < -1.0
    assert stats["avg"]["p_rope"] > 0.95
    assert stats["cum"]["p_rope"] > 0.95


def test_compute_statistics_time_series_hdi_golden():
    """ArviZ 0.22 94% HDIs for a frozen average/cumulative/relative table."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics, _generate_table

    rng = np.random.default_rng(123)
    impact = xr.DataArray(
        rng.normal(size=(2, 200, 3)),
        dims=["chain", "draw", "obs_ind"],
    )
    counterfactual = xr.DataArray(
        10 + rng.normal(size=(2, 200, 3)),
        dims=["chain", "draw", "obs_ind"],
    )

    table = _generate_table(
        _compute_statistics(
            impact,
            counterfactual,
            hdi_prob=0.94,
            cumulative=True,
            relative=True,
        )
    )

    assert tuple(table.loc["average", ["hdi_lower", "hdi_upper"]]) == pytest.approx(
        (-1.136764984172878, 1.1654528765047945),
        rel=1e-12,
        abs=1e-12,
    )
    assert tuple(table.loc["cumulative", ["hdi_lower", "hdi_upper"]]) == pytest.approx(
        (-3.4102949525186337, 3.496358629514383),
        rel=1e-12,
        abs=1e-12,
    )
    assert tuple(
        table.loc["average", ["relative_hdi_lower", "relative_hdi_upper"]]
    ) == pytest.approx(
        (-11.11634243961037, 12.08950890903512),
        rel=1e-12,
        abs=1e-12,
    )
    assert tuple(
        table.loc["cumulative", ["relative_hdi_lower", "relative_hdi_upper"]]
    ) == pytest.approx(
        (-11.116342446857432, 12.089508916593045),
        rel=1e-12,
        abs=1e-12,
    )
    assert "decision" not in table.columns


def test_compute_statistics_with_singleton_treated_unit_dim():
    """Regression test for singleton dims surviving reductions in xarray workflows."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics

    rng = np.random.default_rng(123)
    impact = xr.DataArray(
        rng.normal(loc=2.0, scale=0.1, size=(1, 100, 4, 1)),
        dims=["chain", "draw", "obs_ind", "treated_units"],
        coords={"obs_ind": [0, 1, 2, 3], "treated_units": ["unit_a"]},
    )
    counterfactual = xr.DataArray(
        np.ones((1, 100, 4, 1)) * 10.0,
        dims=["chain", "draw", "obs_ind", "treated_units"],
        coords={"obs_ind": [0, 1, 2, 3], "treated_units": ["unit_a"]},
    )

    stats = _compute_statistics(
        impact,
        counterfactual,
        hdi_prob=0.95,
        direction="two-sided",
        cumulative=True,
        relative=True,
    )

    assert isinstance(stats["avg"]["mean"], float)
    assert isinstance(stats["cum"]["mean"], float)
    assert isinstance(stats["avg"]["relative_mean"], float)
    assert isinstance(stats["cum"]["relative_mean"], float)


def test_compute_statistics_unreduced_treated_units_raises():
    """Unreduced multi-value ``treated_units`` must fail in time-series reporting."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics

    rng = np.random.default_rng(0)
    impact = xr.DataArray(
        rng.normal(loc=1.0, scale=0.1, size=(2, 20, 4, 2)),
        dims=["chain", "draw", "obs_ind", "treated_units"],
        coords={"obs_ind": [0, 1, 2, 3], "treated_units": ["a", "b"]},
    )
    counterfactual = xr.DataArray(
        np.ones((2, 20, 4, 2)) * 10.0,
        dims=["chain", "draw", "obs_ind", "treated_units"],
        coords={"obs_ind": [0, 1, 2, 3], "treated_units": ["a", "b"]},
    )

    with pytest.raises(ValueError):
        _compute_statistics(
            impact,
            counterfactual,
            hdi_prob=0.94,
            direction="two-sided",
            cumulative=True,
            relative=True,
        )


def test_extract_window_canonical_dataarray():
    """_extract_window returns the canonical DataArray unchanged for 'post'."""
    import xarray as xr

    from causalpy.reporting import _extract_window

    datapost = pd.DataFrame(index=pd.Index([10, 11, 12], name="obs_ind"))
    result = SimpleNamespace(
        post_impact=xr.DataArray(
            [1.0, 2.0, 3.0], dims=["obs_ind"], coords={"obs_ind": [10, 11, 12]}
        ),
        datapost=datapost,
    )

    windowed_impact, window_coords = _extract_window(result, window="post")

    assert isinstance(windowed_impact, xr.DataArray)
    assert window_coords.equals(datapost.index)


def test_extract_counterfactual_canonical_dataarray():
    """_extract_counterfactual selects the window from the canonical DataArray."""
    import xarray as xr

    from causalpy.reporting import _extract_counterfactual

    datapost = pd.DataFrame(index=pd.Index([10, 11, 12], name="obs_ind"))
    result = SimpleNamespace(
        post_pred=xr.DataArray(
            [5.0, 6.0, 7.0], dims=["obs_ind"], coords={"obs_ind": [10, 11, 12]}
        ),
        datapost=datapost,
    )

    window_coords = datapost.index[:2]
    counterfactual = _extract_counterfactual(result, window_coords)

    assert isinstance(counterfactual, xr.DataArray)
    np.testing.assert_array_equal(counterfactual.values, np.array([5.0, 6.0]))


def test_compute_statistics_rope_increase():
    """Test ROPE in _compute_statistics for direction='increase'."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics

    rng = np.random.default_rng(42)
    draws = rng.normal(loc=5.0, scale=0.5, size=(1, 200, 3))
    impact = xr.DataArray(
        draws,
        dims=["chain", "draw", "obs_ind"],
        coords={"obs_ind": [0, 1, 2]},
    )
    counterfactual = xr.DataArray(
        np.ones((1, 200, 3)) * 10.0,
        dims=["chain", "draw", "obs_ind"],
        coords={"obs_ind": [0, 1, 2]},
    )

    stats = _compute_statistics(
        impact,
        counterfactual,
        hdi_prob=0.95,
        direction="increase",
        cumulative=True,
        relative=False,
        min_effect=1.0,
    )

    # With draws around +5, virtually all should satisfy effect > 1.0
    assert stats["avg"]["p_rope"] > 0.95
    assert stats["cum"]["p_rope"] > 0.95


def test_compute_statistics_rope_two_sided():
    """Test ROPE in _compute_statistics for direction='two-sided'."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics

    rng = np.random.default_rng(42)
    draws = rng.normal(loc=-5.0, scale=0.5, size=(1, 200, 3))
    impact = xr.DataArray(
        draws,
        dims=["chain", "draw", "obs_ind"],
        coords={"obs_ind": [0, 1, 2]},
    )
    counterfactual = xr.DataArray(
        np.ones((1, 200, 3)) * 10.0,
        dims=["chain", "draw", "obs_ind"],
        coords={"obs_ind": [0, 1, 2]},
    )

    stats = _compute_statistics(
        impact,
        counterfactual,
        hdi_prob=0.95,
        direction="two-sided",
        cumulative=True,
        relative=False,
        min_effect=1.0,
    )

    # With draws around -5, |effect| > 1.0 for virtually all
    assert stats["avg"]["p_rope"] > 0.95
    assert stats["cum"]["p_rope"] > 0.95


def test_compute_statistics_rope_near_threshold():
    """Test ROPE when effect is near the threshold — should give intermediate prob."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics

    rng = np.random.default_rng(42)
    # Draws centered at -2.0 with sd=1.0; min_effect=2.0 tests effect < -2.0
    draws = rng.normal(loc=-2.0, scale=1.0, size=(1, 500, 3))
    impact = xr.DataArray(
        draws,
        dims=["chain", "draw", "obs_ind"],
        coords={"obs_ind": [0, 1, 2]},
    )
    counterfactual = xr.DataArray(
        np.ones((1, 500, 3)) * 10.0,
        dims=["chain", "draw", "obs_ind"],
        coords={"obs_ind": [0, 1, 2]},
    )

    stats = _compute_statistics(
        impact,
        counterfactual,
        hdi_prob=0.95,
        direction="decrease",
        cumulative=False,
        relative=False,
        min_effect=2.0,
    )

    # Mean at -2.0, threshold at -2.0 → should be roughly 0.5
    assert 0.3 < stats["avg"]["p_rope"] < 0.7


def test_compute_statistics_rope_honors_requested_direction():
    """Time-series ``p_rope`` must not flip a requested increase to decrease."""
    import xarray as xr

    from causalpy.reporting import _compute_statistics

    draws = np.full((1, 20, 3), -5.0)
    impact = xr.DataArray(draws, dims=["chain", "draw", "obs_ind"])
    counterfactual = xr.DataArray(
        np.full((1, 20, 3), 10.0), dims=["chain", "draw", "obs_ind"]
    )

    stats = _compute_statistics(
        impact,
        counterfactual,
        hdi_prob=0.95,
        direction="increase",
        cumulative=True,
        relative=False,
        min_effect=1.0,
    )

    assert stats["avg"]["mean"] < 0
    assert stats["cum"]["mean"] < 0
    assert stats["avg"]["p_rope"] == 0.0
    assert stats["cum"]["p_rope"] == 0.0


def test_format_number():
    """Test _format_number helper."""
    from causalpy.reporting import _format_number

    assert _format_number(3.14159, decimals=2) == "3.14"
    assert _format_number(3.14159, decimals=3) == "3.142"
    assert _format_number(10.0, decimals=1) == "10.0"
    assert _format_number(0.001, decimals=4) == "0.0010"


def test_format_rope_bound_is_compact_and_round_trip_safe():
    """ROPE prose remains usable for finite thresholds across float magnitudes."""
    from causalpy.reporting import _format_rope_bound

    smallest = np.nextafter(0.0, 1.0)
    largest = np.finfo(float).max

    assert _format_rope_bound(1.0) == "1"
    assert float(_format_rope_bound(smallest)) == smallest
    assert float(_format_rope_bound(largest)) == largest


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


def test_effect_summary_timeseries_dispatches_on_draws_not_backend():
    """Contract: the container, not backend identity, decides the statistics.

    A prediction container carrying posterior draws gets HDI summaries; a
    singleton (chain=1, draw=1) container falls back to t-based intervals —
    regardless of which backend produced it.
    """
    import xarray as xr

    from causalpy.reporting import _effect_summary_timeseries

    def containers(n_draws):
        rng = np.random.default_rng(42)
        obs_ind = [0, 1, 2, 3]
        impact = xr.DataArray(
            rng.normal(5.0, 0.5, size=(1, n_draws, 4)),
            dims=["chain", "draw", "obs_ind"],
            coords={"obs_ind": obs_ind},
        )
        counterfactual = xr.DataArray(
            np.full((1, n_draws, 4), 10.0),
            dims=["chain", "draw", "obs_ind"],
            coords={"obs_ind": obs_ind},
        )
        return impact, counterfactual, pd.Index(obs_ind)

    impact, counterfactual, window_coords = containers(n_draws=200)
    summary = _effect_summary_timeseries(impact, counterfactual, window_coords)
    assert isinstance(summary, EffectSummary)
    assert "hdi_lower" in summary.table.columns
    assert "ci_lower" not in summary.table.columns

    impact, counterfactual, window_coords = containers(n_draws=1)
    summary = _effect_summary_timeseries(impact, counterfactual, window_coords)
    assert isinstance(summary, EffectSummary)
    assert "ci_lower" in summary.table.columns
    assert "hdi_lower" not in summary.table.columns


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
    import xarray as xr

    from causalpy.reporting import _extract_window

    # Create a minimal mock result
    class MockResult:
        post_impact = xr.DataArray(
            [1.0, 2.0, 3.0], dims=["obs_ind"], coords={"obs_ind": [0, 1, 2]}
        )
        datapost = pd.DataFrame({"y": [1, 2, 3]}, index=[0, 1, 2])

    result = MockResult()

    # Invalid window type (not "post", tuple, or slice)
    with pytest.raises(ValueError, match="window must be"):
        _extract_window(result, window=[1, 2, 3])  # list is invalid


@pytest.mark.integration
def test_compute_statistics_did_ols_missing_interaction_term(
    mock_pymc_sample, did_data
):
    """Test _compute_statistics_did_ols error when interaction term is not found."""
    from sklearn.linear_model import LinearRegression

    from causalpy.reporting import _compute_statistics_did_ols

    df = did_data

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
def test_compute_statistics_rd_ols_fallback_path(mock_pymc_sample, rd_data):
    """Test _compute_statistics_rd_ols uses fallback when coefficient not found."""
    from sklearn.linear_model import LinearRegression

    from causalpy.reporting import _compute_statistics_rd_ols

    df = rd_data
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
    rng = np.random.default_rng(42)
    n_pre = 50
    n_post = 30
    t_pre = np.arange(n_pre)
    t_post = np.arange(n_pre, n_pre + n_post)

    y_pre = 10 + 0.5 * t_pre + rng.normal(0, 1, n_pre)
    y_post = 15 + 0.5 * t_post + rng.normal(0, 1, n_post)

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
    rng = np.random.default_rng(42)
    impact = xr.DataArray(
        rng.normal(1.0, 0.1, (2, 10, 5)),
        dims=["chain", "draw", "obs_ind"],
        coords={"chain": [0, 1], "draw": range(10), "obs_ind": range(5)},
    )

    # Counterfactual with values very close to zero
    counterfactual = xr.DataArray(
        rng.normal(0.0001, 0.00001, (2, 10, 5)),
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
def test_extract_counterfactual_canonical_pymc(mock_pymc_sample, its_data):
    """_extract_counterfactual selects a window from canonical PyMC predictions."""
    from causalpy.reporting import _extract_counterfactual

    df = its_data
    treatment_time = pd.to_datetime("2017-01-01")
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time,
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    window_coords = result.datapost.index[:10]
    counterfactual = _extract_counterfactual(result, window_coords, treated_unit=None)

    assert counterfactual.sizes["obs_ind"] == 10
    assert {"chain", "draw"} <= set(counterfactual.dims)


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
        "decision": _fixed_bayesian_decision((1.0, 4.0), "increase", 0.95),
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
        "decision": _fixed_bayesian_decision((-4.0, -1.0), "decrease", 0.98),
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
        "p_two_sided": 0.30,
        "prob_of_effect": 0.85,
        "decision": _fixed_bayesian_decision((1.0, 4.0), "two-sided", 0.30),
    }

    prose = _generate_prose_scalar(
        stats, "discontinuity", alpha=0.05, direction="two-sided"
    )

    assert "discontinuity" in prose
    assert "two-sided tail probability" in prose
    assert "0.850" not in prose
    assert "0.300" in prose


@pytest.mark.parametrize(
    ("alpha", "coverage"),
    [
        (0.05, "95%"),
        (0.10, "90%"),
        (0.025, "97.5%"),
    ],
)
def test_generate_prose_scalar_renders_coverage_from_alpha(alpha, coverage):
    """Bayesian prose coverage is derived from ``1 - alpha`` without truncation."""
    from causalpy.reporting import _generate_prose_scalar

    prose = _generate_prose_scalar(
        {
            "mean": 1.0,
            "hdi_lower": 0.5,
            "hdi_upper": 1.5,
            "decision": _fixed_bayesian_decision((0.5, 1.5), "increase", 0.9),
        },
        "effect",
        alpha=alpha,
        direction="increase",
    )

    assert f"{coverage} HDI" in prose


@pytest.mark.parametrize(
    ("conclusion", "interval", "expected_verdict"),
    [
        (
            "practically_significant",
            (1.01, 2.0),
            "is entirely outside the ROPE; the effect is practically significant.",
        ),
        (
            "practically_equivalent_to_zero",
            (-1.0, 1.0),
            "is entirely inside the ROPE; the effect is practically equivalent to zero.",
        ),
        (
            "inconclusive",
            (1.0, 2.0),
            "overlaps the ROPE; the result is inconclusive.",
        ),
    ],
)
def test_generate_prose_scalar_renders_each_rope_decision(
    conclusion, interval, expected_verdict
):
    """Scalar prose renders the attached ROPE verdict and complete mass partition."""
    from causalpy.reporting import _generate_prose_scalar

    prose = _generate_prose_scalar(
        {
            "mean": 1.5,
            "hdi_lower": interval[0],
            "hdi_upper": interval[1],
            "decision": _fixed_bayesian_decision(
                interval,
                "increase",
                0.75,
                conclusion=conclusion,
                rope=(-1.0, 1.0),
                masses=(0.1, 0.8, 0.1),
            ),
        },
        "effect",
        alpha=0.05,
        direction="increase",
    )

    assert "posterior probability of an increase is 0.750" in prose
    assert "Using the closed ROPE [-1, 1], the 95% HDI" in prose
    assert expected_verdict in prose
    assert (
        "Posterior mass is 0.100 below, 0.800 inside, and 0.100 above the ROPE."
        in prose
    )


def test_bayesian_prose_uses_attached_decision_without_rederiving_statistics():
    """Scalar and detailed prose read intervals and tails only from decisions."""
    from causalpy.reporting import _generate_prose_detailed, _generate_prose_scalar

    decision = _fixed_bayesian_decision((1.0, 2.0), "increase", 0.123)
    scalar_prose = _generate_prose_scalar(
        {
            "mean": -5.0,
            "hdi_lower": -6.0,
            "hdi_upper": -4.0,
            "p_gt_0": 0.999,
            "decision": decision,
        },
        "effect",
        direction="decrease",
    )
    detailed_prose = _generate_prose_detailed(
        {
            "avg": {
                "mean": -5.0,
                "hdi_lower": -6.0,
                "hdi_upper": -4.0,
                "p_lt_0": 0.999,
                "decision": decision,
            }
        },
        pd.Index([1]),
        direction="decrease",
        cumulative=False,
        relative=False,
    )

    for prose in (scalar_prose, detailed_prose):
        assert "posterior probability of an increase is 0.123" in prose
        assert "posterior probability of a decrease" not in prose
        assert "0.999" not in prose
        assert "HDI [1.00, 2.00]" in prose
        assert "HDI [-6.00, -4.00]" not in prose


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


def test_generate_prose_detailed_basic():
    """Test _generate_prose_detailed generates proper text with observed/cf values."""
    from causalpy.reporting import _generate_prose_detailed

    stats = {
        "avg": {
            "mean": 2.5,
            "hdi_lower": 1.0,
            "hdi_upper": 4.0,
            "p_gt_0": 0.99,
            "decision": _fixed_bayesian_decision((1.0, 4.0), "increase", 0.99),
        }
    }

    window_coords = pd.Index([10, 11, 12, 13, 14])

    prose = _generate_prose_detailed(
        stats,
        window_coords,
        alpha=0.05,
        direction="increase",
        cumulative=False,
        relative=False,
        observed_avg=52.5,
        counterfactual_avg=50.0,
    )

    assert "Post-period" in prose
    assert "10 to 14" in prose
    assert "52.50" in prose
    assert "50.00" in prose
    assert "2.50" in prose
    assert "95%" in prose
    assert "increase" in prose


def test_generate_prose_detailed_counterfactual_interval():
    """Test that counterfactual interval brackets the counterfactual mean."""
    from causalpy.reporting import _generate_prose_detailed

    stats = {
        "avg": {
            "mean": -2.0,
            "hdi_lower": -3.0,
            "hdi_upper": -1.0,
            "p_gt_0": 0.001,
            "decision": _fixed_bayesian_decision((-3.0, -1.0), "increase", 0.001),
        }
    }

    window_coords = pd.Index([10, 11, 12])

    prose = _generate_prose_detailed(
        stats,
        window_coords,
        alpha=0.05,
        direction="increase",
        cumulative=False,
        relative=False,
        observed_avg=18.0,
        counterfactual_avg=20.0,
    )

    # cf_interval_lower = observed - effect_upper = 18 - (-1) = 19.0
    # cf_interval_upper = observed - effect_lower = 18 - (-3) = 21.0
    # Both bracket the counterfactual mean of 20.0
    assert "19.00" in prose
    assert "21.00" in prose
    # The counterfactual mean should appear
    assert "20.00" in prose


def test_generate_prose_detailed_honors_requested_increase_direction():
    """A negative mean must not change a requested increase to decrease."""
    from causalpy.reporting import _generate_prose_detailed

    stats = {
        "avg": {
            "mean": -1.72,
            "hdi_lower": -2.15,
            "hdi_upper": -1.33,
            "p_gt_0": 0.0,
            "decision": _fixed_bayesian_decision((-2.15, -1.33), "increase", 0.0),
        }
    }

    window_coords = pd.Index(range(70, 100))

    prose = _generate_prose_detailed(
        stats,
        window_coords,
        alpha=0.05,
        direction="increase",
        cumulative=False,
        relative=False,
        observed_avg=17.1,
        counterfactual_avg=18.82,
    )

    assert "posterior probability of an increase is 0.000" in prose
    assert "posterior probability of a decrease" not in prose
    assert "does not include zero" not in prose


def test_generate_prose_detailed_honors_requested_decrease_direction():
    """A positive mean must not change a requested decrease to increase."""
    from causalpy.reporting import _generate_prose_detailed

    stats = {
        "avg": {
            "mean": 3.0,
            "hdi_lower": 1.5,
            "hdi_upper": 4.5,
            "p_lt_0": 0.001,
            "decision": _fixed_bayesian_decision((1.5, 4.5), "decrease", 0.001),
        }
    }

    window_coords = pd.Index(range(10, 20))

    prose = _generate_prose_detailed(
        stats,
        window_coords,
        alpha=0.05,
        direction="decrease",
        cumulative=False,
        relative=False,
        observed_avg=53.0,
        counterfactual_avg=50.0,
    )

    assert "posterior probability of a decrease is 0.001" in prose
    assert "posterior probability of an increase" not in prose


def test_generate_prose_detailed_two_sided_uses_tail_probability():
    """Two-sided prose names ``p_two_sided`` as a tail probability."""
    from causalpy.reporting import _generate_prose_detailed

    stats = {
        "avg": {
            "mean": 0.5,
            "hdi_lower": -0.2,
            "hdi_upper": 1.2,
            "p_two_sided": 0.2,
            "prob_of_effect": 0.9,
            "decision": _fixed_bayesian_decision((-0.2, 1.2), "two-sided", 0.2),
        }
    }
    window_coords = pd.Index([10, 11, 12])

    prose = _generate_prose_detailed(
        stats,
        window_coords,
        alpha=0.05,
        direction="two-sided",
        cumulative=False,
        relative=False,
    )

    assert "two-sided tail probability is 0.200" in prose
    assert "posterior probability of an effect" not in prose
    assert "0.900" not in prose


def test_generate_prose_detailed_preserves_custom_prefix_casing():
    """Test that prose preserves caller-provided prefix casing."""
    from causalpy.reporting import _generate_prose_detailed

    stats = {
        "avg": {
            "mean": 2.5,
            "hdi_lower": 1.0,
            "hdi_upper": 4.0,
            "p_gt_0": 0.99,
            "decision": _fixed_bayesian_decision((1.0, 4.0), "increase", 0.99),
        }
    }
    window_coords = pd.Index([10, 11, 12])

    prose = _generate_prose_detailed(
        stats,
        window_coords,
        alpha=0.05,
        direction="increase",
        cumulative=False,
        relative=False,
        prefix="Post-Intervention",
    )

    assert "During the Post-Intervention" in prose


def test_generate_prose_detailed_cumulative():
    """Test _generate_prose_detailed includes cumulative with correct intervals."""
    from causalpy.reporting import _generate_prose_detailed

    stats = {
        "avg": {
            "mean": 2.0,
            "hdi_lower": 1.0,
            "hdi_upper": 3.0,
            "p_gt_0": 0.99,
            "decision": _fixed_bayesian_decision((1.0, 3.0), "increase", 0.99),
        },
        "cum": {
            "mean": 20.0,
            "hdi_lower": 10.0,
            "hdi_upper": 30.0,
            "decision": _fixed_bayesian_decision(
                (10.0, 30.0),
                "increase",
                0.99,
                conclusion="practically_significant",
                rope=(-1.0, 1.0),
                masses=(0.0, 0.0, 1.0),
            ),
        },
    }

    window_coords = pd.Index(range(10, 20))

    prose = _generate_prose_detailed(
        stats,
        window_coords,
        alpha=0.05,
        direction="increase",
        cumulative=True,
        relative=False,
        observed_avg=52.0,
        counterfactual_avg=50.0,
        observed_cum=520.0,
        counterfactual_cum=500.0,
    )

    assert "Summing up" in prose
    assert "520.00" in prose
    assert "500.00" in prose
    # cum_cf_lower = 520 - 30 = 490, cum_cf_upper = 520 - 10 = 510
    assert "490.00" in prose
    assert "510.00" in prose
    assert "The cumulative effect is 20.00 with a 95% HDI [10.00, 30.00]." in prose
    assert (
        "For the cumulative effect, The posterior probability of an increase is 0.990."
        in prose
    )
    assert "the effect is practically significant." in prose
    assert "Using the closed ROPE [-1, 1]" in prose
    assert (
        "Posterior mass is 0.000 below, 0.000 inside, and 1.000 above the ROPE."
        in prose
    )


def test_generate_prose_detailed_with_relative():
    """Test _generate_prose_detailed includes relative effect text."""
    from causalpy.reporting import _generate_prose_detailed

    stats = {
        "avg": {
            "mean": 2.5,
            "hdi_lower": 1.0,
            "hdi_upper": 4.0,
            "p_gt_0": 0.99,
            "relative_mean": 5.0,
            "relative_hdi_lower": 2.0,
            "relative_hdi_upper": 8.0,
            "decision": _fixed_bayesian_decision((1.0, 4.0), "increase", 0.99),
        }
    }

    window_coords = pd.Index([10, 11, 12])

    prose = _generate_prose_detailed(
        stats,
        window_coords,
        alpha=0.05,
        direction="increase",
        cumulative=False,
        relative=True,
        observed_avg=52.5,
        counterfactual_avg=50.0,
    )

    assert "Relative to the counterfactual" in prose
    assert "5.00%" in prose
    assert "2.00%" in prose
    assert "8.00%" in prose


def test_generate_prose_detailed_rope_decision_in_prose():
    """Test that an attached ROPE decision, not ``p_rope``, controls prose."""
    from causalpy.reporting import _generate_prose_detailed

    stats = {
        "avg": {
            "mean": 2.5,
            "hdi_lower": 1.0,
            "hdi_upper": 4.0,
            "p_gt_0": 0.99,
            "p_rope": 0.85,
            "decision": _fixed_bayesian_decision(
                (1.0, 4.0),
                "increase",
                0.99,
                conclusion="practically_significant",
                rope=(-0.004, 0.004),
                masses=(0.0, 0.0, 1.0),
            ),
        }
    }

    window_coords = pd.Index([10, 11, 12])

    prose = _generate_prose_detailed(
        stats,
        window_coords,
        alpha=0.05,
        direction="increase",
        cumulative=False,
        relative=False,
        observed_avg=52.5,
        counterfactual_avg=50.0,
    )

    assert (
        "Using the closed ROPE [-0.004, 0.004], the 95% HDI is entirely outside"
        in prose
    )
    assert (
        "Posterior mass is 0.000 below, 0.000 inside, and 1.000 above the ROPE."
        in prose
    )
    assert "minimum effect size threshold" not in prose


def test_generate_prose_detailed_is_descriptive():
    """Ensure prose uses descriptive language, not prescriptive judgments."""
    from causalpy.reporting import _generate_prose_detailed

    stats = {
        "avg": {
            "mean": 2.5,
            "hdi_lower": 1.0,
            "hdi_upper": 4.0,
            "p_gt_0": 0.99,
            "decision": _fixed_bayesian_decision((1.0, 4.0), "increase", 0.99),
        }
    }

    window_coords = pd.Index([10, 11, 12])

    prose = _generate_prose_detailed(
        stats,
        window_coords,
        alpha=0.05,
        direction="increase",
        cumulative=False,
        relative=False,
        observed_avg=52.5,
        counterfactual_avg=50.0,
    )

    # Should NOT contain prescriptive language from the old four-way branching
    assert "statistically credible" not in prose
    assert "caution is warranted" not in prose
    assert "weak or inconclusive" not in prose
    assert "strong statistical evidence" not in prose
    # Should contain descriptive factual statements
    assert "does not include zero" not in prose
    assert "posterior probability of an increase" in prose
    assert "We recommend" in prose


def test_generate_prose_detailed_ols_is_descriptive():
    """Ensure OLS prose uses descriptive language, not prescriptive judgments."""
    from causalpy.reporting import _generate_prose_detailed_ols

    stats = {
        "avg": {
            "mean": 0.5,
            "ci_lower": -0.2,
            "ci_upper": 1.2,
            "p_value": 0.15,
        }
    }

    window_coords = pd.Index([10, 11, 12])

    prose = _generate_prose_detailed_ols(
        stats,
        window_coords,
        alpha=0.05,
        cumulative=False,
        relative=False,
        observed_avg=50.5,
        counterfactual_avg=50.0,
    )

    # Should NOT contain prescriptive language
    assert "statistically significant" not in prose
    assert "caution is warranted" not in prose
    assert "lack of statistical significance" not in prose
    # Should contain descriptive factual statements
    assert "includes zero" in prose
    assert "p-value" in prose
    assert "We recommend" in prose


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


def test_generate_prose_detailed_ols_basic():
    """Test _generate_prose_detailed_ols generates proper text."""
    from causalpy.reporting import _generate_prose_detailed_ols

    stats = {
        "avg": {
            "mean": 2.5,
            "ci_lower": 1.0,
            "ci_upper": 4.0,
            "p_value": 0.03,
        }
    }

    window_coords = pd.Index([10, 11, 12])

    prose = _generate_prose_detailed_ols(
        stats,
        window_coords,
        alpha=0.05,
        cumulative=False,
        relative=False,
        observed_avg=52.5,
        counterfactual_avg=50.0,
    )

    assert "Post-period" in prose
    assert "10 to 12" in prose
    assert "52.50" in prose
    assert "50.00" in prose
    assert "2.50" in prose
    assert "95% confidence interval" in prose
    assert "does not include zero" in prose


def test_generate_prose_detailed_ols_counterfactual_interval():
    """Test OLS counterfactual interval brackets counterfactual mean."""
    from causalpy.reporting import _generate_prose_detailed_ols

    stats = {
        "avg": {
            "mean": -1.5,
            "ci_lower": -2.5,
            "ci_upper": -0.5,
            "p_value": 0.01,
        }
    }

    window_coords = pd.Index([1, 2, 3])

    prose = _generate_prose_detailed_ols(
        stats,
        window_coords,
        alpha=0.05,
        cumulative=False,
        relative=False,
        observed_avg=18.5,
        counterfactual_avg=20.0,
    )

    # cf_interval_lower = 18.5 - (-0.5) = 19.0
    # cf_interval_upper = 18.5 - (-2.5) = 21.0
    assert "19.00" in prose
    assert "21.00" in prose
    assert "20.00" in prose


def test_generate_prose_detailed_ols_with_cumulative():
    """Test _generate_prose_detailed_ols includes cumulative effect text."""
    from causalpy.reporting import _generate_prose_detailed_ols

    stats = {
        "avg": {
            "mean": 2.5,
            "ci_lower": 1.0,
            "ci_upper": 4.0,
            "p_value": 0.03,
        },
        "cum": {
            "mean": 25.0,
            "ci_lower": 10.0,
            "ci_upper": 40.0,
        },
    }

    window_coords = pd.Index(range(10, 20))

    prose = _generate_prose_detailed_ols(
        stats,
        window_coords,
        alpha=0.05,
        cumulative=True,
        relative=False,
        observed_avg=52.5,
        counterfactual_avg=50.0,
        observed_cum=525.0,
        counterfactual_cum=500.0,
    )

    assert "Summing up" in prose
    assert "525.00" in prose
    assert "500.00" in prose


# ==============================================================================
# Tests for _assumptions_text
# ==============================================================================


def test_assumptions_text_its():
    """Test ITS-specific assumptions text."""
    from causalpy.reporting import _assumptions_text

    text = _assumptions_text("its")
    assert "time-based predictors" in text
    assert "external covariates" in text


def test_assumptions_text_sc():
    """Test SC-specific assumptions text."""
    from causalpy.reporting import _assumptions_text

    text = _assumptions_text("sc")
    assert "control units" in text
    assert "synthetic counterfactual" in text


def test_assumptions_text_default():
    """Test default assumptions text."""
    from causalpy.reporting import _assumptions_text

    text = _assumptions_text(None)
    assert "covariates" in text
    assert "pre-intervention period" in text

    text_piecewise = _assumptions_text("piecewise_its")
    assert text_piecewise == text


# ==============================================================================
# Tests for prose without observed/counterfactual values
# ==============================================================================


def test_prose_detailed_no_observed_values():
    """Test fallback prose when observed/counterfactual are not provided."""
    from causalpy.reporting import _generate_prose_detailed

    stats = {
        "avg": {
            "mean": 2.5,
            "hdi_lower": 1.0,
            "hdi_upper": 4.0,
            "p_gt_0": 0.99,
            "decision": _fixed_bayesian_decision((1.0, 4.0), "increase", 0.99),
        }
    }

    window_coords = pd.Index([10, 11, 12])

    prose = _generate_prose_detailed(
        stats,
        window_coords,
        alpha=0.05,
        direction="increase",
        cumulative=False,
        relative=False,
    )

    assert "estimated average causal effect" in prose
    assert "2.50" in prose


def test_prose_detailed_ols_no_observed_values():
    """Test OLS fallback prose when observed/counterfactual are not provided."""
    from causalpy.reporting import _generate_prose_detailed_ols

    stats = {
        "avg": {
            "mean": 2.5,
            "ci_lower": 1.0,
            "ci_upper": 4.0,
            "p_value": 0.03,
        }
    }

    window_coords = pd.Index([10, 11, 12])

    prose = _generate_prose_detailed_ols(
        stats,
        window_coords,
        alpha=0.05,
        cumulative=False,
        relative=False,
    )

    assert "estimated average causal effect" in prose
    assert "2.50" in prose


# ==============================================================================
# Integration tests for PrePostNEGD experiment
# ==============================================================================


@pytest.mark.integration
def test_effect_summary_prepostnegd_pymc(mock_pymc_sample, anova1_data):
    """Test effect_summary with PrePostNEGD experiment (PyMC)."""
    df = anova1_data
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
def test_effect_summary_prepostnegd_directions(mock_pymc_sample, anova1_data):
    """Test effect_summary with PrePostNEGD with different directions."""
    df = anova1_data
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
def test_effect_summary_prepostnegd_rope(mock_pymc_sample, anova1_data):
    """Test effect_summary with PrePostNEGD with ROPE."""
    df = anova1_data
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
def test_effect_summary_its_relative_false(mock_pymc_sample, its_data):
    """Test effect_summary with ITS and relative=False."""
    df = its_data
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
def test_effect_summary_ols_cumulative_false(mock_pymc_sample, its_data):
    """Test effect_summary with OLS model and cumulative=False."""
    from sklearn.linear_model import LinearRegression

    df = its_data
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
def test_effect_summary_ols_relative_false(mock_pymc_sample, its_data):
    """Test effect_summary with OLS model and relative=False."""
    from sklearn.linear_model import LinearRegression

    df = its_data
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
def test_effect_summary_rope_with_two_sided_its(mock_pymc_sample, its_data):
    """Test effect_summary with ROPE and two-sided direction for ITS."""
    df = its_data
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
def test_effect_summary_rope_with_two_sided_did(mock_pymc_sample, did_data):
    """Test effect_summary with ROPE and two-sided direction for DiD."""
    df = did_data
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
def test_effect_summary_rd_two_sided_with_rope(mock_pymc_sample, rd_data):
    """Test effect_summary with RD, two-sided direction, and ROPE."""
    df = rd_data
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
def test_effect_summary_sc_cumulative_false(mock_pymc_sample, sc_data):
    """Test effect_summary with Synthetic Control and cumulative=False."""
    df = sc_data
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
def test_effect_summary_sc_relative_false(mock_pymc_sample, sc_data):
    """Test effect_summary with Synthetic Control and relative=False."""
    df = sc_data
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
def test_effect_summary_ols_both_false(mock_pymc_sample, its_data):
    """Test effect_summary with OLS model, cumulative=False and relative=False."""
    from sklearn.linear_model import LinearRegression

    df = its_data
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
def test_effect_summary_pymc_both_false(mock_pymc_sample, its_data):
    """Test effect_summary with PyMC model, cumulative=False and relative=False."""
    df = its_data
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
