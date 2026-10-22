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
"""Tests for SyntheticControl design-phase methods."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.checks import DressRehearsalCheck
from causalpy.experiments.sc_results import (
    DonorPoolQualityResult,
    DressRehearsalResult,
    PowerCurveResult,
)

CONTROL_UNITS = ["a", "b", "c", "d", "e", "f", "g"]
TREATED_UNITS = ["actual"]
SAMPLE_KWARGS = {
    "tune": 20,
    "draws": 20,
    "chains": 1,
    "cores": 1,
    "progressbar": False,
    "random_seed": 42,
}


def _tiny_panel(n: int = 16) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    donors = rng.normal(size=(n, 3))
    treated = donors @ np.array([0.5, 0.3, 0.2])
    return pd.DataFrame(
        np.column_stack([treated, donors]),
        index=index,
        columns=["y", "a", "b", "c"],
    )


@pytest.fixture(scope="module")
def design_sc():
    """A fitted design-only PyMC synthetic control on the first 40 sc rows."""
    frame = cp.load_data("sc")
    frame.index = pd.date_range("2020-01-01", periods=len(frame), freq="D")
    return cp.SyntheticControl.from_pre_period(
        frame.iloc[:40],
        CONTROL_UNITS,
        TREATED_UNITS,
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=SAMPLE_KWARGS),
    )


def test_from_pre_period_fits_ols_without_a_post_window():
    design = cp.SyntheticControl.from_pre_period(
        _tiny_panel(),
        ["a", "b", "c"],
        ["y"],
        model=LinearRegression(),
    )

    assert len(design.datapost) == 0
    assert design.result.impact_pre.sizes["obs_ind"] == 16
    assert design.result.impact_post.sizes["obs_ind"] == 0
    assert design.treatment_time > design.datapre.index.max()
    design.summary()
    with pytest.raises(ValueError, match="design-only"):
        design.plot()
    with pytest.raises(ValueError, match="design-only"):
        design.effect_summary()
    with pytest.raises(ValueError, match="design-only"):
        design.get_plot_data()


def test_from_pre_period_accepts_a_numeric_index_and_time_column():
    frame = _tiny_panel().reset_index(names="date")
    frame.index = np.arange(len(frame))
    design = cp.SyntheticControl.from_pre_period(
        frame.drop(columns="date"),
        ["a", "b", "c"],
        ["y"],
        model=LinearRegression(),
    )
    assert design.treatment_time == len(frame)

    dated = cp.SyntheticControl.from_pre_period(
        frame.reset_index(drop=True),
        ["a", "b", "c"],
        ["y"],
        model=LinearRegression(),
        time_column="date",
    )
    assert isinstance(dated.datapre.index, pd.DatetimeIndex)


def test_from_pre_period_rejects_an_index_with_no_pseudo_time():
    frame = _tiny_panel()
    frame.index = [f"row-{i}" for i in range(len(frame))]
    with pytest.raises(TypeError, match="DatetimeIndex or a numeric index"):
        cp.SyntheticControl.from_pre_period(
            frame,
            ["a", "b", "c"],
            ["y"],
            model=LinearRegression(),
        )


@pytest.mark.parametrize(
    "method", ["validate_design", "power_analysis", "donor_pool_quality"]
)
def test_design_methods_reject_an_ols_model(method):
    design = cp.SyntheticControl.from_pre_period(
        _tiny_panel(),
        ["a", "b", "c"],
        ["y"],
        model=LinearRegression(),
    )
    with pytest.raises(TypeError, match="PyMC model"):
        if method == "validate_design":
            design.validate_design(0.1)
        elif method == "power_analysis":
            design.power_analysis([0.1])
        else:
            design.donor_pool_quality()


def test_design_sample_kwargs_merges_parent_settings_and_rejects_non_pymc():
    panel = _tiny_panel()
    treatment_time = panel.index.max() + pd.Timedelta(days=1)
    design = cp.SyntheticControl(
        panel,
        treatment_time,
        ["a", "b", "c"],
        ["y"],
        model=cp.pymc_models.WeightedSumFitter(
            sample_kwargs={"draws": 11, "progressbar": True}
        ),
    )
    merged = design._design_sample_kwargs({"tune": 3})
    assert merged["draws"] == 11
    assert merged["tune"] == 3
    assert merged["progressbar"] is True

    bare = cp.SyntheticControl(
        panel,
        treatment_time,
        ["a", "b", "c"],
        ["y"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs={"draws": 11}),
    )
    assert bare._design_sample_kwargs(None)["progressbar"] is False

    ols = cp.SyntheticControl.from_pre_period(
        panel,
        ["a", "b", "c"],
        ["y"],
        model=LinearRegression(),
    )
    with pytest.raises(TypeError, match="PyMC model"):
        ols._design_sample_kwargs(None)


def test_validate_design_rejects_a_holdout_that_leaves_too_little_pre_period(design_sc):
    with pytest.raises(ValueError, match="need >= 10"):
        design_sc.validate_design(0.1, holdout_periods=35)


def test_validate_design_recovers_a_relative_effect(design_sc):
    result = design_sc.validate_design(0.15, sample_kwargs=SAMPLE_KWARGS)

    assert isinstance(result, DressRehearsalResult)
    assert result.effect_type == "relative"
    assert result.injected_effect == 0.15
    assert np.isfinite(result.recovered_effect_mean)
    assert len(result.recovered_effect_hdi) == 2
    assert isinstance(result.hdi_covers_truth, bool)
    assert result.posterior_samples.sizes["sample"] == 20


def test_validate_design_scores_an_absolute_effect(design_sc):
    result = design_sc.validate_design(
        5.0,
        effect_type="absolute",
        sample_kwargs=SAMPLE_KWARGS,
    )

    assert result.effect_type == "absolute"
    assert result.injected_truth == pytest.approx(5.0 * (40 // 4))


def test_donor_pool_quality_reports_the_pre_period_pool(design_sc):
    result = design_sc.donor_pool_quality()

    assert isinstance(result, DonorPoolQualityResult)
    assert 0 <= result.convex_hull_coverage <= 1
    assert result.weight_concentration >= 1
    assert list(result.per_donor_details["donor"]) == CONTROL_UNITS
    assert "Overall quality" in set(result.summary()["metric"])


def test_power_analysis_returns_a_rate_for_each_effect(design_sc):
    result = design_sc.power_analysis(
        [0.1],
        n_simulations=2,
        sample_kwargs=SAMPLE_KWARGS,
        random_seed=1,
    )

    assert isinstance(result, PowerCurveResult)
    assert result.effect_sizes == [0.1]
    assert len(result.detection_rates) == 1
    assert 0 <= result.detection_rates[0] <= 1
    assert len(result.raw_results[0]) == 2


def test_power_analysis_is_reproducible_for_a_fixed_seed(design_sc):
    kwargs = {"sample_kwargs": SAMPLE_KWARGS, "random_seed": 7, "n_simulations": 1}
    first = design_sc.power_analysis([0.2], **kwargs)
    second = design_sc.power_analysis([0.2], **kwargs)

    assert first.detection_rates == second.detection_rates
    assert first.raw_results[0][0].recovered_effect_mean == pytest.approx(
        second.raw_results[0][0].recovered_effect_mean
    )


def test_power_analysis_rejects_an_unknown_criterion_before_fitting(design_sc):
    with pytest.raises(ValueError, match="Unknown criterion"):
        design_sc.power_analysis([0.1], criterion="not-a-rule")


def test_dress_rehearsal_check_returns_a_check_result(design_sc):
    check = DressRehearsalCheck(injected_effect=0.1, sample_kwargs=SAMPLE_KWARGS)
    result = check.run(design_sc, context=None)

    assert result.check_name == "DressRehearsal"
    assert result.passed is not None
    assert "effect" in result.text
    assert "dress_rehearsal_result" in result.metadata


def test_plot_raises_on_a_design_only_pymc_fit(design_sc):
    with pytest.raises(ValueError, match="design-only"):
        design_sc.plot()


def _rehearsal(
    samples: list[float],
    *,
    recovered: float,
    truth: float,
) -> DressRehearsalResult:
    return DressRehearsalResult(
        injected_effect=0.1,
        effect_type="relative",
        recovered_effect_mean=recovered,
        recovered_effect_hdi=(min(samples), max(samples)),
        hdi_covers_truth=min(samples) <= truth <= max(samples),
        posterior_samples=xr.DataArray(samples, dims="sample"),
        injected_truth=truth,
    )


def _quality(
    correlation: float, coverage: float, weight: float
) -> DonorPoolQualityResult:
    return DonorPoolQualityResult(
        correlation_score=correlation,
        convex_hull_coverage=coverage,
        weight_concentration=weight,
        per_donor_details=pd.DataFrame({"donor": ["a"]}),
    )


def _overall(result: DonorPoolQualityResult) -> str:
    table = result.summary()
    return table.loc[table["metric"] == "Overall quality", "assessment"].item()


def test_power_curve_plot_and_summary_report_each_effect():
    curve = PowerCurveResult(
        effect_sizes=[0.1, 0.2],
        detection_rates=[0.0, 1.0],
        criterion="hdi_excludes_zero",
        raw_results=[
            [_rehearsal([-1.0, 0.5], recovered=-0.25, truth=1.0)],
            [_rehearsal([1.0, 2.0], recovered=1.5, truth=1.0)],
        ],
    )
    empty = PowerCurveResult(
        effect_sizes=[],
        detection_rates=[],
        criterion="hdi_excludes_zero",
        raw_results=[],
    )

    fig, ax = curve.plot()
    empty_fig, empty_ax = empty.plot()
    summary = curve.summary()
    plt.close(fig)
    plt.close(empty_fig)

    assert "n=1" in ax.get_title()
    assert "n=0" in empty_ax.get_title()
    assert list(summary["effect_size"]) == [0.1, 0.2]
    assert list(summary["detection_rate"]) == [0.0, 1.0]
    assert summary["mean_recovery"].tolist() == pytest.approx([-0.25, 1.5])
    assert summary["median_recovery"].tolist() == pytest.approx([-0.25, 1.5])
    assert summary["n_simulations"].tolist() == [1, 1]
    assert empty.summary().empty


def test_donor_pool_quality_labels_span_the_three_bands():
    assert _overall(_quality(0.1, 1.0, 5.0)) == "poor"
    assert _overall(_quality(0.9, 1.0, 4.0)) == "good"
    assert _overall(_quality(0.7, 0.9, 2.0)) == "acceptable"


def test_dress_rehearsal_validate_rejects_the_wrong_experiment_and_model():
    check = DressRehearsalCheck()
    with pytest.raises(TypeError, match="SyntheticControl experiment"):
        check.validate(object())

    design = cp.SyntheticControl.from_pre_period(
        _tiny_panel(),
        ["a", "b", "c"],
        ["y"],
        model=LinearRegression(),
    )
    with pytest.raises(TypeError, match="PyMC model"):
        check.validate(design)


def test_design_methods_reject_bad_controls_before_sampling(design_sc):
    with pytest.raises(ValueError, match="effect_type"):
        design_sc.validate_design(0.1, effect_type="nope")
    with pytest.raises(ValueError, match="n_simulations"):
        design_sc.power_analysis([0.1], n_simulations=0)
    with pytest.raises(ValueError, match="effect_type"):
        design_sc.power_analysis([0.1], effect_type="nope")


def test_criterion_met_applies_each_detection_rule():
    positive = _rehearsal([1.0, 2.0, 3.0], recovered=2.0, truth=2.0)
    mixed = _rehearsal([-1.0, 1.0], recovered=0.0, truth=2.0)
    exact_zero = _rehearsal([0.0, 0.0], recovered=0.0, truth=0.0)
    nonzero_at_zero_truth = _rehearsal([1.0], recovered=1.0, truth=0.0)
    far = _rehearsal([1.0, 1.0], recovered=1.0, truth=2.0)

    assert cp.SyntheticControl._criterion_met(positive, "prob_gt_zero")
    assert not cp.SyntheticControl._criterion_met(mixed, "prob_gt_zero")
    assert cp.SyntheticControl._criterion_met(exact_zero, "recovery_accuracy")
    assert not cp.SyntheticControl._criterion_met(
        nonzero_at_zero_truth, "recovery_accuracy"
    )
    assert cp.SyntheticControl._criterion_met(positive, "recovery_accuracy")
    assert not cp.SyntheticControl._criterion_met(far, "recovery_accuracy")
    with pytest.raises(ValueError, match="Unknown criterion"):
        cp.SyntheticControl._criterion_met(positive, "not-a-rule")


def test_donor_pool_quality_requires_a_fitted_model():
    model = cp.pymc_models.WeightedSumFitter(sample_kwargs=SAMPLE_KWARGS)
    experiment = cp.SyntheticControl.__new__(cp.SyntheticControl)
    experiment.model = model

    with pytest.raises(RuntimeError, match="Call fit"):
        experiment.donor_pool_quality()


def _quality_with_beta(design_sc, beta: xr.DataArray) -> DonorPoolQualityResult:
    idata = design_sc.model.idata.copy()
    posterior = idata.posterior.copy()
    posterior["beta"] = beta
    idata.posterior = posterior
    model = cp.pymc_models.WeightedSumFitter(sample_kwargs=SAMPLE_KWARGS)
    model.idata = idata
    shell = type("Shell", (), {})()
    shell.model = model
    shell.datapre = design_sc.datapre
    shell.control_units = list(design_sc.control_units)
    shell.treated_units = list(design_sc.treated_units)
    shell.pre_design = design_sc.pre_design
    return cp.SyntheticControl.donor_pool_quality(shell)


def test_donor_pool_quality_handles_zero_and_flat_weights(design_sc):
    beta = design_sc.model.idata.posterior["beta"]
    zeroed = _quality_with_beta(design_sc, xr.zeros_like(beta))
    flat_beta = beta.isel(treated_units=0) if "treated_units" in beta.dims else beta
    flat = _quality_with_beta(design_sc, flat_beta)

    n_donors = len(design_sc.control_units)
    assert zeroed.per_donor_details["normalized_weight"].tolist() == pytest.approx(
        [1 / n_donors] * n_donors
    )
    reduced = flat_beta.mean(dim=["chain", "draw"])
    expected = np.abs(np.asarray(reduced.values, dtype=float).ravel())
    expected = expected / expected.sum()
    assert flat.per_donor_details["normalized_weight"].to_numpy() == pytest.approx(
        expected
    )


def test_validate_design_scores_impact_without_a_treated_unit_dimension(
    design_sc, monkeypatch
):
    values = np.zeros((1, 8, 4))
    values[..., 0] = -1.0
    values[..., -1] = 3.0
    impact = xr.DataArray(values, dims=("chain", "draw", "obs_ind"))

    def fake_fit(self, *args, **kwargs):
        self._result = type("Result", (), {"impact_post_cumulative": impact})()
        return self

    monkeypatch.setattr(cp.SyntheticControl, "fit", fake_fit)
    result = design_sc.validate_design(
        0.1, holdout_periods=10, sample_kwargs=SAMPLE_KWARGS
    )

    assert result.recovered_effect_mean == pytest.approx(3.0)
    assert result.effect_type == "relative"


def test_power_analysis_counts_a_miss_without_a_residual_unit_dimension(
    design_sc, monkeypatch
):
    residuals = design_sc.result.impact_pre
    if "treated_units" in residuals.dims:
        residuals = residuals.isel(treated_units=0)
    original = design_sc._result
    design_sc._result = type("Result", (), {"impact_pre": residuals})()
    rehearsal = _rehearsal([-1.0, 1.0], recovered=0.0, truth=2.0)
    monkeypatch.setattr(
        cp.SyntheticControl,
        "_build_noisy_experiment",
        lambda self, *args, **kwargs: design_sc,
    )
    monkeypatch.setattr(
        cp.SyntheticControl,
        "validate_design",
        lambda self, *args, **kwargs: rehearsal,
    )

    try:
        result = design_sc.power_analysis(
            [0.1], n_simulations=1, sample_kwargs=SAMPLE_KWARGS
        )
    finally:
        design_sc._result = original

    assert result.detection_rates == [0.0]


def test_dress_rehearsal_check_text_names_a_hit_and_a_miss():
    hit = _rehearsal([1.0, 2.0], recovered=1.5, truth=1.5)
    miss = _rehearsal([1.0, 2.0], recovered=1.5, truth=9.0)
    hit_check = hit.to_check_result()
    miss_check = miss.to_check_result()
    plt.close(hit_check.figures[0])
    plt.close(miss_check.figures[0])

    assert hit_check.passed is True
    assert "HDI covers" in hit_check.text
    assert miss_check.passed is False
    assert "HDI misses" in miss_check.text


def test_power_analysis_logs_every_fifth_simulation(design_sc, monkeypatch, caplog):
    rehearsal = _rehearsal([1.0, 2.0], recovered=1.5, truth=1.0)
    monkeypatch.setattr(
        cp.SyntheticControl,
        "_build_noisy_experiment",
        lambda self, *args, **kwargs: design_sc,
    )
    monkeypatch.setattr(
        cp.SyntheticControl,
        "validate_design",
        lambda self, *args, **kwargs: rehearsal,
    )

    with caplog.at_level(logging.INFO, logger="causalpy.experiments.synthetic_control"):
        result = design_sc.power_analysis(
            [0.1, 0.2], n_simulations=5, sample_kwargs=SAMPLE_KWARGS
        )

    assert "5/10 simulations complete" in caplog.text
    assert "10/10 simulations complete" in caplog.text
    assert "1/10 simulations complete" not in caplog.text
    assert result.detection_rates == [1.0, 1.0]


def test_donor_pool_quality_rejects_idata_without_a_posterior():
    model = cp.pymc_models.WeightedSumFitter(sample_kwargs=SAMPLE_KWARGS)
    model.idata = xr.DataTree()
    shell = type("Shell", (), {"model": model})()

    with pytest.raises(RuntimeError, match="Call fit"):
        cp.SyntheticControl.donor_pool_quality(shell)


def test_donor_pool_quality_omits_undefined_correlations_from_the_score(design_sc):
    frame = design_sc.datapre.copy()
    for unit in design_sc.control_units:
        frame[unit] = 1.0
    beta = design_sc.model.idata.posterior["beta"]
    idata = design_sc.model.idata.copy()
    posterior = idata.posterior.copy()
    posterior["beta"] = beta
    idata.posterior = posterior
    model = cp.pymc_models.WeightedSumFitter(sample_kwargs=SAMPLE_KWARGS)
    model.idata = idata
    shell = type("Shell", (), {})()
    shell.model = model
    shell.datapre = frame
    shell.control_units = list(design_sc.control_units)
    shell.treated_units = list(design_sc.treated_units)
    shell.pre_design = design_sc.pre_design

    result = cp.SyntheticControl.donor_pool_quality(shell)

    assert np.isnan(result.correlation_score)
    assert result.per_donor_details["mean_correlation"].isna().all()
    assert list(result.per_donor_details["donor"]) == list(design_sc.control_units)
