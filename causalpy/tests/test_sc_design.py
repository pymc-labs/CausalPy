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

import numpy as np
import pandas as pd
import pytest
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
