#   Copyright 2025 - 2026 The PyMC Labs Developers
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
Integration tests for SyntheticControl design methods:
validate_design(), power_analysis(), and donor_pool_quality().
"""

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import xarray as xr

import causalpy as cp
from causalpy.checks.dress_rehearsal import DressRehearsalCheck
from causalpy.experiments.sc_results import (
    DonorPoolQualityResult,
    DressRehearsalResult,
    PowerCurveResult,
)

sample_kwargs = {
    "tune": 20,
    "draws": 20,
    "chains": 2,
    "cores": 2,
    "progressbar": False,
}


CONTROL_UNITS = ["a", "b", "c", "d", "e", "f", "g"]
TREATED_UNITS = ["actual"]


@pytest.fixture(scope="module")
def fitted_sc():
    """A fitted SyntheticControl experiment using the built-in 'sc' dataset."""
    df = cp.load_data("sc")
    treatment_time = 70
    result = cp.SyntheticControl(
        df,
        treatment_time,
        control_units=CONTROL_UNITS,
        treated_units=TREATED_UNITS,
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )
    return result


@pytest.fixture(scope="module")
def design_sc():
    """A design-only SC created with from_pre_period (no post-period data)."""
    df = cp.load_data("sc")
    pre_data = df[df.index < 70]
    return cp.SyntheticControl.from_pre_period(
        pre_data,
        control_units=CONTROL_UNITS,
        treated_units=TREATED_UNITS,
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )


class TestFromPrePeriod:
    """Tests for the prospective workflow via from_pre_period."""

    @pytest.mark.integration
    def test_from_pre_period_creates_design_instance(self, design_sc):
        assert design_sc._design_only is True
        assert hasattr(design_sc, "pre_pred")
        assert hasattr(design_sc, "pre_impact")
        assert not hasattr(design_sc, "post_pred")
        assert not hasattr(design_sc, "post_impact")

    @pytest.mark.integration
    def test_from_pre_period_uses_all_data_as_pre(self, design_sc):
        df = cp.load_data("sc")
        pre_data = df[df.index < 70]
        assert len(design_sc.datapre) == len(pre_data)
        assert len(design_sc.datapost) == 0

    @pytest.mark.integration
    def test_from_pre_period_validate_design(self, design_sc):
        result = design_sc.validate_design(
            injected_effect=0.15,
            sample_kwargs=sample_kwargs,
        )
        assert isinstance(result, DressRehearsalResult)
        assert isinstance(result.recovered_effect_mean, float)

    @pytest.mark.integration
    def test_from_pre_period_donor_pool_quality(self, design_sc):
        result = design_sc.donor_pool_quality()
        assert isinstance(result, DonorPoolQualityResult)
        assert 0 <= result.convex_hull_coverage <= 1

    @pytest.mark.integration
    def test_from_pre_period_power_analysis(self, design_sc):
        result = design_sc.power_analysis(
            effect_sizes=[0.15],
            n_simulations=2,
            sample_kwargs=sample_kwargs,
            random_seed=42,
        )
        assert isinstance(result, PowerCurveResult)
        assert len(result.detection_rates) == 1

    @pytest.mark.integration
    def test_from_pre_period_with_datetime_index(self):
        df = cp.load_data("sc")
        df.index = pd.date_range("2020-01-01", periods=len(df), freq="D")
        pre_data = df.iloc[:70]
        design = cp.SyntheticControl.from_pre_period(
            pre_data,
            control_units=CONTROL_UNITS,
            treated_units=TREATED_UNITS,
            model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
        )
        assert design._design_only is True
        assert isinstance(design.treatment_time, pd.Timestamp)


class TestValidateDesign:
    """Tests for SyntheticControl.validate_design()."""

    @pytest.mark.integration
    def test_validate_design_runs_end_to_end(self, fitted_sc):
        result = fitted_sc.validate_design(
            injected_effect=0.15,
            sample_kwargs=sample_kwargs,
        )
        assert isinstance(result, DressRehearsalResult)
        assert result.injected_effect == 0.15
        assert result.effect_type == "relative"
        assert isinstance(result.recovered_effect_mean, float)
        assert len(result.recovered_effect_hdi) == 2
        assert isinstance(result.hdi_covers_truth, bool)
        assert isinstance(result.posterior_samples, xr.DataArray)

    @pytest.mark.integration
    def test_validate_design_absolute_effect(self, fitted_sc):
        result = fitted_sc.validate_design(
            injected_effect=5.0,
            effect_type="absolute",
            sample_kwargs=sample_kwargs,
        )
        assert result.effect_type == "absolute"
        assert isinstance(result.recovered_effect_mean, float)

    @pytest.mark.integration
    def test_validate_design_custom_holdout(self, fitted_sc):
        result = fitted_sc.validate_design(
            injected_effect=0.10,
            holdout_periods=10,
            sample_kwargs=sample_kwargs,
        )
        assert isinstance(result, DressRehearsalResult)

    @pytest.mark.integration
    def test_validate_design_plot(self, fitted_sc):
        result = fitted_sc.validate_design(
            injected_effect=0.15,
            sample_kwargs=sample_kwargs,
        )
        fig, ax = result.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    @pytest.mark.integration
    def test_validate_design_summary(self, fitted_sc):
        result = fitted_sc.validate_design(
            injected_effect=0.15,
            sample_kwargs=sample_kwargs,
        )
        df = result.summary()
        assert isinstance(df, pd.DataFrame)
        assert "injected_effect" in df.columns
        assert "recovered_mean" in df.columns
        assert "hdi_covers_truth" in df.columns

    @pytest.mark.integration
    def test_validate_design_to_check_result(self, fitted_sc):
        result = fitted_sc.validate_design(
            injected_effect=0.15,
            sample_kwargs=sample_kwargs,
        )
        check_result = result.to_check_result()
        assert check_result.check_name == "DressRehearsal"
        assert isinstance(check_result.passed, bool)
        assert check_result.table is not None
        assert len(check_result.text) > 0

    def test_validate_design_holdout_too_large(self, fitted_sc):
        with pytest.raises(ValueError, match="need >= 10"):
            fitted_sc.validate_design(
                injected_effect=0.15,
                holdout_periods=65,
                sample_kwargs=sample_kwargs,
            )

    def test_validate_design_requires_pymc(self):
        df = cp.load_data("sc")
        from sklearn.linear_model import LinearRegression

        result = cp.SyntheticControl(
            df,
            70,
            control_units=CONTROL_UNITS,
            treated_units=TREATED_UNITS,
            model=LinearRegression(),
        )
        with pytest.raises(TypeError, match="PyMC model"):
            result.validate_design(injected_effect=0.15)


class TestDressRehearsalCheck:
    """Tests for the DressRehearsalCheck pipeline wrapper."""

    @pytest.mark.integration
    def test_check_runs_via_wrapper(self, fitted_sc):
        check = DressRehearsalCheck(
            injected_effect=0.10,
            sample_kwargs=sample_kwargs,
        )
        from causalpy.pipeline import PipelineContext

        ctx = PipelineContext(data=fitted_sc.data)
        ctx.experiment = fitted_sc
        check_result = check.run(fitted_sc, ctx)
        assert check_result.check_name == "DressRehearsal"
        assert isinstance(check_result.passed, bool)

    def test_check_validate_wrong_type(self):
        check = DressRehearsalCheck()
        with pytest.raises(TypeError, match="SyntheticControl"):
            check.validate("not an experiment")


class TestPowerAnalysis:
    """Tests for SyntheticControl.power_analysis()."""

    @pytest.mark.integration
    def test_power_analysis_runs_end_to_end(self, fitted_sc):
        result = fitted_sc.power_analysis(
            effect_sizes=[0.10, 0.20],
            n_simulations=2,
            sample_kwargs=sample_kwargs,
            random_seed=42,
        )
        assert isinstance(result, PowerCurveResult)
        assert len(result.effect_sizes) == 2
        assert len(result.detection_rates) == 2
        assert all(0 <= r <= 1 for r in result.detection_rates)
        assert len(result.raw_results) == 2
        assert len(result.raw_results[0]) == 2

    @pytest.mark.integration
    def test_power_analysis_plot(self, fitted_sc):
        result = fitted_sc.power_analysis(
            effect_sizes=[0.10],
            n_simulations=2,
            sample_kwargs=sample_kwargs,
            random_seed=42,
        )
        fig, ax = result.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    @pytest.mark.integration
    def test_power_analysis_summary(self, fitted_sc):
        result = fitted_sc.power_analysis(
            effect_sizes=[0.10],
            n_simulations=2,
            sample_kwargs=sample_kwargs,
            random_seed=42,
        )
        df = result.summary()
        assert isinstance(df, pd.DataFrame)
        assert "effect_size" in df.columns
        assert "detection_rate" in df.columns
        assert "n_simulations" in df.columns

    @pytest.mark.integration
    def test_power_analysis_reproducible(self, fitted_sc):
        kwargs = {
            "effect_sizes": [0.10],
            "n_simulations": 2,
            "sample_kwargs": sample_kwargs,
            "random_seed": 42,
        }
        r1 = fitted_sc.power_analysis(**kwargs)
        r2 = fitted_sc.power_analysis(**kwargs)
        assert r1.detection_rates == r2.detection_rates

    def test_power_analysis_requires_pymc(self):
        df = cp.load_data("sc")
        from sklearn.linear_model import LinearRegression

        result = cp.SyntheticControl(
            df,
            70,
            control_units=CONTROL_UNITS,
            treated_units=TREATED_UNITS,
            model=LinearRegression(),
        )
        with pytest.raises(TypeError, match="PyMC model"):
            result.power_analysis(effect_sizes=[0.10])


class TestDonorPoolQuality:
    """Tests for SyntheticControl.donor_pool_quality()."""

    @pytest.mark.integration
    def test_donor_pool_quality_runs(self, fitted_sc):
        result = fitted_sc.donor_pool_quality()
        assert isinstance(result, DonorPoolQualityResult)
        assert isinstance(result.correlation_score, float)
        assert isinstance(result.convex_hull_coverage, float)
        assert isinstance(result.weight_concentration, float)
        assert isinstance(result.per_donor_details, pd.DataFrame)

    @pytest.mark.integration
    def test_donor_pool_quality_summary(self, fitted_sc):
        result = fitted_sc.donor_pool_quality()
        df = result.summary()
        assert isinstance(df, pd.DataFrame)
        assert "metric" in df.columns
        assert "assessment" in df.columns
        assert len(df) == 4

    @pytest.mark.integration
    def test_donor_pool_quality_per_donor(self, fitted_sc):
        result = fitted_sc.donor_pool_quality()
        details = result.per_donor_details
        assert len(details) == len(fitted_sc.control_units)
        assert "donor" in details.columns
        assert "mean_correlation" in details.columns
        assert "mean_weight" in details.columns

    @pytest.mark.integration
    def test_donor_pool_quality_coverage_range(self, fitted_sc):
        result = fitted_sc.donor_pool_quality()
        assert 0 <= result.convex_hull_coverage <= 1

    @pytest.mark.integration
    def test_donor_pool_quality_weight_concentration(self, fitted_sc):
        result = fitted_sc.donor_pool_quality()
        assert result.weight_concentration >= 1.0

    def test_donor_pool_quality_requires_pymc(self):
        df = cp.load_data("sc")
        from sklearn.linear_model import LinearRegression

        result = cp.SyntheticControl(
            df,
            70,
            control_units=CONTROL_UNITS,
            treated_units=TREATED_UNITS,
            model=LinearRegression(),
        )
        with pytest.raises(TypeError, match="PyMC model"):
            result.donor_pool_quality()
