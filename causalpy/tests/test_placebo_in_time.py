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
"""Tests for PlaceboInTime hierarchical null model."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.checks.base import Check, CheckResult
from causalpy.checks.placebo_in_time import (
    AssuranceResult,
    PlaceboFoldResult,
    PlaceboInTime,
)
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.pipeline import Pipeline, PipelineContext
from causalpy.steps.sensitivity import _DEFAULT_CHECKS, SensitivityAnalysis

# ---------------------------------------------------------------------------
# Shared sample_kwargs for fast PyMC tests
# ---------------------------------------------------------------------------

_FAST_SAMPLE_KWARGS = {
    "chains": 2,
    "draws": 100,
    "progressbar": False,
    "random_seed": 42,
}

_FAST_HIERARCHICAL_KWARGS = {
    "chains": 2,
    "draws": 50,
    "tune": 50,
    "progressbar": False,
    "random_seed": 42,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_its_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a simple ITS dataset with numeric index and no real effect."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "t": np.arange(n),
            "y": rng.normal(size=n),
        }
    )


def _make_pymc_model():
    """Create a minimal PyMC model for ITS tests."""
    return cp.pymc_models.LinearRegression(sample_kwargs=_FAST_SAMPLE_KWARGS)


def _make_ols_model():
    """Create a minimal OLS-compatible model for validation tests."""
    return cp.create_causalpy_compatible_class(LinearRegression())


def _make_pymc_factory():
    """Factory that creates PyMC ITS experiments."""

    def factory(data, treatment_time):
        """Test factory."""
        return InterruptedTimeSeries(
            data,
            treatment_time=treatment_time,
            formula="y ~ 1 + t",
            model=_make_pymc_model(),
        )

    return factory


def _make_fake_bayesian_experiment(
    data: pd.DataFrame, treatment_time: int
) -> SimpleNamespace:
    """Create a lightweight Bayesian-like experiment for no-sampling checks."""
    n_post = max(int((data.index >= treatment_time).sum()), 1)
    post_impact = xr.DataArray(
        np.ones((1, 2, n_post, 1)),
        dims=("chain", "draw", "obs_ind", "treated_units"),
    )
    return SimpleNamespace(
        data=data,
        treatment_time=treatment_time,
        _model_backend=SimpleNamespace(supports_idata=True),
        model=SimpleNamespace(),
        post_impact=post_impact,
    )


def _fake_status_quo_result(
    fold_means: np.ndarray, fold_sds: np.ndarray
) -> tuple[SimpleNamespace, np.ndarray]:
    """Return deterministic hierarchical outputs without sampling."""
    del fold_means, fold_sds
    return (
        SimpleNamespace(
            posterior={
                "mu_status_quo": xr.DataArray([0.0]),
                "tau_status_quo": xr.DataArray([1.0]),
            }
        ),
        np.array([-1.0, 1.0]),
    )


# ===========================================================================
# Construction tests (unit — no sampling)
# ===========================================================================


def test_default_n_folds():
    """Test default n folds."""
    check = PlaceboInTime()
    assert check.n_folds == 3


def test_custom_n_folds():
    """Test custom n folds."""
    check = PlaceboInTime(n_folds=5)
    assert check.n_folds == 5


def test_invalid_n_folds():
    """Test invalid n folds."""
    with pytest.raises(ValueError, match="n_folds must be >= 1"):
        PlaceboInTime(n_folds=0)


def test_default_sample_kwargs():
    """Test default sample kwargs."""
    check = PlaceboInTime()
    assert check.sample_kwargs["draws"] == 1000
    assert check.sample_kwargs["chains"] == 4
    assert check.sample_kwargs["target_accept"] == 0.97


def test_custom_sample_kwargs():
    """Test custom sample kwargs."""
    check = PlaceboInTime(sample_kwargs={"draws": 200, "chains": 2})
    assert check.sample_kwargs["draws"] == 200
    assert check.sample_kwargs["chains"] == 2
    assert check.sample_kwargs["target_accept"] == 0.97


@pytest.mark.parametrize(
    ("hierarchical_seed", "master_seed"),
    [(None, 17), (23, 17)],
)
def test_hierarchical_sampling_seed_precedence(
    monkeypatch, hierarchical_seed, master_seed
):
    """The master seed controls PPC while an explicit fit seed takes precedence."""
    sample_calls: list[dict] = []
    posterior_predictive_calls: list[dict] = []

    def fake_sample(**kwargs):
        sample_calls.append(kwargs)
        return object()

    def fake_sample_posterior_predictive(idata, **kwargs):
        del idata
        posterior_predictive_calls.append(kwargs)
        return {
            "posterior_predictive": {
                "theta_new": xr.DataArray(
                    np.array([[[1.0], [2.0]]]),
                    dims=("chain", "draw", "new_period"),
                )
            }
        }

    monkeypatch.setattr(pm, "sample", fake_sample)
    monkeypatch.setattr(
        pm, "sample_posterior_predictive", fake_sample_posterior_predictive
    )
    sample_kwargs = {"draws": 2, "chains": 1}
    if hierarchical_seed is not None:
        sample_kwargs["random_seed"] = hierarchical_seed
    check = PlaceboInTime(
        sample_kwargs=sample_kwargs,
        random_seed=master_seed,
    )

    _, theta_new_samples = check._build_status_quo_model(
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
    )

    expected_hierarchical_seed = (
        master_seed if hierarchical_seed is None else hierarchical_seed
    )
    assert sample_calls[0]["random_seed"] == expected_hierarchical_seed
    assert posterior_predictive_calls == [
        {"var_names": ["theta_new"], "random_seed": master_seed}
    ]
    np.testing.assert_array_equal(theta_new_samples, np.array([1.0, 2.0]))


def test_default_threshold_and_prior_scale():
    """Test default threshold and prior scale."""
    check = PlaceboInTime()
    assert check.threshold == 0.95
    assert check.prior_scale == 1.0


def test_custom_threshold():
    """Test custom threshold."""
    check = PlaceboInTime(threshold=0.99)
    assert check.threshold == 0.99


def test_custom_prior_scale():
    """Test custom prior scale."""
    check = PlaceboInTime(prior_scale=2.0)
    assert check.prior_scale == 2.0


def test_expected_effect_prior_without_rope_raises():
    """Test expected effect prior without rope raises."""
    with pytest.raises(ValueError, match="rope_half_width is required"):
        PlaceboInTime(expected_effect_prior=np.array([1.0, 2.0, 3.0]))


def test_expected_effect_prior_with_rope_ok():
    """Test expected effect prior with rope ok."""
    check = PlaceboInTime(
        expected_effect_prior=np.array([1.0, 2.0, 3.0]),
        rope_half_width=0.5,
    )
    assert check.rope_half_width == 0.5


def test_satisfies_check_protocol():
    """Test satisfies check protocol."""
    assert isinstance(PlaceboInTime(), Check)


def test_applicable_methods():
    """Test applicable methods."""
    check = PlaceboInTime()
    assert InterruptedTimeSeries in check.applicable_methods
    assert cp.SyntheticControl in check.applicable_methods


def test_repr_basic():
    """Test repr basic."""
    assert "n_folds=3" in repr(PlaceboInTime())


def test_repr_with_assurance():
    """Test repr with assurance."""
    check = PlaceboInTime(
        expected_effect_prior=np.array([1.0]),
        rope_half_width=0.5,
    )
    assert "assurance=True" in repr(check)


# ===========================================================================
# Validation tests (unit — no sampling)
# ===========================================================================


@pytest.mark.integration
def test_validate_accepts_pymc_its(mock_pymc_sample):
    """Test validate accepts pymc its."""
    df = _make_its_data()
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=150,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    PlaceboInTime().validate(experiment)


def test_validate_rejects_ols_model():
    """Test validate rejects ols model."""
    df = _make_its_data()
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=150,
        formula="y ~ 1 + t",
        model=_make_ols_model(),
    )
    with pytest.raises(TypeError, match="PyMC model"):
        PlaceboInTime().validate(experiment)


def test_validate_rejects_no_treatment_time():
    """Test validate rejects no treatment time."""

    class _FakeExperiment:
        pass

    with pytest.raises(TypeError, match="treatment_time"):
        PlaceboInTime().validate(_FakeExperiment())


# ===========================================================================
# ROPE decision rule tests (unit — no sampling)
# ===========================================================================


def test_rope_decision_positive():
    """Test rope decision positive."""
    samples = np.full(1000, 10.0)
    result = PlaceboInTime.bayesian_rope_decision(samples, 5.0, 0.95)
    assert result == "positive"


def test_rope_decision_null():
    """Test rope decision null."""
    samples = np.full(1000, 0.0)
    result = PlaceboInTime.bayesian_rope_decision(samples, 5.0, 0.95)
    assert result == "null"


def test_rope_decision_indeterminate():
    """Test rope decision indeterminate."""
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=3.0, scale=5.0, size=1000)
    result = PlaceboInTime.bayesian_rope_decision(samples, 5.0, 0.95)
    assert result == "indeterminate"


def test_rope_decision_with_mixed_samples():
    """Test rope decision with mixed samples."""
    samples = np.concatenate([np.full(960, 10.0), np.full(40, 0.0)])
    result = PlaceboInTime.bayesian_rope_decision(samples, 5.0, 0.95)
    assert result == "positive"


def test_rope_decision_barely_below_threshold():
    """Test rope decision barely below threshold."""
    samples = np.concatenate([np.full(940, 10.0), np.full(60, 0.0)])
    result = PlaceboInTime.bayesian_rope_decision(samples, 5.0, 0.95)
    assert result == "indeterminate"


# ===========================================================================
# Assurance formula validation (unit — no sampling)
# ===========================================================================
#
# These tests pin the assurance-simulation formula
# ``true_effect = theta_new + expected_effect`` (the corrected version).
# The old formula ``true_effect = expected_effect`` ignored the status-quo
# baseline, which understated the noise floor under the alternative and
# inflated assurance.  The invariants below would fail under that old
# formula.


def test_assurance_formula_zero_expected_matches_null():
    """When expected_effect == 0, alt and null decision rates should match.

    Under the corrected formula ``true_effect = theta_new + expected_effect``,
    expected_effect=0 makes the alternative distribution identical to the
    null distribution, so TP rate must approximately equal FP rate.

    Under the OLD (buggy) formula ``true_effect = expected_effect``,
    true_effect would collapse to 0 under the alternative and TP rate
    would be ~0 regardless of theta_new — which is what the fix corrects.
    """
    rng = np.random.default_rng(0)
    theta_new_samples = rng.normal(loc=0.0, scale=5.0, size=2000)
    fold_sds = np.array([1.0, 1.0, 1.0])

    check = PlaceboInTime(
        n_folds=2,
        expected_effect_prior=np.zeros(2000),
        rope_half_width=1.0,
        random_seed=123,
    )
    ar = check._compute_assurance(
        theta_new_samples=theta_new_samples,
        fold_sds=fold_sds,
        n_posterior_samples=500,
    )

    # With expected_effect identically zero the alt and null scenarios
    # share the same true-effect distribution.
    assert abs(ar.true_positive_rate - ar.false_positive_rate) < 0.05
    assert abs(ar.true_negative_rate - ar.false_negative_rate) < 0.05


def test_assurance_formula_large_expected_effect_dominates_baseline():
    """Large expected_effect should push true positives well above false positives.

    This pins the sign and direction of the fix: adding ``expected_effect``
    on top of ``theta_new`` (rather than replacing it) means that when
    expected_effect is large and positive the alternative scenario
    detects the intervention much more often than the null does.
    """
    theta_new_samples = np.zeros(1000)  # null baseline tightly around 0
    fold_sds = np.array([0.5])

    check = PlaceboInTime(
        n_folds=2,
        expected_effect_prior=np.full(1000, 10.0),  # large effect
        rope_half_width=1.0,
        random_seed=123,
    )
    ar = check._compute_assurance(
        theta_new_samples=theta_new_samples,
        fold_sds=fold_sds,
        n_posterior_samples=500,
    )

    assert ar.true_positive_rate > 0.9
    assert ar.false_positive_rate < 0.1
    assert ar.true_positive_rate > ar.false_positive_rate


def _assurance_rates(check: PlaceboInTime) -> tuple[float, ...]:
    """Run the assurance simulation on fixed inputs and return its rates."""
    result = check._compute_assurance(
        theta_new_samples=np.linspace(-5.0, 5.0, 500),
        fold_sds=np.array([0.5, 1.0, 2.0]),
        n_posterior_samples=200,
    )
    return (
        result.true_positive_rate,
        result.false_positive_rate,
        result.true_negative_rate,
        result.false_negative_rate,
    )


def _make_assurance_check(random_seed: int | None) -> PlaceboInTime:
    """Build a check whose assurance stage is driven only by ``random_seed``."""
    return PlaceboInTime(
        n_folds=2,
        expected_effect_prior=np.linspace(0.0, 4.0, 500),
        rope_half_width=1.0,
        random_seed=random_seed,
    )


def test_assurance_simulation_is_reproducible_under_master_seed():
    """The assurance stage must not drift between runs sharing a seed."""
    first = _make_assurance_check(2024)
    second = _make_assurance_check(2024)

    assert _assurance_rates(first) == _assurance_rates(second)
    # Repeating on the same instance must also be stable: the stage RNG is
    # derived from the master seed, never carried across calls.
    assert _assurance_rates(first) == _assurance_rates(first)


def test_assurance_simulation_differs_across_master_seeds():
    """Different seeds must actually exercise different simulation draws."""
    assert _assurance_rates(_make_assurance_check(2024)) != _assurance_rates(
        _make_assurance_check(99)
    )


def test_assurance_stage_rng_is_independent_of_prior_draw_stage():
    """Prior draws and simulation noise must not share a stream."""
    check = _make_assurance_check(2024)

    assert not np.array_equal(
        check._rng_for_stage(0).normal(size=50),
        check._rng_for_stage(1).normal(size=50),
    )


# ===========================================================================
# Cumulative impact extraction (integration — needs PyMC)
# ===========================================================================


@pytest.mark.integration
def test_extract_cumulative_impact(mock_pymc_sample):
    """Test extract cumulative impact."""
    df = _make_its_data()
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=150,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    cumulative = PlaceboInTime._extract_cumulative_impact(experiment)

    assert isinstance(cumulative, xr.DataArray)
    assert "sample" in cumulative.dims
    assert cumulative.sizes["sample"] > 0


# ===========================================================================
# Full run tests (integration — needs PyMC)
# ===========================================================================


@pytest.mark.integration
def test_run_produces_check_result(mock_pymc_sample):
    """Test run produces check result."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)

    assert isinstance(result, CheckResult)
    assert result.check_name == "PlaceboInTime"
    assert result.passed is not None
    assert "fold" in result.text.lower()


@pytest.mark.integration
def test_run_produces_fold_results(mock_pymc_sample):
    """Test run produces fold results."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)
    fold_results = result.metadata["fold_results"]

    assert len(fold_results) == 2
    for fr in fold_results:
        assert isinstance(fr, PlaceboFoldResult)
        assert isinstance(fr.experiment, InterruptedTimeSeries)
        assert isinstance(fr.fold_mean, float)
        assert isinstance(fr.fold_sd, float)
        assert fr.cumulative_impact_samples is not None


@pytest.mark.integration
def test_run_metadata_contains_null_distribution(mock_pymc_sample):
    """Test run metadata contains null distribution."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)

    assert "null_samples" in result.metadata
    assert "status_quo_idata" in result.metadata
    assert "actual_cumulative_mean" in result.metadata
    assert "p_effect_outside_null" in result.metadata

    null_samples = result.metadata["null_samples"]
    assert isinstance(null_samples, np.ndarray)
    assert len(null_samples) > 0

    p = result.metadata["p_effect_outside_null"]
    assert 0.0 <= p <= 1.0


@pytest.mark.integration
def test_run_metadata_carries_design_configuration(mock_pymc_sample):
    """Design knobs (ROPE, threshold, prior) and fold_sds round-trip into metadata."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    prior_samples = np.random.default_rng(0).normal(90, 15, size=200)
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
        rope_half_width=25.0,
        threshold=0.9,
        expected_effect_prior=prior_samples,
        random_seed=42,
    )
    result = check.run(experiment)

    assert result.metadata["rope_half_width"] == 25.0
    assert result.metadata["threshold"] == 0.9
    assert result.metadata["expected_effect_prior"] is prior_samples

    fold_sds = result.metadata["fold_sds"]
    assert isinstance(fold_sds, np.ndarray)
    assert fold_sds.shape == (2,)
    assert np.all(fold_sds > 0)
    np.testing.assert_array_equal(
        fold_sds,
        np.array([fr.fold_sd for fr in result.metadata["fold_results"]]),
    )


@pytest.mark.integration
def test_run_metadata_carries_defaults_when_unconfigured(mock_pymc_sample):
    """Configuration metadata is present (None / default) even without ROPE/prior."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)

    assert result.metadata["rope_half_width"] is None
    assert result.metadata["threshold"] == 0.95
    assert result.metadata["expected_effect_prior"] is None
    assert "fold_sds" in result.metadata


@pytest.mark.integration
def test_fold_treatment_times_are_shifted(mock_pymc_sample):
    """Test fold treatment times are shifted."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)
    for fr in result.metadata["fold_results"]:
        assert fr.pseudo_treatment_time < experiment.treatment_time


@pytest.mark.integration
def test_single_fold(mock_pymc_sample):
    """Test single fold."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=1,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)
    assert len(result.metadata["fold_results"]) == 1


@pytest.mark.integration
def test_no_mutable_state_on_check(mock_pymc_sample):
    """Test no mutable state on check."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    check.run(experiment)
    assert not hasattr(check, "fold_results")
    assert not hasattr(check, "_unseeded_custom_priors")


@pytest.mark.integration
def test_standalone_run_without_context(mock_pymc_sample):
    """Standalone use: pass experiment_factory, no PipelineContext."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)
    assert isinstance(result, CheckResult)
    assert result.passed is not None


@pytest.mark.integration
def test_standalone_no_factory_no_context_raises(mock_pymc_sample):
    """Test standalone no factory no context raises."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(n_folds=2, sample_kwargs=_FAST_HIERARCHICAL_KWARGS)
    with pytest.raises(RuntimeError, match="experiment_config"):
        check.run(experiment)


@pytest.mark.integration
def test_run_with_context(mock_pymc_sample):
    """Pipeline-style use: context provides experiment_config."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    ctx = PipelineContext(data=df)
    ctx.experiment = experiment
    ctx.experiment_config = {
        "method": InterruptedTimeSeries,
        "treatment_time": 1500,
        "formula": "y ~ 1 + t",
        "model": _make_pymc_model(),
    }
    check = PlaceboInTime(n_folds=2, sample_kwargs=_FAST_HIERARCHICAL_KWARGS)
    result = check.run(experiment, ctx)
    assert isinstance(result, CheckResult)
    assert result.passed is not None


@pytest.mark.integration
def test_text_contains_hierarchical_summary(mock_pymc_sample):
    """Test text contains hierarchical summary."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)
    assert "mu=" in result.text
    assert "tau=" in result.text
    assert "P(actual outside null)" in result.text


@pytest.mark.integration
def test_fold_fitting_failure_is_skipped(mock_pymc_sample):
    """Test fold fitting failure is skipped."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )

    call_count = 0

    def _failing_factory(data, treatment_time):
        """Factory that raises on first call to test skip logic."""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Simulated fitting failure")
        return InterruptedTimeSeries(
            data,
            treatment_time=treatment_time,
            formula="y ~ 1 + t",
            model=_make_pymc_model(),
        )

    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_failing_factory,
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)
    assert "SKIPPED" in result.text
    assert "failed to fit" in result.text


# ===========================================================================
# Fold eligibility (unit — no sampling)
# ===========================================================================


def test_run_skips_fold_with_insufficient_pre_period(monkeypatch):
    """An early fold is excluded before it can widen the hierarchical null."""
    data = pd.DataFrame({"y": np.zeros(100)}, index=np.arange(100))
    experiment = _make_fake_bayesian_experiment(data, treatment_time=70)
    fitted_times: list[int] = []

    def factory(fold_data, treatment_time):
        fitted_times.append(treatment_time)
        return _make_fake_bayesian_experiment(fold_data, treatment_time)

    check = PlaceboInTime(n_folds=2, experiment_factory=factory)
    monkeypatch.setattr(check, "_build_status_quo_model", _fake_status_quo_result)

    with pytest.warns(
        UserWarning, match="shorter than one full intervention window"
    ) as record:
        result = check.run(experiment)

    assert len(record) == 1
    assert fitted_times == [41]
    assert result.metadata["n_folds_requested"] == 2
    assert result.metadata["n_folds_completed"] == 1
    assert result.metadata["skipped_folds"] == [
        {
            "fold_index": 0,
            "pseudo_treatment_time": 12,
            "observed_pre_period_rows": 12,
            "required_pre_period_rows": 29,
            "reason": "insufficient_pre_period",
        }
    ]
    assert "Fold 1: SKIPPED" in result.text
    assert "Fold 1: pseudo treatment" not in result.text


def test_run_is_inconclusive_when_no_fold_has_enough_pre_period():
    """No eligible folds must not create a fabricated null distribution."""
    data = pd.DataFrame({"y": np.zeros(100)}, index=np.arange(100))
    experiment = _make_fake_bayesian_experiment(data, treatment_time=50)

    def factory(fold_data, treatment_time):  # pragma: no cover - must not fit
        del fold_data, treatment_time
        raise AssertionError("Ineligible folds must be skipped before fitting.")

    check = PlaceboInTime(n_folds=1, experiment_factory=factory)

    with pytest.warns(UserWarning, match="shorter than one full intervention window"):
        result = check.run(experiment)

    assert result.passed is None
    assert "INCONCLUSIVE — no folds completed." in result.text
    assert result.metadata["n_folds_requested"] == 1
    assert result.metadata["n_folds_completed"] == 0
    assert result.metadata["skipped_folds"] == [
        {
            "fold_index": 0,
            "pseudo_treatment_time": 1,
            "observed_pre_period_rows": 1,
            "required_pre_period_rows": 49,
            "reason": "insufficient_pre_period",
        }
    ]
    assert "null_samples" not in result.metadata
    assert "p_effect_outside_null" not in result.metadata


def test_random_run_is_inconclusive_with_no_feasible_folds():
    """A zero-candidate random selection cannot fabricate a null."""
    data = pd.DataFrame({"y": np.zeros(100)}, index=np.arange(100))
    experiment = _make_fake_bayesian_experiment(data, treatment_time=50)

    def factory(fold_data, treatment_time):  # pragma: no cover - must not fit
        del fold_data, treatment_time
        raise AssertionError("No feasible random fold may be fitted.")

    check = PlaceboInTime(
        n_folds=1,
        selection_method="random",
        experiment_factory=factory,
    )

    with pytest.warns(
        UserWarning, match="random selection yielded only 0 of 1"
    ) as record:
        result = check.run(experiment)

    assert len(record) == 1
    assert result.passed is None
    assert "INCONCLUSIVE — no folds completed." in result.text
    assert result.metadata["n_folds_requested"] == 1
    assert result.metadata["n_folds_completed"] == 0
    assert result.metadata["skipped_folds"] == [
        {
            "fold_index": 0,
            "pseudo_treatment_time": None,
            "observed_pre_period_rows": None,
            "required_pre_period_rows": 49,
            "reason": "insufficient_feasible_random_folds",
        }
    ]
    assert "null_samples" not in result.metadata


def test_random_run_uses_maximum_feasible_partial_folds(monkeypatch):
    """Random geometry shortfalls fit only the exact feasible subset."""
    data = pd.DataFrame({"y": np.zeros(201)}, index=np.arange(201))
    experiment = _make_fake_bayesian_experiment(data, treatment_time=150)
    fitted_times: list[int] = []

    def factory(fold_data, treatment_time):
        fitted_times.append(treatment_time)
        return _make_fake_bayesian_experiment(fold_data, treatment_time)

    check = PlaceboInTime(
        n_folds=3,
        selection_method="random",
        min_training_pct=0.10,
        experiment_factory=factory,
        random_seed=42,
    )
    monkeypatch.setattr(check, "_build_status_quo_model", _fake_status_quo_result)

    with pytest.warns(
        UserWarning, match="random selection yielded only 2 of 3"
    ) as record:
        result = check.run(experiment)

    assert len(record) == 1
    assert fitted_times == [50, 100]
    assert result.metadata["n_folds_requested"] == 3
    assert result.metadata["n_folds_completed"] == 2
    assert result.metadata["skipped_folds"] == [
        {
            "fold_index": 2,
            "pseudo_treatment_time": None,
            "observed_pre_period_rows": None,
            "required_pre_period_rows": 50,
            "reason": "insufficient_feasible_random_folds",
        }
    ]
    assert "Fold 3: SKIPPED" in result.text


def test_run_keeps_folds_with_one_full_intervention_window(monkeypatch):
    """Folds with sufficient history still support the existing headline path."""
    data = pd.DataFrame({"y": np.zeros(100)}, index=np.arange(100))
    experiment = _make_fake_bayesian_experiment(data, treatment_time=75)
    fitted_times: list[int] = []

    def factory(fold_data, treatment_time):
        fitted_times.append(treatment_time)
        return _make_fake_bayesian_experiment(fold_data, treatment_time)

    check = PlaceboInTime(n_folds=2, experiment_factory=factory)
    monkeypatch.setattr(check, "_build_status_quo_model", _fake_status_quo_result)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = check.run(experiment)

    assert fitted_times == [27, 51]
    assert result.passed is True
    assert result.metadata["n_folds_requested"] == 2
    assert result.metadata["n_folds_completed"] == 2
    assert result.metadata["skipped_folds"] == []


# ===========================================================================
# Explicit intervention_length tests (unit — no sampling)
# ===========================================================================


def test_intervention_length_defaults_to_none():
    """Test intervention length defaults to none."""
    check = PlaceboInTime()
    assert check.intervention_length is None


@pytest.mark.parametrize(
    "intervention_length",
    [0, -1, -2.5, pd.Timedelta(0), pd.Timedelta(days=-1)],
)
def test_invalid_intervention_length(intervention_length):
    """Test invalid intervention length."""
    with pytest.raises(ValueError, match="intervention_length must be positive"):
        PlaceboInTime(intervention_length=intervention_length)


def test_repr_shows_explicit_intervention_length():
    """Test repr shows explicit intervention length."""
    check = PlaceboInTime(n_folds=2, intervention_length=10)
    assert repr(check) == "PlaceboInTime(n_folds=2, intervention_length=10)"


def test_repr_omits_derived_intervention_length():
    """Test repr omits derived intervention length."""
    assert repr(PlaceboInTime(n_folds=2)) == "PlaceboInTime(n_folds=2)"


def test_explicit_intervention_length_overrides_derived_default():
    """An explicit window length wins over the experiment-derived one."""
    data = pd.DataFrame({"y": np.zeros(100)}, index=np.arange(100))
    experiment = _make_fake_bayesian_experiment(data, treatment_time=70)

    assert PlaceboInTime()._compute_intervention_length(experiment) == 29
    assert (
        PlaceboInTime(intervention_length=10)._compute_intervention_length(experiment)
        == 10
    )


def test_explicit_intervention_length_makes_more_folds_eligible(monkeypatch):
    """A shorter placebo window fits folds that the default would skip."""
    data = pd.DataFrame({"y": np.zeros(100)}, index=np.arange(100))
    experiment = _make_fake_bayesian_experiment(data, treatment_time=70)
    fitted_times: list[int] = []

    def factory(fold_data, treatment_time):
        fitted_times.append(treatment_time)
        return _make_fake_bayesian_experiment(fold_data, treatment_time)

    check = PlaceboInTime(
        n_folds=4,
        intervention_length=10,
        experiment_factory=factory,
    )
    monkeypatch.setattr(check, "_build_status_quo_model", _fake_status_quo_result)

    with pytest.warns(UserWarning, match="placebo windows span 10 observation"):
        result = check.run(experiment)

    assert fitted_times == [30, 40, 50, 60]
    assert result.metadata["n_folds_completed"] == 4
    assert result.metadata["skipped_folds"] == []
    assert result.metadata["intervention_length"] == 10


def test_comparison_window_metadata_is_recorded():
    """Both spans behind the verdict are exposed for interpretation."""
    data = pd.DataFrame({"y": np.zeros(100)}, index=np.arange(100))
    experiment = _make_fake_bayesian_experiment(data, treatment_time=70)

    def factory(fold_data, treatment_time):  # pragma: no cover - not reached
        del fold_data, treatment_time
        raise AssertionError("No eligible fold exists in this configuration.")

    check = PlaceboInTime(n_folds=1, intervention_length=90, experiment_factory=factory)

    with pytest.warns(UserWarning, match="shorter than one full intervention window"):
        result = check.run(experiment)

    assert result.metadata["comparison_window"] == {
        "placebo_window_observations": 30,
        "actual_post_period_observations": 30,
    }
    assert result.metadata["intervention_length"] == 90


def test_derived_intervention_length_does_not_warn_about_comparison_window(monkeypatch):
    """The one-observation half-open artefact must stay silent."""
    data = pd.DataFrame({"y": np.zeros(100)}, index=np.arange(100))
    experiment = _make_fake_bayesian_experiment(data, treatment_time=75)

    def factory(fold_data, treatment_time):
        return _make_fake_bayesian_experiment(fold_data, treatment_time)

    check = PlaceboInTime(n_folds=2, experiment_factory=factory)
    monkeypatch.setattr(check, "_build_status_quo_model", _fake_status_quo_result)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = check.run(experiment)

    assert result.metadata["comparison_window"] == {
        "placebo_window_observations": 24,
        "actual_post_period_observations": 25,
    }


def test_intervention_length_spanning_no_observations_raises():
    """An empty placebo window must fail loudly, not fabricate a null."""
    data = pd.DataFrame({"y": np.zeros(100)}, index=np.arange(100))
    experiment = _make_fake_bayesian_experiment(data, treatment_time=69.5)

    def factory(fold_data, treatment_time):  # pragma: no cover - not reached
        del fold_data, treatment_time
        raise AssertionError("An empty placebo window must fail before fitting.")

    check = PlaceboInTime(
        n_folds=1, intervention_length=0.4, experiment_factory=factory
    )

    with pytest.raises(ValueError, match="spans no observations"):
        check.run(experiment)


def test_explicit_intervention_length_with_datetime_index(monkeypatch):
    """Timedelta windows drive fold geometry on datetime-indexed data."""
    index = pd.date_range("2020-01-01", periods=100, freq="D")
    data = pd.DataFrame({"y": np.zeros(100)}, index=index)
    treatment_time = index[70]
    experiment = _make_fake_bayesian_experiment(data, treatment_time)
    fitted_times: list[pd.Timestamp] = []

    def factory(fold_data, fold_treatment_time):
        fitted_times.append(fold_treatment_time)
        return _make_fake_bayesian_experiment(fold_data, fold_treatment_time)

    check = PlaceboInTime(
        n_folds=3,
        intervention_length=pd.Timedelta(days=10),
        experiment_factory=factory,
    )
    monkeypatch.setattr(check, "_build_status_quo_model", _fake_status_quo_result)

    with pytest.warns(UserWarning, match="placebo windows span 10 observation"):
        result = check.run(experiment)

    assert fitted_times == [index[40], index[50], index[60]]
    assert result.metadata["n_folds_completed"] == 3


def test_explicit_intervention_length_widens_random_candidate_pool():
    """Random selection sees the configured window, not the derived one."""
    data = pd.DataFrame({"y": np.zeros(100)}, index=np.arange(100))

    derived = PlaceboInTime(
        n_folds=3, selection_method="random", random_seed=42
    )._compute_random_fold_treatment_times(data, 70, 29)
    configured = PlaceboInTime(
        n_folds=3,
        selection_method="random",
        intervention_length=10,
        random_seed=42,
    )._compute_random_fold_treatment_times(data, 70, 10)

    assert len(derived) < 3
    assert len(configured) == 3


def test_pipeline_run_derives_independent_fold_seeds(monkeypatch):
    """Pipeline-created folds receive deterministic, distinct model seeds."""
    data = pd.DataFrame({"y": np.zeros(100)}, index=np.arange(100))
    experiment = _make_fake_bayesian_experiment(data, treatment_time=75)
    source_model = SimpleNamespace(sample_kwargs={"random_seed": 1})
    fitted_models: list[SimpleNamespace] = []

    def method(fold_data, treatment_time, model):
        fitted_models.append(model)
        return _make_fake_bayesian_experiment(fold_data, treatment_time)

    context = PipelineContext(data=data)
    context.experiment_config = {
        "method": method,
        "treatment_time": 75,
        "model": source_model,
    }
    check = PlaceboInTime(n_folds=2, random_seed=73)
    monkeypatch.setattr(check, "_build_status_quo_model", _fake_status_quo_result)

    result = check.run(experiment, context)

    assert result.metadata["n_folds_completed"] == 2
    assert [model.sample_kwargs["random_seed"] for model in fitted_models] == [73, 74]
    assert source_model.sample_kwargs["random_seed"] == 1


def test_pipeline_run_seeds_default_model_template(monkeypatch):
    """Pipeline-created folds seed a method's implicit default model."""
    data = pd.DataFrame({"y": np.zeros(100)}, index=np.arange(100))
    experiment = _make_fake_bayesian_experiment(data, treatment_time=75)
    fitted_models: list[SimpleNamespace] = []

    class _DefaultModel:
        def __init__(self):
            self.sample_kwargs: dict[str, int] = {}

    class _Method:
        _default_model_class = _DefaultModel

        def __call__(self, fold_data, treatment_time, model):
            fitted_models.append(model)
            return _make_fake_bayesian_experiment(fold_data, treatment_time)

    context = PipelineContext(data=data)
    context.experiment_config = {
        "method": _Method(),
        "treatment_time": 75,
    }
    check = PlaceboInTime(n_folds=2, random_seed=73)
    monkeypatch.setattr(check, "_build_status_quo_model", _fake_status_quo_result)

    result = check.run(experiment, context)

    assert result.metadata["n_folds_completed"] == 2
    assert [model.sample_kwargs["random_seed"] for model in fitted_models] == [73, 74]


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.correctness
def test_master_seed_makes_full_placebo_run_reproducible():
    """Same-process runs retain exact report, metadata, and null samples."""

    def run_once():
        data = _make_its_data(n=100, seed=91)
        return (
            Pipeline(
                data=data,
                steps=[
                    cp.EstimateEffect(
                        method=InterruptedTimeSeries,
                        treatment_time=75,
                        formula="y ~ 1 + t",
                        model=cp.pymc_models.LinearRegression(
                            sample_kwargs={
                                "chains": 1,
                                "cores": 1,
                                "draws": 20,
                                "tune": 20,
                                "progressbar": False,
                                "random_seed": 41,
                            }
                        ),
                    ),
                    cp.SensitivityAnalysis(
                        checks=[
                            PlaceboInTime(
                                n_folds=2,
                                sample_kwargs={
                                    "chains": 1,
                                    "cores": 1,
                                    "draws": 20,
                                    "tune": 20,
                                    "progressbar": False,
                                },
                                random_seed=73,
                            )
                        ]
                    ),
                ],
            )
            .run()
            .sensitivity_results[0]
        )

    first = run_once()
    second = run_once()

    assert first.text == second.text
    assert first.metadata.keys() == second.metadata.keys()
    for key in (
        "n_folds_requested",
        "n_folds_completed",
        "skipped_folds",
        "actual_cumulative_mean",
        "p_effect_outside_null",
        "rope_half_width",
        "threshold",
        "expected_effect_prior",
        "unseeded_custom_priors",
    ):
        assert first.metadata[key] == second.metadata[key]
    np.testing.assert_array_equal(
        first.metadata["fold_sds"], second.metadata["fold_sds"]
    )
    np.testing.assert_array_equal(
        first.metadata["null_samples"], second.metadata["null_samples"]
    )
    xr.testing.assert_equal(
        first.metadata["status_quo_idata"].posterior,
        second.metadata["status_quo_idata"].posterior,
    )

    first_folds = first.metadata["fold_results"]
    second_folds = second.metadata["fold_results"]
    assert len(first_folds) == len(second_folds) == 2
    for first_fold, second_fold in zip(first_folds, second_folds, strict=True):
        assert first_fold.fold == second_fold.fold
        assert first_fold.pseudo_treatment_time == second_fold.pseudo_treatment_time
        assert first_fold.fold_mean == second_fold.fold_mean
        assert first_fold.fold_sd == second_fold.fold_sd
        np.testing.assert_array_equal(
            first_fold.cumulative_impact_samples,
            second_fold.cumulative_impact_samples,
        )
        pd.testing.assert_frame_equal(
            first_fold.experiment.data, second_fold.experiment.data
        )
        xr.testing.assert_equal(
            first_fold.experiment.post_impact,
            second_fold.experiment.post_impact,
        )
    assert [
        fold.experiment.model.sample_kwargs["random_seed"] for fold in first_folds
    ] == [73, 74]
    assert [
        fold.experiment.model.sample_kwargs["random_seed"] for fold in second_folds
    ] == [73, 74]


# ===========================================================================
# Assurance tests (integration — needs PyMC)
# ===========================================================================


@pytest.mark.integration
def test_assurance_with_numpy_array(mock_pymc_sample):
    """Test assurance with numpy array."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
        expected_effect_prior=np.random.default_rng(42).normal(90, 15, size=200),
        rope_half_width=50.0,
        random_seed=42,
    )
    result = check.run(experiment)

    assert "assurance_result" in result.metadata
    assert "assurance" in result.metadata

    ar = result.metadata["assurance_result"]
    assert isinstance(ar, AssuranceResult)
    assert 0.0 <= ar.true_positive_rate <= 1.0
    assert 0.0 <= ar.false_positive_rate <= 1.0
    assert 0.0 <= ar.true_negative_rate <= 1.0
    assert 0.0 <= ar.false_negative_rate <= 1.0

    null_sum = (
        ar.false_positive_rate + ar.true_negative_rate + ar.null_indeterminate_rate
    )
    assert abs(null_sum - 1.0) < 0.01

    alt_sum = ar.true_positive_rate + ar.false_negative_rate + ar.alt_indeterminate_rate
    assert abs(alt_sum - 1.0) < 0.01


@pytest.mark.integration
def test_assurance_with_rvs_object(mock_pymc_sample):
    """Legacy .rvs(n) priors warn and are recorded as unseeded."""

    class _MockDistribution:
        def rvs(self, n):
            return np.random.default_rng(42).normal(90, 15, size=n)

    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
        expected_effect_prior=_MockDistribution(),
        rope_half_width=50.0,
        random_seed=42,
    )
    with pytest.warns(UserWarning, match="using unseeded legacy .rvs") as record:
        result = check.run(experiment)

    assert (
        len(
            [
                warning
                for warning in record
                if "using unseeded legacy .rvs" in str(warning.message)
            ]
        )
        == 1
    )
    assert "assurance" in result.metadata
    assert isinstance(result.metadata["assurance"], float)
    assert result.metadata["unseeded_custom_priors"][0]["reason"] == (
        "rvs_does_not_accept_random_state"
    )
    assert result.metadata["unseeded_custom_priors"][0]["prior_type"].endswith(
        "._MockDistribution"
    )


@pytest.mark.integration
def test_assurance_text_in_report(mock_pymc_sample):
    """Test assurance text in report."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
        expected_effect_prior=np.full(200, 90.0),
        rope_half_width=50.0,
        random_seed=42,
    )
    result = check.run(experiment)
    assert "Bayesian assurance" in result.text
    assert "Assurance (TP rate)" in result.text
    assert "False Positive rate" in result.text


@pytest.mark.integration
def test_no_assurance_without_prior(mock_pymc_sample):
    """Test no assurance without prior."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)
    assert "assurance_result" not in result.metadata
    assert "assurance" not in result.metadata
    assert "Bayesian assurance" not in result.text


# ===========================================================================
# Default check registration
# ===========================================================================


# ===========================================================================
# Random selection mode — construction tests (unit — no sampling)
# ===========================================================================


def test_selection_method_default():
    """Default selection method is sequential."""
    check = PlaceboInTime()
    assert check.selection_method == "sequential"


def test_selection_method_random():
    """Random selection mode stores parameters."""
    check = PlaceboInTime(
        selection_method="random",
        min_training_pct=0.40,
        min_gap=2,
        exclude_periods={"2020-01"},
        random_seed=99,
    )
    assert check.selection_method == "random"
    assert check.min_training_pct == 0.40
    assert check.min_gap == 2
    assert check.exclude_periods == {"2020-01"}


def test_invalid_selection_method():
    """Invalid selection method raises ValueError."""
    with pytest.raises(ValueError, match="selection_method"):
        PlaceboInTime(selection_method="invalid")


def test_invalid_min_training_pct():
    """min_training_pct outside (0, 1) raises ValueError."""
    with pytest.raises(ValueError, match="min_training_pct"):
        PlaceboInTime(selection_method="random", min_training_pct=0.0)
    with pytest.raises(ValueError, match="min_training_pct"):
        PlaceboInTime(selection_method="random", min_training_pct=1.0)


def test_invalid_min_gap():
    """min_gap < 1 raises ValueError."""
    with pytest.raises(ValueError, match="min_gap"):
        PlaceboInTime(selection_method="random", min_gap=0)


def test_allow_overlap_default_false():
    """allow_overlap defaults to False (non-overlap enforced)."""
    check = PlaceboInTime(selection_method="random")
    assert check.allow_overlap is False


def test_allow_overlap_stores_value():
    """allow_overlap=True is stored on the instance."""
    check = PlaceboInTime(selection_method="random", allow_overlap=True)
    assert check.allow_overlap is True


def test_repr_random_selection():
    """repr includes selection_method when not sequential."""
    check = PlaceboInTime(n_folds=4, selection_method="random")
    r = repr(check)
    assert "selection_method='random'" in r
    assert "n_folds=4" in r


def test_repr_sequential_omits_selection_method():
    """repr omits selection_method when sequential (default)."""
    check = PlaceboInTime(n_folds=3)
    assert "selection_method" not in repr(check)


def test_repr_hides_default_allow_overlap():
    """allow_overlap=False (default) is not shown in repr."""
    check = PlaceboInTime(selection_method="random")
    assert "allow_overlap" not in repr(check)


def test_repr_shows_non_default_allow_overlap():
    """allow_overlap=True is surfaced in repr."""
    check = PlaceboInTime(selection_method="random", allow_overlap=True)
    assert "allow_overlap=True" in repr(check)


# ===========================================================================
# Random fold selection — geometry tests (unit — no sampling)
# ===========================================================================


def test_random_fold_treatment_times_count():
    """Random selection returns exactly n_folds treatment times."""
    n = 200
    data = pd.DataFrame({"y": np.zeros(n)}, index=np.arange(n))
    check = PlaceboInTime(
        n_folds=3,
        selection_method="random",
        min_training_pct=0.20,
        random_seed=42,
    )
    times = check._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=20
    )
    assert len(times) == 3
    # All must be before treatment_time
    assert all(t < 150 for t in times)
    # Sorted
    assert times == sorted(times)


def test_random_fold_selection_requires_one_full_pre_period():
    """Random selection excludes candidates without an intervention-sized history."""
    data = pd.DataFrame({"y": np.zeros(200)}, index=np.arange(200))
    intervention_length = 40
    check = PlaceboInTime(
        n_folds=2,
        selection_method="random",
        min_training_pct=0.10,
        random_seed=42,
    )

    times = check._compute_random_fold_treatment_times(
        data,
        treatment_time=150,
        intervention_length=intervention_length,
    )

    assert all(time >= intervention_length for time in times)
    for time in times:
        observed_rows, required_rows = check._get_fold_pre_period_observation_counts(
            data,
            time,
            intervention_length,
        )
        assert observed_rows >= required_rows


def test_random_fold_treatment_times_reproducible():
    """Same seed produces same selection."""
    n = 200
    data = pd.DataFrame({"y": np.zeros(n)}, index=np.arange(n))
    kwargs = {
        "n_folds": 3,
        "selection_method": "random",
        "min_training_pct": 0.20,
        "random_seed": 42,
    }
    times1 = PlaceboInTime(**kwargs)._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=20
    )
    times2 = PlaceboInTime(**kwargs)._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=20
    )
    assert times1 == times2


def test_random_fold_different_seeds_differ():
    """Different seeds produce different selections."""
    n = 200
    data = pd.DataFrame({"y": np.zeros(n)}, index=np.arange(n))
    times1 = PlaceboInTime(
        n_folds=3,
        selection_method="random",
        min_training_pct=0.20,
        random_seed=42,
    )._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=20
    )
    times2 = PlaceboInTime(
        n_folds=3,
        selection_method="random",
        min_training_pct=0.20,
        random_seed=99,
    )._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=20
    )
    assert times1 != times2


def test_random_fold_respects_min_gap():
    """Selected folds respect the min_gap constraint."""
    n = 200
    data = pd.DataFrame({"y": np.zeros(n)}, index=np.arange(n))
    check = PlaceboInTime(
        n_folds=3,
        selection_method="random",
        min_training_pct=0.10,
        min_gap=5,
        random_seed=42,
    )
    times = check._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=10
    )
    # Gaps between consecutive selected times should be >= min_gap
    # (since they were selected from a candidate list with min_gap spacing)
    for i in range(len(times) - 1):
        assert times[i + 1] - times[i] >= 5


def test_random_fold_returns_empty_without_eligible_periods():
    """A valid configuration with no eligible period returns no folds."""
    n = 200
    data = pd.DataFrame({"y": np.zeros(n)}, index=np.arange(n))
    exclude = {str(i) for i in range(n)}
    check = PlaceboInTime(
        n_folds=1,
        selection_method="random",
        exclude_periods=exclude,
        random_seed=42,
    )

    assert (
        check._compute_random_fold_treatment_times(
            data, treatment_time=150, intervention_length=10
        )
        == []
    )


def test_random_fold_returns_feasible_partial_when_candidates_are_few():
    """A valid but short candidate pool returns its feasible subset."""
    data = pd.DataFrame({"y": np.zeros(10)}, index=np.arange(10))
    check = PlaceboInTime(
        n_folds=5,
        selection_method="random",
        min_training_pct=0.50,
        random_seed=42,
    )

    assert check._compute_random_fold_treatment_times(
        data, treatment_time=8, intervention_length=2
    ) == [4, 6]


def test_random_fold_with_datetime_index():
    """Random selection works with datetime-indexed data."""
    dates = pd.date_range("2020-01-01", periods=100, freq="MS")
    data = pd.DataFrame({"y": np.zeros(100)}, index=dates)
    treatment = pd.Timestamp("2027-01-01")
    check = PlaceboInTime(
        n_folds=3,
        selection_method="random",
        min_training_pct=0.20,
        exclude_periods={"2020-06"},
        random_seed=42,
    )
    times = check._compute_random_fold_treatment_times(
        data,
        treatment_time=treatment,
        intervention_length=pd.DateOffset(months=6),
    )
    assert len(times) == 3
    assert all(t < treatment for t in times)
    # Excluded month should not appear
    for t in times:
        assert t.strftime("%Y-%m") != "2020-06"


# ===========================================================================
# Non-overlap constraint (unit — no sampling)
# ===========================================================================
#
# Default behaviour prevents pseudo-intervention windows from overlapping
# each other; this is the fix for Ben's review points 1 and 2 from the
# 2026-04-23 round.  The two folds share observations if they overlap,
# which violates the exchangeability assumption of the hierarchical
# status-quo model.


def test_random_folds_do_not_overlap_by_default():
    """Default (allow_overlap=False): no two selected windows overlap."""
    n = 200
    data = pd.DataFrame({"y": np.zeros(n)}, index=np.arange(n))
    intervention_length = 20
    check = PlaceboInTime(
        n_folds=3,
        selection_method="random",
        min_training_pct=0.10,
        random_seed=42,
    )
    times = check._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=intervention_length
    )
    # Sorted times should have gaps >= intervention_length
    for i in range(len(times) - 1):
        assert times[i + 1] - times[i] >= intervention_length, (
            f"Folds at {times[i]} and {times[i + 1]} overlap "
            f"with intervention_length={intervention_length}"
        )


def test_random_folds_allow_overlap_when_requested():
    """allow_overlap=True lets folds pack closer than intervention_length.

    We make the non-overlap constraint very tight (large intervention
    relative to the pool) so that the non-overlap default would
    drastically reduce the number of feasible arrangements.  With
    ``allow_overlap=True`` the selection should still succeed and at
    least one pair should be closer than ``intervention_length``.
    """
    n = 200
    data = pd.DataFrame({"y": np.zeros(n)}, index=np.arange(n))
    intervention_length = 40
    check = PlaceboInTime(
        n_folds=3,
        selection_method="random",
        min_training_pct=0.05,
        allow_overlap=True,
        random_seed=0,
    )
    times = check._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=intervention_length
    )
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    assert min(gaps) < intervention_length, (
        "allow_overlap=True should permit at least one gap shorter than "
        f"intervention_length={intervention_length}, got gaps={gaps}"
    )


def test_windows_overlap_helper_numeric():
    """_windows_overlap detects overlap for numeric indices."""
    # Non-overlapping: [0, 10) and [10, 20) share no observations
    assert PlaceboInTime._windows_overlap(0, 10, 10) is False
    # Overlapping: [0, 10) and [5, 15)
    assert PlaceboInTime._windows_overlap(0, 5, 10) is True
    # Order-independent
    assert PlaceboInTime._windows_overlap(5, 0, 10) is True


def test_windows_overlap_helper_datetime():
    """_windows_overlap detects overlap for datetime indices + DateOffset."""
    t_a = pd.Timestamp("2020-01-01")
    t_b = pd.Timestamp("2020-07-01")
    t_c = pd.Timestamp("2020-04-01")
    length = pd.DateOffset(months=6)
    # [2020-01, 2020-07) and [2020-07, 2021-01) are back-to-back: non-overlap
    assert PlaceboInTime._windows_overlap(t_a, t_b, length) is False
    # [2020-01, 2020-07) and [2020-04, 2020-10) overlap in April-July
    assert PlaceboInTime._windows_overlap(t_a, t_c, length) is True


# ===========================================================================
# Maximum feasible random selection (unit — no sampling)
# ===========================================================================
#
# Seeded retries preserve random selection when the requested count is
# feasible. Geometry shortfalls instead return the exact maximum subset so
# the caller can report skipped folds rather than fail the whole analysis.


def test_greedy_retry_preserves_reproducibility():
    """Same seed still produces identical results after retry refactor."""
    n = 200
    data = pd.DataFrame({"y": np.zeros(n)}, index=np.arange(n))
    kwargs = {
        "n_folds": 3,
        "selection_method": "random",
        "min_training_pct": 0.10,
        "min_gap": 5,
        "random_seed": 7,
    }
    t1 = PlaceboInTime(**kwargs)._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=10
    )
    t2 = PlaceboInTime(**kwargs)._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=10
    )
    assert t1 == t2


def test_random_selection_returns_maximum_when_geometry_is_infeasible():
    """Non-overlap constraints return the available two-fold subset."""
    data = pd.DataFrame({"y": np.zeros(200)}, index=np.arange(200))
    check = PlaceboInTime(
        n_folds=3,
        selection_method="random",
        min_training_pct=0.10,
        random_seed=42,
    )

    assert check._compute_random_fold_treatment_times(
        data, treatment_time=150, intervention_length=50
    ) == [50, 100]


def test_random_selection_avoids_the_central_greedy_trap():
    """Partial selection finds both endpoint windows, not one random center."""
    data = pd.DataFrame({"y": np.zeros(401)}, index=np.arange(401))
    check = PlaceboInTime(
        n_folds=102,
        selection_method="random",
        min_training_pct=0.30,
        random_seed=42,
    )

    assert check._compute_random_fold_treatment_times(
        data, treatment_time=300, intervention_length=100
    ) == [100, 200]


# ===========================================================================
# expected_effect_prior cycling warning (unit — no sampling)
# ===========================================================================


def test_draw_expected_effect_samples_warns_when_shorter_than_n():
    """Warn when numpy prior has fewer samples than requested replications."""
    check = PlaceboInTime(
        expected_effect_prior=np.array([1.0, 2.0, 3.0]),
        rope_half_width=0.5,
    )
    with pytest.warns(UserWarning, match="cycled through"):
        out = check._draw_expected_effect_samples(n=100)
    # Behaviour preserved: the raw array is returned and cycling happens
    # at the consumer in ``_compute_assurance``.
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))


def test_draw_expected_effect_samples_no_warning_when_long_enough():
    """No warning when numpy prior already has >= n samples."""
    prior = np.ones(200)
    check = PlaceboInTime(
        expected_effect_prior=prior,
        rope_half_width=0.5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = check._draw_expected_effect_samples(n=50)
    np.testing.assert_array_equal(out, prior)


def test_draw_expected_effect_samples_rvs_no_warning():
    """Objects with .rvs(n) receive n directly and never warn."""

    class _Dist:
        def __init__(self):
            self.last_n: int | None = None

        def rvs(self, n):
            self.last_n = n
            return np.linspace(0.0, 1.0, n)

    dist = _Dist()
    check = PlaceboInTime(
        expected_effect_prior=dist,
        rope_half_width=0.5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = check._draw_expected_effect_samples(n=13)
    assert dist.last_n == 13
    assert len(out) == 13


def test_draw_expected_effect_samples_seeded_rvs_is_reproducible():
    """Seed-aware priors receive a deterministic derived Generator."""

    class _SeedAwareDistribution:
        def __init__(self):
            self.random_states: list[np.random.Generator] = []

        def rvs(self, n, random_state):
            self.random_states.append(random_state)
            return random_state.normal(size=n)

    first_distribution = _SeedAwareDistribution()
    second_distribution = _SeedAwareDistribution()
    first = PlaceboInTime(
        expected_effect_prior=first_distribution,
        rope_half_width=0.5,
        random_seed=71,
    )._draw_expected_effect_samples(n=13)
    second = PlaceboInTime(
        expected_effect_prior=second_distribution,
        rope_half_width=0.5,
        random_seed=71,
    )._draw_expected_effect_samples(n=13)

    np.testing.assert_array_equal(first, second)
    assert isinstance(first_distribution.random_states[0], np.random.Generator)
    assert isinstance(second_distribution.random_states[0], np.random.Generator)


def test_draw_expected_effect_samples_scipy_prior_is_reproducible():
    """SciPy priors receive the derived Generator through random_state."""
    from scipy.stats import norm

    first = PlaceboInTime(
        expected_effect_prior=norm(loc=2.0, scale=0.5),
        rope_half_width=0.5,
        random_seed=71,
    )._draw_expected_effect_samples(n=13)
    second = PlaceboInTime(
        expected_effect_prior=norm(loc=2.0, scale=0.5),
        rope_half_width=0.5,
        random_seed=71,
    )._draw_expected_effect_samples(n=13)

    np.testing.assert_array_equal(first, second)


def test_draw_expected_effect_samples_propagates_seeded_prior_type_error():
    """A TypeError inside a seed-aware prior is not misclassified as legacy."""

    class _FailingSeedAwareDistribution:
        def __init__(self):
            self.calls = 0

        def rvs(self, n, random_state):
            del n, random_state
            self.calls += 1
            raise TypeError("prior calculation failed")

    distribution = _FailingSeedAwareDistribution()
    check = PlaceboInTime(
        expected_effect_prior=distribution,
        rope_half_width=0.5,
        random_seed=71,
    )

    with pytest.raises(TypeError, match="prior calculation failed"):
        check._draw_expected_effect_samples(n=13)
    assert distribution.calls == 1


def test_draw_expected_effect_samples_legacy_rvs_warns_and_is_recorded():
    """Seeded legacy priors retain behavior without silently claiming reproducibility."""

    class _LegacyDistribution:
        def rvs(self, n):
            return np.linspace(0.0, 1.0, n)

    diagnostics: list[dict[str, str]] = []
    check = PlaceboInTime(
        expected_effect_prior=_LegacyDistribution(),
        rope_half_width=0.5,
        random_seed=71,
    )
    with pytest.warns(UserWarning, match="using unseeded legacy .rvs"):
        samples = check._draw_expected_effect_samples(
            n=13,
            unseeded_custom_priors=diagnostics,
        )

    assert len(samples) == 13
    assert diagnostics == [
        {
            "prior_type": (
                f"{_LegacyDistribution.__module__}.{_LegacyDistribution.__qualname__}"
            ),
            "reason": "rvs_does_not_accept_random_state",
        }
    ]
    assert not hasattr(check, "_unseeded_custom_priors")


# ===========================================================================
# Random selection — full run (integration — needs PyMC)
# ===========================================================================


@pytest.mark.integration
def test_run_random_selection(mock_pymc_sample):
    """Full run with random selection mode."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    check = PlaceboInTime(
        n_folds=2,
        selection_method="random",
        min_training_pct=0.20,
        random_seed=42,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)

    assert isinstance(result, CheckResult)
    assert result.check_name == "PlaceboInTime"
    assert result.passed is not None
    assert len(result.metadata["fold_results"]) == 2
    for fr in result.metadata["fold_results"]:
        assert fr.pseudo_treatment_time < experiment.treatment_time


# ===========================================================================
# Default check registration
# ===========================================================================


def test_placebo_in_time_registered_as_default():
    """Test placebo in time registered as default."""
    its_defaults = _DEFAULT_CHECKS.get(InterruptedTimeSeries, [])
    assert PlaceboInTime in its_defaults

    sc_defaults = _DEFAULT_CHECKS.get(cp.SyntheticControl, [])
    assert PlaceboInTime in sc_defaults


def test_default_for_includes_placebo_in_time():
    """Test default for includes placebo in time."""
    step = SensitivityAnalysis.default_for(InterruptedTimeSeries)
    assert any(isinstance(c, PlaceboInTime) for c in step.checks)


# ===========================================================================
# Pipeline integration
# ===========================================================================


@pytest.mark.integration
def test_pipeline_with_placebo_in_time(mock_pymc_sample):
    """Test pipeline with placebo in time."""
    n = 2000
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"t": np.arange(n), "y": rng.normal(size=n)})

    result = Pipeline(
        data=data,
        steps=[
            cp.EstimateEffect(
                method=InterruptedTimeSeries,
                treatment_time=1500,
                formula="y ~ 1 + t",
                model=_make_pymc_model(),
            ),
            cp.SensitivityAnalysis(
                checks=[
                    PlaceboInTime(
                        n_folds=2,
                        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
                    )
                ],
            ),
        ],
    ).run()

    assert result.experiment is not None
    assert len(result.sensitivity_results) == 1
    check_result = result.sensitivity_results[0]
    assert check_result.passed is not None
    assert "null_samples" in check_result.metadata
    fold_results = check_result.metadata["fold_results"]
    assert len(fold_results) == 2
    for fr in fold_results:
        assert isinstance(fr, PlaceboFoldResult)


@pytest.mark.integration
def test_pipeline_with_assurance(mock_pymc_sample):
    """Test pipeline with assurance."""
    n = 2000
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"t": np.arange(n), "y": rng.normal(size=n)})

    result = Pipeline(
        data=data,
        steps=[
            cp.EstimateEffect(
                method=InterruptedTimeSeries,
                treatment_time=1500,
                formula="y ~ 1 + t",
                model=_make_pymc_model(),
            ),
            cp.SensitivityAnalysis(
                checks=[
                    PlaceboInTime(
                        n_folds=2,
                        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
                        expected_effect_prior=np.full(200, 90.0),
                        rope_half_width=50.0,
                        random_seed=42,
                    )
                ],
            ),
        ],
    ).run()

    check_result = result.sensitivity_results[0]
    assert "assurance" in check_result.metadata
    assert isinstance(check_result.metadata["assurance_result"], AssuranceResult)
