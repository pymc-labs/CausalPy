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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.text import Text
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.checks.base import Check, CheckResult
from causalpy.checks.operating_characteristics import (
    AssuranceResult,
    compute_assurance_rates,
)
from causalpy.checks.placebo_in_time import PlaceboFoldResult, PlaceboInTime
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.pipeline import Pipeline, PipelineContext
from causalpy.steps.report import GenerateReport
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
        ).fit()

    return factory


def _fake_plot(*args, **kwargs):
    """Return a throwaway figure so report rendering can encode it."""
    fig, ax = plt.subplots()
    return fig, ax


def _make_fake_bayesian_experiment(
    data: pd.DataFrame, treatment_time: int
) -> SimpleNamespace:
    """Create a lightweight Bayesian-like experiment for no-sampling checks."""
    n_post = max(int((data.index >= treatment_time).sum()), 1)
    post_impact = xr.DataArray(
        np.ones((1, 2, n_post, 1)),
        dims=("chain", "draw", "obs_ind", "treated_units"),
    )
    fake = SimpleNamespace(
        data=data,
        treatment_time=treatment_time,
        _model_backend=SimpleNamespace(supports_idata=True),
        model=SimpleNamespace(),
        result=SimpleNamespace(impact_post=post_impact),
        plot=_fake_plot,
    )
    fake.fit = lambda: fake
    return fake


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0.0},
        {"threshold": 1.0},
        {"threshold": np.nan},
        {"threshold": True},
        {"rope_half_width": -1.0},
        {"rope_half_width": np.inf},
        {"rope_half_width": True},
        {"random_seed": -1},
        {"random_seed": 1.5},
        {"random_seed": True},
    ],
)
def test_invalid_operating_configuration_raises_early(kwargs):
    """Decision, ROPE, and seed controls are valid before fitting starts."""
    with pytest.raises(ValueError):
        PlaceboInTime(**kwargs)


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


@pytest.mark.parametrize("n_design_replications", [0, -1, 1.5, True])
def test_invalid_n_design_replications_raises(n_design_replications):
    """Prior draw counts must be positive, integer, and non-boolean."""
    with pytest.raises(ValueError, match="positive non-bool int"):
        PlaceboInTime(n_design_replications=n_design_replications)


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


def test_repr_with_figures_disabled():
    """A non-default make_figures shows up in the repr."""
    assert "make_figures=False" in repr(PlaceboInTime(make_figures=False))
    assert "make_figures" not in repr(PlaceboInTime())


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
# Exact assurance invariants (unit — no sampling)
# ===========================================================================


def test_exact_assurance_zero_effect_matches_null_rates():
    """A point-mass zero alternative has the exact null operating rates."""
    result = compute_assurance_rates(
        null_samples=np.array([-2.0, 0.0, 2.0]),
        fold_sds=np.array([0.5, 1.0]),
        rope_half_width=1.0,
        threshold=0.95,
        prior=np.array([0.0]),
    )

    assert result.true_positive_rate == result.false_positive_rate
    assert result.false_negative_rate == result.true_negative_rate
    assert result.alt_indeterminate_rate == result.null_indeterminate_rate
    assert result.null_decisions is None
    assert result.alt_decisions is None


def test_exact_assurance_large_effect_is_detected():
    """A large point-mass effect is detected without random replications."""
    result = compute_assurance_rates(
        null_samples=np.zeros(5),
        fold_sds=np.array([0.5]),
        rope_half_width=1.0,
        threshold=0.95,
        prior=np.array([10.0]),
    )

    assert result.true_positive_rate == 1.0
    assert result.false_positive_rate == 0.0
    assert result.false_negative_rate == 0.0
    assert result.alt_indeterminate_rate == 0.0


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
    ).fit()
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
    ).fit()
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
    ).fit()
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
    ).fit()
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
    ).fit()
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
    ).fit()
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
    ).fit()
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
    ).fit()
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
    ).fit()
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
    ).fit()
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
    ).fit()
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
    ).fit()
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
    ).fit()
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
    ).fit()

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
        ).fit()

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


# ===========================================================================
# Single usable fold degeneracy (regression for the tau-scale collapse)
# ===========================================================================


def _make_scaled_fake_experiment(
    data: pd.DataFrame,
    treatment_time: int,
    cumulative_mean: float,
    cumulative_sd: float,
    n_draws: int = 400,
    seed: int = 0,
) -> SimpleNamespace:
    """Create a fake Bayesian experiment with a controlled cumulative impact.

    The returned ``post_impact`` sums (over ``obs_ind``) to a draw-level
    cumulative impact with the requested mean and standard deviation, so the
    downstream hierarchical null and verdict are exercised on a series whose
    scale we choose deterministically.
    """
    rng = np.random.default_rng(seed)
    n_post = max(int((data.index >= treatment_time).sum()), 1)
    totals = rng.normal(cumulative_mean, cumulative_sd, size=n_draws)
    per_obs = (totals / n_post).reshape(1, n_draws, 1, 1)
    post = np.broadcast_to(per_obs, (1, n_draws, n_post, 1)).copy()
    post_impact = xr.DataArray(post, dims=("chain", "draw", "obs_ind", "treated_units"))
    return SimpleNamespace(
        data=data,
        treatment_time=treatment_time,
        _model_backend=SimpleNamespace(supports_idata=True),
        model=SimpleNamespace(),
        result=SimpleNamespace(impact_post=post_impact),
    )


def test_single_usable_fold_skips_assurance_and_renders_report():
    """An inconclusive run must not touch the expected-effect prior."""

    class _GuardPrior:
        def cdf(self, value):  # pragma: no cover - must not be called
            del value
            raise AssertionError("Assurance must not run without a learned null.")

        def sf(self, value):  # pragma: no cover - must not be called
            del value
            raise AssertionError("Assurance must not run without a learned null.")

        def rvs(self, n):  # pragma: no cover - must not be called
            del n
            raise AssertionError("Assurance must not run without a learned null.")

    data = pd.DataFrame({"y": np.zeros(100)}, index=np.arange(100))
    experiment = _make_fake_bayesian_experiment(data, treatment_time=70)
    check = PlaceboInTime(
        n_folds=1,
        intervention_length=30,
        experiment_factory=_make_fake_bayesian_experiment,
        expected_effect_prior=_GuardPrior(),
        rope_half_width=1.0,
    )

    result = check.run(experiment)

    assert result.passed is None
    assert "assurance_result" not in result.metadata
    assert "assurance" not in result.metadata
    assert "null_samples" not in result.metadata
    assert len(result.figures) == 1
    assert any("No null model" in text for text in _figure_texts(result.figures[0]))

    context = PipelineContext(data=data)
    context.experiment = experiment
    context.sensitivity_results = [result]
    context = GenerateReport(include_effect_summary=False).run(context)

    assert "INCONCLUSIVE" in context.report
    plt.close(result.figures[0])


def test_two_fold_fake_run_preserves_status_quo_and_exact_assurance(monkeypatch):
    """A healthy no-MCMC run retains its status-quo object and exact assurance."""
    data = pd.DataFrame({"y": np.zeros(120)}, index=np.arange(120))
    experiment = _make_scaled_fake_experiment(
        data,
        treatment_time=90,
        cumulative_mean=5.0,
        cumulative_sd=1.0,
    )
    status_quo_idata = SimpleNamespace(
        posterior={
            "mu_status_quo": xr.DataArray([0.0]),
            "tau_status_quo": xr.DataArray([1.0]),
        }
    )

    def factory(fold_data, treatment_time):
        return _make_scaled_fake_experiment(
            fold_data,
            treatment_time,
            cumulative_mean=float(treatment_time),
            cumulative_sd=1.0,
            seed=treatment_time,
        )

    def fake_build(fold_means, fold_sds):
        del fold_means, fold_sds
        return status_quo_idata, np.array([-1.0, 1.0])

    check = PlaceboInTime(
        n_folds=2,
        intervention_length=30,
        experiment_factory=factory,
        expected_effect_prior=np.array([10.0]),
        rope_half_width=1.0,
        make_figures=False,
    )
    monkeypatch.setattr(check, "_build_status_quo_model", fake_build)

    result = check.run(experiment)

    assert result.passed is not None
    assert result.metadata["status_quo_idata"] is status_quo_idata
    assert result.metadata["assurance_result"].true_positive_rate == 1.0
    assert result.metadata["assurance_result"].null_decisions is None
    assert result.metadata["assurance_result"].alt_decisions is None


def test_single_fold_directly_does_not_report_supported_on_large_scale():
    """A large-scale series with ``n_folds=1`` must not fabricate SUPPORTED.

    Route 1 to a single usable fold: ``n_folds=1`` requested directly.  With a
    single fold the between-fold spread ``tau_status_quo`` is unidentified and
    collapses to its prior width (~O(1)), so the null distribution loses all
    data scaling.  An actual effect that is well within the fold's own noise at
    the data scale must therefore never be declared "outside the null".
    """
    data = pd.DataFrame({"y": np.zeros(90)}, index=np.arange(90))
    experiment = _make_scaled_fake_experiment(
        data,
        treatment_time=60,
        cumulative_mean=5150.0,
        cumulative_sd=0.0,
    )

    def factory(fold_data, treatment_time):
        return _make_scaled_fake_experiment(
            fold_data,
            treatment_time,
            cumulative_mean=5000.0,
            cumulative_sd=300.0,
            seed=1,
        )

    check = PlaceboInTime(
        n_folds=1,
        intervention_length=30,
        experiment_factory=factory,
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
        random_seed=42,
    )

    result = check.run(experiment)

    assert result.metadata["n_folds_completed"] == 1
    # A single usable fold cannot characterise the null distribution, so the
    # verdict must abstain rather than (spuriously) claim the effect is real.
    assert result.passed is None
    assert "SUPPORTED" not in result.text
    assert "INCONCLUSIVE" in result.text
    # No degenerate null was built.
    assert "null_samples" not in result.metadata
    assert "p_effect_outside_null" not in result.metadata


def test_skips_down_to_single_fold_does_not_report_supported_on_large_scale():
    """Skips reducing ``n_folds>1`` to one usable fold must not report SUPPORTED.

    Route 2 to a single usable fold: ``n_folds=2`` requested, but the earlier
    fold is skipped for insufficient pre-period, leaving exactly one usable
    fold.  This must reach the same abstention as the direct ``n_folds=1``
    route rather than build a degenerate single-fold null.
    """
    data = pd.DataFrame({"y": np.zeros(100)}, index=np.arange(100))
    experiment = _make_scaled_fake_experiment(
        data,
        treatment_time=70,
        cumulative_mean=5150.0,
        cumulative_sd=0.0,
    )

    def factory(fold_data, treatment_time):
        return _make_scaled_fake_experiment(
            fold_data,
            treatment_time,
            cumulative_mean=5000.0,
            cumulative_sd=300.0,
            seed=1,
        )

    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=factory,
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
        random_seed=42,
    )

    with pytest.warns(UserWarning, match="shorter than one full intervention window"):
        result = check.run(experiment)

    assert result.metadata["n_folds_requested"] == 2
    assert result.metadata["n_folds_completed"] == 1
    assert result.passed is None
    assert "SUPPORTED" not in result.text
    assert "INCONCLUSIVE" in result.text
    assert "null_samples" not in result.metadata
    assert "p_effect_outside_null" not in result.metadata


def test_build_status_quo_model_raises_when_between_fold_spread_unidentified():
    """>=2 folds with identical means must not fabricate a scale-free null.

    The single-fold routes are abstained in :meth:`PlaceboInTime.run` before
    the model is built.  This guards the residual root cause directly: when the
    completed folds have no between-fold spread (``np.nanstd(fold_means) == 0``,
    e.g. an almost-constant series with >= 2 folds), the old ``prior_mu_scale``
    fallback to ``1.0`` stripped all data scaling and collapsed the null to an
    O(1) width, flipping the verdict to a spurious SUPPORTED.  It must now fail
    loudly rather than build that null.
    """
    check = PlaceboInTime(sample_kwargs=_FAST_HIERARCHICAL_KWARGS)
    with pytest.raises(ValueError, match="Cannot identify the hierarchical"):
        check._build_status_quo_model(
            np.array([5000.0, 5000.0]),
            np.array([300.0, 300.0]),
        )


def test_identical_folds_are_inconclusive_end_to_end():
    """A near-constant large-scale series must never surface as SUPPORTED.

    Two folds complete (so the single-fold count guard does not apply) but with
    identical cumulative impacts, so the between-fold null spread is
    unidentified.  ``run`` must abstain (INCONCLUSIVE) — mirroring the
    single-fold routes and PlaceboInSpace — rather than build a degenerate null
    or crash the caller.
    """
    data = pd.DataFrame({"y": np.zeros(120)}, index=np.arange(120))
    experiment = _make_scaled_fake_experiment(
        data,
        treatment_time=90,
        cumulative_mean=5150.0,
        cumulative_sd=0.0,
    )

    def factory(fold_data, treatment_time):
        return _make_scaled_fake_experiment(
            fold_data,
            treatment_time,
            cumulative_mean=5000.0,
            cumulative_sd=0.0,
            seed=1,
        )

    check = PlaceboInTime(
        n_folds=2,
        intervention_length=30,
        experiment_factory=factory,
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
        random_seed=42,
    )

    result = check.run(experiment)

    assert result.metadata["n_folds_completed"] == 2
    assert result.passed is None
    assert "SUPPORTED" not in result.text
    assert "INCONCLUSIVE" in result.text
    assert "null_samples" not in result.metadata
    assert "p_effect_outside_null" not in result.metadata


def test_two_folds_with_distinct_means_still_produce_a_verdict():
    """The degeneracy guards must not over-fire on a healthy multi-fold run.

    Two folds complete with *distinct* cumulative impacts, so the between-fold
    spread is identified.  ``run`` must build the null and return a boolean
    verdict rather than abstaining — this pins that the abstention path is
    reached only for genuinely degenerate configurations.
    """
    data = pd.DataFrame({"y": np.zeros(120)}, index=np.arange(120))
    experiment = _make_scaled_fake_experiment(
        data,
        treatment_time=90,
        cumulative_mean=5000.0,
        cumulative_sd=0.0,
    )

    def factory(fold_data, treatment_time):
        # Distinct per-fold means (folds sit at pseudo tt 30 and 60), so
        # np.nanstd(fold_means) > 0 and the null is identified.
        return _make_scaled_fake_experiment(
            fold_data,
            treatment_time,
            cumulative_mean=5000.0 + 2.0 * treatment_time,
            cumulative_sd=300.0,
            seed=1,
        )

    check = PlaceboInTime(
        n_folds=2,
        intervention_length=30,
        experiment_factory=factory,
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
        random_seed=42,
    )

    result = check.run(experiment)

    assert result.metadata["n_folds_completed"] == 2
    assert result.passed is not None
    assert "INCONCLUSIVE" not in result.text
    assert "null_samples" in result.metadata
    assert "p_effect_outside_null" in result.metadata


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
            first_fold.experiment.result.impact_post,
            second_fold.experiment.result.impact_post,
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
    ).fit()
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
    ).fit()
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
    ).fit()
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
    ).fit()
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
# Exact expected-effect-prior routing (unit — no sampling)
# ===========================================================================


def test_compute_assurance_passes_array_prior_directly(monkeypatch):
    """Arrays are consumed unchanged without cycling or draw-count effects."""
    prior = np.array([1.0, 2.0, 3.0])
    captured: list[object] = []

    def fake_compute(*args):
        captured.extend(args)
        return AssuranceResult(0.5, 0.1, 0.2, 0.1, 0.7, 0.4)

    check = PlaceboInTime(
        expected_effect_prior=prior,
        rope_half_width=0.5,
        n_design_replications=1,
    )
    monkeypatch.setattr(
        "causalpy.checks.placebo_in_time.compute_assurance_rates", fake_compute
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check._compute_assurance(
            theta_new_samples=np.array([-1.0, 1.0]),
            fold_sds=np.array([1.0]),
        )

    assert captured[4] is prior


def test_compute_assurance_uses_frozen_prior_without_rvs():
    """Frozen priors follow the exact CDF/SF path and never draw samples."""

    class _FrozenPrior:
        def __init__(self):
            self.cdf_calls = 0
            self.sf_calls = 0

        def cdf(self, value):
            self.cdf_calls += 1
            return np.zeros_like(value, dtype=float)

        def sf(self, value):
            self.sf_calls += 1
            return np.ones_like(value, dtype=float)

        def rvs(self, n):  # pragma: no cover - exact path must not call this
            del n
            raise AssertionError("Frozen priors must not be sampled.")

    prior = _FrozenPrior()
    result = PlaceboInTime(
        expected_effect_prior=prior,
        rope_half_width=1.0,
        n_design_replications=1,
    )._compute_assurance(
        theta_new_samples=np.array([-1.0, 1.0]),
        fold_sds=np.array([0.5]),
    )

    assert result.true_positive_rate == 1.0
    assert result.false_negative_rate == 0.0
    assert prior.cdf_calls > 0
    assert prior.sf_calls > 0


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


def test_compute_assurance_uses_design_count_only_for_rvs_prior():
    """The optional design count controls only RVS-only prior draws."""

    class _RVSOnlyPrior:
        def __init__(self):
            self.draw_counts: list[int] = []

        def rvs(self, n):
            self.draw_counts.append(n)
            return np.full(n, 10.0)

    prior = _RVSOnlyPrior()
    result = PlaceboInTime(
        expected_effect_prior=prior,
        rope_half_width=1.0,
        n_design_replications=3,
    )._compute_assurance(
        theta_new_samples=np.array([-1.0, 1.0]),
        fold_sds=np.array([0.5]),
    )

    assert prior.draw_counts == [3]
    assert result.true_positive_rate == 1.0


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
    ).fit()
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


# ===========================================================================
# Calibration figure
# ===========================================================================


def _make_calibration_check_result(
    n_folds: int = 2,
    with_null: bool = True,
    pseudo_treatment_times: list | None = None,
) -> CheckResult:
    """Build a CheckResult carrying only what plot_calibration reads."""
    rng = np.random.default_rng(0)
    times = pseudo_treatment_times or [100 * (i + 1) for i in range(n_folds)]
    fold_results = [
        PlaceboFoldResult(
            fold=i + 1,
            pseudo_treatment_time=times[i],
            experiment=None,  # type: ignore[arg-type]
            cumulative_impact_samples=xr.DataArray(rng.normal(size=50)),
            fold_mean=float(i),
            fold_sd=1.0,
        )
        for i in range(len(times))
    ]
    metadata: dict = {"fold_results": fold_results}
    if with_null:
        metadata.update(
            null_samples=rng.normal(size=200),
            actual_cumulative_samples=rng.normal(loc=5.0, size=200),
            actual_cumulative_mean=5.0,
            p_effect_outside_null=0.97,
        )
    return CheckResult(check_name="PlaceboInTime", passed=True, metadata=metadata)


def _figure_texts(fig: Figure) -> list[str]:
    """Collect every rendered string in a figure.

    plotnine draws panel titles and legend labels as free text artists rather
    than through ``Axes.set_title``, so assertions read them from here.
    """
    return [text.get_text() for text in fig.findobj(Text)]


def test_plot_calibration_returns_three_panels():
    """The calibration plot has one panel per diagnostic."""
    fig = PlaceboInTime.plot_calibration(_make_calibration_check_result())

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 3
    texts = _figure_texts(fig)
    assert any(text.startswith("A. Placebo fold distributions") for text in texts)
    assert any(text.startswith("B. Learned null distribution") for text in texts)
    assert any(text.startswith("C. Actual effect vs null") for text in texts)
    plt.close(fig)


def test_plot_calibration_honours_figsize():
    """The caller's figure size survives plotnine's own layout pass."""
    fig = PlaceboInTime.plot_calibration(
        _make_calibration_check_result(), figsize=(5.0, 11.0)
    )

    assert tuple(fig.get_size_inches()) == (5.0, 11.0)
    plt.close(fig)


def test_plot_calibration_sets_the_suptitle():
    """The caller's title reaches the drawn figure."""
    fig = PlaceboInTime.plot_calibration(
        _make_calibration_check_result(), title="UK Coal CO2"
    )

    assert "UK Coal CO2" in _figure_texts(fig)
    plt.close(fig)


def test_plot_calibration_labels_datetime_folds_by_year():
    """Datetime pseudo treatment times are formatted, not repr'd."""
    result = _make_calibration_check_result(
        pseudo_treatment_times=[pd.Timestamp("2015-06-01")]
    )
    fig = PlaceboInTime.plot_calibration(result)

    assert "Fold 1 (t*=2015)" in _figure_texts(fig)
    plt.close(fig)


def test_plot_calibration_colors_more_folds_than_the_cycle():
    """More folds than the colour cycle must not raise."""
    result = _make_calibration_check_result(n_folds=12)
    fig = PlaceboInTime.plot_calibration(result)

    texts = _figure_texts(fig)
    assert all(
        any(text.startswith(f"Fold {fold} (") for text in texts)
        for fold in range(1, 13)
    )
    plt.close(fig)


def test_plot_calibration_warns_when_no_null_model():
    """A run without a null model warns instead of printing."""
    result = _make_calibration_check_result(with_null=False)

    with pytest.warns(UserWarning, match="Not enough folds completed"):
        fig = PlaceboInTime.plot_calibration(result)

    assert isinstance(fig, Figure)
    assert any("No null model: 2 folds completed." in t for t in _figure_texts(fig))
    plt.close(fig)


def test_inconclusive_run_still_produces_a_figure():
    """figures[0] is safe to read even when the run reaches no verdict."""
    data = pd.DataFrame({"y": np.zeros(100)}, index=np.arange(100))
    experiment = _make_fake_bayesian_experiment(data, treatment_time=50)

    def factory(fold_data, treatment_time):  # pragma: no cover - must not fit
        del fold_data, treatment_time
        raise AssertionError("Ineligible folds must be skipped before fitting.")

    check = PlaceboInTime(n_folds=1, experiment_factory=factory)

    with pytest.warns(
        UserWarning, match="shorter than one full intervention window"
    ) as record:
        result = check.run(experiment)

    assert result.passed is None
    assert len(result.figures) == 1
    assert any("No null model" in t for t in _figure_texts(result.figures[0]))
    # run() already says so through passed=None and the result text, so it
    # must not also warn about the missing null model.
    assert len(record) == 1
    plt.close(result.figures[0])


def test_inconclusive_run_without_figures_stays_empty():
    """make_figures=False opts out of the placeholder too."""
    data = pd.DataFrame({"y": np.zeros(100)}, index=np.arange(100))
    experiment = _make_fake_bayesian_experiment(data, treatment_time=50)

    def factory(fold_data, treatment_time):  # pragma: no cover - must not fit
        del fold_data, treatment_time
        raise AssertionError("Ineligible folds must be skipped before fitting.")

    check = PlaceboInTime(n_folds=1, experiment_factory=factory, make_figures=False)

    with pytest.warns(UserWarning, match="shorter than one full intervention window"):
        result = check.run(experiment)

    assert result.figures == []


@pytest.mark.integration
def test_run_populates_figures_by_default(mock_pymc_sample):
    """run() attaches the calibration figure to the result."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    ).fit()
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)

    assert len(result.figures) == 1
    assert isinstance(result.figures[0], Figure)
    assert len(result.figures[0].axes) == 3
    plt.close(result.figures[0])


@pytest.mark.integration
def test_run_figure_is_not_registered_with_pyplot(mock_pymc_sample):
    """The default figure must not enter pyplot's figure manager.

    That is what keeps ``make_figures=True`` free of side effects: nothing to
    close, and no auto-display at the end of the notebook cell that ran the
    check.
    """
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    ).fit()
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    plt.close("all")
    result = check.run(experiment)

    assert result.figures
    assert plt.get_fignums() == []


@pytest.mark.integration
def test_check_figure_reaches_the_generated_report(mock_pymc_sample):
    """A figure built by the check survives rendering into the report."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    ).fit()
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
    )
    result = check.run(experiment)

    context = PipelineContext(data=df)
    context.experiment = experiment
    context.sensitivity_results = [result]
    context = GenerateReport(include_effect_summary=False).run(context)

    assert "PlaceboInTime figure" in context.report
    assert "data:image/png;base64," in context.report
    plt.close(result.figures[0])


@pytest.mark.integration
def test_run_metadata_carries_actual_cumulative_samples(mock_pymc_sample):
    """The plotter's actual-effect samples come from run(), not the caller."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    ).fit()
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
        make_figures=False,
    )
    result = check.run(experiment)

    samples = result.metadata["actual_cumulative_samples"]
    assert samples.ndim == 1
    assert (
        samples.size
        == experiment.result.impact_post.sizes["chain"]
        * (experiment.result.impact_post.sizes["draw"])
    )


@pytest.mark.integration
def test_run_without_figures_leaves_figures_empty(mock_pymc_sample):
    """make_figures=False opts out of the figure."""
    df = _make_its_data(n=2000)
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=1500,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    ).fit()
    check = PlaceboInTime(
        n_folds=2,
        experiment_factory=_make_pymc_factory(),
        sample_kwargs=_FAST_HIERARCHICAL_KWARGS,
        make_figures=False,
    )
    result = check.run(experiment)

    assert result.figures == []
