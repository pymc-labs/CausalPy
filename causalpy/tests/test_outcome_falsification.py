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
"""Tests for OutcomeFalsification sensitivity check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.checks.base import Check, CheckResult
from causalpy.checks.outcome_falsification import (
    FalsificationResult,
    OutcomeFalsification,
)
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.pipeline import Pipeline, PipelineContext

# ---------------------------------------------------------------------------
# Shared sample_kwargs for fast PyMC tests
# ---------------------------------------------------------------------------

_FAST_SAMPLE_KWARGS = {
    "chains": 2,
    "draws": 100,
    "progressbar": False,
    "random_seed": 42,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_its_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create ITS dataset with y (main outcome) and z, w (falsification outcomes)."""
    rng = np.random.default_rng(seed)
    treatment_time = int(n * 0.75)
    post = np.where(np.arange(n) >= treatment_time, 1, 0)
    return pd.DataFrame(
        {
            "t": np.arange(n),
            "y": rng.normal(size=n) + 5.0 * post,  # main outcome with effect
            "z": rng.normal(size=n),  # falsification outcome (no effect)
            "w": rng.normal(size=n),  # another falsification outcome (no effect)
        }
    )


def _make_pymc_model():
    """Create a minimal PyMC model for ITS tests."""
    return cp.pymc_models.LinearRegression(sample_kwargs=_FAST_SAMPLE_KWARGS)


def _make_ols_model():
    """Create a minimal OLS-compatible model for validation tests."""
    return cp.create_causalpy_compatible_class(LinearRegression())


# ===========================================================================
# Construction tests (unit -- no sampling)
# ===========================================================================


def test_construction_with_valid_formulas():
    """Test constructor stores formulas and default alpha."""
    check = OutcomeFalsification(formulas=["z ~ 1 + t"])
    assert check.formulas == ["z ~ 1 + t"]
    assert check.alpha == 0.05


def test_construction_multiple_formulas():
    """Test constructor stores multiple formulas."""
    check = OutcomeFalsification(formulas=["z ~ 1 + t", "w ~ 1 + t"])
    assert len(check.formulas) == 2


def test_construction_custom_alpha():
    """Test constructor with custom alpha."""
    check = OutcomeFalsification(formulas=["z ~ 1 + t"], alpha=0.10)
    assert check.alpha == 0.10


def test_construction_empty_formulas_raises():
    """Test empty formulas list raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        OutcomeFalsification(formulas=[])


def test_construction_non_string_formula_raises():
    """Test non-string formula raises TypeError."""
    with pytest.raises(TypeError, match="string"):
        OutcomeFalsification(formulas=[123])


def test_construction_invalid_alpha_zero_raises():
    """Test alpha=0 raises ValueError."""
    with pytest.raises(ValueError, match="alpha must be in"):
        OutcomeFalsification(formulas=["z ~ 1 + t"], alpha=0.0)


def test_construction_invalid_alpha_one_raises():
    """Test alpha=1 raises ValueError."""
    with pytest.raises(ValueError, match="alpha must be in"):
        OutcomeFalsification(formulas=["z ~ 1 + t"], alpha=1.0)


def test_satisfies_check_protocol():
    """Test OutcomeFalsification satisfies the Check protocol."""
    assert isinstance(OutcomeFalsification(formulas=["z ~ 1 + t"]), Check)


def test_applicable_methods():
    """Test applicable methods includes ITS and DiD."""
    check = OutcomeFalsification(formulas=["z ~ 1 + t"])
    assert InterruptedTimeSeries in check.applicable_methods
    assert cp.DifferenceInDifferences in check.applicable_methods


def test_repr():
    """Test __repr__ output."""
    check = OutcomeFalsification(formulas=["z ~ 1 + t"], alpha=0.10)
    r = repr(check)
    assert "OutcomeFalsification" in r
    assert "z ~ 1 + t" in r
    assert "0.1" in r


# ===========================================================================
# Validation tests (unit -- no sampling)
# ===========================================================================


def test_validate_rejects_ols_model():
    """Test validate rejects OLS model."""
    df = _make_its_data()
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=150,
        formula="y ~ 1 + t",
        model=_make_ols_model(),
    )
    with pytest.raises(TypeError, match="PyMC model"):
        OutcomeFalsification(formulas=["z ~ 1 + t"]).validate(experiment)


def test_validate_rejects_no_formula():
    """Test validate rejects experiment without formula attribute."""

    class _FakeExperiment:
        model = None

    with pytest.raises(TypeError, match="formula"):
        OutcomeFalsification(formulas=["z ~ 1 + t"]).validate(_FakeExperiment())


# ===========================================================================
# Integration tests (need PyMC)
# ===========================================================================


@pytest.mark.integration
def test_validate_accepts_pymc_its(mock_pymc_sample):
    """Test validate accepts PyMC ITS experiment."""
    df = _make_its_data()
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=150,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    OutcomeFalsification(formulas=["z ~ 1 + t"]).validate(experiment)


@pytest.mark.integration
def test_run_produces_check_result(mock_pymc_sample):
    """Test run produces a CheckResult with correct structure."""
    df = _make_its_data()
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=150,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    context = PipelineContext(data=df)
    context.experiment = experiment
    context.experiment_config = {
        "method": InterruptedTimeSeries,
        "treatment_time": 150,
        "formula": "y ~ 1 + t",
        "model": _make_pymc_model(),
    }

    check = OutcomeFalsification(formulas=["z ~ 1 + t"])
    result = check.run(experiment, context)

    assert isinstance(result, CheckResult)
    assert result.check_name == "OutcomeFalsification"
    assert result.passed is None  # informational, no pass/fail
    assert result.table is not None
    assert len(result.table) == 1
    assert "formula" in result.table.columns
    assert "effect_mean" in result.table.columns


@pytest.mark.integration
def test_run_table_has_hdi_columns(mock_pymc_sample):
    """Test the table includes HDI interval columns."""
    df = _make_its_data()
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=150,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    context = PipelineContext(data=df)
    context.experiment = experiment
    context.experiment_config = {
        "method": InterruptedTimeSeries,
        "treatment_time": 150,
        "formula": "y ~ 1 + t",
        "model": _make_pymc_model(),
    }

    check = OutcomeFalsification(formulas=["z ~ 1 + t"])
    result = check.run(experiment, context)

    table = result.table
    assert "hdi_95%_lower" in table.columns
    assert "hdi_95%_upper" in table.columns
    assert not pd.isna(table["effect_mean"].iloc[0])


@pytest.mark.integration
def test_run_with_multiple_formulas(mock_pymc_sample):
    """Test run with multiple falsification formulas produces multi-row table."""
    df = _make_its_data()
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=150,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    context = PipelineContext(data=df)
    context.experiment = experiment
    context.experiment_config = {
        "method": InterruptedTimeSeries,
        "treatment_time": 150,
        "formula": "y ~ 1 + t",
        "model": _make_pymc_model(),
    }

    check = OutcomeFalsification(formulas=["z ~ 1 + t", "w ~ 1 + t"])
    result = check.run(experiment, context)

    assert result.table is not None
    assert len(result.table) == 2
    assert result.table["formula"].tolist() == ["z ~ 1 + t", "w ~ 1 + t"]


@pytest.mark.integration
def test_run_handles_failed_formula(mock_pymc_sample):
    """Test that a mix of valid and invalid formulas is handled gracefully.

    Note: formulas referencing nonexistent columns trigger a patsy/Python 3.13
    traceback formatting bug (KeyError in patsy.eval), so we test with a
    syntactically invalid formula instead.
    """
    df = _make_its_data()
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=150,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    context = PipelineContext(data=df)
    context.experiment = experiment
    context.experiment_config = {
        "method": InterruptedTimeSeries,
        "treatment_time": 150,
        "formula": "y ~ 1 + t",
        "model": _make_pymc_model(),
    }

    # Mix one good formula with one that has bad syntax
    check = OutcomeFalsification(formulas=["z ~ 1 + t", "~ bad syntax ~"])
    result = check.run(experiment, context)

    # Should still produce a result -- one success, one failure
    assert isinstance(result, CheckResult)
    assert result.table is not None
    assert len(result.table) == 2


@pytest.mark.integration
def test_metadata_contains_results(mock_pymc_sample):
    """Test metadata contains falsification_results and alpha."""
    df = _make_its_data()
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=150,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    context = PipelineContext(data=df)
    context.experiment = experiment
    context.experiment_config = {
        "method": InterruptedTimeSeries,
        "treatment_time": 150,
        "formula": "y ~ 1 + t",
        "model": _make_pymc_model(),
    }

    check = OutcomeFalsification(formulas=["z ~ 1 + t"])
    result = check.run(experiment, context)

    assert "falsification_results" in result.metadata
    assert "alpha" in result.metadata
    assert result.metadata["alpha"] == 0.05

    frs = result.metadata["falsification_results"]
    assert len(frs) == 1
    assert isinstance(frs[0], FalsificationResult)
    assert frs[0].formula == "z ~ 1 + t"
    assert isinstance(frs[0].effect_mean, float)
    assert isinstance(frs[0].hdi_lower, float)
    assert isinstance(frs[0].hdi_upper, float)


@pytest.mark.integration
def test_text_contains_formula_and_effect(mock_pymc_sample):
    """Test text output contains formula and effect size."""
    df = _make_its_data()
    experiment = InterruptedTimeSeries(
        df,
        treatment_time=150,
        formula="y ~ 1 + t",
        model=_make_pymc_model(),
    )
    context = PipelineContext(data=df)
    context.experiment = experiment
    context.experiment_config = {
        "method": InterruptedTimeSeries,
        "treatment_time": 150,
        "formula": "y ~ 1 + t",
        "model": _make_pymc_model(),
    }

    check = OutcomeFalsification(formulas=["z ~ 1 + t"])
    result = check.run(experiment, context)

    assert "z ~ 1 + t" in result.text
    assert "95% HDI" in result.text
    assert "effect" in result.text.lower()


# ===========================================================================
# Pipeline integration test
# ===========================================================================


@pytest.mark.integration
def test_pipeline_integration(mock_pymc_sample):
    """Test OutcomeFalsification works within a full Pipeline."""
    df = _make_its_data()

    result = Pipeline(
        data=df,
        steps=[
            cp.EstimateEffect(
                method=InterruptedTimeSeries,
                treatment_time=150,
                formula="y ~ 1 + t",
                model=_make_pymc_model(),
            ),
            cp.SensitivityAnalysis(
                checks=[
                    OutcomeFalsification(formulas=["z ~ 1 + t"]),
                ],
            ),
        ],
    ).run()

    assert len(result.sensitivity_results) == 1
    check_result = result.sensitivity_results[0]
    assert check_result.check_name == "OutcomeFalsification"
    assert check_result.passed is None  # informational
    assert check_result.table is not None
