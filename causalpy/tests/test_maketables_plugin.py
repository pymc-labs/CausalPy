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
"""Tests for optional maketables plugin hooks on experiment objects."""

import arviz as az
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.maketables_adapters import _extract_hdi_bounds

sample_kwargs = {
    "chains": 2,
    "draws": 100,
    "progressbar": False,
    "random_seed": 42,
}


@pytest.mark.integration
def test_maketables_coef_table_pymc_contract(mock_pymc_sample):
    """PyMC-backed experiments should expose canonical maketables columns."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    table = result.__maketables_coef_table__

    assert isinstance(table, pd.DataFrame)
    assert table.index.name == "Coefficient"
    for col in ["b", "se", "p", "t", "ci95l", "ci95u"]:
        assert col in table.columns
    assert list(table.index) == list(result.labels)
    assert table["b"].notna().all()
    assert table["se"].notna().all()
    assert table["p"].isna().all()
    assert table["ci95l"].notna().all()
    assert table["ci95u"].notna().all()


@pytest.mark.integration
def test_maketables_coef_table_sklearn_contract(mock_pymc_sample):
    """Sklearn-backed experiments should expose canonical columns with NaN inference."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )

    table = result.__maketables_coef_table__

    assert isinstance(table, pd.DataFrame)
    assert table.index.name == "Coefficient"
    for col in ["b", "se", "p", "t", "ci95l", "ci95u"]:
        assert col in table.columns
    assert list(table.index) == list(result.labels)
    assert table["b"].notna().all()
    assert table["se"].isna().all()
    assert table["p"].isna().all()
    assert table["t"].isna().all()


@pytest.mark.integration
def test_maketables_multi_treated_unit_requires_explicit_selection(mock_pymc_sample):
    """PyMC coefficient export raises for ambiguous multi-treated-unit outputs."""
    rng = np.random.default_rng(42)
    n_obs = 50
    n_control = 3
    n_treated = 2
    time_index = pd.date_range("2020-01-01", periods=n_obs, freq="D")
    treatment_time = time_index[30]

    control_data = {
        f"control_{i}": np.cumsum(rng.normal(0.0, 0.2, n_obs)) + 10.0 + i
        for i in range(n_control)
    }
    treated_data = {}
    for j in range(n_treated):
        base = sum(
            (i + 1) / (n_control * (n_control + 1) / 2) * control_data[f"control_{i}"]
            for i in range(n_control)
        )
        treatment_effect = np.zeros(n_obs)
        treatment_effect[30:] = 2.0 + 0.1 * j
        treated_data[f"treated_{j}"] = (
            base + treatment_effect + rng.normal(0.0, 0.1, n_obs)
        )

    df = pd.DataFrame({**control_data, **treated_data}, index=time_index)

    result = cp.SyntheticControl(
        df,
        treatment_time,
        control_units=[f"control_{i}" for i in range(n_control)],
        treated_units=[f"treated_{j}" for j in range(n_treated)],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )

    with pytest.raises(ValueError, match="Ambiguous multi-treated-unit"):
        _ = result.__maketables_coef_table__


@pytest.mark.integration
def test_maketables_stat_and_metadata_hooks(mock_pymc_sample):
    """Stats, labels, defaults, depvar, and vcov hooks return stable metadata."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )

    assert result.__maketables_stat__("N") is not None
    assert result.__maketables_stat__("unknown_key") is None
    assert result.__maketables_depvar__ == result.outcome_variable_name
    assert result.__maketables_vcov_info__["se_type"] == "Not available"
    assert result.__maketables_stat_labels__["N"] == "N"
    assert "N" in (result.__maketables_default_stat_keys__ or [])


def test_maketables_depvar_fallback_for_ipw():
    """Experiments with outcome_variable should expose it via depvar hook."""
    df = cp.load_data("nhefs")
    result = cp.InversePropensityWeighting(
        data=df,
        formula="trt ~ 1 + age + race",
        outcome_variable="outcome",
        weighting_scheme="robust",
    )

    assert result.__maketables_depvar__ == "outcome"


@pytest.mark.integration
def test_maketables_hdi_prob_user_control(mock_pymc_sample):
    """Users can control HDI level for maketables coefficient intervals."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    result.set_maketables_options(hdi_prob=0.8)

    table = result.__maketables_coef_table__
    first_label = result.labels[0]
    draws = (
        result.model.idata.posterior["beta"]
        .isel(treated_units=0)
        .sel(coeffs=first_label)
    )
    hdi = az.hdi(draws, hdi_prob=0.8)
    expected_lower, expected_upper = _extract_hdi_bounds(hdi)

    assert table.loc[first_label, "ci95l"] == pytest.approx(expected_lower)
    assert table.loc[first_label, "ci95u"] == pytest.approx(expected_upper)


def test_maketables_hdi_prob_validation(mock_pymc_sample):
    """Invalid HDI probability should raise an explicit ValueError."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with pytest.raises(ValueError, match="hdi_prob must be in \\(0, 1\\)"):
        result.set_maketables_options(hdi_prob=1.2)
