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
"""Deterministic Bayesian parameter-recovery contracts for DiD and SC.

These tests deliberately reuse the extraction and compatibility semantics from the
permanent migration-baseline harness without participating in its artifact protocol.
They test known data-generating parameters, not a historical PyMC posterior.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

import causalpy as cp

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPO_ROOT / "scripts" / "migration_baseline" / "harness.py"

PARAMETER_RECOVERY_SEED = 157
PARAMETER_RECOVERY_CHAINS = 4
PARAMETER_RECOVERY_DRAWS = 500
PARAMETER_RECOVERY_TUNE = 500
EXPECTED_DRAW_SHAPE = (
    PARAMETER_RECOVERY_CHAINS,
    PARAMETER_RECOVERY_DRAWS,
)

# Tolerance calibration policy. Every limit below is set from the statistics
# actually produced by the seeded fits (seed 157, 4 chains, 500 draws/tune), with
# roughly 2x-4x headroom, so the gates stay informative instead of unfalsifiable:
#
#   gate                                observed        limit
#   DiD |z| (worst coefficient)         1.97            2.5
#   DiD posterior SD (worst)            0.072           0.16
#   DiD draw-wise R^2                   0.968           >= 0.90
#   SC  |z| (worst estimand)            1.25            3.0
#   SC  weight posterior SD (worst)     0.0089          0.025
#   SC  average-impact posterior SD     0.00039         0.002
#   SC  first/last impact posterior SD  0.012           0.05
#   SC  cumulative posterior SD         0.012           0.05
#   SC  counterfactual RMSE             0.011           0.03
#   SC  counterfactual max abs error    0.015           0.05
#   SC  draw-wise R^2                   0.835           >= 0.80
#
# The errors are dominated by the fixed simulated data realization, not by Monte
# Carlo noise (MCSE is ~2% of each posterior SD here), so the headroom absorbs
# sampler and platform differences without letting a genuine regression through.
DID_UNITS_PER_GROUP = 50
DID_TRUE_COEFFICIENTS = {
    "Intercept": 4.0,
    "group": -1.0,
    "post_treatment": 1.5,
    "group:post_treatment": 2.0,
}
DID_MAX_STANDARDIZED_ERROR = 2.5
DID_MAX_POSTERIOR_SD = 0.16
DID_MIN_DRAW_WISE_R2 = 0.90

SC_TREATMENT_TIME = 64
SC_N_POST = 31
SC_CONTROL_UNITS = ("a", "b", "c")
SC_TRUE_WEIGHTS = {"a": 0.20, "b": 0.50, "c": 0.30}
SC_EFFECT_PATH = np.linspace(-0.50, -2.00, SC_N_POST)
SC_TRUE_AVERAGE_IMPACT = -1.25
SC_TRUE_FINAL_CUMULATIVE_IMPACT = -38.75
SC_MAX_STANDARDIZED_ERROR = 3.0
SC_MAX_WEIGHT_POSTERIOR_SD = 0.025
SC_MAX_AVERAGE_IMPACT_POSTERIOR_SD = 0.002
SC_MAX_IMPACT_POSTERIOR_SD = 0.05
SC_MAX_CUMULATIVE_POSTERIOR_SD = 0.05
SC_MIN_DRAW_WISE_R2 = 0.80
SC_MAX_COUNTERFACTUAL_RMSE = 0.03
SC_MAX_COUNTERFACTUAL_ABSOLUTE_ERROR = 0.05


def _load_harness_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "migration_baseline_harness_parameter_recovery", HARNESS_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_harness_module()


def _simulate_did_records() -> tuple[tuple[dict[str, Any], ...], dict[str, float]]:
    """Return a seeded, identified two-period DiD DGP and its coefficient truths."""
    rng = np.random.default_rng(PARAMETER_RECOVERY_SEED)
    records: list[dict[str, Any]] = []

    for group in (0, 1):
        for unit in range(DID_UNITS_PER_GROUP):
            for post_treatment in (0, 1):
                mean = (
                    DID_TRUE_COEFFICIENTS["Intercept"]
                    + DID_TRUE_COEFFICIENTS["group"] * group
                    + DID_TRUE_COEFFICIENTS["post_treatment"] * post_treatment
                    + DID_TRUE_COEFFICIENTS["group:post_treatment"]
                    * group
                    * post_treatment
                )
                records.append(
                    {
                        "unit": f"group-{group}-unit-{unit}",
                        "t": post_treatment,
                        "group": group,
                        "post_treatment": post_treatment,
                        "y": mean + rng.normal(scale=0.25),
                    }
                )

    return tuple(records), DID_TRUE_COEFFICIENTS.copy()


def _simulate_synthetic_control_records() -> tuple[
    tuple[dict[str, Any], ...], dict[str, np.ndarray]
]:
    """Return an identified simplex DGP with noiseless post-treatment outcomes."""
    rng = np.random.default_rng(PARAMETER_RECOVERY_SEED)
    time = np.arange(SC_TREATMENT_TIME + SC_N_POST)
    phase = 2 * np.pi * time / 16
    controls = np.column_stack(
        (
            5 + np.sin(phase),
            5 + np.cos(phase),
            5 - np.sin(phase) - np.cos(phase),
        )
    )
    weights = np.array([SC_TRUE_WEIGHTS[unit] for unit in SC_CONTROL_UNITS])
    counterfactual = controls @ weights
    actual = counterfactual.copy()
    actual[:SC_TREATMENT_TIME] += rng.normal(scale=0.06, size=SC_TREATMENT_TIME)
    actual[SC_TREATMENT_TIME:] += SC_EFFECT_PATH

    records = tuple(
        {
            "t": int(time_point),
            "a": float(control_values[0]),
            "b": float(control_values[1]),
            "c": float(control_values[2]),
            "actual": float(outcome),
        }
        for time_point, control_values, outcome in zip(
            time, controls, actual, strict=True
        )
    )
    return records, {
        "counterfactual": counterfactual,
        "effect_path": SC_EFFECT_PATH.copy(),
    }


def _finite_scalar(value: Any, label: str) -> float:
    """Return one finite scalar using the baseline harness validation semantics."""
    return harness._finite_scalar(value, label, np)


def _summarize_focal_draws(draws: Any, label: str) -> dict[str, float]:
    """Summarize one focal scalar without relaxing the baseline's fixed-shape path."""
    values = np.asarray(draws, dtype=float)
    if values.shape != EXPECTED_DRAW_SHAPE:
        raise AssertionError(
            f"{label} must have shape {EXPECTED_DRAW_SHAPE}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise AssertionError(f"{label} contains non-finite posterior draws")

    interval = harness._hdi_interval(values.reshape(-1), az, np)
    if interval.shape != (2,):
        raise AssertionError(f"{label} returned invalid HDI shape {interval.shape}")

    summary = {
        "mean": _finite_scalar(values.mean(), f"{label} mean"),
        "posterior_sd": _finite_scalar(values.std(ddof=1), f"{label} posterior SD"),
        "rhat": _finite_scalar(az.rhat(values, method="rank"), f"{label} R-hat"),
        "ess_bulk": _finite_scalar(az.ess(values, method="bulk"), f"{label} bulk ESS"),
        "ess_tail": _finite_scalar(harness._tail_ess(values, az), f"{label} tail ESS"),
        "hdi_lower": _finite_scalar(interval[0], f"{label} HDI lower"),
        "hdi_upper": _finite_scalar(interval[1], f"{label} HDI upper"),
    }
    if summary["posterior_sd"] <= 0:
        raise AssertionError(f"{label} has non-positive posterior SD")
    if summary["hdi_lower"] > summary["hdi_upper"]:
        raise AssertionError(f"{label} has inverted {harness.HDI_PROB:.2f} HDI bounds")
    sampling = harness.REGISTERED_SAMPLING
    if summary["rhat"] > sampling.max_rhat:
        raise AssertionError(
            f"{label} has R-hat {summary['rhat']:.6g}, above {sampling.max_rhat:.6g}"
        )
    if summary["ess_bulk"] < sampling.min_ess_bulk:
        raise AssertionError(
            f"{label} has bulk ESS {summary['ess_bulk']:.6g}, below "
            f"{sampling.min_ess_bulk:.6g}"
        )
    if summary["ess_tail"] < sampling.min_ess_tail:
        raise AssertionError(
            f"{label} has tail ESS {summary['ess_tail']:.6g}, below "
            f"{sampling.min_ess_tail:.6g}"
        )
    return summary


def _capture_focal_series(
    name: str,
    data: Any,
    *,
    expected_dims: tuple[str, ...] | None = None,
    expected_name: str | None = None,
) -> dict[str, Any]:
    """Capture focal scalar summaries with #1048 coordinate semantics."""
    if expected_dims is not None and tuple(data.dims) != expected_dims:
        raise AssertionError(
            f"{name} dimensions changed: expected {expected_dims!r}, "
            f"got {tuple(data.dims)!r}"
        )
    if expected_name is not None and data.name != expected_name:
        raise AssertionError(
            f"{name} must be extracted from {expected_name!r}, got {data.name!r}"
        )

    value_dimensions = [
        dimension for dimension in data.dims if dimension not in {"chain", "draw"}
    ]
    ordered = data.transpose("chain", "draw", *value_dimensions)
    semantics = harness._array_semantics(ordered)
    values = np.asarray(ordered.values, dtype=float)
    value_shape = values.shape[2:]
    indexes = [()] if not value_shape else list(np.ndindex(value_shape))
    metrics: list[dict[str, Any]] = []

    for index in indexes:
        selector = {
            dimension: semantics["coords"][dimension][offset]
            for dimension, offset in zip(value_dimensions, index, strict=True)
        }
        draws = values[(slice(None), slice(None), *index)]
        metric_name = harness._metric_id(name, selector)
        metrics.append(
            {
                "id": metric_name,
                "selector": selector,
                "summary": _summarize_focal_draws(draws, metric_name),
            }
        )
    return {"name": name, "semantics": semantics, "metrics": metrics}


def _assert_posterior_array_semantics(
    data: Any,
    *,
    expected_dims: tuple[str, ...],
    expected_name: str,
) -> dict[str, Any]:
    """Assert canonical posterior-array semantics without a family-wise ESS gate."""
    if tuple(data.dims) != expected_dims:
        raise AssertionError(
            f"Expected dimensions {expected_dims!r}, got {tuple(data.dims)!r}"
        )
    if data.name != expected_name:
        raise AssertionError(
            f"Expected posterior variable {expected_name!r}, got {data.name!r}"
        )
    return harness._array_semantics(data)


def _parameter_recovery_gate(
    summary: dict[str, float],
    *,
    truth: float,
    max_standardized_error: float,
    max_posterior_sd: float,
) -> dict[str, float | bool]:
    """Evaluate preregistered recovery and informativeness gates for one scalar."""
    required = ("mean", "posterior_sd", "hdi_lower", "hdi_upper")
    try:
        values = {name: float(summary[name]) for name in required}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Recovery summary is missing a numeric required statistic"
        ) from error

    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"Recovery summary {name!r} must be finite")
    for name, value in {
        "truth": truth,
        "max_standardized_error": max_standardized_error,
        "max_posterior_sd": max_posterior_sd,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"Recovery gate {name!r} must be finite")
    if values["posterior_sd"] <= 0:
        raise ValueError("Recovery summary posterior SD must be positive")
    if values["hdi_lower"] > values["hdi_upper"]:
        raise ValueError("Recovery summary HDI bounds must not be inverted")
    if max_standardized_error < 0 or max_posterior_sd < 0:
        raise ValueError("Recovery gate limits must be non-negative")

    absolute_error = abs(values["mean"] - truth)
    standardized_error = absolute_error / values["posterior_sd"]
    point_recovery_passed = standardized_error <= max_standardized_error
    informativeness_passed = values["posterior_sd"] <= max_posterior_sd
    return {
        "truth": truth,
        "posterior_mean": values["mean"],
        "posterior_sd": values["posterior_sd"],
        "absolute_error": absolute_error,
        "standardized_error": standardized_error,
        "max_standardized_error": max_standardized_error,
        "max_posterior_sd": max_posterior_sd,
        "point_recovery_passed": point_recovery_passed,
        "informativeness_passed": informativeness_passed,
        "hdi_contains_truth_diagnostic": (
            values["hdi_lower"] <= truth <= values["hdi_upper"]
        ),
        "passed": point_recovery_passed and informativeness_passed,
    }


def _assert_recovery_gate(gate: dict[str, float | bool], label: str) -> None:
    """Assert both hard gates while retaining HDI coverage as a diagnostic only."""
    assert gate["point_recovery_passed"], f"{label} point recovery failed: {gate}"
    assert gate["informativeness_passed"], f"{label} informativeness failed: {gate}"
    assert gate["passed"], f"{label} combined recovery gate failed: {gate}"


def _record_gate_diagnostics(
    record_property: Callable[[str, object], None],
    prefix: str,
    gate: dict[str, float | bool],
) -> None:
    """Expose recovery diagnostics without making HDI containment a hard assertion."""
    for name, value in gate.items():
        record_property(f"{prefix}_{name}", str(value))


@pytest.fixture(scope="module")
def did_parameter_recovery_data() -> tuple[pd.DataFrame, dict[str, float]]:
    """Return a fresh DataFrame for the DiD parameter-recovery scenario."""
    records, truths = _simulate_did_records()
    return pd.DataFrame(harness._records_payload(records)), truths


@pytest.fixture(scope="module")
def sc_parameter_recovery_data() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Return a fresh indexed DataFrame for the SC parameter-recovery scenario."""
    records, dgp = _simulate_synthetic_control_records()
    return pd.DataFrame(harness._records_payload(records)).set_index("t"), dgp


@pytest.fixture(scope="module")
def parameter_recovery_sample_kwargs() -> dict[str, Any]:
    """Use #1048's sampler compatibility policy at the smaller test budget."""
    sample_kwargs = harness._sample_kwargs(pm, harness.REGISTERED_SAMPLING)
    sample_kwargs.update(
        {
            "chains": PARAMETER_RECOVERY_CHAINS,
            "draws": PARAMETER_RECOVERY_DRAWS,
            "tune": PARAMETER_RECOVERY_TUNE,
            "random_seed": PARAMETER_RECOVERY_SEED,
        }
    )
    assert sample_kwargs["cores"] == 1
    assert sample_kwargs["target_accept"] == harness.REGISTERED_SAMPLING.target_accept
    return sample_kwargs


@pytest.fixture(scope="module")
def did_parameter_recovery_result(
    did_parameter_recovery_data: tuple[pd.DataFrame, dict[str, float]],
    parameter_recovery_sample_kwargs: dict[str, Any],
) -> cp.DifferenceInDifferences:
    """Fit the seeded DiD scenario once for all of its contract assertions."""
    data, _ = did_parameter_recovery_data
    return cp.DifferenceInDifferences(
        data.copy(),
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(
            sample_kwargs=dict(parameter_recovery_sample_kwargs)
        ),
    )


@pytest.fixture(scope="module")
def sc_parameter_recovery_result(
    sc_parameter_recovery_data: tuple[pd.DataFrame, dict[str, np.ndarray]],
    parameter_recovery_sample_kwargs: dict[str, Any],
) -> cp.SyntheticControl:
    """Fit the seeded SC scenario once for all of its contract assertions."""
    data, _ = sc_parameter_recovery_data
    return cp.SyntheticControl(
        data.copy(),
        treatment_time=SC_TREATMENT_TIME,
        control_units=list(SC_CONTROL_UNITS),
        treated_units=["actual"],
        min_donor_correlation=-1.0,
        model=cp.pymc_models.WeightedSumFitter(
            sample_kwargs=dict(parameter_recovery_sample_kwargs)
        ),
    )


def test_parameter_recovery_dgps_are_identified() -> None:
    """The deterministic DGPs satisfy their identification conditions."""
    did_records, _ = _simulate_did_records()
    did_data = pd.DataFrame(harness._records_payload(did_records))
    did_design = np.column_stack(
        (
            np.ones(len(did_data)),
            did_data["group"],
            did_data["post_treatment"],
            did_data["group"] * did_data["post_treatment"],
        )
    )
    assert np.linalg.matrix_rank(did_design) == 4

    sc_records, _ = _simulate_synthetic_control_records()
    sc_data = pd.DataFrame(harness._records_payload(sc_records)).set_index("t")
    pre_controls = sc_data.loc[
        sc_data.index < SC_TREATMENT_TIME, list(SC_CONTROL_UNITS)
    ].to_numpy()
    contrast_matrix = np.column_stack(
        (
            pre_controls[:, 0] - pre_controls[:, 2],
            pre_controls[:, 1] - pre_controls[:, 2],
        )
    )
    assert np.linalg.matrix_rank(contrast_matrix) == 2
    assert np.linalg.cond(contrast_matrix) <= 3.1
    post_control_sums = sc_data.loc[
        sc_data.index >= SC_TREATMENT_TIME, list(SC_CONTROL_UNITS)
    ].sum(axis=0)
    assert np.ptp(post_control_sums.to_numpy()) > 0


def test_parameter_recovery_gate_rejects_sign_reversal() -> None:
    """A narrow posterior with the opposite sign cannot satisfy point recovery."""
    gate = _parameter_recovery_gate(
        {
            "mean": -2.0,
            "posterior_sd": 0.10,
            "hdi_lower": -2.2,
            "hdi_upper": -1.8,
        },
        truth=2.0,
        max_standardized_error=2.5,
        max_posterior_sd=0.20,
    )

    assert not gate["point_recovery_passed"]
    assert gate["informativeness_passed"]
    assert not gate["passed"]


def test_parameter_recovery_gate_rejects_diffuse_posterior() -> None:
    """A posterior centered on truth still fails when it is uninformatively broad."""
    gate = _parameter_recovery_gate(
        {
            "mean": 2.0,
            "posterior_sd": 0.50,
            "hdi_lower": 1.0,
            "hdi_upper": 3.0,
        },
        truth=2.0,
        max_standardized_error=2.5,
        max_posterior_sd=0.20,
    )

    assert gate["point_recovery_passed"]
    assert not gate["informativeness_passed"]
    assert not gate["passed"]


def test_parameter_recovery_gate_keeps_hdi_coverage_diagnostic() -> None:
    """A single HDI miss remains observable but is never a pass/fail gate."""
    gate = _parameter_recovery_gate(
        {
            "mean": 2.0,
            "posterior_sd": 0.10,
            "hdi_lower": 2.01,
            "hdi_upper": 2.20,
        },
        truth=2.0,
        max_standardized_error=2.5,
        max_posterior_sd=0.20,
    )

    assert gate["point_recovery_passed"]
    assert gate["informativeness_passed"]
    assert gate["passed"]
    assert not gate["hdi_contains_truth_diagnostic"]


@pytest.mark.parametrize(
    ("summary_update", "truth", "max_standardized_error", "max_posterior_sd"),
    [
        ({"posterior_sd": 0.0}, 2.0, 2.5, 0.20),
        ({"posterior_sd": -0.10}, 2.0, 2.5, 0.20),
        ({"posterior_sd": np.nan}, 2.0, 2.5, 0.20),
        ({"mean": np.nan}, 2.0, 2.5, 0.20),
        ({"hdi_upper": np.inf}, 2.0, 2.5, 0.20),
        ({"hdi_lower": 2.20, "hdi_upper": 2.01}, 2.0, 2.5, 0.20),
        ({}, np.inf, 2.5, 0.20),
        ({}, 2.0, np.nan, 0.20),
        ({}, 2.0, -0.01, 0.20),
        ({}, 2.0, 2.5, -0.01),
    ],
)
def test_parameter_recovery_gate_rejects_invalid_inputs(
    summary_update: dict[str, float],
    truth: float,
    max_standardized_error: float,
    max_posterior_sd: float,
) -> None:
    """Invalid uncertainty evidence cannot be coerced into a passing gate."""
    summary = {
        "mean": 2.0,
        "posterior_sd": 0.10,
        "hdi_lower": 1.8,
        "hdi_upper": 2.2,
    }
    summary.update(summary_update)

    with pytest.raises(ValueError):
        _parameter_recovery_gate(
            summary,
            truth=truth,
            max_standardized_error=max_standardized_error,
            max_posterior_sd=max_posterior_sd,
        )


def test_parameter_recovery_gate_accepts_exact_hard_gate_boundaries() -> None:
    """The preregistered limits are inclusive rather than hidden stricter bounds."""
    gate = _parameter_recovery_gate(
        {
            "mean": 2.25,
            "posterior_sd": 0.125,
            "hdi_lower": 1.8,
            "hdi_upper": 2.3,
        },
        truth=2.0,
        max_standardized_error=2.0,
        max_posterior_sd=0.125,
    )

    assert gate["standardized_error"] == pytest.approx(2.0)
    assert gate["point_recovery_passed"]
    assert gate["informativeness_passed"]
    assert gate["passed"]


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.correctness
def test_did_parameter_recovery(
    did_parameter_recovery_data: tuple[pd.DataFrame, dict[str, float]],
    did_parameter_recovery_result: cp.DifferenceInDifferences,
    record_property: Callable[[str, object], None],
) -> None:
    """Recover the named DiD DGP coefficients with informative posterior draws."""
    _, truths = did_parameter_recovery_data
    result = did_parameter_recovery_result
    idata = result.idata
    assert idata is not None
    sampling_quality = harness._sampling_quality(idata, np, harness.REGISTERED_SAMPLING)
    for name, value in sampling_quality.items():
        record_property(f"did_sampling_{name}", str(value))

    causal_impact = harness._canonical_scalar_effect(result.causal_impact)
    causal_impact_series = _capture_focal_series("did.causal_impact", causal_impact)
    causal_impact_metric = harness._series_metric(causal_impact_series, {})
    causal_impact_gate = _parameter_recovery_gate(
        causal_impact_metric["summary"],
        truth=truths["group:post_treatment"],
        max_standardized_error=DID_MAX_STANDARDIZED_ERROR,
        max_posterior_sd=DID_MAX_POSTERIOR_SD,
    )
    _assert_recovery_gate(causal_impact_gate, "DiD causal impact")
    _record_gate_diagnostics(record_property, "did_causal_impact", causal_impact_gate)

    beta = idata.posterior["beta"].sel(treated_units="unit_0")
    beta_series = _capture_focal_series("did.beta", beta)
    assert beta_series["semantics"]["coords"]["coeffs"] == list(DID_TRUE_COEFFICIENTS)
    for coefficient, truth in truths.items():
        metric = harness._series_metric(beta_series, {"coeffs": coefficient})
        gate = _parameter_recovery_gate(
            metric["summary"],
            truth=truth,
            max_standardized_error=DID_MAX_STANDARDIZED_ERROR,
            max_posterior_sd=DID_MAX_POSTERIOR_SD,
        )
        _assert_recovery_gate(gate, f"DiD beta[{coefficient!r}]")
        _record_gate_diagnostics(record_property, f"did_beta_{coefficient}", gate)

    effect_summary = result.effect_summary(alpha=1 - harness.HDI_PROB)
    table_semantics = harness._table_semantics(effect_summary.table)
    assert table_semantics["hdi"]["probability"] == harness.HDI_PROB
    harness._verify_effect_table(
        effect_summary.table,
        [
            {
                "table_row": "treatment_effect",
                "series": "did.causal_impact",
                "selector": {},
            }
        ],
        {"did.causal_impact": causal_impact_series},
        np,
    )

    counterfactual_semantics = _assert_posterior_array_semantics(
        result.y_pred_counterfactual,
        expected_dims=("chain", "draw", "obs_ind", "treated_units"),
        expected_name="mu",
    )
    assert counterfactual_semantics["coords"]["treated_units"] == ["unit_0"]

    fitted_mu = result._model_backend.predict(result.design["X"])
    draw_wise_r2 = harness._draw_wise_r2(result.design["y"], fitted_mu, xr, np)
    r2_series = _capture_focal_series("did.draw_wise_r2", draw_wise_r2)
    r2_metric = r2_series["metrics"][0]
    assert r2_metric["summary"]["mean"] >= DID_MIN_DRAW_WISE_R2
    record_property("did_draw_wise_r2_mean", str(r2_metric["summary"]["mean"]))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.correctness
def test_synthetic_control_parameter_recovery(
    sc_parameter_recovery_data: tuple[pd.DataFrame, dict[str, np.ndarray]],
    sc_parameter_recovery_result: cp.SyntheticControl,
    record_property: Callable[[str, object], None],
) -> None:
    """Recover simplex weights and distinct known post-treatment SC estimands."""
    data, dgp = sc_parameter_recovery_data
    result = sc_parameter_recovery_result
    idata = result.idata
    assert idata is not None
    sampling_quality = harness._sampling_quality(idata, np, harness.REGISTERED_SAMPLING)
    for name, value in sampling_quality.items():
        record_property(f"sc_sampling_{name}", str(value))

    beta = idata.posterior["beta"].sel(treated_units="actual")
    beta_values = np.asarray(beta.values, dtype=float)
    assert np.isfinite(beta_values).all()
    assert (beta_values >= 0).all()
    np.testing.assert_allclose(beta_values.sum(axis=-1), 1.0, atol=1e-12)
    beta_series = _capture_focal_series("sc.beta", beta)
    assert beta_series["semantics"]["coords"]["coeffs"] == list(SC_CONTROL_UNITS)
    for control_unit, truth in SC_TRUE_WEIGHTS.items():
        metric = harness._series_metric(beta_series, {"coeffs": control_unit})
        gate = _parameter_recovery_gate(
            metric["summary"],
            truth=truth,
            max_standardized_error=SC_MAX_STANDARDIZED_ERROR,
            max_posterior_sd=SC_MAX_WEIGHT_POSTERIOR_SD,
        )
        _assert_recovery_gate(gate, f"SC beta[{control_unit!r}]")
        _record_gate_diagnostics(record_property, f"sc_beta_{control_unit}", gate)

    average_impact = result.post_impact.mean(dim="obs_ind")
    final_cumulative_impact = result.post_impact_cumulative.isel(obs_ind=-1)
    first_impact = result.post_impact.isel(obs_ind=0)
    last_impact = result.post_impact.isel(obs_ind=-1)
    average_series = _capture_focal_series("sc.post_average_impact", average_impact)
    cumulative_series = _capture_focal_series(
        "sc.post_final_cumulative_impact", final_cumulative_impact
    )
    first_impact_series = _capture_focal_series("sc.first_post_impact", first_impact)
    last_impact_series = _capture_focal_series("sc.last_post_impact", last_impact)

    recovery_targets = (
        (
            "sc_average_impact",
            harness._series_metric(average_series, {"treated_units": "actual"})[
                "summary"
            ],
            SC_TRUE_AVERAGE_IMPACT,
            SC_MAX_AVERAGE_IMPACT_POSTERIOR_SD,
        ),
        (
            "sc_final_cumulative_impact",
            harness._series_metric(cumulative_series, {"treated_units": "actual"})[
                "summary"
            ],
            SC_TRUE_FINAL_CUMULATIVE_IMPACT,
            SC_MAX_CUMULATIVE_POSTERIOR_SD,
        ),
        (
            "sc_first_post_impact",
            harness._series_metric(first_impact_series, {"treated_units": "actual"})[
                "summary"
            ],
            float(dgp["effect_path"][0]),
            SC_MAX_IMPACT_POSTERIOR_SD,
        ),
        (
            "sc_last_post_impact",
            harness._series_metric(last_impact_series, {"treated_units": "actual"})[
                "summary"
            ],
            float(dgp["effect_path"][-1]),
            SC_MAX_IMPACT_POSTERIOR_SD,
        ),
    )
    for label, summary, truth, max_posterior_sd in recovery_targets:
        gate = _parameter_recovery_gate(
            summary,
            truth=truth,
            max_standardized_error=SC_MAX_STANDARDIZED_ERROR,
            max_posterior_sd=max_posterior_sd,
        )
        _assert_recovery_gate(gate, label)
        _record_gate_diagnostics(record_property, label, gate)

    effect_summary = result.effect_summary(
        alpha=1 - harness.HDI_PROB,
        cumulative=True,
        relative=False,
    )
    table_semantics = harness._table_semantics(effect_summary.table)
    assert table_semantics["hdi"]["probability"] == harness.HDI_PROB
    harness._verify_effect_table(
        effect_summary.table,
        [
            {
                "table_row": "average",
                "series": "sc.post_average_impact",
                "selector": {"treated_units": "actual"},
            },
            {
                "table_row": "cumulative",
                "series": "sc.post_final_cumulative_impact",
                "selector": {"treated_units": "actual"},
            },
        ],
        {
            "sc.post_average_impact": average_series,
            "sc.post_final_cumulative_impact": cumulative_series,
        },
        np,
    )

    post_prediction_semantics = _assert_posterior_array_semantics(
        result.post_pred,
        expected_dims=("chain", "draw", "obs_ind", "treated_units"),
        expected_name="mu",
    )
    expected_post_coordinates = data.index[data.index >= SC_TREATMENT_TIME].tolist()
    assert post_prediction_semantics["coords"]["obs_ind"] == expected_post_coordinates
    assert post_prediction_semantics["coords"]["treated_units"] == ["actual"]
    post_counterfactual_mean = result.post_pred.mean(dim=["chain", "draw"]).sel(
        treated_units="actual"
    )
    counterfactual_error = (
        np.asarray(post_counterfactual_mean.values, dtype=float)
        - dgp["counterfactual"][SC_TREATMENT_TIME:]
    )
    counterfactual_rmse = float(np.sqrt(np.mean(counterfactual_error**2)))
    counterfactual_max_error = float(np.abs(counterfactual_error).max())
    assert counterfactual_rmse <= SC_MAX_COUNTERFACTUAL_RMSE
    assert counterfactual_max_error <= SC_MAX_COUNTERFACTUAL_ABSOLUTE_ERROR
    record_property("sc_counterfactual_rmse", str(counterfactual_rmse))
    record_property("sc_counterfactual_max_error", str(counterfactual_max_error))

    draw_wise_r2 = harness._draw_wise_r2(
        result.pre_design["treated"], result.pre_pred, xr, np
    )
    r2_series = _capture_focal_series("sc.draw_wise_r2", draw_wise_r2)
    r2_metric = r2_series["metrics"][0]
    assert r2_metric["summary"]["mean"] >= SC_MIN_DRAW_WISE_R2
    record_property("sc_draw_wise_r2_mean", str(r2_metric["summary"]["mean"]))
