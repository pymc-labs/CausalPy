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
"""Contracts for the joint augmented synthetic control posterior."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import scipy.stats
from pytensor.graph.traversal import ancestors

from causalpy.experiments.augmented_synthetic_control import (
    AugmentedSyntheticControl,
)
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.synthetic_control import SyntheticControl

DONORS = ["a", "b", "c"]
TREATED = ["u", "v"]
TREATMENT_TIME = 3
WEIGHTS = np.array([[0.2, 0.5, 0.3], [0.0, 1.0, 0.0]])
COEFFICIENTS = np.array(
    [
        [0.1, 0.0, -0.2],
        [0.3, 0.1, 0.0],
        [-0.1, 0.2, 0.4],
        [1.0, -1.0, 0.5],
        [0.0, 0.25, -0.75],
    ]
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [0.5, 1.5, 2.5, 3.5, 4.5],
            "c": [2.0, 0.0, 1.0, 0.0, 2.0],
            "u": [1.2, 1.8, 2.4, 9.0, 8.0],
            "v": [0.7, 1.1, 1.9, 3.0, 7.0],
            "ignored": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=[0, 1, 2, 3, 4],
    )


def _experiment(**kwargs) -> AugmentedSyntheticControl:
    return AugmentedSyntheticControl(
        _frame(),
        TREATMENT_TIME,
        DONORS,
        TREATED,
        **kwargs,
    )


def _recorded_counterfactual(
    donor_post: np.ndarray,
    weights: np.ndarray,
    treated_pre: np.ndarray,
    donor_pre: np.ndarray,
    beta_post: np.ndarray,
) -> np.ndarray:
    """Independent copy of the recorded counterfactual algebra."""
    imbalance = treated_pre.T - weights @ donor_pre.T
    weighted = donor_post @ weights.T
    correction = imbalance @ beta_post.T
    return weighted + correction.T


def _eval(model: pm.Model, name: str, weights: np.ndarray, beta: np.ndarray):
    fn = model.compile_fn(
        model[name],
        inputs=[model["base_weights"], model["prognostic_coefficients"]],
        point_fn=False,
    )
    return np.asarray(fn(weights, beta), dtype=float)


def test_counterfactual_matches_recorded_algebra():
    """The deterministic matches donor post weights plus the imbalance correction."""
    experiment = _experiment()
    panel = experiment._panel
    donor_pre = panel.pre[DONORS].to_numpy(dtype=float)
    donor_post = panel.post[DONORS].to_numpy(dtype=float)
    treated_pre = panel.pre[TREATED].to_numpy(dtype=float)
    treated_post = panel.post[TREATED].to_numpy(dtype=float)
    expected = _recorded_counterfactual(
        donor_post,
        WEIGHTS,
        treated_pre,
        donor_pre,
        COEFFICIENTS[len(panel.pre) :],
    )

    counterfactual = _eval(experiment.model, "counterfactual", WEIGHTS, COEFFICIENTS)
    effect = _eval(experiment.model, "treatment_effect", WEIGHTS, COEFFICIENTS)
    np.testing.assert_allclose(counterfactual, expected)
    np.testing.assert_allclose(effect, treated_post - expected)


def test_counterfactual_uses_only_post_period_coefficients():
    """Pre-period coefficient rows do not enter the post-period counterfactual."""
    model = _experiment().model
    baseline = _eval(model, "counterfactual", WEIGHTS, COEFFICIENTS)
    shifted_pre = COEFFICIENTS.copy()
    shifted_pre[:TREATMENT_TIME] += 10.0
    shifted_post = COEFFICIENTS.copy()
    shifted_post[TREATMENT_TIME:] += 0.5
    np.testing.assert_allclose(
        _eval(model, "counterfactual", WEIGHTS, shifted_pre),
        baseline,
    )
    assert not np.allclose(
        _eval(model, "counterfactual", WEIGHTS, shifted_post),
        baseline,
    )


def test_treated_post_does_not_enter_counterfactual_or_likelihood():
    """Treated post outcomes change the effect only."""
    experiment = _experiment()
    model = experiment.model
    point = model.initial_point()
    logp = model.compile_logp()
    before = logp(point)
    counterfactual = _eval(model, "counterfactual", WEIGHTS, COEFFICIENTS)
    effect = _eval(model, "treatment_effect", WEIGHTS, COEFFICIENTS)

    shifted = experiment._panel.post[TREATED].to_numpy(dtype=float) + 5.0
    with model:
        pm.set_data({"treated_post_data": shifted})
    assert logp(point) == before
    np.testing.assert_allclose(
        _eval(model, "counterfactual", WEIGHTS, COEFFICIENTS),
        counterfactual,
    )
    np.testing.assert_allclose(
        _eval(model, "treatment_effect", WEIGHTS, COEFFICIENTS),
        effect + 5.0,
    )
    assert model["treated_post_data"] not in ancestors([model["counterfactual"]])
    assert model["treated_post_data"] in ancestors([model["treatment_effect"]])


def test_likelihoods_use_the_recorded_outcomes_and_scale():
    """Weights see treated pre outcomes. The prognostic regression sees donors."""
    experiment = _experiment()
    model = experiment.model
    panel = experiment._panel
    n_pre = len(panel.pre)
    donor_pre = panel.pre[DONORS].to_numpy(dtype=float)
    donor_all = np.vstack((donor_pre, panel.post[DONORS].to_numpy(dtype=float)))
    treated_pre = panel.pre[TREATED].to_numpy(dtype=float)

    weight_logp = model.compile_fn(
        pm.logp(model["treated_pre"], treated_pre).sum(),
        inputs=[model["base_weights"]],
        point_fn=False,
    )
    prognostic_logp = model.compile_fn(
        pm.logp(model["donor_outcomes"], donor_all).sum(),
        inputs=[model["prognostic_coefficients"]],
        point_fn=False,
    )
    expected_weights = scipy.stats.norm.logpdf(
        treated_pre,
        loc=donor_pre @ WEIGHTS.T,
        scale=1.0,
    ).sum()
    expected_prognostic = scipy.stats.norm.logpdf(
        donor_all,
        loc=COEFFICIENTS @ donor_pre,
        scale=1.0,
    ).sum()
    np.testing.assert_allclose(weight_logp(WEIGHTS), expected_weights)
    np.testing.assert_allclose(prognostic_logp(COEFFICIENTS), expected_prognostic)

    observed = {rv.name for rv in model.observed_RVs}
    assert observed == {"treated_pre", "donor_outcomes"}
    assert model["donor_outcomes"].eval().shape == (
        n_pre + len(panel.post),
        len(DONORS),
    )
    assert model["treated_pre"].eval().shape == (n_pre, len(TREATED))
    assert len(model.coords["pre_time"]) == n_pre
    assert "treated_units" not in model.named_vars_to_dims["prognostic_coefficients"]

    point = model.initial_point()
    joint = model.compile_logp()
    baseline = joint(point)
    with model:
        pm.set_data({"treated_pre_data": treated_pre + 4.0})
    assert joint(point) != baseline


def test_shrinkage_prior_is_normal_zero_tau_halfnormal():
    """tau is a HalfNormal, and coefficients are Normal(0, tau), not a penalty."""
    model = _experiment().model
    tau = np.array(1.7)
    beta = COEFFICIENTS
    fn = model.compile_fn(
        [
            pm.logp(
                model["prognostic_coefficients"], model["prognostic_coefficients"]
            ).sum(),
            pm.logp(model["tau"], model["tau"]).sum(),
        ],
        inputs=[model["tau"], model["prognostic_coefficients"]],
        point_fn=False,
    )
    got_beta, got_tau = fn(tau, beta)
    np.testing.assert_allclose(
        got_beta,
        scipy.stats.norm.logpdf(beta, loc=0.0, scale=float(tau)).sum(),
    )
    np.testing.assert_allclose(
        got_tau, scipy.stats.halfnorm.logpdf(float(tau), scale=1.0)
    )
    assert type(model["tau"].owner.op).__name__ == "HalfNormalRV"
    assert type(model["prognostic_coefficients"].owner.op).__name__ == "NormalRV"
    assert model["tau"] in ancestors([model["prognostic_coefficients"]])
    assert list(model.potentials) == []


def test_base_weights_are_independent_unit_simplexes():
    """Each treated unit has its own Dirichlet(1, ..., 1) simplex."""
    model = _experiment().model
    fn = model.compile_fn(
        pm.logp(model["base_weights"], model["base_weights"]).sum(),
        inputs=[model["base_weights"]],
        point_fn=False,
    )
    expected = sum(
        float(scipy.stats.dirichlet.logpdf(row, np.ones(len(DONORS))))
        for row in WEIGHTS
    )
    np.testing.assert_allclose(fn(WEIGHTS), expected)
    assert model.named_vars_to_dims["base_weights"] == (
        "treated_units",
        "control_units",
    )
    assert model.named_vars_to_dims["prognostic_coefficients"] == ("time", "pre_time")


def test_one_model_holds_the_wide_panel_split():
    """The estimator is its own class and keeps the shared treatment time post-period."""
    experiment = _experiment()
    assert not issubclass(AugmentedSyntheticControl, SyntheticControl)
    assert not issubclass(AugmentedSyntheticControl, BaseExperiment)
    assert experiment._panel.treatment_time == TREATMENT_TIME
    assert list(experiment._panel.pre.index) == [0, 1, 2]
    assert list(experiment._panel.post.index) == [3, 4]
    assert {rv.name for rv in experiment.model.free_RVs} == {
        "tau",
        "prognostic_coefficients",
        "base_weights",
    }
    assert {rv.name for rv in experiment.model.deterministics} >= {
        "counterfactual",
        "treatment_effect",
    }
    source = inspect.getsource(inspect.getmodule(AugmentedSyntheticControl))
    assert "WeightedSumFitter" not in source


def test_fit_samples_one_joint_posterior(real_pymc_sampling):
    """One sample call returns weights, coefficients, and the counterfactual together."""
    experiment = _experiment()
    experiment.fit(
        sample_kwargs={
            "draws": 4,
            "tune": 4,
            "chains": 1,
            "cores": 1,
            "progressbar": False,
            "compute_convergence_checks": False,
            "random_seed": 1,
        }
    )
    assert experiment.idata is not None
    posterior = experiment.idata.posterior
    for name in (
        "tau",
        "base_weights",
        "prognostic_coefficients",
        "counterfactual",
        "treatment_effect",
    ):
        assert name in posterior
    assert posterior.sizes["chain"] == 1
    assert posterior.sizes["draw"] == 4
    assert "treated_units" in posterior["base_weights"].dims
    assert "treated_units" not in posterior["prognostic_coefficients"].dims
    assert "chain" in posterior["counterfactual"].dims


def test_datetime_panel_uses_one_treatment_time():
    """A timestamp boundary is post-period for every treated unit."""
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    frame = pd.DataFrame(
        {"donor": [1.0, 2.0, 3.0, 4.0], "treated": [1.5, 2.5, 8.0, 9.0]},
        index=index,
    )
    experiment = AugmentedSyntheticControl(
        frame,
        pd.Timestamp("2020-01-03"),
        ["donor"],
        ["treated"],
    )
    assert list(experiment._panel.pre.index) == list(index[:2])
    assert list(experiment._panel.post.index) == list(index[2:])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"control_units": []}, "control_units must be nonempty"),
        ({"treated_units": ["u", "u"]}, "treated_units must be unique"),
        ({"control_units": ["a"], "treated_units": ["a"]}, "overlap"),
        ({"control_units": ["missing"]}, "missing columns"),
        ({"treatment_time": 0}, "pre-period is empty"),
        ({"treatment_time": 5}, "post-period is empty"),
    ],
)
def test_rejects_panels_that_cannot_support_the_likelihoods(kwargs, match):
    """Empty periods and bad unit lists fail before a model is built."""
    arguments = {
        "data": _frame(),
        "treatment_time": TREATMENT_TIME,
        "control_units": DONORS,
        "treated_units": TREATED,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        AugmentedSyntheticControl(**arguments)


def test_rejects_non_finite_outcomes():
    """Missing outcomes are not silently dropped from either likelihood."""
    frame = _frame()
    frame.loc[1, "u"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        AugmentedSyntheticControl(frame, TREATMENT_TIME, DONORS, TREATED)


def test_rejects_a_non_numeric_outcome():
    """A non-numeric outcome cannot enter the donor matrix or the likelihood."""
    frame = _frame()
    frame["a"] = list("abcde")
    with pytest.raises(ValueError, match="numeric"):
        AugmentedSyntheticControl(frame, TREATMENT_TIME, DONORS, TREATED)
