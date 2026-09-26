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
"""Numerical identity for overlap inverse propensity weighting."""

import numpy as np

import causalpy as cp


def _ipw_without_fit(X, y, t, labels):
    """Build an IPW instance whose weighting methods can run without MCMC."""
    ipw = cp.InversePropensityWeighting.__new__(cp.InversePropensityWeighting)
    ipw.X = np.asarray(X, dtype=float)
    ipw.labels = list(labels)
    ipw.y = np.asarray(y, dtype=float)
    ipw.t = np.asarray(t, dtype=float).reshape(-1, 1)
    ipw.outcome_variable = "y"
    return ipw


def test_overlap_ate_matches_hajek_not_full_sample_ate():
    """Overlap weights target the overlap population, not the full-sample ATE.

    Stratum 0 has propensity 0.5 and treatment effect 10. Stratum 1 has
    propensity 0.95 and treatment effect 0. The Hajek overlap contrast
    upweights the first stratum. The mean of the unit-level effects does not.
    """
    n_per_stratum = 200
    stratum = np.array([0] * n_per_stratum + [1] * n_per_stratum)
    propensity = np.where(stratum == 0, 0.5, 0.95)
    effect = np.where(stratum == 0, 10.0, 0.0)
    y0 = np.where(stratum == 0, 0.0, 5.0)
    y1 = y0 + effect
    treatment = np.random.default_rng(1).binomial(1, propensity)
    y = np.where(treatment == 1, y1, y0)
    X = np.column_stack([np.ones(len(treatment)), stratum.astype(float)])
    ipw = _ipw_without_fit(X, y, treatment, ["Intercept", "stratum"])

    ate, treated_mean, control_mean = ipw._compute_ate_overlap(propensity)
    weight = np.where(treatment == 1, 1.0 - propensity, propensity)
    treated = treatment == 1
    control = ~treated
    expected_treated = np.sum(weight[treated] * y[treated]) / np.sum(weight[treated])
    expected_control = np.sum(weight[control] * y[control]) / np.sum(weight[control])
    full_sample_ate = float(np.mean(y1 - y0))

    np.testing.assert_allclose(treated_mean, expected_treated, atol=1e-12)
    np.testing.assert_allclose(control_mean, expected_control, atol=1e-12)
    np.testing.assert_allclose(ate, expected_treated - expected_control, atol=1e-12)
    assert abs(float(ate) - full_sample_ate) > 1.0
