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
"""Doubly robust outcome regressions follow the Patsy intercept convention."""

import numpy as np
from sklearn.linear_model import LinearRegression

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


def _reference_ate(X, y, t, ps, *, fit_intercept: bool) -> float:
    """Doubly robust ATE from an explicit sklearn intercept setting."""
    design = np.column_stack([np.asarray(X, dtype=float), np.asarray(ps, dtype=float)])
    treated = np.asarray(t).astype(bool)
    control_model = LinearRegression(fit_intercept=fit_intercept).fit(
        design[~treated], y[~treated]
    )
    treated_model = LinearRegression(fit_intercept=fit_intercept).fit(
        design[treated], y[treated]
    )
    control_pred = control_model.predict(design)
    treated_pred = treated_model.predict(design)
    weighted_control = (1 - t) * (y - control_pred) / (1 - ps) + control_pred
    weighted_treated = t * (y - treated_pred) / ps + treated_pred
    return float(np.mean(weighted_treated) - np.mean(weighted_control))


def test_doubly_robust_ate_uses_fit_intercept_false():
    """The outcome regression must not add a second intercept.

    With a Patsy intercept column the two sklearn settings span the same
    column space. Without that column they disagree, and the estimator must
    follow ``fit_intercept=False``.
    """
    rng = np.random.default_rng(2)
    n = 80
    x1 = rng.normal(size=n)
    ps = 1 / (1 + np.exp(-(0.2 + 0.5 * x1)))
    t = rng.binomial(1, ps).astype(float)
    y = 1.5 + 2.0 * x1 + 3.0 * t + rng.normal(scale=0.1, size=n)
    designs = {
        "patsy_intercept": (np.column_stack([np.ones(n), x1]), ["Intercept", "x1"]),
        "no_intercept_column": (x1.reshape(-1, 1), ["x1"]),
    }
    for name, (X, labels) in designs.items():
        ate, _, _ = _ipw_without_fit(X, y, t, labels)._compute_ate_doubly_robust(ps)
        without_sklearn_intercept = _reference_ate(X, y, t, ps, fit_intercept=False)
        with_sklearn_intercept = _reference_ate(X, y, t, ps, fit_intercept=True)
        np.testing.assert_allclose(
            ate,
            without_sklearn_intercept,
            atol=1e-10,
            err_msg=name,
        )
        if name == "no_intercept_column":
            assert abs(float(ate) - with_sklearn_intercept) > 1e-3
