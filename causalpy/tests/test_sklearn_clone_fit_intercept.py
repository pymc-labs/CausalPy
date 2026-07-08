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
"""Regression tests for #664: BaseExperiment must not mutate user-supplied sklearn models."""

import warnings

from sklearn.linear_model import LinearRegression

import causalpy as cp


def test_fit_intercept_clone_and_warn():
    """Passing fit_intercept=True must clone the model, override it, and warn (#664)."""
    model = LinearRegression(fit_intercept=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = cp.DifferenceInDifferences(
            cp.load_data("did"),
            formula="y ~ 1 + group*post_treatment",
            time_variable_name="t",
            group_variable_name="group",
            model=model,
        )

    # Original instance must be unchanged
    assert model.fit_intercept is True
    # Internal clone must have fit_intercept=False
    assert result.model.fit_intercept is False
    # A warning must have been emitted
    assert any("fit_intercept" in str(w.message) for w in caught)
