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
"""Tests for the canonical model score container."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from causalpy.experiments.model_adapter import SklearnModelAdapter
from causalpy.plot_utils import extract_r2_score, format_r2_score
from causalpy.skl_models import create_causalpy_compatible_class


def test_sklearn_single_output_score_contract():
    X = np.column_stack((np.ones(8), np.arange(8)))
    y = 1 + 2 * X[:, 1]
    adapter = SklearnModelAdapter(
        create_causalpy_compatible_class(LinearRegression(fit_intercept=False))
    )
    adapter.fit(X, y)

    score = adapter.score(X, y)

    pd.testing.assert_index_equal(score.index, pd.Index(["unit_0_r2"]))
    assert score["unit_0_r2"] == pytest.approx(adapter.model.score(X, y))


def test_sklearn_multi_output_scores_each_unit():
    X = np.column_stack((np.ones(8), np.arange(8)))
    y = np.column_stack((1 + 2 * X[:, 1], X[:, 1] ** 2))
    adapter = SklearnModelAdapter(
        create_causalpy_compatible_class(LinearRegression(fit_intercept=False))
    )
    adapter.fit(X, y)

    score = adapter.score(X, y)
    expected = r2_score(y, adapter.model.predict(X), multioutput="raw_values")

    pd.testing.assert_index_equal(score.index, pd.Index(["unit_0_r2", "unit_1_r2"]))
    np.testing.assert_allclose(score.to_numpy(), expected)
    assert score["unit_0_r2"] != score["unit_1_r2"]


def test_extract_r2_score_requires_unit_key():
    with pytest.raises(ValueError, match="unit_1_r2"):
        extract_r2_score(pd.Series({"unit_0_r2": 0.9}), unit_index=1)


def test_sklearn_score_rejects_multioutput_kwarg():
    X = np.column_stack((np.ones(8), np.arange(8)))
    y = 1 + 2 * X[:, 1]
    adapter = SklearnModelAdapter(
        create_causalpy_compatible_class(LinearRegression(fit_intercept=False))
    )
    adapter.fit(X, y)

    with pytest.raises(ValueError, match="multioutput"):
        adapter.score(X, y, multioutput="uniform_average")


def test_sklearn_score_forwards_r2_score_kwargs():
    X = np.column_stack((np.ones(8), np.arange(8)))
    y = 1 + 2 * X[:, 1]
    sample_weight = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    adapter = SklearnModelAdapter(
        create_causalpy_compatible_class(LinearRegression(fit_intercept=False))
    )
    adapter.fit(X, y)

    score = adapter.score(X, y, sample_weight=sample_weight)
    expected = r2_score(
        y,
        adapter.model.predict(X),
        sample_weight=sample_weight,
    )

    assert score["unit_0_r2"] == pytest.approx(expected)


def test_score_formatting_keys_on_dispersion_presence():
    point_score = pd.Series({"unit_0_r2": 0.91})
    posterior_score = pd.Series({"unit_0_r2": 0.91, "unit_0_r2_std": 0.03})

    assert format_r2_score(point_score, context="on fit data") == (
        "$R^2$ on fit data = 0.91"
    )
    assert format_r2_score(posterior_score, context="on fit data") == (
        "Bayesian $R^2$ on fit data = 0.91 (std = 0.03)"
    )
