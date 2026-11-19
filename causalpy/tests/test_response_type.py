# Copyright 2022 - 2026 The PyMC Labs Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Expectation versus posterior-predictive response selection."""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.experiments._response_type import (
    annotation_text,
    check_response_type,
    finish_response_plot,
)
from causalpy.experiments._results import CausalResult
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.experiments.model_adapter import (
    PyMCModelAdapter,
    SklearnModelAdapter,
    _extract_response,
)


def _draws(values, name):
    return xr.DataArray(
        np.asarray(values, dtype=float)[None, ..., None],
        dims=("chain", "draw", "obs_ind", "treated_units"),
        coords={
            "chain": [0],
            "draw": [0, 1],
            "obs_ind": [0, 1],
            "treated_units": ["unit_0"],
        },
        name=name,
    )


def test_response_type_rejects_unknown_values_and_names_both_quantities():
    """A typo must fail, and the annotation must not hardcode HDI."""
    with pytest.raises(ValueError, match="response_type"):
        check_response_type("hdi")

    assert "conditional expectation" in annotation_text(
        response_type="expectation",
        ci_prob=0.5,
        ci_kind="eti",
        kind="ribbon",
        has_draws=True,
    )
    assert "50% ETI" in annotation_text(
        response_type="expectation",
        ci_prob=0.5,
        ci_kind="eti",
        kind="ribbon",
        has_draws=True,
    )
    prediction = annotation_text(
        response_type="prediction",
        ci_prob=0.94,
        ci_kind="hdi",
        kind="spaghetti",
        has_draws=True,
    )
    assert "observation noise" in prediction
    assert "94%" not in prediction
    assert "HDI" not in prediction


def test_annotation_lands_on_the_figure_and_can_be_suppressed():
    """The note is a figure text, not a title, and False removes nothing else."""
    fig, ax = plt.subplots()
    ax.set_title("keep me")
    finish_response_plot(
        fig,
        None,
        response_type="prediction",
        ci_prob=0.94,
        ci_kind="hdi",
        kind="ribbon",
        show_response_annotation=True,
        has_draws=True,
    )
    assert fig.texts
    assert "94% HDI" in fig.texts[0].get_text()
    assert ax.get_title() == "keep me"
    plt.close(fig)

    fig, _ax = plt.subplots()
    finish_response_plot(
        fig,
        None,
        response_type="prediction",
        ci_prob=0.94,
        ci_kind="hdi",
        kind="ribbon",
        show_response_annotation=False,
        has_draws=True,
    )
    assert fig.texts == []
    plt.close(fig)


def test_pymc_predict_response_selects_mu_or_y_hat_without_changing_predict():
    """predict stays on mu; the new method is the opt-in."""
    mu = _draws([[4.0, 4.0], [4.0, 4.0]], "mu")
    y_hat = _draws([[1.0, 1.0], [9.0, 9.0]], "y_hat")
    tree = xr.DataTree()
    tree["posterior_predictive"] = xr.Dataset({"mu": mu, "y_hat": y_hat})

    class _Model:
        supports_observation_noise = True

        def predict(self, X, *, coords=None, out_of_sample=False, group="posterior"):
            return tree

    adapter = PyMCModelAdapter.__new__(PyMCModelAdapter)
    adapter._model = _Model()
    selected = adapter.predict_response(
        np.ones((2, 1)), quantity="y_hat", group="posterior"
    )
    xr.testing.assert_equal(selected, y_hat)
    xr.testing.assert_equal(adapter.predict(np.ones((2, 1))), mu)
    with pytest.raises(ValueError, match="y_hat"):
        adapter.predict_response(np.ones((2, 1)), quantity="missing", group="posterior")


def test_declared_noise_free_backend_rejects_prediction():
    """A backend that stores mu under both names must not claim observation noise."""
    mu = _draws([[4.0, 4.0], [4.0, 4.0]], "mu")
    tree = xr.DataTree()
    tree["posterior_predictive"] = xr.Dataset({"mu": mu, "y_hat": mu.copy()})

    class _Model:
        supports_observation_noise = False

        def predict(self, X, *, coords=None, out_of_sample=False, group="posterior"):
            return tree

    adapter = PyMCModelAdapter.__new__(PyMCModelAdapter)
    adapter._model = _Model()
    with pytest.raises(ValueError, match="no observation-noise draws"):
        adapter.predict_response(np.ones((2, 1)), quantity="y_hat", group="posterior")
    xr.testing.assert_equal(
        adapter.predict_response(np.ones((2, 1)), quantity="mu", group="posterior"),
        mu,
    )


def test_declared_noise_free_backend_rejects_nan_predictions():
    """NaN positions still mean the two names are one series."""
    mu = _draws([[4.0, np.nan], [4.0, 4.0]], "mu")
    tree = xr.DataTree()
    tree["posterior_predictive"] = xr.Dataset({"mu": mu, "y_hat": mu.copy()})

    class _Model:
        supports_observation_noise = False

        def predict(self, X, *, coords=None, out_of_sample=False, group="posterior"):
            return tree

    adapter = PyMCModelAdapter.__new__(PyMCModelAdapter)
    adapter._model = _Model()
    with pytest.raises(ValueError, match="no observation-noise draws"):
        adapter.predict_response(np.ones((2, 1)), quantity="y_hat", group="posterior")


def test_its_prediction_bundle_uses_y_hat_and_leaves_the_stored_result():
    """Prediction mode rebuilds impact; expectation mode reuses the stored bundle."""
    mu = _draws([[4.0, 4.0], [4.0, 4.0]], "mu")
    y_hat = _draws([[1.0, 1.0], [1.0, 1.0]], "y_hat")
    observed = xr.DataArray(
        [[10.0], [12.0]],
        dims=("obs_ind", "treated_units"),
        coords={"obs_ind": [0, 1], "treated_units": ["unit_0"]},
    )
    stored = CausalResult(
        predictions_pre=mu,
        predictions_post=mu,
        impact_pre=mu,
        impact_post=mu,
        impact_post_cumulative=mu,
        score=None,
    )
    calls = []

    class _Backend:
        def predict_response(self, X, *, quantity, group, out_of_sample=False):
            calls.append((quantity, group, out_of_sample))
            return y_hat

    experiment = InterruptedTimeSeries.__new__(InterruptedTimeSeries)
    experiment.pre_design = {"X": "pre", "y": observed}
    experiment.post_design = {"X": "post", "y": observed}
    experiment._model_backend = _Backend()

    unchanged = experiment._bundle_for_response(
        stored, response_type="expectation", group="posterior"
    )
    assert unchanged is stored
    copied = experiment._bundle_for_response(
        stored, response_type="prediction", group="posterior"
    )
    assert copied is not stored
    xr.testing.assert_equal(stored.predictions_post, mu)
    xr.testing.assert_allclose(
        copied.impact_post, (observed - y_hat).squeeze("treated_units")
    )
    assert calls == [
        ("y_hat", "posterior", False),
        ("y_hat", "posterior", True),
    ]


def _sklearn_its():
    dates = pd.date_range("2020-01-01", periods=16, freq="D")
    data = pd.DataFrame(
        {
            "t": np.arange(len(dates), dtype=float),
            "y": np.arange(len(dates), dtype=float),
        },
        index=dates,
    )
    return cp.InterruptedTimeSeries(
        data,
        treatment_time=dates[8],
        formula="y ~ 1 + t",
        model=LinearRegression(),
    ).fit()


def test_sklearn_its_rejects_prediction_mode_and_keeps_default_plots_unlabeled():
    """OLS has no observation-noise draws, so prediction mode is an error."""
    experiment = _sklearn_its()
    stored = experiment.result.predictions_post.copy()
    fig, _ax = experiment.plot(show=False)
    assert fig.texts == []
    plt.close(fig)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        labeled, _ax = experiment.plot(show=False, show_response_annotation=True)
    assert labeled.texts
    assert "Point prediction" in labeled.texts[0].get_text()
    assert "HDI" not in labeled.texts[0].get_text()
    plt.close(labeled)

    with pytest.raises(ValueError, match="no observation-noise draws"):
        experiment.plot(show=False, response_type="prediction")
    xr.testing.assert_equal(experiment.result.predictions_post, stored)
    assert "observation noise" not in experiment.effect_summary().text


def test_sklearn_did_prediction_mode_does_not_silently_reuse_mu_bands():
    """Coefficient plots must reject the opt-in rather than ignore it."""
    rows = []
    for group in (0, 1):
        for post in (0, 1):
            for i in range(4):
                rows.append(
                    {
                        "group": group,
                        "unit": f"u{group}-{i}",
                        "t": post,
                        "post_treatment": post,
                        "y": 1.0 + group + post + group * post + 0.01 * i,
                    }
                )
    experiment = cp.DifferenceInDifferences(
        pd.DataFrame(rows),
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    ).fit()
    stored = experiment.result.causal_impact
    with pytest.raises(ValueError, match="no observation-noise draws"):
        experiment.plot(response_type="prediction")
    assert experiment.result.causal_impact == stored


def test_point_estimate_adapter_response_contract():
    """Expected draws stay canonical; unsupported targets fail explicitly."""
    X = xr.DataArray([[1.0, 0.0], [1.0, 1.0]], dims=("obs_ind", "coeffs"))
    y = xr.DataArray([[2.0], [5.0]], dims=("obs_ind", "treated_units"))
    adapter = SklearnModelAdapter(LinearRegression(fit_intercept=False))
    adapter.fit(X, y)
    expected = adapter.predict_response(X)
    xr.testing.assert_equal(expected, adapter.predict(X))
    assert expected.dims == ("chain", "draw", "obs_ind", "treated_units")
    assert not adapter.supports_observation_noise
    with pytest.raises(ValueError, match="no observation-noise draws"):
        adapter.predict_response(X, quantity="y_hat")
    with pytest.raises(ValueError, match="quantity must be"):
        adapter.predict_response(X, quantity="misspelled")


@pytest.mark.parametrize("quantity", ["mu", "y_hat"])
def test_response_extractor_rejects_missing_target_instead_of_falling_back(quantity):
    """A malformed backend container cannot silently substitute the other target."""
    other = "y_hat" if quantity == "mu" else "mu"
    tree = xr.DataTree.from_dict(
        {"posterior_predictive": xr.Dataset({other: _draws([[1, 2], [3, 4]], other)})}
    )
    with pytest.raises(ValueError, match=f"no '{quantity}' draws"):
        _extract_response(tree, quantity)
    with pytest.raises(ValueError, match="quantity must be"):
        _extract_response(tree, "misspelled")
