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
"""Response-scale expected_effect is the default for the five locked designs."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.checks.prior_sensitivity import PriorSensitivity
from causalpy.experiments._expected_effect import (
    align_expected_window,
    cumulative_sum,
    cumulative_total,
    headline_mean,
    response_contrast,
    select_headline_effect,
)
from causalpy.experiments._results import CausalResult
from causalpy.pipeline import PipelineContext
from causalpy.tests.test_prepostnegd_planted import _ClosedFormCoefficientModel

LIGHT_SAMPLE = {
    "tune": 20,
    "draws": 20,
    "chains": 2,
    "cores": 2,
    "progressbar": False,
    "random_seed": 42,
}


def test_logit_contrast_helper_uses_known_response_means():
    """Known probabilities, not log-odds. No GLM is fit."""
    treated = xr.DataArray([1 / 3, 2 / 3], dims=("obs_ind",))
    untreated = xr.DataArray([0.2, 0.5], dims=("obs_ind",))
    did = response_contrast(treated, untreated, reduce_dims=("obs_ind",))
    assert float(np.asarray(did)) == pytest.approx(0.15)
    assert float(np.asarray(did)) != pytest.approx(np.log(2))

    mu_treated = xr.DataArray([1 / 3, 2 / 3, 1 / 3, 1 / 3], dims=("obs_ind",))
    mu_untreated = xr.DataArray([0.2, 0.5, 0.2, 0.2], dims=("obs_ind",))
    pooled = response_contrast(mu_treated, mu_untreated, reduce_dims=("obs_ind",))
    assert float(np.asarray(pooled)) == pytest.approx(17 / 120)


def test_identity_link_ols_did_equals_interaction(capsys):
    """Every treated post row, interaction off, equals the fitted coefficient."""
    rows = []
    unit = 0
    for group in (0, 1):
        for post in (0, 1):
            for step in range(3):
                rows.append(
                    {
                        "unit": f"u{unit}",
                        "t": step + 10 * post,
                        "group": group,
                        "post_treatment": post,
                        "y": 1.0 + 2.0 * group + 3.0 * post + 4.0 * group * post,
                    }
                )
                unit += 1
    frame = pd.DataFrame(rows)
    result = cp.DifferenceInDifferences(
        frame,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(fit_intercept=False),
    ).fit()
    coefficient = float(
        result.model.coef_[
            next(
                i
                for i, label in enumerate(result.labels)
                if result._is_treatment_interaction(label)
            )
        ]
    )
    impact = result.result.causal_impact
    assert isinstance(impact, xr.DataArray)
    assert impact.sizes.get("chain") == 1
    assert impact.sizes.get("draw") == 1
    assert float(np.asarray(impact).reshape(())) == pytest.approx(coefficient)
    assert float(np.asarray(impact).reshape(())) == pytest.approx(4.0)
    assert float(
        np.asarray(result.result.expected_effect).reshape(())
    ) == pytest.approx(float(np.asarray(impact).reshape(())))
    result.summary()
    printed = capsys.readouterr().out
    assert "CI_" not in printed
    assert f"Expected effect = {float(np.asarray(impact).reshape(())):.2f}" in printed

    bumped = frame.copy()
    bumped.loc[bumped.index[-1], "y"] += 1
    result.data = bumped
    assert float(
        np.asarray(result._expected_effect("posterior")).reshape(())
    ) == pytest.approx(float(np.asarray(impact).reshape(())))


def test_identity_link_ols_prepost_equals_group_coefficient(capsys):
    """Each enrolled unit twice, pretest fixed, equals the group coefficient."""
    frame = pd.DataFrame(
        {
            "group": [0, 0, 1, 1],
            "pre": [1.0, 2.0, 1.5, 2.5],
            "post": [1.0, 2.0, 1.5 + 4.0, 2.5 + 4.0],
        }
    )
    result = cp.PrePostNEGD(
        frame,
        formula="post ~ 1 + C(group) + pre",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=_ClosedFormCoefficientModel(),
    ).fit()
    coefficient = float(
        result.idata["posterior"]["beta"]
        .sel(coeffs=result.labels[1])
        .values.reshape(-1)[0]
    )
    impact = float(np.asarray(result.result.causal_impact).reshape(-1)[0])
    assert impact == pytest.approx(coefficient)
    assert impact == pytest.approx(4.0)
    assert float(
        np.asarray(result.result.expected_effect).reshape(-1)[0]
    ) == pytest.approx(impact)
    result.summary()
    printed = capsys.readouterr().out
    assert "CI_" not in printed
    assert f"Expected effect = {impact:.2f}" in printed


def test_rd_epsilon_contrast_uses_the_configured_rows(capsys):
    """A slope makes the wrong threshold positions fail."""
    rng = np.random.default_rng(0)
    running = np.linspace(-1, 1, 41)
    treated = (running > 0).astype(float)
    outcome = 2.0 * running + 3.4 * treated + rng.normal(0, 0.01, size=running.size)
    frame = pd.DataFrame({"x": running, "y": outcome, "treated": treated})
    result = cp.RegressionDiscontinuity(
        frame,
        formula="y ~ 1 + x + treated",
        treatment_threshold=0.0,
        epsilon=0.001,
        model=LinearRegression(fit_intercept=False),
    ).fit()
    effect = result.result.discontinuity_at_threshold
    assert isinstance(effect, xr.DataArray)
    assert effect.sizes.get("chain") == 1
    assert effect.sizes.get("draw") == 1
    assert float(np.asarray(effect).reshape(())) == pytest.approx(3.4, abs=0.05)
    assert float(
        np.asarray(result.result.expected_effect).reshape(())
    ) == pytest.approx(float(np.asarray(effect).reshape(())))
    predicted = result._model_backend.predict(result.x_discon_design)
    upper = np.asarray(predicted.isel(obs_ind=1, treated_units=0)).reshape(-1)[0]
    lower = np.asarray(predicted.isel(obs_ind=0, treated_units=0)).reshape(-1)[0]
    assert float(np.asarray(effect).reshape(())) == pytest.approx(float(upper - lower))
    result.summary()
    printed = capsys.readouterr().out
    assert "CI_" not in printed
    point = f"{float(np.asarray(effect).reshape(())):.2f}"
    assert f"Discontinuity at threshold = {point}" in printed


def test_its_and_sc_labels_bump_and_unavailable_path():
    """Known means, a different realized series, and a counterfactual-only fit."""
    index = pd.RangeIndex(4)
    mu_treated = xr.DataArray(
        [3.0, 4.0, 5.0, 6.0], dims=("obs_ind",), coords={"obs_ind": index}
    )
    mu_untreated = xr.DataArray(
        [1.0, 1.0, 1.0, 1.0], dims=("obs_ind",), coords={"obs_ind": index}
    )
    realized = xr.DataArray(
        [10.0, 1.0, 1.0, 1.0], dims=("obs_ind",), coords={"obs_ind": index}
    )
    expected = response_contrast(mu_treated, mu_untreated, reduce_dims=())
    gap = realized - mu_untreated
    assert list(np.asarray(expected)) == [2.0, 3.0, 4.0, 5.0]
    assert headline_mean(expected) == pytest.approx(3.5)
    assert float(cumulative_sum(expected).isel(obs_ind=-1)) == pytest.approx(14.0)
    bumped_gap = gap.copy()
    bumped_gap[{"obs_ind": 1}] = bumped_gap[{"obs_ind": 1}] + 1
    assert float(bumped_gap.isel(obs_ind=1) - gap.isel(obs_ind=1)) == pytest.approx(1.0)
    assert headline_mean(expected) == pytest.approx(3.5)

    its = _fit_its()
    assert its.result.expected_effect is None
    assert "untreated" in its.result.expected_effect_reason
    unavailable = its.effect_summary()
    assert "unavailable" in unavailable.text
    gap_before = its.result.impact_post.copy()
    updated = its.post_design["y"].copy()
    values = np.asarray(updated).copy()
    flat = values.reshape(-1)
    flat[0] = flat[0] + 1
    its.post_design["y"] = updated.copy(data=flat.reshape(updated.shape))
    its._finalize("posterior")
    delta = its.result.impact_post - gap_before
    assert float(np.asarray(delta.isel(obs_ind=0)).reshape(-1)[0]) == pytest.approx(1.0)
    assert float(np.asarray(delta.isel(obs_ind=slice(1, None))).reshape(-1).sum()) == 0
    assert its.result.expected_effect is None

    plotted = expected.expand_dims(chain=[0], draw=[0])
    its._assign_bundle(
        "posterior",
        replace(its.result, expected_effect=plotted, expected_effect_reason=None),
    )
    labeled = its.effect_summary()
    assert labeled.table.index[0] == "expected_effect"
    assert "Expected effect" in labeled.text
    figure, axes = its.plot(show=False)
    titles = [axis.get_title() for axis in axes]
    assert "Expected effect" in titles
    assert "Cumulative expected effect" in titles
    figure.clf()

    synthetic = _fit_sc()
    assert synthetic.result.expected_effect is None
    assert "unavailable" in synthetic.effect_summary().text
    figure, axes = synthetic.plot(show=False)
    assert "unavailable" in axes[1].get_title().lower()
    assert "unavailable" in axes[2].get_title().lower()
    gap = np.asarray(synthetic.result.impact_post.sel(treated_units="actual")).reshape(
        -1
    )
    for line in axes[1].lines:
        ydata = np.asarray(line.get_ydata(), dtype=float).reshape(-1)
        if ydata.size == gap.size:
            assert not np.allclose(ydata, gap)
    figure.clf()
    synthetic._assign_bundle(
        "posterior",
        replace(synthetic.result, expected_effect=plotted, expected_effect_reason=None),
    )
    assert "Expected effect" in synthetic.effect_summary().text


def test_bayesian_identity_does_not_read_the_outcome(monkeypatch):
    """Prior and posterior contrasts match the interaction and ignore y."""
    rows = []
    unit = 0
    for group in (0, 1):
        for post in (0, 1):
            for _ in range(4):
                rows.append(
                    {
                        "unit": f"u{unit}",
                        "t": post,
                        "group": group,
                        "post_treatment": post,
                        "y": 1.0 + group + post + 4.0 * group * post,
                    }
                )
                unit += 1
    frame = pd.DataFrame(rows)
    experiment = cp.DifferenceInDifferences(
        frame,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=LIGHT_SAMPLE),
    )
    experiment.sample_prior_predictive(draws=20, random_seed=42)
    _assert_matches_coefficient(experiment, "prior")
    experiment.fit()
    _assert_matches_coefficient(experiment, "posterior")

    calls = []
    original = experiment._model_backend.predict

    def spy(X, group="posterior"):
        calls.append(np.asarray(X))
        return original(X, group=group)

    monkeypatch.setattr(experiment._model_backend, "predict", spy)
    experiment.data.loc[:, "y"] = np.nan
    again = experiment._expected_effect("posterior")
    assert calls
    assert again.equals(experiment.result.causal_impact)


def test_contrast_helpers_and_guard_clauses():
    """Cover weight, window, and unavailable-name branches the five designs share."""
    treated = xr.DataArray([1.0, 3.0], dims=("obs_ind",))
    untreated = xr.DataArray([0.0, 1.0], dims=("obs_ind",))
    weights = xr.DataArray([1.0, 3.0], dims=("obs_ind",))
    weighted = response_contrast(
        treated, untreated, weights=weights, reduce_dims=("obs_ind",)
    )
    assert float(np.asarray(weighted)) == pytest.approx(1.75)
    point = xr.DataArray(4.0)
    assert float(np.asarray(cumulative_sum(point))) == pytest.approx(4.0)
    assert cumulative_total(point) == pytest.approx(4.0)

    labeled = xr.DataArray(
        [1.0, 2.0, 3.0], dims=("obs_ind",), coords={"obs_ind": [10, 20, 30]}
    )
    windowed = align_expected_window(labeled, pd.Index([10, 20, 30]), pd.Index([20]))
    assert float(np.asarray(windowed).reshape(-1)[0]) == pytest.approx(2.0)
    positional = xr.DataArray(
        [1.0, 2.0, 3.0], dims=("obs_ind",), coords={"obs_ind": [0, 1, 2]}
    )
    post = pd.Index(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    assert align_expected_window(positional, post, post) is positional
    placed = align_expected_window(
        positional, post, pd.Index(pd.to_datetime(["2020-01-02"]))
    )
    assert float(np.asarray(placed).reshape(-1)[0]) == pytest.approx(2.0)
    bundle = CausalResult(
        predictions_pre=labeled,
        predictions_post=labeled,
        impact_pre=labeled,
        impact_post=labeled,
        impact_post_cumulative=labeled,
    )
    with pytest.raises(ValueError, match="effect must be"):
        select_headline_effect(bundle, "not-an-effect")
    assert not hasattr(bundle, "observed_gap")

    its = _fit_its()
    figure, _axes = its.plot(show=False, effect="observed_gap")
    figure.clf()
    with pytest.raises(ValueError, match="effect must be"):
        its.effect_summary(effect="not-an-effect")
    three = cp.InterruptedTimeSeries(
        pd.DataFrame({"t": np.arange(12), "y": np.ones(12)}),
        treatment_time=4,
        treatment_end_time=8,
        formula="y ~ 1",
        model=LinearRegression(),
    ).fit()
    with pytest.raises(ValueError, match="observed_gap"):
        three.effect_summary(period="comparison")

    synthetic = _fit_sc()
    with pytest.warns(UserWarning, match="ignored"):
        synthetic.effect_summary(effect="expected_effect", period="post")
    with pytest.raises(ValueError, match="effect must be"):
        synthetic.effect_summary(effect="not-an-effect")
    plotted = xr.DataArray(
        [[1.0, 2.0, 3.0]],
        dims=("treated_units", "obs_ind"),
        coords={"treated_units": ["actual"], "obs_ind": synthetic.datapost.index},
    )
    synthetic._assign_bundle(
        "posterior",
        replace(synthetic.result, expected_effect=plotted, expected_effect_reason=None),
    )
    assert "Expected effect" in synthetic.effect_summary(treated_unit="actual").text

    rows = []
    unit = 0
    for group in (0, 1):
        for post_flag in (0, 1):
            rows.append(
                {
                    "unit": f"u{unit}",
                    "t": post_flag,
                    "group": group,
                    "post_treatment": post_flag,
                    "y": 1.0 + group + post_flag + 4.0 * group * post_flag,
                }
            )
            unit += 1
    did = cp.DifferenceInDifferences(
        pd.DataFrame(rows),
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(fit_intercept=False),
    ).fit()
    did.data = did.data.query("group == 0")
    with pytest.raises(ValueError, match="No treated post-period"):
        did._expected_effect("posterior")

    prepost = cp.PrePostNEGD(
        pd.DataFrame(
            {
                "group": [0, 0, 1, 1],
                "pre": [1.0, 2.0, 1.5, 2.5],
                "post": [1.0, 2.0, 5.5, 6.5],
            }
        ),
        formula="post ~ 1 + C(group) + pre",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=_ClosedFormCoefficientModel(),
    ).fit()
    bare = prepost.result.causal_impact.squeeze(("chain", "draw"), drop=True)
    prepost._assign_bundle("posterior", replace(prepost.result, causal_impact=bare))
    figure, axes = prepost.plot(show=False)
    assert any("Expected effect" in text.get_text() for text in axes[1].texts)
    figure.clf()

    captured: dict[str, object] = {}

    class _Refit:
        def fit(self):
            return its

    original = its.effect_summary

    def _record(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    its.effect_summary = _record  # type: ignore[method-assign]
    context = PipelineContext(data=its.data)
    context.experiment_config = {
        "method": lambda *_a, **_k: _Refit(),
        "formula": "y ~ 1",
    }
    PriorSensitivity(alternatives=[{"name": "alt", "model": LinearRegression()}]).run(
        its, context
    )
    assert captured.get("effect") == "observed_gap"


def _assert_matches_coefficient(experiment, group: str) -> None:
    bundle = experiment.prior_result if group == "prior" else experiment.result
    impact = bundle.causal_impact
    assert isinstance(impact, xr.DataArray)
    assert "chain" in impact.dims
    assert "draw" in impact.dims
    coefficient = experiment.idata[group]["beta"].sel(
        treated_units="unit_0",
        coeffs=next(
            label
            for label in experiment.labels
            if experiment._is_treatment_interaction(label)
        ),
    )
    xr.testing.assert_allclose(impact, coefficient.reset_coords(drop=True))


def _fit_its():
    time = np.arange(12)
    outcome = np.concatenate([np.full(8, 1.0), np.full(4, 10.0)])
    return cp.InterruptedTimeSeries(
        pd.DataFrame({"t": time, "y": outcome}),
        treatment_time=8,
        formula="y ~ 1",
        model=LinearRegression(),
    ).fit()


def _fit_sc():
    frame = pd.DataFrame(
        {
            "donor": np.linspace(0, 1, 10),
            "actual": np.concatenate([np.linspace(0, 1, 7), np.full(3, 5.0)]),
        },
        index=np.arange(10),
    )
    return cp.SyntheticControl(
        frame,
        treatment_time=7,
        control_units=["donor"],
        treated_units=["actual"],
        model=cp.skl_models.WeightedProportion(),
    ).fit()
