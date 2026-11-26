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
from causalpy.checks.outcome_falsification import OutcomeFalsification
from causalpy.checks.prior_sensitivity import PriorSensitivity
from causalpy.experiments._expected_effect import (
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

    plotted = xr.DataArray(
        [2.0, 3.0, 4.0, 5.0],
        dims=("obs_ind",),
        coords={"obs_ind": its.datapost.index},
    ).expand_dims(chain=[0], draw=[0])
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
    post = synthetic.datapost.index
    synthetic_effect = xr.DataArray(
        np.arange(1.0, len(post) + 1.0),
        dims=("obs_ind",),
        coords={"obs_ind": post},
    ).expand_dims(chain=[0], draw=[0])
    synthetic._assign_bundle(
        "posterior",
        replace(
            synthetic.result,
            expected_effect=synthetic_effect,
            expected_effect_reason=None,
        ),
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
    """Cover weight and unavailable-name branches the five designs share."""
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

    bundle = CausalResult(
        predictions_pre=treated,
        predictions_post=treated,
        impact_pre=treated,
        impact_post=treated,
        impact_post_cumulative=treated,
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


def test_prepost_extra_covariate_uses_each_rows_value():
    """A covariate outside the pretest grid must not stop the contrast.

    The plotted curves hold ``site`` at its mean. The stored contrast uses
    each row's own ``site``, so it is not the effect evaluated at that mean.
    """
    rows = []
    for group in (0, 1):
        for site in (0.0, 2.0):
            for region, pre in (("a", 1.0), ("b", 2.0), ("a", 3.0)):
                rows.append(
                    {
                        "group": group,
                        "pre": pre,
                        "site": site,
                        "region": region,
                        "post": (
                            1.5
                            + 4.0 * group
                            + 3.0 * group * site**2
                            + 0.7 * pre
                            + float(region == "b")
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    own = 4.0 + 3.0 * float(np.mean(frame["site"].to_numpy() ** 2))
    at_mean = 4.0 + 3.0 * float(frame["site"].mean()) ** 2
    assert own != pytest.approx(at_mean)
    result = cp.PrePostNEGD(
        frame,
        formula="post ~ 1 + C(group) * I(site ** 2) + pre + C(region)",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=_ClosedFormCoefficientModel(),
    ).fit()
    impact = float(np.asarray(result.result.causal_impact).reshape(-1)[0])
    assert impact == pytest.approx(own)
    assert impact != pytest.approx(at_mean)
    plotted = result.result.scenario_control.inputs
    assert plotted["site"].nunique() == 1
    assert float(plotted["site"].iloc[0]) == pytest.approx(float(frame["site"].mean()))
    assert set(plotted["region"].unique()) == {"a"}
    result.data.loc[result.data.index[0], "post"] += 1
    recomputed = float(np.asarray(result._expected_effect("posterior")).reshape(-1)[0])
    assert recomputed == pytest.approx(own)


def _broadcast_draws(array: xr.DataArray, n_draw: int = 4) -> xr.DataArray:
    """Repeat a singleton draw so plotting takes the uncertainty path."""
    point = array.isel(chain=0, draw=0, drop=True)
    draws = [
        (point + float(offset)).expand_dims(draw=[offset]) for offset in range(n_draw)
    ]
    return xr.concat(draws, dim="draw").expand_dims(chain=[0])


def _assign_draws(experiment, n_draw: int = 4) -> None:
    bundle = experiment.result
    names = (
        "predictions_pre",
        "predictions_post",
        "impact_pre",
        "impact_post",
        "impact_post_cumulative",
    )
    fields = {
        name: _broadcast_draws(getattr(bundle, name), n_draw)
        for name in names
        if isinstance(getattr(bundle, name), xr.DataArray)
    }
    experiment._assign_bundle("posterior", replace(bundle, **fields))


def _legend_labels(axis) -> list[str]:
    legend = axis.get_legend()
    if legend is None:
        return []
    return [text.get_text() for text in legend.get_texts()]


def _line_values(axis, n_points: int, *, marker: str) -> np.ndarray:
    matches = []
    for line in axis.get_lines():
        values = np.asarray(line.get_ydata(), dtype=float).reshape(-1)
        if values.size == n_points and line.get_marker() == marker:
            matches.append(values)
    assert len(matches) == 1
    return matches[0]


def _series_mean(array: xr.DataArray) -> np.ndarray:
    reduced = array
    if "treated_units" in reduced.dims:
        reduced = reduced.isel(treated_units=0)
    return np.asarray(reduced.mean(("chain", "draw")).values, dtype=float).reshape(-1)


def _fit_its_one_post():
    time = np.arange(5)
    outcome = np.concatenate([np.full(4, 1.0), [10.0]])
    return cp.InterruptedTimeSeries(
        pd.DataFrame({"t": time, "y": outcome}),
        treatment_time=4,
        formula="y ~ 1",
        model=LinearRegression(),
    ).fit()


def test_point_estimate_panels_draw_the_mean_effect():
    """Sklearn panels draw the stored mean and annotate only a stated reason."""
    synthetic = _fit_sc()
    gap = _series_mean(synthetic.result.impact_post)
    cumulative = _series_mean(synthetic.result.impact_post_cumulative)
    figure, axes = synthetic.plot(show=False, effect="observed_gap")
    assert _line_values(axes[1], gap.size, marker=".") == pytest.approx(gap)
    assert _line_values(axes[2], cumulative.size, marker="None") == pytest.approx(
        cumulative
    )
    assert "model fit" in _legend_labels(axes[0])
    assert axes[0].collections
    figure.clf()

    post_index = synthetic.datapost.index
    series = xr.DataArray(
        [1.0, 2.0, 4.0],
        dims=("obs_ind",),
        coords={"obs_ind": post_index},
    ).expand_dims(chain=[0], draw=[0])
    synthetic._assign_bundle(
        "posterior",
        replace(synthetic.result, expected_effect=series, expected_effect_reason=None),
    )
    figure, axes = synthetic.plot(show=False, effect="expected_effect")
    assert _line_values(axes[1], 3, marker=".") == pytest.approx([1.0, 2.0, 4.0])
    assert axes[1].get_title() == "Expected effect"
    assert len(axes[0].texts) == 0
    figure.clf()

    synthetic._assign_bundle(
        "posterior",
        replace(
            synthetic.result,
            expected_effect=None,
            expected_effect_reason="no treated mean in this fit",
        ),
    )
    figure, axes = synthetic.plot(show=False, effect="expected_effect")
    assert [text.get_text() for text in axes[0].texts] == [
        "no treated mean in this fit"
    ]
    assert axes[1].texts
    assert "no treated mean in this fit" in axes[1].texts[0].get_text()
    figure.clf()


def test_uncertainty_legend_names_the_gap_only_for_several_points():
    """Several posterior points name the gap; one point draws the HDI marker."""
    its = _fit_its()
    _assign_draws(its)
    figure, axes = its.plot(show=False, effect="observed_gap")
    assert "Observed gap" in _legend_labels(axes[0])
    assert list(axes[1].containers) == []
    figure.clf()

    one_post = _fit_its_one_post()
    _assign_draws(one_post)
    figure, axes = one_post.plot(show=False, effect="observed_gap")
    labels = _legend_labels(axes[0])
    assert "Observed gap" not in labels
    assert "Counterfactual" in labels
    assert len(axes[1].containers) == 1
    figure.clf()

    synthetic = _fit_sc()
    _assign_draws(synthetic)
    figure, axes = synthetic.plot(show=False, effect="observed_gap")
    assert "Observed gap" in _legend_labels(axes[0])
    figure.clf()


def test_missing_reason_draws_no_unavailable_annotation():
    """No reason means a blank panel, not a repeated title."""
    its = _fit_its()
    its._assign_bundle(
        "posterior",
        replace(its.result, expected_effect=None, expected_effect_reason=None),
    )
    figure, axes = its.plot(show=False, effect="expected_effect")
    assert len(axes[1].texts) == 0
    assert len(axes[2].texts) == 0
    assert axes[1].get_title() == "Expected effect unavailable"
    figure.clf()


def test_expected_effect_summary_can_omit_the_cumulative_sum():
    """cumulative=False keeps the headline mean and drops the running total."""
    its = _fit_its()
    plotted = xr.DataArray(
        [2.0, 3.0, 4.0, 5.0],
        dims=("obs_ind",),
        coords={"obs_ind": its.datapost.index},
    ).expand_dims(chain=[0], draw=[0])
    its._assign_bundle(
        "posterior",
        replace(its.result, expected_effect=plotted, expected_effect_reason=None),
    )
    point = its.effect_summary(cumulative=False)
    assert "cumulative_sum" not in point.table.columns
    assert "Cumulative sum" not in point.text
    assert float(point.table["mean"].iloc[0]) == pytest.approx(3.5)
    full = its.effect_summary()
    assert float(full.table["cumulative_sum"].iloc[0]) == pytest.approx(14.0)
    assert "Cumulative sum" in full.text


def test_transformed_outcome_is_not_dropped_before_the_contrast():
    """np.log(y) is not a column, so the raw outcome stays for the formula."""
    rows = []
    for group in (0, 1):
        for post in (0, 1):
            log_outcome = 0.2 + 0.1 * group + 0.3 * post + 1.5 * group * post
            for _copy in range(4):
                rows.append(
                    {
                        "unit": f"u{len(rows)}",
                        "t": post,
                        "group": group,
                        "post_treatment": post,
                        "y": float(np.exp(log_outcome)),
                    }
                )
    result = cp.DifferenceInDifferences(
        pd.DataFrame(rows),
        formula="np.log(y) ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(fit_intercept=False),
    ).fit()
    assert result.outcome_variable_name == "np.log(y)"
    assert "np.log(y)" not in result.data.columns
    assert "y" in result.data.columns
    impact = float(np.asarray(result.result.causal_impact).reshape(-1)[0])
    assert impact == pytest.approx(1.5)


def test_stated_reference_rejects_empty_or_nonfinite_columns():
    """A plot reference fails closed when a covariate has no usable value."""
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
    prepost.data["region"] = pd.Series(
        [None, None, None, None], index=prepost.data.index, dtype=object
    )
    with pytest.raises(ValueError, match="no values"):
        prepost._stated_reference("region")
    prepost.data["site"] = np.nan
    with pytest.raises(ValueError, match="no finite values"):
        prepost._stated_reference("site")


def test_outcome_falsification_rejects_other_designs():
    """The extractor names the three designs it can read."""
    with pytest.raises(TypeError, match="InterruptedTimeSeries"):
        OutcomeFalsification._extract_effect_stats(object(), alpha=0.05)


def test_prior_sensitivity_omits_effect_for_other_designs():
    """Difference-in-differences summaries do not take an effect argument."""
    rows = []
    for group in (0, 1):
        for post in (0, 1):
            rows.append(
                {
                    "unit": f"u{len(rows)}",
                    "t": post,
                    "group": group,
                    "post_treatment": post,
                    "y": 1.0 + group + post + 4.0 * group * post,
                }
            )
    did = cp.DifferenceInDifferences(
        pd.DataFrame(rows),
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(fit_intercept=False),
    ).fit()
    captured: dict[str, object] = {}
    original = did.effect_summary

    def _record(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    did.effect_summary = _record  # type: ignore[method-assign]

    class _Refit:
        def fit(self):
            return did

    context = PipelineContext(data=did.data)
    context.experiment_config = {
        "method": lambda *_args, **_kwargs: _Refit(),
        "formula": "y ~ 1",
    }
    result = PriorSensitivity(
        alternatives=[{"name": "alt", "model": LinearRegression()}]
    ).run(did, context)
    assert "effect" not in captured
    assert "error" not in result.table.columns
