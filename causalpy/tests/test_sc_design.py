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
"""Contract tests for the fixed-target synthetic-control planner.

The deterministic oracle below is independent of the planner's private
translation. A same-seed analysis fit is the Monte Carlo comparison. Post-period
treated values are not fit inputs, so that comparison should agree up to
graph-compilation noise; the tolerance is not a second sampler's slack.
"""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pymc_extras.prior import Prior

import causalpy as cp
from causalpy._arviz_compat import hdi_bounds
from causalpy.constants import HDI_PROB
from causalpy.design import (
    DecisionRule,
    HorizonEvaluation,
    PlannerResult,
    SyntheticControlPlanner,
    _score_translated_lifts,
)
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pymc_models import WeightedSumFitter
from causalpy.tests.known_truth_panels import exact_convex_hull

SAMPLE_KWARGS = {
    "tune": 20,
    "draws": 20,
    "chains": 1,
    "cores": 1,
    "progressbar": False,
    "random_seed": 42,
}
# Same split, seed, and priors. Widen only if a same-seed rerun exceeds this.
INDEPENDENT_FIT_ATOL = 1e-5
INDEPENDENT_FIT_RTOL = 1e-5
UNIT = exact_convex_hull.treated_units[0]
CONTROLS = list(exact_convex_hull.control_units)
DECISION = DecisionRule("positive", "cumulative", 0.0, 0.95)


def _history() -> pd.DataFrame:
    return exact_convex_hull.untreated.copy()


def _planner(frame: pd.DataFrame, **kwargs) -> SyntheticControlPlanner:
    model = kwargs.pop("model", None)
    if model is None:
        model = WeightedSumFitter(sample_kwargs=dict(SAMPLE_KWARGS))
    return SyntheticControlPlanner(
        frame,
        UNIT,
        CONTROLS,
        kwargs.pop("decision", DECISION),
        model=model,
        **kwargs,
    )


def _direct_fit(frame: pd.DataFrame, duration: int) -> SyntheticControl:
    return SyntheticControl(
        frame,
        frame.index[-duration],
        control_units=CONTROLS,
        treated_units=[UNIT],
        model=WeightedSumFitter(sample_kwargs=dict(SAMPLE_KWARGS)),
    ).fit()


def _oracle_effect(baseline_impact, observed, lift, scale, estimand):
    """Independent translation. The denominator is the observed holdout."""
    observed_values = np.asarray(observed.values, dtype=float)
    if scale == "relative":
        delta = observed_values * lift
    elif scale == "absolute":
        delta = np.full_like(observed_values, lift, dtype=float)
    else:
        raise AssertionError(scale)
    pointwise = np.asarray(baseline_impact.values, dtype=float) + delta
    if estimand == "cumulative":
        reduced = pointwise.sum(axis=-1)
        truth = float(delta.sum())
    elif estimand == "average":
        reduced = pointwise.mean(axis=-1)
        truth = float(delta.mean())
    else:
        raise AssertionError(estimand)
    return reduced.reshape(-1), truth


def _covers(draws: np.ndarray, truth: float) -> bool:
    array = xr.DataArray(draws.reshape(1, -1), dims=("chain", "draw"))
    lower, upper = hdi_bounds(array, prob=HDI_PROB, dim=("chain", "draw"))
    return bool(lower <= truth <= upper)


@pytest.fixture(scope="module")
def fitted_rehearsal():
    planner = _planner(_history())
    result = planner.evaluate(
        durations=[4],
        relative_lifts=[0.1],
        absolute_lifts=[1.0],
    )
    return planner, result


def test_construction_and_screen_do_not_sample(monkeypatch):
    def reject_fit(*args, **kwargs):
        raise AssertionError("construction or screen sampled")

    monkeypatch.setattr(SyntheticControl, "fit", reject_fit)
    monkeypatch.setattr(WeightedSumFitter, "fit", reject_fit)
    planner = _planner(_history())
    assert planner.model.idata is None
    screen = planner.screen()
    assert planner.model.idata is None
    assert set(screen.correlation) == set(CONTROLS)
    assert screen.convex_hull_passes
    assert "correlation" in screen.summary().columns
    assert "weight" not in screen.summary().columns


def test_groups_two_sided_and_unsupported_estimators_fail_before_fitting(monkeypatch):
    def reject_fit(*args, **kwargs):
        raise AssertionError("invalid input reached a fit")

    monkeypatch.setattr(SyntheticControl, "fit", reject_fit)
    history = _history()
    with pytest.raises(TypeError, match="Groups are not supported"):
        SyntheticControlPlanner(history, [UNIT], CONTROLS, DECISION)
    with pytest.raises(TypeError, match="sequence of unit-name strings"):
        SyntheticControlPlanner(history, UNIT, "abc", DECISION)
    with pytest.raises(ValueError, match="Two-sided decisions are not supported"):
        DecisionRule("two-sided", "cumulative", 0.0, 0.95)
    with pytest.raises(ValueError, match="Unsupported estimator"):
        SyntheticControlPlanner(
            history,
            UNIT,
            CONTROLS,
            DECISION,
            estimator="augmented_synthetic_control",
        )
    with pytest.raises(TypeError, match="Unsupported estimator model"):
        SyntheticControlPlanner(
            history,
            UNIT,
            CONTROLS,
            DECISION,
            model=cp.pymc_models.SoftmaxWeightedSumFitter(),
        )


def test_undefined_relative_baselines_are_rejected_before_fitting(monkeypatch):
    calls = []

    def reject_fit(*args, **kwargs):
        calls.append("fit")
        raise AssertionError("undefined relative baseline reached a fit")

    monkeypatch.setattr(SyntheticControl, "fit", reject_fit)
    signed = _history()
    signed.loc[signed.index[-4:], UNIT] = -1.0
    with pytest.raises(ValueError, match="Relative effects are undefined") as exc:
        _planner(signed).evaluate(durations=[4], relative_lifts=[0.1])
    text = str(exc.value)
    median = float(text.split("outcome is ", 1)[1].split(". The applied floor", 1)[0])
    applied = float(text.split("applied floor is ", 1)[1].split(" ", 1)[0])
    assert applied < median
    assert applied < 1e-6
    assert applied == pytest.approx(1e-8 * median)
    assert "outcome (" not in text
    near_zero = _history()
    near_zero.loc[near_zero.index[-4:], UNIT] = 1e-12
    with pytest.raises(ValueError, match="Relative effects are undefined"):
        _planner(near_zero).evaluate(durations=[4], relative_lifts=[0.1])
    assert calls == []


def test_declared_relative_floor_replaces_the_scale_rule(monkeypatch):
    calls = []

    def stop_after_gate(self, **kwargs):
        calls.append(self.treatment_time)
        raise RuntimeError("stop after the relative gate")

    monkeypatch.setattr(SyntheticControl, "fit", stop_after_gate)
    near_zero = _history()
    near_zero.loc[near_zero.index[-4:], UNIT] = 1e-12
    result = _planner(near_zero, relative_floor=0.0).evaluate(
        durations=[4],
        relative_lifts=[0.1],
    )
    assert calls == [near_zero.index[-4]]
    assert result.status == "fit-failed"
    assert result.horizons[0].scores == ()
    assert result.summary()["detected"].isna().all()


def test_zero_training_scale_rejects_relative_lifts_unless_floor_is_declared(
    monkeypatch,
):
    calls = []

    def reject_fit(*args, **kwargs):
        calls.append("fit")
        raise AssertionError("undefined relative baseline reached a fit")

    monkeypatch.setattr(SyntheticControl, "fit", reject_fit)
    zero = _history()
    zero.loc[:, UNIT] = 0.0
    zero.loc[zero.index[-1], UNIT] = 1e-9
    with pytest.raises(ValueError, match="strictly positive"):
        _planner(zero).evaluate(durations=[1], relative_lifts=[0.1])
    spiked = _history()
    spiked.loc[:, UNIT] = 0.0
    spiked.loc[spiked.index[0], UNIT] = 100.0
    spiked.loc[spiked.index[-1], UNIT] = 1e-9
    with pytest.raises(ValueError, match="strictly positive"):
        _planner(spiked).evaluate(durations=[1], relative_lifts=[0.1])
    assert calls == []

    def stop(self, **kwargs):
        calls.append("fit")
        raise RuntimeError("counted")

    monkeypatch.setattr(SyntheticControl, "fit", stop)
    opened = _history()
    opened.loc[:, UNIT] = 0.0
    opened.loc[opened.index[-1], UNIT] = 1e-9
    result = _planner(opened, relative_floor=0.0).evaluate(
        durations=[1], relative_lifts=[0.1]
    )
    assert calls == ["fit"]
    assert result.status == "fit-failed"
    signed = opened.copy()
    signed.loc[signed.index[-1], UNIT] = -1e-9
    with pytest.raises(ValueError, match="undefined"):
        _planner(signed, relative_floor=0.0).evaluate(
            durations=[1], relative_lifts=[0.1]
        )


def test_nonzero_threshold_does_not_decide_the_other_estimand():
    baseline = xr.DataArray(
        np.zeros((1, 1, 4)),
        dims=("chain", "draw", "obs_ind"),
        coords={"obs_ind": range(4)},
    )
    observed = xr.DataArray(np.ones(4), dims="obs_ind", coords={"obs_ind": range(4)})
    decision = DecisionRule("positive", "cumulative", 5.0, 0.9)
    scores = _score_translated_lifts(
        baseline_impact=baseline,
        observed=observed,
        counterfactual=xr.zeros_like(baseline),
        relative_lifts=None,
        absolute_lifts=(2.0,),
        decision=decision,
    )
    by_estimand = {score.estimand: score for score in scores}
    assert by_estimand["cumulative"].truth == pytest.approx(8.0)
    assert by_estimand["cumulative"].detected is True
    assert by_estimand["cumulative"].tail_probability == pytest.approx(1.0)
    assert by_estimand["average"].truth == pytest.approx(2.0)
    assert by_estimand["average"].detected is None
    assert by_estimand["average"].tail_probability is None
    assert by_estimand["average"].mean == pytest.approx(2.0)


def test_failed_fit_does_not_report_a_detection(monkeypatch):
    def fail(self, **kwargs):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(SyntheticControl, "fit", fail)
    result = _planner(_history()).evaluate(durations=[4], absolute_lifts=[1.0])
    assert result.status == "fit-failed"
    assert result.n_inference_fits == 1
    assert result.fit_error is not None
    assert "sampling failed" in result.fit_error
    assert result.horizons[0].scores == ()
    assert "No detection is reported" in result.recommendation()
    assert result.summary()["detected"].isna().all()
    with pytest.raises(ValueError, match="failed fit"):
        result.plot()


def test_translation_coverage_is_not_a_detection_curve():
    baseline = xr.DataArray(
        np.array([[[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [0.2, 0.2]]]),
        dims=("chain", "draw", "obs_ind"),
        coords={"obs_ind": [0, 1]},
    )
    observed = xr.DataArray(
        [10.0, 30.0],
        dims="obs_ind",
        coords={"obs_ind": [0, 1]},
    )
    counterfactual = xr.zeros_like(baseline)
    decision = DecisionRule("positive", "cumulative", 0.0, 0.75)
    scores = _score_translated_lifts(
        baseline_impact=baseline,
        observed=observed,
        counterfactual=counterfactual,
        relative_lifts=(0.0, 0.5),
        absolute_lifts=(0.0, 5.0),
        decision=decision,
    )
    coverages = set()
    detections = set()
    for score in scores:
        expected, truth = _oracle_effect(
            baseline, observed, score.lift, score.effect_scale, score.estimand
        )
        np.testing.assert_allclose(score.effect, expected)
        assert score.truth == pytest.approx(truth)
        if score.effect_scale == "relative" and score.estimand == "cumulative":
            assert score.truth == pytest.approx(score.lift * float(observed.sum()))
        baseline_effect, _ = _oracle_effect(
            baseline, observed, 0.0, score.effect_scale, score.estimand
        )
        assert score.truth_covers is _covers(baseline_effect, 0.0)
        assert score.counterfactual is counterfactual
        coverages.add(score.truth_covers)
        if score.estimand == decision.estimand:
            detections.add(score.detected)
        else:
            assert score.detected is None
            assert score.tail_probability is None
    assert len(coverages) == 1
    assert detections == {False, True}


def test_rehearsal_matches_the_analysis_fit_and_the_oracle(fitted_rehearsal):
    planner, result = fitted_rehearsal
    assert planner.model.idata is None
    assert result.status == "historical-descriptive"
    assert result.horizon_policy == "separate-fits"
    assert result.n_inference_fits == 1
    assert result.relative_baseline_convention == "observed-holdout"
    horizon = result.horizons[0]
    assert horizon.holdout_index.tolist() == _history().index[-4:].tolist()
    assert horizon.training_index.tolist() == _history().index[:-4].tolist()
    scales = {score.effect_scale for score in horizon.scores}
    estimands = {score.estimand for score in horizon.scores}
    assert scales == {"absolute", "relative"}
    assert estimands == {"cumulative", "average"}
    assert any(score.lift == 0.0 for score in horizon.scores)
    for score in horizon.scores:
        assert score.counterfactual is horizon.counterfactual
        expected, truth = _oracle_effect(
            horizon.baseline_impact,
            horizon.observed_holdout,
            score.lift,
            score.effect_scale,
            score.estimand,
        )
        np.testing.assert_allclose(score.effect, expected)
        assert score.truth == pytest.approx(truth)
        baseline_effect, _ = _oracle_effect(
            horizon.baseline_impact,
            horizon.observed_holdout,
            0.0,
            score.effect_scale,
            score.estimand,
        )
        assert score.truth_covers is _covers(baseline_effect, 0.0)
    direct = _direct_fit(_history(), 4)
    np.testing.assert_allclose(
        horizon.counterfactual.values,
        direct.result.predictions_post.sel(treated_units=UNIT).values,
        atol=INDEPENDENT_FIT_ATOL,
        rtol=INDEPENDENT_FIT_RTOL,
    )
    injected = _history()
    injected.loc[injected.index[-4:], UNIT] *= 1.1
    injected_fit = _direct_fit(injected, 4)
    relative = next(
        score
        for score in horizon.scores
        if score.effect_scale == "relative"
        and score.lift == pytest.approx(0.1)
        and score.estimand == "cumulative"
    )
    np.testing.assert_allclose(
        relative.effect,
        injected_fit.result.impact_post.sel(treated_units=UNIT)
        .sum("obs_ind")
        .values.reshape(-1),
        atol=INDEPENDENT_FIT_ATOL,
        rtol=INDEPENDENT_FIT_RTOL,
    )
    shifted = _history()
    shifted.loc[shifted.index[-4:], UNIT] += 1.0
    shifted_fit = _direct_fit(shifted, 4)
    absolute = next(
        score
        for score in horizon.scores
        if score.effect_scale == "absolute"
        and score.lift == pytest.approx(1.0)
        and score.estimand == "cumulative"
    )
    np.testing.assert_allclose(
        absolute.effect,
        shifted_fit.result.impact_post.sel(treated_units=UNIT)
        .sum("obs_ind")
        .values.reshape(-1),
        atol=INDEPENDENT_FIT_ATOL,
        rtol=INDEPENDENT_FIT_RTOL,
    )
    table = result.summary()
    assert {"truth_covers", "detected", "estimand", "status"} <= set(table.columns)
    assert set(table["status"]) == {"historical-descriptive"}
    other = table["estimand"] != result.decision.estimand
    assert table.loc[other, "detected"].isna().all()
    assert table.loc[other, "tail_probability"].isna().all()
    assert table.loc[~other, "detected"].notna().all()
    fig, ax = result.plot(estimand="average")
    assert len(ax.lines) == 1
    plt.close(fig)
    fig, ax = result.plot()
    assert "not a detection curve" in fig.axes[0].get_title()
    assert len(ax.lines) == 2
    plt.close(fig)


def test_lift_grid_does_not_add_fits_and_horizons_are_separate(monkeypatch):
    calls = []
    real_fit = SyntheticControl.fit

    def counting_fit(self, **kwargs):
        calls.append(self.treatment_time)
        return real_fit(self, **kwargs)

    monkeypatch.setattr(SyntheticControl, "fit", counting_fit)
    planner = _planner(_history())
    small = planner.evaluate(durations=[4], relative_lifts=[0.05])
    large = planner.evaluate(
        durations=[3, 4],
        relative_lifts=[0.05, 0.1, 0.2],
        absolute_lifts=[1.0, 2.0],
    )
    assert calls == [
        _history().index[-4],
        _history().index[-3],
        _history().index[-4],
    ]
    assert small.n_inference_fits == 1
    assert large.n_inference_fits == 2
    short, long = large.horizons
    assert short.duration == 3
    assert long.duration == 4
    assert short.holdout_index.tolist() != long.holdout_index[:3].tolist()
    assert set(long.holdout_index).intersection(short.training_index)
    for horizon in large.horizons:
        assert all(
            score.counterfactual is horizon.counterfactual for score in horizon.scores
        )
        zero = next(
            score
            for score in horizon.scores
            if score.lift == 0.0 and score.estimand == "cumulative"
        )
        np.testing.assert_allclose(
            zero.effect,
            horizon.baseline_impact.sum("obs_ind").values.reshape(-1),
        )


def test_duplicate_durations_are_fitted_once(monkeypatch):
    calls = []

    def stop(self, **kwargs):
        calls.append(self.treatment_time)
        raise RuntimeError("counted")

    monkeypatch.setattr(SyntheticControl, "fit", stop)
    result = _planner(_history()).evaluate(
        durations=[4, 4, 3],
        absolute_lifts=[1.0],
    )
    assert calls == [_history().index[-4], _history().index[-3]]
    assert [horizon.duration for horizon in result.horizons] == [4, 3]
    assert result.n_inference_fits == 2
    assert result.status == "fit-failed"


def test_held_out_treated_values_do_not_change_the_fit():
    base = _history()
    perturbed = _history()
    perturbed.loc[perturbed.index[-4:], UNIT] += 50.0
    base_result = _planner(base).evaluate(durations=[4], absolute_lifts=[1.0])
    perturbed_result = _planner(perturbed).evaluate(durations=[4], absolute_lifts=[1.0])
    np.testing.assert_allclose(
        base_result.horizons[0].counterfactual.values,
        perturbed_result.horizons[0].counterfactual.values,
        atol=INDEPENDENT_FIT_ATOL,
        rtol=INDEPENDENT_FIT_RTOL,
    )
    np.testing.assert_allclose(
        base_result.horizons[0].observation_noise_rate,
        perturbed_result.horizons[0].observation_noise_rate,
    )
    assert not np.allclose(
        base_result.horizons[0].observed_holdout.values,
        perturbed_result.horizons[0].observed_holdout.values,
    )


def test_custom_priors_survive_copying_and_refits():
    custom = Prior(
        "Normal",
        sigma=Prior("HalfNormal", sigma=3.0, dims=["treated_units"]),
        dims=["obs_ind", "treated_units"],
    )
    template = WeightedSumFitter(
        priors={"y_hat": custom},
        sample_kwargs=dict(SAMPLE_KWARGS),
    )
    planner = _planner(_history(), model=template)
    first = planner.evaluate(durations=[4], absolute_lifts=[1.0])
    second = planner.evaluate(durations=[4], absolute_lifts=[1.0, 4.0])
    assert first.status == "historical-descriptive"
    assert second.status == "historical-descriptive"
    assert template.idata is None
    assert template._user_priors["y_hat"] is custom
    for result in (first, second):
        horizon = result.horizons[0]
        assert horizon.y_hat_distribution == "Normal"
        assert horizon.user_priors["y_hat"] is custom
        assert horizon.observation_noise_rate is None
        assert horizon.user_priors["y_hat"].parameters["sigma"].distribution == (
            "HalfNormal"
        )
    assert second.n_inference_fits == 1


def _install_posterior(experiment, *, impact: float) -> None:
    holdout = experiment.datapost.index
    coords = {"obs_ind": holdout}
    values = np.full((1, 2, len(holdout)), impact)
    experiment._result = SimpleNamespace(
        predictions_post=xr.DataArray(
            np.zeros_like(values),
            dims=("chain", "draw", "obs_ind"),
            coords=coords,
        ),
        impact_post=xr.DataArray(
            values, dims=("chain", "draw", "obs_ind"), coords=coords
        ),
    )


def _result_shell(scores: tuple = ()) -> PlannerResult:
    horizon = HorizonEvaluation(
        duration=1,
        treatment_time=0,
        training_index=pd.Index([0]),
        holdout_index=pd.Index([1]),
        status="historical-descriptive",
        fit_error=None,
        n_inference_fits=1,
        counterfactual=None,
        baseline_impact=None,
        observed_holdout=None,
        applied_relative_floor=None,
        observation_noise_rate=None,
        y_hat_distribution=None,
        user_priors={},
        scores=scores,
    )
    return PlannerResult(
        status="historical-descriptive",
        fit_error=None,
        decision=DECISION,
        estimator="synthetic_control",
        treated_unit=UNIT,
        control_units=tuple(CONTROLS),
        relative_floor=None,
        relative_baseline_convention="observed-holdout",
        horizon_policy="separate-fits",
        random_seed=None,
        n_inference_fits=1,
        horizons=(horizon,),
    )


def test_rejected_inputs_do_not_fit(monkeypatch):
    def reject_fit(*args, **kwargs):
        raise AssertionError("rejected input reached a fit")

    monkeypatch.setattr(SyntheticControl, "fit", reject_fit)
    history = _history()
    with pytest.raises(TypeError, match="real number"):
        DecisionRule("positive", "cumulative", True, 0.95)
    with pytest.raises(ValueError, match="finite"):
        DecisionRule("positive", "cumulative", np.nan, 0.95)
    with pytest.raises(ValueError, match="Unsupported decision direction"):
        DecisionRule("up", "cumulative", 0.0, 0.95)
    with pytest.raises(ValueError, match="Unsupported estimand"):
        DecisionRule("positive", "total", 0.0, 0.95)
    with pytest.raises(ValueError, match="probability must be strictly above 0"):
        DecisionRule("positive", "cumulative", 0.0, 0.0)
    with pytest.raises(TypeError, match="decision must be a DecisionRule"):
        SyntheticControlPlanner(history, UNIT, CONTROLS, "positive")
    with pytest.raises(TypeError, match="auto_scale_sigma must be a bool"):
        _planner(history, auto_scale_sigma=1)
    with pytest.raises(ValueError, match="relative_floor must be at least 0"):
        _planner(history, relative_floor=-1.0)
    with pytest.raises(TypeError, match="random_seed must be an int"):
        _planner(history, random_seed=True)
    with pytest.raises(ValueError, match="min_donor_correlation must be between"):
        _planner(history, min_donor_correlation=1.5)
    with pytest.raises(ValueError, match="at least one donor"):
        SyntheticControlPlanner(history, UNIT, [], DECISION)
    with pytest.raises(TypeError, match="unit-name strings"):
        SyntheticControlPlanner(history, UNIT, [1], DECISION)
    with pytest.raises(ValueError, match="duplicate donor"):
        SyntheticControlPlanner(history, UNIT, [CONTROLS[0], CONTROLS[0]], DECISION)
    with pytest.raises(ValueError, match="cannot also be a control"):
        SyntheticControlPlanner(history, UNIT, [UNIT, *CONTROLS], DECISION)
    missing = history.drop(columns=CONTROLS[0])
    with pytest.raises(ValueError, match="missing columns"):
        _planner(missing)
    duplicated = history.copy()
    labels = list(duplicated.index)
    labels[1] = labels[0]
    duplicated.index = labels
    with pytest.raises(ValueError, match="index must be unique"):
        _planner(duplicated)
    reversed_history = history.copy()
    reversed_history.index = list(reversed(history.index))
    with pytest.raises(ValueError, match="sorted ascending"):
        _planner(reversed_history)
    with pytest.raises(ValueError, match="at least three rows"):
        _planner(history.iloc[:2])
    non_finite = history.copy()
    non_finite.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _planner(non_finite)
    planner = _planner(history)
    with pytest.raises(KeyError, match="not in the stored history"):
        planner.screen(periods=["not-a-period"])
    with pytest.raises(ValueError, match="at least two observations"):
        planner.screen(periods=history.index[:1])
    with pytest.raises(TypeError, match="sequence of positive integers"):
        planner.evaluate(durations="4", absolute_lifts=[1.0])
    with pytest.raises(ValueError, match="at least one duration"):
        planner.evaluate(durations=[], absolute_lifts=[1.0])
    with pytest.raises(TypeError, match="positive integer"):
        planner.evaluate(durations=[1.5], absolute_lifts=[1.0])
    with pytest.raises(ValueError, match="at least 1"):
        planner.evaluate(durations=[0], absolute_lifts=[1.0])
    with pytest.raises(ValueError, match="training rows"):
        planner.evaluate(durations=[len(history) - 1], absolute_lifts=[1.0])
    with pytest.raises(ValueError, match="at least one absolute or relative"):
        planner.evaluate(durations=[1])
    with pytest.raises(TypeError, match="sequence of finite lifts"):
        planner.evaluate(durations=[1], relative_lifts="0.1")
    with pytest.raises(ValueError, match="is empty"):
        planner.evaluate(durations=[1], absolute_lifts=[])


def test_unfitted_template_records_the_seed_without_sampling(monkeypatch):
    def reject_fit(*args, **kwargs):
        raise AssertionError("template construction sampled")

    monkeypatch.setattr(SyntheticControl, "fit", reject_fit)
    planner = SyntheticControlPlanner(
        _history(),
        UNIT,
        CONTROLS,
        DECISION,
        random_seed=3,
    )
    assert isinstance(planner.model, WeightedSumFitter)
    assert planner.model.idata is None
    assert planner.model.sample_kwargs["random_seed"] == 3
    assert planner.model.sample_kwargs["progressbar"] is False
    unseeded = SyntheticControlPlanner(_history(), UNIT, CONTROLS, DECISION)
    assert unseeded.model.idata is None
    assert "random_seed" not in unseeded.model.sample_kwargs


def test_screen_accepts_a_stored_period_subset():
    history = _history()
    screen = _planner(history).screen(periods=history.index[:4])
    assert screen.periods.equals(history.index[:4])
    assert set(screen.correlation) == set(CONTROLS)


def test_sampling_overrides_reach_the_cloned_fit(monkeypatch):
    seen = []

    def record(self, **kwargs):
        seen.append(dict(self.model.sample_kwargs))
        raise RuntimeError("stop after recording the fit arguments")

    monkeypatch.setattr(SyntheticControl, "fit", record)
    result = _planner(_history(), random_seed=11).evaluate(
        durations=[1],
        absolute_lifts=[1.0],
        sample_kwargs={"tune": 3},
    )
    assert result.status == "fit-failed"
    assert seen[0]["random_seed"] == 11
    assert seen[0]["tune"] == 3


def test_a_fit_that_drifts_the_requested_window_is_refused(monkeypatch):
    def drift_training(self, **kwargs):
        self.data = self.data.iloc[1:].copy()

    monkeypatch.setattr(SyntheticControl, "fit", drift_training)
    with pytest.raises(RuntimeError, match="trained on a different window"):
        _planner(_history()).evaluate(durations=[1], absolute_lifts=[1.0])

    def drift_holdout(self, **kwargs):
        extra = self.data.iloc[[-1]].copy()
        extra.index = [int(self.data.index[-1]) + 1]
        self.data = pd.concat([self.data, extra])

    monkeypatch.setattr(SyntheticControl, "fit", drift_holdout)
    with pytest.raises(RuntimeError, match="held out a different window"):
        _planner(_history()).evaluate(durations=[1], absolute_lifts=[1.0])


def test_a_swapped_fit_model_is_refused(monkeypatch):
    def swap(self, **kwargs):
        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression()
        _install_posterior(self, impact=4.0)

    monkeypatch.setattr(SyntheticControl, "fit", swap)
    with pytest.raises(TypeError, match="remain a WeightedSumFitter"):
        _planner(_history()).evaluate(durations=[1], absolute_lifts=[1.0])


def test_negative_decision_uses_the_lower_tail():
    baseline = xr.DataArray(
        np.full((1, 1, 2), -1.0),
        dims=("chain", "draw", "obs_ind"),
        coords={"obs_ind": [0, 1]},
    )
    observed = xr.DataArray(np.ones(2), dims="obs_ind", coords={"obs_ind": [0, 1]})
    scores = _score_translated_lifts(
        baseline_impact=baseline,
        observed=observed,
        counterfactual=xr.zeros_like(baseline),
        relative_lifts=None,
        absolute_lifts=(0.0,),
        decision=DecisionRule("negative", "cumulative", 0.0, 0.9),
    )
    cumulative = next(score for score in scores if score.estimand == "cumulative")
    assert cumulative.detected is True
    assert cumulative.tail_probability == pytest.approx(1.0)


def test_translated_result_reports_detection_plot_and_mixed_status(monkeypatch):
    def fit(self, **kwargs):
        if self.treatment_time == _history().index[-1]:
            raise RuntimeError("sampling failed")
        _install_posterior(self, impact=4.0)
        self.model.priors["y_hat"].parameters["sigma"] = 1.0

    monkeypatch.setattr(SyntheticControl, "fit", fit)
    result = _planner(_history()).evaluate(
        durations=[4, 1],
        absolute_lifts=[1.0, 1.0],
    )
    assert result.status == "mixed"
    assert result.n_inference_fits == 2
    assert "No single recommendation" in result.recommendation()
    failed = next(horizon for horizon in result.horizons if horizon.duration == 1)
    assert failed.status == "fit-failed"
    scored = next(horizon for horizon in result.horizons if horizon.duration == 4)
    assert scored.observation_noise_rate is None
    absolute = [
        score.lift
        for score in scored.scores
        if score.effect_scale == "absolute" and score.estimand == "cumulative"
    ]
    assert absolute == [0.0, 1.0]
    assert scored.scores[0].detected is True
    fig, ax = result.plot(horizon=4)
    assert ax.get_xlabel() == "Injected absolute lift"
    assert "decision rule met" in [
        text.get_text() for text in ax.get_legend().get_texts()
    ]
    plt.close(fig)
    with pytest.raises(ValueError, match="effect_scale must be"):
        result.plot(horizon=4, effect_scale="nope")
    with pytest.raises(ValueError, match="estimand must be"):
        result.plot(horizon=4, estimand="nope")
    with pytest.raises(ValueError, match="No relative"):
        result.plot(horizon=4, effect_scale="relative")
    with pytest.raises(KeyError, match="No horizon with duration"):
        result.plot(horizon=99)
    success = PlannerResult(
        status="historical-descriptive",
        fit_error=None,
        decision=result.decision,
        estimator=result.estimator,
        treated_unit=result.treated_unit,
        control_units=result.control_units,
        relative_floor=None,
        relative_baseline_convention=result.relative_baseline_convention,
        horizon_policy=result.horizon_policy,
        random_seed=None,
        n_inference_fits=1,
        horizons=(scored,),
    )
    assert success.recommendation().startswith("Historical-descriptive")
    with pytest.raises(ValueError, match="no scores to plot"):
        _result_shell().plot()
