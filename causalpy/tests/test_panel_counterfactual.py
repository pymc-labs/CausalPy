#   Copyright 2025 - 2026 The PyMC Labs Developers
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
"""Contracts for the private wide-panel counterfactual helper."""

import inspect

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt

import causalpy as cp
from causalpy.experiments._panel_counterfactual import (
    counterfactual_impacts,
    panel_effect_summary,
    panel_period_frames,
    panel_plot_frame,
    plot_panel_counterfactual,
    plot_panel_prior_check,
    wide_panel_design,
)
from causalpy.experiments._results import CausalResult

CONTROL_UNITS = ["a", "b", "c", "d", "e", "f", "g"]


def _draws(values: np.ndarray, obs: pd.Index, unit: str = "actual") -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=["chain", "draw", "obs_ind", "treated_units"],
        coords={"obs_ind": obs, "treated_units": [unit]},
    )


def test_treatment_time_is_post_period():
    """The shared split keeps the treatment time out of the pre-period."""
    frame = pd.DataFrame({"y": [0, 1, 2]}, index=[69, 70, 71])
    pre, post = panel_period_frames(frame, 70)
    assert list(pre.index) == [69]
    assert list(post.index) == [70, 71]


def test_wide_panel_design_names_control_and_treated_axes():
    """Control columns are coeffs; treated columns keep the treated_units dim."""
    frame = pd.DataFrame({"donor": [1.0, 2.0], "treated": [3.0, 4.0]}, index=[0, 1])
    design = wide_panel_design(frame, ["donor"], ["treated"])
    assert design["control"].dims == ("obs_ind", "coeffs")
    assert design["treated"].dims == ("obs_ind", "treated_units")
    assert list(design["control"].coords["coeffs"].values) == ["donor"]
    np.testing.assert_allclose(design["treated"].values, [[3.0], [4.0]])


def test_impact_is_observed_minus_counterfactual_on_aligned_time():
    """Impact subtracts in place and cumulative impact is the post-period running sum."""
    pre_index = pd.Index([0, 1], name="obs_ind")
    post_index = pd.Index([2, 3], name="obs_ind")
    treated_pre = _draws(np.array([[[[1.0], [3.0]]]]), pre_index)
    predicted_pre = _draws(np.array([[[[0.5], [1.0]]]]), pre_index)
    treated_post = _draws(np.array([[[[4.0], [7.0]]]]), post_index)
    predicted_post = _draws(np.array([[[[1.0], [2.0]]]]), post_index)

    impact_pre, impact_post, cumulative = counterfactual_impacts(
        treated_pre, predicted_pre, treated_post, predicted_post
    )

    np.testing.assert_allclose(impact_pre.values, [[[[0.5], [2.0]]]])
    np.testing.assert_allclose(impact_post.values, [[[[3.0], [5.0]]]])
    np.testing.assert_allclose(cumulative.values, impact_post.cumsum("obs_ind").values)


def test_misaligned_prediction_time_is_rejected():
    """A mismatched time index is named, for both the pre-period and post-period pair."""
    treated = _draws(np.ones((1, 1, 2, 1)), pd.Index([0, 1]))
    predicted = _draws(np.ones((1, 1, 2, 1)), pd.Index([0, 2]))
    with pytest.raises(
        ValueError,
        match=r"pre-period obs_ind mismatch: treated=\[0, 1\], predictions=\[0, 2\]",
    ):
        counterfactual_impacts(treated, predicted, treated, treated)
    with pytest.raises(
        ValueError,
        match=r"post-period obs_ind mismatch: treated=\[0, 1\], predictions=\[0, 2\]",
    ):
        counterfactual_impacts(treated, treated, treated, predicted)


def test_plot_frame_omits_hdi_for_a_point_estimate():
    """A singleton chain/draw is a point estimate, so the frame has no HDI columns."""
    index = pd.Index([0, 1])
    observed = pd.DataFrame({"actual": [1.0, 2.0]}, index=index)
    prediction = _draws(np.array([[[[1.5], [2.5]]]]), index)
    impact = _draws(np.array([[[[-0.5], [-0.5]]]]), index)
    bundle = CausalResult(
        predictions_pre=prediction,
        predictions_post=prediction,
        impact_pre=impact,
        impact_post=impact,
        impact_post_cumulative=impact,
    )
    frame = panel_plot_frame(observed, observed, bundle, ["actual"])
    assert set(frame.columns) == {"actual", "prediction", "impact"}
    np.testing.assert_allclose(frame["prediction"], [1.5, 2.5, 1.5, 2.5])


def test_plot_frame_names_hdi_columns_from_the_requested_probability():
    """Draws produce HDI columns whose names follow the requested probability."""
    index = pd.Index([0])
    observed = pd.DataFrame({"actual": [0.0]}, index=index)
    prediction = _draws(np.arange(8, dtype=float).reshape(1, 8, 1, 1), index)
    bundle = CausalResult(
        predictions_pre=prediction,
        predictions_post=prediction,
        impact_pre=prediction,
        impact_post=prediction,
        impact_post_cumulative=prediction,
    )
    frame = panel_plot_frame(observed, observed, bundle, ["actual"], hdi_prob=0.5)
    assert {"pred_hdi_lower_50", "pred_hdi_upper_50"} <= set(frame.columns)
    assert "pred_hdi_lower_94" not in frame.columns


def test_unknown_treated_unit_names_the_available_units():
    """The plot-data lookup uses the same missing-unit error as SyntheticControl."""
    index = pd.Index([0])
    observed = pd.DataFrame({"actual": [0.0]}, index=index)
    prediction = _draws(np.ones((1, 1, 1, 1)), index)
    bundle = CausalResult(
        predictions_pre=prediction,
        predictions_post=prediction,
        impact_pre=prediction,
        impact_post=prediction,
        impact_post_cumulative=prediction,
    )
    with pytest.raises(ValueError, match="Available units: \\['actual'\\]"):
        panel_plot_frame(observed, observed, bundle, ["actual"], treated_unit="missing")


def test_fit_inputs_exclude_treated_post_outcomes(sc_data):
    """Construction may store post-period treated outcomes, but fit does not receive them."""
    experiment = cp.SyntheticControl(
        sc_data,
        70,
        control_units=CONTROL_UNITS,
        treated_units=["actual"],
        model=cp.skl_models.WeightedProportion(),
    )
    predictors, outcome, _coords = experiment._fit_inputs()
    assert predictors.obs_ind.max() < 70
    assert outcome.obs_ind.max() < 70
    assert experiment.post_design["treated"].obs_ind.min() >= 70


def test_fitted_impact_and_plot_data_follow_the_helper(sc_data):
    """An OLS synthetic control still reports observed minus counterfactual."""
    experiment = cp.SyntheticControl(
        sc_data,
        70,
        control_units=CONTROL_UNITS,
        treated_units=["actual"],
        model=cp.skl_models.WeightedProportion(),
    ).fit()
    impact = experiment.result.impact_post.sel(treated_units="actual")
    expected = experiment.post_design["treated"].sel(
        treated_units="actual"
    ) - experiment.result.predictions_post.sel(treated_units="actual")
    xr.testing.assert_allclose(impact, expected.transpose(*impact.dims))
    xr.testing.assert_allclose(
        experiment.result.impact_post_cumulative,
        experiment.result.impact_post.cumsum("obs_ind"),
    )
    plot_data = experiment.get_plot_data()
    np.testing.assert_allclose(
        plot_data.loc[experiment.datapost.index, "impact"],
        impact.mean(["chain", "draw"]).values,
    )
    figure, axes = experiment.plot(plot_predictors=True)
    assert isinstance(figure, plt.Figure)
    assert len(axes) == 3
    plt.close(figure)


def test_effect_summary_does_not_default_to_synthetic_control_assumptions(sc_data):
    """Synthetic control must opt into its assumptions text; the helper has no ``sc`` default."""
    parameter = inspect.signature(panel_effect_summary).parameters["experiment_type"]
    assert parameter.default is inspect.Parameter.empty
    experiment = cp.SyntheticControl(
        sc_data,
        70,
        control_units=CONTROL_UNITS,
        treated_units=["actual"],
        model=cp.skl_models.WeightedProportion(),
    ).fit()
    summary = experiment.effect_summary()
    assert (
        "control units used to construct the synthetic counterfactual" in summary.text
    )


def test_point_estimate_observations_use_the_supplied_index():
    """A point-estimate figure uses the index arguments, not the treated ``obs_ind`` coordinate."""
    pre_index = pd.Index([0, 1, 2])
    post_index = pd.Index([3, 4])
    pre_treated = xr.DataArray(
        [1.0, 1.0, 1.0], dims=["obs_ind"], coords={"obs_ind": [10, 11, 12]}
    )
    post_treated = xr.DataArray(
        [1.0, 1.0], dims=["obs_ind"], coords={"obs_ind": [13, 14]}
    )
    pre_pred = _draws(np.zeros((1, 1, 3, 1)), pre_index).sel(treated_units="actual")
    post_pred = _draws(np.zeros((1, 1, 2, 1)), post_index).sel(treated_units="actual")
    control = xr.DataArray(
        np.zeros((3, 1)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": pre_index, "coeffs": ["donor"]},
    )
    post_control = xr.DataArray(
        np.zeros((2, 1)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": post_index, "coeffs": ["donor"]},
    )
    _figure, axes = plot_panel_counterfactual(
        pre_index=pre_index,
        post_index=post_index,
        pre_pred=pre_pred,
        post_pred=post_pred,
        pre_impact=pre_pred,
        post_impact=post_pred,
        post_impact_cumulative=post_pred,
        pre_treated=pre_treated,
        post_treated=post_treated,
        pre_control=control,
        post_control=post_control,
        treatment_time=3,
        title="point estimate",
        style={"ci_prob": 0.94, "kind": "ribbon", "ci_kind": "hdi", "num_samples": 2},
        figsize=(4, 6),
        plot_predictors=False,
    )
    observation_x = [
        list(line.get_xdata())
        for line in axes[0].get_lines()
        if line.get_marker() == "."
    ]
    assert observation_x == [[0, 1, 2], [3, 4]]
    plt.close(_figure)


def test_prior_check_figure_is_one_panel():
    """The shared prior figure drops the impact panels."""
    index = pd.Index([0, 1])
    prediction = _draws(np.zeros((1, 2, 2, 1)), index)
    treated = prediction.sel(treated_units="actual")
    control = xr.DataArray(
        np.zeros((2, 1)),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": index, "coeffs": ["donor"]},
    )
    figure, axes = plot_panel_prior_check(
        pre_index=index,
        post_index=index,
        pre_pred=treated,
        post_pred=treated,
        pre_treated=treated.isel(chain=0, draw=0),
        post_treated=treated.isel(chain=0, draw=0),
        pre_control=control,
        post_control=control,
        treatment_time=1,
        style={
            "ci_prob": 0.94,
            "kind": "ribbon",
            "ci_kind": "hdi",
            "num_samples": 2,
        },
        figsize=(4, 3),
        plot_predictors=False,
    )
    assert axes[0].get_title() == "Prior predictive check"
    assert len(axes) == 1
    plt.close(figure)
