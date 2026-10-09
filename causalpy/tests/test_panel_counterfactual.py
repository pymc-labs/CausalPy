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
"""Contracts for the private wide-panel counterfactual seam."""

import inspect

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt

import causalpy as cp
import causalpy.experiments._panel_counterfactual as panel_helper
from causalpy.experiments._panel_counterfactual import WidePanel
from causalpy.experiments._results import CausalResult

CONTROL_UNITS = ["a", "b", "c", "d", "e", "f", "g"]
_STYLE = {"ci_prob": 0.94, "kind": "ribbon", "ci_kind": "hdi", "num_samples": 2}


def _draws(values: np.ndarray, obs: pd.Index, unit: str = "actual") -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=["chain", "draw", "obs_ind", "treated_units"],
        coords={"obs_ind": obs, "treated_units": [unit]},
    )


def _panel(
    pre: pd.DataFrame,
    post: pd.DataFrame,
    treatment_time: int | float | pd.Timestamp,
    *,
    control_units: list[str] | None = None,
    treated_units: list[str] | None = None,
) -> WidePanel:
    donors = control_units if control_units is not None else ["donor"]
    treated = treated_units if treated_units is not None else ["actual"]
    return WidePanel(
        treatment_time,
        tuple(donors),
        tuple(treated),
        pre,
        post,
    )


def test_treatment_time_is_post_period():
    """The shared split keeps the treatment time out of the pre-period."""
    panel = WidePanel.from_frame(
        pd.DataFrame({"y": [0, 1, 2]}, index=[69, 70, 71]),
        70,
        [],
        ["y"],
    )
    assert list(panel.pre.index) == [69]
    assert list(panel.post.index) == [70, 71]


def test_call_surface_is_the_panel_not_exploded_helpers():
    """New experiments hold a panel. They do not import the old free functions."""
    assert not hasattr(panel_helper, "plot_panel_counterfactual")
    assert not hasattr(panel_helper, "wide_panel_design")
    assert not hasattr(panel_helper, "counterfactual_impacts")
    panel = WidePanel.from_frame(
        pd.DataFrame({"donor": [1.0], "treated": [2.0]}, index=[0]),
        0,
        ["donor"],
        ["treated"],
    )
    assert not hasattr(panel, "treated_post")
    assert not hasattr(panel, "treated")
    assert not hasattr(panel, "fit_inputs")


def test_control_axis_is_obs_ind_by_control_units():
    """The shared donor axis is control_units, not the fitter's coeffs name."""
    panel = WidePanel.from_frame(
        pd.DataFrame({"donor": [1.0, 2.0], "treated": [3.0, 4.0]}, index=[0, 1]),
        2,
        ["donor"],
        ["treated"],
    )
    control = panel.control("pre")
    assert control.dims == ("obs_ind", "control_units")
    assert list(control.coords["control_units"].values) == ["donor"]
    assert panel.control("post").dims == ("obs_ind", "control_units")
    assert panel.treated_pre.dims == ("obs_ind", "treated_units")
    np.testing.assert_allclose(panel.treated_pre.values, [[3.0], [4.0]])


def test_synthetic_control_renames_control_to_coeffs_and_stores_the_split(sc_data):
    """The fitter boundary keeps coeffs. Period frames are not re-split."""
    experiment = cp.SyntheticControl(
        sc_data,
        70,
        control_units=CONTROL_UNITS,
        treated_units=["actual"],
        model=cp.skl_models.WeightedProportion(),
    )
    assert experiment.pre_design["control"].dims == ("obs_ind", "coeffs")
    assert list(experiment.pre_design["control"].coords["coeffs"].values) == (
        CONTROL_UNITS
    )
    assert experiment.pre_design["treated"].dims == ("obs_ind", "treated_units")
    pd.testing.assert_frame_equal(experiment.datapre, experiment._panel.pre)
    pd.testing.assert_frame_equal(experiment.datapost, experiment._panel.post)
    assert experiment.datapre is not experiment._panel.pre
    assert experiment.datapost is not experiment._panel.post


def test_mutating_datapost_does_not_change_impact(sc_data):
    """Caller writes into the returned period frames leave impact alone."""
    experiment = cp.SyntheticControl(
        sc_data,
        70,
        control_units=CONTROL_UNITS,
        treated_units=["actual"],
        model=cp.skl_models.WeightedProportion(),
    )
    experiment.fit()
    impact_before = experiment.result.impact_post.values.copy()
    treated_before = experiment.post_design["treated"].values.copy()
    returned_post = experiment.datapost
    returned_pre = experiment.datapre
    returned_post.loc[:, "actual"] = returned_post["actual"] + 1000
    returned_pre.iloc[:, 0] = returned_pre.iloc[:, 0] + 1000
    experiment.fit()
    np.testing.assert_allclose(experiment.result.impact_post.values, impact_before)
    np.testing.assert_allclose(experiment.post_design["treated"].values, treated_before)


def test_impact_is_observed_minus_counterfactual_on_aligned_time():
    """Impact subtracts in place and cumulative impact is the post-period running sum."""
    pre_index = pd.Index([0, 1], name="obs_ind")
    post_index = pd.Index([2, 3], name="obs_ind")
    panel = _panel(
        pd.DataFrame({"donor": [0.0, 0.0], "actual": [1.0, 3.0]}, index=pre_index),
        pd.DataFrame({"donor": [0.0, 0.0], "actual": [4.0, 7.0]}, index=post_index),
        2,
    )
    impact_pre, impact_post, cumulative = panel.impacts(
        _draws(np.array([[[[0.5], [1.0]]]]), pre_index),
        _draws(np.array([[[[1.0], [2.0]]]]), post_index),
    )

    np.testing.assert_allclose(impact_pre.values, [[[[0.5], [2.0]]]])
    np.testing.assert_allclose(impact_post.values, [[[[3.0], [5.0]]]])
    np.testing.assert_allclose(cumulative.values, impact_post.cumsum("obs_ind").values)


def test_misaligned_prediction_time_is_rejected():
    """A mismatched time index is named, for both the pre-period and post-period pair."""
    pre = pd.DataFrame({"donor": [0.0, 0.0], "actual": [1.0, 1.0]}, index=[0, 1])
    post = pd.DataFrame({"donor": [0.0, 0.0], "actual": [1.0, 1.0]}, index=[2, 3])
    panel = _panel(pre, post, 2)
    aligned_pre = _draws(np.ones((1, 1, 2, 1)), pd.Index([0, 1]))
    aligned_post = _draws(np.ones((1, 1, 2, 1)), pd.Index([2, 3]))
    misaligned = _draws(np.ones((1, 1, 2, 1)), pd.Index([0, 2]))
    with pytest.raises(
        ValueError,
        match=r"pre-period obs_ind mismatch: treated=\[0, 1\], predictions=\[0, 2\]",
    ):
        panel.impacts(misaligned, aligned_post)
    with pytest.raises(
        ValueError,
        match=r"post-period obs_ind mismatch: treated=\[2, 3\], predictions=\[0, 2\]",
    ):
        panel.impacts(aligned_pre, misaligned)


def test_plot_frame_keeps_donor_columns_and_omits_hdi_for_a_point_estimate():
    """A singleton chain/draw is a point estimate, so the frame has no HDI columns."""
    index = pd.Index([0, 1])
    observed = pd.DataFrame({"donor": [0.0, 0.0], "actual": [1.0, 2.0]}, index=index)
    prediction = _draws(np.array([[[[1.5], [2.5]]]]), index)
    impact = _draws(np.array([[[[-0.5], [-0.5]]]]), index)
    bundle = CausalResult(
        predictions_pre=prediction,
        predictions_post=prediction,
        impact_pre=impact,
        impact_post=impact,
        impact_post_cumulative=impact,
    )
    frame = _panel(observed, observed, 0).plot_data(bundle)
    assert set(frame.columns) == {"donor", "actual", "prediction", "impact"}
    np.testing.assert_allclose(frame["prediction"], [1.5, 2.5, 1.5, 2.5])


def test_plot_frame_names_hdi_columns_from_the_requested_probability():
    """Draws produce HDI columns whose names follow the requested probability."""
    index = pd.Index([0])
    observed = pd.DataFrame({"donor": [0.0], "actual": [0.0]}, index=index)
    prediction = _draws(np.arange(8, dtype=float).reshape(1, 8, 1, 1), index)
    bundle = CausalResult(
        predictions_pre=prediction,
        predictions_post=prediction,
        impact_pre=prediction,
        impact_post=prediction,
        impact_post_cumulative=prediction,
    )
    frame = _panel(observed, observed, 0).plot_data(bundle, hdi_prob=0.5)
    assert {"pred_hdi_lower_50", "pred_hdi_upper_50"} <= set(frame.columns)
    assert "pred_hdi_lower_94" not in frame.columns


def test_unknown_treated_unit_names_the_available_units():
    """Plot data and the effect summary share the missing-unit error."""
    index = pd.Index([0])
    observed = pd.DataFrame({"donor": [0.0], "actual": [0.0]}, index=index)
    prediction = _draws(np.ones((1, 1, 1, 1)), index)
    bundle = CausalResult(
        predictions_pre=prediction,
        predictions_post=prediction,
        impact_pre=prediction,
        impact_post=prediction,
        impact_post_cumulative=prediction,
    )
    panel = _panel(observed, observed, 0)
    with pytest.raises(ValueError, match="Available units: \\['actual'\\]"):
        panel.plot_data(bundle, treated_unit="missing")
    with pytest.raises(ValueError, match="Available units: \\['actual'\\]"):
        panel.effect_summary(
            bundle,
            group="posterior",
            experiment_type="other",
            treated_unit="missing",
        )


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
    assert set(CONTROL_UNITS) <= set(plot_data.columns)
    figure, axes = experiment.plot(plot_predictors=True)
    assert isinstance(figure, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert len(axes) == 3
    plt.close(figure)


def test_effect_summary_does_not_default_to_synthetic_control_assumptions(sc_data):
    """Synthetic control must opt into its assumptions text; the helper has no ``sc`` default."""
    parameter = inspect.signature(WidePanel.effect_summary).parameters[
        "experiment_type"
    ]
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


def _plot_kwargs() -> dict:
    return {
        "group": "posterior",
        "treated_unit": None,
        "title": "point estimate",
        "style": _STYLE,
        "figsize": (4, 6),
        "plot_predictors": False,
    }


def test_plot_and_plot_data_reject_a_misaligned_time_index():
    """A permuted coordinate fails in the figure path, the summary, and names the bad field."""
    pre = pd.DataFrame(
        {"donor": [0.0, 0.0, 0.0], "actual": [1.0, 2.0, 3.0]}, index=[0, 1, 2]
    )
    post = pd.DataFrame({"donor": [0.0, 0.0], "actual": [4.0, 5.0]}, index=[3, 4])
    panel = _panel(pre, post, 3)
    aligned_pre = _draws(np.ones((1, 1, 3, 1)), pd.Index([0, 1, 2]))
    aligned_post = _draws(np.ones((1, 1, 2, 1)), pd.Index([3, 4]))
    reversed_pre = _draws(np.array([[[[30.0], [20.0], [10.0]]]]), pd.Index([2, 1, 0]))
    reversed_post = _draws(np.array([[[[50.0], [40.0]]]]), pd.Index([4, 3]))
    shifted = CausalResult(
        predictions_pre=_draws(np.zeros((1, 1, 3, 1)), pd.Index([10, 11, 12])),
        predictions_post=_draws(np.zeros((1, 1, 2, 1)), pd.Index([13, 14])),
        impact_pre=aligned_pre,
        impact_post=aligned_post,
        impact_post_cumulative=aligned_post,
    )
    cases = (
        (
            CausalResult(
                predictions_pre=reversed_pre,
                predictions_post=reversed_post,
                impact_pre=aligned_pre,
                impact_post=aligned_post,
                impact_post_cumulative=aligned_post,
            ),
            r"pre-period obs_ind mismatch: treated=\[0, 1, 2\], predictions=\[2, 1, 0\]",
        ),
        (
            shifted,
            r"pre-period obs_ind mismatch: treated=\[0, 1, 2\], predictions=\[10, 11, 12\]",
        ),
        (
            CausalResult(
                predictions_pre=aligned_pre,
                predictions_post=aligned_post,
                impact_pre=aligned_pre,
                impact_post=reversed_post,
                impact_post_cumulative=aligned_post,
            ),
            r"post-period obs_ind mismatch: treated=\[3, 4\], impact=\[4, 3\]",
        ),
        (
            CausalResult(
                predictions_pre=aligned_pre,
                predictions_post=aligned_post,
                impact_pre=aligned_pre,
                impact_post=aligned_post,
                impact_post_cumulative=reversed_post,
            ),
            r"post-period obs_ind mismatch: treated=\[3, 4\], cumulative impact=\[4, 3\]",
        ),
    )
    for bundle, message in cases:
        with pytest.raises(ValueError, match=message):
            panel.plot_data(bundle)
        with pytest.raises(ValueError, match=message):
            panel.plot(bundle, **_plot_kwargs())
        with pytest.raises(ValueError, match=message):
            panel.effect_summary(bundle, group="posterior", experiment_type="other")


def test_prediction_without_treated_units_is_rejected():
    """A shared prediction is not broadcast onto every treated unit."""
    pre = pd.DataFrame(
        {"donor": [0.0, 0.0], "t1": [1.0, 2.0], "t2": [3.0, 4.0]}, index=[0, 1]
    )
    post = pd.DataFrame(
        {"donor": [0.0, 0.0], "t1": [4.0, 7.0], "t2": [40.0, 70.0]}, index=[2, 3]
    )
    panel = _panel(pre, post, 2, treated_units=["t1", "t2"])
    aligned_pre = xr.DataArray(
        np.zeros((1, 1, 2, 2)),
        dims=["chain", "draw", "obs_ind", "treated_units"],
        coords={"obs_ind": [0, 1], "treated_units": ["t1", "t2"]},
    )
    shared_post = xr.DataArray(
        np.array([[[1.0, 2.0]]]),
        dims=["chain", "draw", "obs_ind"],
        coords={"obs_ind": [2, 3]},
    )
    with pytest.raises(
        ValueError,
        match=(
            r"post-period predictions lack a treated_units dimension. "
            r"Available units: \['t1', 't2'\]"
        ),
    ):
        panel.impacts(aligned_pre, shared_post)
    per_unit_post = xr.DataArray(
        np.array([[[[1.0, 1.0], [2.0, 2.0]]]]),
        dims=["chain", "draw", "obs_ind", "treated_units"],
        coords={"obs_ind": [2, 3], "treated_units": ["t1", "t2"]},
    )
    _, impact_post, _ = panel.impacts(aligned_pre, per_unit_post)
    np.testing.assert_allclose(
        impact_post.sel(treated_units="t1").values, [[[3.0, 5.0]]]
    )


def test_point_estimate_observations_use_the_panel_index():
    """Marker x and the fit line follow the stored index, not array position."""
    pre_index = pd.Index([0, 1, 2])
    post_index = pd.Index([3, 4])
    panel = _panel(
        pd.DataFrame({"donor": 0.0, "actual": 1.0}, index=pre_index),
        pd.DataFrame({"donor": 0.0, "actual": 1.0}, index=post_index),
        3,
    )
    pre_fit = np.array([30.0, 10.0, 20.0])
    post_fit = np.array([50.0, 40.0])
    bundle = CausalResult(
        predictions_pre=_draws(pre_fit.reshape(1, 1, 3, 1), pre_index),
        predictions_post=_draws(post_fit.reshape(1, 1, 2, 1), post_index),
        impact_pre=_draws(np.zeros((1, 1, 3, 1)), pre_index),
        impact_post=_draws(np.zeros((1, 1, 2, 1)), post_index),
        impact_post_cumulative=_draws(np.zeros((1, 1, 2, 1)), post_index),
    )
    frame = panel.plot_data(bundle)
    np.testing.assert_allclose(frame["prediction"], [30.0, 10.0, 20.0, 50.0, 40.0])
    figure, axes = panel.plot(bundle, **_plot_kwargs())
    observation_x = [
        list(line.get_xdata())
        for line in axes[0].get_lines()
        if line.get_marker() == "."
    ]
    fit = next(line for line in axes[0].get_lines() if line.get_label() == "model fit")
    counterfactual = next(
        line for line in axes[0].get_lines() if line.get_label() == "Counterfactual"
    )
    assert observation_x == [[0, 1, 2], [3, 4]]
    np.testing.assert_allclose(fit.get_xdata(), [0, 1, 2])
    np.testing.assert_allclose(fit.get_ydata(), pre_fit)
    np.testing.assert_allclose(counterfactual.get_xdata(), [3, 4])
    np.testing.assert_allclose(counterfactual.get_ydata(), post_fit)
    plt.close(figure)


def test_prior_check_figure_is_one_panel():
    """The shared prior figure drops the impact panels."""
    index = pd.Index([0, 1])
    observed = pd.DataFrame({"donor": [0.0, 0.0], "actual": [1.0, 1.0]}, index=index)
    prediction = _draws(np.zeros((1, 2, 2, 1)), index)
    figure, axes = _panel(observed, observed, 1).plot(
        CausalResult(
            predictions_pre=prediction,
            predictions_post=prediction,
            impact_pre=prediction,
            impact_post=prediction,
            impact_post_cumulative=prediction,
        ),
        group="prior",
        treated_unit=None,
        title="ignored",
        style=_STYLE,
        figsize=(4, 3),
        plot_predictors=False,
    )
    assert axes[0].get_title() == "Prior predictive check"
    assert len(axes) == 1
    assert not isinstance(axes, np.ndarray)
    plt.close(figure)
