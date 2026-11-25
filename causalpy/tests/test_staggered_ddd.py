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
"""Tests for staggered difference-in-differences-in-differences."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.data.simulate_data import generate_staggered_did_data

# Light draws check retrieval against the aggregated estimate, not calibration.
LIGHT_SAMPLE = {
    "tune": 20,
    "draws": 20,
    "chains": 2,
    "cores": 2,
    "progressbar": False,
}


def _panel(
    *,
    effect: float,
    sugar: str,
    unit_offset: int,
    seed: int,
    keep_never_treated: bool = True,
) -> pd.DataFrame:
    frame = generate_staggered_did_data(
        n_units=12,
        n_time_periods=8,
        treatment_cohorts={4: 6},
        treatment_effects={0: effect},
        sigma=0.05,
        seed=seed,
    )
    if not keep_never_treated:
        frame = frame.loc[np.isfinite(frame["treatment_time"])].copy()
    frame["unit"] = frame["unit"] + unit_offset
    frame["sugar"] = sugar
    return frame


def _stacked(**kwargs) -> pd.DataFrame:
    high = _panel(effect=4.0, sugar="high", unit_offset=0, seed=1, **kwargs)
    low = _panel(effect=1.0, sugar="low", unit_offset=100, seed=2, **kwargs)
    return pd.concat([high, low], ignore_index=True)


def _fit_ols(data: pd.DataFrame, **kwargs):
    return cp.StaggeredDifferenceInDifferencesInDifferences(
        data,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        level_variable_name="sugar",
        treatment_time_variable_name="treatment_time",
        model=LinearRegression(),
        **kwargs,
    ).fit()


def test_contrast_draws_use_within_draw_differences():
    difference = cp.StaggeredDifferenceInDifferencesInDifferences.contrast_draws(
        np.array([7.0, 9.0, 11.0]),
        np.array([5.0, 3.0, 1.0]),
    )
    np.testing.assert_allclose(difference, [2.0, 6.0, 10.0])


def test_ols_public_draws_match_identified_estimates():
    result = _fit_ols(_stacked(), reference_level="low")
    for level, experiment in result.experiments.items():
        draws = experiment.att_event_time_draws()
        assert isinstance(draws, pd.Series)
        assert "chain" not in getattr(draws, "dims", ())
        identified = experiment.result.att_event_time
        identified = identified.loc[identified["identified"]]
        np.testing.assert_allclose(
            draws.to_numpy(),
            identified["att"].to_numpy(),
        )
        overall = experiment.att_overall_draws()
        assert isinstance(overall, float)
        post = identified.loc[identified["event_time"] >= 0, "att"]
        assert overall == pytest.approx(float(post.mean()))
        assert level in {"high", "low"}


def test_hand_computed_contrast_matches_public_draws():
    result = _fit_ols(_stacked(), contrasts=[("high", "low")])
    high = result.experiments["high"]
    low = result.experiments["low"]
    shared = sorted(
        set(high.att_event_time_draws().index) & set(low.att_event_time_draws().index)
    )
    expected = (
        high.att_event_time_draws().loc[shared] - low.att_event_time_draws().loc[shared]
    )
    table = result.contrast_att_event_time()
    assert table["level_a"].unique().tolist() == ["high"]
    assert table["level_b"].unique().tolist() == ["low"]
    np.testing.assert_allclose(table["att"].to_numpy(), expected.to_numpy())
    overall = result.contrast_att_overall()
    assert overall["att"].iloc[0] == pytest.approx(
        high.att_overall_draws() - low.att_overall_draws()
    )
    assert "att_lower" not in table.columns


def test_one_sided_unidentified_event_time_emits_no_contrast():
    high = _panel(effect=4.0, sugar="high", unit_offset=0, seed=3)
    low = _panel(
        effect=1.0,
        sugar="low",
        unit_offset=100,
        seed=4,
        keep_never_treated=False,
    )
    result = _fit_ols(pd.concat([high, low], ignore_index=True), reference_level="low")
    high_times = set(result.experiments["high"].att_event_time_draws().index)
    low_times = set(result.experiments["low"].att_event_time_draws().index)
    one_sided = high_times - low_times
    assert one_sided
    assert all(event_time >= 0 for event_time in one_sided)
    contrasted = set(result.contrast_att_event_time()["event_time"])
    assert one_sided.isdisjoint(contrasted)
    assert contrasted == high_times & low_times


def test_reference_level_returns_other_minus_reference_only():
    data = _stacked()
    mid = _panel(effect=2.0, sugar="mid", unit_offset=200, seed=5)
    data = pd.concat([data, mid], ignore_index=True)
    result = _fit_ols(data, reference_level="low")
    table = result.contrast_att_event_time()
    pairs = set(zip(table["level_a"], table["level_b"], strict=True))
    assert pairs == {("high", "low"), ("mid", "low")}
    only = _fit_ols(data, contrasts=[("mid", "high")])
    only_pairs = set(
        zip(
            only.contrast_att_event_time()["level_a"],
            only.contrast_att_event_time()["level_b"],
            strict=True,
        )
    )
    assert only_pairs == {("mid", "high")}


def test_plot_values_match_tables():
    result = _fit_ols(_stacked(), reference_level="low")
    fig, axes = result.plot()
    level_table = result.level_att_event_time()
    level_table = level_table.loc[level_table["identified"]]
    plotted_levels = []
    for line in axes[0].get_lines():
        if line.get_label().startswith("_"):
            continue
        plotted_levels.append(line.get_ydata())
        match = level_table.loc[level_table["level"] == line.get_label(), "att"]
        np.testing.assert_allclose(line.get_ydata(), match.to_numpy())
    assert plotted_levels
    contrast = result.contrast_att_event_time()
    for line in axes[1].get_lines():
        if line.get_label().startswith("_"):
            continue
        np.testing.assert_allclose(line.get_ydata(), contrast["att"].to_numpy())
        assert line.get_label() == "high - low"
    overall = result.contrast_att_overall()
    overall_line = [
        line for line in axes[2].get_lines() if not line.get_label().startswith("_")
    ]
    np.testing.assert_allclose(overall_line[0].get_ydata(), overall["att"].to_numpy())
    assert axes[0].get_title() == "Event-time ATT by level"
    assert axes[1].get_title() == "Event-time contrast"
    assert axes[2].get_title() == "Overall post-treatment contrast"
    plt.close(fig)


@pytest.mark.integration
def test_bayesian_draw_means_match_base_estimates():
    data = _stacked()
    model = cp.pymc_models.LinearRegression(sample_kwargs=LIGHT_SAMPLE)
    result = cp.StaggeredDifferenceInDifferencesInDifferences(
        data,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        level_variable_name="sugar",
        treatment_time_variable_name="treatment_time",
        model=model,
        reference_level="low",
    ).fit()
    for experiment in result.experiments.values():
        draws = experiment.att_event_time_draws()
        assert isinstance(draws, xr.DataArray)
        assert draws.dims == ("chain", "draw", "event_time")
        identified = experiment.result.att_event_time
        identified = identified.loc[identified["identified"]]
        means = draws.mean(("chain", "draw")).values
        np.testing.assert_allclose(means, identified["att"].to_numpy())
        overall = experiment.att_overall_draws()
        post = identified.loc[identified["event_time"] >= 0, "att"]
        assert float(overall.mean()) == pytest.approx(float(post.mean()))

    high = result.experiments["high"].att_event_time_draws()
    low = result.experiments["low"].att_event_time_draws()
    shared = sorted(
        set(high.coords["event_time"].values.tolist())
        & set(low.coords["event_time"].values.tolist())
    )
    hand = result.contrast_draws(
        high.sel(event_time=shared).values,
        low.sel(event_time=shared).values,
    )
    table = result.contrast_att_event_time()
    np.testing.assert_allclose(table["att"].to_numpy(), hand.mean(axis=(0, 1)))
    assert table["att_lower"].notna().all()
    assert (table["att_upper"] >= table["att_lower"]).all()
    # The interval is not the difference of the subgroup endpoints.
    high_table = result.experiments["high"].result.att_event_time
    low_table = result.experiments["low"].result.att_event_time
    high_row = high_table.set_index("event_time").loc[shared]
    low_row = low_table.set_index("event_time").loc[shared]
    endpoint_gap = high_row["att_upper"].to_numpy() - low_row["att_upper"].to_numpy()
    assert not np.allclose(table["att_upper"].to_numpy(), endpoint_gap)
