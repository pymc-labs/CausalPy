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
"""Recovery and validation tests for the three-arm geo lift workflow."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt

import causalpy as cp
from causalpy.data.simulate_data import generate_up_down_geolift_data


def test_up_down_simulator_reproducible_and_exposes_potential_outcomes():
    first = generate_up_down_geolift_data(seed=123)
    second = generate_up_down_geolift_data(seed=123)
    for field in (
        "revenue",
        "no_intervention",
        "effects",
        "spend_baseline",
        "spend_planned",
        "spend_realized",
    ):
        pd.testing.assert_frame_equal(getattr(first, field), getattr(second, field))
    assert first.arms == second.arms
    assert first.treatment_time == second.treatment_time
    pd.testing.assert_frame_equal(first.revenue - first.no_intervention, first.effects)
    up = first.arms["up"]
    down = first.arms["down"]
    controls = first.arms["control"]
    post = first.effects.loc[first.treatment_time :]
    assert (post[up] > 0).all().all()
    assert (post[down] < 0).all().all()
    assert (post[controls] == 0).all().all()
    assert (first.effects.loc[: first.treatment_time].iloc[:-1] == 0).all().all()
    baseline_search = first.spend_baseline.xs("search", axis=1, level="channel").iloc[0]
    treated = up + down
    assert baseline_search[controls].min() < baseline_search[treated].min()
    assert baseline_search[controls].max() > baseline_search[treated].max()
    assert baseline_search[treated].nunique() == len(treated)
    pre_revenue = first.no_intervention.loc[
        first.no_intervention.index < first.treatment_time
    ]
    assert (pre_revenue[treated].min(axis=1) > pre_revenue[controls].min(axis=1)).all()
    assert (pre_revenue[treated].max(axis=1) < pre_revenue[controls].max(axis=1)).all()
    assert pre_revenue[controls].mean().max() - pre_revenue[controls].mean().min() > 50
    assert set(first.spend_baseline.columns.get_level_values("channel")) == {
        "search",
        "display",
    }
    assert (
        (
            first.spend_realized.loc[:, pd.IndexSlice[:, "display"]]
            == first.spend_baseline.loc[:, pd.IndexSlice[:, "display"]]
        )
        .all()
        .all()
    )
    planned_change = (
        first.spend_planned.loc[first.treatment_time :, pd.IndexSlice[:, "search"]]
        - first.spend_baseline.loc[first.treatment_time :, pd.IndexSlice[:, "search"]]
    )
    assert np.allclose(planned_change.sum(axis=1), 0)


def test_up_down_simulator_no_effect_and_no_delivery():
    no_effect = generate_up_down_geolift_data(seed=123, effect_scale=0)
    assert (no_effect.effects == 0).all().all()
    pd.testing.assert_frame_equal(no_effect.revenue, no_effect.no_intervention)
    no_delivery = generate_up_down_geolift_data(seed=123, delivery_fraction=0)
    assert (no_delivery.effects == 0).all().all()
    pd.testing.assert_frame_equal(
        no_delivery.spend_realized, no_delivery.spend_baseline
    )


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"n_pre": 0}, "counts must be positive"),
        ({"n_control": 1}, "At least two control geos"),
        ({"effect_scale": -1}, "effect_scale"),
        ({"delivery_fraction": 1.2}, "delivery_fraction"),
    ],
)
def test_up_down_simulator_rejects_invalid_design(settings, message):
    with pytest.raises(ValueError, match=message):
        generate_up_down_geolift_data(seed=123, **settings)


@pytest.mark.integration
def test_up_down_no_effect_recovers_near_zero(real_pymc_sampling):
    simulated = generate_up_down_geolift_data(
        seed=615,
        n_pre=32,
        n_post=8,
        n_control=5,
        n_up=1,
        n_down=1,
        effect_scale=0,
    )
    result = cp.UpDownGeoLift(
        simulated.revenue,
        simulated.treatment_time,
        simulated.arms,
        model=cp.pymc_models.SoftmaxWeightedSumFitter(
            sample_kwargs={
                "draws": 100,
                "tune": 100,
                "chains": 2,
                "cores": 1,
                "random_seed": 615,
                "target_accept": 0.95,
                "progressbar": False,
            }
        ),
    ).fit()
    table = result.arm_effect_table()
    assert (table.loc[table.level == "geo", "mean"].abs() < 1.5).all()


@pytest.mark.parametrize(
    ("arms", "message"),
    [
        ({"up": ["up_1"], "control": ["control_1"]}, "exactly"),
        ({"up": ["up_1"], "down": ["up_1"], "control": ["control_1"]}, "overlap"),
        ({"up": ["up_1"], "down": ["down_1"], "control": []}, "at least one"),
        ({"up": ["up_1"], "down": ["down_1"], "control": ["absent"]}, "missing"),
    ],
)
def test_up_down_assignment_validation(arms, message):
    simulated = generate_up_down_geolift_data(seed=123)
    with pytest.raises(ValueError, match=message):
        cp.UpDownGeoLift(simulated.revenue, simulated.treatment_time, arms)


def test_up_down_revenue_panel_requires_complete_unique_periods():
    simulated = generate_up_down_geolift_data(seed=123)
    duplicate = simulated.revenue.copy()
    duplicate.index = duplicate.index.where(
        duplicate.index != duplicate.index[1], duplicate.index[0]
    )
    with pytest.raises(ValueError, match="unique and sorted"):
        cp.UpDownGeoLift(duplicate, simulated.treatment_time, simulated.arms)
    missing = simulated.revenue.copy()
    missing.loc[missing.index[0], simulated.arms["up"][0]] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        cp.UpDownGeoLift(missing, simulated.treatment_time, simulated.arms)


@pytest.fixture(scope="module")
def fitted_up_down(real_pymc_sampling):
    simulated = generate_up_down_geolift_data(
        seed=415, n_pre=32, n_post=8, n_control=5, n_up=2, n_down=2
    )
    result = cp.UpDownGeoLift(
        simulated.revenue,
        simulated.treatment_time,
        simulated.arms,
        model=cp.pymc_models.SoftmaxWeightedSumFitter(
            sample_kwargs={
                "draws": 100,
                "tune": 100,
                "chains": 2,
                "cores": 1,
                "random_seed": 415,
                "target_accept": 0.95,
                "progressbar": False,
            }
        ),
    ).fit()
    return simulated, result


@pytest.mark.integration
def test_up_down_recovers_signed_geo_effects_and_joint_arm_draws(fitted_up_down):
    simulated, result = fitted_up_down
    impact = result.impact_draws
    assert {"chain", "draw", "obs_ind", "treated_units"} <= set(impact.dims)
    assert set(impact.treated_units.to_numpy()) == set(
        simulated.arms["up"] + simulated.arms["down"]
    )
    arm = result.arm_impact_draws
    xr.testing.assert_allclose(
        arm.sel(arm="up").drop_vars("arm"),
        impact.sel(treated_units=simulated.arms["up"])
        .mean("treated_units")
        .drop_vars("arm", errors="ignore"),
    )
    geo_mean = result.aggregate_draws()["geo"].mean(("chain", "draw"))
    with pytest.raises(ValueError, match="aggregate"):
        result.aggregate_draws(aggregate="median")
    with pytest.raises(ValueError, match="Window boundaries"):
        result.aggregate_draws(start=simulated.revenue.index[0])
    truth = simulated.effects.loc[simulated.treatment_time :].mean()
    for geo in simulated.arms["up"] + simulated.arms["down"]:
        estimate = float(geo_mean.sel(treated_units=geo))
        assert np.sign(estimate) == np.sign(truth[geo])
        assert abs(estimate - truth[geo]) < 2.5
    table = result.arm_effect_table()
    assert set(table["arm"]) == {"up", "down"}
    assert len(table) == 6
    assert table.loc[(table.level == "arm") & (table.arm == "up"), "mean"].item() > 0
    assert table.loc[(table.level == "arm") & (table.arm == "down"), "mean"].item() < 0


@pytest.mark.integration
def test_up_down_plot_labels_geo_and_revenue_units(fitted_up_down):
    simulated, result = fitted_up_down
    for arm in ("up", "down"):
        geo = simulated.arms[arm][0]
        fig, axes = result.plot(treated_unit=geo, show=False)
        assert len(axes) == 3
        assert axes[0].get_title().startswith(f"{geo}:")
        assert axes[0].get_ylabel() == "Revenue (USD/week)"
        assert axes[1].get_ylabel() == "Impact (USD/week)"
        assert axes[2].get_ylabel() == "Cumulative impact (USD)"
        plt.close(fig)


@pytest.mark.integration
def test_up_down_mmm_export_matches_window_and_validates_spend(fitted_up_down):
    simulated, result = fitted_up_down
    start = simulated.revenue.index[-5]
    end = simulated.revenue.index[-2]
    rows = result.to_mmm_lift(
        spend_baseline=simulated.spend_baseline,
        spend_realized=simulated.spend_realized,
        channel=simulated.tested_channel,
        spend_unit="USD/week",
        outcome_unit="USD/week",
        start=start,
        end=end,
    )
    assert len(rows) == 4
    assert set(rows.arm) == {"up", "down"}
    assert (rows.n_periods == 4).all()
    assert (rows.window_start == start).all()
    assert (rows.window_end == end).all()
    assert (rows.loc[rows.arm == "up", "delta_x"] > 0).all()
    assert (rows.loc[rows.arm == "down", "delta_x"] < 0).all()
    assert (rows["delta_x"] * rows["delta_y"] >= 0).all()
    for row in rows.itertuples():
        baseline = simulated.spend_baseline.loc[start:end, (row.geo, "search")]
        realized = simulated.spend_realized.loc[start:end, (row.geo, "search")]
        assert row.x == pytest.approx(baseline.mean())
        assert row.delta_x == pytest.approx((realized - baseline).mean())
        draws = result.aggregate_draws(start=start, end=end)["geo"].sel(
            treated_units=row.geo
        )
        assert row.delta_y == pytest.approx(float(draws.mean()))
        assert row.sigma == pytest.approx(float(draws.std()))

    missing = simulated.spend_realized.drop(index=start)
    with pytest.raises(ValueError, match="align"):
        result.to_mmm_lift(
            spend_baseline=simulated.spend_baseline,
            spend_realized=missing,
            channel="search",
            spend_unit="USD/week",
            outcome_unit="USD/week",
            start=start,
            end=end,
        )
    negative = simulated.spend_realized.copy()
    negative.loc[start, (simulated.arms["down"][0], "search")] = -1
    with pytest.raises(ValueError, match="below zero"):
        result.to_mmm_lift(
            spend_baseline=simulated.spend_baseline,
            spend_realized=negative,
            channel="search",
            spend_unit="USD/week",
            outcome_unit="USD/week",
            start=start,
            end=end,
        )
    missing_value = simulated.spend_realized.copy()
    missing_value.loc[start, (simulated.arms["up"][0], "search")] = np.nan
    with pytest.raises(ValueError, match="missing or nonfinite"):
        result.to_mmm_lift(
            spend_baseline=simulated.spend_baseline,
            spend_realized=missing_value,
            channel="search",
            spend_unit="USD/week",
            outcome_unit="USD/week",
            start=start,
            end=end,
        )
    varying = simulated.spend_realized.copy()
    varying.loc[start, (simulated.arms["up"][0], "search")] += 1
    with pytest.raises(ValueError, match="stable"):
        result.to_mmm_lift(
            spend_baseline=simulated.spend_baseline,
            spend_realized=varying,
            channel="search",
            spend_unit="USD/week",
            outcome_unit="USD/week",
            start=start,
            end=end,
        )
    wrong_unit = simulated.spend_baseline.copy()
    wrong_unit.attrs["unit"] = "GBP/week"
    with pytest.raises(ValueError, match="spend_unit"):
        result.to_mmm_lift(
            spend_baseline=wrong_unit,
            spend_realized=simulated.spend_realized,
            channel="search",
            spend_unit="USD/week",
            outcome_unit="USD/week",
            start=start,
            end=end,
        )
    with pytest.raises(ValueError, match="outcome_unit"):
        result.to_mmm_lift(
            spend_baseline=simulated.spend_baseline,
            spend_realized=simulated.spend_realized,
            channel="search",
            spend_unit="USD/week",
            outcome_unit="GBP/week",
            start=start,
            end=end,
        )


@pytest.mark.integration
@pytest.mark.parametrize("frame_name", ["spend_baseline", "spend_realized"])
def test_up_down_mmm_export_rejects_duplicate_spend_columns(fitted_up_down, frame_name):
    simulated, result = fitted_up_down
    baseline = simulated.spend_baseline.copy()
    realized = simulated.spend_realized.copy()
    frame = baseline if frame_name == "spend_baseline" else realized
    geo = simulated.arms["up"][0]
    frame.insert(
        len(frame.columns),
        (geo, "search"),
        frame[(geo, "search")],
        allow_duplicates=True,
    )
    with pytest.raises(ValueError, match="columns must be unique"):
        result.to_mmm_lift(
            spend_baseline=baseline,
            spend_realized=realized,
            channel="search",
            spend_unit="USD/week",
            outcome_unit="USD/week",
        )


@pytest.mark.integration
def test_up_down_mmm_export_rejects_zero_spend_change(fitted_up_down):
    simulated, result = fitted_up_down
    realized = simulated.spend_realized.copy()
    geo = simulated.arms["up"][0]
    realized[(geo, "search")] = simulated.spend_baseline[(geo, "search")]
    with pytest.raises(ValueError, match="spend change must be nonzero"):
        result.to_mmm_lift(
            spend_baseline=simulated.spend_baseline,
            spend_realized=realized,
            channel="search",
            spend_unit="USD/week",
            outcome_unit="USD/week",
        )


@pytest.mark.integration
def test_up_down_mmm_export_rejects_zero_lift(fitted_up_down, monkeypatch):
    simulated, result = fitted_up_down
    zero_draws = result.aggregate_draws()
    zero_draws["geo"] = xr.zeros_like(zero_draws["geo"])
    monkeypatch.setattr(result, "aggregate_draws", lambda **_: zero_draws)
    with pytest.raises(ValueError, match="Estimated lift must be nonzero"):
        result.to_mmm_lift(
            spend_baseline=simulated.spend_baseline,
            spend_realized=simulated.spend_realized,
            channel="search",
            spend_unit="USD/week",
            outcome_unit="USD/week",
        )
