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
"""
Tests for plot utility functions
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from causalpy.plot_utils import dataarray_draws, get_hdi_to_df


@pytest.mark.integration
def test_panel_axes_filters_colorbars():
    from plotnine import aes, geom_point, ggplot

    from causalpy.plot_utils import panel_axes

    p = ggplot() + geom_point(pd.DataFrame({"x": [1, 2], "y": [1, 2]}), aes("x", "y"))
    fig = p.draw(show=False)
    axes = panel_axes(fig)
    assert axes
    assert all(a.get_subplotspec() is not None for a in axes)
    plt.close(fig)


@pytest.mark.integration
def test_plot_spec_overlay_runs_once():
    from plotnine import aes, geom_point, ggplot

    from causalpy.plot_utils import PlotSpec, panel_axes

    calls: list[int] = []

    def overlay(_fig, axes):
        calls.append(len(axes))

    p = ggplot() + geom_point(pd.DataFrame({"x": [1], "y": [1]}), aes("x", "y"))
    spec = PlotSpec(p, overlay=overlay, n_panels=1)
    fig = spec.plot.draw(show=False)
    axes = panel_axes(fig, spec.n_panels)
    spec.overlay(fig, axes)
    assert calls == [1]
    plt.close(fig)


@pytest.mark.integration
def test_get_hdi_to_df_with_coordinate_dimensions():
    """
    Regression test for bug where get_hdi_to_df returned string coordinate values
    instead of numeric HDI values when xarray had named coordinate dimensions.

    This bug manifested in multi-cell synthetic control experiments where columns
    like 'pred_hdi_upper_94' contained the string "treated_agg" instead of
    numeric upper bound values.

    See: https://github.com/pymc-labs/CausalPy/issues/532
    """
    # Create a mock xarray DataArray similar to what's produced in synthetic control
    # with a coordinate dimension like 'treated_units'
    np.random.seed(42)
    n_chains = 2
    n_draws = 100
    n_obs = 10

    # Simulate posterior samples with a named coordinate
    data = np.random.normal(loc=5.0, scale=0.5, size=(n_chains, n_draws, n_obs))

    xr_data = xr.DataArray(
        data,
        dims=["chain", "draw", "obs_ind"],
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "obs_ind": np.arange(n_obs),
            "treated_units": "treated_agg",  # This coordinate caused the bug
        },
    )

    # Call get_hdi_to_df
    result = get_hdi_to_df(xr_data, hdi_prob=0.94)

    # Assertions to verify the bug is fixed
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"

    # Check that we have exactly 2 columns (lower and higher)
    assert result.shape[1] == 2, f"Expected 2 columns, got {result.shape[1]}"

    # Check column names
    assert "lower" in result.columns, "Should have 'lower' column"
    assert "higher" in result.columns, "Should have 'higher' column"

    # CRITICAL: Check that columns contain numeric data, not strings
    assert result["lower"].dtype in [
        np.float64,
        np.float32,
    ], f"'lower' column should be numeric, got {result['lower'].dtype}"
    assert result["higher"].dtype in [
        np.float64,
        np.float32,
    ], f"'higher' column should be numeric, got {result['higher'].dtype}"

    # Check that no string values like 'treated_agg' appear in the data
    assert not (result["lower"].astype(str).str.contains("treated_agg").any()), (
        "'lower' column should not contain coordinate string values"
    )
    assert not (result["higher"].astype(str).str.contains("treated_agg").any()), (
        "'higher' column should not contain coordinate string values"
    )

    # Verify HDI ordering
    assert (result["lower"] <= result["higher"]).all(), (
        "'lower' should be <= 'higher' for all rows"
    )

    # Verify reasonable HDI values (should be around the mean of 5.0)
    assert result["lower"].min() > 3.0, "HDI lower bounds should be reasonable"
    assert result["higher"].max() < 7.0, "HDI upper bounds should be reasonable"


@pytest.fixture
def synthetic_posterior_draws():
    rng = np.random.default_rng(42)
    obs_ind = pd.date_range("2020-01-01", periods=20, freq="D")
    values = rng.normal(
        loc=np.linspace(10, 12, len(obs_ind)),
        scale=1,
        size=(2, 100, len(obs_ind)),
    )
    da = xr.DataArray(
        values,
        dims=["chain", "draw", "obs_ind"],
        coords={"chain": range(2), "draw": range(100), "obs_ind": obs_ind},
    )
    return dataarray_draws(da)


def test_dataarray_draws_selects_requested_treated_unit():
    da = xr.DataArray(
        np.arange(16).reshape(1, 2, 2, 4),
        dims=["chain", "draw", "treated_units", "obs_ind"],
        coords={"treated_units": ["a", "b"]},
    )

    draws = dataarray_draws(da, treated_unit="b")

    assert "treated_units" not in draws.columns
    assert draws["mu"].to_list() == [4, 5, 6, 7, 12, 13, 14, 15]


def test_summarize_draws_preserves_requested_interval_mass(synthetic_posterior_draws):
    from causalpy.plot_utils import summarize_draws

    narrow = summarize_draws(
        synthetic_posterior_draws,
        group_by="obs_ind",
        ci_prob=0.5,
        interval="eti",
    )
    wide = summarize_draws(
        synthetic_posterior_draws,
        group_by="obs_ind",
        ci_prob=0.9,
        interval="eti",
    )

    assert np.allclose(narrow["mu"], wide["mu"])
    assert (wide["mu_lower"] <= narrow["mu_lower"]).all()
    assert (wide["mu_upper"] >= narrow["mu_upper"]).all()


def test_spaghetti_draws_samples_complete_paths(synthetic_posterior_draws):
    from causalpy.plot_utils import label_draws, spaghetti_draws

    sampled = spaghetti_draws(
        label_draws(synthetic_posterior_draws, series="posterior", panel="top"),
        group_by=["panel", "series", "obs_ind"],
        num_samples=3,
    )

    assert sampled["_draw_id"].nunique() == 3
    assert sampled.groupby("_draw_id").size().eq(20).all()
    assert sampled["_line_id"].nunique() == 3
    assert sampled.groupby("_line_id")[["panel", "series"]].nunique().eq(1).all().all()


def test_posterior_histogram_tiles_use_draw_proportions(synthetic_posterior_draws):
    from causalpy.plot_utils import posterior_histogram_tiles

    tiles = posterior_histogram_tiles(synthetic_posterior_draws, "obs_ind")
    assert {"obs_ind", "y", "width", "height", "density"} <= set(tiles.columns)
    assert pd.api.types.is_datetime64_any_dtype(tiles["obs_ind"])
    assert (
        tiles.groupby("obs_ind")["density"]
        .sum()
        .between(0.99, 1.01, inclusive="both")
        .all()
    )


def test_histogram_layers_share_explicit_y_grid(synthetic_posterior_draws):
    from causalpy.plot_utils import (
        histogram_y_edges,
        posterior_histogram_tiles,
    )

    shifted = synthetic_posterior_draws.with_columns(
        mu=synthetic_posterior_draws["mu"] + 10
    )
    edges = histogram_y_edges(synthetic_posterior_draws, shifted)
    tiles = pd.concat(
        [
            posterior_histogram_tiles(
                synthetic_posterior_draws,
                "obs_ind",
                panel="first",
                y_edges=edges,
            ),
            posterior_histogram_tiles(
                shifted,
                "obs_ind",
                panel="second",
                y_edges=edges,
            ),
        ],
        ignore_index=True,
    )

    assert set(tiles["panel"]) == {"first", "second"}
    assert tiles.groupby("panel")["y"].agg(["min", "max"]).nunique().eq(1).all()


@pytest.mark.integration
def test_posterior_histogram_tiles_render_with_plotnine(synthetic_posterior_draws):
    from plotnine import ggplot

    from causalpy.plot_utils import histogram_tile_layers, posterior_histogram_tiles

    tiles = posterior_histogram_tiles(synthetic_posterior_draws, "obs_ind")
    p = ggplot()
    for layer in histogram_tile_layers(tiles, "obs_ind"):
        p += layer
    fig = p.draw(show=False)
    assert fig.axes
    plt.close(fig)


def test_posterior_kind_layers_spaghetti_requires_df(synthetic_posterior_draws):
    from causalpy.plot_utils import posterior_kind_layers, summarize_draws

    bands = summarize_draws(
        synthetic_posterior_draws,
        group_by="obs_ind",
        ci_prob=0.94,
    ).assign(series="a")
    with pytest.raises(ValueError, match="spaghetti_df"):
        posterior_kind_layers(bands, "spaghetti", x="obs_ind", y="mu")
