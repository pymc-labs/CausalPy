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
import polars as pl
import pytest
import xarray as xr

from causalpy.plot_utils import dataarray_draws


@pytest.mark.integration
def test_panel_axes_filters_colorbars():
    import plotnine as p9

    from causalpy.plot_utils import panel_axes

    p = p9.ggplot() + p9.geom_point(
        pd.DataFrame({"x": [1, 2], "y": [1, 2]}), p9.aes("x", "y")
    )
    fig = p.draw(show=False)
    axes = panel_axes(fig)
    assert axes
    assert all(a.get_subplotspec() is not None for a in axes)
    plt.close(fig)


@pytest.mark.integration
def test_plot_spec_overlay_runs_once():
    import plotnine as p9

    from causalpy.plot_utils import PlotSpec, panel_axes

    calls: list[int] = []

    def overlay(_fig, axes):
        calls.append(len(axes))

    p = p9.ggplot() + p9.geom_point(
        pd.DataFrame({"x": [1], "y": [1]}), p9.aes("x", "y")
    )
    spec = PlotSpec(p, overlay=overlay, n_panels=1)
    fig = spec.plot.draw(show=False)
    axes = panel_axes(fig, spec.n_panels)
    spec.overlay(fig, axes)
    assert calls == [1]
    plt.close(fig)


def test_dataarray_summary_ignores_scalar_string_coordinates():
    """Regression test for #532's string coordinate leaking into HDI columns."""
    from causalpy.plot_utils import summarize_draws

    np.random.seed(42)
    n_chains = 2
    n_draws = 100
    n_obs = 10
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

    result = summarize_draws(
        dataarray_draws(xr_data),
        group_by="obs_ind",
        ci_prob=0.94,
    )

    assert pd.api.types.is_numeric_dtype(result["mu_lower"])
    assert pd.api.types.is_numeric_dtype(result["mu_upper"])
    assert (result["mu_lower"] <= result["mu_upper"]).all()


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


def test_histogram_layers_keep_series_in_separate_geoms(synthetic_posterior_draws):
    from causalpy.plot_utils import label_draws, posterior_kind_layers

    shifted = synthetic_posterior_draws.with_columns(
        mu=synthetic_posterior_draws["mu"] + 10
    )
    draws = pl.concat(
        [
            label_draws(
                synthetic_posterior_draws,
                series="first",
            ),
            label_draws(
                shifted,
                series="second",
            ),
        ]
    )
    _, layers = posterior_kind_layers(
        draws,
        "histogram",
        x="obs_ind",
        group_by=["series", "obs_ind"],
        ci_prob=0.94,
        colors={"first": "blue", "second": "orange"},
    )
    bin_layers = [layer for layer in layers if type(layer).__name__ == "geom_bin_2d"]

    assert len(bin_layers) == 2
    assert [layer.data["series"].drop_duplicates().item() for layer in bin_layers] == [
        "first",
        "second",
    ]


@pytest.mark.integration
def test_posterior_histogram_layers_render_with_plotnine(synthetic_posterior_draws):
    import plotnine as p9

    from causalpy.plot_utils import label_draws, posterior_kind_layers

    draws = label_draws(synthetic_posterior_draws, series="posterior")
    _, layers = posterior_kind_layers(
        draws,
        "histogram",
        x="obs_ind",
        group_by=["series", "obs_ind"],
        ci_prob=0.94,
        colors={"posterior": "orange"},
    )
    p = p9.ggplot()
    for layer in layers:
        p += layer
    fig = p.draw(show=False)
    assert fig.axes
    plt.close(fig)


def test_posterior_kind_layers_prepares_spaghetti(synthetic_posterior_draws):
    from causalpy.plot_utils import label_draws, posterior_kind_layers

    draws = label_draws(synthetic_posterior_draws, series="a")
    bands, layers = posterior_kind_layers(
        draws,
        "spaghetti",
        x="obs_ind",
        group_by=["series", "obs_ind"],
        ci_prob=0.94,
        num_samples=3,
    )

    assert len(bands) == 20
    assert len(layers) == 2
    assert layers[0].data["_draw_id"].nunique() == 3


def test_validate_posterior_plot_options_rejects_invalid_kind():
    from causalpy.plot_utils import validate_posterior_plot_options

    with pytest.raises(ValueError, match="Unknown kind"):
        validate_posterior_plot_options("bands")


def test_validate_posterior_plot_options_rejects_invalid_ci_kind():
    from causalpy.plot_utils import validate_posterior_plot_options

    with pytest.raises(ValueError, match="Unknown ci_kind"):
        validate_posterior_plot_options("ribbon", ci_kind="quantile")


def test_validate_posterior_plot_options_rejects_non_positive_num_samples():
    from causalpy.plot_utils import validate_posterior_plot_options

    with pytest.raises(ValueError, match="num_samples must be positive"):
        validate_posterior_plot_options("ribbon", num_samples=0)


def test_spaghetti_draws_isolates_paths_across_series(synthetic_posterior_draws):
    from causalpy.plot_utils import label_draws, spaghetti_draws

    draws = pl.concat(
        [
            label_draws(synthetic_posterior_draws, series="control"),
            label_draws(synthetic_posterior_draws, series="treatment"),
        ],
        how="diagonal_relaxed",
    )
    sampled = spaghetti_draws(
        draws,
        group_by=["series", "obs_ind"],
        num_samples=4,
    )

    assert sampled["_line_id"].nunique() == 8
    assert sampled.groupby("series")["_line_id"].nunique().eq(4).all()
    assert sampled.groupby("_line_id")["series"].nunique().eq(1).all()


def _assert_explicit_causal_panel_schema(panel_data) -> None:
    draw_fields = (
        "fitted",
        "counterfactual",
        "pre_effect",
        "post_effect",
        "cumulative_effect",
    )
    for field in draw_fields:
        draws = getattr(panel_data, field)
        if draws is not None:
            assert {"chain", "draw", "obs_ind", "mu"} <= set(draws.columns)
            assert not (
                set(draws.columns)
                & {"panel", "series", "mu_lower", "mu_upper", "y1", "y2"}
            )
    assert set(panel_data.observations.columns) == {"obs_ind", "value"}


@pytest.mark.integration
def test_its_causal_panel_data_uses_explicit_quantities(mock_pymc_sample, its_data):
    """Extractor exposes named posterior quantities, not render artifacts."""
    import pandas as pd

    import causalpy as cp
    from causalpy.tests.test_hdi_prob_wiring import sample_kwargs

    result = cp.InterruptedTimeSeries(
        its_data,
        pd.to_datetime("2017-01-01"),
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    panel_data = result._causal_panel_data()

    _assert_explicit_causal_panel_schema(panel_data)
    assert panel_data.pre_effect is not None


@pytest.mark.integration
def test_sc_causal_panel_data_uses_explicit_quantities(mock_pymc_sample, sc_data):
    """Synthetic Control extractor matches the explicit panel contract."""
    import causalpy as cp
    from causalpy.tests.test_hdi_prob_wiring import sample_kwargs

    result = cp.SyntheticControl(
        sc_data,
        70,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )
    panel_data = result._causal_panel_data(treated_unit="actual")

    _assert_explicit_causal_panel_schema(panel_data)


@pytest.mark.integration
def test_sdid_causal_panel_data_uses_explicit_quantities(mock_pymc_sample, sc_data):
    """SDiD extractor matches the explicit panel contract."""
    import causalpy as cp
    from causalpy.tests.test_hdi_prob_wiring import sample_kwargs

    result = cp.SyntheticDifferenceInDifferences(
        sc_data,
        70,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.SyntheticDifferenceInDifferencesWeightFitter(
            sample_kwargs=sample_kwargs
        ),
    )
    panel_data = result._causal_panel_data(treated_unit="actual")

    _assert_explicit_causal_panel_schema(panel_data)
