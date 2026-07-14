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
"""Structural tests for posterior plot kinds and the stable (Figure, Axes) contract."""

from __future__ import annotations

import matplotlib.axes
import matplotlib.figure
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

import causalpy as cp
from causalpy.data.simulate_data import generate_piecewise_its_data
from causalpy.tests.conftest import setup_regression_kink_data
from causalpy.tests.test_hdi_prob_wiring import sample_kwargs

POSTERIOR_KINDS = ("ribbon", "histogram", "spaghetti")


def _axes_list(ax: matplotlib.axes.Axes | np.ndarray | list) -> list:
    if isinstance(ax, matplotlib.axes.Axes):
        return [ax]
    if isinstance(ax, np.ndarray):
        return list(ax.flat)
    return list(ax)


def assert_figure_axes_contract(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes | np.ndarray | list,
    *,
    min_axes: int = 1,
) -> list[matplotlib.axes.Axes]:
    assert isinstance(fig, plt.Figure)
    axes = _axes_list(ax)
    assert len(axes) >= min_axes
    for a in axes:
        assert isinstance(a, matplotlib.axes.Axes)
    return axes


@pytest.mark.integration
@pytest.mark.parametrize("kind", POSTERIOR_KINDS)
def test_rd_plot_kinds_return_figure_axes(mock_pymc_sample, rd_data, kind):
    result = cp.RegressionDiscontinuity(
        rd_data,
        formula="y ~ 1 + bs(x, df=6) + treated",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=0.5,
        epsilon=0.001,
    )
    fig, ax = result.plot(show=False, kind=kind, num_samples=10)
    axes = assert_figure_axes_contract(fig, ax, min_axes=1)
    if kind == "spaghetti":
        assert sum(len(a.lines) for a in axes) > 10


@pytest.mark.integration
@pytest.mark.parametrize("kind", POSTERIOR_KINDS)
def test_rk_plot_kinds_return_figure_axes(mock_pymc_sample, kind):
    kink = 0.5
    df = setup_regression_kink_data(kink)
    result = cp.RegressionKink(
        df,
        formula=f"y ~ 1 + x + I((x-{kink})*treated)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        kink_point=kink,
    )
    fig, ax = result.plot(show=False, kind=kind, num_samples=10)
    axes = assert_figure_axes_contract(fig, ax, min_axes=1)
    if kind == "spaghetti":
        assert sum(len(a.lines) for a in axes) > 10


@pytest.mark.integration
@pytest.mark.parametrize("kind", POSTERIOR_KINDS)
def test_its_plot_kinds_return_three_panels(mock_pymc_sample, its_data, kind):
    result = cp.InterruptedTimeSeries(
        its_data,
        pd.to_datetime("2017-01-01"),
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    fig, ax = result.plot(show=False, kind=kind, num_samples=10)
    axes = assert_figure_axes_contract(fig, ax, min_axes=3)
    assert len(axes) == 3
    if kind == "spaghetti":
        assert sum(len(a.lines) for a in axes) > 10


@pytest.mark.integration
@pytest.mark.parametrize("kind", POSTERIOR_KINDS)
def test_sc_plot_kinds_return_three_panels(mock_pymc_sample, sc_data, kind):
    result = cp.SyntheticControl(
        sc_data,
        70,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )
    fig, ax = result.plot(show=False, kind=kind, num_samples=10)
    axes = assert_figure_axes_contract(fig, ax, min_axes=3)
    assert len(axes) == 3


@pytest.mark.integration
@pytest.mark.parametrize("kind", POSTERIOR_KINDS)
def test_sdid_plot_kinds_return_three_panels(mock_pymc_sample, sc_data, kind):
    result = cp.SyntheticDifferenceInDifferences(
        sc_data,
        70,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.SyntheticDifferenceInDifferencesWeightFitter(
            sample_kwargs=sample_kwargs
        ),
    )
    fig, ax = result.plot(show=False, kind=kind, num_samples=10)
    axes = assert_figure_axes_contract(fig, ax, min_axes=3)
    assert len(axes) == 3


@pytest.mark.integration
@pytest.mark.parametrize("kind", POSTERIOR_KINDS)
def test_did_plot_kinds_return_figure_axes(mock_pymc_sample, did_data, kind):
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    fig, ax = result.plot(show=False, kind=kind, num_samples=10)
    assert_figure_axes_contract(fig, ax, min_axes=1)


@pytest.mark.integration
def test_did_plot_supports_boolean_time(mock_pymc_sample, did_data):
    did_data = did_data.assign(t=did_data["t"].astype(bool))
    result = cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    fig, ax = result.plot(show=False, kind="histogram")

    assert_figure_axes_contract(fig, ax)


@pytest.mark.integration
@pytest.mark.parametrize("kind", POSTERIOR_KINDS)
def test_prepost_plot_kinds_return_two_panels(mock_pymc_sample, anova1_data, kind):
    result = cp.PrePostNEGD(
        anova1_data,
        formula="post ~ 1 + C(group) + pre",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    fig, ax = result.plot(show=False, kind=kind, num_samples=10)
    axes = assert_figure_axes_contract(fig, ax, min_axes=2)
    assert len(axes) == 2
    assert axes[0].get_xlabel() == "Pretest"
    assert axes[0].get_ylabel() == "Posttest"
    assert axes[1].get_title().startswith("mean =")
    legend = axes[0].get_legend()
    assert legend is not None
    assert {text.get_text() for text in legend.get_texts()} == {
        "Control group",
        "Treatment group",
    }


@pytest.mark.integration
@pytest.mark.parametrize("kind", POSTERIOR_KINDS)
def test_piecewise_plot_kinds_return_three_panels(mock_pymc_sample, kind):
    df, _ = generate_piecewise_its_data(N=100, seed=42)
    result = cp.PiecewiseITS(
        df,
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    fig, ax = result.plot(show=False, kind=kind, num_samples=10)
    axes = assert_figure_axes_contract(fig, ax, min_axes=3)
    assert len(axes) == 3


@pytest.mark.integration
def test_rd_plot_eti_path_returns_figure_axes(mock_pymc_sample, rd_data):
    result = cp.RegressionDiscontinuity(
        rd_data,
        formula="y ~ 1 + bs(x, df=6) + treated",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=0.5,
        epsilon=0.001,
    )
    fig, ax = result.plot(
        show=False,
        kind="ribbon",
        ci_kind="eti",
        figsize=(5, 4),
    )
    assert_figure_axes_contract(fig, ax)
    assert np.allclose(fig.get_size_inches(), (5, 4))


@pytest.mark.integration
def test_rd_plot_rejects_invalid_posterior_kind(mock_pymc_sample, rd_data):
    result = cp.RegressionDiscontinuity(
        rd_data,
        formula="y ~ 1 + bs(x, df=6) + treated",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=0.5,
        epsilon=0.001,
    )

    with pytest.raises(ValueError, match="Unknown kind"):
        result.plot(show=False, kind="invalid")  # type: ignore[arg-type]


@pytest.mark.integration
def test_its_plot_eti_path_returns_three_panels(mock_pymc_sample, its_data):
    result = cp.InterruptedTimeSeries(
        its_data,
        pd.to_datetime("2017-01-01"),
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    fig, ax = result.plot(show=False, kind="ribbon", ci_kind="eti")
    assert_figure_axes_contract(fig, ax, min_axes=3)
