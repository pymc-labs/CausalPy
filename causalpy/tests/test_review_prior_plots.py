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
"""Regression coverage for prior diagnostic plot controls."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection, QuadMesh

import causalpy as cp
from causalpy.data.simulate_data import generate_piecewise_its_data
from causalpy.tests.conftest import setup_regression_kink_data


@pytest.fixture(scope="module")
def multi_unit_prior_sc():
    data = cp.load_data("sc")
    data["actual2"] = data["actual"] + 50
    experiment = cp.SyntheticControl(
        data,
        treatment_time=70,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual", "actual2"],
        model=cp.pymc_models.WeightedSumFitter(),
    )
    experiment.sample_prior_predictive(draws=40, random_seed=42)
    return experiment


def test_prior_sc_plot_selects_requested_unit(multi_unit_prior_sc):
    experiment = multi_unit_prior_sc
    fig, axes = experiment.plot(group="prior", treated_unit="actual2", show=False)
    try:
        observed = [line for line in axes[0].lines if line.get_marker() == "."]
        np.testing.assert_allclose(
            observed[0].get_ydata(), experiment.datapre["actual2"]
        )
        np.testing.assert_allclose(
            observed[1].get_ydata(), experiment.datapost["actual2"]
        )
        predicted = axes[0].lines[0].get_ydata()
        np.testing.assert_allclose(
            predicted,
            experiment.prior_result.predictions_pre.sel(treated_units="actual2")
            .mean(("chain", "draw"))
            .values,
        )
    finally:
        plt.close(fig)


def test_prior_sc_plot_rejects_unknown_unit(multi_unit_prior_sc):
    with pytest.raises(ValueError, match="treated_unit 'NOT_A_UNIT' not found"):
        multi_unit_prior_sc.plot(group="prior", treated_unit="NOT_A_UNIT", show=False)


@pytest.fixture(
    scope="module",
    params=["its", "sc", "piecewise", "did", "prepost", "rd", "rk", "sdid"],
)
def prior_plot_case(request):
    name = request.param
    model = cp.pymc_models.LinearRegression()
    if name == "its":
        experiment = cp.InterruptedTimeSeries(
            request.getfixturevalue("its_data"),
            treatment_time=pd.Timestamp("2017-06-01"),
            formula="y ~ 1 + t",
            model=model,
        )
    elif name == "sc":
        experiment = request.getfixturevalue("multi_unit_prior_sc")
    elif name == "piecewise":
        data, _ = generate_piecewise_its_data(N=60, seed=42)
        experiment = cp.PiecewiseITS(
            data, formula="y ~ 1 + t + step(t, 30)", model=model
        )
    elif name == "did":
        experiment = cp.DifferenceInDifferences(
            request.getfixturevalue("did_data"),
            formula="y ~ 1 + group*post_treatment",
            time_variable_name="t",
            group_variable_name="group",
            model=model,
        )
    elif name == "prepost":
        experiment = cp.PrePostNEGD(
            request.getfixturevalue("anova1_data"),
            formula="post ~ 1 + C(group) + pre",
            group_variable_name="group",
            pretreatment_variable_name="pre",
            model=model,
        )
    elif name == "rd":
        data = request.getfixturevalue("rd_data")
        experiment = cp.RegressionDiscontinuity(
            data.assign(treated=data["treated"].astype(int)),
            formula="y ~ 1 + x + treated",
            treatment_threshold=0.5,
            model=model,
        )
    elif name == "rk":
        experiment = cp.RegressionKink(
            setup_regression_kink_data(0.0),
            formula="y ~ 1 + x + I(x*treated)",
            kink_point=0.0,
            model=model,
        )
    else:
        experiment = cp.SyntheticDifferenceInDifferences(
            cp.load_data("sc"),
            treatment_time=70,
            control_units=["a", "b", "c", "d", "e", "f", "g"],
            treated_units=["actual"],
            model=cp.pymc_models.SyntheticDifferenceInDifferencesWeightFitter(),
        )
    if not experiment.has_prior_predictive:
        experiment.sample_prior_predictive(draws=40, random_seed=42)

    bundle = experiment.prior_result
    if name in {"did", "prepost"}:
        prediction = bundle.scenario_control.prediction
    elif name in {"rd", "rk"}:
        prediction = bundle.predictions
    else:
        prediction = bundle.predictions_pre
    return name, experiment, prediction.isel(treated_units=0)


@pytest.mark.parametrize("ci_prob", [0.2, 0.8])
def test_prior_plot_uses_requested_interval(prior_plot_case, ci_prob):
    name, experiment, prediction = prior_plot_case
    size_kwargs = {} if name == "sdid" else {"figsize": (3, 3)}
    fig, axes = experiment.plot(
        group="prior", ci_prob=ci_prob, ci_kind="eti", show=False, **size_kwargs
    )
    try:
        axes = np.atleast_1d(axes)
        assert len(axes) == 1
        if size_kwargs:
            np.testing.assert_allclose(fig.get_size_inches(), (3, 3))
        band = next(
            artist
            for artist in axes[0].collections
            if isinstance(artist, PolyCollection)
        )
        vertices = band.get_paths()[0].vertices
        quantiles = prediction.quantile(
            [(1 - ci_prob) / 2, (1 + ci_prob) / 2], dim=("chain", "draw")
        )
        np.testing.assert_allclose(
            [vertices[:, 1].min(), vertices[:, 1].max()],
            [quantiles.isel(quantile=0).min(), quantiles.isel(quantile=1).max()],
        )
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("num_samples", [3, 5])
def test_prior_plot_renders_requested_draw_count(prior_plot_case, num_samples):
    name, experiment, prediction = prior_plot_case
    fig, axes = experiment.plot(
        group="prior", kind="spaghetti", num_samples=num_samples, show=False
    )
    try:
        ax = np.atleast_1d(axes)[0]
        sample_lines = [
            line
            for line in ax.lines
            if line.get_alpha() == 0.1 and line.get_linewidth() == 0.5
        ]
        n_series = 1 if name in {"rd", "rk"} else 2
        assert len(sample_lines) == n_series * num_samples
        samples = prediction.stack(sample=("chain", "draw")).transpose("sample", ...)
        for line in sample_lines[:num_samples]:
            assert any(
                np.allclose(line.get_ydata(), sample) for sample in samples.values
            )
        assert not any(isinstance(artist, PolyCollection) for artist in ax.collections)
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_prior_plot_renders_histogram_density(prior_plot_case):
    name, experiment, _ = prior_plot_case
    fig, axes = experiment.plot(group="prior", kind="histogram", show=False)
    try:
        ax = np.atleast_1d(axes)[0]
        meshes = [artist for artist in ax.collections if isinstance(artist, QuadMesh)]
        assert len(meshes) == (1 if name in {"rd", "rk"} else 2)
        for mesh in meshes:
            np.testing.assert_allclose(np.asarray(mesh.get_array()).max(axis=0), 1)
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_prior_sc_plot_overlays_requested_donors(multi_unit_prior_sc):
    experiment = multi_unit_prior_sc
    fig, axes = experiment.plot(group="prior", plot_predictors=True, show=False)
    try:
        donors = [line for line in axes[0].lines if line.get_zorder() == 1]
        assert len(donors) == 2 * len(experiment.control_units)
        n_controls = len(experiment.control_units)
        for line, unit in zip(
            donors[:n_controls], experiment.control_units, strict=True
        ):
            np.testing.assert_allclose(line.get_ydata(), experiment.datapre[unit])
        for line, unit in zip(
            donors[n_controls:], experiment.control_units, strict=True
        ):
            np.testing.assert_allclose(line.get_ydata(), experiment.datapost[unit])
    finally:
        plt.close(fig)
