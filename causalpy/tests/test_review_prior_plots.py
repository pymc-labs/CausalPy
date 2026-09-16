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
import pytest

import causalpy as cp


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
