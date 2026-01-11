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
Tests for hdi_type parameter functionality across experiment classes.

These tests specifically target the hdi_type="prediction" and show_hdi_annotation=True
branches to ensure code coverage.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import causalpy as cp

# Minimal sample kwargs for fast testing
SAMPLE_KWARGS = {
    "tune": 5,
    "draws": 10,
    "chains": 1,
    "progressbar": False,
    "random_seed": 42,
}


def _setup_regression_kink_data(kink):
    """Set up data for regression kink design tests."""
    seed = 42
    rng = np.random.default_rng(seed)
    N = 50
    beta = [0, -1, 0, 2, 0]
    sigma = 0.05
    x = rng.uniform(-1, 1, N)
    y = _reg_kink_function(x, beta, kink) + rng.normal(0, sigma, N)
    return pd.DataFrame({"x": x, "y": y, "treated": x >= kink})


def _reg_kink_function(x, beta, kink):
    """Utility function for regression kink design."""
    return (
        beta[0]
        + beta[1] * x
        + beta[2] * x**2
        + beta[3] * (x - kink) * (x >= kink)
        + beta[4] * ((x - kink) ** 2) * (x >= kink)
    )


@pytest.fixture
def its_result():
    """Create a minimal ITS result for testing."""
    df = cp.load_data("its")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=pd.Timestamp("2017-01-01"),
        formula="y ~ 1 + t",
        model=cp.pymc_models.LinearRegression(sample_kwargs=SAMPLE_KWARGS),
    )
    return result


@pytest.fixture
def sc_result():
    """Create a minimal Synthetic Control result for testing."""
    df = cp.load_data("sc")
    result = cp.SyntheticControl(
        df,
        treatment_time=70,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=SAMPLE_KWARGS),
    )
    return result


@pytest.fixture
def did_result():
    """Create a minimal DiD result for testing."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=SAMPLE_KWARGS),
    )
    return result


@pytest.fixture
def rd_result():
    """Create a minimal RD result for testing."""
    df = cp.load_data("rd")
    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + x + treated",
        treatment_threshold=0.5,
        model=cp.pymc_models.LinearRegression(sample_kwargs=SAMPLE_KWARGS),
        running_variable_name="x",
    )
    return result


@pytest.fixture
def rkink_result():
    """Create a minimal RegressionKink result for testing."""
    kink = 0.5
    df = _setup_regression_kink_data(kink)
    result = cp.RegressionKink(
        df,
        formula=f"y ~ 1 + x + I((x-{kink})*treated)",
        kink_point=kink,
        model=cp.pymc_models.LinearRegression(sample_kwargs=SAMPLE_KWARGS),
    )
    return result


@pytest.fixture
def prepostnegd_result():
    """Create a minimal PrePostNEGD result for testing."""
    df = cp.load_data("anova1")
    result = cp.PrePostNEGD(
        df,
        formula="post ~ 1 + pre + group",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=cp.pymc_models.LinearRegression(sample_kwargs=SAMPLE_KWARGS),
    )
    return result


class TestHdiTypePlotting:
    """Test hdi_type parameter for plotting methods."""

    @pytest.mark.integration
    def test_its_plot_prediction_hdi(self, its_result):
        """Test ITS plot with hdi_type='prediction'."""
        fig, ax = its_result.plot(hdi_type="prediction")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.integration
    def test_its_plot_show_annotation(self, its_result):
        """Test ITS plot with show_hdi_annotation=True."""
        fig, ax = its_result.plot(show_hdi_annotation=True)
        # Check that annotation was added to the title
        title = ax[0].get_title()
        assert "HDI" in title or "94%" in title
        plt.close(fig)

    @pytest.mark.integration
    def test_its_plot_prediction_with_annotation(self, its_result):
        """Test ITS plot with both hdi_type='prediction' and annotation."""
        fig, ax = its_result.plot(hdi_type="prediction", show_hdi_annotation=True)
        title = ax[0].get_title()
        assert "posterior predictive" in title or "ŷ" in title
        plt.close(fig)

    @pytest.mark.integration
    def test_sc_plot_prediction_hdi(self, sc_result):
        """Test Synthetic Control plot with hdi_type='prediction'."""
        fig, ax = sc_result.plot(hdi_type="prediction")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.integration
    def test_sc_plot_show_annotation(self, sc_result):
        """Test Synthetic Control plot with show_hdi_annotation=True."""
        fig, ax = sc_result.plot(show_hdi_annotation=True)
        title = ax[0].get_title()
        assert "HDI" in title or "94%" in title
        plt.close(fig)

    @pytest.mark.integration
    def test_did_plot_prediction_hdi(self, did_result):
        """Test DiD plot with hdi_type='prediction'."""
        fig, ax = did_result.plot(hdi_type="prediction")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.integration
    def test_did_plot_show_annotation(self, did_result):
        """Test DiD plot with show_hdi_annotation=True."""
        fig, ax = did_result.plot(show_hdi_annotation=True)
        title = ax.get_title()
        assert "HDI" in title or "94%" in title
        plt.close(fig)

    @pytest.mark.integration
    def test_rd_plot_prediction_hdi(self, rd_result):
        """Test RD plot with hdi_type='prediction'."""
        fig, ax = rd_result.plot(hdi_type="prediction")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.integration
    def test_rd_plot_show_annotation(self, rd_result):
        """Test RD plot with show_hdi_annotation=True."""
        fig, ax = rd_result.plot(show_hdi_annotation=True)
        title = ax.get_title()
        assert "HDI" in title or "94%" in title
        plt.close(fig)

    @pytest.mark.integration
    def test_rkink_plot_prediction_hdi(self, rkink_result):
        """Test RegressionKink plot with hdi_type='prediction'."""
        fig, ax = rkink_result.plot(hdi_type="prediction")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.integration
    def test_rkink_plot_show_annotation(self, rkink_result):
        """Test RegressionKink plot with show_hdi_annotation=True."""
        fig, ax = rkink_result.plot(show_hdi_annotation=True)
        title = ax.get_title()
        assert "HDI" in title or "94%" in title
        plt.close(fig)

    @pytest.mark.integration
    def test_prepostnegd_plot_prediction_hdi(self, prepostnegd_result):
        """Test PrePostNEGD plot with hdi_type='prediction'."""
        fig, ax = prepostnegd_result.plot(hdi_type="prediction")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.integration
    def test_prepostnegd_plot_show_annotation(self, prepostnegd_result):
        """Test PrePostNEGD plot with show_hdi_annotation=True."""
        fig, ax = prepostnegd_result.plot(show_hdi_annotation=True)
        # PrePostNEGD returns a list of axes
        title = ax[0].get_title()
        assert "HDI" in title or "94%" in title
        plt.close(fig)


class TestHdiTypeEffectSummary:
    """Test hdi_type parameter for effect_summary methods."""

    @pytest.mark.integration
    def test_its_effect_summary_prediction(self, its_result):
        """Test ITS effect_summary with hdi_type='prediction'."""
        summary = its_result.effect_summary(hdi_type="prediction")
        assert summary is not None
        assert hasattr(summary, "table")

    @pytest.mark.integration
    def test_sc_effect_summary_prediction(self, sc_result):
        """Test Synthetic Control effect_summary with hdi_type='prediction'."""
        summary = sc_result.effect_summary(hdi_type="prediction")
        assert summary is not None
        assert hasattr(summary, "table")


class TestHdiTypeGetPlotData:
    """Test hdi_type parameter for get_plot_data_bayesian methods."""

    @pytest.mark.integration
    def test_its_get_plot_data_prediction(self, its_result):
        """Test ITS get_plot_data_bayesian with hdi_type='prediction'."""
        df = its_result.get_plot_data_bayesian(hdi_type="prediction")
        assert isinstance(df, pd.DataFrame)
        assert "prediction" in df.columns

    @pytest.mark.integration
    def test_sc_get_plot_data_prediction(self, sc_result):
        """Test Synthetic Control get_plot_data_bayesian with hdi_type='prediction'."""
        df = sc_result.get_plot_data_bayesian(hdi_type="prediction")
        assert isinstance(df, pd.DataFrame)
        assert "prediction" in df.columns
