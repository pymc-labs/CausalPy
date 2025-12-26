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
"""
Tests for the plot() method's show parameter functionality.

This module tests that the plot() method correctly handles the show parameter
to control automatic plot display in Jupyter notebooks.
"""

from unittest.mock import patch

import pytest
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

import causalpy as cp

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


@pytest.mark.integration
def test_plot_show_parameter_default_true_pymc(mock_pymc_sample):
    """
    Test that plot() calls plt.show() by default (show=True) for PyMC models.

    This ensures plots auto-display in Jupyter notebooks when using
    the pattern: fig, ax = result.plot()
    """
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with patch("matplotlib.pyplot.show") as mock_show:
        fig, ax = result.plot()
        # Verify plt.show() was called (default behavior)
        mock_show.assert_called_once()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_plot_show_parameter_explicit_true_pymc(mock_pymc_sample):
    """
    Test that plot(show=True) calls plt.show() for PyMC models.
    """
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with patch("matplotlib.pyplot.show") as mock_show:
        fig, ax = result.plot(show=True)
        # Verify plt.show() was called
        mock_show.assert_called_once()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_plot_show_parameter_false_pymc(mock_pymc_sample):
    """
    Test that plot(show=False) does NOT call plt.show() for PyMC models.

    This allows users to modify the figure before displaying it manually.
    """
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with patch("matplotlib.pyplot.show") as mock_show:
        fig, ax = result.plot(show=False)
        # Verify plt.show() was NOT called
        mock_show.assert_not_called()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_plot_show_parameter_default_true_skl():
    """
    Test that plot() calls plt.show() by default (show=True) for scikit-learn models.
    """
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )

    with patch("matplotlib.pyplot.show") as mock_show:
        fig, ax = result.plot()
        # Verify plt.show() was called (default behavior)
        mock_show.assert_called_once()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)


@pytest.mark.integration
def test_plot_show_parameter_false_skl():
    """
    Test that plot(show=False) does NOT call plt.show() for scikit-learn models.
    """
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )

    with patch("matplotlib.pyplot.show") as mock_show:
        fig, ax = result.plot(show=False)
        # Verify plt.show() was NOT called
        mock_show.assert_not_called()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
