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
"""Tests for HierarchicalInterruptedTimeSeries and HierarchicalLaunchITS."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import causalpy as cp
from causalpy.pymc_models import HierarchicalLaunchITS

SAMPLE_KWARGS = {"progressbar": False, "random_seed": 42}


def _make_panel(n_units: int = 4, T: int = 40, true_lift: float = 8.0, seed: int = 0):
    """Build a small synthetic panel for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        launch = rng.integers(10, T - 10)
        for t in range(T):
            emails = rng.normal(100, 10)
            price = rng.normal(20, 1)
            post = float(t >= launch)
            y = 50 + 0.2 * emails - 0.5 * price + true_lift * post + rng.normal(0, 2)
            rows.append(
                {
                    "product": i,
                    "week_idx": t,
                    "launch_week": launch,
                    "sales": y,
                    "emails": emails,
                    "price": price,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def panel():
    """Return a small staggered-launch panel fixture."""
    return _make_panel()


def _fit(df, effect_type, **kwargs):
    """Instantiate and fit a HierarchicalInterruptedTimeSeries experiment."""
    return cp.HierarchicalInterruptedTimeSeries(
        data=df,
        formula="sales ~ 0 + emails + price",
        unit_col="product",
        time_col="week_idx",
        treatment_time_col="launch_week",
        effect_type=effect_type,
        model=HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS),
        **kwargs,
    )


class TestInstant:
    """Tests for the instant (single post-launch shift) effect type."""

    def test_fit_and_attributes(self, panel, mock_pymc_sample):
        """Fit instant model and verify key attributes exist."""
        result = _fit(panel, "instant")
        assert isinstance(result, cp.HierarchicalInterruptedTimeSeries)
        assert result.effect_type == "instant"
        assert result._n_units == 4
        assert "lift" in result.model.idata.posterior
        assert "mu_lift" in result.model.idata.posterior
        assert result.impact.shape[-1] == len(panel)

    def test_plot(self, panel, mock_pymc_sample):
        """Plot instant model and verify figure is returned."""
        result = _fit(panel, "instant")
        fig, _ = result.plot(show=False)
        assert fig is not None
        plt.close(fig)

    def test_predictive_new_unit(self, panel, mock_pymc_sample):
        """Draw from predictive distribution for a new unit (instant)."""
        result = _fit(panel, "instant")
        draws = result.predictive_for_new_unit(size=50, random_seed=0)
        assert draws.shape == (50,)
        assert np.isfinite(draws).all()

    def test_effect_summary(self, panel, mock_pymc_sample):
        """Effect summary table contains mu_lift for instant model."""
        result = _fit(panel, "instant")
        es = result.effect_summary()
        assert "mu_lift" in es.table.index

    def test_summary(self, panel, mock_pymc_sample, capsys):
        """Summary prints experiment info and mu_lift for instant model."""
        result = _fit(panel, "instant")
        result.summary()
        out = capsys.readouterr().out
        assert "instant" in out
        assert "mu_lift" in out.lower() or "E[mu_lift]" in out


class TestEventStudy:
    """Tests for the event-study (binned post-launch) effect type."""

    def test_fit(self, panel, mock_pymc_sample):
        """Fit event-study model and verify delta dimensions and plot."""
        result = _fit(panel, "event_study", bin_edges=[0, 4, 8, 12, 30])
        post = result.model.idata.posterior
        assert tuple(post["delta"].dims[-2:]) == ("unit", "event_bin")
        assert post["delta"].sizes["event_bin"] == 4
        fig, _ = result.plot(show=False)
        plt.close(fig)

    def test_predictive_and_summary(self, panel, mock_pymc_sample):
        """Draw from event-study predictive and check summary table."""
        result = _fit(panel, "event_study", bin_edges=[0, 4, 8, 12, 30])
        draws = result.predictive_for_new_unit(size=20, random_seed=1)
        assert draws.shape == (20, 4)
        es = result.effect_summary()
        assert any("mu_delta" in name for name in es.table.index)

    def test_summary(self, panel, mock_pymc_sample, capsys):
        """Summary prints bin-level mu_delta for event-study model."""
        result = _fit(panel, "event_study", bin_edges=[0, 4, 8, 12, 30])
        result.summary()
        out = capsys.readouterr().out
        assert "mu_delta" in out
        assert "Placebo check" not in out  # only placebo gets the check


class TestPlacebo:
    """Tests for the placebo (pre-launch lead) effect type."""

    def test_fit(self, panel, mock_pymc_sample):
        """Fit placebo model and verify pre/post bin counts and check text."""
        result = _fit(
            panel,
            "placebo",
            bin_edges=[0, 4, 8, 30],
            placebo_edges=[-8, -4, 0],
        )
        assert result._n_pre_bins == 2
        assert result._n_post_bins == 3
        # The placebo check text should report a result
        text = result._placebo_check_text()
        assert "Placebo check" in text

    def test_summary(self, panel, mock_pymc_sample, capsys):
        """Summary prints bin-level mu_delta and placebo check for placebo model."""
        result = _fit(
            panel,
            "placebo",
            bin_edges=[0, 4, 8, 30],
            placebo_edges=[-8, -4, 0],
        )
        result.summary()
        out = capsys.readouterr().out
        assert "mu_delta" in out
        assert "Placebo check" in out

    def test_plot(self, panel, mock_pymc_sample):
        """Plot placebo model exercises the placebo branch of _plot_event_study."""
        result = _fit(
            panel,
            "placebo",
            bin_edges=[0, 4, 8, 30],
            placebo_edges=[-8, -4, 0],
        )
        fig, _ = result.plot(show=False)
        assert fig is not None
        plt.close(fig)

    def test_effect_summary(self, panel, mock_pymc_sample):
        """Effect summary for placebo model includes per-bin mu_delta rows."""
        result = _fit(
            panel,
            "placebo",
            bin_edges=[0, 4, 8, 30],
            placebo_edges=[-8, -4, 0],
        )
        es = result.effect_summary()
        assert any("mu_delta" in name for name in es.table.index)
        # Should have rows for pre + post bins
        assert len(es.table) == result._n_pre_bins + result._n_post_bins


class TestEdgeCases:
    """Tests for edge-case branches."""

    def test_placebo_check_no_pre_bins(self, panel, mock_pymc_sample):
        """_placebo_check_text returns early message when no pre-launch bins."""
        result = _fit(panel, "event_study", bin_edges=[0, 4, 8, 12, 30])
        text = result._placebo_check_text()
        assert "no pre-launch bins" in text

    def test_no_covariates(self, panel, mock_pymc_sample):
        """Fit with a formula that has no covariates (empty X)."""
        result = cp.HierarchicalInterruptedTimeSeries(
            data=panel,
            formula="sales ~ 0",
            unit_col="product",
            time_col="week_idx",
            treatment_time_col="launch_week",
            effect_type="instant",
            model=HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS),
        )
        assert result.X.sizes["coeffs"] == 0

    def test_with_seasonality(self, panel, mock_pymc_sample):
        """Fit with Fourier seasonality enabled."""
        result = _fit(panel, "instant", seasonality={"period": 20, "K": 2})
        assert "beta_season" in result.model.idata.posterior

    def test_time_trend_in_posterior(self, panel, mock_pymc_sample):
        """Time trend random effects appear in the posterior by default."""
        result = _fit(panel, "instant")
        post = result.model.idata.posterior
        assert "gamma" in post
        assert "mu_gamma" in post
        assert "sigma_gamma" in post
        assert post["gamma"].dims == ("chain", "draw", "unit")

    def test_predictive_unfitted_model(self, panel):
        """Raise RuntimeError when calling predictive on an unfitted model."""
        # Build experiment object without fitting by manually constructing
        model = HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS)
        with pytest.raises(RuntimeError, match="not fitted"):
            # Manually create the experiment skeleton and call predictive
            result = cp.HierarchicalInterruptedTimeSeries.__new__(
                cp.HierarchicalInterruptedTimeSeries
            )
            result.model = model
            result.effect_type = "instant"
            result.predictive_for_new_unit()


class TestValidation:
    """Tests for input validation and error messages."""

    def test_non_numeric_time_column(self, panel):
        """Raise ValueError when time column is not numeric."""
        bad = panel.copy()
        bad["week_idx"] = bad["week_idx"].astype(str)
        with pytest.raises(ValueError, match="must be numeric"):
            cp.HierarchicalInterruptedTimeSeries(
                data=bad,
                formula="sales ~ 0 + emails",
                unit_col="product",
                time_col="week_idx",
                treatment_time_col="launch_week",
                model=HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS),
            )

    def test_non_numeric_treatment_time_column(self, panel):
        """Raise ValueError when treatment_time column is not numeric."""
        bad = panel.copy()
        bad["launch_week"] = bad["launch_week"].astype(str)
        with pytest.raises(ValueError, match="must be numeric"):
            cp.HierarchicalInterruptedTimeSeries(
                data=bad,
                formula="sales ~ 0 + emails",
                unit_col="product",
                time_col="week_idx",
                treatment_time_col="launch_week",
                model=HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS),
            )

    def test_invalid_effect_type(self, panel):
        """Raise ValueError for an unrecognised effect_type."""
        with pytest.raises(ValueError, match="effect_type must be"):
            cp.HierarchicalInterruptedTimeSeries(
                data=panel,
                formula="sales ~ 0 + emails",
                unit_col="product",
                time_col="week_idx",
                treatment_time_col="launch_week",
                effect_type="bogus",
                model=HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS),
            )

    def test_unsorted_placebo_edges(self, panel):
        """Raise ValueError when placebo_edges are not sorted."""
        with pytest.raises(ValueError, match="sorted"):
            cp.HierarchicalInterruptedTimeSeries(
                data=panel,
                formula="sales ~ 0 + emails",
                unit_col="product",
                time_col="week_idx",
                treatment_time_col="launch_week",
                effect_type="placebo",
                bin_edges=[0, 4, 8],
                placebo_edges=[0, -4, -8],
                model=HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS),
            )

    def test_outcome_not_in_data(self, panel):
        """Raise ValueError when formula outcome column is missing."""
        with pytest.raises(ValueError, match="Outcome variable"):
            cp.HierarchicalInterruptedTimeSeries(
                data=panel,
                formula="nonexistent ~ 0 + emails",
                unit_col="product",
                time_col="week_idx",
                treatment_time_col="launch_week",
                model=HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS),
            )

    def test_missing_column(self, panel):
        """Raise ValueError when a required column is absent."""
        with pytest.raises(ValueError, match="Missing required columns"):
            cp.HierarchicalInterruptedTimeSeries(
                data=panel,
                formula="sales ~ 0 + emails",
                unit_col="nope",
                time_col="week_idx",
                treatment_time_col="launch_week",
                model=HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS),
            )

    def test_event_study_requires_bins(self, panel):
        """Raise ValueError when event_study lacks bin_edges."""
        with pytest.raises(ValueError, match="requires `bin_edges`"):
            cp.HierarchicalInterruptedTimeSeries(
                data=panel,
                formula="sales ~ 0 + emails",
                unit_col="product",
                time_col="week_idx",
                treatment_time_col="launch_week",
                effect_type="event_study",
                model=HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS),
            )

    def test_placebo_requires_placebo_edges(self, panel):
        """Raise ValueError when placebo lacks placebo_edges."""
        with pytest.raises(ValueError, match="placebo_edges"):
            cp.HierarchicalInterruptedTimeSeries(
                data=panel,
                formula="sales ~ 0 + emails",
                unit_col="product",
                time_col="week_idx",
                treatment_time_col="launch_week",
                effect_type="placebo",
                bin_edges=[0, 4, 8],
                model=HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS),
            )

    def test_unsorted_bins(self, panel):
        """Raise ValueError when bin_edges are not sorted."""
        with pytest.raises(ValueError, match="sorted"):
            cp.HierarchicalInterruptedTimeSeries(
                data=panel,
                formula="sales ~ 0 + emails",
                unit_col="product",
                time_col="week_idx",
                treatment_time_col="launch_week",
                effect_type="event_study",
                bin_edges=[4, 0, 8],
                model=HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS),
            )

    def test_rejects_non_hierarchical_model(self, panel):
        """Raise TypeError when model is not HierarchicalLaunchITS."""
        with pytest.raises(TypeError, match="HierarchicalLaunchITS"):
            cp.HierarchicalInterruptedTimeSeries(
                data=panel,
                formula="sales ~ 0 + emails",
                unit_col="product",
                time_col="week_idx",
                treatment_time_col="launch_week",
                model=cp.pymc_models.LinearRegression(sample_kwargs=SAMPLE_KWARGS),
            )
