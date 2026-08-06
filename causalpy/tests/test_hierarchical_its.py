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

    def test_plot_unit(self, panel, mock_pymc_sample):
        """plot_unit returns a figure for a valid unit."""
        result = _fit(panel, "instant")
        fig, (ax1, ax2) = result.plot_unit(unit_id=0)
        assert fig is not None
        assert ax1.get_title().startswith("Unit 0")
        assert ax2.get_title().startswith("Unit 0")
        plt.close(fig)

    def test_plot_unit_invalid(self, panel, mock_pymc_sample):
        """plot_unit raises ValueError for a missing unit."""
        result = _fit(panel, "instant")
        with pytest.raises(ValueError, match="not found"):
            result.plot_unit(unit_id=9999)

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

    def test_print_coefficients_does_not_crash(self, panel, mock_pymc_sample, capsys):
        """print_coefficients outputs population-level summaries without error."""
        result = _fit(panel, "instant")
        result.print_coefficients()
        out = capsys.readouterr().out
        assert "mu_lift" in out

    def test_effect_summary_direction_decrease(self, panel, mock_pymc_sample):
        """effect_summary with direction='decrease' emits prob_negative column."""
        result = _fit(panel, "instant")
        es = result.effect_summary(direction="decrease")
        assert "prob_negative" in es.table.columns

    def test_effect_summary_direction_two_sided(self, panel, mock_pymc_sample):
        """effect_summary with direction='two-sided' emits prob_nonzero column."""
        result = _fit(panel, "instant")
        es = result.effect_summary(direction="two-sided")
        assert "prob_nonzero" in es.table.columns

    def test_effect_summary_unsupported_param_warns(self, panel, mock_pymc_sample):
        """effect_summary warns when an unsupported parameter is non-default."""
        result = _fit(panel, "instant")
        with pytest.warns(UserWarning, match="not yet supported"):
            result.effect_summary(min_effect=0.5)


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

    def test_print_coefficients_event_study(self, panel, mock_pymc_sample, capsys):
        """print_coefficients on an event-study model prints mu_delta, not mu_lift."""
        result = _fit(panel, "event_study", bin_edges=[0, 4, 8, 12, 30])
        result.print_coefficients()
        out = capsys.readouterr().out
        assert "mu_delta" in out
        assert "mu_lift" not in out


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

    def test_placebo_counterfactual_keeps_pre_bins(self, panel, mock_pymc_sample):
        """Counterfactual aux keeps pre-launch D columns active, zeros post columns."""
        result = _fit(
            panel,
            "placebo",
            bin_edges=[0, 5, 10],
            placebo_edges=[-5, -2, 0],
        )
        aux_cf = result._aux(effect_on=False)
        n_pre = result._n_pre_bins
        np.testing.assert_array_equal(aux_cf["D"][:, :n_pre], result._D[:, :n_pre])
        assert (aux_cf["D"][:, n_pre:] == 0).all()


class TestSaturation:
    """Tests for the saturation (Hill-curve) effect type."""

    def test_fit_and_attributes(self, panel, mock_pymc_sample):
        """Fit saturation model and verify key posterior variables exist."""
        result = _fit(panel, "saturation")
        assert isinstance(result, cp.HierarchicalInterruptedTimeSeries)
        assert result.effect_type == "saturation"
        post = result.model.idata.posterior
        for name in ("L", "k", "s", "mu_logL", "sigma_logL", "mu_logk", "sigma_logk"):
            assert name in post
        assert result.impact.shape[-1] == len(panel)

    def test_plot(self, panel, mock_pymc_sample):
        """Plot saturation model and verify figure is returned."""
        result = _fit(panel, "saturation")
        fig, _ = result.plot(show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_ci_prob_narrower_band(self, panel, mock_pymc_sample):
        """A smaller ci_prob produces a narrower HDI band and a matching legend label."""
        result = _fit(panel, "saturation")
        fig_default, ax_default = result.plot(show=False)
        band_default = ax_default.collections[0]
        default_width = np.ptp(band_default.get_paths()[0].vertices[:, 1])

        fig_narrow, ax_narrow = result.plot(show=False, ci_prob=0.5)
        assert any(
            "50% HDI" in t.get_text() for t in ax_narrow.get_legend().get_texts()
        )
        band_narrow = ax_narrow.collections[0]
        narrow_width = np.ptp(band_narrow.get_paths()[0].vertices[:, 1])
        assert narrow_width <= default_width
        plt.close(fig_default)
        plt.close(fig_narrow)

    def test_plot_invalid_ci_prob(self, panel, mock_pymc_sample):
        """plot raises ValueError for ci_prob outside (0, 1]."""
        result = _fit(panel, "saturation")
        with pytest.raises(ValueError, match="ci_prob must be in"):
            result.plot(show=False, ci_prob=1.5)

    def test_plot_unit(self, panel, mock_pymc_sample):
        """plot_unit works for the saturation effect type (generic mu-based plot)."""
        result = _fit(panel, "saturation")
        fig, (ax1, ax2) = result.plot_unit(unit_id=0)
        assert fig is not None
        plt.close(fig)

    def test_predictive_new_unit(self, panel, mock_pymc_sample):
        """Draw (L, k, s) samples for a hypothetical new unit."""
        result = _fit(panel, "saturation")
        draws = result.predictive_for_new_unit(size=50, random_seed=0)
        assert draws.shape == (50, 3)
        assert np.isfinite(draws).all()
        # L and k must be positive (log-scale parameterization)
        assert (draws[:, 0] > 0).all()
        assert (draws[:, 1] > 0).all()

    def test_summary(self, panel, mock_pymc_sample, capsys):
        """Summary prints ceiling lift / half-saturation time / Hill exponent."""
        result = _fit(panel, "saturation")
        result.summary()
        out = capsys.readouterr().out
        assert "ceiling lift" in out
        assert "half-saturation time" in out
        assert "Hill exponent" in out

    def test_print_coefficients(self, panel, mock_pymc_sample, capsys):
        """print_coefficients prints L/k/s summaries without error."""
        result = _fit(panel, "saturation")
        result.print_coefficients()
        out = capsys.readouterr().out
        assert "ceiling lift" in out
        assert "half-saturation time" in out

    def test_effect_summary(self, panel, mock_pymc_sample):
        """Effect summary table contains L/k/s rows for saturation model."""
        result = _fit(panel, "saturation")
        es = result.effect_summary()
        assert "L (ceiling lift)" in es.table.index
        assert "k (half-saturation time)" in es.table.index
        assert "s (Hill exponent)" in es.table.index

    def test_counterfactual_zeroes_effect(self, panel, mock_pymc_sample):
        """Counterfactual aux zeroes `tau_since`, which zeroes the Hill effect entirely."""
        result = _fit(panel, "saturation")
        aux_cf = result._aux(effect_on=False)
        assert (aux_cf["tau_since"] == 0).all()
        assert "post" not in aux_cf


class TestSaturationRecovery:
    """Non-mocked recovery test for the saturation effect type."""

    @staticmethod
    def _hill(x, k, s):
        x = np.clip(x, 0, None)
        return x**s / (k**s + x**s)

    def _make_saturation_panel(self, n_units=3, T=60, L=10.0, k=8.0, s=2.5, seed=1):
        """Synthesize a panel with a known Hill-curve DGP."""
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n_units):
            launch = 20
            for t in range(T):
                post = float(t >= launch)
                tau_since = max(t - launch, 0)
                effect = post * L * self._hill(tau_since, k, s)
                y = 50 + effect + rng.normal(0, 1.5)
                rows.append(
                    {
                        "product": i,
                        "week_idx": t,
                        "launch_week": launch,
                        "sales": y,
                    }
                )
        return pd.DataFrame(rows), {"L": L, "k": k, "s": s}

    @pytest.mark.slow
    def test_recovers_saturation_parameters(self):
        """Fit with real (non-mocked) sampling and check L/k are in a plausible range."""
        df, truth = self._make_saturation_panel()
        result = cp.HierarchicalInterruptedTimeSeries(
            data=df,
            formula="sales ~ 0",
            unit_col="product",
            time_col="week_idx",
            treatment_time_col="launch_week",
            effect_type="saturation",
            model=HierarchicalLaunchITS(
                sample_kwargs={
                    "tune": 20,
                    "draws": 20,
                    "chains": 2,
                    "progressbar": False,
                    "random_seed": 42,
                }
            ),
        )
        post = result.model.idata.posterior
        L_hat = float(np.exp(post["mu_logL"]).mean())
        k_hat = float(np.exp(post["mu_logk"]).mean())
        # Loose bounds given the tiny draw count - this is a smoke test for
        # sign/indexing bugs in the Hill-curve code path, not a convergence test.
        assert 0.25 * truth["L"] < L_hat < 4 * truth["L"]
        assert 0.1 * truth["k"] < k_hat < 10 * truth["k"]


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

    def test_formula_with_intercept_dropped(self, panel, mock_pymc_sample):
        """Formula with an explicit intercept is silently dropped."""
        result = cp.HierarchicalInterruptedTimeSeries(
            data=panel,
            formula="sales ~ emails",  # patsy adds Intercept; we drop it
            unit_col="product",
            time_col="week_idx",
            treatment_time_col="launch_week",
            effect_type="instant",
            model=HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS),
        )
        assert "Intercept" not in result.labels

    def test_print_coefficients_no_covariates(self, panel, mock_pymc_sample, capsys):
        """print_coefficients on a no-covariates model skips mu_beta/sigma_beta."""
        result = cp.HierarchicalInterruptedTimeSeries(
            data=panel,
            formula="sales ~ 0",
            unit_col="product",
            time_col="week_idx",
            treatment_time_col="launch_week",
            effect_type="instant",
            model=HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS),
        )
        result.print_coefficients()
        out = capsys.readouterr().out
        assert "mu_lift" in out
        assert "mu_beta" not in out

    def test_with_seasonality(self, panel, mock_pymc_sample):
        """Fit with Fourier seasonality enabled."""
        result = _fit(panel, "instant", seasonality={"period": 20, "K": 2})
        assert "beta_season" in result.model.idata.posterior
        fourier_coord = result.model.idata.posterior.coords["fourier"].values.tolist()
        assert fourier_coord == ["f_sin_1", "f_cos_1", "f_sin_2", "f_cos_2"]
        # Design columns should be non-degenerate (have variance), not just present.
        assert np.all(result._F.std(axis=0) > 0)

    def test_time_trend_in_posterior(self, panel, mock_pymc_sample):
        """Time trend random effects appear in the posterior by default."""
        result = _fit(panel, "instant")
        post = result.model.idata.posterior
        assert "gamma" in post
        assert "mu_gamma" in post
        assert "sigma_gamma" in post
        assert post["gamma"].dims == ("chain", "draw", "unit")

    def test_ar_residuals(self, panel, mock_pymc_sample):
        """AR(1) residuals add rho and z_ar to the posterior."""
        result = cp.HierarchicalInterruptedTimeSeries(
            data=panel,
            formula="sales ~ 0 + emails + price",
            unit_col="product",
            time_col="week_idx",
            treatment_time_col="launch_week",
            effect_type="instant",
            ar_residuals=True,
            model=HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS),
        )
        post = result.model.idata.posterior
        assert "rho" in post
        assert "z_ar" in post
        assert "sigma_ar" in post
        assert post["rho"].dims == ("chain", "draw", "unit")
        assert "time_step" in post["z_ar"].dims

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

    def test_ar_residuals_unbalanced_panel(self, panel):
        """Raise ValueError when ar_residuals=True on an unbalanced panel."""
        # Drop some rows from one unit to make it unbalanced
        bad = panel[~((panel["product"] == 0) & (panel["week_idx"] > 35))].copy()
        with pytest.raises(ValueError, match="balanced panel"):
            cp.HierarchicalInterruptedTimeSeries(
                data=bad,
                formula="sales ~ 0 + emails",
                unit_col="product",
                time_col="week_idx",
                treatment_time_col="launch_week",
                effect_type="instant",
                ar_residuals=True,
                model=HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS),
            )

    def test_constant_time_column(self, mock_pymc_sample):
        """Time std == 0 falls back to 1.0 without error."""
        # All rows have the same time value → std == 0
        df = pd.DataFrame(
            {
                "product": [0, 1],
                "week_idx": [5, 5],
                "launch_week": [10, 10],
                "sales": [50.0, 55.0],
                "emails": [100.0, 110.0],
            }
        )
        result = cp.HierarchicalInterruptedTimeSeries(
            data=df,
            formula="sales ~ 0 + emails",
            unit_col="product",
            time_col="week_idx",
            treatment_time_col="launch_week",
            effect_type="instant",
            model=HierarchicalLaunchITS(sample_kwargs=SAMPLE_KWARGS),
        )
        assert result._time_std == 1.0

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

    def test_inconsistent_treatment_time_per_unit(self, panel):
        """Raise ValueError when one unit has two different launch_week values."""
        df = panel.copy()
        mask = df["product"] == 0
        last_idx = df[mask].index[-1]
        df.loc[last_idx, "launch_week"] = df.loc[last_idx, "launch_week"] + 99
        with pytest.raises(ValueError, match="not constant within units"):
            _fit(df, "instant")

    def test_ar_residuals_unsorted_panel(self, mock_pymc_sample):
        """within_unit_tidx must be identical regardless of input row order."""
        panel = _make_panel()
        shuffled = panel.sample(frac=1, random_state=99).reset_index(drop=True)
        result_sorted = _fit(panel, "instant", ar_residuals=True)
        result_shuffled = _fit(shuffled, "instant", ar_residuals=True)
        np.testing.assert_array_equal(
            result_sorted._within_unit_tidx,
            result_shuffled._within_unit_tidx,
        )

    def test_unknown_kwargs_raises(self, panel):
        """Raise TypeError when unexpected keyword arguments are passed."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            _fit(panel, "instant", seasonalty=2)  # typo: missing 'i'

    def test_bin_edges_too_short(self, panel):
        """Raise ValueError when bin_edges has fewer than 2 entries."""
        with pytest.raises(ValueError, match="at least 2 entries"):
            _fit(panel, "event_study", bin_edges=[4])

    def test_placebo_edges_too_short(self, panel):
        """Raise ValueError when placebo_edges has fewer than 2 entries."""
        with pytest.raises(ValueError, match="at least 2 entries"):
            _fit(
                panel,
                "placebo",
                bin_edges=[0, 4, 8],
                placebo_edges=[-4],
            )

    def test_placebo_edges_overlap_bin_edges(self, panel):
        """Raise ValueError when placebo_edges extend past the first bin_edges entry."""
        with pytest.raises(ValueError, match="must not overlap"):
            _fit(
                panel,
                "placebo",
                bin_edges=[0, 4, 8],
                placebo_edges=[-8, -4, 2],
            )

    def test_seasonality_missing_keys(self, panel):
        """Raise ValueError when seasonality dict is missing required keys."""
        with pytest.raises(ValueError, match="missing required key"):
            _fit(panel, "instant", seasonality={"period": 52})

    def test_seasonality_non_positive_period(self, panel):
        """Raise ValueError when seasonality['period'] is not > 0."""
        with pytest.raises(ValueError, match="period.*must be > 0"):
            _fit(panel, "instant", seasonality={"period": 0, "K": 2})

    def test_seasonality_invalid_K(self, panel):
        """Raise ValueError when seasonality['K'] is < 1."""
        with pytest.raises(ValueError, match=r"\['K'\] must be >= 1"):
            _fit(panel, "instant", seasonality={"period": 52, "K": 0})

    def test_bin_edges_cover_no_observations(self, panel):
        """Raise ValueError when bin_edges don't overlap the observed event-time range."""
        with pytest.raises(ValueError, match="No observations fall within any bin"):
            _fit(panel, "event_study", bin_edges=[1000, 1001])
