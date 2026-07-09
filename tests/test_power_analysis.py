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
"""Tests for causalpy.checks.power_analysis module."""

import numpy as np
import pytest

from causalpy.checks.base import CheckResult
from causalpy.checks.power_analysis import (
    LogisticFit,
    PowerCurveResult,
    _logistic,
    _simulate_detection_rate,
    power_analysis,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pit_result(
    null_mean: float = 0.0,
    null_std: float = 50.0,
    fold_sd_mean: float = 30.0,
    n_null_samples: int = 4000,
    n_folds: int = 4,
    rope_half_width: float = 20.0,
    threshold: float = 0.95,
) -> CheckResult:
    """Create a mock PlaceboInTime CheckResult with realistic metadata."""
    rng = np.random.default_rng(42)
    null_samples = rng.normal(null_mean, null_std, size=n_null_samples)
    fold_sds = rng.uniform(fold_sd_mean * 0.5, fold_sd_mean * 1.5, size=n_folds)

    return CheckResult(
        check_name="PlaceboInTime",
        passed=True,
        text="Mock PlaceboInTime result",
        metadata={
            "null_samples": null_samples,
            "fold_sds": fold_sds,
            "rope_half_width": rope_half_width,
            "threshold": threshold,
            "actual_cumulative_mean": 150.0,
            "p_effect_outside_null": 0.97,
        },
    )


# ---------------------------------------------------------------------------
# Tests for _logistic helper
# ---------------------------------------------------------------------------


class TestLogistic:
    """Tests for the logistic function."""

    def test_midpoint_is_half(self):
        """At x=x0, logistic should return 0.5."""
        assert _logistic(np.array([5.0]), k=1.0, x0=5.0) == pytest.approx(0.5)

    def test_monotonically_increasing(self):
        """Logistic should be monotonically increasing for k > 0."""
        x = np.linspace(0, 10, 100)
        y = _logistic(x, k=1.0, x0=5.0)
        assert np.all(np.diff(y) > 0)

    def test_bounds(self):
        """Logistic should be bounded in [0, 1]."""
        x = np.linspace(-100, 100, 1000)
        y = _logistic(x, k=0.5, x0=0.0)
        assert np.all(y >= 0)
        assert np.all(y <= 1)

    def test_steepness(self):
        """Higher k should produce steeper transition."""
        x = np.array([5.5])
        y_shallow = _logistic(x, k=1.0, x0=5.0)
        y_steep = _logistic(x, k=10.0, x0=5.0)
        assert y_steep > y_shallow


# ---------------------------------------------------------------------------
# Tests for LogisticFit
# ---------------------------------------------------------------------------


class TestLogisticFit:
    """Tests for the LogisticFit dataclass."""

    def test_predict(self):
        """predict() should match _logistic."""
        fit = LogisticFit(k=0.05, x0=100.0)
        x = np.linspace(0, 200, 50)
        expected = _logistic(x, 0.05, 100.0)
        np.testing.assert_allclose(fit.predict(x), expected)

    def test_mde_at_50_percent(self):
        """MDE at 50% power should equal x0 (midpoint)."""
        fit = LogisticFit(k=0.05, x0=100.0)
        assert fit.mde(0.5) == pytest.approx(100.0, rel=1e-10)

    def test_mde_at_80_percent(self):
        """MDE at 80% should be > x0."""
        fit = LogisticFit(k=0.05, x0=100.0)
        mde_80 = fit.mde(0.80)
        assert mde_80 > 100.0

    def test_mde_increases_with_power(self):
        """Higher power threshold should require larger effect."""
        fit = LogisticFit(k=0.05, x0=100.0)
        assert fit.mde(0.90) > fit.mde(0.80) > fit.mde(0.50)

    def test_mde_invalid_threshold(self):
        """Should raise ValueError for invalid power_threshold."""
        fit = LogisticFit(k=0.05, x0=100.0)
        with pytest.raises(ValueError, match="power_threshold must be in"):
            fit.mde(0.0)
        with pytest.raises(ValueError, match="power_threshold must be in"):
            fit.mde(1.0)


# ---------------------------------------------------------------------------
# Tests for _simulate_detection_rate
# ---------------------------------------------------------------------------


class TestSimulateDetectionRate:
    """Tests for the detection rate simulation."""

    def test_zero_effect_low_detection(self):
        """At effect_size=0, detection should be low (near FPR)."""
        rng = np.random.default_rng(123)
        null_samples = rng.normal(0, 50, size=2000)
        fold_sds = np.array([30.0, 35.0, 25.0])

        rate = _simulate_detection_rate(
            effect_size=0.0,
            null_samples=null_samples,
            fold_sds=fold_sds,
            rope_half_width=20.0,
            threshold=0.95,
            n_simulations=500,
            n_posterior_samples=1000,
            rng=rng,
        )
        # Should be low — mostly null or indeterminate
        assert rate < 0.30

    def test_large_effect_high_detection(self):
        """At very large effect, detection should be high."""
        rng = np.random.default_rng(456)
        null_samples = rng.normal(0, 50, size=2000)
        fold_sds = np.array([30.0, 35.0, 25.0])

        rate = _simulate_detection_rate(
            effect_size=500.0,
            null_samples=null_samples,
            fold_sds=fold_sds,
            rope_half_width=20.0,
            threshold=0.95,
            n_simulations=500,
            n_posterior_samples=1000,
            rng=rng,
        )
        assert rate > 0.80

    def test_monotonic_in_effect_size(self):
        """Detection rate should generally increase with effect size."""
        rng = np.random.default_rng(789)
        null_samples = rng.normal(0, 50, size=2000)
        fold_sds = np.array([30.0, 35.0, 25.0])

        effects = [0, 50, 100, 200, 400]
        rates = []
        for eff in effects:
            rate = _simulate_detection_rate(
                effect_size=eff,
                null_samples=null_samples,
                fold_sds=fold_sds,
                rope_half_width=20.0,
                threshold=0.95,
                n_simulations=300,
                n_posterior_samples=1000,
                rng=np.random.default_rng(789 + eff),
            )
            rates.append(rate)

        # Not strictly monotonic due to MC noise, but overall trend
        assert rates[-1] > rates[0]
        assert rates[-1] > 0.70


# ---------------------------------------------------------------------------
# Tests for power_analysis (grid strategy)
# ---------------------------------------------------------------------------


class TestPowerAnalysisGrid:
    """Tests for the grid strategy."""

    def test_basic_grid(self):
        """Grid strategy should return correct structure."""
        pit_result = _make_pit_result()
        result = power_analysis(
            pit_result,
            effect_sizes=[0, 50, 100, 150, 200],
            n_simulations=100,
            strategy="grid",
            random_seed=42,
        )

        assert isinstance(result, PowerCurveResult)
        assert result.strategy == "grid"
        assert len(result.effect_sizes) == 5
        assert len(result.detection_rates) == 5
        assert result.fitted_curve is None
        assert result.mde is None
        assert result.n_simulations == 100

    def test_grid_detection_increases(self):
        """Detection rate should increase with effect size."""
        pit_result = _make_pit_result()
        result = power_analysis(
            pit_result,
            effect_sizes=[0, 50, 100, 200, 400],
            n_simulations=200,
            strategy="grid",
            random_seed=42,
        )

        # Overall trend should be increasing
        assert result.detection_rates[-1] > result.detection_rates[0]

    def test_grid_default_effect_sizes(self):
        """Should generate default effect sizes when None."""
        pit_result = _make_pit_result()
        result = power_analysis(
            pit_result,
            effect_sizes=None,
            n_simulations=50,
            strategy="grid",
            random_seed=42,
        )

        assert len(result.effect_sizes) == 8  # default is 8 points

    def test_grid_reproducible(self):
        """Same seed should give same results."""
        pit_result = _make_pit_result()
        r1 = power_analysis(
            pit_result,
            effect_sizes=[0, 100, 200],
            n_simulations=100,
            strategy="grid",
            random_seed=42,
        )
        r2 = power_analysis(
            pit_result,
            effect_sizes=[0, 100, 200],
            n_simulations=100,
            strategy="grid",
            random_seed=42,
        )
        np.testing.assert_array_equal(r1.detection_rates, r2.detection_rates)


# ---------------------------------------------------------------------------
# Tests for power_analysis (sigmoid strategy)
# ---------------------------------------------------------------------------


class TestPowerAnalysisSigmoid:
    """Tests for the sigmoid strategy."""

    def test_basic_sigmoid(self):
        """Sigmoid strategy should return fitted curve and MDE."""
        pit_result = _make_pit_result()
        result = power_analysis(
            pit_result,
            effect_sizes=[0, 400],
            n_simulations=200,
            strategy="sigmoid",
            n_evaluation_points=7,
            random_seed=42,
        )

        assert isinstance(result, PowerCurveResult)
        assert result.strategy == "sigmoid"
        assert len(result.effect_sizes) == 7
        assert result.fitted_curve is not None
        assert result.smooth_effect_sizes is not None
        assert result.smooth_detection_rates is not None
        assert result.mde is not None
        assert result.mde > 0

    def test_sigmoid_mde_reasonable(self):
        """MDE should be within the evaluated range for well-behaved data."""
        pit_result = _make_pit_result(null_std=50.0, rope_half_width=20.0)
        result = power_analysis(
            pit_result,
            effect_sizes=[0, 500],
            n_simulations=300,
            strategy="sigmoid",
            n_evaluation_points=7,
            random_seed=42,
        )

        # MDE should be positive and within a reasonable range
        assert result.mde is not None
        assert result.mde > 0
        assert result.mde < 500  # within our range

    def test_sigmoid_smooth_curve_shape(self):
        """Smooth curve should be monotonically increasing."""
        pit_result = _make_pit_result()
        result = power_analysis(
            pit_result,
            effect_sizes=[0, 400],
            n_simulations=200,
            strategy="sigmoid",
            n_evaluation_points=7,
            random_seed=42,
        )

        assert result.smooth_detection_rates is not None
        # Should be monotonically non-decreasing
        diffs = np.diff(result.smooth_detection_rates)
        assert np.all(diffs >= -1e-10)  # allow tiny numerical noise

    def test_sigmoid_default_range(self):
        """Should use default range when effect_sizes is None."""
        pit_result = _make_pit_result()
        result = power_analysis(
            pit_result,
            effect_sizes=None,
            n_simulations=100,
            strategy="sigmoid",
            n_evaluation_points=5,
            random_seed=42,
        )

        assert len(result.effect_sizes) == 5
        assert result.fitted_curve is not None

    def test_sigmoid_fewer_evaluations_than_grid(self):
        """Sigmoid with 5 points should be faster than grid with 20 points.

        (This is a structural test — we just verify the point counts.)
        """
        pit_result = _make_pit_result()
        grid_result = power_analysis(
            pit_result,
            effect_sizes=np.linspace(0, 400, 20).tolist(),
            n_simulations=50,
            strategy="grid",
            random_seed=42,
        )
        sigmoid_result = power_analysis(
            pit_result,
            effect_sizes=[0, 400],
            n_simulations=50,
            strategy="sigmoid",
            n_evaluation_points=5,
            random_seed=42,
        )

        assert len(grid_result.effect_sizes) == 20
        assert len(sigmoid_result.effect_sizes) == 5
        # Sigmoid should still produce a smooth curve with 200 points
        assert len(sigmoid_result.smooth_effect_sizes) == 200


# ---------------------------------------------------------------------------
# Tests for error handling
# ---------------------------------------------------------------------------


class TestPowerAnalysisErrors:
    """Tests for error handling."""

    def test_missing_null_samples(self):
        """Should raise ValueError if no null_samples in metadata."""
        bad_result = CheckResult(
            check_name="PlaceboInTime",
            passed=True,
            metadata={"fold_sds": [30.0]},
        )
        with pytest.raises(ValueError, match="does not contain a learned null"):
            power_analysis(bad_result)

    def test_missing_rope(self):
        """Should raise ValueError if no rope_half_width."""
        bad_result = CheckResult(
            check_name="PlaceboInTime",
            passed=True,
            metadata={
                "null_samples": np.zeros(100),
                "fold_sds": np.array([30.0]),
            },
        )
        with pytest.raises(ValueError, match="no rope_half_width"):
            power_analysis(bad_result)

    def test_invalid_strategy(self):
        """Should raise ValueError for unknown strategy."""
        pit_result = _make_pit_result()
        with pytest.raises(ValueError, match="strategy must be"):
            power_analysis(pit_result, strategy="unknown")


# ---------------------------------------------------------------------------
# Tests for PowerCurveResult.plot
# ---------------------------------------------------------------------------


class TestPowerCurvePlot:
    """Tests for the plot method."""

    def test_plot_grid_returns_figure(self):
        """Grid plot should return a matplotlib figure."""
        pit_result = _make_pit_result()
        result = power_analysis(
            pit_result,
            effect_sizes=[0, 50, 100, 200, 400],
            n_simulations=50,
            strategy="grid",
            random_seed=42,
        )
        import matplotlib

        matplotlib.use("Agg")
        fig = result.plot()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_sigmoid_returns_figure(self):
        """Sigmoid plot should return a figure with MDE annotation."""
        pit_result = _make_pit_result()
        result = power_analysis(
            pit_result,
            effect_sizes=[0, 400],
            n_simulations=100,
            strategy="sigmoid",
            n_evaluation_points=5,
            random_seed=42,
        )
        import matplotlib

        matplotlib.use("Agg")
        fig = result.plot()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_custom_axes(self):
        """Should accept custom axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pit_result = _make_pit_result()
        result = power_analysis(
            pit_result,
            effect_sizes=[0, 100, 200],
            n_simulations=50,
            strategy="grid",
            random_seed=42,
        )
        fig, ax = plt.subplots()
        returned_fig = result.plot(ax=ax)
        assert returned_fig is fig
        plt.close(fig)
