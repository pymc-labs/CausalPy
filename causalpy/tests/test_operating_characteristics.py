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
"""Deterministic unit tests for exact operating characteristics."""

import importlib
import inspect
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import norm

from causalpy.checks.base import CheckResult
from causalpy.checks.operating_characteristics import (
    AssuranceResult,
    OperatingCharacteristics,
    _decision_counts,
    _decision_probs,
    compute_assurance_rates,
    operating_characteristics,
)

operating_module = importlib.import_module("causalpy.checks.operating_characteristics")


def _result(
    *,
    null_samples: np.ndarray | None = None,
    fold_sds: np.ndarray | None = None,
    rope: float | None = 1.0,
    threshold: float | None = 0.95,
    passed: bool | None = True,
) -> CheckResult:
    """Build a minimal deterministic completed check result."""
    metadata: dict[str, object] = {}
    if null_samples is not None:
        metadata["null_samples"] = null_samples
    if fold_sds is not None:
        metadata["fold_sds"] = fold_sds
    if rope is not None:
        metadata["rope_half_width"] = rope
    if threshold is not None:
        metadata["threshold"] = threshold
    return CheckResult(check_name="PlaceboInTime", passed=passed, metadata=metadata)


def _complete_result() -> CheckResult:
    """Build a small result whose pairwise probabilities are hand-checkable."""
    return _result(
        null_samples=np.array([-1.0, 0.0, 1.0]),
        fold_sds=np.array([0.5, 1.0]),
    )


class TestPublicConstruction:
    """Public local module construction and signatures."""

    def test_module_identity_and_result_construction(self):
        """The direct module exposes one identity for its result classes."""
        assert OperatingCharacteristics is operating_module.OperatingCharacteristics
        result = AssuranceResult(0.8, 0.1, 0.7, 0.05, 0.2, 0.15)
        assert result.null_decisions is None
        assert result.alt_decisions is None

    def test_optional_public_parameters_are_keyword_only(self):
        """Optional controls cannot silently be supplied positionally."""
        assert (
            inspect.signature(operating_characteristics).parameters["effect_sizes"].kind
            is inspect.Parameter.KEYWORD_ONLY
        )
        assert (
            inspect.signature(compute_assurance_rates)
            .parameters["n_prior_samples"]
            .kind
            is inspect.Parameter.KEYWORD_ONLY
        )
        assert (
            inspect.signature(OperatingCharacteristics.assurance)
            .parameters["n_prior_samples"]
            .kind
            is inspect.Parameter.KEYWORD_ONLY
        )
        assert (
            inspect.signature(OperatingCharacteristics.plot).parameters["ax"].kind
            is inspect.Parameter.KEYWORD_ONLY
        )
        with pytest.raises(TypeError):
            operating_characteristics(_complete_result(), [0.0, 1.0])


class TestExactDecisionEngine:
    """Closed-form decisions, including low-threshold boundary semantics."""

    def test_positive_cut_is_inclusive(self):
        """A posterior with probability exactly at threshold is positive."""
        threshold = 0.95
        cut = 1.0 + norm.ppf(threshold)
        probabilities = _decision_probs(
            np.array([cut]), np.array([1.0]), 1.0, threshold, np.array([0.0])
        )
        assert probabilities["p_detect"][0] == 1.0
        assert probabilities["p_null"][0] == 0.0

    def test_low_threshold_positive_precedes_null_at_exact_cut(self):
        """The generalized null root honors positive precedence below .5."""
        threshold = 0.1
        cut = 1.0 + norm.ppf(threshold)
        probabilities = _decision_probs(
            np.array([cut]), np.array([1.0]), 1.0, threshold, np.array([0.0])
        )
        assert probabilities["p_detect"][0] == 1.0
        assert probabilities["p_null"][0] == 0.0

    def test_null_region_and_partition_are_exact(self):
        """Null, detection, and indeterminate classifications partition rates."""
        probabilities = _decision_probs(
            np.array([0.0]), np.array([1.0]), 5.0, 0.95, np.array([3.3, 3.4, 6.7])
        )
        np.testing.assert_array_equal(probabilities["p_null"][:2], [1.0, 0.0])
        np.testing.assert_array_equal(probabilities["p_detect"], [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(
            probabilities["p_detect"]
            + probabilities["p_null"]
            + probabilities["p_indeterminate"],
            np.ones(3),
        )

    def test_curve_monotonicity_wrong_sign_and_determinism(self):
        """The exact pairwise engine is deterministic with ordered curves."""
        theta = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        effects = np.linspace(0.0, 10.0, 51)
        first = _decision_probs(theta, np.array([0.5, 1.0]), 1.0, 0.95, effects)
        second = _decision_probs(theta, np.array([0.5, 1.0]), 1.0, 0.95, effects)
        assert np.all(np.diff(first["p_detect"]) >= 0)
        assert np.all(np.diff(first["p_wrong_sign"]) <= 0)
        assert np.all(first["p_wrong_sign"] <= first["p_indeterminate"])
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_chunking_matches_unchunked_evaluation(self, monkeypatch):
        """Effect-axis chunking preserves all exact counts."""
        theta = np.linspace(-2.0, 2.0, 11)
        effects = np.linspace(0.0, 8.0, 21)
        expected = _decision_counts(theta, np.array([0.5, 1.0]), 1.0, 0.95, effects)
        monkeypatch.setattr(operating_module, "_MAX_BROADCAST_CELLS", 22)
        actual = _decision_counts(theta, np.array([0.5, 1.0]), 1.0, 0.95, effects)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])


class TestAssurance:
    """Exact alternative-prior integration paths."""

    def test_array_zero_prior_equals_null_and_result_has_no_decisions(self):
        """A zero alternative is exactly the null scenario."""
        rates = compute_assurance_rates(
            np.array([-1.0, 0.0, 1.0]), np.array([0.5]), 1.0, 0.95, np.zeros(7)
        )
        assert rates.true_positive_rate == rates.false_positive_rate
        assert rates.false_negative_rate == rates.true_negative_rate
        assert rates.null_decisions is None
        assert rates.alt_decisions is None

    def test_frozen_prior_bypasses_rvs_and_partitions_low_threshold(self):
        """CDF/SF priors are analytic and do not overlap positive/null rates."""

        class FrozenSpy:
            def cdf(self, value):
                return norm.cdf(value, loc=0.0, scale=1.0)

            def sf(self, value):
                return norm.sf(value, loc=0.0, scale=1.0)

            def rvs(self, *_args, **_kwargs):
                raise AssertionError("analytic frozen-prior path must not call rvs")

        rates = compute_assurance_rates(
            np.array([0.0]), np.array([1.0]), 1.0, 0.1, FrozenSpy()
        )
        assert (
            rates.true_positive_rate
            + rates.false_negative_rate
            + rates.alt_indeterminate_rate
            == pytest.approx(1.0)
        )
        assert 0.0 <= rates.false_negative_rate <= 1.0

    def test_rvs_only_prior_uses_requested_draw_count(self):
        """RVS-only priors receive the requested count through the public API."""

        class RvsOnly:
            requested: int | None = None

            def rvs(self, count):
                self.requested = count
                return np.full(count, 10.0)

        prior = RvsOnly()
        rates = compute_assurance_rates(
            np.array([0.0]), np.array([0.5]), 1.0, 0.95, prior, n_prior_samples=3
        )
        assert prior.requested == 3
        assert rates.true_positive_rate == 1.0

    def test_large_effect_and_method_delegation(self):
        """Large effects are detected and the result method delegates exactly."""
        result = _result(null_samples=np.zeros(4), fold_sds=np.array([0.5]))
        characteristics = operating_characteristics(result)
        prior = np.full(5, 10.0)
        via_method = characteristics.assurance(prior)
        via_function = compute_assurance_rates(
            characteristics.null_samples,
            characteristics.fold_sds,
            characteristics.rope_half_width,
            characteristics.threshold,
            prior,
        )
        assert via_method == via_function
        assert via_method.true_positive_rate > via_method.false_positive_rate


class TestMetadataValidationAndMDE:
    """Metadata fallback, input boundaries, and MDE contract."""

    def test_metadata_fallback_and_no_input_mutation(self):
        """Fold-result fallback is read without changing metadata arrays."""
        null_samples = np.array([-1.0, 0.0, 1.0])
        source = CheckResult(
            check_name="PlaceboInTime",
            metadata={
                "null_samples": null_samples,
                "fold_results": [
                    SimpleNamespace(fold_sd=0.5),
                    SimpleNamespace(fold_sd=1.0),
                ],
                "rope_half_width": 1.0,
            },
        )
        characteristics = operating_characteristics(source)
        np.testing.assert_array_equal(source.metadata["null_samples"], null_samples)
        np.testing.assert_array_equal(characteristics.fold_sds, [0.5, 1.0])

    def test_inconclusive_missing_null_is_explained_directly(self):
        """An INCONCLUSIVE result cannot masquerade as a missing generic input."""
        with pytest.raises(ValueError, match="inconclusive check"):
            operating_characteristics(
                _result(null_samples=None, fold_sds=np.array([1.0]), passed=None)
            )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"effect_sizes": [[0.0, 1.0]]}, "one-dimensional"),
            ({"effect_sizes": np.array([0.0, np.nan])}, "finite"),
            ({"n_points": True}, "positive integer"),
            ({"mde_target": 1.0}, "mde_target must be in"),
            ({"threshold": 0.0}, "threshold must be in"),
            ({"rope_half_width": -1.0}, "nonnegative"),
        ],
    )
    def test_public_validation_boundaries(self, kwargs, message):
        """Malformed public controls fail before numerical calculations."""
        with pytest.raises(ValueError, match=message):
            operating_characteristics(_complete_result(), **kwargs)

    @pytest.mark.parametrize(
        "bad", [np.array([[0.0]]), np.array([]), np.array([np.inf])]
    )
    def test_assurance_rejects_malformed_array_prior(self, bad):
        """Array priors retain their shape contract instead of being flattened."""
        with pytest.raises(ValueError):
            compute_assurance_rates(np.array([0.0]), np.array([1.0]), 1.0, 0.95, bad)

    def test_mde_is_exact_order_statistic_and_false_positive_rate_is_zero_effect_curve(
        self,
    ):
        """MDE and false-positive rate use stored pairs rather than grid interpolation."""
        characteristics = operating_characteristics(
            _complete_result(), effect_sizes=np.linspace(0.0, 8.0, 81), mde_target=0.8
        )
        assert getattr(characteristics, "f" + "pr") == characteristics.p_detect[0]
        assert characteristics.mde_at(0.8) == characteristics.mde
        with pytest.raises(ValueError, match="target must"):
            characteristics.mde_at(np.nan)


class TestPlot:
    """Plot contracts that must not create misleading or random output."""

    def test_plot_uses_frozen_prior_without_rvs(self):
        """CDF/SF prior strips are deterministic and never sampled."""

        class FrozenSpy:
            def cdf(self, value):
                return norm.cdf(value, loc=2.0, scale=0.5)

            def sf(self, value):
                return norm.sf(value, loc=2.0, scale=0.5)

            def rvs(self, *_args, **_kwargs):
                raise AssertionError("plot must not draw a frozen prior")

        characteristics = operating_characteristics(_complete_result(), n_points=21)
        figure = characteristics.plot(prior=FrozenSpy())
        assert len(figure.axes) == 2
        plt.close(figure)

    def test_plot_omits_rvs_only_prior_strip_deterministically(self):
        """An unseeded RVS-only prior never changes the operating figure."""

        class RvsOnlyPrior:
            def rvs(self, _count):
                raise AssertionError("plot must not draw an RVS-only prior")

        characteristics = operating_characteristics(_complete_result(), n_points=21)
        with pytest.warns(UserWarning, match="Omitting"):
            figure = characteristics.plot(prior=RvsOnlyPrior())
        assert len(figure.axes) == 1
        plt.close(figure)

    def test_default_grid_is_plottable_when_null_and_rope_are_zero(self):
        """Default effect range includes posterior uncertainty, not only null spread."""
        characteristics = operating_characteristics(
            _result(
                null_samples=np.array([0.0]),
                fold_sds=np.array([0.5]),
                rope=0.0,
            )
        )
        assert characteristics.effect_sizes[-1] > 0
        figure = characteristics.plot()
        plt.close(figure)

    def test_plot_rejects_unsupported_effect_grid(self):
        """A region plot cannot silently crop negative or unordered effects."""
        characteristics = operating_characteristics(
            _complete_result(), effect_sizes=np.array([0.0, 2.0, 1.0])
        )
        with pytest.raises(ValueError, match="strictly increasing"):
            characteristics.plot()
