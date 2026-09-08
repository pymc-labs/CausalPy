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
"""Exact operating characteristics for placebo-calibrated designs.

For a simulated posterior ``Normal(theta + effect, sigma)``, this module
computes ROPE decisions exactly over every learned-null/fold-SD pair.  It does
not simulate posterior draws or mutate the result metadata used as its input.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Protocol, cast

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm

from causalpy.checks.base import CheckResult

# Maximum number of (effect, theta, sigma) cells evaluated in one broadcast.
_MAX_BROADCAST_CELLS = 20_000_000


class _FrozenPrior(Protocol):
    """Distribution protocol for exact CDF/SF integration."""

    def cdf(self, value: np.ndarray) -> np.ndarray: ...

    def sf(self, value: np.ndarray) -> np.ndarray: ...


@dataclass
class AssuranceResult:
    """Closed-form operating rates under null and alternative scenarios.

    The raw decision arrays are always ``None``: exact calculations have no
    simulated replications to retain.
    """

    true_positive_rate: float
    false_positive_rate: float
    true_negative_rate: float
    false_negative_rate: float
    null_indeterminate_rate: float
    alt_indeterminate_rate: float
    null_decisions: np.ndarray | None = field(default=None, repr=False)
    alt_decisions: np.ndarray | None = field(default=None, repr=False)


def _finite_scalar(name: str, value: object) -> float:
    """Return a finite real scalar or raise a descriptive ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar, got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _finite_1d(name: str, values: object, *, positive: bool = False) -> np.ndarray:
    """Validate a finite, nonempty one-dimensional numeric array."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"{name} must be a finite, nonempty one-dimensional array."
        ) from err
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a finite, nonempty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if positive and not np.all(array > 0):
        raise ValueError(f"{name} must contain only strictly positive values.")
    return array


def _validate_inputs(
    null_samples: object, fold_sds: object, rope_half_width: object, threshold: object
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Validate common exact-engine inputs before numerical calculations."""
    theta = _finite_1d("null_samples", null_samples)
    sigmas = _finite_1d("fold_sds", fold_sds, positive=True)
    rope = _finite_scalar("rope_half_width", rope_half_width)
    if rope < 0:
        raise ValueError(
            f"rope_half_width must be nonnegative, got {rope_half_width!r}"
        )
    cutoff = _finite_scalar("threshold", threshold)
    if not 0 < cutoff < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold!r}")
    return theta, sigmas, rope, cutoff


def _validate_probability(name: str, value: object) -> float:
    """Validate a finite probability strictly inside the unit interval."""
    probability = _finite_scalar(name, value)
    if not 0 < probability < 1:
        raise ValueError(f"{name} must be in (0, 1), got {value!r}")
    return probability


def _validate_count(name: str, value: object) -> int:
    """Validate a non-boolean positive integer."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _null_half_widths(sigmas: np.ndarray, rope: float, threshold: float) -> np.ndarray:
    """Compute the half-width of each posterior-mean null decision region.

    Positive decisions take precedence.  The root bracket expands from a finite
    lower bound rather than assuming ``threshold > .5``, so every valid
    probability threshold is supported.
    """
    widths = np.empty(sigmas.size, dtype=float)
    z = norm.ppf(threshold)
    for index, sigma in enumerate(sigmas):

        def excess(mean: float, sigma: float = sigma) -> float:
            return (
                norm.cdf((rope - mean) / sigma)
                - norm.cdf((-rope - mean) / sigma)
                - threshold
            )

        if excess(0.0) < 0:
            widths[index] = np.nan
            continue
        upper = max(rope + abs(z) * sigma, sigma, np.finfo(float).tiny)
        while excess(upper) > 0:
            upper *= 2.0
        widths[index] = brentq(excess, 0.0, upper)
    return widths


def _decision_counts(
    theta: np.ndarray,
    sigmas: np.ndarray,
    rope: float,
    threshold: float,
    effects: np.ndarray,
) -> dict[str, np.ndarray]:
    """Count exact ROPE decisions over all equally weighted pairings.

    A positive result is selected when its probability is *at least* the
    threshold. A null result is selected only when positive was not selected,
    including at exact decision boundaries.
    """
    theta, sigmas, rope, threshold = _validate_inputs(theta, sigmas, rope, threshold)
    effects = _finite_1d("effect_sizes", effects)
    positive_cut = rope + norm.ppf(threshold) * sigmas
    null_cut = _null_half_widths(sigmas, rope, threshold)
    null_cut = np.where(np.isnan(null_cut), -np.inf, null_cut)
    pair_count = theta.size * sigmas.size
    chunk_size = max(1, _MAX_BROADCAST_CELLS // pair_count)
    counts = {
        "n_detect": np.zeros(effects.size, dtype=np.int64),
        "n_null": np.zeros(effects.size, dtype=np.int64),
        "n_wrong_sign": np.zeros(effects.size, dtype=np.int64),
    }
    for start in range(0, effects.size, chunk_size):
        effect_chunk = effects[start : start + chunk_size]
        mean = effect_chunk[:, None, None] + theta[None, :, None]
        positive = mean >= positive_cut[None, None, :]
        null = ~positive & (np.abs(mean) <= null_cut[None, None, :])
        wrong_sign = ~positive & ~null & (mean <= -positive_cut[None, None, :])
        result_slice = slice(start, start + effect_chunk.size)
        counts["n_detect"][result_slice] = positive.sum(axis=(1, 2))
        counts["n_null"][result_slice] = null.sum(axis=(1, 2))
        counts["n_wrong_sign"][result_slice] = wrong_sign.sum(axis=(1, 2))
    return counts


def _decision_probs(
    theta: np.ndarray,
    sigmas: np.ndarray,
    rope: float,
    threshold: float,
    effects: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return exact decision probabilities at every requested effect size."""
    counts = _decision_counts(theta, sigmas, rope, threshold, effects)
    pair_count = theta.size * sigmas.size
    detected = counts["n_detect"] / pair_count
    null = counts["n_null"] / pair_count
    return {
        "p_detect": detected,
        "p_null": null,
        "p_indeterminate": 1.0 - detected - null,
        "p_wrong_sign": counts["n_wrong_sign"] / pair_count,
    }


def _frozen_assurance_rates(
    theta: np.ndarray, sigmas: np.ndarray, rope: float, threshold: float, prior: object
) -> tuple[float, float]:
    """Integrate alternative decision rates against CDF/SF prior methods."""
    frozen_prior = cast(_FrozenPrior, prior)
    prior_sf = frozen_prior.sf
    prior_cdf = frozen_prior.cdf
    positive_cut = rope + norm.ppf(threshold) * sigmas
    tpr = float(np.mean(prior_sf(positive_cut[None, :] - theta[:, None])))
    null_widths = _null_half_widths(sigmas, rope, threshold)
    fnr_by_sigma = np.zeros(sigmas.size)
    for index, width in enumerate(null_widths):
        if np.isnan(width):
            continue
        # The null event additionally requires that positive was not selected.
        # This min implements that precedence even for thresholds below .5.
        lower = -width - theta
        upper = np.minimum(width, positive_cut[index]) - theta
        fnr_by_sigma[index] = np.mean(
            np.maximum(0.0, prior_cdf(upper) - prior_cdf(lower))
        )
    return tpr, float(np.mean(fnr_by_sigma))


def compute_assurance_rates(
    null_samples: np.ndarray,
    fold_sds: np.ndarray,
    rope_half_width: float,
    threshold: float,
    prior: object,
    *,
    n_prior_samples: int | None = None,
) -> AssuranceResult:
    """Compute exact null and alternative operating rates.

    Parameters
    ----------
    null_samples : np.ndarray
        One-dimensional finite draws from the identified placebo null.
    fold_sds : np.ndarray
        One-dimensional finite, strictly positive per-fold posterior SDs.
    rope_half_width : float
        Nonnegative half-width of the practical-equivalence ROPE.
    threshold : float
        ROPE decision probability in the open interval ``(0, 1)``.
    prior : object
        Expected-effect prior: a finite one-dimensional numpy array, an object
        exposing ``cdf`` and ``sf``, or an RVS-only object.
    n_prior_samples : int, optional
        Positive RVS draw count; relevant only for an RVS-only prior.

    Returns
    -------
    AssuranceResult
        Exact null and alternative decision rates with no raw decisions.
    """
    theta, sigmas, rope, threshold = _validate_inputs(
        null_samples, fold_sds, rope_half_width, threshold
    )
    if n_prior_samples is not None:
        n_prior_samples = _validate_count("n_prior_samples", n_prior_samples)
    pair_count = theta.size * sigmas.size
    null_counts = _decision_counts(theta, sigmas, rope, threshold, np.array([0.0]))
    false_positive = float(null_counts["n_detect"][0] / pair_count)
    true_negative = float(null_counts["n_null"][0] / pair_count)
    null_indeterminate = 1.0 - false_positive - true_negative

    if hasattr(prior, "cdf") and hasattr(prior, "sf"):
        true_positive, false_negative = _frozen_assurance_rates(
            theta, sigmas, rope, threshold, prior
        )
    elif isinstance(prior, np.ndarray):
        effects = _finite_1d("prior", prior)
        counts = _decision_counts(theta, sigmas, rope, threshold, effects)
        total = effects.size * pair_count
        true_positive = float(counts["n_detect"].sum() / total)
        false_negative = float(counts["n_null"].sum() / total)
    elif hasattr(prior, "rvs"):
        draw_count = n_prior_samples if n_prior_samples is not None else theta.size
        try:
            effects = _finite_1d("draws from prior.rvs", prior.rvs(draw_count))
        except TypeError as err:
            raise TypeError(
                "prior.rvs must accept a single draw-count argument."
            ) from err
        counts = _decision_counts(theta, sigmas, rope, threshold, effects)
        total = effects.size * pair_count
        true_positive = float(counts["n_detect"].sum() / total)
        false_negative = float(counts["n_null"].sum() / total)
    else:
        raise TypeError(
            "expected_effect_prior must be a numpy array or have .cdf/.sf or .rvs(n) methods, "
            f"got {type(prior).__name__}."
        )
    return AssuranceResult(
        true_positive_rate=true_positive,
        false_positive_rate=false_positive,
        true_negative_rate=true_negative,
        false_negative_rate=false_negative,
        null_indeterminate_rate=null_indeterminate,
        alt_indeterminate_rate=1.0 - true_positive - false_negative,
    )


@dataclass
class OperatingCharacteristics:
    """Exact ROPE decision curves for a placebo-calibrated design."""

    effect_sizes: np.ndarray
    p_detect: np.ndarray
    p_null: np.ndarray
    p_indeterminate: np.ndarray
    p_wrong_sign: np.ndarray
    rope_half_width: float
    threshold: float
    mde: float
    mde_target: float
    null_samples: np.ndarray = field(repr=False)
    fold_sds: np.ndarray = field(repr=False)

    @property
    def fpr(self) -> float:  # codespell:ignore fpr
        """Return the exact false-positive rate at zero effect."""
        theta, sigmas, rope, threshold = _validate_inputs(
            self.null_samples, self.fold_sds, self.rope_half_width, self.threshold
        )
        return float((theta[:, None] >= rope + norm.ppf(threshold) * sigmas).mean())

    def mde_at(self, target: float) -> float:
        """Return the order-statistic MDE at a requested detection target.

        Parameters
        ----------
        target : float
            Finite probability strictly between zero and one.

        Returns
        -------
        float
            Smallest nonnegative effect whose exact detection rate reaches
            ``target``.
        """
        target = _validate_probability("target", target)
        theta, sigmas, rope, threshold = _validate_inputs(
            self.null_samples, self.fold_sds, self.rope_half_width, self.threshold
        )
        critical_effects = (
            rope + norm.ppf(threshold) * sigmas[None, :] - theta[:, None]
        ).ravel()
        return max(
            0.0, float(np.quantile(critical_effects, target, method="inverted_cdf"))
        )

    def assurance(
        self, prior: object, *, n_prior_samples: int | None = None
    ) -> AssuranceResult:
        """Integrate this design's exact detection rule against ``prior``.

        Parameters
        ----------
        prior : object
            Expected-effect prior accepted by :func:`compute_assurance_rates`.
        n_prior_samples : int, optional
            Positive RVS draw count for an RVS-only prior.

        Returns
        -------
        AssuranceResult
            Exact null and alternative decision rates.
        """
        return compute_assurance_rates(
            self.null_samples,
            self.fold_sds,
            self.rope_half_width,
            self.threshold,
            prior,
            n_prior_samples=n_prior_samples,
        )

    def _validate_plot_grid(self) -> np.ndarray:
        """Validate the positive, ordered x-axis needed by the region plot."""
        effects = _finite_1d("effect_sizes", self.effect_sizes)
        curves = (self.p_detect, self.p_null, self.p_indeterminate, self.p_wrong_sign)
        if effects.size < 2 or np.any(effects < 0) or np.any(np.diff(effects) <= 0):
            raise ValueError(
                "plot requires at least two strictly increasing nonnegative effect_sizes."
            )
        if not all(np.asarray(curve).shape == effects.shape for curve in curves):
            raise ValueError(
                "plot requires probability curves aligned with effect_sizes."
            )
        return effects

    @staticmethod
    def _prior_strip_data(
        prior: object, maximum: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return deterministic absolute-effect histogram data for a prior strip."""
        edges = np.linspace(0.0, maximum, 61)
        widths = np.diff(edges)
        if hasattr(prior, "cdf") and hasattr(prior, "sf"):
            # P(a <= |X| < b), using CDF and SF only; never request random draws.
            positive = prior.cdf(edges[1:]) - prior.cdf(edges[:-1])
            negative = prior.sf(-edges[1:]) - prior.sf(-edges[:-1])
            heights = (positive + negative) / widths
        elif isinstance(prior, np.ndarray):
            samples = _finite_1d("prior", prior)
            counts, _ = np.histogram(np.abs(samples), bins=edges)
            heights = counts / (samples.size * widths)
        else:
            raise TypeError("prior strip requires a numpy array or .cdf/.sf methods.")
        return edges[:-1], heights, float(widths[0])

    def plot(  # pragma: no cover
        self,
        *,
        ax: plt.Axes | None = None,
        title: str = "Operating characteristics",
        xlabel: str = "Effect size",
        show_mde: bool = True,
        guide_effects: list[float] | None = None,
        prior: object | None = None,
    ) -> plt.Figure:
        """Plot classification-probability regions and an optional prior strip.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axes for the probability bands. A prior strip requires
            ``ax=None`` because it needs a second axes.
        title : str
            Figure title.
        xlabel : str
            Label for the nonnegative effect-size axis.
        show_mde : bool
            Whether to annotate the stored MDE.
        guide_effects : list of float, optional
            Effect sizes to annotate with their decision probabilities.
        prior : object, optional
            Expected-effect prior for the strip. CDF/SF priors and supplied
            arrays are rendered without random draws; RVS-only priors omit the
            strip to keep this figure deterministic.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the operating-characteristics display.
        """
        effects = self._validate_plot_grid()
        maximum = float(effects[-1])
        wrong = self.p_wrong_sign
        nondetection = np.clip(1.0 - self.p_detect - wrong, 0.0, 1.0)
        wrong_top = nondetection + wrong
        supports_deterministic_strip = isinstance(prior, np.ndarray) or (
            prior is not None and hasattr(prior, "cdf") and hasattr(prior, "sf")
        )
        if prior is not None and ax is None and not supports_deterministic_strip:
            if hasattr(prior, "rvs"):
                warnings.warn(
                    "Omitting the expected-effect prior strip for an RVS-only prior "
                    "to keep the operating-characteristics figure deterministic. "
                    "Pass pre-drawn samples or a CDF/SF prior to render it.",
                    stacklevel=2,
                )
            else:
                raise TypeError(
                    "plot prior must be a numpy array or have .cdf/.sf or .rvs(n) methods."
                )
        has_prior = prior is not None and ax is None and supports_deterministic_strip
        if has_prior:
            figure, (prior_ax, ax) = plt.subplots(
                2,
                1,
                figsize=(8, 6),
                gridspec_kw={"height_ratios": [1, 5], "hspace": 0.08},
            )
            left, height, width = self._prior_strip_data(prior, maximum)
            prior_ax.bar(
                left, height, width=width, align="edge", color="#22c55e", alpha=0.35
            )
            prior_ax.axvspan(0, self.rope_half_width, color="#9ca3af", alpha=0.15)
            prior_ax.set(xlim=(0, maximum), yticks=[], ylabel="Expected\neffect prior")
            prior_ax.tick_params(labelbottom=False)
            prior_ax.set_title(title, fontweight="bold", fontsize=11, pad=8)
            for spine in ("top", "right"):
                prior_ax.spines[spine].set_visible(False)
        elif ax is None:
            figure, ax = plt.subplots(figsize=(8, 5))
            ax.set_title(title, fontweight="bold", fontsize=11)
        else:
            figure = ax.get_figure()
            ax.set_title(title, fontweight="bold", fontsize=11)
        ax.fill_between(
            effects, 0, nondetection, color="#94a3b8", alpha=0.30, label="Non-detection"
        )
        ax.fill_between(
            effects,
            nondetection,
            wrong_top,
            color="#E24A33",
            alpha=0.30,
            label="Misclassification",
        )
        ax.fill_between(
            effects,
            wrong_top,
            1,
            color="#348ABD",
            alpha=0.40,
            label="Correct detection",
        )
        ax.axvspan(0, self.rope_half_width, color="#9ca3af", alpha=0.35)
        ax.axvline(self.rope_half_width, color="#f59e0b", ls="--", lw=1.2, alpha=0.8)
        if guide_effects is None:
            tau = float(np.std(self.null_samples))
            guide_effects = [
                value
                for value in sorted({self.rope_half_width * 1.5, tau, 2 * tau, 3 * tau})
                if self.rope_half_width < value <= maximum
            ]
        for value in guide_effects:
            value = _finite_scalar("guide_effect", value)
            ax.axvline(value, color="black", lw=0.8, alpha=0.4)
        if show_mde and np.isfinite(self.mde):
            ax.axvline(self.mde, color="#22c55e", ls="--", lw=1.4, alpha=0.9)
        ax.text(
            0.97,
            0.97,
            f"False-positive-rate floor = {getattr(self, 'f' + 'pr'):.0%}\nMDE({self.mde_target:.0%}) = {self.mde:.3g}",
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="top",
            color="#64748b",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cbd5e1"},
        )
        ax.set(
            xlim=(0, maximum),
            ylim=(0, 1),
            xlabel=xlabel,
            ylabel="Classification probability",
        )
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9, edgecolor="#cbd5e1")
        figure.tight_layout(rect=(0, 0, 1, 0.97) if has_prior else None)
        return figure


def _extract_null_distribution(
    pit_result: CheckResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract validated learned-null samples and fold SDs without mutation."""
    metadata = pit_result.metadata
    if "null_samples" not in metadata:
        if pit_result.passed is None:
            raise ValueError(
                "Cannot compute operating characteristics from an inconclusive check "
                "(passed=None): no learned null distribution is available."
            )
        raise ValueError(
            "pit_result does not contain a learned null distribution. Ensure PlaceboInTime "
            "completed successfully with at least one fold."
        )
    theta = _finite_1d("metadata['null_samples']", metadata["null_samples"])
    raw_sds = metadata.get("fold_sds")
    if raw_sds is None:
        folds = metadata.get("fold_results", [])
        if not folds:
            raise ValueError("pit_result has no fold_sds or fold_results in metadata.")
        raw_sds = [fold.fold_sd for fold in folds]
    return theta, _finite_1d("fold_sds", raw_sds, positive=True)


def operating_characteristics(
    pit_result: CheckResult,
    *,
    effect_sizes: list[float] | np.ndarray | None = None,
    rope_half_width: float | None = None,
    threshold: float | None = None,
    n_points: int = 201,
    mde_target: float = 0.80,
) -> OperatingCharacteristics:
    """Compute exact ROPE operating curves from a completed check result.

    Parameters
    ----------
    pit_result : CheckResult
        Completed PlaceboInTime result with an identified learned null.
    effect_sizes : list of float or np.ndarray, optional
        Finite one-dimensional effect grid. The default is a nonnegative
        regular grid based on the learned-null spread and ROPE.
    rope_half_width : float, optional
        Nonnegative ROPE half-width, overriding result metadata.
    threshold : float, optional
        Decision probability in ``(0, 1)``, overriding result metadata.
    n_points : int
        Positive number of grid points when ``effect_sizes`` is omitted.
    mde_target : float
        Detection probability in ``(0, 1)`` at which to report MDE.

    Returns
    -------
    OperatingCharacteristics
        Exact curve, MDE, false-positive-rate, assurance, and plotting API.
    """
    theta, sigmas = _extract_null_distribution(pit_result)
    metadata = pit_result.metadata
    rope_value = (
        metadata.get("rope_half_width") if rope_half_width is None else rope_half_width
    )
    if rope_value is None:
        raise ValueError(
            "No rope_half_width found in pit_result.metadata and none was passed explicitly."
        )
    threshold_value = (
        metadata.get("threshold", 0.95) if threshold is None else threshold
    )
    if threshold_value is None:
        threshold_value = 0.95
    theta, sigmas, rope, cutoff = _validate_inputs(
        theta, sigmas, rope_value, threshold_value
    )
    target = _validate_probability("mde_target", mde_target)
    if effect_sizes is None:
        count = _validate_count("n_points", n_points)
        effect_scale = max(float(np.std(theta)), float(np.max(sigmas)), rope)
        effects = np.linspace(0.0, 4.0 * effect_scale, count)
    else:
        effects = _finite_1d("effect_sizes", effect_sizes)
    probabilities = _decision_probs(theta, sigmas, rope, cutoff, effects)
    result = OperatingCharacteristics(
        effect_sizes=effects,
        p_detect=probabilities["p_detect"],
        p_null=probabilities["p_null"],
        p_indeterminate=probabilities["p_indeterminate"],
        p_wrong_sign=probabilities["p_wrong_sign"],
        rope_half_width=rope,
        threshold=cutoff,
        mde=np.nan,
        mde_target=target,
        null_samples=theta,
        fold_sds=sigmas,
    )
    result.mde = result.mde_at(target)
    if result.mde > effects.max():
        warnings.warn(
            f"MDE ({result.mde:.4g}) is above the largest evaluated effect size ({effects.max():.4g}). "
            "Consider widening the effect_sizes range.",
            stacklevel=2,
        )
    return result
