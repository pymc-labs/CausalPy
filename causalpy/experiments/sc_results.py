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
"""Result types for Synthetic Control design methods.

These objects are returned by :meth:`SyntheticControl.from_pre_period`,
:meth:`SyntheticControl.validate_design`, :meth:`SyntheticControl.power_analysis`,
and :meth:`SyntheticControl.donor_pool_quality`. They are not experiment
classes, so they stay in this module rather than ``experiments/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from causalpy.constants import HDI_PROB

if TYPE_CHECKING:  # pragma: no cover
    from causalpy.checks.base import CheckResult


@dataclass
class DressRehearsalResult:
    """Result of a dress-rehearsal design validation.

    Produced by :meth:`SyntheticControl.validate_design`.

    Attributes
    ----------
    injected_effect : float
        The effect that was injected. Interpretation depends on
        ``effect_type``.
    effect_type : str
        ``"relative"`` or ``"absolute"``.
    recovered_effect_mean : float
        Posterior mean of the cumulative impact at the end of the
        pseudo-post window.
    recovered_effect_hdi : tuple of float
        Lower and upper bounds of the cumulative-impact HDI. The
        probability mass is :data:`causalpy.constants.HDI_PROB`.
    hdi_covers_truth : bool
        Whether that HDI contains the injected truth.
    posterior_samples : xarray.DataArray
        Stacked posterior samples of the final cumulative impact.
    injected_truth : float
        Absolute cumulative effect implied by the injection, in outcome
        units. This is the quantity the HDI is compared against.
    """

    injected_effect: float
    effect_type: str
    recovered_effect_mean: float
    recovered_effect_hdi: tuple[float, float]
    hdi_covers_truth: bool
    posterior_samples: xr.DataArray
    injected_truth: float

    def summary(self) -> pd.DataFrame:
        """Return a one-row summary of the dress rehearsal."""
        lower, upper = self.recovered_effect_hdi
        return pd.DataFrame(
            [
                {
                    "injected_effect": self.injected_effect,
                    "effect_type": self.effect_type,
                    "injected_truth": self.injected_truth,
                    "recovered_effect_mean": self.recovered_effect_mean,
                    "hdi_lower": lower,
                    "hdi_upper": upper,
                    "hdi_covers_truth": self.hdi_covers_truth,
                }
            ]
        )

    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        """Plot the recovered cumulative-impact posterior against the truth."""
        samples = np.asarray(self.posterior_samples.values, dtype=float).ravel()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(samples, bins=30, color="C0", alpha=0.75, density=True)
        ax.axvline(
            self.injected_truth,
            color="C3",
            ls="--",
            lw=2,
            label=f"injected truth ({self.injected_truth:.3g})",
        )
        ax.axvline(
            self.recovered_effect_mean,
            color="C0",
            lw=2,
            label=f"posterior mean ({self.recovered_effect_mean:.3g})",
        )
        lower, upper = self.recovered_effect_hdi
        ax.axvspan(lower, upper, color="C0", alpha=0.15, label=f"{HDI_PROB:.0%} HDI")
        ax.set_xlabel("Cumulative impact")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Dress rehearsal ({self.effect_type} effect {self.injected_effect:g})"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig, ax

    def to_check_result(self) -> CheckResult:
        """Convert to a ``CheckResult`` for sensitivity pipeline integration."""
        lower, upper = self.recovered_effect_hdi
        covers = "covers" if self.hdi_covers_truth else "misses"
        from causalpy.checks.base import CheckResult

        return CheckResult(
            check_name="DressRehearsal",
            passed=self.hdi_covers_truth,
            table=self.summary(),
            text=(
                f"Dress rehearsal with {self.effect_type} effect "
                f"{self.injected_effect}: recovered mean "
                f"{self.recovered_effect_mean:.3f} "
                f"(HDI [{lower:.3f}, {upper:.3f}]). "
                f"HDI {covers} the injected truth ({self.injected_truth:.3f})."
            ),
            figures=[self.plot()[0]],
            metadata={"dress_rehearsal_result": self},
        )


@dataclass
class PowerCurveResult:
    """Result of a simulation-based power analysis.

    Produced by :meth:`SyntheticControl.power_analysis`. The curve is a
    heuristic under an iid Gaussian residual model, not a sampling
    distribution of the treatment effect.

    Attributes
    ----------
    effect_sizes : list of float
        Candidate effect sizes evaluated.
    detection_rates : list of float
        Fraction of simulations where the criterion was met, per effect size.
    criterion : str
        The detection criterion used.
    raw_results : list of list of DressRehearsalResult
        Nested list: per effect size, per simulation.
    """

    effect_sizes: list[float]
    detection_rates: list[float]
    criterion: str
    raw_results: list[list[DressRehearsalResult]] = field(repr=False)

    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        """Plot the detection rate at each requested effect size.

        Points are not joined. The method does not estimate a continuous
        power function, and a line between two rates would invent one. The
        dashed line is an 80% reference, not a result of this run.
        """
        n_simulations = len(self.raw_results[0]) if self.raw_results else 0
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            self.effect_sizes,
            self.detection_rates,
            "o",
            color="C0",
            label="Detection rate",
        )
        ax.axhline(0.8, color="C3", ls="--", alpha=0.7, label="80% reference")
        ax.set_xlabel("Effect size")
        ax.set_ylabel("Detection rate")
        ax.set_title(
            f"Detection rates (n={n_simulations}, criterion: {self.criterion})"
        )
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, ax

    def summary(self) -> pd.DataFrame:
        """Return per-effect-size detection and recovery statistics."""
        rows = []
        for effect_size, rate, sim_results in zip(
            self.effect_sizes,
            self.detection_rates,
            self.raw_results,
            strict=True,
        ):
            means = [result.recovered_effect_mean for result in sim_results]
            rows.append(
                {
                    "effect_size": effect_size,
                    "detection_rate": rate,
                    "mean_recovery": float(np.mean(means)),
                    "median_recovery": float(np.median(means)),
                    "n_simulations": len(sim_results),
                }
            )
        return pd.DataFrame(rows)


@dataclass
class DonorPoolQualityResult:
    """Result of a donor-pool quality assessment.

    Produced by :meth:`SyntheticControl.donor_pool_quality`.

    Attributes
    ----------
    correlation_score : float
        Mean correlation between each treated unit and each donor.
    convex_hull_coverage : float
        Share of pre-period times at which the treated value lies between the
        lowest and highest donor. Necessary for a convex combination, not
        proof of one.
    weight_concentration : float
        Effective number of donors, ``1 / sum(w_i^2)``.
    per_donor_details : pandas.DataFrame
        Per-donor correlation and weight.
    """

    correlation_score: float
    convex_hull_coverage: float
    weight_concentration: float
    per_donor_details: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Return a qualitative summary of the three donor-pool metrics."""
        quality = self._overall_quality()
        return pd.DataFrame(
            [
                {
                    "metric": "Mean donor correlation",
                    "value": f"{self.correlation_score:.3f}",
                    "assessment": _band(self.correlation_score, 0.8, 0.5),
                },
                {
                    "metric": "Convex hull coverage",
                    "value": f"{self.convex_hull_coverage:.1%}",
                    "assessment": _band(self.convex_hull_coverage, 0.95, 0.80),
                },
                {
                    "metric": "Effective number of donors",
                    "value": f"{self.weight_concentration:.2f}",
                    "assessment": _band(self.weight_concentration, 3, 1.5),
                },
                {
                    "metric": "Overall quality",
                    "value": quality,
                    "assessment": quality,
                },
            ]
        )

    def _overall_quality(self) -> str:
        scores = [
            _band(self.correlation_score, 0.8, 0.5),
            _band(self.convex_hull_coverage, 0.95, 0.80),
            _band(self.weight_concentration, 3, 1.5),
        ]
        if "poor" in scores:
            return "poor"
        if all(score == "good" for score in scores):
            return "good"
        return "acceptable"


def _band(value: float, good: float, acceptable: float) -> str:
    if value > good:
        return "good"
    if value > acceptable:
        return "acceptable"
    return "poor"
