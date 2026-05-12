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
Result classes for Synthetic Control experiment design methods.

Contains ``DressRehearsalResult``, ``PowerCurveResult``, and
``DonorPoolQualityResult`` — returned by ``SyntheticControl.validate_design()``,
``.power_analysis()``, and ``.donor_pool_quality()`` respectively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from causalpy.checks.base import CheckResult


@dataclass
class DressRehearsalResult:
    """Result of a dress-rehearsal design validation.

    Produced by :meth:`SyntheticControl.validate_design`.

    Attributes
    ----------
    injected_effect : float
        The effect that was injected into the pseudo-post window.
    effect_type : str
        Whether the injected effect was ``"relative"`` or ``"absolute"``.
    recovered_effect_mean : float
        Posterior mean of the cumulative impact in the pseudo-post window.
    recovered_effect_hdi : tuple[float, float]
        94% HDI of the cumulative impact posterior.
    hdi_covers_truth : bool
        Whether the HDI interval contains the injected truth.
    posterior_samples : xr.DataArray
        Raw posterior draws of cumulative impact.
    injected_truth : float
        The actual numeric value injected (after conversion from relative
        to absolute if applicable).
    """

    injected_effect: float
    effect_type: str
    recovered_effect_mean: float
    recovered_effect_hdi: tuple[float, float]
    hdi_covers_truth: bool
    posterior_samples: xr.DataArray
    injected_truth: float

    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        """Plot injected vs recovered effect with HDI band."""
        fig, ax = plt.subplots(figsize=(7, 4))

        samples = self.posterior_samples.values.flatten()
        ax.hist(
            samples,
            bins=40,
            density=True,
            alpha=0.6,
            color="C0",
            label="Posterior of cumulative impact",
        )
        ax.axvline(
            self.injected_truth,
            color="C3",
            lw=2,
            ls="--",
            label=f"Injected truth = {self.injected_truth:.3f}",
        )
        ax.axvline(
            self.recovered_effect_mean,
            color="C0",
            lw=2,
            label=f"Posterior mean = {self.recovered_effect_mean:.3f}",
        )
        ax.axvspan(
            self.recovered_effect_hdi[0],
            self.recovered_effect_hdi[1],
            alpha=0.2,
            color="C0",
            label="94% HDI",
        )

        status = "covers truth" if self.hdi_covers_truth else "misses truth"
        ax.set_title(f"Dress Rehearsal: HDI {status}")
        ax.set_xlabel("Cumulative impact")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig, ax

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame with recovery statistics."""
        return pd.DataFrame(
            [
                {
                    "injected_effect": self.injected_effect,
                    "effect_type": self.effect_type,
                    "injected_truth": self.injected_truth,
                    "recovered_mean": self.recovered_effect_mean,
                    "recovered_hdi_lower": self.recovered_effect_hdi[0],
                    "recovered_hdi_upper": self.recovered_effect_hdi[1],
                    "hdi_covers_truth": self.hdi_covers_truth,
                    "bias": self.recovered_effect_mean - self.injected_truth,
                    "relative_bias": (
                        (self.recovered_effect_mean - self.injected_truth)
                        / self.injected_truth
                        if self.injected_truth != 0
                        else np.nan
                    ),
                }
            ]
        )

    def to_check_result(self) -> CheckResult:
        """Convert to a ``CheckResult`` for sensitivity pipeline integration."""
        from causalpy.checks.base import CheckResult

        return CheckResult(
            check_name="DressRehearsal",
            passed=self.hdi_covers_truth,
            table=self.summary(),
            text=(
                f"Dress rehearsal with {self.effect_type} effect "
                f"{self.injected_effect}: recovered mean "
                f"{self.recovered_effect_mean:.3f} "
                f"(HDI [{self.recovered_effect_hdi[0]:.3f}, "
                f"{self.recovered_effect_hdi[1]:.3f}]). "
                f"HDI {'covers' if self.hdi_covers_truth else 'misses'} "
                f"the injected truth ({self.injected_truth:.3f})."
            ),
            figures=[self.plot()[0]],
            metadata={
                "dress_rehearsal_result": self,
            },
        )


@dataclass
class PowerCurveResult:
    """Result of a simulation-based Bayesian power analysis.

    Produced by :meth:`SyntheticControl.power_analysis`.

    Attributes
    ----------
    effect_sizes : list[float]
        Candidate effect sizes evaluated.
    detection_rates : list[float]
        Fraction of simulations where the criterion was met, per effect size.
    criterion : str
        The detection criterion used.
    raw_results : list[list[DressRehearsalResult]]
        Nested list: per effect size, per simulation.
    """

    effect_sizes: list[float]
    detection_rates: list[float]
    criterion: str
    raw_results: list[list[DressRehearsalResult]] = field(repr=False)

    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        """Power curve: effect size vs detection rate."""
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(
            self.effect_sizes,
            self.detection_rates,
            "o-",
            color="C0",
            lw=2,
            markersize=8,
        )
        ax.axhline(0.8, color="C3", ls="--", alpha=0.7, label="80% detection")
        ax.set_xlabel("Effect size")
        ax.set_ylabel("Detection rate")
        ax.set_title(f"Bayesian Power Curve (criterion: {self.criterion})")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, ax

    def summary(self) -> pd.DataFrame:
        """DataFrame with per-effect-size summary statistics."""
        rows = []
        for es, dr, sim_results in zip(
            self.effect_sizes, self.detection_rates, self.raw_results, strict=True
        ):
            means = [r.recovered_effect_mean for r in sim_results]
            rows.append(
                {
                    "effect_size": es,
                    "detection_rate": dr,
                    "mean_recovery": np.mean(means),
                    "median_recovery": np.median(means),
                    "n_simulations": len(sim_results),
                }
            )
        return pd.DataFrame(rows)


@dataclass
class DonorPoolQualityResult:
    """Result of a donor pool quality assessment.

    Produced by :meth:`SyntheticControl.donor_pool_quality`.

    Attributes
    ----------
    correlation_score : float
        Mean pairwise correlation between treated and control units.
    convex_hull_coverage : float
        Fraction of pre-period time points where treated is within
        the donor envelope.
    weight_concentration : float
        Effective number of donors (1 / sum(w_i^2)), measuring
        how concentrated the Dirichlet weights are.
    per_donor_details : pd.DataFrame
        Per-donor statistics: correlation, mean weight, etc.
    """

    correlation_score: float
    convex_hull_coverage: float
    weight_concentration: float
    per_donor_details: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Formatted summary with per-metric scores and qualitative assessment."""
        quality = self._overall_quality()

        metrics = pd.DataFrame(
            [
                {
                    "metric": "Mean donor correlation",
                    "value": f"{self.correlation_score:.3f}",
                    "assessment": (
                        "good"
                        if self.correlation_score > 0.8
                        else "acceptable"
                        if self.correlation_score > 0.5
                        else "poor"
                    ),
                },
                {
                    "metric": "Convex hull coverage",
                    "value": f"{self.convex_hull_coverage:.1%}",
                    "assessment": (
                        "good"
                        if self.convex_hull_coverage > 0.95
                        else "acceptable"
                        if self.convex_hull_coverage > 0.80
                        else "poor"
                    ),
                },
                {
                    "metric": "Effective number of donors",
                    "value": f"{self.weight_concentration:.2f}",
                    "assessment": (
                        "good"
                        if self.weight_concentration > 3
                        else "acceptable"
                        if self.weight_concentration > 1.5
                        else "poor"
                    ),
                },
                {
                    "metric": "Overall quality",
                    "value": quality,
                    "assessment": quality,
                },
            ]
        )
        return metrics

    def _overall_quality(self) -> str:
        """Compute an overall qualitative assessment."""
        scores = []
        scores.append(
            "good"
            if self.correlation_score > 0.8
            else "acceptable"
            if self.correlation_score > 0.5
            else "poor"
        )
        scores.append(
            "good"
            if self.convex_hull_coverage > 0.95
            else "acceptable"
            if self.convex_hull_coverage > 0.80
            else "poor"
        )
        scores.append(
            "good"
            if self.weight_concentration > 3
            else "acceptable"
            if self.weight_concentration > 1.5
            else "poor"
        )

        if "poor" in scores:
            return "poor"
        if all(s == "good" for s in scores):
            return "good"
        return "acceptable"
