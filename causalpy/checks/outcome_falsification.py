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
Outcome falsification sensitivity check.

Re-fits the same causal method with alternative outcome formulas and
reports their estimated effect sizes.  The researcher interprets
whether the pattern of effects across outcomes is consistent with
their causal story.

Inspired by the "causal detective" approach in Gallea (2026),
*The Causal Mindset Handbook*, where Hanlon's London fog study
tests the pollution hypothesis by comparing effect sizes across
outcomes (pneumonia deaths vs. accident deaths vs. crime deaths).

Supports formula-based experiments: InterruptedTimeSeries,
DifferenceInDifferences, PiecewiseITS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from causalpy.checks.base import CheckResult, clone_model
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.diff_in_diff import DifferenceInDifferences
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.experiments.piecewise_its import PiecewiseITS
from causalpy.pipeline import PipelineContext
from causalpy.pymc_models import PyMCModel

logger = logging.getLogger(__name__)


@dataclass
class FalsificationResult:
    """Result for a single falsification formula.

    Attributes
    ----------
    formula : str
        The falsification formula used.
    experiment : BaseExperiment
        The fitted experiment for this formula.
    effect_mean : float
        Posterior mean of the estimated effect.
    hdi_lower : float
        Lower bound of the HDI for the effect.
    hdi_upper : float
        Upper bound of the HDI for the effect.
    """

    formula: str
    experiment: BaseExperiment
    effect_mean: float
    hdi_lower: float
    hdi_upper: float


class OutcomeFalsification:
    """Outcome falsification sensitivity check.

    Re-fits the experiment with alternative formulas and reports the
    estimated effect size for each.  This is an informational check --
    it does not make pass/fail judgments.  The researcher compares
    effect sizes across outcomes to assess whether the pattern is
    consistent with their causal story.

    Parameters
    ----------
    formulas : list[str]
        Falsification formulas.  Each must be a complete patsy formula
        (e.g., ``"beer ~ 1 + year"``).  The outcome variable must
        exist in the data.
    alpha : float, default 0.05
        Significance level.  The HDI probability is ``1 - alpha``.

    Examples
    --------
    >>> import causalpy as cp  # doctest: +SKIP
    >>> check = cp.checks.OutcomeFalsification(  # doctest: +SKIP
    ...     formulas=["beer ~ 1 + year", "lnincome ~ 1 + year"],
    ... )
    """

    applicable_methods: set[type[BaseExperiment]] = {
        InterruptedTimeSeries,
        DifferenceInDifferences,
        PiecewiseITS,
    }

    def __init__(
        self,
        formulas: list[str],
        alpha: float = 0.05,
    ) -> None:
        if not formulas:
            raise ValueError("formulas must be a non-empty list of formula strings.")
        if not all(isinstance(f, str) for f in formulas):
            raise TypeError("Each formula must be a string.")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.formulas = list(formulas)
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, experiment: BaseExperiment) -> None:
        """Verify the experiment is compatible with OutcomeFalsification.

        Raises
        ------
        TypeError
            If the experiment does not use a formula, or uses a
            non-PyMC model.
        """
        if not hasattr(experiment, "formula"):
            raise TypeError(
                f"{type(experiment).__name__} does not use a formula. "
                f"OutcomeFalsification requires formula-based experiments "
                f"(InterruptedTimeSeries, DifferenceInDifferences, PiecewiseITS)."
            )
        if not isinstance(experiment.model, PyMCModel):
            raise TypeError(
                f"OutcomeFalsification requires a PyMC model for posterior "
                f"extraction, but got {type(experiment.model).__name__}. "
                f"Use a PyMC model (e.g. cp.pymc_models.LinearRegression)."
            )

    # ------------------------------------------------------------------
    # Factory helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_experiment(
        context: PipelineContext,
        formula: str,
    ) -> BaseExperiment:
        """Create and fit an experiment with a different formula.

        Uses ``context.experiment_config`` to reconstruct the experiment
        with all original parameters except the formula.
        """
        if context.experiment_config is None:
            raise RuntimeError(
                "No experiment_config in context. Use EstimateEffect "
                "before SensitivityAnalysis, or pass an explicit "
                "experiment_factory."
            )

        config = context.experiment_config
        method = config["method"]
        kwargs = {k: v for k, v in config.items() if k != "method"}

        kwargs["formula"] = formula

        if "model" in kwargs and kwargs["model"] is not None:
            kwargs["model"] = clone_model(kwargs["model"])

        return method(context.data, **kwargs)

    # ------------------------------------------------------------------
    # Effect extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_effect_stats(
        experiment: BaseExperiment,
        alpha: float,
    ) -> dict[str, float]:
        """Extract mean and HDI from a fitted experiment.

        Uses the experiment's ``effect_summary()`` method which provides
        standardized output across experiment types.  The returned table
        always contains ``hdi_lower`` and ``hdi_upper`` columns.
        """
        summary = experiment.effect_summary(
            alpha=alpha, direction="two-sided", cumulative=False, relative=False
        )
        table = summary.table
        row = table.iloc[0]

        return {
            "mean": float(row["mean"]),
            "hdi_lower": float(row["hdi_lower"]),
            "hdi_upper": float(row["hdi_upper"]),
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext,
    ) -> CheckResult:
        """Run outcome falsification analysis.

        For each falsification formula, fits the experiment with
        the alternative formula and reports the estimated effect size
        with HDI intervals.

        This is an informational check (``passed=None``).  The
        researcher interprets whether the pattern of effect sizes
        across outcomes supports their causal story.

        Returns
        -------
        CheckResult
            With ``passed=None`` (informational).  The table contains
            effect sizes and HDI intervals for each falsification
            formula.
        """
        self.validate(experiment)

        results: list[FalsificationResult] = []
        rows: list[dict[str, Any]] = []
        failed_formulas: list[str] = []

        hdi_pct = int(round((1 - self.alpha) * 100))

        for formula in self.formulas:
            logger.info(
                "OutcomeFalsification: fitting formula '%s'",
                formula,
            )
            try:
                alt_experiment = self._build_experiment(context, formula)
                stats = self._extract_effect_stats(alt_experiment, self.alpha)

                fr = FalsificationResult(
                    formula=formula,
                    experiment=alt_experiment,
                    effect_mean=stats["mean"],
                    hdi_lower=stats["hdi_lower"],
                    hdi_upper=stats["hdi_upper"],
                )
                results.append(fr)
                rows.append(
                    {
                        "formula": formula,
                        "effect_mean": stats["mean"],
                        f"hdi_{hdi_pct}%_lower": stats["hdi_lower"],
                        f"hdi_{hdi_pct}%_upper": stats["hdi_upper"],
                    }
                )

            except Exception:
                logger.warning(
                    "OutcomeFalsification: failed for formula '%s'",
                    formula,
                    exc_info=True,
                )
                failed_formulas.append(formula)
                rows.append(
                    {
                        "formula": formula,
                        "effect_mean": float("nan"),
                        f"hdi_{hdi_pct}%_lower": float("nan"),
                        f"hdi_{hdi_pct}%_upper": float("nan"),
                    }
                )

        table = pd.DataFrame(rows) if rows else None

        parts = [
            f"Outcome falsification: {len(self.formulas)} formula(s), {hdi_pct}% HDI.",
        ]
        if failed_formulas:
            parts.append(
                f"  {len(failed_formulas)} formula(s) failed to fit: {failed_formulas}"
            )
        for fr in results:
            parts.append(
                f"  {fr.formula}: "
                f"effect = {fr.effect_mean:.4f} "
                f"[{fr.hdi_lower:.4f}, {fr.hdi_upper:.4f}]"
            )
        text = "\n".join(parts)

        return CheckResult(
            check_name="OutcomeFalsification",
            passed=None,
            table=table,
            text=text,
            metadata={
                "falsification_results": results,
                "alpha": self.alpha,
            },
        )

    def __repr__(self) -> str:
        return f"OutcomeFalsification(formulas={self.formulas!r}, alpha={self.alpha})"
