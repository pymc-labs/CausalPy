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
import warnings
from dataclasses import dataclass
from typing import Any

import pandas as pd
from patsy import PatsyError

from causalpy.checks.base import CheckResult, clone_model
from causalpy.custom_exceptions import DataException, FormulaException
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.diff_in_diff import DifferenceInDifferences
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.experiments.piecewise_its import PiecewiseITS
from causalpy.pipeline import PipelineContext

logger = logging.getLogger(__name__)

_STORE_EXPERIMENTS_WARN_THRESHOLD = 3


@dataclass
class FalsificationResult:
    """Result for a single falsification formula.

    Attributes
    ----------
    formula : str
        The falsification formula used.
    effect_mean : float
        Posterior mean of the estimated effect.
    hdi_lower : float
        Lower bound of the HDI for the effect.
    hdi_upper : float
        Upper bound of the HDI for the effect.
    experiment : BaseExperiment | None
        The fitted experiment for this formula.  ``None`` when
        ``OutcomeFalsification`` was run with ``store_experiments=False``;
        in that case only the summary statistics above are retained.
    """

    formula: str
    effect_mean: float
    hdi_lower: float
    hdi_upper: float
    experiment: BaseExperiment | None = None


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
    store_experiments : bool, default True
        If ``True`` (default), each ``FalsificationResult`` retains a
        reference to the fitted experiment (including its
        ``InferenceData``), which lets users inspect posteriors but
        can be memory-heavy for many formulas.  Set to ``False`` to
        keep only the summary statistics (``effect_mean``,
        ``hdi_lower``, ``hdi_upper``).  A one-off warning is emitted at
        :meth:`run` when ``store_experiments=True`` and at least
        ``3`` formulas are supplied, because the combined
        ``InferenceData`` footprint of several fitted experiments can
        easily reach hundreds of MB on larger datasets
        (e.g. :class:`PiecewiseITS`).

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
        store_experiments: bool = True,
    ) -> None:
        if not formulas:
            raise ValueError("formulas must be a non-empty list of formula strings.")
        if not all(isinstance(f, str) for f in formulas):
            raise TypeError("Each formula must be a string.")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.formulas = list(formulas)
        self.alpha = alpha
        self.store_experiments = store_experiments

    def validate(self, experiment: BaseExperiment) -> None:
        """Verify the experiment is compatible with OutcomeFalsification.

        Parameters
        ----------
        experiment : BaseExperiment
            Candidate experiment to validate.

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
        if not experiment._model_backend.supports_idata:
            raise TypeError(
                f"OutcomeFalsification requires a PyMC model or another backend "
                f"with InferenceData for posterior "
                f"extraction, but got {type(experiment.model).__name__}. "
                f"Use a PyMC model (e.g. cp.pymc_models.LinearRegression)."
            )

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

    @staticmethod
    def _extract_effect_stats(
        experiment: BaseExperiment,
        alpha: float,
    ) -> dict[str, float]:
        """Extract mean and HDI from a fitted experiment.

        Uses the experiment's ``effect_summary()`` method which provides
        standardized output across experiment types.  The returned table
        always contains ``hdi_lower`` and ``hdi_upper`` columns.

        Raises
        ------
        RuntimeError
            If ``effect_summary()`` returns an empty table or one that
            does not expose the expected ``mean`` / ``hdi_lower`` /
            ``hdi_upper`` columns.  This is a defensive guard against
            future schema changes in ``effect_summary``: the
            single-effect contract assumed here (``iloc[0]`` is *the*
            effect row) should fail loudly rather than silently
            returning the wrong row.
        """
        summary = experiment.effect_summary(
            alpha=alpha, direction="two-sided", cumulative=False, relative=False
        )
        table = summary.table

        required_columns = {"mean", "hdi_lower", "hdi_upper"}
        if table is None or len(table) == 0:
            raise RuntimeError(
                f"{type(experiment).__name__}.effect_summary() returned an "
                f"empty table; cannot extract falsification effect statistics."
            )
        missing = required_columns - set(table.columns)
        if missing:
            raise RuntimeError(
                f"{type(experiment).__name__}.effect_summary() table is "
                f"missing required column(s) {sorted(missing)}; "
                f"OutcomeFalsification expects a single-effect summary "
                f"with columns {sorted(required_columns)}."
            )

        row = table.iloc[0]

        return {
            "mean": float(row["mean"]),
            "hdi_lower": float(row["hdi_lower"]),
            "hdi_upper": float(row["hdi_upper"]),
        }

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

        Parameters
        ----------
        experiment : BaseExperiment
            The fitted experiment to check.
        context : PipelineContext
            Pipeline context providing ``experiment_config`` for re-fits.

        Returns
        -------
        CheckResult
            With ``passed=None`` (informational).  The table contains
            effect sizes and HDI intervals for each falsification
            formula.
        """
        self.validate(experiment)

        if (
            self.store_experiments
            and len(self.formulas) >= _STORE_EXPERIMENTS_WARN_THRESHOLD
        ):
            warnings.warn(
                f"OutcomeFalsification will store {len(self.formulas)} fitted "
                f"experiments (each with its own InferenceData).  The combined "
                f"footprint can reach hundreds of MB on large datasets or "
                f"models with many posterior samples.  Pass "
                f"store_experiments=False if you only need the summary "
                f"statistics (effect_mean, hdi_lower, hdi_upper).",
                stacklevel=2,
            )

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
                    effect_mean=stats["mean"],
                    hdi_lower=stats["hdi_lower"],
                    hdi_upper=stats["hdi_upper"],
                    experiment=alt_experiment if self.store_experiments else None,
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

            except (
                PatsyError,
                FormulaException,
                DataException,
                ValueError,
                KeyError,
                RuntimeError,
            ) as exc:
                logger.warning(
                    "OutcomeFalsification: failed for formula '%s'",
                    formula,
                    exc_info=True,
                )
                warnings.warn(
                    f"OutcomeFalsification: formula {formula!r} failed to fit "
                    f"({type(exc).__name__}: {exc}); skipping.",
                    stacklevel=2,
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
        """Return a string representation, showing only non-default flags."""
        parts = [f"formulas={self.formulas!r}"]
        if self.alpha != 0.05:
            parts.append(f"alpha={self.alpha}")
        if not self.store_experiments:
            parts.append("store_experiments=False")
        return f"OutcomeFalsification({', '.join(parts)})"
