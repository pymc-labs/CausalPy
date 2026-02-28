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
Placebo-in-time sensitivity check.

Tests whether the intervention effect appears in pre-intervention periods
where no treatment actually occurred.  If the model detects spurious
effects during placebo periods, this casts doubt on the causal claim.

Supports experiments with a ``treatment_time`` parameter
(InterruptedTimeSeries, SyntheticControl).  Works with both datetime
and numeric indices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from causalpy.checks.base import CheckResult, clone_model
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pipeline import PipelineContext
from causalpy.pymc_models import PyMCModel
from causalpy.reporting import EffectSummary

logger = logging.getLogger(__name__)

MIN_FOLD_OBSERVATIONS = 3


@dataclass
class PlaceboFoldResult:
    """Result of a single placebo fold.

    Attributes
    ----------
    fold : int
        Fold number (1-indexed).
    pseudo_treatment_time : Any
        The shifted treatment time for this fold.
    experiment : BaseExperiment
        The fitted experiment for this fold.
    effect_summary : EffectSummary or None
        The effect summary for this fold, if available.
    effect_is_null : bool or None
        Whether the fold's effect interval contains zero (i.e. no spurious
        effect detected). ``True`` means the placebo fold passed.
    """

    fold: int
    pseudo_treatment_time: Any
    experiment: BaseExperiment
    effect_summary: EffectSummary | None = None
    effect_is_null: bool | None = None


class PlaceboInTime:
    """Placebo-in-time sensitivity check.

    Shifts the treatment time backward into the pre-intervention period
    to create ``n_folds`` placebo experiments.  If any placebo fold shows
    a significant effect (its confidence/credible interval excludes zero),
    the overall check fails, casting doubt on the causal claim.

    Parameters
    ----------
    n_folds : int, default 3
        Number of placebo folds to create.  Must be >= 1.
    experiment_factory : callable, optional
        Custom factory ``(data, treatment_time) -> BaseExperiment``.
        If ``None`` (default), the factory is derived from the pipeline's
        ``experiment_config``.

    Examples
    --------
    >>> import causalpy as cp  # doctest: +SKIP
    >>> check = cp.checks.PlaceboInTime(n_folds=3)  # doctest: +SKIP
    """

    applicable_methods: set[type[BaseExperiment]] = {
        InterruptedTimeSeries,
        SyntheticControl,
    }

    def __init__(
        self,
        n_folds: int = 3,
        experiment_factory: Any | None = None,
    ) -> None:
        if n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        self.n_folds = n_folds
        self.experiment_factory = experiment_factory

    def validate(self, experiment: BaseExperiment) -> None:
        """Check that the experiment has a treatment_time attribute."""
        if not hasattr(experiment, "treatment_time"):
            raise TypeError(
                f"{type(experiment).__name__} does not have a treatment_time "
                f"attribute. PlaceboInTime requires experiments with an "
                f"explicit treatment time."
            )

    def _get_factory(self, context: PipelineContext) -> Any:
        """Return a factory function ``(data, treatment_time) -> experiment``."""
        if self.experiment_factory is not None:
            return self.experiment_factory

        config = context.experiment_config
        if config is None:
            raise RuntimeError(
                "No experiment_config in context and no experiment_factory "
                "provided. Use EstimateEffect before SensitivityAnalysis, "
                "or pass an explicit experiment_factory to PlaceboInTime."
            )

        method = config["method"]
        kwargs = {k: v for k, v in config.items() if k != "method"}

        def _factory(data: pd.DataFrame, treatment_time: Any) -> BaseExperiment:
            """Create a fresh experiment with the given treatment time."""
            kw = dict(kwargs)
            kw["treatment_time"] = treatment_time
            if "model" in kw and kw["model"] is not None:
                kw["model"] = clone_model(kw["model"])
            return method(data, **kw)

        return _factory

    def _compute_intervention_length(
        self,
        experiment: BaseExperiment,
    ) -> Any:
        """Compute the intervention length from the experiment and data.

        For experiments with ``treatment_end_time``, uses that interval.
        Otherwise, uses the distance from treatment_time to the end of data.
        """
        treatment_time = experiment.treatment_time  # type: ignore[attr-defined]
        data = experiment.data  # type: ignore[attr-defined]

        treatment_end = getattr(experiment, "treatment_end_time", None)
        if treatment_end is not None:
            return treatment_end - treatment_time

        if hasattr(data, "index"):
            return data.index.max() - treatment_time

        raise ValueError("Cannot determine intervention length from experiment.")

    def _compute_fold_treatment_times(
        self,
        treatment_time: Any,
        intervention_length: Any,
    ) -> list[Any]:
        """Compute pseudo-treatment times for each fold."""
        return [
            treatment_time - (self.n_folds - fold) * intervention_length
            for fold in range(self.n_folds)
        ]

    def _get_fold_data(
        self,
        data: pd.DataFrame,
        pseudo_treatment_time: Any,
        intervention_length: Any,
    ) -> pd.DataFrame:
        """Extract data up to the end of the placebo intervention window."""
        pseudo_end = pseudo_treatment_time + intervention_length
        return data.loc[data.index < pseudo_end].copy()

    @staticmethod
    def _fold_effect_is_null(experiment: BaseExperiment) -> tuple[bool, EffectSummary]:
        """Check whether the fold's effect interval contains zero.

        Returns ``(is_null, effect_summary)`` where ``is_null`` is ``True``
        when the confidence/credible interval includes zero (no spurious
        effect detected).
        """
        summary = experiment.effect_summary()
        table = summary.table
        if isinstance(experiment.model, PyMCModel):
            lower = table["hdi_lower"].iloc[0]
            upper = table["hdi_upper"].iloc[0]
        else:
            lower = table["ci_lower"].iloc[0]
            upper = table["ci_upper"].iloc[0]
        return bool(lower <= 0 <= upper), summary

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext,
    ) -> CheckResult:
        """Run placebo-in-time analysis.

        Creates ``n_folds`` placebo experiments by shifting the treatment
        time backward.  Each fold uses only data before the placebo
        intervention window ends, ensuring no actual treatment data leaks.

        A fold **passes** when its effect interval (HDI for Bayesian, CI
        for OLS) contains zero, indicating no spurious effect.  The overall
        check passes when every completed fold passes.

        Returns
        -------
        CheckResult
            With ``passed`` indicating whether all folds found null effects,
            and ``metadata["fold_results"]`` containing the
            :class:`PlaceboFoldResult` objects.
        """
        factory = self._get_factory(context)
        treatment_time = experiment.treatment_time  # type: ignore[attr-defined]
        data = experiment.data  # type: ignore[attr-defined]
        intervention_length = self._compute_intervention_length(experiment)

        fold_treatment_times = self._compute_fold_treatment_times(
            treatment_time, intervention_length
        )

        fold_results: list[PlaceboFoldResult] = []
        fold_summaries: list[str] = []
        fold_passes: list[bool] = []
        skipped_folds: list[int] = []

        for fold_idx, pseudo_tt in enumerate(fold_treatment_times):
            fold_num = fold_idx + 1
            logger.info(
                "PlaceboInTime fold %d/%d: pseudo_treatment_time=%s",
                fold_num,
                self.n_folds,
                pseudo_tt,
            )

            fold_data = self._get_fold_data(data, pseudo_tt, intervention_length)

            if len(fold_data) < MIN_FOLD_OBSERVATIONS:
                logger.warning(
                    "Fold %d has only %d observations (minimum %d), skipping.",
                    fold_num,
                    len(fold_data),
                    MIN_FOLD_OBSERVATIONS,
                )
                skipped_folds.append(fold_num)
                fold_summaries.append(
                    f"Fold {fold_num}: SKIPPED (only {len(fold_data)} "
                    f"observations, need >= {MIN_FOLD_OBSERVATIONS})"
                )
                continue

            try:
                fold_experiment = factory(fold_data, pseudo_tt)
                is_null, effect_summary = self._fold_effect_is_null(fold_experiment)
            except Exception:
                logger.warning(
                    "Fold %d failed to fit (pseudo_treatment_time=%s), skipping.",
                    fold_num,
                    pseudo_tt,
                )
                skipped_folds.append(fold_num)
                fold_summaries.append(
                    f"Fold {fold_num}: SKIPPED (experiment failed to fit "
                    f"at pseudo treatment time {pseudo_tt})"
                )
                continue

            fold_passes.append(is_null)
            fold_result = PlaceboFoldResult(
                fold=fold_num,
                pseudo_treatment_time=pseudo_tt,
                experiment=fold_experiment,
                effect_summary=effect_summary,
                effect_is_null=is_null,
            )
            fold_results.append(fold_result)

            status = "PASS (null)" if is_null else "FAIL (spurious effect)"
            fold_summaries.append(
                f"Fold {fold_num}: pseudo treatment at {pseudo_tt} — {status}"
            )

        n_completed = len(fold_results)
        n_skipped = len(skipped_folds)
        all_passed = all(fold_passes) if fold_passes else None

        if all_passed is True:
            verdict = "PASSED — no spurious effects detected in any fold."
        elif all_passed is False:
            n_failed = sum(not p for p in fold_passes)
            verdict = (
                f"FAILED — {n_failed} of {n_completed} fold(s) detected "
                f"a spurious effect."
            )
        else:
            verdict = "INCONCLUSIVE — no folds completed."

        parts = [
            f"Placebo-in-time analysis: {n_completed} of {self.n_folds} folds completed"
        ]
        if n_skipped:
            parts[0] += f" ({n_skipped} skipped)"
        parts[0] += "."
        parts.append(verdict)
        parts.extend(fold_summaries)
        text = "\n".join(parts)

        return CheckResult(
            check_name="PlaceboInTime",
            passed=all_passed,
            text=text,
            metadata={"fold_results": fold_results},
        )

    def __repr__(self) -> str:
        """Return a string representation of the check."""
        return f"PlaceboInTime(n_folds={self.n_folds})"
