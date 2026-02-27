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

import copy
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from causalpy.checks.base import CheckResult
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pipeline import PipelineContext

logger = logging.getLogger(__name__)


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
    """

    fold: int
    pseudo_treatment_time: Any
    experiment: BaseExperiment


class PlaceboInTime:
    """Placebo-in-time sensitivity check.

    Shifts the treatment time backward into the pre-intervention period
    to create ``n_folds`` placebo experiments.  If any placebo fold shows
    a significant effect, the causal claim may be spurious.

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
    >>> check = cp.checks.PlaceboInTime(n_folds=3)
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
        self.fold_results: list[PlaceboFoldResult] = []

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
            kw = dict(kwargs)
            kw["treatment_time"] = treatment_time
            if "model" in kw and kw["model"] is not None:
                kw["model"] = copy.deepcopy(kw["model"])
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

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext,
    ) -> CheckResult:
        """Run placebo-in-time analysis.

        Creates ``n_folds`` placebo experiments by shifting the treatment
        time backward.  Each fold uses only data before the placebo
        intervention window ends, ensuring no actual treatment data leaks.

        Returns
        -------
        CheckResult
            With ``metadata["fold_results"]`` containing the
            :class:`PlaceboFoldResult` objects.
        """
        factory = self._get_factory(context)
        treatment_time = experiment.treatment_time  # type: ignore[attr-defined]
        data = experiment.data  # type: ignore[attr-defined]
        intervention_length = self._compute_intervention_length(experiment)

        fold_treatment_times = self._compute_fold_treatment_times(
            treatment_time, intervention_length
        )

        self.fold_results = []
        fold_summaries: list[str] = []

        for fold_idx, pseudo_tt in enumerate(fold_treatment_times):
            fold_num = fold_idx + 1
            logger.info(
                "PlaceboInTime fold %d/%d: pseudo_treatment_time=%s",
                fold_num,
                self.n_folds,
                pseudo_tt,
            )

            fold_data = self._get_fold_data(data, pseudo_tt, intervention_length)

            if len(fold_data) < 3:
                logger.warning(
                    "Fold %d has only %d observations, skipping.",
                    fold_num,
                    len(fold_data),
                )
                continue

            fold_experiment = factory(fold_data, pseudo_tt)
            fold_result = PlaceboFoldResult(
                fold=fold_num,
                pseudo_treatment_time=pseudo_tt,
                experiment=fold_experiment,
            )
            self.fold_results.append(fold_result)

            fold_summaries.append(f"Fold {fold_num}: pseudo treatment at {pseudo_tt}")

        n_completed = len(self.fold_results)
        text = (
            f"Placebo-in-time analysis: {n_completed} of {self.n_folds} "
            f"folds completed.\n" + "\n".join(fold_summaries)
        )

        return CheckResult(
            check_name="PlaceboInTime",
            passed=None,
            text=text,
            metadata={"fold_results": self.fold_results},
        )
