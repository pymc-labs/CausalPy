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
Persistence diagnostic check for three-period ITS experiments.

Wraps ``InterruptedTimeSeries.analyze_persistence()`` to report
whether the intervention effect persists after the intervention ends.
Only applicable when ``treatment_end_time`` is set.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from causalpy.checks.base import CheckResult
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.pipeline import PipelineContext


class PersistenceCheck:
    """Check whether the ITS effect persists after the intervention ends.

    Wraps ``InterruptedTimeSeries.analyze_persistence()``.
    Only applicable to three-period ITS designs (with ``treatment_end_time``).

    Parameters
    ----------
    hdi_prob : float, default 0.95
        HDI probability (Bayesian models only).
    direction : str, default "increase"
        Tail probability direction.
    """

    applicable_methods: set[type[BaseExperiment]] = {InterruptedTimeSeries}

    def __init__(
        self,
        hdi_prob: float = 0.95,
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
    ) -> None:
        self.hdi_prob = hdi_prob
        self.direction = direction

    def validate(self, experiment: BaseExperiment) -> None:
        """Verify the experiment is a three-period ITS with treatment_end_time."""
        if not isinstance(experiment, InterruptedTimeSeries):
            raise TypeError(
                "PersistenceCheck requires an InterruptedTimeSeries experiment."
            )
        if (
            not hasattr(experiment, "treatment_end_time")
            or experiment.treatment_end_time is None
        ):
            raise ValueError(
                "PersistenceCheck requires a three-period ITS design "
                "(treatment_end_time must be set)."
            )

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext,
    ) -> CheckResult:
        """Run persistence analysis and report whether the effect decays."""
        its: Any = experiment
        persistence = its.analyze_persistence(
            hdi_prob=self.hdi_prob,
            direction=self.direction,
        )

        table = pd.DataFrame(
            [
                {
                    "metric": "mean_effect_during",
                    "value": persistence["mean_effect_during"],
                },
                {
                    "metric": "mean_effect_post",
                    "value": persistence["mean_effect_post"],
                },
                {
                    "metric": "persistence_ratio",
                    "value": persistence["persistence_ratio"],
                },
                {
                    "metric": "total_effect_during",
                    "value": persistence["total_effect_during"],
                },
                {
                    "metric": "total_effect_post",
                    "value": persistence["total_effect_post"],
                },
            ]
        )

        ratio = persistence["persistence_ratio"]
        text = (
            f"Persistence analysis: the effect persistence ratio is "
            f"{ratio:.2f}. A ratio near 1.0 indicates the effect fully "
            f"persists after the intervention ends; near 0.0 indicates "
            f"the effect decays completely."
        )

        return CheckResult(
            check_name="PersistenceCheck",
            passed=None,
            table=table,
            text=text,
            metadata={"persistence": persistence},
        )
