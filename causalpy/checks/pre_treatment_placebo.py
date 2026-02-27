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
Pre-treatment placebo check for Staggered DiD experiments.

Evaluates whether pre-treatment event-study estimates are near zero,
which validates the parallel trends assumption.
"""

from __future__ import annotations

import numpy as np

from causalpy.checks.base import CheckResult
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.staggered_did import StaggeredDifferenceInDifferences
from causalpy.pipeline import PipelineContext


class PreTreatmentPlaceboCheck:
    """Check that pre-treatment event-study estimates are near zero.

    Wraps the pre-treatment placebo effects already computed by
    ``StaggeredDifferenceInDifferences`` in ``att_event_time_``.

    Parameters
    ----------
    threshold : float, default 0.05
        Significance threshold for determining if pre-treatment effects
        are significantly different from zero.
    """

    applicable_methods: set[type[BaseExperiment]] = {
        StaggeredDifferenceInDifferences,
    }

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    def validate(self, experiment: BaseExperiment) -> None:
        if not isinstance(experiment, StaggeredDifferenceInDifferences):
            raise TypeError(
                "PreTreatmentPlaceboCheck requires a "
                "StaggeredDifferenceInDifferences experiment."
            )
        if not hasattr(experiment, "att_event_time_"):
            raise ValueError(
                "Experiment does not have att_event_time_. "
                "Ensure the experiment has been fitted."
            )

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext,
    ) -> CheckResult:
        sdid = experiment
        att_et = sdid.att_event_time_  # type: ignore[attr-defined]

        pre_treatment = att_et[att_et["event_time"] < 0].copy()

        if pre_treatment.empty:
            return CheckResult(
                check_name="PreTreatmentPlaceboCheck",
                passed=None,
                text="No pre-treatment event times available for placebo check.",
            )

        mean_pre_att = pre_treatment["att"].mean()
        max_abs_pre_att = pre_treatment["att"].abs().max()

        passed = bool(
            np.isclose(mean_pre_att, 0, atol=max_abs_pre_att * self.threshold)
        )

        if passed:
            text = (
                f"Pre-treatment placebo check passed: mean pre-treatment "
                f"ATT = {mean_pre_att:.4f}, consistent with parallel trends."
            )
        else:
            text = (
                f"Pre-treatment placebo check failed: mean pre-treatment "
                f"ATT = {mean_pre_att:.4f}, suggesting possible violation "
                f"of the parallel trends assumption."
            )

        return CheckResult(
            check_name="PreTreatmentPlaceboCheck",
            passed=passed,
            table=pre_treatment,
            text=text,
            metadata={
                "mean_pre_att": mean_pre_att,
                "max_abs_pre_att": max_abs_pre_att,
            },
        )
