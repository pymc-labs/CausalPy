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

Evaluates whether each pre-treatment event-study estimate differs from zero.
Equal-and-opposite pre-trends are not evidence of parallel trends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t

from causalpy.checks.base import CheckResult
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.staggered_did import StaggeredDifferenceInDifferences
from causalpy.pipeline import PipelineContext


class PreTreatmentPlaceboCheck:
    """Check that each pre-treatment event-study estimate is near zero.

    Wraps the pre-treatment placebo effects already computed by
    ``StaggeredDifferenceInDifferences`` in its result bundle
    (``result.att_event_time``). Estimates are judged one event time at a
    time. A mean of zero produced by cancelling pre-trends is not a pass.

    Parameters
    ----------
    threshold : float, default 0.05
        Significance threshold for determining if pre-treatment effects
        are significantly different from zero. Used as the two-sided alpha
        when an event-time row has ``att_std`` and ``n_obs``. Stored
        ``att_lower`` / ``att_upper`` bounds are significant when they
        exclude zero. A point estimate with neither a standard deviation
        nor an interval differs from zero unless it is numerically zero:
        ``threshold`` is not an absolute tolerance on the ATT.
    """

    applicable_methods: set[type[BaseExperiment]] = {
        StaggeredDifferenceInDifferences,
    }

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    def validate(self, experiment: BaseExperiment) -> None:
        """Verify the experiment is a fitted StaggeredDifferenceInDifferences.

        Parameters
        ----------
        experiment : BaseExperiment
            Candidate experiment to validate.
        """
        if not isinstance(experiment, StaggeredDifferenceInDifferences):
            raise TypeError(
                "PreTreatmentPlaceboCheck requires a "
                "StaggeredDifferenceInDifferences experiment."
            )
        if not experiment.is_fitted:
            raise ValueError(
                "Experiment has not been fitted. "
                "Ensure fit() has been called before running this check."
            )

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext,
    ) -> CheckResult:
        """Evaluate pre-treatment event-study ATTs for evidence of pre-trends.

        Parameters
        ----------
        experiment : BaseExperiment
            The fitted StaggeredDifferenceInDifferences experiment.
        context : PipelineContext
            Pipeline context (unused; required by the check protocol).
        """
        sdid = experiment
        att_et = sdid.result.att_event_time
        pre_treatment = att_et[att_et["event_time"] < 0].copy()
        if "identified" in pre_treatment.columns:
            pre_treatment = pre_treatment[pre_treatment["identified"].fillna(False)]

        if pre_treatment.empty:
            return CheckResult(
                check_name="PreTreatmentPlaceboCheck",
                passed=None,
                text="No pre-treatment event times available for placebo check.",
            )

        usable = pre_treatment[np.isfinite(pre_treatment["att"].to_numpy(dtype=float))]
        if usable.empty:
            return CheckResult(
                check_name="PreTreatmentPlaceboCheck",
                passed=None,
                text="No pre-treatment event times available for placebo check.",
                table=pre_treatment,
            )

        differs = np.fromiter(
            (
                _estimate_differs_from_zero(row, self.threshold)
                for _, row in usable.iterrows()
            ),
            dtype=bool,
            count=len(usable),
        )
        differing = usable.iloc[np.flatnonzero(differs)]
        passed = differing.empty
        mean_pre_att = float(usable["att"].mean())
        max_abs_pre_att = float(usable["att"].abs().max())
        event_times = [int(value) for value in differing["event_time"].tolist()]

        if passed:
            text = (
                "Pre-treatment placebo check passed: no pre-treatment "
                f"event-study estimate differs from zero at threshold "
                f"{self.threshold}, consistent with parallel trends."
            )
        else:
            listed = ", ".join(str(value) for value in event_times)
            text = (
                "Pre-treatment placebo check failed: pre-treatment "
                f"event-study estimates at event times {listed} differ from "
                "zero, suggesting a possible violation of the parallel "
                "trends assumption."
            )

        return CheckResult(
            check_name="PreTreatmentPlaceboCheck",
            passed=passed,
            table=pre_treatment,
            text=text,
            metadata={
                "mean_pre_att": mean_pre_att,
                "max_abs_pre_att": max_abs_pre_att,
                "event_times_differing": event_times,
            },
        )


def _estimate_differs_from_zero(row: pd.Series, alpha: float) -> bool:
    """Return whether one event-time estimate differs from zero.

    ``att_std`` is the sample standard deviation of the observation-level
    effects in that event-time bin, so the standard error of the mean is
    ``att_std / sqrt(n_obs)``.
    """
    att = float(row["att"])
    if not np.isfinite(att):
        return False

    std = _finite_or_none(row, "att_std")
    n_obs = _finite_or_none(row, "n_obs")
    if std is not None and n_obs is not None and n_obs > 1 and std >= 0:
        se = std / np.sqrt(n_obs)
        if se == 0.0 or not np.isfinite(se):
            return not np.isclose(att, 0.0)
        p_value = float(2 * t.sf(abs(att / se), n_obs - 1))
        return p_value < alpha

    lower = _finite_or_none(row, "att_lower")
    upper = _finite_or_none(row, "att_upper")
    if lower is not None and upper is not None:
        return not (lower <= 0.0 <= upper)

    return not np.isclose(att, 0.0)


def _finite_or_none(row: pd.Series, column: str) -> float | None:
    if column not in row.index:
        return None
    value = row[column]
    if value is None or not np.isfinite(value):
        return None
    return float(value)
