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
        Two-sided significance level for each pre-treatment estimate.
        Applied when a row has a usable ``att_std`` and ``n_obs`` > 1.
        Stored ``att_lower`` / ``att_upper`` bounds are compared at this
        level only when their coverage, read from ``result.hdi_prob``,
        determines the result. Otherwise that row is not reported as a
        pass or a fail at ``threshold``. A point estimate with no usable
        standard error and no interval is not a significance test:
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

        coverage = _stored_coverage(experiment)
        judgments = [
            _row_judgment(row, self.threshold, coverage) for _, row in usable.iterrows()
        ]
        differs = np.fromiter(
            (status == "differs" for status, _tested in judgments),
            dtype=bool,
            count=len(judgments),
        )
        untested = np.fromiter(
            (status == "untested" for status, _tested in judgments),
            dtype=bool,
            count=len(judgments),
        )
        tested_at_threshold = [tested for _status, tested in judgments]
        event_times = [
            int(value) for value in usable.iloc[np.flatnonzero(differs)]["event_time"]
        ]
        untested_times = [
            int(value) for value in usable.iloc[np.flatnonzero(untested)]["event_time"]
        ]
        mean_pre_att = float(usable["att"].mean())
        max_abs_pre_att = float(usable["att"].abs().max())

        if event_times:
            passed: bool | None = False
            listed = ", ".join(str(value) for value in event_times)
            text = (
                "Pre-treatment placebo check failed: pre-treatment "
                f"event-study estimates at event times {listed} differ from "
                f"zero at threshold {self.threshold}, suggesting a possible "
                "violation of the parallel trends assumption."
            )
        elif untested_times:
            passed = None
            listed = ", ".join(str(value) for value in untested_times)
            text = (
                "Pre-treatment placebo check was not completed: event times "
                f"{listed} were not tested at threshold {self.threshold} "
                "because they lack a standard error or interval that decides "
                "that level."
            )
        elif all(tested_at_threshold):
            passed = True
            text = (
                "Pre-treatment placebo check passed: no pre-treatment "
                f"event-study estimate differs from zero at threshold "
                f"{self.threshold}, consistent with parallel trends."
            )
        elif np.all(np.isclose(usable["att"].to_numpy(dtype=float), 0.0)):
            passed = True
            text = (
                "Pre-treatment placebo check passed: pre-treatment "
                "event-study estimates are zero."
            )
        else:
            passed = True
            text = (
                "Pre-treatment placebo check passed: no pre-treatment "
                "event-study estimate differs from zero."
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
                "event_times_untested": untested_times,
            },
        )


def _row_judgment(
    row: pd.Series, alpha: float, coverage: float | None
) -> tuple[str, bool]:
    """Classify one event-time row.

    Returns ``("differs"|"compatible"|"untested", tested_at_threshold)``.
    ``tested_at_threshold`` is true only when the status was decided by a
    comparison at ``alpha``.
    """
    att = float(row["att"])
    if not np.isfinite(att):
        return "untested", False

    std = _finite_or_none(row, "att_std")
    n_obs = _finite_or_none(row, "n_obs")
    if std is not None and n_obs is not None and n_obs > 1 and std >= 0:
        se = std / np.sqrt(n_obs)
        if se == 0.0 or not np.isfinite(se):
            differs = not np.isclose(att, 0.0)
            return ("differs" if differs else "compatible"), True
        p_value = float(2 * t.sf(abs(att / se), n_obs - 1))
        return ("differs" if p_value < alpha else "compatible"), True

    lower = _finite_or_none(row, "att_lower")
    upper = _finite_or_none(row, "att_upper")
    if lower is not None and upper is not None:
        return _bounds_judgment(lower, upper, alpha, coverage)

    if np.isclose(att, 0.0):
        return "compatible", False
    return "untested", False


def _bounds_judgment(
    lower: float, upper: float, alpha: float, coverage: float | None
) -> tuple[str, bool]:
    """Judge equal-tailed bounds only when their coverage decides ``alpha``.

    A stored central interval at coverage ``c`` rejects zero at level
    ``1 - c``. A wider requested interval is decided only when that stored
    interval already includes zero; a narrower one is decided only when the
    stored interval already excludes zero. Every other comparison is
    untested, including unknown coverage.
    """
    if (
        coverage is None
        or not np.isfinite(coverage)
        or not (0.0 < coverage < 1.0)
        or not np.isfinite(alpha)
        or not (0.0 < alpha < 1.0)
        or lower > upper
    ):
        return "untested", False

    stored_alpha = 1.0 - coverage
    excludes_zero = not (lower <= 0.0 <= upper)
    if excludes_zero:
        if alpha > stored_alpha or np.isclose(alpha, stored_alpha):
            return "differs", True
        return "untested", False
    if alpha < stored_alpha or np.isclose(alpha, stored_alpha):
        return "compatible", True
    return "untested", False


def _stored_coverage(experiment: BaseExperiment) -> float | None:
    result = getattr(experiment, "result", None)
    if result is None:
        return None
    raw = getattr(result, "hdi_prob", None)
    if isinstance(raw, bool) or not isinstance(raw, (int, float, np.floating)):
        return None
    coverage = float(raw)
    if not np.isfinite(coverage) or not (0.0 < coverage < 1.0):
        return None
    return coverage


def _finite_or_none(row: pd.Series, column: str) -> float | None:
    if column not in row.index:
        return None
    value = row[column]
    if value is None or not np.isfinite(value):
        return None
    return float(value)
