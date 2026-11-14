#   Copyright 2026 - 2026 The PyMC Labs Developers
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
"""Predeclared qualification protocol for issue #914.

These constants were fixed before the repeated-data study was run. They are
not a theorem about the estimator, and they must not be edited to match
observed rates. The 94% HDI is an interval convention. It is not an alpha and
not a false-positive tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from causalpy.constants import HDI_PROB
from causalpy.design import DecisionRule

PROTOCOL_ID = "issue-914-qualification-v1"
MASTER_SEED = 914

# Design worked example: 100 independent panels at detection probability 0.8
# have a binomial standard error of about 0.04. This delivery does not claim
# a tighter detection-probability standard error than that.
N_PANELS = 100
CALIBRATION_PANELS = 50
MAX_CLAIMED_DETECTION_SE = 0.04
DETECTION_TARGET = 0.80
RATE_INTERVAL_LEVEL = 0.95

ESTIMAND: Literal["cumulative"] = "cumulative"
DIRECTION: Literal["positive"] = "positive"
# Effect-unit threshold. Not a probability and not ``1 - HDI_PROB``.
DECISION_THRESHOLD = 0.0
# Tail probability the shipped ``DecisionRule`` requires. Not an FPR tolerance.
DECISION_PROBABILITY = 0.95

# Interval convention only. Coverage is not the decision and not a power curve.
COVERAGE_HDI_PROB = HDI_PROB

# Withheld. A number here would pretend the HDI or the tail probability is a
# theorem about null error. Observed null decisions are reported with a
# Clopper-Pearson interval instead.
NULL_ERROR_TOLERANCE: float | None = None
NULL_TOLERANCE_REASON = (
    "No numeric null-error tolerance is claimed. The 94% HDI is an interval "
    "convention, and the 0.95 tail probability is the decision rule. Neither "
    "is a false-positive tolerance."
)

ESTIMATOR = "synthetic_control"
MODEL_NAME = "WeightedSumFitter"

N_TIME = 32
DURATION = 8
N_DONORS = 3
DONORS = ("c0", "c1", "c2")
TREATED = "treated"
WEIGHTS = (0.5, 0.25, 0.25)
# Amplitude 4 keeps the mean inside a noisy hull more often than unit-scale
# factors with residual SD 1. That choice keeps the correctly specified label
# from also being a routine convex-hull failure. It was fixed before the study.
FACTOR_SCALE = 4.0
RESIDUAL_SD = 1.0
AR_PHI = 0.6
MISFIT_LEVEL = 4.0
MISFIT_TREND = 0.15
# Per-period absolute lifts. Cumulative truth is lift times duration.
# Relative lifts are not used: a noisy path can cross zero, and the shipped
# planner rejects a non-positive relative baseline.
SMALL_LIFT = 0.25
LARGE_LIFT = 1.0
ABSOLUTE_LIFTS = (SMALL_LIFT, LARGE_LIFT)

# Light relative to the model default. Two chains so the shipped calibration
# seam can compute rank r-hat; one chain leaves diagnostics partial and the
# product then refuses the learned-null curve. This is not a convergence claim.
SAMPLE_KWARGS = {
    "tune": 40,
    "draws": 40,
    "chains": 2,
    "cores": 1,
    "progressbar": False,
}

SCENARIOS = (
    "correctly_specified_noisy",
    "persistent_level_trend_misfit",
    "serially_correlated_errors",
)
SCENARIO_CODES = {
    "correctly_specified_noisy": 1,
    "persistent_level_trend_misfit": 2,
    "serially_correlated_errors": 3,
}
HISTORICAL_CODE = 4

# One generated history, not 100 independent panels. Origins overlap each
# other and do not intersect the reserved audit holdout.
HISTORICAL_LENGTH = 40
HISTORICAL_HORIZON = 8
HISTORICAL_SELECTION_ORIGINS = (16, 20, 24)
HISTORICAL_AUDIT_ORIGINS = (32,)
HISTORICAL_MIN_TRAINING = 12

SELECTED_DURATION_AUDIT = (
    "gap: no public duration-freeze API. Inspected origin/main at "
    "c4fa86726b040ba7164b9396eb5849c90d271b37 and open pull requests. "
    "Issue #721 has no open pull request. SyntheticControlPlanner has no "
    "freeze method. This delivery does not invent a private duration selector. "
    "The selected-duration audit is unmet."
)

FOLD_SD_ASSESSMENT = (
    "Fold standard deviations are posterior standard deviations of the "
    "cumulative impact, not residual observation standard deviations and not "
    "future observation noise. The effect posterior subtracts counterfactual "
    "conditional-mean draws from observed outcomes. Crossing each learned-null "
    "draw with every fold standard deviation treats bias and precision as "
    "independent. A curve fit on calibration panels does not by itself "
    "transport to audit panels or to a future experiment. Numerical exactness "
    "of the engine is not qualification."
)

# Phrases a supported claim must not contain. The policy generates the claim
# text; this list is the guard a reviewer can check.
FORBIDDEN_CLAIM_PHRASES = (
    "launch recommendation",
    "recommend launching",
    "prospective power",
    "validated false-positive",
    "universal pass threshold",
)


@dataclass(frozen=True)
class QualificationProtocol:
    """The closed protocol. Attribute access only; not a public product API."""

    protocol_id: str = PROTOCOL_ID
    master_seed: int = MASTER_SEED
    n_panels: int = N_PANELS
    calibration_panels: int = CALIBRATION_PANELS
    max_claimed_detection_se: float = MAX_CLAIMED_DETECTION_SE
    detection_target: float = DETECTION_TARGET
    rate_interval_level: float = RATE_INTERVAL_LEVEL
    estimand: str = ESTIMAND
    direction: str = DIRECTION
    decision_threshold: float = DECISION_THRESHOLD
    decision_probability: float = DECISION_PROBABILITY
    coverage_hdi_prob: float = COVERAGE_HDI_PROB
    null_error_tolerance: float | None = NULL_ERROR_TOLERANCE
    estimator: str = ESTIMATOR
    n_time: int = N_TIME
    duration: int = DURATION
    residual_sd: float = RESIDUAL_SD
    ar_phi: float = AR_PHI
    misfit_level: float = MISFIT_LEVEL
    misfit_trend: float = MISFIT_TREND
    small_lift: float = SMALL_LIFT
    large_lift: float = LARGE_LIFT
    historical_length: int = HISTORICAL_LENGTH
    historical_horizon: int = HISTORICAL_HORIZON
    historical_selection_origins: tuple[int, ...] = HISTORICAL_SELECTION_ORIGINS
    historical_audit_origins: tuple[int, ...] = HISTORICAL_AUDIT_ORIGINS
    historical_min_training: int = HISTORICAL_MIN_TRAINING
    selected_duration_audit: str = SELECTED_DURATION_AUDIT

    @property
    def treatment_time(self) -> int:
        """First holdout label when the index is ``RangeIndex(n_time)``."""
        return self.n_time - self.duration

    def decision_rule(self) -> DecisionRule:
        """The shipped one-sided rule this study scores. Not an HDI rule."""
        return DecisionRule(
            "positive",
            "cumulative",
            threshold=self.decision_threshold,
            probability=self.decision_probability,
        )

    def cumulative_truth(self, lift: float) -> float:
        """Cumulative truth of a per-period absolute lift on this duration."""
        return float(lift) * self.duration


PROTOCOL = QualificationProtocol()


def sample_kwargs(random_seed: int) -> dict[str, int | bool]:
    """Copy the light sampling settings and attach one sampler seed.

    The seed is computation variability for that panel. It is not another dataset.
    """
    copied: dict[str, int | bool] = dict(SAMPLE_KWARGS)
    copied["random_seed"] = random_seed
    return copied
