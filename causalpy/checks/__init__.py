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
"""Sensitivity and diagnostic checks for causal inference experiments."""

from causalpy.checks.base import Check, CheckResult
from causalpy.checks.convex_hull import ConvexHullCheck
from causalpy.checks.persistence import PersistenceCheck
from causalpy.checks.placebo_in_time import PlaceboFoldResult, PlaceboInTime
from causalpy.checks.pre_treatment_placebo import PreTreatmentPlaceboCheck
from causalpy.checks.prior_sensitivity import PriorSensitivity
from causalpy.steps.sensitivity import register_default_check

__all__ = [
    "Check",
    "CheckResult",
    "ConvexHullCheck",
    "PersistenceCheck",
    "PlaceboFoldResult",
    "PlaceboInTime",
    "PreTreatmentPlaceboCheck",
    "PriorSensitivity",
]

register_default_check(PlaceboInTime, PlaceboInTime.applicable_methods)
