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
"""CausalPy experiment module"""

from .diff_in_diff import DifferenceInDifferences
from .event_study import EventStudy
from .instrumental_variable import InstrumentalVariable
from .interrupted_time_series import InterruptedTimeSeries
from .inverse_propensity_weighting import InversePropensityWeighting
from .prepostnegd import PrePostNEGD
from .regression_discontinuity import RegressionDiscontinuity
from .regression_kink import RegressionKink
from .staggered_did import StaggeredDifferenceInDifferences
from .synthetic_control import SyntheticControl

__all__ = [
    "DifferenceInDifferences",
    "EventStudy",
    "InstrumentalVariable",
    "InversePropensityWeighting",
    "PrePostNEGD",
    "RegressionDiscontinuity",
    "RegressionKink",
    "StaggeredDifferenceInDifferences",
    "SyntheticControl",
    "InterruptedTimeSeries",
]
