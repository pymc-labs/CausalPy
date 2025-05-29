#   Copyright 2022 - 2025 The PyMC Labs Developers
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
This module exists to maintain the old API where experiment classes were defined in the
`causalpy.skl_experiments` namespace. They have moved to `causalpy.experiments` and
this module is a thin wrapper around the new classes to maintain backwards
compatibility. A deprecation warning is delivered to the user. This module may be
removed in a future release.
"""

import warnings

from .experiments.diff_in_diff import (
    DifferenceInDifferences as NewDifferenceInDifferences,
)
from .experiments.interrupted_time_series import (
    InterruptedTimeSeries as DeprecatedInterruptedTimeSeriesFromExperiments,
)
from .experiments.regression_discontinuity import (
    RegressionDiscontinuity as NewRegressionDiscontinuity,
)
from .experiments.synthetic_control import (
    SyntheticControl as NewSyntheticControl,
)

# Ensure deprecation warnings are always shown in Jupyter Notebooks
warnings.simplefilter("always", DeprecationWarning)
RED = "\033[91m"
RESET = "\033[0m"
BLUE = "\033[1m"


def SyntheticControl(*args, **kwargs):
    warnings.warn(
        f"""{RED}cp.pymc_experiments.SyntheticControl is deprecated and will be removed in a future release. Please use:
        import causalpy as cp
        cp.SyntheticControl(...){RESET}""",
        DeprecationWarning,
        stacklevel=2,
    )
    return NewSyntheticControl(*args, **kwargs)


def DifferenceInDifferences(*args, **kwargs):
    warnings.warn(
        f"""{RED}cp.pymc_experiments.DifferenceInDifferences is deprecated and will be removed in a future release. Please use:
        import causalpy as cp
        cp.DifferenceInDifferences(...){RESET}""",
        DeprecationWarning,
        stacklevel=2,
    )
    return NewDifferenceInDifferences(*args, **kwargs)


def InterruptedTimeSeries(*args, **kwargs):
    """DEPRECATED: Please use cp.StructuralTimeSeries(...)"""
    warnings.warn(
        f"""{RED}cp.skl_experiments.InterruptedTimeSeries is deprecated and will be removed in a future release. Please use:
        {BLUE}cp.InterruptedTimeSeries(...){RESET}""",
        DeprecationWarning,
        stacklevel=2,
    )
    return DeprecatedInterruptedTimeSeriesFromExperiments(*args, **kwargs)


def RegressionDiscontinuity(*args, **kwargs):
    warnings.warn(
        f"""{RED}cp.pymc_experiments.RegressionDiscontinuity is deprecated and will be removed in a future release. Please use:
        import causalpy as cp
        cp.RegressionDiscontinuity(...){RESET}""",
        DeprecationWarning,
        stacklevel=2,
    )
    return NewRegressionDiscontinuity(*args, **kwargs)
