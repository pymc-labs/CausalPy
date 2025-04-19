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
`causalpy.pymc_experiments` namespace. They have moved to `causalpy.experiments` and
this module is a thin wrapper around the new classes to maintain backwards
compatibility. A deprecation warning is delivered to the user. This module may be
removed in a future release.
"""

import warnings

from .experiments.diff_in_diff import (
    DifferenceInDifferences as NewDifferenceInDifferences,
)
from .experiments.instrumental_variable import (
    InstrumentalVariable as NewInstrumentalVariable,
)
from .experiments.interrupted_time_series import (
    InterruptedTimeSeries as NewInterruptedTimeSeries,
)
from .experiments.inverse_propensity_weighting import (
    InversePropensityWeighting as NewInversePropensityWeighting,
)
from .experiments.prepostnegd import PrePostNEGD as NewPrePostNEGD
from .experiments.regression_discontinuity import (
    RegressionDiscontinuity as NewRegressionDiscontinuity,
)
from .experiments.regression_kink import RegressionKink as NewRegressionKink
from .experiments.synthetic_control import (
    SyntheticControl as NewSyntheticControl,
)

# Ensure deprecation warnings are always shown in Jupyter Notebooks
warnings.simplefilter("always", DeprecationWarning)
RED = "\033[91m"
RESET = "\033[0m"


def PrePostNEGD(*args, **kwargs):
    warnings.warn(
        f"""{RED}cp.pymc_experiments.PrePostNEGD is deprecated and will be removed in a future release. Please use:
        import causalpy as cp
        cp.PrePostNEGD(...){RESET}""",
        DeprecationWarning,
        stacklevel=2,
    )
    return NewPrePostNEGD(*args, **kwargs)


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
    warnings.warn(
        f"""{RED}cp.pymc_experiments.InterruptedTimeSeries is deprecated and will be removed in a future release. Please use:
        import causalpy as cp
        cp.InterruptedTimeSeries(...){RESET}""",
        DeprecationWarning,
        stacklevel=2,
    )
    return NewInterruptedTimeSeries(*args, **kwargs)


def SyntheticControl(*args, **kwargs):
    warnings.warn(
        f"""{RED}cp.pymc_experiments.SyntheticControl is deprecated and will be removed in a future release. Please use:
        import causalpy as cp
        cp.SyntheticControl(...){RESET}""",
        DeprecationWarning,
        stacklevel=2,
    )
    return NewSyntheticControl(*args, **kwargs)


def RegressionKink(*args, **kwargs):
    warnings.warn(
        f"""{RED}cp.pymc_experiments.RegressionKink is deprecated and will be removed in a future release. Please use:
        import causalpy as cp
        cp.RegressionKink(...){RESET}""",
        DeprecationWarning,
        stacklevel=2,
    )
    return NewRegressionKink(*args, **kwargs)


def RegressionDiscontinuity(*args, **kwargs):
    warnings.warn(
        f"""{RED}cp.pymc_experiments.RegressionDiscontinuity is deprecated and will be removed in a future release. Please use:
        import causalpy as cp
        cp.RegressionDiscontinuity(...){RESET}""",
        DeprecationWarning,
        stacklevel=2,
    )
    return NewRegressionDiscontinuity(*args, **kwargs)


def InversePropensityWeighting(*args, **kwargs):
    warnings.warn(
        f"""{RED}cp.pymc_experiments.InversePropensityWeighting is deprecated and will be removed in a future release. Please use:
        import causalpy as cp
        cp.InversePropensityWeighting(...){RESET}""",
        DeprecationWarning,
        stacklevel=2,
    )
    return NewInversePropensityWeighting(*args, **kwargs)


def InstrumentalVariable(*args, **kwargs):
    warnings.warn(
        f"""{RED}cp.pymc_experiments.InstrumentalVariable is deprecated and will be removed in a future release. Please use:
        import causalpy as cp
        cp.InstrumentalVariable(...){RESET}""",
        DeprecationWarning,
        stacklevel=2,
    )
    return NewInstrumentalVariable(*args, **kwargs)
