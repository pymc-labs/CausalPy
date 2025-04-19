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
import arviz as az

import causalpy.pymc_experiments as pymc_experiments  # to be deprecated
import causalpy.pymc_models as pymc_models
import causalpy.skl_experiments as skl_experiments  # to be deprecated
import causalpy.skl_models as skl_models
from causalpy.skl_models import create_causalpy_compatible_class
from causalpy.version import __version__

from .data import load_data
from .experiments.diff_in_diff import DifferenceInDifferences
from .experiments.instrumental_variable import InstrumentalVariable
from .experiments.interrupted_time_series import InterruptedTimeSeries
from .experiments.inverse_propensity_weighting import InversePropensityWeighting
from .experiments.prepostnegd import PrePostNEGD
from .experiments.regression_discontinuity import RegressionDiscontinuity
from .experiments.regression_kink import RegressionKink
from .experiments.synthetic_control import SyntheticControl

az.style.use("arviz-darkgrid")

__all__ = [
    "__version__",
    "DifferenceInDifferences",
    "create_causalpy_compatible_class",
    "InstrumentalVariable",
    "InterruptedTimeSeries",
    "InversePropensityWeighting",
    "load_data",
    "PrePostNEGD",
    "pymc_experiments",  # to be deprecated
    "pymc_models",
    "RegressionDiscontinuity",
    "RegressionKink",
    "skl_experiments",  # to be deprecated
    "skl_models",
    "SyntheticControl",
]
