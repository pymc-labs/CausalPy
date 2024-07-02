#   Copyright 2024 The PyMC Labs Developers
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

import causalpy.pymc_models as pymc_models
import causalpy.skl_models as skl_models
from causalpy.version import __version__

from .data import load_data
from .exp_inverse_propensity_weighting import InversePropensityWeighting
from .expt_diff_in_diff import DifferenceInDifferences
from .expt_instrumental_variable import InstrumentalVariable
from .expt_prepostfit import InterruptedTimeSeries, SyntheticControl
from .expt_prepostnegd import PrePostNEGD
from .expt_regression_discontinuity import RegressionDiscontinuity
from .expt_regression_kink import RegressionKink

az.style.use("arviz-darkgrid")

__all__ = [
    "InterruptedTimeSeries",
    "SyntheticControl",
    "DifferenceInDifferences",
    "PrePostNEGD",
    "RegressionDiscontinuity",
    "RegressionKink",
    "InstrumentalVariable",
    "InversePropensityWeighting",
    "pymc_models",
    "skl_models",
    "load_data",
    "__version__",
]
