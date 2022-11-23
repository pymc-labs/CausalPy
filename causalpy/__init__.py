import arviz as az

import causalpy.pymc_experiments
import causalpy.pymc_models
import causalpy.skl_experiments
import causalpy.skl_models
from causalpy.version import __version__

from .data import load_data

az.style.use("arviz-darkgrid")
