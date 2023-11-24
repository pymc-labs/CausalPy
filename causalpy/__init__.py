import arviz as az

from causalpy import pymc_experiments, pymc_models, skl_experiments, skl_models
from causalpy.version import __version__

from .data import load_data

az.style.use("arviz-darkgrid")

__all__ = [
    "pymc_experiments",
    "pymc_models",
    "skl_experiments",
    "skl_models",
    "load_data",
    "__version__",
]
