#   Copyright 2025 The PyMC Labs Developers
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
Base class for quasi experimental designs.
"""

from abc import abstractmethod

from sklearn.base import RegressorMixin

from causalpy.pymc_models import PyMCModel
from causalpy.skl_models import create_causalpy_compatible_class


class BaseExperiment:
    """Base class for quasi experimental designs."""

    supports_bayes: bool
    supports_ols: bool

    def __init__(self, model=None):
        # Ensure we've made any provided Scikit Learn model (as identified as being type
        # RegressorMixin) compatible with CausalPy by appending our custom methods.
        if isinstance(model, RegressorMixin):
            model = create_causalpy_compatible_class(model)

        if model is not None:
            self.model = model

        if isinstance(self.model, PyMCModel) and not self.supports_bayes:
            raise ValueError("Bayesian models not supported.")

        if isinstance(self.model, RegressorMixin) and not self.supports_ols:
            raise ValueError("OLS models not supported.")

        if self.model is None:
            raise ValueError("model not set or passed.")

    @property
    def idata(self):
        """Return the InferenceData object of the model. Only relevant for PyMC models."""
        return self.model.idata

    def print_coefficients(self, round_to=None):
        """Ask the model to print its coefficients."""
        self.model.print_coefficients(self.labels, round_to)

    def plot(self, *args, **kwargs) -> tuple:
        """Plot the model.

        Internally, this function dispatches to either `bayesian_plot` or `ols_plot`
        depending on the model type.
        """
        if isinstance(self.model, PyMCModel):
            return self.bayesian_plot(*args, **kwargs)
        elif isinstance(self.model, RegressorMixin):
            return self.ols_plot(*args, **kwargs)
        else:
            raise ValueError("Unsupported model type")

    @abstractmethod
    def bayesian_plot(self, *args, **kwargs):
        """Abstract method for plotting the model."""
        raise NotImplementedError("bayesian_plot method not yet implemented")

    @abstractmethod
    def ols_plot(self, *args, **kwargs):
        """Abstract method for plotting the model."""
        raise NotImplementedError("ols_plot method not yet implemented")
