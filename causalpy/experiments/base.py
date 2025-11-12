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
Base class for quasi experimental designs.
"""

from abc import abstractmethod
from typing import Any, Union

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import RegressorMixin

from causalpy.pymc_models import PyMCModel
from causalpy.skl_models import create_causalpy_compatible_class


class BaseExperiment:
    """Base class for quasi experimental designs."""

    labels: list[str]

    supports_bayes: bool
    supports_ols: bool

    def __init__(self, model: Union[PyMCModel, RegressorMixin] | None = None) -> None:
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

    def fit(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("fit method not implemented")

    @property
    def idata(self) -> az.InferenceData:
        """Return the InferenceData object of the model. Only relevant for PyMC models."""
        return self.model.idata

    def print_coefficients(self, round_to: int | None = None) -> None:
        """Ask the model to print its coefficients.

        Parameters
        ----------
        round_to : int, optional
            Number of significant figures to round to. Defaults to None,
            in which case 2 significant figures are used.
        """
        self.model.print_coefficients(self.labels, round_to)

    def plot(self, *args: Any, **kwargs: Any) -> tuple:
        """Plot the model.

        Internally, this function dispatches to either `_bayesian_plot` or `_ols_plot`
        depending on the model type.
        """
        # Apply arviz-darkgrid style only during plotting, then revert
        with plt.style.context(az.style.library["arviz-darkgrid"]):
            if isinstance(self.model, PyMCModel):
                return self._bayesian_plot(*args, **kwargs)
            elif isinstance(self.model, RegressorMixin):
                return self._ols_plot(*args, **kwargs)
            else:
                raise ValueError("Unsupported model type")

    @abstractmethod
    def _bayesian_plot(self, *args: Any, **kwargs: Any) -> tuple:
        """Abstract method for plotting the model."""
        raise NotImplementedError("_bayesian_plot method not yet implemented")

    @abstractmethod
    def _ols_plot(self, *args: Any, **kwargs: Any) -> tuple:
        """Abstract method for plotting the model."""
        raise NotImplementedError("_ols_plot method not yet implemented")

    def get_plot_data(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Recover the data of an experiment along with the prediction and causal impact information.

        Internally, this function dispatches to either :func:`get_plot_data_bayesian` or :func:`get_plot_data_ols`
        depending on the model type.
        """
        if isinstance(self.model, PyMCModel):
            return self.get_plot_data_bayesian(*args, **kwargs)
        elif isinstance(self.model, RegressorMixin):
            return self.get_plot_data_ols(*args, **kwargs)
        else:
            raise ValueError("Unsupported model type")

    @abstractmethod
    def get_plot_data_bayesian(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Abstract method for recovering plot data."""
        raise NotImplementedError("get_plot_data_bayesian method not yet implemented")

    @abstractmethod
    def get_plot_data_ols(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Abstract method for recovering plot data."""
        raise NotImplementedError("get_plot_data_ols method not yet implemented")
