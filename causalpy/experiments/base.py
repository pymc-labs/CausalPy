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
"""
Base class for quasi experimental designs.
"""

from abc import abstractmethod
from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import RegressorMixin, clone

from causalpy.pymc_models import PyMCModel
from causalpy.reporting import EffectSummary
from causalpy.skl_models import create_causalpy_compatible_class


class BaseExperiment:
    """Base class for quasi experimental designs."""

    labels: list[str]

    supports_bayes: bool
    supports_ols: bool

    def __init__(self, model: PyMCModel | RegressorMixin | None = None) -> None:
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

    def _ensure_sklearn_fit_intercept_false(self) -> None:
        """Ensure scikit-learn models use fit_intercept=False without mutating user models."""
        if not isinstance(self.model, RegressorMixin):
            return
        if not hasattr(self.model, "fit_intercept"):
            return
        if getattr(self.model, "fit_intercept", False) is False:
            return

        try:
            cloned_model = clone(self.model)
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive for non-cloneable estimators
            raise ValueError(
                "This experiment requires a scikit-learn estimator with "
                "fit_intercept=False. Set fit_intercept=False on your estimator or "
                "pass an estimator that supports sklearn.base.clone()."
            ) from exc

        if hasattr(cloned_model, "set_params"):
            try:
                cloned_model.set_params(fit_intercept=False)
            except ValueError:
                cloned_model.fit_intercept = False
        else:
            cloned_model.fit_intercept = False

        self.model = create_causalpy_compatible_class(cloned_model)

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

    @abstractmethod
    def effect_summary(
        self,
        *,
        window: Literal["post"] | tuple | slice = "post",
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        cumulative: bool = True,
        relative: bool = True,
        min_effect: float | None = None,
        treated_unit: str | None = None,
        period: Literal["intervention", "post", "comparison"] | None = None,
        prefix: str = "Post-period",
        **kwargs: Any,
    ) -> EffectSummary:
        """
        Generate a decision-ready summary of causal effects.

        Parameters
        ----------
        window : str, tuple, or slice, default="post"
            Time window for analysis (ITS/SC only, ignored for DiD/RD):
            - "post": All post-treatment time points (default)
            - (start, end): Tuple of start and end times (handles both datetime and integer indices)
            - slice: Python slice object for integer indices
        direction : {"increase", "decrease", "two-sided"}, default="increase"
            Direction for tail probability calculation (PyMC only, ignored for OLS):
            - "increase": P(effect > 0)
            - "decrease": P(effect < 0)
            - "two-sided": Two-sided p-value, report 1-p as "probability of effect"
        alpha : float, default=0.05
            Significance level for HDI/CI intervals (1-alpha confidence level)
        cumulative : bool, default=True
            Whether to include cumulative effect statistics (ITS/SC only, ignored for DiD/RD)
        relative : bool, default=True
            Whether to include relative effect statistics (% change vs counterfactual)
            (ITS/SC only, ignored for DiD/RD)
        min_effect : float, optional
            Region of Practical Equivalence (ROPE) threshold (PyMC only, ignored for OLS).
            If provided, reports P(|effect| > min_effect) for two-sided or P(effect > min_effect) for one-sided.
        treated_unit : str, optional
            For multi-unit experiments (Synthetic Control), specify which treated unit
            to analyze. If None and multiple units exist, uses first unit.
        period : {"intervention", "post", "comparison"}, optional
            For experiments with multiple periods (e.g., three-period ITS), specify
            which period to summarize. Defaults to None for standard behavior.
        prefix : str, optional
            Prefix for prose generation (e.g., "During intervention", "Post-intervention").
            Defaults to "Post-period".

        Returns
        -------
        EffectSummary
            Object with .table (DataFrame) and .text (str) attributes
        """
        raise NotImplementedError("effect_summary method not yet implemented")
