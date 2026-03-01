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

from abc import ABC, abstractmethod
from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import RegressorMixin

from causalpy.maketables_adapters import get_maketables_adapter
from causalpy.pymc_models import PyMCModel
from causalpy.reporting import EffectSummary
from causalpy.skl_models import create_causalpy_compatible_class


class BaseExperiment(ABC):
    """Base class for quasi experimental designs.

    Subclasses should set ``_default_model_class`` to a PyMC model class
    (e.g. ``LinearRegression``) so that ``model=None`` instantiates a sensible
    Bayesian default. To use an OLS/sklearn model, pass one explicitly.

    Notes
    -----
    Optional ``maketables`` integration is exposed through ``__maketables_*``
    hooks. Users can control the HDI interval level used by
    ``ETable(result)`` via :meth:`set_maketables_options`, for example:
    ``result.set_maketables_options(hdi_prob=0.95)``.
    """

    labels: list[str]

    supports_bayes: bool
    supports_ols: bool

    _default_model_class: type[PyMCModel] | None = None

    def __init__(self, model: PyMCModel | RegressorMixin | None = None) -> None:
        # Ensure we've made any provided Scikit Learn model (as identified as being type
        # RegressorMixin) compatible with CausalPy by appending our custom methods.
        if isinstance(model, RegressorMixin):
            model = create_causalpy_compatible_class(model)

        if model is None and self._default_model_class is not None:
            model = self._default_model_class()

        if model is not None:
            self.model = model

        if getattr(self, "model", None) is None:
            raise ValueError("model not set or passed.")

        if isinstance(self.model, PyMCModel) and not self.supports_bayes:
            raise ValueError("Bayesian models not supported.")

        if isinstance(self.model, RegressorMixin) and not self.supports_ols:
            raise ValueError("OLS models not supported.")

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

    def set_maketables_options(self, *, hdi_prob: float | None = None) -> None:
        """Set optional maketables rendering options for this experiment.

        Parameters
        ----------
        hdi_prob : float, optional
            Bayesian HDI probability used for PyMC coefficient interval columns in
            ``__maketables_coef_table__`` and therefore in ``ETable(result)``.
            Must satisfy ``0 < hdi_prob < 1``.

        Examples
        --------
        >>> result.set_maketables_options(hdi_prob=0.95)  # doctest: +SKIP
        >>> # Subsequent ETable(result) calls use 95% HDI bounds
        """
        if hdi_prob is not None:
            hdi_prob = float(hdi_prob)
            if not 0 < hdi_prob < 1:
                msg = f"hdi_prob must be in (0, 1), got {hdi_prob!r}"
                raise ValueError(msg)
            self._maketables_hdi_prob = hdi_prob

    @property
    def __maketables_coef_table__(self) -> pd.DataFrame:
        """Optional maketables plugin hook for coefficient tables.

        For PyMC-backed experiments, interval columns use the HDI probability set
        by :meth:`set_maketables_options` (or backend defaults if not set).
        """
        return get_maketables_adapter(self.model).coef_table(self)

    def __maketables_stat__(self, key: str) -> Any:
        """Optional maketables plugin hook for model-level statistics."""
        return get_maketables_adapter(self.model).stat(self, key)

    @property
    def __maketables_depvar__(self) -> str:
        """Optional maketables plugin hook for dependent variable name."""
        return str(
            getattr(
                self,
                "outcome_variable_name",
                getattr(self, "outcome_variable", "y"),
            )
        )

    @property
    def __maketables_vcov_info__(self) -> dict[str, Any]:
        """Optional maketables plugin hook for variance-covariance info."""
        return get_maketables_adapter(self.model).vcov_info(self)

    @property
    def __maketables_stat_labels__(self) -> dict[str, str] | None:
        """Optional maketables plugin hook for statistic labels."""
        return get_maketables_adapter(self.model).stat_labels(self)

    @property
    def __maketables_default_stat_keys__(self) -> list[str] | None:
        """Optional maketables plugin hook for default statistic rows."""
        return get_maketables_adapter(self.model).default_stat_keys(self)

    def plot(self, *args: Any, show: bool = True, **kwargs: Any) -> tuple:
        """Plot the model.

        Internally, this function dispatches to either `_bayesian_plot` or `_ols_plot`
        depending on the model type.

        Parameters
        ----------
        show : bool, optional
            Whether to automatically display the plot. Defaults to True.
            Set to False if you want to modify the figure before displaying it.
        """
        # Apply arviz-darkgrid style only during plotting, then revert
        with plt.style.context(az.style.library["arviz-darkgrid"]):
            if isinstance(self.model, PyMCModel):
                fig, ax = self._bayesian_plot(*args, **kwargs)
            elif isinstance(self.model, RegressorMixin):
                fig, ax = self._ols_plot(*args, **kwargs)
            else:
                raise ValueError("Unsupported model type")

        if show:
            plt.show()

        return fig, ax

    def _bayesian_plot(self, *args: Any, **kwargs: Any) -> tuple:
        """Plot results for Bayesian models. Override in subclasses that support Bayesian."""
        raise NotImplementedError("_bayesian_plot method not yet implemented")

    def _ols_plot(self, *args: Any, **kwargs: Any) -> tuple:
        """Plot results for OLS models. Override in subclasses that support OLS."""
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

    def get_plot_data_bayesian(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Return plot data for Bayesian models. Override in subclasses that support Bayesian."""
        raise NotImplementedError("get_plot_data_bayesian method not yet implemented")

    def get_plot_data_ols(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Return plot data for OLS models. Override in subclasses that support OLS."""
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
            Object with .table (DataFrame) and .text (str) attributes.
            The .text attribute contains a detailed multi-paragraph narrative report.
        """
        raise NotImplementedError("effect_summary method not yet implemented")
