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
from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

from causalpy.pymc_models import PyMCModel
from causalpy.reporting import (
    EffectSummary,
    _compute_statistics,
    _compute_statistics_did_ols,
    _compute_statistics_ols,
    _detect_experiment_type,
    _effect_summary_did,
    _effect_summary_rd,
    _effect_summary_rkink,
    _extract_counterfactual,
    _extract_window,
    _generate_prose,
    _generate_prose_did_ols,
    _generate_prose_ols,
    _generate_table,
    _generate_table_did_ols,
    _generate_table_ols,
)
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

    def effect_summary(
        self,
        window: Literal["post"] | tuple | slice = "post",
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        cumulative: bool = True,
        relative: bool = True,
        min_effect: float | None = None,
        treated_unit: str | None = None,
        period: Literal["intervention", "post", "comparison"] | None = None,
        prefix: str = "Post-period",
        **kwargs: dict,
    ) -> EffectSummary:
        """
        Generate a decision-ready summary of causal effects.

        Supports Interrupted Time Series (ITS), Synthetic Control, Difference-in-Differences (DiD),
        and Regression Discontinuity (RD) experiments. Works with both PyMC (Bayesian) and OLS models.
        Automatically detects experiment type and model type, generating appropriate summary.

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
        # Detect experiment type
        experiment_type = _detect_experiment_type(self)

        # Check if PyMC or OLS model
        is_pymc = isinstance(self.model, PyMCModel)

        if experiment_type == "rd":
            # Regression Discontinuity: scalar effect, no time dimension
            return _effect_summary_rd(
                self,
                direction=direction,
                alpha=alpha,
                min_effect=min_effect,
            )
        elif experiment_type == "rkink":
            # Regression Kink: scalar effect (gradient change at kink point)
            return _effect_summary_rkink(
                self,
                direction=direction,
                alpha=alpha,
                min_effect=min_effect,
            )
        elif experiment_type == "did":
            # Difference-in-Differences: scalar effect, no time dimension
            if is_pymc:
                return _effect_summary_did(
                    self,
                    direction=direction,
                    alpha=alpha,
                    min_effect=min_effect,
                )
            else:
                # OLS DiD
                stats = _compute_statistics_did_ols(self, alpha=alpha)
                table = _generate_table_did_ols(stats)
                text = _generate_prose_did_ols(stats, alpha=alpha)
                return EffectSummary(table=table, text=text)
        else:
            # ITS or Synthetic Control: time-series effects
            # Handle period parameter for three-period designs (e.g., ITS with treatment_end_time)
            if period is not None:
                # Validate period parameter
                valid_periods = ["intervention", "post", "comparison"]
                if period not in valid_periods:
                    raise ValueError(
                        f"period must be one of {valid_periods}, got '{period}'"
                    )

                # Check if this experiment supports three-period designs (has treatment_end_time)
                if not (
                    hasattr(self, "treatment_end_time")
                    and self.treatment_end_time is not None
                ):
                    raise ValueError(
                        f"Period '{period}' not available. This experiment may not support three-period designs. "
                        "Provide treatment_end_time to enable period-specific analysis."
                    )

                if period == "comparison":
                    # Comparison period: delegate to subclass method
                    if hasattr(self, "_comparison_period_summary"):
                        return self._comparison_period_summary(
                            direction=direction,
                            alpha=alpha,
                            cumulative=cumulative,
                            relative=relative,
                            min_effect=min_effect,
                        )
                    else:
                        raise NotImplementedError(
                            "comparison period summary not implemented for this experiment type"
                        )

                # For "intervention" or "post" periods, use _extract_window with tuple windows
                # At this point we know treatment_end_time exists (checked above)
                # These attributes are specific to ITS experiments with three-period designs
                if period == "intervention":
                    # Intervention period: treatment_time <= index < treatment_end_time
                    # Since _extract_window uses <= end (inclusive), we need to exclude treatment_end_time
                    # Find the last index that is < treatment_end_time
                    # Note: treatment_end_time > treatment_time is already validated in __init__
                    datapost = self.datapost  # type: ignore[attr-defined]
                    treatment_end_time = self.treatment_end_time  # type: ignore[attr-defined]
                    treatment_time = self.treatment_time  # type: ignore[attr-defined]
                    intervention_indices = datapost.index[
                        datapost.index < treatment_end_time
                    ]
                    # Use the last index before treatment_end_time as the end bound
                    window = (treatment_time, intervention_indices.max())
                    prefix = "During intervention"
                elif period == "post":
                    # Post-intervention period: index >= treatment_end_time (inclusive)
                    datapost = self.datapost  # type: ignore[attr-defined]
                    treatment_end_time = self.treatment_end_time  # type: ignore[attr-defined]
                    window = (treatment_end_time, datapost.index.max())
                    prefix = "Post-intervention"

                # Extract windowed impact data using calculated window
                windowed_impact, window_coords = _extract_window(
                    self, window, treated_unit=treated_unit
                )

                # Extract counterfactual for relative effects
                counterfactual = _extract_counterfactual(
                    self, window_coords, treated_unit=treated_unit
                )
            else:
                # No period specified, use standard flow
                windowed_impact, window_coords = _extract_window(
                    self, window, treated_unit=treated_unit
                )
                counterfactual = _extract_counterfactual(
                    self, window_coords, treated_unit=treated_unit
                )

            if is_pymc:
                # PyMC model: use posterior draws
                hdi_prob = 1 - alpha
                stats = _compute_statistics(
                    windowed_impact,
                    counterfactual,
                    hdi_prob=hdi_prob,
                    direction=direction,
                    cumulative=cumulative,
                    relative=relative,
                    min_effect=min_effect,
                )

                # Generate table
                table = _generate_table(stats, cumulative=cumulative, relative=relative)

                # Generate prose
                text = _generate_prose(
                    stats,
                    window_coords,
                    alpha=alpha,
                    direction=direction,
                    cumulative=cumulative,
                    relative=relative,
                    prefix=prefix,
                )
            else:
                # OLS model: use point estimates and CIs
                # Convert to numpy arrays if needed
                if hasattr(windowed_impact, "values"):
                    impact_array = windowed_impact.values
                else:
                    impact_array = np.asarray(windowed_impact)
                if hasattr(counterfactual, "values"):
                    counterfactual_array = counterfactual.values
                else:
                    counterfactual_array = np.asarray(counterfactual)

                stats = _compute_statistics_ols(
                    impact_array,
                    counterfactual_array,
                    alpha=alpha,
                    cumulative=cumulative,
                    relative=relative,
                )

                # Generate table
                table = _generate_table_ols(
                    stats, cumulative=cumulative, relative=relative
                )

                # Generate prose
                text = _generate_prose_ols(
                    stats,
                    window_coords,
                    alpha=alpha,
                    cumulative=cumulative,
                    relative=relative,
                    prefix=prefix,
                )

            return EffectSummary(table=table, text=text)
