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
            # Extract windowed impact data
            windowed_impact, window_coords = _extract_window(
                self, window, treated_unit=treated_unit
            )

            # Extract counterfactual for relative effects
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
                )

            return EffectSummary(table=table, text=text)

    # =========================================================================
    # maketables plugin support (zero-coupling approach)
    # These dunder attributes enable automatic integration with maketables
    # for generating publication-ready coefficient tables.
    # See: https://py-econometrics.github.io/maketables/docs/AddingMethods.html
    # =========================================================================

    @property
    def __maketables_coef_table__(self) -> pd.DataFrame:
        """Return coefficient table for maketables integration.

        Returns a DataFrame with columns:
        - b: coefficient estimate (posterior mean for Bayesian, point estimate for OLS)
        - se: standard error (posterior std for Bayesian, None for OLS)
        - t: t-statistic (None for Bayesian, None for OLS currently)
        - p: p-value (two-tailed posterior probability for Bayesian, None for OLS)
        - ci_lower: lower bound of 94% HDI (Bayesian) or None (OLS)
        - ci_upper: upper bound of 94% HDI (Bayesian) or None (OLS)
        """
        if isinstance(self.model, PyMCModel):
            return self._maketables_coef_table_bayesian()
        else:
            return self._maketables_coef_table_ols()

    def _maketables_coef_table_bayesian(self) -> pd.DataFrame:
        """Build coefficient table from PyMC posterior."""
        # Get flattened samples for statistics calculations
        beta_samples = az.extract(self.model.idata.posterior, var_names="beta")

        # Handle multi-unit case: select first treated unit if multiple exist
        if "treated_units" in beta_samples.dims:
            beta_samples = beta_samples.isel(treated_units=0)

        # Compute statistics for each coefficient
        rows = []
        for i, label in enumerate(self.labels):
            # Use flattened samples for mean, std, probabilities
            coef_samples = beta_samples.isel(coeffs=i)
            mean_val = float(coef_samples.mean())
            std_val = float(coef_samples.std())
            p_pos = float((coef_samples > 0).mean())
            p_neg = float((coef_samples < 0).mean())
            p_two_sided = min(p_pos, p_neg) * 2

            # Compute HDI from samples using numpy for simplicity
            sorted_samples = np.sort(coef_samples.values)
            n = len(sorted_samples)
            interval_size = int(np.ceil(0.94 * n))
            n_intervals = n - interval_size
            widths = sorted_samples[interval_size:] - sorted_samples[:n_intervals]
            min_idx = np.argmin(widths)
            ci_lower = float(sorted_samples[min_idx])
            ci_upper = float(sorted_samples[min_idx + interval_size])

            rows.append(
                {
                    "b": mean_val,
                    "se": std_val,
                    "t": None,  # Not applicable for Bayesian
                    "p": p_two_sided,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )

        df = pd.DataFrame(rows, index=self.labels)
        df.index.name = "Coefficient"
        return df

    def _maketables_coef_table_ols(self) -> pd.DataFrame:
        """Build coefficient table from OLS model."""
        coeffs = self.model.get_coeffs()
        # Note: OLS SE/t/p require access to residuals which we don't store
        rows = []
        for label, coef in zip(self.labels, coeffs, strict=False):
            rows.append(
                {
                    "b": coef,
                    "se": None,  # OLS SE requires access to residuals
                    "t": None,
                    "p": None,
                    "ci_lower": None,
                    "ci_upper": None,
                }
            )
        df = pd.DataFrame(rows, index=self.labels)
        df.index.name = "Coefficient"
        return df

    def __maketables_stat__(self, key: str):
        """Return statistics for maketables integration.

        Supported keys:
        - N: number of observations
        - model_type: "PyMC" or "OLS"
        - experiment_type: type of causal experiment
        - r2: R-squared (if available)
        """
        stat_map = {
            "N": len(self.data) if hasattr(self, "data") else None,
            "model_type": "PyMC" if isinstance(self.model, PyMCModel) else "OLS",
            "experiment_type": getattr(self, "expt_type", self.__class__.__name__),
        }
        # Add R² if available
        if hasattr(self, "score") and self.score is not None:
            stat_map["r2"] = self.score
        return stat_map.get(key)

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable name for maketables integration."""
        if hasattr(self, "outcome_variable_name"):
            return self.outcome_variable_name
        return "y"

    @property
    def __maketables_vcov_info__(self) -> dict:
        """Return variance-covariance info for maketables integration."""
        return {
            "vcov_type": "Bayesian" if isinstance(self.model, PyMCModel) else "OLS",
            "clustervar": None,
        }
