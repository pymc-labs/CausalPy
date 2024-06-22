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

"""
Experiment ROPE class to estimate Bayesian Power Analysis.

"""

import warnings
from typing import Any, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike
from scipy.stats import norm


# Define custom error classes
class AlphaValueError(Exception):
    """Error for when alpha value is out of the allowed range."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.message = f"Alpha value '{alpha}' is out of the allowed range [0, 1]."
        super().__init__(self.message)


class CorrectionValueError(Exception):
    """Error for when correction value is invalid."""

    def __init__(self, correction: Any) -> None:
        self.correction = correction
        self.message = (
            f"Correction value '{correction}' is invalid. "
            "It must be a bool, Pandas series (`pd.Series`), or Dictionary."
        )
        super().__init__(self.message)


class ROPE:
    """
    A class to evaluate and define the Region of Practical Equivalence (ROPE) based on a null model
    using PyMC. The ROPE method is used to determine whether a parameter estimate is
    practically equivalent to a specified value.

    Attributes
    ----------
    post_y : numpy.ndarray
        Observed data or outcome variable.
    post_pred : arviz.InferenceData
        An object containing the posterior predictive distributions.

    References
    ----------
    - The Bayesian New Statistics: Hypothesis testing, estimation, meta-analysis, and power analysis from a Bayesian perspective by John K. Kruschke · Torrin M. Liddell.
      [Link](https://link.springer.com/article/10.3758/s13423-016-1221-4#Sec19)
    """

    def __init__(self, post_y, post_pred):
        self.post_y = post_y
        self.post_pred = post_pred

    def _validate_alpha(self, alpha: float):
        """
        Validates the alpha level for credible interval calculations.

        Parameters
        ----------
        alpha : float
            Significance level for credible intervals, must be between 0 and 1.

        Raises
        ------
        AlphaValueError
            If alpha is not between 0 and 1.
        """
        if not (0 <= alpha <= 1):
            raise AlphaValueError(alpha)

    def _validate_correction(
        self, correction: Union[bool, Dict[str, float], pd.Series]
    ):
        """
        Validates the correction parameters.

        Parameters
        ----------
        correction : bool, dict, or pd.Series
            Correction parameter to adjust the posterior estimates. If a dictionary or Series, it must
            contain "cumulative" and "mean" keys or indices with numeric values.

        Raises
        ------
        CorrectionValueError
            If correction is not a valid type or does not have the required structure.
        """
        if not isinstance(correction, (bool, pd.Series, dict)):
            raise CorrectionValueError(correction)
        if isinstance(correction, pd.Series):
            if set(correction.index) != {"cumulative", "mean"}:
                raise CorrectionValueError(correction)
            if not all(
                isinstance(value, (float, int)) and not isinstance(value, bool)
                for value in correction.values
            ):
                raise CorrectionValueError(correction)
        elif isinstance(correction, dict):
            if set(correction.keys()) != {"cumulative", "mean"}:
                raise CorrectionValueError(correction)
            if not all(
                isinstance(value, (float, int)) and not isinstance(value, bool)
                for value in correction.values()
            ):
                raise CorrectionValueError(correction)

    @staticmethod
    def compute_bayesian_tail_probability(posterior: ArrayLike, x: float) -> float:
        """
        Calculate the probability of a given value being in a distribution defined by the posterior.

        Parameters
        ----------
        posterior : array-like
            Posterior distribution samples.
        x : float
            A numeric value for which to calculate the probability of being in the distribution.

        Returns
        -------
        float
            Probability of x being in the posterior distribution.
        """
        lower_bound, upper_bound = min(posterior), max(posterior)
        mean, std = np.mean(posterior), np.std(posterior)

        cdf_lower = norm.cdf(lower_bound, mean, std)
        cdf_upper = 1 - norm.cdf(upper_bound, mean, std)
        cdf_x = norm.cdf(x, mean, std)

        if cdf_x <= 0.5:
            probability = 2 * (cdf_x - cdf_lower) / (1 - cdf_lower - cdf_upper)
        else:
            probability = 2 * (1 - cdf_x + cdf_lower) / (1 - cdf_lower - cdf_upper)

        return probability  # abs(probability)

    def _calculate_posterior_mde(
        self, alpha: float, correction: Union[bool, Dict[str, float], pd.Series]
    ) -> Dict:
        """
        Calculates the posterior Minimum Detectable Effect (MDE) and  credible intervals,
        based on the defined ROPE (Region of Practical Equivalence).

        Parameters
        ----------
        alpha : float
            Significance level for credible intervals (ROPE Area).
        correction : bool, dict, or pd.Series
            Correction parameter to adjust the posterior estimates.

        Returns
        -------
        dict
            A dictionary containing the posterior estimation, results, samples, null model error,
            credible intervals, and posterior MDE.
        """
        self._validate_alpha(alpha)
        self._validate_correction(correction)

        results = {}
        credible_intervals = (alpha * 100) / 2

        # Cumulative calculations
        cumulative_results = self.post_y.sum()
        _mu_samples_cumulative = (
            self.post_pred["posterior_predictive"]["mu"]
            .stack(sample=("chain", "draw"))
            .sum("obs_ind")
        )
        # Mean calculations
        mean_results = self.post_y.mean()
        _mu_samples_mean = (
            self.post_pred["posterior_predictive"]["mu"]
            .stack(sample=("chain", "draw"))
            .mean("obs_ind")
        )

        if correction and not isinstance(correction, bool):
            _mu_samples_cumulative += correction["cumulative"]
            _mu_samples_mean += correction["mean"]

        # Posterior Mean
        results["posterior_estimation"] = {
            "cumulative": np.mean(_mu_samples_cumulative.values),
            "mean": np.mean(_mu_samples_mean.values),
        }
        results["results"] = {
            "cumulative": cumulative_results,
            "mean": mean_results,
        }
        results["samples"] = {
            "cumulative": _mu_samples_cumulative,
            "mean": _mu_samples_mean,
        }
        results["null_model_error"] = {
            "cumulative": results["results"]["cumulative"]
            - results["posterior_estimation"]["cumulative"],
            "mean": results["results"]["mean"]
            - results["posterior_estimation"]["mean"],
        }
        if correction:
            _mu_samples_cumulative += results["null_model_error"]["cumulative"]
            _mu_samples_mean += results["null_model_error"]["mean"]

        results["credible_interval"] = {
            "cumulative": [
                np.percentile(_mu_samples_cumulative, credible_intervals),
                np.percentile(_mu_samples_cumulative, 100 - credible_intervals),
            ],
            "mean": [
                np.percentile(_mu_samples_mean, credible_intervals),
                np.percentile(_mu_samples_mean, 100 - credible_intervals),
            ],
        }
        cumulative_upper_mde = (
            results["credible_interval"]["cumulative"][1]
            - results["posterior_estimation"]["cumulative"]
        )
        cumulative_lower_mde = (
            results["posterior_estimation"]["cumulative"]
            - results["credible_interval"]["cumulative"][0]
        )
        mean_upper_mde = (
            results["credible_interval"]["mean"][1]
            - results["posterior_estimation"]["mean"]
        )
        mean_lower_mde = (
            results["posterior_estimation"]["mean"]
            - results["credible_interval"]["mean"][0]
        )
        results["posterior_mde"] = {
            "cumulative": (cumulative_upper_mde + cumulative_lower_mde) / 2,
            "mean": (mean_upper_mde + mean_lower_mde) / 2,
        }

        return results

    def causal_effect_summary(self, alpha: float = 0.05, **kwargs) -> pd.DataFrame:
        """
        Provides a summary of causal effects, including the Bayesian tail probabilities and credible intervals.

        Parameters
        ----------
        alpha : float, optional
            Significance level for credible intervals (default is 0.05).
        **kwargs : dict, optional
            Additional keyword arguments, including 'correction' for posterior adjustments.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the summary of causal effects.
        """
        warnings.warn("The power function is experimental.")

        correction = kwargs.get("correction", False)
        results = self._calculate_posterior_mde(alpha=alpha, correction=correction)

        results["bayesian_tail_probability"] = {
            "cumulative": self.compute_bayesian_tail_probability(
                posterior=results["samples"]["cumulative"],
                x=results["results"]["cumulative"],
            ),
            "mean": self.compute_bayesian_tail_probability(
                posterior=results["samples"]["mean"], x=results["results"]["mean"]
            ),
        }

        results["causal_effect"] = {
            "cumulative": results["results"]["cumulative"]
            - results["posterior_estimation"]["cumulative"],
            "mean": results["results"]["mean"]
            - results["posterior_estimation"]["mean"],
        }

        return pd.DataFrame(results).drop(
            columns=["samples", "null_model_error", "posterior_mde"]
        )

    def mde_summary(
        self, alpha: float = 0.05, correction: bool = False
    ) -> pd.DataFrame:
        """
        Provides a summary of the posterior Minimum Detectable Effect (MDE) estimations.
        Based on the defined ROPE.

        Parameters
        ----------
        alpha : float, optional
            Significance level for credible intervals (default is 0.05).
        correction : bool, optional
            Whether to apply correction to the posterior estimates (default is False).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the summary of posterior MDE.
        """
        warnings.warn("The mde summary function is experimental.")
        return pd.DataFrame(
            self._calculate_posterior_mde(alpha=alpha, correction=correction)
        )

    def plot_power_distribution(
        self, alpha: float = 0.05, correction: bool = False, **kwargs
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plots the power distribution, visualizing the posterior distribution, credible intervals, and key statistics.

        Parameters
        ----------
        alpha : float, optional
            Significance level for credible intervals (default is 0.05).
        correction : bool, optional
            Whether to apply correction to the posterior estimates (default is False).
        **kwargs : dict, optional
            Additional keyword arguments for customizing the plot appearance.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            A tuple containing the Matplotlib Figure object and an array of Axes objects containing the plot.
        """
        if not isinstance(correction, bool):
            raise CorrectionValueError(correction)

        # Default parameters for customization
        default_params = {
            "title": "Mu Posterior",
            "figsize": (20, 6),
            "hist_color": "C0",
            "kde_color": "C1",
            "ci_color": "C0",
            "mean_color": "C3",
            "posterior_mean_color": "C4",
            "alpha": 0.6,
            "ci_alpha": 0.3,
            "hist_bins": 30,
            "title_fontsize": 14,
            "xlabel_fontsize": 12,
            "xlable_title": "Mu",
            "ylabel_fontsize": 12,
            "legend_fontsize": 10,
        }

        # Update defaults with any user-provided keyword arguments
        plot_params = {**default_params, **kwargs}

        _estimates = self._calculate_posterior_mde(alpha=alpha, correction=correction)

        fig, axs = plt.subplots(
            1, 2, figsize=plot_params["figsize"]
        )  # Two subplots side by side

        for i, key in enumerate(["mean", "cumulative"]):
            _mu_samples = self.post_pred["posterior_predictive"]["mu"].stack(
                sample=("chain", "draw")
            )
            if key == "mean":
                _mu_samples = _mu_samples.mean("obs_ind")
            elif key == "cumulative":
                _mu_samples = _mu_samples.sum("obs_ind")

            if correction:
                _mu_samples += _estimates["null_model_error"][key]

            sns.histplot(
                _mu_samples,
                bins=plot_params["hist_bins"],
                kde=True,
                ax=axs[i],
                color=plot_params["hist_color"],
                stat="density",
                alpha=plot_params["alpha"],
            )
            kde_x, kde_y = (
                sns.kdeplot(
                    _mu_samples, color=plot_params["kde_color"], fill=True, ax=axs[i]
                )
                .get_lines()[0]
                .get_data()
            )

            max_density = max(kde_y)
            axs[i].set_ylim(0, max_density + 0.05 * max_density)

            axs[i].fill_betweenx(
                y=np.linspace(0, max_density + 0.05 * max_density, 100),
                x1=_estimates["credible_interval"][key][0],
                x2=_estimates["credible_interval"][key][1],
                color=plot_params["ci_color"],
                alpha=plot_params["ci_alpha"],
                label="C.I",
            )

            axs[i].axvline(
                _estimates["results"][key],
                color=plot_params["mean_color"],
                linestyle="-",
                label="Real Mean",
            )
            if not correction:
                axs[i].axvline(
                    _estimates["posterior_estimation"][key],
                    color=plot_params["posterior_mean_color"],
                    linestyle="--",
                    label="Posterior Mean",
                )

            axs[i].set_title(
                plot_params["title"] + f" {key}", fontsize=plot_params["title_fontsize"]
            )
            axs[i].set_xlabel(
                plot_params["xlable_title"], fontsize=plot_params["xlabel_fontsize"]
            )
            axs[i].set_ylabel("Density", fontsize=plot_params["ylabel_fontsize"])
            axs[i].legend(fontsize=plot_params["legend_fontsize"])

        return fig, axs
