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
Power analysis utilities for quasi-experimental designs.

Given a completed ``PlaceboInTime`` check, estimates the probability of
detecting a true effect as a function of effect size (i.e. a power curve).
Two strategies are supported: brute-force grid evaluation and a faster
sigmoid-fit approach that extracts the MDE analytically.

References
----------
.. [1] Gelman, A. & Carlin, J. (2014). Beyond Power Calculations:
   Assessing Type S (Sign) and Type M (Magnitude) Errors.
   *Perspectives on Psychological Science*, 9(6), 641-651.
.. [2] Kruschke, J. K. (2013). Bayesian estimation supersedes the t test.
   *Journal of Experimental Psychology: General*, 142(2), 573-603.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from causalpy.checks.base import CheckResult


@dataclass
class LogisticFit:
    """Parameters of the fitted two-parameter logistic curve.

    The detection probability is modelled as:

    .. math::
        P(\\text{detect} \\mid x) = \\frac{1}{1 + \\exp(-k(x - x_0))}

    Attributes
    ----------
    k : float
        Steepness (slope) parameter.
    x0 : float
        Midpoint — the effect size at which detection probability is 50%.
    """

    k: float
    x0: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the fitted logistic at effect sizes ``x``.

        Parameters
        ----------
        x : np.ndarray
            Effect sizes at which to evaluate.

        Returns
        -------
        np.ndarray
            Predicted detection probabilities.
        """
        return _logistic(np.asarray(x, dtype=float), self.k, self.x0)

    def mde(self, power_threshold: float = 0.80) -> float:
        """Extract the Minimum Detectable Effect from the fitted logistic.

        Parameters
        ----------
        power_threshold : float, default 0.80
            The power level at which to compute the MDE.

        Returns
        -------
        float
            The smallest absolute effect size achieving the given power.
        """
        if not 0 < power_threshold < 1:
            raise ValueError(
                f"power_threshold must be in (0, 1), got {power_threshold}"
            )
        if self.k == 0:
            return np.inf
        # Invert the logistic: x = x0 + (1/k) * ln(tau / (1 - tau))
        tau = power_threshold
        return self.x0 + (1.0 / self.k) * np.log(tau / (1.0 - tau))


@dataclass
class PowerCurveResult:
    """Result of a power analysis computation.

    Attributes
    ----------
    effect_sizes : np.ndarray
        The effect sizes at which detection probability was evaluated.
    detection_rates : np.ndarray
        Monte Carlo detection probability at each evaluated effect size.
    strategy : str
        The strategy used: ``"grid"`` or ``"sigmoid"``.
    n_simulations : int
        Number of Monte Carlo replications per evaluation point.
    rope_half_width : float
        ROPE half-width used for the decision rule.
    threshold : float
        Posterior probability cutoff for an actionable decision.
    fitted_curve : LogisticFit or None
        The fitted logistic parameters (only for ``strategy="sigmoid"``).
    smooth_effect_sizes : np.ndarray or None
        Fine-grained effect sizes for the reconstructed smooth curve
        (only for ``strategy="sigmoid"``).
    smooth_detection_rates : np.ndarray or None
        Detection probabilities on the smooth grid, from the fitted
        logistic (only for ``strategy="sigmoid"``).
    mde : float or None
        Minimum Detectable Effect at 80% power (only for
        ``strategy="sigmoid"``).
    """

    effect_sizes: np.ndarray
    detection_rates: np.ndarray
    strategy: str
    n_simulations: int
    rope_half_width: float
    threshold: float
    fitted_curve: LogisticFit | None = None
    smooth_effect_sizes: np.ndarray | None = field(default=None, repr=False)
    smooth_detection_rates: np.ndarray | None = field(default=None, repr=False)
    mde: float | None = None

    def plot(  # pragma: no cover
        self,
        power_threshold: float = 0.80,
        ax: plt.Axes | None = None,
        title: str = "Power Curve",
        xlabel: str = "Effect size",
        ylabel: str = "Detection probability",
        show_mde: bool = True,
    ) -> plt.Figure:
        """Plot the power curve.

        Parameters
        ----------
        power_threshold : float, default 0.80
            Power level at which to draw a horizontal reference line.
        ax : plt.Axes or None
            Axes to plot on.  If ``None``, creates a new figure.
        title : str
            Plot title.
        xlabel : str
            X-axis label.
        ylabel : str
            Y-axis label.
        show_mde : bool, default True
            Whether to annotate the MDE on the plot (sigmoid only).

        Returns
        -------
        plt.Figure
            The figure containing the power curve.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        # Plot raw evaluation points
        ax.scatter(
            self.effect_sizes,
            self.detection_rates,
            color="#348ABD",
            s=60,
            zorder=5,
            label="Evaluated points",
            edgecolors="white",
            linewidths=0.8,
        )

        # Plot smooth fitted curve if available
        if (
            self.smooth_effect_sizes is not None
            and self.smooth_detection_rates is not None
        ):
            ax.plot(
                self.smooth_effect_sizes,
                self.smooth_detection_rates,
                color="#348ABD",
                lw=2,
                label="Fitted logistic",
            )

        # Connect points for grid strategy
        if self.strategy == "grid":
            sorted_idx = np.argsort(self.effect_sizes)
            ax.plot(
                self.effect_sizes[sorted_idx],
                self.detection_rates[sorted_idx],
                color="#348ABD",
                lw=1.5,
                alpha=0.6,
            )

        # Power threshold line
        ax.axhline(
            power_threshold,
            color="#E24A33",
            ls="--",
            lw=1.2,
            alpha=0.7,
            label=f"Power = {power_threshold:.0%}",
        )

        # MDE annotation
        if show_mde and self.mde is not None:
            ax.axvline(
                self.mde,
                color="#22c55e",
                ls="--",
                lw=1.2,
                alpha=0.8,
            )
            ax.annotate(
                f"MDE = {self.mde:.3f}",
                xy=(self.mde, power_threshold),
                xytext=(self.mde + 0.02 * ax.get_xlim()[1], power_threshold + 0.05),
                fontsize=9,
                color="#22c55e",
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#22c55e", "lw": 1.2},
            )

        # ROPE region
        ax.axvspan(
            0,
            self.rope_half_width,
            color="#9ca3af",
            alpha=0.15,
            label="Below ROPE",
        )

        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="lower right", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig


def _logistic(x: np.ndarray, k: float, x0: float) -> np.ndarray:
    """Standard two-parameter logistic: 1 / (1 + exp(-k*(x - x0)))."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _simulate_detection_rate(
    effect_size: float,
    null_samples: np.ndarray,
    fold_sds: np.ndarray,
    rope_half_width: float,
    threshold: float,
    n_simulations: int,
    n_posterior_samples: int,
    rng: np.random.Generator,
) -> float:
    """Run Monte Carlo simulation at a single effect size.

    Draws null noise, adds the effect, simulates a posterior, and
    checks whether the ROPE rule fires.  Returns the fraction of
    replications that result in a positive decision.
    """
    detections = 0
    for _ in range(n_simulations):
        # Draw null component (structural noise)
        null_component = float(rng.choice(null_samples))
        # True effect = null noise + injected effect
        true_effect = null_component + effect_size
        # Simulate estimation uncertainty
        sigma = float(rng.choice(fold_sds))
        simulated_posterior = rng.normal(
            loc=true_effect, scale=sigma, size=n_posterior_samples
        )
        # Apply ROPE decision
        prob_positive = float((simulated_posterior > rope_half_width).mean())
        if prob_positive >= threshold:
            detections += 1

    return detections / n_simulations


def _extract_null_distribution(
    pit_result: CheckResult,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Pull null_samples, fold_sds, rope_half_width, threshold from metadata.

    Raises ValueError if any required key is missing.
    """
    meta = pit_result.metadata
    if "null_samples" not in meta:
        raise ValueError(
            "pit_result does not contain a learned null distribution. "
            "Ensure PlaceboInTime completed successfully with at least "
            "one fold."
        )

    null_samples = np.asarray(meta["null_samples"]).ravel()
    fold_sds_raw = meta.get("fold_sds")
    if fold_sds_raw is None:
        fold_results = meta.get("fold_results", [])
        if not fold_results:
            raise ValueError("pit_result has no fold_sds or fold_results in metadata.")
        fold_sds_raw = [fr.fold_sd for fr in fold_results]
    fold_sds = np.asarray(fold_sds_raw)

    rope_half_width = meta.get("rope_half_width")
    if rope_half_width is None:
        raise ValueError(
            "pit_result has no rope_half_width in metadata. "
            "PlaceboInTime must be configured with a rope_half_width."
        )
    rope_half_width = float(rope_half_width)

    threshold = float(meta.get("threshold", 0.95))
    return null_samples, fold_sds, rope_half_width, threshold


def _build_effect_sizes(
    strategy: str,
    effect_sizes: list[float] | np.ndarray | None,
    tau: float,
    rope_half_width: float,
    n_evaluation_points: int,
) -> np.ndarray:
    """Construct the evaluation grid from user inputs or sensible defaults."""
    if strategy == "grid":
        if effect_sizes is None:
            return np.linspace(0, max(4.0 * tau, 3.0 * rope_half_width), 8)
        return np.asarray(effect_sizes, dtype=float)
    elif strategy == "sigmoid":
        if effect_sizes is None:
            range_min, range_max = 0.0, max(4.0 * tau, 3.0 * rope_half_width)
        elif len(effect_sizes) == 2:
            range_min, range_max = float(effect_sizes[0]), float(effect_sizes[1])
        else:
            range_min = float(np.min(effect_sizes))
            range_max = float(np.max(effect_sizes))
        return np.linspace(range_min, range_max, n_evaluation_points)
    else:
        raise ValueError(f"strategy must be 'grid' or 'sigmoid', got {strategy!r}")


def _fit_sigmoid(
    effect_sizes_arr: np.ndarray,
    detection_rates: np.ndarray,
    tau: float,
    power_threshold: float,
) -> tuple[LogisticFit | None, np.ndarray | None, np.ndarray | None, float | None]:
    """Fit a logistic to the detection rates and extract MDE.

    Returns (fitted_curve, smooth_x, smooth_y, mde).  All four are None
    if curve_fit fails.
    """
    try:
        x0_guess = float(effect_sizes_arr[len(effect_sizes_arr) // 2])
        k_guess = 1.0 / max(tau, 1e-8)

        popt, _ = curve_fit(
            _logistic,
            effect_sizes_arr,
            detection_rates,
            p0=[k_guess, x0_guess],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=5000,
        )
        fitted_curve = LogisticFit(k=popt[0], x0=popt[1])

        smooth_effect_sizes = np.linspace(
            float(effect_sizes_arr[0]),
            float(effect_sizes_arr[-1]),
            200,
        )
        smooth_detection_rates = fitted_curve.predict(smooth_effect_sizes)
        mde = fitted_curve.mde(power_threshold)

        if mde < effect_sizes_arr[0] or mde > effect_sizes_arr[-1]:
            warnings.warn(
                f"Fitted MDE ({mde:.4f}) is outside the evaluated range "
                f"[{effect_sizes_arr[0]:.4f}, {effect_sizes_arr[-1]:.4f}]. "
                f"Consider widening the effect_sizes range.",
                stacklevel=3,
            )
        return fitted_curve, smooth_effect_sizes, smooth_detection_rates, mde
    except (RuntimeError, ValueError) as e:
        warnings.warn(
            f"Sigmoid fitting failed: {e}. "
            f"Returning raw evaluation points without fitted curve.",
            stacklevel=3,
        )
        return None, None, None, None


def power_analysis(
    pit_result: CheckResult,
    effect_sizes: list[float] | np.ndarray | None = None,
    n_simulations: int = 200,
    strategy: Literal["grid", "sigmoid"] = "grid",
    n_evaluation_points: int = 5,
    power_threshold: float = 0.80,
    n_posterior_samples: int = 1000,
    random_seed: int | None = None,
) -> PowerCurveResult:
    """Compute a power curve from a completed PlaceboInTime check.

    Uses the learned null distribution to simulate what the decision
    rule would conclude at each hypothetical effect size.  Two modes:

    * ``"grid"`` — brute-force evaluation at every requested point.
    * ``"sigmoid"`` — evaluate at a few points, fit a logistic, and
      read off the MDE.  Faster when you only need the MDE.

    How it works
    ------------
    At each effect size the function runs ``n_simulations`` Monte Carlo
    replications.  Each replication draws a null component from the
    status-quo posterior, adds the hypothetical effect, simulates a
    posterior around that total (using the observed fold SDs), and
    applies the ROPE rule.  The fraction of replications that trigger
    a positive decision is the estimated power.

    For the sigmoid strategy a two-parameter logistic is fitted:

    .. math::
        P(\\text{detect} \\mid x) = \\frac{1}{1 + \\exp(-k(x - x_0))}

    and the MDE is obtained by inverting at the desired power level.

    Parameters
    ----------
    pit_result : CheckResult
        A completed ``PlaceboInTime`` check result containing the learned
        null distribution in its metadata (keys: ``"null_samples"``,
        ``"fold_sds"``, ``"rope_half_width"``, ``"threshold"``).
    effect_sizes : list or ndarray or None
        For ``strategy="grid"``: the exact effect sizes to evaluate.
        For ``strategy="sigmoid"``: a two-element ``[min, max]`` range
        within which evaluation points are placed.  If ``None``,
        defaults to ``np.linspace(0, 4 * tau, 8)`` for grid or
        ``[0, 4 * tau]`` for sigmoid, where ``tau`` is the null SD.
    n_simulations : int, default 200
        Number of Monte Carlo replications per evaluation point.
    strategy : {"grid", "sigmoid"}, default "grid"
        Estimation strategy.
    n_evaluation_points : int, default 5
        Number of evaluation points for the sigmoid strategy.
    power_threshold : float, default 0.80
        Power level for MDE extraction (sigmoid strategy).
    n_posterior_samples : int, default 1000
        Number of posterior draws per simulated experiment.
    random_seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    PowerCurveResult
        Contains evaluated points, optional fitted curve, and MDE.

    Raises
    ------
    ValueError
        If ``pit_result`` does not contain the required metadata.

    Examples
    --------
    >>> import causalpy as cp
    >>> # After running a PlaceboInTime check:
    >>> # result = pipeline.run()
    >>> # pit_check = result.sensitivity_results[0]
    >>> # curve = cp.checks.power_analysis(pit_check, strategy="sigmoid")
    >>> # curve.mde  # Minimum Detectable Effect at 80% power
    >>> # curve.plot()
    """
    null_samples, fold_sds, rope_half_width, threshold = _extract_null_distribution(
        pit_result
    )
    tau = float(null_samples.std())

    effect_sizes_arr = _build_effect_sizes(
        strategy, effect_sizes, tau, rope_half_width, n_evaluation_points
    )

    # Run Monte Carlo simulation at each evaluation point
    rng = np.random.default_rng(random_seed)
    detection_rates = np.zeros(len(effect_sizes_arr))
    for i, eff in enumerate(effect_sizes_arr):
        detection_rates[i] = _simulate_detection_rate(
            effect_size=eff,
            null_samples=null_samples,
            fold_sds=fold_sds,
            rope_half_width=rope_half_width,
            threshold=threshold,
            n_simulations=n_simulations,
            n_posterior_samples=n_posterior_samples,
            rng=rng,
        )

    # Fit sigmoid if requested
    fitted_curve = None
    smooth_effect_sizes = None
    smooth_detection_rates = None
    mde = None
    if strategy == "sigmoid":
        fitted_curve, smooth_effect_sizes, smooth_detection_rates, mde = _fit_sigmoid(
            effect_sizes_arr, detection_rates, tau, power_threshold
        )

    return PowerCurveResult(
        effect_sizes=effect_sizes_arr,
        detection_rates=detection_rates,
        strategy=strategy,
        n_simulations=n_simulations,
        rope_half_width=rope_half_width,
        threshold=threshold,
        fitted_curve=fitted_curve,
        smooth_effect_sizes=smooth_effect_sizes,
        smooth_detection_rates=smooth_detection_rates,
        mde=mde,
    )
