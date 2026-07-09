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
Power analysis for geo-experiment and quasi-experiment designs.

Provides efficient power curve estimation via two strategies:

* **grid** — evaluate detection probability at every effect size in a
  user-supplied list (brute-force Monte Carlo).
* **sigmoid** — evaluate at a small number of points (default 5), fit
  a two-parameter logistic curve, and extract the MDE analytically.
  Achieves ~40–60% computation reduction while producing smoother curves.

The power analysis operates on the learned null distribution from a
completed ``PlaceboInTime`` check.  It simulates what the design would
conclude at each hypothetical true effect size using the same ROPE-based
Bayesian decision rule.

References
----------
Issue: https://github.com/pymc-labs/CausalPy/issues/820
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.optimize import curve_fit

from causalpy.checks.base import CheckResult


@dataclass
class LogisticFit:
    """Parameters of the fitted two-parameter logistic curve.

    The model is::

        P(detect | x) = 1 / (1 + exp(-k * (x - x0)))

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

    def plot(
        self,
        power_threshold: float = 0.80,
        ax: Any = None,
        title: str = "Power Curve",
        xlabel: str = "Effect size",
        ylabel: str = "Detection probability",
        show_mde: bool = True,
    ) -> Any:
        """Plot the power curve.

        Parameters
        ----------
        power_threshold : float, default 0.80
            Power level at which to draw a horizontal reference line.
        ax : matplotlib.axes.Axes or None
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
        matplotlib.figure.Figure
            The figure containing the power curve.
        """
        import matplotlib.pyplot as plt

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
    """Two-parameter logistic function.

    Parameters
    ----------
    x : np.ndarray
        Input values (effect sizes).
    k : float
        Steepness parameter.
    x0 : float
        Midpoint parameter.

    Returns
    -------
    np.ndarray
        Logistic output in [0, 1].
    """
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
    """Simulate detection probability at a single effect size.

    For each replication:
    1. Draw a null component from the learned status-quo distribution.
    2. Add the hypothetical true effect to get the "true" total effect.
    3. Simulate a posterior by drawing from Normal(true_effect, sigma).
    4. Apply the ROPE decision rule.
    5. Count "positive" decisions as detections.

    Parameters
    ----------
    effect_size : float
        The hypothetical true absolute cumulative effect.
    null_samples : np.ndarray
        Posterior predictive draws from the status-quo model.
    fold_sds : np.ndarray
        Per-fold posterior standard deviations.
    rope_half_width : float
        ROPE half-width for the decision rule.
    threshold : float
        Posterior probability cutoff for a "positive" decision.
    n_simulations : int
        Number of Monte Carlo replications.
    n_posterior_samples : int
        Number of posterior draws per simulated experiment.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    float
        Estimated detection probability (fraction of "positive" decisions).
    """
    detections = 0
    for i in range(n_simulations):
        # Draw null component (structural noise)
        null_component = float(null_samples[i % len(null_samples)])
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

    Estimates detection probability as a function of true effect size
    using the learned null distribution from a fitted ``PlaceboInTime``
    check.  Supports two strategies:

    * ``"grid"`` — evaluate at every effect size in ``effect_sizes``
      (brute-force).
    * ``"sigmoid"`` — evaluate at ``n_evaluation_points`` within the
      range of ``effect_sizes``, fit a two-parameter logistic, and
      extract the MDE analytically.  Reduces computation by ~40–60%.

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
    # Validate input
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

    # Compute null scale for default ranges
    tau = float(null_samples.std())

    # Set up effect sizes based on strategy
    if strategy == "grid":
        if effect_sizes is None:
            effect_sizes_arr = np.linspace(0, max(4.0 * tau, 3.0 * rope_half_width), 8)
        else:
            effect_sizes_arr = np.asarray(effect_sizes, dtype=float)
    elif strategy == "sigmoid":
        if effect_sizes is None:
            range_min, range_max = 0.0, max(4.0 * tau, 3.0 * rope_half_width)
        elif len(effect_sizes) == 2:
            range_min, range_max = float(effect_sizes[0]), float(effect_sizes[1])
        else:
            # Use min/max of provided list as range
            range_min = float(np.min(effect_sizes))
            range_max = float(np.max(effect_sizes))
        effect_sizes_arr = np.linspace(range_min, range_max, n_evaluation_points)
    else:
        raise ValueError(f"strategy must be 'grid' or 'sigmoid', got {strategy!r}")

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
        # Fit the two-parameter logistic
        try:
            # Initial guesses: x0 near the midpoint, k ~ 1/tau
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

            # Generate smooth curve
            smooth_effect_sizes = np.linspace(
                float(effect_sizes_arr[0]),
                float(effect_sizes_arr[-1]),
                200,
            )
            smooth_detection_rates = fitted_curve.predict(smooth_effect_sizes)

            # Extract MDE
            mde = fitted_curve.mde(power_threshold)

            # Warn if MDE is outside the evaluated range
            if mde < effect_sizes_arr[0] or mde > effect_sizes_arr[-1]:
                warnings.warn(
                    f"Fitted MDE ({mde:.4f}) is outside the evaluated range "
                    f"[{effect_sizes_arr[0]:.4f}, {effect_sizes_arr[-1]:.4f}]. "
                    f"Consider widening the effect_sizes range.",
                    stacklevel=2,
                )
        except RuntimeError as e:
            warnings.warn(
                f"Sigmoid fitting failed: {e}. "
                f"Returning raw evaluation points without fitted curve.",
                stacklevel=2,
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
