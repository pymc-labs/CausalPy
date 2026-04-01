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
Placebo-in-time sensitivity check with hierarchical null model.

Builds a hierarchical Bayesian model of the "status quo" (no-effect)
distribution from placebo folds, then compares the actual intervention
effect against that learned null.  Optionally computes Bayesian
assurance (operating characteristics) against a user-supplied
expected-effect prior.

Supports experiments with a ``treatment_time`` parameter
(InterruptedTimeSeries, SyntheticControl).  Requires a PyMC model
for posterior extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from causalpy.checks.base import CheckResult, clone_model
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pipeline import PipelineContext
from causalpy.pymc_models import PyMCModel

logger = logging.getLogger(__name__)

MIN_FOLD_OBSERVATIONS = 3

_DEFAULT_SAMPLE_KWARGS: dict[str, Any] = {
    "draws": 1000,
    "chains": 4,
    "target_accept": 0.97,
}


@dataclass
class PlaceboFoldResult:
    """Result of a single placebo fold.

    Attributes
    ----------
    fold : int
        Fold number (1-indexed).
    pseudo_treatment_time : Any
        The shifted treatment time for this fold.
    experiment : BaseExperiment
        The fitted experiment for this fold.
    cumulative_impact_samples : xr.DataArray
        Posterior samples of the cumulative (summed) impact for this fold.
    fold_mean : float
        Posterior mean of the cumulative impact.
    fold_sd : float
        Posterior standard deviation of the cumulative impact.
    """

    fold: int
    pseudo_treatment_time: Any
    experiment: BaseExperiment
    cumulative_impact_samples: xr.DataArray
    fold_mean: float
    fold_sd: float


@dataclass
class AssuranceResult:
    """Bayesian operating characteristics from design-level simulation.

    Attributes
    ----------
    true_positive_rate : float
        P(decide "positive" | alternative true).  This *is* the assurance.
    false_positive_rate : float
        P(decide "positive" | null true).
    true_negative_rate : float
        P(decide "null" | null true).
    false_negative_rate : float
        P(decide "null" | alternative true).
    null_indeterminate_rate : float
        P(decide "indeterminate" | null true).
    alt_indeterminate_rate : float
        P(decide "indeterminate" | alternative true).
    null_decisions : np.ndarray
        Raw decision strings under the null scenario.
    alt_decisions : np.ndarray
        Raw decision strings under the alternative scenario.
    """

    true_positive_rate: float
    false_positive_rate: float
    true_negative_rate: float
    false_negative_rate: float
    null_indeterminate_rate: float
    alt_indeterminate_rate: float
    null_decisions: np.ndarray = field(repr=False)
    alt_decisions: np.ndarray = field(repr=False)


class PlaceboInTime:
    """Placebo-in-time sensitivity check with hierarchical null model.

    Shifts the treatment time backward into the pre-intervention period
    to create ``n_folds`` placebo experiments.  Extracts the posterior
    cumulative impact from each fold, then fits a hierarchical Bayesian
    model to characterise the "status quo" distribution of effects when
    no intervention occurred.  The actual intervention's cumulative
    effect is compared against this learned null.

    When ``expected_effect_prior`` and ``rope_half_width`` are provided,
    additionally computes Bayesian assurance (operating characteristics)
    via simulation.

    Parameters
    ----------
    n_folds : int, default 3
        Number of placebo folds to create.  Must be >= 1.
    experiment_factory : callable, optional
        Custom factory ``(data, treatment_time) -> BaseExperiment``.
        If ``None`` (default), the factory is derived from the pipeline's
        ``experiment_config``.  Required for standalone (non-pipeline) use.
    sample_kwargs : dict, optional
        MCMC settings for the hierarchical status-quo model.
        Defaults to ``{"draws": 1000, "chains": 4, "target_accept": 0.97}``.
    threshold : float, default 0.95
        Probability cutoff.  Used both for ``passed`` (P(actual effect
        outside null) must exceed this) and for the ROPE decision rule
        when computing assurance.
    prior_scale : float, default 1.0
        Multiplier for auto-computed prior widths on the hierarchical
        model.  The priors are
        ``mu ~ Normal(center, 5 * prior_scale * data_scale)`` and
        ``tau ~ HalfNormal(2 * prior_scale * data_scale)``.
    expected_effect_prior : distribution or array, optional
        Prior belief about the true total effect under the alternative
        hypothesis.  Accepts any object with an ``.rvs(n)`` method
        (PreliZ, scipy) or a numpy array of pre-drawn samples.  When
        provided together with ``rope_half_width``, assurance analysis
        runs automatically.
    rope_half_width : float, optional
        Half-width of the ROPE interval ``[-rope, +rope]``.  Required
        when ``expected_effect_prior`` is provided.
    n_design_replications : int, optional
        Number of simulation replications for assurance.  Defaults to
        ``min(theta_new.size, expected_effect_samples.size)``.
    random_seed : int, optional
        RNG seed for the assurance simulation.

    Examples
    --------
    >>> import causalpy as cp  # doctest: +SKIP
    >>> check = cp.checks.PlaceboInTime(n_folds=3)  # doctest: +SKIP
    """

    applicable_methods: set[type[BaseExperiment]] = {
        InterruptedTimeSeries,
        SyntheticControl,
    }

    def __init__(
        self,
        n_folds: int = 3,
        experiment_factory: Any | None = None,
        sample_kwargs: dict[str, Any] | None = None,
        threshold: float = 0.95,
        prior_scale: float = 1.0,
        expected_effect_prior: Any | None = None,
        rope_half_width: float | None = None,
        n_design_replications: int | None = None,
        random_seed: int | None = None,
    ) -> None:
        if n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        if expected_effect_prior is not None and rope_half_width is None:
            raise ValueError(
                "rope_half_width is required when expected_effect_prior is "
                "provided.  Specify the ROPE half-width that defines "
                "practical significance."
            )
        self.n_folds = n_folds
        self.experiment_factory = experiment_factory
        self.sample_kwargs = {**_DEFAULT_SAMPLE_KWARGS, **(sample_kwargs or {})}
        self.threshold = threshold
        self.prior_scale = prior_scale
        self.expected_effect_prior = expected_effect_prior
        self.rope_half_width = rope_half_width
        self.n_design_replications = n_design_replications
        self.random_seed = random_seed

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, experiment: BaseExperiment) -> None:
        """Check the experiment is compatible with PlaceboInTime.

        Raises
        ------
        TypeError
            If the experiment lacks ``treatment_time`` or uses a non-PyMC
            model.
        """
        if not hasattr(experiment, "treatment_time"):
            raise TypeError(
                f"{type(experiment).__name__} does not have a treatment_time "
                f"attribute. PlaceboInTime requires experiments with an "
                f"explicit treatment time."
            )
        if not isinstance(experiment.model, PyMCModel):
            raise TypeError(
                f"PlaceboInTime requires a PyMC model for posterior "
                f"extraction, but got {type(experiment.model).__name__}. "
                f"Use a PyMC model (e.g. cp.pymc_models.LinearRegression)."
            )

    # ------------------------------------------------------------------
    # Factory helpers (reused from original)
    # ------------------------------------------------------------------

    def _get_factory(self, context: PipelineContext | None) -> Any:
        """Return a factory ``(data, treatment_time) -> experiment``."""
        if self.experiment_factory is not None:
            return self.experiment_factory

        if context is None or context.experiment_config is None:
            raise RuntimeError(
                "No experiment_config in context and no experiment_factory "
                "provided.  Use EstimateEffect before SensitivityAnalysis, "
                "or pass an explicit experiment_factory to PlaceboInTime."
            )

        config = context.experiment_config
        method = config["method"]
        kwargs = {k: v for k, v in config.items() if k != "method"}

        def _factory(data: pd.DataFrame, treatment_time: Any) -> BaseExperiment:
            """Create a fresh experiment with the given treatment time."""
            kw = dict(kwargs)
            kw["treatment_time"] = treatment_time
            if "model" in kw and kw["model"] is not None:
                kw["model"] = clone_model(kw["model"])
            return method(data, **kw)

        return _factory

    # ------------------------------------------------------------------
    # Fold geometry (reused from original)
    # ------------------------------------------------------------------

    def _compute_intervention_length(self, experiment: BaseExperiment) -> Any:
        """Compute intervention length from the experiment."""
        treatment_time = experiment.treatment_time  # type: ignore[attr-defined]
        data = experiment.data  # type: ignore[attr-defined]

        treatment_end = getattr(experiment, "treatment_end_time", None)
        if treatment_end is not None:
            return treatment_end - treatment_time

        if hasattr(data, "index"):
            return data.index.max() - treatment_time

        raise ValueError("Cannot determine intervention length from experiment.")

    def _compute_fold_treatment_times(
        self, treatment_time: Any, intervention_length: Any
    ) -> list[Any]:
        """Compute pseudo-treatment times for each fold."""
        return [
            treatment_time - (self.n_folds - fold) * intervention_length
            for fold in range(self.n_folds)
        ]

    def _get_fold_data(
        self,
        data: pd.DataFrame,
        pseudo_treatment_time: Any,
        intervention_length: Any,
    ) -> pd.DataFrame:
        """Extract data up to the end of the placebo intervention window."""
        pseudo_end = pseudo_treatment_time + intervention_length
        return data.loc[data.index < pseudo_end].copy()

    # ------------------------------------------------------------------
    # Posterior extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_cumulative_impact(experiment: BaseExperiment) -> xr.DataArray:
        """Extract posterior cumulative impact from a fitted experiment.

        Returns an ``xr.DataArray`` with a single ``sample`` dimension
        obtained by summing over ``obs_ind`` and stacking
        ``(chain, draw)``.
        """
        post_impact = experiment.post_impact  # type: ignore[attr-defined]

        if "treated_units" in post_impact.dims:
            post_impact = post_impact.isel(treated_units=0)

        cumulative = post_impact.sum("obs_ind")
        return cumulative.stack(sample=("chain", "draw"))

    # ------------------------------------------------------------------
    # Hierarchical status-quo model
    # ------------------------------------------------------------------

    def _build_status_quo_model(
        self,
        fold_means: np.ndarray,
        fold_sds: np.ndarray,
    ) -> tuple[Any, np.ndarray]:
        """Fit the hierarchical status-quo model and return theta_new.

        Parameters
        ----------
        fold_means : np.ndarray
            Per-fold posterior means of cumulative impact.
        fold_sds : np.ndarray
            Per-fold posterior SDs of cumulative impact.

        Returns
        -------
        tuple[InferenceData, np.ndarray]
            ``(idata, theta_new_samples)`` where ``theta_new_samples``
            are draws from the posterior predictive for a new null
            period.
        """
        n_folds = len(fold_means)
        fold_sds = np.where(fold_sds < 1e-6, 1e-6, fold_sds)

        prior_mu_center = float(np.nanmean(fold_means))
        prior_mu_scale = float(np.nanstd(fold_means))
        if prior_mu_scale <= 0.0:
            prior_mu_scale = 1.0

        scale = self.prior_scale
        coords = {"fold": np.arange(n_folds)}

        with pm.Model(coords=coords) as model:
            observed_fold_means = pm.Data(
                "observed_fold_means", fold_means, dims="fold"
            )
            observed_fold_sd = pm.Data("observed_fold_sd", fold_sds, dims="fold")

            mu_status_quo = pm.Normal(
                "mu_status_quo",
                mu=prior_mu_center,
                sigma=5.0 * scale * prior_mu_scale,
            )
            tau_status_quo = pm.HalfNormal(
                "tau_status_quo",
                sigma=2.0 * scale * prior_mu_scale,
            )

            fold_z = pm.Normal("fold_z", mu=0.0, sigma=1.0, dims="fold")
            fold_true_effect = pm.Deterministic(
                "fold_true_effect",
                mu_status_quo + tau_status_quo * fold_z,
                dims="fold",
            )

            pm.Normal(
                "likelihood_fold_means",
                mu=fold_true_effect,
                sigma=observed_fold_sd,
                observed=observed_fold_means,
                dims="fold",
            )

            idata = pm.sample(**self.sample_kwargs)

        with model:
            model.add_coords({"new_period": np.arange(1)})
            pm.Normal(
                "theta_new",
                mu=mu_status_quo,
                sigma=tau_status_quo,
                dims="new_period",
            )
            pp = pm.sample_posterior_predictive(idata, var_names=["theta_new"])

        theta_new_samples = (
            pp["posterior_predictive"]["theta_new"]
            .stack(sample=("chain", "draw"))
            .values.squeeze()
        )

        return idata, theta_new_samples

    # ------------------------------------------------------------------
    # ROPE decision rule
    # ------------------------------------------------------------------

    @staticmethod
    def bayesian_rope_decision(
        posterior_samples: np.ndarray,
        rope_half_width: float,
        threshold: float,
    ) -> str:
        """Apply a ROPE-based Bayesian decision rule.

        Parameters
        ----------
        posterior_samples : np.ndarray
            Posterior draws of the total effect.
        rope_half_width : float
            Half-width of the ROPE interval ``[-rope, +rope]``.
        threshold : float
            Minimum posterior probability required to make a decision.

        Returns
        -------
        str
            One of ``"positive"``, ``"null"``, or ``"indeterminate"``.
        """
        samples = np.asarray(posterior_samples).ravel()
        prob_positive = float((samples > rope_half_width).mean())
        prob_null = float((np.abs(samples) <= rope_half_width).mean())

        if prob_positive >= threshold:
            return "positive"
        elif prob_null >= threshold:
            return "null"
        else:
            return "indeterminate"

    # ------------------------------------------------------------------
    # Assurance (operating characteristics)
    # ------------------------------------------------------------------

    def _draw_expected_effect_samples(self, n: int) -> np.ndarray:
        """Draw samples from the expected-effect prior."""
        prior = self.expected_effect_prior
        if prior is None:
            raise ValueError("expected_effect_prior is not set.")
        if isinstance(prior, np.ndarray):
            return prior
        if hasattr(prior, "rvs"):
            return np.asarray(prior.rvs(n))  # type: ignore[union-attr]
        raise TypeError(
            f"expected_effect_prior must be a numpy array or have an "
            f".rvs(n) method, got {type(prior).__name__}."
        )

    def _compute_assurance(
        self,
        theta_new_samples: np.ndarray,
        fold_sds: np.ndarray,
        n_posterior_samples: int,
    ) -> AssuranceResult:
        """Simulate decisions under null and alternative to get assurance.

        Parameters
        ----------
        theta_new_samples : np.ndarray
            Draws from the status-quo posterior predictive.
        fold_sds : np.ndarray
            Per-fold posterior SDs (used to simulate estimation noise).
        n_posterior_samples : int
            Number of posterior draws to simulate per replication.

        Returns
        -------
        AssuranceResult
        """
        expected_samples = self._draw_expected_effect_samples(len(theta_new_samples))
        n_reps = self.n_design_replications
        if n_reps is None:
            n_reps = min(len(theta_new_samples), len(expected_samples))

        rng = np.random.default_rng(self.random_seed)
        rope = self.rope_half_width
        if rope is None:
            raise ValueError(
                "rope_half_width must be set for assurance."
            )  # pragma: no cover

        null_decisions: list[str] = []
        for i in range(n_reps):
            true_effect = float(theta_new_samples[i % len(theta_new_samples)])
            sigma = float(rng.choice(fold_sds))
            simulated_posterior = rng.normal(
                loc=true_effect, scale=sigma, size=n_posterior_samples
            )
            null_decisions.append(
                self.bayesian_rope_decision(simulated_posterior, rope, self.threshold)
            )

        alt_decisions: list[str] = []
        for i in range(n_reps):
            true_effect = float(expected_samples[i % len(expected_samples)])
            sigma = float(rng.choice(fold_sds))
            simulated_posterior = rng.normal(
                loc=true_effect, scale=sigma, size=n_posterior_samples
            )
            alt_decisions.append(
                self.bayesian_rope_decision(simulated_posterior, rope, self.threshold)
            )

        null_arr = np.array(null_decisions)
        alt_arr = np.array(alt_decisions)

        return AssuranceResult(
            true_positive_rate=float((alt_arr == "positive").mean()),
            false_positive_rate=float((null_arr == "positive").mean()),
            true_negative_rate=float((null_arr == "null").mean()),
            false_negative_rate=float((alt_arr == "null").mean()),
            null_indeterminate_rate=float((null_arr == "indeterminate").mean()),
            alt_indeterminate_rate=float((alt_arr == "indeterminate").mean()),
            null_decisions=null_arr,
            alt_decisions=alt_arr,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext | None = None,
    ) -> CheckResult:
        """Run placebo-in-time analysis with hierarchical null model.

        Creates ``n_folds`` placebo experiments by shifting the treatment
        time backward.  Extracts posterior cumulative impact from each
        fold, then fits a hierarchical Bayesian model to characterise
        the status-quo distribution.  Compares the actual intervention
        effect against this null.

        When ``expected_effect_prior`` was provided at construction,
        also runs Bayesian assurance simulation.

        Can be used standalone (``context=None``) when
        ``experiment_factory`` was provided, or within a pipeline.

        Returns
        -------
        CheckResult
            With ``passed`` indicating whether the actual effect is
            clearly outside the null distribution, and rich metadata
            including the null samples and optional assurance results.
        """
        self.validate(experiment)
        factory = self._get_factory(context)
        treatment_time = experiment.treatment_time  # type: ignore[attr-defined]
        data = experiment.data  # type: ignore[attr-defined]
        intervention_length = self._compute_intervention_length(experiment)

        actual_cumulative = self._extract_cumulative_impact(experiment)
        actual_cumulative_mean = float(actual_cumulative.mean().values)

        fold_treatment_times = self._compute_fold_treatment_times(
            treatment_time, intervention_length
        )

        fold_results: list[PlaceboFoldResult] = []
        fold_summaries: list[str] = []
        skipped_folds: list[int] = []

        for fold_idx, pseudo_tt in enumerate(fold_treatment_times):
            fold_num = fold_idx + 1
            logger.info(
                "PlaceboInTime fold %d/%d: pseudo_treatment_time=%s",
                fold_num,
                self.n_folds,
                pseudo_tt,
            )

            fold_data = self._get_fold_data(data, pseudo_tt, intervention_length)

            if len(fold_data) < MIN_FOLD_OBSERVATIONS:
                logger.warning(
                    "Fold %d has only %d observations (minimum %d), skipping.",
                    fold_num,
                    len(fold_data),
                    MIN_FOLD_OBSERVATIONS,
                )
                skipped_folds.append(fold_num)
                fold_summaries.append(
                    f"Fold {fold_num}: SKIPPED (only {len(fold_data)} "
                    f"observations, need >= {MIN_FOLD_OBSERVATIONS})"
                )
                continue

            try:
                fold_experiment = factory(fold_data, pseudo_tt)
                cum_samples = self._extract_cumulative_impact(fold_experiment)
                f_mean = float(cum_samples.mean().values)
                f_sd = float(cum_samples.std().values)
            except Exception:
                logger.warning(
                    "Fold %d failed to fit (pseudo_treatment_time=%s), skipping.",
                    fold_num,
                    pseudo_tt,
                    exc_info=True,
                )
                skipped_folds.append(fold_num)
                fold_summaries.append(
                    f"Fold {fold_num}: SKIPPED (experiment failed to fit "
                    f"at pseudo treatment time {pseudo_tt})"
                )
                continue

            fold_result = PlaceboFoldResult(
                fold=fold_num,
                pseudo_treatment_time=pseudo_tt,
                experiment=fold_experiment,
                cumulative_impact_samples=cum_samples,
                fold_mean=f_mean,
                fold_sd=f_sd,
            )
            fold_results.append(fold_result)
            fold_summaries.append(
                f"Fold {fold_num}: pseudo treatment at {pseudo_tt} "
                f"— mean={f_mean:.2f}, sd={f_sd:.2f}"
            )

        n_completed = len(fold_results)
        n_skipped = len(skipped_folds)

        if n_completed < 1:
            parts = [
                f"Placebo-in-time analysis: 0 folds completed ({n_skipped} skipped).",
                "INCONCLUSIVE — no folds completed.",
            ]
            parts.extend(fold_summaries)
            return CheckResult(
                check_name="PlaceboInTime",
                passed=None,
                text="\n".join(parts),
                metadata={"fold_results": fold_results},
            )

        fold_means = np.array([fr.fold_mean for fr in fold_results])
        fold_sds = np.array([fr.fold_sd for fr in fold_results])

        idata, theta_new_samples = self._build_status_quo_model(fold_means, fold_sds)

        p_outside = float(
            (np.abs(actual_cumulative_mean) > np.abs(theta_new_samples)).mean()
        )
        passed = p_outside > self.threshold

        mu_post_mean = float(idata.posterior["mu_status_quo"].mean().values)
        tau_post_mean = float(idata.posterior["tau_status_quo"].mean().values)

        parts = [
            f"Placebo-in-time analysis: {n_completed} of {self.n_folds} folds completed"
        ]
        if n_skipped:
            parts[0] += f" ({n_skipped} skipped)"
        parts[0] += "."
        parts.append(
            f"Hierarchical status-quo model: "
            f"mu={mu_post_mean:.2f}, tau={tau_post_mean:.2f}."
        )
        parts.append(
            f"Actual cumulative impact: {actual_cumulative_mean:.2f}. "
            f"P(actual outside null) = {p_outside:.3f}."
        )
        if passed:
            parts.append("SUPPORTED — actual effect is outside the null distribution.")
        else:
            parts.append(
                "NOT SUPPORTED — actual effect is within the null distribution."
            )
        parts.extend(fold_summaries)
        text = "\n".join(parts)

        metadata: dict[str, Any] = {
            "fold_results": fold_results,
            "status_quo_idata": idata,
            "null_samples": theta_new_samples,
            "actual_cumulative_mean": actual_cumulative_mean,
            "p_effect_outside_null": p_outside,
        }

        n_posterior_samples = len(actual_cumulative.values)

        if self.expected_effect_prior is not None:
            assurance_result = self._compute_assurance(
                theta_new_samples, fold_sds, n_posterior_samples
            )
            metadata["assurance_result"] = assurance_result
            metadata["assurance"] = assurance_result.true_positive_rate

            text += (
                f"\n\nBayesian assurance (operating characteristics):\n"
                f"  Under NULL (status quo true):\n"
                f"    False Positive rate : "
                f"{assurance_result.false_positive_rate:.3f}\n"
                f"    True Negative rate  : "
                f"{assurance_result.true_negative_rate:.3f}\n"
                f"    Indeterminate rate  : "
                f"{assurance_result.null_indeterminate_rate:.3f}\n"
                f"  Under ALTERNATIVE (expected effect true):\n"
                f"    Assurance (TP rate) : "
                f"{assurance_result.true_positive_rate:.3f}\n"
                f"    False Negative rate : "
                f"{assurance_result.false_negative_rate:.3f}\n"
                f"    Indeterminate rate  : "
                f"{assurance_result.alt_indeterminate_rate:.3f}"
            )

        return CheckResult(
            check_name="PlaceboInTime",
            passed=passed,
            text=text,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        """Return a string representation of the check."""
        parts = [f"n_folds={self.n_folds}"]
        if self.expected_effect_prior is not None:
            parts.append("assurance=True")
        return f"PlaceboInTime({', '.join(parts)})"
