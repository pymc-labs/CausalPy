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

Supports two fold-selection strategies:

* **sequential** (default) — evenly-spaced sliding windows stepping
  backward from the actual treatment time.
* **random** — randomly sampled eligible windows from the
  pre-intervention period, with constraints on minimum training
  fraction, minimum gap between folds, and optional period exclusion.

Supports experiments with a ``treatment_time`` parameter
(InterruptedTimeSeries, SyntheticControl).  Requires a Bayesian model
backend (PyMC or pymc-forecast) for posterior extraction.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from causalpy.checks.base import CheckResult, clone_model
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pipeline import PipelineContext

logger = logging.getLogger(__name__)

MIN_FOLD_OBSERVATIONS = 3
MAX_RANDOM_SELECTION_RETRIES = 16

# Placebo windows are half-open (``[t, t + intervention_length)``) while the
# actual effect is summarised over the whole post-intervention period.  With
# the derived default ``intervention_length`` those two spans differ by at most
# the single observation sitting on the closing edge of the window, which is a
# geometry artefact rather than a misconfiguration, so the comparison-window
# warning only fires beyond that.
COMPARISON_WINDOW_OBSERVATION_TOLERANCE = 1

_DEFAULT_SAMPLE_KWARGS: dict[str, Any] = {
    "draws": 1000,
    "chains": 4,
    "target_accept": 0.97,
}


def _is_non_positive_length(length: Any) -> bool:
    """Return whether a window length is orderable against zero and not positive.

    Numeric and ``pd.Timedelta`` lengths are compared against a zero of the
    same kind.  Calendar offsets such as ``pd.DateOffset`` are not orderable,
    so they pass this check and are validated at run time by the
    intervention-window observation count in :meth:`PlaceboInTime.run`.
    """
    if isinstance(length, pd.Timedelta):
        return length <= pd.Timedelta(0)
    if isinstance(length, (int, float, np.integer, np.floating)):
        return bool(length <= 0)
    return False


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
        Number of placebo folds to create.  Must be >= 1.  Each fold
        consumes one ``intervention_length`` of pre-treatment history, so on
        a short pre-period a large ``n_folds`` produces ineligible folds that
        are skipped; shorten ``intervention_length`` to fit more folds.
    selection_method : {"sequential", "random"}, default "sequential"
        How to choose placebo windows.

        * ``"sequential"`` — evenly-spaced sliding windows stepping
          backward from the treatment time (original behaviour).
        * ``"random"`` — randomly sample eligible windows from the
          pre-intervention period, subject to ``min_training_pct``,
          ``min_gap``, and ``exclude_periods`` constraints.
        Every placebo fold must have at least one full intervention window
        of observed pre-treatment history. Sequential folds that do not meet
        this rule are skipped with a warning and deterministic
        ``skipped_folds`` metadata; random selection excludes them from its
        candidate pool.
    min_training_pct : float, default 0.30
        *(random mode only)* Minimum fraction of total pre-period
        observations that must precede each candidate placebo window.

        Note: the eligible pre-period is further shortened because a
        candidate's pseudo-intervention window must also end before the
        actual treatment.  With the derived default
        ``intervention_length`` (see below) that window is roughly the
        post-period length, which can make the effective eligible window
        much smaller than ``(1 - min_training_pct)`` suggests; pass an
        explicit ``intervention_length`` to widen it.
    min_gap : int, default 1
        *(random mode only)* Minimum number of pre-intervention
        observations between any two selected folds, measured as
        positions in the sorted pre-period index.  The default of ``1``
        only forbids picking the same candidate twice; use a larger
        value to spread folds further apart.  When ``allow_overlap``
        is ``False`` (the default) non-overlap of pseudo-intervention
        windows is enforced independently of ``min_gap``.
    allow_overlap : bool, default False
        *(random mode only)* If ``False`` (the default), selected
        pseudo-intervention windows are required to be non-overlapping
        in index/time units.  Two folds at times ``t_a`` and ``t_b``
        are considered non-overlapping when
        ``abs(t_a - t_b) >= intervention_length``.  Set to ``True`` to
        allow overlapping windows, which relaxes the constraint at the
        cost of violating the exchangeability assumption of the
        hierarchical status-quo model (each fold mean is treated as an
        independent draw from a common ``mu_status_quo``).
    exclude_periods : set[str] | None, default None
        *(random mode only)* Set of period labels to exclude from
        candidate selection. For datetime-indexed data, use
        ``"YYYY-MM"`` strings; for numeric indices, use string
        representations of the index values.
    experiment_factory : callable, optional
        Custom factory ``(data, treatment_time) -> BaseExperiment``.
        If ``None`` (default), the factory is derived from the pipeline's
        ``experiment_config``. Required for standalone (non-pipeline) use.
        This is the escape hatch for adapting the model to the eligible
        placebo-fold data; custom factories remain responsible for any
        model-specific randomness they introduce.
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
        hypothesis. Accepts any object with an ``.rvs(n)`` method
        (PreliZ, scipy) or a numpy array of pre-drawn samples. When
        ``random_seed`` is set, distributions exposing ``random_state``
        receive a derived Generator; legacy ``.rvs(n)`` distributions remain
        supported but emit a reproducibility warning and are recorded in
        result metadata. Provided together with ``rope_half_width``, assurance
        analysis runs automatically.
    rope_half_width : float, optional
        Half-width of the ROPE interval ``[-rope, +rope]``.  Required
        when ``expected_effect_prior`` is provided.
    n_design_replications : int, optional
        Number of simulation replications for assurance.  Defaults to
        ``min(theta_new.size, expected_effect_samples.size)``.
    random_seed : int, optional
        Master RNG seed for random fold selection, automatically constructed
        placebo-fold fits (using ``random_seed + fold_index``), hierarchical
        posterior predictive sampling, and assurance simulation. It also
        seeds the hierarchical ``pm.sample`` call unless
        ``sample_kwargs["random_seed"]`` is explicitly supplied, which takes
        precedence for that call only.
    intervention_length : int, float, ``pd.Timedelta`` or ``pd.DateOffset``, optional
        Length of each placebo intervention window, in index units.  When
        ``None`` (default) the length is derived from the experiment:
        ``treatment_end_time - treatment_time`` when the experiment defines
        an explicit intervention window, otherwise
        ``data.index.max() - treatment_time`` (roughly the post-period
        length).

        Set this explicitly to fit more well-supported folds into a short
        pre-period.  The derived default consumes one post-period worth of
        history per fold, so when the pre-period is only a few times longer
        than the post-period the earliest folds fail the eligibility rule
        above and are skipped.  A shorter ``intervention_length`` shortens
        both the placebo window and the history each fold requires, so more
        folds become eligible.

        The actual effect is still summarised over the full
        post-intervention period, so a window materially shorter than that
        period compares a long actual cumulative impact against a null built
        from short windows, which inflates ``P(actual outside null)``.  Both
        observation counts are recorded in
        ``metadata["comparison_window"]`` and a warning is emitted when they
        disagree by more than the one-observation half-open-window artefact
        of the derived default.

    Examples
    --------
    >>> import causalpy as cp  # doctest: +SKIP
    >>> check = cp.checks.PlaceboInTime(n_folds=3)  # doctest: +SKIP

    Random selection with constraints:

    >>> check = cp.checks.PlaceboInTime(  # doctest: +SKIP
    ...     n_folds=4,
    ...     selection_method="random",
    ...     min_training_pct=0.30,
    ...     min_gap=2,
    ...     random_seed=42,
    ... )
    """

    applicable_methods: set[type[BaseExperiment]] = {
        InterruptedTimeSeries,
        SyntheticControl,
    }

    def __init__(
        self,
        n_folds: int = 3,
        selection_method: Literal["sequential", "random"] = "sequential",
        min_training_pct: float = 0.30,
        min_gap: int = 1,
        allow_overlap: bool = False,
        exclude_periods: set[str] | None = None,
        experiment_factory: Any | None = None,
        sample_kwargs: dict[str, Any] | None = None,
        threshold: float = 0.95,
        prior_scale: float = 1.0,
        expected_effect_prior: Any | None = None,
        rope_half_width: float | None = None,
        n_design_replications: int | None = None,
        random_seed: int | None = None,
        intervention_length: Any | None = None,
    ) -> None:
        if n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        if intervention_length is not None and _is_non_positive_length(
            intervention_length
        ):
            raise ValueError(
                f"intervention_length must be positive, got {intervention_length!r}"
            )
        if selection_method not in ("sequential", "random"):
            raise ValueError(
                f"selection_method must be 'sequential' or 'random', "
                f"got {selection_method!r}"
            )
        if not 0 < min_training_pct < 1:
            raise ValueError(
                f"min_training_pct must be in (0, 1), got {min_training_pct}"
            )
        if min_gap < 1:
            raise ValueError(f"min_gap must be >= 1, got {min_gap}")
        if expected_effect_prior is not None and rope_half_width is None:
            raise ValueError(
                "rope_half_width is required when expected_effect_prior is "
                "provided.  Specify the ROPE half-width that defines "
                "practical significance."
            )
        self.n_folds = n_folds
        self.selection_method = selection_method
        self.min_training_pct = min_training_pct
        self.min_gap = min_gap
        self.allow_overlap = allow_overlap
        self.exclude_periods = exclude_periods
        self.experiment_factory = experiment_factory
        self.sample_kwargs = {**_DEFAULT_SAMPLE_KWARGS, **(sample_kwargs or {})}
        self.threshold = threshold
        self.prior_scale = prior_scale
        self.expected_effect_prior = expected_effect_prior
        self.rope_half_width = rope_half_width
        self.n_design_replications = n_design_replications
        self.random_seed = random_seed
        self.intervention_length = intervention_length

    def validate(self, experiment: BaseExperiment) -> None:
        """Check the experiment is compatible with PlaceboInTime.

        Parameters
        ----------
        experiment : BaseExperiment
            Candidate experiment to validate.

        Raises
        ------
        TypeError
            If the experiment lacks ``treatment_time`` or uses a
            non-Bayesian model backend.
        """
        if not hasattr(experiment, "treatment_time"):
            raise TypeError(
                f"{type(experiment).__name__} does not have a treatment_time "
                f"attribute. PlaceboInTime requires experiments with an "
                f"explicit treatment time."
            )
        # Any InferenceData-capable backend (PyMCModel or PyMCForecastModel)
        # yields the draw-level post_impact this check consumes.
        backend = getattr(experiment, "_model_backend", None)
        if backend is None or not backend.supports_idata:
            raise TypeError(
                f"PlaceboInTime requires a Bayesian model backend for "
                f"posterior extraction, but got "
                f"{type(experiment.model).__name__}. Use a PyMC model "
                f"(e.g. cp.pymc_models.LinearRegression) or a pymc-forecast "
                f"backend (cp.pymc_forecast_models.PyMCForecastModel)."
            )

    @staticmethod
    def _clone_model_for_fold(model: Any, random_seed: int | None) -> Any:
        """Clone a model and apply a fold-specific seed when supported."""
        if random_seed is None:
            return clone_model(model)

        sample_kwargs = getattr(model, "sample_kwargs", None)
        if isinstance(sample_kwargs, dict):
            cloned_model = clone_model(model)
            cloned_model.sample_kwargs = {
                **cloned_model.sample_kwargs,
                "random_seed": random_seed,
            }
            return cloned_model

        if hasattr(model, "random_seed"):
            original_seed = model.random_seed
            model.random_seed = random_seed
            try:
                return clone_model(model)
            finally:
                model.random_seed = original_seed

        return clone_model(model)

    def _get_factory(self, context: PipelineContext | None) -> Any:
        """Return a factory that accepts data, treatment time, and a fold seed."""
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
        model_template = kwargs.get("model")
        if model_template is None:
            context_experiment = getattr(context, "experiment", None)
            if context_experiment is not None:
                model_template = context_experiment.model
        if model_template is None:
            default_model_class = getattr(method, "_default_model_class", None)
            if default_model_class is not None:
                model_template = default_model_class()

        def _factory(
            data: pd.DataFrame,
            treatment_time: Any,
            fold_random_seed: int | None = None,
        ) -> BaseExperiment:
            """Create a fresh experiment with the given treatment time."""
            kw = dict(kwargs)
            kw["treatment_time"] = treatment_time
            if model_template is not None:
                kw["model"] = self._clone_model_for_fold(
                    model_template, fold_random_seed
                )
            return method(data, **kw)

        return _factory

    def _compute_intervention_length(self, experiment: BaseExperiment) -> Any:
        """Return the configured placebo window length, or derive it."""
        if self.intervention_length is not None:
            return self.intervention_length

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
        """Compute pseudo-treatment times for each fold (sequential mode)."""
        return [
            treatment_time - (self.n_folds - fold) * intervention_length
            for fold in range(self.n_folds)
        ]

    def _compute_random_fold_treatment_times(
        self,
        data: pd.DataFrame,
        treatment_time: Any,
        intervention_length: Any,
    ) -> list[Any]:
        """Randomly select pseudo-treatment times from the pre-period.

        The algorithm proceeds in two stages.

        1. **Candidate pool.**  Walks the sorted pre-intervention
           index and keeps each position that satisfies *all* of:

           * its period label is not in
             :attr:`exclude_periods`;
           * its position in the sorted pre-period index is at least
             ``ceil(min_training_pct * n_total)`` so each placebo fold
             has enough training data ahead of it;
           * its pseudo-intervention window
             ``[idx, idx + intervention_length)`` ends before the real
             ``treatment_time`` (so the placebo and real intervention
             cannot overlap in time).
           * its pre-period contains at least as many observations as the
             original intervention window, so every selected fold has one
             full intervention window of fitting history.

           If candidate eligibility or geometry constraints make the requested
           number infeasible, the method returns the exact maximum feasible
           subset rather than raising.

        2. **Random selection.**  When the maximum feasible subset
           contains :attr:`n_folds` values, the method uses
           :meth:`_try_greedy_selection` to select a random subset subject to
           :attr:`min_gap` (positional distance in the candidate pool) and
           :attr:`allow_overlap` (non-overlap of the pseudo windows in
           time/index units). Greedy without backtracking can paint itself
           into a corner, so up to :data:`MAX_RANDOM_SELECTION_RETRIES` passes
           are attempted. If all seeded attempts miss a full selection, the
           method falls back to the known feasible deterministic subset.

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset (must have a sorted index).
        treatment_time : Any
            The actual treatment time.
        intervention_length : Any
            Length of the intervention window.

        Returns
        -------
        list[Any]
            Sorted pseudo-treatment times. The list can contain fewer than
            :attr:`n_folds` values when eligibility or geometry constraints
            make a full selection infeasible.

        """
        pre_data = data.loc[data.index < treatment_time]
        if pre_data.empty:
            return []

        all_indices = pre_data.index.sort_values()
        n_total = len(all_indices)
        min_training = int(np.ceil(self.min_training_pct * n_total))
        exclude = self.exclude_periods or set()
        required_pre_period_rows = self._get_intervention_window_observation_count(
            data, treatment_time, intervention_length
        )

        # Each candidate carries its position in ``all_indices`` so
        # ``min_gap`` can be enforced as an observation-count distance
        # between selected folds, not a candidate-list distance.
        candidates: list[tuple[int, Any]] = []
        for pos, idx_val in enumerate(all_indices):
            if hasattr(idx_val, "strftime"):
                label = idx_val.strftime("%Y-%m")
            else:
                label = str(idx_val)
            if label in exclude:
                continue

            if pos < min_training:
                continue

            pseudo_end = idx_val + intervention_length
            if pseudo_end > treatment_time:
                continue
            pre_period_rows = int(all_indices.searchsorted(idx_val, side="left"))
            if pre_period_rows < required_pre_period_rows:
                continue

            candidates.append((pos, idx_val))

        if not candidates:
            return []

        maximum_selection = self._maximum_feasible_selection(
            candidates, intervention_length
        )
        if len(maximum_selection) < self.n_folds:
            return sorted(candidates[i][1] for i in maximum_selection)

        for attempt in range(MAX_RANDOM_SELECTION_RETRIES):
            # Deterministic sub-seeds: successive attempts reshuffle choices
            # in a reproducible way when ``random_seed`` is set and remain
            # non-deterministic (as expected) when it isn't.
            sub_seed: int | None
            if self.random_seed is None:
                sub_seed = None
            else:
                sub_seed = int(self.random_seed) + attempt
            rng = np.random.default_rng(sub_seed)
            selected = self._try_greedy_selection(candidates, intervention_length, rng)
            if len(selected) == self.n_folds:
                return sorted(candidates[i][1] for i in selected)

        return sorted(candidates[i][1] for i in maximum_selection[: self.n_folds])

    def _maximum_feasible_selection(
        self,
        candidates: list[tuple[int, Any]],
        intervention_length: Any,
    ) -> list[int]:
        """Return an exact maximum-cardinality subset of ordered candidates.

        Taking the earliest compatible candidate is optimal: every candidate
        that can follow a later start can also follow an earlier compatible
        start because both positional gaps and intervention windows are
        forward-ordered.
        """
        selected: list[int] = []
        for i, (pos_i, idx_val_i) in enumerate(candidates):
            if not selected:
                selected.append(i)
                continue

            pos_last, idx_val_last = candidates[selected[-1]]
            if pos_i - pos_last < self.min_gap:
                continue
            if not self.allow_overlap and self._windows_overlap(
                idx_val_i, idx_val_last, intervention_length
            ):
                continue
            selected.append(i)
        return selected

    def _try_greedy_selection(
        self,
        candidates: list[tuple[int, Any]],
        intervention_length: Any,
        rng: np.random.Generator,
    ) -> list[int]:
        """Select until the requested count is reached or no candidate remains."""
        pool = list(range(len(candidates)))
        selected: list[int] = []

        for _ in range(self.n_folds):
            valid: list[int] = []
            for i in pool:
                pos_i, idx_val_i = candidates[i]
                ok = True
                for s in selected:
                    pos_s, idx_val_s = candidates[s]
                    if abs(pos_i - pos_s) < self.min_gap:
                        ok = False
                        break
                    if not self.allow_overlap and self._windows_overlap(
                        idx_val_i, idx_val_s, intervention_length
                    ):
                        ok = False
                        break
                if ok:
                    valid.append(i)
            if not valid:
                break
            pick = int(rng.choice(valid))
            selected.append(pick)
            pool.remove(pick)
        return selected

    @staticmethod
    def _windows_overlap(idx_a: Any, idx_b: Any, intervention_length: Any) -> bool:
        """Return ``True`` iff two half-open intervention windows share a point.

        Each window starting at index ``idx`` is treated as the
        half-open interval ``[idx, idx + intervention_length)`` (start
        inclusive, end exclusive).  Under that convention, the windows
        ``[a, a + L)`` and ``[b, b + L)`` overlap iff
        ``abs(a - b) < L`` -- two back-to-back windows at distance
        exactly ``L`` are considered non-overlapping.  The
        ``idx + intervention_length`` arithmetic, rather than a direct
        Timedelta computation, lets the same expression handle numeric
        indices and datetime indices with ``pd.DateOffset`` uniformly.
        """
        earlier, later = (idx_a, idx_b) if idx_a <= idx_b else (idx_b, idx_a)
        return later < earlier + intervention_length

    def _get_fold_data(
        self,
        data: pd.DataFrame,
        pseudo_treatment_time: Any,
        intervention_length: Any,
    ) -> pd.DataFrame:
        """Extract data up to the end of the placebo intervention window."""
        pseudo_end = pseudo_treatment_time + intervention_length
        return data.loc[data.index < pseudo_end].copy()

    @staticmethod
    def _get_intervention_window_observation_count(
        data: pd.DataFrame,
        treatment_time: Any,
        intervention_length: Any,
    ) -> int:
        """Count observations in one full, in-range intervention window."""
        intervention_end = treatment_time + intervention_length
        index = data.index
        return int(((index >= treatment_time) & (index < intervention_end)).sum())

    @staticmethod
    def _describe_comparison_window(
        data: pd.DataFrame,
        treatment_time: Any,
        placebo_window_rows: int,
    ) -> dict[str, int]:
        """Compare the placebo window against the actual post-period span.

        The hierarchical null is built from cumulative impacts summed over
        placebo windows of ``placebo_window_rows`` observations, while the
        actual cumulative impact is summed over every post-intervention
        observation.  When the placebo windows are materially shorter the two
        quantities are not on the same footing and ``P(actual outside null)``
        is optimistic, so a warning is emitted.
        """
        actual_post_period_rows = int((data.index >= treatment_time).sum())
        excess = actual_post_period_rows - placebo_window_rows
        if excess > COMPARISON_WINDOW_OBSERVATION_TOLERANCE:
            warnings.warn(
                f"PlaceboInTime placebo windows span {placebo_window_rows} "
                f"observation(s) but the actual effect is summarised over "
                f"{actual_post_period_rows} post-intervention observation(s). "
                "The actual cumulative impact therefore accumulates over a "
                "longer span than the null distribution it is compared "
                "against, which inflates P(actual outside null). Lengthen "
                "intervention_length, or interpret the verdict as an upper "
                "bound.",
                stacklevel=3,
            )
        return {
            "placebo_window_observations": placebo_window_rows,
            "actual_post_period_observations": actual_post_period_rows,
        }

    @staticmethod
    def _get_fold_pre_period_observation_counts(
        data: pd.DataFrame,
        pseudo_treatment_time: Any,
        required_pre_period_rows: int,
    ) -> tuple[int, int]:
        """Return observed and required pre-period rows for a placebo fold."""
        observed_pre_period_rows = int((data.index < pseudo_treatment_time).sum())
        return observed_pre_period_rows, required_pre_period_rows

    @staticmethod
    def _make_skipped_fold_metadata(
        fold_index: int,
        pseudo_treatment_time: Any,
        observed_pre_period_rows: int | None,
        required_pre_period_rows: int | None,
        reason: str,
    ) -> dict[str, Any]:
        """Build deterministic metadata for a skipped placebo fold."""
        return {
            "fold_index": fold_index,
            "pseudo_treatment_time": pseudo_treatment_time,
            "observed_pre_period_rows": observed_pre_period_rows,
            "required_pre_period_rows": required_pre_period_rows,
            "reason": reason,
        }

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
        tuple[DataTree, np.ndarray]
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

            sample_kwargs = dict(self.sample_kwargs)
            if "random_seed" not in sample_kwargs and self.random_seed is not None:
                sample_kwargs["random_seed"] = self.random_seed
            idata = pm.sample(**sample_kwargs)

        with model:
            model.add_coords({"new_period": np.arange(1)})
            pm.Normal(
                "theta_new",
                mu=mu_status_quo,
                sigma=tau_status_quo,
                dims="new_period",
            )
            posterior_predictive_seed = self.random_seed
            if posterior_predictive_seed is None:
                posterior_predictive_seed = sample_kwargs.get("random_seed")
            posterior_predictive_kwargs: dict[str, Any] = {"var_names": ["theta_new"]}
            if posterior_predictive_seed is not None:
                posterior_predictive_kwargs["random_seed"] = posterior_predictive_seed
            pp = pm.sample_posterior_predictive(idata, **posterior_predictive_kwargs)

        theta_new_samples = (
            pp["posterior_predictive"]["theta_new"]
            .stack(sample=("chain", "draw"))
            .values.squeeze()
        )

        return idata, theta_new_samples

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

    def _rng_for_stage(self, stage: int) -> np.random.Generator:
        """Return an independent reproducible generator for one check stage."""
        if self.random_seed is None:
            return np.random.default_rng()
        seed_sequence = np.random.SeedSequence(int(self.random_seed))
        return np.random.default_rng(seed_sequence.spawn(2)[stage])

    @staticmethod
    def _rvs_accepts_random_state(prior: Any) -> bool:
        """Return whether ``prior.rvs`` explicitly supports ``random_state``."""
        try:
            parameters = inspect.signature(prior.rvs).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            or (
                parameter.name == "random_state"
                and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
            )
            for parameter in parameters
        )

    def _draw_expected_effect_samples(
        self,
        n: int,
        *,
        unseeded_custom_priors: list[dict[str, str]] | None = None,
    ) -> np.ndarray:
        """Draw samples from the expected-effect prior.

        Parameters
        ----------
        n : int
            Desired number of samples.
        unseeded_custom_priors : list[dict[str, str]], optional
            Run-local diagnostic records for legacy distributions that do not
            expose ``random_state``. With a master seed, seed-aware
            distributions receive ``.rvs(n, random_state=...)``; legacy
            distributions fall back to ``.rvs(n)`` with a warning. Pre-drawn
            numpy arrays are returned as-is, and :meth:`_compute_assurance`
            cycles through them via ``i % len(prior)`` when the array is
            shorter than the number of replications. A warning is emitted in
            this case because short arrays can introduce spurious structure
            in the simulated decisions.

        Returns
        -------
        np.ndarray
            Samples from the expected-effect prior.
        """
        prior = self.expected_effect_prior
        if prior is None:
            raise ValueError("expected_effect_prior is not set.")
        if isinstance(prior, np.ndarray):
            if len(prior) < n:
                warnings.warn(
                    f"expected_effect_prior has {len(prior)} samples, fewer "
                    f"than the {n} replications requested by the assurance "
                    f"simulation; the array will be cycled through via "
                    f"index % len(prior).  Pass a longer array or an object "
                    f"with an .rvs(n) method (e.g. a PreliZ/scipy "
                    f"distribution) to avoid cycling.",
                    stacklevel=2,
                )
            return prior
        if hasattr(prior, "rvs"):
            if self.random_seed is None:
                return np.asarray(prior.rvs(n))  # type: ignore[union-attr]
            if self._rvs_accepts_random_state(prior):
                return np.asarray(
                    prior.rvs(  # type: ignore[union-attr]
                        n, random_state=self._rng_for_stage(0)
                    )
                )

            prior_type = f"{type(prior).__module__}.{type(prior).__qualname__}"
            if unseeded_custom_priors is not None:
                unseeded_custom_priors.append(
                    {
                        "prior_type": prior_type,
                        "reason": "rvs_does_not_accept_random_state",
                    }
                )
            warnings.warn(
                "expected_effect_prior.rvs does not expose random_state; "
                "using unseeded legacy .rvs(n). Assurance simulation is "
                "not reproducible for this custom prior; result metadata "
                "marks its type as unseeded.",
                stacklevel=2,
            )
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
        *,
        unseeded_custom_priors: list[dict[str, str]] | None = None,
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
        unseeded_custom_priors : list[dict[str, str]], optional
            Run-local diagnostic records for legacy expected-effect priors.

        Returns
        -------
        AssuranceResult
        """
        expected_samples = self._draw_expected_effect_samples(
            len(theta_new_samples),
            unseeded_custom_priors=unseeded_custom_priors,
        )
        n_reps = self.n_design_replications
        if n_reps is None:
            n_reps = min(len(theta_new_samples), len(expected_samples))

        rng = self._rng_for_stage(1)
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
            # Under the alternative, the observed effect is the expected
            # treatment effect added on top of the null baseline noise,
            # matching the paper's formulation: theta_new + expected_effect.
            null_component = float(theta_new_samples[i % len(theta_new_samples)])
            treatment_component = float(expected_samples[i % len(expected_samples)])
            true_effect = null_component + treatment_component
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

        Parameters
        ----------
        experiment : BaseExperiment
            The fitted experiment whose treatment time will be shifted to
            generate placebo folds.
        context : PipelineContext or None, default None
            Pipeline context providing ``experiment_config`` for re-fits.
            If ``None``, an explicit ``experiment_factory`` must have been
            supplied at construction time.

        Returns
        -------
        CheckResult
            With ``passed`` indicating whether the actual effect is
            clearly outside the null distribution, and rich metadata
            including the null samples and optional assurance results.
        """
        self.validate(experiment)
        unseeded_custom_priors: list[dict[str, str]] = []
        factory = self._get_factory(context)
        treatment_time = experiment.treatment_time  # type: ignore[attr-defined]
        data = experiment.data  # type: ignore[attr-defined]
        intervention_length = self._compute_intervention_length(experiment)
        required_pre_period_rows = self._get_intervention_window_observation_count(
            data, treatment_time, intervention_length
        )
        if required_pre_period_rows < 1:
            raise ValueError(
                f"intervention_length={intervention_length!r} spans no "
                f"observations at treatment_time={treatment_time!r}, so no "
                "placebo window can be built. Pass a longer "
                "intervention_length."
            )
        comparison_window = self._describe_comparison_window(
            data, treatment_time, required_pre_period_rows
        )

        actual_cumulative = self._extract_cumulative_impact(experiment)
        actual_cumulative_mean = float(actual_cumulative.mean().values)

        if self.selection_method == "random":
            fold_treatment_times = self._compute_random_fold_treatment_times(
                data, treatment_time, intervention_length
            )
        else:
            fold_treatment_times = self._compute_fold_treatment_times(
                treatment_time, intervention_length
            )

        fold_results: list[PlaceboFoldResult] = []
        fold_summaries: list[str] = []
        skipped_folds: list[dict[str, Any]] = []
        insufficient_pre_period_folds: list[dict[str, Any]] = []
        random_selection_shortfall_folds: list[dict[str, Any]] = []
        random_selection_shortfall_summaries: list[str] = []
        if (
            self.selection_method == "random"
            and len(fold_treatment_times) < self.n_folds
        ):
            for fold_idx in range(len(fold_treatment_times), self.n_folds):
                random_selection_shortfall_folds.append(
                    self._make_skipped_fold_metadata(
                        fold_idx,
                        None,
                        None,
                        required_pre_period_rows,
                        "insufficient_feasible_random_folds",
                    )
                )
                random_selection_shortfall_summaries.append(
                    f"Fold {fold_idx + 1}: SKIPPED (no feasible pseudo "
                    "treatment time after random eligibility and geometry "
                    "constraints)"
                )
            skipped_folds.extend(random_selection_shortfall_folds)

        for fold_idx, pseudo_tt in enumerate(fold_treatment_times):
            fold_num = fold_idx + 1
            logger.info(
                "PlaceboInTime fold %d/%d: pseudo_treatment_time=%s",
                fold_num,
                self.n_folds,
                pseudo_tt,
            )

            observed_pre_period_rows, _ = self._get_fold_pre_period_observation_counts(
                data,
                pseudo_tt,
                required_pre_period_rows,
            )
            if observed_pre_period_rows < required_pre_period_rows:
                skipped_fold = self._make_skipped_fold_metadata(
                    fold_idx,
                    pseudo_tt,
                    observed_pre_period_rows,
                    required_pre_period_rows,
                    "insufficient_pre_period",
                )
                skipped_folds.append(skipped_fold)
                insufficient_pre_period_folds.append(skipped_fold)
                fold_summaries.append(
                    f"Fold {fold_num}: SKIPPED (only "
                    f"{observed_pre_period_rows} pre-treatment observations, "
                    f"need >= {required_pre_period_rows} for one full "
                    f"intervention window)"
                )
                continue

            fold_data = self._get_fold_data(data, pseudo_tt, intervention_length)

            if len(fold_data) < MIN_FOLD_OBSERVATIONS:
                logger.warning(
                    "Fold %d has only %d observations (minimum %d), skipping.",
                    fold_num,
                    len(fold_data),
                    MIN_FOLD_OBSERVATIONS,
                )
                skipped_folds.append(
                    self._make_skipped_fold_metadata(
                        fold_idx,
                        pseudo_tt,
                        observed_pre_period_rows,
                        required_pre_period_rows,
                        "insufficient_fold_observations",
                    )
                )
                fold_summaries.append(
                    f"Fold {fold_num}: SKIPPED (only {len(fold_data)} "
                    f"observations, need >= {MIN_FOLD_OBSERVATIONS})"
                )
                continue

            try:
                fold_random_seed = (
                    None
                    if self.random_seed is None
                    else int(self.random_seed) + fold_idx
                )
                if self.experiment_factory is None:
                    fold_experiment = factory(
                        fold_data,
                        pseudo_tt,
                        fold_random_seed=fold_random_seed,
                    )
                else:
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
                skipped_folds.append(
                    self._make_skipped_fold_metadata(
                        fold_idx,
                        pseudo_tt,
                        observed_pre_period_rows,
                        required_pre_period_rows,
                        "experiment_failed_to_fit",
                    )
                )
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

        fold_summaries.extend(random_selection_shortfall_summaries)
        if insufficient_pre_period_folds or random_selection_shortfall_folds:
            warning_parts: list[str] = []
            if insufficient_pre_period_folds:
                warning_parts.append(
                    f"{len(insufficient_pre_period_folds)} fold(s) had "
                    "pre-treatment history shorter than one full intervention "
                    "window"
                )
            if random_selection_shortfall_folds:
                warning_parts.append(
                    f"random selection yielded only "
                    f"{len(fold_treatment_times)} of {self.n_folds} requested "
                    "feasible fold(s) after eligibility and geometry constraints"
                )
            warnings.warn(
                "PlaceboInTime skipped folds because "
                + "; ".join(warning_parts)
                + ". Use fewer folds or an experiment_factory tailored to "
                "the eligible fold data; skipped_folds metadata records the "
                "observed and required pre-period rows.",
                stacklevel=2,
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
                metadata={
                    "fold_results": fold_results,
                    "n_folds_requested": self.n_folds,
                    "n_folds_completed": n_completed,
                    "skipped_folds": skipped_folds,
                    "intervention_length": intervention_length,
                    "comparison_window": comparison_window,
                    "rope_half_width": self.rope_half_width,
                    "threshold": self.threshold,
                    "expected_effect_prior": self.expected_effect_prior,
                    "unseeded_custom_priors": unseeded_custom_priors,
                },
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
            "n_folds_requested": self.n_folds,
            "n_folds_completed": n_completed,
            "skipped_folds": skipped_folds,
            "intervention_length": intervention_length,
            "comparison_window": comparison_window,
            "fold_sds": fold_sds,
            "status_quo_idata": idata,
            "null_samples": theta_new_samples,
            "actual_cumulative_mean": actual_cumulative_mean,
            "p_effect_outside_null": p_outside,
            "rope_half_width": self.rope_half_width,
            "threshold": self.threshold,
            "expected_effect_prior": self.expected_effect_prior,
            "unseeded_custom_priors": unseeded_custom_priors,
        }

        n_posterior_samples = len(actual_cumulative.values)

        if self.expected_effect_prior is not None:
            assurance_result = self._compute_assurance(
                theta_new_samples,
                fold_sds,
                n_posterior_samples,
                unseeded_custom_priors=unseeded_custom_priors,
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
        if self.intervention_length is not None:
            parts.append(f"intervention_length={self.intervention_length!r}")
        if self.selection_method != "sequential":
            parts.append(f"selection_method={self.selection_method!r}")
        if self.allow_overlap:
            parts.append("allow_overlap=True")
        if self.expected_effect_prior is not None:
            parts.append("assurance=True")
        return f"PlaceboInTime({', '.join(parts)})"
