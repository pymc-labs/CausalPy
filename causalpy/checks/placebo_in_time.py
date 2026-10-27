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
Placebo-in-time calibration and descriptive historical comparisons.

Explicit historical schedules can report average observed-minus-expected differences without calibration. The default retains hierarchical calibration of cumulative effects.

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

Explicit ``treatment_times`` can schedule historical windows independently of
their length. With ``calibrate=False``, report average historical differences
and sampling diagnostics without fitting a null or issuing a causal verdict.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal, cast

import matplotlib as mpl
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from plotnine import (
    aes,
    after_stat,
    geom_histogram,
    geom_point,
    geom_segment,
    geom_text,
    geom_vline,
    ggplot,
    labs,
    scale_colour_manual,
    scale_fill_manual,
    scale_y_continuous,
    theme_void,
)

from causalpy._arviz_compat import hdi
from causalpy.checks._plot_helpers import draw_figure
from causalpy.checks.base import CheckResult, clone_model
from causalpy.checks.operating_characteristics import (
    AssuranceResult,
    compute_assurance_rates,
)
from causalpy.constants import HDI_PROB
from causalpy.date_utils import validate_treatment_time_against_index
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pipeline import PipelineContext

logger = logging.getLogger(__name__)

MIN_FOLD_OBSERVATIONS = 3
MAX_RANDOM_SELECTION_RETRIES = 16
_DEFAULT_PLOT_TITLE = "Placebo-in-Time calibration"
_DEFAULT_FIGSIZE = (7.0, 9.0)
_CALENDAR_MISMATCH_REASON = (
    "comparison index does not match the complete requested frequency grid"
)

# The hierarchical status-quo (null) model estimates the between-fold spread
# ``tau_status_quo`` from the completed folds.  That spread is unidentified from
# a single fold: ``prior_mu_scale = np.nanstd([x]) == 0.0`` falls back to
# ``1.0``, stripping all data scaling from both the ``mu`` and ``tau`` priors
# and collapsing the null distribution to a prior-driven O(1) width.  On a
# large-scale series that flips the verdict to a spurious SUPPORTED, so at least
# this many usable folds are required before a verdict is issued.
MIN_USABLE_FOLDS = 2

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


class _NullModelUnidentifiedError(ValueError):
    """Raised when the hierarchical null's between-fold spread is unidentified.

    A subclass of :class:`ValueError` so existing ``except ValueError`` callers
    keep working.  :meth:`PlaceboInTime.run` catches it and abstains
    (INCONCLUSIVE) rather than building a scale-free null that could report a
    spurious verdict.
    """


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
    average_impact_samples : xr.DataArray or None, default None
        Posterior samples of the average impact across comparison observations, populated for descriptive runs. Cumulative fields retain their summed-impact meaning.
    """

    fold: int
    pseudo_treatment_time: Any
    experiment: BaseExperiment
    cumulative_impact_samples: xr.DataArray
    fold_mean: float
    fold_sd: float
    average_impact_samples: xr.DataArray | None = None


@dataclass(frozen=True)
class _FoldPlan:
    """Geometry and training requirement for one requested fold."""

    fold: int
    start: Any
    end: Any
    comparison_end: Any
    n_training: int | None
    n_comparison: int | None
    required_training: int


@dataclass
class _FoldRecord:
    """Single execution record from which all public fold views are derived."""

    plan: _FoldPlan
    status: Literal["ineligible", "failed", "completed"] = "ineligible"
    reason: str | None = None
    result: PlaceboFoldResult | None = None

    def schedule(self) -> dict[str, Any]:
        """Expose requested geometry; fit failures do not alter the plan."""
        return {
            "fold": self.plan.fold,
            "pseudo_treatment_time": self.plan.start,
            "comparison_end": self.plan.comparison_end,
            "window_end_exclusive": self.plan.end,
            "n_training": self.plan.n_training,
            "n_comparison": self.plan.n_comparison,
            "reason": self.reason
            if self.status == "ineligible"
            and self.reason
            not in {"insufficient_pre_period", "insufficient_fold_observations"}
            else None,
        }

    def skipped_metadata(self) -> dict[str, Any]:
        """Preserve the calibrated skipped-fold metadata contract."""
        return {
            "fold_index": self.plan.fold - 1,
            "pseudo_treatment_time": self.plan.start,
            "observed_pre_period_rows": self.plan.n_training,
            "required_pre_period_rows": self.plan.required_training,
            "reason": "experiment_failed_to_fit"
            if self.status == "failed"
            else self.reason,
        }

    def summary(self) -> str:
        """Render the legacy per-fold prose from the recorded outcome."""
        prefix = f"Fold {self.plan.fold}: "
        if self.result is not None:
            return (
                prefix + f"pseudo treatment at {self.plan.start} "
                f"— mean={self.result.fold_mean:.2f}, sd={self.result.fold_sd:.2f}"
            )
        if self.status == "failed":
            detail = (
                f"experiment failed to fit at pseudo treatment time {self.plan.start}"
            )
        elif self.reason == "insufficient_pre_period":
            detail = (
                f"only {self.plan.n_training} pre-treatment observations, "
                f"need >= {self.plan.required_training} for one full intervention window"
            )
        elif self.reason == "insufficient_fold_observations":
            assert self.plan.n_training is not None
            assert self.plan.n_comparison is not None
            detail = (
                f"only {self.plan.n_training + self.plan.n_comparison} "
                f"observations, need >= {MIN_FOLD_OBSERVATIONS}"
            )
        elif self.reason == "insufficient_feasible_random_folds":
            detail = "no feasible pseudo treatment time after random eligibility and geometry constraints"
        else:
            detail = str(self.reason)
        return prefix + f"SKIPPED ({detail})"


class PlaceboInTime:
    """Historical placebo refits with optional hierarchical calibration.

    Shifts the treatment time backward into the pre-intervention period
    to create ``n_folds`` placebo experiments.  Extracts the posterior
    cumulative impact from each fold, then fits a hierarchical Bayesian
    model to characterise the "status quo" distribution of effects when
    no intervention occurred.  The actual intervention's cumulative
    effect is compared against this learned null. With ``calibrate=False``, report average historical differences and diagnostics without fitting a null or issuing a causal verdict.

    When ``expected_effect_prior`` and ``rope_half_width`` are provided,
    additionally computes exact closed-form Bayesian assurance (operating
    characteristics).

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
        In random selection, if ``False`` (the default), selected
        pseudo-intervention windows are required to be non-overlapping
        in index/time units.  Two folds at times ``t_a`` and ``t_b``
        are considered non-overlapping when
        ``abs(t_a - t_b) >= intervention_length``.  Set to ``True`` to
        allow overlapping windows, which relaxes the constraint at the
        cost of violating the exchangeability assumption of the
        hierarchical status-quo model (each fold mean is treated as an
        independent draw from a common ``mu_status_quo``).
        Explicitly scheduled overlapping windows also require this flag and
        ``calibrate=False``; calibrated explicit schedules cannot overlap.
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
        Finite probability cutoff in ``(0, 1)``. Used both for ``passed``
        (P(actual effect outside null) must exceed this) and for the ROPE
        decision rule when computing assurance.
    prior_scale : float, default 1.0
        Multiplier for auto-computed prior widths on the hierarchical
        model.  The priors are
        ``mu ~ Normal(center, 5 * prior_scale * data_scale)`` and
        ``tau ~ HalfNormal(2 * prior_scale * data_scale)``.
    expected_effect_prior : distribution or array, optional
        Prior belief about the true total effect under the alternative
        hypothesis. Accepts a frozen distribution exposing ``.cdf`` and
        ``.sf``, any object with an ``.rvs(n)`` method (PreliZ, scipy), or a
        numpy array of pre-drawn samples. Frozen distributions and arrays are
        evaluated directly by the exact closed-form assurance calculation.
        When ``random_seed`` is set, RVS-only distributions exposing
        ``random_state`` receive a derived Generator; legacy ``.rvs(n)``
        distributions remain supported but emit a reproducibility warning and
        are recorded in result metadata. Provided together with
        ``rope_half_width``, assurance analysis runs automatically.
    rope_half_width : float, optional
        Finite nonnegative half-width of the ROPE interval ``[-rope, +rope]``.
        Required when ``expected_effect_prior`` is provided.
    n_design_replications : int, optional
        Number of prior samples drawn only for RVS-only expected-effect
        priors. Defaults to the number of status-quo samples. It has no
        effect for numpy arrays or frozen priors with ``.cdf`` and ``.sf``.
    random_seed : int, optional
        posterior predictive sampling, and RVS-only expected-effect-prior
        sampling. It also seeds the hierarchical ``pm.sample`` call unless
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
    make_figures : bool, default True
        Whether :meth:`run` appends a figure to ``CheckResult.figures``. Calibrated runs use :meth:`plot_calibration`; descriptive runs use :meth:`plot_folds`. With this control enabled, ``figures[0]`` is available even when all fits fail.
    treatment_times : sequence or None, optional
        Explicit, strictly ascending unique starts, copied at construction and exactly ``n_folds`` long. Incompatible with random selection. Starts must match the data index type, as ``treatment_time`` does (``pd.Timestamp`` on a ``DatetimeIndex``). Every historical window is validated before fitting. Overlapping scheduled windows require ``allow_overlap=True`` and ``calibrate=False``.
    min_training_rows : int or None, optional
        Nonnegative minimum fitting-history count in every selection mode. Descriptive folds additionally require at least their planned comparison count in training rows; calibrated folds retain the original actual-window history requirement.
    comparison_frequency : str or None, optional
        Pandas calendar frequency for complete historical comparison grids (for example, ``"MS"``). Requires a datetime index. Each observed half-open window must exactly match its requested grid.
    calibrate : bool, default True
        Fit the hierarchical null and compare the real cumulative effect. With ``False``, report informational average historical differences, retain failed/ineligible rows, and omit actual-effect extraction, null fitting and assurance. Expected-effect/ROPE assurance configuration is incompatible with descriptive runs.
    hdi_prob : float, default HDI_PROB
        Probability mass for descriptive average-difference HDIs. Defaults to :data:`~causalpy.constants.HDI_PROB` (currently 0.94).

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
        make_figures: bool = True,
        *,
        treatment_times: Sequence[Any] | None = None,
        min_training_rows: int | None = None,
        comparison_frequency: str | None = None,
        calibrate: bool = True,
        hdi_prob: float = HDI_PROB,
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
        if (
            isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, (int, float, np.integer, np.floating))
            or not np.isfinite(threshold)
            or not 0 < threshold < 1
        ):
            raise ValueError(
                f"threshold must be a finite probability in (0, 1), got {threshold!r}"
            )
        if rope_half_width is not None and (
            isinstance(rope_half_width, (bool, np.bool_))
            or not isinstance(rope_half_width, (int, float, np.integer, np.floating))
            or not np.isfinite(rope_half_width)
            or rope_half_width < 0
        ):
            raise ValueError(
                "rope_half_width must be a finite nonnegative real number, "
                f"got {rope_half_width!r}"
            )
        if random_seed is not None and (
            isinstance(random_seed, (bool, np.bool_))
            or not isinstance(random_seed, Integral)
            or random_seed < 0
        ):
            raise ValueError(
                f"random_seed must be a nonnegative integer or None, got {random_seed!r}"
            )
        if expected_effect_prior is not None and rope_half_width is None:
            raise ValueError(
                "rope_half_width is required when expected_effect_prior is "
                "provided.  Specify the ROPE half-width that defines "
                "practical significance."
            )
        if n_design_replications is not None and (
            isinstance(n_design_replications, (bool, np.bool_))
            or not isinstance(n_design_replications, Integral)
            or n_design_replications < 1
        ):
            raise ValueError(
                "n_design_replications must be a positive non-bool int when "
                f"provided, got {n_design_replications!r}"
            )
        if not isinstance(calibrate, bool):
            raise ValueError("calibrate must be a bool")
        if min_training_rows is not None and (
            isinstance(min_training_rows, (bool, np.bool_))
            or not isinstance(min_training_rows, Integral)
            or min_training_rows < 0
        ):
            raise ValueError("min_training_rows must be a nonnegative integer or None")
        if (
            isinstance(hdi_prob, (bool, np.bool_))
            or not isinstance(hdi_prob, (int, float, np.integer, np.floating))
            or not np.isfinite(hdi_prob)
            or not 0 < hdi_prob < 1
        ):
            raise ValueError("hdi_prob must be a finite probability in (0, 1)")
        if comparison_frequency is not None:
            if not isinstance(comparison_frequency, str) or not comparison_frequency:
                raise ValueError(
                    "comparison_frequency must be a pandas frequency string"
                )
            pd.tseries.frequencies.to_offset(comparison_frequency)
        if not calibrate and any(
            value is not None
            for value in (expected_effect_prior, rope_half_width, n_design_replications)
        ):
            raise ValueError("Assurance configuration requires calibrate=True")
        times = None
        if treatment_times is not None:
            if isinstance(treatment_times, (str, bytes, Mapping)):
                raise TypeError(
                    "treatment_times must be a sequence of requested starts"
                )
            times = tuple(treatment_times)
            if not times or len(times) != n_folds:
                raise ValueError(
                    "treatment_times must be nonempty and have length n_folds"
                )
            if any(
                not earlier < later
                for earlier, later in zip(times, times[1:], strict=False)
            ):
                raise ValueError(
                    "treatment_times must be strictly ascending and unique"
                )
            if selection_method == "random":
                raise ValueError(
                    "treatment_times is incompatible with random selection"
                )
        self.treatment_times = times
        self.min_training_rows = min_training_rows
        self.comparison_frequency = comparison_frequency
        self.calibrate = calibrate
        self.hdi_prob = float(hdi_prob)
        self.n_folds = n_folds
        self.selection_method = selection_method
        self.min_training_pct = min_training_pct
        self.min_gap = min_gap
        self.allow_overlap = allow_overlap
        self.exclude_periods = exclude_periods
        self.experiment_factory = experiment_factory
        self.sample_kwargs = {**_DEFAULT_SAMPLE_KWARGS, **(sample_kwargs or {})}
        self.threshold = float(threshold)
        self.prior_scale = prior_scale
        self.expected_effect_prior = expected_effect_prior
        self.rope_half_width = (
            None if rope_half_width is None else float(rope_half_width)
        )
        self.n_design_replications = n_design_replications
        self.random_seed = None if random_seed is None else int(random_seed)
        self.intervention_length = intervention_length
        self.make_figures = make_figures

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
        # yields the draw-level posterior impact this check consumes.
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
            # This factory receives a slice of experiment.data, which is already
            # normalized: time_column has been moved onto the index, so the
            # column no longer exists and replaying the argument would fail.
            # The other checks re-fit from the caller's original data and do
            # need it, so this is dropped here rather than at the source.
            kw.pop("time_column", None)
            if model_template is not None:
                kw["model"] = self._clone_model_for_fold(
                    model_template, fold_random_seed
                )
            return method(data, **kw).fit()

        return _factory

    def _compute_intervention_length(self, experiment: BaseExperiment) -> Any:
        """Return the configured placebo window length, or derive it."""
        if self.intervention_length is not None:
            return self.intervention_length

        treatment_time = experiment.treatment_time  # type: ignore[attr-defined]
        data = experiment.data

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
            candidate = self._plan_window(
                data, treatment_time, intervention_length, idx_val, 0
            )
            if candidate.reason is not None:
                continue
            assert candidate.plan.n_training is not None
            if candidate.plan.n_training < candidate.plan.required_training:
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
    def _extract_cumulative_impact(experiment: BaseExperiment) -> xr.DataArray:
        """Extract posterior cumulative impact from a fitted experiment.

        Returns an ``xr.DataArray`` with a single ``sample`` dimension
        obtained by summing over ``obs_ind`` and stacking
        ``(chain, draw)``.
        """
        post_impact = experiment.result.impact_post

        if "treated_units" in post_impact.dims:
            post_impact = post_impact.isel(treated_units=0)

        cumulative = post_impact.sum("obs_ind")
        return cumulative.stack(sample=("chain", "draw"))

    def _plan_window(
        self,
        data: pd.DataFrame,
        treatment_time: Any,
        intervention_length: Any,
        start: Any,
        fold: int,
    ) -> _FoldRecord:
        """Calculate geometry once for candidate eligibility and execution."""
        if self.comparison_frequency is not None and not isinstance(
            data.index, pd.DatetimeIndex
        ):
            raise ValueError("comparison_frequency requires a DatetimeIndex")
        end = start + intervention_length
        observed = data.index[(data.index >= start) & (data.index < end)]
        reason = None
        # Default calibrated callers retain their historical eligibility rules.
        if (
            self.treatment_times is not None
            or not self.calibrate
            or self.comparison_frequency is not None
        ):
            if start not in data.index:
                reason = "requested start is absent from the data index"
            elif end <= start or end > treatment_time:
                reason = "window must have positive length and end no later than the real intervention"
            elif len(observed) == 0:
                reason = "comparison window is empty"
            elif self.comparison_frequency is not None:
                expected = pd.date_range(
                    start, end, freq=self.comparison_frequency, inclusive="left"
                )
                if not observed.equals(expected):
                    reason = _CALENDAR_MISMATCH_REASON
        required_rows = (
            self._get_intervention_window_observation_count(
                data, treatment_time, intervention_length
            )
            if self.calibrate
            else len(observed)
        )
        return _FoldRecord(
            plan=_FoldPlan(
                fold=fold,
                start=start,
                end=end,
                comparison_end=observed[-1] if len(observed) else None,
                n_training=int((data.index < start).sum()),
                n_comparison=len(observed),
                required_training=max(self.min_training_rows or 0, required_rows),
            ),
            reason=reason,
        )

    def _plan_windows(
        self,
        data: pd.DataFrame,
        treatment_time: Any,
        intervention_length: Any,
        starts: list[Any],
    ) -> list[_FoldRecord]:
        """Preflight all requested geometry before spending any fitting budget."""
        if self.comparison_frequency is not None and not isinstance(
            data.index, pd.DatetimeIndex
        ):
            raise ValueError("comparison_frequency requires a DatetimeIndex")
        if self.treatment_times is not None:
            for start in starts:
                validate_treatment_time_against_index(
                    data.index, start, name="treatment_times"
                )
        records = [
            self._plan_window(
                data, treatment_time, intervention_length, start, position
            )
            for position, start in enumerate(starts, start=1)
        ]
        if self.treatment_times is not None:
            for record in records:
                if record.reason is not None:
                    raise ValueError(
                        f"Invalid placebo window at {record.plan.start!r}: {record.reason}"
                    )
            overlaps = any(
                self._windows_overlap(earlier, later, intervention_length)
                for earlier, later in zip(starts, starts[1:], strict=False)
            )
            if overlaps and (not self.allow_overlap or self.calibrate):
                raise ValueError(
                    "Scheduled overlapping windows require allow_overlap=True and calibrate=False"
                )
        required_rows = self._get_intervention_window_observation_count(
            data, treatment_time, intervention_length
        )
        for fold in range(len(records) + 1, self.n_folds + 1):
            records.append(
                _FoldRecord(
                    plan=_FoldPlan(fold, None, None, None, None, None, required_rows),
                    reason="insufficient_feasible_random_folds",
                )
            )
        return records

    def _descriptive_row(self, record: _FoldRecord) -> dict[str, Any]:
        """Derive one table row from a fold's plan and optional fitted result."""
        row = {
            key: value
            for key, value in record.schedule().items()
            if key not in {"window_end_exclusive", "reason"}
        } | {
            "status": record.status,
            "reason": record.reason,
            "mean": np.nan,
            "hdi_lower": np.nan,
            "hdi_upper": np.nan,
            "hdi_prob": self.hdi_prob,
            "divergences": None,
            "max_rhat": None,
            "min_bulk_ess": None,
            "diagnostics_status": "unavailable",
            "diagnostics_reason": "No completed placebo fit",
        }
        if record.result is not None:
            samples = record.result.average_impact_samples
            assert samples is not None
            bounds = hdi(samples, prob=self.hdi_prob, dim="sample")
            row.update(
                mean=float(samples.mean().values),
                hdi_lower=float(bounds.sel(hdi="lower").values),
                hdi_upper=float(bounds.sel(hdi="higher").values),
            )
            row.update(self._fold_diagnostics(record.result.experiment))
        return row

    @staticmethod
    def _fold_diagnostics(experiment: BaseExperiment) -> dict[str, Any]:
        """Consume the adapter's full-fit diagnostic contract without backend dispatch."""
        try:
            return dict(experiment._model_backend.sampling_diagnostics())
        except Exception as error:
            return {
                "divergences": None,
                "max_rhat": None,
                "min_bulk_ess": None,
                "diagnostics_status": "unavailable",
                "diagnostics_reason": f"Sampling diagnostics unavailable: {type(error).__name__}: {error}",
            }

    def _descriptive_result(
        self,
        records: list[_FoldRecord],
        intervention_length: Any,
    ) -> CheckResult:
        """Summarize historical average differences without constructing a null."""
        table = pd.DataFrame([self._descriptive_row(record) for record in records])
        completed = table.loc[table.status == "completed"]
        count = len(completed)
        summary = {
            "n_requested": self.n_folds,
            "n_completed": count,
            "mean_difference": float(completed["mean"].mean()) if count else None,
            "mean_absolute_difference": float(completed["mean"].abs().mean())
            if count
            else None,
            "mean_interval_width": float(
                (completed.hdi_upper - completed.hdi_lower).mean()
            )
            if count
            else None,
            "intervals_excluding_zero": int(
                ((completed.hdi_lower > 0) | (completed.hdi_upper < 0)).sum()
            )
            if count
            else None,
        }
        text = (
            f"Historical placebo differences: {count} of {self.n_folds} requested fits completed. "
            "Calibration was not requested."
        )
        if count:
            text += (
                f" Mean average difference across completed fits: {summary['mean_difference']:.2f}; "
                f"mean absolute average difference: {summary['mean_absolute_difference']:.2f}; "
                f"{summary['intervals_excluding_zero']} of {count} HDIs exclude zero. "
                "These are descriptive differences, not a false-positive rate; overlapping windows and shared training histories are dependent."
            )
        else:
            text += " No completed placebo fits; differences and interval summaries are unavailable."
        return self._attach_figures(
            CheckResult(
                check_name="PlaceboInTime",
                passed=None,
                table=table,
                text=text,
                metadata={
                    "analysis_mode": "descriptive",
                    "calibration_status": "not_requested",
                    "schedule": [record.schedule() for record in records],
                    "descriptive_summary": summary,
                    "fold_results": [
                        record.result for record in records if record.result is not None
                    ],
                    "n_folds_requested": self.n_folds,
                    "n_folds_completed": count,
                    "skipped_folds": self._skipped_fold_metadata(records),
                    "intervention_length": intervention_length,
                },
            )
        )

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

        Raises
        ------
        _NullModelUnidentifiedError
            If the between-fold spread is unidentified, i.e.
            ``np.nanstd(fold_means)`` is not positive and finite. Building the
            null in that case would collapse it to a prior-driven width and
            could report a spurious verdict. A subclass of ``ValueError``.
        """
        n_folds = len(fold_means)
        fold_sds = np.where(fold_sds < 1e-6, 1e-6, fold_sds)

        prior_mu_center = float(np.nanmean(fold_means))
        prior_mu_scale = float(np.nanstd(fold_means))
        # A non-positive (or non-finite) between-fold spread means the null's
        # scale is unidentified from these folds. Silently substituting a bare
        # ``1.0`` here strips all data scaling from the ``mu`` and ``tau``
        # priors and collapses the null to a prior-driven O(1) width, which on
        # a large-scale series flips the verdict to a spurious SUPPORTED. Fail
        # loudly instead. ``run`` already abstains (INCONCLUSIVE) before
        # reaching this point when fewer than ``MIN_USABLE_FOLDS`` folds
        # complete; this guards the residual case of >= 2 folds whose
        # cumulative impacts coincide (e.g. an almost-constant series).
        if not np.isfinite(prior_mu_scale) or prior_mu_scale <= 0.0:
            raise _NullModelUnidentifiedError(
                "Cannot identify the hierarchical status-quo null: the "
                f"{n_folds} completed placebo fold(s) have no between-fold "
                "spread in their cumulative impacts (np.nanstd(fold_means) is "
                "not positive). Use more folds or a longer pre-intervention "
                "span so the placebo windows differ."
            )

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
        """Draw samples from an RVS-only expected-effect prior.

        Parameters
        ----------
        n : int
            Desired number of samples.
        unseeded_custom_priors : list[dict[str, str]], optional
            Run-local diagnostic records for legacy distributions that do not
            expose ``random_state``. With a master seed, seed-aware
            distributions receive ``.rvs(n, random_state=...)``; legacy
            distributions fall back to ``.rvs(n)`` with a warning.

        Returns
        -------
        np.ndarray
            Samples from the expected-effect prior.
        """
        prior = self.expected_effect_prior
        if prior is None:
            raise ValueError("expected_effect_prior is not set.")
        if hasattr(prior, "rvs"):
            if self.random_seed is None:
                return np.asarray(prior.rvs(n))
            if self._rvs_accepts_random_state(prior):
                return np.asarray(prior.rvs(n, random_state=self._rng_for_stage(0)))

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
                "using unseeded legacy .rvs(n). Assurance analysis is "
                "not reproducible for this custom prior; result metadata "
                "marks its type as unseeded.",
                stacklevel=2,
            )
            return np.asarray(prior.rvs(n))
        raise TypeError(
            f"expected_effect_prior must have an .rvs(n) method, got "
            f"{type(prior).__name__}."
        )

    def _compute_assurance(
        self,
        theta_new_samples: np.ndarray,
        fold_sds: np.ndarray,
        *,
        unseeded_custom_priors: list[dict[str, str]] | None = None,
    ) -> AssuranceResult:
        """Compute exact assurance rates for the learned status-quo distribution."""
        prior = self.expected_effect_prior
        if prior is None:
            raise ValueError(
                "expected_effect_prior must be set for assurance."
            )  # pragma: no cover

        if not isinstance(prior, np.ndarray) and not (
            hasattr(prior, "cdf") and hasattr(prior, "sf")
        ):
            n_prior_samples = self.n_design_replications or len(theta_new_samples)
            prior = self._draw_expected_effect_samples(
                n_prior_samples,
                unseeded_custom_priors=unseeded_custom_priors,
            )

        rope = self.rope_half_width
        if rope is None:
            raise ValueError(
                "rope_half_width must be set for assurance."
            )  # pragma: no cover

        return compute_assurance_rates(
            theta_new_samples,
            fold_sds,
            rope,
            self.threshold,
            prior,
        )

    def _execute_fold(
        self,
        record: _FoldRecord,
        data: pd.DataFrame,
        factory: Any,
    ) -> None:
        """Execute one eligible plan and record its outcome in place."""
        plan = record.plan
        if record.reason is not None:
            return
        assert plan.n_training is not None and plan.n_comparison is not None
        logger.info(
            "PlaceboInTime fold %d/%d: pseudo_treatment_time=%s",
            plan.fold,
            self.n_folds,
            plan.start,
        )
        if plan.n_training < plan.required_training:
            record.reason = "insufficient_pre_period"
            return
        fold_data = data.loc[data.index < plan.end].copy()
        if len(fold_data) < MIN_FOLD_OBSERVATIONS:
            logger.warning(
                "Fold %d has only %d observations (minimum %d), skipping.",
                plan.fold,
                len(fold_data),
                MIN_FOLD_OBSERVATIONS,
            )
            record.reason = "insufficient_fold_observations"
            return
        try:
            fold_random_seed = (
                None
                if self.random_seed is None
                else int(self.random_seed) + plan.fold - 1
            )
            if self.experiment_factory is None:
                experiment = factory(
                    fold_data, plan.start, fold_random_seed=fold_random_seed
                )
            else:
                experiment = factory(fold_data, plan.start)
            if not self.calibrate and experiment.result.impact_post.size == 0:
                raise ValueError("Historical impact draws contain no posterior samples")
            if not self.calibrate and not bool(
                np.isfinite(experiment.result.impact_post).all().values
            ):
                raise ValueError("Historical impact draws contain non-finite values")
            cumulative = self._extract_cumulative_impact(experiment)
            record.result = PlaceboFoldResult(
                fold=plan.fold,
                pseudo_treatment_time=plan.start,
                experiment=experiment,
                cumulative_impact_samples=cumulative,
                fold_mean=float(cumulative.mean().values),
                fold_sd=float(cumulative.std().values),
                average_impact_samples=None
                if self.calibrate
                else cumulative / plan.n_comparison,
            )
            record.status = "completed"
        except Exception as error:
            record.status = "failed"
            record.reason = f"{type(error).__name__}: {error}"
            logger.warning(
                "Fold %d failed to fit (pseudo_treatment_time=%s), skipping.",
                plan.fold,
                plan.start,
                exc_info=True,
            )

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext | None = None,
    ) -> CheckResult:
        """Run historical placebo refits and optional hierarchical calibration.

        Creates ``n_folds`` historical experiments at generated or explicit
        treatment times. By default, extracts posterior cumulative impact
        from each fold, fits a hierarchical Bayesian status-quo model, and
        compares the actual intervention effect against this null. With
        ``calibrate=False``, reports draw-wise average historical impacts
        and sampling diagnostics without extracting the real effect.
        When ``expected_effect_prior`` was provided at construction,
        also runs exact closed-form Bayesian assurance calculations.

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
            Calibrated results indicate whether the actual cumulative effect is clearly outside the learned null. Descriptive results have ``passed=None``, an average-difference table with one row per planned fold, schedule geometry and completed-fit aggregates, with no calibration quantities. Failed or ineligible folds keep their original positions and seeds. Descriptive summaries include weak completed fits; their diagnostics remain visible.
        """
        self.validate(experiment)
        factory = self._get_factory(context)
        treatment_time = experiment.treatment_time  # type: ignore[attr-defined]
        data = experiment.data
        intervention_length = self._compute_intervention_length(experiment)
        required_pre_period_rows = self._get_intervention_window_observation_count(
            data, treatment_time, intervention_length
        )
        comparison_window = None
        if self.calibrate:
            if required_pre_period_rows < 1:
                raise ValueError(
                    f"intervention_length={intervention_length!r} spans no "
                    f"observations at treatment_time={treatment_time!r}, so no "
                    "placebo window can be built. Pass a longer intervention_length."
                )
            comparison_window = self._describe_comparison_window(
                data, treatment_time, required_pre_period_rows
            )

        if self.treatment_times is not None:
            fold_treatment_times = list(self.treatment_times)
        elif self.selection_method == "random":
            fold_treatment_times = self._compute_random_fold_treatment_times(
                data, treatment_time, intervention_length
            )
        else:
            fold_treatment_times = self._compute_fold_treatment_times(
                treatment_time, intervention_length
            )
        records = self._plan_windows(
            data, treatment_time, intervention_length, fold_treatment_times
        )
        for record in records:
            self._execute_fold(record, data, factory)

        self._warn_skipped_folds(records)
        if not self.calibrate:
            return self._descriptive_result(records, intervention_length)
        return self._calibrated_result(
            experiment, records, intervention_length, comparison_window
        )

    @staticmethod
    def _skipped_fold_metadata(records: list[_FoldRecord]) -> list[dict[str, Any]]:
        """Derive skips in legacy order: random shortfalls, then attempted folds."""
        skipped = sorted(
            (record for record in records if record.status != "completed"),
            key=lambda record: record.plan.start is not None,
        )
        return [record.skipped_metadata() for record in skipped]

    def _warn_skipped_folds(self, records: list[_FoldRecord]) -> None:
        """Explain training, calendar and selection shortfalls from recorded outcomes."""
        insufficient_pre_period_folds = [
            record for record in records if record.reason == "insufficient_pre_period"
        ]
        calendar_mismatch_folds = [
            record for record in records if record.reason == _CALENDAR_MISMATCH_REASON
        ]
        random_selection_shortfall_folds = [
            record
            for record in records
            if record.reason == "insufficient_feasible_random_folds"
        ]
        if (
            insufficient_pre_period_folds
            or calendar_mismatch_folds
            or random_selection_shortfall_folds
        ):
            warning_parts: list[str] = []
            if insufficient_pre_period_folds:
                warning_parts.append(
                    f"{len(insufficient_pre_period_folds)} fold(s) had "
                    + (
                        "pre-treatment history below the required training minimum"
                        if self.min_training_rows is not None
                        else "pre-treatment history shorter than one full intervention window"
                    )
                )
            if calendar_mismatch_folds:
                warning_parts.append(
                    f"{len(calendar_mismatch_folds)} fold(s) had a comparison "
                    "window that does not match the complete "
                    f"comparison_frequency={self.comparison_frequency!r} grid"
                )
            if random_selection_shortfall_folds:
                warning_parts.append(
                    f"random selection yielded only "
                    f"{sum(record.plan.start is not None for record in records)} of {self.n_folds} requested "
                    "feasible fold(s) after "
                    + (
                        "eligibility and geometry constraints"
                        if self.comparison_frequency is None
                        else "eligibility, geometry and comparison_frequency="
                        f"{self.comparison_frequency!r} grid constraints"
                    )
                )
            calendar_advice = (
                ", or check that comparison_frequency matches the data index "
                "frequency with no missing dates"
                if calendar_mismatch_folds
                or (
                    random_selection_shortfall_folds
                    and self.comparison_frequency is not None
                )
                else ""
            )
            warnings.warn(
                "PlaceboInTime skipped folds because "
                + "; ".join(warning_parts)
                + ". Use fewer folds or an experiment_factory tailored to "
                "the eligible fold data"
                + calendar_advice
                + "; skipped_folds metadata records the "
                "observed and required pre-period rows.",
                stacklevel=3,
            )

    def _calibrated_result(
        self,
        experiment: BaseExperiment,
        records: list[_FoldRecord],
        intervention_length: Any,
        comparison_window: dict[str, int] | None,
    ) -> CheckResult:
        """Calibrate completed cumulative effects while retaining legacy outputs."""
        unseeded_custom_priors: list[dict[str, str]] = []
        actual_cumulative = self._extract_cumulative_impact(experiment)
        actual_cumulative_mean = float(actual_cumulative.mean().values)
        fold_results = [
            record.result for record in records if record.result is not None
        ]
        skipped_folds = self._skipped_fold_metadata(records)
        fold_summaries = [record.summary() for record in records]
        n_completed = len(fold_results)
        n_skipped = len(skipped_folds)

        # A verdict requires a hierarchical null whose between-fold spread is
        # identified.  It is not when fewer than ``MIN_USABLE_FOLDS`` folds
        # complete (``np.nanstd`` of one fold is 0), nor when the completed
        # folds have coincident cumulative impacts (``np.nanstd`` still 0).  In
        # both cases the null loses all data scaling and collapses to a
        # prior-driven width, which can flip the verdict to a spurious
        # SUPPORTED.  Abstain (INCONCLUSIVE) instead — mirroring PlaceboInSpace,
        # which returns ``passed=None`` when it lacks enough units to
        # characterise its null.  The count is checked here; the coincident-fold
        # case is detected inside ``_build_status_quo_model`` (which raises) so
        # that monkeypatched builds and direct callers stay consistent.
        fold_means = np.array([fr.fold_mean for fr in fold_results])
        fold_sds = np.array([fr.fold_sd for fr in fold_results])

        inconclusive: tuple[str, str] | None = None
        idata = None
        theta_new_samples = None
        if n_completed == 0:
            inconclusive = (
                f"Placebo-in-time analysis: 0 folds completed ({n_skipped} skipped).",
                "INCONCLUSIVE — no folds completed.",
            )
        elif n_completed < MIN_USABLE_FOLDS:
            inconclusive = (
                f"Placebo-in-time analysis: {n_completed} of {self.n_folds} "
                f"folds completed ({n_skipped} skipped).",
                f"INCONCLUSIVE — only {n_completed} usable fold; at least "
                f"{MIN_USABLE_FOLDS} are required to identify the between-fold "
                "status-quo spread. A single fold leaves the null distribution "
                "unidentified, so no verdict is issued.",
            )

        if inconclusive is None:
            try:
                idata, theta_new_samples = self._build_status_quo_model(
                    fold_means, fold_sds
                )
            except _NullModelUnidentifiedError:
                inconclusive = (
                    f"Placebo-in-time analysis: {n_completed} of "
                    f"{self.n_folds} folds completed ({n_skipped} skipped).",
                    f"INCONCLUSIVE — the {n_completed} usable folds have "
                    "coincident cumulative impacts, so the between-fold "
                    "status-quo spread is unidentified. Building a null from it "
                    "would collapse to a prior-driven width, so no verdict is "
                    "issued.",
                )

        if inconclusive is not None:
            summary, verdict = inconclusive
            parts = [summary, verdict]
            parts.extend(fold_summaries)
            return self._attach_figures(
                CheckResult(
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
            )

        # Reaching here means the null model was built successfully.
        assert idata is not None and theta_new_samples is not None

        assert actual_cumulative is not None and actual_cumulative_mean is not None
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
            "actual_cumulative_samples": np.asarray(actual_cumulative.values).ravel(),
            "actual_cumulative_mean": actual_cumulative_mean,
            "p_effect_outside_null": p_outside,
            "rope_half_width": self.rope_half_width,
            "threshold": self.threshold,
            "expected_effect_prior": self.expected_effect_prior,
            "unseeded_custom_priors": unseeded_custom_priors,
        }

        if self.expected_effect_prior is not None:
            assurance_result = self._compute_assurance(
                theta_new_samples,
                fold_sds,
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

        return self._attach_figures(
            CheckResult(
                check_name="PlaceboInTime",
                passed=passed,
                text=text,
                metadata=metadata,
            )
        )

    def _attach_figures(self, result: CheckResult) -> CheckResult:
        """Add the calibration figure to *result* when ``make_figures`` is on.

        Every path through :meth:`run` goes through here, so a caller can
        read ``figures[0]`` without first checking whether the run reached a
        verdict; an inconclusive run gets the annotated placeholder.
        """
        if self.make_figures and result.metadata.get("analysis_mode") == "descriptive":
            result.figures.append(self.plot_folds({"Placebo": result}))
        elif self.make_figures:
            result.figures.append(
                self._calibration_figure(
                    result,
                    _DEFAULT_PLOT_TITLE,
                    _DEFAULT_FIGSIZE,
                    warn_on_missing_null=False,
                )
            )
        return result

    @staticmethod
    def plot_folds(
        results: Mapping[str, CheckResult],
        *,
        title: str = "Historical placebo differences",
        figsize: tuple[float, float] = (7.0, 6.0),
    ) -> Figure:
        """Plot recorded average differences and HDIs for matching historical windows.

        Parameters
        ----------
        results : Mapping[str, CheckResult]
            Labelled descriptive results returned by :meth:`run`. Requested schedules and window geometry must match, including failed fits.
        title : str, default "Historical placebo differences"
            Figure title.
        figsize : tuple of float, default (7, 6)
            Figure size in inches.

        Returns
        -------
        matplotlib.figure.Figure
            Shared-axis comparison with a zero reference and failed-fold annotations.
        """
        if not results:
            raise ValueError(
                "plot_folds requires at least one labelled descriptive result"
            )
        frames = []
        reference = None
        date_labels = None
        for model_position, (label, result) in enumerate(results.items()):
            if (
                result.metadata.get("analysis_mode") != "descriptive"
                or result.table is None
            ):
                raise ValueError(
                    "plot_folds requires descriptive PlaceboInTime results"
                )
            schedule = pd.DataFrame(result.metadata["schedule"]).drop(columns="reason")
            if reference is None:
                reference = schedule
                date_labels = [
                    str(value.date()) if hasattr(value, "date") else str(value)
                    for value in schedule.pseudo_treatment_time
                ]
            elif not reference.equals(schedule):
                raise ValueError(
                    "Paired results require matching requested schedules and windows"
                )
            frame = result.table.copy()
            frame["model"] = label
            frame["position"] = -frame["fold"] + (
                model_position - (len(results) - 1) / 2
            ) * 0.6 / len(results)
            frame["annotation"] = [
                f"{label}: {status}" if status != "completed" else ""
                for status in frame.status
            ]
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        completed = frame.loc[frame.status == "completed"]
        missing = frame.loc[frame.status != "completed"]
        assert reference is not None and date_labels is not None
        cycle = cast(Any, mpl.rcParams["axes.prop_cycle"]).by_key().get("color")
        if not cycle:
            cycle = cast(Any, mpl.rcParamsDefault["axes.prop_cycle"]).by_key()["color"]
        model_colors = {
            label: to_hex(cycle[position % len(cycle)])
            for position, label in enumerate(results)
        }
        plot = (
            ggplot(frame)
            + geom_vline(xintercept=0, linetype="dashed", colour="black")
            + geom_segment(
                data=completed,
                mapping=aes(
                    x="hdi_lower",
                    xend="hdi_upper",
                    y="position",
                    yend="position",
                    colour="model",
                ),
            )
            + geom_point(
                data=completed, mapping=aes(x="mean", y="position", colour="model")
            )
            + geom_text(
                data=missing,
                mapping=aes(x=0, y="position", label="annotation", colour="model"),
                size=8,
                ha="left",
                show_legend=False,
            )
            + scale_y_continuous(
                breaks=[-int(fold) for fold in reference.fold], labels=date_labels
            )
            + scale_colour_manual(values=model_colors)
            + labs(
                x="Average difference (outcome units)",
                y="Requested placebo start",
                colour="Model",
                caption="Recorded expected-path HDIs; dependent historical windows.",
            )
        )
        if completed.empty:
            plot += labs(subtitle="No completed placebo fits")
        return PlaceboInTime._draw(plot, title, figsize)

    @staticmethod
    def plot_calibration(
        check_result: CheckResult,
        title: str = _DEFAULT_PLOT_TITLE,
        figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    ) -> Figure:
        """Plot the three-panel calibration diagnostic for a placebo run.

        Panel A shows the per-fold cumulative impact posteriors, panel B the
        hierarchical status-quo null pooled from them, and panel C the null
        against the actual effect.

        Parameters
        ----------
        check_result : CheckResult
            Result returned by :meth:`run`.  Everything the panels need is
            read from its ``metadata``.
        title : str, default "Placebo-in-Time calibration"
            Figure suptitle.
        figsize : tuple of float, default (7, 9)
            Size of the drawn figure, in inches.

        Returns
        -------
        matplotlib.figure.Figure
            The drawn composition.  When the run produced no null model, a
            single annotated panel is returned instead and a warning is
            emitted.
        """
        return PlaceboInTime._calibration_figure(
            check_result, title, figsize, warn_on_missing_null=True
        )

    @staticmethod
    def _calibration_figure(
        check_result: CheckResult,
        title: str,
        figsize: tuple[float, float],
        warn_on_missing_null: bool,
    ) -> Figure:
        """Build the calibration figure, optionally warning on a missing null.

        :meth:`run` suppresses the warning because it already reports the
        same condition through ``passed=None`` and the result text; a direct
        call to :meth:`plot_calibration` has no such context and gets it.
        """
        metadata = check_result.metadata
        if metadata.get("calibration_status") == "not_requested":
            raise ValueError(
                "Cannot plot calibration: calibration was not requested for this descriptive result"
            )
        fold_results = metadata["fold_results"]

        if "null_samples" not in metadata:
            return PlaceboInTime._plot_missing_null(
                fold_results, title, figsize, warn_on_missing_null
            )

        null_samples = np.asarray(metadata["null_samples"]).ravel()
        actual_samples = np.asarray(metadata["actual_cumulative_samples"]).ravel()
        null_mean = float(np.mean(null_samples))

        fold_labels = [
            f"Fold {fold_result.fold} "
            f"(t*={PlaceboInTime._format_fold_time(fold_result.pseudo_treatment_time)})"
            for fold_result in fold_results
        ]
        fold_samples = [
            np.asarray(fold_result.cumulative_impact_samples.values).ravel()
            for fold_result in fold_results
        ]
        fold_frame = pd.DataFrame(
            {
                "cumulative_impact": np.concatenate(fold_samples),
                "fold": pd.Categorical(
                    np.repeat(fold_labels, [len(s) for s in fold_samples]),
                    categories=fold_labels,
                ),
            }
        )
        fold_mean_frame = pd.DataFrame(
            {
                "fold": pd.Categorical(fold_labels, categories=fold_labels),
                "fold_mean": [fold_result.fold_mean for fold_result in fold_results],
            }
        )
        # The palette follows the active matplotlib cycle so the panels keep
        # their colours if a caller restyles the surrounding report.
        cycle = cast(Any, mpl.rcParams["axes.prop_cycle"]).by_key()["color"]
        fold_colors = [to_hex(cycle[i % len(cycle)]) for i in range(len(fold_labels))]

        panel_a = (
            ggplot(fold_frame, aes("cumulative_impact"))
            + geom_histogram(
                aes(y=after_stat("density"), fill="fold"),
                bins=40,
                alpha=0.45,
                position="identity",
            )
            + geom_vline(
                data=fold_mean_frame,
                mapping=aes(xintercept="fold_mean", colour="fold"),
                linetype="dashed",
                show_legend=False,
            )
            + geom_vline(xintercept=0, linetype="dotted", alpha=0.5)
            + scale_fill_manual(values=fold_colors)
            + scale_colour_manual(values=fold_colors)
            + labs(
                x="Cumulative impact",
                y="Density",
                fill="",
                title="A. Placebo fold distributions",
            )
        )

        panel_b = (
            ggplot(pd.DataFrame({"cumulative_impact": null_samples}))
            + geom_histogram(
                aes("cumulative_impact", after_stat("density")),
                bins=50,
                fill="#94a3b8",
                alpha=0.5,
            )
            + geom_vline(xintercept=0, linetype="dotted", alpha=0.5)
            + geom_vline(xintercept=null_mean, colour="#64748b", linetype="dashed")
            + labs(
                x="Cumulative impact",
                y="Density",
                title=f"B. Learned null distribution (mean = {null_mean:.1f})",
            )
        )

        sources = ["Null (status quo)", "Actual effect"]
        comparison_frame = pd.DataFrame(
            {
                "cumulative_impact": np.concatenate([null_samples, actual_samples]),
                "distribution": pd.Categorical(
                    np.repeat(sources, [null_samples.size, actual_samples.size]),
                    categories=sources,
                ),
            }
        )
        panel_c = (
            ggplot(comparison_frame, aes("cumulative_impact"))
            + geom_histogram(
                aes(y=after_stat("density"), fill="distribution"),
                bins=50,
                alpha=0.4,
                position="identity",
            )
            + scale_fill_manual(values=["#94a3b8", "#E24A33"])
            + labs(
                x="Cumulative impact",
                y="Density",
                fill="",
                title=(
                    "C. Actual effect vs null "
                    f"($p_{{cal}}$ = {metadata['p_effect_outside_null']:.3f})"
                ),
            )
        )

        return PlaceboInTime._draw(panel_a / panel_b / panel_c, title, figsize)

    @staticmethod
    def _format_fold_time(pseudo_treatment_time: Any) -> str:
        """Format a pseudo treatment time for a fold label."""
        if hasattr(pseudo_treatment_time, "strftime"):
            return f"{pseudo_treatment_time:%Y}"
        return f"{pseudo_treatment_time}"

    @staticmethod
    def _draw(plot: Any, title: str, figsize: tuple[float, float]) -> Figure:
        """Draw a plotnine plot or composition and stamp the suptitle on it."""
        return draw_figure(plot, title, figsize)

    @staticmethod
    def _plot_missing_null(
        fold_results: list[PlaceboFoldResult],
        title: str,
        figsize: tuple[float, float],
        warn: bool,
    ) -> Figure:
        """Return an annotated placeholder when no null model was built."""
        if warn:
            warnings.warn(
                f"Not enough folds completed to build a null model "
                f"({len(fold_results)} completed), so the calibration panels "
                f"cannot be drawn.",
                UserWarning,
                stacklevel=4,
            )
        lines = [f"No null model: {len(fold_results)} folds completed."]
        lines.extend(
            f"Fold {fold_result.fold}: mean={fold_result.fold_mean:.2f}, "
            f"sd={fold_result.fold_sd:.2f}"
            for fold_result in fold_results
        )
        placeholder = (
            ggplot(pd.DataFrame({"x": [0.0], "y": [0.0], "label": ["\n".join(lines)]}))
            + geom_text(aes("x", "y", label="label"))
            + theme_void()
        )
        return PlaceboInTime._draw(placeholder, title, figsize)

    def __repr__(self) -> str:
        """Return a string representation of the check."""
        parts = [f"n_folds={self.n_folds}"]
        if not self.calibrate:
            parts.append("calibrate=False")
        if self.treatment_times is not None:
            parts.append(f"treatment_times={self.treatment_times!r}")
        if self.intervention_length is not None:
            parts.append(f"intervention_length={self.intervention_length!r}")
        if self.selection_method != "sequential":
            parts.append(f"selection_method={self.selection_method!r}")
        if self.allow_overlap:
            parts.append("allow_overlap=True")
        if self.expected_effect_prior is not None:
            parts.append("assurance=True")
        if not self.make_figures:
            parts.append("make_figures=False")
        return f"PlaceboInTime({', '.join(parts)})"
