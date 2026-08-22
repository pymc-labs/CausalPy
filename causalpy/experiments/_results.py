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
"""Per-experiment result bundles for the lazy experiment lifecycle.

Every bundle is *fully populated* whenever it exists: an experiment either has
no bundle yet (the corresponding lifecycle verb has not run) or a complete one.
Nothing derived from model draws lives directly on the experiment object; the
two public slots ``experiment.result`` (posterior group) and
``experiment.prior_result`` (prior group) hold these bundles and back the
``is_fitted`` / ``has_prior_predictive`` state predicates.

All prediction/impact fields are typed :class:`xarray.DataArray` on the
canonical ``("chain", "draw", "obs_ind"[, ...])`` dimensions produced by
:class:`~causalpy.experiments.model_adapter.ModelAdapter.predict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import xarray as xr

__all__ = [
    "CausalResult",
    "CoefficientResult",
    "DiscontinuityResult",
    "GroupComparisonScenario",
    "KinkResult",
    "StaggeredDifferenceInDifferencesResult",
    "SyntheticDifferenceInDifferencesResult",
]


@dataclass(frozen=True)
class CausalResult:
    """Result bundle for predict-contrast experiments.

    Used by :class:`~causalpy.experiments.interrupted_time_series.InterruptedTimeSeries`,
    :class:`~causalpy.experiments.synthetic_control.SyntheticControl`, and
    :class:`~causalpy.experiments.piecewise_its.PiecewiseITS`.

    For ``PiecewiseITS`` ``predictions_pre`` carries the fitted expectation over
    the full observation window and ``predictions_post`` / ``impact_post`` /
    ``impact_post_cumulative`` carry the post-first-interruption slices consumed
    by the reporting helpers.
    """

    predictions_pre: xr.DataArray
    predictions_post: xr.DataArray
    impact_pre: xr.DataArray
    impact_post: xr.DataArray
    impact_post_cumulative: xr.DataArray
    score: pd.Series | None = None


@dataclass(frozen=True)
class SyntheticDifferenceInDifferencesResult(CausalResult):
    """Result bundle for :class:`~causalpy.experiments.synthetic_difference_in_differences.SyntheticDifferenceInDifferences`.

    The ``CausalResult`` prediction/impact fields are reconstructed from the
    synthetic-control imputation; ``tau_posterior`` carries the analytic
    double-difference treatment-effect draws with dimensions
    ``("chain", "draw")``.
    """

    tau_posterior: xr.DataArray = field(kw_only=True)


@dataclass(frozen=True)
class GroupComparisonScenario:
    """One scenario plotted or summarized by a group-comparison experiment."""

    inputs: pd.DataFrame
    prediction: xr.DataArray


@dataclass(frozen=True)
class CoefficientResult:
    """Result bundle for coefficient-contrast experiments (DiD, PrePostNEGD).

    ``causal_impact`` holds draws of the treatment-effect coefficient (or its
    algebraically-equivalent prediction contrast) with canonical coefficient
    dimensions.
    """

    causal_impact: xr.DataArray
    scenario_control: GroupComparisonScenario
    scenario_treated: GroupComparisonScenario
    scenario_counterfactual: GroupComparisonScenario | None = None
    score: pd.Series | None = None


@dataclass(frozen=True)
class DiscontinuityResult:
    """Result bundle for :class:`~causalpy.experiments.regression_discontinuity.RegressionDiscontinuity`."""

    predictions: xr.DataArray
    discontinuity_at_threshold: xr.DataArray
    score: pd.Series | None = None


@dataclass(frozen=True)
class KinkResult:
    """Result bundle for :class:`~causalpy.experiments.regression_kink.RegressionKink`."""

    predictions: xr.DataArray
    gradient_change: xr.DataArray
    score: pd.Series | None = None


@dataclass(frozen=True)
class StaggeredDifferenceInDifferencesResult:
    """Result bundle for :class:`~causalpy.experiments.staggered_did.StaggeredDifferenceInDifferences`.

    ``att_group_time`` and ``att_event_time`` are aggregated ATT tables;
    ``y_pred`` retains the raw counterfactual draws so placebos and alternate
    HDI levels can re-derive effects without resampling.
    """

    att_group_time: pd.DataFrame
    att_event_time: pd.DataFrame
    y_pred: xr.DataArray
    hdi_prob: float
    score: pd.Series | None = None
