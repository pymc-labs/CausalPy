# Copyright 2022 - 2026 The PyMC Labs Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tables and plots for canonical planning results.

This module reads :class:`~causalpy.planning.contracts.PlanningResult` and
:class:`~causalpy.planning.contracts.CanonicalEvaluation` only. It does not
import an estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from causalpy._arviz_compat import hdi_bounds
from causalpy.constants import HDI_PROB
from causalpy.planning.contracts import CanonicalEvaluation, PlanningResult

EVALUATION_COLUMNS: tuple[str, ...] = (
    "estimator",
    "duration",
    "effect_scale",
    "effect_magnitude",
    "cumulative_total_mean",
    "average_per_period_mean",
    "average_per_treated_unit_mean",
    "relative_total_mean",
    "posterior_tail_probability",
    "decided",
    "hdi_covers_threshold",
    "repeated_experiment_power",
    "historical_detection_share",
)


def evaluation_table(result: PlanningResult) -> pd.DataFrame:
    """Return one row per evaluation.

    ``repeated_experiment_power`` and ``historical_detection_share`` come from
    the planning result, not from the posterior tail. On the reference path
    both are ``None``.

    Parameters
    ----------
    result : PlanningResult
        Canonical evaluations from a history-only plan.

    Returns
    -------
    pandas.DataFrame
        One row per duration and effect, with the columns in
        :data:`EVALUATION_COLUMNS`.
    """
    rows = [
        {
            "estimator": evaluation.estimator,
            "duration": evaluation.duration,
            "effect_scale": evaluation.scenario.scale,
            "effect_magnitude": evaluation.scenario.magnitude,
            "cumulative_total_mean": evaluation.summary.cumulative_total_mean,
            "average_per_period_mean": evaluation.summary.average_per_period_mean,
            "average_per_treated_unit_mean": (
                evaluation.summary.average_per_treated_unit_mean
            ),
            "relative_total_mean": evaluation.summary.relative_total_mean,
            "posterior_tail_probability": (
                evaluation.decision.posterior_tail_probability
            ),
            "decided": evaluation.decision.decided,
            "hdi_covers_threshold": evaluation.decision.hdi_covers_threshold,
            "repeated_experiment_power": result.repeated_experiment_power,
            "historical_detection_share": result.historical_detection_share,
        }
        for evaluation in result.evaluations
    ]
    return pd.DataFrame(rows, columns=list(EVALUATION_COLUMNS))


def plot_evaluation(evaluation: CanonicalEvaluation) -> Figure:
    """Plot the posterior conditional-mean impact path and its HDI.

    A single treated unit is drawn on its own outcome scale. Several units
    are summed within each draw before the mean and interval are taken.
    The figure is not a launch recommendation.

    Parameters
    ----------
    evaluation : CanonicalEvaluation
        One translated horizon. The plot reads its draws and does not fit.

    Returns
    -------
    matplotlib.figure.Figure
        The impact path. The caller owns the figure and should close it.
    """
    import matplotlib.pyplot as plt

    impact = evaluation.draws.unit_impact
    path = (
        impact.isel(treated_units=0)
        if impact.sizes["treated_units"] == 1
        else impact.sum(dim="treated_units")
    )
    mean = path.mean(dim=("chain", "draw"))
    lower: list[float] = []
    upper: list[float] = []
    for position in range(path.sizes["obs_ind"]):
        lo, hi = hdi_bounds(
            path.isel(obs_ind=position),
            prob=HDI_PROB,
            dim=("chain", "draw"),
        )
        lower.append(float(lo))
        upper.append(float(hi))
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(mean.sizes["obs_ind"])
    ax.fill_between(x, lower, upper, alpha=0.25, label=f"{HDI_PROB:.0%} HDI")
    ax.plot(x, np.asarray(mean.values), label="posterior conditional mean")
    ax.set_xlabel("holdout period")
    ax.set_ylabel(evaluation.outcome_scale)
    ax.set_title(
        f"{evaluation.estimator}: duration {evaluation.duration}, "
        f"{evaluation.scenario.scale} effect {evaluation.scenario.magnitude}"
    )
    ax.legend()
    return fig
