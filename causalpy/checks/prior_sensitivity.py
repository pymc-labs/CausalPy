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
Prior sensitivity check for Bayesian causal inference experiments.

Re-fits the experiment with alternative prior specifications and
compares posterior estimates to assess how sensitive the conclusions
are to prior choices.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import pandas as pd

from causalpy.checks.base import CheckResult
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.diff_in_diff import DifferenceInDifferences
from causalpy.experiments.instrumental_variable import InstrumentalVariable
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.experiments.inverse_propensity_weighting import (
    InversePropensityWeighting,
)
from causalpy.experiments.prepostnegd import PrePostNEGD
from causalpy.experiments.regression_discontinuity import RegressionDiscontinuity
from causalpy.experiments.regression_kink import RegressionKink
from causalpy.experiments.staggered_did import StaggeredDifferenceInDifferences
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pipeline import PipelineContext
from causalpy.pymc_models import PyMCModel

logger = logging.getLogger(__name__)


class PriorSensitivity:
    """Re-fit the experiment with alternative models/priors and compare.

    Each alternative is specified as a dict with ``"name"`` and ``"model"``
    keys.  The check re-instantiates the experiment for each alternative
    model and compares the resulting effect summaries.

    Parameters
    ----------
    alternatives : list of dict
        Each dict must have ``"name"`` (str) and ``"model"`` (PyMCModel
        or RegressorMixin) keys.

    Examples
    --------
    >>> import causalpy as cp  # doctest: +SKIP
    >>> check = cp.checks.PriorSensitivity(  # doctest: +SKIP
    ...     alternatives=[
    ...         {"name": "diffuse", "model": cp.pymc_models.LinearRegression(...)},
    ...         {"name": "tight", "model": cp.pymc_models.LinearRegression(...)},
    ...     ]
    ... )
    """

    applicable_methods: set[type[BaseExperiment]] = {
        InterruptedTimeSeries,
        DifferenceInDifferences,
        SyntheticControl,
        StaggeredDifferenceInDifferences,
        RegressionDiscontinuity,
        RegressionKink,
        PrePostNEGD,
        InversePropensityWeighting,
        InstrumentalVariable,
    }

    def __init__(self, alternatives: list[dict[str, Any]]) -> None:
        if not alternatives:
            raise ValueError("alternatives must be a non-empty list")
        for i, alt in enumerate(alternatives):
            if "name" not in alt or "model" not in alt:
                raise ValueError(
                    f"Alternative {i} must have 'name' and 'model' keys, "
                    f"got keys: {list(alt.keys())}"
                )
        self.alternatives = alternatives

    def validate(self, experiment: BaseExperiment) -> None:
        """Verify the experiment uses a Bayesian (PyMC) model."""
        if not isinstance(experiment.model, PyMCModel):
            raise TypeError(
                "PriorSensitivity requires a Bayesian (PyMC) model. "
                f"Got {type(experiment.model).__name__}."
            )

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext,
    ) -> CheckResult:
        """Re-fit with each alternative model and compare effect estimates."""
        if context.experiment_config is None:
            raise RuntimeError(
                "No experiment_config in context. Use EstimateEffect "
                "before SensitivityAnalysis."
            )

        method = context.experiment_config["method"]
        base_kwargs = {
            k: v
            for k, v in context.experiment_config.items()
            if k not in ("method", "model")
        }

        rows: list[dict[str, Any]] = []

        for alt in self.alternatives:
            name = alt["name"]
            model = copy.deepcopy(alt["model"])
            logger.info("PriorSensitivity: fitting with '%s'", name)

            alt_experiment = method(context.data, model=model, **base_kwargs)

            try:
                summary = alt_experiment.effect_summary()
                row: dict[str, Any] = {"prior_spec": name}
                if summary.table is not None and not summary.table.empty:
                    for col in summary.table.columns:
                        row[col] = summary.table[col].iloc[0]
                rows.append(row)
            except (NotImplementedError, Exception) as exc:
                logger.warning(
                    "PriorSensitivity: effect_summary() failed for '%s': %s",
                    name,
                    exc,
                )
                rows.append({"prior_spec": name, "error": str(exc)})

        table = pd.DataFrame(rows) if rows else None

        text = (
            f"Prior sensitivity analysis: compared {len(self.alternatives)} "
            f"alternative prior specifications."
        )

        return CheckResult(
            check_name="PriorSensitivity",
            passed=None,
            table=table,
            text=text,
        )
