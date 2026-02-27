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
Bandwidth sensitivity check for Regression Discontinuity / Kink designs.

Re-fits the experiment with multiple bandwidth values and compares
effect estimates to assess sensitivity to the bandwidth choice.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from causalpy.checks.base import CheckResult, clone_model
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.regression_discontinuity import RegressionDiscontinuity
from causalpy.experiments.regression_kink import RegressionKink
from causalpy.pipeline import PipelineContext

logger = logging.getLogger(__name__)


class BandwidthSensitivity:
    """Re-fit with multiple bandwidths and compare effect estimates.

    Parameters
    ----------
    bandwidths : list of float
        Bandwidth values to test.  ``np.inf`` means no bandwidth restriction.

    Examples
    --------
    >>> import causalpy as cp  # doctest: +SKIP
    >>> check = cp.checks.BandwidthSensitivity(  # doctest: +SKIP
    ...     bandwidths=[0.5, 1.0, 2.0, np.inf]
    ... )
    """

    applicable_methods: set[type[BaseExperiment]] = {
        RegressionDiscontinuity,
        RegressionKink,
    }

    def __init__(self, bandwidths: list[float] | None = None) -> None:
        self.bandwidths = bandwidths or [0.25, 0.5, 1.0, 2.0, np.inf]

    def validate(self, experiment: BaseExperiment) -> None:
        """Verify the experiment is an RD or RKink instance."""
        if not isinstance(experiment, (RegressionDiscontinuity, RegressionKink)):
            raise TypeError(
                "BandwidthSensitivity requires a RegressionDiscontinuity "
                "or RegressionKink experiment."
            )

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext,
    ) -> CheckResult:
        """Re-fit the experiment at multiple bandwidths and compare estimates."""
        if context.experiment_config is None:
            raise RuntimeError(
                "No experiment_config in context. Use EstimateEffect "
                "before SensitivityAnalysis."
            )

        method = context.experiment_config["method"]
        base_kwargs = {
            k: v
            for k, v in context.experiment_config.items()
            if k not in ("method", "bandwidth")
        }

        rows: list[dict[str, Any]] = []
        for bw in self.bandwidths:
            logger.info("BandwidthSensitivity: fitting with bandwidth=%s", bw)
            kw = dict(base_kwargs)
            kw["bandwidth"] = bw
            if "model" in kw and kw["model"] is not None:
                kw["model"] = clone_model(kw["model"])

            try:
                alt_experiment = method(context.data, **kw)
                summary = alt_experiment.effect_summary()
                row: dict[str, Any] = {"bandwidth": bw}
                if summary.table is not None and not summary.table.empty:
                    for col in summary.table.columns:
                        row[col] = summary.table[col].iloc[0]
                rows.append(row)
            except Exception as exc:
                logger.warning(
                    "BandwidthSensitivity: failed for bandwidth=%s: %s",
                    bw,
                    exc,
                )
                rows.append({"bandwidth": bw, "error": str(exc)})

        table = pd.DataFrame(rows) if rows else None

        text = (
            f"Bandwidth sensitivity analysis: compared {len(self.bandwidths)} "
            f"bandwidth values. Examine the table for consistency of effect "
            f"estimates across bandwidths."
        )

        return CheckResult(
            check_name="BandwidthSensitivity",
            passed=None,
            table=table,
            text=text,
        )
