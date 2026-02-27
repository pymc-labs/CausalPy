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
Leave-one-out sensitivity check for Synthetic Control experiments.

Drops each control unit one at a time, refits, and assesses how
much the effect estimate changes.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from causalpy.checks.base import CheckResult, clone_model
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pipeline import PipelineContext

logger = logging.getLogger(__name__)


class LeaveOneOut:
    """Drop each control unit, refit, and compare effect estimates.

    Assesses how sensitive the synthetic control weights and effect
    estimates are to individual donor units.

    Examples
    --------
    >>> import causalpy as cp  # doctest: +SKIP
    >>> check = cp.checks.LeaveOneOut()  # doctest: +SKIP
    """

    applicable_methods: set[type[BaseExperiment]] = {SyntheticControl}

    def validate(self, experiment: BaseExperiment) -> None:
        """Verify the experiment is a SyntheticControl instance."""
        if not isinstance(experiment, SyntheticControl):
            raise TypeError("LeaveOneOut requires a SyntheticControl experiment.")

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext,
    ) -> CheckResult:
        """Drop each control unit in turn and compare effect estimates."""
        if context.experiment_config is None:
            raise RuntimeError(
                "No experiment_config in context. Use EstimateEffect "
                "before SensitivityAnalysis."
            )

        method = context.experiment_config["method"]
        base_kwargs = {
            k: v
            for k, v in context.experiment_config.items()
            if k not in ("method", "control_units")
        }
        all_controls: list[str] = context.experiment_config["control_units"]

        if len(all_controls) < 2:
            return CheckResult(
                check_name="LeaveOneOut",
                passed=None,
                text="Cannot run leave-one-out with fewer than 2 control units.",
            )

        rows: list[dict[str, Any]] = []
        for dropped in all_controls:
            remaining = [c for c in all_controls if c != dropped]
            logger.info("LeaveOneOut: dropping '%s'", dropped)

            kw = dict(base_kwargs)
            kw["control_units"] = remaining
            if "model" in kw and kw["model"] is not None:
                kw["model"] = clone_model(kw["model"])

            try:
                alt_experiment = method(context.data, **kw)
                summary = alt_experiment.effect_summary()
                row: dict[str, Any] = {"dropped_unit": dropped}
                if summary.table is not None and not summary.table.empty:
                    for col in summary.table.columns:
                        row[col] = summary.table[col].iloc[0]
                rows.append(row)
            except Exception as exc:
                logger.warning(
                    "LeaveOneOut: failed when dropping '%s': %s",
                    dropped,
                    exc,
                )
                rows.append({"dropped_unit": dropped, "error": str(exc)})

        table = pd.DataFrame(rows) if rows else None

        text = (
            f"Leave-one-out analysis: dropped each of {len(all_controls)} "
            f"control units. Examine the table for consistency of effect "
            f"estimates."
        )

        return CheckResult(
            check_name="LeaveOneOut",
            passed=None,
            table=table,
            text=text,
        )
