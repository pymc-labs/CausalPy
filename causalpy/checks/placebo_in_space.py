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
Placebo-in-space sensitivity check for Synthetic Control experiments.

Treats each control unit as if it were the treated unit (while excluding
the actual treated unit from the donor pool) and checks whether
spurious effects appear.
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


class PlaceboInSpace:
    """Treat each control unit as if treated and check for spurious effects.

    For each control unit, re-fits the synthetic control using the
    remaining controls as donors.  If the placebo effects are as large
    as the actual effect, the causal claim is weakened.

    Examples
    --------
    >>> import causalpy as cp  # doctest: +SKIP
    >>> check = cp.checks.PlaceboInSpace()  # doctest: +SKIP
    """

    applicable_methods: set[type[BaseExperiment]] = {SyntheticControl}

    def validate(self, experiment: BaseExperiment) -> None:
        """Verify the experiment is a SyntheticControl instance."""
        if not isinstance(experiment, SyntheticControl):
            raise TypeError("PlaceboInSpace requires a SyntheticControl experiment.")

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext,
    ) -> CheckResult:
        """Treat each control unit as treated and compare effect magnitudes."""
        if context.experiment_config is None:
            raise RuntimeError(
                "No experiment_config in context. Use EstimateEffect "
                "before SensitivityAnalysis."
            )

        method = context.experiment_config["method"]
        base_kwargs = {
            k: v
            for k, v in context.experiment_config.items()
            if k not in ("method", "control_units", "treated_units")
        }
        all_controls: list[str] = context.experiment_config["control_units"]
        actual_treated: list[str] = context.experiment_config["treated_units"]

        if len(all_controls) < 2:
            return CheckResult(
                check_name="PlaceboInSpace",
                passed=None,
                text="Cannot run placebo-in-space with fewer than 2 control units.",
            )

        rows: list[dict[str, Any]] = []
        for placebo_treated in all_controls:
            donors = [
                c
                for c in all_controls
                if c != placebo_treated and c not in actual_treated
            ]

            if len(donors) < 1:
                logger.warning(
                    "PlaceboInSpace: not enough donors when treating '%s'",
                    placebo_treated,
                )
                continue

            logger.info("PlaceboInSpace: treating '%s' as treated", placebo_treated)

            kw = dict(base_kwargs)
            kw["control_units"] = donors
            kw["treated_units"] = [placebo_treated]
            if "model" in kw and kw["model"] is not None:
                kw["model"] = clone_model(kw["model"])

            try:
                alt_experiment = method(context.data, **kw)
                summary = alt_experiment.effect_summary()
                row: dict[str, Any] = {"placebo_treated": placebo_treated}
                if summary.table is not None and not summary.table.empty:
                    for col in summary.table.columns:
                        row[col] = summary.table[col].iloc[0]
                rows.append(row)
            except Exception as exc:
                logger.warning(
                    "PlaceboInSpace: failed for '%s': %s",
                    placebo_treated,
                    exc,
                )
                rows.append({"placebo_treated": placebo_treated, "error": str(exc)})

        table = pd.DataFrame(rows) if rows else None

        text = (
            f"Placebo-in-space analysis: tested {len(all_controls)} control "
            f"units as placebo treated units. If placebo effects are "
            f"comparable to the actual effect, the causal claim may be "
            f"weakened."
        )

        return CheckResult(
            check_name="PlaceboInSpace",
            passed=None,
            table=table,
            text=text,
        )
