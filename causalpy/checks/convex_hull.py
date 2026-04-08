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
Convex hull diagnostic check for Synthetic Control experiments.

Verifies that pre-treatment values of treated units fall within
the range of control units — a key assumption of the synthetic
control method.
"""

from __future__ import annotations

import pandas as pd

from causalpy.checks.base import CheckResult
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pipeline import PipelineContext
from causalpy.utils import check_convex_hull_violation


class ConvexHullCheck:
    """Check that treated unit values lie within the convex hull of controls.

    Wraps the existing ``SyntheticControl._check_convex_hull()`` logic.
    """

    applicable_methods: set[type[BaseExperiment]] = {SyntheticControl}

    def validate(self, experiment: BaseExperiment) -> None:
        """Verify the experiment is a SyntheticControl instance."""
        if not isinstance(experiment, SyntheticControl):
            raise TypeError("ConvexHullCheck requires a SyntheticControl experiment.")

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext,
    ) -> CheckResult:
        """Run the convex hull violation check on pre-treatment data."""
        sc = experiment
        datapre_control = sc.datapre_control  # type: ignore[attr-defined]
        datapre_treated = sc.datapre_treated  # type: ignore[attr-defined]

        all_results = []
        total_violations = 0
        all_pass = True

        for unit_idx in range(datapre_treated.shape[1]):
            treated_series = datapre_treated[:, unit_idx]
            result = check_convex_hull_violation(treated_series, datapre_control)
            all_results.append(result)
            total_violations += result["n_violations"]
            if not result["passes"]:
                all_pass = False

        rows = []
        treated_units = getattr(
            sc, "treated_units", [f"unit_{i}" for i in range(len(all_results))]
        )  # type: ignore[attr-defined]
        for unit_name, res in zip(treated_units, all_results, strict=True):
            rows.append(
                {
                    "treated_unit": unit_name,
                    "passes": res["passes"],
                    "n_violations": res["n_violations"],
                    "pct_above": res["pct_above"],
                    "pct_below": res["pct_below"],
                }
            )

        table = pd.DataFrame(rows) if rows else None

        if all_pass:
            text = (
                "Convex hull check passed: all treated unit values lie "
                "within the range of control units in the pre-treatment period."
            )
        else:
            text = (
                f"Convex hull check failed: {total_violations} violations "
                f"detected. Some treated unit values fall outside the range "
                f"of control units, which may compromise the synthetic "
                f"control fit."
            )

        return CheckResult(
            check_name="ConvexHullCheck",
            passed=all_pass,
            table=table,
            text=text,
        )
