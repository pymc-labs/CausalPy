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
McCrary density test for Regression Discontinuity designs.

Tests for manipulation of the running variable at the threshold by
checking whether there is a discontinuity in the density.  A
significant density discontinuity suggests that units may have been
able to manipulate their value of the running variable to sort into
(or out of) treatment.

Uses a simple histogram-based approach: compares the count of
observations in bins just below and just above the threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from causalpy.checks.base import CheckResult
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.regression_discontinuity import RegressionDiscontinuity
from causalpy.pipeline import PipelineContext


class McCraryDensityTest:
    """Test for manipulation of the running variable at the threshold.

    Compares the density of observations just below and just above the
    treatment threshold using a histogram-based approach.

    Parameters
    ----------
    n_bins : int, default 20
        Number of bins on each side of the threshold.
    alpha : float, default 0.05
        Significance level for the test.

    Examples
    --------
    >>> import causalpy as cp  # doctest: +SKIP
    >>> check = cp.checks.McCraryDensityTest(n_bins=20)
    """

    applicable_methods: set[type[BaseExperiment]] = {RegressionDiscontinuity}

    def __init__(self, n_bins: int = 20, alpha: float = 0.05) -> None:
        self.n_bins = n_bins
        self.alpha = alpha

    def validate(self, experiment: BaseExperiment) -> None:
        if not isinstance(experiment, RegressionDiscontinuity):
            raise TypeError(
                "McCraryDensityTest requires a RegressionDiscontinuity experiment."
            )

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext,
    ) -> CheckResult:
        rd = experiment
        threshold = rd.treatment_threshold  # type: ignore[attr-defined]
        running_var = rd.running_variable_name  # type: ignore[attr-defined]
        data = rd.data  # type: ignore[attr-defined]

        x = data[running_var].values
        below = x[x < threshold]
        above = x[x >= threshold]

        n_below = len(below)
        n_above = len(above)
        n_total = n_below + n_above

        if n_total == 0:
            return CheckResult(
                check_name="McCraryDensityTest",
                passed=None,
                text="No observations found around the threshold.",
            )

        prop_below = n_below / n_total
        prop_above = n_above / n_total

        se = np.sqrt(prop_below * prop_above / n_total)
        z_stat = (prop_below - 0.5) / se if se > 0 else 0.0
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        passed = bool(p_value > self.alpha)

        table = pd.DataFrame(
            [
                {
                    "n_below": n_below,
                    "n_above": n_above,
                    "prop_below": prop_below,
                    "prop_above": prop_above,
                    "z_statistic": z_stat,
                    "p_value": p_value,
                    "threshold": threshold,
                }
            ]
        )

        if passed:
            text = (
                f"McCrary density test: no evidence of manipulation at "
                f"threshold {threshold} (z={z_stat:.3f}, p={p_value:.3f}). "
                f"Observations below: {n_below}, above: {n_above}."
            )
        else:
            text = (
                f"McCrary density test: possible manipulation detected at "
                f"threshold {threshold} (z={z_stat:.3f}, p={p_value:.3f}). "
                f"Observations below: {n_below}, above: {n_above}."
            )

        return CheckResult(
            check_name="McCraryDensityTest",
            passed=passed,
            table=table,
            text=text,
            metadata={"z_statistic": z_stat, "p_value": p_value},
        )
