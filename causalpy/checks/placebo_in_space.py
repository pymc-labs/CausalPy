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
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import xarray as xr

from causalpy.checks.base import CheckResult, clone_model
from causalpy.experiments._results import CausalResult
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pipeline import PipelineContext

logger = logging.getLogger(__name__)


class _MspeStats(TypedDict):
    """Pre-period MSPE, post-period MSPE and their ratio for one unit."""

    pre_mspe: float
    post_mspe: float
    mspe_ratio: float


def _mspe(impact: xr.DataArray) -> float:
    """Mean squared prediction error of one unit's impact series.

    The residuals are averaged over the posterior before squaring, so the
    result is a scalar per unit rather than a distribution.  Point-estimate
    backends carry singleton ``chain``/``draw`` dimensions, so the same
    reduction is correct for them.

    Both reductions propagate missing values instead of skipping them, so a
    unit with any missing residual reports a non-finite error rather than an
    error computed from whatever happened to be present.
    """
    residuals = impact.mean(dim=("chain", "draw"), skipna=False)
    return float((residuals**2).mean(skipna=False))


def _mspe_stats(result: CausalResult, unit: str) -> _MspeStats:
    """Pre-period MSPE, post-period MSPE and their ratio for one unit.

    The ratio is the quantity used in section 3.4 and Figure 8 of Abadie,
    Diamond and Hainmueller (2010), so the values here are directly
    comparable to the ones published there.

    Three degenerate cases are reported apart rather than raising, so a
    single pathological donor does not sink the whole check:

    - positive post-period error over a zero pre-period error is ``inf``,
      a unit that ranks above every unit with a defined finite ratio;
    - zero over zero is ``nan``, an undefined ratio;
    - a non-finite pre- or post-period error is ``nan`` as well.
    """
    pre = _mspe(result.impact_pre.sel(treated_units=unit))
    post = _mspe(result.impact_post.sel(treated_units=unit))
    if not (np.isfinite(pre) and np.isfinite(post)):
        ratio = float("nan")
    elif pre > 0:
        ratio = post / pre
    elif post > 0:
        ratio = float("inf")
    else:
        ratio = float("nan")
    return {"pre_mspe": pre, "post_mspe": post, "mspe_ratio": ratio}


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
        """Verify the experiment is a SyntheticControl instance.

        Parameters
        ----------
        experiment : BaseExperiment
            Candidate experiment to validate.
        """
        if not isinstance(experiment, SyntheticControl):
            raise TypeError("PlaceboInSpace requires a SyntheticControl experiment.")

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext,
    ) -> CheckResult:
        """Treat each control unit as treated and compare effect magnitudes.

        Parameters
        ----------
        experiment : BaseExperiment
            The fitted SyntheticControl experiment.
        context : PipelineContext
            Pipeline context providing ``experiment_config`` for re-fits.
        """
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
            if placebo_treated in actual_treated:
                # A treated unit cannot be its own placebo, and ranking it twice would break the ratio plot.
                logger.warning(
                    "PlaceboInSpace: skipping '%s', which is listed as both a "
                    "control and a treated unit",
                    placebo_treated,
                )
                continue

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
                alt_experiment = method(context.data, **kw).fit()
                summary = alt_experiment.effect_summary()
            except Exception as exc:
                logger.warning(
                    "PlaceboInSpace: failed for '%s': %s",
                    placebo_treated,
                    exc,
                )
                rows.append({"placebo_treated": placebo_treated, "error": str(exc)})
                continue

            # Outside the try: a failed fit is a property of the placebo unit, an error in computing the statistic is a bug and must surface.
            row: dict[str, Any] = {"placebo_treated": placebo_treated}
            if summary.table is not None and not summary.table.empty:
                for col in summary.table.columns:
                    row[col] = summary.table[col].iloc[0]
            row.update(_mspe_stats(alt_experiment.result, placebo_treated))
            rows.append(row)

        table = pd.DataFrame(rows) if rows else None

        # The config can name treated units the fitted experiment does not carry, so only units present in its impact coordinates get a baseline; the rest are simply absent from the metadata. An unfitted experiment has no impact arrays, so it gets no baseline either.
        baseline_mspe: dict[str, _MspeStats] = {}
        if isinstance(experiment, SyntheticControl) and experiment.is_fitted:
            fitted_units = set(
                experiment.result.impact_pre.coords["treated_units"].values
            )
            baseline_mspe = {
                unit: _mspe_stats(experiment.result, unit)
                for unit in actual_treated
                if unit in fitted_units
            }

        text = (
            f"Placebo-in-space analysis: tested {len(all_controls)} control "
            f"units as placebo treated units. If placebo effects are "
            f"comparable to the actual effect, the causal claim may be "
            f"weakened."
        )
        if table is not None and "mspe_ratio" in table.columns:
            text += (
                " The post/pre MSPE ratio in the `mspe_ratio` column is the "
                "statistic to rank units by."
            )

        return CheckResult(
            check_name="PlaceboInSpace",
            passed=None,
            table=table,
            text=text,
            metadata={"baseline_mspe": baseline_mspe},
        )
