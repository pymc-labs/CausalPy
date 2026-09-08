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
import warnings
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from plotnine import (
    aes,
    coord_flip,
    geom_col,
    geom_hline,
    ggplot,
    labs,
    scale_fill_manual,
)

from causalpy.checks._plot_helpers import draw_figure
from causalpy.checks.base import CheckResult, clone_model
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pipeline import PipelineContext

logger = logging.getLogger(__name__)

_DEFAULT_PLOT_TITLE = "Placebo-in-space: post/pre MSPE ratio"
_DEFAULT_FIGSIZE = (7.0, 8.0)
# Grey for the donor pool, red for the actual treated unit, matching the
# palette the other check figures use.
_PLACEBO_COLOUR = "#94a3b8"
_TREATED_COLOUR = "#E24A33"


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
    sample_dims = [dim for dim in ("chain", "draw") if dim in impact.dims]
    residuals = impact.mean(dim=sample_dims, skipna=False) if sample_dims else impact
    return float((residuals**2).mean(skipna=False))


def _mspe_stats(experiment: Any, unit: str) -> dict[str, float]:
    """Pre-period MSPE, post-period MSPE and their ratio for one unit.

    The ratio is the quantity used in section 5.2 of Abadie, Diamond and
    Hainmueller (2010), so the values here are directly comparable to the
    ones published there.

    Three degenerate cases are reported apart rather than raising, so a
    single pathological donor does not sink the whole check:

    - positive post-period error over a zero pre-period error is ``inf``,
      a unit that ranks above every unit with a defined finite ratio;
    - zero over zero is ``nan``, an undefined ratio;
    - a non-finite pre- or post-period error is ``nan`` as well.

    :meth:`PlaceboInSpace.plot_mspe_ratio` keeps ``inf`` in the permutation
    reference set and drops only ``nan``; the figure itself can draw neither.
    """
    pre = _mspe(experiment.pre_impact.sel(treated_units=unit))
    post = _mspe(experiment.post_impact.sel(treated_units=unit))
    if not (np.isfinite(pre) and np.isfinite(post)):
        ratio = float("nan")
    elif pre > 0:
        ratio = post / pre
    elif post > 0:
        ratio = float("inf")
    else:
        ratio = float("nan")
    return {"pre_mspe": pre, "post_mspe": post, "mspe_ratio": ratio}


def _permutation_pvalue(ratios: np.ndarray, treated_ratio: float) -> float:
    """Share of units whose MSPE ratio is at least the treated unit's.

    ``ratios`` is the reference set for one treated unit: the placebo units
    plus that unit itself, and no other treated unit.  Because the treated
    unit is in its own reference set, the smallest attainable value is
    ``1 / n_units``: with a donor pool of 19, a treated unit that ranks
    first gives ``p = 0.05``.
    """
    return float(np.mean(ratios >= treated_ratio))


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
                row.update(_mspe_stats(alt_experiment, placebo_treated))
                rows.append(row)
            except Exception as exc:
                logger.warning(
                    "PlaceboInSpace: failed for '%s': %s",
                    placebo_treated,
                    exc,
                )
                rows.append({"placebo_treated": placebo_treated, "error": str(exc)})

        table = pd.DataFrame(rows) if rows else None

        # The config can name treated units the fitted experiment does not
        # carry, so only units present in its impact coordinates get a
        # baseline; the rest are simply absent from the metadata.
        baseline_mspe: dict[str, dict[str, float]] = {}
        if isinstance(experiment, SyntheticControl):
            fitted_units = set(
                np.asarray(
                    experiment.pre_impact.coords["treated_units"].values
                ).tolist()
            )
            baseline_mspe = {
                unit: _mspe_stats(experiment, unit)
                for unit in actual_treated
                if unit in fitted_units
            }

        text = (
            f"Placebo-in-space analysis: tested {len(all_controls)} control "
            f"units as placebo treated units. If placebo effects are "
            f"comparable to the actual effect, the causal claim may be "
            f"weakened. The post/pre MSPE ratio in the `mspe_ratio` column "
            f"is the statistic to rank units by; see "
            f"`PlaceboInSpace.plot_mspe_ratio`."
        )

        return CheckResult(
            check_name="PlaceboInSpace",
            passed=None,
            table=table,
            text=text,
            metadata={"baseline_mspe": baseline_mspe},
        )

    @staticmethod
    def plot_mspe_ratio(
        check_result: CheckResult,
        *,
        title: str = _DEFAULT_PLOT_TITLE,
        figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
        show_pvalue: bool = True,
    ) -> Figure:
        """Plot the post/pre MSPE ratio of every unit, treated unit highlighted.

        This is the inferential view recommended in Abadie, Diamond and
        Hainmueller (2010), section 5.2, and the ratio is theirs unchanged:
        mean squared prediction error after the intervention over mean squared
        prediction error before it.  A unit with a large ratio tracks its
        synthetic control closely before the intervention and diverges after
        it, which is the signature of either a real effect or a structural
        break.  Inference is the permutation rank of the treated unit's ratio
        within the donor distribution, so the treated unit standing out is the
        evidence, not the size of the ratio on its own.

        Prefer this over the raw effect sizes in ``check_result.table``: a
        donor whose pre-period fit is poor can show a large post-period
        divergence without that meaning anything, and dividing by the
        pre-period MSPE is what removes it.

        With several actual treated units, each one is ranked against the
        placebo units and itself only, never against the other treated units,
        so one treated unit's p-value does not depend on the others.

        The reference set is every unit whose ratio is defined, infinite ratios
        included, so the reported p-value is conditional on the units that
        fitted successfully.  Units with an undefined ratio, meaning a failed
        placebo fit or a zero post-period error over a zero pre-period error,
        are outside both the figure and the p-value.  Infinite ratios count
        towards the p-value but cannot be drawn on a finite axis, so they are
        left out of the bars alone.

        Parameters
        ----------
        check_result : CheckResult
            Result returned by :meth:`run`.  The bars come from its
            ``mspe_ratio`` column and the highlighted unit(s) from
            ``metadata["baseline_mspe"]``.
        title : str, default "Placebo-in-space: post/pre MSPE ratio"
            Figure suptitle.
        figsize : tuple of float, default (7, 8)
            Size of the drawn figure, in inches.
        show_pvalue : bool, default True
            Whether to report the permutation p-value of each treated unit as
            a subtitle.

        Returns
        -------
        matplotlib.figure.Figure
            The drawn figure.

        Raises
        ------
        ValueError
            If the result carries no ``mspe_ratio`` column, or if no unit has a
            finite ratio to draw.

        Warns
        -----
        UserWarning
            If any unit has an undefined ratio, or a defined but infinite one,
            and so cannot be drawn.
        """
        table = check_result.table
        if table is None or "mspe_ratio" not in table.columns:
            raise ValueError(
                "Cannot plot: the CheckResult has no 'mspe_ratio' column. "
                "This happens when no placebo fit succeeded, or when the "
                "result predates MSPE reporting."
            )

        baseline: dict[str, dict[str, float]] = check_result.metadata.get(
            "baseline_mspe", {}
        )
        frame = pd.DataFrame(
            {
                "unit": list(table["placebo_treated"]) + list(baseline),
                "mspe_ratio": list(table["mspe_ratio"])
                + [stats["mspe_ratio"] for stats in baseline.values()],
                "role": ["Placebo"] * len(table) + ["Treated"] * len(baseline),
            }
        )

        # An undefined ratio means the unit was never ranked at all: it is out
        # of the figure and out of every reference set below.
        defined = frame["mspe_ratio"].notna()
        if not defined.all():
            dropped = ", ".join(frame.loc[~defined, "unit"].astype(str))
            warnings.warn(
                f"Dropping {int((~defined).sum())} unit(s) with an undefined "
                f"post/pre MSPE ratio: {dropped}. A failed placebo fit, or a "
                f"zero post-period error over a zero pre-period error, leaves "
                f"the ratio undefined. These units are outside the reported "
                f"p-value as well as the figure.",
                UserWarning,
                stacklevel=2,
            )
            frame = frame[defined]

        # An infinite ratio is ranked, it just cannot be drawn on a finite
        # axis, so it leaves the bars and stays in the reference set.
        finite = np.isfinite(frame["mspe_ratio"])
        if not finite.all():
            unplotted = ", ".join(frame.loc[~finite, "unit"].astype(str))
            warnings.warn(
                f"Leaving {int((~finite).sum())} unit(s) with an infinite "
                f"post/pre MSPE ratio out of the figure: {unplotted}. A "
                f"positive post-period error over a zero pre-period error is a "
                f"defined infinite ratio, so these units still count towards "
                f"the reported p-value.",
                UserWarning,
                stacklevel=2,
            )
        plot_frame = frame[finite].copy()
        if plot_frame.empty:
            raise ValueError("Cannot plot: no unit has a finite MSPE ratio.")

        # Ordering by ratio is what makes the figure readable: the treated
        # unit's rank is the inference, so it has to be visible at a glance.
        order = plot_frame.sort_values("mspe_ratio")["unit"].tolist()
        plot_frame["unit"] = pd.Categorical(plot_frame["unit"], categories=order)

        subtitle = ""
        if show_pvalue and baseline:
            # Each treated unit is ranked against the placebo pool plus itself.
            # Including the other treated units would make one treated unit's
            # p-value depend on how many others were fitted.
            placebo_ratios = frame.loc[
                frame["role"] == "Placebo", "mspe_ratio"
            ].to_numpy()
            annotations = []
            for unit, stats in baseline.items():
                ratio = stats["mspe_ratio"]
                if np.isnan(ratio):
                    continue
                reference = np.append(placebo_ratios, ratio)
                annotations.append(
                    f"{unit}: p = {_permutation_pvalue(reference, ratio):.3f}"
                )
            if annotations:
                # Named for Abadie because the effect-summary columns already
                # carry an unrelated `p_value`, and the two sit side by side in
                # the same CheckResult.
                subtitle = "Abadie permutation p-value.  " + ",  ".join(annotations)

        plot = (
            ggplot(plot_frame, aes("unit", "mspe_ratio", fill="role"))
            + geom_col()
            + geom_hline(yintercept=1.0, linetype="dashed", alpha=0.5)
            + coord_flip()
            + scale_fill_manual(
                values={"Placebo": _PLACEBO_COLOUR, "Treated": _TREATED_COLOUR}
            )
            + labs(x="", y="Post/pre MSPE ratio", fill="", subtitle=subtitle)
        )
        return draw_figure(plot, title, figsize)
