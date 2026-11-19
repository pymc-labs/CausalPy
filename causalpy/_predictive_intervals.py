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
"""Equal-tailed intervals for future realized outcomes and impacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from causalpy.reporting import EffectSummary, _format_probability_as_percent

_ASSUMPTIONS = (
    "The untreated dynamics and covariate relationships must continue after the "
    "intervention, with no concurrent unmodeled change explaining the gap. Pooled "
    "one-step errors assume approximately exchangeable innovations. Intervals "
    "describe future outcome variation conditional on the fitted model and supplied "
    "covariates. They exclude parameter refitting, uncertain future covariates, "
    "and uncertainty about causal identification. The stated level applies to "
    "each displayed point or aggregate separately."
)


def _prediction_interval(
    samples: xr.DataArray, prob: float
) -> tuple[xr.DataArray, xr.DataArray]:
    if not 0 < prob <= 1:
        raise ValueError("Prediction interval probability must be in (0, 1].")
    tail = (1 - prob) / 2
    bounds = samples.quantile([tail, 1 - tail], dim="simulation", skipna=False)
    return bounds.isel(quantile=0, drop=True), bounds.isel(quantile=1, drop=True)


def _effect_interval(
    values: xr.DataArray, prob: float, *, prefix: str = ""
) -> dict[str, float]:
    lower, upper = _prediction_interval(values, prob)
    return {
        f"{prefix}mean": float(values.mean(skipna=False)),
        f"{prefix}pi_lower": float(lower),
        f"{prefix}pi_upper": float(upper),
    }


def _stable_ratio(
    numerator: xr.DataArray, denominator: xr.DataArray, prob: float
) -> xr.DataArray:
    lower, upper = _prediction_interval(denominator, prob)
    if bool(((lower <= 0) & (upper >= 0)).any()) or bool((denominator == 0).any()):
        return xr.full_like(numerator, np.nan)
    ratio = numerator / denominator
    return ratio if bool(np.isfinite(ratio).all()) else xr.full_like(numerator, np.nan)


def _effect_rows(
    impact: xr.DataArray,
    paths: xr.DataArray,
    prob: float,
    *,
    cumulative: bool,
    relative: bool,
) -> dict[str, dict[str, float]]:
    average = _effect_interval(impact.mean("obs_ind"), prob)
    rows = {"average": average}
    if relative:
        # Match the sklearn average of period percentages, within each path.
        average_lift = 100 * _stable_ratio(impact, paths, prob).mean(
            "obs_ind", skipna=False
        )
        average.update(_effect_interval(average_lift, prob, prefix="relative_"))
    if cumulative:
        total = impact.sum("obs_ind")
        cumulative_row = _effect_interval(total, prob)
        if relative:
            total_lift = 100 * _stable_ratio(total, paths.sum("obs_ind"), prob)
            cumulative_row.update(
                _effect_interval(total_lift, prob, prefix="relative_")
            )
        rows["cumulative"] = cumulative_row
    return rows


def _format_effect(row: dict[str, float], prefix: str = "") -> str:
    estimate = row[f"{prefix}mean"]
    if not np.isfinite(estimate):
        return "unavailable (unstable denominator)"
    return f"{estimate:.2f} [{row[f'{prefix}pi_lower']:.2f}, {row[f'{prefix}pi_upper']:.2f}]"


def _prediction_summary(
    rows: dict[str, dict[str, float]],
    lines: list[str],
    paths: xr.DataArray,
    prob: float,
) -> EffectSummary:
    count = paths.attrs["calibration_count"]
    calibration = f"Calibrated using {count} finite held-out one-step errors."
    if paths.attrs["calibration_degenerate"]:
        calibration += " The pooled error distribution is degenerate."
    level = _format_probability_as_percent(prob)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.attrs["prediction_interval_prob"] = prob
    text = "\n\n".join(
        [
            f"Estimates and {level} bootstrap prediction intervals for realized impact.",
            "\n".join(lines),
            calibration,
            _ASSUMPTIONS,
        ]
    )
    return EffectSummary(table=table, text=text)


def _summarize_realized_effect(
    observed: xr.DataArray,
    paths: xr.DataArray,
    prob: float,
    *,
    cumulative: bool,
    relative: bool,
    prefix: str,
) -> EffectSummary:
    rows = _effect_rows(
        observed - paths, paths, prob, cumulative=cumulative, relative=relative
    )
    lines = [
        f"{prefix} {name} effect is {_format_effect(row)}."
        for name, row in rows.items()
    ]
    if relative:
        lines.extend(
            f"{prefix} {name} relative effect in percent is {_format_effect(row, 'relative_')}."
            for name, row in rows.items()
        )
    return _prediction_summary(rows, lines, paths, prob)


def _compare_realized_effects(
    observed: xr.DataArray,
    paths: xr.DataArray,
    treatment_end_time: int | float | pd.Timestamp,
    prob: float,
    *,
    cumulative: bool,
    relative: bool,
) -> EffectSummary:
    during = paths.obs_ind < treatment_end_time
    if not bool(during.any()) or bool(during.all()):
        raise ValueError("Period comparison requires observations in both periods.")
    impact = observed - paths
    periods = {"intervention": during, "post_intervention": ~during}
    rows: dict[str, dict[str, float]] = {}
    lines = []
    for period, mask in periods.items():
        effects = _effect_rows(
            impact.isel(obs_ind=mask),
            paths.isel(obs_ind=mask),
            prob,
            cumulative=cumulative,
            relative=relative,
        )
        rows[period] = effects["average"].copy()
        for effect, row in effects.items():
            period_label = period.replace("_", "-")
            lines.append(
                f"The {period_label} {effect} effect is {_format_effect(row)}."
            )
            if relative:
                lines.append(
                    f"The {period_label} {effect} relative effect in percent is "
                    f"{_format_effect(row, 'relative_')}."
                )
        if cumulative:
            rows[period].update(
                {f"total_{key}": value for key, value in effects["cumulative"].items()}
            )
    if relative:
        during_mean = impact.isel(obs_ind=during).mean("obs_ind")
        post_mean = impact.isel(obs_ind=~during).mean("obs_ind")
        ratio = 100 * _stable_ratio(post_mean, during_mean, prob)
        interval = _effect_interval(ratio, prob)
        rows["post_intervention"].update(
            persistence_ratio_pct=interval["mean"],
            persistence_pi_lower=interval["pi_lower"],
            persistence_pi_upper=interval["pi_upper"],
        )
        lines.append(
            f"Persistence as a percentage of intervention impact is {_format_effect(interval)}."
        )
    return _prediction_summary(rows, lines, paths, prob)
