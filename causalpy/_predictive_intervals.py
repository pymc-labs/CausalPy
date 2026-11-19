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


def _format_prediction_level(prob: float) -> str:
    """Format a predictive level consistently when specified as mass or alpha."""
    # The binary complement of alpha can differ slightly from a direct probability.
    percent = np.format_float_positional(
        prob * 100, precision=12, unique=False, trim="-"
    )
    if percent in ("0", "100") and 0 < prob < 1:
        return _format_probability_as_percent(prob)
    return f"{percent}%"


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
    prob: float,
    *,
    cumulative: bool,
    counterfactual: xr.DataArray | None = None,
) -> dict[str, dict[str, float]]:
    average = _effect_interval(impact.mean("obs_ind"), prob)
    rows = {"average": average}
    if counterfactual is not None:
        # Match the sklearn average of period percentages, within each path.
        average_lift = 100 * _stable_ratio(impact, counterfactual, prob).mean(
            "obs_ind", skipna=False
        )
        average.update(_effect_interval(average_lift, prob, prefix="relative_"))
    if cumulative:
        total = impact.sum("obs_ind")
        cumulative_row = _effect_interval(total, prob)
        if counterfactual is not None:
            total_lift = 100 * _stable_ratio(total, counterfactual.sum("obs_ind"), prob)
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


def _calibration_text(paths: xr.DataArray) -> str:
    count = paths.attrs["calibration_count"]
    pool = paths.attrs["residual_pool_count"]
    text = f"Held-out calibration measured {count} finite one-step errors."
    if pool != count:
        text += f" The bootstrap sampled from a pool of {pool} of those errors."
    if paths.attrs["calibration_degenerate"]:
        text += " The pooled error distribution is degenerate."
    return text


def _observations_text(paths: xr.DataArray) -> str:
    index = paths.get_index("obs_ind")
    label = "observation" if len(index) == 1 else "observations"
    return f"{len(index)} {label} from {index.min()} through {index.max()}"


def _prediction_summary(
    rows: dict[str, dict[str, float]],
    lines: list[str],
    paths: xr.DataArray,
    prob: float,
) -> EffectSummary:
    level = _format_prediction_level(prob)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.attrs["prediction_interval_prob"] = prob
    text = "\n\n".join(
        [
            f"Estimates and {level} bootstrap prediction intervals for realized impact.",
            "\n".join(lines),
            _calibration_text(paths),
            _ASSUMPTIONS,
        ]
    )
    return EffectSummary(table=table, text=text)


def _effect_lines(
    prefix: str, rows: dict[str, dict[str, float]], relative: bool
) -> list[str]:
    lines = [
        f"{prefix} {name} effect is {_format_effect(row)}."
        for name, row in rows.items()
    ]
    if relative:
        lines.extend(
            f"{prefix} {name} relative effect in percent is {_format_effect(row, 'relative_')}."
            for name, row in rows.items()
        )
    return lines


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
        observed - paths,
        prob,
        cumulative=cumulative,
        counterfactual=paths if relative else None,
    )
    lines = [
        f"{prefix} includes {_observations_text(paths)}.",
        *_effect_lines(prefix, rows, relative),
    ]
    return _prediction_summary(rows, lines, paths, prob)


def _split_realized_periods(
    samples: xr.DataArray,
    treatment_end_time: int | float | pd.Timestamp,
) -> tuple[xr.DataArray, xr.DataArray]:
    during = samples.get_index("obs_ind") < treatment_end_time
    if not bool(during.any()) or bool(during.all()):
        raise ValueError("Period comparison requires observations in both periods.")
    return samples.isel(obs_ind=during), samples.isel(obs_ind=~during)


def _persistence_interval(
    during: xr.DataArray, post: xr.DataArray, prob: float
) -> dict[str, float]:
    ratio = _stable_ratio(post.mean("obs_ind"), during.mean("obs_ind"), prob)
    return _effect_interval(ratio, prob)


def _compare_realized_effects(
    observed: xr.DataArray,
    paths: xr.DataArray,
    treatment_end_time: int | float | pd.Timestamp,
    prob: float,
    *,
    cumulative: bool,
    relative: bool,
) -> EffectSummary:
    during_paths, post_paths = _split_realized_periods(paths, treatment_end_time)
    during_impact = (
        observed.sel(obs_ind=during_paths.get_index("obs_ind")) - during_paths
    )
    post_impact = observed.sel(obs_ind=post_paths.get_index("obs_ind")) - post_paths
    rows: dict[str, dict[str, float]] = {}
    lines = []
    for period, impact, counterfactual in (
        ("intervention", during_impact, during_paths),
        ("post_intervention", post_impact, post_paths),
    ):
        effects = _effect_rows(
            impact,
            prob,
            cumulative=cumulative,
            counterfactual=counterfactual if relative else None,
        )
        rows[period] = effects["average"].copy()
        period_label = period.replace("_", "-")
        lines.append(
            f"The {period_label} period includes {_observations_text(impact)}."
        )
        lines.extend(_effect_lines(f"The {period_label}", effects, relative))
        if cumulative:
            rows[period].update(
                {f"total_{key}": value for key, value in effects["cumulative"].items()}
            )
    if relative:
        interval = {
            key: 100 * value
            for key, value in _persistence_interval(
                during_impact, post_impact, prob
            ).items()
        }
        rows["post_intervention"].update(
            persistence_ratio_pct=interval["mean"],
            persistence_pi_lower=interval["pi_lower"],
            persistence_pi_upper=interval["pi_upper"],
        )
        lines.append(
            f"Persistence as a percentage of intervention impact is {_format_effect(interval)}."
        )
    return _prediction_summary(rows, lines, paths, prob)


def _analyze_realized_persistence(
    observed: xr.DataArray,
    paths: xr.DataArray,
    treatment_end_time: int | float | pd.Timestamp,
    prob: float,
) -> dict[str, float]:
    during, post = _split_realized_periods(observed - paths, treatment_end_time)
    result = {}
    level = _format_prediction_level(prob)
    print(
        f"Effect persistence with {level} bootstrap prediction intervals for realized impact."
    )
    for period, label, impact in (
        ("during", "Intervention", during),
        ("post", "Post-intervention", post),
    ):
        print(f"{label} period includes {_observations_text(impact)}.")
        effects = _effect_rows(impact, prob, cumulative=True)
        for metric, estimand in (("mean", "average"), ("total", "cumulative")):
            row = effects[estimand]
            key = f"{metric}_effect_{period}"
            result[key] = row["mean"]
            result[f"{key}_pi_lower"] = row["pi_lower"]
            result[f"{key}_pi_upper"] = row["pi_upper"]
            print(f"  {metric.capitalize()} effect is {_format_effect(row)}.")
    ratio = _persistence_interval(during, post, prob)
    result.update(
        persistence_ratio=ratio["mean"],
        persistence_ratio_pi_lower=ratio["pi_lower"],
        persistence_ratio_pi_upper=ratio["pi_upper"],
    )
    print(f"Persistence ratio is {_format_effect(ratio)}.")
    return result
