#   Copyright 2025 - 2026 The PyMC Labs Developers
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
"""Fit the five known-truth panels and score posterior HDIs.

Not a public API. Tests and the committed known-truth note share this module.
It does not add a panel generator.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import xarray as xr

from causalpy._arviz_compat import hdi
from causalpy.constants import HDI_PROB
from causalpy.experiments.augmented_synthetic_control import AugmentedSyntheticControl
from causalpy.experiments.interactive_fixed_effects import (
    InteractiveFixedEffectsSyntheticControl,
)
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pymc_models import WeightedSumFitter
from causalpy.tests import known_truth_panels

NOTE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "source"
    / "knowledgebase"
    / "bayesian_synthetic_control_known_truth_report.md"
)

# One setting for every recorded inside/outside bit. Nightly refits use this
# mapping and no other.
REPORT_SAMPLE: dict[str, int | float | bool] = {
    "draws": 100,
    "tune": 100,
    "chains": 2,
    "cores": 1,
    "random_seed": 1241,
    "progressbar": False,
    "target_accept": 0.95,
}

PANELS = {
    "exact_convex_hull": known_truth_panels.exact_convex_hull,
    "systematic_inexact": known_truth_panels.systematic_inexact,
    "latent_factor": known_truth_panels.latent_factor,
    "donor_shock": known_truth_panels.donor_shock,
    "placebo": known_truth_panels.placebo,
}

MODELS = (
    "SyntheticControl",
    "InteractiveFixedEffectsSyntheticControl",
    "AugmentedSyntheticControl",
)

# Posterior name -> planted series it is scored against.
POSTERIOR_TARGET = {
    "SyntheticControl": {
        "predictions_post": "untreated",
        "impact_post": "effect",
    },
    "InteractiveFixedEffectsSyntheticControl": {
        "predictions_post": "untreated",
        "impact_post": "effect",
    },
    "AugmentedSyntheticControl": {
        "counterfactual": "untreated",
        "treatment_effect": "effect",
    },
}


def fit_panel(
    model_name: str,
    panel_name: str,
    *,
    data: pd.DataFrame | None = None,
    sample: Mapping[str, int | float | bool] | None = None,
):
    """Fit one recorded model on one existing panel. Does not add a panel."""
    panel = PANELS[panel_name]
    frame = panel.data if data is None else data
    controls = list(panel.control_units)
    treated = list(panel.treated_units)
    kwargs = dict(REPORT_SAMPLE if sample is None else sample)
    if model_name == "SyntheticControl":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            experiment = SyntheticControl(
                frame,
                panel.treatment_time,
                controls,
                treated,
                model=WeightedSumFitter(sample_kwargs=kwargs),
            )
            experiment.fit()
        return experiment
    if model_name == "InteractiveFixedEffectsSyntheticControl":
        return InteractiveFixedEffectsSyntheticControl(
            frame,
            panel.treatment_time,
            controls,
            treated,
        ).fit(**kwargs)
    if model_name == "AugmentedSyntheticControl":
        return AugmentedSyntheticControl(
            frame,
            panel.treatment_time,
            controls,
            treated,
        ).fit(kwargs)
    raise ValueError(f"unknown model {model_name}")


def posterior_draws(
    model_name: str, fitted, panel_name: str
) -> dict[str, xr.DataArray]:
    """Return the recorded posterior series for the single treated unit."""
    panel = PANELS[panel_name]
    unit = panel.treated_units[0]
    if model_name == "AugmentedSyntheticControl":
        posterior = fitted.idata.posterior
        return {
            name: posterior[name].sel(treated_units=unit)
            for name in POSTERIOR_TARGET[model_name]
        }
    result = fitted.result
    return {
        name: getattr(result, name).sel(treated_units=unit)
        for name in POSTERIOR_TARGET[model_name]
    }


def planted_series(panel_name: str, target: str) -> pd.Series:
    """Post-period planted untreated path or effect for the treated unit."""
    panel = PANELS[panel_name]
    unit = panel.treated_units[0]
    frame = panel.untreated if target == "untreated" else panel.effect
    post = frame.index >= panel.treatment_time
    return frame.loc[post, unit]


def inside_flags(draws: xr.DataArray, truth: pd.Series) -> dict[int, int]:
    """1 if the planted value is inside the HDI at that post time, else 0."""
    time_dim = "obs_ind" if "obs_ind" in draws.dims else "post_time"
    bounds = hdi(draws, prob=HDI_PROB, dim=("chain", "draw"))
    flags: dict[int, int] = {}
    for time in draws.coords[time_dim].values:
        lower = float(bounds.sel({time_dim: time, "hdi": "lower"}))
        higher = float(bounds.sel({time_dim: time, "hdi": "higher"}))
        value = float(truth.loc[time])
        flags[int(time)] = int(lower <= value <= higher)
    return flags


def score_fit(
    model_name: str,
    panel_name: str,
    *,
    sample: Mapping[str, int | float | bool] | None = None,
) -> dict[tuple[str, int], int]:
    """Map ``(posterior_name, post_time)`` to inside (1) or outside (0)."""
    fitted = fit_panel(model_name, panel_name, sample=sample)
    draws = posterior_draws(model_name, fitted, panel_name)
    scored: dict[tuple[str, int], int] = {}
    for posterior_name, target in POSTERIOR_TARGET[model_name].items():
        flags = inside_flags(draws[posterior_name], planted_series(panel_name, target))
        for time, flag in flags.items():
            scored[(posterior_name, time)] = flag
    return scored
