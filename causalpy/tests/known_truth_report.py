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
"""Fit known-truth panels for the perturbation test.

Not a public API. Does not add a panel generator.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import pandas as pd
import xarray as xr

from causalpy.experiments.augmented_synthetic_control import AugmentedSyntheticControl
from causalpy.experiments.interactive_fixed_effects import (
    InteractiveFixedEffectsSyntheticControl,
)
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pymc_models import WeightedSumFitter
from causalpy.tests import known_truth_panels

# Default sample when the caller does not pass one. The perturbation test
# passes its own short sample.
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

# Posterior series ``posterior_draws`` returns for the treated unit.
POSTERIOR_NAMES = {
    "SyntheticControl": ("predictions_post", "impact_post"),
    "InteractiveFixedEffectsSyntheticControl": ("predictions_post", "impact_post"),
    "AugmentedSyntheticControl": ("counterfactual", "treatment_effect"),
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
    names = POSTERIOR_NAMES[model_name]
    if model_name == "AugmentedSyntheticControl":
        posterior = fitted.idata.posterior
        return {name: posterior[name].sel(treated_units=unit) for name in names}
    result = fitted.result
    return {name: getattr(result, name).sel(treated_units=unit) for name in names}
