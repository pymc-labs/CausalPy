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
"""Boundary regression tests for the COVID tutorial's damped trend construction."""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest
from packaging.specifiers import SpecifierSet
from pytensor import scan

import causalpy as cp
from causalpy.pymc_forecast_models import PyMCForecastModel

pf = pytest.importorskip("pymc_forecast", minversion="0.3")
if not SpecifierSet(">=0.3,<0.4").contains(pf.__version__, prereleases=False):
    pytest.skip("requires pymc-forecast>=0.3,<0.4", allow_module_level=True)

LEVEL_SCALE = 0.15
SLOPE_SCALE = 0.1


def _damped_trend(covariates, data=None):
    """Keep only the tutorial's trend, so its forecast boundary is observable."""
    horizon = pf.Horizon.from_data(covariates, data)
    initial_level = pm.Normal("initial_level", 5, 0.5)
    initial_slope = pm.Normal("initial_slope", 0.5, 0.2)
    damping = pm.Beta("damping", 20, 2)
    level_z = pf.innovations(horizon, "level_z", pm.Normal.dist(0, 1))
    slope_z = pf.innovations(horizon, "slope_z", pm.Normal.dist(0, 1))
    slope = scan(
        fn=lambda innovation, previous, damping, scale: (
            damping * previous + scale * innovation
        ),
        sequences=[slope_z],
        outputs_info=[initial_slope],
        non_sequences=[damping, pt.as_tensor_variable(SLOPE_SCALE)],
        strict=True,
        return_updates=False,
    )
    level = initial_level + pt.cumsum(slope) + LEVEL_SCALE * pt.cumsum(level_z)
    pf.predict(horizon, pm.Normal.dist(0, 0.1), level, expected_observation=level)


@pytest.fixture(scope="module")
def damped_result(real_pymc_sampling):
    """Fit once; forecast draws outnumber fit draws to exercise shared replay."""
    dates = pd.date_range("2010-01-01", periods=22, freq="MS")
    frame = pd.DataFrame(
        {"y": 5 + 0.5 * np.cumsum(0.9 ** np.arange(1, len(dates) + 1))},
        index=dates,
    )
    return cp.InterruptedTimeSeries(
        frame,
        dates[14],
        formula="y ~ 0",
        model=PyMCForecastModel(
            _damped_trend,
            forecaster_kwargs={
                "draws": 100,
                "tune": 100,
                "chains": 2,
                "sample_kwargs": {"cores": 1, "target_accept": 0.9},
                "progressbar": False,
            },
            num_samples=400,
            random_seed=42,
        ),
    ).fit()


def _terminal_state(posterior):
    """Independent closed form for the damped impulse response over history."""
    damping = posterior["damping"].values
    slope_z = posterior["slope_z"].values
    n_training = slope_z.shape[-1]
    powers = damping[..., None] ** np.arange(n_training)
    weights = np.cumsum(powers, axis=-1)[..., ::-1]
    terminal_slope = damping**n_training * posterior[
        "initial_slope"
    ].values + SLOPE_SCALE * np.sum(powers[..., ::-1] * slope_z, axis=-1)
    terminal_level = (
        posterior["initial_level"].values
        + posterior["initial_slope"].values
        * np.sum(damping[..., None] * powers, axis=-1)
        + SLOPE_SCALE * np.sum(weights * slope_z, axis=-1)
        + LEVEL_SCALE * posterior["level_z"].values.sum(axis=-1)
    )
    return terminal_level, terminal_slope


@pytest.mark.integration
def test_damped_forecast_continues_each_posterior_terminal_state(damped_result):
    """Historical innovations replay, while fresh future uncertainty accumulates."""
    posterior = damped_result.model.idata.posterior
    terminal_level, terminal_slope = _terminal_state(posterior)
    fitted_terminal = damped_result.result.predictions_pre.isel(
        obs_ind=-1, treated_units=0
    ).values
    np.testing.assert_allclose(fitted_terminal, terminal_level, atol=1e-10)

    future = damped_result.result.predictions_post.isel(treated_units=0).values
    damping = posterior["damping"].values
    # Check the boundary and a distant step separately: restarting the slope, replaying future innovations, and dropping level noise break these checks.
    for horizon in (1, future.shape[-1]):
        powers = damping[..., None] ** np.arange(horizon)
        impulse_weights = np.cumsum(powers, axis=-1)
        conditional_mean = terminal_level + terminal_slope * np.sum(
            damping[..., None] * powers, axis=-1
        )
        conditional_variance = LEVEL_SCALE**2 * horizon + SLOPE_SCALE**2 * np.sum(
            impulse_weights**2, axis=-1
        )
        standardized = (future[..., horizon - 1] - conditional_mean) / np.sqrt(
            conditional_variance
        )
        assert abs(standardized.mean()) < 0.15
        assert standardized.std() == pytest.approx(1, rel=0.15)


@pytest.mark.integration
def test_damped_forecast_draws_reconstruct_from_new_innovations(damped_result):
    """Future level and slope noise belong to the same shared posterior draw."""
    model = damped_result.model
    samples = pf.prediction_samples(
        model.forecaster.forecast(
            future_index=damped_result.datapost.index.values,
            posterior=model.idata,
            random_seed=model.random_seed,
        )
    )
    level, slope = _terminal_state(model.idata.posterior)
    damping = model.idata.posterior["damping"].values
    reconstructed = []
    for step in range(samples.sizes["time_future"]):
        slope = (
            damping * slope + SLOPE_SCALE * samples["slope_z_future"].values[..., step]
        )
        level = (
            level + slope + LEVEL_SCALE * samples["level_z_future"].values[..., step]
        )
        reconstructed.append(level)
    reconstructed = np.stack(reconstructed, axis=-1)
    np.testing.assert_allclose(
        samples["expected_observation_future"].values, reconstructed, atol=1e-10
    )
    np.testing.assert_allclose(
        damped_result.result.predictions_post.isel(treated_units=0).values,
        reconstructed,
        atol=1e-10,
    )


@pytest.mark.integration
def test_damping_one_replays_undamped_local_linear_trend(damped_result):
    """At damping=1 the recurrence reduces to the cumulative local linear trend."""
    model = damped_result.model
    posterior = model.idata.posterior.to_dataset()
    # Replay existing posterior innovations with damping fixed at the limiting value. No additional fit or sampling from a degenerate Beta is needed.
    undamped = posterior.assign(damping=posterior["damping"] * 0 + 1)
    samples = pf.prediction_samples(
        model.forecaster.forecast(
            future_index=damped_result.datapost.index.values,
            posterior=undamped,
            random_seed=43,
        )
    )
    historical_slope = posterior["initial_slope"].values[..., None] + (
        SLOPE_SCALE * np.cumsum(posterior["slope_z"].values, axis=-1)
    )
    terminal_level = (
        posterior["initial_level"].values
        + historical_slope.sum(axis=-1)
        + LEVEL_SCALE * posterior["level_z"].values.sum(axis=-1)
    )
    future_slope = historical_slope[..., -1, None] + SLOPE_SCALE * np.cumsum(
        samples["slope_z_future"].values, axis=-1
    )
    future_level = (
        terminal_level[..., None]
        + np.cumsum(future_slope, axis=-1)
        + LEVEL_SCALE * np.cumsum(samples["level_z_future"].values, axis=-1)
    )
    np.testing.assert_allclose(
        samples["expected_observation_future"].values, future_level, atol=1e-10
    )
