#   Copyright 2026 - 2026 The PyMC Labs Developers
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
"""Stochastic panels for the issue #914 qualification study.

The untreated path is a predeclared mean plus one residual draw. Treatment is
injected once into that path. The generator does not redraw noise when the
lift changes, and it does not stack a second residual on an already noisy path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from causalpy.historical_validation import ScheduleRequest
from causalpy.tests.qualification_protocol import (
    DONORS,
    FACTOR_SCALE,
    HISTORICAL_CODE,
    MASTER_SEED,
    MISFIT_LEVEL,
    MISFIT_TREND,
    PROTOCOL,
    SCENARIO_CODES,
    TREATED,
    WEIGHTS,
)


@dataclass(frozen=True)
class StochasticPanel:
    """One untreated panel and the residual that built it.

    ``data_seed_entropy`` identifies the stream. It is not a sampler seed and
    not a second dataset.
    """

    scenario: str
    panel_index: int
    data_seed_entropy: tuple[int, int, int]
    untreated: pd.DataFrame
    mean: pd.DataFrame
    residual: pd.DataFrame


def scenario_code(scenario: str) -> int:
    """Stable integer for a named scenario. Unknown names are a caller error."""
    try:
        return SCENARIO_CODES[scenario]
    except KeyError as error:
        raise KeyError(f"Unknown qualification scenario {scenario!r}.") from error


def seed_entropy(code: int, index: int) -> tuple[int, int, int]:
    """The three integers that identify one data or history stream."""
    return (MASTER_SEED, int(code), int(index))


def data_generator(code: int, index: int) -> np.random.Generator:
    """Data stream for one panel. Spawned apart from the sampler stream."""
    data_sequence, _sampler_sequence = np.random.SeedSequence(
        list(seed_entropy(code, index))
    ).spawn(2)
    return np.random.default_rng(data_sequence)


def sampler_seed(code: int, index: int) -> int:
    """Sampler seed for one panel. Not an independent dataset."""
    _data_sequence, sampler_sequence = np.random.SeedSequence(
        list(seed_entropy(code, index))
    ).spawn(2)
    return int(np.random.default_rng(sampler_sequence).integers(0, 2**31 - 1))


def _ar1(generator: np.random.Generator, n_time: int) -> np.ndarray:
    """Stationary AR(1) with the protocol innovation scale and lag-1 coefficient."""
    phi = PROTOCOL.ar_phi
    innovation_sd = PROTOCOL.residual_sd
    stationary_sd = innovation_sd / np.sqrt(1.0 - phi**2)
    values = np.empty(n_time, dtype=float)
    values[0] = generator.normal(0.0, stationary_sd)
    innovations = generator.normal(0.0, innovation_sd, size=n_time - 1)
    for time in range(1, n_time):
        values[time] = phi * values[time - 1] + innovations[time - 1]
    return values


def residual_frame(
    generator: np.random.Generator, n_time: int, *, serial: bool
) -> pd.DataFrame:
    """One residual draw. Columns are independent of each other."""
    columns = (*DONORS, TREATED)
    if serial:
        noise = np.column_stack([_ar1(generator, n_time) for _column in columns])
    else:
        noise = generator.normal(0.0, PROTOCOL.residual_sd, size=(n_time, len(columns)))
    return pd.DataFrame(
        noise, columns=columns, index=pd.RangeIndex(n_time, name="time")
    )


def mean_frame(n_time: int, *, misfit: bool) -> pd.DataFrame:
    """Donor factors and the treated mean.

    The reference treated mean is the exact convex combination. The misfit
    mean adds a level and a linear trend that the donor span cannot absorb.
    """
    time = np.arange(n_time, dtype=float)
    donors = np.column_stack(
        [
            FACTOR_SCALE * np.sin(2.0 * np.pi * time / 12.0),
            FACTOR_SCALE * np.cos(2.0 * np.pi * time / 10.0),
            FACTOR_SCALE * np.sin(2.0 * np.pi * time / 8.0 + 0.4),
        ]
    )
    weights = np.asarray(WEIGHTS, dtype=float)
    treated = donors @ weights
    if misfit:
        treated = treated + MISFIT_LEVEL + MISFIT_TREND * time
    frame = pd.DataFrame(
        donors, columns=DONORS, index=pd.RangeIndex(n_time, name="time")
    )
    frame[TREATED] = treated
    return frame


def _untreated_frame(
    scenario: str, code: int, index: int, n_time: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    misfit = scenario == "persistent_level_trend_misfit"
    serial = scenario == "serially_correlated_errors"
    mean = mean_frame(n_time, misfit=misfit)
    residual = residual_frame(data_generator(code, index), n_time, serial=serial)
    return mean + residual, mean, residual


def generate_untreated(scenario: str, panel_index: int) -> StochasticPanel:
    """Build one untreated panel. Does not inject treatment."""
    if panel_index < 0 or panel_index >= PROTOCOL.n_panels:
        raise ValueError(
            f"panel_index must be in 0..{PROTOCOL.n_panels - 1}, got {panel_index}."
        )
    code = scenario_code(scenario)
    untreated, mean, residual = _untreated_frame(
        scenario, code, panel_index, PROTOCOL.n_time
    )
    return StochasticPanel(
        scenario=scenario,
        panel_index=panel_index,
        data_seed_entropy=seed_entropy(code, panel_index),
        untreated=untreated,
        mean=mean,
        residual=residual,
    )


def inject_treatment(untreated: pd.DataFrame, lift: float) -> pd.DataFrame:
    """Add ``lift`` once to the treated holdout. Does not redraw the residual.

    The holdout is the last ``duration`` rows, which is the block the shipped
    planner treats as post-treatment when that duration is requested.
    """
    injected = untreated.copy()
    holdout = injected.index[-PROTOCOL.duration :]
    injected.loc[holdout, TREATED] = injected.loc[holdout, TREATED] + float(lift)
    return injected


def historical_replay_frame() -> pd.DataFrame:
    """One correctly specified history plus one residual draw.

    This is product evidence for replay, not an independent experiment and not
    one of the 100 panels. The stream is keyed by the historical code so it
    cannot collide with a panel stream.
    """
    untreated, _mean, _residual = _untreated_frame(
        "correctly_specified_noisy",
        HISTORICAL_CODE,
        0,
        PROTOCOL.historical_length,
    )
    if len(untreated) != PROTOCOL.historical_length:
        raise RuntimeError("Historical replay frame length drifted from the protocol.")
    return untreated


def historical_schedule_request() -> ScheduleRequest:
    """The frozen historical geometry. Not a duration selector."""
    return ScheduleRequest(
        selection_origins=PROTOCOL.historical_selection_origins,
        audit_origins=PROTOCOL.historical_audit_origins,
        horizons=(PROTOCOL.historical_horizon,),
        cadence=1,
        min_training_periods=PROTOCOL.historical_min_training,
    )
