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
"""Outcome-only known-truth panels for tests.

Not a public API. Callers use the five module attributes. The class name is
not part of the contract. Values are exact: ``data`` equals ``untreated`` plus
``effect`` in every cell, with no observation noise and no sampling.
"""

from dataclasses import dataclass

import pandas as pd

_TIME = pd.Index(range(8), dtype="int64")
_TREATMENT_TIME = 4
_CONTROL_UNITS = ("c0", "c1", "c2")
_TREATED_UNITS = ("treated",)
_COLUMNS = list(_CONTROL_UNITS + _TREATED_UNITS)


@dataclass(frozen=True)
class _KnownTruthPanel:
    """Module-private frozen panel. Callers use attribute access only."""

    data: pd.DataFrame
    untreated: pd.DataFrame
    effect: pd.DataFrame
    treatment_time: int
    control_units: tuple[str, ...]
    treated_units: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", self.data.copy())
        object.__setattr__(self, "untreated", self.untreated.copy())
        object.__setattr__(self, "effect", self.effect.copy())


def _frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    ordered = {name: columns[name] for name in _COLUMNS}
    return pd.DataFrame(ordered, index=_TIME)


def _effect(treated_post: float) -> pd.DataFrame:
    effect = pd.DataFrame(0.0, index=_TIME, columns=_COLUMNS)
    post = _TIME >= _TREATMENT_TIME
    effect.loc[post, _TREATED_UNITS[0]] = treated_post
    return effect


def _panel(untreated: pd.DataFrame, treated_post: float) -> _KnownTruthPanel:
    effect = _effect(treated_post)
    return _KnownTruthPanel(
        data=untreated + effect,
        untreated=untreated,
        effect=effect,
        treatment_time=_TREATMENT_TIME,
        control_units=_CONTROL_UNITS,
        treated_units=_TREATED_UNITS,
    )


def _exact_untreated() -> pd.DataFrame:
    """Treated path is weights (1/2, 1/4, 1/4) of the control columns."""
    c0 = [0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0]
    c1 = [0.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0]
    c2 = [0.0, 0.0, 0.0, 8.0, 0.0, 8.0, 8.0, 8.0]
    treated = [
        0.5 * a + 0.25 * b + 0.25 * c for a, b, c in zip(c0, c1, c2, strict=True)
    ]
    return _frame({"c0": c0, "c1": c1, "c2": c2, "treated": treated})


def _inexact_untreated() -> pd.DataFrame:
    """Treated pre-period sits outside the convex hull of the controls."""
    c0 = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    c1 = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    c2 = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    treated = [3.0, -1.0, 3.0, -1.0, 3.0, -1.0, 3.0, -1.0]
    return _frame({"c0": c0, "c1": c1, "c2": c2, "treated": treated})


def _latent_untreated() -> pd.DataFrame:
    """Exact rank-2 product. Treated loadings are outside the control hull."""
    factor_a = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    factor_b = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    loadings = {
        "c0": (1.0, 0.0),
        "c1": (0.0, 1.0),
        "c2": (1.0, 1.0),
        "treated": (2.0, -1.0),
    }
    columns = {
        name: [load_a * a + load_b * b for a, b in zip(factor_a, factor_b, strict=True)]
        for name, (load_a, load_b) in loadings.items()
    }
    return _frame(columns)


def _donor_shock_untreated() -> pd.DataFrame:
    """One control post path is shocked off the path that builds treated."""
    c0 = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    c1 = [0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    generated_c2 = [0.0, 0.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    shocked_c2 = [0.0, 0.0, 1.0, 3.0, 8.0, 8.0, 8.0, 8.0]
    treated = [
        0.5 * a + 0.25 * b + 0.25 * c
        for a, b, c in zip(c0, c1, generated_c2, strict=True)
    ]
    return _frame({"c0": c0, "c1": c1, "c2": shocked_c2, "treated": treated})


exact_convex_hull = _panel(_exact_untreated(), treated_post=2.0)
systematic_inexact = _panel(_inexact_untreated(), treated_post=2.0)
latent_factor = _panel(_latent_untreated(), treated_post=2.0)
donor_shock = _panel(_donor_shock_untreated(), treated_post=0.0)
placebo = _panel(_exact_untreated(), treated_post=0.0)
