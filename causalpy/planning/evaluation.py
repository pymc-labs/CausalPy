# Copyright 2022 - 2026 The PyMC Labs Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared candidate result for treatment search and grouped studies.

``search`` and ``grouped_study`` both return this type. It lives here so
neither module imports the other to name it.
"""

from __future__ import annotations

from dataclasses import dataclass

from causalpy.planning.power import EffectByDuration


@dataclass(frozen=True)
class _DiagnosticRecord:
    """Sampling diagnostics stored on one candidate evaluation.

    ``divergences`` is null when any replicate count is missing. It is not
    the power curve's required integer.
    """

    divergences: int | None
    max_rhat: float | None
    min_bulk_ess: float | None
    status: str | None
    reason: str | None


@dataclass
class _CandidateEvaluation:
    study: EffectByDuration | None
    status: str
    reason: str | None
    n_fits: int
    unit_scale: float | None = None
    diagnostics: _DiagnosticRecord | None = None
    failed_runs: int = 0


def _finished_evaluation(
    study: EffectByDuration,
    n_fits: int,
    *,
    unit_scale: float | None = None,
    diagnostics: _DiagnosticRecord | None = None,
) -> _CandidateEvaluation:
    """Label a finished curve. A withheld curve is not a completed selection."""
    if study.power_withheld:
        return _CandidateEvaluation(
            study,
            "withheld",
            study.withhold_reason,
            n_fits,
            unit_scale=unit_scale,
            diagnostics=diagnostics,
        )
    return _CandidateEvaluation(
        study,
        "completed",
        None,
        n_fits,
        unit_scale=unit_scale,
        diagnostics=diagnostics,
    )
