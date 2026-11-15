# Copyright 2022 - 2026 The PyMC Labs Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Protocols and ownership for the planning seams.

Issue #1440 adds ``causalpy.planning.adapters`` with an ``ADAPTERS`` dict
mapping estimator names to adapter classes. Those classes accept ``model``,
``sample_kwargs``, and ``random_seed`` as keyword arguments and implement
:class:`EstimatorAdapter`. They must not edit ``reference.py``.

Issue #1439 adds power and precision against :class:`PowerEvaluator`.
Issue #1441 adds candidate generation against :class:`CandidateGenerator`.
Issue #319 integrates the three and owns post-search qualification.
Search, simulation, reporting, and plotting consume canonical results.
They do not import concrete estimator classes.
"""

from __future__ import annotations

from typing import Protocol

from causalpy.planning.contracts import (
    CandidateRequest,
    CandidateTable,
    CanonicalEvaluation,
    CanonicalFit,
    Capabilities,
    DecisionRule,
    EffectScenario,
    PowerRequest,
    RepeatedExperimentPower,
    WindowRequest,
)

OWNERSHIP: tuple[tuple[str, str], ...] = (
    (
        "causalpy/planning/contracts.py",
        "#1438 canonical types, mapping, validation matrix",
    ),
    (
        "causalpy/planning/reference.py",
        "#1438 history-only plan and synthetic-control adapter",
    ),
    ("causalpy/planning/report.py", "#1438 tables and plots of canonical results"),
    ("causalpy/planning/seams.py", "#1438 protocols and this ownership map"),
    ("causalpy/planning/__init__.py", "#1438 public planning exports"),
    (
        "causalpy/planning/power.py",
        "#1439 power, MDE, precision, repeated-data scenarios",
    ),
    (
        "causalpy/planning/adapters.py",
        "#1440 augmented and IFE adapters, groups of 1-3",
    ),
    (
        "causalpy/planning/candidates.py",
        "#1441 combination generation and screening tables",
    ),
    ("integration after 1439-1441", "#319 full-procedure audit"),
)


class EstimatorAdapter(Protocol):
    """Fit one window and translate effects without exposing the estimator."""

    def capabilities(self) -> Capabilities:
        """Return evidence-backed flags. Unset flags are not implied."""
        ...

    def fit_window(self, request: WindowRequest) -> CanonicalFit:
        """Fit one horizon. Do not bake an effect into the fit.

        Parameters
        ----------
        request : WindowRequest
            History slice, units, treatment time, and sampling settings.
        """
        ...

    def translate(
        self,
        fit: CanonicalFit,
        scenario: EffectScenario,
        decision: DecisionRule,
        *,
        fit_index: int,
        outcome_scale: str,
    ) -> CanonicalEvaluation:
        """Translate one fitted window into one canonical evaluation.

        Parameters
        ----------
        fit : CanonicalFit
            One horizon fitted without a planned effect.
        scenario : EffectScenario
            Effect to add without refitting.
        decision : DecisionRule
            One-sided rule applied to the translated estimand.
        fit_index : int
            Index of this horizon's single fit within the call.
        outcome_scale : str
            Label for the outcome units.
        """
        ...


class PowerEvaluator(Protocol):
    """Repeated-data power. Not a posterior tail probability."""

    def repeated_experiment_power(
        self,
        request: PowerRequest,
    ) -> RepeatedExperimentPower:
        """Estimate the probability of the declared analysis decision.

        Parameters
        ----------
        request : PowerRequest
            Scenario, decision, and repeated-data budget.
        """
        ...


class CandidateGenerator(Protocol):
    """Screen candidate groups. Screening does not fit a model."""

    def generate(self, request: CandidateRequest) -> CandidateTable:
        """Return plot-ready rows with no posterior and no estimator class.

        Parameters
        ----------
        request : CandidateRequest
            Constraints and the maximum number of rows.
        """
        ...
