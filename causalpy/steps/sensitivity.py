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
SensitivityAnalysis pipeline step.

A container step that holds a list of pluggable ``Check`` objects and
runs them against the fitted experiment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from causalpy.checks.base import Check, CheckResult
from causalpy.experiments.base import BaseExperiment
from causalpy.pipeline import PipelineContext

logger = logging.getLogger(__name__)

# Registry mapping experiment types to their default checks.
# Populated by individual check modules via ``register_default_check``.
_DEFAULT_CHECKS: dict[type[BaseExperiment], list[type]] = {}


def register_default_check(
    check_class: type,
    experiment_types: set[type[BaseExperiment]],
) -> None:
    """Register a check class as a default for the given experiment types.

    Called by check modules at import time so that
    ``SensitivityAnalysis.default_for`` can auto-select checks.
    """
    for exp_type in experiment_types:
        _DEFAULT_CHECKS.setdefault(exp_type, []).append(check_class)


@dataclass
class SensitivitySummary:
    """Aggregate result of all sensitivity checks.

    Attributes
    ----------
    results : list[CheckResult]
        Individual check results.
    all_passed : bool or None
        ``True`` if every check with a pass/fail criterion passed, ``False``
        if any failed, or ``None`` if no check had a pass/fail criterion.
    text : str
        Combined prose summary.
    """

    results: list[CheckResult] = field(default_factory=list)
    all_passed: bool | None = None
    text: str = ""

    @classmethod
    def from_results(cls, results: list[CheckResult]) -> SensitivitySummary:
        """Build a summary from a list of check results."""
        verdicts = [r.passed for r in results if r.passed is not None]
        all_passed = all(verdicts) if verdicts else None

        texts = [r.text for r in results if r.text]
        combined_text = "\n\n".join(texts)

        return cls(results=list(results), all_passed=all_passed, text=combined_text)


class SensitivityAnalysis:
    """Pipeline step that runs a suite of sensitivity / diagnostic checks.

    Parameters
    ----------
    checks : list of Check
        The checks to run against the fitted experiment.

    Examples
    --------
    >>> import causalpy as cp  # doctest: +SKIP
    >>> step = cp.SensitivityAnalysis(  # doctest: +SKIP
    ...     checks=[
    ...         cp.checks.PlaceboInTime(n_folds=4),
    ...         cp.checks.PriorSensitivity(priors=[...]),
    ...     ]
    ... )
    """

    def __init__(self, checks: list[Any] | None = None) -> None:
        self.checks: list[Any] = list(checks) if checks else []

    @classmethod
    def default_for(cls, method: type[BaseExperiment]) -> SensitivityAnalysis:
        """Create a ``SensitivityAnalysis`` pre-loaded with all registered
        default checks for *method*.

        Parameters
        ----------
        method : type[BaseExperiment]
            The experiment class to look up defaults for.

        Returns
        -------
        SensitivityAnalysis
            Instance with applicable default checks instantiated.
        """
        check_classes = _DEFAULT_CHECKS.get(method, [])
        checks = [cc() for cc in check_classes]
        return cls(checks=checks)

    def validate(self, context: PipelineContext) -> None:
        """Validate that checks are well-formed.

        At validation time the experiment may not yet be fitted, so we
        only check structural issues (e.g. that each object satisfies the
        Check protocol).

        Raises
        ------
        TypeError
            If any item in ``checks`` does not satisfy the ``Check`` protocol.
        """
        for i, check in enumerate(self.checks):
            if not isinstance(check, Check):
                raise TypeError(
                    f"Check {i} ({type(check).__name__}) does not satisfy the "
                    f"Check protocol"
                )

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run all checks against the fitted experiment.

        Raises
        ------
        RuntimeError
            If no experiment has been fitted (``context.experiment is None``).
        TypeError
            If a check is not applicable to the experiment type.
        """
        if context.experiment is None:
            raise RuntimeError(
                "SensitivityAnalysis requires a fitted experiment in the "
                "pipeline context. Add an EstimateEffect step before "
                "SensitivityAnalysis."
            )

        experiment = context.experiment
        experiment_type = type(experiment)
        results: list[CheckResult] = []

        for check in self.checks:
            if experiment_type not in check.applicable_methods:
                raise TypeError(
                    f"{type(check).__name__} is not applicable to "
                    f"{experiment_type.__name__}. Applicable methods: "
                    f"{[m.__name__ for m in check.applicable_methods]}"
                )
            check.validate(experiment)
            logger.info("Running check: %s", type(check).__name__)
            result = check.run(experiment, context)
            results.append(result)

        summary = SensitivitySummary.from_results(results)
        context.sensitivity_results = results
        context.report = summary  # overwritten by GenerateReport if present

        return context

    def __repr__(self) -> str:
        check_names = [type(c).__name__ for c in self.checks]
        return f"SensitivityAnalysis(checks={check_names})"
