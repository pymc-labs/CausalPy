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
Base classes for sensitivity / diagnostic checks.

Every check implements the :class:`Check` protocol and returns a
:class:`CheckResult`.  Checks declare which experiment types they
apply to via ``applicable_methods``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from causalpy.experiments.base import BaseExperiment
from causalpy.pipeline import PipelineContext


def clone_model(model: Any) -> Any:
    """Create a fresh, unfitted copy of a model.

    PyMC models cannot survive ``copy.deepcopy`` (the class identity is
    lost), so we use their ``_clone()`` method instead.  For all other
    model types we fall back to ``copy.deepcopy``.
    """
    if hasattr(model, "_clone"):
        return model._clone()
    return copy.deepcopy(model)


@dataclass
class CheckResult:
    """Result of a single sensitivity / diagnostic check.

    Attributes
    ----------
    check_name : str
        Human-readable name of the check.
    passed : bool or None
        ``True`` if the check passed, ``False`` if it failed, or ``None``
        if the check is purely informational (no pass/fail criterion).
    table : pd.DataFrame or None
        Optional diagnostic statistics table.
    text : str
        Prose summary of the check result.
    figures : list
        Optional matplotlib figures produced by the check.
    metadata : dict
        Arbitrary extra data that downstream steps (e.g. ``GenerateReport``)
        can use.
    """

    check_name: str
    passed: bool | None = None
    table: pd.DataFrame | None = None
    text: str = ""
    figures: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Check(Protocol):
    """Protocol that individual sensitivity checks must satisfy.

    Attributes
    ----------
    applicable_methods : set[type[BaseExperiment]]
        Experiment classes this check can be applied to.
    """

    applicable_methods: set[type[BaseExperiment]]

    def validate(self, experiment: BaseExperiment) -> None:
        """Verify the check is applicable to the given experiment.

        Raises
        ------
        TypeError
            If the experiment type is not in ``applicable_methods``.
        """
        ...

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext,
    ) -> CheckResult:
        """Execute the check and return a result.

        Parameters
        ----------
        experiment : BaseExperiment
            The fitted experiment to check.
        context : PipelineContext
            The pipeline context (provides experiment_config, data, etc.).

        Returns
        -------
        CheckResult
        """
        ...
