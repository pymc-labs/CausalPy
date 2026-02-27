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
Pipeline orchestration for composable causal inference workflows.

Provides a ``Pipeline`` class that chains steps (``EstimateEffect``,
``SensitivityAnalysis``, ``GenerateReport``) into a reproducible,
lazily-validated workflow.  All steps are validated before any fitting
begins so that configuration errors surface before expensive MCMC sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from causalpy.experiments.base import BaseExperiment
from causalpy.reporting import EffectSummary


@dataclass
class PipelineContext:
    """Mutable container that accumulates results as pipeline steps execute.

    Each step reads from and writes to this context, building up a complete
    record of the analysis.

    Attributes
    ----------
    data : pd.DataFrame
        The input dataset.
    experiment : BaseExperiment or None
        The fitted experiment object, populated by ``EstimateEffect``.
    experiment_config : dict or None
        The configuration used to create the experiment (method class +
        keyword arguments), so that downstream steps like
        ``SensitivityAnalysis`` can derive experiment factories.
    effect_summary : EffectSummary or None
        The effect summary from the primary experiment.
    sensitivity_results : list
        Accumulated sensitivity / diagnostic check results.
    report : object or None
        Generated report artifact, populated by ``GenerateReport``.
    """

    data: pd.DataFrame
    experiment: BaseExperiment | None = None
    experiment_config: dict[str, Any] | None = None
    effect_summary: EffectSummary | None = None
    sensitivity_results: list[Any] = field(default_factory=list)
    report: Any = None


@dataclass
class PipelineResult:
    """Immutable result returned by :meth:`Pipeline.run`.

    Attributes
    ----------
    experiment : BaseExperiment or None
        The fitted experiment.
    effect_summary : EffectSummary or None
        The effect summary from the experiment.
    sensitivity_results : list
        Results of all sensitivity / diagnostic checks.
    report : object or None
        Generated report artifact.
    """

    experiment: BaseExperiment | None
    effect_summary: EffectSummary | None
    sensitivity_results: list[Any]
    report: Any

    @classmethod
    def from_context(cls, context: PipelineContext) -> PipelineResult:
        """Build a ``PipelineResult`` from a completed ``PipelineContext``."""
        return cls(
            experiment=context.experiment,
            effect_summary=context.effect_summary,
            sensitivity_results=list(context.sensitivity_results),
            report=context.report,
        )


@runtime_checkable
class Step(Protocol):
    """Protocol that all pipeline steps must satisfy.

    Implementations must provide two methods:

    * ``validate`` -- called *before* any step runs.  Should raise on
      configuration errors (wrong types, missing parameters, etc.).
    * ``run`` -- called sequentially.  Receives the shared
      ``PipelineContext``, mutates it, and returns it.
    """

    def validate(self, context: PipelineContext) -> None: ...

    def run(self, context: PipelineContext) -> PipelineContext: ...


class Pipeline:
    """Orchestrate a sequence of causal-inference steps.

    The pipeline validates *all* steps before executing any of them,
    ensuring configuration errors are caught before potentially expensive
    model fitting.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to analyse.
    steps : list of Step
        Ordered sequence of pipeline steps.

    Examples
    --------
    >>> import causalpy as cp  # doctest: +SKIP
    >>> result = cp.Pipeline(
    ...     data=df,
    ...     steps=[
    ...         cp.EstimateEffect(
    ...             method=cp.InterruptedTimeSeries,
    ...             treatment_time=pd.Timestamp("2020-01-01"),
    ...             formula="y ~ 1 + t",
    ...             model=cp.pymc_models.LinearRegression(),
    ...         ),
    ...     ],
    ... ).run()
    """

    def __init__(self, data: pd.DataFrame, steps: list[Step]) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"data must be a pandas DataFrame, got {type(data).__name__}"
            )
        if not steps:
            raise ValueError("steps must be a non-empty list")
        for i, step in enumerate(steps):
            if not isinstance(step, Step):
                raise TypeError(
                    f"Step {i} ({type(step).__name__}) does not satisfy the "
                    f"Step protocol (must implement validate and run)"
                )
        self.data = data
        self.steps = list(steps)

    def run(self) -> PipelineResult:
        """Validate all steps, then execute them sequentially.

        Returns
        -------
        PipelineResult
            The accumulated results of the pipeline.

        Raises
        ------
        Exception
            Re-raises any exception from validation or step execution.
        """
        context = PipelineContext(data=self.data)

        for step in self.steps:
            step.validate(context)

        for step in self.steps:
            context = step.run(context)

        return PipelineResult.from_context(context)
