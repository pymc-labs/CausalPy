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
EstimateEffect pipeline step.

Wraps experiment construction as a deferred configuration object so that
the pipeline can validate all steps before executing any fitting.
"""

from __future__ import annotations

import logging
from typing import Any

from causalpy.experiments.base import BaseExperiment
from causalpy.pipeline import PipelineContext

logger = logging.getLogger(__name__)


class EstimateEffect:
    """Pipeline step that fits a causal experiment.

    Captures the experiment class and its keyword arguments.  When the
    pipeline runs, instantiates the experiment with the pipeline's data
    (which triggers fitting) and stores the result in the context.

    Parameters
    ----------
    method : type[BaseExperiment]
        The experiment class to instantiate (e.g. ``cp.InterruptedTimeSeries``).
    **kwargs
        Keyword arguments forwarded to the experiment constructor.  The
        ``data`` argument is supplied by the pipeline and must *not* be
        included here.

    Examples
    --------
    >>> import causalpy as cp  # doctest: +SKIP
    >>> step = cp.EstimateEffect(  # doctest: +SKIP
    ...     method=cp.InterruptedTimeSeries,
    ...     treatment_time=pd.Timestamp("2020-01-01"),
    ...     formula="y ~ 1 + t",
    ...     model=cp.pymc_models.LinearRegression(),
    ... )
    """

    def __init__(self, method: type[BaseExperiment], **kwargs: Any) -> None:
        self.method = method
        self.kwargs = kwargs

    def validate(self, context: PipelineContext) -> None:
        """Check that the step is properly configured.

        Raises
        ------
        TypeError
            If *method* is not a subclass of ``BaseExperiment``.
        ValueError
            If ``data`` is passed in kwargs (it comes from the pipeline).
        """
        if not (
            isinstance(self.method, type) and issubclass(self.method, BaseExperiment)
        ):
            raise TypeError(
                f"method must be a BaseExperiment subclass, got {self.method!r}"
            )
        if "data" in self.kwargs:
            raise ValueError(
                "Do not pass 'data' to EstimateEffect; it is supplied by the Pipeline."
            )

    def run(self, context: PipelineContext) -> PipelineContext:
        """Instantiate and fit the experiment.

        The experiment constructor receives ``context.data`` as its first
        positional argument, followed by all captured keyword arguments.

        Returns
        -------
        PipelineContext
            Updated context with ``experiment``, ``experiment_config``,
            and (if available) ``effect_summary`` populated.
        """
        logger.info("Fitting %s", self.method.__name__)
        experiment = self.method(context.data, **self.kwargs)

        context.experiment = experiment
        context.experiment_config = {
            "method": self.method,
            **self.kwargs,
        }

        try:
            context.effect_summary = experiment.effect_summary()
        except NotImplementedError as exc:
            logger.debug(
                "effect_summary() not available for %s: %s",
                self.method.__name__,
                exc,
            )

        return context

    def __repr__(self) -> str:
        """Return a string representation of the step."""
        kwarg_str = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"EstimateEffect(method={self.method.__name__}, {kwarg_str})"
