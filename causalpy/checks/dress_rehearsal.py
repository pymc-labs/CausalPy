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
"""
Dress rehearsal diagnostic check for Synthetic Control experiments.

Wraps ``SyntheticControl.validate_design()`` as a ``Check`` for use
in the sensitivity analysis pipeline.
"""

from __future__ import annotations

from typing import Literal

from causalpy.checks.base import CheckResult
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pipeline import PipelineContext


class DressRehearsalCheck:
    """Pipeline-compatible dress rehearsal check for Synthetic Control.

    Calls :meth:`SyntheticControl.validate_design` and wraps the
    result as a :class:`CheckResult`.

    Parameters
    ----------
    injected_effect : float
        Effect to inject (see ``validate_design``).
    holdout_periods : int or None
        Pseudo-post window length.
    effect_type : {"relative", "absolute"}
        How the injected effect is applied.
    sample_kwargs : dict or None
        MCMC sampling arguments for the refitted model.
    """

    applicable_methods: set[type[BaseExperiment]] = {SyntheticControl}

    def __init__(
        self,
        injected_effect: float = 0.10,
        holdout_periods: int | None = None,
        effect_type: Literal["relative", "absolute"] = "relative",
        sample_kwargs: dict | None = None,
    ) -> None:
        self.injected_effect = injected_effect
        self.holdout_periods = holdout_periods
        self.effect_type = effect_type
        self.sample_kwargs = sample_kwargs

    def validate(self, experiment: BaseExperiment) -> None:
        """Verify the experiment is a SyntheticControl with a PyMC model."""
        if not isinstance(experiment, SyntheticControl):
            raise TypeError(
                "DressRehearsalCheck requires a SyntheticControl experiment."
            )
        from causalpy.pymc_models import PyMCModel

        if not isinstance(experiment.model, PyMCModel):
            raise TypeError(
                "DressRehearsalCheck requires a PyMC model for posterior extraction."
            )

    def run(
        self,
        experiment: BaseExperiment,
        context: PipelineContext,
    ) -> CheckResult:
        """Run the dress rehearsal and return a ``CheckResult``."""
        sc: SyntheticControl = experiment  # type: ignore[assignment]
        result = sc.validate_design(
            injected_effect=self.injected_effect,
            holdout_periods=self.holdout_periods,
            effect_type=self.effect_type,
            sample_kwargs=self.sample_kwargs,
        )
        return result.to_check_result()

    def __repr__(self) -> str:
        """Return a readable string representation."""
        return (
            f"DressRehearsalCheck(injected_effect={self.injected_effect}, "
            f"effect_type='{self.effect_type}')"
        )
