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
"""Internal result bundles for experiment classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class Scenario:
    """A prediction scenario: design inputs plus model output."""

    inputs: pd.DataFrame
    prediction: Any


@dataclass
class CausalResult:
    """Fitted counterfactual comparison for a pre/post time-series experiment."""

    predictions_pre: Any
    predictions_post: Any
    impact_pre: Any
    impact_post: Any
    impact_post_cumulative: Any
    score: Any | None = None


@dataclass
class PriorCalibration:
    """OLS fits used to calibrate default IV priors."""

    first_stage: Any
    second_stage: Any
    naive: Any
    beta_first: Any
    beta_second: Any
    beta_naive: dict[str, Any]
