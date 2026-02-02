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
Tests for BaseExperiment abstract method enforcement.
"""

import pytest

from causalpy.experiments.base import BaseExperiment
from causalpy.reporting import EffectSummary


def test_baseexperiment_enforces_missing_bayesian_methods():
    class IncompleteExperiment(BaseExperiment):
        supports_bayes = True
        supports_ols = False

        def effect_summary(self, *args, **kwargs) -> EffectSummary:
            raise NotImplementedError

    with pytest.raises(TypeError):
        IncompleteExperiment()
