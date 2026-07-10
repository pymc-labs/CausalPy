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
Shared constants for the CausalPy package.
"""

#: Default probability mass for highest-density-interval (HDI) bands and
#: summary intervals across CausalPy plots and reports. Matches the
#: :func:`arviz.hdi` default of 0.94. Override on a per-call basis by passing
#: ``hdi_prob`` to the relevant method (e.g.
#: :meth:`causalpy.experiments.interrupted_time_series.InterruptedTimeSeries.analyze_persistence`),
#: or globally for ``maketables`` rendering via
#: :meth:`causalpy.experiments.base.BaseExperiment.set_maketables_options`.
HDI_PROB: float = 0.94

#: Default font size used in matplotlib legends across CausalPy plots.
LEGEND_FONT_SIZE: int = 12
