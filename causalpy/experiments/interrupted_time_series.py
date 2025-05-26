#   Copyright 2022 - 2025 The PyMC Labs Developers
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
Interrupted Time Series Analysis (DEPRECATED)
"""

import warnings

from .structural_time_series import BasisExpansionTimeSeries


class InterruptedTimeSeries(BasisExpansionTimeSeries):
    """
    DEPRECATED: This class is deprecated and will be removed in a future version.
    Please use BasisExpansionTimeSeries instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The InterruptedTimeSeries class is deprecated and will be removed in a "
            "future version. Please use BasisExpansionTimeSeries instead.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)
