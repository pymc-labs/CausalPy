#   Copyright 2024 The PyMC Labs Developers
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
Utility functions
"""

import numpy as np
import pandas as pd


def _is_variable_dummy_coded(series: pd.Series) -> bool:
    """Check if a data in the provided Series is dummy coded. It should be 0 or 1
    only."""
    return len(set(series).difference(set([0, 1]))) == 0


def _series_has_2_levels(series: pd.Series) -> bool:
    """Check that the variable in the provided Series has 2 levels"""
    return len(pd.Categorical(series).categories) == 2


def round_num(n, round_to):
    """
    Return a string representing a number with `round_to` significant figures.

    Parameters
    ----------
    n : float
        number to round
    round_to : int
        number of significant figures
    """
    sig_figs = _format_sig_figs(n, round_to)
    return f"{n:.{sig_figs}g}"


def _format_sig_figs(value, default=None):
    """Get a default number of significant figures.

    Gives the integer part or `default`, whichever is bigger.

    Examples
    --------
    0.1234 --> 0.12
    1.234  --> 1.2
    12.34  --> 12
    123.4  --> 123
    """
    if default is None:
        default = 2
    if value == 0:
        return 1
    return max(int(np.log10(np.abs(value))) + 1, default)
