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
Tests for the simulated data functions
"""

import numpy as np
import pandas as pd


def test_generate_multicell_geolift_data():
    """
    Test the generate_multicell_geolift_data function.
    """
    from causalpy.data.simulate_data import generate_multicell_geolift_data

    df = generate_multicell_geolift_data()
    assert isinstance(df, pd.DataFrame)
    assert np.all(df >= 0), "Found negative values in dataset"


def test_generate_geolift_data():
    """
    Test the generate_geolift_data function.
    """
    from causalpy.data.simulate_data import generate_geolift_data

    df = generate_geolift_data()
    assert isinstance(df, pd.DataFrame)
    assert np.all(df >= 0), "Found negative values in dataset"
