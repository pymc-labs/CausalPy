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
Tests that example data can be loaded into data frames.
"""

import pandas as pd
import pytest

import causalpy as cp

tests = [
    "banks",
    "brexit",
    "covid",
    "did",
    "drinking",
    "its",
    "its simple",
    "rd",
    "sc",
    "anova1",
]


@pytest.mark.parametrize("dataset_name", tests)
def test_data_loading(dataset_name):
    """
    Checks that test data can be loaded into data frames and that there are no
    missing values in any column.
    """
    df = cp.load_data(dataset_name)
    assert isinstance(df, pd.DataFrame)
    # Check that there are no missing values in any column
    assert df.isnull().sum().sum() == 0
