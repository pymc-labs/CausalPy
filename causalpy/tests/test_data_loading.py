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
Tests that example data can be loaded into data frames.
"""

import pandas as pd
import pytest

import causalpy as cp
from causalpy.data.datasets import REAL_WORLD_DATASETS, SYNTHETIC_DATASETS

all_datasets = list(SYNTHETIC_DATASETS.keys()) + list(REAL_WORLD_DATASETS.keys())


@pytest.mark.parametrize("dataset_name", all_datasets)
def test_data_loading(dataset_name):
    """Checks that test data can be loaded into data frames."""
    df = cp.load_data(dataset_name)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.parametrize("dataset_name", list(SYNTHETIC_DATASETS.keys()))
def test_synthetic_data_no_nulls(dataset_name):
    """Synthetic datasets should never contain missing values."""
    df = cp.load_data(dataset_name)
    assert df.isnull().sum().sum() == 0


def test_synthetic_data_reproducibility():
    """Verify that synthetic datasets are deterministic across calls."""
    for key in SYNTHETIC_DATASETS:
        df1 = cp.load_data(key)
        df2 = cp.load_data(key)
        assert df1.equals(df2), f"load_data({key!r}) is not reproducible"


def test_unknown_dataset_raises():
    """Verify that requesting a nonexistent dataset raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        cp.load_data("nonexistent_dataset")
