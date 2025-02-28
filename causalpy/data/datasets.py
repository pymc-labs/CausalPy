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
Functions to load example datasets
"""

import pathlib

import pandas as pd

import causalpy as cp

DATASETS = {
    "banks": {"filename": "banks.csv"},
    "brexit": {"filename": "GDP_in_dollars_billions.csv"},
    "covid": {"filename": "deaths_and_temps_england_wales.csv"},
    "did": {"filename": "did.csv"},
    "drinking": {"filename": "drinking.csv"},
    "its": {"filename": "its.csv"},
    "its simple": {"filename": "its_simple.csv"},
    "rd": {"filename": "regression_discontinuity.csv"},
    "sc": {"filename": "synthetic_control.csv"},
    "anova1": {"filename": "ancova_generated.csv"},
    "geolift1": {"filename": "geolift1.csv"},
    "geolift_multi_cell": {"filename": "geolift_multi_cell.csv"},
    "risk": {"filename": "AJR2001.csv"},
    "nhefs": {"filename": "nhefs.csv"},
    "schoolReturns": {"filename": "schoolingReturns.csv"},
}


def _get_data_home() -> pathlib.PosixPath:
    """Return the path of the data directory"""
    return pathlib.Path(cp.__file__).parents[1] / "causalpy" / "data"


def load_data(dataset: str = None) -> pd.DataFrame:
    """Loads the requested dataset and returns a pandas DataFrame.

    :param dataset: The desired dataset to load
    """

    if dataset in DATASETS:
        data_dir = _get_data_home()
        datafile = DATASETS[dataset]
        file_path = data_dir / datafile["filename"]
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Dataset {dataset} not found!")
