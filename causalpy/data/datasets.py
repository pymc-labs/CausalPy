import os
import pathlib

import pandas as pd

import causalpy as cp

DATASETS = {
    "banks": {"filename": "banks.csv"},
    "did": {"filename": "did.csv"},
    "drinking": {"filename": "drinking.csv"},
    "its": {"filename": "its.csv"},
    "its simple": {"filename": "its_simple.csv"},
    "rd": {"filename": "regression_discontinuity.csv"},
    "sc": {"filename": "synthetic_control.csv"},
}


def get_data_home():
    """Return the path of the data directory"""
    return pathlib.Path(cp.__file__).parents[1] / "causalpy" / "data"


def load_data(dataset: str = None):

    if dataset in DATASETS:

        data_dir = get_data_home()
        datafile = DATASETS[dataset]
        file_path = data_dir / datafile["filename"]
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Dataset {dataset} not found!")
