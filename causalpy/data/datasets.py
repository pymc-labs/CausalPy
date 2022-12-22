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
