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
    "pisa18": {"filename": "PISA18sampleScale.csv"},
    "nets": {"filename": "nets_df.csv"},
    "lalonde": {"filename": "lalonde.csv"},
    "zipcodes": {"filename": "zipcodes_data.csv"},
    "nevo": {"filename": "data_nevo.csv"},
}


def _get_data_home() -> pathlib.Path:
    """Return the path of the data directory"""
    return pathlib.Path(cp.__file__).parents[1] / "causalpy" / "data"


def load_data(dataset: str) -> pd.DataFrame:
    """Load example datasets for causal inference analysis.

    This function loads pre-packaged datasets that are used in CausalPy's
    documentation and examples. These datasets demonstrate various causal
    inference methods including difference-in-differences, regression
    discontinuity, synthetic control, interrupted time series, and more.

    Parameters
    ----------
    dataset : str
        Name of the dataset to load. Available datasets are:

        - ``"banks"`` - Historic banking closures data for difference-in-differences
        - ``"brexit"`` - UK GDP data for estimating causal impact of Brexit
        - ``"covid"`` - Deaths and temperature data for England and Wales
        - ``"did"`` - Difference-in-differences example dataset
        - ``"drinking"`` - Minimum legal drinking age data for regression discontinuity
        - ``"its"`` - Interrupted time series example dataset
        - ``"its simple"`` - Simplified interrupted time series dataset
        - ``"rd"`` - Regression discontinuity example dataset
        - ``"sc"`` - Synthetic control example dataset
        - ``"anova1"`` - ANCOVA example with pre/post treatment nonequivalent groups
        - ``"geolift1"`` - Single treatment geo-lift dataset for synthetic control
        - ``"geolift_multi_cell"`` - Multi-cell geo-lift dataset for synthetic control
        - ``"risk"`` - Acemoglu, Johnson & Robinson (2001) data for instrumental variables
        - ``"nhefs"`` - National Health and Nutrition Examination Survey data
        - ``"schoolReturns"`` - Schooling returns data for instrumental variable analysis
        - ``"pisa18"`` - PISA 2018 sample data
        - ``"nets"`` - National Supported Work Demonstration dataset
        - ``"lalonde"`` - LaLonde dataset for propensity score analysis
        - ``"zipcodes"`` - Geo-experimentation zipcode data for comparative interrupted time
          series analysis. Based on synthetic data from Juan Orduz's blog post on
          `time-based regression for geo-experiments <https://juanitorduz.github.io/time_based_regression_pymc/>`_.

    Returns
    -------
    pd.DataFrame
        The requested dataset as a pandas DataFrame.

    Raises
    ------
    ValueError
        If the requested dataset name is not found in the available datasets.

    Examples
    --------
    Load the difference-in-differences example dataset:

    >>> import causalpy as cp
    >>> df = cp.load_data("did")

    Load the regression discontinuity dataset:

    >>> df = cp.load_data("rd")
    """

    if dataset in DATASETS:
        data_dir = _get_data_home()
        datafile = DATASETS[dataset]
        file_path = data_dir / datafile["filename"]
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Dataset {dataset} not found!")
