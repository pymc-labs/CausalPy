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

from pathlib import Path

import pandas as pd

from .simulate_data import (
    RANDOM_SEED,
    generate_ancova_data,
    generate_did,
    generate_geolift_data,
    generate_multicell_geolift_data,
    generate_regression_discontinuity_data,
    generate_synthetic_control_data,
    generate_time_series_data_seasonal,
    generate_time_series_data_simple,
)

_DATA_DIR = Path(__file__).parent

# Synthetic datasets are generated programmatically for reproducibility.
# .reset_index() on ITS functions because generators set date as the index,
# but the old CSV-based load_data returned date as a column.
SYNTHETIC_DATASETS: dict[str, callable] = {
    "did": lambda: generate_did(seed=RANDOM_SEED),
    "rd": lambda: generate_regression_discontinuity_data(
        true_treatment_threshold=0.5, seed=RANDOM_SEED
    ),
    "sc": lambda: generate_synthetic_control_data(seed=RANDOM_SEED)[0],
    "its": lambda: generate_time_series_data_seasonal(
        treatment_time=pd.to_datetime("2017-01-01"), seed=RANDOM_SEED
    ).reset_index(),
    "its simple": lambda: generate_time_series_data_simple(
        treatment_time=pd.to_datetime("2015-01-01"), seed=RANDOM_SEED
    ).reset_index(),
    "anova1": lambda: generate_ancova_data(seed=RANDOM_SEED),
    "geolift1": lambda: generate_geolift_data(seed=RANDOM_SEED).reset_index(),
    "geolift_multi_cell": lambda: generate_multicell_geolift_data(
        seed=RANDOM_SEED
    ).reset_index(),
}

# Real-world datasets remain as CSV files shipped with the package.
REAL_WORLD_DATASETS: dict[str, str] = {
    "banks": "banks.csv",
    "brexit": "GDP_in_dollars_billions.csv",
    "covid": "deaths_and_temps_england_wales.csv",
    "drinking": "drinking.csv",
    "risk": "AJR2001.csv",
    "nhefs": "nhefs.csv",
    "schoolReturns": "schoolingReturns.csv",
    "pisa18": "PISA18sampleScale.csv",
    "nets": "nets_df.csv",
    "lalonde": "lalonde.csv",
    "zipcodes": "zipcodes_data.csv",
    "nevo": "data_nevo.csv",
}


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
        - ``"nevo"`` - Berry, Levinsohn, and Pakes (1995) cereal data for BLP estimation

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
    if dataset in SYNTHETIC_DATASETS:
        return SYNTHETIC_DATASETS[dataset]()
    elif dataset in REAL_WORLD_DATASETS:
        return pd.read_csv(_DATA_DIR / REAL_WORLD_DATASETS[dataset])
    else:
        raise ValueError(f"Dataset {dataset!r} not found!")
