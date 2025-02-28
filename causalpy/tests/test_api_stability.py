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
Tests API stability for specific external calls
"""

import matplotlib.pyplot as plt
import pandas as pd

import causalpy as cp
from causalpy.experiments.prepostfit import SyntheticControl
from causalpy.pymc_models import WeightedSumFitter

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


def test_causal_inference_and_discovery_with_python_example():
    """Test example used in Alexander Molak's book 'Causal Inference and Discovery in Python'
    Chapter 11 (pages 304-307)
    """
    data = pd.read_csv(r"./causalpy/data/gt_social_media_data.csv")
    data.index = pd.to_datetime(data["date"])
    data = data.drop("date", axis=1)
    treatment_index = pd.to_datetime("2022-10-28")

    # Build the model
    model = cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs)
    assert isinstance(
        model, WeightedSumFitter
    ), "model is not an instance of WeightedSumFitter"

    formula = "twitter ~ 0 + tiktok + linkedin + instagram"

    # Run the experiment and plot results
    results = cp.pymc_experiments.SyntheticControl(
        data,
        treatment_index,
        formula=formula,
        model=model,
    )
    assert isinstance(
        results, SyntheticControl
    ), "results is not an instance of SyntheticControl"

    fig, ax = results.plot()
    assert isinstance(fig, plt.Figure)
