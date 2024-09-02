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
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest

import causalpy as cp
from causalpy.pymc_models import PyMCModel

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


class MyToyModel(PyMCModel):
    """
    A subclass of PyMCModel with a simple regression model for use in tests.
    """

    def build_model(self, X, y, coords):
        """
        Required .build_model() method of a PyMCModel subclass

        This is a basic 1-variable linear regression model for use in tests.
        """
        with self:
            X_ = pm.Data(name="X", value=X)
            y_ = pm.Data(name="y", value=y)
            beta = pm.Normal("beta", mu=0, sigma=1, shape=X_.shape[1])
            sigma = pm.HalfNormal("sigma", sigma=1)
            mu = pm.Deterministic("mu", pm.math.dot(X_, beta))
            pm.Normal("y_hat", mu=mu, sigma=sigma, observed=y_)


class TestPyMCModel:
    """
    Related tests that check aspects of PyMCModel objects.
    """

    def test_init(self):
        """
        Test initialization.

        Creates PyMCModel() object and checks that idata is None and no sample
        kwargs are specified.
        """
        mb = PyMCModel()
        assert mb.idata is None
        assert mb.sample_kwargs == {}

    @pytest.mark.parametrize(
        argnames="coords", argvalues=[{"a": 1}, None], ids=["coords-dict", "coord-None"]
    )
    @pytest.mark.parametrize(
        argnames="y", argvalues=[np.ones(3), None], ids=["y-array", "y-None"]
    )
    @pytest.mark.parametrize(
        argnames="X", argvalues=[np.ones(2), None], ids=["X-array", "X-None"]
    )
    def test_model_builder(self, X, y, coords) -> None:
        """
        Tests that a PyMCModel() object without a .build_model() method raises
        appropriate exception.
        """
        with pytest.raises(
            NotImplementedError, match="This method must be implemented by a subclass"
        ):
            PyMCModel().build_model(X=X, y=y, coords=coords)

    def test_fit_build_not_implemented(self):
        """
        Tests that a PyMCModel() object without a .fit() method raises appropriate
        exception.
        """
        with pytest.raises(
            NotImplementedError, match="This method must be implemented by a subclass"
        ):
            PyMCModel().fit(X=np.ones(2), y=np.ones(3), coords={"a": 1})

    @pytest.mark.parametrize(
        argnames="coords",
        argvalues=[None, {"a": 1}],
        ids=["None-coords", "dict-coords"],
    )
    def test_fit_predict(self, coords, rng) -> None:
        """
        Test fit and predict methods on MyToyModel.

        Generates normal data, fits the model, makes predictions, scores the model
        then:
        1. checks that model.idata is az.InferenceData type
        2. checks that beta, sigma, mu, and y_hat can be extract from idata
        3. checks score is a pandas series of the correct shape
        4. checks that predictions are az.InferenceData type
        """
        X = rng.normal(loc=0, scale=1, size=(20, 2))
        y = rng.normal(loc=0, scale=1, size=(20,))
        model = MyToyModel(sample_kwargs={"chains": 2, "draws": 2})
        model.fit(X, y, coords=coords)
        predictions = model.predict(X=X)
        score = model.score(X=X, y=y)
        assert isinstance(model.idata, az.InferenceData)
        assert az.extract(data=model.idata, var_names=["beta"]).shape == (2, 2 * 2)
        assert az.extract(data=model.idata, var_names=["sigma"]).shape == (2 * 2,)
        assert az.extract(data=model.idata, var_names=["mu"]).shape == (20, 2 * 2)
        assert az.extract(
            data=model.idata, group="posterior_predictive", var_names=["y_hat"]
        ).shape == (20, 2 * 2)
        assert isinstance(score, pd.Series)
        assert score.shape == (2,)
        assert isinstance(predictions, az.InferenceData)


def test_idata_property():
    """Test that we can access the idata property of the model"""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group + t + group:post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert hasattr(result, "idata")
    assert isinstance(result.idata, az.InferenceData)


seeds = [1234, 42, 123456789]


@pytest.mark.parametrize("seed", seeds)
def test_result_reproducibility(seed):
    """Test that we can reproduce the results from the model. We could in theory test
    this with all the model and experiment types, but what is being targeted is
    the PyMCModel.fit method, so we should be safe testing with just one model. Here
    we use the DifferenceInDifferences experiment class."""
    # Load the data
    df = cp.load_data("did")
    # Set a random seed
    sample_kwargs["random_seed"] = seed
    # Calculate the result twice
    result1 = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group + t + group:post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    result2 = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group + t + group:post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert np.all(result1.idata.posterior.mu == result2.idata.posterior.mu)
    assert np.all(result1.idata.prior.mu == result2.idata.prior.mu)
    assert np.all(
        result1.idata.prior_predictive.y_hat == result2.idata.prior_predictive.y_hat
    )
