import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest

import causalpy as cp
from causalpy.pymc_models import ModelBuilder

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


class MyToyModel(ModelBuilder):
    def build_model(self, X, y, coords):
        with self:
            X_ = pm.MutableData(name="X", value=X)
            y_ = pm.MutableData(name="y", value=y)
            beta = pm.Normal("beta", mu=0, sigma=1, shape=X_.shape[1])
            sigma = pm.HalfNormal("sigma", sigma=1)
            mu = pm.Deterministic("mu", pm.math.dot(X_, beta))
            pm.Normal("y_hat", mu=mu, sigma=sigma, observed=y_)


class TestModelBuilder:
    def test_init(self):
        mb = ModelBuilder()
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
        with pytest.raises(
            NotImplementedError, match="This method must be implemented by a subclass"
        ):
            ModelBuilder().build_model(X=X, y=y, coords=coords)

    def test_fit_build_not_implemented(self):
        with pytest.raises(
            NotImplementedError, match="This method must be implemented by a subclass"
        ):
            ModelBuilder().fit(X=np.ones(2), y=np.ones(3), coords={"a": 1})

    @pytest.mark.parametrize(
        argnames="coords",
        argvalues=[None, {"a": 1}],
        ids=["None-coords", "dict-coords"],
    )
    def test_fit_predict(self, coords, rng) -> None:
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
    result = cp.pymc_experiments.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group + t + treated:group",
        time_variable_name="t",
        group_variable_name="group",
        treated=1,
        untreated=0,
        prediction_model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert hasattr(result, "idata")
    assert isinstance(result.idata, az.InferenceData)
