import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


class ModelBuilder(pm.Model):
    """
    This is a wrapper around pm.Model to give scikit-learn like API
    """

    def __init__(self):
        super().__init__()
        self.idata = None

    def build_model(self, X, y, coords):
        raise NotImplementedError

    def _data_setter(self, X):
        with self.model:
            pm.set_data({"X": X})

    def fit(self, X, y, coords):
        self.build_model(X, y, coords)
        with self.model:
            self.idata = pm.sample()
            self.idata.extend(pm.sample_prior_predictive())
            self.idata.extend(pm.sample_posterior_predictive(self.idata))
        return self.idata

    def predict(self, X):
        self._data_setter(X)
        with self.model:  # sample with new input data
            post_pred = pm.sample_posterior_predictive(self.idata)
        return post_pred

    def score(self, X, y):
        return 0.0
