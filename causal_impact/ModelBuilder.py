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

    def build_model(self, X, y):
        raise NotImplementedError

    def _data_setter(self, X):
        with self.model:
            pm.set_data({"X": X})

    def fit(self, X, y):
        self.build_model(X, y)
        with self.model:
            self.idata = pm.sample()
            print("pm.sample DONE")
            self.idata.extend(pm.sample_prior_predictive())
            print("sample_prior_predictive DONE")
            self.idata.extend(pm.sample_posterior_predictive(self.idata))
            print("sample_posterior_predictive DONE")
        return self.idata

    def predict(self, X):
        self._data_setter(X)
        print("_data_setter DONE")
        with self.model:  # sample with new input data
            print("pm.sample_posterior_predictive")
            post_pred = pm.sample_posterior_predictive(self.idata)
        return post_pred

    def score(self, X, y):
        return 0.0
