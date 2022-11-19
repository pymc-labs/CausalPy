import arviz as az
import numpy as np
import pymc as pm
from arviz import r2_score


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
        """Draw samples from posterior, prior predictive, and posterior predictive distributions."""
        self.build_model(X, y, coords)
        with self.model:
            self.idata = pm.sample()
            self.idata.extend(pm.sample_prior_predictive())
            self.idata.extend(pm.sample_posterior_predictive(self.idata))
        return self.idata

    def predict(self, X):
        """Predict data given input data `X`"""
        self._data_setter(X)
        with self.model:  # sample with new input data
            post_pred = pm.sample_posterior_predictive(
                self.idata, var_names=["y_hat", "mu"]
            )
        return post_pred

    def score(self, X, y):
        """Score the Bayesian :math:`R^2` given inputs ``X`` and outputs ``y``.

        .. caution::

           The Bayesian :math:`R^2` is not the same as the traditional coefficient of determination, https://en.wikipedia.org/wiki/Coefficient_of_determination.

        """
        yhat = self.predict(X)
        yhat = az.extract(
            yhat, group="posterior_predictive", var_names="y_hat"
        ).T.values
        # Note: First argument must be a 1D array
        return r2_score(y.flatten(), yhat)

    # .stack(sample=("chain", "draw")


class WeightedSumFitter(ModelBuilder):
    """Used for synthetic control experiments"""

    def build_model(self, X, y, coords):
        """Defines the PyMC model"""
        with self:
            self.add_coords(coords)
            n_predictors = X.shape[1]
            X = pm.MutableData("X", X, dims=["obs_ind", "coeffs"])
            y = pm.MutableData("y", y[:, 0], dims="obs_ind")
            beta = pm.Dirichlet("beta", a=np.ones(n_predictors), dims="coeffs")
            sigma = pm.HalfNormal("sigma", 1)
            mu = pm.Deterministic("mu", pm.math.dot(X, beta), dims="obs_ind")
            pm.Normal("y_hat", mu, sigma, observed=y, dims="obs_ind")


class LinearRegression(ModelBuilder):
    """Custom PyMC model for linear regression"""

    def build_model(self, X, y, coords):
        """Defines the PyMC model"""
        with self:
            self.add_coords(coords)
            X = pm.MutableData("X", X, dims=["obs_ind", "coeffs"])
            y = pm.MutableData("y", y[:, 0], dims="obs_ind")
            beta = pm.Normal("beta", 0, 50, dims="coeffs")
            sigma = pm.HalfNormal("sigma", 1)
            mu = pm.Deterministic("mu", pm.math.dot(X, beta), dims="obs_ind")
            pm.Normal("y_hat", mu, sigma, observed=y, dims="obs_ind")
