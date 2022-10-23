from causal_impact.ModelBuilder import ModelBuilder
import pymc as pm
import numpy as np


class WeightedSumFitter(ModelBuilder):
    def build_model(self, X, y, coords):
        print("building model")
        with self:
            self.add_coords(coords)
            n_predictors = X.shape[1]
            X = pm.MutableData("X", X, dims=["obs_ind", "coeffs"])
            y = pm.MutableData("y", y[:, 0], dims="obs_ind")
            beta = pm.Dirichlet("beta", a=np.ones(n_predictors))
            sigma = pm.HalfNormal("sigma", 1)
            mu = pm.Deterministic("mu", pm.math.dot(X, beta))
            pm.Normal("y_hat", mu, sigma, observed=y, dims="obs_ind")
