from causal_impact.ModelBuilder import ModelBuilder
import pymc as pm
import numpy as np


class WeightedSumFitter(ModelBuilder):
    def build_model(self, X, y):
        print("building model")
        with self:
            n_predictors = X.shape[1]
            X = pm.MutableData("X", X)
            y = pm.MutableData("y", y)
            beta = pm.Dirichlet("beta", a=np.ones(n_predictors))
            sigma = pm.HalfNormal("sigma", 1)
            mu = pm.Deterministic("mu", pm.math.dot(X, beta))
            pm.Normal("y_hat", mu, sigma, shape=y.eval().shape, observed=y)
