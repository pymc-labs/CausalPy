from typing import Any, Dict, Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from arviz import r2_score


class ModelBuilder(pm.Model):
    """
    This is a wrapper around pm.Model to give scikit-learn like API.
    """

    def __init__(self, sample_kwargs: Optional[Dict[str, Any]] = None):
        """
        :param sample_kwargs: A dictionary of kwargs that get unpacked and passed to the
            :func:`pymc.sample` function. Defaults to an empty dictionary.
        """
        super().__init__()
        self.idata = None
        self.sample_kwargs = sample_kwargs if sample_kwargs is not None else {}

    def build_model(self, X, y, coords) -> None:
        """Build the model.

        Example
        -------
        >>> class CausalPyModel(ModelBuilder):
        >>>    def build_model(self, X, y):
        >>>        with self:
        >>>            X_ = pm.MutableData(name="X", value=X)
        >>>            y_ = pm.MutableData(name="y", value=y)
        >>>            beta = pm.Normal("beta", mu=0, sigma=1, shape=X_.shape[1])
        >>>            sigma = pm.HalfNormal("sigma", sigma=1)
        >>>            mu = pm.Deterministic("mu", pm.math.dot(X_, beta))
        >>>            pm.Normal("y_hat", mu=mu, sigma=sigma, observed=y_)
        """
        raise NotImplementedError("This method must be implemented by a subclass")

    def _data_setter(self, X) -> None:
        with self.model:
            pm.set_data({"X": X})

    def fit(self, X, y, coords: Optional[Dict[str, Any]] = None) -> None:
        """Draw samples from posterior, prior predictive, and posterior predictive
        distributions.
        """
        self.build_model(X, y, coords)
        with self.model:
            self.idata = pm.sample(**self.sample_kwargs)
            self.idata.extend(pm.sample_prior_predictive())
            self.idata.extend(
                pm.sample_posterior_predictive(self.idata, progressbar=False)
            )
        return self.idata

    def predict(self, X):
        """Predict data given input data `X`"""
        self._data_setter(X)
        with self.model:  # sample with new input data
            post_pred = pm.sample_posterior_predictive(
                self.idata, var_names=["y_hat", "mu"], progressbar=False
            )
        return post_pred

    def score(self, X, y) -> pd.Series:
        """Score the Bayesian :math:`R^2` given inputs ``X`` and outputs ``y``.

        .. caution::

            The Bayesian :math:`R^2` is not the same as the traditional coefficient of
            determination, https://en.wikipedia.org/wiki/Coefficient_of_determination.

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
            # TODO: There we should allow user-specified priors here
            beta = pm.Dirichlet("beta", a=np.ones(n_predictors), dims="coeffs")
            # beta = pm.Dirichlet(
            #     name="beta", a=(1 / n_predictors) * np.ones(n_predictors),
            #     dims="coeffs"
            # )
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


class InstrumentalVariableRegression(ModelBuilder):
    """Custom PyMC model for instrumental linear regression"""

    def build_model(self, X, Z, y, t, coords, priors):
        """Specify model with treatment regression and focal regression data and priors

        :param X: A pandas dataframe used to predict our outcome y
        :param Z: A pandas dataframe used to predict our treatment variable t
        :param y: An array of values representing our focal outcome y
        :param t: An array of values representing the treatment t of
                  which we're interested in estimating the causal impact
        :param coords: A dictionary with the coordinate names for our
                       instruments and covariates
        :param priors: An optional dictionary of priors for the mus and
                      sigmas of both regressions

        :code:`priors = {"mus": [0, 0], "sigmas": [1, 1], "eta": 2, "lkj_sd": 2}`

        """

        # --- Priors ---
        with self:
            self.add_coords(coords)
            beta_t = pm.Normal(
                name="beta_t",
                mu=priors["mus"][0],
                sigma=priors["sigmas"][0],
                dims="instruments",
            )
            beta_z = pm.Normal(
                name="beta_z",
                mu=priors["mus"][1],
                sigma=priors["sigmas"][1],
                dims="covariates",
            )
            sd_dist = pm.HalfCauchy.dist(beta=priors["lkj_sd"], shape=2)
            chol, corr, sigmas = pm.LKJCholeskyCov(
                name="chol_cov",
                eta=priors["eta"],
                n=2,
                sd_dist=sd_dist,
            )
            # compute and store the covariance matrix
            pm.Deterministic(name="cov", var=pt.dot(l=chol, r=chol.T))

            # --- Parameterization ---
            mu_y = pm.Deterministic(name="mu_y", var=pm.math.dot(X, beta_z))
            # focal regression
            mu_t = pm.Deterministic(name="mu_t", var=pm.math.dot(Z, beta_t))
            # instrumental regression
            mu = pm.Deterministic(name="mu", var=pt.stack(tensors=(mu_y, mu_t), axis=1))

            # --- Likelihood ---
            pm.MvNormal(
                name="likelihood",
                mu=mu,
                chol=chol,
                observed=np.stack(arrays=(y.flatten(), t.flatten()), axis=1),
                shape=(X.shape[0], 2),
            )

    def fit(self, X, Z, y, t, coords, priors):
        """Draw samples from posterior, prior predictive, and posterior predictive
        distributions.
        """
        self.build_model(X, Z, y, t, coords, priors)
        with self.model:
            self.idata = pm.sample(**self.sample_kwargs)
            self.idata.extend(pm.sample_prior_predictive())
            self.idata.extend(
                pm.sample_posterior_predictive(self.idata, progressbar=False)
            )
        return self.idata
