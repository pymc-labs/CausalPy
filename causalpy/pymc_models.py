"PyMC based meta-learners."
from typing import Any, Dict, Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
from arviz import r2_score
from pymc.distributions.distribution import DistributionMeta


class ModelBuilder(pm.Model):
    """
    This is a wrapper around pm.Model to give scikit-learn like API
    """

    def __init__(self, sample_kwargs: Optional[Dict[str, Any]] = None):
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


class BARTModel(ModelBuilder):
    "Class for building BART based models for meta-learners."

    def __init__(self, sample_kwargs=None, m=20, sigma=1):
        self.m = m
        self.sigma = sigma
        super().__init__(sample_kwargs)

    def build_model(self, X, y, coords=None):
        with self:
            self.add_coords(coords)
            X_ = pm.MutableData("X", X, dims=["obs_ind", "coeffs"])
            mu = pmb.BART("mu", X_, y, m=self.m, dims="obs_ind")
            pm.Normal("y_hat", mu=mu, sigma=self.sigma, observed=y, dims="obs_ind")


class LogisticRegression(ModelBuilder):
    """
    Custom PyMC model for logistic regression.

    Parameters
    ----------
    coeff_distribution :    PyMC distribution.
                Prior distribution of coefficient vector.
    distribution_kwargs
                Keyword arguments for prior distribution.
    sample_kwargs
                Keyword arguments for sampler.

    Examples
    --------
    >>> import numpy as np
    >>> import pymc as pm
    >>> from causalpy.pymc_models import LogisticRegression
    >>>
    >>> X = np.random.rand(10, 10)
    >>> y = np.random.rand(10)
    >>> m = LogisticRegression(
    >>>         coeff_distribution=pm.Cauchy,
    >>>         coeff_distribution_kwargs={"alpha": 0, "beta": 1}
    >>> )
    >>>
    >>> m.fit(X, y)
    """

    def __init__(
        self,
        sample_kwargs=None,
        coeff_distribution: DistributionMeta = pm.Normal,
        coeff_distribution_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.coeff_distribution = coeff_distribution
        if coeff_distribution_kwargs is None:
            self.coeff_distribution_kwargs = {"mu": 0, "sigma": 50}
        else:
            self.coeff_distribution_kwargs = coeff_distribution_kwargs

        super().__init__(sample_kwargs)

    def build_model(self, X, y, coords) -> None:
        with self:
            self.add_coords(coords)
            X_ = pm.MutableData("X", X, dims=["obs_ind", "coeffs"])
            beta = self.coeff_distribution(
                "beta", dims="coeffs", **self.coeff_distribution_kwargs
            )
            mu = pm.Deterministic(
                "mu", pm.math.sigmoid(pm.math.dot(X_, beta)), dims="obs_ind"
            )
            pm.Bernoulli("y_hat", mu, observed=y, dims="obs_ind")
