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
"""Custom PyMC models for causal inference"""

from typing import Any, Dict, Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from arviz import r2_score

from causalpy.utils import round_num


class PyMCModel(pm.Model):
    """A wrapper class for PyMC models. This provides a scikit-learn like interface with
    methods like `fit`, `predict`, and `score`. It also provides other methods which are
    useful for causal inference.

    Example
    -------
    >>> import causalpy as cp
    >>> import numpy as np
    >>> import pymc as pm
    >>> from causalpy.pymc_models import PyMCModel
    >>> class MyToyModel(PyMCModel):
    ...     def build_model(self, X, y, coords):
    ...         with self:
    ...             X_ = pm.Data(name="X", value=X)
    ...             y_ = pm.Data(name="y", value=y)
    ...             beta = pm.Normal("beta", mu=0, sigma=1, shape=X_.shape[1])
    ...             sigma = pm.HalfNormal("sigma", sigma=1)
    ...             mu = pm.Deterministic("mu", pm.math.dot(X_, beta))
    ...             pm.Normal("y_hat", mu=mu, sigma=sigma, observed=y_)
    >>> rng = np.random.default_rng(seed=42)
    >>> X = rng.normal(loc=0, scale=1, size=(20, 2))
    >>> y = rng.normal(loc=0, scale=1, size=(20,))
    >>> model = MyToyModel(
    ...     sample_kwargs={
    ...         "chains": 2,
    ...         "draws": 2000,
    ...         "progressbar": False,
    ...         "random_seed": 42,
    ...     }
    ... )
    >>> model.fit(X, y)
    Inference data...
    >>> model.score(X, y)  # doctest: +ELLIPSIS
    r2        ...
    r2_std    ...
    dtype: float64
    >>> X_new = rng.normal(loc=0, scale=1, size=(20, 2))
    >>> model.predict(X_new)
    Inference data...
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
        """Build the model, must be implemented by subclass."""
        raise NotImplementedError("This method must be implemented by a subclass")

    def _data_setter(self, X) -> None:
        """
        Set data for the model.

        This method is used internally to register new data for the model for
        prediction.

        NOTE: We are actively changing the `X`. Often, this matrix will have a different
        number of rows than the original data. So to make the shapes work, we need to
        update all data nodes in the model to have the correct shape. The values are not
        used, so we set them to 0. In our case, we just have data nodes X and y, but if
        in the future we get more complex models with more data nodes, then we'll need
        to update all of them - ideally programmatically.
        """
        new_no_of_observations = X.shape[0]
        with self:
            pm.set_data(
                {"X": X, "y": np.zeros(new_no_of_observations)},
                coords={"obs_ind": np.arange(new_no_of_observations)},
            )

    def fit(self, X, y, coords: Optional[Dict[str, Any]] = None) -> None:
        """Draw samples from posterior, prior predictive, and posterior predictive
        distributions, placing them in the model's idata attribute.
        """

        # Ensure random_seed is used in sample_prior_predictive() and
        # sample_posterior_predictive() if provided in sample_kwargs.
        random_seed = self.sample_kwargs.get("random_seed", None)

        self.build_model(X, y, coords)
        with self:
            self.idata = pm.sample(**self.sample_kwargs)
            self.idata.extend(pm.sample_prior_predictive(random_seed=random_seed))
            self.idata.extend(
                pm.sample_posterior_predictive(
                    self.idata, progressbar=False, random_seed=random_seed
                )
            )
        return self.idata

    def predict(self, X):
        """
        Predict data given input data `X`

        .. caution::
            Results in KeyError if model hasn't been fit.
        """

        # Ensure random_seed is used in sample_prior_predictive() and
        # sample_posterior_predictive() if provided in sample_kwargs.
        random_seed = self.sample_kwargs.get("random_seed", None)
        self._data_setter(X)
        with self:  # sample with new input data
            post_pred = pm.sample_posterior_predictive(
                self.idata,
                var_names=["y_hat", "mu"],
                progressbar=False,
                random_seed=random_seed,
            )
        return post_pred

    def score(self, X, y) -> pd.Series:
        """Score the Bayesian :math:`R^2` given inputs ``X`` and outputs ``y``.

        Note that the score is based on a comparison of the observed data ``y`` and the
        model's expected value of the data, `mu`.

        .. caution::

            The Bayesian :math:`R^2` is not the same as the traditional coefficient of
            determination, https://en.wikipedia.org/wiki/Coefficient_of_determination.

        """
        mu = self.predict(X)
        mu = az.extract(mu, group="posterior_predictive", var_names="mu").T.values
        # Note: First argument must be a 1D array
        return r2_score(y.flatten(), mu)

    def calculate_impact(self, y_true, y_pred):
        pre_data = xr.DataArray(y_true, dims=["obs_ind"])
        impact = pre_data - y_pred["posterior_predictive"]["y_hat"]
        return impact.transpose(..., "obs_ind")

    def calculate_cumulative_impact(self, impact):
        return impact.cumsum(dim="obs_ind")

    def print_coefficients(self, labels, round_to=None) -> None:
        def print_row(
            max_label_length: int, name: str, coeff_samples: xr.DataArray, round_to: int
        ) -> None:
            """Print one row of the coefficient table"""
            formatted_name = f"  {name: <{max_label_length}}"
            formatted_val = f"{round_num(coeff_samples.mean().data, round_to)}, 94% HDI [{round_num(coeff_samples.quantile(0.03).data, round_to)}, {round_num(coeff_samples.quantile(1 - 0.03).data, round_to)}]"  # noqa: E501
            print(f"  {formatted_name}  {formatted_val}")

        print("Model coefficients:")
        coeffs = az.extract(self.idata.posterior, var_names="beta")

        # Determine the width of the longest label
        max_label_length = max(len(name) for name in labels + ["sigma"])

        for name in labels:
            coeff_samples = coeffs.sel(coeffs=name)
            print_row(max_label_length, name, coeff_samples, round_to)

        # Add coefficient for measurement std
        coeff_samples = az.extract(self.idata.posterior, var_names="sigma")
        name = "sigma"
        print_row(max_label_length, name, coeff_samples, round_to)


class LinearRegression(PyMCModel):
    r"""
    Custom PyMC model for linear regression.

    Defines the PyMC model

    .. math::
        \beta &\sim \mathrm{Normal}(0, 50) \\
        \sigma &\sim \mathrm{HalfNormal}(1) \\
        \mu &= X \cdot \beta \\
        y &\sim \mathrm{Normal}(\mu, \sigma) \\

    Example
    --------
    >>> import causalpy as cp
    >>> import numpy as np
    >>> from causalpy.pymc_models import LinearRegression
    >>> rd = cp.load_data("rd")
    >>> X = rd[["x", "treated"]]
    >>> y = np.asarray(rd["y"]).reshape((rd["y"].shape[0],1))
    >>> lr = LinearRegression(sample_kwargs={"progressbar": False})
    >>> lr.fit(X, y, coords={
    ...                 'coeffs': ['x', 'treated'],
    ...                 'obs_ind': np.arange(rd.shape[0])
    ...                },
    ... )
    Inference data...
    """  # noqa: W605

    def build_model(self, X, y, coords):
        """
        Defines the PyMC model
        """
        with self:
            self.add_coords(coords)
            X = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            y = pm.Data("y", y[:, 0], dims="obs_ind")
            beta = pm.Normal("beta", 0, 50, dims="coeffs")
            sigma = pm.HalfNormal("sigma", 1)
            mu = pm.Deterministic("mu", pm.math.dot(X, beta), dims="obs_ind")
            pm.Normal("y_hat", mu, sigma, observed=y, dims="obs_ind")


class WeightedSumFitter(PyMCModel):
    r"""
    Used for synthetic control experiments.

    Defines the PyMC model:

    .. math::
        \sigma &\sim \mathrm{HalfNormal}(1) \\
        \beta &\sim \mathrm{Dirichlet}(1,...,1) \\
        \mu &= X \cdot \beta \\
        y &\sim \mathrm{Normal}(\mu, \sigma) \\

    Example
    --------
    >>> import causalpy as cp
    >>> import numpy as np
    >>> from causalpy.pymc_models import WeightedSumFitter
    >>> sc = cp.load_data("sc")
    >>> X = sc[['a', 'b', 'c', 'd', 'e', 'f', 'g']]
    >>> y = np.asarray(sc['actual']).reshape((sc.shape[0], 1))
    >>> wsf = WeightedSumFitter(sample_kwargs={"progressbar": False})
    >>> wsf.fit(X,y)
    Inference data...
    """  # noqa: W605

    def build_model(self, X, y, coords):
        """
        Defines the PyMC model
        """
        with self:
            self.add_coords(coords)
            n_predictors = X.shape[1]
            X = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            y = pm.Data("y", y[:, 0], dims="obs_ind")
            # TODO: There we should allow user-specified priors here
            beta = pm.Dirichlet("beta", a=np.ones(n_predictors), dims="coeffs")
            # beta = pm.Dirichlet(
            #     name="beta", a=(1 / n_predictors) * np.ones(n_predictors),
            #     dims="coeffs"
            # )
            sigma = pm.HalfNormal("sigma", 1)
            mu = pm.Deterministic("mu", pm.math.dot(X, beta), dims="obs_ind")
            pm.Normal("y_hat", mu, sigma, observed=y, dims="obs_ind")


class InstrumentalVariableRegression(PyMCModel):
    """Custom PyMC model for instrumental linear regression

    Example
    --------
    >>> import causalpy as cp
    >>> import numpy as np
    >>> from causalpy.pymc_models import InstrumentalVariableRegression
    >>> N = 10
    >>> e1 = np.random.normal(0, 3, N)
    >>> e2 = np.random.normal(0, 1, N)
    >>> Z = np.random.uniform(0, 1, N)
    >>> ## Ensure the endogeneity of the the treatment variable
    >>> X = -1 + 4 * Z + e2 + 2 * e1
    >>> y = 2 + 3 * X + 3 * e1
    >>> t = X.reshape(10, 1)
    >>> y = y.reshape(10, 1)
    >>> Z = np.asarray([[1, Z[i]] for i in range(0, 10)])
    >>> X = np.asarray([[1, X[i]] for i in range(0, 10)])
    >>> COORDS = {"instruments": ["Intercept", "Z"], "covariates": ["Intercept", "X"]}
    >>> sample_kwargs = {
    ...     "tune": 5,
    ...     "draws": 10,
    ...     "chains": 2,
    ...     "cores": 2,
    ...     "target_accept": 0.95,
    ...     "progressbar": False,
    ... }
    >>> iv_reg = InstrumentalVariableRegression(sample_kwargs=sample_kwargs)
    >>> iv_reg.fit(
    ...     X,
    ...     Z,
    ...     y,
    ...     t,
    ...     COORDS,
    ...     {
    ...         "mus": [[-2, 4], [0.5, 3]],
    ...         "sigmas": [1, 1],
    ...         "eta": 2,
    ...         "lkj_sd": 1,
    ...     },
    ...     None,
    ... )
    Inference data...
    """

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
                      :code:`priors = {"mus": [0, 0], "sigmas": [1, 1],
                      "eta": 2, "lkj_sd": 2}`
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
            sd_dist = pm.Exponential.dist(priors["lkj_sd"], shape=2)
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

    def sample_predictive_distribution(self, ppc_sampler="jax"):
        """Function to sample the Multivariate Normal posterior predictive
        Likelihood term in the IV class. This can be slow without
        using the JAX sampler compilation method. If using the
        JAX sampler it will sample only the posterior predictive distribution.
        If using the PYMC sampler if will sample both the prior
        and posterior predictive distributions."""
        random_seed = self.sample_kwargs.get("random_seed", None)

        if ppc_sampler == "jax":
            with self:
                self.idata.extend(
                    pm.sample_posterior_predictive(
                        self.idata,
                        random_seed=random_seed,
                        compile_kwargs={"mode": "JAX"},
                    )
                )
        elif ppc_sampler == "pymc":
            with self:
                self.idata.extend(pm.sample_prior_predictive(random_seed=random_seed))
                self.idata.extend(
                    pm.sample_posterior_predictive(
                        self.idata,
                        random_seed=random_seed,
                    )
                )

    def fit(self, X, Z, y, t, coords, priors, ppc_sampler=None):
        """Draw samples from posterior distribution and potentially
        from the prior and posterior predictive distributions. The
        fit call can take values for the
        ppc_sampler = ['jax', 'pymc', None]
        We default to None, so the user can determine if they wish
        to spend time sampling the posterior predictive distribution
        independently.
        """

        # Ensure random_seed is used in sample_prior_predictive() and
        # sample_posterior_predictive() if provided in sample_kwargs.
        # Use JAX for ppc sampling of multivariate likelihood

        self.build_model(X, Z, y, t, coords, priors)
        with self:
            self.idata = pm.sample(**self.sample_kwargs)
        self.sample_predictive_distribution(ppc_sampler=ppc_sampler)
        return self.idata


class PropensityScore(PyMCModel):
    r"""
    Custom PyMC model for inverse propensity score models

    .. note:
        Generally, the `.fit()` method should be used rather than
        calling `.build_model()` directly.

    Defines the PyMC model

    .. math::
        \beta &\sim \mathrm{Normal}(0, 1) \\
        \sigma &\sim \mathrm{HalfNormal}(1) \\
        \mu &= X \cdot \beta \\
        p &= \text{logit}^{-1}(\mu) \\
        t &\sim \mathrm{Bernoulli}(p)

    Example
    --------
    >>> import causalpy as cp
    >>> import numpy as np
    >>> from causalpy.pymc_models import PropensityScore
    >>> df = cp.load_data('nhefs')
    >>> X = df[["age", "race"]]
    >>> t = np.asarray(df["trt"])
    >>> ps = PropensityScore(sample_kwargs={"progressbar": False})
    >>> ps.fit(X, t, coords={
    ...                 'coeffs': ['age', 'race'],
    ...                 'obs_ind': np.arange(df.shape[0])
    ...                },
    ... )
    Inference...
    """  # noqa: W605

    def build_model(self, X, t, coords):
        "Defines the PyMC propensity model"
        with self:
            self.add_coords(coords)
            X_data = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            t_data = pm.Data("t", t.flatten(), dims="obs_ind")
            b = pm.Normal("b", mu=0, sigma=1, dims="coeffs")
            mu = pm.math.dot(X_data, b)
            p = pm.Deterministic("p", pm.math.invlogit(mu))
            pm.Bernoulli("t_pred", p=p, observed=t_data, dims="obs_ind")

    def fit(self, X, t, coords):
        """Draw samples from posterior, prior predictive, and posterior predictive
        distributions. We overwrite the base method because the base method assumes
        a variable y and we use t to indicate the treatment variable here.
        """
        # Ensure random_seed is used in sample_prior_predictive() and
        # sample_posterior_predictive() if provided in sample_kwargs.
        random_seed = self.sample_kwargs.get("random_seed", None)

        self.build_model(X, t, coords)
        with self:
            self.idata = pm.sample(**self.sample_kwargs)
            self.idata.extend(pm.sample_prior_predictive(random_seed=random_seed))
            self.idata.extend(
                pm.sample_posterior_predictive(
                    self.idata, progressbar=False, random_seed=random_seed
                )
            )
        return self.idata


class BayesianStructuralTimeSeries(PyMCModel):
    r"""
    Bayesian Structural Time Series Model.

    This model allows for the inclusion of trend, seasonality (via Fourier series),
    and optional exogenous regressors.

    .. math::
        \text{trend} &\sim \text{LinearTrend}(...) \\
        \text{seasonality} &\sim \text{YearlyFourier}(...) \\
        \beta &\sim \mathrm{Normal}(0, \sigma_{\beta}) \quad \text{(if X is provided)} \\
        \sigma &\sim \mathrm{HalfNormal}(\sigma_{err}) \\
        \mu &= \text{trend_component} + \text{seasonality_component} [+ X \cdot \beta] \\
        y &\sim \mathrm{Normal}(\mu, \sigma)

    Parameters
    ----------
    n_order : int, optional
        The number of Fourier components for the yearly seasonality. Defaults to 3.
    n_changepoints_trend : int, optional
        The number of changepoints for the linear trend component. Defaults to 10.
    sample_kwargs : dict, optional
        A dictionary of kwargs that get unpacked and passed to the
        :func:`pymc.sample` function. Defaults to an empty dictionary.
    trend_component : Optional[Any], optional
        A custom trend component model. If None, the default pymc-marketing trend component is used.
    seasonality_component : Optional[Any], optional
        A custom seasonality component model. If None, the default pymc-marketing seasonality `YearlyFourier` component is used.
    """  # noqa: W605

    def __init__(
        self,
        n_order: int = 3,
        n_changepoints_trend: int = 10,
        prior_sigma: float = 5,
        trend_component: Optional[Any] = None,
        seasonality_component: Optional[Any] = None,
        sample_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(sample_kwargs=sample_kwargs)

        # Store original configuration parameters
        self.n_order = n_order
        self.n_changepoints_trend = n_changepoints_trend
        self.prior_sigma = prior_sigma

        # Attempt to import pymc_marketing components
        _PymcMarketingLinearTrend = None
        _PymcMarketingYearlyFourier = None
        pymc_marketing_available = False
        try:
            from pymc_marketing.mmm import LinearTrend as PymcMLinearTrend
            from pymc_marketing.mmm import YearlyFourier as PymcMYearlyFourier

            _PymcMarketingLinearTrend = PymcMLinearTrend
            _PymcMarketingYearlyFourier = PymcMYearlyFourier
            pymc_marketing_available = True
        except ImportError:
            # pymc-marketing is not available. This is handled conditionally below.
            pass

        if seasonality_component is not None:
            self._yearly_fourier = seasonality_component
        else:
            if not pymc_marketing_available:
                raise ImportError(
                    "pymc-marketing is required for the default yearly seasonality component. "
                    "Please install it with `pip install pymc-marketing` or "
                    "provide a custom 'seasonality_component'."
                )
            self._yearly_fourier = _PymcMarketingYearlyFourier(n_order=self.n_order)

        if trend_component is not None:
            self._linear_trend = trend_component
        else:
            if not pymc_marketing_available:
                raise ImportError(
                    "pymc-marketing is required for the default linear trend component. "
                    "Please install it with `pip install pymc-marketing` or "
                    "provide a custom 'trend_component'."
                )
            self._linear_trend = _PymcMarketingLinearTrend(
                n_changepoints=self.n_changepoints_trend
            )

    def build_model(self, X, y, coords):
        """
        Defines the PyMC model.

        Parameters
        ----------
        X : array-like or None
            Exogenous variables. If None, the model only includes trend and
            seasonality.
        y : array-like
            The target variable.
        coords : dict
            Coordinates for PyMC model. Must include 'time_for_trend' and
            'time_for_seasonality'. If X is provided, 'coeffs' must also be
            included.
        """
        with self:
            self.add_coords(coords)

            # Time data for trend and seasonality
            # These are expected to be passed in via coords by the experiment class
            # or user.
            time_for_trend = coords.get("time_for_trend")
            time_for_seasonality = coords.get("time_for_seasonality")

            if time_for_trend is None:
                raise ValueError(
                    "'time_for_trend' must be provided in coords for the trend component."  # noqa E501
                )
            if time_for_seasonality is None:
                raise ValueError(
                    "'time_for_seasonality' must be provided in coords for the seasonality component."  # noqa E501
                )

            t_trend_data = pm.Data(
                "t_trend_data", time_for_trend, dims="obs_ind", mutable=True
            )  # noqa E501
            t_season_data = pm.Data(
                "t_season_data", time_for_seasonality, dims="obs_ind", mutable=True
            )  # noqa E501

            # Seasonal component
            season_component = pm.Deterministic(
                "season_component",
                self._yearly_fourier.apply(t_season_data),
                dims="obs_ind",
            )

            # Trend component
            trend_component = pm.Deterministic(
                "trend_component",
                self._linear_trend.apply(t_trend_data),
                dims="obs_ind",
            )

            # Initialize mu with trend and seasonality
            mu_ = trend_component + season_component

            # Exogenous regressors (optional)
            if X is not None and X.shape[1] > 0:
                if "coeffs" not in coords:
                    raise ValueError(
                        "'coeffs' must be provided in coords when X is not None."
                    )
                X_data = pm.Data("X", X, dims=["obs_ind", "coeffs"], mutable=True)
                # Priors for beta coefficients
                # TODO: Allow user-specified priors for beta
                beta = pm.Normal("beta", mu=0, sigma=10, dims="coeffs")
                mu_ = mu_ + pm.math.dot(X_data, beta)
            # If X is None, mu_ remains as trend + season

            # Make mu_ an explicit deterministic variable named "mu"
            mu = pm.Deterministic("mu", mu_, dims="obs_ind")

            # Likelihood
            sigma = pm.HalfNormal("sigma", sigma=self.prior_sigma)
            y_data = pm.Data("y", y.flatten(), dims="obs_ind", mutable=True)
            pm.Normal("y_hat", mu=mu, sigma=sigma, observed=y_data, dims="obs_ind")

    def fit(self, X, y, coords: Optional[Dict[str, Any]] = None) -> None:
        """Draw samples from posterior, prior predictive, and posterior predictive
        distributions, placing them in the model's idata attribute.
        This overrides the base PyMCModel.fit() to ensure 'mu' is included in
        posterior predictive sampling for BSTS.
        """

        random_seed = self.sample_kwargs.get("random_seed", None)

        self.build_model(X, y, coords)
        with self:
            self.idata = pm.sample(**self.sample_kwargs)
            self.idata.extend(pm.sample_prior_predictive(random_seed=random_seed))
            self.idata.extend(
                pm.sample_posterior_predictive(
                    self.idata,
                    var_names=["y_hat", "mu"],  # Ensure mu is sampled
                    progressbar=self.sample_kwargs.get("progressbar", True),
                    random_seed=random_seed,
                )
            )
        return self.idata

    def _data_setter(
        self,
        X_pred,
        time_for_trend_pred: Optional[np.ndarray] = None,
        time_for_seasonality_pred: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set data for the model for prediction.

        For BayesianStructuralTimeSeries, this method updates exogenous variables X_pred
        and, crucially, the time features for trend (time_for_trend_pred) and
        seasonality (time_for_seasonality_pred).
        """
        if time_for_trend_pred is None:
            raise ValueError(
                "time_for_trend_pred must be provided for prediction with BSTS."
            )
        if time_for_seasonality_pred is None:
            raise ValueError(
                "time_for_seasonality_pred must be provided for prediction with BSTS."
            )

        new_no_of_observations = len(time_for_trend_pred)

        if len(time_for_seasonality_pred) != new_no_of_observations:
            raise ValueError(
                "Shape mismatch: time_for_seasonality_pred length "
                f"({len(time_for_seasonality_pred)}) "
                "does not match time_for_trend_pred length "
                f"({new_no_of_observations})."
            )

        new_obs_inds = np.arange(new_no_of_observations)
        data_to_set = {
            "y": np.zeros(new_no_of_observations),  # For prediction, y is dummy
            "t_trend_data": time_for_trend_pred,
            "t_season_data": time_for_seasonality_pred,
        }
        coords_to_set = {"obs_ind": new_obs_inds}

        if "X" in self.named_vars:  # Model was built with exogenous variable X
            if X_pred is None:
                raise ValueError(
                    "Model was built with exogenous variable X. "
                    "New X data (X_pred) must be provided for prediction, not None."
                )
            if X_pred.shape[0] != new_no_of_observations:
                raise ValueError(
                    "Shape mismatch: X_pred number of rows "
                    f"({X_pred.shape[0]}) "
                    "does not match time_for_trend_pred length "
                    f"({new_no_of_observations})."
                )
            data_to_set["X"] = X_pred
        elif X_pred is not None:
            # Model does not have 'X' pm.Data, but X_pred was provided.
            print(
                "Warning: X_pred provided, but the model was not built with exogenous variables X. "
                "X_pred will be ignored."
            )

        # If model was built without X, and X_pred is None, this is fine.

        with self:
            pm.set_data(data_to_set, coords=coords_to_set)

    def predict(
        self,
        X,
        time_for_trend_pred: Optional[np.ndarray] = None,
        time_for_seasonality_pred: Optional[np.ndarray] = None,
    ):
        """
        Predict data given input data X and new time features.
        Overrides PyMCModel.predict to handle specific time features for BSTS.
        """
        random_seed = self.sample_kwargs.get("random_seed", None)
        self._data_setter(
            X,
            time_for_trend_pred=time_for_trend_pred,
            time_for_seasonality_pred=time_for_seasonality_pred,
        )
        with self:  # sample with new input data
            post_pred = pm.sample_posterior_predictive(
                self.idata,
                var_names=["y_hat", "mu"],
                progressbar=self.sample_kwargs.get(
                    "progressbar", False
                ),  # Consistent with base
                random_seed=random_seed,
            )
        return post_pred

    def score(
        self,
        X,
        y,
        time_for_trend_pred: Optional[np.ndarray] = None,
        time_for_seasonality_pred: Optional[np.ndarray] = None,
    ) -> pd.Series:
        """Score the Bayesian R2 given inputs X, y and new time features.
        Overrides PyMCModel.score to handle specific time features for BSTS.
        """
        # Predict with new data (X and time features)
        pred_output = self.predict(
            X,
            time_for_trend_pred=time_for_trend_pred,
            time_for_seasonality_pred=time_for_seasonality_pred,
        )
        # Extract mu for R2 calculation
        mu_pred = az.extract(
            pred_output, group="posterior_predictive", var_names="mu"
        ).T.values
        # Note: First argument must be a 1D array
        return r2_score(y.flatten(), mu_pred)
