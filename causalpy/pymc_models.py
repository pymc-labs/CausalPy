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

from typing import Any, Dict, List, Optional

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

    def predict(
        self,
        X,
        coords: Optional[Dict[str, Any]] = None,
        out_of_sample: Optional[bool] = False,
        **kwargs,
    ):
        """
        Predict data given input data `X`

        .. caution::
            Results in KeyError if model hasn't been fit.
        """

        # Ensure random_seed is used in sample_prior_predictive() and
        # sample_posterior_predictive() if provided in sample_kwargs.
        random_seed = self.sample_kwargs.get("random_seed", None)
        # Base _data_setter doesn't use coords, but subclasses might override _data_setter to use it.
        # If a subclass needs coords in _data_setter, it should handle it.
        self._data_setter(X)
        with self:  # sample with new input data
            pp = pm.sample_posterior_predictive(
                self.idata,
                var_names=["y_hat", "mu"],
                progressbar=False,
                random_seed=random_seed,
            )

        # TODO: This is a bit of a hack. Maybe it could be done properly in _data_setter?
        if isinstance(X, xr.DataArray):
            pp["posterior_predictive"] = pp["posterior_predictive"].assign_coords(
                obs_ind=X.obs_ind
            )

        return pp

    def score(
        self, X, y, coords: Optional[Dict[str, Any]] = None, **kwargs
    ) -> pd.Series:
        """Score the Bayesian :math:`R^2` given inputs ``X`` and outputs ``y``.

        Note that the score is based on a comparison of the observed data ``y`` and the
        model's expected value of the data, `mu`.

        .. caution::

            The Bayesian :math:`R^2` is not the same as the traditional coefficient of
            determination, https://en.wikipedia.org/wiki/Coefficient_of_determination.

        """
        mu = self.predict(X)
        mu = az.extract(mu, group="posterior_predictive", var_names="mu").T
        return r2_score(y.data, mu.data)

    def calculate_impact(
        self, y_true: xr.DataArray, y_pred: az.InferenceData
    ) -> xr.DataArray:
        impact = y_true - y_pred["posterior_predictive"]["y_hat"]
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
    >>> import xarray as xr
    >>> from causalpy.pymc_models import LinearRegression
    >>> rd = cp.load_data("rd")
    >>> rd["treated"] = rd["treated"].astype(int)
    >>> coeffs = ["x", "treated"]
    >>> X = xr.DataArray(
    ...     rd[coeffs].values,
    ...     dims=["obs_ind", "coeffs"],
    ...     coords={"obs_ind": rd.index, "coeffs": coeffs},
    ... )
    >>> y = xr.DataArray(
    ...     rd["y"].values,
    ...     dims=["obs_ind"],
    ...     coords={"obs_ind": rd.index},
    ... )
    >>> lr = LinearRegression(sample_kwargs={"progressbar": False})
    >>> coords={"coeffs": coeffs, "obs_ind": np.arange(rd.shape[0])}
    >>> lr.fit(X, y, coords=coords)
    Inference data...
    """  # noqa: W605

    def build_model(self, X, y, coords):
        """
        Defines the PyMC model
        """
        with self:
            self.add_coords(coords)
            X = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            y = pm.Data("y", y, dims="obs_ind")
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
    >>> wsf.fit(X, y)
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
            beta = pm.Dirichlet("beta", a=np.ones(n_predictors), dims="coeffs")
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


class BayesianBasisExpansionTimeSeries(PyMCModel):
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
        Only used if seasonality_component is None.
    n_changepoints_trend : int, optional
        The number of changepoints for the linear trend component. Defaults to 10.
        Only used if trend_component is None.
    prior_sigma : float, optional
        Prior standard deviation for the observation noise. Defaults to 5.
    trend_component : Optional[Any], optional
        A custom trend component model. If None, the default pymc-marketing LinearTrend component is used.
        Must have an `apply(time_data)` method that returns a PyMC tensor.
    seasonality_component : Optional[Any], optional
        A custom seasonality component model. If None, the default pymc-marketing YearlyFourier component is used.
        Must have an `apply(time_data)` method that returns a PyMC tensor.
    sample_kwargs : dict, optional
        A dictionary of kwargs that get unpacked and passed to the
        :func:`pymc.sample` function. Defaults to an empty dictionary.
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
        self._first_fit_timestamp: Optional[pd.Timestamp] = None
        self._exog_var_names: Optional[List[str]] = None

        # Store custom components (fix the bug where they were swapped)
        self._custom_trend_component = trend_component
        self._custom_seasonality_component = seasonality_component

        # Initialize and validate components
        self._trend_component = None
        self._seasonality_component = None
        self._validate_and_initialize_components()

    def _validate_and_initialize_components(self):
        """
        Validate and initialize trend and seasonality components.
        This separates validation from model building for cleaner code.
        """
        # Validate pymc-marketing availability if using default components
        if (
            self._custom_trend_component is None
            or self._custom_seasonality_component is None
        ):
            try:
                from pymc_marketing.mmm import LinearTrend, YearlyFourier

                self._PymcMarketingLinearTrend = LinearTrend
                self._PymcMarketingYearlyFourier = YearlyFourier
            except ImportError:
                raise ImportError(
                    "pymc-marketing is required when using default trend or seasonality components. "
                    "Please install it with `pip install pymc-marketing` or provide custom components."
                )

        # Validate custom components have required methods
        if self._custom_trend_component is not None:
            if not hasattr(self._custom_trend_component, "apply"):
                raise ValueError(
                    "Custom trend_component must have an 'apply' method that accepts time data "
                    "and returns a PyMC tensor."
                )

        if self._custom_seasonality_component is not None:
            if not hasattr(self._custom_seasonality_component, "apply"):
                raise ValueError(
                    "Custom seasonality_component must have an 'apply' method that accepts time data "
                    "and returns a PyMC tensor."
                )

    def _get_trend_component(self):
        """Get the trend component, creating default if needed."""
        if self._custom_trend_component is not None:
            return self._custom_trend_component

        # Create default trend component
        if self._trend_component is None:
            self._trend_component = self._PymcMarketingLinearTrend(
                n_changepoints=self.n_changepoints_trend
            )
        return self._trend_component

    def _get_seasonality_component(self):
        """Get the seasonality component, creating default if needed."""
        if self._custom_seasonality_component is not None:
            return self._custom_seasonality_component

        # Create default seasonality component
        if self._seasonality_component is None:
            self._seasonality_component = self._PymcMarketingYearlyFourier(
                n_order=self.n_order
            )
        return self._seasonality_component

    def _prepare_time_and_exog_features(
        self,
        X_exog_array: Optional[np.ndarray],
        datetime_index: pd.DatetimeIndex,
        exog_names_from_coords: Optional[List[str]] = None,
    ):
        """
        Prepares time features from datetime_index and processes exogenous variables from X_exog_array.
        Exogenous variable names are taken from exog_names_from_coords (expected to be a list).
        """
        if not isinstance(datetime_index, pd.DatetimeIndex):
            raise ValueError("`datetime_index` must be a pandas DatetimeIndex.")

        num_obs = len(datetime_index)

        if X_exog_array is not None:
            if not isinstance(X_exog_array, np.ndarray):
                raise TypeError("X_exog_array must be a NumPy array or None.")
            if X_exog_array.ndim == 1:
                X_exog_array = X_exog_array.reshape(-1, 1)
            if X_exog_array.shape[0] != num_obs:
                raise ValueError(
                    f"Shape mismatch: X_exog_array rows ({X_exog_array.shape[0]}) and length of `datetime_index` ({num_obs}) must be equal."
                )
            if exog_names_from_coords and X_exog_array.shape[1] != len(
                exog_names_from_coords
            ):
                raise ValueError(
                    f"Mismatch: X_exog_array has {X_exog_array.shape[1]} columns, but {len(exog_names_from_coords)} names provided."
                )
        else:  # No exogenous variables passed as array
            if exog_names_from_coords:
                # This implies exog_names were given, but no array. Could mean an empty array for 0 columns was intended.
                if X_exog_array is None:
                    X_exog_array = np.empty((num_obs, 0))

        # Ensure exog_names_from_coords is a list for internal processing
        processed_exog_names = []
        if exog_names_from_coords is not None:
            if isinstance(exog_names_from_coords, str):
                processed_exog_names = [exog_names_from_coords]
            elif isinstance(exog_names_from_coords, (list, tuple)):
                processed_exog_names = list(exog_names_from_coords)
            else:
                raise TypeError(
                    f"exog_names_from_coords should be a list, tuple, or string, not {type(exog_names_from_coords)}"
                )

        # Set or validate self._exog_var_names (must be a list)
        if X_exog_array is not None and X_exog_array.shape[1] > 0:
            if not processed_exog_names:
                raise ValueError(
                    "Logic error: processed_exog_names should be set if X_exog_array has columns."
                )
            if self._exog_var_names is None:
                self._exog_var_names = processed_exog_names  # Ensures it's a list
            elif (
                self._exog_var_names != processed_exog_names
            ):  # List-to-list comparison
                raise ValueError(
                    f"Exogenous variable names mismatch. Model fit with {self._exog_var_names}, "
                    f"but current call provides {processed_exog_names}."
                )
        elif (
            self._exog_var_names is None
        ):  # No exog vars in this call, and none set before
            self._exog_var_names = []  # Explicitly an empty list

        if self._first_fit_timestamp is None:
            self._first_fit_timestamp = datetime_index[0]

        time_for_trend = (
            (datetime_index - self._first_fit_timestamp).days / 365.25
        ).values
        time_for_seasonality = datetime_index.dayofyear.values

        # X_values to be used by PyMC; None if no exog vars
        X_values_for_pymc = X_exog_array if self._exog_var_names else None
        if X_values_for_pymc is not None and X_values_for_pymc.shape[1] == 0:
            X_values_for_pymc = (
                None  # Treat 0-column array as no exog vars for PyMC part
            )

        return time_for_trend, time_for_seasonality, X_values_for_pymc, num_obs

    def build_model(
        self, X: Optional[np.ndarray], y: np.ndarray, coords: Dict[str, Any]
    ):
        """
        Defines the PyMC model.

        Parameters
        ----------
        X : np.ndarray or None
            NumPy array of exogenous regressors. Can be None if no exogenous variables.
        y : np.ndarray
            The target variable.
        coords : dict
            Coordinates dictionary. Must contain "datetime_index" (pd.DatetimeIndex).
            If X is provided and has columns, coords must also contain "coeffs" (List[str]).
        """
        datetime_index = coords.pop("datetime_index", None)
        if not isinstance(datetime_index, pd.DatetimeIndex):
            raise ValueError(
                "`coords` must contain 'datetime_index' of type pd.DatetimeIndex."
            )

        # Get exog_names from coords["coeffs"] if X_exog_array is present
        exog_names_from_coords = coords.get("coeffs")

        (
            time_for_trend,
            time_for_seasonality,
            X_values_for_pymc,  # NumPy array for PyMC or None
            num_obs,
        ) = self._prepare_time_and_exog_features(
            X, datetime_index, exog_names_from_coords
        )

        model_coords = {
            "obs_ind": np.arange(num_obs),
        }

        # Start with a copy of the input coords (datetime_index was already popped)
        if coords:
            model_coords.update(coords)

        # Ensure "coeffs" in model_coords (if present from input) is a list
        if "coeffs" in model_coords:
            current_coeffs = model_coords["coeffs"]
            if isinstance(current_coeffs, str):
                model_coords["coeffs"] = [current_coeffs]
            elif isinstance(current_coeffs, tuple):
                model_coords["coeffs"] = list(current_coeffs)
            elif not isinstance(current_coeffs, list):
                # If it's something else weird, raise error or clear it
                # so self._exog_var_names can take precedence if needed.
                raise TypeError(
                    f"Unexpected type for 'coeffs' in input coords: {type(current_coeffs)}"
                )

        # self._exog_var_names is the source of truth for coefficient names, ensure it's a list (done in _prepare)
        # Override or set "coeffs" in model_coords based on self._exog_var_names
        if self._exog_var_names:
            if (
                "coeffs" in model_coords
                and model_coords["coeffs"] != self._exog_var_names
            ):
                # This implies a mismatch between what user provided in coords["coeffs"]
                # and what _prepare_time_and_exog_features decided based on X and coords["coeffs"]
                # This should ideally be caught earlier or be consistent.
                # For now, let's assume _prepare_time_and_exog_features's derivation (self._exog_var_names) is correct.
                print(
                    f"Warning: Discrepancy in 'coeffs'. Using derived: {self._exog_var_names} over input: {model_coords['coeffs']}"
                )
            model_coords["coeffs"] = self._exog_var_names
        elif "coeffs" in model_coords and model_coords["coeffs"]:
            # No exog vars determined by _prepare..., but coords has non-empty coeffs
            raise ValueError(
                f"Model determined no exogenous variables (self._exog_var_names is {self._exog_var_names}), "
                f"but input coords provided 'coeffs': {model_coords['coeffs']}. "
                f"If no exog vars, provide empty list or omit 'coeffs'."
            )
        elif (
            "coeffs" not in model_coords and self._exog_var_names
        ):  # Should not happen if logic is right
            model_coords["coeffs"] = self._exog_var_names

        with self:
            self.add_coords(model_coords)

            # Time data for trend and seasonality
            t_trend_data = pm.Data(
                "t_trend_data", time_for_trend, dims="obs_ind", mutable=True
            )
            t_season_data = pm.Data(
                "t_season_data", time_for_seasonality, dims="obs_ind", mutable=True
            )

            # Get validated components (no more ugly imports in build_model!)
            trend_component_instance = self._get_trend_component()
            seasonality_component_instance = self._get_seasonality_component()

            # Seasonal component
            season_component = pm.Deterministic(
                "season_component",
                seasonality_component_instance.apply(t_season_data),
                dims="obs_ind",
            )

            # Trend component
            trend_component_values = trend_component_instance.apply(t_trend_data)
            trend_component = pm.Deterministic(
                "trend_component",
                trend_component_values,
                dims="obs_ind",
            )

            # Initialize mu with trend and seasonality
            mu_ = trend_component + season_component

            # Exogenous regressors (optional)
            if (
                X_values_for_pymc is not None and self._exog_var_names
            ):  # self._exog_var_names is guaranteed list
                # self.coords["coeffs"] should be an xarray.Coordinate object here.
                # Its .values attribute is a numpy array. So list(self.coords["coeffs"].values) is a list.
                model_coord_coeffs_list = (
                    list(self.coords["coeffs"]) if "coeffs" in self.coords else []
                )
                if (
                    "coeffs" not in self.coords
                    or model_coord_coeffs_list != self._exog_var_names
                ):
                    raise ValueError(
                        f"Mismatch between internal exogenous variable names ('{self._exog_var_names}') "
                        f"and model coordinates for 'coeffs' ({model_coord_coeffs_list})."
                    )
                if X_values_for_pymc.shape[1] != len(self._exog_var_names):
                    raise ValueError(
                        f"Shape mismatch: X_values_for_pymc has {X_values_for_pymc.shape[1]} columns, but "
                        f"{len(self._exog_var_names)} names in self._exog_var_names ({self._exog_var_names})."
                    )
                X_data = pm.Data(
                    "X", X_values_for_pymc, dims=["obs_ind", "coeffs"], mutable=True
                )
                beta = pm.Normal("beta", mu=0, sigma=10, dims="coeffs")
                mu_ = mu_ + pm.math.dot(X_data, beta)

            # Make mu_ an explicit deterministic variable named "mu"
            mu = pm.Deterministic("mu", mu_, dims="obs_ind")

            # Likelihood
            sigma = pm.HalfNormal("sigma", sigma=self.prior_sigma)
            y_data = pm.Data("y", y.flatten(), dims="obs_ind", mutable=True)
            pm.Normal("y_hat", mu=mu, sigma=sigma, observed=y_data, dims="obs_ind")

    def fit(
        self, X: Optional[np.ndarray], y: np.ndarray, coords: Dict[str, Any]
    ) -> None:
        """Draw samples from posterior, prior predictive, and posterior predictive
        distributions, placing them in the model's idata attribute.
        Parameters
        ----------
        X : np.ndarray or None
            NumPy array of exogenous regressors. Can be None or an array with 0 columns
            if no exogenous variables.
        y : np.ndarray
            The target variable.
        coords : dict
            Coordinates dictionary. Must contain "datetime_index" (pd.DatetimeIndex).
            If X is provided and has columns, coords must also contain "coeffs" (List[str]).
        """

        random_seed = self.sample_kwargs.get("random_seed", None)
        # X can be None if no exog vars, _prepare_... handles it.
        self.build_model(X, y, coords=coords)
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
        X_pred: Optional[np.ndarray],
        coords_pred: Dict[
            str, Any
        ],  # Must contain "datetime_index" for prediction period
    ) -> None:
        """
        Set data for the model for prediction.
        X_pred contains exogenous variables for the prediction period.
        coords_pred must contain "datetime_index" for the prediction period.
        """
        datetime_index_pred = coords_pred.get("datetime_index")
        if not isinstance(datetime_index_pred, pd.DatetimeIndex):
            raise ValueError(
                "`coords_pred` must contain 'datetime_index' for prediction."
            )

        # For _data_setter, exog_names are already known (self._exog_var_names from fit)
        # We pass self._exog_var_names so _prepare_time_and_exog_features can validate
        # the shape of X_pred_numpy if it's provided.
        (
            time_for_trend_pred_vals,
            time_for_seasonality_pred_vals,
            X_exog_pred_vals,  # NumPy array for PyMC or None
            num_obs_pred,
        ) = self._prepare_time_and_exog_features(
            X_pred, datetime_index_pred, self._exog_var_names
        )

        new_obs_inds = np.arange(num_obs_pred)

        data_to_set = {
            "y": np.zeros(num_obs_pred),
            "t_trend_data": time_for_trend_pred_vals,
            "t_season_data": time_for_seasonality_pred_vals,
        }
        coords_to_set = {"obs_ind": new_obs_inds}

        if (
            "X" in self.named_vars
        ):  # Model was built with exogenous variable X (i.e. self._exog_var_names is not empty)
            if (
                X_exog_pred_vals is None and self._exog_var_names
            ):  # Check if exog_var_names expects something
                raise ValueError(
                    "Model was built with exogenous variables. "
                    "New X data (X_pred) must provide these (or index_for_time_pred if X_pred is array)."
                )
            if (
                self._exog_var_names
                and X_exog_pred_vals is not None
                and X_exog_pred_vals.shape[1] != len(self._exog_var_names)
            ):
                raise ValueError(
                    f"Shape mismatch for exogenous prediction variables. Expected {len(self._exog_var_names)} columns, "
                    f"got {X_exog_pred_vals.shape[1]}."
                )
            data_to_set["X"] = X_exog_pred_vals  # Can be None if no exog vars
        elif X_exog_pred_vals is not None:
            print(
                "Warning: X_pred provided exogenous variables, but the model was not "
                "built with exogenous variables. These will be ignored."
            )

        # Ensure "X" is set to None if no exog vars, even if "X" data var exists but model has no coeffs
        if not self._exog_var_names and "X" in self.named_vars:
            # Pass an array with 0 columns for the X data variable if no exog vars expected
            if X_exog_pred_vals is not None and X_exog_pred_vals.shape[1] > 0:
                # This should not happen if self._exog_var_names is empty
                print(
                    "Warning: Model expects no exog vars, but X_exog_pred_vals has columns. Forcing to 0 columns."
                )
                data_to_set["X"] = np.empty((num_obs_pred, 0))
            elif X_exog_pred_vals is None:
                data_to_set["X"] = np.empty((num_obs_pred, 0))
            else:  # X_exog_pred_vals has 0 columns already
                data_to_set["X"] = X_exog_pred_vals

        with self:
            pm.set_data(data_to_set, coords=coords_to_set)

    def predict(
        self,
        X: Optional[np.ndarray],
        coords: Dict[str, Any],  # Must contain "datetime_index" for prediction period
        out_of_sample: Optional[bool] = False,
    ):
        """
        Predict data given input X and coords for prediction period.
        coords must contain "datetime_index". If X has columns, coords should also have "coeffs".
        However, for prediction, exog var names are already known by the model.
        """
        random_seed = self.sample_kwargs.get("random_seed", None)
        self._data_setter(X, coords_pred=coords)
        with self:
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
        X: Optional[np.ndarray],
        y: np.ndarray,
        coords: Dict[str, Any],  # Must contain "datetime_index" for score period
    ) -> pd.Series:
        """Score the Bayesian R2.
        coords must contain "datetime_index". If X has columns, coords should also have "coeffs".
        However, for scoring, exog var names are already known by the model.
        """
        pred_output = self.predict(X, coords=coords)
        mu_pred = az.extract(
            pred_output, group="posterior_predictive", var_names="mu"
        ).T.values
        # Note: First argument must be a 1D array
        return r2_score(y.flatten(), mu_pred)


class StateSpaceTimeSeries(PyMCModel):
    """
    State-space time series model using pymc_extras.statespace.structural.

    Parameters
    ----------
    level_order : int, optional
        Order of the local level/trend component. Defaults to 2.
    seasonal_length : int, optional
        Seasonal period (e.g., 12 for monthly data with annual seasonality). Defaults to 12.
    trend_component : optional
        Custom state-space trend component.
    seasonality_component : optional
        Custom state-space seasonal component.
    sample_kwargs : dict, optional
        Kwargs passed to `pm.sample`.
    mode : str, optional
        Mode passed to `build_statespace_graph` (e.g., "JAX").
    """

    def __init__(
        self,
        level_order: int = 2,
        seasonal_length: int = 12,
        trend_component: Optional[Any] = None,
        seasonality_component: Optional[Any] = None,
        sample_kwargs: Optional[Dict[str, Any]] = None,
        mode: str = "JAX",
    ):
        super().__init__(sample_kwargs=sample_kwargs)
        self._custom_trend_component = trend_component
        self._custom_seasonality_component = seasonality_component
        self.level_order = level_order
        self.seasonal_length = seasonal_length
        self.mode = mode
        self.ss_mod = None
        self._validate_and_initialize_components()

    def _validate_and_initialize_components(self):
        """
        Validate and initialize trend and seasonality components.
        This separates validation from model building for cleaner code.
        """
        # Validate pymc-extras availability if using default components
        if (
            self._custom_trend_component is None
            or self._custom_seasonality_component is None
        ):
            try:
                from pymc_extras.statespace import structural as st

                self._PymcExtrasLevelTrendComponent = st.LevelTrendComponent
                self._PymcExtrasFrequencySeasonality = st.FrequencySeasonality
            except ImportError:
                raise ImportError(
                    "pymc-extras is required when using default trend or seasonality components. "
                    "Please install it with `conda install -c conda-forge pymc-extras` or provide custom components."
                )

        # Validate custom components have required methods
        if self._custom_trend_component is not None:
            if not hasattr(self._custom_trend_component, "apply"):
                raise ValueError(
                    "Custom trend_component must have an 'apply' method that accepts time data "
                    "and returns a PyMC tensor."
                )

        if self._custom_seasonality_component is not None:
            if not hasattr(self._custom_seasonality_component, "apply"):
                raise ValueError(
                    "Custom seasonality_component must have an 'apply' method that accepts time data "
                    "and returns a PyMC tensor."
                )

        # Initialize components
        self._trend_component = None
        self._seasonality_component = None

    def _get_trend_component(self):
        """Get the trend component, creating default if needed."""
        if self._custom_trend_component is not None:
            return self._custom_trend_component

        # Create default trend component
        if self._trend_component is None:
            self._trend_component = self._PymcExtrasLevelTrendComponent(
                order=self.level_order
            )
        return self._trend_component

    def _get_seasonality_component(self):
        """Get the seasonality component, creating default if needed."""
        if self._custom_seasonality_component is not None:
            return self._custom_seasonality_component

        # Create default seasonality component
        if self._seasonality_component is None:
            self._seasonality_component = self._PymcExtrasFrequencySeasonality(
                season_length=self.seasonal_length, name="freq"
            )
        return self._seasonality_component

    def build_model(
        self, X: Optional[np.ndarray], y: np.ndarray, coords: Dict[str, Any]
    ) -> None:
        """
        Build the PyMC state-space model.  `coords` must include:
          - 'datetime_index': a pandas.DatetimeIndex matching `y`.
        """
        coords = coords.copy()
        datetime_index = coords.pop("datetime_index", None)
        if not isinstance(datetime_index, pd.DatetimeIndex):
            raise ValueError(
                "coords must contain 'datetime_index' of type pandas.DatetimeIndex."
            )
        self._train_index = datetime_index

        # Instantiate components and build state-space object
        trend = self._get_trend_component()
        season = self._get_seasonality_component()
        combined = trend + season
        self.ss_mod = combined.build()

        # Extract parameter dims (order: initial_trend, sigma_trend, seasonal, P0)
        initial_trend_dims, sigma_trend_dims, annual_dims, P0_dims = (
            self.ss_mod.param_dims.values()
        )
        coordinates = {**coords, **self.ss_mod.coords}

        # Build model
        with pm.Model(coords=coordinates) as self.second_model:
            # Add coords for statespace (includes 'time' and 'state' dims)
            P0_diag = pm.Gamma("P0_diag", alpha=2, beta=1, dims=P0_dims[0])
            _P0 = pm.Deterministic("P0", pt.diag(P0_diag), dims=P0_dims)
            _initial_trend = pm.Normal(
                "initial_trend", sigma=50, dims=initial_trend_dims
            )
            _annual_seasonal = pm.ZeroSumNormal("freq", sigma=80, dims=annual_dims)

            _sigma_trend = pm.Gamma(
                "sigma_trend", alpha=2, beta=5, dims=sigma_trend_dims
            )
            _sigma_monthly_season = pm.Gamma("sigma_freq", alpha=2, beta=1)

            # Attach the state-space graph using the observed data
            df = pd.DataFrame({"y": y.flatten()}, index=datetime_index)
            self.ss_mod.build_statespace_graph(df[["y"]], mode=self.mode)

    def fit(
        self, X: Optional[np.ndarray], y: np.ndarray, coords: Dict[str, Any]
    ) -> az.InferenceData:
        """
        Fit the model, drawing posterior samples.
        Returns the InferenceData with parameter draws.
        """
        self.build_model(X, y, coords)
        with self.second_model:
            self.idata = pm.sample(**self.sample_kwargs)
            self.idata.extend(
                pm.sample_posterior_predictive(
                    self.idata,
                )
            )
        self.conditional_idata = self._smooth()
        return self._prepare_idata()

    def _prepare_idata(self):
        if self.idata is None:
            raise RuntimeError("Model must be fit before smoothing.")

        new_idata = self.idata.copy()
        # Get smoothed posterior and sum over state dimension
        smoothed = self.conditional_idata.isel(observed_state=0).rename(
            {"smoothed_posterior": "y_hat"}
        )
        y_hat_summed = smoothed.y_hat.sum(dim="state")

        # Rename 'time' to 'obs_ind' to match CausalPy conventions
        if "time" in y_hat_summed.dims:
            y_hat_final = y_hat_summed.rename({"time": "obs_ind"})
        else:
            y_hat_final = y_hat_summed

        new_idata["posterior_predictive"]["y_hat"] = y_hat_final
        new_idata["posterior_predictive"]["mu"] = y_hat_final

        return new_idata

    def _smooth(self) -> xr.Dataset:
        """
        Run the Kalman smoother / conditional posterior sampler.
        Returns an xarray Dataset with 'smoothed_posterior'.
        """
        if self.idata is None:
            raise RuntimeError("Model must be fit before smoothing.")
        return self.ss_mod.sample_conditional_posterior(self.idata)

    def _forecast(self, start: pd.Timestamp, periods: int) -> xr.Dataset:
        """
        Forecast future values.
        `start` is the timestamp of the last observed point, and `periods` is the number of steps ahead.
        Returns an xarray Dataset with 'forecast_observed'.
        """
        if self.idata is None:
            raise RuntimeError("Model must be fit before forecasting.")
        return self.ss_mod.forecast(self.idata, start=start, periods=periods)

    def predict(
        self,
        X: Optional[np.ndarray],
        coords: Dict[str, Any],
        out_of_sample: Optional[bool] = False,
    ) -> xr.Dataset:
        """
        Wrapper around forecast: expects coords with 'datetime_index' of future points.
        """
        if not out_of_sample:
            return self._prepare_idata()
        else:
            idx = coords.get("datetime_index")
            if not isinstance(idx, pd.DatetimeIndex):
                raise ValueError(
                    "coords must contain 'datetime_index' for prediction period."
                )
            last = self._train_index[-1]  # start forecasting after the last observed
            temp_idata = self._forecast(start=last, periods=len(idx))
            new_idata = temp_idata.copy()

            # Rename 'time' to 'obs_ind' to match CausalPy conventions
            if "time" in new_idata.dims:
                new_idata = new_idata.rename({"time": "obs_ind"})

            # Extract the forecasted observed data and assign it to 'y_hat'
            new_idata["y_hat"] = new_idata["forecast_observed"].isel(observed_state=0)

            # Assign 'y_hat' to 'mu' for consistency
            new_idata["mu"] = new_idata["y_hat"]

            return new_idata

    def score(
        self, X: Optional[np.ndarray], y: np.ndarray, coords: Dict[str, Any]
    ) -> pd.Series:
        """
        Compute R^2 between observed and mean forecast.
        """
        pred = self.predict(X, coords)
        fc = pred["posterior_predictive"]["y_hat"]  # .isel(observed_state=0)

        # Use all posterior samples to compute Bayesian R²
        # fc has shape (chain, draw, time), we want (n_samples, time)
        fc_samples = fc.stack(
            sample=["chain", "draw"]
        ).T.values  # Shape: (time, n_samples)

        # Use arviz.r2_score to get both r2 and r2_std
        return r2_score(y.flatten(), fc_samples)
