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

    def score(self, X, y) -> pd.Series:
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


class InterventionTimeEstimator(PyMCModel):
    r"""
    Custom PyMC model to estimate the time an intervention took place.

    defines the PyMC model :

    .. math::
        \beta &\sim \mathrm{Normal}(0, 1) \\
        \mu &= \beta \cdot X\\
        \\
        \tau &\sim \mathrm{Uniform}(0, 1) \\
        w &= sigmoid(t-\tau) \\
        \\
        \text{level} &\sim \mathrm{Normal}(0, 1) \\
        \text{trend} &\sim \mathrm{Normal}(0, 1) \\
        A &\sim \mathrm{Normal}(0, 1) \\
        \lambda &\sim \mathrm{HalfNormal}(0, 1) \\
        \text{impulse} &= A \cdot exp(-\lambda \cdot |t-\tau|) \\
        \mu_{in} &= \text{level} + \text{trend} \cdot (t-\tau) + \text{impulse}\\
        \\
        \sigma &\sim \mathrm{HalfNormal}(0, 1) \\
        \mu_{ts} &= \mu + \mu_{in} \\
        \\
        y &\sim \mathrm{Normal}(\mu_{ts}, \sigma)

    Example
        --------
        >>> import causalpy as cp
        >>> import numpy as np
        >>> from patsy import build_design_matrices, dmatrices
        >>> from causalpy.pymc_models import InterventionTimeEstimator as ITE
        >>> data = cp.load_data("its")
        >>> formula="y ~ 1 + t + C(month)"
        >>> y, X = dmatrices(formula, data)
        >>> outcome_variable_name = y.design_info.column_names[0]
        >>> labels = X.design_info.column_names
        >>> _y, _X = np.asarray(y), np.asarray(X)
        >>> _X = xr.DataArray(
        ... _X,
        ... dims=["obs_ind", "coeffs"],
        ... coords={
        ...     "obs_ind": data.index,
        ...     "coeffs": labels,
        ...     },
        ... )
        >>> _y = xr.DataArray(
        ...     _y[:, 0],
        ...     dims=["obs_ind"],
        ...     coords={"obs_ind": data.index},
        ...     )
        >>> COORDS = {"coeffs":labels, "obs_ind": np.arange(_X.shape[0])}
        >>> model = ITE(sample_kwargs={"draws" : 10, "tune":10, "progressbar":False})
        >>> model.set_time_range(None, data)
        >>> model.fit(X=_X, y=_y, coords=COORDS)
        Inference ...
    """

    def __init__(
        self, time_variable_name: str, treatment_type_effect=None, sample_kwargs=None
    ):
        """
        Initializes the InterventionTimeEstimator model.

        :param time_variable_name: name of the column representing time among the covariates
        :param treatment_type_effect: Optional dictionary that specifies prior parameters for the
            intervention effects. Expected keys are:
                - "level": [mu, sigma]
                - "trend": [mu, sigma]
                - "impulse": [mu, sigma1, sigma2]
            If a key is missing, the corresponding effect is ignored.
            If the associated list is incomplete, default values will be used.
        :param sample_kwargs: Optional dictionary of arguments passed to pm.sample().
        """
        self.time_variable_name = time_variable_name

        if treatment_type_effect is None:
            treatment_type_effect = {}
        if not isinstance(treatment_type_effect, dict):
            raise TypeError("treatment_type_effect must be a dictionary.")

        super().__init__(sample_kwargs)
        self.treatment_type_effect = treatment_type_effect

    def build_model(self, X, y, coords):
        """
        Defines the PyMC model

        :param X: An array of the covariates
        :param y: An array of values representing our outcome y
        :param coords: Dictionary of named coordinates for PyMC variables (e.g., {"obs_ind": range(n_obs), "coeffs": range(n_covariates)}).

        Assumes the following attributes are already defined in self:
            - self.timeline: the index of the column in X representing time.
            - self.time_range: a tuple (lower_bound, upper_bound) for the intervention time.
            - self.treatment_type_effect: a dictionary specifying which intervention effects to include and their priors.
        """
        DEFAULT_BETA_PRIOR = (0, 5)
        DEFAULT_LEVEL_PRIOR = (0, 5)
        DEFAULT_TREND_PRIOR = (0, 0.5)
        DEFAULT_IMPULSE_PRIOR = (0, 5, 5)

        with self:
            self.add_coords(coords)

            t = pm.Data("t", X.sel(coeffs=self.time_variable_name), dims="obs_ind")
            X = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            y = pm.Data("y", y, dims="obs_ind")

            lower_bound = pm.Data("lower_bound", self.time_range[0])
            upper_bound = pm.Data("upper_bound", self.time_range[1])

            # --- Priors ---
            treatment_time = pm.Uniform(
                "treatment_time", lower=lower_bound, upper=upper_bound
            )
            beta = pm.Normal(
                name="beta",
                mu=DEFAULT_BETA_PRIOR[0],
                sigma=DEFAULT_BETA_PRIOR[1],
                dims="coeffs",
            )

            # --- Intervention effect ---
            mu_in_components = []

            if "level" in self.treatment_type_effect:
                mu, sigma = (
                    DEFAULT_LEVEL_PRIOR
                    if len(self.treatment_type_effect["level"]) != 2
                    else (
                        self.treatment_type_effect["level"][0],
                        self.treatment_type_effect["level"][1],
                    )
                )
                level = pm.Normal(
                    "level",
                    mu=mu,
                    sigma=sigma,
                )
                mu_in_components.append(level)
            if "trend" in self.treatment_type_effect:
                mu, sigma = (
                    DEFAULT_TREND_PRIOR
                    if len(self.treatment_type_effect["trend"]) != 2
                    else (
                        self.treatment_type_effect["trend"][0],
                        self.treatment_type_effect["trend"][1],
                    )
                )
                trend = pm.Normal("trend", mu=mu, sigma=sigma)
                mu_in_components.append(trend * (t - treatment_time))
            if "impulse" in self.treatment_type_effect:
                mu, sigma1, sigma2 = (
                    DEFAULT_IMPULSE_PRIOR
                    if len(self.treatment_type_effect["impulse"]) != 3
                    else (
                        self.treatment_type_effect["impulse"][0],
                        self.treatment_type_effect["impulse"][1],
                        self.treatment_type_effect["impulse"][2],
                    )
                )
                impulse_amplitude = pm.Normal("impulse_amplitude", mu=mu, sigma=sigma1)
                decay_rate = pm.HalfNormal("decay_rate", sigma=sigma2)
                impulse = pm.Deterministic(
                    "impulse",
                    impulse_amplitude
                    * pm.math.exp(-decay_rate * pm.math.abs(t - treatment_time)),
                )
                mu_in_components.append(impulse)

            # --- Parameterization ---
            weight = pm.math.sigmoid(t - treatment_time)
            # Compute and store the base time series
            mu = pm.Deterministic(name="mu", var=pm.math.dot(X, beta))
            # Compute and store the modelled intervention effect
            mu_in = (
                pm.Deterministic(name="mu_in", var=sum(mu_in_components))
                if len(mu_in_components) > 0
                else 0
            )
            # Compute and store the sum of the base time series and the intervention's effect
            mu_ts = pm.Deterministic("mu_ts", mu + weight * mu_in)
            sigma = pm.HalfNormal("sigma", 1)

            # --- Likelihood ---
            # Likelihood of the base time series
            pm.Normal("y_hat", mu=mu, sigma=sigma, dims="obs_ind")
            # Likelihodd of the base time series and the intervention's effect
            pm.Normal("y_ts", mu=mu_ts, sigma=sigma, observed=y, dims="obs_ind")

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
                var_names=["y_hat", "y_ts", "mu", "mu_ts", "mu_in"],
                progressbar=False,
                random_seed=random_seed,
            )
        return post_pred

    def calculate_impact(
        self, y_true: xr.DataArray, y_pred: az.InferenceData
    ) -> xr.DataArray:
        impact = y_true.data - y_pred["posterior_predictive"]["y_hat"]
        return impact.transpose(..., "obs_ind")

    def _data_setter(self, X) -> None:
        """
        Set data for the model.

        This method is used internally to register new data for the model for
        prediction.
        """
        new_no_of_observations = X.shape[0]
        with self:
            pm.set_data(
                {
                    "X": X,
                    "t": X.sel(coeffs=self.time_variable_name),
                    "y": np.zeros(new_no_of_observations),
                },
                coords={"obs_ind": np.arange(new_no_of_observations)},
            )

    def score(self, X, y) -> pd.Series:
        """
        Score the Bayesian :math:`R^2` given inputs ``X`` and outputs ``y``.
        """
        mu_ts = self.predict(X)
        mu_ts = az.extract(mu_ts, group="posterior_predictive", var_names="mu_ts").T
        # Note: First argument must be a 1D array
        return r2_score(y.data, mu_ts)

    def set_time_range(self, time_range, data):
        """
        Set time_range.

        :param time_range: tuple or None
            If not None, a tuple of two values (start_label, end_label) that correspond
            to index labels in the 't' column of the `data` DataFrame
        :param data: pandas.DataFrame.
        """
        if time_range is None:
            self.time_range = (
                data[self.time_variable_name].min(),
                data[self.time_variable_name].max(),
            )
        else:
            self.time_range = (
                data[self.time_variable_name].loc[time_range[0]],
                data[self.time_variable_name].loc[time_range[1]],
            )

    def get_time_variable_name(self):
        return self.time_variable_name
