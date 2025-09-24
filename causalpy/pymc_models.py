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
from patsy import dmatrix

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
    ...             self.add_coords(coords)
    ...             X_ = pm.Data(name="X", value=X)
    ...             y_ = pm.Data(name="y", value=y)
    ...             beta = pm.Normal(
    ...                 "beta", mu=0, sigma=1, shape=(y.shape[1], X.shape[1])
    ...             )
    ...             sigma = pm.HalfNormal("sigma", sigma=1, shape=y.shape[1])
    ...             mu = pm.Deterministic(
    ...                 "mu", pm.math.dot(X_, beta.T), dims=["obs_ind", "treated_units"]
    ...             )
    ...             pm.Normal("y_hat", mu=mu, sigma=sigma, observed=y_)
    >>> rng = np.random.default_rng(seed=42)
    >>> X = xr.DataArray(
    ...     rng.normal(loc=0, scale=1, size=(20, 2)),
    ...     dims=["obs_ind", "coeffs"],
    ...     coords={"obs_ind": np.arange(20), "coeffs": ["coeff_0", "coeff_1"]},
    ... )
    >>> y = xr.DataArray(
    ...     rng.normal(loc=0, scale=1, size=(20, 1)),
    ...     dims=["obs_ind", "treated_units"],
    ...     coords={"obs_ind": np.arange(20), "treated_units": ["unit_0"]},
    ... )
    >>> model = MyToyModel(
    ...     sample_kwargs={
    ...         "chains": 2,
    ...         "draws": 2000,
    ...         "progressbar": False,
    ...         "random_seed": 42,
    ...     }
    ... )
    >>> model.fit(
    ...     X,
    ...     y,
    ...     coords={
    ...         "coeffs": ["coeff_0", "coeff_1"],
    ...         "obs_ind": np.arange(20),
    ...         "treated_units": ["unit_0"],
    ...     },
    ... )
    Inference data...
    >>> model.score(X, y)  # doctest: +ELLIPSIS
    unit_0_r2        ...
    unit_0_r2_std    ...
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

    def _data_setter(self, X: xr.DataArray) -> None:
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

        # Use integer indices for obs_ind to avoid datetime compatibility issues with PyMC
        obs_coords = np.arange(new_no_of_observations)

        with self:
            # Get the number of treated units from the model coordinates
            treated_units_coord = getattr(self, "coords", {}).get(
                "treated_units", ["unit_0"]
            )
            n_treated_units = len(treated_units_coord)

            # Always use 2D format for consistency
            pm.set_data(
                {"X": X, "y": np.zeros((new_no_of_observations, n_treated_units))},
                coords={"obs_ind": obs_coords},
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

    def predict(self, X: xr.DataArray):
        """
        Predict data given input data `X`

        .. caution::
            Results in KeyError if model hasn't been fit.
        """

        # Ensure random_seed is used in sample_prior_predictive() and
        # sample_posterior_predictive() if provided in sample_kwargs.
        random_seed = self.sample_kwargs.get("random_seed", None)
        self._data_setter(X)
        with self:
            pp = pm.sample_posterior_predictive(
                self.idata,
                var_names=["y_hat", "mu"],
                progressbar=False,
                random_seed=random_seed,
            )

        # Assign coordinates from input X to ensure xarray operations work correctly
        # This is necessary because PyMC uses integer indices internally, but we need
        # to preserve the original coordinates (e.g., datetime indices) for proper
        # alignment with other xarray operations like calculate_impact()
        if isinstance(X, xr.DataArray) and "obs_ind" in X.coords:
            pp["posterior_predictive"] = pp["posterior_predictive"].assign_coords(
                obs_ind=X.obs_ind
            )

        return pp

    def score(self, X: xr.DataArray, y: xr.DataArray) -> pd.Series:
        """Score the Bayesian :math:`R^2` given inputs ``X`` and outputs ``y``.

        Note that the score is based on a comparison of the observed data ``y`` and the
        model's expected value of the data, `mu`.

        .. caution::

            The Bayesian :math:`R^2` is not the same as the traditional coefficient of
            determination, https://en.wikipedia.org/wiki/Coefficient_of_determination.

        """
        mu = self.predict(X)
        mu_data = az.extract(mu, group="posterior_predictive", var_names="mu")

        scores = {}

        # Always iterate over treated_units dimension - no branching needed!
        for i, unit in enumerate(mu_data.coords["treated_units"].values):
            unit_mu = mu_data.sel(treated_units=unit).T  # (sample, obs_ind)
            unit_y = y.sel(treated_units=unit).data
            unit_score = r2_score(unit_y, unit_mu.data)
            scores[f"unit_{i}_r2"] = unit_score["r2"]
            scores[f"unit_{i}_r2_std"] = unit_score["r2_std"]

        return pd.Series(scores)

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

        def print_coefficients_for_unit(
            unit_coeffs: xr.DataArray,
            unit_sigma: xr.DataArray,
            labels: list,
            round_to: int,
        ) -> None:
            """Print coefficients for a single unit"""
            # Determine the width of the longest label
            max_label_length = max(len(name) for name in labels + ["sigma"])

            for name in labels:
                coeff_samples = unit_coeffs.sel(coeffs=name)
                print_row(max_label_length, name, coeff_samples, round_to)

            # Add coefficient for measurement std
            print_row(max_label_length, "sigma", unit_sigma, round_to)

        print("Model coefficients:")
        coeffs = az.extract(self.idata.posterior, var_names="beta")

        # Always has treated_units dimension - no branching needed!
        treated_units = coeffs.coords["treated_units"].values
        for unit in treated_units:
            if len(treated_units) > 1:
                print(f"\nTreated unit: {unit}")

            unit_coeffs = coeffs.sel(treated_units=unit)
            unit_sigma = az.extract(self.idata.posterior, var_names="sigma").sel(
                treated_units=unit
            )
            print_coefficients_for_unit(unit_coeffs, unit_sigma, labels, round_to or 2)


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
    ...     rd["y"].values[:, None],
    ...     dims=["obs_ind", "treated_units"],
    ...     coords={"obs_ind": rd.index, "treated_units": ["unit_0"]},
    ... )
    >>> lr = LinearRegression(sample_kwargs={"progressbar": False})
    >>> coords={"coeffs": coeffs, "obs_ind": np.arange(rd.shape[0]), "treated_units": ["unit_0"]}
    >>> lr.fit(X, y, coords=coords)
    Inference data...
    """  # noqa: W605

    def build_model(self, X, y, coords):
        """
        Defines the PyMC model
        """
        with self:
            # Ensure treated_units coordinate exists for consistency
            if "treated_units" not in coords:
                coords = coords.copy()
                coords["treated_units"] = ["unit_0"]

            self.add_coords(coords)
            X = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            y = pm.Data("y", y, dims=["obs_ind", "treated_units"])
            beta = pm.Normal("beta", 0, 50, dims=["treated_units", "coeffs"])
            sigma = pm.HalfNormal("sigma", 1, dims="treated_units")
            mu = pm.Deterministic(
                "mu", pt.dot(X, beta.T), dims=["obs_ind", "treated_units"]
            )
            pm.Normal("y_hat", mu, sigma, observed=y, dims=["obs_ind", "treated_units"])


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
    >>> import xarray as xr
    >>> from causalpy.pymc_models import WeightedSumFitter
    >>> sc = cp.load_data("sc")
    >>> control_units = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    >>> X = xr.DataArray(
    ...     sc[control_units].values,
    ...     dims=["obs_ind", "coeffs"],
    ...     coords={"obs_ind": sc.index, "coeffs": control_units},
    ... )
    >>> y = xr.DataArray(
    ...     sc['actual'].values.reshape((sc.shape[0], 1)),
    ...     dims=["obs_ind", "treated_units"],
    ...     coords={"obs_ind": sc.index, "treated_units": ["actual"]},
    ... )
    >>> coords = {
    ...     "coeffs": control_units,
    ...     "treated_units": ["actual"],
    ...     "obs_ind": np.arange(sc.shape[0]),
    ... }
    >>> wsf = WeightedSumFitter(sample_kwargs={"progressbar": False})
    >>> wsf.fit(X, y, coords=coords)
    Inference data...
    """  # noqa: W605

    def build_model(self, X, y, coords):
        """
        Defines the PyMC model
        """
        with self:
            self.add_coords(coords)
            n_predictors = X.sizes["coeffs"]
            X = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            y = pm.Data("y", y, dims=["obs_ind", "treated_units"])
            beta = pm.Dirichlet(
                "beta", a=np.ones(n_predictors), dims=["treated_units", "coeffs"]
            )
            sigma = pm.HalfNormal("sigma", 1, dims="treated_units")
            mu = pm.Deterministic(
                "mu", pt.dot(X, beta.T), dims=["obs_ind", "treated_units"]
            )
            pm.Normal("y_hat", mu, sigma, observed=y, dims=["obs_ind", "treated_units"])


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
            mu_y = pm.Deterministic(name="mu_y", var=pt.dot(X, beta_z))
            # focal regression
            mu_t = pm.Deterministic(name="mu_t", var=pt.dot(Z, beta_t))
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
    ...                prior={'b': [0, 1]},
    ... )
    Inference...
    """  # noqa: W605

    def build_model(self, X, t, coords, prior, noncentred):
        "Defines the PyMC propensity model"
        with self:
            self.add_coords(coords)
            X_data = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            t_data = pm.Data("t", t.flatten(), dims="obs_ind")
            if noncentred:
                mu_beta, sigma_beta = prior["b"]
                beta_std = pm.Normal("beta_std", 0, 1, dims="coeffs")
                b = pm.Deterministic(
                    "beta_", mu_beta + sigma_beta * beta_std, dims="coeffs"
                )
            else:
                b = pm.Normal("b", mu=prior["b"][0], sigma=prior["b"][1], dims="coeffs")
            mu = pm.math.dot(X_data, b)
            p = pm.Deterministic("p", pm.math.invlogit(mu))
            pm.Bernoulli("t_pred", p=p, observed=t_data, dims="obs_ind")

    def fit(self, X, t, coords, prior={"b": [0, 1]}, noncentred=True):
        """Draw samples from posterior, prior predictive, and posterior predictive
        distributions. We overwrite the base method because the base method assumes
        a variable y and we use t to indicate the treatment variable here.
        """
        # Ensure random_seed is used in sample_prior_predictive() and
        # sample_posterior_predictive() if provided in sample_kwargs.
        random_seed = self.sample_kwargs.get("random_seed", None)

        self.build_model(X, t, coords, prior, noncentred)
        with self:
            self.idata = pm.sample(**self.sample_kwargs)
            self.idata.extend(pm.sample_prior_predictive(random_seed=random_seed))
            self.idata.extend(
                pm.sample_posterior_predictive(
                    self.idata, progressbar=False, random_seed=random_seed
                )
            )
        return self.idata

    def fit_outcome_model(
        self,
        X_outcome,
        y,
        coords,
        priors={
            "b_outcome": [0, 1],
            "sigma": 1,
            "beta_ps": [0, 1],
        },
        noncentred=True,
        normal_outcome=True,
        spline_component=False,
        winsorize_boundary=0.0,
    ):
        """
        Fit a Bayesian outcome model using covariates and previously estimated propensity scores.

        This function implements the second stage of a modular two-step causal inference procedure.
        It uses propensity scores extracted from a prior treatment model (via `self.fit()`) to adjust
        for confounding when estimating treatment effects on an outcome variable `y`.

        Parameters
        ----------
        X_outcome : array-like, shape (n_samples, n_covariates)
            Covariate matrix for the outcome model.

        y : array-like, shape (n_samples,)
            Observed outcome variable.

        coords : dict
            Coordinate dictionary for named dimensions in the PyMC model. Should include
            a key "outcome_coeffs" for `X_outcome`.

        priors : dict, optional
            Dictionary specifying priors for outcome model parameters:
                - "b_outcome": list [mean, std] for regression coefficients.
                - "sigma": standard deviation of the outcome noise (default 1).

        noncentred : bool, default True
            If True, use a non-centred parameterization for the outcome coefficients.

        normal_outcome : bool, default True
            If True, assume a Normal likelihood for the outcome.
            If False, use a Student-t likelihood with unknown degrees of freedom.

        spline_component : bool, default False
            If True, include a spline basis expansion on the propensity score to allow
            flexible (nonlinear) adjustment. Uses B-splines with 30 internal knots.

        winsorize_boundary : float, default 0.0
            If we wish to winsorize the propensity score this can be set to clip the high
            and low values of the propensity at 0 + winsorize_boundary and 1-winsorize_boundary

        Returns
        -------
        idata_outcome : arviz.InferenceData
            The posterior and prior predictive samples from the outcome model.

        model_outcome : pm.Model
            The PyMC model object.

        Raises
        ------
        AttributeError
            If the `self.idata` attribute is not available, which indicates that
            `fit()` (i.e., the treatment model) has not been called yet.

        Notes
        -----
        - This model uses a sampled version of the propensity score (`p`) from the
        posterior of the treatment model, randomly selecting one posterior draw
        per call. This term is estimated initially in the InversePropensity
        class initialisation.
        - The term `beta_ps[0] * p` captures both
        main effects of the propensity score.
        - Including spline adjustment enables modeling nonlinear relationships
        between the propensity score and the outcome.

        """
        if not hasattr(self, "idata"):
            raise AttributeError("""Object is missing required attribute 'idata'
                                 so cannot proceed. Call fit() first""")
        propensity_scores = az.extract(self.idata)["p"]
        random_seed = self.sample_kwargs.get("random_seed", None)

        with pm.Model(coords=coords) as model_outcome:
            X_data_outcome = pm.Data("X_outcome", X_outcome)
            Y_data_ = pm.Data("Y", y)

            if noncentred:
                mu_beta, sigma_beta = priors["b_outcome"]
                beta_std = pm.Normal("beta_std", 0, 1, dims="outcome_coeffs")
                beta = pm.Deterministic(
                    "beta_", mu_beta + sigma_beta * beta_std, dims="outcome_coeffs"
                )
            else:
                beta = pm.Normal(
                    "beta_",
                    priors["b_outcome"][0],
                    priors["b_outcome"][1],
                    dims="outcome_coeffs",
                )

            beta_ps = pm.Normal("beta_ps", priors["beta_ps"][0], priors["beta_ps"][1])

            chosen = np.random.choice(range(propensity_scores.shape[1]))
            p = propensity_scores[:, chosen].values
            p = np.clip(p, winsorize_boundary, 1 - winsorize_boundary)

            mu_outcome = pm.math.dot(X_data_outcome, beta) + beta_ps * p

            if spline_component:
                beta_ps_spline = pm.Normal(
                    "beta_ps_spline",
                    priors["beta_ps"][0],
                    priors["beta_ps"][1],
                    size=34,
                )
                B = dmatrix(
                    "bs(ps, knots=knots, degree=3, include_intercept=True, lower_bound=0, upper_bound=1) - 1",
                    {"ps": p, "knots": np.linspace(0, 1, 30)},
                )
                B_f = np.asarray(B, order="F")
                splines_summed = pm.Deterministic(
                    "spline_features", pm.math.dot(B_f, beta_ps_spline.T)
                )
                mu_outcome = pm.math.dot(X_data_outcome, beta) + splines_summed

            sigma = pm.HalfNormal("sigma", priors["sigma"])

            if normal_outcome:
                _ = pm.Normal("like", mu_outcome, sigma, observed=Y_data_)
            else:
                nu = pm.Exponential("nu", lam=1 / 10)
                _ = pm.StudentT(
                    "like", nu=nu, mu=mu_outcome, sigma=sigma, observed=Y_data_
                )

            idata_outcome = pm.sample_prior_predictive(random_seed=random_seed)
            idata_outcome.extend(pm.sample(**self.sample_kwargs))

        return idata_outcome, model_outcome


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
        >>> model = ITE(treatment_effect_type="level", sample_kwargs={"draws" : 10, "tune":10, "progressbar":False})
        >>> model.set_time_range(None, data)
        >>> model.fit(X=_X, y=_y, coords=COORDS)
        Inference ...
    """

    def __init__(
        self,
        treatment_effect_type: str | list[str],
        treatment_effect_param=None,
        sample_kwargs=None,
    ):
        """
        Initializes the InterventionTimeEstimator model.

        :param treatment_effect_type: Optional dictionary that specifies prior parameters for the
            intervention effects. Expected keys are:
                - "level": [mu, sigma]
                - "trend": [mu, sigma]
                - "impulse": [mu, sigma1, sigma2]
            If a key is missing, the corresponding effect is ignored.
            If the associated list is incomplete, default values will be used.
        :param sample_kwargs: Optional dictionary of arguments passed to pm.sample().
        """

        super().__init__(sample_kwargs)

        # Hardcoded default priors
        self.DEFAULT_BETA_PRIOR = (0, 5)
        self.DEFAULT_LEVEL_PRIOR = (0, 5)
        self.DEFAULT_TREND_PRIOR = (0, 0.5)
        self.DEFAULT_IMPULSE_PRIOR = (0, 5, 5)

        # Make sure we get a list of all expected effects
        if isinstance(treatment_effect_type, str):
            self.treatment_effect_type = [treatment_effect_type]
        else:
            self.treatment_effect_type = treatment_effect_type

        # Defining the priors here
        self.treatment_effect_param = (
            {} if treatment_effect_param is None else treatment_effect_param
        )

        if "level" in self.treatment_effect_type:
            if (
                "level" not in self.treatment_effect_param
                or len(self.treatment_effect_param["level"]) != 2
            ):
                self.treatment_effect_param["level"] = self.DEFAULT_LEVEL_PRIOR
            else:
                self.treatment_effect_param["level"] = self.treatment_effect_param[
                    "level"
                ]

        if "trend" in self.treatment_effect_type:
            if (
                "trend" not in self.treatment_effect_param
                or len(self.treatment_effect_param["trend"]) != 2
            ):
                self.treatment_effect_param["trend"] = self.DEFAULT_TREND_PRIOR
            else:
                self.treatment_effect_param["trend"] = self.treatment_effect_param[
                    "trend"
                ]

        if "impulse" in self.treatment_effect_type:
            if (
                "impulse" not in self.treatment_effect_param
                or len(self.treatment_effect_param["impulse"]) != 3
            ):
                self.treatment_effect_param["impulse"] = self.DEFAULT_IMPULSE_PRIOR
            else:
                self.treatment_effect_param["impulse"] = self.treatment_effect_param[
                    "impulse"
                ]

    def build_model(self, X, y, coords):
        """
        Defines the PyMC model

        :param X: An array of the covariates
        :param y: An array of values representing our outcome y
        :param coords: Dictionary of named coordinates for PyMC variables (e.g., {"obs_ind": range(n_obs), "coeffs": range(n_covariates)}).

        Assumes the following attributes are already defined in self:
            - self.timeline: the index of the column in X representing time.
            - self.time_range: a tuple (lower_bound, upper_bound) for the intervention time.
            - self.treatment_effect_type: a dictionary specifying which intervention effects to include and their priors.
        """

        with self:
            self.add_coords(coords)

            t = pm.Data("t", np.arange(len(X)), dims="obs_ind")
            X = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            y = pm.Data("y", y, dims=["obs_ind", "treated_units"])

            lower_bound = pm.Data("lower_bound", self.time_range[0])
            upper_bound = pm.Data("upper_bound", self.time_range[1])

            # --- Priors ---
            treatment_time = pm.Uniform(
                "treatment_time", lower=lower_bound, upper=upper_bound
            )
            delta_t = pm.Deterministic(
                name="delta_t",
                var=(t - treatment_time)[:, None],
                dims=["obs_ind", "treated_units"],
            )
            beta = pm.Normal(
                name="beta",
                mu=self.DEFAULT_BETA_PRIOR[0],
                sigma=self.DEFAULT_BETA_PRIOR[1],
                dims=["treated_units", "coeffs"],
            )

            # --- Intervention effect ---
            mu_in_components = []

            if "level" in self.treatment_effect_param:
                level = pm.Normal(
                    "level",
                    mu=self.treatment_effect_param["level"][0],
                    sigma=self.treatment_effect_param["level"][1],
                    dims=["obs_ind", "treated_units"],
                )
                mu_in_components.append(level)
            if "trend" in self.treatment_effect_param:
                trend = pm.Normal(
                    "trend",
                    mu=self.treatment_effect_param["trend"][0],
                    sigma=self.treatment_effect_param["trend"][1],
                    dims=["obs_ind", "treated_units"],
                )
                mu_in_components.append(trend * delta_t)
            if "impulse" in self.treatment_effect_param:
                impulse_amplitude = pm.Normal(
                    "impulse_amplitude",
                    mu=self.treatment_effect_param["impulse"][0],
                    sigma=self.treatment_effect_param["impulse"][1],
                    dims=["obs_ind", "treated_units"],
                )
                decay_rate = pm.HalfNormal(
                    "decay_rate",
                    sigma=self.treatment_effect_param["impulse"][2],
                    dims=["obs_ind", "treated_units"],
                )
                impulse = pm.Deterministic(
                    "impulse",
                    impulse_amplitude * pm.math.exp(-decay_rate * pm.math.abs(delta_t)),
                    dims=["obs_ind", "treated_units"],
                )
                mu_in_components.append(impulse)

            # --- Parameterization ---
            weight = pm.math.sigmoid(delta_t)
            # Compute and store the base time series
            mu = pm.Deterministic(
                name="mu", var=pm.math.dot(X, beta.T), dims=["obs_ind", "treated_units"]
            )
            # Compute and store the modelled intervention effect
            mu_in = (
                pm.Deterministic(
                    name="mu_in",
                    var=sum(mu_in_components),
                    dims=["obs_ind", "treated_units"],
                )
                if len(mu_in_components) > 0
                else pm.Data(
                    name="mu_in",
                    vars=np.zeros((X.sizes["obs_ind"], y.sizes["treated_units"])),
                    dims=["obs_ind", "treated_units"],
                )
            )
            # Compute and store the sum of the base time series and the intervention's effect
            mu_ts = pm.Deterministic(
                "mu_ts", mu + weight * mu_in, dims=["obs_ind", "treated_units"]
            )
            sigma = pm.HalfNormal("sigma", 1, dims="treated_units")

            # --- Likelihood ---
            # Likelihood of the base time series
            pm.Normal("y_hat", mu=mu, sigma=sigma, dims=["obs_ind", "treated_units"])
            # Likelihodd of the base time series and the intervention's effect
            pm.Normal(
                "y_ts",
                mu=mu_ts,
                sigma=sigma,
                observed=y,
                dims=["obs_ind", "treated_units"],
            )

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
                var_names=["y_hat", "y_ts", "mu", "mu_ts", "mu_in"],
                progressbar=False,
                random_seed=random_seed,
            )

        # TODO: This is a bit of a hack. Maybe it could be done properly in _data_setter?
        if isinstance(X, xr.DataArray):
            pp["posterior_predictive"] = pp["posterior_predictive"].assign_coords(
                obs_ind=X.obs_ind
            )

        return pp

    def _data_setter(self, X) -> None:
        """
        Set data for the model.

        This method is used internally to register new data for the model for
        prediction.
        """
        n_obs = X.shape[0]
        with self:
            treated_units_coord = getattr(self, "coords", {}).get(
                "treated_units", ["unit_0"]
            )
            pm.set_data(
                {
                    "X": X,
                    "t": np.arange(n_obs)[:, None],
                    "y": np.zeros((n_obs, treated_units_coord)),
                },
                coords={"obs_ind": np.arange(n_obs)},
            )

    def score(self, X, y) -> pd.Series:
        """
        Score the Bayesian :math:`R^2` given inputs ``X`` and outputs ``y``.
        """
        mu_ts = self.predict(X)
        mu_ts = az.extract(mu_ts, group="posterior_predictive", var_names="mu_ts").T
        # Note: First argument must be a 1D array
        return r2_score(y.data, mu_ts.data)

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
                0,
                len(data),
            )
        else:
            self.time_range = (
                data.index.get_loc(time_range[0]),
                data.index.get_loc(time_range[1]),
            )
