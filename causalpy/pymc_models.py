#   Copyright 2022 - 2026 The PyMC Labs Developers
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

import warnings
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from arviz import r2_score
from patsy import dmatrix
from pymc_extras.prior import Prior

from causalpy.utils import round_num
from causalpy.variable_selection_priors import VariableSelectionPrior


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

    default_priors: dict[str, Prior] = {}

    def priors_from_data(self, X, y) -> dict[str, Any]:
        """
        Generate priors dynamically based on the input data.

        This method allows models to set sensible priors that adapt to the scale
        and characteristics of the actual data being analyzed. It's called during
        the `fit()` method before model building, allowing data-driven prior
        specification that can improve model performance and convergence.

        The priors returned by this method are merged with any user-specified
        priors (passed via the `priors` parameter in `__init__`), with
        user-specified priors taking precedence in case of conflicts.

        Parameters
        ----------
        X : xarray.DataArray
            Input features/covariates with dimensions ["obs_ind", "coeffs"].
            Used to understand the scale and structure of predictors.
        y : xarray.DataArray
            Target variable with dimensions ["obs_ind", "treated_units"].
            Used to understand the scale and structure of the outcome.

        Returns
        -------
        Dict[str, Prior]
            Dictionary mapping parameter names to Prior objects. The keys should
            match parameter names used in the model's `build_model()` method.

        Notes
        -----
        The base implementation returns an empty dictionary, meaning no
        data-driven priors are set by default. Subclasses should override
        this method to implement data-adaptive prior specification.

        **Priority Order for Priors:**
        1. User-specified priors (passed to `__init__`)
        2. Data-driven priors (from this method)
        3. Default priors (from `default_priors` property)

        Examples
        --------
        A typical implementation might scale priors based on data variance:

        >>> def priors_from_data(self, X, y):
        ...     y_std = float(y.std())
        ...     return {
        ...         "sigma": Prior("HalfNormal", sigma=y_std, dims="treated_units"),
        ...         "beta": Prior(
        ...             "Normal",
        ...             mu=0,
        ...             sigma=2 * y_std,
        ...             dims=["treated_units", "coeffs"],
        ...         ),
        ...     }

        Or set shape parameters based on data dimensions:

        >>> def priors_from_data(self, X, y):
        ...     n_predictors = X.shape[1]
        ...     return {
        ...         "beta": Prior(
        ...             "Dirichlet",
        ...             a=np.ones(n_predictors),
        ...             dims=["treated_units", "coeffs"],
        ...         )
        ...     }

        See Also
        --------
        WeightedSumFitter.priors_from_data : Example implementation that sets
            Dirichlet prior shape based on number of control units.
        """
        return {}

    def __init__(
        self,
        sample_kwargs: dict[str, Any] | None = None,
        priors: dict[str, Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        sample_kwargs : dict, optional
            Dictionary of kwargs that get unpacked and passed to the
            :func:`pymc.sample` function. Defaults to an empty dictionary
            if None.
        priors : dict, optional
            Dictionary of priors for the model. Defaults to None, in which
            case default priors are used.
        """
        super().__init__()
        self.idata = None
        self.sample_kwargs = sample_kwargs if sample_kwargs is not None else {}

        self.priors = {**self.default_priors, **(priors or {})}

    def build_model(
        self, X: xr.DataArray, y: xr.DataArray, coords: dict[str, Any] | None
    ) -> None:
        raise NotImplementedError(
            "This method must be implemented by a subclass"
        )  # pragma: no cover

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

    def fit(
        self, X: xr.DataArray, y: xr.DataArray, coords: dict[str, Any] | None = None
    ) -> az.InferenceData:
        """Draw samples from posterior, prior predictive, and posterior
        predictive distributions.

        Parameters
        ----------
        X : xr.DataArray
            Input features as an xarray DataArray.
        y : xr.DataArray
            Target variable as an xarray DataArray.
        coords : dict, optional
            Dictionary with coordinate names for named dimensions.
            Defaults to None.

        Returns
        -------
        az.InferenceData
            InferenceData object containing the samples.
        """

        # Ensure random_seed is used in sample_prior_predictive() and
        # sample_posterior_predictive() if provided in sample_kwargs.
        random_seed = self.sample_kwargs.get("random_seed", None)

        # Merge priors with precedence: user-specified > data-driven > defaults
        # Data-driven priors are computed first, then user-specified priors override them
        self.priors = {**self.priors_from_data(X, y), **self.priors}

        self.build_model(X, y, coords)
        with self:
            self.idata = pm.sample(**self.sample_kwargs)
            if self.idata is None:
                raise RuntimeError("pm.sample() returned None")
            self.idata.extend(pm.sample_prior_predictive(random_seed=random_seed))
            self.idata.extend(
                pm.sample_posterior_predictive(
                    self.idata, progressbar=False, random_seed=random_seed
                )
            )
        return self.idata

    def predict(
        self,
        X: xr.DataArray,
        coords: dict[str, Any] | None = None,
        out_of_sample: bool | None = False,
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

    def score(self, X, y, coords: dict[str, Any] | None = None, **kwargs) -> pd.Series:
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
        """
        Calculate the causal impact as the difference between observed and predicted values.

        The impact is calculated using the posterior expectation (`mu`) rather than the
        posterior predictive (`y_hat`). This means the causal impact represents the
        difference from the expected value of the model, excluding observation noise.
        This approach provides a cleaner measure of the causal effect by focusing on
        the systematic difference rather than including sampling variability from the
        observation noise term.

        Parameters
        ----------
        y_true : xr.DataArray
            The observed outcome values with dimensions ["obs_ind", "treated_units"].
        y_pred : az.InferenceData
            The posterior predictive samples containing the "mu" variable, which
            represents the expected value (mean) of the outcome.

        Returns
        -------
        xr.DataArray
            The causal impact with dimensions ending in "obs_ind". The impact includes
            posterior uncertainty from the model parameters but excludes observation noise.

        Notes
        -----
        By using `mu` (the posterior expectation) rather than `y_hat` (the posterior
        predictive with observation noise), the uncertainty in the impact reflects:
        - Parameter uncertainty in the fitted model
        - Uncertainty in the counterfactual prediction

        But excludes:
        - Observation-level noise (sigma)

        This makes the impact plots focus on the systematic causal effect rather than
        individual observation variability.
        """
        y_hat = y_pred["posterior_predictive"]["mu"]
        # Ensure the coordinate type and values match along obs_ind so xarray can align
        if "obs_ind" in y_hat.dims and "obs_ind" in getattr(y_true, "coords", {}):
            try:
                # Assign the same coordinate values (e.g., DatetimeIndex) to prediction
                y_hat = y_hat.assign_coords(obs_ind=y_true["obs_ind"])  # type: ignore[index]
            except Exception:
                # If assignment fails, fall back to position-based subtraction
                # by temporarily dropping coords to avoid dtype promotion issues
                y_hat = y_hat.reset_coords(names=["obs_ind"], drop=True)
                y_true = y_true.reset_coords(names=["obs_ind"], drop=True)
        impact = y_true - y_hat
        return impact.transpose(..., "obs_ind")

    def calculate_cumulative_impact(self, impact: xr.DataArray) -> xr.DataArray:
        return impact.cumsum(dim="obs_ind")

    def print_coefficients(
        self, labels: list[str], round_to: int | None = None
    ) -> None:
        """Print the model coefficients with their labels.

        Parameters
        ----------
        labels : list of str
            List of strings representing the coefficient names.
        round_to : int, optional
            Number of significant figures to round to. Defaults to None,
            in which case 2 significant figures are used.
        """
        if self.idata is None:
            raise RuntimeError("Model has not been fit")

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
            max_label_length = max(len(name) for name in labels + ["y_hat_sigma"])

            for name in labels:
                coeff_samples = unit_coeffs.sel(coeffs=name)
                print_row(max_label_length, name, coeff_samples, round_to)

            # Add coefficient for measurement std
            print_row(max_label_length, "y_hat_sigma", unit_sigma, round_to)

        print("Model coefficients:")
        coeffs = az.extract(self.idata.posterior, var_names="beta")

        # Check if sigma or y_hat_sigma variable exists
        sigma_var_name = None
        if "sigma" in self.idata.posterior:
            sigma_var_name = "sigma"
        elif "y_hat_sigma" in self.idata.posterior:
            sigma_var_name = "y_hat_sigma"
        else:
            raise ValueError(
                "Neither 'sigma' nor 'y_hat_sigma' found in posterior"
            )  # pragma: no cover

        treated_units = coeffs.coords["treated_units"].values
        for unit in treated_units:
            if len(treated_units) > 1:
                print(f"\nTreated unit: {unit}")

            unit_coeffs = coeffs.sel(treated_units=unit)
            unit_sigma = az.extract(self.idata.posterior, var_names=sigma_var_name).sel(
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

    default_priors = {
        "beta": Prior("Normal", mu=0, sigma=50, dims=["treated_units", "coeffs"]),
        "y_hat": Prior(
            "Normal",
            sigma=Prior("HalfNormal", sigma=1, dims=["treated_units"]),
            dims=["obs_ind", "treated_units"],
        ),
    }

    def build_model(
        self, X: xr.DataArray, y: xr.DataArray, coords: dict[str, Any] | None
    ) -> None:
        """
        Defines the PyMC model
        """
        with self:
            # Ensure treated_units coordinate exists for consistency
            if coords is not None and "treated_units" not in coords:
                coords = coords.copy()
                coords["treated_units"] = ["unit_0"]

            self.add_coords(coords)
            X = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            y = pm.Data("y", y, dims=["obs_ind", "treated_units"])
            beta = self.priors["beta"].create_variable("beta")
            mu = pm.Deterministic(
                "mu", pt.dot(X, beta.T), dims=["obs_ind", "treated_units"]
            )
            self.priors["y_hat"].create_likelihood_variable("y_hat", mu=mu, observed=y)


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

    default_priors = {
        "y_hat": Prior(
            "Normal",
            sigma=Prior("HalfNormal", sigma=1, dims=["treated_units"]),
            dims=["obs_ind", "treated_units"],
        ),
    }

    def priors_from_data(self, X, y) -> dict[str, Any]:
        """
        Set Dirichlet prior for weights based on number of control units.

        For synthetic control models, this method sets the shape parameter of the
        Dirichlet prior on the control unit weights (`beta`) to be uniform across
        all available control units. This ensures that all control units have
        equal prior probability of contributing to the synthetic control.

        Parameters
        ----------
        X : xarray.DataArray
            Control unit data with shape (n_obs, n_control_units).
        y : xarray.DataArray
            Treated unit outcome data.

        Returns
        -------
        Dict[str, Prior]
            Dictionary containing:
            - "beta": Dirichlet prior with shape=(1,...,1) for n_control_units
        """
        n_predictors = X.shape[1]
        return {
            "beta": Prior(
                "Dirichlet", a=np.ones(n_predictors), dims=["treated_units", "coeffs"]
            ),
        }

    def build_model(
        self, X: xr.DataArray, y: xr.DataArray, coords: dict[str, Any] | None
    ) -> None:
        """
        Defines the PyMC model
        """
        with self:
            self.add_coords(coords)
            X = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            y = pm.Data("y", y, dims=["obs_ind", "treated_units"])
            beta = self.priors["beta"].create_variable("beta")
            mu = pm.Deterministic(
                "mu", pt.dot(X, beta.T), dims=["obs_ind", "treated_units"]
            )
            self.priors["y_hat"].create_likelihood_variable("y_hat", mu=mu, observed=y)


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

    def build_model(  # type: ignore
        self,
        X: np.ndarray,
        Z: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        coords: dict[str, Any],
        priors: dict[str, Any],
        vs_prior_type: Literal["spike_and_slab", "horseshoe", "normal"] | None = None,
        vs_hyperparams: dict[str, Any] | None = None,
        binary_treatment: bool = False,
    ) -> None:
        """Specify model with treatment regression and focal regression
        data and priors.

        Parameters
        ----------
        X : np.ndarray
            Array used to predict our outcome y.
        Z : np.ndarray
            Array used to predict our treatment variable t.
        y : np.ndarray
            Array of values representing our focal outcome y.
        t : np.ndarray
            Array representing the treatment t of which we're interested
            in estimating the causal impact.
        coords : dict
            Dictionary with the coordinate names for our instruments and
            covariates.
        priors : dict
            Dictionary of priors for the mus and sigmas of both
            regressions. Example: ``priors = {"mus": [0, 0],
            "sigmas": [1, 1], "eta": 2, "lkj_sd": 2}``.
        vs_prior_type: An optional string. Can be "spike_and_slab"
                              or "horseshoe" or "normal
        vs_hyperparams: An optional dictionary of priors for the
                               variable selection hyperparameters
        binary_treatment: A flag for determining the relevant
                                likelihood to be used.

        """

        # --- Priors ---
        with self:
            self.add_coords(coords)

            if vs_prior_type and ("mus" in priors or "sigmas" in priors):
                warnings.warn(
                    "Variable selection priors specified. "
                    "The 'mus' and 'sigmas' in the priors dict will be ignored "
                    "for beta coefficients in the treatment equation."
                    "Only 'eta' and 'lkj_sd' will be used from the priors dict"
                    "where applicable.",
                    stacklevel=2,
                )

            # Create coefficient priors
            if vs_prior_type:
                if vs_hyperparams is None:
                    vs_hyperparams = {}
                # Use variable selection priors
                self.vs_prior_treatment = VariableSelectionPrior(
                    vs_prior_type, vs_hyperparams
                )
                self.vs_prior_outcome = VariableSelectionPrior(
                    vs_prior_type, vs_hyperparams
                )

                beta_t = self.vs_prior_treatment.create_prior(
                    name="beta_t", n_params=Z.shape[1], dims="instruments", X=Z
                )
                if vs_hyperparams.get("outcome", False):
                    beta_z = self.vs_prior_outcome.create_prior(
                        name="beta_z", n_params=X.shape[1], dims="covariates", X=X
                    )
                else:  # Fallback to standard normal priors for outcome
                    beta_z = pm.Normal(
                        name="beta_z",
                        mu=priors["mus"][1],
                        sigma=priors["sigmas"][1],
                        dims="covariates",
                    )
            else:
                # Use standard normal priors
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

            if binary_treatment:
                # Binary treatment formulation with correlated latent errors
                sigma_U = pm.Exponential("sigma_U", priors.get("sigma_U", 1.0))

                # Correlation/Sensitivity parameter with bounds
                # 'rho' represents the coupling between the Logistic latent error (V)
                # and the Normal outcome error (U).
                # Note: Because V follows a Standard Logistic distribution (heavy tails),
                # this value is not directly comparable to a Normal-Normal Pearson rho.
                # It acts as the sensitivity parameter in the Control Function approach.
                rho_lower = priors.get("rho_bounds", [-0.99, 0.99])[0]
                rho_upper = priors.get("rho_bounds", [-0.99, 0.99])[1]

                # Use tanh transform to keep correlation in valid range
                rho_unconstr = pm.Normal("rho_unconstr", 0, 0.5)
                rho = pm.Deterministic("rho", pm.math.tanh(rho_unconstr))

                # Clip to ensure numerical stability
                rho_clipped = pt.clip(rho, rho_lower + 0.01, rho_upper - 0.01)

                u = pm.Uniform("u", 0, 1, shape=X.shape[0])
                # 2. Transform to Standard Logistic space
                # This is the "residual" in the treatment equation
                V = pm.Deterministic("V", pt.log(u / (1 - u)))

                # Treatment equation (logit link for binary treatment)
                # much more stable than probit link in practice
                mu_treatment = pm.Deterministic("mu_t", pt.dot(Z, beta_t) + V)
                p_t = pm.math.invlogit(mu_treatment)
                pm.Bernoulli("likelihood_treatment", p=p_t, observed=t.flatten())

                # Conditional Outcome equation formulation
                mu_outcome = pm.Deterministic("mu_y", pt.dot(X, beta_z))
                sigma_v_logistic = pm.math.sqrt(pt.pi**2 / 3)
                expected_U = rho_clipped * (sigma_U / sigma_v_logistic) * V

                conditional_mu_y = mu_outcome + expected_U
                conditional_sigma_y = sigma_U * pm.math.sqrt(1 - rho_clipped**2)
                pm.Normal(
                    "likelihood_outcome",
                    mu=conditional_mu_y,
                    sigma=conditional_sigma_y,
                    observed=y.flatten(),
                )

            else:
                sd_dist = pm.Exponential.dist(priors["lkj_sd"], shape=2)
                chol, _, _ = pm.LKJCholeskyCov(
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
                mu = pm.Deterministic(
                    name="mu", var=pt.stack(tensors=(mu_y, mu_t), axis=1)
                )

                # --- Likelihood ---
                pm.MvNormal(
                    name="likelihood",
                    mu=mu,
                    chol=chol,
                    observed=np.stack(arrays=(y.flatten(), t.flatten()), axis=1),
                    shape=(X.shape[0], 2),
                )

    def sample_predictive_distribution(self, ppc_sampler: str | None = "jax") -> None:
        """Function to sample the Multivariate Normal posterior predictive
        Likelihood term in the IV class. This can be slow without
        using the JAX sampler compilation method. If using the
        JAX sampler it will sample only the posterior predictive distribution.
        If using the PYMC sampler if will sample both the prior
        and posterior predictive distributions."""
        random_seed = self.sample_kwargs.get("random_seed", None)

        if ppc_sampler == "jax":
            if self.idata is not None:
                with self:
                    self.idata.extend(
                        pm.sample_posterior_predictive(
                            self.idata,
                            random_seed=random_seed,
                            compile_kwargs={"mode": "JAX"},
                        )
                    )
        elif ppc_sampler == "pymc" and self.idata is not None:
            with self:
                self.idata.extend(pm.sample_prior_predictive(random_seed=random_seed))
                self.idata.extend(
                    pm.sample_posterior_predictive(
                        self.idata,
                        random_seed=random_seed,
                    )
                )

    def fit(  # type: ignore[override]
        self,
        X: np.ndarray,
        Z: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        coords: dict[str, Any],
        priors: dict[str, Any],
        ppc_sampler: Literal["jax", "pymc"] | None = None,
        vs_prior_type: Literal["spike_and_slab", "horseshoe", "normal"] | None = None,
        vs_hyperparams: dict[str, Any] | None = None,
        binary_treatment: bool = False,
    ) -> az.InferenceData:  # type: ignore[override]
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

        self.build_model(
            X, Z, y, t, coords, priors, vs_prior_type, vs_hyperparams, binary_treatment
        )
        with self:
            self.idata = pm.sample(**self.sample_kwargs)
        self.sample_predictive_distribution(ppc_sampler=ppc_sampler)
        assert self.idata is not None
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

    default_priors = {
        "b": Prior("Normal", mu=0, sigma=1, dims="coeffs"),
    }

    def build_model(  # type: ignore
        self,
        X: np.ndarray,
        t: np.ndarray,
        coords: dict[str, Any],
        prior: dict[str, Any] | None = None,
        noncentred: bool = True,
    ) -> None:
        "Defines the PyMC propensity model"
        with self:
            self.add_coords(coords)
            X_data = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            t_data = pm.Data("t", t.flatten(), dims="obs_ind")
            b = self.priors["b"].create_variable("b")
            mu = pt.dot(X_data, b)
            p = pm.Deterministic("p", pm.math.invlogit(mu))
            pm.Bernoulli("t_pred", p=p, observed=t_data, dims="obs_ind")

    def fit(  # type: ignore
        self,
        X: np.ndarray,
        t: np.ndarray,
        coords: dict[str, Any],
        prior: dict[str, list] | None = None,
        noncentred: bool = True,
    ) -> az.InferenceData:
        """Draw samples from posterior, prior predictive, and posterior predictive
        distributions. We overwrite the base method because the base method assumes
        a variable y and we use t to indicate the treatment variable here.
        """
        if prior is None:
            prior = {"b": [0, 1]}
        # Ensure random_seed is used in sample_prior_predictive() and
        # sample_posterior_predictive() if provided in sample_kwargs.
        random_seed = self.sample_kwargs.get("random_seed", None)

        self.build_model(X, t, coords, prior, noncentred)
        with self:
            self.idata = pm.sample(**self.sample_kwargs)
            if self.idata is not None:
                self.idata.extend(pm.sample_prior_predictive(random_seed=random_seed))
                self.idata.extend(
                    pm.sample_posterior_predictive(
                        self.idata, progressbar=False, random_seed=random_seed
                    )
                )
        assert self.idata is not None
        return self.idata

    def fit_outcome_model(
        self,
        X_outcome: pd.DataFrame,
        y: pd.Series,
        coords: dict[str, Any],
        priors: dict[str, Any] | None = None,
        noncentred: bool = True,
        normal_outcome: bool = True,
        spline_component: bool = False,
        winsorize_boundary: float = 0.0,
        spline_knots: int = 30,
    ) -> tuple[az.InferenceData, pm.Model]:
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

        spline_knots: int, default 30
            The number of knots we use in the 0 - 1 interval to create our spline function

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
        if priors is None:
            priors = {
                "b_outcome": [0, 1],
                "sigma": 1,
                "beta_ps": [0, 1],
            }
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
                    size=spline_knots + 4,
                )
                B = dmatrix(
                    "bs(ps, knots=knots, degree=3, include_intercept=True, lower_bound=0, upper_bound=1) - 1",
                    {"ps": p, "knots": np.linspace(0, 1, spline_knots)},
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
        \mu &= \text{trend_component} + \text{seasonality_component} + X \cdot \beta \quad \text{(if X is provided)} \\
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
        trend_component: Any | None = None,
        seasonality_component: Any | None = None,
        sample_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(sample_kwargs=sample_kwargs)

        # Warn that this is experimental
        warnings.warn(
            "BayesianBasisExpansionTimeSeries is experimental and its API may change in future versions. "
            "Not recommended for production use.",
            FutureWarning,
            stacklevel=2,
        )

        # Store original configuration parameters
        self.n_order = n_order
        self.n_changepoints_trend = n_changepoints_trend
        self.prior_sigma = prior_sigma
        self._first_fit_timestamp: pd.Timestamp | None = None
        self._exog_var_names: list[str] | None = None

        # Store custom components (fix the bug where they were swapped)
        self._custom_trend_component = trend_component
        self._custom_seasonality_component = seasonality_component

        # Initialize and validate components
        self._trend_component = None
        self._seasonality_component = None
        self._validate_and_initialize_components()

    def _validate_and_initialize_components(self):
        """
        Validate custom components only. Optional dependencies are imported lazily
        when default components are actually needed.
        """
        # Validate custom components have required methods
        if self._custom_trend_component is not None and not hasattr(
            self._custom_trend_component, "apply"
        ):
            raise ValueError(
                "Custom trend_component must have an 'apply' method that accepts time data "
                "and returns a PyMC tensor."
            )

        if self._custom_seasonality_component is not None and not hasattr(
            self._custom_seasonality_component, "apply"
        ):
            raise ValueError(
                "Custom seasonality_component must have an 'apply' method that accepts time data "
                "and returns a PyMC tensor."
            )

    def _get_trend_component(self):
        """Get the trend component, creating default if needed."""
        if self._custom_trend_component is not None:
            return self._custom_trend_component

        # Create default trend component (lazy import of pymc-marketing)
        if self._trend_component is None:
            try:
                from pymc_marketing.mmm import LinearTrend
            except ImportError as err:
                raise ImportError(
                    "BayesianBasisExpansionTimeSeries requires pymc-marketing when default trend "
                    "component is used. Install it with `pip install pymc-marketing`."
                ) from err
            self._trend_component = LinearTrend(
                n_changepoints=self.n_changepoints_trend
            )
        return self._trend_component

    def _get_seasonality_component(self):
        """Get the seasonality component, creating default if needed."""
        if self._custom_seasonality_component is not None:
            return self._custom_seasonality_component

        # Create default seasonality component (lazy import of pymc-marketing)
        if self._seasonality_component is None:
            try:
                from pymc_marketing.mmm import YearlyFourier
            except ImportError as err:
                raise ImportError(
                    "BayesianBasisExpansionTimeSeries requires pymc-marketing when default seasonality "
                    "component is used. Install it with `pip install pymc-marketing`."
                ) from err
            self._seasonality_component = YearlyFourier(n_order=self.n_order)
        return self._seasonality_component

    def _prepare_time_and_exog_features(
        self,
        X: xr.DataArray | None,
    ) -> tuple[np.ndarray, np.ndarray, xr.DataArray | None, int]:
        """
        Prepares time features and processes exogenous variables from X.

        Parameters
        ----------
        X : xr.DataArray or None
            Input features with dims ["obs_ind", "coeffs"]. The obs_ind coordinate
            must contain datetime values. Can be None or have 0 columns if no
            exogenous variables.

        Returns
        -------
        tuple
            (time_for_trend, time_for_seasonality, X_for_pymc, num_obs)
            - time_for_trend: numpy array of time values for trend component
            - time_for_seasonality: numpy array of day-of-year values
            - X_for_pymc: xarray DataArray for exogenous vars, or None if no exog vars
            - num_obs: number of observations
        """
        if X is None:
            raise ValueError(
                "X cannot be None. Pass an empty DataArray if no exog vars."
            )

        if not isinstance(X, xr.DataArray):
            raise TypeError("X must be an xarray DataArray.")

        # Extract datetime index from X coordinates
        if "obs_ind" not in X.coords:
            raise ValueError("X must have 'obs_ind' coordinate.")

        obs_ind_vals = X.coords["obs_ind"].values
        if len(obs_ind_vals) == 0:
            raise ValueError("X must have at least one observation.")

        # Check if obs_ind contains datetime values
        if not isinstance(obs_ind_vals[0], (np.datetime64, pd.Timestamp)):
            raise ValueError(
                "X.coords['obs_ind'] must contain datetime values (np.datetime64 or pd.Timestamp)."
            )

        datetime_index = pd.DatetimeIndex(obs_ind_vals)
        num_obs = len(datetime_index)

        # Extract coefficient names from X coordinates
        exog_names: list[str] = []
        if "coeffs" in X.coords:
            coeffs_vals = X.coords["coeffs"].values
            if len(coeffs_vals) > 0:
                exog_names = list(coeffs_vals)

        # Validate dimensions
        if X.shape[0] != num_obs:
            raise ValueError(
                f"Shape mismatch: X has {X.shape[0]} rows but datetime_index has {num_obs} entries."
            )

        if X.shape[1] != len(exog_names):
            raise ValueError(
                f"Mismatch: X has {X.shape[1]} columns, but {len(exog_names)} coefficient names provided."
            )

        # Set or validate self._exog_var_names
        if X.shape[1] > 0:
            if self._exog_var_names is None:
                self._exog_var_names = exog_names
            elif self._exog_var_names != exog_names:
                raise ValueError(
                    f"Exogenous variable names mismatch. Model fit with {self._exog_var_names}, "
                    f"but current call provides {exog_names}."
                )
        elif self._exog_var_names is None:
            # No exog vars in this call, and none set before
            self._exog_var_names = []

        # Set first fit timestamp if not set
        if self._first_fit_timestamp is None:
            self._first_fit_timestamp = datetime_index[0]

        # Compute time features (these are numpy arrays)
        time_for_trend = (
            (datetime_index - self._first_fit_timestamp).days / 365.25
        ).values
        time_for_seasonality = datetime_index.dayofyear.values

        # Determine X to use for PyMC (return as xarray or None)
        X_for_pymc: xr.DataArray | None = None
        if self._exog_var_names and X.shape[1] > 0:
            X_for_pymc = X  # Keep as xarray
        # else: no exog vars, return None

        return time_for_trend, time_for_seasonality, X_for_pymc, num_obs

    def build_model(
        self, X: xr.DataArray, y: xr.DataArray, coords: dict[str, Any] | None
    ) -> None:
        """
        Defines the PyMC model.

        Parameters
        ----------
        X : xr.DataArray
            Input features with dims ["obs_ind", "coeffs"]. Can have 0 columns if
            no exogenous variables. The obs_ind coordinate must contain datetime values.
        y : xr.DataArray
            Target variable with dims ["obs_ind", "treated_units"].
        coords : dict, optional
            Coordinates dictionary. Can contain "datetime_index" for backwards compatibility,
            but datetime is preferentially extracted from X.coords['obs_ind'].
        """
        # Prepare time features and validate X
        # This extracts datetime from X.coords['obs_ind'] and validates exog vars
        (
            time_for_trend,
            time_for_seasonality,
            X_for_pymc,  # xarray DataArray or None
            num_obs,
        ) = self._prepare_time_and_exog_features(X)

        # Build model coordinates
        model_coords = {
            "obs_ind": np.arange(num_obs),
            "treated_units": ["unit_0"],
        }

        # Add coeffs coordinate if we have exogenous variables
        if self._exog_var_names:
            model_coords["coeffs"] = self._exog_var_names  # type: ignore[assignment]

        with self:
            self.add_coords(model_coords)

            # Time data for trend and seasonality
            t_trend_data = pm.Data(
                "t_trend_data",
                time_for_trend,
                dims="obs_ind",
            )
            t_season_data = pm.Data(
                "t_season_data",
                time_for_seasonality,
                dims="obs_ind",
            )

            # Get validated components
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
            if X_for_pymc is not None:
                # Use xarray directly with pm.Data
                X_data = pm.Data("X", X_for_pymc, dims=["obs_ind", "coeffs"])
                beta = pm.Normal("beta", mu=0, sigma=10, dims="coeffs")
                mu_ = mu_ + pm.math.dot(X_data, beta)

            # Make mu_ an explicit deterministic variable with treated_units dimension
            # Expand dims to include treated_units for consistency with other models
            mu = pm.Deterministic("mu", mu_[:, None], dims=["obs_ind", "treated_units"])

            # Likelihood - also with treated_units dimension
            # Use xarray directly with pm.Data
            sigma = pm.HalfNormal("sigma", sigma=self.prior_sigma, dims="treated_units")
            y_data = pm.Data("y", y, dims=["obs_ind", "treated_units"])
            pm.Normal(
                "y_hat",
                mu=mu,
                sigma=sigma,
                observed=y_data,
                dims=["obs_ind", "treated_units"],
            )

    def fit(
        self, X: xr.DataArray, y: xr.DataArray, coords: dict[str, Any] | None = None
    ) -> az.InferenceData:
        """Draw samples from posterior, prior predictive, and posterior predictive
        distributions, placing them in the model's idata attribute.

        Parameters
        ----------
        X : xr.DataArray
            Input features with dims ["obs_ind", "coeffs"]. Can have 0 columns if
            no exogenous variables.
        y : xr.DataArray
            Target variable with dims ["obs_ind", "treated_units"].
        coords : dict
            Coordinates dictionary. Must contain "datetime_index" (pd.DatetimeIndex).
        """
        random_seed = self.sample_kwargs.get("random_seed", None)
        self.build_model(X, y, coords=coords)
        with self:
            self.idata = pm.sample(**self.sample_kwargs)
            if self.idata is not None:
                self.idata.extend(pm.sample_prior_predictive(random_seed=random_seed))
                self.idata.extend(
                    pm.sample_posterior_predictive(
                        self.idata,
                        var_names=["y_hat", "mu"],
                        progressbar=self.sample_kwargs.get("progressbar", True),
                        random_seed=random_seed,
                    )
                )
        return self.idata  # type: ignore[return-value]

    def _data_setter(self, X: xr.DataArray) -> None:
        """
        Set data for the model for prediction.

        Parameters
        ----------
        X : xr.DataArray
            Input features with dims ["obs_ind", "coeffs"]. Must have datetime
            coordinates on obs_ind.
        """
        # Prepare time features and get X for PyMC (as xarray or None)
        (
            time_for_trend_pred_vals,
            time_for_seasonality_pred_vals,
            X_for_pymc,  # xarray or None
            num_obs_pred,
        ) = self._prepare_time_and_exog_features(X)

        new_obs_inds = np.arange(num_obs_pred)

        # Create dummy y data with proper shape
        dummy_y = xr.DataArray(
            np.zeros((num_obs_pred, 1)),
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": new_obs_inds, "treated_units": ["unit_0"]},
        )

        data_to_set = {
            "y": dummy_y,
            "t_trend_data": time_for_trend_pred_vals,
            "t_season_data": time_for_seasonality_pred_vals,
        }
        coords_to_set = {"obs_ind": new_obs_inds}

        # Handle exogenous variables
        if "X" in self.named_vars:
            if X_for_pymc is None and self._exog_var_names:
                raise ValueError(
                    "Model was built with exogenous variables. "
                    "New X data must provide these."
                )
            if X_for_pymc is not None:
                # Use xarray directly
                data_to_set["X"] = X_for_pymc
            else:
                # Model expects X but we have none - create empty xarray
                empty_X = xr.DataArray(
                    np.empty((num_obs_pred, 0)),
                    dims=["obs_ind", "coeffs"],
                    coords={"obs_ind": new_obs_inds, "coeffs": []},
                )
                data_to_set["X"] = empty_X
        elif X_for_pymc is not None:
            warnings.warn(
                "X provided exogenous variables, but the model was not "
                "built with exogenous variables. These will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        with self:
            pm.set_data(data_to_set, coords=coords_to_set)

    def predict(
        self,
        X: xr.DataArray,
        coords: dict[str, Any] | None = None,
        out_of_sample: bool | None = False,
        **kwargs: Any,
    ) -> az.InferenceData:
        """
        Predict data given input X.

        Parameters
        ----------
        X : xr.DataArray
            Input features with dims ["obs_ind", "coeffs"]. Must have datetime
            coordinates on obs_ind.
        coords : dict, optional
            Not used, kept for API compatibility.
        out_of_sample : bool, optional
            Not used, kept for API compatibility.

        Returns
        -------
        az.InferenceData
            Posterior predictive samples.
        """
        random_seed = self.sample_kwargs.get("random_seed", None)
        self._data_setter(X)
        with self:
            post_pred = pm.sample_posterior_predictive(
                self.idata,
                var_names=["y_hat", "mu"],
                progressbar=self.sample_kwargs.get("progressbar", False),
                random_seed=random_seed,
            )

        # Assign coordinates from input X for proper alignment
        if isinstance(X, xr.DataArray) and "obs_ind" in X.coords:
            post_pred["posterior_predictive"] = post_pred[
                "posterior_predictive"
            ].assign_coords(obs_ind=X.obs_ind)

        return post_pred

    def score(
        self,
        X: xr.DataArray,
        y: xr.DataArray,
        coords: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        """Score the Bayesian R^2.

        Parameters
        ----------
        X : xr.DataArray
            Input features with dims ["obs_ind", "coeffs"].
        y : xr.DataArray
            Target variable with dims ["obs_ind", "treated_units"].
        coords : dict, optional
            Not used, kept for API compatibility.

        Returns
        -------
        pd.Series
            R² score and standard deviation for each treated unit.
        """
        # Use base class score method now that we have treated_units dimension
        return super().score(X, y, coords=coords, **kwargs)


class StateSpaceTimeSeries(PyMCModel):
    """
    State-space time series model using :class:`pymc-extras.statespace.structural`.

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
        Pytensor compile mode passed to `build_statespace_graph`. Defaults to None.
    """

    def __init__(
        self,
        level_order: int = 2,
        seasonal_length: int = 12,
        trend_component: Any | None = None,
        seasonality_component: Any | None = None,
        sample_kwargs: dict[str, Any] | None = None,
        mode: str | None = None,
    ):
        super().__init__(sample_kwargs=sample_kwargs)

        # Warn that this is experimental
        warnings.warn(
            "StateSpaceTimeSeries is experimental and its API may change in future versions. "
            "Not recommended for production use.",
            FutureWarning,
            stacklevel=2,
        )

        self._custom_trend_component = trend_component
        self._custom_seasonality_component = seasonality_component
        self.level_order = level_order
        self.seasonal_length = seasonal_length
        self.mode = mode
        self.ss_mod: Any = None
        self.second_model: pm.Model | None = None  # Created in build_model()
        self._validate_and_initialize_components()

    def _validate_and_initialize_components(self):
        """
        Validate custom components only. Optional dependencies are imported lazily
        when default components are actually needed.
        """
        # Validate custom components have required methods
        if self._custom_trend_component is not None and not hasattr(
            self._custom_trend_component, "apply"
        ):
            raise ValueError(
                "Custom trend_component must have an 'apply' method that accepts time data "
                "and returns a PyMC tensor."
            )

        if self._custom_seasonality_component is not None and not hasattr(
            self._custom_seasonality_component, "apply"
        ):
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

        # Create default trend component (lazy import of pymc-extras)
        if self._trend_component is None:
            try:
                from pymc_extras.statespace import structural as st
            except ImportError as err:
                raise ImportError(
                    "StateSpaceTimeSeries requires pymc-extras when default trend component is used. "
                    "Install it with `conda/mamba/micromamba install -c conda-forge pymc-extras`."
                ) from err
            self._trend_component = st.LevelTrendComponent(order=self.level_order)
        return self._trend_component

    def _get_seasonality_component(self):
        """Get the seasonality component, creating default if needed."""
        if self._custom_seasonality_component is not None:
            return self._custom_seasonality_component

        # Create default seasonality component (lazy import of pymc-extras)
        if self._seasonality_component is None:
            try:
                from pymc_extras.statespace import structural as st
            except ImportError as err:
                raise ImportError(
                    "StateSpaceTimeSeries requires pymc-extras when default seasonality component is used. "
                    "Install it with `conda/mamba/micromamba install -c conda-forge pymc-extras`."
                ) from err
            self._seasonality_component = st.FrequencySeasonality(
                season_length=self.seasonal_length, name="freq"
            )
        return self._seasonality_component

    def build_model(
        self,
        X: xr.DataArray | None = None,
        y: xr.DataArray | None = None,
        coords: dict[str, Any] | None = None,
    ) -> None:
        """
        Build the PyMC state-space model.

        Parameters
        ----------
        X : xr.DataArray, optional
            Input features with dims ["obs_ind", "coeffs"]. Not used by state-space
            models, but kept for API compatibility.
        y : xr.DataArray
            Target variable with dims ["obs_ind", "treated_units"]. Must have datetime
            coordinates on obs_ind.
        coords : dict, optional
            Coordinates dictionary. Can contain "datetime_index" for backwards compatibility,
            but datetime is preferentially extracted from y.coords['obs_ind'].
        """
        if y is None:
            raise ValueError(
                "y must be provided for StateSpaceTimeSeries.build_model()"
            )

        # Extract datetime index from y coordinates
        if "obs_ind" not in y.coords:
            raise ValueError("y must have 'obs_ind' coordinate.")

        obs_ind_vals = y.coords["obs_ind"].values
        if len(obs_ind_vals) == 0:
            raise ValueError("y must have at least one observation.")

        # Check if obs_ind contains datetime values
        if isinstance(obs_ind_vals[0], (np.datetime64, pd.Timestamp)):
            datetime_index = pd.DatetimeIndex(obs_ind_vals)
        elif coords is not None and "datetime_index" in coords:
            # Fallback to coords dict for backwards compatibility
            datetime_index = coords["datetime_index"]
            if not isinstance(datetime_index, pd.DatetimeIndex):
                raise ValueError(
                    "coords['datetime_index'] must be a pd.DatetimeIndex if provided."
                )
        else:
            raise ValueError(
                "y.coords['obs_ind'] must contain datetime values or "
                "coords must contain 'datetime_index' (pd.DatetimeIndex)."
            )

        self._train_index = datetime_index

        # Instantiate components and build state-space object
        trend = self._get_trend_component()
        season = self._get_seasonality_component()
        combined = trend + season
        self.ss_mod = combined.build()

        # Extract parameter dims (order: initial_trend, sigma_trend, seasonal, P0)
        if self.ss_mod is None:
            raise RuntimeError("State space model not initialized")
        initial_trend_dims, sigma_trend_dims, annual_dims, P0_dims = (
            self.ss_mod.param_dims.values()
        )

        # Build coordinates for the model
        coordinates = self.ss_mod.coords.copy()
        if coords:
            # Merge with user-provided coords (excluding datetime_index and obs_ind which are handled separately)
            coords_copy = coords.copy()
            coords_copy.pop("datetime_index", None)
            coords_copy.pop(
                "obs_ind", None
            )  # obs_ind handled by state-space model's time dimension
            coordinates.update(coords_copy)

        # Build model
        with pm.Model(coords=coordinates) as self.second_model:
            # Add coords for statespace (includes 'time' and 'state' dims)
            P0_diag = pm.Gamma("P0_diag", alpha=2, beta=1, dims=P0_dims[0])
            _P0 = pm.Deterministic("P0", pt.diag(P0_diag), dims=P0_dims)
            _initial_trend = pm.Normal(
                "initial_level_trend", sigma=50, dims=initial_trend_dims
            )
            _annual_seasonal = pm.ZeroSumNormal(
                "params_freq", sigma=80, dims=annual_dims
            )

            _sigma_trend = pm.Gamma(
                "sigma_level_trend", alpha=2, beta=5, dims=sigma_trend_dims
            )
            _sigma_monthly_season = pm.Gamma("sigma_freq", alpha=2, beta=1)

            # Attach the state-space graph using the observed data
            # Extract values from xarray for pandas DataFrame
            y_values = (
                y.isel(treated_units=0).values
                if "treated_units" in y.dims
                else y.values
            )
            df = pd.DataFrame({"y": y_values.flatten()}, index=datetime_index)
            if self.ss_mod is not None:
                self.ss_mod.build_statespace_graph(df[["y"]], mode=self.mode)

    def fit(
        self,
        X: xr.DataArray | None = None,
        y: xr.DataArray | None = None,
        coords: dict[str, Any] | None = None,
    ) -> az.InferenceData:
        """
        Fit the model, drawing posterior samples.

        Parameters
        ----------
        X : xr.DataArray, optional
            Input features with dims ["obs_ind", "coeffs"]. Not used by state-space
            models, but kept for API compatibility.
        y : xr.DataArray
            Target variable with dims ["obs_ind", "treated_units"]. Must have datetime
            coordinates on obs_ind.
        coords : dict, optional
            Coordinates dictionary. Can contain "datetime_index" for backwards compatibility.

        Returns
        -------
        az.InferenceData
            InferenceData with parameter draws.
        """
        if y is None:
            raise ValueError("y must be provided for StateSpaceTimeSeries.fit()")
        self.build_model(X, y, coords)
        if self.second_model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        with self.second_model:
            self.idata = pm.sample(**self.sample_kwargs)
            if self.idata is not None:
                self.idata.extend(
                    pm.sample_posterior_predictive(
                        self.idata,
                    )
                )
        self.conditional_idata = self._smooth()
        return self._prepare_idata()

    def _prepare_idata(self) -> az.InferenceData:
        """Prepare InferenceData with proper dimensions including treated_units."""
        if self.idata is None:
            raise RuntimeError("Model must be fit before smoothing.")

        new_idata = self.idata.copy()
        # Get smoothed posterior and sum over state dimension
        smoothed = self.conditional_idata.isel(observed_state=0).rename(
            {"smoothed_posterior_observed": "y_hat"}
        )
        y_hat_summed = smoothed.y_hat.copy()

        # Rename 'time' to 'obs_ind' to match CausalPy conventions
        if "time" in y_hat_summed.dims:
            y_hat_final = y_hat_summed.rename({"time": "obs_ind"})
        else:
            y_hat_final = y_hat_summed

        # Add treated_units dimension for consistency with other models
        y_hat_with_units = y_hat_final.expand_dims({"treated_units": ["unit_0"]})

        new_idata["posterior_predictive"]["y_hat"] = y_hat_with_units
        new_idata["posterior_predictive"]["mu"] = y_hat_with_units

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
        if self.ss_mod is None:
            raise RuntimeError("State space model not initialized")
        return self.ss_mod.forecast(self.idata, start=start, periods=periods)

    def predict(
        self,
        X: xr.DataArray | None = None,
        coords: dict[str, Any] | None = None,
        out_of_sample: bool | None = False,
        **kwargs: Any,
    ) -> az.InferenceData:
        """
        Predict data given input X.

        Parameters
        ----------
        X : xr.DataArray, optional
            Input features with dims ["obs_ind", "coeffs"]. Must have datetime
            coordinates on obs_ind for out-of-sample predictions. Not required for
            in-sample predictions.
        coords : dict, optional
            Not used directly, datetime extracted from X coordinates.
        out_of_sample : bool, optional
            If True, forecast future values. If False, return in-sample predictions.

        Returns
        -------
        az.InferenceData
            Posterior predictive samples with y_hat and mu.
        """
        if not out_of_sample:
            return self._prepare_idata()
        else:
            # Extract datetime from X coordinates
            if X is None:
                raise ValueError(
                    "X must be provided for out-of-sample predictions with datetime coordinates"
                )
            if not hasattr(X, "coords") or "obs_ind" not in X.coords:
                raise ValueError(
                    "X must have 'obs_ind' coordinate with datetime values for prediction"
                )

            obs_ind_vals = X.coords["obs_ind"].values
            if len(obs_ind_vals) == 0 or not isinstance(
                obs_ind_vals[0], (np.datetime64, pd.Timestamp)
            ):
                raise ValueError("X 'obs_ind' coordinate must contain datetime values")

            idx = pd.DatetimeIndex(obs_ind_vals)
            last = self._train_index[-1]  # start forecasting after the last observed
            forecast_data = self._forecast(start=last, periods=len(idx))
            forecast_copy = forecast_data.copy()

            # Rename 'time' to 'obs_ind' to match CausalPy conventions
            if "time" in forecast_copy.dims:
                forecast_copy = forecast_copy.rename({"time": "obs_ind"})

            # Extract the forecasted observed data and add treated_units dimension
            y_hat = forecast_copy["forecast_observed"].isel(observed_state=0)
            y_hat_with_units = y_hat.expand_dims({"treated_units": ["unit_0"]})

            # Wrap in InferenceData for consistency
            result = az.InferenceData(
                posterior_predictive=xr.Dataset(
                    {"y_hat": y_hat_with_units, "mu": y_hat_with_units}
                )
            )

            # Assign coordinates from input X for proper alignment
            if isinstance(X, xr.DataArray) and "obs_ind" in X.coords:
                result["posterior_predictive"] = result[
                    "posterior_predictive"
                ].assign_coords(obs_ind=X.obs_ind)

            return result

    def score(
        self,
        X: xr.DataArray | None = None,
        y: xr.DataArray | None = None,
        coords: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        """
        Score the Bayesian R^2 given inputs X and outputs y.

        Parameters
        ----------
        X : xr.DataArray, optional
            Input features. Not used by state-space models, but kept for API compatibility.
        y : xr.DataArray
            Target variable with dims ["obs_ind", "treated_units"].
        coords : dict, optional
            Not used, kept for API compatibility.

        Returns
        -------
        pd.Series
            R² score and standard deviation for each treated unit.
        """
        # Use base class implementation - X is accepted but not used by predict()
        return super().score(X, y, coords, **kwargs)


class TransferFunctionLinearRegression(PyMCModel):
    r"""
    Bayesian Transfer Function model for Graded Intervention Time Series.

    This model jointly estimates transform parameters (adstock, saturation) and
    regression coefficients within a Bayesian framework using PyMC.

    The model applies transforms to treatment variables using pymc-marketing
    functions, allowing full Bayesian inference on all parameters including
    the transform parameters themselves.

    Parameters
    ----------
    saturation_type : str or None
        Type of saturation transform. Options: "hill", "logistic", "michaelis_menten", None.
        If None, no saturation is applied.
    adstock_config : dict or None
        Configuration for adstock transform. Required keys:
        - "half_life_prior": dict with prior specification (e.g., {"dist": "Gamma", "alpha": 4, "beta": 2})
        - "l_max": int, maximum lag
        - "normalize": bool, whether to normalize weights
        If None, no adstock is applied.
    saturation_config : dict or None
        Configuration for saturation transform. Structure depends on saturation_type:
        - For "hill": {"slope_prior": {...}, "kappa_prior": {...}}
        - For "logistic": {"lam_prior": {...}}
        - For "michaelis_menten": {"alpha_prior": {...}, "lam_prior": {...}}
    coef_constraint : str, default="unconstrained"
        Constraint on treatment coefficients: "nonnegative" or "unconstrained".
    sample_kwargs : dict, optional
        Additional kwargs passed to pm.sample().

    Notes
    -----
    The current implementation uses independent Normal errors.

    **Autocorrelation in Errors: Implementation Challenges**

    For time series with autocorrelated residuals, see ``TransferFunctionARRegression``,
    which implements AR(1) errors via quasi-differencing. This note explains why AR
    errors require special handling and the design decisions made.

    *Why Standard PyMC AR Approaches Fail:*

    The fundamental issue is that **observed data cannot depend on model parameters**
    in PyMC's computational graph. For AR errors, we want:

    .. math::
        y[t] = \mu[t] + \epsilon[t]
        \epsilon[t] = \rho \cdot \epsilon[t-1] + \nu[t]

    But ``\epsilon[t] = y[t] - \mu[t]`` depends on ``\mu`` (which depends on ``\beta``,
    ``\theta``, ``half_life``, etc.), so this fails::

        # This FAILS with TypeError
        residuals = y - mu  # depends on parameters!
        pm.AR("epsilon", rho, observed=residuals)  # ❌

    *Implemented Solution (TransferFunctionARRegression):*

    Uses **quasi-differencing** following Box & Tiao (1975):

    .. math::
        y[t] - \rho \cdot y[t-1] = \mu[t] - \rho \cdot \mu[t-1] + \nu[t]

    This transforms to independent innovations ``\nu[t]``, allowing manual log-likelihood
    via ``pm.Potential``. This is the theoretically correct Box & Tiao intervention
    analysis model where AR(1) represents the noise structure itself.

    *Alternative: AR as Latent Component (Not Implemented):*

    An alternative that avoids the ``observed=`` issue would be:

    .. math::
        y[t] = \mu_{baseline}[t] + ar[t] + \epsilon_{obs}[t]
        ar[t] = \rho \cdot ar[t-1] + \eta[t]
        \epsilon_{obs}[t] \sim N(0, \sigma^2_{obs})

    This could use PyMC's built-in ``pm.AR`` since ``ar[t]`` is latent (unobserved)::

        # This WOULD work (but changes model interpretation)
        ar = pm.AR("ar", rho=[rho_param], sigma=sigma_ar, shape=n_obs)
        mu_total = mu_baseline + ar
        pm.Normal("y", mu=mu_total, sigma=sigma_obs, observed=y_data)  # ✅

    **Why We Don't Use This Approach:**

    1. **Different model class**: Represents AR + white noise, not pure AR errors.
       Has two variance parameters (``\sigma_{ar}``, ``\sigma_{obs}``) vs. one (``\sigma``).

    2. **Theoretical mismatch**: Box & Tiao's intervention analysis models autocorrelation
       in the noise process itself, not as an additional latent component. The AR process
       IS the residual structure, not a separate mean component.

    3. **Identifiability concerns**: With both AR and white noise, parameters may be
       poorly identified unless ``\sigma_{obs} \approx 0`` (which defeats the purpose).

    4. **Interpretation**: The latent AR component would represent a time-varying offset
       rather than residual autocorrelation, changing the causal interpretation.

    **When to Use Each Model:**

    - **TransferFunctionLinearRegression** (this class): When residuals show minimal
      autocorrelation, or computational efficiency is critical.

    - **TransferFunctionARRegression**: When residual diagnostics show significant
      autocorrelation (e.g., ACF plots, Durbin-Watson test), and you want to follow
      the classical Box & Tiao specification.

    **Prior Customization**:

    Priors are managed using the ``Prior`` class from ``pymc_extras`` and can be
    customized via the ``priors`` parameter::

        from pymc_extras.prior import Prior

        model = cp.pymc_models.TransferFunctionLinearRegression(
            saturation_type=None,
            adstock_config={...},
            priors={
                "beta": Prior(
                    "Normal", mu=0, sigma=100, dims=["treated_units", "coeffs"]
                ),
                "sigma": Prior("HalfNormal", sigma=50, dims=["treated_units"]),
            },
        )

    By default, data-informed priors are set automatically via ``priors_from_data()``:

    - Baseline coefficients (``beta``): ``Normal(0, 5 * std(y))``
    - Treatment coefficients (``theta_treatment``): ``Normal(0, 2 * std(y))`` or ``HalfNormal(2 * std(y))``
    - Error std (``sigma``): ``HalfNormal(2 * std(y))``

    This adaptive approach ensures priors are reasonable regardless of data scale.

    Examples
    --------
    Basic usage::

        import causalpy as cp

        model = cp.pymc_models.TransferFunctionLinearRegression(
            saturation_type=None,
            adstock_config={
                "half_life_prior": {"dist": "Gamma", "alpha": 4, "beta": 2},
                "l_max": 8,
                "normalize": True,
            },
            sample_kwargs={"chains": 4, "draws": 2000, "tune": 1000},
        )
    """

    def __init__(
        self,
        saturation_type: str | None = None,
        adstock_config: dict | None = None,
        saturation_config: dict | None = None,
        coef_constraint: str = "unconstrained",
        sample_kwargs: dict[str, Any] | None = None,
        priors: dict[str, Any] | None = None,
    ):
        """Initialize TransferFunctionLinearRegression model."""
        super().__init__(sample_kwargs=sample_kwargs, priors=priors)

        # Validate that at least one transform is specified
        if saturation_type is None and adstock_config is None:
            raise ValueError(
                "At least one of saturation_type or adstock_config must be specified."
            )

        self.saturation_type = saturation_type
        self.adstock_config = adstock_config
        self.saturation_config = saturation_config or {}
        self.coef_constraint = coef_constraint

        # Store for later use
        self.treatment_names = None
        self.n_treatments = None

    def priors_from_data(self, X, y) -> dict[str, Any]:
        """
        Generate data-informed priors that scale with outcome variable.

        Computes priors for baseline coefficients, treatment coefficients,
        and error standard deviation based on the standard deviation of y.
        This ensures priors are reasonable regardless of data scale.

        Parameters
        ----------
        X : xr.DataArray
            Baseline design matrix (n_obs, n_baseline_features).
        y : xr.DataArray
            Outcome variable (n_obs, 1).

        Returns
        -------
        Dict[str, Prior]
            Dictionary with Prior objects for beta, theta_treatment, and sigma.

        Notes
        -----
        The returned dictionary contains Prior objects with the following structure::

            {
                "beta": Prior(
                    "Normal", mu=0, sigma=5 * y_scale, dims=["treated_units", "coeffs"]
                ),
                "theta_treatment": Prior(
                    "Normal",
                    mu=0,
                    sigma=2 * y_scale,
                    dims=["treated_units", "treatment_names"],
                ),
                "sigma": Prior("HalfNormal", sigma=2 * y_scale, dims=["treated_units"]),
            }

        where ``y_scale = std(y)``.
        """
        y_scale = float(np.std(y))

        priors = {
            "beta": Prior(
                "Normal",
                mu=0,
                sigma=5 * y_scale,
                dims=["treated_units", "coeffs"],
            ),
            "sigma": Prior(
                "HalfNormal",
                sigma=2 * y_scale,
                dims=["treated_units"],
            ),
        }

        # Treatment coefficient prior depends on constraint
        if self.coef_constraint == "nonnegative":
            priors["theta_treatment"] = Prior(
                "HalfNormal",
                sigma=2 * y_scale,
                dims=["treated_units", "treatment_names"],
            )
        else:
            priors["theta_treatment"] = Prior(
                "Normal",
                mu=0,
                sigma=2 * y_scale,
                dims=["treated_units", "treatment_names"],
            )

        return priors

    def build_model(self, X, y, coords, treatment_data):
        """
        Build the PyMC model with transforms.

        Parameters
        ----------
        X : xr.DataArray
            Baseline design matrix (n_obs, n_baseline_features).
        y : xr.DataArray
            Outcome variable (n_obs, 1).
        coords : dict
            Coordinate names for PyMC.
        treatment_data : xr.DataArray
            Raw treatment variables (n_obs, n_treatments).
        """
        from pymc_marketing.mmm.transformers import (
            geometric_adstock,
            hill_function,
            logistic_saturation,
            michaelis_menten,
        )

        self.treatment_names = treatment_data.coords.get(
            "treatment_names",
            [f"treatment_{i}" for i in range(treatment_data.shape[1])],
        ).values.tolist()
        self.n_treatments = treatment_data.shape[1]

        with self:
            self.add_coords(coords)

            # Register data
            X_data = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            y_data = pm.Data("y", y, dims=["obs_ind", "treated_units"])
            treatment_raw = pm.Data(
                "treatment_raw", treatment_data, dims=["obs_ind", "treatment_names"]
            )

            # ==================================================================
            # Transform Parameters (if applicable)
            # ==================================================================
            transform_params = {}

            # Adstock transform parameters
            if self.adstock_config is not None:
                half_life_prior_config = self.adstock_config.get(
                    "half_life_prior", {"dist": "Gamma", "alpha": 4, "beta": 2}
                )
                if half_life_prior_config["dist"] == "Gamma":
                    half_life = pm.Gamma(
                        "half_life",
                        alpha=half_life_prior_config["alpha"],
                        beta=half_life_prior_config["beta"],
                    )
                elif half_life_prior_config["dist"] == "HalfNormal":
                    half_life = pm.HalfNormal(
                        "half_life",
                        sigma=half_life_prior_config["sigma"],
                    )
                else:
                    raise ValueError(
                        f"Unsupported prior distribution: {half_life_prior_config['dist']}"
                    )

                transform_params["half_life"] = half_life

                # Convert half_life to alpha for geometric_adstock
                alpha_adstock = pm.Deterministic(
                    "alpha_adstock", pt.power(0.5, 1.0 / half_life)
                )

            # Saturation transform parameters
            if self.saturation_type == "hill":
                slope_prior = self.saturation_config.get(
                    "slope_prior", {"dist": "HalfNormal", "sigma": 2}
                )
                kappa_prior = self.saturation_config.get(
                    "kappa_prior", {"dist": "HalfNormal", "sigma": 10}
                )

                if slope_prior["dist"] == "HalfNormal":
                    slope = pm.HalfNormal("slope", sigma=slope_prior["sigma"])
                elif slope_prior["dist"] == "Gamma":
                    slope = pm.Gamma(
                        "slope", alpha=slope_prior["alpha"], beta=slope_prior["beta"]
                    )

                if kappa_prior["dist"] == "HalfNormal":
                    kappa = pm.HalfNormal("kappa", sigma=kappa_prior["sigma"])
                elif kappa_prior["dist"] == "Gamma":
                    kappa = pm.Gamma(
                        "kappa", alpha=kappa_prior["alpha"], beta=kappa_prior["beta"]
                    )

                transform_params["slope"] = slope
                transform_params["kappa"] = kappa

            elif self.saturation_type == "logistic":
                lam_prior = self.saturation_config.get(
                    "lam_prior", {"dist": "HalfNormal", "sigma": 1}
                )
                if lam_prior["dist"] == "HalfNormal":
                    lam = pm.HalfNormal("lam", sigma=lam_prior["sigma"])
                elif lam_prior["dist"] == "Gamma":
                    lam = pm.Gamma(
                        "lam", alpha=lam_prior["alpha"], beta=lam_prior["beta"]
                    )
                transform_params["lam"] = lam

            elif self.saturation_type == "michaelis_menten":
                alpha_prior = self.saturation_config.get(
                    "alpha_prior", {"dist": "HalfNormal", "sigma": 1}
                )
                lam_prior = self.saturation_config.get(
                    "lam_prior", {"dist": "HalfNormal", "sigma": 100}
                )

                if alpha_prior["dist"] == "HalfNormal":
                    alpha_sat = pm.HalfNormal("alpha_sat", sigma=alpha_prior["sigma"])
                elif alpha_prior["dist"] == "Gamma":
                    alpha_sat = pm.Gamma(
                        "alpha_sat",
                        alpha=alpha_prior["alpha"],
                        beta=alpha_prior["beta"],
                    )

                if lam_prior["dist"] == "HalfNormal":
                    lam = pm.HalfNormal("lam", sigma=lam_prior["sigma"])
                elif lam_prior["dist"] == "Gamma":
                    lam = pm.Gamma(
                        "lam", alpha=lam_prior["alpha"], beta=lam_prior["beta"]
                    )

                transform_params["alpha_sat"] = alpha_sat
                transform_params["lam"] = lam

            # ==================================================================
            # Apply Transforms to Treatment Variables
            # ==================================================================
            treatment_transformed_list = []

            for i in range(self.n_treatments):
                treatment_i = treatment_raw[:, i]

                # Apply saturation
                if self.saturation_type == "hill":
                    treatment_i = hill_function(treatment_i, slope=slope, kappa=kappa)
                elif self.saturation_type == "logistic":
                    treatment_i = logistic_saturation(treatment_i, lam=lam)
                elif self.saturation_type == "michaelis_menten":
                    treatment_i = michaelis_menten(
                        treatment_i, alpha=alpha_sat, lam=lam
                    )

                # Apply adstock
                if self.adstock_config is not None:
                    l_max = self.adstock_config.get("l_max", 12)
                    normalize = self.adstock_config.get("normalize", True)
                    treatment_i = geometric_adstock(
                        treatment_i,
                        alpha=alpha_adstock,
                        l_max=l_max,
                        normalize=normalize,
                        mode="After",  # Causal: only past affects present
                    )

                treatment_transformed_list.append(treatment_i)

            # Stack transformed treatments
            treatment_transformed = pt.stack(treatment_transformed_list, axis=1)

            # ==================================================================
            # Regression Coefficients (with data-informed priors)
            # ==================================================================
            # Baseline coefficients: data-informed priors set via priors_from_data()
            beta = self.priors["beta"].create_variable("beta")

            # Treatment coefficients: data-informed priors set via priors_from_data()
            theta_treatment = self.priors["theta_treatment"].create_variable(
                "theta_treatment"
            )

            # ==================================================================
            # Mean Function
            # ==================================================================
            # Baseline contribution
            mu_baseline = pt.dot(X_data, beta.T)  # (n_obs, n_units)

            # Treatment contribution
            mu_treatment = pt.dot(
                treatment_transformed, theta_treatment.T
            )  # (n_obs, n_units)

            # Combined mean
            mu = pm.Deterministic(
                "mu", mu_baseline + mu_treatment, dims=["obs_ind", "treated_units"]
            )

            # ==================================================================
            # Likelihood
            # ==================================================================
            # Error std: data-informed prior set via priors_from_data()
            sigma = self.priors["sigma"].create_variable("sigma")

            # For now, use independent Normal errors
            # Note: AR(1) errors in regression context require more complex implementation
            # See: https://discourse.pymc.io/t/regression-with-ar-1-errors/
            # Future enhancement: Implement AR(1) errors via state-space or custom likelihood
            pm.Normal(
                "y_hat",
                mu=mu,
                sigma=sigma,
                observed=y_data,
                dims=["obs_ind", "treated_units"],
            )

    def fit(self, X, y, coords, treatment_data):
        """
        Fit the Transfer Function model.

        This method overrides the base PyMCModel.fit() to accept treatment_data.

        Parameters
        ----------
        X : xr.DataArray
            Baseline design matrix (n_obs, n_baseline_features).
        y : xr.DataArray
            Outcome variable (n_obs, 1).
        coords : dict
            Coordinate names for PyMC model.
        treatment_data : xr.DataArray
            Raw treatment variables (n_obs, n_treatments).

        Returns
        -------
        idata : arviz.InferenceData
            Inference data with posterior, prior predictive, and posterior predictive.
        """
        # Ensure random_seed is used in sample_prior_predictive() and
        # sample_posterior_predictive() if provided in sample_kwargs.
        random_seed = self.sample_kwargs.get("random_seed", None)

        # Merge priors with precedence: user-specified > data-driven > defaults
        # Data-driven priors are computed first, then user-specified priors override them
        self.priors = {**self.priors_from_data(X, y), **self.priors}

        # Build the model with treatment data
        self.build_model(X, y, coords, treatment_data)

        with self:
            self.idata = pm.sample(**self.sample_kwargs)
            self.idata.extend(pm.sample_prior_predictive(random_seed=random_seed))
            self.idata.extend(
                pm.sample_posterior_predictive(
                    self.idata, progressbar=False, random_seed=random_seed
                )
            )
        return self.idata


class TransferFunctionARRegression(PyMCModel):
    r"""
    Bayesian Transfer Function model with AR(1) errors for Graded Intervention Time Series.

    This model extends the Transfer Function framework by explicitly modeling autocorrelation
    in the errors using an AR(1) process implemented via quasi-differencing. This approach
    properly accounts for temporal correlation in the residuals while jointly estimating
    transform parameters (adstock, saturation) and regression coefficients.

    Mathematical Framework
    ----------------------
    The standard regression model with AR(1) errors is:

    .. math::
        y[t] = \mu[t] + \epsilon[t]
        \epsilon[t] = \rho \cdot \epsilon[t-1] + \nu[t]
        \nu[t] \sim N(0, \sigma_\nu^2)

    where :math:`\mu[t]` is the regression mean (baseline + transformed treatment effects),
    :math:`\rho` is the AR(1) coefficient (|ρ| < 1), and :math:`\nu[t]` is white noise.

    Quasi-Differencing Transformation
    ----------------------------------
    To enable Bayesian inference, we apply quasi-differencing:

    .. math::
        y[t] - \rho \cdot y[t-1] = \mu[t] - \rho \cdot \mu[t-1] + \nu[t]

    This transforms the model into one with independent errors :math:`\nu[t]`, which can
    be directly sampled in PyMC. The quasi-differenced likelihood is:

    - For t=0: :math:`y[0] \sim N(\mu[0], \sigma_\nu / \sqrt{1-\rho^2})` (stationary initial condition)
    - For t>0: :math:`y[t] - \rho \cdot y[t-1] \sim N(\mu[t] - \rho \cdot \mu[t-1], \sigma_\nu)`

    Advantages Over Independent Errors
    -----------------------------------
    - **Proper uncertainty quantification**: Accounts for temporal correlation in credible intervals
    - **More efficient inference**: Correctly models the error structure
    - **Better parameter recovery**: Avoids bias from ignoring autocorrelation
    - **Diagnostic information**: The :math:`\rho` parameter indicates strength of temporal dependence

    When to Use
    -----------
    Use this model when:

    - Time series data exhibits autocorrelation in residuals
    - Standard transfer function model shows diagnostic issues (e.g., correlated residuals)
    - You need proper uncertainty propagation with temporal dependence

    Use the standard TransferFunctionLinearRegression when:

    - Residuals show minimal autocorrelation
    - Computational efficiency is critical (AR model is slower)
    - You want a simpler baseline model for comparison

    Parameters
    ----------
    saturation_type : str or None
        Type of saturation transform. Options: "hill", "logistic", "michaelis_menten", None.
        If None, no saturation is applied.
    adstock_config : dict or None
        Configuration for adstock transform. Required keys:
        - "half_life_prior": dict with prior specification (e.g., {"dist": "Gamma", "alpha": 4, "beta": 2})
        - "l_max": int, maximum lag
        - "normalize": bool, whether to normalize weights
        If None, no adstock is applied.
    saturation_config : dict or None
        Configuration for saturation transform. Structure depends on saturation_type:
        - For "hill": {"slope_prior": {...}, "kappa_prior": {...}}
        - For "logistic": {"lam_prior": {...}}
        - For "michaelis_menten": {"alpha_prior": {...}, "lam_prior": {...}}
    coef_constraint : str, default="unconstrained"
        Constraint on treatment coefficients: "nonnegative" or "unconstrained".
    sample_kwargs : dict, optional
        Additional kwargs passed to pm.sample().

    Notes
    -----
    - The AR(1) coefficient :math:`\rho` has a Uniform(-0.99, 0.99) prior by default
    - The quasi-differencing approach ensures the model remains computationally tractable
    - Posterior predictive sampling requires forward simulation of the AR process
    - Convergence can be slower than the independent errors model; consider increasing tune/draws

    **Prior Customization**:

    Priors are managed using the ``Prior`` class from ``pymc_extras`` and can be
    customized via the ``priors`` parameter::

        from pymc_extras.prior import Prior

        model = cp.pymc_models.TransferFunctionARRegression(
            saturation_type=None,
            adstock_config={...},
            priors={
                "beta": Prior(
                    "Normal", mu=0, sigma=100, dims=["treated_units", "coeffs"]
                ),
                "rho": Prior(
                    "Uniform", lower=-0.95, upper=0.95, dims=["treated_units"]
                ),
            },
        )

    By default, data-informed priors are set automatically via ``priors_from_data()``:

    - Baseline coefficients (``beta``): ``Normal(0, 5 * std(y))``
    - Treatment coefficients (``theta_treatment``): ``Normal(0, 2 * std(y))`` or ``HalfNormal(2 * std(y))``
    - Error std (``sigma``): ``HalfNormal(2 * std(y))``
    - AR(1) coefficient (``rho``): ``Uniform(-0.99, 0.99)``

    This adaptive approach ensures priors are reasonable regardless of data scale.

    Examples
    --------
    Basic usage::

        import causalpy as cp

        model = cp.pymc_models.TransferFunctionARRegression(
            saturation_type=None,
            adstock_config={
                "half_life_prior": {"dist": "Gamma", "alpha": 4, "beta": 2},
                "l_max": 8,
                "normalize": True,
            },
            sample_kwargs={"chains": 4, "draws": 2000, "tune": 1000},
        )
        result = cp.GradedInterventionTimeSeries(
            data=df,
            y_column="outcome",
            treatment_names=["treatment"],
            base_formula="1 + time + covariate",
            model=model,
        )

    References
    ----------
    .. [1] Pankratz, A. (1991). "Forecasting with Dynamic Regression Models". Wiley.
    .. [2] Box, G. E., & Jenkins, G. M. (2015). "Time Series Analysis: Forecasting and Control". Wiley.
    """

    def __init__(
        self,
        saturation_type: str | None = None,
        adstock_config: dict | None = None,
        saturation_config: dict | None = None,
        coef_constraint: str = "unconstrained",
        sample_kwargs: dict[str, Any] | None = None,
        priors: dict[str, Any] | None = None,
    ):
        """Initialize TransferFunctionARRegression model."""
        super().__init__(sample_kwargs=sample_kwargs, priors=priors)

        # Validate that at least one transform is specified
        if saturation_type is None and adstock_config is None:
            raise ValueError(
                "At least one of saturation_type or adstock_config must be specified."
            )

        self.saturation_type = saturation_type
        self.adstock_config = adstock_config
        self.saturation_config = saturation_config or {}
        self.coef_constraint = coef_constraint

        # Store for later use
        self.treatment_names = None
        self.n_treatments = None

    def priors_from_data(self, X, y) -> dict[str, Any]:
        """
        Generate data-informed priors including AR(1) coefficient.

        Similar to TransferFunctionLinearRegression but also includes
        a prior for the AR(1) coefficient rho.

        Parameters
        ----------
        X : xr.DataArray
            Baseline design matrix.
        y : xr.DataArray
            Outcome variable.

        Returns
        -------
        Dict[str, Prior]
            Dictionary with Prior objects for beta, theta_treatment, sigma, and rho.

        Notes
        -----
        The returned dictionary contains Prior objects with the following structure::

            {
                "beta": Prior(
                    "Normal", mu=0, sigma=5 * y_scale, dims=["treated_units", "coeffs"]
                ),
                "theta_treatment": Prior(
                    "Normal",
                    mu=0,
                    sigma=2 * y_scale,
                    dims=["treated_units", "treatment_names"],
                ),
                "sigma": Prior("HalfNormal", sigma=2 * y_scale, dims=["treated_units"]),
                "rho": Prior(
                    "Uniform", lower=-0.99, upper=0.99, dims=["treated_units"]
                ),
            }

        where ``y_scale = std(y)``.
        """
        y_scale = float(np.std(y))

        priors = {
            "beta": Prior(
                "Normal",
                mu=0,
                sigma=5 * y_scale,
                dims=["treated_units", "coeffs"],
            ),
            "sigma": Prior(
                "HalfNormal",
                sigma=2 * y_scale,
                dims=["treated_units"],
            ),
            "rho": Prior(
                "Uniform",
                lower=-0.99,
                upper=0.99,
                dims=["treated_units"],
            ),
        }

        # Treatment coefficient prior depends on constraint
        if self.coef_constraint == "nonnegative":
            priors["theta_treatment"] = Prior(
                "HalfNormal",
                sigma=2 * y_scale,
                dims=["treated_units", "treatment_names"],
            )
        else:
            priors["theta_treatment"] = Prior(
                "Normal",
                mu=0,
                sigma=2 * y_scale,
                dims=["treated_units", "treatment_names"],
            )

        return priors

    def build_model(self, X, y, coords, treatment_data):
        """
        Build the PyMC model with transforms and AR(1) errors using quasi-differencing.

        Parameters
        ----------
        X : xr.DataArray
            Baseline design matrix (n_obs, n_baseline_features).
        y : xr.DataArray
            Outcome variable (n_obs, 1).
        coords : dict
            Coordinate names for PyMC.
        treatment_data : xr.DataArray
            Raw treatment variables (n_obs, n_treatments).
        """
        from pymc_marketing.mmm.transformers import (
            geometric_adstock,
            hill_function,
            logistic_saturation,
            michaelis_menten,
        )

        self.treatment_names = treatment_data.coords.get(
            "treatment_names",
            [f"treatment_{i}" for i in range(treatment_data.shape[1])],
        ).values.tolist()
        self.n_treatments = treatment_data.shape[1]

        with self:
            self.add_coords(coords)

            # Add coordinate for differenced observations (needed for AR(1) likelihood)
            # Note: y has shape (n_obs, 1), so differenced has shape (n_obs-1, 1)
            n_obs = y.shape[0]
            self.add_coords({"obs_ind_diff": range(n_obs - 1)})

            # Register data
            X_data = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            y_data = pm.Data("y", y, dims=["obs_ind", "treated_units"])
            treatment_raw = pm.Data(
                "treatment_raw", treatment_data, dims=["obs_ind", "treatment_names"]
            )

            # ==================================================================
            # Transform Parameters (if applicable)
            # ==================================================================
            transform_params = {}

            # Adstock transform parameters
            if self.adstock_config is not None:
                half_life_prior_config = self.adstock_config.get(
                    "half_life_prior", {"dist": "Gamma", "alpha": 4, "beta": 2}
                )
                if half_life_prior_config["dist"] == "Gamma":
                    half_life = pm.Gamma(
                        "half_life",
                        alpha=half_life_prior_config["alpha"],
                        beta=half_life_prior_config["beta"],
                    )
                elif half_life_prior_config["dist"] == "HalfNormal":
                    half_life = pm.HalfNormal(
                        "half_life",
                        sigma=half_life_prior_config["sigma"],
                    )
                else:
                    raise ValueError(
                        f"Unsupported prior distribution: {half_life_prior_config['dist']}"
                    )

                transform_params["half_life"] = half_life

                # Convert half_life to alpha for geometric_adstock
                alpha_adstock = pm.Deterministic(
                    "alpha_adstock", pt.power(0.5, 1.0 / half_life)
                )

            # Saturation transform parameters
            if self.saturation_type == "hill":
                slope_prior = self.saturation_config.get(
                    "slope_prior", {"dist": "HalfNormal", "sigma": 2}
                )
                kappa_prior = self.saturation_config.get(
                    "kappa_prior", {"dist": "HalfNormal", "sigma": 10}
                )

                if slope_prior["dist"] == "HalfNormal":
                    slope = pm.HalfNormal("slope", sigma=slope_prior["sigma"])
                elif slope_prior["dist"] == "Gamma":
                    slope = pm.Gamma(
                        "slope", alpha=slope_prior["alpha"], beta=slope_prior["beta"]
                    )

                if kappa_prior["dist"] == "HalfNormal":
                    kappa = pm.HalfNormal("kappa", sigma=kappa_prior["sigma"])
                elif kappa_prior["dist"] == "Gamma":
                    kappa = pm.Gamma(
                        "kappa", alpha=kappa_prior["alpha"], beta=kappa_prior["beta"]
                    )

                transform_params["slope"] = slope
                transform_params["kappa"] = kappa

            elif self.saturation_type == "logistic":
                lam_prior = self.saturation_config.get(
                    "lam_prior", {"dist": "HalfNormal", "sigma": 1}
                )
                if lam_prior["dist"] == "HalfNormal":
                    lam = pm.HalfNormal("lam", sigma=lam_prior["sigma"])
                elif lam_prior["dist"] == "Gamma":
                    lam = pm.Gamma(
                        "lam", alpha=lam_prior["alpha"], beta=lam_prior["beta"]
                    )
                transform_params["lam"] = lam

            elif self.saturation_type == "michaelis_menten":
                alpha_prior = self.saturation_config.get(
                    "alpha_prior", {"dist": "HalfNormal", "sigma": 1}
                )
                lam_prior = self.saturation_config.get(
                    "lam_prior", {"dist": "HalfNormal", "sigma": 100}
                )

                if alpha_prior["dist"] == "HalfNormal":
                    alpha_sat = pm.HalfNormal("alpha_sat", sigma=alpha_prior["sigma"])
                elif alpha_prior["dist"] == "Gamma":
                    alpha_sat = pm.Gamma(
                        "alpha_sat",
                        alpha=alpha_prior["alpha"],
                        beta=alpha_prior["beta"],
                    )

                if lam_prior["dist"] == "HalfNormal":
                    lam = pm.HalfNormal("lam", sigma=lam_prior["sigma"])
                elif lam_prior["dist"] == "Gamma":
                    lam = pm.Gamma(
                        "lam", alpha=lam_prior["alpha"], beta=lam_prior["beta"]
                    )

                transform_params["alpha_sat"] = alpha_sat
                transform_params["lam"] = lam

            # ==================================================================
            # Apply Transforms to Treatment Variables
            # ==================================================================
            treatment_transformed_list = []

            for i in range(self.n_treatments):
                treatment_i = treatment_raw[:, i]

                # Apply saturation
                if self.saturation_type == "hill":
                    treatment_i = hill_function(treatment_i, slope=slope, kappa=kappa)
                elif self.saturation_type == "logistic":
                    treatment_i = logistic_saturation(treatment_i, lam=lam)
                elif self.saturation_type == "michaelis_menten":
                    treatment_i = michaelis_menten(
                        treatment_i, alpha=alpha_sat, lam=lam
                    )

                # Apply adstock
                if self.adstock_config is not None:
                    l_max = self.adstock_config.get("l_max", 12)
                    normalize = self.adstock_config.get("normalize", True)
                    treatment_i = geometric_adstock(
                        treatment_i,
                        alpha=alpha_adstock,
                        l_max=l_max,
                        normalize=normalize,
                        mode="After",  # Causal: only past affects present
                    )

                treatment_transformed_list.append(treatment_i)

            # Stack transformed treatments
            treatment_transformed = pt.stack(treatment_transformed_list, axis=1)

            # ==================================================================
            # Regression Coefficients (with data-informed priors)
            # ==================================================================
            # Baseline coefficients: data-informed priors set via priors_from_data()
            beta = self.priors["beta"].create_variable("beta")

            # Treatment coefficients: data-informed priors set via priors_from_data()
            theta_treatment = self.priors["theta_treatment"].create_variable(
                "theta_treatment"
            )

            # ==================================================================
            # Mean Function
            # ==================================================================
            # Baseline contribution
            mu_baseline = pt.dot(X_data, beta.T)  # (n_obs, n_units)

            # Treatment contribution
            mu_treatment = pt.dot(
                treatment_transformed, theta_treatment.T
            )  # (n_obs, n_units)

            # Combined mean
            mu = pm.Deterministic(
                "mu", mu_baseline + mu_treatment, dims=["obs_ind", "treated_units"]
            )

            # ==================================================================
            # AR(1) Likelihood via Quasi-Differencing
            # ==================================================================
            # AR(1) parameter: data-informed prior set via priors_from_data()
            rho = self.priors["rho"].create_variable("rho")

            # Innovation standard deviation: data-informed prior set via priors_from_data()
            sigma = self.priors["sigma"].create_variable("sigma")

            # Quasi-differencing approach using manual log-likelihood
            # We can't use y_diff as observed data because it depends on rho

            # For t > 0: compute residuals of quasi-differenced model
            # residuals = (y[t] - rho * y[t-1]) - (mu[t] - rho * mu[t-1])
            y_diff = y_data[1:, :] - rho * y_data[:-1, :]
            mu_diff = mu[1:, :] - rho * mu[:-1, :]
            residuals_diff = y_diff - mu_diff

            # Compute log-likelihood for t > 0 (Gaussian log-likelihood formula)
            # log p(y[t] | y[t-1], params) = -0.5 * ((y_diff - mu_diff) / sigma)^2 - log(sigma) - 0.5 * log(2π)
            n_diff = y_data.shape[0] - 1  # Number of differenced observations
            n_units = y_data.shape[1]  # Number of units
            logp_diff = (
                -0.5 * pt.sum((residuals_diff / sigma) ** 2)
                - n_diff * pt.sum(pt.log(sigma))
                - 0.5 * n_diff * n_units * pt.log(2 * np.pi)
            )
            pm.Potential("logp_diff", logp_diff)

            # Initial observation (t=0) with stationary distribution
            # For AR(1), the stationary variance is sigma^2 / (1 - rho^2)
            residuals_0 = y_data[0, :] - mu[0, :]
            sigma_0 = sigma / pt.sqrt(1.0 - rho**2)

            # Compute log-likelihood for t=0
            logp_0 = (
                -0.5 * pt.sum((residuals_0 / sigma_0) ** 2)
                - pt.sum(pt.log(sigma_0))
                - 0.5 * n_units * pt.log(2 * np.pi)
            )
            pm.Potential("logp_0", logp_0)

    def fit(self, X, y, coords, treatment_data):
        """
        Fit the Transfer Function AR(1) model.

        This method overrides the base PyMCModel.fit() to accept treatment_data.

        Parameters
        ----------
        X : xr.DataArray
            Baseline design matrix (n_obs, n_baseline_features).
        y : xr.DataArray
            Outcome variable (n_obs, 1).
        coords : dict
            Coordinate names for PyMC model.
        treatment_data : xr.DataArray
            Raw treatment variables (n_obs, n_treatments).

        Returns
        -------
        idata : arviz.InferenceData
            Inference data with posterior, prior predictive, and posterior predictive.
        """
        # Ensure random_seed is used in sample_prior_predictive() and
        # sample_posterior_predictive() if provided in sample_kwargs.
        random_seed = self.sample_kwargs.get("random_seed", None)

        # Merge priors with precedence: user-specified > data-driven > defaults
        # Data-driven priors are computed first, then user-specified priors override them
        self.priors = {**self.priors_from_data(X, y), **self.priors}

        # Build the model with treatment data
        self.build_model(X, y, coords, treatment_data)

        with self:
            self.idata = pm.sample(**self.sample_kwargs)
            self.idata.extend(pm.sample_prior_predictive(random_seed=random_seed))
            self.idata.extend(
                pm.sample_posterior_predictive(
                    self.idata, progressbar=False, random_seed=random_seed
                )
            )
        return self.idata
