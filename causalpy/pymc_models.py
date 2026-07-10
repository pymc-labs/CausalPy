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
"""Custom PyMC models for causal inference."""

import inspect
import warnings
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr
from arviz import r2_score
from patsy import dmatrix
from pymc_extras.prior import Prior

from causalpy.constants import HDI_PROB
from causalpy.utils import round_num
from causalpy.variable_selection_priors import VariableSelectionPrior


def _as_xtensor_obs_ind(x: Any) -> Any:
    """Convert a tensor-like input to an obs_ind xtensor."""
    import pytensor.xtensor as ptx

    return ptx.as_xtensor(x, dims=("obs_ind",))


def _uses_xtensor_api(function: Any) -> bool:
    """Return True when an upstream transform expects xtensor inputs."""
    try:
        return "as_xtensor" in inspect.getsource(function)
    except (OSError, TypeError):
        code = getattr(function, "__code__", None)
        return code is not None and "as_xtensor" in code.co_names


def _call_time_component_apply(
    component: Any,
    t: Any,
) -> Any:
    """Call time components across tensor and xtensor variants."""
    parameters = inspect.signature(component.apply).parameters
    if _uses_xtensor_api(component.apply) or (
        "sum" in parameters and "result_callback" not in parameters
    ):
        t = _as_xtensor_obs_ind(t)
    result = component.apply(t)
    return getattr(result, "values", result)


def _call_seasonality_component_apply(
    seasonality_component: Any,
    dayofperiod: Any,
) -> Any:
    """Call seasonality components across tensor and xtensor variants."""
    return _call_time_component_apply(seasonality_component, dayofperiod)


class PyMCModel(pm.Model):
    """A wrapper class for PyMC models. This provides a scikit-learn like interface with
    methods like `fit`, `predict`, and `score`. It also provides other methods which are
    useful for causal inference.

    The base :meth:`_data_setter` assumes the model graph contains mutable data
    nodes named ``"X"`` (predictors) and ``"y"`` (target).  Subclasses that use
    different data nodes should override :meth:`_data_setter`.  See
    :class:`BayesianBasisExpansionTimeSeries` for an example.

    Parameters
    ----------
    sample_kwargs : dict, optional
        Dictionary of kwargs that get unpacked and passed to the
        :func:`pymc.sample` function. Defaults to an empty dictionary if
        ``None``.
    priors : dict, optional
        Dictionary of priors for the model. Defaults to ``None``, in which
        case default priors are used.

    Examples
    --------
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

        See Also
        --------
        WeightedSumFitter.priors_from_data : Example implementation that sets
            Dirichlet prior shape based on number of control units.

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
        self._user_priors = priors

        self.priors = {**self.default_priors, **(priors or {})}

    def _clone(self) -> "PyMCModel":
        """Create a fresh, unfitted copy with the same configuration.

        ``copy.deepcopy`` of a ``pm.Model`` subclass loses its class
        identity, so this method constructs a new instance from the
        stored init parameters instead.
        """
        return type(self)(
            sample_kwargs=dict(self.sample_kwargs),
            priors=self._user_priors,
        )

    def build_model(
        self, X: xr.DataArray, y: xr.DataArray, coords: dict[str, Any] | None
    ) -> None:
        """Construct the PyMC model graph.

        Subclasses must override this method to declare priors, deterministic
        nodes, and the likelihood for the model.

        Parameters
        ----------
        X : xarray.DataArray
            Input features with dimensions ``["obs_ind", "coeffs"]``.
        y : xarray.DataArray
            Target variable with dimensions ``["obs_ind", "treated_units"]``.
        coords : dict or None
            Mapping of named dimensions to coordinate labels for the
            underlying ``pm.Model``.

        Raises
        ------
        NotImplementedError
            Always, when called on the base class.
        """
        raise NotImplementedError(
            "This method must be implemented by a subclass"
        )  # pragma: no cover

    def _data_setter(self, X: xr.DataArray) -> None:
        """
        Set data for the model for prediction.

        This method is called by :meth:`predict` to register new predictor data
        and reshape the target placeholder so that ``pm.sample_posterior_predictive``
        can run with the new observation count.

        The base implementation updates mutable data nodes named ``"X"`` and
        ``"y"``.  Subclasses that use different data nodes should override this
        method.  See :class:`BayesianBasisExpansionTimeSeries` for an example.
        """
        for name in ("X", "y"):
            if name not in self.named_vars:
                raise ValueError(
                    f"Data node '{name}' not found in model. "
                    f"If your model uses different data node names, "
                    f"override _data_setter() (see "
                    f"BayesianBasisExpansionTimeSeries for an example)."
                )

        new_no_of_observations = X.shape[0]

        # Use integer indices for obs_ind to avoid datetime compatibility issues with PyMC
        obs_coords = np.arange(new_no_of_observations)

        with self:
            # Get the number of treated units from the model coordinates
            treated_units_coord = getattr(self, "coords", {}).get(
                "treated_units", ["unit_0"]
            )
            n_treated_units = len(treated_units_coord)

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
        Predict data given input data `X`.

        .. caution::
            Results in KeyError if model hasn't been fit.

        Parameters
        ----------
        X : xr.DataArray
            Input features for which predictions are required.
        coords : dict, optional
            Coordinate names for named dimensions. Forwarded to subclass
            ``_data_setter`` overrides; ignored by the base implementation.
        out_of_sample : bool, optional
            Marker for out-of-sample prediction. Reserved for subclasses;
            the base implementation does not act on it.
        **kwargs
            Reserved for subclass extensions.
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

        Parameters
        ----------
        X : xr.DataArray
            Input features.
        y : xr.DataArray
            Observed targets to score against the posterior predictive mean.
        coords : dict, optional
            Coordinate names for named dimensions. Forwarded to
            :meth:`predict`; ignored by the base implementation.
        **kwargs
            Reserved for subclass extensions.
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
        """Cumulative sum of pointwise causal impact along ``obs_ind``.

        Parameters
        ----------
        impact : xarray.DataArray
            Pointwise causal impact, typically the output of
            :meth:`calculate_impact`.

        Returns
        -------
        xarray.DataArray
            Cumulative impact accumulated along the ``obs_ind`` dimension.
        """
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

        def _print_row(
            max_label_length: int, name: str, coeff_samples: xr.DataArray, round_to: int
        ) -> None:
            """Print one row of the coefficient table."""
            formatted_name = f"  {name: <{max_label_length}}"
            formatted_val = f"{round_num(coeff_samples.mean().data, round_to)}, {HDI_PROB * 100:.0f}% HDI [{round_num(coeff_samples.quantile((1 - HDI_PROB) / 2).data, round_to)}, {round_num(coeff_samples.quantile(1 - (1 - HDI_PROB) / 2).data, round_to)}]"  # noqa: E501
            print(f"  {formatted_name}  {formatted_val}")

        def _print_coefficients_for_unit(
            unit_coeffs: xr.DataArray,
            unit_sigma: xr.DataArray,
            labels: list,
            round_to: int,
        ) -> None:
            """Print coefficients for a single unit."""
            # Determine the width of the longest label
            max_label_length = max(len(name) for name in labels + ["y_hat_sigma"])

            for name in labels:
                coeff_samples = unit_coeffs.sel(coeffs=name)
                _print_row(max_label_length, name, coeff_samples, round_to)

            # Add coefficient for measurement std
            _print_row(max_label_length, "y_hat_sigma", unit_sigma, round_to)

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
            _print_coefficients_for_unit(unit_coeffs, unit_sigma, labels, round_to or 2)


class LinearRegression(PyMCModel):
    r"""
    Custom PyMC model for linear regression.

    Defines the PyMC model

    .. math::
        \beta &\sim \mathrm{Normal}(0, 50) \\
        \sigma &\sim \mathrm{HalfNormal}(1) \\
        \mu &= X \cdot \beta \\
        y &\sim \mathrm{Normal}(\mu, \sigma) \\

    Examples
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
        Define the PyMC model.

        Parameters
        ----------
        X : xr.DataArray
            Design matrix with dims ``("obs_ind", "coeffs")``.
        y : xr.DataArray
            Outcome with dims ``("obs_ind", "treated_units")``.
        coords : dict or None
            Coordinate names for the model's named dimensions.
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

    Examples
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
        Define the PyMC model.

        Parameters
        ----------
        X : xr.DataArray
            Design matrix with dims ``("obs_ind", "coeffs")``.
        y : xr.DataArray
            Outcome with dims ``("obs_ind", "treated_units")``.
        coords : dict or None
            Coordinate names for the model's named dimensions.
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


def _softmax_simplex_weights(
    name: str,
    prior: "Prior",
    n_rows: int,
    dims: list[str],
) -> "pt.TensorVariable":
    """Create simplex weights via softmax-over-Normal-logits with pinned reference.

    Pins the first logit to zero (removing softmax shift invariance), samples
    ``N - 1`` unconstrained Normal logits, and applies softmax to produce
    weights on the simplex.

    Parameters
    ----------
    name : str
        Name for the PyMC Deterministic variable (e.g., "beta", "omega").
    prior : Prior
        Prior for the raw logits. Must be a Normal prior with appropriate dims.
    n_rows : int
        Number of rows in the weight matrix (e.g., n_treated for SC, 1 for SDiD).
    dims : list[str]
        Dimension names for the output Deterministic.

    Returns
    -------
    pt.TensorVariable
        Simplex weights as a PyMC Deterministic.
    """
    if prior.distribution != "Normal":
        raise ValueError(
            f"_softmax_simplex_weights expects a Normal prior, got {prior.distribution}"
        )
    raw = prior.create_variable(f"{name}_raw")
    if n_rows == 1 and raw.ndim == 1:
        # When the prior has no "treated_units" dim (e.g. SDiD's omega_raw with
        # dims=["coeffs_raw"]), PyMC creates a 1D tensor.  Concatenate along
        # axis 0 to produce a 1D simplex of shape (N,).
        zero_logit = pt.zeros((1,))
        tilde = pt.concatenate([zero_logit, raw], axis=0)
        return pm.Deterministic(name, pt.special.softmax(tilde, axis=-1), dims=dims)
    zero_logit = pt.zeros((n_rows, 1))
    tilde = pt.concatenate([zero_logit, raw], axis=-1)
    return pm.Deterministic(name, pt.special.softmax(tilde, axis=-1), dims=dims)


class SoftmaxWeightedSumFitter(PyMCModel):
    r"""
    Weighted sum model with softmax-over-Normal-logits parameterization.

    An alternative to :class:`WeightedSumFitter` for synthetic control experiments.
    Instead of a Dirichlet prior on the simplex weights, this model places Normal
    priors on unconstrained logits and maps them to the simplex via the softmax
    transform. The first logit is pinned to zero to remove the softmax's shift
    invariance.

    Defines the PyMC model:

    .. math::
        \tilde{\beta}_1 &= 0 \\
        \tilde{\beta}_{j} &\sim \mathrm{Normal}(0, \sigma) \quad j = 2, \ldots, N \\
        \beta &= \mathrm{softmax}(\tilde{\beta}) \\
        \mu &= X \cdot \beta \\
        y &\sim \mathrm{Normal}(\mu, \sigma_y) \\

    Notes
    -----
    The softmax-Normal parameterization and the Dirichlet prior used by
    :class:`WeightedSumFitter` both produce simplex-valued weights, but they
    encode different prior beliefs and regularization behavior:

    - **Dirichlet** (:class:`WeightedSumFitter`): With concentration ``a=1`` the
      prior is uniform on the simplex. Setting ``a < 1`` encourages sparsity
      (weights concentrating on fewer donors), while ``a > 1`` encourages
      uniformity. Regularization strength is controlled by the concentration
      parameter.

    - **Softmax-Normal** (this class): The prior scale ``sigma`` on the logits
      controls regularization. Small ``sigma`` shrinks logits toward zero,
      producing near-uniform weights (DiD-like behavior). Large ``sigma`` allows
      the data to concentrate weight on a few well-matching control units
      (SC-like behavior). The default ``sigma=1.0`` provides moderate
      regularization.

    This parameterization is motivated by the Bayesian Synthetic
    Difference-in-Differences (SDiD) formulation, where the prior scale plays
    the role of the :math:`\ell_2` regularization parameter :math:`\zeta` in the
    frequentist SDiD of Arkhangelsky et al. (2021).

    Examples
    --------
    >>> import causalpy as cp
    >>> import numpy as np
    >>> import xarray as xr
    >>> from causalpy.pymc_models import SoftmaxWeightedSumFitter
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
    >>> wsf = SoftmaxWeightedSumFitter(sample_kwargs={"progressbar": False})
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
        Set Normal prior for logit weights based on number of control units.

        The prior is placed on ``N - 1`` unconstrained logits (the first logit is
        pinned to zero). The default scale ``sigma=1.0`` provides moderate
        regularization, equivalent to ``zeta=1.0`` in the frequentist SDiD.

        Unlike :meth:`WeightedSumFitter.priors_from_data`, which must read
        ``X.shape[1]`` to size the Dirichlet concentration vector, the Normal
        prior here broadcasts automatically via its ``dims``, so the data shape
        is not needed.

        To control regularization strength, pass a custom ``beta_raw`` prior::

            # Tighter regularization (more DiD-like, near-uniform weights):
            model = SoftmaxWeightedSumFitter(
                priors={
                    "beta_raw": Prior(
                        "Normal", mu=0, sigma=0.1, dims=["treated_units", "coeffs_raw"]
                    )
                }
            )

            # Looser regularization (more SC-like, data-driven sparse weights):
            model = SoftmaxWeightedSumFitter(
                priors={
                    "beta_raw": Prior(
                        "Normal", mu=0, sigma=10, dims=["treated_units", "coeffs_raw"]
                    )
                }
            )

        Parameters
        ----------
        X : xarray.DataArray
            Control unit data with shape (n_obs, n_control_units).
        y : xarray.DataArray
            Treated unit outcome data.

        Returns
        -------
        dict[str, Prior]
            Dictionary containing:

            - "beta_raw": Normal prior with dims ["treated_units", "coeffs_raw"]
        """
        return {
            "beta_raw": Prior(
                "Normal",
                mu=0,
                sigma=1.0,
                dims=["treated_units", "coeffs_raw"],
            ),
        }

    def build_model(
        self, X: xr.DataArray, y: xr.DataArray, coords: dict[str, Any] | None
    ) -> None:
        """
        Build the PyMC model with softmax-parameterized simplex weights.

        Parameters
        ----------
        X : xr.DataArray
            Design matrix with dims ``("obs_ind", "coeffs")``.
        y : xr.DataArray
            Outcome with dims ``("obs_ind", "treated_units")``.
        coords : dict or None
            Coordinate names for the model's named dimensions.
        """
        if not coords or "coeffs" not in coords:
            raise ValueError(
                "coords must include 'coeffs' for SoftmaxWeightedSumFitter"
            )
        coeffs_raw = coords["coeffs"][1:]

        with self:
            coords_with_raw = dict(coords) if coords else {}
            coords_with_raw["coeffs_raw"] = coeffs_raw
            self.add_coords(coords_with_raw)

            X = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            y = pm.Data("y", y, dims=["obs_ind", "treated_units"])

            beta = _softmax_simplex_weights(
                name="beta",
                prior=self.priors["beta_raw"],
                n_rows=y.shape[1],
                dims=["treated_units", "coeffs"],
            )

            mu = pm.Deterministic(
                "mu", pt.dot(X, beta.T), dims=["obs_ind", "treated_units"]
            )
            self.priors["y_hat"].create_likelihood_variable("y_hat", mu=mu, observed=y)


class SyntheticDifferenceInDifferencesWeightFitter(PyMCModel):
    r"""
    Bayesian weight fitter for Synthetic Difference-in-Differences.

    Encodes both the unit-weight module and the time-weight module in a single
    PyMC model. Unit weights balance control units against treated units in the
    pre-treatment period; time weights balance pre-treatment periods against
    post-treatment periods for control units. Both use the softmax-over-Normal-logits
    parameterization with a pinned reference level.

    The treatment effect is **not** estimated inside this model. It is computed
    analytically from the weight posteriors via the double-difference formula
    in the experiment class.

    Defines the PyMC model:

    .. math::
        \omega &= \mathrm{softmax}(0, \tilde{\omega}_2, \ldots, \tilde{\omega}_{N_\text{co}}) \\
        \bar{Y}_{\text{tr},t} &\sim \mathrm{Normal}(\omega_0 + \boldsymbol{\omega}^\top \mathbf{Y}_{\text{co},t},\; \sigma_\omega) \\
        \lambda &= \mathrm{softmax}(0, \tilde{\lambda}_2, \ldots, \tilde{\lambda}_{T_\text{pre}}) \\
        \bar{Y}_{i,\text{post}} &\sim \mathrm{Normal}(\lambda_0 + \boldsymbol{\lambda}^\top \mathbf{Y}_{i,\text{pre}},\; \sigma_\lambda)

    Notes
    -----
    This model implements the cut-posterior formulation of Bayesian SDiD.
    Modules 1 and 2 share no parameters and are
    conditionally independent given the data. Running them in a single MCMC
    call is a convenience; the important property is that no treatment-effect
    likelihood feeds back into the weight posteriors.

    The prior scales on the logits play the role of the regularization
    parameter in the frequentist SDiD:

    - ``omega_raw`` default ``sigma=1.0`` (``zeta_omega=1.0``): moderate
      regularization, allowing weights between SC-sparse and DiD-uniform.
    - ``lam_raw`` default ``sigma=100.0`` (``zeta_lambda=0.01``): essentially
      flat, letting time weights concentrate on the most informative
      pre-treatment periods.

    References
    ----------
    .. [1] Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., &
       Wager, S. (2021). Synthetic Difference-in-Differences. *American
       Economic Review*, 111(12), 4088-4118.
    """  # noqa: W605

    default_priors: dict[str, Prior] = {}

    def priors_from_data(self, X, y) -> dict[str, Any]:
        """
        Set default priors for unit and time weight modules.

        Parameters
        ----------
        X : dict
            Dict with keys ``"unit"`` and ``"time"``, each an xarray.DataArray.
        y : dict
            Dict with keys ``"unit"`` and ``"time"``, each an xarray.DataArray.

        Returns
        -------
        dict[str, Prior]
            Priors for omega_raw, lam_raw, omega0, lambda0, sigma_omega,
            sigma_lambda.
        """
        return {
            "omega_raw": Prior(
                "Normal",
                mu=0,
                sigma=1.0,
                dims=["coeffs_raw"],
            ),
            "lam_raw": Prior(
                "Normal",
                mu=0,
                sigma=100.0,
                dims=["obs_ind_raw"],
            ),
            "omega0": Prior("Normal", mu=0, sigma=5.0),
            "lambda0": Prior("Normal", mu=0, sigma=5.0),
            "sigma_omega": Prior("HalfNormal", sigma=1.0),
            "sigma_lambda": Prior("HalfNormal", sigma=1.0),
        }

    def build_model(self, X, y, coords: dict[str, Any] | None) -> None:
        """
        Build the PyMC model with both unit-weight and time-weight modules.

        Parameters
        ----------
        X : dict
            Mapping with ``"unit"`` and ``"time"`` design matrices for the
            unit-weight and time-weight modules respectively.
        y : dict
            Mapping with ``"unit"`` and ``"time"`` outcome arrays for the
            unit-weight and time-weight modules respectively.
        coords : dict or None
            Coordinate names for the model's named dimensions.
        """
        with self:
            self.add_coords(coords)

            # Data
            X_unit = pm.Data("X_unit", X["unit"], dims=["obs_ind", "coeffs"])
            y_unit = pm.Data("y_unit", y["unit"], dims=["obs_ind"])
            X_time = pm.Data("X_time", X["time"], dims=["coeffs", "obs_ind"])
            y_time = pm.Data("y_time", y["time"], dims=["coeffs"])

            # Module 1: Unit weights
            omega = _softmax_simplex_weights(
                name="omega",
                prior=self.priors["omega_raw"],
                n_rows=1,
                dims=["coeffs"],
            )  # shape (N_co,) — 1D from helper when n_rows=1
            omega0 = self.priors["omega0"].create_variable("omega0")
            sigma_omega = self.priors["sigma_omega"].create_variable("sigma_omega")
            mu_omega = omega0 + pt.dot(X_unit, omega)
            pm.Normal("omega_match", mu=mu_omega, sigma=sigma_omega, observed=y_unit)

            # Module 2: Time weights
            lam = _softmax_simplex_weights(
                name="lam",
                prior=self.priors["lam_raw"],
                n_rows=1,
                dims=["obs_ind"],
            )  # shape (T_pre,) — 1D from helper when n_rows=1
            lambda0 = self.priors["lambda0"].create_variable("lambda0")
            sigma_lambda = self.priors["sigma_lambda"].create_variable("sigma_lambda")
            mu_lambda = lambda0 + pt.dot(X_time, lam)
            pm.Normal("lambda_match", mu=mu_lambda, sigma=sigma_lambda, observed=y_time)


class InstrumentalVariableRegression(PyMCModel):
    """Custom PyMC model for instrumental linear regression.

    Examples
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
        vs_prior_type : {"spike_and_slab", "horseshoe", "normal"}, optional
            Optional variable-selection prior type. ``None`` falls back to
            standard normal priors.
        vs_hyperparams : dict, optional
            Hyperparameters for the variable-selection prior. Only consulted
            when ``vs_prior_type`` is set.
        binary_treatment : bool, default False
            Whether the treatment ``t`` is binary; selects the relevant
            likelihood term.
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
        and posterior predictive distributions.

        Parameters
        ----------
        ppc_sampler : {"jax", "pymc"}, optional
            Backend used for posterior predictive sampling. ``"jax"`` (the
            default) is much faster for the multivariate Normal likelihood;
            ``"pymc"`` additionally samples the prior predictive.
        """
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

        Parameters
        ----------
        X : np.ndarray
            Array used to predict the outcome ``y``.
        Z : np.ndarray
            Array used to predict the treatment variable ``t``.
        y : np.ndarray
            Focal outcome.
        t : np.ndarray
            Treatment whose causal impact is being estimated.
        coords : dict
            Coordinate names for the instruments and covariates.
        priors : dict
            Prior specification dictionary forwarded to :meth:`build_model`.
        ppc_sampler : {"jax", "pymc"}, optional
            Backend for posterior predictive sampling. ``None`` skips it.
        vs_prior_type : {"spike_and_slab", "horseshoe", "normal"}, optional
            Variable-selection prior type, forwarded to :meth:`build_model`.
        vs_hyperparams : dict, optional
            Hyperparameters for the variable-selection prior.
        binary_treatment : bool, default False
            Whether the treatment ``t`` is binary.
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
        return self.idata


class PropensityScore(PyMCModel):
    r"""Custom PyMC model for inverse propensity score models.

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

    Examples
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
        """Define the PyMC propensity model.

        Parameters
        ----------
        X : np.ndarray
            Covariate matrix used to predict the treatment.
        t : np.ndarray
            Observed treatment indicator (0/1).
        coords : dict
            Coordinate names for named dimensions of the model.
        prior : dict, optional
            Prior specification overrides; see :attr:`default_priors` for
            the expected keys.
        noncentred : bool, default True
            Reserved for future non-centred parameterisations of the
            coefficient prior. Currently informational only.
        """
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

        Parameters
        ----------
        X : np.ndarray
            Covariate matrix used to predict the treatment.
        t : np.ndarray
            Observed treatment indicator (0/1).
        coords : dict
            Coordinate names for named dimensions of the model.
        prior : dict, optional
            Prior specification overrides. Defaults to ``{"b": [0, 1]}``.
        noncentred : bool, default True
            Forwarded to :meth:`build_model`.
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

        spline_knots : int, default 30
            The number of knots we use in the 0 - 1 interval to create our spline function.

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
            raise AttributeError(
                """Object is missing required attribute 'idata'
                                 so cannot proceed. Call fit() first"""
            )
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
    priors : dict, optional
        Dictionary of priors for the model. Defaults to ``None``, in which
        case default priors are used.
    """  # noqa: W605

    def __init__(
        self,
        n_order: int = 3,
        n_changepoints_trend: int = 10,
        prior_sigma: float = 5,
        trend_component: Any | None = None,
        seasonality_component: Any | None = None,
        sample_kwargs: dict[str, Any] | None = None,
        priors: dict[str, Any] | None = None,
    ):
        super().__init__(sample_kwargs=sample_kwargs, priors=priors)

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

    def _clone(self) -> "PyMCModel":
        """Create a fresh, unfitted copy with the same configuration."""
        return type(self)(
            n_order=self.n_order,
            n_changepoints_trend=self.n_changepoints_trend,
            prior_sigma=self.prior_sigma,
            trend_component=self._custom_trend_component,
            seasonality_component=self._custom_seasonality_component,
            sample_kwargs=dict(self.sample_kwargs),
            priors=self._user_priors,
        )

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
            ``(time_for_trend, time_for_seasonality, X_for_pymc, num_obs)``:

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
                _call_seasonality_component_apply(
                    seasonality_component_instance, t_season_data
                ),
                dims="obs_ind",
            )

            # Trend component
            trend_component_values = _call_time_component_apply(
                trend_component_instance, t_trend_data
            )
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
        **kwargs
            Reserved for forward-compatibility; not consumed by this
            implementation.

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
        **kwargs
            Forwarded to :meth:`PyMCModel.score`.

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
    priors : dict, optional
        Dictionary of priors for the model. Defaults to ``None``, in which
        case default priors are used.
    """

    def __init__(
        self,
        level_order: int = 2,
        seasonal_length: int = 12,
        trend_component: Any | None = None,
        seasonality_component: Any | None = None,
        sample_kwargs: dict[str, Any] | None = None,
        mode: str | None = None,
        priors: dict[str, Any] | None = None,
    ):
        super().__init__(sample_kwargs=sample_kwargs, priors=priors)

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

    def _clone(self) -> "PyMCModel":
        """Create a fresh, unfitted copy with the same configuration."""
        return type(self)(
            level_order=self.level_order,
            seasonal_length=self.seasonal_length,
            trend_component=self._custom_trend_component,
            seasonality_component=self._custom_seasonality_component,
            sample_kwargs=dict(self.sample_kwargs),
            mode=self.mode,
            priors=self._user_priors,
        )

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
            # Keep Normal (not ZeroSumNormal): frequency-state coefficients are
            # unconstrained here; see PR #679 for rationale and context.
            _annual_seasonal = pm.Normal("params_freq", sigma=80, dims=annual_dims)

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
        **kwargs
            Reserved for forward-compatibility; not consumed by this
            implementation.

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
        **kwargs
            Forwarded to :meth:`PyMCModel.score`.

        Returns
        -------
        pd.Series
            R² score and standard deviation for each treated unit.
        """
        # Use base class implementation - X is accepted but not used by predict()
        return super().score(X, y, coords, **kwargs)


class HierarchicalLaunchITS(PyMCModel):
    r"""
    Hierarchical Bayesian interrupted time series model for multi-unit panels
    with unit-specific treatment (launch) times.

    Each unit (e.g. a product) has its own launch time. Per-unit intercepts,
    covariate effects and launch "lift" are partially pooled toward shared
    population-level hyperparameters, which lets sparse units borrow strength
    from well-observed ones and yields a posterior distribution over the
    population lift that can be used to forecast the effect of a *new* unit.

    Four effect parameterizations are supported via the experiment layer:

    - ``"instant"``: a single post-launch level shift per unit,
      ``lift[unit] ~ Normal(mu_lift, sigma_lift)``.
    - ``"event_study"``: per-unit dynamic coefficients ``delta[unit, event_bin]``
      over user-specified post-launch event-time bins (pre-launch is the
      implicit reference).
    - ``"placebo"``: the event-study form extended with pre-launch leads; the
      most negative leads serve as placebos that should be indistinguishable
      from zero if the no-anticipation assumption holds.
    - ``"saturation"``: the post-launch effect follows a Hill (logistic-type)
      saturation curve in event time rather than an instant level shift,
      ``effect(t) = post_t * L[unit] * tau^s / (k[unit]^s + tau^s)`` where
      ``tau = max(t - launch, 0)``. ``L`` (ceiling lift) and ``k``
      (half-saturation time) are hierarchical per-unit (log scale,
      non-centered); the Hill exponent ``s`` is a single shared
      population-level shape parameter. This describes an effect that ramps
      up smoothly after launch and asymptotes, rather than jumping instantly
      to its final size.

    All hierarchical parameters use a non-centered parametrization.

    Expected inputs
    ---------------
    - ``X`` : ``xr.DataArray`` with dims ``["obs_ind", "coeffs"]`` holding the
      standardized covariate design (built by the experiment from a patsy
      formula, *without* an intercept — the hierarchical ``alpha`` plays that
      role).
    - ``y`` : ``xr.DataArray`` with dims ``["obs_ind", "treated_units"]``; the
      single ``treated_units`` entry is retained for API consistency with the
      rest of CausalPy — the per-unit hierarchy is indexed via the ``unit``
      coord instead.
    - ``aux`` (passed to :meth:`fit`/:meth:`predict`) : dict with keys
      ``effect_type``, ``unit_idx`` (int array of length ``n_obs``) and, as
      appropriate, ``F`` (Fourier basis), ``post`` (0/1 indicator), ``D``
      (one-hot event-bin design) and ``tau_since`` (event time clipped at 0,
      for ``"saturation"``).

    See :class:`causalpy.experiments.hierarchical_interrupted_time_series.HierarchicalInterruptedTimeSeries`
    for the user-facing experiment wrapper that assembles ``aux`` from a long
    panel DataFrame.
    """

    def __init__(
        self,
        sample_kwargs: dict[str, Any] | None = None,
        priors: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(sample_kwargs=sample_kwargs, priors=priors)
        self._aux: dict[str, Any] | None = None
        self._pred_aux: dict[str, Any] | None = None

    def _clone(self) -> "HierarchicalLaunchITS":
        return type(self)(
            sample_kwargs=dict(self.sample_kwargs),
            priors=self._user_priors,
        )

    def priors_from_data(self, X, y) -> dict[str, Any]:
        """Data-adaptive priors keyed to the scale of ``y``."""
        y_mean = float(np.asarray(y).mean())
        y_std = float(np.asarray(y).std())
        if not np.isfinite(y_std) or y_std == 0.0:
            y_std = 1.0
        return {
            "mu_alpha": Prior("Normal", mu=y_mean, sigma=2 * y_std),
            "sigma_alpha": Prior("HalfNormal", sigma=y_std),
            "mu_gamma": Prior("Normal", mu=0.0, sigma=2 * y_std),
            "sigma_gamma": Prior("HalfNormal", sigma=y_std),
            "mu_lift": Prior("Normal", mu=0.0, sigma=2 * y_std),
            "sigma_lift": Prior("HalfNormal", sigma=y_std),
            "mu_delta": Prior("Normal", mu=0.0, sigma=2 * y_std, dims="event_bin"),
            "sigma_delta": Prior("HalfNormal", sigma=y_std, dims="event_bin"),
            "mu_rho": Prior("Normal", mu=0.0, sigma=0.5),
            "sigma_rho": Prior("HalfNormal", sigma=0.25),
            "sigma_ar": Prior("HalfNormal", sigma=y_std),
            # Saturation (Hill-curve) effect priors. `mu_logL` is data-adaptive,
            # centering the ceiling lift `L` near the outcome's scale (same
            # spirit as `mu_lift`). `mu_logk`/`s` use fixed, generic defaults
            # on the log/natural scale (like `mu_rho`/`sigma_rho` above) —
            # override via `priors=` if `time_col` isn't in "a handful of
            # periods to ramp up" units (e.g. daily rather than weekly data).
            "mu_logL": Prior("Normal", mu=float(np.log(max(y_std, 1e-3))), sigma=1.5),
            "sigma_logL": Prior("HalfNormal", sigma=1.0),
            "mu_logk": Prior("Normal", mu=float(np.log(5.0)), sigma=1.5),
            "sigma_logk": Prior("HalfNormal", sigma=0.75),
            "s": Prior("Gamma", mu=2.0, sigma=1.5),
            "sigma": Prior("HalfNormal", sigma=y_std),
        }

    def fit(  # type: ignore[override]
        self,
        X: xr.DataArray,
        y: xr.DataArray,
        coords: dict[str, Any] | None = None,
        aux: dict[str, Any] | None = None,
    ) -> az.InferenceData:
        """Fit the hierarchical launch ITS model.

        Parameters
        ----------
        X, y, coords :
            Standard PyMCModel inputs. ``coords`` must include a ``unit`` entry
            (and an ``event_bin`` entry for the event-study / placebo variants).
        aux : dict
            Auxiliary design inputs; see the class docstring.
        """
        if aux is None:
            raise ValueError(
                "HierarchicalLaunchITS.fit() requires the `aux` keyword argument"
            )
        self._aux = aux
        return super().fit(X, y, coords)

    def build_model(
        self, X: xr.DataArray, y: xr.DataArray, coords: dict[str, Any] | None
    ) -> None:
        if self._aux is None:
            raise RuntimeError(
                "HierarchicalLaunchITS.build_model called without aux. "
                "Use .fit(..., aux=...) rather than calling build_model directly."
            )
        aux = self._aux
        effect_type = aux["effect_type"]
        unit_idx_np = np.asarray(aux["unit_idx"], dtype=np.int64)
        F_np = aux.get("F")
        D_np = aux.get("D")
        post_np = aux.get("post")

        X_values = np.asarray(X)
        n_obs, n_coeffs = X_values.shape

        with self:
            if coords is None:
                coords = {}
            coords = dict(coords)
            coords.setdefault("treated_units", ["unit_0"])
            coords.setdefault("obs_ind", np.arange(n_obs))
            if "coeffs" not in coords:
                coords["coeffs"] = [f"x{i}" for i in range(n_coeffs)]
            self.add_coords(coords)

            X_ = pm.Data("X", X_values, dims=["obs_ind", "coeffs"])
            y_ = pm.Data("y", np.asarray(y), dims=["obs_ind", "treated_units"])
            unit_idx_ = pm.Data("unit_idx", unit_idx_np, dims="obs_ind")

            # Hierarchical intercept (non-centered)
            mu_alpha = self.priors["mu_alpha"].create_variable("mu_alpha")
            sigma_alpha = self.priors["sigma_alpha"].create_variable("sigma_alpha")
            z_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims="unit")
            alpha = pm.Deterministic(
                "alpha", mu_alpha + z_alpha * sigma_alpha, dims="unit"
            )

            # Hierarchical per-unit time trend (non-centered)
            time_np = aux.get("time")
            if time_np is not None:
                time_ = pm.Data(
                    "time",
                    np.asarray(time_np, dtype=np.float64),
                    dims="obs_ind",
                )
                mu_gamma = self.priors["mu_gamma"].create_variable("mu_gamma")
                sigma_gamma = self.priors["sigma_gamma"].create_variable("sigma_gamma")
                z_gamma = pm.Normal("z_gamma", 0.0, 1.0, dims="unit")
                gamma = pm.Deterministic(
                    "gamma", mu_gamma + z_gamma * sigma_gamma, dims="unit"
                )
                trend_contrib = gamma[unit_idx_] * time_
            else:
                trend_contrib = pt.zeros(n_obs)

            # Hierarchical per-unit covariate effects (non-centered)
            if n_coeffs > 0:
                mu_beta = pm.Normal("mu_beta", 0.0, 2.0, dims="coeffs")
                sigma_beta = pm.HalfNormal("sigma_beta", 1.0, dims="coeffs")
                z_beta = pm.Normal("z_beta", 0.0, 1.0, dims=["unit", "coeffs"])
                beta = pm.Deterministic(
                    "beta",
                    mu_beta[None, :] + z_beta * sigma_beta[None, :],
                    dims=["unit", "coeffs"],
                )
                cov_contrib = pt.sum(X_ * beta[unit_idx_], axis=1)
            else:
                cov_contrib = pt.zeros(n_obs)

            # Shared Fourier seasonality
            if F_np is not None:
                F_ = pm.Data("F", np.asarray(F_np), dims=["obs_ind", "fourier"])
                beta_season = pm.Normal("beta_season", 0.0, 5.0, dims="fourier")
                season_contrib = pt.dot(F_, beta_season)
            else:
                season_contrib = pt.zeros(n_obs)

            # Effect component
            if effect_type == "instant":
                if post_np is None:
                    raise ValueError("effect_type='instant' requires aux['post']")
                post_ = pm.Data(
                    "post", np.asarray(post_np, dtype=np.float64), dims="obs_ind"
                )
                mu_lift = self.priors["mu_lift"].create_variable("mu_lift")
                sigma_lift = self.priors["sigma_lift"].create_variable("sigma_lift")
                z_lift = pm.Normal("z_lift", 0.0, 1.0, dims="unit")
                lift = pm.Deterministic(
                    "lift", mu_lift + z_lift * sigma_lift, dims="unit"
                )
                effect_contrib = lift[unit_idx_] * post_
            elif effect_type in ("event_study", "placebo"):
                if D_np is None:
                    raise ValueError(f"effect_type='{effect_type}' requires aux['D']")
                D_ = pm.Data(
                    "D",
                    np.asarray(D_np, dtype=np.float64),
                    dims=["obs_ind", "event_bin"],
                )
                mu_delta = self.priors["mu_delta"].create_variable("mu_delta")
                sigma_delta = self.priors["sigma_delta"].create_variable("sigma_delta")
                z_delta = pm.Normal("z_delta", 0.0, 1.0, dims=["unit", "event_bin"])
                delta = pm.Deterministic(
                    "delta",
                    mu_delta[None, :] + z_delta * sigma_delta[None, :],
                    dims=["unit", "event_bin"],
                )
                effect_contrib = pt.sum(D_ * delta[unit_idx_], axis=1)
            elif effect_type == "saturation":
                tau_since_np = aux.get("tau_since")
                if post_np is None or tau_since_np is None:
                    raise ValueError(
                        "effect_type='saturation' requires aux['post'] and "
                        "aux['tau_since']"
                    )
                post_ = pm.Data(
                    "post", np.asarray(post_np, dtype=np.float64), dims="obs_ind"
                )
                tau_since_ = pm.Data(
                    "tau_since",
                    np.asarray(tau_since_np, dtype=np.float64),
                    dims="obs_ind",
                )

                # Ceiling lift L[unit], hierarchical on the log scale (non-centered)
                mu_logL = self.priors["mu_logL"].create_variable("mu_logL")
                sigma_logL = self.priors["sigma_logL"].create_variable("sigma_logL")
                z_logL = pm.Normal("z_logL", 0.0, 1.0, dims="unit")
                L = pm.Deterministic(
                    "L", pt.exp(mu_logL + z_logL * sigma_logL), dims="unit"
                )

                # Half-saturation time k[unit], hierarchical on the log scale
                mu_logk = self.priors["mu_logk"].create_variable("mu_logk")
                sigma_logk = self.priors["sigma_logk"].create_variable("sigma_logk")
                z_logk = pm.Normal("z_logk", 0.0, 1.0, dims="unit")
                k = pm.Deterministic(
                    "k", pt.exp(mu_logk + z_logk * sigma_logk), dims="unit"
                )

                # Hill exponent s: shared population-level shape parameter
                s = self.priors["s"].create_variable("s")

                tau_safe = pt.maximum(tau_since_, 0.0)
                k_obs = k[unit_idx_]
                hill_curve = tau_safe**s / (k_obs**s + tau_safe**s + 1e-6)
                effect_contrib = post_ * L[unit_idx_] * hill_curve
            else:
                raise ValueError(
                    f"Unknown effect_type {effect_type!r}; expected "
                    "'instant', 'event_study', 'placebo' or 'saturation'"
                )

            # Optional hierarchical AR(1) residuals via scan
            within_unit_tidx_np = aux.get("within_unit_tidx")
            n_time_steps = aux.get("n_time_steps")
            if within_unit_tidx_np is not None and n_time_steps is not None:
                within_unit_tidx_ = pm.Data(
                    "within_unit_tidx",
                    np.asarray(within_unit_tidx_np, dtype=np.int64),
                    dims="obs_ind",
                )
                n_units = len(coords.get("unit", []))

                # Hierarchical AR coefficient (non-centered, tanh → stationarity)
                mu_rho = self.priors["mu_rho"].create_variable("mu_rho")
                sigma_rho = self.priors["sigma_rho"].create_variable("sigma_rho")
                z_rho = pm.Normal("z_rho", 0.0, 1.0, dims="unit")
                rho_raw = mu_rho + z_rho * sigma_rho
                rho = pm.Deterministic("rho", pt.tanh(rho_raw), dims="unit")

                # Pre-drawn innovations: (unit, time_step)
                z_ar = pm.Normal("z_ar", 0.0, 1.0, dims=["unit", "time_step"])
                sigma_ar = self.priors["sigma_ar"].create_variable("sigma_ar")

                def ar_step(z_t, prev, rho_vec):
                    """One AR(1) step: prev * rho + innovation."""
                    return rho_vec * prev + z_t

                # Scan along time axis, vectorised across units
                # z_ar.T is (time_step, unit); scan iterates over axis 0
                ar_resid_matrix = pytensor.scan(
                    fn=ar_step,
                    sequences=[z_ar.T],
                    outputs_info=[pt.zeros(n_units)],
                    non_sequences=[rho],
                    n_steps=n_time_steps,
                    return_updates=False,
                )
                # ar_resid_matrix: (time_step, unit)
                ar_contrib = sigma_ar * ar_resid_matrix[within_unit_tidx_, unit_idx_]
            else:
                ar_contrib = pt.zeros(n_obs)

            mu_row = (
                alpha[unit_idx_]
                + trend_contrib
                + cov_contrib
                + season_contrib
                + effect_contrib
                + ar_contrib
            )
            mu = pm.Deterministic(
                "mu", mu_row[:, None], dims=["obs_ind", "treated_units"]
            )
            sigma = self.priors["sigma"].create_variable("sigma")
            pm.Normal(
                "y_hat",
                mu=mu,
                sigma=sigma,
                observed=y_,
                dims=["obs_ind", "treated_units"],
            )

    def _data_setter(self, X: xr.DataArray) -> None:
        aux = self._pred_aux if self._pred_aux is not None else self._aux
        if aux is None:
            raise RuntimeError("HierarchicalLaunchITS has no aux data set")

        n = X.shape[0]
        obs_coords = np.arange(n)
        named = set(self.named_vars)
        data_to_set: dict[str, Any] = {
            "X": np.asarray(X),
            "y": np.zeros((n, 1)),
            "unit_idx": np.asarray(aux["unit_idx"], dtype=np.int64),
        }
        if "time" in named:
            time_val = aux.get("time")
            if time_val is None:
                raise ValueError("Model has time data but aux['time'] is missing")
            data_to_set["time"] = np.asarray(time_val, dtype=np.float64)
        if "F" in named:
            F_val = aux.get("F")
            if F_val is None:
                raise ValueError("Model has Fourier data but aux['F'] is missing")
            data_to_set["F"] = np.asarray(F_val)
        if "post" in named:
            post_val = aux.get("post")
            if post_val is None:
                raise ValueError("Model has post data but aux['post'] is missing")
            data_to_set["post"] = np.asarray(post_val, dtype=np.float64)
        if "D" in named:
            D_val = aux.get("D")
            if D_val is None:
                raise ValueError("Model has D data but aux['D'] is missing")
            data_to_set["D"] = np.asarray(D_val, dtype=np.float64)
        if "tau_since" in named:
            tau_val = aux.get("tau_since")
            if tau_val is None:
                raise ValueError(
                    "Model has tau_since data but aux['tau_since'] is missing"
                )
            data_to_set["tau_since"] = np.asarray(tau_val, dtype=np.float64)
        if "within_unit_tidx" in named:
            tidx_val = aux.get("within_unit_tidx")
            if tidx_val is None:
                raise ValueError("Model has within_unit_tidx but aux is missing it")
            data_to_set["within_unit_tidx"] = np.asarray(tidx_val, dtype=np.int64)

        with self:
            pm.set_data(data_to_set, coords={"obs_ind": obs_coords})

    def predict(  # type: ignore[override]
        self,
        X: xr.DataArray,
        coords: dict[str, Any] | None = None,
        out_of_sample: bool | None = False,
        aux: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> az.InferenceData:
        """Posterior predictive using the supplied ``aux`` bundle."""
        self._pred_aux = aux
        try:
            return super().predict(X, coords=coords, out_of_sample=out_of_sample)
        finally:
            self._pred_aux = None
