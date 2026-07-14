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
"""Adapter that lets a ``pymc_forecast`` forecasting model act as a model
provider behind CausalPy's experiment API.

CausalPy keeps identification, counterfactual construction, and placebo
methods; ``pymc_forecast`` provides the fitted forecasting model. The wrapper
maps CausalPy's backend protocol onto the ``pymc_forecast`` drivers:

- ``fit(X, y)`` constructs and fits the forecasting model on the pre-period.
- ``predict(X)`` (in-sample) uses ``predict_in_sample()``.
- ``predict(X, out_of_sample=True)`` draws the counterfactual with
  ``forecast(future_covariates=...)`` when the design matrix has columns, or
  ``forecast(future_index=...)`` for a covariate-free trend/seasonal model.
- Draw-level samples are extracted with ``prediction_samples()`` and the
  documented output schema dims are renamed onto CausalPy coords
  (``time`` / ``time_future`` -> ``obs_ind``, ``series`` -> ``treated_units``).

**When to reach for this backend vs the existing PyMCModel classes:** use
:class:`PyMCForecastModel` when the counterfactual is best expressed as a
proper forecasting model (local level / trend, stochastic seasonality, ARIMA-
style dynamics) built with ``pymc_forecast`` primitives, and you want its
priors, inference backends (ADVI / NUTS / Pathfinder), and forecasting
machinery. Stick with the native :class:`~causalpy.pymc_models.PyMCModel`
classes (e.g. ``LinearRegression``, ``BayesianBasisExpansionTimeSeries``) for
plain regression-style counterfactuals or when you need model coefficients
tied to the patsy design matrix.

Requires the optional dependency ``pymc-forecast`` (``pip install
causalpy[forecast]``).

Notes
-----
``pymc_forecast`` exposes draw-level posterior-predictive samples of the
observed variable rather than a separate noise-free expectation, so this
backend reports the posterior predictive as both ``mu`` and ``y_hat``. Causal
impact therefore includes observation-level noise, unlike native
``PyMCModel`` backends where impact is computed from the noise-free ``mu``.
"""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from arviz import r2_score

from causalpy.constants import HDI_PROB
from causalpy.utils import round_num

__all__ = ["PyMCForecastModel"]


def _import_pymc_forecast():
    """Import ``pymc_forecast`` lazily, failing with install instructions."""
    try:
        import pymc_forecast
    except ImportError as err:  # pragma: no cover - exercised without extra
        raise ImportError(
            "PyMCForecastModel requires the optional dependency 'pymc-forecast'. "
            "Install it with `pip install causalpy[forecast]` or "
            "`pip install pymc-forecast>=0.1.0`."
        ) from err
    return pymc_forecast


class PyMCForecastModel:
    """Wrap a ``pymc_forecast`` model as a CausalPy time-series backend.

    Parameters
    ----------
    model_fn : callable or pymc_forecast.ForecastingModel
        The forecasting model body ``(Horizon, covariates) -> None`` or a
        :class:`pymc_forecast.ForecastingModel` instance. Priors flow through
        this object (pymc-extras ``Prior``), preserving the transparent-prior
        ethos.
    forecaster : type, optional
        The ``pymc_forecast`` forecaster class used to fit the model. Defaults
        to :class:`pymc_forecast.HMCForecaster` (NUTS), matching CausalPy's
        native PyMC backends. Pass :class:`pymc_forecast.Forecaster` (ADVI) or
        :class:`pymc_forecast.PathfinderForecaster` for faster approximate
        inference — but check convergence before trusting the counterfactual.
    forecaster_kwargs : dict, optional
        Extra keyword arguments for the forecaster constructor (e.g.
        ``{"draws": 500}`` for MCMC or ``{"num_steps": 20_000}`` for ADVI).
    num_samples : int, default 500
        Number of posterior(-predictive) draws for in-sample prediction and
        forecasting.
    random_seed : int, optional
        Seed passed to fitting and every predictive call.

    Examples
    --------
    >>> import causalpy as cp
    >>> import pandas as pd
    >>> import pymc as pm
    >>> import pytensor.tensor as pt
    >>> import pymc_forecast  # doctest: +SKIP
    >>> def local_level(h, covariates):  # doctest: +SKIP
    ...     drift = pymc_forecast.time_series(
    ...         h, "drift", lambda name, dims: pm.Normal(name, 0, 0.1, dims=dims)
    ...     )
    ...     sigma = pm.HalfNormal("sigma", 1)
    ...     pymc_forecast.predict(
    ...         h,
    ...         lambda name, mu, dims, observed: pm.Normal(
    ...             name, mu, sigma, dims=dims, observed=observed
    ...         ),
    ...         pt.cumsum(drift),
    ...     )
    >>> result = cp.InterruptedTimeSeries(  # doctest: +SKIP
    ...     df,
    ...     treatment_time,
    ...     formula="y ~ 0",
    ...     model=cp.pymc_forecast_models.PyMCForecastModel(local_level),
    ... )
    """

    def __init__(
        self,
        model_fn: Any,
        forecaster: type | None = None,
        forecaster_kwargs: dict[str, Any] | None = None,
        num_samples: int = 500,
        random_seed: int | None = None,
    ) -> None:
        self._pf = _import_pymc_forecast()
        self.model_fn = model_fn
        self.forecaster_cls = (
            forecaster if forecaster is not None else self._pf.HMCForecaster
        )
        self.forecaster_kwargs = dict(forecaster_kwargs or {})
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.forecaster: Any = None
        self.idata: az.InferenceData | None = None
        self._treated_units: list[str] = ["unit_0"]
        self._has_covariates = False

    # -- fitting -----------------------------------------------------------

    def fit(
        self, X: xr.DataArray, y: xr.DataArray, coords: dict[str, Any] | None = None
    ) -> az.InferenceData:
        """Construct and fit the forecasting model on the pre-period.

        Parameters
        ----------
        X : xr.DataArray
            Design matrix with dims ``["obs_ind", "coeffs"]`` whose
            ``obs_ind`` coordinate carries the real (datetime or numeric)
            index. Columns are passed to ``pymc_forecast`` as covariates; a
            zero-column design (formula ``"y ~ 0"``) fits a covariate-free
            model.
        y : xr.DataArray
            Outcome with dims ``["obs_ind", "treated_units"]``. Must contain
            exactly one treated unit.
        coords : dict, optional
            Ignored; the real coordinates are read from ``X`` and ``y``.
        """
        if y.sizes["treated_units"] != 1:
            raise ValueError(
                "PyMCForecastModel supports a single treated unit, got "
                f"{y.sizes['treated_units']}."
            )
        self._treated_units = [str(u) for u in y.treated_units.values]
        data = y.isel(treated_units=0, drop=True).rename({"obs_ind": "time"})
        covariates = self._as_covariates(X)
        self._has_covariates = covariates is not None
        self.forecaster = self.forecaster_cls(
            self.model_fn,
            data,
            covariates,
            random_seed=self.random_seed,
            **self.forecaster_kwargs,
        )
        posterior = self.forecaster.draw_posterior(
            self.num_samples, random_seed=self.random_seed
        )
        self.idata = az.InferenceData(posterior=posterior)
        return self.idata

    @staticmethod
    def _as_covariates(X: xr.DataArray) -> xr.DataArray | None:
        """Map a patsy design matrix onto ``pymc_forecast`` covariates."""
        if X.sizes["coeffs"] == 0:
            return None
        return X.rename({"obs_ind": "time", "coeffs": "covariate"})

    # -- prediction --------------------------------------------------------

    def predict(
        self,
        X: xr.DataArray,
        coords: dict[str, Any] | None = None,
        out_of_sample: bool | None = False,
        **kwargs: Any,
    ) -> az.InferenceData:
        """Predict in-sample (pre-period) or forecast the counterfactual.

        Parameters
        ----------
        X : xr.DataArray
            Design matrix with dims ``["obs_ind", "coeffs"]``. In-sample
            prediction replays the training window (``X`` supplies only the
            output coordinates); out-of-sample prediction conditions the
            forecast on ``X``'s columns as future covariates (or, for a
            covariate-free model, forecasts over ``X``'s ``obs_ind`` index).
        coords : dict, optional
            Not used, kept for API compatibility.
        out_of_sample : bool, default False
            ``True`` draws the post-period counterfactual ("as if untreated").
        **kwargs
            Reserved for forward-compatibility; not consumed.

        Returns
        -------
        az.InferenceData
            With a ``posterior_predictive`` group holding draw-level ``mu``
            and ``y_hat`` variables with dims
            ``(chain, draw, obs_ind, treated_units)``.
        """
        if self.forecaster is None:
            raise RuntimeError("Model has not been fit yet.")
        if out_of_sample:
            if self._has_covariates:
                result = self.forecaster.forecast(
                    future_covariates=self._as_covariates(X),
                    num_samples=self.num_samples,
                    random_seed=self.random_seed,
                )
            else:
                result = self.forecaster.forecast(
                    future_index=X.obs_ind.values,
                    num_samples=self.num_samples,
                    random_seed=self.random_seed,
                )
            samples = self._pf.prediction_samples(result)[self._pf.FORECAST_VAR]
            samples = samples.rename({self._pf.FUTURE_DIM: "obs_ind"})
        else:
            result = self.forecaster.predict_in_sample(
                num_samples=self.num_samples, random_seed=self.random_seed
            )
            samples = self._pf.prediction_samples(result)[self._pf.OBS_VAR]
            samples = samples.rename({self._pf.TIME_DIM: "obs_ind"})
        return self._to_inference_data(samples, X.obs_ind.values)

    def _to_inference_data(
        self, samples: xr.DataArray, obs_ind: np.ndarray
    ) -> az.InferenceData:
        """Rename schema dims onto CausalPy coords and wrap as InferenceData."""
        if "series" in samples.dims:
            samples = samples.rename({"series": "treated_units"})
        else:
            samples = samples.expand_dims(treated_units=self._treated_units)
        samples = samples.assign_coords(obs_ind=obs_ind).transpose(
            "chain", "draw", "obs_ind", "treated_units"
        )
        # pymc_forecast exposes the draw-level posterior predictive of the
        # observed variable; there is no separate noise-free expectation, so
        # mu and y_hat are the same samples (see module docstring).
        ds = xr.Dataset({"mu": samples, "y_hat": samples})
        return az.InferenceData(posterior_predictive=ds)

    # -- scoring and impact ------------------------------------------------

    def score(
        self, X: xr.DataArray, y: xr.DataArray, coords: dict[str, Any] | None = None
    ) -> pd.Series:
        """Bayesian :math:`R^2` of the in-sample posterior predictive vs ``y``.

        Matches the ``PyMCModel.score`` output shape: one ``unit_{i}_r2`` /
        ``unit_{i}_r2_std`` pair per treated unit.

        Parameters
        ----------
        X : xr.DataArray
            Design matrix with dims ``["obs_ind", "coeffs"]``.
        y : xr.DataArray
            Observed outcomes with dims ``["obs_ind", "treated_units"]``.
        coords : dict, optional
            Not used, kept for API compatibility.
        """
        pred = self.predict(X)
        mu = az.extract(pred, group="posterior_predictive", var_names="mu")
        scores = {}
        for i, unit in enumerate(mu.coords["treated_units"].values):
            unit_mu = mu.sel(treated_units=unit).transpose("sample", "obs_ind")
            unit_y = y.sel(treated_units=unit).data
            unit_score = r2_score(unit_y, unit_mu.data)
            scores[f"unit_{i}_r2"] = unit_score["r2"]
            scores[f"unit_{i}_r2_std"] = unit_score["r2_std"]
        return pd.Series(scores)

    def calculate_impact(
        self, y_true: xr.DataArray, y_pred: az.InferenceData
    ) -> xr.DataArray:
        """Causal impact as observed minus counterfactual, at the draw level.

        Parameters
        ----------
        y_true : xr.DataArray
            Observed outcomes with dims ``["obs_ind", "treated_units"]``.
        y_pred : az.InferenceData
            Counterfactual prediction from :meth:`predict`.
        """
        y_hat = y_pred["posterior_predictive"]["mu"]
        y_hat = y_hat.assign_coords(obs_ind=y_true["obs_ind"])
        impact = y_true - y_hat
        return impact.transpose(..., "obs_ind")

    def calculate_cumulative_impact(self, impact: xr.DataArray) -> xr.DataArray:
        """Cumulative sum of pointwise causal impact along ``obs_ind``.

        Parameters
        ----------
        impact : xr.DataArray
            Pointwise causal impact, typically the output of
            :meth:`calculate_impact`.
        """
        return impact.cumsum(dim="obs_ind")

    def print_coefficients(
        self, labels: list[str], round_to: int | None = None
    ) -> None:
        """Print posterior means and HDIs of the model's scalar parameters.

        Forecasting-model parameters do not map onto the patsy design-matrix
        ``labels``, so those are ignored; every scalar variable in the fitted
        posterior is reported instead. Time-varying latents are skipped.

        Parameters
        ----------
        labels : list of str
            Design-matrix labels; ignored by forecasting models.
        round_to : int, optional
            Number of significant figures to round to. Defaults to None,
            in which case 2 significant figures are used.
        """
        if self.idata is None:
            raise RuntimeError("Model has not been fit yet.")
        posterior = self.idata.posterior
        scalar_vars = [
            name
            for name, da in posterior.data_vars.items()
            if set(da.dims) == {"chain", "draw"}
        ]
        print("Model parameters:")
        if not scalar_vars:
            print("  (no scalar parameters in posterior)")
            return
        max_label_length = max(len(name) for name in scalar_vars)
        for name in scalar_vars:
            samples = posterior[name]
            formatted_val = (
                f"{round_num(samples.mean().data, round_to)}, "
                f"{HDI_PROB * 100:.0f}% HDI "
                f"[{round_num(samples.quantile((1 - HDI_PROB) / 2).data, round_to)}, "
                f"{round_num(samples.quantile(1 - (1 - HDI_PROB) / 2).data, round_to)}]"
            )
            print(f"  {name: <{max_label_length}}  {formatted_val}")
