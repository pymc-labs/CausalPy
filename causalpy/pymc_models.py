#   Copyright 2024 The PyMC Labs Developers
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
import pandas as pd
import pymc as pm
import xarray as xr
from arviz import r2_score

from causalpy.plotting import BayesianPlotComponent, PlotComponent
from causalpy.utils import round_num


class BayesianModel(pm.Model):
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
        """
        with self:
            pm.set_data({"X": X})

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

    def calculate_impact(self, y_true, y_pred):
        pre_data = xr.DataArray(y_true, dims=["obs_ind"])
        impact = pre_data - y_pred["posterior_predictive"]["y_hat"]
        return impact.transpose(..., "obs_ind")

    def calculate_cumulative_impact(self, impact):
        return impact.cumsum(dim="obs_ind")

    def get_plot_component(self) -> PlotComponent:
        return BayesianPlotComponent()

    def print_coefficients(self, labels, round_to=None) -> None:
        def print_row(
            max_label_length: int, name: str, coeff_samples: xr.DataArray, round_to: int
        ) -> None:
            """Print one row of the coefficient table"""
            formatted_name = f"  {name: <{max_label_length}}"
            formatted_val = f"{round_num(coeff_samples.mean().data, round_to)}, 94% HDI [{round_num(coeff_samples.quantile(0.03).data, round_to)}, {round_num(coeff_samples.quantile(1-0.03).data, round_to)}]"  # noqa: E501
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


class LinearRegression(BayesianModel):
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
