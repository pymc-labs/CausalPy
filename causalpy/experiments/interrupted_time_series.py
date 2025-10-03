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
"""
Interrupted Time Series Analysis
"""

from typing import List, Union

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from patsy import build_design_matrices, dmatrices
from sklearn.base import RegressorMixin

from causalpy.custom_exceptions import BadIndexException
from causalpy.plot_utils import get_hdi_to_df, plot_xY
from causalpy.pymc_models import (
    BayesianBasisExpansionTimeSeries,
    PyMCModel,
    StateSpaceTimeSeries,
)
from causalpy.utils import round_num

from .base import BaseExperiment

LEGEND_FONT_SIZE = 12


class InterruptedTimeSeries(BaseExperiment):
    """
    The class for interrupted time series analysis.

    :param data:
        A pandas dataframe
    :param treatment_time:
        The time when treatment occurred, should be in reference to the data index
    :param formula:
        A statistical model formula
    :param model:
        A PyMC model

    Example
    --------
    >>> import causalpy as cp
    >>> df = (
    ...     cp.load_data("its")
    ...     .assign(date=lambda x: pd.to_datetime(x["date"]))
    ...     .set_index("date")
    ... )
    >>> treatment_time = pd.to_datetime("2017-01-01")
    >>> seed = 42
    >>> result = cp.InterruptedTimeSeries(
    ...     df,
    ...     treatment_time,
    ...     formula="y ~ 1 + t + C(month)",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={
    ...             "target_accept": 0.95,
    ...             "random_seed": seed,
    ...             "progressbar": False,
    ...         }
    ...     ),
    ... )
    """

    expt_type = "Interrupted Time Series"
    supports_ols = True
    supports_bayes = True

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_time: Union[int, float, pd.Timestamp],
        formula: str,
        model=None,
        **kwargs,
    ) -> None:
        super().__init__(model=model)
        # rename the index to "obs_ind"
        data.index.name = "obs_ind"
        self.input_validation(data, treatment_time)
        self.treatment_time = treatment_time
        # set experiment type - usually done in subclasses
        self.expt_type = "Pre-Post Fit"
        # split data in to pre and post intervention
        self.datapre = data[data.index < self.treatment_time]
        self.datapost = data[data.index >= self.treatment_time]

        self.formula = formula

        # set things up with pre-intervention data
        y, X = dmatrices(formula, self.datapre)
        self.outcome_variable_name = y.design_info.column_names[0]
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.pre_y, self.pre_X = np.asarray(y), np.asarray(X)
        # process post-intervention data
        (new_y, new_x) = build_design_matrices(
            [self._y_design_info, self._x_design_info], self.datapost
        )
        self.post_X = np.asarray(new_x)
        self.post_y = np.asarray(new_y)
        # turn into xarray.DataArray's
        self.pre_X = xr.DataArray(
            self.pre_X,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": self.datapre.index,
                "coeffs": self.labels,
            },
        )
        self.pre_y = xr.DataArray(
            self.pre_y,  # Keep 2D shape
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": self.datapre.index, "treated_units": ["unit_0"]},
        )
        self.post_X = xr.DataArray(
            self.post_X,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": self.datapost.index,
                "coeffs": self.labels,
            },
        )
        self.post_y = xr.DataArray(
            self.post_y,  # Keep 2D shape
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": self.datapost.index, "treated_units": ["unit_0"]},
        )

        # fit the model to the observed (pre-intervention) data
        if isinstance(self.model, PyMCModel):
            is_bsts_like = isinstance(
                self.model, (BayesianBasisExpansionTimeSeries, StateSpaceTimeSeries)
            )

            if is_bsts_like:
                # BSTS/StateSpace models expect numpy arrays and datetime coords
                X_fit = self.pre_X.values if self.pre_X.shape[1] > 0 else None
                y_fit = self.pre_y.isel(treated_units=0).values
                pre_coords = {"datetime_index": self.datapre.index}
                if X_fit is not None:
                    pre_coords["coeffs"] = self.labels
                self.model.fit(X=X_fit, y=y_fit, coords=pre_coords)
            else:
                # General PyMC models expect xarray with treated_units
                COORDS = {
                    "coeffs": self.labels,
                    "obs_ind": np.arange(self.pre_X.shape[0]),
                    "treated_units": ["unit_0"],
                }
                self.model.fit(X=self.pre_X, y=self.pre_y, coords=COORDS)
        elif isinstance(self.model, RegressorMixin):
            # For OLS models, use 1D y data
            self.model.fit(X=self.pre_X, y=self.pre_y.isel(treated_units=0))
        else:
            raise ValueError("Model type not recognized")

        # score the goodness of fit to the pre-intervention data
        if isinstance(self.model, PyMCModel):
            is_bsts_like = isinstance(
                self.model, (BayesianBasisExpansionTimeSeries, StateSpaceTimeSeries)
            )
            if is_bsts_like:
                X_score = self.pre_X.values if self.pre_X.shape[1] > 0 else None
                y_score = self.pre_y.isel(treated_units=0).values
                score_coords = {"datetime_index": self.datapre.index}
                if X_score is not None:
                    score_coords["coeffs"] = self.labels
                self.score = self.model.score(X=X_score, y=y_score, coords=score_coords)
            else:
                self.score = self.model.score(X=self.pre_X, y=self.pre_y)
        elif isinstance(self.model, RegressorMixin):
            self.score = self.model.score(
                X=self.pre_X, y=self.pre_y.isel(treated_units=0)
            )

        # get the model predictions of the observed (pre-intervention) data
        if isinstance(self.model, PyMCModel):
            is_bsts_like = isinstance(
                self.model, (BayesianBasisExpansionTimeSeries, StateSpaceTimeSeries)
            )
            if is_bsts_like:
                X_pre_predict = self.pre_X.values if self.pre_X.shape[1] > 0 else None
                pre_pred_coords = {"datetime_index": self.datapre.index}
                self.pre_pred = self.model.predict(
                    X=X_pre_predict, coords=pre_pred_coords
                )
                if not isinstance(self.pre_pred, az.InferenceData):
                    self.pre_pred = az.InferenceData(posterior_predictive=self.pre_pred)
            else:
                self.pre_pred = self.model.predict(X=self.pre_X)
        elif isinstance(self.model, RegressorMixin):
            self.pre_pred = self.model.predict(X=self.pre_X)

        # calculate the counterfactual (post period)
        if isinstance(self.model, PyMCModel):
            is_bsts_like = isinstance(
                self.model, (BayesianBasisExpansionTimeSeries, StateSpaceTimeSeries)
            )
            if is_bsts_like:
                X_post_predict = (
                    self.post_X.values if self.post_X.shape[1] > 0 else None
                )
                post_pred_coords = {"datetime_index": self.datapost.index}
                self.post_pred = self.model.predict(
                    X=X_post_predict, coords=post_pred_coords, out_of_sample=True
                )
                if not isinstance(self.post_pred, az.InferenceData):
                    self.post_pred = az.InferenceData(
                        posterior_predictive=self.post_pred
                    )
            else:
                self.post_pred = self.model.predict(X=self.post_X)
        elif isinstance(self.model, RegressorMixin):
            self.post_pred = self.model.predict(X=self.post_X)

        # calculate impact - use appropriate y data format for each model type
        if isinstance(self.model, PyMCModel):
            is_bsts_like = isinstance(
                self.model, (BayesianBasisExpansionTimeSeries, StateSpaceTimeSeries)
            )
            if is_bsts_like:
                pre_y_for_impact = self.pre_y.isel(treated_units=0)
                post_y_for_impact = self.post_y.isel(treated_units=0)
                self.pre_impact = self.model.calculate_impact(
                    pre_y_for_impact, self.pre_pred
                )
                self.post_impact = self.model.calculate_impact(
                    post_y_for_impact, self.post_pred
                )
            else:
                # PyMC models with treated_units use 2D data
                self.pre_impact = self.model.calculate_impact(self.pre_y, self.pre_pred)
                self.post_impact = self.model.calculate_impact(
                    self.post_y, self.post_pred
                )
        elif isinstance(self.model, RegressorMixin):
            # SKL models work with 1D data
            self.pre_impact = self.model.calculate_impact(
                self.pre_y.isel(treated_units=0), self.pre_pred
            )
            self.post_impact = self.model.calculate_impact(
                self.post_y.isel(treated_units=0), self.post_pred
            )

        self.post_impact_cumulative = self.model.calculate_cumulative_impact(
            self.post_impact
        )

    def input_validation(self, data, treatment_time):
        """Validate the input data and model formula for correctness"""
        if isinstance(data.index, pd.DatetimeIndex) and not isinstance(
            treatment_time, pd.Timestamp
        ):
            raise BadIndexException(
                "If data.index is DatetimeIndex, treatment_time must be pd.Timestamp."
            )
        if not isinstance(data.index, pd.DatetimeIndex) and isinstance(
            treatment_time, pd.Timestamp
        ):
            raise BadIndexException(
                "If data.index is not DatetimeIndex, treatment_time must be pd.Timestamp."  # noqa: E501
            )

    def summary(self, round_to=None) -> None:
        """Print summary of main results and model coefficients.

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers
        """
        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        self.print_coefficients(round_to)

    def _bayesian_plot(
        self, round_to=None, **kwargs
    ) -> tuple[plt.Figure, List[plt.Axes]]:
        """
        Plot the results

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """
        counterfactual_label = "Counterfactual"

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))
        # TOP PLOT --------------------------------------------------
        # pre-intervention period
        pre_mu = self.pre_pred["posterior_predictive"].mu
        pre_mu_plot = (
            pre_mu.isel(treated_units=0) if "treated_units" in pre_mu.dims else pre_mu
        )
        h_line, h_patch = plot_xY(
            self.datapre.index,
            pre_mu_plot,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C0"},
        )
        handles = [(h_line, h_patch)]
        labels = ["Pre-intervention period"]

        (h,) = ax[0].plot(
            self.datapre.index,
            self.pre_y.isel(treated_units=0)
            if hasattr(self.pre_y, "isel")
            else self.pre_y[:, 0],
            "k.",
            label="Observations",
        )
        handles.append(h)
        labels.append("Observations")

        # post intervention period
        post_mu = self.post_pred["posterior_predictive"].mu
        post_mu_plot = (
            post_mu.isel(treated_units=0)
            if "treated_units" in post_mu.dims
            else post_mu
        )
        h_line, h_patch = plot_xY(
            self.datapost.index,
            post_mu_plot,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C1"},
        )
        handles.append((h_line, h_patch))
        labels.append(counterfactual_label)

        ax[0].plot(
            self.datapost.index,
            self.post_y.isel(treated_units=0)
            if hasattr(self.post_y, "isel")
            else self.post_y[:, 0],
            "k.",
        )
        # Shaded causal effect
        post_pred_mu = az.extract(
            self.post_pred, group="posterior_predictive", var_names="mu"
        )
        if "treated_units" in post_pred_mu.dims:
            post_pred_mu = post_pred_mu.isel(treated_units=0)
        post_pred_mu = post_pred_mu.mean("sample")
        h = ax[0].fill_between(
            self.datapost.index,
            y1=post_pred_mu,
            y2=self.post_y.isel(treated_units=0)
            if hasattr(self.post_y, "isel")
            else self.post_y[:, 0],
            color="C0",
            alpha=0.25,
        )
        handles.append(h)
        labels.append("Causal impact")

        # Title with R^2, supporting both unit_0_r2 and r2 keys
        r2_val = None
        r2_std_val = None
        try:
            if isinstance(self.score, pd.Series):
                if "unit_0_r2" in self.score.index:
                    r2_val = self.score["unit_0_r2"]
                    r2_std_val = self.score.get("unit_0_r2_std", None)
                elif "r2" in self.score.index:
                    r2_val = self.score["r2"]
                    r2_std_val = self.score.get("r2_std", None)
        except Exception:
            pass
        title_str = "Pre-intervention Bayesian $R^2$"
        if r2_val is not None:
            title_str += f": {round_num(r2_val, round_to)}"
            if r2_std_val is not None:
                title_str += f"\n(std = {round_num(r2_std_val, round_to)})"
        ax[0].set(title=title_str)

        # MIDDLE PLOT -----------------------------------------------
        pre_impact_plot = (
            self.pre_impact.isel(treated_units=0)
            if hasattr(self.pre_impact, "dims")
            and "treated_units" in self.pre_impact.dims
            else self.pre_impact
        )
        plot_xY(
            self.datapre.index,
            pre_impact_plot,
            ax=ax[1],
            plot_hdi_kwargs={"color": "C0"},
        )
        post_impact_plot = (
            self.post_impact.isel(treated_units=0)
            if hasattr(self.post_impact, "dims")
            and "treated_units" in self.post_impact.dims
            else self.post_impact
        )
        plot_xY(
            self.datapost.index,
            post_impact_plot,
            ax=ax[1],
            plot_hdi_kwargs={"color": "C1"},
        )
        ax[1].axhline(y=0, c="k")
        post_impact_mean = (
            self.post_impact.mean(["chain", "draw"])
            if hasattr(self.post_impact, "mean")
            else self.post_impact
        )
        if (
            hasattr(post_impact_mean, "dims")
            and "treated_units" in post_impact_mean.dims
        ):
            post_impact_mean = post_impact_mean.isel(treated_units=0)
        ax[1].fill_between(
            self.datapost.index,
            y1=post_impact_mean,
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[1].set(title="Causal Impact")

        # BOTTOM PLOT -----------------------------------------------
        ax[2].set(title="Cumulative Causal Impact")
        post_cum_plot = (
            self.post_impact_cumulative.isel(treated_units=0)
            if hasattr(self.post_impact_cumulative, "dims")
            and "treated_units" in self.post_impact_cumulative.dims
            else self.post_impact_cumulative
        )
        plot_xY(
            self.datapost.index,
            post_cum_plot,
            ax=ax[2],
            plot_hdi_kwargs={"color": "C1"},
        )
        ax[2].axhline(y=0, c="k")

        # Intervention line
        for i in [0, 1, 2]:
            ax[i].axvline(
                x=self.treatment_time,
                ls="-",
                lw=3,
                color="r",
            )

        ax[0].legend(
            handles=(h_tuple for h_tuple in handles),
            labels=labels,
            fontsize=LEGEND_FONT_SIZE,
        )

        return fig, ax

    def _ols_plot(self, round_to=None, **kwargs) -> tuple[plt.Figure, List[plt.Axes]]:
        """
        Plot the results

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """
        counterfactual_label = "Counterfactual"

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

        ax[0].plot(self.datapre.index, self.pre_y, "k.")
        ax[0].plot(self.datapost.index, self.post_y, "k.")

        ax[0].plot(self.datapre.index, self.pre_pred, c="k", label="model fit")
        ax[0].plot(
            self.datapost.index,
            self.post_pred,
            label=counterfactual_label,
            ls=":",
            c="k",
        )
        ax[0].set(
            title=f"$R^2$ on pre-intervention data = {round_num(self.score, round_to)}"
        )

        ax[1].plot(self.datapre.index, self.pre_impact, "k.")
        ax[1].plot(
            self.datapost.index,
            self.post_impact,
            "k.",
            label=counterfactual_label,
        )
        ax[1].axhline(y=0, c="k")
        ax[1].set(title="Causal Impact")

        ax[2].plot(self.datapost.index, self.post_impact_cumulative, c="k")
        ax[2].axhline(y=0, c="k")
        ax[2].set(title="Cumulative Causal Impact")

        # Shaded causal effect
        ax[0].fill_between(
            self.datapost.index,
            y1=np.squeeze(self.post_pred),
            y2=np.squeeze(self.post_y),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[1].fill_between(
            self.datapost.index,
            y1=np.squeeze(self.post_impact),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )

        # Intervention line
        # TODO: make this work when treatment_time is a datetime
        for i in [0, 1, 2]:
            ax[i].axvline(
                x=self.treatment_time,
                ls="-",
                lw=3,
                color="r",
                label="Treatment time",
            )

        ax[0].legend(fontsize=LEGEND_FONT_SIZE)

        return (fig, ax)

    def get_plot_data_bayesian(self, hdi_prob: float = 0.94) -> pd.DataFrame:
        """
        Recover the data of the experiment along with the prediction and causal impact information.

        :param hdi_prob:
            Prob for which the highest density interval will be computed. The default value is defined as the default from the :func:`arviz.hdi` function.
        """
        if isinstance(self.model, PyMCModel):
            hdi_pct = int(round(hdi_prob * 100))

            pred_lower_col = f"pred_hdi_lower_{hdi_pct}"
            pred_upper_col = f"pred_hdi_upper_{hdi_pct}"
            impact_lower_col = f"impact_hdi_lower_{hdi_pct}"
            impact_upper_col = f"impact_hdi_upper_{hdi_pct}"

            pre_data = self.datapre.copy()
            post_data = self.datapost.copy()

            pre_mu = az.extract(
                self.pre_pred, group="posterior_predictive", var_names="mu"
            )
            post_mu = az.extract(
                self.post_pred, group="posterior_predictive", var_names="mu"
            )
            if "treated_units" in pre_mu.dims:
                pre_mu = pre_mu.isel(treated_units=0)
            if "treated_units" in post_mu.dims:
                post_mu = post_mu.isel(treated_units=0)
            pre_data["prediction"] = pre_mu.mean("sample").values
            post_data["prediction"] = post_mu.mean("sample").values

            hdi_pre_pred = get_hdi_to_df(
                self.pre_pred["posterior_predictive"].mu, hdi_prob=hdi_prob
            )
            hdi_post_pred = get_hdi_to_df(
                self.post_pred["posterior_predictive"].mu, hdi_prob=hdi_prob
            )
            # If treated_units present, select unit_0; otherwise use directly
            if (
                isinstance(hdi_pre_pred.index, pd.MultiIndex)
                and "treated_units" in hdi_pre_pred.index.names
            ):
                pre_data[[pred_lower_col, pred_upper_col]] = hdi_pre_pred.xs(
                    "unit_0", level="treated_units"
                ).set_index(pre_data.index)
                post_data[[pred_lower_col, pred_upper_col]] = hdi_post_pred.xs(
                    "unit_0", level="treated_units"
                ).set_index(post_data.index)
            else:
                pre_data[[pred_lower_col, pred_upper_col]] = hdi_pre_pred.set_index(
                    pre_data.index
                )
                post_data[[pred_lower_col, pred_upper_col]] = hdi_post_pred.set_index(
                    post_data.index
                )

            pre_impact_mean = (
                self.pre_impact.mean(dim=["chain", "draw"])
                if hasattr(self.pre_impact, "mean")
                else self.pre_impact
            )
            post_impact_mean = (
                self.post_impact.mean(dim=["chain", "draw"])
                if hasattr(self.post_impact, "mean")
                else self.post_impact
            )
            if (
                hasattr(pre_impact_mean, "dims")
                and "treated_units" in pre_impact_mean.dims
            ):
                pre_impact_mean = pre_impact_mean.isel(treated_units=0)
            if (
                hasattr(post_impact_mean, "dims")
                and "treated_units" in post_impact_mean.dims
            ):
                post_impact_mean = post_impact_mean.isel(treated_units=0)
            pre_data["impact"] = pre_impact_mean.values
            post_data["impact"] = post_impact_mean.values

            # Compute impact HDIs directly via quantiles over posterior dims to avoid column shape issues
            alpha = 1 - hdi_prob
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2

            pre_lower_da = self.pre_impact.quantile(lower_q, dim=["chain", "draw"])
            pre_upper_da = self.pre_impact.quantile(upper_q, dim=["chain", "draw"])
            post_lower_da = self.post_impact.quantile(lower_q, dim=["chain", "draw"])
            post_upper_da = self.post_impact.quantile(upper_q, dim=["chain", "draw"])

            # If a treated_units dim remains for some models, select unit_0
            if hasattr(pre_lower_da, "dims") and "treated_units" in pre_lower_da.dims:
                pre_lower_da = pre_lower_da.sel(treated_units="unit_0")
                pre_upper_da = pre_upper_da.sel(treated_units="unit_0")
            if hasattr(post_lower_da, "dims") and "treated_units" in post_lower_da.dims:
                post_lower_da = post_lower_da.sel(treated_units="unit_0")
                post_upper_da = post_upper_da.sel(treated_units="unit_0")

            pre_data[impact_lower_col] = (
                pre_lower_da.to_series().reindex(pre_data.index).values
            )
            pre_data[impact_upper_col] = (
                pre_upper_da.to_series().reindex(pre_data.index).values
            )
            post_data[impact_lower_col] = (
                post_lower_da.to_series().reindex(post_data.index).values
            )
            post_data[impact_upper_col] = (
                post_upper_da.to_series().reindex(post_data.index).values
            )

            self.plot_data = pd.concat([pre_data, post_data])

            return self.plot_data
        else:
            raise ValueError("Unsupported model type")

    def get_plot_data_ols(self) -> pd.DataFrame:
        """
        Recover the data of the experiment along with the prediction and causal impact information.
        """
        pre_data = self.datapre.copy()
        post_data = self.datapost.copy()
        pre_data["prediction"] = self.pre_pred
        post_data["prediction"] = self.post_pred
        pre_data["impact"] = self.pre_impact
        post_data["impact"] = self.post_impact
        self.plot_data = pd.concat([pre_data, post_data])

        return self.plot_data
