#   Copyright 2025 - 2025 The PyMC Labs Developers
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
Synthetic Control Experiment
"""

from typing import List, Union

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.base import RegressorMixin

from causalpy.custom_exceptions import BadIndexException
from causalpy.plot_utils import get_hdi_to_df, plot_xY
from causalpy.pymc_models import PyMCModel
from causalpy.utils import round_num

from .base import BaseExperiment

LEGEND_FONT_SIZE = 12


class SyntheticControl(BaseExperiment):
    """The class for the synthetic control experiment.

    :param data:
        A pandas dataframe
    :param treatment_time:
        The time when treatment occurred, should be in reference to the data index
    :param control_units:
        A list of control units to be used in the experiment
    :param treated_units:
        A list of treated units to be used in the experiment
    :param model:
        A PyMC model

    Example
    --------
    >>> import causalpy as cp
    >>> df = cp.load_data("sc")
    >>> treatment_time = 70
    >>> seed = 42
    >>> result = cp.SyntheticControl(
    ...     df,
    ...     treatment_time,
    ...     control_units=["a", "b", "c", "d", "e", "f", "g"],
    ...     treated_units=["actual"],
    ...     model=cp.pymc_models.WeightedSumFitter(
    ...         sample_kwargs={
    ...             "target_accept": 0.95,
    ...             "random_seed": seed,
    ...             "progressbar": False,
    ...         }
    ...     ),
    ... )

    Notes
    -----
    For Bayesian models, the causal impact is calculated using the posterior expectation
    (``mu``) rather than the posterior predictive (``y_hat``). This means the impact and
    its uncertainty represent the systematic causal effect, excluding observation-level
    noise. The uncertainty bands in the plots reflect parameter uncertainty and
    counterfactual prediction uncertainty, but not individual observation variability.
    """

    supports_ols = True
    supports_bayes = True

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_time: Union[int, float, pd.Timestamp],
        control_units: list[str],
        treated_units: list[str],
        model: Union[PyMCModel, RegressorMixin] | None = None,
        **kwargs: dict,
    ) -> None:
        super().__init__(model=model)
        # rename the index to "obs_ind"
        data.index.name = "obs_ind"
        self.input_validation(data, treatment_time)
        self.treatment_time = treatment_time
        self.control_units = control_units
        self.labels = control_units
        self.treated_units = treated_units
        self.expt_type = "SyntheticControl"
        # split data in to pre and post intervention
        self.datapre = data[data.index < self.treatment_time]
        self.datapost = data[data.index >= self.treatment_time]

        # split data into the 4 quadrants (pre/post, control/treated) and store as
        # xarray.DataArray objects.
        # NOTE: if we have renamed/ensured the index is named "obs_ind", then it will
        # make constructing the xarray DataArray objects easier.
        self.datapre_control = xr.DataArray(
            self.datapre[self.control_units],
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": self.datapre[self.control_units].index,
                "coeffs": self.control_units,
            },
        )
        self.datapre_treated = xr.DataArray(
            self.datapre[self.treated_units],
            dims=["obs_ind", "treated_units"],
            coords={
                "obs_ind": self.datapre[self.treated_units].index,
                "treated_units": self.treated_units,
            },
        )
        self.datapost_control = xr.DataArray(
            self.datapost[self.control_units],
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": self.datapost[self.control_units].index,
                "coeffs": self.control_units,
            },
        )
        self.datapost_treated = xr.DataArray(
            self.datapost[self.treated_units],
            dims=["obs_ind", "treated_units"],
            coords={
                "obs_ind": self.datapost[self.treated_units].index,
                "treated_units": self.treated_units,
            },
        )

        # fit the model to the observed (pre-intervention) data
        if isinstance(self.model, PyMCModel):
            COORDS = {
                # key must stay as "coeffs" unless we can find a way to auto identify
                # the predictor dimension name. "coeffs" is assumed by
                # PyMCModel.print_coefficients for example.
                "coeffs": self.control_units,
                "treated_units": self.treated_units,
                "obs_ind": np.arange(self.datapre.shape[0]),
            }
            self.model.fit(
                X=self.datapre_control,
                y=self.datapre_treated,
                coords=COORDS,
            )
        elif isinstance(self.model, RegressorMixin):
            self.model.fit(
                X=self.datapre_control.data,
                y=self.datapre_treated.isel(treated_units=0).data,
            )
        else:
            raise ValueError("Model type not recognized")

        # score the goodness of fit to the pre-intervention data
        self.score = self.model.score(
            X=self.datapre_control,
            y=self.datapre_treated,
        )

        # get the model predictions of the observed (pre-intervention) data
        self.pre_pred = self.model.predict(X=self.datapre_control)

        # calculate the counterfactual
        self.post_pred = self.model.predict(X=self.datapost_control)
        self.pre_impact = self.model.calculate_impact(
            self.datapre_treated, self.pre_pred
        )

        self.post_impact = self.model.calculate_impact(
            self.datapost_treated, self.post_pred
        )

        self.post_impact_cumulative = self.model.calculate_cumulative_impact(
            self.post_impact
        )

    def input_validation(
        self, data: pd.DataFrame, treatment_time: Union[int, float, pd.Timestamp]
    ) -> None:
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

    def summary(self, round_to: int | None = None) -> None:
        """Print summary of main results and model coefficients.

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers
        """
        print(f"{self.expt_type:=^80}")
        print(f"Control units: {self.control_units}")
        if len(self.treated_units) > 1:
            print(f"Treated units: {self.treated_units}")
        else:
            print(f"Treated unit: {self.treated_units[0]}")
        self.print_coefficients(round_to)

    def _bayesian_plot(
        self,
        round_to: int | None = None,
        treated_unit: str | None = None,
        **kwargs: dict,
    ) -> tuple[plt.Figure, List[plt.Axes]]:
        """
        Plot the results for a specific treated unit

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        :param treated_unit:
            Which treated unit to plot. Must be a string name of the treated unit.
            If None, plots the first treated unit.
        """
        counterfactual_label = "Counterfactual"

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))
        # TOP PLOT --------------------------------------------------
        # pre-intervention period

        # Get treated unit name - default to first unit if None
        treated_unit = (
            treated_unit if treated_unit is not None else self.treated_units[0]
        )

        if treated_unit not in self.treated_units:
            raise ValueError(
                f"treated_unit '{treated_unit}' not found. Available units: {self.treated_units}"
            )

        pre_pred = self.pre_pred["posterior_predictive"].mu.sel(
            treated_units=treated_unit
        )
        post_pred = self.post_pred["posterior_predictive"].mu.sel(
            treated_units=treated_unit
        )

        h_line, h_patch = plot_xY(
            self.datapre.index,
            pre_pred,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C0"},
        )
        handles = [(h_line, h_patch)]
        labels = ["Pre-intervention period"]

        # Plot observations for primary treated unit
        (h,) = ax[0].plot(
            self.datapre.index,
            self.datapre_treated.sel(treated_units=treated_unit),
            "k.",
            label="Observations",
        )
        handles.append(h)
        labels.append("Observations")

        # post intervention period
        h_line, h_patch = plot_xY(
            self.datapost.index,
            post_pred,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C1"},
        )
        handles.append((h_line, h_patch))
        labels.append(counterfactual_label)

        ax[0].plot(
            self.datapost.index,
            self.datapost_treated.sel(treated_units=treated_unit),
            "k.",
        )
        # Shaded causal effect for primary treated unit
        h = ax[0].fill_between(
            self.datapost.index,
            y1=post_pred.mean(dim=["chain", "draw"]).values,
            y2=self.datapost_treated.sel(treated_units=treated_unit).values,
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        handles.append(h)
        labels.append("Causal impact")

        ax[0].set(title=f"{self._get_score_title(treated_unit, round_to)}")

        # MIDDLE PLOT -----------------------------------------------
        plot_xY(
            self.datapre.index,
            self.pre_impact.sel(treated_units=treated_unit),
            ax=ax[1],
            plot_hdi_kwargs={"color": "C0"},
        )
        plot_xY(
            self.datapost.index,
            self.post_impact.sel(treated_units=treated_unit),
            ax=ax[1],
            plot_hdi_kwargs={"color": "C1"},
        )
        ax[1].axhline(y=0, c="k")
        ax[1].fill_between(
            self.datapost.index,
            y1=self.post_impact.mean(["chain", "draw"]).sel(treated_units=treated_unit),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[1].set(title="Causal Impact")

        # BOTTOM PLOT -----------------------------------------------
        ax[2].set(title="Cumulative Causal Impact")
        plot_xY(
            self.datapost.index,
            self.post_impact_cumulative.sel(treated_units=treated_unit),
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

        plot_predictors = kwargs.get("plot_predictors", False)
        if plot_predictors:
            # plot control units as well
            ax[0].plot(
                self.datapre.index,
                self.datapre_control,
                "-",
                c=[0.8, 0.8, 0.8],
                zorder=1,
            )
            ax[0].plot(
                self.datapost.index,
                self.datapost_control,
                "-",
                c=[0.8, 0.8, 0.8],
                zorder=1,
            )

        return fig, ax

    def _ols_plot(
        self,
        round_to: int | None = None,
        treated_unit: str | None = None,
        **kwargs: dict,
    ) -> tuple[plt.Figure, List[plt.Axes]]:
        """
        Plot the results for OLS model for a specific treated unit

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        :param treated_unit:
            Which treated unit to plot. Must be a string name of the treated unit.
            If None, plots the first treated unit.
        """
        counterfactual_label = "Counterfactual"

        # Get treated unit name - default to first unit if None
        treated_unit = (
            treated_unit if treated_unit is not None else self.treated_units[0]
        )

        if treated_unit not in self.treated_units:
            raise ValueError(
                f"treated_unit '{treated_unit}' not found. Available units: {self.treated_units}"
            )

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

        ax[0].plot(
            self.datapre_treated["obs_ind"],
            self.datapre_treated.sel(treated_units=treated_unit),
            "k.",
        )
        ax[0].plot(
            self.datapost_treated["obs_ind"],
            self.datapost_treated.sel(treated_units=treated_unit),
            "k.",
        )

        ax[0].plot(self.datapre.index, self.pre_pred, c="k", label="model fit")
        ax[0].plot(
            self.datapost.index,
            self.post_pred,
            label=counterfactual_label,
            ls=":",
            c="k",
        )
        ax[0].set(title=f"{self._get_score_title(treated_unit, round_to)}")
        # Shaded causal effect
        post_pred_values = np.squeeze(self.post_pred)

        ax[0].fill_between(
            self.datapost.index,
            y1=post_pred_values,
            y2=np.squeeze(self.datapost_treated.sel(treated_units=treated_unit).data),
            color="C0",
            alpha=0.25,
            label="Causal impact",
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

    def get_plot_data_bayesian(
        self, hdi_prob: float = 0.94, treated_unit: str | None = None
    ) -> pd.DataFrame:
        """
        Recover the data of the PrePostFit experiment along with the prediction and causal impact information.

        :param hdi_prob:
            Prob for which the highest density interval will be computed. The default value is defined as the default from the :func:`arviz.hdi` function.
        :param treated_unit:
            Which treated unit to extract data for. Must be a string name of the treated unit.
            If None, uses the first treated unit.
        """
        if not isinstance(self.model, PyMCModel):
            raise ValueError("Unsupported model type")

        hdi_pct = int(round(hdi_prob * 100))

        pred_lower_col = f"pred_hdi_lower_{hdi_pct}"
        pred_upper_col = f"pred_hdi_upper_{hdi_pct}"
        impact_lower_col = f"impact_hdi_lower_{hdi_pct}"
        impact_upper_col = f"impact_hdi_upper_{hdi_pct}"

        pre_data = self.datapre.copy()
        post_data = self.datapost.copy()

        # Get treated unit name - default to first unit if None
        treated_unit = (
            treated_unit if treated_unit is not None else self.treated_units[0]
        )

        if treated_unit not in self.treated_units:
            raise ValueError(
                f"treated_unit '{treated_unit}' not found. Available units: {self.treated_units}"
            )

        # Extract predictions - handle multi-unit case
        pre_pred_vals = az.extract(
            self.pre_pred, group="posterior_predictive", var_names="mu"
        ).mean("sample")
        post_pred_vals = az.extract(
            self.post_pred, group="posterior_predictive", var_names="mu"
        ).mean("sample")

        # Extract predictions for the specified treated unit (always has treated_units dimension)
        pre_data["prediction"] = pre_pred_vals.sel(treated_units=treated_unit).values
        post_data["prediction"] = post_pred_vals.sel(treated_units=treated_unit).values

        # HDI intervals for predictions (always use treated_units dimension)
        pre_hdi = get_hdi_to_df(
            self.pre_pred["posterior_predictive"].mu.sel(treated_units=treated_unit),
            hdi_prob=hdi_prob,
        )
        post_hdi = get_hdi_to_df(
            self.post_pred["posterior_predictive"].mu.sel(treated_units=treated_unit),
            hdi_prob=hdi_prob,
        )

        # Extract only the lower and upper columns and ensure proper indexing
        pre_lower_upper = pre_hdi.iloc[:, [0, -1]].values  # Get first and last columns
        post_lower_upper = post_hdi.iloc[:, [0, -1]].values

        pre_data[[pred_lower_col, pred_upper_col]] = pre_lower_upper
        post_data[[pred_lower_col, pred_upper_col]] = post_lower_upper

        # Impact data - always use primary unit for main dataframe
        pre_data["impact"] = (
            self.pre_impact.mean(dim=["chain", "draw"])
            .sel(treated_units=treated_unit)
            .values
        )
        post_data["impact"] = (
            self.post_impact.mean(dim=["chain", "draw"])
            .sel(treated_units=treated_unit)
            .values
        )
        # Impact HDI intervals (always use treated_units dimension)
        pre_impact_hdi = get_hdi_to_df(
            self.pre_impact.sel(treated_units=treated_unit), hdi_prob=hdi_prob
        )
        post_impact_hdi = get_hdi_to_df(
            self.post_impact.sel(treated_units=treated_unit), hdi_prob=hdi_prob
        )

        # Extract only the lower and upper columns for impact HDI
        pre_impact_lower_upper = pre_impact_hdi.iloc[:, [0, -1]].values
        post_impact_lower_upper = post_impact_hdi.iloc[:, [0, -1]].values

        pre_data[[impact_lower_col, impact_upper_col]] = pre_impact_lower_upper
        post_data[[impact_lower_col, impact_upper_col]] = post_impact_lower_upper

        self.plot_data = pd.concat([pre_data, post_data])

        return self.plot_data

    def _get_score_title(self, treated_unit: str, round_to: int | None = 2) -> str:
        """Generate appropriate score title for the specified treated unit"""
        if isinstance(self.model, PyMCModel):
            # Bayesian model - get unit-specific R² scores using unified format
            unit_index = self.treated_units.index(treated_unit)
            r2_val = round_num(
                self.score[f"unit_{unit_index}_r2"],
                round_to if round_to is not None else 2,
            )
            r2_std_val = round_num(
                self.score[f"unit_{unit_index}_r2_std"],
                round_to if round_to is not None else 2,
            )
            return f"Pre-intervention Bayesian $R^2$: {r2_val} (std = {r2_std_val})"
        else:
            # OLS model - simple float score
            return f"$R^2$ on pre-intervention data = {round_num(float(self.score), round_to if round_to is not None else 2)}"
