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
from patsy import dmatrices
from sklearn.base import RegressorMixin

from causalpy.custom_exceptions import BadIndexException
from causalpy.plot_utils import get_hdi_to_df, plot_xY
from causalpy.pymc_models import PyMCModel
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
        data.index.name = "obs_ind"
        self.input_validation(data, treatment_time)
        self.treatment_time = treatment_time
        self.expt_type = "Pre-Post Fit"
        self.formula = formula
        self._data = self._build_data(data)
        self.algorithm()

    def algorithm(self) -> None:
        """Execute the core interrupted time series algorithm.

        This method implements the standard interrupted time series analysis workflow:
        1. Fit model on pre-intervention data
        2. Score model goodness of fit
        3. Generate predictions for pre and post periods
        4. Calculate causal impact and cumulative impact
        """
        # 1. Fit the model to the observed (pre-intervention) data
        if isinstance(self.model, PyMCModel):
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

        # 2. Score the goodness of fit to the pre-intervention data
        self.score = self.model.score(X=self.pre_X, y=self.pre_y)

        # 3a. Get the model predictions of the observed (pre-intervention) data
        self.pre_pred = self.model.predict(X=self.pre_X)

        # 3b. Calculate the counterfactual
        self.post_pred = self.model.predict(X=self.post_X)

        # 4a. Calculate impact - use appropriate y data format for each model type
        if isinstance(self.model, PyMCModel):
            # PyMC models work with 2D data
            self.pre_impact = self.model.calculate_impact(self.pre_y, self.pre_pred)
            self.post_impact = self.model.calculate_impact(self.post_y, self.post_pred)
        elif isinstance(self.model, RegressorMixin):
            # SKL models work with 1D data
            self.pre_impact = self.model.calculate_impact(
                self.pre_y.isel(treated_units=0), self.pre_pred
            )
            self.post_impact = self.model.calculate_impact(
                self.post_y.isel(treated_units=0), self.post_pred
            )

        # 4b. Calculate cumulative impact
        self.post_impact_cumulative = self.model.calculate_cumulative_impact(
            self.post_impact
        )

    @staticmethod
    def _is_pre_intervention(obs_ind, treatment_time) -> bool:
        """Check if observation indices are pre-intervention.

        :param obs_ind: Observation indices to check
        :param treatment_time: The treatment time threshold
        :return: Boolean mask for pre-intervention observations
        """
        return obs_ind < treatment_time

    @staticmethod
    def _is_post_intervention(obs_ind, treatment_time) -> bool:
        """Check if observation indices are post-intervention.

        :param obs_ind: Observation indices to check
        :param treatment_time: The treatment time threshold
        :return: Boolean mask for post-intervention observations
        """
        return obs_ind >= treatment_time

    def _build_data(self, data: pd.DataFrame) -> xr.Dataset:
        """Build the experiment dataset as unified time series."""
        # Build design matrices for the complete dataset directly
        y_full, X_full = dmatrices(self.formula, data)

        # Store metadata from the design matrices
        self.outcome_variable_name = y_full.design_info.column_names[0]
        self._y_design_info = y_full.design_info
        self._x_design_info = X_full.design_info
        self.labels = X_full.design_info.column_names

        # Return complete time series as a single xarray Dataset
        return xr.Dataset(
            {
                "X": xr.DataArray(
                    np.asarray(X_full),
                    dims=["obs_ind", "coeffs"],
                    coords={
                        "obs_ind": data.index,
                        "coeffs": self.labels,
                    },
                ),
                "y": xr.DataArray(
                    np.asarray(y_full),
                    dims=["obs_ind", "treated_units"],
                    coords={
                        "obs_ind": data.index,
                        "treated_units": ["unit_0"],
                    },
                ),
            }
        )

    # Properties for pre/post intervention data access
    @property
    def pre_X(self) -> xr.DataArray:
        """Pre-intervention features."""
        return self._data.X.sel(
            obs_ind=self._is_pre_intervention(self._data.X.obs_ind, self.treatment_time)
        )

    @property
    def pre_y(self) -> xr.DataArray:
        """Pre-intervention outcomes."""
        return self._data.y.sel(
            obs_ind=self._is_pre_intervention(self._data.y.obs_ind, self.treatment_time)
        )

    @property
    def post_X(self) -> xr.DataArray:
        """Post-intervention features."""
        return self._data.X.sel(
            obs_ind=self._is_post_intervention(
                self._data.X.obs_ind, self.treatment_time
            )
        )

    @property
    def post_y(self) -> xr.DataArray:
        """Post-intervention outcomes."""
        return self._data.y.sel(
            obs_ind=self._is_post_intervention(
                self._data.y.obs_ind, self.treatment_time
            )
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
        h_line, h_patch = plot_xY(
            self.pre_X.obs_ind,
            self.pre_pred["posterior_predictive"].mu.isel(treated_units=0),
            ax=ax[0],
            plot_hdi_kwargs={"color": "C0"},
        )
        handles = [(h_line, h_patch)]
        labels = ["Pre-intervention period"]

        (h,) = ax[0].plot(
            self.pre_X.obs_ind,
            self.pre_y.isel(treated_units=0)
            if hasattr(self.pre_y, "isel")
            else self.pre_y[:, 0],
            "k.",
            label="Observations",
        )
        handles.append(h)
        labels.append("Observations")

        # post intervention period
        h_line, h_patch = plot_xY(
            self.post_X.obs_ind,
            self.post_pred["posterior_predictive"].mu.isel(treated_units=0),
            ax=ax[0],
            plot_hdi_kwargs={"color": "C1"},
        )
        handles.append((h_line, h_patch))
        labels.append(counterfactual_label)

        ax[0].plot(
            self.post_X.obs_ind,
            self.post_y.isel(treated_units=0)
            if hasattr(self.post_y, "isel")
            else self.post_y[:, 0],
            "k.",
        )
        # Shaded causal effect
        post_pred_mu = (
            az.extract(self.post_pred, group="posterior_predictive", var_names="mu")
            .isel(treated_units=0)
            .mean("sample")
        )  # Add .mean("sample") to get 1D array
        h = ax[0].fill_between(
            self.post_X.obs_ind,
            y1=post_pred_mu,
            y2=self.post_y.isel(treated_units=0)
            if hasattr(self.post_y, "isel")
            else self.post_y[:, 0],
            color="C0",
            alpha=0.25,
        )
        handles.append(h)
        labels.append("Causal impact")

        ax[0].set(
            title=f"""
            Pre-intervention Bayesian $R^2$: {round_num(self.score["unit_0_r2"], round_to)}
            (std = {round_num(self.score["unit_0_r2_std"], round_to)})
            """
        )

        # MIDDLE PLOT -----------------------------------------------
        plot_xY(
            self.pre_X.obs_ind,
            self.pre_impact.isel(treated_units=0),
            ax=ax[1],
            plot_hdi_kwargs={"color": "C0"},
        )
        plot_xY(
            self.post_X.obs_ind,
            self.post_impact.isel(treated_units=0),
            ax=ax[1],
            plot_hdi_kwargs={"color": "C1"},
        )
        ax[1].axhline(y=0, c="k")
        ax[1].fill_between(
            self.post_X.obs_ind,
            y1=self.post_impact.mean(["chain", "draw"]).isel(treated_units=0),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[1].set(title="Causal Impact")

        # BOTTOM PLOT -----------------------------------------------
        ax[2].set(title="Cumulative Causal Impact")
        plot_xY(
            self.post_X.obs_ind,
            self.post_impact_cumulative.isel(treated_units=0),
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

        ax[0].plot(self.pre_X.obs_ind, self.pre_y, "k.")
        ax[0].plot(self.post_X.obs_ind, self.post_y, "k.")

        ax[0].plot(self.pre_X.obs_ind, self.pre_pred, c="k", label="model fit")
        ax[0].plot(
            self.post_X.obs_ind,
            self.post_pred,
            label=counterfactual_label,
            ls=":",
            c="k",
        )
        ax[0].set(
            title=f"$R^2$ on pre-intervention data = {round_num(self.score, round_to)}"
        )

        ax[1].plot(self.pre_X.obs_ind, self.pre_impact, "k.")
        ax[1].plot(
            self.post_X.obs_ind,
            self.post_impact,
            "k.",
            label=counterfactual_label,
        )
        ax[1].axhline(y=0, c="k")
        ax[1].set(title="Causal Impact")

        ax[2].plot(self.post_X.obs_ind, self.post_impact_cumulative, c="k")
        ax[2].axhline(y=0, c="k")
        ax[2].set(title="Cumulative Causal Impact")

        # Shaded causal effect
        ax[0].fill_between(
            self.post_X.obs_ind,
            y1=np.squeeze(self.post_pred),
            y2=np.squeeze(self.post_y),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[1].fill_between(
            self.post_X.obs_ind,
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

            # Reconstruct DataFrame structure from our xarray data
            pre_data = pd.DataFrame(
                {self.outcome_variable_name: self.pre_y.isel(treated_units=0).values},
                index=self.pre_y.obs_ind.values,
            )

            post_data = pd.DataFrame(
                {self.outcome_variable_name: self.post_y.isel(treated_units=0).values},
                index=self.post_y.obs_ind.values,
            )

            pre_data["prediction"] = (
                az.extract(self.pre_pred, group="posterior_predictive", var_names="mu")
                .mean("sample")
                .isel(treated_units=0)
                .values
            )
            post_data["prediction"] = (
                az.extract(self.post_pred, group="posterior_predictive", var_names="mu")
                .mean("sample")
                .isel(treated_units=0)
                .values
            )
            hdi_pre_pred = get_hdi_to_df(
                self.pre_pred["posterior_predictive"].mu, hdi_prob=hdi_prob
            )
            hdi_post_pred = get_hdi_to_df(
                self.post_pred["posterior_predictive"].mu, hdi_prob=hdi_prob
            )
            # Select the single unit from the MultiIndex results
            pre_data[[pred_lower_col, pred_upper_col]] = hdi_pre_pred.xs(
                "unit_0", level="treated_units"
            ).set_index(pre_data.index)
            post_data[[pred_lower_col, pred_upper_col]] = hdi_post_pred.xs(
                "unit_0", level="treated_units"
            ).set_index(post_data.index)

            pre_data["impact"] = (
                self.pre_impact.mean(dim=["chain", "draw"]).isel(treated_units=0).values
            )
            post_data["impact"] = (
                self.post_impact.mean(dim=["chain", "draw"])
                .isel(treated_units=0)
                .values
            )
            hdi_pre_impact = get_hdi_to_df(self.pre_impact, hdi_prob=hdi_prob)
            hdi_post_impact = get_hdi_to_df(self.post_impact, hdi_prob=hdi_prob)
            # Select the single unit from the MultiIndex results
            pre_data[[impact_lower_col, impact_upper_col]] = hdi_pre_impact.xs(
                "unit_0", level="treated_units"
            ).set_index(pre_data.index)
            post_data[[impact_lower_col, impact_upper_col]] = hdi_post_impact.xs(
                "unit_0", level="treated_units"
            ).set_index(post_data.index)

            self.plot_data = pd.concat([pre_data, post_data])

            return self.plot_data
        else:
            raise ValueError("Unsupported model type")

    def get_plot_data_ols(self) -> pd.DataFrame:
        """
        Recover the data of the experiment along with the prediction and causal impact information.
        """
        # Reconstruct DataFrame structure from our xarray data
        pre_data = pd.DataFrame(
            {self.outcome_variable_name: self.pre_y.isel(treated_units=0).values},
            index=self.pre_y.obs_ind.values,
        )

        post_data = pd.DataFrame(
            {self.outcome_variable_name: self.post_y.isel(treated_units=0).values},
            index=self.post_y.obs_ind.values,
        )

        pre_data["prediction"] = self.pre_pred
        post_data["prediction"] = self.post_pred
        pre_data["impact"] = self.pre_impact
        post_data["impact"] = self.post_impact
        self.plot_data = pd.concat([pre_data, post_data])

        return self.plot_data
