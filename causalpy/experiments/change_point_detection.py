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

This module implements interrupted time series (ITS) analysis for causal inference,
supporting both traditional scenarios where the intervention time is known and
advanced scenarios where the intervention time must be inferred from the data.

Overview
--------
Interrupted time series analysis is a quasi-experimental design used to evaluate
the impact of an intervention by comparing time series data before and after the
intervention occurs. This module provides a flexible framework that can handle:

1. **Known intervention times**: Traditional ITS where you specify exactly when
   the treatment occurred (e.g., policy implementation date)
2. **Unknown intervention times**: Advanced ITS where the model infers when an
   intervention likely occurred based on observed changes in the data

Treatment Time Handler Architecture
----------------------------------
The core design pattern in this module is the Strategy pattern implemented through
the `TreatmentTimeHandler` hierarchy. This architecture was necessary because known
and unknown treatment times require fundamentally different approaches:

**Why the Handler Architecture?**

- **Data Processing**: Known times require splitting data at a specific point;
  unknown times need the full dataset for inference
- **Model Training**: Known times train only on pre-intervention data; unknown
  times train on all available data to detect the changepoint
- **Uncertainty Handling**: Known times have deterministic splits; unknown times
  have probabilistic splits with confidence intervals
- **Visualization**: Different plotting strategies for certain vs. uncertain
  intervention times

**Handler Classes:**

1. **TreatmentTimeHandler (Abstract Base Class)**

   - Defines the interface that all concrete handlers must implement
   - Ensures consistent API regardless of whether treatment time is known/unknown
   - Abstract methods: data_preprocessing, data_postprocessing, plot_intervention_line,
     plot_impact_cumulative
   - Optional method: plot_treated_counterfactual (only needed for unknown times)

2. **KnownTreatmentTimeHandler**

   - Handles traditional ITS scenarios with predetermined intervention times
   - **Data Preprocessing**: Filters data to pre-intervention period only for training
   - **Data Postprocessing**: Creates clean pre/post splits at the known time point
   - **Plotting**: Draws single vertical line at the intervention time
   - **Use Case**: Policy evaluations, clinical trials, A/B tests with known start dates

3. **UnknownTreatmentTimeHandler**

   - Handles advanced ITS scenarios where intervention time is inferred
   - **Data Preprocessing**: Uses full dataset and constrains model's search window
   - **Data Postprocessing**: Extracts inferred treatment time from posterior samples,
     creates probabilistic pre/post splits, handles uncertainty propagation
   - **Plotting**: Draws intervention line with uncertainty bands (HDI), shows
     "treated counterfactual" predictions
   - **Use Case**: Exploratory analysis, natural experiments, detecting unknown
     structural breaks

The handler pattern ensures that:

- The main `InterruptedTimeSeries` class maintains a clean, unified API
- Different treatment time scenarios are handled with appropriate algorithms
- New handler types can be easily added (e.g., multiple intervention times)
- Code is maintainable and testable with clear separation of concerns

Usage Examples
--------------
Known treatment time (traditional approach):

>>> result = cp.ChangePointDetection(
...     data=df,
...     time_range=None
...     formula="y ~ 1 + t + C(month)",
...     model=cp.pymc_models.LinearChangePointDetection(),
... )

The module automatically selects the appropriate handler based on the treatment_time
parameter and model type, providing a seamless user experience while maintaining
the flexibility to handle diverse analytical scenarios.
"""

from typing import Iterable, List, Union

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from patsy import dmatrices

from causalpy.custom_exceptions import BadIndexException, ModelException
from causalpy.experiments.base import BaseExperiment
from causalpy.plot_utils import get_hdi_to_df, plot_xY
from causalpy.pymc_models import PyMCModel
from causalpy.utils import round_num

LEGEND_FONT_SIZE = 12


class ChangePointDetection(BaseExperiment):
    """
    The class for detecting turning point in time series.

    :param data:
        A pandas dataframe
    :param formula:
        A statistical model formula
    :param treatment_time_range:
        The time range when treatment could've occurred,
        should be in reference to the data index
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

    expt_type = "Change Point Detection"
    supports_ols = False
    supports_bayes = True

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        time_range: Union[Iterable, None] = None,
        model=None,
        **kwargs,
    ) -> None:
        super().__init__(model=model)

        # rename the index to "obs_ind"
        data.index.name = "obs_ind"
        self.input_validation(data, time_range, model)

        # set experiment type - usually done in subclasses
        self.expt_type = "Pre-Post Fit"

        self.time_range = time_range
        self.formula = formula

        # Define the time interval over which the model will perform inference
        model.set_time_range(self.time_range, data)

        # Preprocess the data according to the given formula
        y, X = dmatrices(formula, data)

        self.outcome_variable_name = y.design_info.column_names[0]
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.y, self.X = np.asarray(y), np.asarray(X)

        # turn into xarray.DataArray's
        self.X = xr.DataArray(
            self.X,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": data.index,
                "coeffs": self.labels,
            },
        )
        self.y = xr.DataArray(
            self.y,  # Keep 2D shape
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": data.index, "treated_units": ["unit_0"]},
        )

        # fit the model to the observed data
        if isinstance(self.model, PyMCModel):
            COORDS = {
                "coeffs": self.labels,
                "obs_ind": np.arange(self.X.shape[0]),
                "treated_units": ["unit_0"],
            }
            idata = self.model.fit(X=self.X, y=self.y, coords=COORDS)
        else:
            raise ValueError("Model type not recognized")

        # score the goodness of fit to the pre-intervention data
        self.score = self.model.score(X=self.X, y=self.y)

        # Getting inferred change point values
        cp_samples = idata.posterior["change_point"].values
        cp_mean = int(cp_samples.mean().item())

        # Actual timestamp (index) corresponding to inferred change point
        self.changepoint = data.index[cp_mean]

        # --- Slice data into pre/post change point ---
        self.datapre = data.head(cp_mean)
        self.datapost = data.iloc[cp_mean:]

        # --- Slice covariates into pre/post change point ---
        self.pre_y = self.y.isel(obs_ind=slice(0, cp_mean))
        self.pre_X = self.X.isel(obs_ind=slice(0, cp_mean))
        self.post_y = self.y.isel(obs_ind=slice(cp_mean, None))
        self.post_X = self.X.isel(obs_ind=slice(cp_mean, None))

        # --- Predict outcomes using the model ---
        pred = model.predict(X=self.X)
        self.pre_pred = pred.isel(obs_ind=slice(0, cp_mean))
        self.post_pred = pred.isel(obs_ind=slice(cp_mean, None))

        # --- Estimate causal impact ---
        impact = model.calculate_impact(y, pred)
        self.pre_impact = impact.isel(obs_ind=slice(0, cp_mean))
        self.post_impact = impact.isel(obs_ind=slice(cp_mean, None))

        # --- Create a mask to isolate post-change point period ---
        # Timeline reshaped to match broadcasting with change point
        # (Probably could be better implemented though)
        timeline = [
            [[i for i in range(len(data))] for _ in range(len(cp_samples[0]))]
            for _ in range(len(cp_samples))
        ]
        timeline_broadcast = np.array(timeline)
        tt_broadcast = cp_samples[:, :, None].astype(int)
        mask = (timeline_broadcast >= tt_broadcast).astype(int)
        mask = mask[:, :, np.newaxis, :]
        post_impact_masked = impact * mask

        # --- Compute cumulative post-change point impact ---
        post_impact_masked = impact * mask
        self.post_impact_cumulative = model.calculate_cumulative_impact(
            post_impact_masked
        )

    def input_validation(self, data, time_range, model):
        """Validate the input data and model formula for correctness"""
        if not hasattr(model, "set_time_range"):
            raise ModelException("Provided model must have a 'set_time_range' method")
        if time_range is not None and len(time_range) != 2:
            raise BadIndexException(
                "Provided time_range must be of length 2 : (start, end)"
            )
        if isinstance(data.index, pd.DatetimeIndex) and not (
            time_range is None
            or (
                isinstance(time_range, Iterable)
                and all(isinstance(t, pd.Timestamp) for t in time_range)
            )
        ):
            raise BadIndexException(
                "If data.index is DatetimeIndex, time_range must "
                "be of type Iterable[pd.Timestamp]."
            )
        if not isinstance(data.index, pd.DatetimeIndex) and (
            isinstance(time_range, Iterable)
            and all(isinstance(t, pd.Timestamp) for t in time_range)
        ):
            raise BadIndexException(
                "If data.index is not DatetimeIndex, time_range must"
                "not be of type Iterable[pd.Timestamp]."  # noqa: E501
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
        handles = []
        labels = []

        # Treated counterfactual
        # Plot predicted values after change point (with HDI)
        h_line, h_patch = plot_xY(
            self.datapre.index,
            self.pre_pred["posterior_predictive"].mu_ts.isel(treated_units=0),
            ax=ax[0],
            plot_hdi_kwargs={"color": "yellowgreen"},
        )

        h_line, h_patch = plot_xY(
            self.datapost.index,
            self.post_pred["posterior_predictive"].mu_ts.isel(treated_units=0),
            ax=ax[0],
            plot_hdi_kwargs={"color": "yellowgreen"},
        )

        handles.append((h_line, h_patch))
        labels.append("Treated counterfactual")

        # pre-intervention period
        h_line, h_patch = plot_xY(
            self.datapre.index,
            self.pre_pred["posterior_predictive"].mu.isel(treated_units=0),
            ax=ax[0],
            plot_hdi_kwargs={"color": "C0"},
        )
        handles.append((h_line, h_patch))
        labels.append("Pre-intervention period")

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
        h_line, h_patch = plot_xY(
            self.datapost.index,
            self.post_pred["posterior_predictive"].mu.isel(treated_units=0),
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
        post_pred_mu = (
            az.extract(self.post_pred, group="posterior_predictive", var_names="mu")
            .isel(treated_units=0)
            .mean("sample")
        )  # Add .mean("sample") to get 1D array
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

        ax[0].set(
            title=f"""
            Pre-intervention Bayesian $R^2$: {round_num(self.score["unit_0_r2"], round_to)}
            (std = {round_num(self.score["unit_0_r2_std"], round_to)})
            """
        )

        # MIDDLE PLOT -----------------------------------------------
        plot_xY(
            self.datapre.index,
            self.pre_impact.isel(treated_units=0),
            ax=ax[1],
            plot_hdi_kwargs={"color": "C0"},
        )
        plot_xY(
            self.datapost.index,
            self.post_impact.isel(treated_units=0),
            ax=ax[1],
            plot_hdi_kwargs={"color": "C1"},
        )
        ax[1].axhline(y=0, c="k")
        ax[1].fill_between(
            self.datapost.index,
            y1=self.post_impact.mean(["chain", "draw"]).isel(treated_units=0),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[1].set(title="Causal Impact")

        # BOTTOM PLOT -----------------------------------------------
        # Concatenate the time indices
        full_index = self.datapre.index.append(self.datapost.index)
        ax[2].set(title="Cumulative Causal Impact")
        plot_xY(
            full_index,
            self.post_impact_cumulative.isel(treated_units=0),
            ax=ax[2],
            plot_hdi_kwargs={"color": "C1"},
        )
        ax[2].axhline(y=0, c="k")

        # Plot vertical line marking change point (with HDI if it's inferred)
        data = pd.concat([self.datapre, self.datapost])
        # Extract the HDI (uncertainty interval) of the change point
        hdi = az.hdi(self.idata, var_names=["change_point"])["change_point"].values
        x1 = data.index[int(hdi[0])]
        x2 = data.index[int(hdi[1])]

        for i in [0, 1, 2]:
            ymin, ymax = ax[i].get_ylim()

            # Vertical line for inferred change point
            ax[i].plot(
                [self.changepoint, self.changepoint],
                [ymin, ymax],
                ls="-",
                lw=3,
                color="r",
                solid_capstyle="butt",
            )

            # Shaded region for HDI of change point
            ax[i].fill_betweenx(
                y=[ymin, ymax],
                x1=x1,
                x2=x2,
                alpha=0.1,
                color="r",
            )

        ax[0].legend(
            handles=(h_tuple for h_tuple in handles),
            labels=labels,
            fontsize=LEGEND_FONT_SIZE,
        )

        return fig, ax

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

    def plot_change_point(self):
        """
        display the posterior estimates of the change point
        """
        if "change_point" not in self.idata.posterior.data_vars:
            raise ValueError(
                "Variable 'change_point' not found in inference data (idata)."
            )

        az.plot_trace(self.idata, var_names="change_point")
