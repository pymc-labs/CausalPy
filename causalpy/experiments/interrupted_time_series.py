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
        self.data = self._build_data(data)
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
                "obs_ind": np.arange(self.data.X.sel(period="pre").shape[0]),
                "treated_units": ["unit_0"],
            }
            self.model.fit(
                X=self.data.X.sel(period="pre"),
                y=self.data.y.sel(period="pre"),
                coords=COORDS,
            )
        elif isinstance(self.model, RegressorMixin):
            # For OLS models, use 1D y data
            self.model.fit(
                X=self.data.X.sel(period="pre"),
                y=self.data.y.sel(period="pre").isel(treated_units=0),
            )
        else:
            raise ValueError("Model type not recognized")

        # 2. Score the goodness of fit to the pre-intervention data
        self.score = self.model.score(
            X=self.data.X.sel(period="pre"), y=self.data.y.sel(period="pre")
        )

        # 3. Generate predictions for the full dataset using unified approach
        # This creates predictions aligned with our complete time series
        if isinstance(self.model, PyMCModel):
            # PyMC models expect xarray DataArrays
            self.predictions = self.model.predict(X=self.data.X)
            # Add period coordinate to predictions - InferenceData handles multiple data arrays
            self.predictions = self.predictions.assign_coords(
                period=("obs_ind", self.data.period.data)
            )
        else:
            # Sklearn models expect numpy arrays
            pred_array = self.model.predict(X=self.data.X.values)
            # Create xarray DataArray with period coordinate
            self.predictions = xr.DataArray(
                pred_array,
                dims=["obs_ind"],
                coords={
                    "obs_ind": self.data.obs_ind,
                    "period": ("obs_ind", self.data.period.data),
                },
            ).set_xindex("period")

        # 4. Calculate impact
        if isinstance(self.model, PyMCModel):
            # Calculate impact for the entire time series at once
            self.impact = self.model.calculate_impact(self.data.y, self.predictions)
            # Assign period coordinate to unified impact and set index
            self.impact = self.impact.assign_coords(
                period=("obs_ind", self.data.period.data)
            ).set_xindex("period")
        else:
            # For sklearn: calculate unified impact as DataArray
            observed_values = self.data.y.isel(treated_units=0).values
            predicted_values = self.predictions.values
            impact_values = observed_values - predicted_values

            self.impact = xr.DataArray(
                impact_values,
                dims=["obs_ind"],
                coords={
                    "obs_ind": self.data.obs_ind,
                    "period": ("obs_ind", self.data.period.data),
                },
            ).set_xindex("period")

        # 5. Calculate cumulative impact (only on post-intervention period)
        post_impact = self.impact.sel(period="post")
        if isinstance(self.model, PyMCModel):
            self.post_impact_cumulative = self.model.calculate_cumulative_impact(
                post_impact
            )
        else:
            # For sklearn: simple cumulative sum
            self.post_impact_cumulative = post_impact.cumsum()

    def _build_data(self, data: pd.DataFrame) -> xr.Dataset:
        """Build the experiment dataset as unified time series with period coordinate."""
        # Build design matrices for the complete dataset directly
        y_full, X_full = dmatrices(self.formula, data)

        # Store metadata from the design matrices
        self.outcome_variable_name = y_full.design_info.column_names[0]
        self._y_design_info = y_full.design_info
        self._x_design_info = X_full.design_info
        self.labels = X_full.design_info.column_names

        # Create period coordinate based on treatment time
        period_coord = xr.where(data.index < self.treatment_time, "pre", "post")

        # Return as a xarray.Dataset
        common_coords = {
            "obs_ind": data.index,
            "period": ("obs_ind", period_coord),
        }

        return xr.Dataset(
            {
                "X": xr.DataArray(
                    np.asarray(X_full),
                    dims=["obs_ind", "coeffs"],
                    coords={**common_coords, "coeffs": self.labels},
                ),
                "y": xr.DataArray(
                    np.asarray(y_full),
                    dims=["obs_ind", "treated_units"],
                    coords={**common_coords, "treated_units": ["unit_0"]},
                ),
            }
        ).set_xindex("period")

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

    def plot(self, round_to=None, **kwargs) -> tuple[plt.Figure, List[plt.Axes]]:
        """Plot the interrupted time series analysis results.

        Creates a unified plot that works for both Bayesian (PyMC) and OLS (sklearn) models,
        automatically detecting the model type and adjusting the visualization accordingly.

        Parameters
        ----------
        round_to : int, optional
            Number of decimal places to round displayed values. Defaults to 2.
            Use None to return raw numbers.
        **kwargs
            Additional keyword arguments passed to matplotlib plotting functions

        Returns
        -------
        tuple[plt.Figure, List[plt.Axes]]
            Matplotlib figure and list of axes objects
        """
        # Get plot data using the appropriate method based on model type
        if isinstance(self.model, PyMCModel):
            plot_data = self.get_plot_data_bayesian(**kwargs)
            has_hdi = True
        else:
            plot_data = self.get_plot_data_ols()
            has_hdi = False

        # Extract period masks and observation indices for cleaner plotting
        pre_mask = self.data.period == "pre"
        post_mask = self.data.period == "post"
        pre_obs_ind = self.data.X.sel(period="pre").obs_ind
        post_obs_ind = self.data.X.sel(period="post").obs_ind

        # Convert xarray boolean masks to pandas boolean arrays for DataFrame indexing
        pre_mask_pd = pre_mask.values
        post_mask_pd = post_mask.values

        counterfactual_label = "Counterfactual"
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))
        handles = []
        labels = []

        # TOP PLOT - Observations and Predictions -------------------------
        # Plot observations (same for both model types)
        (h,) = ax[0].plot(
            pre_obs_ind,
            plot_data[self.outcome_variable_name][pre_mask_pd],
            "k.",
            label="Observations",
        )
        handles.append(h)
        labels.append("Observations")

        ax[0].plot(
            post_obs_ind,
            plot_data[self.outcome_variable_name][post_mask_pd],
            "k.",
        )

        # Plot predictions with appropriate styling
        if isinstance(self.model, PyMCModel):
            # Bayesian: plot mean predictions as lines
            (h,) = ax[0].plot(
                pre_obs_ind,
                plot_data["prediction"][pre_mask_pd],
                c="C0",
                label="Pre-intervention period",
            )
            handles.append(h)
            labels.append("Pre-intervention period")

            (h,) = ax[0].plot(
                post_obs_ind,
                plot_data["prediction"][post_mask_pd],
                c="C1",
                label=counterfactual_label,
            )
            handles.append(h)
            labels.append(counterfactual_label)
        else:
            # OLS: plot predictions as lines
            ax[0].plot(
                pre_obs_ind,
                plot_data["prediction"][pre_mask_pd],
                c="k",
                label="model fit",
            )
            ax[0].plot(
                post_obs_ind,
                plot_data["prediction"][post_mask_pd],
                label=counterfactual_label,
                ls=":",
                c="k",
            )

        # Add HDI bands if available (Bayesian only)
        if has_hdi:
            hdi_prob = kwargs.get("hdi_prob", 0.94)
            hdi_pct = int(round(hdi_prob * 100))

            # Pre-intervention HDI
            ax[0].fill_between(
                pre_obs_ind,
                plot_data[f"pred_hdi_lower_{hdi_pct}"][pre_mask_pd],
                plot_data[f"pred_hdi_upper_{hdi_pct}"][pre_mask_pd],
                alpha=0.3,
                color="C0",
            )

            # Post-intervention HDI
            ax[0].fill_between(
                post_obs_ind,
                plot_data[f"pred_hdi_lower_{hdi_pct}"][post_mask_pd],
                plot_data[f"pred_hdi_upper_{hdi_pct}"][post_mask_pd],
                alpha=0.3,
                color="C1",
            )

        # Shaded causal effect
        h = ax[0].fill_between(
            post_obs_ind,
            plot_data["prediction"][post_mask_pd],
            plot_data[self.outcome_variable_name][post_mask_pd],
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        handles.append(h)
        labels.append("Causal impact")

        # Set title based on model type
        if isinstance(self.model, PyMCModel):
            title = f"""
            Pre-intervention Bayesian $R^2$: {round_num(self.score["unit_0_r2"], round_to)}
            (std = {round_num(self.score["unit_0_r2_std"], round_to)})
            """
        else:
            title = (
                f"$R^2$ on pre-intervention data = {round_num(self.score, round_to)}"
            )
        ax[0].set(title=title)

        # MIDDLE PLOT - Causal Impact -----------------------------------
        ax[1].plot(pre_obs_ind, plot_data["impact"][pre_mask_pd], "k.")
        ax[1].plot(
            post_obs_ind,
            plot_data["impact"][post_mask_pd],
            "k.",
            label=counterfactual_label,
        )
        ax[1].axhline(y=0, c="k")

        # Add HDI for impact if available
        if has_hdi:
            ax[1].fill_between(
                pre_obs_ind,
                plot_data[f"impact_hdi_lower_{hdi_pct}"][pre_mask_pd],
                plot_data[f"impact_hdi_upper_{hdi_pct}"][pre_mask_pd],
                alpha=0.3,
                color="C0",
            )
            ax[1].fill_between(
                post_obs_ind,
                plot_data[f"impact_hdi_lower_{hdi_pct}"][post_mask_pd],
                plot_data[f"impact_hdi_upper_{hdi_pct}"][post_mask_pd],
                alpha=0.3,
                color="C1",
            )

        # Shaded causal impact
        ax[1].fill_between(
            post_obs_ind,
            plot_data["impact"][post_mask_pd],
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[1].set(title="Causal Impact")

        # BOTTOM PLOT - Cumulative Impact -------------------------------
        # Ensure cumulative impact is 1D for plotting (mean over all dims except obs_ind)
        cum_impact = self.post_impact_cumulative
        if (
            hasattr(cum_impact, "dims")
            and hasattr(cum_impact, "mean")
            and cum_impact.ndim > 1
        ):
            # Find all dims except obs_ind
            dims_to_mean = [d for d in cum_impact.dims if d != "obs_ind"]
            cum_impact = cum_impact.mean(dim=dims_to_mean)
        ax[2].plot(post_obs_ind, cum_impact, c="k")
        ax[2].axhline(y=0, c="k")
        ax[2].set(title="Cumulative Causal Impact")

        # Add intervention lines to all plots
        for i in range(3):
            ax[i].axvline(x=self.treatment_time, ls="-", lw=3, color="r")

        # Legend for top plot
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
        if not isinstance(self.model, PyMCModel):
            raise ValueError("Unsupported model type")

        hdi_pct = int(round(hdi_prob * 100))

        # Start with the outcome data from our unified dataset
        plot_data = pd.DataFrame(
            {self.outcome_variable_name: self.data.y.isel(treated_units=0).values},
            index=self.data.y.obs_ind.values,
        )

        # Extract predictions directly from unified predictions object
        pred_mu = self.predictions["posterior_predictive"].mu.isel(treated_units=0)
        plot_data["prediction"] = pred_mu.mean(dim=["chain", "draw"]).values

        # Extract impact directly from unified impact - no more calculation needed!
        plot_data["impact"] = (
            self.impact.mean(dim=["chain", "draw"]).isel(treated_units=0).values
        )

        # Calculate HDI bounds directly using arviz
        pred_hdi = az.hdi(pred_mu, hdi_prob=hdi_prob)
        impact_hdi = az.hdi(self.impact.isel(treated_units=0), hdi_prob=hdi_prob)

        # Extract HDI bounds from xarray Dataset results
        pred_var_name = list(pred_hdi.data_vars.keys())[0]
        impact_var_name = list(impact_hdi.data_vars.keys())[0]

        pred_hdi_data = pred_hdi[pred_var_name]
        impact_hdi_data = impact_hdi[impact_var_name]

        plot_data[f"pred_hdi_lower_{hdi_pct}"] = pred_hdi_data.isel(hdi=0).values
        plot_data[f"pred_hdi_upper_{hdi_pct}"] = pred_hdi_data.isel(hdi=1).values
        plot_data[f"impact_hdi_lower_{hdi_pct}"] = impact_hdi_data.isel(hdi=0).values
        plot_data[f"impact_hdi_upper_{hdi_pct}"] = impact_hdi_data.isel(hdi=1).values

        self.plot_data = plot_data
        return plot_data

    def get_plot_data_ols(self) -> pd.DataFrame:
        """
        Recover the data of the experiment along with the prediction and causal impact information.
        """
        # Create unified DataFrame from our xarray data
        plot_data = pd.DataFrame(
            {self.outcome_variable_name: self.data.y.isel(treated_units=0).values},
            index=self.data.y.obs_ind.values,
        )

        # Extract directly from unified data structures - ultimate simplification!
        plot_data["prediction"] = self.predictions.values
        plot_data["impact"] = self.impact.values

        self.plot_data = plot_data
        return self.plot_data
