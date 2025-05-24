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
Structured Time Series Analysis using Bayesian Structural Time Series Models.
"""

from typing import List, Optional

import arviz as az
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from patsy import dmatrices

from causalpy.custom_exceptions import BadIndexException
from causalpy.plot_utils import get_hdi_to_df, plot_xY
from causalpy.pymc_models import BayesianStructuralTimeSeries
from causalpy.utils import round_num

from .base import BaseExperiment

LEGEND_FONT_SIZE = 12


class StructuredTimeSeries(BaseExperiment):
    r"""
    A class for time series analysis using Bayesian Structural Time Series (BSTS) models.

    This experiment type is similar to Interrupted Time Series but specifically
    uses the `BayesianStructuralTimeSeries` model, which handles trend and
    seasonality internally based on time features.

    :param data:
        A pandas DataFrame with a DatetimeIndex.
    :param treatment_time:
        The time when treatment occurred. Must be a pandas Timestamp.
    :param formula:
        A patsy-style formula string for exogenous regressors (e.g., "~ x1 + x2").
        If no exogenous regressors are needed, use "~ 0" or "~ 1" (the intercept
        will be handled by the BSTS model's trend component).
    :param model:
        An instance of `causalpy.pymc_models.BayesianStructuralTimeSeries`.
    :param time_variable_name_for_trend: str, optional
        The name for the numeric time variable used for the trend component.
        Defaults to "time_numeric". This is generated as days since the
        start of the data, scaled by 1/365.25.
    :param time_variable_name_for_seasonality: str, optional
        The name for the numeric time variable used for the seasonality component.
        Defaults to "day_of_year". This is generated from the DatetimeIndex.

    Example
    --------
    >>> import causalpy as cp
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(seed=123)
    >>> dates = pd.date_range(start="2019-01-01", end="2022-12-31", freq="D")
    >>> n_obs = len(dates)
    >>> trend_actual = np.linspace(0, 2, n_obs)
    >>> seasonality_actual = 3 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    >>> x1_actual = rng.normal(0, 1, n_obs)
    >>> noise_actual = rng.normal(0, 0.3, n_obs)
    >>> y_values = trend_actual + seasonality_actual + 1.5 * x1_actual + noise_actual
    >>> df = pd.DataFrame({"y": y_values, "x1": x1_actual}, index=dates)
    >>> treatment_time = pd.Timestamp("2021-01-01")
    >>> bsts_model = cp.pymc_models.BayesianStructuralTimeSeries(
    ...     n_order=3,
    ...     n_changepoints_trend=10,
    ...     sample_kwargs={
    ...         "chains": 1,
    ...         "draws": 500,
    ...         "tune": 200,
    ...         "progressbar": False,
    ...         "random_seed": 42,
    ...     },
    ... )
    >>> result = cp.StructuredTimeSeries(
    ...     df,
    ...     treatment_time,
    ...     formula="y ~ x1",  # Exogenous regressor x1
    ...     model=bsts_model,
    ... )
    >>> result.summary()  # doctest: +ELLIPSIS
    =========================Structured Time Series Fit=========================
    Formula: y ~ x1
    Model coefficients:
    ...
    >>> fig, ax = result.plot()
    """

    expt_type = "Structured Time Series"
    supports_bayes = True
    supports_ols = False  # BSTS is inherently Bayesian

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_time: pd.Timestamp,
        formula: str,
        model: BayesianStructuralTimeSeries,
        time_variable_name_for_trend: str = "time_numeric",
        time_variable_name_for_seasonality: str = "day_of_year",
    ) -> None:
        super().__init__(model=model)
        self.input_validation(data, treatment_time)
        self.treatment_time = treatment_time
        self.expt_type = "Structured Time Series Fit"  # More specific for summary
        self.formula = formula

        if not isinstance(self.model, BayesianStructuralTimeSeries):
            raise TypeError(
                "Model must be an instance of BayesianStructuralTimeSeries."
            )

        # Prepare time features for the entire dataset
        data_with_time_features = data.copy()
        data_with_time_features[time_variable_name_for_trend] = (
            data_with_time_features.index - data_with_time_features.index[0]
        ).days / 365.25
        data_with_time_features[time_variable_name_for_seasonality] = (
            data_with_time_features.index.dayofyear
        )

        # Split data into pre and post intervention
        self.datapre = data_with_time_features[
            data_with_time_features.index < self.treatment_time
        ].copy()
        self.datapost = data_with_time_features[
            data_with_time_features.index >= self.treatment_time
        ].copy()

        # Patsy for exogenous variables X, and target y
        # The formula should only contain exogenous regressors, e.g., "y ~ x1 + x2"
        # or "y ~ 0" if no exogenous variables are used.
        y_df, X_df_patsy = dmatrices(
            formula, data_with_time_features, return_type="dataframe"
        )
        self.outcome_variable_name = y_df.columns[0]

        X_df_exog = X_df_patsy.copy()

        self.labels = list(X_df_exog.columns)  # Labels for actual exogenous regressors

        # Split X (exogenous only) and y
        self.pre_X_regressors = X_df_exog[X_df_exog.index < self.treatment_time]
        self.post_X_regressors = X_df_exog[X_df_exog.index >= self.treatment_time]
        self.pre_y = y_df[y_df.index < self.treatment_time][
            self.outcome_variable_name
        ].values.reshape(-1, 1)
        self.post_y = y_df[y_df.index >= self.treatment_time][
            self.outcome_variable_name
        ].values.reshape(-1, 1)

        # Handle case with no regressors (e.g. y ~ 0 or y ~ 1, after Intercept removal)
        if (
            not self.labels
        ):  # Check if self.labels is empty (i.e., X_df_exog has no columns)
            self.pre_X_fit = None
            self.post_X_pred = None
            coords_coeffs = {}
        else:
            self.pre_X_fit = self.pre_X_regressors.values
            self.post_X_pred = self.post_X_regressors.values
            coords_coeffs = {"coeffs": self.labels}

        # Coordinates for the BSTS model
        pre_coords = {
            "obs_ind": np.arange(self.datapre.shape[0]),
            "time_for_trend": self.datapre[time_variable_name_for_trend].values,
            "time_for_seasonality": self.datapre[
                time_variable_name_for_seasonality
            ].values,
            **coords_coeffs,
        }

        # Fit the model to the pre-intervention data
        self.model.fit(
            X=self.pre_X_fit,
            y=self.pre_y,
            coords=pre_coords,
        )

        # Score the goodness of fit to the pre-intervention data
        self.score = self.model.score(
            X=self.pre_X_fit,
            y=self.pre_y,
            time_for_trend_pred=self.datapre[time_variable_name_for_trend].values,
            time_for_seasonality_pred=self.datapre[
                time_variable_name_for_seasonality
            ].values,
        )

        # Get model predictions for the pre-intervention period (in-sample)
        self.pre_pred = self.model.predict(
            X=self.pre_X_fit,
            time_for_trend_pred=self.datapre[time_variable_name_for_trend].values,
            time_for_seasonality_pred=self.datapre[
                time_variable_name_for_seasonality
            ].values,
        )

        # Calculate counterfactual for the post-intervention period
        self.post_pred = self.model.predict(
            X=self.post_X_pred,  # Use post-intervention exogenous vars if any
            time_for_trend_pred=self.datapost[time_variable_name_for_trend].values,
            time_for_seasonality_pred=self.datapost[
                time_variable_name_for_seasonality
            ].values,
        )

        # Calculate impacts
        self.pre_impact = self.model.calculate_impact(
            self.pre_y.flatten(), self.pre_pred
        )
        self.post_impact = self.model.calculate_impact(
            self.post_y.flatten(), self.post_pred
        )
        self.post_impact_cumulative = self.model.calculate_cumulative_impact(
            self.post_impact
        )

    def input_validation(self, data, treatment_time):
        """Validate the input data and model formula for correctness"""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise BadIndexException("Data must have a pandas DatetimeIndex.")
        if not isinstance(treatment_time, pd.Timestamp):
            raise BadIndexException("treatment_time must be a pandas Timestamp.")
        if treatment_time <= data.index.min() or treatment_time >= data.index.max():
            raise ValueError(
                "treatment_time must be within the range of the data index."
            )

    def summary(self, round_to: Optional[int] = 2) -> None:
        r"""Print summary of main results and model coefficients.

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" for raw numbers.
        """
        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        # Coefficients for exogenous variables are in 'beta'
        # Coefficients for seasonality are in 'fourier_beta' (from pymc-marketing)
        # Coefficients for trend are in 'delta', 'k', 'm' (from pymc-marketing)
        # Sigma for overall error

        # We can extend print_coefficients in BSTS or handle it here
        # For now, let's print what PyMCModel.print_coefficients provides (beta, sigma)
        # and then add specific BSTS components if they exist.

        print("\nModel coefficients (Exogenous Regressors and Error Sigma):")
        if self.labels:  # If there were exogenous regressors
            self.model.print_coefficients(labels=self.labels, round_to=round_to)
        else:
            print("  No exogenous regressors in the model.")
            # Still print sigma if it exists
            sigma_samples = az.extract(self.model.idata.posterior, var_names="sigma")
            print(
                f"  sigma: {round_num(sigma_samples.mean().data, round_to)}, 94% HDI [{round_num(sigma_samples.quantile(0.03).data, round_to)}, {round_num(sigma_samples.quantile(1 - 0.03).data, round_to)}]"
            )

        # TODO: Add printing for trend and seasonality components if desired,
        # e.g. by accessing self.model.idata.posterior['fourier_beta'], self.model.idata.posterior['delta'] etc.
        # This would require knowing the internal variable names of the components.

    def _bayesian_plot(
        self, round_to: Optional[int] = 2, **kwargs
    ) -> tuple[plt.Figure, List[plt.Axes]]:
        """
        Plot the results. This is specific to Bayesian models.

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use \"None\" for raw numbers.
        """
        counterfactual_label = "Counterfactual (Predicted Post-treatment)"
        fig, ax = plt.subplots(
            3, 1, sharex=True, figsize=(10, 12)
        )  # Increased figure size

        # TOP PLOT: Observed vs. Predicted
        # Pre-intervention fit
        h_line_pre, h_patch_pre = plot_xY(
            self.datapre.index,
            self.pre_pred["posterior_predictive"].mu,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C0", "fill_kwargs": {"alpha": 0.3}},
            label="Model Fit (Pre-treatment)",
        )
        handles = [(h_line_pre, h_patch_pre)]
        labels = ["Model Fit (Pre-treatment)"]

        # Observed data points
        (h_obs,) = ax[0].plot(
            self.datapre.index, self.pre_y, "ko", ms=3, label="Observations (Pre)"
        )
        handles.append(h_obs)
        labels.append("Observations (Pre)")
        ax[0].plot(
            self.datapost.index, self.post_y, "ko", ms=3, label="Observations (Post)"
        )  # Add to legend if distinct pre/post obs needed

        # Post-intervention counterfactual
        h_line_post, h_patch_post = plot_xY(
            self.datapost.index,
            self.post_pred["posterior_predictive"].mu,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C1", "fill_kwargs": {"alpha": 0.3}},
            label=counterfactual_label,
        )
        handles.append((h_line_post, h_patch_post))
        labels.append(counterfactual_label)

        # Shaded causal impact area
        h_impact_fill = ax[0].fill_between(
            self.datapost.index,
            y1=az.extract(
                self.post_pred, group="posterior_predictive", var_names="mu"
            ).mean("sample"),
            y2=np.squeeze(self.post_y),
            color="C2",  # Different color for impact
            alpha=0.3,
            label="Estimated Causal Impact Region",
        )
        handles.append(h_impact_fill)
        labels.append("Estimated Causal Impact Region")

        ax[0].set_title(
            f"Observed vs. Predicted {self.outcome_variable_name}\n"
            f"Pre-intervention Bayesian $R^2$: {round_num(self.score.r2, round_to)} "
            f"(std = {round_num(self.score.r2_std, round_to)})"
        )
        ax[0].set_ylabel(self.outcome_variable_name)

        # MIDDLE PLOT: Pointwise Causal Impact
        plot_xY(
            self.datapre.index,  # Show zero impact pre-treatment
            self.pre_impact,
            ax=ax[1],
            plot_hdi_kwargs={"color": "grey", "fill_kwargs": {"alpha": 0.2}},
        )
        plot_xY(
            self.datapost.index,
            self.post_impact,
            ax=ax[1],
            plot_hdi_kwargs={"color": "C2", "fill_kwargs": {"alpha": 0.3}},
        )
        ax[1].axhline(y=0, c="k", linestyle="--", linewidth=0.8)
        ax[1].set_title("Pointwise Causal Impact")
        ax[1].set_ylabel("Impact")

        # BOTTOM PLOT: Cumulative Causal Impact
        plot_xY(
            self.datapost.index,
            self.post_impact_cumulative,
            ax=ax[2],
            plot_hdi_kwargs={"color": "C2", "fill_kwargs": {"alpha": 0.3}},
        )
        ax[2].axhline(y=0, c="k", linestyle="--", linewidth=0.8)
        ax[2].set_title("Cumulative Causal Impact")
        ax[2].set_ylabel("Cumulative Impact")
        ax[2].tick_params(axis="x", rotation=45)

        # Intervention line for all subplots
        for i in [0, 1, 2]:
            ax[i].axvline(
                x=self.treatment_time,
                linestyle="-",
                linewidth=1.5,
                color="r",
                label="Treatment Time" if i == 0 else "",  # Legend only for first plot
            )

        if not any(label == "Treatment Time" for label in labels):
            handles.append(ax[0].lines[-1])  # Add treatment line to legend
            labels.append("Treatment Time")

        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.05),
            fontsize=LEGEND_FONT_SIZE - 2,
        )
        fig.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make space for legend
        return fig, ax

    def get_plot_data(self, hdi_prob: float = 0.94) -> pd.DataFrame:
        """
        Recover the data of the experiment along with the prediction and causal impact information.

        :param hdi_prob:
            Prob for which the highest density interval will be computed. The default value is defined as the default from the :func:`arviz.hdi` function.
        """
        hdi_pct = int(round(hdi_prob * 100))

        pred_lower_col = f"pred_hdi_lower_{hdi_pct}"
        pred_upper_col = f"pred_hdi_upper_{hdi_pct}"
        impact_lower_col = f"impact_hdi_lower_{hdi_pct}"
        impact_upper_col = f"impact_hdi_upper_{hdi_pct}"
        cum_impact_lower_col = f"cum_impact_hdi_lower_{hdi_pct}"
        cum_impact_upper_col = f"cum_impact_hdi_upper_{hdi_pct}"

        # Combine original y with predictions and impacts
        # Ensure index is consistent
        pre_df = pd.DataFrame(
            {
                self.outcome_variable_name: self.pre_y.flatten(),
                "prediction": az.extract(
                    self.pre_pred, group="posterior_predictive", var_names="mu"
                )
                .mean("sample")
                .values,
                "impact": self.pre_impact.mean(dim=["chain", "draw"]).values,
            },
            index=self.datapre.index,
        )

        hdi_pred_pre = get_hdi_to_df(
            self.pre_pred["posterior_predictive"].mu, hdi_prob=hdi_prob
        )
        pre_df[pred_lower_col] = hdi_pred_pre["lower"].values
        pre_df[pred_upper_col] = hdi_pred_pre["upper"].values

        hdi_impact_pre = get_hdi_to_df(self.pre_impact, hdi_prob=hdi_prob)
        pre_df[impact_lower_col] = hdi_impact_pre["lower"].values
        pre_df[impact_upper_col] = hdi_impact_pre["upper"].values

        # For pre-treatment, cumulative impact is not typically shown or is zero.
        # We can add NaNs or zeros if consistency is needed. Here, we omit it.

        post_df = pd.DataFrame(
            {
                self.outcome_variable_name: self.post_y.flatten(),
                "prediction": az.extract(
                    self.post_pred, group="posterior_predictive", var_names="mu"
                )
                .mean("sample")
                .values,
                "impact": self.post_impact.mean(dim=["chain", "draw"]).values,
                "cumulative_impact": self.post_impact_cumulative.mean(
                    dim=["chain", "draw"]
                ).values,
            },
            index=self.datapost.index,
        )

        hdi_pred_post = get_hdi_to_df(
            self.post_pred["posterior_predictive"].mu, hdi_prob=hdi_prob
        )
        post_df[pred_lower_col] = hdi_pred_post["lower"].values
        post_df[pred_upper_col] = hdi_pred_post["upper"].values

        hdi_impact_post = get_hdi_to_df(self.post_impact, hdi_prob=hdi_prob)
        post_df[impact_lower_col] = hdi_impact_post["lower"].values
        post_df[impact_upper_col] = hdi_impact_post["upper"].values

        hdi_cum_impact_post = get_hdi_to_df(
            self.post_impact_cumulative, hdi_prob=hdi_prob
        )
        post_df[cum_impact_lower_col] = hdi_cum_impact_post["lower"].values
        post_df[cum_impact_upper_col] = hdi_cum_impact_post["upper"].values

        # Concatenate pre and post dataframes
        plot_data = pd.concat([pre_df, post_df])

        # Add original exogenous regressors if they exist
        if self.pre_X_fit is not None and self.post_X_pred is not None:
            # Ensure that we are using the original column names (self.labels)
            # X_df was created with these columns. We use .values for self.pre_X_regressors etc.
            # so we need to reconstruct the DataFrame with correct column names if needed.

            # self.pre_X_regressors and self.post_X_regressors are already DataFrames with correct columns
            # from X_df_exog. So, no need to reconstruct with .values and new DataFrame call.
            X_orig_df = pd.concat([self.pre_X_regressors, self.post_X_regressors])

            # Select only the labels that were part of the exogenous regressors
            # self.labels already contains only the true exogenous regressors (no Intercept from patsy default)
            # So, if self.labels is not empty, X_orig_df[self.labels] should be correct.
            # The X_orig_df was built from X_df_exog which already had Intercept removed.

            # Simplified logic: self.labels contains the exact exogenous columns we need.
            # X_orig_df was constructed from X_df_exog which has these columns.
            if self.labels:  # If there are any exogenous regressors
                plot_data = pd.concat([plot_data, X_orig_df[self.labels]], axis=1)

        self.plot_data = plot_data
        return self.plot_data

    def plot(self, **kwargs):
        """Plot the results of the causal inference analysis.
        Since BSTS is Bayesian, this directly calls `_bayesian_plot`.
        """
        if not isinstance(self.model, BayesianStructuralTimeSeries):
            raise NotImplementedError(
                "Plotting is only supported for BayesianStructuralTimeSeries models."
            )
        return self._bayesian_plot(**kwargs)
