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

from causalpy.custom_exceptions import BadIndexException, ModelException
from causalpy.experiments.base import BaseExperiment
from causalpy.plot_utils import get_hdi_to_df, plot_xY
from causalpy.pymc_models import PyMCModel
from causalpy.utils import round_num

LEGEND_FONT_SIZE = 12


class UnknownTreatmentTimeHandler:
    """
    A utility class for managing data preprocessing, postprocessing,
    and plotting steps for models that infer unknown treatment times.

    This handler prepares input data for the model, extracts relevant
    outputs after inference, and structures them for further analysis
    and visualization.
    """

    def data_preprocessing(self, data, treatment_time, model):
        """
        Preprocesses the input data by constraining the model's
        treatment time inference window.
        """
        # Restrict model's treatment time inference to given range
        model.set_time_range(treatment_time, data)
        return data

    def data_postprocessing(
        self, model, data, idata, treatment_time, y, X, pre_y, pre_X
    ):
        """
        Postprocesses model outputs and input data using the inferred
        treatment time. Slices the data into pre/post segments, generates
        predictions and impact estimates, and prepares them for analysis.
        """
        # --- Return ---
        res = {}

        # --- Retrieve timeline and inferred treatment time ---
        time_var = model.time_variable_name
        timeline = data[time_var]

        tt_samples = idata.posterior["treatment_time"].values
        tt_mean = int(tt_samples.mean().item())

        # Actual timestamp (index) corresponding to inferred treatment
        tt = data[timeline == tt_mean].index[0]
        # Index of the inferred treatment time in the data
        tt_idx = data.index.get_loc(tt)
        res["treatment_time"] = tt

        # --- Slice data into pre/post-treatment ---
        res["datapre"] = data.head(tt_idx)
        res["datapost"] = data.iloc[tt_idx:]

        # --- Slice covariates into pre/post treatment time ---
        res["pre_y"] = pre_y.isel(obs_ind=slice(0, tt_idx))
        res["pre_X"] = pre_X.isel(obs_ind=slice(0, tt_idx))
        res["post_y"] = pre_y.isel(obs_ind=slice(tt_idx, None))
        res["post_X"] = pre_X.isel(obs_ind=slice(tt_idx, None))

        # --- Predict outcomes using the model ---
        pred = model.predict(X=pre_X)
        res["pre_pred"] = pred.isel(obs_ind=slice(0, tt_idx))
        res["post_pred"] = pred.isel(obs_ind=slice(tt_idx, None))

        # --- Estimate causal impact ---
        impact = model.calculate_impact(pre_y, pred)
        res["pre_impact"] = impact.isel(obs_ind=slice(0, tt_idx))
        res["post_impact"] = impact.isel(obs_ind=slice(tt_idx, None))

        # --- Create a mask to isolate post-treatment period ---
        # Timeline reshaped to match broadcasting with treatment time
        timeline_reshape = timeline.values.reshape(1, 1, -1)
        tt_broadcast = tt_samples[:, :, None].astype(int)
        mask = (timeline_reshape >= tt_broadcast).astype(int)

        # --- Compute cumulative post-treatment impact ---
        post_impact = impact * mask
        res["post_impact_cumulative"] = model.calculate_cumulative_impact(post_impact)

        return res

    def plot_treated_counterfactual(
        self, ax, handles, labels, datapre, datapost, pre_pred, post_pred
    ):
        """
        Plot the predicted post-intervention trajectory, including its
        Highest Density Interval (HDI), on the first subplot.
        """
        # Plot predicted values under treatment (with HDI)
        h_line, h_patch = plot_xY(
            datapre.index,
            pre_pred["posterior_predictive"].mu_ts,
            ax=ax[0],
            plot_hdi_kwargs={"color": "yellowgreen"},
        )

        h_line, h_patch = plot_xY(
            datapost.index,
            post_pred["posterior_predictive"].mu_ts,
            ax=ax[0],
            plot_hdi_kwargs={"color": "yellowgreen"},
        )

        handles.append((h_line, h_patch))
        labels.append("Treated counterfactual")

    def plot_impact_cumulative(self, ax, datapre, datapost, post_impact_cumulative):
        """
        Plot the cumulative causal impact over the full time series.
        """
        # Concatenate the time indices
        full_index = datapre.index.append(datapost.index)
        ax[2].set(title="Cumulative Causal Impact")
        plot_xY(
            full_index,
            post_impact_cumulative,
            ax=ax[2],
            plot_hdi_kwargs={"color": "C1"},
        )

    def plot_intervention_line(
        self, ax, model, idata, datapre, datapost, treatment_time
    ):
        """
        Draw a vertical line at the inferred treatment time and shade the HDI interval around it.
        """
        data = pd.concat([datapre, datapost])
        timeline = data[model.time_variable_name]

        # Extract the HDI (uncertainty interval) of the treatment time
        hdi = az.hdi(idata, var_names=["treatment_time"])["treatment_time"].values
        x1 = data[timeline == int(hdi[0])].index[0]
        x2 = data[timeline == int(hdi[1])].index[0]

        for i in [0, 1, 2]:
            ymin, ymax = ax[i].get_ylim()

            # Vertical line for inferred treatment time
            ax[i].plot(
                [treatment_time, treatment_time],
                [ymin, ymax],
                ls="-",
                lw=3,
                color="r",
                solid_capstyle="butt",
            )

            # Shaded region for HDI of treatment time
            ax[i].fill_betweenx(
                y=[ymin, ymax],
                x1=x1,
                x2=x2,
                alpha=0.1,
                color="r",
            )


class KnownTreatmentTimeHandler:
    """
    Handles data preprocessing, postprocessing, and plotting logic for models
    where the treatment time is known in advance.
    """

    def data_preprocessing(self, data, treatment_time, model):
        """
        Preprocess the data by selecting only the pre-treatment period for model fitting.
        """
        # Use only data before treatment for training the model
        return data[data.index < treatment_time]

    def data_postprocessing(
        self, model, data, idata, treatment_time, y, X, pre_y, pre_X
    ):
        """
        Splits data and computes predictions and causal impact metrics.
        """
        res = {
            "treatment_time": treatment_time,
            "datapre": data[data.index < treatment_time],
            "datapost": data[data.index >= treatment_time],
            "pre_y": pre_y,
            "pre_X": pre_X,
        }

        # --- Build post-treatment design matrices ---
        (new_y, new_x) = build_design_matrices(
            [y.design_info, X.design_info], res["datapost"]
        )
        post_X = np.asarray(new_x)
        post_y = np.asarray(new_y)
        post_X = xr.DataArray(
            post_X,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": res["datapost"].index,
                "coeffs": X.design_info.column_names,
            },
        )
        post_y = xr.DataArray(
            post_y[:, 0],
            dims=["obs_ind"],
            coords={"obs_ind": res["datapost"].index},
        )
        res["post_y"] = post_y
        res["post_X"] = post_X

        # --- Predictions (counterfactual under treatment) ---
        res["pre_pred"] = model.predict(X=pre_X)
        res["post_pred"] = model.predict(X=post_X)

        # --- Impacts ---
        res["pre_impact"] = model.calculate_impact(res["pre_y"], res["pre_pred"])
        res["post_impact"] = model.calculate_impact(res["post_y"], res["post_pred"])
        res["post_impact_cumulative"] = model.calculate_cumulative_impact(
            res["post_impact"]
        )

        return res

    def plot_treated_counterfactual(
        self, sax, handles, labels, datapre, datapost, pre_pred, post_pred
    ):
        """
        Placeholder method to maintain interface compatibility with UnknownTreatmentTimeHandler.
        """
        pass

    def plot_impact_cumulative(self, ax, datapre, datapost, post_impact_cumulative):
        """
        Plot the cumulative causal impact for the post-intervention period.
        """
        ax[2].set(title="Cumulative Causal Impact")
        plot_xY(
            datapost.index,
            post_impact_cumulative,
            ax=ax[2],
            plot_hdi_kwargs={"color": "C1"},
        )

    def plot_intervention_line(
        self, ax, model, idata, datapre, datapost, treatment_time
    ):
        """
        Plot a vertical line at the known treatment time on all subplots.
        """
        # --- Plot a vertical line at the known treatment time
        for i in [0, 1, 2]:
            ax[i].axvline(
                x=treatment_time, ls="-", lw=3, color="r", solid_capstyle="butt"
            )


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
        treatment_time: Union[int, float, pd.Timestamp, tuple, None],
        formula: str,
        model=None,
        **kwargs,
    ) -> None:
        super().__init__(model=model)

        # rename the index to "obs_ind"
        data.index.name = "obs_ind"
        self.input_validation(data, treatment_time, model)
        # set experiment type - usually done in subclasses
        self.expt_type = "Pre-Post Fit"

        self.treatment_time = treatment_time
        self.formula = formula

        # Getting the right handler
        if treatment_time is None or isinstance(treatment_time, tuple):
            self.handler = UnknownTreatmentTimeHandler()
        else:
            self.handler = KnownTreatmentTimeHandler()

        # Preprocessing based on handler type
        self.datapre = self.handler.data_preprocessing(
            data, self.treatment_time, self.model
        )

        y, X = dmatrices(formula, self.datapre)
        # set things up with pre-intervention data
        self.outcome_variable_name = y.design_info.column_names[0]
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.pre_y, self.pre_X = np.asarray(y), np.asarray(X)

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
            self.pre_y[:, 0],
            dims=["obs_ind"],
            coords={"obs_ind": self.datapre.index},
        )

        # fit the model to the observed (pre-intervention) data
        if isinstance(self.model, PyMCModel):
            COORDS = {"coeffs": self.labels, "obs_ind": np.arange(X.shape[0])}
            idata = self.model.fit(X=self.pre_X, y=self.pre_y, coords=COORDS)
        elif isinstance(self.model, RegressorMixin):
            self.model.fit(X=self.pre_X, y=self.pre_y)
            idata = None
        else:
            raise ValueError("Model type not recognized")

        # score the goodness of fit to the pre-intervention data
        self.score = self.model.score(X=self.pre_X, y=self.pre_y)

        # Postprocessing with handler
        results = self.handler.data_postprocessing(
            self.model, data, idata, treatment_time, y, X, self.pre_y, self.pre_X
        )

        # Inject all results into self
        for k, v in results.items():
            setattr(self, k, v)

    def input_validation(self, data, treatment_time, model):
        """Validate the input data and model formula for correctness"""
        if treatment_time is None and not hasattr(model, "set_time_range"):
            raise ModelException(
                "If treatment_time is None, provided model must have a 'set_time_range' method"
            )
        if isinstance(treatment_time, tuple) and not hasattr(model, "set_time_range"):
            raise ModelException(
                "If treatment_time is a tuple, provided model must have a 'set_time_range' method"
            )
        if isinstance(data.index, pd.DatetimeIndex) and not isinstance(
            treatment_time, (pd.Timestamp, tuple, type(None))
        ):
            raise BadIndexException(
                "If data.index is DatetimeIndex, treatment_time must be pd.Timestamp."
            )
        if not isinstance(data.index, pd.DatetimeIndex) and isinstance(
            treatment_time, (pd.Timestamp)
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
        handles = []
        labels = []

        # Treated counterfactual (only for unknown treatment time)
        self.handler.plot_treated_counterfactual(
            ax,
            handles,
            labels,
            self.datapre,
            self.datapost,
            self.pre_pred,
            self.post_pred,
        )

        # pre-intervention period
        h_line, h_patch = plot_xY(
            self.datapre.index,
            self.pre_pred["posterior_predictive"].mu,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C0"},
        )
        handles.append((h_line, h_patch))
        labels.append("Pre-intervention period")

        (h,) = ax[0].plot(self.datapre.index, self.pre_y, "k.", label="Observations")
        handles.append(h)
        labels.append("Observations")

        # post intervention period
        h_line, h_patch = plot_xY(
            self.datapost.index,
            self.post_pred["posterior_predictive"].mu,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C1"},
        )
        handles.append((h_line, h_patch))
        labels.append(counterfactual_label)

        ax[0].plot(self.datapost.index, self.post_y, "k.")
        # Shaded causal effect
        h = ax[0].fill_between(
            self.datapost.index,
            y1=az.extract(
                self.post_pred, group="posterior_predictive", var_names="mu"
            ).mean("sample"),
            y2=np.squeeze(self.post_y),
            color="C0",
            alpha=0.25,
        )
        handles.append(h)
        labels.append("Causal impact")

        ax[0].set(
            title=f"""
            Pre-intervention Bayesian $R^2$: {round_num(self.score.r2, round_to)}
            (std = {round_num(self.score.r2_std, round_to)})
            """
        )

        # MIDDLE PLOT -----------------------------------------------
        plot_xY(
            self.datapre.index,
            self.pre_impact,
            ax=ax[1],
            plot_hdi_kwargs={"color": "C0"},
        )
        plot_xY(
            self.datapost.index,
            self.post_impact,
            ax=ax[1],
            plot_hdi_kwargs={"color": "C1"},
        )
        ax[1].axhline(y=0, c="k")
        ax[1].fill_between(
            self.datapost.index,
            y1=self.post_impact.mean(["chain", "draw"]),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[1].set(title="Causal Impact")

        # BOTTOM PLOT -----------------------------------------------
        self.handler.plot_impact_cumulative(
            ax, self.datapre, self.datapost, self.post_impact_cumulative
        )
        ax[2].axhline(y=0, c="k")

        # Plot vertical line marking treatment time (with HDI if it's inferred)
        self.handler.plot_intervention_line(
            ax, self.model, self.idata, self.datapre, self.datapost, self.treatment_time
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

            pre_data["prediction"] = (
                az.extract(self.pre_pred, group="posterior_predictive", var_names="mu")
                .mean("sample")
                .values
            )
            post_data["prediction"] = (
                az.extract(self.post_pred, group="posterior_predictive", var_names="mu")
                .mean("sample")
                .values
            )
            pre_data[[pred_lower_col, pred_upper_col]] = get_hdi_to_df(
                self.pre_pred["posterior_predictive"].mu, hdi_prob=hdi_prob
            ).set_index(pre_data.index)
            post_data[[pred_lower_col, pred_upper_col]] = get_hdi_to_df(
                self.post_pred["posterior_predictive"].mu, hdi_prob=hdi_prob
            ).set_index(post_data.index)

            pre_data["impact"] = self.pre_impact.mean(dim=["chain", "draw"]).values
            post_data["impact"] = self.post_impact.mean(dim=["chain", "draw"]).values
            pre_data[[impact_lower_col, impact_upper_col]] = get_hdi_to_df(
                self.pre_impact, hdi_prob=hdi_prob
            ).set_index(pre_data.index)
            post_data[[impact_lower_col, impact_upper_col]] = get_hdi_to_df(
                self.post_impact, hdi_prob=hdi_prob
            ).set_index(post_data.index)

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

    def plot_treatment_time(self):
        """
        display the posterior estimates of the treatment time
        """
        if "treatment_time" not in self.idata.posterior.data_vars:
            raise ValueError(
                "Variable 'treatment_time' not found in inference data (idata)."
            )

        az.plot_trace(self.idata, var_names="treatment_time")
