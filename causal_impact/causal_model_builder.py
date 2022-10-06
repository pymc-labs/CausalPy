import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from causal_impact.plot_utils import plot_xY, format_x_axis
from pymc_experimental.model_builder import ModelBuilder
from typing import Dict, Union


class CausalModelBuilder(ModelBuilder):
    def predict(
        self,
        data_prediction: Dict[str, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        point_estimate: bool = True,
    ):
        """
        Uses model to predict on unseen data.
        Parameters
        ---------
        data_prediction : Dictionary of string and either of numpy array, pandas dataframe or pandas Series
            It is the data we need to make prediction on using the model.
        point_estimate : bool
            Adds point like estimate used as mean passed as
        Returns
        -------
        returns dictionary of sample's posterior predict.
        Examples
        --------
        >>> data, model_config, sampler_config = LinearModel.create_sample_input()
        >>> model = LinearModel(model_config, sampler_config)
        >>> idata = model.fit(data)
        >>> x_pred = []
        >>> prediction_data = pd.DataFrame({'input':x_pred})
        # only point estimate
        >>> pred_mean = model.predict(prediction_data)
        # samples
        >>> pred_samples = model.predict(prediction_data, point_estimate=False)
        """

        # STEP 1 - PREDICT
        if data_prediction is not None:  # set new input data
            self._data_setter(data_prediction)

        with self.model:  # sample with new input data
            self.idata_predict = pm.sample_posterior_predictive(self.idata)

        # TODO STEP 2 - CALCULATE CAUSAL IMPACT
        self._calc_causal_impact()

        # TODO: DO WE WANT THIS EXPORTING OF POINT ESTIMATE?
        # reshape output
        post_pred = self._extract_samples(self.idata_predict)
        if point_estimate:  # average, if point-like estimate desired
            for key in post_pred:
                post_pred[key] = post_pred[key].mean(axis=0)

        return post_pred

    def _calc_causal_impact(self):
        target_var = self.model_config["target_var"]
        # POST -----
        post_data = xr.DataArray(self.idata_predict.constant_data.y, dims=["y_dim_0"])
        post_prediction = self.idata_predict.posterior_predictive["y_model"].rename(
            {"y_model_dim_0": "y_dim_0"}
        )
        self.causal_impact_post = (post_data - post_prediction).transpose(
            ..., "y_dim_0"
        )

        # CUMULATIVE IMPACT: post -----
        self.post_cumulative_impact = self.causal_impact_post.cumsum(dim="y_dim_0")

        # PRE -----
        pre_data = xr.DataArray(self.idata.constant_data.y, dims=["y_dim_0"])
        pre_prediction = self.idata.posterior_predictive["y_model"].rename(
            {"y_model_dim_0": "y_dim_0"}
        )
        # do the calculation by taking the difference
        self.causal_impact_pre = (pre_data - pre_prediction).transpose(..., "y_dim_0")

    def plot(self):
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        self.plot_data_and_fit(ax[0])
        self.plot_causal_impact(ax[1])
        self.plot_causal_impact_cumulative(ax[2])

    def plot_data_and_fit(self, ax=None):
        if ax is None:
            ax = plt.gca()
        # synthetic control: pre ----------------------------------------------
        # TODO: we really want the original dataframe index for the x-axis
        x_vals = np.arange(self.idata.posterior_predictive["y_model"].shape[2])
        plot_xY(x_vals, self.idata.posterior_predictive["y_model"], ax)
        # synthetic control: post
        # TODO: we really want the original dataframe index for the x-axis
        x_vals = np.max(x_vals) + np.arange(
            self.idata_predict.posterior_predictive["y_model"].shape[2]
        )
        plot_xY(x_vals, self.idata_predict.posterior_predictive["y_model"], ax)

        ax.set(title="Data and Counterfactual")

    def plot_causal_impact(self, ax=None):
        if ax is None:
            ax = plt.gca()
        """Plot the inferred causal impact (aka lift analysis)"""
        # TODO: we really want the original dataframe index for the x-axis
        x_vals = np.arange(self.idata.posterior_predictive["y_model"].shape[2])
        plot_xY(x_vals, self.causal_impact_pre, ax)

        # TODO: we really want the original dataframe index for the x-axis
        x_vals = np.max(x_vals) + np.arange(
            self.idata_predict.posterior_predictive["y_model"].shape[2]
        )
        plot_xY(x_vals, self.causal_impact_post, ax)

        # ax.axvline(x=self.treatment_date, linewidth=3, c="k", ls="--")
        # format_x_axis(ax)
        ax.axhline(y=0, color="k")
        ax.set(title="Lift analysis / Causal Impact")

    def plot_causal_impact_cumulative(self, ax=None):
        if ax is None:
            ax = plt.gca()
        """Plot the inferred causal impact (aka lift analysis), but cumulative over time,since the intervention."""
        # TODO: we really want the original dataframe index for the x-axis
        x_vals = self.idata.posterior_predictive["y_model"].shape[2] + np.arange(
            self.idata_predict.posterior_predictive["y_model"].shape[2]
        )
        plot_xY(x_vals, self.post_cumulative_impact, ax)
        # ax.axvline(x=self.treatment_date, linewidth=3, c="k", ls="--")
        ax.set(title="Lift analysis / Causal Impact (Cumulative)")
