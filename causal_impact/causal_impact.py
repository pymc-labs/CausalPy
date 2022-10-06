import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from copy import copy
from causal_impact.plot_utils import plot_xY, format_x_axis


class CausalBase:
    """Base class that does core counterfactual inference tasks"""

    def __init__(
        self, df, treatment_date, target_var, predictor_vars, RANDOM_SEED=1234
    ) -> None:
        self.df = copy(df)
        self.treatment_date = treatment_date
        self.target_var = target_var
        self.predictor_vars = predictor_vars
        self.n_predictors = len(self.predictor_vars)

        # validate inputs
        if type(self.df.index) is pd.DatetimeIndex:
            if type(self.treatment_date) is not pd.Timestamp:
                raise Exception(
                    "If index of dataframe is a timeseries, then treatment_date needs to be a timeseries type."
                )

        # create indicator column of pre/post
        self.df["pre"] = self.df.index < self.treatment_date

        # split into separate dataframes for pre and post treatment
        self.pre = self.df[self.df.index < self.treatment_date]
        self.post = self.df[self.df.index >= self.treatment_date]

        # build model
        self.model = self.build_model()

        # PRIOR PREDICTIVE CHECK
        print("PRIOR PREDICTIVE CHECK")
        with self.model:
            self.idata_pre = pm.sample_prior_predictive(random_seed=RANDOM_SEED)

        # INFERENCE
        print("INFERENCE")
        with self.model:
            self.idata_pre.extend(
                pm.sample(random_seed=RANDOM_SEED, target_accept=0.95, tune=2000)
            )

        # POSTERIOR PREDICTIVE CHECK
        print("POSTERIOR PREDICTIVE CHECK")
        with self.model:
            self.idata_pre.extend(
                pm.sample_posterior_predictive(self.idata_pre, random_seed=RANDOM_SEED)
            )

        # COUNTERFACTUAL
        print("COUNTERFACTUAL INFERENCE")
        with self.model:
            pm.set_data(
                {
                    "X": self.post[self.predictor_vars].to_numpy(),
                    "y": self.post[self.target_var].to_numpy(),
                }
            )
            self.idata_post = pm.sample_posterior_predictive(
                self.idata_pre, var_names=["obs"], random_seed=RANDOM_SEED
            )

        # Calculate causal impact (difference between observed and counterfactual) and cumulative
        self.calc_causal_impact()

    def build_model(self):
        raise NotImplementedError("Please implement this method")

    def plot(self):
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        self.plot_data_and_fit(ax[0])
        self.plot_causal_impact(ax[1])
        self.plot_causal_impact_cumulative(ax[2])

    def plot_data_and_fit(self, ax=None):
        if ax is None:
            ax = plt.gca()
        # synthetic control: pre
        plot_xY(self.pre.index, self.idata_pre.posterior_predictive["obs"], ax)
        # synthetic control: post
        plot_xY(self.post.index, self.idata_post.posterior_predictive["obs"], ax)
        # plot observed data
        ax.plot(self.df.index, self.df[self.target_var], label=self.target_var)
        # formatting
        ax.axvline(x=self.treatment_date, linewidth=3, c="k", ls="--")
        ax.legend()
        format_x_axis(ax)
        ax.set(title="Data and Counterfactual")

    def plot_causal_impact(self, ax=None):
        if ax is None:
            ax = plt.gca()
        """Plot the inferred causal impact (aka lift analysis)"""
        plot_xY(self.pre.index, self.causal_impact_pre, ax)
        plot_xY(self.post.index, self.causal_impact_post, ax)
        ax.axvline(x=self.treatment_date, linewidth=3, c="k", ls="--")
        format_x_axis(ax)
        ax.axhline(y=0, color="k")
        ax.set(title="Lift analysis / Causal Impact")

    def plot_causal_impact_cumulative(self, ax=None):
        if ax is None:
            ax = plt.gca()
        """Plot the inferred causal impact (aka lift analysis), but cumulative over time,since the intervention."""
        plot_xY(self.post.index, self.post_cumulative_impact, ax)
        ax.axvline(x=self.treatment_date, linewidth=3, c="k", ls="--")
        ax.set(title="Lift analysis / Causal Impact (Cumulative)")

    def calc_causal_impact(self):
        # POST -----
        # convert deaths into an XArray object with a labelled dimension to help in the next step
        post_data = xr.DataArray(
            self.post[self.target_var].to_numpy(), dims=["obs_dim_0"]
        )
        # do the calculation by taking the difference
        self.causal_impact_post = (
            post_data - self.idata_post.posterior_predictive["obs"]
        ).transpose(..., "obs_dim_0")

        # PRE -----
        pre_data = xr.DataArray(
            self.pre[self.target_var].to_numpy(), dims=["obs_dim_0"]
        )
        # do the calculation by taking the difference
        self.causal_impact_pre = (
            pre_data - self.idata_pre.posterior_predictive["obs"]
        ).transpose(..., "obs_dim_0")

        # CUMULATIVE IMPACT: post -----
        self.post_cumulative_impact = self.causal_impact_post.cumsum(dim="obs_dim_0")


# class SyntheticControl(CausalBase):
#     """This model is intended for use in synthetic control contexts. It's main feature is that it has a Dirichlet prior over the coefficients."""

#     def build_model(self):
#         COORDS = {
#             "predictors": self.predictor_vars,
#             "obs": np.arange(self.pre.shape[0]),
#         }

#         with pm.Model(coords=COORDS) as model:
#             # observed predictors and outcome
#             X = pm.MutableData("X", self.pre[self.predictor_vars].to_numpy())
#             y = pm.MutableData("y", self.pre[self.target_var].to_numpy())
#             # priors
#             beta = pm.Dirichlet(
#                 "beta",
#                 a=np.ones(self.n_predictors),
#                 dims="predictors",
#             )
#             #  linear model
#             mu = pm.Deterministic("mu", pm.math.dot(X, beta))
#             sigma = pm.HalfNormal("sigma", 1)
#             # likelihood
#             pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

#         return model


class LinearModel(CausalBase):
    """Simple linear regression model. Useful for interrupted time series contexts with predictor variables"""

    def build_model(self):
        COORDS = {
            "predictors": self.predictor_vars,
            "obs": np.arange(self.pre.shape[0]),
        }

        with pm.Model(coords=COORDS) as model:
            # observed predictors and outcome
            X = pm.MutableData("X", self.pre[self.predictor_vars].to_numpy())
            y = pm.MutableData("y", self.pre[self.target_var].to_numpy())
            # priors
            beta = pm.Normal("beta", mu=0, sigma=2, dims="predictors")
            #  linear model
            mu = pm.Deterministic("mu", pm.math.dot(X, beta))
            sigma = pm.HalfNormal("sigma", 1)
            # likelihood
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        return model
