import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from patsy import build_design_matrices, dmatrices

from causalpy.plot_utils import plot_xY

LEGEND_FONT_SIZE = 12


class ExperimentalDesign:
    prediction_model = None

    def __init__(self, prediction_model=None, **kwargs):
        if prediction_model is not None:
            self.prediction_model = prediction_model
        if self.prediction_model is None:
            raise ValueError("fitting_model not set or passed.")


class TimeSeriesExperiment(ExperimentalDesign):
    def __init__(self, data, treatment_time, formula, prediction_model=None, **kwargs):
        super().__init__(prediction_model=prediction_model, **kwargs)
        self.treatment_time = treatment_time
        # split data in to pre and post intervention
        self.datapre = data[data.index <= self.treatment_time]
        self.datapost = data[data.index > self.treatment_time]

        self.formula = formula

        # set things up with pre-intervention data
        y, X = dmatrices(formula, self.datapre)
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

        # DEVIATION FROM SKL EXPERIMENT CODE =============================
        # fit the model to the observed (pre-intervention) data
        COORDS = {"coeffs": self.labels, "obs_indx": np.arange(self.pre_X.shape[0])}
        self.prediction_model.fit(X=self.pre_X, y=self.pre_y, coords=COORDS)
        # ================================================================

        # score the goodness of fit to the pre-intervention data
        self.score = self.prediction_model.score(X=self.pre_X, y=self.pre_y)

        # get the model predictions of the observed (pre-intervention) data
        self.pre_pred = self.prediction_model.predict(X=self.pre_X)

        # calculate the counterfactual
        self.post_pred = self.prediction_model.predict(X=self.post_X)

        # causal impact pre (ie the residuals of the model fit to observed)
        pre_data = xr.DataArray(self.pre_y[:, 0], dims=["obs_ind"])
        self.pre_impact = pre_data - self.pre_pred["posterior_predictive"].y_hat

        # causal impact post (ie the residuals of the model fit to observed)
        post_data = xr.DataArray(self.post_y[:, 0], dims=["obs_ind"])
        self.post_impact = post_data - self.post_pred["posterior_predictive"].y_hat

        # cumulative impact post
        self.post_impact_cumulative = self.post_impact.cumsum(dim="obs_ind")

    def plot(self):
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

        # pre-intervention period
        plot_xY(
            self.datapre.index, self.pre_pred["posterior_predictive"].y_hat, ax=ax[0]
        )
        ax[0].plot(self.datapre.index, self.pre_y, "k.")
        # post intervention period
        plot_xY(
            self.datapost.index, self.post_pred["posterior_predictive"].y_hat, ax=ax[0]
        )
        ax[0].plot(self.datapost.index, self.post_y, "k.")
        ax[0].set(title=f"$R^2$ on pre-intervention data = {self.score:.3f}")

        plot_xY(self.datapre.index, self.pre_impact, ax=ax[1])
        plot_xY(self.datapost.index, self.post_impact, ax=ax[1])
        ax[1].axhline(y=0, c="k")
        ax[1].set(title="Causal Impact")

        ax[2].set(title="Cumulative Causal Impact")
        plot_xY(self.datapost.index, self.post_impact_cumulative, ax=ax[2])

        # Intervention line
        for i in [0, 1, 2]:
            ax[i].axvline(
                x=self.treatment_time,
                ls="-",
                lw=3,
                color="r",
                label="treatment time",
            )
        return (fig, ax)


class SyntheticControl(TimeSeriesExperiment):
    def plot(self):
        fig, ax = super().plot()
        # plot control units as well
        ax[0].plot(self.datapre.index, self.pre_X, "-", c=[0.8, 0.8, 0.8], zorder=1)
        ax[0].plot(self.datapost.index, self.post_X, "-", c=[0.8, 0.8, 0.8], zorder=1)
        return (fig, ax)


class InterruptedTimeSeries(TimeSeriesExperiment):
    pass


class DifferenceInDifferences(ExperimentalDesign):
    """Note: there is no pre/post intervention data distinction for DiD, we fit all the data available."""

    def __init__(
        self,
        data,
        formula,
        time_variable_name="t",
        outcome_variable_name="y",
        prediction_model=None,
        **kwargs,
    ):
        super().__init__(prediction_model=prediction_model, **kwargs)
        self.data = data
        self.formula = formula
        self.time_variable_name = time_variable_name
        self.outcome_variable_name = outcome_variable_name
        y, X = dmatrices(formula, self.data)
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.y, self.X = np.asarray(y), np.asarray(X)

        # TODO: `treated` is a deterministic function of group and time, so this should be a function rather than supplied data

        # DEVIATION FROM SKL EXPERIMENT CODE =============================
        # fit the model to the observed (pre-intervention) data
        COORDS = {"coeffs": self.labels, "obs_indx": np.arange(self.X.shape[0])}
        self.prediction_model.fit(X=self.X, y=self.y, coords=COORDS)
        # ================================================================

        # predicted outcome for control group
        self.x_pred_control = pd.DataFrame(
            {"group": [0, 0], "t": [0.0, 1.0], "treated": [0, 0]}
        )
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred_control)
        self.y_pred_control = self.prediction_model.predict(np.asarray(new_x))

        # predicted outcome for treatment group
        self.x_pred_treatment = pd.DataFrame(
            {"group": [1, 1], "t": [0.0, 1.0], "treated": [0, 1]}
        )
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred_treatment)
        self.y_pred_treatment = self.prediction_model.predict(np.asarray(new_x))

        # predicted outcome for counterfactual
        self.x_pred_counterfactual = pd.DataFrame(
            {"group": [1], "t": [1.0], "treated": [0]}
        )
        (new_x,) = build_design_matrices(
            [self._x_design_info], self.x_pred_counterfactual
        )
        self.y_pred_counterfactual = self.prediction_model.predict(np.asarray(new_x))

        # calculate causal impact
        # TODO: This should most likely be posterior estimate, not posterior predictive
        self.causal_impact = (
            self.y_pred_treatment["posterior_predictive"]
            .y_hat.isel({"obs_ind": 1})
            .mean()
            .data
            - self.y_pred_counterfactual["posterior_predictive"].y_hat.mean().data
        )

    def plot(self):
        fig, ax = plt.subplots()

        # Plot raw data
        sns.lineplot(
            self.data,
            x=self.time_variable_name,
            y=self.outcome_variable_name,
            hue="group",
            units="unit",
            estimator=None,
            alpha=0.25,
            ax=ax,
        )
        # Plot model fit to control group
        parts = ax.violinplot(
            az.extract(
                self.y_pred_control, group="posterior_predictive", var_names="y_hat"
            ).values.T,
            positions=self.x_pred_control[self.time_variable_name].values,
            showmeans=False,
            showmedians=False,
            widths=0.2,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("C0")
            pc.set_edgecolor("None")
            pc.set_alpha(0.5)

        # Plot model fit to treatment group
        parts = ax.violinplot(
            az.extract(
                self.y_pred_treatment, group="posterior_predictive", var_names="y_hat"
            ).values.T,
            positions=self.x_pred_treatment[self.time_variable_name].values,
            showmeans=False,
            showmedians=False,
            widths=0.2,
        )
        # Plot counterfactual - post-test for treatment group IF no treatment had occurred.
        parts = ax.violinplot(
            az.extract(
                self.y_pred_counterfactual,
                group="posterior_predictive",
                var_names="y_hat",
            ).values.T,
            positions=self.x_pred_counterfactual[self.time_variable_name].values,
            showmeans=False,
            showmedians=False,
            widths=0.2,
        )
        # arrow to label the causal impact
        y_pred_treatment = (
            self.y_pred_treatment["posterior_predictive"]
            .y_hat.isel({"obs_ind": 1})
            .mean()
            .data
        )
        y_pred_counterfactual = (
            self.y_pred_counterfactual["posterior_predictive"].y_hat.mean().data
        )
        ax.annotate(
            "",
            xy=(1.15, y_pred_counterfactual),
            xycoords="data",
            xytext=(1.15, y_pred_treatment),
            textcoords="data",
            arrowprops={"arrowstyle": "<->", "color": "green", "lw": 3},
        )
        ax.annotate(
            "causal\nimpact",
            xy=(1.15, np.mean([y_pred_counterfactual, y_pred_treatment])),
            xycoords="data",
            xytext=(5, 0),
            textcoords="offset points",
            color="green",
            va="center",
        )
        # formatting
        ax.set(
            xlim=[-0.15, 1.25],
            xticks=[0, 1],
            xticklabels=["pre", "post"],
            title=f"Causal impact = {self.causal_impact:.2f}",
        )
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        return (fig, ax)


class RegressionDiscontinuity(ExperimentalDesign):
    """Note: there is no pre/post intervention data distinction, we fit all the data available."""

    def __init__(
        self,
        data,
        formula,
        treatment_threshold,
        prediction_model=None,
        running_variable_name="x",
        outcome_variable_name="y",
        **kwargs,
    ):
        super().__init__(prediction_model=prediction_model, **kwargs)
        self.data = data
        self.formula = formula
        self.running_variable_name = running_variable_name
        self.outcome_variable_name = outcome_variable_name
        self.treatment_threshold = treatment_threshold
        y, X = dmatrices(formula, self.data)
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.y, self.X = np.asarray(y), np.asarray(X)

        # TODO: `treated` is a deterministic function of x and treatment_threshold, so this could be a function rather than supplied data

        # DEVIATION FROM SKL EXPERIMENT CODE =============================
        # fit the model to the observed (pre-intervention) data
        COORDS = {"coeffs": self.labels, "obs_indx": np.arange(self.X.shape[0])}
        self.prediction_model.fit(X=self.X, y=self.y, coords=COORDS)
        # ================================================================

        # score the goodness of fit to all data
        self.score = self.prediction_model.score(X=self.X, y=self.y)

        # get the model predictions of the observed data
        xi = np.linspace(np.min(self.data["x"]), np.max(self.data["x"]), 200)
        self.x_pred = pd.DataFrame({"x": xi, "treated": self._is_treated(xi)})
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred)
        self.pred = self.prediction_model.predict(X=np.asarray(new_x))

        # calculate the counterfactual
        xi = xi[xi > self.treatment_threshold]
        self.x_counterfact = pd.DataFrame({"x": xi, "treated": np.zeros(xi.shape)})
        (new_x,) = build_design_matrices([self._x_design_info], self.x_counterfact)
        self.pred_counterfac = self.prediction_model.predict(X=np.asarray(new_x))

    def _is_treated(self, x):
        return np.greater_equal(x, self.treatment_threshold)

    def plot(self):
        fig, ax = plt.subplots(2, 1, figsize=(7, 8))
        # Plot raw data
        sns.scatterplot(
            self.data,
            x=self.running_variable_name,
            y=self.outcome_variable_name,
            c="k",  # hue="treated",
            ax=ax[0],
        )
        # Plot model fit to data
        plot_xY(
            self.x_pred[self.running_variable_name],
            self.pred["posterior_predictive"].y_hat,
            ax=ax[0],
        )
        # # Plot counterfactual
        plot_xY(
            self.x_counterfact[self.running_variable_name],
            self.pred_counterfac["posterior_predictive"].y_hat,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C2"},
        )
        # Shaded causal effect
        # TODO
        # Intervention line
        ax[0].axvline(
            x=self.treatment_threshold,
            ls="-",
            lw=3,
            color="r",
            label="treatment threshold",
        )
        ax[0].set(title=f"$R^2$ on all data = {self.score:.3f}")
        ax[0].legend(fontsize=LEGEND_FONT_SIZE)

        # Plot causal effect estimate ------------------------
        coeff_name = (
            "treated[T.True]"  # NOTE: get rid of this hard coded variable name!
        )
        beta = self.prediction_model.idata["posterior"]["beta"].sel(
            {"coeffs": coeff_name}
        )
        az.plot_posterior(beta, ref_val=0, ax=ax[1])
        ax[1].set(title=f"Causal impact", xlabel=coeff_name)
        return (fig, ax)
