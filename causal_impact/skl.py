import matplotlib.pyplot as plt
import numpy as np
from patsy import dmatrices, build_design_matrices
import seaborn as sns
import pandas as pd

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

        # fit the model to the observed (pre-intervention) data
        self.prediction_model.fit(X=self.pre_X, y=self.pre_y)

        # score the goodness of fit to the pre-intervention data
        self.score = self.prediction_model.score(X=self.pre_X, y=self.pre_y)

        # get the model predictions of the observed (pre-intervention) data
        self.pre_pred = self.prediction_model.predict(X=self.pre_X)

        # calculate the counterfactual
        self.post_pred = self.prediction_model.predict(X=self.post_X)

        # causal impact pre (ie the residuals of the model fit to observed)
        self.pre_impact = self.pre_y - self.pre_pred
        # causal impact post (ie the impact of the intervention)
        self.post_impact = self.post_y - self.post_pred

        # cumulative impact post
        self.post_impact_cumulative = np.cumsum(self.post_impact)

    def plot(self):
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

        ax[0].plot(self.datapre.index, self.pre_y, "k.")
        ax[0].plot(self.datapost.index, self.post_y, "k.")

        ax[0].plot(self.datapre.index, self.pre_pred, c="k", label="model fit")
        ax[0].plot(
            self.datapost.index,
            self.post_pred,
            label="counterfactual",
            ls=":",
            c="k",
        )
        ax[0].set(title=f"$R^2$ on pre-intervention data = {self.score:.3f}")

        ax[1].plot(self.datapre.index, self.pre_impact, "k.")
        ax[1].plot(
            self.datapost.index,
            self.post_impact,
            "k.",
            label="counterfactual",
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
            label="causal impact",
        )
        ax[1].fill_between(
            self.datapost.index,
            y1=np.squeeze(self.post_impact),
            color="C0",
            alpha=0.25,
            label="causal impact",
        )

        # Intervention line
        # TODO: make this work when self.treatment_time is a datetime
        for i in [0, 1, 2]:
            ax[i].axvline(
                x=self.treatment_time,
                ls="-",
                lw=3,
                color="r",
                label="treatment time",
            )

        ax[0].legend(fontsize=LEGEND_FONT_SIZE)

        return (fig, ax)


# InterruptedTimeSeries and SyntheticControl are basically the same thing but with different
# predictor variables. So we just have a TimeSeriesExperiment class and InterruptedTimeSeries
# and SyntheticControl are both equal to the TimeSeriesExperiment class


class InterruptedTimeSeries(TimeSeriesExperiment):
    pass


class SyntheticControl(TimeSeriesExperiment):
    def plot(self):
        fig, ax = super().plot()
        # plot control units as well
        ax[0].plot(self.datapre.index, self.pre_X, "-", c=[0.8, 0.8, 0.8], zorder=1)
        ax[0].plot(self.datapost.index, self.post_X, "-", c=[0.8, 0.8, 0.8], zorder=1)
        return (fig, ax)


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

        # fit the model to all the data
        self.prediction_model.fit(X=self.X, y=self.y)

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
        ax.plot(
            self.x_pred_control[self.time_variable_name],
            self.y_pred_control,
            "o",
            c="C0",
            markersize=10,
            label="model fit (control group)",
        )
        # Plot model fit to treatment group
        ax.plot(
            self.x_pred_treatment[self.time_variable_name],
            self.y_pred_treatment,
            "o",
            c="C1",
            markersize=10,
            label="model fit (treament group)",
        )
        # Plot counterfactual - post-test for treatment group IF no treatment had occurred.
        ax.plot(
            self.x_pred_counterfactual[self.time_variable_name],
            self.y_pred_counterfactual,
            "go",
            markersize=10,
            label="counterfactual",
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

        # fit the model to all the data
        self.prediction_model.fit(X=self.X, y=self.y)

        # score the goodness of fit to all data
        self.score = self.prediction_model.score(X=self.X, y=self.y)

        # get the model predictions of the observed data
        xi = np.linspace(np.min(self.data["x"]), np.max(self.data["x"]), 1000)
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
        fig, ax = plt.subplots()
        # Plot raw data
        sns.scatterplot(
            self.data,
            x=self.running_variable_name,
            y=self.outcome_variable_name,
            c="k",  # hue="treated",
            ax=ax,
        )
        # Plot model fit to data
        ax.plot(
            self.x_pred[self.running_variable_name],
            self.pred,
            "k",
            markersize=10,
            label="model fit",
        )
        # Plot counterfactual
        ax.plot(
            self.x_counterfact[self.running_variable_name],
            self.pred_counterfac,
            markersize=10,
            ls=":",
            c="k",
            label="counterfactual",
        )
        # Shaded causal effect
        ax.fill_between(
            self.x_counterfact[self.running_variable_name],
            y1=np.squeeze(self.pred_counterfac),
            y2=np.squeeze(self.pred[-len(np.squeeze(self.pred_counterfac)) :]),
            color="C0",
            alpha=0.25,
            label="inferred causal impact",
        )
        # Intervention line
        ax.axvline(
            x=self.treatment_threshold,
            ls="-",
            lw=3,
            color="r",
            label="treatment threshold",
        )
        ax.set(title=f"$R^2$ on all data = {self.score:.3f}")
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        return (fig, ax)
