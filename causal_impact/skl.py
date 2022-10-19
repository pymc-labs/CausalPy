import matplotlib.pyplot as plt
import numpy as np
from patsy import dmatrices, build_design_matrices
import seaborn as sns
import pandas as pd


class PlotterMixin:
    def plot(self):
        fig, ax = plt.subplots(3, 1, sharex=True)

        ax[0].plot(self.datapre.index, self.pre_y, "ko")
        ax[0].plot(self.datapost.index, self.post_y, "ko")

        ax[0].plot(
            self.datapre.index, self.pre_pred, label="model fit to observed data"
        )
        ax[0].plot(
            self.datapost.index, self.post_pred, label="estimated counterfactual"
        )

        ax[0].legend()

        ax[1].plot(self.datapre.index, self.pre_impact)
        ax[1].plot(self.datapost.index, self.post_impact)
        ax[1].axhline(y=0, c="k")
        ax[1].set(title="Causal Impact")

        ax[2].plot(self.datapost.index, self.post_impact_cumulative)
        ax[2].set(title="Cumulative Causal Impact")


class ExperimentalDesign:
    prediction_model = None

    def __init__(self, prediction_model=None, **kwargs):
        if prediction_model is not None:
            self.prediction_model = prediction_model
        if self.prediction_model is None:
            raise ValueError("fitting_model not set or passed.")


class InterruptedTimeSeries(ExperimentalDesign, PlotterMixin):
    def __init__(self, datapre, datapost, prediction_model=None, **kwargs):
        super().__init__(prediction_model=prediction_model, **kwargs)
        self.datapre = datapre
        self.datapost = datapost

        # extract the data we need
        # NOTE: the X predictors here is literally just [0, 1, 2, 3, ...]
        self.pre_X = self.datapre["linear_trend"].values.reshape(-1, 1)
        self.pre_y = self.datapre["timeseries"].values
        self.post_X = self.datapost["linear_trend"].values.reshape(-1, 1)
        self.post_y = self.datapost["timeseries"].values

        # fit the model to the observed (pre-intervention) data
        self.prediction_model.fit(X=self.pre_X, y=self.pre_y)

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


class SyntheticControl(ExperimentalDesign, PlotterMixin):
    def __init__(
        self, datapre, datapost, target_col, controls, prediction_model=None, **kwargs
    ):
        super().__init__(prediction_model=prediction_model, **kwargs)
        self.datapre = datapre
        self.datapost = datapost
        self.target_col = target_col
        self.controls = controls

        # extract the data we need
        self.pre_X = self.datapre[self.controls].values
        self.pre_y = self.datapre[self.target_col].values.reshape(-1, 1)
        self.post_X = self.datapost[self.controls].values
        self.post_y = self.datapost[self.target_col].values.reshape(-1, 1)

        # fit the model to the observed (pre-intervention) data
        self.prediction_model.fit(X=self.pre_X, y=self.pre_y)

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


class DifferenceInDifferences(ExperimentalDesign):
    """Note: there is no 'predict data' for DiD, we fit all the data available."""

    def __init__(self, data, formula, prediction_model=None, **kwargs):
        super().__init__(prediction_model=prediction_model, **kwargs)
        self.data = data
        self.formula = formula
        y, X = dmatrices(formula, data)
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
        self.y_pred_control = np.dot(np.asarray(new_x), self.prediction_model.coef_.T)

        # predicted outcome for treatment group
        self.x_pred_treatment = pd.DataFrame(
            {"group": [1, 1], "t": [0.0, 1.0], "treated": [0, 1]}
        )
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred_treatment)
        self.y_pred_treatment = np.dot(np.asarray(new_x), self.prediction_model.coef_.T)

        # predicted outcome for counterfactual
        self.x_pred_counterfactual = pd.DataFrame(
            {"group": [1], "t": [1.0], "treated": [0]}
        )
        (new_x,) = build_design_matrices(
            [self._x_design_info], self.x_pred_counterfactual
        )
        self.y_pred_counterfactual = np.dot(
            np.asarray(new_x), self.prediction_model.coef_.T
        )

        # self.pre_pred = self.prediction_model.predict(X=self.X)

        # # calculate the counterfactual
        # self.post_pred = self.prediction_model.predict(X=self.post_X)

        # # causal impact pre (ie the residuals of the model fit to observed)
        # self.pre_impact = self.pre_y - self.pre_pred
        # # causal impact post (ie the impact of the intervention)
        # self.post_impact = self.post_y - self.post_pred

        # # cumulative impact post
        # self.post_impact_cumulative = np.cumsum(self.post_impact)

    def plot(self):
        fig, ax = plt.subplots()

        # plot raw data
        sns.lineplot(
            self.data, x="t", y="y", hue="group", units="unit", estimator=None, ax=ax
        )
        sns.scatterplot(self.data, x="t", y="y", hue="group", ax=ax)

        # #Plot model fit to control group
        ax.plot(
            self.x_pred_control["t"],
            self.y_pred_control,
            "ko",
            markersize=20,
            alpha=0.5,
            label="model fit (control group)",
        )

        # Plot model fit to treatment group
        ax.plot(
            self.x_pred_treatment["t"],
            self.y_pred_treatment,
            "ro",
            markersize=20,
            alpha=0.5,
            label="model fit (treament group)",
        )

        # Plot counterfactual - post-test for treatment group IF no treatment had occurred.
        ax.plot(
            self.x_pred_counterfactual["t"],
            self.y_pred_counterfactual,
            "go",
            markersize=20,
            alpha=0.5,
            label="model fit (treament group)",
        )

        # ax.legend()
