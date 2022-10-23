import matplotlib.pyplot as plt
import numpy as np
from patsy import dmatrices, build_design_matrices
import seaborn as sns
import pandas as pd
from causal_impact.plot_utils import plot_xY

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
        COORDS = {"coeffs": self.labels, "obs_indx": np.arange(self.pre_X.shape[0])}
        self.prediction_model.fit(X=self.pre_X, y=self.pre_y, coords=COORDS)

        # score the goodness of fit to the pre-intervention data
        self.score = self.prediction_model.score(X=self.pre_X, y=self.pre_y)

        # get the model predictions of the observed (pre-intervention) data
        self.pre_pred = self.prediction_model.predict(X=self.pre_X)

        # calculate the counterfactual
        self.post_pred = self.prediction_model.predict(X=self.post_X)

        # TODO
        # # causal impact pre (ie the residuals of the model fit to observed)
        # self.pre_impact = self.pre_y - self.pre_pred
        # # causal impact post (ie the impact of the intervention)
        # self.post_impact = self.post_y - self.post_pred

        # # cumulative impact post
        # self.post_impact_cumulative = np.cumsum(self.post_impact)

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
    pass
