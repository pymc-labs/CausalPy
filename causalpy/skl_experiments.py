import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from patsy import build_design_matrices, dmatrices

LEGEND_FONT_SIZE = 12


class ExperimentalDesign:
    """Base class for experiment designs"""

    model = None
    outcome_variable_name = None

    def __init__(self, model=None, **kwargs):
        if model is not None:
            self.model = model
        if self.model is None:
            raise ValueError("fitting_model not set or passed.")


class PrePostFit(ExperimentalDesign):
    """A class to analyse quasi-experiments where parameter estimation is based on just
    the pre-intervention data."""

    def __init__(
        self,
        data,
        treatment_time,
        formula,
        model=None,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
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
        self.outcome_variable_name = y.design_info.column_names[0]
        self.pre_y, self.pre_X = np.asarray(y), np.asarray(X)
        # process post-intervention data
        (new_y, new_x) = build_design_matrices(
            [self._y_design_info, self._x_design_info], self.datapost
        )
        self.post_X = np.asarray(new_x)
        self.post_y = np.asarray(new_y)

        # fit the model to the observed (pre-intervention) data
        self.model.fit(X=self.pre_X, y=self.pre_y)

        # score the goodness of fit to the pre-intervention data
        self.score = self.model.score(X=self.pre_X, y=self.pre_y)

        # get the model predictions of the observed (pre-intervention) data
        self.pre_pred = self.model.predict(X=self.pre_X)

        # calculate the counterfactual
        self.post_pred = self.model.predict(X=self.post_X)

        # causal impact pre (ie the residuals of the model fit to observed)
        self.pre_impact = self.pre_y - self.pre_pred
        # causal impact post (ie the impact of the intervention)
        self.post_impact = self.post_y - self.post_pred

        # cumulative impact post
        self.post_impact_cumulative = np.cumsum(self.post_impact)

    def plot(self, counterfactual_label="Counterfactual", **kwargs):
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
        ax[0].set(title=f"$R^2$ on pre-intervention data = {self.score:.3f}")

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
        # TODO: make this work when self.treatment_time is a datetime
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

    def get_coeffs(self):
        return np.squeeze(self.model.coef_)

    def plot_coeffs(self):
        df = pd.DataFrame(
            {"predictor variable": self.labels, "ols_coef": self.get_coeffs()}
        )
        sns.barplot(
            data=df,
            x="ols_coef",
            y="predictor variable",
            palette=sns.color_palette("husl"),
        )


class InterruptedTimeSeries(PrePostFit):
    """Interrupted time series analysis"""

    expt_type = "Interrupted Time Series"


class SyntheticControl(PrePostFit):
    """A wrapper around the PrePostFit class"""

    def plot(self, plot_predictors=False, **kwargs):
        """Plot the results"""
        fig, ax = super().plot(counterfactual_label="Synthetic control", **kwargs)
        if plot_predictors:
            # plot control units as well
            ax[0].plot(self.datapre.index, self.pre_X, "-", c=[0.8, 0.8, 0.8], zorder=1)
            ax[0].plot(
                self.datapost.index, self.post_X, "-", c=[0.8, 0.8, 0.8], zorder=1
            )
        return (fig, ax)


class DifferenceInDifferences(ExperimentalDesign):
    """
    .. note::

        There is no pre/post intervention data distinction for DiD, we fit all the data
        available.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        time_variable_name: str,
        group_variable_name: str,
        treated: str,
        untreated: str,
        model=None,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.data = data
        self.formula = formula
        self.time_variable_name = time_variable_name
        self.group_variable_name = group_variable_name
        self.treated = treated  # level of the group_variable_name that was treated
        self.untreated = (
            untreated  # level of the group_variable_name that was untreated
        )
        y, X = dmatrices(formula, self.data)
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.y, self.X = np.asarray(y), np.asarray(X)
        self.outcome_variable_name = y.design_info.column_names[0]

        # fit the model to all the data
        self.model.fit(X=self.X, y=self.y)

        # predicted outcome for control group
        self.x_pred_control = (
            self.data
            # just the untreated group
            .query(f"{self.group_variable_name} == @self.untreated")
            # drop the outcome variable
            .drop(self.outcome_variable_name, axis=1)
            # We may have multiple units per time point, we only want one time point
            .groupby(self.time_variable_name)
            .first()
            .reset_index()
        )
        assert not self.x_pred_control.empty
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred_control)
        self.y_pred_control = self.model.predict(np.asarray(new_x))

        # predicted outcome for treatment group
        self.x_pred_treatment = (
            self.data
            # just the treated group
            .query(f"{self.group_variable_name} == @self.treated")
            # drop the outcome variable
            .drop(self.outcome_variable_name, axis=1)
            # We may have multiple units per time point, we only want one time point
            .groupby(self.time_variable_name)
            .first()
            .reset_index()
        )
        assert not self.x_pred_treatment.empty
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred_treatment)
        self.y_pred_treatment = self.model.predict(np.asarray(new_x))

        # predicted outcome for counterfactual. This is given by removing the influence
        # of the interaction term between the group and the post_treatment variable
        self.x_pred_counterfactual = (
            self.data
            # just the treated group
            .query(f"{self.group_variable_name} == @self.treated")
            # just the treatment period(s)
            .query("post_treatment == True")
            # drop the outcome variable
            .drop(self.outcome_variable_name, axis=1)
            # We may have multiple units per time point, we only want one time point
            .groupby(self.time_variable_name)
            .first()
            .reset_index()
        )
        assert not self.x_pred_counterfactual.empty
        (new_x,) = build_design_matrices(
            [self._x_design_info], self.x_pred_counterfactual, return_type="dataframe"
        )
        # INTERVENTION: set the interaction term between the group and the
        # post_treatment variable to zero. This is the counterfactual.
        for i, label in enumerate(self.labels):
            if "post_treatment" in label and self.group_variable_name in label:
                new_x.iloc[:, i] = 0
        self.y_pred_counterfactual = self.model.predict(np.asarray(new_x))

        # calculate causal impact
        # This is the coefficient on the interaction term
        # TODO: THIS IS NOT YET CORRECT
        self.causal_impact = self.y_pred_treatment[1] - self.y_pred_counterfactual[0]

    def plot(self):
        """Plot results"""
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
        # Plot counterfactual - post-test for treatment group IF no treatment
        # had occurred.
        ax.plot(
            self.x_pred_counterfactual[self.time_variable_name],
            self.y_pred_counterfactual,
            "go",
            markersize=10,
            label="counterfactual",
        )
        # arrow to label the causal impact
        ax.annotate(
            "",
            xy=(1.05, self.y_pred_counterfactual),
            xycoords="data",
            xytext=(1.05, self.y_pred_treatment[1]),
            textcoords="data",
            arrowprops={"arrowstyle": "<->", "color": "green", "lw": 3},
        )
        ax.annotate(
            "causal\nimpact",
            xy=(1.05, np.mean([self.y_pred_counterfactual, self.y_pred_treatment[1]])),
            xycoords="data",
            xytext=(5, 0),
            textcoords="offset points",
            color="green",
            va="center",
        )
        # formatting
        ax.set(
            xlim=[-0.05, 1.1],
            xticks=[0, 1],
            xticklabels=["pre", "post"],
            title=f"Causal impact = {self.causal_impact[0]:.2f}",
        )
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        return (fig, ax)


class RegressionDiscontinuity(ExperimentalDesign):
    """
    Analyse data from regression discontinuity experiments.

    .. note::

        There is no pre/post intervention data distinction for the regression
        discontinuity design, we fit all the data available.

    """

    def __init__(
        self,
        data,
        formula,
        treatment_threshold,
        model=None,
        running_variable_name="x",
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.data = data
        self.formula = formula
        self.running_variable_name = running_variable_name
        self.treatment_threshold = treatment_threshold
        y, X = dmatrices(formula, self.data)
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.y, self.X = np.asarray(y), np.asarray(X)
        self.outcome_variable_name = y.design_info.column_names[0]

        # TODO: `treated` is a deterministic function of x and treatment_threshold, so
        # this could be a function rather than supplied data

        # fit the model to all the data
        self.model.fit(X=self.X, y=self.y)

        # score the goodness of fit to all data
        self.score = self.model.score(X=self.X, y=self.y)

        # get the model predictions of the observed data
        xi = np.linspace(
            np.min(self.data[self.running_variable_name]),
            np.max(self.data[self.running_variable_name]),
            1000,
        )
        self.x_pred = pd.DataFrame(
            {self.running_variable_name: xi, "treated": self._is_treated(xi)}
        )
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred)
        self.pred = self.model.predict(X=np.asarray(new_x))

        # calculate discontinuity by evaluating the difference in model expectation on
        # either side of the discontinuity
        # NOTE: `"treated": np.array([0, 1])`` assumes treatment is applied above
        # (not below) the threshold
        self.x_discon = pd.DataFrame(
            {
                self.running_variable_name: np.array(
                    [self.treatment_threshold - 0.001, self.treatment_threshold + 0.001]
                ),
                "treated": np.array([0, 1]),
            }
        )
        (new_x,) = build_design_matrices([self._x_design_info], self.x_discon)
        self.pred_discon = self.model.predict(X=np.asarray(new_x))
        self.discontinuity_at_threshold = np.squeeze(self.pred_discon[1]) - np.squeeze(
            self.pred_discon[0]
        )

    def _is_treated(self, x):
        """Returns ``True`` if ``x`` is greater than or equal to the treatment
        threshold.

        .. warning::

            Assumes treatment is given to those ABOVE the treatment threshold.
        """
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
        # create strings to compose title
        r2 = f"$R^2$ on all data = {self.score:.3f}"
        discon = f"Discontinuity at threshold = {self.discontinuity_at_threshold:.2f}"
        ax.set(title=r2 + "\n" + discon)
        # Intervention line
        ax.axvline(
            x=self.treatment_threshold,
            ls="-",
            lw=3,
            color="r",
            label="treatment threshold",
        )
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        return (fig, ax)

    def summary(self):
        """Print text output summarising the results"""
        print("Difference in Differences experiment")
        print(f"Formula: {self.formula}")
        print(f"Running variable: {self.running_variable_name}")
        print(f"Threshold on running variable: {self.treatment_threshold}")
        print("\nResults:")
        print(f"Discontinuity at threshold = {self.discontinuity_at_threshold:.2f}")
        print("Model coefficients:")
        for name, val in zip(self.labels, self.model.coef_[0]):
            print(f"\t{name}\t\t{val}")
