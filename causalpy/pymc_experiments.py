import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from patsy import build_design_matrices, dmatrices

from causalpy.plot_utils import plot_xY

LEGEND_FONT_SIZE = 12
az.style.use("arviz-darkgrid")


class ExperimentalDesign:
    """Base class"""

    prediction_model = None
    expt_type = None

    def __init__(self, prediction_model=None, **kwargs):
        if prediction_model is not None:
            self.prediction_model = prediction_model
        if self.prediction_model is None:
            raise ValueError("fitting_model not set or passed.")

    @property
    def idata(self):
        """Access to the InferenceData object"""
        return self.prediction_model.idata

    def print_coefficients(self):
        """Prints the model coefficients"""
        print("Model coefficients:")
        coeffs = az.extract(self.idata.posterior, var_names="beta")
        # Note: f"{name: <30}" pads the name with spaces so that we have alignment of
        # the stats despite variable names of different lengths
        for name in self.labels:
            coeff_samples = coeffs.sel(coeffs=name)
            print(
                f"{name: <30}{coeff_samples.mean().data:.2f}, 94% HDI [{coeff_samples.quantile(0.03).data:.2f}, {coeff_samples.quantile(1-0.03).data:.2f}]"  # noqa: E501
            )
        # add coeff for measurement std
        coeff_samples = az.extract(
            self.prediction_model.idata.posterior, var_names="sigma"
        )
        name = "sigma"
        print(
            f"{name: <30}{coeff_samples.mean().data:.2f}, 94% HDI [{coeff_samples.quantile(0.03).data:.2f}, {coeff_samples.quantile(1-0.03).data:.2f}]"  # noqa: E501
        )


class TimeSeriesExperiment(ExperimentalDesign):
    """A class to analyse time series quasi-experiments"""

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_time: int,
        formula: str,
        prediction_model=None,
        **kwargs,
    ) -> None:
        super().__init__(prediction_model=prediction_model, **kwargs)
        self.treatment_time = treatment_time
        # split data in to pre and post intervention
        self.datapre = data[data.index <= self.treatment_time]
        self.datapost = data[data.index > self.treatment_time]

        self.formula = formula

        # set things up with pre-intervention data
        y, X = dmatrices(formula, self.datapre)
        self.outcome_variable_name = y.design_info.column_names[0]
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
        self.pre_impact = (
            pre_data - self.pre_pred["posterior_predictive"].y_hat
        ).transpose(..., "obs_ind")

        # causal impact post (ie the residuals of the model fit to observed)
        post_data = xr.DataArray(self.post_y[:, 0], dims=["obs_ind"])
        self.post_impact = (
            post_data - self.post_pred["posterior_predictive"].y_hat
        ).transpose(..., "obs_ind")

        # cumulative impact post
        self.post_impact_cumulative = self.post_impact.cumsum(dim="obs_ind")

    def plot(self):

        """Plot the results"""
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

        # TOP PLOT --------------------------------------------------
        # pre-intervention period
        plot_xY(
            self.datapre.index,
            self.pre_pred["posterior_predictive"].y_hat,
            ax=ax[0],
        )
        ax[0].plot(self.datapre.index, self.pre_y, "k.", label="Observations")
        # post intervention period
        plot_xY(
            self.datapost.index,
            self.post_pred["posterior_predictive"].y_hat,
            ax=ax[0],
            include_label=False,
        )
        ax[0].plot(self.datapost.index, self.post_y, "k.")
        # Shaded causal effect
        ax[0].fill_between(
            self.datapost.index,
            y1=az.extract(
                self.post_pred, group="posterior_predictive", var_names="y_hat"
            ).mean("sample"),
            y2=np.squeeze(self.post_y),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[0].set(
            title=f"""
            Pre-intervention Bayesian $R^2$: {self.score.r2:.3f}
            (std = {self.score.r2_std:.3f})
            """
        )

        # MIDDLE PLOT -----------------------------------------------
        plot_xY(
            self.datapre.index,
            self.pre_impact,
            ax=ax[1],
        )
        plot_xY(
            self.datapost.index,
            self.post_impact,
            ax=ax[1],
            include_label=False,
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

        ax[2].set(title="Cumulative Causal Impact")
        plot_xY(
            self.datapost.index,
            self.post_impact_cumulative,
            ax=ax[2],
        )
        ax[2].axhline(y=0, c="k")

        # Intervention line
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

    def summary(self):
        """Print text output summarising the results"""

        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        # TODO: extra experiment specific outputs here
        self.print_coefficients()


class SyntheticControl(TimeSeriesExperiment):
    """A wrapper around the TimeSeriesExperiment class"""

    expt_type = "Synthetic Control"

    def plot(self, plot_predictors=False):
        """Plot the results"""
        fig, ax = super().plot()
        if plot_predictors:
            # plot control units as well
            ax[0].plot(self.datapre.index, self.pre_X, "-", c=[0.8, 0.8, 0.8], zorder=1)
            ax[0].plot(
                self.datapost.index, self.post_X, "-", c=[0.8, 0.8, 0.8], zorder=1
            )
        return (fig, ax)


class DifferenceInDifferences(ExperimentalDesign):
    """A class to analyse data from Difference in Difference settings.

    .. note::

        There is no pre/post intervention data distinction for DiD, we fit all the
        data available.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        time_variable_name: str,
        group_variable_name: str,
        treated: str,
        untreated: str,
        prediction_model=None,
        **kwargs,
    ):
        super().__init__(prediction_model=prediction_model, **kwargs)
        self.data = data
        self.expt_type = "Difference in Differences"
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

        # Input validation ----------------------------------------------------
        # Check that `treated` appears in the module formula
        assert (
            "treated" in formula
        ), "A predictor column called `treated` should be in the provided dataframe"
        # Check that we have `treated` in the incoming dataframe
        assert (
            "treated" in self.data.columns
        ), "Require a boolean column labelling observations which are `treated`"
        # Check for `unit` in the incoming dataframe.
        # *This is only used for plotting purposes*
        assert (
            "unit" in self.data.columns
        ), """
        Require a `unit` column to label unique units.
        This is used for plotting purposes
        """
        # Check that `group_variable_name` has TWO levels, representing the
        # treated/untreated. But it does not matter what the actual names of
        # the levels are.
        assert (
            len(pd.Categorical(self.data[self.group_variable_name]).categories) == 2
        ), f"""
            There must be 2 levels of the grouping variable {self.group_variable_name}
            .I.e. the treated and untreated.
        """

        # TODO: `treated` is a deterministic function of group and time, so this could
        # be a function rather than supplied data

        # DEVIATION FROM SKL EXPERIMENT CODE =============================
        # fit the model to the observed (pre-intervention) data
        COORDS = {"coeffs": self.labels, "obs_indx": np.arange(self.X.shape[0])}
        self.prediction_model.fit(X=self.X, y=self.y, coords=COORDS)
        # ================================================================

        time_levels = self.data[self.time_variable_name].unique()

        # predicted outcome for control group
        self.x_pred_control = pd.DataFrame(
            {
                self.group_variable_name: [self.untreated, self.untreated],
                self.time_variable_name: time_levels,
                "treated": [0, 0],
            }
        )
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred_control)
        self.y_pred_control = self.prediction_model.predict(np.asarray(new_x))

        # predicted outcome for treatment group
        self.x_pred_treatment = pd.DataFrame(
            {
                self.group_variable_name: [self.treated, self.treated],
                self.time_variable_name: time_levels,
                "treated": [0, 1],
            }
        )
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred_treatment)
        self.y_pred_treatment = self.prediction_model.predict(np.asarray(new_x))

        # predicted outcome for counterfactual
        self.x_pred_counterfactual = pd.DataFrame(
            {
                self.group_variable_name: [self.treated],
                self.time_variable_name: time_levels[1],
                "treated": [0],
            }
        )
        (new_x,) = build_design_matrices(
            [self._x_design_info], self.x_pred_counterfactual
        )
        self.y_pred_counterfactual = self.prediction_model.predict(np.asarray(new_x))

        # calculate causal impact
        self.causal_impact = (
            self.y_pred_treatment["posterior_predictive"].mu.isel({"obs_ind": 1})
            - self.y_pred_counterfactual["posterior_predictive"].mu.squeeze()
        )
        # self.causal_impact = (
        #     self.y_pred_treatment["posterior_predictive"]
        #     .mu.isel({"obs_ind": 1})
        #     .stack(samples=["chain", "draw"])
        #     - self.y_pred_counterfactual["posterior_predictive"]
        #     .mu.stack(samples=["chain", "draw"])
        #     .squeeze()
        # )

    def plot(self):
        """Plot the results"""
        fig, ax = plt.subplots()

        # Plot raw data
        # NOTE: This will not work when there is just ONE unit in each group
        sns.lineplot(
            self.data,
            x=self.time_variable_name,
            y=self.outcome_variable_name,
            hue=self.group_variable_name,
            units="unit",  # NOTE: assumes we have a `unit` predictor variable
            estimator=None,
            alpha=0.5,
            ax=ax,
        )
        # Plot model fit to control group
        parts = ax.violinplot(
            az.extract(
                self.y_pred_control, group="posterior_predictive", var_names="mu"
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
                self.y_pred_treatment, group="posterior_predictive", var_names="mu"
            ).values.T,
            positions=self.x_pred_treatment[self.time_variable_name].values,
            showmeans=False,
            showmedians=False,
            widths=0.2,
        )

        for pc in parts["bodies"]:
            pc.set_facecolor("C1")
            pc.set_edgecolor("None")
            pc.set_alpha(0.5)
        # Plot counterfactual - post-test for treatment group IF no treatment
        # had occurred.
        parts = ax.violinplot(
            az.extract(
                self.y_pred_counterfactual,
                group="posterior_predictive",
                var_names="mu",
            ).values.T,
            positions=self.x_pred_counterfactual[self.time_variable_name].values,
            showmeans=False,
            showmedians=False,
            widths=0.2,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("C2")
            pc.set_edgecolor("None")
            pc.set_alpha(0.5)
        # arrow to label the causal impact
        self._plot_causal_impact_arrow(ax)
        # formatting
        ax.set(
            xticks=self.x_pred_treatment[self.time_variable_name].values,
            title=self._causal_impact_summary_stat(),
        )
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        return (fig, ax)

    def _plot_causal_impact_arrow(self, ax):
        """
        draw a vertical arrow between `y_pred_counterfactual` and
        `y_pred_counterfactual`
        """
        # Calculate y values to plot the arrow between
        y_pred_treatment = (
            self.y_pred_treatment["posterior_predictive"]
            .mu.isel({"obs_ind": 1})
            .mean()
            .data
        )
        y_pred_counterfactual = (
            self.y_pred_counterfactual["posterior_predictive"].mu.mean().data
        )
        # Calculate the x position to plot at
        diff = np.ptp(self.x_pred_treatment[self.time_variable_name].values)
        x = np.max(self.x_pred_treatment[self.time_variable_name].values) + 0.1 * diff
        # Plot the arrow
        ax.annotate(
            "",
            xy=(x, y_pred_counterfactual),
            xycoords="data",
            xytext=(x, y_pred_treatment),
            textcoords="data",
            arrowprops={"arrowstyle": "<-", "color": "green", "lw": 3},
        )
        # Plot text annotation next to arrow
        ax.annotate(
            "causal\nimpact",
            xy=(x, np.mean([y_pred_counterfactual, y_pred_treatment])),
            xycoords="data",
            xytext=(5, 0),
            textcoords="offset points",
            color="green",
            va="center",
        )

    def _causal_impact_summary_stat(self):
        percentiles = self.causal_impact.quantile([0.03, 1 - 0.03]).values
        ci = r"$CI_{94\%}$" + f"[{percentiles[0]:.2f}, {percentiles[1]:.2f}]"
        causal_impact = f"{self.causal_impact.mean():.2f}, "
        return f"Causal impact = {causal_impact + ci}"

    def summary(self):
        """Print text output summarising the results"""

        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print("\nResults:")
        # TODO: extra experiment specific outputs here
        print(self._causal_impact_summary_stat())
        self.print_coefficients()


class RegressionDiscontinuity(ExperimentalDesign):
    """
    A class to analyse regression discontinuity experiments.

    :param data: A pandas dataframe
    :param formula: A statistical model formula
    :param treatment_threshold: A scalar threshold value at which the treatment
                                is applied
    :param prediction_model: A PyMC model
    :param running_variable_name: The name of the predictor variable that the treatment
                                  threshold is based upon

    .. note::

        There is no pre/post intervention data distinction for the regression
        discontinuity design, we fit all the data available.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        treatment_threshold: float,
        prediction_model=None,
        running_variable_name: str = "x",
        **kwargs,
    ):
        super().__init__(prediction_model=prediction_model, **kwargs)
        self.expt_type = "Regression Discontinuity"
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

        # DEVIATION FROM SKL EXPERIMENT CODE =============================
        # fit the model to the observed (pre-intervention) data
        COORDS = {"coeffs": self.labels, "obs_indx": np.arange(self.X.shape[0])}
        self.prediction_model.fit(X=self.X, y=self.y, coords=COORDS)
        # ================================================================

        # score the goodness of fit to all data
        self.score = self.prediction_model.score(X=self.X, y=self.y)

        # get the model predictions of the observed data
        xi = np.linspace(
            np.min(self.data[self.running_variable_name]),
            np.max(self.data[self.running_variable_name]),
            200,
        )
        self.x_pred = pd.DataFrame(
            {self.running_variable_name: xi, "treated": self._is_treated(xi)}
        )
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred)
        self.pred = self.prediction_model.predict(X=np.asarray(new_x))

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
        self.pred_discon = self.prediction_model.predict(X=np.asarray(new_x))
        self.discontinuity_at_threshold = (
            self.pred_discon["posterior_predictive"].sel(obs_ind=1)["mu"]
            - self.pred_discon["posterior_predictive"].sel(obs_ind=0)["mu"]
        )

    def _is_treated(self, x):
        """Returns ``True`` if `x` is greater than or equal to the treatment threshold.

        .. warning::

            Assumes treatment is given to those ABOVE the treatment threshold.
        """
        return np.greater_equal(x, self.treatment_threshold)

    def plot(self):
        """Plot the results"""
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
        plot_xY(
            self.x_pred[self.running_variable_name],
            self.pred["posterior_predictive"].mu,
            ax=ax,
        )
        # create strings to compose title
        title_info = f"{self.score.r2:.3f} (std = {self.score.r2_std:.3f})"
        r2 = f"Bayesian $R^2$ on all data = {title_info}"
        percentiles = self.discontinuity_at_threshold.quantile([0.03, 1 - 0.03]).values
        ci = r"$CI_{94\%}$" + f"[{percentiles[0]:.2f}, {percentiles[1]:.2f}]"
        discon = f"""
            Discontinuity at threshold = {self.discontinuity_at_threshold.mean():.2f},
            """
        ax.set(title=r2 + "\n" + discon + ci)
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

        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print(f"Running variable: {self.running_variable_name}")
        print(f"Threshold on running variable: {self.treatment_threshold}")
        print("\nResults:")
        print(
            f"Discontinuity at threshold = {self.discontinuity_at_threshold.mean():.2f}"
        )
        self.print_coefficients()


class PrePostNEGD(ExperimentalDesign):
    """A class to analyse data from pretest/posttest designs"""

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        group_variable_name: str,
        pretreatment_variable_name: str,
        prediction_model=None,
        **kwargs,
    ):
        super().__init__(prediction_model=prediction_model, **kwargs)
        self.data = data
        self.expt_type = "Pretest/posttest Nonequivalent Group Design"
        self.formula = formula
        self.group_variable_name = group_variable_name
        self.pretreatment_variable_name = pretreatment_variable_name

        y, X = dmatrices(formula, self.data)
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.y, self.X = np.asarray(y), np.asarray(X)
        self.outcome_variable_name = y.design_info.column_names[0]

        # Input validation ----------------------------------------------------
        # Check that `group_variable_name` has TWO levels, representing the
        # treated/untreated. But it does not matter what the actual names of
        # the levels are.
        assert (
            len(pd.Categorical(self.data[self.group_variable_name]).categories) == 2
        ), f"""
            There must be 2 levels of the grouping variable {self.group_variable_name}
            .I.e. the treated and untreated.
        """

        # fit the model to the observed (pre-intervention) data
        COORDS = {"coeffs": self.labels, "obs_indx": np.arange(self.X.shape[0])}
        self.prediction_model.fit(X=self.X, y=self.y, coords=COORDS)

        # Calculate the posterior predictive for the treatment and control for an
        # interpolated set of pretest values
        # get the model predictions of the observed data
        self.pred_xi = np.linspace(
            np.min(self.data[self.pretreatment_variable_name]),
            np.max(self.data[self.pretreatment_variable_name]),
            200,
        )
        # untreated
        x_pred_untreated = pd.DataFrame(
            {
                self.pretreatment_variable_name: self.pred_xi,
                self.group_variable_name: np.zeros(self.pred_xi.shape),
            }
        )
        (new_x,) = build_design_matrices([self._x_design_info], x_pred_untreated)
        self.pred_untreated = self.prediction_model.predict(X=np.asarray(new_x))
        # treated
        x_pred_untreated = pd.DataFrame(
            {
                self.pretreatment_variable_name: self.pred_xi,
                self.group_variable_name: np.ones(self.pred_xi.shape),
            }
        )
        (new_x,) = build_design_matrices([self._x_design_info], x_pred_untreated)
        self.pred_treated = self.prediction_model.predict(X=np.asarray(new_x))

        # Evaluate causal impact as equal to the trestment effect
        self.causal_impact = self.idata.posterior["beta"].sel(
            {"coeffs": self._get_treatment_effect_coeff()}
        )

        # ================================================================

    def plot(self):
        """Plot the results"""
        fig, ax = plt.subplots(
            2, 1, figsize=(7, 9), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Plot raw data
        sns.scatterplot(
            x="pre",
            y="post",
            hue="group",
            alpha=0.5,
            data=self.data,
            ax=ax[0],
        )
        ax[0].set(xlabel="Pretest", ylabel="Posttest")

        # plot posterior predictive of untreated
        plot_xY(
            self.pred_xi,
            self.pred_untreated["posterior_predictive"].y_hat,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C0"},
        )

        # plot posterior predictive of treated
        plot_xY(
            self.pred_xi,
            self.pred_treated["posterior_predictive"].y_hat,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C1"},
        )

        ax[0].legend(fontsize=LEGEND_FONT_SIZE)

        # Plot estimated caual impact / treatment effect
        az.plot_posterior(self.causal_impact, ref_val=0, ax=ax[1])
        ax[1].set(title="Estimated treatment effect")
        return fig, ax

    def _causal_impact_summary_stat(self):
        percentiles = self.causal_impact.quantile([0.03, 1 - 0.03]).values
        ci = r"$CI_{94\%}$" + f"[{percentiles[0]:.2f}, {percentiles[1]:.2f}]"
        causal_impact = f"{self.causal_impact.mean():.2f}, "
        return f"Causal impact = {causal_impact + ci}"

    def summary(self):
        """Print text output summarising the results"""

        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print("\nResults:")
        # TODO: extra experiment specific outputs here
        print(self._causal_impact_summary_stat())
        self.print_coefficients()

    def _get_treatment_effect_coeff(self) -> str:
        """Find the beta regression coefficient corresponding to the
        group (i.e. treatment) effect.
        For example if self.group_variable_name is 'group' and
        the labels are `['Intercept', 'C(group)[T.1]', 'pre']`
        then we want `C(group)[T.1]`.
        """
        for label in self.labels:
            if ("group" in label) & (":" not in label):
                return label

        raise NameError("Unable to find coefficient name for the treatment effect")
