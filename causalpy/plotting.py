#   Copyright 2024 The PyMC Labs Developers
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
from abc import ABC

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from causalpy.plot_utils import plot_xY
from causalpy.utils import round_num

LEGEND_FONT_SIZE = 12
az.style.use("arviz-darkgrid")


class PlotComponent(ABC):
    """Abstract Base Class for PlotComponent."""

    pass


class BayesianPlotComponent(PlotComponent):
    """Plotting component for Bayesian models."""

    @staticmethod
    def plot_pre_post(results, round_to=None):
        """Generate plot for pre-post experiment types, such as Interrupted Time Series
        and Synthetic Control."""
        datapre = results.datapre
        datapost = results.datapost
        pre_y = results.pre_y
        post_y = results.post_y
        pre_pred = results.pre_pred
        post_pred = results.post_pred
        pre_impact = results.pre_impact
        post_impact = results.post_impact
        post_impact_cumulative = results.post_impact_cumulative
        treatment_time = results.treatment_time
        score = results.score

        counterfactual_label = "Counterfactual"

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))
        # TOP PLOT --------------------------------------------------
        # pre-intervention period
        h_line, h_patch = plot_xY(
            datapre.index,
            pre_pred["posterior_predictive"].mu,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C0"},
        )
        handles = [(h_line, h_patch)]
        labels = ["Pre-intervention period"]

        (h,) = ax[0].plot(datapre.index, pre_y, "k.", label="Observations")
        handles.append(h)
        labels.append("Observations")

        # post intervention period
        h_line, h_patch = plot_xY(
            datapost.index,
            post_pred["posterior_predictive"].mu,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C1"},
        )
        handles.append((h_line, h_patch))
        labels.append(counterfactual_label)

        ax[0].plot(datapost.index, post_y, "k.")
        # Shaded causal effect
        h = ax[0].fill_between(
            datapost.index,
            y1=az.extract(post_pred, group="posterior_predictive", var_names="mu").mean(
                "sample"
            ),
            y2=np.squeeze(post_y),
            color="C0",
            alpha=0.25,
        )
        handles.append(h)
        labels.append("Causal impact")

        ax[0].set(
            title=f"""
            Pre-intervention Bayesian $R^2$: {round_num(score.r2, round_to)}
            (std = {round_num(score.r2_std, round_to)})
            """
        )

        # MIDDLE PLOT -----------------------------------------------
        plot_xY(
            datapre.index,
            pre_impact,
            ax=ax[1],
            plot_hdi_kwargs={"color": "C0"},
        )
        plot_xY(
            datapost.index,
            post_impact,
            ax=ax[1],
            plot_hdi_kwargs={"color": "C1"},
        )
        ax[1].axhline(y=0, c="k")
        ax[1].fill_between(
            datapost.index,
            y1=post_impact.mean(["chain", "draw"]),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[1].set(title="Causal Impact")

        # BOTTOM PLOT -----------------------------------------------
        ax[2].set(title="Cumulative Causal Impact")
        plot_xY(
            datapost.index,
            post_impact_cumulative,
            ax=ax[2],
            plot_hdi_kwargs={"color": "C1"},
        )
        ax[2].axhline(y=0, c="k")

        # Intervention line
        for i in [0, 1, 2]:
            ax[i].axvline(
                x=treatment_time,
                ls="-",
                lw=3,
                color="r",
            )

        ax[0].legend(
            handles=(h_tuple for h_tuple in handles),
            labels=labels,
            fontsize=LEGEND_FONT_SIZE,
        )

        return fig, ax

    @staticmethod
    def plot_difference_in_differences(results, round_to=None):
        """Generate plot for difference-in-differences"""

        def _plot_causal_impact_arrow(results, ax):
            """
            draw a vertical arrow between `y_pred_counterfactual` and
            `y_pred_counterfactual`
            """
            # Calculate y values to plot the arrow between
            y_pred_treatment = (
                results.y_pred_treatment["posterior_predictive"]
                .mu.isel({"obs_ind": 1})
                .mean()
                .data
            )
            y_pred_counterfactual = (
                results.y_pred_counterfactual["posterior_predictive"].mu.mean().data
            )
            # Calculate the x position to plot at
            # Note that we force to be float to avoid a type error using np.ptp with boolean
            # values
            diff = np.ptp(
                np.array(
                    results.x_pred_treatment[results.time_variable_name].values
                ).astype(float)
            )
            x = (
                np.max(results.x_pred_treatment[results.time_variable_name].values)
                + 0.1 * diff
            )
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

        fig, ax = plt.subplots()

        # Plot raw data
        sns.scatterplot(
            results.data,
            x=results.time_variable_name,
            y=results.outcome_variable_name,
            hue=results.group_variable_name,
            alpha=1,
            legend=False,
            markers=True,
            ax=ax,
        )

        # Plot model fit to control group
        time_points = results.x_pred_control[results.time_variable_name].values
        h_line, h_patch = plot_xY(
            time_points,
            results.y_pred_control.posterior_predictive.mu,
            ax=ax,
            plot_hdi_kwargs={"color": "C0"},
            label="Control group",
        )
        handles = [(h_line, h_patch)]
        labels = ["Control group"]

        # Plot model fit to treatment group
        time_points = results.x_pred_control[results.time_variable_name].values
        h_line, h_patch = plot_xY(
            time_points,
            results.y_pred_treatment.posterior_predictive.mu,
            ax=ax,
            plot_hdi_kwargs={"color": "C1"},
            label="Treatment group",
        )
        handles.append((h_line, h_patch))
        labels.append("Treatment group")

        # Plot counterfactual - post-test for treatment group IF no treatment
        # had occurred.
        time_points = results.x_pred_counterfactual[results.time_variable_name].values
        if len(time_points) == 1:
            parts = ax.violinplot(
                az.extract(
                    results.y_pred_counterfactual,
                    group="posterior_predictive",
                    var_names="mu",
                ).values.T,
                positions=results.x_pred_counterfactual[
                    results.time_variable_name
                ].values,
                showmeans=False,
                showmedians=False,
                widths=0.2,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("C0")
                pc.set_edgecolor("None")
                pc.set_alpha(0.5)
        else:
            h_line, h_patch = plot_xY(
                time_points,
                results.y_pred_counterfactual.posterior_predictive.mu,
                ax=ax,
                plot_hdi_kwargs={"color": "C2"},
                label="Counterfactual",
            )
            handles.append((h_line, h_patch))
            labels.append("Counterfactual")

        # arrow to label the causal impact
        _plot_causal_impact_arrow(results, ax)

        # formatting
        ax.set(
            xticks=results.x_pred_treatment[results.time_variable_name].values,
            title=results._causal_impact_summary_stat(round_to),
        )
        ax.legend(
            handles=(h_tuple for h_tuple in handles),
            labels=labels,
            fontsize=LEGEND_FONT_SIZE,
        )
        return fig, ax

    # def plot_synthetic_control(self, experiment, impact):
    #     pass

    @staticmethod
    def plot_pre_post_negd(results, round_to=None):
        """Generate plot for ANOVA-like experiments with non-equivalent group designs."""
        fig, ax = plt.subplots(
            2, 1, figsize=(7, 9), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Plot raw data
        sns.scatterplot(
            x="pre",
            y="post",
            hue="group",
            alpha=0.5,
            data=results.data,
            legend=True,
            ax=ax[0],
        )
        ax[0].set(xlabel="Pretest", ylabel="Posttest")

        # plot posterior predictive of untreated
        h_line, h_patch = plot_xY(
            results.pred_xi,
            results.pred_untreated["posterior_predictive"].mu,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C0"},
            label="Control group",
        )
        handles = [(h_line, h_patch)]
        labels = ["Control group"]

        # plot posterior predictive of treated
        h_line, h_patch = plot_xY(
            results.pred_xi,
            results.pred_treated["posterior_predictive"].mu,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C1"},
            label="Treatment group",
        )
        handles.append((h_line, h_patch))
        labels.append("Treatment group")

        ax[0].legend(
            handles=(h_tuple for h_tuple in handles),
            labels=labels,
            fontsize=LEGEND_FONT_SIZE,
        )

        # Plot estimated caual impact / treatment effect
        az.plot_posterior(results.causal_impact, ref_val=0, ax=ax[1], round_to=round_to)
        ax[1].set(title="Estimated treatment effect")
        return fig, ax

    @staticmethod
    def plot_regression_discontinuity(results, round_to=None):
        """Generate plot for regression discontinuity designs."""
        fig, ax = plt.subplots()
        # Plot raw data
        sns.scatterplot(
            results.data,
            x=results.running_variable_name,
            y=results.outcome_variable_name,
            c="k",
            ax=ax,
        )

        # Plot model fit to data
        h_line, h_patch = plot_xY(
            results.x_pred[results.running_variable_name],
            results.pred["posterior_predictive"].mu,
            ax=ax,
            plot_hdi_kwargs={"color": "C1"},
        )
        handles = [(h_line, h_patch)]
        labels = ["Posterior mean"]

        # create strings to compose title
        title_info = f"{round_num(results.score.r2, round_to)} (std = {round_num(results.score.r2_std, round_to)})"
        r2 = f"Bayesian $R^2$ on all data = {title_info}"
        percentiles = results.discontinuity_at_threshold.quantile(
            [0.03, 1 - 0.03]
        ).values
        ci = (
            r"$CI_{94\%}$"
            + f"[{round_num(percentiles[0], round_to)}, {round_num(percentiles[1], round_to)}]"
        )
        discon = f"""
            Discontinuity at threshold = {round_num(results.discontinuity_at_threshold.mean(), round_to)},
            """
        ax.set(title=r2 + "\n" + discon + ci)
        # Intervention line
        ax.axvline(
            x=results.treatment_threshold,
            ls="-",
            lw=3,
            color="r",
            label="treatment threshold",
        )
        ax.legend(
            handles=(h_tuple for h_tuple in handles),
            labels=labels,
            fontsize=LEGEND_FONT_SIZE,
        )
        return (fig, ax)

    @staticmethod
    def plot_regression_kink(results, round_to=None):
        """Generate plot for regression kink designs."""
        fig, ax = plt.subplots()
        # Plot raw data
        sns.scatterplot(
            results.data,
            x=results.running_variable_name,
            y=results.outcome_variable_name,
            c="k",  # hue="treated",
            ax=ax,
        )

        # Plot model fit to data
        h_line, h_patch = plot_xY(
            results.x_pred[results.running_variable_name],
            results.pred["posterior_predictive"].mu,
            ax=ax,
            plot_hdi_kwargs={"color": "C1"},
        )
        handles = [(h_line, h_patch)]
        labels = ["Posterior mean"]

        # create strings to compose title
        title_info = f"{round_num(results.score.r2, round_to)} (std = {round_num(results.score.r2_std, round_to)})"
        r2 = f"Bayesian $R^2$ on all data = {title_info}"
        percentiles = results.gradient_change.quantile([0.03, 1 - 0.03]).values
        ci = (
            r"$CI_{94\%}$"
            + f"[{round_num(percentiles[0], round_to)}, {round_num(percentiles[1], round_to)}]"
        )
        grad_change = f"""
            Change in gradient = {round_num(results.gradient_change.mean(), round_to)},
            """
        ax.set(title=r2 + "\n" + grad_change + ci)
        # Intervention line
        ax.axvline(
            x=results.kink_point,
            ls="-",
            lw=3,
            color="r",
            label="treatment threshold",
        )
        ax.legend(
            handles=(h_tuple for h_tuple in handles),
            labels=labels,
            fontsize=LEGEND_FONT_SIZE,
        )
        return fig, ax


class OLSPlotComponent(PlotComponent):
    """Plotting component for OLS models."""

    @staticmethod
    def plot_pre_post(results, round_to=None):
        """Generate plot for pre-post experiment types, such as Interrupted Time Series
        and Synthetic Control."""
        counterfactual_label = "Counterfactual"

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

        ax[0].plot(results.datapre.index, results.pre_y, "k.")
        ax[0].plot(results.datapost.index, results.post_y, "k.")

        ax[0].plot(results.datapre.index, results.pre_pred, c="k", label="model fit")
        ax[0].plot(
            results.datapost.index,
            results.post_pred,
            label=counterfactual_label,
            ls=":",
            c="k",
        )
        ax[0].set(
            title=f"$R^2$ on pre-intervention data = {round_num(results.score, round_to)}"
        )

        ax[1].plot(results.datapre.index, results.pre_impact, "k.")
        ax[1].plot(
            results.datapost.index,
            results.post_impact,
            "k.",
            label=counterfactual_label,
        )
        ax[1].axhline(y=0, c="k")
        ax[1].set(title="Causal Impact")

        ax[2].plot(results.datapost.index, results.post_impact_cumulative, c="k")
        ax[2].axhline(y=0, c="k")
        ax[2].set(title="Cumulative Causal Impact")

        # Shaded causal effect
        ax[0].fill_between(
            results.datapost.index,
            y1=np.squeeze(results.post_pred),
            y2=np.squeeze(results.post_y),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[1].fill_between(
            results.datapost.index,
            y1=np.squeeze(results.post_impact),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )

        # Intervention line
        # TODO: make this work when treatment_time is a datetime
        for i in [0, 1, 2]:
            ax[i].axvline(
                x=results.treatment_time,
                ls="-",
                lw=3,
                color="r",
                label="Treatment time",
            )

        ax[0].legend(fontsize=LEGEND_FONT_SIZE)

        return (fig, ax)

    @staticmethod
    def plot_difference_in_differences(results, round_to=None):
        """Generate plot for difference-in-differences"""
        fig, ax = plt.subplots()

        # Plot raw data
        sns.lineplot(
            results.data,
            x=results.time_variable_name,
            y=results.outcome_variable_name,
            hue="group",
            units="unit",
            estimator=None,
            alpha=0.25,
            ax=ax,
        )
        # Plot model fit to control group
        ax.plot(
            results.x_pred_control[results.time_variable_name],
            results.y_pred_control,
            "o",
            c="C0",
            markersize=10,
            label="model fit (control group)",
        )
        # Plot model fit to treatment group
        ax.plot(
            results.x_pred_treatment[results.time_variable_name],
            results.y_pred_treatment,
            "o",
            c="C1",
            markersize=10,
            label="model fit (treament group)",
        )
        # Plot counterfactual - post-test for treatment group IF no treatment
        # had occurred.
        ax.plot(
            results.x_pred_counterfactual[results.time_variable_name],
            results.y_pred_counterfactual,
            "go",
            markersize=10,
            label="counterfactual",
        )
        # arrow to label the causal impact
        ax.annotate(
            "",
            xy=(1.05, results.y_pred_counterfactual),
            xycoords="data",
            xytext=(1.05, results.y_pred_treatment[1]),
            textcoords="data",
            arrowprops={"arrowstyle": "<->", "color": "green", "lw": 3},
        )
        ax.annotate(
            "causal\nimpact",
            xy=(
                1.05,
                np.mean(
                    [results.y_pred_counterfactual[0], results.y_pred_treatment[1]]
                ),
            ),
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
            title=f"Causal impact = {round_num(results.causal_impact, round_to)}",
        )
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        return (fig, ax)

    @staticmethod
    def plot_regression_discontinuity(results, round_to=None) -> tuple:
        """Generate plot for regression discontinuity designs."""
        fig, ax = plt.subplots()
        # Plot raw data
        sns.scatterplot(
            results.data,
            x=results.running_variable_name,
            y=results.outcome_variable_name,
            c="k",  # hue="treated",
            ax=ax,
        )
        # Plot model fit to data
        ax.plot(
            results.x_pred[results.running_variable_name],
            results.pred,
            "k",
            markersize=10,
            label="model fit",
        )
        # create strings to compose title
        r2 = f"$R^2$ on all data = {round_num(results.score, round_to)}"
        discon = f"Discontinuity at threshold = {round_num(results.discontinuity_at_threshold, round_to)}"
        ax.set(title=r2 + "\n" + discon)
        # Intervention line
        ax.axvline(
            x=results.treatment_threshold,
            ls="-",
            lw=3,
            color="r",
            label="treatment threshold",
        )
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        return (fig, ax)
