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

from causalpy.plot_utils import plot_xY
from causalpy.utils import round_num

LEGEND_FONT_SIZE = 12
az.style.use("arviz-darkgrid")


class PlotComponent(ABC):
    pass


class BayesianPlotComponent(PlotComponent):
    def plot_pre_post(self, results, round_to=None):
        # unpack results dictionary
        datapre = results["datapre"]
        datapost = results["datapost"]
        pre_y = results["pre_y"]
        post_y = results["post_y"]
        pre_pred = results["pre_pred"]
        post_pred = results["post_pred"]
        pre_impact = results["pre_impact"]
        post_impact = results["post_impact"]
        post_impact_cumulative = results["post_impact_cumulative"]
        treatment_time = results["treatment_time"]
        score = results["score"]

        counterfactual_label = "Counterfactual"
        # datapre, datapost, pre_y, post_y, pre_pred, treatment_time
        print("BayesianPlotComponent - plot_pre_post")

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

    def plot_difference_in_differences(self, experiment, impact):
        pass

    def plot_synthetic_control(self, experiment, impact):
        pass


class OLSPlotComponent(PlotComponent):
    def plot_pre_post(self, results, round_to=None):
        # unpack results dictionary
        datapre = results["datapre"]
        datapost = results["datapost"]
        pre_y = results["pre_y"]
        post_y = results["post_y"]
        pre_pred = results["pre_pred"]
        post_pred = results["post_pred"]
        pre_impact = results["pre_impact"]
        post_impact = results["post_impact"]
        post_impact_cumulative = results["post_impact_cumulative"]
        treatment_time = results["treatment_time"]
        score = results["score"]

        counterfactual_label = "Counterfactual"

        print("OLSPlotComponent - plot_pre_post")

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

        ax[0].plot(datapre.index, pre_y, "k.")
        ax[0].plot(datapost.index, post_y, "k.")

        ax[0].plot(datapre.index, pre_pred, c="k", label="model fit")
        ax[0].plot(
            datapost.index,
            post_pred,
            label=counterfactual_label,
            ls=":",
            c="k",
        )
        ax[0].set(
            title=f"$R^2$ on pre-intervention data = {round_num(score, round_to)}"
        )

        ax[1].plot(datapre.index, pre_impact, "k.")
        ax[1].plot(
            datapost.index,
            post_impact,
            "k.",
            label=counterfactual_label,
        )
        ax[1].axhline(y=0, c="k")
        ax[1].set(title="Causal Impact")

        ax[2].plot(datapost.index, post_impact_cumulative, c="k")
        ax[2].axhline(y=0, c="k")
        ax[2].set(title="Cumulative Causal Impact")

        # Shaded causal effect
        ax[0].fill_between(
            datapost.index,
            y1=np.squeeze(post_pred),
            y2=np.squeeze(post_y),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[1].fill_between(
            datapost.index,
            y1=np.squeeze(post_impact),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )

        # Intervention line
        # TODO: make this work when treatment_time is a datetime
        for i in [0, 1, 2]:
            ax[i].axvline(
                x=treatment_time,
                ls="-",
                lw=3,
                color="r",
                label="Treatment time",
            )

        ax[0].legend(fontsize=LEGEND_FONT_SIZE)

        return (fig, ax)

    def plot_difference_in_differences(self, experiment, impact):
        pass

    def plot_synthetic_control(self, experiment, impact):
        pass
