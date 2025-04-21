#   Copyright 2022 - 2025 The PyMC Labs Developers
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
"""
Difference in differences
"""

import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from patsy import build_design_matrices, dmatrices
from sklearn.base import RegressorMixin

from causalpy.custom_exceptions import (
    DataException,
    FormulaException,
)
from causalpy.plot_utils import plot_xY
from causalpy.pymc_models import PyMCModel
from causalpy.utils import _is_variable_dummy_coded, convert_to_string, round_num

from .base import BaseExperiment

LEGEND_FONT_SIZE = 12


class DifferenceInDifferences(BaseExperiment):
    """A class to analyse data from Difference in Difference settings.

    .. note::

        There is no pre/post intervention data distinction for DiD, we fit all the
        data available.
    :param data:
        A pandas dataframe
    :param formula:
        A statistical model formula
    :param time_variable_name:
        Name of the data column for the time variable
    :param group_variable_name:
        Name of the data column for the group variable
    :param model:
        A PyMC model for difference in differences

    Example
    --------
    >>> import causalpy as cp
    >>> df = cp.load_data("did")
    >>> seed = 42
    >>> result = cp.DifferenceInDifferences(
    ...     df,
    ...     formula="y ~ 1 + group*post_treatment",
    ...     time_variable_name="t",
    ...     group_variable_name="group",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={
    ...             "target_accept": 0.95,
    ...             "random_seed": seed,
    ...             "progressbar": False,
    ...         }
    ...     ),
    ... )
    """

    supports_ols = True
    supports_bayes = True

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        time_variable_name: str,
        group_variable_name: str,
        model=None,
        **kwargs,
    ) -> None:
        super().__init__(model=model)

        self.data = data
        self.expt_type = "Difference in Differences"
        self.formula = formula
        self.time_variable_name = time_variable_name
        self.group_variable_name = group_variable_name
        self.input_validation()

        y, X = dmatrices(formula, self.data)
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.y, self.X = np.asarray(y), np.asarray(X)
        self.outcome_variable_name = y.design_info.column_names[0]

        # fit model
        if isinstance(self.model, PyMCModel):
            COORDS = {"coeffs": self.labels, "obs_ind": np.arange(self.X.shape[0])}
            self.model.fit(X=self.X, y=self.y, coords=COORDS)
        elif isinstance(self.model, RegressorMixin):
            self.model.fit(X=self.X, y=self.y)
        else:
            raise ValueError("Model type not recognized")

        # predicted outcome for control group
        self.x_pred_control = (
            self.data
            # just the untreated group
            .query(f"{self.group_variable_name} == 0")
            # drop the outcome variable
            .drop(self.outcome_variable_name, axis=1)
            # We may have multiple units per time point, we only want one time point
            .groupby(self.time_variable_name)
            .first()
            .reset_index()
        )
        if self.x_pred_control.empty:
            raise ValueError("x_pred_control is empty")
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred_control)
        self.y_pred_control = self.model.predict(np.asarray(new_x))

        # predicted outcome for treatment group
        self.x_pred_treatment = (
            self.data
            # just the treated group
            .query(f"{self.group_variable_name} == 1")
            # drop the outcome variable
            .drop(self.outcome_variable_name, axis=1)
            # We may have multiple units per time point, we only want one time point
            .groupby(self.time_variable_name)
            .first()
            .reset_index()
        )
        if self.x_pred_treatment.empty:
            raise ValueError("x_pred_treatment is empty")
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred_treatment)
        self.y_pred_treatment = self.model.predict(np.asarray(new_x))

        # predicted outcome for counterfactual. This is given by removing the influence
        # of the interaction term between the group and the post_treatment variable
        self.x_pred_counterfactual = (
            self.data
            # just the treated group
            .query(f"{self.group_variable_name} == 1")
            # just the treatment period(s)
            .query("post_treatment == True")
            # drop the outcome variable
            .drop(self.outcome_variable_name, axis=1)
            # We may have multiple units per time point, we only want one time point
            .groupby(self.time_variable_name)
            .first()
            .reset_index()
        )
        if self.x_pred_counterfactual.empty:
            raise ValueError("x_pred_counterfactual is empty")
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
        if isinstance(self.model, PyMCModel):
            # This is the coefficient on the interaction term
            coeff_names = self.model.idata.posterior.coords["coeffs"].data
            for i, label in enumerate(coeff_names):
                if "post_treatment" in label and self.group_variable_name in label:
                    self.causal_impact = self.model.idata.posterior["beta"].isel(
                        {"coeffs": i}
                    )
        elif isinstance(self.model, RegressorMixin):
            # This is the coefficient on the interaction term
            # TODO: THIS IS NOT YET CORRECT ?????
            self.causal_impact = (
                self.y_pred_treatment[1] - self.y_pred_counterfactual[0]
            )[0]
        else:
            raise ValueError("Model type not recognized")

    def input_validation(self):
        """Validate the input data and model formula for correctness"""
        if "post_treatment" not in self.formula:
            raise FormulaException(
                "A predictor called `post_treatment` should be in the formula"
            )

        if "post_treatment" not in self.data.columns:
            raise DataException(
                "Require a boolean column labelling observations which are `treated`"
            )

        if "unit" not in self.data.columns:
            raise DataException(
                "Require a `unit` column to label unique units. This is used for plotting purposes"  # noqa: E501
            )

        if _is_variable_dummy_coded(self.data[self.group_variable_name]) is False:
            raise DataException(
                f"""The grouping variable {self.group_variable_name} should be dummy
                coded. Consisting of 0's and 1's only."""
            )

    def summary(self, round_to=None) -> None:
        """Print summary of main results and model coefficients.

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers
        """
        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print("\nResults:")
        print(self._causal_impact_summary_stat(round_to))
        self.print_coefficients(round_to)

    def _causal_impact_summary_stat(self, round_to=None) -> str:
        """Computes the mean and 94% credible interval bounds for the causal impact."""
        return f"Causal impact = {convert_to_string(self.causal_impact, round_to=round_to)}"

    def _bayesian_plot(self, round_to=None, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the results

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """

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
            self.data,
            x=self.time_variable_name,
            y=self.outcome_variable_name,
            hue=self.group_variable_name,
            alpha=1,
            legend=False,
            markers=True,
            ax=ax,
        )

        # Plot model fit to control group
        time_points = self.x_pred_control[self.time_variable_name].values
        h_line, h_patch = plot_xY(
            time_points,
            self.y_pred_control.posterior_predictive.mu,
            ax=ax,
            plot_hdi_kwargs={"color": "C0"},
            label="Control group",
        )
        handles = [(h_line, h_patch)]
        labels = ["Control group"]

        # Plot model fit to treatment group
        time_points = self.x_pred_control[self.time_variable_name].values
        h_line, h_patch = plot_xY(
            time_points,
            self.y_pred_treatment.posterior_predictive.mu,
            ax=ax,
            plot_hdi_kwargs={"color": "C1"},
            label="Treatment group",
        )
        handles.append((h_line, h_patch))
        labels.append("Treatment group")

        # Plot counterfactual - post-test for treatment group IF no treatment
        # had occurred.
        time_points = self.x_pred_counterfactual[self.time_variable_name].values
        if len(time_points) == 1:
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
                pc.set_facecolor("C0")
                pc.set_edgecolor("None")
                pc.set_alpha(0.5)
        else:
            h_line, h_patch = plot_xY(
                time_points,
                self.y_pred_counterfactual.posterior_predictive.mu,
                ax=ax,
                plot_hdi_kwargs={"color": "C2"},
                label="Counterfactual",
            )
            handles.append((h_line, h_patch))
            labels.append("Counterfactual")

        # arrow to label the causal impact
        _plot_causal_impact_arrow(self, ax)

        # formatting
        ax.set(
            xticks=self.x_pred_treatment[self.time_variable_name].values,
            title=self._causal_impact_summary_stat(round_to),
        )
        ax.legend(
            handles=(h_tuple for h_tuple in handles),
            labels=labels,
            fontsize=LEGEND_FONT_SIZE,
        )
        return fig, ax

    def _ols_plot(self, round_to=None, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """Generate plot for difference-in-differences"""
        round_to = kwargs.get("round_to")
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
            xy=(
                1.05,
                np.mean([self.y_pred_counterfactual[0], self.y_pred_treatment[1]]),
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
            title=f"Causal impact = {round_num(self.causal_impact, round_to)}",
        )
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        return fig, ax
