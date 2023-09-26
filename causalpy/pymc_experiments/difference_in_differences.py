import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from patsy import build_design_matrices, dmatrices

from causalpy.custom_exceptions import DataException, FormulaException
from causalpy.plot_utils import plot_xY
from causalpy.pymc_experiments.experimental_design import ExperimentalDesign
from causalpy.utils import _is_variable_dummy_coded

LEGEND_FONT_SIZE = 12
az.style.use("arviz-darkgrid")


class DifferenceInDifferences(ExperimentalDesign):
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
    >>> result = cp.pymc_experiments.DifferenceInDifferences(
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
    ...     )
    ...  )
    >>> result.summary() # doctest: +NUMBER
    ===========================Difference in Differences============================
    Formula: y ~ 1 + group*post_treatment
    <BLANKLINE>
    Results:
    Causal impact = 0.5, $CI_{94%}$[0.4, 0.6]
    Model coefficients:
    Intercept                     1.0, 94% HDI [1.0, 1.1]
    post_treatment[T.True]        0.9, 94% HDI [0.9, 1.0]
    group                         0.1, 94% HDI [0.0, 0.2]
    group:post_treatment[T.True]  0.5, 94% HDI [0.4, 0.6]
    sigma                         0.0, 94% HDI [0.0, 0.1]
    """

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        time_variable_name: str,
        group_variable_name: str,
        model=None,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.data = data
        self.expt_type = "Difference in Differences"
        self.formula = formula
        self.time_variable_name = time_variable_name
        self.group_variable_name = group_variable_name
        self._input_validation()

        y, X = dmatrices(formula, self.data)
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.y, self.X = np.asarray(y), np.asarray(X)
        self.outcome_variable_name = y.design_info.column_names[0]

        COORDS = {"coeffs": self.labels, "obs_indx": np.arange(self.X.shape[0])}
        self.model.fit(X=self.X, y=self.y, coords=COORDS)

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
        assert not self.x_pred_control.empty
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
        assert not self.x_pred_treatment.empty
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

        # calculate causal impact.
        # This is the coefficient on the interaction term
        coeff_names = self.idata.posterior.coords["coeffs"].data
        for i, label in enumerate(coeff_names):
            if "post_treatment" in label and self.group_variable_name in label:
                self.causal_impact = self.idata.posterior["beta"].isel({"coeffs": i})

    def _input_validation(self):
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

    def plot(self):
        """Plot the results.
        Creating the combined mean + HDI legend entries is a bit involved.
        """
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
        self._plot_causal_impact_arrow(ax)

        # formatting
        ax.set(
            xticks=self.x_pred_treatment[self.time_variable_name].values,
            title=self._causal_impact_summary_stat(),
        )
        ax.legend(
            handles=(h_tuple for h_tuple in handles),
            labels=labels,
            fontsize=LEGEND_FONT_SIZE,
        )
        return fig, ax

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
        # Note that we force to be float to avoid a type error using np.ptp with boolean
        # values
        diff = np.ptp(
            np.array(self.x_pred_treatment[self.time_variable_name].values).astype(
                float
            )
        )
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

    def _causal_impact_summary_stat(self) -> str:
        """Computes the mean and 94% credible interval bounds for the causal impact."""
        percentiles = self.causal_impact.quantile([0.03, 1 - 0.03]).values
        ci = "$CI_{94%}$" + f"[{percentiles[0]:.2f}, {percentiles[1]:.2f}]"
        causal_impact = f"{self.causal_impact.mean():.2f}, "
        return f"Causal impact = {causal_impact + ci}"

    def summary(self) -> None:
        """
        Print text output summarising the results
        """

        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print("\nResults:")
        # TODO: extra experiment specific outputs here
        print(self._causal_impact_summary_stat())
        self.print_coefficients()
