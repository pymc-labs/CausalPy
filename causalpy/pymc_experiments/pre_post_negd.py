import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from patsy import build_design_matrices, dmatrices

from causalpy.custom_exceptions import DataException
from causalpy.plot_utils import plot_xY
from causalpy.pymc_experiments.experimental_design import ExperimentalDesign
from causalpy.utils import _series_has_2_levels

LEGEND_FONT_SIZE = 12
az.style.use("arviz-darkgrid")


class PrePostNEGD(ExperimentalDesign):
    """
    A class to analyse data from pretest/posttest designs

    :param data:
        A pandas dataframe
    :param formula:
        A statistical model formula
    :param group_variable_name:
        Name of the column in data for the group variable
    :param pretreatment_variable_name:
        Name of the column in data for the pretreatment variable
    :param model:
        A PyMC model

    Example
    --------
    >>> import causalpy as cp
    >>> df = cp.load_data("anova1")
    >>> seed = 42
    >>> result = cp.pymc_experiments.PrePostNEGD(
    ...     df,
    ...     formula="post ~ 1 + C(group) + pre",
    ...     group_variable_name="group",
    ...     pretreatment_variable_name="pre",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={
    ...             "target_accept": 0.95,
    ...             "random_seed": seed,
    ...             "progressbar": False,
    ...         }
    ...     )
    ... )
    >>> result.summary() # doctest: +NUMBER
    ==================Pretest/posttest Nonequivalent Group Design===================
    Formula: post ~ 1 + C(group) + pre
    <BLANKLINE>
    Results:
    Causal impact = 1.8, $CI_{94%}$[1.6, 2.0]
    Model coefficients:
    Intercept                     -0.4, 94% HDI [-1.2, 0.2]
    C(group)[T.1]                 1.8, 94% HDI [1.6, 2.0]
    pre                           1.0, 94% HDI [0.9, 1.1]
    sigma                         0.5, 94% HDI [0.4, 0.5]
    """

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        group_variable_name: str,
        pretreatment_variable_name: str,
        model=None,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.data = data
        self.expt_type = "Pretest/posttest Nonequivalent Group Design"
        self.formula = formula
        self.group_variable_name = group_variable_name
        self.pretreatment_variable_name = pretreatment_variable_name
        self._input_validation()

        y, X = dmatrices(formula, self.data)
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.y, self.X = np.asarray(y), np.asarray(X)
        self.outcome_variable_name = y.design_info.column_names[0]

        # fit the model to the observed (pre-intervention) data
        COORDS = {"coeffs": self.labels, "obs_indx": np.arange(self.X.shape[0])}
        self.model.fit(X=self.X, y=self.y, coords=COORDS)

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
        self.pred_untreated = self.model.predict(X=np.asarray(new_x))
        # treated
        x_pred_untreated = pd.DataFrame(
            {
                self.pretreatment_variable_name: self.pred_xi,
                self.group_variable_name: np.ones(self.pred_xi.shape),
            }
        )
        (new_x,) = build_design_matrices([self._x_design_info], x_pred_untreated)
        self.pred_treated = self.model.predict(X=np.asarray(new_x))

        # Evaluate causal impact as equal to the trestment effect
        self.causal_impact = self.idata.posterior["beta"].sel(
            {"coeffs": self._get_treatment_effect_coeff()}
        )

        # ================================================================

    def _input_validation(self) -> None:
        """Validate the input data and model formula for correctness"""
        if not _series_has_2_levels(self.data[self.group_variable_name]):
            raise DataException(
                f"""
                There must be 2 levels of the grouping variable
                {self.group_variable_name}. I.e. the treated and untreated.
                """
            )

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
            legend=True,
            ax=ax[0],
        )
        ax[0].set(xlabel="Pretest", ylabel="Posttest")

        # plot posterior predictive of untreated
        h_line, h_patch = plot_xY(
            self.pred_xi,
            self.pred_untreated["posterior_predictive"].mu,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C0"},
            label="Control group",
        )
        handles = [(h_line, h_patch)]
        labels = ["Control group"]

        # plot posterior predictive of treated
        h_line, h_patch = plot_xY(
            self.pred_xi,
            self.pred_treated["posterior_predictive"].mu,
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
        az.plot_posterior(self.causal_impact, ref_val=0, ax=ax[1])
        ax[1].set(title="Estimated treatment effect")
        return fig, ax

    def _causal_impact_summary_stat(self) -> str:
        """Computes the mean and 94% credible interval bounds for the causal impact."""
        percentiles = self.causal_impact.quantile([0.03, 1 - 0.03]).values
        ci = r"$CI_{94%}$" + f"[{percentiles[0]:.2f}, {percentiles[1]:.2f}]"
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
