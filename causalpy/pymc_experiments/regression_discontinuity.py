"""
regression_discontinuity.py

Provides RegressionDiscontinuity experiment class
"""

import warnings
from typing import Optional

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


class RegressionDiscontinuity(ExperimentalDesign):
    """
    A class to analyse sharp regression discontinuity experiments.

    :param data:
        A pandas dataframe
    :param formula:
        A statistical model formula
    :param treatment_threshold:
        A scalar threshold value at which the treatment is applied
    :param model:
        A PyMC model
    :param running_variable_name:
        The name of the predictor variable that the treatment threshold is based upon
    :param epsilon:
        A small scalar value which determines how far above and below the treatment
        threshold to evaluate the causal impact.
    :param bandwidth:
        Data outside of the bandwidth (relative to the discontinuity) is not used to fit
        the model.

    Example
    --------
    >>> import causalpy as cp
    >>> df = cp.load_data("rd")
    >>> seed = 42
    >>> result = cp.pymc_experiments.RegressionDiscontinuity(
    ...     df,
    ...     formula="y ~ 1 + x + treated + x:treated",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={
    ...             "draws": 2000,
    ...             "target_accept": 0.95,
    ...             "random_seed": seed,
    ...             "progressbar": False,
    ...         },
    ...     ),
    ...     treatment_threshold=0.5,
    ... )
    >>> result.summary() # doctest: +NUMBER
    ============================Regression Discontinuity============================
    Formula: y ~ 1 + x + treated + x:treated
    Running variable: x
    Threshold on running variable: 0.5
    <BLANKLINE>
    Results:
    Discontinuity at threshold = 0.91
    Model coefficients:
    Intercept                     0.0, 94% HDI [0.0, 0.1]
    treated[T.True]               2.4, 94% HDI [1.6, 3.2]
    x                             1.3, 94% HDI [1.1, 1.5]
    x:treated[T.True]             -3.0, 94% HDI [-4.1, -2.0]
    sigma                         0.3, 94% HDI [0.3, 0.4]
    """

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        treatment_threshold: float,
        model=None,
        running_variable_name: str = "x",
        epsilon: float = 0.001,
        bandwidth: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.expt_type = "Regression Discontinuity"
        self.data = data
        self.formula = formula
        self.running_variable_name = running_variable_name
        self.treatment_threshold = treatment_threshold
        self.epsilon = epsilon
        self.bandwidth = bandwidth
        self._input_validation()

        if self.bandwidth is not None:
            fmin = self.treatment_threshold - self.bandwidth
            fmax = self.treatment_threshold + self.bandwidth
            filtered_data = self.data.query(f"{fmin} <= x <= {fmax}")
            if len(filtered_data) <= 10:
                warnings.warn(
                    f"Choice of bandwidth parameter has lead to only {len(filtered_data)} remaining datapoints. Consider increasing the bandwidth parameter.",  # noqa: E501
                    UserWarning,
                )
            y, X = dmatrices(formula, filtered_data)
        else:
            y, X = dmatrices(formula, self.data)

        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.y, self.X = np.asarray(y), np.asarray(X)
        self.outcome_variable_name = y.design_info.column_names[0]

        # DEVIATION FROM SKL EXPERIMENT CODE =============================
        # fit the model to the observed (pre-intervention) data
        COORDS = {"coeffs": self.labels, "obs_indx": np.arange(self.X.shape[0])}
        self.model.fit(X=self.X, y=self.y, coords=COORDS)
        # ================================================================

        # score the goodness of fit to all data
        self.score = self.model.score(X=self.X, y=self.y)

        # get the model predictions of the observed data
        if self.bandwidth is not None:
            xi = np.linspace(fmin, fmax, 200)
        else:
            xi = np.linspace(
                np.min(self.data[self.running_variable_name]),
                np.max(self.data[self.running_variable_name]),
                200,
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
                    [
                        self.treatment_threshold - self.epsilon,
                        self.treatment_threshold + self.epsilon,
                    ]
                ),
                "treated": np.array([0, 1]),
            }
        )
        (new_x,) = build_design_matrices([self._x_design_info], self.x_discon)
        self.pred_discon = self.model.predict(X=np.asarray(new_x))
        self.discontinuity_at_threshold = (
            self.pred_discon["posterior_predictive"].sel(obs_ind=1)["mu"]
            - self.pred_discon["posterior_predictive"].sel(obs_ind=0)["mu"]
        )

    def _input_validation(self):
        """Validate the input data and model formula for correctness"""
        if "treated" not in self.formula:
            raise FormulaException(
                "A predictor called `treated` should be in the formula"
            )

        if _is_variable_dummy_coded(self.data["treated"]) is False:
            raise DataException(
                """The treated variable should be dummy coded. Consisting of 0's and 1's only."""  # noqa: E501
            )

    def _is_treated(self, x):
        """Returns ``True`` if `x` is greater than or equal to the treatment threshold.

        .. warning::

            Assumes treatment is given to those ABOVE the treatment threshold.
        """
        return np.greater_equal(x, self.treatment_threshold)

    def plot(self):
        """
        Plot the results
        """
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
        h_line, h_patch = plot_xY(
            self.x_pred[self.running_variable_name],
            self.pred["posterior_predictive"].mu,
            ax=ax,
            plot_hdi_kwargs={"color": "C1"},
        )
        handles = [(h_line, h_patch)]
        labels = ["Posterior mean"]

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
        ax.legend(
            handles=(h_tuple for h_tuple in handles),
            labels=labels,
            fontsize=LEGEND_FONT_SIZE,
        )
        return (fig, ax)

    def summary(self) -> None:
        """
        Print text output summarising the results
        """

        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print(f"Running variable: {self.running_variable_name}")
        print(f"Threshold on running variable: {self.treatment_threshold}")
        print("\nResults:")
        print(
            f"Discontinuity at threshold = {self.discontinuity_at_threshold.mean():.2f}"
        )
        self.print_coefficients()
