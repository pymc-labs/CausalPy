"""
Experiment routines for PyMC models.

- ExperimentalDesign base class
- Pre-Post Fit
- Interrupted Time Series
- Synthetic Control
- Difference in differences
- Regression Discontinuity
- Pretest/Posttest Nonequivalent Group Design

"""

import warnings  # noqa: I001
from typing import Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from patsy import build_design_matrices, dmatrices
from sklearn.linear_model import LinearRegression as sk_lin_reg

from causalpy.custom_exceptions import (
    BadIndexException,  # NOQA
    DataException,
    FormulaException,
)
from causalpy.plot_utils import plot_xY
from causalpy.utils import _is_variable_dummy_coded, round_num

LEGEND_FONT_SIZE = 12
az.style.use("arviz-darkgrid")


class ExperimentalDesign:
    """
    Base class for other experiment types

    See subclasses for examples of most methods
    """

    model = None
    expt_type = None

    def __init__(self, model=None, **kwargs):
        if model is not None:
            self.model = model
        if self.model is None:
            raise ValueError("fitting_model not set or passed.")

    @property
    def idata(self):
        """
        Access to the models InferenceData object
        """

        return self.model.idata

    def print_coefficients(self, round_to=None) -> None:
        """
        Prints the model coefficients

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.

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
        ...             sample_kwargs={
        ...                 "draws": 2000,
        ...                 "random_seed": seed,
        ...                 "progressbar": False
        ...             }),
        ...  )
        >>> result.print_coefficients(round_to=1) # doctest: +NUMBER
        Model coefficients:
        Intercept                     1, 94% HDI [1, 1]
        post_treatment[T.True]        1, 94% HDI [0.9, 1]
        group                         0.2, 94% HDI [0.09, 0.2]
        group:post_treatment[T.True]  0.5, 94% HDI [0.4, 0.6]
        sigma                         0.08, 94% HDI [0.07, 0.1]
        """
        print("Model coefficients:")
        coeffs = az.extract(self.idata.posterior, var_names="beta")
        # Note: f"{name: <30}" pads the name with spaces so that we have alignment of
        # the stats despite variable names of different lengths
        for name in self.labels:
            coeff_samples = coeffs.sel(coeffs=name)
            print(
                f"{name: <30}{round_num(coeff_samples.mean().data, round_to)}, 94% HDI [{round_num(coeff_samples.quantile(0.03).data, round_to)}, {round_num(coeff_samples.quantile(1-0.03).data, round_to)}]"  # noqa: E501
            )
        # add coeff for measurement std
        coeff_samples = az.extract(self.model.idata.posterior, var_names="sigma")
        name = "sigma"
        print(
            f"{name: <30}{round_num(coeff_samples.mean().data, round_to)}, 94% HDI [{round_num(coeff_samples.quantile(0.03).data, round_to)}, {round_num(coeff_samples.quantile(1-0.03).data, round_to)}]"  # noqa: E501
        )


class PrePostFit(ExperimentalDesign):
    """
    A class to analyse quasi-experiments where parameter estimation is based on just
    the pre-intervention data.

    :param data:
        A pandas dataframe
    :param treatment_time:
        The time when treatment occured, should be in reference to the data index
    :param formula:
        A statistical model formula
    :param model:
        A PyMC model

    Example
    --------
    >>> import causalpy as cp
    >>> sc = cp.load_data("sc")
    >>> treatment_time = 70
    >>> seed = 42
    >>> result = cp.pymc_experiments.PrePostFit(
    ...     sc,
    ...     treatment_time,
    ...     formula="actual ~ 0 + a + g",
    ...     model=cp.pymc_models.WeightedSumFitter(
    ...         sample_kwargs={
    ...             "draws": 400,
    ...             "target_accept": 0.95,
    ...             "random_seed": seed,
    ...             "progressbar": False
    ...         }
    ...     ),
    ... )
    >>> result.summary(round_to=1) # doctest: +NUMBER
    ==================================Pre-Post Fit==================================
    Formula: actual ~ 0 + a + g
    Model coefficients:
    a                             0.6, 94% HDI [0.6, 0.6]
    g                             0.4, 94% HDI [0.4, 0.4]
    sigma                         0.8, 94% HDI [0.6, 0.9]
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_time: Union[int, float, pd.Timestamp],
        formula: str,
        model=None,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._input_validation(data, treatment_time)

        self.treatment_time = treatment_time
        # set experiment type - usually done in subclasses
        self.expt_type = "Pre-Post Fit"
        # split data in to pre and post intervention
        self.datapre = data[data.index < self.treatment_time]
        self.datapost = data[data.index >= self.treatment_time]

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
        self.model.fit(X=self.pre_X, y=self.pre_y, coords=COORDS)
        # ================================================================

        # score the goodness of fit to the pre-intervention data
        self.score = self.model.score(X=self.pre_X, y=self.pre_y)

        # get the model predictions of the observed (pre-intervention) data
        self.pre_pred = self.model.predict(X=self.pre_X)

        # calculate the counterfactual
        self.post_pred = self.model.predict(X=self.post_X)

        # causal impact pre (ie the residuals of the model fit to observed)
        pre_data = xr.DataArray(self.pre_y[:, 0], dims=["obs_ind"])
        self.pre_impact = (
            pre_data - self.pre_pred["posterior_predictive"].mu
        ).transpose(..., "obs_ind")

        # causal impact post (ie the residuals of the model fit to observed)
        post_data = xr.DataArray(self.post_y[:, 0], dims=["obs_ind"])
        self.post_impact = (
            post_data - self.post_pred["posterior_predictive"].mu
        ).transpose(..., "obs_ind")

        # cumulative impact post
        self.post_impact_cumulative = self.post_impact.cumsum(dim="obs_ind")

    def _input_validation(self, data, treatment_time):
        """Validate the input data and model formula for correctness"""
        if isinstance(data.index, pd.DatetimeIndex) and not isinstance(
            treatment_time, pd.Timestamp
        ):
            raise BadIndexException(
                "If data.index is DatetimeIndex, treatment_time must be pd.Timestamp."
            )
        if not isinstance(data.index, pd.DatetimeIndex) and isinstance(
            treatment_time, pd.Timestamp
        ):
            raise BadIndexException(
                "If data.index is not DatetimeIndex, treatment_time must be pd.Timestamp."  # noqa: E501
            )

    def plot(self, counterfactual_label="Counterfactual", round_to=None, **kwargs):
        """
        Plot the results

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

        # TOP PLOT --------------------------------------------------
        # pre-intervention period
        h_line, h_patch = plot_xY(
            self.datapre.index,
            self.pre_pred["posterior_predictive"].mu,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C0"},
        )
        handles = [(h_line, h_patch)]
        labels = ["Pre-intervention period"]

        (h,) = ax[0].plot(self.datapre.index, self.pre_y, "k.", label="Observations")
        handles.append(h)
        labels.append("Observations")

        # post intervention period
        h_line, h_patch = plot_xY(
            self.datapost.index,
            self.post_pred["posterior_predictive"].mu,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C1"},
        )
        handles.append((h_line, h_patch))
        labels.append(counterfactual_label)

        ax[0].plot(self.datapost.index, self.post_y, "k.")
        # Shaded causal effect
        h = ax[0].fill_between(
            self.datapost.index,
            y1=az.extract(
                self.post_pred, group="posterior_predictive", var_names="mu"
            ).mean("sample"),
            y2=np.squeeze(self.post_y),
            color="C0",
            alpha=0.25,
        )
        handles.append(h)
        labels.append("Causal impact")

        ax[0].set(
            title=f"""
            Pre-intervention Bayesian $R^2$: {round_num(self.score.r2, round_to)}
            (std = {round_num(self.score.r2_std, round_to)})
            """
        )

        # MIDDLE PLOT -----------------------------------------------
        plot_xY(
            self.datapre.index,
            self.pre_impact,
            ax=ax[1],
            plot_hdi_kwargs={"color": "C0"},
        )
        plot_xY(
            self.datapost.index,
            self.post_impact,
            ax=ax[1],
            plot_hdi_kwargs={"color": "C1"},
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
            plot_hdi_kwargs={"color": "C1"},
        )
        ax[2].axhline(y=0, c="k")

        # Intervention line
        for i in [0, 1, 2]:
            ax[i].axvline(
                x=self.treatment_time,
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

    def summary(self, round_to=None) -> None:
        """
        Print text output summarising the results

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """

        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        # TODO: extra experiment specific outputs here
        self.print_coefficients(round_to)


class InterruptedTimeSeries(PrePostFit):
    """
    A wrapper around PrePostFit class

    :param data:
        A pandas dataframe
    :param treatment_time:
        The time when treatment occured, should be in reference to the data index
    :param formula:
        A statistical model formula
    :param model:
        A PyMC model

    Example
    --------
    >>> import causalpy as cp
    >>> df = (
    ...     cp.load_data("its")
    ...     .assign(date=lambda x: pd.to_datetime(x["date"]))
    ...     .set_index("date")
    ... )
    >>> treatment_time = pd.to_datetime("2017-01-01")
    >>> seed = 42
    >>> result = cp.pymc_experiments.InterruptedTimeSeries(
    ...     df,
    ...     treatment_time,
    ...     formula="y ~ 1 + t + C(month)",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={
    ...             "target_accept": 0.95,
    ...             "random_seed": seed,
    ...             "progressbar": False,
    ...         }
    ...     )
    ... )
    """

    expt_type = "Interrupted Time Series"


class SyntheticControl(PrePostFit):
    """A wrapper around the PrePostFit class

    :param data:
        A pandas dataframe
    :param treatment_time:
        The time when treatment occured, should be in reference to the data index
    :param formula:
        A statistical model formula
    :param model:
        A PyMC model

    Example
    --------
    >>> import causalpy as cp
    >>> df = cp.load_data("sc")
    >>> treatment_time = 70
    >>> seed = 42
    >>> result = cp.pymc_experiments.SyntheticControl(
    ...     df,
    ...     treatment_time,
    ...     formula="actual ~ 0 + a + b + c + d + e + f + g",
    ...     model=cp.pymc_models.WeightedSumFitter(
    ...         sample_kwargs={
    ...             "target_accept": 0.95,
    ...             "random_seed": seed,
    ...             "progressbar": False,
    ...         }
    ...     ),
    ... )
    """

    expt_type = "Synthetic Control"

    def plot(self, plot_predictors=False, **kwargs):
        """Plot the results

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """
        fig, ax = super().plot(counterfactual_label="Synthetic control", **kwargs)
        if plot_predictors:
            # plot control units as well
            ax[0].plot(self.datapre.index, self.pre_X, "-", c=[0.8, 0.8, 0.8], zorder=1)
            ax[0].plot(
                self.datapost.index, self.post_X, "-", c=[0.8, 0.8, 0.8], zorder=1
            )
        return fig, ax


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

    def plot(self, round_to=None):
        """Plot the results.

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
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
            title=self._causal_impact_summary_stat(round_to),
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

    def _causal_impact_summary_stat(self, round_to=None) -> str:
        """Computes the mean and 94% credible interval bounds for the causal impact."""
        percentiles = self.causal_impact.quantile([0.03, 1 - 0.03]).values
        ci = (
            "$CI_{94\\%}$"
            + f"[{round_num(percentiles[0], round_to)}, {round_num(percentiles[1], round_to)}]"
        )
        causal_impact = f"{round_num(self.causal_impact.mean(), round_to)}, "
        return f"Causal impact = {causal_impact + ci}"

    def summary(self, round_to=None) -> None:
        """
        Print text output summarising the results.

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """

        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print("\nResults:")
        print(round_num(self._causal_impact_summary_stat(), round_to))
        self.print_coefficients(round_to)


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
    ...             "draws": 100,
    ...             "target_accept": 0.95,
    ...             "random_seed": seed,
    ...             "progressbar": False,
    ...         },
    ...     ),
    ...     treatment_threshold=0.5,
    ... )
    """

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        treatment_threshold: float,
        model=None,
        running_variable_name: str = "x",
        epsilon: float = 0.001,
        bandwidth: float = np.inf,
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

        if self.bandwidth is not np.inf:
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
        if self.bandwidth is not np.inf:
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

    def plot(self, round_to=None):
        """
        Plot the results

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """
        fig, ax = plt.subplots()
        # Plot raw data
        sns.scatterplot(
            self.data,
            x=self.running_variable_name,
            y=self.outcome_variable_name,
            c="k",
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
        title_info = f"{round_num(self.score.r2, round_to)} (std = {round_num(self.score.r2_std, round_to)})"
        r2 = f"Bayesian $R^2$ on all data = {title_info}"
        percentiles = self.discontinuity_at_threshold.quantile([0.03, 1 - 0.03]).values
        ci = (
            r"$CI_{94\%}$"
            + f"[{round_num(percentiles[0], round_to)}, {round_num(percentiles[1], round_to)}]"
        )
        discon = f"""
            Discontinuity at threshold = {round_num(self.discontinuity_at_threshold.mean(), round_to)},
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
        return fig, ax

    def summary(self, round_to: None) -> None:
        """
        Print text output summarising the results

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """

        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print(f"Running variable: {self.running_variable_name}")
        print(f"Threshold on running variable: {self.treatment_threshold}")
        print("\nResults:")
        print(
            f"Discontinuity at threshold = {round_num(self.discontinuity_at_threshold.mean(), round_to)}"
        )
        self.print_coefficients(round_to)


class RegressionKink(ExperimentalDesign):
    """
    A class to analyse sharp regression kink experiments.

    :param data:
        A pandas dataframe
    :param formula:
        A statistical model formula
    :param kink_point:
        A scalar threshold value at which there is a change in the first derivative of
        the assignment function
    :param model:
        A PyMC model
    :param running_variable_name:
        The name of the predictor variable that the kink_point is based upon
    :param epsilon:
        A small scalar value which determines how far above and below the kink point to
        evaluate the causal impact.
    :param bandwidth:
        Data outside of the bandwidth (relative to the discontinuity) is not used to fit
        the model.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        kink_point: float,
        model=None,
        running_variable_name: str = "x",
        epsilon: float = 0.001,
        bandwidth: float = np.inf,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.expt_type = "Regression Kink"
        self.data = data
        self.formula = formula
        self.running_variable_name = running_variable_name
        self.kink_point = kink_point
        self.epsilon = epsilon
        self.bandwidth = bandwidth
        self._input_validation()

        if self.bandwidth is not np.inf:
            fmin = self.kink_point - self.bandwidth
            fmax = self.kink_point + self.bandwidth
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

        COORDS = {"coeffs": self.labels, "obs_indx": np.arange(self.X.shape[0])}
        self.model.fit(X=self.X, y=self.y, coords=COORDS)

        # score the goodness of fit to all data
        self.score = self.model.score(X=self.X, y=self.y)

        # get the model predictions of the observed data
        if self.bandwidth is not np.inf:
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

        # evaluate gradient change around kink point
        mu_kink_left, mu_kink, mu_kink_right = self._probe_kink_point()
        self.gradient_change = self._eval_gradient_change(
            mu_kink_left, mu_kink, mu_kink_right, epsilon
        )

    @staticmethod
    def _eval_gradient_change(mu_kink_left, mu_kink, mu_kink_right, epsilon):
        """Evaluate the gradient change at the kink point.
        It works by evaluating the model below the kink point, at the kink point,
        and above the kink point.
        This is a static method for ease of testing.
        """
        gradient_left = (mu_kink - mu_kink_left) / epsilon
        gradient_right = (mu_kink_right - mu_kink) / epsilon
        gradient_change = gradient_right - gradient_left
        return gradient_change

    def _probe_kink_point(self):
        # Create a dataframe to evaluate predicted outcome at the kink point and either
        # side
        x_predict = pd.DataFrame(
            {
                self.running_variable_name: np.array(
                    [
                        self.kink_point - self.epsilon,
                        self.kink_point,
                        self.kink_point + self.epsilon,
                    ]
                ),
                "treated": np.array([0, 1, 1]),
            }
        )
        (new_x,) = build_design_matrices([self._x_design_info], x_predict)
        predicted = self.model.predict(X=np.asarray(new_x))
        # extract predicted mu values
        mu_kink_left = predicted["posterior_predictive"].sel(obs_ind=0)["mu"]
        mu_kink = predicted["posterior_predictive"].sel(obs_ind=1)["mu"]
        mu_kink_right = predicted["posterior_predictive"].sel(obs_ind=2)["mu"]
        return mu_kink_left, mu_kink, mu_kink_right

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

        if self.bandwidth <= 0:
            raise ValueError("The bandwidth must be greater than zero.")

        if self.epsilon <= 0:
            raise ValueError("Epsilon must be greater than zero.")

    def _is_treated(self, x):
        """Returns ``True`` if `x` is greater than or equal to the treatment threshold."""  # noqa: E501
        return np.greater_equal(x, self.kink_point)

    def plot(self, round_to=None):
        """
        Plot the results

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
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
        title_info = f"{round_num(self.score.r2, round_to)} (std = {round_num(self.score.r2_std, round_to)})"
        r2 = f"Bayesian $R^2$ on all data = {title_info}"
        percentiles = self.gradient_change.quantile([0.03, 1 - 0.03]).values
        ci = (
            r"$CI_{94\%}$"
            + f"[{round_num(percentiles[0], round_to)}, {round_num(percentiles[1], round_to)}]"
        )
        grad_change = f"""
            Change in gradient = {round_num(self.gradient_change.mean(), round_to)},
            """
        ax.set(title=r2 + "\n" + grad_change + ci)
        # Intervention line
        ax.axvline(
            x=self.kink_point,
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

    def summary(self, round_to=None) -> None:
        """
        Print text output summarising the results

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """

        print(
            f"""
        {self.expt_type:=^80}
        Formula: {self.formula}
        Running variable: {self.running_variable_name}
        Kink point on running variable: {self.kink_point}

        Results:
        Change in slope at kink point = {round_num(self.gradient_change.mean(), round_to)}
        """
        )
        self.print_coefficients(round_to)


class PrePostNEGD(ExperimentalDesign):
    """
    A class to analyse data from pretest/posttest designs

    :param data:
        A pandas dataframe
    :param formula:
        A statistical model formula
    :param group_variable_name:
        Name of the column in data for the group variable, should be either
        binary or boolean
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
    >>> result.summary(round_to=1) # doctest: +NUMBER
    ==================Pretest/posttest Nonequivalent Group Design===================
    Formula: post ~ 1 + C(group) + pre
    <BLANKLINE>
    Results:
    Causal impact = 2, $CI_{94%}$[2, 2]
    Model coefficients:
    Intercept                     -0.5, 94% HDI [-1, 0.2]
    C(group)[T.1]                 2, 94% HDI [2, 2]
    pre                           1, 94% HDI [1, 1]
    sigma                         0.5, 94% HDI [0.5, 0.6]
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
        (new_x_untreated,) = build_design_matrices(
            [self._x_design_info], x_pred_untreated
        )
        self.pred_untreated = self.model.predict(X=np.asarray(new_x_untreated))
        # treated
        x_pred_treated = pd.DataFrame(
            {
                self.pretreatment_variable_name: self.pred_xi,
                self.group_variable_name: np.ones(self.pred_xi.shape),
            }
        )
        (new_x_treated,) = build_design_matrices([self._x_design_info], x_pred_treated)
        self.pred_treated = self.model.predict(X=np.asarray(new_x_treated))

        # Evaluate causal impact as equal to the trestment effect
        self.causal_impact = self.idata.posterior["beta"].sel(
            {"coeffs": self._get_treatment_effect_coeff()}
        )

        # ================================================================

    def _input_validation(self) -> None:
        """Validate the input data and model formula for correctness"""
        if not _is_variable_dummy_coded(self.data[self.group_variable_name]):
            raise DataException(
                f"""
                There must be 2 levels of the grouping variable
                {self.group_variable_name}. I.e. the treated and untreated.
                """
            )

    def plot(self, round_to=None):
        """Plot the results

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """
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
        az.plot_posterior(self.causal_impact, ref_val=0, ax=ax[1], round_to=round_to)
        ax[1].set(title="Estimated treatment effect")
        return fig, ax

    def _causal_impact_summary_stat(self, round_to) -> str:
        """Computes the mean and 94% credible interval bounds for the causal impact."""
        percentiles = self.causal_impact.quantile([0.03, 1 - 0.03]).values
        ci = (
            r"$CI_{94%}$"
            + f"[{round_num(percentiles[0], round_to)}, {round_num(percentiles[1], round_to)}]"
        )
        causal_impact = f"{round_num(self.causal_impact.mean(), round_to)}, "
        return f"Causal impact = {causal_impact + ci}"

    def summary(self, round_to=None) -> None:
        """
        Print text output summarising the results

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """

        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print("\nResults:")
        # TODO: extra experiment specific outputs here
        print(self._causal_impact_summary_stat(round_to))
        self.print_coefficients(round_to)

    def _get_treatment_effect_coeff(self) -> str:
        """Find the beta regression coefficient corresponding to the
        group (i.e. treatment) effect.
        For example if self.group_variable_name is 'group' and
        the labels are `['Intercept', 'C(group)[T.1]', 'pre']`
        then we want `C(group)[T.1]`.
        """
        for label in self.labels:
            if (self.group_variable_name in label) & (":" not in label):
                return label

        raise NameError("Unable to find coefficient name for the treatment effect")


class InstrumentalVariable(ExperimentalDesign):
    """
    A class to analyse instrumental variable style experiments.

    :param instruments_data: A pandas dataframe of instruments
                             for our treatment variable. Should contain
                             instruments Z, and treatment t
    :param data: A pandas dataframe of covariates for fitting
                 the focal regression of interest. Should contain covariates X
                 including treatment t and outcome y
    :param instruments_formula: A statistical model formula for
                                the instrumental stage regression
                                e.g. t ~ 1 + z1 + z2 + z3
    :param formula: A statistical model formula for the \n
                    focal regression e.g. y ~ 1 + t + x1 + x2 + x3
    :param model: A PyMC model
    :param priors: An optional dictionary of priors for the
                   mus and sigmas of both regressions. If priors are not
                   specified we will substitue MLE estimates for the beta
                   coefficients. Greater control can be achieved
                   by specifying the priors directly e.g. priors = {
                                    "mus": [0, 0],
                                    "sigmas": [1, 1],
                                    "eta": 2,
                                    "lkj_sd": 2,
                                    }

    Example
    --------
    >>> import pandas as pd
    >>> import causalpy as cp
    >>> from causalpy.pymc_experiments import InstrumentalVariable
    >>> from causalpy.pymc_models import InstrumentalVariableRegression
    >>> import numpy as np
    >>> N = 100
    >>> e1 = np.random.normal(0, 3, N)
    >>> e2 = np.random.normal(0, 1, N)
    >>> Z = np.random.uniform(0, 1, N)
    >>> ## Ensure the endogeneity of the the treatment variable
    >>> X = -1 + 4 * Z + e2 + 2 * e1
    >>> y = 2 + 3 * X + 3 * e1
    >>> test_data = pd.DataFrame({"y": y, "X": X, "Z": Z})
    >>> sample_kwargs = {
    ...     "tune": 1,
    ...     "draws": 5,
    ...     "chains": 1,
    ...     "cores": 4,
    ...     "target_accept": 0.95,
    ...     "progressbar": False,
    ...     }
    >>> instruments_formula = "X  ~ 1 + Z"
    >>> formula = "y ~  1 + X"
    >>> instruments_data = test_data[["X", "Z"]]
    >>> data = test_data[["y", "X"]]
    >>> iv = InstrumentalVariable(
    ...         instruments_data=instruments_data,
    ...         data=data,
    ...         instruments_formula=instruments_formula,
    ...         formula=formula,
    ...         model=InstrumentalVariableRegression(sample_kwargs=sample_kwargs),
    ... )
    """

    def __init__(
        self,
        instruments_data: pd.DataFrame,
        data: pd.DataFrame,
        instruments_formula: str,
        formula: str,
        model=None,
        priors=None,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.expt_type = "Instrumental Variable Regression"
        self.data = data
        self.instruments_data = instruments_data
        self.formula = formula
        self.instruments_formula = instruments_formula
        self.model = model
        self._input_validation()

        y, X = dmatrices(formula, self.data)
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.y, self.X = np.asarray(y), np.asarray(X)
        self.outcome_variable_name = y.design_info.column_names[0]

        t, Z = dmatrices(instruments_formula, self.instruments_data)
        self._t_design_info = t.design_info
        self._z_design_info = Z.design_info
        self.labels_instruments = Z.design_info.column_names
        self.t, self.Z = np.asarray(t), np.asarray(Z)
        self.instrument_variable_name = t.design_info.column_names[0]

        self.get_naive_OLS_fit()
        self.get_2SLS_fit()

        # fit the model to the data
        COORDS = {"instruments": self.labels_instruments, "covariates": self.labels}
        self.coords = COORDS
        if priors is None:
            priors = {
                "mus": [self.ols_beta_first_params, self.ols_beta_second_params],
                "sigmas": [1, 1],
                "eta": 2,
                "lkj_sd": 2,
            }
        self.priors = priors
        self.model.fit(
            X=self.X, Z=self.Z, y=self.y, t=self.t, coords=COORDS, priors=self.priors
        )

    def get_2SLS_fit(self):
        """
        Two Stage Least Squares Fit

        This function is called by the experiment, results are used for
        priors if none are provided.
        """
        first_stage_reg = sk_lin_reg().fit(self.Z, self.t)
        fitted_Z_values = first_stage_reg.predict(self.Z)
        X2 = self.data.copy(deep=True)
        X2[self.instrument_variable_name] = fitted_Z_values
        _, X2 = dmatrices(self.formula, X2)
        second_stage_reg = sk_lin_reg().fit(X=X2, y=self.y)
        betas_first = list(first_stage_reg.coef_[0][1:])
        betas_first.insert(0, first_stage_reg.intercept_[0])
        betas_second = list(second_stage_reg.coef_[0][1:])
        betas_second.insert(0, second_stage_reg.intercept_[0])
        self.ols_beta_first_params = betas_first
        self.ols_beta_second_params = betas_second
        self.first_stage_reg = first_stage_reg
        self.second_stage_reg = second_stage_reg

    def get_naive_OLS_fit(self):
        """
        Naive Ordinary Least Squares

        This function is called by the experiment.
        """
        ols_reg = sk_lin_reg().fit(self.X, self.y)
        beta_params = list(ols_reg.coef_[0][1:])
        beta_params.insert(0, ols_reg.intercept_[0])
        self.ols_beta_params = dict(zip(self._x_design_info.column_names, beta_params))
        self.ols_reg = ols_reg

    def _input_validation(self):
        """Validate the input data and model formula for correctness"""
        treatment = self.instruments_formula.split("~")[0]
        test = treatment.strip() in self.instruments_data.columns
        test = test & (treatment.strip() in self.data.columns)
        if not test:
            raise DataException(
                f"""
                The treatment variable:
                {treatment} must appear in the instrument_data to be used
                as an outcome variable and in the data object to be used as a covariate.
                """
            )
        Z = self.data[treatment.strip()]
        check_binary = len(np.unique(Z)) > 2
        if check_binary:
            warnings.warn(
                """Warning. The treatment variable is not Binary.
                This is not necessarily a problem but it violates
                the assumption of a simple IV experiment.
                The coefficients should be interpreted appropriately."""
            )
