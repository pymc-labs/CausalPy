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
Inverse propensity weighting
"""

from typing import List

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from patsy import dmatrices
from sklearn.linear_model import LinearRegression as sk_lin_reg

from causalpy.custom_exceptions import DataException

from .base import BaseExperiment


class InversePropensityWeighting(BaseExperiment):
    """
    A class to analyse inverse propensity weighting experiments.

    :param data:
        A pandas dataframe
    :param formula:
        A statistical model formula for the propensity model
    :param outcome_variable
        A string denoting the outcome variable in datq to be reweighted
    :param weighting_scheme:
        A string denoting which weighting scheme to use among: 'raw', 'robust',
        'doubly robust' or 'overlap'. See Aronow and Miller "Foundations
        of Agnostic Statistics" for discussion and computation of these
        weighting schemes.
    :param model:
        A PyMC model

    Example
    --------
    >>> import causalpy as cp
    >>> df = cp.load_data("nhefs")
    >>> seed = 42
    >>> result = cp.InversePropensityWeighting(
    ...     df,
    ...     formula="trt ~ 1 + age + race",
    ...     outcome_variable="outcome",
    ...     weighting_scheme="robust",
    ...     model=cp.pymc_models.PropensityScore(
    ...         sample_kwargs={
    ...             "draws": 100,
    ...             "target_accept": 0.95,
    ...             "random_seed": seed,
    ...             "progressbar": False,
    ...         },
    ...     ),
    ... )
    """

    supports_ols = False
    supports_bayes = True

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        outcome_variable: str,
        weighting_scheme: str,
        model=None,
        **kwargs,
    ):
        super().__init__(model=model)
        self.expt_type = "Inverse Propensity Score Weighting"
        self.data = data
        self.formula = formula
        self.outcome_variable = outcome_variable
        self.weighting_scheme = weighting_scheme
        self.input_validation()

        t, X = dmatrices(formula, self.data)
        self._t_design_info = t.design_info
        self._t_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.t, self.X = np.asarray(t), np.asarray(X)
        self.y = self.data[self.outcome_variable]

        COORDS = {"obs_ind": list(range(self.X.shape[0])), "coeffs": self.labels}
        self.coords = COORDS
        self.model.fit(X=self.X, t=self.t, coords=COORDS)

    def input_validation(self):
        """Validate the input data and model formula for correctness"""
        treatment = self.formula.split("~")[0]
        test = treatment.strip() in self.data.columns
        test = test & (self.outcome_variable in self.data.columns)
        if not test:
            raise DataException(
                f"""
                The treatment variable:
                {treatment} must appear in the data to be used
                as an outcome variable. And {self.outcome_variable}
                must also be available in the data to be re-weighted
                """
            )
        T = self.data[treatment.strip()]
        check_binary = len(np.unique(T)) > 2
        if check_binary:
            raise DataException(
                """Warning. The treatment variable is not 0-1 Binary.
                """
            )

    def make_robust_adjustments(self, ps):
        """This estimator is discussed in Aronow
        and Miller's book as being related to the
        Horvitz Thompson method"""
        X = pd.DataFrame(self.X, columns=self.labels)
        X["ps"] = ps
        X[self.outcome_variable] = self.y
        t = self.t.flatten()
        p_of_t = np.mean(t)
        X["i_ps"] = np.where(t == 1, (p_of_t / X["ps"]), (1 - p_of_t) / (1 - X["ps"]))
        n_ntrt = X[t == 0].shape[0]
        n_trt = X[t == 1].shape[0]
        outcome_trt = X[t == 1][self.outcome_variable]
        outcome_ntrt = X[t == 0][self.outcome_variable]
        i_propensity0 = X[t == 0]["i_ps"]
        i_propensity1 = X[t == 1]["i_ps"]
        weighted_outcome1 = outcome_trt * i_propensity1
        weighted_outcome0 = outcome_ntrt * i_propensity0
        return weighted_outcome0, weighted_outcome1, n_ntrt, n_trt

    def make_raw_adjustments(self, ps):
        """This estimator is discussed in Aronow and
        Miller as the simplest of base form of
        inverse propensity weighting schemes"""
        X = pd.DataFrame(self.X, columns=self.labels)
        X["ps"] = ps
        X[self.outcome_variable] = self.y
        t = self.t.flatten()
        X["ps"] = np.where(t, X["ps"], 1 - X["ps"])
        X["i_ps"] = 1 / X["ps"]
        n_ntrt = n_trt = len(X)
        outcome_trt = X[t == 1][self.outcome_variable]
        outcome_ntrt = X[t == 0][self.outcome_variable]
        i_propensity0 = X[t == 0]["i_ps"]
        i_propensity1 = X[t == 1]["i_ps"]
        weighted_outcome1 = outcome_trt * i_propensity1
        weighted_outcome0 = outcome_ntrt * i_propensity0
        return weighted_outcome0, weighted_outcome1, n_ntrt, n_trt

    def make_overlap_adjustments(self, ps):
        """This weighting scheme was adapted from
        Lucy D’Agostino McGowan's blog on
        Propensity Score Weights referenced in
        the primary CausalPy explainer notebook"""
        X = pd.DataFrame(self.X, columns=self.labels)
        X["ps"] = ps
        X[self.outcome_variable] = self.y
        t = self.t.flatten()
        X["i_ps"] = np.where(t, (1 - X["ps"]) * t, X["ps"] * (1 - t))
        n_ntrt = (1 - t[t == 0]) * X[t == 0]["i_ps"]
        n_trt = t[t == 1] * X[t == 1]["i_ps"]
        outcome_trt = X[t == 1][self.outcome_variable]
        outcome_ntrt = X[t == 0][self.outcome_variable]
        i_propensity0 = X[t == 0]["i_ps"]
        i_propensity1 = X[t == 1]["i_ps"]
        weighted_outcome1 = t[t == 1] * outcome_trt * i_propensity1
        weighted_outcome0 = (1 - t[t == 0]) * outcome_ntrt * i_propensity0
        return weighted_outcome0, weighted_outcome1, n_ntrt, n_trt

    def make_doubly_robust_adjustment(self, ps):
        """The doubly robust weighting scheme is also
        discussed in Aronow and Miller, but a bit more generally
        than our implementation here. Here we have specified
        the outcome model to be a simple OLS model.
        In this way the compromise between the outcome model and
        the propensity model is always done with OLS."""
        X = pd.DataFrame(self.X, columns=self.labels)
        X["ps"] = ps
        t = self.t.flatten()
        m0 = sk_lin_reg().fit(X[t == 0].astype(float), self.y[t == 0])
        m1 = sk_lin_reg().fit(X[t == 1].astype(float), self.y[t == 1])
        m0_pred = m0.predict(X)
        m1_pred = m1.predict(X)
        ## Compromise between outcome and treatment assignment model
        weighted_outcome0 = (1 - t) * (self.y - m0_pred) / (1 - X["ps"]) + m0_pred
        weighted_outcome1 = t * (self.y - m1_pred) / X["ps"] + m1_pred
        return weighted_outcome0, weighted_outcome1, None, None

    def get_ate(self, i, idata, method="doubly_robust"):
        ### Post processing the sample posterior distribution for propensity scores
        ### One sample at a time.
        ps = idata["posterior"]["p"].stack(z=("chain", "draw"))[:, i].values
        if method == "robust":
            (
                weighted_outcome_ntrt,
                weighted_outcome_trt,
                n_ntrt,
                n_trt,
            ) = self.make_robust_adjustments(ps)
            ntrt = weighted_outcome_ntrt.sum() / n_ntrt
            trt = weighted_outcome_trt.sum() / n_trt
        elif method == "raw":
            (
                weighted_outcome_ntrt,
                weighted_outcome_trt,
                n_ntrt,
                n_trt,
            ) = self.make_raw_adjustments(ps)
            ntrt = weighted_outcome_ntrt.sum() / n_ntrt
            trt = weighted_outcome_trt.sum() / n_trt
        elif method == "overlap":
            (
                weighted_outcome_ntrt,
                weighted_outcome_trt,
                n_ntrt,
                n_trt,
            ) = self.make_overlap_adjustments(ps)
            ntrt = np.sum(weighted_outcome_ntrt) / np.sum(n_ntrt)
            trt = np.sum(weighted_outcome_trt) / np.sum(n_trt)
        else:
            (
                weighted_outcome_ntrt,
                weighted_outcome_trt,
                n_ntrt,
                n_trt,
            ) = self.make_doubly_robust_adjustment(ps)
            trt = np.mean(weighted_outcome_trt)
            ntrt = np.mean(weighted_outcome_ntrt)
        ate = trt - ntrt
        return [ate, trt, ntrt]

    def plot_ate(
        self, idata=None, method=None, prop_draws=100, ate_draws=300
    ) -> tuple[plt.Figure, List[plt.Axes]]:
        if idata is None:
            idata = self.model.idata
        if method is None:
            method = self.weighting_scheme

        def plot_weights(bins, top0, top1, ax, color="population"):
            colors_dict = {
                "population": ["orange", "skyblue", 0.6],
                "pseudo_population": ["grey", "grey", 0.1],
            }

            ax.axhline(0, c="gray", linewidth=1)
            bars0 = ax.bar(
                bins[:-1] + 0.025,
                top0,
                width=0.04,
                facecolor=colors_dict[color][0],
                alpha=colors_dict[color][2],
            )
            bars1 = ax.bar(
                bins[:-1] + 0.025,
                -top1,
                width=0.04,
                facecolor=colors_dict[color][1],
                alpha=colors_dict[color][2],
            )

            for bars in (bars0, bars1):
                for bar in bars:
                    bar.set_edgecolor("black")

        def make_hists(idata, i, axs, method=method):
            p_i = az.extract(idata)["p"][:, i].values
            if method == "raw":
                weight0 = 1 / (1 - p_i[self.t.flatten() == 0])
                weight1 = 1 / (p_i[self.t.flatten() == 1])
            elif method == "overlap":
                t = self.t.flatten()
                weight1 = (1 - p_i[t == 1]) * t[t == 1]
                weight0 = p_i[t == 0] * (1 - t[t == 0])
            else:
                t = self.t.flatten()
                p_of_t = np.mean(t)
                weight1 = p_of_t / p_i[t == 1]
                weight0 = (1 - p_of_t) / (1 - p_i[t == 0])
            bins = np.arange(0.025, 0.99, 0.005)
            top0, _ = np.histogram(p_i[self.t.flatten() == 0], bins=bins)
            top1, _ = np.histogram(p_i[self.t.flatten() == 1], bins=bins)
            plot_weights(bins, top0, top1, axs[0])
            top0, _ = np.histogram(
                p_i[self.t.flatten() == 0], bins=bins, weights=weight0
            )
            top1, _ = np.histogram(
                p_i[self.t.flatten() == 1], bins=bins, weights=weight1
            )
            plot_weights(bins, top0, top1, axs[0], color="pseudo_population")

        mosaic = """AAAAAA
                    BBBBCC"""

        fig, axs = plt.subplot_mosaic(mosaic, figsize=(20, 13))
        axs = [axs[k] for k in axs.keys()]
        axs[0].axvline(
            0.1, linestyle="--", label="Low Extreme Propensity Scores", color="black"
        )
        axs[0].axvline(
            0.9, linestyle="--", label="Hi Extreme Propensity Scores", color="black"
        )
        axs[0].set_title(
            "Weighted and Unweighted Draws from the Posterior \n  Propensity Scores Distribution",
            fontsize=20,
        )
        axs[0].set_ylabel("Counts of Observations")
        axs[0].set_xlabel("Propensity Scores")
        custom_lines = [
            Line2D([0], [0], color="skyblue", lw=2),
            Line2D([0], [0], color="orange", lw=2),
            Line2D([0], [0], color="grey", lw=2),
            Line2D([0], [0], color="black", lw=2, linestyle="--"),
        ]

        axs[0].legend(
            custom_lines,
            ["Treatment PS", "Control PS", "Weighted Pseudo Population", "Extreme PS"],
        )

        [make_hists(idata, i, axs) for i in range(prop_draws)]
        ate_df = pd.DataFrame(
            [self.get_ate(i, idata, method=method) for i in range(ate_draws)],
            columns=["ATE", "Y(1)", "Y(0)"],
        )
        axs[1].hist(
            ate_df["Y(1)"],
            label="E(Y(1))",
            ec="black",
            bins=10,
            alpha=0.6,
            color="skyblue",
        )
        axs[1].hist(
            ate_df["Y(0)"],
            label="E(Y(0))",
            ec="black",
            bins=10,
            alpha=0.6,
            color="orange",
        )
        axs[1].legend()
        axs[1].set_xlabel(self.outcome_variable)
        axs[1].set_title(
            f"The Outcomes \n Under the {method} re-weighting scheme", fontsize=20
        )
        axs[2].hist(
            ate_df["ATE"],
            label="ATE",
            ec="black",
            bins=10,
            color="grey",
            alpha=0.6,
        )
        axs[2].set_xlabel("Difference")
        axs[2].axvline(ate_df["ATE"].mean(), label="E(ATE)")
        axs[2].legend()
        axs[2].set_title("Average Treatment Effect", fontsize=20)

        return fig, axs

    def weighted_percentile(self, data, weights, perc):
        """
        perc : percentile in [0-1]!
        """
        if not (0 <= perc <= 1):
            raise ValueError("Percentile must be between 0 and 1.")
        ix = np.argsort(data)
        data = data[ix]  # sort data
        weights = weights[ix]  # sort weights
        cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(
            weights
        )  # 'like' a CDF function
        return np.interp(perc, cdf, data)

    def plot_balance_ecdf(
        self, covariate, idata=None, weighting_scheme=None
    ) -> tuple[plt.Figure, List[plt.Axes]]:
        """
        Plotting function takes a single covariate and shows the
        differences in the ECDF between the treatment and control
        groups before and after weighting. It provides a visual
        check on the balance achieved by using the different weighting
        schemes
        """
        if idata is None:
            idata = self.model.idata
        if weighting_scheme is None:
            weighting_scheme = self.weighting_scheme

        ps = az.extract(idata)["p"].mean(dim="sample").values
        X = pd.DataFrame(self.X, columns=self.labels)
        X["ps"] = ps
        t = self.t.flatten()
        if weighting_scheme == "raw":
            w1 = 1 / ps[t == 1]
            w0 = 1 / (1 - ps[t == 0])
        elif weighting_scheme == "robust":
            p_of_t = np.mean(t)
            w1 = p_of_t / (ps[t == 1])
            w0 = (1 - p_of_t) / (1 - ps[t == 0])
        else:
            w1 = (1 - ps[t == 1]) * t[t == 1]
            w0 = ps[t == 0] * (1 - t[t == 0])
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))
        raw_trt = [
            self.weighted_percentile(
                X[t == 1][covariate].values, np.ones(len(X[t == 1])), p
            )
            for p in np.linspace(0, 1, 1000)
        ]
        raw_ntrt = [
            self.weighted_percentile(
                X[t == 0][covariate].values, np.ones(len(X[t == 0])), p
            )
            for p in np.linspace(0, 1, 1000)
        ]
        w_trt = [
            self.weighted_percentile(X[t == 1][covariate].values, w1, p)
            for p in np.linspace(0, 1, 1000)
        ]
        w_ntrt = [
            self.weighted_percentile(X[t == 0][covariate].values, w0, p)
            for p in np.linspace(0, 1, 1000)
        ]
        axs[0].plot(
            np.linspace(0, 1, 1000), raw_trt, color="skyblue", label="Raw Treated"
        )
        axs[0].plot(
            np.linspace(0, 1, 1000), raw_ntrt, color="orange", label="Raw Control"
        )
        axs[0].set_title(f"ECDF \n Raw: {covariate}")
        axs[1].set_title(
            f"ECDF \n Weighted {weighting_scheme} adjustment for {covariate}"
        )
        axs[1].plot(
            np.linspace(0, 1, 1000), w_trt, color="skyblue", label="Reweighted Treated"
        )
        axs[1].plot(
            np.linspace(0, 1, 1000),
            w_ntrt,
            color="orange",
            label="Reweighted Control",
        )
        axs[1].set_xlabel("Quantiles")
        axs[0].set_xlabel("Quantiles")
        axs[1].legend()
        axs[0].legend()
        # TODO: for some reason ax is type numpy.ndarray, so we need to convert this back to a list to conform to the expected return type.
        return fig, list(axs)
