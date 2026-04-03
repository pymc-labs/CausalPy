#   Copyright 2025 - 2026 The PyMC Labs Developers
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
Synthetic Difference-in-Differences Experiment
"""

import warnings
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.base import RegressorMixin

from causalpy.custom_exceptions import BadIndexException
from causalpy.date_utils import _combine_datetime_indices, format_date_axes
from causalpy.plot_utils import plot_xY
from causalpy.pymc_models import PyMCModel, SyntheticDifferenceInDifferencesWeightFitter
from causalpy.reporting import EffectSummary

from .base import BaseExperiment

LEGEND_FONT_SIZE = 12


class SyntheticDifferenceInDifferences(BaseExperiment):
    """Bayesian Synthetic Difference-in-Differences experiment.

    Combines the synthetic control method's unit weighting with
    difference-in-differences time weighting. The treatment effect (tau) is
    computed analytically from the posterior weight distributions via the
    double-difference formula, rather than being estimated inside the MCMC
    model (cut-posterior formulation).

    :param data:
        A pandas dataframe in wide format (columns = units, rows = time periods).
    :param treatment_time:
        The time when treatment occurred, should be in reference to the data index.
    :param control_units:
        A list of control unit column names.
    :param treated_units:
        A list of treated unit column names.
    :param model:
        A SyntheticDifferenceInDifferencesWeightFitter instance.
        Defaults to SyntheticDifferenceInDifferencesWeightFitter.

    Example
    --------
    >>> import causalpy as cp
    >>> df = cp.load_data("sc")
    >>> treatment_time = 70
    >>> result = cp.SyntheticDifferenceInDifferences(
    ...     df,
    ...     treatment_time,
    ...     control_units=["a", "b", "c", "d", "e", "f", "g"],
    ...     treated_units=["actual"],
    ...     model=cp.pymc_models.SyntheticDifferenceInDifferencesWeightFitter(
    ...         sample_kwargs={
    ...             "tune": 20,
    ...             "draws": 20,
    ...             "chains": 2,
    ...             "cores": 2,
    ...             "progressbar": False,
    ...         }
    ...     ),
    ... )

    Notes
    -----
    This implements Bayesian SDiD method. The model fits two weight modules via
    MCMC:

    - **Unit weights** (omega): balance control units against treated units in the
      pre-treatment period, similar to synthetic control.
    - **Time weights** (lambda): balance pre-treatment periods against
      post-treatment periods for control units.

    The treatment effect is then computed analytically via the double-difference:

    .. math::
        \\tau = \\bar{\\Delta}_{\\text{post}} - \\boldsymbol{\\lambda}^\\top \\boldsymbol{\\Delta}_{\\text{pre}}

    where :math:`\\Delta_t = y_{\\text{tr},t} - (\\omega_0 + \\boldsymbol{\\omega}^\\top \\mathbf{Y}_{\\text{co},t})`
    is the gap between the observed treated outcome and the synthetic control at
    time *t*.

    References
    ----------
    .. [1] Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., &
       Wager, S. (2021). Synthetic Difference-in-Differences. *American
       Economic Review*, 111(12), 4088-4118.
    """

    supports_ols = True
    supports_bayes = True
    _default_model_class = SyntheticDifferenceInDifferencesWeightFitter

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_time: int | float | pd.Timestamp,
        control_units: list[str],
        treated_units: list[str],
        model: PyMCModel | RegressorMixin | None = None,
        **kwargs: dict,
    ) -> None:
        super().__init__(model=model)
        # rename the index to "obs_ind"
        data.index.name = "obs_ind"
        self.data = data
        self.input_validation(data, treatment_time)
        self.treatment_time = treatment_time
        self.control_units = control_units
        self.labels = control_units
        self.treated_units = treated_units
        self.expt_type = "SyntheticDifferenceInDifferences"
        self._prepare_data()
        self.algorithm()

    @property
    def datapre(self) -> pd.DataFrame:
        """Data from before the treatment time (exclusive).

        Pre-period: index < treatment_time
        """
        return self.data[self.data.index < self.treatment_time]

    @property
    def datapost(self) -> pd.DataFrame:
        """Data from on or after the treatment time (inclusive).

        Post-period: index >= treatment_time
        """
        return self.data[self.data.index >= self.treatment_time]

    def input_validation(
        self, data: pd.DataFrame, treatment_time: int | float | pd.Timestamp
    ) -> None:
        """Validate the input data for correctness."""
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

    def _prepare_data(self) -> None:
        """Prepare xarray DataArrays for control and treated units in pre/post periods.

        Also constructs the dict-based inputs expected by
        SyntheticDifferenceInDifferencesWeightFitter.
        """
        # Four-quadrant split as xarray DataArrays (same as SyntheticControl)
        self.datapre_control = xr.DataArray(
            self.datapre[self.control_units],
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": self.datapre[self.control_units].index,
                "coeffs": self.control_units,
            },
        )
        self.datapre_treated = xr.DataArray(
            self.datapre[self.treated_units],
            dims=["obs_ind", "treated_units"],
            coords={
                "obs_ind": self.datapre[self.treated_units].index,
                "treated_units": self.treated_units,
            },
        )
        self.datapost_control = xr.DataArray(
            self.datapost[self.control_units],
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": self.datapost[self.control_units].index,
                "coeffs": self.control_units,
            },
        )
        self.datapost_treated = xr.DataArray(
            self.datapost[self.treated_units],
            dims=["obs_ind", "treated_units"],
            coords={
                "obs_ind": self.datapost[self.treated_units].index,
                "treated_units": self.treated_units,
            },
        )

    def algorithm(self) -> None:
        """Run the SDiD algorithm: fit weight modules, compute tau analytically.

        Steps:
        1. Prepare dict-based X/y inputs for the weight fitter.
        2. Fit the model (both omega and lambda modules via MCMC).
        3. Extract weight posteriors.
        4. Compute the synthetic control for all time points.
        5. Compute gaps (treated - synthetic control).
        6. Compute tau via the double-difference formula.
        7. Construct xarray objects for reporting compatibility.
        """
        if isinstance(self.model, RegressorMixin):
            raise NotImplementedError(
                "OLS estimation for SyntheticDifferenceInDifferences is not yet "
                "implemented. Please use a PyMC model."
            )

        # Full panel data as numpy
        Y_co = self.data[self.control_units].values.T  # (N_co, T)
        y_tr = self.data[self.treated_units].values.mean(
            axis=1
        )  # (T,) mean across treated units

        T_pre = self.datapre.shape[0]

        # ---- Prepare inputs for SyntheticDifferenceInDifferencesWeightFitter ----
        # Module 1 (unit weights): X_unit = Y_co_pre.T (T_pre x N_co),
        #                          y_unit = y_tr_pre (T_pre,)
        X_unit = xr.DataArray(
            Y_co[:, :T_pre].T,  # (T_pre, N_co)
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": np.arange(T_pre),
                "coeffs": self.control_units,
            },
        )
        y_unit = xr.DataArray(
            y_tr[:T_pre],
            dims=["obs_ind"],
            coords={"obs_ind": np.arange(T_pre)},
        )

        # Module 2 (time weights): X_time = Y_co_pre (N_co x T_pre),
        #                          y_time = Y_co_post_mean (N_co,)
        Y_co_post_mean = Y_co[:, T_pre:].mean(axis=1)  # (N_co,)
        X_time = xr.DataArray(
            Y_co[:, :T_pre],  # (N_co, T_pre)
            dims=["coeffs", "obs_ind"],
            coords={
                "coeffs": self.control_units,
                "obs_ind": np.arange(T_pre),
            },
        )
        y_time = xr.DataArray(
            Y_co_post_mean,
            dims=["coeffs"],
            coords={"coeffs": self.control_units},
        )

        X = {"unit": X_unit, "time": X_time}
        y = {"unit": y_unit, "time": y_time}

        COORDS = {
            "coeffs": self.control_units,
            "obs_ind": np.arange(T_pre),
            "coeffs_raw": self.control_units[1:],
            "obs_ind_raw": list(range(1, T_pre)),
        }

        self.model.fit(X=X, y=y, coords=COORDS)
        assert self.model.idata is not None, "Model fitting failed to produce idata"

        # ---- Extract weight posteriors ----
        omega = self.model.idata.posterior["omega"].values  # (chain, draw, N_co)
        lam = self.model.idata.posterior["lam"].values  # (chain, draw, T_pre)
        omega0 = self.model.idata.posterior["omega0"].values  # (chain, draw)

        n_chains, n_draws = omega.shape[0], omega.shape[1]

        # ---- Compute synthetic control for ALL time points ----
        # sc_t = omega0 + omega @ Y_co_t for each time t
        sc_all = omega0[..., np.newaxis] + np.einsum(
            "cdn,nt->cdt", omega, Y_co
        )  # (chain, draw, T)

        # ---- Compute gaps ----
        gaps = y_tr[np.newaxis, np.newaxis, :] - sc_all  # (chain, draw, T)

        # ---- Tau via double-difference ----
        # tau = mean(gaps_post) - lam^T @ gaps_pre
        gaps_post_mean = gaps[..., T_pre:].mean(axis=-1)  # (chain, draw)
        lam_gaps_pre = (lam * gaps[..., :T_pre]).sum(axis=-1)  # (chain, draw)
        tau = gaps_post_mean - lam_gaps_pre  # (chain, draw)

        # Store tau posterior as xarray
        self.tau_posterior = xr.DataArray(
            tau,
            dims=["chain", "draw"],
            coords={
                "chain": np.arange(n_chains),
                "draw": np.arange(n_draws),
            },
        )

        # ---- Build xarray objects for reporting compatibility ----
        # The reporting helpers expect:
        # - self.post_pred with .posterior_predictive["mu"] having dims
        #   (chain, draw, obs_ind) and optionally treated_units
        # - self.post_impact as xr.DataArray with dims (chain, draw, obs_ind)
        #   and optionally treated_units
        # - self.pre_pred, self.pre_impact similarly for pre-period

        # Synthetic control predictions for pre and post periods
        sc_pre = sc_all[..., :T_pre]  # (chain, draw, T_pre)
        sc_post = sc_all[..., T_pre:]  # (chain, draw, T_post)

        # Pre-period predictions (synthetic control, no treatment effect)
        self.pre_pred = self._build_inference_data(
            sc_pre,
            self.datapre.index,
            n_chains,
            n_draws,
        )

        # Post-period counterfactual (what would have happened without treatment)
        self.post_pred = self._build_inference_data(
            sc_post,
            self.datapost.index,
            n_chains,
            n_draws,
        )

        # Impact: observed - counterfactual
        # For pre-period, the "treated" is the actual observed treated outcome
        y_tr_pre = self.datapre[self.treated_units].values.mean(axis=1)  # (T_pre,)
        y_tr_post = self.datapost[self.treated_units].values.mean(axis=1)  # (T_post,)

        pre_impact_vals = (
            y_tr_pre[np.newaxis, np.newaxis, :] - sc_pre
        )  # (chain, draw, T_pre)
        post_impact_vals = (
            y_tr_post[np.newaxis, np.newaxis, :] - sc_post
        )  # (chain, draw, T_post)

        # Add a singleton treated_units dim for compatibility with reporting helpers
        self.pre_impact = xr.DataArray(
            pre_impact_vals[..., np.newaxis],
            dims=["chain", "draw", "obs_ind", "treated_units"],
            coords={
                "chain": np.arange(n_chains),
                "draw": np.arange(n_draws),
                "obs_ind": self.datapre.index,
                "treated_units": [self.treated_units[0]],
            },
        )

        self.post_impact = xr.DataArray(
            post_impact_vals[..., np.newaxis],
            dims=["chain", "draw", "obs_ind", "treated_units"],
            coords={
                "chain": np.arange(n_chains),
                "draw": np.arange(n_draws),
                "obs_ind": self.datapost.index,
                "treated_units": [self.treated_units[0]],
            },
        )

        self.post_impact_cumulative = self.post_impact.cumsum(dim="obs_ind")

    def _build_inference_data(
        self,
        mu_vals: np.ndarray,
        index: pd.Index,
        n_chains: int,
        n_draws: int,
    ) -> az.InferenceData:
        """Build an InferenceData-like object with posterior_predictive group.

        Constructs a minimal InferenceData containing a ``mu`` variable in the
        ``posterior_predictive`` group, shaped to be compatible with the
        reporting helpers that expect SC-style predictions.

        Parameters
        ----------
        mu_vals : np.ndarray
            Array of shape (chain, draw, T) with the mean predictions.
        index : pd.Index
            Time index for the obs_ind coordinate.
        n_chains : int
            Number of MCMC chains.
        n_draws : int
            Number of MCMC draws per chain.

        Returns
        -------
        az.InferenceData
            InferenceData with posterior_predictive group containing ``mu``.
        """
        # Add a singleton treated_units dim: (chain, draw, T, 1)
        mu_4d = mu_vals[..., np.newaxis]

        mu_da = xr.DataArray(
            mu_4d,
            dims=["chain", "draw", "obs_ind", "treated_units"],
            coords={
                "chain": np.arange(n_chains),
                "draw": np.arange(n_draws),
                "obs_ind": index,
                "treated_units": [self.treated_units[0]],
            },
        )
        ds = xr.Dataset({"mu": mu_da})
        return az.InferenceData(posterior_predictive=ds)

    def summary(self, round_to: int | None = None) -> None:
        """Print summary of main results.

        :param round_to:
            Number of decimals used to round results. Defaults to 2.
            Use ``None`` to return raw numbers.
        """
        round_to = round_to if round_to is not None else 2

        print(f"{self.expt_type:=^80}")
        print(f"Control units: {self.control_units}")
        if len(self.treated_units) > 1:
            print(f"Treated units: {self.treated_units}")
        else:
            print(f"Treated unit: {self.treated_units[0]}")

        tau_mean = float(self.tau_posterior.mean())
        tau_hdi = az.hdi(self.tau_posterior.values.flatten(), hdi_prob=0.94)
        print(
            f"Average treatment effect on the treated (ATT): "
            f"{round(tau_mean, round_to)}"
        )
        print(
            f"  94% HDI: [{round(float(tau_hdi[0]), round_to)}, "
            f"{round(float(tau_hdi[1]), round_to)}]"
        )

    @staticmethod
    def _convert_treatment_time_for_axis(
        axis: plt.Axes, treatment_time: int | float | pd.Timestamp
    ) -> int | float | pd.Timestamp:
        """Convert treatment time into the plotting units expected by a specific axis."""
        try:
            return axis.xaxis.convert_units(treatment_time)
        except (TypeError, ValueError):
            return treatment_time

    def _bayesian_plot(
        self,
        round_to: int | None = None,
        **kwargs: dict,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot the results: counterfactual, impact, and cumulative impact.

        :param round_to:
            Number of decimals used to round results. Defaults to 2.
            Use ``None`` to return raw numbers.
        """
        treated_unit = self.treated_units[0]

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

        # ---- TOP PLOT: Observed vs counterfactual ----
        pre_pred = self.pre_pred.posterior_predictive["mu"].sel(
            treated_units=treated_unit
        )
        post_pred = self.post_pred.posterior_predictive["mu"].sel(
            treated_units=treated_unit
        )

        # Pre-intervention synthetic control fit
        h_line, h_patch = plot_xY(
            self.datapre.index,
            pre_pred,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C0"},
        )
        handles = [(h_line, h_patch)]
        labels = ["Pre-intervention fit"]

        # Observed treated outcome
        (h,) = ax[0].plot(
            self.datapre.index,
            self.datapre[self.treated_units].values.mean(axis=1),
            "k.",
            label="Observations",
        )
        handles.append(h)
        labels.append("Observations")

        # Post-intervention counterfactual
        h_line, h_patch = plot_xY(
            self.datapost.index,
            post_pred,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C1"},
        )
        handles.append((h_line, h_patch))
        labels.append("Counterfactual")

        ax[0].plot(
            self.datapost.index,
            self.datapost[self.treated_units].values.mean(axis=1),
            "k.",
        )

        # Shaded causal effect
        h = ax[0].fill_between(
            self.datapost.index,
            y1=post_pred.mean(dim=["chain", "draw"]).values,
            y2=self.datapost[self.treated_units].values.mean(axis=1),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        handles.append(h)
        labels.append("Causal impact")

        tau_mean = float(self.tau_posterior.mean())
        r_to = round_to if round_to is not None else 2
        ax[0].set(title=f"SDiD: ATT = {round(tau_mean, r_to)}")

        # ---- MIDDLE PLOT: Impact ----
        plot_xY(
            self.datapre.index,
            self.pre_impact.sel(treated_units=treated_unit),
            ax=ax[1],
            plot_hdi_kwargs={"color": "C0"},
        )
        plot_xY(
            self.datapost.index,
            self.post_impact.sel(treated_units=treated_unit),
            ax=ax[1],
            plot_hdi_kwargs={"color": "C1"},
        )
        ax[1].axhline(y=0, c="k")
        ax[1].fill_between(
            self.datapost.index,
            y1=self.post_impact.mean(["chain", "draw"])
            .sel(treated_units=treated_unit)
            .values,
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[1].set(title="Causal Impact")

        # ---- BOTTOM PLOT: Cumulative impact ----
        ax[2].set(title="Cumulative Causal Impact")
        plot_xY(
            self.datapost.index,
            self.post_impact_cumulative.sel(treated_units=treated_unit),
            ax=ax[2],
            plot_hdi_kwargs={"color": "C1"},
        )
        ax[2].axhline(y=0, c="k")

        # Intervention line
        for i in [0, 1, 2]:
            treatment_time = self._convert_treatment_time_for_axis(
                ax[i], self.treatment_time
            )
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

        # Apply intelligent date formatting if data has datetime index
        if isinstance(self.datapre.index, pd.DatetimeIndex):
            full_index = _combine_datetime_indices(
                pd.DatetimeIndex(self.datapre.index),
                pd.DatetimeIndex(self.datapost.index),
            )
            format_date_axes(ax, full_index)

        return fig, ax

    def _ols_plot(self, *args: Any, **kwargs: Any) -> tuple:
        """OLS not supported for SDiD."""
        raise NotImplementedError(
            "OLS models are not supported for "
            "SyntheticDifferenceInDifferences. Use a Bayesian model."
        )

    def effect_summary(
        self,
        *,
        window: Literal["post"] | tuple | slice = "post",
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        cumulative: bool = True,
        relative: bool = True,
        min_effect: float | None = None,
        treated_unit: str | None = None,
        period: Literal["intervention", "post", "comparison"] | None = None,
        prefix: str = "Post-period",
        **kwargs: Any,
    ) -> EffectSummary:
        """Generate a decision-ready summary of causal effects for SDiD.

        Parameters
        ----------
        window : str, tuple, or slice, default="post"
            Time window for analysis.
        direction : {"increase", "decrease", "two-sided"}, default="increase"
            Direction for tail probability calculation.
        alpha : float, default=0.05
            Significance level for HDI intervals.
        cumulative : bool, default=True
            Whether to include cumulative effect statistics.
        relative : bool, default=True
            Whether to include relative effect statistics.
        min_effect : float, optional
            ROPE threshold.
        treated_unit : str, optional
            Which treated unit to analyze. If None, uses first unit.
        period : str, optional
            Ignored for SDiD (two-period design only).
        prefix : str, optional
            Prefix for prose generation. Defaults to "Post-period".

        Returns
        -------
        EffectSummary
            Object with .table (DataFrame) and .text (str) attributes.
        """
        from causalpy.reporting import (
            _compute_statistics,
            _extract_counterfactual,
            _extract_window,
            _generate_prose_detailed,
            _generate_table,
        )

        if period is not None:
            warnings.warn(
                f"period='{period}' is ignored for SyntheticDifferenceInDifferences "
                "(two-period design only). "
                "Results reflect the entire post-treatment period. "
                "Use the 'window' parameter to analyze specific time ranges.",
                UserWarning,
                stacklevel=2,
            )

        # Extract windowed impact data
        windowed_impact, window_coords = _extract_window(
            self, window, treated_unit=treated_unit
        )

        # Extract counterfactual for relative effects
        counterfactual = _extract_counterfactual(
            self, window_coords, treated_unit=treated_unit
        )

        hdi_prob = 1 - alpha
        stats = _compute_statistics(
            windowed_impact,
            counterfactual,
            hdi_prob=hdi_prob,
            direction=direction,
            cumulative=cumulative,
            relative=relative,
            min_effect=min_effect,
        )

        table = _generate_table(stats, cumulative=cumulative, relative=relative)

        # Compute observed/counterfactual averages for prose
        time_dim = "obs_ind"
        cf_avg = float(counterfactual.mean(dim=[time_dim, "chain", "draw"]).values)
        obs_avg = cf_avg + stats["avg"]["mean"]
        cf_cum = float(
            counterfactual.sum(dim=time_dim).mean(dim=["chain", "draw"]).values
        )
        obs_cum = cf_cum + stats["cum"]["mean"] if cumulative else None

        text = _generate_prose_detailed(
            stats,
            window_coords,
            alpha=alpha,
            direction=direction,
            cumulative=cumulative,
            relative=relative,
            prefix=prefix,
            observed_avg=obs_avg,
            counterfactual_avg=cf_avg,
            observed_cum=obs_cum,
            counterfactual_cum=cf_cum if cumulative else None,
            experiment_type="sc",
        )

        return EffectSummary(table=table, text=text)
