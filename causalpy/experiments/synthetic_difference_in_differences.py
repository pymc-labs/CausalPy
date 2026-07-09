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
Synthetic Difference-in-Differences Experiment.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from matplotlib import pyplot as plt
from plotnine import aes, element_text, geom_vline, theme
from sklearn.base import RegressorMixin

from causalpy.constants import HDI_PROB
from causalpy.custom_exceptions import BadIndexException
from causalpy.plot_utils import (
    PlotSpec,
    add_causal_panel_legend,
    build_causal_panel_plot,
    dataarray_draws,
    interval_kind,
    label_draws,
    posterior_histogram_tiles,
    prediction_draws,
    spaghetti_draws,
    summarize_draws,
)
from causalpy.pymc_models import PyMCModel, SyntheticDifferenceInDifferencesWeightFitter
from causalpy.reporting import EffectSummary

from .base import BaseExperiment


@dataclass(frozen=True)
class _SyntheticDiDPlotData:
    """Tidy tables consumed by the declarative synthetic DiD plot."""

    intervals: pd.DataFrame
    observations: pd.DataFrame
    effect_area: pd.DataFrame
    posterior_paths: pd.DataFrame | None
    posterior_density: pd.DataFrame | None


class SyntheticDifferenceInDifferences(BaseExperiment):
    """Bayesian Synthetic Difference-in-Differences experiment.

    Combines the synthetic control method's unit weighting with
    difference-in-differences time weighting. The treatment effect (tau) is
    computed analytically from the posterior weight distributions via the
    double-difference formula, rather than being estimated inside the MCMC
    model (cut-posterior formulation).

    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe in wide format (columns = units, rows = time periods).
    treatment_time : int, float or pandas.Timestamp
        The time when treatment occurred, should be in reference to the data
        index.
    control_units : list of str
        A list of control unit column names.
    treated_units : list of str
        A list of treated unit column names.
    model : PyMCModel or sklearn.base.RegressorMixin, optional
        A ``SyntheticDifferenceInDifferencesWeightFitter`` instance. Defaults
        to ``SyntheticDifferenceInDifferencesWeightFitter``.
    **kwargs : dict
        Additional keyword arguments (currently unused).

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

    Examples
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
    """

    supports_ols = True
    supports_bayes = True
    _default_model_class = SyntheticDifferenceInDifferencesWeightFitter
    _deprecated_design_aliases = {
        "datapre_control": ("pre_design", "control"),
        "datapre_treated": ("pre_design", "treated"),
        "datapost_control": ("post_design", "control"),
        "datapost_treated": ("post_design", "treated"),
    }

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
        """Validate the input data for correctness.

        Parameters
        ----------
        data : pandas.DataFrame
            A dataframe in wide format (columns = units, rows = time periods).
        treatment_time : int, float or pandas.Timestamp
            The time when treatment occurred, should be in reference to the
            data index.
        """
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
        """Bundle control and treated data into ``xr.Dataset`` objects per period.

        Builds ``pre_design`` / ``post_design`` datasets with ``control`` and
        ``treated`` variables, mirroring :class:`SyntheticControl`.
        """
        self.pre_design = xr.Dataset(
            {
                "control": xr.DataArray(
                    self.datapre[self.control_units],
                    dims=["obs_ind", "coeffs"],
                    coords={
                        "obs_ind": self.datapre[self.control_units].index,
                        "coeffs": self.control_units,
                    },
                ),
                "treated": xr.DataArray(
                    self.datapre[self.treated_units],
                    dims=["obs_ind", "treated_units"],
                    coords={
                        "obs_ind": self.datapre[self.treated_units].index,
                        "treated_units": self.treated_units,
                    },
                ),
            }
        )
        self.post_design = xr.Dataset(
            {
                "control": xr.DataArray(
                    self.datapost[self.control_units],
                    dims=["obs_ind", "coeffs"],
                    coords={
                        "obs_ind": self.datapost[self.control_units].index,
                        "coeffs": self.control_units,
                    },
                ),
                "treated": xr.DataArray(
                    self.datapost[self.treated_units],
                    dims=["obs_ind", "treated_units"],
                    coords={
                        "obs_ind": self.datapost[self.treated_units].index,
                        "treated_units": self.treated_units,
                    },
                ),
            }
        )

    def algorithm(self) -> None:
        """Run the SDiD algorithm: fit weight modules, compute tau analytically.

        The method is a thin orchestrator that delegates each step to a
        private helper so that the individual pieces can be unit tested in
        isolation:

        1. :meth:`_build_weight_fitter_inputs` prepares the dict-based ``X``,
           ``y`` and ``coords`` inputs for the weight fitter.
        2. :meth:`PyMCModel.fit` fits both the omega and lambda modules via
           MCMC.
        3. :meth:`_extract_weight_posteriors` pulls the posterior weight
           arrays out of the fitted model.
        4. :meth:`_compute_synthetic_and_gaps` builds the synthetic control
           trajectory and the gap between treated and synthetic.
        5. :meth:`_compute_tau` evaluates the double-difference ATT.
        6. :meth:`_build_reporting_objects` constructs the xarray objects
           required by the reporting helpers.
        """
        if self._model_backend.is_ols:
            raise NotImplementedError(
                "OLS estimation for SyntheticDifferenceInDifferences is not yet "
                "implemented. Please use a PyMC model."
            )

        Y_co = self.data[self.control_units].to_numpy().T  # (N_co, T)
        y_tr = self.data[self.treated_units].to_numpy().mean(axis=1)  # (T,)
        T_pre = self.datapre.shape[0]

        X, y, coords = self._build_weight_fitter_inputs(Y_co, y_tr, T_pre)
        self._model_backend.fit(X=X, y=y, coords=coords)
        if self.model.idata is None:
            raise AttributeError("Model fitting failed to produce idata")

        omega, omega0, lam, n_chains, n_draws = self._extract_weight_posteriors()
        sc_all, gaps = self._compute_synthetic_and_gaps(omega, omega0, Y_co, y_tr)
        self.tau_posterior = self._compute_tau(gaps, lam, T_pre, n_chains, n_draws)
        self._build_reporting_objects(sc_all, T_pre, n_chains, n_draws)

    def _build_weight_fitter_inputs(
        self,
        Y_co: np.ndarray,
        y_tr: np.ndarray,
        T_pre: int,
    ) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray], dict[str, Any]]:
        """Construct the dict-based inputs consumed by the weight fitter.

        The weight fitter expects two modules: a *unit* module that regresses
        the pre-period treated outcome on the pre-period control panel, and a
        *time* module that regresses the post-period control mean on the
        pre-period control panel.

        Parameters
        ----------
        Y_co : np.ndarray
            Control outcomes with shape ``(N_co, T)``.
        y_tr : np.ndarray
            Mean treated outcomes with shape ``(T,)``.
        T_pre : int
            Number of pre-treatment time periods.

        Returns
        -------
        X : dict of str to xr.DataArray
            ``{"unit": X_unit, "time": X_time}`` design matrices.
        y : dict of str to xr.DataArray
            ``{"unit": y_unit, "time": y_time}`` response arrays.
        coords : dict
            Coordinates passed to PyMC during model construction.
        """
        # Module 1 (unit weights): X_unit = Y_co_pre.T (T_pre x N_co),
        #                          y_unit = y_tr_pre (T_pre,)
        X_unit = xr.DataArray(
            Y_co[:, :T_pre].T,
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
        Y_co_post_mean = Y_co[:, T_pre:].mean(axis=1)
        X_time = xr.DataArray(
            Y_co[:, :T_pre],
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
        coords = {
            "coeffs": self.control_units,
            "obs_ind": np.arange(T_pre),
            "coeffs_raw": self.control_units[1:],
            "obs_ind_raw": list(range(1, T_pre)),
        }
        return X, y, coords

    def _extract_weight_posteriors(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Pull posterior samples of the weight parameters from the model.

        Returns
        -------
        omega : np.ndarray
            Unit-weight posterior with shape ``(chain, draw, N_co)``.
        omega0 : np.ndarray
            Unit intercept posterior with shape ``(chain, draw)``.
        lam : np.ndarray
            Time-weight posterior with shape ``(chain, draw, T_pre)``.
        n_chains : int
            Number of MCMC chains.
        n_draws : int
            Number of draws per chain.
        """
        if self.model.idata is None:
            raise RuntimeError("Model has not been fit")
        posterior = self.model.idata.posterior
        omega = posterior["omega"].to_numpy()
        lam = posterior["lam"].to_numpy()
        omega0 = posterior["omega0"].to_numpy()
        n_chains, n_draws = omega.shape[0], omega.shape[1]
        return omega, omega0, lam, n_chains, n_draws

    @staticmethod
    def _compute_synthetic_and_gaps(
        omega: np.ndarray,
        omega0: np.ndarray,
        Y_co: np.ndarray,
        y_tr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the synthetic control trajectory and the treatment gap.

        For each posterior draw :math:`(c, d)` and time :math:`t` the
        synthetic control is
        :math:`\\mathrm{sc}_t = \\omega_0 + \\boldsymbol{\\omega}^\\top
        \\mathbf{Y}_{\\text{co}, t}`, and the gap is
        :math:`\\Delta_t = y_{\\text{tr}, t} - \\mathrm{sc}_t`.

        Parameters
        ----------
        omega : np.ndarray
            Unit-weight posterior with shape ``(chain, draw, N_co)``.
        omega0 : np.ndarray
            Unit intercept posterior with shape ``(chain, draw)``.
        Y_co : np.ndarray
            Control outcomes with shape ``(N_co, T)``.
        y_tr : np.ndarray
            Mean treated outcomes with shape ``(T,)``.

        Returns
        -------
        sc_all : np.ndarray
            Synthetic control with shape ``(chain, draw, T)``.
        gaps : np.ndarray
            Treated minus synthetic, shape ``(chain, draw, T)``.
        """
        sc_all = omega0[..., np.newaxis] + np.einsum("cdn,nt->cdt", omega, Y_co)
        gaps = y_tr[np.newaxis, np.newaxis, :] - sc_all
        return sc_all, gaps

    @staticmethod
    def _compute_tau(
        gaps: np.ndarray,
        lam: np.ndarray,
        T_pre: int,
        n_chains: int,
        n_draws: int,
    ) -> xr.DataArray:
        """Compute the ATT posterior via the SDiD double-difference formula.

        :math:`\\tau = \\bar{\\Delta}_{\\text{post}} -
        \\boldsymbol{\\lambda}^\\top \\boldsymbol{\\Delta}_{\\text{pre}}`.

        Parameters
        ----------
        gaps : np.ndarray
            Treated-minus-synthetic gaps with shape ``(chain, draw, T)``.
        lam : np.ndarray
            Time-weight posterior with shape ``(chain, draw, T_pre)``.
        T_pre : int
            Number of pre-treatment time periods.
        n_chains : int
            Number of MCMC chains.
        n_draws : int
            Number of draws per chain.

        Returns
        -------
        xr.DataArray
            Posterior samples of tau with dims ``(chain, draw)``.
        """
        gaps_post_mean = gaps[..., T_pre:].mean(axis=-1)
        lam_gaps_pre = (lam * gaps[..., :T_pre]).sum(axis=-1)
        tau = gaps_post_mean - lam_gaps_pre
        return xr.DataArray(
            tau,
            dims=["chain", "draw"],
            coords={
                "chain": np.arange(n_chains),
                "draw": np.arange(n_draws),
            },
        )

    def _build_reporting_objects(
        self,
        sc_all: np.ndarray,
        T_pre: int,
        n_chains: int,
        n_draws: int,
    ) -> None:
        """Build the xarray objects consumed by the reporting helpers.

        Sets the following attributes on ``self``:

        - ``pre_pred`` / ``post_pred``: ``az.InferenceData`` objects holding
          the synthetic-control predictions in a ``posterior_predictive``
          group.
        - ``pre_impact`` / ``post_impact``: ``xr.DataArray`` of observed
          minus counterfactual with dims ``(chain, draw, obs_ind,
          treated_units)``.
        - ``post_impact_cumulative``: cumulative sum of ``post_impact`` along
          the time axis.

        Parameters
        ----------
        sc_all : np.ndarray
            Synthetic control predictions for every time point, shape
            ``(chain, draw, T)``.
        T_pre : int
            Number of pre-treatment time periods.
        n_chains : int
            Number of MCMC chains.
        n_draws : int
            Number of draws per chain.
        """
        sc_pre = sc_all[..., :T_pre]
        sc_post = sc_all[..., T_pre:]

        self.pre_pred = self._build_inference_data(
            sc_pre, self.datapre.index, n_chains, n_draws
        )
        self.post_pred = self._build_inference_data(
            sc_post, self.datapost.index, n_chains, n_draws
        )

        y_tr_pre = self.datapre[self.treated_units].values.mean(axis=1)
        y_tr_post = self.datapost[self.treated_units].values.mean(axis=1)

        pre_impact_vals = y_tr_pre[np.newaxis, np.newaxis, :] - sc_pre
        post_impact_vals = y_tr_post[np.newaxis, np.newaxis, :] - sc_post

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

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2. Use
            ``None`` to return raw numbers.
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

    def plot(
        self,
        *,
        round_to: int | None = None,
        ci_prob: float = HDI_PROB,
        hdi_prob: float | None = None,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        figsize: tuple[float, float] = (7, 11),
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot SDiD results: counterfactual, period impact, and cumulative impact.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round the ATT in the title. Defaults to
            2. Use ``None`` for raw values.
        ci_prob : float
            Probability mass of the highest density interval drawn around the
            posterior predictive, causal impact, and cumulative impact bands.
            Must be in ``(0, 1]``. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
        hdi_prob : float, optional
            Deprecated. Use ``ci_prob`` instead.
        kind : {"ribbon", "spaghetti", "histogram"}, optional
            How posterior uncertainty is rendered. Defaults to ``"ribbon"``
            (mean + credible band).
        ci_kind : {"hdi", "eti"}, optional
            Credible interval type when ``kind="ribbon"``. Defaults to
            ``"hdi"``.
        num_samples : int, optional
            Number of posterior draws to overlay when ``kind="spaghetti"``.
            Defaults to 50.
        figsize : tuple of (float, float)
            Width and height of the figure in inches. Defaults to ``(7, 11)``
            so the three panels and date tick labels have room.
        show : bool, optional
            Whether to call :func:`matplotlib.pyplot.show` after drawing.
            Defaults to ``True``.
        legend_kwargs : dict, optional
            Keyword arguments applied to the top-axis legend in place after
            the figure is built. Supported keys include ``loc``,
            ``bbox_to_anchor``, ``fontsize``, ``frameon``, ``title``, and
            optionally ``bbox_transform`` alongside ``bbox_to_anchor``. See
            :meth:`~causalpy.experiments.base.BaseExperiment._render_plot`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the three stacked panels (plotnine base
            plus matplotlib overlays for the treatment line and date
            formatting).
        ax : numpy.ndarray
            Array of the three :class:`matplotlib.axes.Axes` instances.
        """
        if hdi_prob is not None:
            warnings.warn(
                "hdi_prob is deprecated and will be removed in a future release. "
                "Use ci_prob instead.",
                FutureWarning,
                stacklevel=2,
            )
            ci_prob = hdi_prob
        return self._render_plot(
            show=show,
            legend_kwargs=legend_kwargs,
            round_to=round_to,
            ci_prob=ci_prob,
            kind=kind,
            ci_kind=ci_kind,
            num_samples=num_samples,
            figsize=figsize,
        )

    def _prepare_bayesian_plot_data(
        self,
        *,
        treated_unit: str,
        panels: tuple[str, str, str],
        ci_prob: float,
        interval: Literal["hdi", "eti"],
        kind: Literal["ribbon", "histogram", "spaghetti"],
        num_samples: int,
    ) -> _SyntheticDiDPlotData:
        """Prepare tidy posterior and observed tables for plotting."""
        top, middle, bottom = panels
        pre_predictions = prediction_draws(
            self.pre_pred,
            pd.DataFrame({"obs_ind": self.datapre.index}),
            treated_unit=treated_unit,
        )
        post_predictions = prediction_draws(
            self.post_pred,
            pd.DataFrame({"obs_ind": self.datapost.index}),
            treated_unit=treated_unit,
        )
        pre_effect = dataarray_draws(self.pre_impact, treated_unit=treated_unit)
        post_effect = dataarray_draws(self.post_impact, treated_unit=treated_unit)
        cumulative_effect = dataarray_draws(
            self.post_impact_cumulative, treated_unit=treated_unit
        )
        panel_draws = pl.concat(
            [
                label_draws(
                    pre_predictions,
                    series="Pre-intervention fit",
                    panel=top,
                ),
                label_draws(
                    post_predictions,
                    series="Counterfactual",
                    panel=top,
                ),
                label_draws(pre_effect, series="pre", panel=middle),
                label_draws(post_effect, series="post", panel=middle),
                label_draws(cumulative_effect, series="post", panel=bottom),
            ],
            how="diagonal_relaxed",
        )
        grouping = ["panel", "series", "obs_ind"]
        intervals = summarize_draws(
            panel_draws,
            group_by=grouping,
            ci_prob=ci_prob,
            interval=interval,
        )
        observations = pd.DataFrame(
            {
                "obs_ind": self.data.index,
                "y": self.data[self.treated_units].mean(axis=1).to_numpy(),
                "series": "Observations",
                "panel": top,
            }
        )
        post_prediction = intervals.query(
            "panel == @top and series == 'Counterfactual'"
        )
        post_impact = intervals.query("panel == @middle and series == 'post'")
        post_observations = observations.loc[
            observations["obs_ind"].isin(self.datapost.index.tolist()),
            ["obs_ind", "y"],
        ]
        effect_area = pd.concat(
            [
                post_prediction[["obs_ind", "mu"]]
                .merge(post_observations, on="obs_ind")
                .rename(columns={"mu": "y1", "y": "y2"})
                .assign(panel=top),
                post_impact[["obs_ind", "mu"]]
                .rename(columns={"mu": "y1"})
                .assign(y2=0.0, panel=middle),
            ],
            ignore_index=True,
        )
        posterior_paths = (
            spaghetti_draws(
                panel_draws,
                group_by=grouping,
                num_samples=num_samples,
            )
            if kind == "spaghetti"
            else None
        )
        posterior_density = (
            pd.concat(
                [
                    posterior_histogram_tiles(
                        panel_draws.filter(pl.col("panel") == panel),
                        "obs_ind",
                        x_col="obs_ind",
                        panel=panel,
                    )
                    for panel in panels
                ],
                ignore_index=True,
            )
            if kind == "histogram"
            else None
        )
        return _SyntheticDiDPlotData(
            intervals=intervals,
            observations=observations,
            effect_area=effect_area,
            posterior_paths=posterior_paths,
            posterior_density=posterior_density,
        )

    def _bayesian_plot(
        self,
        round_to: int | None = None,
        ci_prob: float = HDI_PROB,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        figsize: tuple[float, float] = (7, 11),
        **kwargs: Any,
    ) -> PlotSpec:
        """Build the Bayesian synthetic DiD plot from tidy declarative layers."""
        treated_unit = self.treated_units[0]
        tau_mean = float(self.tau_posterior.mean())
        r_to = round_to if round_to is not None else 2
        panels = (
            f"SDiD: ATT = {round(tau_mean, r_to)}",
            "Causal Impact",
            "Cumulative Causal Impact",
        )
        plot_data = self._prepare_bayesian_plot_data(
            treated_unit=treated_unit,
            panels=panels,
            ci_prob=ci_prob,
            interval=interval_kind(ci_kind),
            kind=kind,
            num_samples=num_samples,
        )
        colors = {
            "Pre-intervention fit": "#1f77b4",
            "Counterfactual": "#ff7f0e",
            "Observations": "black",
            "pre": "#1f77b4",
            "post": "#ff7f0e",
        }
        p = build_causal_panel_plot(
            plot_data.intervals,
            plot_data.observations,
            panels=list(panels),
            colors=colors,
            show_panel_titles=True,
            kind=kind,
            shade_df=plot_data.effect_area,
            spaghetti_df=plot_data.posterior_paths,
            histogram_tiles=plot_data.posterior_density,
            figsize=figsize,
        )
        p += geom_vline(
            pd.DataFrame({"obs_ind": [self.treatment_time]}),
            aes(xintercept="obs_ind"),
            color="red",
            size=2,
        )
        if isinstance(self.data.index, pd.DatetimeIndex):
            p += theme(axis_text_x=element_text(rotation=45, ha="right"))

        def add_legend(_fig: plt.Figure, axes: list[plt.Axes]) -> None:
            add_causal_panel_legend(
                axes[0],
                labels=[
                    "Pre-intervention fit",
                    "Observations",
                    "Counterfactual",
                    "Causal impact",
                ],
                colors={**colors, "Causal impact": "#1f77b4"},
                area_labels={"Causal impact"},
            )

        return PlotSpec(p, overlay=add_legend, n_panels=3)

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
        **kwargs : dict
            Additional keyword arguments (currently unused).

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
