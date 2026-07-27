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
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.base import RegressorMixin

from causalpy.constants import HDI_PROB
from causalpy.custom_exceptions import BadIndexException
from causalpy.date_utils import _combine_datetime_indices, format_date_axes
from causalpy.plot_utils import _PosteriorPlotStyle, plot_posterior_over_x
from causalpy.pymc_models import PyMCModel, SyntheticDifferenceInDifferencesWeightFitter
from causalpy.reporting import EffectSummary

from .base import BaseExperiment


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
    **Estimate extraction**

    The Bayesian weight model produces posterior draws of synthetic-control unit weights and pre-period time weights. For each draw, the class constructs treated-minus-synthetic gaps and evaluates the weighted double-difference analytically to obtain the scalar ``tau_posterior`` ATT; the effect is not read from a regression coefficient or obtained by population-standardized g-computation. The time-indexed ``post_impact`` consumed by ``effect_summary()`` is the post-period treated-minus-synthetic trajectory rather than this time-weighted scalar.

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
        # Backend-identity check is justified here: capability validation
        # (trust boundary), not statistical dispatch.
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
        posterior = self._model_backend.require_idata().posterior
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

        - ``pre_pred`` / ``post_pred``: ``xr.DataArray`` synthetic-control
          predictions with canonical dims ``(chain, draw, obs_ind,
          treated_units)``.
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

        self.pre_pred = self._build_prediction(
            sc_pre, self.datapre.index, n_chains, n_draws
        )
        self.post_pred = self._build_prediction(
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

    def _build_prediction(
        self,
        mu_vals: np.ndarray,
        index: pd.Index,
        n_chains: int,
        n_draws: int,
    ) -> xr.DataArray:
        """Build a prediction DataArray with canonical dimensions.

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
        xr.DataArray
            Predictions with dims ``(chain, draw, obs_ind, treated_units)``.
        """
        return xr.DataArray(
            mu_vals[..., np.newaxis],
            dims=["chain", "draw", "obs_ind", "treated_units"],
            coords={
                "chain": np.arange(n_chains),
                "draw": np.arange(n_draws),
                "obs_ind": index,
                "treated_units": [self.treated_units[0]],
            },
        )

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
        kind : {"ribbon", "histogram", "spaghetti"}, optional
            How posterior uncertainty is rendered via
            :func:`~causalpy.plot_utils.plot_posterior_over_x`. Defaults to ``"ribbon"``.
            For ``"spaghetti"``, legends use draw lines rather than a shaded
            band. For ``"histogram"``, uncertainty is shown as a 2D density
            heatmap with a mean line overlay (no ribbon patch for legends).
        ci_kind : {"hdi", "eti"}, optional
            Credible interval type when ``kind="ribbon"``. Defaults to
            ``"hdi"``.
        num_samples : int, optional
            Number of posterior draws when ``kind="spaghetti"``. Defaults
            to 50. Ignored for other kinds.

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
            The figure containing the three stacked panels.
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

    def _plot(
        self,
        round_to: int | None = None,
        ci_prob: float = HDI_PROB,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        **kwargs: Any,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot the results: counterfactual, impact, and cumulative impact.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2. Use
            ``None`` to return raw numbers.
        hdi_prob : float, optional
            Probability mass of the credible interval. Must be in ``(0, 1]``.
            Defaults to :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
        kind : {"ribbon", "histogram", "spaghetti"}, optional
            How posterior uncertainty is rendered. Defaults to ``"ribbon"``.
        ci_kind : {"hdi", "eti"}, optional
            Credible interval type when ``kind="ribbon"``. Defaults to ``"hdi"``.
        num_samples : int, optional
            Number of posterior draws when ``kind="spaghetti"``. Defaults to 50.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure containing the plots.
        ax : list of matplotlib.axes.Axes
            The three axes (counterfactual, impact, cumulative impact).
        """
        style: _PosteriorPlotStyle = {
            "ci_prob": ci_prob,
            "kind": kind,
            "ci_kind": ci_kind,
            "num_samples": num_samples,
        }
        treated_unit = self.treated_units[0]

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

        # ---- TOP PLOT: Observed vs counterfactual ----
        pre_pred = self.pre_pred.sel(treated_units=treated_unit)
        post_pred = self.post_pred.sel(treated_units=treated_unit)

        # Pre-intervention synthetic control fit
        h_line, h_patch = plot_posterior_over_x(
            self.datapre.index,
            pre_pred,
            ax=ax[0],
            **style,
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
        h_line, h_patch = plot_posterior_over_x(
            self.datapost.index,
            post_pred,
            ax=ax[0],
            **style,
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
        plot_posterior_over_x(
            self.datapre.index,
            self.pre_impact.sel(treated_units=treated_unit),
            ax=ax[1],
            **style,
            plot_hdi_kwargs={"color": "C0"},
        )
        plot_posterior_over_x(
            self.datapost.index,
            self.post_impact.sel(treated_units=treated_unit),
            ax=ax[1],
            **style,
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
        plot_posterior_over_x(
            self.datapost.index,
            self.post_impact_cumulative.sel(treated_units=treated_unit),
            ax=ax[2],
            **style,
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
        )

        # Apply intelligent date formatting if data has datetime index
        if isinstance(self.datapre.index, pd.DatetimeIndex):
            full_index = _combine_datetime_indices(
                pd.DatetimeIndex(self.datapre.index),
                pd.DatetimeIndex(self.datapost.index),
            )
            format_date_axes(ax, full_index)

        return fig, ax

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
