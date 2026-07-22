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
"""Synthetic Control Experiment."""

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.base import RegressorMixin

from causalpy.constants import HDI_PROB, LEGEND_FONT_SIZE
from causalpy.custom_exceptions import BadIndexException
from causalpy.date_utils import _combine_datetime_indices, format_date_axes
from causalpy.experiments.model_adapter import build_coords
from causalpy.plot_utils import (
    _PosteriorPlotStyle,
    extract_r2_score,
    get_hdi_to_df,
    has_posterior_draws,
    plot_posterior_over_x,
)
from causalpy.pymc_models import PyMCModel, WeightedSumFitter
from causalpy.reporting import EffectSummary
from causalpy.utils import check_convex_hull_violation, round_num

from .base import BaseExperiment


class SyntheticControl(BaseExperiment):
    """The class for the synthetic control experiment.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas dataframe.
    treatment_time : int, float, or pd.Timestamp
        The time when treatment occurred, in reference to the data index.
    control_units : list of str
        A list of control units to be used in the experiment.
    treated_units : list of str
        A list of treated units to be used in the experiment.
    model : PyMCModel, RegressorMixin, or None, default None
        A PyMC or sklearn model. Defaults to :class:`WeightedSumFitter`.
    min_donor_correlation : float, default 0.0
        Minimum acceptable Pearson correlation between each control unit and
        treated unit in the pre-treatment period. Control units below this
        threshold trigger a ``UserWarning``. Defaults to ``0.0`` (warn on
        negatively correlated donors).
    **kwargs
        Additional keyword arguments forwarded to :class:`BaseExperiment`.

    Notes
    -----
    **Estimate extraction**

    The model learns control-unit weights from pre-intervention outcomes and applies them to post-intervention controls to construct a synthetic untreated trajectory. Pointwise impact is the observed treated outcome minus this synthetic counterfactual, and cumulative impact is its running sum. Bayesian backends subtract the posterior conditional expectation ``mu`` rather than noisy posterior-predictive draws ``y_hat``; OLS subtracts its weighted point prediction.

    Examples
    --------
    >>> import causalpy as cp
    >>> df = cp.load_data("sc")
    >>> treatment_time = 70
    >>> seed = 42
    >>> result = cp.SyntheticControl(
    ...     df,
    ...     treatment_time,
    ...     control_units=["a", "b", "c", "d", "e", "f", "g"],
    ...     treated_units=["actual"],
    ...     model=cp.pymc_models.WeightedSumFitter(
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
    _default_model_class = WeightedSumFitter
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
        min_donor_correlation: float = 0.0,
        **kwargs: Any,
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
        if self._model_backend.is_ols and len(treated_units) > 1:
            raise ValueError(
                "OLS/sklearn synthetic control supports only a single treated "
                f"unit, but {len(treated_units)} were given: {treated_units}. "
                "Use a PyMC model (e.g. WeightedSumFitter) for multiple treated "
                "units, or run a separate experiment per treated unit."
            )
        if not (-1 <= min_donor_correlation <= 1):
            raise ValueError(
                f"min_donor_correlation must be between -1 and 1, "
                f"got {min_donor_correlation}."
            )
        self.min_donor_correlation = min_donor_correlation
        self.expt_type = "SyntheticControl"
        self._prepare_data()
        self._check_donor_correlations()
        self._check_convex_hull()
        self.algorithm()

    def _check_convex_hull(self) -> None:
        """Check convex hull assumption and warn if violated."""
        # Aggregate violations across all treated units
        total_violations = 0
        total_above = 0
        total_below = 0
        n_units = len(self.treated_units)
        n_pre_points = self.pre_design["treated"].shape[0]

        for i in range(n_units):
            unit_check = check_convex_hull_violation(
                self.pre_design["treated"].isel(treated_units=i),
                self.pre_design["control"],
            )
            total_violations += unit_check["n_violations"]
            total_above += unit_check["pct_above"] * n_pre_points / 100
            total_below += unit_check["pct_below"] * n_pre_points / 100

        total_points = n_units * n_pre_points
        hull_check = {
            "passes": total_violations == 0,
            "n_violations": total_violations,
            "pct_above": 100 * total_above / total_points if total_points > 0 else 0,
            "pct_below": 100 * total_below / total_points if total_points > 0 else 0,
        }

        if not hull_check["passes"]:
            warnings.warn(
                f"Convex hull assumption may be violated: {hull_check['n_violations']} "
                f"pre-intervention time points ({hull_check['pct_above']:.1f}% above, "
                f"{hull_check['pct_below']:.1f}% below control range). "
                "The synthetic control method requires the treated unit to lie within "
                "the convex hull of control units. Consider: (1) adding more diverse "
                "control units, (2) using a model with an intercept (e.g., ITS with "
                "control predictors), or (3) using the Augmented Synthetic Control Method. "
                "See glossary term 'Convex hull condition' for more details.",
                UserWarning,
                stacklevel=2,
            )

    def _check_donor_correlations(self) -> None:
        """Warn if any control unit has low pre-treatment correlation with treated units.

        Computes pairwise Pearson correlations between each control and treated
        unit in the pre-treatment period. Control units correlated below
        ``self.min_donor_correlation`` — or whose correlation is undefined
        (``NaN``, e.g. constant-valued donors) — are reported via
        :func:`warnings.warn`.
        """
        pre = self.datapre
        flagged: dict[str, list[tuple[str, float | None]]] = {}

        for treated in self.treated_units:
            treated_series = pre[treated]
            low: list[tuple[str, float | None]] = []
            for control in self.control_units:
                r = treated_series.corr(pre[control])
                if pd.isna(r):
                    low.append((control, None))
                elif r < self.min_donor_correlation:
                    low.append((control, float(r)))
            if low:
                flagged[treated] = low

        if flagged:
            parts: list[str] = []
            for treated, controls in flagged.items():
                details = []
                for name, corr_val in controls:
                    if corr_val is None:
                        details.append(f"'{name}' (r=undefined, likely constant)")
                    else:
                        details.append(f"'{name}' (r={corr_val:.3f})")
                parts.append(
                    f"Control units [{', '.join(details)}] have pre-treatment "
                    f"correlation below {self.min_donor_correlation} or undefined "
                    f"with treated unit '{treated}'."
                )
            msg = (
                " ".join(parts)
                + " Consider excluding them from the donor pool."
                + " Use cp.plot_correlations() to inspect."
                + " See Abadie (2021) for guidance on donor pool selection."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

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

    def _prepare_data(self) -> None:
        """Bundle control and treated data into ``xr.Dataset`` objects per period."""
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
        """Run the experiment algorithm: fit model, predict, and calculate causal impact."""
        # fit the model to the observed (pre-intervention) data
        self._model_backend.fit(
            X=self.pre_design["control"],
            y=self.pre_design["treated"],
            coords=build_coords(
                self.control_units,
                self.datapre.shape[0],
                treated_units=self.treated_units,
            ),
        )

        # score the goodness of fit to the pre-intervention data
        self.score = self._model_backend.score(
            X=self.pre_design["control"],
            y=self.pre_design["treated"],
        )

        # get the model predictions of the observed (pre-intervention) data
        self.pre_pred = self._model_backend.predict(X=self.pre_design["control"])

        # calculate the counterfactual
        self.post_pred = self._model_backend.predict(X=self.post_design["control"])
        # Impact below relies on exact obs_ind alignment; a mismatch (e.g. a bare
        # ndarray X getting arange coords) would silently corrupt the subtraction.
        assert self.pre_design["treated"].obs_ind.equals(self.pre_pred.obs_ind)
        assert self.post_design["treated"].obs_ind.equals(self.post_pred.obs_ind)
        self.pre_impact = (self.pre_design["treated"] - self.pre_pred).transpose(
            ..., "obs_ind", "treated_units"
        )
        self.post_impact = (self.post_design["treated"] - self.post_pred).transpose(
            ..., "obs_ind", "treated_units"
        )
        self.post_impact_cumulative = self.post_impact.cumsum(dim="obs_ind")

    def input_validation(
        self, data: pd.DataFrame, treatment_time: int | float | pd.Timestamp
    ) -> None:
        """Validate the input data and model formula for correctness.

        Parameters
        ----------
        data : pd.DataFrame
            The experiment data.
        treatment_time : int, float, or pd.Timestamp
            The treatment time, expected to be compatible with ``data.index``.
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

    def _pre_treatment_correlations(self) -> dict[str, float]:
        """Compute Pearson correlation between each treated unit and its
        synthetic control prediction in the pre-treatment period.

        Returns
        -------
        dict[str, float]
            Mapping from treated unit name to correlation coefficient.
        """
        correlations: dict[str, float] = {}
        for unit in self.treated_units:
            observed = (
                self.pre_design["treated"].sel(treated_units=unit).values.flatten()
            )
            predicted = (
                self.pre_pred.sel(treated_units=unit)
                .mean(dim=["chain", "draw"])
                .values.flatten()
            )
            correlations[unit] = float(np.corrcoef(observed, predicted)[0, 1])
        return correlations

    def summary(self, round_to: int | None = None) -> None:
        """Print summary of main results and model coefficients.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2. Use
            ``None`` to return raw numbers.
        """
        print(f"{self.expt_type:=^80}")
        print(f"Control units: {self.control_units}")
        if len(self.treated_units) > 1:
            print(f"Treated units: {self.treated_units}")
        else:
            print(f"Treated unit: {self.treated_units[0]}")
        self.print_coefficients(round_to)
        corrs = self._pre_treatment_correlations()
        for unit, r in corrs.items():
            print(f"Pre-treatment correlation ({unit}): {r:.4f}")

    @staticmethod
    def _convert_treatment_time_for_axis(
        axis: plt.Axes, treatment_time: int | float | pd.Timestamp
    ) -> int | float | pd.Timestamp:
        """
        Convert treatment time into the plotting units expected by a specific axis.
        """
        try:
            return axis.xaxis.convert_units(treatment_time)
        except (TypeError, ValueError):
            return treatment_time

    def plot(
        self,
        *,
        round_to: int | None = None,
        treated_unit: str | None = None,
        ci_prob: float = HDI_PROB,
        hdi_prob: float | None = None,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        plot_predictors: bool = False,
        figsize: tuple[float, float] = (7, 8),
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot the synthetic control results for a specific treated unit.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round numerical results in the figure
            title (e.g. the Bayesian :math:`R^2`). Defaults to ``None``,
            in which case 2 significant figures are used.
        treated_unit : str, optional
            Which treated unit to plot. Must be one of the names supplied
            via ``treated_units`` at construction time. Defaults to ``None``,
            which selects the first treated unit.
        ci_prob : float
            Probability mass of the highest density interval drawn around the
            posterior predictive, causal impact, and cumulative impact bands.
            Must be in ``(0, 1]``. Ignored for OLS models. Defaults to
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

        plot_predictors : bool
            Whether to overlay the donor (control) unit trajectories on the
            top panel. Defaults to ``False``.
        figsize : tuple of (float, float)
            Width and height of the figure in inches, passed to
            :func:`matplotlib.pyplot.subplots`. Defaults to ``(7, 8)``.
        show : bool
            Whether to automatically display the plot. Defaults to ``True``.
            Set to ``False`` if you want to modify the figure before
            displaying it.
        legend_kwargs : dict, optional
            Keyword arguments to adjust legend placement and styling.
            Supported keys: ``loc``, ``bbox_to_anchor``, ``fontsize``,
            ``frameon``, ``title`` (``bbox_transform`` is accepted alongside
            ``bbox_to_anchor``). The existing legend is modified **in
            place** so that custom handles are preserved.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure that was created.
        ax : list[matplotlib.axes.Axes]
            The three axes (top: predictions, middle: causal impact,
            bottom: cumulative impact).
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
            treated_unit=treated_unit,
            ci_prob=ci_prob,
            kind=kind,
            ci_kind=ci_kind,
            num_samples=num_samples,
            plot_predictors=plot_predictors,
            figsize=figsize,
        )

    def _plot(
        self,
        round_to: int | None = None,
        treated_unit: str | None = None,
        ci_prob: float = HDI_PROB,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        plot_predictors: bool = False,
        figsize: tuple[float, float] = (7, 8),
        **kwargs: Any,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plot the results for a specific treated unit.

        Consumes the canonical prediction container from any backend.
        Uncertainty bands are drawn only when the container carries posterior
        draws; point-estimate backends (singleton ``chain``/``draw``) get bare
        lines.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2. Use ``None``
            to return raw numbers.
        treated_unit : str, optional
            Which treated unit to plot. Must be a string name of the treated unit.
            If ``None``, plots the first treated unit.
        ci_prob : float, optional
            Probability mass of the credible interval drawn around the
            posterior predictive, causal impact, and cumulative impact bands.
            Must be in ``(0, 1]``. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
        plot_predictors : bool, optional
            Whether to overlay control-unit trajectories. Defaults to ``False``.
        figsize : tuple of (float, float), optional
            Width and height of the figure in inches. Defaults to ``(7, 8)``.
        """
        counterfactual_label = "Counterfactual"
        with_uncertainty = has_posterior_draws(self.pre_pred)
        style: _PosteriorPlotStyle = {
            "ci_prob": ci_prob,
            "kind": kind,
            "ci_kind": ci_kind,
            "num_samples": num_samples,
        }

        # Get treated unit name - default to first unit if None
        treated_unit = (
            treated_unit if treated_unit is not None else self.treated_units[0]
        )

        if treated_unit not in self.treated_units:
            raise ValueError(
                f"treated_unit '{treated_unit}' not found. Available units: {self.treated_units}"
            )

        pre_pred = self.pre_pred.sel(treated_units=treated_unit)
        post_pred = self.post_pred.sel(treated_units=treated_unit)
        pre_impact = self.pre_impact.sel(treated_units=treated_unit)
        post_impact = self.post_impact.sel(treated_units=treated_unit)
        post_impact_cumulative = self.post_impact_cumulative.sel(
            treated_units=treated_unit
        )
        pre_treated = self.pre_design["treated"].sel(treated_units=treated_unit)
        post_treated = self.post_design["treated"].sel(treated_units=treated_unit)

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize)
        # TOP PLOT --------------------------------------------------
        handles: list[Any] = []
        labels: list[str] = []
        if with_uncertainty:
            # pre-intervention period
            h_line, h_patch = plot_posterior_over_x(
                self.datapre.index,
                pre_pred,
                ax=ax[0],
                **style,
                plot_hdi_kwargs={"color": "C0"},
            )
            handles.append((h_line, h_patch))
            labels.append("Pre-intervention period")

            # Plot observations for primary treated unit
            (h,) = ax[0].plot(
                self.datapre.index,
                pre_treated,
                "k.",
                label="Observations",
            )
            handles.append(h)
            labels.append("Observations")

            # post intervention period
            h_line, h_patch = plot_posterior_over_x(
                self.datapost.index,
                post_pred,
                ax=ax[0],
                **style,
                plot_hdi_kwargs={"color": "C1"},
            )
            handles.append((h_line, h_patch))
            labels.append(counterfactual_label)

            ax[0].plot(self.datapost.index, post_treated, "k.")
        else:
            ax[0].plot(pre_treated["obs_ind"], pre_treated, "k.")
            ax[0].plot(post_treated["obs_ind"], post_treated, "k.")
            ax[0].plot(
                self.datapre.index,
                pre_pred.mean(dim=["chain", "draw"]),
                c="k",
                label="model fit",
            )
            ax[0].plot(
                self.datapost.index,
                post_pred.mean(dim=["chain", "draw"]),
                label=counterfactual_label,
                ls=":",
                c="k",
            )

        # Shaded causal effect
        h = ax[0].fill_between(
            self.datapost.index,
            y1=post_pred.mean(dim=["chain", "draw"]).values,
            y2=post_treated.values,
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        if with_uncertainty:
            handles.append(h)
            labels.append("Causal impact")

        ax[0].set(title=f"{self._get_score_title(treated_unit, round_to)}")

        # MIDDLE PLOT -----------------------------------------------
        if with_uncertainty:
            plot_posterior_over_x(
                self.datapre.index,
                pre_impact,
                ax=ax[1],
                **style,
                plot_hdi_kwargs={"color": "C0"},
            )
            plot_posterior_over_x(
                self.datapost.index,
                post_impact,
                ax=ax[1],
                **style,
                plot_hdi_kwargs={"color": "C1"},
            )
        else:
            ax[1].plot(self.datapre.index, pre_impact.mean(dim=["chain", "draw"]), "k.")
            ax[1].plot(
                self.datapost.index,
                post_impact.mean(dim=["chain", "draw"]),
                "k.",
                label=counterfactual_label,
            )
        ax[1].axhline(y=0, c="k")
        ax[1].fill_between(
            self.datapost.index,
            y1=post_impact.mean(dim=["chain", "draw"]),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[1].set(title="Causal Impact")

        # BOTTOM PLOT -----------------------------------------------
        if with_uncertainty:
            plot_posterior_over_x(
                self.datapost.index,
                post_impact_cumulative,
                ax=ax[2],
                **style,
                plot_hdi_kwargs={"color": "C1"},
            )
        else:
            ax[2].plot(
                self.datapost.index,
                post_impact_cumulative.mean(dim=["chain", "draw"]),
                c="k",
            )
        ax[2].axhline(y=0, c="k")
        ax[2].set(title="Cumulative Causal Impact")

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
                label=None if with_uncertainty else "Treatment time",
            )

        if with_uncertainty:
            ax[0].legend(
                handles=(h_tuple for h_tuple in handles),
                labels=labels,
                fontsize=LEGEND_FONT_SIZE,
            )
        else:
            # Collect labelled artists (including the treatment line)
            ax[0].legend(fontsize=LEGEND_FONT_SIZE)

        if plot_predictors:
            # plot control units as well
            ax[0].plot(
                self.datapre.index,
                self.pre_design["control"],
                "-",
                c=[0.8, 0.8, 0.8],
                zorder=1,
            )
            ax[0].plot(
                self.datapost.index,
                self.post_design["control"],
                "-",
                c=[0.8, 0.8, 0.8],
                zorder=1,
            )

        # Apply intelligent date formatting if data has datetime index
        if isinstance(self.datapre.index, pd.DatetimeIndex):
            # Combine pre and post indices for full date range
            full_index = _combine_datetime_indices(
                pd.DatetimeIndex(self.datapre.index),
                pd.DatetimeIndex(self.datapost.index),
            )
            format_date_axes(ax, full_index)

        return fig, ax

    def get_plot_data(
        self, hdi_prob: float = HDI_PROB, treated_unit: str | None = None
    ) -> pd.DataFrame:
        """
        Recover the data of the experiment along with the prediction and causal impact information.

        HDI columns are included only when the prediction container carries
        posterior draws (point-estimate backends return just ``prediction``
        and ``impact``).

        Parameters
        ----------
        hdi_prob : float, default :data:`~causalpy.constants.HDI_PROB`
            Probability mass of the highest density interval. Defaults to
            the project-wide :data:`~causalpy.constants.HDI_PROB`. Ignored
            when the prediction container has no posterior draws.
        treated_unit : str, optional
            Which treated unit to extract data for. Must be a string name
            of the treated unit. If ``None``, uses the first treated unit.
        """
        with_uncertainty = has_posterior_draws(self.pre_pred)
        hdi_pct = int(round(hdi_prob * 100))

        pre_data = self.datapre.copy()
        post_data = self.datapost.copy()

        # Get treated unit name - default to first unit if None
        treated_unit = (
            treated_unit if treated_unit is not None else self.treated_units[0]
        )

        if treated_unit not in self.treated_units:
            raise ValueError(
                f"treated_unit '{treated_unit}' not found. Available units: {self.treated_units}"
            )

        pre_pred = self.pre_pred.sel(treated_units=treated_unit)
        post_pred = self.post_pred.sel(treated_units=treated_unit)
        pre_impact = self.pre_impact.sel(treated_units=treated_unit)
        post_impact = self.post_impact.sel(treated_units=treated_unit)

        pre_data["prediction"] = pre_pred.mean(dim=["chain", "draw"]).values
        post_data["prediction"] = post_pred.mean(dim=["chain", "draw"]).values

        if with_uncertainty:
            pred_lower_col = f"pred_hdi_lower_{hdi_pct}"
            pred_upper_col = f"pred_hdi_upper_{hdi_pct}"
            pre_hdi = get_hdi_to_df(pre_pred, hdi_prob=hdi_prob)
            post_hdi = get_hdi_to_df(post_pred, hdi_prob=hdi_prob)
            # Extract only the lower and upper columns
            pre_data[[pred_lower_col, pred_upper_col]] = pre_hdi.iloc[:, [0, -1]].values
            post_data[[pred_lower_col, pred_upper_col]] = post_hdi.iloc[
                :, [0, -1]
            ].values

        pre_data["impact"] = pre_impact.mean(dim=["chain", "draw"]).values
        post_data["impact"] = post_impact.mean(dim=["chain", "draw"]).values

        if with_uncertainty:
            impact_lower_col = f"impact_hdi_lower_{hdi_pct}"
            impact_upper_col = f"impact_hdi_upper_{hdi_pct}"
            pre_impact_hdi = get_hdi_to_df(pre_impact, hdi_prob=hdi_prob)
            post_impact_hdi = get_hdi_to_df(post_impact, hdi_prob=hdi_prob)
            pre_data[[impact_lower_col, impact_upper_col]] = pre_impact_hdi.iloc[
                :, [0, -1]
            ].values
            post_data[[impact_lower_col, impact_upper_col]] = post_impact_hdi.iloc[
                :, [0, -1]
            ].values

        self.plot_data = pd.concat([pre_data, post_data])

        return self.plot_data

    def _get_score_title(self, treated_unit: str, round_to: int | None = 2) -> str:
        """Generate appropriate score title for the specified treated unit"""
        r_to = round_to if round_to is not None else 2
        r2_val, r2_std_val = extract_r2_score(
            self.score, unit_index=self.treated_units.index(treated_unit)
        )
        assert r2_val is not None  # both backends' score containers carry R^2
        if r2_std_val is not None:
            return (
                f"Pre-intervention Bayesian $R^2$: {round_num(r2_val, r_to)} "
                f"(std = {round_num(r2_std_val, r_to)})"
            )
        return f"$R^2$ on pre-intervention data = {round_num(r2_val, r_to)}"

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
        """
        Generate a decision-ready summary of causal effects for Synthetic Control.

        Parameters
        ----------
        window : str, tuple, or slice, default="post"
            Time window for analysis:

            - "post": All post-treatment time points (default)
            - (start, end): Tuple of start and end times (handles both datetime and integer indices)
            - slice: Python slice object for integer indices
        direction : {"increase", "decrease", "two-sided"}, default="increase"
            Direction for tail probability calculation (PyMC only, ignored for OLS).
        alpha : float, default=0.05
            Significance level for HDI/CI intervals (1-alpha confidence level).
        cumulative : bool, default=True
            Whether to include cumulative effect statistics.
        relative : bool, default=True
            Whether to include relative effect statistics (% change vs counterfactual).
        min_effect : float, optional
            Region of Practical Equivalence (ROPE) threshold (PyMC only, ignored for OLS).
        treated_unit : str, optional
            For multi-unit experiments, specify which treated unit to analyze.
            If None and multiple units exist, uses first unit.
        period : {"intervention", "post", "comparison"}, optional
            Ignored for Synthetic Control (two-period design only).
        prefix : str, optional
            Prefix for prose generation. Defaults to "Post-period".
        **kwargs
            Reserved for forward-compatibility; not consumed by this
            implementation.

        Returns
        -------
        EffectSummary
            Object with .table (DataFrame) and .text (str) attributes.
            The .text attribute contains a detailed multi-paragraph narrative report.
        """
        from causalpy.reporting import (
            _effect_summary_timeseries,
            _extract_counterfactual,
            _extract_window,
        )

        # Warn if period parameter is provided (not supported for Synthetic Control)
        if period is not None:
            warnings.warn(
                f"period='{period}' is ignored for SyntheticControl (two-period design only). "
                "Results reflect the entire post-treatment period. "
                "Use the 'window' parameter to analyze specific time ranges.",
                UserWarning,
                stacklevel=2,
            )

        windowed_impact, window_coords = _extract_window(
            self, window, treated_unit=treated_unit
        )
        counterfactual = _extract_counterfactual(
            self, window_coords, treated_unit=treated_unit
        )
        return _effect_summary_timeseries(
            self,
            windowed_impact,
            counterfactual,
            window_coords,
            direction=direction,
            alpha=alpha,
            cumulative=cumulative,
            relative=relative,
            min_effect=min_effect,
            prefix=prefix,
            experiment_type="sc",
        )
