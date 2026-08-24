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
from causalpy.date_utils import (
    _combine_datetime_indices,
    format_date_axes,
    validate_treatment_time_against_index,
)
from causalpy.experiments._results import CausalResult
from causalpy.experiments.model_adapter import PyMCModelAdapter, build_coords
from causalpy.input_data import DataFrameLike, to_pandas_with_time_index
from causalpy.plot_utils import (
    _PosteriorPlotStyle,
    format_r2_score,
    get_hdi_to_df,
    has_posterior_draws,
    plot_posterior_over_x,
)
from causalpy.pymc_models import (
    _LEGACY_Y_HAT_PRIOR,
    PyMCModel,
    WeightedSumFitter,
    _uses_stock_y_hat_default,
)
from causalpy.reporting import EffectSummary
from causalpy.utils import check_convex_hull_violation

from .base import BaseExperiment


class SyntheticControl(BaseExperiment[CausalResult]):
    """The class for the synthetic control experiment.

    Parameters
    ----------
    data : dataframe-like
        Any eager dataframe Narwhals supports. For a pandas dataframe the index
        carries the time axis. Dataframes from other libraries have no index,
        so those callers must pass ``time_column``.
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
    auto_scale_sigma : bool, default True
        If ``True`` (default) and the model still carries the weighted-sum
        fitters' stock ``y_hat`` prior, that ``sigma ~ HalfNormal(1)`` default is
        replaced by ``sigma ~ Exponential(2/s)``. The scale is computed per
        treated unit, with *s* the standard deviation of that unit's
        pre-treatment data, so units on different scales are each calibrated
        separately. Set to ``False`` to keep the original ``HalfNormal(1)``
        default; the experiment then fits a copy of the model with that prior
        pinned explicitly, leaving the instance you passed in untouched. A model
        constructed with an explicit ``y_hat`` prior is never rescaled either
        way.
    time_column : str, optional
        Column holding the time axis. It becomes the index of the data. Required
        for non-pandas inputs, which carry no index. If None (default), the
        pandas index of ``data`` is used. Passing it for data that already has a
        meaningful index raises, since only one of the two can be the time axis.

    Notes
    -----
    **Lazy lifecycle**

    Construction only validates input and builds the control/treated design
    matrices — nothing is sampled. Call :meth:`fit` to run posterior inference
    (it returns ``self``, so construction and fitting chain in one
    expression), and optionally :meth:`sample_prior_predictive` first for
    prior predictive checks (``plot(group="prior")``,
    ``effect_summary(group="prior")``). Results live on ``exp.result`` /
    ``exp.prior_result``.

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
    ... ).fit()
    """

    supports_ols = True
    supports_bayes = True
    _default_model_class = WeightedSumFitter

    def __init__(
        self,
        data: DataFrameLike,
        treatment_time: int | float | pd.Timestamp,
        control_units: list[str],
        treated_units: list[str],
        model: PyMCModel | RegressorMixin | None = None,
        min_donor_correlation: float = 0.0,
        auto_scale_sigma: bool = True,
        time_column: str | None = None,
    ) -> None:
        super().__init__(model=model)
        # to_pandas_with_time_index returns a copy, so index metadata is
        # normalized on an owned frame rather than the caller's.
        data = to_pandas_with_time_index(data, time_column)
        data.index.name = "obs_ind"
        self.data = data
        self.input_validation(data, treatment_time)
        self.treatment_time = treatment_time
        self.control_units = control_units
        self.labels = control_units
        self.treated_units = treated_units
        self.auto_scale_sigma = auto_scale_sigma
        if not auto_scale_sigma:
            self._pin_legacy_sigma_prior()
        # Backend-identity check is justified here: constructor-time
        # capability validation (trust boundary), not statistical dispatch.
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

    def _pin_legacy_sigma_prior(self) -> None:
        """Swap in a model that carries the legacy noise prior explicitly.

        Automatic scaling only reaches models that still declare the stock
        ``y_hat`` default and were not given an explicit ``y_hat`` prior, so
        those are the only models the opt-out has to touch. Expressing the
        opt-out as an ordinary user prior on a fresh instance — rather than as a
        fit-local flag — means it survives refits and later clones, such as the
        ones the sensitivity checks make, and leaves the caller's own model
        untouched.
        """
        model = self.model
        if not isinstance(model, PyMCModel) or not _uses_stock_y_hat_default(model):
            return
        user_priors = model._user_priors or {}
        if "y_hat" in user_priors:
            return
        # Route the opt-out through ``_clone`` rather than ``type(model)(...)``:
        # subclasses with extra ``__init__`` parameters carry them through their
        # ``_clone`` override, so a direct reconstruction here would silently
        # drop that configuration. ``_clone`` takes the pinned prior set as an
        # override, keeping the sole re-instantiation site inside ``_clone``.
        pinned = model._clone(priors={**user_priors, "y_hat": _LEGACY_Y_HAT_PRIOR})
        self.model = pinned
        self._model_backend = PyMCModelAdapter(pinned)

    def _fit_inputs(
        self,
    ) -> tuple[xr.DataArray, xr.DataArray, dict[str, Any]]:
        """Return the pre-period control/treated matrices and coordinates for build."""
        control_pre = self.pre_design["control"]
        return (
            control_pre,
            self.pre_design["treated"],
            build_coords(
                self.control_units,
                self.datapre.shape[0],
                treated_units=self.treated_units,
            ),
        )

    def _finalize(self, group: Literal["prior", "posterior"]) -> None:
        """Compute the group's result bundle from its draws and assign it.

        The body is the historical ``algorithm()`` with the draw group
        threaded through prediction and scoring. Posterior fits score against
        the observed pre-period; prior draws are not scored (R² against
        observed data is not informative under a prior).
        """
        control_pre = self.pre_design["control"]
        treated_pre = self.pre_design["treated"]
        treated_post = self.post_design["treated"]

        # get the model predictions of the observed (pre-intervention) data
        predictions_pre = self._model_backend.predict(X=control_pre, group=group)

        # calculate the counterfactual
        predictions_post = self._model_backend.predict(
            X=self.post_design["control"], group=group
        )
        # Impact below relies on exact obs_ind alignment; a mismatch (e.g. a bare
        # ndarray X getting arange coords) would silently corrupt the subtraction.
        assert treated_pre.obs_ind.equals(predictions_pre.obs_ind)
        assert treated_post.obs_ind.equals(predictions_post.obs_ind)
        impact_pre = (treated_pre - predictions_pre).transpose(
            ..., "obs_ind", "treated_units"
        )
        impact_post = (treated_post - predictions_post).transpose(
            ..., "obs_ind", "treated_units"
        )
        impact_post_cumulative = impact_post.cumsum(dim="obs_ind")

        score = None
        if group == "posterior":
            # score the goodness of fit to the pre-intervention data
            score = self._model_backend.score(X=control_pre, y=treated_pre)

        bundle = CausalResult(
            predictions_pre=predictions_pre,
            predictions_post=predictions_post,
            impact_pre=impact_pre,
            impact_post=impact_post,
            impact_post_cumulative=impact_post_cumulative,
            score=score,
        )
        self._assign_bundle(group, bundle)

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
        validate_treatment_time_against_index(data.index, treatment_time)

    def _pre_treatment_correlations(self) -> dict[str, float]:
        """Compute Pearson correlation between each treated unit and its
        synthetic control prediction in the pre-treatment period.

        Posterior-only: reads the fitted posterior bundle.

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
                self.result.predictions_pre.sel(treated_units=unit)
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
        group: Literal["prior", "posterior"] = "posterior",
        round_to: int | None = None,
        treated_unit: str | None = None,
        ci_prob: float = HDI_PROB,
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
        group : {"prior", "posterior"}, default "posterior"
            Which draw group to plot. ``"prior"`` renders the reduced
            prior-check panel set — the prior-implied counterfactual against
            the observed series only — and requires
            :meth:`sample_prior_predictive`; ``"posterior"`` (default)
            renders the full three-panel layout and requires :meth:`fit`.
            The two groups intentionally return different axes layouts.
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
        return self._render_plot(
            show=show,
            legend_kwargs=legend_kwargs,
            group=group,
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
        *,
        group: Literal["prior", "posterior"] = "posterior",
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
        Plot the posterior or prior-check figure for a specific treated unit.

        Consumes the resolved group bundle injected by
        :meth:`~causalpy.experiments.base.BaseExperiment._render_plot`.
        Uncertainty bands are drawn only when the container carries draws;
        point-estimate backends (singleton ``chain``/``draw``) get bare lines.

        Parameters
        ----------
        group : {"prior", "posterior"}
            ``"prior"`` renders the reduced single-panel prior-check figure
            via :meth:`_plot_prior_checks`; ``"posterior"`` renders the full
            three-panel layout.
        round_to : int, optional
            Number of decimals used to round results. Defaults to ``None``,
            in which case 2 significant figures are used.
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
        bundle = self._require_bundle(group)
        if group == "prior":
            return self._plot_prior_checks(bundle=bundle)

        counterfactual_label = "Counterfactual"
        with_uncertainty = has_posterior_draws(bundle.predictions_pre)
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

        pre_pred = bundle.predictions_pre.sel(treated_units=treated_unit)
        post_pred = bundle.predictions_post.sel(treated_units=treated_unit)
        pre_impact = bundle.impact_pre.sel(treated_units=treated_unit)
        post_impact = bundle.impact_post.sel(treated_units=treated_unit)
        post_impact_cumulative = bundle.impact_post_cumulative.sel(
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

        ax[0].set(
            title=f"{self._get_score_title(bundle.score, treated_unit, round_to)}"
        )

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

    def _plot_prior_checks(
        self, *, bundle: CausalResult
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Render the reduced prior-check panel set.

        Prior-implied bands are typically far wider than the data, so the
        impact panels are dropped rather than autoscaled into uselessness.
        The question a prior check answers is whether the prior counterfactual
        is plausible against the observed series — one panel suffices.
        """
        pre_pred = bundle.predictions_pre.isel(treated_units=0)
        post_pred = bundle.predictions_post.isel(treated_units=0)
        pre_treated = self.pre_design["treated"].isel(treated_units=0)
        post_treated = self.post_design["treated"].isel(treated_units=0)

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        style: _PosteriorPlotStyle = {
            "ci_prob": HDI_PROB,
            "kind": "ribbon",
            "ci_kind": "hdi",
            "num_samples": 50,
        }
        h_line, h_patch = plot_posterior_over_x(
            self.datapre.index,
            pre_pred,
            ax=ax,
            **style,
            plot_hdi_kwargs={"color": "C0"},
        )
        ax.plot(self.datapre.index, pre_treated, "k.", label="Observations")
        plot_posterior_over_x(
            self.datapost.index,
            post_pred,
            ax=ax,
            **style,
            plot_hdi_kwargs={"color": "C1"},
        )
        ax.plot(self.datapost.index, post_treated, "k.", zorder=3)
        treatment_time = self._convert_treatment_time_for_axis(ax, self.treatment_time)
        ax.axvline(x=treatment_time, ls="-", lw=3, color="r", zorder=1.5)
        ax.legend(
            handles=[(h_line, h_patch)],
            labels=["Prior counterfactual"],
            fontsize=LEGEND_FONT_SIZE,
        )
        ax.set(title="Prior predictive check")

        if isinstance(self.datapre.index, pd.DatetimeIndex):
            full_index = _combine_datetime_indices(
                pd.DatetimeIndex(self.datapre.index),
                pd.DatetimeIndex(self.datapost.index),
            )
            format_date_axes([ax], full_index)

        return fig, [ax]

    def get_plot_data(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        hdi_prob: float = HDI_PROB,
        treated_unit: str | None = None,
    ) -> pd.DataFrame:
        """
        Recover the data of the experiment along with the prediction and causal impact information.

        HDI columns are included only when the prediction container carries
        posterior draws (point-estimate backends return just ``prediction``
        and ``impact``).

        Parameters
        ----------
        group : {"prior", "posterior"}, default "posterior"
            Which draw group to summarize. ``"prior"`` requires
            :meth:`sample_prior_predictive`; ``"posterior"`` requires
            :meth:`fit`.
        hdi_prob : float, default :data:`~causalpy.constants.HDI_PROB`
            Probability mass of the highest density interval. Defaults to
            the project-wide :data:`~causalpy.constants.HDI_PROB`. Ignored
            when the prediction container has no posterior draws.
        treated_unit : str, optional
            Which treated unit to extract data for. Must be a string name
            of the treated unit. If ``None``, uses the first treated unit.

        Returns
        -------
        pd.DataFrame
            Observed data with ``prediction`` and ``impact`` columns plus HDI
            bounds when draws are available. Not cached on the experiment.
        """
        bundle = self._require_bundle(group)
        with_uncertainty = has_posterior_draws(bundle.predictions_pre)
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

        pre_pred = bundle.predictions_pre.sel(treated_units=treated_unit)
        post_pred = bundle.predictions_post.sel(treated_units=treated_unit)
        pre_impact = bundle.impact_pre.sel(treated_units=treated_unit)
        post_impact = bundle.impact_post.sel(treated_units=treated_unit)

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

        return pd.concat([pre_data, post_data])

    def _get_score_title(
        self, score: pd.Series | None, treated_unit: str, round_to: int | None = 2
    ) -> str:
        """Generate appropriate score title for the specified treated unit"""
        return format_r2_score(
            score,
            unit_index=self.treated_units.index(treated_unit),
            round_to=round_to,
            context="on pre-intervention data",
        )

    def effect_summary(
        self,
        *,
        group: Literal["prior", "posterior"] = "posterior",
        window: Literal["post"] | tuple | slice = "post",
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        cumulative: bool = True,
        relative: bool = True,
        min_effect: float | None = None,
        treated_unit: str | None = None,
        period: Literal["intervention", "post", "comparison"] | None = None,
        prefix: str = "Post-period",
    ) -> EffectSummary:
        """
        Generate a decision-ready summary of causal effects for Synthetic Control.

        Parameters
        ----------
        group : {"prior", "posterior"}, default "posterior"
            Which draw group to summarize. ``"prior"`` requires
            :meth:`sample_prior_predictive` and produces prior-appropriate
            prose — under a neutral prior, ``P(effect > 0)`` should sit near
            0.5, so a tail probability far from 0.5 flags a design-matrix or
            prior-specification problem rather than a causal finding.
            ``"posterior"`` requires :meth:`fit`.
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

        # Resolve the group's bundle once; helpers consume containers.
        bundle = self._require_bundle(group)

        windowed_impact, window_coords = _extract_window(
            bundle.impact_post,
            self.datapost.index,
            window,
            treated_unit=treated_unit,
        )
        counterfactual = _extract_counterfactual(
            bundle.predictions_post, window_coords, treated_unit=treated_unit
        )
        return _effect_summary_timeseries(
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
            group=group,
        )
