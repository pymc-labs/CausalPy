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

import logging
import warnings
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.base import RegressorMixin

from causalpy._arviz_compat import hdi_bounds
from causalpy.constants import HDI_PROB
from causalpy.date_utils import validate_treatment_time_against_index
from causalpy.experiments._panel_counterfactual import WidePanel, resolve_treated_unit
from causalpy.experiments._results import CausalResult
from causalpy.experiments.model_adapter import PyMCModelAdapter, build_coords
from causalpy.experiments.sc_results import (
    DonorPoolQualityResult,
    DressRehearsalResult,
    PowerCurveResult,
)
from causalpy.input_data import DataFrameLike, to_pandas_with_time_index
from causalpy.plot_utils import _PosteriorPlotStyle, format_r2_score
from causalpy.pymc_models import (
    _LEGACY_Y_HAT_PRIOR,
    PyMCModel,
    WeightedSumFitter,
    _uses_stock_y_hat_default,
)
from causalpy.reporting import EffectSummary
from causalpy.utils import check_convex_hull_violation, has_posterior_draws

from .base import BaseExperiment

logger = logging.getLogger(__name__)


def _align_prior_draws(model: PyMCModel) -> None:
    """Keep a design refit's prior phase on the posterior draw count.

    ``sample_prior_predictive`` rejects posterior-only keys such as ``tune``
    and ``chains``, so only ``draws`` and ``random_seed`` are copied.
    """
    prior = dict(model.prior_sample_kwargs)
    if "draws" in model.sample_kwargs:
        prior["draws"] = model.sample_kwargs["draws"]
    if "random_seed" in model.sample_kwargs:
        prior["random_seed"] = model.sample_kwargs["random_seed"]
    model.prior_sample_kwargs = prior


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

    **Design before a post-period exists**

    :meth:`from_pre_period` fits the supplied rows as pre-period only.
    :meth:`donor_pool_quality`, :meth:`validate_design`, and
    :meth:`power_analysis` then ask whether that donor pool can recover an
    effect you inject. They require a PyMC model. ``power_analysis`` adds iid
    Gaussian noise and is a heuristic, not a sampling distribution of the
    treatment effect.

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
    _panel: WidePanel

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
        # to_pandas_with_time_index returns a copy, so index metadata is normalized on an owned frame rather than the caller's.
        pandas_data = to_pandas_with_time_index(data, time_column)
        pandas_data.index.name = "obs_ind"
        self.data = pandas_data
        self.input_validation(pandas_data, treatment_time)
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

        Pre-period: index < treatment_time. This re-splits ``self.data``, so a
        later write there is visible on the next read. The frame is a copy:
        in-place writes do not change ``self.data`` or the stored panel that
        fit, design, and impact use.
        """
        return self.data.loc[self.data.index < self.treatment_time].copy()

    @property
    def datapost(self) -> pd.DataFrame:
        """Data from on or after the treatment time (inclusive).

        Post-period: index >= treatment_time. This re-splits ``self.data``, so a
        later write there is visible on the next read. The frame is a copy:
        in-place writes do not change ``self.data`` or the stored panel that
        fit, design, and impact use.
        """
        return self.data.loc[self.data.index >= self.treatment_time].copy()

    def _prepare_data(self) -> None:
        """Store the wide panel and the weighted-sum design views."""
        self._panel = WidePanel.from_frame(
            self.data,
            self.treatment_time,
            self.control_units,
            self.treated_units,
        )
        self.pre_design = self._weighted_sum_design("pre")
        self.post_design = self._weighted_sum_design("post")

    def _weighted_sum_design(self, period: Literal["pre", "post"]) -> xr.Dataset:
        """Rename the donor axis to ``coeffs`` for ``WeightedSumFitter``.

        The shared panel keeps ``control_units``. Post-period treated outcomes
        come from :meth:`WidePanel._treated_post`, so ``post_design["treated"]``
        cannot drift from what impact subtracts. That method stays private.
        New experiments use :meth:`WidePanel.impacts`.
        """
        control = self._panel.control(period).rename({"control_units": "coeffs"})
        if period == "pre":
            treated = self._panel.treated_pre
        else:
            treated = self._panel._treated_post()
        return xr.Dataset({"control": control, "treated": treated})

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
        """Return the pre-period control/treated matrices and coordinates for build.

        Post-period treated outcomes are excluded. They stay on the panel and
        on ``post_design`` for impact and display, and are not a fit or tuning
        input. ``pre_design["control"]`` is the donor array renamed to
        ``coeffs`` for ``WeightedSumFitter``.
        """
        control_pre = self.pre_design["control"]
        return (
            control_pre,
            self.pre_design["treated"],
            build_coords(
                self.control_units,
                len(self._panel.pre),
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

        predictions_pre = self._model_backend.predict(X=control_pre, group=group)
        if len(self._panel.post) == 0:
            # A design-only fit has no post window. Predicting on an empty
            # design matrix breaks the PyMC graph, so build a zero-length
            # post prediction with the pre-prediction dimensions instead.
            predictions_post = predictions_pre.isel(obs_ind=slice(0, 0)).assign_coords(
                obs_ind=self._panel.post.index
            )
        else:
            predictions_post = self._model_backend.predict(
                X=self.post_design["control"], group=group
            )
        impact_pre, impact_post, impact_post_cumulative = self._panel.impacts(
            predictions_pre, predictions_post
        )

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
    ) -> tuple[plt.Figure, list[plt.Axes]] | tuple[plt.Figure, np.ndarray]:
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
            Uncertainty styling, ``figsize``, and ``plot_predictors`` apply to both groups; ``round_to`` only affects posterior annotations.
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
        ax : list[matplotlib.axes.Axes] or numpy.ndarray
            Posterior axes are the ndarray from ``plt.subplots`` (top:
            predictions, middle: causal impact, bottom: cumulative impact).
            Prior axes are a one-element list.

        Raises
        ------
        ValueError
            If the experiment has no post-period observations. Design-only
            instances from :meth:`from_pre_period` support
            :meth:`donor_pool_quality`, :meth:`validate_design`, and
            :meth:`power_analysis` instead.

        """
        self._require_post_period("plot")
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
    ) -> tuple[plt.Figure, list[plt.Axes]] | tuple[plt.Figure, np.ndarray]:
        """
        Plot the posterior or prior-check figure for a specific treated unit.

        Consumes the resolved group bundle injected by
        :meth:`~causalpy.experiments.base.BaseExperiment._render_plot`.
        Uncertainty bands are drawn only when the container carries draws;
        point-estimate backends (singleton ``chain``/``draw``) get bare lines.

        Parameters
        ----------
        group : {"prior", "posterior"}
            ``"prior"`` renders the reduced single-panel prior-check figure;
            ``"posterior"`` renders the full three-panel layout.
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
        style: _PosteriorPlotStyle = {
            "ci_prob": ci_prob,
            "kind": kind,
            "ci_kind": ci_kind,
            "num_samples": num_samples,
        }
        treated_unit = resolve_treated_unit(self.treated_units, treated_unit)
        title = (
            self._get_score_title(bundle.score, treated_unit, round_to)
            if group == "posterior"
            else ""
        )
        return self._panel.plot(
            bundle,
            group=group,
            treated_unit=treated_unit,
            title=title,
            style=style,
            figsize=figsize,
            plot_predictors=plot_predictors,
        )

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
            Observation columns follow a later write to ``self.data``.
            ``prediction``, ``impact``, and HDI columns stay computed even
            when the input frame already uses those names.

        Raises
        ------
        ValueError
            If the experiment has no post-period observations.

        """
        self._require_post_period("get_plot_data")
        bundle = self._require_bundle(group)
        frame = self._panel.plot_data(
            bundle,
            treated_unit=treated_unit,
            hdi_prob=hdi_prob,
        )
        observed = pd.concat([self.datapre, self.datapost])
        computed = {"prediction", "impact"}
        if has_posterior_draws(bundle.predictions_pre):
            hdi_pct = int(round(hdi_prob * 100))
            computed.update(
                {
                    f"pred_hdi_lower_{hdi_pct}",
                    f"pred_hdi_upper_{hdi_pct}",
                    f"impact_hdi_lower_{hdi_pct}",
                    f"impact_hdi_upper_{hdi_pct}",
                }
            )
        overlay = [column for column in observed.columns if column not in computed]
        if overlay:
            frame.loc[:, overlay] = observed.loc[:, overlay]
        return frame

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

        Raises
        ------
        ValueError
            If the experiment has no post-period observations.

        """
        self._require_post_period("effect_summary")
        if period is not None:
            warnings.warn(
                f"period='{period}' is ignored for SyntheticControl (two-period design only). "
                "Results reflect the entire post-treatment period. "
                "Use the 'window' parameter to analyze specific time ranges.",
                UserWarning,
                stacklevel=2,
            )

        return self._panel.effect_summary(
            self._require_bundle(group),
            group=group,
            window=window,
            direction=direction,
            alpha=alpha,
            cumulative=cumulative,
            relative=relative,
            min_effect=min_effect,
            treated_unit=treated_unit,
            prefix=prefix,
            experiment_type="sc",
        )

    def _require_post_period(self, method: str) -> None:
        """Raise if a post-period method is called on a design-only fit."""
        if len(self.datapost) == 0:
            raise ValueError(
                f"{method} is not available for a design-only synthetic control "
                "with no post-period observations. Use donor_pool_quality(), "
                "validate_design(), or power_analysis() to assess the design."
            )

    def _design_sample_kwargs(
        self, sample_kwargs: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Merge a design refit's sampling overrides onto the parent model."""
        model = self.model
        if not isinstance(model, PyMCModel):
            raise TypeError(
                f"Design refits require a PyMC model. Got {type(model).__name__}."
            )
        refit_kwargs = dict(model.sample_kwargs)
        if sample_kwargs is not None:
            refit_kwargs.update(sample_kwargs)
        refit_kwargs.setdefault("progressbar", False)
        return refit_kwargs

    def _clone_for_design(self, sample_kwargs: dict[str, Any] | None) -> PyMCModel:
        """Return an unfitted clone whose both phases use the design sample size."""
        from causalpy.checks.base import clone_model

        cloned = clone_model(self.model)
        cloned.sample_kwargs = self._design_sample_kwargs(sample_kwargs)
        _align_prior_draws(cloned)
        return cloned

    @classmethod
    def from_pre_period(
        cls,
        data: DataFrameLike,
        control_units: list[str],
        treated_units: list[str],
        model: PyMCModel | RegressorMixin | None = None,
        *,
        min_donor_correlation: float = 0.0,
        auto_scale_sigma: bool = True,
        time_column: str | None = None,
    ) -> Self:
        """Create a design-phase synthetic control fitted only on pre-period data.

        Use this before an experiment is run. Every supplied row is treated as
        pre-period. The returned instance is fitted, so
        :meth:`donor_pool_quality`, :meth:`validate_design`, and
        :meth:`power_analysis` are ready to call. :meth:`plot`,
        :meth:`get_plot_data`, and :meth:`effect_summary` raise because there
        is no post-period window.

        Parameters
        ----------
        data : DataFrameLike
            Historical panel. All rows are used for fitting.
        control_units : list of str
            Donor columns.
        treated_units : list of str
            Treated columns.
        model : PyMCModel or RegressorMixin, optional
            Model to fit. Defaults to ``WeightedSumFitter``.
        min_donor_correlation : float, default 0.0
            Minimum acceptable donor correlation. See the constructor.
        auto_scale_sigma : bool, default True
            Forwarded to the constructor.
        time_column : str, optional
            Column to use as the time index when ``data`` is not already indexed.

        Returns
        -------
        SyntheticControl
            A fitted design-only instance.

        Raises
        ------
        TypeError
            If the index is neither a ``DatetimeIndex`` nor numeric, so a
            pseudo treatment time cannot be placed after the last observation.
        """
        frame = to_pandas_with_time_index(data, time_column)
        if isinstance(frame.index, pd.DatetimeIndex):
            treatment_time = frame.index.max() + pd.Timedelta(days=1)
        elif pd.api.types.is_numeric_dtype(frame.index):
            treatment_time = frame.index.max() + 1
        else:
            raise TypeError(
                "from_pre_period requires a DatetimeIndex or a numeric index "
                "so a pseudo treatment time can be placed after the last "
                "observation."
            )
        experiment = cls(
            frame,
            treatment_time,
            control_units,
            treated_units,
            model=model,
            min_donor_correlation=min_donor_correlation,
            auto_scale_sigma=auto_scale_sigma,
        )
        if isinstance(experiment.model, PyMCModel):
            _align_prior_draws(experiment.model)
        return experiment.fit()

    def validate_design(
        self,
        injected_effect: float,
        *,
        holdout_periods: int | None = None,
        effect_type: Literal["relative", "absolute"] = "relative",
        sample_kwargs: dict[str, Any] | None = None,
    ) -> DressRehearsalResult:
        """Inject a known effect into a holdout window and check recovery.

        The last ``holdout_periods`` pre-period rows become a pseudo-post
        window. A fresh model is fit on the remaining pre-period. The reported
        recovery is the first treated unit's cumulative impact. The injected
        truth is that same unit's cumulative injection, so a one-unit design
        matches the historical comparison. The effect is still applied to
        every treated column.

        Parameters
        ----------
        injected_effect : float
            Effect to inject. ``"relative"`` multiplies treated values by
            ``1 + injected_effect``. ``"absolute"`` adds ``injected_effect``.
        holdout_periods : int, optional
            Pseudo-post length. ``None`` uses about 25% of the pre-period,
            at least 1.
        effect_type : {"relative", "absolute"}, default "relative"
            How ``injected_effect`` is applied.
        sample_kwargs : dict, optional
            MCMC overrides for the refit. Merged onto the parent model's
            ``sample_kwargs``. The prior phase uses the same draw count.

        Returns
        -------
        DressRehearsalResult
            Recovery of the injected effect for the first treated unit.

        Raises
        ------
        TypeError
            If the model is not a ``PyMCModel``.
        ValueError
            If the holdout would leave fewer than 10 pre-period rows, or
            ``effect_type`` is unknown.
        """
        if not isinstance(self.model, PyMCModel):
            raise TypeError(
                "validate_design() requires a PyMC model. "
                f"Got {type(self.model).__name__}."
            )
        if effect_type not in {"relative", "absolute"}:
            raise ValueError(
                f"effect_type must be 'relative' or 'absolute', got {effect_type!r}."
            )

        pre = self.datapre
        n_pre = len(pre)
        if holdout_periods is None:
            holdout_periods = max(1, n_pre // 4)
        remaining = n_pre - holdout_periods
        if remaining < 10:
            raise ValueError(
                f"Holdout of {holdout_periods} periods leaves only {remaining} "
                "pre-period rows (need >= 10)."
            )

        pseudo_treatment_time = pre.index[remaining]
        pseudo_pre = pre.iloc[:remaining].copy()
        pseudo_post = pre.iloc[remaining:].copy()
        scored_unit = self.treated_units[0]
        if effect_type == "relative":
            injected_truth = float(pseudo_post[scored_unit].sum() * injected_effect)
            for column in self.treated_units:
                pseudo_post[column] = pseudo_post[column] * (1 + injected_effect)
        else:
            injected_truth = float(injected_effect * len(pseudo_post))
            for column in self.treated_units:
                pseudo_post[column] = pseudo_post[column] + injected_effect

        refitted = SyntheticControl(
            pd.concat([pseudo_pre, pseudo_post]),
            pseudo_treatment_time,
            list(self.control_units),
            list(self.treated_units),
            model=self._clone_for_design(sample_kwargs),
            min_donor_correlation=self.min_donor_correlation,
            auto_scale_sigma=self.auto_scale_sigma,
        ).fit()

        cum_impact = refitted.result.impact_post_cumulative
        if "treated_units" in cum_impact.dims:
            cum_impact = cum_impact.isel(treated_units=0)
        samples = cum_impact.isel(obs_ind=-1).stack(sample=("chain", "draw"))
        recovered_mean = float(samples.mean().item())
        recovered_hdi = hdi_bounds(np.asarray(samples.values), prob=HDI_PROB)
        lower, upper = recovered_hdi
        return DressRehearsalResult(
            injected_effect=injected_effect,
            effect_type=effect_type,
            recovered_effect_mean=recovered_mean,
            recovered_effect_hdi=recovered_hdi,
            hdi_covers_truth=lower <= injected_truth <= upper,
            posterior_samples=samples,
            injected_truth=injected_truth,
        )

    def power_analysis(
        self,
        effect_sizes: list[float],
        *,
        n_simulations: int = 50,
        criterion: Literal[
            "hdi_excludes_zero", "prob_gt_zero", "recovery_accuracy"
        ] = "hdi_excludes_zero",
        effect_type: Literal["relative", "absolute"] = "relative",
        holdout_periods: int | None = None,
        sample_kwargs: dict[str, Any] | None = None,
        random_seed: int | None = None,
    ) -> PowerCurveResult:
        """Estimate a heuristic power curve by repeating a dress rehearsal.

        For each candidate effect size, the treated pre-period is perturbed
        with iid Gaussian noise and :meth:`validate_design` is run. The noise
        scale is the pre-period residual standard deviation of the first
        treated unit. Every treated unit receives that same residual draw.
        This is not a sampling distribution of the treatment effect, and it
        does not model autocorrelation, seasonality, or heteroscedasticity.
        Those limitations are why an iid Gaussian curve can overstate power
        for a time-series donor pool.

        Parameters
        ----------
        effect_sizes : list of float
            Candidate effects passed to :meth:`validate_design`.
        n_simulations : int, default 50
            Simulations per effect size.
        criterion : {"hdi_excludes_zero", "prob_gt_zero", "recovery_accuracy"}
            Detection rule. ``hdi_excludes_zero`` asks whether the cumulative
            impact HDI excludes 0. ``prob_gt_zero`` asks whether
            ``P(impact > 0) > 0.95``. ``recovery_accuracy`` asks whether the
            posterior mean is within 20% of the injected truth.
        effect_type : {"relative", "absolute"}, default "relative"
            Passed to :meth:`validate_design`.
        holdout_periods : int, optional
            Passed to :meth:`validate_design`.
        sample_kwargs : dict, optional
            MCMC overrides for every refit.
        random_seed : int, optional
            Seed for the residual draws. MCMC reproducibility also requires
            ``random_seed`` inside ``sample_kwargs``.

        Returns
        -------
        PowerCurveResult
            Detection rate and recovery statistics at each requested effect size.

        Raises
        ------
        TypeError
            If the model is not a ``PyMCModel``.
        ValueError
            If ``criterion`` is unknown.
        """
        if not isinstance(self.model, PyMCModel):
            raise TypeError(
                "power_analysis() requires a PyMC model. "
                f"Got {type(self.model).__name__}."
            )
        if n_simulations < 1:
            raise ValueError(f"n_simulations must be >= 1, got {n_simulations}.")
        if criterion not in {
            "hdi_excludes_zero",
            "prob_gt_zero",
            "recovery_accuracy",
        }:
            raise ValueError(f"Unknown criterion: {criterion!r}.")
        if effect_type not in {"relative", "absolute"}:
            raise ValueError(
                f"effect_type must be 'relative' or 'absolute', got {effect_type!r}."
            )

        residuals = self.result.impact_pre
        if "treated_units" in residuals.dims:
            residuals = residuals.isel(treated_units=0)
        residual_std = float(
            residuals.mean(dim=["chain", "draw"]).std(dim="obs_ind").item()
        )
        rng = np.random.default_rng(random_seed)
        pre = self.datapre
        all_results: list[list[DressRehearsalResult]] = []
        detection_rates: list[float] = []
        total = len(effect_sizes) * n_simulations
        completed = 0

        for effect_size in effect_sizes:
            sim_results: list[DressRehearsalResult] = []
            detections = 0
            for _ in range(n_simulations):
                noisy = self._build_noisy_experiment(
                    pre, residual_std, rng, sample_kwargs
                )
                result = noisy.validate_design(
                    effect_size,
                    holdout_periods=holdout_periods,
                    effect_type=effect_type,
                    sample_kwargs=sample_kwargs,
                )
                sim_results.append(result)
                if self._criterion_met(result, criterion):
                    detections += 1
                completed += 1
                if completed % 5 == 0 or completed == total:
                    logger.info(
                        "Power analysis progress: %d/%d simulations complete",
                        completed,
                        total,
                    )
            all_results.append(sim_results)
            detection_rates.append(detections / n_simulations)

        return PowerCurveResult(
            effect_sizes=list(effect_sizes),
            detection_rates=detection_rates,
            criterion=criterion,
            raw_results=all_results,
        )

    def _build_noisy_experiment(
        self,
        pre: pd.DataFrame,
        residual_std: float,
        rng: np.random.Generator,
        sample_kwargs: dict[str, Any] | None,
    ) -> "SyntheticControl":
        """Fit a design-only SyntheticControl on a noisy copy of the pre-period.

        Always returns this class, not a subclass. ``UpDownGeoLift`` does not
        accept the unit arguments this constructor passes.
        """
        noisy = pre.copy()
        noise = rng.normal(0, residual_std, size=len(pre))
        for column in self.treated_units:
            noisy[column] = noisy[column] + noise
        return SyntheticControl.from_pre_period(
            noisy,
            list(self.control_units),
            list(self.treated_units),
            model=self._clone_for_design(sample_kwargs),
            min_donor_correlation=self.min_donor_correlation,
            auto_scale_sigma=self.auto_scale_sigma,
        )

    @staticmethod
    def _criterion_met(result: DressRehearsalResult, criterion: str) -> bool:
        """Return whether one dress rehearsal meets the detection rule."""
        samples = np.asarray(result.posterior_samples.values, dtype=float).ravel()
        if criterion == "hdi_excludes_zero":
            lower, upper = hdi_bounds(samples, prob=HDI_PROB)
            return bool(lower > 0 or upper < 0)
        if criterion == "prob_gt_zero":
            return bool(float((samples > 0).mean()) > 0.95)
        if criterion == "recovery_accuracy":
            if result.injected_truth == 0:
                return abs(result.recovered_effect_mean) < 1e-6
            relative_error = abs(
                (result.recovered_effect_mean - result.injected_truth)
                / result.injected_truth
            )
            return relative_error < 0.2
        raise ValueError(f"Unknown criterion: {criterion!r}.")

    def donor_pool_quality(self) -> DonorPoolQualityResult:
        """Score donor correlation, convex-hull coverage, and weight concentration.

        Weight concentration uses the posterior mean of ``beta``. When that
        posterior has a treated-unit dimension, the first treated unit is
        reported. An undefined donor correlation, such as a constant donor,
        is omitted from the composite correlation score and kept in
        ``per_donor_details``. The experiment must already be fitted.

        Returns
        -------
        DonorPoolQualityResult
            Composite donor-pool scores and the per-donor table.

        Raises
        ------
        TypeError
            If the model is not a ``PyMCModel``.
        RuntimeError
            If the model has no posterior draws.
        """
        if not isinstance(self.model, PyMCModel):
            raise TypeError(
                "donor_pool_quality() requires a PyMC model. "
                f"Got {type(self.model).__name__}."
            )
        idata = self.model.idata
        if idata is None or "posterior" not in idata.children:
            raise RuntimeError(
                "donor_pool_quality() requires a fitted PyMC model. Call fit() first."
            )

        pre = self.datapre
        correlations: dict[str, float] = {}
        for control in self.control_units:
            unit_corrs = [
                pre[treated].corr(pre[control]) for treated in self.treated_units
            ]
            correlations[control] = float(
                np.nanmean(np.asarray(unit_corrs, dtype=float))
            )

        defined = [value for value in correlations.values() if np.isfinite(value)]
        correlation_score = float(np.mean(defined)) if defined else float("nan")

        treated = self.pre_design["treated"]
        control_design = self.pre_design["control"]
        n_violations = 0
        n_points = 0
        n_pre = int(treated.sizes["obs_ind"])
        for unit_index in range(len(self.treated_units)):
            hull = check_convex_hull_violation(
                treated.isel(treated_units=unit_index).values,
                control_design.values,
            )
            n_violations += hull["n_violations"]
            n_points += n_pre
        coverage = 1.0 - (n_violations / n_points) if n_points else 0.0

        beta = idata.posterior["beta"].mean(dim=["chain", "draw"])
        if "treated_units" in beta.dims:
            beta = beta.isel(treated_units=0)
        weights = np.abs(np.asarray(beta.values, dtype=float).ravel())
        weight_sum = float(weights.sum())
        if weight_sum > 0:
            normalized = weights / weight_sum
        else:
            normalized = np.ones_like(weights) / len(weights)
        herfindahl = float(np.sum(normalized**2))
        effective_donors = 1.0 / herfindahl if herfindahl > 0 else float(len(weights))
        details = pd.DataFrame(
            {
                "donor": list(self.control_units),
                "mean_correlation": [correlations[name] for name in self.control_units],
                "mean_weight": weights,
                "normalized_weight": normalized,
            }
        )
        return DonorPoolQualityResult(
            correlation_score=correlation_score,
            convex_hull_coverage=coverage,
            weight_concentration=effective_donors,
            per_donor_details=details,
        )
