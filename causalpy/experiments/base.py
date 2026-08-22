#   Copyright 2022 - 2026 The PyMC Labs Developers
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
Base class for quasi experimental designs.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.base import RegressorMixin

from causalpy.custom_exceptions import (
    GroupNotSampleedException,
    PriorPredictiveNotSupportedException,
)
from causalpy.experiments.model_adapter import ModelAdapter, make_model_adapter
from causalpy.maketables_adapters import coefficient_table, get_maketables_adapter
from causalpy.pymc_forecast_models import PyMCForecastModel
from causalpy.pymc_models import PyMCModel
from causalpy.reporting import EffectSummary

logger = logging.getLogger(__name__)


def _apply_legend_kwargs(legend: Any, kwargs: dict[str, Any]) -> None:
    """Mutate an existing Legend in place without recreating it.

    This preserves custom handles (e.g. ``(Line2D, PolyCollection)`` tuples
    built by :func:`~causalpy.plot_utils.plot_posterior_over_x` with
    ``kind="ribbon"``) that would be lost if the legend were rebuilt with
    ``ax.legend()``.

    Supported keys: ``loc``, ``bbox_to_anchor``, ``bbox_transform`` (only
    with ``bbox_to_anchor``), ``fontsize``, ``frameon``, ``title``.

    Raises
    ------
    TypeError
        If *kwargs* contains keys that cannot be applied in place.
    """
    _SUPPORTED = {
        "loc",
        "bbox_to_anchor",
        "bbox_transform",
        "fontsize",
        "frameon",
        "title",
    }
    unsupported = set(kwargs) - _SUPPORTED
    if unsupported:
        raise TypeError(
            f"legend_kwargs keys not supported for in-place mutation: "
            f"{sorted(unsupported)}. Supported keys: {sorted(_SUPPORTED)}"
        )
    if "bbox_transform" in kwargs and "bbox_to_anchor" not in kwargs:
        raise TypeError(
            "bbox_transform requires bbox_to_anchor to be specified as well"
        )

    if "loc" in kwargs:
        loc = kwargs["loc"]
        # set_loc is public in matplotlib >= 3.8; fall back to the stable
        # private helper for older versions, converting string names to
        # numeric codes since _set_loc may not accept strings.
        if hasattr(legend, "set_loc"):
            legend.set_loc(loc)
        else:
            if isinstance(loc, str):  # pragma: no cover
                loc = legend.codes.get(loc, loc)
            legend._set_loc(loc)  # pragma: no cover
    if "bbox_to_anchor" in kwargs:
        legend.set_bbox_to_anchor(
            kwargs["bbox_to_anchor"], kwargs.get("bbox_transform")
        )
    if "fontsize" in kwargs:
        for text in legend.get_texts():
            text.set_fontsize(kwargs["fontsize"])
    if "frameon" in kwargs:
        legend.set_frame_on(kwargs["frameon"])
    if "title" in kwargs:
        legend.set_title(kwargs["title"])


class BaseExperiment(ABC):
    """Base class for quasi experimental designs.

    Subclasses should set ``_default_model_class`` to a PyMC model class
    (e.g. ``LinearRegression``) so that ``model=None`` instantiates a sensible
    Bayesian default. To use an OLS/sklearn model — or, for experiments that
    declare ``supports_pymc_forecast``, a
    :class:`~causalpy.pymc_forecast_models.PyMCForecastModel` — pass one
    explicitly.

    Parameters
    ----------
    model : PyMCModel, RegressorMixin, PyMCForecastModel, or None, default None
        Model instance to use. If ``None`` and ``_default_model_class`` is set,
        an instance of that default class is constructed.

    Notes
    -----
    Optional ``maketables`` integration is exposed through ``__maketables_*``
    hooks. Users can control the HDI interval level used by
    ``ETable(result)`` via :meth:`set_maketables_options`, for example:
    ``result.set_maketables_options(hdi_prob=0.95)``.
    """

    labels: list[str]
    data: pd.DataFrame

    supports_bayes: bool
    supports_ols: bool
    supports_pymc_forecast: bool = False

    _default_model_class: type[PyMCModel] | None = None

    @staticmethod
    def _build_design_dataset(
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        *,
        obs_ind: np.ndarray | pd.Index,
        coeffs: list[str],
        treated_units: list[str] | None = None,
    ) -> xr.Dataset:
        """Build a standard ``xr.Dataset`` from raw design matrices.

        Parameters
        ----------
        X_raw : np.ndarray
            Predictor matrix, shape ``(n_obs, n_coeffs)``.
        y_raw : np.ndarray
            Outcome matrix, shape ``(n_obs, n_units)``.
        obs_ind : array-like
            Observation index coordinates.
        coeffs : list[str]
            Coefficient / column names for ``X_raw``.
        treated_units : list[str], optional
            Names for the treated-unit dimension of ``y_raw``.
            Defaults to ``["unit_0"]``.
        """
        if treated_units is None:
            treated_units = ["unit_0"]
        return xr.Dataset(
            {
                "X": xr.DataArray(
                    X_raw,
                    dims=["obs_ind", "coeffs"],
                    coords={"obs_ind": obs_ind, "coeffs": coeffs},
                ),
                "y": xr.DataArray(
                    y_raw,
                    dims=["obs_ind", "treated_units"],
                    coords={"obs_ind": obs_ind, "treated_units": treated_units},
                ),
            }
        )

    _model_backend: ModelAdapter

    #: Whether this experiment produces grouped result bundles. Experiments
    #: that store no draw-derived state (IV, IPW, PanelRegression) set False
    #: and key their fitted-state off the backend instead.
    _supports_results: bool = True

    def __init__(
        self, model: PyMCModel | RegressorMixin | PyMCForecastModel | None = None
    ) -> None:
        adapter = make_model_adapter(
            model,
            default_model_class=self._default_model_class,
            supports_bayes=self.supports_bayes,
            supports_ols=self.supports_ols,
            supports_pymc_forecast=self.supports_pymc_forecast,
        )
        self._model_backend = adapter
        self.model = adapter.model
        self._result: Any | None = None
        self._prior_result: Any | None = None

    @property
    def model(self) -> PyMCModel | RegressorMixin | PyMCForecastModel:
        """The underlying model instance.

        Assigning a new model is the documented reset: it swaps in the fresh
        instance and clears ``idata``, ``result``, and ``prior_result``,
        because graph identity *is* the model instance. Prior revision works
        through assignment (``exp.model = Model(priors={...})``); there is no
        ``set_priors()``.
        """
        return self._model_backend.model

    @model.setter
    def model(self, value: PyMCModel | RegressorMixin | PyMCForecastModel) -> None:
        """Install *value* as the backend model and reset all lifecycle state.

        Parameters
        ----------
        value : PyMCModel, RegressorMixin, or PyMCForecastModel
            The new backend model instance.

        Notes
        -----
        The swap clears ``idata``, ``result``, and ``prior_result``: graph
        identity is the model instance, so stale draws would be incoherent.
        This assignment is the documented prior-revision mechanism.
        """
        self._model_backend = make_model_adapter(
            value,
            default_model_class=self._default_model_class,
            supports_bayes=self.supports_bayes,
            supports_ols=self.supports_ols,
            supports_pymc_forecast=self.supports_pymc_forecast,
        )
        self._result = None
        self._prior_result = None

    @property
    def idata(self) -> xr.DataTree | None:
        """Return fitted DataTree when the model backend supports it."""
        return self._model_backend.idata

    @property
    def is_configured(self) -> bool:
        """Whether construction succeeded: design matrices are ready.

        Always ``True`` on a successfully constructed experiment; sampling has
        not happened unless the other lifecycle predicates say so.
        """
        return True

    @property
    def is_built(self) -> bool:
        """Whether the model graph / fit design exists (no draws implied)."""
        return self._model_backend.is_built or self.is_fitted

    @property
    def is_fitted(self) -> bool:
        """Whether posterior draws and the posterior result bundle exist."""
        if self._supports_results:
            return self._result is not None
        return self._model_backend.has_posterior

    @property
    def has_prior_predictive(self) -> bool:
        """Whether prior draws and the prior result bundle exist."""
        if self._supports_results:
            return self._prior_result is not None
        return self._model_backend.has_prior

    @property
    def result(self) -> Any:
        """Posterior-group result bundle; raises before :meth:`fit`."""
        if not self._supports_results:
            raise NotImplementedError(
                f"{type(self).__name__} does not produce a grouped result "
                "bundle; inspect the backend draws via .idata instead."
            )
        if self._result is None:
            raise GroupNotSampleedException(
                f"No posterior draws are available. Call "
                f"{type(self).__name__}.fit() first.",
                group="posterior",
            )
        return self._result

    @property
    def prior_result(self) -> Any:
        """Prior-group result bundle; raises before prior sampling."""
        if not self._supports_results:
            raise NotImplementedError(
                f"{type(self).__name__} does not produce a grouped result "
                "bundle; inspect the backend draws via .idata instead."
            )
        if self._prior_result is None:
            raise GroupNotSampleedException(
                f"No prior predictive draws are available. Call "
                f"{type(self).__name__}.sample_prior_predictive() first.",
                group="prior",
            )
        return self._prior_result

    def build(self) -> Self:
        """Construct the model graph without sampling anything.

        Public, idempotent, and auto-called by both sampling verbs, so users
        never have to call it — but calling it explicitly makes the spec
        inspectable (``pm.model_to_graphviz(exp.model)``,
        ``exp.model.basic_RVs``, merged priors) before any compute is spent.

        Returns
        -------
        Self
            The same experiment, for chaining.
        """
        if not self._model_backend.is_built:
            X, y, coords = self._fit_inputs()
            self._model_backend.build(X=X, y=y, coords=coords)
        return self

    def sample_prior_predictive(self, **kwargs: Any) -> Self:
        """Run the optional prior phase and populate :attr:`prior_result`.

        Keyword arguments override ``model.prior_sample_kwargs`` for this call
        only. Re-running overwrites the previous prior groups and
        ``prior_result`` without touching any posterior state.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`pymc.sample_prior_predictive`, overriding the
            model's stored ``prior_sample_kwargs`` for this call only.

        Returns
        -------
        Self
            The same experiment, for chaining.

        Raises
        ------
        PriorPredictiveNotSupportedException
            If the model backend declares no prior predictive capability.
        """
        if not self._model_backend.supports_prior_predictive:
            raise PriorPredictiveNotSupportedException(
                f"The {type(self.model).__name__} backend does not support "
                "prior predictive sampling."
            )
        self.build()
        resolved = {
            **getattr(self.model, "prior_sample_kwargs", {}),
            **kwargs,
        }
        self._model_backend.sample_prior_predictive(**resolved)
        if self._supports_results:
            self._finalize("prior")
        return self

    def fit(self, **kwargs: Any) -> Self:
        """Run the posterior phase and populate :attr:`result`.

        Builds the graph (idempotent), samples NUTS plus posterior predictive
        draws, then — when the backend supports a prior phase and no prior
        state exists yet — fills the prior groups and :attr:`prior_result` so
        ``idata`` is as complete as the historical eager fit produced. The
        posterior runs FIRST because forward-sampling machinery conditions
        through the graph's mutable data nodes; re-arming them for every
        sampling call keeps each phase's draws computed from the right design.
        Standalone prior checks stay cheap: call
        :meth:`sample_prior_predictive` directly before :meth:`fit`.
        Re-running overwrites posterior state only and warns; prior state is
        preserved.

        Parameters
        ----------
        **kwargs
            Forwarded to the posterior sampler, overriding the model's stored
            ``sample_kwargs`` for this call only.

        Returns
        -------
        Self
            The same experiment, for chaining. This turns every pre-1.0 call
            site into a one-token migration:
            ``cp.InterruptedTimeSeries(...).fit()``.
        """
        self.build()
        if self._model_backend.has_posterior:
            warnings.warn(
                f"Refitting {type(self).__name__}: the previous posterior "
                "draws will be replaced. Prior-phase state, if any, is "
                "preserved.",
                UserWarning,
                stacklevel=2,
            )
        self._model_backend.sample_posterior(**kwargs)
        if self._supports_results:
            self._finalize("posterior")
        if (
            self._model_backend.supports_prior_predictive
            and not self.has_prior_predictive
        ):
            self.sample_prior_predictive()
        return self

    def _fit_inputs(self) -> tuple[Any, Any, dict[str, Any] | None]:
        """Return ``(X, y, coords)`` handed to the backend at build time."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _fit_inputs() or override "
            "the lifecycle verbs."
        )

    def _finalize(self, group: Literal["prior", "posterior"]) -> None:
        """Compute the group's result bundle from its draws and assign it."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _finalize(group)."
        )

    def _resolve_group(self, group: str) -> Any:
        """Guard and resolve the read-method draw group.

        Raises :class:`~causalpy.custom_exceptions.GroupNotSampleedException`
        naming the missing lifecycle verb; returns ``None`` for experiments
        without result bundles (their read methods guard on fitted state
        instead).

        Parameters
        ----------
        group : {"prior", "posterior"}
            Requested draw group.
        """
        if group not in ("prior", "posterior"):
            raise ValueError(f"group must be 'prior' or 'posterior', got {group!r}")
        if not self._supports_results:
            if not self.is_fitted:
                raise GroupNotSampleedException(
                    f"No posterior draws are available. Call "
                    f"{type(self).__name__}.fit() first.",
                    group="posterior",
                )
            return None
        if group == "prior":
            if self._prior_result is None:
                raise GroupNotSampleedException(
                    f"No prior predictive draws are available. Call "
                    f"{type(self).__name__}.sample_prior_predictive() first.",
                    group="prior",
                )
            return self._prior_result
        if self._result is None:
            raise GroupNotSampleedException(
                f"No posterior draws are available. Call "
                f"{type(self).__name__}.fit() first.",
                group="posterior",
            )
        return self._result

    def print_coefficients(self, round_to: int | None = None) -> None:
        """Ask the model to print its posterior coefficients.
        Posterior-only by design: prior coefficient draws include merged
        data-driven defaults the user never typed. To inspect those, call
        :meth:`build` first and read ``exp.model.priors`` /
        ``exp.model.basic_RVs`` directly — that is a better tool for prior
        inspection than a second printing path.

        Parameters
        ----------
        round_to : int, optional
            Number of significant figures to round to. Defaults to None,
            in which case 2 significant figures are used.

        Raises
        ------
        GroupNotSampleedException
            If the experiment has not been fitted yet.
        """
        if not self.is_fitted:
            raise GroupNotSampleedException(
                f"No posterior draws are available. Call "
                f"{type(self).__name__}.fit() before printing coefficients.",
                group="posterior",
            )
        self._model_backend.print_coefficients(self.labels, round_to)

    def set_maketables_options(self, *, hdi_prob: float | None = None) -> None:
        """Set optional maketables rendering options for this experiment.

        Parameters
        ----------
        hdi_prob : float, optional
            Bayesian HDI probability used for PyMC coefficient interval columns in
            ``__maketables_coef_table__`` and therefore in ``ETable(result)``.
            Must satisfy ``0 < hdi_prob < 1``.

        Examples
        --------
        >>> result.set_maketables_options(hdi_prob=0.95)  # doctest: +SKIP
        >>> # Subsequent ETable(result) calls use 95% HDI bounds
        """
        if hdi_prob is not None:
            hdi_prob = float(hdi_prob)
            if not 0 < hdi_prob < 1:
                msg = f"hdi_prob must be in (0, 1), got {hdi_prob!r}"
                raise ValueError(msg)
            self._maketables_hdi_prob = hdi_prob

    @property
    def __maketables_coef_table__(self) -> pd.DataFrame:
        """Optional maketables plugin hook for coefficient tables.

        Interval columns use the HDI probability set by
        :meth:`set_maketables_options` when the canonical coefficient container
        carries posterior draws.
        """
        return coefficient_table(self)

    def __maketables_stat__(self, key: str) -> Any:
        """Optional maketables plugin hook for model-level statistics."""
        return get_maketables_adapter(self._model_backend).stat(self, key)

    @property
    def __maketables_depvar__(self) -> str:
        """Optional maketables plugin hook for dependent variable name."""
        return str(
            getattr(
                self,
                "outcome_variable_name",
                getattr(self, "outcome_variable", "y"),
            )
        )

    @property
    def __maketables_vcov_info__(self) -> dict[str, Any]:
        """Optional maketables plugin hook for variance-covariance info."""
        return get_maketables_adapter(self._model_backend).vcov_info(self)

    @property
    def __maketables_stat_labels__(self) -> dict[str, str] | None:
        """Optional maketables plugin hook for statistic labels."""
        return get_maketables_adapter(self._model_backend).stat_labels(self)

    @property
    def __maketables_default_stat_keys__(self) -> list[str] | None:
        """Optional maketables plugin hook for default statistic rows."""
        return get_maketables_adapter(self._model_backend).default_stat_keys(self)

    def _render_plot(
        self,
        *,
        show: bool,
        legend_kwargs: dict[str, Any] | None,
        group: Literal["prior", "posterior"] = "posterior",
        **draw_kwargs: Any,
    ) -> tuple:
        """Template Method shared by every subclass's public ``plot``.

        Each :class:`BaseExperiment` subclass exposes its own explicit,
        kwarg-only public ``plot()`` (issue
        `#886 <https://github.com/pymc-labs/CausalPy/issues/886>`_) and
        forwards the call here. This helper:

        1. Applies the ``arviz-darkgrid`` style for the duration of the
           draw call.
        2. Calls the subclass's backend-agnostic :meth:`_plot`.
        3. Mutates the resulting legend(s) in place when *legend_kwargs*
           is supplied, preserving custom handles built by the subclass.
        4. Optionally calls :func:`matplotlib.pyplot.show`.

        ``BaseExperiment`` deliberately does **not** define a public
        ``plot()`` method: that would inherit a generic
        ``*args, **kwargs`` signature into every subclass and re-introduce
        the discoverability problem described in #886. Subclasses are
        instead required to declare their own ``plot()`` with an explicit
        keyword-only signature and call ``self._render_plot(...)``.

        The ``group`` keyword is guarded here — once for every subclass —
        via :meth:`_resolve_group`: the group-not-sampled error is raised in
        exactly one place. The literal group is forwarded to the subclass's
        ``_plot(group=..., ...)`` which resolves its own bundle; prior-group
        plots render the reduced panel set (counterfactual vs observations
        only).

        Parameters
        ----------
        show : bool
            Whether to call :func:`matplotlib.pyplot.show` after drawing.
        legend_kwargs : dict, optional
            Keyword arguments to adjust legend placement and styling. The
            existing legend is modified **in place** so that custom
            handles (e.g. ``(Line2D, PolyCollection)`` tuples built by
            :func:`~causalpy.plot_utils.plot_posterior_over_x` with
            ``kind="ribbon"``) are preserved.
            Supported keys: ``loc``, ``bbox_to_anchor``, ``fontsize``,
            ``frameon``, ``title``. ``bbox_transform`` is accepted
            alongside ``bbox_to_anchor``.
        **draw_kwargs
            Subclass-specific drawing parameters forwarded verbatim to
            ``_plot``. May include ``kind``, ``ci_kind``, ``ci_prob``, and
            ``num_samples`` for
            :func:`~causalpy.plot_utils.plot_posterior_over_x`.

        Notes
        -----
        **Legend handling and ``plot_posterior_over_x`` return types:** :func:`~causalpy.plot_utils.plot_posterior_over_x`
        returns ``(Line2D, PolyCollection)`` for ``kind="ribbon"`` but
        ``(list[Line2D], None)`` for ``kind="histogram"`` or ``"spaghetti"``.
        Subclass ``_plot`` implementations that assemble
        matplotlib legends from those return values should only pack
        ``(line, patch)`` tuples when calling ``plot_posterior_over_x`` with ``kind="ribbon"``
        (the default). Many current experiment plots always use the ribbon
        default and never forward ``kind``; if a subclass forwards non-ribbon
        kinds, it must build legend handles accordingly. The base class applies
        ``legend_kwargs`` by mutating an existing legend in place, which preserves
        whatever handle objects the subclass attached (including tuple handles
        used for ribbon mean+band).

        Examples
        --------
        Move the legend outside the plot area to avoid overlap:

        >>> fig, ax = result.plot(  # doctest: +SKIP
        ...     show=False,
        ...     legend_kwargs={"loc": "upper left", "bbox_to_anchor": (1.04, 1)},
        ... )
        """
        # The guard runs here, once for every subclass; the subclass's
        # ``_plot`` resolves its own bundle from the group so that override
        # signatures stay keyword-only with defaults (LSP-clean).
        self._resolve_group(group)
        with plt.style.context("arviz-darkgrid"):
            fig, ax = self._plot(group=group, **draw_kwargs)

        # Apply legend customization if requested.  We mutate the existing
        # Legend object in place so that custom handles — especially the
        # (Line2D, PolyCollection) tuples built by plot_posterior_over_x with
        # kind="ribbon" — are preserved
        # exactly as the subclass created them.
        if legend_kwargs is not None:
            # Normalise ax to a flat list so we can iterate uniformly.
            if hasattr(ax, "flat"):
                axes = list(ax.flat)
            elif isinstance(ax, list):
                axes = ax
            else:
                axes = [ax]
            for a in axes:
                legend = a.get_legend()
                if legend is not None:
                    _apply_legend_kwargs(legend, legend_kwargs)
            # Recompute layout when the legend is placed outside the axes
            # so it is not clipped (some subclass plots already call
            # tight_layout before we get here).
            if "bbox_to_anchor" in legend_kwargs:
                fig.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def _plot(self, **kwargs: Any) -> tuple:
        """Draw the experiment figure; called by :meth:`_render_plot`.

        Subclasses implement a single backend-agnostic ``_plot`` that declares
        ``group`` among its keyword-only parameters and resolves the group's
        result bundle itself via ``self._resolve_group(group)`` — never flat
        draw-derived attributes. ``group == "prior"`` renders the reduced
        panel set: the prior-implied counterfactual against observed data
        only, dropping impact and cumulative-impact panels, whose axis
        scaling is meaningless under a prior. Uncertainty rendering should
        key on data properties (e.g.
        :func:`~causalpy.utils.has_posterior_draws`), not backend identity.
        """
        raise NotImplementedError("_plot method not yet implemented")

    @abstractmethod
    def effect_summary(
        self, *, group: Literal["prior", "posterior"] = "posterior"
    ) -> EffectSummary:
        """Generate a decision-ready summary of causal effects.

        Parameters
        ----------
        group : {"prior", "posterior"}, default "posterior"
            ``"prior"`` resolves the prior bundle (after
            :meth:`sample_prior_predictive`) with prior-appropriate prose;
            ``"posterior"`` (default) requires :meth:`fit`. Concrete
            experiments declare additional keyword-only parameters that their
            own effect-summary implementation supports.

        Returns
        -------
        EffectSummary
            Object with ``.table`` (DataFrame) and ``.text`` (str) attributes.
        """
        raise NotImplementedError("effect_summary method not yet implemented")

    def generate_report(
        self,
        *,
        include_plots: bool = True,
        include_effect_summary: bool = True,
        output_file: str | Path | None = None,
    ) -> str:
        """Generate a self-contained HTML report for this experiment.

        This is a convenience wrapper around
        :class:`~causalpy.steps.report.GenerateReport` that does not require
        a full pipeline.

        Parameters
        ----------
        include_plots : bool, default True
            Embed diagnostic plots in the report.
        include_effect_summary : bool, default True
            Include the effect-summary section.
        output_file : str or Path, optional
            If provided, write the HTML report to this path.

        Returns
        -------
        str
            The rendered HTML report.
        """
        from causalpy.pipeline import PipelineContext
        from causalpy.steps.report import GenerateReport

        ctx = PipelineContext(data=self.data)
        ctx.experiment = self
        if include_effect_summary:
            try:
                ctx.effect_summary = self.effect_summary()
            except NotImplementedError:
                # Experiments without an effect-summary implementation keep
                # reporting; a missing draw group must NOT be swallowed —
                # GroupNotSampleedException propagates so the report tells
                # the user to call fit() first.
                logger.debug(
                    "effect_summary() not available for %s",
                    type(self).__name__,
                )

        step = GenerateReport(
            include_plots=include_plots,
            include_effect_summary=include_effect_summary,
            include_sensitivity=False,
            output_file=output_file,
        )
        step.run(ctx)
        return ctx.report
