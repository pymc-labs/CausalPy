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

import contextlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import RegressorMixin

from causalpy.maketables_adapters import get_maketables_adapter
from causalpy.pymc_models import PyMCModel
from causalpy.reporting import EffectSummary
from causalpy.skl_models import create_causalpy_compatible_class


def _apply_legend_kwargs(legend: Any, kwargs: dict[str, Any]) -> None:
    """Mutate an existing Legend in place without recreating it.

    This preserves custom handles (e.g. ``(Line2D, PolyCollection)`` tuples
    built by :func:`~causalpy.plot_utils.plot_xY`) that would be lost if the
    legend were rebuilt with ``ax.legend()``.

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
    Bayesian default. To use an OLS/sklearn model, pass one explicitly.

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

    _default_model_class: type[PyMCModel] | None = None

    def __init__(self, model: PyMCModel | RegressorMixin | None = None) -> None:
        # Ensure we've made any provided Scikit Learn model (as identified as being type
        # RegressorMixin) compatible with CausalPy by appending our custom methods.
        if isinstance(model, RegressorMixin):
            model = create_causalpy_compatible_class(model)

        if model is None and self._default_model_class is not None:
            model = self._default_model_class()

        if model is not None:
            self.model = model

        if getattr(self, "model", None) is None:
            raise ValueError("model not set or passed.")

        if isinstance(self.model, PyMCModel) and not self.supports_bayes:
            raise ValueError("Bayesian models not supported.")

        if isinstance(self.model, RegressorMixin) and not self.supports_ols:
            raise ValueError("OLS models not supported.")

    def fit(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("fit method not implemented")

    @property
    def idata(self) -> az.InferenceData:
        """Return the InferenceData object of the model. Only relevant for PyMC models."""
        return self.model.idata

    def print_coefficients(self, round_to: int | None = None) -> None:
        """Ask the model to print its coefficients.

        Parameters
        ----------
        round_to : int, optional
            Number of significant figures to round to. Defaults to None,
            in which case 2 significant figures are used.
        """
        self.model.print_coefficients(self.labels, round_to)

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

        For PyMC-backed experiments, interval columns use the HDI probability set
        by :meth:`set_maketables_options` (or backend defaults if not set).
        """
        return get_maketables_adapter(self.model).coef_table(self)

    def __maketables_stat__(self, key: str) -> Any:
        """Optional maketables plugin hook for model-level statistics."""
        return get_maketables_adapter(self.model).stat(self, key)

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
        return get_maketables_adapter(self.model).vcov_info(self)

    @property
    def __maketables_stat_labels__(self) -> dict[str, str] | None:
        """Optional maketables plugin hook for statistic labels."""
        return get_maketables_adapter(self.model).stat_labels(self)

    @property
    def __maketables_default_stat_keys__(self) -> list[str] | None:
        """Optional maketables plugin hook for default statistic rows."""
        return get_maketables_adapter(self.model).default_stat_keys(self)

    def _render_plot(
        self,
        *,
        show: bool,
        legend_kwargs: dict[str, Any] | None,
        **draw_kwargs: Any,
    ) -> tuple:
        """Template Method shared by every subclass's public ``plot``.

        Each :class:`BaseExperiment` subclass exposes its own explicit,
        kwarg-only public ``plot()`` (issue
        `#886 <https://github.com/pymc-labs/CausalPy/issues/886>`_) and
        forwards the call here. This helper:

        1. Applies the ``arviz-darkgrid`` style for the duration of the
           draw call.
        2. Dispatches to :meth:`_bayesian_plot` or :meth:`_ols_plot` based
           on the model type.
        3. Mutates the resulting legend(s) in place when *legend_kwargs*
           is supplied, preserving custom handles built by the subclass.
        4. Optionally calls :func:`matplotlib.pyplot.show`.

        ``BaseExperiment`` deliberately does **not** define a public
        ``plot()`` method: that would inherit a generic
        ``*args, **kwargs`` signature into every subclass and re-introduce
        the discoverability problem described in #886. Subclasses are
        instead required to declare their own ``plot()`` with an explicit
        keyword-only signature and call ``self._render_plot(...)``.

        Parameters
        ----------
        show : bool
            Whether to call :func:`matplotlib.pyplot.show` after drawing.
        legend_kwargs : dict, optional
            Keyword arguments to adjust legend placement and styling. The
            existing legend is modified **in place** so that custom
            handles (e.g. ``(Line2D, PolyCollection)`` tuples built by
            :func:`~causalpy.plot_utils.plot_xY`) are preserved.
            Supported keys: ``loc``, ``bbox_to_anchor``, ``fontsize``,
            ``frameon``, ``title``. ``bbox_transform`` is accepted
            alongside ``bbox_to_anchor``.
        **draw_kwargs
            Subclass-specific drawing parameters forwarded verbatim to
            ``_bayesian_plot`` / ``_ols_plot``.
        """
        with plt.style.context(az.style.library["arviz-darkgrid"]):
            if isinstance(self.model, PyMCModel):
                fig, ax = self._bayesian_plot(**draw_kwargs)
            elif isinstance(self.model, RegressorMixin):
                fig, ax = self._ols_plot(**draw_kwargs)
            else:
                raise ValueError("Unsupported model type")

        # Apply legend customization if requested.  We mutate the existing
        # Legend object in place so that custom handles — especially the
        # (Line2D, PolyCollection) tuples built by plot_xY — are preserved
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

    def _bayesian_plot(self, *args: Any, **kwargs: Any) -> tuple:
        """Plot results for Bayesian models. Override in subclasses that support Bayesian."""
        raise NotImplementedError("_bayesian_plot method not yet implemented")

    def _ols_plot(self, *args: Any, **kwargs: Any) -> tuple:
        """Plot results for OLS models. Override in subclasses that support OLS."""
        raise NotImplementedError("_ols_plot method not yet implemented")

    def get_plot_data(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Recover the data of an experiment along with the prediction and causal impact information.

        Internally, this function dispatches to either :func:`get_plot_data_bayesian` or :func:`get_plot_data_ols`
        depending on the model type.
        """
        if isinstance(self.model, PyMCModel):
            return self.get_plot_data_bayesian(*args, **kwargs)
        elif isinstance(self.model, RegressorMixin):
            return self.get_plot_data_ols(*args, **kwargs)
        else:
            raise ValueError("Unsupported model type")

    def get_plot_data_bayesian(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Return plot data for Bayesian models. Override in subclasses that support Bayesian."""
        raise NotImplementedError("get_plot_data_bayesian method not yet implemented")

    def get_plot_data_ols(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Return plot data for OLS models. Override in subclasses that support OLS."""
        raise NotImplementedError("get_plot_data_ols method not yet implemented")

    @abstractmethod
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
        Generate a decision-ready summary of causal effects.

        Parameters
        ----------
        window : str, tuple, or slice, default="post"
            Time window for analysis (ITS/SC only, ignored for DiD/RD):

            - "post": All post-treatment time points (default)
            - (start, end): Tuple of start and end times (handles both datetime and integer indices)
            - slice: Python slice object for integer indices
        direction : {"increase", "decrease", "two-sided"}, default="increase"
            Direction for tail probability calculation (PyMC only, ignored for OLS):

            - "increase": P(effect > 0)
            - "decrease": P(effect < 0)
            - "two-sided": Two-sided p-value, report 1-p as "probability of effect"
        alpha : float, default=0.05
            Significance level for HDI/CI intervals (1-alpha confidence level).
            For Bayesian models the effective HDI probability is
            ``hdi_prob = 1 - alpha``. Note that this is independent of the
            project-wide :data:`~causalpy.constants.HDI_PROB` constant
            (currently 0.94) used by :meth:`plot` and
            :meth:`get_plot_data_bayesian`, so the same experiment may report
            a 95% HDI in :meth:`effect_summary` and a 94% HDI in :meth:`plot`
            with default settings.
        cumulative : bool, default=True
            Whether to include cumulative effect statistics (ITS/SC only, ignored for DiD/RD)
        relative : bool, default=True
            Whether to include relative effect statistics (% change vs counterfactual)
            (ITS/SC only, ignored for DiD/RD)
        min_effect : float, optional
            Region of Practical Equivalence (ROPE) threshold (PyMC only, ignored for OLS).
            If provided, reports ``P(|effect| > min_effect)`` for two-sided or
            ``P(effect > min_effect)`` for one-sided.
        treated_unit : str, optional
            For multi-unit experiments (Synthetic Control), specify which treated unit
            to analyze. If None and multiple units exist, uses first unit.
        period : {"intervention", "post", "comparison"}, optional
            For experiments with multiple periods (e.g., three-period ITS), specify
            which period to summarize. Defaults to None for standard behavior.
        prefix : str, optional
            Prefix for prose generation (e.g., "During intervention", "Post-intervention").
            Defaults to "Post-period".

        Returns
        -------
        EffectSummary
            Object with .table (DataFrame) and .text (str) attributes.
            The .text attribute contains a detailed multi-paragraph narrative report.
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
            with contextlib.suppress(Exception):
                ctx.effect_summary = self.effect_summary()

        step = GenerateReport(
            include_plots=include_plots,
            include_effect_summary=include_effect_summary,
            include_sensitivity=False,
            output_file=output_file,
        )
        step.run(ctx)
        return ctx.report
