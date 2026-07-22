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
"""Interrupted Time Series Analysis."""

import warnings
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from patsy import build_design_matrices
from sklearn.base import RegressorMixin

from causalpy.constants import HDI_PROB, LEGEND_FONT_SIZE
from causalpy.custom_exceptions import BadIndexException
from causalpy.date_utils import _combine_datetime_indices, format_date_axes
from causalpy.experiments.model_adapter import build_coords
from causalpy.formula_utils import build_formula_matrices
from causalpy.plot_utils import (
    _PosteriorPlotStyle,
    extract_r2_score,
    get_hdi_to_df,
    has_posterior_draws,
    plot_posterior_over_x,
)
from causalpy.pymc_forecast_models import PyMCForecastModel
from causalpy.pymc_models import LinearRegression, PyMCModel
from causalpy.reporting import EffectSummary
from causalpy.utils import _as_scalar, round_num

from .base import BaseExperiment


class InterruptedTimeSeries(BaseExperiment):
    """
    The class for interrupted time series analysis.

    Supports both two-period (permanent intervention) and three-period (temporary
    intervention) designs. When ``treatment_end_time`` is provided, the analysis
    splits the post-intervention period into an intervention period and a
    post-intervention period, enabling analysis of effect persistence and decay.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas dataframe with time series data. The index should be either
        a DatetimeIndex or numeric (integer/float), with unique values in
        monotonically increasing order.
    treatment_time : Union[int, float, pd.Timestamp]
        The time when treatment occurred, should be in reference to the data index.
        Must match the index type (DatetimeIndex requires pd.Timestamp).
        **INCLUSIVE**: Observations at exactly ``treatment_time`` are included in the
        post-intervention period (uses ``>=`` comparison).
    formula : str
        A statistical model formula using patsy syntax (e.g., "y ~ 1 + t + C(month)").
    model : Union[PyMCModel, RegressorMixin, PyMCForecastModel], optional
        A PyMC (Bayesian) or sklearn (OLS) model. If None, defaults to a PyMC
        LinearRegression model. Alternatively, a
        :class:`~causalpy.pymc_forecast_models.PyMCForecastModel` wrapping a
        ``pymc_forecast`` forecasting model can serve as the counterfactual
        backend (requires the optional ``pymc-forecast`` dependency); see
        :mod:`causalpy.pymc_forecast_models` for when to prefer it.
    treatment_end_time : Union[int, float, pd.Timestamp], optional
        The time when treatment ended, enabling three-period analysis. Must be
        greater than ``treatment_time`` and within the data range. If None (default),
        the analysis assumes a permanent intervention (two-period design).
        **INCLUSIVE**: Observations at exactly ``treatment_end_time`` are included in the
        post-intervention period (uses ``>=`` comparison).
    **kwargs : dict
        Additional keyword arguments passed to the model.

    Notes
    -----
    **Estimate extraction**

    The model is fitted to pre-intervention observations and predicts the untreated trajectory after the intervention. Pointwise impact is the observed post-intervention outcome minus that one-sided counterfactual prediction, and cumulative impact is its running sum. Bayesian backends subtract the posterior conditional expectation ``mu`` rather than noisy posterior-predictive draws ``y_hat``; OLS subtracts its point prediction.

    This fit-predict-subtract procedure is a reduced-form estimator. From a Bayesian structural perspective, the same impact can be viewed as the response to an intervention shock in a state-space model of the outcome series; see the knowledgebase page on structural causal models for the reduced-form versus structural distinction.

    The three-period design is useful for analyzing temporary interventions such as:

    - Marketing campaigns with defined start and end dates
    - Policy trials or pilot programs
    - Clinical treatments with limited duration
    - Seasonal interventions

    Use ``effect_summary(period="intervention")`` to analyze effects during the
    intervention, and ``effect_summary(period="post")`` to analyze effect persistence
    after the intervention ends.

    Examples
    --------
    **Two-period design (permanent intervention):**

    >>> import causalpy as cp
    >>> df = (
    ...     cp.load_data("its")
    ...     .assign(date=lambda x: pd.to_datetime(x["date"]))
    ...     .set_index("date")
    ... )
    >>> treatment_time = pd.to_datetime("2017-01-01")
    >>> result = cp.InterruptedTimeSeries(
    ...     df,
    ...     treatment_time,
    ...     formula="y ~ 1 + t + C(month)",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={"random_seed": 42, "progressbar": False}
    ...     ),
    ... )

    **Three-period design (temporary intervention):**

    >>> treatment_time = pd.to_datetime("2017-01-01")
    >>> treatment_end_time = pd.to_datetime("2017-06-01")
    >>> result = cp.InterruptedTimeSeries(
    ...     df,
    ...     treatment_time,
    ...     formula="y ~ 1 + t + C(month)",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={"random_seed": 42, "progressbar": False}
    ...     ),
    ...     treatment_end_time=treatment_end_time,
    ... )
    >>> # Get period-specific effect summaries
    >>> intervention_summary = result.effect_summary(period="intervention")
    >>> post_summary = result.effect_summary(period="post")
    """

    supports_ols = True
    supports_bayes = True
    supports_pymc_forecast = True
    _default_model_class = LinearRegression
    _deprecated_design_aliases = {
        "pre_X": ("pre_design", "X"),
        "pre_y": ("pre_design", "y"),
        "post_X": ("post_design", "X"),
        "post_y": ("post_design", "y"),
    }

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_time: int | float | pd.Timestamp,
        formula: str,
        model: PyMCModel | RegressorMixin | PyMCForecastModel | None = None,
        treatment_end_time: int | float | pd.Timestamp | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model)
        self.pre_design: xr.Dataset
        self.post_design: xr.Dataset
        data.index.name = "obs_ind"
        self.data = data
        self.input_validation(data, treatment_time, treatment_end_time)
        self.treatment_time = treatment_time
        self.treatment_end_time = treatment_end_time
        self.expt_type = "Pre-Post Fit"
        self.formula = formula
        self._build_design_matrices()
        self._prepare_data()
        self.algorithm()

    def _build_design_matrices(self) -> None:
        """Build design matrices for pre and post intervention periods using patsy."""
        y, X = build_formula_matrices(self.formula, self.datapre)
        self.outcome_variable_name = y.design_info.column_names[0]
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self._pre_y_raw, self._pre_X_raw = np.asarray(y), np.asarray(X)
        (new_y, new_x) = build_design_matrices(
            [self._y_design_info, self._x_design_info], self.datapost
        )
        self._post_X_raw = np.asarray(new_x)
        self._post_y_raw = np.asarray(new_y)

    def _prepare_data(self) -> None:
        """Bundle design matrices into ``xr.Dataset`` objects for pre and post periods."""
        self.pre_design = self._build_design_dataset(
            self._pre_X_raw,
            self._pre_y_raw,
            obs_ind=self.datapre.index,
            coeffs=self.labels,
        )
        self.post_design = self._build_design_dataset(
            self._post_X_raw,
            self._post_y_raw,
            obs_ind=self.datapost.index,
            coeffs=self.labels,
        )
        del self._pre_X_raw, self._pre_y_raw, self._post_X_raw, self._post_y_raw

    def algorithm(self) -> None:
        """Run the experiment algorithm: fit model, predict, and calculate causal impact."""
        pre_X = self.pre_design["X"]
        pre_y = self.pre_design["y"]
        post_X = self.post_design["X"]
        post_y = self.post_design["y"]

        self._model_backend.fit(
            X=pre_X,
            y=pre_y,
            coords=build_coords(
                self.labels,
                pre_X.shape[0],
                datetime_index=self.datapre.index,
            ),
        )

        self.score = self._model_backend.score(X=pre_X, y=pre_y)

        self.pre_pred = self._model_backend.predict(X=pre_X)
        self.post_pred = self._model_backend.predict(X=post_X, out_of_sample=True)
        # Impact below relies on exact obs_ind alignment; a mismatch (e.g. a bare
        # ndarray X getting arange coords) would silently corrupt the subtraction.
        assert pre_y.obs_ind.equals(self.pre_pred.obs_ind)
        assert post_y.obs_ind.equals(self.post_pred.obs_ind)
        self.pre_impact = (
            pre_y.isel(treated_units=0) - self.pre_pred.isel(treated_units=0)
        ).transpose(..., "obs_ind")
        self.post_impact = (
            post_y.isel(treated_units=0) - self.post_pred.isel(treated_units=0)
        ).transpose(..., "obs_ind")
        self.post_impact_cumulative = self.post_impact.cumsum(dim="obs_ind")

        # Split post period into intervention and post-intervention if treatment_end_time is provided
        if self.treatment_end_time is not None:
            self._split_post_period()

    def input_validation(
        self,
        data: pd.DataFrame,
        treatment_time: int | float | pd.Timestamp,
        treatment_end_time: int | float | pd.Timestamp | None = None,
    ) -> None:
        """Validate the input data and model formula for correctness.

        Parameters
        ----------
        data : pd.DataFrame
            The experiment data.
        treatment_time : int, float, or pd.Timestamp
            Start of the treatment period.
        treatment_end_time : int, float, pd.Timestamp, or None, default None
            Optional end of the treatment period for three-period designs.
        """
        if not data.index.is_unique or not data.index.is_monotonic_increasing:
            raise BadIndexException(
                "data.index must be unique and monotonically increasing. "
                "Sort the data and remove duplicate index values before fitting."
            )
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
        if treatment_end_time is not None:
            # Validate treatment_end_time matches index type
            if isinstance(data.index, pd.DatetimeIndex) and not isinstance(
                treatment_end_time, pd.Timestamp
            ):
                raise BadIndexException(
                    "If data.index is DatetimeIndex, treatment_end_time must be pd.Timestamp."
                )
            if not isinstance(data.index, pd.DatetimeIndex) and isinstance(
                treatment_end_time, pd.Timestamp
            ):
                raise BadIndexException(
                    "If data.index is not DatetimeIndex, treatment_end_time must not be pd.Timestamp."
                )
            # Validate treatment_end_time > treatment_time
            # Type check: we've already validated both match the index type, so they're compatible
            # NOTE: Both treatment_time and treatment_end_time are INCLUSIVE (>=) in their respective periods
            if treatment_end_time <= treatment_time:  # type: ignore[operator]
                raise ValueError(
                    f"treatment_end_time ({treatment_end_time}) must be greater than treatment_time ({treatment_time})"
                )
            # Validate treatment_end_time is within data range
            # NOTE: treatment_end_time is INCLUSIVE, so it can equal data.index.max()
            if treatment_end_time > data.index.max():  # type: ignore[operator]
                raise ValueError(
                    f"treatment_end_time ({treatment_end_time}) is beyond the data range (max: {data.index.max()})"
                )

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

    def _split_post_period(self) -> None:
        """Split post period into intervention and post-intervention periods.

        Creates new attributes for data, predictions, and impacts for each period.
        Only called when treatment_end_time is provided.

        Key insight: intervention_pred and post_intervention_pred are slices of post_pred,
        not new computations. The model makes one continuous forecast (post_pred), which is
        then sliced into two periods for analysis.

        NOTE: treatment_end_time is INCLUSIVE (>=) in post-intervention period.

        - Intervention period: treatment_time <= index < treatment_end_time
        - Post-intervention period: index >= treatment_end_time (inclusive)
        """
        # 1. Create boolean masks based on treatment_end_time
        # NOTE: treatment_end_time is INCLUSIVE (>=) in post-intervention period
        # Intervention period: index < treatment_end_time (exclusive)
        # Post-intervention period: index >= treatment_end_time (inclusive)
        during_mask = self.datapost.index < self.treatment_end_time
        post_mask = self.datapost.index >= self.treatment_end_time

        # 2. Split datapost into data_intervention and data_post_intervention
        self.data_intervention = self.datapost[during_mask]
        self.data_post_intervention = self.datapost[post_mask]

        intervention_coords = self.data_intervention.index
        post_intervention_coords = self.data_post_intervention.index
        self.intervention_pred = self.post_pred.sel(obs_ind=intervention_coords)
        self.post_intervention_pred = self.post_pred.sel(
            obs_ind=post_intervention_coords
        )

        self.intervention_impact = self.post_impact.sel(obs_ind=intervention_coords)
        self.post_intervention_impact = self.post_impact.sel(
            obs_ind=post_intervention_coords
        )
        self.intervention_impact_cumulative = self.intervention_impact.cumsum(
            dim="obs_ind"
        )
        self.post_intervention_impact_cumulative = self.post_intervention_impact.cumsum(
            dim="obs_ind"
        )

    def _comparison_period_summary(
        self,
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        cumulative: bool = True,
        relative: bool = True,
        min_effect: float | None = None,
    ):
        """Generate comparative summary between intervention and post-intervention periods.

        Parameters
        ----------
        direction : {"increase", "decrease", "two-sided"}, default="increase"
            Direction for tail probability calculation (PyMC only)
        alpha : float, default=0.05
            Significance level for HDI/CI intervals
        cumulative : bool, default=True
            Whether to include cumulative effect statistics
        relative : bool, default=True
            Whether to include relative effect statistics
        min_effect : float, optional
            Region of Practical Equivalence (ROPE) threshold (PyMC only)

        Returns
        -------
        EffectSummary
            Object with .table (DataFrame) and .text (str) attributes
        """
        from causalpy.reporting import _extract_hdi_bounds

        has_draws = has_posterior_draws(self.intervention_impact)
        time_dim = "obs_ind"
        hdi_prob = 1 - alpha
        prob_persisted: float | None

        if has_draws:
            # PyMC: Compute statistics for both periods
            intervention_avg = self.intervention_impact.mean(dim=time_dim)
            intervention_mean = _as_scalar(intervention_avg.mean(dim=["chain", "draw"]))
            intervention_hdi = az.hdi(intervention_avg, hdi_prob=hdi_prob)
            intervention_lower, intervention_upper = _extract_hdi_bounds(
                intervention_hdi, hdi_prob
            )

            post_avg = self.post_intervention_impact.mean(dim=time_dim)
            post_mean = _as_scalar(post_avg.mean(dim=["chain", "draw"]))
            post_hdi = az.hdi(post_avg, hdi_prob=hdi_prob)
            post_lower, post_upper = _extract_hdi_bounds(post_hdi, hdi_prob)

            # Persistence ratio: post_mean / intervention_mean (as percentage)
            epsilon = 1e-8
            persistence_ratio_pct = (post_mean / (intervention_mean + epsilon)) * 100

            # Probability that some effect persisted (P(post_mean > 0))
            prob_persisted = _as_scalar((post_avg > 0).mean())

            # Build simple table
            table = pd.DataFrame(
                {
                    "mean": [intervention_mean, post_mean],
                    "hdi_lower": [intervention_lower, post_lower],
                    "hdi_upper": [intervention_upper, post_upper],
                    "persistence_ratio_pct": [None, persistence_ratio_pct],
                    "prob_persisted": [None, prob_persisted],
                },
                index=["intervention", "post_intervention"],
            )

            # Generate simple prose
            hdi_pct = int(hdi_prob * 100)
            text = (
                f"Effect persistence: The post-intervention effect "
                f"({post_mean:.1f}, {hdi_pct}% HDI [{post_lower:.1f}, {post_upper:.1f}]) "
                f"was {persistence_ratio_pct:.1f}% of the intervention effect "
                f"({intervention_mean:.1f}, {hdi_pct}% HDI [{intervention_lower:.1f}, {intervention_upper:.1f}]), "
                f"with a posterior probability of {prob_persisted:.2f} that some effect persisted "
                f"beyond the intervention period."
            )

        else:
            # OLS: Compute statistics for both periods
            from causalpy.reporting import _compute_statistics_ols

            intervention_stats = _compute_statistics_ols(
                np.asarray(self.intervention_impact).ravel(),
                np.asarray(self.intervention_pred).ravel(),
                alpha=alpha,
                cumulative=False,
                relative=False,
            )

            post_stats = _compute_statistics_ols(
                np.asarray(self.post_intervention_impact).ravel(),
                np.asarray(self.post_intervention_pred).ravel(),
                alpha=alpha,
                cumulative=False,
                relative=False,
            )

            # Persistence ratio (as percentage)
            epsilon = 1e-8
            persistence_ratio_pct = (
                post_stats["avg"]["mean"]
                / (intervention_stats["avg"]["mean"] + epsilon)
            ) * 100

            # For OLS, use 1 - p-value as proxy for probability
            prob_persisted = (
                1 - post_stats["avg"]["p_value"]
                if "p_value" in post_stats["avg"]
                else None
            )

            # Build simple table
            table_data = {
                "mean": [
                    intervention_stats["avg"]["mean"],
                    post_stats["avg"]["mean"],
                ],
                "ci_lower": [
                    intervention_stats["avg"]["ci_lower"],
                    post_stats["avg"]["ci_lower"],
                ],
                "ci_upper": [
                    intervention_stats["avg"]["ci_upper"],
                    post_stats["avg"]["ci_upper"],
                ],
                "persistence_ratio_pct": [None, persistence_ratio_pct],
            }
            if prob_persisted is not None:
                table_data["prob_persisted"] = [None, prob_persisted]

            table = pd.DataFrame(
                table_data,
                index=["intervention", "post_intervention"],
            )

            # Generate simple prose
            ci_pct = int((1 - alpha) * 100)
            if prob_persisted is not None:
                text = (
                    f"Effect persistence: The post-intervention effect "
                    f"({post_stats['avg']['mean']:.1f}, {ci_pct}% CI [{post_stats['avg']['ci_lower']:.1f}, {post_stats['avg']['ci_upper']:.1f}]) "
                    f"was {persistence_ratio_pct:.1f}% of the intervention effect "
                    f"({intervention_stats['avg']['mean']:.1f}, {ci_pct}% CI [{intervention_stats['avg']['ci_lower']:.1f}, {intervention_stats['avg']['ci_upper']:.1f}]), "
                    f"with a probability of {prob_persisted:.2f} that some effect persisted "
                    f"beyond the intervention period."
                )
            else:
                text = (
                    f"Effect persistence: The post-intervention effect "
                    f"({post_stats['avg']['mean']:.1f}, {ci_pct}% CI [{post_stats['avg']['ci_lower']:.1f}, {post_stats['avg']['ci_upper']:.1f}]) "
                    f"was {persistence_ratio_pct:.1f}% of the intervention effect "
                    f"({intervention_stats['avg']['mean']:.1f}, {ci_pct}% CI [{intervention_stats['avg']['ci_lower']:.1f}, {intervention_stats['avg']['ci_upper']:.1f}])."
                )

        return EffectSummary(table=table, text=text)

    def summary(self, round_to: int | None = None) -> None:
        """Print summary of main results and model coefficients.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2. Use
            ``None`` to return raw numbers.
        """
        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        self.print_coefficients(round_to)

    def plot(
        self,
        *,
        round_to: int | None = 2,
        ci_prob: float = HDI_PROB,
        hdi_prob: float | None = None,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        figsize: tuple[float, float] = (7, 8),
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot the interrupted time-series results.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round numerical results in the figure
            title (e.g. the Bayesian :math:`R^2`). Defaults to 2. Use
            ``None`` to render raw numbers.
        ci_prob : float
            Probability mass of the credible interval drawn around the
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
            ci_prob=ci_prob,
            kind=kind,
            ci_kind=ci_kind,
            num_samples=num_samples,
            figsize=figsize,
        )

    @staticmethod
    def _draw_singleton_hdi_marker(
        ax: plt.Axes,
        x: Any,
        Y: xr.DataArray,
        color: str,
        hdi_prob: float = HDI_PROB,
    ) -> Any:
        """Overlay a median dot + HDI errorbar for a single post-period datum.

        When ``plot_posterior_over_x`` is called with ``kind="ribbon"`` and
        HDI intervals, ``arviz.plot_hdi`` renders a degenerate zero-area polygon
        when the post-period contains a single observation, so neither the median
        line nor the HDI ribbon is visible. Drawing an explicit point and errorbar
        makes both the central tendency and the uncertainty plain to read in that
        edge case. Returns the matplotlib ``ErrorbarContainer`` so callers can use
        it as a legend handle.
        """
        Y_plot = Y.isel(treated_units=0) if "treated_units" in Y.dims else Y
        median = float(np.asarray(Y_plot.median(("chain", "draw")).values).item())
        hdi = az.hdi(Y_plot, hdi_prob=hdi_prob)
        data_var = list(hdi.data_vars)[0]
        bounds = np.asarray(hdi[data_var].values).reshape(-1)
        lower, upper = float(bounds[0]), float(bounds[1])
        return ax.errorbar(
            x,
            [median],
            yerr=[[median - lower], [upper - median]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            zorder=3,
        )

    def _plot(
        self,
        round_to: int | None = 2,
        ci_prob: float = HDI_PROB,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        figsize: tuple[float, float] = (7, 8),
        **kwargs: Any,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plot the results.

        Consumes the canonical prediction container from any backend.
        Uncertainty bands are drawn only when the container carries posterior
        draws; point-estimate backends (singleton ``chain``/``draw``) get bare
        lines.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2. Use ``None``
            to return raw numbers.
        ci_prob : float, optional
            Probability mass of the credible interval drawn around the
            posterior predictive, causal impact, and cumulative impact bands.
            Must be in ``(0, 1]``. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
        figsize : tuple of (float, float), optional
            Width and height of the figure in inches. Defaults to ``(7, 8)``.
        """
        counterfactual_label = "Counterfactual"
        with_uncertainty = has_posterior_draws(self.pre_pred)
        single_post_obs = len(self.datapost) <= 1
        style: _PosteriorPlotStyle = {
            "ci_prob": ci_prob,
            "kind": kind,
            "ci_kind": ci_kind,
            "num_samples": num_samples,
        }

        pre_pred = self.pre_pred.isel(treated_units=0)
        post_pred = self.post_pred.isel(treated_units=0)
        pre_y = self.pre_design["y"].isel(treated_units=0)
        post_y = self.post_design["y"].isel(treated_units=0)

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

            (h,) = ax[0].plot(
                self.datapre.index,
                pre_y,
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
            if single_post_obs:
                # plot_posterior_over_x's HDI ribbon collapses to a zero-area polygon for a
                # single post-period datum; overlay an explicit median + HDI
                # errorbar so the counterfactual is still visible. Use the
                # errorbar artist itself as the legend handle so the legend
                # matches what is actually drawn.
                errbar = self._draw_singleton_hdi_marker(
                    ax[0], self.datapost.index, self.post_pred, color="C1"
                )
                handles.append(errbar)
            else:
                handles.append((h_line, h_patch))
            labels.append(counterfactual_label)

            ax[0].plot(self.datapost.index, post_y, "k.", zorder=3)
        else:
            ax[0].plot(self.datapre.index, pre_y, "k.")
            ax[0].plot(
                self.datapre.index,
                pre_pred.mean(("chain", "draw")),
                c="k",
                label="model fit",
            )
            ax[0].plot(self.datapost.index, post_y, "k.")
            ax[0].plot(
                self.datapost.index,
                post_pred.mean(("chain", "draw")),
                label=counterfactual_label,
                ls=":",
                c="k",
            )

        # Shaded causal effect (only meaningful when there are >=2 post-period
        # points; with a single datum the fill_between collapses to nothing,
        # so we omit the legend entry to avoid misleading the reader).
        h = ax[0].fill_between(
            self.datapost.index,
            y1=post_pred.mean(("chain", "draw")),
            y2=post_y,
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        if with_uncertainty and not single_post_obs:
            handles.append(h)
            labels.append("Causal impact")

        # Title with R^2; scores carrying a dispersion entry render as Bayesian
        r2_val, r2_std_val = extract_r2_score(self.score)
        assert r2_val is not None  # both backends' score containers carry R^2
        if r2_std_val is not None:
            title_str = (
                f"Pre-intervention Bayesian $R^2$: {round_num(r2_val, round_to)}"
                f"\n(std = {round_num(r2_std_val, round_to)})"
            )
        else:
            title_str = (
                f"$R^2$ on pre-intervention data = {round_num(r2_val, round_to)}"
            )
        ax[0].set(title=title_str)

        # MIDDLE PLOT -----------------------------------------------
        if with_uncertainty:
            plot_posterior_over_x(
                self.datapre.index,
                self.pre_impact,
                ax=ax[1],
                **style,
                plot_hdi_kwargs={"color": "C0"},
            )
            plot_posterior_over_x(
                self.datapost.index,
                self.post_impact,
                ax=ax[1],
                **style,
                plot_hdi_kwargs={"color": "C1"},
            )
            if single_post_obs:
                self._draw_singleton_hdi_marker(
                    ax[1], self.datapost.index, self.post_impact, color="C1"
                )
        else:
            ax[1].plot(
                self.datapre.index, self.pre_impact.mean(("chain", "draw")), "k."
            )
            ax[1].plot(
                self.datapost.index,
                self.post_impact.mean(("chain", "draw")),
                "k.",
                label=counterfactual_label,
            )
        ax[1].axhline(y=0, c="k")
        ax[1].fill_between(
            self.datapost.index,
            y1=self.post_impact.mean(("chain", "draw")),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[1].set(title="Causal Impact")

        # BOTTOM PLOT -----------------------------------------------
        if with_uncertainty:
            plot_posterior_over_x(
                self.datapost.index,
                self.post_impact_cumulative,
                ax=ax[2],
                **style,
                plot_hdi_kwargs={"color": "C1"},
            )
            if single_post_obs:
                self._draw_singleton_hdi_marker(
                    ax[2], self.datapost.index, self.post_impact_cumulative, color="C1"
                )
        else:
            ax[2].plot(
                self.datapost.index,
                self.post_impact_cumulative.mean(("chain", "draw")),
                c="k",
            )
        ax[2].axhline(y=0, c="k")
        ax[2].set(title="Cumulative Causal Impact")

        # Intervention lines. Use a thin dashed black style and a zorder just
        # below the data so the treatment marker reads as a neutral
        # annotation rather than data, and never occludes data points or HDI
        # ribbons - important for the edge case of very few post-treatment
        # observations where the marker can land exactly on top of the only
        # post-period datum.
        for i in [0, 1, 2]:
            ax[i].axvline(
                x=self.treatment_time,
                ls="--",
                lw=1.5,
                color="k",
                zorder=1.5,
                label="Treatment start" if i == 0 else None,
            )
            if self.treatment_end_time is not None:
                ax[i].axvline(
                    x=self.treatment_end_time,
                    ls=":",
                    lw=1.5,
                    color="k",
                    zorder=1.5,
                    label="Treatment end" if i == 0 else None,
                )

        if with_uncertainty:
            ax[0].legend(
                handles=(h_tuple for h_tuple in handles),
                labels=labels,
                fontsize=LEGEND_FONT_SIZE,
            )
        else:
            # Collect labelled artists (including the treatment lines)
            ax[0].legend(fontsize=LEGEND_FONT_SIZE)

        # Apply intelligent date formatting if data has datetime index
        if isinstance(self.datapre.index, pd.DatetimeIndex):
            # Combine pre and post indices for full date range
            full_index = _combine_datetime_indices(
                pd.DatetimeIndex(self.datapre.index),
                pd.DatetimeIndex(self.datapost.index),
            )
            format_date_axes(ax, full_index)

        return fig, ax

    def get_plot_data(self, hdi_prob: float = HDI_PROB) -> pd.DataFrame:
        """
        Recover the data of the experiment along with the prediction and causal impact information.

        HDI columns are included only when the prediction container carries
        posterior draws (point-estimate backends return just ``prediction``
        and ``impact``).

        Parameters
        ----------
        hdi_prob : float, default :data:`~causalpy.constants.HDI_PROB`
            Probability mass of the highest density interval. Defaults to the
            project-wide :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
            Ignored when the prediction container has no posterior draws.
        """
        with_uncertainty = has_posterior_draws(self.pre_pred)
        hdi_pct = int(round(hdi_prob * 100))

        pre_data = self.datapre.copy()
        post_data = self.datapost.copy()

        pre_mu = self.pre_pred.isel(treated_units=0)
        post_mu = self.post_pred.isel(treated_units=0)
        pre_data["prediction"] = pre_mu.mean(("chain", "draw")).values
        post_data["prediction"] = post_mu.mean(("chain", "draw")).values

        if with_uncertainty:
            pred_lower_col = f"pred_hdi_lower_{hdi_pct}"
            pred_upper_col = f"pred_hdi_upper_{hdi_pct}"
            hdi_pre_pred = get_hdi_to_df(self.pre_pred, hdi_prob=hdi_prob)
            hdi_post_pred = get_hdi_to_df(self.post_pred, hdi_prob=hdi_prob)
            pre_data[[pred_lower_col, pred_upper_col]] = hdi_pre_pred.xs(
                "unit_0", level="treated_units"
            ).set_index(pre_data.index)
            post_data[[pred_lower_col, pred_upper_col]] = hdi_post_pred.xs(
                "unit_0", level="treated_units"
            ).set_index(post_data.index)

        pre_data["impact"] = self.pre_impact.mean(dim=["chain", "draw"]).values
        post_data["impact"] = self.post_impact.mean(dim=["chain", "draw"]).values

        if with_uncertainty:
            impact_lower_col = f"impact_hdi_lower_{hdi_pct}"
            impact_upper_col = f"impact_hdi_upper_{hdi_pct}"
            # Compute impact HDIs directly via quantiles over posterior dims to avoid column shape issues
            alpha = 1 - hdi_prob
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2

            pre_data[impact_lower_col] = (
                self.pre_impact.quantile(lower_q, dim=["chain", "draw"])
                .to_series()
                .reindex(pre_data.index)
                .values
            )
            pre_data[impact_upper_col] = (
                self.pre_impact.quantile(upper_q, dim=["chain", "draw"])
                .to_series()
                .reindex(pre_data.index)
                .values
            )
            post_data[impact_lower_col] = (
                self.post_impact.quantile(lower_q, dim=["chain", "draw"])
                .to_series()
                .reindex(post_data.index)
                .values
            )
            post_data[impact_upper_col] = (
                self.post_impact.quantile(upper_q, dim=["chain", "draw"])
                .to_series()
                .reindex(post_data.index)
                .values
            )

        self.plot_data = pd.concat([pre_data, post_data])

        return self.plot_data

    def analyze_persistence(
        self,
        hdi_prob: float = HDI_PROB,
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
    ) -> dict[str, Any]:
        """Analyze effect persistence between intervention and post-intervention periods.

        Computes mean effects, persistence ratio, and total (cumulative) impacts for both periods.
        The persistence ratio is the post-intervention mean effect divided by the intervention
        mean effect (as a decimal, e.g., 0.30 means 30% persistence, 1.5 means 150%).
        Note: The ratio can exceed 1.0 if the post-intervention effect is larger than the
        intervention effect.

        Automatically prints a summary of the results.

        Parameters
        ----------
        hdi_prob : float
            Probability for the HDI interval (Bayesian models only). Defaults
            to :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
        direction : {"increase", "decrease", "two-sided"}, default="increase"
            Direction for tail probability calculation (Bayesian models only)

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            - "mean_effect_during": Mean effect during intervention period
            - "mean_effect_post": Mean effect during post-intervention period
            - "persistence_ratio": Post-intervention mean effect divided by intervention mean (decimal, can exceed 1.0)
            - "total_effect_during": Total (cumulative) effect during intervention period
            - "total_effect_post": Total (cumulative) effect during post-intervention period

        Raises
        ------
        ValueError
            If treatment_end_time is not provided (two-period design)

        Examples
        --------
        >>> import causalpy as cp
        >>> import pandas as pd
        >>> df = (
        ...     cp.load_data("its")
        ...     .assign(date=lambda x: pd.to_datetime(x["date"]))
        ...     .set_index("date")
        ... )
        >>> result = cp.InterruptedTimeSeries(
        ...     df,
        ...     treatment_time=pd.Timestamp("2017-01-01"),
        ...     treatment_end_time=pd.Timestamp("2017-06-01"),
        ...     formula="y ~ 1 + t + C(month)",
        ...     model=cp.pymc_models.LinearRegression(
        ...         sample_kwargs={"random_seed": 42, "progressbar": False}
        ...     ),
        ... )
        >>> persistence = result.analyze_persistence()  # doctest: +SKIP
        ... # Note: Results are automatically printed to console
        >>> persistence["persistence_ratio"]  # doctest: +SKIP
        -1.224
        """
        if self.treatment_end_time is None:
            raise ValueError(
                "analyze_persistence() requires treatment_end_time to be provided. "
                "This method is only available for three-period designs."
            )

        has_draws = has_posterior_draws(self.intervention_impact)
        time_dim = "obs_ind"

        if has_draws:
            # PyMC: Compute statistics using xarray operations
            from causalpy.reporting import _extract_hdi_bounds

            # Intervention period
            intervention_avg = self.intervention_impact.mean(dim=time_dim)
            intervention_mean = _as_scalar(intervention_avg.mean(dim=["chain", "draw"]))
            intervention_hdi = az.hdi(intervention_avg, hdi_prob=hdi_prob)
            intervention_lower, intervention_upper = _extract_hdi_bounds(
                intervention_hdi, hdi_prob
            )

            # Post-intervention period
            post_avg = self.post_intervention_impact.mean(dim=time_dim)
            post_mean = _as_scalar(post_avg.mean(dim=["chain", "draw"]))
            post_hdi = az.hdi(post_avg, hdi_prob=hdi_prob)
            post_lower, post_upper = _extract_hdi_bounds(post_hdi, hdi_prob)

            # Cumulative (total) impacts
            intervention_cum = self.intervention_impact_cumulative.isel({time_dim: -1})
            intervention_cum_mean = _as_scalar(
                intervention_cum.mean(dim=["chain", "draw"])
            )

            post_cum = self.post_intervention_impact_cumulative.isel({time_dim: -1})
            post_cum_mean = _as_scalar(post_cum.mean(dim=["chain", "draw"]))

            # Persistence ratio: post_mean / intervention_mean (as decimal, not percentage)
            epsilon = 1e-8
            persistence_ratio = post_mean / (intervention_mean + epsilon)

            result = {
                "mean_effect_during": intervention_mean,
                "mean_effect_post": post_mean,
                "persistence_ratio": float(persistence_ratio),
                "total_effect_during": intervention_cum_mean,
                "total_effect_post": post_cum_mean,
            }
            # Store HDI bounds for printing
            intervention_ci_lower = intervention_lower
            intervention_ci_upper = intervention_upper
            post_ci_lower = post_lower
            post_ci_upper = post_upper
        else:
            # OLS: Compute statistics using numpy operations
            from causalpy.reporting import _compute_statistics_ols

            # Compute statistics for intervention period
            intervention_stats = _compute_statistics_ols(
                np.asarray(self.intervention_impact).ravel(),
                np.asarray(self.intervention_pred).ravel(),
                alpha=1 - hdi_prob,
                cumulative=True,
                relative=False,
            )

            # Compute statistics for post-intervention period
            post_stats = _compute_statistics_ols(
                np.asarray(self.post_intervention_impact).ravel(),
                np.asarray(self.post_intervention_pred).ravel(),
                alpha=1 - hdi_prob,
                cumulative=True,
                relative=False,
            )

            # Persistence ratio (as decimal)
            epsilon = 1e-8
            persistence_ratio = post_stats["avg"]["mean"] / (
                intervention_stats["avg"]["mean"] + epsilon
            )

            result = {
                "mean_effect_during": intervention_stats["avg"]["mean"],
                "mean_effect_post": post_stats["avg"]["mean"],
                "persistence_ratio": float(persistence_ratio),
                "total_effect_during": intervention_stats["cum"]["mean"],
                "total_effect_post": post_stats["cum"]["mean"],
            }
            # Store CI bounds for printing
            intervention_ci_lower = intervention_stats["avg"]["ci_lower"]
            intervention_ci_upper = intervention_stats["avg"]["ci_upper"]
            post_ci_lower = post_stats["avg"]["ci_lower"]
            post_ci_upper = post_stats["avg"]["ci_upper"]

        # Print results
        hdi_pct = int(hdi_prob * 100)
        ci_label = "HDI" if has_draws else "CI"
        print("=" * 60)
        print("Effect Persistence Analysis")
        print("=" * 60)
        print("\nDuring intervention period:")
        print(f"  Mean effect: {result['mean_effect_during']:.2f}")
        print(
            f"  {hdi_pct}% {ci_label}: [{intervention_ci_lower:.2f}, {intervention_ci_upper:.2f}]"
        )
        print(f"  Total effect: {result['total_effect_during']:.2f}")
        print("\nPost-intervention period:")
        print(f"  Mean effect: {result['mean_effect_post']:.2f}")
        print(f"  {hdi_pct}% {ci_label}: [{post_ci_lower:.2f}, {post_ci_upper:.2f}]")
        print(f"  Total effect: {result['total_effect_post']:.2f}")
        print(f"\nPersistence ratio: {result['persistence_ratio']:.3f}")
        print(
            f"  ({result['persistence_ratio'] * 100:.1f}% of intervention effect persisted)"
        )
        print("=" * 60)

        return result

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
        Generate a decision-ready summary of causal effects for Interrupted Time Series.

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
            Ignored for Interrupted Time Series (single unit).
        period : {"intervention", "post", "comparison"}, optional
            For three-period designs (with treatment_end_time), specify which period to summarize.
            Defaults to None for standard behavior.
        prefix : str, optional
            Prefix for prose generation (e.g., "During intervention", "Post-intervention").
            Defaults to "Post-period".
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

        # Handle period parameter for three-period designs
        if period is not None:
            # Validate period parameter
            valid_periods = ["intervention", "post", "comparison"]
            if period not in valid_periods:
                raise ValueError(
                    f"period must be one of {valid_periods}, got '{period}'"
                )

            # Check if this experiment supports three-period designs
            if not (
                hasattr(self, "treatment_end_time")
                and self.treatment_end_time is not None
            ):
                raise ValueError(
                    f"Period '{period}' not available. This experiment may not support three-period designs. "
                    "Provide treatment_end_time to enable period-specific analysis."
                )

            if period == "comparison":
                # Comparison period: delegate to subclass method
                return self._comparison_period_summary(
                    direction=direction,
                    alpha=alpha,
                    cumulative=cumulative,
                    relative=relative,
                    min_effect=min_effect,
                )

            # For "intervention" or "post" periods, use _extract_window with tuple windows
            if period == "intervention":
                # Intervention period: treatment_time <= index < treatment_end_time
                intervention_indices = self.datapost.index[
                    self.datapost.index < self.treatment_end_time
                ]
                # Use the last index before treatment_end_time as the end bound
                window = (self.treatment_time, intervention_indices.max())
                prefix = "During intervention"
            elif period == "post":
                # Post-intervention period: index >= treatment_end_time (inclusive)
                window = (self.treatment_end_time, self.datapost.index.max())
                prefix = "Post-intervention"

            # Extract windowed impact data using calculated window
            windowed_impact, window_coords = _extract_window(
                self, window, treated_unit=treated_unit
            )

            # Extract counterfactual for relative effects
            counterfactual = _extract_counterfactual(
                self, window_coords, treated_unit=treated_unit
            )
        else:
            # No period specified, use standard flow
            windowed_impact, window_coords = _extract_window(
                self, window, treated_unit=treated_unit
            )
            counterfactual = _extract_counterfactual(
                self, window_coords, treated_unit=treated_unit
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
            experiment_type="its",
        )
