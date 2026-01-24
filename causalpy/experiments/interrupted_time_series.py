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
Interrupted Time Series Analysis
"""

from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from patsy import build_design_matrices, dmatrices
from sklearn.base import RegressorMixin

from causalpy.custom_exceptions import BadIndexException
from causalpy.date_utils import _combine_datetime_indices, format_date_axes
from causalpy.plot_utils import (
    ResponseType,
    _log_response_type_effect_summary_once,
    _log_response_type_info_once,
    add_hdi_annotation,
    get_hdi_to_df,
    plot_xY,
)
from causalpy.pymc_models import PyMCModel
from causalpy.reporting import EffectSummary
from causalpy.utils import round_num

from .base import BaseExperiment

LEGEND_FONT_SIZE = 12


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
        a DatetimeIndex or numeric (integer/float).
    treatment_time : Union[int, float, pd.Timestamp]
        The time when treatment occurred, should be in reference to the data index.
        Must match the index type (DatetimeIndex requires pd.Timestamp).
        **INCLUSIVE**: Observations at exactly ``treatment_time`` are included in the
        post-intervention period (uses ``>=`` comparison).
    formula : str
        A statistical model formula using patsy syntax (e.g., "y ~ 1 + t + C(month)").
    model : Union[PyMCModel, RegressorMixin], optional
        A PyMC (Bayesian) or sklearn (OLS) model. If None, defaults to a PyMC
        LinearRegression model.
    treatment_end_time : Union[int, float, pd.Timestamp], optional
        The time when treatment ended, enabling three-period analysis. Must be
        greater than ``treatment_time`` and within the data range. If None (default),
        the analysis assumes a permanent intervention (two-period design).
        **INCLUSIVE**: Observations at exactly ``treatment_end_time`` are included in the
        post-intervention period (uses ``>=`` comparison).
    **kwargs : dict
        Additional keyword arguments passed to the model.

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

    Notes
    -----
    For Bayesian models, the causal impact is calculated using the posterior expectation
    (``mu``) rather than the posterior predictive (``y_hat``). This means the impact and
    its uncertainty represent the systematic causal effect, excluding observation-level
    noise. The uncertainty bands in the plots reflect parameter uncertainty and
    counterfactual prediction uncertainty, but not individual observation variability.

    The three-period design is useful for analyzing temporary interventions such as:
    - Marketing campaigns with defined start and end dates
    - Policy trials or pilot programs
    - Clinical treatments with limited duration
    - Seasonal interventions

    Use ``effect_summary(period="intervention")`` to analyze effects during the
    intervention, and ``effect_summary(period="post")`` to analyze effect persistence
    after the intervention ends.
    """

    expt_type = "Interrupted Time Series"
    supports_ols = True
    supports_bayes = True

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_time: int | float | pd.Timestamp,
        formula: str,
        model: PyMCModel | RegressorMixin | None = None,
        treatment_end_time: int | float | pd.Timestamp | None = None,
        **kwargs: dict,
    ) -> None:
        super().__init__(model=model)
        self.pre_y: xr.DataArray
        self.post_y: xr.DataArray
        # rename the index to "obs_ind"
        data.index.name = "obs_ind"
        self.input_validation(data, treatment_time, treatment_end_time)
        self.treatment_time = treatment_time
        self.treatment_end_time = treatment_end_time
        # set experiment type - usually done in subclasses
        self.expt_type = "Pre-Post Fit"
        # split data in to pre and post intervention
        # NOTE: treatment_time is INCLUSIVE (>=) in post-period
        # Pre-period: index < treatment_time (exclusive)
        # Post-period: index >= treatment_time (inclusive)
        self.datapre = data[data.index < self.treatment_time]
        self.datapost = data[data.index >= self.treatment_time]

        self.formula = formula

        # set things up with pre-intervention data
        y, X = dmatrices(formula, self.datapre)
        self.outcome_variable_name = y.design_info.column_names[0]
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.pre_y, self.pre_X = np.asarray(y), np.asarray(X)
        # process post-intervention data
        (new_y, new_x) = build_design_matrices(
            [self._y_design_info, self._x_design_info], self.datapost
        )
        self.post_X = np.asarray(new_x)
        self.post_y = np.asarray(new_y)
        # turn into xarray.DataArray's
        self.pre_X = xr.DataArray(
            self.pre_X,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": self.datapre.index,
                "coeffs": self.labels,
            },
        )
        self.pre_y = xr.DataArray(
            self.pre_y,  # Keep 2D shape
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": self.datapre.index, "treated_units": ["unit_0"]},
        )
        self.post_X = xr.DataArray(
            self.post_X,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": self.datapost.index,
                "coeffs": self.labels,
            },
        )
        self.post_y = xr.DataArray(
            self.post_y,  # Keep 2D shape
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": self.datapost.index, "treated_units": ["unit_0"]},
        )

        # fit the model to the observed (pre-intervention) data
        # All PyMC models now accept xr.DataArray with consistent API
        if isinstance(self.model, PyMCModel):
            COORDS: dict[str, Any] = {
                "coeffs": self.labels,
                "obs_ind": np.arange(self.pre_X.shape[0]),
                "treated_units": ["unit_0"],
                "datetime_index": self.datapre.index,  # For time series models
            }
            self.model.fit(X=self.pre_X, y=self.pre_y, coords=COORDS)
        elif isinstance(self.model, RegressorMixin):
            # For OLS models, use 1D y data
            self.model.fit(X=self.pre_X, y=self.pre_y.isel(treated_units=0))
        else:
            raise ValueError("Model type not recognized")

        # score the goodness of fit to the pre-intervention data
        if isinstance(self.model, PyMCModel):
            self.score = self.model.score(X=self.pre_X, y=self.pre_y)
        elif isinstance(self.model, RegressorMixin):
            self.score = self.model.score(
                X=self.pre_X, y=self.pre_y.isel(treated_units=0)
            )

        # get the model predictions of the observed (pre-intervention) data
        if isinstance(self.model, (PyMCModel, RegressorMixin)):
            self.pre_pred = self.model.predict(X=self.pre_X)

        # calculate the counterfactual (post period)
        if isinstance(self.model, PyMCModel):
            self.post_pred = self.model.predict(X=self.post_X, out_of_sample=True)
        elif isinstance(self.model, RegressorMixin):
            self.post_pred = self.model.predict(X=self.post_X)

        # Calculate impact - all PyMC models now use 2D data with treated_units.
        # TODO: REFACTOR TARGET - Currently, stored impacts use the model expectation
        # (mu) by default. When users request response_type="prediction" in plot(), the
        # y_hat-based impact is calculated on-the-fly in _bayesian_plot(). This works
        # but is not ideal: consider storing both mu and y_hat based impacts, or
        # refactoring to always calculate on-demand. See calculate_impact() for details.
        if isinstance(self.model, PyMCModel):
            self.pre_impact = self.model.calculate_impact(self.pre_y, self.pre_pred)
            self.post_impact = self.model.calculate_impact(self.post_y, self.post_pred)
        elif isinstance(self.model, RegressorMixin):
            # SKL models work with 1D data
            self.pre_impact = self.model.calculate_impact(
                self.pre_y.isel(treated_units=0), self.pre_pred
            )
            self.post_impact = self.model.calculate_impact(
                self.post_y.isel(treated_units=0), self.post_pred
            )

        self.post_impact_cumulative = self.model.calculate_cumulative_impact(
            self.post_impact
        )

        # Split post period into intervention and post-intervention if treatment_end_time is provided
        if self.treatment_end_time is not None:
            self._split_post_period()

    def input_validation(
        self,
        data: pd.DataFrame,
        treatment_time: int | float | pd.Timestamp,
        treatment_end_time: int | float | pd.Timestamp | None = None,
    ) -> None:
        """Validate the input data and model formula for correctness"""
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

        # Split predictions and impacts
        # Handle both PyMC (xarray) and OLS (numpy) cases
        is_pymc = isinstance(self.model, PyMCModel)

        if is_pymc:
            # PyMC: use xarray selection
            # Dimension is always "obs_ind" in CausalPy
            time_dim = "obs_ind"

            # Get indices for selection
            intervention_coords = self.data_intervention.index
            post_intervention_coords = self.data_post_intervention.index

            # 3. Split post_pred into intervention_pred and post_intervention_pred
            # These are slices of post_pred, not new computations
            # For PyMC models, post_pred is guaranteed to be az.InferenceData
            # (regular PyMC models return it directly, BSTS-like models are wrapped in __init__)
            intervention_pred_dataset = self.post_pred.posterior_predictive.sel(
                {time_dim: intervention_coords}
            )
            post_intervention_pred_dataset = self.post_pred.posterior_predictive.sel(
                {time_dim: post_intervention_coords}
            )

            # Create new InferenceData objects with the sliced posterior_predictive
            # This maintains the same structure as post_pred
            self.intervention_pred = az.InferenceData(
                posterior_predictive=intervention_pred_dataset
            )
            self.post_intervention_pred = az.InferenceData(
                posterior_predictive=post_intervention_pred_dataset
            )

            # 4. Split post_impact into intervention_impact and post_intervention_impact
            # Similarly, these are slices of the existing post_impact calculation
            if "treated_units" in self.post_impact.dims:
                post_impact_sel = self.post_impact.isel(treated_units=0)
            else:
                post_impact_sel = self.post_impact
            self.intervention_impact = post_impact_sel.sel(
                {time_dim: intervention_coords}
            )
            self.post_intervention_impact = post_impact_sel.sel(
                {time_dim: post_intervention_coords}
            )

            # 5. Calculate cumulative impacts for each period using the sliced impacts
            self.intervention_impact_cumulative = (
                self.model.calculate_cumulative_impact(self.intervention_impact)
            )
            self.post_intervention_impact_cumulative = (
                self.model.calculate_cumulative_impact(self.post_intervention_impact)
            )
        else:
            # OLS: use numpy array indexing with position-based selection
            # For OLS models, post_pred is guaranteed to be numpy array
            intervention_indices = [
                self.datapost.index.get_loc(coord)
                for coord in self.data_intervention.index
            ]
            post_intervention_indices = [
                self.datapost.index.get_loc(coord)
                for coord in self.data_post_intervention.index
            ]

            # 3. Split post_pred (numpy array for OLS) - slices of post_pred
            self.intervention_pred = self.post_pred[intervention_indices]
            self.post_intervention_pred = self.post_pred[post_intervention_indices]

            # 4. Split post_impact (numpy array for OLS) - slices of post_impact
            self.intervention_impact = self.post_impact[intervention_indices]
            self.post_intervention_impact = self.post_impact[post_intervention_indices]

            # 5. Calculate cumulative impacts for each period using the sliced impacts
            self.intervention_impact_cumulative = (
                self.model.calculate_cumulative_impact(self.intervention_impact)
            )
            self.post_intervention_impact_cumulative = (
                self.model.calculate_cumulative_impact(self.post_intervention_impact)
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

        is_pymc = isinstance(self.model, PyMCModel)
        time_dim = "obs_ind"
        hdi_prob = 1 - alpha
        prob_persisted: float | None

        if is_pymc:
            # PyMC: Compute statistics for both periods
            intervention_avg = self.intervention_impact.mean(dim=time_dim)
            intervention_mean = float(
                intervention_avg.mean(dim=["chain", "draw"]).values
            )
            intervention_hdi = az.hdi(intervention_avg, hdi_prob=hdi_prob)
            intervention_lower, intervention_upper = _extract_hdi_bounds(
                intervention_hdi, hdi_prob
            )

            post_avg = self.post_intervention_impact.mean(dim=time_dim)
            post_mean = float(post_avg.mean(dim=["chain", "draw"]).values)
            post_hdi = az.hdi(post_avg, hdi_prob=hdi_prob)
            post_lower, post_upper = _extract_hdi_bounds(post_hdi, hdi_prob)

            # Persistence ratio: post_mean / intervention_mean (as percentage)
            epsilon = 1e-8
            persistence_ratio_pct = (post_mean / (intervention_mean + epsilon)) * 100

            # Probability that some effect persisted (P(post_mean > 0))
            prob_persisted = float((post_avg > 0).mean().values)

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
                self.intervention_impact.values
                if hasattr(self.intervention_impact, "values")
                else np.asarray(self.intervention_impact),
                self.intervention_pred,
                alpha=alpha,
                cumulative=False,
                relative=False,
            )

            post_stats = _compute_statistics_ols(
                self.post_intervention_impact.values
                if hasattr(self.post_intervention_impact, "values")
                else np.asarray(self.post_intervention_impact),
                self.post_intervention_pred,
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

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers
        """
        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        self.print_coefficients(round_to)

    def _bayesian_plot(
        self,
        round_to: int | None = 2,
        response_type: ResponseType = "expectation",
        show_hdi_annotation: bool = False,
        **kwargs: dict,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plot the results

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2.
            Use None to return raw numbers.
        response_type : {"expectation", "prediction"}, default="expectation"
            The response type to display in the HDI band:

            - ``"expectation"``: HDI of the model expectation (μ). This shows
              uncertainty from model parameters only, excluding observation noise.
              Results in narrower intervals that represent the uncertainty in
              the expected value of the outcome.
            - ``"prediction"``: HDI of the posterior predictive (ŷ). This includes
              observation noise (σ) in addition to parameter uncertainty, resulting
              in wider intervals that represent the full predictive uncertainty
              for new observations.
        show_hdi_annotation : bool, default=False
            Whether to display a text annotation at the bottom of the figure
            explaining what the HDI represents. Set to False to hide the annotation.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple[plt.Figure, list[plt.Axes]]
            The matplotlib figure and axes.
        """
        # Log HDI type info once per session
        _log_response_type_info_once()

        counterfactual_label = "Counterfactual"

        # Select the variable name based on response_type
        var_name = "mu" if response_type == "expectation" else "y_hat"

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))
        # TOP PLOT --------------------------------------------------
        # pre-intervention period
        pre_pred_var = self.pre_pred["posterior_predictive"][var_name]
        pre_pred_plot = (
            pre_pred_var.isel(treated_units=0)
            if "treated_units" in pre_pred_var.dims
            else pre_pred_var
        )
        h_line, h_patch = plot_xY(
            self.datapre.index,
            pre_pred_plot,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C0"},
        )
        handles = [(h_line, h_patch)]
        labels = ["Pre-intervention period"]

        (h,) = ax[0].plot(
            self.datapre.index,
            self.pre_y.isel(treated_units=0)
            if hasattr(self.pre_y, "isel")
            else self.pre_y[:, 0],
            "k.",
            label="Observations",
        )
        handles.append(h)
        labels.append("Observations")

        # post intervention period
        post_pred_var = self.post_pred["posterior_predictive"][var_name]
        post_pred_plot = (
            post_pred_var.isel(treated_units=0)
            if "treated_units" in post_pred_var.dims
            else post_pred_var
        )
        h_line, h_patch = plot_xY(
            self.datapost.index,
            post_pred_plot,
            ax=ax[0],
            plot_hdi_kwargs={"color": "C1"},
        )
        handles.append((h_line, h_patch))
        labels.append(counterfactual_label)

        ax[0].plot(
            self.datapost.index,
            self.post_y.isel(treated_units=0)
            if hasattr(self.post_y, "isel")
            else self.post_y[:, 0],
            "k.",
        )
        # Shaded causal effect - always use mu for the fill_between mean line
        post_pred_mu = az.extract(
            self.post_pred, group="posterior_predictive", var_names="mu"
        )
        if "treated_units" in post_pred_mu.dims:
            post_pred_mu = post_pred_mu.isel(treated_units=0)
        post_pred_mu = post_pred_mu.mean("sample")
        h = ax[0].fill_between(
            self.datapost.index,
            y1=post_pred_mu,
            y2=self.post_y.isel(treated_units=0)
            if hasattr(self.post_y, "isel")
            else self.post_y[:, 0],
            color="C0",
            alpha=0.25,
        )
        handles.append(h)
        labels.append("Causal impact")

        # Title with R^2, supporting both unit_0_r2 and r2 keys
        r2_val = None
        r2_std_val = None
        try:
            if isinstance(self.score, pd.Series):
                if "unit_0_r2" in self.score.index:
                    r2_val = self.score["unit_0_r2"]
                    r2_std_val = self.score.get("unit_0_r2_std", None)
                elif "r2" in self.score.index:
                    r2_val = self.score["r2"]
                    r2_std_val = self.score.get("r2_std", None)
        except Exception:
            pass
        title_str = "Pre-intervention Bayesian $R^2$"
        if r2_val is not None:
            title_str += f": {round_num(r2_val, round_to)}"
            if r2_std_val is not None:
                title_str += f"\n(std = {round_num(r2_std_val, round_to)})"
        ax[0].set(title=title_str)

        # MIDDLE PLOT -----------------------------------------------
        # Calculate impact for plotting based on response_type
        if response_type == "expectation":
            # Use stored mu-based impact
            pre_impact_for_plot = self.pre_impact
            post_impact_for_plot = self.post_impact
        else:
            # Calculate y_hat-based impact on demand
            pre_impact_for_plot = self.model.calculate_impact(
                self.pre_y, self.pre_pred, response_type="prediction"
            )
            post_impact_for_plot = self.model.calculate_impact(
                self.post_y, self.post_pred, response_type="prediction"
            )

        pre_impact_plot = (
            pre_impact_for_plot.isel(treated_units=0)
            if hasattr(pre_impact_for_plot, "dims")
            and "treated_units" in pre_impact_for_plot.dims
            else pre_impact_for_plot
        )
        plot_xY(
            self.datapre.index,
            pre_impact_plot,
            ax=ax[1],
            plot_hdi_kwargs={"color": "C0"},
        )
        post_impact_plot = (
            post_impact_for_plot.isel(treated_units=0)
            if hasattr(post_impact_for_plot, "dims")
            and "treated_units" in post_impact_for_plot.dims
            else post_impact_for_plot
        )
        plot_xY(
            self.datapost.index,
            post_impact_plot,
            ax=ax[1],
            plot_hdi_kwargs={"color": "C1"},
        )
        ax[1].axhline(y=0, c="k")
        post_impact_mean = (
            post_impact_for_plot.mean(["chain", "draw"])
            if hasattr(post_impact_for_plot, "mean")
            else post_impact_for_plot
        )
        if (
            hasattr(post_impact_mean, "dims")
            and "treated_units" in post_impact_mean.dims
        ):
            post_impact_mean = post_impact_mean.isel(treated_units=0)
        ax[1].fill_between(
            self.datapost.index,
            y1=post_impact_mean,
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[1].set(title="Causal Impact")

        # BOTTOM PLOT -----------------------------------------------
        ax[2].set(title="Cumulative Causal Impact")
        # Calculate cumulative impact based on response_type
        if response_type == "expectation":
            post_impact_cumulative_for_plot = self.post_impact_cumulative
        else:
            post_impact_cumulative_for_plot = self.model.calculate_cumulative_impact(
                post_impact_for_plot
            )

        post_cum_plot = (
            post_impact_cumulative_for_plot.isel(treated_units=0)
            if hasattr(post_impact_cumulative_for_plot, "dims")
            and "treated_units" in post_impact_cumulative_for_plot.dims
            else post_impact_cumulative_for_plot
        )
        plot_xY(
            self.datapost.index,
            post_cum_plot,
            ax=ax[2],
            plot_hdi_kwargs={"color": "C1"},
        )
        ax[2].axhline(y=0, c="k")

        # Intervention lines
        for i in [0, 1, 2]:
            ax[i].axvline(
                x=self.treatment_time,
                ls="-",
                lw=3,
                color="r",
                label="Treatment start" if i == 0 else None,
            )
            if self.treatment_end_time is not None:
                ax[i].axvline(
                    x=self.treatment_end_time,
                    ls="--",
                    lw=2,
                    color="orange",
                    label="Treatment end" if i == 0 else None,
                )

        ax[0].legend(
            handles=(h_tuple for h_tuple in handles),
            labels=labels,
            fontsize=LEGEND_FONT_SIZE,
        )

        # Apply intelligent date formatting if data has datetime index
        if isinstance(self.datapre.index, pd.DatetimeIndex):
            # Combine pre and post indices for full date range
            full_index = _combine_datetime_indices(
                pd.DatetimeIndex(self.datapre.index),
                pd.DatetimeIndex(self.datapost.index),
            )
            format_date_axes(ax, full_index)

        # Add HDI type annotation to the top subplot's title
        if show_hdi_annotation:
            add_hdi_annotation(ax[0], response_type)

        return fig, ax

    def _ols_plot(
        self, round_to: int | None = 2, **kwargs: dict
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plot the results

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """
        counterfactual_label = "Counterfactual"

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

        ax[0].plot(self.datapre.index, self.pre_y, "k.")
        ax[0].plot(self.datapre.index, self.pre_pred, c="k", label="model fit")

        ax[0].plot(self.datapost.index, self.post_y, "k.")
        ax[0].plot(
            self.datapost.index,
            self.post_pred,
            label=counterfactual_label,
            ls=":",
            c="k",
        )
        # Shaded causal effect
        ax[0].fill_between(
            self.datapost.index,
            y1=np.squeeze(self.post_pred),
            y2=np.squeeze(self.post_y),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )

        ax[0].set(
            title=f"$R^2$ on pre-intervention data = {round_num(float(self.score), round_to)}"
        )

        ax[1].plot(self.datapre.index, self.pre_impact, "k.")
        ax[1].plot(
            self.datapost.index,
            self.post_impact,
            "k.",
            label=counterfactual_label,
        )
        ax[1].axhline(y=0, c="k")
        # Shaded causal effect
        ax[1].fill_between(
            self.datapost.index,
            y1=np.squeeze(self.post_impact),
            color="C0",
            alpha=0.25,
            label="Causal impact",
        )
        ax[1].set(title="Causal Impact")

        ax[2].plot(self.datapost.index, self.post_impact_cumulative, c="k")
        ax[2].axhline(y=0, c="k")
        ax[2].set(title="Cumulative Causal Impact")

        # Intervention lines
        for i in [0, 1, 2]:
            ax[i].axvline(
                x=self.treatment_time,
                ls="-",
                lw=3,
                color="r",
                label="Treatment start" if i == 0 else None,
            )
            if self.treatment_end_time is not None:
                ax[i].axvline(
                    x=self.treatment_end_time,
                    ls="--",
                    lw=2,
                    color="orange",
                    label="Treatment end" if i == 0 else None,
                )

        ax[0].legend(fontsize=LEGEND_FONT_SIZE)

        # Apply intelligent date formatting if data has datetime index
        if isinstance(self.datapre.index, pd.DatetimeIndex):
            # Combine pre and post indices for full date range
            full_index = _combine_datetime_indices(
                pd.DatetimeIndex(self.datapre.index),
                pd.DatetimeIndex(self.datapost.index),
            )
            format_date_axes(ax, full_index)

        return (fig, ax)

    def get_plot_data_bayesian(
        self,
        hdi_prob: float = 0.94,
        response_type: ResponseType = "expectation",
    ) -> pd.DataFrame:
        """
        Recover the data of the experiment along with the prediction and causal impact information.

        Parameters
        ----------
        hdi_prob : float, default=0.94
            Probability for which the highest density interval will be computed.
            The default value is defined as the default from the :func:`arviz.hdi` function.
        response_type : {"expectation", "prediction"}, default="expectation"
            The response type to use for predictions and impact:

            - ``"expectation"``: Uses the model expectation (μ). Excludes observation
              noise, focusing on the systematic causal effect.
            - ``"prediction"``: Uses the full posterior predictive (ŷ). Includes
              observation noise, showing the full predictive uncertainty.
        """

        if isinstance(self.model, PyMCModel):
            # Map semantic response_type to internal variable name
            var_name = "mu" if response_type == "expectation" else "y_hat"
            hdi_pct = int(round(hdi_prob * 100))

            pred_lower_col = f"pred_hdi_lower_{hdi_pct}"
            pred_upper_col = f"pred_hdi_upper_{hdi_pct}"
            impact_lower_col = f"impact_hdi_lower_{hdi_pct}"
            impact_upper_col = f"impact_hdi_upper_{hdi_pct}"

            pre_data = self.datapre.copy()
            post_data = self.datapost.copy()

            pre_mu = az.extract(
                self.pre_pred, group="posterior_predictive", var_names=var_name
            )
            post_mu = az.extract(
                self.post_pred, group="posterior_predictive", var_names=var_name
            )
            if "treated_units" in pre_mu.dims:
                pre_mu = pre_mu.isel(treated_units=0)
            if "treated_units" in post_mu.dims:
                post_mu = post_mu.isel(treated_units=0)
            pre_data["prediction"] = pre_mu.mean("sample").values
            post_data["prediction"] = post_mu.mean("sample").values

            hdi_pre_pred = get_hdi_to_df(
                self.pre_pred["posterior_predictive"][var_name], hdi_prob=hdi_prob
            )
            hdi_post_pred = get_hdi_to_df(
                self.post_pred["posterior_predictive"][var_name], hdi_prob=hdi_prob
            )
            # If treated_units present, select unit_0; otherwise use directly
            if (
                isinstance(hdi_pre_pred.index, pd.MultiIndex)
                and "treated_units" in hdi_pre_pred.index.names
            ):
                pre_data[[pred_lower_col, pred_upper_col]] = hdi_pre_pred.xs(
                    "unit_0", level="treated_units"
                ).set_index(pre_data.index)
                post_data[[pred_lower_col, pred_upper_col]] = hdi_post_pred.xs(
                    "unit_0", level="treated_units"
                ).set_index(post_data.index)
            else:
                pre_data[[pred_lower_col, pred_upper_col]] = hdi_pre_pred.set_index(
                    pre_data.index
                )
                post_data[[pred_lower_col, pred_upper_col]] = hdi_post_pred.set_index(
                    post_data.index
                )

            # Select impact based on response_type
            if response_type == "expectation":
                pre_impact = self.pre_impact
                post_impact = self.post_impact
            else:
                # Calculate y_hat-based impact on demand
                pre_impact = self.model.calculate_impact(
                    self.pre_y, self.pre_pred, response_type="prediction"
                )
                post_impact = self.model.calculate_impact(
                    self.post_y, self.post_pred, response_type="prediction"
                )

            pre_impact_mean = (
                pre_impact.mean(dim=["chain", "draw"])
                if hasattr(pre_impact, "mean")
                else pre_impact
            )
            post_impact_mean = (
                post_impact.mean(dim=["chain", "draw"])
                if hasattr(post_impact, "mean")
                else post_impact
            )
            if (
                hasattr(pre_impact_mean, "dims")
                and "treated_units" in pre_impact_mean.dims
            ):
                pre_impact_mean = pre_impact_mean.isel(treated_units=0)
            if (
                hasattr(post_impact_mean, "dims")
                and "treated_units" in post_impact_mean.dims
            ):
                post_impact_mean = post_impact_mean.isel(treated_units=0)
            pre_data["impact"] = pre_impact_mean.values
            post_data["impact"] = post_impact_mean.values

            # Compute impact HDIs directly via quantiles over posterior dims to avoid column shape issues
            alpha = 1 - hdi_prob
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2

            pre_lower_da = pre_impact.quantile(lower_q, dim=["chain", "draw"])
            pre_upper_da = pre_impact.quantile(upper_q, dim=["chain", "draw"])
            post_lower_da = post_impact.quantile(lower_q, dim=["chain", "draw"])
            post_upper_da = post_impact.quantile(upper_q, dim=["chain", "draw"])

            # If a treated_units dim remains for some models, select unit_0
            if hasattr(pre_lower_da, "dims") and "treated_units" in pre_lower_da.dims:
                pre_lower_da = pre_lower_da.sel(treated_units="unit_0")
                pre_upper_da = pre_upper_da.sel(treated_units="unit_0")
            if hasattr(post_lower_da, "dims") and "treated_units" in post_lower_da.dims:
                post_lower_da = post_lower_da.sel(treated_units="unit_0")
                post_upper_da = post_upper_da.sel(treated_units="unit_0")

            pre_data[impact_lower_col] = (
                pre_lower_da.to_series().reindex(pre_data.index).values
            )
            pre_data[impact_upper_col] = (
                pre_upper_da.to_series().reindex(pre_data.index).values
            )
            post_data[impact_lower_col] = (
                post_lower_da.to_series().reindex(post_data.index).values
            )
            post_data[impact_upper_col] = (
                post_upper_da.to_series().reindex(post_data.index).values
            )

            self.plot_data = pd.concat([pre_data, post_data])

            return self.plot_data
        else:
            raise ValueError("Unsupported model type")

    def get_plot_data_ols(self) -> pd.DataFrame:
        """
        Recover the data of the experiment along with the prediction and causal impact information.
        """
        pre_data = self.datapre.copy()
        post_data = self.datapost.copy()
        pre_data["prediction"] = self.pre_pred
        post_data["prediction"] = self.post_pred
        pre_data["impact"] = self.pre_impact
        post_data["impact"] = self.post_impact
        self.plot_data = pd.concat([pre_data, post_data])

        return self.plot_data

    def analyze_persistence(
        self,
        hdi_prob: float = 0.95,
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        response_type: ResponseType = "expectation",
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
        hdi_prob : float, default=0.95
            Probability for HDI interval (Bayesian models only)
        direction : {"increase", "decrease", "two-sided"}, default="increase"
            Direction for tail probability calculation (Bayesian models only)
        response_type : {"expectation", "prediction"}, default="expectation"
            The response type to use for effect analysis (Bayesian models only):

            - ``"expectation"``: Uses the model expectation (μ). Excludes observation
              noise, focusing on the systematic causal effect.
            - ``"prediction"``: Uses the full posterior predictive (ŷ). Includes
              observation noise, showing the full predictive uncertainty.

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

        is_pymc = isinstance(self.model, PyMCModel)
        time_dim = "obs_ind"

        if is_pymc:
            # PyMC: Compute statistics using xarray operations
            from causalpy.reporting import _extract_hdi_bounds

            # Select impact based on response_type
            if response_type == "expectation":
                intervention_impact = self.intervention_impact
                post_intervention_impact = self.post_intervention_impact
                intervention_impact_cumulative = self.intervention_impact_cumulative
                post_intervention_impact_cumulative = (
                    self.post_intervention_impact_cumulative
                )
            else:
                # Calculate y_hat-based impact on demand
                # Get y values for intervention and post-intervention periods from post_y
                intervention_y = self.post_y.sel(obs_ind=self.data_intervention.index)
                post_intervention_y = self.post_y.sel(
                    obs_ind=self.data_post_intervention.index
                )
                intervention_impact = self.model.calculate_impact(
                    intervention_y, self.intervention_pred, response_type="prediction"
                )
                post_intervention_impact = self.model.calculate_impact(
                    post_intervention_y,
                    self.post_intervention_pred,
                    response_type="prediction",
                )
                intervention_impact_cumulative = self.model.calculate_cumulative_impact(
                    intervention_impact
                )
                post_intervention_impact_cumulative = (
                    self.model.calculate_cumulative_impact(post_intervention_impact)
                )

            # Intervention period
            intervention_avg = intervention_impact.mean(dim=time_dim)
            intervention_mean = float(
                intervention_avg.mean(dim=["chain", "draw"]).values
            )
            intervention_hdi = az.hdi(intervention_avg, hdi_prob=hdi_prob)
            intervention_lower, intervention_upper = _extract_hdi_bounds(
                intervention_hdi, hdi_prob
            )

            # Post-intervention period
            post_avg = post_intervention_impact.mean(dim=time_dim)
            post_mean = float(post_avg.mean(dim=["chain", "draw"]).values)
            post_hdi = az.hdi(post_avg, hdi_prob=hdi_prob)
            post_lower, post_upper = _extract_hdi_bounds(post_hdi, hdi_prob)

            # Cumulative (total) impacts
            intervention_cum = intervention_impact_cumulative.isel({time_dim: -1})
            intervention_cum_mean = float(
                intervention_cum.mean(dim=["chain", "draw"]).values
            )

            post_cum = post_intervention_impact_cumulative.isel({time_dim: -1})
            post_cum_mean = float(post_cum.mean(dim=["chain", "draw"]).values)

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

            # Get counterfactual predictions for each period
            intervention_counterfactual = self.intervention_pred
            post_counterfactual = self.post_intervention_pred

            # Compute statistics for intervention period
            intervention_stats = _compute_statistics_ols(
                self.intervention_impact.values
                if hasattr(self.intervention_impact, "values")
                else np.asarray(self.intervention_impact),
                intervention_counterfactual,
                alpha=1 - hdi_prob,
                cumulative=True,
                relative=False,
            )

            # Compute statistics for post-intervention period
            post_stats = _compute_statistics_ols(
                self.post_intervention_impact.values
                if hasattr(self.post_intervention_impact, "values")
                else np.asarray(self.post_intervention_impact),
                post_counterfactual,
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
        ci_label = "HDI" if is_pymc else "CI"
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
        response_type: ResponseType = "expectation",
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
        response_type : {"expectation", "prediction"}, default="expectation"
            Response type to compute effect sizes:

            - ``"expectation"``: Effect size HDI based on model expectation (μ).
              Excludes observation noise, focusing on the systematic causal effect.
            - ``"prediction"``: Effect size HDI based on posterior predictive (ŷ).
              Includes observation noise, showing full predictive uncertainty.

        Returns
        -------
        EffectSummary
            Object with .table (DataFrame) and .text (str) attributes
        """
        from causalpy.reporting import (
            _compute_statistics,
            _compute_statistics_ols,
            _extract_counterfactual,
            _extract_window,
            _generate_prose,
            _generate_prose_ols,
            _generate_table,
            _generate_table_ols,
        )

        # Log HDI type info once per session (for PyMC models only)
        is_pymc = isinstance(self.model, PyMCModel)
        if is_pymc:
            _log_response_type_effect_summary_once()

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
                self, window, treated_unit=treated_unit, response_type=response_type
            )

            # Extract counterfactual for relative effects
            counterfactual = _extract_counterfactual(
                self,
                window_coords,
                treated_unit=treated_unit,
                response_type=response_type,
            )
        else:
            # No period specified, use standard flow
            windowed_impact, window_coords = _extract_window(
                self, window, treated_unit=treated_unit, response_type=response_type
            )
            counterfactual = _extract_counterfactual(
                self,
                window_coords,
                treated_unit=treated_unit,
                response_type=response_type,
            )

        if is_pymc:
            # PyMC model: use posterior draws
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

            # Generate table
            table = _generate_table(stats, cumulative=cumulative, relative=relative)

            # Generate prose
            text = _generate_prose(
                stats,
                window_coords,
                alpha=alpha,
                direction=direction,
                cumulative=cumulative,
                relative=relative,
                prefix=prefix,
            )
        else:
            # OLS model: use point estimates and CIs
            # Convert to numpy arrays if needed
            if hasattr(windowed_impact, "values"):
                impact_array = windowed_impact.values
            else:
                impact_array = np.asarray(windowed_impact)
            if hasattr(counterfactual, "values"):
                counterfactual_array = counterfactual.values
            else:
                counterfactual_array = np.asarray(counterfactual)

            stats = _compute_statistics_ols(
                impact_array,
                counterfactual_array,
                alpha=alpha,
                cumulative=cumulative,
                relative=relative,
            )

            # Generate table
            table = _generate_table_ols(stats, cumulative=cumulative, relative=relative)

            # Generate prose
            text = _generate_prose_ols(
                stats,
                window_coords,
                alpha=alpha,
                cumulative=cumulative,
                relative=relative,
                prefix=prefix,
            )

        return EffectSummary(table=table, text=text)
