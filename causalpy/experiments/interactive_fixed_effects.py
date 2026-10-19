#   Copyright 2026 - 2026 The PyMC Labs Developers
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
"""Bayesian interactive fixed-effects synthetic control.

The untreated path is Xu's linear factor mean: an additive unit intercept,
an additive time intercept, and a loading-factor product. Rank is not
selected. A multiplicative gamma process shrinks unused factor scales toward
zero, and posterior intervals marginalize that prior.

Shared factors and shared time intercepts are updated from the control panel
only. The control likelihood uses the one shared positive residual scale, not
a unit observation scale. That is one random variable. Treated pre-period
outcomes may update that scale's draws after the factor and time-intercept
sample, and must not resample those draws. Loadings use the updated draws.
The factor posterior is the control-panel posterior under that scale as
informed by the control panel. It is not recomputed under the treated-pre
update. Treated post-intervention outcomes never enter the likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from matplotlib import pyplot as plt

from causalpy.constants import HDI_PROB
from causalpy.date_utils import validate_treatment_time_against_index
from causalpy.experiments._panel_counterfactual import WidePanel, resolve_treated_unit
from causalpy.experiments._results import CausalResult
from causalpy.plot_utils import _PosteriorPlotStyle, format_r2_score
from causalpy.utils import _bayesian_r2_score

_DRAW_DIMS = ("chain", "draw")
_FACTORS = "factors"
_DEFAULT_MAX_FACTORS = 4
_MGP_A1 = 2.0
_MGP_A2 = 3.0
_RESIDUAL_CAUCHY_BETA = 2.5
_SCALE_UPDATE_STEPS = 12
_SCALE_FD_STEP = 1e-4
_INTERCEPT_SD = 5.0


@dataclass(frozen=True)
class InteractiveFixedEffectsResult(CausalResult):
    """Result bundle for interactive fixed-effects synthetic control.

    ``predictions_*`` are the untreated counterfactual. Posterior intervals on
    those fields marginalize the shrinkage prior. They are not conditional on
    a selected rank. ``score`` is the pre-period prediction error, fit to
    pre-period treated outcomes only.

    ``shrinkage_state`` holds posterior draws of one positive scale per factor,
    up to ``max_factors``. It is not a selected integer rank and not a derived
    effective-rank count.
    """

    shrinkage_state: xr.DataArray = field(kw_only=True)


class InteractiveFixedEffectsSyntheticControl:
    """Interactive fixed-effects synthetic control on a :class:`WidePanel`.

    Not a :class:`~causalpy.experiments.synthetic_control.SyntheticControl`
    subclass. Several treated units share one ``treatment_time``. Control units
    are donors. There is no ``n_factors`` argument: ``max_factors`` is an upper
    bound, not a selected rank.

    ``build`` constructs the control-panel graph. Its likelihood uses the one
    shared positive residual scale, not a unit observation scale. ``fit``
    samples the control-panel posterior of the shared factors, shared time
    intercepts, per-factor shrinkage scales, and that one scale. Treated
    pre-period outcomes may update the scale's draws after that sample. They
    must not resample the factor or time-intercept draws. Loadings use the
    updated draws.

    The untreated counterfactual is not a donor-row prediction. Raw factors and
    raw loadings are not identifiers: a rotation or a sign flip leaves the
    product unchanged. This sampler does not impose orthonormality.

    Parameters
    ----------
    data : pandas.DataFrame
        Balanced wide panel indexed by time. Used columns must have no
        missing values.
    treatment_time : int, float, or pandas.Timestamp
        Shared first post-period time. Rows at this time are post-period.
    control_units : sequence of str
        Donor columns.
    treated_units : sequence of str
        Treated columns. One or more.
    max_factors : int, default 4
        Upper bound on the number of factors. Not a selected rank.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_time: int | float | pd.Timestamp,
        control_units: list[str] | tuple[str, ...],
        treated_units: list[str] | tuple[str, ...],
        *,
        max_factors: int = _DEFAULT_MAX_FACTORS,
    ) -> None:
        if isinstance(max_factors, bool) or not isinstance(max_factors, int):
            raise TypeError("max_factors must be an int, not a selected rank.")
        if max_factors < 1:
            raise ValueError("max_factors must be at least 1.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        validate_treatment_time_against_index(data.index, treatment_time)
        control = tuple(control_units)
        treated = tuple(treated_units)
        if not control or not treated:
            raise ValueError(
                "v1 requires one or more treated units and control units as donors."
            )
        overlap = set(control) & set(treated)
        if overlap:
            raise ValueError(
                f"control_units and treated_units overlap: {sorted(overlap)}."
            )
        missing_columns = [
            name for name in control + treated if name not in data.columns
        ]
        if missing_columns:
            raise KeyError(f"panel is missing columns: {missing_columns}.")
        used = data.loc[:, list(control + treated)]
        if used.isna().any().any():
            raise ValueError(
                "missing values in the used columns are out of bounds for v1."
            )
        panel = WidePanel.from_frame(data, treatment_time, control, treated)
        if panel.pre.empty or panel.post.empty:
            raise ValueError("the panel must have both a pre-period and a post-period.")
        self.panel = panel
        self.max_factors = max_factors
        self._control_model: pm.Model | None = None
        self.result: InteractiveFixedEffectsResult | None = None

    @property
    def control_units(self) -> tuple[str, ...]:
        """Donor names, in the order supplied at construction."""
        return self.panel.control_units

    @property
    def treated_units(self) -> tuple[str, ...]:
        """Treated names, in the order supplied at construction."""
        return self.panel.treated_units

    def build(self) -> Self:
        """Construct the control-panel graph without sampling.

        Idempotent. The graph's observed outcomes are control units at every
        time. Treated outcomes, including pre-period outcomes, are not in this
        graph: they must not update factors or time intercepts.
        """
        if self._control_model is None:
            self._control_model = _control_model(
                self._control_outcomes(), self.max_factors
            )
        return self

    def fit(
        self,
        *,
        draws: int = 40,
        tune: int = 80,
        chains: int = 1,
        cores: int = 1,
        random_seed: int | None = None,
        progressbar: bool = False,
        **sample_kwargs: Any,
    ) -> Self:
        """Sample the joint posterior and populate :attr:`result`.

        Keyword arguments are forwarded to the control-panel :func:`pymc.sample`
        call. The treated conditional sample is aligned to those draws: one
        conditional draw per control-panel draw. Before that sample, the one
        shared residual scale's draws are updated from treated pre-period
        outcomes. The conditional sample uses those updated draws as the
        loading likelihood scale. It does not resample the scale, the factors,
        or the time intercepts, and it does not take caller-supplied factor
        draws.

        Treated post-intervention outcomes are read only when impacts are
        aligned, after sampling. Changing them does not change the untreated
        trajectory.

        Parameters
        ----------
        draws : int, default 40
            Control-panel posterior draws per chain.
        tune : int, default 80
            Control-panel tuning steps.
        chains : int, default 1
            Control-panel chains.
        cores : int, default 1
            Cores for the control-panel sample.
        random_seed : int, optional
            Seed for the control-panel sample. The conditional sample uses the
            next integer when this is set.
        progressbar : bool, default False
            Whether PyMC shows a progress bar.
        **sample_kwargs
            Extra arguments for the control-panel :func:`pymc.sample` call.

        Returns
        -------
        InteractiveFixedEffectsSyntheticControl
            The fitted experiment.
        """
        self.build()
        assert self._control_model is not None
        control_idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            progressbar=progressbar,
            model=self._control_model,
            initvals=_control_initvals(self._control_model, self._control_outcomes()),
            **sample_kwargs,
        )
        conditional = _sample_treated_conditional(
            self._control_outcomes(),
            self.panel.treated_pre,
            _as_dataset(control_idata),
            random_seed=None if random_seed is None else random_seed + 1,
            progressbar=progressbar,
        )
        posterior = _as_dataset(control_idata)
        self.result = self.compose_result(
            unit_intercept=conditional["unit_intercept"],
            time_intercept=posterior["time_intercept"],
            loadings=conditional["loadings"],
            factors=posterior["shared_factors"],
            shrinkage_state=posterior["shrinkage_state"],
        )
        return self

    def compose_result(
        self,
        *,
        unit_intercept: xr.DataArray,
        time_intercept: xr.DataArray,
        loadings: xr.DataArray,
        factors: xr.DataArray,
        shrinkage_state: xr.DataArray,
    ) -> InteractiveFixedEffectsResult:
        """Compose the frozen result from component draws.

        The untreated counterfactual at each time is the additive unit
        intercept plus the additive time intercept plus the loading-factor
        product. Impact uses :meth:`WidePanel.impacts`. ``score`` uses
        pre-period treated outcomes only. This method does not update
        shrinkage from treated post-intervention outcomes and does not select
        a rank.

        Parameters
        ----------
        unit_intercept : xarray.DataArray
            Draws with dimensions ``chain``, ``draw``, and ``treated_units``.
        time_intercept : xarray.DataArray
            Draws with dimensions ``chain``, ``draw``, and ``obs_ind``.
        loadings : xarray.DataArray
            Per-treated-unit loadings. The ``factors`` axis has length
            ``max_factors``.
        factors : xarray.DataArray
            Shared factors with dimensions ``chain``, ``draw``, ``obs_ind``,
            and ``factors``.
        shrinkage_state : xarray.DataArray
            One positive scale per factor, dimensions ``chain``, ``draw``,
            and ``factors``.

        Returns
        -------
        InteractiveFixedEffectsResult
            Predictions, impacts, pre-period score, and shrinkage state.
        """
        self._require_shrinkage_state(shrinkage_state)
        self._require_factor_axis(loadings, "loadings")
        self._require_factor_axis(factors, "factors")
        counterfactual = untreated_counterfactual(
            unit_intercept, time_intercept, loadings, factors
        )
        predictions_pre = counterfactual.sel(obs_ind=self.panel.pre.index)
        predictions_post = counterfactual.sel(obs_ind=self.panel.post.index)
        impact_pre, impact_post, impact_post_cumulative = self.panel.impacts(
            predictions_pre, predictions_post
        )
        return InteractiveFixedEffectsResult(
            predictions_pre=predictions_pre,
            predictions_post=predictions_post,
            impact_pre=impact_pre,
            impact_post=impact_post,
            impact_post_cumulative=impact_post_cumulative,
            score=pre_period_score(predictions_pre, self.panel.treated_pre),
            shrinkage_state=shrinkage_state,
        )

    def plot(
        self,
        *,
        round_to: int | None = None,
        treated_unit: str | None = None,
        ci_prob: float = HDI_PROB,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        plot_predictors: bool = False,
        figsize: tuple[float, float] = (7, 8),
        show: bool = True,
    ) -> tuple[plt.Figure, list[plt.Axes] | np.ndarray]:
        """Plot the reconstructed untreated trajectory and impact panels.

        Requires :meth:`fit`. Bands are posterior intervals for the untreated
        trajectory and the impacts. They marginalize the shrinkage prior and
        do not claim a selected-rank interval or nominal coverage. Raw factor
        series are not drawn: they are not uniquely identified.

        Parameters
        ----------
        round_to : int, optional
            Significant figures for the pre-period :math:`R^2` in the title.
        treated_unit : str, optional
            Treated unit to draw. Defaults to the first treated name.
        ci_prob : float, default 0.94
            Interval mass. Must be in ``(0, 1]``.
        kind : {"ribbon", "histogram", "spaghetti"}, default "ribbon"
            How posterior uncertainty is rendered.
        ci_kind : {"hdi", "eti"}, default "hdi"
            Interval type when ``kind="ribbon"``.
        num_samples : int, default 50
            Draw lines when ``kind="spaghetti"``.
        plot_predictors : bool, default False
            Overlay donor trajectories on the counterfactual panel.
        figsize : tuple of float, default (7, 8)
            Figure size in inches.
        show : bool, default True
            Whether to display the figure.

        Returns
        -------
        tuple of matplotlib.figure.Figure and numpy.ndarray
            Three panels: untreated trajectory, point impact, cumulative impact.
        """
        if self.result is None:
            raise RuntimeError("Call fit() before plot().")
        style: _PosteriorPlotStyle = {
            "ci_prob": ci_prob,
            "kind": kind,
            "ci_kind": ci_kind,
            "num_samples": num_samples,
        }
        unit = resolve_treated_unit(self.treated_units, treated_unit)
        unit_index = self.treated_units.index(unit)
        title = format_r2_score(
            self.result.score,
            unit_index=unit_index,
            round_to=round_to,
            context="on pre-period treated outcomes",
        )
        figure, axes = self.panel.plot(
            self.result,
            group="posterior",
            treated_unit=unit,
            title=title,
            style=style,
            figsize=figsize,
            plot_predictors=plot_predictors,
        )
        if show:
            plt.show()
        return figure, axes

    def _control_outcomes(self) -> xr.DataArray:
        """Control outcomes at every time. Treated columns are not included."""
        pre = self.panel.control("pre")
        post = self.panel.control("post")
        return xr.concat([pre, post], dim="obs_ind")

    def _require_shrinkage_state(self, shrinkage_state: xr.DataArray) -> None:
        if not isinstance(shrinkage_state, xr.DataArray):
            raise TypeError("shrinkage_state must be an xarray.DataArray of scales.")
        if shrinkage_state.dtype.kind in {"i", "u", "b"}:
            raise TypeError("shrinkage_state must be scale draws, not an integer rank.")
        dims = tuple(shrinkage_state.dims)
        if dims != (*_DRAW_DIMS, _FACTORS):
            raise ValueError(
                "shrinkage_state dims must be ('chain', 'draw', 'factors'), "
                f"got {dims}."
            )
        if shrinkage_state.sizes[_FACTORS] != self.max_factors:
            raise ValueError(
                "shrinkage_state must have one scale per factor up to "
                f"max_factors={self.max_factors}, got {shrinkage_state.sizes[_FACTORS]}."
            )

    def _require_factor_axis(self, array: xr.DataArray, name: str) -> None:
        if name == "factors":
            _require_shared_factors(array)
        if _FACTORS not in array.dims:
            raise ValueError(f"{name} must have a {_FACTORS!r} dimension.")
        if array.sizes[_FACTORS] != self.max_factors:
            raise ValueError(
                f"{name} factor axis must have length max_factors="
                f"{self.max_factors}, got {array.sizes[_FACTORS]}."
            )


def untreated_counterfactual(
    unit_intercept: xr.DataArray,
    time_intercept: xr.DataArray,
    loadings: xr.DataArray,
    factors: xr.DataArray,
) -> xr.DataArray:
    """Return unit intercept + time intercept + loading-factor product.

    Raw factors and raw loadings are not unique. The returned trajectory is
    the identified product, not a selected-factor reconstruction.

    Parameters
    ----------
    unit_intercept : xarray.DataArray
        Additive unit intercept. Must include ``treated_units``.
    time_intercept : xarray.DataArray
        Additive time intercept. Must include ``obs_ind``.
    loadings : xarray.DataArray
        Loadings with a ``factors`` dimension.
    factors : xarray.DataArray
        Shared factors. Dimensions must be ``chain``, ``draw``, ``obs_ind``,
        and ``factors``.

    Returns
    -------
    xarray.DataArray
        Untreated counterfactual.
    """
    _require_shared_factors(factors)
    product = (loadings * factors).sum(dim=_FACTORS)
    return unit_intercept + time_intercept + product


_SHARED_FACTOR_DIMS = frozenset((*_DRAW_DIMS, "obs_ind", _FACTORS))


def _require_shared_factors(factors: xr.DataArray) -> None:
    """Reject a factor array that is not shared across treated units."""
    dims = set(factors.dims)
    if dims != _SHARED_FACTOR_DIMS:
        raise ValueError(
            "factors must be the shared fit with dimensions "
            "('chain', 'draw', 'obs_ind', 'factors'), "
            f"got {tuple(factors.dims)}."
        )


def pre_period_score(
    predictions_pre: xr.DataArray,
    treated_pre: xr.DataArray,
) -> pd.Series:
    """Bayesian :math:`R^2` of pre-period predictions against pre-period outcomes.

    Post-period outcomes are not an argument.

    Parameters
    ----------
    predictions_pre : xarray.DataArray
        Counterfactual draws on the pre-period.
    treated_pre : xarray.DataArray
        Observed treated outcomes strictly before treatment.

    Returns
    -------
    pandas.Series
        ``unit_{i}_r2`` and ``unit_{i}_r2_std`` for each treated unit.
    """
    stacked = predictions_pre.stack(sample=_DRAW_DIMS)
    scores: dict[str, float] = {}
    for i, unit in enumerate(treated_pre.coords["treated_units"].values):
        unit_mu = stacked.sel(treated_units=unit).transpose("sample", "obs_ind")
        unit_y = treated_pre.sel(treated_units=unit)
        unit_score = _bayesian_r2_score(unit_y.to_numpy(), unit_mu.to_numpy())
        scores[f"unit_{i}_r2"] = float(unit_score["r2"])
        scores[f"unit_{i}_r2_std"] = float(unit_score["r2_std"])
    return pd.Series(scores)


def _factor_names(max_factors: int) -> list[str]:
    return [f"f{i}" for i in range(max_factors)]


def _control_model(control_y: xr.DataArray, max_factors: int) -> pm.Model:
    """Gaussian factor graph on control cells only.

    The likelihood uses the one shared positive residual scale. A unit
    observation scale is not this model. Treated outcomes are not in this
    graph, so they do not update these factor or time-intercept draws.
    """
    factor_names = _factor_names(max_factors)
    alpha = np.array([_MGP_A1 if i == 0 else _MGP_A2 for i in range(max_factors)])
    with pm.Model(
        coords={
            "obs_ind": control_y.coords["obs_ind"].values,
            "control_units": control_y.coords["control_units"].values,
            "factors": factor_names,
        }
    ) as model:
        delta = pm.Gamma("mgp_delta", alpha=alpha, beta=1.0, dims="factors")
        precision = pt.cumprod(delta)
        pm.Deterministic("shrinkage_state", 1.0 / pt.sqrt(precision), dims="factors")
        sigma = pm.HalfCauchy("sigma", beta=_RESIDUAL_CAUCHY_BETA)
        unit_intercept = pm.Normal(
            "control_unit_intercept", 0.0, _INTERCEPT_SD, dims="control_units"
        )
        time_intercept = pm.Normal("time_intercept", 0.0, _INTERCEPT_SD, dims="obs_ind")
        shared_factors = pm.Normal(
            "shared_factors", 0.0, 1.0, dims=("obs_ind", "factors")
        )
        shrinkage = model["shrinkage_state"]
        loadings = pm.Normal(
            "control_loadings",
            0.0,
            shrinkage,
            dims=("control_units", "factors"),
        )
        product = pt.sum(
            loadings.dimshuffle("x", 0, 1) * shared_factors.dimshuffle(0, "x", 1),
            axis=2,
        )
        mu = unit_intercept + time_intercept.dimshuffle(0, "x") + product
        observed = control_y.transpose("obs_ind", "control_units")
        pm.Normal(
            "control_y",
            mu,
            sigma,
            observed=observed.values,
            dims=("obs_ind", "control_units"),
        )
    return model


def _control_initvals(
    model: pm.Model, control_y: xr.DataArray
) -> dict[str, np.ndarray]:
    """SVD start for the factor product. Not a rank selector."""
    values = control_y.transpose("obs_ind", "control_units").values
    centered = values - values.mean(axis=0, keepdims=True)
    left, singular, right = np.linalg.svd(centered, full_matrices=False)
    n_factors = int(model.dim_lengths["factors"].eval())
    rank = min(n_factors, left.shape[1])
    factors = np.zeros((values.shape[0], n_factors))
    loadings = np.zeros((values.shape[1], n_factors))
    if rank:
        scale = np.sqrt(np.maximum(singular[:rank], 1e-8))
        factors[:, :rank] = left[:, :rank] * scale
        loadings[:, :rank] = right[:rank, :].T * scale
    return {
        "shared_factors": factors,
        "control_loadings": loadings,
        "mgp_delta": np.array([1.0 if i == 0 else 1.5 for i in range(n_factors)]),
        "sigma": np.array(0.5),
    }


def _scale_log_posterior(
    log_scale: float,
    *,
    gram: np.ndarray,
    centered: np.ndarray,
    control_rss: float,
    n_control: int,
) -> float:
    """Profile the one scale given factors, integrating intercepts and loadings."""
    scale_sq = np.exp(2.0 * log_scale)
    covariance = scale_sq * np.eye(gram.shape[0]) + gram
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0 or not np.isfinite(logdet):
        return -1e300
    solved = np.linalg.solve(covariance, centered)
    quadratic = float(np.sum(centered * solved))
    n_units = centered.shape[1]
    prior = -np.log1p(scale_sq / _RESIDUAL_CAUCHY_BETA**2)
    control = -n_control * log_scale - control_rss / (2.0 * scale_sq)
    return -0.5 * n_units * logdet - 0.5 * quadratic + control + prior


def _updated_residual_scale(
    sigma: np.ndarray,
    control_rss: np.ndarray,
    n_control: int,
    factors_pre: np.ndarray,
    shrinkage: np.ndarray,
    treated_centered: np.ndarray,
) -> np.ndarray:
    """Move each control-panel scale draw using treated pre-period outcomes.

    The target conditions on the sampled factors. It uses the control residuals
    at those draws and the treated pre-period after integrating the unit
    intercept and loadings under their priors. Factors are not resampled. This
    is not a second residual-scale random variable.
    """
    updated = np.empty(sigma.shape, dtype=float)
    n_time = treated_centered.shape[1]
    for index in range(sigma.shape[0]):
        design = np.column_stack((np.ones(n_time), factors_pre[index]))
        prior_sd = np.concatenate(([_INTERCEPT_SD], shrinkage[index]))
        gram = (design * prior_sd**2) @ design.T
        centered = treated_centered[index]
        rss = float(control_rss[index])
        log_s = float(np.log(max(float(sigma[index]), 1e-8)))
        for _ in range(_SCALE_UPDATE_STEPS):
            current = _scale_log_posterior(
                log_s,
                gram=gram,
                centered=centered,
                control_rss=rss,
                n_control=n_control,
            )
            above = _scale_log_posterior(
                log_s + _SCALE_FD_STEP,
                gram=gram,
                centered=centered,
                control_rss=rss,
                n_control=n_control,
            )
            below = _scale_log_posterior(
                log_s - _SCALE_FD_STEP,
                gram=gram,
                centered=centered,
                control_rss=rss,
                n_control=n_control,
            )
            gradient = (above - below) / (2.0 * _SCALE_FD_STEP)
            curvature = (above - 2.0 * current + below) / _SCALE_FD_STEP**2
            if not np.isfinite(gradient) or not np.isfinite(curvature):
                break
            if curvature < -1e-6:
                step = -gradient / curvature
            else:
                step = 0.5 if gradient > 0.0 else -0.5
            step = float(np.clip(step, -1.5, 1.5))
            accepted = log_s
            for fraction in (1.0, 0.5, 0.25):
                trial = log_s + fraction * step
                trial_value = _scale_log_posterior(
                    trial,
                    gram=gram,
                    centered=centered,
                    control_rss=rss,
                    n_control=n_control,
                )
                if trial_value >= current - 1e-6:
                    accepted = trial
                    break
            log_s = accepted
        updated[index] = float(np.exp(log_s))
    return updated


def _sample_treated_conditional(
    control_y: xr.DataArray,
    treated_pre: xr.DataArray,
    control_posterior: xr.Dataset,
    *,
    random_seed: int | None,
    progressbar: bool,
) -> dict[str, xr.DataArray]:
    """Sample treated intercepts and loadings at the updated residual scale.

    Factors and time intercepts are data from the control-panel posterior.
    The one shared residual scale is not a random variable here: its draws are
    updated from treated pre-period outcomes, then held fixed. Loadings use
    those updated draws. Treated pre-period outcomes cannot move the factor or
    time-intercept draws.
    """
    posterior = _as_dataset(control_posterior)
    stacked = posterior.stack(sample=_DRAW_DIMS)
    draw_index = stacked.indexes["sample"]
    stacked = stacked.drop_vars(["sample", "chain", "draw"]).assign_coords(
        sample=np.arange(len(draw_index))
    )
    factors = stacked["shared_factors"].transpose("sample", "obs_ind", "factors")
    time_intercept = stacked["time_intercept"].transpose("sample", "obs_ind")
    shrinkage = stacked["shrinkage_state"].transpose("sample", "factors")
    control_mu = _control_mean(stacked)
    residuals = control_y - control_mu
    control_rss = (residuals**2).sum(("obs_ind", "control_units")).values
    n_control = int(control_y.sizes["obs_ind"] * control_y.sizes["control_units"])
    pre_times = treated_pre.coords["obs_ind"]
    factor_pre = (
        factors.sel(obs_ind=pre_times).transpose("sample", "obs_ind", "factors").values
    )
    xi_pre = time_intercept.sel(obs_ind=pre_times).transpose("sample", "obs_ind").values
    treated_values = treated_pre.transpose("obs_ind", "treated_units").values
    treated_centered = treated_values[None, :, :] - xi_pre[:, :, None]
    updated_scale = _updated_residual_scale(
        np.asarray(stacked["sigma"].values, dtype=float),
        np.asarray(control_rss, dtype=float),
        n_control,
        factor_pre,
        np.asarray(shrinkage.values, dtype=float),
        treated_centered,
    )
    treated_observed = np.broadcast_to(
        treated_values,
        (
            factors.sizes["sample"],
            treated_pre.sizes["obs_ind"],
            treated_pre.sizes["treated_units"],
        ),
    ).copy()
    with pm.Model(
        coords={
            "sample": factors.coords["sample"].values,
            "treated_units": treated_pre.coords["treated_units"].values,
            "factors": factors.coords["factors"].values,
        }
    ) as model:
        residual_scale = pm.Data("residual_scale", updated_scale, dims="sample")
        scale = residual_scale.dimshuffle(0, "x", "x")
        unit_intercept = pm.Normal(
            "unit_intercept",
            0.0,
            _INTERCEPT_SD,
            dims=("sample", "treated_units"),
        )
        shrink = np.broadcast_to(
            shrinkage.values[:, None, :],
            (
                factors.sizes["sample"],
                treated_pre.sizes["treated_units"],
                factors.sizes["factors"],
            ),
        ).copy()
        loadings = pm.Normal(
            "loadings",
            0.0,
            shrink,
            dims=("sample", "treated_units", "factors"),
        )
        mu = (
            unit_intercept.dimshuffle(0, "x", 1)
            + xi_pre[:, :, None]
            + pt.sum(
                loadings.dimshuffle(0, "x", 1, 2) * factor_pre[:, :, None, :], axis=3
            )
        )
        pm.Normal("treated_pre", mu, scale, observed=treated_observed)
        idata = pm.sample(
            draws=1,
            tune=40,
            chains=1,
            cores=1,
            random_seed=random_seed,
            progressbar=progressbar,
            model=model,
            compute_convergence_checks=False,
        )
    drawn = _as_dataset(idata).isel(chain=0, draw=-1)
    return {
        "unit_intercept": _restore_draws(drawn["unit_intercept"], draw_index),
        "loadings": _restore_draws(drawn["loadings"], draw_index),
    }


def _control_mean(stacked: xr.Dataset) -> xr.DataArray:
    loadings = stacked["control_loadings"].transpose(
        "sample", "control_units", "factors"
    )
    factors = stacked["shared_factors"].transpose("sample", "obs_ind", "factors")
    product = (loadings * factors).sum("factors")
    return stacked["control_unit_intercept"] + stacked["time_intercept"] + product


def _as_dataset(posterior: xr.Dataset | Any) -> xr.Dataset:
    """Return a posterior dataset from a Dataset, DataTree, or InferenceData."""
    if isinstance(posterior, xr.Dataset):
        return posterior
    children = getattr(posterior, "children", None)
    if children and "posterior" in children:
        return _as_dataset(posterior["posterior"])
    dataset = getattr(posterior, "dataset", None)
    if dataset is not None and hasattr(dataset, "data_vars"):
        if isinstance(dataset, xr.Dataset):
            return dataset
        return xr.Dataset(dataset)
    raise TypeError("control posterior must be an xarray Dataset or a PyMC DataTree.")


def _restore_draws(array: xr.DataArray, draw_index: pd.Index) -> xr.DataArray:
    """Restore ``chain`` and ``draw`` from the integer conditional index."""
    coords = xr.Coordinates.from_pandas_multiindex(draw_index, "sample")
    restored = array.assign_coords(coords).unstack("sample")
    return restored.transpose("chain", "draw", ...)
