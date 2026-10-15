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
"""Interactive fixed-effects synthetic control: frozen call surface only.

This module does not sample. The prior family, and rotation or scale
constraints beyond "unused factor scales can go to zero," are not frozen.
Do not add a prior here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import xarray as xr

from causalpy.date_utils import validate_treatment_time_against_index
from causalpy.experiments._panel_counterfactual import WidePanel
from causalpy.experiments._results import CausalResult
from causalpy.utils import _bayesian_r2_score

_DRAW_DIMS = ("chain", "draw")
_FACTORS = "factors"


@dataclass(frozen=True)
class InteractiveFixedEffectsResult(CausalResult):
    """Result bundle for interactive fixed-effects synthetic control.

    Inherits the :class:`~causalpy.experiments._results.CausalResult`
    prediction and impact fields. ``predictions_*`` are the untreated
    counterfactual. ``score`` is the pre-period prediction error, fit to
    pre-period treated outcomes only.

    ``shrinkage_state`` holds posterior draws of one scale per factor, up to
    ``max_factors``. It is not a selected integer rank and not a derived
    effective-rank count. Posterior intervals on the inherited prediction
    fields are whatever draws the caller composed; this class does not
    condition them on a selected rank.
    """

    shrinkage_state: xr.DataArray = field(kw_only=True)


class InteractiveFixedEffectsSyntheticControl:
    """Holds a balanced :class:`WidePanel` for interactive fixed effects.

    Not a :class:`~causalpy.experiments.synthetic_control.SyntheticControl`
    subclass. Several treated units share one ``treatment_time``. Control
    units are donors. There is no ``n_factors`` argument: ``max_factors`` is
    an upper bound, not a selected rank.

    This class does not fit a factor model. Composing a result from component
    draws does not read treated post-intervention outcomes except as the
    observed side of impact.

    Parameters
    ----------
    data : pandas.DataFrame
        Balanced wide panel indexed by time. Used columns must have no
        missing values.
    treatment_time : int, float, or pandas.Timestamp
        Shared first post-period time.
    control_units : sequence of str
        Donor columns.
    treated_units : sequence of str
        Treated columns. One or more.
    max_factors : int
        Upper bound on the number of factors. Not a selected rank.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_time: int | float | pd.Timestamp,
        control_units: list[str] | tuple[str, ...],
        treated_units: list[str] | tuple[str, ...],
        *,
        max_factors: int,
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
        panel = WidePanel.from_frame(
            data,
            treatment_time,
            control,
            treated,
        )
        if panel.pre.empty or panel.post.empty:
            raise ValueError("the panel must have both a pre-period and a post-period.")
        self.panel = panel
        self.max_factors = max_factors

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
        pre-period treated outcomes only. ``shrinkage_state`` is stored as
        given after a shape check. This method does not update shrinkage
        from treated post-intervention outcomes and does not select a rank.

        Parameters
        ----------
        unit_intercept : xarray.DataArray
            Draws with dimensions ``chain``, ``draw``, and ``treated_units``.
        time_intercept : xarray.DataArray
            Draws with dimensions ``chain``, ``draw``, and ``obs_ind``,
            covering the stored pre-period and post-period times.
        loadings : xarray.DataArray
            Per-treated-unit loadings with dimensions ``chain``, ``draw``,
            ``treated_units``, and ``factors``. Length of ``factors`` is
            ``max_factors``. These must already be pre-period loadings; this
            method does not refit them.
        factors : xarray.DataArray
            Shared factors with dimensions ``chain``, ``draw``, ``obs_ind``,
            and ``factors``.
        shrinkage_state : xarray.DataArray
            Posterior draws of one scale per factor, dimensions ``chain``,
            ``draw``, and ``factors``, length ``max_factors``.

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
                f"max_factors={self.max_factors}, got "
                f"{shrinkage_state.sizes[_FACTORS]}."
            )

    def _require_factor_axis(self, array: xr.DataArray, name: str) -> None:
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

    Parameters
    ----------
    unit_intercept : xarray.DataArray
        Additive unit intercept. Must include ``treated_units``.
    time_intercept : xarray.DataArray
        Additive time intercept. Must include ``obs_ind``.
    loadings : xarray.DataArray
        Loadings with a ``factors`` dimension.
    factors : xarray.DataArray
        Factors with a ``factors`` dimension aligned to ``loadings``.

    Returns
    -------
    xarray.DataArray
        Untreated counterfactual. Not the loading-factor product alone, and
        not the unit intercept alone.
    """
    product = (loadings * factors).sum(dim=_FACTORS)
    return unit_intercept + time_intercept + product


def pre_period_score(
    predictions_pre: xr.DataArray,
    treated_pre: xr.DataArray,
) -> pd.Series:
    """Bayesian :math:`R^2` of pre-period predictions against pre-period outcomes.

    Parameters
    ----------
    predictions_pre : xarray.DataArray
        Counterfactual draws on the pre-period, with ``chain``, ``draw``,
        ``obs_ind``, and ``treated_units``.
    treated_pre : xarray.DataArray
        Observed treated outcomes strictly before treatment.

    Returns
    -------
    pandas.Series
        ``unit_{i}_r2`` and ``unit_{i}_r2_std`` for each treated unit, in
        treated-unit order. Post-period outcomes are not an argument.
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
