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
"""Joint-posterior augmented synthetic control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from causalpy.date_utils import validate_treatment_time_against_index
from causalpy.experiments._panel_counterfactual import WidePanel
from causalpy.input_data import DataFrameLike, to_pandas_with_time_index

# Frozen observation scale for both Normal likelihoods. A fixed scale, not a prior.
_OBSERVATION_SCALE = 1.0


class AugmentedSyntheticControl:
    """Sample a joint augmented synthetic control posterior.

    One PyMC model draws simplex base weights and one shared donor prognostic
    correction. Credible intervals of the counterfactual and the treatment
    effect are Bayesian intervals for that full model, not a cut posterior.

    Weights are independent across treated units. Each unit's weights are
    ``Dirichlet`` with concentration ones, with a Normal likelihood on that
    unit's pre-period outcomes and mean equal to the donor pre-period matrix
    times the unit's weights. The prognostic regression uses donor outcomes at
    each time as the response and that donor's full pre-period outcome vector
    as covariates, with time-specific coefficients and no intercept.
    Coefficients have a ``Normal(0, tau)`` prior and ``tau`` is ``HalfNormal``
    with scale 1. Both Normal likelihoods use observation scale 1. All units
    share one ``treatment_time``.

    Parameters
    ----------
    data : DataFrameLike
        Wide outcome panel. The time axis is the index, or ``time_column``.
    treatment_time : int, float, or pandas.Timestamp
        First post-period time. Rows at this time are post-period.
    control_units : sequence of str
        Donor column names.
    treated_units : sequence of str
        Treated column names.
    time_column : str, optional
        Column to use as the time axis when ``data`` has no index.

    Raises
    ------
    DataException
        If the time axis cannot be taken from ``data``.
    BadIndexException
        If ``treatment_time`` cannot be compared with the time axis.
    ValueError
        If the unit columns, the pre-period, or the post-period cannot support
        the recorded likelihoods.

    Notes
    -----
    The untreated counterfactual at post-period time ``t`` for treated unit
    ``u`` is ``donor_post[t] @ w_u + (X_u - X_donors @ w_u) @ beta[t]``.
    ``X_u`` and each donor row of ``X_donors`` are full pre-period outcome
    vectors. ``beta[t]`` is the prognostic coefficient vector at time ``t``.
    The treatment effect is the treated post-period outcome minus that
    counterfactual. Treated post-period outcomes do not enter the
    counterfactual or either likelihood.
    """

    def __init__(
        self,
        data: DataFrameLike,
        treatment_time: int | float | pd.Timestamp,
        control_units: Sequence[str],
        treated_units: Sequence[str],
        *,
        time_column: str | None = None,
    ) -> None:
        pandas_data = to_pandas_with_time_index(data, time_column)
        validate_treatment_time_against_index(pandas_data.index, treatment_time)
        donors = _unique_names(control_units, "control_units")
        treated = _unique_names(treated_units, "treated_units")
        _reject_overlap(donors, treated)
        _reject_missing_columns(pandas_data, donors, treated)
        if not pandas_data.index.is_unique:
            raise ValueError("data.index must be unique.")

        panel = WidePanel.from_frame(
            pandas_data,
            treatment_time,
            donors,
            treated,
        )
        if panel.pre.empty:
            raise ValueError(
                "The pre-period is empty. treatment_time must be after the first time."
            )
        if panel.post.empty:
            raise ValueError(
                "The post-period is empty. treatment_time must be at or before "
                "the last time."
            )

        self.data = pandas_data
        self.treatment_time = treatment_time
        self.control_units = donors
        self.treated_units = treated
        self._panel = panel
        self.model = _build_model(panel)
        self.idata: az.InferenceData | None = None

    def fit(
        self, sample_kwargs: Mapping[str, Any] | None = None
    ) -> AugmentedSyntheticControl:
        """Sample the joint posterior.

        Parameters
        ----------
        sample_kwargs : mapping, optional
            Arguments forwarded to ``pymc.sample``. ``None`` uses that
            function's defaults.

        Returns
        -------
        AugmentedSyntheticControl
            The fitted instance. The joint posterior, including the
            counterfactual and the treatment effect, is stored on ``idata``.
        """
        kwargs = {} if sample_kwargs is None else dict(sample_kwargs)
        with self.model:
            self.idata = pm.sample(**kwargs)
        return self


def _unique_names(names: Sequence[str], label: str) -> tuple[str, ...]:
    if len(names) == 0:
        raise ValueError(f"{label} must be nonempty.")
    unique = tuple(names)
    if len(set(unique)) != len(unique):
        raise ValueError(f"{label} must be unique.")
    return unique


def _reject_overlap(donors: Sequence[str], treated: Sequence[str]) -> None:
    overlap = sorted(set(donors) & set(treated))
    if overlap:
        raise ValueError(
            f"control_units and treated_units overlap: {', '.join(overlap)}."
        )


def _reject_missing_columns(
    data: pd.DataFrame, donors: Sequence[str], treated: Sequence[str]
) -> None:
    missing = [name for name in (*donors, *treated) if name not in data.columns]
    if missing:
        raise ValueError(f"data is missing columns: {missing}.")


def _finite_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    selected = frame.loc[:, list(columns)]
    try:
        values = selected.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("outcome columns must be numeric.") from error
    if not np.isfinite(values).all():
        raise ValueError("outcome columns must be finite.")
    return values


def _build_model(panel: WidePanel) -> pm.Model:
    """Build the one joint model for the recorded algebra.

    Likelihoods do not share observations: weights see treated pre-period
    outcomes only, and the prognostic regression sees donor outcomes only.
    The counterfactual and the treatment effect are deterministics of each
    joint draw. Treated post-period outcomes are data for the effect only.
    """
    n_pre = panel.pre.shape[0]
    donor_all = np.vstack(
        (
            _finite_matrix(panel.pre, panel.control_units),
            _finite_matrix(panel.post, panel.control_units),
        )
    )
    treated_pre = _finite_matrix(panel.pre, panel.treated_units)
    treated_post = _finite_matrix(panel.post, panel.treated_units)
    n_donors = len(panel.control_units)
    coords = {
        "treated_units": list(panel.treated_units),
        "control_units": list(panel.control_units),
        "pre_time": list(panel.pre.index),
        "post_time": list(panel.post.index),
        "time": list(panel.pre.index) + list(panel.post.index),
    }

    with pm.Model(coords=coords) as model:
        donor_outcomes = pm.Data(
            "donor_outcomes_data",
            donor_all,
            dims=["time", "control_units"],
        )
        treated_pre_data = pm.Data(
            "treated_pre_data",
            treated_pre,
            dims=["pre_time", "treated_units"],
        )
        treated_post_data = pm.Data(
            "treated_post_data",
            treated_post,
            dims=["post_time", "treated_units"],
        )
        tau = pm.HalfNormal("tau", sigma=1.0)
        beta = pm.Normal(
            "prognostic_coefficients",
            mu=0.0,
            sigma=tau,
            dims=["time", "pre_time"],
        )
        weights = pm.Dirichlet(
            "base_weights",
            a=np.ones(n_donors),
            dims=["treated_units", "control_units"],
        )

        donor_pre = donor_outcomes[:n_pre]
        donor_post = donor_outcomes[n_pre:]
        x_donors = donor_pre.T
        x_treated = treated_pre_data.T

        # Normal likelihood of pre-period treated outcomes on X_donors_pre @ w_u.
        pm.Normal(
            "treated_pre",
            mu=pt.dot(donor_pre, weights.T),
            sigma=_OBSERVATION_SCALE,
            observed=treated_pre_data,
            dims=["pre_time", "treated_units"],
        )
        # Donor outcomes at each time, covariates = full pre-period vector.
        # No intercept. Time-specific coefficients share the shrinkage prior.
        pm.Normal(
            "donor_outcomes",
            mu=pt.dot(beta, donor_pre),
            sigma=_OBSERVATION_SCALE,
            observed=donor_outcomes,
            dims=["time", "control_units"],
        )

        imbalance = x_treated - pt.dot(weights, x_donors)
        beta_post = beta[n_pre:]
        correction = pt.dot(imbalance, beta_post.T)
        weighted_donors = pt.dot(donor_post, weights.T)
        counterfactual = pm.Deterministic(
            "counterfactual",
            weighted_donors + correction.T,
            dims=["post_time", "treated_units"],
        )
        pm.Deterministic(
            "treatment_effect",
            treated_post_data - counterfactual,
            dims=["post_time", "treated_units"],
        )
    return model
