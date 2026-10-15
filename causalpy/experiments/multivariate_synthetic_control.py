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
"""Multivariate Synthetic Control Experiment.

Save as ``causalpy/experiments/multivariate_synthetic_control.py``.
"""

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from causalpy.custom_exceptions import DataException, GroupNotSampledException
from causalpy.experiments._results import CausalResult
from causalpy.experiments.model_adapter import build_coords
from causalpy.input_data import DataFrameLike, to_pandas
from causalpy.pymc_models import MvWeightedSumFitter, PyMCModel
from causalpy.utils import round_num

from .synthetic_control import SyntheticControl


class MultivariateSyntheticControl(SyntheticControl):
    """Synthetic control that also matches on additional predictors.

    Besides the outcome, each unit can carry extra time-varying predictors
    (e.g. industrialisation, wages, unemployment). Donor weights ``W`` are
    learned from the outcome trajectory and from the predictors, and a
    separate set of predictor weights ``V`` (shared across units) says how
    much each predictor matters for the match.

    Parameters
    ----------
    data : dataframe-like
        **Long** format: one row per ``(unit, time)`` with the outcome and the
        predictors as columns.
    treatment_time : int, float, or pd.Timestamp
        Time of treatment, in the units of ``time_column``.
    outcome : str
        Name of the outcome column.
    predictors : list of str
        Additional predictor columns (the outcome is *not* repeated here).
        Predictors are standardised using pre-treatment statistics.
    control_units, treated_units : list of str
        Values of ``unit_column`` for donors and treated units.
    unit_column, time_column : str
        Names of the unit and time columns.
    model : PyMCModel or None, default None
        Defaults to :class:`~causalpy.pymc_models.MvWeightedSumFitter`.
        sklearn models are not supported.
    min_donor_correlation : float, default 0.0
        As in :class:`SyntheticControl`; donor correlations are computed on the
        outcome only.
    auto_scale_sigma : bool, default True
        As in :class:`SyntheticControl`; the data-driven prior on the outcome
        noise is based on the outcome only.

    Notes
    -----
    The counterfactual is built from the outcome only. Predictors inform the
    weights but are never forecast, so they need not be observed after
    treatment (missing post-treatment predictor values are ignored).
    Predictors must be observed for every unit at every pre-treatment time.
    """

    supports_ols = False
    supports_bayes = True
    _default_model_class = MvWeightedSumFitter

    def __init__(
        self,
        data: DataFrameLike,
        treatment_time: int | float | pd.Timestamp,
        outcome: str,
        predictors: list[str],
        control_units: list[str],
        treated_units: list[str],
        unit_column: str,
        time_column: str,
        model: PyMCModel | None = None,
        min_donor_correlation: float = 0.0,
        auto_scale_sigma: bool = True,
    ) -> None:
        predictors = [p for p in predictors if p != outcome]
        if len(predictors) == 0:
            raise ValueError(
                "MultivariateSyntheticControl needs at least one predictor besides "
                "the outcome; use SyntheticControl for outcome-only analysis."
            )
        # Everything the parent's __init__ -> _prepare_data() will need must be
        # set before calling super().__init__ (BaseExperiment.__init__ only
        # installs the model, so this ordering is safe).
        self.outcome = outcome
        self.predictors = predictors
        self.features = [outcome, *predictors]
        self.outcome_variable_name = outcome  # used by maketables
        self._feature_frames = self._pivot_long_panel(
            to_pandas(data),
            unit_column=unit_column,
            time_column=time_column,
            features=self.features,
            units=[*control_units, *treated_units],
        )
        super().__init__(
            self._feature_frames[outcome],  # wide outcome: time x unit
            treatment_time,
            control_units=control_units,
            treated_units=treated_units,
            model=model,
            min_donor_correlation=min_donor_correlation,
            auto_scale_sigma=auto_scale_sigma,
        )
        self.expt_type = "MultivariateSyntheticControl"

    @staticmethod
    def _pivot_long_panel(
        long: pd.DataFrame,
        *,
        unit_column: str,
        time_column: str,
        features: list[str],
        units: list[str],
    ) -> dict[str, pd.DataFrame]:
        """Pivot long data into one wide (time x unit) frame per feature."""
        needed = [unit_column, time_column, *features]
        missing = [c for c in needed if c not in long.columns]
        if missing:
            raise DataException(
                f"Columns {missing} not found in data. Available: {list(long.columns)}."
            )
        if long.duplicated([time_column, unit_column]).any():
            raise DataException(
                f"Data has duplicate ({unit_column!r}, {time_column!r}) rows; "
                "there must be exactly one row per unit and time."
            )
        absent = sorted(set(units) - set(long[unit_column].unique()), key=str)
        if absent:
            raise DataException(f"Units {absent} not found in '{unit_column}'.")

        long = long[long[unit_column].isin(units)]
        frames = {}
        for feature in features:
            wide = long.pivot(index=time_column, columns=unit_column, values=feature)
            wide = wide.sort_index()[units]
            wide.columns.name = None
            frames[feature] = wide
        return frames

    def _prepare_data(self) -> None:
        """Build the outcome-only designs (parent) plus the multivariate designs."""
        super()._prepare_data()  # self.pre_design / self.post_design (outcome only)
        units = [*self.control_units, *self.treated_units]
        pre_idx, post_idx = self.datapre.index, self.datapost.index
        pre_frames = {
            f: self._feature_frames[f].loc[pre_idx, units] for f in self.features
        }
        post_frames = {
            f: self._feature_frames[f].loc[post_idx, units] for f in self.features
        }

        for name, frame in pre_frames.items():
            if frame.isna().any().any():
                raise DataException(
                    f"Feature '{name}' has missing values before treatment; every "
                    "feature must be observed for every unit at every pre-treatment time."
                )
        if post_frames[self.outcome].isna().any().any():
            raise DataException(
                f"Outcome '{self.outcome}' has missing values after treatment."
            )

        # Standardise auxiliary predictors with pooled pre-treatment statistics
        # (the outcome keeps its original scale so impacts stay interpretable).
        self._feature_stats: dict[str, tuple[float, float]] = {}
        for name in self.predictors:
            values = pre_frames[name].to_numpy(dtype=float)
            mean, std = float(values.mean()), float(values.std(ddof=1))
            if not np.isfinite(std) or std <= 0:
                raise DataException(
                    f"Predictor '{name}' is constant before treatment and cannot "
                    "inform the weights."
                )
            self._feature_stats[name] = (mean, std)

        self.pre_design_mv = self._build_mv_design(pre_frames, pre_idx)
        self.post_design_mv = self._build_mv_design(post_frames, post_idx)

    def _build_mv_design(
        self, frames: dict[str, pd.DataFrame], index: pd.Index
    ) -> xr.Dataset:
        """Stack features into ``control (obs, coeffs, features)`` and
        ``treated (obs, features, treated_units)`` arrays."""
        control_parts, treated_parts = [], []
        for feature in self.features:
            frame = frames[feature]
            if feature != self.outcome:
                mean, std = self._feature_stats[feature]
                # Post-treatment predictors are never used for prediction, so
                # missing values there are neutralised to 0.
                frame = ((frame - mean) / std).fillna(0.0)
            control_parts.append(frame[self.control_units].to_numpy(dtype=float))
            treated_parts.append(frame[self.treated_units].to_numpy(dtype=float))
        return xr.Dataset(
            {
                "control": xr.DataArray(
                    np.stack(control_parts, axis=-1),
                    dims=["obs_ind", "coeffs", "features"],
                    coords={
                        "obs_ind": index,
                        "coeffs": self.control_units,
                        "features": self.features,
                    },
                ),
                "treated": xr.DataArray(
                    np.stack(treated_parts, axis=1),
                    dims=["obs_ind", "features", "treated_units"],
                    coords={
                        "obs_ind": index,
                        "features": self.features,
                        "treated_units": self.treated_units,
                    },
                ),
            }
        )

    def _fit_inputs(self):
        """Return the pre-period 3D designs and coordinates for ``build``."""
        return (
            self.pre_design_mv["control"],
            self.pre_design_mv["treated"],
            build_coords(
                self.control_units,
                self.datapre.shape[0],
                treated_units=self.treated_units,
                features=self.features,
            ),
        )

    def _finalize(self, group: Literal["prior", "posterior"]) -> None:
        """Same as ``SyntheticControl._finalize`` but with the 3D designs.

        Predictions are for the outcome only (``mu``), so impacts are computed
        against the outcome-only treated series held by the parent.
        """
        control_pre = self.pre_design_mv["control"]
        control_post = self.post_design_mv["control"]
        treated_pre = self.pre_design["treated"]
        treated_post = self.post_design["treated"]

        predictions_pre = self._model_backend.predict(X=control_pre, group=group)
        predictions_post = self._model_backend.predict(X=control_post, group=group)
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
            score = self._model_backend.score(X=control_pre, y=treated_pre)

        self._assign_bundle(
            group,
            CausalResult(
                predictions_pre=predictions_pre,
                predictions_post=predictions_post,
                impact_pre=impact_pre,
                impact_post=impact_post,
                impact_post_cumulative=impact_post_cumulative,
                score=score,
            ),
        )

    def predictor_weights(self) -> xr.DataArray:
        """Posterior draws of the predictor weights ``v`` (dims chain, draw, features_aux)."""
        if not self.is_fitted:
            raise GroupNotSampledException(
                f"No posterior draws are available. Call {type(self).__name__}.fit() first.",
                group="posterior",
            )
        return self.model.require_group("posterior")["v"]

    def summary(self, round_to: int | None = None) -> None:
        """Print the usual summary, plus ``v`` when it is learned.

        ``v`` measures how well each predictor can be matched, not how much it
        matters for the outcome.

        Parameters
        ----------
        round_to : int or None, default None
            Number of significant figures used when printing the values.
        """
        super().summary(round_to)
        if getattr(self.model, "v_prior", "learned") != "learned":
            return  # fixed uniform weights: nothing to report
        v_mean = self.predictor_weights().mean(dim=["chain", "draw"])
        print("Predictor weights (v):")
        width = max(len(str(name)) for name in self.predictors)
        for name in self.predictors:
            value = round_num(float(v_mean.sel(features_aux=name)), round_to)
            print(f"  {str(name):<{width}}  {value}")
