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
"""Adapt held-out recursive skforecast simulations to the ITS result contract."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import r2_score

from causalpy.custom_exceptions import BadIndexException, GroupNotSampledException
from causalpy.experiments.model_adapter import ModelAdapter
from causalpy.skl_models import SkforecastModel, _validate_recursive_forecaster


def _dated_grid(index: pd.DatetimeIndex) -> pd.DatetimeIndex | None:
    if index.freq is not None:
        return index
    clocks = (index,) if index.tz is None else (index, index.tz_convert("UTC"))
    for clock in clocks:
        if clock.inferred_freq is not None:
            try:
                dated = pd.DatetimeIndex(clock, freq=clock.inferred_freq)
            except ValueError:
                continue
            return dated if index.tz is None else dated.tz_convert(index.tz)
    return None


def _native_calendar_matches(
    index: pd.DatetimeIndex, pre_size: int, window_size: int
) -> bool:
    from skforecast.utils import expand_index

    # Some timezone-aware calendar grids are interpreted differently by
    # skforecast depending on which timestamps its lag window contains.
    for end in range(window_size, pre_size + 1):
        stop = end + 1 if end < pre_size else len(index)
        if not expand_index(index[end - window_size : end], stop - end).equals(
            index[end:stop]
        ):
            return False
    return True


def _regular_index(
    index: pd.Index, pre_size: int, window_size: int
) -> pd.DatetimeIndex | pd.RangeIndex:
    if isinstance(index, pd.DatetimeIndex):
        dated = _dated_grid(index)
        if dated is not None:
            freq = dated.freq
            if (
                dated.tz is None
                or (
                    isinstance(freq, pd.offsets.Tick)
                    and not isinstance(freq, pd.offsets.Day)
                )
                or _native_calendar_matches(dated, pre_size, window_size)
            ):
                return dated

            # Preserve the original feature clock when the grid is an elapsed
            # day. Calendar offsets can also be represented on their stable
            # wall clock or UTC clock, with public coordinates carried separately.
            clocks = [
                (dated.tz_localize(None), freq),
                (dated.tz_convert("UTC"), freq),
            ]
            if isinstance(freq, pd.offsets.Day):
                clocks.insert(0, (dated, pd.offsets.Hour(24 * freq.n)))
            for clock, offset in clocks:
                try:
                    return pd.DatetimeIndex(
                        pd.DatetimeIndex(clock, freq=None), freq=offset
                    )
                except ValueError:
                    continue
    elif pd.api.types.is_integer_dtype(index.dtype):
        start, second, last = map(int, (index[0], index[1], index[-1]))
        step = second - start
        if step > 0:
            candidate = pd.RangeIndex(start, last + step, step, name=index.name)
            if candidate.equals(index):
                return candidate
    raise BadIndexException(
        "SkforecastModel requires a regular datetime or integer grid without gaps."
    )


def _exog(X: xr.DataArray, index: pd.Index) -> pd.DataFrame | None:
    original = X.get_index("obs_ind")
    if (
        isinstance(original, pd.DatetimeIndex)
        and isinstance(index, pd.DatetimeIndex)
        and original.tz is not None
    ):
        original = (
            original.tz_localize(None)
            if index.tz is None
            else original.tz_convert(index.tz)
        )
    if not original.equals(index):
        raise BadIndexException(
            "Observations and exogenous inputs must align to the regular forecast grid."
        )
    frame = X.to_pandas().set_axis(index)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("SkforecastModel requires finite exogenous inputs.")
    return None if frame.empty else frame


class _SkforecastAdapter(ModelAdapter):
    def __init__(self, model: SkforecastModel) -> None:
        _validate_recursive_forecaster(model.forecaster)
        # Snapshot the configuration at installation, like the sklearn adapter.
        self._model = deepcopy(model)
        self._fit_inputs: (
            tuple[xr.DataArray, xr.DataArray, dict[str, Any] | None] | None
        ) = None
        self._forecaster: Any = None
        self._pre_prediction: xr.DataArray
        self._last_window: pd.Series
        self._score: pd.Series
        self._calibration_count: int
        self._residual_pool_count: int
        self._calibration_degenerate: bool

    @property
    def model(self) -> SkforecastModel:
        return self._model

    @property
    def kind(self) -> Literal["skforecast"]:
        return "skforecast"

    @property
    def calibration_size(self) -> int:
        return self._model.calibration_size

    @property
    def idata(self) -> None:
        return None

    @property
    def is_built(self) -> bool:
        return self._fit_inputs is not None

    @property
    def has_posterior(self) -> bool:
        return self._forecaster is not None

    def build(
        self, X: xr.DataArray, y: xr.DataArray, *, coords: dict[str, Any] | None = None
    ) -> None:
        if self._fit_inputs is None:
            self._fit_inputs = (X, y, coords)

    def sample_posterior(self, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError("Set the bootstrap controls on SkforecastModel.")
        assert self._fit_inputs is not None
        X, y, coords = self._fit_inputs
        self.fit(X, y, coords=coords)

    def fit(
        self, X: xr.DataArray, y: xr.DataArray, *, coords: dict[str, Any] | None = None
    ) -> None:
        from skforecast.exceptions import ResidualsUsageWarning

        forecaster = deepcopy(self._model.forecaster)
        training_size = y.sizes["obs_ind"] - self.calibration_size
        window_size = forecaster.window_size
        if training_size <= window_size:
            raise ValueError(
                "The training prefix must have more rows than the forecaster's "
                f"window_size ({window_size}) after reserving calibration_size "
                f"({self.calibration_size}) rows."
            )
        pre_index = y.get_index("obs_ind")
        full_index = (coords or {}).get("forecast_index", pre_index)
        index = _regular_index(full_index, len(pre_index), window_size)[
            : len(pre_index)
        ]
        outcome = pd.Series(y.values[:, 0], index=index, name="y")
        if not np.isfinite(outcome).all():
            raise ValueError(
                "SkforecastModel requires finite pre-intervention outcomes."
            )
        exog = _exog(X, index)
        forecaster.fit(
            y=outcome.iloc[:training_size],
            exog=None if exog is None else exog.iloc[:training_size],
        )

        predictions = pd.Series(np.nan, index=index)
        for i in range(window_size, len(outcome)):
            prediction = forecaster.predict(
                steps=1,
                last_window=outcome.iloc[i - window_size : i],
                exog=None if exog is None else exog.iloc[i : i + 1],
            )
            if not prediction.index.equals(index[i : i + 1]):
                raise BadIndexException(
                    "One-step prediction is misaligned to its observation."
                )
            predictions.iloc[i] = prediction.iloc[0]

        actual = outcome.iloc[training_size:]
        predicted = predictions.iloc[training_size:]
        finite = np.isfinite(actual - predicted)
        actual, predicted = actual[finite], predicted[finite]
        if len(actual) < 2:
            raise ValueError(
                "SkforecastModel requires at least two finite calibration errors."
            )
        # The upstream setter also bins errors. Sparse-bin warnings are immaterial
        # because the simulation only uses the pooled residual distribution.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ResidualsUsageWarning)
            forecaster.set_out_sample_residuals(
                y_true=actual, y_pred=predicted, random_state=self._model.random_state
            )
        residuals = forecaster.out_sample_residuals_
        if residuals.size < 2 or not np.isfinite(residuals).all():
            raise ValueError(
                "The forecaster must store at least two finite calibration errors."
            )
        degenerate = bool(np.all(residuals == residuals[0]))
        if degenerate:
            warnings.warn(
                "The held-out calibration errors are degenerate. Every simulation "
                "will use the same innovation.",
                UserWarning,
                stacklevel=2,
            )
        self._pre_prediction = y.copy(data=predictions.to_numpy()[:, None]).expand_dims(
            chain=[0], draw=[0]
        )
        self._last_window = outcome.iloc[-window_size:]
        self._score = pd.Series({"unit_0_r2": r2_score(actual, predicted)})
        self._calibration_count = len(actual)
        self._residual_pool_count = residuals.size
        self._calibration_degenerate = degenerate
        self._forecaster = forecaster

    def predict(
        self,
        X: xr.DataArray,
        *,
        coords: dict[str, Any] | None = None,
        out_of_sample: bool = False,
        group: Literal["prior", "posterior"] = "posterior",
    ) -> xr.DataArray:
        if out_of_sample:
            return self.predict_counterfactual(X, group=group)[0]
        self._require_fitted(group)
        return self._pre_prediction

    def _require_fitted(self, group: Literal["prior", "posterior"]) -> None:
        if group != "posterior" or self._forecaster is None:
            raise GroupNotSampledException(
                "SkforecastModel requires a successful fit() and has no prior draws.",
                group=group,
            )

    def predict_counterfactual(
        self, X: xr.DataArray, *, group: Literal["prior", "posterior"]
    ) -> tuple[xr.DataArray, xr.DataArray]:
        from skforecast.utils import expand_index

        self._require_fitted(group)
        index = X.get_index("obs_ind")
        future_grid = expand_index(self._last_window.index, steps=len(index))
        paths = self._forecaster.predict_bootstrapping(
            steps=len(index),
            last_window=self._last_window,
            exog=_exog(X, future_grid),
            n_boot=self._model.n_boot,
            random_state=self._model.random_state,
            use_in_sample_residuals=False,
            use_binned_residuals=False,
        )
        if not paths.index.equals(future_grid):
            raise BadIndexException(
                "Simulated paths are misaligned to the observation grid."
            )
        values = paths.to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(
                "SkforecastModel produced non-finite counterfactual paths."
            )
        trajectories = xr.DataArray(
            values.T[:, :, None],
            dims=("simulation", "obs_ind", "treated_units"),
            coords={
                "simulation": np.arange(self._model.n_boot),
                "obs_ind": index,
                "treated_units": self._pre_prediction.treated_units,
            },
            attrs={
                "calibration_size": self.calibration_size,
                "calibration_count": self._calibration_count,
                "residual_pool_count": self._residual_pool_count,
                "calibration_degenerate": self._calibration_degenerate,
                "forecaster": type(self._forecaster).__name__,
                "estimator": type(self._forecaster.estimator).__name__,
            },
        )
        point = trajectories.mean("simulation", keep_attrs=False).expand_dims(
            chain=[0], draw=[0]
        )
        return point, trajectories

    def score(
        self, X: Any, y: Any, *, coords: dict[str, Any] | None = None
    ) -> pd.Series:
        self._require_fitted("posterior")
        return self._score

    def coefficients(
        self, *, group: Literal["prior", "posterior"] = "posterior"
    ) -> xr.DataArray:
        raise NotImplementedError(
            "SkforecastModel has no CausalPy design-matrix coefficient table."
        )
