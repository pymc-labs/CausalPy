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
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
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


def _same_labels(left: pd.Index, right: pd.Index) -> bool:
    """Compare ordered observation labels independently of pandas storage dtype."""
    return len(left) == len(right) and bool((left == right).all())


def _native_calendar_matches(
    index: pd.DatetimeIndex, pre_size: int, window_size: int
) -> bool:
    from pytz.exceptions import InvalidTimeError
    from skforecast.utils import expand_index

    # Some timezone-aware calendar grids are interpreted differently by
    # skforecast depending on which timestamps its lag window contains.
    try:
        for end in range(window_size, pre_size + 1):
            stop = end + 1 if end < pre_size else len(index)
            if not _same_labels(
                expand_index(index[end - window_size : end], stop - end),
                index[end:stop],
            ):
                return False
    except InvalidTimeError:
        # Native expansion can reach a nonexistent or ambiguous wall time
        # even when the supplied observations follow a regular grid.
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
                candidate = pd.date_range(
                    start=clock[0], periods=len(clock), freq=offset, name=clock.name
                )
                if _same_labels(candidate, clock):
                    return candidate
    elif pd.api.types.is_integer_dtype(index.dtype):
        start, second, last = map(int, (index[0], index[1], index[-1]))
        step = second - start
        if step > 0:
            candidate = pd.RangeIndex(start, last + step, step, name=index.name)
            if _same_labels(candidate, index):
                return candidate
    raise BadIndexException(
        "SkforecastModel requires a regular datetime or integer grid without gaps."
    )


def _exog(X: xr.DataArray, observed: pd.Index, grid: pd.Index) -> pd.DataFrame | None:
    if not _same_labels(X.get_index("obs_ind"), observed):
        raise BadIndexException(
            "Observations and exogenous inputs must align to the regular forecast grid."
        )
    frame = X.to_pandas().set_axis(grid)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("SkforecastModel requires finite exogenous inputs.")
    return None if frame.empty else frame


def _translate_index(
    index: pd.Index, *, source: pd.Index, target: pd.Index
) -> pd.Index:
    positions = source.get_indexer(index)
    if (positions < 0).any():
        raise BadIndexException("Model dates must align to the observed forecast grid.")
    return target.take(positions)


class _PreprocessorOnObservedIndex:
    """Translate pandas labels at the native preprocessing boundary."""

    def __init__(
        self,
        preprocessor: Any,
        to_observed: Callable[[pd.Index], pd.Index] | None,
        output: Callable[[Any, pd.Index, pd.Index], Any],
    ) -> None:
        self._preprocessor = preprocessor
        self._to_observed = to_observed
        self._output = output

    def _apply(self, method: Callable, data: Any) -> Any:
        if not isinstance(data, (pd.Index, pd.Series, pd.DataFrame)):
            return method(data)
        index_only = isinstance(data, pd.Index)
        model_index = data if index_only else data.index
        observed_index = model_index
        if self._to_observed is not None:
            observed_index = self._to_observed(model_index)
            data = observed_index if index_only else data.set_axis(observed_index)
        transformed = method(data)
        if isinstance(transformed, (pd.Series, pd.DataFrame)):
            return self._output(transformed, observed_index, model_index)
        return transformed

    def fit_transform(self, data: Any) -> Any:
        return self._apply(self._preprocessor.fit_transform, data)

    def transform(self, data: Any) -> Any:
        return self._apply(self._preprocessor.transform, data)

    def transform_batch(self, data: Any) -> Any:
        return self._apply(self._preprocessor.transform_batch, data)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_preprocessor"), name)


def _align_transformed_rows(
    data: Any, observed_index: pd.Index, model_index: pd.Index, *, name: str
) -> Any:
    index = data.index
    positional = (
        isinstance(index, pd.RangeIndex)
        and index.name is None
        and len(index) == len(observed_index)
        and index.equals(pd.RangeIndex(len(data)))
    )
    if not positional and not _same_labels(index, observed_index):
        raise BadIndexException(
            f"Transformed {name} must keep their observation labels in order."
        )
    return data.set_axis(model_index)


def _configure_native_preprocessors(
    forecaster: Any, grid: pd.Index, observed: pd.Index
) -> None:
    """Keep caller timestamps at the native preprocessing boundary."""
    to_observed = None
    if not grid.equals(observed):
        to_observed = partial(_translate_index, source=grid, target=observed)

        def on_model_grid(
            data: Any, _observed_index: pd.Index, _model_index: pd.Index
        ) -> Any:
            return data.set_axis(
                _translate_index(data.index, source=observed, target=grid)
            )

        if forecaster.window_features is not None:
            forecaster.window_features = [
                _PreprocessorOnObservedIndex(features, to_observed, on_model_grid)
                for features in forecaster.window_features
            ]
        if forecaster.weight_func is not None:
            weight_func = forecaster.weight_func

            def observed_weights(index: pd.Index) -> np.ndarray:
                return weight_func(to_observed(index))

            forecaster.weight_func = observed_weights
    for attribute, name in (
        ("calendar_features", "calendar features"),
        ("transformer_y", "outcomes"),
        ("transformer_exog", "exogenous covariates"),
    ):
        preprocessor = getattr(forecaster, attribute)
        if preprocessor is not None:
            setattr(
                forecaster,
                attribute,
                _PreprocessorOnObservedIndex(
                    preprocessor,
                    to_observed,
                    partial(_align_transformed_rows, name=name),
                ),
            )


@dataclass(frozen=True, slots=True)
class _CalibratedForecast:
    forecaster: Any
    predictions_pre: xr.DataArray
    last_window: pd.Series
    future_index: pd.Index
    future_grid: pd.Index
    score: pd.Series
    calibration_count: int
    residual_pool_count: int
    calibration_degenerate: bool


class _SkforecastAdapter(ModelAdapter):
    def __init__(self, model: SkforecastModel) -> None:
        _validate_recursive_forecaster(model.forecaster)
        # Snapshot the configuration at installation, like the sklearn adapter.
        self._model = deepcopy(model)
        self._fit_inputs: (
            tuple[xr.DataArray, xr.DataArray, dict[str, Any] | None] | None
        ) = None
        self._fitted: _CalibratedForecast | None = None

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
        return self._fitted is not None

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
        pre_size = y.sizes["obs_ind"]
        training_size = pre_size - self.calibration_size
        window_size = forecaster.window_size
        if training_size <= window_size:
            raise ValueError(
                "The training prefix must have more rows than the forecaster's "
                f"window_size ({window_size}) after reserving calibration_size "
                f"({self.calibration_size}) rows."
            )
        full_index = (coords or {}).get("forecast_index")
        if not isinstance(full_index, pd.Index):
            raise ValueError("SkforecastModel requires the full ITS observation index.")
        grid = _regular_index(full_index, pre_size, window_size)
        _configure_native_preprocessors(forecaster, grid, full_index)
        index = grid[:pre_size]
        outcome = pd.Series(
            y.values[:, 0], index=index, name=(coords or {}).get("outcome_name", "y")
        )
        if not np.isfinite(outcome).all():
            raise ValueError(
                "SkforecastModel requires finite pre-intervention outcomes."
            )
        exog = _exog(X, full_index[:pre_size], index)
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
            if not _same_labels(prediction.index, index[i : i + 1]):
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
        predictions_pre = y.copy(data=predictions.to_numpy()[:, None]).expand_dims(
            chain=[0], draw=[0]
        )
        self._fitted = _CalibratedForecast(
            forecaster=forecaster,
            predictions_pre=predictions_pre,
            last_window=outcome.iloc[-window_size:],
            future_index=full_index[pre_size:],
            future_grid=grid[pre_size:],
            score=pd.Series({"unit_0_r2": r2_score(actual, predicted)}),
            calibration_count=len(actual),
            residual_pool_count=residuals.size,
            calibration_degenerate=degenerate,
        )

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
        return self._require_fitted(group).predictions_pre

    def _require_fitted(
        self, group: Literal["prior", "posterior"]
    ) -> _CalibratedForecast:
        if group != "posterior" or self._fitted is None:
            raise GroupNotSampledException(
                "SkforecastModel requires a successful fit() and has no prior draws.",
                group=group,
            )
        return self._fitted

    def predict_counterfactual(
        self, X: xr.DataArray, *, group: Literal["prior", "posterior"]
    ) -> tuple[xr.DataArray, xr.DataArray]:
        fitted = self._require_fitted(group)
        paths = fitted.forecaster.predict_bootstrapping(
            steps=len(fitted.future_index),
            last_window=fitted.last_window,
            exog=_exog(X, fitted.future_index, fitted.future_grid),
            n_boot=self._model.n_boot,
            random_state=self._model.random_state,
            use_in_sample_residuals=False,
            use_binned_residuals=False,
        )
        if not _same_labels(paths.index, fitted.future_grid):
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
                "obs_ind": fitted.future_index,
                "treated_units": fitted.predictions_pre.treated_units,
            },
            attrs={
                "calibration_size": self.calibration_size,
                "calibration_count": fitted.calibration_count,
                "residual_pool_count": fitted.residual_pool_count,
                "calibration_degenerate": fitted.calibration_degenerate,
                "forecaster": type(fitted.forecaster).__name__,
                "estimator": type(fitted.forecaster.estimator).__name__,
            },
        )
        point = trajectories.mean("simulation", keep_attrs=False).expand_dims(
            chain=[0], draw=[0]
        )
        return point, trajectories

    def score(
        self, X: Any, y: Any, *, coords: dict[str, Any] | None = None
    ) -> pd.Series:
        return self._require_fitted("posterior").score

    def coefficients(
        self, *, group: Literal["prior", "posterior"] = "posterior"
    ) -> xr.DataArray:
        raise NotImplementedError(
            "SkforecastModel has no CausalPy design-matrix coefficient table."
        )
