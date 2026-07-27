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
Patsy stateful transforms for Piecewise Interrupted Time Series analysis.

This module provides `step` and `ramp` transforms for use in patsy formulas,
enabling flexible specification of level and slope changes at intervention points.
It also provides the internal `elapsed` transform used to represent bare datetime
predictors as continuous elapsed days; use ``C(date)`` for categorical date effects.

Examples
--------
>>> import causalpy as cp
>>> # Numeric time with level and slope change at t=50
>>> formula = "y ~ 1 + t + step(t, 50) + ramp(t, 50)"

>>> # Datetime time with intervention
>>> formula = "y ~ 1 + date + step(date, '2020-06-01') + ramp(date, '2020-06-01')"

>>> # Different effects per intervention
>>> formula = "y ~ 1 + t + step(t, 50) + step(t, 100) + ramp(t, 100)"
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import patsy


class StepTransform:
    """
    Stateful transform for step function (level change) at threshold.

    Creates a binary indicator: 1 if time >= threshold, 0 otherwise.

    Works with both numeric and datetime time columns. For datetime,
    the threshold can be specified as a string ('2020-01-01') or
    pd.Timestamp.

    The transform is "stateful" because it remembers the datetime origin
    from the training data, ensuring consistent behavior when predicting
    on new data.

    Notes
    -----
    Per the patsy stateful transform protocol, ``x`` and ``threshold`` are
    supplied to :meth:`memorize_chunk` and :meth:`transform` rather than to
    the constructor; see those methods for parameter details.

    Examples
    --------
    >>> # Numeric time
    >>> formula = "y ~ 1 + t + step(t, 50)"

    >>> # Datetime time with string threshold
    >>> formula = "y ~ 1 + date + step(date, '2020-06-01')"

    >>> # Datetime time with Timestamp threshold
    >>> formula = "y ~ 1 + date + step(date, pd.Timestamp('2020-06-01'))"
    """

    def __init__(self) -> None:
        self._is_datetime: bool = False
        self._origin: pd.Timestamp | None = None

    def _is_datetime_like(self, x: Any) -> bool:
        """Check if x is datetime-like."""
        return (
            pd.api.types.is_datetime64_any_dtype(x)
            or isinstance(x, pd.DatetimeIndex)
            or (hasattr(x, "dtype") and pd.api.types.is_datetime64_any_dtype(x.dtype))
        )

    def memorize_chunk(
        self, x: Any, threshold: int | float | str | pd.Timestamp
    ) -> None:
        """
        Detect datetime and store origin during patsy's first pass.

        Parameters
        ----------
        x : array-like
            Time values (numeric or datetime).
        threshold : int, float, str, or pd.Timestamp
            The intervention time. For datetime ``x`` it may be a string
            like ``'2020-01-01'`` or a :class:`pd.Timestamp`.
        """
        if self._is_datetime_like(x):
            self._is_datetime = True
            x_dt = pd.to_datetime(x)
            x_min: pd.Timestamp = pd.Timestamp(x_dt.min())  # type: ignore[assignment]
            if self._origin is None:
                self._origin = x_min
            else:
                # Handle chunked data - keep the overall minimum
                self._origin = min(self._origin, x_min)  # type: ignore[assignment]

    def memorize_finish(self) -> None:
        """Called after all chunks processed - finalize state."""
        pass

    def transform(
        self, x: Any, threshold: int | float | str | pd.Timestamp
    ) -> np.ndarray:
        """
        Transform ``x`` into step function values.

        Parameters
        ----------
        x : array-like
            Time values (numeric or datetime).
        threshold : int, float, str, or pd.Timestamp
            The intervention time, in the same domain as ``x``.

        Returns
        -------
        np.ndarray
            Binary indicator with 1 where ``x >= threshold`` and 0 elsewhere.
        """
        if self._is_datetime and self._origin is not None:
            # Convert x to days from origin
            x_dt = pd.to_datetime(x)
            # Handle both DatetimeIndex and Series
            if isinstance(x_dt, pd.DatetimeIndex):
                x_numeric: np.ndarray = np.asarray(
                    (x_dt - self._origin).total_seconds() / (24 * 3600)
                )
            else:
                x_numeric = np.asarray(
                    (x_dt - self._origin).dt.total_seconds() / (24 * 3600)
                )

            # Convert threshold to days from origin
            threshold_dt = self._parse_threshold(threshold)
            t_numeric = (threshold_dt - self._origin).total_seconds() / (24 * 3600)
        else:
            x_numeric = np.asarray(x, dtype=float)
            t_numeric = float(threshold)  # type: ignore[arg-type]

        return (x_numeric >= t_numeric).astype(float)

    def _parse_threshold(
        self, threshold: int | float | str | pd.Timestamp
    ) -> pd.Timestamp:
        """Parse threshold to pd.Timestamp, handling various input types."""
        if isinstance(threshold, pd.Timestamp):
            return threshold
        else:
            # Assume it's something pandas can convert (str or numeric)
            return pd.Timestamp(threshold)  # type: ignore[arg-type, return-value]


class RampTransform:
    """
    Stateful transform for ramp function (slope change) at threshold.

    Creates a ramp: max(0, time - threshold). For datetime, the ramp
    values are in days.

    Works with both numeric and datetime time columns. For datetime,
    the threshold can be specified as a string ('2020-01-01') or
    pd.Timestamp.

    Notes
    -----
    Per the patsy stateful transform protocol, ``x`` and ``threshold`` are
    supplied to :meth:`memorize_chunk` and :meth:`transform` rather than to
    the constructor; see those methods for parameter details.

    For datetime inputs, the ramp values represent days since the threshold.
    This means the slope coefficient will be interpreted as "change per day".

    Examples
    --------
    >>> # Numeric time - ramp is in same units as t
    >>> formula = "y ~ 1 + t + ramp(t, 50)"

    >>> # Datetime time - ramp is in DAYS
    >>> formula = "y ~ 1 + date + ramp(date, '2020-06-01')"
    """

    def __init__(self) -> None:
        self._is_datetime: bool = False
        self._origin: pd.Timestamp | None = None

    def _is_datetime_like(self, x: Any) -> bool:
        """Check if x is datetime-like."""
        return (
            pd.api.types.is_datetime64_any_dtype(x)
            or isinstance(x, pd.DatetimeIndex)
            or (hasattr(x, "dtype") and pd.api.types.is_datetime64_any_dtype(x.dtype))
        )

    def memorize_chunk(
        self, x: Any, threshold: int | float | str | pd.Timestamp
    ) -> None:
        """
        Detect datetime and store origin during patsy's first pass.

        Parameters
        ----------
        x : array-like
            Time values (numeric or datetime).
        threshold : int, float, str, or pd.Timestamp
            The intervention time. For datetime ``x`` it may be a string
            like ``'2020-01-01'`` or a :class:`pd.Timestamp`.
        """
        if self._is_datetime_like(x):
            self._is_datetime = True
            x_dt = pd.to_datetime(x)
            x_min: pd.Timestamp = pd.Timestamp(x_dt.min())  # type: ignore[assignment]
            if self._origin is None:
                self._origin = x_min
            else:
                self._origin = min(self._origin, x_min)  # type: ignore[assignment]

    def memorize_finish(self) -> None:
        """Called after all chunks processed."""
        pass

    def transform(
        self, x: Any, threshold: int | float | str | pd.Timestamp
    ) -> np.ndarray:
        """
        Transform ``x`` into ramp function values.

        Parameters
        ----------
        x : array-like
            Time values (numeric or datetime).
        threshold : int, float, str, or pd.Timestamp
            The intervention time, in the same domain as ``x``.

        Returns
        -------
        np.ndarray
            Ramp values ``max(0, x - threshold)``. For datetime inputs, the
            difference is expressed in days.
        """
        if self._is_datetime and self._origin is not None:
            # Convert x to days from origin
            x_dt = pd.to_datetime(x)
            # Handle both DatetimeIndex and Series
            if isinstance(x_dt, pd.DatetimeIndex):
                x_numeric: np.ndarray = np.asarray(
                    (x_dt - self._origin).total_seconds() / (24 * 3600)
                )
            else:
                x_numeric = np.asarray(
                    (x_dt - self._origin).dt.total_seconds() / (24 * 3600)
                )

            # Convert threshold to days from origin
            threshold_dt = self._parse_threshold(threshold)
            t_numeric = (threshold_dt - self._origin).total_seconds() / (24 * 3600)
        else:
            x_numeric = np.asarray(x, dtype=float)
            t_numeric = float(threshold)  # type: ignore[arg-type]

        return np.maximum(0.0, x_numeric - t_numeric)

    def _parse_threshold(
        self, threshold: int | float | str | pd.Timestamp
    ) -> pd.Timestamp:
        """Parse threshold to pd.Timestamp."""
        if isinstance(threshold, pd.Timestamp):
            return threshold
        else:
            # Assume it's something pandas can convert (str or numeric)
            return pd.Timestamp(threshold)  # type: ignore[arg-type, return-value]


class ElapsedDaysTransform:
    """Stateful transform that represents datetimes as days since the fitted origin."""

    def __init__(self) -> None:
        self._origin: pd.Timestamp | None = None

    def memorize_chunk(self, x: Any) -> None:
        """Store the earliest datetime encountered during Patsy's fitting pass.

        Parameters
        ----------
        x : array-like
            Datetime values from a Patsy fitting chunk.
        """
        x_dt = pd.to_datetime(x)
        x_min = pd.Timestamp(x_dt.min())
        if self._origin is None:
            self._origin = x_min
        else:
            self._origin = min(self._origin, x_min)

    def memorize_finish(self) -> None:
        """Called after all chunks processed."""
        pass

    def transform(self, x: Any) -> np.ndarray:
        """Return elapsed days from the fitted origin.

        Parameters
        ----------
        x : array-like
            Datetime values to encode.
        """
        if self._origin is None:
            raise RuntimeError("elapsed() was used before its origin was initialized.")
        x_dt = pd.to_datetime(x)
        if isinstance(x_dt, pd.DatetimeIndex):
            return np.asarray((x_dt - self._origin).total_seconds() / (24 * 3600))
        return np.asarray((x_dt - self._origin).dt.total_seconds() / (24 * 3600))


# Create callable stateful transforms for use in formulas
step = patsy.stateful_transform(StepTransform)  # type: ignore[attr-defined]
ramp = patsy.stateful_transform(RampTransform)  # type: ignore[attr-defined]
elapsed = patsy.stateful_transform(ElapsedDaysTransform)  # type: ignore[attr-defined]

__all__ = [
    "step",
    "ramp",
    "elapsed",
    "StepTransform",
    "RampTransform",
    "ElapsedDaysTransform",
]
