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
"""Historical schedules, zero-lift replay, and the calibration seam.

:class:`~causalpy.design.SyntheticControlPlanner` fits one horizon and
translates that fit. It does not replay a history or feed operating
characteristics. This module is that next seam. It does not change the planner
contract, and it is not :class:`~causalpy.checks.placebo_in_time.PlaceboInTime`.

Placebo-in-time window planning stays where it is. That planner is coupled to
one experiment's treatment time, a calibrate mode, and optional random
selection. Extracting it would risk changing public fold geometry, so historical
schedules are a separate contract: common origins, explicit horizons, selection
origins kept apart from later audit origins, exclusions, cadence, and a
reject-missing policy.

The learned-null fit is not copied. When a calibration record is defensible it
calls :func:`causalpy.checks.placebo_in_time._fit_learned_null`, the same
hierarchical model PlaceboInTime already uses. Operating characteristics stay
the only probability engine. A calibration record can feed that engine. It is
never marked validated.

Ferman and Pinto (2021) and Abadie (2021) explain why an imperfect pretreatment
fit can leave a bias that more windows do not remove. This module reports that
bias. It does not correct it, and those papers are not acceptance thresholds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np
import pandas as pd

from causalpy._arviz_compat import hdi_bounds
from causalpy.checks.placebo_in_time import (
    _fit_learned_null,
    _NullModelUnidentifiedError,
)
from causalpy.constants import HDI_PROB
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.input_data import to_pandas_with_time_index
from causalpy.pymc_models import WeightedSumFitter

if TYPE_CHECKING:
    from causalpy.design import DecisionRule

__all__ = [
    "CalibrationRecord",
    "EvidenceLevel",
    "HISTORICAL_REPLAY_ASSUMPTIONS",
    "HistoricalReplay",
    "HistoricalSchedule",
    "ScheduleCounts",
    "ScheduleRequest",
    "WindowPlan",
    "WindowReplay",
    "replay_history",
    "validate_schedule",
]

EvidenceLevel = Literal[
    "historical-descriptive",
    "model-conditional",
    "inconclusive",
    "insufficient-history",
    "fit-failed",
]
WindowRole = Literal["selection", "audit"]
WindowStatus = Literal["usable", "incomplete", "missing"]
ReplayStatus = Literal["scored", "fit-failed", "not-run", "reserved"]

HORIZON_POLICY = "separate-fits"
TRAINING_POLICY = "pre-origin-only"
MISSING_DATA_POLICY = "reject"
DONOR_POLICY = "donors-through-window-end"
_DEFAULT_MIN_TRAINING = 2
_LEARNED_NULL_PRIOR_SCALE = 1.0
_MORE_WINDOWS = (
    "More windows do not eliminate bias. They can make a stable bias visible; "
    "they are not a correction."
)

HISTORICAL_REPLAY_ASSUMPTIONS: tuple[str, ...] = (
    "The learned-null model is a Normal approximation conditional on the "
    "supplied fold means and fold standard deviations, not an empirical "
    "false-positive rate.",
    "Imperfect pretreatment fit can leave a bias that does not vanish as more "
    "windows are added (Ferman and Pinto, 2021; Abadie, 2021). This report "
    "does not correct that bias.",
    "Donor outcomes inside the analysis window are used to predict the "
    "counterfactual. Held-out treated outcomes are scoring inputs only.",
    "Data outside the analysis window is not passed to the estimator.",
    "Overlapping windows and shared training rows are dependent. MCMC draws "
    "measure computation variability, not independent experiments.",
    "The same caller seed is reused for every window, so sampling noise is "
    "shared rather than independent-experiment noise.",
    "Prefix reuse across horizons is disabled. Each origin and horizon is a "
    "separate fit, and a shorter horizon does not see a longer horizon's "
    "extra donor rows.",
    "No automatic bias correction, detrending, or noise injection is applied.",
)


def _require_python_int(name: str, value: object, *, minimum: int) -> int:
    """Return a Python int at or above ``minimum``, rejecting numpy integers."""
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int, got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}.")
    return value


def _require_seed(random_seed: object) -> int | None:
    """Return a Python seed, or None."""
    if random_seed is None:
        return None
    if type(random_seed) is not int:
        raise TypeError(
            "random_seed must be a Python int or None, "
            f"got {type(random_seed).__name__}."
        )
    return random_seed


def _copy_index(index: pd.Index) -> pd.Index:
    """Return an owned index so a later caller mutation cannot move the plan."""
    return index.copy()


def _empty_index_like(index: pd.Index) -> pd.Index:
    """Return an empty index of the same class as ``index``."""
    return index[:0].copy()


def _label_position(index: pd.Index, label: Any) -> int | None:
    """Return the unique position of ``label``, or None when it is absent."""
    try:
        position = index.get_indexer([label])[0]
    except (TypeError, ValueError, KeyError):
        return None
    if int(position) < 0:
        return None
    return int(position)


def _same_labels(left: pd.Index, right: pd.Index) -> bool:
    """Compare labels by value, including numpy integers and timestamps."""
    if len(left) != len(right):
        return False
    return all(a == b for a, b in zip(left, right, strict=True))


def _finite_block(
    frame: pd.DataFrame, labels: pd.Index, columns: list[str]
) -> pd.Index:
    """Return analysis labels whose outcome block is not finite."""
    if len(labels) == 0:
        return labels
    values = frame.loc[list(labels), columns].to_numpy(dtype=float)
    bad_rows = ~np.isfinite(values).all(axis=1)
    return labels[bad_rows]


@dataclass(frozen=True)
class ScheduleRequest:
    """Validated historical-schedule contract.

    Selection origins and audit origins are separate. Audit origins must sit
    strictly after every selection origin in the history index; an earlier
    audit is not untouched. Horizons are Python ints because each origin and
    horizon is a separate fit. ``np.int64`` is rejected so a duration cannot
    silently take a second convention from the planner's known limitation.

    ``min_training_periods`` defaults to 2 to match
    :class:`~causalpy.design.SyntheticControlPlanner`. That floor is not a
    universal statistical minimum.

    Attributes
    ----------
    selection_origins : tuple
        Origins used to score bias and, when defensible, to feed the
        learned-null model. Required and unique.
    horizons : tuple of int
        Positive unique Python ints. Each is fitted separately. Prefix reuse
        is not available.
    audit_origins : tuple
        Later origins reserved as an untouched audit. They are planned and
        not fitted unless replay is asked to fit them. Fitting an audit
        origin means it is no longer untouched.
    exclusions : tuple
        Labels removed from training and holdout. A holdout exclusion makes
        that window incomplete. Unknown labels are a contract error.
    cadence : int, str, pandas offset, or None
        ``None`` takes the next ``horizon`` index labels and does not check
        calendar gaps. An int cadence expects ``origin + i * cadence`` on a
        numeric index. A date frequency expects each generated timestamp to
        be present. Gaps are recorded, not imputed.
    min_training_periods : int
        Minimum pre-origin rows after exclusions. Default 2.
    missing_data : {'reject'}
        The only supported policy. Non-finite outcomes mark the window
        incomplete and are not imputed.
    training_policy : {'pre-origin-only'}
        The only supported policy. Training is every non-excluded observation
        strictly before the origin. It is an expanding prefix, not a rolling
        window, so later origins share earlier training history.
    """

    selection_origins: tuple[Any, ...]
    horizons: tuple[int, ...]
    audit_origins: tuple[Any, ...] = ()
    exclusions: tuple[Any, ...] = ()
    cadence: int | str | pd.DateOffset | pd.Timedelta | None = None
    min_training_periods: int = _DEFAULT_MIN_TRAINING
    missing_data: Literal["reject"] = "reject"
    training_policy: Literal["pre-origin-only"] = "pre-origin-only"

    def __post_init__(self) -> None:
        """Reject policies and durations this seam does not implement."""
        selection = tuple(self.selection_origins)
        audit = tuple(self.audit_origins)
        exclusions = tuple(self.exclusions)
        if not selection:
            raise ValueError("selection_origins must name at least one origin.")
        if len(selection) != len(set(map(repr, selection))):
            raise ValueError("selection_origins must be unique.")
        if len(audit) != len(set(map(repr, audit))):
            raise ValueError("audit_origins must be unique.")
        overlap = set(map(repr, selection)) & set(map(repr, audit))
        if overlap:
            raise ValueError(
                "an origin cannot be both a selection origin and an audit origin."
            )
        horizons = tuple(
            _require_python_int("horizon", horizon, minimum=1)
            for horizon in self.horizons
        )
        if not horizons:
            raise ValueError("horizons must name at least one duration.")
        if len(horizons) != len(set(horizons)):
            raise ValueError("horizons must be unique. Prefix reuse is not enabled.")
        minimum = _require_python_int(
            "min_training_periods", self.min_training_periods, minimum=1
        )
        if self.missing_data != "reject":
            raise ValueError(
                "missing_data must be 'reject'. Historical validation does not impute."
            )
        if self.training_policy != "pre-origin-only":
            raise ValueError(
                "training_policy must be 'pre-origin-only'. Other training "
                "windows are not supported."
            )
        _validate_cadence(self.cadence)
        object.__setattr__(self, "selection_origins", selection)
        object.__setattr__(self, "audit_origins", audit)
        object.__setattr__(self, "exclusions", exclusions)
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "min_training_periods", minimum)


def _validate_cadence(cadence: Any) -> None:
    """Reject cadence types this seam cannot turn into expected labels."""
    if cadence is None or isinstance(cadence, str):
        if isinstance(cadence, str) and not cadence:
            raise ValueError("cadence string must not be empty.")
        return
    if isinstance(cadence, bool):
        raise TypeError("cadence must not be a boolean.")
    if isinstance(cadence, int):
        if cadence < 1:
            raise ValueError(f"integer cadence must be at least 1, got {cadence}.")
        return
    if isinstance(cadence, (pd.DateOffset, pd.Timedelta)):
        return
    raise TypeError(
        "cadence must be None, a positive int, a date-frequency string, "
        "a pandas DateOffset, or a Timedelta, "
        f"got {type(cadence).__name__}."
    )


@dataclass(frozen=True)
class WindowPlan:
    """One origin and horizon, including failures.

    A window is never dropped. ``missing`` means the origin is absent.
    ``incomplete`` means the window was planned and cannot be scored: a short
    training prefix, a cadence gap, an exclusion in the holdout, a non-finite
    outcome, or a horizon that runs past the history. ``usable`` is the only
    status replay will fit.

    Attributes
    ----------
    role : {'selection', 'audit'}
        Selection windows may be scored. Audit windows are reserved unless
        replay is asked to fit them.
    origin : object
        First holdout label, as stored on the history index.
    horizon : int
        Number of holdout labels this window asked for.
    status : {'usable', 'incomplete', 'missing'}
        Whether the window can be scored.
    reasons : tuple of str
        Why the window is not usable. Empty when it is usable.
    training_index : pandas.Index
        Non-excluded labels strictly before the origin.
    holdout_index : pandas.Index
        Planned holdout labels that are present, including labels later
        removed by an exclusion. Empty when the origin is missing.
    excluded_training : pandas.Index
        Training labels removed by the exclusion list.
    excluded_holdout : pandas.Index
        Planned holdout labels removed by the exclusion list.
    analysis_index : pandas.Index
        Training plus holdout, with exclusions removed. This is the only
        index replay passes to the estimator.
    forbidden_index : pandas.Index
        History labels outside ``analysis_index``. Replay does not pass them
        to the estimator.
    training_count : int
        Rows in ``training_index``.
    holdout_count : int
        Planned present holdout labels, before exclusions.
    """

    role: WindowRole
    origin: Any
    horizon: int
    status: WindowStatus
    reasons: tuple[str, ...]
    training_index: pd.Index
    holdout_index: pd.Index
    excluded_training: pd.Index
    excluded_holdout: pd.Index
    analysis_index: pd.Index
    forbidden_index: pd.Index
    training_count: int
    holdout_count: int

    @property
    def reason(self) -> str | None:
        """Joined failure text, or None when the window is usable."""
        if not self.reasons:
            return None
        return "; ".join(self.reasons)


@dataclass(frozen=True)
class ScheduleCounts:
    """Requested, usable, incomplete, and missing window counts.

    Attributes
    ----------
    requested : int
        Selection and audit windows asked for. Equal to the length of the
        windows tuple. Nothing has been dropped.
    usable : int
        Windows that can be scored.
    incomplete : int
        Windows retained with a recorded reason.
    missing : int
        Windows whose origin is absent from the history.
    selection_requested : int
        Windows in the selection role.
    selection_usable : int
        Usable selection windows.
    audit_requested : int
        Windows in the audit role.
    audit_usable : int
        Usable audit windows. Usable does not mean fitted.
    """

    requested: int
    usable: int
    incomplete: int
    missing: int
    selection_requested: int
    selection_usable: int
    audit_requested: int
    audit_usable: int


@dataclass(frozen=True)
class HistoricalSchedule:
    """A validated schedule and the failures it retained.

    Attributes
    ----------
    request : ScheduleRequest
        The contract that produced this schedule.
    treated_unit : str
        The single treated unit.
    control_units : tuple of str
        Donor units. Replay must use the same units.
    history_index : pandas.Index
        Owned copy of the history index the windows were planned against.
    windows : tuple of WindowPlan
        Every requested origin and horizon, including failures.
    counts : ScheduleCounts
        Requested and usable counts. Failures stay in ``windows``.
    overlapping_pairs : tuple of tuple of int
        Positions of windows whose analysis indexes intersect. These are not
        independent experiments.
    shared_training_pairs : tuple of tuple of int
        Positions of windows whose training indexes intersect. Later origins
        share earlier training history under the pre-origin-only policy.
    audit_reserved : bool
        True when at least one audit origin was requested.
    selection_audit_separated : bool
        True when every requested origin is present and every audit origin
        sits strictly after every selection origin. False when an origin is
        missing, so the untouched-audit claim cannot be verified.
    assumptions : tuple of str
        The handoff assumptions. The same text is on
        :data:`HISTORICAL_REPLAY_ASSUMPTIONS`.
    """

    request: ScheduleRequest
    treated_unit: str
    control_units: tuple[str, ...]
    history_index: pd.Index
    windows: tuple[WindowPlan, ...]
    counts: ScheduleCounts
    overlapping_pairs: tuple[tuple[int, int], ...]
    shared_training_pairs: tuple[tuple[int, int], ...]
    audit_reserved: bool
    selection_audit_separated: bool
    assumptions: tuple[str, ...] = HISTORICAL_REPLAY_ASSUMPTIONS

    def exact_dates(self) -> pd.DataFrame:
        """Return origin, training, and holdout endpoints for every window.

        Missing endpoints are null. Incomplete windows are included.

        Returns
        -------
        pandas.DataFrame
            One row per requested window, in schedule order.
        """
        rows = [_window_date_row(window) for window in self.windows]
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        """Return the schedule contract as a table.

        Returns
        -------
        pandas.DataFrame
            Role, origin, horizon, status, reason, counts, and endpoints.
            Dependence counts are schedule attributes, not a per-row claim
            that the windows are independent.
        """
        rows = []
        for position, window in enumerate(self.windows):
            row = _window_date_row(window)
            row.update(
                {
                    "position": position,
                    "status": window.status,
                    "reason": window.reason,
                    "training_count": window.training_count,
                    "holdout_count": window.holdout_count,
                    "excluded_training": list(window.excluded_training),
                    "excluded_holdout": list(window.excluded_holdout),
                    "training_policy": self.request.training_policy,
                    "missing_data": self.request.missing_data,
                    "horizon_policy": HORIZON_POLICY,
                    "donor_policy": DONOR_POLICY,
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)


def _window_date_row(window: WindowPlan) -> dict[str, Any]:
    """Return endpoint fields for one window, with nulls when unknown."""
    return {
        "role": window.role,
        "origin": window.origin,
        "horizon": window.horizon,
        "training_start": _endpoint(window.training_index, 0),
        "training_end": _endpoint(window.training_index, -1),
        "holdout_start": _endpoint(window.holdout_index, 0),
        "holdout_end": _endpoint(window.holdout_index, -1),
    }


def _endpoint(index: pd.Index, position: int) -> Any:
    """Return one label, or null when the index is empty."""
    if len(index) == 0:
        return pd.NA
    return index[position]


def validate_schedule(
    history: pd.DataFrame,
    treated_unit: str,
    control_units: Sequence[str],
    request: ScheduleRequest,
    *,
    time_column: str | None = None,
) -> HistoricalSchedule:
    """Validate a historical schedule without fitting.

    Parameters
    ----------
    history : pandas.DataFrame
        Panel indexed by time, or a frame whose time axis is ``time_column``.
        The caller's frame is not mutated.
    treated_unit : str
        The single unit the decision concerns.
    control_units : sequence of str
        Donor units. Must be unique and must not include the treated unit.
    request : ScheduleRequest
        Origins, horizons, exclusions, cadence, and policies.
    time_column : str, optional
        Column to use as the time index. Passed through
        :func:`causalpy.input_data.to_pandas_with_time_index`.

    Returns
    -------
    HistoricalSchedule
        Every requested window, including incomplete and missing ones, plus
        exact endpoints and usable counts.

    Raises
    ------
    TypeError
        If a duration, cadence, or unit argument has an unsupported type.
    ValueError
        If the request mixes selection and audit, names an unknown exclusion,
        or asks for a cadence the index cannot support. An audit origin that
        is not strictly after every selection origin is also a contract error.
    DataException
        If the history index is missing, duplicated, or not sorted.
    """
    if not isinstance(request, ScheduleRequest):
        raise TypeError(
            f"request must be a ScheduleRequest, got {type(request).__name__}."
        )
    frame = _history_frame(history, time_column)
    treated, controls = _require_units(frame, treated_unit, control_units)
    _require_known_exclusions(frame.index, request.exclusions)
    _require_cadence_matches_index(frame.index, request.cadence)
    _require_audit_after_selection(frame.index, request)

    windows: list[WindowPlan] = []
    roles: tuple[tuple[WindowRole, tuple[Any, ...]], ...] = (
        ("selection", request.selection_origins),
        ("audit", request.audit_origins),
    )
    for role, origins in roles:
        for origin in origins:
            for horizon in request.horizons:
                windows.append(
                    _plan_window(
                        frame,
                        treated,
                        controls,
                        role=role,
                        origin=origin,
                        horizon=horizon,
                        request=request,
                    )
                )
    planned = tuple(windows)
    overlap, shared = _pair_positions(planned)
    present_selection = sum(
        window.role == "selection" and window.status != "missing" for window in planned
    )
    # An origin can produce several horizons. Separation fails if any requested
    # origin is absent, which shows up as a missing window.
    missing_origin = any(window.status == "missing" for window in planned)
    return HistoricalSchedule(
        request=request,
        treated_unit=treated,
        control_units=controls,
        history_index=_copy_index(frame.index),
        windows=planned,
        counts=_schedule_counts(planned),
        overlapping_pairs=overlap,
        shared_training_pairs=shared,
        audit_reserved=bool(request.audit_origins),
        selection_audit_separated=not missing_origin and present_selection > 0,
        assumptions=HISTORICAL_REPLAY_ASSUMPTIONS,
    )


def _history_frame(history: pd.DataFrame, time_column: str | None) -> pd.DataFrame:
    """Return an owned, sorted, unique time-indexed frame."""
    frame = to_pandas_with_time_index(history, time_column)
    frame.index = frame.index.copy()
    return frame


def _require_units(
    frame: pd.DataFrame, treated_unit: str, control_units: Sequence[str]
) -> tuple[str, tuple[str, ...]]:
    """Return validated unit names and reject non-numeric outcome columns."""
    if not isinstance(treated_unit, str):
        raise TypeError(
            f"treated_unit must be a string, got {type(treated_unit).__name__}."
        )
    controls = tuple(control_units)
    if not controls:
        raise ValueError("at least one control unit is required.")
    if any(not isinstance(name, str) for name in controls):
        raise TypeError("control_units must be strings.")
    if len(controls) != len(set(controls)):
        raise ValueError("control_units must be unique.")
    if treated_unit in controls:
        raise ValueError("treated_unit cannot also be a control unit.")
    missing = [name for name in (treated_unit, *controls) if name not in frame.columns]
    if missing:
        raise ValueError(f"history is missing column(s) {missing}.")
    for name in (treated_unit, *controls):
        if not pd.api.types.is_numeric_dtype(frame[name]):
            raise TypeError(f"column {name!r} must be numeric.")
    return treated_unit, controls


def _require_known_exclusions(index: pd.Index, exclusions: tuple[Any, ...]) -> None:
    """Raise if an exclusion names a label the history does not have."""
    unknown = [label for label in exclusions if _label_position(index, label) is None]
    if unknown:
        raise ValueError(f"exclusions {unknown!r} are not in the history index.")


def _require_cadence_matches_index(index: pd.Index, cadence: Any) -> None:
    """Reject a cadence that cannot be checked against this index."""
    datetime_index = isinstance(index, pd.DatetimeIndex)
    if isinstance(cadence, int) and datetime_index:
        raise ValueError(
            "integer cadence is not a date frequency. Pass a date-frequency "
            "string, DateOffset, or Timedelta for a datetime index."
        )
    if isinstance(cadence, (str, pd.DateOffset, pd.Timedelta)) and not datetime_index:
        raise ValueError(
            "a date cadence requires a datetime index. Pass an integer cadence "
            "or None for a numeric index."
        )


def _require_audit_after_selection(index: pd.Index, request: ScheduleRequest) -> None:
    """Raise when a present audit origin is not strictly later than selection."""
    selection_positions = [
        position
        for origin in request.selection_origins
        if (position := _label_position(index, origin)) is not None
    ]
    audit_positions = [
        position
        for origin in request.audit_origins
        if (position := _label_position(index, origin)) is not None
    ]
    if (
        selection_positions
        and audit_positions
        and min(audit_positions) <= max(selection_positions)
    ):
        raise ValueError(
            "audit origins must sit strictly after every selection origin "
            "in the history index. An earlier audit is not untouched."
        )


def _plan_window(
    frame: pd.DataFrame,
    treated_unit: str,
    control_units: tuple[str, ...],
    *,
    role: WindowRole,
    origin: Any,
    horizon: int,
    request: ScheduleRequest,
) -> WindowPlan:
    """Plan one window, retaining every failure instead of dropping it."""
    index = frame.index
    position = _label_position(index, origin)
    empty = _empty_index_like(index)
    if position is None:
        return WindowPlan(
            role=role,
            origin=origin,
            horizon=horizon,
            status="missing",
            reasons=(f"origin {origin!r} is absent from the history index.",),
            training_index=empty,
            holdout_index=empty,
            excluded_training=empty,
            excluded_holdout=empty,
            analysis_index=empty,
            forbidden_index=_copy_index(index),
            training_count=0,
            holdout_count=0,
        )
    stored_origin = index[position]
    holdout, gap_reason = _expected_holdout(
        index, position, stored_origin, horizon, request.cadence
    )
    pre = index[:position]
    excluded_training = pre[pre.isin(request.exclusions)]
    training = pre[~pre.isin(request.exclusions)]
    excluded_holdout = holdout[holdout.isin(request.exclusions)]
    analysis_holdout = holdout[~holdout.isin(request.exclusions)]
    analysis = training.append(analysis_holdout)
    forbidden = index[~index.isin(analysis)]
    reasons: list[str] = []
    if gap_reason is not None:
        reasons.append(gap_reason)
    if len(excluded_holdout):
        reasons.append(
            f"exclusion punches a hole in the holdout at {list(excluded_holdout)!r}."
        )
    if len(training) < request.min_training_periods:
        reasons.append(
            f"training has {len(training)} row(s) after exclusions; "
            f"min_training_periods is {request.min_training_periods}."
        )
    bad_labels = _finite_block(frame, analysis, [treated_unit, *control_units])
    if len(bad_labels):
        reasons.append(
            "non-finite outcome on "
            f"{list(bad_labels)!r}; missing data is rejected, not imputed."
        )
    status: WindowStatus = "incomplete" if reasons else "usable"
    return WindowPlan(
        role=role,
        origin=stored_origin,
        horizon=horizon,
        status=status,
        reasons=tuple(reasons),
        training_index=_copy_index(training),
        holdout_index=_copy_index(holdout),
        excluded_training=_copy_index(excluded_training),
        excluded_holdout=_copy_index(excluded_holdout),
        analysis_index=_copy_index(analysis),
        forbidden_index=_copy_index(forbidden),
        training_count=int(len(training)),
        holdout_count=int(len(holdout)),
    )


def _expected_holdout(
    index: pd.Index,
    position: int,
    origin: Any,
    horizon: int,
    cadence: Any,
) -> tuple[pd.Index, str | None]:
    """Return present holdout labels and a gap reason, if any."""
    if cadence is None:
        end = position + horizon
        present = index[position:end]
        if len(present) < horizon:
            missing = horizon - len(present)
            return present, (
                f"horizon {horizon} extends {missing} label(s) past the history."
            )
        return present, None
    expected, error = _cadence_labels(index, origin, horizon, cadence)
    if error is not None:
        return _empty_index_like(index), error
    positions = index.get_indexer(expected)
    missing_labels = [expected[i] for i, pos in enumerate(positions) if int(pos) < 0]
    present_positions = [int(pos) for pos in positions if int(pos) >= 0]
    present = (
        index[present_positions] if present_positions else _empty_index_like(index)
    )
    if missing_labels:
        return present, (f"cadence {cadence!r} is missing label(s) {missing_labels!r}.")
    return present, None


def _cadence_labels(
    index: pd.Index, origin: Any, horizon: int, cadence: Any
) -> tuple[pd.Index, str | None]:
    """Build the labels a declared cadence requires, without imputing gaps."""
    if isinstance(cadence, int):
        try:
            values = [origin + i * cadence for i in range(horizon)]
        except TypeError as exc:
            return _empty_index_like(index), (
                f"integer cadence cannot be applied to origin {origin!r}: {exc}."
            )
        return pd.Index(values), None
    if isinstance(cadence, pd.Timedelta):
        values = [origin + i * cadence for i in range(horizon)]
        return pd.DatetimeIndex(values), None
    try:
        expected = pd.date_range(origin, periods=horizon, freq=cadence)
    except (TypeError, ValueError) as exc:
        return _empty_index_like(index), (
            f"cannot build a holdout from cadence {cadence!r}: {exc}."
        )
    return expected, None


def _pair_positions(
    windows: Sequence[WindowPlan],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Return analysis-overlap and shared-training pairs by window position."""
    overlap: list[tuple[int, int]] = []
    shared: list[tuple[int, int]] = []
    for left_at, left in enumerate(windows):
        for right_at, right in enumerate(windows):
            if right_at <= left_at:
                continue
            if len(left.analysis_index.intersection(right.analysis_index)):
                overlap.append((left_at, right_at))
            if len(left.training_index.intersection(right.training_index)):
                shared.append((left_at, right_at))
    return tuple(overlap), tuple(shared)


def _schedule_counts(windows: Sequence[WindowPlan]) -> ScheduleCounts:
    """Count requested and usable windows without dropping failures."""

    def count(role: str | None, status: str | None) -> int:
        return sum(
            (role is None or window.role == role)
            and (status is None or window.status == status)
            for window in windows
        )

    return ScheduleCounts(
        requested=len(windows),
        usable=count(None, "usable"),
        incomplete=count(None, "incomplete"),
        missing=count(None, "missing"),
        selection_requested=count("selection", None),
        selection_usable=count("selection", "usable"),
        audit_requested=count("audit", None),
        audit_usable=count("audit", "usable"),
    )


@dataclass(frozen=True)
class WindowReplay:
    """One window after replay.

    Audit windows stay ``reserved`` unless fitting was requested. Incomplete
    and missing windows stay ``not-run``. A fit that raises is ``fit-failed``
    and is kept. Scored windows carry both cumulative and average-period
    zero-lift impacts. Bias is the posterior mean minus zero, because no
    effect was injected. It is not a corrected effect.

    Attributes
    ----------
    plan : WindowPlan
        The schedule row this replay executed.
    status : {'scored', 'fit-failed', 'not-run', 'reserved'}
        What replay did with the window.
    reason : str or None
        Failure or reservation text. None when the window was scored.
    cumulative_mean : float or None
        Posterior mean of the cumulative impact. None when not scored.
    cumulative_sd : float or None
        Posterior standard deviation of the cumulative impact.
    cumulative_hdi_lower : float or None
        Lower cumulative HDI bound at :data:`causalpy.constants.HDI_PROB`.
    cumulative_hdi_upper : float or None
        Upper cumulative HDI bound.
    average_mean : float or None
        Posterior mean of the per-period impact. Use this, not the cumulative
        mean, to compare windows of different lengths.
    average_sd : float or None
        Posterior standard deviation of the per-period impact.
    average_hdi_lower : float or None
        Lower per-period HDI bound.
    average_hdi_upper : float or None
        Upper per-period HDI bound.
    bias_cumulative : float or None
        Cumulative posterior mean under a zero injection.
    bias_average : float or None
        Per-period posterior mean under a zero injection.
    tail_probability : float or None
        Share of declared-estimand draws beyond the decision threshold.
        The other estimand is still reported, but its tail is not this field.
    detected : bool or None
        Whether the declared one-sided rule fired. False does not clear the
        bias columns.
    diagnostics_status : str or None
        ``available``, ``partial``, or ``unavailable`` after a fit attempt.
    diagnostics_reason : str or None
        Why diagnostics are not available, when that is the case.
    n_inference_fits : int
        Fit attempts for this window. Reserved and not-run windows are 0.
    counterfactual_mean : numpy.ndarray or None
        Posterior mean of the counterfactual on the holdout, copied so a
        caller cannot mutate the replay.
    """

    plan: WindowPlan
    status: ReplayStatus
    reason: str | None
    cumulative_mean: float | None = None
    cumulative_sd: float | None = None
    cumulative_hdi_lower: float | None = None
    cumulative_hdi_upper: float | None = None
    average_mean: float | None = None
    average_sd: float | None = None
    average_hdi_lower: float | None = None
    average_hdi_upper: float | None = None
    bias_cumulative: float | None = None
    bias_average: float | None = None
    tail_probability: float | None = None
    detected: bool | None = None
    diagnostics_status: str | None = None
    diagnostics_reason: str | None = None
    n_inference_fits: int = 0
    counterfactual_mean: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Own the counterfactual vector."""
        if self.counterfactual_mean is not None:
            object.__setattr__(
                self,
                "counterfactual_mean",
                np.array(self.counterfactual_mean, dtype=float, copy=True),
            )

    @property
    def cumulative_hdi_above_zero(self) -> bool | None:
        """True when the cumulative interval lies entirely above zero."""
        if self.cumulative_hdi_lower is None:
            return None
        return self.cumulative_hdi_lower > 0.0

    @property
    def cumulative_hdi_below_zero(self) -> bool | None:
        """True when the cumulative interval lies entirely below zero."""
        if self.cumulative_hdi_upper is None:
            return None
        return self.cumulative_hdi_upper < 0.0

    @property
    def average_hdi_above_zero(self) -> bool | None:
        """True when the per-period interval lies entirely above zero."""
        if self.average_hdi_lower is None:
            return None
        return self.average_hdi_lower > 0.0

    @property
    def average_hdi_below_zero(self) -> bool | None:
        """True when the per-period interval lies entirely below zero."""
        if self.average_hdi_upper is None:
            return None
        return self.average_hdi_upper < 0.0


@dataclass(frozen=True)
class CalibrationRecord:
    """Typed inputs for the existing operating-characteristics engine.

    ``applicable`` is true only when finite learned-null samples exist and the
    evidence level is ``model-conditional``. ``curve_validated`` is always
    false. Windows and MCMC draws are not independent experiments, so those
    flags cannot be set true and ``independent_experiment_count`` stays unset.

    Direct construction is the downstream contract. :meth:`conditional_curve`
    refuses unless the record is applicable, and it calls
    :func:`causalpy.checks.operating_characteristics.operating_characteristics_from_calibration`
    rather than a second curve.

    Attributes
    ----------
    evidence_level : {'historical-descriptive', 'model-conditional', 'inconclusive', 'insufficient-history', 'fit-failed'}
        Whether the learned-null inputs may be fed to the conditional engine.
        ``model-conditional`` is not a validated launch.
    reasons : tuple of str
        Why the record has this evidence level, including dependence.
    applicable : bool
        True only when ``null_samples`` may be fed to the engine.
    curve_validated : bool
        Always false. A true value is rejected.
    null_samples : numpy.ndarray or None
        Learned-null draws. Present only when ``applicable`` is true.
    fold_sds : numpy.ndarray or None
        Raw posterior standard deviations of the scored selection windows.
        Not floored. The engine rejects non-positive values itself.
    fold_means : numpy.ndarray or None
        Posterior cumulative means of the scored selection windows.
    rope_half_width : float or None
        ROPE half-width stored for the engine. This is not the decision
        rule's effect threshold.
    threshold : float or None
        Decision probability stored for the engine, defaulting to 0.95.
        This is not :attr:`causalpy.design.DecisionRule.threshold`.
    scored_window_count : int
        Selection windows that scored and could enter the null model.
    overlapping_pair_count : int
        Scored selection windows whose analysis indexes intersect.
    shared_training_pair_count : int
        Scored selection windows whose training indexes intersect.
    windows_independent : bool
        Always false.
    mcmc_draws_independent : bool
        Always false.
    independent_experiment_count : int or None
        Always None. Dependence is disclosed, not counted as independence.
    model : str
        ``learned-null-normal`` when samples exist, otherwise
        ``not-applicable``.
    """

    evidence_level: EvidenceLevel
    reasons: tuple[str, ...]
    applicable: bool
    curve_validated: bool = False
    null_samples: np.ndarray | None = None
    fold_sds: np.ndarray | None = None
    fold_means: np.ndarray | None = None
    rope_half_width: float | None = None
    threshold: float | None = None
    scored_window_count: int = 0
    overlapping_pair_count: int = 0
    shared_training_pair_count: int = 0
    windows_independent: bool = False
    mcmc_draws_independent: bool = False
    independent_experiment_count: int | None = None
    model: str = "not-applicable"

    def __post_init__(self) -> None:
        """Reject a validated curve and own any arrays."""
        if self.curve_validated:
            raise ValueError(
                "A calibration record cannot be marked curve-validated. "
                "The conditional model is not a validated false-positive rate."
            )
        if self.windows_independent or self.mcmc_draws_independent:
            raise ValueError("Windows and MCMC draws are not independent experiments.")
        if self.independent_experiment_count is not None:
            raise ValueError(
                "independent_experiment_count stays unset. Dependence is "
                "disclosed, not counted as independent experiments."
            )
        samples = _owned_array(self.null_samples)
        fold_sds = _owned_array(self.fold_sds)
        fold_means = _owned_array(self.fold_means)
        object.__setattr__(self, "null_samples", samples)
        object.__setattr__(self, "fold_sds", fold_sds)
        object.__setattr__(self, "fold_means", fold_means)
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if self.applicable:
            if samples is None or fold_sds is None or fold_means is None:
                raise ValueError(
                    "applicable requires finite null_samples, fold_sds, and fold_means."
                )
            if self.evidence_level != "model-conditional":
                raise ValueError(
                    "applicable requires evidence_level 'model-conditional'."
                )
            if self.model != "learned-null-normal":
                raise ValueError(
                    "an applicable record must name model 'learned-null-normal'."
                )
            if self.rope_half_width is None or self.threshold is None:
                raise ValueError(
                    "applicable requires rope_half_width and threshold. "
                    "The kernel will not invent them."
                )
            if (
                not np.isfinite(samples).all()
                or not np.isfinite(fold_sds).all()
                or not np.isfinite(fold_means).all()
                or np.any(fold_sds <= 0.0)
            ):
                raise ValueError(
                    "applicable requires finite samples and positive fold_sds."
                )
        elif samples is not None:
            raise ValueError(
                "an inapplicable record must not carry null_samples. "
                "Store raw fold means only when the engine will not be fed."
            )

    def conditional_curve(
        self,
        *,
        effect_sizes: list[float] | np.ndarray | None = None,
        rope_half_width: float | None = None,
        threshold: float | None = None,
        n_points: int = 201,
        mde_target: float = 0.80,
    ) -> Any:
        """Feed this record to the existing operating-characteristics engine.

        Parameters
        ----------
        effect_sizes : list of float or numpy.ndarray, optional
            Finite effect grid. The engine's default grid is used when omitted.
        rope_half_width : float, optional
            Overrides the ROPE half-width stored on this record.
        threshold : float, optional
            Overrides the decision probability stored on this record.
        n_points : int, optional
            Grid size when ``effect_sizes`` is omitted.
        mde_target : float, optional
            Detection probability at which the engine reports an MDE.

        Returns
        -------
        causalpy.checks.operating_characteristics.OperatingCharacteristics
            A conditional curve. It is not marked validated.

        Raises
        ------
        ValueError
            If this record is not applicable. The message includes ``reasons``.
        """
        from causalpy.checks.operating_characteristics import (
            operating_characteristics_from_calibration,
        )

        return operating_characteristics_from_calibration(
            self,
            effect_sizes=effect_sizes,
            rope_half_width=rope_half_width,
            threshold=threshold,
            n_points=n_points,
            mde_target=mde_target,
        )


def _owned_array(values: np.ndarray | None) -> np.ndarray | None:
    """Return a float copy, or None."""
    if values is None:
        return None
    return np.array(values, dtype=float, copy=True)


@dataclass(frozen=True)
class HistoricalReplay:
    """Zero-lift historical replay and its calibration record.

    The replay evidence level is never ``model-conditional``. A scored window
    is historical description. The calibration record separately says whether
    the learned-null model was identified. Audit scores, when a caller forces
    them, do not enter the calibration record.

    Attributes
    ----------
    schedule : HistoricalSchedule
        The schedule that was executed.
    decision : DecisionRule
        The one-sided rule used for ``detected``. Both estimands are still
        reported.
    windows : tuple of WindowReplay
        Every schedule window, including reserved, failed, and not-run rows.
    evidence_level : {'historical-descriptive', 'inconclusive', 'insufficient-history', 'fit-failed'}
        What the replay itself supports. Not a validated power claim.
    reasons : tuple of str
        Replay-level reasons. Calibration reasons live on ``calibration``.
    calibration : CalibrationRecord
        The seam downstream operating-characteristics calls consume.
    bias_cumulative_mean : float or None
        Mean of scored selection cumulative biases. None if none scored.
    bias_average_mean : float or None
        Mean of scored selection per-period biases.
    bias_cumulative_sd : float or None
        Sample standard deviation of cumulative bias across scored selection
        windows. This is variation across dependent windows, not a standard
        error of independent experiments. None when fewer than two scored.
    persistent_bias : bool or None
        True only when at least two scored selection windows share a strict
        cumulative-bias sign and each cumulative interval excludes zero.
        False when scored windows do not show that pattern. None when nothing
        scored. True means the bias is visible, not that it was removed.
    n_inference_fits : int
        Window fit attempts plus the learned-null fit when it ran.
    assumptions : tuple of str
        Handoff assumptions. Also :data:`HISTORICAL_REPLAY_ASSUMPTIONS`.
    """

    schedule: HistoricalSchedule
    decision: DecisionRule
    windows: tuple[WindowReplay, ...]
    evidence_level: EvidenceLevel
    reasons: tuple[str, ...]
    calibration: CalibrationRecord
    bias_cumulative_mean: float | None
    bias_average_mean: float | None
    bias_cumulative_sd: float | None
    persistent_bias: bool | None
    n_inference_fits: int
    assumptions: tuple[str, ...] = HISTORICAL_REPLAY_ASSUMPTIONS

    def summary(self) -> pd.DataFrame:
        """Return per-window scores, intervals, and directional flags.

        An interval above zero is favorable for a positive rule and
        unfavorable for a negative rule. The columns report both readings.
        A false detection does not clear the bias columns.

        Returns
        -------
        pandas.DataFrame
            One row per schedule window. Unscored numeric fields are null.
        """
        rows = []
        for window in self.windows:
            row = _window_date_row(window.plan)
            row.update(
                {
                    "status": window.status,
                    "reason": window.reason,
                    "cumulative_mean": window.cumulative_mean,
                    "cumulative_sd": window.cumulative_sd,
                    "cumulative_hdi_lower": window.cumulative_hdi_lower,
                    "cumulative_hdi_upper": window.cumulative_hdi_upper,
                    "average_mean": window.average_mean,
                    "average_sd": window.average_sd,
                    "average_hdi_lower": window.average_hdi_lower,
                    "average_hdi_upper": window.average_hdi_upper,
                    "bias_cumulative": window.bias_cumulative,
                    "bias_average": window.bias_average,
                    "cumulative_hdi_above_zero": window.cumulative_hdi_above_zero,
                    "cumulative_hdi_below_zero": window.cumulative_hdi_below_zero,
                    "average_hdi_above_zero": window.average_hdi_above_zero,
                    "average_hdi_below_zero": window.average_hdi_below_zero,
                    "tail_probability": window.tail_probability,
                    "detected": window.detected,
                    "diagnostics_status": window.diagnostics_status,
                    "diagnostics_reason": window.diagnostics_reason,
                    "n_inference_fits": window.n_inference_fits,
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def recommendation(self) -> str:
        """Return the handoff text. It does not claim that bias was removed.

        Returns
        -------
        str
            Evidence, signed bias, the dependence disclosure, and the
            calibration result. Always states that more windows do not
            eliminate bias.
        """
        scored = [
            window
            for window in self.windows
            if window.status == "scored" and window.plan.role == "selection"
        ]
        parts = [
            (
                f"Replay evidence is {self.evidence_level}. "
                f"{len(scored)} selection window(s) scored, "
                f"{self.n_inference_fits} inference fit(s) ran."
            ),
            " ".join(self.reasons),
            _MORE_WINDOWS,
        ]
        if scored and self.bias_cumulative_mean is not None:
            variation = (
                f" Cross-window cumulative-bias standard deviation is "
                f"{self.bias_cumulative_sd:.4g}; that is variation across "
                "dependent windows, not a standard error."
                if self.bias_cumulative_sd is not None
                else ""
            )
            parts.append(
                f"Cumulative bias mean is {self.bias_cumulative_mean:.4g}. "
                f"Average-period bias mean is {self.bias_average_mean:.4g}."
                f"{variation} Both detection directions are in the summary: "
                "an interval above zero is favorable for a positive rule and "
                "unfavorable for a negative rule."
            )
        if self.persistent_bias is True:
            parts.append(
                "The scored selection windows agree on the sign of cumulative "
                "bias and each cumulative interval excludes zero. That "
                "persistent bias is visible; it has not been removed."
            )
        elif self.persistent_bias is False:
            parts.append(
                "The scored windows do not show a stable same-sign cumulative "
                "bias whose intervals all exclude zero. That is not evidence "
                "that bias was eliminated."
            )
        else:
            parts.append(
                "No selection window was scored, so persistence cannot be assessed."
            )
        calibration = self.calibration
        parts.append(
            "Overlapping scored pairs: "
            f"{calibration.overlapping_pair_count}. Shared-training scored "
            f"pairs: {calibration.shared_training_pair_count}. "
            "Overlapping windows and shared training rows are not independent "
            "experiments. MCMC draws are not independent experiments."
        )
        parts.append(
            f"Calibration evidence is {calibration.evidence_level}. "
            + " ".join(calibration.reasons)
            + " The conditional operating-characteristics model is not a "
            "validated false-positive rate, and curve_validated is false."
        )
        if not self.schedule.audit_reserved:
            parts.append("No untouched later audit was reserved.")
        elif not self.schedule.selection_audit_separated:
            parts.append(
                "The untouched-audit claim cannot be verified because an "
                "origin is missing."
            )
        else:
            parts.append(
                "Audit windows were planned and were not used as calibration inputs."
            )
        return " ".join(part for part in parts if part)


def replay_history(
    history: pd.DataFrame,
    treated_unit: str,
    control_units: Sequence[str],
    schedule: HistoricalSchedule,
    decision: DecisionRule,
    *,
    model: WeightedSumFitter | None = None,
    random_seed: int | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    donor_correlation: float | None = None,
    auto_scale_sigma: bool = True,
    fit_audit: bool = False,
    fit_learned_null: bool = True,
    rope_half_width: float = 0.0,
    threshold: float = 0.95,
    time_column: str | None = None,
) -> HistoricalReplay:
    """Replay a validated schedule on its history and build a calibration record.

    Parameters
    ----------
    history : pandas.DataFrame
        The same history the schedule was built from, or a copy. A frame that
        no longer contains the scheduled labels is rejected. The caller's
        frame is not mutated.
    treated_unit : str
        Must match the schedule.
    control_units : sequence of str
        Must match the schedule.
    schedule : HistoricalSchedule
        Output of :func:`validate_schedule`.
    decision : DecisionRule
        One-sided rule used only to set ``detected``. Bias columns are filled
        either way.
    model : WeightedSumFitter, optional
        Unfitted template. Each window receives a clone. When omitted,
        ``sample_kwargs`` must include ``draws`` so replay does not fall
        back to a 1000-draw, 4-chain default.
    random_seed : int, optional
        Seed copied onto every clone. An explicit ``sample_kwargs`` seed
        overrides it, matching the planner.
    sample_kwargs : dict, optional
        Merged onto each clone. Must include ``draws`` when ``model`` is
        omitted.
    donor_correlation : float, optional
        Passed to synthetic control as ``min_donor_correlation``. The default
        is the estimator default, 0.
    auto_scale_sigma : bool, optional
        Passed through to synthetic control. Default true.
    fit_audit : bool, optional
        Default false. Audit windows stay reserved and out of the null model.
        True fits them and means they are no longer untouched; their scores
        still do not enter calibration.
    fit_learned_null : bool, optional
        Default true. The null model runs only when at least two selection
        windows scored, their diagnostics are available, and their posterior
        standard deviations are positive. Otherwise the record stays
        inapplicable and names the reason.
    rope_half_width : float, optional
        Stored on the calibration record for the operating-characteristics
        engine. Default 0. This is not the decision rule's effect threshold.
    threshold : float, optional
        Decision probability stored for the engine. Default 0.95. This is not
        :attr:`causalpy.design.DecisionRule.threshold`.
    time_column : str, optional
        Time column, when the history is not already indexed by time.

    Returns
    -------
    HistoricalReplay
        Per-window scores, bias, and a calibration record. No
        ``CheckResult`` is constructed.

    Raises
    ------
    TypeError
        If the schedule, decision, model, or seed has the wrong type.
    ValueError
        If the history or units do not match the schedule, or if sampling
        settings would silently use the heavy model default.
    RuntimeError
        If a constructed experiment's training or holdout index does not match
        the window plan. That is a contract failure, not a fit-failed row.
    DataException
        If the history index is missing, duplicated, or not sorted.
    """
    from causalpy.design import DecisionRule as DecisionRuleRuntime

    if not isinstance(schedule, HistoricalSchedule):
        raise TypeError(
            f"schedule must be a HistoricalSchedule, got {type(schedule).__name__}."
        )
    if not isinstance(decision, DecisionRuleRuntime):
        raise TypeError(
            f"decision must be a DecisionRule, got {type(decision).__name__}."
        )
    seed = _require_seed(random_seed)
    if not isinstance(rope_half_width, (int, float)) or isinstance(
        rope_half_width, bool
    ):
        raise TypeError("rope_half_width must be a real number.")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("threshold must be a real number.")
    frame = _history_frame(history, time_column)
    treated, controls = _require_units(frame, treated_unit, control_units)
    if treated != schedule.treated_unit or controls != schedule.control_units:
        raise ValueError("replay units do not match the schedule.")
    _require_scheduled_labels(frame, schedule)
    template = _replay_template(model, sample_kwargs, seed)
    correlation = 0.0 if donor_correlation is None else donor_correlation

    replays: list[WindowReplay] = []
    fit_attempts = 0
    for window in schedule.windows:
        replay, attempts = _replay_window(
            frame,
            treated,
            controls,
            window,
            decision,
            template=template,
            donor_correlation=correlation,
            auto_scale_sigma=auto_scale_sigma,
            fit_audit=fit_audit,
        )
        replays.append(replay)
        fit_attempts += attempts
    windows = tuple(replays)
    calibration, null_fits = _calibration_from_windows(
        windows,
        rope_half_width=float(rope_half_width),
        threshold=float(threshold),
        fit_learned_null=fit_learned_null,
        sample_kwargs=dict(template.sample_kwargs),
        random_seed=seed,
    )
    evidence, reasons, bias_mean, bias_average, bias_sd, persistent = _replay_evidence(
        windows
    )
    return HistoricalReplay(
        schedule=schedule,
        decision=decision,
        windows=windows,
        evidence_level=evidence,
        reasons=reasons,
        calibration=calibration,
        bias_cumulative_mean=bias_mean,
        bias_average_mean=bias_average,
        bias_cumulative_sd=bias_sd,
        persistent_bias=persistent,
        n_inference_fits=fit_attempts + null_fits,
        assumptions=HISTORICAL_REPLAY_ASSUMPTIONS,
    )


def _require_scheduled_labels(
    frame: pd.DataFrame, schedule: HistoricalSchedule
) -> None:
    """Reject a history that no longer contains the planned labels."""
    for window in schedule.windows:
        for label in window.analysis_index:
            if _label_position(frame.index, label) is None:
                raise ValueError(
                    f"replay history is missing scheduled label {label!r}. "
                    "Pass the same history the schedule was built from."
                )


def _replay_template(
    model: WeightedSumFitter | None,
    sample_kwargs: dict[str, Any] | None,
    random_seed: int | None,
) -> WeightedSumFitter:
    """Return an unfitted template whose sampling settings are explicit."""
    if sample_kwargs is not None and not isinstance(sample_kwargs, dict):
        raise TypeError("sample_kwargs must be a dict or None.")
    if model is None:
        if not sample_kwargs or "draws" not in sample_kwargs:
            raise ValueError(
                "Pass a WeightedSumFitter template or sample_kwargs that "
                "include 'draws'. Historical replay does not fall back to a "
                "1000-draw, 4-chain default."
            )
        model = WeightedSumFitter(sample_kwargs=dict(sample_kwargs))
    if not isinstance(model, WeightedSumFitter):
        raise TypeError(
            "Historical replay supports Bayesian simplex synthetic control "
            f"only (WeightedSumFitter), got {type(model).__name__}."
        )
    cloned = model._clone()
    merged = dict(cloned.sample_kwargs)
    if random_seed is not None:
        merged["random_seed"] = random_seed
    if sample_kwargs:
        merged.update(sample_kwargs)
    merged.setdefault("progressbar", False)
    if "draws" not in merged:
        raise ValueError(
            "The model template has no 'draws' in sample_kwargs. Pass "
            "sample_kwargs that include 'draws' so replay does not silently "
            "use a heavy default."
        )
    cloned.sample_kwargs = merged
    return cloned


def _replay_window(
    frame: pd.DataFrame,
    treated_unit: str,
    control_units: tuple[str, ...],
    window: WindowPlan,
    decision: DecisionRule,
    *,
    template: WeightedSumFitter,
    donor_correlation: float,
    auto_scale_sigma: bool,
    fit_audit: bool,
) -> tuple[WindowReplay, int]:
    """Fit one window, or record why it was not fitted."""
    if window.status != "usable":
        return (
            WindowReplay(
                plan=window,
                status="not-run",
                reason=window.reason,
            ),
            0,
        )
    if window.role == "audit" and not fit_audit:
        return (
            WindowReplay(
                plan=window,
                status="reserved",
                reason="audit window reserved and not fitted.",
            ),
            0,
        )
    analysis = frame.loc[list(window.analysis_index), [treated_unit, *control_units]]
    cloned = template._clone()
    cloned.sample_kwargs = dict(template.sample_kwargs)
    experiment = SyntheticControl(
        analysis.copy(),
        window.origin,
        control_units=list(control_units),
        treated_units=[treated_unit],
        model=cloned,
        min_donor_correlation=donor_correlation,
        auto_scale_sigma=auto_scale_sigma,
    )
    _require_window_boundaries(experiment, window)
    try:
        experiment.fit()
    except Exception as exc:
        diagnostics = _read_diagnostics(experiment)
        return (
            WindowReplay(
                plan=window,
                status="fit-failed",
                reason=f"{type(exc).__name__}: {exc}",
                diagnostics_status=diagnostics.get("diagnostics_status"),
                diagnostics_reason=diagnostics.get("diagnostics_reason"),
                n_inference_fits=1,
            ),
            1,
        )
    return _score_window(experiment, window, decision), 1


def _require_window_boundaries(
    experiment: SyntheticControl, window: WindowPlan
) -> None:
    """Raise if the estimator can see a label outside the analysis window."""
    permitted_holdout = window.holdout_index[
        ~window.holdout_index.isin(window.excluded_holdout)
    ]
    checks = (
        ("training index", experiment.datapre.index, window.training_index),
        ("holdout index", experiment.datapost.index, permitted_holdout),
        ("analysis index", experiment.data.index, window.analysis_index),
    )
    for name, actual, expected in checks:
        if not _same_labels(pd.Index(actual), pd.Index(expected)):
            raise RuntimeError(
                f"Synthetic control {name} {list(actual)!r} does not match "
                f"the planned window {list(expected)!r}. Refusing to fit a "
                "window that would see data outside its contract."
            )
    leaked = [
        label
        for label in window.forbidden_index
        if _label_position(pd.Index(experiment.data.index), label) is not None
    ]
    if leaked:
        raise RuntimeError(
            f"Forbidden label(s) {leaked!r} are inside the estimator frame."
        )
    pre_obs = pd.Index(experiment.pre_design["treated"].coords["obs_ind"].values)
    post_obs = pd.Index(experiment.post_design["control"].coords["obs_ind"].values)
    if not _same_labels(pre_obs, window.training_index):
        raise RuntimeError(
            "pre-period design rows do not match the training index. Priors "
            "and screening would see data outside the window."
        )
    if not _same_labels(post_obs, permitted_holdout):
        raise RuntimeError("post-period donor rows do not match the holdout index.")


def _read_diagnostics(experiment: SyntheticControl) -> dict[str, Any]:
    """Read the fit's sampling diagnostics without a second diagnostic engine."""
    try:
        return dict(experiment._model_backend.sampling_diagnostics())
    except Exception as exc:
        return {
            "diagnostics_status": "unavailable",
            "diagnostics_reason": f"{type(exc).__name__}: {exc}",
        }


def _score_window(
    experiment: SyntheticControl,
    window: WindowPlan,
    decision: DecisionRule,
) -> WindowReplay:
    """Summarize one successful zero-lift fit. Both estimands are kept."""
    impact = experiment.result.impact_post.sel(
        treated_units=experiment.treated_units[0]
    )
    cumulative = impact.sum("obs_ind")
    average = impact.mean("obs_ind")
    cumulative_mean = float(cumulative.mean().item())
    average_mean = float(average.mean().item())
    cumulative_sd = float(cumulative.std().item())
    average_sd = float(average.std().item())
    cumulative_lower, cumulative_upper = hdi_bounds(
        cumulative, prob=HDI_PROB, dim=("chain", "draw")
    )
    average_lower, average_upper = hdi_bounds(
        average, prob=HDI_PROB, dim=("chain", "draw")
    )
    declared = cumulative if decision.estimand == "cumulative" else average
    flat = np.asarray(declared.values, dtype=float).reshape(-1)
    if decision.direction == "positive":
        tail = float(np.mean(flat > decision.threshold))
    else:
        tail = float(np.mean(flat < decision.threshold))
    counterfactual = experiment.result.predictions_post.sel(
        treated_units=experiment.treated_units[0]
    )
    diagnostics = _read_diagnostics(experiment)
    return WindowReplay(
        plan=window,
        status="scored",
        reason=None,
        cumulative_mean=cumulative_mean,
        cumulative_sd=cumulative_sd,
        cumulative_hdi_lower=cumulative_lower,
        cumulative_hdi_upper=cumulative_upper,
        average_mean=average_mean,
        average_sd=average_sd,
        average_hdi_lower=average_lower,
        average_hdi_upper=average_upper,
        bias_cumulative=cumulative_mean,
        bias_average=average_mean,
        tail_probability=tail,
        detected=bool(tail >= decision.probability),
        diagnostics_status=diagnostics.get("diagnostics_status"),
        diagnostics_reason=diagnostics.get("diagnostics_reason"),
        n_inference_fits=1,
        counterfactual_mean=np.asarray(
            counterfactual.mean(("chain", "draw")).values, dtype=float
        ),
    )


def _replay_evidence(
    windows: Sequence[WindowReplay],
) -> tuple[
    EvidenceLevel,
    tuple[str, ...],
    float | None,
    float | None,
    float | None,
    bool | None,
]:
    """Classify the replay. It is never model-conditional."""
    selection = [window for window in windows if window.plan.role == "selection"]
    attempted = [window for window in selection if window.n_inference_fits]
    scored = [window for window in selection if window.status == "scored"]
    failed = [window for window in selection if window.status == "fit-failed"]
    reasons: list[str] = []
    if not any(window.plan.status == "usable" for window in selection):
        reasons.append("No usable selection window was available to score.")
        return "insufficient-history", tuple(reasons), None, None, None, None
    if attempted and not scored:
        reasons.append(
            "Every attempted selection fit failed: "
            + "; ".join(window.reason or "fit failed" for window in failed)
        )
        return "fit-failed", tuple(reasons), None, None, None, None
    if failed:
        reasons.append(
            "Some selection windows failed and were retained: "
            + "; ".join(window.reason or "fit failed" for window in failed)
        )
    cumulative = np.array([window.bias_cumulative for window in scored], dtype=float)
    average = np.array([window.bias_average for window in scored], dtype=float)
    bias_sd = float(np.std(cumulative, ddof=1)) if len(scored) >= 2 else None
    excludes = [
        bool(window.cumulative_hdi_above_zero or window.cumulative_hdi_below_zero)
        for window in scored
    ]
    signs = {int(np.sign(value)) for value in cumulative if value != 0.0}
    if not scored:
        persistent = None
    elif (
        len(scored) >= 2 and all(excludes) and len(signs) == 1 and 0.0 not in cumulative
    ):
        persistent = True
    else:
        persistent = False
    reasons.append(
        "Scored windows are historical description of the zero-lift impact, "
        "not a conditional power curve."
    )
    return (
        "historical-descriptive",
        tuple(reasons),
        float(np.mean(cumulative)),
        float(np.mean(average)),
        bias_sd,
        persistent,
    )


class _CalibrationShared(TypedDict):
    """Fields shared by applicable and inapplicable calibration records."""

    rope_half_width: float
    threshold: float
    scored_window_count: int
    overlapping_pair_count: int
    shared_training_pair_count: int


def _calibration_from_windows(
    windows: Sequence[WindowReplay],
    *,
    rope_half_width: float,
    threshold: float,
    fit_learned_null: bool,
    sample_kwargs: dict[str, Any],
    random_seed: int | None,
) -> tuple[CalibrationRecord, int]:
    """Build the calibration seam from scored selection windows only."""
    scored = [
        window
        for window in windows
        if window.status == "scored" and window.plan.role == "selection"
    ]
    overlap, shared = _pair_positions([window.plan for window in scored])
    base: _CalibrationShared = {
        "rope_half_width": rope_half_width,
        "threshold": threshold,
        "scored_window_count": len(scored),
        "overlapping_pair_count": len(overlap),
        "shared_training_pair_count": len(shared),
    }
    dependence = _dependence_reason(len(overlap), len(shared))
    if len(scored) < 2:
        failed = [
            window
            for window in windows
            if window.status == "fit-failed" and window.plan.role == "selection"
        ]
        level: EvidenceLevel = (
            "fit-failed" if failed and not scored else "insufficient-history"
        )
        failed_text = (
            " Failed selection windows were retained: "
            + "; ".join(window.reason or "fit failed" for window in failed)
            if failed
            else ""
        )
        reason = (
            "Fewer than two scored selection windows. The learned-null model "
            "was not identified, and no conditional curve was produced." + failed_text
        )
        return _inapplicable(level, (reason, dependence), scored, base), 0
    diagnostics_bad = [
        window for window in scored if window.diagnostics_status != "available"
    ]
    if diagnostics_bad:
        named = ", ".join(repr(window.plan.origin) for window in diagnostics_bad)
        return _inapplicable(
            "inconclusive",
            (
                "Sampling diagnostics are not available for every scored "
                f"selection window ({named}). The learned-null model was not "
                "fed. Partial or unavailable diagnostics are inconclusive, "
                "not a conditional curve.",
                dependence,
            ),
            scored,
            base,
        ), 0
    fold_sds = np.array([window.cumulative_sd for window in scored], dtype=float)
    if not np.isfinite(fold_sds).all() or np.any(fold_sds <= 0.0):
        return _inapplicable(
            "inconclusive",
            (
                "A scored window has a non-positive or non-finite posterior "
                "standard deviation. Those raw values are not floored, and "
                "the conditional engine was not fed.",
                dependence,
            ),
            scored,
            base,
        ), 0
    if not fit_learned_null:
        return _inapplicable(
            "historical-descriptive",
            (
                "Learned-null fit was not requested. Replay evidence stays "
                "descriptive, and the conditional model was not fed.",
                dependence,
            ),
            scored,
            base,
        ), 0
    means = np.array([window.bias_cumulative for window in scored], dtype=float)
    try:
        _, theta = _fit_learned_null(
            means,
            fold_sds,
            prior_scale=_LEARNED_NULL_PRIOR_SCALE,
            sample_kwargs=dict(sample_kwargs),
            random_seed=random_seed,
        )
    except _NullModelUnidentifiedError as exc:
        return _inapplicable(
            "inconclusive",
            (str(exc), dependence),
            scored,
            base,
        ), 1
    except Exception as exc:
        return _inapplicable(
            "fit-failed",
            (
                f"Learned-null fit failed: {type(exc).__name__}: {exc}",
                dependence,
            ),
            scored,
            base,
        ), 1
    reasons = (
        "The learned-null Normal model was identified from the scored "
        "selection windows. The curve is conditional on those inputs, not "
        "an empirical false-positive rate.",
        dependence,
        "Learned-null prior_scale is the PlaceboInTime default of 1. "
        "It is not a noise menu.",
    )
    return CalibrationRecord(
        evidence_level="model-conditional",
        reasons=reasons,
        applicable=True,
        curve_validated=False,
        null_samples=np.asarray(theta, dtype=float),
        fold_sds=fold_sds,
        fold_means=means,
        model="learned-null-normal",
        **base,
    ), 1


def _dependence_reason(overlap: int, shared: int) -> str:
    """Disclose dependence without treating it as a validated-curve barrier."""
    return (
        f"{overlap} scored selection pair(s) overlap and {shared} share "
        "training rows. Those windows are not independent experiments, and "
        "MCMC draws are not independent experiments."
    )


def _inapplicable(
    level: EvidenceLevel,
    reasons: tuple[str, ...],
    scored: Sequence[WindowReplay],
    base: _CalibrationShared,
) -> CalibrationRecord:
    """Return a record that stores raw fold summaries and no null samples."""
    fold_means = (
        np.array([window.bias_cumulative for window in scored], dtype=float)
        if scored
        else None
    )
    fold_sds = (
        np.array([window.cumulative_sd for window in scored], dtype=float)
        if scored
        else None
    )
    return CalibrationRecord(
        evidence_level=level,
        reasons=reasons,
        applicable=False,
        curve_validated=False,
        null_samples=None,
        fold_sds=fold_sds,
        fold_means=fold_means,
        model="not-applicable",
        **base,
    )
