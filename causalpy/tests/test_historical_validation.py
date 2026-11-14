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
"""Historical schedule, zero-lift replay, and the calibration seam.

Schedule tests do not sample. Boundary tests spy on the estimator frame and
also compare same-seed fits. The persistent-bias panel puts the treated unit
outside the donor convex hull, so a weighted sum cannot absorb the level.
More windows make that bias visible. They do not remove it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pymc_extras.prior import Prior

from causalpy.checks.base import CheckResult
from causalpy.checks.operating_characteristics import (
    operating_characteristics,
    operating_characteristics_from_calibration,
)
from causalpy.custom_exceptions import DataException
from causalpy.design import (
    DecisionRule,
    replay_history,
    validate_schedule,
)
from causalpy.experiments.model_adapter import ModelAdapter
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.historical_validation import (
    CalibrationRecord,
    ScheduleRequest,
    WindowReplay,
    _score_window,
)
from causalpy.pymc_models import WeightedSumFitter

LIGHT = {"tune": 15, "draws": 15, "chains": 1, "cores": 1, "progressbar": False}
DONORS = ("a", "b", "c", "d")
RULE = DecisionRule("positive", "cumulative", threshold=0.0, probability=0.95)


def _request(*origins: object, **kwargs: object) -> ScheduleRequest:
    payload = {"selection_origins": origins, "horizons": (2,), "cadence": 1}
    payload.update(kwargs)
    return ScheduleRequest(**payload)  # type: ignore[arg-type]


def _integer_frame(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {name: rng.normal(size=n) for name in ("y", *DONORS)},
        index=pd.RangeIndex(n, name="time"),
    )
    frame["extra"] = 1_000.0
    return frame


def _bias_frame() -> pd.DataFrame:
    """Treated level sits outside the donor hull on every row."""
    t = np.arange(12)
    frame = pd.DataFrame(
        {name: np.sin(t / 2.0 + i) for i, name in enumerate(DONORS)},
        index=pd.RangeIndex(12, name="time"),
    )
    frame["y"] = 4.0
    return frame


def _exact_frame() -> pd.DataFrame:
    """Treated path is weights (1/2, 1/4, 1/4) of three donors."""
    c0 = np.array([0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0])
    c1 = np.array([0.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0])
    c2 = np.array([0.0, 0.0, 0.0, 8.0, 0.0, 8.0, 8.0, 8.0])
    treated = 0.5 * c0 + 0.25 * c1 + 0.25 * c2
    return pd.DataFrame(
        {"y": treated, "c0": c0, "c1": c1, "c2": c2},
        index=pd.RangeIndex(8, name="time"),
    )


def _wide_template() -> WeightedSumFitter:
    """Wide noise prior so a misspecified level still samples in a light test.

    The prior does not add an intercept. Simplex weights still cannot match a
    treated level outside the donor hull.
    """
    return WeightedSumFitter(
        priors={
            "y_hat": Prior(
                "Normal",
                sigma=5.0,
                dims=["obs_ind", "treated_units"],
            )
        },
        sample_kwargs=dict(LIGHT),
    )


def _replay(frame, request, **kwargs):
    schedule = validate_schedule(frame, "y", list(DONORS), request)
    return replay_history(
        frame,
        "y",
        DONORS,
        schedule,
        RULE,
        model=_wide_template(),
        random_seed=7,
        **kwargs,
    )


def test_schedule_geometry_keeps_failures_and_forbids_outside_rows():
    frame = _integer_frame()
    original = frame.copy()
    schedule = validate_schedule(
        frame,
        "y",
        DONORS,
        _request(
            4, 8, horizons=(2, 3), cadence=2, exclusions=(1, 6), audit_origins=(11,)
        ),
    )
    pd.testing.assert_frame_equal(frame, original)
    by_key = {(window.origin, window.horizon): window for window in schedule.windows}
    first = by_key[(4, 2)]
    assert list(first.training_index) == [0, 2, 3]
    assert list(first.holdout_index) == [4, 6]
    assert 1 in first.excluded_training
    assert 6 in first.excluded_holdout
    assert first.status == "incomplete"
    assert "6" in (first.reason or "")
    assert 6 not in first.analysis_index
    assert 5 in first.forbidden_index
    assert 7 in first.forbidden_index
    assert 9 in first.forbidden_index
    second = by_key[(8, 3)]
    assert second.horizon == 3
    assert list(second.holdout_index) != list(first.holdout_index)
    assert schedule.counts.requested == 6
    assert schedule.counts.missing == 0
    assert schedule.overlapping_pairs
    assert schedule.shared_training_pairs
    assert schedule.audit_reserved is True
    assert schedule.selection_audit_separated is True
    dates = schedule.exact_dates()
    assert len(dates) == 6
    assert set(dates["role"]) == {"selection", "audit"}


def test_absent_origin_is_retained_and_contract_errors_raise():
    frame = _integer_frame()
    schedule = validate_schedule(frame, "y", DONORS, _request(4, 99))
    missing = [window for window in schedule.windows if window.origin == 99]
    assert len(missing) == 1
    assert missing[0].status == "missing"
    assert schedule.counts.missing == 1
    assert schedule.selection_audit_separated is False
    with pytest.raises(ValueError, match="not in the history index"):
        validate_schedule(frame, "y", DONORS, _request(4, exclusions=(50,)))
    with pytest.raises(ValueError, match="strictly after"):
        validate_schedule(frame, "y", DONORS, _request(6, audit_origins=(4,)))
    with pytest.raises(ValueError, match="unique"):
        ScheduleRequest(selection_origins=(4, 4), horizons=(2,))
    with pytest.raises(TypeError, match="boolean"):
        ScheduleRequest(selection_origins=(4,), horizons=(2,), cadence=True)
    with pytest.raises(TypeError, match="Python int"):
        ScheduleRequest(selection_origins=(4,), horizons=(np.int64(2),))
    with pytest.raises(ValueError, match="datetime index"):
        validate_schedule(
            frame,
            "y",
            DONORS,
            ScheduleRequest(selection_origins=(4,), horizons=(2,), cadence="W"),
        )


def test_datetime_cadence_records_gaps_instead_of_imputing():
    index = pd.date_range("2020-01-05", periods=8, freq="W")
    frame = pd.DataFrame(
        {name: np.arange(8, dtype=float) for name in ("y", *DONORS)},
        index=index,
    )
    origin = index[3]
    schedule = validate_schedule(
        frame,
        "y",
        DONORS,
        ScheduleRequest(selection_origins=(origin,), horizons=(2,), cadence="W"),
    )
    window = schedule.windows[0]
    assert window.status == "usable"
    assert list(window.holdout_index) == list(index[3:5])
    gapped = frame.drop(index[4])
    gapped_schedule = validate_schedule(
        gapped,
        "y",
        DONORS,
        ScheduleRequest(selection_origins=(origin,), horizons=(2,), cadence="W"),
    )
    gapped_window = gapped_schedule.windows[0]
    assert gapped_window.status == "incomplete"
    assert "missing label" in (gapped_window.reason or "")
    assert index[4] not in gapped_window.analysis_index


def test_anchored_cadence_that_misses_the_origin_is_incomplete():
    index = pd.date_range("2020-01-01", periods=40, freq="D")
    frame = pd.DataFrame(
        {name: np.arange(40, dtype=float) for name in ("y", *DONORS)},
        index=index,
    )
    origin = pd.Timestamp("2020-01-08")
    snapped = validate_schedule(
        frame,
        "y",
        DONORS,
        ScheduleRequest(selection_origins=(origin,), horizons=(4,), cadence="W"),
    )
    window = snapped.windows[0]
    assert window.status == "incomplete"
    assert "does not start at the origin" in (window.reason or "")
    assert origin not in window.holdout_index
    assert pd.Timestamp("2020-01-12") not in window.holdout_index
    sunday = pd.Timestamp("2020-01-05")
    aligned = validate_schedule(
        frame,
        "y",
        DONORS,
        ScheduleRequest(selection_origins=(sunday,), horizons=(2,), cadence="W"),
    )
    assert aligned.windows[0].status == "usable"
    assert aligned.windows[0].holdout_index[0] == sunday
    wednesday = validate_schedule(
        frame,
        "y",
        DONORS,
        ScheduleRequest(selection_origins=(origin,), horizons=(2,), cadence="W-WED"),
    )
    assert wednesday.windows[0].status == "usable"
    assert wednesday.windows[0].holdout_index[0] == origin
    month_end = validate_schedule(
        frame,
        "y",
        DONORS,
        ScheduleRequest(selection_origins=(origin,), horizons=(2,), cadence="ME"),
    )
    assert month_end.windows[0].status == "incomplete"
    assert "does not start at the origin" in (month_end.windows[0].reason or "")


def test_timezone_aware_schedule_reaches_the_fit(monkeypatch):
    index = pd.date_range("2020-01-05", periods=8, freq="W", tz="UTC")
    frame = pd.DataFrame(
        {name: np.arange(8, dtype=float) for name in ("y", *DONORS)},
        index=index,
    )
    schedule = validate_schedule(
        frame,
        "y",
        DONORS,
        ScheduleRequest(selection_origins=(index[3],), horizons=(2,), cadence="W"),
    )
    assert schedule.windows[0].status == "usable"
    fitted: list[object] = []

    def fit(self):
        fitted.append(self.treatment_time)
        raise RuntimeError("sentinel fit")

    monkeypatch.setattr(SyntheticControl, "fit", fit)
    replay = replay_history(
        frame,
        "y",
        DONORS,
        schedule,
        RULE,
        model=_wide_template(),
        random_seed=1,
    )
    assert fitted == [index[3]]
    assert replay.windows[0].status == "fit-failed"
    assert "outside the window" not in (replay.windows[0].reason or "")


def test_partial_nonfinite_impact_draws_are_not_scored():
    import xarray as xr

    frame = _integer_frame()
    window = validate_schedule(frame, "y", DONORS, _request(4)).windows[0]
    impact = xr.DataArray(
        np.array([[[[1.0, 1.0], [2.0, np.nan], [3.0, 3.0]]]]),
        dims=("treated_units", "chain", "draw", "obs_ind"),
        coords={"treated_units": ["y"]},
    )

    class _Result:
        impact_post = impact

    class _Experiment:
        treated_units = ["y"]
        result = _Result()

    scored = _score_window(_Experiment(), window, RULE)  # type: ignore[arg-type]
    assert scored.status == "fit-failed"
    assert scored.bias_cumulative is None
    assert "non-finite impact" in (scored.reason or "")
    assert scored.n_inference_fits == 1


def _scored_replay_window(*args, **kwargs):
    window = args[3]
    return (
        WindowReplay(
            plan=window,
            status="scored",
            reason=None,
            cumulative_sd=0.2,
            bias_cumulative=float(window.origin),
            bias_average=0.05,
            diagnostics_status="available",
            n_inference_fits=1,
        ),
        1,
    )


def test_nonfinite_learned_null_samples_retain_scored_windows(monkeypatch):
    frame = _integer_frame()
    schedule = validate_schedule(frame, "y", DONORS, _request(4, 6))
    monkeypatch.setattr(
        "causalpy.historical_validation._replay_window", _scored_replay_window
    )
    monkeypatch.setattr(
        "causalpy.historical_validation._fit_learned_null",
        lambda *args, **kwargs: (None, np.array([np.nan, 1.0])),
    )
    replay = replay_history(
        frame,
        "y",
        DONORS,
        schedule,
        RULE,
        model=_wide_template(),
        random_seed=1,
    )
    assert [window.status for window in replay.windows] == ["scored", "scored"]
    assert replay.calibration.applicable is False
    assert replay.calibration.null_samples is None
    assert replay.calibration.evidence_level == "inconclusive"
    assert "finite one-dimensional" in " ".join(replay.calibration.reasons)


def test_zero_dimensional_learned_null_sample_feeds_the_existing_kernel(monkeypatch):
    frame = _integer_frame()
    schedule = validate_schedule(frame, "y", DONORS, _request(4, 6))
    monkeypatch.setattr(
        "causalpy.historical_validation._replay_window", _scored_replay_window
    )
    monkeypatch.setattr(
        "causalpy.historical_validation._fit_learned_null",
        lambda *args, **kwargs: (None, np.array(0.25)),
    )
    replay = replay_history(
        frame,
        "y",
        DONORS,
        schedule,
        RULE,
        model=_wide_template(),
        random_seed=1,
    )
    samples = replay.calibration.null_samples
    assert replay.calibration.applicable is True
    assert samples is not None
    assert samples.shape == (1,)
    assert np.isfinite(samples).all()
    curve = operating_characteristics_from_calibration(
        replay.calibration, effect_sizes=[0.0, 0.5]
    )
    assert np.isfinite(curve.p_detect).all()


def test_caller_label_aliases_resolve_to_stored_index_labels():
    index = pd.date_range("2020-01-05", periods=8, freq="W")
    frame = pd.DataFrame(
        {name: np.arange(8, dtype=float) for name in ("y", *DONORS)},
        index=index,
    )
    excluded = index[4].strftime("%Y-%m-%d")
    schedule = validate_schedule(
        frame,
        "y",
        DONORS,
        ScheduleRequest(
            selection_origins=(index[3].strftime("%Y-%m-%d"),),
            horizons=(2,),
            cadence="W",
            exclusions=(excluded,),
        ),
    )
    window = schedule.windows[0]
    assert window.origin == index[3]
    assert window.status == "incomplete"
    assert list(window.excluded_holdout) == [index[4]]
    assert index[4] not in window.analysis_index
    with pytest.raises(ValueError, match="same history label"):
        validate_schedule(
            frame,
            "y",
            DONORS,
            ScheduleRequest(
                selection_origins=(index[3], index[3].strftime("%Y-%m-%d")),
                horizons=(2,),
                cadence="W",
            ),
        )
    with pytest.raises(ValueError, match="strictly after"):
        validate_schedule(
            frame,
            "y",
            DONORS,
            ScheduleRequest(
                selection_origins=(index[1],),
                audit_origins=(index[1].strftime("%Y-%m-%d"),),
                horizons=(2,),
                cadence="W",
            ),
        )
    integer = _integer_frame()
    with pytest.raises(ValueError, match="same history label"):
        validate_schedule(
            integer,
            "y",
            DONORS,
            ScheduleRequest(selection_origins=(4, np.int64(4)), horizons=(2,)),
        )


def test_unsorted_or_duplicate_index_is_rejected():
    frame = _integer_frame()
    duplicated = frame.copy()
    duplicated.index = pd.Index([0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10], name="time")
    with pytest.raises(DataException, match="duplicate"):
        validate_schedule(duplicated, "y", DONORS, _request(4))
    unsorted = frame.copy()
    unsorted.index = pd.Index([0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11], name="time")
    with pytest.raises(DataException, match="not sorted"):
        validate_schedule(unsorted, "y", DONORS, _request(4, 6))
    schedule = validate_schedule(frame, "y", DONORS, _request(4))
    with pytest.raises(DataException, match="not sorted"):
        replay_history(
            unsorted,
            "y",
            DONORS,
            schedule,
            RULE,
            sample_kwargs=LIGHT,
        )


def test_selection_horizon_that_reaches_an_audit_holdout_is_rejected():
    frame = _integer_frame(16)
    with pytest.raises(ValueError, match="audit holdout"):
        validate_schedule(
            frame,
            "y",
            DONORS,
            _request(4, horizons=(8,), cadence=1, audit_origins=(8,)),
        )
    separated = validate_schedule(
        frame,
        "y",
        DONORS,
        _request(4, horizons=(2,), cadence=1, audit_origins=(8,)),
    )
    assert separated.selection_audit_separated is True
    assert separated.windows[1].role == "audit"
    assert set(separated.windows[0].analysis_index).isdisjoint(
        separated.windows[1].holdout_index
    )


def test_nonfinite_replay_holdout_is_not_scored():
    frame = _integer_frame(8)
    schedule = validate_schedule(frame, "y", DONORS, _request(4))
    assert schedule.windows[0].status == "usable"
    replay_frame = frame.copy()
    replay_frame.loc[5, "y"] = np.nan
    replay = replay_history(
        replay_frame,
        "y",
        DONORS,
        schedule,
        RULE,
        sample_kwargs=LIGHT,
    )
    window = replay.windows[0]
    assert window.status == "not-run"
    assert window.n_inference_fits == 0
    assert "non-finite" in (window.reason or "")
    assert window.bias_cumulative is None
    assert "not imputed" in replay.recommendation()


def test_boundary_mismatch_does_not_drop_later_windows(monkeypatch):
    frame = _integer_frame()
    schedule = validate_schedule(frame, "y", DONORS, _request(4, 6))
    fitted: list[object] = []

    def check(experiment, window):
        if window.origin == 4:
            raise RuntimeError("training index does not match the planned window")

    def fit(self):
        fitted.append(self.treatment_time)
        raise RuntimeError("sentinel fit")

    monkeypatch.setattr(
        "causalpy.historical_validation._require_window_boundaries",
        check,
    )
    monkeypatch.setattr(SyntheticControl, "fit", fit)
    replay = replay_history(
        frame,
        "y",
        DONORS,
        schedule,
        RULE,
        model=_wide_template(),
        random_seed=1,
    )
    assert replay.windows[0].status == "not-run"
    assert "planned window" in (replay.windows[0].reason or "")
    assert replay.windows[0].n_inference_fits == 0
    assert replay.windows[1].status == "fit-failed"
    assert fitted == [6]


def test_nonfinite_training_is_rejected_without_imputation():
    frame = _integer_frame()
    frame.loc[2, "y"] = np.nan
    schedule = validate_schedule(frame, "y", DONORS, _request(4))
    window = schedule.windows[0]
    assert window.status == "incomplete"
    assert "non-finite" in (window.reason or "")
    assert 2 in window.training_index
    assert not frame["y"].isna().all()


def test_replay_refuses_a_window_that_would_see_forbidden_rows(monkeypatch):
    frame = _integer_frame()
    seen: list[pd.DataFrame] = []

    def record(self, **kwargs):
        seen.append(self.data.copy())
        raise RuntimeError("stop after recording the estimator frame")

    monkeypatch.setattr(SyntheticControl, "fit", record)
    request = _request(4, cadence=2)
    replay = replay_history(
        frame,
        "y",
        DONORS,
        validate_schedule(frame, "y", DONORS, request),
        RULE,
        model=_wide_template(),
        random_seed=1,
    )
    assert len(seen) == 1
    window = replay.schedule.windows[0]
    assert list(seen[0].index) == list(window.analysis_index)
    assert "extra" not in seen[0].columns
    assert set(seen[0].columns) == {"y", *DONORS}
    for label in window.forbidden_index:
        assert label not in seen[0].index
    assert replay.windows[0].status == "fit-failed"
    assert "stop after recording" in (replay.windows[0].reason or "")


def test_replay_does_not_construct_a_check_result(monkeypatch):
    constructed: list[str] = []
    original = CheckResult.__init__

    def spy(self, *args, **kwargs):
        constructed.append("built")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CheckResult, "__init__", spy)
    frame = _integer_frame()
    monkeypatch.setattr(
        SyntheticControl,
        "fit",
        lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("no result")),
    )
    replay_history(
        frame,
        "y",
        DONORS,
        validate_schedule(frame, "y", DONORS, _request(4)),
        RULE,
        model=_wide_template(),
        random_seed=1,
    )
    assert constructed == []


def test_forbidden_data_does_not_change_the_fit_and_holdout_data_does():
    frame = _exact_frame()
    controls = ["c0", "c1", "c2"]
    request = ScheduleRequest(selection_origins=(4,), horizons=(2,), cadence=1)
    schedule = validate_schedule(frame, "y", controls, request)
    template = WeightedSumFitter(sample_kwargs=dict(LIGHT))

    def run(panel: pd.DataFrame):
        return replay_history(
            panel,
            "y",
            controls,
            schedule,
            RULE,
            model=template,
            random_seed=11,
        )

    baseline = run(frame)
    forbidden = frame.copy()
    forbidden.loc[7, "c0"] = 1_000.0
    forbidden.loc[7, "y"] = -1_000.0
    unchanged = run(forbidden)
    treated = frame.copy()
    treated.loc[[4, 5], "y"] = treated.loc[[4, 5], "y"] + 5.0
    shifted = run(treated)
    donors = frame.copy()
    donors.loc[[4, 5], "c0"] = donors.loc[[4, 5], "c0"] + 40.0
    moved = run(donors)

    base_window = baseline.windows[0]
    same_window = unchanged.windows[0]
    assert base_window.status == "scored"
    np.testing.assert_allclose(
        base_window.counterfactual_mean,
        same_window.counterfactual_mean,
        atol=1e-4,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        base_window.cumulative_mean,
        same_window.cumulative_mean,
        atol=1e-4,
        rtol=1e-4,
    )
    assert shifted.windows[0].cumulative_mean > base_window.cumulative_mean + 5.0
    assert not np.allclose(
        moved.windows[0].counterfactual_mean,
        base_window.counterfactual_mean,
        atol=1.0,
    )


def test_persistent_bias_stays_visible_and_is_not_called_eliminated():
    frame = _bias_frame()
    replay = _replay(frame, _request(4, 8))
    scored = [window for window in replay.windows if window.status == "scored"]
    assert len(scored) == 2
    assert all(
        window.bias_cumulative is not None and window.bias_cumulative > 2.0
        for window in scored
    )
    assert all(np.sign(window.bias_cumulative) == 1 for window in scored)
    assert replay.persistent_bias is True
    text = replay.recommendation()
    assert "More windows do not eliminate bias." in text
    assert "has not been removed" in text
    assert "eliminated" not in text.replace("do not eliminate", "")
    assert "not independent experiments" in text
    assert replay.calibration.applicable is False
    assert replay.calibration.null_samples is None
    assert replay.calibration.evidence_level == "inconclusive"
    assert replay.evidence_level == "historical-descriptive"


def test_incomplete_fit_is_retained_and_audit_windows_stay_reserved(monkeypatch):
    frame = _integer_frame()
    calls: list[int] = []

    def boom(self, **kwargs):
        calls.append(1)
        raise RuntimeError("sampler exploded")

    monkeypatch.setattr(SyntheticControl, "fit", boom)
    schedule = validate_schedule(
        frame,
        "y",
        DONORS,
        _request(4, 8, audit_origins=(10,)),
    )
    replay = replay_history(
        frame,
        "y",
        DONORS,
        schedule,
        RULE,
        model=_wide_template(),
        random_seed=3,
        fit_audit=False,
    )
    statuses = [window.status for window in replay.windows]
    assert statuses.count("fit-failed") == 2
    assert "reserved" in statuses
    assert len(calls) == 2
    assert replay.evidence_level == "fit-failed"
    assert all(window.bias_cumulative is None for window in replay.windows)
    assert "sampler exploded" in replay.recommendation()


def test_calibration_feeds_the_existing_kernel_without_validating_the_curve(
    monkeypatch,
):
    frame = _exact_frame() + np.random.default_rng(11).normal(0.0, 0.25, size=(8, 4))
    controls = ["c0", "c1", "c2"]
    request = ScheduleRequest(selection_origins=(3, 5), horizons=(2,), cadence=1)
    schedule = validate_schedule(frame, "y", controls, request)

    def available(self):
        return {
            "divergences": 0,
            "max_rhat": 1.0,
            "min_bulk_ess": 100.0,
            "diagnostics_status": "available",
            "diagnostics_reason": None,
        }

    monkeypatch.setattr(ModelAdapter, "sampling_diagnostics", available)
    replay = replay_history(
        frame,
        "y",
        controls,
        schedule,
        RULE,
        model=WeightedSumFitter(sample_kwargs=dict(LIGHT)),
        random_seed=5,
        fit_learned_null=True,
    )
    record = replay.calibration
    assert record.applicable is True
    assert record.curve_validated is False
    assert record.evidence_level == "model-conditional"
    assert record.null_samples is not None
    assert record.fold_sds is not None
    assert np.isfinite(record.null_samples).all()
    assert "not a validated false-positive rate" in replay.recommendation()
    assert record.overlapping_pair_count >= 1
    curve = record.conditional_curve()
    direct = operating_characteristics_from_calibration(record)
    pit = operating_characteristics(
        CheckResult(
            check_name="historical",
            passed=True,
            metadata={
                "null_samples": record.null_samples,
                "fold_sds": record.fold_sds,
                "rope_half_width": record.rope_half_width,
                "threshold": record.threshold,
            },
        )
    )
    np.testing.assert_allclose(curve.p_detect, direct.p_detect)
    np.testing.assert_allclose(direct.p_detect, pit.p_detect)
    np.testing.assert_allclose(direct.p_null, pit.p_null)
    assert direct.rope_half_width == pit.rope_half_width


def test_inapplicable_calibration_record_does_not_invent_a_curve():
    record = CalibrationRecord(
        evidence_level="inconclusive",
        reasons=("diagnostics are partial",),
        applicable=False,
        curve_validated=False,
        null_samples=None,
        fold_sds=np.array([0.2, 0.4]),
        fold_means=np.array([1.0, 1.2]),
        rope_half_width=0.0,
        threshold=0.95,
        scored_window_count=2,
        overlapping_pair_count=1,
        shared_training_pair_count=1,
        model="not-applicable",
    )
    with pytest.raises(ValueError, match="cannot feed"):
        operating_characteristics_from_calibration(record)
    with pytest.raises(ValueError, match="cannot be marked curve-validated"):
        CalibrationRecord(
            evidence_level="model-conditional",
            reasons=("no",),
            applicable=True,
            curve_validated=True,
            null_samples=np.array([0.0, 0.1]),
            fold_sds=np.array([0.2, 0.3]),
            fold_means=np.array([0.0, 0.1]),
            rope_half_width=0.0,
            threshold=0.95,
            scored_window_count=2,
            overlapping_pair_count=0,
            shared_training_pair_count=0,
            model="learned-null-normal",
        )


def test_adapter_matches_a_hand_built_check_result():
    theta = np.array([0.0, 0.2, -0.1, 0.4])
    sigmas = np.array([0.5, 0.8])
    record = CalibrationRecord(
        evidence_level="model-conditional",
        reasons=("conditional on the learned-null model",),
        applicable=True,
        curve_validated=False,
        null_samples=theta,
        fold_sds=sigmas,
        fold_means=np.array([0.0, 0.1]),
        rope_half_width=0.1,
        threshold=0.9,
        scored_window_count=2,
        overlapping_pair_count=0,
        shared_training_pair_count=1,
        model="learned-null-normal",
    )
    adapted = operating_characteristics_from_calibration(
        record, effect_sizes=[0.0, 0.5, 1.0]
    )
    direct = operating_characteristics(
        CheckResult(
            check_name="placebo",
            passed=True,
            metadata={
                "null_samples": theta,
                "fold_sds": sigmas,
                "rope_half_width": 0.1,
                "threshold": 0.9,
            },
        ),
        effect_sizes=[0.0, 0.5, 1.0],
    )
    np.testing.assert_allclose(adapted.p_detect, direct.p_detect)
    np.testing.assert_allclose(adapted.p_wrong_sign, direct.p_wrong_sign)
    assert adapted.mde == pytest.approx(direct.mde)


def test_public_names_are_importable_from_the_planner_module():
    from causalpy.checks import operating_characteristics_from_calibration as exported
    from causalpy.design import (
        HISTORICAL_REPLAY_ASSUMPTIONS,
        HistoricalReplay,
        HistoricalSchedule,
        WindowPlan,
    )

    assert exported is operating_characteristics_from_calibration
    assert HistoricalSchedule.__module__ == "causalpy.historical_validation"
    assert HistoricalReplay.__module__ == "causalpy.historical_validation"
    assert WindowPlan.__module__ == "causalpy.historical_validation"
    assert HISTORICAL_REPLAY_ASSUMPTIONS
    assert "Ferman" in HISTORICAL_REPLAY_ASSUMPTIONS[1]
    assert "Abadie" in HISTORICAL_REPLAY_ASSUMPTIONS[1]
