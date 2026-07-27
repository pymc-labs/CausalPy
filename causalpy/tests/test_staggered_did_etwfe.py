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
Unit tests for the ETWFE (extended two-way fixed effects) index construction
used by the staggered difference-in-differences experiment.

These tests are deliberately pure numpy/pandas: they exercise the module-level
private helpers in ``causalpy.experiments.staggered_did`` without constructing an
experiment or fitting any model.
"""

import warnings
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

import causalpy as cp
from causalpy.custom_exceptions import DataException
from causalpy.data.simulate_data import generate_staggered_did_data
from causalpy.experiments.staggered_did import (
    _ETWFE_K_SENTINEL,
    _build_etwfe_index,
    _format_cohort,
    _mundlak_means,
)
from causalpy.pymc_models import ETWFERegression, LinearRegression

NEVER = np.inf


def make_panel(n_units: int = 4, n_periods: int = 6) -> pd.DataFrame:
    """Build a small balanced panel with cohorts {2, 4} and one never-treated unit.

    Units 0 and 1 adopt at t=2, unit 2 adopts at t=4, unit 3 never adopts.
    """
    g_map = {0: 2.0, 1: 2.0, 2: 4.0, 3: NEVER}
    rows = []
    for unit in range(n_units):
        for time in range(n_periods):
            g = g_map[unit]
            rows.append(
                {
                    "unit": unit,
                    "time": time,
                    "G": g,
                    "treated": int(time >= g),
                    "y": float(unit) + 0.5 * time,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def panel() -> pd.DataFrame:
    """A 4 unit x 6 period panel with cohorts {2, 4} and one never-treated unit."""
    return make_panel()


COHORTS = [2.0, 4.0]


def build(panel: pd.DataFrame, **kwargs):
    """Call ``_build_etwfe_index`` on the fixture panel with the usual column names."""
    return _build_etwfe_index(
        panel,
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        never_treated_value=NEVER,
        cohorts=COHORTS,
        **kwargs,
    )


def build_quiet(panel: pd.DataFrame, **kwargs):
    """Build the index, ignoring the empty-cell UserWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return build(panel, **kwargs)


# ---------------------------------------------------------------------------
# 1. no leads -> effect indicator is exactly the treated column
# ---------------------------------------------------------------------------


def test_no_leads_effect_indicator_equals_treated(panel):
    """With n_leads=0 the effect indicator reduces to the treated column."""
    idx = build_quiet(panel)
    np.testing.assert_array_equal(
        idx.effect_indicator, panel["treated"].to_numpy(dtype=float)
    )
    # k_max_obs: unit 0/1 treated up to t=5 with G=2 -> k=3
    np.testing.assert_array_equal(idx.ev_grid, np.array([0, 1, 2, 3]))


def test_hand_computed_arrays(panel):
    """Every index array matches hand-computed values on the fixture panel."""
    idx = build_quiet(panel)

    np.testing.assert_array_equal(idx.unit_levels, np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(idx.time_levels, np.array([0, 1, 2, 3, 4, 5]))
    assert idx.cohorts == [2.0, 4.0]

    np.testing.assert_array_equal(idx.unit_idx, panel["unit"].to_numpy())
    np.testing.assert_array_equal(idx.time_idx, panel["time"].to_numpy())

    # cohort positions: units 0,1 -> 0; unit 2 -> 1; unit 3 (never) -> 0
    expected_cohort = np.repeat([0, 0, 1, 0], 6)
    np.testing.assert_array_equal(idx.cohort_idx, expected_cohort)

    # event-time positions: grid is [0,1,2,3] so position == k for treated cells
    # unit 0/1: t=0..5, G=2 -> k = -2,-1,0,1,2,3 ; in scope only for k>=0
    expected_ev = np.array(
        [0, 0, 0, 1, 2, 3]  # unit 0
        + [0, 0, 0, 1, 2, 3]  # unit 1
        + [0, 0, 0, 0, 0, 1]  # unit 2 (G=4 -> k=0 at t=4, k=1 at t=5)
        + [0, 0, 0, 0, 0, 0]  # unit 3 never treated
    )
    np.testing.assert_array_equal(idx.ev_idx, expected_ev)

    # never-treated rows carry the sentinel in k_eff
    never_rows = panel["unit"] == 3
    assert np.all(idx.k_eff[never_rows.to_numpy()] == _ETWFE_K_SENTINEL)
    # eventually-treated rows carry the real (top-binned) event time
    np.testing.assert_array_equal(
        idx.k_eff[panel["unit"].to_numpy() == 0], np.array([-2, -1, 0, 1, 2, 3])
    )


# ---------------------------------------------------------------------------
# 2. the grid omits the reference event time
# ---------------------------------------------------------------------------


def test_ev_grid_omits_reference_event_time(panel):
    """With leads, the reference event time has no column in the grid."""
    idx = build_quiet(panel, n_leads=3, reference_event_time=-1)
    np.testing.assert_array_equal(idx.ev_grid, np.array([-3, -2, 0, 1, 2, 3]))
    assert -1 not in idx.ev_grid.tolist()


def test_reference_cell_is_a_control(panel):
    """Observations at the reference event time get effect_indicator == 0."""
    idx = build_quiet(panel, n_leads=3)
    k = panel["time"].to_numpy() - panel["G"].to_numpy()
    at_reference = k == -1
    assert at_reference.sum() > 0
    assert np.all(idx.effect_indicator[at_reference] == 0.0)


# ---------------------------------------------------------------------------
# 3. indices are never negative, for any number of leads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_leads", [0, 1, 2, 3, 4])
def test_indices_never_negative(panel, n_leads):
    """No index array ever contains a negative sentinel, for any n_leads."""
    idx = build_quiet(panel, n_leads=n_leads)
    assert idx.ev_idx.min() >= 0
    assert idx.cohort_idx.min() >= 0
    assert idx.unit_idx.min() >= 0
    assert idx.time_idx.min() >= 0
    assert idx.ev_idx.max() < idx.ev_grid.size
    assert idx.cohort_idx.max() < len(idx.cohorts)
    assert set(np.unique(idx.effect_indicator)).issubset({0.0, 1.0})


@pytest.mark.parametrize("n_leads", [0, 1, 2, 3, 4])
def test_treated_cells_always_in_scope(panel, n_leads):
    """Treated observations always carry effect_indicator == 1."""
    idx = build_quiet(panel, n_leads=n_leads)
    treated = panel["treated"].to_numpy(dtype=bool)
    assert np.all(idx.effect_indicator[treated] == 1.0)


# ---------------------------------------------------------------------------
# 4. top-binning of long-run treated cells
# ---------------------------------------------------------------------------


def test_top_binning_keeps_effect_indicator_and_maps_to_last_column():
    """Treated cells beyond max_event_time bin into the last column, not into controls."""
    # one unit treated at t=0, observed to t=5 -> k = 0..5 including k=5
    rows = []
    for time in range(6):
        rows.append({"unit": 0, "time": time, "G": 0.0, "treated": 1})
        rows.append({"unit": 1, "time": time, "G": NEVER, "treated": 0})
    df = pd.DataFrame(rows)

    idx = _build_etwfe_index(
        df,
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        never_treated_value=NEVER,
        cohorts=[0.0],
        max_event_time=2,
    )
    np.testing.assert_array_equal(idx.ev_grid, np.array([0, 1, 2]))

    k = (df["time"] - df["G"]).to_numpy()
    tail = (df["treated"].to_numpy() == 1) & (k >= 2)
    # k = 2, 3, 4, 5 all land in the k=2 column and stay in scope
    assert np.all(idx.ev_idx[tail] == 2)
    assert np.all(idx.effect_indicator[tail] == 1.0)
    assert np.all(idx.k_eff[tail] == 2)

    # the binned observations are counted, they do not vanish
    assert idx.cell_counts.sum() == int(df["treated"].sum())
    assert idx.cell_counts[0, 2] == 4


def test_top_binning_preserves_total_treated_count(panel):
    """Binning moves treated observations between columns but never drops them."""
    full = build_quiet(panel)
    binned = build_quiet(panel, max_event_time=1)
    assert binned.cell_counts.sum() == full.cell_counts.sum()
    np.testing.assert_array_equal(binned.ev_grid, np.array([0, 1]))


# ---------------------------------------------------------------------------
# 5. far leads stay controls
# ---------------------------------------------------------------------------


def test_far_leads_are_controls(panel):
    """Pre-treatment cells beyond n_leads are not binned; they remain controls."""
    idx = build_quiet(panel, n_leads=1)
    k = panel["time"].to_numpy() - panel["G"].to_numpy()
    # with n_leads=1 and ref=-1 there are no lead cells at all
    np.testing.assert_array_equal(
        idx.effect_indicator, panel["treated"].to_numpy(dtype=float)
    )

    idx3 = build_quiet(panel, n_leads=3)
    far = np.isfinite(k) & (k < -3)
    if far.any():
        assert np.all(idx3.effect_indicator[far] == 0.0)
    # k = -4 exists for unit 2 (G=4, t=0); it is a control
    unit2_t0 = (panel["unit"] == 2) & (panel["time"] == 0)
    assert idx3.effect_indicator[unit2_t0.to_numpy()][0] == 0.0
    # k = -2 for unit 2 (t=2) is inside the lead window and is in scope
    unit2_t2 = (panel["unit"] == 2) & (panel["time"] == 2)
    assert idx3.effect_indicator[unit2_t2.to_numpy()][0] == 1.0


def test_never_treated_never_in_scope(panel):
    """Never-treated units never load on the effect surface."""
    for n_leads in range(5):
        idx = build_quiet(panel, n_leads=n_leads)
        never = (panel["unit"] == 3).to_numpy()
        assert np.all(idx.effect_indicator[never] == 0.0)
        assert np.all(idx.ev_idx[never] == 0)
        assert np.all(idx.cohort_idx[never] == 0)


# ---------------------------------------------------------------------------
# 6-7. ATT weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_leads", [0, 2, 3, 4])
def test_att_weights_sum_to_one_and_zero_in_lead_columns(panel, n_leads):
    """Weights are a proper average over treated cells only."""
    idx = build_quiet(panel, n_leads=n_leads)
    assert idx.att_weights.sum() == pytest.approx(1.0)
    lead_cols = np.flatnonzero(idx.ev_grid < 0)
    if lead_cols.size:
        assert np.all(idx.att_weights[:, lead_cols] == 0.0)
    # and the lead columns are genuinely populated in cell_counts
    if n_leads >= 2:
        assert idx.cell_counts[:, lead_cols].sum() > 0


def test_att_weights_match_groupby_recomputation(panel):
    """Weights match an independent groupby().size() computation."""
    idx = build_quiet(panel, n_leads=3)

    treated = panel[panel["treated"] == 1].copy()
    treated["k"] = (treated["time"] - treated["G"]).astype(int)
    counts = treated.groupby(["G", "k"]).size()
    total = counts.sum()

    expected = np.zeros_like(idx.att_weights)
    for (g, k), n in counts.items():
        i = idx.cohorts.index(g)
        j = int(np.flatnonzero(idx.ev_grid == k)[0])
        expected[i, j] = n / total

    np.testing.assert_allclose(idx.att_weights, expected)


# ---------------------------------------------------------------------------
# 8. unbalanced panel
# ---------------------------------------------------------------------------


def test_unbalanced_panel(panel):
    """Dropping rows keeps the weights normalised and the counts consistent."""
    rng = np.random.default_rng(42)
    keep = rng.random(len(panel)) > 0.2
    # keep at least the cells that define the grid
    unbalanced = panel[keep].reset_index(drop=True)

    idx = build_quiet(unbalanced, n_leads=2)

    assert idx.att_weights.sum() == pytest.approx(1.0)
    assert idx.effect_indicator.shape[0] == len(unbalanced)
    assert idx.cell_counts.sum() == int(idx.effect_indicator.sum())
    assert idx.ev_idx.min() >= 0

    treated = unbalanced[unbalanced["treated"] == 1].copy()
    treated["k"] = (treated["time"] - treated["G"]).astype(int)
    counts = treated.groupby(["G", "k"]).size()
    for (g, k), n in counts.items():
        i = idx.cohorts.index(g)
        j = int(np.flatnonzero(idx.ev_grid == k)[0])
        assert idx.att_weights[i, j] == pytest.approx(n / counts.sum())


def test_unbalanced_panel_with_shuffled_index(panel):
    """Index construction is positional and survives a non-monotonic row index."""
    shuffled = panel.sample(frac=1.0, random_state=0)
    idx = build_quiet(shuffled, n_leads=2)
    np.testing.assert_array_equal(
        idx.effect_indicator[shuffled["treated"].to_numpy() == 1],
        np.ones(int(shuffled["treated"].sum())),
    )
    assert idx.att_weights.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 9. Mundlak means
# ---------------------------------------------------------------------------


def test_mundlak_means_raw_and_centred(panel):
    """Raw means match a groupby transform; centred means have zero sample mean."""
    du_c, dt_c, du_r, dt_r = _mundlak_means(
        panel,
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
    )

    expected_unit = panel.groupby("unit")["treated"].transform("mean").to_numpy()
    expected_time = panel.groupby("time")["treated"].transform("mean").to_numpy()
    np.testing.assert_allclose(du_r, expected_unit)
    np.testing.assert_allclose(dt_r, expected_time)

    np.testing.assert_allclose(du_c, du_r - du_r.mean())
    np.testing.assert_allclose(dt_c, dt_r - dt_r.mean())
    assert du_c.mean() == pytest.approx(0.0, abs=1e-12)
    assert dt_c.mean() == pytest.approx(0.0, abs=1e-12)

    # never-treated unit has a raw mean of exactly zero
    never = (panel["unit"] == 3).to_numpy()
    assert np.all(du_r[never] == 0.0)


def test_mundlak_means_unbalanced(panel):
    """Mundlak means are observation-aligned on an unbalanced panel."""
    unbalanced = panel.drop(index=[0, 7, 13]).reset_index(drop=True)
    du_c, dt_c, du_r, dt_r = _mundlak_means(
        unbalanced,
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
    )
    assert du_r.shape == (len(unbalanced),)
    np.testing.assert_allclose(
        du_r, unbalanced.groupby("unit")["treated"].transform("mean").to_numpy()
    )
    assert du_c.mean() == pytest.approx(0.0, abs=1e-12)
    assert dt_c.mean() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 10. empty cells and empty columns
# ---------------------------------------------------------------------------


def test_empty_cells_are_recorded_without_warning(panel):
    """Individually empty (cohort, event time) cells are recorded, not warned about.

    Sparse corners are the normal shape of a staggered panel -- the last-adopting
    cohort never reaches the largest event times -- so warning on them would fire
    on essentially every real panel. They are recorded for downstream reporting.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        idx = build(panel, n_leads=2)
    # cohort 4 has no k=2 or k=3 observations (it adopts at t=4, panel ends t=5)
    assert (4.0, 2) in idx.dropped_cells
    assert (4.0, 3) in idx.dropped_cells
    # every recorded cell really is empty
    for g, k in idx.dropped_cells:
        i = idx.cohorts.index(g)
        cols = np.flatnonzero(idx.ev_grid == k)
        if cols.size:
            assert idx.cell_counts[i, int(cols[0])] == 0


def test_empty_column_is_dropped_and_shrinks_grid():
    """An event-time column empty for every cohort is removed from the grid."""
    # Two cohorts adopting at t=1 and t=2, panel ends at t=3, so k=3 exists only
    # for the first cohort. Add a never-treated unit for identification.
    rows = []
    g_map = {0: 1.0, 1: 2.0, 2: NEVER}
    for unit, g in g_map.items():
        for time in range(4):
            rows.append({"unit": unit, "time": time, "G": g, "treated": int(time >= g)})
    df = pd.DataFrame(rows)

    # n_leads=3 with ref=-1: leads are k in {-3, -2}. No unit has k=-3
    # (the earliest cohort adopts at t=1, so its minimum k is -1), so the k=-3
    # column is empty for every cohort and must be removed.
    with pytest.warns(UserWarning, match="removed from the ETWFE event-time grid"):
        idx = _build_etwfe_index(
            df,
            unit_variable_name="unit",
            time_variable_name="time",
            treated_variable_name="treated",
            never_treated_value=NEVER,
            cohorts=[1.0, 2.0],
            n_leads=3,
        )

    assert -3 not in idx.ev_grid.tolist()
    assert -2 in idx.ev_grid.tolist()
    assert idx.ev_idx.min() >= 0
    assert idx.ev_idx.max() < idx.ev_grid.size
    # every surviving column has at least one observation
    assert np.all(idx.cell_counts.sum(axis=0) > 0)
    assert idx.att_weights.sum() == pytest.approx(1.0)
    assert idx.att_weights.shape == idx.cell_counts.shape


def test_no_warning_when_every_cell_is_populated():
    """A fully populated design does not warn."""
    rows = []
    g_map = {0: 1.0, 1: 1.0, 2: NEVER}
    for unit, g in g_map.items():
        for time in range(3):
            rows.append({"unit": unit, "time": time, "G": g, "treated": int(time >= g)})
    df = pd.DataFrame(rows)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        idx = _build_etwfe_index(
            df,
            unit_variable_name="unit",
            time_variable_name="time",
            treated_variable_name="treated",
            never_treated_value=NEVER,
            cohorts=[1.0],
        )
    assert idx.dropped_cells == []
    np.testing.assert_array_equal(idx.ev_grid, np.array([0, 1]))


# ---------------------------------------------------------------------------
# 11. validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_leads, reference_event_time",
    [(0, -2), (0, 0), (0, 1), (2, -3), (2, 0), (2, 1), (3, -4)],
)
def test_reference_event_time_validation(panel, n_leads, reference_event_time):
    """Out-of-range reference event times raise ValueError."""
    with pytest.raises(ValueError, match="reference_event_time"):
        build(panel, n_leads=n_leads, reference_event_time=reference_event_time)


@pytest.mark.parametrize("n_leads, reference_event_time", [(0, -1), (2, -1), (2, -2)])
def test_reference_event_time_accepted(panel, n_leads, reference_event_time):
    """In-range reference event times are accepted."""
    idx = build_quiet(panel, n_leads=n_leads, reference_event_time=reference_event_time)
    assert reference_event_time not in idx.ev_grid.tolist()


def test_lead_window_is_complement_of_reference(panel):
    """Every estimated lead column carries observations, for any reference.

    The lead window must be the exact complement of ``reference_event_time``
    within ``[-n_leads, -1]``. If it were hard-coded to ``k <= -2`` then with
    ``reference_event_time=-2`` the ``k=-1`` column would appear in the grid
    while no observation could ever load on it.
    """
    idx = build_quiet(panel, n_leads=2, reference_event_time=-2)
    assert idx.ev_grid.tolist()[:1] == [-1]
    # k = -1 is an estimated lead column, so it must carry observations
    col = int(np.flatnonzero(idx.ev_grid == -1)[0])
    assert idx.cell_counts[:, col].sum() > 0
    # and the observations loading on it really are the k == -1 cells
    on_col = (idx.ev_idx == col) & (idx.effect_indicator == 1.0)
    assert np.all(idx.k_eff[on_col] == -1)


def test_default_reference_lead_window_unchanged(panel):
    """With the default reference, k=-1 cells stay controls."""
    idx = build_quiet(panel, n_leads=3)
    assert -1 not in idx.ev_grid.tolist()
    eventually_treated = idx.k_eff != _ETWFE_K_SENTINEL
    at_ref = eventually_treated & (idx.k_eff == -1)
    assert at_ref.any()
    assert np.all(idx.effect_indicator[at_ref] == 0.0)


def test_negative_n_leads_raises(panel):
    """Negative n_leads raises ValueError."""
    with pytest.raises(ValueError, match="n_leads"):
        build(panel, n_leads=-1)


def test_negative_max_event_time_raises(panel):
    """Negative max_event_time raises ValueError."""
    with pytest.raises(ValueError, match="max_event_time"):
        build(panel, max_event_time=-1)


def test_missing_g_column_raises(panel):
    """A panel without the precomputed G column raises ValueError."""
    with pytest.raises(ValueError, match="'G' column"):
        build(panel.drop(columns=["G"]))


def test_unknown_cohort_raises(panel):
    """A treatment time absent from the supplied cohorts raises ValueError."""
    with pytest.raises(ValueError, match="appear in the data but not"):
        _build_etwfe_index(
            panel,
            unit_variable_name="unit",
            time_variable_name="time",
            treated_variable_name="treated",
            never_treated_value=NEVER,
            cohorts=[2.0],
        )


def test_no_treated_observations_raises(panel):
    """A panel with no treated observations raises ValueError."""
    untreated = panel.copy()
    untreated["treated"] = 0
    with pytest.raises(ValueError, match="No treated observations"):
        build(untreated)


def test_no_cohorts_raises(panel):
    """An empty cohort list raises ValueError."""
    with pytest.raises(ValueError, match="No adopting cohorts"):
        _build_etwfe_index(
            panel,
            unit_variable_name="unit",
            time_variable_name="time",
            treated_variable_name="treated",
            never_treated_value=NEVER,
            cohorts=[],
        )


def test_never_treated_value_can_be_nan():
    """NaN works as the never-treated sentinel."""
    rows = []
    g_map = {0: 1.0, 1: np.nan}
    for unit, g in g_map.items():
        for time in range(3):
            treated = 0 if pd.isna(g) else int(time >= g)
            rows.append({"unit": unit, "time": time, "G": g, "treated": treated})
    df = pd.DataFrame(rows)

    idx = _build_etwfe_index(
        df,
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        never_treated_value=np.nan,
        cohorts=[1.0, np.nan],
    )
    assert idx.cohorts == [1.0]
    np.testing.assert_array_equal(
        idx.effect_indicator, df["treated"].to_numpy(dtype=float)
    )
    assert idx.att_weights.sum() == pytest.approx(1.0)


def test_index_is_frozen(panel):
    """The returned index bundle is immutable."""
    idx = build_quiet(panel)
    with pytest.raises(FrozenInstanceError):
        idx.ev_grid = np.array([0])


# ===========================================================================
# WP3 -- experiment wiring for estimator="etwfe"
#
# Every *numeric* assertion below runs on the scikit-learn path. The
# ``mock_pymc_sample`` fixture replaces ``pm.sample`` with a prior predictive
# draw, so posterior recovery is impossible on the PyMC path; those tests check
# schema and plumbing only.
# ===========================================================================

BASE_KWARGS = {
    "formula": "y ~ 1 + C(unit) + C(time)",
    "unit_variable_name": "unit",
    "time_variable_name": "time",
    "treated_variable_name": "treated",
    "treatment_time_variable_name": "treatment_time",
}


def noise_free_panel(**kwargs) -> pd.DataFrame:
    """Noise-free panel with cohort-heterogeneous, growing treatment effects.

    With ``sigma=0`` the generator satisfies ``y == y0 + tau`` exactly, so a
    correctly specified saturated design must reproduce ``tau[g, k]`` and the
    treated-cell-averaged ATT to machine precision.
    """
    defaults: dict = {
        "n_units": 24,
        "n_time_periods": 12,
        "treatment_cohorts": {4: 8, 8: 8},
        "treatment_effects": lambda k: 1 + 0.4 * k,
        "cohort_effect_scale": {4: 1.0, 8: 1.6},
        "sigma": 0.0,
        "seed": 42,
    }
    defaults.update(kwargs)
    return generate_staggered_did_data(**defaults)


def noisy_panel(**kwargs) -> pd.DataFrame:
    """Small noisy panel used for the plumbing/schema tests."""
    defaults: dict = {
        "n_units": 16,
        "n_time_periods": 10,
        "treatment_cohorts": {3: 5, 6: 5},
        "sigma": 0.4,
        "seed": 7,
    }
    defaults.update(kwargs)
    return generate_staggered_did_data(**defaults)


def fit_ols(data: pd.DataFrame, **kwargs):
    """Fit the ETWFE experiment with a plain OLS model."""
    return cp.StaggeredDifferenceInDifferences(
        data,
        model=SklearnLinearRegression(),
        estimator="etwfe",
        **{**BASE_KWARGS, **kwargs},
    )


@pytest.fixture(scope="module")
def ols_result():
    """A default (no leads, no covariates) OLS ETWFE fit on noise-free data."""
    return fit_ols(noise_free_panel())


# ---------------------------------------------------------------------------
# Tier 1 -- exact algebraic recovery on noise-free data
# ---------------------------------------------------------------------------


def test_etwfe_ols_recovers_true_tau_surface_exactly(ols_result):
    """On noise-free data the saturated OLS design reproduces the generator exactly.

    This is the cheapest possible proof that the index construction and the
    weight matrix are right: any error in ``ev_idx``, ``cohort_idx``,
    ``effect_indicator``, top-binning or ``W`` shows up here as a large
    deviation, with no MCMC involved.
    """
    df = noise_free_panel()
    truth = (
        df.loc[df["treated"] == 1]
        .assign(k=lambda d: (d["time"] - d["treatment_time"]).astype(int))
        .groupby(["treatment_time", "k"])["tau"]
        .mean()
    )

    surface = ols_result.tau_surface_
    assert len(surface) == len(truth)
    for _, row in surface.iterrows():
        expected = truth.loc[(row["cohort"], row["event_time"])]
        assert row["att"] == pytest.approx(expected, abs=1e-8)

    true_att = df.loc[df["treated"] == 1, "tau"].mean()
    assert ols_result.att_ == pytest.approx(true_att, abs=1e-8)


def test_etwfe_ols_exact_with_covariates_and_leads():
    """Exactness survives additive covariates and estimated lead terms."""
    df = noise_free_panel(n_covariates=2)
    result = fit_ols(df, covariates=["x1", "x2"], n_leads=3)

    true_att = df.loc[df["treated"] == 1, "tau"].mean()
    assert result.att_ == pytest.approx(true_att, abs=1e-8)

    # With no anticipation and no noise, every lead coefficient is exactly zero.
    leads = result.tau_surface_[result.tau_surface_["event_time"] < 0]
    assert len(leads) > 0
    np.testing.assert_allclose(leads["att"].to_numpy(), 0.0, atol=1e-8)


def test_etwfe_ols_exact_on_unbalanced_panel():
    """Dropping rows at random does not disturb the index alignment."""
    df = noise_free_panel()
    unbalanced = (
        df.sample(frac=0.9, random_state=1)
        .sort_values(["unit", "time"])
        .reset_index(drop=True)
    )
    result = fit_ols(unbalanced)
    true_att = unbalanced.loc[unbalanced["treated"] == 1, "tau"].mean()
    assert result.att_ == pytest.approx(true_att, abs=1e-8)


# ---------------------------------------------------------------------------
# OLS smoke and internal consistency
# ---------------------------------------------------------------------------


def test_etwfe_ols_smoke(ols_result):
    """The OLS ETWFE fit exposes the full documented attribute surface."""
    assert ols_result.estimator == "etwfe"
    assert ols_result.conditioning == "dummy"
    assert isinstance(ols_result.att_, float)
    assert isinstance(ols_result.att_se_, float)
    assert ols_result.att_se_ >= 0.0

    assert list(ols_result.att_event_time_.columns) == [
        "event_time",
        "att",
        "att_std",
        "n_obs",
    ]
    assert list(ols_result.att_group_time_.columns) == [
        "cohort",
        "time",
        "att",
        "att_std",
        "n_obs",
    ]
    assert set(ols_result.tau_surface_.columns) == {
        "cohort",
        "event_time",
        "att",
        "att_std",
        "n_obs",
    }
    assert isinstance(ols_result.att_weights_, pd.DataFrame)
    assert ols_result.att_weights_.to_numpy().sum() == pytest.approx(1.0)
    assert isinstance(ols_result.event_time_grid_, np.ndarray)
    assert ols_result.etwfe_formula_.startswith("y ~ 1 + C(unit) + C(time)")
    assert "y_hat0" in ols_result.data_.columns
    assert "tau_hat" in ols_result.data_.columns
    assert not hasattr(ols_result, "hdi_prob_")

    # group-time cells are post-treatment only and satisfy t = g + k
    assert (
        ols_result.att_group_time_["time"] >= ols_result.att_group_time_["cohort"]
    ).all()


def test_etwfe_ols_att_is_weighted_sum_of_surface(ols_result):
    """``att_`` is exactly the ATT-weighted sum over the estimated surface."""
    weights = ols_result.att_weights_
    total = sum(
        weights.loc[row["cohort"], row["event_time"]] * row["att"]
        for _, row in ols_result.tau_surface_.iterrows()
    )
    assert ols_result.att_ == pytest.approx(total, abs=1e-10)


def test_etwfe_ols_column_weights_sum_to_one(ols_result):
    """Every event-time column aggregates a proper weighted average."""
    for position in range(ols_result.event_time_grid_.size):
        weights, n_obs = ols_result._etwfe_column_weights(position)
        assert n_obs > 0
        assert weights.sum() == pytest.approx(1.0)
        # weights load only on cell columns, never on unit/time effects
        cell_columns = set(ols_result._etwfe_cell_columns.values())
        assert set(np.flatnonzero(weights)).issubset(cell_columns)


def test_etwfe_tau_hat_reproduces_true_effect(ols_result):
    """On a perfect fit ``tau_hat`` equals the model-implied cell effect."""
    df = noise_free_panel()
    treated = ols_result.data_["treated"] == 1
    np.testing.assert_allclose(
        ols_result.data_.loc[treated, "tau_hat"].to_numpy(),
        df.loc[treated.to_numpy(), "tau"].to_numpy(),
        atol=1e-8,
    )
    # untreated rows keep the imputation path's NaN convention
    assert ols_result.data_.loc[~treated, "tau_hat"].isna().all()


# ---------------------------------------------------------------------------
# Event-time grid behaviour
# ---------------------------------------------------------------------------


def test_etwfe_no_reference_row_and_no_leads_by_default(ols_result):
    """Without leads the event-time table is post-treatment only."""
    event_times = ols_result.att_event_time_["event_time"].to_numpy()
    assert (event_times >= 0).all()
    assert -1 not in event_times


def test_etwfe_leads_add_negative_rows_but_never_the_reference():
    """``n_leads=3`` estimates -3 and -2; the reference -1 is omitted entirely."""
    result = fit_ols(noise_free_panel(), n_leads=3)
    event_times = set(result.att_event_time_["event_time"].tolist())
    assert {-3, -2}.issubset(event_times)
    assert -1 not in event_times
    assert -1 not in set(result.tau_surface_["event_time"].tolist())
    assert -1 not in set(result.event_time_grid_.tolist())


def test_etwfe_max_event_time_binning_preserves_treated_n_obs():
    """Top-binning moves treated observations between columns but loses none."""
    df = noise_free_panel()
    unbinned = fit_ols(df)
    binned = fit_ols(df, max_event_time=2)

    assert binned.event_time_grid_.max() == 2
    assert (
        binned.att_event_time_["n_obs"].sum() == unbinned.att_event_time_["n_obs"].sum()
    )
    assert binned.att_event_time_["n_obs"].sum() == int(df["treated"].sum())


def test_etwfe_event_window_filters_reporting_only():
    """``event_window`` trims the report; it does not change what is estimated."""
    df = noise_free_panel()
    result = fit_ols(df, n_leads=3, event_window=(0, 2))
    assert result.att_event_time_["event_time"].tolist() == [0, 1, 2]
    # the leads were still estimated, they are just not reported
    assert -3 in set(result.tau_surface_["event_time"].tolist())


# ---------------------------------------------------------------------------
# Covariates and standard errors
# ---------------------------------------------------------------------------


def test_etwfe_covariates_change_labels():
    """Requested covariates appear as design columns."""
    df = noise_free_panel(n_covariates=1)
    without = fit_ols(df)
    with_cov = fit_ols(df, covariates=["x1"])
    assert "x1" not in without.labels
    assert "x1" in with_cov.labels
    assert len(with_cov.labels) == len(without.labels) + 1


def test_etwfe_unknown_covariate_raises():
    """A covariate that is not a column is caught during validation."""
    with pytest.raises(DataException, match="not found in data"):
        fit_ols(noise_free_panel(), covariates=["nope"])


def test_etwfe_cluster_se_exceeds_classical_under_serial_correlation():
    """Clustering by unit inflates the standard error when errors are unit-persistent."""
    df = (
        noise_free_panel(sigma=0.1).sort_values(["unit", "time"]).reset_index(drop=True)
    )
    rng = np.random.default_rng(0)
    n_periods = df["time"].nunique()
    n_units = df["unit"].nunique()
    walk = np.concatenate(
        [np.cumsum(rng.normal(0, 0.6, n_periods)) for _ in range(n_units)]
    )
    df["y"] = df["y"] + walk

    cluster = fit_ols(df, se_type="cluster")
    classical = fit_ols(df, se_type="classical")

    assert cluster.att_ == pytest.approx(classical.att_)
    assert cluster.att_se_ > classical.att_se_


def test_etwfe_bad_se_type_raises():
    """``se_type`` is validated."""
    with pytest.raises(ValueError, match="se_type"):
        fit_ols(noise_free_panel(), se_type="robust")


# ---------------------------------------------------------------------------
# Argument validation and estimator dispatch
# ---------------------------------------------------------------------------


def test_etwfe_warns_about_ignored_formula_rhs():
    """Extra right-hand-side terms are ignored, loudly."""
    df = noise_free_panel(n_covariates=1)
    with pytest.warns(UserWarning, match="ignores the formula right-hand side"):
        result = fit_ols(df, formula="y ~ 1 + C(unit) + C(time) + x1")
    # the term really is ignored: no x1 column in the design
    assert "x1" not in result.labels


def test_etwfe_canonical_formula_does_not_warn():
    """The released canonical formula must stay warning-free."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fit_ols(noise_free_panel())


@pytest.mark.parametrize(
    "kwargs, name",
    [
        ({"conditioning": "dummy"}, "conditioning"),
        ({"n_leads": 2}, "n_leads"),
        ({"max_event_time": 3}, "max_event_time"),
        ({"covariates": ["x1"]}, "covariates"),
        ({"se_type": "classical"}, "se_type"),
    ],
)
def test_imputation_rejects_etwfe_only_arguments(kwargs, name):
    """Silently ignoring a statistically meaningful argument is not acceptable."""
    with pytest.raises(ValueError, match=name):
        cp.StaggeredDifferenceInDifferences(
            noise_free_panel(n_covariates=1),
            model=SklearnLinearRegression(),
            estimator="imputation",
            **BASE_KWARGS,
            **kwargs,
        )


def test_mundlak_conditioning_with_sklearn_raises():
    """Mundlak needs partial pooling, so it is a PyMC-only option."""
    with pytest.raises(ValueError, match="not available for scikit-learn"):
        fit_ols(noise_free_panel(), conditioning="mundlak")


def test_unknown_estimator_raises():
    """``estimator`` is validated before anything else happens."""
    with pytest.raises(ValueError, match="estimator must be"):
        cp.StaggeredDifferenceInDifferences(
            noise_free_panel(),
            model=SklearnLinearRegression(),
            estimator="wooldridge",
            **BASE_KWARGS,
        )


def test_etwfe_rejects_non_etwfe_pymc_model():
    """A generic PyMC model cannot carry the saturated ETWFE design."""
    with pytest.raises(ValueError, match="ETWFERegression"):
        cp.StaggeredDifferenceInDifferences(
            noise_free_panel(),
            model=LinearRegression(),
            estimator="etwfe",
            **BASE_KWARGS,
        )


def test_imputation_path_is_untouched():
    """The default estimator still runs the imputation algorithm unchanged."""
    result = cp.StaggeredDifferenceInDifferences(
        noisy_panel(),
        model=SklearnLinearRegression(),
        **BASE_KWARGS,
    )
    assert result.estimator == "imputation"
    assert result.conditioning is None
    assert result.att_ is None
    assert not hasattr(result, "tau_surface_")
    assert list(result.att_event_time_.columns) == [
        "event_time",
        "att",
        "att_std",
        "n_obs",
    ]


# ---------------------------------------------------------------------------
# Reporting entry points on the OLS path
# ---------------------------------------------------------------------------


def test_etwfe_ols_reporting_entry_points(ols_result):
    """``summary``/``plot``/``effect_summary``/``get_plot_data`` all run."""
    ols_result.summary()
    fig, axes = ols_result.plot()
    assert len(axes) == 1
    plt.close(fig)
    assert not ols_result.get_plot_data().empty
    assert ols_result.effect_summary().text


# ---------------------------------------------------------------------------
# PyMC path -- schema and plumbing only (mocked sampling)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("conditioning", ["mundlak", "dummy"])
def test_etwfe_bayesian_schema(mock_pymc_sample, conditioning):
    """Both conditionings produce the documented Bayesian result schema."""
    result = cp.StaggeredDifferenceInDifferences(
        noisy_panel(),
        estimator="etwfe",
        conditioning=conditioning,
        n_leads=2,
        **BASE_KWARGS,
    )
    assert isinstance(result.model, ETWFERegression)
    assert result.conditioning == conditioning
    assert list(result.att_event_time_.columns) == [
        "event_time",
        "att",
        "att_lower",
        "att_upper",
        "n_obs",
    ]
    assert list(result.att_group_time_.columns) == [
        "cohort",
        "time",
        "att",
        "att_lower",
        "att_upper",
    ]
    assert set(result.tau_surface_.columns) == {
        "cohort",
        "event_time",
        "att",
        "att_lower",
        "att_upper",
        "n_obs",
    }
    assert result.hdi_prob_ == pytest.approx(0.94)
    assert -1 not in set(result.att_event_time_["event_time"].tolist())
    assert (result.att_group_time_["time"] >= result.att_group_time_["cohort"]).all()

    assert isinstance(result.att_, xr.DataArray)
    assert {"chain", "draw"}.issubset(result.att_.dims)
    assert result.att_se_ is None

    assert "y_hat0" in result.data_.columns
    assert "tau_hat" in result.data_.columns
    assert result.data_["y_hat0"].notna().all()

    if conditioning == "mundlak":
        assert "dbar_unit" in result.data_.columns
        assert "dbar_time" in result.data_.columns


@pytest.mark.integration
def test_etwfe_bayesian_default_model(mock_pymc_sample):
    """``model=None`` with ``estimator='etwfe'`` instantiates ``ETWFERegression``."""
    result = cp.StaggeredDifferenceInDifferences(
        noisy_panel(), estimator="etwfe", **BASE_KWARGS
    )
    assert isinstance(result.model, ETWFERegression)
    assert result.conditioning == "mundlak"
    # the class default is untouched for the imputation estimator
    assert cp.StaggeredDifferenceInDifferences._default_model_class is LinearRegression


@pytest.mark.integration
def test_etwfe_bayesian_reporting_entry_points(mock_pymc_sample):
    """Every downstream reporting entry point runs on the Bayesian ETWFE path."""
    result = cp.StaggeredDifferenceInDifferences(
        noisy_panel(), estimator="etwfe", **BASE_KWARGS
    )
    result.summary()
    fig, axes = result.plot()
    assert len(axes) == 1
    plt.close(fig)
    assert result.effect_summary().text

    default = result.get_plot_data()
    assert list(default.columns) == [
        "event_time",
        "att",
        "att_lower",
        "att_upper",
        "n_obs",
    ]
    narrow = result.get_plot_data(hdi_prob=0.5)
    assert len(narrow) == len(default)
    assert (narrow["att_upper"] - narrow["att_lower"]).sum() < (
        default["att_upper"] - default["att_lower"]
    ).sum()


@pytest.mark.integration
def test_etwfe_bayesian_covariates(mock_pymc_sample):
    """Covariates are carried into the model's ``coeffs`` coordinate."""
    df = noisy_panel(n_covariates=2)
    result = cp.StaggeredDifferenceInDifferences(
        df, estimator="etwfe", covariates=["x1", "x2"], **BASE_KWARGS
    )
    # ``labels`` advertises the whole ETWFE parameter block (see WP5); the
    # covariates are the trailing entries and are the only ones indexing ``beta``.
    assert result.labels[-2:] == ["x1", "x2"]
    assert result._etwfe_covariate_labels == ["x1", "x2"]
    assert "beta" in result.model.idata.posterior


# ===========================================================================
# WP5 -- reporting prose and the maketables coefficient hook
# ===========================================================================

#: The phrase that the ETWFE headline adds to the effect-summary prose. The
#: imputation estimator must never produce it.
ATT_PHRASE = "Average effect on the treated (in-model, treated-cell weighted)"


def test_effect_summary_ols_reports_in_model_att_with_se(ols_result):
    """The OLS ETWFE headline leads with the aggregated ATT and its SE."""
    summary = ols_result.effect_summary()
    assert summary.text.startswith(ATT_PHRASE)
    assert f"{ols_result.att_:.2f}" in summary.text
    assert f"(SE {ols_result.att_se_:.2f})" in summary.text


def test_effect_summary_table_is_unchanged_by_the_att_headline(ols_result):
    """``.table`` stays exactly ``att_event_time_``.

    Adding an ATT row would break every downstream ``event_time < 0`` filter --
    the pre-treatment placebo check and both plot methods rely on it.
    """
    summary = ols_result.effect_summary()
    pd.testing.assert_frame_equal(summary.table, ols_result.att_event_time_)


@pytest.mark.integration
def test_effect_summary_bayesian_reports_in_model_att_with_hdi(mock_pymc_sample):
    """The Bayesian ETWFE headline reports the posterior of the in-model ATT."""
    result = cp.StaggeredDifferenceInDifferences(
        noisy_panel(), estimator="etwfe", **BASE_KWARGS
    )
    summary = result.effect_summary()
    assert summary.text.startswith(ATT_PHRASE)
    assert f"{int(result.hdi_prob_ * 100)}% HDI" in summary.text
    assert f"{float(result.att_.mean()):.2f}" in summary.text
    # and the table is still purely the event-time ATTs
    assert list(summary.table.columns) == list(result.att_event_time_.columns)


def test_effect_summary_imputation_prose_is_unchanged():
    """The imputation prose must not acquire the ETWFE headline.

    ``att_`` is a *class* attribute defaulting to None, so ``hasattr`` is True
    even here; the reporting code has to gate on the value.
    """
    result = cp.StaggeredDifferenceInDifferences(
        noisy_panel(),
        model=SklearnLinearRegression(),
        **BASE_KWARGS,
    )
    assert hasattr(result, "att_")
    assert result.att_ is None
    text = result.effect_summary().text
    assert ATT_PHRASE not in text
    assert text.startswith("Staggered DiD analysis:")


def test_detect_experiment_type_still_routes_etwfe_to_staggered_did(ols_result):
    """``_detect_experiment_type`` keys on ``att_event_time_``, which ETWFE has."""
    from causalpy.reporting import _detect_experiment_type

    assert _detect_experiment_type(ols_result) == "staggered_did"


def test_maketables_coef_table_etwfe_ols(ols_result):
    """The OLS ETWFE design already exposes patsy column names, so this renders."""
    table = ols_result.__maketables_coef_table__
    assert list(table.columns) == ["b", "se", "p", "t", "ci95l", "ci95u"]
    assert len(table) == len(ols_result.labels)


def test_maketables_coef_draws_hook_is_none_for_imputation():
    """The escape hatch is inert unless the fit is ETWFE *and* Bayesian."""
    result = cp.StaggeredDifferenceInDifferences(
        noisy_panel(), model=SklearnLinearRegression(), **BASE_KWARGS
    )
    assert result.__maketables_coef_draws__ is None


def test_maketables_coef_draws_hook_is_none_for_etwfe_ols(ols_result):
    """The OLS path resolves coefficients through the sklearn adapter instead."""
    assert ols_result.__maketables_coef_draws__ is None


def test_maketables_coef_draws_hook_absent_on_other_experiments():
    """No other experiment class defines the hook, so ``getattr`` yields None."""
    import causalpy as _cp

    df = _cp.load_data("did")
    result = _cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=SklearnLinearRegression(),
    )
    assert getattr(result, "__maketables_coef_draws__", None) is None


@pytest.mark.integration
def test_maketables_coef_table_etwfe_pymc_without_covariates(mock_pymc_sample):
    """A covariate-free ETWFE fit has no ``beta``; the hook must carry it."""
    result = cp.StaggeredDifferenceInDifferences(
        noisy_panel(), estimator="etwfe", **BASE_KWARGS
    )
    assert "beta" not in result.model.idata.posterior

    draws = result.__maketables_coef_draws__
    assert isinstance(draws, xr.DataArray)
    assert list(draws.coords["coeffs"].values) == result.labels
    assert result.labels[0] == "att"
    assert "g_u" in result.labels and "g_t" in result.labels

    table = result.__maketables_coef_table__
    assert list(table.index) == result.labels
    assert list(table.columns) == ["b", "se", "p", "t", "ci95l", "ci95u"]
    assert np.isfinite(table["b"]).all()


@pytest.mark.integration
def test_maketables_coef_table_etwfe_pymc_with_covariates(mock_pymc_sample):
    """Covariate rows are appended to the ETWFE parameter block, in order."""
    result = cp.StaggeredDifferenceInDifferences(
        noisy_panel(n_covariates=2),
        estimator="etwfe",
        covariates=["x1", "x2"],
        **BASE_KWARGS,
    )
    table = result.__maketables_coef_table__
    assert list(table.index)[-2:] == ["x1", "x2"]
    assert np.isfinite(table["b"]).all()


@pytest.mark.integration
def test_etwfe_pymc_print_coefficients_runs_with_rich_labels(mock_pymc_sample):
    """``labels`` now carries non-covariate names, which ``beta`` cannot index."""
    result = cp.StaggeredDifferenceInDifferences(
        noisy_panel(n_covariates=1),
        estimator="etwfe",
        covariates=["x1"],
        **BASE_KWARGS,
    )
    result.print_coefficients()


# ===========================================================================
# WP6 -- runtime identification checks
# ===========================================================================


def thin_cell_panel() -> pd.DataFrame:
    """Panel whose cohorts are too small to populate their cells."""
    return generate_staggered_did_data(
        n_units=10,
        n_time_periods=6,
        treatment_cohorts={2: 2, 4: 2},
        sigma=0.2,
        seed=11,
    )


def test_thin_cell_warning_names_the_worst_offenders():
    """The sparse-cell warning has to say *which* cells are thin."""
    with pytest.warns(UserWarning, match="fewer than 5 observations") as record:
        fit_ols(thin_cell_panel())
    message = str(record[0].message)
    assert "(cohort=2, event_time=0): 2 obs" in message
    # the list is capped so the message stays readable
    assert message.count("event_time=") <= 5
    assert "and 1 more." in message


def test_thin_cell_warning_silent_on_a_well_populated_panel():
    """A panel with 8 observations per cell must not trip the threshold."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fit_ols(noise_free_panel())


def test_mostly_empty_surface_warns_once():
    """Sparse corners are normal; a surface that is *mostly* holes is not."""
    df = generate_staggered_did_data(
        n_units=26,
        n_time_periods=10,
        treatment_cohorts={2: 6, 7: 6, 8: 6, 9: 6},
        sigma=0.2,
        seed=3,
    )
    with pytest.warns(UserWarning, match="have no observations") as record:
        fit_ols(df)
    empty_warnings = [w for w in record if "have no observations" in str(w.message)]
    assert len(empty_warnings) == 1
    assert "18 of 32" in str(empty_warnings[0].message)


def test_empty_cells_are_absent_from_the_effect_surface():
    """Cells with no observations produce no row in ``tau_surface_``."""
    df = generate_staggered_did_data(
        n_units=26,
        n_time_periods=10,
        treatment_cohorts={2: 6, 7: 6, 8: 6, 9: 6},
        sigma=0.2,
        seed=3,
    )
    with pytest.warns(UserWarning):
        result = fit_ols(df)
    estimated = set(
        zip(
            result.tau_surface_["cohort"],
            result.tau_surface_["event_time"],
            strict=True,
        )
    )
    for cohort, event_time in result._etwfe_index.dropped_cells:
        assert (cohort, event_time) not in estimated


@pytest.mark.integration
def test_leads_without_never_treated_units_warn(mock_pymc_sample):
    """Leads with no never-treated group is genuine under-identification."""
    df = generate_staggered_did_data(
        n_units=10,
        n_time_periods=8,
        treatment_cohorts={3: 5, 5: 5},
        sigma=0.2,
        seed=1,
    )
    assert (df["treatment_time"] == np.inf).sum() == 0
    with pytest.warns(UserWarning, match="no never-treated units"):
        cp.StaggeredDifferenceInDifferences(
            df, estimator="etwfe", n_leads=2, **BASE_KWARGS
        )


def test_leads_with_never_treated_units_do_not_warn():
    """The under-identification warning is specific to the degenerate case."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_ols(noise_free_panel(), n_leads=3)
    assert not any("never-treated units" in str(w.message) for w in caught)


def test_ols_rank_deficient_design_raises():
    """Both units adopting together makes the time dummy equal the cell dummy."""
    df = pd.DataFrame(
        {
            "unit": [0, 0, 1, 1],
            "time": [0, 1, 0, 1],
            "treatment_time": [1.0, 1.0, 1.0, 1.0],
            "treated": [0, 1, 0, 1],
            "y": [1.0, 2.5, 0.5, 2.0],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(DataException, match="rank deficient"):
            fit_ols(df)


def test_ols_rank_deficiency_message_reports_the_deficiency_count():
    """The error names how many columns are unidentified, not just that some are."""
    df = pd.DataFrame(
        {
            "unit": [0, 0, 1, 1],
            "time": [0, 1, 0, 1],
            "treatment_time": [1.0, 1.0, 1.0, 1.0],
            "treated": [0, 1, 0, 1],
            "y": [1.0, 2.5, 0.5, 2.0],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(DataException) as excinfo:
            fit_ols(df)
    assert "1 deficiency" in str(excinfo.value)


@pytest.mark.integration
@pytest.mark.parametrize("conditioning", ["mundlak", "dummy"])
def test_no_rhat_warning_under_mocked_sampling(mock_pymc_sample, conditioning):
    """Regression test: the mocked sampler yields one chain, so R-hat is NaN.

    An unguarded ``rhat > 1.01`` comparison would warn on every mocked fit in the
    suite. The check has to be guarded by ``np.isfinite``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cp.StaggeredDifferenceInDifferences(
            noisy_panel(),
            estimator="etwfe",
            conditioning=conditioning,
            **BASE_KWARGS,
        )
    assert not any("R-hat" in str(w.message) for w in caught)


def test_convergence_check_is_a_no_op_for_ols(ols_result):
    """The OLS path has no posterior to diagnose."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ols_result._check_etwfe_convergence()
    assert not any("R-hat" in str(w.message) for w in caught)


# ===========================================================================
# WP8 -- plotting
# ===========================================================================


def test_ols_error_bars_use_the_etwfe_standard_error_directly(ols_result):
    """``att_std`` is already a linear-combination SE on the ETWFE path.

    Dividing it again by ``sqrt(n_obs)`` -- which is right for the imputation
    path, where ``att_std`` is a within-group sample standard deviation -- would
    make the ETWFE error bars roughly ``sqrt(n_obs)`` times too narrow.
    """
    table = ols_result.att_event_time_
    np.testing.assert_allclose(
        ols_result._ols_error_bar_se(table), table["att_std"].to_numpy()
    )


def test_ols_error_bars_keep_dividing_by_sqrt_n_for_imputation():
    """The imputation branch is unchanged."""
    result = cp.StaggeredDifferenceInDifferences(
        noisy_panel(), model=SklearnLinearRegression(), **BASE_KWARGS
    )
    table = result.att_event_time_
    np.testing.assert_allclose(
        result._ols_error_bar_se(table),
        table["att_std"].to_numpy() / np.sqrt(table["n_obs"].to_numpy()),
    )


def test_etwfe_event_study_marks_the_reference_event_time(ols_result):
    """Readers should see the omitted baseline, not an unexplained hole."""
    fig, axes = ols_result.plot(show=False)
    labels = [text.get_text() for text in axes[0].get_legend().get_texts()]
    assert "reference (normalised)" in labels
    assert ols_result.reference_event_time in set(axes[0].get_xticks())
    plt.close(fig)


def test_imputation_event_study_has_no_reference_marker():
    """The imputation estimator normalises nothing, so there is no marker."""
    result = cp.StaggeredDifferenceInDifferences(
        noisy_panel(), model=SklearnLinearRegression(), **BASE_KWARGS
    )
    fig, axes = result.plot(show=False)
    labels = [text.get_text() for text in axes[0].get_legend().get_texts()]
    assert "reference (normalised)" not in labels
    plt.close(fig)


def test_plot_tau_surface_ols(ols_result):
    """One panel per cohort that appears in the effect surface."""
    fig, axes = ols_result.plot_tau_surface()
    assert len(axes) == len(ols_result.cohorts)
    for ax, cohort in zip(axes, ols_result.cohorts, strict=True):
        assert _format_cohort(cohort) in ax.get_title()
    plt.close(fig)


def test_plot_tau_surface_with_leads_covers_negative_event_times():
    """Lead cells belong on the surface plot; they are the placebo evidence."""
    result = fit_ols(noise_free_panel(), n_leads=3)
    fig, axes = result.plot_tau_surface()
    assert len(axes) == len(result.cohorts)
    xmin = min(
        np.asarray(line.get_xdata(), dtype=float).min()
        for ax in axes
        for line in ax.get_lines()
        if len(line.get_xdata())
    )
    assert xmin < 0
    plt.close(fig)


@pytest.mark.integration
def test_plot_tau_surface_bayesian(mock_pymc_sample):
    """The Bayesian panel draws a posterior mean line plus an HDI band."""
    result = cp.StaggeredDifferenceInDifferences(
        noisy_panel(), estimator="etwfe", **BASE_KWARGS
    )
    fig, axes = result.plot_tau_surface(hdi_prob=0.8)
    assert len(axes) == len(result.cohorts)
    labels = [text.get_text() for text in axes[0].get_legend().get_texts()]
    assert "80% HDI" in labels
    assert "posterior mean" in labels
    plt.close(fig)


def test_plot_tau_surface_raises_for_imputation():
    """There is no ``tau_surface_`` on the imputation path."""
    result = cp.StaggeredDifferenceInDifferences(
        noisy_panel(), model=SklearnLinearRegression(), **BASE_KWARGS
    )
    with pytest.raises(ValueError, match="only available for estimator='etwfe'"):
        result.plot_tau_surface()
