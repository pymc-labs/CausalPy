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
Tests that column-based experiments accept non-pandas dataframes.

Each experiment is built twice, once from the pandas fixture and once from the
same data as a Polars dataframe. The two runs must agree. Experiments that
treat the pandas index as a time axis are not covered here; they need an
explicit time column first.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.data.simulate_data import (
    generate_ancova_data,
    generate_piecewise_its_data,
    generate_staggered_did_data,
)

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}


def to_polars(data: pd.DataFrame) -> pl.DataFrame:
    """Convert a pandas fixture to Polars, dropping the index."""
    return pl.from_pandas(data.reset_index(drop=True))


def assert_data_matches(result_pandas, result_polars, attribute="data"):
    """The stored dataframe must agree between the two backends.

    The index is dropped before comparing because the two backends are not
    expected to agree on it. A Polars input has no index to carry over, so it
    gets a positional one. That contract is asserted separately, per
    experiment, by ``assert_positional_obs_ind``.
    """
    left = getattr(result_pandas, attribute).reset_index(drop=True)
    right = getattr(result_polars, attribute).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def assert_positional_obs_ind(result_polars, attribute="data"):
    """A converted input carries a positional observation coordinate.

    Polars and PyArrow have no index to carry over, so conversion assigns
    ``0..n-1``. Asserted for every experiment rather than one representative,
    since the coordinate reaches user-facing results.
    """
    index = getattr(result_polars, attribute).index
    assert index.tolist() == list(range(len(index)))


def assert_model_input_matches(result_pandas, result_polars):
    """The model must see identical inputs from both backends.

    Sampling is mocked suite-wide, so posterior draws from two separate
    constructions are not comparable. The design matrices are a deterministic
    function of the data and the formula, which is what a conversion bug would
    actually corrupt.
    """
    assert result_polars.labels == result_pandas.labels
    if hasattr(result_pandas, "design"):
        pairs = [
            (result_pandas.design[key], result_polars.design[key]) for key in ("X", "y")
        ]
    else:
        pairs = [
            (result_pandas.X, result_polars.X),
            (result_pandas.y, result_polars.y),
        ]
    for from_pandas, from_polars in pairs:
        np.testing.assert_allclose(
            np.asarray(from_polars, dtype=float), np.asarray(from_pandas, dtype=float)
        )


def test_did_accepts_polars(did_data):
    """DifferenceInDifferences gives the same fit from pandas and Polars."""

    def build(data):
        return cp.DifferenceInDifferences(
            data,
            formula="y ~ 1 + group*post_treatment",
            time_variable_name="t",
            group_variable_name="group",
            model=LinearRegression(),
        )

    from_pandas = build(did_data)
    from_polars = build(to_polars(did_data))

    assert_data_matches(from_pandas, from_polars)
    assert_positional_obs_ind(from_polars)
    assert from_polars.causal_impact == pytest.approx(from_pandas.causal_impact)


def test_regression_discontinuity_accepts_polars(rd_data):
    """RegressionDiscontinuity gives the same fit from pandas and Polars."""

    def build(data):
        return cp.RegressionDiscontinuity(
            data,
            formula="y ~ 1 + x + treated",
            model=LinearRegression(),
            treatment_threshold=0.5,
        )

    from_pandas = build(rd_data)
    from_polars = build(to_polars(rd_data))

    assert_data_matches(from_pandas, from_polars)
    assert_positional_obs_ind(from_polars)
    assert np.asarray(from_polars.discontinuity_at_threshold).item() == pytest.approx(
        np.asarray(from_pandas.discontinuity_at_threshold).item()
    )


def test_regression_kink_accepts_polars(mock_pymc_sample):
    """RegressionKink accepts a Polars dataframe."""
    from causalpy.tests.conftest import setup_regression_kink_data

    kink = 0.5
    data = setup_regression_kink_data(kink)

    def build(frame):
        return cp.RegressionKink(
            frame,
            formula=f"y ~ 1 + x + I((x-{kink})*treated)",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
            kink_point=kink,
        )

    from_pandas = build(data)
    from_polars = build(to_polars(data))

    assert_data_matches(from_pandas, from_polars)
    assert_positional_obs_ind(from_polars)
    assert_model_input_matches(from_pandas, from_polars)


def test_prepostnegd_accepts_polars(mock_pymc_sample):
    """PrePostNEGD accepts a Polars dataframe."""
    data = generate_ancova_data(seed=42)

    def build(frame):
        return cp.PrePostNEGD(
            frame,
            formula="post ~ 1 + C(group) + pre",
            group_variable_name="group",
            pretreatment_variable_name="pre",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )

    from_pandas = build(data)
    from_polars = build(to_polars(data))

    assert_data_matches(from_pandas, from_polars)
    assert_positional_obs_ind(from_polars)
    assert_model_input_matches(from_pandas, from_polars)


def test_inverse_propensity_weighting_accepts_polars(mock_pymc_sample):
    """InversePropensityWeighting accepts a Polars dataframe."""
    data = cp.load_data("nhefs")

    def build(frame):
        return cp.InversePropensityWeighting(
            data=frame,
            formula="trt ~ 1 + age + race",
            outcome_variable="outcome",
            weighting_scheme="robust",
            model=cp.pymc_models.PropensityScore(sample_kwargs=sample_kwargs),
        )

    from_pandas = build(data)
    from_polars = build(to_polars(data))

    assert_data_matches(from_pandas, from_polars)
    assert_positional_obs_ind(from_polars)
    assert_model_input_matches(from_pandas, from_polars)


def test_instrumental_variable_accepts_polars(mock_pymc_sample):
    """InstrumentalVariable converts both of its dataframe arguments."""
    df = cp.load_data("risk")
    instruments_data = df[["risk", "logmort0"]]
    data = df[["loggdp", "risk"]]

    def build(instruments, covariates):
        return cp.InstrumentalVariable(
            instruments_data=instruments,
            data=covariates,
            instruments_formula="risk ~ 1 + logmort0",
            formula="loggdp ~ 1 + risk",
            model=cp.pymc_models.InstrumentalVariableRegression(
                sample_kwargs=sample_kwargs
            ),
        )

    from_pandas = build(instruments_data, data)
    from_polars = build(to_polars(instruments_data), to_polars(data))

    assert_data_matches(from_pandas, from_polars)
    assert_positional_obs_ind(from_polars)
    assert_data_matches(from_pandas, from_polars, attribute="instruments_data")
    assert_model_input_matches(from_pandas, from_polars)


def test_panel_regression_accepts_polars(mock_pymc_sample):
    """PanelRegression accepts a Polars dataframe."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "unit": np.repeat([f"unit_{i}" for i in range(5)], 10),
            "time": np.tile(range(10), 5),
            "treatment": np.tile([0] * 5 + [1] * 5, 5),
            "x1": rng.normal(size=50),
            "y": rng.normal(size=50),
        }
    )

    def build(frame):
        return cp.PanelRegression(
            data=frame,
            formula="y ~ C(unit) + C(time) + treatment + x1",
            unit_fe_variable="unit",
            time_fe_variable="time",
            fe_method="dummies",
            model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        )

    from_pandas = build(data)
    from_polars = build(to_polars(data))

    assert_data_matches(from_pandas, from_polars)
    assert_positional_obs_ind(from_polars)
    assert_model_input_matches(from_pandas, from_polars)


def test_staggered_did_accepts_polars(mock_pymc_sample):
    """StaggeredDifferenceInDifferences accepts a Polars dataframe."""
    data = generate_staggered_did_data(n_units=30, n_time_periods=15, seed=42)

    def build(frame):
        return cp.StaggeredDifferenceInDifferences(
            frame,
            formula="y ~ 1 + C(unit) + C(time)",
            unit_variable_name="unit",
            time_variable_name="time",
            treated_variable_name="treated",
            treatment_time_variable_name="treatment_time",
            model=LinearRegression(),
        )

    from_pandas = build(data)
    from_polars = build(to_polars(data))

    assert_data_matches(from_pandas, from_polars)
    assert_positional_obs_ind(from_polars)
    pd.testing.assert_frame_equal(
        from_pandas.att_event_time_.reset_index(drop=True),
        from_polars.att_event_time_.reset_index(drop=True),
        check_dtype=False,
    )


def test_piecewise_its_accepts_polars():
    """PiecewiseITS reads time from the step() column, so Polars works."""
    data, _ = generate_piecewise_its_data(N=100, seed=42)

    def build(frame):
        return cp.PiecewiseITS(
            frame,
            formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
            model=LinearRegression(),
        )

    from_pandas = build(data)
    from_polars = build(to_polars(data))

    assert_data_matches(from_pandas, from_polars)
    assert_positional_obs_ind(from_polars)
    assert from_polars.time_col == from_pandas.time_col


def test_polars_input_leaves_caller_frame_alone(did_data):
    """Constructing an experiment does not mutate the caller's dataframe."""
    data = to_polars(did_data)
    before = data.clone()
    cp.DifferenceInDifferences(
        data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )
    assert data.equals(before)


def test_pandas_input_index_is_not_renamed(did_data):
    """Constructing an experiment no longer renames the caller's index."""
    data = did_data.copy()
    data.index.name = "my_index"
    cp.DifferenceInDifferences(
        data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )
    assert data.index.name == "my_index"


def test_panel_regression_pandas_input_index_is_not_renamed(mock_pymc_sample):
    """PanelRegression also leaves the caller's index name alone."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "unit": np.repeat([f"unit_{i}" for i in range(5)], 10),
            "time": np.tile(range(10), 5),
            "treatment": np.tile([0] * 5 + [1] * 5, 5),
            "x1": rng.normal(size=50),
            "y": rng.normal(size=50),
        }
    )
    data.index.name = "my_index"
    cp.PanelRegression(
        data=data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    assert data.index.name == "my_index"


def test_polars_input_gets_positional_obs_ind(did_data):
    """A dataframe without an index gets a positional observation coordinate.

    Pandas callers keep whatever their index holds. Polars and PyArrow have no
    index to carry over, so conversion assigns ``0..n-1``. That coordinate is
    row identity for these column-based classes, not a semantic axis, and no
    existing caller can regress: a frame with no index never had labels to
    lose. Pinned here so the contract is explicit rather than incidental.
    """

    def build(data):
        return cp.DifferenceInDifferences(
            data,
            formula="y ~ 1 + group*post_treatment",
            time_variable_name="t",
            group_variable_name="group",
            model=LinearRegression(),
        )

    labelled = did_data.copy()
    labelled.index = pd.Index([f"row_{i}" for i in range(len(labelled))], name="key")

    from_pandas = build(labelled)
    from_polars = build(to_polars(did_data))

    assert from_pandas.data.index.tolist() == labelled.index.tolist()
    assert from_polars.data.index.tolist() == list(range(len(did_data)))
