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
"""Tests for simultaneous-adoption event study."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.data.simulate_data import generate_event_study_data
from causalpy.experiments._results import EventStudyResult

SAMPLE_KWARGS = {
    "tune": 20,
    "draws": 20,
    "chains": 1,
    "cores": 1,
    "progressbar": False,
    "random_seed": 42,
}


def _ols():
    """Return an OLS model with the CausalPy fit contract."""
    return cp.create_causalpy_compatible_class(LinearRegression())


def _panel(**kwargs) -> pd.DataFrame:
    """Return a small simultaneous-adoption panel."""
    defaults = {
        "n_units": 8,
        "n_time": 8,
        "treatment_time": 4,
        "seed": 1,
    }
    defaults.update(kwargs)
    return generate_event_study_data(**defaults)


def _fit(model, **kwargs):
    """Configure and fit an event study on the small panel."""
    experiment = cp.EventStudy(
        _panel(),
        formula="y ~ 1 + C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-2, 2),
        reference_event_time=-1,
        model=model,
        **kwargs,
    )
    return experiment.fit()


def test_generate_event_study_data_shape_and_treatment_time():
    """The simulator is a balanced panel with one treatment time."""
    df = generate_event_study_data(
        n_units=20, n_time=20, treatment_time=10, treated_fraction=0.5, seed=42
    )
    assert df.shape == (400, 5)
    assert list(df.columns) == ["unit", "time", "y", "treat_time", "treated"]
    treated = df.loc[df["treated"] == 1, "treat_time"].unique()
    assert treated.tolist() == [10]
    assert df.loc[df["treated"] == 0, "treat_time"].isna().all()
    assert df.groupby("unit").size().nunique() == 1


def test_configuration_does_not_sample():
    """Building the design does not fit the model."""
    experiment = cp.EventStudy(
        _panel(),
        formula="y ~ 1 + C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        model=_ols(),
    )
    assert experiment.is_fitted is False
    with pytest.raises(cp.GroupNotSampledException):
        _ = experiment.result


def test_sklearn_fit_normalizes_reference_period():
    """The reference event time is exact zero and the window is complete."""
    result = _fit(_ols())
    effects = result.result.event_time_effects
    assert isinstance(result.result, EventStudyResult)
    assert list(effects.coords["event_time"].values) == [-2, -1, 0, 1, 2]
    assert float(effects.sel(event_time=-1).mean()) == 0.0
    table = result.get_plot_data()
    assert table.loc[table["event_time"] == -1, "mean"].item() == 0.0
    summary = result.effect_summary()
    assert "point estimates" in summary.text
    fig, ax = result.plot(show=False)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_pymc_fit_stores_draws_on_the_bundle():
    """PyMC sampling stores event-time draws and a decision summary."""
    result = _fit(cp.pymc_models.LinearRegression(sample_kwargs=dict(SAMPLE_KWARGS)))
    effects = result.result.event_time_effects
    assert isinstance(effects, xr.DataArray)
    assert set(effects.dims) == {"chain", "draw", "event_time"}
    assert effects.sizes["draw"] == SAMPLE_KWARGS["draws"]
    assert float(effects.sel(event_time=-1).mean()) == 0.0
    summary = result.effect_summary()
    assert "hdi_lower" in summary.table.columns
    assert "parallel" in summary.text
    plot_data = result.get_plot_data()
    assert {"mean", "hdi_lower", "hdi_upper", "is_reference"} <= set(plot_data.columns)
    reference = plot_data.loc[plot_data["is_reference"]]
    assert reference["mean"].item() == 0.0
    assert reference[["hdi_lower", "hdi_upper"]].isna().all().all()
    assert summary.table.loc[summary.table["is_reference"], "p_gt_0"].isna().all()


def test_prior_summary_is_labeled_as_a_prior_check():
    """Prior summaries are design checks, not causal estimates."""
    experiment = cp.EventStudy(
        _panel(),
        formula="y ~ 1 + C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        event_window=(-2, 2),
        model=cp.pymc_models.LinearRegression(
            sample_kwargs=dict(SAMPLE_KWARGS),
            prior_sample_kwargs={"draws": 20, "random_seed": 1},
        ),
    )
    experiment.sample_prior_predictive()
    text = experiment.effect_summary(group="prior").text
    assert text.startswith("Prior predictive check")
    fig, _ = experiment.plot(group="prior", show=False)
    plt.close(fig)


def test_effect_summary_before_fit_raises():
    """Summaries do not invent a result before sampling."""
    experiment = cp.EventStudy(
        _panel(),
        formula="y ~ 1 + C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        model=_ols(),
    )
    with pytest.raises(cp.GroupNotSampledException):
        experiment.effect_summary()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda df: df.drop(columns="treat_time"), "Required column"),
        (lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True), "duplicate"),
    ],
)
def test_invalid_panel_raises_before_fit(mutate, match):
    """Broken panels fail at configuration."""
    df = mutate(_panel())
    with pytest.raises(cp.DataException, match=match):
        cp.EventStudy(
            df,
            formula="y ~ 1 + C(unit) + C(time)",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            model=_ols(),
        )


def test_missing_formula_raises():
    """A formula is required."""
    with pytest.raises(cp.FormulaException, match="Formula must be provided"):
        cp.EventStudy(
            _panel(),
            formula="not a formula",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            model=_ols(),
        )


def test_reference_outside_window_raises():
    """The omitted period has to sit inside the estimated window."""
    with pytest.raises(cp.DataException, match="reference_event_time"):
        cp.EventStudy(
            _panel(),
            formula="y ~ 1 + C(unit) + C(time)",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            event_window=(-2, 2),
            reference_event_time=9,
            model=_ols(),
        )


def test_inverted_event_window_raises():
    """The window must run from an earlier event time to a later one."""
    with pytest.raises(cp.DataException, match="event_window"):
        cp.EventStudy(
            _panel(),
            formula="y ~ 1 + C(unit) + C(time)",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            event_window=(2, -2),
            model=_ols(),
        )


def test_no_treated_units_raises():
    """A panel of only never-treated units is not an event study."""
    df = _panel()
    df["treat_time"] = np.nan
    with pytest.raises(cp.DataException, match="No treated units"):
        cp.EventStudy(
            df,
            formula="y ~ 1 + C(unit) + C(time)",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            model=_ols(),
        )


def test_event_time_column_collision_raises():
    """The experiment owns the event-time dummy names."""
    df = _panel()
    df["event_time_0"] = 0.0
    with pytest.raises(cp.DataException, match="collide"):
        cp.EventStudy(
            df,
            formula="y ~ 1 + event_time_0",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            model=_ols(),
        )


def test_staggered_adoption_is_rejected():
    """Different treatment times are a different estimator."""
    df = _panel()
    treated = df.loc[df["treated"] == 1, "unit"].unique()
    df.loc[df["unit"] == treated[0], "treat_time"] = 3
    with pytest.raises(cp.DataException, match="StaggeredDifferenceInDifferences"):
        cp.EventStudy(
            df,
            formula="y ~ 1 + C(unit) + C(time)",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            model=_ols(),
        )


def test_unbalanced_panel_warns_but_configures():
    """An unbalanced panel is a warning, not a silent drop of the design."""
    df = _panel().drop(index=0)
    with pytest.warns(UserWarning, match="unbalanced panel"):
        experiment = cp.EventStudy(
            df,
            formula="y ~ 1 + C(unit) + C(time)",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            model=_ols(),
        )
    assert experiment.is_fitted is False


def test_time_gaps_warn():
    """A hole in the time grid warns because event time uses subtraction."""
    df = _panel()
    df = df.loc[df["time"] != 3]
    with pytest.warns(UserWarning, match="gaps in time series"):
        cp.EventStudy(
            df,
            formula="y ~ 1 + C(unit) + C(time)",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            model=_ols(),
        )


def test_extreme_missingness_warns():
    """Dropping most rows to missing outcomes warns."""
    df = _panel(n_units=10, n_time=10)
    df.loc[df.index[:60], "y"] = np.nan
    with pytest.warns(UserWarning, match="removed"):
        cp.EventStudy(
            df,
            formula="y ~ 1 + C(unit) + C(time)",
            unit_col="unit",
            time_col="time",
            treat_time_col="treat_time",
            model=_ols(),
        )


def test_polars_input_is_copied():
    """A Polars frame is accepted and not mutated."""
    polars = pytest.importorskip("polars")
    df = _panel()
    frame = polars.from_pandas(df)
    experiment = cp.EventStudy(
        frame,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_col="unit",
        time_col="time",
        treat_time_col="treat_time",
        model=_ols(),
    )
    assert "_event_time" not in frame.columns
    assert experiment.data["_event_time"].notna().any()


def test_summary_prints_the_reference_period(capsys):
    """The text summary names the omitted event time."""
    result = _fit(_ols())
    result.summary(round_to=2)
    captured = capsys.readouterr().out
    assert "Reference event time: -1" in captured
    assert "event_time" in captured
