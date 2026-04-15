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
"""Tests for optional maketables plugin hooks on experiment objects."""

import arviz as az
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.data.simulate_data import (
    generate_piecewise_its_data,
    generate_staggered_did_data,
)
from causalpy.maketables_adapters import (
    PyMCMaketablesAdapter,
    SklearnMaketablesAdapter,
    _canonical_frame,
    _extract_hdi_bounds,
    _get_maketables_hdi_prob,
    _safe_observation_count,
    _safe_r2_value,
    get_maketables_adapter,
)

sample_kwargs = {
    "chains": 2,
    "draws": 100,
    "progressbar": False,
    "random_seed": 42,
}


@pytest.mark.integration
def test_maketables_coef_table_pymc_contract(mock_pymc_sample):
    """PyMC-backed experiments should expose canonical maketables columns."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    table = result.__maketables_coef_table__

    assert isinstance(table, pd.DataFrame)
    assert table.index.name == "Coefficient"
    for col in ["b", "se", "p", "t", "ci95l", "ci95u"]:
        assert col in table.columns
    assert list(table.index) == list(result.labels)
    assert table["b"].notna().all()
    assert table["se"].notna().all()
    assert table["p"].isna().all()
    assert table["ci95l"].notna().all()
    assert table["ci95u"].notna().all()


@pytest.mark.integration
def test_maketables_coef_table_sklearn_contract(mock_pymc_sample):
    """Sklearn-backed experiments should expose canonical columns with NaN inference."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )

    table = result.__maketables_coef_table__

    assert isinstance(table, pd.DataFrame)
    assert table.index.name == "Coefficient"
    for col in ["b", "se", "p", "t", "ci95l", "ci95u"]:
        assert col in table.columns
    assert list(table.index) == list(result.labels)
    assert table["b"].notna().all()
    assert table["se"].isna().all()
    assert table["p"].isna().all()
    assert table["t"].isna().all()


@pytest.mark.integration
def test_maketables_multi_treated_unit_requires_explicit_selection(mock_pymc_sample):
    """PyMC coefficient export raises for ambiguous multi-treated-unit outputs."""
    rng = np.random.default_rng(42)
    n_obs = 50
    n_control = 3
    n_treated = 2
    time_index = pd.date_range("2020-01-01", periods=n_obs, freq="D")
    treatment_time = time_index[30]

    control_data = {
        f"control_{i}": np.cumsum(rng.normal(0.0, 0.2, n_obs)) + 10.0 + i
        for i in range(n_control)
    }
    treated_data = {}
    for j in range(n_treated):
        base = sum(
            (i + 1) / (n_control * (n_control + 1) / 2) * control_data[f"control_{i}"]
            for i in range(n_control)
        )
        treatment_effect = np.zeros(n_obs)
        treatment_effect[30:] = 2.0 + 0.1 * j
        treated_data[f"treated_{j}"] = (
            base + treatment_effect + rng.normal(0.0, 0.1, n_obs)
        )

    df = pd.DataFrame({**control_data, **treated_data}, index=time_index)

    result = cp.SyntheticControl(
        df,
        treatment_time,
        control_units=[f"control_{i}" for i in range(n_control)],
        treated_units=[f"treated_{j}" for j in range(n_treated)],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )

    with pytest.raises(ValueError, match="Ambiguous multi-treated-unit"):
        _ = result.__maketables_coef_table__


@pytest.mark.integration
def test_maketables_stat_and_metadata_hooks(mock_pymc_sample):
    """Stats, labels, defaults, depvar, and vcov hooks return stable metadata."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=LinearRegression(),
    )

    assert result.__maketables_stat__("N") is not None
    assert result.__maketables_stat__("unknown_key") is None
    assert result.__maketables_depvar__ == result.outcome_variable_name
    assert result.__maketables_vcov_info__["se_type"] == "Not available"
    assert result.__maketables_stat_labels__["N"] == "N"
    assert "N" in (result.__maketables_default_stat_keys__ or [])


def test_maketables_depvar_fallback_for_ipw():
    """Experiments with outcome_variable should expose it via depvar hook."""
    df = cp.load_data("nhefs")
    result = cp.InversePropensityWeighting(
        data=df,
        formula="trt ~ 1 + age + race",
        outcome_variable="outcome",
        weighting_scheme="robust",
    )

    assert result.__maketables_depvar__ == "outcome"


@pytest.mark.integration
def test_maketables_hdi_prob_user_control(mock_pymc_sample):
    """Users can control HDI level for maketables coefficient intervals."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )
    result.set_maketables_options(hdi_prob=0.8)

    table = result.__maketables_coef_table__
    first_label = result.labels[0]
    draws = (
        result.model.idata.posterior["beta"]
        .isel(treated_units=0)
        .sel(coeffs=first_label)
    )
    hdi = az.hdi(draws, hdi_prob=0.8)
    expected_lower, expected_upper = _extract_hdi_bounds(hdi)

    assert table.loc[first_label, "ci95l"] == pytest.approx(expected_lower)
    assert table.loc[first_label, "ci95u"] == pytest.approx(expected_upper)


def test_maketables_hdi_prob_validation(mock_pymc_sample):
    """Invalid HDI probability should raise an explicit ValueError."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    with pytest.raises(ValueError, match="hdi_prob must be in \\(0, 1\\)"):
        result.set_maketables_options(hdi_prob=1.2)


@pytest.mark.integration
def test_maketables_prepostnegd_pymc_contract(mock_pymc_sample):
    """PrePostNEGD should expose canonical maketables coefficient table columns."""
    df = cp.load_data("anova1")
    result = cp.PrePostNEGD(
        df,
        formula="post ~ 1 + C(group) + pre",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    table = result.__maketables_coef_table__
    assert table.index.name == "Coefficient"
    for col in ["b", "se", "p", "ci95l", "ci95u"]:
        assert col in table.columns
    assert table["ci95l"].notna().all()
    assert table["ci95u"].notna().all()


@pytest.mark.integration
def test_maketables_regression_kink_pymc_contract(mock_pymc_sample):
    """RegressionKink should expose canonical maketables coefficient table columns."""
    rng = np.random.default_rng(42)
    kink_point = 0.5
    beta = [1, 0.5, 0, 0.5, 0]
    n_obs = 120
    x = rng.uniform(-1, 1, n_obs)
    treated = (x >= kink_point).astype(int)
    y = (
        beta[0]
        + beta[1] * x
        + beta[2] * x**2
        + beta[3] * (x - kink_point) * treated
        + beta[4] * (x - kink_point) ** 2 * treated
        + rng.normal(0, 0.1, n_obs)
    )
    df = pd.DataFrame({"x": x, "y": y, "treated": treated})

    result = cp.RegressionKink(
        df,
        formula="y ~ 1 + x + I(x**2) + I((x-0.5)*treated) + I(((x-0.5)**2)*treated)",
        kink_point=kink_point,
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    table = result.__maketables_coef_table__
    assert table.index.name == "Coefficient"
    assert "ci95l" in table.columns
    assert "ci95u" in table.columns


@pytest.mark.integration
def test_maketables_piecewise_its_pymc_contract(mock_pymc_sample):
    """PiecewiseITS should expose maketables coefficient table for PyMC backend."""
    df, _ = generate_piecewise_its_data(N=120, seed=42)
    result = cp.PiecewiseITS(
        df,
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    table = result.__maketables_coef_table__
    assert table.index.name == "Coefficient"
    assert {"b", "se", "p", "ci95l", "ci95u"}.issubset(set(table.columns))


@pytest.mark.integration
def test_maketables_staggered_did_pymc_contract(mock_pymc_sample):
    """Staggered DiD should expose maketables coefficient table for PyMC backend."""
    df = generate_staggered_did_data(
        n_units=30,
        n_time_periods=15,
        treatment_cohorts={5: 10, 10: 10},
        seed=42,
    )
    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        treatment_time_variable_name="treatment_time",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    table = result.__maketables_coef_table__
    assert table.index.name == "Coefficient"
    assert "ci95l" in table.columns
    assert "ci95u" in table.columns


@pytest.mark.integration
def test_maketables_interrupted_time_series_pymc_contract(mock_pymc_sample):
    """InterruptedTimeSeries should expose maketables coefficient table for PyMC backend."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=pd.to_datetime("2017-01-01"),
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    table = result.__maketables_coef_table__
    assert table.index.name == "Coefficient"
    assert {"b", "se", "p", "ci95l", "ci95u"}.issubset(set(table.columns))
    assert list(table.index) == list(result.labels)


@pytest.mark.integration
def test_maketables_regression_discontinuity_pymc_contract(mock_pymc_sample):
    """RegressionDiscontinuity should expose maketables coefficient table for PyMC backend."""
    df = cp.load_data("rd")
    df["treated"] = df["treated"].astype(int)
    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + x + treated + x:treated",
        treatment_threshold=0.5,
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    table = result.__maketables_coef_table__
    assert table.index.name == "Coefficient"
    assert "ci95l" in table.columns
    assert "ci95u" in table.columns
    assert list(table.index) == list(result.labels)


@pytest.mark.integration
def test_maketables_synthetic_control_single_treated_pymc_contract(mock_pymc_sample):
    """SyntheticControl should export coefficients for single-treated-unit PyMC models."""
    df = cp.load_data("sc")
    result = cp.SyntheticControl(
        df,
        treatment_time=70,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )

    table = result.__maketables_coef_table__
    assert table.index.name == "Coefficient"
    assert {"b", "se", "p", "ci95l", "ci95u"}.issubset(set(table.columns))
    assert list(table.index) == list(result.labels)


@pytest.mark.integration
def test_maketables_inverse_propensity_weighting_pymc_contract(mock_pymc_sample):
    """IPW should export maketables coefficients from PropensityScore posterior draws."""
    df = cp.load_data("nhefs")
    result = cp.InversePropensityWeighting(
        data=df,
        formula="trt ~ 1 + age + race",
        outcome_variable="outcome",
        weighting_scheme="robust",
        model=cp.pymc_models.PropensityScore(sample_kwargs=sample_kwargs),
    )

    table = result.__maketables_coef_table__
    assert table.index.name == "Coefficient"
    assert {"b", "se", "p", "ci95l", "ci95u"}.issubset(set(table.columns))
    assert list(table.index) == list(result.labels)


@pytest.mark.integration
def test_maketables_instrumental_variable_pymc_contract(mock_pymc_sample):
    """InstrumentalVariable should export maketables coefficients for outcome equation."""
    df = cp.load_data("risk")
    result = cp.InstrumentalVariable(
        instruments_data=df[["risk", "logmort0"]],
        data=df[["loggdp", "risk"]],
        instruments_formula="risk ~ 1 + logmort0",
        formula="loggdp ~ 1 + risk",
        model=cp.pymc_models.InstrumentalVariableRegression(
            sample_kwargs=sample_kwargs
        ),
    )

    table = result.__maketables_coef_table__
    assert table.index.name == "Coefficient"
    assert {"b", "se", "p", "ci95l", "ci95u"}.issubset(set(table.columns))
    assert list(table.index) == list(result.labels)


@pytest.mark.integration
def test_maketables_interrupted_time_series_sklearn_contract(mock_pymc_sample):
    """InterruptedTimeSeries should expose canonical maketables columns for sklearn."""
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    result = cp.InterruptedTimeSeries(
        df,
        treatment_time=pd.to_datetime("2017-01-01"),
        formula="y ~ 1 + t + C(month)",
        model=LinearRegression(),
    )

    table = result.__maketables_coef_table__
    assert table.index.name == "Coefficient"
    assert list(table.index) == list(result.labels)
    for col in ["b", "se", "p", "t", "ci95l", "ci95u"]:
        assert col in table.columns
    assert table["b"].notna().all()
    assert table["se"].isna().all()
    assert table["p"].isna().all()
    assert table["t"].isna().all()


@pytest.mark.integration
def test_maketables_regression_discontinuity_sklearn_contract(mock_pymc_sample):
    """RegressionDiscontinuity should expose canonical maketables columns for sklearn."""
    df = cp.load_data("rd")
    df["treated"] = df["treated"].astype(int)
    result = cp.RegressionDiscontinuity(
        df,
        formula="y ~ 1 + x + treated + x:treated",
        treatment_threshold=0.5,
        model=LinearRegression(),
    )

    table = result.__maketables_coef_table__
    assert table.index.name == "Coefficient"
    assert list(table.index) == list(result.labels)
    assert table["b"].notna().all()
    assert table["se"].isna().all()


@pytest.mark.integration
def test_maketables_piecewise_its_sklearn_contract(mock_pymc_sample):
    """PiecewiseITS should expose canonical maketables columns for sklearn."""
    df, _ = generate_piecewise_its_data(N=120, seed=42)
    result = cp.PiecewiseITS(
        df,
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
        model=LinearRegression(),
    )

    table = result.__maketables_coef_table__
    assert table.index.name == "Coefficient"
    assert list(table.index) == list(result.labels)
    assert table["b"].notna().all()
    assert table["se"].isna().all()


@pytest.mark.integration
def test_maketables_synthetic_control_single_treated_sklearn_contract(mock_pymc_sample):
    """SyntheticControl should expose canonical maketables columns for sklearn."""
    df = cp.load_data("sc")
    result = cp.SyntheticControl(
        df,
        treatment_time=70,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=LinearRegression(),
    )

    table = result.__maketables_coef_table__
    assert table.index.name == "Coefficient"
    assert list(table.index) == list(result.labels)
    assert table["b"].notna().all()
    assert table["se"].isna().all()


@pytest.mark.integration
def test_maketables_staggered_did_sklearn_contract(mock_pymc_sample):
    """StaggeredDiD should expose canonical maketables columns for sklearn."""
    df = generate_staggered_did_data(
        n_units=30,
        n_time_periods=15,
        treatment_cohorts={5: 10, 10: 10},
        seed=42,
    )
    result = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        treatment_time_variable_name="treatment_time",
        model=LinearRegression(),
    )

    table = result.__maketables_coef_table__
    assert table.index.name == "Coefficient"
    assert list(table.index) == list(result.labels)
    assert table["b"].notna().all()
    assert table["se"].isna().all()


@pytest.mark.integration
def test_maketables_missing_pymc_coef_variable_raises(mock_pymc_sample):
    """PyMC export should fail clearly when posterior has no supported coef variable."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    posterior = result.model.idata.posterior.rename({"beta": "beta_missing"})
    result.model.idata = az.InferenceData(posterior=posterior)

    with pytest.raises(
        ValueError,
        match="must expose one of 'beta', 'b', or 'beta_z'",
    ):
        _ = result.__maketables_coef_table__


@pytest.mark.integration
def test_maketables_incompatible_pymc_label_dim_raises(mock_pymc_sample):
    """PyMC export should fail clearly when coefficient label dimension is unsupported."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    posterior = result.model.idata.posterior.rename({"coeffs": "bad_coeff_dim"})
    result.model.idata = az.InferenceData(posterior=posterior)

    with pytest.raises(
        ValueError,
        match="do not include a label dimension compatible with experiment labels",
    ):
        _ = result.__maketables_coef_table__


# ---------------------------------------------------------------------------
# Unit tests for internal adapter helpers
# ---------------------------------------------------------------------------


class _Stub:
    """Lightweight test stub accepting arbitrary keyword attributes."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestSafeObservationCount:
    def test_returns_count_from_data_attr(self):
        stub = _Stub(data=np.zeros((42, 3)))
        assert _safe_observation_count(stub) == 42

    def test_returns_count_from_X_attr(self):
        stub = _Stub(X=np.zeros((10, 2)))
        assert _safe_observation_count(stub) == 10

    def test_returns_none_when_no_attrs(self):
        stub = _Stub()
        assert _safe_observation_count(stub) is None

    def test_skips_none_attrs(self):
        stub = _Stub(data=None, X=None, datapre=np.zeros((5, 1)))
        assert _safe_observation_count(stub) == 5

    def test_skips_attr_without_shape(self):
        stub = _Stub(data="not an array", datapre=np.zeros((7,)))
        assert _safe_observation_count(stub) == 7

    def test_skips_empty_shape(self):
        stub = _Stub(data=np.float64(3.0), datapost=np.zeros((8,)))
        assert _safe_observation_count(stub) == 8


class TestSafeR2Value:
    def test_returns_none_when_no_score(self):
        assert _safe_r2_value(_Stub()) is None

    def test_returns_none_when_score_is_none(self):
        assert _safe_r2_value(_Stub(score=None)) is None

    def test_returns_float_for_numeric_score(self):
        assert _safe_r2_value(_Stub(score=0.85)) == pytest.approx(0.85)

    def test_returns_float_for_numpy_int(self):
        assert _safe_r2_value(_Stub(score=np.int64(1))) == pytest.approx(1.0)

    def test_series_with_r2_key(self):
        s = pd.Series({"r2": 0.9, "r2_std": 0.05})
        result = _safe_r2_value(_Stub(score=s))
        assert result == pytest.approx(0.9)

    def test_series_with_no_r2_keys(self):
        s = pd.Series({"rmse": 1.5, "mae": 0.8})
        assert _safe_r2_value(_Stub(score=s)) is None

    def test_series_with_only_nan_r2(self):
        s = pd.Series({"r2": np.nan})
        assert _safe_r2_value(_Stub(score=s)) is None

    def test_returns_none_for_string_score(self):
        assert _safe_r2_value(_Stub(score="high")) is None

    def test_returns_none_for_dict_score(self):
        assert _safe_r2_value(_Stub(score={"r2": 0.9})) is None

    def test_exception_in_series_ops_returns_none(self):
        class _ExplodingSeries(pd.Series):
            def __getitem__(self, key):
                raise RuntimeError("kaboom")

        s = _ExplodingSeries({"r2_mean": 0.9})
        assert _safe_r2_value(_Stub(score=s)) is None


class TestCanonicalFrame:
    def test_fills_nan_when_ci_not_provided(self):
        frame = _canonical_frame(
            labels=["a", "b"],
            b=np.array([1.0, 2.0]),
            se=np.array([0.1, 0.2]),
            p=np.array([0.01, 0.02]),
        )
        assert frame["ci95l"].isna().all()
        assert frame["ci95u"].isna().all()
        assert frame.index.name == "Coefficient"


class TestExtractHdiBoundsDataArray:
    def test_dataarray_path(self):
        import xarray as xr

        da = xr.DataArray([1.0, 3.0], dims=["hdi"], coords={"hdi": ["lower", "higher"]})
        lo, hi = _extract_hdi_bounds(da)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(3.0)

    def test_dataset_with_explicit_var(self):
        import xarray as xr

        da = xr.DataArray([1.5, 2.5], dims=["hdi"], coords={"hdi": ["lower", "higher"]})
        ds = xr.Dataset({"my_var": da, "other_var": da * 2})
        lo, hi = _extract_hdi_bounds(ds, var_name="my_var")
        assert lo == pytest.approx(1.5)
        assert hi == pytest.approx(2.5)

    def test_dataset_without_var_name_uses_first(self):
        import xarray as xr

        da = xr.DataArray([2.0, 4.0], dims=["hdi"], coords={"hdi": ["lower", "higher"]})
        ds = xr.Dataset({"only_var": da})
        lo, hi = _extract_hdi_bounds(ds)
        assert lo == pytest.approx(2.0)
        assert hi == pytest.approx(4.0)

    def test_dataset_with_missing_var_name_uses_first(self):
        import xarray as xr

        da = xr.DataArray([0.5, 1.5], dims=["hdi"], coords={"hdi": ["lower", "higher"]})
        ds = xr.Dataset({"actual": da})
        lo, hi = _extract_hdi_bounds(ds, var_name="nonexistent")
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(1.5)


class TestGetMaketablesHdiProb:
    def test_explicit_override(self):
        stub = _Stub(_maketables_hdi_prob=0.89)
        assert _get_maketables_hdi_prob(stub) == pytest.approx(0.89)

    def test_fallback_to_hdi_prob_attr(self):
        stub = _Stub(hdi_prob_=0.95)
        assert _get_maketables_hdi_prob(stub) == pytest.approx(0.95)

    def test_default_094(self):
        stub = _Stub()
        assert _get_maketables_hdi_prob(stub) == pytest.approx(0.94)

    def test_none_hdi_prob_attr_falls_to_default(self):
        stub = _Stub(hdi_prob_=None)
        assert _get_maketables_hdi_prob(stub) == pytest.approx(0.94)

    def test_non_convertible_raises(self):
        stub = _Stub(_maketables_hdi_prob="not_a_number")
        with pytest.raises(ValueError, match="Invalid HDI probability"):
            _get_maketables_hdi_prob(stub)

    def test_out_of_range_raises(self):
        stub = _Stub(_maketables_hdi_prob=1.5)
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            _get_maketables_hdi_prob(stub)

    def test_zero_raises(self):
        stub = _Stub(_maketables_hdi_prob=0.0)
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            _get_maketables_hdi_prob(stub)


class TestGetMaketablesAdapter:
    def test_unsupported_model_raises(self):
        with pytest.raises(TypeError, match="Unsupported model backend"):
            get_maketables_adapter("not_a_model")


class TestSklearnAdapterUnit:
    def test_coef_table_success(self):
        adapter = SklearnMaketablesAdapter()
        model = _Stub(get_coeffs=lambda: [1.0, 2.0])
        stub = _Stub(labels=["a", "b"], model=model)
        frame = adapter.coef_table(stub)
        assert list(frame.index) == ["a", "b"]
        assert frame["b"].notna().all()
        assert frame["se"].isna().all()

    def test_stat_returns_expected_keys(self):
        adapter = SklearnMaketablesAdapter()
        stub = _Stub(data=np.zeros((20, 3)), score=0.75)
        assert adapter.stat(stub, "N") == 20
        assert adapter.stat(stub, "r2") == pytest.approx(0.75)
        assert adapter.stat(stub, "model_type") == "ols"
        assert adapter.stat(stub, "unknown") is None

    def test_vcov_info(self):
        adapter = SklearnMaketablesAdapter()
        info = adapter.vcov_info(_Stub())
        assert info["se_type"] == "Not available"

    def test_stat_labels(self):
        adapter = SklearnMaketablesAdapter()
        labels = adapter.stat_labels(_Stub())
        assert labels is not None
        assert "N" in labels

    def test_default_stat_keys_includes_r2_when_available(self):
        adapter = SklearnMaketablesAdapter()
        stub = _Stub(score=0.9)
        keys = adapter.default_stat_keys(stub)
        assert keys is not None
        assert "r2" in keys

    def test_default_stat_keys_excludes_r2_when_none(self):
        adapter = SklearnMaketablesAdapter()
        stub = _Stub()
        keys = adapter.default_stat_keys(stub)
        assert keys is not None
        assert "r2" not in keys


class TestSklearnAdapterCoefMismatch:
    def test_coef_count_mismatch_raises(self):
        adapter = SklearnMaketablesAdapter()
        model = _Stub(get_coeffs=lambda: [1.0, 2.0])
        stub = _Stub(labels=["a", "b", "c"], model=model)
        with pytest.raises(ValueError, match="Coefficient count mismatch"):
            adapter.coef_table(stub)


class TestSklearnAdapterNoLabelsRaises:
    def test_empty_labels_raises(self):
        adapter = SklearnMaketablesAdapter()
        stub = _Stub(labels=[])
        with pytest.raises(ValueError, match="no coefficient labels"):
            adapter.coef_table(stub)


@pytest.mark.integration
def test_maketables_pymc_stat_hooks(mock_pymc_sample):
    """PyMC adapter stat/metadata methods should return expected types."""
    df = cp.load_data("did")
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )

    adapter = PyMCMaketablesAdapter()
    assert adapter.stat(result, "N") is not None
    assert adapter.stat(result, "model_type") == "bayesian"
    assert adapter.stat(result, "se_type") == "Bayesian posterior"
    assert adapter.stat(result, "experiment_type") == "DifferenceInDifferences"
    assert adapter.stat(result, "unknown_key") is None
    assert adapter.vcov_info(result) == {"se_type": "Bayesian posterior", "vcov": None}
    labels = adapter.stat_labels(result)
    assert labels is not None
    assert labels["N"] == "N"
    assert "Bayesian R2" in labels.values()
    defaults = adapter.default_stat_keys(result)
    assert defaults is not None
    assert "N" in defaults


@pytest.mark.integration
def test_maketables_pymc_no_labels_raises(mock_pymc_sample):
    """PyMC adapter should raise when experiment has empty labels."""
    adapter = PyMCMaketablesAdapter()
    stub = _Stub(labels=[])
    with pytest.raises(ValueError, match="no coefficient labels"):
        adapter.coef_table(stub)
