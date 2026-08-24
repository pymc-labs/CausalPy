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
"""Prior-phase contract tests (issue #1092).

Every experiment whose Bayesian backend declares prior-predictive support
must serve ``plot(group="prior")`` with the reduced single-panel layout and
an effect summary framed as a plausibility check rather than a causal claim.
Point-estimate and forecast backends must raise their documented capability
errors instead. Adapter-level guard paths are exercised directly so a guard
that fails open cannot slip through.
"""

import numpy as np
import pandas as pd
import pytest

import causalpy as cp
from causalpy.custom_exceptions import (
    GroupNotSampledException,
    PriorPredictiveNotSupportedException,
)
from causalpy.data.simulate_data import (
    generate_piecewise_its_data,
    generate_staggered_did_data,
)
from causalpy.experiments import model_adapter

SAMPLE_KWARGS = {
    "draws": 20,
    "tune": 20,
    "chains": 1,
    "progressbar": False,
    "random_seed": 42,
}
PRIOR_KWARGS = {"draws": 25, "random_seed": 42}

PRIOR_FRAMING = "Prior predictive check"


def _linear_model():
    return cp.pymc_models.LinearRegression(sample_kwargs=dict(SAMPLE_KWARGS))


# ---------------------------------------------------------------------------
# Fitted-experiment factories: one tiny fit each, shared session-wide by the
# parametrized assertions below.
# ---------------------------------------------------------------------------


def _make_its(its_data):
    return cp.InterruptedTimeSeries(
        its_data,
        treatment_time=pd.Timestamp("2017-06-01"),
        formula="y ~ 1 + t",
        model=_linear_model(),
    ).fit()


def _make_its_three_period(its_data):
    return cp.InterruptedTimeSeries(
        its_data,
        treatment_time=pd.Timestamp("2017-06-01"),
        treatment_end_time=pd.Timestamp("2017-09-01"),
        formula="y ~ 1 + t",
        model=_linear_model(),
    ).fit()


def _make_sc(sc_data):
    return cp.SyntheticControl(
        sc_data,
        70,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=dict(SAMPLE_KWARGS)),
    ).fit()


def _make_piecewise():
    df, _ = generate_piecewise_its_data(N=100, seed=42)
    return cp.PiecewiseITS(
        df,
        formula="y ~ 1 + t + step(t, 50)",
        model=_linear_model(),
    ).fit()


def _make_did(did_data):
    return cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=_linear_model(),
    ).fit()


def _make_prepostnegd(anova1_data):
    return cp.PrePostNEGD(
        anova1_data,
        formula="post ~ 1 + C(group) + pre",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=_linear_model(),
    ).fit()


def _make_rd(rd_data):
    rd = rd_data.assign(treated=lambda x: x["treated"].astype(int))
    return cp.RegressionDiscontinuity(
        rd,
        formula="y ~ 1 + x + treated",
        treatment_threshold=0.5,
        model=_linear_model(),
    ).fit()


def _make_rk():
    from causalpy.tests.conftest import setup_regression_kink_data

    return cp.RegressionKink(
        setup_regression_kink_data(0.0),
        formula="y ~ 1 + x + I((x-0.0)*treated)",
        kink_point=0.0,
        model=_linear_model(),
    ).fit()


def _make_sdid():
    return cp.SyntheticDifferenceInDifferences(
        cp.load_data("sc"),
        70,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.SyntheticDifferenceInDifferencesWeightFitter(
            sample_kwargs=dict(SAMPLE_KWARGS)
        ),
    ).fit()


def _make_staggered():
    df = generate_staggered_did_data(
        n_units=20,
        n_time_periods=10,
        treatment_cohorts={4: 6, 8: 6},
        seed=42,
    )
    return cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        treatment_time_variable_name="treatment_time",
        model=_linear_model(),
    ).fit()


@pytest.fixture(scope="session")
def fitted_its(its_data):
    return _make_its(its_data)


@pytest.fixture(scope="session")
def fitted_its_three_period(its_data):
    return _make_its_three_period(its_data)


@pytest.fixture(scope="session")
def fitted_sc(sc_data):
    return _make_sc(sc_data)


@pytest.fixture(scope="session")
def fitted_piecewise():
    return _make_piecewise()


@pytest.fixture(scope="session")
def fitted_did(did_data):
    return _make_did(did_data)


@pytest.fixture(scope="session")
def fitted_prepostnegd(anova1_data):
    return _make_prepostnegd(anova1_data)


@pytest.fixture(scope="session")
def fitted_rd(rd_data):
    return _make_rd(rd_data)


@pytest.fixture(scope="session")
def fitted_rk():
    return _make_rk()


@pytest.fixture(scope="session")
def fitted_sdid():
    return _make_sdid()


@pytest.fixture(scope="session")
def fitted_staggered():
    return _make_staggered()


# ---------------------------------------------------------------------------
# Prior-phase contract per experiment
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "experiment_name",
    [
        "its",
        "its_three_period",
        "sc",
        "piecewise",
        "did",
        "prepostnegd",
        "rd",
        "rk",
        "sdid",
        "staggered",
    ],
)
def test_prior_plot_reduced_layout(request, experiment_name):
    """plot(group='prior') renders exactly one axes for every capable experiment."""
    exp = request.getfixturevalue(f"fitted_{experiment_name}")
    exp.sample_prior_predictive(**PRIOR_KWARGS)
    fig, ax = exp.plot(group="prior", show=False)
    assert len(np.atleast_1d(ax)) == 1


@pytest.mark.integration
@pytest.mark.parametrize(
    "experiment_name",
    ["its", "sc", "piecewise", "did", "prepostnegd", "rd", "rk", "sdid", "staggered"],
)
def test_prior_effect_summary_is_framed_as_plausibility_check(request, experiment_name):
    """effect_summary(group='prior') prose is prefixed as a non-causal check."""
    exp = request.getfixturevalue(f"fitted_{experiment_name}")
    summary = exp.effect_summary(group="prior")
    # Scalar-effect summaries prepend the framing; SDID weaves it into the
    # sentence as the period label. Either way the prose must be framed as
    # a plausibility check rather than a causal claim.
    assert PRIOR_FRAMING in summary.text


@pytest.mark.integration
def test_posterior_plot_keeps_full_layout_after_prior(fitted_its):
    """The posterior layout is unchanged after a prior phase ran."""
    fitted_its.sample_prior_predictive(**PRIOR_KWARGS)
    fig, ax = fitted_its.plot(show=False)
    assert len(list(ax)) == 3


@pytest.mark.integration
def test_prior_get_plot_data_runs_without_caching(fitted_its):
    """get_plot_data(group='prior') returns a frame without caching it on self."""
    fitted_its.sample_prior_predictive(**PRIOR_KWARGS)
    out = fitted_its.get_plot_data(group="prior")
    assert isinstance(out, pd.DataFrame)
    assert not hasattr(fitted_its, "plot_data")


# ---------------------------------------------------------------------------
# Experiment/base guard paths
# ---------------------------------------------------------------------------


def test_resolve_group_rejects_unknown_group(fitted_its):
    with pytest.raises(ValueError, match="group must be"):
        fitted_its._resolve_group("bogus")


def test_no_bundle_experiment_guard_names_fit():
    """_supports_results=False experiments still fail fast pre-fit."""

    class _Panel:
        pass

    units = [f"unit_{i}" for i in range(4)]
    rows = []
    rng = np.random.default_rng(0)
    for u_idx, unit in enumerate(units):
        effect = float(rng.normal())
        for t in range(8):
            rows.append(
                {
                    "unit": unit,
                    "time": t,
                    "treatment": int(t >= 4 and u_idx < 2),
                    "x1": float(rng.normal()),
                    "y": effect + 0.1 * t + float(rng.normal()),
                }
            )
    data = pd.DataFrame(rows)

    from sklearn.linear_model import LinearRegression as SkLinearRegression

    exp = cp.PanelRegression(
        data=data,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=SkLinearRegression(),
    )
    with pytest.raises(GroupNotSampledException, match=r"fit\(\) first"):
        exp._resolve_group("posterior")


def _iv_inputs(n: int = 60):
    rng = np.random.default_rng(7)
    e1 = rng.normal(0, 3, n)
    e2 = rng.normal(0, 1, n)
    Z = rng.uniform(0, 1, n)
    X = -1 + 4 * Z + e2 + 2 * e1
    y = 2 + 3 * X + 3 * e1
    df = pd.DataFrame({"y": y, "X": X, "Z": Z})
    return {
        "data": df[["y", "X"]],
        "instruments_data": df[["X", "Z"]],
        "formula": "y ~ 1 + X",
        "instruments_formula": "X ~ 1 + Z",
    }


@pytest.mark.integration
def test_generate_report_swallows_only_not_implemented():
    """IV has no effect summary: report generation degrades gracefully, while a
    missing draw group propagates so the user learns to call fit()."""
    iv_exp = cp.InstrumentalVariable(**_iv_inputs()).fit()
    html = iv_exp.generate_report(include_plots=False)
    assert len(html) > 0

    # An experiment WITH a summary implementation must propagate the
    # group-not-sampled error instead of silently omitting the section.
    rng = np.random.default_rng(1)
    n = 30
    its_df = pd.DataFrame(
        {"t": np.arange(n, dtype=float), "y": rng.normal(size=n)},
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )
    unfitted = cp.InterruptedTimeSeries(
        its_df,
        treatment_time=pd.Timestamp("2020-01-15"),
        formula="y ~ 1 + t",
        model=_linear_model(),
    )
    with pytest.raises(GroupNotSampledException):
        unfitted.generate_report(include_plots=False)


@pytest.mark.integration
def test_iv_build_is_noop_and_refit_warns_and_overwrites():
    """IV.build() returns self untouched; refitting warns then overwrites."""
    iv_exp = cp.InstrumentalVariable(**_iv_inputs())
    assert iv_exp.build() is iv_exp

    iv_exp.fit()
    n_draws = iv_exp.idata["posterior"].sizes["draw"]
    with pytest.warns(UserWarning, match="previous posterior"):
        iv_exp.fit()
    assert iv_exp.idata["posterior"].sizes["draw"] == n_draws


def test_format_r2_score_none_renders_empty():
    from causalpy.plot_utils import format_r2_score

    assert format_r2_score(None) == ""


# ---------------------------------------------------------------------------
# PyMCModel guard paths
# ---------------------------------------------------------------------------


def _tiny_xy(n: int = 10):
    rng = np.random.default_rng(0)
    X = xr_canonical(rng.normal(size=(n, 2)), dims=("obs_ind", "coeffs"), n=n)
    y = xr_canonical(rng.normal(size=(n, 1)), dims=("obs_ind", "treated_units"), n=n)
    return X, y


def xr_canonical(values, *, dims, n):
    """Build a minimal canonical DataArray with integer obs_ind coords."""
    import xarray as xr

    other = "coeffs" if "coeffs" in dims else "treated_units"
    labels = [f"c{i}" for i in range(values.shape[-1])]
    return xr.DataArray(
        values,
        dims=list(dims),
        coords={"obs_ind": np.arange(n), other: labels},
    )


def test_model_requires_built_graph_before_sampling():
    model = cp.pymc_models.LinearRegression()
    with pytest.raises(RuntimeError, match="has not been built"):
        model.sample_prior_predictive()
    with pytest.raises(RuntimeError, match="has not been built"):
        model.sample_posterior()


def test_model_require_group_guards():
    model = cp.pymc_models.LinearRegression()
    with pytest.raises(ValueError, match="group must be"):
        model.require_group("bogus")
    with pytest.raises(GroupNotSampledException, match=r"sample_prior_predictive\(\)"):
        model.require_group("prior")


def test_sample_returning_none_raises_clearly(monkeypatch):
    """pm.sample() returning None surfaces as RuntimeError, not silent success."""
    import causalpy.pymc_models as pm_mod

    X, y = _tiny_xy()
    model = cp.pymc_models.LinearRegression()
    model.build(
        X,
        y,
        coords={
            "coeffs": list(X.coords["coeffs"].values),
            "obs_ind": np.arange(X.shape[0]),
            "treated_units": ["unit_0"],
        },
    )
    monkeypatch.setattr(pm_mod.pm, "sample", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="returned None"):
        model.sample_posterior()


# ---------------------------------------------------------------------------
# Adapter guard paths (no sampling required)
# ---------------------------------------------------------------------------


def _sklearn_rd(rd_data):
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    return cp.RegressionDiscontinuity(
        rd_data.assign(treated=lambda x: x["treated"].astype(int)),
        formula="y ~ 1 + x + treated",
        treatment_threshold=0.5,
        model=SkLinearRegression(),
    )


def test_sklearn_adapter_capability_guards(rd_data):
    """Direct sklearn-adapter calls raise the documented capability errors."""
    exp = _sklearn_rd(rd_data)
    backend = exp._model_backend

    with pytest.raises(PriorPredictiveNotSupportedException):
        backend.sample_prior_predictive()

    # Idempotent build records once; a second call is a no-op.
    X = exp.design["X"]
    backend.build(X, exp.design["y"])
    recorded = backend._fit_inputs
    backend.build(X, exp.design["y"])
    assert backend._fit_inputs is recorded

    with pytest.raises(TypeError, match="no sampling overrides"):
        backend.sample_posterior(draws=5)

    with pytest.raises(GroupNotSampledException, match="'prior' draws"):
        backend.predict(X.isel(obs_ind=slice(0, 5)), group="prior")

    with pytest.raises(GroupNotSampledException, match="'prior' draws"):
        backend.coefficients(group="prior")

    from sklearn.linear_model import LinearRegression as SkLinearRegression

    with pytest.raises(RuntimeError, match="not been recorded"):
        model_adapter.SklearnModelAdapter(SkLinearRegression()).sample_posterior()

    with pytest.raises(GroupNotSampledException, match=r"fit\(\) first"):
        backend.coefficients()


def test_sklearn_adapter_coefficients_after_fit(rd_data):
    exp = _sklearn_rd(rd_data).fit()
    coefs = exp._model_backend.coefficients()
    assert "chain" in coefs.dims


def test_base_adapter_defaults():
    """The abstract base's capability defaults are False; its verbs raise."""

    class MinimalAdapter(model_adapter.ModelAdapter):
        @property
        def model(self):
            return object()

        @property
        def kind(self):
            return "pymc"

        @property
        def idata(self):
            return None

        def fit(self, X, y, *, coords=None):
            return None

        def predict(self, X, *, coords=None, out_of_sample=False, group="posterior"):
            raise AssertionError

        def sample_posterior(self, **kwargs):
            raise AssertionError

        def score(self, X, y, *, coords=None):
            raise AssertionError

        def coefficients(self, *, group="posterior"):
            raise AssertionError

    adapter = MinimalAdapter()
    assert adapter.is_built is False
    assert adapter.has_posterior is False
    assert adapter.has_prior is False
    assert adapter.supports_prior_predictive is False
    with pytest.raises(PriorPredictiveNotSupportedException):
        adapter.sample_prior_predictive()
    with pytest.raises(NotImplementedError):
        adapter.build(None, None)


def test_pymc_adapter_mixed_mapping_inputs_raise():
    backend = model_adapter.PyMCModelAdapter(cp.pymc_models.LinearRegression())
    with pytest.raises(TypeError, match="both be mappings or both be arrays"):
        backend.build({"unit": object()}, [1, 2])


def test_pymc_adapter_prior_capability_error():
    backend = model_adapter.PyMCModelAdapter(
        cp.pymc_models.InstrumentalVariableRegression()
    )
    assert backend.supports_prior_predictive is False
    with pytest.raises(PriorPredictiveNotSupportedException, match="Instrumental"):
        backend.sample_prior_predictive()


def test_pymc_adapter_coefficients_group_guard():
    backend = model_adapter.PyMCModelAdapter(cp.pymc_models.LinearRegression())
    with pytest.raises(GroupNotSampledException, match=r"sample_prior_predictive\(\)"):
        backend.coefficients(group="prior")


def test_forecast_adapter_guards():
    pytest.importorskip("pymc_forecast")
    from causalpy.pymc_forecast_models import PyMCForecastModel

    backend = model_adapter.PyMCForecastAdapter(
        PyMCForecastModel(model_fn=lambda: None)
    )
    with pytest.raises(PriorPredictiveNotSupportedException):
        backend.sample_prior_predictive()

    with pytest.raises(RuntimeError, match="have not been recorded"):
        backend.sample_posterior()

    backend.build(np.zeros((3, 1)), np.zeros((3, 1)))
    with pytest.raises(TypeError, match="no sampling overrides"):
        backend.sample_posterior(draws=5)

    with pytest.raises(GroupNotSampledException, match="'prior' draws"):
        backend.predict(np.zeros((3, 1)), group="prior")


@pytest.mark.integration
def test_rd_ols_effect_summary_via_helper(rd_data):
    """The OLS posterior path flows through the shared RD helper."""
    exp = _sklearn_rd(rd_data).fit()
    summary = exp.effect_summary()
    assert isinstance(summary.text, str) and summary.text


def test_rd_helper_requires_experiment_on_ols_path(fitted_rd):
    """Singleton-draw bundles on the OLS path need the fitted experiment."""
    import xarray as xr

    from causalpy.reporting import _effect_summary_rd

    n = fitted_rd.result.predictions.sizes["obs_ind"]
    point = xr.DataArray(
        np.zeros((1, 1, n)),
        dims=("chain", "draw", "obs_ind"),
        coords={"chain": [0], "draw": [0], "obs_ind": range(n)},
    )
    singleton = type(fitted_rd.result)(
        predictions=point,
        discontinuity_at_threshold=point.isel(obs_ind=0),
        score=None,
    )
    with pytest.raises(TypeError, match="pass experiment=self"):
        _effect_summary_rd(singleton)


def test_extract_window_datetime_slice():
    """Slicing a datetime post index goes through the slice branch."""
    from causalpy.reporting import _extract_window

    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    impact = xr_zero(idx)
    windowed, coords = _extract_window(
        impact, idx, slice(pd.Timestamp("2020-01-02"), None)
    )
    assert len(coords) == 4


def xr_zero(index):
    import xarray as xr

    return xr.DataArray(
        np.zeros((1, 1, len(index))),
        dims=("chain", "draw", "obs_ind"),
        coords={"chain": [0], "draw": [0], "obs_ind": index},
    )


def test_propensity_score_build_is_idempotent():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 2))
    t = rng.integers(0, 2, 20).astype(float)
    model = cp.pymc_models.PropensityScore()
    coords = {"coeffs": ["a", "b"], "obs_ind": np.arange(20)}
    model.build(X, t, coords=coords)
    nodes = dict(model._build_data_nodes)
    model.build(X, t, coords=coords)
    assert model._build_data_nodes == nodes


def test_state_space_predict_rejects_prior_group():
    model = cp.pymc_models.StateSpaceTimeSeries()
    with pytest.raises(GroupNotSampledException, match="Kalman"):
        model.predict(group="prior")
