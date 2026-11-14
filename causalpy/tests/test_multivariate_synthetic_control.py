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
"""Tests for :class:`MultivariateSyntheticControl` and :class:`MvWeightedSumFitter`.

Save as ``causalpy/tests/test_multivariate_synthetic_control.py``.

Structure tests use ``mock_pymc_sample`` and never run real MCMC. The single
``correctness`` test uses ``real_pymc_sampling`` to check that the extra
predictors actually correct the bias of an outcome-only synthetic control.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
from pymc_extras.prior import Prior
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.custom_exceptions import DataException, GroupNotSampledException
from causalpy.experiments.multivariate_synthetic_control import (
    MultivariateSyntheticControl,
)
from causalpy.pymc_models import MvWeightedSumFitter

sample_kwargs = {
    "tune": 20,
    "draws": 20,
    "chains": 2,
    "cores": 2,
    "progressbar": False,
    "random_seed": 1,
}
DONORS = ["a", "b", "c", "d"]
N_TIME = 24
TREATMENT_TIME = 16


def _make_long_data(predictors=("x1", "x2"), seed=0):
    """Small long-format panel: one outcome, extra predictors, a +2 effect."""
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.normal(size=N_TIME)) / 3
    rows = []
    for k, unit in enumerate([*DONORS, "treated"]):
        level = 1.0 + 0.2 * k
        for i in range(N_TIME):
            effect = 2.0 if unit == "treated" and i >= TREATMENT_TIME else 0.0
            row = {
                "unit": unit,
                "time": i,
                "y": 10 + level * trend[i] + effect + rng.normal(0, 0.1),
            }
            for j, name in enumerate(predictors):
                row[name] = 5 + (j + 1) * level + 0.3 * trend[i] + rng.normal(0, 0.1)
            rows.append(row)
    return pd.DataFrame(rows)


def _build(data, predictors=("x1", "x2"), control_units=None, **kwargs):
    """Construct the experiment with light sampling settings."""
    kwargs.setdefault("model", MvWeightedSumFitter(sample_kwargs=sample_kwargs))
    return MultivariateSyntheticControl(
        data,
        treatment_time=TREATMENT_TIME,
        outcome="y",
        predictors=list(predictors),
        control_units=DONORS if control_units is None else control_units,
        treated_units=["treated"],
        unit_column="unit",
        time_column="time",
        **kwargs,
    )


def _arrays(n_features, n_obs=10, n_donors=3):
    """Donor (X) and treated (y) arrays in the layout the model expects."""
    rng = np.random.default_rng(0)
    names = ["y", *[f"x{i}" for i in range(1, n_features)]]
    obs = np.arange(n_obs)
    X = xr.DataArray(
        rng.normal(size=(n_obs, n_donors, n_features)),
        dims=["obs_ind", "coeffs", "features"],
        coords={
            "obs_ind": obs,
            "coeffs": [f"d{i}" for i in range(n_donors)],
            "features": names,
        },
    )
    y = xr.DataArray(
        rng.normal(size=(n_obs, n_features, 1)),
        dims=["obs_ind", "features", "treated_units"],
        coords={"obs_ind": obs, "features": names, "treated_units": ["t0"]},
    )
    return X, y


# ---------------------------------------------------------------------------
# MvWeightedSumFitter: construction and priors (no sampling)
# ---------------------------------------------------------------------------


def test_invalid_v_prior_is_rejected():
    with pytest.raises(ValueError, match="v_prior"):
        MvWeightedSumFitter(v_prior="sparse")  # type: ignore[arg-type]


def test_build_model_rejects_dict_inputs():
    donors = xr.DataArray(np.zeros((3, 2, 2)), dims=["obs_ind", "coeffs", "features"])
    treated = xr.DataArray(
        np.zeros((3, 2, 1)), dims=["obs_ind", "features", "treated_units"]
    )
    model = MvWeightedSumFitter()
    with pytest.raises(TypeError, match="single DataArray"):
        model.build_model(
            {"donors": donors},
            {"treated": treated},
            coords={"features": ["outcome", "predictor"]},
        )


def test_clone_preserves_v_prior_and_sample_kwargs():
    model = MvWeightedSumFitter(v_prior="uniform", sample_kwargs=sample_kwargs)
    clone = model._clone()
    assert clone is not model
    assert isinstance(clone, MvWeightedSumFitter)
    assert clone.v_prior == "uniform"
    assert clone.sample_kwargs == model.sample_kwargs


def test_clone_accepts_replacement_priors():
    clone = MvWeightedSumFitter()._clone(
        priors={"sigma_feat": Prior("HalfNormal", sigma=2)}
    )
    assert "sigma_feat" in clone._user_priors


def test_priors_from_data_requires_an_auxiliary_predictor():
    X, y = _arrays(n_features=1)
    with pytest.raises(ValueError, match="auxiliary predictor"):
        MvWeightedSumFitter().priors_from_data(X, y)


@pytest.mark.parametrize(
    ("v_prior", "n_features", "has_v"),
    [("learned", 3, True), ("learned", 2, False), ("uniform", 3, False)],
)
def test_priors_from_data_creates_v_only_when_it_is_learned(v_prior, n_features, has_v):
    X, y = _arrays(n_features)
    priors = MvWeightedSumFitter(v_prior=v_prior).priors_from_data(X, y)
    assert ("v" in priors) is has_v
    assert {"beta", "sigma_feat", "y_hat"} <= set(priors)


def test_build_model_requires_feature_coords():
    X, y = _arrays(n_features=2)
    with pytest.raises(ValueError, match="features"):
        MvWeightedSumFitter().build_model(X, y, coords=None)


def test_data_setter_requires_a_built_model():
    X, _ = _arrays(n_features=2)
    with pytest.raises(ValueError, match="not found"):
        MvWeightedSumFitter()._data_setter(X)


# ---------------------------------------------------------------------------
# MultivariateSyntheticControl: data preparation and validation
# ---------------------------------------------------------------------------


def test_is_exported_and_a_synthetic_control():
    assert cp.MultivariateSyntheticControl is MultivariateSyntheticControl
    assert issubclass(MultivariateSyntheticControl, cp.SyntheticControl)


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_designs_are_stacked_and_standardised():
    exp = _build(_make_long_data())
    control = exp.pre_design_mv["control"]
    treated = exp.pre_design_mv["treated"]

    assert exp.features == ["y", "x1", "x2"]
    assert control.dims == ("obs_ind", "coeffs", "features")
    assert treated.dims == ("obs_ind", "features", "treated_units")
    assert control.sizes["coeffs"] == len(DONORS)
    assert control.sizes["obs_ind"] == TREATMENT_TIME

    # The outcome keeps its original scale: it is what the counterfactual uses.
    np.testing.assert_allclose(
        control.sel(features="y").values, exp.pre_design["control"].values
    )
    # Predictors are standardised with pooled pre-treatment statistics.
    for name in ("x1", "x2"):
        pooled = np.concatenate(
            [
                control.sel(features=name).values.ravel(),
                treated.sel(features=name).values.ravel(),
            ]
        )
        assert pooled.mean() == pytest.approx(0.0, abs=1e-9)
        assert pooled.std(ddof=1) == pytest.approx(1.0)


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_post_treatment_predictors_may_be_missing():
    data = _make_long_data()
    data.loc[data["time"] >= TREATMENT_TIME, ["x1", "x2"]] = np.nan
    exp = _build(data)
    post = exp.post_design_mv["control"]
    assert np.all(post.sel(features=["x1", "x2"]).values == 0.0)
    assert np.all(np.isfinite(post.sel(features="y").values))


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_polars_input_matches_pandas():
    data = _make_long_data()
    from_pandas = _build(data)
    from_polars = _build(pl.from_pandas(data))
    xr.testing.assert_allclose(
        from_pandas.pre_design_mv["control"], from_polars.pre_design_mv["control"]
    )
    xr.testing.assert_allclose(
        from_pandas.post_design_mv["treated"], from_polars.post_design_mv["treated"]
    )


def test_missing_column_raises():
    with pytest.raises(DataException, match="not found in data"):
        _build(_make_long_data().drop(columns="x2"))


def test_duplicate_unit_time_rows_raise():
    data = _make_long_data()
    with pytest.raises(DataException, match="duplicate"):
        _build(pd.concat([data, data.iloc[[0]]]))


def test_unknown_unit_raises():
    with pytest.raises(DataException, match="not found in 'unit'"):
        _build(_make_long_data(), control_units=[*DONORS, "missing"])


@pytest.mark.parametrize("predictors", [[], ["y"]])
def test_needs_at_least_one_extra_predictor(predictors):
    with pytest.raises(ValueError, match="at least one predictor"):
        _build(_make_long_data(), predictors=predictors)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_constant_predictor_raises():
    data = _make_long_data()
    data["x1"] = 1.0
    with pytest.raises(DataException, match="constant"):
        _build(data)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_missing_pre_treatment_predictor_raises():
    data = _make_long_data()
    data.loc[(data["unit"] == "a") & (data["time"] == 0), "x1"] = np.nan
    with pytest.raises(DataException, match="before treatment"):
        _build(data)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_missing_post_treatment_outcome_raises():
    data = _make_long_data()
    data.loc[(data["unit"] == "treated") & (data["time"] == TREATMENT_TIME), "y"] = (
        np.nan
    )
    with pytest.raises(DataException, match="after treatment"):
        _build(data)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sklearn_models_are_rejected():
    with pytest.raises(ValueError, match="OLS models not supported"):
        _build(_make_long_data(), model=LinearRegression())


# ---------------------------------------------------------------------------
# MultivariateSyntheticControl: fitting (sampling is mocked)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    ("v_prior", "predictors"),
    [
        ("learned", ("x1", "x2")),
        ("uniform", ("x1", "x2")),
        ("learned", ("x1",)),
    ],
)
def test_fit_populates_results_and_predictor_weights(
    mock_pymc_sample, v_prior, predictors
):
    exp = _build(
        _make_long_data(predictors=predictors),
        predictors=predictors,
        model=MvWeightedSumFitter(v_prior=v_prior, sample_kwargs=sample_kwargs),
    ).fit()

    impact = exp.result.impact_post
    assert impact.sizes["obs_ind"] == N_TIME - TREATMENT_TIME
    assert "treated_units" in impact.dims

    v = exp.predictor_weights()
    assert v.sizes["features_aux"] == len(predictors)
    np.testing.assert_allclose(v.sum("features_aux"), 1.0, rtol=1e-6)
    if v_prior == "uniform" or len(predictors) == 1:
        np.testing.assert_allclose(v, 1 / len(predictors))


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    ("v_prior", "expected"), [("learned", True), ("uniform", False)]
)
def test_summary_reports_predictor_weights_only_when_learned(
    mock_pymc_sample, capsys, v_prior, expected
):
    exp = _build(
        _make_long_data(),
        model=MvWeightedSumFitter(v_prior=v_prior, sample_kwargs=sample_kwargs),
    ).fit()
    exp.summary(round_to=2)
    assert ("Predictor weights (v)" in capsys.readouterr().out) is expected


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_predictor_weights_require_a_fit():
    exp = _build(_make_long_data())
    with pytest.raises(GroupNotSampledException, match="fit"):
        exp.predictor_weights()


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_auto_scale_sigma_applies_to_the_outcome_likelihood(mock_pymc_sample):
    exp = _build(_make_long_data()).fit()
    sigma = exp.model.priors["y_hat"].parameters["sigma"]
    assert sigma.distribution == "Exponential"


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_opt_out_of_auto_scaling_keeps_the_v_prior(mock_pymc_sample):
    """The opt-out fits a clone, so ``_clone`` must carry ``v_prior`` along."""
    model = MvWeightedSumFitter(v_prior="uniform", sample_kwargs=sample_kwargs)
    exp = _build(_make_long_data(), model=model, auto_scale_sigma=False).fit()
    assert exp.model is not model
    assert exp.model.v_prior == "uniform"
    sigma = exp.model.priors["y_hat"].parameters["sigma"]
    assert sigma.distribution == "HalfNormal"


# ---------------------------------------------------------------------------
# Correctness: extra predictors remove the bias an outcome-only fit cannot
# ---------------------------------------------------------------------------


def _make_exposure_scenario(seed=0, n_time=60, treatment_time=40, n_donors=10):
    """Donors share almost the same pre-treatment outcome path but differ in
    their exposure to a shock that only starts after treatment.

    The treated unit is more exposed (2.5) than the average donor (1.0), and the
    true effect is +2. Pre-treatment GDP cannot reveal exposure, but
    ``industry`` and ``exports`` are proxies of it, so only a model that matches
    on them recovers the counterfactual. ``irrelevant`` and ``noise`` are
    uninformative.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_time)
    common = np.cumsum(rng.normal(size=n_time)) / 3 + np.sin(t / 5)
    shock = np.where(
        t >= treatment_time, 2.0 * (1 - np.exp(-(t - treatment_time) / 6)), 0.0
    )
    donors = [f"c{i}" for i in range(n_donors)]
    exposure = rng.permutation(np.linspace(-2, 4, n_donors))
    a = 10 + rng.normal(0, 0.03, n_donors)
    b = 1 + rng.normal(0, 0.03, n_donors)
    z = rng.normal(size=n_donors)

    def unit_rows(name, a_u, b_u, exposure_u, z_u, effect):
        return [
            {
                "unit": name,
                "time": i,
                "gdp": a_u
                + b_u * common[i]
                + exposure_u * shock[i]
                + effect * (i >= treatment_time)
                + rng.normal(0, 0.05),
                "industry": 5 + 2 * exposure_u + 0.3 * common[i] + rng.normal(0, 0.1),
                "exports": -1 + 1.5 * exposure_u + 0.2 * common[i] + rng.normal(0, 0.1),
                "irrelevant": z_u + rng.normal(0, 0.1),
                "noise": rng.normal(),
            }
            for i in range(n_time)
        ]

    rows = []
    for k, name in enumerate(donors):
        rows += unit_rows(name, a[k], b[k], exposure[k], z[k], 0.0)
    rows += unit_rows("treated", a.mean(), b.mean(), 2.5, 0.3, 2.0)
    return pd.DataFrame(rows), donors, treatment_time


@pytest.mark.correctness
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_predictors_recover_the_effect_an_outcome_only_fit_misses(real_pymc_sampling):
    """With informative predictors the effect is close to the true +2.

    An outcome-only synthetic control is biased upward by about +2 in this
    scenario (its donors cannot be told apart before treatment).
    """
    data, donors, treatment_time = _make_exposure_scenario()
    exp = MultivariateSyntheticControl(
        data,
        treatment_time=treatment_time,
        outcome="gdp",
        predictors=["industry", "exports", "irrelevant", "noise"],
        control_units=donors,
        treated_units=["treated"],
        unit_column="unit",
        time_column="time",
        min_donor_correlation=-1.0,
        model=MvWeightedSumFitter(
            sample_kwargs={
                "draws": 300,
                "tune": 600,
                "chains": 2,
                "cores": 2,
                "target_accept": 0.95,
                "random_seed": 0,
                "progressbar": False,
            }
        ),
    ).fit()
    effect = float(
        exp.result.impact_post.sel(treated_units="treated").mean(
            ("chain", "draw", "obs_ind")
        )
    )
    assert effect == pytest.approx(2.0, abs=0.5)
