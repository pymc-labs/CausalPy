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
"""Tests for the ``auto_scale_sigma`` feature of :class:`SyntheticControl`.

The feature replaces the stock fitters' ``sigma ~ HalfNormal(1)`` likelihood
prior with ``sigma ~ Exponential(2/s)``, where *s* is the pre-treatment
standard deviation of the treated data, computed *per treated unit*. A unit whose
spread cannot be estimated falls back to ``s = 1`` with a warning, so data that
fitted under the legacy default keeps fitting.

The tests use ``mock_pymc_sample`` where model construction is needed and never
run real MCMC.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pymc_extras.prior import Prior

import causalpy as cp
from causalpy.pymc_models import SoftmaxWeightedSumFitter, WeightedSumFitter

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2, "progressbar": False}

# Both weighted-sum fitters share the same per-treated-unit ``y_hat`` sigma prior,
# so auto-scaling must behave identically for each.
FITTERS = [WeightedSumFitter, SoftmaxWeightedSumFitter]


def _make_data(treated_scales, n=60, treatment_time=45, seed=42):
    """Build a synthetic-control dataset with one treated column per entry in
    ``treated_scales``.

    A bespoke builder is used here (rather than ``cp.load_data("sc")``) because
    these tests need treated units on deliberately different scales, and a
    constant pre-treatment series — cases the canned ``sc`` dataset cannot
    express. Each treated unit is the control mean scaled by its factor, so
    units can be placed orders of magnitude apart on demand.
    """
    rng = np.random.default_rng(seed)
    controls = {
        c: rng.normal(10, 2, n).cumsum() / 10 + rng.normal(0, 1, n)
        for c in ["a", "b", "c"]
    }
    df = pd.DataFrame(controls)
    base = df[["a", "b", "c"]].mean(axis=1)
    treated_units = []
    for i, scale in enumerate(treated_scales):
        name = f"treated_{i}"
        df[name] = scale * base + rng.normal(0, scale, n)
        treated_units.append(name)
    return df, treatment_time, treated_units


def _sigma_prior(result):
    """Return the inner ``sigma`` Prior of the model's ``y_hat`` prior."""
    return result.model.priors["y_hat"].parameters["sigma"]


def _fitter(cls=WeightedSumFitter, **kwargs):
    return cls(sample_kwargs={**sample_kwargs, "random_seed": 1}, **kwargs)


def _expected_lam(df, treatment_time, treated_units):
    """Independently compute 2/s per unit with pandas over the pre-treatment
    rows (``index < treatment_time``), to cross-check the xarray-based impl."""
    pre = df[df.index < treatment_time][treated_units]
    return (2 / pre.std(ddof=1)).values


def _pre_treatment_design(df, treatment_time, treated_units):
    """Return labeled pre-treatment control and treated arrays."""
    pre = df[df.index < treatment_time]
    X = xr.DataArray(
        pre[["a", "b", "c"]].to_numpy(),
        dims=["obs_ind", "coeffs"],
        coords={"obs_ind": pre.index, "coeffs": ["a", "b", "c"]},
    )
    y = xr.DataArray(
        pre[treated_units].to_numpy(),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": pre.index, "treated_units": treated_units},
    )
    return X, y


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_auto_scale_sets_exponential_prior_matching_2_over_s(
    mock_pymc_sample, fitter_cls
):
    """After fit, the y_hat sigma prior is Exponential with lam = 2/s."""
    df, tt, treated = _make_data([1.0])
    model = _fitter(fitter_cls)
    result = cp.SyntheticControl(
        df,
        tt,
        control_units=["a", "b", "c"],
        treated_units=treated,
        model=model,
    )
    sigma = _sigma_prior(result)
    assert sigma.distribution == "Exponential"
    np.testing.assert_allclose(
        np.asarray(sigma.parameters["lam"]), _expected_lam(df, tt, treated)
    )
    assert result.model is model


@pytest.mark.integration
@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_direct_fit_consumes_data_scaled_prior(mock_pymc_sample, fitter_cls):
    """The fitter-level prior is used when a stock fitter is fit directly."""
    df, tt, treated = _make_data([1.0])
    X, y = _pre_treatment_design(df, tt, treated)
    model = _fitter(fitter_cls)
    model.fit(
        X,
        y,
        coords={
            "obs_ind": X.obs_ind.values,
            "coeffs": X.coeffs.values,
            "treated_units": y.treated_units.values,
        },
    )
    sigma = model.priors["y_hat"].parameters["sigma"]
    assert sigma.distribution == "Exponential"
    np.testing.assert_allclose(
        np.asarray(sigma.parameters["lam"]), _expected_lam(df, tt, treated)
    )


@pytest.mark.integration
@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_direct_fit_accepts_dimension_only_treated_outcomes(
    mock_pymc_sample, fitter_cls
):
    """Direct fitting supports a treated-units dimension without labels."""
    df, tt, treated = _make_data([1.0])
    X, y = _pre_treatment_design(df, tt, treated)
    y = xr.DataArray(y.values, dims=["obs_ind", "treated_units"])
    model = _fitter(fitter_cls)
    model.fit(X, y)
    sigma = model.priors["y_hat"].parameters["sigma"]
    np.testing.assert_allclose(
        np.asarray(sigma.parameters["lam"]), _expected_lam(df, tt, treated)
    )


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_auto_scale_false_preserves_halfnormal_default(mock_pymc_sample, fitter_cls):
    """auto_scale_sigma=False leaves the HalfNormal(1) default untouched."""
    df, tt, treated = _make_data([1.0])
    result = cp.SyntheticControl(
        df,
        tt,
        control_units=["a", "b", "c"],
        treated_units=treated,
        model=_fitter(fitter_cls),
        auto_scale_sigma=False,
    )
    sigma = _sigma_prior(result)
    assert sigma.distribution == "HalfNormal"
    assert sigma.parameters["sigma"] == 1


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_auto_scale_false_skips_scale_estimation(mock_pymc_sample, fitter_cls):
    """The legacy opt-out never inspects a constant treated pre-period."""
    df, tt, treated = _make_data([1.0])
    df["treated_0"] = 5.0
    result = cp.SyntheticControl(
        df,
        tt,
        control_units=["a", "b", "c"],
        treated_units=treated,
        model=_fitter(fitter_cls),
        auto_scale_sigma=False,
    )
    sigma = _sigma_prior(result)
    assert sigma.distribution == "HalfNormal"
    assert sigma.parameters["sigma"] == 1


@pytest.mark.integration
@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_constant_treated_series_still_fits_with_auto_scaling(
    mock_pymc_sample, fitter_cls
):
    """A constant treated pre-period fits as it did before data scaling."""
    df, tt, treated = _make_data([1.0])
    df["treated_0"] = 5.0
    with pytest.warns(UserWarning, match="Cannot estimate the pre-treatment"):
        result = cp.SyntheticControl(
            df,
            tt,
            control_units=["a", "b", "c"],
            treated_units=treated,
            model=_fitter(fitter_cls),
        )
    sigma = _sigma_prior(result)
    assert sigma.distribution == "Exponential"
    assert np.asarray(sigma.parameters["lam"]) == 2.0


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_user_supplied_y_hat_prior_is_respected(mock_pymc_sample, fitter_cls):
    """An explicit y_hat prior disables auto-scaling (guard not triggered)."""
    df, tt, treated = _make_data([1.0])
    custom = Prior(
        "Normal",
        sigma=Prior("HalfNormal", sigma=42, dims=["treated_units"]),
        dims=["obs_ind", "treated_units"],
    )
    model = _fitter(fitter_cls, priors={"y_hat": custom})
    result = cp.SyntheticControl(
        df, tt, control_units=["a", "b", "c"], treated_units=treated, model=model
    )
    sigma = _sigma_prior(result)
    # Untouched: still the user's HalfNormal(42), not the auto Exponential.
    assert sigma.distribution == "HalfNormal"
    assert sigma.parameters["sigma"] == 42
    assert model.priors["y_hat"] is custom
    assert result.model._user_priors == {"y_hat": custom}
    assert result.model is model


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_multiple_treated_units_get_per_unit_lam(mock_pymc_sample, fitter_cls):
    """With multiple treated units on different scales, lam is a per-unit vector
    of 2/s_i — not a single broadcast scalar."""
    df, tt, treated = _make_data([1.0, 100.0])  # two units, ~100x apart in scale
    result = cp.SyntheticControl(
        df,
        tt,
        control_units=["a", "b", "c"],
        treated_units=treated,
        model=_fitter(fitter_cls),
    )
    lam = np.asarray(_sigma_prior(result).parameters["lam"])
    assert lam.shape == (2,)
    np.testing.assert_allclose(lam, _expected_lam(df, tt, treated))
    # The two rates must genuinely differ; a shared scalar would fail this.
    assert not np.isclose(lam[0], lam[1])


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_reused_model_does_not_carry_auto_scaled_prior_into_opt_out(
    mock_pymc_sample, fitter_cls
):
    """A scaled fit cannot turn a later opt-out fit into an Exponential prior."""
    first_df, tt, treated = _make_data([1.0])
    source_model = _fitter(fitter_cls)
    first = cp.SyntheticControl(
        first_df,
        tt,
        control_units=["a", "b", "c"],
        treated_units=treated,
        model=source_model,
    )
    second_df, _, _ = _make_data([100.0])
    second = cp.SyntheticControl(
        second_df,
        tt,
        control_units=["a", "b", "c"],
        treated_units=treated,
        model=first.model,
        auto_scale_sigma=False,
    )
    second_sigma = _sigma_prior(second)
    assert second_sigma.distribution == "HalfNormal"
    assert second_sigma.parameters["sigma"] == 1
    assert first.model is source_model
    assert source_model.priors["y_hat"].parameters["sigma"].distribution == (
        "Exponential"
    )
    assert second.model is not source_model
    assert source_model._clone().priors["y_hat"].parameters["sigma"].distribution == (
        "HalfNormal"
    )


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_reused_model_recomputes_the_auto_scale(mock_pymc_sample, fitter_cls):
    """A cloned model derives its next prior from the next treated outcome."""
    first_df, tt, treated = _make_data([1.0])
    first = cp.SyntheticControl(
        first_df,
        tt,
        control_units=["a", "b", "c"],
        treated_units=treated,
        model=_fitter(fitter_cls),
    )
    second_df, _, _ = _make_data([100.0])
    second = cp.SyntheticControl(
        second_df,
        tt,
        control_units=["a", "b", "c"],
        treated_units=treated,
        model=first.model._clone(),
    )
    first_lam = np.asarray(_sigma_prior(first).parameters["lam"])
    second_lam = np.asarray(_sigma_prior(second).parameters["lam"])
    np.testing.assert_allclose(second_lam, _expected_lam(second_df, tt, treated))
    assert not np.isclose(first_lam[0], second_lam[0])


@pytest.mark.parametrize("fitter_cls", FITTERS)
@pytest.mark.parametrize("invalid_kind", ["nan", "infinite"])
def test_non_finite_treated_outcome_raises_actionable_error(fitter_cls, invalid_kind):
    """A non-finite treated outcome is a data error and is rejected before fit."""
    df, tt, treated = _make_data([1.0, 10.0])
    df.loc[0, "treated_0"] = np.nan if invalid_kind == "nan" else np.inf
    X, y = _pre_treatment_design(df, tt, treated)
    with pytest.raises(ValueError, match="non-finite values") as error:
        _fitter(fitter_cls).priors_from_data(X, y)
    assert "treated_1" not in str(error.value)
    assert "treated_0" in str(error.value)


@pytest.mark.parametrize("fitter_cls", FITTERS)
@pytest.mark.parametrize("degenerate_kind", ["constant", "single_observation"])
def test_unestimable_scale_warns_and_falls_back_per_unit(fitter_cls, degenerate_kind):
    """A unit with no estimable spread keeps the legacy scale; others do not."""
    if degenerate_kind == "single_observation":
        df, tt, treated = _make_data([1.0], n=2, treatment_time=1)
    else:
        df, tt, treated = _make_data([1.0, 10.0])
        df["treated_0"] = 5.0
    X, y = _pre_treatment_design(df, tt, treated)
    with pytest.warns(UserWarning, match="Cannot estimate the pre-treatment") as record:
        priors = _fitter(fitter_cls).priors_from_data(X, y)
    message = str(record[0].message)
    assert "treated_0" in message
    lam = np.asarray(priors["y_hat"].parameters["sigma"].parameters["lam"])
    assert lam[0] == 2.0
    if degenerate_kind == "constant":
        # The healthy unit keeps its own data-derived rate.
        assert "treated_1" not in message
        np.testing.assert_allclose(lam[1], _expected_lam(df, tt, treated)[1])


@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_dimension_only_degenerate_scale_uses_index_label(fitter_cls):
    """Dimension-only outcomes report their generated treated-unit index."""
    df, tt, treated = _make_data([1.0])
    X, y = _pre_treatment_design(df, tt, treated)
    y = xr.DataArray(np.ones_like(y.values), dims=["obs_ind", "treated_units"])
    with pytest.warns(UserWarning, match="Cannot estimate the pre-treatment") as record:
        _fitter(fitter_cls).priors_from_data(X, y)
    assert "'0'" in str(record[0].message)


@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_user_y_hat_prior_bypasses_invalid_scale_calculation(fitter_cls):
    """An explicit likelihood prior wins before invalid default scales are read."""
    df, tt, treated = _make_data([1.0])
    df["treated_0"] = 5.0
    X, y = _pre_treatment_design(df, tt, treated)
    custom = Prior(
        "Normal",
        sigma=Prior("HalfNormal", sigma=42, dims=["treated_units"]),
        dims=["obs_ind", "treated_units"],
    )
    priors = _fitter(fitter_cls, priors={"y_hat": custom}).priors_from_data(X, y)
    assert "y_hat" not in priors


@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_subclass_default_y_hat_prior_is_not_auto_overridden(fitter_cls):
    """A subclass that declares its own noise default has already chosen one."""
    df, tt, treated = _make_data([1.0])
    X, y = _pre_treatment_design(df, tt, treated)
    custom = Prior(
        "Normal",
        sigma=Prior("HalfNormal", sigma=42, dims=["treated_units"]),
        dims=["obs_ind", "treated_units"],
    )
    custom_fitter = type(
        "CustomFitter",
        (fitter_cls,),
        {"default_priors": {"y_hat": custom}},
    )
    priors = custom_fitter().priors_from_data(X, y)
    assert "y_hat" not in priors


@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_subclass_inheriting_the_stock_default_is_auto_scaled(fitter_cls):
    """A subclass that leaves the noise default alone is still scaled."""
    df, tt, treated = _make_data([1.0])
    X, y = _pre_treatment_design(df, tt, treated)
    subclass = type("SubclassedFitter", (fitter_cls,), {})
    sigma = subclass().priors_from_data(X, y)["y_hat"].parameters["sigma"]
    assert sigma.distribution == "Exponential"
    np.testing.assert_allclose(
        np.asarray(sigma.parameters["lam"]), _expected_lam(df, tt, treated)
    )


@pytest.mark.parametrize("fitter_cls", FITTERS)
@pytest.mark.parametrize(
    ("scale", "falls_back"),
    [
        (np.finfo(float).tiny / 2, True),
        (np.finfo(float).max, False),
    ],
)
def test_finite_scale_rate_boundaries_are_checked(
    monkeypatch, fitter_cls, scale, falls_back
):
    """A subnormal spread overflows its rate and falls back; a huge one does not."""
    df, tt, treated = _make_data([1.0])
    X, y = _pre_treatment_design(df, tt, treated)
    monkeypatch.setattr(
        np,
        "std",
        lambda *_args, **_kwargs: np.array([scale]),
    )
    if falls_back:
        with pytest.warns(UserWarning, match="Cannot estimate the pre-treatment"):
            priors = _fitter(fitter_cls).priors_from_data(X, y)
        assert np.asarray(priors["y_hat"].parameters["sigma"].parameters["lam"]) == 2.0
    else:
        sigma = _fitter(fitter_cls).priors_from_data(X, y)["y_hat"].parameters["sigma"]
        rate = np.asarray(sigma.parameters["lam"])
        assert np.all(np.isfinite(rate))
        assert np.all(rate > 0)


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_opt_out_survives_refit_and_cloning(mock_pymc_sample, fitter_cls):
    """The opt-out is carried by the model, so later fits keep the legacy prior."""
    df, tt, treated = _make_data([1.0])
    result = cp.SyntheticControl(
        df,
        tt,
        control_units=["a", "b", "c"],
        treated_units=treated,
        model=_fitter(fitter_cls),
        auto_scale_sigma=False,
    )
    X, y = _pre_treatment_design(*_make_data([100.0]))
    refitted = result.model._clone()
    refitted.fit(X, y)
    sigma = refitted.priors["y_hat"].parameters["sigma"]
    assert sigma.distribution == "HalfNormal"
    assert sigma.parameters["sigma"] == 1


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_opt_out_leaves_the_callers_model_untouched(mock_pymc_sample, fitter_cls):
    """Opting out fits a copy, so the caller's own instance is not reconfigured."""
    df, tt, treated = _make_data([1.0])
    source_model = _fitter(fitter_cls)
    result = cp.SyntheticControl(
        df,
        tt,
        control_units=["a", "b", "c"],
        treated_units=treated,
        model=source_model,
        auto_scale_sigma=False,
    )
    assert result.model is not source_model
    assert source_model._user_priors is None
    assert source_model.idata is None
