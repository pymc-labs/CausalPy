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

The feature replaces ``WeightedSumFitter``'s default ``sigma ~ HalfNormal(1)``
prior with ``sigma ~ Exponential(2/s)``, where *s* is the pre-treatment standard
deviation of the treated data, computed *per treated unit*.

These assertions inspect the model's ``y_hat`` prior, which ``algorithm()`` sets
before fitting, so they rely on ``mock_pymc_sample`` and never need real MCMC.
"""

import numpy as np
import pandas as pd
import pytest
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


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_auto_scale_sets_exponential_prior_matching_2_over_s(
    mock_pymc_sample, fitter_cls
):
    """After fit, the y_hat sigma prior is Exponential with lam = 2/s."""
    df, tt, treated = _make_data([1.0])
    result = cp.SyntheticControl(
        df,
        tt,
        control_units=["a", "b", "c"],
        treated_units=treated,
        model=_fitter(fitter_cls),
    )
    sigma = _sigma_prior(result)
    assert sigma.distribution == "Exponential"
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
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("fitter_cls", FITTERS)
def test_constant_pre_treatment_series_raises(mock_pymc_sample, fitter_cls):
    """A zero/undefined per-unit std raises a clear error rather than producing
    an infinite Exponential rate."""
    df, tt, treated = _make_data([1.0, 1.0])
    df["treated_0"] = 5.0  # constant pre-treatment -> std == 0 for this unit
    with pytest.raises(ValueError, match="auto-scale the sigma prior"):
        cp.SyntheticControl(
            df,
            tt,
            control_units=["a", "b", "c"],
            treated_units=treated,
            model=_fitter(fitter_cls),
        )
