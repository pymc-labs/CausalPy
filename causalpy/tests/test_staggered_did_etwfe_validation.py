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
Ground-truth validation suite for the ETWFE staggered DiD estimator.

This module answers one question: *does the estimator recover the effects that
generated the data?* It is deliberately separate from
``test_staggered_did_etwfe.py``, which tests plumbing, schemas and error paths.

The suite is organised as four tiers of increasing cost and decreasing
sharpness.

Tier 1
    Exact algebraic recovery on noise-free data via the sklearn/OLS path. With
    a correctly specified saturated design and ``sigma=0`` the residuals vanish
    and the coefficients are unique given full rank, so every assertion is an
    ``atol=1e-8`` equality rather than a tolerance band. This is the
    load-bearing proof of the whole feature and it runs in milliseconds. Any
    error in ``ev_idx``, ``cohort_idx``, ``effect_indicator``, top-binning or
    the ATT weight matrix surfaces here as a large deviation.

Tier 2
    Statistical recovery under noise, still on the OLS path, plus the
    naive-TWFE contrast that *is* the motivation for the estimator.

Tier 3
    Real MCMC. Confirms the Bayesian implementation is calibrated: the true ATT
    falls inside the posterior HDI and the chains converge.

Tier 4
    Cross-estimator agreement. Fits the same panel with real MCMC and with OLS
    and asserts they agree. This is the only mechanism that transfers Tier 1's
    algebraic exactness onto the Bayesian path.

Tiers 3 and 4 must **not** use the ``mock_pymc_sample`` fixture -- it replaces
``pm.sample`` with prior predictive sampling, which is precisely the failure
mode the other tiers exist to cover. Because that fixture is *session* scoped
in ``conftest.py``, its patch survives for the rest of the session once any
earlier test has requested it, so these tests take the ``real_pymc_sample``
fixture below to restore genuine sampling explicitly.
"""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
from patsy import dmatrices
from pymc_extras.prior import Prior
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

import causalpy as cp
from causalpy.data.simulate_data import generate_staggered_did_data
from causalpy.pymc_models import ETWFERegression

# Common experiment arguments. The canonical released formula is used verbatim:
# the ETWFE path takes only its left-hand side, which is the point.
BASE_KWARGS = {
    "formula": "y ~ 1 + C(unit) + C(time)",
    "unit_variable_name": "unit",
    "time_variable_name": "time",
    "treated_variable_name": "treated",
    "treatment_time_variable_name": "treatment_time",
}

# Exact-equality tolerance for the noise-free OLS path.
EXACT = 1e-8


# ===========================================================================
# Helpers and fixtures
# ===========================================================================


def make_noise_free_panel(**overrides) -> pd.DataFrame:
    """Noise-free panel with cohort-heterogeneous, growing treatment effects.

    ``sigma=0`` makes the generator satisfy ``y == y0 + tau`` exactly, so the
    frame doubles as an algebraic ground truth. Effects grow in event time and
    the later cohort's profile is scaled by 1.6, which is exactly the setting in
    which a single two-way fixed effects coefficient is biased.

    Parameters
    ----------
    **overrides
        Keyword arguments forwarded to
        :func:`causalpy.data.simulate_data.generate_staggered_did_data`,
        overriding the defaults below.

    Returns
    -------
    pd.DataFrame
        Panel with ``unit, time, treated, treatment_time, y, y0, tau`` columns.
    """
    defaults: dict = {
        "n_units": 24,
        "n_time_periods": 12,
        # Cohort sizes are deliberately UNEQUAL. With equal sizes every treated
        # (g, k) cell holds the same number of observations, the ATT weight
        # matrix is uniform, and a weighted average is numerically identical to
        # an unweighted one -- so the exactness tests below would pass even if
        # the aggregation dropped the weights entirely. 10 vs 6 makes the
        # weights non-uniform, and an unweighted aggregation then misses the
        # true ATT by ~0.016, far outside the 1e-8 tolerance these tests use.
        "treatment_cohorts": {4: 10, 8: 6},
        "treatment_effects": lambda k: 1 + 0.4 * k,
        "cohort_effect_scale": {4: 1.0, 8: 1.6},
        "sigma": 0.0,
        "seed": 7,
    }
    defaults.update(overrides)
    return generate_staggered_did_data(**defaults)


def make_noisy_panel(**overrides) -> pd.DataFrame:
    """Larger, noisy panel with the same cohort-heterogeneous effect structure.

    Parameters
    ----------
    **overrides
        Keyword arguments forwarded to
        :func:`causalpy.data.simulate_data.generate_staggered_did_data`.

    Returns
    -------
    pd.DataFrame
        Panel data.
    """
    defaults: dict = {
        "n_units": 40,
        "n_time_periods": 16,
        # Unequal for the same reason as the noise-free panel: uniform ATT
        # weights would make weighted and unweighted aggregation indistinguishable.
        "treatment_cohorts": {5: 14, 10: 10},
        "treatment_effects": lambda k: 1 + 0.4 * k,
        "cohort_effect_scale": {5: 1.0, 10: 1.8},
        "sigma": 0.1,
        "seed": 11,
    }
    defaults.update(overrides)
    return generate_staggered_did_data(**defaults)


def make_mcmc_panel(**overrides) -> pd.DataFrame:
    """Small panel sized for the real-MCMC tiers.

    Kept deliberately small: Tiers 3 and 4 run on every pull request because
    ``.github/workflows/ci.yml`` invokes ``pytest`` with no ``-m`` deselection.

    Parameters
    ----------
    **overrides
        Keyword arguments forwarded to
        :func:`causalpy.data.simulate_data.generate_staggered_did_data`.

    Returns
    -------
    pd.DataFrame
        Panel data.
    """
    defaults: dict = {
        # 5 units per cohort, not 4: five observations per (g, k) cell is the
        # threshold below which the identification check warns about thin cells,
        # and that warning is already covered by its own test elsewhere.
        "n_units": 14,
        "n_time_periods": 10,
        "treatment_cohorts": {3: 5, 6: 5},
        "treatment_effects": lambda k: 1 + 0.4 * k,
        "cohort_effect_scale": {3: 1.0, 6: 1.5},
        "sigma": 0.3,
        "seed": 42,
    }
    defaults.update(overrides)
    return generate_staggered_did_data(**defaults)


def true_att(df: pd.DataFrame) -> float:
    """Average-over-the-treated ATT implied by the generator.

    ``tau`` is stored per observation, so the mean over treated rows is exactly
    the ``<W, T>`` estimand the ETWFE model targets.

    Parameters
    ----------
    df : pd.DataFrame
        Panel produced by the generator.

    Returns
    -------
    float
        The true ATT.
    """
    return float(df.loc[df["treated"] == 1, "tau"].mean())


def true_tau_surface(df: pd.DataFrame) -> pd.Series:
    """True ``tau[g, k]`` for every treated cell, indexed by ``(cohort, k)``.

    Parameters
    ----------
    df : pd.DataFrame
        Panel produced by the generator.

    Returns
    -------
    pd.Series
        Series indexed by ``(treatment_time, event_time)``.
    """
    treated = df.loc[df["treated"] == 1].copy()
    treated["k"] = (treated["time"] - treated["treatment_time"]).astype(int)
    return treated.groupby(["treatment_time", "k"])["tau"].mean()


def fit_etwfe_ols(df: pd.DataFrame, **kwargs):
    """Fit the ETWFE experiment through the plain-OLS path.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    **kwargs
        Extra experiment keyword arguments, e.g. ``n_leads`` or ``covariates``.

    Returns
    -------
    causalpy.experiments.staggered_did.StaggeredDifferenceInDifferences
        The fitted experiment.
    """
    return cp.StaggeredDifferenceInDifferences(
        df,
        model=SklearnLinearRegression(),
        estimator="etwfe",
        **{**BASE_KWARGS, **kwargs},
    )


def naive_twfe_delta(df: pd.DataFrame) -> float:
    """Single-coefficient two-way fixed effects estimate of the ATT.

    Fits ``y ~ 1 + C(unit) + C(time) + treated`` by least squares and returns
    the coefficient on ``treated``. This is the estimator ETWFE replaces, and
    with cohort-heterogeneous, growing effects it is the one that is biased.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.

    Returns
    -------
    float
        The single-delta TWFE coefficient.
    """
    y, X = dmatrices("y ~ 1 + C(unit) + C(time) + treated", df)
    coefs, *_ = np.linalg.lstsq(np.asarray(X), np.asarray(y), rcond=None)
    position = X.design_info.column_names.index("treated")
    return float(np.ravel(coefs)[position])


@pytest.fixture(scope="module")
def noise_free_df() -> pd.DataFrame:
    """The canonical noise-free Tier 1 panel."""
    return make_noise_free_panel()


@pytest.fixture(scope="module")
def ols_fit(noise_free_df):
    """A default ETWFE OLS fit (no leads, no covariates) on noise-free data."""
    return fit_etwfe_ols(noise_free_df)


@pytest.fixture
def real_pymc_sample():
    """Force genuine MCMC, undoing any active ``mock_pymc_sample`` patch.

    ``conftest.mock_pymc_sample`` is *session* scoped and swaps ``pm.sample``
    for prior predictive sampling, restoring it only at the end of the session.
    Once any earlier test in the run has requested it, every later test would
    silently sample from the prior. The MCMC tiers below would then pass
    vacuously, which defeats their entire purpose, so they take this fixture to
    pin the genuine sampler for their duration.

    Yields
    ------
    None
    """
    from pymc.distributions.continuous import Flat, HalfFlat
    from pymc.sampling.mcmc import sample as genuine_sample

    saved = (pm.sample, pm.Flat, pm.HalfFlat)
    pm.sample, pm.Flat, pm.HalfFlat = genuine_sample, Flat, HalfFlat
    try:
        yield
    finally:
        pm.sample, pm.Flat, pm.HalfFlat = saved


# ===========================================================================
# Tier 1 -- exact algebraic recovery, noise-free, sklearn
# ===========================================================================


def test_tier1_tau_surface_matches_generator_cell_by_cell(noise_free_df, ols_fit):
    """Every estimated ``tau[g, k]`` equals the generator's value exactly.

    If this fails the saturated design is mis-specified or an index array is
    wrong: ``cohort_idx`` mapping rows to the wrong cohort column, ``ev_idx``
    off by one, or the patsy ``_gk_cell`` labels not lining up with the
    coefficient positions. Under ``sigma=0`` OLS is exact, so any deviation
    above 1e-8 is structural, not statistical.
    """
    truth = true_tau_surface(noise_free_df)
    surface = ols_fit.tau_surface_

    assert len(surface) == len(truth)
    for _, row in surface.iterrows():
        expected = truth.loc[(row["cohort"], row["event_time"])]
        assert row["att"] == pytest.approx(expected, abs=EXACT)


def test_tier1_att_equals_treated_cell_mean_tau(noise_free_df, ols_fit):
    """``att_`` reproduces ``df.loc[treated, "tau"].mean()`` exactly.

    This is the headline estimand. A failure means the ATT weight matrix ``W``
    is not the average-over-the-treated weighting -- for example weights
    normalised over the wrong set of cells, or lead columns leaking non-zero
    weight -- even if every individual cell coefficient is correct.
    """
    assert isinstance(ols_fit.att_, float)
    assert ols_fit.att_ == pytest.approx(true_att(noise_free_df), abs=EXACT)


def test_tier1_lead_coefficients_are_exactly_zero(noise_free_df):
    """With no anticipation and no noise, every estimated lead is exactly zero.

    Failure here means lead cells are entering the design incorrectly: an
    ``ev_idx`` that aliases a lead onto a post-treatment column, an
    ``effect_indicator`` that marks the reference period as in-scope, or a
    reference event time that has not been pinned by omission. Adding leads
    must also leave the ATT untouched, since ``W`` is built from treated cells
    only.
    """
    result = fit_etwfe_ols(noise_free_df, n_leads=3)

    leads = result.tau_surface_[result.tau_surface_["event_time"] < 0]
    assert len(leads) > 0
    assert set(leads["event_time"]) == {-2, -3}  # reference -1 is omitted
    np.testing.assert_allclose(leads["att"].to_numpy(), 0.0, atol=EXACT)

    assert result.att_ == pytest.approx(true_att(noise_free_df), abs=EXACT)


def test_tier1_exact_with_additive_covariates():
    """Exactness survives additive covariates, and their coefficients recover.

    A failure means the covariate columns are being appended to the design in
    the wrong place, or their contribution is being absorbed into the effect
    surface. The generator folds ``1.5 * x1 - 0.8 * x2`` into ``y0``, so with
    ``sigma=0`` those coefficients must come back exactly too.
    """
    df = make_noise_free_panel(n_covariates=2, covariate_coefs=[1.5, -0.8])
    result = fit_etwfe_ols(df, covariates=["x1", "x2"])

    assert result.att_ == pytest.approx(true_att(df), abs=EXACT)

    truth = true_tau_surface(df)
    for _, row in result.tau_surface_.iterrows():
        expected = truth.loc[(row["cohort"], row["event_time"])]
        assert row["att"] == pytest.approx(expected, abs=EXACT)

    coefs = dict(zip(result.labels, result._etwfe_coefs, strict=True))
    assert coefs["x1"] == pytest.approx(1.5, abs=EXACT)
    assert coefs["x2"] == pytest.approx(-0.8, abs=EXACT)


def test_tier1_exact_on_unbalanced_panel():
    """Dropping ~10% of rows at random does not disturb index alignment.

    This is the test that catches index-alignment bugs. Every index array is
    built by position over the observation frame; anything that silently
    assumes a balanced panel, a contiguous ``RangeIndex``, or that
    ``groupby`` order matches row order will produce a badly wrong surface
    here while passing on the balanced panel above.
    """
    df = make_noise_free_panel()
    unbalanced = df.sample(frac=0.9, random_state=3).sort_values(["unit", "time"])

    for frame in (unbalanced.reset_index(drop=True), unbalanced):
        result = fit_etwfe_ols(frame)
        assert result.att_ == pytest.approx(true_att(frame), abs=EXACT)

        truth = true_tau_surface(frame)
        for _, row in result.tau_surface_.iterrows():
            expected = truth.loc[(row["cohort"], row["event_time"])]
            assert row["att"] == pytest.approx(expected, abs=EXACT)


def test_tier1_event_time_atts_reconcile_with_the_surface(noise_free_df, ols_fit):
    """Post rows of ``att_event_time_`` are the cell-count-weighted surface.

    Two reconciliations in one: against the estimator's own ``tau_surface_``
    (so the aggregation weights are internally consistent) and against the
    generator (so they are also *correct*). A failure means
    ``_etwfe_column_weights`` is normalising over the wrong denominator, which
    would bias the event-study plot without touching the scalar ``att_``.
    """
    surface = ols_fit.tau_surface_
    df = noise_free_df
    treated = df.loc[df["treated"] == 1].copy()
    treated["k"] = (treated["time"] - treated["treatment_time"]).astype(int)

    post = ols_fit.att_event_time_[ols_fit.att_event_time_["event_time"] >= 0]
    assert len(post) > 0

    for _, row in post.iterrows():
        k = int(row["event_time"])
        cells = surface[surface["event_time"] == k]
        weighted = float((cells["att"] * cells["n_obs"]).sum() / cells["n_obs"].sum())
        assert row["att"] == pytest.approx(weighted, abs=EXACT)

        observed = treated.loc[treated["k"] == k, "tau"].mean()
        assert row["att"] == pytest.approx(observed, abs=EXACT)
        assert int(row["n_obs"]) == int((treated["k"] == k).sum())


def test_tier1_att_weights_are_treated_cell_shares(noise_free_df, ols_fit):
    """``att_weights_`` is exactly the treated-cell share matrix ``N_gk / N``.

    A failure means the weights are not the average-over-the-treated weighting
    the ATT claims to use, which would make ``att_`` a different estimand than
    the one documented even when it happens to be close numerically.
    """
    df = noise_free_df
    treated = df.loc[df["treated"] == 1].copy()
    treated["k"] = (treated["time"] - treated["treatment_time"]).astype(int)
    shares = treated.groupby(["treatment_time", "k"]).size() / len(treated)

    weights = ols_fit.att_weights_
    assert weights.to_numpy().sum() == pytest.approx(1.0, abs=EXACT)

    # Guard the *discriminating power* of every exactness test on this panel:
    # if the cohort sizes ever became equal the weight matrix would go uniform,
    # and a weighted aggregation would be numerically identical to an unweighted
    # one -- so those tests would still pass with the weights dropped entirely.
    nonzero = weights.to_numpy()[weights.to_numpy() > 0]
    assert len(np.unique(np.round(nonzero, 12))) > 1, (
        "ATT weights are uniform on this panel, so the exactness tests can no "
        "longer distinguish weighted from unweighted aggregation. Restore "
        "unequal cohort sizes in make_noise_free_panel()."
    )
    for (cohort, k), share in shares.items():
        assert weights.loc[cohort, k] == pytest.approx(share, abs=EXACT)

    # Every cell with no treated observation behind it carries exactly zero
    # weight, so no unpopulated cell can contribute to the estimand.
    populated = set(shares.index)
    for cohort in weights.index:
        for k in weights.columns:
            if (cohort, k) not in populated:
                assert weights.loc[cohort, k] == 0.0


# ===========================================================================
# Tier 2 -- noisy recovery and the naive-TWFE contrast, sklearn
# ===========================================================================


@pytest.mark.integration
def test_tier2_recovers_att_under_noise():
    """The OLS ETWFE point estimate and its cluster SE cover the truth.

    Exactness is gone once ``sigma > 0``, so this is the calibration check: the
    point estimate must land close to the truth and the reported standard error
    must be honest enough that a two-standard-error band contains it. A failure
    here with Tier 1 passing points at the variance-covariance code
    (``_etwfe_ols_vcov``) rather than at the design.
    """
    df = make_noisy_panel()
    result = fit_etwfe_ols(df)
    truth = true_att(df)

    assert abs(result.att_ - truth) < 0.15
    assert result.att_se_ > 0.0
    assert result.att_ - 2 * result.att_se_ <= truth <= result.att_ + 2 * result.att_se_


@pytest.mark.integration
def test_tier2_etwfe_beats_naive_single_delta_twfe():
    """ETWFE is closer to the truth than single-delta TWFE. This is the motivation.

    On cohort-heterogeneous, growing-effect data the single ``treated``
    coefficient is contaminated by already-treated units acting as controls
    (the Goodman-Bacon decomposition's "bad comparisons"). If this assertion
    ever fails, either the estimator has stopped disaggregating by cohort and
    event time, or the panel has lost the heterogeneity that makes the
    correction necessary -- and the feature has no reason to exist.
    """
    df = make_noisy_panel()
    truth = true_att(df)

    etwfe_error = abs(fit_etwfe_ols(df).att_ - truth)
    twfe_error = abs(naive_twfe_delta(df) - truth)

    # A comfortable margin: the gap on this design is large, so a small change
    # in seed or panel size cannot flip the comparison.
    assert twfe_error > etwfe_error + 0.15


# ===========================================================================
# Tier 3 -- real MCMC calibration
# ===========================================================================


# NOTE ON THE ``slow`` MARKER (applies to Tier 3 and Tier 4 below).
#
# These two are the ONLY tests anywhere in the suite that run a real sampler
# against the in-model ``att`` deterministic. Every other Bayesian test uses the
# ``mock_pymc_sample`` fixture, which replaces ``pm.sample`` with prior
# predictive draws and therefore cannot check recovery at all.
#
# CI currently runs ``pytest`` with no ``-m`` deselection, so they execute on
# every PR (~23s combined, measured). If a future change adds
# ``-m "not slow"``, these drop out and the Bayesian path loses its only
# genuine numerical coverage -- Tier 1 would still prove the OLS path
# algebraically, but nothing would tie that guarantee to the sampler. If the
# runtime ever needs cutting, shrink the panels rather than deselecting.
@pytest.mark.slow
def test_tier3_bayesian_posterior_covers_true_att(real_pymc_sample):
    """Real MCMC: the true ATT sits inside the 94% HDI and the chains converge.

    This is the only tier that exercises the actual sampler on the in-model
    ``att`` deterministic. A failure means either the posterior is
    mis-centred -- the Mundlak conditioning or the partial pooling of ``tau``
    biasing the aggregate -- or the geometry is bad enough that the sampler
    cannot explore it, which would make every interval this estimator reports
    untrustworthy.
    """
    df = make_mcmc_panel()
    result = cp.StaggeredDifferenceInDifferences(
        df,
        model=ETWFERegression(
            sample_kwargs={
                "tune": 400,
                "draws": 400,
                "chains": 2,
                "cores": 2,
                "target_accept": 0.9,
                "random_seed": 42,
                "progressbar": False,
            }
        ),
        estimator="etwfe",
        conditioning="mundlak",
        **BASE_KWARGS,
    )

    # Guard against a mocked sampler having produced these draws.
    assert "sample_stats" in result.model.idata

    truth = true_att(df)
    # Flattened deliberately: arviz reads a 2-D array as (draw, shape), not
    # (chain, draw), so pooling the chains into one dimension is unambiguous.
    draws = np.asarray(result.att_).ravel()
    lower, upper = az.hdi(draws, hdi_prob=0.94)
    assert lower <= truth <= upper

    rhat = float(az.rhat(result.model.idata, var_names=["att"])["att"])
    assert rhat < 1.05


# ===========================================================================
# Tier 4 -- cross-estimator agreement
# ===========================================================================


@pytest.mark.slow
def test_tier4_bayesian_agrees_with_ols_on_the_same_panel(real_pymc_sample):
    """Real MCMC with diffuse priors reproduces the OLS ETWFE estimate.

    **This is the only mechanism that transfers Tier 1's algebraic exactness
    onto the Bayesian path.** Tier 1 proves the OLS design recovers the
    generator to machine precision, but it says nothing about the PyMC model,
    which builds its own linear predictor from index arrays. Pinning the
    Bayesian posterior mean to the OLS point estimate -- with priors made
    deliberately diffuse so shrinkage cannot explain a gap -- is what carries
    that guarantee across. A failure means the two paths disagree about what
    the model *is*: a different linear predictor, a different effect indicator,
    or a different ATT weighting.

    ``conditioning="dummy"`` is used deliberately: free unit intercepts and a
    fixed-scale ``beta_t`` are the Bayesian analogue of the OLS design, so the
    two are estimating the same thing and any residual gap is real.
    """
    df = make_mcmc_panel()
    sd_y = float(df["y"].std())

    ols = fit_etwfe_ols(df)

    diffuse = {
        "alpha_dummy": Prior("Normal", mu=0, sigma=10 * sd_y, dims="units"),
        "beta_t_dummy": Prior("ZeroSumNormal", sigma=10 * sd_y, dims="periods"),
        "tau_bar": Prior("Normal", mu=0, sigma=10 * sd_y, dims="ev"),
        "sd_dev": Prior("HalfNormal", sigma=5 * sd_y),
    }
    bayes = cp.StaggeredDifferenceInDifferences(
        df,
        model=ETWFERegression(
            sample_kwargs={
                "tune": 400,
                "draws": 400,
                "chains": 2,
                "cores": 2,
                "target_accept": 0.9,
                "random_seed": 42,
                "progressbar": False,
            },
            priors=diffuse,
        ),
        estimator="etwfe",
        conditioning="dummy",
        **BASE_KWARGS,
    )

    assert "sample_stats" in bayes.model.idata

    gap = abs(float(np.asarray(bayes.att_).mean()) - ols.att_)
    assert gap < 0.1 * sd_y
