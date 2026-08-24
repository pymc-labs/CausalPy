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
"""Lifecycle tests for the lazy experiment lifecycle (issue #1092).

Structural behavior (guards, predicates, return types, invalidation) runs
under cheap forward sampling. The prior path gets tiny real sampling because
genuine prior draws are required to verify canonical containers and
independent draw sizes. No seed-pinned numerical baselines: parity comes from
the unchanged integration suite.
"""

import numpy as np
import pandas as pd
import pytest

import causalpy as cp
from causalpy.custom_exceptions import (
    GroupNotSampledException,
    PriorPredictiveNotSupportedException,
)


@pytest.fixture(scope="module", autouse=True)
def real_pymc_sampling():
    """Undo the session-scoped ``mock_pymc_sample`` patch for this module.

    The conftest registers PyMC's ``mock_sample`` (prior sampling instead of
    MCMC) session-wide, so once any earlier test module requests it,
    ``pm.sample`` stays mocked for the rest of the run. Several tests here
    assert posterior-vs-prior relationships that are meaningless when the
    "posterior" is itself prior sampling — restore the real sampler for this
    module only.
    """
    import pymc as pm
    import pymc.sampling.mcmc

    patched = pm.sample
    pm.sample = pymc.sampling.mcmc.sample
    yield
    pm.sample = patched


TINY_SAMPLE_KWARGS = {
    "draws": 20,
    "tune": 20,
    "chains": 1,
    "progressbar": False,
    "random_seed": 42,
}
PRIOR_SAMPLE_KWARGS = {"draws": 30, "random_seed": 42}


def _make_its(data, **model_kwargs):
    """Build a lazy-configured ITS on synthetic simple data."""
    defaults = {"sample_kwargs": dict(TINY_SAMPLE_KWARGS)}
    defaults.update(model_kwargs)
    return cp.InterruptedTimeSeries(
        data,
        treatment_time=pd.Timestamp("2017-06-01"),
        formula="y ~ 1 + t",
        model=cp.pymc_models.LinearRegression(**defaults),
    )


# ----------------------------------------------------------------------------
# Structural behavior
# ----------------------------------------------------------------------------


@pytest.mark.integration
def test_constructor_is_lazy(its_data):
    """__init__ validates and builds design matrices but never samples."""
    exp = _make_its(its_data)
    assert exp.is_configured
    assert not exp.is_built
    assert not exp.is_fitted
    assert not exp.has_prior_predictive
    assert exp.idata is None


@pytest.mark.integration
def test_read_guards_name_the_missing_call(its_data):
    """Every group-requiring read fails fast naming the lifecycle verb."""
    exp = _make_its(its_data)

    with pytest.raises(GroupNotSampledException, match=r"fit\(\)"):
        _ = exp.result
    with pytest.raises(GroupNotSampledException, match=r"fit\(\)"):
        exp.plot(show=False)
    with pytest.raises(GroupNotSampledException, match=r"fit\(\)"):
        exp.effect_summary()
    with pytest.raises(GroupNotSampledException, match=r"sample_prior_predictive"):
        exp.plot(group="prior", show=False)
    with pytest.raises(GroupNotSampledException, match=r"sample_prior_predictive"):
        _ = exp.prior_result
    with pytest.raises(GroupNotSampledException, match=r"fit\(\)"):
        exp.print_coefficients()


@pytest.mark.integration
def test_build_is_public_idempotent_and_inspectable(its_data):
    """build() returns Self, constructs the graph, and is a no-op afterwards."""
    import pymc as pm

    exp = _make_its(its_data)
    out = exp.build()
    assert out is exp
    assert exp.is_built
    assert len(exp.model.basic_RVs) > 0
    # Graph inspection works without any draws existing
    graph = pm.model_to_graphviz(exp.model)
    assert graph.source
    assert exp.idata is None
    # Idempotent: second call is a no-op returning Self
    assert exp.build() is exp


@pytest.mark.integration
def test_fit_returns_self_and_populates_state(its_data):
    exp = _make_its(its_data)
    out = exp.fit()
    assert out is exp
    assert exp.is_fitted
    assert exp.has_prior_predictive  # fit fills the absent prior phase first
    assert exp.result is not None
    assert exp.prior_result is not None
    assert exp.idata is not None


@pytest.mark.integration
def test_sample_prior_predictive_returns_self(its_data):
    exp = _make_its(its_data)
    out = exp.sample_prior_predictive(**PRIOR_SAMPLE_KWARGS)
    assert out is exp
    assert exp.has_prior_predictive
    assert not exp.is_fitted
    # Posterior reads stay guarded after a prior-only phase
    with pytest.raises(GroupNotSampledException, match=r"fit\(\)"):
        exp.plot(show=False)


@pytest.mark.integration
def test_model_assignment_resets_all_state(its_data):
    exp = _make_its(its_data)
    exp.fit()
    assert exp.is_fitted and exp.has_prior_predictive

    exp.model = cp.pymc_models.LinearRegression(
        priors={"beta": [0, 0.5]},
        sample_kwargs=dict(TINY_SAMPLE_KWARGS),
    )
    assert exp.idata is None
    assert not exp.is_fitted
    assert not exp.has_prior_predictive
    assert not exp.is_built


@pytest.mark.integration
def test_second_fit_warns_and_preserves_prior(its_data):
    exp = _make_its(its_data, prior_sample_kwargs=dict(PRIOR_SAMPLE_KWARGS))
    exp.fit()
    prior_before = exp._prior_result
    n_prior_draws = exp.idata["prior"].sizes["draw"]

    with pytest.warns(UserWarning, match="previous posterior"):
        exp.fit()

    assert exp._prior_result is prior_before
    assert exp.idata["prior"].sizes["draw"] == n_prior_draws


@pytest.mark.integration
def test_prior_and_posterior_draw_sizes_are_independent(its_data):
    exp = _make_its(its_data, prior_sample_kwargs={"draws": 37, "random_seed": 7})
    exp.fit()
    assert exp.idata["prior"].sizes["draw"] == 37
    assert exp.idata["posterior"].sizes["draw"] == TINY_SAMPLE_KWARGS["draws"]


@pytest.mark.integration
def test_sklearn_backend_rejects_prior_phase(its_data):
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    exp = cp.InterruptedTimeSeries(
        its_data,
        treatment_time=pd.Timestamp("2017-06-01"),
        formula="y ~ 1 + t",
        model=SkLinearRegression(),
    )
    with pytest.raises(PriorPredictiveNotSupportedException, match="LinearRegression"):
        exp.sample_prior_predictive()
    out = exp.fit()
    assert out is exp
    assert exp.is_fitted
    assert not exp.has_prior_predictive
    fig, ax = exp.plot(show=False)
    assert fig is not None
    assert len(list(ax)) > 0


# ----------------------------------------------------------------------------
# Tiny real sampling — canonical containers and prior-phase behavior
# ----------------------------------------------------------------------------


@pytest.mark.integration
def test_prior_predictions_use_canonical_container(its_data):
    exp = _make_its(its_data, prior_sample_kwargs=dict(PRIOR_SAMPLE_KWARGS))
    exp.sample_prior_predictive()
    preds = exp.prior_result.predictions_post
    assert {"chain", "draw", "obs_ind", "treated_units"} <= set(preds.dims)
    assert preds.sizes["draw"] == PRIOR_SAMPLE_KWARGS["draws"]


@pytest.mark.integration
def test_prior_plot_has_reduced_layout(its_data):
    exp = _make_its(its_data, prior_sample_kwargs=dict(PRIOR_SAMPLE_KWARGS))
    exp.sample_prior_predictive()
    _, ax_prior = exp.plot(group="prior", show=False)
    assert len(np.atleast_1d(ax_prior)) < 3  # impact panels dropped

    exp.fit()
    _, ax_post = exp.plot(show=False)
    assert len(list(ax_post)) == 3


@pytest.mark.slow
@pytest.mark.integration
def test_posterior_hdi_narrower_than_prior_hdi(its_data):
    """Property: data-informed posterior contracts relative to the prior."""
    from causalpy._arviz_compat import hdi_bounds

    exp = _make_its(its_data, prior_sample_kwargs=dict(PRIOR_SAMPLE_KWARGS))
    exp.sample_prior_predictive()
    prior_lower, prior_upper = hdi_bounds(
        exp.prior_result.impact_post.mean("obs_ind"), prob=0.94
    )
    prior_width = float(np.asarray(prior_upper - prior_lower))
    exp.fit()
    post_lower, post_upper = hdi_bounds(
        exp.result.impact_post.mean("obs_ind"), prob=0.94
    )
    post_width = float(np.asarray(post_upper - post_lower))
    assert post_width < prior_width


@pytest.mark.slow
@pytest.mark.integration
def test_prior_effect_tail_probability_near_neutral(its_data):
    """Under a symmetric zero-mean prior, P(effect > 0) should sit near 0.5."""
    exp = _make_its(its_data, prior_sample_kwargs={"draws": 200, "random_seed": 3})
    exp.sample_prior_predictive()
    summary = exp.effect_summary(group="prior")
    table = summary.table
    prob_col = "p_gt_0" if "p_gt_0" in table.columns else table.columns[-1]
    prob = float(table[prob_col].iloc[0])
    assert 0.25 < prob < 0.75


class _MockTrendComponent:
    """Minimal trend component: zero contribution, valid apply()."""

    def apply(self, time_data):
        return time_data * 0


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: cp.pymc_models.LinearRegression(
            sample_kwargs={
                "draws": 20,
                "tune": 20,
                "chains": 1,
                "progressbar": False,
                "random_seed": 42,
            }
        ),
        # Exercises the BBETS build() override that records the extra
        # trend/seasonality pm.Data nodes for re-arming.
        lambda: cp.pymc_models.BayesianBasisExpansionTimeSeries(
            trend_component=_MockTrendComponent(),
            seasonality_component=_MockTrendComponent(),
            sample_kwargs={
                "draws": 10,
                "tune": 10,
                "chains": 1,
                "progressbar": False,
                "random_seed": 42,
            },
        ),
    ],
    ids=["linear-regression", "bbets"],
)
def test_refit_after_predict_restores_training_design(its_data, model_factory):
    """fit → plot (forecast-window conditioning) → fit must re-sample on the
    training design; guards the base X/y re-arm and, on the BBETS variant,
    the trend/seasonality node re-arm added for the round-1 review blocker.
    """
    import warnings

    exp = cp.InterruptedTimeSeries(
        its_data,
        treatment_time=pd.Timestamp("2017-06-01"),
        formula="y ~ 1 + t",
        model=model_factory(),
    )
    exp.fit()
    n_train = exp.idata["posterior"]["mu"].sizes["obs_ind"]

    # Forward sampling through plot() conditions the shared data nodes on
    # the forecast window. The recorded build-time values must still hold
    # training lengths, and a second documented no-op build() call must not
    # overwrite them with the poisoned live-node values.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp.plot(show=False)
    expected_shapes = {
        name: value.shape for name, value in exp.model._build_data_nodes.items()
    }
    for name, shape in expected_shapes.items():
        if name in ("X", "y", "t_trend_data", "t_season_data"):
            assert shape[0] == n_train, f"{name} recorded {shape}, not training"

    exp.model.build(
        exp.pre_design["X"],
        exp.pre_design["y"],
        coords=None,
    )
    assert {
        name: value.shape for name, value in exp.model._build_data_nodes.items()
    } == expected_shapes

    # The next fit re-arms from those recordings and samples on the
    # training design again.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp.fit()
    assert exp.idata["posterior"]["mu"].sizes["obs_ind"] == n_train


@pytest.mark.integration
def test_prior_after_fit_overwrites_prior_preserves_posterior(its_data):
    """sample_prior_predictive() after a completed fit replaces prior draws
    and the prior bundle while leaving posterior groups untouched."""
    exp = _make_its(its_data, prior_sample_kwargs={"draws": 25, "random_seed": 1})
    exp.fit()
    post_sizes = dict(exp.idata["posterior"]["mu"].sizes)
    post_draws = exp.idata["posterior"].sizes["draw"]
    prior_bundle = exp._prior_result

    exp.sample_prior_predictive(draws=40, random_seed=2)

    assert exp.idata["prior"].sizes["draw"] == 40
    assert dict(exp.idata["posterior"]["mu"].sizes) == post_sizes
    assert exp.idata["posterior"].sizes["draw"] == post_draws
    assert exp._prior_result is not prior_bundle  # recomputed
    assert exp.result is not None


@pytest.mark.integration
def test_exceptions_are_exported():
    """Guard exceptions are part of the public contract."""
    for name in (
        "BadIndexException",
        "DataException",
        "FormulaException",
        "GroupNotSampledException",
        "PriorPredictiveNotSupportedException",
    ):
        assert hasattr(cp, name), f"cp.{name} missing"


def test_rebuild_after_design_mutation_raises_instead_of_stale_graph():
    """exp.build() after the design data changed fails loudly.

    The graph is built exactly once per instance; silently keeping a stale
    graph while the inputs changed would be the worst of both worlds.
    ``_fit_inputs()`` reads the design dataset, so mutating it is the
    reachable path for post-build drift (mutating ``exp.data`` cannot reach
    an already-materialized design).
    """
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    rng = np.random.default_rng(3)
    n = 30
    df = pd.DataFrame({"t": np.arange(n), "y": rng.normal(size=n)})
    exp = cp.InterruptedTimeSeries(
        df,
        treatment_time=20,
        formula="y ~ 1 + t",
        model=SkLinearRegression(),
    )
    exp.fit()

    exp.pre_design["y"] = exp.pre_design["y"] * 1000.0
    with pytest.raises(RuntimeError, match="already built with different inputs"):
        exp.build()
