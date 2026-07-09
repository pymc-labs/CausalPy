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
Regression tests verifying that ``ci_prob`` (and its deprecated alias
``hdi_prob``) is wired through Bayesian plot methods so that user-supplied
values actually change the rendered credible-interval bands.

These tests guard against the regression described in GitHub issue
`pymc-labs/CausalPy#890`_, where ``result.plot(hdi_prob=...)`` was silently
swallowed by ``**kwargs`` rather than reaching the underlying
:func:`causalpy.plot_utils.plot_posterior_over_x` (and equivalent) calls.

``hdi_prob`` was the original parameter name. It was renamed to ``ci_prob``
when ETI support was added (since the parameter controls the *credible interval*
width, not only HDI). ``hdi_prob`` remains accepted with a ``FutureWarning``.

.. _pymc-labs/CausalPy#890: https://github.com/pymc-labs/CausalPy/issues/890
"""

import importlib
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression as SkLinearRegression

import causalpy as cp
from causalpy.constants import HDI_PROB
from causalpy.data.simulate_data import (
    generate_piecewise_its_data,
    generate_staggered_did_data,
)
from causalpy.tests.conftest import setup_regression_kink_data

sample_kwargs = {"tune": 20, "draws": 20, "chains": 2, "cores": 2}

# Each entry maps a "spy target" (dotted import path of the callable used by
# the experiment's plot path) to the kwarg name that ``hdi_prob`` flows into.
# ``plot_posterior_over_x`` targets use ``"ci_prob"`` (the canonical name); other callables
# such as ``az.plot_posterior`` and ``az.plot_forest`` use ``"hdi_prob"``.
# Migrated (plotnine) experiments thread ``ci_prob`` into ``td.point_interval``
# as ``probs=(ci_prob,)``; those targets add a third element that extracts the
# scalar from the recorded tuple.
_SpyTarget = tuple[str, str] | tuple[str, str, Callable[[Any], float | None]]


def _resolve_dotted(dotted: str) -> Any:
    """Resolve a dotted path that may include attribute access through an alias.

    For example, ``causalpy.experiments.prepostnegd.az.plot_posterior`` is not
    importable as a module path because ``az`` is an alias inside the module,
    not a real submodule. We import the longest valid module prefix and then
    walk the remaining attributes.
    """
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for part in parts[i:]:
            obj = getattr(obj, part)
        return obj
    raise ImportError(f"Could not resolve {dotted!r}")  # pragma: no cover


def _record_hdi_prob_calls(
    targets: list[_SpyTarget],
) -> tuple[ExitStack, list[float | None]]:
    """Patch each ``targets`` callable with a spy that records its ``hdi_prob`` kwarg.

    Returns the ``ExitStack`` (to be used as a context manager) and a list that
    will be populated with the recorded values. Each call appends one entry,
    or ``None`` when the kwarg was omitted.
    """
    recorded: list[float | None] = []
    stack = ExitStack()

    for target in targets:
        dotted, kwarg = target[0], target[1]
        transform = target[2] if len(target) > 2 else (lambda v: v)
        real = _resolve_dotted(dotted)

        def make_spy(real_callable=real, kwarg_name=kwarg, extract=transform):
            """Build a side_effect that records ``kwarg_name`` and forwards to ``real_callable``."""

            def spy(*args, **kwargs):
                """Record ``hdi_prob`` then delegate to the real callable."""
                value = kwargs.get(kwarg_name)
                recorded.append(value if value is None else extract(value))
                return real_callable(*args, **kwargs)

            return spy

        stack.enter_context(patch(dotted, side_effect=make_spy()))

    return stack, recorded


def _assert_threads(recorded: list[float | None], expected: float) -> None:
    """Assert that every recorded ``ci_prob`` value equals ``expected``."""
    assert len(recorded) > 0, "no spied call sites were exercised"
    assert all(value == expected for value in recorded), (
        f"Expected every spied call to receive hdi_prob={expected}, "
        f"but recorded values were {recorded}"
    )


# ---------------------------------------------------------------------------
# Module-scoped fitted experiment fixtures (each fits the model once).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_its(its_data):
    """Fit an InterruptedTimeSeries once for reuse across wiring tests."""
    return cp.InterruptedTimeSeries(
        its_data,
        pd.to_datetime("2017-01-01"),
        formula="y ~ 1 + t + C(month)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )


@pytest.fixture(scope="module")
def fitted_sc(sc_data):
    """Fit a SyntheticControl once for reuse across wiring tests."""
    return cp.SyntheticControl(
        sc_data,
        70,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=sample_kwargs),
    )


@pytest.fixture(scope="module")
def fitted_did(did_data):
    """Fit a DifferenceInDifferences once for reuse across wiring tests."""
    return cp.DifferenceInDifferences(
        did_data,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )


@pytest.fixture(scope="module")
def fitted_rd(rd_data):
    """Fit a RegressionDiscontinuity once for reuse across wiring tests."""
    return cp.RegressionDiscontinuity(
        rd_data,
        formula="y ~ 1 + bs(x, df=6) + treated",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        treatment_threshold=0.5,
        epsilon=0.001,
    )


@pytest.fixture(scope="module")
def fitted_rkink():
    """Fit a RegressionKink once for reuse across wiring tests."""
    kink = 0.5
    df = setup_regression_kink_data(kink)
    return cp.RegressionKink(
        df,
        formula=f"y ~ 1 + x + I((x-{kink})*treated)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
        kink_point=kink,
    )


@pytest.fixture(scope="module")
def fitted_prepost(anova1_data):
    """Fit a PrePostNEGD once for reuse across wiring tests."""
    return cp.PrePostNEGD(
        anova1_data,
        formula="post ~ 1 + C(group) + pre",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )


@pytest.fixture(scope="module")
def fitted_piecewise():
    """Fit a PiecewiseITS once for reuse across wiring tests."""
    df, _ = generate_piecewise_its_data(N=100, seed=42)
    return cp.PiecewiseITS(
        df,
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )


@pytest.fixture(scope="module")
def fitted_panel():
    """Fit a PanelRegression once for reuse across wiring tests."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    for u_idx in range(10):
        unit_effect = rng.normal()
        for t in range(20):
            treatment = 1 if (t >= 10 and u_idx < 5) else 0
            x1 = rng.normal()
            y = unit_effect + 0.1 * t + 2.0 * treatment + 0.5 * x1 + 0.1 * rng.normal()
            rows.append(
                {
                    "unit": f"u{u_idx}",
                    "time": t,
                    "treatment": treatment,
                    "x1": x1,
                    "y": y,
                }
            )
    df = pd.DataFrame(rows)
    return cp.PanelRegression(
        data=df,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )


@pytest.fixture(scope="module")
def fitted_staggered():
    """Fit a Bayesian StaggeredDifferenceInDifferences for the staggered tests."""
    df = generate_staggered_did_data(
        n_units=30, n_time_periods=15, treatment_cohorts={5: 10, 10: 10}, seed=42
    )
    return cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        treatment_time_variable_name="treatment_time",
        model=cp.pymc_models.LinearRegression(sample_kwargs=sample_kwargs),
    )


@pytest.fixture(scope="module")
def fitted_staggered_ols():
    """Fit an OLS StaggeredDifferenceInDifferences (kept for future use)."""
    df = generate_staggered_did_data(
        n_units=30, n_time_periods=15, treatment_cohorts={5: 10, 10: 10}, seed=42
    )
    return cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        treatment_time_variable_name="treatment_time",
        model=SkLinearRegression(),
    )


@pytest.fixture(scope="module")
def fitted_sdid():
    """Fit a SyntheticDifferenceInDifferences once for reuse across wiring tests."""
    df = cp.load_data("sc")
    return cp.SyntheticDifferenceInDifferences(
        df,
        70,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=cp.pymc_models.SyntheticDifferenceInDifferencesWeightFitter(
            sample_kwargs=sample_kwargs,
        ),
    )


# ---------------------------------------------------------------------------
# Per-class wiring tests. Each verifies that user-supplied ``hdi_prob`` reaches
# the underlying HDI primitives, and that the canonical default is forwarded
# when the parameter is omitted.
# ---------------------------------------------------------------------------


_PARAMS = [0.50, 0.75, 0.99]


def _check_threading(
    fitted,
    targets: list[_SpyTarget],
    ci_prob: float,
    plot_kwarg: str = "ci_prob",
) -> None:
    """Call ``fitted.plot(ci_prob=...)`` and assert ``ci_prob`` reaches every target.

    ``plot_kwarg`` controls the kwarg name used to call ``plot()`` — use
    ``"hdi_prob"`` for classes not yet migrated to ``ci_prob``.
    """
    stack, recorded = _record_hdi_prob_calls(targets)
    with stack:
        fitted.plot(**{plot_kwarg: ci_prob})
    _assert_threads(recorded, ci_prob)


def _check_default(
    fitted,
    targets: list[_SpyTarget],
) -> None:
    """Call ``fitted.plot()`` (no kwarg) and assert every target gets HDI_PROB or None."""
    stack, recorded = _record_hdi_prob_calls(targets)
    with stack:
        fitted.plot()
    # When the experiment forwards the default, the recorded values are
    # either ``HDI_PROB`` itself or ``None`` if the experiment relies on the
    # downstream callable's own default. Both are acceptable.
    assert len(recorded) > 0, "no spied call sites were exercised"
    assert all(value in (HDI_PROB, None) for value in recorded), (
        f"Expected every spied call to receive HDI_PROB ({HDI_PROB}) or None, "
        f"but recorded values were {recorded}"
    )


_ITS_TARGETS: list[_SpyTarget] = [
    ("causalpy.experiments.interrupted_time_series.plot_posterior_over_x", "ci_prob"),
]
_SC_TARGETS: list[_SpyTarget] = [
    ("causalpy.experiments.synthetic_control.plot_posterior_over_x", "ci_prob"),
]
_DID_TARGETS: list[_SpyTarget] = [
    (
        "causalpy.experiments.diff_in_diff.td.point_interval",
        "probs",
        lambda probs: probs[0] if probs else None,
    ),
]
_RD_TARGETS: list[_SpyTarget] = [
    (
        "causalpy.experiments.regression_discontinuity.td.point_interval",
        "probs",
        lambda probs: probs[0] if probs else None,
    ),
]
_RKINK_TARGETS: list[_SpyTarget] = [
    ("causalpy.experiments.regression_kink.plot_posterior_over_x", "ci_prob"),
]
_PREPOST_TARGETS: list[_SpyTarget] = [
    (
        "causalpy.experiments.prepostnegd.td.point_interval",
        "probs",
        lambda probs: probs[0] if probs else None,
    ),
]
_PIECEWISE_TARGETS: list[_SpyTarget] = [
    ("causalpy.experiments.piecewise_its.plot_posterior_over_x", "ci_prob"),
]
_PANEL_TARGETS: list[_SpyTarget] = [
    ("causalpy.experiments.panel_regression.az.plot_forest", "hdi_prob"),
]


@pytest.mark.integration
@pytest.mark.parametrize("ci_prob", _PARAMS)
def test_its_plot_threads_ci_prob(mock_pymc_sample, fitted_its, ci_prob):
    """ITS ``plot(ci_prob=...)`` reaches every ``plot_posterior_over_x`` call."""
    _check_threading(fitted_its, _ITS_TARGETS, ci_prob)


@pytest.mark.integration
def test_its_plot_default_ci_prob(mock_pymc_sample, fitted_its):
    """ITS default ``plot()`` forwards ``HDI_PROB`` to every ``plot_posterior_over_x`` call (as ``ci_prob``)."""
    _check_default(fitted_its, _ITS_TARGETS)


@pytest.mark.integration
@pytest.mark.parametrize("ci_prob", _PARAMS)
def test_sc_plot_threads_ci_prob(mock_pymc_sample, fitted_sc, ci_prob):
    """Synthetic Control ``plot(ci_prob=...)`` reaches every ``plot_posterior_over_x`` call."""
    _check_threading(fitted_sc, _SC_TARGETS, ci_prob)


@pytest.mark.integration
def test_sc_plot_default_ci_prob(mock_pymc_sample, fitted_sc):
    """Synthetic Control default ``plot()`` forwards ``HDI_PROB`` as ``ci_prob``."""
    _check_default(fitted_sc, _SC_TARGETS)


@pytest.mark.integration
@pytest.mark.parametrize("ci_prob", _PARAMS)
def test_did_plot_threads_ci_prob(mock_pymc_sample, fitted_did, ci_prob):
    """DiD ``plot(ci_prob=...)`` reaches ``td.point_interval``."""
    _check_threading(fitted_did, _DID_TARGETS, ci_prob)


@pytest.mark.integration
def test_did_plot_default_ci_prob(mock_pymc_sample, fitted_did):
    """DiD default ``plot()`` forwards ``HDI_PROB`` as ``ci_prob``."""
    _check_default(fitted_did, _DID_TARGETS)


@pytest.mark.integration
@pytest.mark.parametrize("ci_prob", _PARAMS)
def test_rd_plot_threads_ci_prob(mock_pymc_sample, fitted_rd, ci_prob):
    """RD ``plot(ci_prob=...)`` reaches ``td.point_interval``."""
    _check_threading(fitted_rd, _RD_TARGETS, ci_prob)


@pytest.mark.integration
def test_rd_plot_default_ci_prob(mock_pymc_sample, fitted_rd):
    """RD default ``plot()`` forwards ``HDI_PROB`` as ``ci_prob``."""
    _check_default(fitted_rd, _RD_TARGETS)


@pytest.mark.integration
@pytest.mark.parametrize("ci_prob", _PARAMS)
def test_rkink_plot_threads_ci_prob(mock_pymc_sample, fitted_rkink, ci_prob):
    """Regression Kink ``plot(ci_prob=...)`` reaches every ``plot_posterior_over_x`` call."""
    _check_threading(fitted_rkink, _RKINK_TARGETS, ci_prob)


@pytest.mark.integration
def test_rkink_plot_default_ci_prob(mock_pymc_sample, fitted_rkink):
    """Regression Kink default ``plot()`` forwards ``HDI_PROB`` as ``ci_prob``."""
    _check_default(fitted_rkink, _RKINK_TARGETS)


@pytest.mark.integration
@pytest.mark.parametrize("ci_prob", _PARAMS)
def test_prepost_plot_threads_ci_prob(mock_pymc_sample, fitted_prepost, ci_prob):
    """PrePostNEGD ``plot(ci_prob=...)`` reaches ``td.point_interval``."""
    _check_threading(fitted_prepost, _PREPOST_TARGETS, ci_prob)


@pytest.mark.integration
def test_prepost_plot_default_ci_prob(mock_pymc_sample, fitted_prepost):
    """PrePostNEGD default ``plot()`` forwards ``HDI_PROB`` as ``ci_prob``."""
    _check_default(fitted_prepost, _PREPOST_TARGETS)


@pytest.mark.integration
@pytest.mark.parametrize("ci_prob", _PARAMS)
def test_piecewise_plot_threads_ci_prob(mock_pymc_sample, fitted_piecewise, ci_prob):
    """PiecewiseITS ``plot(ci_prob=...)`` reaches every ``plot_posterior_over_x`` call."""
    _check_threading(fitted_piecewise, _PIECEWISE_TARGETS, ci_prob)


@pytest.mark.integration
def test_piecewise_plot_default_ci_prob(mock_pymc_sample, fitted_piecewise):
    """PiecewiseITS default ``plot()`` forwards ``HDI_PROB`` as ``ci_prob``."""
    _check_default(fitted_piecewise, _PIECEWISE_TARGETS)


@pytest.mark.integration
@pytest.mark.parametrize("ci_prob", _PARAMS)
def test_panel_plot_threads_ci_prob(mock_pymc_sample, fitted_panel, ci_prob):
    """PanelRegression ``plot(hdi_prob=...)`` reaches ``az.plot_forest``.

    PanelRegression has not yet been migrated to ``ci_prob`` (it does not
    support ETI/spaghetti/histogram), so we call ``plot(hdi_prob=...)`` here.
    """
    _check_threading(fitted_panel, _PANEL_TARGETS, ci_prob, plot_kwarg="hdi_prob")


@pytest.mark.integration
def test_panel_plot_default_ci_prob(mock_pymc_sample, fitted_panel):
    """PanelRegression default ``plot()`` forwards ``HDI_PROB`` to ``az.plot_forest``."""
    _check_default(fitted_panel, _PANEL_TARGETS)


# ---------------------------------------------------------------------------
# Deprecated hdi_prob alias: verify it still works with a FutureWarning.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_deprecated_hdi_prob_still_wired(mock_pymc_sample, fitted_its):
    """``plot(hdi_prob=...)`` still reaches ``plot_posterior_over_x`` via the deprecated alias.

    Ensures that existing user code does not silently break after the rename
    to ``ci_prob``. The deprecated alias must emit a ``FutureWarning`` and
    forward the value to ``plot_posterior_over_x`` as ``ci_prob``.
    """
    stack, recorded = _record_hdi_prob_calls(
        [
            (
                "causalpy.experiments.interrupted_time_series.plot_posterior_over_x",
                "ci_prob",
            )
        ]
    )
    import warnings

    with stack, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted_its.plot(hdi_prob=0.75)

    assert any(issubclass(w.category, FutureWarning) for w in caught), (
        "Expected a FutureWarning when hdi_prob is passed to plot()"
    )
    _assert_threads(recorded, 0.75)


@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture_name",
    [
        "fitted_did",
        "fitted_rd",
        "fitted_rkink",
        "fitted_prepost",
        "fitted_piecewise",
        "fitted_sc",
        "fitted_sdid",
    ],
)
def test_deprecated_hdi_prob_alias_emits_futurewarning(
    mock_pymc_sample, request, fixture_name
):
    """hdi_prob deprecated alias emits FutureWarning on all migrated experiment classes."""
    fitted = request.getfixturevalue(fixture_name)
    with pytest.warns(FutureWarning, match="hdi_prob is deprecated"):
        fitted.plot(hdi_prob=0.75)


# ---------------------------------------------------------------------------
# StaggeredDifferenceInDifferences has different semantics: HDI bounds are
# computed during effect aggregation (at fit time), not at plot time. The
# plot reads cached ``att_lower``/``att_upper`` columns. To prevent silent
# kwarg swallowing, ``_bayesian_plot`` raises a clear ``ValueError`` when the
# caller supplies an ``hdi_prob`` that does not match the cached value.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_staggered_plot_uses_cached_hdi_prob(mock_pymc_sample, fitted_staggered):
    """Default ``plot()`` and ``plot(hdi_prob=cached)`` both succeed."""
    fig1, _ = fitted_staggered.plot()
    fig2, _ = fitted_staggered.plot(hdi_prob=fitted_staggered.hdi_prob_)
    assert fig1 is not None
    assert fig2 is not None


@pytest.mark.integration
def test_staggered_plot_rejects_mismatched_hdi_prob(mock_pymc_sample, fitted_staggered):
    """Supplying a non-cached ``hdi_prob`` must raise rather than silently no-op."""
    other = 0.50 if fitted_staggered.hdi_prob_ != 0.50 else 0.99
    with pytest.raises(ValueError, match="HDI bounds are computed during"):
        fitted_staggered.plot(hdi_prob=other)


@pytest.mark.integration
def test_staggered_group_time_plot_rejects_mismatched_hdi_prob(
    mock_pymc_sample, fitted_staggered
):
    """The group-time plot must also reject a non-cached ``hdi_prob``."""
    other = 0.50 if fitted_staggered.hdi_prob_ != 0.50 else 0.99
    with pytest.raises(ValueError, match="HDI bounds are computed during"):
        fitted_staggered.plot_group_time(hdi_prob=other)
