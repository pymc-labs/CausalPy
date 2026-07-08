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
Unit tests for the private helper methods of ``SyntheticDifferenceInDifferences``.

The ``algorithm`` method is a thin orchestrator that delegates to a handful of
private helpers. Testing them directly lets us catch regressions in the
individual steps without paying the cost of a full MCMC run.
"""

from types import SimpleNamespace
from typing import Any, Protocol

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from causalpy.custom_exceptions import BadIndexException
from causalpy.experiments.synthetic_difference_in_differences import (
    SyntheticDifferenceInDifferences,
)


class _StubFittableModel(Protocol):
    """Minimal model protocol for SDiD stub tests."""

    idata: Any

    def fit(self, X: Any, y: Any, *, coords: Any = None) -> Any: ...


class _StubModelAdapter:
    """Minimal adapter for SDiD unit-test stubs that bypass ``__init__``."""

    def __init__(self, model: _StubFittableModel) -> None:
        self.model = model
        self.is_ols = False
        self.is_bayesian = False

    def fit(self, X: Any, y: Any, *, coords: Any | None = None) -> Any:
        return self.model.fit(X=X, y=y, coords=coords)


def _make_experiment_stub(
    *,
    control_units: list[str] | None = None,
    treated_units: list[str] | None = None,
    data: pd.DataFrame | None = None,
    treatment_time: int | None = None,
    model: _StubFittableModel | None = None,
) -> SyntheticDifferenceInDifferences:
    """Construct an SDiD instance without running ``__init__``/``algorithm``.

    The private helpers only rely on a handful of attributes. Bypassing
    ``__init__`` keeps these tests fast and side-effect free.
    """
    stub = object.__new__(SyntheticDifferenceInDifferences)
    stub.control_units = control_units or []
    stub.labels = list(stub.control_units)
    stub.treated_units = treated_units or []
    if data is not None:
        data.index.name = "obs_ind"
        stub.data = data
    if treatment_time is not None:
        stub.treatment_time = treatment_time
    if model is not None:
        stub.model = model
        stub._model_backend = _StubModelAdapter(model)  # type: ignore[assignment]
    return stub


@pytest.fixture
def toy_panel():
    """A deterministic 2-control, 5-period panel with treatment at t=3."""
    rng = np.random.default_rng(0)
    control_units = ["c0", "c1"]
    treated_units = ["t0"]
    T, T_pre = 5, 3
    data = pd.DataFrame(
        {
            "c0": rng.normal(size=T),
            "c1": rng.normal(size=T),
            "t0": rng.normal(size=T),
        }
    )
    return SimpleNamespace(
        data=data,
        control_units=control_units,
        treated_units=treated_units,
        treatment_time=T_pre,
        T=T,
        T_pre=T_pre,
    )


class TestBuildWeightFitterInputs:
    """Unit tests for ``_build_weight_fitter_inputs``."""

    def test_shapes_and_coords(self, toy_panel):
        stub = _make_experiment_stub(control_units=toy_panel.control_units)
        Y_co = toy_panel.data[toy_panel.control_units].to_numpy().T
        y_tr = toy_panel.data[toy_panel.treated_units].to_numpy().mean(axis=1)

        X, y, coords = stub._build_weight_fitter_inputs(Y_co, y_tr, toy_panel.T_pre)

        assert set(X.keys()) == {"unit", "time"}
        assert set(y.keys()) == {"unit", "time"}

        # Unit module: design is (T_pre, N_co), response is (T_pre,).
        assert X["unit"].dims == ("obs_ind", "coeffs")
        assert X["unit"].shape == (toy_panel.T_pre, len(toy_panel.control_units))
        assert y["unit"].dims == ("obs_ind",)
        assert y["unit"].shape == (toy_panel.T_pre,)

        # Time module: design is (N_co, T_pre), response is (N_co,).
        assert X["time"].dims == ("coeffs", "obs_ind")
        assert X["time"].shape == (len(toy_panel.control_units), toy_panel.T_pre)
        assert y["time"].dims == ("coeffs",)
        assert y["time"].shape == (len(toy_panel.control_units),)

        assert coords["coeffs"] == toy_panel.control_units
        assert list(coords["obs_ind"]) == list(range(toy_panel.T_pre))
        assert coords["coeffs_raw"] == toy_panel.control_units[1:]
        assert coords["obs_ind_raw"] == list(range(1, toy_panel.T_pre))

    def test_values_match_hand_computed(self, toy_panel):
        """``X_time`` must equal the pre-period control panel and
        ``y_time`` the post-period control mean."""
        stub = _make_experiment_stub(control_units=toy_panel.control_units)
        Y_co = toy_panel.data[toy_panel.control_units].to_numpy().T
        y_tr = toy_panel.data[toy_panel.treated_units].to_numpy().mean(axis=1)

        X, y, _ = stub._build_weight_fitter_inputs(Y_co, y_tr, toy_panel.T_pre)

        np.testing.assert_allclose(X["unit"].to_numpy(), Y_co[:, : toy_panel.T_pre].T)
        np.testing.assert_allclose(y["unit"].to_numpy(), y_tr[: toy_panel.T_pre])
        np.testing.assert_allclose(X["time"].to_numpy(), Y_co[:, : toy_panel.T_pre])
        np.testing.assert_allclose(
            y["time"].to_numpy(), Y_co[:, toy_panel.T_pre :].mean(axis=1)
        )


class TestExtractWeightPosteriors:
    """Unit tests for ``_extract_weight_posteriors``."""

    def test_returns_expected_arrays_and_counts(self):
        n_chains, n_draws, n_co, T_pre = 2, 4, 3, 5
        omega_true = np.arange(n_chains * n_draws * n_co, dtype=float).reshape(
            n_chains, n_draws, n_co
        )
        lam_true = np.arange(n_chains * n_draws * T_pre, dtype=float).reshape(
            n_chains, n_draws, T_pre
        )
        omega0_true = np.arange(n_chains * n_draws, dtype=float).reshape(
            n_chains, n_draws
        )
        posterior = xr.Dataset(
            {
                "omega": (("chain", "draw", "coeffs"), omega_true),
                "lam": (("chain", "draw", "obs_ind"), lam_true),
                "omega0": (("chain", "draw"), omega0_true),
            }
        )
        idata = az.InferenceData(posterior=posterior)
        stub = _make_experiment_stub(model=SimpleNamespace(idata=idata))

        omega, omega0, lam, n_ch, n_dr = stub._extract_weight_posteriors()

        np.testing.assert_array_equal(omega, omega_true)
        np.testing.assert_array_equal(lam, lam_true)
        np.testing.assert_array_equal(omega0, omega0_true)
        assert (n_ch, n_dr) == (n_chains, n_draws)


class TestComputeSyntheticAndGaps:
    """Unit tests for the static ``_compute_synthetic_and_gaps`` helper."""

    def test_matches_explicit_formula(self):
        n_chains, n_draws, n_co, T = 2, 3, 4, 6
        rng = np.random.default_rng(1)
        omega = rng.normal(size=(n_chains, n_draws, n_co))
        omega0 = rng.normal(size=(n_chains, n_draws))
        Y_co = rng.normal(size=(n_co, T))
        y_tr = rng.normal(size=(T,))

        sc_all, gaps = SyntheticDifferenceInDifferences._compute_synthetic_and_gaps(
            omega, omega0, Y_co, y_tr
        )

        assert sc_all.shape == (n_chains, n_draws, T)
        assert gaps.shape == (n_chains, n_draws, T)

        expected_sc = np.empty((n_chains, n_draws, T))
        for c in range(n_chains):
            for d in range(n_draws):
                expected_sc[c, d, :] = omega0[c, d] + omega[c, d] @ Y_co
        np.testing.assert_allclose(sc_all, expected_sc)
        np.testing.assert_allclose(gaps, y_tr[np.newaxis, np.newaxis, :] - expected_sc)


class TestComputeTau:
    """Unit tests for the static ``_compute_tau`` helper."""

    def test_double_difference_formula(self):
        n_chains, n_draws, T_pre, T_post = 2, 3, 4, 3
        T = T_pre + T_post
        rng = np.random.default_rng(2)
        gaps = rng.normal(size=(n_chains, n_draws, T))
        lam = rng.normal(size=(n_chains, n_draws, T_pre))

        tau = SyntheticDifferenceInDifferences._compute_tau(
            gaps, lam, T_pre, n_chains, n_draws
        )

        assert isinstance(tau, xr.DataArray)
        assert tau.dims == ("chain", "draw")
        assert tau.shape == (n_chains, n_draws)

        expected = gaps[..., T_pre:].mean(axis=-1) - (lam * gaps[..., :T_pre]).sum(
            axis=-1
        )
        np.testing.assert_allclose(tau.to_numpy(), expected)

    def test_tau_is_zero_when_gaps_are_zero(self):
        n_chains, n_draws, T_pre, T_post = 2, 3, 4, 5
        T = T_pre + T_post
        gaps = np.zeros((n_chains, n_draws, T))
        lam = np.random.default_rng(3).normal(size=(n_chains, n_draws, T_pre))

        tau = SyntheticDifferenceInDifferences._compute_tau(
            gaps, lam, T_pre, n_chains, n_draws
        )
        np.testing.assert_allclose(tau.to_numpy(), 0.0)


class TestBuildReportingObjects:
    """Unit tests for ``_build_reporting_objects``."""

    def test_sets_expected_attributes(self, toy_panel):
        stub = _make_experiment_stub(
            control_units=toy_panel.control_units,
            treated_units=toy_panel.treated_units,
            data=toy_panel.data,
            treatment_time=toy_panel.treatment_time,
        )

        n_chains, n_draws = 2, 3
        sc_all = np.random.default_rng(4).normal(size=(n_chains, n_draws, toy_panel.T))

        stub._build_reporting_objects(sc_all, toy_panel.T_pre, n_chains, n_draws)

        # pre_pred / post_pred are InferenceData with a 'mu' variable in
        # the posterior_predictive group.
        assert isinstance(stub.pre_pred, az.InferenceData)
        assert isinstance(stub.post_pred, az.InferenceData)
        assert "mu" in stub.pre_pred.posterior_predictive
        assert "mu" in stub.post_pred.posterior_predictive

        # pre_impact / post_impact are xr.DataArrays with the correct dims.
        for impact, expected_len in (
            (stub.pre_impact, toy_panel.T_pre),
            (stub.post_impact, toy_panel.T - toy_panel.T_pre),
        ):
            assert isinstance(impact, xr.DataArray)
            assert impact.dims == ("chain", "draw", "obs_ind", "treated_units")
            assert impact.shape == (n_chains, n_draws, expected_len, 1)

        # cumulative is cumsum of post_impact along time.
        np.testing.assert_allclose(
            stub.post_impact_cumulative.to_numpy(),
            stub.post_impact.cumsum(dim="obs_ind").to_numpy(),
        )

    def test_impact_values_match_observed_minus_counterfactual(self, toy_panel):
        stub = _make_experiment_stub(
            control_units=toy_panel.control_units,
            treated_units=toy_panel.treated_units,
            data=toy_panel.data,
            treatment_time=toy_panel.treatment_time,
        )
        n_chains, n_draws = 1, 2
        sc_all = np.random.default_rng(5).normal(size=(n_chains, n_draws, toy_panel.T))

        stub._build_reporting_objects(sc_all, toy_panel.T_pre, n_chains, n_draws)

        y_tr_pre = (
            toy_panel.data.iloc[: toy_panel.T_pre][toy_panel.treated_units]
            .to_numpy()
            .mean(axis=1)
        )
        y_tr_post = (
            toy_panel.data.iloc[toy_panel.T_pre :][toy_panel.treated_units]
            .to_numpy()
            .mean(axis=1)
        )

        expected_pre = (
            y_tr_pre[np.newaxis, np.newaxis, :] - sc_all[..., : toy_panel.T_pre]
        )
        expected_post = (
            y_tr_post[np.newaxis, np.newaxis, :] - sc_all[..., toy_panel.T_pre :]
        )

        np.testing.assert_allclose(stub.pre_impact.to_numpy()[..., 0], expected_pre)
        np.testing.assert_allclose(stub.post_impact.to_numpy()[..., 0], expected_post)


class TestInputValidation:
    """Both ``BadIndexException`` branches in ``input_validation``."""

    def test_datetime_index_requires_timestamp_treatment_time(self):
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0]},
            index=pd.date_range("2020-01-01", periods=3),
        )
        stub = _make_experiment_stub()
        with pytest.raises(BadIndexException, match="DatetimeIndex"):
            stub.input_validation(df, treatment_time=2)

    def test_int_index_rejects_timestamp_treatment_time(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        stub = _make_experiment_stub()
        with pytest.raises(BadIndexException, match="DatetimeIndex"):
            stub.input_validation(df, treatment_time=pd.Timestamp("2020-01-01"))


class TestErrorBranches:
    """Error paths raised when the underlying model has no ``idata``."""

    def test_extract_weight_posteriors_raises_when_idata_is_none(self):
        stub = _make_experiment_stub(model=SimpleNamespace(idata=None))
        with pytest.raises(RuntimeError, match="Model has not been fit"):
            stub._extract_weight_posteriors()

    def test_algorithm_raises_when_fit_leaves_idata_none(self, toy_panel):
        class _NoIdataModel:
            """Mimics a PyMCModel whose ``fit`` fails to populate ``idata``."""

            idata = None

            def fit(self, X, y, coords):
                return None

        stub = _make_experiment_stub(
            control_units=toy_panel.control_units,
            treated_units=toy_panel.treated_units,
            data=toy_panel.data,
            treatment_time=toy_panel.treatment_time,
            model=_NoIdataModel(),
        )
        with pytest.raises(AttributeError, match="failed to produce idata"):
            stub.algorithm()


class TestSummaryMultiTreated:
    """``summary`` prints the list when more than one treated unit is present."""

    def test_multi_treated_branch(self, capsys):
        stub = _make_experiment_stub(
            control_units=["c0"],
            treated_units=["t0", "t1"],
        )
        stub.expt_type = "SyntheticDifferenceInDifferences"
        stub.tau_posterior = xr.DataArray(
            np.array([[1.0, 1.5], [0.5, 2.0]]),
            dims=["chain", "draw"],
        )

        stub.summary()

        captured = capsys.readouterr().out
        assert "Treated units: ['t0', 't1']" in captured
        assert "Treated unit:" not in captured


class TestConvertTreatmentTimeForAxis:
    """``_convert_treatment_time_for_axis`` falls back when conversion fails."""

    def test_returns_input_when_convert_units_raises_typeerror(self):
        class _XAxis:
            @staticmethod
            def convert_units(_value):
                raise TypeError("cannot convert")

        axis = SimpleNamespace(xaxis=_XAxis())
        result = SyntheticDifferenceInDifferences._convert_treatment_time_for_axis(
            axis, 42
        )
        assert result == 42

    def test_returns_input_when_convert_units_raises_valueerror(self):
        class _XAxis:
            @staticmethod
            def convert_units(_value):
                raise ValueError("bad value")

        axis = SimpleNamespace(xaxis=_XAxis())
        ts = pd.Timestamp("2020-01-01")
        result = SyntheticDifferenceInDifferences._convert_treatment_time_for_axis(
            axis, ts
        )
        assert result == ts
