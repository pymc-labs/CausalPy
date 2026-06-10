#   Copyright 2025 - 2026 The PyMC Labs Developers
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
Tests for uncovered conditional logic in time series models.

This test file focuses on code coverage for edge cases and error handling
in BayesianBasisExpansionTimeSeries and StateSpaceTimeSeries.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import causalpy as cp


class MockComponent:
    """Mock component with apply method for testing custom components."""

    def apply(self, time_data):
        return time_data * 0


class MockComponentNoApply:
    """Mock component without apply method to test validation."""

    pass


class TestBayesianBasisExpansionTimeSeriesCoverage:
    """Test uncovered branches in BayesianBasisExpansionTimeSeries."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_pymc_marketing(self):
        """Skip entire class when pymc-marketing not installed (needed for default BSTS components)."""
        pytest.importorskip(
            "pymc_marketing",
            reason="pymc-marketing optional for default BSTS components",
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range(start="2020-01-01", end="2020-03-01", freq="D")
        n_obs = len(dates)
        y_values = np.random.randn(n_obs)

        X_da = xr.DataArray(
            np.zeros((n_obs, 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates, "coeffs": []},
        )
        y_da = xr.DataArray(
            y_values.reshape(-1, 1),
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": dates, "treated_units": ["unit_0"]},
        )
        return X_da, y_da

    def test_custom_trend_component_without_apply_method(self):
        """Test validation error when custom trend component lacks apply method."""
        with pytest.raises(
            ValueError,
            match="Custom trend_component must have an 'apply' method",
        ):
            cp.pymc_models.BayesianBasisExpansionTimeSeries(
                trend_component=MockComponentNoApply(),
                sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            )

    def test_custom_seasonality_component_without_apply_method(self):
        """Test validation error when custom seasonality component lacks apply method."""
        with pytest.raises(
            ValueError,
            match="Custom seasonality_component must have an 'apply' method",
        ):
            cp.pymc_models.BayesianBasisExpansionTimeSeries(
                seasonality_component=MockComponentNoApply(),
                sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            )

    def test_custom_components_with_apply_method(self, sample_data):
        """Test that custom components with apply method work."""
        X_da, y_da = sample_data

        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            trend_component=MockComponent(),
            seasonality_component=MockComponent(),
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )

        # Should not raise
        idata = model.fit(X_da, y_da)
        assert idata is not None

    def test_prepare_time_features_none_x(self):
        """Test error when X is None in _prepare_time_and_exog_features."""
        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        with pytest.raises(ValueError, match="X cannot be None"):
            model._prepare_time_and_exog_features(None)

    def test_prepare_time_features_not_xarray(self):
        """Test error when X is not an xarray DataArray."""
        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        with pytest.raises(TypeError, match="X must be an xarray DataArray"):
            model._prepare_time_and_exog_features(np.array([[1, 2, 3]]))

    def test_prepare_time_features_no_obs_ind_coord(self):
        """Test error when X lacks obs_ind coordinate."""
        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        X_bad = xr.DataArray(np.zeros((10, 0)), dims=["time", "coeffs"])

        with pytest.raises(ValueError, match="X must have 'obs_ind' coordinate"):
            model._prepare_time_and_exog_features(X_bad)

    def test_prepare_time_features_empty_obs_ind(self):
        """Test error when X has empty obs_ind."""
        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        X_bad = xr.DataArray(
            np.zeros((0, 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": [], "coeffs": []},
        )

        with pytest.raises(ValueError, match="X must have at least one observation"):
            model._prepare_time_and_exog_features(X_bad)

    def test_prepare_time_features_non_datetime_obs_ind(self):
        """Test error when obs_ind doesn't contain datetime values."""
        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        X_bad = xr.DataArray(
            np.zeros((10, 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": np.arange(10), "coeffs": []},
        )

        with pytest.raises(
            ValueError,
            match="X.coords\\['obs_ind'\\] must contain datetime values",
        ):
            model._prepare_time_and_exog_features(X_bad)

    def test_data_setter_error_x_mismatch(self, sample_data):
        """Test error when X exog var names don't match between fit and predict."""
        X_da, y_da = sample_data

        # Fit model without exogenous variables (empty X)
        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False}
        )
        model.fit(X_da, y_da)

        # Create X with exogenous variables for prediction
        dates_new = pd.date_range(start="2020-03-02", end="2020-03-10", freq="D")
        X_with_exog = xr.DataArray(
            np.random.randn(len(dates_new), 1),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates_new, "coeffs": ["x1"]},
        )

        # Should raise error about mismatch (model fit with [], trying to predict with ["x1"])
        with pytest.raises(
            ValueError,
            match="Exogenous variable names mismatch",
        ):
            model.predict(X_with_exog)

    def test_data_setter_error_missing_exog_vars(self, sample_data):
        """Test error when model expects exog vars but prediction X doesn't provide them."""
        X_da, y_da = sample_data
        dates = X_da.coords["obs_ind"].values

        # Create X with exogenous variables for fitting
        X_with_exog = xr.DataArray(
            np.random.randn(len(dates), 1),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates, "coeffs": ["x1"]},
        )

        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False}
        )
        model.fit(X_with_exog, y_da)

        # Try to predict with empty X
        dates_new = pd.date_range(start="2020-03-02", end="2020-03-10", freq="D")
        X_empty = xr.DataArray(
            np.zeros((len(dates_new), 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates_new, "coeffs": []},
        )

        with pytest.raises(
            ValueError,
            match="Model was built with exogenous variables",
        ):
            model.predict(X_empty)


class TestBayesianBasisExpansionTimeSeriesDataTree:
    """DataTree migration contracts for BayesianBasisExpansionTimeSeries.

    Uses MockComponent so these tests do not depend on pymc-marketing default
    components (which may be unavailable or incompatible in the pymc 6 env).
    """

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range(start="2020-01-01", end="2020-03-01", freq="D")
        n_obs = len(dates)
        y_values = np.random.default_rng(0).normal(size=n_obs)

        X_da = xr.DataArray(
            np.zeros((n_obs, 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates, "coeffs": []},
        )
        y_da = xr.DataArray(
            y_values.reshape(-1, 1),
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": dates, "treated_units": ["unit_0"]},
        )
        return X_da, y_da

    def test_fit_returns_datatree_with_expected_groups(
        self, sample_data, mock_pymc_sample
    ):
        """fit() returns a DataTree with posterior / prior / predictive groups."""
        X_da, y_da = sample_data
        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            trend_component=MockComponent(),
            seasonality_component=MockComponent(),
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )

        idata = model.fit(X_da, y_da)

        assert isinstance(idata, xr.DataTree)
        assert "posterior" in idata
        assert "prior_predictive" in idata
        assert "posterior_predictive" in idata

    def test_predict_preserves_datetime_obs_ind(self, sample_data, mock_pymc_sample):
        """predict() keeps the exact datetime obs_ind coordinates from X."""
        X_da, y_da = sample_data
        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            trend_component=MockComponent(),
            seasonality_component=MockComponent(),
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )
        model.fit(X_da, y_da)

        pred = model.predict(X_da)

        assert isinstance(pred, xr.DataTree)
        assert "posterior_predictive" in pred
        obs_ind = pred["posterior_predictive"]["mu"].coords["obs_ind"]
        np.testing.assert_array_equal(
            obs_ind.values.astype("datetime64[ns]"),
            X_da.coords["obs_ind"].values.astype("datetime64[ns]"),
        )


class TestStateSpaceTimeSeriesCoverage:
    """Test uncovered branches in StateSpaceTimeSeries."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range(start="2020-01-01", end="2020-02-01", freq="D")
        n_obs = len(dates)
        y_values = np.random.default_rng(0).normal(size=n_obs) + 10

        y_da = xr.DataArray(
            y_values.reshape(-1, 1),
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": dates, "treated_units": ["unit_0"]},
        )
        return y_da

    def test_custom_trend_component_wrong_type(self):
        """Test validation error when custom trend component is not a
        statespace component."""
        with pytest.raises(
            ValueError,
            match="Custom trend_component must be a pymc-extras structural",
        ):
            cp.pymc_models.StateSpaceTimeSeries(
                trend_component=MockComponentNoApply(),
                sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            )

    def test_custom_seasonality_component_wrong_type(self):
        """Test validation error when custom seasonality component is not a
        statespace component."""
        with pytest.raises(
            ValueError,
            match="Custom seasonality_component must be a pymc-extras structural",
        ):
            cp.pymc_models.StateSpaceTimeSeries(
                seasonality_component=MockComponentNoApply(),
                sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            )

    def test_custom_statespace_components_accepted(self, sample_data):
        """Real pymc-extras structural components are valid custom components."""
        from pymc_extras.statespace import structural as st

        model = cp.pymc_models.StateSpaceTimeSeries(
            trend_component=st.LevelTrend(order=1),
            seasonality_component=st.FrequencySeasonality(season_length=7, name="freq"),
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
        )
        model.build_model(y=sample_data)

        assert "initial_level_trend" in (rv.name for rv in model.free_RVs)

    def test_component_base_class_moved_upstream(self, monkeypatch):
        """A moved upstream base class gives an actionable error."""
        import sys

        from pymc_extras.statespace import structural as st

        component = st.LevelTrend(order=1)
        monkeypatch.setitem(
            sys.modules, "pymc_extras.statespace.models.structural.core", None
        )

        with pytest.raises(ImportError, match="does not expose it at that path"):
            cp.pymc_models.StateSpaceTimeSeries(
                trend_component=component,
                sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            )

    def test_backwards_compatibility_coords_datetime_index(self, sample_data):
        """Test backwards compatibility with coords['datetime_index']."""
        y_da = sample_data
        dates = pd.DatetimeIndex(y_da.coords["obs_ind"].values)

        # Create y with integer obs_ind (old API)
        y_old_api = xr.DataArray(
            y_da.values,
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": np.arange(len(dates)), "treated_units": ["unit_0"]},
        )

        # Pass datetime via coords dict
        coords = {"datetime_index": dates}

        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )

        # Should not raise - uses backwards compatibility path
        idata = model.fit(y=y_old_api, coords=coords)
        assert idata is not None

    def test_coords_datetime_index_not_datetimeindex(self, sample_data):
        """Test error when coords['datetime_index'] is not a DatetimeIndex."""
        y_da = sample_data
        n_obs = len(y_da)

        # Create y with integer obs_ind
        y_old_api = xr.DataArray(
            y_da.values,
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
        )

        # Pass non-DatetimeIndex via coords dict
        coords = {"datetime_index": np.arange(n_obs)}  # Not a DatetimeIndex!

        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )

        with pytest.raises(
            ValueError,
            match="coords\\['datetime_index'\\] must be a pd.DatetimeIndex",
        ):
            model.fit(y=y_old_api, coords=coords)

    def test_build_model_y_none(self):
        """Test error when y is None in build_model."""
        model = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        with pytest.raises(
            ValueError,
            match="y must be provided for StateSpaceTimeSeries.build_model",
        ):
            model.build_model(X=None, y=None)

    def test_build_model_y_no_obs_ind(self):
        """Test error when y lacks obs_ind coordinate."""
        model = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        y_bad = xr.DataArray(
            np.random.randn(10, 1),
            dims=["time", "treated_units"],
            coords={"time": np.arange(10), "treated_units": ["unit_0"]},
        )

        with pytest.raises(ValueError, match="y must have 'obs_ind' coordinate"):
            model.build_model(y=y_bad)

    def test_build_model_y_empty_obs_ind(self):
        """Test error when y has empty obs_ind."""
        model = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        y_bad = xr.DataArray(
            np.zeros((0, 1)),
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": [], "treated_units": ["unit_0"]},
        )

        with pytest.raises(ValueError, match="y must have at least one observation"):
            model.build_model(y=y_bad)

    def test_build_model_rejects_multiple_treated_units(self, sample_data):
        """State-space models reject unsupported multi-unit outcomes."""
        y_multi = xr.DataArray(
            np.repeat(sample_data.values, 2, axis=1),
            dims=["obs_ind", "treated_units"],
            coords={
                "obs_ind": sample_data.coords["obs_ind"],
                "treated_units": ["first", "second"],
            },
        )
        model = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        with pytest.raises(
            ValueError, match="supports exactly one treated unit, got 2"
        ):
            model.build_model(y=y_multi)

    def test_build_model_rejects_empty_treated_units(self, sample_data):
        """State-space models reject empty treated-unit dimensions clearly."""
        y_empty = xr.DataArray(
            np.empty((sample_data.sizes["obs_ind"], 0)),
            dims=["obs_ind", "treated_units"],
            coords={
                "obs_ind": sample_data.coords["obs_ind"],
                "treated_units": [],
            },
        )
        model = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        with pytest.raises(
            ValueError, match="supports exactly one treated unit, got 0"
        ):
            model.build_model(y=y_empty)

    def test_build_model_requires_treated_units_dimension(self, sample_data):
        """State-space models reject 1D outcomes before score can fail."""
        y_1d = xr.DataArray(
            sample_data.values[:, 0],
            dims=["obs_ind"],
            coords={"obs_ind": sample_data.coords["obs_ind"]},
        )
        model = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        with pytest.raises(ValueError, match="requires a treated_units dimension"):
            model.build_model(y=y_1d)

    def test_fit_and_score_support_unlabeled_single_treated_unit(
        self, sample_data, mock_pymc_sample
    ):
        """A singleton treated-unit dimension uses its xarray index label."""
        y_unlabeled = xr.DataArray(
            sample_data.values,
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": sample_data.coords["obs_ind"]},
        )
        X = xr.DataArray(
            np.zeros((y_unlabeled.sizes["obs_ind"], 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": y_unlabeled.coords["obs_ind"], "coeffs": []},
        )
        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )

        prediction = model.fit(X=X, y=y_unlabeled)
        score = model.score(X=X, y=y_unlabeled)

        assert prediction["posterior_predictive"]["y_hat"].coords[
            "treated_units"
        ].values.tolist() == [0]
        assert "unit_0_r2" in score

    def test_fit_y_none(self):
        """Test error when y is None in fit."""
        model = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        with pytest.raises(
            ValueError,
            match="y must be provided for StateSpaceTimeSeries.fit",
        ):
            model.fit(y=None)

    def test_fit_predict_datatree_uses_obs_ind_not_time(
        self, sample_data, mock_pymc_sample
    ):
        """fit/predict return DataTree with y_hat/mu on obs_ind (not time).

        Also checks OOS predict preserves the exact supplied datetime obs_ind.
        """
        y_da = sample_data.assign_coords(treated_units=["treated"])
        dates = y_da.coords["obs_ind"].values
        dummy_X = xr.DataArray(
            np.zeros((len(dates), 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates, "coeffs": []},
        )
        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )

        idata = model.fit(X=dummy_X, y=y_da)
        assert isinstance(idata, xr.DataTree)
        assert "posterior_predictive" in idata
        assert model.idata is idata
        fitted_pp = model.idata["posterior_predictive"]
        assert "y_hat" in fitted_pp and "mu" in fitted_pp
        assert set(fitted_pp.to_dataset().data_vars) == {"y_hat", "mu"}
        assert "obs_ind" in fitted_pp["y_hat"].dims
        np.testing.assert_array_equal(
            fitted_pp["y_hat"].coords["treated_units"].values,
            y_da.coords["treated_units"].values,
        )
        assert fitted_pp["y_hat"].dims == (
            "chain",
            "draw",
            "obs_ind",
            "treated_units",
        )
        np.testing.assert_array_equal(
            fitted_pp["y_hat"].coords["obs_ind"].values, dates
        )

        in_sample = model.predict(X=dummy_X, out_of_sample=False)
        assert isinstance(in_sample, xr.DataTree)
        in_pp = in_sample["posterior_predictive"]
        assert "y_hat" in in_pp and "mu" in in_pp
        assert set(in_pp.to_dataset().data_vars) == {"y_hat", "mu"}
        assert "obs_ind" in in_pp["y_hat"].dims
        assert "time" not in in_pp["y_hat"].dims
        assert "obs_ind" in in_pp["mu"].dims
        assert "time" not in in_pp["mu"].dims
        np.testing.assert_array_equal(
            in_pp["y_hat"].coords["treated_units"].values,
            y_da.coords["treated_units"].values,
        )
        assert in_pp["y_hat"].dims == (
            "chain",
            "draw",
            "obs_ind",
            "treated_units",
        )
        np.testing.assert_array_equal(in_pp["y_hat"].coords["obs_ind"].values, dates)

        oos_dates = pd.date_range(
            start=pd.Timestamp(dates[-1]) + pd.Timedelta(days=1),
            periods=5,
            freq="D",
        )
        X_oos = xr.DataArray(
            np.zeros((len(oos_dates), 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": oos_dates, "coeffs": []},
        )
        oos = model.predict(X=X_oos, out_of_sample=True)
        assert isinstance(oos, xr.DataTree)
        oos_pp = oos["posterior_predictive"]
        assert "y_hat" in oos_pp and "mu" in oos_pp
        assert set(oos_pp.to_dataset().data_vars) == {"y_hat", "mu"}
        assert "obs_ind" in oos_pp["y_hat"].dims
        assert "time" not in oos_pp["y_hat"].dims
        np.testing.assert_array_equal(
            oos_pp["y_hat"].coords["obs_ind"].values.astype("datetime64[ns]"),
            oos_dates.values.astype("datetime64[ns]"),
        )
        np.testing.assert_array_equal(
            oos_pp["y_hat"].coords["treated_units"].values,
            y_da.coords["treated_units"].values,
        )
        assert oos_pp["y_hat"].dims == (
            "chain",
            "draw",
            "obs_ind",
            "treated_units",
        )

    def test_missing_y_values_handled(self, sample_data):
        """The Kalman filter handles NaN in y; predictions stay finite."""
        y_da = sample_data.copy()
        y_da[5, 0] = np.nan
        y_da[12, 0] = np.nan

        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={
                "draws": 10,
                "tune": 10,
                "chains": 1,
                "progressbar": False,
            },
        )
        idata = model.fit(X=None, y=y_da)

        y_hat = idata.posterior_predictive["y_hat"]
        assert np.isfinite(y_hat.isel(obs_ind=[5, 12]).values).all()

    def test_short_series(self):
        """A short series (n=25) fits and predicts in sample."""
        dates = pd.date_range(start="2020-01-01", periods=25, freq="D")
        y_da = xr.DataArray(
            (10 + np.random.randn(25)).reshape(-1, 1),
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": dates, "treated_units": ["unit_0"]},
        )
        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={
                "draws": 10,
                "tune": 10,
                "chains": 1,
                "progressbar": False,
            },
        )
        model.fit(X=None, y=y_da)
        pred = model.predict(X=None)
        assert pred.posterior_predictive["y_hat"].sizes["obs_ind"] == 25

    def test_seasonal_length_below_two_raises(self):
        """seasonal_length=1 is rejected with a clear error."""
        with pytest.raises(ValueError, match="seasonal_length must be at least 2"):
            cp.pymc_models.StateSpaceTimeSeries(
                seasonal_length=1,
                sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            )

    def test_integer_index_without_datetime_raises(self):
        """Integer obs_ind without a datetime_index fallback is rejected."""
        y_int = xr.DataArray(
            np.random.randn(30, 1),
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": np.arange(30), "treated_units": ["unit_0"]},
        )
        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
        )
        with pytest.raises(ValueError, match="must contain datetime values"):
            model.build_model(y=y_int)

    def test_forecast_index_mismatch_warns(self, sample_data):
        """Out-of-sample dates that break the training frequency warn."""
        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={
                "draws": 10,
                "tune": 10,
                "chains": 1,
                "progressbar": False,
            },
        )
        model.fit(X=None, y=sample_data)

        last = pd.Timestamp(sample_data.coords["obs_ind"].values[-1])
        gapped = pd.date_range(last + pd.Timedelta(days=4), periods=5, freq="D")
        X_gapped = xr.DataArray(
            np.zeros((5, 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": gapped, "coeffs": []},
        )
        with pytest.warns(UserWarning, match="relabeled onto X's dates"):
            model.predict(X=X_gapped, out_of_sample=True)

    def test_default_model_free_rvs(self, sample_data):
        """Lock the default model contract: same five RVs as before the
        priors refactor."""
        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
        )
        model.build_model(y=sample_data)

        assert sorted(rv.name for rv in model.free_RVs) == [
            "P0_diag",
            "initial_level_trend",
            "params_freq",
            "sigma_freq",
            "sigma_level_trend",
        ]

    def test_user_priors_override(self, sample_data):
        """Test that user-supplied priors replace the defaults."""
        from pymc_extras.prior import Prior

        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            priors={"sigma_freq": Prior("HalfNormal", sigma=1)},
        )
        model.build_model(y=sample_data)

        assert model.priors["sigma_freq"].distribution == "HalfNormal"
        assert model["sigma_freq"].owner.op.name == "halfnormal"
        # Untouched defaults still apply
        assert model["sigma_level_trend"].owner.op.name == "gamma"

    def test_extract_exog_names(self):
        """Intercept handling: dropped silently when alone, with a warning
        when other covariates remain."""
        import warnings

        model = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        def make_X(names):
            return xr.DataArray(
                np.zeros((5, len(names))),
                dims=["obs_ind", "coeffs"],
                coords={"obs_ind": np.arange(5), "coeffs": names},
            )

        assert model._extract_exog_names(None) == []

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # intercept-only must not warn
            assert model._extract_exog_names(make_X(["Intercept"])) == []
            assert model._extract_exog_names(make_X(["x1", "x2"])) == ["x1", "x2"]

        with pytest.warns(UserWarning, match="Dropping the 'Intercept' column"):
            assert model._extract_exog_names(make_X(["Intercept", "x1"])) == ["x1"]

    def test_predict_missing_exog_columns(self, sample_data):
        """Predicting out of sample without the fit-time covariates raises."""
        y_da = sample_data
        n = len(y_da)
        X = xr.DataArray(
            np.random.randn(n, 1),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": y_da.coords["obs_ind"], "coeffs": ["x1"]},
        )

        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={
                "draws": 10,
                "tune": 10,
                "chains": 1,
                "progressbar": False,
            },
        )
        model.fit(X=X, y=y_da)

        future = pd.date_range(
            start=pd.Timestamp(y_da.coords["obs_ind"].values[-1])
            + pd.Timedelta(days=1),
            periods=5,
            freq="D",
        )
        X_bad = xr.DataArray(
            np.random.randn(5, 1),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": future, "coeffs": ["other"]},
        )

        with pytest.raises(ValueError, match="missing exogenous columns"):
            model.predict(X=X_bad, out_of_sample=True)

    def test_vs_prior_without_covariates_raises(self, sample_data):
        """vs_prior_type without covariates is a configuration error."""
        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            vs_prior_type="spike_and_slab",
        )
        with pytest.raises(ValueError, match="no exogenous covariates"):
            model.build_model(y=sample_data)

    def test_vs_prior_invalid_type(self):
        """Unknown vs_prior_type fails at construction."""
        with pytest.raises(ValueError, match="Unknown prior_type"):
            cp.pymc_models.StateSpaceTimeSeries(
                sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
                vs_prior_type="lasso",  # type: ignore[arg-type]
            )

    def test_vs_prior_beta_exog_precedence_warning(self):
        """Passing both vs_prior_type and a beta_exog prior warns.

        The warning must name the caller: ``pm.Model``'s metaclass calls
        ``__init__``, so ``stacklevel=2`` would blame ``pymc/model/core.py``
        instead of the line that passed both arguments.
        """
        from pymc_extras.prior import Prior

        with pytest.warns(UserWarning, match="variable selection prior takes") as recs:
            cp.pymc_models.StateSpaceTimeSeries(
                sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
                vs_prior_type="spike_and_slab",
                priors={"beta_exog": Prior("Normal", mu=0, sigma=1)},
            )

        precedence = [rec for rec in recs if "takes precedence" in str(rec.message)]
        assert len(precedence) == 1
        assert precedence[0].filename == __file__

    def test_vs_helpers_require_configuration_and_fit(self):
        """Helper methods guard against missing config and missing fit."""
        plain = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
        )
        with pytest.raises(ValueError, match="not configured with vs_prior_type"):
            plain.get_inclusion_probabilities()
        with pytest.raises(ValueError, match="not configured with vs_prior_type"):
            plain.get_shrinkage_factors()

        unfit = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            vs_prior_type="spike_and_slab",
        )
        with pytest.raises(RuntimeError, match="must be fit first"):
            unfit.get_inclusion_probabilities()
        with pytest.raises(RuntimeError, match="must be fit first"):
            unfit.get_shrinkage_factors()

    def test_vs_spike_and_slab_structure(self, sample_data, mock_pymc_sample):
        """Spike-and-slab on covariates: selection variables in the
        posterior and a well-formed inclusion-probability table.

        Structure-only by design: the suite mocks pm.sample session-wide,
        so posterior values are prior draws.
        """
        y_da = sample_data
        n = len(y_da)
        X = xr.DataArray(
            np.random.randn(n, 2),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": y_da.coords["obs_ind"], "coeffs": ["x1", "x2"]},
        )
        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={
                "draws": 10,
                "tune": 10,
                "chains": 1,
                "progressbar": False,
            },
            vs_prior_type="spike_and_slab",
        )
        model.fit(X=X, y=y_da)

        assert "beta_exog" in model.idata.posterior
        assert "gamma_beta_exog" in model.idata.posterior

        incl = model.get_inclusion_probabilities()
        assert isinstance(incl, pd.DataFrame)
        assert list(incl.columns) == ["prob", "selected", "gamma_mean"]
        assert list(incl.index) == ["x1", "x2"]

    def test_vs_horseshoe_structure(self, sample_data, mock_pymc_sample):
        """Horseshoe on covariates: shrinkage factor table is well formed.

        Structure-only by design: the suite mocks pm.sample session-wide.
        """
        y_da = sample_data
        n = len(y_da)
        X = xr.DataArray(
            np.random.randn(n, 2),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": y_da.coords["obs_ind"], "coeffs": ["x1", "x2"]},
        )
        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={
                "draws": 10,
                "tune": 10,
                "chains": 1,
                "progressbar": False,
            },
            vs_prior_type="horseshoe",
        )
        model.fit(X=X, y=y_da)

        assert "beta_exog" in model.idata.posterior
        shrink = model.get_shrinkage_factors()
        assert isinstance(shrink, pd.DataFrame)
        assert list(shrink.index) == ["x1", "x2"]

        # Inclusion probabilities are a spike-and-slab concept
        with pytest.raises(ValueError, match="spike_and_slab"):
            model.get_inclusion_probabilities()

    def test_vs_normal_structure(self, sample_data, mock_pymc_sample):
        """Normal "selection" prior: a plain Normal on beta_exog, no selection.

        Structure-only by design: the suite mocks pm.sample session-wide.
        The factory builds neither the spike-and-slab indicators nor the
        horseshoe scales for this option, so both diagnostics must refuse.
        """
        y_da = sample_data
        n = len(y_da)
        X = xr.DataArray(
            np.random.default_rng(seed=42).normal(size=(n, 2)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": y_da.coords["obs_ind"], "coeffs": ["x1", "x2"]},
        )
        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={
                "draws": 10,
                "tune": 10,
                "chains": 1,
                "progressbar": False,
            },
            vs_prior_type="normal",
        )
        model.fit(X=X, y=y_da)

        posterior = model.idata.posterior
        assert "beta_exog" in posterior
        assert list(posterior["beta_exog"].coords["state_exog"].values) == [
            "x1",
            "x2",
        ]
        assert "gamma_beta_exog" not in posterior
        assert "tau_beta_exog" not in posterior

        with pytest.raises(ValueError, match="spike_and_slab"):
            model.get_inclusion_probabilities()
        with pytest.raises(ValueError, match="horseshoe"):
            model.get_shrinkage_factors()

    def test_vs_clone_preserves_config(self):
        """_clone carries the variable selection configuration."""
        model = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            vs_prior_type="horseshoe",
            vs_hyperparams={"nu": 5},
        )
        clone = model._clone()
        assert clone.vs_prior_type == "horseshoe"
        assert clone.vs_hyperparams == {"nu": 5}
        assert clone.vs_prior is not None

    def test_clone_preserves_config(self):
        """Test that _clone carries over the full model configuration."""
        from pymc_extras.prior import Prior

        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            priors={"sigma_freq": Prior("HalfNormal", sigma=1)},
        )
        clone = model._clone()

        assert clone.level_order == 1
        assert clone.seasonal_length == 7
        assert clone.sample_kwargs == model.sample_kwargs
        assert clone.priors["sigma_freq"].distribution == "HalfNormal"

    def test_missing_prior_raises(self, sample_data):
        """Test error when a state-space parameter has no prior."""
        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
        )
        model.priors.pop("sigma_freq")

        with pytest.raises(
            ValueError,
            match="No prior found for state-space parameters: \\['sigma_freq'\\]",
        ):
            model.build_model(y=sample_data)

    def test_p0_diag_prior_override(self, sample_data):
        """A user prior for the P0 diagonal reaches the model."""
        from pymc_extras.prior import Prior

        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            priors={"P0_diag": Prior("HalfNormal", sigma=1)},
        )
        model.build_model(y=sample_data)

        assert model["P0_diag"].owner.op.name == "halfnormal"
        assert "P0" in [det.name for det in model.deterministics]

    def test_prior_options_are_preserved(self, sample_data):
        """Options beyond the distribution parameters survive dim resolution.

        `centered` and `transform` live on the Prior itself, not in its
        parameters, so rebuilding a prior from distribution plus parameters
        would silently drop them.
        """
        from pymc_extras.prior import Prior

        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            priors={
                "initial_level_trend": Prior("Normal", mu=0, sigma=50, centered=False),
                "params_freq": Prior("Normal", mu=0, sigma=80, transform="log"),
            },
        )
        model.build_model(y=sample_data)

        free_rvs = {rv.name for rv in model.free_RVs}
        deterministics = {det.name for det in model.deterministics}
        # Non-centered priors sample an offset; transformed priors sample a raw.
        assert "initial_level_trend_offset" in free_rvs
        assert "params_freq_raw" in free_rvs
        assert {"initial_level_trend", "params_freq"} <= deterministics
        # The caller's Prior objects are left untouched.
        assert model.priors["initial_level_trend"].dims is None

    def test_rebuild_raises_with_guidance(self, sample_data):
        """Building twice fails with a message that points at the way out."""
        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
        )
        model.build_model(y=sample_data)

        with pytest.raises(RuntimeError, match="already built"):
            model.build_model(y=sample_data)

    def test_predict_out_of_sample_x_none(self, sample_data):
        """Test error when X is None for out-of-sample predictions."""
        y_da = sample_data

        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )

        # Create dummy X for fit (state-space doesn't use it)
        dates = y_da.coords["obs_ind"].values
        dummy_X = xr.DataArray(
            np.zeros((len(dates), 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates, "coeffs": []},
        )
        model.fit(X=dummy_X, y=y_da)

        with pytest.raises(
            ValueError,
            match="X must be provided for out-of-sample predictions",
        ):
            model.predict(X=None, out_of_sample=True)

    def test_predict_out_of_sample_x_no_coords(self, sample_data):
        """Test error when X lacks coords for out-of-sample predictions."""
        y_da = sample_data

        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )

        # Fit model
        dates = y_da.coords["obs_ind"].values
        dummy_X = xr.DataArray(
            np.zeros((len(dates), 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates, "coeffs": []},
        )
        model.fit(X=dummy_X, y=y_da)

        # Try to predict with numpy array (no coords)
        X_no_coords = np.zeros((5, 0))

        with pytest.raises(
            ValueError,
            match="X must have 'obs_ind' coordinate with datetime values",
        ):
            model.predict(X=X_no_coords, out_of_sample=True)

    def test_score_y_none(self, sample_data):
        """Test error when y is None in score."""
        y_da = sample_data

        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )

        dates = y_da.coords["obs_ind"].values
        dummy_X = xr.DataArray(
            np.zeros((len(dates), 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates, "coeffs": []},
        )
        model.fit(X=dummy_X, y=y_da)

        # StateSpaceTimeSeries.score calls super().score() which doesn't validate y
        # So it raises AttributeError when trying to call y.sel()
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute"):
            model.score(X=dummy_X, y=None)


class TestTimeSeriesModelClonePreservesPriors:
    """Regression tests: _clone() must forward user-supplied priors."""

    def test_bayesian_basis_expansion_clone_forwards_priors(self):
        """BayesianBasisExpansionTimeSeries._clone() keeps user priors."""
        pytest.importorskip(
            "pymc_marketing",
            reason="pymc-marketing optional for default BSTS components",
        )
        custom_priors = {"sentinel": "value"}
        original = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            priors=custom_priors,
        )
        cloned = original._clone()
        assert cloned._user_priors == custom_priors
        assert cloned._user_priors is not None

    def test_state_space_clone_forwards_priors(self):
        """StateSpaceTimeSeries._clone() keeps user priors."""
        pytest.importorskip(
            "pymc_extras",
            reason="pymc-extras optional for state-space model",
        )
        custom_priors = {"sentinel": "value"}
        original = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
            priors=custom_priors,
        )
        cloned = original._clone()
        assert cloned._user_priors == custom_priors
        assert cloned._user_priors is not None

    def test_bayesian_basis_expansion_clone_accepts_priors_override(self):
        """BSTS._clone(priors=...) overrides priors and keeps extra init args.

        The ``auto_scale_sigma=False`` opt-out pins the legacy prior by cloning
        with a ``priors`` override; the subclass override must accept it and
        still carry ``n_order`` and friends, or the base and override signatures
        drift apart.
        """
        pytest.importorskip(
            "pymc_marketing",
            reason="pymc-marketing optional for default BSTS components",
        )
        original = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            n_order=7,
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            priors={"sentinel": "value"},
        )
        override = {"y_hat": "pinned"}
        cloned = original._clone(priors=override)
        assert cloned._user_priors == override  # override applied
        assert cloned.n_order == 7  # extra init arg preserved through the clone

    def test_state_space_clone_accepts_priors_override(self):
        """StateSpaceTimeSeries._clone(priors=...) overrides priors, keeps args."""
        pytest.importorskip(
            "pymc_extras",
            reason="pymc-extras optional for state-space model",
        )
        original = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
            priors={"sentinel": "value"},
        )
        override = {"y_hat": "pinned"}
        cloned = original._clone(priors=override)
        assert cloned._user_priors == override
        assert cloned.level_order == 1

    def test_all_shipped_pymc_model_clones_accept_priors_override(self):
        """Every shipped ``PyMCModel._clone`` must accept the ``priors`` override.

        The ``auto_scale_sigma=False`` opt-out pins the legacy prior via
        ``_clone(priors=...)``; a subclass whose override dropped the parameter
        would silently lose the pin. Since the design threads ``priors`` per
        override rather than through a single template method, this guards the
        exact drift that would reintroduce the bug one inheritance level up.
        """
        import inspect

        from causalpy.pymc_models import PyMCModel

        def _subclasses(cls):
            for sub in cls.__subclasses__():
                yield sub
                yield from _subclasses(sub)

        shipped = [
            cls
            for cls in _subclasses(PyMCModel)
            if cls.__module__ == "causalpy.pymc_models"
        ]
        assert shipped  # sanity: the subclasses were discovered
        for cls in shipped:
            params = inspect.signature(cls._clone).parameters
            assert "priors" in params, f"{cls.__name__}._clone drops priors override"


class TestTimeSeriesModelCloneIsUnfitted:
    """Regression tests: ``_clone()`` returns a fresh model with no fitted state.

    The whole point of ``_clone()`` is to give sensitivity checks a model
    they can refit from scratch without inheriting the original's
    posterior, sampling history, or any cached state.  These tests assert
    the clone really is a clean copy: ``idata`` is ``None``, the clone is
    a distinct instance from the original, and fitting the original after
    cloning does not leak into the clone.
    """

    def test_linear_regression_clone_has_no_fitted_state(self):
        """``LinearRegression._clone()`` returns a model with idata=None."""
        original = cp.pymc_models.LinearRegression(
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )
        cloned = original._clone()
        assert cloned is not original
        assert cloned.idata is None

    def test_bayesian_basis_expansion_clone_has_no_fitted_state(self):
        """``BayesianBasisExpansionTimeSeries._clone()`` returns idata=None."""
        pytest.importorskip(
            "pymc_marketing",
            reason="pymc-marketing optional for default BSTS components",
        )
        original = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
        )
        cloned = original._clone()
        assert cloned is not original
        assert cloned.idata is None

    def test_state_space_clone_has_no_fitted_state(self):
        """``StateSpaceTimeSeries._clone()`` returns idata=None."""
        pytest.importorskip(
            "pymc_extras",
            reason="pymc-extras optional for state-space model",
        )
        original = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )
        cloned = original._clone()
        assert cloned is not original
        assert cloned.idata is None

    def test_clone_after_fit_does_not_inherit_idata(self):
        """Cloning a fitted model still produces a fresh, unfitted instance.

        Uses ``LinearRegression`` (the cheapest fittable PyMC model) to
        actually drive a fit with a tiny ``sample_kwargs`` budget, then
        asserts the cloned model has no posterior and that the original's
        ``idata`` is untouched after the clone.
        """
        rng = np.random.default_rng(0)
        n = 30
        X = xr.DataArray(
            rng.normal(size=(n, 1)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": np.arange(n), "coeffs": ["x1"]},
        )
        y = xr.DataArray(
            rng.normal(size=(n, 1)),
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": np.arange(n), "treated_units": ["unit_0"]},
        )

        original = cp.pymc_models.LinearRegression(
            sample_kwargs={
                "draws": 10,
                "tune": 10,
                "chains": 1,
                "progressbar": False,
                "random_seed": 42,
            },
        )
        original.fit(X=X, y=y)
        assert original.idata is not None  # sanity check the fit landed

        cloned = original._clone()

        assert cloned is not original
        assert cloned.idata is None
        assert original.idata is not None  # original is untouched
