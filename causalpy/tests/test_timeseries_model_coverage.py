#   Copyright 2025 - 2025 The PyMC Labs Developers
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


class TestStateSpaceTimeSeriesCoverage:
    """Test uncovered branches in StateSpaceTimeSeries."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range(start="2020-01-01", end="2020-02-01", freq="D")
        n_obs = len(dates)
        y_values = np.random.randn(n_obs) + 10

        y_da = xr.DataArray(
            y_values.reshape(-1, 1),
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": dates, "treated_units": ["unit_0"]},
        )
        return y_da

    def test_custom_trend_component_without_apply_method(self):
        """Test validation error when custom trend component lacks apply method."""
        with pytest.raises(
            ValueError,
            match="Custom trend_component must have an 'apply' method",
        ):
            cp.pymc_models.StateSpaceTimeSeries(
                trend_component=MockComponentNoApply(),
                sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            )

    def test_custom_seasonality_component_without_apply_method(self):
        """Test validation error when custom seasonality component lacks apply method."""
        with pytest.raises(
            ValueError,
            match="Custom seasonality_component must have an 'apply' method",
        ):
            cp.pymc_models.StateSpaceTimeSeries(
                seasonality_component=MockComponentNoApply(),
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
