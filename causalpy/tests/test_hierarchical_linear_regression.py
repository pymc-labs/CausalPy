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
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pymc_extras.prior import Prior

from causalpy.data.simulate_data import generate_hlr_data
from causalpy.pymc_models import HierarchicalLinearRegression


class TestHierarchicalLinearRegression:
    data: ClassVar[dict[str, Any]]
    X: ClassVar[xr.DataArray]
    Z: ClassVar[xr.DataArray]
    y: ClassVar[xr.DataArray]
    group_idx: ClassVar[np.ndarray]
    coords: ClassVar[dict[str, Any]]
    sample_kwargs: ClassVar[dict[str, Any]]
    priors: ClassVar[dict[str, Prior]]
    model_true: ClassVar[dict[str, np.ndarray]]
    model: ClassVar[HierarchicalLinearRegression]
    idata: ClassVar[xr.DataTree]

    @staticmethod
    def _expect_value_error(fn, expected_message: str) -> None:
        with pytest.raises(ValueError, match=expected_message):
            fn()

    @staticmethod
    def _prepare_data(
        XY: pd.DataFrame,
        Z_: pd.DataFrame,
        params: dict[str, np.ndarray],
        *,
        seed: int = 42,
        sample_kwargs: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Map HLR data into HierarchicalLinearRegression inputs."""
        obs_ind = XY.index.to_numpy()
        group_idx = XY["group_idx"].to_numpy()
        n_groups = int(XY["group_idx"].nunique())

        _X = XY[["1", "x1"]].to_numpy()
        _Z = Z_.to_numpy()
        _y = XY["y"].to_numpy()

        X = xr.DataArray(
            _X,
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": obs_ind, "coeffs": ["1", "x1"]},
        )
        Z = xr.DataArray(
            _Z,
            dims=["obs_ind", "random_coeffs"],
            coords={"obs_ind": obs_ind, "random_coeffs": list(Z_.columns)},
        )
        y = xr.DataArray(
            _y[:, None],
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": obs_ind, "treated_units": ["unit_0"]},
        )

        coords = {
            "obs_ind": obs_ind,
            "coeffs": ["1", "x1"],
            "random_coeffs": list(Z_.columns),
            "treated_units": ["unit_0"],
            "groups": np.arange(n_groups),
        }

        resolved_sample_kwargs: dict[str, Any] = {
            "draws": 500,
            "tune": 500,
            "chains": 2,
            "target_accept": 0.90,
            "progressbar": False,
            "random_seed": seed,
        }
        if sample_kwargs is not None:
            resolved_sample_kwargs.update(sample_kwargs)

        return {
            "model_kwargs": {
                "X": X,
                "Z": Z,
                "y": y,
                "group_idx": group_idx,
                "coords": coords,
            },
            "sample_kwargs": resolved_sample_kwargs,
            "priors": {
                "beta_fixed": Prior(
                    "Normal", mu=0, sigma=10, dims=["treated_units", "coeffs"]
                ),
                "sigma_fixed": Prior("HalfNormal", sigma=1, dims=["treated_units"]),
                "sigma_random": Prior(
                    "HalfNormal", sigma=1, dims=["treated_units", "random_coeffs"]
                ),
                "beta_group": Prior(
                    "Normal",
                    mu=0,
                    sigma=1,
                    dims=["groups", "treated_units", "random_coeffs"],
                ),
            },
            "model_true": params,
        }

    @classmethod
    def _setup_data(
        cls,
        *,
        seed: int = 42,
        n_groups: int = 12,
        n_obs_per_group: int = 40,
        beta_fixed_true: tuple[float, float] = (2.0, 1.4),
        sigma_random_true: tuple[float, float] = (0.7, 0.5),
        sigma_noise: float = 0.8,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        XY, Z, params = generate_hlr_data(
            seed=seed,
            n_groups=n_groups,
            n_obs_per_group=n_obs_per_group,
            beta_fixed_true=beta_fixed_true,
            sigma_random_true=sigma_random_true,
            sigma_noise=sigma_noise,
        )
        return cls._prepare_data(
            XY,
            Z,
            params,
            seed=seed,
            sample_kwargs=sample_kwargs,
        )

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def _setup_model(cls, real_pymc_sampling) -> None:
        cls.data = cls._setup_data(
            sample_kwargs={
                "draws": 120,
                "tune": 120,
                "chains": 2,
                "cores": 1,
            }
        )
        cls.X = cls.data["model_kwargs"]["X"]
        cls.Z = cls.data["model_kwargs"]["Z"]
        cls.y = cls.data["model_kwargs"]["y"]
        cls.group_idx = cls.data["model_kwargs"]["group_idx"]
        cls.coords = cls.data["model_kwargs"]["coords"]
        cls.sample_kwargs = cls.data["sample_kwargs"]
        cls.priors = cls.data["priors"]
        cls.model_true = cls.data["model_true"]

        cls.model = HierarchicalLinearRegression(
            sample_kwargs=cls.sample_kwargs, priors=cls.priors
        )
        cls.model.build(
            X=cls.X,
            Z=cls.Z,
            y=cls.y,
            group_idx=cls.group_idx,
            coords=cls.coords,
        )
        cls.idata = cls.model.sample_posterior()

    def test_default_priors_build_model(self) -> None:
        model = HierarchicalLinearRegression()

        model.build_model(
            X=self.X,
            Z=self.Z,
            y=self.y,
            group_idx=self.group_idx,
            coords=self.coords,
        )

        assert "beta_fixed" in model.named_vars
        assert "sigma_fixed" in model.named_vars
        assert "sigma_random" in model.named_vars
        assert "beta_group" in model.named_vars
        assert "beta_random" in model.named_vars

    def test_partial_custom_priors_do_not_fall_back_to_defaults(self) -> None:
        priors = {
            "beta_fixed": self.model.priors["beta_fixed"],
            "sigma_fixed": self.model.priors["sigma_fixed"],
            "sigma_random": self.model.priors["sigma_random"],
        }

        self._expect_value_error(
            lambda: HierarchicalLinearRegression(priors=priors).build(
                X=self.X,
                Z=self.Z,
                y=self.y,
                group_idx=self.group_idx,
                coords=self.coords,
                non_centered=True,
            ),
            "Missing required custom priors for non-centered parameterization: beta_group.",
        )

    def test_centered_requires_sigma_random_prior(self) -> None:
        priors = {key: self.model.priors[key] for key in ["beta_fixed", "sigma_fixed"]}
        self._expect_value_error(
            lambda: HierarchicalLinearRegression(priors=priors).build(
                X=self.X,
                Z=self.Z,
                y=self.y,
                group_idx=self.group_idx,
                coords=self.coords,
                non_centered=False,
            ),
            "Missing required custom priors for centered parameterization: sigma_random.",
        )

    def test_noncentered_requires_beta_group_prior(self) -> None:
        priors = {
            key: self.model.priors[key]
            for key in ["beta_fixed", "sigma_fixed", "sigma_random"]
        }
        self._expect_value_error(
            lambda: HierarchicalLinearRegression(priors=priors).build(
                X=self.X,
                Z=self.Z,
                y=self.y,
                group_idx=self.group_idx,
                coords=self.coords,
                non_centered=True,
            ),
            "Missing required custom priors for non-centered parameterization: beta_group.",
        )

    def test_centered_build_model_uses_centered_random_effects(self) -> None:
        priors = {
            key: self.model.priors[key]
            for key in ["beta_fixed", "sigma_fixed", "sigma_random"]
        }
        model = HierarchicalLinearRegression(priors=priors)

        model.build_model(
            X=self.X,
            Z=self.Z,
            y=self.y,
            group_idx=self.group_idx,
            coords=self.coords,
            non_centered=False,
        )

        assert "beta_random" in model.named_vars
        assert "beta_group" not in model.named_vars

    def test_fit_extends_prior_and_posterior_predictive(self, mock_pymc_sample) -> None:
        model = HierarchicalLinearRegression(
            sample_kwargs={
                "draws": 2,
                "tune": 2,
                "chains": 1,
                "progressbar": False,
                "random_seed": self.sample_kwargs["random_seed"],
            },
            priors=self.priors,
        )
        result = model.fit(**self.data["model_kwargs"])

        assert result is model.idata
        assert isinstance(result, xr.DataTree)
        assert {
            "posterior",
            "prior",
            "prior_predictive",
            "posterior_predictive",
        } <= set(result.children)

    @pytest.mark.correctness
    def test_fixed_effect_parameter_recovery(self) -> None:
        beta_fixed_mean = (
            self.idata.posterior["beta_fixed"].mean(dim=("chain", "draw")).to_numpy()[0]
        )
        np.testing.assert_allclose(
            beta_fixed_mean,
            self.model_true["beta_fixed_true"],
            atol=0.50,
        )

    @pytest.mark.correctness
    def test_random_scale_parameter_recovery(self) -> None:
        sigma_random_mean = (
            self.idata.posterior["sigma_random"]
            .mean(dim=("chain", "draw"))
            .to_numpy()[0]
        )
        np.testing.assert_allclose(
            sigma_random_mean,
            self.model_true["sigma_random_true"],
            atol=0.50,
        )
