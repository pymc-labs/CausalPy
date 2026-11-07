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
"""Tests for hierarchical difference in differences."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from causalpy.custom_exceptions import (
    DataException,
    FormulaException,
    GroupNotSampledException,
)
from causalpy.data.simulate_data import generate_hdid_data
from causalpy.experiments import hierarchical_difference_in_differences as hdid_module
from causalpy.experiments.hierarchical_difference_in_differences import (
    HierarchicalDifferenceInDifferences,
)
from causalpy.pymc_models import HierarchicalLinearRegression


@pytest.mark.correctness
@pytest.mark.parametrize("seed", [42, 656, 20260296])
def test_hdid_recovers_variance_parameters_and_icc(seed, genuine_pymc_sampling):
    """Recover generating parameters through the real simulator and HDID fit."""
    data, _ = generate_hdid_data(
        seed=seed,
        n_stores_total=200,
        n_stores_treated=100,
        n_months=4,
        pre_months=2,
        units_min=5,
        units_max=5,
        units_lam=5.0,
        sigma_store_intercept=1.3,
        sigma_treatment_slope=1.5,
        sigma_noise=2.0,
        intercept=5.0,
        post_effect=1.0,
        trend_weight=0.0,
        include_covariate_effects=False,
        include_region_effects=False,
    )
    experiment = hdid_module.HierarchicalDifferenceInDifferences(
        data,
        "purchase_amount ~ 1 + post_treatment + treated + post_treatment:treated "
        "+ (1 + post_treatment:treated | store_id)",
        time_variable_name="month",
        unit_variable_name="customer_id",
        model=HierarchicalLinearRegression(
            sample_kwargs={
                "draws": 2000,
                "tune": 2000,
                "chains": 4,
                "cores": 1,
                "target_accept": 0.95,
                "progressbar": False,
                "random_seed": seed,
            },
            prior_sample_kwargs={"draws": 10, "random_seed": seed},
        ),
    ).fit()

    posterior = experiment.idata.posterior
    draws = {
        term: posterior["beta_fixed"].sel(coeffs=term, drop=True).squeeze(drop=True)
        for term in ["1", "post_treatment", "treated", "post_treatment:treated"]
    }
    draws.update(
        {
            "intercept_sd": posterior["sigma_random"]
            .sel(random_coeffs="1|store_id", drop=True)
            .squeeze(drop=True),
            "response_sd": posterior["sigma_random"]
            .sel(random_coeffs="post_treatment:treated|store_id", drop=True)
            .squeeze(drop=True),
            "noise_sd": posterior["sigma_fixed"].squeeze(drop=True),
            "icc": experiment.result.icc.squeeze(drop=True),
        }
    )
    # The population target uses the generating variances, not the empirical
    # covariance of the particular store effects drawn for this dataset.
    treated_post_fraction = (data["treated"] * data["post_treatment"]).mean()
    random_variance = 1.3**2 + treated_post_fraction * 1.5**2
    truth = {
        "1": 5.0,
        "post_treatment": 1.0,
        "treated": 2.2,
        "post_treatment:treated": 4.0,
        "intercept_sd": 1.3,
        "response_sd": 1.5,
        "noise_sd": 2.0,
        "icc": random_variance / (random_variance + 2.0**2),
    }
    summary = az.summary(xr.Dataset(draws), ci_kind="hdi", ci_prob=0.94)
    print(summary.assign(truth=pd.Series(truth)).to_string())
    assert int(experiment.idata.sample_stats["diverging"].sum()) == 0
    assert summary["r_hat"].max() < 1.01
    assert summary["ess_bulk"].min() > 400
    assert summary["ess_tail"].min() > 400

    # Sampling uncertainty varies across simulated datasets. Require both
    # agreement within posterior uncertainty and informative absolute precision.
    max_sd = pd.Series([0.4, 0.2, 0.5, 0.5, 0.25, 0.4, 0.1, 0.08], index=truth)
    standardized_error = (summary["mean"] - pd.Series(truth)).abs() / summary["sd"]
    assert (standardized_error < 3.5).all(), standardized_error
    assert (summary["sd"] < max_sd).all(), summary["sd"]
    assert abs(summary.loc["icc", "mean"] - truth["icc"]) < 0.1


class _FixedPosteriorHDiDModel(HierarchicalLinearRegression):
    """PyMCModel test double with deterministic hierarchical posterior draws."""

    def __init__(
        self,
        beta_fixed: np.ndarray | None = None,
        beta_random: np.ndarray | None = None,
        mu: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.beta_fixed = beta_fixed
        self.beta_random = beta_random
        self.mu = mu

    def build(self, **kwargs: Any) -> None:
        self.build_kwargs = kwargs
        self._built = True

    def sample_posterior(self, **kwargs: Any) -> xr.DataTree:
        if self.idata is None:
            self.idata = xr.DataTree()
        self.idata["posterior"] = self._sample_draws()
        return self.idata

    def sample_prior_predictive(self, **kwargs: Any) -> xr.DataTree:
        if self.idata is None:
            self.idata = xr.DataTree()
        self.idata["prior"] = self._sample_draws() + 1
        return self.idata

    def _sample_draws(self) -> xr.Dataset:
        kwargs = self.build_kwargs
        beta_fixed = self.beta_fixed
        if beta_fixed is None:
            beta_fixed = np.zeros((1, 2, 1, len(kwargs["coords"]["coeffs"])))
        beta_random = self.beta_random
        if beta_random is None:
            beta_random = np.zeros(
                (
                    1,
                    2,
                    len(kwargs["coords"]["groups"]),
                    1,
                    len(kwargs["coords"]["random_coeffs"]),
                )
            )
        mu = self.mu
        if mu is None:
            mu = np.zeros((1, 2, len(kwargs["coords"]["obs_ind"]), 1))
        sigma_fixed = np.ones((1, 2, 1))
        sigma_fixed[0, 1, 0] = 1.2
        sigma_random = np.ones((1, 2, 1, len(kwargs["coords"]["random_coeffs"])))
        sigma_random[0, 1, :, :] = 1.5
        posterior = xr.Dataset(
            data_vars={
                "beta_fixed": (
                    ["chain", "draw", "treated_units", "coeffs"],
                    beta_fixed,
                ),
                "sigma_fixed": (
                    ["chain", "draw", "treated_units"],
                    sigma_fixed,
                ),
                "sigma_random": (
                    ["chain", "draw", "treated_units", "random_coeffs"],
                    sigma_random,
                ),
                "beta_random": (
                    ["chain", "draw", "groups", "treated_units", "random_coeffs"],
                    beta_random,
                ),
                "mu": (
                    ["chain", "draw", "obs_ind", "treated_units"],
                    mu,
                ),
            },
            coords={
                "chain": [0],
                "draw": [0, 1],
                "treated_units": kwargs["coords"]["treated_units"],
                "coeffs": kwargs["coords"]["coeffs"],
                "random_coeffs": kwargs["coords"]["random_coeffs"],
                "groups": kwargs["coords"]["groups"],
                "obs_ind": kwargs["coords"]["obs_ind"],
            },
        )
        return posterior


class TestHierarchicalDifferenceInDifferencesInterface:
    @staticmethod
    def _panel_data() -> pd.DataFrame:
        rows = []
        for store_idx in range(10):
            store_id = f"s{store_idx + 1}"
            treated = int(store_idx >= 5)
            for customer_idx in range(3):
                for month in [1, 2]:
                    rows.append(
                        {
                            "store_id": store_id,
                            "customer_id": f"{store_id}_c{customer_idx}",
                            "month": month,
                            "treated": treated,
                            "post_treatment": int(month == 2),
                            "y": float(10 + treated + month),
                        }
                    )
        return pd.DataFrame(rows)

    @staticmethod
    def _matrices(
        data: pd.DataFrame,
        *,
        include_did: bool = True,
        include_random_did: bool = True,
    ) -> SimpleNamespace:
        fixed = {
            "1": np.ones(len(data)),
            "post_treatment": data["post_treatment"].to_numpy(),
            "treated": data["treated"].to_numpy(),
        }
        if include_did:
            fixed["post_treatment:treated"] = (
                data["post_treatment"].to_numpy() * data["treated"].to_numpy()
            )
        rhs = pd.DataFrame(fixed, index=data.index)
        lhs = pd.DataFrame({"y": data["y"].to_numpy()}, index=data.index)
        random_effects = {"1|store_id": np.ones(len(data))}
        if include_random_did:
            random_effects["post_treatment:treated|store_id"] = (
                data["post_treatment"].to_numpy() * data["treated"].to_numpy()
            )
        z_matrix = pd.DataFrame(random_effects, index=data.index)
        group_idx, group_labels = pd.factorize(data["store_id"], sort=False)
        return SimpleNamespace(
            lhs=lhs,
            rhs=rhs,
            Z=z_matrix,
            metadata={
                "outcome_name": "y",
                "has_random_effects": True,
                "fixed_effect_names": list(rhs.columns),
                "random_effect_names": list(z_matrix.columns),
                "group": {
                    "variable": "store_id",
                    "labels": [str(label) for label in group_labels.tolist()],
                    "n_groups": int(len(group_labels)),
                    "idx": group_idx.astype(np.int32),
                    "components": [
                        "(post_treatment:treated | store_id)"
                        if include_random_did
                        else "(1 | store_id)"
                    ],
                },
            },
        )

    @staticmethod
    def _patch_parse_formula(
        monkeypatch: pytest.MonkeyPatch, matrices: SimpleNamespace
    ) -> None:
        monkeypatch.setattr(
            hdid_module, "parse_formula", lambda formula, data: matrices
        )

    @staticmethod
    def _experiment(
        data: pd.DataFrame,
        *,
        model: HierarchicalLinearRegression | None = None,
        non_centered: bool = True,
        fit: bool = True,
    ) -> HierarchicalDifferenceInDifferences:
        if model is None:
            model = _FixedPosteriorHDiDModel()
        experiment = HierarchicalDifferenceInDifferences(
            data=data,
            formula=(
                "y ~ 1 + post_treatment + treated + post_treatment:treated "
                "+ (post_treatment:treated | store_id)"
            ),
            time_variable_name="month",
            unit_variable_name="customer_id",
            model=model,
            non_centered=non_centered,
        )
        return experiment.fit() if fit else experiment

    def test_group_variable_is_inferred_from_random_effects_formula(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that the grouping variable is inferred from the random-effects formula."""
        data = self._panel_data()
        matrices = self._matrices(data)
        self._patch_parse_formula(monkeypatch, matrices)

        result = self._experiment(data)

        assert result.group_variable_name == "store_id"
        assert result.group_labels == [f"s{idx}" for idx in range(1, 11)]
        np.testing.assert_array_equal(
            result.group_idx, matrices.metadata["group"]["idx"]
        )

    def test_group_variable_is_inferred_from_real_parser(self) -> None:
        """Validate that the parser provides random-effects group metadata."""
        data = self._panel_data()

        result = self._experiment(data)

        assert result.group_variable_name == "store_id"
        assert result.group_labels == [f"s{idx}" for idx in range(1, 11)]
        assert result.n_groups == 10
        assert "post_treatment:treated|store_id" in result.random_effect_labels

    def test_icc_uses_mean_observation_level_random_variance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate generalized ICC across the observed random-effects design."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data))
        result = self._experiment(data)

        idata = result.idata
        assert idata is not None
        sigma_random = idata.posterior["sigma_random"] ** 2
        sigma_fixed = idata.posterior["sigma_fixed"] ** 2
        mean_random_variance = (
            (result.Z**2 * sigma_random).sum("random_coeffs").mean("obs_ind")
        )
        expected = mean_random_variance / (mean_random_variance + sigma_fixed)

        assert result.result.icc is not None
        xr.testing.assert_allclose(result.result.icc, expected)

    @pytest.mark.parametrize("missing_variable", ["sigma_fixed", "sigma_random"])
    def test_posterior_data(
        self,
        monkeypatch: pytest.MonkeyPatch,
        missing_variable: str,
    ) -> None:
        """Skip ICC calculation when posterior variance data are unavailable."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data))
        result = self._experiment(data)
        idata = result.idata
        assert idata is not None
        del idata.posterior[missing_variable]

        assert result._compute_icc(idata.posterior.to_dataset()) is None

    def test_observation_random_variance_uses_full_covariance_matrix(self) -> None:
        """Validate that ICC variance includes random-effect covariance."""
        design = xr.DataArray(
            [[1.0, 0.0], [1.0, 1.0]],
            dims=("obs_ind", "random_coeffs"),
            coords={"random_coeffs": ["intercept", "slope"]},
        )
        covariance = xr.DataArray(
            [[2.0, 0.5], [0.5, 3.0]],
            dims=("random_coeffs_row", "random_coeffs_column"),
            coords={
                "random_coeffs_row": ["intercept", "slope"],
                "random_coeffs_column": ["intercept", "slope"],
            },
        )

        result = HierarchicalDifferenceInDifferences._observation_random_variance(
            design, covariance
        )

        xr.testing.assert_allclose(
            result,
            xr.DataArray([2.0, 6.0], dims=("obs_ind",)),
        )

    def test_formula_includes_random_effects(self) -> None:
        """Reject parsed DiD formulas without a random-effects term."""
        data = self._panel_data()

        with pytest.raises(FormulaException, match="requires a random-effects term"):
            HierarchicalDifferenceInDifferences(
                data=data,
                formula="y ~ 1 + post_treatment * treated",
                time_variable_name="month",
                unit_variable_name="customer_id",
                model=_FixedPosteriorHDiDModel(),
            )

    def test_formula_includes_did_interaction(self) -> None:
        """Reject parsed hierarchical formulas without the DiD interaction."""
        data = self._panel_data()

        with pytest.raises(FormulaException, match="exactly one DiD interaction"):
            HierarchicalDifferenceInDifferences(
                data=data,
                formula="y ~ 1 + post_treatment + treated + (1 | store_id)",
                time_variable_name="month",
                unit_variable_name="customer_id",
                model=_FixedPosteriorHDiDModel(),
            )

    @pytest.mark.parametrize(
        ("case", "error", "match"),
        [
            ("missing-group", FormulaException, "grouping variable"),
            ("multiple-components", FormulaException, "exactly one random-effects"),
            ("missing-random-labels", FormulaException, "no random-effect columns"),
            ("inconsistent-labels", FormulaException, "metadata is inconsistent"),
            ("invalid-group-indices", DataException, "invalid group indices"),
        ],
        ids=[
            "missing-group",
            "multiple-components",
            "missing-random-labels",
            "inconsistent-labels",
            "invalid-group-indices",
        ],
    )
    def test_parser_metadata_validation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case: str,
        error: type[Exception],
        match: str,
    ) -> None:
        """Reject malformed mixed-model metadata returned by the parser."""
        data = self._panel_data()
        matrices = self._matrices(data)
        if case == "missing-group":
            matrices.metadata["group"]["variable"] = None
        elif case == "multiple-components":
            matrices.metadata["group"]["components"].append("(1 | store_id)")
        elif case == "missing-random-labels":
            matrices.metadata["random_effect_names"] = []
            matrices.Z = matrices.Z.iloc[:, :0]
        elif case == "inconsistent-labels":
            matrices.metadata["group"]["labels"] = matrices.metadata["group"]["labels"][
                :-1
            ]
        elif case == "invalid-group-indices":
            matrices.metadata["group"]["idx"] = matrices.metadata["group"]["idx"][:-1]
        self._patch_parse_formula(monkeypatch, matrices)

        with pytest.raises(error, match=match):
            self._experiment(data)

    def test_prepare_data_uses_mixed_model_matrices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that data preparation consumes mixed model matrices."""
        data = self._panel_data()
        matrices = self._matrices(data)
        self._patch_parse_formula(monkeypatch, matrices)

        result = self._experiment(data)

        assert result.outcome_variable_name == "y"
        assert result.labels == list(matrices.rhs.columns)
        assert result.random_effect_labels == list(matrices.Z.columns)
        assert result.coords["groups"] == [f"s{idx}" for idx in range(1, 11)]
        assert result.X.dims == ("obs_ind", "coeffs")
        assert result.Z.dims == ("obs_ind", "random_coeffs")
        assert result.y.dims == ("obs_ind", "treated_units")

    def test_att_selects_did_coefficient_posterior(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that ATT is the posterior for the DiD fixed-effect coefficient."""
        data = self._panel_data()
        matrices = self._matrices(data)
        self._patch_parse_formula(monkeypatch, matrices)
        beta_fixed = np.zeros((1, 2, 1, len(matrices.metadata["fixed_effect_names"])))
        did_idx = matrices.metadata["fixed_effect_names"].index(
            "post_treatment:treated"
        )
        beta_fixed[:, :, :, did_idx] = np.array([[[2.5], [3.5]]])
        model = _FixedPosteriorHDiDModel(beta_fixed=beta_fixed)

        result = self._experiment(data, model=model)

        assert result.did_term == "post_treatment:treated"
        idata = result.idata
        assert idata is not None
        expected = idata.posterior["beta_fixed"].sel(coeffs=result.did_term)
        xr.testing.assert_identical(result.result.causal_impact, expected)
        xr.testing.assert_identical(result.result.causal_impact, expected)

    def test_group_effects_selects_random_effects_posterior(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that group effects are the random-effects posterior."""
        data = self._panel_data()
        matrices = self._matrices(data)
        self._patch_parse_formula(monkeypatch, matrices)

        result = self._experiment(data)

        idata = result.idata
        assert idata is not None
        expected = idata.posterior["beta_random"]
        xr.testing.assert_identical(result.result.group_effects, expected)

    def test_plot_group_effects_prepares_selected_forest_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate group-effect selection and ArviZ forest-plot wiring."""
        data = self._panel_data()
        matrices = self._matrices(data)
        self._patch_parse_formula(monkeypatch, matrices)
        random_coeff = "post_treatment:treated|store_id"
        result = self._experiment(data)
        idata = result.idata
        assert idata is not None
        posterior_vars = set(idata.posterior.data_vars)
        recorded: dict[str, Any] = {}

        def mock_plot_forest(plot_data: xr.Dataset, **kwargs: Any) -> np.ndarray:
            recorded["plot_data"] = plot_data.posterior.to_dataset()
            recorded["kwargs"] = kwargs
            _, ax = plt.subplots()
            return SimpleNamespace(get_viz=lambda *args, **kwargs: ax)

        monkeypatch.setattr(hdid_module.az, "plot_forest", mock_plot_forest)

        fig, ax = result.plot_group_effects(
            random_coeff=random_coeff,
            hdi_prob=0.8,
            combined=False,
            show=False,
        )

        plot_data = recorded["plot_data"]
        expected = (
            result.result.group_effects.sel(random_coeffs=random_coeff, drop=True)
            .squeeze("treated_units", drop=True)
            .rename("group_deviation")
        )
        xr.testing.assert_identical(plot_data["group_deviation"], expected)
        assert plot_data["group_deviation"].dims == ("chain", "draw", "groups")
        assert plot_data.coords["groups"].values.tolist() == result.group_labels
        assert recorded["kwargs"]["var_names"] == ["group_deviation"]
        assert recorded["kwargs"]["combined"] is False
        assert recorded["kwargs"]["ci_probs"] == (0.5, 0.8)
        assert recorded["kwargs"]["ci_kind"] == "hdi"
        assert recorded["kwargs"]["figure_kwargs"]["figsize"] == (8, 5.0)
        assert recorded["kwargs"]["labels"] == ["groups"]
        assert ax.get_xlabel() == "Deviation from population coefficient"
        assert ax.get_title() == "Posterior group deviations (80% HDI)"
        np.testing.assert_allclose(ax.lines[-1].get_xdata(), [0, 0])
        assert set(idata.posterior.data_vars) == posterior_vars
        plt.close(fig)

        fig, _ = result.plot_group_effects(show=False)

        assert "random_coeffs" in recorded["plot_data"]["group_deviation"].dims
        plt.close(fig)

    def test_plot_group_effects_returns_figure_and_axis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that selected group effects render with ArviZ."""
        data = self._panel_data()
        matrices = self._matrices(data)
        self._patch_parse_formula(monkeypatch, matrices)
        result = self._experiment(data)
        shown = False

        def mock_show() -> None:
            nonlocal shown
            shown = True

        monkeypatch.setattr(plt, "show", mock_show)

        fig, ax = result.plot_group_effects(
            random_coeff="post_treatment:treated|store_id",
            show=True,
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Deviation from population coefficient"
        assert shown
        plt.close(fig)

    @pytest.mark.parametrize(
        ("case", "match"),
        [
            ("unknown-coefficient", "Available coefficients"),
            ("missing-posterior", "does not contain beta_random"),
        ],
        ids=["unknown-coefficient", "missing-posterior"],
    )
    def test_plot_group_effects_validation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case: str,
        match: str,
    ) -> None:
        """Validate that group-effect plotting reports available coefficients."""
        data = self._panel_data()
        matrices = self._matrices(data)
        self._patch_parse_formula(monkeypatch, matrices)
        result = self._experiment(data)
        idata = result.idata
        assert idata is not None
        if case == "missing-posterior":
            del idata.posterior["beta_random"]

        with pytest.raises(ValueError, match=match):
            result.plot_group_effects(random_coeff="unknown", show=False)

    @pytest.mark.parametrize(
        "case",
        [
            "complete",
            "missing-idata",
            "missing-mu",
            "no-counterfactual",
            "dataarray-hdi",
        ],
    )
    def test_get_plot_data_returns_observed_panel_copy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case: str,
    ) -> None:
        """Validate that plot data starts from the observed panel."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data))
        result = self._experiment(data)
        idata = result.idata
        assert idata is not None
        if case == "missing-idata":
            result.model = _FixedPosteriorHDiDModel()
            with pytest.raises(GroupNotSampledException):
                result.get_plot_data()
            return
        elif case == "missing-mu":
            del idata.posterior["mu"]
        elif case == "no-counterfactual":
            result._result = replace(result.result, y_pred_counterfactual=None)
        elif case == "dataarray-hdi":
            original_hdi = hdid_module.az.hdi

            def dataarray_hdi(*args: Any, **kwargs: Any) -> xr.DataArray:
                hdi = original_hdi(*args, **kwargs)
                if isinstance(hdi, xr.Dataset):
                    return next(iter(hdi.data_vars.values()))
                return hdi

            monkeypatch.setattr(hdid_module.az, "hdi", dataarray_hdi)

        plot_data = result.get_plot_data()

        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape[0] == data.shape[0]
        observed_columns = {
            "store_id",
            "customer_id",
            "month",
            "treated",
            "post_treatment",
            "y",
        }
        fitted_columns = {
            "y_fitted",
            "y_fitted_lower",
            "y_fitted_upper",
        }
        counterfactual_columns = {
            "y_counterfactual",
            "y_counterfactual_lower",
            "y_counterfactual_upper",
        }
        assert observed_columns <= set(plot_data.columns)
        if case in {"missing-idata", "missing-mu"}:
            assert fitted_columns.isdisjoint(plot_data.columns)
            assert counterfactual_columns.isdisjoint(plot_data.columns)
        elif case == "no-counterfactual":
            assert fitted_columns <= set(plot_data.columns)
            assert counterfactual_columns.isdisjoint(plot_data.columns)
        else:
            assert fitted_columns | counterfactual_columns <= set(plot_data.columns)

    def test_counterfactual_removes_fixed_and_random_did_terms(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate counterfactual predictions remove fixed and group-level DiD terms."""
        data = self._panel_data()
        matrices = self._matrices(data, include_random_did=True)
        self._patch_parse_formula(monkeypatch, matrices)

        coeffs = matrices.metadata["fixed_effect_names"]
        random_coeffs = matrices.metadata["random_effect_names"]
        n_groups = matrices.metadata["group"]["n_groups"]
        n_obs = data.shape[0]
        beta_fixed = np.zeros((1, 2, 1, len(coeffs)))
        beta_fixed[0, :, 0, coeffs.index("post_treatment:treated")] = [5.0, 6.0]
        beta_random = np.zeros((1, 2, n_groups, 1, len(random_coeffs)))
        beta_random[
            0,
            :,
            :,
            0,
            random_coeffs.index("post_treatment:treated|store_id"),
        ] = np.array([2.0, 3.0])[:, None]
        mu = np.zeros((1, 2, n_obs, 1))
        mu[0, 0, :, 0] = 20.0
        mu[0, 1, :, 0] = 24.0
        model = _FixedPosteriorHDiDModel(
            beta_fixed=beta_fixed,
            beta_random=beta_random,
            mu=mu,
        )

        result = self._experiment(data, model=model)
        assert result.result.y_pred_counterfactual is not None
        plot_data = result.get_plot_data()
        treated_post = plot_data["treated"].astype(bool) & plot_data[
            "post_treatment"
        ].astype(bool)

        np.testing.assert_allclose(
            plot_data.loc[treated_post, "y_counterfactual"],
            14.0,
        )
        np.testing.assert_allclose(
            plot_data.loc[~treated_post, "y_counterfactual"],
            22.0,
        )

    @pytest.mark.parametrize(
        "case",
        [
            "missing-mu",
            "missing-beta-fixed",
            "missing-beta-random",
            "no-random-slope",
        ],
    )
    def test_counterfactual(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case: str,
    ) -> None:
        """Handle unavailable posterior data and models without random DiD slopes."""
        data = self._panel_data()
        include_random_did = case != "no-random-slope"
        matrices = self._matrices(data, include_random_did=include_random_did)
        self._patch_parse_formula(monkeypatch, matrices)

        coeffs = matrices.metadata["fixed_effect_names"]
        beta_fixed = np.zeros((1, 2, 1, len(coeffs)))
        beta_fixed[..., coeffs.index("post_treatment:treated")] = 2.0
        mu = np.full((1, 2, data.shape[0], 1), 10.0)
        result = self._experiment(
            data,
            model=_FixedPosteriorHDiDModel(beta_fixed=beta_fixed, mu=mu),
        )
        idata = result.idata
        assert idata is not None

        missing_variable = case.removeprefix("missing-").replace("-", "_")
        if case.startswith("missing-"):
            del idata.posterior[missing_variable]

        counterfactual = result._compute_counterfactual(idata.posterior.to_dataset())

        if case in {"missing-mu", "missing-beta-fixed"}:
            assert counterfactual is None
        else:
            assert counterfactual is not None
            did_design = result.X.sel(coeffs=result.did_term)
            expected = idata.posterior["mu"] - (
                idata.posterior["beta_fixed"].sel(coeffs=result.did_term) * did_design
            ).transpose(*idata.posterior["mu"].dims)
            xr.testing.assert_allclose(counterfactual, expected)

    def test_public_plot_returns_figure_and_axis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that the public plot method renders Matplotlib objects."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data))
        result = self._experiment(data)

        fig, ax = result.plot(show=False)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    @pytest.mark.parametrize("case", ["complete", "missing-mu", "no-counterfactual"])
    def test_plot_hook_returns_figure_and_axis(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case: str,
    ) -> None:
        """Validate that the plot hook returns Matplotlib objects."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data))
        result = self._experiment(data)
        idata = result.idata
        assert idata is not None
        if case == "missing-mu":
            del idata.posterior["mu"]
        elif case == "no-counterfactual":
            result._result = replace(result.result, y_pred_counterfactual=None)

        fig, ax = result._plot()

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    @pytest.mark.parametrize("case", ["complete", "no-icc", "missing-sigma"])
    def test_variance_components(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case: str,
    ) -> None:
        """Validate variance plotting and unavailable posterior components."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data))
        result = self._experiment(data)
        idata = result.idata
        assert idata is not None
        if case == "no-icc":
            result._result = replace(result.result, icc=None)
        elif case == "missing-sigma":
            del idata.posterior["sigma_fixed"]
        posterior_vars = set(idata.posterior.data_vars)

        if case == "missing-sigma":
            with pytest.raises(ValueError, match="missing variance component"):
                result.plot_variance_components(show=False)
            return

        shown = False

        def mock_show() -> None:
            nonlocal shown
            shown = True

        monkeypatch.setattr(plt, "show", mock_show)
        fig, ax = result.plot_variance_components(show=True)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert set(idata.posterior.data_vars) == posterior_vars
        assert shown
        plt.close(fig)

    @pytest.mark.parametrize(
        ("method_name", "kwargs"),
        [
            ("plot_group_effects", {"show": False}),
            ("plot_variance_components", {"show": False}),
            ("plot", {"show": False}),
            ("get_plot_data", {}),
        ],
        ids=["group-effects", "variance", "plot", "plot-data"],
    )
    def test_invalid_hdi_prob(
        self,
        monkeypatch: pytest.MonkeyPatch,
        method_name: str,
        kwargs: dict[str, Any],
    ) -> None:
        """Reject invalid HDI probabilities across public reporting methods."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data))
        result = self._experiment(data)

        with pytest.raises(ValueError, match="hdi_prob must be in"):
            getattr(result, method_name)(hdi_prob=0, **kwargs)

    @pytest.mark.parametrize(
        "case",
        ["complete", "missing-posterior", "no-random-att", "unrounded"],
    )
    def test_summary_prints_hierarchical_sections(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        case: str,
    ) -> None:
        """Validate that summary reports hierarchical model quantities."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data))
        result = self._experiment(data)
        idata = result.idata
        assert idata is not None
        if case == "missing-posterior":
            idata["posterior"] = idata.posterior.to_dataset().drop_vars(
                ["sigma_fixed", "sigma_random", "beta_fixed", "beta_random"]
            )
        elif case == "no-random-att":
            result.random_effect_labels = ["1|store_id"]

        result.summary(round_to=None if case == "unrounded" else 2)

        captured = capsys.readouterr().out
        assert "Results:" in captured
        if case == "missing-posterior":
            assert "\nVariance components:\n" not in captured
            assert "\nFixed effects:\n" not in captured
            assert "Group-level ATT deviations:" not in captured
        elif case == "no-random-att":
            assert "\nVariance components:\n" in captured
            assert "\nFixed effects:\n" in captured
            assert "Group-level ATT deviations:" not in captured
        else:
            assert "\nVariance components:\n" in captured
            assert "\nFixed effects:\n" in captured
            assert "Group-level ATT deviations:" in captured
        assert "Model coefficients:" not in captured

    def test_effect_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return a scalar posterior summary for the population ATT."""
        data = self._panel_data()
        matrices = self._matrices(data)
        self._patch_parse_formula(monkeypatch, matrices)
        beta_fixed = np.zeros((1, 2, 1, len(matrices.metadata["fixed_effect_names"])))
        did_idx = matrices.metadata["fixed_effect_names"].index(
            "post_treatment:treated"
        )
        beta_fixed[0, :, 0, did_idx] = [1.0, 2.0]
        result = self._experiment(
            data,
            model=_FixedPosteriorHDiDModel(beta_fixed=beta_fixed),
        )

        summary = result.effect_summary(
            direction="two-sided",
            alpha=0.1,
            min_effect=0.5,
        )

        assert summary.table.index.tolist() == ["treatment_effect"]
        assert {
            "mean",
            "median",
            "hdi_lower",
            "hdi_upper",
            "p_two_sided",
            "prob_of_effect",
            "p_rope",
        } <= set(summary.table.columns)
        assert summary.text

    def test_build_delegates_to_hierarchical_regression(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that fitting delegates to hierarchical regression inputs."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data))
        model = _FixedPosteriorHDiDModel()
        result = self._experiment(data, model=model, non_centered=False)

        assert model.build_kwargs["X"] is result.X
        assert model.build_kwargs["Z"] is result.Z
        assert model.build_kwargs["y"] is result.y
        np.testing.assert_array_equal(model.build_kwargs["group_idx"], result.group_idx)
        assert model.build_kwargs["coords"] is result.coords
        assert model.build_kwargs["non_centered"] is False

    @pytest.mark.integration
    def test_real_hierarchical_regression_backend_with_mocked_sampling(
        self, mock_pymc_sample: None
    ) -> None:
        """Validate the default backend and small-sample warnings."""
        data = self._panel_data()
        selected_stores = ["s1", "s2", "s6", "s7", "s8"]
        data = data[
            data["store_id"].isin(selected_stores)
            & data["customer_id"].str.endswith("_c0")
        ].copy()

        with pytest.warns(UserWarning) as warnings:
            result = HierarchicalDifferenceInDifferences(
                data=data,
                formula=(
                    "y ~ 1 + post_treatment + treated + post_treatment:treated "
                    "+ (post_treatment:treated | store_id)"
                ),
                time_variable_name="month",
                unit_variable_name="customer_id",
            )

        result.fit(draws=4, tune=4, chains=1, progressbar=False, random_seed=42)
        assert isinstance(result.model, HierarchicalLinearRegression)
        warning_messages = [str(warning.message) for warning in warnings]
        assert any("fewer than 10" in message for message in warning_messages)
        assert any(
            "fewer than 5 observations" in message for message in warning_messages
        )
        assert result.did_term == "post_treatment:treated"
        assert result.result.y_pred_counterfactual is not None
        assert result.result.causal_impact.dims == ("chain", "draw", "treated_units")
        idata = result.idata
        assert idata is not None
        assert {
            "beta_fixed",
            "beta_random",
            "sigma_fixed",
            "sigma_random",
            "mu",
        } <= set(idata.posterior.data_vars)

    def test_validation_rejects_unbalanced_panel_before_fit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that unbalanced panels are rejected before fitting."""
        data = self._panel_data().iloc[:-1].copy()
        self._patch_parse_formula(monkeypatch, self._matrices(data))

        with pytest.raises(DataException, match="balanced panel"):
            self._experiment(data)

    def test_validation_rejects_duplicate_unit_time_observations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that every unit has at most one observation per period."""
        data = self._panel_data()
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        self._patch_parse_formula(monkeypatch, self._matrices(data))

        with pytest.raises(
            DataException, match="exactly one observation per unit and time period"
        ):
            self._experiment(data)

    def test_validation_rejects_treatment_changes_within_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that group-level treatment assignment is time invariant."""
        data = self._panel_data()
        data.loc[(data["store_id"] == "s1") & (data["month"] == 2), "treated"] = 1
        self._patch_parse_formula(monkeypatch, self._matrices(data))

        with pytest.raises(DataException, match="must remain constant"):
            self._experiment(data)

    def test_validation_rejects_inconsistent_post_values_within_period(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that each period has one shared pre/post status."""
        data = self._panel_data()
        data.loc[data.index[0], "post_treatment"] = 1
        self._patch_parse_formula(monkeypatch, self._matrices(data))

        with pytest.raises(DataException, match="one value within each time period"):
            self._experiment(data)

    def test_validation_rejects_post_to_pre_transition(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that the post indicator cannot return to pre-treatment."""
        data = self._panel_data()
        third_period = data.loc[data["month"] == 2].copy()
        third_period["month"] = 3
        third_period["post_treatment"] = 0
        data = pd.concat([data, third_period], ignore_index=True)
        self._patch_parse_formula(monkeypatch, self._matrices(data))

        with pytest.raises(DataException, match="single transition"):
            self._experiment(data)

    @pytest.mark.parametrize(
        ("case", "match"),
        [
            ("missing-column", "missing required HDiD columns"),
            ("nonnumeric-outcome", "must be numeric"),
            ("missing-time", "contains missing values"),
            ("missing-outcome", "contains missing values"),
            ("group-switching", "must belong to exactly one"),
            ("staggered-adoption", "Staggered adoption is not supported"),
            ("no-post-period", "Both pre- and post-treatment periods"),
            ("too-few-groups", "at least 3 random-effects groups"),
            ("too-few-observations", "at least 2 observations"),
        ],
        ids=[
            "missing-column",
            "nonnumeric-outcome",
            "missing-time",
            "missing-outcome",
            "group-switching",
            "staggered-adoption",
            "no-post-period",
            "too-few-groups",
            "too-few-observations",
        ],
    )
    def test_data_validation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case: str,
        match: str,
    ) -> None:
        """Reject malformed panel data before fitting the model."""
        data = self._panel_data()
        if case == "missing-column":
            matrices = self._matrices(data)
            data = data.drop(columns=["month"])
        else:
            if case == "nonnumeric-outcome":
                data["y"] = "invalid"
            elif case == "missing-time":
                data.loc[data.index[0], "month"] = np.nan
            elif case == "missing-outcome":
                data.loc[data.index[0], "y"] = np.nan
            elif case == "group-switching":
                data.loc[
                    (data["customer_id"] == "s1_c0") & (data["month"] == 2),
                    "store_id",
                ] = "s2"
            elif case == "staggered-adoption":
                third_period = data.loc[data["month"] == 2].copy()
                third_period["month"] = 3
                data = pd.concat([data, third_period], ignore_index=True)
                data.loc[
                    (data["store_id"] == "s6") & (data["month"] == 2),
                    "post_treatment",
                ] = 0
            elif case == "no-post-period":
                data["post_treatment"] = 0
            elif case == "too-few-groups":
                data = data[data["store_id"].isin(["s1", "s6"])].copy()
            elif case == "too-few-observations":
                data = data.groupby("store_id", observed=True).head(1).copy()
            matrices = self._matrices(data)
        self._patch_parse_formula(monkeypatch, matrices)

        with pytest.raises(DataException, match=match):
            self._experiment(data)

    @pytest.mark.parametrize(
        ("column", "invalid_value"),
        [
            ("treated", 0.5),
            ("post_treatment", 0.5),
            ("treated", np.nan),
            ("post_treatment", np.nan),
        ],
    )
    def test_validation_rejects_invalid_binary_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
        column: str,
        invalid_value: float,
    ) -> None:
        """Validate that binary indicators contain only complete zero-one values."""
        data = self._panel_data()
        data[column] = data[column].astype(float)
        data.loc[data.index[0], column] = invalid_value
        self._patch_parse_formula(monkeypatch, self._matrices(data))

        with pytest.raises(DataException, match="must be binary-coded"):
            self._experiment(data)

    @pytest.mark.parametrize(
        ("case", "match"),
        [
            ("missing-interaction", "exactly one DiD interaction"),
            ("rank-deficient", "rank deficient"),
        ],
        ids=["missing-interaction", "rank-deficient"],
    )
    def test_formula_validation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case: str,
        match: str,
    ) -> None:
        """Reject invalid fixed-effects designs."""
        data = self._panel_data()
        matrices = self._matrices(data, include_did=case != "missing-interaction")
        if case == "rank-deficient":
            matrices.rhs["treated_copy"] = matrices.rhs["treated"]
            matrices.metadata["fixed_effect_names"] = list(matrices.rhs.columns)
        self._patch_parse_formula(monkeypatch, matrices)

        with pytest.raises(FormulaException, match=match):
            self._experiment(data)


@pytest.fixture
def unfitted_hdid():
    """A small real hierarchical graph, with sampling controlled by each test."""
    helpers = TestHierarchicalDifferenceInDifferencesInterface
    return helpers._experiment(
        helpers._panel_data(),
        model=HierarchicalLinearRegression(
            sample_kwargs={
                "draws": 4,
                "tune": 4,
                "chains": 1,
                "progressbar": False,
                "random_seed": 42,
            },
            prior_sample_kwargs={"draws": 4, "random_seed": 42},
        ),
        fit=False,
    )


def test_hdid_construction_is_lazy_and_build_is_idempotent(unfitted_hdid):
    experiment = unfitted_hdid
    assert experiment.idata is None
    assert not experiment.model.basic_RVs
    assert not experiment.is_fitted
    assert not experiment.has_prior_predictive
    assert experiment.build() is experiment
    variables = tuple(experiment.model.basic_RVs)
    assert variables
    assert experiment.build() is experiment
    assert tuple(experiment.model.basic_RVs) == variables
    assert experiment.idata is None


@pytest.mark.parametrize(
    "method_name",
    [
        "plot",
        "get_plot_data",
        "summary",
        "effect_summary",
        "plot_group_effects",
        "plot_variance_components",
    ],
)
@pytest.mark.parametrize("group", ["prior", "posterior"])
def test_hdid_readers_require_the_requested_phase(unfitted_hdid, method_name, group):
    with pytest.raises(GroupNotSampledException):
        getattr(unfitted_hdid, method_name)(group=group)


def test_hdid_sampling_phases_refit_and_model_replacement(
    unfitted_hdid, mock_pymc_sample
):
    experiment = unfitted_hdid
    assert experiment.sample_prior_predictive() is experiment
    prior = experiment.prior_result
    assert experiment.has_prior_predictive
    assert not experiment.is_fitted
    xr.testing.assert_identical(
        prior.causal_impact,
        experiment.model.require_group("prior")["beta_fixed"].sel(
            coeffs=experiment.did_term
        ),
    )
    assert "Prior predictive check" in experiment.effect_summary(group="prior").text
    with pytest.raises(GroupNotSampledException):
        experiment.effect_summary()
    fig, ax = experiment.plot(group="prior", show=False)
    assert "Prior population effect" in ax.get_title()
    assert not any(text.get_text() == "Fitted treated\nsample gap" for text in ax.texts)
    plt.close(fig)
    assert "y_counterfactual" in experiment.get_plot_data(group="prior")

    assert experiment.fit() is experiment
    posterior = experiment.result
    assert experiment.prior_result is prior
    xr.testing.assert_identical(
        posterior.causal_impact,
        experiment.model.require_group("posterior")["beta_fixed"].sel(
            coeffs=experiment.did_term
        ),
    )
    experiment.sample_prior_predictive(draws=5)
    assert experiment.result is posterior
    refreshed_prior = experiment.prior_result
    assert refreshed_prior is not prior
    with pytest.warns(UserWarning, match="Refitting"):
        experiment.fit()
    assert experiment.result is not posterior
    assert experiment.prior_result is refreshed_prior

    experiment.model = HierarchicalLinearRegression()
    assert not experiment.is_fitted
    assert not experiment.has_prior_predictive
    assert experiment.idata is None
    with pytest.raises(GroupNotSampledException):
        _ = experiment.result
    with pytest.raises(GroupNotSampledException):
        _ = experiment.prior_result


@pytest.mark.parametrize("random_modifier", [False, True])
def test_hdid_rejects_unsupported_treatment_modifiers(random_modifier):
    data = TestHierarchicalDifferenceInDifferencesInterface._panel_data()
    data["x"] = np.arange(len(data)) % 3
    fixed = "y ~ 1 + post_treatment * treated + x"
    random = "(post_treatment:treated | store_id)"
    if random_modifier:
        random = "(post_treatment:treated + post_treatment:treated:x | store_id)"
    else:
        fixed += " + post_treatment:treated:x"
    with pytest.raises(FormulaException, match="treatment-effect modifiers"):
        HierarchicalDifferenceInDifferences(
            data, fixed + " + " + random, "month", "customer_id"
        )


@pytest.mark.parametrize("changed_input", ["Z", "group_idx", "non_centered"])
def test_hlr_rebuild_checks_hierarchical_inputs(unfitted_hdid, changed_input):
    experiment = unfitted_hdid.build()
    kwargs = {
        "X": experiment.X,
        "Z": experiment.Z,
        "y": experiment.y,
        "group_idx": experiment.group_idx,
        "coords": experiment.coords,
        "non_centered": experiment.non_centered,
    }
    if changed_input == "Z":
        kwargs["Z"] = experiment.Z + 1
    elif changed_input == "group_idx":
        kwargs["group_idx"] = experiment.group_idx[::-1]
    else:
        kwargs["non_centered"] = not experiment.non_centered
    with pytest.raises(RuntimeError, match="different inputs"):
        experiment.model.build(**kwargs)


@pytest.mark.parametrize("method", ["predict", "score"])
def test_hlr_rejects_prediction_before_changing_design(unfitted_hdid, method):
    experiment = unfitted_hdid.build()
    original = experiment.model["X"].get_value().copy()
    args = [experiment.X.isel(obs_ind=slice(None, None, -1))]
    if method == "score":
        args.append(experiment.y)
    with pytest.raises(
        NotImplementedError, match="joint updates of X, Z and group_idx"
    ):
        getattr(experiment.model, method)(*args)
    np.testing.assert_array_equal(experiment.model["X"].get_value(), original)


@pytest.mark.parametrize("method", ["sample_prior_predictive", "fit"])
def test_hlr_sampling_restores_hierarchical_training_data(
    unfitted_hdid, mock_pymc_sample, method
):
    experiment = unfitted_hdid.build()
    model = experiment.model
    original = {
        name: model[name].get_value().copy() for name in ("X", "Z", "y", "group_idx")
    }
    with model:
        pm.set_data({name: np.zeros_like(value) for name, value in original.items()})

    getattr(experiment, method)()

    for name, value in original.items():
        np.testing.assert_array_equal(model[name].get_value(), value)
