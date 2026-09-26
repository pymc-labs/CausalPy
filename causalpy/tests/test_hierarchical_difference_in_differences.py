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

from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from arviz import InferenceData

from causalpy.custom_exceptions import DataException, FormulaException
from causalpy.experiments import hierarchical_difference_in_differences as hdid_module
from causalpy.experiments.hierarchical_difference_in_differences import (
    HierarchicalDifferenceInDifferences,
)
from causalpy.pymc_models import HierarchicalLinearRegression, PyMCModel


class _FixedPosteriorHDiDModel(PyMCModel):
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

    def fit(self, **kwargs: Any) -> None:  # type: ignore[override]
        self.fit_kwargs = kwargs
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
        self.idata = InferenceData(posterior=posterior)


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
        model: PyMCModel | None = None,
        non_centered: bool = True,
    ) -> HierarchicalDifferenceInDifferences:
        if model is None:
            model = _FixedPosteriorHDiDModel()
        return HierarchicalDifferenceInDifferences(
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

        assert result.icc is not None
        xr.testing.assert_allclose(result.icc, expected)

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

        result._compute_icc()

        assert result.icc is None

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
        xr.testing.assert_identical(result.att, expected)
        xr.testing.assert_identical(result.causal_impact, expected)

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
        xr.testing.assert_identical(result.group_effects, expected)

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
            recorded["plot_data"] = plot_data
            recorded["kwargs"] = kwargs
            _, ax = plt.subplots()
            return np.array([ax])

        monkeypatch.setattr(hdid_module.az, "plot_forest", mock_plot_forest)

        fig, ax = result.plot_group_effects(
            random_coeff=random_coeff,
            hdi_prob=0.8,
            combined=False,
            show=False,
        )

        plot_data = recorded["plot_data"]
        expected = (
            result.group_effects.sel(random_coeffs=random_coeff, drop=True)
            .squeeze("treated_units", drop=True)
            .rename("group_deviation")
        )
        xr.testing.assert_identical(plot_data["group_deviation"], expected)
        assert plot_data["group_deviation"].dims == ("chain", "draw", "groups")
        assert plot_data.coords["groups"].values.tolist() == result.group_labels
        assert recorded["kwargs"]["var_names"] == ["group_deviation"]
        assert recorded["kwargs"]["combined"] is False
        assert recorded["kwargs"]["hdi_prob"] == 0.8
        assert recorded["kwargs"]["figsize"] == (8, 5.0)
        assert isinstance(
            recorded["kwargs"]["labeller"], hdid_module.az.labels.NoVarLabeller
        )
        assert ax.get_xlabel() == "Deviation from population coefficient"
        assert ax.get_title() == "Group-Level Deviations (80% HDI)"
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
            result.model.idata = None
        elif case == "missing-mu":
            del idata.posterior["mu"]
        elif case == "no-counterfactual":
            result.y_pred_counterfactual = None
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
        assert result.y_pred_counterfactual is not None
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

        result._compute_counterfactual()

        if case in {"missing-mu", "missing-beta-fixed"}:
            assert result.y_pred_counterfactual is None
        else:
            assert result.y_pred_counterfactual is not None
            did_design = result.X.sel(coeffs=result.did_term)
            expected = idata.posterior["mu"] - (
                idata.posterior["beta_fixed"].sel(coeffs=result.did_term) * did_design
            ).transpose(*idata.posterior["mu"].dims)
            xr.testing.assert_allclose(result.y_pred_counterfactual, expected)

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
            result.y_pred_counterfactual = None

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
            result.icc = None
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
            idata.posterior = idata.posterior.drop_vars(
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

    def test_fit_model_delegates_to_hierarchical_regression(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that fitting delegates to hierarchical regression inputs."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data))
        model = _FixedPosteriorHDiDModel()
        result = self._experiment(data, model=model, non_centered=False)

        assert model.fit_kwargs["X"] is result.X
        assert model.fit_kwargs["Z"] is result.Z
        assert model.fit_kwargs["y"] is result.y
        np.testing.assert_array_equal(model.fit_kwargs["group_idx"], result.group_idx)
        assert model.fit_kwargs["coords"] is result.coords
        assert model.fit_kwargs["non_centered"] is False

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

        assert isinstance(result.model, HierarchicalLinearRegression)
        warning_messages = [str(warning.message) for warning in warnings]
        assert any("fewer than 10" in message for message in warning_messages)
        assert any(
            "fewer than 5 observations" in message for message in warning_messages
        )
        assert result.did_term == "post_treatment:treated"
        assert result.y_pred_counterfactual is not None
        assert result.causal_impact.dims == ("chain", "draw", "treated_units")
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
