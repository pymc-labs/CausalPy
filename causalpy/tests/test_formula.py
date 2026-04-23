import pytest
import numpy as np

from causalpy.formula import Parser, parse_formula


class TestParser:
    def test_parse_extracts_lhs_rhs_and_random(self):
        mixed_formula = Parser.parse("y ~ 1 + post * treated + (1 + x1 | store_id)")
        assert mixed_formula.lhs == "y"
        assert mixed_formula.rhs == "1 + post * treated"
        assert mixed_formula.fixed_formula == "y ~ 1 + post * treated"
        assert mixed_formula.has_random_effects is True
        assert len(mixed_formula.random_components) == 1
        assert mixed_formula.random_components[0].grouping == "store_id"
        assert mixed_formula.random_components[0].formula_rhs == "1 + x1"

    def test_parse_rejects_multiple_grouping_variables(self):
        with pytest.raises(ValueError, match="Multiple grouping variables"):
            Parser.parse("y ~ x1 + (1 | group) + (1 | store_id)")

    def test_parse_rejects_multiple_random_components(self):
        with pytest.raises(ValueError, match="Multiple random components"):
            Parser.parse("y ~ x1 + (1 | store_id) + (x1 | store_id)")


class TestMixedModelFormula:
    def test_get_model_matrix_fixed_only_returns_empty_z(self, mixed_effect_model_data):
        mixed_formula = Parser.parse("y ~ 1 + post * treated + size")
        matrices = mixed_formula.get_model_matrix(mixed_effect_model_data)

        assert matrices.lhs.shape[0] == matrices.rhs.shape[0]
        assert matrices.Z.shape[0] == matrices.rhs.shape[0]
        assert matrices.Z.shape[1] == 0
        assert matrices.metadata["has_random_effects"] is False
        assert matrices.metadata["group"]["variable"] is None
        assert matrices.metadata["group"]["n_groups"] == 0

    def test_get_model_matrix_single_random_term(self, mixed_effect_model_data):
        mixed_formula = Parser.parse(
            "y ~ 1 + post * treated + size + (1 + x1 | store_id)"
        )
        matrices = mixed_formula.get_model_matrix(mixed_effect_model_data)

        assert matrices.lhs.shape[0] == matrices.rhs.shape[0]
        assert matrices.Z.shape[0] == matrices.rhs.shape[0]
        assert list(matrices.Z.columns) == ["1|store_id", "x1|store_id"]

        assert matrices.metadata["has_random_effects"] is True
        assert matrices.metadata["group"]["variable"] == "store_id"
        assert (
            matrices.metadata["group"]["n_groups"]
            == mixed_effect_model_data["store_id"].nunique()
        )

    def test_get_model_matrix_rejects_missing_group_values(
        self, mixed_effect_model_data
    ):
        broken = mixed_effect_model_data.copy()
        broken.loc[0, "store_id"] = np.nan
        mixed_formula = Parser.parse("y ~ 1 + x1 + (1 | store_id)")

        with pytest.raises(ValueError, match="contains missing values"):
            mixed_formula.get_model_matrix(broken)


class TestMixedModelMatrices:
    def test_aliases_and_core_fields(self, mixed_effect_model_data):
        matrices = parse_formula(
            "y ~ 1 + post * treated + size + (1 + x1 | store_id)",
            mixed_effect_model_data,
        )

        assert matrices.y is matrices.lhs
        assert matrices.X is matrices.rhs
        assert matrices.lhs.shape[0] == matrices.rhs.shape[0]
        assert matrices.rhs.shape[0] == matrices.Z.shape[0]

    def test_model_spec_fields_exist(self, mixed_effect_model_data):
        matrices = parse_formula(
            "y ~ 1 + post * treated + size + (1 + x1 | store_id)",
            mixed_effect_model_data,
        )

        assert "model_spec" in matrices.metadata
        assert "fixed_model_spec" in matrices.metadata
        assert "random_model_spec" in matrices.metadata
        assert matrices.model_spec is matrices.metadata["model_spec"]
        assert matrices.metadata["model_spec"] is matrices.metadata["fixed_model_spec"]
        assert "group" in matrices.metadata
