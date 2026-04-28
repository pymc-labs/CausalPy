#   Copyright 2026 - 2026 The PyMC Labs Developers
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
import re
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from formulaic import Formula

_RE_RANDOM_COMPONENT = re.compile(r"\(\s*([^|()]+?)\s*\|\s*([^|()]+?)\s*\)")


def _canonical_component_name(name: Any) -> str:
    """Normalize Formulaic intercept labels to stable naming."""
    text = str(name).strip()
    return "1" if text == "Intercept" else text


def _assert_unique_names(names: list[str], *, context: str) -> None:
    """Raise an explicit error when normalized names collide."""
    if len(set(names)) != len(names):
        raise ValueError(
            f"Duplicate {context} names detected after normalization: {names}"
        )


def _build_dataframe(
    matrix: Any,
    context: str,
    rename: Callable[[str], str],
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Build a DataFrame with normalized, unique column names."""
    frame = pd.DataFrame(matrix, copy=True)
    frame.columns = [rename(_canonical_component_name(col)) for col in frame.columns]
    _assert_unique_names(list(frame.columns), context=context)
    if index is not None:
        frame.index = index
    return frame


@dataclass(frozen=True, slots=True)
class RandomComponent:
    """
    Parsed random-effects component using lme4 grouping syntax ``(expr | group)``.

    Examples
    --------
    >>> rc = RandomComponent(expr="1 + x1", grouping="store_id")
    >>> str(rc)
    '(1 + x1 | store_id)'
    >>> rc.formula_rhs
    '1 + x1'
    """

    expr: str
    grouping: str

    @property
    def _elements(self) -> list[str]:
        return [element.strip() for element in self.expr.split("+") if element.strip()]

    @property
    def has_intercept(self) -> bool:
        """Whether this component includes a random intercept."""
        elements = set(self._elements)
        return "0" not in elements and "-1" not in elements

    @property
    def slopes(self) -> list[str]:
        """Return non-intercept random-slope elements."""
        return [
            element for element in self._elements if element not in ("0", "-1", "1")
        ]

    @property
    def formula_rhs(self) -> str:
        """Return the Formulaic RHS equivalent of the random expression."""
        rhs_terms: list[str] = []
        if self.has_intercept:
            rhs_terms.append("1")
        rhs_terms.extend(self.slopes)
        return " + ".join(rhs_terms) if rhs_terms else "0"

    def __str__(self) -> str:
        return f"({self.expr} | {self.grouping})"


@dataclass(frozen=True, slots=True)
class MixedModelMatrices:
    """
    Mixed-model matrix container with Formulaic-compatible ``lhs``/``rhs`` access.

    Implements ``lhs``/``rhs`` and ``model_spec`` while extending with mixed-effects
    attributes (`Z`, `metadata`, `fixed_model_spec`, `random_model_spec`).

    Examples
    --------
    >>> data = pd.DataFrame(
    ...     {
    ...         "y": [0.1, -0.2, 0.3, 0.0],
    ...         "x1": [1.0, 0.0, 1.0, 0.5],
    ...         "store_id": ["s1", "s1", "s2", "s2"],
    ...     }
    ... )
    >>> mm = parse_formula("y ~ 1 + x1 + (1 + x1 | store_id)", data)
    >>> mm.lhs.shape[0] == mm.rhs.shape[0] == mm.Z.shape[0]
    True
    >>> mm.metadata["group"]["variable"]
    'store_id'
    """

    lhs: pd.DataFrame
    rhs: pd.DataFrame
    Z: pd.DataFrame
    metadata: dict[str, Any]

    @property
    def y(self) -> pd.DataFrame:
        """Alias for ``lhs`` outcome matrix."""
        return self.lhs

    @property
    def X(self) -> pd.DataFrame:
        """Alias for ``rhs`` fixed-effects design matrix."""
        return self.rhs

    @property
    def model_spec(self):
        """Fixed-effects Formulaic model specification used for ``rhs``."""
        return self.metadata["model_spec"]

    @property
    def fixed_model_spec(self):
        """Explicit fixed-effects model specification alias."""
        return self.metadata["fixed_model_spec"]

    @property
    def random_model_spec(self):
        """Random-effects model specification used for ``Z``."""
        return self.metadata["random_model_spec"]


@dataclass(frozen=True, slots=True)
class MixedModelFormula:
    """
    Parsed mixed-effects formula object produced by ``Parser.parse(...)``.

    This lightweight representation stores:
    - ``lhs``: left-hand side variable name
    - ``rhs``: fixed-effects right-hand side expression
    - ``random_components``: parsed lme4-style ``(expr | group)`` components

    ``lhs``/``rhs`` naming intentionally mirrors Formulaic's
    structured formula conventions for easier migration.

    ``get_model_matrix(...)`` materializes aligned pandas ``lhs``, ``rhs``, and
    ``Z`` matrices using Formulaic and returns them in ``MixedModelMatrices``
    with minimal mixed-effects metadata.

    Examples
    --------
    >>> mf = Parser.parse("y ~ 1 + x1 + (1 + x1 | store_id)")
    >>> mf.lhs, mf.rhs
    ('y', '1 + x1')
    >>> mf.has_random_effects
    True
    """

    lhs: str
    rhs: str
    random_components: tuple[RandomComponent, ...]
    _formula: str

    @property
    def has_random_effects(self) -> bool:
        """Whether the parsed formula contains random components."""
        return len(self.random_components) > 0

    @property
    def grouping_variables(self) -> list[str]:
        """Unique grouping-variable names in declaration order."""
        return list(
            OrderedDict.fromkeys(
                component.grouping for component in self.random_components
            )
        )

    @property
    def fixed_formula(self) -> str:
        """Fixed-effects formula string built from ``lhs`` and ``rhs``."""
        return f"{self.lhs} ~ {self.rhs}"

    @property
    def formula(self) -> Formula:
        """Formulaic ``Formula`` instance for fixed-effects materialization."""
        return Formula(self.fixed_formula)

    def __str__(self) -> str:
        rhs_parts = [self.rhs] + [
            str(component) for component in self.random_components
        ]
        return f"{self.lhs} ~ {' + '.join(rhs_parts)}"

    def get_model_matrix(
        self,
        data: Any,
        context: Mapping[str, Any] | None = None,
        drop_rows: set[int] | None = None,
        **attr_overrides: Any,
    ) -> MixedModelMatrices:
        """
        Build unified mixed-model matrices (`lhs`, `rhs`, `Z`) and metadata.

        Signature mirrors Formulaic's ``ModelSpec.get_model_matrix(...)``.
        ``attr_overrides`` are forwarded to Formulaic as model-spec overrides.
        """
        grouping_variables = self.grouping_variables
        if len(grouping_variables) > 1:
            raise ValueError(
                f"Current version supports a single grouping variable, got: {grouping_variables}"
            )
        if len(self.random_components) > 1:
            raise ValueError(
                "Current version supports a single random component. "
                f"Found {len(self.random_components)} components: {[str(c) for c in self.random_components]}"
            )

        spec_overrides = dict(attr_overrides)

        # Shared drop_rows keeps lhs/rhs/Z aligned under missing-data dropping.
        shared_drop_rows: set[int] = drop_rows if drop_rows is not None else set()

        fixed_mm = self.formula.get_model_matrix(
            data,
            context=context,
            drop_rows=shared_drop_rows,
            **spec_overrides,
        )

        rhs = _build_dataframe(
            fixed_mm.rhs,
            context="fixed-effect",
            rename=lambda name: name,
        )

        lhs = pd.DataFrame(fixed_mm.lhs, copy=True)
        if lhs.shape[1] != 1:
            raise ValueError(
                f"Expected a single outcome column for mixed regression, got {lhs.shape[1]}."
            )
        lhs.columns = [self.lhs]

        fixed_model_spec = fixed_mm.rhs.model_spec

        Z = pd.DataFrame(index=rhs.index)
        random_model_spec = None
        random_effect_names: list[str] = []
        group: dict[str, np.ndarray | int | list[Any] | str | None] = {
            "variable": None,
            "labels": [],
            "n_groups": 0,
            "idx": np.asarray([], dtype=np.int32),
            "components": [],
        }

        if self.has_random_effects:
            grouping = grouping_variables[0]
            if grouping not in data.columns:
                raise ValueError(
                    f"Grouping variable '{grouping}' not found in data. Available columns: {list(data.columns)}"
                )

            # Group metadata must follow the same kept rows as lhs/rhs/Z.
            grouping_values = data.loc[rhs.index, grouping]
            if grouping_values.isna().any():
                raise ValueError(
                    f"Grouping variable '{grouping}' contains missing values."
                )

            group_idx, group_labels = pd.factorize(grouping_values, sort=False)
            group_idx = group_idx.astype(np.int32, copy=False)
            group_labels_list = [str(label) for label in group_labels.tolist()]

            random_component = self.random_components[0]
            random_mm = Formula(random_component.formula_rhs).get_model_matrix(
                data,
                context=context,
                drop_rows=shared_drop_rows,
                **spec_overrides,
            )
            Z = _build_dataframe(
                random_mm,
                context="random-effect",
                rename=lambda name: f"{name}|{random_component.grouping}",
                index=rhs.index,
            )

            random_model_spec = random_mm.model_spec
            random_effect_names = list(Z.columns)
            group = {
                "variable": grouping,
                "labels": group_labels_list,
                "n_groups": int(group_labels.shape[0]),
                "idx": group_idx,
                "components": [str(random_component)],
            }

        metadata = {
            "outcome_name": self.lhs,
            "has_random_effects": self.has_random_effects,
            "fixed_effect_names": list(rhs.columns),
            "random_effect_names": random_effect_names,
            "group": group,
            "model_spec": fixed_model_spec,
            "fixed_model_spec": fixed_model_spec,
            "random_model_spec": random_model_spec,
            "raw_formula": self._formula,
            "fixed_formula": self.fixed_formula,
        }

        return MixedModelMatrices(lhs=lhs, rhs=rhs, Z=Z, metadata=metadata)


class Parser:
    """
    Parser converts a formula string into ``MixedModelFormula``.

    Examples
    --------
    >>> parsed = Parser.parse("y ~ 1 + x1 + (1 | store_id)")
    >>> parsed.fixed_formula
    'y ~ 1 + x1'
    >>> parsed.random_components[0].grouping
    'store_id'
    """

    _RE_PATTERN = _RE_RANDOM_COMPONENT

    @classmethod
    def parse(cls, formula: str) -> MixedModelFormula:
        """
        Parse a mixed-effects formula into fixed and random components.

        Supports lme4-style random syntax ``(expr | group)`` and returns a
        ``MixedModelFormula`` for later materialization. This parser currently
        supports one grouping variable and one random component.
        """
        if "~" not in formula:
            raise ValueError(f"Formula must contain '~': {formula!r}")

        lhs, rhs = formula.split("~", maxsplit=1)
        lhs_name = lhs.strip()
        if not lhs_name:
            raise ValueError(f"Formula must have a left-hand side: {formula!r}")

        random_components = tuple(
            RandomComponent(expr=expr.strip(), grouping=grouping.strip())
            for expr, grouping in cls._RE_PATTERN.findall(rhs)
        )

        grouping_vars = {component.grouping for component in random_components}

        if len(grouping_vars) > 1:
            raise ValueError(
                f"Multiple grouping variables are not supported in current version: {grouping_vars}."
            )
        if len(random_components) > 1:
            raise ValueError(
                "Multiple random components are not supported in current version. "
                f"Found: {[str(t) for t in random_components]}"
            )

        fixed_rhs = cls._RE_PATTERN.sub("", rhs)
        fixed_rhs = re.sub(r"\+\s*\+", "+", fixed_rhs)
        fixed_rhs = fixed_rhs.strip().strip("+").strip()
        if not fixed_rhs:
            fixed_rhs = "1"

        return MixedModelFormula(
            lhs=lhs_name,
            rhs=fixed_rhs,
            random_components=random_components,
            _formula=formula.strip(),
        )


def parse_formula(
    formula: str,
    data: pd.DataFrame,
) -> MixedModelMatrices:
    """Parse and materialize a formula against ``data`` in a single call."""
    return Parser.parse(formula).get_model_matrix(data=data)
