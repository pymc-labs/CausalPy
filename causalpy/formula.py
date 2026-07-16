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
from dataclasses import dataclass
from typing import Any

import pandas as pd


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
        elements = set(self._elements)
        return "0" not in elements and "-1" not in elements

    @property
    def slopes(self) -> list[str]:
        return [element for element in self._elements if element not in ("0", "-1", "1")]

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
        return self.lhs

    @property
    def X(self) -> pd.DataFrame:
        return self.rhs

    @property
    def model_spec(self):
        return self.metadata["model_spec"]

    @property
    def fixed_model_spec(self):
        return self.metadata["fixed_model_spec"]

    @property
    def random_model_spec(self):
        return self.metadata["random_model_spec"]
