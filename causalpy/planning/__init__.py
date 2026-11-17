# Copyright 2022 - 2026 The PyMC Labs Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""History-only planning for synthetic control.

The primary group estimand is the draw-aligned sum of unit-level posterior
conditional-mean impacts. Averages per period and per treated unit are
transforms of that sum. A raw sum of treated outcomes is not comparable to a
simplex donor combination, and summing marginal intervals is not group
inference. An aggregate-first shared-weight model is a different estimator
and is not implemented here.

:class:`HistoryOnlyPlan` fits the ordinary synthetic-control analysis once
per horizon and translates effects without refitting. A registered adapter,
or one passed in, is called instead. Construction does not sample.

Group fitting, augmented synthetic control, and interactive fixed effects
are the issue #1440 seam. Power is the issue #1439 seam. Candidate screening
is the issue #1441 seam. Grouped treatment search is the issue #319 seam:
:func:`search` screens with :class:`CandidateGenerator` and evaluates power
through :class:`causalpy.planning.power.ReferencePowerEvaluator` or by
translating :class:`HistoryOnlyPlan` fits. It freezes one explicit selection
and does not add a second scoring implementation. :class:`CandidateRequest`
stores the screen. It does not enumerate or fit.
"""

from causalpy.planning.contracts import (
    CANDIDATE_COLUMNS,
    CANDIDATE_FILTER_REASONS,
    DRAW_ALIGNED_UNIT_SUM,
    FIXED_DESIGN_FIT_CEILING,
    REFERENCE_CAPABILITIES,
    REJECTED_ESTIMATOR_MAPPINGS,
    RELATIVE_BASELINE_CONVENTION,
    VALIDATION_MATRIX,
    AnalysisDecision,
    BalanceConstraint,
    CandidateRequest,
    CandidateTable,
    CanonicalEvaluation,
    CanonicalFit,
    Capabilities,
    DataBoundary,
    DecisionRule,
    EffectScenario,
    EstimandDraws,
    EstimandSummary,
    FitDiagnostics,
    PlanningResult,
    PowerCriterion,
    PowerRequest,
    Provenance,
    RepeatedExperimentPower,
    ValidationCell,
    WindowRequest,
    aggregate_unit_impacts,
    decide,
    require_relative_baseline,
    unit_period_effect,
)
from causalpy.planning.reference import (
    AnalysisHandoff,
    HistoryOnlyPlan,
    SyntheticControlAdapter,
)
from causalpy.planning.report import evaluation_table, plot_evaluation
from causalpy.planning.seams import (
    OWNERSHIP,
    CandidateGenerator,
    EstimatorAdapter,
    PowerEvaluator,
)
from causalpy.planning.search import (
    EVALUATION_ORDER,
    GROUPED_DATA_SEED_SCHEME,
    GROUPED_MECHANISM,
    SEARCH_COLUMNS,
    SELECTION_OBJECTIVE,
    AuditReport,
    ComputationalBudget,
    FrozenDesign,
    ScreeningComparison,
    SearchResult,
    SelectionValidation,
    compare_screening,
    plot_search,
    sampler_seed,
    search,
    validate_selection,
)

__all__ = [
    "CANDIDATE_COLUMNS",
    "CANDIDATE_FILTER_REASONS",
    "DRAW_ALIGNED_UNIT_SUM",
    "EVALUATION_ORDER",
    "FIXED_DESIGN_FIT_CEILING",
    "GROUPED_DATA_SEED_SCHEME",
    "GROUPED_MECHANISM",
    "OWNERSHIP",
    "REFERENCE_CAPABILITIES",
    "REJECTED_ESTIMATOR_MAPPINGS",
    "RELATIVE_BASELINE_CONVENTION",
    "SEARCH_COLUMNS",
    "SELECTION_OBJECTIVE",
    "VALIDATION_MATRIX",
    "AnalysisDecision",
    "AnalysisHandoff",
    "AuditReport",
    "BalanceConstraint",
    "CandidateGenerator",
    "CandidateRequest",
    "CandidateTable",
    "CanonicalEvaluation",
    "CanonicalFit",
    "Capabilities",
    "ComputationalBudget",
    "DataBoundary",
    "DecisionRule",
    "EffectScenario",
    "EstimandDraws",
    "EstimandSummary",
    "EstimatorAdapter",
    "FitDiagnostics",
    "FrozenDesign",
    "HistoryOnlyPlan",
    "PlanningResult",
    "PowerCriterion",
    "PowerEvaluator",
    "PowerRequest",
    "Provenance",
    "RepeatedExperimentPower",
    "ScreeningComparison",
    "SearchResult",
    "SelectionValidation",
    "SyntheticControlAdapter",
    "ValidationCell",
    "WindowRequest",
    "aggregate_unit_impacts",
    "compare_screening",
    "decide",
    "evaluation_table",
    "plot_evaluation",
    "plot_search",
    "require_relative_baseline",
    "sampler_seed",
    "search",
    "unit_period_effect",
    "validate_selection",
]
