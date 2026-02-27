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
"""Tests for cross-cutting sensitivity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.checks.base import Check, CheckResult
from causalpy.checks.convex_hull import ConvexHullCheck
from causalpy.checks.persistence import PersistenceCheck
from causalpy.checks.pre_treatment_placebo import PreTreatmentPlaceboCheck
from causalpy.checks.prior_sensitivity import PriorSensitivity
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.experiments.synthetic_control import SyntheticControl
from causalpy.pipeline import PipelineContext

# ---------------------------------------------------------------------------
# ConvexHullCheck tests
# ---------------------------------------------------------------------------


class TestConvexHullCheck:
    """Tests for ConvexHullCheck (Synthetic Control)."""

    def test_satisfies_check_protocol(self):
        assert isinstance(ConvexHullCheck(), Check)

    def test_applicable_methods(self):
        assert SyntheticControl in ConvexHullCheck().applicable_methods

    def test_validate_rejects_non_sc(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({"t": np.arange(n), "y": np.random.normal(size=n)})
        model = cp.create_causalpy_compatible_class(LinearRegression())
        its = InterruptedTimeSeries(
            df, treatment_time=70, formula="y ~ 1 + t", model=model
        )
        with pytest.raises(TypeError, match="SyntheticControl"):
            ConvexHullCheck().validate(its)

    def test_run_on_sc(self):
        df = cp.load_data("sc")
        model = cp.create_causalpy_compatible_class(LinearRegression())
        sc = SyntheticControl(
            df,
            treatment_time=70,
            control_units=["a", "b", "c", "d", "e", "f", "g"],
            treated_units=["actual"],
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = sc

        check = ConvexHullCheck()
        check.validate(sc)
        result = check.run(sc, ctx)

        assert isinstance(result, CheckResult)
        assert result.check_name == "ConvexHullCheck"
        assert result.passed is not None
        assert result.table is not None


# ---------------------------------------------------------------------------
# PersistenceCheck tests
# ---------------------------------------------------------------------------


class TestPersistenceCheck:
    """Tests for PersistenceCheck (three-period ITS)."""

    def test_satisfies_check_protocol(self):
        assert isinstance(PersistenceCheck(), Check)

    def test_applicable_methods(self):
        assert InterruptedTimeSeries in PersistenceCheck().applicable_methods

    def test_validate_rejects_two_period_its(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({"t": np.arange(n), "y": np.random.normal(size=n)})
        model = cp.create_causalpy_compatible_class(LinearRegression())
        its = InterruptedTimeSeries(
            df, treatment_time=70, formula="y ~ 1 + t", model=model
        )
        with pytest.raises(ValueError, match="three-period"):
            PersistenceCheck().validate(its)

    def test_run_on_three_period_its(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({"t": np.arange(n), "y": np.random.normal(size=n)})
        model = cp.create_causalpy_compatible_class(LinearRegression())
        its = InterruptedTimeSeries(
            df,
            treatment_time=50,
            treatment_end_time=70,
            formula="y ~ 1 + t",
            model=model,
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = its

        check = PersistenceCheck()
        check.validate(its)
        result = check.run(its, ctx)

        assert isinstance(result, CheckResult)
        assert result.check_name == "PersistenceCheck"
        assert result.passed is None
        assert result.table is not None
        assert "persistence" in result.metadata


# ---------------------------------------------------------------------------
# PreTreatmentPlaceboCheck tests
# ---------------------------------------------------------------------------


class TestPreTreatmentPlaceboCheck:
    """Tests for PreTreatmentPlaceboCheck (Staggered DiD)."""

    def test_satisfies_check_protocol(self):
        assert isinstance(PreTreatmentPlaceboCheck(), Check)

    def test_applicable_methods(self):
        from causalpy.experiments.staggered_did import (
            StaggeredDifferenceInDifferences,
        )

        assert (
            StaggeredDifferenceInDifferences
            in PreTreatmentPlaceboCheck().applicable_methods
        )

    def test_validate_rejects_non_staggered_did(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({"t": np.arange(n), "y": np.random.normal(size=n)})
        model = cp.create_causalpy_compatible_class(LinearRegression())
        its = InterruptedTimeSeries(
            df, treatment_time=70, formula="y ~ 1 + t", model=model
        )
        with pytest.raises(TypeError, match="StaggeredDifferenceInDifferences"):
            PreTreatmentPlaceboCheck().validate(its)


# ---------------------------------------------------------------------------
# PriorSensitivity tests
# ---------------------------------------------------------------------------


class TestPriorSensitivity:
    """Tests for PriorSensitivity check."""

    def test_satisfies_check_protocol(self):
        check = PriorSensitivity(
            alternatives=[{"name": "test", "model": LinearRegression()}]
        )
        assert isinstance(check, Check)

    def test_empty_alternatives_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            PriorSensitivity(alternatives=[])

    def test_missing_keys_raises(self):
        with pytest.raises(ValueError, match="'name' and 'model'"):
            PriorSensitivity(alternatives=[{"name": "test"}])

    def test_validate_rejects_ols_model(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({"t": np.arange(n), "y": np.random.normal(size=n)})
        model = cp.create_causalpy_compatible_class(LinearRegression())
        its = InterruptedTimeSeries(
            df, treatment_time=70, formula="y ~ 1 + t", model=model
        )
        check = PriorSensitivity(
            alternatives=[{"name": "test", "model": LinearRegression()}]
        )
        with pytest.raises(TypeError, match="Bayesian"):
            check.validate(its)

    def test_applicable_to_all_experiment_types(self):
        check = PriorSensitivity(
            alternatives=[{"name": "test", "model": LinearRegression()}]
        )
        assert len(check.applicable_methods) == 9
