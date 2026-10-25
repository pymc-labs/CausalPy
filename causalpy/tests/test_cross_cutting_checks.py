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

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.checks.base import Check, CheckResult, clone_model
from causalpy.checks.convex_hull import ConvexHullCheck
from causalpy.checks.persistence import PersistenceCheck
from causalpy.checks.pre_treatment_placebo import PreTreatmentPlaceboCheck
from causalpy.checks.prior_sensitivity import PriorSensitivity
from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries
from causalpy.experiments.staggered_did import StaggeredDifferenceInDifferences
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

    def test_run_detects_convex_hull_violation(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "a": np.random.normal(0, 0.1, n),
                "b": np.random.normal(0, 0.1, n),
                "treated": np.random.normal(10, 0.1, n),
            }
        )
        model = cp.create_causalpy_compatible_class(LinearRegression())
        sc = SyntheticControl(
            df,
            treatment_time=70,
            control_units=["a", "b"],
            treated_units=["treated"],
            model=model,
        )
        ctx = PipelineContext(data=df)
        result = ConvexHullCheck().run(sc, ctx)
        assert result.passed is False
        assert "failed" in result.text
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
        ).fit()
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

    def test_run_missing_experiment_config_raises(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({"t": np.arange(n), "y": np.random.normal(size=n)})
        model = cp.create_causalpy_compatible_class(LinearRegression())
        its = InterruptedTimeSeries(
            df, treatment_time=70, formula="y ~ 1 + t", model=model
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = its
        check = PriorSensitivity(
            alternatives=[
                {
                    "name": "alt",
                    "model": cp.create_causalpy_compatible_class(LinearRegression()),
                }
            ]
        )
        with pytest.raises(RuntimeError, match="experiment_config"):
            check.run(its, ctx)

    def test_run_with_ols_alternatives(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({"t": np.arange(n), "y": np.random.normal(size=n)})
        model = cp.create_causalpy_compatible_class(LinearRegression())
        its = InterruptedTimeSeries(
            df, treatment_time=70, formula="y ~ 1 + t", model=model
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = its
        ctx.experiment_config = {
            "method": InterruptedTimeSeries,
            "treatment_time": 70,
            "formula": "y ~ 1 + t",
            "model": model,
        }
        check = PriorSensitivity(
            alternatives=[
                {
                    "name": "alt1",
                    "model": cp.create_causalpy_compatible_class(LinearRegression()),
                },
                {
                    "name": "alt2",
                    "model": cp.create_causalpy_compatible_class(LinearRegression()),
                },
            ]
        )
        result = check.run(its, ctx)
        assert isinstance(result, CheckResult)
        assert result.check_name == "PriorSensitivity"
        assert result.passed is None
        assert result.table is not None
        assert len(result.table) == 2

    def test_run_with_failing_effect_summary(self):
        from unittest.mock import patch

        np.random.seed(42)
        n = 100
        df = pd.DataFrame({"t": np.arange(n), "y": np.random.normal(size=n)})
        model = cp.create_causalpy_compatible_class(LinearRegression())
        its = InterruptedTimeSeries(
            df, treatment_time=70, formula="y ~ 1 + t", model=model
        )
        ctx = PipelineContext(data=df)
        ctx.experiment = its
        ctx.experiment_config = {
            "method": InterruptedTimeSeries,
            "treatment_time": 70,
            "formula": "y ~ 1 + t",
            "model": model,
        }
        check = PriorSensitivity(
            alternatives=[
                {
                    "name": "alt",
                    "model": cp.create_causalpy_compatible_class(LinearRegression()),
                }
            ]
        )
        with patch.object(
            InterruptedTimeSeries,
            "effect_summary",
            side_effect=RuntimeError("summary failed"),
        ):
            result = check.run(its, ctx)
        assert result.table is not None
        assert "error" in result.table.columns


# ---------------------------------------------------------------------------
# PreTreatmentPlaceboCheck additional tests
# ---------------------------------------------------------------------------


class TestPreTreatmentPlaceboCheckRun:
    """Additional tests covering run() and edge cases."""

    def test_validate_rejects_unfitted_experiment(self):
        mock_sdid = Mock(spec=StaggeredDifferenceInDifferences)
        mock_sdid.is_fitted = False
        with pytest.raises(ValueError, match="not been fitted"):
            PreTreatmentPlaceboCheck().validate(mock_sdid)

    def test_run_passing_pre_treatment(self):
        mock_exp = Mock()
        mock_exp.result.att_event_time = pd.DataFrame(
            {
                "event_time": [-3, -2, -1, 0, 1, 2],
                "att": [0.0, 0.0, 0.0, 0.5, 0.6, 0.7],
            }
        )
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))
        result = PreTreatmentPlaceboCheck().run(mock_exp, ctx)
        assert isinstance(result, CheckResult)
        assert result.check_name == "PreTreatmentPlaceboCheck"
        assert result.passed is True
        assert "passed" in result.text
        assert result.table is not None
        assert "mean_pre_att" in result.metadata

    def test_run_failing_pre_treatment(self):
        mock_exp = Mock()
        mock_exp.result.att_event_time = pd.DataFrame(
            {
                "event_time": [-3, -2, -1, 0, 1],
                "att": [5.0, 4.0, 6.0, 0.5, 0.6],
            }
        )
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))
        result = PreTreatmentPlaceboCheck().run(mock_exp, ctx)
        assert result.passed is False
        assert "failed" in result.text

    def test_cancelling_point_estimates_fail(self):
        """Equal-and-opposite pre-trends are not a parallel-trends pass."""
        mock_exp = Mock()
        mock_exp.result.att_event_time = pd.DataFrame(
            {
                "event_time": [-2, -1, 0],
                "att": [100.0, -100.0, 5.0],
            }
        )
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))
        result = PreTreatmentPlaceboCheck(threshold=0.05).run(mock_exp, ctx)
        assert result.passed is False
        assert "consistent with parallel trends" not in result.text
        assert result.metadata["event_times_differing"] == [-2, -1]

    def test_cancelling_estimates_fail_for_other_magnitudes(self):
        mock_exp = Mock()
        mock_exp.result.att_event_time = pd.DataFrame(
            {
                "event_time": [-3, -2, -1],
                "att": [10.0, -6.0, -4.0],
            }
        )
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))
        result = PreTreatmentPlaceboCheck().run(mock_exp, ctx)
        assert result.passed is False
        assert result.metadata["event_times_differing"] == [-3, -2, -1]

    def test_small_cancelling_point_estimates_fail(self):
        """threshold is not an absolute tolerance that lets cancelling ATTs pass."""
        mock_exp = Mock()
        mock_exp.result.att_event_time = pd.DataFrame(
            {
                "event_time": [-2, -1],
                "att": [0.2, -0.2],
            }
        )
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))
        result = PreTreatmentPlaceboCheck(threshold=0.05).run(mock_exp, ctx)
        assert result.passed is False

    def test_only_a_later_pre_treatment_estimate_failing_fails(self):
        mock_exp = Mock()
        mock_exp.result.att_event_time = pd.DataFrame(
            {
                "event_time": [-2, -1],
                "att": [0.0, 80.0],
            }
        )
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))
        result = PreTreatmentPlaceboCheck().run(mock_exp, ctx)
        assert result.passed is False
        assert result.metadata["event_times_differing"] == [-1]

    def test_insignificant_opposite_estimates_pass(self):
        mock_exp = Mock()
        mock_exp.result.att_event_time = pd.DataFrame(
            {
                "event_time": [-2, -1, 0],
                "att": [100.0, -100.0, 5.0],
                "att_std": [1000.0, 1000.0, 1.0],
                "n_obs": [30, 30, 30],
            }
        )
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))
        result = PreTreatmentPlaceboCheck(threshold=0.05).run(mock_exp, ctx)
        assert result.passed is True

    def test_one_significant_estimate_fails_even_if_mean_cancels(self):
        mock_exp = Mock()
        mock_exp.result.att_event_time = pd.DataFrame(
            {
                "event_time": [-2, -1],
                "att": [100.0, -100.0],
                "att_std": [1.0, 1000.0],
                "n_obs": [30, 30],
            }
        )
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))
        result = PreTreatmentPlaceboCheck(threshold=0.05).run(mock_exp, ctx)
        assert result.passed is False
        assert result.metadata["event_times_differing"] == [-2]

    def test_threshold_is_the_two_sided_alpha(self):
        from scipy.stats import t

        se = 1.0 / np.sqrt(30)
        critical = float(t.ppf(1 - 0.025, 29)) * se
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))

        def _result(att: float):
            mock_exp = Mock()
            mock_exp.result.att_event_time = pd.DataFrame(
                {
                    "event_time": [-1],
                    "att": [att],
                    "att_std": [1.0],
                    "n_obs": [30],
                }
            )
            return PreTreatmentPlaceboCheck(threshold=0.05).run(mock_exp, ctx)

        assert _result(critical).passed is True
        assert _result(critical + 0.01).passed is False

    def test_intervals_excluding_zero_fail_when_mean_cancels(self):
        mock_exp = Mock()
        mock_exp.result.att_event_time = pd.DataFrame(
            {
                "event_time": [-2, -1],
                "att": [100.0, -100.0],
                "att_lower": [90.0, -110.0],
                "att_upper": [110.0, -90.0],
            }
        )
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))
        result = PreTreatmentPlaceboCheck().run(mock_exp, ctx)
        assert result.passed is False
        assert result.metadata["event_times_differing"] == [-2, -1]

    def test_intervals_covering_zero_pass(self):
        mock_exp = Mock()
        mock_exp.result.att_event_time = pd.DataFrame(
            {
                "event_time": [-2, -1],
                "att": [100.0, -100.0],
                "att_lower": [-5.0, -150.0],
                "att_upper": [150.0, 5.0],
            }
        )
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))
        result = PreTreatmentPlaceboCheck().run(mock_exp, ctx)
        assert result.passed is True

    def test_non_identified_rows_are_not_evidence(self):
        mock_exp = Mock()
        mock_exp.result.att_event_time = pd.DataFrame(
            {
                "event_time": [-2, -1],
                "att": [np.nan, 0.0],
                "identified": [False, True],
            }
        )
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))
        result = PreTreatmentPlaceboCheck().run(mock_exp, ctx)
        assert result.passed is True

    def test_run_empty_pre_treatment(self):
        mock_exp = Mock()
        mock_exp.result.att_event_time = pd.DataFrame(
            {
                "event_time": [0, 1, 2],
                "att": [0.5, 0.6, 0.7],
            }
        )
        ctx = PipelineContext(data=pd.DataFrame({"x": [1]}))
        result = PreTreatmentPlaceboCheck().run(mock_exp, ctx)
        assert result.passed is None
        assert "No pre-treatment" in result.text


# ---------------------------------------------------------------------------
# clone_model tests
# ---------------------------------------------------------------------------


class TestCloneModel:
    """Tests for clone_model utility."""

    def test_clone_model_with_clone_method(self):
        class _Cloneable:
            def _clone(self):
                return _Cloneable()

        original = _Cloneable()
        cloned = clone_model(original)
        assert cloned is not original
        assert isinstance(cloned, _Cloneable)

    def test_clone_model_deepcopy_fallback(self):
        model = LinearRegression()
        model.some_attr = "test"
        cloned = clone_model(model)
        assert cloned is not model
        assert cloned.some_attr == "test"


# ---------------------------------------------------------------------------
# PersistenceCheck additional tests
# ---------------------------------------------------------------------------


class TestPersistenceCheckEdgeCases:
    """Additional edge case tests for PersistenceCheck."""

    def test_validate_rejects_non_its(self):
        df = cp.load_data("sc")
        model = cp.create_causalpy_compatible_class(LinearRegression())
        sc = SyntheticControl(
            df,
            treatment_time=70,
            control_units=["a", "b", "c"],
            treated_units=["actual"],
            model=model,
        )
        with pytest.raises(TypeError, match="InterruptedTimeSeries"):
            PersistenceCheck().validate(sc)
