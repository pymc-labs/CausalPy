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
"""Difference in differences in differences contrasts ordinary DiD fits."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from patsy import PatsyError
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy._arviz_compat import hdi_bounds
from causalpy.custom_exceptions import DataException
from causalpy.experiments import difference_in_differences_in_differences as ddd_module
from causalpy.experiments.base import BaseExperiment
from causalpy.experiments.difference_in_differences_in_differences import (
    DifferenceInDifferencesInDifferences,
)
from causalpy.pymc_models import LinearRegression as BayesianLinearRegression

FORMULA = "y ~ 1 + group * post_treatment"
TREATED_POST = {"none": 3.0, "lite": 6.0, "full": 10.0}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def sales_frame(
    *,
    tiers: Sequence[str] = ("none", "lite", "full"),
    drop: tuple[str, str] | None = None,
    lite_times: tuple[int, int] = (0, 1),
    lite_post_at_time_1: int = 1,
) -> pd.DataFrame:
    """Return the unadjusted two-period sales panel, with optional holes."""
    rows: list[dict[str, object]] = []
    for tier in tiers:
        times = lite_times if tier == "lite" else (0, 1)
        for group, rise in ((0, 1.0), (1, TREATED_POST[tier])):
            for time in times:
                if tier == "lite" and time == 1:
                    post = lite_post_at_time_1
                elif tier == "lite":
                    post = int(time == times[-1])
                else:
                    post = int(time == 1)
                if drop == (tier, "treated post") and group == 1 and post == 1:
                    continue
                rows.append(
                    {
                        "unit": f"{tier}-g{group}",
                        "t": time,
                        "group": group,
                        "post_treatment": post,
                        "tier": tier,
                        "y": 0.0 if post == 0 else rise,
                    }
                )
    return pd.DataFrame(rows)


def fit_sales(**kwargs) -> DifferenceInDifferencesInDifferences:
    """Fit the sales fixture with ordinary least squares."""
    data = kwargs.pop("data", sales_frame())
    return DifferenceInDifferencesInDifferences(
        data,
        FORMULA,
        time_variable_name="t",
        group_variable_name="group",
        level_variable_name="tier",
        model=LinearRegression(fit_intercept=False),
        **kwargs,
    ).fit()


def contrast_table(experiment: DifferenceInDifferencesInDifferences) -> pd.DataFrame:
    """Return contrast rows in requested order."""
    table = experiment.table()
    return table.loc[table["quantity"] == "contrast"].reset_index(drop=True)


def test_reference_level_reports_ordinary_did_and_signed_contrasts() -> None:
    """Unadjusted sales tiers are 2, 5, and 9, with contrasts 3 and 7 versus none."""
    experiment = fit_sales(reference_level="none")
    table = experiment.table()
    levels = table.loc[table["quantity"] == "level"].set_index("label")
    contrasts = contrast_table(experiment)

    assert list(levels.index) == ["none", "lite", "full"]
    assert levels.loc["none", "estimate"] == pytest.approx(2.0)
    assert levels.loc["lite", "estimate"] == pytest.approx(5.0)
    assert levels.loc["full", "estimate"] == pytest.approx(9.0)
    assert list(contrasts["label"]) == ["lite - none", "full - none"]
    assert contrasts["estimate"].tolist() == pytest.approx([3.0, 7.0])
    assert contrasts["uncertainty"].tolist() == ["point", "point"]
    assert contrasts["sd"].isna().all()
    assert contrasts["hdi_lower"].isna().all()


def test_explicit_pair_returns_only_the_requested_direction() -> None:
    """An explicit full-minus-lite pair is 4 and does not add the reference contrasts."""
    experiment = fit_sales(contrasts=[("full", "lite")])
    contrasts = contrast_table(experiment)
    levels = experiment.table()
    levels = levels.loc[levels["quantity"] == "level", "label"].tolist()

    assert levels == ["lite", "full"]
    assert list(contrasts["label"]) == ["full - lite"]
    assert contrasts["estimate"].iloc[0] == pytest.approx(4.0)

    reversed_pair = fit_sales(contrasts=[("lite", "full")])
    assert contrast_table(reversed_pair)["estimate"].iloc[0] == pytest.approx(-4.0)


def test_child_experiment_keeps_the_ordinary_counterfactual() -> None:
    """Each level's fitted DiD, not a zeroed triple interaction, is the stored result."""
    experiment = fit_sales(reference_level="none")
    child = experiment.experiments["full"]

    assert child.formula == FORMULA
    assert child.result.scenario_counterfactual is not None
    impact = np.asarray(child.result.causal_impact.values, dtype=float).reshape(-1)
    assert impact.size == 1
    assert float(impact[0]) == pytest.approx(9.0)
    assert not hasattr(experiment, "scenario_counterfactual")


def test_plot_labels_and_positions_match_the_table() -> None:
    """The contrast plot uses the table's labels and estimates, including a negative pair."""
    experiment = fit_sales(contrasts=[("full", "lite"), ("lite", "none")])
    ax = experiment.plot(show=False)
    contrasts = contrast_table(experiment)
    markers = next(line for line in ax.lines if line.get_label() == "contrast")

    assert [text.get_text() for text in ax.get_yticklabels()] == contrasts[
        "label"
    ].tolist()
    assert markers.get_xdata().tolist() == pytest.approx(contrasts["estimate"].tolist())
    assert ax.get_xlabel() == "Signed contrast (outcome units)"
    labeled = experiment.plot(show=False, outcome_units="sales units")
    assert labeled.get_xlabel() == "Signed contrast (sales units)"
    assert not any(line.get_label() == "_interval" for line in ax.lines)


def test_plot_intervals_match_the_table_hdi() -> None:
    """Interval lines span the table HDI, including a point that sits outside it."""
    summary = DifferenceInDifferencesInDifferences.summarize_paired_effects(
        {"full": [9.0, 11.0, 7.0], "lite": [5.0, 3.0, 1.0]},
        [("full", "lite")],
        point_estimates={"full": 20.0, "lite": 5.0},
    )
    experiment = fit_sales(contrasts=[("full", "lite")])
    experiment._effect_draws = {
        "full": np.array([9.0, 11.0, 7.0]),
        "lite": np.array([5.0, 3.0, 1.0]),
    }
    experiment._point_estimates = {"full": 20.0, "lite": 5.0}
    experiment._uncertainty = "unit_bootstrap"
    ax = experiment.plot(show=False)
    interval = next(line for line in ax.lines if line.get_label() == "_interval")
    expected = summary.table.loc[summary.table["quantity"] == "contrast"].iloc[0]

    assert interval.get_xdata().tolist() == pytest.approx(
        [expected["hdi_lower"], expected["hdi_upper"]]
    )
    marker = next(line for line in ax.lines if line.get_label() == "contrast")
    assert marker.get_xdata().tolist() == pytest.approx([expected["estimate"]])
    assert "paired differences" in ax.get_title()


def test_plot_show_displays_the_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default plot path displays the figure."""
    shown = False

    def _show(*_args, **_kwargs) -> None:
        nonlocal shown
        shown = True

    monkeypatch.setattr(ddd_module.plt, "show", _show)
    fit_sales(contrasts=[("full", "lite")]).plot()
    assert shown


def test_unfitted_reads_raise_before_group_not_sampled() -> None:
    """Table, draws, plot, and experiments fail with the fit() call, not a group name."""
    experiment = DifferenceInDifferencesInDifferences(
        sales_frame(),
        FORMULA,
        time_variable_name="t",
        group_variable_name="group",
        level_variable_name="tier",
        model=LinearRegression(fit_intercept=False),
        reference_level="none",
    )
    with pytest.raises(RuntimeError, match="Call fit\\(\\)"):
        experiment.table()
    with pytest.raises(RuntimeError, match="Call fit\\(\\)"):
        experiment.contrast_draws()
    with pytest.raises(RuntimeError, match="Call fit\\(\\)"):
        experiment.plot(show=False)
    with pytest.raises(RuntimeError, match="Call fit\\(\\)"):
        _ = experiment.experiments


def test_paired_draws_keep_positive_spread_when_marginal_errors_match() -> None:
    """(7, 5), (9, 3), (11, 1) summarize as [2, 6, 10], not a difference of standard errors."""
    summary = DifferenceInDifferencesInDifferences.summarize_paired_effects(
        {"a": [7.0, 9.0, 11.0], "b": [5.0, 3.0, 1.0]},
        [("a", "b")],
    )
    differences = summary.contrast_draws["a - b"].to_numpy()
    contrast = summary.table.loc[summary.table["quantity"] == "contrast"].iloc[0]
    marginal_sd = float(np.std([7.0, 9.0, 11.0], ddof=1))

    assert differences.tolist() == pytest.approx([2.0, 6.0, 10.0])
    assert contrast["estimate"] == pytest.approx(6.0)
    assert contrast["sd"] == pytest.approx(4.0)
    assert marginal_sd == pytest.approx(float(np.std([5.0, 3.0, 1.0], ddof=1)))
    assert contrast["sd"] != pytest.approx(0.0)
    assert [contrast["hdi_lower"], contrast["hdi_upper"]] == pytest.approx(
        hdi_bounds(differences)
    )


def test_point_estimates_stay_fixed_when_draws_supply_only_the_interval() -> None:
    """A supplied full-sample point is not replaced by the mean of the paired draws."""
    summary = DifferenceInDifferencesInDifferences.summarize_paired_effects(
        {"full": [0.0, 0.0], "lite": [0.0, 0.0]},
        [("full", "lite")],
        point_estimates={"full": 9.0, "lite": 5.0},
        uncertainty="unit_bootstrap",
    )
    contrast = summary.table.loc[summary.table["quantity"] == "contrast"].iloc[0]

    assert contrast["estimate"] == pytest.approx(4.0)
    assert contrast["sd"] == pytest.approx(0.0)
    assert contrast["uncertainty"] == "unit_bootstrap"


def test_mismatched_draw_lengths_are_not_trimmed() -> None:
    """Independent fits with different draw counts raise instead of aligning by truncation."""
    with pytest.raises(DataException, match="not trimmed"):
        DifferenceInDifferencesInDifferences.summarize_paired_effects(
            {"a": [1.0, 2.0], "b": [1.0]},
            [("a", "b")],
        )


def test_dataarray_draws_align_chain_then_draw() -> None:
    """A transposed chain/draw array is subtracted in chain-major order, not label order."""
    chain_major = xr.DataArray([[7.0, 9.0], [11.0, 1.0]], dims=("chain", "draw"))
    transposed = chain_major.transpose("draw", "chain")
    other = xr.DataArray([[5.0, 3.0], [1.0, 0.0]], dims=("chain", "draw"))
    summary = DifferenceInDifferencesInDifferences.summarize_paired_effects(
        {"a": transposed, "b": other},
        [("a", "b")],
    )

    assert summary.contrast_draws["a - b"].tolist() == pytest.approx(
        [2.0, 6.0, 10.0, 1.0]
    )


def test_unexpected_effect_dimension_names_the_level() -> None:
    """A non-sample dimension is an error, not a silently squeezed coordinate."""
    effect = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("draw", "unit"))
    with pytest.raises(DataException, match="Level 'a'"):
        DifferenceInDifferencesInDifferences.summarize_paired_effects(
            {"a": effect, "b": [1.0, 2.0, 3.0, 4.0]},
            [("a", "b")],
        )


def test_request_validation_rejects_ambiguous_or_empty_contrasts() -> None:
    """Reference mode and explicit pairs are mutually exclusive and individually required."""
    data = sales_frame()
    with pytest.raises(ValueError, match="not both"):
        DifferenceInDifferencesInDifferences(
            data,
            FORMULA,
            "t",
            "group",
            level_variable_name="tier",
            reference_level="none",
            contrasts=[("full", "lite")],
        )
    with pytest.raises(ValueError, match="reference_level or"):
        DifferenceInDifferencesInDifferences(
            data,
            FORMULA,
            "t",
            "group",
            level_variable_name="tier",
        )
    with pytest.raises(ValueError, match="at least one"):
        DifferenceInDifferencesInDifferences(
            data,
            FORMULA,
            "t",
            "group",
            level_variable_name="tier",
            contrasts=[],
        )
    with pytest.raises(ValueError, match="ordered pair"):
        DifferenceInDifferencesInDifferences(
            data,
            FORMULA,
            "t",
            "group",
            level_variable_name="tier",
            contrasts=["full"],
        )


def test_missing_level_cell_window_and_coding_name_the_level() -> None:
    """Panel problems name the affected level instead of dropping it."""
    with pytest.raises(DataException, match="gold"):
        fit_sales(data=sales_frame(), reference_level="gold")
    with pytest.raises(DataException, match="no other level"):
        fit_sales(data=sales_frame(tiers=("none",)), reference_level="none")
    with pytest.raises(DataException, match="lite.*treated post"):
        fit_sales(
            data=sales_frame(drop=("lite", "treated post")), reference_level="none"
        )
    with pytest.raises(DataException, match="does not share"):
        fit_sales(data=sales_frame(lite_times=(0, 2)), reference_level="none")
    with pytest.raises(DataException, match="not one intervention date"):
        fit_sales(data=sales_frame(lite_post_at_time_1=0), reference_level="none")


def test_missing_outcome_is_named_as_a_missing_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-finite outcome is not a present cell, including in a bootstrap replicate."""
    data = sales_frame()
    hole = data["tier"].eq("lite") & data["group"].eq(1) & data["post_treatment"].eq(1)
    data.loc[hole, "y"] = np.nan
    with pytest.raises(DataException, match="lite.*treated post"):
        fit_sales(data=data, reference_level="none")

    finite = sales_frame()
    kept = finite.loc[hole].copy()
    kept["unit"] = "lite-good"
    finite.loc[hole, "y"] = np.nan
    mixed = pd.concat([finite, kept], ignore_index=True)

    class _DropFiniteTreated:
        def choice(self, units, size, replace):
            del size, replace
            return np.asarray([unit for unit in units if unit != "lite-good"])

    monkeypatch.setattr(ddd_module, "default_rng", lambda seed: _DropFiniteTreated())
    with pytest.raises(DataException, match="bootstrap replicate 0") as exc_info:
        fit_sales(data=mixed, contrasts=[("lite", "none")]).fit(n_bootstrap=1)
    message = str(exc_info.value)
    assert "lite" in message
    assert "treated post" in message


def test_unused_level_with_a_missing_cell_is_not_part_of_the_contrast() -> None:
    """A level outside the requested pairs is ignored rather than silently dropped."""
    data = sales_frame(drop=("none", "treated post"))
    experiment = fit_sales(data=data, contrasts=[("full", "lite")])

    assert contrast_table(experiment)["estimate"].iloc[0] == pytest.approx(4.0)


def test_missing_column_and_non_dummy_coding_raise() -> None:
    """The shared DiD columns are required and dummy coded before any level is fit."""
    data = sales_frame().drop(columns=["unit"])
    with pytest.raises(DataException, match="unit"):
        fit_sales(data=data, reference_level="none")
    coded = sales_frame()
    coded["group"] = coded["group"].replace({1: 2})
    with pytest.raises(DataException, match="dummy coded"):
        fit_sales(data=coded, reference_level="none")
    post = sales_frame()
    post["post_treatment"] = post["post_treatment"].replace({1: 2})
    with pytest.raises(DataException, match="post_treatment"):
        fit_sales(data=post, reference_level="none")


def test_bootstrap_arguments_are_rejected_when_they_do_not_apply() -> None:
    """Bootstrap controls are a point-estimate path, not a posterior setting."""
    experiment = DifferenceInDifferencesInDifferences(
        sales_frame(),
        FORMULA,
        "t",
        "group",
        level_variable_name="tier",
        model=LinearRegression(fit_intercept=False),
        contrasts=[("full", "lite")],
    )
    with pytest.raises(ValueError, match="random_seed"):
        experiment.fit(random_seed=0)
    with pytest.raises(ValueError, match="at least 1"):
        experiment.fit(n_bootstrap=0)
    bayesian = DifferenceInDifferencesInDifferences(
        sales_frame(),
        FORMULA,
        "t",
        "group",
        level_variable_name="tier",
        model=BayesianLinearRegression(),
        contrasts=[("full", "lite")],
    )
    with pytest.raises(ValueError, match="Posterior draws"):
        bayesian.fit(n_bootstrap=2)
    default_model = DifferenceInDifferencesInDifferences(
        sales_frame(),
        FORMULA,
        "t",
        "group",
        level_variable_name="tier",
        contrasts=[("full", "lite")],
    )
    with pytest.raises(ValueError, match="Posterior draws"):
        default_model.fit(n_bootstrap=2)


def test_shared_units_reject_independent_posterior_draws() -> None:
    """A unit in two levels cannot borrow an unpaired posterior interval."""
    data = _shared_unit_frame()
    for model in (None, BayesianLinearRegression()):
        experiment = DifferenceInDifferencesInDifferences(
            data,
            FORMULA,
            "t",
            "group",
            level_variable_name="tier",
            contrasts=[("full", "lite")],
            model=model,
        )
        with pytest.raises(DataException, match="0-u1") as exc_info:
            experiment.fit()
        message = str(exc_info.value)
        assert "lite" in message
        assert "full" in message
        assert "n_bootstrap" in message
    fitted = DifferenceInDifferencesInDifferences(
        data,
        FORMULA,
        "t",
        "group",
        level_variable_name="tier",
        contrasts=[("full", "lite")],
        model=LinearRegression(fit_intercept=False),
    ).fit()
    assert contrast_table(fitted)["estimate"].iloc[0] == pytest.approx(4.0)


def test_post_treatment_level_recode_is_named_and_not_fit() -> None:
    """A pretreated unit that gains a level after treatment is not a silent contrast."""
    extra = pd.DataFrame(
        [
            {
                "unit": "recode",
                "t": 0,
                "group": 1,
                "post_treatment": 0,
                "tier": "lite",
                "y": 0.0,
            },
            {
                "unit": "recode",
                "t": 1,
                "group": 1,
                "post_treatment": 1,
                "tier": "full",
                "y": 100.0,
            },
        ]
    )
    recoded = pd.concat([sales_frame(), extra], ignore_index=True)
    with pytest.raises(DataException, match="recode") as exc_info:
        fit_sales(data=recoded, contrasts=[("full", "lite")])
    assert "full" in str(exc_info.value)
    unused_origin = extra.copy()
    unused_origin.loc[unused_origin["t"] == 0, "tier"] = "gold"
    recoded_from_gold = pd.concat([sales_frame(), unused_origin], ignore_index=True)
    with pytest.raises(DataException, match="recode") as unused_info:
        fit_sales(data=recoded_from_gold, contrasts=[("full", "lite")])
    assert "full" in str(unused_info.value)


def test_within_unit_levels_and_post_only_units_stay_eligible() -> None:
    """Levels held in both windows, and a unit seen only after treatment, are kept."""
    rows: list[dict[str, object]] = []
    for tier, treated_rise in (("lite", 6.0), ("full", 10.0)):
        for group, rise in ((0, 1.0), (1, treated_rise)):
            for time, post in ((0, 0), (1, 1)):
                rows.append(
                    {
                        "unit": f"{tier}-g{group}",
                        "t": time,
                        "group": group,
                        "post_treatment": post,
                        "tier": tier,
                        "y": 0.0 if post == 0 else rise,
                    }
                )
        for time, post in ((0, 0), (1, 1)):
            rows.append(
                {
                    "unit": "both",
                    "t": time,
                    "group": 1,
                    "post_treatment": post,
                    "tier": tier,
                    "y": 0.0 if post == 0 else treated_rise,
                }
            )
    rows.append(
        {
            "unit": "late",
            "t": 1,
            "group": 0,
            "post_treatment": 1,
            "tier": "lite",
            "y": 1.0,
        }
    )
    experiment = fit_sales(data=pd.DataFrame(rows), contrasts=[("full", "lite")])
    assert contrast_table(experiment)["estimate"].iloc[0] == pytest.approx(4.0)


def test_bootstrap_resamples_units_once_and_keeps_the_full_sample_contrast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One unit sample is applied to every level, and the estimate stays the fitted contrast."""
    data = _shared_unit_frame()
    calls: list[np.ndarray] = []

    class _FixedRng:
        def choice(self, units, size, replace):
            del replace
            sample = np.asarray(units)[:size]
            calls.append(sample)
            return sample

    monkeypatch.setattr(ddd_module, "default_rng", lambda seed: _FixedRng())
    experiment = DifferenceInDifferencesInDifferences(
        data,
        FORMULA,
        time_variable_name="t",
        group_variable_name="group",
        level_variable_name="tier",
        model=LinearRegression(fit_intercept=False),
        contrasts=[("full", "lite")],
    ).fit(n_bootstrap=2, random_seed=0)

    assert len(calls) == 2
    assert contrast_table(experiment)["estimate"].iloc[0] == pytest.approx(4.0)
    assert experiment.contrast_draws().shape == (2, 1)
    assert experiment.table()["uncertainty"].iloc[-1] == "unit_bootstrap"


def test_bootstrap_names_a_replicate_that_loses_a_comparison_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replicate that drops a cell raises and names the level; it is not skipped."""

    class _DropTreated:
        def choice(self, units, size, replace):
            del size, replace
            return np.asarray([unit for unit in units if not str(unit).endswith("-g1")])

    monkeypatch.setattr(ddd_module, "default_rng", lambda seed: _DropTreated())
    with pytest.raises(DataException, match="bootstrap replicate 0"):
        fit_sales(contrasts=[("full", "lite")]).fit(n_bootstrap=1)


def test_class_is_a_tier_one_export_without_reordering_neighbors() -> None:
    """The wrapper is exported beside DifferenceInDifferences and is not an experiment subclass."""
    names = list(cp.__all__)
    did = names.index("DifferenceInDifferences")
    assert names[did + 1] == "DifferenceInDifferencesInDifferences"
    source = Path(cp.__file__).read_text(encoding="utf-8")
    assert source.index("difference_in_differences_in_differences") > source.index(
        "diff_in_diff import DifferenceInDifferences"
    )
    assert not issubclass(DifferenceInDifferencesInDifferences, BaseExperiment)


def test_each_level_receives_a_fresh_pymc_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A posterior model is cloned per level, not deep-copied or shared."""
    seen: list[object] = []

    def _fit(self):
        seen.append(self.model)
        tier = self.data["tier"].iloc[0]
        impact = {"none": 2.0, "lite": 5.0, "full": 9.0}[tier]
        self._result = SimpleNamespace(causal_impact=np.array([impact]))
        return self

    monkeypatch.setattr(ddd_module.DifferenceInDifferences, "fit", _fit)
    original = BayesianLinearRegression()
    experiment = DifferenceInDifferencesInDifferences(
        sales_frame(),
        FORMULA,
        time_variable_name="t",
        group_variable_name="group",
        level_variable_name="tier",
        model=original,
        reference_level="none",
    ).fit()

    assert len(seen) == 3
    assert all(isinstance(model, BayesianLinearRegression) for model in seen)
    assert all(model is not original for model in seen)
    assert len({id(model) for model in seen}) == 3
    assert contrast_table(experiment)["estimate"].tolist() == pytest.approx([3.0, 7.0])


def test_bootstrap_names_a_level_whose_units_are_all_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replicate that drops a whole level names that level instead of crashing."""

    class _DropLite:
        def choice(self, units, size, replace):
            del size, replace
            return np.asarray(
                [unit for unit in units if not str(unit).startswith("lite")]
            )

    monkeypatch.setattr(ddd_module, "default_rng", lambda seed: _DropLite())
    with pytest.raises(DataException, match="bootstrap replicate 0") as exc_info:
        fit_sales(contrasts=[("full", "lite")]).fit(n_bootstrap=1)
    message = str(exc_info.value)
    assert "lite" in message
    assert "treated post" in message


def test_invalid_formula_is_a_formula_error_not_a_missing_cell() -> None:
    """A name Patsy cannot evaluate is not reported as a missing comparison cell."""
    experiment = DifferenceInDifferencesInDifferences(
        sales_frame(),
        "y ~ 1 + group * post_treatment + missing",
        time_variable_name="t",
        group_variable_name="group",
        level_variable_name="tier",
        model=LinearRegression(fit_intercept=False),
        reference_level="none",
    )
    with pytest.raises(PatsyError, match="missing"):
        experiment.fit()


def test_duplicate_contrast_text_keeps_both_requested_rows() -> None:
    """Repeated pair text stays two contrasts, with a suffix so the labels differ."""
    experiment = fit_sales(contrasts=[("full", "lite"), ("full", "lite")])
    table = contrast_table(experiment)

    assert table["label"].tolist() == ["full - lite", "full - lite (2)"]
    assert table["estimate"].tolist() == pytest.approx([4.0, 4.0])
    assert experiment.contrast_draws().columns.tolist() == table["label"].tolist()


def test_missing_level_column_is_not_reported_as_an_unknown_level() -> None:
    """A missing third-dimension column is a missing column in both contrast modes."""
    data = sales_frame().drop(columns=["tier"])
    with pytest.raises(DataException, match="Missing required columns") as reference:
        DifferenceInDifferencesInDifferences(
            data,
            FORMULA,
            time_variable_name="t",
            group_variable_name="group",
            level_variable_name="tier",
            model=LinearRegression(fit_intercept=False),
            reference_level="none",
        )
    with pytest.raises(DataException, match="Missing required columns") as explicit:
        DifferenceInDifferencesInDifferences(
            data,
            FORMULA,
            time_variable_name="t",
            group_variable_name="group",
            level_variable_name="tier",
            model=LinearRegression(fit_intercept=False),
            contrasts=[("full", "lite")],
        )
    assert "tier" in str(reference.value)
    assert "tier" in str(explicit.value)
    assert "not in column" not in str(reference.value)
    assert "not in column" not in str(explicit.value)


def test_summarize_rejects_empty_draws_unknown_levels_and_bad_labels() -> None:
    """The summarizer names a level with no draws and refuses an unknown uncertainty label."""
    draws = {"full": [11.0, 1.0], "lite": [9.0, 3.0]}
    with pytest.raises(DataException, match="lite"):
        DifferenceInDifferencesInDifferences.summarize_paired_effects(
            {"full": [11.0, 1.0], "lite": []},
            contrasts=[("full", "lite")],
        )
    with pytest.raises(DataException, match="gold"):
        DifferenceInDifferencesInDifferences.summarize_paired_effects(
            draws,
            contrasts=[("full", "gold")],
        )
    with pytest.raises(ValueError, match="unit_bootstrap"):
        DifferenceInDifferencesInDifferences.summarize_paired_effects(
            draws,
            contrasts=[("full", "lite")],
            uncertainty="stderr",
        )
    with pytest.raises(DataException, match="lite"):
        DifferenceInDifferencesInDifferences.summarize_paired_effects(
            draws,
            contrasts=[("full", "lite")],
            point_estimates={"full": 9.0},
        )


def test_plot_uses_a_caller_supplied_figure_size() -> None:
    """An explicit figure size is used instead of the default that grows with the table."""
    ax = fit_sales(contrasts=[("full", "lite")]).plot(show=False, figsize=(8.0, 3.0))
    assert tuple(ax.figure.get_size_inches()) == pytest.approx((8.0, 3.0))


def _shared_unit_frame() -> pd.DataFrame:
    """Return two tiers whose unit ids are shared, with full's treated rise larger by 4."""
    rows: list[dict[str, object]] = []
    for tier, treated_rise in (("lite", 6.0), ("full", 10.0)):
        for group, rise in ((0, 1.0), (1, treated_rise)):
            for unit in ("u1", "u2"):
                for time, post in ((0, 0), (1, 1)):
                    rows.append(
                        {
                            "unit": f"{group}-{unit}",
                            "t": time,
                            "group": group,
                            "post_treatment": post,
                            "tier": tier,
                            "y": 0.0 if post == 0 else rise,
                        }
                    )
    return pd.DataFrame(rows)
