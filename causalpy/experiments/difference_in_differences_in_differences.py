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
"""Difference in differences in differences as a contrast of ordinary DiD fits."""

from __future__ import annotations

import copy
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.random import default_rng
from numpy.typing import ArrayLike
from patsy import PatsyError
from sklearn.base import RegressorMixin

from causalpy._arviz_compat import hdi_bounds
from causalpy.constants import HDI_PROB
from causalpy.custom_exceptions import DataException
from causalpy.experiments.diff_in_diff import DifferenceInDifferences
from causalpy.formula_utils import build_formula_matrices
from causalpy.input_data import DataFrameLike, to_pandas
from causalpy.pymc_models import PyMCModel
from causalpy.utils import _is_variable_dummy_coded

_Uncertainty = Literal["draws", "point", "posterior", "unit_bootstrap"]
_CELL_NAMES = ("treated pre", "treated post", "control pre", "control post")


@dataclass(frozen=True)
class _PairedContrastSummary:
    """Tidy paired-contrast table and the within-draw differences behind it."""

    table: pd.DataFrame
    contrast_draws: pd.DataFrame


def _clone_model(
    model: PyMCModel | RegressorMixin | None,
) -> PyMCModel | RegressorMixin | None:
    """Return a fresh model so level fits do not share mutable state."""
    if model is None:
        return None
    clone = getattr(model, "_clone", None)
    if callable(clone):
        return clone()
    return copy.deepcopy(model)


def _is_post(series: pd.Series) -> pd.Series:
    """Return a boolean mask for the post-treatment code, without comparing to True."""
    return pd.to_numeric(series, errors="coerce").eq(1)


def _is_treated(series: pd.Series) -> pd.Series:
    """Return a boolean mask for the treated group code."""
    return pd.to_numeric(series, errors="coerce").eq(1)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise if a column the difference-in-differences specification needs is absent."""
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise DataException(f"Missing required columns: {missing}.")


def _resolve_level(
    observed: Sequence[Hashable],
    requested: Hashable,
    *,
    column: str,
) -> Hashable:
    """Return the observed level equal to ``requested``, preserving the data's value."""
    matches = [level for level in observed if level == requested]
    if not matches:
        raise DataException(f"Level {requested!r} is not in column {column!r}.")
    return matches[0]


def _as_pairs(
    contrasts: Sequence[Sequence[Hashable]],
) -> list[tuple[Hashable, Hashable]]:
    """Return explicit contrasts as ordered pairs, rejecting a bare pair or a string."""
    if len(contrasts) == 0:
        raise ValueError("contrasts must contain at least one ordered pair.")
    pairs: list[tuple[Hashable, Hashable]] = []
    for pair in contrasts:
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise ValueError("Each contrast must be an ordered pair (a, b).")
        pairs.append((pair[0], pair[1]))
    return pairs


def _missing_cells(
    frame: pd.DataFrame,
    *,
    group_variable_name: str,
    post_treatment_variable_name: str,
) -> list[str]:
    """Return the treated/control by pre/post cells with no rows."""
    treated = _is_treated(frame[group_variable_name])
    post = _is_post(frame[post_treatment_variable_name])
    masks = (
        treated & ~post,
        treated & post,
        ~treated & ~post,
        ~treated & post,
    )
    return [
        name for name, mask in zip(_CELL_NAMES, masks, strict=True) if not mask.any()
    ]


def _rows_used_by_formula(frame: pd.DataFrame, formula: str) -> pd.DataFrame:
    """Return rows the formula keeps, so a missing outcome is not a present cell."""
    if frame.empty:
        return frame
    try:
        outcome, _design = build_formula_matrices(
            formula, frame.reset_index(drop=True), return_type="dataframe"
        )
    except PatsyError:
        return frame
    return frame.iloc[outcome.index.to_numpy()]


def _window_times(
    frame: pd.DataFrame,
    *,
    time_variable_name: str,
    post_treatment_variable_name: str,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Return sorted unique pre and post time values."""
    post = _is_post(frame[post_treatment_variable_name])
    pre_times = tuple(sorted(frame.loc[~post, time_variable_name].unique(), key=str))
    post_times = tuple(sorted(frame.loc[post, time_variable_name].unique(), key=str))
    return pre_times, post_times


def _draw_vector(effect: ArrayLike, *, level: Hashable) -> np.ndarray:
    """Flatten effect draws in chain-then-draw order.

    Unlabeled arrays are flattened in C order. An xarray array is squeezed out
    of size-1 extra dimensions and transposed to ``("chain", "draw")`` before
    flattening, so two levels are aligned by position rather than by coordinate
    labels.
    """
    if isinstance(effect, xr.DataArray):
        data = effect
        extra = [dim for dim in data.dims if dim not in ("chain", "draw")]
        unsqueezable = [dim for dim in extra if int(data.sizes[dim]) != 1]
        if unsqueezable:
            raise DataException(
                f"Level {level!r} effect draws have unexpected dimensions "
                f"{unsqueezable}."
            )
        if extra:
            data = data.squeeze(drop=True)
        ordered = [dim for dim in ("chain", "draw") if dim in data.dims]
        if ordered:
            data = data.transpose(*ordered)
        values = np.asarray(data.values, dtype=float)
    else:
        values = np.asarray(effect, dtype=float)
    vector = values.reshape(-1)
    if vector.size == 0:
        raise DataException(f"Level {level!r} has no effect draws.")
    return vector


def _levels_by_unit(
    frame: pd.DataFrame, level_variable_name: str
) -> dict[Hashable, set[Hashable]]:
    """Return the levels observed for each assignment unit."""
    grouped: dict[Hashable, set[Hashable]] = {}
    for unit, level in zip(frame["unit"], frame[level_variable_name], strict=True):
        grouped.setdefault(unit, set()).add(level)
    return grouped


def _contrast_label(level_a: Hashable, level_b: Hashable, *, suffix: int) -> str:
    """Return a stable contrast label, disambiguating duplicate pair text."""
    label = f"{level_a} - {level_b}"
    if suffix:
        return f"{label} ({suffix + 1})"
    return label


def _sample_standard_deviation(values: np.ndarray) -> float:
    """Return the sample standard deviation, or NaN when a single draw has no spread."""
    if values.size < 2:
        return np.nan
    return float(np.std(values, ddof=1))


def _interval(values: np.ndarray, hdi_prob: float) -> tuple[float, float]:
    """Return an HDI for two or more draws, or a pair of NaNs for a point."""
    if values.size < 2:
        return np.nan, np.nan
    return hdi_bounds(values, prob=hdi_prob)


class DifferenceInDifferencesInDifferences:
    """Contrast ordinary difference-in-differences fits across a third dimension.

    ``DDD(a, b) = DiD(a) - DiD(b)``. Each level is fit with
    :class:`~causalpy.experiments.diff_in_diff.DifferenceInDifferences` and the
    same formula, time column, group column, and post-treatment column. The
    contrast is the difference of those fitted effects. It is not a pooled
    triple-interaction coefficient, and its interval is not a difference of
    standard errors.

    Pass ``reference_level`` to compare every other observed level with that
    reference, in data appearance order. Pass ``contrasts`` to request explicit
    ordered pairs and no others. The two controls are mutually exclusive.

    Parameters
    ----------
    data : dataframe-like
        Any eager dataframe Narwhals supports, such as pandas, Polars, or
        PyArrow. Converted to pandas internally. Must contain ``unit`` plus the
        time, group, post-treatment, and level columns. Group and
        post-treatment columns are dummy coded with 0 and 1.
    formula : str
        Within-level difference-in-differences formula, passed unchanged to
        each level. It must already be a formula
        :class:`~causalpy.experiments.diff_in_diff.DifferenceInDifferences`
        accepts.
    time_variable_name : str
        Name of the data column for the time variable.
    group_variable_name : str
        Name of the data column for the treated-versus-control indicator.
    post_treatment_variable_name : str, optional
        Name of the data column indicating the post-treatment period.
        Defaults to ``"post_treatment"``.
    model : PyMCModel or RegressorMixin, optional
        Template copied once per level. ``None`` uses the ordinary
        difference-in-differences default, a PyMC ``LinearRegression``.
        The template itself is not fit.
    level_variable_name : str
        Name of the pretreatment third-dimension column. Every analyzed level
        must contain treated and control observations in both windows.
    reference_level : hashable, optional
        Level subtracted from every other observed level. Required unless
        ``contrasts`` is passed.
    contrasts : sequence of pair, optional
        Explicit ordered pairs ``(a, b)`` meaning ``DiD(a) - DiD(b)``, returned
        in the requested order. Required unless ``reference_level`` is passed.

    Raises
    ------
    ValueError
        If both ``reference_level`` and ``contrasts`` are passed, if neither is
        passed, or if a contrast is not an ordered pair.
    DataException
        If a requested level is absent, a comparison cell is missing, the
        post-treatment code is inconsistent, levels do not share one
        pre/post window, or a unit gains a level only after treatment.

    See Also
    --------
    causalpy.DifferenceInDifferences : The ordinary difference in differences
        fit once per level.

    Notes
    -----
    Construction validates the panel and resolves the contrasts. Call
    :meth:`fit` to fit one difference in differences per analyzed level.
    ``experiments`` then maps each level to that fit, whose
    ``scenario_counterfactual`` is the ordinary treated-post counterfactual.
    This class does not build a second counterfactual by zeroing an
    interaction column.

    Contrast uncertainty is the spread of within-draw or unit-bootstrap
    differences. A bootstrap resamples ``unit`` once per replicate, keeps every
    period and analyzed level of a drawn unit together, and refits both sides
    on that replicate. The reported estimate stays the full-sample fitted
    difference; bootstrap draws supply the interval and are not required to be
    centered on it. Posterior draws are aligned by chain-then-draw position
    only when no assignment unit appears in more than one analyzed level.
    When a unit does span levels, independent posterior draws are not a valid
    interval: fit a point-estimate model and pass ``n_bootstrap``. Mismatched
    draw counts raise :class:`~causalpy.custom_exceptions.DataException`
    rather than trimming either chain.
    """

    def __init__(
        self,
        data: DataFrameLike,
        formula: str,
        time_variable_name: str,
        group_variable_name: str,
        post_treatment_variable_name: str = "post_treatment",
        model: PyMCModel | RegressorMixin | None = None,
        *,
        level_variable_name: str,
        reference_level: Hashable | None = None,
        contrasts: Sequence[Sequence[Hashable]] | None = None,
    ) -> None:
        pandas_data = to_pandas(data)
        pandas_data.index.name = "obs_ind"
        self.data = pandas_data
        self.formula = formula
        self.time_variable_name = time_variable_name
        self.group_variable_name = group_variable_name
        self.post_treatment_variable_name = post_treatment_variable_name
        self.model = model
        self.level_variable_name = level_variable_name
        self.reference_level = reference_level
        observed = (
            list(pd.unique(self.data[level_variable_name]))
            if (level_variable_name in self.data.columns)
            else []
        )
        self.contrasts = self._resolve_contrasts(observed, reference_level, contrasts)
        self._level_order = self._levels_in_data_order(observed)
        self._validate_panel()
        self._experiments: dict[Hashable, DifferenceInDifferences] | None = None
        self._effect_draws: dict[Hashable, np.ndarray] | None = None
        self._point_estimates: dict[Hashable, float] | None = None
        self._uncertainty: _Uncertainty | None = None

    def _resolve_contrasts(
        self,
        observed: Sequence[Hashable],
        reference_level: Hashable | None,
        contrasts: Sequence[Sequence[Hashable]] | None,
    ) -> tuple[tuple[Hashable, Hashable], ...]:
        """Resolve reference mode or explicit pairs, but not both and not neither."""
        if contrasts is not None and reference_level is not None:
            raise ValueError(
                "Pass reference_level or contrasts, not both. Explicit "
                "contrasts are the complete result."
            )
        if contrasts is None:
            if reference_level is None:
                raise ValueError(
                    "Pass reference_level or an explicit list of contrasts."
                )
            if self.level_variable_name not in self.data.columns:
                raise DataException(
                    f"Missing required columns: [{self.level_variable_name!r}]."
                )
            reference = _resolve_level(
                observed,
                reference_level,
                column=self.level_variable_name,
            )
            pairs = [(level, reference) for level in observed if level != reference]
            if not pairs:
                raise DataException(
                    f"reference_level {reference_level!r} leaves no other level "
                    "to contrast."
                )
            return tuple(pairs)
        requested = _as_pairs(contrasts)
        if self.level_variable_name not in self.data.columns:
            raise DataException(
                f"Missing required columns: [{self.level_variable_name!r}]."
            )
        return tuple(
            (
                _resolve_level(observed, level_a, column=self.level_variable_name),
                _resolve_level(observed, level_b, column=self.level_variable_name),
            )
            for level_a, level_b in requested
        )

    def _levels_in_data_order(
        self, observed: Sequence[Hashable]
    ) -> tuple[Hashable, ...]:
        """Return analyzed levels in appearance order, including a reference level."""
        needed = {level for pair in self.contrasts for level in pair}
        return tuple(level for level in observed if level in needed)

    def _rows_for_level(self, level: Hashable) -> pd.DataFrame:
        """Return a copy of one level's rows."""
        return self.data.loc[self.data[self.level_variable_name] == level].copy()

    def _analyzed_frame(self) -> pd.DataFrame:
        """Return analyzed levels only, so an unused level cannot change the fit."""
        return pd.concat(
            [self._rows_for_level(level) for level in self._level_order],
            ignore_index=True,
        )

    def _validate_panel(self) -> None:
        """Check columns, coding, one intervention date, cells, and shared windows."""
        required = [
            self.time_variable_name,
            self.group_variable_name,
            self.post_treatment_variable_name,
            self.level_variable_name,
            "unit",
        ]
        _require_columns(self.data, required)
        analyzed = self._analyzed_frame()
        group = analyzed[self.group_variable_name]
        post = analyzed[self.post_treatment_variable_name]
        if not _is_variable_dummy_coded(group):
            raise DataException(
                f"The grouping variable {self.group_variable_name} should be "
                "dummy coded. Consisting of 0's and 1's only."
            )
        if not _is_variable_dummy_coded(post):
            raise DataException(
                f"The post-treatment column {self.post_treatment_variable_name!r} "
                "must be dummy coded with 0 and 1 only, the same coding in every level."
            )
        self._require_one_intervention(analyzed)
        self._require_stable_pre_levels()
        self._require_comparison_cells(context="in the supplied data")
        self._require_shared_windows()

    def _require_stable_pre_levels(self) -> None:
        """Reject a level that a pretreated unit gains only after treatment."""
        post = _is_post(self.data[self.post_treatment_variable_name])
        before = _levels_by_unit(self.data.loc[~post], self.level_variable_name)
        after = _levels_by_unit(self.data.loc[post], self.level_variable_name)
        problems: list[str] = []
        for unit, post_levels in after.items():
            pre_levels = before.get(unit)
            if not pre_levels:
                continue
            gained = [level for level in post_levels if level not in pre_levels]
            if not gained:
                continue
            problems.append(f"unit {unit!r} is {gained} after treatment but not before")
        if problems:
            raise DataException(
                "A pretreated unit must not gain a level after treatment. "
                + "; ".join(problems)
                + "."
            )

    def _require_one_intervention(self, analyzed: pd.DataFrame) -> None:
        """Reject a time value that is post-treatment in one level and pre in another."""
        coded = analyzed.assign(
            _post=_is_post(analyzed[self.post_treatment_variable_name])
        )
        for time, group in coded.groupby(self.time_variable_name, observed=True):
            if int(group["_post"].nunique()) <= 1:
                continue
            post_levels = group.loc[group["_post"], self.level_variable_name].unique()
            pre_levels = group.loc[~group["_post"], self.level_variable_name].unique()
            raise DataException(
                f"Time {time!r} is not one intervention date: post-treatment in "
                f"levels {list(post_levels)} and pre-treatment in levels "
                f"{list(pre_levels)}."
            )

    def _require_comparison_cells(
        self, *, context: str, frame: pd.DataFrame | None = None
    ) -> None:
        """Name every missing treated/control by pre/post cell instead of dropping it."""
        problems: list[str] = []
        for level in self._level_order:
            level_frame = (
                self._rows_for_level(level)
                if frame is None
                else frame.loc[frame[self.level_variable_name] == level]
            )
            missing = _missing_cells(
                _rows_used_by_formula(level_frame, self.formula),
                group_variable_name=self.group_variable_name,
                post_treatment_variable_name=self.post_treatment_variable_name,
            )
            problems.extend(f"{level!r} {cell}" for cell in missing)
        if problems:
            raise DataException(
                "Each level needs treated and control observations in both "
                f"windows. Missing comparison cells {context}: "
                + "; ".join(problems)
                + "."
            )

    def _require_shared_windows(self) -> None:
        """Require identical pre and post time sets across analyzed levels."""
        windows = {
            level: _window_times(
                self._rows_for_level(level),
                time_variable_name=self.time_variable_name,
                post_treatment_variable_name=self.post_treatment_variable_name,
            )
            for level in self._level_order
        }
        first = self._level_order[0]
        reference_window = windows[first]
        for level, window in windows.items():
            if window == reference_window:
                continue
            raise DataException(
                f"Level {level!r} does not share the pre/post window of level "
                f"{first!r}. {level!r} has pre {window[0]} and post {window[1]}; "
                f"{first!r} has pre {reference_window[0]} and post "
                f"{reference_window[1]}."
            )

    def fit(
        self,
        *,
        n_bootstrap: int | None = None,
        random_seed: int | None = None,
    ) -> DifferenceInDifferencesInDifferences:
        """Fit one difference in differences per level, then store contrast draws.

        Parameters
        ----------
        n_bootstrap : int, optional
            Number of unit-bootstrap replicates. Omit it to use each level's own
            effect draws. A point-estimate fit then has no interval until this
            is set. Cannot be combined with a PyMC model, whose posterior draws
            already form the contrast.
        random_seed : int, optional
            Seed for the unit bootstrap. Valid only when ``n_bootstrap`` is set.

        Returns
        -------
        DifferenceInDifferencesInDifferences
            This experiment, so the call can be chained.

        Raises
        ------
        ValueError
            If ``n_bootstrap`` is not a positive integer, ``random_seed`` is
            set without it, or a posterior model is combined with the bootstrap.
        DataException
            If assignment units span analyzed levels and the model supplies
            independent posterior draws, or if a bootstrap replicate loses a
            comparison cell. The replicate is not dropped.
        """
        self._reject_independent_posterior_for_shared_units()
        self._validate_bootstrap(n_bootstrap, random_seed)
        experiments = {
            level: self._fit_level(self._rows_for_level(level))
            for level in self._level_order
        }
        if n_bootstrap is None:
            effect_draws = {
                level: _draw_vector(experiment.result.causal_impact, level=level)
                for level, experiment in experiments.items()
            }
            uncertainty: _Uncertainty = (
                "posterior"
                if any(values.size > 1 for values in effect_draws.values())
                else "point"
            )
            point_estimates = None
        else:
            effect_draws = self._bootstrap_draws(n_bootstrap, random_seed)
            uncertainty = "unit_bootstrap"
            # The reported contrast stays the full-sample fitted difference.
            # Bootstrap replicates supply the interval and are not substituted
            # for that point.
            point_estimates = {
                level: float(
                    np.mean(_draw_vector(experiment.result.causal_impact, level=level))
                )
                for level, experiment in experiments.items()
            }
        self._experiments = experiments
        self._effect_draws = effect_draws
        self._point_estimates = point_estimates
        self._uncertainty = uncertainty
        return self

    def _spanning_units(self) -> list[tuple[Hashable, list[Hashable]]]:
        """Return assignment units that appear in more than one analyzed level."""
        analyzed = self._analyzed_frame()
        levels_by_unit: dict[Hashable, list[Hashable]] = {}
        for level in self._level_order:
            subset = analyzed.loc[analyzed[self.level_variable_name] == level, "unit"]
            for unit in pd.unique(subset):
                levels = levels_by_unit.setdefault(unit, [])
                if level not in levels:
                    levels.append(level)
        return [
            (unit, levels) for unit, levels in levels_by_unit.items() if len(levels) > 1
        ]

    def _reject_independent_posterior_for_shared_units(self) -> None:
        """Refuse an unpaired posterior interval when one unit spans levels."""
        if self.model is not None and not isinstance(self.model, PyMCModel):
            return
        spanning = self._spanning_units()
        if not spanning:
            return
        listed = "; ".join(
            f"unit {unit!r} appears in levels {levels}" for unit, levels in spanning
        )
        raise DataException(
            "Assignment units span analyzed levels, so independent posterior "
            "draws are not a valid contrast interval. "
            f"{listed}. Fit a point-estimate model and pass n_bootstrap so "
            "each replicate keeps those units together."
        )

    def _validate_bootstrap(
        self,
        n_bootstrap: int | None,
        random_seed: int | None,
    ) -> None:
        """Reject a bootstrap that would mix posterior draws with unit replicates."""
        if random_seed is not None and n_bootstrap is None:
            raise ValueError(
                "random_seed applies to n_bootstrap. Omit it, or pass n_bootstrap."
            )
        if n_bootstrap is None:
            return
        if (
            isinstance(n_bootstrap, bool)
            or not isinstance(n_bootstrap, int)
            or n_bootstrap < 1
        ):
            raise ValueError("n_bootstrap must be an integer of at least 1.")
        if self.model is None or isinstance(self.model, PyMCModel):
            raise ValueError(
                "n_bootstrap is for a point-estimate model. Posterior draws "
                "already form the contrast; omit n_bootstrap."
            )

    def _fit_level(self, frame: pd.DataFrame) -> DifferenceInDifferences:
        """Fit one ordinary difference in differences on a copied model."""
        experiment = DifferenceInDifferences(
            frame,
            formula=self.formula,
            time_variable_name=self.time_variable_name,
            group_variable_name=self.group_variable_name,
            post_treatment_variable_name=self.post_treatment_variable_name,
            model=_clone_model(self.model),
        )
        return experiment.fit()

    def _bootstrap_draws(
        self,
        n_bootstrap: int,
        random_seed: int | None,
    ) -> dict[Hashable, np.ndarray]:
        """Resample units once per replicate and refit every analyzed level."""
        analyzed = self._analyzed_frame()
        units = np.asarray(pd.unique(analyzed["unit"]))
        rng = default_rng(random_seed)
        samples = [
            rng.choice(units, size=len(units), replace=True) for _ in range(n_bootstrap)
        ]
        collected: dict[Hashable, list[np.ndarray]] = {
            level: [] for level in self._level_order
        }
        for replicate, sample in enumerate(samples):
            resampled = _resample_units(analyzed, sample)
            self._require_comparison_cells(
                context=f"in bootstrap replicate {replicate}",
                frame=resampled,
            )
            for level in self._level_order:
                subset = resampled.loc[resampled[self.level_variable_name] == level]
                vector = _draw_vector(
                    self._fit_level(subset).result.causal_impact,
                    level=level,
                )
                if vector.size != 1:
                    raise ValueError(
                        "n_bootstrap is for a point-estimate model. Posterior "
                        "draws already form the contrast; omit n_bootstrap."
                    )
                collected[level].append(vector)
        return {level: np.concatenate(chunks) for level, chunks in collected.items()}

    def _require_fit(self) -> None:
        """Raise before a result method is used on an unfitted experiment."""
        if (
            self._experiments is None
            or self._effect_draws is None
            or self._uncertainty is None
        ):
            raise RuntimeError(
                "Call fit() before reading difference-in-differences contrasts."
            )

    @property
    def experiments(self) -> Mapping[Hashable, DifferenceInDifferences]:
        """Fitted difference-in-differences experiments, one per analyzed level.

        Returns
        -------
        mapping of hashable to DifferenceInDifferences
            Available after :meth:`fit`. Each value is that level's ordinary
            experiment, including its treated-post counterfactual.
        """
        self._require_fit()
        assert self._experiments is not None
        return self._experiments

    def contrast_draws(self) -> pd.DataFrame:
        """Return within-draw or within-replicate contrast differences.

        Returns
        -------
        DataFrame
            One row per draw or bootstrap replicate and one column per
            requested contrast. Column names match ``table()["label"]``.
        """
        return self._summary().contrast_draws

    def table(self, *, hdi_prob: float = HDI_PROB) -> pd.DataFrame:
        """Return level estimates and signed contrasts with paired-difference uncertainty.

        Parameters
        ----------
        hdi_prob : float, optional
            Probability mass of the highest density interval. Defaults to
            ``causalpy.constants.HDI_PROB``. A single draw leaves the interval
            empty rather than inventing a standard error.

        Returns
        -------
        DataFrame
            Columns ``quantity``, ``label``, ``level_a``, ``level_b``,
            ``estimate``, ``sd``, ``hdi_lower``, ``hdi_upper``, ``n_draws``,
            and ``uncertainty``. Level rows use the data's appearance order.
            Contrast rows follow the requested pairs. ``estimate`` is the
            fitted difference. ``sd`` and the interval describe the paired
            draws, not a difference of marginal standard errors.
        """
        return self._summary(hdi_prob=hdi_prob).table

    def plot(
        self,
        *,
        hdi_prob: float = HDI_PROB,
        figsize: tuple[float, float] | None = None,
        outcome_units: str | None = None,
        show: bool = True,
    ) -> Axes:
        """Plot signed contrasts with the same labels and intervals as :meth:`table`.

        Parameters
        ----------
        hdi_prob : float, optional
            Probability mass of the highest density interval drawn around each
            contrast. Defaults to ``causalpy.constants.HDI_PROB``.
        figsize : tuple of float, optional
            Figure size in inches. The default grows with the number of contrasts.
        outcome_units : str, optional
            Unit named on the horizontal axis, for example ``"sales units"``.
            When omitted, the axis says ``"outcome units"``.
        show : bool, optional
            Whether to display the figure. Defaults to ``True``.

        Returns
        -------
        matplotlib.axes.Axes
            Axes whose y labels are the contrast labels and whose marker
            positions are the contrast estimates. Interval lines, when present,
            span the table's highest density interval.
        """
        summary = self.table(hdi_prob=hdi_prob)
        contrasts = summary.loc[summary["quantity"] == "contrast"].reset_index(
            drop=True
        )
        estimates = contrasts["estimate"].to_numpy(dtype=float)
        y = np.arange(len(contrasts))[::-1]
        if figsize is None:
            figsize = (7.0, max(2.5, 0.55 * len(contrasts) + 1.4))
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        has_interval = bool(contrasts["hdi_lower"].notna().all())
        if has_interval:
            for yi, lower, upper in zip(
                y,
                contrasts["hdi_lower"].to_numpy(dtype=float),
                contrasts["hdi_upper"].to_numpy(dtype=float),
                strict=True,
            ):
                ax.plot(
                    [lower, upper],
                    [yi, yi],
                    color="C0",
                    linewidth=2,
                    solid_capstyle="butt",
                    label="_interval",
                    zorder=2,
                )
        ax.plot(estimates, y, "o", color="C0", label="contrast", zorder=3)
        ax.axvline(0.0, color="0.6", linewidth=1, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(contrasts["label"].tolist())
        title = "Difference-in-differences-in-differences"
        if int(contrasts["n_draws"].iloc[0]) > 1:
            title += (
                "\nintervals from paired differences, not subtracted standard errors"
            )
        ax.set_title(title)
        unit = outcome_units if outcome_units is not None else "outcome units"
        ax.set_xlabel(f"Signed contrast ({unit})")
        finite = estimates[np.isfinite(estimates)]
        if has_interval:
            finite = np.concatenate(
                [
                    finite,
                    contrasts["hdi_lower"].to_numpy(dtype=float),
                    contrasts["hdi_upper"].to_numpy(dtype=float),
                ]
            )
        finite = finite[np.isfinite(finite)]
        if finite.size:
            lo = min(0.0, float(np.min(finite)))
            hi = max(0.0, float(np.max(finite)))
            pad = max(hi - lo, 1.0) * 0.25
            ax.set_xlim(lo - pad, hi + pad * 2)
        if show:
            plt.show()
        return ax

    def _summary(self, hdi_prob: float = HDI_PROB) -> _PairedContrastSummary:
        """Build the paired summary from the stored fits."""
        self._require_fit()
        assert self._effect_draws is not None
        assert self._uncertainty is not None
        return self.summarize_paired_effects(
            self._effect_draws,
            self.contrasts,
            point_estimates=self._point_estimates,
            hdi_prob=hdi_prob,
            uncertainty=self._uncertainty,
            level_order=self._level_order,
        )

    @staticmethod
    def summarize_paired_effects(
        effect_draws: Mapping[Hashable, ArrayLike],
        contrasts: Sequence[Sequence[Hashable]],
        *,
        point_estimates: Mapping[Hashable, float] | None = None,
        hdi_prob: float = HDI_PROB,
        uncertainty: _Uncertainty = "draws",
        level_order: Sequence[Hashable] | None = None,
    ) -> _PairedContrastSummary:
        """Summarize paired effect draws without subtracting standard errors.

        Parameters
        ----------
        effect_draws : mapping of hashable to array-like
            Effect draws for each level, aligned by position. A one-dimensional
            sequence is already that order. An xarray array is aligned on
            ``chain`` then ``draw``.
        contrasts : sequence of pair
            Ordered pairs ``(a, b)`` meaning the draw-wise difference
            ``effect(a) - effect(b)``.
        point_estimates : mapping of hashable to float, optional
            Full-sample level estimates. When omitted, each estimate is the
            mean of that level's draws. Contrast estimates are the corresponding
            difference either way. The interval always comes from the paired
            draws.
        hdi_prob : float, optional
            Probability mass of the highest density interval. Defaults to
            ``causalpy.constants.HDI_PROB``.
        uncertainty : {"draws", "point", "posterior", "unit_bootstrap"}, optional
            Label stored on the table. It does not change the arithmetic.
        level_order : sequence of hashable, optional
            Order of the level rows. Levels outside the contrasts are ignored.
            When omitted, level rows follow first appearance in ``contrasts``.

        Returns
        -------
        summary
            Object with ``table`` and ``contrast_draws``. ``contrast_draws`` has
            one column per contrast and one row per paired difference.

        Raises
        ------
        DataException
            If a contrast level has no draws, draw counts differ, or a supplied
            point estimate is missing.
        ValueError
            If ``uncertainty`` is not one of the documented labels, or a
            contrast is not an ordered pair.
        """
        if uncertainty not in ("draws", "point", "posterior", "unit_bootstrap"):
            raise ValueError(
                "uncertainty must be 'draws', 'point', 'posterior', or "
                f"'unit_bootstrap', got {uncertainty!r}."
            )
        pairs = _as_pairs(contrasts)
        needed = list(dict.fromkeys(level for pair in pairs for level in pair))
        vectors = {
            level: _draw_vector(effect_draws[level], level=level)
            if level in effect_draws
            else None
            for level in needed
        }
        missing = [level for level, values in vectors.items() if values is None]
        if missing:
            raise DataException(
                f"Level {missing[0]!r} has no effect draws to contrast."
            )
        lengths = {
            level: int(values.size)
            for level, values in vectors.items()
            if values is not None
        }
        if len(set(lengths.values())) != 1:
            raise DataException(
                "Paired contrasts need the same number of draws in every "
                f"level, got {lengths}. Independent fits are aligned by draw "
                "index; they are not trimmed."
            )
        points = _resolve_points(needed, vectors, point_estimates)
        labels = _unique_labels(pairs)
        difference_columns: dict[str, np.ndarray] = {}
        for (level_a, level_b), label in zip(pairs, labels, strict=True):
            left = vectors[level_a]
            right = vectors[level_b]
            if left is None or right is None:
                raise DataException(
                    f"Level {level_a!r} or {level_b!r} has no effect draws to contrast."
                )
            difference_columns[label] = left - right
        contrast_draws = pd.DataFrame(difference_columns)
        if level_order is None:
            ordered_levels = needed
        else:
            ordered_levels = [level for level in level_order if level in needed]
            ordered_levels.extend(
                level for level in needed if level not in ordered_levels
            )
        rows: list[dict[str, Any]] = []
        for level in ordered_levels:
            values = vectors[level]
            assert values is not None
            lower, upper = _interval(values, hdi_prob)
            rows.append(
                {
                    "quantity": "level",
                    "label": str(level),
                    "level_a": level,
                    "level_b": pd.NA,
                    "estimate": points[level],
                    "sd": _sample_standard_deviation(values),
                    "hdi_lower": lower,
                    "hdi_upper": upper,
                    "n_draws": int(values.size),
                    "uncertainty": uncertainty,
                }
            )
        for (level_a, level_b), label in zip(pairs, labels, strict=True):
            values = difference_columns[label]
            lower, upper = _interval(values, hdi_prob)
            rows.append(
                {
                    "quantity": "contrast",
                    "label": label,
                    "level_a": level_a,
                    "level_b": level_b,
                    "estimate": points[level_a] - points[level_b],
                    "sd": _sample_standard_deviation(values),
                    "hdi_lower": lower,
                    "hdi_upper": upper,
                    "n_draws": int(values.size),
                    "uncertainty": uncertainty,
                }
            )
        table = pd.DataFrame(rows)
        return _PairedContrastSummary(table=table, contrast_draws=contrast_draws)


def _resolve_points(
    levels: Sequence[Hashable],
    vectors: Mapping[Hashable, np.ndarray | None],
    point_estimates: Mapping[Hashable, float] | None,
) -> dict[Hashable, float]:
    """Return fitted points, or the draw means when no full-sample points were supplied."""
    if point_estimates is None:
        return {
            level: float(np.mean(values))
            for level, values in vectors.items()
            if values is not None
        }
    missing = [level for level in levels if level not in point_estimates]
    if missing:
        raise DataException(
            f"Level {missing[0]!r} has draws but no full-sample point estimate."
        )
    return {level: float(point_estimates[level]) for level in levels}


def _unique_labels(pairs: Sequence[tuple[Hashable, Hashable]]) -> list[str]:
    """Return contrast labels, suffixing duplicates so columns stay unique."""
    seen: dict[str, int] = {}
    labels: list[str] = []
    for level_a, level_b in pairs:
        base = f"{level_a} - {level_b}"
        suffix = seen.get(base, 0)
        seen[base] = suffix + 1
        labels.append(_contrast_label(level_a, level_b, suffix=suffix))
    return labels


def _resample_units(frame: pd.DataFrame, units: ArrayLike) -> pd.DataFrame:
    """Duplicate whole unit histories in draw order, keeping ids unchanged.

    Duplicate unit ids are the cluster bootstrap: a drawn unit is weighted
    again, rather than renamed into a new unit that would change a formula
    with unit fixed effects.
    """
    parts: list[pd.DataFrame] = []
    for unit in np.asarray(units):
        rows = frame.loc[frame["unit"] == unit]
        if rows.empty:
            continue
        parts.append(rows)
    if not parts:
        raise DataException("Bootstrap replicate has no rows.")
    return pd.concat(parts, ignore_index=True)
