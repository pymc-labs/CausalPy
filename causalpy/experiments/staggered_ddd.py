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
"""Staggered difference-in-differences-in-differences.

Fits the imputation staggered DiD estimator once per predefined third-dimension
level, then contrasts those ATTs. It does not fit a second imputation model
and does not read private helpers of the child experiments.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.base import RegressorMixin, clone

from causalpy.constants import HDI_PROB
from causalpy.custom_exceptions import DataException
from causalpy.experiments.staggered_did import StaggeredDifferenceInDifferences
from causalpy.input_data import DataFrameLike, to_pandas
from causalpy.pymc_models import PyMCModel

Draws = xr.DataArray | pd.Series | float


class StaggeredDifferenceInDifferencesInDifferences:
    """Contrast staggered DiD ATTs across a predefined third dimension.

    Each level is fit with
    :class:`~causalpy.experiments.staggered_did.StaggeredDifferenceInDifferences`
    using the same formula, unit and time definition, treatment columns, and
    event window. At event time ``e``, the contrast is ``ATT(a, e) - ATT(b, e)``
    only when both levels identify ``e``. A named ``reference_level`` returns
    one contrast per other level, in the direction other-minus-reference.
    ``contrasts`` returns only those ordered pairs, in the requested direction.
    The overall post-treatment summary is contrasted the same way.

    Bayesian intervals are highest-density intervals of within-draw
    differences. They are not differences of subgroup standard errors or
    interval endpoints. Independent draws are valid when the level fits
    factorize, which this class requires by rejecting a unit that appears in
    more than one level. OLS contrasts are point differences, with no interval.

    Parameters
    ----------
    data : dataframe-like
        Panel data containing the third-dimension column. Converted to pandas.
    formula : str
        Untreated-outcome formula passed to each level's staggered DiD fit.
        Do not include the level column; it is constant inside a level.
    unit_variable_name : str
        Column identifying units. A unit may belong to only one level.
    time_variable_name : str
        Column identifying time periods.
    level_variable_name : str
        Column of predefined third-dimension levels, such as a sugar profile.
    treated_variable_name : str, optional
        Treatment indicator column. Defaults to ``"treated"``.
    treatment_time_variable_name : str, optional
        Unit-level treatment time column. If None, treatment time is inferred.
    never_treated_value : Any, optional
        Value marking never-treated units. Defaults to ``np.inf``.
    model : PyMCModel or RegressorMixin, optional
        Unfitted model template. Each level receives its own copy. Defaults
        to the staggered DiD default model.
    event_window : tuple of int, optional
        Event-time window passed unchanged to every level.
    reference_event_time : int, optional
        Passed unchanged to every level. Defaults to ``-1``.
    reference_level : Any, optional
        Level subtracted from every other level. Mutually exclusive with
        ``contrasts``.
    contrasts : sequence of (Any, Any), optional
        Ordered pairs ``(a, b)`` whose contrast is ``ATT(a) - ATT(b)``.
        Mutually exclusive with ``reference_level``.

    Notes
    -----
    Construction validates the panel and builds one unfitted staggered DiD
    experiment per level. Call :meth:`fit` before reading contrasts.
    """

    def __init__(
        self,
        data: DataFrameLike,
        formula: str,
        unit_variable_name: str,
        time_variable_name: str,
        level_variable_name: str,
        treated_variable_name: str = "treated",
        treatment_time_variable_name: str | None = None,
        never_treated_value: Any = np.inf,
        model: PyMCModel | RegressorMixin | None = None,
        event_window: tuple[int, int] | None = None,
        reference_event_time: int = -1,
        *,
        reference_level: Any = None,
        contrasts: list[tuple[Any, Any]] | None = None,
    ) -> None:
        panel = to_pandas(data)
        self.formula = formula
        self.unit_variable_name = unit_variable_name
        self.time_variable_name = time_variable_name
        self.level_variable_name = level_variable_name
        self.event_window = event_window
        self._validate_panel(panel)
        self.levels = list(pd.unique(panel[level_variable_name]))
        if len(self.levels) < 2:
            raise DataException(
                "Staggered DDD needs at least two third-dimension levels."
            )
        self._validate_contrast_request(reference_level, contrasts)
        self.reference_level = reference_level
        self.contrasts = contrasts
        self._pairs = self._resolve_pairs()
        self._validate_units(panel)

        experiments: dict[Any, StaggeredDifferenceInDifferences] = {}
        for level in self.levels:
            level_data = panel.loc[panel[level_variable_name] == level].copy()
            experiments[level] = StaggeredDifferenceInDifferences(
                level_data,
                formula=formula,
                unit_variable_name=unit_variable_name,
                time_variable_name=time_variable_name,
                treated_variable_name=treated_variable_name,
                treatment_time_variable_name=treatment_time_variable_name,
                never_treated_value=never_treated_value,
                model=_fresh_model(model),
                event_window=event_window,
                reference_event_time=reference_event_time,
            )
        self.experiments = experiments
        self._validate_adoption_timing()

    def fit(self) -> Self:
        """Fit each level's staggered DiD experiment and return this object."""
        for experiment in self.experiments.values():
            experiment.fit()
        return self

    def level_att_event_time(self) -> pd.DataFrame:
        """Event-time ATT table for every level, from each fit's public result.

        Returns
        -------
        pandas.DataFrame
            Child ``result.att_event_time`` rows with a ``level`` column.
        """
        frames = []
        for level, experiment in self.experiments.items():
            table = experiment.result.att_event_time.copy()
            table.insert(0, "level", level)
            frames.append(table)
        return pd.concat(frames, ignore_index=True)

    def contrast_att_event_time(
        self,
        *,
        hdi_prob: float = HDI_PROB,
    ) -> pd.DataFrame:
        """Event-time contrasts identified on both sides of each pair.

        Parameters
        ----------
        hdi_prob : float, optional
            Probability mass of the Bayesian HDI of the within-draw
            difference. Defaults to :data:`~causalpy.constants.HDI_PROB`
            (currently 0.94). Ignored for OLS point contrasts.

        Returns
        -------
        pandas.DataFrame
            Columns ``level_a``, ``level_b``, ``event_time``, and ``att``.
            Bayesian rows also have ``att_lower`` and ``att_upper``. An
            event time identified on only one side is absent.
        """
        rows: list[dict[str, Any]] = []
        bayesian = False
        for level_a, level_b in self._pairs:
            left = self.experiments[level_a].att_event_time_draws()
            right = self.experiments[level_b].att_event_time_draws()
            for event_time in _shared_event_times(left, right):
                difference = _difference_at(left, right, event_time)
                row: dict[str, Any] = {
                    "level_a": level_a,
                    "level_b": level_b,
                    "event_time": event_time,
                    "att": _point(difference),
                }
                if isinstance(difference, np.ndarray):
                    bayesian = True
                    lower, upper = _hdi(difference, hdi_prob)
                    row["att_lower"] = lower
                    row["att_upper"] = upper
                rows.append(row)
        columns = ["level_a", "level_b", "event_time", "att"]
        if bayesian:
            columns.extend(["att_lower", "att_upper"])
        return pd.DataFrame(rows, columns=columns)

    def contrast_att_overall(
        self,
        *,
        hdi_prob: float = HDI_PROB,
    ) -> pd.DataFrame:
        """Contrast of each pair's overall post-treatment ATT.

        A pair is omitted when either level has no identified post-treatment
        event time. The difference is within-draw for Bayesian fits.

        Parameters
        ----------
        hdi_prob : float, optional
            Probability mass of the Bayesian HDI. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94). Ignored
            for OLS point contrasts.

        Returns
        -------
        pandas.DataFrame
            Columns ``level_a``, ``level_b``, and ``att``, plus
            ``att_lower`` and ``att_upper`` for Bayesian fits.
        """
        rows: list[dict[str, Any]] = []
        bayesian = False
        for level_a, level_b in self._pairs:
            try:
                left = self.experiments[level_a].att_overall_draws()
                right = self.experiments[level_b].att_overall_draws()
            except DataException:
                continue
            difference = _subtract(left, right)
            row: dict[str, Any] = {
                "level_a": level_a,
                "level_b": level_b,
                "att": _point(difference),
            }
            if isinstance(difference, np.ndarray):
                bayesian = True
                lower, upper = _hdi(difference, hdi_prob)
                row["att_lower"] = lower
                row["att_upper"] = upper
            rows.append(row)
        columns = ["level_a", "level_b", "att"]
        if bayesian:
            columns.extend(["att_lower", "att_upper"])
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def contrast_draws(draws_a: np.ndarray, draws_b: np.ndarray) -> np.ndarray:
        """Difference within each draw or shared bootstrap replicate.

        Parameters
        ----------
        draws_a : numpy.ndarray
            Effect draws for the first side.
        draws_b : numpy.ndarray
            Effect draws for the second side. Must have the same shape as
            ``draws_a``.

        Returns
        -------
        numpy.ndarray
            ``draws_a - draws_b``. Not a difference of standard errors or
            interval endpoints.

        Raises
        ------
        DataException
            When the two arrays do not share a shape.
        """
        return _subtract_arrays(draws_a, draws_b)

    def plot(
        self,
        *,
        figsize: tuple[float, float] = (10, 10),
        hdi_prob: float = HDI_PROB,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot level curves, contrast curves, and the overall contrast.

        Plotted values are the tables from :meth:`level_att_event_time`,
        :meth:`contrast_att_event_time`, and :meth:`contrast_att_overall`.

        Parameters
        ----------
        figsize : tuple of float, optional
            Width and height of the figure in inches. Defaults to ``(10, 10)``.
        hdi_prob : float, optional
            Probability mass of Bayesian intervals. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94). Ignored
            for OLS point contrasts.

        Returns
        -------
        tuple of matplotlib.figure.Figure and list of matplotlib.axes.Axes
            Three axes: event-time ATT by level, event-time contrast, and
            overall post-treatment contrast.
        """
        level_table = self.level_att_event_time()
        contrast_table = self.contrast_att_event_time(hdi_prob=hdi_prob)
        overall = self.contrast_att_overall(hdi_prob=hdi_prob)
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        _plot_level_curves(axes[0], level_table)
        _plot_contrast_curves(axes[1], contrast_table)
        _plot_overall(axes[2], overall)
        fig.tight_layout()
        return fig, list(axes)

    def _validate_panel(self, panel: pd.DataFrame) -> None:
        required = [
            self.unit_variable_name,
            self.time_variable_name,
            self.level_variable_name,
        ]
        missing = [name for name in required if name not in panel.columns]
        if missing:
            raise DataException(f"Missing required columns: {missing}")
        if panel[self.level_variable_name].isna().any():
            raise DataException(
                f"{self.level_variable_name} has missing levels. "
                "Levels must be predefined."
            )

    def _validate_contrast_request(
        self,
        reference_level: Any,
        contrasts: list[tuple[Any, Any]] | None,
    ) -> None:
        if (reference_level is None) == (contrasts is None):
            raise DataException(
                "Pass reference_level or contrasts, not both and not neither."
            )
        if contrasts is None:
            if reference_level not in self.levels:
                raise DataException(
                    f"reference_level {reference_level!r} is not in "
                    f"{self.level_variable_name}."
                )
            return
        if len(contrasts) == 0:
            raise DataException("contrasts must contain at least one ordered pair.")
        for pair in contrasts:
            if len(pair) != 2:
                raise DataException(
                    "Each contrast must be an ordered pair (level_a, level_b)."
                )
            for level in pair:
                if level not in self.levels:
                    raise DataException(
                        f"Contrast level {level!r} is not in "
                        f"{self.level_variable_name}."
                    )

    def _resolve_pairs(self) -> list[tuple[Any, Any]]:
        if self.contrasts is not None:
            return [(pair[0], pair[1]) for pair in self.contrasts]
        return [
            (level, self.reference_level)
            for level in self.levels
            if level != self.reference_level
        ]

    def _validate_units(self, panel: pd.DataFrame) -> None:
        unit_levels = panel.groupby(self.unit_variable_name, observed=True)[
            self.level_variable_name
        ].nunique()
        shared = unit_levels[unit_levels > 1]
        if len(shared) > 0:
            names = ", ".join(str(unit) for unit in shared.index[:5])
            raise DataException(
                "A unit appears in more than one third-dimension level "
                f"({names}). Independent level fits would ignore that "
                "dependence. Split shared units before contrasting."
            )

    def _validate_adoption_timing(self) -> None:
        timings = {
            level: _adoption_times(experiment)
            for level, experiment in self.experiments.items()
        }
        expected = next(iter(timings.values()))
        mismatched = [level for level, timing in timings.items() if timing != expected]
        if mismatched:
            raise DataException(
                "Every level must use the same adoption timing. "
                f"Levels {mismatched} do not match {expected}."
            )


def _fresh_model(
    model: PyMCModel | RegressorMixin | None,
) -> PyMCModel | RegressorMixin | None:
    if model is None:
        return None
    cloner = getattr(model, "_clone", None)
    if callable(cloner):
        return cloner()
    return clone(model)


def _adoption_times(experiment: StaggeredDifferenceInDifferences) -> tuple[Any, ...]:
    return tuple(experiment.cohorts)


def _event_times(draws: Draws) -> list[int]:
    if isinstance(draws, pd.Series):
        return [int(event_time) for event_time in draws.index]
    if isinstance(draws, xr.DataArray):
        return [int(event_time) for event_time in draws.coords["event_time"].values]
    return []


def _shared_event_times(left: Draws, right: Draws) -> list[int]:
    right_times = set(_event_times(right))
    return [
        event_time for event_time in _event_times(left) if event_time in right_times
    ]


def _value_at(draws: Draws, event_time: int) -> xr.DataArray | float:
    if isinstance(draws, pd.Series):
        return float(draws.loc[event_time])
    if isinstance(draws, xr.DataArray):
        return draws.sel(event_time=event_time)
    raise DataException("Event-time draws are missing.")


def _subtract(left: Draws, right: Draws) -> np.ndarray | float:
    if isinstance(left, xr.DataArray) and isinstance(right, xr.DataArray):
        return _subtract_arrays(left.values, right.values)
    if isinstance(left, pd.Series) or isinstance(right, pd.Series):
        raise DataException("Cannot subtract event-time tables as overall draws.")
    return float(left) - float(right)


def _difference_at(left: Draws, right: Draws, event_time: int) -> np.ndarray | float:
    return _subtract(_value_at(left, event_time), _value_at(right, event_time))


def _subtract_arrays(draws_a: np.ndarray, draws_b: np.ndarray) -> np.ndarray:
    left = np.asarray(draws_a, dtype=float)
    right = np.asarray(draws_b, dtype=float)
    if left.shape != right.shape:
        raise DataException(
            "Contrast draws must share a shape so each difference stays "
            f"within a draw. Got {left.shape} and {right.shape}."
        )
    return left - right


def _point(difference: np.ndarray | float) -> float:
    if isinstance(difference, np.ndarray):
        return float(difference.mean())
    return float(difference)


def _hdi(difference: np.ndarray, hdi_prob: float) -> tuple[float, float]:
    from causalpy._arviz_compat import hdi_bounds

    return hdi_bounds(difference, prob=hdi_prob, flatten_chains_draws=True)


def _errorbar(
    ax: plt.Axes,
    x: np.ndarray,
    frame: pd.DataFrame,
    label: str,
    *,
    connect: bool = True,
) -> None:
    if {"att_lower", "att_upper"}.issubset(frame.columns):
        yerr = np.vstack(
            [
                frame["att"].to_numpy() - frame["att_lower"].to_numpy(),
                frame["att_upper"].to_numpy() - frame["att"].to_numpy(),
            ]
        )
        ax.errorbar(x, frame["att"], yerr=yerr, fmt="o", capsize=3, label=label)
        return
    ax.plot(
        x,
        frame["att"],
        marker="o",
        linestyle="-" if connect else "none",
        label=label,
    )


def _plot_level_curves(ax: plt.Axes, table: pd.DataFrame) -> None:
    ax.set_title("Event-time ATT by level")
    ax.set_xlabel("Event time")
    ax.set_ylabel("ATT")
    identified = table
    if "identified" in table.columns:
        identified = table.loc[table["identified"].astype(bool)]
    for level, frame in identified.groupby("level", sort=False, observed=True):
        _errorbar(ax, frame["event_time"].to_numpy(), frame, str(level))
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.legend(title="Level")


def _plot_contrast_curves(ax: plt.Axes, table: pd.DataFrame) -> None:
    ax.set_title("Event-time contrast")
    ax.set_xlabel("Event time")
    ax.set_ylabel("ATT difference")
    if len(table) == 0:
        ax.text(0.5, 0.5, "No shared identified event time", ha="center", va="center")
        return
    grouped = table.groupby(["level_a", "level_b"], sort=False, observed=True)
    for (level_a, level_b), frame in grouped:
        _errorbar(
            ax,
            frame["event_time"].to_numpy(),
            frame,
            f"{level_a} - {level_b}",
        )
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.legend(title="Contrast")


def _plot_overall(ax: plt.Axes, table: pd.DataFrame) -> None:
    ax.set_title("Overall post-treatment contrast")
    ax.set_ylabel("ATT difference")
    if len(table) == 0:
        ax.set_xticks([])
        ax.text(0.5, 0.5, "No overall contrast", ha="center", va="center")
        return
    labels = [f"{row.level_a} - {row.level_b}" for row in table.itertuples()]
    x = np.arange(len(table))
    _errorbar(ax, x, table, "Overall contrast", connect=False)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Contrast")
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.legend()
