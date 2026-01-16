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
"""
Panel fixed effects regression experiment.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from patsy import dmatrices
from scipy import stats
from sklearn.base import RegressorMixin

from causalpy.custom_exceptions import DataException, FormulaException
from causalpy.plot_utils import get_hdi_to_df
from causalpy.pymc_models import PyMCModel
from causalpy.reporting import EffectSummary

from .base import BaseExperiment

LEGEND_FONT_SIZE = 12


class PanelRegression(BaseExperiment):
    """Panel fixed effects regression with optional within transformation."""

    supports_ols = True
    supports_bayes = True

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        unit_fe_variable: str,
        time_fe_variable: str | None = None,
        fe_method: Literal["dummies", "within"] = "dummies",
        model: PyMCModel | RegressorMixin | None = None,
        **kwargs: dict,
    ) -> None:
        super().__init__(model=model)
        self.expt_type = "Panel Regression"
        self.formula = formula
        self.unit_fe_variable = unit_fe_variable
        self.time_fe_variable = time_fe_variable
        self.fe_method = fe_method.lower()
        self._group_means: dict[str, pd.DataFrame] = {}

        self.original_data = data.copy()
        self.original_data.index.name = "obs_ind"

        self.input_validation()

        self.n_units = int(self.original_data[self.unit_fe_variable].nunique())
        self.n_periods = (
            int(self.original_data[self.time_fe_variable].nunique())
            if self.time_fe_variable is not None
            else None
        )
        self._all_units = list(self.original_data[self.unit_fe_variable].unique())

        self.model_data = self._prepare_model_data(self.original_data)

        y_df, X_df = dmatrices(self.formula, self.model_data, return_type="dataframe")
        self._design_data = self.model_data.loc[y_df.index].copy()
        self._design_data_raw = self.original_data.loc[y_df.index].copy()

        self._y_design_info = y_df.design_info
        self._x_design_info = X_df.design_info
        self.labels = list(X_df.columns)
        self.outcome_variable_name = y_df.design_info.column_names[0]

        self.y = xr.DataArray(
            y_df.values,
            dims=["obs_ind", "treated_units"],
            coords={
                "obs_ind": np.arange(y_df.shape[0]),
                "treated_units": ["unit_0"],
            },
        )
        self.X = xr.DataArray(
            X_df.values,
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": np.arange(X_df.shape[0]), "coeffs": self.labels},
        )

        if isinstance(self.model, PyMCModel):
            coords = {
                "coeffs": self.labels,
                "obs_ind": np.arange(self.X.shape[0]),
                "treated_units": ["unit_0"],
            }
            self.model.fit(X=self.X, y=self.y, coords=coords)
        elif isinstance(self.model, RegressorMixin):
            if hasattr(self.model, "fit_intercept"):
                self.model.fit_intercept = False
            self.model.fit(X=self.X, y=self.y)
        else:
            raise ValueError("Model type not recognized")

        self.score = self.model.score(X=self.X, y=self.y)
        self.pred = self.model.predict(X=self.X)

        self._observed = np.squeeze(y_df.values)
        self._pred_mean = self._prediction_mean()
        self.residuals = self._observed - self._pred_mean

    def _prepare_model_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.fe_method == "within":
            data = self._within_transform(data, self.unit_fe_variable)
            if self.time_fe_variable is not None:
                data = self._within_transform(data, self.time_fe_variable)
        return data

    def _within_transform(self, data: pd.DataFrame, group_var: str) -> pd.DataFrame:
        data = data.copy()
        numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
        exclude_cols = {group_var, self.unit_fe_variable}
        if self.time_fe_variable is not None:
            exclude_cols.add(self.time_fe_variable)
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        if not numeric_cols:
            return data
        group_means = data.groupby(group_var)[numeric_cols].transform("mean")
        self._group_means[group_var] = data.groupby(group_var)[numeric_cols].mean()
        data.loc[:, numeric_cols] = data[numeric_cols].sub(group_means)
        return data

    def input_validation(self) -> None:
        if self.unit_fe_variable not in self.original_data.columns:
            raise DataException(
                f"unit_fe_variable '{self.unit_fe_variable}' not in data columns"
            )
        if self.time_fe_variable is not None and (
            self.time_fe_variable not in self.original_data.columns
        ):
            raise DataException(
                f"time_fe_variable '{self.time_fe_variable}' not in data columns"
            )
        if self.fe_method not in {"dummies", "within"}:
            raise ValueError("fe_method must be one of {'dummies', 'within'}")

        unit_term = self._category_term(self.unit_fe_variable)
        time_term = (
            self._category_term(self.time_fe_variable)
            if self.time_fe_variable is not None
            else None
        )

        has_unit_term = self._formula_has_term(unit_term)
        has_time_term = (
            self._formula_has_term(time_term) if time_term is not None else False
        )

        if self.fe_method == "dummies" and not has_unit_term:
            raise FormulaException(
                f"fe_method='dummies' requires {unit_term} in the formula"
            )
        if self.fe_method == "within" and has_unit_term:
            raise FormulaException(
                f"fe_method='within' should not include {unit_term} in the formula"
            )
        if self.fe_method == "within" and has_time_term:
            raise FormulaException(
                f"fe_method='within' should not include {time_term} in the formula"
            )

    def _category_term(self, variable: str | None) -> str:
        if variable is None:
            return ""
        return f"C({variable})"

    def _formula_has_term(self, term: str | None) -> bool:
        if term is None or term == "":
            return False
        return term in self.formula

    def _prediction_mean(self) -> np.ndarray:
        if isinstance(self.model, PyMCModel):
            pred = self.pred
            mu = az.extract(pred, group="posterior_predictive", var_names="mu")
            if "treated_units" in mu.dims:
                mu = mu.sel(treated_units=mu.coords["treated_units"].values[0])
            return mu.mean("sample").values
        return np.asarray(self.pred).squeeze()

    def _plot_data_base(self) -> pd.DataFrame:
        plot_df = self._design_data_raw.copy()
        plot_df[self.outcome_variable_name] = self._observed
        plot_df["prediction"] = self._pred_mean
        plot_df["residual"] = self.residuals
        return plot_df

    def summary(self, round_to: int | None = None) -> None:
        """Print summary of panel dimensions and coefficients."""
        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print(f"Units: {self.n_units} ({self.unit_fe_variable})")
        if self.n_periods is not None:
            print(f"Periods: {self.n_periods} ({self.time_fe_variable})")
        print(f"FE method: {self.fe_method}")
        self.print_coefficients(round_to)

    def plot_coefficients(
        self,
        var_names: list[str] | None = None,
        hdi_prob: float = 0.94,
    ) -> Figure:
        """Forest plot of covariate coefficients excluding FE coefficients."""
        if var_names is None:
            var_names = self._non_fe_labels()
        missing = [name for name in var_names if name not in self.labels]
        if missing:
            raise ValueError(f"Unknown coefficient names: {missing}")

        fig, ax = plt.subplots(figsize=(7, max(2, 0.3 * len(var_names))))

        if isinstance(self.model, PyMCModel):
            assert self.model.idata is not None
            coeffs: xr.DataArray = az.extract(
                self.model.idata.posterior, var_names="beta"
            )
            coeffs = coeffs.sel(treated_units=coeffs.coords["treated_units"].values[0])
            coeffs = coeffs.sel(coeffs=var_names)
            hdi = az.hdi(coeffs, hdi_prob=hdi_prob)
            means = coeffs.mean("sample").values
            lower = hdi.sel(hdi="lower").values
            upper = hdi.sel(hdi="higher").values
            y_pos = np.arange(len(var_names))
            ax.errorbar(
                means,
                y_pos,
                xerr=[means - lower, upper - means],
                fmt="o",
                color="C0",
                capsize=3,
            )
        else:
            coef_map = dict(zip(self.labels, self.model.get_coeffs(), strict=False))
            means = np.array([coef_map[name] for name in var_names])
            y_pos = np.arange(len(var_names))
            ax.plot(means, y_pos, "o", color="C0")

        ax.axvline(0, color="k", lw=1, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(var_names)
        ax.set_title("Coefficient estimates")
        ax.set_xlabel("Estimate")
        return fig

    def plot_unit_effects(
        self,
        highlight: list[str] | None = None,
        label_extreme: int = 0,
    ) -> Figure:
        """Plot distribution of unit fixed effects (dummies only)."""
        if self.fe_method != "dummies":
            raise ValueError(
                "plot_unit_effects is only available for fe_method='dummies'"
            )

        unit_effects = self._unit_effects_mean()
        if unit_effects.empty:
            raise ValueError("No unit fixed effects found in the model coefficients")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(unit_effects.to_numpy(), bins=30, alpha=0.7, color="C0")
        ax.set_title("Unit fixed effects")
        ax.set_xlabel("Effect estimate")
        ax.set_ylabel("Count")

        if highlight is not None:
            for unit in highlight:
                if unit in unit_effects.index:
                    ax.axvline(unit_effects.loc[unit], color="C1", lw=2)

        if label_extreme > 0:
            extreme = pd.concat(
                [
                    unit_effects.nsmallest(label_extreme),
                    unit_effects.nlargest(label_extreme),
                ]
            )
            for unit, value in extreme.items():
                ax.annotate(
                    str(unit),
                    xy=(value, 0),
                    xytext=(0, 8),
                    textcoords="offset points",
                    rotation=90,
                    ha="center",
                )

        return fig

    def plot_trajectories(
        self,
        units: list[str] | None = None,
        n_sample: int = 10,
        select: Literal["random", "extreme", "high_variance"] = "random",
        show_mean: bool = True,
    ) -> Figure:
        """Plot observed vs predicted trajectories for selected units."""
        if self.time_fe_variable is None:
            raise ValueError("plot_trajectories requires time_fe_variable to be set")

        plot_df = self._plot_data_base()
        plot_df = plot_df.sort_values(self.time_fe_variable)

        plot_units = self._select_units(plot_df, units, n_sample, select)
        n_units = len(plot_units)
        ncols = min(3, n_units)
        nrows = int(np.ceil(n_units / ncols))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(4 * ncols, 2.5 * nrows),
            sharex=True,
        )
        axes = np.atleast_1d(axes).ravel()

        mean_series = None
        if show_mean:
            mean_series = (
                plot_df.groupby(self.time_fe_variable)[
                    [self.outcome_variable_name, "prediction"]
                ]
                .mean()
                .reset_index()
            )

        for ax, unit in zip(axes, plot_units, strict=False):
            unit_data = plot_df[plot_df[self.unit_fe_variable] == unit]
            ax.plot(
                unit_data[self.time_fe_variable],
                unit_data[self.outcome_variable_name],
                "o-",
                ms=3,
                label="Observed",
            )
            ax.plot(
                unit_data[self.time_fe_variable],
                unit_data["prediction"],
                "o-",
                ms=3,
                label="Predicted",
            )
            if mean_series is not None:
                ax.plot(
                    mean_series[self.time_fe_variable],
                    mean_series[self.outcome_variable_name],
                    color="k",
                    alpha=0.4,
                    label="Mean observed",
                )
            ax.set_title(f"{self.unit_fe_variable}={unit}")

        for ax in axes[n_units:]:
            ax.axis("off")

        axes[0].legend(fontsize=LEGEND_FONT_SIZE)
        fig.tight_layout()
        return fig

    def plot_residuals(
        self,
        kind: Literal["scatter", "histogram", "qq"] = "scatter",
        by: str | None = None,
    ) -> Figure:
        """Residual diagnostic plots."""
        plot_df = self._plot_data_base()

        if by is not None:
            group_var = self._resolve_group_var(by)
            plot_df = self._sample_groups(plot_df, group_var)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(
                data=plot_df,
                x=group_var,
                y="residual",
                ax=ax,
            )
            ax.set_title(f"Residuals by {group_var}")
            ax.set_xlabel(group_var)
            ax.set_ylabel("Residual")
            ax.tick_params(axis="x", rotation=45)
            return fig

        fig, ax = plt.subplots(figsize=(6, 4))
        if kind == "scatter":
            ax.scatter(plot_df["prediction"], plot_df["residual"], alpha=0.6)
            ax.axhline(0, color="k", lw=1)
            ax.set_xlabel("Fitted values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs fitted")
        elif kind == "histogram":
            ax.hist(plot_df["residual"], bins=30, alpha=0.7, color="C0")
            ax.set_xlabel("Residual")
            ax.set_ylabel("Count")
            ax.set_title("Residual distribution")
        elif kind == "qq":
            stats.probplot(plot_df["residual"], dist="norm", plot=ax)
            ax.set_title("Normal Q-Q plot")
        else:
            raise ValueError("kind must be one of {'scatter', 'histogram', 'qq'}")
        return fig

    def _resolve_group_var(self, by: str) -> str:
        if by == "unit":
            return self.unit_fe_variable
        if by == "time":
            if self.time_fe_variable is None:
                raise ValueError("time_fe_variable is required for by='time'")
            return self.time_fe_variable
        if by not in self._plot_data_base().columns:
            raise ValueError(f"'{by}' not found in plot data columns")
        return by

    def _sample_groups(self, plot_df: pd.DataFrame, group_var: str) -> pd.DataFrame:
        unique_groups = np.asarray(plot_df[group_var].unique(), dtype=object)
        if len(unique_groups) <= 30:
            return plot_df
        warnings.warn(
            f"Sampling 30 groups for '{group_var}' residual plot (out of {len(unique_groups)}).",
            UserWarning,
            stacklevel=2,
        )
        rng = np.random.default_rng(42)
        sample_groups = rng.choice(unique_groups, size=30, replace=False)
        return plot_df[plot_df[group_var].isin(sample_groups)]

    def _select_units(
        self,
        plot_df: pd.DataFrame,
        units: list[str] | None,
        n_sample: int,
        select: str,
    ) -> list[str]:
        all_units = list(plot_df[self.unit_fe_variable].unique())
        if units is not None:
            missing = [unit for unit in units if unit not in all_units]
            if missing:
                raise ValueError(f"Units not found in data: {missing}")
            return units
        if len(all_units) <= n_sample:
            return all_units
        if select == "random":
            rng = np.random.default_rng(42)
            return list(rng.choice(all_units, size=n_sample, replace=False))
        if select == "extreme":
            effects = self._unit_effects_for_selection(plot_df)
            extreme = pd.concat(
                [effects.nsmallest(n_sample // 2), effects.nlargest(n_sample // 2)]
            )
            return list(extreme.index)
        if select == "high_variance":
            variances = (
                plot_df.groupby(self.unit_fe_variable)[self.outcome_variable_name]
                .var()
                .sort_values(ascending=False)
            )
            return list(variances.head(n_sample).index)
        raise ValueError("select must be one of {'random', 'extreme', 'high_variance'}")

    def _unit_effects_for_selection(self, plot_df: pd.DataFrame) -> pd.Series:
        if self.fe_method == "dummies":
            return self._unit_effects_mean()
        return (
            plot_df.groupby(self.unit_fe_variable)[self.outcome_variable_name]
            .mean()
            .sort_values()
        )

    def _unit_effects_mean(self) -> pd.Series:
        unit_label_map = self._unit_effect_label_map()
        if not unit_label_map:
            return pd.Series(dtype=float)
        coeff_labels = list(unit_label_map.keys())
        if isinstance(self.model, PyMCModel):
            assert self.model.idata is not None
            coeffs: xr.DataArray = az.extract(
                self.model.idata.posterior, var_names="beta"
            )
            coeffs = coeffs.sel(treated_units=coeffs.coords["treated_units"].values[0])
            coeffs = coeffs.sel(coeffs=coeff_labels)
            means = coeffs.mean("sample").values
        else:
            coef_map = dict(zip(self.labels, self.model.get_coeffs(), strict=False))
            means = np.array([coef_map[label] for label in coeff_labels])
        unit_ids = [unit_label_map[label] for label in coeff_labels]
        return pd.Series(means, index=unit_ids).sort_values()

    def _unit_effect_label_map(self) -> dict[str, str]:
        pattern = rf"C\({re.escape(self.unit_fe_variable)}\)\[T\.(.+)\]"
        label_map = {}
        for label in self.labels:
            match = re.match(pattern, label)
            if match:
                label_map[label] = match.group(1)
        return label_map

    def _non_fe_labels(self) -> list[str]:
        unit_term = self._category_term(self.unit_fe_variable)
        time_term = self._category_term(self.time_fe_variable)
        labels = [
            label
            for label in self.labels
            if unit_term not in label and (time_term is None or time_term not in label)
        ]
        return labels

    def get_plot_data_ols(self) -> pd.DataFrame:
        """Return data with fitted values and residuals for OLS models."""
        if isinstance(self.model, PyMCModel):
            raise ValueError("Unsupported model type")
        self.plot_data = self._plot_data_base()
        return self.plot_data

    def get_plot_data_bayesian(self, hdi_prob: float = 0.94) -> pd.DataFrame:
        """Return data with fitted values, residuals, and prediction HDI."""
        if not isinstance(self.model, PyMCModel):
            raise ValueError("Unsupported model type")

        hdi_pct = int(round(hdi_prob * 100))
        pred_lower_col = f"pred_hdi_lower_{hdi_pct}"
        pred_upper_col = f"pred_hdi_upper_{hdi_pct}"

        plot_df = self._plot_data_base()
        hdi = get_hdi_to_df(
            self.pred["posterior_predictive"].mu.isel(treated_units=0),
            hdi_prob=hdi_prob,
        )
        plot_df[[pred_lower_col, pred_upper_col]] = hdi.iloc[:, [0, -1]].values
        self.plot_data = plot_df
        return self.plot_data

    def effect_summary(self, **kwargs: Any) -> EffectSummary:
        raise NotImplementedError(
            "effect_summary is not implemented for PanelRegression"
        )
