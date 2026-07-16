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
"""Hierarchical difference in differences."""

import warnings
from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from causalpy.constants import HDI_PROB, LEGEND_FONT_SIZE
from causalpy.custom_exceptions import DataException, FormulaException
from causalpy.experiments.base import BaseExperiment
from causalpy.formula import parse_formula
from causalpy.pymc_models import HierarchicalLinearRegression, PyMCModel
from causalpy.reporting import EffectSummary


class HierarchicalDifferenceInDifferences(BaseExperiment):
    """Analyse hierarchical difference-in-differences designs.

    This Bayesian experiment class extends
    :class:`~causalpy.experiments.DifferenceInDifferences` to panel data where
    observations are nested within groups, such as patients within clinics or
    customers within stores. The formula declares the outcome, fixed effects,
    and group-level random effects using lme4-style syntax, for example
    ``(post_treatment:treated | store_id)``. The random-effects grouping
    variable is inferred from the formula and is distinct from the
    treated/control indicator.

    The reported causal impact is the posterior distribution for the
    population-average coefficient on the interaction between
    ``post_treatment_variable_name`` and ``treated_variable_name``. Group-level
    random effects are available for partial-pooling diagnostics and
    heterogeneity inspection.

    Parameters
    ----------
    data : pd.DataFrame
        Balanced panel data for a simultaneous-adoption DiD design. Each unit
        must appear in each observed time period.
    formula : str
        Mixed-effects formula containing one random-effects component. For
        example,
        ``"purchase_amount ~ 1 + post_treatment + treated + post_treatment:treated + age + (post_treatment:treated | store_id)"``.
    time_variable_name : str
        Column identifying time periods.
    unit_variable_name : str
        Column identifying panel units.
    treated_variable_name : str, default="treated"
        Binary column identifying treated/control assignment.
    post_treatment_variable_name : str, default="post_treatment"
        Binary column identifying post-treatment periods.
    model : PyMCModel, optional
        Bayesian model backend. Defaults to
        :class:`~causalpy.pymc_models.HierarchicalLinearRegression`.
    non_centered : bool, default=True
        Whether to use the non-centered hierarchical parameterization.
    **kwargs
        Additional keyword arguments accepted for API compatibility with other
        experiment classes.

    Attributes
    ----------
    causal_impact : xr.DataArray
        Posterior samples for the population-average ATT.
    icc : xr.DataArray or None
        Intraclass correlation for the random-intercept component when the
        posterior contains the required variance parameters.

    Notes
    -----
    This estimator requires one random-effects component, one random-effects
    grouping variable, a balanced panel with one observation per unit and time
    period, fixed random-effects group membership for each unit, treatment
    assignment that remains constant within each random-effects group, and a
    shared transition from pre- to post-treatment periods.

    For MCMC diagnostics, use ArviZ directly on ``result.model.idata``, for
    example :func:`arviz.plot_trace` or :func:`arviz.plot_ppc`.
    """

    supports_ols = False
    supports_bayes = True
    _default_model_class = HierarchicalLinearRegression

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        time_variable_name: str,
        unit_variable_name: str,
        *,
        treated_variable_name: str = "treated",
        post_treatment_variable_name: str = "post_treatment",
        model: PyMCModel | None = None,
        non_centered: bool = True,
        **kwargs: Any,
    ) -> None:
        self.expt_type = "Hierarchical Difference in Differences"
        self.formula = formula
        self.time_variable_name = time_variable_name
        self.unit_variable_name = unit_variable_name
        self.treated_variable_name = treated_variable_name
        self.post_treatment_variable_name = post_treatment_variable_name
        self.non_centered = non_centered

        self.data = data.copy()
        self.data.index.name = "obs_ind"
        self.icc: xr.DataArray | None = None
        self.y_pred_counterfactual: xr.DataArray | None = None

        if model is None:
            model = HierarchicalLinearRegression()
        super().__init__(model=model)

        self._build_design_matrices()
        self.input_validation()
        self._prepare_data()
        self.algorithm()

    def _build_design_matrices(self) -> None:
        """Build mixed-model matrices from the formula."""
        self.matrices = parse_formula(self.formula, self.data)
        self._formula_info = self.matrices.metadata
        self.outcome_variable_name = self._formula_info["outcome_name"]
        self.labels = list(self._formula_info["fixed_effect_names"])
        self.random_effect_labels = list(self._formula_info["random_effect_names"])
        self.group_info = dict(self._formula_info["group"])
        if not self._formula_info["has_random_effects"]:
            raise FormulaException(
                "HierarchicalDifferenceInDifferences requires a random-effects term, "
                "for example '(post_treatment:treated | store_id)'."
            )
        self.group_variable_name = self._resolve_group_variable()

    def _resolve_group_variable(self) -> str:
        """Infer the random-effects grouping variable."""
        inferred_group_variable = self.group_info["variable"]
        if not isinstance(inferred_group_variable, str) or not inferred_group_variable:
            raise FormulaException(
                "Formula parser did not return a random-effects grouping variable."
            )
        return inferred_group_variable

    def algorithm(self) -> None:
        """Fit the model and compute derived posterior quantities."""
        self._fit_model()
        self._extract_att()
        self._extract_group_effects()
        self._compute_counterfactual()
        self._compute_icc()

    def input_validation(self) -> None:
        """Validate panel structure, treatment timing, and model matrices."""
        self._check_required_columns()
        self._check_appropriate_data_types()
        self._check_balanced_panel()
        self._check_no_missing_outcomes()
        self._check_no_group_switching()
        self._check_treatment_group_level()
        self._check_sufficient_groups_and_observations()
        self._detect_staggered_adoption()
        self._check_clear_pre_post_periods()
        self._check_valid_random_effects_syntax()
        self._check_no_perfect_multicollinearity()

    def _check_required_columns(self) -> None:
        required_columns = {
            self.outcome_variable_name,
            self.time_variable_name,
            self.unit_variable_name,
            self.group_variable_name,
            self.treated_variable_name,
            self.post_treatment_variable_name,
        }
        missing = sorted(required_columns.difference(self.data.columns))
        if missing:
            raise DataException(f"Data is missing required HDiD columns: {missing}.")

    def _check_balanced_panel(self) -> None:
        panel_keys = [self.unit_variable_name, self.time_variable_name]
        if self.data.duplicated(panel_keys).any():
            raise DataException(
                "Hierarchical DiD requires exactly one observation per unit and "
                "time period."
            )

        panel_counts = self.data.groupby(self.unit_variable_name, observed=True)[
            self.time_variable_name
        ].nunique()
        n_periods = self.data[self.time_variable_name].nunique()
        if panel_counts.min() != n_periods or panel_counts.max() != n_periods:
            raise DataException(
                "Hierarchical DiD requires a balanced panel: every unit must appear "
                "in every observed time period."
            )

    def _check_no_group_switching(self) -> None:
        groups_per_unit = self.data.groupby(self.unit_variable_name, observed=True)[
            self.group_variable_name
        ].nunique()
        if groups_per_unit.max() != 1:
            raise DataException(
                "Each unit must belong to exactly one random-effects group across "
                "all periods; random-effects group switching is not supported."
            )

    def _check_treatment_group_level(self) -> None:
        treated_per_group = self.data.groupby(self.group_variable_name, observed=True)[
            self.treated_variable_name
        ].nunique()
        if treated_per_group.max() != 1:
            raise DataException(
                "Treatment assignment must remain constant within each "
                "random-effects group across all periods."
            )

    def _check_no_missing_outcomes(self) -> None:
        if self.data[self.outcome_variable_name].isna().any():
            raise DataException(
                f"Outcome column {self.outcome_variable_name!r} contains missing values."
            )

    def _check_clear_pre_post_periods(self) -> None:
        post_values_per_period = self.data.groupby(
            self.time_variable_name, observed=True
        )[self.post_treatment_variable_name].nunique()
        if post_values_per_period.max() != 1:
            raise DataException(
                "The post indicator must have exactly one value within each time "
                "period."
            )

        post_by_period = (
            self.data[[self.time_variable_name, self.post_treatment_variable_name]]
            .drop_duplicates()
            .sort_values(self.time_variable_name)[self.post_treatment_variable_name]
            .astype(int)
        )
        if set(post_by_period) != {0, 1}:
            raise DataException(
                "Both pre- and post-treatment periods are required. The post indicator "
                "must contain both 0 and 1 values."
            )
        if (post_by_period.diff().dropna() < 0).any():
            raise DataException(
                "The post indicator must define a single transition from "
                "pre-treatment to post-treatment periods."
            )

    def _detect_staggered_adoption(self) -> None:
        group_time = (
            self.data.assign(
                _adopted=lambda frame: (
                    frame[self.treated_variable_name].astype(int)
                    * frame[self.post_treatment_variable_name].astype(int)
                )
            )
            .groupby(
                [self.group_variable_name, self.time_variable_name], observed=True
            )["_adopted"]
            .max()
            .reset_index()
        )
        first_adoption_period = (
            group_time.loc[group_time["_adopted"] == 1]
            .groupby(self.group_variable_name, observed=True)[self.time_variable_name]
            .min()
        )
        if first_adoption_period.nunique() > 1:
            raise DataException(
                "Staggered adoption is not supported in HierarchicalDifferenceInDifferences; "
                "all treated groups must begin treatment in the same period."
            )

    def _check_valid_random_effects_syntax(self) -> None:
        self._validate_random_effects()

    def _validate_random_effects(self) -> None:
        """Validate random-effects metadata produced by the formula parser."""
        if len(self._formula_info.get("group", {}).get("components", [])) != 1:
            raise FormulaException(
                "HierarchicalDifferenceInDifferences supports exactly one random-effects component."
            )
        if not self.random_effect_labels:
            raise FormulaException("Formula parser returned no random-effect columns.")
        if self.group_variable_name not in self.data.columns:
            raise FormulaException(
                f"Random-effects grouping variable {self.group_variable_name!r} is not present in data."
            )
        if (
            len(self._formula_info["group"]["labels"])
            != self._formula_info["group"]["n_groups"]
        ):
            raise FormulaException("Random-effects grouping metadata is inconsistent.")

    def _check_appropriate_data_types(self) -> None:
        if not pd.api.types.is_numeric_dtype(self.data[self.outcome_variable_name]):
            raise DataException(
                f"Outcome column {self.outcome_variable_name!r} must be numeric."
            )
        if self.data[self.time_variable_name].isna().any():
            raise DataException(
                f"Time column {self.time_variable_name!r} contains missing values."
            )
        for binary_column in [
            self.treated_variable_name,
            self.post_treatment_variable_name,
        ]:
            values = self.data[binary_column]
            if values.isna().any() or not values.isin([0, 1]).all():
                raise DataException(
                    f"Column {binary_column!r} must be binary-coded as 0/1 or False/True."
                )

    def _check_sufficient_groups_and_observations(self) -> None:
        n_groups = self.data[self.group_variable_name].nunique()
        if n_groups < 3:
            raise DataException(
                "Hierarchical DiD requires at least 3 random-effects groups."
            )
        if n_groups < 10:
            warnings.warn(
                "Hierarchical DiD has fewer than 10 random-effects groups; random-effect variance estimates may be unstable.",
                UserWarning,
                stacklevel=2,
            )
        observations_per_group = self.data.groupby(
            self.group_variable_name, observed=True
        ).size()
        if observations_per_group.min() < 2:
            raise DataException(
                "Each random-effects group must have at least 2 observations."
            )
        if observations_per_group.min() < 5:
            warnings.warn(
                "At least one random-effects group has fewer than 5 observations; group-level effects may be weakly identified.",
                UserWarning,
                stacklevel=2,
            )

    def _check_no_perfect_multicollinearity(self) -> None:
        matrix_rank = np.linalg.matrix_rank(self.matrices.rhs.to_numpy())
        if matrix_rank < self.matrices.rhs.shape[1]:
            raise FormulaException(
                "The fixed-effects design matrix is rank deficient; check for perfect "
                "multicollinearity in the formula."
            )

    def _create_group_indices(self) -> None:
        """Create integer indices and stable labels for random-effects groups."""
        self.group_idx = np.asarray(self.group_info["idx"], dtype=np.int32)
        self.group_labels = list(self.group_info["labels"])
        self.n_groups = int(self.group_info["n_groups"])
        if self.group_idx.ndim != 1 or self.group_idx.shape[0] != self.data.shape[0]:
            raise DataException("Formula parser returned invalid group indices.")
        if len(self.group_labels) != self.n_groups:
            raise DataException("Formula parser returned inconsistent group labels.")

    def _prepare_data(self) -> None:
        """Convert parser outputs into model-ready xarray objects."""
        self._create_group_indices()
        obs_ind = np.arange(self.matrices.rhs.shape[0])
        self.coords = {
            "obs_ind": obs_ind,
            "coeffs": self.labels,
            "random_coeffs": self.random_effect_labels,
            "treated_units": ["unit_0"],
            "groups": self.group_labels,
        }
        self.X = xr.DataArray(
            self.matrices.rhs.to_numpy(),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": obs_ind, "coeffs": self.labels},
        )
        self.Z = xr.DataArray(
            self.matrices.Z.to_numpy(),
            dims=["obs_ind", "random_coeffs"],
            coords={"obs_ind": obs_ind, "random_coeffs": self.random_effect_labels},
        )
        self.y = xr.DataArray(
            self.matrices.lhs.to_numpy(),
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": obs_ind, "treated_units": ["unit_0"]},
        )
        self.did_term = self._identify_did_term()

    def _identify_did_term(self) -> str:
        """Find the unique fixed-effect coefficient representing post x treated."""
        candidate_pairs = {
            f"{self.post_treatment_variable_name}:{self.treated_variable_name}",
            f"{self.treated_variable_name}:{self.post_treatment_variable_name}",
        }
        matches = [label for label in self.labels if label in candidate_pairs]
        if len(matches) != 1:
            raise FormulaException(
                "Formula must contain exactly one DiD interaction term matching "
                f"{sorted(candidate_pairs)}. Found: {matches}."
            )
        return matches[0]

    def _fit_model(self) -> None:
        """Fit the hierarchical regression backend."""
        self.model.fit(  # type: ignore[call-arg]
            X=self.X,
            Z=self.Z,
            y=self.y,
            group_idx=self.group_idx,
            coords=self.coords,
            non_centered=self.non_centered,
        )

    def _extract_att(self) -> None:
        """Extract posterior samples for the population-average ATT coefficient."""
        did_idx = self.labels.index(self.did_term)
        posterior = self._model_backend.require_idata().posterior
        self.att = posterior["beta_fixed"].sel(coeffs=self.did_term)
        self.causal_impact = self.att
        self._did_term_index = did_idx

    def _extract_group_effects(self) -> None:
        """Extract posterior samples for group-level random-effect deviations."""
        posterior = self._model_backend.require_idata().posterior
        self.group_effects = posterior["beta_random"]

    def _compute_icc(self) -> None:
        """Compute the random-intercept intraclass correlation when available."""
        posterior = self._model_backend.require_idata().posterior
        if "sigma_random" not in posterior or "sigma_fixed" not in posterior:
            self.icc = None
            return
        random_intercept_label = next(
            (label for label in self.random_effect_labels if label.startswith("1|")),
            self.random_effect_labels[0],
        )
        sigma_group = posterior["sigma_random"].sel(
            random_coeffs=random_intercept_label
        )
        sigma_fixed = posterior["sigma_fixed"]
        self.icc = sigma_group**2 / (sigma_group**2 + sigma_fixed**2)

    @staticmethod
    def _hdi_dataarray(
        samples: xr.DataArray, hdi_prob: float = HDI_PROB
    ) -> xr.DataArray:
        """Return ArviZ HDI output as a data array."""
        hdi = az.hdi(samples, hdi_prob=hdi_prob)
        if isinstance(hdi, xr.Dataset):
            return next(iter(hdi.data_vars.values()))
        return hdi

    @classmethod
    def _scalar_hdi_bounds(
        cls, samples: xr.DataArray, hdi_prob: float = HDI_PROB
    ) -> tuple[float, float]:
        """Return scalar lower and upper HDI bounds."""
        hdi_data = cls._hdi_dataarray(samples, hdi_prob=hdi_prob)
        lower = float(np.asarray(hdi_data.sel(hdi="lower")).reshape(()))
        upper = float(np.asarray(hdi_data.sel(hdi="higher")).reshape(()))
        return lower, upper

    def _compute_counterfactual(self) -> None:
        """Compute counterfactual predictions for the treated outcome.

        The counterfactual removes the fixed DiD interaction and any matching
        group-level random DiD slope from the posterior fitted outcomes. The
        resulting posterior samples are stored on ``y_pred_counterfactual``.
        """
        posterior = self._model_backend.require_idata().posterior
        required = {"mu", "beta_fixed"}
        if required.difference(posterior.data_vars):
            self.y_pred_counterfactual = None
            return

        mu = posterior["mu"]
        fixed_att = posterior["beta_fixed"].sel(coeffs=self.did_term)
        did_design = self.X.sel(coeffs=self.did_term)
        did_contribution = (fixed_att * did_design).transpose(*mu.dims)

        random_did_labels = self._did_random_effect_labels()
        if random_did_labels and "beta_random" in posterior:
            obs_group_idx = xr.DataArray(
                self.group_idx,
                dims=["obs_ind"],
                coords={"obs_ind": self.X.coords["obs_ind"]},
            )
            beta_random = posterior["beta_random"].sel(random_coeffs=random_did_labels)
            beta_random_by_obs = beta_random.isel(groups=obs_group_idx)
            Z_did = self.Z.sel(random_coeffs=random_did_labels)
            random_did_contribution = (beta_random_by_obs * Z_did).sum(
                dim="random_coeffs"
            )
            did_contribution = did_contribution + random_did_contribution.transpose(
                *mu.dims
            )

        self.y_pred_counterfactual = mu - did_contribution

    def _did_random_effect_labels(self) -> list[str]:
        """Return random-effect labels corresponding to the DiD interaction."""
        candidate_pairs = {
            f"{self.post_treatment_variable_name}:{self.treated_variable_name}",
            f"{self.treated_variable_name}:{self.post_treatment_variable_name}",
        }
        return [
            label
            for label in self.random_effect_labels
            if label.split("|", maxsplit=1)[0] in candidate_pairs
        ]

    def _append_posterior_columns(
        self,
        plot_data: pd.DataFrame,
        *,
        prefix: str,
        samples: xr.DataArray,
        hdi_prob: float,
    ) -> None:
        """Add posterior mean and HDI columns to a plot-data frame."""
        values = samples.squeeze("treated_units", drop=True)
        hdi_data = self._hdi_dataarray(values, hdi_prob=hdi_prob)
        plot_data[prefix] = values.mean(dim=("chain", "draw")).values.reshape(-1)
        plot_data[f"{prefix}_lower"] = hdi_data.sel(hdi="lower").values.reshape(-1)
        plot_data[f"{prefix}_upper"] = hdi_data.sel(hdi="higher").values.reshape(-1)

    def _posterior_trend_summary(
        self,
        samples: xr.DataArray,
        *,
        treated_value: int,
        hdi_prob: float,
    ) -> pd.DataFrame:
        """Summarise posterior outcomes by treatment status and time period."""
        data = self.data.reset_index(drop=True)
        values = samples.squeeze("treated_units", drop=True)
        subset = data[data[self.treated_variable_name].astype(int) == treated_value]
        rows = []
        for time_value, indices in subset.groupby(
            self.time_variable_name, observed=True
        ).groups.items():
            period_samples = values.isel(obs_ind=list(indices)).mean(dim="obs_ind")
            lower, upper = self._scalar_hdi_bounds(period_samples, hdi_prob=hdi_prob)
            rows.append(
                {
                    self.time_variable_name: time_value,
                    "mean": float(period_samples.mean()),
                    "lower": lower,
                    "upper": upper,
                }
            )
        return pd.DataFrame(rows).sort_values(self.time_variable_name)

    def _variance_components_posterior(self) -> xr.Dataset | None:
        """Return posterior draws for variance components and ICC."""
        posterior = self._model_backend.require_idata().posterior
        required = {"sigma_random", "sigma_fixed"}
        if required.difference(posterior.data_vars):
            return None

        components = xr.Dataset(
            {
                "random_effect_variance": posterior["sigma_random"] ** 2,
                "residual_variance": posterior["sigma_fixed"] ** 2,
            }
        )
        if self.icc is not None:
            components["icc"] = self.icc
        return components

    def plot_group_effects(
        self,
        *,
        random_coeff: str | None = None,
        hdi_prob: float = HDI_PROB,
        combined: bool = True,
        show: bool = True,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot posterior random-effects group deviations.

        Parameters
        ----------
        random_coeff : str, optional
            Random-effect coefficient to plot. If omitted, all random-effect
            coefficients are shown.
        hdi_prob : float
            Probability mass of the highest density interval. Must be in
            ``(0, 1)``. Defaults to :data:`~causalpy.constants.HDI_PROB`
            (currently 0.94).
        combined : bool
            Whether to combine chains in the ArviZ forest plot. Defaults to
            ``True``.
        show : bool
            Whether to automatically display the plot. Defaults to ``True``.
        **kwargs
            Additional keyword arguments forwarded to :func:`arviz.plot_forest`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure that was created.
        ax : matplotlib.axes.Axes
            Axes containing the forest plot.

        Notes
        -----
        Treatment-response coefficients are directly informed by observations
        only for treated groups. For control groups, the treatment-response
        deviation does not enter the likelihood because the DiD interaction is
        always zero. Their displayed deviations are therefore prior-centered,
        model-implied values rather than separately identified treatment-response
        deviations.
        """
        posterior = self._model_backend.require_idata().posterior
        if "beta_random" not in posterior:
            raise ValueError(
                "Posterior does not contain beta_random random-effects group deviations."
            )
        if not 0 < hdi_prob < 1:
            raise ValueError("hdi_prob must be in (0, 1).")

        group_effects = self.group_effects
        if random_coeff is not None:
            available_coeffs = group_effects.coords["random_coeffs"].values.tolist()
            if random_coeff not in available_coeffs:
                raise ValueError(
                    f"Unknown random-effect coefficient {random_coeff!r}. "
                    f"Available coefficients are {available_coeffs}."
                )
            group_effects = group_effects.sel(random_coeffs=random_coeff, drop=True)

        if group_effects.sizes.get("treated_units") == 1:
            group_effects = group_effects.squeeze("treated_units", drop=True)

        plot_data = group_effects.rename("group_deviation").to_dataset()
        plot_dims = set(group_effects.dims).difference({"chain", "draw"})
        n_rows = int(np.prod([group_effects.sizes[dim] for dim in plot_dims]))
        kwargs.setdefault("labeller", az.labels.NoVarLabeller())
        kwargs.setdefault("figsize", (8, max(4, 0.4 * n_rows + 1)))

        axes = az.plot_forest(
            plot_data,
            var_names=["group_deviation"],
            combined=combined,
            hdi_prob=hdi_prob,
            **kwargs,
        )
        ax = axes.ravel()[0] if hasattr(axes, "ravel") else axes[0]
        fig = ax.figure
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Deviation from population coefficient")
        ax.set_title(f"Group-Level Deviations ({hdi_prob:.0%} HDI)")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_variance_components(
        self,
        *,
        hdi_prob: float = HDI_PROB,
        show: bool = True,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot posterior distributions for hierarchical variance components.

        Parameters
        ----------
        hdi_prob : float
            Probability mass of the highest density interval. Must be in
            ``(0, 1)``. Defaults to :data:`~causalpy.constants.HDI_PROB`
            (currently 0.94).
        show : bool
            Whether to automatically display the plot. Defaults to ``True``.
        **kwargs
            Additional keyword arguments forwarded to
            :func:`arviz.plot_posterior`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure that was created.
        ax : matplotlib.axes.Axes
            First axes returned by ArviZ.
        """
        if not 0 < hdi_prob < 1:
            raise ValueError("hdi_prob must be in (0, 1).")

        components = self._variance_components_posterior()
        if components is None:
            raise ValueError(
                "Posterior is missing variance component variables: "
                "['sigma_fixed', 'sigma_random']."
            )
        idata = az.InferenceData(posterior=components)

        axes = az.plot_posterior(
            idata,
            var_names=list(components.data_vars),
            hdi_prob=hdi_prob,
            **kwargs,
        )
        ax = axes.ravel()[0] if hasattr(axes, "ravel") else axes
        fig = ax.figure
        fig.suptitle("Hierarchical DiD Variance Components")
        if show:
            plt.show()
        return fig, ax

    def summary(self, round_to: int | None = 2) -> None:
        """Print a summary of the hierarchical DiD results.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2. Use
            ``None`` to return raw numbers.
        """
        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print(f"Outcome: {self.outcome_variable_name}")
        print(f"Random-effects groups: {self.n_groups} ({self.group_variable_name})")
        print(
            f"Units: {self.data[self.unit_variable_name].nunique()} ({self.unit_variable_name})"
        )
        print(
            f"Time periods: {self.data[self.time_variable_name].nunique()} ({self.time_variable_name})"
        )
        print(f"Observations: {self.data.shape[0]}")
        print(f"DiD term: {self.did_term}")
        print(f"Fixed effects: {len(self.labels)}")
        print(f"Random effects: {len(self.random_effect_labels)}")
        print("\nResults:")
        print(self._causal_impact_summary_stat(round_to))
        variance_components = self._variance_components_summary(round_to)
        if variance_components is not None:
            print("\nVariance components:")
            print(variance_components.to_string())

        fixed_effects = self._fixed_effects_summary(round_to)
        if fixed_effects is not None:
            print("\nFixed effects:")
            print(fixed_effects.to_string())

        random_att = self._random_att_summary(round_to)
        if random_att is not None:
            print("\nGroup-level ATT deviations:")
            print(random_att.to_string())

    def _posterior_summary(
        self,
        *,
        var_names: list[str],
        round_to: int | None = 2,
        coords: dict[str, list[str]] | None = None,
    ) -> pd.DataFrame | None:
        """Return an ArviZ summary table for selected posterior variables."""
        idata = self._model_backend.require_idata()
        missing_vars = set(var_names).difference(idata.posterior.data_vars)
        if missing_vars:
            return None

        summary = az.summary(idata, var_names=var_names, coords=coords)
        columns = [
            column
            for column in ["mean", "sd", "hdi_3%", "hdi_97%"]
            if column in summary.columns
        ]
        table = summary[columns].copy()
        return table.round(round_to) if round_to is not None else table

    def _fixed_effects_summary(self, round_to: int | None = 2) -> pd.DataFrame | None:
        """Return posterior summaries for population-level coefficients."""
        return self._posterior_summary(var_names=["beta_fixed"], round_to=round_to)

    def _random_att_summary(self, round_to: int | None = 2) -> pd.DataFrame | None:
        """Return posterior summaries for group-level ATT deviations."""
        random_att_labels = self._did_random_effect_labels()
        if not random_att_labels:
            return None
        return self._posterior_summary(
            var_names=["beta_random"],
            coords={"random_coeffs": random_att_labels},
            round_to=round_to,
        )

    def _variance_components_summary(
        self, round_to: int | None = 2
    ) -> pd.DataFrame | None:
        """Return summaries for random-effect variance, residual variance, and ICC."""
        components = self._variance_components_posterior()
        if components is None:
            return None

        summary = az.summary(
            az.InferenceData(posterior=components),
            var_names=list(components.data_vars),
        )
        columns = [
            column
            for column in ["mean", "sd", "hdi_3%", "hdi_97%"]
            if column in summary.columns
        ]
        table = summary[columns].copy()

        return table.round(round_to) if round_to is not None else table

    def _causal_impact_summary_stat(self, round_to: int | None = None) -> str:
        """Return a compact ATT summary string."""
        att_mean = float(self.causal_impact.mean())
        lower, upper = self._scalar_hdi_bounds(self.causal_impact, hdi_prob=HDI_PROB)
        if round_to is not None:
            att_mean = round(att_mean, round_to)
            lower = round(lower, round_to)
            upper = round(upper, round_to)
        return f"Causal impact = {att_mean} ({HDI_PROB:.0%} HDI [{lower}, {upper}])"

    def plot(
        self,
        *,
        round_to: int | None = 2,
        hdi_prob: float = HDI_PROB,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot observed trends, fitted trends, and the HDiD counterfactual.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round numerical results in the figure
            title. Defaults to 2. Use ``None`` to return raw numbers.
        hdi_prob : float
            Probability mass of the highest density interval drawn around
            posterior fitted and counterfactual trajectories. Must be in
            ``(0, 1)``. Defaults to :data:`~causalpy.constants.HDI_PROB`
            (currently 0.94).
        figsize : tuple of (float, float), optional
            Width and height of the figure in inches, passed to
            :func:`matplotlib.pyplot.subplots`. Defaults to ``None``.
        show : bool
            Whether to automatically display the plot. Defaults to ``True``.
        legend_kwargs : dict, optional
            Keyword arguments to adjust legend placement and styling.
            Supported keys are defined by :meth:`BaseExperiment._render_plot`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure that was created.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """
        return self._render_plot(
            show=show,
            legend_kwargs=legend_kwargs,
            round_to=round_to,
            hdi_prob=hdi_prob,
            figsize=figsize,
        )

    def _plot(
        self,
        round_to: int | None = 2,
        hdi_prob: float = HDI_PROB,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot observed, fitted, and counterfactual outcome trends."""
        if not 0 < hdi_prob < 1:
            raise ValueError("hdi_prob must be in (0, 1).")
        posterior = self._model_backend.require_idata().posterior

        trend = self.data.groupby(
            [self.time_variable_name, self.treated_variable_name], observed=True
        )[self.outcome_variable_name].mean()

        fig, ax = plt.subplots(figsize=figsize)
        for treated_value, values in trend.groupby(
            level=self.treated_variable_name, sort=False
        ):
            label = "Treated" if int(treated_value) == 1 else "Control"
            time_values = values.index.get_level_values(self.time_variable_name)
            ax.plot(
                time_values,
                values.to_numpy(),
                marker="o",
                linestyle="",
                label=f"Observed {label.lower()}",
            )

        if "mu" in posterior:
            fitted_specs = [
                ("Control fit", 0, "C0", "-"),
                ("Treated fit", 1, "C1", "-"),
            ]
            for label, treated_value, color, linestyle in fitted_specs:
                summary = self._posterior_trend_summary(
                    posterior["mu"],
                    treated_value=treated_value,
                    hdi_prob=hdi_prob,
                )
                ax.plot(
                    summary[self.time_variable_name],
                    summary["mean"],
                    color=color,
                    linestyle=linestyle,
                    label=label,
                )
                ax.fill_between(
                    summary[self.time_variable_name],
                    summary["lower"],
                    summary["upper"],
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                )

            counterfactual = self.y_pred_counterfactual
            if counterfactual is not None:
                counterfactual_summary = self._posterior_trend_summary(
                    counterfactual,
                    treated_value=1,
                    hdi_prob=hdi_prob,
                )
                ax.plot(
                    counterfactual_summary[self.time_variable_name],
                    counterfactual_summary["mean"],
                    color="C2",
                    linestyle="--",
                    label="Counterfactual",
                )
                ax.fill_between(
                    counterfactual_summary[self.time_variable_name],
                    counterfactual_summary["lower"],
                    counterfactual_summary["upper"],
                    color="C2",
                    alpha=0.2,
                    linewidth=0,
                )
                self._plot_causal_impact_arrow(
                    ax,
                    treated_summary=self._posterior_trend_summary(
                        posterior["mu"],
                        treated_value=1,
                        hdi_prob=hdi_prob,
                    ),
                    counterfactual_summary=counterfactual_summary,
                )

        post_times = self.data.loc[
            self.data[self.post_treatment_variable_name].astype(bool),
            self.time_variable_name,
        ]
        treatment_start = post_times.min()
        ax.axvline(
            treatment_start,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Treatment start",
        )

        ax.set_title(self._causal_impact_summary_stat(round_to))
        ax.set_xlabel(self.time_variable_name)
        ax.set_ylabel(self.outcome_variable_name)
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        return fig, ax

    def _plot_causal_impact_arrow(
        self,
        ax: plt.Axes,
        *,
        treated_summary: pd.DataFrame,
        counterfactual_summary: pd.DataFrame,
    ) -> None:
        """Annotate the fitted treated/counterfactual gap in the post period."""
        post_times = self.data.loc[
            self.data[self.post_treatment_variable_name].astype(bool),
            self.time_variable_name,
        ]
        time_value = post_times.max()
        treated_row = treated_summary[
            treated_summary[self.time_variable_name] == time_value
        ]
        counterfactual_row = counterfactual_summary[
            counterfactual_summary[self.time_variable_name] == time_value
        ]
        treated_mean = float(treated_row["mean"].iloc[0])
        counterfactual_mean = float(counterfactual_row["mean"].iloc[0])
        ax.annotate(
            "",
            xy=(time_value, counterfactual_mean),
            xytext=(time_value, treated_mean),
            arrowprops={"arrowstyle": "<-", "color": "green", "lw": 2},
        )
        ax.annotate(
            "causal\nimpact",
            xy=(time_value, np.mean([counterfactual_mean, treated_mean])),
            xytext=(5, 0),
            textcoords="offset points",
            color="green",
            va="center",
        )

    def get_plot_data(self, hdi_prob: float = HDI_PROB, **kwargs: Any) -> pd.DataFrame:
        """Return observed data plus fitted posterior summaries when available.

        Parameters
        ----------
        hdi_prob : float
            Probability mass of the highest density interval. Must be in
            ``(0, 1)``. Defaults to :data:`~causalpy.constants.HDI_PROB`
            (currently 0.94).
        **kwargs
            Reserved for forward-compatibility; not consumed by this
            implementation.

        Returns
        -------
        pd.DataFrame
            Copy of the experiment data with posterior fitted outcome summaries.
            If available, counterfactual summaries are included as ``y_counterfactual``,
            ``y_counterfactual_lower``, and ``y_counterfactual_upper``.
        """
        if not 0 < hdi_prob < 1:
            raise ValueError("hdi_prob must be in (0, 1).")

        plot_data = self.data.copy().reset_index(drop=True)
        idata = self.idata
        if idata is not None and "mu" in idata.posterior:
            mu = idata.posterior["mu"]
            self._append_posterior_columns(
                plot_data,
                prefix="y_fitted",
                samples=mu,
                hdi_prob=hdi_prob,
            )
            counterfactual = self.y_pred_counterfactual
            if counterfactual is not None:
                self._append_posterior_columns(
                    plot_data,
                    prefix="y_counterfactual",
                    samples=counterfactual,
                    hdi_prob=hdi_prob,
                )
        return plot_data

    def effect_summary(
        self,
        *,
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        min_effect: float | None = None,
        **kwargs: Any,
    ) -> EffectSummary:
        """Generate a decision-ready summary of the HDiD ATT.

        HDiD reports the population-average ATT using the same scalar posterior
        summary machinery as DifferenceInDifferences.

        Parameters
        ----------
        direction : {"increase", "decrease", "two-sided"}, default="increase"
            Direction for tail probability calculation.
        alpha : float, default=0.05
            Significance level for HDI intervals. The reported HDI probability
            is ``1 - alpha``.
        min_effect : float, optional
            Region of Practical Equivalence (ROPE) threshold.
        **kwargs
            Reserved for forward-compatibility; not consumed by this
            implementation.

        Returns
        -------
        EffectSummary
            Object with ``.table`` and ``.text`` attributes.
        """
        from causalpy.reporting import _effect_summary_did

        return _effect_summary_did(
            self,
            direction=direction,
            alpha=alpha,
            min_effect=min_effect,
        )
