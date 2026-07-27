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
"""Staggered Difference in Differences (Imputation-based).

This module implements the imputation-based staggered DiD estimator, following
the approach of Borusyak, Jaravel, and Spiess (2024). It handles settings where
different units receive treatment at different times.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from patsy import PatsyError
from sklearn.base import RegressorMixin

from causalpy.constants import HDI_PROB, LEGEND_FONT_SIZE
from causalpy.custom_exceptions import DataException, FormulaException
from causalpy.experiments.model_adapter import build_coords
from causalpy.formula_utils import build_formula_matrices
from causalpy.plot_utils import has_posterior_draws
from causalpy.pymc_models import ETWFERegression, LinearRegression, PyMCModel
from causalpy.reporting import EffectSummary

from .base import BaseExperiment


class StaggeredDifferenceInDifferences(BaseExperiment):
    """A class to analyse data from staggered adoption Difference-in-Differences settings.

    This class implements the Borusyak, Jaravel, and Spiess (BJS, 2024)
    imputation estimator for staggered adoption settings. It fits a model on
    untreated observations only (pre-treatment periods for eventually-treated
    units plus all periods for never-treated units), then predicts
    counterfactual outcomes for all observations. Treatment effects are computed
    as the difference between observed and predicted outcomes for treated
    observations.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas dataframe with panel data (unit x time observations).
    formula : str
        A statistical model formula. Recommended: "y ~ 1 + C(unit) + C(time)"
        for unit and time fixed effects.
    unit_variable_name : str
        Name of the column identifying units.
    time_variable_name : str
        Name of the column identifying time periods.
    treated_variable_name : str, optional
        Name of the column indicating treatment status (0/1). Defaults to "treated".
    treatment_time_variable_name : str, optional
        Name of the column containing unit-level treatment time (G_i).
        If None, treatment time is inferred from the treated_variable_name column.
    never_treated_value : Any, optional
        Value indicating never-treated units in treatment_time column.
        Defaults to np.inf.
    model : PyMCModel or RegressorMixin, optional
        A model for the untreated outcome. Defaults to LinearRegression.
    event_window : tuple[int, int], optional
        Tuple (min_event_time, max_event_time) to restrict event-time aggregation.
        If None, uses all available event-times.
    reference_event_time : int, optional
        Event-time whose effect is normalised to zero. Used by the ETWFE
        estimator, where the corresponding column is *omitted* from the effect
        surface. Must satisfy ``-n_leads <= reference_event_time <= -1``.
        Defaults to -1. Unused (reserved) by the imputation estimator.
    estimator : {"imputation", "etwfe"}, optional
        Which estimator to run. ``"imputation"`` (default) is the
        Borusyak-Jaravel-Spiess fit-on-untreated-then-impute approach.
        ``"etwfe"`` is Wooldridge's extended two-way fixed effects (Mundlak)
        estimator: a saturated regression on the **full sample** with one
        treatment effect per (cohort, event-time) cell.
    conditioning : {"mundlak", "dummy"}, optional
        How the two-way effects are conditioned in the ETWFE estimator. ``None``
        (default) resolves to ``"mundlak"`` for PyMC models and ``"dummy"`` for
        scikit-learn models. ``"mundlak"`` is rejected for scikit-learn models
        (see Notes). Only valid when ``estimator="etwfe"``.
    n_leads : int, optional
        Number of pre-treatment lead terms the ETWFE estimator should estimate.
        Defaults to 0 (post-treatment cells only). Only valid when
        ``estimator="etwfe"``.
    max_event_time : int, optional
        Largest event-time given its own column in the ETWFE effect surface.
        Treated observations beyond it are top-binned into that column. If None
        (default), every observed treated event-time gets its own column. Only
        valid when ``estimator="etwfe"``.
    covariates : list[str], optional
        Names of additional covariate columns to include additively in the ETWFE
        design. The formula's right-hand side is **ignored** by the ETWFE
        estimator, so covariates must be supplied here. Only valid when
        ``estimator="etwfe"``.
    se_type : {"cluster", "classical"}, optional
        Standard error type for the OLS ETWFE path. ``"cluster"`` (default) is a
        cluster-by-unit sandwich estimator. Only valid when
        ``estimator="etwfe"``.
    **kwargs
        Additional keyword arguments forwarded to :class:`BaseExperiment`.

    Attributes
    ----------
    data_ : pd.DataFrame
        Augmented data with G (treatment time), event_time, y_hat0 (counterfactual),
        and tau_hat (treatment effect) columns.
    att_group_time_ : pd.DataFrame
        Group-time ATT estimates: ATT(g, t) for each cohort g and calendar time t.
        Includes an ``identified`` column; non-identified cells have ``NaN`` estimates.
    att_event_time_ : pd.DataFrame
        Event-time ATT estimates: ATT(e) for each event-time e = t - G.
        Includes an ``identified`` column; non-identified cells have ``NaN`` estimates.
    non_identified_periods_ : set
        Calendar periods with no untreated observations.
    non_identified_cohorts_ : set
        Treatment cohorts with at least one non-identified post-treatment ATT(g, t).
    att_ : xarray.DataArray or float
        ETWFE only. The aggregated average-over-the-treated ATT. On the PyMC path
        this is the posterior of the in-model ``att`` deterministic; on the OLS
        path it is the point estimate ``w'b``.
    att_se_ : float or None
        ETWFE only. Standard error of ``att_`` on the OLS path; ``None`` on the
        PyMC path, where ``att_`` carries its own posterior.
    tau_surface_ : pd.DataFrame
        ETWFE only. Long-form ``(cohort, event_time, att, ...)`` table covering
        every estimated cell, including lead cells.
    att_weights_ : pd.DataFrame
        ETWFE only. The average-over-the-treated weight matrix
        ``w_gk = N_gk / sum(N_gk)``, cohorts x event-times. Lead columns are zero.
    event_time_grid_ : np.ndarray
        ETWFE only. The event-times actually estimated, reference omitted.
    etwfe_formula_ : str
        ETWFE only, OLS path. The generated saturated patsy formula.
    estimator, conditioning, n_leads, se_type
        The resolved configuration, echoed back. ``conditioning`` is ``None``
        for the imputation estimator and the resolved ``"mundlak"`` /
        ``"dummy"`` value for ETWFE.

    Notes
    -----
    **Estimate extraction**

    The Borusyak-Jaravel-Spiess imputation estimator fits the untreated outcome model using only observations that are not yet treated or never treated. It predicts each treated observation's untreated potential outcome, subtracts that prediction from the observed outcome, and averages the resulting one-sided contrasts into group-time and event-time ATTs. Bayesian aggregation retains posterior uncertainty in ``mu``; OLS aggregation uses point predictions and standard-error approximations.

    Like Interrupted Time Series, this fit-predict-subtract procedure is a reduced-form estimator. The corresponding structural contrast is a saturated regression as in Wooldridge's extended two-way fixed effects (ETWFE) framework, which CausalPy does not currently implement.

    This estimator requires the following identifying assumptions:

    1. **Absorbing treatment**: Once a unit receives treatment, it must remain
       treated in all subsequent periods. Treatment cannot be reversed or
       temporarily suspended. This is validated at runtime.
    2. **Parallel trends**: In the absence of treatment, treated and control
       units would have followed parallel outcome trajectories.
    3. **No anticipation**: Units do not change their behavior in anticipation
       of future treatment.
    4. **Untreated support at each calendar period**: The time fixed effect
       :math:`\\gamma_t` for calendar period :math:`t` is identified only if at
       least one unit is untreated in that period. Without never-treated units,
       post-treatment effects for the last-treated cohort (and any calendar
       periods where every unit is already treated) are not identified. CausalPy
       warns when this condition fails and marks the affected ``ATT(g, t)`` and
       ``ATT(e)`` cells as non-identified in the output tables.

    **Panel Balance**: This implementation supports both balanced and unbalanced panel
    data. While balanced panels (where each unit is observed in every time period) are
    common in staggered DiD applications, the imputation-based approach of Borusyak et
    al. (2024) can accommodate unbalanced panels. The key requirement is that treatment
    timing is well-defined for each unit, not that all units are observed in all periods.
    Unit and observation counts in the summary output are computed without assuming
    balanced panels.

    **ETWFE and the formula argument**: the ``estimator="etwfe"`` path builds its
    own saturated design and uses only the **left-hand side** of ``formula``. A
    ``UserWarning`` names any right-hand-side term beyond ``1``, ``0``,
    ``C(unit)`` and ``C(time)``. This keeps the canonical
    ``"y ~ 1 + C(unit) + C(time)"`` call working when a user simply flips
    ``estimator=``.

    **Mundlak conditioning requires PyMC**: with free unit dummies the Mundlak
    unit mean is exactly collinear with them, so a pseudo-inverse would silently
    drop it and "Mundlak OLS" would be numerically identical to the dummy fit.
    Genuine Mundlak conditioning needs partial pooling, i.e. the PyMC path.

    **ETWFE covariates enter additively**. Wooldridge's centred-covariate by
    ``(g, k)`` interactions are not implemented; this is future work.

    **The Mundlak time coefficient ``g_t`` must not be interpreted.** Under
    ``conditioning="mundlak"`` the Mundlak time mean ``dbar_time`` is a
    deterministic function of the calendar period ``t`` alone, so it lies exactly
    in the span of the time effects ``beta_t``. ``g_t`` is therefore identified
    only by its prior: its posterior carries no information from the data, and
    reading it as "the effect of average exposure in a period" is a mistake. This
    is a property of the Mundlak device, not a defect of the implementation.
    **The ATT is unaffected.** ``tau`` -- and hence ``att_`` -- is identified off
    within-cell variation, which is orthogonal to any function of ``t`` alone, so
    the collinearity between ``dbar_time`` and ``beta_t`` moves posterior mass
    between two nuisance parameters without touching the estimand. ``g_u`` is
    better behaved, because the unit intercepts are only partially pooled and
    shrinkage identifies it, but it too is a nuisance parameter.

    References
    ----------
    Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting Event Study Designs:
    Robust and Efficient Estimation. Review of Economic Studies.

    Wooldridge, J. M. (2021). Two-Way Fixed Effects, the Two-Way Mundlak
    Regression, and Difference-in-Differences Estimators. Working paper.

    Mundlak, Y. (1978). On the Pooling of Time Series and Cross Section Data.
    Econometrica, 46(1), 69-85.

    Examples
    --------
    >>> import causalpy as cp
    >>> from causalpy.data.simulate_data import generate_staggered_did_data
    >>> df = generate_staggered_did_data(n_units=30, n_time_periods=15, seed=42)
    >>> result = cp.StaggeredDifferenceInDifferences(
    ...     df,
    ...     formula="y ~ 1 + C(unit) + C(time)",
    ...     unit_variable_name="unit",
    ...     time_variable_name="time",
    ...     treated_variable_name="treated",
    ...     treatment_time_variable_name="treatment_time",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={
    ...             "tune": 100,
    ...             "draws": 200,
    ...             "chains": 2,
    ...             "progressbar": False,
    ...         }
    ...     ),
    ... )  # doctest: +SKIP

    The same call switched onto Wooldridge's extended two-way fixed effects
    estimator. Only ``estimator``, ``conditioning`` and the model class change;
    the ATT is then available as ``att_``, a posterior of the in-model ``att``
    deterministic, and the full effect surface as ``tau_surface_``:

    >>> result = cp.StaggeredDifferenceInDifferences(
    ...     df,
    ...     formula="y ~ 1 + C(unit) + C(time)",
    ...     unit_variable_name="unit",
    ...     time_variable_name="time",
    ...     treated_variable_name="treated",
    ...     treatment_time_variable_name="treatment_time",
    ...     estimator="etwfe",
    ...     conditioning="mundlak",
    ...     n_leads=4,
    ...     model=cp.pymc_models.ETWFERegression(
    ...         sample_kwargs={
    ...             "tune": 500,
    ...             "draws": 500,
    ...             "chains": 4,
    ...             "progressbar": False,
    ...         }
    ...     ),
    ... )  # doctest: +SKIP
    >>> float(result.att_.mean())  # doctest: +SKIP
    >>> result.tau_surface_.head()  # doctest: +SKIP
    >>> fig, axes = result.plot_tau_surface()  # doctest: +SKIP
    """

    supports_ols = True
    supports_bayes = True
    # NOTE: keep this a plain (unannotated) assignment. The architecture
    # inventory check introspects it via AST and only matches ``ast.Assign``,
    # so an annotated assignment reads as "no default model".
    _default_model_class = LinearRegression

    #: ETWFE-only results. Declared here so that the two estimator paths, which
    #: populate them with different types, type-check.
    att_: "xr.DataArray | float | None" = None
    att_se_: float | None = None

    #: Model predictions. An ``arviz.InferenceData`` for PyMC models and a plain
    #: array for scikit-learn models, hence the deliberately loose annotation.
    y_pred: Any

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        unit_variable_name: str,
        time_variable_name: str,
        treated_variable_name: str = "treated",
        treatment_time_variable_name: str | None = None,
        never_treated_value: Any = np.inf,
        model: PyMCModel | RegressorMixin | None = None,
        event_window: tuple[int, int] | None = None,
        reference_event_time: int = -1,
        estimator: Literal["imputation", "etwfe"] = "imputation",
        conditioning: Literal["mundlak", "dummy"] | None = None,
        n_leads: int = 0,
        max_event_time: int | None = None,
        covariates: list[str] | None = None,
        se_type: Literal["cluster", "classical"] = "cluster",
        **kwargs: Any,
    ) -> None:
        # NOTE: kwargs is accepted for API compatibility with other experiment classes
        # and is intentionally not used inside this constructor.
        if estimator not in ("imputation", "etwfe"):
            raise ValueError(
                f"estimator must be 'imputation' or 'etwfe', got {estimator!r}"
            )
        self.estimator = estimator
        if estimator == "etwfe" and model is None:
            # Instance attribute shadows the class attribute, so the imputation
            # default is untouched for every other instance.
            self._default_model_class = ETWFERegression

        super().__init__(model=model)

        # Store parameters
        self.expt_type = "Staggered Difference in Differences"
        self.formula = formula
        self.unit_variable_name = unit_variable_name
        self.time_variable_name = time_variable_name
        self.treated_variable_name = treated_variable_name
        self.treatment_time_variable_name = treatment_time_variable_name
        self.never_treated_value = never_treated_value
        self.event_window = event_window
        self.reference_event_time = reference_event_time
        self.n_leads = n_leads
        self.max_event_time = max_event_time
        self.covariates = list(covariates) if covariates is not None else None
        self.se_type = se_type
        self._conditioning_arg = conditioning
        self.conditioning = self._resolve_conditioning(conditioning)

        # Make a copy of data to avoid modifying the original
        data = data.copy()
        data.index.name = "obs_ind"

        # Input validation
        self.data = data
        self.input_validation()

        # Step 1: Compute treatment time G_i for each unit
        self._compute_treatment_times()

        # Step 2: Compute event time for each observation
        self._compute_event_times()

        # Step 3: Identify untreated observations (training set)
        self._identify_untreated_observations()

        # Step 3b: Check calendar-period identification support. This is about
        # the panel, not the estimator, so it runs for both.
        self._check_att_identification()

        # Step 4: Build design matrices. This is the first step whose *meaning*
        # differs between estimators: imputation needs an untreated-only training
        # matrix, ETWFE needs a full-sample saturated design plus index arrays.
        if self.estimator == "etwfe":
            self._build_etwfe_design()
        else:
            self._build_design_matrices()

        self.algorithm()

    def _resolve_conditioning(
        self, conditioning: Literal["mundlak", "dummy"] | None
    ) -> Literal["mundlak", "dummy"] | None:
        """Resolve the ``conditioning`` argument against the model type.

        Called after ``super().__init__()`` so that ``self.model`` exists.

        Parameters
        ----------
        conditioning : {"mundlak", "dummy"} or None
            The user-supplied value.

        Returns
        -------
        {"mundlak", "dummy"} or None
            The resolved value, or ``None`` for the imputation estimator.

        Raises
        ------
        ValueError
            If ``conditioning`` is not a recognised value, or if
            ``"mundlak"`` is requested with a scikit-learn model.
        """
        if self.estimator != "etwfe":
            return None
        if conditioning is not None and conditioning not in ("mundlak", "dummy"):
            raise ValueError(
                f"conditioning must be 'mundlak', 'dummy' or None, got {conditioning!r}"
            )
        if self._model_backend.is_bayesian:
            return conditioning or "mundlak"
        if conditioning == "mundlak":
            raise ValueError(
                "conditioning='mundlak' is not available for scikit-learn models. "
                "The OLS ETWFE design gives every unit a free dummy, and the "
                "Mundlak unit mean D_bar_unit is constant within unit, hence "
                "exactly collinear with those dummies. The pseudo-inverse would "
                "silently drop it and the resulting 'Mundlak OLS' fit would be "
                "numerically identical to conditioning='dummy'. Genuine Mundlak "
                "conditioning needs partial pooling, which is what the PyMC path "
                "(model=cp.pymc_models.ETWFERegression()) provides."
            )
        return "dummy"

    def algorithm(self) -> None:
        """Run the experiment algorithm for the selected estimator."""
        if self.estimator == "etwfe":
            self._algorithm_etwfe()
        else:
            self._algorithm_imputation()

    def _algorithm_imputation(self) -> None:
        """Fit on untreated cells, impute counterfactuals, and aggregate effects."""
        # Step 5: Fit model on untreated observations
        self._fit_model()

        # Step 6: Predict counterfactuals for all observations
        self._predict_counterfactuals()

        # Step 7: Compute treatment effects
        self._compute_treatment_effects()

        # Step 8: Aggregate to group-time and event-time ATTs
        self._aggregate_effects()

    def _algorithm_etwfe(self) -> None:
        """Fit the saturated ETWFE design, extract effects, and populate results."""
        self._fit_model_etwfe()
        self._check_etwfe_convergence()
        self._populate_etwfe_fitted_values()
        if self._model_backend.is_bayesian:
            self._compute_etwfe_effects_bayesian()
        else:
            self._compute_etwfe_effects_ols()

    def input_validation(self) -> None:
        """Validate the input data and parameters."""
        # Check required columns exist
        required_cols = [
            self.unit_variable_name,
            self.time_variable_name,
        ]

        for col in required_cols:
            if col not in self.data.columns:
                raise DataException(f"Required column '{col}' not found in data")

        # Check treated variable exists (either directly or via treatment_time)
        if self.treatment_time_variable_name is not None:
            if self.treatment_time_variable_name not in self.data.columns:
                raise DataException(
                    f"Treatment time column '{self.treatment_time_variable_name}' "
                    "not found in data"
                )
        elif self.treated_variable_name not in self.data.columns:
            raise DataException(
                f"Treated column '{self.treated_variable_name}' not found in data. "
                "Either provide treated_variable_name or treatment_time_variable_name."
            )

        # Validate absorbing treatment (once treated, always treated)
        self._validate_absorbing_treatment()

        # Estimator-specific validation
        if self.estimator == "etwfe":
            self._validate_etwfe_arguments()
        else:
            self._validate_imputation_arguments()

    def _validate_imputation_arguments(self) -> None:
        """Reject ETWFE-only arguments supplied to the imputation estimator.

        Silently ignoring a statistically meaningful argument is worse than
        erroring, so every ETWFE-only argument left at a non-default value is
        named explicitly.

        Raises
        ------
        ValueError
            If any ETWFE-only argument is non-default.
        """
        offenders = []
        if self._conditioning_arg is not None:
            offenders.append("conditioning")
        if self.n_leads != 0:
            offenders.append("n_leads")
        if self.max_event_time is not None:
            offenders.append("max_event_time")
        if self.covariates is not None:
            offenders.append("covariates")
        if self.se_type != "cluster":
            offenders.append("se_type")
        if offenders:
            named = ", ".join(offenders)
            raise ValueError(
                f"The argument(s) {named} only apply to estimator='etwfe' but "
                "estimator='imputation' was requested. Either pass "
                "estimator='etwfe' or drop the argument(s)."
            )

    def _validate_etwfe_arguments(self) -> None:
        """Validate ETWFE-specific arguments and warn about the ignored formula RHS.

        Raises
        ------
        ValueError
            If ``se_type`` is unrecognised, if a PyMC model other than
            :class:`~causalpy.pymc_models.ETWFERegression` was supplied, or if a
            requested covariate is not a column of the data.

        Warns
        -----
        UserWarning
            If the formula's right-hand side carries terms beyond ``1``, ``0``,
            ``C(unit)`` and ``C(time)``, all of which the ETWFE design ignores.
        """
        if self.se_type not in ("cluster", "classical"):
            raise ValueError(
                f"se_type must be 'cluster' or 'classical', got {self.se_type!r}"
            )

        if self._model_backend.is_bayesian and not isinstance(
            self.model, ETWFERegression
        ):
            raise ValueError(
                "estimator='etwfe' with a Bayesian model requires "
                "causalpy.pymc_models.ETWFERegression (the saturated design is "
                "built inside that model's fit()). Got "
                f"{type(self.model).__name__}. Pass model=None to get the default."
            )

        # Only the LHS of the formula is used; warn about anything else on the RHS.
        rhs = self.formula.split("~", 1)[1] if "~" in self.formula else ""
        allowed = {
            "1",
            "0",
            f"C({self.unit_variable_name})",
            f"C({self.time_variable_name})",
        }
        extras = [
            term
            for term in (t.strip() for t in rhs.split("+"))
            if term and term not in allowed
        ]
        if extras:
            warnings.warn(
                f"estimator='etwfe' ignores the formula right-hand side; the "
                f"term(s) {extras} will have no effect. ETWFE builds its own "
                "saturated design over unit, time and (cohort, event-time) "
                "cells. Pass additional covariates via covariates=[...] instead.",
                UserWarning,
                stacklevel=2,
            )

        if self.covariates:
            missing = [c for c in self.covariates if c not in self.data.columns]
            if missing:
                raise DataException(f"Covariate column(s) {missing} not found in data.")

    def _validate_absorbing_treatment(self) -> None:
        """Validate that treatment is absorbing (once treated, always treated)."""
        if self.treated_variable_name not in self.data.columns:
            # Will infer from treatment_time, skip validation here
            return

        for unit in self.data[self.unit_variable_name].unique():
            unit_data = self.data[
                self.data[self.unit_variable_name] == unit
            ].sort_values(self.time_variable_name)
            treated_values = unit_data[self.treated_variable_name].values

            # Find first treated period
            treated_indices = np.where(treated_values == 1)[0]
            if len(treated_indices) == 0:
                continue  # Never treated

            first_treated_idx = treated_indices[0]

            # Check all subsequent periods are also treated
            if not np.all(treated_values[first_treated_idx:] == 1):
                raise DataException(
                    f"Treatment is not absorbing for unit {unit}. "
                    "Once a unit is treated, it must remain treated in all "
                    "subsequent periods."
                )

    def _compute_treatment_times(self) -> None:
        """Compute treatment time G_i for each unit."""
        if self.treatment_time_variable_name is not None:
            # Use provided treatment time column
            # Get unique treatment time per unit
            g_map = (
                self.data.groupby(self.unit_variable_name)[
                    self.treatment_time_variable_name
                ]
                .first()
                .to_dict()
            )
            self.data["G"] = self.data[self.unit_variable_name].map(g_map)
        else:
            # Infer from treated variable: G = min{t : D_it = 1}
            g_map = {}
            for unit in self.data[self.unit_variable_name].unique():
                unit_data = self.data[self.data[self.unit_variable_name] == unit]
                treated_times = unit_data.loc[
                    unit_data[self.treated_variable_name] == 1, self.time_variable_name
                ]
                if len(treated_times) == 0:
                    g_map[unit] = self.never_treated_value
                else:
                    g_map[unit] = treated_times.min()
            self.data["G"] = self.data[self.unit_variable_name].map(g_map)

        # Store unique cohorts (excluding never-treated)
        self.cohorts = sorted(
            [g for g in self.data["G"].unique() if g != self.never_treated_value]
        )

    def _compute_event_times(self) -> None:
        """Compute event time (t - G) for each observation."""
        self.data["event_time"] = self.data[self.time_variable_name] - self.data["G"]
        # Set event_time to NaN for never-treated units
        self.data.loc[self.data["G"] == self.never_treated_value, "event_time"] = np.nan

    def _identify_untreated_observations(self) -> None:
        """Identify untreated observations for the training set."""
        # Untreated if: (t < G) OR (never-treated)
        is_never_treated = self.data["G"] == self.never_treated_value
        is_pre_treatment = self.data[self.time_variable_name] < self.data["G"]
        self.data["_is_untreated"] = is_never_treated | is_pre_treatment

        # Verify we have some training data
        n_untreated = self.data["_is_untreated"].sum()
        if n_untreated == 0:
            raise DataException(
                "No untreated observations found. Cannot fit the model. "
                "Ensure there are never-treated units or pre-treatment periods."
            )

    def _get_periods_without_untreated_support(self) -> set[Any]:
        """Return calendar periods with zero untreated observations."""
        untreated_periods = set(
            self.data.loc[self.data["_is_untreated"], self.time_variable_name].unique()
        )
        all_periods = set(self.data[self.time_variable_name].unique())
        return all_periods - untreated_periods

    def _get_non_identified_cohorts(self, periods: set[Any]) -> set[Any]:
        """Return cohorts with post-treatment cells in non-identified periods."""
        non_identified_cohorts: set[Any] = set()
        for cohort in self.cohorts:
            for period in periods:
                if period >= cohort:
                    non_identified_cohorts.add(cohort)
                    break
        return non_identified_cohorts

    def _check_att_identification(self) -> None:
        """Detect non-identified ATT cells and warn when untreated support is missing."""
        self.non_identified_periods_ = self._get_periods_without_untreated_support()
        self.non_identified_cohorts_ = self._get_non_identified_cohorts(
            self.non_identified_periods_
        )

        if not self.non_identified_periods_:
            return

        periods_str = ", ".join(str(p) for p in sorted(self.non_identified_periods_))
        cohorts_str = ", ".join(str(c) for c in sorted(self.non_identified_cohorts_))
        warnings.warn(
            "No untreated observations in calendar period(s) "
            f"{{{periods_str}}}; treatment effects for cohort(s) "
            f"{{{cohorts_str}}} are not identified at the affected post-treatment "
            "cells. Provide never-treated units or restrict the event window. "
            "Non-identified ATT(g, t) and ATT(e) cells are marked in the output "
            "tables (identified=False) with NaN estimates.",
            UserWarning,
            stacklevel=2,
        )

    def _is_calendar_period_identified(self, period: Any) -> bool:
        """Return whether calendar period ``period`` has untreated support."""
        return period not in self.non_identified_periods_

    def _is_event_time_att_identified(self, event_time: int) -> bool:
        """Return whether aggregated ATT(e) is identified."""
        for cohort in self.cohorts:
            period = cohort + event_time
            has_contributing_obs = (
                (self.data["G"] == cohort)
                & (self.data[self.time_variable_name] == period)
                & (self.data["event_time"] == event_time)
            ).any()
            if has_contributing_obs and not self._is_calendar_period_identified(period):
                return False
        return True

    def _mark_non_identified_att_rows(self, att_df: pd.DataFrame) -> pd.DataFrame:
        """Add ``identified`` column and mask non-identified point estimates."""
        if len(att_df) == 0:
            att_df = att_df.copy()
            att_df["identified"] = pd.Series(dtype=bool)
            return att_df

        att_df = att_df.copy()
        if "cohort" in att_df.columns and "time" in att_df.columns:
            att_df["identified"] = att_df["time"].map(
                self._is_calendar_period_identified
            )
        elif "event_time" in att_df.columns:
            att_df["identified"] = att_df["event_time"].apply(
                lambda e: self._is_event_time_att_identified(int(e))
            )
        else:
            att_df["identified"] = True

        value_columns = [
            col
            for col in ("att", "att_lower", "att_upper", "att_std")
            if col in att_df.columns
        ]
        for col in value_columns:
            att_df.loc[~att_df["identified"], col] = np.nan
        return att_df

    def _build_design_matrices(self) -> None:
        """Build design matrices using patsy."""
        # Build design matrix for the full data
        try:
            y, X = build_formula_matrices(self.formula, self.data)
        except PatsyError as err:
            raise FormulaException(f"Unable to evaluate formula: {err}") from err
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.outcome_variable_name = y.design_info.column_names[0]

        # Store full design matrix
        self.X_full = np.asarray(X)
        self.y_full = np.asarray(y)
        self._observed_outcome = pd.Series(self.y_full.ravel(), index=self.data.index)

        # Get untreated subset for training
        untreated_mask = np.asarray(self.data["_is_untreated"].values, dtype=bool)
        self.X_train = self.X_full[untreated_mask]
        self.y_train = self.y_full[untreated_mask]

    # -----------------------------------------------------------------------
    # ETWFE path
    # -----------------------------------------------------------------------

    def _build_etwfe_design(self) -> None:
        """Build the ETWFE index bundle and the model-specific design.

        Unlike :meth:`_build_design_matrices`, which trains on untreated cells
        only, the ETWFE design is fit on the **full sample**: that is the
        defining difference between the two estimators.

        Sets ``_etwfe_index``, ``event_time_grid_``, ``att_weights_``,
        ``outcome_variable_name`` and ``labels``; then either the xarray inputs
        for the PyMC path or the patsy design matrices for the OLS path.
        """
        self.outcome_variable_name = self.formula.split("~")[0].strip()

        index = _build_etwfe_index(
            self.data,
            unit_variable_name=self.unit_variable_name,
            time_variable_name=self.time_variable_name,
            treated_variable_name=self.treated_variable_name,
            never_treated_value=self.never_treated_value,
            cohorts=self.cohorts,
            n_leads=self.n_leads,
            reference_event_time=self.reference_event_time,
            max_event_time=self.max_event_time,
        )
        self._etwfe_index = index
        self.event_time_grid_ = index.ev_grid
        self.att_weights_ = pd.DataFrame(
            index.att_weights,
            index=pd.Index(index.cohorts, name="cohort"),
            columns=pd.Index(index.ev_grid, name="event_time"),
        )

        if self._model_backend.is_bayesian:
            self._build_etwfe_design_bayesian()
        elif self._model_backend.is_ols:
            self._build_etwfe_design_ols()
        else:  # pragma: no cover - defensive, BaseExperiment already checks
            raise ValueError("Model type not recognized")

        self._check_etwfe_identification()

    def _etwfe_coef_labels(self) -> list[str]:
        """Coefficient labels for the Bayesian ETWFE fit.

        The ETWFE posterior has no single ``beta``-style coefficient vector -- with
        no covariates it has no ``beta`` at all -- so the labels advertised to
        ``print_coefficients`` and to maketables are assembled by hand from the
        parameters that a reader actually wants in a coefficient table.

        Returns
        -------
        list of str
            ``["att", "tau_bar[k]", ..., "g_u", "g_t", <covariates>]``. The
            Mundlak entries are present only under ``conditioning="mundlak"``.
        """
        names = ["att"]
        names += [f"tau_bar[{int(k)}]" for k in self._etwfe_index.ev_grid]
        if self.conditioning == "mundlak":
            names += ["g_u", "g_t"]
        names += list(self.covariates or [])
        return names

    def _build_etwfe_design_bayesian(self) -> None:
        """Assemble the xarray inputs and coords for :class:`ETWFERegression`."""
        index = self._etwfe_index
        n_obs = len(self.data)
        covariates = list(self.covariates or [])
        # ``labels`` advertises the full ETWFE parameter block (see
        # :meth:`_etwfe_coef_labels`); the covariate names are kept separately
        # because they are the only ones that index into ``beta``.
        self._etwfe_covariate_labels = covariates
        self.labels = self._etwfe_coef_labels()

        X_values = (
            self.data[covariates].to_numpy(dtype=float)
            if covariates
            else np.empty((n_obs, 0), dtype=float)
        )
        obs_ind = np.arange(n_obs)
        self._etwfe_X = xr.DataArray(
            X_values,
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": obs_ind, "coeffs": covariates},
        )
        self._etwfe_y = xr.DataArray(
            self.data[[self.outcome_variable_name]].to_numpy(dtype=float),
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": obs_ind, "treated_units": ["unit_0"]},
        )
        self._etwfe_coords = {
            "obs_ind": obs_ind,
            "treated_units": ["unit_0"],
            "coeffs": covariates,
            "units": index.unit_levels,
            "periods": index.time_levels,
            "cohorts": index.cohorts,
            "ev": index.ev_grid,
        }

        if self.conditioning == "mundlak":
            dbar_unit, dbar_time, dbar_unit_raw, dbar_time_raw = _mundlak_means(
                self.data,
                unit_variable_name=self.unit_variable_name,
                time_variable_name=self.time_variable_name,
                treated_variable_name=self.treated_variable_name,
            )
            # Centred versions go to the model (orthogonalises g_u against mu_a
            # without changing the estimand); the raw means are kept for
            # inspection.
            self._etwfe_dbar_unit: np.ndarray | None = dbar_unit
            self._etwfe_dbar_time: np.ndarray | None = dbar_time
            self.data["dbar_unit"] = dbar_unit_raw
            self.data["dbar_time"] = dbar_time_raw
        else:
            self._etwfe_dbar_unit = None
            self._etwfe_dbar_time = None

    def _build_etwfe_design_ols(self) -> None:
        """Build the saturated patsy design for the OLS ETWFE path.

        A single ``_gk_cell`` categorical carries the whole effect surface. Its
        patsy reference level is ``"__none__"``, the level given to every
        out-of-scope observation, so each cell coefficient reads directly against
        the untreated baseline -- Wooldridge's saturated parametrisation.
        """
        index = self._etwfe_index
        cell_labels, label_to_cell = _etwfe_cell_labels(index)
        self.data[_GK_CELL] = cell_labels
        self._etwfe_label_to_cell = label_to_cell

        if not np.any(cell_labels == _GK_NONE):
            raise DataException(
                "Every observation loads on the ETWFE effect surface, leaving no "
                "untreated baseline for the saturated design. Reduce n_leads, or "
                "supply never-treated units."
            )

        rhs = [
            "1",
            f"C({self.unit_variable_name})",
            f"C({self.time_variable_name})",
            f"C({_GK_CELL}, Treatment(reference='{_GK_NONE}'))",
        ] + list(self.covariates or [])
        self.etwfe_formula_ = f"{self.outcome_variable_name} ~ " + " + ".join(rhs)

        try:
            y, X = build_formula_matrices(self.etwfe_formula_, self.data)
        except PatsyError as err:
            raise FormulaException(f"Unable to evaluate formula: {err}") from err
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.X_full = np.asarray(X)
        self.y_full = np.asarray(y)

        prefix = f"C({_GK_CELL}, Treatment(reference='{_GK_NONE}'))[T."
        column_positions = {name: j for j, name in enumerate(self.labels)}
        self._etwfe_cell_columns = {
            cell: column_positions[prefix + label + "]"]
            for label, cell in label_to_cell.items()
            if prefix + label + "]" in column_positions
        }

    def _check_etwfe_identification(self) -> None:
        """Warn (or raise) about identification problems in the ETWFE design.

        Called from :meth:`_build_etwfe_design` once the index bundle and the
        model-specific design exist, so that every check runs *before* any
        sampling or fitting cost is incurred.

        The checks are, in order:

        1. **Thin cells.** Any retained ``(cohort, event time)`` cell with fewer
           than :data:`_ETWFE_MIN_CELL_COUNT` observations is estimated from very
           little data. Under partial pooling that is survivable; on the OLS path
           it produces a very noisy coefficient.
        2. **Empty cells.** Individual empty cells are the normal shape of a
           staggered panel (the last-adopting cohort never reaches the largest
           event times), so this fires only once the surface is *mostly* holes.
           Event times that were empty for every cohort have already been removed
           from the grid, and warned about, by :func:`_build_etwfe_index`; they
           are excluded here so the two warnings do not overlap.
        3. **Leads without a never-treated group.** Estimating lead terms with no
           never-treated units leaves the pre-treatment profile identified only by
           differences in adoption timing -- genuine under-identification.
        4. **OLS rank deficiency.** Exact and cheap: if the saturated design does
           not have full column rank, the pseudo-inverse will silently return one
           of infinitely many solutions and the reported cell effects are
           arbitrary. That is a hard error.

        Raises
        ------
        DataException
            If the OLS ETWFE design matrix is rank deficient.

        Warns
        -----
        UserWarning
            For thin cells, a mostly-empty effect surface, or lead terms
            estimated without a never-treated comparison group.
        """
        index = self._etwfe_index

        # --- 1. thin cells ----------------------------------------------------
        thin_rows, thin_cols = np.where(
            (index.cell_counts > 0) & (index.cell_counts < _ETWFE_MIN_CELL_COUNT)
        )
        if thin_rows.size:
            order = np.argsort(index.cell_counts[thin_rows, thin_cols])
            worst = [
                f"(cohort={_format_cohort(index.cohorts[int(thin_rows[i])])}, "
                f"event_time={int(index.ev_grid[int(thin_cols[i])])}): "
                f"{int(index.cell_counts[int(thin_rows[i]), int(thin_cols[i])])} obs"
                for i in order[:_ETWFE_MAX_REPORTED_CELLS]
            ]
            more = thin_rows.size - len(worst)
            suffix = f" and {more} more" if more > 0 else ""
            warnings.warn(
                f"{thin_rows.size} (cohort, event time) cell(s) in the ETWFE "
                f"effect surface have fewer than {_ETWFE_MIN_CELL_COUNT} "
                f"observations: {', '.join(worst)}{suffix}. Effects for these "
                "cells are estimated from very little data.",
                UserWarning,
                stacklevel=2,
            )

        # --- 2. empty cells ---------------------------------------------------
        retained = {int(k) for k in index.ev_grid}
        empty = [(g, k) for g, k in index.dropped_cells if int(k) in retained]
        n_cells = len(index.cohorts) * index.ev_grid.size
        if n_cells and len(empty) > _ETWFE_EMPTY_CELL_SHARE * n_cells:
            listed = [
                f"(cohort={_format_cohort(g)}, event_time={int(k)})"
                for g, k in empty[:_ETWFE_MAX_REPORTED_CELLS]
            ]
            more = len(empty) - len(listed)
            suffix = f" and {more} more" if more > 0 else ""
            warnings.warn(
                f"{len(empty)} of {n_cells} (cohort, event time) cells in the "
                f"ETWFE effect surface have no observations: "
                f"{', '.join(listed)}{suffix}. Empty cells carry no likelihood "
                "contribution; under partial pooling they simply sample from the "
                "shared event-time profile, and on the OLS path they are absent "
                "from the design. Consider lowering max_event_time.",
                UserWarning,
                stacklevel=2,
            )

        # --- 3. leads with no never-treated group ------------------------------
        if self.n_leads > 0:
            never_treated = _never_treated_mask(
                self.data["G"], self.never_treated_value
            )
            n_never_treated = int(
                self.data.loc[never_treated, self.unit_variable_name].nunique()
            )
            if n_never_treated == 0:
                warnings.warn(
                    f"n_leads={self.n_leads} lead term(s) are being estimated but "
                    "the panel contains no never-treated units. The pre-treatment "
                    "profile is then identified only by differences in adoption "
                    "timing, and the lead coefficients are not a clean test of "
                    "parallel trends. Interpret them with care.",
                    UserWarning,
                    stacklevel=2,
                )

        # --- 4. OLS rank deficiency -------------------------------------------
        if self._model_backend.is_ols:
            n_columns = int(self.X_full.shape[1])
            rank = int(np.linalg.matrix_rank(self.X_full))
            if rank < n_columns:
                raise DataException(
                    f"The saturated ETWFE design matrix is rank deficient: "
                    f"{n_columns} columns but rank {rank} ({n_columns - rank} "
                    "deficiency). The cell effects are not identified and the "
                    "pseudo-inverse would silently return an arbitrary solution. "
                    "This usually means there is no clean untreated comparison "
                    "left: reduce n_leads, reduce max_event_time, or supply "
                    "never-treated units."
                )

    def _check_etwfe_convergence(self) -> None:
        """Warn if the in-model ATT has not converged (PyMC path only).

        Notes
        -----
        The ``sd_dev``/``dev`` block plus the Mundlak means is a realistic
        divergence source, so the aggregated ATT is worth checking directly rather
        than trusting a global summary.

        The finiteness guard is load-bearing: with a single chain -- which is what
        the test suite's mocked sampler produces -- R-hat is undefined, and an
        unguarded comparison would warn on every mocked fit.
        """
        if not self._model_backend.is_bayesian:
            return
        # The adapter's ``idata`` honestly returns None when the backend cannot
        # supply one or has not been fit, so no getattr probing is needed.
        idata = self._model_backend.idata
        if idata is None or "posterior" not in idata:
            return
        if "att" not in idata["posterior"]:  # pragma: no cover - defensive
            return
        try:
            rhat = float(np.asarray(az.rhat(idata, var_names=["att"])["att"]))
        except Exception:  # pragma: no cover - arviz refuses single-chain input
            return
        if np.isfinite(rhat) and rhat > _ETWFE_RHAT_THRESHOLD:
            warnings.warn(
                f"R-hat for the in-model ATT is {rhat:.3f}, above "
                f"{_ETWFE_RHAT_THRESHOLD}. The posterior for att_ may not have "
                "converged. Increase tune/draws, raise target_accept, or switch "
                "to conditioning='dummy'.",
                UserWarning,
                stacklevel=2,
            )

    def print_coefficients(self, round_to: int | None = None) -> None:
        """Ask the model to print its coefficients.

        Overrides :meth:`~causalpy.experiments.base.BaseExperiment.print_coefficients`
        because on the Bayesian ETWFE path ``self.labels`` advertises the whole
        parameter block (``att``, ``tau_bar[k]``, ``g_u``, ``g_t``, covariates) for
        the benefit of maketables, whereas
        :meth:`~causalpy.pymc_models.ETWFERegression.print_coefficients` expects
        only the covariate names, which are the ones that index into ``beta``.

        Parameters
        ----------
        round_to : int, optional
            Number of significant figures to round to. Defaults to None,
            in which case 2 significant figures are used.
        """
        if self.estimator == "etwfe" and self._model_backend.is_bayesian:
            self.model.print_coefficients(
                list(getattr(self, "_etwfe_covariate_labels", [])), round_to
            )
            return
        super().print_coefficients(round_to)

    @property
    def __maketables_coef_draws__(self) -> "xr.DataArray | None":
        """Posterior coefficient draws for the maketables export hook.

        ``maketables_adapters._resolve_pymc_coef_draws`` looks for a ``beta``-like
        variable, which a covariate-free ETWFE fit simply does not have. This hook
        is the documented escape route: it assembles the ETWFE parameters that
        belong in a coefficient table into one array on a ``coeffs`` dimension
        whose coordinate matches :attr:`labels`.

        Returns
        -------
        xarray.DataArray or None
            ``None`` for every non-ETWFE or non-PyMC fit, which restores the
            adapter's standard resolution path.
        """
        if self.estimator != "etwfe" or not self._model_backend.is_bayesian:
            return None
        posterior = self._etwfe_idata["posterior"]

        pieces: list[xr.DataArray] = [posterior["att"]]
        tau_bar = posterior["tau_bar"]
        pieces += [
            tau_bar.sel(ev=ev_value, drop=True)
            for ev_value in tau_bar.coords["ev"].values
        ]
        pieces += [posterior[name] for name in ("g_u", "g_t") if name in posterior]
        if "beta" in posterior:
            beta = posterior["beta"]
            if "treated_units" in beta.dims:
                beta = beta.isel(treated_units=0, drop=True)
            pieces += [
                beta.sel(coeffs=name, drop=True)
                for name in getattr(self, "_etwfe_covariate_labels", [])
            ]

        stacked = xr.concat(
            [piece.reset_coords(drop=True) for piece in pieces], dim="coeffs"
        )
        return stacked.assign_coords(coeffs=self._etwfe_coef_labels())

    @property
    def _etwfe_idata(self) -> az.InferenceData:
        """The fitted PyMC model's inference data.

        Delegates to the backend adapter rather than probing ``self.model``
        with ``getattr``: ARCHITECTURE.md is explicit that capability is
        discovered through ``ModelAdapter``, not through ``AttributeError``.

        Raises
        ------
        RuntimeError
            If the backend is not Bayesian or the model has not been fit.
        """
        return self._model_backend.require_idata()

    def _fit_model_etwfe(self) -> None:
        """Fit the ETWFE model on the full sample.

        Both branches go through ``self._model_backend`` rather than touching
        ``self.model`` directly, so backend coercion, prediction
        canonicalisation and the ``fit_intercept=False`` clone-and-warn all stay
        in :mod:`causalpy.experiments.model_adapter` (see ARCHITECTURE.md).
        """
        backend = self._model_backend
        if backend.is_bayesian:
            index = self._etwfe_index
            # The panel index arrays ride the adapter's ``**fit_kwargs``
            # passthrough; ETWFERegression.fit widens the standard signature to
            # receive them.
            backend.fit(
                X=self._etwfe_X,
                y=self._etwfe_y,
                coords=self._etwfe_coords,
                unit_idx=index.unit_idx,
                time_idx=index.time_idx,
                cohort_idx=index.cohort_idx,
                ev_idx=index.ev_idx,
                effect_indicator=index.effect_indicator,
                att_weights=index.att_weights,
                dbar_unit=self._etwfe_dbar_unit,
                dbar_time=self._etwfe_dbar_time,
                conditioning=self.conditioning or "mundlak",
            )
            # ETWFE predicts in-sample only -- every index array is bound to
            # this panel -- so pass the design back in rather than new data.
            # The adapter returns the canonical (chain, draw, obs_ind,
            # treated_units) container the rest of the class dispatches on via
            # ``has_posterior_draws``.
            self.y_pred = backend.predict(X=self._etwfe_X)
        elif backend.is_ols:
            backend.fit(X=self.X_full, y=self.y_full)
            self._etwfe_coefs = np.asarray(backend.coefficients(), dtype=float).ravel()
            # Singleton chain/draw dims, so ``has_posterior_draws`` reports
            # False and the point-estimate branches are taken downstream.
            self.y_pred = backend.predict(X=self.X_full)
            self._etwfe_fitted = np.squeeze(np.asarray(self.y_pred.values, dtype=float))
        else:  # pragma: no cover - defensive, BaseExperiment already checks
            raise ValueError("Model type not recognized")

    def _populate_etwfe_fitted_values(self) -> None:
        """Populate ``y_hat0`` and ``tau_hat`` from the fitted effect surface.

        Notes
        -----
        Unlike the imputation estimator, where ``y_hat0`` is an out-of-sample
        prediction from a model that never saw the treated cells, the ETWFE
        ``y_hat0`` is **model-implied**: it is the in-sample fitted value with the
        estimated cell effect subtracted back out,
        ``y_hat0 = mu - E_it * tau[g, k]``. ``tau_hat = y - y_hat0`` therefore
        equals the estimated cell effect plus the fit residual, and reduces to the
        cell effect exactly when the model fits perfectly.
        """
        index = self._etwfe_index
        if self._model_backend.is_bayesian:
            idata = self._etwfe_idata
            mu = (
                idata["posterior_predictive"]["mu"]
                .mean(dim=["chain", "draw"])
                .isel(treated_units=0)
                .values
            )
            tau_mean = idata["posterior"]["tau"].mean(dim=["chain", "draw"]).values
        else:
            mu = self._etwfe_fitted
            tau_mean = self._etwfe_tau_matrix_ols()

        tau_obs = index.effect_indicator * tau_mean[index.cohort_idx, index.ev_idx]
        self.data["y_hat0"] = np.asarray(mu, dtype=float) - tau_obs
        self.data["tau_hat"] = np.nan
        treated_mask = ~self.data["_is_untreated"]
        self.data.loc[treated_mask, "tau_hat"] = (
            self.data.loc[treated_mask, self.outcome_variable_name]
            - self.data.loc[treated_mask, "y_hat0"]
        )
        self.data_ = self.data.copy()

    def _etwfe_tau_matrix_ols(self) -> np.ndarray:
        """Point estimates of ``tau[g, k]`` laid out as a ``(cohorts, ev)`` matrix.

        Cells absent from the design (no observations) are left at zero.
        """
        index = self._etwfe_index
        tau = np.zeros((len(index.cohorts), index.ev_grid.size), dtype=float)
        for (gi, ei), column in self._etwfe_cell_columns.items():
            tau[gi, ei] = self._etwfe_coefs[column]
        return tau

    def _etwfe_ols_vcov(self) -> np.ndarray:
        """Variance-covariance matrix of the OLS ETWFE coefficients.

        Returns
        -------
        np.ndarray
            ``(p, p)`` matrix. Cluster-by-unit sandwich when
            ``se_type="cluster"``, classical homoskedastic otherwise.

        Warns
        -----
        UserWarning
            If the fitter is not plain ordinary least squares, in which case the
            sandwich formula is only an approximation.
        """
        from sklearn.linear_model import LinearRegression as SklearnLinearRegression

        if not isinstance(self.model, SklearnLinearRegression):
            warnings.warn(
                f"Standard errors for estimator='etwfe' assume ordinary least "
                f"squares, but the fitted model is {type(self.model).__name__}. "
                "The reported standard errors are approximate.",
                UserWarning,
                stacklevel=2,
            )

        X = self.X_full
        n, p = X.shape
        resid = self.y_full.ravel() - self._etwfe_fitted
        xtx_inv = np.linalg.pinv(X.T @ X)
        rank = int(np.linalg.matrix_rank(X))
        dof = max(n - rank, 1)

        if self.se_type == "classical":
            sigma2 = float(resid @ resid) / dof
            return sigma2 * xtx_inv

        units = self.data[self.unit_variable_name].to_numpy()
        unique_units = pd.unique(units)
        meat = np.zeros((p, p), dtype=float)
        for unit in unique_units:
            mask = units == unit
            score = X[mask].T @ resid[mask]
            meat += np.outer(score, score)
        n_clusters = len(unique_units)
        correction = (
            (n_clusters / (n_clusters - 1)) * ((n - 1) / dof) if n_clusters > 1 else 1.0
        )
        return correction * (xtx_inv @ meat @ xtx_inv)

    def _etwfe_column_weights(self, ev_position: int) -> tuple[np.ndarray, int]:
        """Weights over the coefficient vector that aggregate one event-time column.

        The in-column weights are ``w_g = M_gk / sum_g M_gk`` with ``M_gk`` the
        number of in-scope observations in cell ``(g, k)``, so the resulting
        aggregate is a proper linear combination of the fitted coefficients.

        Parameters
        ----------
        ev_position : int
            Position of the event time within ``event_time_grid_``.

        Returns
        -------
        tuple[np.ndarray, int]
            The weight vector over all design columns, and the total number of
            in-scope observations behind it.
        """
        index = self._etwfe_index
        counts = index.cell_counts[:, ev_position]
        present = [
            gi
            for gi in range(len(index.cohorts))
            if counts[gi] > 0 and (gi, ev_position) in self._etwfe_cell_columns
        ]
        weights = np.zeros(self.X_full.shape[1], dtype=float)
        if not present:
            return weights, 0
        total = int(counts[present].sum())
        for gi in present:
            weights[self._etwfe_cell_columns[(gi, ev_position)]] += counts[gi] / total
        return weights, total

    def _compute_etwfe_effects_ols(self) -> None:
        """Extract ATTs, their standard errors and the effect surface from OLS."""
        index = self._etwfe_index
        beta = self._etwfe_coefs
        vcov = self._etwfe_ols_vcov()

        # --- scalar ATT: a linear combination over the treated cells ----------
        att_weight_vector = np.zeros(self.X_full.shape[1], dtype=float)
        for (gi, ei), column in self._etwfe_cell_columns.items():
            att_weight_vector[column] += index.att_weights[gi, ei]
        self.att_ = float(att_weight_vector @ beta)
        self.att_se_ = float(
            np.sqrt(max(float(att_weight_vector @ vcov @ att_weight_vector), 0.0))
        )

        # --- event-time ATTs ---------------------------------------------------
        rows: list[dict] = []
        for ei, event_time in enumerate(index.ev_grid):
            weights, n_obs = self._etwfe_column_weights(ei)
            if n_obs == 0:
                continue
            rows.append(
                {
                    "event_time": int(event_time),
                    "att": float(weights @ beta),
                    "att_std": float(
                        np.sqrt(max(float(weights @ vcov @ weights), 0.0))
                    ),
                    "n_obs": n_obs,
                }
            )
        self.att_event_time_ = self._mark_non_identified_att_rows(
            self._apply_event_window(pd.DataFrame(rows))
        )

        # --- effect surface and group-time ATTs --------------------------------
        surface: list[dict] = []
        group_time: list[dict] = []
        for (gi, ei), column in sorted(self._etwfe_cell_columns.items()):
            cohort = index.cohorts[gi]
            event_time = int(index.ev_grid[ei])
            std = float(np.sqrt(max(float(vcov[column, column]), 0.0)))
            n_obs = int(index.cell_counts[gi, ei])
            surface.append(
                {
                    "cohort": cohort,
                    "event_time": event_time,
                    "att": float(beta[column]),
                    "att_std": std,
                    "n_obs": n_obs,
                }
            )
            if event_time >= 0:
                group_time.append(
                    {
                        "cohort": cohort,
                        "time": _as_calendar_time(cohort, event_time),
                        "att": float(beta[column]),
                        "att_std": std,
                        "n_obs": n_obs,
                    }
                )
        self.tau_surface_ = pd.DataFrame(surface)
        self.att_group_time_ = self._mark_non_identified_att_rows(
            pd.DataFrame(group_time)
        )

    def _compute_etwfe_effects_bayesian(self, hdi_prob: float = HDI_PROB) -> None:
        """Extract ATTs and the effect surface from the ETWFE posterior.

        Parameters
        ----------
        hdi_prob : float, optional
            Probability mass for the interval bounds. Defaults to
            :data:`causalpy.constants.HDI_PROB`.
        """
        self.hdi_prob_ = hdi_prob
        index = self._etwfe_index
        posterior = self._etwfe_idata["posterior"]

        # The in-model ATT deterministic: read its draws straight off the
        # posterior rather than reconstructing it from differenced predictions.
        self.att_ = posterior["att"]
        self.att_se_ = None

        tau_draws = posterior["tau"].values  # (chain, draw, cohorts, ev)
        lower_pct = (1 - hdi_prob) / 2 * 100
        upper_pct = (1 + hdi_prob) / 2 * 100

        self.att_event_time_ = self._mark_non_identified_att_rows(
            self._apply_event_window(
                self._etwfe_att_event_time_bayesian(tau_draws, lower_pct, upper_pct)
            )
        )

        surface: list[dict] = []
        group_time: list[dict] = []
        for gi, cohort in enumerate(index.cohorts):
            for ei, event_time in enumerate(index.ev_grid):
                n_obs = int(index.cell_counts[gi, ei])
                if n_obs == 0:
                    continue
                draws = tau_draws[:, :, gi, ei]
                record = {
                    "cohort": cohort,
                    "event_time": int(event_time),
                    "att": float(draws.mean()),
                    "att_lower": float(np.percentile(draws, lower_pct)),
                    "att_upper": float(np.percentile(draws, upper_pct)),
                    "n_obs": n_obs,
                }
                surface.append(record)
                if event_time >= 0:
                    group_time.append(
                        {
                            "cohort": cohort,
                            "time": _as_calendar_time(cohort, int(event_time)),
                            "att": record["att"],
                            "att_lower": record["att_lower"],
                            "att_upper": record["att_upper"],
                        }
                    )
        self.tau_surface_ = pd.DataFrame(surface)
        self.att_group_time_ = self._mark_non_identified_att_rows(
            pd.DataFrame(group_time)
        )

    def _etwfe_att_event_time_bayesian(
        self, tau_draws: np.ndarray, lower_pct: float, upper_pct: float
    ) -> pd.DataFrame:
        """Aggregate the posterior effect surface down to event-time ATTs.

        Parameters
        ----------
        tau_draws : np.ndarray
            Posterior draws of ``tau`` with shape ``(chain, draw, cohorts, ev)``.
        lower_pct, upper_pct : float
            Percentiles for the interval bounds.

        Returns
        -------
        pd.DataFrame
            Columns ``event_time, att, att_lower, att_upper, n_obs``.
        """
        index = self._etwfe_index
        rows: list[dict] = []
        for ei, event_time in enumerate(index.ev_grid):
            counts = index.cell_counts[:, ei].astype(float)
            total = counts.sum()
            if total == 0:
                continue
            weights = counts / total
            draws = np.tensordot(tau_draws[:, :, :, ei], weights, axes=([2], [0]))
            rows.append(
                {
                    "event_time": int(event_time),
                    "att": float(draws.mean()),
                    "att_lower": float(np.percentile(draws, lower_pct)),
                    "att_upper": float(np.percentile(draws, upper_pct)),
                    "n_obs": int(total),
                }
            )
        return pd.DataFrame(rows)

    def _apply_event_window(self, table: pd.DataFrame) -> pd.DataFrame:
        """Filter an event-time table to ``event_window``.

        ``event_window`` restricts *reporting* only; ``n_leads`` and
        ``max_event_time`` control what is *estimated*.
        """
        if self.event_window is None or table.empty:
            return table
        keep = (table["event_time"] >= self.event_window[0]) & (
            table["event_time"] <= self.event_window[1]
        )
        return table[keep].reset_index(drop=True)

    def _fit_model(self) -> None:
        """Fit the model on untreated observations only."""
        n_train = self.X_train.shape[0]
        X_train_xr = xr.DataArray(
            self.X_train,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": np.arange(n_train),
                "coeffs": self.labels,
            },
        )
        y_train_xr = xr.DataArray(
            self.y_train,
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": np.arange(n_train), "treated_units": ["unit_0"]},
        )
        self._model_backend.fit(
            X=X_train_xr,
            y=y_train_xr,
            coords=build_coords(self.labels, n_train),
        )

    def _predict_counterfactuals(self) -> None:
        """Predict counterfactual outcomes for all observations."""
        n_full = self.X_full.shape[0]
        X_full_xr = xr.DataArray(
            self.X_full,
            dims=["obs_ind", "coeffs"],
            coords={
                "obs_ind": np.arange(n_full),
                "coeffs": self.labels,
            },
        )
        self.y_pred = self._model_backend.predict(X=X_full_xr)
        self.data["y_hat0"] = (
            self.y_pred.mean(dim=["chain", "draw"]).isel(treated_units=0).values
        )

    def _compute_treatment_effects(self) -> None:
        """Compute treatment effects tau_hat = y - y_hat0 for treated observations."""
        self.data["tau_hat"] = np.nan  # Initialize with NaN
        treated_mask = ~self.data["_is_untreated"]
        self.data.loc[treated_mask, "tau_hat"] = (
            self._observed_outcome.loc[treated_mask]
            - self.data.loc[treated_mask, "y_hat0"]
        )

        # Store augmented data
        self.data_ = self.data.copy()

    def _aggregate_effects(self) -> None:
        """Aggregate effects to group-time and event-time ATTs.

        This method aggregates individual treatment effects into:
        1. Group-time ATTs: ATT(g, t) for each cohort g and calendar time t
        2. Event-time ATTs: ATT(e) for each event-time e = t - G

        For event-time ATTs, this includes both:

        - Post-treatment effects (event_time >= 0): actual treatment effects
        - Pre-treatment effects (event_time < 0): placebo/residual checks

        Pre-treatment effects are computed as residuals (y - y_hat0) for
        eventually-treated units before they receive treatment. These serve
        as a placebo check - if the parallel trends assumption holds, they
        should be centered around zero.
        """
        treated_data = self.data[~self.data["_is_untreated"]].copy()

        # Also get pre-treatment data for eventually-treated units (placebo check)
        # These are observations where: G != never_treated_value AND event_time < 0
        is_eventually_treated = self.data["G"] != self.never_treated_value
        is_pre_treatment = self.data["event_time"] < 0
        pretreatment_data = self.data[is_eventually_treated & is_pre_treatment].copy()

        # The two helpers compute genuinely different statistics: HDI bounds
        # need posterior draws, sample dispersion needs only point residuals.
        if has_posterior_draws(self.y_pred):
            self._aggregate_effects_bayesian(treated_data, pretreatment_data)
        else:
            self._aggregate_effects_ols(treated_data, pretreatment_data)

    def _aggregate_effects_bayesian(
        self,
        treated_data: pd.DataFrame,
        pretreatment_data: pd.DataFrame,
        hdi_prob: float = HDI_PROB,
    ) -> None:
        """Aggregate effects for Bayesian model with posterior uncertainty.

        Parameters
        ----------
        treated_data : pd.DataFrame
            DataFrame containing only treated observations (event_time >= 0)
        pretreatment_data : pd.DataFrame
            DataFrame containing pre-treatment observations from eventually-treated
            units (event_time < 0) for placebo check
        hdi_prob : float, optional
            Probability mass for the HDI interval bounds. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).
        """
        # Store the HDI probability used for interval computation
        self.hdi_prob_ = hdi_prob
        lower_pct = (1 - hdi_prob) / 2 * 100
        upper_pct = (1 + hdi_prob) / 2 * 100

        # Get posterior draws for mu
        mu_draws = self.y_pred.isel(treated_units=0)

        # Get observed y for all observations
        y_observed = self._observed_outcome.to_numpy()

        # Compute tau draws for all observations
        # tau_draws has shape (chain, draw, obs_ind)
        tau_draws_all = y_observed - mu_draws.values

        # Get treated observation indices for group-time ATTs
        _is_untreated = np.asarray(self.data["_is_untreated"].values, dtype=bool)
        treated_mask = ~_is_untreated
        treated_indices = np.where(treated_mask)[0]
        tau_draws_treated = tau_draws_all[:, :, treated_indices]
        event_time_treated = np.asarray(treated_data["event_time"].values)

        # --- Group-time ATTs (post-treatment only) ---
        gt_groups = treated_data.groupby(["G", self.time_variable_name]).groups
        att_gt_rows: list[dict] = []
        for key, idx in gt_groups.items():
            g_val = key[0]  # type: ignore[index]
            t_val = key[1]  # type: ignore[index]
            # Find positions in treated_indices
            positions = [np.where(treated_indices == i)[0][0] for i in idx]
            tau_gt = tau_draws_treated[:, :, positions].mean(axis=2)
            att_gt_rows.append(
                {
                    "cohort": g_val,
                    "time": t_val,
                    "att": float(tau_gt.mean()),
                    "att_lower": float(np.percentile(tau_gt, lower_pct)),
                    "att_upper": float(np.percentile(tau_gt, upper_pct)),
                }
            )
        self.att_group_time_ = self._mark_non_identified_att_rows(
            pd.DataFrame(att_gt_rows)
        )

        # --- Event-time ATTs (including pre-treatment placebo) ---
        att_et_rows: list[dict] = []

        # Pre-treatment placebo effects (event_time < 0)
        if len(pretreatment_data) > 0:
            pretreat_indices = pretreatment_data.index.values
            pretreat_idx_positions = np.array(
                [np.where(self.data.index == idx)[0][0] for idx in pretreat_indices]
            )
            tau_draws_pretreat = tau_draws_all[:, :, pretreat_idx_positions]
            event_time_pretreat = np.asarray(pretreatment_data["event_time"].values)

            event_times_pre = np.unique(
                event_time_pretreat[~np.isnan(event_time_pretreat)]
            )
            # Apply event window filter if specified
            if self.event_window is not None:
                event_times_pre = event_times_pre[
                    (event_times_pre >= self.event_window[0])
                    & (event_times_pre <= self.event_window[1])
                ]

            for e in sorted(event_times_pre):
                e_mask = event_time_pretreat == e
                if e_mask.sum() == 0:
                    continue
                positions_arr = np.where(e_mask)[0]
                tau_e = tau_draws_pretreat[:, :, positions_arr].mean(axis=2)
                att_et_rows.append(
                    {
                        "event_time": int(e),
                        "att": float(tau_e.mean()),
                        "att_lower": float(np.percentile(tau_e, lower_pct)),
                        "att_upper": float(np.percentile(tau_e, upper_pct)),
                        "n_obs": int(e_mask.sum()),
                    }
                )

        # Post-treatment effects (event_time >= 0)
        event_times_post = np.unique(event_time_treated[~np.isnan(event_time_treated)])
        if self.event_window is not None:
            event_times_post = event_times_post[
                (event_times_post >= self.event_window[0])
                & (event_times_post <= self.event_window[1])
            ]

        for e in sorted(event_times_post):
            e_mask = event_time_treated == e
            if e_mask.sum() == 0:
                continue
            positions_arr = np.where(e_mask)[0]
            tau_e = tau_draws_treated[:, :, positions_arr].mean(axis=2)
            att_et_rows.append(
                {
                    "event_time": int(e),
                    "att": float(tau_e.mean()),
                    "att_lower": float(np.percentile(tau_e, lower_pct)),
                    "att_upper": float(np.percentile(tau_e, upper_pct)),
                    "n_obs": int(e_mask.sum()),
                }
            )

        self.att_event_time_ = self._mark_non_identified_att_rows(
            pd.DataFrame(att_et_rows)
        )

    def _aggregate_effects_ols(
        self, treated_data: pd.DataFrame, pretreatment_data: pd.DataFrame
    ) -> None:
        """Aggregate effects for OLS model (point estimates only).

        Parameters
        ----------
        treated_data : pd.DataFrame
            DataFrame containing only treated observations (event_time >= 0)
        pretreatment_data : pd.DataFrame
            DataFrame containing pre-treatment observations from eventually-treated
            units (event_time < 0) for placebo check
        """
        # --- Group-time ATTs (post-treatment only) ---
        att_gt = (
            treated_data.groupby(["G", self.time_variable_name])["tau_hat"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        att_gt.columns = ["cohort", "time", "att", "att_std", "n_obs"]
        self.att_group_time_ = self._mark_non_identified_att_rows(att_gt)

        # --- Event-time ATTs (including pre-treatment placebo) ---
        # Compute tau_hat for pre-treatment observations (residuals)
        if len(pretreatment_data) > 0:
            pretreatment_data = pretreatment_data.copy()
            pretreatment_data["tau_hat"] = (
                self._observed_outcome.loc[pretreatment_data.index].to_numpy()
                - pretreatment_data["y_hat0"].to_numpy()
            )

        # Combine pre-treatment and post-treatment for event-time aggregation
        event_data = pd.concat([pretreatment_data, treated_data], ignore_index=True)

        # Apply event window filter if specified
        if self.event_window is not None:
            event_data = event_data[
                (event_data["event_time"] >= self.event_window[0])
                & (event_data["event_time"] <= self.event_window[1])
            ]

        att_et = (
            event_data.groupby("event_time")["tau_hat"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        att_et.columns = ["event_time", "att", "att_std", "n_obs"]
        att_et["event_time"] = att_et["event_time"].astype(int)
        self.att_event_time_ = self._mark_non_identified_att_rows(att_et)

    def summary(
        self, round_to: int | None = 2, include_group_time: bool = False
    ) -> None:
        """Print summary of main results.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals for rounding. Defaults to 2.
        include_group_time : bool
            Whether to print the disaggregated cohort-by-calendar-time
            ``ATT(g, t)`` table after the event-time estimates. Defaults to
            ``False``.
        """
        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        if self.estimator == "etwfe":
            print(f"Estimator: ETWFE (conditioning={self.conditioning})")
            if self.att_se_ is not None:
                print(f"Overall ATT: {self.att_:.4g} (se {self.att_se_:.4g})")
            elif self.att_ is not None:
                print(f"Overall ATT: {float(np.asarray(self.att_).mean()):.4g}")
        print(f"Number of units: {self.data[self.unit_variable_name].nunique()}")
        print(f"Number of time periods: {self.data[self.time_variable_name].nunique()}")
        print(f"Treatment cohorts: {self.cohorts}")
        n_never_treated = self.data.loc[
            self.data["G"] == self.never_treated_value, self.unit_variable_name
        ].nunique()
        print(f"Never-treated units: {n_never_treated}")
        print("\nEvent-time estimates:")
        att_et = self.att_event_time_.copy()
        # Add indicator column for clarity
        att_et["type"] = att_et["event_time"].apply(
            lambda x: "placebo" if x < 0 else "ATT"
        )
        # Reorder columns to put type first
        cols = ["event_time", "type"] + [
            c for c in att_et.columns if c not in ["event_time", "type"]
        ]
        print(att_et[cols].to_string(index=False))
        if include_group_time:
            print("\nGroup-time estimates:")
            print(self.att_group_time_.to_string(index=False))
        print("\nModel coefficients:")
        self.print_coefficients(round_to)

    def _draw_reference_marker(self, ax: plt.Axes, *, label: bool = True) -> bool:
        """Mark the omitted reference event time at zero on an event-study axis.

        The ETWFE effect surface pins the reference event time by *omitting* its
        column, so no estimate exists there. Without a marker a reader sees a hole
        in the event study and has to guess why. Does nothing for the imputation
        estimator, which normalises nothing.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to draw on.
        label : bool, default=True
            Whether to attach a legend label to the marker.

        Returns
        -------
        bool
            Whether a marker was drawn.
        """
        if self.estimator != "etwfe":
            return False
        ax.plot(
            [self.reference_event_time],
            [0.0],
            marker="o",
            markersize=8,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.5,
            linestyle="none",
            zorder=4,
            label="reference (normalised)" if label else None,
        )
        return True

    def _ols_error_bar_se(self, table: pd.DataFrame) -> np.ndarray:
        """Standard errors behind the OLS event-study error bars.

        The ``att_std`` column means different things on the two estimator paths,
        and conflating them makes the ETWFE bars roughly ``sqrt(n_obs)`` times too
        narrow:

        - **imputation**: ``att_std`` is the *sample standard deviation* of the
          imputed treatment effects within the event-time group, so the standard
          error of their mean is ``att_std / sqrt(n_obs)``.
        - **ETWFE**: ``att_std`` is *already a standard error* -- the
          linear-combination SE ``sqrt(w'Vw)`` of the aggregated cell effects --
          and must be used as it stands.

        Parameters
        ----------
        table : pd.DataFrame
            A slice of ``att_event_time_`` carrying ``att_std`` and ``n_obs``.

        Returns
        -------
        np.ndarray
            One standard error per row.
        """
        att_std = np.asarray(table["att_std"], dtype=float)
        if self.estimator == "etwfe":
            return att_std
        return att_std / np.sqrt(np.asarray(table["n_obs"], dtype=float))

    def plot_tau_surface(
        self, hdi_prob: float = HDI_PROB
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot the ETWFE effect surface, one panel per adoption cohort.

        The aggregated event study collapses every cohort into a single line, which
        is exactly the information that the extended two-way fixed effects
        parametrisation exists to preserve. This figure puts each cohort's
        ``tau[g, k]`` profile in its own panel, so cohort heterogeneity -- effects
        that grow faster for later adopters, say -- is visible directly.

        Parameters
        ----------
        hdi_prob : float, optional
            Probability mass for the credible band on the Bayesian path. Defaults
            to :data:`causalpy.constants.HDI_PROB`. Ignored on the OLS path, which
            draws a 95% normal-approximation interval from the coefficient
            standard errors.

        Returns
        -------
        tuple[plt.Figure, list[plt.Axes]]
            The figure and one axis per cohort, in cohort order.

        Raises
        ------
        ValueError
            If the experiment was run with ``estimator="imputation"``, which
            produces no ``tau_surface_``.
        """
        if self.estimator != "etwfe":
            raise ValueError(
                "plot_tau_surface() is only available for estimator='etwfe'. The "
                "imputation estimator produces no (cohort, event-time) effect "
                "surface -- it imputes counterfactuals observation by observation "
                "-- so there is nothing to plot. Use plot() for the event study, "
                "or re-run with estimator='etwfe'."
            )

        index = self._etwfe_index
        surface = self.tau_surface_
        cohorts = [g for g in index.cohorts if (surface["cohort"] == g).any()]

        fig, axes_array = plt.subplots(
            len(cohorts),
            1,
            figsize=(9, 3.0 * len(cohorts)),
            sharex=True,
            squeeze=False,
        )
        axes: list[plt.Axes] = list(axes_array.ravel())

        is_bayesian = self._model_backend.is_bayesian
        if is_bayesian:
            tau_draws = self._etwfe_idata["posterior"]["tau"].values
            lower_pct = (1 - hdi_prob) / 2 * 100
            upper_pct = (1 + hdi_prob) / 2 * 100
            band_label = f"{int(hdi_prob * 100)}% HDI"
            line_label = "posterior mean"
        else:
            band_label = "95% CI"
            line_label = "point estimate"

        for ax, cohort in zip(axes, cohorts, strict=True):
            gi = index.cohorts.index(cohort)
            rows = surface[surface["cohort"] == cohort].sort_values("event_time")
            event_times = rows["event_time"].to_numpy(dtype=float)
            mean = rows["att"].to_numpy(dtype=float)

            if is_bayesian:
                positions = [
                    int(np.flatnonzero(index.ev_grid == int(k))[0])
                    for k in rows["event_time"]
                ]
                draws = tau_draws[:, :, gi, positions]
                lower = np.percentile(draws, lower_pct, axis=(0, 1))
                upper = np.percentile(draws, upper_pct, axis=(0, 1))
            else:
                se = rows["att_std"].to_numpy(dtype=float)
                lower = mean - 1.96 * se
                upper = mean + 1.96 * se

            ax.fill_between(
                event_times, lower, upper, alpha=0.25, color="C0", label=band_label
            )
            ax.plot(event_times, mean, marker="o", color="C0", label=line_label)
            ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.7)
            ax.axvline(x=-0.5, color="red", linestyle="-", linewidth=1.5, alpha=0.7)
            self._draw_reference_marker(ax)
            ax.set_title(f"Cohort {_format_cohort(cohort)}", fontsize=12)
            ax.set_ylabel("Effect", fontsize=11)
            ax.legend(fontsize=LEGEND_FONT_SIZE)

        axes[-1].set_xlabel("Event Time (periods relative to treatment)", fontsize=12)
        fig.suptitle("ETWFE effect surface by cohort", fontsize=14)
        fig.tight_layout()
        return fig, axes

    def plot(
        self,
        *,
        hdi_prob: float | None = None,
        figsize: tuple[float, float] = (10, 6),
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot the staggered difference-in-differences event study.

        Parameters
        ----------
        hdi_prob : float, optional
            Probability mass of the highest density interval shown by the
            error bars. Unlike most other CausalPy experiments, ``hdi_prob``
            for staggered DiD is fixed at fit time during effect aggregation
            and the resulting bounds are cached on the instance. If
            supplied here, the value must match the cached
            :attr:`hdi_prob_`; otherwise a :class:`ValueError` is raised.
            Pass ``None`` (the default) to plot using the cached value.
            Ignored for OLS models.
        figsize : tuple of (float, float)
            Width and height of the figure in inches, passed to
            :func:`matplotlib.pyplot.subplots`. Defaults to ``(10, 6)``.
        show : bool
            Whether to automatically display the plot. Defaults to ``True``.
        legend_kwargs : dict, optional
            Keyword arguments to adjust legend placement and styling.
            Supported keys: ``loc``, ``bbox_to_anchor``, ``fontsize``,
            ``frameon``, ``title`` (``bbox_transform`` is accepted alongside
            ``bbox_to_anchor``). The existing legend is modified **in
            place** so that custom handles are preserved.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure that was created.
        ax : list[matplotlib.axes.Axes]
            A single-element list containing the event-study axes.
        """
        return self._render_plot(
            show=show,
            legend_kwargs=legend_kwargs,
            hdi_prob=hdi_prob,
            figsize=figsize,
        )

    def plot_group_time(
        self,
        *,
        hdi_prob: float | None = None,
        layout: Literal["facet", "overlay"] = "facet",
        x_axis: Literal["event_time", "calendar_time"] = "event_time",
        include_placebo: bool = True,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot cohort-specific ``ATT(g, t)`` trajectories.

        Parameters
        ----------
        hdi_prob : float, optional
            Probability mass of the highest density interval shown by the
            uncertainty bands. As with :meth:`plot`, Bayesian ``ATT(g, t)``
            bounds are cached during effect aggregation. If supplied here, the
            value must match the cached :attr:`hdi_prob_`; otherwise a
            :class:`ValueError` is raised. Pass ``None`` (the default) to plot
            using the cached value. Ignored for OLS models.
        layout : {"facet", "overlay"}
            Plot layout. ``"facet"`` draws one row per cohort and
            ``"overlay"`` draws all cohorts on a single axes. Defaults to
            ``"facet"``.
        x_axis : {"event_time", "calendar_time"}
            Time scale for the cohort trajectories. ``"event_time"`` plots
            each cohort against periods since treatment, giving an
            ``ATT(g, e)`` view derived from ``ATT(g, t)``. ``"calendar_time"``
            plots each cohort against calendar time ``t``. Defaults to
            ``"event_time"``.
        include_placebo : bool
            Whether to include pre-treatment residual estimates for
            eventually-treated cohorts as placebo diagnostics. Defaults to
            ``True``.
        figsize : tuple of (float, float), optional
            Width and height of the figure in inches, passed to
            :func:`matplotlib.pyplot.subplots`. Defaults to a height scaled by
            the number of cohorts when ``layout="facet"`` and ``(10, 6)``
            when ``layout="overlay"``.
        show : bool
            Whether to automatically display the plot. Defaults to ``True``.
        legend_kwargs : dict, optional
            Keyword arguments to adjust legend placement and styling.
            Supported keys: ``loc``, ``bbox_to_anchor``, ``fontsize``,
            ``frameon``, ``title`` (``bbox_transform`` is accepted alongside
            ``bbox_to_anchor``). The existing legend is modified **in place**
            so that custom handles are preserved.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure that was created.
        ax : list[matplotlib.axes.Axes]
            Axes containing the cohort trajectories. The list has one axes
            per cohort when ``layout="facet"`` and one axes when
            ``layout="overlay"``.
        """
        return self._render_plot(
            show=show,
            legend_kwargs=legend_kwargs,
            hdi_prob=hdi_prob,
            layout=layout,
            x_axis=x_axis,
            include_placebo=include_placebo,
            figsize=figsize,
            view="group_time",
        )

    def _plot(
        self,
        hdi_prob: float | None = None,
        figsize: tuple[float, float] | None = (10, 6),
        view: Literal["event_time", "group_time"] = "event_time",
        layout: Literal["facet", "overlay"] = "facet",
        x_axis: Literal["event_time", "calendar_time"] = "event_time",
        include_placebo: bool = True,
        **kwargs: Any,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot the event study or cohort trajectories.

        Parameters
        ----------
        hdi_prob : float, optional
            Probability mass of the highest density interval shown by the
            error bars. Unlike most other CausalPy experiments, ``hdi_prob``
            for ``StaggeredDiD`` is fixed at fit time during effect
            aggregation (see ``_aggregate_effects_bayesian``) and the
            resulting bounds are cached on the instance. If supplied here,
            the value must match the cached
            :attr:`~causalpy.experiments.staggered_did.StaggeredDiD.hdi_prob_`;
            otherwise a :class:`ValueError` is raised. Pass ``None`` (the
            default) to plot using the cached value. Ignored for
            point-estimate models.
        figsize : tuple of (float, float), optional
            Width and height of the figure in inches. Defaults to ``(10, 6)``.
        view : {"event_time", "group_time"}, optional
            Plot view to render. ``"event_time"`` draws the aggregated event
            study and ``"group_time"`` draws cohort-specific ``ATT(g, t)``
            trajectories. Defaults to ``"event_time"``.
        layout : {"facet", "overlay"}, optional
            Plot layout for the ``"group_time"`` view. Defaults to
            ``"facet"``.
        x_axis : {"event_time", "calendar_time"}, optional
            Time scale for the ``"group_time"`` view. Defaults to
            ``"event_time"``.
        include_placebo : bool, optional
            Whether to include pre-treatment residual estimates in the
            ``"group_time"`` view. Defaults to ``True``.

        Returns
        -------
        tuple[plt.Figure, list[plt.Axes]]
            Figure and axes objects.
        """
        with_uncertainty = has_posterior_draws(self.y_pred)
        if with_uncertainty and hdi_prob is not None and hdi_prob != self.hdi_prob_:
            raise ValueError(
                "StaggeredDiD HDI bounds are computed during effect "
                "aggregation, not at plot time. The cached HDI probability "
                f"is {self.hdi_prob_}, but plot() received hdi_prob="
                f"{hdi_prob}. To plot at a different HDI probability, "
                "re-fit the experiment so that aggregation uses the desired "
                "value, or omit hdi_prob to use the cached value."
            )
        if view == "group_time":
            return self._plot_group_time(
                figsize=figsize,
                layout=layout,
                x_axis=x_axis,
                include_placebo=include_placebo,
            )
        if view != "event_time":
            raise ValueError("view must be 'event_time' or 'group_time'")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        att_et = self.att_event_time_.copy()

        # Separate pre-treatment (placebo) and post-treatment (ATT)
        pre_treatment = att_et[att_et["event_time"] < 0]
        post_treatment = att_et[att_et["event_time"] >= 0]

        # Plot pre-treatment placebo estimates (different style)
        if len(pre_treatment) > 0:
            if with_uncertainty:
                ax.errorbar(
                    pre_treatment["event_time"],
                    pre_treatment["att"],
                    yerr=[
                        pre_treatment["att"] - pre_treatment["att_lower"],
                        pre_treatment["att_upper"] - pre_treatment["att"],
                    ],
                    fmt="s",  # Square markers for placebo
                    capsize=4,
                    capthick=2,
                    markersize=7,
                    color="gray",
                    alpha=0.7,
                    label=f"Placebo estimate ({int(self.hdi_prob_ * 100)}% HDI)",
                )
            else:
                ax.scatter(
                    pre_treatment["event_time"],
                    pre_treatment["att"],
                    s=60,
                    color="gray",
                    marker="s",  # Square markers for placebo
                    zorder=3,
                    alpha=0.7,
                    label="Placebo estimate",
                )
                # Add error bars if std available
                if "att_std" in pre_treatment.columns:
                    se = self._ols_error_bar_se(pre_treatment)
                    ax.errorbar(
                        pre_treatment["event_time"],
                        pre_treatment["att"],
                        yerr=1.96 * se,
                        fmt="none",
                        capsize=4,
                        capthick=2,
                        color="gray",
                        alpha=0.5,
                    )

        # Plot post-treatment ATT estimates
        if len(post_treatment) > 0:
            if with_uncertainty:
                ax.errorbar(
                    post_treatment["event_time"],
                    post_treatment["att"],
                    yerr=[
                        post_treatment["att"] - post_treatment["att_lower"],
                        post_treatment["att_upper"] - post_treatment["att"],
                    ],
                    fmt="o",
                    capsize=4,
                    capthick=2,
                    markersize=8,
                    color="C0",
                    label=f"ATT estimate ({int(self.hdi_prob_ * 100)}% HDI)",
                )
            else:
                ax.scatter(
                    post_treatment["event_time"],
                    post_treatment["att"],
                    s=80,
                    color="C0",
                    zorder=3,
                    label="ATT estimate",
                )
                # Add error bars if std available
                if "att_std" in post_treatment.columns:
                    se = self._ols_error_bar_se(post_treatment)
                    ax.errorbar(
                        post_treatment["event_time"],
                        post_treatment["att"],
                        yerr=1.96 * se,
                        fmt="none",
                        capsize=4,
                        capthick=2,
                        color="C0",
                        alpha=0.7,
                    )

        # Add horizontal line at zero
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.7)

        # Add vertical line at event_time = 0 (treatment onset)
        ax.axvline(x=-0.5, color="red", linestyle="-", linewidth=2, alpha=0.7)

        # Shade pre-treatment region
        event_min = att_et["event_time"].min()
        if event_min < 0:
            ax.axvspan(
                event_min - 0.5,
                -0.5,
                alpha=0.1,
                color="gray",
            )

        # ETWFE pins the reference event time by omitting its column, so no
        # estimate exists there. Mark it, otherwise the reader sees an
        # unexplained hole in the event study.
        drew_reference = self._draw_reference_marker(ax)

        # Labels and formatting
        ax.set_xlabel("Event Time (periods relative to treatment)", fontsize=12)
        ax.set_ylabel("Effect Estimate", fontsize=12)
        ax.set_title("Staggered DiD Event Study", fontsize=14)
        ax.legend(fontsize=LEGEND_FONT_SIZE)

        # Set integer ticks for event time, including the omitted reference
        ticks = list(att_et["event_time"].values)
        if drew_reference and self.reference_event_time not in ticks:
            ticks = sorted([*ticks, self.reference_event_time])
        ax.set_xticks(ticks)

        return fig, [ax]

    def _plot_group_time(
        self,
        figsize: tuple[float, float] | None = None,
        layout: Literal["facet", "overlay"] = "facet",
        x_axis: Literal["event_time", "calendar_time"] = "event_time",
        include_placebo: bool = True,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot cohort-time ``ATT(g, t)`` trajectories."""
        att_gt, x_col, x_label, y_label = self._get_group_time_plot_data(
            x_axis=x_axis, include_placebo=include_placebo
        )
        cohort_groups = list(att_gt.groupby("cohort", sort=True))
        sharex = x_axis == "event_time"
        fig, axes = self._make_group_time_axes(
            att_gt=att_gt,
            layout=layout,
            figsize=figsize,
            sharex=sharex,
            sharey=layout == "facet",
        )

        for cohort_idx, (cohort, cohort_data) in enumerate(cohort_groups):
            ax = axes[cohort] if layout == "facet" else axes["overlay"]
            self._plot_group_time_segment(
                ax=ax,
                cohort_data=cohort_data[cohort_data["type"] == "placebo"],
                x_col=x_col,
                line_type="placebo",
                color="gray" if layout == "facet" else f"C{cohort_idx % 10}",
                label=(
                    "Placebo estimate"
                    if layout == "facet"
                    else f"Cohort {cohort} placebo"
                ),
            )
            self._plot_group_time_segment(
                ax=ax,
                cohort_data=cohort_data[cohort_data["type"] == "ATT"],
                x_col=x_col,
                line_type="ATT",
                color="C0" if layout == "facet" else f"C{cohort_idx % 10}",
                label="ATT estimate" if layout == "facet" else f"Cohort {cohort} ATT",
            )
            self._format_group_time_axis(
                ax=ax,
                cohort=cohort if layout == "facet" else None,
                x_label=self._get_group_time_axis_label(
                    x_label=x_label,
                    layout=layout,
                    sharex=sharex,
                    axis_index=cohort_idx,
                    n_axes=len(cohort_groups),
                ),
                y_label=y_label,
                x_axis=x_axis,
                treatment_time=cohort,
            )
            ax.legend(fontsize=LEGEND_FONT_SIZE)

        if layout == "overlay":
            axes["overlay"].legend(title="Treatment cohort", fontsize=LEGEND_FONT_SIZE)

        return fig, list(axes.values())

    def _get_group_time_plot_data(
        self,
        x_axis: Literal["event_time", "calendar_time"],
        include_placebo: bool,
    ) -> tuple[pd.DataFrame, str, str, str]:
        """Return cohort-time data with the requested plotting time scale."""
        if x_axis not in {"event_time", "calendar_time"}:
            raise ValueError("x_axis must be 'event_time' or 'calendar_time'")

        att_gt = self.att_group_time_.sort_values(["cohort", "time"]).copy()
        att_gt["type"] = "ATT"
        if include_placebo:
            att_gt = pd.concat(
                [self._get_group_time_placebo_data(), att_gt],
                ignore_index=True,
                sort=False,
            ).sort_values(["cohort", "time"])

        y_label = "ATT(g, e)" if x_axis == "event_time" else "ATT(g, t)"
        if include_placebo:
            y_label = f"{y_label} / placebo"

        if x_axis == "event_time":
            att_gt["event_time"] = att_gt["time"] - att_gt["cohort"]
            return (
                att_gt,
                "event_time",
                "Event Time (periods relative to treatment)",
                y_label,
            )
        return att_gt, "time", "Calendar Time", y_label

    def _get_group_time_placebo_data(self) -> pd.DataFrame:
        """Return cohort-time placebo estimates for eventually-treated units.

        The two helpers compute genuinely different statistics: HDI bounds
        need posterior draws, sample dispersion needs only point residuals.
        """
        if has_posterior_draws(self.y_pred):
            return self._get_group_time_placebo_data_bayesian()
        return self._get_group_time_placebo_data_ols()

    def _get_group_time_placebo_observations(self) -> pd.DataFrame:
        """Return pre-treatment observations for eventually-treated units."""
        is_eventually_treated = self.data["G"] != self.never_treated_value
        is_pre_treatment = self.data["event_time"] < 0
        return self.data[is_eventually_treated & is_pre_treatment].copy()

    def _get_group_time_placebo_data_bayesian(self) -> pd.DataFrame:
        """Return Bayesian cohort-time placebo estimates with HDI bounds."""
        pretreatment_data = self._get_group_time_placebo_observations()
        if len(pretreatment_data) == 0:
            return pd.DataFrame()

        hdi_prob = getattr(self, "hdi_prob_", HDI_PROB)
        lower_pct = (1 - hdi_prob) / 2 * 100
        upper_pct = (1 + hdi_prob) / 2 * 100
        mu_draws = self.y_pred.isel(treated_units=0)
        y_observed = self._observed_outcome.to_numpy()
        tau_draws_all = y_observed - mu_draws.values

        att_gt_rows: list[dict[str, Any]] = []
        gt_groups = pretreatment_data.groupby(["G", self.time_variable_name]).groups
        for key, idx in gt_groups.items():
            g_val = key[0]  # type: ignore[index]
            t_val = key[1]  # type: ignore[index]
            positions = [np.where(self.data.index == i)[0][0] for i in idx]
            tau_gt = tau_draws_all[:, :, positions].mean(axis=2)
            att_gt_rows.append(
                {
                    "cohort": g_val,
                    "time": t_val,
                    "att": float(tau_gt.mean()),
                    "att_lower": float(np.percentile(tau_gt, lower_pct)),
                    "att_upper": float(np.percentile(tau_gt, upper_pct)),
                    "n_obs": len(positions),
                    "type": "placebo",
                }
            )
        return pd.DataFrame(att_gt_rows)

    def _get_group_time_placebo_data_ols(self) -> pd.DataFrame:
        """Return OLS cohort-time placebo residual estimates."""
        pretreatment_data = self._get_group_time_placebo_observations()
        if len(pretreatment_data) == 0:
            return pd.DataFrame()

        pretreatment_data["tau_hat"] = (
            self._observed_outcome.loc[pretreatment_data.index].to_numpy()
            - pretreatment_data["y_hat0"].to_numpy()
        )
        att_gt = (
            pretreatment_data.groupby(["G", self.time_variable_name])["tau_hat"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        att_gt.columns = ["cohort", "time", "att", "att_std", "n_obs"]
        att_gt["type"] = "placebo"
        return att_gt

    def _make_group_time_axes(
        self,
        att_gt: pd.DataFrame,
        layout: Literal["facet", "overlay"],
        figsize: tuple[float, float] | None,
        sharex: bool,
        sharey: bool,
    ) -> tuple[plt.Figure, dict[Any, plt.Axes]]:
        """Create axes for cohort trajectory plots."""
        cohorts = list(att_gt["cohort"].drop_duplicates())
        if layout == "overlay":
            fig, ax = plt.subplots(
                1, 1, figsize=figsize or (10, 6), layout="constrained"
            )
            return fig, {"overlay": ax}
        if layout != "facet":
            raise ValueError("layout must be 'facet' or 'overlay'")

        fig_height = max(2.5 * len(cohorts), 3.0)
        fig, axes_arr = plt.subplots(
            len(cohorts),
            1,
            figsize=figsize or (10, fig_height),
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
            layout="constrained",
        )
        return fig, {
            cohort: axes_arr[row_idx, 0] for row_idx, cohort in enumerate(cohorts)
        }

    def _format_group_time_axis(
        self,
        ax: plt.Axes,
        cohort: Any | None,
        x_label: str,
        y_label: str,
        x_axis: Literal["event_time", "calendar_time"],
        treatment_time: Any,
    ) -> None:
        """Apply shared formatting for cohort trajectory axes."""
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        if x_axis == "event_time":
            ax.axvline(x=-0.5, color="red", linestyle="-", linewidth=1, alpha=0.5)
        elif cohort is not None:
            ax.axvline(
                x=treatment_time - 0.5,
                color="red",
                linestyle="-",
                linewidth=1,
                alpha=0.5,
            )
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        if cohort is None:
            ax.set_title("Staggered DiD Cohort Trajectories", fontsize=14)
        else:
            ax.set_title(f"Cohort {cohort}", fontsize=12)

    def _get_group_time_axis_label(
        self,
        x_label: str,
        layout: Literal["facet", "overlay"],
        sharex: bool,
        axis_index: int,
        n_axes: int,
    ) -> str:
        """Return an x-axis label only where it helps the figure."""
        if layout == "facet" and sharex and axis_index < n_axes - 1:
            return ""
        return x_label

    def _plot_group_time_segment(
        self,
        ax: plt.Axes,
        cohort_data: pd.DataFrame,
        x_col: str,
        line_type: Literal["placebo", "ATT"],
        color: str,
        label: str,
    ) -> None:
        """Plot one placebo or ATT segment for a cohort.

        Uncertainty rendering keys on which columns the aggregation produced:
        HDI bounds (``att_lower`` / ``att_upper``) draw a shaded band, sample
        dispersion (``att_std`` / ``n_obs``) draws 1.96-SE error bars, and
        bare point estimates draw a plain line.
        """
        if len(cohort_data) == 0:
            return

        marker = "s" if line_type == "placebo" else "o"
        linestyle = "--" if line_type == "placebo" else "-"
        if {"att_lower", "att_upper"}.issubset(cohort_data.columns):
            alpha = 0.15 if line_type == "placebo" else 0.2
            ax.plot(
                cohort_data[x_col],
                cohort_data["att"],
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=label,
            )
            ax.fill_between(
                cohort_data[x_col],
                cohort_data["att_lower"],
                cohort_data["att_upper"],
                color=color,
                alpha=alpha,
            )
        elif {"att_std", "n_obs"}.issubset(cohort_data.columns):
            se = cohort_data["att_std"] / np.sqrt(cohort_data["n_obs"])
            ax.errorbar(
                cohort_data[x_col],
                cohort_data["att"],
                yerr=1.96 * se,
                fmt=f"{marker}{linestyle}",
                capsize=4,
                capthick=2,
                color=color,
                label=label,
            )
        else:
            ax.plot(
                cohort_data[x_col],
                cohort_data["att"],
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=label,
            )

    def get_plot_data(self, hdi_prob: float = HDI_PROB) -> pd.DataFrame:
        """Get event-time plotting data.

        Parameters
        ----------
        hdi_prob : float, optional
            Probability for HDI interval. Only used by models carrying
            posterior draws; when it differs from the value cached at fit
            time, the intervals are recomputed. Defaults to
            :data:`~causalpy.constants.HDI_PROB` (currently 0.94).

        Returns
        -------
        pd.DataFrame
            DataFrame with ``event_time`` and ``att`` columns plus
            ``att_lower`` / ``att_upper`` HDI bounds (posterior draws) or
            ``att_std`` / ``n_obs`` dispersion columns (point estimates).
            Includes both pre-treatment (placebo) and post-treatment effects.
        """
        # If there are no posterior draws, or the requested hdi_prob matches
        # what was used during aggregation, return the pre-computed results
        stored_hdi_prob = getattr(self, "hdi_prob_", HDI_PROB)
        if not has_posterior_draws(self.y_pred) or np.isclose(
            hdi_prob, stored_hdi_prob
        ):
            return self.att_event_time_.copy()

        if self.estimator == "etwfe":
            # ETWFE aggregates the posterior effect surface directly; there are no
            # differenced posterior predictive draws to recompute from.
            tau_draws = self._etwfe_idata["posterior"]["tau"].values
            return self._apply_event_window(
                self._etwfe_att_event_time_bayesian(
                    tau_draws,
                    (1 - hdi_prob) / 2 * 100,
                    (1 + hdi_prob) / 2 * 100,
                )
            )

        # Recompute intervals with the requested hdi_prob
        lower_pct = (1 - hdi_prob) / 2 * 100
        upper_pct = (1 + hdi_prob) / 2 * 100

        # Get posterior draws for mu
        mu_draws = self.y_pred.isel(treated_units=0)

        # Get observed y for all observations
        y_observed = self._observed_outcome.to_numpy()

        # Compute tau draws for all observations
        tau_draws_all = y_observed - mu_draws.values

        att_et_rows: list[dict] = []

        # Pre-treatment placebo effects (eventually-treated units, event_time < 0)
        is_eventually_treated = self.data["G"] != self.never_treated_value
        is_pre_treatment = self.data["event_time"] < 0
        pretreatment_data = self.data[is_eventually_treated & is_pre_treatment].copy()

        if len(pretreatment_data) > 0:
            pretreat_indices = pretreatment_data.index.values
            pretreat_idx_positions = np.array(
                [np.where(self.data.index == idx)[0][0] for idx in pretreat_indices]
            )
            tau_draws_pretreat = tau_draws_all[:, :, pretreat_idx_positions]
            event_time_pretreat = np.asarray(pretreatment_data["event_time"].values)

            event_times_pre = np.unique(
                event_time_pretreat[~np.isnan(event_time_pretreat)]
            )
            if self.event_window is not None:
                event_times_pre = event_times_pre[
                    (event_times_pre >= self.event_window[0])
                    & (event_times_pre <= self.event_window[1])
                ]

            for e in sorted(event_times_pre):
                e_mask = event_time_pretreat == e
                if e_mask.sum() == 0:
                    continue
                positions_arr = np.where(e_mask)[0]
                tau_e = tau_draws_pretreat[:, :, positions_arr].mean(axis=2)
                att_et_rows.append(
                    {
                        "event_time": int(e),
                        "att": float(tau_e.mean()),
                        "att_lower": float(np.percentile(tau_e, lower_pct)),
                        "att_upper": float(np.percentile(tau_e, upper_pct)),
                        "n_obs": int(e_mask.sum()),
                    }
                )

        # Post-treatment effects (treated observations, event_time >= 0)
        _is_untreated = np.asarray(self.data["_is_untreated"].values, dtype=bool)
        treated_mask = ~_is_untreated
        treated_indices = np.where(treated_mask)[0]
        tau_draws_treated = tau_draws_all[:, :, treated_indices]

        treated_data = self.data[~self.data["_is_untreated"]].copy()
        event_time_treated = np.asarray(treated_data["event_time"].values)

        event_times_post = np.unique(event_time_treated[~np.isnan(event_time_treated)])
        if self.event_window is not None:
            event_times_post = event_times_post[
                (event_times_post >= self.event_window[0])
                & (event_times_post <= self.event_window[1])
            ]

        for e in sorted(event_times_post):
            e_mask = event_time_treated == e
            if e_mask.sum() == 0:
                continue
            positions_arr = np.where(e_mask)[0]
            tau_e = tau_draws_treated[:, :, positions_arr].mean(axis=2)
            att_et_rows.append(
                {
                    "event_time": int(e),
                    "att": float(tau_e.mean()),
                    "att_lower": float(np.percentile(tau_e, lower_pct)),
                    "att_upper": float(np.percentile(tau_e, upper_pct)),
                    "n_obs": int(e_mask.sum()),
                }
            )

        return self._mark_non_identified_att_rows(pd.DataFrame(att_et_rows))

    def effect_summary(
        self,
        *,
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        min_effect: float | None = None,
        **kwargs: Any,
    ) -> EffectSummary:
        """
        Generate a decision-ready summary of causal effects for Staggered Difference-in-Differences.

        Parameters
        ----------
        direction : {"increase", "decrease", "two-sided"}, default="increase"
            Direction for tail probability calculation (PyMC only, ignored for OLS).
        alpha : float, default=0.05
            Significance level for HDI/CI intervals (1-alpha confidence level).
        min_effect : float, optional
            Region of Practical Equivalence (ROPE) threshold (PyMC only, ignored for OLS).
        **kwargs
            Reserved for forward-compatibility; not consumed by this
            implementation.

        Returns
        -------
        EffectSummary
            Object with .table (DataFrame) and .text (str) attributes
        """
        from causalpy.reporting import _effect_summary_staggered_did

        return _effect_summary_staggered_did(
            self,
            direction=direction,
            alpha=alpha,
            min_effect=min_effect,
        )


# ---------------------------------------------------------------------------
# ETWFE (Wooldridge / Mundlak) index construction
#
# These are deliberately module-level *pure* functions rather than methods, so
# that the indexing logic -- which is the load-bearing part of the extended
# two-way fixed effects estimator -- can be unit tested without constructing an
# experiment object.
# ---------------------------------------------------------------------------

#: Sentinel written into ``_ETWFEIndex.k_eff`` for never-treated observations,
#: for which the event time ``t - G`` is undefined. It is never used as an index.
_ETWFE_K_SENTINEL: int = int(np.iinfo(np.int64).min)

#: Name of the temporary categorical column carrying the ``(cohort, event time)``
#: cell each observation loads on, used by the OLS ETWFE patsy design.
_GK_CELL: str = "_gk_cell"

#: Level of :data:`_GK_CELL` given to out-of-scope observations. It is the patsy
#: reference level, so every cell coefficient reads against the untreated baseline.
_GK_NONE: str = "__none__"

#: Cells with fewer than this many observations are reported by
#: :meth:`StaggeredDifferenceInDifferences._check_etwfe_identification`.
_ETWFE_MIN_CELL_COUNT: int = 5

#: Maximum number of offending cells named in an identification warning, so the
#: message stays readable on a large panel.
_ETWFE_MAX_REPORTED_CELLS: int = 5

#: Share of empty ``(cohort, event time)`` cells above which the effect surface is
#: reported as mostly holes. Sparse corners are the *normal* shape of a staggered
#: panel -- the last-adopting cohort never reaches the largest event times -- so
#: this threshold is deliberately permissive.
_ETWFE_EMPTY_CELL_SHARE: float = 0.5

#: R-hat above which the in-model ATT is reported as possibly unconverged.
_ETWFE_RHAT_THRESHOLD: float = 1.01


def _format_cohort(cohort: Any) -> str:
    """Render a cohort label compactly, without a spurious ``.0`` suffix.

    Parameters
    ----------
    cohort : Any
        The adoption cohort value.

    Returns
    -------
    str
        A short string form of the cohort.

    Examples
    --------
    >>> _format_cohort(4.0)
    '4'
    >>> _format_cohort(2.5)
    '2.5'
    """
    try:
        as_float = float(cohort)
    except (TypeError, ValueError):
        return str(cohort)
    if as_float.is_integer():
        return str(int(as_float))
    return str(as_float)


def _as_calendar_time(cohort: Any, event_time: int) -> Any:
    """Convert ``(cohort, event time)`` to the calendar time ``G + k``.

    Parameters
    ----------
    cohort : Any
        Adoption time ``G``.
    event_time : int
        Event time ``k``.

    Returns
    -------
    Any
        ``G + k``, narrowed to ``int`` when the result is integral, so the
        ``time`` column matches the panel's own integer time labels.

    Examples
    --------
    >>> _as_calendar_time(4.0, 2)
    6
    """
    value = cohort + event_time
    try:
        as_float = float(value)
    except (TypeError, ValueError):  # pragma: no cover - exotic time labels
        return value
    return int(as_float) if as_float.is_integer() else value


def _etwfe_cell_labels(index: "_ETWFEIndex") -> tuple[np.ndarray, dict[str, tuple]]:
    """Build the ``_gk_cell`` categorical column and its inverse mapping.

    Parameters
    ----------
    index : _ETWFEIndex
        Index bundle from :func:`_build_etwfe_index`.

    Returns
    -------
    tuple
        ``(labels, label_to_cell)`` where ``labels`` is an object array of length
        ``n_obs`` holding the categorical level of each observation, and
        ``label_to_cell`` maps each in-scope level back to its
        ``(cohort position, event-time position)`` pair.

    Raises
    ------
    ValueError
        If two distinct cells would render to the same label. This can only
        happen with pathological cohort values and indicates that the labels
        cannot be used as categorical levels.
    """
    n_obs = index.effect_indicator.shape[0]
    labels = np.full(n_obs, _GK_NONE, dtype=object)
    in_scope = index.effect_indicator == 1.0
    cohort_strings = [_format_cohort(g) for g in index.cohorts]

    label_to_cell: dict[str, tuple] = {}
    for gi, cohort_string in enumerate(cohort_strings):
        for ei, event_time in enumerate(index.ev_grid):
            label = f"g{cohort_string}_k{int(event_time)}"
            if label in label_to_cell or label == _GK_NONE:
                raise ValueError(
                    f"Cohort/event-time labels are not unique: {label!r} would "
                    "name more than one cell of the ETWFE effect surface."
                )
            label_to_cell[label] = (gi, ei)

    if np.any(in_scope):
        scoped = np.array(
            [
                f"g{cohort_strings[int(gi)]}_k{int(index.ev_grid[int(ei)])}"
                for gi, ei in zip(
                    index.cohort_idx[in_scope], index.ev_idx[in_scope], strict=True
                )
            ],
            dtype=object,
        )
        labels[in_scope] = scoped
    return labels, label_to_cell


@dataclass(frozen=True)
class _ETWFEIndex:
    """Index bundle describing the ETWFE (extended two-way fixed effects) design.

    Every array is observation-aligned with the ``data`` frame it was built from
    (same length, same row order). The integer index arrays are **always
    non-negative**: out-of-scope observations are given index ``0`` and are
    neutralised by ``effect_indicator == 0`` rather than by a negative sentinel.
    This matters because with lead terms, ``-1`` is a legitimate event time and a
    ``-1`` sentinel would silently alias onto a real column.

    Attributes
    ----------
    unit_idx : np.ndarray
        Integer array of shape ``(n_obs,)`` giving the position of each
        observation's unit within ``unit_levels``.
    time_idx : np.ndarray
        Integer array of shape ``(n_obs,)`` giving the position of each
        observation's time period within ``time_levels``.
    cohort_idx : np.ndarray
        Integer array of shape ``(n_obs,)`` giving the position of each
        observation's adoption cohort within ``cohorts``. Never-treated
        observations get ``0`` (they are neutralised by ``effect_indicator``).
    ev_idx : np.ndarray
        Integer array of shape ``(n_obs,)`` giving the column of the effect
        surface each observation loads on. Out-of-scope observations get ``0``.
    effect_indicator : np.ndarray
        Float array of shape ``(n_obs,)`` with values in ``{0.0, 1.0}``. Equals
        the treated indicator when ``n_leads == 0``, and additionally switches on
        for estimated lead (pre-treatment) cells otherwise.
    k_eff : np.ndarray
        Integer array of shape ``(n_obs,)`` with the top-binned event time
        ``min(t - G, k_max_est)`` for treated cells and the raw event time
        ``t - G`` elsewhere. Never-treated observations carry
        ``_ETWFE_K_SENTINEL``.
    unit_levels : np.ndarray
        Sorted unique unit labels.
    time_levels : np.ndarray
        Sorted unique time labels.
    cohorts : list
        Sorted adopting cohorts. Never-treated units are excluded.
    ev_grid : np.ndarray
        Integer array of the event times actually estimated, in ascending order.
        The reference event time is **omitted** (the effect there is normalised
        to zero by leaving the column out of the design, not by masking it).
    att_weights : np.ndarray
        Float array of shape ``(n_cohorts, n_ev)`` holding
        ``w_gk = N_gk / sum(N_gk)`` computed from **treated cells only**, so lead
        columns are exactly zero. Sums to 1.
    cell_counts : np.ndarray
        Integer array of shape ``(n_cohorts, n_ev)`` counting **all** in-scope
        observations per cell, so unlike ``att_weights`` it is non-zero in lead
        columns.
    dropped_cells : list
        List of ``(cohort, event_time)`` pairs with no in-scope observations.
        Individual empty cells are retained in the effect surface (under partial
        pooling they simply sample from the shared profile); event-time columns
        that are empty for *every* cohort are removed from ``ev_grid``. All empty
        cells are recorded here regardless, for downstream reporting.
    """

    unit_idx: np.ndarray
    time_idx: np.ndarray
    cohort_idx: np.ndarray
    ev_idx: np.ndarray
    effect_indicator: np.ndarray
    k_eff: np.ndarray
    unit_levels: np.ndarray
    time_levels: np.ndarray
    cohorts: list
    ev_grid: np.ndarray
    att_weights: np.ndarray
    cell_counts: np.ndarray
    dropped_cells: list


def _never_treated_mask(g_values: pd.Series, never_treated_value: Any) -> np.ndarray:
    """Boolean mask flagging never-treated observations.

    Parameters
    ----------
    g_values : pd.Series
        Unit-level treatment times, one entry per observation.
    never_treated_value : Any
        Sentinel value marking never-treated units. NaN is handled explicitly
        because ``NaN != NaN``.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(n_obs,)``, True for never-treated rows.
    """
    if pd.isna(never_treated_value):
        return np.asarray(pd.isna(g_values))
    return np.asarray(g_values == never_treated_value)


def _ev_positions(
    ev_grid: np.ndarray, k_eff: np.ndarray, in_scope: np.ndarray
) -> np.ndarray:
    """Map effective event times onto their column position in ``ev_grid``.

    Parameters
    ----------
    ev_grid : np.ndarray
        Ascending integer array of estimated event times.
    k_eff : np.ndarray
        Effective (top-binned) event time per observation.
    in_scope : np.ndarray
        Boolean mask of observations that load on the effect surface.

    Returns
    -------
    np.ndarray
        Non-negative integer array of shape ``(n_obs,)``. Out-of-scope
        observations get ``0``.

    Raises
    ------
    ValueError
        If an in-scope observation's effective event time is absent from
        ``ev_grid``. This indicates an internal inconsistency, not user error.
    """
    positions = np.zeros(k_eff.shape[0], dtype=np.int64)
    if not np.any(in_scope):
        return positions
    k_in = k_eff[in_scope]
    if ev_grid.size == 0:
        raise ValueError(
            "Internal error building the ETWFE index: the event-time grid is "
            "empty but some observations are in scope."
        )
    found = np.clip(np.searchsorted(ev_grid, k_in), 0, ev_grid.size - 1)
    matched = ev_grid[found] == k_in
    if not np.all(matched):
        missing = sorted({int(v) for v in k_in[~matched]})
        raise ValueError(
            f"Internal error building the ETWFE index: effective event times "
            f"{missing} are in scope but absent from the event-time grid."
        )
    positions[in_scope] = found
    return positions


def _count_cells(
    cohort_idx: np.ndarray,
    ev_idx: np.ndarray,
    mask: np.ndarray,
    n_cohorts: int,
    n_ev: int,
) -> np.ndarray:
    """Count masked observations falling in each ``(cohort, event time)`` cell.

    Parameters
    ----------
    cohort_idx : np.ndarray
        Non-negative cohort positions, one per observation.
    ev_idx : np.ndarray
        Non-negative event-time positions, one per observation.
    mask : np.ndarray
        Boolean mask selecting the observations to count.
    n_cohorts : int
        Number of adopting cohorts (rows of the returned matrix).
    n_ev : int
        Number of estimated event times (columns of the returned matrix).

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_cohorts, n_ev)``.
    """
    counts = np.zeros((n_cohorts, n_ev), dtype=np.int64)
    if np.any(mask):
        np.add.at(counts, (cohort_idx[mask], ev_idx[mask]), 1)
    return counts


def _build_etwfe_index(
    data: pd.DataFrame,
    *,
    unit_variable_name: str,
    time_variable_name: str,
    treated_variable_name: str,
    never_treated_value: Any,
    cohorts: list,
    n_leads: int = 0,
    reference_event_time: int = -1,
    max_event_time: int | None = None,
) -> _ETWFEIndex:
    """Build the observation-level index arrays for the ETWFE design.

    The estimator saturates the treatment effect over cohort ``g`` and event time
    ``k = t - G``, so every observation needs to know (i) which unit and period it
    belongs to, (ii) which cell of the ``tau[g, k]`` surface it loads on, and
    (iii) whether it loads on that surface at all. This function computes all of
    that from a panel that already carries the ``"G"`` column produced by
    :meth:`StaggeredDifferenceInDifferences._compute_treatment_times`.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data. Must already contain a ``"G"`` column (unit-level treatment
        time, ``never_treated_value`` for never-treated units) and the 0/1
        treated column. ``G`` is **not** recomputed here.
    unit_variable_name : str
        Name of the unit column.
    time_variable_name : str
        Name of the time column.
    treated_variable_name : str
        Name of the 0/1 treatment status column.
    never_treated_value : Any
        Value of ``G`` marking never-treated units (typically ``np.inf``).
    cohorts : list
        Adopting cohorts. Sorted and de-duplicated internally; any entry equal to
        ``never_treated_value`` is dropped.
    n_leads : int, default=0
        Number of pre-treatment lead terms to estimate. With ``n_leads == 0`` the
        effect indicator reduces exactly to the treated column.
    reference_event_time : int, default=-1
        Event time normalised to zero by omission from ``ev_grid``. Must satisfy
        ``-n_leads <= reference_event_time <= -1`` when ``n_leads > 0``, and must
        be ``-1`` when ``n_leads == 0``.
    max_event_time : int, optional
        Largest event time given its own column. Treated observations beyond it
        are **top-binned** into that column and keep ``effect_indicator == 1``;
        turning them into controls would leak their real effect into the unit and
        time effects. If None, the largest observed treated event time is used.

    Returns
    -------
    _ETWFEIndex
        Frozen bundle of index arrays, the event-time grid, the ATT weight
        matrix and cell diagnostics.

    Raises
    ------
    ValueError
        If ``n_leads`` is negative, ``reference_event_time`` is out of range,
        ``max_event_time`` is negative, the ``"G"`` column is missing, there are
        no treated observations, or an eventually-treated unit's ``G`` is absent
        from ``cohorts``.

    Warns
    -----
    UserWarning
        If any ``(cohort, event time)`` cell has no in-scope observations.

    Notes
    -----
    The scheme is *non-negative indices plus a separate effect indicator*::

        eventually_treated = G != never_treated_value
        k                  = t - G
        is_lead_cell       = eventually_treated & (-n_leads <= k <= -2)
                                                & (k != reference_event_time)
        is_treated_cell    = treated == 1
        in_scope           = is_lead_cell | is_treated_cell
        effect_indicator   = in_scope.astype(float)
        k_eff              = where(is_treated_cell, minimum(k, k_max_est), k)
        ev_idx             = where(in_scope, pos_map[k_eff], 0)
        cohort_idx         = where(eventually_treated, cohort_map[G], 0)

    Lead-side cells with ``k < -n_leads``, and cells at the reference event time,
    are *not* binned: they are genuine not-yet-treated controls and supply the
    identifying variation.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "unit": [0, 0, 0, 1, 1, 1],
    ...         "time": [0, 1, 2, 0, 1, 2],
    ...         "treated": [0, 1, 1, 0, 0, 0],
    ...         "G": [1, 1, 1, np.inf, np.inf, np.inf],
    ...     }
    ... )
    >>> idx = _build_etwfe_index(
    ...     df,
    ...     unit_variable_name="unit",
    ...     time_variable_name="time",
    ...     treated_variable_name="treated",
    ...     never_treated_value=np.inf,
    ...     cohorts=[1],
    ... )
    >>> idx.ev_grid
    array([0, 1])
    >>> idx.effect_indicator
    array([0., 1., 1., 0., 0., 0.])
    >>> float(idx.att_weights.sum())
    1.0
    """
    # ---- validation -------------------------------------------------------
    if int(n_leads) != n_leads or n_leads < 0:
        raise ValueError(f"n_leads must be a non-negative integer, got {n_leads!r}.")
    n_leads = int(n_leads)

    if n_leads == 0:
        if reference_event_time != -1:
            raise ValueError(
                "reference_event_time must be -1 when n_leads == 0, got "
                f"{reference_event_time!r}."
            )
    elif not (-n_leads <= reference_event_time <= -1):
        raise ValueError(
            f"reference_event_time must satisfy -n_leads <= ref <= -1 "
            f"(i.e. between {-n_leads} and -1) when n_leads={n_leads}, got "
            f"{reference_event_time!r}."
        )
    reference_event_time = int(reference_event_time)

    if max_event_time is not None and max_event_time < 0:
        raise ValueError(
            f"max_event_time must be non-negative, got {max_event_time!r}."
        )

    if "G" not in data.columns:
        raise ValueError(
            "data must already contain the 'G' column (unit-level treatment "
            "time). Call _compute_treatment_times() before building the index."
        )

    # ---- unit / time positions -------------------------------------------
    unit_levels = np.unique(data[unit_variable_name].to_numpy())
    time_levels = np.unique(data[time_variable_name].to_numpy())
    unit_idx = np.searchsorted(unit_levels, data[unit_variable_name].to_numpy()).astype(
        np.int64
    )
    time_idx = np.searchsorted(time_levels, data[time_variable_name].to_numpy()).astype(
        np.int64
    )

    # ---- cohorts ----------------------------------------------------------
    if pd.isna(never_treated_value):
        cohort_list = sorted({g for g in cohorts if not pd.isna(g)})
    else:
        cohort_list = sorted({g for g in cohorts if g != never_treated_value})
    n_cohorts = len(cohort_list)
    if n_cohorts == 0:
        raise ValueError("No adopting cohorts supplied; nothing to estimate.")
    cohort_map = {g: i for i, g in enumerate(cohort_list)}

    # ---- event times ------------------------------------------------------
    n_obs = len(data)
    g_series = data["G"]
    never_mask = _never_treated_mask(g_series, never_treated_value)
    eventually_treated = ~never_mask

    k = np.zeros(n_obs, dtype=np.int64)
    if np.any(eventually_treated):
        g_num = pd.to_numeric(g_series[eventually_treated]).to_numpy(dtype=float)
        t_num = pd.to_numeric(data[time_variable_name][eventually_treated]).to_numpy(
            dtype=float
        )
        k[eventually_treated] = np.rint(t_num - g_num).astype(np.int64)

    is_treated_cell = np.asarray(data[treated_variable_name]).astype(int) == 1
    if not np.any(is_treated_cell):
        raise ValueError(
            "No treated observations found; the ETWFE effect surface is empty."
        )

    k_max_obs = int(k[is_treated_cell].max())
    k_max_est = k_max_obs if max_event_time is None else min(k_max_obs, max_event_time)

    # ---- effect indicator and top-binning ---------------------------------
    # A lead cell is any eventually-treated pre-treatment cell inside the lead
    # window that is not the omitted reference. Using ``k <= -1`` rather than
    # ``k <= -2`` makes the lead window the exact complement of
    # ``reference_event_time`` within [-n_leads, -1], so it stays correct when
    # the reference is not the default -1. With ref == -1 the two are identical.
    is_lead_cell = (
        eventually_treated & (k >= -n_leads) & (k <= -1) & (k != reference_event_time)
    )
    in_scope = is_lead_cell | is_treated_cell
    effect_indicator = in_scope.astype(float)

    k_eff = np.where(is_treated_cell, np.minimum(k, k_max_est), k)
    k_eff = np.where(never_mask, _ETWFE_K_SENTINEL, k_eff).astype(np.int64)

    # ---- cohort positions -------------------------------------------------
    unknown = [
        g for g in pd.unique(g_series[eventually_treated]) if g not in cohort_map
    ]
    if unknown:
        raise ValueError(
            f"Treatment times {sorted(unknown)} appear in the data but not in "
            f"the supplied cohorts {cohort_list}."
        )
    mapped = g_series.map(cohort_map).fillna(0).to_numpy(dtype=np.int64)
    cohort_idx = np.where(eventually_treated, mapped, 0).astype(np.int64)

    # ---- event-time grid, indices, counts ---------------------------------
    ev_grid = np.array(
        [j for j in range(-n_leads, k_max_est + 1) if j != reference_event_time],
        dtype=np.int64,
    )
    ev_idx = _ev_positions(ev_grid, k_eff, in_scope)
    cell_counts = _count_cells(cohort_idx, ev_idx, in_scope, n_cohorts, ev_grid.size)

    # ---- empty cells ------------------------------------------------------
    empty_rows, empty_cols = np.where(cell_counts == 0)
    dropped_cells = [
        (cohort_list[int(i)], int(ev_grid[int(j)]))
        for i, j in zip(empty_rows, empty_cols, strict=True)
    ]

    fully_empty = np.flatnonzero(cell_counts.sum(axis=0) == 0)
    removed_event_times = [int(j) for j in ev_grid[fully_empty]]
    if fully_empty.size:
        keep = np.setdiff1d(np.arange(ev_grid.size), fully_empty)
        ev_grid = ev_grid[keep]
        ev_idx = _ev_positions(ev_grid, k_eff, in_scope)
        cell_counts = _count_cells(
            cohort_idx, ev_idx, in_scope, n_cohorts, ev_grid.size
        )

    # Individually empty (cohort, event time) cells are the normal shape of a
    # staggered panel -- the last-adopting cohort simply never reaches the
    # largest event times -- so they are recorded in ``dropped_cells`` for
    # downstream reporting but are NOT warned about here. Warning on them would
    # fire on essentially every real panel. Only an event time that is empty for
    # *every* cohort is actionable, because the grid changes shape as a result.
    if removed_event_times:
        warnings.warn(
            f"Event time(s) {removed_event_times} have no observations for any "
            "cohort and have been removed from the ETWFE event-time grid.",
            UserWarning,
            stacklevel=2,
        )

    # ---- ATT weights (treated cells only) ---------------------------------
    treated_counts = _count_cells(
        cohort_idx, ev_idx, is_treated_cell, n_cohorts, ev_grid.size
    )
    att_weights = treated_counts.astype(float) / float(treated_counts.sum())

    return _ETWFEIndex(
        unit_idx=unit_idx,
        time_idx=time_idx,
        cohort_idx=cohort_idx,
        ev_idx=ev_idx,
        effect_indicator=effect_indicator,
        k_eff=k_eff,
        unit_levels=unit_levels,
        time_levels=time_levels,
        cohorts=cohort_list,
        ev_grid=ev_grid,
        att_weights=att_weights,
        cell_counts=cell_counts,
        dropped_cells=dropped_cells,
    )


def _mundlak_means(
    data: pd.DataFrame,
    *,
    unit_variable_name: str,
    time_variable_name: str,
    treated_variable_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute unit-level and time-level Mundlak treatment means.

    The Mundlak device replaces the wall of unit dummies with the unit's average
    treatment exposure (and symmetrically for time periods), which is what allows
    the ETWFE model to use partially pooled intercepts instead of one free
    parameter per unit.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data containing the unit, time and treated columns.
    unit_variable_name : str
        Name of the unit column.
    time_variable_name : str
        Name of the time column.
    treated_variable_name : str
        Name of the 0/1 treatment status column.

    Returns
    -------
    tuple of np.ndarray
        ``(dbar_unit_centred, dbar_time_centred, dbar_unit_raw, dbar_time_raw)``,
        each of shape ``(n_obs,)`` and aligned with ``data``.

    Notes
    -----
    The centred versions subtract the sample mean of the corresponding raw array.
    Centring orthogonalises the Mundlak coefficients against the intercept, which
    improves posterior geometry without changing the estimand. The raw means are
    returned as well so they can be stored on the experiment for inspection.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "unit": [0, 0, 1, 1],
    ...         "time": [0, 1, 0, 1],
    ...         "treated": [0, 1, 0, 0],
    ...     }
    ... )
    >>> _, _, dbar_unit_raw, dbar_time_raw = _mundlak_means(
    ...     df,
    ...     unit_variable_name="unit",
    ...     time_variable_name="time",
    ...     treated_variable_name="treated",
    ... )
    >>> dbar_unit_raw
    array([0.5, 0.5, 0. , 0. ])
    >>> dbar_time_raw
    array([0. , 0.5, 0. , 0.5])
    """
    dbar_unit_raw = (
        data.groupby(unit_variable_name)[treated_variable_name]
        .transform("mean")
        .to_numpy(dtype=float)
    )
    dbar_time_raw = (
        data.groupby(time_variable_name)[treated_variable_name]
        .transform("mean")
        .to_numpy(dtype=float)
    )
    dbar_unit_centred = dbar_unit_raw - dbar_unit_raw.mean()
    dbar_time_centred = dbar_time_raw - dbar_time_raw.mean()
    return dbar_unit_centred, dbar_time_centred, dbar_unit_raw, dbar_time_raw
