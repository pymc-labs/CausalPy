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
"""Instrumental variable regression."""

import warnings  # noqa: I001

import numpy as np
import pandas as pd
from causalpy.experiments._design import build_patsy_design
from causalpy.experiments._results import PriorCalibration
from patsy import dmatrices
from sklearn.linear_model import LinearRegression as sk_lin_reg

from causalpy.custom_exceptions import DataException
from causalpy.pymc_models import InstrumentalVariableRegression

from .base import BaseExperiment
from causalpy.reporting import EffectSummary
from typing import Any, Literal


class InstrumentalVariable(BaseExperiment):
    """A class to analyse instrumental variable style experiments.

    Parameters
    ----------
    instruments_data : pd.DataFrame
        A pandas dataframe of instruments for our treatment variable.
        Should contain instruments Z, and treatment t.
    data : pd.DataFrame
        A pandas dataframe of covariates for fitting the focal regression
        of interest. Should contain covariates X including treatment t and
        outcome y.
    instruments_formula : str
        A statistical model formula for the instrumental stage regression,
        e.g. ``t ~ 1 + z1 + z2 + z3``.
    formula : str
        A statistical model formula for the focal regression,
        e.g. ``y ~ 1 + t + x1 + x2 + x3``.
    model : InstrumentalVariableRegression, optional
        A PyMC model. Defaults to InstrumentalVariableRegression.
    priors : dict, optional
        Dictionary of priors for the mus and sigmas of both regressions.
        If priors are not specified we will substitute MLE estimates for
        the beta coefficients. Example: ``priors = {"mus": [0, 0],
        "sigmas": [1, 1], "eta": 2, "lkj_sd": 2}``.
    vs_prior_type : str or None, default=None
        Type of variable selection prior: 'spike_and_slab', 'horseshoe', or None.
        If None, uses standard normal priors.
    vs_hyperparams : dict, optional
        Hyperparameters for variable selection priors. Only used if vs_prior_type
        is not None.
    binary_treatment : bool, default=False
        A indicator for whether the treatment to be modelled is binary or not.
        Determines which PyMC model we use to model the joint outcome and
        treatment.
    **kwargs
        Additional keyword arguments forwarded to :class:`BaseExperiment`.

    Example
    --------
    >>> import pandas as pd
    >>> import causalpy as cp
    >>> from causalpy.pymc_models import InstrumentalVariableRegression
    >>> import numpy as np
    >>> N = 100
    >>> e1 = np.random.normal(0, 3, N)
    >>> e2 = np.random.normal(0, 1, N)
    >>> Z = np.random.uniform(0, 1, N)
    >>> ## Ensure the endogeneity of the the treatment variable
    >>> X = -1 + 4 * Z + e2 + 2 * e1
    >>> y = 2 + 3 * X + 3 * e1
    >>> test_data = pd.DataFrame({"y": y, "X": X, "Z": Z})
    >>> sample_kwargs = {
    ...     "tune": 1,
    ...     "draws": 5,
    ...     "chains": 1,
    ...     "cores": 4,
    ...     "target_accept": 0.95,
    ...     "progressbar": False,
    ... }
    >>> instruments_formula = "X  ~ 1 + Z"
    >>> formula = "y ~  1 + X"
    >>> instruments_data = test_data[["X", "Z"]]
    >>> data = test_data[["y", "X"]]
    >>> iv = cp.InstrumentalVariable(
    ...     instruments_data=instruments_data,
    ...     data=data,
    ...     instruments_formula=instruments_formula,
    ...     formula=formula,
    ...     model=InstrumentalVariableRegression(sample_kwargs=sample_kwargs),
    ... )
    >>> # With variable selection
    >>> iv = cp.InstrumentalVariable(
    ...     instruments_data=instruments_data,
    ...     data=data,
    ...     instruments_formula=instruments_formula,
    ...     formula=formula,
    ...     model=InstrumentalVariableRegression(sample_kwargs=sample_kwargs),
    ...     vs_prior_type="spike_and_slab",
    ...     vs_hyperparams={"slab_sigma": 5.0},
    ... )
    """

    supports_ols = False
    supports_bayes = True
    _default_model_class = InstrumentalVariableRegression

    def __init__(
        self,
        instruments_data: pd.DataFrame,
        data: pd.DataFrame,
        instruments_formula: str,
        formula: str,
        model: InstrumentalVariableRegression | None = None,
        priors: dict | None = None,
        vs_prior_type=None,
        vs_hyperparams=None,
        binary_treatment=False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model)
        self.expt_type = "Instrumental Variable Regression"
        self.data = data
        self.instruments_data = instruments_data
        self.formula = formula
        self.instruments_formula = instruments_formula
        self.vs_prior_type = vs_prior_type
        self.vs_hyperparams = vs_hyperparams or {}
        self.binary_treatment = binary_treatment
        self.use_vs_prior_outcome = self.vs_hyperparams.get("outcome", False)
        self.input_validation()
        self._build_design_matrices()

        # Store user-provided priors (will set defaults in algorithm() if None)
        self.priors = priors
        self._prior_calibration: PriorCalibration | None = None

        self.algorithm()

    def _build_design_matrices(self) -> None:
        """Build design matrices for outcome and instrument formulas."""
        self._design, self.X, self.y = build_patsy_design(self.formula, self.data)
        self.labels = self._design.labels
        self.outcome_variable_name = self._design.outcome_name

        self._instrument_design, self.Z, self.t = build_patsy_design(
            self.instruments_formula, self.instruments_data
        )
        self.labels_instruments = self._instrument_design.labels
        self.instrument_variable_name = self._instrument_design.outcome_name

    def algorithm(self) -> None:
        """Run the experiment algorithm: fit OLS, 2SLS, and Bayesian IV model."""
        self.get_naive_OLS_fit()
        self.get_2SLS_fit()
        prior_calibration = self._prior_calibration
        assert prior_calibration is not None  # populated by get_*_fit above

        # fit the model to the data
        COORDS = {"instruments": self.labels_instruments, "covariates": self.labels}
        self.coords = COORDS
        # Only set default priors if user didn't provide custom priors
        if self.priors is None:
            if self.binary_treatment:
                # Different default priors for binary treatment
                self.priors = {
                    "mus": [
                        prior_calibration.beta_first,
                        prior_calibration.beta_second,
                    ],
                    "sigmas": [1, 1],
                    "sigma_U": 1.0,
                    "rho_bounds": [-0.99, 0.99],
                }
            else:
                # Original continuous treatment priors
                self.priors = {
                    "mus": [
                        prior_calibration.beta_first,
                        prior_calibration.beta_second,
                    ],
                    "sigmas": [1, 1],
                    "eta": 2,
                    "lkj_sd": 1,
                }
        self.model.fit(  # type: ignore[call-arg,union-attr]
            X=self.X,
            Z=self.Z,
            y=self.y,
            t=self.t,
            coords=COORDS,
            priors=self.priors,
            vs_prior_type=self.vs_prior_type,
            vs_hyperparams=self.vs_hyperparams,
            binary_treatment=self.binary_treatment,
        )

    def input_validation(self) -> None:
        """Validate the input data and model formula for correctness."""
        treatment = self.instruments_formula.split("~")[0]
        test = treatment.strip() in self.instruments_data.columns
        test = test & (treatment.strip() in self.data.columns)
        if not test:
            raise DataException(
                f"""
                The treatment variable:
                {treatment} must appear in the instrument_data to be used
                as an outcome variable and in the data object to be used as a covariate.
                """
            )
        Z = self.data[treatment.strip()]
        check_binary = len(np.unique(Z)) > 2
        if check_binary:
            warnings.warn(
                """Warning. The treatment variable is not Binary.
                We will use the multivariate normal likelihood
                for continuous treatment.""",
                stacklevel=2,
            )

    def get_2SLS_fit(self) -> None:
        """Two Stage Least Squares Fit.

        This function is called by the experiment, results are used for
        priors if none are provided.
        """
        first_stage_reg = sk_lin_reg().fit(self.Z, self.t)
        fitted_Z_values = first_stage_reg.predict(self.Z)
        X2 = self.data.copy(deep=True)
        X2[self.instrument_variable_name] = fitted_Z_values
        _, X2 = dmatrices(self.formula, X2)
        second_stage_reg = sk_lin_reg().fit(X=X2, y=self.y)
        betas_first = list(first_stage_reg.coef_[0][1:])
        betas_first.insert(0, first_stage_reg.intercept_[0])
        betas_second = list(second_stage_reg.coef_[0][1:])
        betas_second.insert(0, second_stage_reg.intercept_[0])
        if self._prior_calibration is None:
            self._prior_calibration = PriorCalibration(
                first_stage=first_stage_reg,
                second_stage=second_stage_reg,
                naive=None,
                beta_first=betas_first,
                beta_second=betas_second,
                beta_naive={},
            )
        else:
            self._prior_calibration.first_stage = first_stage_reg
            self._prior_calibration.second_stage = second_stage_reg
            self._prior_calibration.beta_first = betas_first
            self._prior_calibration.beta_second = betas_second

    def get_naive_OLS_fit(self) -> None:
        """Naive Ordinary Least Squares.

        This function is called by the experiment.
        """
        ols_reg = sk_lin_reg().fit(self.X, self.y)
        beta_params = list(ols_reg.coef_[0][1:])
        beta_params.insert(0, ols_reg.intercept_[0])
        beta_naive = dict(zip(self._design.labels, beta_params, strict=False))
        if self._prior_calibration is None:
            self._prior_calibration = PriorCalibration(
                first_stage=None,
                second_stage=None,
                naive=ols_reg,
                beta_first=[],
                beta_second=[],
                beta_naive=beta_naive,
            )
        else:
            self._prior_calibration.naive = ols_reg
            self._prior_calibration.beta_naive = beta_naive

    def plot(
        self,
        *,
        show: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Plot the results.

        Notes
        -----
        Plotting is not yet implemented for instrumental variable
        experiments. This stub exists so every experiment subclass
        offers an explicit, kwarg-only ``plot()`` signature
        (issue `#886 <https://github.com/pymc-labs/CausalPy/issues/886>`_).

        Parameters
        ----------
        show : bool
            Reserved; ignored. Defaults to ``True``.
        legend_kwargs : dict, optional
            Reserved; ignored.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("Plot method not implemented.")

    def summary(self, round_to: int | None = None) -> None:
        """Print summary of main results and model coefficients.

        Parameters
        ----------
        round_to : int, optional
            Number of decimals used to round results. Defaults to 2. Use
            ``None`` to return raw numbers.
        """
        raise NotImplementedError("Summary method not implemented.")

    def effect_summary(
        self,
        *,
        window: Literal["post"] | tuple | slice = "post",
        direction: Literal["increase", "decrease", "two-sided"] = "increase",
        alpha: float = 0.05,
        cumulative: bool = True,
        relative: bool = True,
        min_effect: float | None = None,
        treated_unit: str | None = None,
        period: Literal["intervention", "post", "comparison"] | None = None,
        prefix: str = "Post-period",
        **kwargs: Any,
    ) -> EffectSummary:
        """
        Generate a decision-ready summary of causal effects.

        Note: effect_summary is not yet implemented for InstrumentalVariable experiments.

        Parameters
        ----------
        window : str, tuple, or slice, default "post"
            Time window for analysis (unused for InstrumentalVariable).
        direction : {"increase", "decrease", "two-sided"}, default "increase"
            Direction for tail probability calculation.
        alpha : float, default 0.05
            Significance level for HDI/CI intervals.
        cumulative : bool, default True
            Whether to include cumulative effect statistics.
        relative : bool, default True
            Whether to include relative effect statistics.
        min_effect : float, optional
            Region of Practical Equivalence (ROPE) threshold.
        treated_unit : str, optional
            For multi-unit experiments, the unit to analyse.
        period : {"intervention", "post", "comparison"}, optional
            Period selector for three-period designs.
        prefix : str, default "Post-period"
            Prefix for prose generation.
        **kwargs
            Reserved for forward-compatibility.
        """
        raise NotImplementedError(
            "effect_summary is not yet implemented for InstrumentalVariable experiments."
        )
