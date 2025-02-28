#   Copyright 2022 - 2025 The PyMC Labs Developers
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
Instrumental variable regression
"""

import warnings  # noqa: I001

import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LinearRegression as sk_lin_reg

from causalpy.custom_exceptions import DataException
from .base import BaseExperiment


class InstrumentalVariable(BaseExperiment):
    """
    A class to analyse instrumental variable style experiments.

    :param instruments_data: A pandas dataframe of instruments
                             for our treatment variable. Should contain
                             instruments Z, and treatment t
    :param data: A pandas dataframe of covariates for fitting
                 the focal regression of interest. Should contain covariates X
                 including treatment t and outcome y
    :param instruments_formula: A statistical model formula for
                                the instrumental stage regression
                                e.g. t ~ 1 + z1 + z2 + z3
    :param formula: A statistical model formula for the \n
                    focal regression e.g. y ~ 1 + t + x1 + x2 + x3
    :param model: A PyMC model
    :param priors: An optional dictionary of priors for the
                   mus and sigmas of both regressions. If priors are not
                   specified we will substitute MLE estimates for the beta
                   coefficients. Greater control can be achieved
                   by specifying the priors directly e.g. priors = {
                                    "mus": [0, 0],
                                    "sigmas": [1, 1],
                                    "eta": 2,
                                    "lkj_sd": 2,
                                    }

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
    """

    supports_ols = False
    supports_bayes = True

    def __init__(
        self,
        instruments_data: pd.DataFrame,
        data: pd.DataFrame,
        instruments_formula: str,
        formula: str,
        model=None,
        priors=None,
        **kwargs,
    ):
        super().__init__(model=model)
        self.expt_type = "Instrumental Variable Regression"
        self.data = data
        self.instruments_data = instruments_data
        self.formula = formula
        self.instruments_formula = instruments_formula
        self.model = model
        self.input_validation()

        y, X = dmatrices(formula, self.data)
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.y, self.X = np.asarray(y), np.asarray(X)
        self.outcome_variable_name = y.design_info.column_names[0]

        t, Z = dmatrices(instruments_formula, self.instruments_data)
        self._t_design_info = t.design_info
        self._z_design_info = Z.design_info
        self.labels_instruments = Z.design_info.column_names
        self.t, self.Z = np.asarray(t), np.asarray(Z)
        self.instrument_variable_name = t.design_info.column_names[0]

        self.get_naive_OLS_fit()
        self.get_2SLS_fit()

        # fit the model to the data
        COORDS = {"instruments": self.labels_instruments, "covariates": self.labels}
        self.coords = COORDS
        if priors is None:
            priors = {
                "mus": [self.ols_beta_first_params, self.ols_beta_second_params],
                "sigmas": [1, 1],
                "eta": 2,
                "lkj_sd": 1,
            }
        self.priors = priors
        self.model.fit(
            X=self.X, Z=self.Z, y=self.y, t=self.t, coords=COORDS, priors=self.priors
        )

    def input_validation(self):
        """Validate the input data and model formula for correctness"""
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
                This is not necessarily a problem but it violates
                the assumption of a simple IV experiment.
                The coefficients should be interpreted appropriately."""
            )

    def get_2SLS_fit(self):
        """
        Two Stage Least Squares Fit

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
        self.ols_beta_first_params = betas_first
        self.ols_beta_second_params = betas_second
        self.first_stage_reg = first_stage_reg
        self.second_stage_reg = second_stage_reg

    def get_naive_OLS_fit(self):
        """
        Naive Ordinary Least Squares

        This function is called by the experiment.
        """
        ols_reg = sk_lin_reg().fit(self.X, self.y)
        beta_params = list(ols_reg.coef_[0][1:])
        beta_params.insert(0, ols_reg.intercept_[0])
        self.ols_beta_params = dict(zip(self._x_design_info.column_names, beta_params))
        self.ols_reg = ols_reg

    def plot(self, round_to=None):
        """
        Plot the results

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers.
        """
        raise NotImplementedError("Plot method not implemented.")

    def summary(self, round_to=None) -> None:
        """Print summary of main results and model coefficients.

        :param round_to:
            Number of decimals used to round results. Defaults to 2. Use "None" to return raw numbers
        """
        raise NotImplementedError("Summary method not implemented.")
