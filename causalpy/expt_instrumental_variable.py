#   Copyright 2024 The PyMC Labs Developers
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
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LinearRegression as sk_lin_reg

from causalpy.data_validation import IVDataValidator
from causalpy.experiments import ExperimentalDesign

# from causalpy.pymc_models import PyMCModel
# from causalpy.utils import round_num


class InstrumentalVariable(ExperimentalDesign, IVDataValidator):
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
        super().__init__(model=model, **kwargs)
        self.expt_type = "Instrumental Variable Regression"
        self.data = data
        self.instruments_data = instruments_data
        self.formula = formula
        self.instruments_formula = instruments_formula
        self.model = model
        self._input_validation()

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

    def plot(self):
        raise NotImplementedError("Plot method not implemented.")

    def summary(self, round_to=None) -> None:
        raise NotImplementedError("Summary method not implemented.")
