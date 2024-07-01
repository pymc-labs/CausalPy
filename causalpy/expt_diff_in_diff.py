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
from patsy import build_design_matrices, dmatrices

from causalpy.data_validation import DiDDataValidator
from causalpy.experiments import ExperimentalDesign
from causalpy.pymc_models import PyMCModel
from causalpy.utils import round_num


class DifferenceInDifferences(ExperimentalDesign, DiDDataValidator):
    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        time_variable_name: str,
        group_variable_name: str,
        model=None,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)

        self.data = data
        self.expt_type = "Difference in Differences"
        self.formula = formula
        self.time_variable_name = time_variable_name
        self.group_variable_name = group_variable_name
        self._input_validation()

        y, X = dmatrices(formula, self.data)
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.y, self.X = np.asarray(y), np.asarray(X)
        self.outcome_variable_name = y.design_info.column_names[0]

        # ******** THIS IS SUBOPTIMAL AT THE MOMENT ************************************
        if isinstance(self.model, PyMCModel):
            COORDS = {"coeffs": self.labels, "obs_indx": np.arange(self.X.shape[0])}
            self.model.fit(X=self.X, y=self.y, coords=COORDS)
        else:
            self.model.fit(X=self.X, y=self.y)
        # ******************************************************************************

        # predicted outcome for control group
        self.x_pred_control = (
            self.data
            # just the untreated group
            .query(f"{self.group_variable_name} == 0")
            # drop the outcome variable
            .drop(self.outcome_variable_name, axis=1)
            # We may have multiple units per time point, we only want one time point
            .groupby(self.time_variable_name)
            .first()
            .reset_index()
        )
        assert not self.x_pred_control.empty
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred_control)
        self.y_pred_control = self.model.predict(np.asarray(new_x))

        # predicted outcome for treatment group
        self.x_pred_treatment = (
            self.data
            # just the treated group
            .query(f"{self.group_variable_name} == 1")
            # drop the outcome variable
            .drop(self.outcome_variable_name, axis=1)
            # We may have multiple units per time point, we only want one time point
            .groupby(self.time_variable_name)
            .first()
            .reset_index()
        )
        assert not self.x_pred_treatment.empty
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred_treatment)
        self.y_pred_treatment = self.model.predict(np.asarray(new_x))

        # predicted outcome for counterfactual. This is given by removing the influence
        # of the interaction term between the group and the post_treatment variable
        self.x_pred_counterfactual = (
            self.data
            # just the treated group
            .query(f"{self.group_variable_name} == 1")
            # just the treatment period(s)
            .query("post_treatment == True")
            # drop the outcome variable
            .drop(self.outcome_variable_name, axis=1)
            # We may have multiple units per time point, we only want one time point
            .groupby(self.time_variable_name)
            .first()
            .reset_index()
        )
        assert not self.x_pred_counterfactual.empty
        (new_x,) = build_design_matrices(
            [self._x_design_info], self.x_pred_counterfactual, return_type="dataframe"
        )
        # INTERVENTION: set the interaction term between the group and the
        # post_treatment variable to zero. This is the counterfactual.
        for i, label in enumerate(self.labels):
            if "post_treatment" in label and self.group_variable_name in label:
                new_x.iloc[:, i] = 0
        self.y_pred_counterfactual = self.model.predict(np.asarray(new_x))

        # ******** THIS IS SUBOPTIMAL AT THE MOMENT ************************************
        if isinstance(self.model, PyMCModel):
            # calculate causal impact &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            # This is the coefficient on the interaction term
            coeff_names = self.model.idata.posterior.coords["coeffs"].data
            for i, label in enumerate(coeff_names):
                if "post_treatment" in label and self.group_variable_name in label:
                    self.causal_impact = self.model.idata.posterior["beta"].isel(
                        {"coeffs": i}
                    )
            # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        else:
            # calculate causal impact
            # This is the coefficient on the interaction term
            # TODO: THIS IS NOT YET CORRECT
            self.causal_impact = (
                self.y_pred_treatment[1] - self.y_pred_counterfactual[0]
            )
        # ******************************************************************************

    def plot(self):
        # Get a BayesianPlotComponent or OLSPlotComponent depending on the model
        plot_component = self.model.get_plot_component()
        plot_component.plot_difference_in_differences(self)

    def summary(self, round_to=None) -> None:
        print(f"{self.expt_type:=^80}")
        print(f"Formula: {self.formula}")
        print("\nResults:")
        print(self._causal_impact_summary_stat(round_to))
        self.print_coefficients(round_to)

    def _causal_impact_summary_stat(self, round_to=None) -> str:
        """Computes the mean and 94% credible interval bounds for the causal impact."""
        percentiles = self.causal_impact.quantile([0.03, 1 - 0.03]).values
        ci = (
            "$CI_{94\\%}$"
            + f"[{round_num(percentiles[0], round_to)}, {round_num(percentiles[1], round_to)}]"
        )
        causal_impact = f"{round_num(self.causal_impact.mean(), round_to)}, "
        return f"Causal impact = {causal_impact + ci}"
