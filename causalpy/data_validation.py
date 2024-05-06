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
import warnings  # noqa: I001

import pandas as pd
import numpy as np
from causalpy.custom_exceptions import (
    BadIndexException,  # NOQA
    DataException,
    FormulaException,
)
from causalpy.utils import _is_variable_dummy_coded


class PrePostFitDataValidator:
    """Mixin class for validating the input data and model formula for PrePostFit"""

    def _input_validation(self, data, treatment_time):
        """Validate the input data and model formula for correctness"""
        if isinstance(data.index, pd.DatetimeIndex) and not isinstance(
            treatment_time, pd.Timestamp
        ):
            raise BadIndexException(
                "If data.index is DatetimeIndex, treatment_time must be pd.Timestamp."
            )
        if not isinstance(data.index, pd.DatetimeIndex) and isinstance(
            treatment_time, pd.Timestamp
        ):
            raise BadIndexException(
                "If data.index is not DatetimeIndex, treatment_time must be pd.Timestamp."  # noqa: E501
            )


class DiDDataValidator:
    """Mixin class for validating the input data and model formula for Difference in Differences experiments."""

    def _input_validation(self):
        """Validate the input data and model formula for correctness"""
        if "post_treatment" not in self.formula:
            raise FormulaException(
                "A predictor called `post_treatment` should be in the formula"
            )

        if "post_treatment" not in self.data.columns:
            raise DataException(
                "Require a boolean column labelling observations which are `treated`"
            )

        if "unit" not in self.data.columns:
            raise DataException(
                "Require a `unit` column to label unique units. This is used for plotting purposes"  # noqa: E501
            )

        if _is_variable_dummy_coded(self.data[self.group_variable_name]) is False:
            raise DataException(
                f"""The grouping variable {self.group_variable_name} should be dummy
                coded. Consisting of 0's and 1's only."""
            )


class RDDataValidator:
    """Mixin class for validating the input data and model formula for Regression Discontinuity experiments."""

    def _input_validation(self):
        """Validate the input data and model formula for correctness"""
        if "treated" not in self.formula:
            raise FormulaException(
                "A predictor called `treated` should be in the formula"
            )

        if _is_variable_dummy_coded(self.data["treated"]) is False:
            raise DataException(
                """The treated variable should be dummy coded. Consisting of 0's and 1's only."""  # noqa: E501
            )


class RegressionKinkDataValidator:
    """Mixin class for validating the input data and model formula for Regression Kink experiments."""

    def _input_validation(self):
        """Validate the input data and model formula for correctness"""
        if "treated" not in self.formula:
            raise FormulaException(
                "A predictor called `treated` should be in the formula"
            )

        if _is_variable_dummy_coded(self.data["treated"]) is False:
            raise DataException(
                """The treated variable should be dummy coded. Consisting of 0's and 1's only."""  # noqa: E501
            )

        if self.bandwidth <= 0:
            raise ValueError("The bandwidth must be greater than zero.")

        if self.epsilon <= 0:
            raise ValueError("Epsilon must be greater than zero.")


class PrePostNEGDDataValidator:
    """Mixin class for validating the input data and model formula for PrePostNEGD experiments."""

    def _input_validation(self) -> None:
        """Validate the input data and model formula for correctness"""
        if not _is_variable_dummy_coded(self.data[self.group_variable_name]):
            raise DataException(
                f"""
                There must be 2 levels of the grouping variable
                {self.group_variable_name}. I.e. the treated and untreated.
                """
            )


class IVDataValidator:
    """Mixin class for validating the input data and model formula for IV experiments."""

    def _input_validation(self):
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


class PropensityDataValidator:
    """Mixin class for validating the input data and model formula for Propensity Weighting experiments."""

    def _input_validation(self):
        """Validate the input data and model formula for correctness"""
        treatment = self.formula.split("~")[0]
        test = treatment.strip() in self.data.columns
        test = test & (self.outcome_variable in self.data.columns)
        if not test:
            raise DataException(
                f"""
                The treatment variable:
                {treatment} must appear in the data to be used
                as an outcome variable. And {self.outcome_variable}
                must also be available in the data to be re-weighted
                """
            )
        T = self.data[treatment.strip()]
        check_binary = len(np.unique(T)) > 2
        if check_binary:
            raise DataException(
                """Warning. The treatment variable is not 0-1 Binary.
                """
            )
