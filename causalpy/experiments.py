import warnings
from typing import Optional

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrices

from causalpy.custom_exceptions import DataException, FormulaException
from causalpy.utils import _is_variable_dummy_coded


class ExperimentalDesign:
    """
    Base class for other experiment types

    See subclasses for examples of most methods
    """

    model = None
    outcome_variable_name = None
    expt_type = None

    def __init__(self, model=None, **kwargs):
        if model is not None:
            self.model = model
        if self.model is None:
            raise ValueError("fitting_model not set or passed.")

    @property
    def idata(self):
        """
        Access to the models InferenceData object
        """

        return self.model.idata


class RDD(ExperimentalDesign):
    def __init__(
        self,
        data,
        formula,
        treatment_threshold,
        model=None,
        running_variable_name="x",
        epsilon: float = 0.001,
        bandwidth: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.expt_type = "Regression Discontinuity"
        self.data = data
        self.formula = formula
        self.running_variable_name = running_variable_name
        self.treatment_threshold = treatment_threshold
        self.epsilon = epsilon
        self.bandwidth = bandwidth
        self._input_validation()

        # REGRESSION DISCONTINUITY ALGORITHM ~~~~~~~~~~~~~~~~~~~~~
        y, X = self.bandwidth_clip(formula)
        self.process_design_matrix(y, X)
        self.fit(X, y)
        self.score = self.model.score(X, y)
        self.calc_model_predictions()
        self.calc_discontinuity()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def fit(self, X, y):
        self.model.fit(X, y)

    def process_design_matrix(self, y, X):
        self._y_design_info = y.design_info
        self._x_design_info = X.design_info
        self.labels = X.design_info.column_names
        self.y, self.X = np.asarray(y), np.asarray(X)
        self.outcome_variable_name = y.design_info.column_names[0]

    def calc_model_predictions(self):
        # get the model predictions of the observed data
        if self.bandwidth is not None:
            fmin = self.treatment_threshold - self.bandwidth
            fmax = self.treatment_threshold + self.bandwidth
            xi = np.linspace(fmin, fmax, 200)
        else:
            xi = np.linspace(
                np.min(self.data[self.running_variable_name]),
                np.max(self.data[self.running_variable_name]),
                200,
            )
        self.x_pred = pd.DataFrame(
            {self.running_variable_name: xi, "treated": self._is_treated(xi)}
        )
        (new_x,) = build_design_matrices([self._x_design_info], self.x_pred)
        self.pred = self.model.predict(X=np.asarray(new_x))

    def bandwidth_clip(self, formula):
        if self.bandwidth is not None:
            fmin = self.treatment_threshold - self.bandwidth
            fmax = self.treatment_threshold + self.bandwidth
            filtered_data = self.data.query(f"{fmin} <= x <= {fmax}")
            if len(filtered_data) <= 10:
                warnings.warn(
                    f"Choice of bandwidth parameter has lead to only {len(filtered_data)} remaining datapoints. Consider increasing the bandwidth parameter.",  # noqa: E501
                    UserWarning,
                )
            y, X = dmatrices(formula, filtered_data)
        else:
            y, X = dmatrices(formula, self.data)
        return y, X

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

    def _is_treated(self, x):
        """Returns ``True`` if ``x`` is greater than or equal to the treatment
        threshold.

        .. warning::

            Assumes treatment is given to those ABOVE the treatment threshold.
        """
        return np.greater_equal(x, self.treatment_threshold)
