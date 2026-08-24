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
"""
Custom Exceptions for CausalPy.
"""


class BadIndexException(Exception):
    """Custom exception used when we have a mismatch in types between the dataframe
    index and an event, typically a treatment or intervention.

    Parameters
    ----------
    message : str
        Human-readable description of the index mismatch.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class FormulaException(Exception):
    """Exception raised given when there is some error in a user-provided model
    formula.

    Parameters
    ----------
    message : str
        Human-readable description of the formula problem.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class DataException(Exception):
    """Exception raised given when there is some error in user-provided dataframe.

    Parameters
    ----------
    message : str
        Human-readable description of the data problem.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class GroupNotSampledException(Exception):
    """Raised when a read method requests a draw group that has not been sampled.

    Under the lazy lifecycle an experiment holds no draws until
    :meth:`~causalpy.experiments.base.BaseExperiment.fit` (posterior group) or
    :meth:`~causalpy.experiments.base.BaseExperiment.sample_prior_predictive`
    (prior group) is called. This exception carries the missing group and the
    call that would populate it so the fix is actionable.

    Parameters
    ----------
    message : str
        Human-readable description naming the missing group and the call to
        make.
    group : str, optional
        The draw group that was requested but not sampled.
    """

    def __init__(self, message: str, group: str | None = None):
        super().__init__(message)
        self.message = message
        self.group = group


class PriorPredictiveNotSupportedException(Exception):
    """Raised when prior predictive sampling is requested of a backend that cannot do it.

    Prior-phase support is a property of the model backend. OLS/sklearn
    models, ``PyMCForecastModel``, and the PyMC state-space and instrumental
    variable models do not expose a prior predictive phase.

    Parameters
    ----------
    message : str
        Human-readable description naming the model class that lacks the
        capability.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
