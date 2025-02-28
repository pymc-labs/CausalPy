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
Custom Exceptions for CausalPy.
"""


class BadIndexException(Exception):
    """Custom exception used when we have a mismatch in types between the dataframe
    index and an event, typically a treatment or intervention."""

    def __init__(self, message: str):
        self.message = message


class FormulaException(Exception):
    """Exception raised given when there is some error in a user-provided model
    formula"""

    def __init__(self, message: str):
        self.message = message


class DataException(Exception):
    """Exception raised given when there is some error in user-provided dataframe"""

    def __init__(self, message: str):
        self.message = message
