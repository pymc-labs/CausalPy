class BadIndexException(Exception):
    """Custom exception used when we have a mismatch in types between the dataframe
    index and an event, typically a treatment or intervention."""

    def __init__(self, message):
        self.message = message


class FormulaException(Exception):
    """Exception raised given when there is some error in a user-provided model
    formula"""

    def __init__(self, message):
        self.message = message


class DataException(Exception):
    """Exception raised given when there is some error in user-provided dataframe"""

    def __init__(self, message):
        self.message = message
