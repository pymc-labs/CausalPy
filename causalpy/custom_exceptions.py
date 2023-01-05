class BadIndexException(Exception):
    """Custom exception used when we have a mismatch in types between the dataframe
    index and an event, typically a treatment or intervention."""

    def __init__(self, message):
        self.message = message
