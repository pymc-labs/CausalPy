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
