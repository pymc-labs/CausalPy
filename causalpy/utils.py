import pandas as pd

from causalpy.pymc_models import ModelBuilder


def _fit(model, X, y, coords):
    """Fits model to X, y, where model is either a sklearn model or a ModelBuilder
    instance. In the later case it passes coords, in the first case coords is ignored."""
    if isinstance(model, ModelBuilder):
        model.fit(X, y, coords)
    else:
        model.fit(X, y)


def _is_variable_dummy_coded(series: pd.Series) -> bool:
    """Check if a data in the provided Series is dummy coded. It should be 0 or 1
    only."""
    return len(set(series).difference(set([0, 1]))) == 0


def _series_has_2_levels(series: pd.Series) -> bool:
    """Check that the variable in the provided Series has 2 levels"""
    return len(pd.Categorical(series).categories) == 2
