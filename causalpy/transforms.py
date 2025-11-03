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
Transform specifications and utilities for Transfer Function ITS.

This module provides dataclasses for specifying saturation, adstock, and lag
transforms for treatment channels. It leverages pymc-marketing's battle-tested
transform implementations for consistency with the PyMC ecosystem.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
from pymc_marketing.mmm.transformers import (
    geometric_adstock,
    hill_function,
    logistic_saturation,
    michaelis_menten,
)


@dataclass
class Saturation:
    """Saturation transform specification using pymc-marketing implementations.

    Saturation transforms model diminishing returns in the response to increasing
    exposure levels (e.g., ad spend).

    Parameters
    ----------
    kind : str
        Type of saturation function. Options:
        - "hill": Hill function with slope and kappa parameters
        - "logistic": Logistic saturation with lam parameter
        - "michaelis_menten": Michaelis-Menten function with alpha and lam
        - None: No saturation transform
    slope : float, optional
        Hill function slope parameter (s). Higher values create steeper curves.
        Required when kind="hill".
    kappa : float, optional
        Hill function half-saturation point (k). The exposure level at which
        the response reaches 50% of maximum. Required when kind="hill".
    lam : float, optional
        Lambda parameter for logistic or Michaelis-Menten saturation.
        Required when kind="logistic" or kind="michaelis_menten".
    alpha : float, optional
        Alpha parameter for Michaelis-Menten saturation.
        Required when kind="michaelis_menten".

    Examples
    --------
    >>> # Hill saturation with half-saturation at 10000 units
    >>> sat = Saturation(kind="hill", slope=2.0, kappa=10000)
    >>> # Logistic saturation
    >>> sat = Saturation(kind="logistic", lam=0.5)

    Notes
    -----
    For future extensions, additional saturation functions from pymc-marketing
    can be added, such as tanh_saturation, root_saturation, etc.
    """

    kind: str
    slope: Optional[float] = None
    kappa: Optional[float] = None
    lam: Optional[float] = None
    alpha: Optional[float] = None

    def __post_init__(self):
        """Validate that required parameters are provided for the chosen kind."""
        if self.kind == "hill":
            if self.slope is None or self.kappa is None:
                raise ValueError(
                    "Hill saturation requires 'slope' and 'kappa' parameters"
                )
        elif self.kind == "logistic":
            if self.lam is None:
                raise ValueError("Logistic saturation requires 'lam' parameter")
        elif self.kind == "michaelis_menten":
            if self.alpha is None or self.lam is None:
                raise ValueError(
                    "Michaelis-Menten saturation requires 'alpha' and 'lam' parameters"
                )
        elif self.kind is not None:
            raise ValueError(
                f"Unknown saturation kind: {self.kind}. "
                f"Options are: 'hill', 'logistic', 'michaelis_menten', or None"
            )


@dataclass
class Adstock:
    """Adstock (carryover) transform specification using pymc-marketing geometric adstock.

    Adstock transforms model the carryover effect where exposure in one period
    affects outcomes in subsequent periods (e.g., advertising carryover).

    Parameters
    ----------
    alpha : float, optional
        Geometric decay rate, must be in (0, 1). Higher values indicate longer
        carryover. Either alpha or half_life must be provided.
    half_life : float, optional
        Half-life of the carryover effect in the same units as the data frequency
        (e.g., weeks for weekly data). The number of periods until the effect
        decays to 50% of its original value. Either alpha or half_life must be provided.
    l_max : int, default=12
        Maximum lag for the convolution (truncation point). Should be long enough
        to capture the full carryover effect but not unnecessarily long.
    normalize : bool, default=True
        If True, normalize the adstock weights to sum to 1. When normalized,
        the treatment coefficient represents the long-run cumulative effect per
        unit of (saturated) exposure.

    Examples
    --------
    >>> # Adstock with 3-week half-life, normalized
    >>> adstock = Adstock(half_life=3, l_max=12, normalize=True)
    >>> # Adstock with direct alpha specification
    >>> adstock = Adstock(alpha=0.8, l_max=10, normalize=False)

    Notes
    -----
    The geometric adstock function applies a geometric decay:
    g_t = x_t + alpha * x_{t-1} + alpha^2 * x_{t-2} + ...

    When normalize=True, the weights sum to 1, so the long-run effect per unit
    exposure is directly interpretable from the coefficient.

    FUTURE: Additional adstock forms available in pymc-marketing include
    delayed_adstock (with a delay parameter theta) and weibull_adstock for
    more flexible decay patterns.
    """

    alpha: Optional[float] = None
    half_life: Optional[float] = None
    l_max: int = 12
    normalize: bool = True

    def __post_init__(self):
        """Convert half_life to alpha if provided, and validate parameters."""
        if self.half_life is not None and self.alpha is None:
            self.alpha = np.power(0.5, 1 / self.half_life)
        elif self.alpha is None and self.half_life is None:
            raise ValueError("Must provide either 'alpha' or 'half_life'")

        if self.alpha is not None and not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")


@dataclass
class Lag:
    """Discrete lag (delay) transform specification.

    A simple shift that delays the effect by a fixed number of periods.

    Parameters
    ----------
    k : int, default=0
        Number of periods to delay. k=0 means no delay (immediate effect).
        k=1 means the effect appears one period after exposure, etc.

    Examples
    --------
    >>> # No delay
    >>> lag = Lag(k=0)
    >>> # 2-period delay
    >>> lag = Lag(k=2)
    """

    k: int = 0

    def __post_init__(self):
        """Validate lag parameter."""
        if self.k < 0:
            raise ValueError(f"Lag k must be non-negative, got {self.k}")


@dataclass
class Treatment:
    """Treatment channel specification for Transfer Function ITS.

    A treatment channel represents a time-varying intervention (e.g., media spend,
    policy intensity) along with its transformation pipeline.

    Parameters
    ----------
    name : str
        Column name in the data DataFrame containing the raw exposure series.
    transforms : List[Union[Saturation, Adstock, Lag]], optional
        Ordered list of transforms to apply. Transforms are applied in sequence:
        Saturation → Adstock → Lag. Default is an empty list (no transforms).
    coef_constraint : str, default="nonnegative"
        Constraint on the treatment coefficient. Options:
        - "nonnegative": Coefficient must be >= 0 (typical for media effects)
        - "unconstrained": No constraint on coefficient sign

    Examples
    --------
    >>> # TV spend with Hill saturation and 3-week adstock
    >>> tv = Treatment(
    ...     name="tv_spend",
    ...     transforms=[
    ...         Saturation(kind="hill", slope=2.0, kappa=10000),
    ...         Adstock(half_life=3, normalize=True),
    ...     ],
    ...     coef_constraint="nonnegative",
    ... )
    >>> # Simple treatment with adstock only
    >>> promo = Treatment(name="promo_intensity", transforms=[Adstock(half_life=2)])

    Notes
    -----
    The transform order is fixed: Saturation → Adstock → Lag. This ordering
    reflects the typical causal sequence: first, exposure saturates, then the
    saturated effect carries over across time, and finally an optional delay
    can be applied.

    FUTURE: Grid search for optimal transform parameters can be added by
    iterating over parameter combinations and selecting based on AICc or
    pre-period RMSE.
    """

    name: str
    transforms: List[Union[Saturation, Adstock, Lag]] = field(default_factory=list)
    coef_constraint: str = "nonnegative"

    def __post_init__(self):
        """Validate treatment specification."""
        if self.coef_constraint not in ["nonnegative", "unconstrained"]:
            raise ValueError(
                f"coef_constraint must be 'nonnegative' or 'unconstrained', "
                f"got '{self.coef_constraint}'"
            )


def apply_saturation(x: np.ndarray, saturation: Saturation) -> np.ndarray:
    """Apply saturation transform using pymc-marketing functions.

    Parameters
    ----------
    x : np.ndarray
        Input series (1D array).
    saturation : Saturation
        Saturation specification.

    Returns
    -------
    np.ndarray
        Saturated series.

    Examples
    --------
    >>> x = np.array([100, 500, 1000, 5000])
    >>> sat = Saturation(kind="hill", slope=2.0, kappa=1000)
    >>> x_sat = apply_saturation(x, sat)
    """
    if saturation.kind is None:
        return x

    result = None
    if saturation.kind == "hill":
        # pymc-marketing hill_function(x, slope, kappa)
        result = hill_function(x, slope=saturation.slope, kappa=saturation.kappa)
    elif saturation.kind == "logistic":
        # pymc-marketing logistic_saturation(x, lam)
        result = logistic_saturation(x, lam=saturation.lam)
    elif saturation.kind == "michaelis_menten":
        # pymc-marketing michaelis_menten(x, alpha, lam)
        result = michaelis_menten(x, alpha=saturation.alpha, lam=saturation.lam)
    else:
        raise ValueError(f"Unknown saturation kind: {saturation.kind}")

    # Ensure we return a numpy array, not a PyTensor symbolic tensor
    # If the result is a PyTensor tensor, evaluate it
    if hasattr(result, "eval"):
        return result.eval()
    return np.asarray(result)


def apply_adstock(x: np.ndarray, adstock: Adstock) -> np.ndarray:
    """Apply adstock transform using pymc-marketing geometric_adstock.

    Parameters
    ----------
    x : np.ndarray
        Input series (1D array).
    adstock : Adstock
        Adstock specification.

    Returns
    -------
    np.ndarray
        Adstocked series.

    Examples
    --------
    >>> x = np.array([0, 0, 100, 0, 0])
    >>> adstock = Adstock(half_life=2, l_max=4, normalize=True)
    >>> x_adstock = apply_adstock(x, adstock)
    """
    # pymc-marketing geometric_adstock(x, alpha, l_max, normalize, mode)
    # mode="After" means only past values affect current (causal)
    result = geometric_adstock(
        x,
        alpha=adstock.alpha,
        l_max=adstock.l_max,
        normalize=adstock.normalize,
        mode="After",
    )

    # Ensure we return a numpy array, not a PyTensor symbolic tensor
    # If the result is a PyTensor tensor, evaluate it
    if hasattr(result, "eval"):
        return result.eval()
    return np.asarray(result)


def apply_lag(x: np.ndarray, lag: Lag) -> np.ndarray:
    """Apply discrete lag (delay) transform.

    Parameters
    ----------
    x : np.ndarray
        Input series (1D array).
    lag : Lag
        Lag specification.

    Returns
    -------
    np.ndarray
        Lagged series. The first k values are filled with 0.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> lag = Lag(k=2)
    >>> x_lagged = apply_lag(x, lag)
    >>> x_lagged
    array([0, 0, 1, 2, 3])
    """
    if lag.k == 0:
        return x

    # Shift the array and fill the beginning with zeros
    lagged = np.zeros_like(x)
    lagged[lag.k :] = x[: -lag.k]
    return lagged


def apply_treatment_transforms(x: np.ndarray, treatment: Treatment) -> np.ndarray:
    """Apply the full transform pipeline for a treatment channel.

    Transforms are applied in the order: Saturation → Adstock → Lag.

    Parameters
    ----------
    x : np.ndarray
        Raw exposure series (1D array).
    treatment : Treatment
        Treatment specification with transform pipeline.

    Returns
    -------
    np.ndarray
        Fully transformed series.

    Examples
    --------
    >>> x = np.array([100, 200, 300, 200, 100])
    >>> treatment = Treatment(
    ...     name="tv",
    ...     transforms=[
    ...         Saturation(kind="hill", slope=1.0, kappa=200),
    ...         Adstock(half_life=2, normalize=True),
    ...     ],
    ... )
    >>> x_transformed = apply_treatment_transforms(x, treatment)

    Notes
    -----
    The transform order is enforced to match the typical causal sequence:
    1. Saturation (diminishing returns within period)
    2. Adstock (carryover across periods)
    3. Lag (discrete delay)

    FUTURE: When estimating transforms via grid search, this function will be
    called repeatedly with different parameter values to find optimal settings
    based on AICc or pre-period fit.
    """
    result = x.copy()

    # Apply transforms in order: Saturation → Adstock → Lag
    for transform in treatment.transforms:
        if isinstance(transform, Saturation):
            result = apply_saturation(result, transform)
        elif isinstance(transform, Adstock):
            result = apply_adstock(result, transform)
        elif isinstance(transform, Lag):
            result = apply_lag(result, transform)
        else:
            raise ValueError(f"Unknown transform type: {type(transform)}")

    return result
