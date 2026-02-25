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
Transform specifications and utilities for Transfer Function ITS.

This module provides a strategy pattern implementation for saturation, adstock,
and lag transforms for treatment channels. It leverages pymc-marketing's
battle-tested transform implementations for consistency with the PyMC ecosystem.

The strategy pattern (following pymc-marketing design) provides:
- Common interface for all transforms via .apply() method
- Extensibility without modifying existing code
- Easy parameter retrieval via .get_params() method
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from pymc_marketing.mmm.transformers import (
    ConvMode,
    geometric_adstock,
    hill_function,
    logistic_saturation,
    michaelis_menten,
)

# ============================================================================
# Strategy Pattern Base Classes
# ============================================================================


class SaturationTransform(ABC):
    """Base class for saturation transforms.

    Saturation transforms model diminishing returns in the response to increasing
    exposure levels (e.g., ad spend, policy intensity).

    Following the strategy pattern, all saturation transforms must implement:
    - apply(x): Transform input array
    - get_params(): Return dictionary of parameters
    """

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply saturation transform to input array.

        Parameters
        ----------
        x : np.ndarray
            Input series (1D array).

        Returns
        -------
        np.ndarray
            Saturated series.
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Return transform parameters as a dictionary.

        Returns
        -------
        dict
            Dictionary of parameter names and values.
        """
        pass


class AdstockTransform(ABC):
    """Base class for adstock (carryover) transforms.

    Adstock transforms model the carryover effect where exposure in one period
    affects outcomes in subsequent periods (e.g., advertising carryover).

    Following the strategy pattern, all adstock transforms must implement:
    - apply(x): Transform input array
    - get_params(): Return dictionary of parameters
    """

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply adstock transform to input array.

        Parameters
        ----------
        x : np.ndarray
            Input series (1D array).

        Returns
        -------
        np.ndarray
            Adstocked series.
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Return transform parameters as a dictionary.

        Returns
        -------
        dict
            Dictionary of parameter names and values.
        """
        pass


class LagTransform(ABC):
    """Base class for lag (delay) transforms.

    Lag transforms apply a simple shift that delays the effect by a fixed
    number of periods.

    Following the strategy pattern, all lag transforms must implement:
    - apply(x): Transform input array
    - get_params(): Return dictionary of parameters
    """

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply lag transform to input array.

        Parameters
        ----------
        x : np.ndarray
            Input series (1D array).

        Returns
        -------
        np.ndarray
            Lagged series.
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Return transform parameters as a dictionary.

        Returns
        -------
        dict
            Dictionary of parameter names and values.
        """
        pass


# ============================================================================
# Concrete Saturation Implementations
# ============================================================================


class HillSaturation(SaturationTransform):
    """Hill saturation function.

    Models diminishing returns using the Hill function, commonly used in
    pharmacology and marketing mix modeling.

    Parameters
    ----------
    slope : float
        Hill function slope parameter (s). Higher values create steeper curves.
    kappa : float
        Hill function half-saturation point (k). The exposure level at which
        the response reaches 50% of maximum.

    Examples
    --------
    >>> saturation = HillSaturation(slope=2.0, kappa=5000)
    >>> x = np.array([1000, 5000, 10000])
    >>> x_saturated = saturation.apply(x)
    """

    def __init__(self, slope: float, kappa: float):
        """Initialize Hill saturation with parameters."""
        self.slope = slope
        self.kappa = kappa

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply Hill saturation transform."""
        result = hill_function(x, slope=self.slope, kappa=self.kappa)
        # Ensure we return a numpy array, not a PyTensor symbolic tensor
        if hasattr(result, "eval"):
            return result.eval()
        return np.asarray(result)

    def get_params(self) -> dict:
        """Return Hill saturation parameters."""
        return {"slope": self.slope, "kappa": self.kappa}


class LogisticSaturation(SaturationTransform):
    """Logistic saturation function.

    Models diminishing returns using the logistic function.

    Parameters
    ----------
    lam : float
        Lambda parameter controlling the saturation rate.

    Examples
    --------
    >>> saturation = LogisticSaturation(lam=0.5)
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> x_saturated = saturation.apply(x)
    """

    def __init__(self, lam: float):
        """Initialize logistic saturation with parameter."""
        self.lam = lam

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply logistic saturation transform."""
        result = logistic_saturation(x, lam=self.lam)
        # Ensure we return a numpy array, not a PyTensor symbolic tensor
        if hasattr(result, "eval"):
            return result.eval()
        return np.asarray(result)

    def get_params(self) -> dict:
        """Return logistic saturation parameters."""
        return {"lam": self.lam}


class MichaelisMentenSaturation(SaturationTransform):
    """Michaelis-Menten saturation function.

    Models diminishing returns using the Michaelis-Menten equation from
    enzyme kinetics.

    Parameters
    ----------
    alpha : float
        Maximum saturation level.
    lam : float
        Half-saturation constant.

    Examples
    --------
    >>> saturation = MichaelisMentenSaturation(alpha=1.0, lam=100)
    >>> x = np.array([50, 100, 200, 500])
    >>> x_saturated = saturation.apply(x)
    """

    def __init__(self, alpha: float, lam: float):
        """Initialize Michaelis-Menten saturation with parameters."""
        self.alpha = alpha
        self.lam = lam

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply Michaelis-Menten saturation transform."""
        result = michaelis_menten(x, alpha=self.alpha, lam=self.lam)
        # Ensure we return a numpy array, not a PyTensor symbolic tensor
        if hasattr(result, "eval"):
            return result.eval()
        return np.asarray(result)

    def get_params(self) -> dict:
        """Return Michaelis-Menten saturation parameters."""
        return {"alpha": self.alpha, "lam": self.lam}


# ============================================================================
# Saturation Factory
# ============================================================================

SATURATION_TYPES: dict[str, type[SaturationTransform]] = {
    "hill": HillSaturation,
    "logistic": LogisticSaturation,
    "michaelis_menten": MichaelisMentenSaturation,
}


def create_saturation(saturation_type: str, **kwargs) -> SaturationTransform:
    """Create a saturation transform from a type string and parameters.

    Parameters
    ----------
    saturation_type : str
        One of ``"hill"``, ``"logistic"``, or ``"michaelis_menten"``.
    **kwargs
        Parameters forwarded to the chosen saturation class constructor.

    Returns
    -------
    SaturationTransform
        An instance of the requested saturation transform.

    Raises
    ------
    ValueError
        If ``saturation_type`` is not a recognised type.

    Examples
    --------
    >>> sat = create_saturation("hill", slope=2.0, kappa=5.0)
    >>> isinstance(sat, HillSaturation)
    True
    """
    cls = SATURATION_TYPES.get(saturation_type)
    if cls is None:
        raise ValueError(
            f"Unknown saturation type: {saturation_type!r}. "
            f"Choose from: {sorted(SATURATION_TYPES.keys())}"
        )
    return cls(**kwargs)


# ============================================================================
# Concrete Adstock Implementations
# ============================================================================


class GeometricAdstock(AdstockTransform):
    """Geometric adstock function.

    Models carryover effects using geometric decay, where past exposures
    decay exponentially over time.

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
    >>> adstock = GeometricAdstock(half_life=3, l_max=12, normalize=True)
    >>> x = np.array([0, 0, 100, 0, 0])
    >>> x_adstocked = adstock.apply(x)

    >>> # Adstock with direct alpha specification
    >>> adstock = GeometricAdstock(alpha=0.8, l_max=10, normalize=False)

    Notes
    -----
    The geometric adstock function applies a geometric decay:
    g_t = x_t + alpha * x_{t-1} + alpha^2 * x_{t-2} + ...

    When normalize=True, the weights sum to 1, so the long-run effect per unit
    exposure is directly interpretable from the coefficient.
    """

    def __init__(
        self,
        alpha: float | None = None,
        half_life: float | None = None,
        l_max: int = 12,
        normalize: bool = True,
    ):
        """Initialize geometric adstock with parameters."""
        # Convert half_life to alpha if provided
        if half_life is not None and alpha is None:
            self.alpha = np.power(0.5, 1 / half_life)
            self.half_life = half_life
        elif alpha is not None:
            self.alpha = alpha
            # Calculate half_life from alpha for get_params()
            self.half_life = np.log(0.5) / np.log(alpha)
        else:
            raise ValueError("Must provide either 'alpha' or 'half_life'")

        if self.alpha is not None and not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")

        self.l_max = l_max
        self.normalize = normalize

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply geometric adstock transform."""
        # pymc-marketing geometric_adstock(x, alpha, l_max, normalize, mode)
        # mode="After" means only past values affect current (causal)
        result = geometric_adstock(
            x,
            alpha=self.alpha,
            l_max=self.l_max,
            normalize=self.normalize,
            mode=ConvMode.After,
        )

        # Ensure we return a numpy array, not a PyTensor symbolic tensor
        if hasattr(result, "eval"):
            return result.eval()
        return np.asarray(result)

    def get_params(self) -> dict:
        """Return geometric adstock parameters."""
        return {
            "alpha": self.alpha,
            "half_life": self.half_life,
            "l_max": self.l_max,
            "normalize": self.normalize,
        }


# ============================================================================
# Concrete Lag Implementations
# ============================================================================


class DiscreteLag(LagTransform):
    """Discrete lag (delay) transform.

    A simple shift that delays the effect by a fixed number of periods.

    Parameters
    ----------
    k : int, default=0
        Number of periods to delay. k=0 means no delay (immediate effect).
        k=1 means the effect appears one period after exposure, etc.

    Examples
    --------
    >>> # No delay
    >>> lag = DiscreteLag(k=0)
    >>> # 2-period delay
    >>> lag = DiscreteLag(k=2)
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> x_lagged = lag.apply(x)
    >>> x_lagged
    array([0, 0, 1, 2, 3])
    """

    def __init__(self, k: int = 0):
        """Initialize discrete lag with parameter."""
        if k < 0:
            raise ValueError(f"Lag k must be non-negative, got {k}")
        self.k = k

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply discrete lag transform."""
        if self.k == 0:
            return x

        # Shift the array and fill the beginning with zeros
        lagged = np.zeros_like(x)
        lagged[self.k :] = x[: -self.k]
        return lagged

    def get_params(self) -> dict:
        """Return discrete lag parameters."""
        return {"k": self.k}


# ============================================================================
# Treatment Class
# ============================================================================


@dataclass
class Treatment:
    """Treatment channel specification for Transfer Function ITS.

    A treatment channel represents a time-varying intervention (e.g., media spend,
    policy intensity) along with its transformation pipeline.

    Parameters
    ----------
    name : str
        Column name in the data DataFrame containing the raw exposure series.
    saturation : SaturationTransform, optional
        Saturation transform to apply (e.g., HillSaturation, LogisticSaturation).
        Default is None (no saturation).
    adstock : AdstockTransform, optional
        Adstock transform to apply (e.g., GeometricAdstock).
        Default is None (no adstock).
    lag : LagTransform, optional
        Lag transform to apply (e.g., DiscreteLag).
        Default is None (no lag).
    coef_constraint : str, default="nonnegative"
        Constraint on the treatment coefficient. Options:
        - "nonnegative": Coefficient must be >= 0 (typical for media effects)
        - "unconstrained": No constraint on coefficient sign

    Examples
    --------
    >>> # Communication intensity with Hill saturation and 4-week adstock
    >>> comm = Treatment(
    ...     name="comm_intensity",
    ...     saturation=HillSaturation(slope=2.0, kappa=5),
    ...     adstock=GeometricAdstock(half_life=4, normalize=True),
    ...     coef_constraint="unconstrained",
    ... )

    >>> # Simple treatment with adstock only
    >>> promo = Treatment(
    ...     name="promo_intensity",
    ...     adstock=GeometricAdstock(half_life=2),
    ... )

    Notes
    -----
    Transforms are applied in the order: Saturation → Adstock → Lag.
    This ordering reflects the typical causal sequence: first, exposure saturates,
    then the saturated effect carries over across time, and finally an optional
    delay can be applied.

    The strategy pattern allows easy extension with new transform types without
    modifying this class or the TransferFunctionITS implementation.
    """

    name: str
    saturation: SaturationTransform | None = None
    adstock: AdstockTransform | None = None
    lag: LagTransform | None = None
    coef_constraint: str = "nonnegative"

    def __post_init__(self):
        """Validate treatment specification."""
        if self.coef_constraint not in ["nonnegative", "unconstrained"]:
            raise ValueError(
                f"coef_constraint must be 'nonnegative' or 'unconstrained', "
                f"got '{self.coef_constraint}'"
            )
