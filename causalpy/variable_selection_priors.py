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
Generic variable selection priors for PyMC models using pymc-extras Prior class.

This module provides reusable prior specifications that can be applied to any
PyMC model with coefficient vectors (beta parameters). Supports spike-and-slab
and horseshoe priors for automatic variable selection and shrinkage, built on
top of the pymc-extras Prior infrastructure.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc_extras.prior import Prior


def _relaxed_bernoulli_transform(
    p: Union[float, pt.TensorVariable], temperature: float = 0.1
):
    """
    Transform function for relaxed (continuous) Bernoulli distribution.

    This provides a continuous approximation to a Bernoulli distribution,
    useful for gradient-based inference. As temperature → 0, this approaches
    a true binary distribution.

    Parameters
    ----------
    p : float or PyMC variable
        Probability parameter
    temperature : float, default=0.1
        Temperature parameter (lower = more binary)

    Returns
    -------
    function
        Transform function that takes uniform random variable
    """

    def transform(u):
        logit_p = pt.log(p) - pt.log(1 - p)
        return pm.math.sigmoid((logit_p + pt.log(u) - pt.log(1 - u)) / temperature)

    return transform


class SpikeAndSlabPrior:
    """
    Spike-and-slab prior using pymc-extras Prior class.

    Creates a mixture prior with a point mass at zero (spike) and a diffuse
    normal distribution (slab), implemented as:

    β_j = γ_j × β_j^raw

    where γ_j ∈ [0,1] is a relaxed indicator and β_j^raw ~ N(0, σ_slab²).

    Parameters
    ----------
    pi_alpha : float, default=2
        Beta prior alpha for selection probability
    pi_beta : float, default=2
        Beta prior beta for selection probability
    slab_sigma : float, default=2
        Standard deviation of slab (non-zero) component
    temperature : float, default=0.1
        Relaxation parameter for binary approximation (lower = more binary)
    dims : str or tuple, optional
        Dimension names for the coefficient vector

    Example
    -------
    >>> spike_slab = SpikeAndSlabPrior(dims="features")
    >>> with pm.Model():
    ...     beta = spike_slab.create_variable("beta")
    """

    def __init__(
        self,
        pi_alpha: float = 2,
        pi_beta: float = 2,
        slab_sigma: float = 2,
        temperature: float = 0.1,
        dims: Optional[Union[str, tuple]] = None,
    ):
        self.pi_alpha = pi_alpha
        self.pi_beta = pi_beta
        self.slab_sigma = slab_sigma
        self.temperature = temperature
        self.dims = dims if isinstance(dims, tuple) or dims is None else (dims,)

    def create_variable(self, name: str) -> pm.Deterministic:
        """
        Create spike-and-slab variable.

        Parameters
        ----------
        name : str
            Name for the coefficient vector

        Returns
        -------
        pm.Deterministic
            Coefficient vector with spike-and-slab prior
        """
        # Selection probability using Prior class
        pi_prior = Prior("Beta", alpha=self.pi_alpha, beta=self.pi_beta)
        pi = pi_prior.create_variable(f"pi_{name}")

        # Raw coefficients (slab component) using Prior class
        slab_prior = Prior("Normal", mu=0, sigma=self.slab_sigma, dims=self.dims)
        beta_raw = slab_prior.create_variable(f"{name}_raw")

        # Selection indicators using relaxed Bernoulli
        # We use Uniform and transform it
        u = pm.Uniform(f"gamma_{name}_u", 0, 1, dims=self.dims)
        transform_fn = _relaxed_bernoulli_transform(pi, self.temperature)
        gamma = pm.Deterministic(f"gamma_{name}", transform_fn(u), dims=self.dims)

        # Actual coefficients
        return pm.Deterministic(name, gamma * beta_raw, dims=self.dims)


class HorseshoePrior:
    """
    Regularized horseshoe prior using pymc-extras Prior class.

    Provides continuous shrinkage with heavy tails, allowing strong signals
    to escape shrinkage while weak signals are dampened:

    β_j = τ · λ̃_j · β_j^raw

    where λ̃_j = √(c²λ_j² / (c² + τ²λ_j²)) is the regularized local shrinkage.

    Parameters
    ----------
    tau0 : float, optional
        Global shrinkage parameter. If None, computed from data.
    nu : float, default=3
        Degrees of freedom for half-t prior on tau
    c2_alpha : float, default=2
        InverseGamma alpha for regularization parameter
    c2_beta : float, default=2
        InverseGamma beta for regularization parameter
    dims : str or tuple, optional
        Dimension names for the coefficient vector

    Example
    -------
    >>> horseshoe = HorseshoePrior(dims="features")
    >>> with pm.Model():
    ...     beta = horseshoe.create_variable("beta")
    """

    def __init__(
        self,
        tau0: Optional[float] = None,
        nu: float = 3,
        c2_alpha: float = 2,
        c2_beta: float = 2,
        dims: Optional[Union[str, tuple]] = None,
    ):
        self.tau0 = tau0
        self.nu = nu
        self.c2_alpha = c2_alpha
        self.c2_beta = c2_beta
        self.dims = dims if isinstance(dims, tuple) or dims is None else (dims,)

    def create_variable(self, name: str) -> pm.Deterministic:
        """
        Create horseshoe variable.

        Parameters
        ----------
        name : str
            Name for the coefficient vector

        Returns
        -------
        pm.Deterministic
            Coefficient vector with horseshoe prior
        """
        # Global shrinkage using Prior class
        tau_prior = Prior("HalfStudentT", nu=self.nu, sigma=self.tau0 or 1.0)
        tau = tau_prior.create_variable(f"tau_{name}")

        # Local shrinkage parameters using Prior class
        lambda_prior = Prior("HalfCauchy", beta=1.0, dims=self.dims)
        lambda_ = lambda_prior.create_variable(f"lambda_{name}")

        # Regularization parameter using Prior class
        c2_prior = Prior("InverseGamma", alpha=self.c2_alpha, beta=self.c2_beta)
        c2 = c2_prior.create_variable(f"c2_{name}")

        # Regularized local shrinkage
        lambda_tilde = pm.Deterministic(
            f"lambda_tilde_{name}",
            pm.math.sqrt(c2 * lambda_**2 / (c2 + tau**2 * lambda_**2)),
            dims=self.dims,
        )

        # Raw coefficients using Prior class
        raw_prior = Prior("Normal", mu=0, sigma=1, dims=self.dims)
        beta_raw = raw_prior.create_variable(f"{name}_raw")

        # Actual coefficients
        return pm.Deterministic(name, beta_raw * lambda_tilde * tau, dims=self.dims)


class VariableSelectionPrior:
    """
    Factory for creating variable selection priors on coefficient vectors.

    This class provides a unified interface for different types of variable
    selection priors that can be applied to any beta coefficient in a PyMC model.
    Built on top of pymc-extras Prior class for consistency and interoperability.

    Supported prior types:
    - 'spike_and_slab': Mixture prior with near-zero spike and diffuse slab
    - 'horseshoe': Continuous shrinkage with adaptive regularization
    - 'normal': Standard normal prior (no selection, for comparison)

    Parameters
    ----------
    prior_type : str
        Type of prior: 'spike_and_slab', 'horseshoe', or 'normal'
    hyperparams : dict, optional
        Hyperparameters specific to the chosen prior type. If None, defaults are used.

        For 'spike_and_slab':
            - pi_alpha: float (default=2) - Beta prior alpha for selection probability
            - pi_beta: float (default=2) - Beta prior beta for selection probability
            - slab_sigma: float (default=2) - SD of slab (non-zero) component
            - temperature: float (default=0.1) - Relaxation parameter for binary approximation

        For 'horseshoe':
            - tau0: float (default=None) - Global shrinkage, auto-computed if None
            - nu: float (default=3) - Degrees of freedom for half-t prior on tau
            - c2_alpha: float (default=2) - InverseGamma alpha for regularization
            - c2_beta: float (default=2) - InverseGamma beta for regularization

        For 'normal':
            - mu: float or array (default=0) - Prior mean
            - sigma: float or array (default=1) - Prior SD

    Example
    -------
    >>> import pymc as pm
    >>> from variable_selection_priors import VariableSelectionPrior
    >>>
    >>> # Create spike-and-slab prior
    >>> vs_prior = VariableSelectionPrior("spike_and_slab")
    >>>
    >>> with pm.Model() as model:
    ...     # Create coefficients with variable selection
    ...     beta = vs_prior.create_prior(
    ...         name="beta",
    ...         n_params=5,
    ...         dims="features",
    ...         X=X_train,  # For computing tau0 in horseshoe
    ...     )
    """

    def __init__(self, prior_type: str, hyperparams: Optional[Dict[str, Any]] = None):
        """Initialize the variable selection prior factory."""
        self.prior_type = prior_type.lower()
        self.hyperparams = hyperparams or {}

        if self.prior_type not in ["spike_and_slab", "horseshoe", "normal"]:
            raise ValueError(
                f"Unknown prior_type: {prior_type}. "
                "Must be 'spike_and_slab', 'horseshoe', or 'normal'"
            )

        # Will be set when create_prior is called
        self._prior_instance = None

    def _get_default_hyperparams(
        self, n_params: int, X: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Get default hyperparameters for the chosen prior type.

        Parameters
        ----------
        n_params : int
            Number of parameters (dimension of beta vector)
        X : array-like, optional
            Design matrix for computing data-adaptive defaults (horseshoe only)

        Returns
        -------
        dict
            Default hyperparameters
        """
        if self.prior_type == "spike_and_slab":
            return {
                "pi_alpha": 2,
                "pi_beta": 2,
                "slab_sigma": 2,
                "temperature": 0.1,
            }

        elif self.prior_type == "horseshoe":
            # Compute tau0 using rule of thumb from Piironen & Vehtari (2017)
            if X is not None:
                p = n_params
                p0 = min(5.0, p / 2)  # Expected number of nonzero coefficients
                sigma_est = 1.0
                n = X.shape[0]
                tau0 = (p0 / (p - p0)) * (sigma_est / np.sqrt(n))
            else:
                # Fallback if no data provided
                tau0 = 1.0 / np.sqrt(n_params)

            return {
                "tau0": tau0,
                "nu": 3,
                "c2_alpha": 2,
                "c2_beta": 2,
            }

        else:  # normal
            return {
                "mu": 0,
                "sigma": 1,
            }

    def create_prior(
        self,
        name: str,
        n_params: int,
        dims: Optional[Union[str, tuple]] = None,
        X: Optional[np.ndarray] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> Union[pm.Deterministic, pm.Distribution]:
        """
        Create the specified prior on a coefficient vector.

        This is the main method to use. It creates the appropriate prior type
        based on the configuration and returns the PyMC variable.

        Parameters
        ----------
        name : str
            Name for the coefficient vector (e.g., 'beta', 'b', 'coef')
        n_params : int
            Number of parameters (length of coefficient vector)
        dims : str or tuple, optional
            Dimension name(s) for the coefficient vector
        X : array-like, optional
            Design matrix for computing data-adaptive hyperparameters
            (used only for horseshoe priors)
        hyperparams : dict, optional
            Override default hyperparameters for this specific prior instance

        Returns
        -------
        PyMC variable
            The coefficient vector with the specified prior

        Example
        -------
        >>> vs_prior = VariableSelectionPrior("horseshoe")
        >>> with pm.Model() as model:
        ...     beta = vs_prior.create_prior(
        ...         "beta", n_params=10, dims="features", X=X_train
        ...     )
        """
        # Merge instance and call-specific hyperparameters
        default_hp = self._get_default_hyperparams(n_params, X)
        merged_hp = {**default_hp, **self.hyperparams}
        if hyperparams:
            merged_hp.update(hyperparams)

        # Normalize dims
        if isinstance(dims, str):
            dims = (dims,)

        # Create the appropriate prior
        if self.prior_type == "spike_and_slab":
            self._prior_instance = SpikeAndSlabPrior(
                pi_alpha=merged_hp["pi_alpha"],
                pi_beta=merged_hp["pi_beta"],
                slab_sigma=merged_hp["slab_sigma"],
                temperature=merged_hp["temperature"],
                dims=dims,
            )
            return self._prior_instance.create_variable(name)

        elif self.prior_type == "horseshoe":
            self._prior_instance = HorseshoePrior(
                tau0=merged_hp["tau0"],
                nu=merged_hp["nu"],
                c2_alpha=merged_hp["c2_alpha"],
                c2_beta=merged_hp["c2_beta"],
                dims=dims,
            )
            return self._prior_instance.create_variable(name)

        else:  # normal
            # Use Prior class directly for normal
            normal_prior = Prior(
                "Normal", mu=merged_hp["mu"], sigma=merged_hp["sigma"], dims=dims
            )
            return normal_prior.create_variable(name)

    def get_inclusion_probabilities(
        self, idata, param_name: str, threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Extract variable inclusion probabilities from fitted model.

        Only applicable for spike-and-slab priors. Returns the posterior
        probability that each coefficient is "selected" (non-zero).

        Parameters
        ----------
        idata : arviz.InferenceData
            Fitted model inference data
        param_name : str
            Name of the coefficient parameter (must match name in create_prior)
        threshold : float, default=0.5
            Threshold for considering a variable "selected"

        Returns
        -------
        dict
            Dictionary with keys:
            - 'probabilities': Array of inclusion probabilities per coefficient
            - 'selected': Boolean array indicating which are selected
            - 'gamma_mean': Mean of gamma (indicator) variables

        Raises
        ------
        ValueError
            If prior_type is not 'spike_and_slab' or gamma variables not found

        Example
        -------
        >>> result = vs_prior.get_inclusion_probabilities(idata, "beta")
        >>> print(f"Selected features: {result['selected']}")
        >>> print(f"Inclusion probs: {result['probabilities']}")
        """
        if self.prior_type != "spike_and_slab":
            raise ValueError(
                "Inclusion probabilities only available for 'spike_and_slab' priors"
            )

        gamma_name = f"gamma_{param_name}"

        if gamma_name not in idata.posterior:
            raise ValueError(
                f"Could not find '{gamma_name}' in posterior. "
                f"Make sure you used the correct parameter name."
            )

        import arviz as az

        # Extract gamma values
        gamma = az.extract(idata.posterior[gamma_name])

        # Compute inclusion probabilities
        probabilities = (gamma > threshold).mean(dim="sample").values
        gamma_mean = gamma.mean(dim="sample").values
        selected = probabilities > threshold

        return {
            "probabilities": probabilities,
            "selected": selected,
            "gamma_mean": gamma_mean,
        }

    def get_shrinkage_factors(self, idata, param_name: str) -> Dict[str, np.ndarray]:
        """
        Extract shrinkage factors from horseshoe prior.

        Only applicable for horseshoe priors. Returns the effective shrinkage
        applied to each coefficient: κ_j = τ · λ̃_j

        Parameters
        ----------
        idata : arviz.InferenceData
            Fitted model inference data
        param_name : str
            Name of the coefficient parameter

        Returns
        -------
        dict
            Dictionary with keys:
            - 'shrinkage_factors': Array of shrinkage factors per coefficient
            - 'tau': Global shrinkage parameter
            - 'lambda_tilde': Regularized local shrinkage parameters

        Raises
        ------
        ValueError
            If prior_type is not 'horseshoe' or required variables not found

        Example
        -------
        >>> result = vs_prior.get_shrinkage_factors(idata, "beta")
        >>> print(f"Shrinkage factors: {result['shrinkage_factors']}")
        """
        if self.prior_type != "horseshoe":
            raise ValueError("Shrinkage factors only available for 'horseshoe' priors")

        import arviz as az

        tau_name = f"tau_{param_name}"
        lambda_tilde_name = f"lambda_tilde_{param_name}"

        if tau_name not in idata.posterior:
            raise ValueError(f"Could not find '{tau_name}' in posterior")
        if lambda_tilde_name not in idata.posterior:
            raise ValueError(f"Could not find '{lambda_tilde_name}' in posterior")

        # Extract components
        tau = az.extract(idata.posterior[tau_name])
        lambda_tilde = az.extract(idata.posterior[lambda_tilde_name])

        # Compute shrinkage factors
        shrinkage_factors = (tau * lambda_tilde).mean(dim="sample").values

        return {
            "shrinkage_factors": shrinkage_factors,
            "tau": tau.mean().values,
            "lambda_tilde": lambda_tilde.mean(dim="sample").values,
        }


def create_variable_selection_prior(
    prior_type: str,
    name: str,
    n_params: int,
    dims: Optional[Union[str, tuple]] = None,
    X: Optional[np.ndarray] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> Union[pm.Deterministic, pm.Distribution]:
    """
    Convenience function to create a variable selection prior in one call.

    This is a shorthand for creating a VariableSelectionPrior instance and
    calling create_prior() in one step.

    Parameters
    ----------
    prior_type : str
        Type of prior: 'spike_and_slab', 'horseshoe', or 'normal'
    name : str
        Name for the coefficient vector
    n_params : int
        Number of parameters
    dims : str or tuple, optional
        Dimension name(s)
    X : array-like, optional
        Design matrix for data-adaptive hyperparameters
    hyperparams : dict, optional
        Custom hyperparameters

    Returns
    -------
    PyMC variable
        The coefficient vector with specified prior

    Example
    -------
    >>> with pm.Model() as model:
    ...     X = pm.Data("X", X_train)
    ...     beta = create_variable_selection_prior(
    ...         "spike_and_slab", "beta", n_params=X_train.shape[1], dims="features"
    ...     )
    ...     mu = pm.math.dot(X, beta)
    """
    vs_prior = VariableSelectionPrior(prior_type, hyperparams)
    return vs_prior.create_prior(name, n_params, dims, X)
