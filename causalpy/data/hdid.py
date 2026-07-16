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
Functions that generate data for hierarchical difference-in-differences
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from causalpy.data.datasets import _DATA_DIR


@dataclass(slots=True, frozen=True)
class Region:
    """Region labels and additive fixed effects.

    Parameters
    ----------
    labels : tuple[str, ...]
        Region names used to materialize the `region` column.
    effects : tuple[float, ...]
        Region-level effects aligned with `labels` by position.
    """

    labels: tuple[str, ...] = ("north", "south", "east", "west")
    effects: tuple[float, ...] = (0.8, -0.3, 0.4, 0.0)


@dataclass(slots=True, frozen=True)
class Store:
    """Store-level simulation parameters.

    Parameters
    ----------
    units_min : int
        Lower clipping bound for unit counts per store.
    units_max : int
        Upper clipping bound for unit counts per store.
    units_distribution : str
        Random draw function name on `np.random.Generator` for unit counts (for example `"poisson"`).
    units_parameters : dict[str, float]
        Dictionary passed as `**kwargs` to `rng.<units_distribution>(...)` (for example `{"lam": 500.0}`).
    urban_distribution : str
        Random draw function name on `np.random.Generator` for urban flags (for example `"binomial"`).
    urban_parameters : dict[str, float]
        Dictionary passed as `**kwargs` to `rng.<urban_distribution>(...)`.
    size_distribution : str
        Random draw function name on `np.random.Generator` for store size (for example `"lognormal"`).
    size_parameters : dict[str, float]
        Dictionary passed as `**kwargs` to `rng.<size_distribution>(...)`.
    """

    units_min: int = 150
    units_max: int = 800
    units_distribution: str = "poisson"
    units_parameters: dict[str, float] = field(default_factory=lambda: {"lam": 500.0})
    urban_distribution: str = "binomial"
    urban_parameters: dict[str, float] = field(
        default_factory=lambda: {"n": 1, "p": 0.60}
    )
    size_distribution: str = "lognormal"
    size_parameters: dict[str, float] = field(
        default_factory=lambda: {"mean": 4.0, "sigma": 0.35}
    )

    def sample(self, rng: np.random.Generator, prefix: str, size: int) -> np.ndarray:
        """Sample store attributes using a configured distribution.

        Parameters
        ----------
        rng : np.random.Generator
            Random generator used for deterministic reproducibility.
        prefix : str
            Name of the variable to sample. For example, `prefix="size"` uses `size_distribution` and `size_parameters`.
        size : int
            Number of values to draw.

        Returns
        -------
        np.ndarray
            Drawn values for the requested store attribute.
        """

        distribution = getattr(self, f"{prefix}_distribution")
        params = getattr(self, f"{prefix}_parameters")
        return getattr(rng, distribution)(size=size, **params)


@dataclass(slots=True, frozen=True)
class Customer:
    """Customer-level simulation parameters.

    Parameters
    ----------
    age_min : float
        Lower clipping bound for sampled ages.
    age_max : float
        Upper clipping bound for sampled ages.
    age_distribution : str
        Random draw function name on `np.random.Generator` for age (for example `"normal"`).
    age_parameters : dict[str, float]
        Dictionary passed as `**kwargs` to `rng.<age_distribution>(...)`.
    tenure_min : float
        Lower clipping bound for sampled tenure.
    tenure_max : float
        Upper clipping bound for sampled tenure.
    tenure_distribution : str
        Random draw function name on `np.random.Generator` for tenure (for example `"gamma"`).
    tenure_parameters : dict[str, float]
        Dictionary passed as `**kwargs` to `rng.<tenure_distribution>(...)`.
    """

    age_min: float = 18.0
    age_max: float = 90.0
    age_distribution: str = "normal"
    age_parameters: dict[str, float] = field(
        default_factory=lambda: {"loc": 42.0, "scale": 12.0}
    )
    tenure_min: float = 0.2
    tenure_max: float = 25.0
    tenure_distribution: str = "gamma"
    tenure_parameters: dict[str, float] = field(
        default_factory=lambda: {"shape": 2.4, "scale": 2.2}
    )

    def sample(self, rng: np.random.Generator, prefix: str, size: int) -> np.ndarray:
        """Sample customer attributes from configured distributions.

        Parameters
        ----------
        rng : np.random.Generator
            Random generator used for deterministic reproducibility.
        prefix : str
            Name of the variable to sample. For example, `prefix="age"` uses `age_distribution` and `age_parameters`.
        size : int
            Number of values to draw.

        Returns
        -------
        np.ndarray
            Drawn values for the requested customer attribute.
        """

        distribution = getattr(self, f"{prefix}_distribution")
        params = getattr(self, f"{prefix}_parameters")
        return getattr(rng, distribution)(size=size, **params)


@dataclass(slots=True, frozen=True)
class History:
    """Parameters for generating customer baseline pre-treatment purchase history.

    Parameters
    ----------
    base : float
        Baseline level in the deterministic history mean (`history_mu`).
    store_intercept_weight : float
        Weight on the store random intercept in `history_mu`.
    urban_weight : float
        Weight on the urban indicator in `history_mu`.
    noise_distribution : str
        Random draw function name on `np.random.Generator` for history noise (for example `"normal"`).
    noise_parameters : dict[str, float]
        Dictionary passed as `**kwargs` to `rng.<noise_distribution>(...)`.
    minimum : float
        Lower clipping bound applied after adding history noise.
    """

    base: float = 120.0
    store_intercept_weight: float = 4.0
    urban_weight: float = 6.0
    noise_distribution: str = "normal"
    noise_parameters: dict[str, float] = field(
        default_factory=lambda: {"loc": 0.0, "scale": 25.0}
    )
    minimum: float = 20.0

    def sample(self, rng: np.random.Generator, prefix: str, size: int) -> np.ndarray:
        """Sample stochastic components for baseline history generation.

        Parameters
        ----------
        rng : np.random.Generator
            Random generator used for deterministic reproducibility.
        prefix : str
            Name of the variable to sample. For example, `prefix="noise"` uses `noise_distribution` and `noise_parameters`.
        size : int
            Number of values to draw.

        Returns
        -------
        np.ndarray
            Drawn values for the requested history component.
        """

        distribution = getattr(self, f"{prefix}_distribution")
        params = getattr(self, f"{prefix}_parameters")
        return getattr(rng, distribution)(size=size, **params)


@dataclass(slots=True, frozen=True)
class Outcome:
    """Outcome-model coefficients and stochastic terms.

    Parameters
    ----------
    intercept : float
        Baseline intercept for the conditional mean.
    trend_weight : float
        Common linear month trend shared across groups.
    post_effect : float
        Common post-period level shift (set to `0.0` by default so controls do not jump at intervention time).
    treated_effect : float
        Baseline treated-vs-control level difference.
    att_effect : float
        Average treatment effect for treated stores in post periods.
    age_weight : float
        Coefficient for centered age.
    age_center : float
        Age-centering constant.
    tenure_weight : float
        Coefficient for customer tenure.
    history_weight : float
        Coefficient for centered purchase history.
    history_center : float
        Purchase-history centering constant.
    log_store_size_weight : float
        Coefficient for log store size.
    urban_weight : float
        Coefficient for urban store indicator.
    noise_distribution : str
        Random draw function name on `np.random.Generator` for observation noise (for example `"normal"`).
    noise_parameters : dict[str, float]
        Dictionary passed as `**kwargs` to `rng.<noise_distribution>(...)`.
    store_intercept_distribution : str
        Random draw function name on `np.random.Generator` for store random intercepts (for example `"normal"`).
    store_intercept_parameters : dict[str, float]
        Dictionary passed as `**kwargs` to `rng.<store_intercept_distribution>(...)`.
    treatment_slope_distribution : str
        Random draw function name on `np.random.Generator` for store treatment-slope deviations.
    treatment_slope_parameters : dict[str, float]
        Dictionary passed as `**kwargs` to `rng.<treatment_slope_distribution>(...)`.
    """

    intercept: float = 80.0
    trend_weight: float = 0.55
    post_effect: float = 0.0
    treated_effect: float = 2.2
    att_effect: float = 4.0
    age_weight: float = 0.10
    age_center: float = 40.0
    tenure_weight: float = 0.75
    history_weight: float = 0.06
    history_center: float = 120.0
    log_store_size_weight: float = 0.30
    urban_weight: float = 0.80
    noise_distribution: str = "normal"
    noise_parameters: dict[str, float] = field(
        default_factory=lambda: {"loc": 0.0, "scale": 4.0}
    )
    store_intercept_distribution: str = "normal"
    store_intercept_parameters: dict[str, float] = field(
        default_factory=lambda: {"loc": 0.0, "scale": 1.3}
    )
    treatment_slope_distribution: str = "normal"
    treatment_slope_parameters: dict[str, float] = field(
        default_factory=lambda: {"loc": 0.0, "scale": 0.45}
    )

    def sample(self, rng: np.random.Generator, prefix: str, size: int) -> np.ndarray:
        """Sample outcome-related random components.

        Parameters
        ----------
        rng : np.random.Generator
            Random generator used for deterministic reproducibility.
        prefix : str
            Name of the variable to sample. For example, `prefix="store_intercept"` uses `store_intercept_distribution` and `store_intercept_parameters`.
        size : int
            Number of values to draw.

        Returns
        -------
        np.ndarray
            Drawn values for the requested outcome component.
        """

        distribution = getattr(self, f"{prefix}_distribution")
        params = getattr(self, f"{prefix}_parameters")
        return getattr(rng, distribution)(size=size, **params)


@dataclass(slots=True)
class Config:
    """Top-level configuration for hierarchical DiD simulation.

    Parameters
    ----------
    seed : int
        Random seed used to initialize reproducible draws.
    n_stores_total : int
        Total number of stores in the panel.
    n_stores_treated : int
        Number of treated stores.
    n_months : int
        Number of periods per customer.
    pre_months : int
        Number of pre-treatment periods.
    region : Region
        Region-level parameter object.
    store : Store
        Store-level parameter object.
    customer : Customer
        Customer-level parameter object.
    history : History
        Pre-period history parameter object.
    outcome : Outcome
        Outcome-model parameter object.
    output_path : Path
        Default output location for the simulated panel dataset.
    run_validation : bool
        Whether to run panel validation after generation.
    """

    seed: int = 656
    n_stores_total: int = 200
    n_stores_treated: int = 100
    n_months: int = 12
    pre_months: int = 6
    region: Region = field(default_factory=Region)
    store: Store = field(default_factory=Store)
    customer: Customer = field(default_factory=Customer)
    history: History = field(default_factory=History)
    outcome: Outcome = field(default_factory=Outcome)
    output_path: Path = _DATA_DIR / "hdid_data.csv.gz"
    run_validation: bool = True
    store_ids: np.ndarray = field(init=False, repr=False)
    treated_idx: np.ndarray = field(init=False, repr=False)
    treated: np.ndarray = field(init=False, repr=False)
    region_idx: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build derived index arrays used by the simulator.

        Raises
        ------
        ValueError
            If `n_stores_treated` is greater than or equal to `n_stores_total`.
        """

        if self.n_stores_treated >= self.n_stores_total:
            raise ValueError("n_stores_treated must be smaller than n_stores_total")

        rng = np.random.default_rng(seed=self.seed)

        self.store_ids = np.arange(1, self.n_stores_total + 1, dtype=np.int32)

        self.region_idx = rng.integers(
            low=0,
            high=len(self.region.labels),
            size=self.n_stores_total,
            dtype=np.int16,
        )

        self.treated_idx = rng.choice(
            a=self.n_stores_total, size=self.n_stores_treated, replace=False
        ).astype(np.int32)

        self.treated = np.isin(np.arange(self.n_stores_total), self.treated_idx).astype(
            np.int8
        )


class HDiDSimulator:
    r"""Hierarchical DiD panel simulator.

    Draws a balanced panel with treatment assigned at the store level.
    In simplified notation, the simulated outcome for customer \(i\) in store \(g\) at month \(t\) is

    .. math::

        Y_{igt} = \alpha + b_g + \gamma t + \lambda Post_t + \delta Treat_g
        + (\tau + u_g) Post_t Treat_g + x_{igt}^{\top}\beta + r_g + \epsilon_{igt}

    where \(b_g\) is a store random intercept, \(u_g\) is a store-level
    treatment-slope deviation, \(\tau\) is the population ATT, \(r_g\) is
    the region effect, and \(\epsilon_{igt}\) is observation noise. The
    covariate term \(x_{igt}^{\top}\beta\) expands to customer age, customer
    tenure, pre-period purchase history, log store size, and urban status.

    Parameters
    ----------
    config : Config
        Fully specified simulation configuration.
    """

    def __init__(self, config: Config):
        """Initialize simulator state and validate shared configuration.

        Parameters
        ----------
        config : Config
            Simulation configuration object.

        Raises
        ------
        ValueError
            If region labels and effects have mismatched lengths.
        """

        self.config = config
        self.rng = np.random.default_rng(seed=config.seed)
        self.region = config.region
        self.store = config.store
        self.customer = config.customer
        self.history = config.history
        self.outcome = config.outcome
        self.region_labels = np.array(self.region.labels, dtype=object)
        self.region_effects = np.array(self.region.effects, dtype=np.float32)
        if len(self.region_labels) != len(self.region_effects):
            raise ValueError("region.labels and region.effects must have equal length")
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate high-level simulator configuration constraints.

        Raises
        ------
        ValueError
            If treated-store count, time settings, or clipping bounds are invalid.
        """

        if self.config.n_stores_treated >= self.config.n_stores_total:
            raise ValueError("n_stores_treated must be smaller than n_stores_total")
        if not 1 <= self.config.pre_months < self.config.n_months:
            raise ValueError(
                "pre_months must be >=1 and strictly smaller than n_months"
            )
        if self.store.units_min > self.store.units_max:
            raise ValueError("store.units_min cannot exceed store.units_max")

    def _draw_store_level(self) -> dict[str, np.ndarray]:
        """Draw store-level latent variables and covariates.

        Returns
        -------
        dict[str, np.ndarray]
            Store-level arrays used to expand the unit-level panel.
        """

        store_ids = self.config.store_ids

        region_idx = self.config.region_idx

        treated = self.config.treated

        _n_units = self.store.sample(
            rng=self.rng, prefix="units", size=self.config.n_stores_total
        )
        n_units = np.clip(_n_units, self.store.units_min, self.store.units_max).astype(
            np.int32
        )

        urban = self.store.sample(
            rng=self.rng, prefix="urban", size=self.config.n_stores_total
        ).astype(np.int8)
        store_size = self.store.sample(
            rng=self.rng, prefix="size", size=self.config.n_stores_total
        ).astype(np.float32)
        store_intercept = self.outcome.sample(
            rng=self.rng, prefix="store_intercept", size=self.config.n_stores_total
        ).astype(np.float32)
        store_treatment_slope = self.outcome.sample(
            rng=self.rng, prefix="treatment_slope", size=self.config.n_stores_total
        ).astype(np.float32)

        return {
            "store_id": store_ids,
            "treated": treated,
            "n_units": n_units,
            "region_idx": region_idx,
            "urban": urban,
            "store_size": store_size,
            "store_intercept": store_intercept,
            "store_treatment_slope": store_treatment_slope,
        }

    def _draw_unit_level(self, stores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        r"""Draw customer-level covariates nested within stores.

        Creates the customer-level part of the HDiD DGP before the
        panel is expanded over months. For customer \(i\) in store \(g\),
        it draws age, tenure, and pre-period purchase history, then attaches
        the store-level treatment assignment, region, store size, random
        intercept \(b_g\), and treatment-slope deviation \(u_g\).

        The pre-period history covariate is generated conditionally on store
        characteristics:

        .. math::

            H_{ig} = h_0 + h_b b_g + h_u Urban_g + \eta_{ig}

        then clipped at ``history.minimum`` and stored as
        ``pre_purchase_history``. This value later enters the simulated
        outcome through the covariate term \(x_{igt}^{\top}\beta\).

        Parameters
        ----------
        stores : dict[str, np.ndarray]
            Store-level arrays produced by :meth:`_draw_store_level`, including
            treatment assignment, region, store size, store intercepts, treatment
            slope deviations, and customer counts per store.

        Returns
        -------
        dict[str, np.ndarray]
            Customer-level arrays before month expansion. Each returned array has
            one row per customer and carries both customer covariates and the
            store-level quantities needed by :meth:`simulate`.
        """

        store_idx_for_units = np.repeat(
            np.arange(self.config.n_stores_total, dtype=np.int32), stores["n_units"]
        )

        n_units_total = int(store_idx_for_units.size)

        urban = stores["urban"][store_idx_for_units]

        store_intercept = stores["store_intercept"][store_idx_for_units]

        customer_age = np.clip(
            self.customer.sample(rng=self.rng, prefix="age", size=n_units_total),
            self.customer.age_min,
            self.customer.age_max,
        ).astype(np.float32)
        customer_tenure = np.clip(
            self.customer.sample(rng=self.rng, prefix="tenure", size=n_units_total),
            self.customer.tenure_min,
            self.customer.tenure_max,
        ).astype(np.float32)
        history_mu = (
            self.history.base
            + self.history.store_intercept_weight * store_intercept
            + self.history.urban_weight * urban
        )
        pre_purchase_history = np.clip(
            history_mu
            + self.history.sample(rng=self.rng, prefix="noise", size=n_units_total),
            self.history.minimum,
            None,
        ).astype(np.float32)

        customer_id = np.array(
            [
                f"S{store_id_:03d}_U{unit_idx:04d}"
                for store_id_, n_units_ in zip(
                    stores["store_id"], stores["n_units"], strict=False
                )
                for unit_idx in range(1, int(n_units_) + 1)
            ],
            dtype=object,
        )

        return {
            "store_id": stores["store_id"][store_idx_for_units],
            "customer_id": customer_id,
            "treated": stores["treated"][store_idx_for_units],
            "region_idx": stores["region_idx"][store_idx_for_units],
            "urban": urban,
            "store_size": stores["store_size"][store_idx_for_units],
            "store_intercept": store_intercept,
            "store_treatment_slope": stores["store_treatment_slope"][
                store_idx_for_units
            ],
            "customer_age": customer_age,
            "customer_tenure": customer_tenure,
            "pre_purchase_history": pre_purchase_history,
        }

    def simulate(self) -> pd.DataFrame:
        r"""Generate the balanced customer-store-month HDiD panel.

        The method composes the full simulator DGP in three stages:

        1. draw store-level treatment assignment, region, store size, random
           intercepts \(b_g\), and treatment-slope deviations \(u_g\);
        2. draw customers nested in those stores with covariates
           \(x_{igt}\), including pre-period purchase history;
        3. expand each customer over months and draw the outcome from

        .. math::

            Y_{igt} = \alpha + b_g + \gamma t + \lambda Post_t + \delta Treat_g
            + (\tau + u_g) Post_t Treat_g + x_{igt}^{\top}\beta + r_g
            + \epsilon_{igt}.

        The returned ``purchase_amount`` is this latent mean plus observation
        noise, rounded to two decimals. Treatment is store-level,
        ``post_treatment`` is determined by ``config.pre_months``, and the
        generated panel is balanced by customer and month.

        Returns
        -------
        pd.DataFrame
            Simulated customer-month panel with identifiers, treatment flags,
            outcome, customer covariates, store covariates, and region labels.

        Raises
        ------
        RuntimeError
            If generated data fails panel validity checks.
        """

        stores = self._draw_store_level()
        units = self._draw_unit_level(stores=stores)

        n_units_total = len(units["customer_id"])
        months = np.arange(1, self.config.n_months + 1, dtype=np.int16)
        month = np.tile(months, n_units_total)
        post_treatment = (month > self.config.pre_months).astype(np.int8)

        store_id = np.repeat(units["store_id"], self.config.n_months)
        customer_id = np.repeat(units["customer_id"], self.config.n_months)
        treated = np.repeat(units["treated"], self.config.n_months)
        urban = np.repeat(units["urban"], self.config.n_months)
        store_size = np.repeat(units["store_size"], self.config.n_months)
        store_intercept = np.repeat(units["store_intercept"], self.config.n_months)
        store_treatment_slope = np.repeat(
            units["store_treatment_slope"], self.config.n_months
        )
        customer_age = np.repeat(units["customer_age"], self.config.n_months)
        customer_tenure = np.repeat(units["customer_tenure"], self.config.n_months)
        pre_purchase_history = np.repeat(
            units["pre_purchase_history"], self.config.n_months
        )
        region_idx = np.repeat(units["region_idx"], self.config.n_months)
        region_effect = self.region_effects[region_idx]

        mu = (
            self.outcome.intercept
            + store_intercept
            + self.outcome.trend_weight * month
            + self.outcome.post_effect * post_treatment
            + self.outcome.treated_effect * treated
            + (self.outcome.att_effect + store_treatment_slope)
            * post_treatment
            * treated
            + self.outcome.age_weight * (customer_age - self.outcome.age_center)
            + self.outcome.tenure_weight * customer_tenure
            + self.outcome.history_weight
            * (pre_purchase_history - self.outcome.history_center)
            + self.outcome.log_store_size_weight * np.log(store_size)
            + self.outcome.urban_weight * urban
            + region_effect
        )

        purchase_amount = np.round(
            mu + self.outcome.sample(rng=self.rng, prefix="noise", size=mu.shape[0]),
            2,
        ).astype(np.float32)

        panel = pd.DataFrame(
            {
                "store_id": store_id.astype(np.int32),
                "customer_id": customer_id,
                "month": month.astype(np.int16),
                "treated": treated.astype(np.int8),
                "post_treatment": post_treatment.astype(np.int8),
                "purchase_amount": purchase_amount,
                "customer_age": customer_age.astype(np.float32),
                "customer_tenure": customer_tenure.astype(np.float32),
                "pre_purchase_history": pre_purchase_history.astype(np.float32),
                "store_size": store_size.astype(np.float32),
                "region": pd.Categorical(
                    values=self.region_labels[region_idx],
                    categories=self.region_labels.tolist(),
                ),
                "urban": urban.astype(np.int8),
            }
        )

        if self.config.run_validation:
            self.validate_panel(panel=panel)

        return panel

    def validate_panel(self, panel: pd.DataFrame) -> None:
        """Validate structural assumptions for the simulated panel.

        Parameters
        ----------
        panel : pd.DataFrame
            Simulated panel dataset.

        Raises
        ------
        RuntimeError
            If the panel is unbalanced or treatment is not store-level.
        """

        periods_per_customer = panel.groupby(by="customer_id", observed=True)[
            "month"
        ].nunique()
        if (
            periods_per_customer.min() != self.config.n_months
            or periods_per_customer.max() != self.config.n_months
        ):
            raise RuntimeError("Generated panel is not balanced by unit")
        if panel.groupby(by="store_id", observed=True)["treated"].nunique().max() != 1:
            raise RuntimeError("Treatment assignment is not store-level")

    @staticmethod
    def estimate_icc(panel: pd.DataFrame) -> float:
        """Estimate empirical ICC from store-level outcome clustering.

        Parameters
        ----------
        panel : pd.DataFrame
            Simulated panel dataset.

        Returns
        -------
        float
            Ratio of between-store variance to total variance.
        """

        store_means_var = (
            panel.groupby(by="store_id", observed=True)["purchase_amount"].mean().var()
        )
        total_var = panel["purchase_amount"].var()
        return float(store_means_var / total_var)

    def save(self, panel: pd.DataFrame, path: Path | None = None) -> Path:
        """Persist simulated panel data to a compressed CSV file.

        Parameters
        ----------
        panel : pd.DataFrame
            Simulated panel dataset.
        path : Path | None, default=None
            Destination path. Uses `config.output_path` when omitted.

        Returns
        -------
        Path
            Resolved output path written to disk.
        """

        out_path = Path(path or self.config.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(path_or_buf=out_path, index=False, compression="gzip")
        return out_path
