# Placebo-in-time Analysis

## Overview

The `PlaceboAnalysis` class implements a placebo-in-time sensitivity analysis for causal inference experiments. This technique helps validate causal claims by testing whether the intervention effect appears in periods where no intervention actually occurred.

## When to Use

Use `PlaceboAnalysis` when you want to:

1. **Validate causal claims**: Test if your model would detect spurious effects in pre-intervention periods where no treatment occurred
2. **Check model specification**: Verify that your model isn't picking up pre-existing trends or patterns that could be mistaken for treatment effects
3. **Assess robustness**: Demonstrate that the observed effect is specific to the actual intervention period and not a general pattern in the data
4. **Strengthen inference**: Provide additional evidence that the treatment effect is real by showing no effects in placebo periods

## Implementation

Since this class is not yet part of the core library, you must define it in your code:

```python
from typing import Any, Callable
from pydantic import BaseModel
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import causalpy as cp
import pymc as pm
import arviz as az

logger = logging.getLogger(__name__)

class PlaceboAnalysis(BaseModel):
    """
    Run sensitivity analysis for any causalpy experiment using a factory pattern.

    The factory function allows complete flexibility in choosing and configuring
    any causalpy experiment type (SyntheticControl, InterruptedTimeSeries,
    DifferenceInDifferences, RegressionDiscontinuity, etc.).

    Parameters
    ----------
    experiment_factory : Callable
        A function that creates and returns a fitted causalpy experiment.
        Signature: (dataset: pd.DataFrame, treatment_time: pd.Timestamp,
                   treatment_time_end: pd.Timestamp) -> causalpy result
    dataset : pd.DataFrame
        The full dataset with datetime index
    intervention_start_date : str
        Start date of the intervention period (YYYY-MM-DD format)
    intervention_end_date : str
        End date of the intervention period (YYYY-MM-DD format)
    n_cuts : int
        Number of cuts for cross-validation (n_cuts - 1 folds will be created)
    """

    model_config = {"arbitrary_types_allowed": True}

    experiment_factory: Callable
    dataset: pd.DataFrame
    intervention_start_date: str
    intervention_end_date: str
    n_cuts: int = 2

    def _validate_cuts(self, n_cuts: int) -> None:
        """Validate that n_cuts is at least 2."""
        if n_cuts < 2:
            raise ValueError("n_cuts must be >= 2 (n_cuts - 1 folds will be created).")

    def _prepare_pre_data(self, treatment_time: pd.Timestamp) -> pd.DataFrame:
        """Extract pre-intervention data."""
        pre_df = self.dataset.loc[self.dataset.index < treatment_time].copy()
        if pre_df.empty:
            raise ValueError("No observations strictly before treatment_time in dataset.")
        return pre_df

    def _calculate_intervention_length(
        self, treatment_time: pd.Timestamp, treatment_time_end: pd.Timestamp
    ) -> pd.Timedelta:
        """Calculate the length of the intervention period."""
        treatment_time = pd.Timestamp(treatment_time)
        treatment_time_end = pd.Timestamp(treatment_time_end)
        intervention_length = treatment_time_end - treatment_time
        if intervention_length <= pd.Timedelta(0):
            raise ValueError("treatment_time_end must be after treatment_time to compute a positive intervention length.")
        return intervention_length

    def _validate_sufficient_data(
        self,
        pre_df: pd.DataFrame,
        treatment_time: pd.Timestamp,
        intervention_length: pd.Timedelta,
        n_cuts: int,
    ) -> None:
        """Validate that there's sufficient pre-intervention data for the requested folds."""
        pre_start = pre_df.index.min()
        earliest_needed = treatment_time - (n_cuts - 1) * intervention_length
        if pre_start > earliest_needed:
            max_cuts = 1 + int((treatment_time - pre_start) // intervention_length)
            raise ValueError(
                "Not enough pre-period for requested folds. "
                f"Earliest required: {earliest_needed.date()}, available starts: {pre_start.date()}. "
                f"Try n_cuts <= {max_cuts}."
            )

    def _create_fold_data(
        self,
        pre_df: pd.DataFrame,
        fold: int,
        n_cuts: int,
        treatment_time: pd.Timestamp,
        intervention_length: pd.Timedelta,
    ) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
        """Create data for a specific fold."""
        pseudo_start = treatment_time - (n_cuts - fold) * intervention_length
        pseudo_end = pseudo_start + intervention_length
        fold_df = pre_df.loc[pre_df.index < pseudo_end].sort_index()

        pre_mask = fold_df.index < pseudo_start
        post_mask = (fold_df.index >= pseudo_start) & (fold_df.index < pseudo_end)

        if pre_mask.sum() == 0 or post_mask.sum() == 0:
            raise ValueError(
                f"Fold {fold}: insufficient data. pre_n={pre_mask.sum()}, post_n={post_mask.sum()} "
                f"for window [{pseudo_start} .. {pseudo_end})."
            )

        return fold_df, pseudo_start, pseudo_end

    def _fit_model(
        self, fold_df: pd.DataFrame, pseudo_start: pd.Timestamp, pseudo_end: pd.Timestamp
    ) -> Any:
        """
        Fit the experiment using the provided factory function.
        """
        logger.info(f"Fitting model for fold with treatment_time={pseudo_start}, treatment_time_end={pseudo_end}")
        return self.experiment_factory(fold_df, pseudo_start, pseudo_end)

    def run(self) -> list[dict[str, Any]]:
        """
        Run the sensitivity analysis across all folds.
        """
        n_cuts = self.n_cuts
        treatment_time = pd.Timestamp(self.intervention_start_date)
        treatment_time_end = pd.Timestamp(self.intervention_end_date)

        self._validate_cuts(n_cuts)
        pre_df = self._prepare_pre_data(treatment_time)
        intervention_length = self._calculate_intervention_length(treatment_time, treatment_time_end)
        self._validate_sufficient_data(pre_df, treatment_time, intervention_length, n_cuts)

        results: list[dict[str, Any]] = []
        for fold in range(1, n_cuts):
            fold_df, pseudo_start, pseudo_end = self._create_fold_data(
                pre_df, fold, n_cuts, treatment_time, intervention_length
            )

            model_result = self._fit_model(fold_df, pseudo_start, pseudo_end)

            results.append(
                {
                    "fold": fold,
                    "pseudo_start": pseudo_start,
                    "pseudo_end": pseudo_end,
                    "result": model_result,
                }
            )

        return results
```

## Example Usage

```python
# 1. Define a factory function
def its_factory(dataset, treatment_time, treatment_time_end):
    formula = "target ~ 1 + feature"
    return cp.InterruptedTimeSeries(
        dataset,
        treatment_time,
        formula=formula,
        model=cp.pymc_models.LinearRegression(sample_kwargs={"random_seed": 42})
    )

# 2. Run sensitivity analysis
sensitivity = PlaceboAnalysis(
    experiment_factory=its_factory,
    dataset=df.set_index("date"),
    intervention_start_date="2024-01-01",
    intervention_end_date="2024-01-30",
    n_cuts=4
)
results = sensitivity.run()

# 3. Plot results
for r in results:
    r["result"].plot()
plt.show()
```

## Visualization: Posterior Cumulative Distribution

```python
# Extract and stack post-impact samples
sensitivity_post_impact = xr.concat(
    [
        r["result"]
        .post_impact.sum("obs_ind")   # sum over days in pseudo window
        .isel(treated_units=0)
        .stack(sample=("chain", "draw"))
        for r in results
    ],
    dim="fold",
)

# Convert sample coordinate for plotting
sensitivity_post_impact_numeric = sensitivity_post_impact.assign_coords(
    sample=np.arange(len(sensitivity_post_impact.sample))
)

# Plot histograms
n_folds = sensitivity_post_impact.sizes["fold"]
fold_means = [sensitivity_post_impact_numeric.isel(fold=i).mean().item() for i in range(n_folds)]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i in range(n_folds):
    fold_data = sensitivity_post_impact_numeric.isel(fold=i)
    fold_data.plot.hist(ax=ax, alpha=0.7, label=f'Fold {i} (mean: {fold_means[i]:.1f})')
ax.legend()
plt.show()
```

## Advanced: Hierarchical Status Quo Model

Build a hierarchical Bayesian model to characterize the "status quo" distribution of placebo effects.

```python
# 1. Extract summaries
n_folds = sensitivity_post_impact.sizes["fold"]
n_samples = sensitivity_post_impact.sizes["sample"]
fold_means = sensitivity_post_impact.mean(dim="sample").values
fold_sds = sensitivity_post_impact.std(dim="sample").values
fold_sds = np.where(fold_sds < 1e-6, 1e-6, fold_sds) # Guard against degenerate SDs

coords_meta = {"fold": np.arange(n_folds)}
prior_mu_center = float(np.nanmean(fold_means))
prior_mu_scale = float(np.nanstd(fold_means)) if np.nanstd(fold_means) > 0.0 else 1.0

# 2. Define and fit model
n_chains = 4
draws_per_chain_meta = n_samples // n_chains

with pm.Model(coords=coords_meta) as meta_status_quo_model:
    observed_fold_means = pm.Data("observed_fold_means", fold_means, dims="fold")
    observed_fold_sd = pm.Data("observed_fold_sd", fold_sds, dims="fold")

    mu_status_quo = pm.Normal("mu_status_quo", mu=prior_mu_center, sigma=5.0 * prior_mu_scale)
    tau_status_quo = pm.HalfNormal("tau_status_quo", sigma=2.0 * prior_mu_scale)
    fold_standard_normal = pm.Normal("fold_standard_normal", mu=0.0, sigma=1.0, dims="fold")

    fold_true_total_effect = pm.Deterministic(
        "fold_true_total_effect",
        mu_status_quo + tau_status_quo * fold_standard_normal,
        dims="fold",
    )

    likelihood_fold_means = pm.Normal(
        "likelihood_fold_means",
        mu=fold_true_total_effect,
        sigma=observed_fold_sd,
        observed=observed_fold_means,
        dims="fold",
    )

    idata_meta_status_quo = pm.sample(
        draws=draws_per_chain_meta,
        chains=n_chains,
        target_accept=0.97,
    )

# 3. Posterior predictive for new period
with meta_status_quo_model:
    meta_status_quo_model.add_coords({"new_period": np.arange(1)})
    theta_new = pm.Normal("theta_new", mu=mu_status_quo, sigma=tau_status_quo, dims="new_period")
    posterior_predictive_status_quo = pm.sample_posterior_predictive(idata_meta_status_quo, var_names=["theta_new"])

# Plot result
theta_new_samples = posterior_predictive_status_quo["posterior_predictive"]["theta_new"].stack(sample=("chain", "draw")).values.squeeze()
plt.hist(theta_new_samples, bins=40, density=True, alpha=0.6, label="θ_new ~ N(μ, τ)")
plt.show()
```
