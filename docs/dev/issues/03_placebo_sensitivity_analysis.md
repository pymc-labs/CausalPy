# Feature Request: Placebo Tests & Sensitivity Analysis for PanelRegression

> **Prerequisite:** This feature builds upon the Panel Fixed Effects PR ([#670](https://github.com/pymc-labs/CausalPy/pull/670)), which must be merged first.

## Summary

Add systematic placebo testing capabilities to `PanelRegression` and other experiment classes, including placebo-in-time tests, placebo-in-units tests (permutation/randomization inference), and Bayesian posterior comparison under permuted treatments.

## Motivation

### Why This Matters

Placebo tests are essential for validating causal claims. They answer the question: **"Could we have obtained this result by chance?"**

A treatment effect estimate alone is not sufficient for a credible causal claim. Reviewers and stakeholders want to see that:
1. The effect appears only where and when treatment actually occurred
2. There's no "effect" in pre-treatment periods (parallel trends)
3. The effect is distinguishable from random noise

If a placebo test shows significant "effects" where none should exist, it undermines the credibility of the main finding.

### User Story

> "I've estimated a treatment effect of 0.5 using panel regression. Before publishing, I need to show this isn't spurious. I want to run 100 placebo tests where I randomly reassign which units are treated, and show that my actual effect is larger than 95% of the placebo effects."

### Precedent in CausalPy

CausalPy already has placebo infrastructure:
- `StaggeredDifferenceInDifferences` includes pre-treatment placebo checks
- The existing approach computes ATT for pre-treatment periods, which should be ~0 if parallel trends holds

### Precedent in pyfixest: Randomization Inference

pyfixest provides `ritest()` for randomization inference (RI). This is a powerful approach for testing causal claims without relying on asymptotic approximations.

```python
# pyfixest randomization inference
fit = pf.feols("Y ~ X1 + X2", data=data)
fit.ritest("X1", reps=1000)
```

The output includes:
- **H0:** The null hypothesis being tested (e.g., "X1 = 0")
- **ri-type:** The type of test ("randomization-c" or "randomization-t")
- **Estimate:** The observed coefficient
- **Pr(>|t|):** The randomization inference p-value
- **Standard error and CI of the p-value**

## What is Randomization Inference?

Randomization Inference (RI), also called **permutation testing**, is a non-parametric approach to statistical inference. Instead of relying on asymptotic theory (Central Limit Theorem), it directly simulates the distribution of test statistics under the null hypothesis.

### The Logic

1. **Observe** the actual treatment effect: τ = 0.52
2. **Ask:** "If treatment had no effect, how likely would we be to see a coefficient this large?"
3. **Answer by simulation:** Randomly permute treatment assignment many times, re-estimate the model each time, and see where the actual coefficient falls in the distribution

### Why This Works

Under the **sharp null hypothesis** (treatment has exactly zero effect for all units), the observed outcomes would be the same regardless of treatment assignment. Therefore, any treatment assignment is equally likely to produce any pattern of results.

By permuting treatment, we generate the **null distribution** of the test statistic. If our observed statistic falls in the extreme tail of this distribution, we reject the null.

### Two Types in pyfixest

pyfixest offers two variants:

| Type | Test Statistic | Speed | Robustness |
|------|----------------|-------|------------|
| `randomization-c` | Coefficient β | Fast | Less robust |
| `randomization-t` | t-statistic β/SE | Slow | More robust (Wu & Ding, 2021) |

The "randomization-c" approach only permutes the treatment variable and re-estimates coefficients. The "randomization-t" approach also recomputes standard errors, making it more robust to heteroskedasticity.

### Cluster Randomization

If treatment is assigned at the cluster level (e.g., firms, states), the permutation should respect this:

```python
# Cluster-level permutation
fit.ritest("treated", cluster="firm", reps=1000)
```

This ensures treatment is held constant within clusters during permutation.

## Proposed API for CausalPy

### Placebo-in-Time Test

Pretend treatment happened at a different time and check for "effects":

```python
result = cp.PanelRegression(
    data=df,
    formula="y ~ treated + x1",
    unit_fe_variable="unit",
    time_fe_variable="time",
    model=cp.pymc_models.LinearRegression(...)
)

# Run placebo-in-time test
placebo_result = result.placebo_test_time(
    actual_treatment_time=2020,
    placebo_times=[2015, 2016, 2017, 2018, 2019],
    treatment_var="treated",
    sample_kwargs={"draws": 500, "tune": 200},  # Faster for placebos
)

# Visualize
placebo_result.plot()  # Shows actual effect vs placebo distribution

# Summary
placebo_result.summary()
# Actual effect: 0.52 [0.31, 0.73]
# Placebo effects: mean=0.02, std=0.15
# Rank of actual among placebos: 6/6 (p < 0.01)
```

### Placebo-in-Units Test (Permutation/Randomization Inference)

Randomly reassign treatment across units:

```python
# Run placebo-in-units test (permutation test)
placebo_result = result.placebo_test_permutation(
    treatment_var="treated",
    n_permutations=100,
    cluster="unit",  # Permute at unit level
    seed=42,
    sample_kwargs={"draws": 500, "tune": 200},
)

# Visualize
placebo_result.plot()  # Histogram of placebo effects with actual marked

# Get posterior probability that actual > placebos
placebo_result.prob_actual_exceeds_placebos()  # e.g., 0.97

# Bayesian interpretation
placebo_result.bayes_factor()  # Evidence ratio for effect vs no effect
```

### Combined Sensitivity Analysis

```python
# Comprehensive sensitivity analysis
sensitivity = result.sensitivity_analysis(
    treatment_var="treated",
    include=["placebo_time", "placebo_units", "leave_one_out"],
    n_permutations=100,
)

sensitivity.plot()  # Multi-panel diagnostic plot
sensitivity.summary()
```

## Implementation Details

### Placebo-in-Time Logic

The idea is simple: if the treatment effect is real, it should only appear *after* treatment actually occurred. If we pretend treatment happened at an earlier time, we should see no effect.

```python
def placebo_test_time(
    self,
    actual_treatment_time,
    placebo_times,
    treatment_var,
    sample_kwargs=None
):
    """
    Run placebo-in-time test.

    For each placebo time, re-fit the model pretending treatment started at that time.
    The actual treatment effect should be much larger than any placebo effect.

    Parameters
    ----------
    actual_treatment_time : int or datetime
        When treatment actually started
    placebo_times : list
        List of placebo treatment times (should be before actual_treatment_time)
    treatment_var : str
        Name of the treatment variable in the formula
    sample_kwargs : dict, optional
        Override sampling parameters for placebo models (e.g., fewer draws)

    Returns
    -------
    PlaceboTimeResult
        Object containing actual and placebo effects with visualization methods
    """
    placebo_effects = []

    for t_placebo in placebo_times:
        # Create placebo treatment variable
        df_placebo = self.data.copy()
        df_placebo[treatment_var] = (
            df_placebo[self.time_fe_variable] >= t_placebo
        ).astype(int)

        # Re-fit model with same specification
        placebo_model = PanelRegression(
            data=df_placebo,
            formula=self.formula,
            unit_fe_variable=self.unit_fe_variable,
            time_fe_variable=self.time_fe_variable,
            fe_method=self.fe_method,
            model=self._clone_model(),  # Fresh model with same config
            sample_kwargs=sample_kwargs or self.sample_kwargs,
        )

        # Extract treatment coefficient (full posterior for Bayesian)
        placebo_effects.append(
            placebo_model.get_coefficient_posterior(treatment_var)
        )

    return PlaceboTimeResult(
        actual_effect=self.get_coefficient_posterior(treatment_var),
        placebo_effects=placebo_effects,
        placebo_times=placebo_times,
    )
```

### Placebo-in-Units Logic (Permutation Test)

This follows the randomization inference approach: permute treatment assignment and re-estimate.

```python
def placebo_test_permutation(
    self,
    treatment_var,
    n_permutations=100,
    cluster=None,
    seed=None,
    sample_kwargs=None
):
    """
    Run permutation-based placebo test (randomization inference).

    Randomly permute treatment assignment across units and re-estimate.
    The actual treatment effect should fall in the extreme tail of the
    permutation distribution.

    Parameters
    ----------
    treatment_var : str
        Name of the treatment variable
    n_permutations : int
        Number of permutations (more = more precise p-value)
    cluster : str, optional
        If provided, permute treatment at this level (e.g., "unit")
    seed : int, optional
        Random seed for reproducibility
    sample_kwargs : dict, optional
        Override sampling parameters for placebo models

    Returns
    -------
    PlaceboPermutationResult
        Object containing actual effect, placebo distribution, and p-value
    """
    rng = np.random.default_rng(seed)
    placebo_effects = []

    if cluster is not None:
        # Get cluster-level treatment status
        cluster_treatment = self.data.groupby(cluster)[treatment_var].max()
        clusters = cluster_treatment.index.tolist()
        treatment_values = cluster_treatment.values
    else:
        # Observation-level permutation
        treatment_values = self.data[treatment_var].values

    for _ in tqdm(range(n_permutations), desc="Running permutations"):
        # Permute treatment assignment
        permuted_treatment = rng.permutation(treatment_values)

        # Create permuted dataset
        df_permuted = self.data.copy()
        if cluster is not None:
            cluster_to_treatment = dict(zip(clusters, permuted_treatment))
            df_permuted[treatment_var] = df_permuted[cluster].map(cluster_to_treatment)
        else:
            df_permuted[treatment_var] = permuted_treatment

        # Re-fit model
        placebo_model = PanelRegression(
            data=df_permuted,
            formula=self.formula,
            unit_fe_variable=self.unit_fe_variable,
            time_fe_variable=self.time_fe_variable,
            fe_method=self.fe_method,
            model=self._clone_model(),
            sample_kwargs=sample_kwargs or {"draws": 200, "tune": 100},
        )

        # For Bayesian: store full posterior
        # For OLS: store point estimate
        placebo_effects.append(
            placebo_model.get_coefficient_posterior(treatment_var)
        )

    return PlaceboPermutationResult(
        actual_effect=self.get_coefficient_posterior(treatment_var),
        placebo_effects=placebo_effects,
        n_permutations=n_permutations,
    )
```

### Result Classes

```python
@dataclass
class PlaceboTimeResult:
    """Results from placebo-in-time test."""

    actual_effect: xr.DataArray  # Full posterior (or float for OLS)
    actual_effect_hdi: tuple[float, float]
    placebo_effects: list  # List of posteriors
    placebo_times: list

    def plot(self, ax=None):
        """Plot actual effect vs placebo effects over time."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot placebo effects
        for i, (t, effect) in enumerate(zip(self.placebo_times, self.placebo_effects)):
            if isinstance(effect, xr.DataArray):
                mean = float(effect.mean())
                hdi = az.hdi(effect)
                ax.errorbar(t, mean, yerr=[[mean-hdi[0]], [hdi[1]-mean]],
                           fmt='o', color='gray', alpha=0.6)
            else:
                ax.scatter(t, effect, color='gray', alpha=0.6)

        # Plot actual effect
        actual_mean = float(self.actual_effect.mean())
        ax.scatter(self.actual_time, actual_mean, color='red', s=100, zorder=5)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel("Treatment Time")
        ax.set_ylabel("Estimated Effect")
        ax.set_title("Placebo-in-Time Test")

        return ax

    def summary(self):
        """Print summary of placebo test results."""
        actual_mean = float(self.actual_effect.mean())
        placebo_means = [float(p.mean()) for p in self.placebo_effects]

        print(f"Actual effect: {actual_mean:.3f} {self.actual_effect_hdi}")
        print(f"Placebo effects: mean={np.mean(placebo_means):.3f}, std={np.std(placebo_means):.3f}")
        print(f"Rank of actual among placebos: {self.rank()}/{len(self.placebo_effects)+1}")

    def rank(self):
        """Rank of actual effect among all effects (higher = more extreme)."""
        all_means = [float(p.mean()) for p in self.placebo_effects] + [float(self.actual_effect.mean())]
        return sorted(all_means, reverse=True).index(float(self.actual_effect.mean())) + 1


@dataclass
class PlaceboPermutationResult:
    """Results from permutation-based placebo test."""

    actual_effect: xr.DataArray  # Full posterior
    placebo_effects: list  # List of posteriors
    n_permutations: int

    def plot(self, ax=None):
        """Plot histogram of placebo effects with actual marked."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Get means of placebo effects
        placebo_means = [float(p.mean()) for p in self.placebo_effects]
        actual_mean = float(self.actual_effect.mean())

        # Histogram of placebos
        ax.hist(placebo_means, bins=30, alpha=0.7, color='steelblue',
                label='Placebo distribution')

        # Vertical line for actual
        ax.axvline(actual_mean, color='red', linewidth=2,
                   label=f'Actual ({actual_mean:.3f})')

        ax.set_xlabel("Estimated Effect")
        ax.set_ylabel("Frequency")
        ax.set_title("Permutation Test")
        ax.legend()

        return ax

    def prob_actual_exceeds_placebos(self):
        """
        P(actual > placebo) - Bayesian probability that actual effect
        exceeds the placebo distribution.
        """
        actual_mean = float(self.actual_effect.mean())
        placebo_means = [float(p.mean()) for p in self.placebo_effects]
        return np.mean([actual_mean > p for p in placebo_means])

    def empirical_pvalue(self, two_sided=True):
        """
        Frequentist p-value from permutation distribution.

        p = (# placebos with |effect| >= |actual|) / n_permutations
        """
        actual_mean = float(self.actual_effect.mean())
        placebo_means = [float(p.mean()) for p in self.placebo_effects]

        if two_sided:
            return np.mean([abs(p) >= abs(actual_mean) for p in placebo_means])
        else:
            return np.mean([p >= actual_mean for p in placebo_means])
```

### Visualization

**Placebo-in-Time Plot:**
```
Effect
  │
  │                           ●  Actual (T=2020)
  │
  │  ○─────○─────○─────○─────○   Placebos (T=2015-2019)
  │
  └─────────────────────────────
     2015  2016  2017  2018  2019  2020
                Placebo treatment time
```

**Permutation Test Plot:**
```
       Histogram of Placebo Effects
  │
  │    ▄▄▄▄▄
  │   ▄██████▄
  │  ▄████████▄
  │ ▄██████████▄              │ Actual
  │▄████████████▄             │ (0.52)
  └──────────────────────────────
   -0.3    0     0.3     0.6
```

## Differentiation from pyfixest

| Aspect | pyfixest `ritest()` | CausalPy (proposed) |
|--------|---------------------|---------------------|
| Inference type | Frequentist p-value | Bayesian posterior comparison |
| Output | p-value, rank, SE of p-value | P(actual > placebos), full posteriors |
| Uncertainty | CI on p-value | HDI on effects, posterior probability |
| Visualization | Optional histogram | Built-in density plots with HDI |
| Computation | Fast (Numba-accelerated) | Slower (full MCMC per permutation) |
| Existing infrastructure | Standalone method | Builds on CausalPy experiment pattern |

## Computational Considerations

### The Challenge

Permutation tests with Bayesian models are computationally expensive:
- 100 permutations × full MCMC (1000 draws) = 100,000+ samples
- Each permutation requires fitting a complete model

### Mitigations

1. **Reduced sampling for placebos:** Use fewer draws for placebo models since we only need rough estimates of the distribution
   ```python
   sample_kwargs={"draws": 200, "tune": 100}  # vs 1000/500 for main model
   ```

2. **Parallelization:** Fit placebo models in parallel (future enhancement)
   ```python
   from joblib import Parallel, delayed
   placebo_effects = Parallel(n_jobs=-1)(
       delayed(fit_placebo)(i) for i in range(n_permutations)
   )
   ```

3. **Approximate methods:** Consider posterior-based approximations that don't require re-fitting

## Blockers & Prerequisites

### Required Before Implementation

1. **PR #670 merged:** `PanelRegression` must be stable
2. **Model cloning mechanism:** Need to re-create models with same configuration
3. **Efficient re-fitting:** Consider reduced `sample_kwargs` for placebos

### Technical Challenges

- **Computational cost:** 100 permutations × full MCMC = expensive
- **Solution:** Allow reduced `sample_kwargs` for placebos
- **Alternative:** Use posterior predictive for approximate placebo (no re-fit)

## Effort Estimate

| Component | Complexity |
|-----------|------------|
| Placebo-in-time implementation | Medium |
| Permutation test implementation | Medium |
| Result classes and plotting | Medium |
| Integration with existing experiment classes | Low |
| Parallelization (optional) | High |
| Tests and documentation | Medium |
| **Total** | **Medium-High** (~3-4 days) |

## Acceptance Criteria

- [ ] `placebo_test_time()` method on `PanelRegression`
- [ ] `placebo_test_permutation()` method on `PanelRegression`
- [ ] Result classes with `plot()` and `summary()` methods
- [ ] Works with Bayesian models (full posterior comparison)
- [ ] Works with OLS models (point estimate comparison)
- [ ] Configurable `sample_kwargs` for faster placebo fitting
- [ ] Cluster-level permutation option
- [ ] Unit tests for both placebo types
- [ ] Example in documentation showing interpretation

## Related Issues / PRs

- PR #670: `PanelRegression` (prerequisite)
- Existing `PlaceboTest` class in `StaggeredDifferenceInDifferences`
- pyfixest `ritest()` for inspiration

## References

- Wu, J. & Ding, P. (2021). Randomization Tests for Weak Null Hypotheses in Randomized Experiments. *Journal of the American Statistical Association*.
- Fisher, R. A. (1935). *The Design of Experiments*. (Original work on permutation testing)

## Labels

`enhancement`, `panel-regression`, `sensitivity-analysis`, `bayesian`
