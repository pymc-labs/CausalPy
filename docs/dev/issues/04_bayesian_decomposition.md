# Feature Request: Bayesian Decomposition / Mediation Analysis

> **Prerequisite:** This feature builds upon the Panel Fixed Effects PR ([#670](https://github.com/pymc-labs/CausalPy/pull/670)), which must be merged first.

## Summary

Add a `decompose()` method to `PanelRegression` that performs Gelbach-style decomposition with full Bayesian uncertainty quantification, allowing researchers to understand how much of a treatment effect is explained by specific mechanisms or mediators.

## What is Gelbach Decomposition?

### The Core Question

When we find a treatment effect, the natural follow-up question is: **"Why does this effect exist? What explains it?"**

Consider this example:

> You find that a job training program increases wages by 15%. Management asks: "Is this because trained workers get better credentials, switch to higher-paying industries, or work longer hours?"

Gelbach decomposition provides a rigorous way to answer this question by decomposing the total effect into components explained by different mechanisms.

### The Intuition

Imagine you run two regressions:

**Short regression (without mediators):**
```
wage ~ treatment
β_short = 0.15  ← "Total effect"
```

**Long regression (with mediators):**
```
wage ~ treatment + credentials + industry + hours
β_long = 0.05   ← "Direct effect" (what's left after controlling for mechanisms)
```

The difference (0.15 - 0.05 = 0.10) is the **explained effect** - the portion of the treatment effect that operates *through* the mediators.

Gelbach decomposition goes further by attributing this explained effect to each specific mediator: "0.04 from credentials, 0.03 from industry, 0.03 from hours."

### The Math

The Gelbach decomposition identity states:

```
β_short = β_long + Σⱼ (γⱼ × δⱼ)
```

Where:
- **β_short** = coefficient on treatment in short regression (without mediators)
- **β_long** = coefficient on treatment in long regression (with mediators)
- **γⱼ** = effect of treatment on mediator j (from regressing mediator on treatment)
- **δⱼ** = effect of mediator j on outcome (coefficient in long regression)
- **γⱼ × δⱼ** = indirect effect through mediator j

This is exact - it's not an approximation but an algebraic identity.

### Visual Representation

```
                   ┌──────────────┐
                   │  Credentials │
                   │   (γ₁ × δ₁)  │
              ┌───►│    = 0.04    │───┐
              │    └──────────────┘   │
              │                       │
              │    ┌──────────────┐   │
              │    │   Industry   │   │
 ┌─────────┐  │───►│   (γ₂ × δ₂)  │───│───► ┌────────┐
 │Treatment│──┤    │    = 0.03    │   │     │ Wages  │
 │         │  │    └──────────────┘   ├────►│        │
 │ β_short │  │                       │     │ Total  │
 │ = 0.15  │  │    ┌──────────────┐   │     │ effect │
 └─────────┘  │───►│    Hours     │───│     │ = 0.15 │
              │    │   (γ₃ × δ₃)  │   │     └────────┘
              │    │    = 0.03    │   │
              │    └──────────────┘   │
              │                       │
              └───────────────────────┘
                    Direct effect
                    (β_long = 0.05)
```

### Academic Foundation

This approach was formalized by **Jonah Gelbach** in his 2016 paper "When Do Covariates Matter? And Which Ones, and How Much?" in the *Journal of Labor Economics*.

Key insight: The standard approach of comparing coefficients between regressions is actually doing implicit decomposition. Gelbach makes this explicit and provides standard errors.

### Important Caveat: This is NOT Causal Mediation

**Gelbach decomposition is correlation-based, not causal.** It tells you how the coefficient *changes* when you add mediators, not whether the mediators are *causal mechanisms*.

For true causal mediation, you need additional assumptions:
- Sequential ignorability (no unmeasured confounders of mediator-outcome relationship)
- No treatment-induced confounding
- Correct model specification

CausalPy documentation should be clear about this distinction.

## Motivation

### Why This Matters for CausalPy

When researchers find a treatment effect, stakeholders often want to understand *why* the effect exists:

| Domain | Question |
|--------|----------|
| Policy | "Does the education intervention work through attendance, study habits, or peer effects?" |
| Marketing | "Is the ad campaign effect driven by awareness, consideration, or purchase intent?" |
| Healthcare | "Does the drug work through reducing inflammation, improving blood flow, or both?" |

### User Story

> "I've found that a training program increases productivity by 15%. Management wants to know: Is this because trained workers use better tools, work longer hours, or are intrinsically more skilled? I need to decompose the total effect into these components with uncertainty estimates."

### Why Bayesian?

pyfixest's `decompose()` uses **bootstrap** for uncertainty quantification. This involves:
1. Resample data B times
2. Re-run decomposition on each bootstrap sample
3. Compute percentile confidence intervals

CausalPy's Bayesian approach offers advantages:

| Aspect | Bootstrap (pyfixest) | Bayesian (CausalPy) |
|--------|----------------------|---------------------|
| Uncertainty source | Resampling | Posterior distribution |
| Interpretation | "95% of bootstrap samples had effect in this range" | "95% probability effect is in this range" |
| Probability queries | Not natural | Direct: "P(mediator explains >25%)?" |
| Computational cost | Fast (OLS each resample) | Slower (MCMC) |
| Handling of small samples | Can be unreliable | Natural regularization via priors |

## How pyfixest Implements This

### API

```python
import pyfixest as pf

# Fit the full model
fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

# Decompose the effect of x1 into components explained by x21, x22, x23
gb = fit.decompose(decomp_var="x1", reps=1000)

# View results
gb.tidy()   # DataFrame with effects and CIs
gb.etable() # Formatted table
gb.coefplot() # Waterfall chart
```

### Output

```
                     coefficients  ci_lower  ci_upper  panels
direct_effect             0.200     0.150     0.250  Levels (units)
full_effect               0.500     0.400     0.600  Levels (units)
explained_effect          0.300     0.220     0.380  Levels (units)
x21                       0.150     0.100     0.200  Levels (units)
x22                       0.100     0.050     0.150  Levels (units)
x23                       0.050     0.010     0.090  Levels (units)
```

### The Algorithm

pyfixest implements this in `pyfixest/estimation/decomposition.py`:

```python
def compute_gelbach(self, X1, X2, Y, X, agg_first):
    """
    X1 = design matrix with treatment only (short model)
    X2 = design matrix with mediators only
    X = full design matrix (treatment + mediators)
    Y = outcome
    """
    # Step 1: Direct effect (short regression)
    direct_effect = lsqr(X1, Y)[0]  # β_short

    # Step 2: Full regression
    beta_full = lsqr(X, Y)[0]  # β_long and δs
    beta2 = beta_full[mediator_indices]  # δs: mediator coefficients

    # Step 3: Compute indirect effects
    if agg_first:
        # Aggregate first approach (faster for many mediators)
        H = X2 @ np.diag(beta2)  # X2_j * δ_j for each mediator
        # Regress each H column on X1 to get γ_j
        delta = [lsqr(X1, H[:, j])[0] for j in range(H.shape[1])]
    else:
        # Standard approach
        # For each mediator, regress mediator on treatment to get γ
        gamma = [lsqr(X1, X2[:, j])[0] for j in range(X2.shape[1])]
        delta = gamma * beta2  # Element-wise: γ_j × δ_j

    return GelbachResults(
        direct_effect=direct_effect,
        full_effect=beta_full[treatment_idx],
        explained_effect=sum(delta),
        mediator_effects={name: delta[i] for i, name in enumerate(mediator_names)}
    )
```

## Proposed API for CausalPy

### Basic Usage

```python
# Fit base model (without mediators)
base_result = cp.PanelRegression(
    data=df,
    formula="y ~ treated",
    unit_fe_variable="unit",
    model=cp.pymc_models.LinearRegression(...)
)

# Fit full model (with mediators)
full_result = cp.PanelRegression(
    data=df,
    formula="y ~ treated + mediator1 + mediator2 + mediator3",
    unit_fe_variable="unit",
    model=cp.pymc_models.LinearRegression(...)
)

# Decomposition
decomp = cp.decompose(
    base_model=base_result,
    full_model=full_result,
    treatment_var="treated",
    mediators=["mediator1", "mediator2", "mediator3"],
    hdi_prob=0.94
)

# Results
decomp.summary()
```

**Output:**
```
Gelbach Decomposition
=====================
Treatment variable: treated

Total effect (base): 0.50 [0.35, 0.65]
Direct effect (full): 0.20 [0.08, 0.32]

Explained by mediators:
  mediator1: 0.15 [0.05, 0.25] (30% of total)
  mediator2: 0.10 [0.02, 0.18] (20% of total)
  mediator3: 0.05 [-0.02, 0.12] (10% of total)

Bayesian probabilities:
  P(mediator1 explains > 25%): 0.72
  P(any mediator explains > 50%): 0.15
  P(direct effect is majority): 0.35
```

### Probability Queries

Unique to Bayesian approach - ask probabilistic questions:

```python
# What's the probability mediator1 explains more than 25%?
decomp.prob_explains_more_than("mediator1", threshold=0.25)
# Returns: 0.72

# What's the probability the direct effect is < 50% of total?
decomp.prob_direct_less_than(fraction=0.5)
# Returns: 0.65

# Rank mediators by explanatory power
decomp.rank_mediators()  # Returns probabilistic ranking
```

### Visualization

```python
# Waterfall chart with HDI
decomp.plot()
```

```
        Total Effect (β_short)
            │
            │ ████████████████████████████  0.50
            │                              [0.35, 0.65]
            │
            │ ─ mediator1 ─────────────────  -0.15
            │ ██████████                    [0.05, 0.25]
            │
            │ ─ mediator2 ─────────────────  -0.10
            │ ██████                        [0.02, 0.18]
            │
            │ ─ mediator3 ─────────────────  -0.05
            │ ███                           [-0.02, 0.12]
            │
            │ = Direct Effect (β_long) ────  0.20
            │ ████████                      [0.08, 0.32]
            │
            └──────────────────────────────────────
```

## Implementation Details

### Core Algorithm for Bayesian Decomposition

```python
def decompose(
    base_model,
    full_model,
    treatment_var: str,
    mediators: list[str],
    hdi_prob: float = 0.94
) -> BayesianDecomposition:
    """
    Perform Gelbach decomposition with Bayesian uncertainty.

    Parameters
    ----------
    base_model : PanelRegression
        Model without mediators (y ~ treatment)
    full_model : PanelRegression
        Model with mediators (y ~ treatment + mediators)
    treatment_var : str
        Name of treatment variable
    mediators : list[str]
        Names of mediator variables
    hdi_prob : float
        Width of HDI intervals

    Returns
    -------
    BayesianDecomposition
        Object with decomposition results and plotting methods
    """
    # Step 1: Extract posteriors from both models
    # β_short: total effect from base model
    beta_short = base_model.idata.posterior[treatment_var]  # Shape: (chain, draw)

    # β_long: direct effect from full model
    beta_long = full_model.idata.posterior[treatment_var]

    # Step 2: For each mediator, compute indirect effect
    indirect_effects = {}

    for mediator in mediators:
        # Step 2a: Fit auxiliary model: mediator ~ treatment
        # This gives γ: effect of treatment on mediator
        mediator_model = fit_auxiliary_regression(
            data=base_model.data,
            formula=f"{mediator} ~ {treatment_var}",
            unit_fe_variable=base_model.unit_fe_variable,
        )
        gamma = mediator_model.idata.posterior[treatment_var]

        # Step 2b: Get δ: coefficient on mediator in full model
        delta = full_model.idata.posterior[mediator]

        # Step 2c: Indirect effect = γ × δ
        # Element-wise multiplication across posterior samples
        indirect_effects[mediator] = gamma * delta

    # Step 3: Sum of indirect effects should equal β_short - β_long
    total_indirect = sum(indirect_effects.values())

    # Verify decomposition identity (within numerical tolerance)
    residual = beta_short - beta_long - total_indirect
    assert residual.mean().abs() < 0.01, "Decomposition identity violated"

    return BayesianDecomposition(
        total_effect=beta_short,
        direct_effect=beta_long,
        indirect_effects=indirect_effects,
        mediators=mediators,
        hdi_prob=hdi_prob,
    )
```

### Result Class

```python
@dataclass
class BayesianDecomposition:
    """Results from Bayesian Gelbach decomposition."""

    total_effect: xr.DataArray  # β_short posterior samples
    direct_effect: xr.DataArray  # β_long posterior samples
    indirect_effects: dict[str, xr.DataArray]  # mediator -> γδ posterior
    mediators: list[str]
    hdi_prob: float

    def summary(self, round_to: int = 3) -> None:
        """Print formatted summary of decomposition."""
        print("Gelbach Decomposition")
        print("=" * 50)

        # Total effect
        total_mean = float(self.total_effect.mean())
        total_hdi = az.hdi(self.total_effect, hdi_prob=self.hdi_prob)
        print(f"Total effect: {total_mean:.{round_to}f} [{total_hdi[0]:.{round_to}f}, {total_hdi[1]:.{round_to}f}]")

        # Direct effect
        direct_mean = float(self.direct_effect.mean())
        direct_hdi = az.hdi(self.direct_effect, hdi_prob=self.hdi_prob)
        print(f"Direct effect: {direct_mean:.{round_to}f} [{direct_hdi[0]:.{round_to}f}, {direct_hdi[1]:.{round_to}f}]")

        # Indirect effects
        print("\nExplained by mediators:")
        for mediator in self.mediators:
            effect = self.indirect_effects[mediator]
            mean = float(effect.mean())
            hdi = az.hdi(effect, hdi_prob=self.hdi_prob)
            pct = 100 * mean / total_mean
            print(f"  {mediator}: {mean:.{round_to}f} [{hdi[0]:.{round_to}f}, {hdi[1]:.{round_to}f}] ({pct:.1f}%)")

    def plot(self, kind: str = "waterfall", ax=None) -> tuple:
        """
        Plot decomposition results.

        Parameters
        ----------
        kind : str
            "waterfall" for bar chart, "forest" for coefficient plot
        """
        if kind == "waterfall":
            return self._plot_waterfall(ax)
        elif kind == "forest":
            return self._plot_forest(ax)

    # Bayesian probability queries
    def prob_explains_more_than(self, mediator: str, threshold: float) -> float:
        """
        P(indirect_effect[mediator] / total_effect > threshold)

        Example: "What's the probability mediator1 explains more than 25%?"
        """
        ratio = self.indirect_effects[mediator] / self.total_effect
        return float((ratio > threshold).mean())

    def prob_direct_less_than(self, fraction: float) -> float:
        """
        P(direct_effect / total_effect < fraction)

        Example: "What's the probability the direct effect is less than half the total?"
        """
        ratio = self.direct_effect / self.total_effect
        return float((ratio < fraction).mean())

    def rank_mediators(self) -> pd.DataFrame:
        """
        Probabilistic ranking of mediators by explanatory power.

        For each posterior sample, rank mediators by contribution.
        Return probability each mediator is most important.
        """
        # Stack all indirect effects
        effects = np.stack([
            self.indirect_effects[m].values.flatten()
            for m in self.mediators
        ], axis=1)  # Shape: (n_samples, n_mediators)

        # For each sample, rank by absolute contribution
        ranks = np.argsort(np.argsort(-np.abs(effects), axis=1), axis=1) + 1

        # Probability each mediator is rank 1
        prob_first = (ranks == 1).mean(axis=0)

        return pd.DataFrame({
            "mediator": self.mediators,
            "mean_effect": [float(self.indirect_effects[m].mean()) for m in self.mediators],
            "prob_most_important": prob_first,
        }).sort_values("prob_most_important", ascending=False)
```

## Handling Fixed Effects

When models include fixed effects:
- Unit FE absorbed - doesn't affect decomposition
- The decomposition operates on the treatment coefficient only
- Both models should have the same FE structure for valid comparison

## Important Caveats

### This is NOT Causal Mediation

**Critical disclaimer for documentation:**

> **Note:** Gelbach decomposition shows how the treatment coefficient changes when mediators are added. It does NOT establish causal mediation pathways.
>
> For true causal mediation analysis, you need:
> - Sequential ignorability (no unmeasured confounders)
> - No treatment-induced confounding
> - Correct causal ordering
>
> This method is best interpreted as "accounting for differences" rather than "explaining causal mechanisms."

### When is it Valid?

Gelbach decomposition is most appropriate when:
1. Mediators are measured before or simultaneously with outcome
2. Interest is in statistical (not causal) explanation
3. Goal is sensitivity analysis: "How robust is the treatment effect to adding controls?"

## Differentiation from pyfixest

| Aspect | pyfixest | CausalPy (proposed) |
|--------|----------|---------------------|
| Uncertainty | Bootstrap CIs | Full posterior from MCMC |
| Probability queries | Not available | P(mediator explains > X%) |
| Visualization | Waterfall with CIs | Waterfall with HDI |
| Ranking | Point estimate order | Probabilistic ranking |
| Interpretation | Frequentist | Bayesian probability statements |
| Computation | Fast (OLS + bootstrap) | Slower (MCMC) |

## Blockers & Prerequisites

### Required Before Implementation

1. **PR #670 merged:** `PanelRegression` must be stable
2. **Posterior access:** Need consistent API for extracting coefficient posteriors
3. **Auxiliary regression fitting:** May need lightweight model fitting for γ estimation

### Technical Considerations

- **Model consistency:** Base and full models should have same FE structure
- **Overlapping samples:** Both models should be fit on same data
- **Mediator endogeneity:** Document clearly that this is correlation, not causation

## Effort Estimate

| Component | Complexity |
|-----------|------------|
| Core decomposition logic | Medium |
| Auxiliary regression for γ | Low |
| Posterior probability queries | Low |
| Waterfall chart visualization | Medium |
| Result class and summary | Low |
| Documentation with caveats | Medium |
| Tests | Medium |
| **Total** | **Medium** (~2-3 days) |

## Acceptance Criteria

- [ ] `cp.decompose()` function implemented
- [ ] Works with `PanelRegression` (Bayesian models)
- [ ] Computes total effect, direct effect, and indirect effects
- [ ] Provides probability queries (e.g., `prob_explains_more_than()`)
- [ ] Waterfall chart visualization with HDI
- [ ] Clear documentation about non-causal interpretation
- [ ] Unit tests for decomposition identity
- [ ] Example showing interpretation

## Related Issues / PRs

- PR #670: `PanelRegression` (prerequisite)
- pyfixest `GelbachDecomposition` for inspiration

## References

- Gelbach, J. B. (2016). When Do Covariates Matter? And Which Ones, and How Much? *Journal of Labor Economics*, 34(2), 509-543. [SSRN link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1425737)
- Imai, K., Keele, L., & Tingley, D. (2010). A General Approach to Causal Mediation Analysis. *Psychological Methods*, 15(4), 309-334. (For contrast with causal mediation)

## Labels

`enhancement`, `panel-regression`, `mediation`, `bayesian`
