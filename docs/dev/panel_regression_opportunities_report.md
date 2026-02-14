# CausalPy PanelRegression: Current Capabilities & Opportunities

This document analyzes the current `PanelRegression` implementation (PR #670) and identifies opportunities for enhancement inspired by pyfixest's feature set. The goal is not to duplicate pyfixest, but to leverage fixed effects capabilities within CausalPy's Bayesian and causal inference framework.

---

## Related PRs in Progress

Before diving into the `PanelRegression` PR (#670), it's important to understand what's being built in parallel:

### PR #584: Event Study (Dynamic DiD)

**Status:** Open, in active development

**What it adds:**
- New `EventStudy` experiment class for dynamic difference-in-differences
- Event-time coefficients (βₖ) with Bayesian HDI visualization
- Automatic computation of time-relative-to-treatment
- Pre-treatment trend visualization (parallel trends diagnostic)
- Both PyMC (Bayesian) and sklearn (OLS) support
- Warning about staggered treatment timing (TWFE bias)

**Key features:**
```python
result = cp.EventStudy(
    data=df,
    formula="y ~ C(unit) + C(time)",
    unit_col="unit",
    time_col="time",
    treat_time_col="treat_time",
    event_window=(-5, 5),
    reference_event_time=-1,
    model=cp.pymc_models.LinearRegression(...)
)
result.plot()  # Event study plot with HDI
result.get_event_time_summary()  # Coefficients table
```

### PR #614: Piecewise ITS

**Status:** Open, in active development

**What it adds:**
- New `PiecewiseITS` experiment class for segmented regression
- `step(time, threshold)` and `ramp(time, threshold)` patsy transforms
- Multiple known intervention points in a single model
- Level and slope change estimation at each intervention

**Key features:**
```python
result = cp.PiecewiseITS(
    data=df,
    formula="y ~ 1 + t + step(t, 50) + ramp(t, 50) + step(t, 100)",
    model=cp.pymc_models.LinearRegression(...)
)
```

---

## Current PR #670 Capabilities (PanelRegression)

### Core Features (Already Implemented)

| Feature | Description | Status |
|---------|-------------|--------|
| **Two FE Methods** | `dummies` (explicit dummy variables) and `within` (demeaning transformation) | ✅ Complete |
| **Unit Fixed Effects** | Control for time-invariant unit-level confounders | ✅ Complete |
| **Time Fixed Effects** | Control for time-varying common shocks | ✅ Complete |
| **PyMC Support** | Full Bayesian inference with posterior distributions | ✅ Complete |
| **Scikit-learn Support** | OLS estimation via sklearn interface | ✅ Complete |
| **Panel Dimensions** | Track `n_units` and `n_periods` | ✅ Complete |
| **Within Transformation** | Automatic demeaning for large panels | ✅ Complete |
| **Group Mean Storage** | Store means for fixed effect recovery | ✅ Complete |

### Visualization (Already Implemented)

| Feature | Description | Status |
|---------|-------------|--------|
| **Coefficient Plots** | Forest plot (Bayesian) / bar chart (OLS) of non-FE coefficients | ✅ Complete |
| **Unit Effects Distribution** | Histogram of unit fixed effects (dummies method) | ✅ Complete |
| **Trajectory Plots** | Unit-level actual vs. fitted over time with HDI | ✅ Complete |
| **Residual Diagnostics** | Scatter, histogram, and Q-Q plots | ✅ Complete |

### Reporting (Already Implemented)

| Feature | Description | Status |
|---------|-------------|--------|
| **Summary Method** | Panel dimensions, FE method, filtered coefficients | ✅ Complete |
| **Print Coefficients** | Exclude FE dummies for cleaner output | ✅ Complete |

---

## Opportunities: What PyFixest Does That Could Inspire CausalPy

### Already Being Addressed (Other PRs)

#### ✅ Event Study / Dynamic Treatment Effects → PR #584

**Status:** Being implemented as separate `EventStudy` class

This is being addressed in PR #584 with a dedicated experiment class rather than as part of `PanelRegression`. The implementation includes:
- Event-time coefficient estimation with HDI
- Pre-treatment trend visualization
- Parallel trends diagnostic
- Warning about staggered treatment TWFE bias

**Relationship to PanelRegression:**
- `EventStudy` is conceptually related but serves a different purpose
- `PanelRegression` = general panel model with FE
- `EventStudy` = specifically designed for dynamic DiD with event-time visualization
- Users needing event study plots should use `EventStudy`, not `PanelRegression`

---

### Already Implemented in CausalPy

#### ✅ Staggered DiD → `StaggeredDifferenceInDifferences`

**Status:** Already implemented in `causalpy/experiments/staggered_did.py`

CausalPy already has a comprehensive staggered DiD implementation using the **Borusyak, Jaravel, and Spiess (2024)** imputation-based estimator:

**What it does:**
- Fits model on untreated observations only (pre-treatment + never-treated)
- Predicts counterfactuals for treated observations
- Computes treatment effects as observed - predicted
- Aggregates to **group-time ATTs**: ATT(g, t) by cohort and calendar time
- Aggregates to **event-time ATTs**: ATT(e) for each event-time e = t - G
- Includes **pre-treatment placebo checks** (should be ~0 if parallel trends holds)
- Full **Bayesian posterior** with HDI for all estimates
- Validates **absorbing treatment** assumption
- Supports **balanced and unbalanced panels**

**Key features:**
```python
result = cp.StaggeredDifferenceInDifferences(
    data=df,
    formula="y ~ 1 + C(unit) + C(time)",
    unit_variable_name="unit",
    time_variable_name="time",
    treated_variable_name="treated",
    treatment_time_variable_name="treatment_time",
    model=cp.pymc_models.LinearRegression(...)
)
result.att_event_time_   # Event-time ATTs with HDI
result.att_group_time_   # Group-time ATTs
result.plot()            # Event study visualization
```

**Differentiation from pyfixest:**
- Full Bayesian posterior (not just point estimates)
- HDI intervals for all ATT estimates
- No need for bootstrap - uncertainty comes from MCMC

---

### Remaining Opportunities

#### 1. Counterfactual Predictions & Visualization

**Pyfixest approach:** Standard `predict()` with new data.

**CausalPy opportunity:**
- `plot_counterfactual()` method showing actual vs. predicted under D=0
- Particularly powerful with Bayesian models (HDI around counterfactual)
- Similar visualization to synthetic control

**Implementation complexity:** Low-Medium
- Need to handle treatment variable zeroing
- Integrate with posterior predictive

**Differentiation from pyfixest:**
- Full posterior predictive distribution for counterfactual
- Causal interpretation built into the framing

---

#### 2. Model Comparison Tables

**Pyfixest approach:** `etable()` for side-by-side comparison.

**CausalPy opportunity:**
- Compare multiple `PanelRegression` specifications
- Show treatment effect across models
- Include model fit statistics (R², WAIC for Bayesian)

**Implementation complexity:** Low
- Format coefficients and statistics
- Output to markdown/LaTeX

**Differentiation from pyfixest:**
- Include Bayesian model comparison metrics (LOO, WAIC)
- Posterior summary statistics

---

#### 3. Placebo Tests / Sensitivity Analysis

**Pyfixest approach:** `ritest()` method for randomization inference.

**CausalPy opportunity:**
- Placebo-in-time tests (already partially in codebase)
- Placebo-in-units tests
- Compare posterior distributions under permuted treatments

**Implementation complexity:** Medium
- Need efficient resampling
- Bayesian approach: re-fit model with permuted treatment

**Differentiation from pyfixest:**
- Full posterior comparison, not just p-values
- Could leverage existing placebo infrastructure in CausalPy

---

#### 4. Bayesian Decomposition / Mediation

**Pyfixest approach:** `decompose()` method for Gelbach-style mediation analysis.

**CausalPy opportunity:**
- Decompose treatment effect by adding covariates
- Show "how much of the raw effect is explained by X"
- Bayesian posterior for each component's contribution

**Implementation complexity:** Medium
- Compare posteriors from models with/without mediators
- Natural uncertainty quantification from MCMC

**Differentiation from pyfixest:**
- Full posterior on each decomposition component
- No need for bootstrap - uncertainty comes from posterior
- Probability statements about mediation (e.g., P(mediator explains >50%))

---

### Lower Priority / Long-term Opportunities

#### 5. High-Dimensional Fixed Effects Performance

**Pyfixest approach:** Rust/Numba/GPU backends for demeaning.

**CausalPy opportunity:**
- Current `within` method is pure Python/Pandas
- Could optimize for very large panels

**Implementation complexity:** High
- Would need Numba or similar acceleration
- Memory management for large panels

**Differentiation from pyfixest:**
- Less critical since CausalPy focuses on interpretability over speed
- Bayesian models have their own computational challenges

---

## CausalPy's Unique Advantages (Not in pyfixest)

> **Note:** Multiple hypothesis corrections (Bonferroni, Romano-Wolf) are not included here. Bayesian inference with proper priors provides natural regularization through shrinkage and hierarchical modeling, making frequentist-style MHT corrections unnecessary.

These are CausalPy strengths that should be preserved and enhanced:

| Advantage | Description |
|-----------|-------------|
| **Bayesian Inference** | Full posterior distributions, not just point estimates |
| **HDI for All Quantities** | Credible intervals for predictions, effects, counterfactuals |
| **Causal Framing** | Explicit language about treatment effects and causality |
| **Posterior Predictive Checks** | Bayesian model validation |
| **Integration with PyMC** | Access to full MCMC diagnostics, priors, hierarchical models |
| **Unified API** | Same interface for DiD, ITS, SC, RD, Panel |
| **Hierarchical Modeling** | Partial pooling, shrinkage, borrowing strength across units |

---

## Recommended Roadmap

### Currently In Progress (Parallel PRs)

| PR | Feature | Status |
|----|---------|--------|
| **#670** | `PanelRegression` - General panel FE class | Open |
| **#584** | `EventStudy` - Dynamic DiD with event-time plots | Open |
| **#614** | `PiecewiseITS` - Multiple intervention ITS | Open |

### Phase 1: Core PRs (Current Focus)
Focus on getting the three open PRs merged:
- **PanelRegression** (#670): Two-way fixed effects, dummies/within methods
- **EventStudy** (#584): Dynamic treatment effects, parallel trends
- **PiecewiseITS** (#614): Segmented regression with multiple interventions

### Phase 2: Bayesian Enhancements for PanelRegression
1. **Counterfactual plotting** - Visualize actual vs. counterfactual with posterior HDI
2. **Hierarchical priors for unit effects** - Partial pooling across units
3. **Posterior predictive checks** - Model validation specific to panel data

### Phase 3: Enhanced Sensitivity Analysis
1. **Placebo tests** - Systematic placebo-in-time and placebo-in-units
2. **Bayesian decomposition** - Mechanism analysis with posterior uncertainty
3. **Model comparison** - LOO/WAIC across specifications

### Phase 4: Cross-Cutting Features
1. **Decomposition methods** - Mechanism analysis (Gelbach-style)
2. **Model comparison tables** - Side-by-side robustness checks
3. **Integration between experiment classes** - Unified workflows

---

## Implementation Notes

### What NOT to Implement

1. **GPU acceleration** - Overkill for CausalPy's use cases
2. **Rust backends** - Complexity not justified
3. **Full pyfixest formula syntax** - Use patsy, don't reinvent
4. **Poisson/GLM with FE** - Out of scope for panel class
5. **OLS-specific inference enhancements** - Focus on Bayesian; OLS users can use pyfixest
   - No clustered standard errors
   - No HAC standard errors
   - No on-the-fly variance adjustment

### Design Principles

1. **Bayesian First:** Features should leverage posterior distributions
2. **Causal Framing:** Methods should use causal language
3. **Simplicity:** Don't replicate pyfixest's full complexity
4. **Integration:** Build on existing CausalPy patterns

### API Considerations

```python
# Example future API extensions for PanelRegression (Bayesian focus)

# Counterfactual predictions with posterior uncertainty
result.plot_counterfactual(units=["unit_1", "unit_2"])
result.predict(newdata=df_counterfactual)  # Returns posterior predictive

# Bayesian model diagnostics
result.posterior_predictive_check()
result.plot_unit_effects_posterior()  # Shrinkage visualization

# Sensitivity analysis
result.diagnose_staggered_timing()  # Check for TWFE issues
result.placebo_test(n_placebos=100)

# Note: Event study plots are in the separate EventStudy class (PR #584)
# event_result = cp.EventStudy(data, formula, ...)
# event_result.plot()  # Event-time coefficient plot with HDI
```

---

## Summary

### What's Already Available + Being Built

CausalPy has significant panel/DiD capabilities, with more coming in parallel PRs:

**Already in CausalPy:**
| Class | Purpose |
|-------|---------|
| `StaggeredDifferenceInDifferences` | Imputation-based staggered DiD (Borusyak et al. 2024) |
| `DifferenceInDifferences` | Standard two-period DiD |
| `SyntheticControl` | Synthetic control method |

**In Progress (Open PRs):**
| PR | Class | Purpose |
|----|-------|---------|
| **#670** | `PanelRegression` | General panel models with unit/time FE |
| **#584** | `EventStudy` | Dynamic DiD with event-time coefficient plots |
| **#614** | `PiecewiseITS` | Segmented regression with multiple interventions |

### PanelRegression (#670) Provides

- ✅ Two FE estimation methods (dummies + within)
- ✅ Full Bayesian inference with PyMC (primary focus)
- ✅ Panel-aware visualization (trajectories, unit effects, residuals)
- ✅ Proper handling of large panels via within transformation

### Remaining Opportunities for Future Work (Bayesian Focus)

1. **Counterfactual visualization** (high priority) - plot actual vs. counterfactual with posterior HDI
2. **Hierarchical panel models** (medium priority) - partial pooling across units
3. **Model comparison** (medium priority) - LOO/WAIC for comparing specifications
4. **Bayesian decomposition** (medium priority) - mechanism analysis with full posterior

### CausalPy's Differentiators vs. pyfixest

- **Bayesian inference** with full posterior distributions
- **Causal framing** in all methods and documentation
- **Unified API** across DiD, ITS, SC, RD, Panel methods
- **Uncertainty quantification** via HDI, not just point estimates
- **Not competing on performance** - focus on interpretability and causal inference
