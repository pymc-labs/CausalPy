# PanelRegression Feature Request Issues

This folder contains detailed feature request specifications for future enhancements to CausalPy's panel regression and related capabilities. These were inspired by analyzing `pyfixest` functionality while maintaining CausalPy's Bayesian-first philosophy.

## Prerequisites

All features require **PR #670 (`PanelRegression`)** to be merged first.

## Issue Index

### High Priority

| # | Feature | File | Complexity | Description |
|---|---------|------|------------|-------------|
| 1 | **Counterfactual Visualization** | [01_counterfactual_visualization.md](01_counterfactual_visualization.md) | Medium | Plot actual vs. counterfactual outcomes with HDI |
| 2 | **Model Comparison Tables** | [02_model_comparison_tables.md](02_model_comparison_tables.md) | Medium | Side-by-side comparison with WAIC/LOO |
| 3 | **Placebo Tests** | [03_placebo_sensitivity_analysis.md](03_placebo_sensitivity_analysis.md) | Medium-High | Placebo-in-time and permutation/randomization inference |
| 4 | **Bayesian Decomposition** | [04_bayesian_decomposition.md](04_bayesian_decomposition.md) | Medium | Gelbach-style mediation with posteriors |

### Lower Priority

| # | Feature | File | Complexity | Description |
|---|---------|------|------------|-------------|
| 5 | **High-Dim FE Performance** | [05_high_dimensional_fe_performance.md](05_high_dimensional_fe_performance.md) | Medium | Numba backend for large panels |

## Feature Summaries

### 1. Counterfactual Visualization

Visualize what would have happened without treatment. Shows actual outcomes vs. predicted counterfactual with full Bayesian uncertainty (HDI bands). Similar to existing patterns in `SyntheticControl` and `InterruptedTimeSeries`.

**Key API:**
```python
result.plot_counterfactual(units=["unit_1"], treatment_var="treated")
```

### 2. Model Comparison Tables

Publication-ready side-by-side comparison of multiple model specifications. Includes Bayesian model comparison metrics (WAIC, LOO) instead of AIC/BIC. Outputs to DataFrame, Markdown, LaTeX, HTML.

**Key API:**
```python
cp.compare_models([model1, model2, model3], stats=["n_obs", "waic", "loo"])
```

### 3. Placebo Tests & Sensitivity Analysis

Systematic placebo testing including:
- **Placebo-in-time:** Pretend treatment happened at different times
- **Permutation tests (randomization inference):** Shuffle treatment assignment and compare

Inspired by pyfixest's `ritest()` but with Bayesian uncertainty quantification.

**Key API:**
```python
result.placebo_test_permutation(treatment_var="treated", n_permutations=100)
```

### 4. Bayesian Decomposition (Gelbach-style)

Decompose treatment effects to understand *what explains them*. Based on Gelbach (2016) but with full Bayesian posteriors instead of bootstrap.

**What is Gelbach decomposition?** It answers: "How much of the treatment effect is explained by specific mediators?" For example: "The training program increases wages by 15%. Of this, 8% is explained by credentials, 4% by industry switching, and 3% remains unexplained."

**Key API:**
```python
decomp = cp.decompose(base_model, full_model, treatment_var="treated",
                       mediators=["credentials", "industry"])
decomp.prob_explains_more_than("credentials", threshold=0.25)  # P(credentials explains >25%)
```

### 5. High-Dimensional FE Performance

Optional Numba acceleration for the within transformation on very large panels. Lower priority because MCMC sampling is typically the bottleneck for Bayesian models.

**Key API:**
```python
cp.PanelRegression(..., demeaner_backend="numba")
```

## How to Use These Files

Each file is structured to serve as a GitHub issue draft:

1. **Summary:** One-line description
2. **Motivation:** Why this matters, user story
3. **Proposed API:** Code examples showing desired interface
4. **Implementation Details:** Technical approach with code snippets
5. **Blockers & Prerequisites:** What must come first
6. **Effort Estimate:** Complexity breakdown
7. **Acceptance Criteria:** Checklist for completion
8. **Labels:** Suggested GitHub labels

### Creating GitHub Issues

When ready to implement a feature:

```bash
# Review the file
cat docs/dev/issues/01_counterfactual_visualization.md

# Create the issue (after user review)
gh issue create --title "Feature: Counterfactual Visualization for PanelRegression" \
    --body-file docs/dev/issues/01_counterfactual_visualization.md \
    --label "enhancement" --label "panel-regression"
```

## Context

These features emerged from analyzing [pyfixest](https://github.com/py-econometrics/pyfixest), a high-performance fixed effects package. CausalPy's approach differs significantly:

| Aspect | pyfixest | CausalPy |
|--------|----------|----------|
| **Focus** | Frequentist, performance | Bayesian, interpretability |
| **Uncertainty** | Standard errors, p-values | HDI, posterior distributions |
| **Probability statements** | Not available | Natural: "P(effect > 0.5)?" |
| **Model comparison** | AIC, BIC | WAIC, LOO, ELPD |
| **Inference** | Bootstrap, asymptotic | MCMC, posterior |
| **Speed** | Primary concern (Rust) | Secondary (Python/Numba) |

These features adapt pyfixest's functionality to CausalPy's Bayesian philosophy.

## What These Features Provide That pyfixest Doesn't

| Feature | pyfixest Output | CausalPy Output |
|---------|-----------------|-----------------|
| **Counterfactual** | Point predictions | Posterior predictive with HDI |
| **Model comparison** | p-values, stars | Credible intervals, probability statements |
| **Placebo tests** | Frequentist p-value | P(actual > placebos), full posterior |
| **Decomposition** | Bootstrap CIs | Full posterior, probabilistic ranking |

## Related Documentation

- [Panel Regression Opportunities Report](../panel_regression_opportunities_report.md) - Full analysis
- [pyfixest Functionality Report](../pyfixest_functionality_report.md) - Detailed pyfixest review
- [PR #670](https://github.com/pymc-labs/CausalPy/pull/670) - PanelRegression implementation
