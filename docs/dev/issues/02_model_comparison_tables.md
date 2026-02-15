# Feature Request: Model Comparison Tables for CausalPy Experiments

> **Prerequisite:** This feature builds upon the Panel Fixed Effects PR ([#670](https://github.com/pymc-labs/CausalPy/pull/670)), which must be merged first.

## Summary

Add a `compare_models()` function that creates publication-ready side-by-side comparison tables for multiple CausalPy experiment results, with Bayesian model comparison metrics (LOO, WAIC) alongside traditional statistics.

## Motivation

### Why This Matters

Robustness checks are fundamental to credible causal inference. Researchers routinely fit multiple model specifications to show that results are not sensitive to:
- Different control variables
- Different fixed effects structures
- Different sample restrictions
- Different functional forms

A single coefficient estimate is never sufficient for a credible causal claim. Reviewers and stakeholders expect to see sensitivity analysis showing the treatment effect is stable across reasonable alternative specifications.

Currently, comparing CausalPy models requires manual extraction and formatting of results—a tedious and error-prone process.

### User Story

> "I've fitted 4 different `PanelRegression` specifications to test robustness. I need a single table showing treatment effects across all models, with standard errors/HDI, sample sizes, and model fit statistics. This table will go directly into my paper."

### Precedent in pyfixest

pyfixest provides `etable()` for side-by-side regression tables. Here's how it works:

```python
import pyfixest as pf

# Fit multiple models
fit1 = pf.feols("Y ~ X1 + X2 | f1", df)
fit2 = pf.feols("Y ~ X1 + X2 | f1 + f2", df)

# Create comparison table
pf.etable([fit1, fit2])
```

The `etable()` function:
1. Extracts coefficients and standard errors from each model
2. Aligns variables across models (shows "—" for missing variables)
3. Shows significance stars based on p-values
4. Includes model fit statistics (R², N, fixed effects indicators)
5. Outputs to multiple formats (DataFrame, Markdown, LaTeX, HTML via Great Tables)

pyfixest delegates to the `maketables` package for formatting:

```python
# From pyfixest/report/summarize.py
table = maketables.ETable(
    models=models_list,
    signif_code=signif_code,
    coef_fmt=coef_fmt,
    custom_stats=custom_stats,
    ...
)
```

### Other Precedents

- **statsmodels:** `summary_col()` for regression comparison
- **R stargazer:** Publication-ready regression tables
- **R modelsummary:** Modern alternative to stargazer

## Proposed API

### Option 1: Standalone Function (Recommended)

```python
import causalpy as cp

# Fit multiple models
model1 = cp.PanelRegression(df, "y ~ treated", unit_fe_variable="unit", ...)
model2 = cp.PanelRegression(df, "y ~ treated + x1", unit_fe_variable="unit", ...)
model3 = cp.PanelRegression(df, "y ~ treated + x1 + x2", unit_fe_variable="unit", ...)
model4 = cp.PanelRegression(df, "y ~ treated + x1 + x2", unit_fe_variable="unit",
                             time_fe_variable="time", ...)

# Compare models
table = cp.compare_models(
    [model1, model2, model3, model4],
    model_names=["(1) Base", "(2) + Controls", "(3) + More", "(4) Two-way FE"],
    coefficients=["treated"],  # Which coefficients to show (None = all non-FE)
    stats=["n_obs", "n_units", "r_squared", "waic", "loo"],  # Which statistics
    hdi_prob=0.94,  # For Bayesian models
    output="dataframe"  # or "markdown", "latex", "html"
)
```

### Option 2: Method on Experiment Class

```python
# Compare with other models
model1.compare_with(
    [model2, model3, model4],
    model_names=["(1)", "(2)", "(3)", "(4)"]
)
```

**Recommendation:** Option 1 (standalone function) is more flexible and follows pyfixest's pattern.

### Output Format Examples

#### DataFrame Output

```
                    (1) Base    (2) + Controls  (3) + More   (4) Two-way FE
treated              0.523          0.487         0.491          0.445
                   [0.31, 0.74]  [0.29, 0.68]  [0.30, 0.69]   [0.26, 0.63]
x1                      —           0.121         0.098          0.102
                                 [0.02, 0.22]  [-0.01, 0.21]  [0.01, 0.19]
x2                      —             —           0.045          0.039
                                              [-0.05, 0.14]  [-0.04, 0.12]
───────────────────────────────────────────────────────────────────────────
N                     500           500           500            500
Units                  50            50            50             50
Unit FE               Yes           Yes           Yes            Yes
Time FE                No            No            No            Yes
R²                   0.234         0.267         0.271          0.412
WAIC                 1523          1498          1495           1389
LOO                  1525          1501          1498           1392
```

Note the key differences from pyfixest:
- **HDI intervals** instead of standard errors in parentheses
- **WAIC/LOO** for Bayesian model comparison instead of AIC/BIC
- **No significance stars** (Bayesian credible intervals are more informative)

#### Markdown Output

```markdown
| Coefficient | (1) Base | (2) + Controls | (3) + More | (4) Two-way FE |
|-------------|----------|----------------|------------|----------------|
| treated     | 0.523    | 0.487          | 0.491      | 0.445          |
|             | [0.31, 0.74] | [0.29, 0.68] | [0.30, 0.69] | [0.26, 0.63] |
| ...         | ...      | ...            | ...        | ...            |
```

## Understanding Bayesian Model Comparison Metrics

### What is WAIC?

**WAIC (Widely Applicable Information Criterion)** is a Bayesian generalization of AIC that estimates out-of-sample prediction accuracy. It uses the full posterior distribution rather than a point estimate.

```python
import arviz as az

# Compute WAIC from InferenceData
waic = az.waic(model.idata)
print(waic.waic)  # WAIC value (lower is better)
print(waic.waic_se)  # Standard error of WAIC
```

### What is LOO-CV?

**LOO (Leave-One-Out Cross-Validation)** uses Pareto-smoothed importance sampling to efficiently approximate leaving out each observation. It's generally preferred over WAIC for model comparison.

```python
# Compute LOO from InferenceData
loo = az.loo(model.idata)
print(loo.loo)  # LOO value (lower is better)
print(loo.pareto_k)  # Diagnostic - values > 0.7 indicate problems
```

### Why These Matter for Causal Inference

When comparing model specifications:
- **WAIC/LOO closer to 0:** Better predictive accuracy
- **Difference in WAIC/LOO:** Indicates strength of preference between models
- **Standard errors:** Quantify uncertainty in the comparison

ArviZ provides `az.compare()` for formal model comparison:

```python
# Compare multiple models
comparison = az.compare({
    "Model 1": model1.idata,
    "Model 2": model2.idata,
    "Model 3": model3.idata,
})
```

## Implementation Details

### Core Logic

1. **Extract coefficients** from each model's posterior/estimates
2. **Compute comparison statistics** (WAIC, LOO for Bayesian)
3. **Align coefficients** across models (handle missing/different variables)
4. **Format output** with intervals and statistics

### Coefficient Extraction

```python
def _extract_coefficients(model, hdi_prob=0.94):
    """Extract coefficient summaries from a CausalPy model."""
    if isinstance(model.model, PyMCModel):
        # Bayesian model - extract from posterior
        posterior = model.idata.posterior

        results = {}
        for var in model.labels:
            if var in posterior.data_vars:
                samples = posterior[var]
                results[var] = {
                    "mean": float(samples.mean()),
                    "hdi_lower": float(az.hdi(samples, hdi_prob=hdi_prob)["lower"]),
                    "hdi_upper": float(az.hdi(samples, hdi_prob=hdi_prob)["upper"]),
                }
        return results
    else:
        # OLS model - extract coefficients and (if available) confidence intervals
        coef = model.model.coef_
        return {
            label: {"mean": coef[i], "se": getattr(model.model, "standard_errors_", [None]*len(coef))[i]}
            for i, label in enumerate(model.labels)
        }
```

### Statistics Computation

```python
def _compute_model_stats(model, stats_requested):
    """Compute requested statistics for a model."""
    stats = {}

    if "n_obs" in stats_requested:
        stats["N"] = model.n_obs

    if "n_units" in stats_requested and hasattr(model, "n_units"):
        stats["Units"] = model.n_units

    if "r_squared" in stats_requested:
        stats["R²"] = model.score

    if isinstance(model.model, PyMCModel):
        if "waic" in stats_requested:
            try:
                waic = az.waic(model.idata)
                stats["WAIC"] = f"{waic.waic:.1f}"
            except Exception:
                stats["WAIC"] = "—"

        if "loo" in stats_requested:
            try:
                loo = az.loo(model.idata)
                stats["LOO"] = f"{loo.loo:.1f}"
            except Exception:
                stats["LOO"] = "—"

    return stats
```

### Handling Mixed Model Types

The function should work with:
- Multiple `PanelRegression` results
- Mix of Bayesian and OLS models
- Potentially other experiment classes (`DifferenceInDifferences`, etc.)

```python
# Mixed model types
bayes_model = cp.PanelRegression(df, "y ~ treated", model=cp.pymc_models.LinearRegression())
ols_model = cp.PanelRegression(df, "y ~ treated", model=LinearRegression())

cp.compare_models([bayes_model, ols_model],
                   stats=["n_obs", "r_squared"])  # Only show common stats
```

### Log-Likelihood Requirement

WAIC and LOO require `log_likelihood` in the InferenceData. The PyMC models may need to be updated to ensure this is computed:

```python
# In pymc_models.py, ensure log_likelihood is computed
with model:
    idata = pm.sample(..., idata_kwargs={"log_likelihood": True})
```

### Statistics to Support

| Statistic | Bayesian | OLS | Description |
|-----------|----------|-----|-------------|
| `n_obs` | ✅ | ✅ | Number of observations |
| `n_units` | ✅ | ✅ | Number of panel units |
| `n_periods` | ✅ | ✅ | Number of time periods |
| `r_squared` | ✅ | ✅ | R² (Bayesian: posterior mean) |
| `waic` | ✅ | ❌ | Widely Applicable IC |
| `loo` | ✅ | ❌ | Leave-One-Out CV |
| `elpd_diff` | ✅ | ❌ | ELPD difference from best model |
| `aic` | ❌ | ✅ | Akaike IC (OLS only) |
| `bic` | ❌ | ✅ | Bayesian IC (OLS only) |
| `unit_fe` | ✅ | ✅ | Whether unit FE included |
| `time_fe` | ✅ | ✅ | Whether time FE included |
| `fe_method` | ✅ | ✅ | "dummies" or "within" |

### Output Formats

| Format | Use Case |
|--------|----------|
| `"dataframe"` | Further processing in Python |
| `"markdown"` | README, Jupyter, documentation |
| `"latex"` | Academic papers |
| `"html"` | Web reports, notebooks |

## Differentiation from pyfixest

| Aspect | pyfixest | CausalPy (proposed) |
|--------|----------|---------------------|
| Model comparison | AIC, BIC, R² | WAIC, LOO, ELPD (Bayesian) |
| Uncertainty | Standard errors, p-values | HDI intervals |
| Significance | Stars based on p-values | Credible interval excludes 0 |
| Model types | OLS, IV, Poisson | Bayesian + OLS experiments |
| Underlying library | maketables | Custom or pandas styling |

## Blockers & Prerequisites

### Required Before Implementation

1. **PR #670 merged:** `PanelRegression` must be stable
2. **Log-likelihood in PyMC models:** Ensure WAIC/LOO can be computed
3. **Consistent coefficient access:** Standardize how coefficients are extracted across experiment classes

### Potential Challenges

- **ArviZ dependency:** Already a dependency via PyMC
- **Log-likelihood computation:** May need to add to `pymc_models.py`
- **Cross-experiment compatibility:** Need common interface for coefficient extraction

## Effort Estimate

| Component | Complexity |
|-----------|------------|
| Coefficient extraction logic | Low |
| Statistics computation (WAIC, LOO) | Medium |
| Output formatting (4 formats) | Medium |
| Cross-model alignment | Low |
| Tests and documentation | Medium |
| **Total** | **Medium** (~2-3 days) |

## Acceptance Criteria

- [ ] `cp.compare_models()` function implemented
- [ ] Works with `PanelRegression` (both Bayesian and OLS)
- [ ] Works with `DifferenceInDifferences`, `SyntheticControl`, `InterruptedTimeSeries`
- [ ] Outputs to DataFrame, Markdown, LaTeX, HTML
- [ ] Includes Bayesian comparison metrics (WAIC, LOO) when available
- [ ] Handles missing coefficients gracefully (shows "—")
- [ ] Unit tests for each output format
- [ ] Example in documentation

## Related Issues / PRs

- PR #670: `PanelRegression` (prerequisite)
- ArviZ model comparison: `az.compare()`
- pyfixest `etable()` for inspiration

## Labels

`enhancement`, `reporting`, `bayesian`
