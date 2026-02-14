# PyFixest Functionality Report

This document summarizes the feature set of the [pyfixest](https://github.com/py-econometrics/pyfixest) package to inform potential enhancements for CausalPy's `PanelRegression` class.

**Package Overview:** PyFixest is a Python package for fast high-dimensional fixed effects regression. It closely mirrors the R `fixest` package API and is focused on performance-optimized frequentist estimation.

---

## Core Estimation Capabilities

### 1. Linear Regression with Fixed Effects (`feols`)

**What it does:**
- OLS and WLS regression with fixed-effects demeaning via Frisch-Waugh-Lovell theorem
- Supports high-dimensional fixed effects (thousands of levels)
- Within-transformation via efficient iterative demeaning algorithms

**Key features:**
- Formula syntax: `"Y ~ X1 + X2 | FE1 + FE2"` where `|` separates covariates from fixed effects
- Interacted fixed effects: `fe1^fe2` for interactions
- Multiple estimation: `"Y + Y2 ~ X1 | csw0(f1, f2)"` estimates multiple models at once
- Collinearity detection with configurable tolerance
- Singleton fixed effect removal

**Example:**
```python
import pyfixest as pf
data = pf.get_data()
pf.feols("Y ~ X1 | f1 + f2", data=data).summary()
```

### 2. Instrumental Variables (`feols` with three-part formula)

**What it does:**
- 2SLS estimation with fixed effects
- Three-part formula syntax: `"Y ~ 1 | FE | X1 ~ Z1"`

**Key features:**
- Instruments specified after second `|`
- First stage statistics reported
- Can combine with high-dimensional fixed effects

### 3. Poisson Regression (`fepois`)

**What it does:**
- Poisson regression with high-dimensional fixed effects
- Implements the `ppmlhdfe` algorithm from Stata

**Key features:**
- Same formula syntax as `feols`
- IRLS estimation with fixed effect demeaning at each iteration
- Separation detection

### 4. Quantile Regression (`quantreg`)

**What it does:**
- Quantile regression using an interior point solver
- Estimate effects at different quantiles of the distribution

### 5. GLMs (Probit, Logit, Gaussian)

**What it does:**
- Generalized Linear Models
- Currently without fixed effects demeaning (work in progress)

---

## Variance-Covariance Estimators

PyFixest offers a comprehensive set of variance estimators:

### Standard Errors Types
| Type | Description |
|------|-------------|
| `"iid"` | Homoskedastic standard errors |
| `"hetero"` / `"HC1"` | Heteroskedasticity-robust (White) |
| `"HC2"`, `"HC3"` | Alternative HC corrections |
| `{"CRV1": "cluster_var"}` | Cluster-robust (one-way) |
| `{"CRV3": "cluster_var"}` | Cluster-robust with small-sample correction |
| Two-way clustering | Supported via dictionary syntax |
| `"NW"` | Newey-West HAC (time series) |
| `"DK"` | Driscoll-Kraay HAC (panel) |

**Notable feature:** Standard errors can be adjusted "on-the-fly" after estimation:
```python
fit = pf.feols("Y ~ X1 | f1", data=data)
fit.vcov("hetero").summary()  # Change SE type without re-estimating
```

### Causal Cluster Variance (CCV)

Implements Abadie et al. (2023) "When Should You Adjust Standard Errors for Clustering?" - accounts for both sampling uncertainty and design uncertainty in cluster settings.

---

## Difference-in-Differences Estimators

### 1. Two-Way Fixed Effects (TWFE)

**Class:** `TWFE`

Standard difference-in-differences with unit and time fixed effects:
- Formula: `y ~ is_treated | unit_id + time_id`
- Automatic treatment variable creation based on timing
- Handles "never treated" units

### 2. Gardner's Two-Stage Estimator (DID2S)

**Class:** `DID2S`

Implements Gardner (2021) two-step estimator:
1. First stage: Estimate unit and time fixed effects using only untreated observations
2. Second stage: Regress residualized outcome on treatment indicator

**Why useful:** Corrects for certain biases in staggered DiD with heterogeneous treatment effects.

### 3. Saturated Event Study (Sun & Abraham)

**Class:** `SaturatedEventStudy`

Implements Sun & Abraham (2021) interaction-weighted estimator:
- Fully interacted event study with cohort-specific effect curves
- Aggregation methods to recover ATT by period
- Tests for treatment effect heterogeneity across cohorts
- Visualization of cohort-specific and aggregated effects

**Methods:**
- `aggregate(agg="period")` - Aggregate across cohorts
- `test_treatment_heterogeneity()` - Wald test for heterogeneous effects
- `iplot()` / `iplot_aggregate()` - Visualization

### 4. Local Projections DiD (LPDID)

**Class:** `LPDID`

Implements local projections for DiD following Dube et al. (2023):
- Estimates dynamic treatment effects via separate regressions for each horizon
- More flexible than parametric event study models
- Natural handling of pre/post treatment windows

---

## Inference Features

### 1. Wild Cluster Bootstrap

Integration with [wildboottest](https://github.com/py-econometrics/wildboottest):
- Robust inference with few clusters
- Multiple bootstrap types supported

### 2. Randomization Inference (`ritest`)

Fast randomization inference implementation:
- "Fast" algorithm for OLS (leverages FWL theorem)
- "Slow" algorithm for general models
- Cluster-aware resampling
- Visualization of null distribution

### 3. Multiple Hypothesis Testing

**Functions:**
- `bonferroni(models, param)` - Bonferroni correction
- `rwolf(models, param, reps, seed)` - Romano-Wolf stepdown procedure

**Features:**
- Works across multiple model specifications
- Bootstrap-based critical values
- Can use either wild bootstrap or randomization inference

### 4. Simultaneous Confidence Intervals

Multiplier bootstrap for simultaneous inference on multiple coefficients.

### 5. Wald Tests

General Wald test functionality for linear hypothesis testing:
```python
model.wald_test(R=restriction_matrix, distribution="chi2")
```

---

## Regression Decomposition (Gelbach)

**Class:** `GelbachDecomposition`

Implements Gelbach (2016) decomposition (equivalent to linear mediation):
- Decomposes the effect of a focal variable into explained/unexplained components
- Shows which mediator variables drive the difference between "short" and "long" regressions

**Features:**
- Bootstrap confidence intervals
- Cluster-aware bootstrap
- Multiple output formats (table, waterfall chart)
- Combine covariates into groups

**Methods:**
- `fit()` - Compute decomposition
- `bootstrap()` - Bootstrap CIs
- `tidy()` / `etable()` - Formatted output
- `coefplot()` - Waterfall visualization

---

## Visualization

### 1. Coefficient Plots (`iplot`, `coefplot`)

**Features:**
- Multiple plotting backends: `lets_plot` (default) and `matplotlib`
- Confidence intervals with adjustable alpha
- Horizontal/vertical orientation
- Keep/drop coefficients with regex patterns
- Custom labels and axis intercepts
- Multiple models on same plot

### 2. Event Study Plots

Built-in for DiD classes:
- Pre/post treatment coefficients
- Confidence bands
- Reference period highlighting

### 3. Publication-Ready Tables (`etable`)

**Output formats:**
- Great Tables (HTML)
- LaTeX booktabs
- Markdown
- Plain DataFrame

**Features:**
- Side-by-side model comparison
- Custom coefficient ordering
- Significance stars
- R² and other statistics
- Fixed effects indicators

---

## Performance Features

### 1. GPU Acceleration

Optional CuPy backend for GPU-accelerated demeaning:
```python
pf.feols("Y ~ X1 | f1 + f2", data=data, demean_backend="cupy64")
```

### 2. Multiple Demeaning Backends

| Backend | Description |
|---------|-------------|
| `"numba"` | Default, CPU JIT-compiled |
| `"rust"` | Rust implementation |
| `"jax"` | JAX (CPU/GPU) |
| `"cupy"` / `"cupy32"` / `"cupy64"` | GPU via CuPy |
| `"scipy"` | Sparse matrix operations |

### 3. Data Compression

Lossless compression using sufficient statistics for faster estimation on large datasets:
```python
pf.feols("Y ~ X1 | f1", data=data, use_compression=True)
```

### 4. Rust Core Operations

Some operations (collinearity detection, demeaning, CRV1 computation) implemented in Rust for performance.

---

## Formula Syntax Features

### Multiple Estimation Syntax

Estimate many models efficiently:

| Syntax | Meaning |
|--------|---------|
| `Y + Y2 ~ X1` | Multiple dependent variables |
| `Y ~ X1 + X2 + sw(X3, X4)` | Stepwise: include X3, then X4 |
| `Y ~ csw(X1, X2)` | Cumulative stepwise: X1, then X1+X2 |
| `Y ~ csw0(X1, X2)` | Start with no vars, then add |
| `Y ~ X1 \| sw(f1, f2)` | Stepwise fixed effects |

### Interactions

- `i(var1, var2)` - Interact variables for coefficient extraction
- `fe1^fe2` - Interacted fixed effects

---

## Post-Estimation Methods

### Core Methods
| Method | Description |
|--------|-------------|
| `.summary()` | Print formatted summary |
| `.tidy()` | Return tidy DataFrame |
| `.coef()` | Extract coefficients |
| `.se()` | Extract standard errors |
| `.tstat()` | Extract t-statistics |
| `.pvalue()` | Extract p-values |
| `.confint()` | Confidence intervals |
| `.resid()` | Residuals |
| `.predict()` | Predictions (in/out of sample) |
| `.fixef()` | Extract fixed effects |

### Advanced Methods
| Method | Description |
|--------|-------------|
| `.vcov()` | Change variance estimator |
| `.wald_test()` | Hypothesis testing |
| `.ritest()` | Randomization inference |
| `.wildboottest()` | Wild cluster bootstrap |
| `.decompose()` | Gelbach decomposition |

---

## Key Design Principles

1. **Fixest API Compatibility:** Mirrors R `fixest` syntax closely
2. **Performance First:** Optimized for large datasets with high-dimensional FEs
3. **Frequentist Focus:** No Bayesian inference
4. **Modularity:** Separate estimation and inference steps
5. **Multiple Backends:** Users can choose compute backends
6. **On-the-fly Adjustment:** Re-compute SEs without re-estimating model

---

## Limitations / Gaps

- No Bayesian inference
- GLM fixed effects still work-in-progress
- No built-in synthetic control methods
- No explicit causal framing (focuses on estimation mechanics)
- No uncertainty quantification for fixed effect estimates

---

## References

- Bergé, L. (2018). Efficient estimation of maximum likelihood models with multiple fixed-effects: the R package FENmlm. CREA Discussion Paper.
- Gardner, J. (2021). Two-stage differences in differences. Working Paper.
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. Journal of Econometrics.
- Abadie, A., et al. (2023). When should you adjust standard errors for clustering? Quarterly Journal of Economics.
- Gelbach, J. B. (2016). When do covariates matter? And which ones, and how much? Journal of Labor Economics.
- Dube, A., Girardi, D., Jordà, Ò., & Taylor, A. M. (2023). A local projections approach to difference-in-differences event studies. NBER Working Paper.
