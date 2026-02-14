# Panel Regression Feature Ideas

This document captures potential enhancements for the `PanelRegression` class, drawing inspiration from the R and Python ecosystems (e.g., `fixest`, `plm`, `linearmodels`, `pyfixest`).

---

## 1. Counterfactual Predictions

**What:** Predict outcomes under the counterfactual scenario where `D=0` for treated units, then plot actual vs counterfactual trajectories.

**Why useful:** Visualizes the treatment effect as the gap between observed and counterfactual outcomes over time. Similar to synthetic control visualizations.

**Considerations:**
- Straightforward for Bayesian models (just set `treatment=0` in prediction)
- Could show HDI around counterfactual
- May be redundant if the treatment effect is already summarized well

**Priority:** Medium - nice visualization but may not add much beyond coefficient estimates

---

## 2. Event Study Plots

**What:** Plot treatment effect coefficients relative to the treatment time (e.g., coefficients for t-3, t-2, t-1, t, t+1, t+2...).

**Why useful:**
- Standard way to visualize dynamic treatment effects
- Tests parallel trends assumption (pre-treatment coefficients should be ~0)
- Shows how effects evolve post-treatment

**Implementation:**
- Requires reformulating the model with time-to-treatment indicators
- `fixest` has `i(time_to_treat, ref = -1)` syntax

**Priority:** High - very commonly requested, standard in applied work

---

## 3. Clustered Standard Errors

**What:** Cluster standard errors at the unit level (or other levels) for OLS models.

**Why useful:** Panel data typically has within-unit correlation that inflates standard errors if ignored.

**Current state:** Bayesian models naturally handle this through hierarchical structure, but OLS doesn't.

**Implementation:** Could use `statsmodels` clustered covariance or implement manually.

**Priority:** High for OLS users

---

## 4. Goodman-Bacon Decomposition

**What:** Decompose the TWFE estimator into a weighted average of all 2x2 DiD comparisons.

**Why useful:**
- Diagnoses problems with staggered treatment timing
- Shows which comparisons drive the overall estimate
- Identifies negative weights (bad comparisons)

**Reference:** Goodman-Bacon (2021)

**Priority:** Medium-High - increasingly standard in applied work

---

## 5. Pre-Trend Testing / Placebo Tests

**What:** Formal tests for parallel trends in pre-treatment period.

**Options:**
- Joint F-test on pre-treatment leads
- Plot with confidence bands
- Rambachan & Roth (2023) sensitivity analysis

**Priority:** Medium - event study plots often suffice

---

## 6. Hausman Test (Fixed vs Random Effects)

**What:** Test whether fixed effects are necessary or if random effects would suffice.

**Why useful:** Random effects are more efficient if the assumption holds.

**Considerations:** Less relevant in causal inference context where FE is usually required.

**Priority:** Low - FE is typically the right choice for causal inference

---

## 7. Correlated Random Effects (Mundlak Approach)

**What:** Include group means of time-varying covariates in a random effects model.

**Why useful:**
- Equivalent to FE for coefficients of interest
- But also estimates coefficients on time-invariant variables (which FE can't do)

**Priority:** Medium - useful when time-invariant covariates matter

---

## 8. Heterogeneous Treatment Effects

**What:** Allow treatment effects to vary by:
- Time since treatment
- Unit characteristics
- Treatment intensity

**Implementation:** Interaction terms, but could provide helper methods.

**Priority:** Medium

---

## 9. Staggered Difference-in-Differences

**What:** Modern DiD estimators that handle staggered treatment timing correctly.

**Options:**
- Callaway & Sant'Anna (2021)
- Sun & Abraham (2021)
- de Chaisemartin & D'Haultfœuille (2020)

**Why useful:** Standard TWFE can be biased with staggered timing and heterogeneous effects.

**Priority:** High - major gap in current causal inference tools

---

## 10. Bootstrap Inference

**What:** Bootstrap confidence intervals, especially useful for:
- Small number of clusters
- Non-standard test statistics

**Options:**
- Wild cluster bootstrap
- Pairs cluster bootstrap

**Priority:** Medium

---

## 11. First Differences Estimator

**What:** Alternative to within transformation: regress $\Delta y_{it}$ on $\Delta x_{it}$.

**Why useful:**
- More efficient under certain error structures
- Useful comparison to FE

**Priority:** Low - within transformation is more common

---

## 12. Diagnostics Suite

**What:** Comprehensive diagnostics beyond current residual plots:
- Serial correlation tests (Wooldridge test)
- Heteroskedasticity tests
- Cross-sectional dependence tests (Pesaran CD)
- Multicollinearity diagnostics

**Priority:** Medium

---

## 13. Model Comparison Tools

**What:** Easy comparison of multiple specifications:
- Side-by-side coefficient tables
- Information criteria (AIC, BIC)
- R² comparison

**Similar to:** `fixest::etable()` or `stargazer`

**Priority:** Medium - useful for robustness checks

---

## 14. Prediction Methods

**What:** Enhanced prediction capabilities:
- `predict(newdata=...)`
- Out-of-sample prediction
- Prediction intervals

**Priority:** Medium

---

## 15. Integration with Other CausalPy Methods

**What:** Seamless handoff to other causal methods:
- If FE assumptions fail → suggest synthetic control
- Connection to DiD class
- Combine with propensity score methods

**Priority:** Low - design consideration

---

## Summary: Priority Ranking

| Priority | Feature |
|----------|---------|
| **High** | Event study plots |
| **High** | Clustered standard errors (OLS) |
| **High** | Staggered DiD estimators |
| **Medium-High** | Goodman-Bacon decomposition |
| **Medium** | Counterfactual predictions |
| **Medium** | Bootstrap inference |
| **Medium** | Heterogeneous treatment effects |
| **Medium** | Diagnostics suite |
| **Medium** | Model comparison tools |
| **Low** | Hausman test |
| **Low** | First differences |

---

## References

- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*.
- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*.
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*.
- de Chaisemartin, C., & D'Haultfœuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. *American Economic Review*.
- Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends. *Review of Economic Studies*.
