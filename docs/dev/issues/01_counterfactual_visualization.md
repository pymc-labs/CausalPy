# Feature Request: Counterfactual Predictions & Visualization for PanelRegression

> **Prerequisite:** This feature builds upon the Panel Fixed Effects PR ([#670](https://github.com/pymc-labs/CausalPy/pull/670)), which must be merged first.

## Summary

Add a `plot_counterfactual()` method to `PanelRegression` that visualizes observed outcomes versus predicted counterfactual outcomes (what would have happened without treatment), with full Bayesian uncertainty quantification.

## Motivation

### Why This Matters

Counterfactual visualization is central to causal inference. When we estimate a treatment effect, we're implicitly computing:

```
τ = Y(1) - Y(0)
```

Where Y(0) is the counterfactual outcome under no treatment. Currently, `PanelRegression` estimates treatment coefficients but doesn't provide a direct way to visualize what individual units would have looked like without treatment.

### The Fundamental Problem of Causal Inference

The core challenge in causal inference is that we can never observe both potential outcomes for the same unit. If Company A adopts a new policy in 2020, we observe their revenue *with* the policy, but we can never directly observe what their revenue would have been *without* the policy. This unobserved quantity is the **counterfactual**.

Panel regression with fixed effects helps by:
1. **Unit fixed effects:** Controlling for time-invariant differences between units
2. **Time fixed effects:** Controlling for common shocks affecting all units
3. **Treatment coefficient:** Estimating the average difference between treated and control conditions

But the coefficient alone doesn't tell the full story. Stakeholders often want to see:
- "What would Company A's trajectory have looked like without the policy?"
- "How big is the gap between what actually happened and what would have happened?"
- "How confident are we in this counterfactual prediction?"

### User Story

> "I've fitted a panel model with treatment effects. I want to show stakeholders what Company X's revenue trajectory would have looked like if they hadn't adopted the new policy. I need a plot showing actual revenue vs. counterfactual revenue with uncertainty bands."

### Precedent in CausalPy

This pattern already exists in other CausalPy experiment classes:
- `SyntheticControl.plot()` shows actual vs. synthetic counterfactual with HDI bands
- `DifferenceInDifferences.plot()` shows pre/post with counterfactual projection
- `InterruptedTimeSeries.plot()` shows actual vs. counterfactual trend

Here's how `SyntheticControl` does it:

```python
# From causalpy/experiments/synthetic_control.py
# Post-intervention period - showing counterfactual
h_line, h_patch = plot_xY(
    self.datapost.index,
    post_pred,  # Counterfactual prediction
    ax=ax[0],
    plot_hdi_kwargs={"color": "C1"},
)

# Shaded causal effect
h = ax[0].fill_between(
    self.datapost.index,
    y1=post_pred.mean(dim=["chain", "draw"]).values,  # Counterfactual
    y2=self.datapost_treated.sel(treated_units=treated_unit).values,  # Actual
    color="C0",
    alpha=0.25,
    label="Causal impact",
)
```

`PanelRegression` should offer similar capabilities.

## Proposed API

```python
# Basic usage
result = cp.PanelRegression(
    data=df,
    formula="y ~ 1 + treated + x1 + x2",
    unit_fe_variable="unit",
    time_fe_variable="time",
    model=cp.pymc_models.LinearRegression(...)
)

# Counterfactual plot for specific units
fig, ax = result.plot_counterfactual(
    units=["unit_1", "unit_2"],  # Which units to plot
    treatment_var="treated",     # Name of treatment indicator
    hdi_prob=0.94,               # Credible interval width
    show_effect=True,            # Shade the treatment effect region
)

# Get counterfactual predictions as data
cf_data = result.get_counterfactual_predictions(
    treatment_var="treated",
    hdi_prob=0.94
)
# Returns DataFrame with: unit, time, y_actual, y_counterfactual, y_cf_lower, y_cf_upper, effect
```

### Alternative: Integrated into existing plot method

```python
# Could also be a mode in the main plot method
result.plot(kind="counterfactual", units=["unit_1", "unit_2"])
result.plot(kind="trajectories")  # Current behavior
result.plot(kind="coefficients")  # Current behavior
```

## How This Differs from pyfixest

pyfixest provides a standard `predict(newdata)` method that returns point estimates:

```python
# pyfixest approach
model.predict(newdata=df_counterfactual)
# Returns: array of point predictions, no uncertainty
```

CausalPy's Bayesian approach offers significant advantages:

| Aspect | pyfixest | CausalPy (proposed) |
|--------|----------|---------------------|
| Prediction | `predict(newdata)` point estimate | Full posterior predictive distribution |
| Uncertainty | None built-in | HDI bands on counterfactual |
| Visualization | Not provided | Purpose-built counterfactual plot |
| Framing | Statistical prediction | Causal counterfactual |
| Output | Single number per observation | Distribution of plausible values |

## Implementation Details

### Core Logic

1. **Identify treatment variable** in the formula
2. **Create counterfactual design matrix** by setting treatment = 0
3. **Generate posterior predictive** for counterfactual scenario
4. **Compute treatment effect** as actual - counterfactual (with full posterior)
5. **Visualize** with HDI bands

### Step-by-Step Algorithm

```python
def get_counterfactual_predictions(self, treatment_var: str, hdi_prob: float = 0.94):
    """
    Compute counterfactual predictions (what would have happened without treatment).

    Parameters
    ----------
    treatment_var : str
        Name of the binary treatment variable in the formula
    hdi_prob : float
        Width of the HDI interval (default 0.94)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: unit, time, y_actual, y_counterfactual_mean,
        y_cf_lower, y_cf_upper, effect_mean, effect_lower, effect_upper
    """
    # Step 1: Create counterfactual data (treatment = 0 for all)
    data_cf = self.data.copy()
    data_cf[treatment_var] = 0

    # Step 2: Build counterfactual design matrix
    # (reuses patsy formula parsing from fit)
    y_cf, X_cf = dmatrices(self.formula, data_cf)

    # Step 3: Generate posterior predictive
    # For Bayesian models, this uses the full posterior
    with self.model:
        pm.set_data({"X": X_cf})
        cf_posterior = pm.sample_posterior_predictive(
            self.idata,
            var_names=["mu", "y_pred"]
        )

    # Step 4: Compute effect = actual - counterfactual
    y_actual = self.data[self.outcome_var].values
    y_cf_samples = cf_posterior.posterior_predictive["mu"]
    effect_samples = y_actual - y_cf_samples

    # Step 5: Summarize with HDI
    y_cf_mean = y_cf_samples.mean(dim=["chain", "draw"])
    y_cf_hdi = az.hdi(y_cf_samples, hdi_prob=hdi_prob)

    return pd.DataFrame({
        "unit": self.data[self.unit_fe_variable],
        "time": self.data[self.time_fe_variable],
        "y_actual": y_actual,
        "y_counterfactual_mean": y_cf_mean.values,
        "y_cf_lower": y_cf_hdi.sel(hdi="lower").values,
        "y_cf_upper": y_cf_hdi.sel(hdi="higher").values,
        "effect_mean": effect_samples.mean(dim=["chain", "draw"]).values,
    })
```

### Handling the Treatment Variable

The method needs to identify which variable represents treatment. Options:
- **Explicit parameter:** `treatment_var="treated"` (safest)
- **Inference from formula:** Look for binary 0/1 variable (risky)
- **Store at fit time:** If user specifies during `__init__`

**Recommendation:** Require explicit `treatment_var` parameter.

### Fixed Effects in Counterfactual

When computing counterfactuals:
- **Unit FE:** Keep the same (unit identity doesn't change)
- **Time FE:** Keep the same (time period doesn't change)
- **Only change:** Treatment indicator and any treatment interactions

### Within Transformation Complexity

For `fe_method="within"`:
- Demeaned data doesn't include unit intercepts
- Need to add back unit means when plotting absolute counterfactuals
- Already handled in `plot_trajectories()` - can reuse that logic

```python
# For within transformation, recover absolute predictions
if self.fe_method == "within":
    # Add back group means stored during transformation
    y_cf_absolute = y_cf_demeaned + self.unit_means + self.time_means
```

### Visualization Design

```
     │
  Y  │    ●───●───●───●     Actual (solid line)
     │         ╲
     │          ╲  Treatment effect (shaded)
     │           ╲
     │    ○───○───○───○     Counterfactual (dashed line + HDI band)
     │
     └────────────────────
           Pre    │  Post
                  ▲
            Treatment onset
```

### Dependencies

- Requires `posterior_predictive` from PyMC model (already computed)
- Needs treatment variable identification
- For OLS: point estimates only (no HDI)

## Blockers & Prerequisites

### Required Before Implementation

1. **PR #670 merged:** `PanelRegression` class must be stable
2. **Clear treatment variable handling:** Decide on API for specifying treatment

### No Hard Blockers

This is a relatively self-contained feature that doesn't require changes to other parts of CausalPy.

## Effort Estimate

| Component | Complexity |
|-----------|------------|
| Core counterfactual prediction logic | Low |
| Integration with within transformation | Medium |
| Visualization with HDI | Low |
| Tests and documentation | Medium |
| **Total** | **Medium** (~1-2 days) |

## Acceptance Criteria

- [ ] `plot_counterfactual()` method added to `PanelRegression`
- [ ] Works with both `fe_method="dummies"` and `fe_method="within"`
- [ ] Shows HDI bands for Bayesian models
- [ ] Falls back gracefully for OLS models (point estimates only)
- [ ] `get_counterfactual_predictions()` returns DataFrame with all quantities
- [ ] Unit tests covering basic functionality
- [ ] Example in docstring

## Related Issues / PRs

- PR #670: `PanelRegression` (prerequisite)
- Existing pattern: `SyntheticControl.plot()` counterfactual visualization

## Labels

`enhancement`, `panel-regression`, `visualization`
