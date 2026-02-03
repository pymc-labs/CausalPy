# Estimands in CausalPy

Understanding **what** a method estimates is just as important as knowing **how** to use it. This page documents the causal estimands, computation approaches, and key assumptions for CausalPy's core methods.

## Why Estimands Matter

Different causal inference methods target different causal quantities. Misunderstanding what a method estimates can lead to:

- Choosing the wrong method for your research question
- Misinterpreting the magnitude or meaning of effects
- Drawing incorrect conclusions about generalizability

CausalPy methods fall into two broad categories:

- **Parametric interaction models** (DiD, ANCOVA): Estimate treatment effects via model coefficients representing population-level averages
- **Pre-post counterfactual models** (ITS, SC): Estimate time-varying, unit-specific impacts by comparing observed outcomes to counterfactual predictions

This distinction affects how you should interpret results and plots.

:::{note}
The estimand is not always uniquely determined by the experiment class. Some methods have explicit parameters that change the target (e.g., IPW's `weighting_scheme` selects between ATE and ATO). Others produce "local" effects whose scope depends on design choices (e.g., RD bandwidth, IV instrument). The descriptions below assume standard usage.
:::

## Quick Reference

- **Difference-in-Differences**: {term}`ATT` via interaction coefficient
- **Interrupted Time Series**: Time-varying causal impact for a treated unit
- **Synthetic Control**: Time-varying causal impact for a treated unit
- **Regression Discontinuity**: Local treatment effect at the cutoff

---

## Difference-in-Differences (DiD)

### Estimand

The {term}`Average treatment effect on the treated` (ATT): the average causal effect of treatment on the units that received it, in the post-treatment period.

### Computation

CausalPy estimates the ATT as the coefficient on the interaction term between the group indicator and the post-treatment indicator:

```
causal_impact = β[group × post_treatment]
```

This coefficient represents the difference in the change over time between the treatment and control groups.

### Key Assumptions

- **{term}`Parallel trends assumption`**: In the absence of treatment, the treatment and control groups would have followed the same trend over time
- **No anticipation**: Units do not change behavior before treatment begins
- **SUTVA (Stable Unit Treatment Value Assumption)**: No spillovers between units; treatment effect is the same regardless of how many others are treated

### Interpretation Note

:::{note}
The plots show counterfactual trajectories (what would have happened to the treatment group without treatment), but the **reported causal impact is the interaction coefficient**, not the visual difference at a single time point. For experiments with multiple post-treatment periods, the coefficient represents the average effect across all post-treatment observations.
:::

---

## Interrupted Time Series (ITS)

### Estimand

**Time-varying causal impact** for a single treated unit (or aggregate): the difference between the observed outcome and the counterfactual prediction at each post-intervention time point.

This is **not** a population-level {term}`ATE` or {term}`ATT`. It is a unit-specific, time-indexed effect.

### Computation

CausalPy computes the causal impact as:

```
impact(t) = Y_observed(t) - E[Y_counterfactual(t)]
```

Where:
- The counterfactual is predicted by a model trained only on pre-intervention data
- For Bayesian models, the impact uses the posterior expectation (`mu`) rather than the posterior predictive (`y_hat`), representing the systematic causal effect excluding observation-level noise

The cumulative impact sums these effects over the post-intervention period.

### Key Assumptions

- **Stable pre-intervention relationship**: The model correctly captures the pre-intervention trend and seasonality
- **No concurrent shocks**: No other events occur at treatment time that would affect the outcome
- **Correct counterfactual model**: The model specification (trend, seasonality, covariates) accurately represents what would have happened without intervention

### Interpretation Note

:::{note}
The causal impact varies over time and is specific to the treated unit. Summarizing with a single number (e.g., average impact) loses information about the temporal pattern. The `effect_summary()` method provides both point-in-time and cumulative statistics with appropriate uncertainty intervals.
:::

---

## Synthetic Control (SC)

### Estimand

**Time-varying causal impact** for a treated unit: the difference between the observed outcome and a synthetic counterfactual constructed from weighted control units.

Like ITS, this is **not** a population-level effect. It estimates what would have happened to the specific treated unit if it had not received treatment.

### Computation

CausalPy computes the causal impact as:

```
impact(t) = Y_treated(t) - Σ(w_i × Y_control_i(t))
```

Where:
- Weights `w_i` are learned from pre-intervention data to best approximate the treated unit
- For `WeightedSumFitter`, weights are constrained to be non-negative and sum to 1
- For Bayesian models, the impact uses the posterior expectation (`mu`) rather than the posterior predictive

### Key Assumptions

- **Parallel trends in weighted combination**: The weighted combination of control units would have followed the same trajectory as the treated unit in the absence of treatment
- **No spillovers**: Treatment of one unit does not affect control units
- **Convex hull coverage**: The treated unit's pre-intervention characteristics can be well-approximated by a weighted combination of controls
- **No concurrent shocks**: No other events differentially affect the treated unit at treatment time

### Interpretation Note

:::{note}
The synthetic control method is designed for comparative case studies with a single treated unit (or small number of treated units). The effect is specific to that unit and time period—generalization to other units or time periods requires additional assumptions.
:::

---

## Regression Discontinuity (RD)

### Estimand

**Local average treatment effect at the cutoff**: the causal effect of treatment for units exactly at the threshold where treatment assignment changes.

This is a highly local estimate—it applies to units at the margin of treatment, not to the full population.

### Computation

CausalPy estimates the discontinuity as:

```
causal_impact = lim(x→c⁺) E[Y|X=x] - lim(x→c⁻) E[Y|X=x]
```

Where `c` is the treatment threshold. In practice, this is computed as the jump in predicted outcomes at the cutoff, either via:
- The coefficient on the treatment indicator (sharp RD)
- The difference in model predictions just above and below the threshold

### Key Assumptions

- **Continuity at cutoff**: In the absence of treatment, the conditional expectation of the outcome would be continuous at the threshold
- **No manipulation**: Units cannot precisely control their value of the {term}`running variable` to sort around the cutoff
- **Local validity**: The treatment effect is only identified at the cutoff; extrapolation requires additional assumptions
- **Correct functional form**: The relationship between the running variable and outcome is correctly specified on both sides of the cutoff

### Interpretation Note

:::{note}
The RD effect is **local** to the cutoff. Units far from the threshold may experience different treatment effects. The bandwidth parameter controls how much data is used for estimation—narrower bandwidths are more local but have higher variance.
:::

---

## Summary: Choosing the Right Method

When selecting a method, consider:

- **Do you have a control group?** DiD requires treatment and control groups; ITS does not
- **Is treatment based on a threshold?** RD is appropriate when treatment is assigned based on a cutoff
- **Do you have multiple control units to construct a synthetic counterfactual?** SC requires a pool of untreated units
- **What population does your effect apply to?**
  - DiD → treated units (ATT)
  - ITS/SC → the specific treated unit(s) at each time point
  - RD → units at the treatment threshold

For methods not covered here (IV, IPW, ANCOVA), see the respective notebook documentation and the {doc}`glossary` for estimand definitions.
