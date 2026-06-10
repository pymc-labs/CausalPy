# Counterfactual Analysis

## The Parallel-World Benchmark

The ideal causal comparison is the same unit in two parallel worlds, with only the treatment changed. Since that is impossible, every CausalPy analysis uses an observed proxy counterfactual. The first review step is to name that proxy and judge how credible it is.

## Common Proxy Counterfactuals

| Method | Proxy counterfactual | Main identifying burden |
|---|---|---|
| Difference-in-differences | Untreated group's trend | Parallel trends |
| Interrupted time series | Pre-treatment trend projected forward | Trend continuity and no concurrent shock |
| Synthetic control | Weighted combination of donor units | Donor relevance and convex-hull support |
| Regression discontinuity | Units just below or above the threshold | Continuity near the cutoff |
| Inverse propensity weighting | Reweighted comparison group | No unmeasured confounding and overlap |
| Instrumental variables | Variation induced by the instrument | Instrument relevance, exclusion, and monotonicity |

## Review Questions

- What is being used as the stand-in for the missing counterfactual?
- Are treated and comparison units systematically different before treatment?
- Are before/after comparisons vulnerable to seasonality, trend breaks, regression to the mean, or concurrent events?
- Does the comparison happen at the right level: person, unit, geography, market, time period, or treatment dose?
- Does the method extrapolate beyond observed support?

## Red Flags

- Before/after comparison without a control or clear trend argument.
- Convenient comparison group chosen because it is available rather than credible.
- Donor pool that cannot reproduce the treated unit before treatment.
- Threshold design where units can manipulate which side of the cutoff they fall on.
- Observational treatment with weak overlap or obvious unmeasured confounders.
