# Counterfactual Analysis

## The "Parallel World" Key

The only way to perfectly measure a causal effect would be to have two parallel worlds, changing just one thing. Since we can't, we must find proxy counterfactuals — observable situations that approximate the ideal.

## Evaluating a Proxy Counterfactual

For any causal claim, ask:

**What is the proxy counterfactual?**
The observed situation used as a stand-in for the true counterfactual. Common choices:

| Method | Proxy Counterfactual | Ideal Approximation |
|---|---|---|
| DiD | Untreated group's trend | Assumes parallel trends |
| ITS | Pre-treatment trend projected forward | Assumes trend continuity |
| SC | Weighted combination of donor units | Assumes convex hull, relevance |
| RD | Units just below the threshold | Assumes continuity at threshold |
| IPW | Reweighted comparison group | Assumes no unmeasured confounders |

**How far is it from the ideal?**
For each proxy, identify the gaps:

1. **Selection differences**: Are treated and untreated groups systematically different? (e.g., loyalty card members vs. non-members are different types of shoppers)
2. **Temporal differences**: Are you comparing different time periods? (e.g., before vs. after, but seasonality exists)
3. **Contextual differences**: Are you comparing different contexts? (e.g., urban vs. rural, but lifestyle differs)
4. **Scale differences**: Is the treatment dose different? (e.g., extrapolating from small to large dose)

## Red Flags in Counterfactual Choice

- **Before-after without control**: Comparing the same unit before and after treatment — cannot distinguish treatment from time trends, regression to the mean, or concurrent events
- **Convenience comparison**: Using whatever data is available rather than selecting the most comparable group
- **Survivorship bias**: Only observing units that survived the treatment period
- **Wrong level of comparison**: e.g., e-scooter company comparing to cars, when actual substitution is for walking/public transport
