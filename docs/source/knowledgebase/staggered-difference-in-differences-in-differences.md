---
myst:
  html_meta:
    description: "Staggered difference-in-differences-in-differences is a contrast of subgroup staggered DiD estimates, not a second imputation estimator."
    "og:description": "Why a sugar tax's average effect hides the high-sugar versus low-sugar question, and how to contrast those staggered DiDs without subtracting their intervals."
---

# One average hides who the tax hit

A city-by-city sugar tax can look like a single number: sales fell by about 2 after adoption. That average is the attractive summary, and it answers the wrong question if the policy was aimed at high-sugar brands. Suppose high-sugar brands fell by 4 and low-sugar brands fell by 1. The incremental impact is 3, in sales units, not "somewhere between the two published intervals."

Subtracting those intervals feels careful. Each group already has a lower and upper bound, so the gap between the bounds looks like a range for the difference. It is not. Two groups can have the same standard error and still have a difference that moves from draw to draw. The worked numbers are below. The panel those numbers belong to is the {doc}`sugar-tax notebook <../notebooks/staggered-difference-in-differences-in-differences-sklearn>`.

## Two staggered DiDs, then a subtraction

Keep the brands in view. High-sugar and low-sugar are groups defined before the tax, not groups discovered from the outcome. The panel unit is the brand. A brand is treated when its city adopts, so the staggered calendar is the cities' rollout, but the estimator's unit is the brand, not the city. High-sugar and low-sugar brands are different units. Each group is fit with the imputation staggered difference-in-differences already in CausalPy, using the same adoption times, the same unit and time columns, and the same event window.

The extra question, at event time `e`, is how much larger the high-sugar average treatment effect on the treated was than the low-sugar one:

```text
ATT(high, e) - ATT(low, e)
```

A named reference group returns one contrast per other group, in the direction other-minus-reference. An explicit list of ordered pairs returns only those pairs, in the requested direction. The same subtraction applies to the overall post-treatment summary each staggered fit already reports: the unweighted mean of its identified post-treatment event times.

This is not a new imputation of the untreated outcome, and it is not a two-way fixed-effects triple interaction. The untreated outcome is still imputed inside each group by `StaggeredDifferenceInDifferences`. The wrapper only subtracts the public draws those fits return.

## Compare a time only when both groups identify it

An event time enters the contrast only when both groups identify it. If high-sugar brands have never-treated cities and low-sugar brands do not, a late event time can be identified for high sugar and unidentified for low sugar. That time gets no contrast row. It is not filled with zero, and it is not compared against a missing cell. Pre-tax event times that both groups identify can still appear; they are a check that the groups were not already diverging, not a second treatment effect.

## Subtract the draws, not the bounds

Take three paired illustrations, not the output of an OLS fit. OLS returns a point estimate and has no posterior draws. A Bayesian fit stores one draw of each group's event-time ATT in `att_event_time_draws`, and one draw of the overall post-treatment average in `att_overall_draws`. The contrast uses the same subtraction those illustrations show.

| Draw | High sugar | Low sugar | High minus low |
| --- | --- | --- | --- |
| 1 | 7 | 5 | 2 |
| 2 | 9 | 3 | 6 |
| 3 | 11 | 1 | 10 |

The differences are 2, 6, and 10. Their spread is the uncertainty of the contrast. The two sides have the same standard error, so subtracting standard errors, or subtracting the published interval endpoints, would report no uncertainty at all. `contrast_draws` does the within-draw subtraction. It does not subtract standard errors.

Independent draws are valid here because the brands do not appear in both groups. A brand that sat in both groups would share shocks across the two fits, and subtracting independent draws would ignore that dependence.

## What the 3 does not license

The high-minus-low gap says how much harder the tax hit high-sugar brands than low-sugar brands, in the outcome's units, under the staggered DiD assumptions inside each group. It does not say the tax was well designed, that a continuous sugar dose has a slope, or that a shock shared by both groups has been removed. A shared shock cancels in the difference only if, absent the tax, the treated-minus-untreated trend would have matched across the two sugar profiles. That is an assumption, not a result the subtraction creates.

The {doc}`sugar-tax notebook <../notebooks/staggered-difference-in-differences-in-differences-sklearn>` fits the two groups and prints the contrast in the simulated sales units.
