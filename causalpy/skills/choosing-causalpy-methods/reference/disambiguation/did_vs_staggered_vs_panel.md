# DiD vs Staggered DiD vs Panel Regression

Use this card when the user has panel or repeated group-time data with treatment timing.

## Deciding Question

Is treatment timing common, staggered and absorbing, or not a quasi-experimental timing design at all?

## Choose `DifferenceInDifferences`

- There is one treated group and one control group observed before and after a common intervention timing.
- The formula can express a single treated-group by post-period interaction.
- The identifying assumption is parallel trends for treated and control groups.

## Choose `StaggeredDifferenceInDifferences`

- Units adopt treatment at different times.
- Treatment is absorbing: once treated, a unit stays treated.
- The estimand is a cohort, event-time, or overall ATT path under no anticipation and parallel trends.
- Each calendar period in the estimand needs at least one untreated unit. Without never-treated or not-yet-treated units, time fixed effects are not identified and CausalPy warns or fails.

## Choose `PanelRegression`

- The target is coefficient-level fixed-effects regression, not a named DiD design.
- The user wants unit and/or time fixed effects for adjusted association or for a formula they will defend separately.
- The route should warn that `PanelRegression` is not a causal design by itself and does not implement `effect_summary()`.

## Choose Neither

- Return Not implemented in CausalPy for non-absorbing staggered treatment, repeated treatment on/off, or a request for a specific alternative staggered estimator such as Callaway-Sant'Anna, Sun-Abraham, or Gardner.
