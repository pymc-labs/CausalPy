# IPW vs IV vs Panel Regression

Use this card when treatment is observational or potentially confounded.

## Deciding Question

What is the identification strategy: measured-confounder adjustment with overlap, a valid instrument, or fixed-effects regression?

## Choose `InversePropensityWeighting`

- Treatment is binary.
- Confounders needed for conditional exchangeability are measured.
- Positivity and overlap are plausible.
- The target is a weighted ATE-style contrast, and the user does not need `effect_summary()` or unified `plot()`.

## Choose `InstrumentalVariable`

- Treatment is endogenous or unmeasured confounding is central.
- There is a credible instrument with relevance, exclusion, and exogeneity arguments.
- The user accepts Bayesian IV with custom inspection rather than standard `summary()`, `effect_summary()`, and `plot()` outputs.

## Choose `PanelRegression`

- The data is a panel and the target is fixed-effects coefficient estimation.
- Unit and/or time effects are part of the adjustment strategy.
- The answer should be framed as coefficient-level regression unless the treatment assignment story supports causal interpretation.

## Choose Neither

- Return Not identifiable yet if confounders are unmeasured and no credible instrument exists.
- Return Not implemented in CausalPy for matching, multi-arm treatment, continuous dose-response treatment, causal forests, or mediation.
