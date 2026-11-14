# Skill Triggers

Use this reference for discovery when the user's question does not mention "causal" explicitly but still needs a quasi-experimental design before analysis code.

## Plain-Language Impact Questions

- Did the campaign, policy, product change, or rollout work?
- What was the effect, lift, or impact on the outcome?
- Did sales, engagement, or conversions change after the intervention?

## Design Keywords

- Before-after or pre/post comparisons
- Treated vs control groups
- Counterfactuals, ATE, ATT
- Interrupted time series, comparative interrupted time series (CITS), comparison or control series
- Difference-in-differences (DiD), staggered adoption, event study, panel data
- Synthetic control, synthetic difference-in-differences (SDiD)
- Regression discontinuity, regression kink
- Instrumental variable (IV), inverse propensity weighting (IPW)
- Pre/post nonequivalent groups (PrePostNEGD)

## When Not To Use This Skill

- The method is already chosen — use `running-causalpy-experiments` instead.
- The user wants to stress-test an existing claim — use `causal-detective`.
- The user only needs bundled example data — use `example-datasets`.
