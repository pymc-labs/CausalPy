# Interactive fixed effects

`InteractiveFixedEffectsSyntheticControl` holds a balanced `WidePanel`. It is not a `SyntheticControl` subclass. v1 is outcome-only. This module does not sample.

## Frozen call surface

`compose_result` builds an `InteractiveFixedEffectsResult`, a subclass of `CausalResult`. New diagnostics are on that subclass, not new fields of `CausalResult`.

- Untreated counterfactual at each time: additive unit intercept + additive time intercept + loading-factor product.
- Impact: observed treated outcome minus that counterfactual, via `WidePanel.impacts`. Changing treated post-intervention outcomes updates impacts only.
- Pre-period prediction error: `score`, Bayesian \(R^2\) against pre-period treated outcomes only.
- Shrinkage state: `shrinkage_state`, posterior draws of one scale per factor up to `max_factors`. Not a selected integer rank and not an effective-rank count.

There is no `n_factors` argument and no Xu cross-validation. `max_factors` is an upper bound, not a selected rank. Posterior intervals on the composed counterfactual are the intervals of those draws. This module does not condition them on a selected rank.

## Not frozen

The prior family is not frozen. Rotation or scale constraints beyond "unused factor scales can go to zero" are not frozen. This page does not name a prior family. Do not treat a missing sampler as a selected-rank estimator.

## Out of bounds

Missing values in the used columns, staggered treatment times, covariates, and a classical numerical-parity target are out of bounds for v1. The panel is the balanced wide panel `WidePanel` already stores: one shared `treatment_time`, one or more treated units, and control units as donors.
