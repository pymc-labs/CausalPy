# Known-truth checks for Bayesian synthetic control

This note is the comparison #1240 may cite. It is not a gallery notebook and it is not linked from the gallery. A coverage claim on these panels is a claim about noiseless data: each panel's `data` equals planted `untreated + effect` in every cell. The five panels are the existing fixtures in `causalpy/tests/known_truth_panels.py`. This note does not add observation noise, a second panel generator, or a Monte Carlo rate over datasets.

Each recorded 0/1 is whether the planted post-period value lies inside the highest density interval of the posterior of the conditional mean. The interval probability is the library default `causalpy.constants.HDI_PROB` (0.94). The interval is not an equal-tailed interval and not a posterior-predictive `y_hat` interval. A 0/1 is not a frequentist coverage rate and not a simulation-based calibration result. Do not summarize these rows as a percentage.

One treated unit, post period only (`index >= treatment_time`). Refits that produced the table used `draws=100`, `tune=100`, `chains=2`, `cores=1`, `random_seed=1241`, `target_accept=0.95`, and `progressbar=False`. `InteractiveFixedEffectsSyntheticControl` uses its default `max_factors` upper bound. That bound is not a selected rank.

## Interval construction

- `SyntheticControl` and `InteractiveFixedEffectsSyntheticControl`: highest density interval of posterior `predictions_post` against planted `untreated`, and of posterior `impact_post` against planted `effect`.
- `AugmentedSyntheticControl`: highest density interval of the joint-model posterior deterministic `counterfactual` against planted `untreated`, and of `treatment_effect` against planted `effect`. That is the full joint model, not a `WeightedSumFitter` adjustment, and not `CausalResult.impact_post`.

## Inside or outside

`inside` is 1 when the planted value is inside that interval at that post time, and 0 when it is outside.

| panel | model | posterior | time | inside |
| --- | --- | --- | --- | --- |
| exact_convex_hull | SyntheticControl | predictions_post | 4 | 1 |
| exact_convex_hull | SyntheticControl | predictions_post | 5 | 1 |
| exact_convex_hull | SyntheticControl | predictions_post | 6 | 1 |
| exact_convex_hull | SyntheticControl | predictions_post | 7 | 1 |
| exact_convex_hull | SyntheticControl | impact_post | 4 | 1 |
| exact_convex_hull | SyntheticControl | impact_post | 5 | 1 |
| exact_convex_hull | SyntheticControl | impact_post | 6 | 1 |
| exact_convex_hull | SyntheticControl | impact_post | 7 | 1 |
| exact_convex_hull | InteractiveFixedEffectsSyntheticControl | predictions_post | 4 | 1 |
| exact_convex_hull | InteractiveFixedEffectsSyntheticControl | predictions_post | 5 | 1 |
| exact_convex_hull | InteractiveFixedEffectsSyntheticControl | predictions_post | 6 | 1 |
| exact_convex_hull | InteractiveFixedEffectsSyntheticControl | predictions_post | 7 | 1 |
| exact_convex_hull | InteractiveFixedEffectsSyntheticControl | impact_post | 4 | 1 |
| exact_convex_hull | InteractiveFixedEffectsSyntheticControl | impact_post | 5 | 1 |
| exact_convex_hull | InteractiveFixedEffectsSyntheticControl | impact_post | 6 | 1 |
| exact_convex_hull | InteractiveFixedEffectsSyntheticControl | impact_post | 7 | 1 |
| exact_convex_hull | AugmentedSyntheticControl | counterfactual | 4 | 1 |
| exact_convex_hull | AugmentedSyntheticControl | counterfactual | 5 | 1 |
| exact_convex_hull | AugmentedSyntheticControl | counterfactual | 6 | 1 |
| exact_convex_hull | AugmentedSyntheticControl | counterfactual | 7 | 1 |
| exact_convex_hull | AugmentedSyntheticControl | treatment_effect | 4 | 1 |
| exact_convex_hull | AugmentedSyntheticControl | treatment_effect | 5 | 1 |
| exact_convex_hull | AugmentedSyntheticControl | treatment_effect | 6 | 1 |
| exact_convex_hull | AugmentedSyntheticControl | treatment_effect | 7 | 1 |
| systematic_inexact | SyntheticControl | predictions_post | 4 | 0 |
| systematic_inexact | SyntheticControl | predictions_post | 5 | 0 |
| systematic_inexact | SyntheticControl | predictions_post | 6 | 0 |
| systematic_inexact | SyntheticControl | predictions_post | 7 | 0 |
| systematic_inexact | SyntheticControl | impact_post | 4 | 0 |
| systematic_inexact | SyntheticControl | impact_post | 5 | 0 |
| systematic_inexact | SyntheticControl | impact_post | 6 | 0 |
| systematic_inexact | SyntheticControl | impact_post | 7 | 0 |
| systematic_inexact | InteractiveFixedEffectsSyntheticControl | predictions_post | 4 | 1 |
| systematic_inexact | InteractiveFixedEffectsSyntheticControl | predictions_post | 5 | 1 |
| systematic_inexact | InteractiveFixedEffectsSyntheticControl | predictions_post | 6 | 1 |
| systematic_inexact | InteractiveFixedEffectsSyntheticControl | predictions_post | 7 | 1 |
| systematic_inexact | InteractiveFixedEffectsSyntheticControl | impact_post | 4 | 1 |
| systematic_inexact | InteractiveFixedEffectsSyntheticControl | impact_post | 5 | 1 |
| systematic_inexact | InteractiveFixedEffectsSyntheticControl | impact_post | 6 | 1 |
| systematic_inexact | InteractiveFixedEffectsSyntheticControl | impact_post | 7 | 1 |
| systematic_inexact | AugmentedSyntheticControl | counterfactual | 4 | 0 |
| systematic_inexact | AugmentedSyntheticControl | counterfactual | 5 | 1 |
| systematic_inexact | AugmentedSyntheticControl | counterfactual | 6 | 0 |
| systematic_inexact | AugmentedSyntheticControl | counterfactual | 7 | 1 |
| systematic_inexact | AugmentedSyntheticControl | treatment_effect | 4 | 0 |
| systematic_inexact | AugmentedSyntheticControl | treatment_effect | 5 | 1 |
| systematic_inexact | AugmentedSyntheticControl | treatment_effect | 6 | 0 |
| systematic_inexact | AugmentedSyntheticControl | treatment_effect | 7 | 1 |
| latent_factor | SyntheticControl | predictions_post | 4 | 0 |
| latent_factor | SyntheticControl | predictions_post | 5 | 0 |
| latent_factor | SyntheticControl | predictions_post | 6 | 0 |
| latent_factor | SyntheticControl | predictions_post | 7 | 0 |
| latent_factor | SyntheticControl | impact_post | 4 | 0 |
| latent_factor | SyntheticControl | impact_post | 5 | 0 |
| latent_factor | SyntheticControl | impact_post | 6 | 0 |
| latent_factor | SyntheticControl | impact_post | 7 | 0 |
| latent_factor | InteractiveFixedEffectsSyntheticControl | predictions_post | 4 | 1 |
| latent_factor | InteractiveFixedEffectsSyntheticControl | predictions_post | 5 | 1 |
| latent_factor | InteractiveFixedEffectsSyntheticControl | predictions_post | 6 | 1 |
| latent_factor | InteractiveFixedEffectsSyntheticControl | predictions_post | 7 | 1 |
| latent_factor | InteractiveFixedEffectsSyntheticControl | impact_post | 4 | 1 |
| latent_factor | InteractiveFixedEffectsSyntheticControl | impact_post | 5 | 1 |
| latent_factor | InteractiveFixedEffectsSyntheticControl | impact_post | 6 | 1 |
| latent_factor | InteractiveFixedEffectsSyntheticControl | impact_post | 7 | 1 |
| latent_factor | AugmentedSyntheticControl | counterfactual | 4 | 1 |
| latent_factor | AugmentedSyntheticControl | counterfactual | 5 | 1 |
| latent_factor | AugmentedSyntheticControl | counterfactual | 6 | 1 |
| latent_factor | AugmentedSyntheticControl | counterfactual | 7 | 1 |
| latent_factor | AugmentedSyntheticControl | treatment_effect | 4 | 1 |
| latent_factor | AugmentedSyntheticControl | treatment_effect | 5 | 1 |
| latent_factor | AugmentedSyntheticControl | treatment_effect | 6 | 1 |
| latent_factor | AugmentedSyntheticControl | treatment_effect | 7 | 1 |
| donor_shock | SyntheticControl | predictions_post | 4 | 0 |
| donor_shock | SyntheticControl | predictions_post | 5 | 0 |
| donor_shock | SyntheticControl | predictions_post | 6 | 0 |
| donor_shock | SyntheticControl | predictions_post | 7 | 0 |
| donor_shock | SyntheticControl | impact_post | 4 | 0 |
| donor_shock | SyntheticControl | impact_post | 5 | 0 |
| donor_shock | SyntheticControl | impact_post | 6 | 0 |
| donor_shock | SyntheticControl | impact_post | 7 | 0 |
| donor_shock | InteractiveFixedEffectsSyntheticControl | predictions_post | 4 | 1 |
| donor_shock | InteractiveFixedEffectsSyntheticControl | predictions_post | 5 | 1 |
| donor_shock | InteractiveFixedEffectsSyntheticControl | predictions_post | 6 | 1 |
| donor_shock | InteractiveFixedEffectsSyntheticControl | predictions_post | 7 | 1 |
| donor_shock | InteractiveFixedEffectsSyntheticControl | impact_post | 4 | 1 |
| donor_shock | InteractiveFixedEffectsSyntheticControl | impact_post | 5 | 1 |
| donor_shock | InteractiveFixedEffectsSyntheticControl | impact_post | 6 | 1 |
| donor_shock | InteractiveFixedEffectsSyntheticControl | impact_post | 7 | 1 |
| donor_shock | AugmentedSyntheticControl | counterfactual | 4 | 0 |
| donor_shock | AugmentedSyntheticControl | counterfactual | 5 | 0 |
| donor_shock | AugmentedSyntheticControl | counterfactual | 6 | 0 |
| donor_shock | AugmentedSyntheticControl | counterfactual | 7 | 0 |
| donor_shock | AugmentedSyntheticControl | treatment_effect | 4 | 0 |
| donor_shock | AugmentedSyntheticControl | treatment_effect | 5 | 0 |
| donor_shock | AugmentedSyntheticControl | treatment_effect | 6 | 0 |
| donor_shock | AugmentedSyntheticControl | treatment_effect | 7 | 0 |
| placebo | SyntheticControl | predictions_post | 4 | 1 |
| placebo | SyntheticControl | predictions_post | 5 | 1 |
| placebo | SyntheticControl | predictions_post | 6 | 1 |
| placebo | SyntheticControl | predictions_post | 7 | 1 |
| placebo | SyntheticControl | impact_post | 4 | 1 |
| placebo | SyntheticControl | impact_post | 5 | 1 |
| placebo | SyntheticControl | impact_post | 6 | 1 |
| placebo | SyntheticControl | impact_post | 7 | 1 |
| placebo | InteractiveFixedEffectsSyntheticControl | predictions_post | 4 | 1 |
| placebo | InteractiveFixedEffectsSyntheticControl | predictions_post | 5 | 1 |
| placebo | InteractiveFixedEffectsSyntheticControl | predictions_post | 6 | 1 |
| placebo | InteractiveFixedEffectsSyntheticControl | predictions_post | 7 | 1 |
| placebo | InteractiveFixedEffectsSyntheticControl | impact_post | 4 | 1 |
| placebo | InteractiveFixedEffectsSyntheticControl | impact_post | 5 | 1 |
| placebo | InteractiveFixedEffectsSyntheticControl | impact_post | 6 | 1 |
| placebo | InteractiveFixedEffectsSyntheticControl | impact_post | 7 | 1 |
| placebo | AugmentedSyntheticControl | counterfactual | 4 | 1 |
| placebo | AugmentedSyntheticControl | counterfactual | 5 | 1 |
| placebo | AugmentedSyntheticControl | counterfactual | 6 | 1 |
| placebo | AugmentedSyntheticControl | counterfactual | 7 | 1 |
| placebo | AugmentedSyntheticControl | treatment_effect | 4 | 1 |
| placebo | AugmentedSyntheticControl | treatment_effect | 5 | 1 |
| placebo | AugmentedSyntheticControl | treatment_effect | 6 | 1 |
| placebo | AugmentedSyntheticControl | treatment_effect | 7 | 1 |

## What the intervals do and do not establish

On `latent_factor`, `InteractiveFixedEffectsSyntheticControl` improves on `SyntheticControl`. The untreated panel is an exact rank-2 factor product and is not a convex combination of the donors. At every post time, the interactive fixed-effects intervals for `predictions_post` and `impact_post` contain the planted untreated path and the planted effect. The `SyntheticControl` intervals for those same posterior series exclude both. That is a difference in which conditional-mean interval contains the planted series, not a ranking by training error.

On `donor_shock`, a better pre-period fit does not by itself establish causal validity. The treated pre-period path is an exact convex combination of the donors, so `SyntheticControl` can match that pre-period path, and the planted treated effect is zero. One donor's post-period path is shocked off the path that generated the treated unit's untreated trajectory. That shock is not part of the treated untreated trajectory. At every post time the `SyntheticControl` intervals for `predictions_post` and `impact_post` exclude the planted untreated path and the planted effect. The joint `AugmentedSyntheticControl` intervals for `counterfactual` and `treatment_effect` also exclude those planted values at every post time: the prognostic covariates are pre-period donor vectors, so the post-period donor shock is not repaired by a closer pre-period weight fit. Containing, or missing, the planted series on this panel is not evidence that pre-period fit is the causal claim.
