---
myst:
  html_meta:
    description: "How to compare Bayesian synthetic control, augmented synthetic control, and interactive fixed effects without ranking them by training fit."
"og:description": "How to compare Bayesian synthetic control, augmented synthetic control, and interactive fixed effects without ranking them by training fit."
---

# Comparing Bayesian synthetic-control models

This note is the cross-model audit. It distinguishes a fit diagnostic from a causal identification check, and it says what a posterior interval is allowed to be called. Method specifications stay on [Augmented synthetic control](augmented-synthetic-control.md) and [Interactive fixed-effects synthetic control](interactive-fixed-effects.md). This page does not restate those likelihoods.

The worked comparison is [the how-to notebook](../notebooks/comparing-bayesian-synthetic-control-pymc.ipynb). Coverage evidence is the [known-truth report](bayesian_synthetic_control_known_truth_report.md). This note does not reimplement that study.

## Fit diagnostics are not identification checks

A fit diagnostic describes how the recorded model uses pre-intervention and untreated information. A causal identification check asks whether that fit is the untreated path the estimand needs. Lower pre-period error does not answer the second question. Do not rank `SyntheticControl`, `AugmentedSyntheticControl`, and `InteractiveFixedEffectsSyntheticControl` by training fit, and do not treat the lowest training error as evidence that the estimator is causally valid.

Comparisons and tuning use only pre-intervention treated outcomes and untreated (control) outcomes. Treated outcomes at or after `treatment_time` are impact and display. They are not a tuning target.

## What each model actually exposes

The audit uses only quantities the owning estimators already freeze. It does not add a second diagnostic implementation.

| Model | Frozen quantity | What it means here | What it is not |
| --- | --- | --- | --- |
| `SyntheticControl` with `WeightedSumFitter` | `predictions_pre`, `predictions_post`, `impact_pre`, `impact_post` | Posterior of the weighted-sum model that experiment fitted | A prognostic correction, and not an adjustment applied after `WeightedSumFitter` |
| `AugmentedSyntheticControl` | `base_weights`, `prognostic_coefficients`, `counterfactual`, `treatment_effect`, `tau` | Joint-model posterior names | A second estimator |
| `InteractiveFixedEffectsSyntheticControl` | `shrinkage_state`, `score` | Posterior draws of one positive scale per factor, up to `max_factors`; pre-period prediction error on pre-period treated outcomes only | A selected integer rank, and not an effective-rank count |

`shrinkage_state` is the factor information this comparison is allowed to use. `max_factors` is an upper bound, not a rank chosen on the post-period. `score` is the pre-period prediction error.

This comparison does not depend on baseline weights, effective weights, negative-weight magnitude, or a time-varying correction. Those names are not part of the augmented result this note audits. Their absence is not a defect to send back to the augmented estimator, and it is not a reason to edit that estimator.

## Bayesian interval labeling

Label an interval as Bayesian only when the full fitted model produced it.

- A `SyntheticControl` band is a Bayesian interval for `WeightedSumFitter` when it is the posterior of `predictions_*` or `impact_*` from that fit. It is not a Bayesian interval for augmented synthetic control.
- An `AugmentedSyntheticControl` band is a Bayesian interval for the joint model when it is the posterior of `counterfactual` or `treatment_effect` on `idata`. A deterministic adjustment of draws from `WeightedSumFitter` is not that posterior. Do not label the adjustment as a full augmented posterior.
- An `InteractiveFixedEffectsSyntheticControl` band is a Bayesian interval for that model when it is the posterior of `predictions_*` or `impact_*` after marginalizing the shrinkage prior. It is not a confidence interval for a cross-validated rank.

None of these intervals claims nominal frequentist coverage under unrestricted misspecification.

## What the known-truth report shows

The [known-truth report](bayesian_synthetic_control_known_truth_report.md) records, for each post time, whether the planted value lies inside the highest density interval of the named posterior. Each 0/1 is that inside-or-outside flag. It is not a frequentist coverage rate and not a simulation-based calibration result. Do not summarize those rows as a percentage, and do not promise nominal coverage from them.

Two data-generating processes in that report have different pre-fit behavior.

On `latent_factor` the untreated panel is an exact rank-2 factor product and is not a convex combination of the donors. At every post time the interactive fixed-effects intervals for `predictions_post` and `impact_post` contain the planted untreated path and the planted effect. The `SyntheticControl` intervals for those same series exclude both. That is a difference in which conditional-mean interval contains the planted series. It is not a ranking by training error. The augmented joint-model intervals for `counterfactual` and `treatment_effect` also contain the planted series at every post time on that panel. That agreement is not identification.

On `donor_shock` the treated pre-period path is an exact convex combination of the donors, so `SyntheticControl` can match that pre-period path, and the planted treated effect is zero. One donor's post-period path is shocked off the path that generated the treated unit's untreated trajectory. At every post time the `SyntheticControl` intervals for `predictions_post` and `impact_post` exclude the planted untreated path and the planted effect. The joint `AugmentedSyntheticControl` intervals for `counterfactual` and `treatment_effect` also exclude those planted values at every post time. A closer pre-period weight fit does not repair a post-period donor shock. That is the failure mode: improved pre-period fit does not by itself establish causal validity.

Interactive fixed-effects intervals contain the planted series at every post time on `donor_shock` as well. Containing the planted series on one noiseless panel is still not a coverage claim, and it is not a reason to prefer that estimator because its pre-period score is smaller.
