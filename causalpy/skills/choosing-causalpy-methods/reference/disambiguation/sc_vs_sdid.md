# Synthetic Control vs Synthetic Difference-in-Differences

Use this card when the user has a wide panel with treated units and donor units.

## Deciding Question

Is the design just donor-unit weighting, or does it require both donor-unit weights and pre-period time weights?

## Choose `SyntheticControl`

- The data is a wide panel with unit columns and a time index.
- The counterfactual is a weighted combination of untreated donor units.
- The main design check is whether donor weights reproduce the treated unit's pre-period path.

## Choose `SyntheticDifferenceInDifferences`

- The same donor-panel setup applies, but the design explicitly uses both unit weights and pre-period time weights.
- The estimand is an SDiD-style ATT rather than a donor-weight-only synthetic control gap.
- The user accepts a Bayesian workflow in practice.

## Choose Neither

- Return Not implemented in CausalPy if the user needs augmented synthetic control, generalized synthetic control, or matrix completion.
- Return Not identifiable yet if donor units are not credible or pre-period fit cannot be assessed.
