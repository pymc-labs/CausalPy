# Threat Catalog

Use this catalog to turn "could something else explain this?" into named, testable alternatives.

## Confounding

A confounder affects both treatment assignment and the outcome, opening a backdoor path between treatment and outcome.

Ask:

- What variables plausibly affect both the treatment and the outcome?
- Were they measured before treatment?
- Are they adjusted for in the formula, design, weights, or donor selection?
- Would omitting them bias the effect upward, downward, or ambiguously?

Do not confuse confounders with mediators or colliders. Mediators sit on the causal path and should usually not be controlled away when estimating the total effect. Colliders can create bias when conditioned on.

## Selection Bias

Treatment and comparison groups differ because of how units enter the data, survive to observation, or select into treatment.

Ask:

- Are treated and untreated units comparable at baseline?
- Is treatment adoption related to expected outcome changes?
- Are only surviving, active, or observed units included?
- Is overlap weak in propensity-score or weighting workflows?

Useful checks include `PreTreatmentPlaceboCheck`, `ConvexHullCheck`, balance diagnostics, and placebo analyses.

## Reverse Causation

The claimed outcome may affect treatment assignment, or treatment and outcome may reinforce each other.

Ask:

- Does treatment clearly happen before the outcome response?
- Could anticipation effects move the outcome before the treatment date?
- Could the outcome trigger the intervention?

Useful checks include pre-treatment placebo tests and placebo-in-time analyses.

## Measurement Error

The observed variable may not measure the intended treatment, outcome, running variable, or confounder.

Ask:

- Is measurement quality changing over time or across groups?
- Is measurement intensity related to treatment?
- Does the proxy outcome capture the theoretical construct?
- Would error attenuate the effect or create systematic bias?

Outcome falsification and sensitivity to alternative outcome definitions can help, but many measurement threats require domain evidence.

## Common Shocks

A simultaneous event may affect the treated unit and outcome at the same time as treatment.

Ask:

- What else changed around the intervention date?
- Did untreated units show similar changes?
- Are macro shocks, policy changes, seasonality, or campaign overlap plausible?

Useful checks include `PlaceboInSpace`, `PlaceboInTime`, and untreated-unit diagnostics.

## Specification Sensitivity

The result may depend on modeling choices rather than a stable causal signal.

Ask:

- Does the result change under reasonable formulas, bandwidths, priors, donor pools, or time windows?
- Are posterior diagnostics credible?
- Is the effect driven by a single donor, unit, or period?

Useful checks include `BandwidthSensitivity`, `PriorSensitivity`, `LeaveOneOut`, and robustness plots.

## External Validity

The effect may be real in the observed setting but not transport to another population, time, geography, or treatment scale.

Ask:

- Is the treatment context unusually specific?
- Are there likely diminishing returns or threshold effects?
- Would the mechanism operate in the target population?
- Is the intervention dose comparable?

External validity is usually a judgment call supported by subgroup analysis, replication, and domain knowledge.
