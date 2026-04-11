---
name: causal-detective
description: Structured falsification process for challenging causal claims. Guides users through questioning counterfactuals, hunting confounders, checking bias direction, and running computational falsification tests with CausalPy. Use when validating whether a causal effect is real or when a user says "is this effect real?" or "can I trust this result?"
---

# Causal Detective

A structured investigation process for challenging causal claims — inspired by the detective metaphor from *The Causal Mindset Handbook* (Gallea, 2026). Like a police detective eliminating alibis, we systematically rule out alternative explanations until only the causal claim remains (or doesn't).

This skill combines **critical thinking** (the 5-step Causal Mindset Framework) with **computational falsification** (CausalPy's 11 sensitivity and diagnostic checks).

## The Investigation Process

### Phase 1: Frame the Claim

Before touching any code, establish what's being claimed.

**Questions to answer:**
- What is the causal claim? (X causes Y)
- What is the treatment/intervention?
- What is the outcome being measured?
- What is the proxy counterfactual being used? (What's being compared?)
- How far is this proxy from the ideal "parallel world"?

**Output:** A clear statement: *"We claim that [treatment] caused [outcome], using [counterfactual] as our comparison."*

See [Counterfactual Analysis](reference/counterfactual_analysis.md) for guidance on evaluating proxy counterfactuals.

### Phase 2: Hunt for Alternative Explanations

This is the core detective work. For each question, generate concrete hypotheses.

**Question 1: "Is there something else?"** (Confounders)
- What third variables affect BOTH the treatment and outcome?
- Draw the causal graph — does any variable have arrows pointing to both X and Y?
- Is there selection bias? Are treated and untreated groups systematically different?
- Could there be measurement error that correlates with treatment?

**Question 2: "Could it be the reverse?"** (Reverse causation)
- Could Y be causing X instead of X causing Y?
- Could there be simultaneity — X and Y reinforcing each other?
- Does the timeline make sense? (cause must precede effect)

**Question 3: "What is the direction of bias?"**
- If confounders exist, do they inflate or shrink the estimated effect?
- Is the bias likely to make the effect look bigger (upward bias) or smaller (attenuation bias)?
- Could measurement error be systematically linked to the treatment?

See [Threat Catalog](reference/threat_catalog.md) for the full list of threats with examples.

### Phase 3: Design Falsification Tests

Translate each alternative explanation into a testable prediction, then test it with CausalPy. The logic: if the alternative explanation is true, we should observe a specific pattern in the data. If we don't observe it, we can rule it out.

| Alternative Explanation | Falsification Test | CausalPy Check |
|---|---|---|
| Effect existed before treatment | Test for pre-treatment effects | `PreTreatmentPlaceboCheck` |
| Model picks up noise, not signal | Shift treatment time to placebo periods | `PlaceboInTime` |
| Effect is an artifact of one donor/unit | Remove units one at a time | `LeaveOneOut`, `PlaceboInSpace` |
| Wrong outcome is being affected | Test on an outcome that should NOT change | `OutcomeFalsification` |
| Result sensitive to modeling choices | Vary bandwidth, priors, specifications | `BandwidthSensitivity`, `PriorSensitivity` |
| Effect doesn't persist | Check if effect fades or reverses | `PersistenceCheck` |
| Manipulation at RD threshold | Test for density discontinuity | `McCraryDensityTest` |
| Treated unit outside donor range | Check convex hull | `ConvexHullCheck` |

See [Falsification Tests](reference/falsification_tests.md) for implementation patterns.

### Phase 4: Evaluate the Evidence

After running tests, assess the overall strength of the causal claim.

**For each alternative explanation:**
- Was it testable? If not, discuss it qualitatively.
- Was the falsification test passed? (alternative ruled out)
- Was the falsification test failed? (alternative NOT ruled out — threat remains)
- How confident are we? (check `p_effect_outside_null`, effect sizes, credible intervals)

**Overall verdict categories:**
- **Strong evidence**: All major alternatives ruled out, effect robust to specification changes
- **Moderate evidence**: Some alternatives ruled out, effect stable but some threats remain
- **Weak evidence**: Key alternatives not tested or not ruled out
- **Falsified**: One or more tests indicate the effect is likely an artifact

### Phase 5: Assess Generalizability

Even if the effect is real in this context, can we extrapolate?

- **Across populations**: Would the effect hold for different groups? (age, geography, demographics)
- **Across time**: Was this a one-time context or a stable relationship?
- **Across scale**: If we increase/decrease the treatment, is the effect linear?
- **Across contexts**: Different market, different policy environment, different culture?

## Quick Reference: The 5 Questions

These questions can be asked of ANY causal claim, anywhere:

1. **What is the counterfactual?** — What is being compared, and how far is it from the ideal?
2. **Is there something else?** — What confounders, mediators, or colliders exist?
3. **Could it be the reverse?** — Is the direction of causation correct?
4. **What is the direction of bias?** — Is the effect over- or under-estimated?
5. **Can we extrapolate?** — Does this result generalize beyond this specific context?

## Agents

Three specialized agents support this workflow:

- **threat-assessor** — Identifies confounders, reverse causation, and other threats to a causal claim through structured questioning
- **falsification-runner** — Designs and executes computational falsification tests using CausalPy checks
- **evidence-synthesizer** — Weighs all evidence and produces a final verdict on the causal claim

## References

| Reference | Contents |
|---|---|
| [Counterfactual Analysis](reference/counterfactual_analysis.md) | Evaluating proxy counterfactuals and the "parallel world" key |
| [Threat Catalog](reference/threat_catalog.md) | All threats to causal claims with detection strategies |
| [Falsification Tests](reference/falsification_tests.md) | CausalPy checks mapped to alternative explanations |
