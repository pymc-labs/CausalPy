---
name: choosing-causalpy-methods
description: Selects the appropriate CausalPy quasi-experimental method for causal or impact questions. Use when a user asks whether something worked, changed an outcome, caused an effect, or had impact/lift, even without explicit causal terms; triggers include campaigns, policies, product changes, rollouts, interventions, treatments, before-after comparisons, treated vs control groups, counterfactuals, ATE/ATT, DiD, interrupted time series, synthetic control, regression discontinuity, IV, IPW, panel/event-study questions, and any request that needs a causal design before analysis code.
---

# Choosing CausalPy Methods

Use this skill to translate a user's causal or impact question into a CausalPy experiment choice, including plain-English questions like "did the campaign work?", "what was the effect of the rollout?", or "did the policy change sales?". This is the design-intake skill, not the implementation skill. Optimize for agent use: follow the ordered routing steps, prefer explicit uncertainty over force-fitting, and do not write analysis code until the method route is matched or the user has answered the key ambiguity. Once the method is chosen, hand off to `running-causalpy-experiments` for constructor details, model configuration, priors, summaries, plots, and interpretation.

## Required Intake

Before naming a method, identify these facts. If the request is missing several, ask for the single most decision-relevant missing fact.

1. Estimand: ATE, ATT, local threshold effect, event-study path, cumulative post-period impact, baseline-adjusted group contrast, or coefficient-level association.
2. Assignment mechanism: known intervention time, common pre/post group treatment, staggered adoption, cutoff, kink, instrument, observed treatment with measured confounders, or no credible assignment story.
3. Data topology: single outcome series, wide unit-by-time panel, long unit-time panel, cross-section, or single pre/post observations by unit or group.
4. Controls: none, donor units, treated/control groups, never-treated cohorts, measured covariates, near-cutoff observations, or instruments.
5. Identification support: pre-period history, donor support, overlap/positivity, no anticipation, absorbing treatment, no manipulation at cutoff, instrument validity, or baseline adjustment.
6. Backend/reporting constraints: whether the user needs OLS/sklearn, Bayesian uncertainty, `effect_summary()`, or a unified `plot()`.

## Routing Workflow

Use the canonical routing algorithm in [Decision tree](reference/decision_tree.md). It is deliberately written as text/pseudocode, not a visual decision tree, so agents can follow it linearly. Do not skip from a keyword such as "time series" or "panel" directly to a class; route through assignment mechanism, data topology, controls, and disqualifiers.

When a route is close but not settled, use the disambiguation cards:

- [ITS vs Piecewise ITS](reference/disambiguation/its_vs_piecewise_its.md)
- [ITS vs Synthetic Control vs DiD](reference/disambiguation/its_vs_sc_vs_did.md)
- [DiD vs Staggered DiD vs Panel Regression](reference/disambiguation/did_vs_staggered_vs_panel.md)
- [Synthetic Control vs SDiD](reference/disambiguation/sc_vs_sdid.md)
- [IPW vs IV vs Panel Regression](reference/disambiguation/ipw_vs_iv_vs_panel.md)
- [PrePostNEGD vs DiD](reference/disambiguation/prepostnegd_vs_did.md)

## Output Contracts

Return exactly one of these outcomes.

### Matched

- Recommended method: name one primary CausalPy experiment class.
- Why it fits: tie the recommendation to estimand, assignment mechanism, data topology, and controls.
- Disqualifiers checked: name the main alternatives rejected and why.
- Required columns/data layout: list the minimal structure needed.
- Key assumptions: state what must be credible for causal interpretation.
- Main risks: name likely failure modes and sensitivity checks.
- Next step: route to `running-causalpy-experiments` and the relevant method reference.

### Ambiguous

- Candidate methods: name the top two plausible CausalPy classes.
- What separates them: state the concrete distinction, such as forecast counterfactual vs segmented level/slope model.
- Deciding question: ask one question that will resolve the route.
- Next step: do not write analysis code until the user answers.

### Not Identifiable Yet

- Status: the method may exist in CausalPy, but the available data or assumptions are insufficient.
- Missing requirement: name the specific missing design fact, such as no pre-period, no donor support, no overlap check, or no credible instrument.
- Deciding question: ask for the missing fact or suggest what evidence would be needed.
- Next step: do not force a method recommendation.

### Not Implemented In CausalPy

- Status: state "CausalPy has not implemented the right method."
- Ideal method category: name the method family the user likely needs.
- Why no CausalPy experiment fits: explain the mismatch with assignment, estimand, treatment type, data topology, or reporting needs.
- Closest partial fit: mention a CausalPy class only if it is genuinely useful, and state its limitations.
- What would unlock CausalPy: describe the data or assumption change that would make an implemented experiment appropriate.

## References

- [Decision tree](reference/decision_tree.md)
- [Experiment decision guide](reference/experiment_decision_guide.md)
- [Method capability matrix](reference/method_capability_matrix.md)
- [When CausalPy has not implemented the right method](reference/not_in_causalpy.md)
