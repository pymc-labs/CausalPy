---
name: causal-detective
description: Challenge causal claims through structured threat assessment, counterfactual reasoning, and CausalPy falsification checks. Use when validating whether a causal effect is real or when the user asks "is this effect real?" or "can I trust this result?"
---

# Causal Detective

Use this skill to stress-test a causal claim before trusting or communicating it. The workflow combines qualitative causal reasoning with CausalPy sensitivity and diagnostic checks.

## Investigation Workflow

1. Frame the claim: state the treatment, outcome, estimand, fitted method, and proxy counterfactual.
2. Evaluate the counterfactual: ask how close the proxy is to the ideal parallel-world comparison.
3. Hunt for alternatives: identify concrete confounders, selection effects, reverse causation, measurement issues, common shocks, and external-validity limits.
4. Map threats to tests: choose CausalPy checks that would be expected to fail if each alternative explanation were true.
5. Interpret the evidence: separate threats ruled out by the data from threats that remain untested or unresolved.
6. Communicate the verdict: use cautious language that reflects the strength of the causal evidence rather than treating a fitted effect as proof.

## Core Questions

- What is the counterfactual, and how far is it from the ideal comparison?
- Is there something else that could affect both treatment assignment and the outcome?
- Could the outcome be influencing the treatment, or could the timing be ambiguous?
- If bias exists, would it inflate the effect, shrink it, or make the direction unclear?
- Can this result be generalized across populations, time periods, geographies, or treatment scales?

## CausalPy Checks

| Alternative explanation | Useful check |
|---|---|
| Effect existed before treatment | `cp.checks.PreTreatmentPlaceboCheck` |
| Model detects fake effects in untreated periods | `cp.checks.PlaceboInTime` |
| Result depends on one donor or observation | `cp.checks.LeaveOneOut` |
| Common shocks affect untreated units too | `cp.checks.PlaceboInSpace` |
| Effect appears on outcomes that should not move | `cp.checks.OutcomeFalsification` |
| RD/RK estimate depends on bandwidth | `cp.checks.BandwidthSensitivity` |
| Bayesian result depends on prior choices | `cp.checks.PriorSensitivity` |
| RD threshold may be manipulated | `cp.checks.McCraryDensityTest` |
| Synthetic control extrapolates beyond donors | `cp.checks.ConvexHullCheck` |
| Effect fades, reverses, or is window-specific | `cp.checks.PersistenceCheck` |

## Output Pattern

Return:

- Claim: one sentence.
- Counterfactual quality: good, moderate, or poor with reasoning.
- Threat inventory: named threats with severity, bias direction, and whether each is testable.
- Tests to run or tests run: CausalPy check names and the alternative each test targets.
- Verdict: strong, moderate, suggestive but inconclusive, weak, or likely non-causal.
- What would change the verdict: specific additional data, checks, or domain evidence.

## References

- [Counterfactual analysis](reference/counterfactual_analysis.md)
- [Threat catalog](reference/threat_catalog.md)
- [Falsification tests](reference/falsification_tests.md)
