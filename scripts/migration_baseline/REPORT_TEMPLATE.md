# PyMC 5 → PyMC 6 Migration Baseline Comparison

This is the permanent attachment structure for #1048. It is not evidence by itself: the coordinator must generate a populated Markdown report with `scripts/migration_baseline/harness.py compare` and attach that generated report together with its four fresh schema-v2 input artifacts.

**Result:** `PASS` or `FAIL` from the generated comparison. A schema-v1 result is historical only and does not satisfy this template.

## Scope and attribution

- Reference checkout: `79c0a87072fd4653bfaed1eb085f965594c7f03a` in the PyMC 5 environment.
- Candidate checkout: `18a524a1a8512aaa21c46e0ccddbc54501c9eb1a` in the PyMC 6 migration environment.
- Both sampled checkouts and the executing harness checkout had an empty `git status --porcelain` before CausalPy import. The committed harness implementation checkout is not a sampled candidate checkout.
- Attribution statement: this evidence covers only the fixed migration delta. Any behavior on a later source commit is a separate feature comparison and must not be described as migration drift.

## Comparator, artifact, and batch identity

Record all of the following from the generated report, not from manually transcribed values:

- Comparator path, SHA-256, Git commit, and Git blob SHA-256.
- One canonical batch UUID shared by the four role-bound captures.
- The four distinct capture UUIDs and roles: `reference_first`, `reference_second`, `candidate_first`, and `candidate_second`.
- Each input artifact path and the SHA-256 calculated from the exact byte buffer read by the comparator.
- The fixed scenario-manifest identifier and fixture SHA-256 for every scenario.

The attachment must name a newly created schema-v2 evidence directory and immutable result location. Do not overwrite, relabel, or mix in the historical schema-v1 directory or gist.

## Imported runtime and deterministic protocol

For every capture, record the exact CausalPy editable-install target, Python executable and prefix, imported CausalPy/Python/PyMC/PyTensor/ArviZ versions, imported module paths, platform, and machine. The report must show:

- same-stack repeat captures used identical runtime identity, posterior summaries, and sampling-quality evidence as well as exact raw-draw digests;
- PyMC 5 and PyMC 6 captures used distinct prefixes; matching platform, machine, Python version/implementation, and NumPy/pandas/xarray versions; and the expected PyMC/PyTensor/ArviZ major versions;
- the imported editable CausalPy target matched the selected clean checkout;
- all four artifacts bound the currently executing comparator SHA-256, blob SHA-256, and Git commit.

Record that each capture used PyMC NUTS with four serialized chains (`cores=1`), 1,000 tuning iterations, 1,000 retained draws, master seed `1048`, target acceptance `0.95`, and a maximum tree depth of `12`. State that `cores=1` was mandatory on local macOS and intentionally retained on all platforms to serialize the chain schedule.

Record the explicit tail-ESS probability pair `(0.05, 0.95)`. It is part of the shared protocol rather than an ArviZ-version default.

## Evidence validity gate

Report that both independent captures within each stack produced identical posterior summaries and sampling-quality evidence plus exact same-stack raw-draw digests for every captured metric. The comparator must never compare raw draw digests across stacks.

Report the per-case divergent-draw count, tree-depth result when exposed, finite-value result, maximum rank R-hat, minimum bulk ESS, and minimum tail ESS. A failure in any of these checks invalidates the numerical comparison rather than being treated as a migration measurement.

Confirm that every fixture hash, expected series binding, selector, table schema, coordinate dimension/order/shape/value, and serialized numeric field passed schema validation. A self-consistent artifact that omits output or substitutes fixture data is invalid evidence.

## Hard migration gates

Apply each hard gate to every unrounded posterior scalar from the representative Difference-in-Differences and Synthetic Control scenarios, including each draw-wise R² and counterfactual coordinate:

1. `abs(candidate_mean - reference_mean) <= max(4 * hypot(reference_mcse, candidate_mcse), 1e-6 + 1e-4 * abs(reference_mean))`.
2. `abs(candidate_mean - reference_mean) / pooled_posterior_sd <= 0.1`, with the absolute rule used when the pooled posterior SD is degenerate.
3. Effect-summary table schema, requested `0.94` HDI metadata and labels, metric selectors, dimensions, shapes, and coordinate values are semantically identical.

Mutual mean-in-94%-HDI containment is diagnostic only. It must appear in the metric table but never change the overall decision.

## Representative outputs

Include the generated metric table for the following contracts:

| Scenario | Contract | Required output |
|---|---|---|
| Difference-in-Differences | Public effect summary | Semantic table equality with unrounded treatment-effect mean and explicit 0.94 HDI |
| Difference-in-Differences | In-sample fit | Draw-wise R² derived from conditional expected `mu` |
| Difference-in-Differences | Counterfactual | Treated-post `mu` summaries with exact coordinate equality |
| Synthetic Control | Public effect summary | Semantic table equality with unrounded average and cumulative effect means and explicit 0.94 HDIs |
| Synthetic Control | In-sample fit | Pre-treatment draw-wise R² derived from conditional expected `mu` |
| Synthetic Control | Counterfactual | Post-treatment `mu` summaries with exact coordinate equality |

## Conclusion and follow-on use

State the overall gate result and identify every failed hard gate, if any. A diagnostic HDI-containment failure without a failed hard gate is reported but does not change the conclusion.

For #157, reuse the fixed inputs, summary extraction, and semantic contracts as correctness-test infrastructure, but evaluate known simulation ground truth with independently registered posterior-SD-unit bounds. Do not make PyMC 5 output the correctness oracle.
