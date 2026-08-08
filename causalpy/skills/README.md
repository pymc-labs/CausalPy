# User-Facing Agent Skills

Markdown skills in this directory teach AI coding agents how to use CausalPy for causal inference tasks. They are distributed to end users via [Decision AI Hub](https://hub.decision.ai).

Developer-focused skills (environment setup, PR workflows, testing conventions, etc.) live in `.agents/skills/`. They are **not** included here.

When these user-facing skills change, update the distributed Decision AI Hub copy after the PR lands, or leave an explicit follow-up task if the Hub update cannot happen in the same workflow.

## Layout

| Path | Purpose |
|------|---------|
| `choosing-causalpy-methods/` | Route causal or impact questions to the right CausalPy experiment via ordered intake, disambiguation, and explicit no-fit outcomes |
| `causal-detective/` | Challenging causal claims with threat assessment, counterfactual reasoning, and falsification checks |
| `example-datasets/` | Loading bundled CausalPy example datasets for demos, tutorials, and tests |
| `running-causalpy-experiments/` | Fitting chosen experiments, configuring models and priors, summarizing, plotting, and interpreting results |
| `running-placebo-analysis/` | Placebo-in-time sensitivity checks |
