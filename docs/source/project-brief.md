---
orphan: true
myst:
  html_meta:
    description: "Prepare a causal inference project brief for PyMC Labs: the decision, the intervention and its timing, available data and units, existing experiments, and timeline."
"og:description": "Prepare a causal inference project brief for PyMC Labs: the decision, the intervention and its timing, available data and units, existing experiments, and timeline."
---

# Work with PyMC Labs

## Submit an RFP or project brief

Tell the team behind CausalPy what you want to decide. PyMC Labs can help with lift and geo tests, {term}`quasi-experiments <Quasi-experiment>`, and high-stakes causal analysis.

**Already have an RFP?** Share its scope, submission deadline, and a link to the brief through the inquiry page. A short description is enough if you are still defining the project.

:::{button-link} https://www.pymc-labs.com/contact
:color: primary

Send your project inquiry
:::

Prefer to talk it through? [Book a consulting discovery call](https://calendly.com/benjamin-vincent/causalpy). These calls are for consulting inquiries only, not free technical assistance. Usage questions belong in [GitHub Discussions](https://github.com/pymc-labs/CausalPy/discussions/categories/q-a).

## Prepare your brief

You do not need to answer every question before getting in touch. These prompts help make the first conversation useful.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 1. The question and decision
What should this work help you decide, and who owns that decision? For example, whether a regional campaign changed sales, or whether a policy cutoff changed an outcome.
:::

:::{grid-item-card} 2. The intervention and its timing
What changed, when it started and ended, and which units were exposed? Note rollout, geography, or eligibility rules.
:::

:::{grid-item-card} 3. Available data and units
Which outcomes, units, and time coverage are available? Summarize frequency and known gaps. Do not include personal records in an initial inquiry.
:::

:::{grid-item-card} 4. Existing experiments
Note any randomized tests, geo or lift tests, holdouts, or earlier analyses that could validate an estimate.
:::

:::{grid-item-card} 5. Timeline
When is the decision needed? Who will use the result, and are there procurement or review dates?
:::
::::

{download}`Download the project-brief outline <_static/causal-inference-project-brief.txt>`

The [how-to gallery](notebooks/index.md) shows the {term}`quasi-experimental <Quasi-experiment>` designs CausalPy supports.
