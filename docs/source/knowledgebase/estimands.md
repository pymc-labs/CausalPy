# Estimands in CausalPy

Understanding **what** a method estimates is just as important as knowing **how** to use it. This page introduces a framework for thinking about causal estimands and connects CausalPy's methods to this framework.

## The Estimand Framework

Following {cite:t}`lundberg2021estimand`, we distinguish three interconnected concepts in causal inference:

```text
Theoretical Estimand     -->  Empirical Estimand      -->  Estimator
```

- **Theoretical Estimand**: The causal quantity of interest, defined by the research question. This specifies *what* causal effect we want to know, for *whom*, and over *what time period*. Examples: "the average effect of a job training program on weekly earnings for program participants" or "the cumulative impact of a marketing campaign on sales for the treated region."

- **Empirical Estimand**: A specific, data-linked quantity that can be identified under a set of assumptions. Examples: {term}`ATT`, {term}`ATE`, local treatment effect at a cutoff, time-varying unit-specific impact.

- **Estimator**: The combination of a statistical model and a computational procedure used to produce an estimate from data. The fitted model and the procedure that extracts the causal estimate are distinct choices: choosing Bayesian inference or OLS does not by itself determine how the causal quantity is computed.

### How is the causal estimate extracted?

Counterfactual construction or prediction is the broad prediction-based approach. G-computation is one structured member of that family, not a synonym for every procedure that constructs a {term}`counterfactual`. The hierarchy below classifies common extraction procedures by the computation they perform:

```text
How is the causal estimate extracted?
├── Direct parameter extraction
│   └── A model parameter is exactly the target estimand
├── Counterfactual construction or prediction
│   ├── G-computation and standardization
│   ├── One-sided counterfactual prediction
│   ├── Local prediction contrasts
│   └── Synthetic counterfactual construction
├── Weighting
└── Specialized or hybrid estimators
```

- **Direct parameter extraction** reads an estimand directly from a fitted parameter when the model specification makes that parameter the target quantity.
- **G-computation and standardization** define explicit interventions, predict conditional expected outcomes under each intervention, contrast those predictions, and average or standardize over a specified target population.
- **One-sided counterfactual prediction** compares observed treated outcomes with predicted untreated outcomes. Because it need not predict both {term}`potential outcomes` and standardize their contrast over a population, it does not necessarily follow the full g-computation pattern. Interrupted Time Series is an example.
- **Local prediction contrasts** compare predictions at a boundary rather than standardizing over a population. Regression Discontinuity estimates a level discontinuity, while Regression Kink estimates a change in derivatives.
- **Synthetic counterfactual construction** predicts an untreated trajectory from weighted control units. Synthetic Control belongs here; calling it g-computation requires a broader, carefully justified use of that term.
- **Weighting and specialized or hybrid estimators** include inverse probability weighting (IPW), instrumental variables (IV), synthetic Difference-in-Differences, and doubly robust estimators. These are not mutually exclusive alternatives to the branches above---they are routinely combined with parameter extraction or counterfactual prediction. IPW reweights observed or predicted outcomes before contrasting them, IV reads the effect from a coefficient in a model that accounts for endogeneity, and doubly robust estimators explicitly combine weighting with an outcome model. They appear as separate branches because their defining computation is neither pure parameter extraction nor pure counterfactual prediction.

:::{note}
These categories can be algebraically equivalent in special cases. In the standard identity-link additive OLS Difference-in-Differences model, the group-by-post interaction coefficient is exactly the corresponding predicted treated-versus-counterfactual contrast. Coefficient extraction and prediction-based estimation are therefore computational descriptions, not necessarily disjoint estimators.
:::

This page focuses on the concepts that remain stable across implementations. The exact extraction procedure used by each CausalPy backend is documented with its experiment class in the {doc}`../api/index`.

The connection between theoretical and empirical estimands requires **identification assumptions**---claims about the data-generating process that cannot be tested from data alone. These assumptions are often formalized using Directed Acyclic Graphs (DAGs). See {doc}`quasi_dags` for DAG-based identification strategies for each quasi-experimental method.

The **structure of available data** also constrains which empirical estimands are even *candidates*. Panel data with treated and control groups makes the ATT a candidate; a single time series restricts you to unit-specific impacts; data with a threshold-based assignment mechanism makes local effects at the cutoff a possibility. But data structure alone does not guarantee credibility---the identification assumptions must also be defensible. The choice of empirical estimand is thus jointly determined by the research question, the available data, and the assumptions one is willing to defend.

The estimator is the machinery that transforms data into an estimate. Different estimators make different bias-variance trade-offs and have different data requirements. See {doc}`structural_causal_models` for a deeper treatment of structural versus reduced-form approaches and {doc}`prediction-contract` for the distinction between response-scale and link-scale contrasts.

:::{note}
This is an iterative process: estimates inform new theoretical questions, refining our understanding over time.
:::

---

## CausalPy Methods: From Questions to Estimates

Each quasi-experimental design connects a research question to an empirical estimand under a set of identification assumptions. Understanding which estimand a design targets---and under what assumptions---is essential for valid causal interpretation.

:::{note}
**Shared assumption**: All methods below require the Stable Unit Treatment Value Assumption (SUTVA)---one unit's treatment does not affect another unit's outcomes, and there is only one version of the treatment. This assumption is listed here once rather than repeated under each method.
:::

### Difference-in-Differences

**Typical research questions**: What is the effect of a policy that was implemented in some regions/groups but not others? Did the intervention cause a change in outcomes for the treated group?

**Empirical estimand**: {term}`Average treatment effect on the treated` (ATT)---the average causal effect on units that received treatment in the post-treatment period, relative to their counterfactual trajectory.

**Identification assumptions**:

- {term}`Parallel trends assumption`: Absent treatment, treated and control groups would have followed the same trajectory.
- No anticipation: Units do not change behavior before treatment begins.

See the [Difference in Differences section of quasi_dags](quasi_dags.ipynb#difference-in-differences) for the DAG representation.

**Interpretation note**: A single ATT summarizes the treatment effect across treated post-treatment observations and can conceal variation across units or time.

---

### Interrupted Time Series

**Typical research questions**: Did an intervention (policy change, marketing campaign, etc.) affect a time series? What is the causal impact over time?

**Empirical estimand**: Time-varying causal impact for a single treated unit (or aggregate). This is **not** a population-level {term}`ATE` or {term}`ATT`---it is unit-specific and time-indexed:

$$\text{impact}(t) = Y_{\text{observed}}(t) - \mathbb{E}[Y_{\text{counterfactual}}(t)]$$

**Identification assumptions**:

- Stable pre-intervention relationship: The model correctly captures the pre-intervention trend and seasonality.
- No concurrent shocks: No other events affect the outcome at treatment time.
- Correct counterfactual model: The model specification accurately represents what would have happened without intervention.

See the [Interrupted Time Series section of quasi_dags](quasi_dags.ipynb#interrupted-time-series) for the DAG representation.

**Interpretation note**: The effect varies over time. Summarizing with a single number, such as average impact, loses information about the temporal pattern.

---

### Synthetic Control

**Typical research questions**: What would have happened to a treated unit (e.g., a country, region, or firm) if it had not been treated? What is the causal effect of a unique intervention?

**Empirical estimand**: Time-varying causal impact for a treated unit---the difference between the observed outcome and a synthetic counterfactual constructed from weighted control units:

$$\text{impact}(t) = Y_{\text{treated}}(t) - \sum_i w_i \cdot Y_{\text{control}_i}(t)$$

Like ITS, this is **not** a population-level effect.

**Identification assumptions**:

- Parallel trends in weighted combination: The synthetic control would have followed the same trajectory as the treated unit absent treatment.
- No spillovers: Treatment of one unit does not affect control units.
- Convex hull coverage: The treated unit can be well-approximated by a weighted combination of controls.
- No concurrent shocks: No other events differentially affect the treated unit.

See the [Synthetic Control section of quasi_dags](quasi_dags.ipynb#synthetic-control) for the DAG representation.

**Interpretation note**: The effect is specific to the treated unit and time period. Generalization requires additional assumptions.

---

### Regression Discontinuity

**Typical research questions**: What is the effect of crossing a threshold (e.g., eligibility cutoff, election margin, test score threshold)?

**Empirical estimand**: Local treatment effect at the cutoff---the causal effect for units exactly at the threshold where treatment assignment changes. This is a highly local estimate:

$$\text{effect} = \lim_{x \to c^+} \mathbb{E}[Y|X=x] - \lim_{x \to c^-} \mathbb{E}[Y|X=x]$$

**Identification assumptions**:

- Continuity at cutoff: The conditional expectation of the outcome would be continuous at the threshold absent treatment.
- No manipulation: Units cannot precisely sort around the cutoff.

See the [Regression Discontinuity section of quasi_dags](quasi_dags.ipynb#regression-discontinuity) for the DAG representation.

**Interpretation note**: The effect is **local** to the cutoff. Units far from the threshold may experience different treatment effects. The bandwidth controls how much data contributes to the estimate---narrower bandwidths are more local but have higher variance.

---

## Context-Dependence of Estimands

:::{note}
The empirical estimand is not always uniquely determined by the experiment class. Design choices can change what you are estimating:

- **IPW**: Different weighting schemes can target the ATE or an effect in the overlap population, while doubly robust estimators combine weighting with an outcome model.
- **Regression Discontinuity**: Bandwidth choice affects how local the effect is (see the Regression Discontinuity section above).
- **IV**: The instrument used defines the complier population, determining whose {term}`LATE` you estimate.

The descriptions above assume standard usage. Always consider what your specific design choices imply for interpretation.
:::

---

## Quick Reference

| Method | Empirical Estimand |
|--------|-------------------|
| Difference-in-Differences | {term}`ATT` |
| Interrupted Time Series | Time-varying unit-specific impact |
| Synthetic Control | Time-varying unit-specific impact |
| Regression Discontinuity | Local treatment effect at cutoff |

For methods not covered in detail here, including IV, IPW, and ANCOVA, see the respective notebook documentation, {doc}`quasi_dags` for identification, and the {doc}`glossary` for estimand definitions.
