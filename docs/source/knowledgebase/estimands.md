# Estimands in CausalPy

Understanding **what** a method estimates is just as important as knowing
**how** to use it. This page introduces a framework for thinking about causal
estimands and connects CausalPy's methods to this framework.

## The Estimand Framework

Following {cite:t}`lundberg2021estimand`, we distinguish three interconnected
concepts in causal inference:

```
Theoretical Estimand     -->  Empirical Estimand      -->  Estimator
```

- **Theoretical Estimand**: The causal quantity of interest, defined by the
  research question. This specifies *what* causal effect we want to know, for
  *whom*, and over *what time period*. Examples: "the average effect of a job
  training program on weekly earnings for program participants" or "the
  cumulative impact of a marketing campaign on sales for the treated region."

- **Empirical Estimand**: A specific, data-linked quantity that can be
  identified under a set of assumptions. Examples: {term}`ATT`, {term}`ATE`,
  {term}`LATE`, local effect at a cutoff, time-varying unit-specific impact.

- **Estimator**: The combination of a statistical model and a computational
  procedure used to produce an estimate from data. In CausalPy, this typically
  means choosing a model type (Bayesian via PyMC or OLS via scikit-learn)
  together with a computation to extract the causal quantity (e.g., coefficient
  extraction, g-computation). The same design can be implemented with different
  estimators---for example, a DiD design can use either a Bayesian model with
  g-computation or an OLS model with coefficient extraction.

The connection between theoretical and empirical estimands requires
**identification assumptions**---claims about the data-generating process that
cannot be tested from data alone. These assumptions are often formalized using
Directed Acyclic Graphs (DAGs). See {doc}`quasi_dags` for DAG-based
identification strategies for each quasi-experimental method.

The **structure of available data** also constrains which empirical estimands
are even *candidates*. Panel data with treated and control groups makes the
ATT a candidate; a single time series restricts you to unit-specific impacts;
data with a threshold-based assignment mechanism makes local effects at the
cutoff a possibility. But data structure alone does not guarantee credibility---
the identification assumptions must also be defensible. The choice of empirical
estimand is thus jointly determined by the research question, the available
data, and the assumptions one is willing to defend.

The estimator is the machinery that transforms data into an estimate. Different
estimators make different bias-variance trade-offs and have different data
requirements. See {doc}`structural_causal_models` for a deeper treatment of
structural versus reduced-form approaches.

:::{note}
This is an iterative process: estimates inform new theoretical questions,
refining our understanding over time.
:::

---

## CausalPy Methods: From Questions to Estimates

Each CausalPy experiment class targets a specific empirical estimand and
supports different estimators (Bayesian or OLS). Understanding which estimand
your method targets---and under what assumptions---is essential for valid
causal interpretation.

### Difference-in-Differences

**Typical research questions**: What is the effect of a policy that was
implemented in some regions/groups but not others? Did the intervention cause a
change in outcomes for the treated group?

**Empirical estimand**: {term}`Average treatment effect on the treated` (ATT)---
the average causal effect on units that received treatment in the post-treatment
period, relative to their counterfactual trajectory.

**Identification assumptions**:

- {term}`Parallel trends assumption`: Absent treatment, treated and control
  groups would have followed the same trajectory.
- No anticipation: Units do not change behavior before treatment begins.
- No interference between units: One unit's treatment does not affect another
  unit's outcomes.

See the [Difference in Differences section of quasi_dags](quasi_dags.ipynb#difference-in-differences)
for the DAG representation.

**Estimator**: Coefficient-based. The ATT is estimated as the coefficient on
the group-by-post interaction term. CausalPy supports both Bayesian (PyMC) and
OLS (scikit-learn) models for this design.

**Interpretation note**: Plots show counterfactual trajectories, but the
reported effect is the interaction coefficient---a single summary of the
treatment effect across all post-treatment observations.

---

### Interrupted Time Series

**Typical research questions**: Did an intervention (policy change, marketing
campaign, etc.) affect a time series? What is the causal impact over time?

**Empirical estimand**: Time-varying causal impact for a single treated unit
(or aggregate). This is **not** a population-level {term}`ATE` or {term}`ATT`---
it is unit-specific and time-indexed:

$$\text{impact}(t) = Y_{\text{observed}}(t) - \mathbb{E}[Y_{\text{counterfactual}}(t)]$$

**Identification assumptions**:

- Stable pre-intervention relationship: The model correctly captures the
  pre-intervention trend and seasonality.
- No concurrent shocks: No other events affect the outcome at treatment time.
- Correct counterfactual model: The model specification accurately represents
  what would have happened without intervention.

See the [Interrupted Time Series section of quasi_dags](quasi_dags.ipynb#interrupted-time-series)
for the DAG representation.

**Estimator**: G-computation. A model is fit to pre-intervention data only,
then used to predict counterfactual outcomes in the post-intervention period.
The causal impact is the difference between observed and predicted values.
CausalPy supports both Bayesian (PyMC) and OLS (scikit-learn) models for this
design.

**Interpretation note**: The effect varies over time. Summarizing with a single
number (e.g., average impact) loses information about the temporal pattern. Use
`effect_summary()` for both point-in-time and cumulative statistics.

---

### Synthetic Control

**Typical research questions**: What would have happened to a treated unit
(e.g., a country, region, or firm) if it had not been treated? What is the
causal effect of a unique intervention?

**Empirical estimand**: Time-varying causal impact for a treated unit---the
difference between the observed outcome and a synthetic counterfactual
constructed from weighted control units:

$$\text{impact}(t) = Y_{\text{treated}}(t) - \sum_i w_i \cdot Y_{\text{control}_i}(t)$$

Like ITS, this is **not** a population-level effect.

**Identification assumptions**:

- Parallel trends in weighted combination: The synthetic control would have
  followed the same trajectory as the treated unit absent treatment.
- No spillovers: Treatment of one unit does not affect control units.
- Convex hull coverage: The treated unit can be well-approximated by a weighted
  combination of controls.
- No concurrent shocks: No other events differentially affect the treated unit.

See the [Synthetic Control section of quasi_dags](quasi_dags.ipynb#synthetic-control)
for the DAG representation.

**Estimator**: G-computation. Weights are learned from pre-intervention data
to minimize the distance between the treated unit and the weighted combination
of controls. These weights are then applied to construct the counterfactual in
the post-intervention period. CausalPy supports both Bayesian (PyMC) and OLS
(scikit-learn) models for this design.

**Interpretation note**: The effect is specific to the treated unit and time
period. Generalization requires additional assumptions.

---

### Regression Discontinuity

**Typical research questions**: What is the effect of crossing a threshold
(e.g., eligibility cutoff, election margin, test score threshold)?

**Empirical estimand**: Local average treatment effect at the cutoff---the
causal effect for units exactly at the threshold where treatment assignment
changes. This is a highly local estimate:

$$\text{effect} = \lim_{x \to c^+} \mathbb{E}[Y|X=x] - \lim_{x \to c^-} \mathbb{E}[Y|X=x]$$

**Identification assumptions**:

- Continuity at cutoff: The conditional expectation of the outcome would be
  continuous at the threshold absent treatment.
- No manipulation: Units cannot precisely sort around the cutoff.

See the [Regression Discontinuity section of quasi_dags](quasi_dags.ipynb#regression-discontinuity)
for the DAG representation.

**Estimator**: Coefficient-based. The treatment effect is estimated as the
discontinuity in predicted outcomes at the cutoff, typically using local
polynomials around the {term}`running variable` threshold. CausalPy supports
both Bayesian (PyMC) and OLS (scikit-learn) models for this design.

**Interpretation note**: The effect is **local** to the cutoff. Units far from
the threshold may experience different treatment effects. The bandwidth
parameter controls how much data is used---narrower bandwidths are more local
but have higher variance.

---

## Context-Dependence of Estimands

:::{note}
The empirical estimand is not always uniquely determined by the experiment
class. Design choices can change what you are estimating:

- **IPW**: The `weighting_scheme` parameter selects between ATE (`"raw"`,
  `"robust"`) and the overlap population estimand (`"overlap"`).
- **Regression Discontinuity**: Bandwidth choice affects how local the effect
  is (see the Regression Discontinuity section above).
- **IV**: The instrument used defines the complier population, determining
  whose LATE you estimate.

The descriptions above assume standard usage. Always consider what your
specific design choices imply for interpretation.
:::

---

## Quick Reference

| Method | Empirical Estimand | Computation |
|--------|-------------------|-------------|
| Difference-in-Differences | {term}`ATT` | Coefficient-based |
| Interrupted Time Series | Time-varying unit-specific impact | G-computation |
| Synthetic Control | Time-varying unit-specific impact | G-computation |
| Regression Discontinuity | Local effect at cutoff | Coefficient-based |

For methods not covered here (IV, IPW, ANCOVA), see the respective notebook
documentation, {doc}`quasi_dags` for identification, and the {doc}`glossary`
for estimand definitions.

---

## References

:::{bibliography}
:filter: docname in docnames
:::
