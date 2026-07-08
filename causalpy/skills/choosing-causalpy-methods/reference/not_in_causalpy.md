# When CausalPy Has Not Implemented The Right Method

Use this reference when the user's question is causal but none of the implemented CausalPy experiments match the design. Do not route to the "closest" class just to avoid saying no. Return the Not implemented in CausalPy output contract from `SKILL.md`.

## Common Not-Implemented Routes

| User need | Why no CausalPy experiment fits | Closest partial fit, if any |
|---|---|---|
| Propensity-score matching, coarsened exact matching, Mahalanobis matching, or genetic matching | CausalPy has inverse propensity weighting, not matching estimators. | `InversePropensityWeighting` only if weighting is acceptable and treatment is binary with overlap. |
| Fuzzy regression discontinuity | `RegressionDiscontinuity` is for sharp RD where treatment assignment changes at the cutoff. | `InstrumentalVariable` may be useful only if the cutoff indicator can be defended as an instrument and the user accepts an IV framing. |
| Continuous or dose-response treatment without a kink design | IPW requires binary treatment, and `RegressionKink` only covers a slope change at a known threshold. | External dose-response, generalized propensity score, or outcome-regression workflows. |
| Multi-arm treatment | Implemented treatment-selection tools do not cover multi-valued treatment arms. | Possibly separate binary comparisons, but that changes the estimand and assumptions. |
| Non-absorbing staggered treatment, treatment reversal, or repeated on/off treatment | `StaggeredDifferenceInDifferences` assumes absorbing treatment. | No direct CausalPy route. |
| Alternative staggered DiD estimators such as Callaway-Sant'Anna, Sun-Abraham, or Gardner | CausalPy's staggered class implements a specific imputation-style workflow, not every modern staggered estimator. | `StaggeredDifferenceInDifferences` only if its assumptions and estimand are the intended target. |
| Augmented synthetic control, generalized synthetic control, or matrix completion | `SyntheticControl` and `SyntheticDifferenceInDifferences` do not implement these estimators. | `SyntheticControl` or `SyntheticDifferenceInDifferences` only if donor-weight or SDiD assumptions are acceptable. |
| CATE, causal forests, uplift models, or heterogeneous treatment-effect discovery | CausalPy experiment classes target specific quasi-experimental estimands, not flexible CATE discovery. | No direct CausalPy route. |
| Mediation analysis or path decomposition | CausalPy does not implement mediation estimators. | No direct CausalPy route. |
| Survival, hazard, duration, or time-to-event causal models | CausalPy experiment classes assume regression/time-series style outcomes, not survival likelihoods. | No direct CausalPy route. |
| Interference, network spillovers, or geographic spillover designs | CausalPy experiments generally require no-interference assumptions and do not model spillovers directly. | Synthetic control or ITS can be descriptive only if spillovers are negligible or explicitly out of scope. |
| Cluster-robust frequentist DiD as the primary inferential output | CausalPy has OLS-compatible paths for some classes, but not a dedicated cluster-robust DiD reporting workflow. | `DifferenceInDifferences` only if the available uncertainty output is acceptable. |
| Generic "control for confounders" regression with no assignment story | Panel or regression adjustment alone does not identify a causal effect without an exogeneity story. | `PanelRegression` for coefficient-level association, or `InversePropensityWeighting` if treatment is binary, confounders are measured, and overlap holds. |

## Output Template

Use this structure when routing here:

- Status: CausalPy has not implemented the right method.
- Ideal method category: name the likely method family.
- Why no CausalPy experiment fits: state the missing estimator, unsupported treatment type, unsupported assignment pattern, or missing identification story.
- Closest partial fit: name one only if it is genuinely useful, and state the caveat.
- What would unlock CausalPy: describe the design change, data requirement, or assumption change that would make an implemented class appropriate.
