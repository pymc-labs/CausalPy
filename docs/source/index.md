---
myst:
  html_meta:
    description: "CausalPy is an open-source Python package for Bayesian causal inference in quasi-experiments. Supports synthetic control, difference-in-differences, regression discontinuity, interrupted time series, and more — with full uncertainty quantification via PyMC."
    keywords: "causal inference, Python, Bayesian, quasi-experiment, synthetic control, difference-in-differences, regression discontinuity, interrupted time series, PyMC, causal impact"
---

:::{image} _static/logo.png
:width: 40 %
:align: center
:alt: CausalPy logo
:::

```{include} ../../README.md
:start-after: <!-- docs-start -->
:end-before: <!-- docs-end -->
```

## How CausalPy compares

The Python causal inference ecosystem has several strong packages, each with different strengths. Here's how CausalPy fits in:

| | CausalPy | CausalImpact | DoWhy | CausalML | pyfixest |
|---|---|---|---|---|---|
| **Primary focus** | Quasi-experimental causal effect estimation | Time series impact analysis (BSTS) | End-to-end causal reasoning pipeline (graph → identify → estimate → refute) | Uplift modeling and heterogeneous treatment effects (CATE/ITE) | High-dimensional fixed effects regression (econometrics) |
| **Bayesian estimation** | First-class via PyMC, with full posterior distributions and HDI | Yes — Bayesian Structural Time Series | Limited — primarily frequentist estimators | No | No |
| **Uncertainty quantification** | Credible intervals (HDI), ROPE analysis, tail probabilities | Posterior credible intervals | Confidence intervals, refutation tests | Confidence intervals | Cluster-robust and heteroscedasticity-robust standard errors |
| **Synthetic control** | Yes — Bayesian weighted | Yes | No | No | No |
| **Difference-in-differences** | Yes — including staggered adoption | No | Yes — as one of many estimators | No | Yes — TWFE, Did2s, Sun-Abraham |
| **Regression discontinuity** | Yes — Bayesian and OLS | No | No | No | No |
| **Interrupted time series** | Yes | Yes | No | No | No |
| **Causal discovery** | No — use causal-learn or DoWhy-GCM | No | Yes — via DoWhy-GCM extension | No | No |
| **Uplift / CATE modeling** | No — use CausalML or EconML | No | Yes — via EconML integration | Yes — primary focus | No |
| **Best for** | Teams that need uncertainty-aware estimates from a quasi-experimental design | Rapid impact analysis of a single time series using synthetic control | Projects that require formal identification and robustness testing across a causal DAG | A/B test personalization, targeting optimization, CATE estimation | Fast econometric regressions with many fixed effects |

### When to choose CausalPy

CausalPy is the right choice when you have a **plausible quasi-experimental design** — a policy change, a geo-targeted campaign, a threshold rule — and you want **Bayesian uncertainty quantification** over the causal effect. If you need to reason about causal graphs and formal identification strategies, start with **DoWhy**. If you need heterogeneous treatment effects from A/B tests, look at **CausalML**. If you need fast fixed-effects regressions at scale, **pyfixest** is purpose-built for that.

CausalPy complements these packages rather than replacing them. Many teams use DoWhy for identification and CausalPy for estimation, or pyfixest for initial exploration and CausalPy for Bayesian analysis with full uncertainty.

```{include} ../../README.md
:start-after: <!-- docs-end -->
```

:::{toctree}
:hidden:

knowledgebase/index
api/index
notebooks/index
:::
