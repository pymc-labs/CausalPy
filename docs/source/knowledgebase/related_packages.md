---
myst:
  html_meta:
    "description": "Python packages directly related to CausalPy for causal inference and quasi-experimental analysis."
    "og:description": "Python packages directly related to CausalPy for causal inference and quasi-experimental analysis."
---

# Related packages

CausalPy focuses on Bayesian inference for quasi-experimental designs. The packages below are directly related: either alternative implementations of the same methods, or complementary tools in the Python causal inference ecosystem. This is a curated list rather than an exhaustive one — for a much broader collection of resources see [Awesome Causal Inference](https://github.com/matteocourthoud/awesome-causal-inference). For recommended books and videos, see {doc}`causal_written_resources` and {doc}`causal_video_resources`.

## Alternative causal inference frameworks

* [DoWhy](https://github.com/py-why/dowhy) — a framework for end-to-end causal inference built around explicit causal graphs: model the DAG, identify the estimand, estimate the effect, and refute the result. The most directly comparable project to CausalPy, taking a frequentist and machine-learning approach where CausalPy takes a Bayesian one.
* [EconML](https://github.com/py-why/EconML) — also from the py-why ecosystem, focusing on heterogeneous treatment effect (CATE) estimation with machine-learning methods.
* [DoubleML](https://github.com/DoubleML/doubleml-for-py) — double/debiased machine learning for treatment effect estimation.

## Quasi-experimental methods and econometrics

* [differences](https://github.com/bernardodionisi/differences) — a dedicated package for difference-in-differences estimation.
* [linearmodels](https://github.com/bashtage/linearmodels) — econometric estimators for instrumental variables, panel data, and system regressions; useful for frequentist IV/2SLS workflows.
* [pyfixest](https://github.com/py-econometrics/pyfixest) — fast fixed-effects regression following R's fixest, including event-study and difference-in-differences estimators.

## Specialized tools

* [causalnlp](https://github.com/amaiya/causalnlp) — causal inference when treatments or covariates are text.
* [TFCausalImpact](https://github.com/google/tfcausalimpact) — a TensorFlow Probability port of Google's [CausalImpact](https://github.com/google/CausalImpact) R package, the Bayesian structural time-series approach that inspires CausalPy's interrupted time series analyses.
* [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) — a sister PyMC Labs package for Bayesian marketing mix models and customer lifetime value, sharing the same PyMC ecosystem and workflow.
